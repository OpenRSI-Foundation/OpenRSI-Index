"""Fresh, isolated Judge execution over a paused persistent workspace."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from rsi_harness.errors import (
    ContainerExecNotStartedError,
    InfrastructureError,
    RetryableSubmissionError,
    SetupError,
    StateTransitionError,
    SubmissionError,
)
from rsi_harness.models import (
    VERIFIER_OUTPUT_TRUNCATION_MARKER,
    AgentRunResult,
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerTmpfs,
    ContainerVolumeMount,
    EvaluationRequest,
    GPUAllocation,
    JudgeGPUMode,
    ManagedNetwork,
    ManagedWorkdirVolume,
    NetworkPolicy,
    RewardResult,
    RootfsSnapshotLease,
    RootfsSnapshotMode,
    RunPlan,
    SubmissionReport,
    SubmissionStatus,
    WorkQuiescence,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.environment import (
    MissingRuntimeEnvironmentError,
    resolve_runtime_environment,
    runtime_template_name,
)
from rsi_harness.runtime.gpu import (
    NVIDIA_VISIBLE_DEVICES_ENV,
    nvidia_visible_devices_value,
)
from rsi_harness.runtime.mount_topology import validate_split_workdir_target
from rsi_harness.runtime.network import NetworkPolicyEnforcer, NetworkPolicyLease
from rsi_harness.runtime.redaction import redact_exact_values, redact_text
from rsi_harness.runtime.reward import read_reward
from rsi_harness.runtime.rootfs_snapshot import RootfsSnapshotNotCreatedError

VERIFIER_COMMAND = ("/bin/bash", "/tests/test.sh")
VERIFIER_EXPECTED_GPU_UUIDS_ENV = "RSI_HARNESS_EXPECTED_GPU_UUIDS"
OUTPUT_TRUNCATION_MARKER = VERIFIER_OUTPUT_TRUNCATION_MARKER
_ROOTFS_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_JUDGE_SNAPSHOT_REF = re.compile(r"rsi-harness-rootfs:judge-round-[0-9a-f]{64}\Z")


class _SubmissionRejected(Exception):
    """Internal control flow for a reportable pre-pause submission error."""


class WorkRuntime(Protocol):
    def pause(self, container: ContainerRef) -> None: ...

    def unpause(self, container: ContainerRef) -> None: ...

    def inspect_quiescence(self, container: ContainerRef) -> WorkQuiescence | None: ...


class JudgeRoundRuntime(Protocol):
    def create(self, spec: ContainerSpec) -> ContainerRef: ...

    def start(self, container: ContainerRef) -> None: ...

    def inject_tests(self, container: ContainerRef, source: Path) -> None: ...

    def exec(
        self,
        container: ContainerRef,
        command: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
        environment: dict[str, str] | None = None,
        output_path: Path | None = None,
    ) -> AgentRunResult: ...

    def remove(self, container: ContainerRef) -> None: ...

    def contain_after_remove_failure(self, container: ContainerRef) -> None: ...

    @property
    def safe_to_release_isolation(self) -> bool: ...

    def recovery_context(self, container: ContainerRef | None = None) -> str: ...

    def close(self) -> None: ...


class RootfsSnapshotPort(Protocol):
    def planned_ref(
        self, *, run_id: str, task_id: str, round_id: str, purpose: str
    ) -> str: ...

    def acquire(
        self,
        work: ContainerRef,
        *,
        run_id: str,
        task_id: str,
        round_id: str,
        purpose: str = "judge-round",
        planned_ref: str | None = None,
    ) -> RootfsSnapshotLease: ...

    def release(self, lease: RootfsSnapshotLease) -> None: ...


class ArtifactWriterPort(Protocol):
    def record_submission(self, report: SubmissionReport) -> None: ...


class JudgeResourceMutationObserver(Protocol):
    """Typed durable events emitted immediately around external mutations."""

    def judge_network_planned(self, round_id: str, planned_name: str) -> None: ...

    def judge_network_created(self, round_id: str, network: ManagedNetwork) -> None: ...

    def judge_container_planned(self, round_id: str, planned_name: str) -> None: ...

    def judge_container_created(
        self, round_id: str, container: ContainerRef
    ) -> None: ...

    def judge_policy_planned(self, round_id: str, rule_id: str) -> None: ...

    def judge_policy_installed(
        self, round_id: str, lease: NetworkPolicyLease
    ) -> None: ...

    def judge_container_removed(self, round_id: str, container_id: str) -> None: ...

    def judge_policy_removed(self, round_id: str, rule_id: str) -> None: ...

    def judge_network_removed(self, round_id: str, network_id: str) -> None: ...


class DockerJudgeRoundRuntime:
    """One container, authoritative policy lease, and private Judge bridge."""

    def __init__(
        self,
        *,
        runtime: DockerContainerRuntime,
        provisioner: DockerContainerRuntime,
        enforcer: NetworkPolicyEnforcer,
        network: ManagedNetwork,
        policy: NetworkPolicy,
        round_id: str,
        planned_container_name: str,
        observer: JudgeResourceMutationObserver,
    ) -> None:
        self._runtime = runtime
        self._provisioner = provisioner
        self._enforcer = enforcer
        self._network = network
        self._policy = policy
        self._round_id = round_id
        self._planned_container_name = planned_container_name
        self._observer = observer
        self._lease: NetworkPolicyLease | None = None
        self._closed = False
        self._container: ContainerRef | None = None
        self._safe_to_release_isolation = True

    def create(self, spec: ContainerSpec) -> ContainerRef:
        self._observer.judge_container_planned(
            self._round_id, self._planned_container_name
        )
        container = self._runtime.create(
            spec, planned_name=self._planned_container_name
        )
        self._container = container
        self._safe_to_release_isolation = False
        self._observer.judge_container_created(self._round_id, container)
        try:
            for mount in spec.volume_mounts:
                self._runtime.attest_workdir_volume_mount(container, mount)
            planned_lease = self._enforcer.plan(
                container,
                self._policy,
                network=self._network,
            )
            # Keep the planned identity in memory before installation.  If a
            # partial install cannot roll back, recovery still has the exact
            # rule authority even though apply() cannot return normally.
            self._lease = planned_lease
            lease = self._enforcer.apply(
                container,
                self._policy,
                network=self._network,
                mutation_observer=self,
                planned=planned_lease,
            )
            self._lease = lease
            self._runtime.install_network_policy(container, lease)
            return container
        except Exception as primary:
            recovery_required = "recovery_required" in _error_text(primary)
            cleanup: list[str] = []
            try:
                self.remove(container)
            except Exception as error:
                cleanup.append(f"container removal: {_error_text(error)}")
                try:
                    self.contain_after_remove_failure(container)
                except Exception as containment_error:
                    cleanup.append(
                        "emergency containment unproven: "
                        f"{_error_text(containment_error)}"
                    )
            if self._lease is not None and self._safe_to_release_isolation:
                try:
                    self._enforcer.cleanup(self._lease)
                except Exception as error:
                    self._safe_to_release_isolation = False
                    cleanup.append(
                        "recovery_required: policy cleanup is unproven: "
                        f"{_error_text(error)}"
                    )
                else:
                    self._lease = None
            if recovery_required:
                self._safe_to_release_isolation = False
            detail = _combined_error(_error_text(primary), cleanup)
            raise InfrastructureError(detail or "Judge creation failed") from primary

    def start(self, container: ContainerRef) -> None:
        self._runtime.start(container)

    def inject_tests(self, container: ContainerRef, source: Path) -> None:
        self._runtime.inject_directory(
            container,
            source,
            PurePosixPath("/tests"),
        )

    def exec(
        self,
        container: ContainerRef,
        command: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
        environment: dict[str, str] | None = None,
        output_path: Path | None = None,
    ) -> AgentRunResult:
        return self._runtime.exec(
            container,
            command,
            timeout_seconds=timeout_seconds,
            environment=environment,
            output_path=output_path,
            redact_output=False,
        )

    def remove(self, container: ContainerRef) -> None:
        self._runtime.remove(container)
        self._safe_to_release_isolation = True
        self._observer.judge_container_removed(self._round_id, container.container_id)

    def policy_install_planned(self, rule_id: str) -> None:
        self._observer.judge_policy_planned(self._round_id, rule_id)

    def policy_installed(self, lease: NetworkPolicyLease) -> None:
        self._observer.judge_policy_installed(self._round_id, lease)

    def contain_after_remove_failure(self, container: ContainerRef) -> None:
        stop_error: str | None = None
        try:
            self._runtime.stop(container)
        except Exception as error:
            stop_error = _error_text(error)
        try:
            stopped = self._runtime.is_stopped_or_gone(container)
        except Exception as error:
            detail = f"inspection failed: {_error_text(error)}"
            if stop_error is not None:
                detail = f"stop failed: {stop_error}; {detail}"
            self._safe_to_release_isolation = False
            raise InfrastructureError(
                f"Judge containment is unproven: {detail}"
            ) from error
        if not stopped:
            detail = "authoritative state is still running"
            if stop_error is not None:
                detail = f"stop failed: {stop_error}; {detail}"
            self._safe_to_release_isolation = False
            raise InfrastructureError(f"Judge containment is unproven: {detail}")
        # A successful stop proves execution containment only. It cannot prove
        # that the exact-labeled container was removed, so all dependent
        # isolation remains durable for recovery.
        self._safe_to_release_isolation = False

    @property
    def safe_to_release_isolation(self) -> bool:
        return self._safe_to_release_isolation

    def recovery_context(self, container: ContainerRef | None = None) -> str:
        resolved = container or self._container
        container_id = "unknown" if resolved is None else resolved.container_id
        rule_id = "unknown" if self._lease is None else self._lease.rule_id
        return (
            f"judge_container_id={container_id},"
            f"network_id={self._network.network_id},"
            f"policy_rule_id={rule_id}"
        )

    def close(self) -> None:
        if self._closed:
            return
        if not self._safe_to_release_isolation:
            raise InfrastructureError(
                "recovery_required: refusing to remove Judge policy/network while "
                f"containment is unproven; {self.recovery_context()}"
            )
        if self._lease is not None:
            rule_id = self._lease.rule_id
            try:
                self._enforcer.cleanup(self._lease)
            except Exception as error:
                self._safe_to_release_isolation = False
                raise InfrastructureError(
                    "recovery_required: Judge policy cleanup is unproven: "
                    f"{_error_text(error)}; {self.recovery_context()}"
                ) from error
            self._observer.judge_policy_removed(self._round_id, rule_id)
            self._lease = None
        try:
            self._provisioner.remove_network(self._network)
        except Exception as error:
            self._safe_to_release_isolation = False
            raise InfrastructureError(
                "recovery_required: Judge network removal is unproven: "
                f"{_error_text(error)}; {self.recovery_context()}"
            ) from error
        self._observer.judge_network_removed(self._round_id, self._network.network_id)
        self._closed = True


class DockerJudgeRuntimeFactory:
    """Create a fresh Docker bridge and fail-closed policy owner per round."""

    def __init__(
        self,
        client: Any,
        *,
        run_id: str,
        task_id: str,
        task_source_dir: Path,
        allowed_mount_roots: Sequence[Path],
        network_policy_enforcer: NetworkPolicyEnforcer,
        lifecycle_observer: JudgeResourceMutationObserver,
        work_container: ContainerRef | None = None,
        omit_gpu_device_requests_for_tests: bool = False,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._task_id = task_id
        self._task_source_dir = task_source_dir
        self._allowed_mount_roots = tuple(allowed_mount_roots)
        self._enforcer = network_policy_enforcer
        self._observer = lifecycle_observer
        if work_container is not None and work_container.role != "work":
            raise SetupError("Judge volume authority requires a Work container")
        self._work_container = work_container
        self._omit_gpu_device_requests_for_tests = omit_gpu_device_requests_for_tests

    def __call__(self, plan: RunPlan, round_id: str) -> DockerJudgeRoundRuntime:
        common = {
            "run_id": self._run_id,
            "task_id": self._task_id,
            "role": "judge",
            "task_source_dir": self._task_source_dir,
            "allowed_mount_roots": self._allowed_mount_roots,
            "omit_gpu_device_requests_for_tests": (
                self._omit_gpu_device_requests_for_tests
            ),
        }
        provisioner = DockerContainerRuntime(self._client, **common)
        planned_network = provisioner.planned_network_name(round_id)
        self._observer.judge_network_planned(round_id, planned_network)
        try:
            network = provisioner.create_network(
                round_id,
                internal=plan.task.verifier.network.mode == "no-network",
            )
        except SetupError as primary:
            # The Docker boundary only returns SetupError after exact
            # planned-name/label absence has been proved. Clear the durable
            # plan before allowing snapshot release and Work resume.
            try:
                self._observer.judge_network_removed(round_id, planned_network)
            except BaseException as observer_error:
                raise InfrastructureError(
                    "recovery_required: Judge network is proven absent but "
                    "durable planned authority could not be cleared: "
                    f"{_error_text(observer_error)}; "
                    f"planned_network={planned_network}"
                ) from primary
            raise
        try:
            self._observer.judge_network_created(round_id, network)
            runtime = DockerContainerRuntime(
                self._client,
                **common,
                network=network,
                network_policy_enforcer=self._enforcer,
                exec_output_limit_bytes=plan.task.verifier.output_limit_bytes,
                workdir_volume_references=(
                    () if self._work_container is None else (self._work_container,)
                ),
            )
            return DockerJudgeRoundRuntime(
                runtime=runtime,
                provisioner=provisioner,
                enforcer=self._enforcer,
                network=network,
                policy=plan.task.verifier.network,
                round_id=round_id,
                planned_container_name=runtime.planned_container_name(round_id),
                observer=self._observer,
            )
        except BaseException as primary:
            try:
                provisioner.remove_network(network)
            except BaseException as rollback_error:
                raise InfrastructureError(
                    "recovery_required: Judge network creation rollback is "
                    f"unproven: {_error_text(rollback_error)}; "
                    f"planned_network={planned_network}"
                ) from primary
            try:
                self._observer.judge_network_removed(round_id, network.network_id)
            except BaseException as observer_error:
                raise InfrastructureError(
                    "recovery_required: Judge network rollback is physically "
                    "proven but durable removal authority is unproven: "
                    f"{_error_text(observer_error)}; "
                    f"planned_network={planned_network},"
                    f"network_id={network.network_id}"
                ) from primary
            raise


class JudgeRunner:
    """Evaluate one immutable Work snapshot and recover Work on every path."""

    def __init__(
        self,
        *,
        run_id: str,
        workdir_volume: ManagedWorkdirVolume | None = None,
        work_runtime: WorkRuntime,
        judge_runtime_factory: Callable[[RunPlan, str], JudgeRoundRuntime],
        snapshot_backend: RootfsSnapshotPort,
        artifact_writer: ArtifactWriterPort,
        quiescence_checker: Callable[[GPUAllocation, ContainerRef], None],
        reward_reader: Callable[[Path, str | None], RewardResult] = read_reward,
        monotonic: Callable[[], float] = time.monotonic,
        lifecycle_observer: Any | None = None,
        verifier_secret_env: Mapping[str, str] | None = None,
        event_callback: Callable[[str, object], None] | None = None,
    ) -> None:
        self._run_id = run_id
        self._workdir_volume = workdir_volume
        self._work_runtime = work_runtime
        self._judge_runtime_factory = judge_runtime_factory
        self._snapshot_backend = snapshot_backend
        self._artifact_writer = artifact_writer
        self._quiescence_checker = quiescence_checker
        self.reward_reader = reward_reader
        self._monotonic = monotonic
        self._lifecycle_observer = lifecycle_observer
        self._verifier_secret_env = dict(verifier_secret_env or {})
        self._event_callback = event_callback or (lambda _name, _value: None)
        self.recovery_required = False
        self.submission_closed = False

    def evaluate(self, request: EvaluationRequest) -> SubmissionReport:
        started = self._monotonic()
        plan = request.run_plan
        status = SubmissionStatus.INFRASTRUCTURE_ERROR
        rewards: dict[str, float] = {}
        score: float | None = None
        output = ""
        exit_code: int | None = None
        timed_out = False
        verifier_output_required = False
        full_output_captured = False
        primary_error: str | None = None
        cleanup_errors: list[str] = []
        paused = False
        safe_to_release_isolation = True
        lease: RootfsSnapshotLease | None = None
        planned_snapshot_ref: str | None = None
        judge_round: JudgeRoundRuntime | None = None
        judge: ContainerRef | None = None
        engine_bug: StateTransitionError | None = None
        verifier_error_secrets: set[str] = set()
        persisted_report: SubmissionReport | None = None
        artifact_error: BaseException | None = None
        retry_candidate: SubmissionError | None = None

        try:
            self._validate_feedback_output(request)
            self._validate_workdir_volume(plan)
            if tuple(plan.task.verifier.command) != VERIFIER_COMMAND:
                raise RuntimeError(
                    "RunPlan verifier command is not the fixed Harbor test command"
                )
            self._observe(
                "work_pause_planned",
                work_container_id=request.work_container.container_id,
            )
            paused = True
            self._work_runtime.pause(request.work_container)
            self._observe(
                "work_paused",
                work_container_id=request.work_container.container_id,
            )
            if plan.gpu_plan.judge_mode is JudgeGPUMode.RELEASE_ALL:
                try:
                    self._quiescence_checker(plan.gpu_plan.work, request.work_container)
                except SubmissionError as error:
                    retry_candidate = error
                    primary_error = _error_text(error, verifier_error_secrets)
                    raise _SubmissionRejected from error
            log_dir = self._prepare_log_dir(request)
            planned_snapshot_ref = self._snapshot_backend.planned_ref(
                run_id=self._run_id,
                task_id=plan.task.task_id,
                round_id=request.round_id,
                purpose="judge-round",
            )
            if _JUDGE_SNAPSHOT_REF.fullmatch(planned_snapshot_ref) is None:
                raise InfrastructureError(
                    "rootfs snapshot backend returned an invalid planned Judge "
                    "image reference"
                )
            # A failed durable plan write cannot authorize the commit mutation.
            # Keep Work paused because the runtime write itself is unproven.
            safe_to_release_isolation = False
            self._observe(
                "snapshot_planned",
                round_id=request.round_id,
                planned_snapshot_ref=planned_snapshot_ref,
                source_work_container_id=request.work_container.container_id,
            )
            safe_to_release_isolation = True
            try:
                lease = self._snapshot_backend.acquire(
                    request.work_container,
                    run_id=self._run_id,
                    task_id=plan.task.task_id,
                    round_id=request.round_id,
                    purpose="judge-round",
                    planned_ref=planned_snapshot_ref,
                )
            except RootfsSnapshotNotCreatedError:
                # No commit mutation exists, but the durable planned authority
                # must be cancelled before Work may resume.
                safe_to_release_isolation = False
                self._observe(
                    "snapshot_cancelled",
                    round_id=request.round_id,
                    planned_snapshot_ref=planned_snapshot_ref,
                    source_work_container_id=request.work_container.container_id,
                )
                safe_to_release_isolation = True
                raise
            # The external commit now exists. Do not release it or resume Work
            # until its complete actual identity is durable and attested.
            safe_to_release_isolation = False
            self._attest_snapshot_lease(request, lease, planned_snapshot_ref)
            self._observe(
                "snapshot_acquired",
                snapshot_lease_id=lease.lease_id,
                snapshot_image_id=lease.image_id,
                snapshot_image_ref=lease.image_ref,
                snapshot_source_container_id=lease.source_container_id,
            )
            safe_to_release_isolation = True
            self._observe("judge_planned", round_id=request.round_id)
            judge_round = self._judge_runtime_factory(plan, request.round_id)
            judge_spec, verifier_environment = self._judge_spec(
                request, lease, log_dir, verifier_error_secrets
            )
            judge = judge_round.create(judge_spec)
            self._observe(
                "judge_created",
                judge_container_id=judge.container_id,
                recovery_context=judge_round.recovery_context(judge),
            )
            judge_round.start(judge)
            judge_round.inject_tests(judge, plan.task.source_dir / "tests")
            self._event_callback("judge_exec_started", {"round_id": request.round_id})
            verifier_output_required = True
            result = judge_round.exec(
                judge,
                VERIFIER_COMMAND,
                timeout_seconds=plan.task.verifier.timeout_seconds,
                environment=verifier_environment,
                output_path=request.verifier_output,
            )
            output = _bounded_output(
                result.output,
                limit=plan.task.verifier.output_limit_bytes,
                truncated=result.output_truncated,
            )
            if output:
                self._event_callback(
                    "judge_output",
                    redact_text(redact_exact_values(output, verifier_error_secrets)),
                )
            exit_code = result.exit_code
            timed_out = result.timed_out
            full_output_captured = result.full_output_captured
            if timed_out:
                self._discard_timeout_rewards(log_dir)
                status = SubmissionStatus.VERIFIER_TIMEOUT
                primary_error = "Judge verifier timed out; partial reward was discarded"
            else:
                reward = self.reward_reader(
                    log_dir,
                    plan.task.verifier.primary_reward,
                )
                if reward.error is not None:
                    status = SubmissionStatus.VERIFIER_ERROR
                    primary_error = _redact_error_text(
                        reward.error, verifier_error_secrets
                    )
                else:
                    status = SubmissionStatus.COMPLETED
                    rewards = dict(reward.rewards)
                    score = reward.score
        except _SubmissionRejected:
            pass
        except StateTransitionError as error:
            engine_bug = error
        except Exception as error:
            if isinstance(error, ContainerExecNotStartedError):
                verifier_output_required = False
                full_output_captured = False
            primary_error = _error_text(error, verifier_error_secrets)
            status = SubmissionStatus.INFRASTRUCTURE_ERROR
            score = None
            if "recovery_required" in primary_error:
                safe_to_release_isolation = False
        finally:
            # The evaluator is the sole report owner. Prove the required
            # artifact write while Work is still paused and all round
            # isolation is retained. SubmissionService only publishes the
            # already-persisted result to its in-memory history.
            if engine_bug is None and retry_candidate is None:
                candidate = self._report(
                    request,
                    status=status,
                    rewards=rewards,
                    score=score,
                    output=output,
                    exit_code=exit_code,
                    timed_out=timed_out,
                    verifier_output_required=verifier_output_required,
                    full_output_captured=full_output_captured,
                    started=started,
                    error=primary_error,
                )
                try:
                    self._artifact_writer.record_submission(candidate)
                except BaseException as error:
                    artifact_error = error
                    self.submission_closed = True
                    if paused:
                        self.recovery_required = True
                        safe_to_release_isolation = False
                    try:
                        self._observe(
                            "recovery_required" if paused else "submission_closed",
                            recovery_context=(
                                "required submission artifact persistence failed"
                            ),
                        )
                    except BaseException as marker_error:
                        cleanup_errors.append(
                            "recovery marker persistence: "
                            f"{_error_text(marker_error, verifier_error_secrets)}"
                        )
                else:
                    persisted_report = candidate
            if judge is not None and judge_round is not None:
                if artifact_error is not None:
                    # The exact Judge is the only recoverable authority for a
                    # round whose required report was not made durable. Stop
                    # and attest it, but do not destroy it or any dependency.
                    safe_to_release_isolation = False
                    try:
                        judge_round.contain_after_remove_failure(judge)
                    except Exception as containment_error:
                        cleanup_errors.append(
                            "report-failure containment unproven: "
                            f"{_error_text(containment_error, verifier_error_secrets)}"
                        )
                    else:
                        cleanup_errors.append(
                            "report-failure containment proved exact Judge "
                            "stopped or gone"
                        )
                else:
                    try:
                        judge_round.remove(judge)
                        self._observe(
                            "judge_removed", judge_container_id=judge.container_id
                        )
                    except Exception as error:
                        cleanup_errors.append(
                            "Judge removal: "
                            f"{_error_text(error, verifier_error_secrets)}"
                        )
                        # A stopped-but-retained Judge is still an unremoved exact-
                        # labeled resource. Do not resume Work or allocate a round
                        # that could overwrite its single durable lease authority.
                        safe_to_release_isolation = False
                        try:
                            judge_round.contain_after_remove_failure(judge)
                        except Exception as containment_error:
                            containment_detail = _error_text(
                                containment_error, verifier_error_secrets
                            )
                            cleanup_errors.append(
                                f"emergency containment unproven: {containment_detail}"
                            )
                            safe_to_release_isolation = False
                        else:
                            cleanup_errors.append(
                                "emergency containment proved Judge stopped or gone"
                            )
            if judge_round is not None and not judge_round.safe_to_release_isolation:
                safe_to_release_isolation = False
            if judge_round is not None and safe_to_release_isolation:
                try:
                    judge_round.close()
                except Exception as error:
                    cleanup_errors.append(
                        "Judge network/policy cleanup: "
                        f"{_error_text(error, verifier_error_secrets)}"
                    )
                    safe_to_release_isolation = False
            if not safe_to_release_isolation:
                self.recovery_required = True
                context = (
                    "unknown"
                    if judge_round is None
                    else judge_round.recovery_context(judge)
                )
                snapshot_context = self._snapshot_recovery_context(
                    lease=lease,
                    planned_ref=planned_snapshot_ref,
                    work=request.work_container,
                )
                cleanup_errors.append(
                    "recovery_required: retained Judge policy/network, snapshot, "
                    f"and Work pause; {context},{snapshot_context}"
                )
                self._observe(
                    "recovery_required",
                    recovery_context=f"{context},{snapshot_context}",
                )
            if lease is not None and safe_to_release_isolation:
                try:
                    self._snapshot_backend.release(lease)
                    self._observe("snapshot_released", snapshot_lease_id=lease.lease_id)
                except Exception as error:
                    cleanup_errors.append(
                        "snapshot release: "
                        f"{_error_text(error, verifier_error_secrets)}"
                    )
                    safe_to_release_isolation = False
                    cleanup_errors.append(
                        "recovery_required: snapshot release is unproven; "
                        f"snapshot_lease_id={lease.lease_id},"
                        f"work_container_id={request.work_container.container_id}"
                    )
                    self.recovery_required = True
                    self._observe(
                        "recovery_required",
                        recovery_context=f"snapshot_lease_id={lease.lease_id}",
                    )
            if paused and safe_to_release_isolation:
                try:
                    self._work_runtime.unpause(request.work_container)
                    self._observe(
                        "work_unpaused",
                        work_container_id=request.work_container.container_id,
                    )
                except Exception as error:
                    cleanup_errors.append(
                        f"Work unpause: {_error_text(error, verifier_error_secrets)}"
                    )
                    cleanup_errors.extend(
                        self._restore_work_quiescence(
                            request.work_container,
                            verifier_error_secrets,
                        )
                    )
                    cleanup_errors.append(
                        "recovery_required: Work unpause is unproven; "
                        f"work_container_id={request.work_container.container_id}"
                    )
                    self.recovery_required = True
                    try:
                        self._observe(
                            "recovery_required",
                            recovery_context=(
                                f"work_container_id={request.work_container.container_id}"
                            ),
                        )
                    except Exception as observer_error:
                        cleanup_errors.append(
                            "recovery authority persistence: "
                            f"{_error_text(observer_error, verifier_error_secrets)}"
                        )

        if engine_bug is not None:
            raise engine_bug
        if retry_candidate is not None and not cleanup_errors:
            verifier_error_secrets.clear()
            raise RetryableSubmissionError(
                f"release all Work GPU processes before retrying: {retry_candidate}"
            ) from retry_candidate
        if artifact_error is not None:
            safe_error = _error_text(artifact_error, verifier_error_secrets)
            verifier_error_secrets.clear()
            raise InfrastructureError(
                "required report artifact persistence failed: "
                f"{safe_error or type(artifact_error).__name__}"
            ) from artifact_error
        if cleanup_errors:
            status = SubmissionStatus.INFRASTRUCTURE_ERROR
            score = None
        error = _combined_error(primary_error, cleanup_errors)
        report = self._report(
            request,
            status=status,
            rewards=rewards,
            score=score,
            output=output,
            exit_code=exit_code,
            timed_out=timed_out,
            verifier_output_required=verifier_output_required,
            full_output_captured=full_output_captured,
            started=started,
            error=error,
        )
        if cleanup_errors:
            try:
                self._artifact_writer.record_submission(report)
            except BaseException as persistence_error:
                self.recovery_required = True
                safe_error = _error_text(persistence_error, verifier_error_secrets)
                verifier_error_secrets.clear()
                raise InfrastructureError(
                    "required cleanup report artifact persistence failed: "
                    f"{safe_error or type(persistence_error).__name__}"
                ) from persistence_error
        elif persisted_report is not None:
            report = persisted_report
        if retry_candidate is not None:
            verifier_error_secrets.clear()
            raise InfrastructureError(
                report.error or "recovery_required: Work unpause is unproven"
            )
        verifier_error_secrets.clear()
        return report

    def _restore_work_quiescence(
        self,
        work: ContainerRef,
        secret_values: set[str],
    ) -> list[str]:
        errors: list[str] = []
        observed: WorkQuiescence | None = None
        inspection_proven = False
        try:
            observed = self._work_runtime.inspect_quiescence(work)
            inspection_proven = True
        except Exception as error:
            errors.append(
                f"Work recovery inspection: {_error_text(error, secret_values)}"
            )
        if not inspection_proven or observed is None:
            try:
                self._work_runtime.pause(work)
            except Exception as error:
                errors.append(
                    f"Work recovery pause: {_error_text(error, secret_values)}"
                )
            try:
                observed = self._work_runtime.inspect_quiescence(work)
                inspection_proven = True
            except Exception as error:
                inspection_proven = False
                errors.append(
                    f"Work recovery attestation: {_error_text(error, secret_values)}"
                )
        event = (
            "work_recovery_state" if inspection_proven else "work_quiescence_unproven"
        )
        values = {
            "work_container_id": work.container_id,
            "paused": observed is WorkQuiescence.PAUSED,
            "stopped": observed is WorkQuiescence.STOPPED,
        }
        try:
            self._observe(event, **values)
        except Exception as error:
            errors.append(
                "Work recovery authority persistence: "
                f"{_error_text(error, secret_values)}"
            )
        if not inspection_proven or observed is None:
            errors.append(
                "recovery_required: exact Work quiescence remains unproven; "
                f"work_container_id={work.container_id}"
            )
        return errors

    def _observe(self, name: str, **values: object) -> None:
        if self._lifecycle_observer is not None:
            self._lifecycle_observer.resource_event(name, **values)

    def _report(
        self,
        request: EvaluationRequest,
        *,
        status: SubmissionStatus,
        rewards: dict[str, float],
        score: float | None,
        output: str,
        exit_code: int | None,
        timed_out: bool,
        verifier_output_required: bool,
        full_output_captured: bool,
        started: float,
        error: str | None,
    ) -> SubmissionReport:
        return SubmissionReport(
            round_id=request.round_id,
            status=status,
            rewards=rewards,
            score=score,
            output=output,
            exit_code=exit_code,
            timed_out=timed_out,
            verifier_output_required=verifier_output_required,
            full_output_captured=full_output_captured,
            duration_seconds=max(0.0, self._monotonic() - started),
            error=error,
        )

    @staticmethod
    def _prepare_log_dir(request: EvaluationRequest) -> Path:
        root = request.run_plan.paths.logs.resolve()
        candidate = request.verifier_logs.resolve(strict=False)
        if candidate == root or root not in candidate.parents:
            raise RuntimeError("verifier log directory is outside RunPlan logs")
        if candidate.exists():
            raise RuntimeError("verifier log directory already exists")
        candidate.mkdir(parents=True, mode=0o700)
        return candidate

    def _judge_spec(
        self,
        request: EvaluationRequest,
        lease: RootfsSnapshotLease,
        log_dir: Path,
        error_secret_values: set[str],
    ) -> tuple[ContainerSpec, dict[str, str]]:
        plan = request.run_plan
        try:
            environment = resolve_runtime_environment(
                tuple(
                    (key, value)
                    for key, value in plan.task.verifier.environment
                    if key
                    not in {
                        NVIDIA_VISIBLE_DEVICES_ENV,
                        VERIFIER_EXPECTED_GPU_UUIDS_ENV,
                    }
                ),
                self._verifier_secret_env,
            )
        except MissingRuntimeEnvironmentError as error:
            raise InfrastructureError(
                f"missing verifier runtime environment variable '{error.name}'"
            ) from error
        environment[VERIFIER_EXPECTED_GPU_UUIDS_ENV] = ",".join(
            plan.gpu_plan.judge.uuids
        )
        environment[NVIDIA_VISIBLE_DEVICES_ENV] = nvidia_visible_devices_value(
            plan.gpu_plan.judge
        )
        secret_names = set(plan.task.verifier.secret_env_names)
        error_secret_values.update(
            environment[key]
            for key, template in plan.task.verifier.environment
            if key in environment
            and runtime_template_name(template) in secret_names
            and environment[key]
        )
        return (
            ContainerSpec(
                image=lease.image_id,
                command=("/bin/sleep", "infinity"),
                workdir=plan.workdir,
                user=(
                    plan.images.judge_user
                    or plan.task.verifier.user
                    or plan.task.service.user
                ),
                environment=(),
                shm_size=plan.task.service.shm_size,
                cpus=plan.task.service.cpus,
                memory_mb=plan.task.service.memory_mb,
                storage_mb=plan.task.service.storage_mb,
                tmpfs=(
                    ContainerTmpfs(
                        target=PurePosixPath("/tests"),
                        options="rw,exec,nosuid,nodev,mode=0755",
                    ),
                ),
                mounts=(
                    ContainerMount(
                        source=log_dir,
                        target=PurePosixPath("/logs/verifier"),
                    ),
                ),
                volume_mounts=(
                    (
                        ContainerVolumeMount(
                            volume=self._workdir_volume,
                            target=plan.workdir,
                            read_only=True,
                        ),
                    )
                    if self._workdir_volume is not None
                    else ()
                ),
                gpu_allocation=plan.gpu_plan.judge,
                labels=(("rsi-harness.round-id", request.round_id),),
            ),
            environment,
        )

    def _validate_workdir_volume(self, plan: RunPlan) -> None:
        if plan.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR:
            validate_split_workdir_target(plan.workdir)
            if self._workdir_volume is None:
                raise InfrastructureError(
                    "split WORKDIR Judge authority is unavailable"
                )
            if (
                self._workdir_volume.driver != "local"
                or self._workdir_volume.run_id != self._run_id
                or self._workdir_volume.task_id != plan.task.task_id
                or self._workdir_volume.target != plan.workdir
            ):
                raise InfrastructureError(
                    "split WORKDIR Judge authority is inconsistent"
                )
        elif self._workdir_volume is not None:
            raise InfrastructureError("full-rootfs Judge received volume authority")

    def _validate_feedback_output(self, request: EvaluationRequest) -> None:
        if re.fullmatch(r"agent-[1-9][0-9]*", request.round_id) is None:
            raise InfrastructureError("Judge round identity is invalid")
        feedback_root = (
            request.run_plan.paths.logs
            / "runs"
            / self._run_id
            / request.run_plan.task.task_id
            / "feedback"
        )
        expected = feedback_root / f"{request.round_id}.log"
        if request.verifier_output != expected:
            raise InfrastructureError(
                "Judge feedback output path differs from exact run authority"
            )
        self._attest_directory_without_symlinks(
            request.run_plan.paths.logs, feedback_root
        )

    @staticmethod
    def _attest_directory_without_symlinks(root: Path, directory: Path) -> None:
        try:
            relative = directory.relative_to(root)
        except ValueError as error:
            raise InfrastructureError(
                "Judge feedback directory is outside exact logs authority"
            ) from error
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptors: list[int] = []
        try:
            descriptor = os.open(root, flags)
            descriptors.append(descriptor)
            for component in relative.parts:
                if component in {"", ".", ".."}:
                    raise InfrastructureError(
                        "Judge feedback directory contains unsafe components"
                    )
                descriptor = os.open(component, flags, dir_fd=descriptor)
                descriptors.append(descriptor)
        except OSError as error:
            raise InfrastructureError(
                "Judge feedback directory is missing, non-directory, or symlinked"
            ) from error
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    def _attest_snapshot_lease(
        self,
        request: EvaluationRequest,
        lease: RootfsSnapshotLease,
        planned_ref: str,
    ) -> None:
        plan = request.run_plan
        if (
            lease.purpose != "judge-round"
            or lease.run_id != self._run_id
            or lease.task_id != plan.task.task_id
            or lease.round_id != request.round_id
            or lease.source_container_id != request.work_container.container_id
            or lease.image_ref != planned_ref
            or _JUDGE_SNAPSHOT_REF.fullmatch(lease.image_ref) is None
            or _ROOTFS_IMAGE_ID.fullmatch(lease.image_id) is None
        ):
            raise InfrastructureError(
                "recovery_required: acquired Judge rootfs snapshot authority "
                "does not match the planned paused Work state"
            )

    @staticmethod
    def _snapshot_recovery_context(
        *,
        lease: RootfsSnapshotLease | None,
        planned_ref: str | None,
        work: ContainerRef,
    ) -> str:
        if lease is None:
            return (
                "snapshot_lease_id=unknown,"
                f"snapshot_image_ref={planned_ref or 'unknown'},"
                f"work_container_id={work.container_id}"
            )
        return (
            f"snapshot_lease_id={lease.lease_id},"
            f"snapshot_image_id={lease.image_id},"
            f"snapshot_image_ref={lease.image_ref},"
            f"work_container_id={lease.source_container_id}"
        )

    @staticmethod
    def _discard_timeout_rewards(log_dir: Path) -> None:
        for name in ("reward.json", "reward.txt"):
            (log_dir / name).unlink(missing_ok=True)


def _bounded_output(output: str, *, limit: int, truncated: bool) -> str:
    encoded = output.encode()
    if not truncated and len(encoded) <= limit:
        return output
    marker = OUTPUT_TRUNCATION_MARKER.encode()
    if limit <= len(marker):
        return marker[:limit].decode(errors="ignore")
    budget = limit - len(marker)
    head_limit = (budget + 1) // 2
    tail_limit = budget - head_limit
    head = encoded[:head_limit].decode(errors="ignore")
    tail = encoded[-tail_limit:].decode(errors="ignore") if tail_limit else ""
    value = head + OUTPUT_TRUNCATION_MARKER + tail
    while len(value.encode()) > limit and tail:
        tail = tail[1:]
        value = head + OUTPUT_TRUNCATION_MARKER + tail
    return value


def _redact_error_text(detail: str, secret_values: set[str]) -> str:
    return redact_exact_values(detail, secret_values)


def _error_text(error: BaseException, secret_values: set[str] | None = None) -> str:
    detail = str(error).strip()
    value = detail or type(error).__name__
    return _redact_error_text(value, secret_values or set())


def _combined_error(primary: str | None, cleanup: list[str]) -> str | None:
    if not cleanup:
        return primary
    cleanup_text = "; ".join(cleanup)
    if primary is None:
        return f"cleanup failure: {cleanup_text}"
    return f"primary failure: {primary}; cleanup failure: {cleanup_text}"


__all__ = [
    "DockerJudgeRuntimeFactory",
    "JudgeRunner",
    "JudgeResourceMutationObserver",
    "OUTPUT_TRUNCATION_MARKER",
    "VERIFIER_COMMAND",
    "VERIFIER_EXPECTED_GPU_UUIDS_ENV",
]
