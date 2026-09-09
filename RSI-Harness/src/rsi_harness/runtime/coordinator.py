"""End-to-end ownership of one persistent Agent and explicit Judge rounds."""

from __future__ import annotations

import os
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rsi_harness.errors import InfrastructureError, SetupError, StateTransitionError
from rsi_harness.models import (
    AgentRunResult,
    ContainerRef,
    EvaluationRequest,
    FrozenRewardMap,
    ManagedNetwork,
    ManagedWorkdirVolume,
    RootfsSnapshotLease,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunRequest,
    RunResult,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
    TaskDefinition,
    WorkQuiescence,
)
from rsi_harness.runtime.mount_topology import validate_run_plan_mount_topology
from rsi_harness.runtime.recovery import (
    LeaseStore,
    ResourceLease,
    RetainedImageRollbackAuthority,
    WorkdirVolumeResourceLease,
)
from rsi_harness.runtime.redaction import redact_text
from rsi_harness.runtime.submissions import (
    EmbeddedJudgeServer,
    JudgeEndpoint,
    SubmissionService,
)

_TERMINAL = {
    RunStatus.COMPLETED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
    RunStatus.NO_VALID_SUBMISSION,
}
_ALLOWED = {
    RunStatus.PREPARING: {
        RunStatus.AGENT_RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.AGENT_RUNNING: {
        RunStatus.SNAPSHOTTING,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.NO_VALID_SUBMISSION,
    },
    RunStatus.SNAPSHOTTING: {
        RunStatus.JUDGING,
        RunStatus.AGENT_RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
    RunStatus.JUDGING: {
        RunStatus.AGENT_RUNNING,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    },
}


class _RunCancelled(BaseException):
    pass


@contextmanager
def _cancellation_signals() -> Iterator[None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[signal.Signals, Any] = {}

    def cancel(signum: int, frame: object) -> None:
        del frame
        raise _RunCancelled(f"received signal {signum}")

    try:
        for selected in (signal.SIGINT, signal.SIGTERM):
            previous[selected] = signal.getsignal(selected)
            signal.signal(selected, cancel)
        yield
    finally:
        for selected, handler in previous.items():
            signal.signal(selected, handler)


class SubmissionServerOwner(Protocol):
    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class StartedSubmissionServer:
    """A started owner paired with the endpoint actually bound by that owner."""

    service: Any
    owner: SubmissionServerOwner
    endpoint: JudgeEndpoint


@dataclass(frozen=True, slots=True)
class EmbeddedSubmissionServerFactory:
    """Production composition for SubmissionService + EmbeddedJudgeServer."""

    bind_host: str
    port: int
    bridge_gateway: str
    startup_timeout_seconds: float = 5.0
    shutdown_timeout_seconds: float = 10.0

    def __call__(
        self, evaluator: Any, artifacts: Any, clock: Any
    ) -> StartedSubmissionServer:
        service = SubmissionService(
            evaluator=evaluator,
            artifact_writer=artifacts,
            clock=clock,
        )
        owner = EmbeddedJudgeServer(
            service,
            bind_host=self.bind_host,
            port=self.port,
            bridge_gateway=self.bridge_gateway,
            startup_timeout_seconds=self.startup_timeout_seconds,
            shutdown_timeout_seconds=self.shutdown_timeout_seconds,
        )
        endpoint = owner.start()
        return StartedSubmissionServer(
            service=service,
            owner=owner,
            endpoint=endpoint,
        )


class CoordinatorBackend(Protocol):
    """Mandatory production-capable ports owned in order by the coordinator."""

    def compile(self, request: RunRequest) -> Any: ...

    def allocate(self, definition: Any, selectors: Sequence[str]) -> Any: ...

    def prepare_images(self, definition: Any) -> Any: ...

    def prepare_plan(
        self,
        definition: Any,
        images: Any,
        allocation: Any,
        request: RunRequest,
        run_id: str,
    ) -> Any: ...

    def start_artifacts(self, plan: Any, run_id: str) -> Any: ...

    def start_server(
        self, evaluator: Any, artifacts: Any, clock: Any
    ) -> StartedSubmissionServer: ...

    def plan_network(self, plan: Any, run_id: str) -> str: ...

    def create_network(
        self, plan: Any, run_id: str, planned: str
    ) -> ManagedNetwork: ...

    def remove_network(self, network: ManagedNetwork) -> None: ...

    def plan_work_container(self, plan: Any, run_id: str) -> str: ...

    def plan_workdir_volume(
        self, plan: Any, run_id: str
    ) -> ManagedWorkdirVolume | None: ...

    def create_workdir_volume(
        self, plan: Any, run_id: str, planned: ManagedWorkdirVolume
    ) -> ManagedWorkdirVolume: ...

    def attest_workdir_volume(
        self, work: ContainerRef, volume: ManagedWorkdirVolume
    ) -> None: ...

    def attest_work_feedback_mount(self, work: ContainerRef) -> None: ...

    def remove_workdir_volume(self, volume: ManagedWorkdirVolume) -> None: ...

    def create_work(
        self,
        plan: Any,
        run_id: str,
        network: ManagedNetwork,
        planned_name: str,
        workdir_volume: ManagedWorkdirVolume | None,
    ) -> ContainerRef: ...

    def plan_work_policy(
        self, plan: Any, work: ContainerRef, network: ManagedNetwork
    ) -> str: ...

    def install_work_policy(
        self, plan: Any, work: ContainerRef, network: ManagedNetwork
    ) -> str: ...

    def remove_work_policy(self, rule_id: str) -> None: ...

    def start_work(self, work: ContainerRef) -> None: ...

    def remove_work(self, work: ContainerRef) -> None: ...

    def install_hooks(
        self, plan: Any, work: ContainerRef, submit_url: str, token: str
    ) -> None: ...

    def prepare_agent(
        self, plan: Any, max_submissions: int | None = None
    ) -> Any: ...

    def run_agent(
        self, prepared: Any, work: ContainerRef, timeout: float | None
    ) -> AgentRunResult: ...

    def stop_agent(self, work: ContainerRef) -> None: ...

    def quiesce_work(self, work: ContainerRef) -> WorkQuiescence: ...

    def plan_retained_work(self, plan: Any, work: ContainerRef) -> str: ...

    def retain_work(
        self, plan: Any, work: ContainerRef, planned_ref: str
    ) -> RootfsSnapshotLease: ...

    def release_retained_work(self, retained: RootfsSnapshotLease) -> None: ...

    def evaluate_submission(
        self,
        request: EvaluationRequest,
        *,
        lifecycle_observer: RoundLifecycleObserver,
    ) -> SubmissionReport: ...

    def record_event(self, name: str, value: object) -> None: ...


class RoundLifecycleObserver(Protocol):
    def resource_event(self, name: str, **values: object) -> None: ...


@dataclass(frozen=True, slots=True)
class RunPreparation:
    """One immutable compile/allocation result authorized before runtime mutation."""

    request: RunRequest
    definition: TaskDefinition
    gpu_plan: RunGPUPlan


@dataclass(slots=True)
class ProductionCoordinatorBackend:
    """Concrete typed composition over production ports and phase factories.

    The callables are construction seams, not optional behavior: a composition
    cannot omit evaluation, networking, server ownership, or Agent termination.
    """

    compiler: Any
    allocator: Callable[[Any, Sequence[str]], Any]
    image_preparer: Callable[[Any], Any]
    plan_preparer: Callable[[Any, Any, Any, RunRequest, str], Any]
    artifact_starter: Callable[[Any, str], Any]
    server_starter: Callable[[Any, Any, Any], StartedSubmissionServer]
    network_planner: Callable[[Any, str], str]
    network_creator: Callable[[Any, str, str], ManagedNetwork]
    network_remover: Callable[[ManagedNetwork], None]
    workdir_volume_planner: Callable[
        [Any, str], ManagedWorkdirVolume | None
    ]
    workdir_volume_creator: Callable[
        [Any, str, ManagedWorkdirVolume], ManagedWorkdirVolume
    ]
    workdir_volume_attester: Callable[
        [ContainerRef, ManagedWorkdirVolume], None
    ]
    workdir_volume_remover: Callable[[ManagedWorkdirVolume], None]
    work_name_planner: Callable[[Any, str], str]
    work_creator: Callable[
        [Any, str, ManagedNetwork, str, ManagedWorkdirVolume | None],
        ContainerRef,
    ]
    work_feedback_attester: Callable[[ContainerRef], None]
    work_policy_planner: Callable[[Any, ContainerRef, ManagedNetwork], str]
    work_policy_installer: Callable[[Any, ContainerRef, ManagedNetwork], str]
    work_policy_remover: Callable[[str], None]
    work_starter: Callable[[ContainerRef], None]
    work_remover: Callable[[ContainerRef], None]
    hook_installer: Callable[[Any, ContainerRef, str, str], None]
    agent_preparer: Callable[[Any, int | None], Any]
    agent_runner: Callable[[Any, ContainerRef, float | None], AgentRunResult]
    agent_stopper: Callable[[ContainerRef], None]
    work_quiescer: Callable[[ContainerRef], WorkQuiescence]
    retained_work_planner: Callable[[Any, ContainerRef], str]
    work_retainer: Callable[[Any, ContainerRef, str], RootfsSnapshotLease]
    retained_work_releaser: Callable[[RootfsSnapshotLease], None]
    evaluator: Callable[
        [EvaluationRequest, RoundLifecycleObserver], SubmissionReport
    ]
    event_recorder: Callable[[str, object], None] = lambda _name, _value: None
    preparation: RunPreparation | None = None

    def compile(self, request: RunRequest) -> Any:
        if self.preparation is not None:
            if request != self.preparation.request:
                raise SetupError("coordinator request differs from prevalidated run")
            return self.preparation.definition
        return self.compiler.compile(request.task_dir, request.options)

    def allocate(self, definition: Any, selectors: Sequence[str]) -> Any:
        if self.preparation is not None:
            if definition is not self.preparation.definition:
                raise SetupError(
                    "GPU allocation definition differs from prevalidated run"
                )
            if tuple(selectors) != tuple(self.preparation.request.gpu_selectors):
                raise SetupError("GPU selectors differ from prevalidated run")
            return self.preparation.gpu_plan
        return self.allocator(definition, selectors)

    def prepare_images(self, definition: Any) -> Any:
        return self.image_preparer(definition)

    def prepare_plan(
        self,
        definition: Any,
        images: Any,
        gpu_plan: Any,
        request: RunRequest,
        run_id: str,
    ) -> Any:
        return self.plan_preparer(definition, images, gpu_plan, request, run_id)

    def start_artifacts(self, plan: Any, run_id: str) -> Any:
        return self.artifact_starter(plan, run_id)

    def start_server(
        self, evaluator: Any, artifacts: Any, clock: Any
    ) -> StartedSubmissionServer:
        return self.server_starter(evaluator, artifacts, clock)

    def plan_network(self, plan: Any, run_id: str) -> str:
        return self.network_planner(plan, run_id)

    def create_network(
        self, plan: Any, run_id: str, planned: str
    ) -> ManagedNetwork:
        return self.network_creator(plan, run_id, planned)

    def remove_network(self, network: ManagedNetwork) -> None:
        self.network_remover(network)

    def plan_work_container(self, plan: Any, run_id: str) -> str:
        return self.work_name_planner(plan, run_id)

    def plan_workdir_volume(
        self, plan: Any, run_id: str
    ) -> ManagedWorkdirVolume | None:
        return self.workdir_volume_planner(plan, run_id)

    def create_workdir_volume(
        self, plan: Any, run_id: str, planned: ManagedWorkdirVolume
    ) -> ManagedWorkdirVolume:
        return self.workdir_volume_creator(plan, run_id, planned)

    def attest_workdir_volume(
        self, work: ContainerRef, volume: ManagedWorkdirVolume
    ) -> None:
        self.workdir_volume_attester(work, volume)

    def remove_workdir_volume(self, volume: ManagedWorkdirVolume) -> None:
        self.workdir_volume_remover(volume)

    def create_work(
        self,
        plan: Any,
        run_id: str,
        network: ManagedNetwork,
        planned_name: str,
        workdir_volume: ManagedWorkdirVolume | None,
    ) -> ContainerRef:
        return self.work_creator(
            plan, run_id, network, planned_name, workdir_volume
        )

    def attest_work_feedback_mount(self, work: ContainerRef) -> None:
        self.work_feedback_attester(work)

    def start_work(self, work: ContainerRef) -> None:
        self.work_starter(work)

    def plan_work_policy(
        self, plan: Any, work: ContainerRef, network: ManagedNetwork
    ) -> str:
        return self.work_policy_planner(plan, work, network)

    def install_work_policy(
        self, plan: Any, work: ContainerRef, network: ManagedNetwork
    ) -> str:
        return self.work_policy_installer(plan, work, network)

    def remove_work_policy(self, rule_id: str) -> None:
        self.work_policy_remover(rule_id)

    def remove_work(self, work: ContainerRef) -> None:
        self.work_remover(work)

    def install_hooks(
        self, plan: Any, work: ContainerRef, submit_url: str, token: str
    ) -> None:
        self.hook_installer(plan, work, submit_url, token)

    def prepare_agent(
        self, plan: Any, max_submissions: int | None = None
    ) -> Any:
        return self.agent_preparer(plan, max_submissions)

    def run_agent(
        self, prepared: Any, work: ContainerRef, timeout: float | None
    ) -> AgentRunResult:
        return self.agent_runner(prepared, work, timeout)

    def stop_agent(self, work: ContainerRef) -> None:
        self.agent_stopper(work)

    def quiesce_work(self, work: ContainerRef) -> WorkQuiescence:
        return self.work_quiescer(work)

    def plan_retained_work(self, plan: Any, work: ContainerRef) -> str:
        return self.retained_work_planner(plan, work)

    def retain_work(
        self, plan: Any, work: ContainerRef, planned_ref: str
    ) -> RootfsSnapshotLease:
        return self.work_retainer(plan, work, planned_ref)

    def release_retained_work(self, retained: RootfsSnapshotLease) -> None:
        self.retained_work_releaser(retained)

    def evaluate_submission(
        self,
        request: EvaluationRequest,
        *,
        lifecycle_observer: RoundLifecycleObserver,
    ) -> SubmissionReport:
        return self.evaluator(request, lifecycle_observer)

    def record_event(self, name: str, value: object) -> None:
        self.event_recorder(name, value)


class CoordinatorState:
    """Small strict state machine whose invalid edges are Engine defects."""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = run_id
        self._status = RunStatus.PREPARING
        self._history = [self._status]
        self.engine_errors: list[str] = []

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def history(self) -> tuple[RunStatus, ...]:
        return tuple(self._history)

    def transition(self, status: RunStatus) -> None:
        if status not in _ALLOWED.get(self._status, set()):
            error = StateTransitionError(
                f"invalid coordinator transition {self._status.value} -> "
                f"{status.value} for {self.run_id}"
            )
            self.engine_errors.append(f"StateTransitionError: {error}")
            raise error
        self._status = status
        self._history.append(status)


class _RoundEvaluator:
    def __init__(
        self,
        *,
        backend: CoordinatorBackend,
        state: CoordinatorState,
        transition: Callable[[RunStatus], None],
        persist_recovery: Callable[[str], None],
        update_work: Callable[..., None],
        update_judge: Callable[..., None],
    ) -> None:
        self._backend = backend
        self._state = state
        self._transition = transition
        self._persist_recovery = persist_recovery
        self._update_work = update_work
        self._update_judge = update_judge
        self.reports: list[SubmissionReport] = []
        self.recovery_required = False
        self.recovery_error: str | None = None
        self.submission_closed = False

    def evaluate(self, request: EvaluationRequest) -> SubmissionReport:
        if self.recovery_required or self.submission_closed:
            raise InfrastructureError(
                "submissions are closed after a required runtime write failed"
            )
        try:
            report = self._backend.evaluate_submission(
                request, lifecycle_observer=self
            )
        except StateTransitionError:
            raise
        except Exception:
            if not self.recovery_required and self._state.status in {
                RunStatus.SNAPSHOTTING,
                RunStatus.JUDGING,
            }:
                self._transition(RunStatus.AGENT_RUNNING)
                self._backend.record_event(
                    "work_resume", request.work_container.container_id
                )
            raise
        if not isinstance(report, SubmissionReport):
            raise TypeError("submission evaluator must return SubmissionReport")
        self.reports.append(report)
        self.recovery_required = bool(
            report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
            and report.error
            and "recovery_required" in report.error
        )
        if self.recovery_required:
            self.recovery_error = report.error
            self._persist_recovery(report.error or "recovery_required")
        elif self._state.status != RunStatus.AGENT_RUNNING:
            self._transition(RunStatus.AGENT_RUNNING)
            self._backend.record_event(
                "work_resume", request.work_container.container_id
            )
        return report

    def resource_event(self, name: str, **values: object) -> None:
        """Persist typed lifecycle notifications emitted at mutation boundaries."""
        if name == "workdir_volume_planned":
            planned = values.get("planned")
            if not isinstance(planned, ManagedWorkdirVolume):
                raise InfrastructureError(
                    "WORKDIR volume plan returned untyped authority"
                )
            self._update_work(
                workdir_volume=WorkdirVolumeResourceLease(
                    planned_name=planned.name,
                    planned_target=planned.target,
                    planned_snapshot_mode=planned.snapshot_mode,
                    planned_freshness_nonce=planned.freshness_nonce,
                )
            )
        elif name == "workdir_volume_created":
            planned = values.get("planned")
            if not isinstance(planned, ManagedWorkdirVolume):
                raise InfrastructureError(
                    "WORKDIR volume plan returned untyped authority"
                )
            actual = values.get("actual")
            rollback = values.get("rollback")
            if actual is not None and not isinstance(
                actual, ManagedWorkdirVolume
            ):
                raise InfrastructureError(
                    "WORKDIR volume create returned untyped authority"
                )
            if rollback is not None and not isinstance(
                rollback, ManagedWorkdirVolume
            ):
                raise InfrastructureError(
                    "WORKDIR volume rollback returned untyped authority"
                )
            if (actual is None) == (rollback is None):
                raise InfrastructureError(
                    "WORKDIR volume create event requires one exact authority"
                )
            self._update_work(
                workdir_volume=WorkdirVolumeResourceLease(
                    planned_name=planned.name,
                    planned_target=planned.target,
                    planned_snapshot_mode=planned.snapshot_mode,
                    planned_freshness_nonce=planned.freshness_nonce,
                    actual=actual,
                    rollback=rollback,
                )
            )
        elif name == "workdir_volume_mounted":
            actual = values["actual"]
            if not isinstance(actual, ManagedWorkdirVolume):
                raise InfrastructureError(
                    "WORKDIR volume mount returned untyped authority"
                )
            self._update_work(
                workdir_volume=WorkdirVolumeResourceLease(
                    planned_name=actual.name,
                    planned_target=actual.target,
                    planned_snapshot_mode=actual.snapshot_mode,
                    planned_freshness_nonce=actual.freshness_nonce,
                    actual=actual,
                    mounted=True,
                )
            )
        elif name == "workdir_volume_removed":
            self._update_work(
                workdir_volume=WorkdirVolumeResourceLease()
            )
        elif name == "work_pause_planned":
            if self._state.status == RunStatus.AGENT_RUNNING:
                self._transition(RunStatus.SNAPSHOTTING)
            self._update_work(paused=True)
        elif name == "snapshot_planned":
            round_id = str(values["round_id"])
            self._update_judge(
                round_id=round_id,
                planned_snapshot=f"snapshot:{self._state.run_id}:{round_id}",
                planned_snapshot_ref=values.get("planned_snapshot_ref"),
            )
        elif name == "snapshot_acquired":
            image_authority = values.get("snapshot_image_id") is not None
            self._update_judge(
                snapshot_lease_id=values.get("snapshot_lease_id"),
                snapshot_image_id=values.get("snapshot_image_id"),
                snapshot_image_ref=values.get("snapshot_image_ref"),
                snapshot_source_container_id=values.get(
                    "snapshot_source_container_id"
                ),
                snapshot_merged_path=(
                    None
                    if image_authority
                    else values.get("snapshot_merged_path")
                ),
                snapshot_process_id=(
                    None
                    if image_authority
                    else values.get("snapshot_process_id")
                ),
            )
        elif name == "snapshot_cancelled":
            self._update_judge(
                planned_snapshot=None,
                planned_snapshot_ref=None,
                snapshot_lease_id=None,
                snapshot_image_id=None,
                snapshot_image_ref=None,
                snapshot_source_container_id=None,
                snapshot_merged_path=None,
                snapshot_process_id=None,
            )
        elif name in {"judge_planned", "judge_container_planned"}:
            if self._state.status == RunStatus.SNAPSHOTTING:
                self._transition(RunStatus.JUDGING)
            round_id = str(values.get("round_id", "unknown"))
            planned = values.get(
                "planned_name", f"rsi-{self._state.run_id}-{round_id}-judge"
            )
            self._update_judge(round_id=round_id, planned_container=planned)
        elif name in {"judge_created", "judge_container_created"}:
            self._update_judge(container_id=values.get("judge_container_id"))
        elif name == "judge_network_planned":
            self._update_judge(planned_network=values.get("planned_name"))
        elif name == "judge_network_created":
            self._update_judge(
                network_id=values.get("network_id"),
                network_name=values.get("network_name"),
            )
        elif name == "judge_policy_installed":
            self._update_judge(
                planned_policy_rule_id=values.get("policy_rule_id"),
                policy_rule_id=values.get("policy_rule_id"),
            )
        elif name == "judge_policy_planned":
            self._update_judge(
                planned_policy_rule_id=values.get("policy_rule_id")
            )
        elif name == "judge_removed":
            self._update_judge(container_id=None, planned_container=None)
        elif name == "judge_policy_removed":
            self._update_judge(
                planned_policy_rule_id=None, policy_rule_id=None
            )
        elif name == "judge_network_removed":
            self._update_judge(
                network_id=None, network_name=None, planned_network=None
            )
        elif name == "snapshot_released":
            self._update_judge(
                planned_snapshot=None,
                planned_snapshot_ref=None,
                snapshot_lease_id=None,
                snapshot_image_id=None,
                snapshot_image_ref=None,
                snapshot_source_container_id=None,
                snapshot_merged_path=None,
                snapshot_process_id=None,
            )
        elif name == "work_unpaused":
            if self._state.status in {RunStatus.SNAPSHOTTING, RunStatus.JUDGING}:
                self._transition(RunStatus.AGENT_RUNNING)
            self._update_work(paused=False)
        elif name == "work_recovery_state":
            self._update_work(
                planned_quiescence=None,
                paused=bool(values.get("paused")),
                stopped=bool(values.get("stopped")),
            )
        elif name == "work_quiescence_unproven":
            self._update_work(
                planned_quiescence="pause-if-running",
                paused=False,
                stopped=False,
            )
        elif name == "recovery_required":
            # Close the in-memory submission authority before any durable or
            # artifact operation can itself fail.
            self.recovery_required = True
            context = str(values.get("recovery_context", name))
            self.recovery_error = (
                context
                if "recovery_required" in context
                else f"recovery_required: {context}"
            )
            self._persist_recovery(self.recovery_error)
        elif name == "submission_closed":
            # No isolation authority exists before Work pause. Close only the
            # submission stream; normal coordinator cleanup must still stop
            # and remove the running Work container and network.
            self.submission_closed = True

    def judge_network_planned(self, round_id: str, planned_name: str) -> None:
        self.resource_event(
            "judge_network_planned",
            round_id=round_id,
            planned_name=planned_name,
        )

    def judge_network_created(
        self, round_id: str, network: ManagedNetwork
    ) -> None:
        self.resource_event(
            "judge_network_created",
            round_id=round_id,
            network_id=network.network_id,
            network_name=network.name,
        )

    def judge_container_planned(self, round_id: str, planned_name: str) -> None:
        self.resource_event(
            "judge_container_planned",
            round_id=round_id,
            planned_name=planned_name,
        )

    def judge_container_created(
        self, round_id: str, container: ContainerRef
    ) -> None:
        self.resource_event(
            "judge_container_created",
            round_id=round_id,
            judge_container_id=container.container_id,
        )

    def judge_policy_planned(self, round_id: str, rule_id: str) -> None:
        self.resource_event(
            "judge_policy_planned", round_id=round_id, policy_rule_id=rule_id
        )

    def judge_policy_installed(self, round_id: str, policy: Any) -> None:
        self.resource_event(
            "judge_policy_installed",
            round_id=round_id,
            policy_rule_id=policy.rule_id,
        )

    def judge_container_removed(self, round_id: str, container_id: str) -> None:
        self.resource_event(
            "judge_removed", round_id=round_id, judge_container_id=container_id
        )

    def judge_policy_removed(self, round_id: str, rule_id: str) -> None:
        self.resource_event(
            "judge_policy_removed", round_id=round_id, policy_rule_id=rule_id
        )

    def judge_network_removed(self, round_id: str, network_id: str) -> None:
        self.resource_event(
            "judge_network_removed", round_id=round_id, network_id=network_id
        )

class RunCoordinator:
    """Own the complete run sequence through mandatory injected ports."""

    def __init__(
        self,
        *,
        backend: CoordinatorBackend,
        lease_store: LeaseStore,
        run_id_factory: Callable[[], str] = lambda: uuid.uuid4().hex,
        clock: Any,
    ) -> None:
        self._backend = backend
        self._leases = lease_store
        self._run_id_factory = run_id_factory
        self._clock = clock
        self.phase_history: tuple[RunStatus, ...] = ()

    def run(self, request: RunRequest) -> RunResult:
        run_id = str(self._run_id_factory())
        state = CoordinatorState(run_id=run_id)
        started_at = time.monotonic()
        lease: ResourceLease | None = None
        plan: Any = None
        artifacts: Any = None
        evaluator: _RoundEvaluator | None = None
        server: StartedSubmissionServer | None = None
        network: ManagedNetwork | None = None
        work: ContainerRef | None = None
        planned_workdir_volume: ManagedWorkdirVolume | None = None
        actual_workdir_volume: ManagedWorkdirVolume | None = None
        rollback_workdir_volume: ManagedWorkdirVolume | None = None
        workdir_volume_mounted = False
        workdir_volume_authority_known = False
        setup_recovery_required = False
        setup_recovery_error: str | None = None
        agent_stopped = False
        work_quiescence: WorkQuiescence | None = None
        planned_retained_work: str | None = None
        retained_work: RootfsSnapshotLease | None = None
        retained_work_rollback: RootfsSnapshotLease | None = None
        agent_result: AgentRunResult | None = None
        drained = False
        drain_failed = False
        work_removed = False
        network_removed = False
        work_policy_removed = False
        work_policy_id: str | None = None
        captured: BaseException | None = None
        cancelled = False
        cleanup_errors: list[str] = []
        retention_attempted = False
        retained_identity_durable = False
        cleanup_blocked = False
        authority = threading.RLock()
        resources = ExitStack()

        def write(updated: ResourceLease) -> None:
            nonlocal lease
            with authority:
                if workdir_volume_authority_known:
                    updated = updated.model_copy(
                        update={
                            "work": updated.work.model_copy(
                                update={
                                    "workdir_volume": WorkdirVolumeResourceLease(
                                        planned_name=(
                                            None
                                            if planned_workdir_volume is None
                                            else planned_workdir_volume.name
                                        ),
                                        planned_target=(
                                            None
                                            if planned_workdir_volume is None
                                            else planned_workdir_volume.target
                                        ),
                                        planned_snapshot_mode=(
                                            None
                                            if planned_workdir_volume is None
                                            else planned_workdir_volume.snapshot_mode
                                        ),
                                        planned_freshness_nonce=(
                                            None
                                            if planned_workdir_volume is None
                                            else planned_workdir_volume.freshness_nonce
                                        ),
                                        actual=actual_workdir_volume,
                                        rollback=rollback_workdir_volume,
                                        mounted=workdir_volume_mounted,
                                    )
                                }
                            )
                        }
                    )
                if setup_recovery_required:
                    marker = setup_recovery_error or (
                        "recovery_required: WORKDIR volume authority retained"
                    )
                    detail = updated.error
                    if detail and marker not in detail:
                        detail = f"{marker}; {detail}"
                    else:
                        detail = marker
                    updated = updated.model_copy(
                        update={
                            "recovery_required": True,
                            "error": detail,
                        }
                    )
                if evaluator is not None and evaluator.recovery_required:
                    marker = evaluator.recovery_error or (
                        "recovery_required: in-memory recovery authority retained"
                    )
                    detail = updated.error
                    if detail and marker not in detail:
                        detail = f"{marker}; {detail}"
                    else:
                        detail = marker
                    updated = updated.model_copy(
                        update={
                            "recovery_required": True,
                            "work": updated.work.model_copy(
                                update={"paused": True, "stopped": False}
                            ),
                            "error": detail,
                        }
                    )
                self._leases.write(updated)
                lease = updated

        def transition(status: RunStatus) -> None:
            state.transition(status)
            if lease is None:
                return
            history = lease.phase_history
            if history[-1:] != (status.value,):
                history += (status.value,)
            write(
                lease.model_copy(
                    update={"phase": status.value, "phase_history": history}
                )
            )

        def update_work(**updates: object) -> None:
            assert lease is not None
            write(
                lease.model_copy(
                    update={"work": lease.work.model_copy(update=updates)}
                )
            )

        def update_judge(**updates: object) -> None:
            assert lease is not None
            write(
                lease.model_copy(
                    update={"judge": lease.judge.model_copy(update=updates)}
                )
            )

        def persist_recovery(message: str) -> None:
            assert lease is not None
            # A mutation may have succeeded even when its first identity fsync
            # failed. Overlay every still-live coordinator-local authority on
            # the last durable lease before marking recovery required.
            work_updates: dict[str, object] = {}
            if work is not None and not work_removed:
                work_updates["container_id"] = work.container_id
                if work_quiescence is not None:
                    work_updates.update(
                        {
                            "planned_quiescence": None,
                            "paused": work_quiescence is WorkQuiescence.PAUSED,
                            "stopped": work_quiescence is WorkQuiescence.STOPPED,
                        }
                    )
            elif work_removed:
                work_updates.update(
                    {
                        "container_id": None,
                        "planned_container": None,
                        "paused": False,
                        "stopped": False,
                    }
                )
            if network is not None and not network_removed:
                work_updates.update(
                    {
                        "network_id": network.network_id,
                        "network_name": network.name,
                    }
                )
            if work_policy_id is not None and not work_policy_removed:
                # The planned ID was durably recorded before installation.
                # Preserve it even when a buggy backend returns a different
                # actual ID: recovery must retain both authorities.
                work_updates["policy_rule_id"] = work_policy_id
            if planned_retained_work is not None:
                work_updates["planned_retained_image_ref"] = planned_retained_work
            if retained_work is not None:
                work_updates.update(
                    {
                        "retained_image_id": retained_work.image_id,
                        "retained_image_ref": retained_work.image_ref,
                    }
                )
            if retained_work_rollback is not None:
                work_updates["retained_image_rollback"] = (
                    RetainedImageRollbackAuthority(
                        image_id=retained_work_rollback.image_id,
                        image_ref=retained_work_rollback.image_ref,
                    )
                )
            write(
                lease.model_copy(
                    update={
                        "recovery_required": True,
                        "work": lease.work.model_copy(update=work_updates),
                        "error": redact_text(message),
                    }
                )
            )

        def retain_workdir_authority_after_write_failure(
            stage: str, error: BaseException
        ) -> None:
            nonlocal setup_recovery_required, setup_recovery_error
            setup_recovery_required = True
            setup_recovery_error = redact_text(
                f"recovery_required: {stage}: {error}"
            )
            persist_recovery(setup_recovery_error)

        def emergency_cleanup() -> None:
            nonlocal drained, drain_failed, work_removed, network_removed
            nonlocal work_policy_removed, agent_stopped, work_quiescence
            nonlocal planned_retained_work, retained_work, retention_attempted
            nonlocal retained_work_rollback
            nonlocal cleanup_blocked, retained_identity_durable
            if (
                drain_failed
                or cleanup_blocked
                or (
                    work is not None
                    and retention_attempted
                    and not retained_identity_durable
                )
            ):
                return
            if server is not None and not drained:
                try:
                    server.owner.stop()
                    drained = True
                except BaseException as error:
                    drain_failed = True
                    cleanup_errors.append(self._error(error))
                    if lease is not None:
                        persist_recovery(f"server drain unproven: {error}")
                    return
            if (
                (lease is not None and lease.recovery_required)
                or (evaluator is not None and evaluator.recovery_required)
                or setup_recovery_required
            ):
                return
            if work is not None and not work_removed and not agent_stopped:
                try:
                    self._backend.stop_agent(work)
                    agent_stopped = True
                except BaseException as error:
                    cleanup_errors.append(self._error(error))
                    cleanup_blocked = True
                    if lease is not None:
                        persist_recovery(f"Agent stop is unproven: {error}")
                    return
            if work is not None and not work_removed and work_quiescence is None:
                try:
                    update_work(planned_quiescence="pause-if-running")
                    observed = self._backend.quiesce_work(work)
                    work_quiescence = WorkQuiescence(observed)
                    update_work(
                        planned_quiescence=None,
                        paused=work_quiescence is WorkQuiescence.PAUSED,
                        stopped=work_quiescence is WorkQuiescence.STOPPED,
                    )
                except BaseException as error:
                    cleanup_errors.append(self._error(error))
                    cleanup_blocked = True
                    if lease is not None:
                        persist_recovery(f"Work quiescence is unproven: {error}")
                    return
            if (
                work is not None
                and not work_removed
                and not retention_attempted
                and plan is not None
            ):
                retention_attempted = True
                try:
                    planned_retained_work = self._backend.plan_retained_work(
                        plan, work
                    )
                    update_work(
                        planned_retained_image_ref=planned_retained_work
                    )
                    retained_work = self._backend.retain_work(
                        plan, work, planned_retained_work
                    )
                    if (
                        retained_work.purpose != "retained-work"
                        or retained_work.run_id != run_id
                        or retained_work.task_id != plan.task.task_id
                        or retained_work.round_id != "final"
                        or retained_work.source_container_id != work.container_id
                        or retained_work.image_ref != planned_retained_work
                    ):
                        mismatched = retained_work
                        try:
                            self._backend.release_retained_work(mismatched)
                        except BaseException as rollback_error:
                            retained_work_rollback = mismatched
                            retained_work = None
                            raise InfrastructureError(
                                "retained Work image lease differs from durable "
                                "plan; rollback failed: "
                                f"{rollback_error}"
                            ) from rollback_error
                        retained_work = None
                        raise InfrastructureError(
                            "retained Work image lease differs from durable plan"
                        )
                    try:
                        update_work(
                            retained_image_id=retained_work.image_id,
                            retained_image_ref=retained_work.image_ref,
                        )
                        retained_identity_durable = True
                    except BaseException as identity_error:
                        rollback_error: BaseException | None = None
                        try:
                            self._backend.release_retained_work(retained_work)
                        except BaseException as error:
                            rollback_error = error
                        else:
                            retained_work = None
                        detail = (
                            f"retained Work image identity fsync failed: "
                            f"{identity_error}"
                        )
                        if rollback_error is not None:
                            detail += (
                                "; retained Work image rollback failed: "
                                f"{rollback_error}"
                            )
                        raise InfrastructureError(detail) from identity_error
                except BaseException as error:
                    cleanup_errors.append(self._error(error))
                    cleanup_blocked = True
                    if lease is not None:
                        persist_recovery(f"retained Work image failed: {error}")
                    return
            if work is not None and not work_removed:
                try:
                    self._backend.remove_work(work)
                    work_removed = True
                    update_work(
                        container_id=None,
                        planned_container=None,
                        paused=False,
                        stopped=False,
                    )
                except BaseException as error:
                    cleanup_errors.append(self._error(error))
                    if lease is not None:
                        persist_recovery(f"Work removal unproven: {error}")
                    return
            if work_policy_id is not None and not work_policy_removed:
                try:
                    self._backend.remove_work_policy(work_policy_id)
                    work_policy_removed = True
                    update_work(
                        planned_policy_rule_id=None, policy_rule_id=None
                    )
                except BaseException as error:
                    cleanup_errors.append(self._error(error))
                    if lease is not None:
                        persist_recovery(
                            f"Work policy removal unproven: {error}"
                        )
                    return
            if network is not None and not network_removed:
                try:
                    self._backend.remove_network(network)
                    network_removed = True
                    update_work(
                        network_id=None,
                        network_name=None,
                        planned_network=None,
                    )
                except BaseException as error:
                    cleanup_errors.append(self._error(error))
                    if lease is not None:
                        persist_recovery(f"Work network removal unproven: {error}")

        try:
            with _cancellation_signals(), self._leases.lock(run_id):
                try:
                    definition = self._backend.compile(request)
                    gpu_plan = self._backend.allocate(
                        definition, tuple(request.gpu_selectors)
                    )
                    lease = ResourceLease(
                        run_id=run_id,
                        task_id=definition.task_id,
                        coordinator_pid=os.getpid(),
                        coordinator_started_at=time.time(),
                        phase=RunStatus.PREPARING.value,
                        gpu_plan=gpu_plan,
                    )
                    write(lease)
                    self._backend.record_event("gpu_plan", gpu_plan)
                    images = self._backend.prepare_images(definition)
                    plan = self._backend.prepare_plan(
                        definition, images, gpu_plan, request, run_id
                    )
                    validate_run_plan_mount_topology(plan)
                    write(
                        lease.model_copy(
                            update={
                                "rootfs_snapshot_mode": plan.rootfs_snapshot_mode
                            }
                        )
                    )
                    artifacts = self._backend.start_artifacts(plan, run_id)
                    evaluator = _RoundEvaluator(
                        backend=self._backend,
                        state=state,
                        transition=transition,
                        persist_recovery=persist_recovery,
                        update_work=update_work,
                        update_judge=update_judge,
                    )
                    server = self._backend.start_server(
                        evaluator, artifacts, self._clock
                    )
                    resources.callback(emergency_cleanup)

                    planned_network = self._backend.plan_network(plan, run_id)
                    update_work(planned_network=planned_network)
                    network = self._backend.create_network(
                        plan, run_id, planned_network
                    )
                    update_work(
                        network_id=network.network_id,
                        network_name=network.name,
                    )
                    resources.callback(emergency_cleanup)

                    if (
                        plan.rootfs_snapshot_mode
                        is RootfsSnapshotMode.SPLIT_WORKDIR
                    ):
                        planned_workdir_volume = (
                            self._backend.plan_workdir_volume(plan, run_id)
                        )
                        workdir_volume_authority_known = True
                        if planned_workdir_volume is None:
                            raise InfrastructureError(
                                "split-workdir mode requires planned WORKDIR "
                                "volume authority"
                            )
                        try:
                            evaluator.resource_event(
                                "workdir_volume_planned",
                                planned=planned_workdir_volume,
                            )
                        except BaseException as error:
                            retain_workdir_authority_after_write_failure(
                                "planned WORKDIR volume fsync failed", error
                            )
                            raise
                        try:
                            created_workdir_volume = (
                                self._backend.create_workdir_volume(
                                    plan,
                                    run_id,
                                    planned_workdir_volume,
                                )
                            )
                        except InfrastructureError as error:
                            retain_workdir_authority_after_write_failure(
                                "WORKDIR volume create outcome is unproven",
                                error,
                            )
                            raise
                        except SetupError:
                            planned_workdir_volume = None
                            actual_workdir_volume = None
                            rollback_workdir_volume = None
                            workdir_volume_mounted = False
                            evaluator.resource_event("workdir_volume_removed")
                            raise
                        if not isinstance(
                            created_workdir_volume, ManagedWorkdirVolume
                        ):
                            raise InfrastructureError(
                                "WORKDIR volume create returned untyped authority"
                            )
                        if (
                            created_workdir_volume != planned_workdir_volume
                        ):
                            mismatch = InfrastructureError(
                                "created WORKDIR volume differs from exact "
                                "split-workdir authority"
                            )
                            actual_workdir_volume = None
                            rollback_workdir_volume = created_workdir_volume
                            try:
                                evaluator.resource_event(
                                    "workdir_volume_created",
                                    planned=planned_workdir_volume,
                                    rollback=rollback_workdir_volume,
                                )
                            except BaseException as error:
                                retain_workdir_authority_after_write_failure(
                                    "mismatched WORKDIR volume rollback identity "
                                    "fsync failed",
                                    error,
                                )
                                raise
                            try:
                                self._backend.remove_workdir_volume(
                                    rollback_workdir_volume
                                )
                            except BaseException as rollback_error:
                                retain_workdir_authority_after_write_failure(
                                    "mismatched WORKDIR volume rollback is "
                                    "unproven",
                                    rollback_error,
                                )
                                raise InfrastructureError(
                                    "recovery_required: mismatched WORKDIR volume "
                                    "rollback is unproven: "
                                    f"{rollback_error}"
                                ) from mismatch
                            planned_workdir_volume = None
                            actual_workdir_volume = None
                            rollback_workdir_volume = None
                            workdir_volume_mounted = False
                            evaluator.resource_event("workdir_volume_removed")
                            raise mismatch
                        actual_workdir_volume = created_workdir_volume
                        rollback_workdir_volume = None
                        try:
                            evaluator.resource_event(
                                "workdir_volume_created",
                                planned=planned_workdir_volume,
                                actual=actual_workdir_volume,
                            )
                        except BaseException as error:
                            retain_workdir_authority_after_write_failure(
                                "actual WORKDIR volume identity fsync failed",
                                error,
                            )
                            raise
                    elif (
                        plan.rootfs_snapshot_mode
                        is not RootfsSnapshotMode.FULL_ROOTFS
                    ):
                        raise InfrastructureError(
                            "unsupported rootfs snapshot mode"
                        )

                    planned_work = self._backend.plan_work_container(plan, run_id)
                    update_work(planned_container=planned_work)
                    work = self._backend.create_work(
                        plan,
                        run_id,
                        network,
                        planned_work,
                        actual_workdir_volume,
                    )
                    update_work(container_id=work.container_id)
                    resources.callback(emergency_cleanup)
                    self._backend.attest_work_feedback_mount(work)
                    if actual_workdir_volume is not None:
                        self._backend.attest_workdir_volume(
                            work, actual_workdir_volume
                        )
                        workdir_volume_mounted = True
                        try:
                            evaluator.resource_event(
                                "workdir_volume_mounted",
                                actual=actual_workdir_volume,
                            )
                        except BaseException as error:
                            retain_workdir_authority_after_write_failure(
                                "mounted WORKDIR volume fsync failed", error
                            )
                            raise
                    planned_work_policy = self._backend.plan_work_policy(
                        plan, work, network
                    )
                    update_work(
                        planned_policy_rule_id=planned_work_policy
                    )
                    try:
                        work_policy_id = self._backend.install_work_policy(
                            plan, work, network
                        )
                    except BaseException as error:
                        if "recovery_required" in str(error):
                            work_policy_id = planned_work_policy
                            update_work(policy_rule_id=planned_work_policy)
                            persist_recovery(str(error))
                        raise
                    if work_policy_id != planned_work_policy:
                        raise InfrastructureError(
                            "installed Work policy identity differs from plan"
                        )
                    update_work(policy_rule_id=work_policy_id)
                    resources.callback(emergency_cleanup)
                    self._backend.start_work(work)

                    token = server.service.register(
                        run_id=run_id,
                        run_plan=plan,
                        work_container=work,
                        max_submissions=request.options.max_submissions,
                        cooldown_seconds=request.options.cooldown_seconds,
                    )
                    self._backend.install_hooks(
                        plan, work, server.endpoint.url, token
                    )
                    prepared = self._backend.prepare_agent(
                        plan, request.options.max_submissions
                    )
                    transition(RunStatus.AGENT_RUNNING)
                    resources.callback(emergency_cleanup)
                    agent_result = self._backend.run_agent(
                        prepared, work, plan.task.agent.timeout_seconds
                    )
                except (KeyboardInterrupt, _RunCancelled) as error:
                    cancelled = True
                    captured = error
                except BaseException as error:
                    captured = error

                # Acceptance closes and all in-flight submissions drain before
                # history, aggregation, artifacts, Agent, or Work are touched.
                emergency_cleanup()
                reports = (
                    self._history(server, evaluator) if drained else ()
                )
                infrastructure_round = any(
                    report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
                    for report in reports
                )
                if cancelled:
                    terminal = RunStatus.CANCELLED
                elif captured is not None or cleanup_errors or infrastructure_round:
                    terminal = RunStatus.FAILED
                elif agent_result is not None and agent_result.cancelled:
                    terminal = RunStatus.CANCELLED
                elif agent_result is not None and agent_result.timed_out:
                    terminal = RunStatus.FAILED
                elif any(
                    report.status == SubmissionStatus.COMPLETED
                    for report in reports
                ):
                    terminal = RunStatus.COMPLETED
                else:
                    terminal = RunStatus.NO_VALID_SUBMISSION

                engine_bug = isinstance(captured, StateTransitionError)
                if engine_bug:
                    terminal = RunStatus.FAILED
                if artifacts is not None:
                    diagnostic = "; ".join(
                        tuple(cleanup_errors)
                        + (() if captured is None else (self._error(captured),))
                    )
                    if diagnostic:
                        try:
                            artifacts.record_engine_error(diagnostic)
                        except BaseException as error:
                            cleanup_errors.append(self._error(error))
                            terminal = RunStatus.FAILED
                    try:
                        artifacts.finalize(
                            status=terminal,
                            runtime_seconds=max(
                                0.0, time.monotonic() - started_at
                            ),
                            timed_out=bool(
                                agent_result and agent_result.timed_out
                            ),
                        )
                    except BaseException as error:
                        cleanup_errors.append(self._error(error))
                        terminal = RunStatus.FAILED

                if state.status not in _TERMINAL:
                    transition(terminal)
                if lease is not None:
                    error_text = (
                        None
                        if captured is None and not cleanup_errors
                        else "; ".join(
                            tuple(cleanup_errors)
                            + (() if captured is None else (self._error(captured),))
                        )
                    )
                    write(
                        lease.model_copy(
                            update={
                                "phase": state.status.value,
                                "status": state.status,
                                "error": error_text,
                            }
                        )
                    )
                result = aggregate_result(
                    run_id=run_id,
                    reports=reports,
                    primary_reward=(
                        None if plan is None else plan.task.verifier.primary_reward
                    ),
                    direction=(
                        "maximize" if plan is None else plan.task.score_direction
                    ),
                    terminal=state.status,
                )
                if engine_bug:
                    assert isinstance(captured, StateTransitionError)
                    raise captured
                return result
        finally:
            resources.close()
            self.phase_history = state.history

    @staticmethod
    def _history(
        server: StartedSubmissionServer | None,
        evaluator: _RoundEvaluator | None,
    ) -> tuple[SubmissionReport, ...]:
        if server is not None:
            reports = tuple(server.service.reports)
            if reports:
                return reports
        return () if evaluator is None else tuple(evaluator.reports)

    @staticmethod
    def _error(error: BaseException) -> str:
        return redact_text(f"{type(error).__name__}: {error}")


def aggregate_result(
    *,
    run_id: str,
    reports: Sequence[SubmissionReport],
    primary_reward: str | None,
    direction: Literal["maximize", "minimize"],
    terminal: RunStatus | None = None,
) -> RunResult:
    """Aggregate valid rewards per key without inventing a multi-objective scalar."""
    history = tuple(reports)
    valid = tuple(
        report for report in history if report.status == SubmissionStatus.COMPLETED
    )
    keys = {key for report in valid for key in report.rewards}
    selected = primary_reward
    if selected is None:
        if "reward" in keys:
            selected = "reward"
        elif len(keys) == 1:
            selected = next(iter(keys))

    choose = min if direction == "minimize" else max
    best_rewards = {
        key: choose(report.rewards[key] for report in valid if key in report.rewards)
        for key in sorted(keys)
    }
    scalar_candidates = (
        ()
        if selected is None
        else tuple(report for report in valid if selected in report.rewards)
    )
    best_report: SubmissionReport | None = None
    if scalar_candidates:
        key = selected
        assert key is not None

        def ranking(report: SubmissionReport) -> float:
            return report.rewards[key]

        best_report = (
            min(scalar_candidates, key=ranking)
            if direction == "minimize"
            else max(scalar_candidates, key=ranking)
        )
    status = terminal
    if status is None:
        status = RunStatus.COMPLETED if valid else RunStatus.NO_VALID_SUBMISSION
    return RunResult(
        run_id=run_id,
        status=status,
        total_rounds=len(history),
        best_score=(
            None
            if best_report is None or selected is None
            else best_report.rewards[selected]
        ),
        best_round=None if best_report is None else best_report.round_id,
        best_rewards=FrozenRewardMap(best_rewards),
        reports=history,
    )


__all__ = [
    "CoordinatorBackend",
    "CoordinatorState",
    "EmbeddedSubmissionServerFactory",
    "ProductionCoordinatorBackend",
    "RoundLifecycleObserver",
    "RunPreparation",
    "RunCoordinator",
    "StartedSubmissionServer",
    "aggregate_result",
]
