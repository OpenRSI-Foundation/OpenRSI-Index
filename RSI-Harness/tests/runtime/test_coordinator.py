from __future__ import annotations

import json
import signal
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import (
    InfrastructureError,
    RetryableSubmissionError,
    SetupError,
    StateTransitionError,
)
from rsi_harness.models import (
    AgentRunResult,
    CompileOptions,
    ContainerRef,
    EvaluationRequest,
    GPUAllocation,
    GPUDevice,
    GPURequirement,
    JudgeGPUMode,
    ManagedNetwork,
    ManagedWorkdirVolume,
    RootfsSnapshotLease,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunRequest,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.coordinator import (
    CoordinatorState,
    EmbeddedSubmissionServerFactory,
    ProductionCoordinatorBackend,
    RunCoordinator,
    StartedSubmissionServer,
    aggregate_result,
)
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.submissions import JudgeEndpoint, SubmissionService
from tests.factories import make_run_plan


class ScriptedBackend:
    def __init__(self, tmp_path: Path) -> None:
        self.plan = make_run_plan(tmp_path)
        devices = tuple(
            GPUDevice(index=index, uuid=f"GPU-{letter}", name="Test GPU")
            for index, letter in enumerate("abcd")
        )
        self.gpu_plan = RunGPUPlan(
            authorized_pool=GPUAllocation(devices=devices),
            work=GPUAllocation(devices=devices[:2]),
            judge=GPUAllocation(devices=devices[2:]),
            judge_mode=JudgeGPUMode.DISJOINT,
        )
        self.events: list[tuple[str, object]] = []
        self.service = None
        self.artifacts = None
        self.submit_token: str | None = None
        self.submit_url: str | None = None
        self.prepared_max_submissions: int | None = None
        self._next_report = 0
        self.agent_result = AgentRunResult(exit_code=0)
        self.fail_at: str | None = None
        self.fail_server_stop = False
        self.fail_work_remove = False
        self.fail_work_policy_install_recovery = False
        self.fail_artifact_finalize = False
        self.fail_stop_agent = False
        self.fail_retain_work = False
        self.fail_retained_release = False
        self.work_quiescence = "paused"
        self.retained_image_ref: str | None = None
        self.require_quiescence_plan = False
        self.cancel_snapshot_before_acquire = False
        self.fail_workdir_volume_create_absent = False
        self.fail_workdir_volume_create_ambiguous = False
        self.fail_work_create = False
        self.fail_workdir_volume_attest = False
        self.fail_work_feedback_attest = False
        self.fail_workdir_volume_remove: str | None = None
        self.return_volume_for_full_rootfs = False
        self.return_none_for_split_workdir = False
        self.store: LeaseStore | None = None
        self.workdir_volume = ManagedWorkdirVolume(
            name=f"rsi-harness-workdir-{'1' * 64}",
            run_id="run-1",
            task_id=self.plan.task.task_id,
            target=self.plan.workdir,
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        )
        self.planned_workdir_volume = self.workdir_volume
        self.planned_workdir_volume_name = self.workdir_volume.name
        self.reports = [
            SubmissionReport(
                round_id="agent-1",
                status=SubmissionStatus.COMPLETED,
                rewards={"reward": 0.25, "latency": 7},
                score=0.25,
            ),
            SubmissionReport(
                round_id="agent-2",
                status=SubmissionStatus.COMPLETED,
                rewards={"reward": 1.0, "latency": 5},
                score=1.0,
            ),
        ]

    def compile(self, request: RunRequest):
        self.events.append(("compiler", request.task_dir))
        if self.fail_at == "compile":
            raise RuntimeError("setup bearer SUPER-SECRET must be redacted")
        return self.plan.task

    def allocate(self, definition, selectors):
        self.events.append(("allocation", tuple(selectors)))
        return self.gpu_plan

    def prepare_images(self, definition):
        self.events.append(("images", definition.task_id))
        return self.plan.images

    def prepare_plan(self, definition, images, allocation, request, run_id):
        self.events.append(("preflight", run_id))
        return self.plan

    def initialize_workspace(self, plan, run_id):
        del plan, run_id
        raise AssertionError("production initialized a host workspace")

    def start_artifacts(self, plan, run_id):
        self.events.append(("artifacts", run_id))
        self.artifacts = ArtifactRecorder(
            self.events, fail_finalize=self.fail_artifact_finalize
        )
        return self.artifacts

    def start_server(self, evaluator, artifacts, clock):
        self.events.append(("server_start", None))
        self.service = Service(evaluator, self.reports)
        return StartedSubmissionServer(
            service=self.service,
            owner=Server(self.events, fail=self.fail_server_stop),
            endpoint=JudgeEndpoint(
                url="http://172.30.0.1:9020",
                bind_host="127.0.0.1",
                port=9020,
            ),
        )

    def retain_workspace(self, workspace):
        del workspace
        raise AssertionError("production retained a host workspace")

    def plan_network(self, plan, run_id):
        del plan
        return f"rsi-{run_id}-work"

    def create_network(self, plan, run_id, planned):
        del plan
        self.events.append(("network_create", planned))
        if self.store is not None:
            assert self.store.read(run_id).work.planned_network == planned
        return ManagedNetwork(
            network_id="network-work-1",
            name=planned,
            run_id=run_id,
            task_id=self.plan.task.task_id,
            role="work",
            internal=False,
        )

    def remove_network(self, network):
        self.events.append(("network_remove", network.network_id))

    def plan_work_container(self, plan, run_id):
        planned = f"rsi-{run_id}-{plan.task.task_id}-work"
        self.events.append(("work_plan", planned))
        return planned

    def plan_workdir_volume(self, plan, run_id):
        del run_id
        self.events.append(("volume_plan", plan.rootfs_snapshot_mode))
        if plan.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS:
            return (
                self.planned_workdir_volume
                if self.return_volume_for_full_rootfs
                else None
            )
        if self.return_none_for_split_workdir:
            return None
        return self.planned_workdir_volume

    def create_workdir_volume(self, plan, run_id, planned):
        self.events.append(("volume_create", planned.name))
        if self.store is not None:
            durable = self.store.read(run_id)
            assert durable is not None
            assert durable.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR
            assert durable.work.workdir_volume.planned_name == planned.name
            assert durable.work.workdir_volume.planned_target == plan.workdir
            assert durable.work.workdir_volume.planned_snapshot_mode is (
                RootfsSnapshotMode.SPLIT_WORKDIR
            )
            assert (
                durable.work.workdir_volume.planned_freshness_nonce
                == planned.freshness_nonce
            )
            assert durable.work.workdir_volume.actual is None
            assert durable.work.workdir_volume.rollback is None
        if self.fail_workdir_volume_create_absent:
            raise SetupError("managed WORKDIR volume create is proven absent")
        if self.fail_workdir_volume_create_ambiguous:
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume create outcome is ambiguous"
            )
        return self.workdir_volume

    def attest_workdir_volume(self, work, volume):
        self.events.append(("volume_attest_work_mount", work.container_id))
        if self.fail_workdir_volume_attest:
            raise InfrastructureError(
                "recovery_required: Work WORKDIR mount attestation failed"
            )
        assert volume == self.workdir_volume

    def attest_work_feedback_mount(self, work):
        self.events.append(("feedback_attest_work_mount", work.container_id))
        if self.fail_work_feedback_attest:
            raise InfrastructureError(
                "recovery_required: Work feedback mount attestation failed"
            )

    def remove_workdir_volume(self, volume):
        self.events.append(("volume_remove", volume.name))
        if self.fail_workdir_volume_remove is not None:
            raise InfrastructureError(self.fail_workdir_volume_remove)

    def create_work(self, plan, run_id, network, planned_name, workdir_volume):
        del network, planned_name
        self.events.append(("work_create", "work-1"))
        if self.store is not None:
            durable = self.store.read(run_id)
            assert durable.work.planned_container is not None
            if plan.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR:
                assert durable.work.workdir_volume.actual == workdir_volume
            else:
                assert workdir_volume is None
        if self.fail_work_create:
            raise RuntimeError("Work create failed after volume creation")
        return ContainerRef(container_id="work-1", role="work")

    def start_work(self, work):
        self.events.append(("work_start", work.container_id))
        if self.store is not None:
            durable = self.store.read("run-1")
            assert durable.work.container_id == work.container_id
            assert durable.work.policy_rule_id == "policy-work-1"
            if durable.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR:
                assert durable.work.workdir_volume.mounted is True

    def plan_work_policy(self, plan, work, network):
        del plan, network
        return f"policy-{work.container_id}"

    def install_work_policy(self, plan, work, network):
        del plan, network
        self.events.append(("work_policy_install", work.container_id))
        if self.store is not None:
            assert (
                self.store.read("run-1").work.planned_policy_rule_id == "policy-work-1"
            )
        if self.fail_work_policy_install_recovery:
            raise RuntimeError(
                "recovery_required: partial Work policy rollback is unproven"
            )
        return f"policy-{work.container_id}"

    def remove_work_policy(self, rule_id):
        self.events.append(("work_policy_remove", rule_id))

    def remove_work(self, work):
        self.events.append(("work_remove", work.container_id))
        if self.fail_work_remove:
            raise RuntimeError("Work removal failed")

    def install_hooks(self, plan, work, submit_url, token):
        self.events.append(("hooks", work.container_id))
        self.submit_token = token
        self.submit_url = submit_url

    def prepare_agent(self, plan, max_submissions=None):
        self.events.append(("prompt", plan.task.task_id))
        self.prepared_max_submissions = max_submissions
        if self.fail_at == "prepare_agent":
            raise RuntimeError("prepare failed after hooks")
        return object()

    def run_agent(self, prepared, work, timeout):
        del prepared, timeout
        self.events.append(("agent_start", work.container_id))
        assert self.service is not None
        if isinstance(self.service, Service):
            self.service.submit()
        else:
            assert self.submit_token is not None
            self.service.submit(self.submit_token)
        self.events.append(("agent_writes", work.container_id))
        if isinstance(self.service, Service):
            self.service.submit()
        else:
            assert self.submit_token is not None
            self.service.submit(self.submit_token)
        return self.agent_result

    def stop_agent(self, work):
        self.events.append(("agent_stop", work.container_id))
        if self.fail_stop_agent:
            raise RuntimeError("Agent stop authority is unproven")

    def quiesce_work(self, work):
        if self.require_quiescence_plan:
            assert self.store is not None
            durable = self.store.read("run-1")
            assert durable is not None
            assert durable.work.planned_quiescence == "pause-if-running"
            assert durable.work.paused is False
            assert durable.work.stopped is False
        event = "work_pause" if self.work_quiescence == "paused" else "work_stopped"
        self.events.append((event, work.container_id))
        return self.work_quiescence

    def plan_retained_work(self, plan, work):
        del plan
        self.events.append(("retained_plan", work.container_id))
        return f"rsi-harness-rootfs:retained-work-{'a' * 64}"

    def retain_work(self, plan, work, planned_ref):
        self.events.append(("retained_acquire", work.container_id))
        if self.store is not None:
            durable = self.store.read("run-1")
            assert durable is not None
            assert durable.work.planned_retained_image_ref == planned_ref
        if self.fail_retain_work:
            raise RuntimeError("retained Work commit failed")
        return RootfsSnapshotLease(
            lease_id="retained-lease-1",
            purpose="retained-work",
            run_id="run-1",
            task_id=plan.task.task_id,
            round_id="final",
            source_container_id=work.container_id,
            image_id=f"sha256:{'b' * 64}",
            image_ref=self.retained_image_ref or planned_ref,
        )

    def release_retained_work(self, retained):
        self.events.append(("retained_rollback", retained.image_id))
        if self.fail_retained_release:
            raise RuntimeError("recovery_required: retained image rollback is unproven")

    def evaluate_submission(self, request, *, lifecycle_observer):
        report = self.reports[self._next_report].model_copy(
            update={"round_id": request.round_id}
        )
        self._next_report += 1
        lifecycle_observer.resource_event(
            "work_pause_planned",
            work_container_id=request.work_container.container_id,
        )
        planned_snapshot_ref = f"rsi-harness-rootfs:judge-round-{'a' * 64}"
        lifecycle_observer.resource_event(
            "snapshot_planned",
            round_id=request.round_id,
            planned_snapshot_ref=planned_snapshot_ref,
            source_work_container_id=request.work_container.container_id,
        )
        if self.store is not None:
            planned = self.store.read("run-1")
            assert planned is not None
            assert planned.judge.planned_snapshot_ref == planned_snapshot_ref
            assert planned.judge.snapshot_image_id is None
        if self.cancel_snapshot_before_acquire:
            lifecycle_observer.resource_event(
                "snapshot_cancelled",
                round_id=request.round_id,
                planned_snapshot_ref=planned_snapshot_ref,
                source_work_container_id=request.work_container.container_id,
            )
            if self.store is not None:
                cancelled = self.store.read("run-1")
                assert cancelled is not None
                assert cancelled.judge.planned_snapshot is None
                assert cancelled.judge.planned_snapshot_ref is None
                assert cancelled.judge.snapshot_image_id is None
            lifecycle_observer.resource_event(
                "work_unpaused",
                work_container_id=request.work_container.container_id,
            )
            report = report.model_copy(
                update={
                    "status": SubmissionStatus.INFRASTRUCTURE_ERROR,
                    "score": None,
                    "error": "snapshot acquire failed before commit",
                }
            )
            assert self.artifacts is not None
            self.artifacts.record_submission(report)
            return report
        lifecycle_observer.resource_event(
            "snapshot_acquired",
            snapshot_lease_id=f"snapshot:{request.round_id}",
            snapshot_merged_path=str(request.run_plan.paths.workspace),
            snapshot_process_id=None,
            snapshot_image_id=f"sha256:{'b' * 64}",
            snapshot_image_ref=planned_snapshot_ref,
            snapshot_source_container_id=request.work_container.container_id,
        )
        if self.store is not None:
            actual = self.store.read("run-1")
            assert actual is not None
            assert actual.judge.snapshot_image_id == f"sha256:{'b' * 64}"
            assert actual.judge.snapshot_image_ref == planned_snapshot_ref
            assert actual.judge.snapshot_source_container_id == "work-1"
        lifecycle_observer.resource_event(
            "judge_network_planned",
            round_id=request.round_id,
            planned_name=f"network:{request.round_id}",
        )
        lifecycle_observer.resource_event(
            "judge_network_created",
            round_id=request.round_id,
            network_id=f"network:{request.round_id}",
            network_name=f"network:{request.round_id}",
        )
        lifecycle_observer.resource_event("judge_planned", round_id=request.round_id)
        lifecycle_observer.resource_event(
            "judge_created",
            judge_container_id=f"judge:{request.round_id}",
        )
        lifecycle_observer.resource_event(
            "judge_policy_planned",
            round_id=request.round_id,
            policy_rule_id=f"policy:{request.round_id}",
        )
        lifecycle_observer.resource_event(
            "judge_policy_installed",
            round_id=request.round_id,
            policy_rule_id=f"policy:{request.round_id}",
        )
        self.record_event("judge_create", request.round_id)
        assert self.artifacts is not None
        self.artifacts.record_submission(report)
        if report.error is not None and "recovery_required" in report.error:
            lifecycle_observer.resource_event(
                "recovery_required", recovery_context=report.error
            )
            return report
        lifecycle_observer.resource_event(
            "judge_removed", judge_container_id=f"judge:{request.round_id}"
        )
        lifecycle_observer.resource_event(
            "judge_policy_removed",
            round_id=request.round_id,
            policy_rule_id=f"policy:{request.round_id}",
        )
        lifecycle_observer.resource_event(
            "judge_network_removed",
            round_id=request.round_id,
            network_id=f"network:{request.round_id}",
        )
        lifecycle_observer.resource_event(
            "snapshot_released",
            snapshot_lease_id=f"snapshot:{request.round_id}",
        )
        lifecycle_observer.resource_event(
            "work_unpaused",
            work_container_id=request.work_container.container_id,
        )
        return report

    def record_event(self, name, value):
        self.events.append((name, value))


class ArtifactRecorder:
    def __init__(self, events, *, fail_finalize: bool = False) -> None:
        self.events = events
        self.fail_finalize = fail_finalize
        self.reports: list[SubmissionReport] = []
        self.engine_errors: list[str] = []

    def record_submission(self, report) -> None:
        self.reports.append(report)

    def record_engine_error(self, error: str) -> None:
        self.engine_errors.append(error)

    def finalize(self, **kwargs) -> None:
        self.events.append(("artifacts_finalize", kwargs["status"]))
        if self.fail_finalize:
            raise RuntimeError("artifact durability failed")


class Service:
    def __init__(self, evaluator, reports) -> None:
        del reports
        self.evaluator = evaluator
        self.reports: list[SubmissionReport] = []
        self.rounds = 0
        self.registered = False

    def register(self, **kwargs):
        self.registered = True
        self.work = kwargs["work_container"]
        self.plan = kwargs["run_plan"]
        return "runtime-secret-token"

    def submit(self) -> None:
        self.rounds += 1
        round_id = f"agent-{self.rounds}"
        report = self.evaluator.evaluate(
            EvaluationRequest(
                run_plan=self.plan,
                work_container=self.work,
                round_id=round_id,
                verifier_logs=self.plan.paths.logs / "verifier" / round_id,
                verifier_output=(
                    self.plan.paths.logs / "feedback" / f"{round_id}.log"
                ),
            )
        )
        self.reports.append(report)

    def close(self, **kwargs) -> None:
        pass


class Server:
    def __init__(self, events, *, fail: bool = False) -> None:
        self.events = events
        self.fail = fail

    def stop(self) -> None:
        self.events.append(("server_stop", None))
        if self.fail:
            raise RuntimeError("Authorization: Bearer SERVER-SECRET")


@dataclass
class Clock:
    value: float = 0.0

    def now(self):
        from datetime import UTC, datetime

        return datetime.fromtimestamp(self.value, UTC)


def test_one_gpu_plan_is_durable_before_all_runtime_consumers(tmp_path) -> None:
    """A blank first lease or recomputation would split GPU authority."""

    class OneShotBackend(ScriptedBackend):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.compile_calls = 0
            self.allocate_calls = 0
            self.work_gpu_plan = None
            self.judge_gpu_plan = None

        def compile(self, request: RunRequest):
            self.compile_calls += 1
            assert self.compile_calls == 1
            return super().compile(request)

        def allocate(self, definition, selectors):
            self.allocate_calls += 1
            assert self.allocate_calls == 1
            assert definition is self.plan.task
            return super().allocate(definition, selectors)

        def prepare_images(self, definition):
            assert definition is self.plan.task
            return super().prepare_images(definition)

        def prepare_plan(self, definition, images, allocation, request, run_id):
            assert definition is self.plan.task
            assert allocation is self.gpu_plan
            self.plan = self.plan.model_copy(update={"gpu_plan": allocation})
            return super().prepare_plan(
                definition, images, allocation, request, run_id
            )

        def create_work(self, plan, run_id, network, planned_name, workdir_volume):
            assert plan.gpu_plan is self.gpu_plan
            self.work_gpu_plan = plan.gpu_plan
            return super().create_work(
                plan, run_id, network, planned_name, workdir_volume
            )

        def evaluate_submission(self, request, *, lifecycle_observer):
            assert request.run_plan.gpu_plan is self.gpu_plan
            self.judge_gpu_plan = request.run_plan.gpu_plan
            return super().evaluate_submission(
                request, lifecycle_observer=lifecycle_observer
            )

    backend = OneShotBackend(tmp_path)

    class GPUPlanFirstStore(LeaseStore):
        def write(self, lease) -> None:
            assert lease.gpu_plan is not None
            super().write(lease)

    store = GPUPlanFirstStore(tmp_path / "leases")
    backend.store = store

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.COMPLETED
    assert backend.compile_calls == 1
    assert backend.allocate_calls == 1
    assert backend.work_gpu_plan is backend.gpu_plan
    assert backend.judge_gpu_plan is backend.gpu_plan
    assert backend.work_gpu_plan.work.uuids == ("GPU-a", "GPU-b")
    assert backend.judge_gpu_plan.judge.uuids == ("GPU-c", "GPU-d")
    assert [value for name, value in backend.events if name == "gpu_plan"] == [
        backend.gpu_plan
    ]
    durable = store.read("run-1")
    assert durable is not None
    assert durable.gpu_plan == backend.gpu_plan


def test_gpu_plan_is_durable_before_image_preparation(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)

    class RecordingStore(LeaseStore):
        def write(self, lease) -> None:
            super().write(lease)
            if lease.gpu_plan is not None:
                backend.events.append(("lease_gpu_plan", lease.gpu_plan))

    store = RecordingStore(tmp_path / "leases")
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    durable = next(value for name, value in backend.events if name == "lease_gpu_plan")
    assert durable.authorized_pool.uuids == ("GPU-a", "GPU-b", "GPU-c", "GPU-d")
    assert durable.work.uuids == ("GPU-a", "GPU-b")
    assert durable.judge.uuids == ("GPU-c", "GPU-d")
    assert durable.judge_mode is JudgeGPUMode.DISJOINT
    names = [name for name, _ in backend.events]
    assert (
        names.index("allocation")
        < names.index("lease_gpu_plan")
        < names.index("images")
    )
    assert result.status is RunStatus.COMPLETED


def test_gpu_plan_event_is_published_once_after_durable_lease_before_images(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)

    class RecordingStore(LeaseStore):
        def write(self, lease) -> None:
            super().write(lease)
            if lease.gpu_plan is not None:
                backend.events.append(("lease_gpu_plan", lease.gpu_plan))

    result = RunCoordinator(
        backend=backend,
        lease_store=RecordingStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    gpu_events = [value for name, value in backend.events if name == "gpu_plan"]
    assert gpu_events == [backend.gpu_plan]
    names = [name for name, _ in backend.events]
    assert (
        names.index("lease_gpu_plan")
        < names.index("gpu_plan")
        < names.index("images")
    )
    assert result.status is RunStatus.COMPLETED


def test_gpu_plan_remains_in_completed_lease_after_later_writes(tmp_path) -> None:
    """Catches removing ``lease = updated`` or later writes using stale state."""
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.COMPLETED
    completed = store.read("run-1")
    assert completed is not None
    assert completed.gpu_plan is not None
    assert completed.gpu_plan.authorized_pool.uuids == (
        "GPU-a",
        "GPU-b",
        "GPU-c",
        "GPU-d",
    )
    assert completed.gpu_plan.work.uuids == ("GPU-a", "GPU-b")
    assert completed.gpu_plan.judge.uuids == ("GPU-c", "GPU-d")
    assert completed.gpu_plan.judge_mode is JudgeGPUMode.DISJOINT


def test_gpu_plan_write_failure_prevents_image_and_work_runtime_mutations(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)

    class FailGPUPlanWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failed = False

        def write(self, lease) -> None:
            if lease.gpu_plan is not None and not self.failed:
                self.failed = True
                raise OSError("GPU plan fsync failed")
            super().write(lease)

    store = FailGPUPlanWrite(tmp_path / "leases")
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    assert store.failed is True
    assert result.status is RunStatus.FAILED
    names = [name for name, _ in backend.events]
    assert "images" not in names
    assert "network_create" not in names
    assert "work_create" not in names


def test_split_workdir_volume_is_durable_and_attested_before_work_start(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)

    class RecordingStore(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.previous = None

        def write(self, lease) -> None:
            super().write(lease)
            volume = lease.work.workdir_volume
            previous = (
                None if self.previous is None else self.previous.work.workdir_volume
            )
            if volume.planned_name is not None and (
                previous is None or previous.planned_name is None
            ):
                backend.events.append(
                    ("lease_write_planned_volume", volume.planned_name)
                )
            if volume.actual is not None and (
                previous is None or previous.actual is None
            ):
                backend.events.append(("lease_write_actual_volume", volume.actual.name))
            if volume.mounted and (previous is None or not previous.mounted):
                backend.events.append(
                    ("lease_write_mounted_volume", volume.actual.name)
                )
            self.previous = lease

    store = RecordingStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    coordinator.run(RunRequest(task_dir=tmp_path))

    selected = {
        "volume_plan",
        "lease_write_planned_volume",
        "volume_create",
        "lease_write_actual_volume",
        "work_plan",
        "work_create",
        "volume_attest_work_mount",
        "lease_write_mounted_volume",
        "work_start",
    }
    observed = [name for name, _ in backend.events if name in selected]
    assert observed[:9] == [
        "volume_plan",
        "lease_write_planned_volume",
        "volume_create",
        "lease_write_actual_volume",
        "work_plan",
        "work_create",
        "volume_attest_work_mount",
        "lease_write_mounted_volume",
        "work_start",
    ]


def test_work_feedback_mount_is_attested_before_work_start(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    backend.store = store

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.COMPLETED
    names = [name for name, _ in backend.events]
    assert names.index("feedback_attest_work_mount") < names.index("work_start")


def test_work_feedback_mount_attestation_failure_prevents_work_start(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_work_feedback_attest = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.FAILED
    assert not any(name == "work_start" for name, _ in backend.events)
    assert ("work_remove", "work-1") in backend.events
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is False
    assert retained.work.container_id is None


def test_full_rootfs_mode_calls_no_workdir_volume_ports_and_passes_none_to_work(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.plan = backend.plan.model_copy(
        update={
            "workdir": PurePosixPath("/"),
            "rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS,
            "task": backend.plan.task.model_copy(
                update={
                    "workdir": PurePosixPath("/"),
                    "service": backend.plan.task.service.model_copy(
                        update={"workdir": PurePosixPath("/")}
                    ),
                }
            ),
            "images": backend.plan.images.model_copy(
                update={
                    "workdir": PurePosixPath("/"),
                    "rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS,
                }
            ),
        }
    )
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.COMPLETED
    assert not any(name.startswith("volume_") for name, _ in backend.events)
    retained = store.read("run-1")
    assert retained is not None
    assert retained.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS
    assert retained.work.workdir_volume.actual is None


@pytest.mark.parametrize(
    "workdir",
    (
        "/tests",
        "/tests/cases",
        "/logs",
        "/logs/verifier",
        "/logs/verifier/output",
        "/run",
        "/run/rsi-harness",
        "/run/rsi-harness/staging",
        "/run/rsi-harness/staging/cache",
        "/run/rsi-harness/feedback",
        "/run/rsi-harness/feedback/cache",
    ),
)
def test_split_workdir_mount_topology_fails_before_external_runtime_mutation(
    tmp_path, workdir: str
) -> None:
    """A split mount must not shadow or contain any Engine-owned mount target."""
    backend = ScriptedBackend(tmp_path)
    target = PurePosixPath(workdir)
    backend.plan = backend.plan.model_copy(
        update={
            "workdir": target,
            "task": backend.plan.task.model_copy(
                update={
                    "workdir": target,
                    "service": backend.plan.task.service.model_copy(
                        update={"workdir": target}
                    ),
                }
            ),
            "images": backend.plan.images.model_copy(update={"workdir": target}),
        }
    )
    backend.workdir_volume = backend.workdir_volume.model_copy(
        update={"target": target}
    )
    backend.planned_workdir_volume = backend.workdir_volume
    store = LeaseStore(tmp_path / "leases")
    backend.store = store

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.FAILED
    assert not any(
        name
        in {
            "artifacts",
            "server_start",
            "network_create",
            "volume_create",
            "work_create",
        }
        for name, _ in backend.events
    )
    retained = store.read("run-1")
    assert retained is not None
    assert "mount topology" in (retained.error or "")


def test_production_backend_orders_topology_before_external_runtime_owners(
    tmp_path,
) -> None:
    """The concrete production adapter must preserve coordinator preflight order."""
    ports = ScriptedBackend(tmp_path)
    target = PurePosixPath("/tests")
    ports.plan = ports.plan.model_copy(
        update={
            "workdir": target,
            "task": ports.plan.task.model_copy(
                update={
                    "workdir": target,
                    "service": ports.plan.task.service.model_copy(
                        update={"workdir": target}
                    ),
                }
            ),
            "images": ports.plan.images.model_copy(update={"workdir": target}),
        }
    )

    class Compiler:
        def compile(self, task_dir, options):
            del options
            return ports.compile(RunRequest(task_dir=task_dir))

    backend = ProductionCoordinatorBackend(
        compiler=Compiler(),
        allocator=ports.allocate,
        image_preparer=ports.prepare_images,
        plan_preparer=ports.prepare_plan,
        artifact_starter=ports.start_artifacts,
        server_starter=ports.start_server,
        network_planner=ports.plan_network,
        network_creator=ports.create_network,
        network_remover=ports.remove_network,
        workdir_volume_planner=ports.plan_workdir_volume,
        workdir_volume_creator=ports.create_workdir_volume,
        workdir_volume_attester=ports.attest_workdir_volume,
        workdir_volume_remover=ports.remove_workdir_volume,
        work_name_planner=ports.plan_work_container,
        work_creator=ports.create_work,
        work_feedback_attester=ports.attest_work_feedback_mount,
        work_policy_planner=ports.plan_work_policy,
        work_policy_installer=ports.install_work_policy,
        work_policy_remover=ports.remove_work_policy,
        work_starter=ports.start_work,
        work_remover=ports.remove_work,
        hook_installer=ports.install_hooks,
        agent_preparer=ports.prepare_agent,
        agent_runner=ports.run_agent,
        agent_stopper=ports.stop_agent,
        work_quiescer=ports.quiesce_work,
        retained_work_planner=ports.plan_retained_work,
        work_retainer=ports.retain_work,
        retained_work_releaser=ports.release_retained_work,
        evaluator=lambda request, observer: ports.evaluate_submission(
            request, lifecycle_observer=observer
        ),
        event_recorder=ports.record_event,
    )
    store = LeaseStore(tmp_path / "leases")

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.FAILED
    assert [name for name, _ in ports.events] == [
        "compiler",
        "allocation",
        "gpu_plan",
        "images",
        "preflight",
    ]


def test_retryable_gpu_release_returns_coordinator_to_agent_running_without_round(
    tmp_path,
) -> None:
    class RetryableBackend(ScriptedBackend):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.retry_observed = None

        def start_server(self, evaluator, artifacts, clock):
            self.events.append(("server_start", None))
            self.service = SubmissionService(
                evaluator=evaluator, artifact_writer=artifacts, clock=clock
            )
            return StartedSubmissionServer(
                service=self.service,
                owner=Server(self.events),
                endpoint=JudgeEndpoint(
                    url="http://172.30.0.1:9020", bind_host="127.0.0.1", port=9020
                ),
            )

        def run_agent(self, prepared, work, timeout):
            del prepared, timeout
            assert self.service is not None and self.submit_token is not None
            try:
                self.service.submit(self.submit_token)
            except RetryableSubmissionError:
                assert self.store is not None
                self.retry_observed = self.store.read("run-1")
            self.service.submit(self.submit_token)
            return AgentRunResult(exit_code=0)

        def evaluate_submission(self, request, *, lifecycle_observer):
            if self._next_report == 0:
                self._next_report += 1
                lifecycle_observer.resource_event(
                    "work_pause_planned",
                    work_container_id=request.work_container.container_id,
                )
                lifecycle_observer.resource_event(
                    "work_paused",
                    work_container_id=request.work_container.container_id,
                )
                lifecycle_observer.resource_event(
                    "work_unpaused",
                    work_container_id=request.work_container.container_id,
                )
                raise RetryableSubmissionError(
                    "release all Work GPU processes before retrying: PID 42"
                )
            return super().evaluate_submission(
                request, lifecycle_observer=lifecycle_observer
            )

    backend = RetryableBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert backend.retry_observed is not None
    assert backend.retry_observed.work.paused is False
    assert backend.retry_observed.judge.model_dump(exclude_none=True) == {}
    assert RunStatus.AGENT_RUNNING in coordinator.phase_history
    assert result.total_rounds == 1
    assert [report.round_id for report in result.reports] == ["agent-1"]


def test_proven_absent_workdir_volume_create_cancels_plan_before_mutations(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_workdir_volume_create_absent = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.FAILED
    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.workdir_volume.planned_name is None
    assert retained.work.workdir_volume.actual is None
    assert retained.work.workdir_volume.mounted is False
    assert ("volume_plan", RootfsSnapshotMode.SPLIT_WORKDIR) in backend.events
    assert ("volume_create", backend.workdir_volume.name) in backend.events
    assert not any(name == "work_create" for name, _ in backend.events)
    assert not any(name == "volume_remove" for name, _ in backend.events)


def test_ambiguous_workdir_volume_create_retains_planned_recovery_authority(
    tmp_path,
) -> None:
    """A possible create mutation must retain its durable plan for recovery."""
    backend = ScriptedBackend(tmp_path)
    backend.fail_workdir_volume_create_ambiguous = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status is RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.workdir_volume.planned_name == backend.workdir_volume.name
    assert retained.work.workdir_volume.planned_target == backend.plan.workdir
    assert "create outcome is ambiguous" in (retained.error or "")
    assert not any(name == "work_create" for name, _ in backend.events)
    assert not any(name == "volume_remove" for name, _ in backend.events)


def test_actual_workdir_volume_fsync_failure_overlays_exact_local_authority(
    tmp_path,
) -> None:
    class FailFirstActualVolumeWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failed = False

        def write(self, lease) -> None:
            if lease.work.workdir_volume.actual is not None and not self.failed:
                self.failed = True
                raise OSError("actual WORKDIR volume identity fsync failed")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    store = FailFirstActualVolumeWrite(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert store.failed is True
    assert result.status is RunStatus.FAILED
    assert retained is not None
    assert retained.recovery_required is True
    assert "actual WORKDIR volume identity fsync failed" in (retained.error or "")
    assert retained.work.workdir_volume.actual == backend.workdir_volume
    assert retained.work.workdir_volume.mounted is False
    assert not any(name == "work_create" for name, _ in backend.events)
    assert not any(name == "volume_remove" for name, _ in backend.events)


@pytest.mark.parametrize("failed_state", ("planned", "mounted"))
def test_workdir_volume_fsync_windows_overlay_each_local_authority(
    tmp_path,
    failed_state,
) -> None:
    class FailFirstSelectedVolumeWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failed = False

        def write(self, lease) -> None:
            volume = lease.work.workdir_volume
            selected = (
                volume.planned_name is not None and volume.actual is None
                if failed_state == "planned"
                else volume.mounted
            )
            if selected and not self.failed:
                self.failed = True
                raise OSError(f"{failed_state} WORKDIR volume fsync failed")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    store = FailFirstSelectedVolumeWrite(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert store.failed is True
    assert result.status is RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert f"{failed_state} WORKDIR volume fsync failed" in (retained.error or "")
    assert retained.work.workdir_volume.planned_name == backend.workdir_volume.name
    if failed_state == "planned":
        assert retained.work.workdir_volume.actual is None
        assert not any(name == "volume_create" for name, _ in backend.events)
    else:
        assert retained.work.workdir_volume.actual == backend.workdir_volume
        assert retained.work.workdir_volume.mounted is True
        assert not any(name == "work_start" for name, _ in backend.events)
    assert not any(name == "volume_remove" for name, _ in backend.events)


def test_work_create_failure_retains_workdir_volume_without_false_removal(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_work_create = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status is RunStatus.FAILED
    assert retained is not None
    assert retained.work.workdir_volume.actual == backend.workdir_volume
    assert retained.work.workdir_volume.mounted is False
    assert not any(name == "volume_remove" for name, _ in backend.events)


def test_split_workdir_mount_failure_prevents_registration_hooks_and_agent(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_workdir_volume_attest = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.FAILED
    assert isinstance(backend.service, Service)
    assert backend.service.registered is False
    assert not any(name == "hooks" for name, _ in backend.events)
    assert not any(name == "agent_start" for name, _ in backend.events)
    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.workdir_volume.actual == backend.workdir_volume
    assert retained.work.workdir_volume.mounted is False


def test_split_workdir_mode_without_planned_volume_is_a_typed_engine_error(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.return_none_for_split_workdir = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.FAILED
    assert backend.artifacts is not None
    assert any(
        "InfrastructureError" in error and "split-workdir" in error
        for error in backend.artifacts.engine_errors
    )
    assert not any(name == "volume_create" for name, _ in backend.events)


@pytest.mark.parametrize(
    "updates",
    (
        {"run_id": "other-run"},
        {"task_id": "other-task"},
        {"target": PurePosixPath("/other-workspace")},
    ),
)
def test_split_workdir_mismatched_actual_is_rolled_back_before_work(
    tmp_path,
    updates,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.workdir_volume = backend.workdir_volume.model_copy(update=updates)
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status is RunStatus.FAILED
    assert ("volume_remove", backend.workdir_volume.name) in backend.events
    assert not any(name == "work_create" for name, _ in backend.events)
    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.workdir_volume.planned_name is None
    assert retained.work.workdir_volume.planned_target is None
    assert retained.work.workdir_volume.actual is None
    assert retained.work.workdir_volume.rollback is None
    assert backend.artifacts is not None
    assert any(
        "InfrastructureError" in error and "exact split-workdir" in error
        for error in backend.artifacts.engine_errors
    )


@pytest.mark.parametrize(
    "rollback_error",
    (
        "mismatched WORKDIR volume removal failed",
        "mismatched WORKDIR volume post-remove absence is ambiguous",
    ),
)
def test_split_workdir_mismatch_rollback_failure_retains_exact_authority(
    tmp_path,
    rollback_error,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.workdir_volume = backend.workdir_volume.model_copy(
        update={
            "name": f"rsi-harness-workdir-{'2' * 64}",
            "run_id": "returned-run",
            "task_id": "returned-task",
            "target": PurePosixPath("/returned-workspace"),
        }
    )
    backend.fail_workdir_volume_remove = rollback_error
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status is RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.workdir_volume.planned_name == (
        backend.planned_workdir_volume_name
    )
    assert retained.work.workdir_volume.planned_target == backend.plan.workdir
    assert retained.work.workdir_volume.actual is None
    assert retained.work.workdir_volume.rollback == backend.workdir_volume
    assert retained.work.workdir_volume.mounted is False
    assert rollback_error in (retained.error or "")
    assert ("volume_remove", backend.workdir_volume.name) in backend.events
    assert not any(name == "work_create" for name, _ in backend.events)


@pytest.mark.parametrize("failed_write", ("rollback_identity", "recovery_marker"))
def test_split_workdir_mismatch_fsync_windows_retain_exact_rollback_authority(
    tmp_path,
    failed_write,
) -> None:
    class FailSelectedRollbackWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failed = False

        def write(self, lease) -> None:
            volume = lease.work.workdir_volume
            selected = (
                volume.rollback is not None and not lease.recovery_required
                if failed_write == "rollback_identity"
                else volume.rollback is not None and lease.recovery_required
            )
            if selected and not self.failed:
                self.failed = True
                raise OSError(f"{failed_write} fsync failed")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    backend.workdir_volume = backend.workdir_volume.model_copy(
        update={
            "run_id": "returned-run",
            "task_id": "returned-task",
            "target": PurePosixPath("/returned-workspace"),
        }
    )
    if failed_write == "recovery_marker":
        backend.fail_workdir_volume_remove = (
            "mismatched WORKDIR volume absence is ambiguous"
        )
    store = FailSelectedRollbackWrite(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert store.failed is True
    assert result.status is RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.workdir_volume.planned_name == (
        backend.planned_workdir_volume_name
    )
    assert retained.work.workdir_volume.planned_target == backend.plan.workdir
    assert retained.work.workdir_volume.actual is None
    assert retained.work.workdir_volume.rollback == backend.workdir_volume
    if failed_write == "rollback_identity":
        assert not any(name == "volume_remove" for name, _ in backend.events)
    else:
        assert ("volume_remove", backend.workdir_volume.name) in backend.events
    assert not any(name == "work_create" for name, _ in backend.events)


def test_two_explicit_rounds_reuse_one_work_and_one_agent(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert sum(name == "work_create" for name, _ in backend.events) == 1
    assert sum(name == "agent_start" for name, _ in backend.events) == 1
    assert sum(name == "judge_create" for name, _ in backend.events) == 2
    assert all(
        value == "work-1" for name, value in backend.events if name == "work_resume"
    )
    assert result.total_rounds == 2
    assert result.best_score == 1.0
    assert result.best_round == "agent-2"
    assert result.best_rewards == {"latency": 7.0, "reward": 1.0}
    assert coordinator.phase_history == (
        RunStatus.PREPARING,
        RunStatus.AGENT_RUNNING,
        RunStatus.SNAPSHOTTING,
        RunStatus.JUDGING,
        RunStatus.AGENT_RUNNING,
        RunStatus.SNAPSHOTTING,
        RunStatus.JUDGING,
        RunStatus.AGENT_RUNNING,
        RunStatus.COMPLETED,
    )
    persisted = store.read("run-1")
    assert persisted is not None
    assert persisted.phase_history == tuple(
        status.value for status in coordinator.phase_history
    )
    assert not any(
        name == "judge_create"
        for name, _ in backend.events[
            backend.events.index(("agent_start", "work-1")) + 7 :
        ]
    )


def test_coordinator_passes_submission_budget_to_agent_prompt(tmp_path) -> None:
    """Dropping the run option here leaves the initial Agent prompt uninformed."""
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    backend.store = store

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(
        RunRequest(
            task_dir=tmp_path,
            options=CompileOptions(max_submissions=2),
        )
    )

    assert result.status is RunStatus.COMPLETED
    assert backend.prepared_max_submissions == 2


def test_snapshot_cancelled_event_durably_clears_plan_before_work_resume(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.cancel_snapshot_before_acquire = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    durable = store.read("run-1")
    assert result.total_rounds == 2
    assert durable is not None
    assert durable.judge.planned_snapshot is None
    assert durable.judge.planned_snapshot_ref is None
    assert durable.judge.snapshot_image_id is None


def test_invalid_transition_is_engine_bug_and_never_submission_score(tmp_path) -> None:
    state = CoordinatorState(run_id="run-1")
    with pytest.raises(StateTransitionError, match="preparing.*judging"):
        state.transition(RunStatus.JUDGING)

    assert state.history == (RunStatus.PREPARING,)
    assert state.engine_errors and "StateTransitionError" in state.engine_errors[0]


@pytest.mark.parametrize(
    ("reports", "primary", "direction", "status", "best_score", "best_round"),
    (
        ((), None, "maximize", RunStatus.NO_VALID_SUBMISSION, None, None),
        (
            (
                SubmissionReport(
                    round_id="agent-1",
                    status=SubmissionStatus.COMPLETED,
                    rewards={"accuracy": 0.9, "latency": 4},
                    score=None,
                ),
            ),
            None,
            "maximize",
            RunStatus.COMPLETED,
            None,
            None,
        ),
        (
            (
                SubmissionReport(
                    round_id="agent-1",
                    status=SubmissionStatus.COMPLETED,
                    rewards={"loss": 2, "accuracy": 0.7},
                    score=2,
                ),
                SubmissionReport(
                    round_id="agent-2",
                    status=SubmissionStatus.COMPLETED,
                    rewards={"loss": 1, "accuracy": 0.8},
                    score=1,
                ),
            ),
            "loss",
            "minimize",
            RunStatus.COMPLETED,
            1,
            "agent-2",
        ),
    ),
)
def test_aggregation_counts_verifier_valid_rounds_and_ranks_only_primary(
    reports, primary, direction, status, best_score, best_round
) -> None:
    result = aggregate_result(
        run_id="run-1",
        reports=reports,
        primary_reward=primary,
        direction=direction,
    )

    assert result.status == status
    assert result.best_score == best_score
    assert result.best_round == best_round
    assert result.total_rounds == len(reports)


def test_timeout_and_cancellation_preserve_prior_valid_result(tmp_path) -> None:
    report = SubmissionReport(
        round_id="agent-1",
        status=SubmissionStatus.COMPLETED,
        rewards={"reward": 0.5},
        score=0.5,
    )
    timed_out = aggregate_result(
        run_id="run-1",
        reports=(report,),
        primary_reward=None,
        direction="maximize",
        terminal=RunStatus.FAILED,
    )
    cancelled = timed_out.model_copy(update={"status": RunStatus.CANCELLED})

    assert timed_out.best_score == cancelled.best_score == 0.5
    assert timed_out.reports == cancelled.reports == (report,)


@pytest.mark.parametrize(
    ("agent_result", "expected"),
    (
        (AgentRunResult(exit_code=None, timed_out=True), RunStatus.FAILED),
        (AgentRunResult(exit_code=None, cancelled=True), RunStatus.CANCELLED),
    ),
)
def test_agent_terminal_paths_stop_new_submissions_before_work_teardown(
    tmp_path, agent_result, expected
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.agent_result = agent_result
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=LeaseStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == expected
    names = [name for name, _ in backend.events]
    assert names.index("server_stop") < names.index("agent_stop")
    assert names.index("agent_stop") < names.index("work_remove")
    assert names.index("server_stop") < names.index("artifacts_finalize")


def test_terminal_work_is_quiesced_retained_and_removed_in_exact_order(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.COMPLETED
    names = [name for name, _ in backend.events]
    ordered = (
        "server_stop",
        "agent_stop",
        "work_pause",
        "retained_plan",
        "retained_acquire",
        "work_remove",
        "work_policy_remove",
        "network_remove",
    )
    assert [names.index(name) for name in ordered] == sorted(
        names.index(name) for name in ordered
    )
    retained = store.read("run-1")
    assert retained is not None
    assert retained.recovery_required is False
    assert retained.work.container_id is None
    assert retained.work.paused is False
    assert retained.work.planned_retained_image_ref == (
        f"rsi-harness-rootfs:retained-work-{'a' * 64}"
    )
    assert retained.work.retained_image_id == f"sha256:{'b' * 64}"
    assert retained.work.retained_image_ref == (
        f"rsi-harness-rootfs:retained-work-{'a' * 64}"
    )


def test_unproven_agent_stop_blocks_quiescence_retention_and_removal(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_stop_agent = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.container_id == "work-1"
    assert retained.work.policy_rule_id == "policy-work-1"
    assert retained.work.network_id == "network-work-1"
    assert "Agent stop authority is unproven" in (retained.error or "")
    assert ("agent_stop", "work-1") in backend.events
    assert not any(
        name
        in {
            "work_pause",
            "work_stopped",
            "retained_plan",
            "retained_acquire",
            "work_remove",
            "work_policy_remove",
            "network_remove",
        }
        for name, _ in backend.events
    )


def test_retained_work_commit_failure_keeps_paused_work_authority(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_retain_work = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.container_id == "work-1"
    assert retained.work.paused is True
    assert retained.work.planned_retained_image_ref == (
        f"rsi-harness-rootfs:retained-work-{'a' * 64}"
    )
    assert retained.work.retained_image_id is None
    assert retained.work.retained_image_ref is None
    assert "retained Work commit failed" in (retained.error or "")
    assert not any(
        name in {"work_remove", "work_policy_remove", "network_remove"}
        for name, _ in backend.events
    )


def test_retained_identity_fsync_and_rollback_failure_overlays_local_authority(
    tmp_path,
) -> None:
    class FailFirstRetainedIdentityWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failed = False

        def write(self, lease) -> None:
            if lease.work.retained_image_id is not None and not self.failed:
                self.failed = True
                raise OSError("retained image identity fsync failed")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    backend.fail_retained_release = True
    store = FailFirstRetainedIdentityWrite(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert store.failed is True
    assert retained is not None and retained.recovery_required is True
    assert retained.work.container_id == "work-1"
    assert retained.work.paused is True
    assert retained.work.planned_retained_image_ref == (
        f"rsi-harness-rootfs:retained-work-{'a' * 64}"
    )
    assert retained.work.retained_image_id == f"sha256:{'b' * 64}"
    assert retained.work.retained_image_ref == (
        f"rsi-harness-rootfs:retained-work-{'a' * 64}"
    )
    assert ("retained_rollback", f"sha256:{'b' * 64}") in backend.events
    assert "rollback is unproven" in (retained.error or "")
    assert not any(name == "work_remove" for name, _ in backend.events)


def test_mismatched_retained_lease_and_failed_release_persist_rollback_authority(
    tmp_path,
) -> None:
    mismatch_ref = f"rsi-harness-rootfs:retained-work-{'d' * 64}"
    backend = ScriptedBackend(tmp_path)
    backend.retained_image_ref = mismatch_ref
    backend.fail_retained_release = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.container_id == "work-1"
    assert retained.work.planned_retained_image_ref == (
        f"rsi-harness-rootfs:retained-work-{'a' * 64}"
    )
    assert retained.work.retained_image_id is None
    assert retained.work.retained_image_ref is None
    assert retained.work.retained_image_rollback is not None
    assert retained.work.retained_image_rollback.image_id == f"sha256:{'b' * 64}"
    assert retained.work.retained_image_rollback.image_ref == mismatch_ref
    assert ("retained_rollback", f"sha256:{'b' * 64}") in backend.events
    assert not any(
        name in {"work_remove", "work_policy_remove", "network_remove"}
        for name, _ in backend.events
    )


def test_permanent_planned_retention_write_failure_blocks_every_cleanup_callback(
    tmp_path,
) -> None:
    class FailFromRetainedPlanWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.blocked = False

        def write(self, lease) -> None:
            if self.blocked or lease.work.planned_retained_image_ref is not None:
                self.blocked = True
                raise OSError("permanent retained plan fsync failure")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    store = FailFromRetainedPlanWrite(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    with pytest.raises(OSError, match="permanent retained plan fsync failure"):
        coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.container_id == "work-1"
    assert retained.work.policy_rule_id == "policy-work-1"
    assert retained.work.network_id == "network-work-1"
    assert ("retained_plan", "work-1") in backend.events
    assert not any(name == "retained_acquire" for name, _ in backend.events)
    assert not any(
        name in {"work_remove", "work_policy_remove", "network_remove"}
        for name, _ in backend.events
    )


def test_permanent_actual_retention_write_failure_blocks_every_cleanup_callback(
    tmp_path,
) -> None:
    class FailFromActualRetainedIdentityWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.blocked = False

        def write(self, lease) -> None:
            if self.blocked or lease.work.retained_image_id is not None:
                self.blocked = True
                raise OSError("permanent retained identity fsync failure")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    store = FailFromActualRetainedIdentityWrite(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    with pytest.raises(OSError, match="permanent retained identity fsync failure"):
        coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.container_id == "work-1"
    assert retained.work.policy_rule_id == "policy-work-1"
    assert retained.work.network_id == "network-work-1"
    assert retained.work.planned_retained_image_ref == (
        f"rsi-harness-rootfs:retained-work-{'a' * 64}"
    )
    assert ("retained_acquire", "work-1") in backend.events
    assert ("retained_rollback", f"sha256:{'b' * 64}") in backend.events
    assert not any(
        name in {"work_remove", "work_policy_remove", "network_remove"}
        for name, _ in backend.events
    )


def test_stopped_work_is_retained_without_claiming_it_is_paused(tmp_path) -> None:
    class RecordingStore(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.work_states = []

        def write(self, lease) -> None:
            self.work_states.append(lease.work)
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    backend.work_quiescence = "stopped"
    store = RecordingStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.COMPLETED
    assert retained is not None and retained.recovery_required is False
    assert retained.work.paused is False
    assert any(
        work.stopped is True
        and work.paused is False
        and work.retained_image_id == f"sha256:{'b' * 64}"
        for work in store.work_states
    )
    names = [name for name, _ in backend.events]
    assert names.index("agent_stop") < names.index("work_stopped")
    assert names.index("work_stopped") < names.index("retained_plan")


def test_work_quiescence_plan_is_durable_before_backend_mutation(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.require_quiescence_plan = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.COMPLETED
    assert retained is not None and retained.recovery_required is False
    assert retained.work.planned_quiescence is None


def test_permanent_factual_quiescence_write_failure_retains_durable_plan(
    tmp_path,
) -> None:
    class FailAfterQuiescencePlan(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.plan_written = False

        def write(self, lease) -> None:
            if self.plan_written:
                raise OSError("permanent factual quiescence fsync failure")
            super().write(lease)
            if lease.work.planned_quiescence == "pause-if-running":
                self.plan_written = True

    backend = ScriptedBackend(tmp_path)
    backend.require_quiescence_plan = True
    store = FailAfterQuiescencePlan(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    with pytest.raises(OSError, match="permanent factual quiescence fsync failure"):
        coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.planned_quiescence == "pause-if-running"
    assert retained.work.paused is False
    assert retained.work.stopped is False
    assert retained.work.container_id == "work-1"
    assert retained.work.policy_rule_id == "policy-work-1"
    assert retained.work.network_id == "network-work-1"
    assert not any(
        name
        in {
            "retained_plan",
            "retained_acquire",
            "work_remove",
            "work_policy_remove",
            "network_remove",
        }
        for name, _ in backend.events
    )


def test_ambiguous_work_quiescence_fails_closed_before_retention(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.work_quiescence = "running"
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.container_id == "work-1"
    assert retained.work.paused is False
    assert retained.work.planned_retained_image_ref is None
    assert not any(name == "retained_acquire" for name, _ in backend.events)
    assert not any(name == "work_remove" for name, _ in backend.events)


def test_unproven_server_drain_retains_agent_work_and_network(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_server_stop = True
    store = LeaseStore(tmp_path / "leases")
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.FAILED
    assert result.total_rounds == 0
    assert sum(name == "server_stop" for name, _ in backend.events) == 1
    assert not any(name == "agent_stop" for name, _ in backend.events)
    assert not any(name == "work_remove" for name, _ in backend.events)
    assert not any(name == "network_remove" for name, _ in backend.events)
    persisted = store.read("run-1")
    assert persisted is not None and persisted.recovery_required
    assert persisted.work.container_id == "work-1"
    assert "SERVER-SECRET" not in store.path_for("run-1").read_text()


@pytest.mark.parametrize("failure", ["work", "artifact"])
def test_cleanup_or_artifact_failure_has_one_failed_terminal_history(
    tmp_path, failure
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_work_remove = failure == "work"
    backend.fail_artifact_finalize = failure == "artifact"
    store = LeaseStore(tmp_path / "leases")
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    persisted = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert coordinator.phase_history[-1] == RunStatus.FAILED
    assert persisted is not None
    assert persisted.status == RunStatus.FAILED
    assert persisted.phase_history[-1] == RunStatus.FAILED.value
    assert RunStatus.COMPLETED not in coordinator.phase_history


def test_setup_failure_never_starts_agent_and_redacts_durable_error(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_at = "compile"
    store = LeaseStore(tmp_path / "leases")
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.FAILED
    assert not any(name == "agent_start" for name, _ in backend.events)
    raw = store.path_for("run-1")
    assert not raw.exists() or "SUPER-SECRET" not in raw.read_text()


def test_recovery_required_round_stops_second_submission_and_retains_work(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.reports = [
        SubmissionReport(
            round_id="agent-1",
            status=SubmissionStatus.INFRASTRUCTURE_ERROR,
            error="recovery_required: live Judge containment unproven",
        ),
        backend.reports[1],
    ]
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.FAILED
    assert result.total_rounds == 1
    assert sum(name == "judge_create" for name, _ in backend.events) == 1
    assert not any(name == "work_remove" for name, _ in backend.events)
    assert not any(name == "network_remove" for name, _ in backend.events)
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.work.paused is True
    assert retained.judge.round_id == "agent-1"
    assert retained.judge.container_id == "judge:agent-1"
    assert retained.judge.network_id == "network:agent-1"
    assert retained.judge.network_name == "network:agent-1"
    assert retained.judge.planned_network == "network:agent-1"
    assert retained.judge.policy_rule_id == "policy:agent-1"
    assert retained.judge.planned_policy_rule_id == "policy:agent-1"
    assert retained.judge.snapshot_lease_id == "snapshot:agent-1"
    assert retained.judge.snapshot_merged_path is None
    assert retained.judge.planned_snapshot_ref == (
        f"rsi-harness-rootfs:judge-round-{'a' * 64}"
    )
    assert retained.judge.snapshot_image_id == f"sha256:{'b' * 64}"
    assert retained.judge.snapshot_image_ref == retained.judge.planned_snapshot_ref
    assert retained.judge.snapshot_source_container_id == "work-1"


def test_cleanup_recovery_survives_submission_artifact_failure_without_resume(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.reports[0] = SubmissionReport(
        round_id="agent-1",
        status=SubmissionStatus.INFRASTRUCTURE_ERROR,
        error="recovery_required: policy cleanup remains unproven",
    )
    evaluate_submission = backend.evaluate_submission

    def fail_after_recovery(request, *, lifecycle_observer):
        evaluate_submission(request, lifecycle_observer=lifecycle_observer)
        raise OSError("submission artifact fsync failed")

    backend.evaluate_submission = fail_after_recovery  # type: ignore[method-assign]
    store = LeaseStore(tmp_path / "leases")
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.paused is True
    assert retained.judge.round_id == "agent-1"
    assert retained.judge.container_id == "judge:agent-1"
    assert retained.judge.network_id == "network:agent-1"
    assert retained.judge.policy_rule_id == "policy:agent-1"
    assert retained.judge.snapshot_lease_id == "snapshot:agent-1"
    assert not any(name == "work_resume" for name, _ in backend.events)
    assert not any(name == "work_remove" for name, _ in backend.events)
    assert not any(name == "network_remove" for name, _ in backend.events)


def test_pre_pause_artifact_failure_closes_round_but_normally_contains_work(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)

    def fail_before_pause(request, *, lifecycle_observer):
        del request
        lifecycle_observer.resource_event(
            "submission_closed",
            reason="pre-pause report fsync failed",
        )
        raise OSError("pre-pause report fsync failed")

    backend.evaluate_submission = fail_before_pause  # type: ignore[method-assign]
    store = LeaseStore(tmp_path / "leases")
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is False
    assert retained.work.paused is False
    assert retained.work.container_id is None
    assert retained.work.policy_rule_id is None
    assert retained.work.network_id is None
    assert any(name == "agent_stop" for name, _ in backend.events)
    assert any(name == "work_remove" for name, _ in backend.events)
    assert any(name == "work_policy_remove" for name, _ in backend.events)
    assert any(name == "network_remove" for name, _ in backend.events)
    assert not any(name == "work_resume" for name, _ in backend.events)


def test_failed_recovery_marker_fsync_still_retains_all_runtime_authority(
    tmp_path,
) -> None:
    class FailFirstRecoveryWrite(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failed = False

        def write(self, lease) -> None:
            if lease.recovery_required and not self.failed:
                self.failed = True
                raise OSError("recovery marker fsync failed")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    backend.reports[0] = SubmissionReport(
        round_id="agent-1",
        status=SubmissionStatus.INFRASTRUCTURE_ERROR,
        error="recovery_required: snapshot cleanup remains unproven",
    )
    store = FailFirstRecoveryWrite(tmp_path / "leases")
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert store.failed is True
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.paused is True
    assert retained.work.container_id == "work-1"
    assert retained.work.policy_rule_id == "policy-work-1"
    assert retained.work.network_id == "network-work-1"
    assert retained.judge.container_id == "judge:agent-1"
    assert retained.judge.snapshot_lease_id == "snapshot:agent-1"
    assert not any(name == "agent_stop" for name, _ in backend.events)
    assert not any(name == "work_remove" for name, _ in backend.events)
    assert not any(name == "work_policy_remove" for name, _ in backend.events)
    assert not any(name == "network_remove" for name, _ in backend.events)


def test_partial_work_policy_install_failure_retains_exact_authority(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_work_policy_install_recovery = True
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.container_id == "work-1"
    assert retained.work.network_id == "network-work-1"
    assert retained.work.planned_policy_rule_id == "policy-work-1"
    assert retained.work.policy_rule_id == "policy-work-1"
    assert not any(name == "work_remove" for name, _ in backend.events)
    assert not any(name == "network_remove" for name, _ in backend.events)


def test_prepare_failure_still_stops_agent_and_clears_installed_hooks(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.fail_at = "prepare_agent"
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=LeaseStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.FAILED
    assert ("hooks", "work-1") in backend.events
    assert ("agent_stop", "work-1") in backend.events
    assert backend.events.index(("agent_stop", "work-1")) < backend.events.index(
        ("work_remove", "work-1")
    )


def test_keyboard_interrupt_preserves_completed_round_and_cancels_cleanup(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)

    def interrupt(prepared, work, timeout):
        del prepared, timeout
        backend.events.append(("agent_start", work.container_id))
        assert backend.service is not None
        backend.service.submit()
        raise KeyboardInterrupt

    backend.run_agent = interrupt  # type: ignore[method-assign]
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=LeaseStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.CANCELLED
    assert result.total_rounds == 1
    assert result.best_score == 0.25
    names = [name for name, _ in backend.events]
    assert names.index("server_stop") < names.index("agent_stop")
    assert names.index("agent_stop") < names.index("work_remove")


def test_sigterm_cancels_run_and_restores_previous_handler(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    previous = signal.getsignal(signal.SIGTERM)
    store = LeaseStore(tmp_path / "leases")

    def terminate(prepared, work, timeout):
        del prepared, timeout
        backend.events.append(("agent_start", work.container_id))
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)
        raise AssertionError("SIGTERM handler returned")

    backend.run_agent = terminate  # type: ignore[method-assign]
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.CANCELLED
    assert signal.getsignal(signal.SIGTERM) is previous
    persisted = store.read("run-1")
    assert persisted is not None
    assert persisted.status == RunStatus.CANCELLED
    assert persisted.phase == RunStatus.CANCELLED.value
    assert persisted.work.container_id is None
    assert persisted.work.network_id is None
    assert ("artifacts_finalize", RunStatus.CANCELLED) in backend.events
    names = [name for name, _ in backend.events]
    assert names.index("server_stop") < names.index("agent_stop")


def test_submission_error_round_recovers_and_later_valid_round_wins(tmp_path) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.reports[0] = SubmissionReport(
        round_id="agent-1",
        status=SubmissionStatus.SUBMISSION_ERROR,
        error="Work GPU process is active",
    )
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=LeaseStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.COMPLETED
    assert result.total_rounds == 2
    assert result.best_round == "agent-2"


def test_engine_infrastructure_round_fails_but_preserves_prior_valid_result(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    backend.reports[1] = SubmissionReport(
        round_id="agent-2",
        status=SubmissionStatus.INFRASTRUCTURE_ERROR,
        error="transient Judge runtime failure",
    )
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=LeaseStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.FAILED
    assert result.total_rounds == 2
    assert result.best_score == 0.25
    assert result.best_round == "agent-1"


@pytest.mark.parametrize("cpu_only", (False, True))
def test_concrete_production_composition_runs_two_rounds_with_real_endpoint(
    tmp_path,
    cpu_only,
) -> None:
    ports = ScriptedBackend(tmp_path)
    if cpu_only:
        ports.gpu_plan = RunGPUPlan(
            authorized_pool=GPUAllocation(),
            work=GPUAllocation(),
            judge=GPUAllocation(),
            judge_mode=JudgeGPUMode.FREEZE_ONLY,
        )
        ports.plan = ports.plan.model_copy(
            update={
                "task": ports.plan.task.model_copy(
                    update={"gpu_requirement": GPURequirement(count=0)}
                ),
                "gpu_plan": ports.gpu_plan,
            }
        )
    responses: list[str] = []
    store = LeaseStore(tmp_path / "leases")
    embedded = EmbeddedSubmissionServerFactory(
        bind_host="127.0.0.1",
        port=0,
        bridge_gateway="127.0.0.1",
    )

    def start_server(evaluator, artifacts, clock):
        started = embedded(evaluator, artifacts, clock)
        ports.service = started.service
        ports.events.append(("server_start", started.endpoint.url))
        return started

    def start_artifacts(plan, run_id):
        writer = RunArtifactWriter(plan, run_id=run_id, clock=Clock())
        writer.start()
        ports.artifacts = writer
        return writer

    def run_agent_over_http(prepared, work, timeout):
        del prepared, timeout
        ports.events.append(("agent_start", work.container_id))
        assert ports.submit_url is not None
        assert ports.submit_token is not None
        for index in range(2):
            request = urllib.request.Request(
                f"{ports.submit_url}/api/v1/submit",
                data=b"",
                headers={"Authorization": f"Bearer {ports.submit_token}"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
                responses.append(response.read().decode())
            if index == 0:
                ports.events.append(("agent_writes", work.container_id))
        return AgentRunResult(exit_code=0)

    class Compiler:
        def compile(self, task_dir, options):
            del options
            return ports.compile(RunRequest(task_dir=task_dir))

    backend = ProductionCoordinatorBackend(
        compiler=Compiler(),
        allocator=ports.allocate,
        image_preparer=ports.prepare_images,
        plan_preparer=ports.prepare_plan,
        artifact_starter=start_artifacts,
        server_starter=start_server,
        network_planner=ports.plan_network,
        network_creator=ports.create_network,
        network_remover=ports.remove_network,
        workdir_volume_planner=ports.plan_workdir_volume,
        workdir_volume_creator=ports.create_workdir_volume,
        workdir_volume_attester=ports.attest_workdir_volume,
        workdir_volume_remover=ports.remove_workdir_volume,
        work_name_planner=ports.plan_work_container,
        work_creator=ports.create_work,
        work_feedback_attester=ports.attest_work_feedback_mount,
        work_policy_planner=ports.plan_work_policy,
        work_policy_installer=ports.install_work_policy,
        work_policy_remover=ports.remove_work_policy,
        work_starter=ports.start_work,
        work_remover=ports.remove_work,
        hook_installer=ports.install_hooks,
        agent_preparer=ports.prepare_agent,
        agent_runner=run_agent_over_http,
        agent_stopper=ports.stop_agent,
        work_quiescer=ports.quiesce_work,
        retained_work_planner=ports.plan_retained_work,
        work_retainer=ports.retain_work,
        retained_work_releaser=ports.release_retained_work,
        evaluator=lambda request, observer: ports.evaluate_submission(
            request, lifecycle_observer=observer
        ),
        event_recorder=ports.record_event,
    )
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.total_rounds == 2
    assert result.best_round == "agent-2"
    assert sum(name == "work_create" for name, _ in ports.events) == 1
    assert sum(name == "agent_start" for name, _ in ports.events) == 1
    assert ["round: agent-1" in item for item in responses] == [True, False]
    assert ["round: agent-2" in item for item in responses] == [False, True]
    assert [report.round_id for report in result.reports] == ["agent-1", "agent-2"]
    persisted = store.read("run-1")
    assert persisted is not None and persisted.status == RunStatus.COMPLETED
    assert persisted.gpu_plan == ports.gpu_plan
    final_result = json.loads(
        (ports.plan.paths.logs / "runs/run-1/minimal-gpu/final_result.json").read_text()
    )
    assert final_result["total_rounds"] == 2
    assert final_result["best_round"] == "agent-2"
    endpoint = next(value for name, value in ports.events if name == "server_start")
    assert str(endpoint).startswith("http://127.0.0.1:")


@pytest.mark.parametrize(
    ("failure", "actual_name", "remove_fails", "identity_write_fails"),
    (
        ("adapter", "rsi-run-1-work", False, False),
        ("name-mismatch", "unexpected-work-network", False, False),
        ("name-mismatch", "unexpected-work-network", True, False),
        ("identity-write", "rsi-run-1-work", True, True),
    ),
)
def test_production_network_post_create_failure_uses_durable_cleanup_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    actual_name: str,
    remove_fails: bool,
    identity_write_fails: bool,
) -> None:
    from rsi_harness.runtime import production
    from rsi_harness.runtime.production import _ProductionRunComposition
    from rsi_loop.harness.config import RSILoopConfig

    network_id = "network-created-before-post-create-failure"

    class NetworkIdentityWriteStore(LeaseStore):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.failed = False

        def write(self, lease) -> None:
            if (
                identity_write_fails
                and not self.failed
                and lease.work.network_id == network_id
            ):
                self.failed = True
                raise OSError("forced actual network identity fsync failure")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    store = NetworkIdentityWriteStore(tmp_path / "leases")
    backend.store = store
    networks: dict[str, ManagedNetwork] = {}
    remove_attempts: list[str] = []

    class BoundaryRuntime:
        def __init__(self, _client, **kwargs) -> None:
            self.network = kwargs.get("network")

        @staticmethod
        def planned_network_name(_purpose: str) -> str:
            return "rsi-run-1-work"

        def create_network(self, _purpose: str, *, internal: bool):
            network = ManagedNetwork(
                network_id=network_id,
                name=actual_name,
                run_id="run-1",
                task_id=backend.plan.task.task_id,
                role="work",
                internal=internal,
            )
            networks[network.network_id] = network
            return network

        def remove_network(self, network: ManagedNetwork) -> None:
            remove_attempts.append(network.network_id)
            if remove_fails:
                raise RuntimeError("forced Work network removal failure")
            networks.pop(network.network_id, None)

        @staticmethod
        def planned_container_name(_purpose: str) -> str:
            return "rsi-run-1-minimal-gpu-work"

    def adapter_factory(_config, _runtime):
        if failure == "adapter":
            raise RuntimeError("forced Agent adapter constructor failure")
        return object()

    monkeypatch.setattr(production, "DockerContainerRuntime", BoundaryRuntime)
    for path in (tmp_path / "data", tmp_path / "logs"):
        path.mkdir()
    composition = _ProductionRunComposition(
        client=object(),
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=object(),
        rsi_loop_config=RSILoopConfig(),
        snapshot=object(),
        firewall=object(),
        bind_host="127.0.0.1",
        bridge_gateway="127.0.0.1",
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=adapter_factory,
        quiescence_checker=None,
        api_endpoints=(),
        agent_secret_env={},
        verifier_secret_env={},
    )
    composition.definition = backend.plan.task
    composition.run_id = "run-1"
    composition.enforcer = object()
    composition.artifacts = RunArtifactWriter(backend.plan, run_id="run-1")
    composition.artifacts.start()
    backend.plan_network = composition.plan_network  # type: ignore[method-assign]
    backend.create_network = composition.create_network  # type: ignore[method-assign]
    backend.remove_network = composition.remove_network  # type: ignore[method-assign]
    backend.plan_work_container = composition.plan_work_container  # type: ignore[method-assign]
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.status == RunStatus.FAILED
    assert store.failed is identity_write_fails
    assert remove_attempts == [network_id]
    if remove_fails:
        assert set(networks) == {network_id}
        assert retained.recovery_required is True
        assert retained.work.network_id == network_id
        assert retained.work.network_name == actual_name
        assert retained.work.planned_network == "rsi-run-1-work"
        assert retained.work.paused is False
        assert retained.error is not None
        assert "forced Work network removal failure" in retained.error
    else:
        assert networks == {}
        assert retained.recovery_required is False
        assert retained.work.network_id is None
        assert retained.work.network_name is None
        assert retained.work.planned_network is None
    if failure == "adapter":
        assert retained.error is not None
        assert "Agent adapter constructor failure" in retained.error
    elif failure == "name-mismatch":
        assert retained.error is not None
        assert "created Work network differs from durable plan" in retained.error
    else:
        assert retained.error is not None
        assert "actual network identity fsync failure" in retained.error


def test_policy_mismatch_cleanup_failure_retains_planned_and_actual_ids(
    tmp_path,
) -> None:
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    returned_policy_id = "policy-returned-by-buggy-backend"

    def return_mismatched_policy(plan, work, network):
        del plan, work, network
        return returned_policy_id

    def fail_mismatched_cleanup(rule_id: str) -> None:
        assert rule_id == returned_policy_id
        raise RuntimeError("forced mismatched policy cleanup failure")

    backend.install_work_policy = return_mismatched_policy  # type: ignore[method-assign]
    backend.remove_work_policy = fail_mismatched_cleanup  # type: ignore[method-assign]
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    retained = store.read("run-1")
    assert result.status == RunStatus.FAILED
    assert retained is not None and retained.recovery_required is True
    assert retained.work.planned_policy_rule_id == "policy-work-1"
    assert retained.work.policy_rule_id == returned_policy_id
    assert retained.error is not None
    assert "installed Work policy identity differs from plan" in retained.error
    assert "mismatched policy cleanup failure" in retained.error
