import json
from pathlib import Path, PurePosixPath

import pytest
from docker.errors import APIError
from pydantic import ValidationError

from rsi_harness.errors import (
    ContainerExecNotStartedError,
    InfrastructureError,
    RetryableSubmissionError,
    StateTransitionError,
    SubmissionError,
)
from rsi_harness.models import (
    AgentRunResult,
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerTmpfs,
    ContainerVolumeMount,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    ManagedWorkdirVolume,
    RootfsSnapshotMode,
    RunGPUPlan,
    SubmissionStatus,
    VerifierPlan,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.judge import (
    OUTPUT_TRUNCATION_MARKER,
    VERIFIER_EXPECTED_GPU_UUIDS_ENV,
    DockerJudgeRuntimeFactory,
    JudgeRunner,
)
from rsi_harness.runtime.network import NetworkPolicyEnforcer
from rsi_harness.runtime.submissions import SubmissionClosedError, SubmissionService
from rsi_harness.runtime.workdir_volume import managed_workdir_volume_labels
from rsi_harness.task.digest import hash_tree
from tests.factories import make_run_plan
from tests.fakes import (
    FakeArtifactWriter,
    FakeClock,
    FakeDockerClient,
    FakeDockerContainer,
    FakeFirewallBackend,
    FakeJudgeRuntime,
    FakeJudgeSnapshotBackend,
)


class RecordingJudgeResourceObserver:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def judge_network_planned(self, round_id, planned_name):
        self.events.append(("network_planned", planned_name))

    def judge_network_created(self, round_id, network):
        self.events.append(("network_created", network.network_id))

    def judge_container_planned(self, round_id, planned_name):
        self.events.append(("container_planned", planned_name))

    def judge_container_created(self, round_id, container):
        self.events.append(("container_created", container.container_id))

    def judge_policy_planned(self, round_id, rule_id):
        self.events.append(("policy_planned", rule_id))

    def judge_policy_installed(self, round_id, lease):
        self.events.append(("policy_installed", lease.rule_id))

    def judge_container_removed(self, round_id, container_id):
        self.events.append(("container_removed", container_id))

    def judge_policy_removed(self, round_id, rule_id):
        self.events.append(("policy_removed", rule_id))

    def judge_network_removed(self, round_id, network_id):
        self.events.append(("network_removed", network_id))


def managed_workdir_volume(plan, *, run_id: str = "run-1") -> ManagedWorkdirVolume:
    names = {
        "run-1": (
            "rsi-harness-workdir-"
            "833cb5148a5fca7b86d7eb62fd0c6d32d8014a36b38f9501b03c643f07bb58b0"
        ),
        "run-real": (
            "rsi-harness-workdir-"
            "54beb63c20e83ba95b64b021f2df9793e858f2976e3be744ebf8a20fc00605df"
        ),
    }
    return ManagedWorkdirVolume(
        name=names[run_id],
        run_id=run_id,
        task_id=plan.task.task_id,
        target=plan.workdir,
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce="f" * 64,
    )


def install_managed_workdir_volume(client, plan) -> ManagedWorkdirVolume:
    volume = managed_workdir_volume(plan)
    client.volumes.create(
        volume.name,
        driver="local",
        labels=managed_workdir_volume_labels(volume),
    )
    return volume


def install_work_volume_reference(
    client, plan, *, container_ref: ContainerRef | None = None
) -> ContainerRef:
    volume = managed_workdir_volume(plan)
    resolved = container_ref or ContainerRef(container_id="work-reference", role="work")
    reference = FakeDockerContainer(resolved.container_id)
    reference.attrs["Config"]["Labels"] = {
        "rsi-harness.run-id": volume.run_id,
        "rsi-harness.task-id": volume.task_id,
        "rsi-harness.role": "work",
    }
    reference.attrs["Mounts"] = [
        {
            "Type": "volume",
            "Name": volume.name,
            "Source": volume.name,
            "Destination": str(volume.target),
            "RW": True,
        }
    ]
    client.containers.by_id[reference.id] = reference
    return resolved


def make_harness(tmp_path: Path, *, event_callback=None):
    plan = make_run_plan(tmp_path)
    source = plan.task.source_dir
    (source / "tests").mkdir(parents=True)
    (source / "tests" / "test.sh").write_text("#!/bin/bash\n")
    plan = plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(
                    devices=(
                        GPUDevice(index=0, uuid="GPU-a", name="H100"),
                        GPUDevice(index=1, uuid="GPU-b", name="H100"),
                        GPUDevice(index=2, uuid="GPU-c", name="H100"),
                        GPUDevice(index=3, uuid="GPU-d", name="H100"),
                    )
                ),
                work=GPUAllocation(
                    devices=(
                        GPUDevice(index=0, uuid="GPU-a", name="H100"),
                        GPUDevice(index=1, uuid="GPU-b", name="H100"),
                    )
                ),
                judge=GPUAllocation(
                    devices=(
                        GPUDevice(index=2, uuid="GPU-c", name="H100"),
                        GPUDevice(index=3, uuid="GPU-d", name="H100"),
                    )
                ),
                judge_mode=JudgeGPUMode.DISJOINT,
            ),
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "user": "1001:1002",
                            "environment": (("FIXED", "yes"),),
                            "secret_env_names": ("VERIFIER_TOKEN",),
                            "output_limit_bytes": 80,
                        }
                    )
                }
            ),
        }
    )
    plan.paths.workspace.mkdir(parents=True)
    (
        plan.paths.logs
        / "runs"
        / "run-1"
        / plan.task.task_id
        / "feedback"
    ).mkdir(parents=True)
    log_dir = plan.paths.logs / "verifier"
    runtime = FakeJudgeRuntime(tmp_path / "snapshots")
    snapshot = FakeJudgeSnapshotBackend(tmp_path / "snapshots", runtime)
    artifacts = FakeArtifactWriter()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(plan),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        event_callback=event_callback,
    )
    request = plan, ContainerRef(container_id="work-1", role="work"), log_dir
    return runner, runtime, snapshot, artifacts, request


def evaluate(runner, request, round_id="agent-1"):
    plan, work, log_dir = request
    from rsi_harness.models import EvaluationRequest

    return runner.evaluate(
        EvaluationRequest(
            run_plan=plan,
            work_container=work,
            round_id=round_id,
            verifier_logs=log_dir / round_id,
            verifier_output=(
                plan.paths.logs
                / "runs"
                / "run-1"
                / plan.task.task_id
                / "feedback"
                / f"{round_id}.log"
            ),
        )
    )


def test_judge_emits_exec_progress_and_redacted_output(tmp_path):
    events: list[tuple[str, object]] = []
    runner, runtime, _snapshot, _artifacts, request = make_harness(
        tmp_path,
        event_callback=lambda name, value: events.append((name, value)),
    )
    runtime.exec_result = AgentRunResult(
        exit_code=0,
        output="tests running token=opaque-value\n",
    )

    evaluate(runner, request)

    assert events[0] == ("judge_exec_started", {"round_id": "agent-1"})
    assert events[1][0] == "judge_output"
    assert events[1][1] == "tests running token=[REDACTED]\n"


def test_successful_judge_lifecycle_and_fixed_security_spec(tmp_path):
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    tests_dir = request[0].task.source_dir / "tests"
    nested = tests_dir / "nested"
    nested.mkdir()
    helper = nested / "helper.txt"
    helper.write_text("helper\n")
    helper.chmod(0o640)
    (nested / "helper-link").symlink_to("helper.txt")
    (tests_dir / "test.sh").chmod(0o751)
    source_before = (
        hash_tree(tests_dir),
        tests_dir.stat().st_ino,
        tests_dir.stat().st_mtime_ns,
    )
    runtime.verifier_test_write = (Path("verifier-write.txt"), "private\n")

    report = evaluate(runner, request)

    assert runtime.events == [
        ("pause", "work-1"),
        ("snapshot_acquire", "agent-1"),
        ("judge_create", "agent-1"),
        ("judge_start", "agent-1"),
        ("tests_inject", "agent-1"),
        ("judge_exec", ("/bin/bash", "/tests/test.sh")),
        ("judge_remove", "agent-1"),
        ("judge_close", "agent-1"),
        ("snapshot_release", "agent-1"),
        ("unpause", "work-1"),
    ]
    assert report.status == SubmissionStatus.COMPLETED
    assert report.score == 1
    assert report.rewards == {"reward": 1}
    assert artifacts.reports == [report]

    spec = runtime.created_specs[0]
    plan, _, log_root = request
    assert spec.image == snapshot.acquired[0].image_id
    assert spec.workdir == PurePosixPath("/workspace")
    assert spec.user == "1001:1002"
    assert spec.environment == ()
    assert runtime.exec_environments == [
        {
            "FIXED": "yes",
            "NVIDIA_VISIBLE_DEVICES": "GPU-c,GPU-d",
            VERIFIER_EXPECTED_GPU_UUIDS_ENV: "GPU-c,GPU-d",
        }
    ]
    assert spec.gpu_allocation.uuids == ("GPU-c", "GPU-d")
    assert [(mount.target, mount.read_only) for mount in spec.mounts] == [
        (PurePosixPath("/logs/verifier"), False),
    ]
    assert spec.tmpfs == (
        ContainerTmpfs(
            target=PurePosixPath("/tests"),
            options="rw,exec,nosuid,nodev,mode=0755",
        ),
    )
    assert spec.volume_mounts == (
        ContainerVolumeMount(
            volume=managed_workdir_volume(plan),
            target=PurePosixPath("/workspace"),
            read_only=True,
        ),
    )
    sources = {mount.source for mount in spec.mounts}
    assert plan.task.source_dir not in sources
    assert plan.paths.root not in sources
    assert plan.paths.workspace not in sources
    assert (log_root / "agent-1").resolve() in sources
    assert "VERIFIER_TOKEN" not in dict(spec.environment)
    assert runtime.verifier_modified_tests is True
    assert not (tests_dir / "verifier-write.txt").exists()
    assert not (runtime.root / "private-tests" / "agent-1").exists()
    assert source_before == (
        hash_tree(tests_dir),
        tests_dir.stat().st_ino,
        tests_dir.stat().st_mtime_ns,
    )


def test_phase_gpu_judge_uses_exact_judge_allocation_and_expected_uuid_env(
    tmp_path,
) -> None:
    """Using the Work allocation here would expose the wrong physical devices."""
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert runtime.created_specs[0].gpu_allocation.uuids == ("GPU-c", "GPU-d")
    assert runtime.exec_environments[0][VERIFIER_EXPECTED_GPU_UUIDS_ENV] == (
        "GPU-c,GPU-d"
    )


@pytest.mark.parametrize("cpu_work", (False, True))
def test_judge_gpu_freeze_only_omits_devices_and_expected_uuid_env(
    tmp_path, cpu_work
) -> None:
    """A zero-GPU Judge must not inherit Work devices through snapshot mode."""
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "gpu_requirement": (
                        plan.task.gpu_requirement.model_copy(update={"count": 0})
                        if cpu_work
                        else plan.task.gpu_requirement
                    ),
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "environment": plan.task.verifier.environment
                            + (
                                (
                                    "NVIDIA_VISIBLE_DEVICES",
                                    "${MISSING_TASK_VALUE}",
                                ),
                            )
                        }
                    )
                }
            ),
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "judge": GPUAllocation(),
                    "judge_mode": JudgeGPUMode.FREEZE_ONLY,
                    "work": GPUAllocation() if cpu_work else plan.gpu_plan.work,
                    "authorized_pool": (
                        GPUAllocation() if cpu_work else plan.gpu_plan.authorized_pool
                    ),
                }
            )
        }
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.COMPLETED
    assert runtime.created_specs[0].gpu_allocation.devices == ()
    assert runtime.exec_environments[0][VERIFIER_EXPECTED_GPU_UUIDS_ENV] == ""
    assert runtime.exec_environments[0]["NVIDIA_VISIBLE_DEVICES"] == "void"


@pytest.mark.parametrize(
    ("mode", "volume_update", "message"),
    [
        (RootfsSnapshotMode.SPLIT_WORKDIR, None, "authority is unavailable"),
        (RootfsSnapshotMode.SPLIT_WORKDIR, {"run_id": "other-run"}, "inconsistent"),
        (RootfsSnapshotMode.SPLIT_WORKDIR, {"task_id": "other-task"}, "inconsistent"),
        (RootfsSnapshotMode.SPLIT_WORKDIR, {"target": "/other"}, "inconsistent"),
        (RootfsSnapshotMode.SPLIT_WORKDIR, {"driver": "nfs"}, "inconsistent"),
        (RootfsSnapshotMode.FULL_ROOTFS, {}, "received volume authority"),
    ],
)
def test_judge_rejects_mode_inconsistent_workdir_volume_before_pause(
    tmp_path, mode, volume_update, message
):
    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    plan = plan.model_copy(
        update={
            "rootfs_snapshot_mode": mode,
            "images": plan.images.model_copy(update={"rootfs_snapshot_mode": mode}),
        }
    )
    volume = None
    if volume_update is not None:
        volume = managed_workdir_volume(plan).model_copy(update=volume_update)
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=volume,
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
    )

    report = evaluate(runner, (plan, work, logs))

    assert report.status is SubmissionStatus.INFRASTRUCTURE_ERROR
    assert message in (report.error or "")
    assert not any(name == "pause" for name, _ in runtime.events)
    assert runtime.created_specs == []


def test_full_rootfs_judge_emits_no_workdir_volume_mount(tmp_path):
    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    plan = plan.model_copy(
        update={
            "rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS,
            "images": plan.images.model_copy(
                update={"rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS}
            ),
        }
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=None,
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
    )

    report = evaluate(runner, (plan, work, logs))

    assert report.status is SubmissionStatus.COMPLETED
    assert runtime.created_specs[0].volume_mounts == ()


def test_test_injection_failure_is_infrastructure_and_cleans_in_order(tmp_path):
    runner, runtime, _snapshot, artifacts, request = make_harness(tmp_path)
    runtime.fail_at = "inject"

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "inject failed" in (report.error or "")
    assert runtime.events[-4:] == [
        ("judge_remove", "agent-1"),
        ("judge_close", "agent-1"),
        ("snapshot_release", "agent-1"),
        ("unpause", "work-1"),
    ]
    assert artifacts.reports == [report]


def test_judge_rootfs_writes_never_enter_work_or_a_later_snapshot(tmp_path):
    runner, runtime, snapshot, _artifacts, request = make_harness(tmp_path)
    runtime.judge_rootfs_writes = {"/etc/judge-only": "private"}

    first = evaluate(runner, request, "agent-1")
    second = evaluate(runner, request, "agent-2")

    assert first.status == second.status == SubmissionStatus.COMPLETED
    assert [state["/etc/agent-state"] for state in snapshot.states] == [
        "agent-state",
        "agent-state",
    ]
    assert all("/etc/judge-only" not in state for state in snapshot.states)
    assert "/etc/judge-only" not in runtime.work_rootfs


def test_required_report_write_failure_retains_isolation_and_closes_service(
    tmp_path,
) -> None:
    class FailingWriter:
        def __init__(self) -> None:
            self.calls = 0

        def record_submission(self, report) -> None:
            del report
            self.calls += 1
            raise OSError("required report fsync failed")

    class Observer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.events.append((name, values))

    _runner, runtime, snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, _logs = request
    writer = FailingWriter()
    observer = Observer()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=writer,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )
    service = SubmissionService(
        evaluator=runner,
        artifact_writer=writer,
        clock=FakeClock(),
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
        max_submissions=2,
    )

    with pytest.raises(InfrastructureError, match="required report"):
        service.submit(token)

    assert writer.calls == 1
    assert runner.recovery_required is True
    assert runtime.work_paused is True
    assert ("judge_remove", "agent-1") not in runtime.events
    assert ("judge_contain", "agent-1") in runtime.events
    assert ("snapshot_release", "agent-1") not in runtime.events
    assert ("unpause", "work-1") not in runtime.events
    created = next(
        values for name, values in observer.events if name == "judge_created"
    )
    assert created["judge_container_id"] == "judge-agent-1"
    assert "judge_container_id=judge-agent-1" in str(created["recovery_context"])
    assert snapshot.acquired[0].image_id in runtime.snapshot_images
    assert any(name == "recovery_required" for name, _ in observer.events)
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)
    assert runtime.round_ids == ["agent-1"]


def test_partial_verifier_output_is_not_published_or_allowed_to_release_isolation(
    tmp_path,
) -> None:
    _runner, runtime, snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, _logs = request
    writer = RunArtifactWriter(plan, run_id="run-1")
    writer.start()
    runtime.complete_exec_output = b"partial verifier output"
    runtime.fail_at = "exec"
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(plan),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=writer,
        quiescence_checker=runtime.assert_gpu_quiet,
    )
    service = SubmissionService(
        evaluator=runner, artifact_writer=writer, clock=FakeClock()
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
        max_submissions=2,
    )

    with pytest.raises(InfrastructureError, match="complete verifier output"):
        service.submit(token)

    feedback = writer.feedback_root / "agent-1.log"
    assert not feedback.exists()
    assert runner.recovery_required is True
    assert runtime.work_paused is True
    assert ("judge_remove", "agent-1") not in runtime.events
    assert ("snapshot_release", "agent-1") not in runtime.events
    assert ("unpause", "work-1") not in runtime.events
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)


def test_pre_exec_failure_publishes_empty_feedback_and_releases_isolation(
    tmp_path,
) -> None:
    _runner, runtime, snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, _logs = request
    writer = RunArtifactWriter(plan, run_id="run-1")
    writer.start()

    def fail_before_exec(*_args, **_kwargs):
        raise ContainerExecNotStartedError("Docker exec was not started")

    runtime.exec = fail_before_exec
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(plan),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=writer,
        quiescence_checker=runtime.assert_gpu_quiet,
    )
    service = SubmissionService(
        evaluator=runner, artifact_writer=writer, clock=FakeClock()
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
        max_submissions=2,
    )

    feedback = service.submit(token)

    assert "status: infrastructure_error" in feedback
    assert "/run/rsi-harness/feedback/agent-1.log" in feedback
    assert (writer.feedback_root / "agent-1.log").read_bytes() == b""
    assert runner.recovery_required is False
    assert runner.submission_closed is False
    assert runtime.work_paused is False
    assert ("judge_remove", "agent-1") in runtime.events
    assert ("snapshot_release", "agent-1") in runtime.events
    assert ("unpause", "work-1") in runtime.events


def test_retryable_release_rejection_does_not_invoke_report_writer_or_close_submissions(
    tmp_path,
) -> None:
    class FailingWriter:
        def record_submission(self, report) -> None:
            del report
            raise OSError("pre-pause report fsync failed")

    class Observer:
        def __init__(self) -> None:
            self.events: list[str] = []

        def resource_event(self, name: str, **values: object) -> None:
            del values
            self.events.append(name)

    _runner, runtime, snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, _logs = request
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "judge_mode": JudgeGPUMode.RELEASE_ALL,
                    "judge": plan.gpu_plan.work,
                }
            )
        }
    )
    observer = Observer()

    def reject_before_pause(_allocation, _work) -> None:
        raise SubmissionError("GPU is not quiescent")

    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=FailingWriter(),
        quiescence_checker=reject_before_pause,
        lifecycle_observer=observer,
    )
    service = SubmissionService(
        evaluator=runner,
        artifact_writer=FakeArtifactWriter(),
        clock=FakeClock(),
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
        max_submissions=2,
    )

    with pytest.raises(
        RetryableSubmissionError, match="release all Work GPU processes"
    ):
        service.submit(token)

    assert runtime.work_paused is False
    assert runtime.events == [("pause", "work-1"), ("unpause", "work-1")]
    assert runner.recovery_required is False
    assert runner.submission_closed is False
    assert observer.events == ["work_pause_planned", "work_paused", "work_unpaused"]


@pytest.mark.parametrize("mode", (JudgeGPUMode.FREEZE_ONLY, JudgeGPUMode.DISJOINT))
def test_gpu_modes_without_release_all_pause_before_snapshot_without_quiescence(
    tmp_path, mode
) -> None:
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    judge = (
        GPUAllocation(devices=())
        if mode is JudgeGPUMode.FREEZE_ONLY
        else plan.gpu_plan.judge
    )
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={"judge_mode": mode, "judge": judge}
            )
        }
    )

    def checker(*_args: object) -> None:
        raise AssertionError("quiescence must only run for release-all")

    observer_events: list[str] = []

    class Observer:
        def resource_event(self, name: str, **_values: object) -> None:
            observer_events.append(name)

    runner._quiescence_checker = checker
    runner._lifecycle_observer = Observer()
    report = evaluate(runner, (plan, work, logs))

    assert report.status is SubmissionStatus.COMPLETED
    assert observer_events[:4] == [
        "work_pause_planned",
        "work_paused",
        "snapshot_planned",
        "snapshot_acquired",
    ]


def test_retryable_gpu_release_checks_work_gpus_after_pause(
    tmp_path,
) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "judge_mode": JudgeGPUMode.RELEASE_ALL,
                    "judge": plan.gpu_plan.work,
                }
            )
        }
    )
    observer_events: list[str] = []

    class Observer:
        def resource_event(self, name: str, **_values: object) -> None:
            observer_events.append(name)

    def reject(allocation: GPUAllocation, container: ContainerRef) -> None:
        assert allocation == plan.gpu_plan.work
        assert container == work
        raise SubmissionError("Work still owns GPU process PID 42")

    def snapshot_plan_must_not_run(**_values: object) -> str:
        raise AssertionError("retryable release-all rejection must not plan a snapshot")

    def reward_read_must_not_run(*_args: object) -> object:
        raise AssertionError("retryable release-all rejection must not read rewards")

    runner._quiescence_checker = reject
    runner._lifecycle_observer = Observer()
    snapshot.planned_ref = snapshot_plan_must_not_run
    runner.reward_reader = reward_read_must_not_run

    with pytest.raises(
        RetryableSubmissionError,
        match="release all Work GPU processes.*PID 42",
    ):
        evaluate(runner, (plan, work, logs))

    assert runtime.events[:2] == [("pause", "work-1"), ("unpause", "work-1")]
    assert runtime.work_paused is False
    assert observer_events == ["work_pause_planned", "work_paused", "work_unpaused"]
    assert snapshot.acquired == []
    assert runtime.round_ids == []
    assert artifacts.reports == []
    assert not (logs / "agent-1").exists()
    assert runner.recovery_required is False


def test_retryable_gpu_release_with_unproven_unpause_fails_closed(
    tmp_path,
) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "judge_mode": JudgeGPUMode.RELEASE_ALL,
                    "judge": plan.gpu_plan.work,
                }
            )
        }
    )

    def reject(_allocation: GPUAllocation, _container: ContainerRef) -> None:
        raise SubmissionError("Work still owns GPU process PID 42")

    runtime.fail_at = {"unpause", "inspect_quiescence"}
    runner._quiescence_checker = reject

    with pytest.raises(InfrastructureError, match="recovery_required"):
        evaluate(runner, (plan, work, logs))

    assert runtime.work_paused is True
    assert snapshot.acquired == []
    assert len(artifacts.reports) == 1
    assert runner.recovery_required is True


def test_retryable_gpu_release_recovery_observer_failure_stays_infrastructure(
    tmp_path,
) -> None:
    class Observer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.events.append((name, values))
            if name == "recovery_required":
                raise OSError("recovery lease fsync failed")

    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "judge_mode": JudgeGPUMode.RELEASE_ALL,
                    "judge": plan.gpu_plan.work,
                }
            )
        }
    )
    observer = Observer()

    def reject(_allocation: GPUAllocation, _container: ContainerRef) -> None:
        raise SubmissionError("Work still owns GPU process PID 42")

    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(plan),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=reject,
        lifecycle_observer=observer,
    )
    runtime.fail_at = {"unpause", "inspect_quiescence"}
    service = SubmissionService(
        evaluator=runner, artifact_writer=artifacts, clock=FakeClock()
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
        max_submissions=1,
    )

    with pytest.raises(InfrastructureError, match="recovery_required") as raised:
        service.submit(token)

    assert "PID 42" in str(raised.value)
    assert "recovery lease fsync failed" in str(raised.value)
    assert runtime.work_paused is True
    assert runner.recovery_required is True
    assert len(artifacts.reports) == 1
    assert service.session_state.rounds_allocated == 1
    assert service.reports == ()
    assert any(name == "work_quiescence_unproven" for name, _ in observer.events)
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)


def test_verifier_templates_resolve_from_runtime_only_source_before_create(
    tmp_path,
) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "environment": (
                                ("LITERAL", "visible"),
                                ("TOKEN", "${VERIFIER_TOKEN}"),
                            ),
                            "secret_env_names": ("VERIFIER_TOKEN",),
                        }
                    )
                }
            )
        }
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        verifier_secret_env={"VERIFIER_TOKEN": "runtime-verifier-secret"},
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.COMPLETED
    assert runtime.created_specs[0].environment == ()
    assert runtime.exec_environments[0] == {
        "LITERAL": "visible",
        "NVIDIA_VISIBLE_DEVICES": "GPU-c,GPU-d",
        "RSI_HARNESS_EXPECTED_GPU_UUIDS": "GPU-c,GPU-d",
        "TOKEN": "runtime-verifier-secret",
    }
    assert "runtime-verifier-secret" not in report.model_dump_json()


def test_resolved_verifier_secret_is_exact_redacted_from_runtime_errors(
    tmp_path,
) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    secret = "runtime-verifier-secret"
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "environment": (("TOKEN", "${VERIFIER_TOKEN}"),),
                            "secret_env_names": ("VERIFIER_TOKEN",),
                        }
                    )
                }
            )
        }
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        verifier_secret_env={"VERIFIER_TOKEN": secret},
    )

    def exec_echoing_environment(_container, _command, **kwargs):
        raise RuntimeError(
            f"Docker rejected exec environment {kwargs.get('environment')}"
        )

    runtime.exec = exec_echoing_environment  # type: ignore[method-assign]

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert secret not in (report.error or "")
    assert "[REDACTED]" in (report.error or "")
    assert secret not in artifacts.reports[0].model_dump_json()


def test_missing_verifier_template_is_typed_and_never_creates_judge(tmp_path) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "environment": (("TOKEN", "${VERIFIER_TOKEN}"),),
                            "secret_env_names": ("VERIFIER_TOKEN",),
                        }
                    )
                }
            )
        }
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        verifier_secret_env={},
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "missing verifier runtime environment variable 'VERIFIER_TOKEN'" in (
        report.error or ""
    )
    assert runtime.created_specs == []


def test_engine_expected_gpu_env_overrides_unresolvable_task_template(
    tmp_path,
) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "environment": (
                                (
                                    VERIFIER_EXPECTED_GPU_UUIDS_ENV,
                                    "${TASK_CANNOT_OVERRIDE_THIS}",
                                ),
                            ),
                            "secret_env_names": ("TASK_CANNOT_OVERRIDE_THIS",),
                        }
                    )
                }
            )
        }
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        verifier_secret_env={},
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.COMPLETED
    assert runtime.created_specs[0].environment == ()
    assert runtime.exec_environments[0][VERIFIER_EXPECTED_GPU_UUIDS_ENV] == (
        "GPU-c,GPU-d"
    )


def test_harbor_template_name_with_hyphen_resolves_at_runtime(tmp_path) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={
                            "environment": (("TOKEN", "${HOST-TOKEN}"),),
                            "secret_env_names": ("HOST-TOKEN",),
                        }
                    )
                }
            )
        }
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        verifier_secret_env={"HOST-TOKEN": "runtime-only-value"},
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.COMPLETED
    assert runtime.created_specs[0].environment == ()
    assert runtime.exec_environments[0]["TOKEN"] == ("runtime-only-value")


def test_judge_inherits_service_resources_and_user_when_phase_has_no_override(
    tmp_path,
) -> None:
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "service": plan.task.service.model_copy(
                        update={
                            "shm_size": "4g",
                            "user": "2001:2002",
                            "cpus": 2,
                            "memory_mb": 8192,
                            "storage_mb": 15360,
                        }
                    ),
                    "verifier": plan.task.verifier.model_copy(update={"user": None}),
                }
            )
        }
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.COMPLETED
    assert runtime.created_specs[0].shm_size == "4g"
    assert runtime.created_specs[0].user == "2001:2002"
    assert runtime.created_specs[0].cpus == 2
    assert runtime.created_specs[0].memory_mb == 8192
    assert runtime.created_specs[0].storage_mb == 15360


def test_lifecycle_observer_sees_plans_before_mutations_and_actual_ids_after(
    tmp_path,
) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)

    class Observer:
        def __init__(self) -> None:
            self.events = []

        def resource_event(self, name, **values) -> None:
            self.events.append((name, values, tuple(runtime.events)))

    observer = Observer()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )

    evaluate(runner, request)

    by_name = {name: (values, prior) for name, values, prior in observer.events}
    assert ("pause", "work-1") not in by_name["work_pause_planned"][1]
    assert by_name["work_paused"][0]["work_container_id"] == "work-1"
    assert ("snapshot_acquire", "agent-1") not in by_name["snapshot_planned"][1]
    planned_ref = snapshot.planned_ref(
        run_id="run-1",
        task_id=request[0].task.task_id,
        round_id="agent-1",
        purpose="judge-round",
    )
    assert by_name["snapshot_planned"][0] == {
        "round_id": "agent-1",
        "planned_snapshot_ref": planned_ref,
        "source_work_container_id": "work-1",
    }
    acquired = snapshot.acquired[0]
    assert by_name["snapshot_acquired"][0] == {
        "snapshot_lease_id": acquired.lease_id,
        "snapshot_image_id": acquired.image_id,
        "snapshot_image_ref": acquired.image_ref,
        "snapshot_source_container_id": "work-1",
    }
    assert ("judge_create", "agent-1") not in by_name["judge_planned"][1]
    assert by_name["judge_created"][0]["judge_container_id"] == "judge-agent-1"
    assert by_name["snapshot_released"][0]["snapshot_lease_id"] == acquired.lease_id
    assert by_name["work_unpaused"][0]["work_container_id"] == "work-1"


def test_snapshot_actual_authority_persistence_failure_retains_exact_snapshot_and_pause(
    tmp_path,
) -> None:
    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)

    class Observer:
        def __init__(self) -> None:
            self.attempted_actual: dict[str, object] | None = None

        def resource_event(self, name: str, **values: object) -> None:
            if name == "snapshot_acquired":
                self.attempted_actual = values
                raise OSError("actual snapshot lease fsync failed")

    observer = Observer()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )

    report = evaluate(runner, request)

    acquired = snapshot.acquired[0]
    assert observer.attempted_actual == {
        "snapshot_lease_id": acquired.lease_id,
        "snapshot_image_id": acquired.image_id,
        "snapshot_image_ref": acquired.image_ref,
        "snapshot_source_container_id": "work-1",
    }
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in (report.error or "")
    assert acquired.image_id in (report.error or "")
    assert acquired.image_ref in (report.error or "")
    assert ("snapshot_release", "agent-1") not in runtime.events
    assert ("unpause", "work-1") not in runtime.events
    assert runtime.work_paused is True


def test_proven_snapshot_acquire_failure_cancels_plan_before_work_resume(
    tmp_path,
) -> None:
    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    snapshot.fail_at = "acquire"

    class Observer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object], tuple[object, ...]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.events.append((name, values, tuple(runtime.events)))

    observer = Observer()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )

    report = evaluate(runner, request)

    cancelled = [event for event in observer.events if event[0] == "snapshot_cancelled"]
    assert len(cancelled) == 1
    _, values, prior_runtime_events = cancelled[0]
    assert values == {
        "round_id": "agent-1",
        "planned_snapshot_ref": snapshot.planned_ref(
            run_id="run-1",
            task_id=request[0].task.task_id,
            round_id="agent-1",
            purpose="judge-round",
        ),
        "source_work_container_id": "work-1",
    }
    assert ("unpause", "work-1") not in prior_runtime_events
    assert runtime.work_paused is False
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR


def test_snapshot_plan_cancel_persistence_failure_retains_work_pause(
    tmp_path,
) -> None:
    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    snapshot.fail_at = "acquire"

    class Observer:
        def resource_event(self, name: str, **values: object) -> None:
            del values
            if name == "snapshot_cancelled":
                raise OSError("snapshot cancellation fsync failed")

    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=Observer(),
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in (report.error or "")
    assert runtime.work_paused is True
    assert ("unpause", "work-1") not in runtime.events


def test_ambiguous_snapshot_acquire_retains_plan_and_work_pause(tmp_path) -> None:
    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    snapshot.fail_at = "acquire_ambiguous"
    observed: list[str] = []

    class Observer:
        def resource_event(self, name: str, **values: object) -> None:
            del values
            observed.append(name)

    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=Observer(),
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in (report.error or "")
    assert "snapshot_cancelled" not in observed
    assert runtime.work_paused is True
    assert ("unpause", "work-1") not in runtime.events


def test_state_transition_error_cleans_judge_then_escapes_as_engine_bug(
    tmp_path,
) -> None:
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)

    class Observer:
        def resource_event(self, name, **values) -> None:
            del values
            if name == "judge_planned":
                raise StateTransitionError("impossible judging transition")

    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=Observer(),
    )

    with pytest.raises(StateTransitionError, match="impossible"):
        evaluate(runner, request)

    assert ("snapshot_release", "agent-1") in runtime.events
    assert ("unpause", "work-1") in runtime.events
    assert artifacts.reports == []


@pytest.mark.parametrize(
    ("stage", "expected_tail"),
    [
        ("acquire", [("snapshot_acquire", "agent-1"), ("unpause", "work-1")]),
        ("create", [("snapshot_release", "agent-1"), ("unpause", "work-1")]),
        ("start", [("snapshot_release", "agent-1"), ("unpause", "work-1")]),
        ("exec", [("snapshot_release", "agent-1"), ("unpause", "work-1")]),
        ("reward", [("snapshot_release", "agent-1"), ("unpause", "work-1")]),
        ("remove", [("judge_remove", "agent-1"), ("judge_contain", "agent-1")]),
        ("release", [("snapshot_release", "agent-1")]),
    ],
)
def test_failure_matrix_always_resumes_work_and_attempts_later_cleanup(
    tmp_path, stage, expected_tail
):
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    if stage in {"create", "start", "exec", "remove"}:
        runtime.fail_at = stage
    elif stage in {"acquire", "release"}:
        snapshot.fail_at = stage
    else:
        runner.reward_reader = lambda *_args: (_ for _ in ()).throw(
            RuntimeError("reward failed")
        )

    report = evaluate(runner, request)

    assert runtime.events[-len(expected_tail) :] == expected_tail
    assert runtime.events.count(("unpause", "work-1")) == (
        0 if stage in {"remove", "release"} else 1
    )
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert report.score is None
    assert report.score != 0
    assert stage in report.error
    assert artifacts.reports == [report]


def test_contained_remove_failure_stops_judge_but_requires_durable_recovery(
    tmp_path,
):
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    runtime.fail_at = "remove"

    report = evaluate(runner, request)

    assert runtime.events[-2:] == [
        ("judge_remove", "agent-1"),
        ("judge_contain", "agent-1"),
    ]
    assert ("snapshot_release", "agent-1") not in runtime.events
    assert ("unpause", "work-1") not in runtime.events
    assert runtime.closed_rounds == 0
    assert runtime.work_paused is True
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in report.error
    assert "Judge removal: remove failed" in report.error
    assert "emergency containment proved Judge stopped or gone" in report.error
    assert artifacts.reports == [report]


def test_uncontained_live_judge_retains_all_isolation_resources_and_work_pause(
    tmp_path,
):
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    runtime.fail_at = {"remove", "contain"}

    report = evaluate(runner, request)

    assert runtime.events[-2:] == [
        ("judge_remove", "agent-1"),
        ("judge_contain", "agent-1"),
    ]
    assert ("snapshot_release", "agent-1") not in runtime.events
    assert ("unpause", "work-1") not in runtime.events
    assert runtime.closed_rounds == 0
    assert runtime.work_paused is True
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert report.score is None
    assert "recovery_required" in report.error
    assert "judge_container_id=judge-agent-1" in report.error
    assert "network_id=network-agent-1" in report.error
    assert "policy_rule_id=policy-agent-1" in report.error
    assert f"snapshot_lease_id={snapshot.acquired[0].lease_id}" in report.error
    assert "work_container_id=work-1" in report.error
    assert artifacts.reports == [report]


def test_uncontained_recovery_identifiers_are_durable_before_return(tmp_path):
    _runner, runtime, snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    artifacts = RunArtifactWriter(plan, run_id="run-1")
    artifacts.start()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
    )
    runtime.fail_at = {"remove", "contain"}

    report = evaluate(runner, (plan, work, logs))

    persisted = json.loads(
        (artifacts.root / "submissions" / "agent-1" / "report.json").read_text()
    )
    assert persisted["status"] == "infrastructure_error"
    assert persisted["error"] == report.error
    assert "recovery_required" in persisted["error"]
    assert "judge_container_id=judge-agent-1" in persisted["error"]
    assert runtime.work_paused is True


def test_pause_mutates_state_then_raises_still_attempts_idempotent_unpause(tmp_path):
    runner, runtime, _snapshot, artifacts, request = make_harness(tmp_path)
    runtime.fail_at = "pause"

    report = evaluate(runner, request)

    assert runtime.events == [
        ("pause", "work-1"),
        ("unpause", "work-1"),
    ]
    assert runtime.work_paused is False
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "pause failed" in report.error
    assert artifacts.reports == [report]


def test_unpause_mutates_then_raises_repauses_work_before_recovery(tmp_path):
    class Observer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.events.append((name, values))

    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    observer = Observer()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )
    runtime.fail_at = "unpause"

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in (report.error or "")
    assert runtime.work_paused is True
    assert runtime.events[-4:] == [
        ("unpause", "work-1"),
        ("inspect_quiescence", "work-1"),
        ("pause", "work-1"),
        ("inspect_quiescence", "work-1"),
    ]
    assert any(name == "work_recovery_state" for name, _ in observer.events)


def test_unpause_observer_failure_repauses_work_and_persists_factual_state(
    tmp_path,
) -> None:
    class Observer:
        def __init__(self) -> None:
            self.failed = False
            self.events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            if name == "work_unpaused" and not self.failed:
                self.failed = True
                raise OSError("work unpause lease fsync failed")
            self.events.append((name, values))

    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    observer = Observer()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "work unpause lease fsync failed" in (report.error or "")
    assert "recovery_required" in (report.error or "")
    assert runtime.work_paused is True
    assert runtime.events[-4:] == [
        ("unpause", "work-1"),
        ("inspect_quiescence", "work-1"),
        ("pause", "work-1"),
        ("inspect_quiescence", "work-1"),
    ]
    recovery_state = next(
        values for name, values in observer.events if name == "work_recovery_state"
    )
    assert recovery_state == {
        "work_container_id": "work-1",
        "paused": True,
        "stopped": False,
    }


def test_ambiguous_repause_attestation_never_records_false_paused_authority(
    tmp_path,
) -> None:
    class Observer:
        def __init__(self) -> None:
            self.events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.events.append((name, values))

    _runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    observer = Observer()
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=runtime,
        judge_runtime_factory=runtime.for_round,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )
    runtime.fail_at = {"unpause", "inspect_quiescence"}

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert runtime.events.count(("pause", "work-1")) == 2
    unproven = next(
        values for name, values in observer.events if name == "work_quiescence_unproven"
    )
    assert unproven == {
        "work_container_id": "work-1",
        "paused": False,
        "stopped": False,
    }
    assert "exact Work quiescence remains unproven" in (report.error or "")


def test_primary_and_remove_failure_retain_dependent_cleanup_for_recovery(tmp_path):
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    runtime.fail_at = {"exec", "remove", "close", "unpause"}
    snapshot.fail_at = "release"

    report = evaluate(runner, request)

    assert runtime.events[-2:] == [
        ("judge_remove", "agent-1"),
        ("judge_contain", "agent-1"),
    ]
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert report.score is None
    assert "primary failure: exec failed" in report.error
    assert "Judge removal: remove failed" in report.error
    assert "emergency containment proved Judge stopped or gone" in report.error
    assert "recovery_required" in report.error
    assert "Judge network/policy cleanup: close failed" not in report.error
    assert "snapshot release: release failed" not in report.error
    assert "Work unpause: unpause failed" not in report.error
    assert artifacts.reports == [report]


@pytest.mark.parametrize("cleanup_stage", ["close", "release"])
def test_isolation_or_snapshot_cleanup_failure_requires_recovery_and_keeps_pause(
    tmp_path, cleanup_stage
):
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    if cleanup_stage == "close":
        runtime.fail_at = "close"
    else:
        snapshot.fail_at = "release"

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert report.error is not None and "recovery_required" in report.error
    assert runtime.work_paused is True
    assert ("unpause", "work-1") not in runtime.events
    if cleanup_stage == "close":
        assert ("snapshot_release", "agent-1") not in runtime.events
    assert artifacts.reports == [report]


def test_snapshot_release_failure_closes_actual_submission_service(tmp_path):
    runner, runtime, snapshot, artifacts, request = make_harness(tmp_path)
    snapshot.fail_at = "release"
    plan, work, _logs = request
    service = SubmissionService(
        evaluator=runner,
        artifact_writer=artifacts,
        clock=FakeClock(),
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
    )

    feedback = service.submit(token)

    assert "recovery_required" in feedback
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)
    assert runtime.round_ids == ["agent-1"]
    assert runtime.work_paused is True


def test_timeout_discards_pretermination_reward_without_reading_it(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    runtime.exec_result = AgentRunResult(
        exit_code=None,
        output="partial",
        timed_out=True,
    )
    reads = 0

    def forbidden_reader(*_args):
        nonlocal reads
        reads += 1
        raise AssertionError("timed-out reward must not be read")

    runner.reward_reader = forbidden_reader

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.VERIFIER_TIMEOUT
    assert report.rewards == {}
    assert report.score is None
    assert report.timed_out is True
    assert reads == 0
    log_dir = request[2] / "agent-1"
    assert not (log_dir / "reward.json").exists()
    exec_index = runtime.events.index(("judge_exec", ("/bin/bash", "/tests/test.sh")))
    remove_index = runtime.events.index(("judge_remove", "agent-1"))
    assert exec_index < remove_index


def test_valid_reward_completes_despite_nonzero_test_exit(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    runtime.exec_result = AgentRunResult(exit_code=9, output="failed assertion")

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert report.exit_code == 9
    assert report.score == 1


def test_missing_reward_is_verifier_error_not_zero(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    runtime.reward_payload = None

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.VERIFIER_ERROR
    assert report.score is None
    assert report.rewards == {}
    assert "missing" in report.error.lower()


def test_output_is_bounded_with_explicit_marker_before_artifact_return(tmp_path):
    runner, runtime, _snapshot, artifacts, request = make_harness(tmp_path)
    runtime.exec_result = AgentRunResult(
        exit_code=0,
        output="A" * 80,
        output_truncated=True,
    )

    report = evaluate(runner, request)

    assert OUTPUT_TRUNCATION_MARKER in report.output
    assert len(report.output.encode()) <= 80
    assert artifacts.reports[0].output == report.output


def test_judge_streams_complete_raw_output_to_the_round_feedback_file(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    complete = (
        b"head\nAuthorization: Bearer task-authored-literal\n"
        + b"middle" * 100
        + b"\ntail\n"
    )
    runtime.complete_exec_output = complete
    runtime.exec_result = AgentRunResult(
        exit_code=0,
        output="head\ntail\n",
        output_truncated=True,
        full_output_captured=True,
    )
    expected = (
        request[0].paths.logs
        / "runs/run-1"
        / request[0].task.task_id
        / "feedback/agent-1.log"
    )

    evaluate(runner, request)

    assert runtime.exec_output_paths == [expected]
    assert expected.read_bytes() == complete


def test_judge_rejects_noncanonical_feedback_output_before_pause(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    from rsi_harness.models import EvaluationRequest

    report = runner.evaluate(
        EvaluationRequest(
            run_plan=plan,
            work_container=work,
            round_id="agent-1",
            verifier_logs=log_dir / "agent-1",
            verifier_output=tmp_path / "outside.log",
        )
    )

    assert report.status is SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "feedback output path" in (report.error or "")
    assert not any(event[0] == "pause" for event in runtime.events)
    assert not (tmp_path / "outside.log").exists()


def test_judge_rejects_traversing_round_before_output_or_pause(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    from rsi_harness.models import EvaluationRequest

    escaped = (
        plan.paths.logs
        / "runs/run-1"
        / plan.task.task_id
        / "feedback/../escaped.log"
    ).resolve()
    report = runner.evaluate(
        EvaluationRequest(
            run_plan=plan,
            work_container=work,
            round_id="../escaped",
            verifier_logs=log_dir / "escaped",
            verifier_output=escaped,
        )
    )

    assert report.status is SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "round identity" in (report.error or "")
    assert not any(event[0] == "pause" for event in runtime.events)
    assert not escaped.exists()


def test_judge_rejects_symlinked_feedback_directory_before_pause(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    from rsi_harness.models import EvaluationRequest

    feedback = (
        plan.paths.logs / "runs/run-1" / plan.task.task_id / "feedback"
    )
    redirected = plan.paths.logs / "redirected-feedback"
    redirected.mkdir(parents=True)
    feedback.rmdir()
    feedback.symlink_to(redirected, target_is_directory=True)
    report = runner.evaluate(
        EvaluationRequest(
            run_plan=plan,
            work_container=work,
            round_id="agent-1",
            verifier_logs=log_dir / "agent-1",
            verifier_output=feedback / "agent-1.log",
        )
    )

    assert report.status is SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "feedback directory" in (report.error or "")
    assert not any(event[0] == "pause" for event in runtime.events)
    assert not (redirected / "agent-1.log").exists()


def test_output_limit_rejects_values_smaller_than_explicit_marker():
    marker_size = len(OUTPUT_TRUNCATION_MARKER.encode())

    accepted = VerifierPlan(
        command=("/bin/bash", "/tests/test.sh"),
        output_limit_bytes=marker_size,
    )
    assert accepted.output_limit_bytes == marker_size

    with pytest.raises(ValidationError, match="truncation marker"):
        VerifierPlan(
            command=("/bin/bash", "/tests/test.sh"),
            output_limit_bytes=marker_size - 1,
        )


def test_output_at_exact_marker_limit_remains_explicit(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, logs = request
    marker_size = len(OUTPUT_TRUNCATION_MARKER.encode())
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "verifier": plan.task.verifier.model_copy(
                        update={"output_limit_bytes": marker_size}
                    )
                }
            )
        }
    )
    runtime.exec_result = AgentRunResult(
        exit_code=0,
        output="A" * marker_size,
        output_truncated=True,
    )

    report = evaluate(runner, (plan, work, logs))

    assert report.output == OUTPUT_TRUNCATION_MARKER


def test_each_evaluation_opens_and_closes_a_fresh_round_runtime(tmp_path):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)

    first = evaluate(runner, request, "agent-1")
    second = evaluate(runner, request, "agent-2")

    assert first.status == second.status == SubmissionStatus.COMPLETED
    assert runtime.round_ids == ["agent-1", "agent-2"]
    assert runtime.closed_rounds == 2
    assert [event for event in runtime.events if event[0] == "judge_create"] == [
        ("judge_create", "agent-1"),
        ("judge_create", "agent-2"),
    ]


def test_docker_round_factory_creates_fresh_network_and_policy_per_submission(
    tmp_path,
):
    plan = make_run_plan(tmp_path)
    (plan.task.source_dir / "tests").mkdir(parents=True)
    (plan.task.source_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    logs = tmp_path / "logs" / "verifier"
    logs.mkdir(parents=True)
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    work_reference = install_work_volume_reference(client, plan)
    firewall = FakeFirewallBackend()
    enforcer = NetworkPolicyEnforcer(
        run_id="run-1",
        firewall=firewall,
    )
    observer = RecordingJudgeResourceObserver()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=enforcer,
        lifecycle_observer=observer,
        work_container=work_reference,
    )
    spec = ContainerSpec(
        image=plan.images.judge_ref,
        command=("/bin/sleep", "infinity"),
        workdir=plan.workdir,
        volume_mounts=(
            ContainerVolumeMount(
                volume=managed_workdir_volume(plan),
                target=plan.workdir,
                read_only=True,
            ),
        ),
        mounts=(
            ContainerMount(
                source=plan.task.source_dir / "tests",
                target=PurePosixPath("/tests"),
                read_only=True,
            ),
            ContainerMount(
                source=logs,
                target=PurePosixPath("/logs/verifier"),
            ),
        ),
    )

    for round_id in ("agent-1", "agent-2"):
        runtime = factory(plan, round_id)
        container = runtime.create(spec)
        runtime.start(container)
        runtime.remove(container)
        runtime.close()

    assert [item["name"] for item in client.networks.created] == [
        "rsi-run-1-minimal-gpu-judge-agent-1",
        "rsi-run-1-minimal-gpu-judge-agent-2",
    ]
    assert all(item["internal"] is True for item in client.networks.created)
    assert len([event for event in firewall.events if event[0] == "install"]) == 2
    assert len([event for event in firewall.events if event[0] == "remove"]) == 2
    assert all(network.removed for network in client.networks.by_id.values())
    names = [name for name, _ in observer.events]
    assert names.index("network_planned") < names.index("network_created")
    assert names.index("container_planned") < names.index("container_created")
    assert names.index("policy_planned") < names.index("policy_installed")


@pytest.mark.parametrize("failure_stage", ("policy", "network"))
def test_concrete_round_cleanup_failure_retains_dependent_authority(
    tmp_path,
    failure_stage,
) -> None:
    plan = make_run_plan(tmp_path)
    plan.task.source_dir.mkdir(parents=True, exist_ok=True)
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    firewall = FakeFirewallBackend()
    observer = RecordingJudgeResourceObserver()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=firewall
        ),
        lifecycle_observer=observer,
    )
    runtime = factory(plan, "agent-1")
    container = runtime.create(ContainerSpec(image=plan.images.judge_ref))
    runtime.start(container)
    runtime.remove(container)
    network = client.networks.by_id["network-1"]

    if failure_stage == "policy":

        def fail_policy_remove(_rule_id):
            raise RuntimeError("policy cleanup failed")

        firewall.remove = fail_policy_remove  # type: ignore[method-assign]
    else:

        def fail_network_remove():
            raise RuntimeError("network cleanup failed")

        network.remove = fail_network_remove  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        runtime.close()

    names = [name for name, _ in observer.events]
    assert runtime.safe_to_release_isolation is False
    assert network.removed is False
    assert "network_removed" not in names
    if failure_stage == "policy":
        assert firewall.installed
        assert "policy_removed" not in names
    else:
        assert firewall.installed == {}
        assert "policy_removed" in names


def test_durable_network_plan_failure_aborts_before_docker_mutation(tmp_path):
    class FailingObserver(RecordingJudgeResourceObserver):
        def judge_network_planned(self, round_id, planned_name):
            raise OSError("lease fsync failed")

    plan = make_run_plan(tmp_path)
    plan.task.source_dir.mkdir(parents=True, exist_ok=True)
    client = FakeDockerClient()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=FakeFirewallBackend()
        ),
        lifecycle_observer=FailingObserver(),
    )

    with pytest.raises(OSError, match="fsync"):
        factory(plan, "agent-1")

    assert client.networks.created == []


def test_network_created_observer_failure_durably_observes_proven_rollback(
    tmp_path,
) -> None:
    class FailingObserver(RecordingJudgeResourceObserver):
        def judge_network_created(self, round_id, network):
            del round_id, network
            raise OSError("network lease fsync failed")

    plan = make_run_plan(tmp_path)
    plan.task.source_dir.mkdir(parents=True, exist_ok=True)
    client = FakeDockerClient()
    observer = FailingObserver()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=FakeFirewallBackend()
        ),
        lifecycle_observer=observer,
    )

    with pytest.raises(OSError, match="fsync"):
        factory(plan, "agent-1")

    network = client.networks.by_id["network-1"]
    assert network.removed is True
    assert observer.events[-1] == ("network_removed", network.id)


def test_network_rollback_observer_failure_retains_snapshot_and_work_pause(
    tmp_path,
) -> None:
    class FailingObserver(RecordingJudgeResourceObserver):
        def __init__(self) -> None:
            super().__init__()
            self.lifecycle_events: list[tuple[str, dict[str, object]]] = []

        def judge_network_created(self, round_id, network):
            del round_id, network
            raise OSError("network lease fsync failed")

        def judge_network_removed(self, round_id, network_id):
            del round_id, network_id
            raise OSError("network rollback fsync failed")

        def resource_event(self, name: str, **values: object) -> None:
            self.lifecycle_events.append((name, values))

    _runner, work_runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    client = FakeDockerClient()
    observer = FailingObserver()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=FakeFirewallBackend()
        ),
        lifecycle_observer=observer,
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=work_runtime,
        judge_runtime_factory=factory,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=work_runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in (report.error or "")
    assert client.networks.by_id["network-1"].removed is True
    assert work_runtime.work_paused is True
    assert ("snapshot_release", "agent-1") not in work_runtime.events
    assert ("unpause", "work-1") not in work_runtime.events


def test_network_created_observer_and_rollback_failure_retains_snapshot_and_pause(
    tmp_path,
) -> None:
    class FailingObserver(RecordingJudgeResourceObserver):
        def __init__(self) -> None:
            super().__init__()
            self.lifecycle_events: list[tuple[str, dict[str, object]]] = []

        def judge_network_created(self, round_id, network):
            del round_id, network
            raise OSError("network lease fsync failed")

        def resource_event(self, name: str, **values: object) -> None:
            self.lifecycle_events.append((name, values))

    _runner, work_runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    client = FakeDockerClient()
    observer = FailingObserver()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=FakeFirewallBackend()
        ),
        lifecycle_observer=observer,
    )
    original_create = client.networks.create

    def create_with_failed_rollback(*args, **kwargs):
        network = original_create(*args, **kwargs)

        def fail_remove():
            raise RuntimeError("network rollback failed")

        network.remove = fail_remove
        return network

    client.networks.create = create_with_failed_rollback  # type: ignore[method-assign]
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=work_runtime,
        judge_runtime_factory=factory,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=work_runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in (report.error or "")
    assert work_runtime.work_paused is True
    assert ("snapshot_release", "agent-1") not in work_runtime.events
    assert ("unpause", "work-1") not in work_runtime.events
    assert client.networks.by_id["network-1"].removed is False
    assert any(name == "recovery_required" for name, _ in observer.lifecycle_events)


def test_post_network_runtime_construction_and_rollback_failure_retains_isolation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rsi_harness.runtime.judge as judge_module

    class RecordingObserver(RecordingJudgeResourceObserver):
        def __init__(self) -> None:
            super().__init__()
            self.lifecycle_events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.lifecycle_events.append((name, values))

    _runner, work_runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    client = FakeDockerClient()
    observer = RecordingObserver()
    original_runtime = judge_module.DockerContainerRuntime
    constructions = 0

    def fail_second_runtime(*args, **kwargs):
        nonlocal constructions
        constructions += 1
        if constructions == 2:
            raise RuntimeError("post-network runtime setup failed")
        return original_runtime(*args, **kwargs)

    monkeypatch.setattr(judge_module, "DockerContainerRuntime", fail_second_runtime)
    original_create = client.networks.create

    def create_with_failed_rollback(*args, **kwargs):
        network = original_create(*args, **kwargs)

        def fail_remove():
            raise RuntimeError("network rollback failed")

        network.remove = fail_remove
        return network

    client.networks.create = create_with_failed_rollback  # type: ignore[method-assign]
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=FakeFirewallBackend()
        ),
        lifecycle_observer=observer,
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=work_runtime,
        judge_runtime_factory=factory,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=work_runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )

    report = evaluate(runner, (plan, work, log_dir))

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "recovery_required" in (report.error or "")
    assert work_runtime.work_paused is True
    assert ("snapshot_release", "agent-1") not in work_runtime.events
    assert ("unpause", "work-1") not in work_runtime.events
    assert client.networks.by_id["network-1"].removed is False
    assert any(name == "recovery_required" for name, _ in observer.lifecycle_events)


def test_judge_factory_propagates_explicit_test_gpu_omission(tmp_path):
    plan = make_run_plan(tmp_path)
    (plan.task.source_dir / "tests").mkdir(parents=True)
    client = FakeDockerClient()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=FakeFirewallBackend()
        ),
        lifecycle_observer=RecordingJudgeResourceObserver(),
        omit_gpu_device_requests_for_tests=True,
    )
    runtime = factory(plan, "agent-1")

    runtime.create(
        ContainerSpec(
            image=plan.images.judge_ref,
            gpu_allocation=GPUAllocation(
                devices=(GPUDevice(index=0, uuid="GPU-test", name="Test GPU"),)
            ),
        )
    )

    assert client.containers.created[0]["device_requests"] == []


def test_docker_round_cleans_applied_policy_when_runtime_install_fails(tmp_path):
    class FailingAttestEnforcer(NetworkPolicyEnforcer):
        def attest(self, lease):
            del lease
            raise InfrastructureError("attestation failed")

    plan = make_run_plan(tmp_path)
    (plan.task.source_dir / "tests").mkdir(parents=True)
    (plan.task.source_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    logs = tmp_path / "logs" / "verifier"
    logs.mkdir(parents=True)
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    work_reference = install_work_volume_reference(client, plan)
    firewall = FakeFirewallBackend()
    enforcer = FailingAttestEnforcer(run_id="run-1", firewall=firewall)
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=enforcer,
        lifecycle_observer=RecordingJudgeResourceObserver(),
        work_container=work_reference,
    )
    runtime = factory(plan, "agent-1")
    spec = ContainerSpec(
        image=plan.images.judge_ref,
        command=("/bin/sleep", "infinity"),
        workdir=plan.workdir,
        volume_mounts=(
            ContainerVolumeMount(
                volume=managed_workdir_volume(plan),
                target=plan.workdir,
                read_only=True,
            ),
        ),
        mounts=(
            ContainerMount(
                source=plan.task.source_dir / "tests",
                target=PurePosixPath("/tests"),
                read_only=True,
            ),
            ContainerMount(
                source=logs,
                target=PurePosixPath("/logs/verifier"),
            ),
        ),
    )

    with pytest.raises(InfrastructureError, match="attestation failed"):
        runtime.create(spec)
    runtime.close()

    assert client.containers.by_id["container-1"].removed_kwargs == {
        "force": True,
        "v": True,
    }
    assert [event for event, _ in firewall.events].count("install") == 1
    assert [event for event, _ in firewall.events].count("remove") == 1
    assert client.networks.by_id["network-1"].removed is True


def test_create_rollback_policy_failure_retains_network_and_recovery_authority(
    tmp_path,
) -> None:
    class FailingAttestEnforcer(NetworkPolicyEnforcer):
        def attest(self, lease):
            del lease
            raise InfrastructureError("attestation failed")

    plan = make_run_plan(tmp_path)
    plan.task.source_dir.mkdir(parents=True, exist_ok=True)
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    work_reference = install_work_volume_reference(client, plan)
    firewall = FakeFirewallBackend()
    enforcer = FailingAttestEnforcer(run_id="run-1", firewall=firewall)
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=enforcer,
        lifecycle_observer=RecordingJudgeResourceObserver(),
        work_container=work_reference,
    )
    runtime = factory(plan, "agent-1")

    def fail_policy_remove(_rule_id):
        raise RuntimeError("rollback policy cleanup failed")

    firewall.remove = fail_policy_remove  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        runtime.create(ContainerSpec(image=plan.images.judge_ref))
    with pytest.raises(InfrastructureError, match="recovery_required"):
        runtime.close()

    assert runtime.safe_to_release_isolation is False
    assert firewall.installed
    assert client.networks.by_id["network-1"].removed is False


def test_policy_install_failure_with_unproved_remove_closes_submission_recovery(
    tmp_path,
) -> None:
    """A stopped-but-unremoved container must retain every isolation layer."""

    class FailingAttestEnforcer(NetworkPolicyEnforcer):
        def attest(self, lease):
            del lease
            raise InfrastructureError("production policy attestation failed")

    class LifecycleObserver(RecordingJudgeResourceObserver):
        def __init__(self) -> None:
            super().__init__()
            self.lifecycle_events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.lifecycle_events.append((name, values))

    _runner, work_runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, _logs = request
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    install_work_volume_reference(client, plan, container_ref=work)
    firewall = FakeFirewallBackend()
    observer = LifecycleObserver()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=FailingAttestEnforcer(
            run_id="run-1", firewall=firewall
        ),
        lifecycle_observer=observer,
        work_container=work,
    )
    original_create = client.containers.create

    def create_with_unproved_removal(*args, **kwargs):
        container = original_create(*args, **kwargs)

        def fail_remove(**_remove_kwargs):
            raise APIError("production remove failed")

        container.remove = fail_remove
        return container

    client.containers.create = create_with_unproved_removal  # type: ignore[method-assign]
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=work_runtime,
        judge_runtime_factory=factory,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=work_runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )
    service_writer = FakeArtifactWriter()
    service = SubmissionService(
        evaluator=runner,
        artifact_writer=service_writer,
        clock=FakeClock(),
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
        max_submissions=2,
    )

    feedback = service.submit(token)

    assert "status: infrastructure_error" in feedback
    assert "recovery_required" in feedback
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)
    assert service.session_state.rounds_allocated == 1
    assert len(client.networks.created) == 1
    docker_container = client.containers.by_id["container-1"]
    assert docker_container.attrs["State"]["Running"] is False
    assert client.networks.by_id["network-1"].removed is False
    assert len(firewall.installed) == 1
    assert work_runtime.work_paused is True
    assert ("snapshot_release", "agent-1") not in work_runtime.events
    assert ("unpause", "work-1") not in work_runtime.events
    names = [name for name, _ in observer.events]
    assert "network_created" in names
    assert "container_created" in names
    assert "policy_installed" in names
    assert any(name == "recovery_required" for name, _ in observer.lifecycle_events)


def test_partial_policy_install_rollback_failure_retains_round_and_closes_submits(
    tmp_path,
) -> None:
    class PartialRollbackFirewall(FakeFirewallBackend):
        def install(self, rule_id, rules):
            super().install(rule_id, rules)
            raise InfrastructureError(
                "recovery_required: partial network policy rollback is unproven"
            )

        def remove(self, rule_id):
            self.events.append(("remove", rule_id))
            raise RuntimeError("partial policy still cannot be removed")

    class LifecycleObserver(RecordingJudgeResourceObserver):
        def __init__(self) -> None:
            super().__init__()
            self.lifecycle_events: list[tuple[str, dict[str, object]]] = []

        def resource_event(self, name: str, **values: object) -> None:
            self.lifecycle_events.append((name, values))

    _runner, work_runtime, snapshot, artifacts, request = make_harness(tmp_path)
    plan, work, _logs = request
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    install_work_volume_reference(client, plan, container_ref=work)
    firewall = PartialRollbackFirewall()
    observer = LifecycleObserver()
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=NetworkPolicyEnforcer(
            run_id="run-1", firewall=firewall
        ),
        lifecycle_observer=observer,
        work_container=work,
    )
    runner = JudgeRunner(
        run_id="run-1",
        workdir_volume=managed_workdir_volume(request[0]),
        work_runtime=work_runtime,
        judge_runtime_factory=factory,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=work_runtime.assert_gpu_quiet,
        lifecycle_observer=observer,
    )
    service = SubmissionService(
        evaluator=runner,
        artifact_writer=FakeArtifactWriter(),
        clock=FakeClock(),
    )
    token = service.register(
        run_id="run-1",
        run_plan=plan,
        work_container=work,
        max_submissions=2,
    )

    feedback = service.submit(token)

    assert "recovery_required" in feedback
    with pytest.raises(SubmissionClosedError, match="closed"):
        service.submit(token)
    assert work_runtime.work_paused is True
    assert ("snapshot_release", "agent-1") not in work_runtime.events
    assert ("unpause", "work-1") not in work_runtime.events
    assert client.networks.by_id["network-1"].removed is False
    assert firewall.installed
    assert any(name == "recovery_required" for name, _ in observer.lifecycle_events)


def test_production_round_retains_policy_when_remove_fails_and_judge_stays_live(
    tmp_path,
):
    plan = make_run_plan(tmp_path)
    (plan.task.source_dir / "tests").mkdir(parents=True)
    (plan.task.source_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    logs = tmp_path / "logs" / "verifier"
    logs.mkdir(parents=True)
    client = FakeDockerClient()
    install_managed_workdir_volume(client, plan)
    work_reference = install_work_volume_reference(client, plan)
    firewall = FakeFirewallBackend()
    enforcer = NetworkPolicyEnforcer(run_id="run-1", firewall=firewall)
    factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-1",
        task_id=plan.task.task_id,
        task_source_dir=plan.task.source_dir,
        allowed_mount_roots=(tmp_path.resolve(),),
        network_policy_enforcer=enforcer,
        lifecycle_observer=RecordingJudgeResourceObserver(),
        work_container=work_reference,
    )
    runtime = factory(plan, "agent-1")
    spec = ContainerSpec(
        image=plan.images.judge_ref,
        command=("/bin/sleep", "infinity"),
        workdir=plan.workdir,
        volume_mounts=(
            ContainerVolumeMount(
                volume=managed_workdir_volume(plan),
                target=plan.workdir,
                read_only=True,
            ),
        ),
        mounts=(
            ContainerMount(
                source=plan.task.source_dir / "tests",
                target=PurePosixPath("/tests"),
                read_only=True,
            ),
            ContainerMount(
                source=logs,
                target=PurePosixPath("/logs/verifier"),
            ),
        ),
    )
    container = runtime.create(spec)
    runtime.start(container)
    docker_container = client.containers.by_id[container.container_id]

    def failed_remove(**_kwargs):
        raise APIError("remove failed")

    def ineffective_stop(**_kwargs):
        docker_container.events.append("stop")

    docker_container.remove = failed_remove
    docker_container.stop = ineffective_stop

    with pytest.raises(InfrastructureError, match="failed to remove"):
        runtime.remove(container)
    with pytest.raises(InfrastructureError, match="containment.*unproven"):
        runtime.contain_after_remove_failure(container)
    with pytest.raises(InfrastructureError, match="recovery_required"):
        runtime.close()

    assert docker_container.attrs["State"]["Running"] is True
    assert len(firewall.installed) == 1
    assert client.networks.by_id["network-1"].removed is False
    context = runtime.recovery_context(container)
    assert "judge_container_id=container-1" in context
    assert "network_id=network-1" in context
    assert "policy_rule_id=rsi-run-1-judge-" in context
