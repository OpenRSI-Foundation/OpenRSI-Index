import json
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from rsi_harness.models import (
    AgentHookRequest,
    AgentPlan,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    ContainerSpec,
    ContainerVolumeMount,
    EvaluationRequest,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    ManagedWorkdirVolume,
    NetworkPolicy,
    PreparedAgent,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
    SubmissionReport,
    SubmissionStatus,
    VerifierPlan,
)
from tests.factories import make_run_plan


def test_run_plan_is_frozen_and_contains_no_secret_values(tmp_path):
    """Removing frozen persistence or serializing a secret value is a defect."""
    plan = make_run_plan(tmp_path, secret_env_names=("OPENAI_API_KEY",))

    payload = plan.model_dump_json()

    assert "OPENAI_API_KEY" in payload
    assert "sk-test-secret" not in payload
    with pytest.raises(ValidationError):
        plan.workdir = PurePosixPath("/changed")


def test_gpu_allocation_rejects_duplicate_physical_uuid():
    """Removing physical-device uniqueness would let one GPU be allocated twice."""
    with pytest.raises(ValidationError, match="duplicate"):
        GPUAllocation(
            devices=(
                GPUDevice(index=0, uuid="GPU-a", name="H100"),
                GPUDevice(index=1, uuid="GPU-a", name="H100"),
            )
        )


@pytest.mark.parametrize("phase", ("work", "judge"))
def test_run_gpu_plan_rejects_phase_devices_outside_authorized_pool(phase):
    """A phase allocation outside the lease could access an unreserved GPU."""
    outside_pool = GPUAllocation(
        devices=(GPUDevice(index=0, uuid="GPU-outside", name="H100"),)
    )
    allocations = {"work": GPUAllocation(), "judge": GPUAllocation()}
    allocations[phase] = outside_pool

    with pytest.raises(ValidationError, match="escapes authorized pool"):
        RunGPUPlan(
            authorized_pool=GPUAllocation(),
            work=allocations["work"],
            judge=allocations["judge"],
            judge_mode=JudgeGPUMode.FREEZE_ONLY,
        )


def test_run_gpu_plan_rejects_gpus_for_freeze_only_judge():
    """A frozen judge must not receive a GPU device."""
    judge = GPUAllocation(
        devices=(GPUDevice(index=0, uuid="GPU-judge", name="H100"),)
    )

    with pytest.raises(ValidationError, match="freeze-only"):
        RunGPUPlan(
            authorized_pool=judge,
            work=GPUAllocation(),
            judge=judge,
            judge_mode=JudgeGPUMode.FREEZE_ONLY,
        )


def test_run_gpu_plan_rejects_overlapping_disjoint_devices():
    """Disjoint judging cannot concurrently receive a work GPU."""
    shared = GPUAllocation(
        devices=(GPUDevice(index=0, uuid="GPU-shared", name="H100"),)
    )

    with pytest.raises(ValidationError, match="disjoint"):
        RunGPUPlan(
            authorized_pool=shared,
            work=shared,
            judge=shared,
            judge_mode=JudgeGPUMode.DISJOINT,
        )


def test_run_gpu_plan_rejects_non_overlapping_release_all_devices():
    """Release-all judging must reuse at least one work GPU."""
    work = GPUAllocation(
        devices=(GPUDevice(index=0, uuid="GPU-work", name="H100"),)
    )
    judge = GPUAllocation(
        devices=(GPUDevice(index=1, uuid="GPU-judge", name="H100"),)
    )

    with pytest.raises(ValidationError, match="release-all"):
        RunGPUPlan(
            authorized_pool=GPUAllocation(devices=work.devices + judge.devices),
            work=work,
            judge=judge,
            judge_mode=JudgeGPUMode.RELEASE_ALL,
        )


def test_run_plan_serializes_exact_gpu_plan_without_legacy_allocation(tmp_path):
    """Persisted plans must retain phase UUIDs without the retired allocation key."""
    plan = make_run_plan(tmp_path)
    work = GPUAllocation(
        devices=(GPUDevice(index=0, uuid="GPU-work", name="H100"),)
    )
    judge = GPUAllocation(
        devices=(GPUDevice(index=1, uuid="GPU-judge", name="H100"),)
    )
    gpu_plan = RunGPUPlan(
        authorized_pool=GPUAllocation(devices=work.devices + judge.devices),
        work=work,
        judge=work,
        judge_mode=JudgeGPUMode.RELEASE_ALL,
    )

    restored = type(plan).model_validate(
        {**plan.model_dump(), "gpu_plan": gpu_plan.model_dump()}
    )
    payload = json.loads(restored.model_dump_json())

    assert payload["gpu_plan"]["authorized_pool"]["devices"] == [
        {"index": 0, "uuid": "GPU-work", "name": "H100"},
        {"index": 1, "uuid": "GPU-judge", "name": "H100"},
    ]
    assert payload["gpu_plan"]["work"]["devices"] == [
        {"index": 0, "uuid": "GPU-work", "name": "H100"}
    ]
    assert payload["gpu_plan"]["judge"]["devices"] == [
        {"index": 0, "uuid": "GPU-work", "name": "H100"}
    ]
    assert "allocation" not in payload


@pytest.mark.parametrize(
    "value",
    ("", ".", "workspace", "//", "/workspace/..", "/workspace/../"),
)
def test_run_plan_rejects_invalid_or_isolation_escaping_workdir(value, tmp_path):
    """A workdir must be absolute without lexically escaping to root."""
    plan = make_run_plan(tmp_path)
    with pytest.raises(ValidationError):
        type(plan).model_validate({**plan.model_dump(), "workdir": value})


def test_run_plan_accepts_docker_root_workdir(tmp_path):
    plan = make_run_plan(tmp_path)

    restored = type(plan).model_validate({**plan.model_dump(), "workdir": "/"})

    assert restored.workdir == PurePosixPath("/")


def test_run_paths_reject_relative_persisted_paths(tmp_path):
    """Allowing relative persisted paths makes a saved plan location-dependent."""
    with pytest.raises(ValidationError):
        RunPaths(
            root=Path("runs"),
            workspace=tmp_path / "workspace",
            logs=tmp_path / "logs",
        )


def test_managed_workdir_volume_requires_non_root_target():
    """A root volume would hand a runtime authority over the whole rootfs."""
    with pytest.raises(ValidationError, match="non-root"):
        ManagedWorkdirVolume(
            name="rsi-volume-abc",
            run_id="run-1",
            task_id="task-1",
            target=PurePosixPath("/"),
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        )


@pytest.mark.parametrize("target", ("workspace", "/workspace/../etc"))
def test_managed_workdir_volume_rejects_unsafe_targets(target):
    """Removing workdir validation would allow a volume to escape its authority."""
    with pytest.raises(ValidationError, match="workdir"):
        ManagedWorkdirVolume(
            name="rsi-volume-abc",
            run_id="run-1",
            task_id="task-1",
            target=PurePosixPath(target),
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        )


@pytest.mark.parametrize("name", ("/rsi-volume", "rsi volume", "rsi/volume"))
def test_managed_workdir_volume_rejects_unsafe_names(name):
    """Unsafe volume names could select a Docker resource outside this authority."""
    with pytest.raises(ValidationError, match="unsafe"):
        ManagedWorkdirVolume(
            name=name,
            run_id="run-1",
            task_id="task-1",
            target=PurePosixPath("/workspace"),
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        )


def test_managed_workdir_volume_rejects_nonlocal_driver():
    """A nonlocal driver could turn a private workdir into an external mount."""
    with pytest.raises(ValidationError):
        ManagedWorkdirVolume(
            name="rsi-volume-abc",
            driver="nfs",
            run_id="run-1",
            task_id="task-1",
            target=PurePosixPath("/workspace"),
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        )


def test_managed_workdir_volume_requires_split_mode_and_freshness_identity():
    """Omitting mode or nonce would make a durable split authority reusable."""
    base = {
        "name": "rsi-volume-abc",
        "run_id": "run-1",
        "task_id": "task-1",
        "target": "/workspace",
        "snapshot_mode": "split-workdir",
        "freshness_nonce": "f" * 64,
    }

    expected = ManagedWorkdirVolume.model_validate(base)
    assert expected.snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR
    assert expected.freshness_nonce == "f" * 64
    for field in ("snapshot_mode", "freshness_nonce"):
        payload = dict(base)
        payload.pop(field)
        with pytest.raises(ValidationError, match=field):
            ManagedWorkdirVolume.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("snapshot_mode", "full-rootfs"),
        ("freshness_nonce", "short"),
        ("freshness_nonce", "g" * 64),
    ),
)
def test_managed_workdir_volume_rejects_wrong_mode_or_nonce(field, value):
    """Malformed provenance must never become mountable typed authority."""
    payload = {
        "name": "rsi-volume-abc",
        "run_id": "run-1",
        "task_id": "task-1",
        "target": "/workspace",
        "snapshot_mode": "split-workdir",
        "freshness_nonce": "f" * 64,
        field: value,
    }

    with pytest.raises(ValidationError):
        ManagedWorkdirVolume.model_validate(payload)


def test_container_volume_mount_requires_exact_managed_target():
    """Changing the mount target would detach access from the trusted workdir."""
    volume = ManagedWorkdirVolume(
        name="rsi-volume-abc",
        run_id="run-1",
        task_id="task-1",
        target=PurePosixPath("/workspace"),
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce="f" * 64,
    )

    with pytest.raises(ValidationError, match="differs"):
        ContainerVolumeMount(
            volume=volume,
            target=PurePosixPath("/other"),
            read_only=False,
        )


def test_submission_report_preserves_every_reward_key():
    """Dropping a reward key would corrupt a multi-objective Harbor result."""
    report = SubmissionReport(
        round_id="agent-1",
        status=SubmissionStatus.COMPLETED,
        rewards={"accuracy": 0.8, "latency": 12},
    )

    assert report.rewards == {"accuracy": 0.8, "latency": 12.0}
    assert report.score is None


def test_persisted_rewards_are_immutable_json_objects():
    """Mutable rewards permit post-validation changes to a persisted result."""
    report = SubmissionReport(
        round_id="agent-1",
        status=SubmissionStatus.COMPLETED,
        rewards={"reward": 1},
    )
    payload = report.model_dump_json()
    restored = SubmissionReport.model_validate_json(payload)

    assert json.loads(payload)["rewards"] == {"reward": 1.0}
    assert restored.rewards == {"reward": 1.0}
    with pytest.raises(TypeError):
        report.rewards["reward"] = 0


def test_secret_bearing_runtime_requests_cannot_serialize_tokens(tmp_path):
    """Persisting a bearer token or resolved environment value leaks a secret."""
    plan = make_run_plan(tmp_path)
    container = ContainerRef(container_id="work-1", role="work")
    prepared = PreparedAgent(
        agent_name="codex",
        command=("agent",),
        environment=(("OPENAI_API_KEY", "sk-test-secret"),),
        prompt_path=tmp_path,
    )
    request = AgentHookRequest(
        run_plan=plan,
        container=container,
        submit_url="http://judge.test",
        token="bearer-test-secret",
    )

    for runtime_object in (
        ContainerSpec(image="work", environment=prepared.environment),
        prepared,
        AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path),
        request,
        AgentRunRequest(prepared=prepared, container=container),
        AgentRunResult(exit_code=0),
        EvaluationRequest(
            run_plan=plan,
            work_container=container,
            round_id="agent-1",
            verifier_logs=tmp_path,
            verifier_output=tmp_path / "agent-1.log",
        ),
    ):
        assert is_dataclass(runtime_object)
        assert not hasattr(runtime_object, "model_dump_json")

    with pytest.raises(FrozenInstanceError):
        request.token = "changed"


@pytest.mark.parametrize("request_kind", ("prepared", "prepare", "evaluation"))
def test_runtime_request_paths_are_absolute_path_instances(request_kind, tmp_path):
    """Relative runtime paths make an otherwise fixed run location-dependent."""
    plan = make_run_plan(tmp_path)
    container = ContainerRef(container_id="work-1", role="work")

    def build(path):
        if request_kind == "prepared":
            return PreparedAgent(
                agent_name="codex", command=("agent",), prompt_path=path
            )
        if request_kind == "prepare":
            return AgentPrepareRequest(run_plan=plan, prompt_path=path)
        return EvaluationRequest(
            run_plan=plan,
            work_container=container,
            round_id="agent-1",
            verifier_logs=path,
            verifier_output=(
                path if isinstance(path, str) else path / "agent-1.log"
            ),
        )

    with pytest.raises(ValueError, match="absolute"):
        build("relative")

    request = build(str(tmp_path))
    field = "verifier_logs" if request_kind == "evaluation" else "prompt_path"
    assert getattr(request, field) == tmp_path
    assert isinstance(getattr(request, field), Path)


def test_agent_and_verifier_keep_immutable_network_policy():
    """A bare network mode loses the phase allowlist required by the runtime."""
    policy = NetworkPolicy(
        mode="allowlist",
        allowlist=(" API.EXAMPLE.COM ", "10.0.0.0/24"),
    )
    agent = AgentPlan(name="codex", network=policy)
    verifier = VerifierPlan(command=("/bin/bash", "/tests/test.sh"), network=policy)

    assert agent.network.mode == "allowlist"
    assert verifier.network.allowlist == ("api.example.com", "10.0.0.0/24")
    with pytest.raises(ValidationError):
        NetworkPolicy(mode="allowlist")
