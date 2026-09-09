from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela.allocation import AllocatedNode, AllocatedPools
from rsi_harness.cluster.bluevela.runtime import (
    ApptainerAgentRuntime,
    NativeEngineComposition,
    NativeJudgeEvaluator,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import (
    InfrastructureError,
    RetryableSubmissionError,
    SetupError,
)
from rsi_harness.models import (
    AgentAuthSource,
    AgentRunResult,
    ContainerRef,
    ContainerTmpfs,
    EvaluationRequest,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
)
from tests.factories import make_run_plan


class _Runtime:
    def __init__(self, root: Path) -> None:
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "answer.py").write_text("answer = 42\n")
        self.rounds = root / "rounds"
        self.rounds.mkdir()
        self.payload = SimpleNamespace(run_id="native-run")
        self.events: list[str] = []

    def pause(self, _work: ContainerRef) -> None:
        self.events.append("pause")

    def unpause(self, _work: ContainerRef) -> None:
        self.events.append("unpause")

    def require_work_idle(self) -> None:
        self.events.append("idle")

    def run_judge(self, workspace, request, environment):
        del workspace, environment
        (request.verifier_logs / "reward.txt").write_text("1\n")
        request.verifier_output.write_text("passed\n")
        return AgentRunResult(
            exit_code=0,
            output="passed\n",
            full_output_captured=True,
        )


class _Observer:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def resource_event(self, name: str, **values: object) -> None:
        self.events.append((name, values))


def _claude_plan(tmp_path: Path):
    plan = make_run_plan(tmp_path)
    agent = plan.task.agent.model_copy(update={"name": "claude-code"})
    task = plan.task.model_copy(update={"agent": agent})
    return plan.model_copy(update={"task": task})


def _claude_auth() -> AgentAuthMaterial:
    return AgentAuthMaterial(
        agent_name="claude-code",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.claude"),
                    options="rw,nosuid,nodev,noexec,mode=0700",
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath(".credentials.json"),
                        content=b'{"claudeAiOauth":{"accessToken":"secret"}}',
                    ),
                ),
            ),
        ),
        secret_values=frozenset({"secret"}),
    )


def _multi_pools() -> AllocatedPools:
    def node(host: str, subnet: int) -> AllocatedNode:
        return AllocatedNode(
            host=host,
            slots=4,
            ipv4=f"10.2.{subnet}.1",
            cuda_devices=tuple(f"GPU-{host}-{index}" for index in range(4)),
        )

    return AllocatedPools(
        run_id="native-run",
        work=(node("work-a", 0), node("work-b", 1)),
        verifier=(node("judge-a", 2),),
        gpus_per_node=4,
        digest="a" * 64,
    )


def test_workspace_seed_streams_archive_to_shared_storage(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
            agent_binary=Path(sys.executable),
            agent_launcher="codex",
            agent_companions=(),
            agent_auth=None,
            agent_version="0.149.0",
        ),
        make_run_plan(tmp_path),
    )
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_run(command, **options):
        calls.append((tuple(command), options))
        return AgentRunResult(exit_code=0, output="0.149.0\n")

    monkeypatch.setattr(runtime, "_run", fake_run)

    runtime.initialize()

    seed_command, seed_options = calls[0]
    shell = seed_command[2]
    assert "set -o pipefail" in shell
    assert "command -v tar" in shell
    assert "tar -C /workspace -cf - ." in shell
    assert "tar -C /run/rsi-harness/seed -xpf -" in shell
    assert "cp -a -- /workspace/. /run/rsi-harness/seed/" in shell
    assert seed_options["timeout_seconds"] == 1800
    assert seed_options["mount_workspace"] is False
    assert seed_options["extra_binds"] == (
        (runtime.workspace, PurePosixPath("/run/rsi-harness/seed"), False),
    )


def test_vlmr1_verifier_does_not_mutate_read_only_private_tests() -> None:
    task = (
        Path(__file__).resolve().parents[4]
        / "examples/harbor-task-agent/vlmr1-rec-curriculum"
    )
    verifier = task / "tests/test.sh"
    if not verifier.is_file():
        pytest.skip("optional external task is not included in this checkout")

    script = verifier.read_text()

    assert "chmod -R go-rwx /tests" not in script


def test_workspace_snapshot_uses_path_authority_and_native_artifacts(
    tmp_path: Path,
) -> None:
    run_id = "native-run"
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id=run_id)
    writer.start()
    runtime = _Runtime(tmp_path)
    observer = _Observer()
    work = ContainerRef(container_id="work", role="work")
    request = EvaluationRequest(
        run_plan=plan,
        work_container=work,
        round_id="agent-1",
        verifier_logs=(
            writer.root / "verifier" / "agent-1"
        ),
        verifier_output=writer.feedback_root / "agent-1.log",
    )

    report = NativeJudgeEvaluator(runtime, writer).evaluate(request, observer)

    planned = dict(observer.events)["snapshot_planned"]
    assert "planned_snapshot_ref" not in planned
    assert report.score == 1.0
    assert runtime.events == ["pause", "idle", "unpause"]
    assert (writer.root / "submissions/agent-1/report.json").is_file()
    assert (writer.root / "feedback/agent-1.log").read_text() == "passed\n"


def test_active_remote_work_makes_submission_retryable_before_snapshot(
    tmp_path: Path,
) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="native-run")
    writer.start()
    runtime = _Runtime(tmp_path)

    def active() -> None:
        runtime.events.append("idle")
        raise InfrastructureError("phase broker has active remote requests")

    runtime.require_work_idle = active  # type: ignore[method-assign]
    observer = _Observer()
    request = EvaluationRequest(
        run_plan=plan,
        work_container=ContainerRef(container_id="work", role="work"),
        round_id="agent-1",
        verifier_logs=writer.root / "verifier/agent-1",
        verifier_output=writer.feedback_root / "agent-1.log",
    )

    with pytest.raises(RetryableSubmissionError, match="active remote Work"):
        NativeJudgeEvaluator(runtime, writer).evaluate(request, observer)

    assert runtime.events == ["pause", "idle", "unpause"]
    assert not (runtime.rounds / "agent-1").exists()
    assert not (writer.root / "submissions/agent-1/report.json").exists()


def test_judge_gets_fresh_writable_tmp_without_mutating_preserved_assets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )
    runtime.task_assets.mkdir(parents=True)
    (runtime.task_assets / "test_patch.diff").write_text("original\n")
    (runtime.task_assets / "setup_patch.diff").touch()
    workspace = runtime.rounds / "agent-1" / "workspace"
    workspace.mkdir(parents=True)
    request = EvaluationRequest(
        run_plan=plan,
        work_container=ContainerRef(container_id="work", role="work"),
        round_id="agent-1",
        verifier_logs=tmp_path / "verifier",
        verifier_output=tmp_path / "feedback.log",
    )

    def fake_run(command, **options):
        del command
        tmp_binds = [
            bind
            for bind in options["extra_binds"]
            if bind[1] == PurePosixPath("/tmp")
        ]
        assert len(tmp_binds) == 1
        judge_tmp, _target, read_only = tmp_binds[0]
        assert judge_tmp.parent == node_tmp / "judge"
        assert read_only is False
        assert (judge_tmp / "test_patch.diff").read_text() == "original\n"
        assert (judge_tmp / "setup_patch.diff").is_file()
        assert (judge_tmp / "setup_patch.diff").stat().st_size == 0
        (judge_tmp / "test_patch.diff").write_text("judge mutation\n")
        (judge_tmp / "f2p_output.txt").write_text("passed\n")
        return AgentRunResult(exit_code=0, output="passed\n")

    monkeypatch.setattr(runtime, "_run", fake_run)

    result = runtime.run_judge(workspace, request, {})

    assert result.exit_code == 0
    assert (runtime.task_assets / "test_patch.diff").read_text() == "original\n"
    assert not any((node_tmp / "judge").iterdir())


def test_multinode_judge_uses_only_fresh_judge_pool_and_read_only_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from rsi_harness.cluster.bluevela import judge_controller

    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = profile.model_copy(
        update={
            "scheduler": profile.scheduler.model_copy(
                update={
                    "remote_binary": "/site/bin/remote-launch",
                    "remote_host_flag": "--host",
                }
            ),
            "apptainer": profile.apptainer.model_copy(
                update={"temp_root": node_tmp}
            ),
            "resources": profile.resources.model_copy(
                update={"gpus_per_node": 4}
            ),
        }
    )
    sif = tmp_path / "task.sif"
    checksum = tmp_path / "task.sif.sha256"
    sif.write_bytes(b"sif")
    checksum.write_text("d" * 64 + "  task.sif\n")
    plan = make_run_plan(tmp_path)
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=profile,
            run_id="native-run",
            sif_path=sif,
            sif_sha256_path=checksum,
        ),
        plan,
        allocated_pools=_multi_pools(),
    )
    runtime.control.mkdir(parents=True)
    runtime.rounds.mkdir(parents=True)
    runtime.judge_tmp_root.mkdir(parents=True)
    snapshot = runtime.rounds / "agent-1/workspace"
    snapshot.mkdir(parents=True)
    verifier_logs = tmp_path / "verifier"
    verifier_logs.mkdir()
    request = EvaluationRequest(
        run_plan=plan,
        work_container=runtime.work_ref,
        round_id="agent-1",
        verifier_logs=verifier_logs,
        verifier_output=tmp_path / "feedback.log",
    )
    observed: list[tuple[str, ...]] = []

    def controller(command, *, timeout_seconds, output_path):
        del timeout_seconds
        observed.append(command)
        assert runtime.judge_broker is not None
        assert tuple(node.host for node in runtime.judge_broker.nodes) == (
            "judge-a",
        )
        control_path = Path(command[command.index("--control") + 1])
        control = judge_controller.load_control(control_path)
        binds = {
            (item.source, item.target): item.read_only
            for item in control.worker.binds
        }
        assert binds[(snapshot, plan.workdir)] is True
        assert binds[(plan.task.source_dir / "tests", PurePosixPath("/tests"))] is True
        assert any("paloma" in str(source) for source, _target in binds)
        torchrun_sources = tuple(
            item.source
            for item in control.worker.binds
            if item.target == PurePosixPath("/usr/local/bin/torchrun")
        )
        assert len(torchrun_sources) == 1
        assert torchrun_sources[0].is_relative_to(runtime.control)
        assert torchrun_sources[0].stat().st_mode & 0o111 == 0o111
        assert control.worker.environment["HF_HOME"] == "/tmp/.cache/huggingface"
        assert (
            control.worker.environment["HF_DATASETS_CACHE"]
            == "/tmp/.cache/huggingface/datasets"
        )
        assert all("GPU-work" not in value for value in control.environment.values())
        (verifier_logs / "reward.txt").write_text("1\n")
        output_path.write_text("passed\n")
        return AgentRunResult(exit_code=0, output="passed\n")

    monkeypatch.setattr(runtime, "_run_host_controller", controller)

    result = runtime.run_judge(snapshot, request, {})

    assert result.exit_code == 0
    assert observed[0][:3] == (
        "/site/bin/remote-launch",
        "--host",
        "judge-a",
    )
    assert "work-a" not in " ".join(observed[0])
    assert runtime.judge_broker is None


def test_apptainer_commands_use_run_plan_workdir(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )

    command = runtime._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        containall=True,
        network_mode="public",
    )

    cwd_index = command.index("--cwd")
    assert command[cwd_index + 1] == "/workspace"
    assert (
        "--bind",
        f"{node_tmp / 'work'}:/tmp",
    ) in tuple(zip(command, command[1:], strict=False))
    environment = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for option, value in zip(command, command[1:], strict=False)
        if option == "--env"
    }
    assert environment["HF_HOME"] == "/rsi-cache/huggingface"
    assert environment["HF_DATASETS_CACHE"] == "/rsi-cache/datasets"
    assert environment["RSI_SHARED_DATA_ROOT"] == runtime.profile.apptainer.environment[
        "RSI_SHARED_DATA_ROOT"
    ]
    assert environment["TMPDIR"] == "/tmp"
    assert environment["XDG_CACHE_HOME"] == "/tmp/.cache"
    assert environment["TRITON_CACHE_DIR"] == "/tmp/.cache/triton"
    bind_values = tuple(
        value
        for option, value in zip(command, command[1:], strict=False)
        if option == "--bind"
    )
    assert "/proj:/proj" not in bind_values
    assert any("/rsi-data" in value for value in bind_values)
    assert not any("paloma" in value for value in bind_values)
    assert (
        f"{runtime.profile.storage.hf_home}:/rsi-cache/huggingface"
    ) in bind_values
    assert (
        f"{runtime.profile.storage.hf_datasets_cache}:/rsi-cache/datasets"
    ) in bind_values

    judge_command = runtime._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        containall=True,
        network_mode="no-network",
        phase="judge",
    )
    judge_bind_values = tuple(
        value
        for option, value in zip(judge_command, judge_command[1:], strict=False)
        if option == "--bind"
    )
    assert any(
        "paloma" in value and value.endswith(":ro") for value in judge_bind_values
    )
    assert not any("/rsi-cache" in value for value in judge_bind_values)


@pytest.mark.parametrize("network_mode", ("public", "no-network"))
@pytest.mark.parametrize("phase", ("work", "judge"))
def test_legacy_single_node_retains_mounts_cache_paths_and_environment(
    tmp_path: Path, monkeypatch, network_mode: str, phase: str,
) -> None:
    node_tmp = tmp_path / "node-tmp"
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    base = load_cluster_profile("bluevela")
    # This models an existing site file, which has no mount_policy field.
    data = base.model_dump()
    data["apptainer"].pop("mount_policy")
    data["apptainer"]["extra_binds"] = (Path("/shared/project"),)
    data["apptainer"]["environment"] = {"SITE_DATA": "/shared/project/data"}
    data["apptainer"]["work_environment"] = {"SITE_DATA": "/work-data"}
    data["apptainer"]["judge_environment"] = {"SITE_DATA": "/judge-data"}
    profile = type(base).model_validate(data)
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=profile, run_id="legacy", sif_path=tmp_path / "task.sif"
        ),
        make_run_plan(tmp_path),
    )

    command = runtime._base_command(
        devices=("GPU-fixture",),
        environment={"TASK_SETTING": "preserved"},
        extra_binds=(),
        mount_workspace=True,
        mount_agent_home=True,
        containall=True,
        network_mode=network_mode,
        phase=phase,
    )
    pairs = tuple(zip(command, command[1:], strict=False))
    mounts = {value for option, value in pairs if option == "--bind"}
    expected = {
        f"{runtime.work_tmp}:/tmp",
        f"{runtime.workspace}:/workspace",
        f"{runtime.feedback}:/run/rsi-harness/feedback:ro",
        f"{runtime.agent_home}:/home/agent",
    }
    if network_mode == "public":
        expected.add("/shared/project:/shared/project")
    assert mounts == expected
    environment = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for option, value in pairs if option == "--env"
    }
    assert environment["SITE_DATA"] == "/shared/project/data"
    assert environment["TASK_SETTING"] == "preserved"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
    assert environment["HF_HOME"] == (
        str(profile.storage.hf_home) if network_mode == "public"
        else "/tmp/.cache/huggingface"
    )
    assert environment["HF_DATASETS_CACHE"] == (
        str(profile.storage.hf_datasets_cache) if network_mode == "public"
        else "/tmp/.cache/huggingface/datasets"
    )


def test_judge_authority_is_multinode_only_and_never_exposed_to_work(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    authority_root = tmp_path / "authority"
    task_authority = authority_root / plan.task.task_id
    task_authority.mkdir(parents=True)
    (task_authority / "BASELINE_CERTIFIED.json").write_text("{}\n")
    base = load_cluster_profile("bluevela")
    profile = base.model_copy(
        update={
            "apptainer": base.apptainer.model_copy(
                update={"judge_authority_root": authority_root}
            ),
            "resources": base.resources.model_copy(update={"gpus_per_node": 4}),
        }
    )
    payload = SimpleNamespace(
        profile=profile,
        run_id="native-run",
        sif_path=tmp_path / "task.sif",
    )

    single = ApptainerAgentRuntime(payload, plan)
    single_judge = single._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        containall=True,
        network_mode="no-network",
        phase="judge",
    )
    assert not any("/run-contract" in item for item in single_judge)

    multi = ApptainerAgentRuntime(payload, plan, allocated_pools=_multi_pools())
    work = multi._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        containall=True,
        network_mode="no-network",
        phase="work",
    )
    judge = multi._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        containall=True,
        network_mode="no-network",
        phase="judge",
    )
    assert not any("/run-contract" in item for item in work)
    assert f"{task_authority}:/run-contract:ro" in judge


def test_multinode_work_runtime_injects_only_current_phase_broker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={"temp_root": node_tmp}
            ),
            "resources": profile.resources.model_copy(
                update={"gpus_per_node": 4}
            ),
        }
    )
    sif = tmp_path / "task.sif"
    checksum = tmp_path / "task.sif.sha256"
    sif.write_bytes(b"sif")
    checksum.write_text("d" * 64 + "  task.sif\n")
    payload = SimpleNamespace(
        profile=profile,
        run_id="native-run",
        sif_path=sif,
        sif_sha256_path=checksum,
        agent_binary=Path(sys.executable),
        agent_launcher="codex",
        agent_companions=(),
        agent_auth=None,
        agent_version="0.149.0",
    )
    runtime = ApptainerAgentRuntime(
        payload,
        make_run_plan(tmp_path),
        allocated_pools=_multi_pools(),
    )

    monkeypatch.setattr(
        runtime,
        "_run",
        lambda *_args, **_kwargs: AgentRunResult(exit_code=0, output="0.149.0"),
    )
    runtime.initialize()
    try:
        assert runtime.work_devices == tuple(
            f"GPU-work-a-{index}" for index in range(4)
        )
        assert runtime.work_broker is not None
        assert runtime.work_broker.worker_template.environment[
            "RSI_SHARED_DATA_ROOT"
        ] == profile.apptainer.work_environment["RSI_SHARED_DATA_ROOT"]
        assert runtime.work_broker.worker_template.environment["HF_HOME"] == (
            "/tmp/.cache/huggingface"
        )
        assert runtime.work_broker.worker_template.environment[
            "HF_DATASETS_CACHE"
        ] == "/tmp/.cache/huggingface/datasets"
        assert not any(
            "paloma" in str(binding.source)
            for binding in runtime.work_broker.worker_template.binds
        )
        assert not any(
            binding.source == Path("/proj")
            for binding in runtime.work_broker.worker_template.binds
        )
        assert (runtime.work_broker.root / "READY.json").is_file()
        shim = runtime._copies[PurePosixPath("/usr/local/bin/torchrun")]
        assert shim.is_relative_to(node_tmp)
        assert shim.stat().st_mode & 0o111 == 0o111
        assert shim.read_text().startswith("#!/usr/bin/env python3\n")
        command = runtime._base_command(
            devices=runtime.work_devices,
            environment=None,
            extra_binds=(),
            mount_workspace=True,
            mount_agent_home=True,
            containall=True,
            network_mode="public",
        )
        pairs = tuple(zip(command, command[1:], strict=False))
        assert (
            "--bind",
            f"{runtime.work_broker.root}:/run/rsi-harness/torchrun",
        ) in pairs
        environment = {
            value.split("=", 1)[0]: value.split("=", 1)[1]
            for option, value in pairs
            if option == "--env"
        }
        assert environment["RSI_MULTINODE_ROOT"] == (
            "/run/rsi-harness/torchrun"
        )
        assert environment["RSI_LOCAL_WORLD_SIZE"] == "4"
        assert environment["PREPEND_PATH"] == "/usr/local/bin"
        serialized = " ".join(command)
        assert "LSB_MCPU_HOSTS" not in serialized
        assert "judge-a" not in serialized
        assert "blaunch" not in serialized
    finally:
        runtime.close_multinode()


def test_claude_runtime_mounts_launcher_and_auth_only_from_node_local_storage(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.resolve_agent_auth",
        lambda **_options: _claude_auth(),
    )
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
            agent_binary=Path(sys.executable),
            agent_launcher="claude",
            agent_companions=(),
            agent_auth=AgentAuthSource.LOCAL,
            agent_version="2.1.246",
        ),
        _claude_plan(tmp_path),
    )
    commands: list[tuple[str, ...]] = []

    def fake_run(command, **_options):
        commands.append(tuple(command))
        return AgentRunResult(exit_code=0, output="2.1.246 (Claude Code)\n")

    monkeypatch.setattr(runtime, "_run", fake_run)

    runtime.initialize()

    command = runtime._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=True,
        mount_agent_home=True,
        containall=True,
        network_mode="public",
    )
    bind_values = [
        value
        for option, value in zip(command, command[1:], strict=False)
        if option == "--bind"
    ]
    assert f"{sys.executable}:/usr/local/bin/claude:ro" in bind_values
    auth_bind = next(
        value for value in bind_values if value.endswith(":/home/agent/.claude")
    )
    auth_root = Path(auth_bind.split(":", 1)[0])
    assert auth_root.is_relative_to(node_tmp)
    assert (auth_root / ".credentials.json").read_bytes() == (
        b'{"claudeAiOauth":{"accessToken":"secret"}}'
    )
    assert (auth_root / ".credentials.json").stat().st_mode & 0o777 == 0o600
    assert not any(
        b"secret" in path.read_bytes()
        for path in runtime.plan.paths.root.rglob("*")
        if path.is_file()
    )
    assert ("/usr/local/bin/claude", "--version") in commands


def test_claude_hook_settings_share_writable_node_local_auth_mount(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.resolve_agent_auth",
        lambda **_options: _claude_auth(),
    )
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
            agent_binary=Path(sys.executable),
            agent_launcher="claude",
            agent_companions=(),
            agent_auth=AgentAuthSource.LOCAL,
            agent_version="2.1.246",
        ),
        _claude_plan(tmp_path),
    )
    runtime.agent_home.mkdir(parents=True)
    runtime._install_auth()
    settings = tmp_path / "settings.json"
    settings.write_text('{"hooks":{}}')

    runtime.copy_to(
        runtime.work_ref,
        settings,
        PurePosixPath("/home/agent/.claude/settings.json"),
    )

    auth_root = next(
        source
        for source, target, _read_only in runtime._agent_auth_binds
        if target == PurePosixPath("/home/agent/.claude")
    )
    assert (auth_root / "settings.json").read_text() == '{"hooks":{}}'


def test_claude_apptainer_environment_uses_private_config_directory(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        _claude_plan(tmp_path),
    )

    command = runtime._base_command(
        devices=(),
        environment=None,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        containall=True,
        network_mode="public",
    )
    environment = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for option, value in zip(command, command[1:], strict=False)
        if option == "--env"
    }

    assert environment["CLAUDE_CONFIG_DIR"] == "/home/agent/.claude"


def test_apptainer_network_policy_is_enforced_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    profile = load_cluster_profile("bluevela")
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=profile,
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        make_run_plan(tmp_path),
    )
    options = {
        "devices": (),
        "environment": None,
        "extra_binds": (),
        "mount_workspace": False,
        "mount_agent_home": False,
        "containall": True,
    }

    isolated = runtime._base_command(**options, network_mode="no-network")
    public = runtime._base_command(**options, network_mode="public")

    assert (
        "--net",
        "--network",
        "none",
        "--hostname",
        "localhost",
    ) == isolated[2:7]
    assert "--net" not in public
    assert "--network" not in public
    assert "--hostname" not in public
    isolated_pairs = tuple(zip(isolated, isolated[1:], strict=False))
    public_pairs = tuple(zip(public, public[1:], strict=False))
    assert ("--bind", "/proj:/proj") not in isolated_pairs
    assert ("--bind", "/proj:/proj") not in public_pairs
    isolated_environment = {
        value.split("=", 1)[0]: value.split("=", 1)[1]
        for option, value in isolated_pairs
        if option == "--env"
    }
    assert isolated_environment["HF_HOME"] == "/tmp/.cache/huggingface"
    assert (
        isolated_environment["HF_DATASETS_CACHE"]
        == "/tmp/.cache/huggingface/datasets"
    )
    with pytest.raises(SetupError, match="allowlist"):
        runtime._base_command(**options, network_mode="allowlist")


def test_no_network_agent_command_uses_private_provider_and_submit_relays(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        make_run_plan(tmp_path),
    )
    runtime.configure_agent_network(
        "http://127.0.0.1:43123",
        ("https://localhost:443",),
    )
    try:
        base = runtime._base_command(
            devices=(),
            environment=None,
            extra_binds=(),
            mount_workspace=True,
            mount_agent_home=True,
            containall=True,
            network_mode="no-network",
        )
        assert (
            "--bind",
            f"{node_tmp / 'network'}:/run/rsi-harness/network:ro",
        ) in tuple(zip(base, base[1:], strict=False))
        assert runtime._network_command(
            ("codex", "exec"),
            network_mode="no-network",
            mount_agent_home=True,
        )[:10] == (
            "/usr/bin/env",
            "python3",
            "/usr/local/bin/rsi-netns-relay",
            "--provider-socket",
            "/run/rsi-harness/network/provider.sock",
            "--submit-socket",
            "/run/rsi-harness/network/submit.sock",
            "--submit-port",
            "43123",
            "--",
        )
    finally:
        runtime.close_agent_network()


def test_local_codex_auth_supplies_exact_provider_urls_for_no_network_agent(
    tmp_path: Path,
) -> None:
    from rsi_harness.cluster.bluevela.runtime import _agent_provider_urls

    def resolve(**_options):
        return AgentAuthMaterial(
            agent_name="codex",
            mounts=(),
            secret_values=frozenset(),
            provider_endpoints=(
                "https://chatgpt.com/backend-api/codex",
                "https://auth.openai.com",
            ),
        )

    urls = _agent_provider_urls(
        SimpleNamespace(agent_auth=AgentAuthSource.LOCAL),
        make_run_plan(tmp_path),
        SimpleNamespace(
            agent_api_key=None,
            agent_api_base_url=None,
            http_proxy=None,
            https_proxy=None,
        ),
        auth_resolver=resolve,
    )

    assert urls == (
        "https://chatgpt.com/backend-api/codex",
        "https://auth.openai.com",
    )


def test_native_hook_install_starts_no_network_agent_broker(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )
    composition = object.__new__(NativeEngineComposition)
    composition.runtime = runtime
    composition.provider_urls = ("https://localhost:443",)
    composition.agent = SimpleNamespace(install_hooks=lambda _request: None)
    composition.secrets = set()

    composition.install_hooks(
        plan,
        runtime.work_ref,
        "http://127.0.0.1:43123",
        "token",
    )
    try:
        assert (node_tmp / "network/provider.sock").is_socket()
        assert (node_tmp / "network/submit.sock").is_socket()
    finally:
        runtime.close_agent_network()
