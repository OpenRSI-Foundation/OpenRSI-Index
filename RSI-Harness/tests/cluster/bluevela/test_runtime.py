from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
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
    SubmissionError,
)
from rsi_harness.models import (
    AgentAuthSource,
    AgentRunResult,
    ContainerRef,
    ContainerTmpfs,
    EvaluationRequest,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    RunGPUPlan,
    RunStatus,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.coordinator import (
    CoordinatorState,
    RunCoordinator,
    _RoundEvaluator,
    aggregate_result,
)
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
)
from rsi_harness.runtime.submissions import SubmissionService
from tests.factories import make_run_plan
from tests.fakes import FakeClock

# The last line of a finished isolation probe (initialize's last exec).
PROBE_END = "isolation probe end\n"


class _Runtime:
    def __init__(
        self,
        root: Path,
        *,
        reward_file="reward.txt",
        reward_content="1\n",
    ) -> None:
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "answer.py").write_text("answer = 42\n")
        self.rounds = root / "rounds"
        self.rounds.mkdir()
        self.payload = SimpleNamespace(run_id="native-run")
        self.events: list[str] = []
        self.reward_file = reward_file
        self.reward_content = reward_content

    def renew_operator_login(self, _round_id: str) -> None:
        pass

    def pause(self, _work: ContainerRef) -> None:
        self.events.append("pause")

    def unpause(self, _work: ContainerRef) -> None:
        self.events.append("unpause")

    def require_work_idle(self) -> None:
        self.events.append("idle")

    def run_judge(self, workspace, request, environment, *, trajectory=None):
        del workspace, environment
        self.trajectory = trajectory
        self.trajectory_files = (
            None
            if trajectory is None
            else sorted(path.name for path in trajectory.iterdir())
        )
        (request.verifier_logs / self.reward_file).write_text(self.reward_content)
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
        return AgentRunResult(exit_code=0, output="0.149.0\n" + PROBE_END)

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


@pytest.mark.parametrize(
    ("reward_file", "reward_content"),
    [
        ("reward.txt", "1\n"),
        ("reward.json", '{"reward": 1, "accuracy": 0.75}\n'),
    ],
)
def test_workspace_snapshot_uses_path_authority_and_native_artifacts(
    tmp_path: Path,
    reward_file,
    reward_content,
) -> None:
    run_id = "native-run"
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id=run_id)
    writer.start()
    runtime = _Runtime(
        tmp_path,
        reward_file=reward_file,
        reward_content=reward_content,
    )
    observer = _Observer()
    work = ContainerRef(container_id="work", role="work")
    request = EvaluationRequest(
        run_plan=plan,
        work_container=work,
        round_id="agent-1",
        verifier_logs=(writer.root / "verifier" / "agent-1"),
        verifier_output=writer.feedback_root / "agent-1.log",
    )

    report = NativeJudgeEvaluator(runtime, writer).evaluate(request, observer)

    planned = dict(observer.events)["snapshot_planned"]
    assert "planned_snapshot_ref" not in planned
    assert report.score == 1.0
    assert runtime.events == ["pause", "idle", "unpause"]
    assert (writer.root / "submissions/agent-1/report.json").is_file()
    assert (writer.root / "feedback/agent-1.log").read_text() == "passed\n"
    assert (request.verifier_logs / reward_file).read_text() == reward_content
    if reward_file == "reward.json":
        assert dict(report.rewards) == {"reward": 1.0, "accuracy": 0.75}
    # The Judge read this round's trajectory copy; only its manifest remains.
    assert runtime.trajectory == writer.root / "judge-trajectory" / "agent-1"
    assert runtime.trajectory_files == ["agent_output.txt", "manifest.json"]
    assert sorted(path.name for path in runtime.trajectory.iterdir()) == [
        "manifest.json"
    ]


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


def _single_node_gpu_runtime(
    tmp_path,
    monkeypatch,
    *,
    mode=JudgeGPUMode.RELEASE_ALL,
    selector="2",
):
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(tmp_path / "node-tmp"))
    plan = make_run_plan(tmp_path)
    work = GPUDevice(index=0, uuid=selector, name="allocated GPU")
    spare = GPUDevice(index=1, uuid="3", name="allocated GPU")
    gpu_plan = RunGPUPlan(
        authorized_pool=GPUAllocation(devices=(work, spare)),
        work=GPUAllocation(devices=(work,)),
        judge=GPUAllocation(
            devices=(
                ()
                if mode is JudgeGPUMode.FREEZE_ONLY
                else (spare,)
                if mode is JudgeGPUMode.DISJOINT
                else (work,)
            )
        ),
        judge_mode=mode,
    )
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan.model_copy(update={"gpu_plan": gpu_plan}),
    )
    runtime._process = SimpleNamespace(pid=100, poll=lambda: None)
    return runtime


@pytest.mark.parametrize("agent_exit", ["running", "before_resume", "during_resume"])
def test_final_native_judge_is_registered_after_agent_exit(
    tmp_path: Path,
    monkeypatch,
    agent_exit,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    runtime.workspace.mkdir()
    (runtime.workspace / "answer.py").write_text("answer = 42\n")
    monkeypatch.setattr(runtime, "require_work_idle", lambda: None)
    writer = RunArtifactWriter(runtime.plan, run_id="native-run")
    writer.start()
    judge_rounds = 0

    def judge(_snapshot, request, _environment, *, trajectory=None):
        nonlocal judge_rounds
        judge_rounds += 1
        (request.verifier_logs / "reward.json").write_text(
            json.dumps({"reward": judge_rounds}) + "\n"
        )
        request.verifier_output.write_text("passed\n")
        if judge_rounds == 2 and agent_exit == "before_resume":
            runtime._process = SimpleNamespace(pid=100, poll=lambda: 0)
        return AgentRunResult(exit_code=0, output="passed\n", full_output_captured=True)

    def signal_group(_pid, signum):
        if (
            judge_rounds == 2
            and agent_exit == "during_resume"
            and signum == signal.SIGCONT
        ):
            raise ProcessLookupError("Agent exited before SIGCONT")

    monkeypatch.setattr(runtime, "run_judge", judge)
    monkeypatch.setattr("os.killpg", signal_group)
    native = NativeJudgeEvaluator(runtime, writer)
    state = CoordinatorState(run_id="native-run")
    state.transition(RunStatus.AGENT_RUNNING)
    evaluator = _RoundEvaluator(
        backend=SimpleNamespace(
            evaluate_submission=lambda request, *, lifecycle_observer: native.evaluate(
                request, lifecycle_observer
            ),
        ),
        state=state,
        transition=state.transition,
        persist_recovery=lambda _message: None,
        update_work=lambda **_values: None,
        update_judge=lambda **_values: None,
    )
    service = SubmissionService(
        evaluator=evaluator,
        artifact_writer=writer,
        clock=FakeClock(),
    )
    token = service.register(
        run_id="native-run",
        run_plan=runtime.plan,
        work_container=runtime.work_ref,
    )

    service.submit(token)
    service.submit(token)
    service.close(timeout_seconds=0)
    writer.finalize()

    history = RunCoordinator._history(SimpleNamespace(service=service), evaluator)
    assert len(history) == len(evaluator.reports) == len(service.reports) == 2
    assert history[-1].status is SubmissionStatus.COMPLETED
    assert history[-1].score == 2.0
    assert runtime._paused is False
    final = json.loads((writer.root / "final_result.json").read_text())
    engine = aggregate_result(
        run_id="native-run",
        reports=history,
        primary_reward=runtime.plan.task.verifier.primary_reward,
        direction=runtime.plan.task.score_direction,
        terminal=RunStatus.COMPLETED,
    )
    assert final["total_rounds"] == engine.total_rounds == 2
    assert final["best_round"] == engine.best_round == "agent-2"
    assert final["best_score"] == engine.best_score == 2.0
    assert json.loads((writer.root / "verifier/agent-2/reward.json").read_text()) == {
        "reward": 2
    }


def test_native_unpause_does_not_hide_signal_permission_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    runtime._paused = True

    def denied(_pid, _signum):
        raise PermissionError("signal denied")

    monkeypatch.setattr("os.killpg", denied)

    with pytest.raises(PermissionError, match="signal denied"):
        runtime.unpause(runtime.work_ref)

    assert runtime._paused is True


def _process_state(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return stat.rsplit(")", 1)[1].split()[0]


def test_unpause_and_stop_reach_members_left_after_the_agent_exits(
    tmp_path: Path, monkeypatch
) -> None:
    # The race in pause(): the Agent exits after the liveness check, so
    # SIGSTOP only reaches what it left behind in its process group.
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    leader = subprocess.Popen(
        ["sh", "-c", "sleep 300 & echo $!"],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    child = int(leader.stdout.readline())
    try:
        leader.wait(timeout=10)
        os.killpg(leader.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 10
        while _process_state(child) != "T":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        runtime._process = leader
        runtime._paused = True

        runtime.unpause(runtime.work_ref)
        deadline = time.monotonic() + 10
        while _process_state(child) == "T":
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert runtime._paused is False

        os.killpg(leader.pid, signal.SIGSTOP)
        runtime._paused = True
        runtime.stop(runtime.work_ref)
        deadline = time.monotonic() + 10
        while _process_state(child) not in (None, "Z"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        # Once the group is gone, both calls are no-ops.
        runtime.unpause(runtime.work_ref)
        runtime.stop(runtime.work_ref)
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _gpu_preflight_commands(
    monkeypatch,
    *,
    compute="",
    graphics_pid=999,
    ps=None,
    failure=None,
    inventory="2, GPU-owned, NVIDIA H100\n3, GPU-other, NVIDIA H100\n",
):
    # Child 102 is a grandchild; 103 has a separate session; 104 is reparented
    # but still belongs to Work's dedicated session. PID 999 is unrelated.
    process_table = (
        ps
        if ps is not None
        else (
            "100 1 100\n101 100 100\n102 101 100\n103 101 103\n104 1 100\n999 1 999\n"
        )
    )
    responses = {
        ("ps", "-eo", "pid=,ppid=,sid="): process_table,
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ): inventory,
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ): compute,
        (
            "nvidia-smi",
            "-q",
            "-x",
        ): "<nvidia_smi_log><gpu><uuid>GPU-owned</uuid><processes>"
        f"<process_info><pid>{graphics_pid}</pid><process_type>G</process_type>"
        "</process_info></processes></gpu></nvidia_smi_log>",
    }

    def run(command, **options):
        assert 0 < options["timeout"] <= 30
        if failure is not None:
            raise failure
        return subprocess.CompletedProcess(command, 0, responses[tuple(command)], "")

    monkeypatch.setattr(subprocess, "run", run)


@pytest.mark.parametrize(
    ("pid", "selector"),
    [(100, "2"), (101, "2"), (102, "2"), (103, "2"), (104, "2"), (102, "GPU-owned")],
)
def test_release_all_rejects_work_processes_and_descendants(
    tmp_path: Path,
    monkeypatch,
    pid: int,
    selector: str,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch, selector=selector)
    _gpu_preflight_commands(monkeypatch, compute=f"GPU-owned, {pid}\n")

    with pytest.raises(SubmissionError, match=f"PID {pid}"):
        runtime.require_work_idle()


def test_release_all_rejects_work_graphics_process(tmp_path: Path, monkeypatch) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    _gpu_preflight_commands(monkeypatch, graphics_pid=102)

    with pytest.raises(SubmissionError, match="PID 102"):
        runtime.require_work_idle()


def test_release_all_ignores_unrelated_processes_and_unallocated_gpus(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    _gpu_preflight_commands(monkeypatch, compute="GPU-owned, 999\nGPU-other, 102\n")

    runtime.require_work_idle()


@pytest.mark.parametrize("mode", [JudgeGPUMode.FREEZE_ONLY, JudgeGPUMode.DISJOINT])
def test_nonsharing_modes_do_not_query_work_gpu_processes(
    tmp_path: Path,
    monkeypatch,
    mode,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch, mode=mode)

    def unexpected_query(*_args, **_kwargs):
        pytest.fail("GPU release is not required in this mode")

    monkeypatch.setattr(subprocess, "run", unexpected_query)
    runtime.require_work_idle()


def test_cpu_work_does_not_query_gpu_processes(tmp_path: Path, monkeypatch) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    runtime.plan = make_run_plan(tmp_path)

    def unexpected_query(*_args, **_kwargs):
        pytest.fail("CPU-only Work must not require nvidia-smi")

    monkeypatch.setattr(subprocess, "run", unexpected_query)
    runtime.require_work_idle()


@pytest.mark.parametrize(
    ("ps", "failure"),
    [
        ("999 1 999\n", None),
        ("invalid process table\n", None),
        (None, subprocess.TimeoutExpired("ps", 15)),
        (None, FileNotFoundError("nvidia-smi unavailable")),
    ],
)
def test_unproven_work_gpu_release_rejects_submission(
    tmp_path: Path,
    monkeypatch,
    ps,
    failure,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    _gpu_preflight_commands(monkeypatch, ps=ps, failure=failure)

    with pytest.raises(SubmissionError):
        runtime.require_work_idle()


@pytest.mark.parametrize("inventory", ["malformed\n", "3, GPU-other, NVIDIA H100\n"])
def test_unresolved_work_gpu_allocation_rejects_submission(
    tmp_path: Path,
    monkeypatch,
    inventory,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    _gpu_preflight_commands(monkeypatch, inventory=inventory)

    with pytest.raises(SubmissionError, match="cannot resolve Work GPU allocation"):
        runtime.require_work_idle()


def test_busy_single_node_submission_is_retryable_before_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    runtime.workspace.mkdir()
    runtime.task_assets.mkdir(parents=True)
    _gpu_preflight_commands(monkeypatch, compute="GPU-owned, 102\n")

    def unexpected_judge(*_args, **_kwargs):
        pytest.fail("Judge launched before Work GPU release")

    monkeypatch.setattr(subprocess, "Popen", unexpected_judge)
    signals = []
    monkeypatch.setattr("os.killpg", lambda pid, signum: signals.append((pid, signum)))
    writer = RunArtifactWriter(runtime.plan, run_id="native-run")
    writer.start()
    request = EvaluationRequest(
        run_plan=runtime.plan,
        work_container=runtime.work_ref,
        round_id="agent-1",
        verifier_logs=writer.root / "verifier/agent-1",
        verifier_output=writer.feedback_root / "agent-1.log",
    )
    observer = _Observer()

    with pytest.raises(RetryableSubmissionError, match="PID 102"):
        NativeJudgeEvaluator(runtime, writer).evaluate(request, observer)

    assert signals == [(100, signal.SIGSTOP), (100, signal.SIGCONT)]
    assert not (runtime.rounds / "agent-1").exists()
    assert not request.verifier_logs.exists()
    assert "judge_planned" not in dict(observer.events)
    assert not (writer.root / "submissions/agent-1/report.json").exists()


@pytest.mark.parametrize("phase", ["work", "judge"])
def test_task_runtime_resets_private_controller_umask(
    tmp_path: Path,
    monkeypatch,
    phase,
) -> None:
    runtime = _single_node_gpu_runtime(tmp_path, monkeypatch)
    command = runtime._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="public",
        phase=phase,
    )

    assert "--no-umask" in command


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
            bind for bind in options["extra_binds"] if bind[1] == PurePosixPath("/tmp")
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
        assert (trajectory, PurePosixPath("/logs/agent"), True) in options[
            "extra_binds"
        ]
        # Every Judge: rsi-isolate, read-only.
        isolate = runtime.control / "judge-isolate"
        assert (isolate, PurePosixPath("/opt/rsi-isolate"), True) in options[
            "extra_binds"
        ]
        assert (
            isolate / "rsi-isolate",
            PurePosixPath("/usr/local/bin/rsi-isolate"),
            True,
        ) in options["extra_binds"]
        assert (isolate / "rsi_isolate.py").is_file()
        assert (isolate / "judge_isolation.py").is_file()
        return AgentRunResult(exit_code=0, output="passed\n", full_output_captured=True)

    monkeypatch.setattr(runtime, "_run", fake_run)
    trajectory = tmp_path / "judge-trajectory" / "agent-1"
    trajectory.mkdir(parents=True)

    result = runtime.run_judge(workspace, request, {}, trajectory=trajectory)

    assert result.exit_code == 0
    assert (runtime.task_assets / "test_patch.diff").read_text() == "original\n"
    assert not any((node_tmp / "judge").iterdir())


@pytest.mark.parametrize(
    "cluster_name,agent_name",
    [("bluevela", "codex"), ("slurm", "codex"), ("slurm", "oracle")],
)
@pytest.mark.parametrize("allow_signal_compatibility", [False, True])
def test_multinode_judge_uses_only_fresh_judge_pool_and_read_only_snapshot(
    tmp_path: Path,
    monkeypatch,
    cluster_name,
    agent_name,
    allow_signal_compatibility,
) -> None:
    from rsi_harness.cluster.bluevela import judge_controller

    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={
                    "judge_allow_missing_landlock_signal": allow_signal_compatibility,
                }
            ),
            "adapter": cluster_name,
            "scheduler": load_cluster_profile(cluster_name).scheduler,
        }
    )
    profile = profile.model_copy(
        update={
            "scheduler": profile.scheduler.model_copy(
                update={
                    "remote_binary": "/site/bin/remote-launch",
                    "remote_host_flag": "--host",
                }
            ),
            "apptainer": profile.apptainer.model_copy(update={"temp_root": node_tmp}),
            "resources": profile.resources.model_copy(update={"gpus_per_node": 4}),
        }
    )
    sif = tmp_path / "task.sif"
    checksum = tmp_path / "task.sif.sha256"
    sif.write_bytes(b"sif")
    checksum.write_text("d" * 64 + "  task.sif\n")
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(update={"name": agent_name})
                }
            )
        }
    )
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
    if agent_name == "oracle":
        runtime._oracle_solution = runtime.control / "oracle-solution"
        runtime._oracle_solution.mkdir()
        (runtime._oracle_solution / "solve.sh").write_text("baseline\n")
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
        assert runtime.judge_broker.remote_args == profile.scheduler.remote_args
        assert tuple(node.host for node in runtime.judge_broker.nodes) == ("judge-a",)
        control_path = Path(command[command.index("--control") + 1])
        control = judge_controller.load_control(control_path)
        binds = {
            (item.source, item.target): item.read_only for item in control.worker.binds
        }
        assert binds[(snapshot, plan.workdir)] is True
        assert binds[(plan.task.source_dir / "tests", PurePosixPath("/tests"))] is True
        assert binds[(trajectory, PurePosixPath("/logs/agent"))] is True
        assert not any(
            target == PurePosixPath("/solution") or source == runtime._oracle_solution
            for source, target in binds
        )
        isolate = runtime.control / "judge-isolate"
        assert binds[(isolate, PurePosixPath("/opt/rsi-isolate"))] is True
        assert (
            binds[
                (isolate / "rsi-isolate", PurePosixPath("/usr/local/bin/rsi-isolate"))
            ]
            is True
        )
        assert any("paloma" in str(source) for source, _target in binds)
        # Every Judge node gets the operator's isolation fallback, explicitly.
        assert control.worker.environment["RSI_JUDGE_ISOLATION_FALLBACK"] == "none"
        assert control.environment["RSI_JUDGE_ISOLATION_FALLBACK"] == "none"
        # Only the head controller runs with --net; remote ranks share the
        # host's network namespace.
        assert control.worker.environment["RSI_JUDGE_NETNS"] == "shared"
        assert control.environment["RSI_JUDGE_NETNS"] == "private"
        signal_env = "RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL"
        expected = "1" if allow_signal_compatibility else "0"
        assert control.worker.environment[signal_env] == expected
        assert control.environment[signal_env] == expected
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
        return AgentRunResult(exit_code=0, output="passed\n", full_output_captured=True)

    monkeypatch.setattr(runtime, "_run_host_controller", controller)
    trajectory = tmp_path / "judge-trajectory" / "agent-1"
    trajectory.mkdir(parents=True)

    result = runtime.run_judge(snapshot, request, {}, trajectory=trajectory)

    assert result.exit_code == 0
    prefix_length = 3 + len(profile.scheduler.remote_args)
    assert observed[0][:prefix_length] == (
        "/site/bin/remote-launch",
        *profile.scheduler.remote_args,
        "--host",
        "judge-a",
    )
    assert "work-a" not in " ".join(observed[0])
    assert runtime.judge_broker is None


def test_apptainer_commands_use_run_plan_workdir(tmp_path: Path, monkeypatch) -> None:
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
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="public",
    )

    cwd_index = command.index("--cwd")
    assert command[cwd_index + 1] == "/workspace"
    assert (
        "--bind",
        f"{node_tmp / 'work'}:/tmp",
    ) in tuple(zip(command, command[1:], strict=False))
    environment = runtime._container_environment(
        devices=(), environment=None, mount_agent_home=False, declared="public"
    )
    assert environment["HF_HOME"] == "/rsi-cache/huggingface"
    assert environment["HF_DATASETS_CACHE"] == "/rsi-cache/datasets"
    assert (
        environment["RSI_SHARED_DATA_ROOT"]
        == runtime.profile.apptainer.environment["RSI_SHARED_DATA_ROOT"]
    )
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
    assert (f"{runtime.profile.storage.hf_home}:/rsi-cache/huggingface") in bind_values
    assert (
        f"{runtime.profile.storage.hf_datasets_cache}:/rsi-cache/datasets"
    ) in bind_values

    judge_command = runtime._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
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
    tmp_path: Path,
    monkeypatch,
    network_mode: str,
    phase: str,
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
        extra_binds=(),
        mount_workspace=True,
        mount_agent_home=True,
        network_mode=network_mode,
        phase=phase,
    )
    pairs = tuple(zip(command, command[1:], strict=False))
    mounts = {value for option, value in pairs if option == "--bind"}
    expected = {
        "/etc/resolv.conf:/etc/resolv.conf:ro",
        f"{runtime.work_tmp}:/tmp",
        f"{runtime.workspace}:/workspace",
        f"{runtime.feedback}:/run/rsi-harness/feedback:ro",
        f"{runtime.agent_home}:/home/agent",
    }
    if network_mode == "public":
        expected.add("/shared/project:/shared/project")
    assert mounts == expected
    environment = runtime._container_environment(
        devices=("GPU-fixture",),
        environment={"TASK_SETTING": "preserved"},
        mount_agent_home=True,
        declared=network_mode,
        phase=phase,
    )
    assert environment["SITE_DATA"] == "/shared/project/data"
    assert environment["TASK_SETTING"] == "preserved"
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
    assert environment["HF_HOME"] == (
        str(profile.storage.hf_home)
        if network_mode == "public"
        else "/tmp/.cache/huggingface"
    )
    assert environment["HF_DATASETS_CACHE"] == (
        str(profile.storage.hf_datasets_cache)
        if network_mode == "public"
        else "/tmp/.cache/huggingface/datasets"
    )


@pytest.mark.parametrize("phase", ["work", "judge"])
@pytest.mark.parametrize("network_mode", ["public", "no-network"])
@pytest.mark.parametrize("devices", [(), ("GPU-fixture",)])
def test_apptainer_enables_nvidia_only_for_the_phase_devices(
    tmp_path,
    monkeypatch,
    phase,
    network_mode,
    devices,
):
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(tmp_path / "node-tmp"))
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("slurm"),
            run_id="cpu-phases",
            sif_path=tmp_path / "task.sif",
        ),
        make_run_plan(tmp_path),
    )

    command = runtime._base_command(
        devices=devices,
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode=network_mode,
        phase=phase,
    )

    assert ("--nv" in command) is bool(devices)
    assert "--containall" in command
    assert "--no-umask" in command
    environment = runtime._container_environment(
        devices=devices,
        environment={"CUDA_VISIBLE_DEVICES": "GPU-unexpected"},
        mount_agent_home=False,
        declared=network_mode,
        phase=phase,
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == ",".join(devices)
    assert ("--net" in command) is (network_mode == "no-network")


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
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="no-network",
        phase="judge",
    )
    assert not any("/run-contract" in item for item in single_judge)

    multi = ApptainerAgentRuntime(payload, plan, allocated_pools=_multi_pools())
    work = multi._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="no-network",
        phase="work",
    )
    judge = multi._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="no-network",
        phase="judge",
    )
    assert not any("/run-contract" in item for item in work)
    assert f"{task_authority}:/run-contract:ro" in judge


@pytest.mark.parametrize("cluster_name", ["bluevela", "slurm"])
def test_multinode_work_runtime_injects_only_current_phase_broker(
    tmp_path: Path,
    monkeypatch,
    cluster_name,
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = profile.model_copy(
        update={
            "adapter": cluster_name,
            "scheduler": load_cluster_profile(cluster_name).scheduler,
        }
    )
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(update={"temp_root": node_tmp}),
            "resources": profile.resources.model_copy(update={"gpus_per_node": 4}),
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
        lambda *_args, **_kwargs: AgentRunResult(
            exit_code=0, output="0.149.0\n" + PROBE_END
        ),
    )
    runtime.initialize()
    try:
        assert runtime.work_devices == tuple(
            f"GPU-work-a-{index}" for index in range(4)
        )
        assert runtime.work_broker is not None
        assert runtime.work_broker.remote_args == profile.scheduler.remote_args
        assert runtime.work_broker.remote_binary == profile.scheduler.remote_binary
        assert (
            runtime.work_broker.worker_template.environment["RSI_SHARED_DATA_ROOT"]
            == profile.apptainer.work_environment["RSI_SHARED_DATA_ROOT"]
        )
        assert runtime.work_broker.worker_template.environment["HF_HOME"] == (
            "/tmp/.cache/huggingface"
        )
        assert (
            runtime.work_broker.worker_template.environment["HF_DATASETS_CACHE"]
            == "/tmp/.cache/huggingface/datasets"
        )
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
            extra_binds=(),
            mount_workspace=True,
            mount_agent_home=True,
            network_mode="public",
        )
        pairs = tuple(zip(command, command[1:], strict=False))
        assert (
            "--bind",
            f"{runtime.work_broker.root}:/run/rsi-harness/torchrun",
        ) in pairs
        environment = runtime._container_environment(
            devices=runtime.work_devices,
            environment=None,
            mount_agent_home=True,
            declared="public",
        )
        assert environment["RSI_MULTINODE_ROOT"] == ("/run/rsi-harness/torchrun")
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
        return AgentRunResult(exit_code=0, output="2.1.246 (Claude Code)\n" + PROBE_END)

    monkeypatch.setattr(runtime, "_run", fake_run)

    runtime.initialize()

    command = runtime._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=True,
        mount_agent_home=True,
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

    environment = runtime._container_environment(
        devices=(), environment=None, mount_agent_home=False, declared="public"
    )

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
        "extra_binds": (),
        "mount_workspace": False,
        "mount_agent_home": False,
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
    isolated_environment = runtime._container_environment(
        devices=(), environment=None, mount_agent_home=False, declared="no-network"
    )
    assert isolated_environment["HF_HOME"] == "/tmp/.cache/huggingface"
    assert (
        isolated_environment["HF_DATASETS_CACHE"] == "/tmp/.cache/huggingface/datasets"
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
            extra_binds=(),
            mount_workspace=True,
            mount_agent_home=True,
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


def _codex_runtime(tmp_path: Path, monkeypatch, **payload) -> ApptainerAgentRuntime:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    return ApptainerAgentRuntime(
        SimpleNamespace(
            **{
                "profile": load_cluster_profile("bluevela"),
                "run_id": "native-run",
                "sif_path": tmp_path / "task.sif",
                "agent_binary": Path(sys.executable),
                "agent_launcher": "codex",
                "agent_companions": (),
                "agent_auth": None,
                "agent_version": "0.149.0",
                **payload,
            }
        ),
        make_run_plan(tmp_path),
    )


@pytest.mark.parametrize(
    ("exit_code", "output"),
    [
        (127, ""),
        (0, "codex-cli 0.148.0\n"),
        (0, "codex-cli 0.149.01\n"),
        (0, "codex-cli 0.149.0-beta.1\n"),
        (0, "codex-cli 0.149.0\nwrapper 1.2.3\n"),
        (0, ""),
    ],
)
def test_agent_binary_probe_explains_unusable_container_cli(
    tmp_path, monkeypatch, exit_code, output
):
    binary = tmp_path / "host-wrapper"
    runtime = _codex_runtime(tmp_path, monkeypatch, agent_binary=binary)
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return AgentRunResult(exit_code=exit_code, output=output)

    monkeypatch.setattr(runtime, "_run", run)
    with pytest.raises(InfrastructureError) as caught:
        runtime._smoke_agent()
    message = str(caught.value)
    assert str(binary) in message
    assert f"exited {exit_code}" in message
    assert "expected '0.149.0'" in message
    assert "host-only wrappers" in message
    assert calls[0][0] == ("/usr/local/bin/codex", "--version")
    assert calls[0][1]["timeout_seconds"] == 60


@pytest.mark.parametrize(
    "output",
    [
        "codex-cli 0.149.0\n",
        "Wrapper starting CLI\ncodex-cli v0.149.0\n",
        "CLI 0.149.0\nversion 0.149.0\n",
    ],
)
def test_agent_binary_probe_accepts_exact_version_with_banner(
    tmp_path, monkeypatch, output
):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runtime, "_run", lambda *_a, **_kw: AgentRunResult(exit_code=0, output=output)
    )
    runtime._smoke_agent()


def test_resumed_initialize_reuses_the_run_directories_and_seeds_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _codex_runtime(tmp_path, monkeypatch)
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_options):
        calls.append(tuple(command))
        return AgentRunResult(exit_code=0, output="0.149.0\n" + PROBE_END)

    monkeypatch.setattr(runtime, "_run", fake_run)
    runtime.initialize()
    (runtime.workspace / "answer.py").write_text("answer = 42\n")
    (runtime.agent_home / ".codex").mkdir()
    (runtime.rounds / "agent-1").mkdir()
    # The next attempt's job: a fresh node-local scratch.
    node_tmp = tmp_path / "node-tmp-2"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    resumed = ApptainerAgentRuntime(runtime.payload, runtime.plan)
    calls.clear()
    monkeypatch.setattr(resumed, "_run", fake_run)

    resumed.initialize(resume=True)

    # The smoke check, then the isolation probe; no seed.
    assert calls[0] == ("/usr/local/bin/codex", "--version")
    assert [call[:2] for call in calls[1:]] == [("/bin/sh", "-c")]
    assert (resumed.workspace / "answer.py").read_text() == "answer = 42\n"
    assert (resumed.agent_home / ".codex").is_dir()
    assert (resumed.rounds / "agent-1").is_dir()
    assert (resumed.control / "rsi-submit").is_file()


def test_resumed_initialize_refuses_a_missing_run_directory(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _codex_runtime(tmp_path, monkeypatch)
    runtime.control.mkdir(parents=True)
    (runtime.control / "seeded").write_text("\n")
    monkeypatch.setattr(runtime, "_run", lambda *_a, **_k: pytest.fail("Work started"))

    with pytest.raises(SetupError, match="resumed run directory is missing"):
        runtime.initialize(resume=True)


def test_resumed_initialize_seeds_a_run_that_never_finished_seeding(
    tmp_path: Path, monkeypatch
) -> None:
    """Attempt 1 died copying the image's WORKDIR, before any Agent ran: the
    next attempt seeds the partial workspace and task assets again."""
    runtime = _codex_runtime(tmp_path, monkeypatch)
    runtime.workspace.mkdir(parents=True)
    (runtime.workspace / "partial.py").write_text("trunc")
    runtime.control.mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def fake_run(command, **_options):
        calls.append(tuple(command))
        return AgentRunResult(exit_code=0, output="0.149.0\n" + PROBE_END)

    monkeypatch.setattr(runtime, "_run", fake_run)

    runtime.initialize(resume=True)

    shells = [command[-1] for command in calls]
    assert any("/run/rsi-harness/seed" in shell for shell in shells)
    assert any("/run/rsi-harness/task-tmp" in shell for shell in shells)
    assert (runtime.control / "seeded").is_file()
    assert runtime.task_assets.is_dir() and runtime.rounds.is_dir()


def test_resumed_initialize_refuses_a_multinode_run(
    tmp_path: Path, monkeypatch
) -> None:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    profile = load_cluster_profile("bluevela")
    profile = profile.model_copy(
        update={"resources": profile.resources.model_copy(update={"gpus_per_node": 4})}
    )
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(profile=profile, run_id="native-run"),
        make_run_plan(tmp_path),
        allocated_pools=_multi_pools(),
    )

    with pytest.raises(SetupError, match="multi-node cluster run cannot be resumed"):
        runtime.initialize(resume=True)


@pytest.mark.parametrize(
    ("agent_name", "target", "login", "link", "store"),
    [
        ("codex", "/home/agent/.codex", "auth.json", "sessions", ".codex-sessions"),
        (
            "claude-code",
            "/home/agent/.claude",
            ".credentials.json",
            "projects",
            ".claude-projects",
        ),
    ],
)
def test_local_login_sessions_are_kept_in_the_shared_agent_home(
    tmp_path: Path, monkeypatch, agent_name, target, login, link, store
) -> None:
    """Under --agent-auth local the login directory is node-local; the CLI's
    sessions in it reach the Agent home, which a resumed attempt mounts."""
    material = AgentAuthMaterial(
        agent_name=agent_name,
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath(target),
                    options="rw,nosuid,nodev,noexec,mode=0700",
                ),
                files=(AgentAuthFile(path=PurePosixPath(login), content=b"{}"),),
            ),
        ),
        secret_values=frozenset(),
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.resolve_agent_auth",
        lambda **_options: material,
    )
    runtime = _codex_runtime(tmp_path, monkeypatch, agent_auth=AgentAuthSource.LOCAL)
    agent = runtime.plan.task.agent.model_copy(update={"name": agent_name})
    runtime.plan = runtime.plan.model_copy(
        update={"task": runtime.plan.task.model_copy(update={"agent": agent})}
    )
    runtime.agent_home.mkdir(parents=True)

    runtime._install_auth()

    ((source, bound, _read_only),) = runtime._agent_auth_binds
    assert bound == PurePosixPath(target)
    assert source.is_relative_to(runtime.node_tmp)
    assert os.readlink(source / link) == f"/home/agent/{store}"
    assert (runtime.agent_home / store).is_dir()


def test_agent_output_is_appended_across_attempts(tmp_path: Path, monkeypatch) -> None:
    runtime = _codex_runtime(tmp_path, monkeypatch)
    runtime.workspace.mkdir(parents=True)
    output = tmp_path / "agent_output.txt"
    output.write_text("attempt 1\n")
    monkeypatch.setattr(
        runtime,
        "_base_command",
        lambda **_options: (sys.executable, "-c", "print('attempt 2')"),
    )
    monkeypatch.setattr(runtime, "_network_command", lambda command, **_o: ())

    runtime._run(
        (), devices=(), network_mode="public", output_path=output, timeout_seconds=30
    )

    assert output.read_text() == "attempt 1\nattempt 2\n"


def test_a_cancelled_run_stops_its_process_group_before_reading_its_output(
    tmp_path: Path, monkeypatch
) -> None:
    """The job's SIGTERM reaches only the Engine: Work, still running and
    holding its output open, is stopped at once instead of being waited on."""

    class Cancelled(BaseException):
        pass

    def cancel(*_args):
        raise Cancelled

    runtime = _codex_runtime(tmp_path, monkeypatch)
    pid_file = tmp_path / "child.pid"
    monkeypatch.setattr(
        runtime,
        "_base_command",
        lambda **_options: (
            "sh",
            "-c",
            f"sleep 300 & echo $! > {pid_file}; wait",
        ),
    )
    monkeypatch.setattr(runtime, "_network_command", lambda command, **_o: ())
    previous = signal.signal(signal.SIGALRM, cancel)
    signal.setitimer(signal.ITIMER_REAL, 0.5)
    started = time.monotonic()
    try:
        with pytest.raises(Cancelled):
            runtime._run((), devices=(), network_mode="public", timeout_seconds=300)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)

    assert time.monotonic() - started < 15
    child = int(pid_file.read_text())
    deadline = time.monotonic() + 10
    while _process_state(child) not in (None, "Z"):
        assert time.monotonic() < deadline
        time.sleep(0.05)


class _RecordingPopen:
    """Stands in for subprocess.Popen in _run: records argv and environment."""

    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def __init__(self, argv, *, env, **_options) -> None:
        self.calls.append((tuple(argv), dict(env)))
        self.stdout = iter(())
        self.returncode = 0
        self.pid = 0

    def wait(self, timeout=None) -> int:
        return 0


def test_every_exec_is_contained_and_ignores_inherited_apptainer_settings(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = _codex_runtime(tmp_path, monkeypatch)
    for name in (
        "APPTAINER_BINDPATH",
        "SINGULARITY_BINDPATH",
        "APPTAINER_NO_MOUNT",
        "APPTAINERENV_FORGED",
    ):
        monkeypatch.setenv(name, "/oak")
    monkeypatch.setenv("APPTAINER_TMPDIR", "/lscratch/build-tmp")
    monkeypatch.setenv("APPTAINER_CACHEDIR", "/scratch/cache")
    monkeypatch.setattr(_RecordingPopen, "calls", [])
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)

    runtime._seed()
    runtime._run(
        ("true",),
        devices=(),
        network_mode="public",
        environment={"OPENAI_API_KEY": "sk-never-in-argv", "LIST": "a,b"},
        timeout_seconds=5,
    )

    (seed, _), (image_tmp, _), (agent, environment) = _RecordingPopen.calls
    for argv in (seed, image_tmp, agent):
        assert argv.count("--containall") == 1
        pairs = tuple(zip(argv, argv[1:], strict=False))
        assert ("--bind", "/etc/resolv.conf:/etc/resolv.conf:ro") in pairs
        no_mount = [value for option, value in pairs if option == "--no-mount"]
        assert no_mount == (["hostfs,tmp"] if argv is image_tmp else ["hostfs"])
        assert "--env" not in argv
    assert not any(bind.endswith(":/tmp") for bind in image_tmp)
    assert "sk-never-in-argv" not in " ".join(agent)
    assert environment["APPTAINERENV_OPENAI_API_KEY"] == "sk-never-in-argv"
    assert environment["APPTAINERENV_LIST"] == "a,b"
    assert environment["APPTAINER_TMPDIR"] == "/lscratch/build-tmp"
    assert environment["APPTAINER_CACHEDIR"] == "/scratch/cache"
    assert (
        not {
            "APPTAINER_BIND",
            "APPTAINER_BINDPATH",
            "SINGULARITY_BINDPATH",
            "APPTAINER_NO_MOUNT",
            "APPTAINERENV_FORGED",
        }
        & environment.keys()
    )


def _capture_verbose_codex_result(tmp_path, monkeypatch, *, multiline=True):
    """Exercise production capture with only the Apptainer launcher replaced."""
    runtime = _codex_runtime(tmp_path, monkeypatch)
    session = "01a16125-90ff-70a0-8d09-1a36fffb91d0"
    header = f"OpenAI Codex v0.159.2\nsession id: {session}\nprivate-secret 中文\n"
    terminal = "ERROR: Selected model is at capacity. Please try a different model.\n"
    row = "x" * 998 + "🧪" + ("\n" if multiline else "")
    trailer = "\nprivate-secret END中文\n" + terminal
    script = (
        "import sys; "
        f"sys.stdout.buffer.write({header.encode()!r}); "
        f"sys.stdout.buffer.write({row.encode()!r} * 16010); "
        f"sys.stdout.buffer.write({trailer.encode()!r}); sys.exit(1)"
    )
    monkeypatch.setattr(
        runtime, "_base_command", lambda **_options: (sys.executable, "-c", script)
    )
    monkeypatch.setattr(runtime, "_network_command", lambda command, **_o: ())
    output = tmp_path / "complete-output.log"
    output.write_text("prior attempt\n")
    streamed = []
    result = runtime._run(
        (),
        devices=(),
        network_mode="public",
        timeout_seconds=30,
        output_path=output,
        output_redact_values=("private-secret",),
        output_callback=streamed.append,
    )
    expected = header + row * 16010 + trailer
    from rsi_harness.cluster.bluevela.runtime import _safe_output

    expected = _safe_output(expected, {"private-secret"})
    return result, output, streamed, expected


@pytest.mark.parametrize("multiline", [False, True])
def test_apptainer_capture_retains_verbose_header_and_terminal_error(
    tmp_path, monkeypatch, multiline
):
    result, output, streamed, expected = _capture_verbose_codex_result(
        tmp_path, monkeypatch, multiline=multiline
    )
    assert len(expected.encode()) > 16_000_000
    assert result.exit_code == 1
    assert not result.timed_out
    assert result.output_truncated
    # Feedback is bounded, but the complete verifier log is persisted.
    assert result.full_output_captured
    assert len(result.output.encode()) <= 16_000_000
    assert result.output.startswith("OpenAI Codex v0.159.2\nsession id: ")
    assert result.output.endswith("Please try a different model.\n")
    assert "中文" in result.output and "END中文" in result.output
    assert "�" not in result.output
    assert "private-secret" not in result.output
    assert output.read_text() == "prior attempt\n" + expected
    assert "".join(streamed) == expected


@pytest.mark.parametrize("directory", [False, True])
def test_apptainer_copy_to_owns_temporary_sources_and_replaces_prior_copy(
    tmp_path, monkeypatch, directory
):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    source = tmp_path / "temporary-source"
    if directory:
        source.mkdir()
        (source / "script.sh").write_text("first\n")
        (source / "script.sh").chmod(0o750)
    else:
        source.write_text("first\n")
        source.chmod(0o750)
    target = PurePosixPath("/tmp/copied-input")
    runtime.copy_to(runtime.work_ref, source, target)
    staged = runtime._copies[target]
    original = source / "script.sh" if directory else source
    copied = staged / "script.sh" if directory else staged
    original.write_text("source changed\n")
    assert copied.read_text() == "first\n"
    assert copied.stat().st_mode & 0o777 == 0o750
    if directory:
        import shutil

        shutil.rmtree(source)
    else:
        source.unlink()
    assert copied.read_text() == "first\n"
    assert staged.is_relative_to(runtime.node_tmp / "copied-inputs")
    command = runtime._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=True,
        mount_agent_home=True,
        network_mode="public",
    )
    assert f"{staged}:{target}:ro" in command

    # A repeated copy may change the source type. It replaces only its own
    # staged input, leaving no old directory or stale content behind.
    source.write_text("replacement\n")
    runtime.copy_to(runtime.work_ref, source, target)
    assert not staged.exists()
    assert runtime._copies[target].read_text() == "replacement\n"
    assert len(list((runtime.node_tmp / "copied-inputs").iterdir())) == 1


def test_apptainer_copy_to_failure_preserves_prior_bind_and_builtin_source(
    tmp_path, monkeypatch
):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    builtin = tmp_path / "built-in-agent"
    builtin.write_text("CLI")
    target = PurePosixPath("/usr/local/bin/agent")
    runtime._copies[target] = builtin
    source = tmp_path / "source"
    source.write_text("replacement")
    runtime.copy_to(runtime.work_ref, source, target)
    assert builtin.read_text() == "CLI"
    staged = runtime._copies[target]
    with pytest.raises(FileNotFoundError):
        runtime.copy_to(runtime.work_ref, tmp_path / "missing", target)
    assert runtime._copies[target] == staged
    assert staged.read_text() == "replacement"
    assert len(list((runtime.node_tmp / "copied-inputs").iterdir())) == 1


def test_apptainer_copy_to_preserves_directory_symlinks_without_dereferencing_host(
    tmp_path, monkeypatch
):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    source = tmp_path / "source-tree"
    source.mkdir()
    (source / "local").write_text("local payload")
    (source / "relative").symlink_to("local")
    (source / "dangling").symlink_to("container-created-later")
    outside = tmp_path / "host-only"
    outside.write_text("host-only bytes")
    outside.chmod(0o600)
    (source / "absolute").symlink_to(outside)
    target = PurePosixPath("/tmp/tree")

    runtime.copy_to(runtime.work_ref, source, target)

    copied = runtime._copies[target]
    for name in ("relative", "dangling", "absolute"):
        assert (copied / name).is_symlink()
        assert (copied / name).readlink() == (source / name).readlink()
    assert (copied / "local").read_text() == "local payload"
    assert outside.read_text() == "host-only bytes"
    assert outside.stat().st_mode & 0o777 == 0o600


def test_apptainer_copy_to_cleans_readonly_tree_on_replacement_and_failed_copy(
    tmp_path, monkeypatch
):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    source = tmp_path / "read-only-source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (nested / "file").write_text("payload")
    nested.chmod(0o555)
    source.chmod(0o555)
    target = PurePosixPath("/tmp/tree")
    runtime.copy_to(runtime.work_ref, source, target)
    previous = runtime._copies[target]
    assert previous.stat().st_mode & 0o777 == 0o555
    runtime.copy_to(runtime.work_ref, source, target)
    assert not previous.exists()
    current = runtime._copies[target]

    def fail_after_partial_readonly_copy(_source, destination, **_options):
        nested = destination / "nested"
        nested.mkdir(parents=True)
        (nested / "file").write_text("partial")
        nested.chmod(0o555)
        destination.chmod(0o555)
        raise OSError("copy failed")

    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.shutil.copytree",
        fail_after_partial_readonly_copy,
    )
    with pytest.raises(OSError, match="copy failed"):
        runtime.copy_to(runtime.work_ref, source, target)
    assert runtime._copies[target] == current
    assert (current / "nested/file").read_text() == "payload"
    assert len(list((runtime.node_tmp / "copied-inputs").iterdir())) == 1
    assert source.stat().st_mode & 0o777 == 0o555
    assert (source / "nested").stat().st_mode & 0o777 == 0o555


def test_host_controller_persists_output_beyond_feedback_limit(tmp_path, monkeypatch):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    output = tmp_path / "controller.log"
    script = (
        "import sys; sys.stdout.buffer.write(b'x'*20_000_000 + b'END'); sys.exit(7)"
    )
    result = runtime._run_host_controller(
        (sys.executable, "-c", script), timeout_seconds=20, output_path=output
    )
    assert result.exit_code == 7 and not result.timed_out
    assert result.output_truncated and result.full_output_captured
    assert len(result.output.encode()) <= 16_000_000
    assert result.output.endswith("END")
    assert output.stat().st_size == 20_000_003


@pytest.mark.parametrize("stop_failure", ["timeout", "oserror", "nonzero"])
def test_host_controller_timeout_always_reaps_and_keeps_logs(
    tmp_path, monkeypatch, stop_failure
):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    output = tmp_path / "controller.log"
    pid_file = tmp_path / "pid"
    script = (
        "import os, time; from pathlib import Path; "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "print('started', flush=True); time.sleep(60)"
    )
    from rsi_harness.cluster.bluevela import runtime as module

    monkeypatch.setattr(module, "stop_command", lambda *a, **k: ("stop",))

    def failed_stop(*args, **kwargs):
        if stop_failure == "timeout":
            raise subprocess.TimeoutExpired("stop", 20)
        if stop_failure == "oserror":
            raise OSError("cannot launch stop")
        return subprocess.CompletedProcess(args[0], 3)

    monkeypatch.setattr(module.subprocess, "run", failed_stop)
    result = runtime._run_host_controller(
        (sys.executable, "-c", script), timeout_seconds=0.5, output_path=output
    )
    assert result.timed_out and result.full_output_captured
    assert result.exit_code == -signal.SIGTERM
    assert "started" in output.read_text() and "controller stop" in output.read_text()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_native_evaluator_does_not_claim_incomplete_logs_are_complete(tmp_path):
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="native-run")
    writer.start()
    runtime = _Runtime(tmp_path)
    original = runtime.run_judge

    def incomplete(*args, **kwargs):
        original(*args, **kwargs)
        return AgentRunResult(exit_code=0, output="partial", full_output_captured=False)

    runtime.run_judge = incomplete
    request = EvaluationRequest(
        run_plan=plan,
        work_container=ContainerRef(container_id="work", role="work"),
        round_id="agent-1",
        verifier_logs=writer.root / "verifier/agent-1",
        verifier_output=writer.feedback_root / "agent-1.log",
    )
    with pytest.raises(OSError, match="complete verifier output capture"):
        NativeJudgeEvaluator(runtime, writer).evaluate(request, _Observer())
    report = json.loads((writer.root / "submissions/agent-1/report.json").read_text())
    assert report["full_output_captured"] is False


def test_host_controller_escalates_a_stubborn_process_to_kill(tmp_path, monkeypatch):
    runtime = _codex_runtime(tmp_path, monkeypatch)
    from rsi_harness.cluster.bluevela import runtime as module

    monkeypatch.setattr(
        module, "stop_command", lambda *a, **k: (sys.executable, "-c", "pass")
    )
    result = runtime._run_host_controller(
        (
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)",
        ),
        timeout_seconds=0.5,
        output_path=tmp_path / "controller.log",
    )
    assert result.timed_out and result.exit_code == -signal.SIGKILL
    assert result.full_output_captured and "ready" in result.output
