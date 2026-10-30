from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.bluevela.multinode import MultiNodeRequest
from rsi_harness.cluster.bluevela.remote_worker import (
    build_apptainer_command,
    load_control,
    request_tmp_dir,
)
from rsi_harness.cluster.bluevela.runtime import (
    ApptainerAgentRuntime,
    NativeEngineComposition,
    NativeJudgeEvaluator,
    _agent_provider_urls,
    _ClusterAgentAdapter,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import InfrastructureError, RetryableSubmissionError, SetupError
from rsi_harness.models import (
    AgentAuthSource,
    AgentHookRequest,
    AgentRunResult,
    CompileOptions,
    EvaluationRequest,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from tests.cluster.bluevela.test_adapter import _profile
from tests.cluster.bluevela.test_runtime import (
    PROBE_END,
    _codex_runtime,
    _multi_pools,
    _Observer,
)
from tests.cluster.slurm.test_adapter import RecordingSlurmScheduler
from tests.factories import DEFAULT_TASK_TOML, make_run_plan, write_harbor_task


def setup_oracle(tmp_path, *, gpus=0, judge_gpus=0):
    task = write_harbor_task(
        tmp_path / "task",
        task_toml=(
            DEFAULT_TASK_TOML.replace("gpus = 2", f"gpus = {gpus}")
            + f"\n[metadata.rsi_harness.verifier]\ngpus = {judge_gpus}\n"
        ),
    )
    (task / "solution").mkdir()
    (task / "solution/solve.sh").write_text("#!/bin/sh\nprintf 42 > answer\n")
    local = _profile(tmp_path)
    profile = load_cluster_profile("slurm", {"USER": "alice"}).model_copy(
        update={
            "storage": local.storage,
            "apptainer": local.apptainer,
            "resources": local.resources,
            "builder": local.builder,
        }
    )
    request = ClusterRunRequest(
        task_dir=task.resolve(),
        agent_name="oracle",
        options=CompileOptions(agent_name="oracle"),
        logs_root=tmp_path / "logs",
        dry_run=True,
    )
    return profile, request


@pytest.mark.parametrize("gpus,judge_gpus", [(0, 0), (1, 0), (1, 1)])
def test_oracle_dryrun_needs_no_model_cli_or_auth(
    tmp_path, monkeypatch, gpus, judge_gpus
):
    profile, request = setup_oracle(tmp_path, gpus=gpus, judge_gpus=judge_gpus)
    # The configured path is only available inside a compute allocation.
    profile.apptainer.binary.unlink()
    scheduler = RecordingSlurmScheduler()
    events = {}
    monkeypatch.setattr("shutil.which", lambda _: None)
    adapter = SlurmClusterAdapter(
        profile,
        scheduler=scheduler,
        agent_version_resolver=lambda _: pytest.fail("Oracle queried a model CLI"),
        event_callback=lambda key, value: events.update({key: value}),
    )
    adapter.run(request)
    assert scheduler.events == []
    assert events["dry_run"]["resources"]["total_gpus"] == gpus + judge_gpus
    assert events["dry_run"]["agent_version"] is None
    assert not profile.storage.run_root.exists()


@pytest.mark.parametrize(
    "problem", ["lsf", "auth", "effort", "missing", "symlink", "model_access"]
)
def test_oracle_refuses_unsupported_inputs_before_jobs(tmp_path, problem):
    profile, request = setup_oracle(tmp_path)
    if problem == "lsf":
        profile = _profile(tmp_path)
    elif problem == "auth":
        request = request.model_copy(update={"agent_auth": AgentAuthSource.LOCAL})
    elif problem == "effort":
        request = request.model_copy(update={"reasoning_effort": "medium"})
    elif problem in {"missing", "symlink"}:
        script = request.task_dir / "solution/solve.sh"
        script.unlink()
        if problem == "symlink":
            script.symlink_to(request.task_dir / "instruction.md")
    elif problem == "model_access":
        with (request.task_dir / "task.toml").open("a") as f:
            f.write("model_access = true\n")
    scheduler = RecordingSlurmScheduler()
    cls = BlueVelaClusterAdapter if problem == "lsf" else SlurmClusterAdapter
    with pytest.raises(SetupError):
        cls(profile, scheduler=scheduler).run(request)
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()


@pytest.mark.parametrize("gpus,judge_gpus", [(16, 8), (8, 16), (16, 16)])
def test_multinode_oracle_dryrun_uses_existing_disjoint_phase_plan(
    tmp_path, monkeypatch, gpus, judge_gpus
):
    profile, request = setup_oracle(tmp_path, gpus=gpus, judge_gpus=judge_gpus)
    profile.apptainer.binary.unlink()
    monkeypatch.setattr("shutil.which", lambda _: None)
    scheduler = RecordingSlurmScheduler()
    events = {}
    SlurmClusterAdapter(
        profile,
        scheduler=scheduler,
        agent_version_resolver=lambda _: pytest.fail("Oracle queried a model CLI"),
        event_callback=lambda key, value: events.update({key: value}),
    ).run(request)
    resources = events["dry_run"]["multi_node"]
    assert resources["work"]["gpu_count"] == gpus
    assert resources["verifier"]["gpu_count"] == judge_gpus
    assert resources["total_nodes"] == (gpus + judge_gpus) // 8
    assert events["dry_run"]["agent_version"] is None
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()


@pytest.mark.parametrize(
    "problem", ["empty-work", "empty-judge", "partial-node", "seccomp", "harness-pid"]
)
def test_multinode_oracle_refuses_unsupported_topology_or_isolation_before_jobs(
    tmp_path, problem
):
    profile, request = setup_oracle(
        tmp_path,
        gpus=0 if problem == "empty-work" else 16,
        judge_gpus={"empty-judge": 0, "partial-node": 1}.get(problem, 16),
    )
    expected = "nonempty Work and Judge node pools"
    if problem == "empty-work":
        expected = "CPU Work requires a single-node Slurm allocation"
    elif problem == "partial-node":
        expected = "whole 8-GPU nodes"
    elif problem in {"seccomp", "harness-pid"}:
        values = (
            {"network_isolation": "seccomp"}
            if problem == "seccomp"
            else {"process_namespaces": "harness"}
        )
        profile = profile.model_copy(
            update={"apptainer": profile.apptainer.model_copy(update=values)}
        )
        expected = "need Apptainer's PID and network namespaces"
    scheduler = RecordingSlurmScheduler()
    with pytest.raises(SetupError, match=expected):
        SlurmClusterAdapter(profile, scheduler=scheduler).run(request)
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()


def _oracle_runtime(tmp_path, monkeypatch, *, multi=False):
    monkeypatch.setenv("USER", "alice")
    runtime = _codex_runtime(
        tmp_path,
        monkeypatch,
        agent_launcher="rsi-oracle",
        agent_binary=None,
        agent_version=None,
    )
    runtime.profile = load_cluster_profile("slurm", {"USER": "alice"})
    if multi:
        runtime.profile = runtime.profile.model_copy(
            update={
                "resources": runtime.profile.resources.model_copy(
                    update={"gpus_per_node": 4}
                )
            }
        )
        runtime.allocated_pools = _multi_pools()
        runtime.payload.sif_sha256_path = tmp_path / "task.sif.sha256"
        runtime.payload.sif_sha256_path.write_text("d" * 64 + "  task.sif\n")
    runtime.plan = runtime.plan.model_copy(
        update={
            "task": runtime.plan.task.model_copy(
                update={
                    "agent": runtime.plan.task.agent.model_copy(
                        update={"name": "oracle"}
                    )
                }
            )
        }
    )
    solution = runtime.plan.task.source_dir / "solution"
    solution.mkdir(parents=True)
    (solution / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(
        runtime,
        "_run",
        lambda *a, **k: AgentRunResult(exit_code=0, output=PROBE_END),
    )
    return runtime


def test_solution_is_writable_work_only_and_original_is_preserved(
    tmp_path, monkeypatch
):
    runtime = _oracle_runtime(tmp_path, monkeypatch)
    solution = runtime.plan.task.source_dir / "solution"
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    outside.chmod(0o400)
    (solution / "link").symlink_to(outside)
    runtime.initialize()
    adapter = _ClusterAgentAdapter(SimpleNamespace(), runtime=runtime)
    adapter._stage_solution(
        runtime,
        AgentHookRequest(
            run_plan=runtime.plan,
            container=runtime.work_ref,
            submit_url="http://localhost",
            token="test",
        ),
    )
    assert runtime._oracle_solution != solution
    assert (runtime._oracle_solution / "solve.sh").stat().st_mode & 0o100
    assert outside.stat().st_mode & 0o777 == 0o400
    assert runtime._copies[PurePosixPath("/usr/local/bin/rsi-oracle")].is_file()
    assert runtime._agent_cli_targets == []
    assert _agent_provider_urls(None, runtime.plan, None) == ()

    def command(phase):
        return runtime._base_command(
            devices=(),
            extra_binds=(),
            mount_workspace=phase == "work",
            mount_agent_home=phase == "work",
            network_mode="public",
            phase=phase,
        )

    assert f"{runtime._oracle_solution}:/solution" in command("work")
    assert not any(
        ":/solution" in arg or ":/usr/local/bin/rsi-oracle" in arg
        for arg in command("judge")
    )
    (runtime._oracle_solution / "solve.sh").unlink()
    assert (solution / "solve.sh").is_file()

    # A profile/task environment cannot opt single-node Work out of cleanup.
    environment = runtime._container_environment(
        devices=(),
        environment={"RSI_ORACLE_SOLUTION_CLEANUP": "submit"},
        mount_agent_home=True,
        declared="no-network",
    )
    assert environment["RSI_ORACLE_SOLUTION_CLEANUP"] == "launcher"


def test_multinode_oracle_binds_shared_solution_to_every_work_rank(
    tmp_path, monkeypatch
):
    runtime = _oracle_runtime(tmp_path, monkeypatch, multi=True)
    runtime.initialize()
    try:
        runtime.stage_solution()
        solution = runtime.control / "oracle-solution"
        assert runtime._oracle_solution == solution
        assert not solution.is_relative_to(runtime.node_tmp)
        assert (solution / "solve.sh").is_file()
        broker = runtime.work_broker
        assert broker is not None
        intent = MultiNodeRequest(
            request_id="b" * 32,
            run_id=runtime.payload.run_id,
            pool_digest=runtime.allocated_pools.digest,
            node_count=2,
            local_world_size=4,
            argv=("--nnodes=2", "--nproc-per-node=4", "/solution/train.py"),
            working_directory=runtime.plan.workdir,
            environment={},
        )
        reserved = broker.reserve(intent)
        try:
            control = load_control(reserved.control_path)
            assert tuple(node.host for node in control.lease.nodes) == (
                "work-a",
                "work-b",
            )
            for rank, node in enumerate(control.lease.nodes):
                command = build_apptainer_command(
                    control,
                    request_id=intent.request_id,
                    pool_rank=rank,
                    node_rank=rank,
                    current_host=node.host,
                    local_tmp=request_tmp_dir(control, pool_rank=rank),
                )
                assert f"{solution}:/solution" in command
                assert command[-1] == "/solution/train.py"
                assert f"--node-rank={rank}" in command
                assert not any("judge-a" in part for part in command)
        finally:
            broker.release(reserved)
        environment = runtime._container_environment(
            devices=runtime.work_devices,
            environment=None,
            mount_agent_home=True,
            declared="no-network",
        )
        assert environment["RSI_ORACLE_SOLUTION_CLEANUP"] == "submit"
        judge_environment = runtime._container_environment(
            devices=(),
            environment=None,
            mount_agent_home=False,
            declared="no-network",
            phase="judge",
        )
        assert "RSI_ORACLE_SOLUTION_CLEANUP" not in judge_environment
    finally:
        runtime.close_multinode()


@pytest.mark.parametrize("failure", [None, "busy", "cleanup"])
def test_multinode_oracle_retires_solution_after_idle_before_judge(
    tmp_path, monkeypatch, failure
):
    runtime = _oracle_runtime(tmp_path, monkeypatch, multi=True)
    runtime.initialize()
    runtime.stage_solution()
    solution = runtime._oracle_solution
    original = runtime.plan.task.source_dir / "solution/solve.sh"
    (runtime.workspace / "answer").write_text("42\n")
    events = []
    monkeypatch.setattr(runtime, "pause", lambda _: events.append("pause"))
    monkeypatch.setattr(runtime, "unpause", lambda _: events.append("unpause"))

    def idle():
        assert events == ["pause"]
        assert (solution / "solve.sh").is_file()
        events.append("idle")
        if failure == "busy":
            raise InfrastructureError("phase broker has active remote requests")

    monkeypatch.setattr(runtime, "require_work_idle", idle)
    cleanup = runtime.remove_oracle_solution

    def remove():
        assert events == ["pause", "idle"]
        events.append("cleanup")
        if failure == "cleanup":
            raise InfrastructureError("cannot retire remote Work")
        cleanup()

    monkeypatch.setattr(runtime, "remove_oracle_solution", remove)

    def judge(snapshot, request, environment, *, trajectory):
        assert events == ["pause", "idle", "cleanup"]
        assert not solution.exists()
        assert runtime.work_broker is None
        assert (snapshot / "answer").read_text() == "42\n"
        assert sorted(p.name for p in snapshot.iterdir()) == ["answer"]
        events.append("judge")
        (request.verifier_logs / "reward.txt").write_text("1\n")
        request.verifier_output.write_text("passed\n")
        return AgentRunResult(exit_code=0, output="passed\n", full_output_captured=True)

    monkeypatch.setattr(runtime, "run_judge", judge)
    writer = RunArtifactWriter(runtime.plan, run_id=runtime.payload.run_id)
    writer.start()
    request = EvaluationRequest(
        run_plan=runtime.plan,
        work_container=runtime.work_ref,
        round_id="agent-1",
        verifier_logs=writer.root / "verifier/agent-1",
        verifier_output=writer.feedback_root / "agent-1.log",
    )
    try:
        evaluator = NativeJudgeEvaluator(runtime, writer)
        if failure:
            error = (
                RetryableSubmissionError if failure == "busy" else InfrastructureError
            )
            with pytest.raises(error):
                evaluator.evaluate(request, _Observer())
            assert (solution / "solve.sh").is_file()
            assert "judge" not in events
            assert not (runtime.rounds / "agent-1").exists()
        else:
            assert evaluator.evaluate(request, _Observer()).score == 1.0
            assert events == ["pause", "idle", "cleanup", "judge", "unpause"]
        assert original.is_file()
    finally:
        runtime.close_multinode()


def test_multinode_oracle_cannot_resume(tmp_path, monkeypatch):
    runtime = _oracle_runtime(tmp_path, monkeypatch, multi=True)
    with pytest.raises(SetupError, match="Oracle validation cannot resume"):
        runtime.initialize(resume=True)


@pytest.mark.parametrize("busy_first", [False, True])
def test_multinode_oracle_real_broker_barrier_and_repeated_submissions(
    tmp_path, monkeypatch, busy_first
):
    runtime = _oracle_runtime(tmp_path, monkeypatch, multi=True)
    runtime.initialize()
    runtime.stage_solution()
    solution = runtime._oracle_solution
    broker = runtime.work_broker
    assert broker is not None
    (runtime.workspace / "answer").write_text("42\n")
    monkeypatch.setattr(runtime, "pause", lambda _: None)
    monkeypatch.setattr(runtime, "unpause", lambda _: None)
    observed = []

    def judge(snapshot, request, environment, *, trajectory):
        assert not solution.exists()
        assert runtime.work_broker is None
        assert not (broker.root / "READY.json").exists()
        assert (snapshot / "answer").read_text() == "42\n"
        observed.append(request.round_id)
        (request.verifier_logs / "reward.txt").write_text("1\n")
        request.verifier_output.write_text("passed\n")
        return AgentRunResult(exit_code=0, output="passed\n", full_output_captured=True)

    monkeypatch.setattr(runtime, "run_judge", judge)
    writer = RunArtifactWriter(runtime.plan, run_id=runtime.payload.run_id)
    writer.start()
    evaluator = NativeJudgeEvaluator(runtime, writer)

    def request(round_id):
        return EvaluationRequest(
            run_plan=runtime.plan,
            work_container=runtime.work_ref,
            round_id=round_id,
            verifier_logs=writer.root / "verifier" / round_id,
            verifier_output=writer.feedback_root / f"{round_id}.log",
        )

    try:
        if busy_first:
            reserved = broker.reserve(
                MultiNodeRequest(
                    request_id="c" * 32,
                    run_id=runtime.payload.run_id,
                    pool_digest=runtime.allocated_pools.digest,
                    node_count=2,
                    local_world_size=4,
                    argv=("--nnodes=2", "--nproc-per-node=4", "/solution/train.py"),
                    working_directory=runtime.plan.workdir,
                    environment={},
                )
            )
            try:
                with pytest.raises(RetryableSubmissionError):
                    evaluator.evaluate(request("busy"), _Observer())
                assert runtime.work_broker is broker
                assert (solution / "solve.sh").is_file()
                assert observed == []
                assert not (runtime.rounds / "busy").exists()
            finally:
                broker.release(reserved)
        for round_id in ("agent-1", "agent-2", "agent-3"):
            assert evaluator.evaluate(request(round_id), _Observer()).score == 1.0
        assert observed == ["agent-1", "agent-2", "agent-3"]
        assert (runtime.plan.task.source_dir / "solution/solve.sh").is_file()
    finally:
        runtime.close_multinode()


def test_multinode_oracle_failed_file_cleanup_can_be_retried(tmp_path, monkeypatch):
    from rsi_harness.cluster.bluevela import runtime as module

    runtime = _oracle_runtime(tmp_path, monkeypatch, multi=True)
    runtime.initialize()
    runtime.stage_solution()
    solution = runtime._oracle_solution
    broker = runtime.work_broker
    remove = module.shutil.rmtree

    def denied(path, *args, **kwargs):
        if path == solution:
            raise PermissionError("cannot remove disposable solution")
        return remove(path, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.shutil, "rmtree", denied)
            with pytest.raises(PermissionError):
                runtime.remove_oracle_solution()
        assert runtime.work_broker is None
        assert not (broker.root / "READY.json").exists()
        assert runtime._oracle_solution == solution
        assert (solution / "solve.sh").is_file()
        runtime.remove_oracle_solution()
        assert runtime._oracle_solution is None
        assert not solution.exists()
        assert (runtime.plan.task.source_dir / "solution/solve.sh").is_file()
    finally:
        runtime.close_multinode()


def test_multinode_oracle_keeps_solution_when_workers_cannot_stop(
    tmp_path, monkeypatch
):
    runtime = _oracle_runtime(tmp_path, monkeypatch, multi=True)
    runtime.control.mkdir(parents=True)
    runtime.stage_solution()
    solution = runtime._oracle_solution

    def stop():
        raise InfrastructureError("multi-node broker did not stop")

    broker = SimpleNamespace(stop=stop)
    runtime.work_broker = broker
    with pytest.raises(InfrastructureError, match="broker did not stop"):
        runtime.remove_oracle_solution()
    assert runtime.work_broker is broker
    assert runtime._oracle_solution == solution
    assert (solution / "solve.sh").is_file()


@pytest.mark.parametrize("sandbox", [False, True])
def test_no_network_oracle_work_submits_through_the_socket(
    tmp_path, monkeypatch, sandbox
):
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={"agent": plan.task.agent.model_copy(update={"name": "oracle"})}
            )
        }
    )
    assert plan.task.agent.network.mode == "no-network"
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("slurm", {"USER": "alice"}),
            run_id="oracle-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )
    endpoint = tmp_path / "endpoint"
    if sandbox:
        runtime.work_sandbox = endpoint
    composition = object.__new__(NativeEngineComposition)
    composition.runtime = runtime
    composition.provider_urls = _agent_provider_urls(None, plan, None)
    composition.agent = SimpleNamespace(install_hooks=lambda _request: None)
    composition.secrets = set()

    composition.install_hooks(plan, runtime.work_ref, "http://127.0.0.1:43123", "token")
    try:
        # No provider: only the submit socket, and the command runs unwrapped.
        assert sorted(path.name for path in (node_tmp / "network").iterdir()) == [
            "submit.sock"
        ]
        assert runtime._network_command(
            ("rsi-oracle",), network_mode="no-network", mount_agent_home=True
        ) == ("rsi-oracle",)
        base = runtime._base_command(
            devices=(),
            extra_binds=(),
            mount_workspace=True,
            mount_agent_home=True,
            network_mode="no-network",
            phase="work",
        )
        binds = set(zip(base, base[1:], strict=False))
        assert (
            "--bind",
            f"{node_tmp / 'network'}:/run/rsi-harness/network:ro",
        ) in binds
        assert (
            ("--bind", f"{endpoint}:/run/rsi-harness/sandbox:ro") in binds
        ) is sandbox
    finally:
        runtime.close_agent_network()
    assert not (node_tmp / "network/submit.sock").exists()


@pytest.mark.parametrize(
    "updates,expected,reason",
    [
        ({}, 1, None),
        ({"agent_exit_code": 7}, 1, "solve.sh exited 7"),
        ({}, 2, "expected 2"),
        ({"agent_exit_code": None}, 1, "no exit code"),
    ],
)
def test_native_engine_applies_oracle_result_checks(
    tmp_path, monkeypatch, updates, expected, reason
):
    from rsi_harness.cluster.bluevela import runtime as module
    from rsi_harness.errors import InfrastructureError
    from tests.factories import make_run_plan
    from tests.test_cli import _oracle_result

    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={"agent": plan.task.agent.model_copy(update={"name": "oracle"})}
            )
        }
    )
    plan.paths.root.mkdir(parents=True, exist_ok=True)
    result = _oracle_result(**updates)
    payload = SimpleNamespace(
        run_id="oracle",
        resume=None,
        options=CompileOptions(agent_name="oracle", max_submissions=expected),
    )
    composition = SimpleNamespace(
        backend=lambda: None,
        clock=None,
        coordinator_options=lambda: {},
        request=lambda: None,
        runtime=SimpleNamespace(close_multinode=lambda: None),
    )
    monkeypatch.setattr(module, "NativeEngineComposition", lambda *a, **k: composition)
    monkeypatch.setattr(
        module, "RunCoordinator", lambda **k: SimpleNamespace(run=lambda _: result)
    )
    monkeypatch.setattr(module, "validate_native_artifacts", lambda *a: plan.paths.root)
    if reason:
        with pytest.raises(InfrastructureError, match=reason):
            module.run_native_engine(payload, plan)
    else:
        module.run_native_engine(payload, plan)
    assert (plan.paths.root / "engine-result.json").is_file()


@pytest.mark.parametrize("appears_after", [0, 2, None])
def test_artifact_visibility_wait_is_bounded_and_requires_all_files(
    tmp_path, monkeypatch, appears_after
):
    from rsi_harness.cluster.bluevela import runtime as module
    from rsi_harness.errors import InfrastructureError
    from tests.factories import make_run_plan

    plan = make_run_plan(tmp_path)
    root = plan.paths.logs / "runs" / "oracle" / plan.task.task_id
    files = (
        "final_result.json",
        "agent_output.txt",
        "run_agent.log",
        "submissions/agent-1/report.json",
        "feedback/agent-1.log",
    )
    clock = [0.0]
    sleeps = []

    def expose():
        for name in files:
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("evidence")

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds
        if appears_after is not None and clock[0] >= appears_after:
            expose()

    monkeypatch.setattr(module, "recorded_submissions", lambda _: [])
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", sleep)
    if appears_after == 0:
        expose()
    if appears_after is None:
        with pytest.raises(InfrastructureError, match="missing artifacts"):
            module.validate_native_artifacts(plan, "oracle", visibility_timeout=3)
        assert sum(sleeps) == 3
    else:
        assert (
            module.validate_native_artifacts(plan, "oracle", visibility_timeout=3)
            == root
        )
        assert sum(sleeps) == appears_after
