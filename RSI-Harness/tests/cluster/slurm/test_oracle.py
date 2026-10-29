from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.bluevela.runtime import (
    _agent_provider_urls,
    _ClusterAgentAdapter,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentAuthSource,
    AgentHookRequest,
    AgentRunResult,
    CompileOptions,
)
from tests.cluster.bluevela.test_adapter import _profile
from tests.cluster.bluevela.test_runtime import _codex_runtime
from tests.cluster.slurm.test_adapter import RecordingSlurmScheduler
from tests.factories import DEFAULT_TASK_TOML, write_harbor_task


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
    "problem", ["lsf", "multi", "auth", "effort", "missing", "symlink", "model_access"]
)
def test_oracle_refuses_unsupported_inputs_before_jobs(tmp_path, problem):
    profile, request = setup_oracle(tmp_path, gpus=16 if problem == "multi" else 0)
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


def test_solution_is_writable_work_only_and_original_is_preserved(
    tmp_path, monkeypatch
):
    runtime = _codex_runtime(
        tmp_path,
        monkeypatch,
        agent_launcher="rsi-oracle",
        agent_binary=None,
        agent_version=None,
    )
    runtime.profile = load_cluster_profile("slurm", {"USER": "alice"})
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
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    outside.chmod(0o400)
    (solution / "link").symlink_to(outside)
    monkeypatch.setattr(
        runtime, "_run", lambda *a, **k: AgentRunResult(exit_code=0, output="")
    )
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
            environment={},
            extra_binds=(),
            mount_workspace=phase == "work",
            mount_agent_home=phase == "work",
            containall=True,
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
