"""The oracle Agent: a task's solution/solve.sh run on the real run path."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopAgentAdapter,
    rsi_loop_agent_environment,
    rsi_loop_oneshot,
    validate_agent_config,
)
from rsi_harness.models import (
    AgentAuthSource,
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunResult,
    CompileOptions,
    ContainerRef,
    NetworkPolicy,
    RunRequest,
)
from rsi_harness.runtime import production
from rsi_loop.harness.agent import create_agent, get_agent_class, oracle
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan, write_harbor_task

LAUNCHER = Path(oracle.__file__).resolve().parent / "oracle" / "rsi-oracle"
WORK = ContainerRef(container_id="work-1", role="work")
CREDENTIALED = RSILoopConfig(
    http_proxy="http://proxy.internal:8080",
    https_proxy="http://proxy.internal:8443",
    no_proxy="localhost",
    agent_api_key="sk-must-not-reach-work",
    agent_api_base_url="http://203.0.113.7:8000/v1",
    agent_model="gpt-test",
    agent_extra_env={"OPENAI_API_KEY": "sk-extra", "EXTRA": "value"},
    nodejs_mirror_url="https://nodejs.mirror.test",
)


class StagingRuntime:
    """Record copies (keeping their content) and execs, like the Work runtime."""

    def __init__(self, tmp_path: Path, *, exit_code: int = 0) -> None:
        self.copies: list[tuple[PurePosixPath, Path]] = []
        self.executions: list[dict[str, object]] = []
        self._tmp = tmp_path / "copied"
        self._tmp.mkdir()
        self._exit_code = exit_code

    def copy_to(self, container, source: Path, target: PurePosixPath) -> None:
        assert container == WORK
        kept = self._tmp / f"{len(self.copies)}-{source.name}"
        shutil.copyfile(source, kept)
        self.copies.append((target, kept))

    def exec(self, container, command, **options) -> AgentRunResult:
        assert container == WORK
        self.executions.append({"command": command, **options})
        return AgentRunResult(exit_code=self._exit_code, output="exec output")


def _oracle_plan(tmp_path: Path, *, network: str = "no-network"):
    plan = make_run_plan(tmp_path)
    policy = NetworkPolicy(
        mode=network,
        allowlist=("example.test",) if network == "allowlist" else (),
    )
    agent = plan.task.agent.model_copy(update={"name": "oracle", "network": policy})
    return plan.model_copy(
        update={"task": plan.task.model_copy(update={"agent": agent})}
    )


def _write_solution(source_dir: Path) -> Path:
    solution = source_dir / "solution"
    (solution / "data").mkdir(parents=True)
    (solution / "solve.sh").write_text("#!/bin/bash\necho solved > answer\n")
    (solution / "solve.sh").chmod(0o644)
    (solution / "data" / "patch.diff").write_text("diff\n")
    (solution / "data" / "tool").write_text("#!/bin/sh\n")
    (solution / "data" / "tool").chmod(0o750)
    (solution / "data" / "link").symlink_to("patch.diff")
    return solution


def _hook_request(plan) -> AgentHookRequest:
    return AgentHookRequest(
        run_plan=plan, container=WORK, submit_url="http://control", token="tok"
    )


def test_oracle_is_registered_without_model_cli_or_credential() -> None:
    agent_class = get_agent_class("oracle")

    assert "oracle" in RSILoopAgentAdapter(RSILoopConfig()).available_agents()
    assert agent_class.runs_task_solution is True
    assert agent_class.install_cmds == []
    assert agent_class.requires_model is False
    assert agent_class.reasoning_efforts == frozenset()
    assert agent_class.extra_launchers == ()
    assert agent_class.image_files == {
        "/usr/local/bin/rsi-oracle": ("harness/agent/oracle/rsi-oracle", 0o755)
    }
    agent = create_agent("oracle", RSILoopConfig())
    assert agent.format_oneshot_cmd(model="m") is None
    # No model, key or endpoint is needed to set it up.
    assert validate_agent_config(RSILoopConfig(), "oracle").name == "oracle"
    # Every model Agent stays a model Agent.
    for name in RSILoopAgentAdapter(RSILoopConfig()).available_agents():
        if name != "oracle":
            assert get_agent_class(name).runs_task_solution is False


def test_oracle_command_quotes_the_working_directory() -> None:
    agent = create_agent("oracle", RSILoopConfig())

    assert agent.format_run_cmd("/p", cwd="/workspace") == (
        "rsi-oracle /workspace /solution"
    )
    assert agent.format_run_cmd("/p", cwd="/my work") == (
        "rsi-oracle '/my work' /solution"
    )


def test_oracle_environment_carries_only_proxy_settings() -> None:
    agent = create_agent("oracle", CREDENTIALED)

    assert rsi_loop_agent_environment(CREDENTIALED, agent, "gpt-test") == {
        "http_proxy": "http://proxy.internal:8080",
        "HTTP_PROXY": "http://proxy.internal:8080",
        "https_proxy": "http://proxy.internal:8443",
        "HTTPS_PROXY": "http://proxy.internal:8443",
        "no_proxy": "localhost",
        "NO_PROXY": "localhost",
    }


def test_a_judge_cannot_call_the_oracle(tmp_path: Path) -> None:
    oneshot = rsi_loop_oneshot(CREDENTIALED, _oracle_plan(tmp_path).task.agent)

    assert oneshot["command"] is None
    assert "cannot run one-shot" in oneshot["unsupported"]
    assert "sk-must-not-reach-work" not in str(oneshot)


@pytest.mark.parametrize("network", ["no-network", "allowlist", "public"])
def test_oracle_pins_no_provider_endpoint(tmp_path: Path, network: str) -> None:
    definition = _oracle_plan(tmp_path, network=network).task

    # Not even the no-network refusal for a missing provider endpoint.
    assert production._agent_endpoints(definition, RSILoopConfig()) == ()
    assert production._agent_endpoints(definition, CREDENTIALED) == ()


@pytest.mark.parametrize(("max_submissions", "count"), [(None, "1"), (3, "3")])
def test_prepare_runs_the_launcher_with_the_submission_count(
    tmp_path: Path, max_submissions: int | None, count: str
) -> None:
    prepared = RSILoopAgentAdapter(CREDENTIALED).prepare(
        AgentPrepareRequest(
            run_plan=_oracle_plan(tmp_path),
            prompt_path=tmp_path / "prompt.md",
            max_submissions=max_submissions,
        )
    )

    assert prepared.agent_name == "oracle"
    assert prepared.command == (
        "/bin/bash",
        "-c",
        "rsi-oracle /workspace /solution",
    )
    environment = dict(prepared.environment)
    assert environment["RSI_ORACLE_SUBMISSIONS"] == count
    assert environment["RSI_ORACLE_SOLUTION_CLEANUP"] == "launcher"
    assert "sk-must-not-reach-work" not in str(prepared.environment)
    assert "OPENAI_API_KEY" not in environment
    assert "EXTRA" not in environment


def test_install_hooks_stages_the_solution_as_root_before_the_exec(
    tmp_path: Path,
) -> None:
    plan = _oracle_plan(tmp_path)
    _write_solution(plan.task.source_dir)
    runtime = StagingRuntime(tmp_path)

    RSILoopAgentAdapter(CREDENTIALED, runtime=runtime).install_hooks(
        _hook_request(plan)
    )

    assert [target for target, _ in runtime.copies] == [
        PurePosixPath("/tmp/rsi-oracle-solution.tar")
    ]
    with tarfile.open(runtime.copies[0][1]) as archive:
        members = {member.name: member for member in archive.getmembers()}
        assert set(members) == {
            "data",
            "data/link",
            "data/patch.diff",
            "data/tool",
            "solve.sh",
        }
        assert members["data/link"].issym()
        assert members["data/link"].linkname == "patch.diff"
        assert members["data/tool"].mode & 0o777 == 0o750
        assert all(m.uid == 0 and m.uname == "" for m in members.values())
        solve = archive.extractfile("solve.sh")
        assert solve is not None and b"echo solved" in solve.read()
    [install] = runtime.executions
    assert install["user"] == "root"
    script = install["command"][-1]
    assert install["command"][:2] == ("/bin/sh", "-c")
    for step in (
        "rm -rf /solution",
        "install -d -m 0755 /solution",
        "tar -xf /tmp/rsi-oracle-solution.tar -C /solution --no-same-owner",
        "chown -hR agent:agent /solution",
        "chmod u+x /solution/solve.sh",
        "rm -f /tmp/rsi-oracle-solution.tar",
    ):
        assert step in script


def test_install_script_extracts_an_owned_runnable_solution(tmp_path: Path) -> None:
    """The staged archive and script, run against a scratch root."""
    plan = _oracle_plan(tmp_path)
    _write_solution(plan.task.source_dir)
    runtime = StagingRuntime(tmp_path)
    RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime).install_hooks(
        _hook_request(plan)
    )
    root = tmp_path / "root"
    (root / "tmp").mkdir(parents=True)
    shutil.copyfile(runtime.copies[0][1], root / "tmp" / "rsi-oracle-solution.tar")
    script = runtime.executions[0]["command"][-1]
    me = f"{os.getuid()}:{os.getgid()}"
    script = (
        script.replace("/solution", f"{root}/solution")
        .replace("/tmp/rsi-oracle-solution.tar", f"{root}/tmp/rsi-oracle-solution.tar")
        .replace("agent:agent", me)
    )

    subprocess.run(["/bin/sh", "-c", script], check=True)

    solution = root / "solution"
    assert os.access(solution / "solve.sh", os.X_OK)
    assert (solution / "data" / "link").is_symlink()
    assert (solution / "data" / "patch.diff").read_text() == "diff\n"
    assert not (root / "tmp" / "rsi-oracle-solution.tar").exists()


def test_failed_solution_staging_is_an_infrastructure_error(tmp_path: Path) -> None:
    plan = _oracle_plan(tmp_path)
    _write_solution(plan.task.source_dir)
    adapter = RSILoopAgentAdapter(
        RSILoopConfig(), runtime=StagingRuntime(tmp_path, exit_code=2)
    )

    with pytest.raises(InfrastructureError, match="staging the task solution"):
        adapter.install_hooks(_hook_request(plan))


def test_model_agents_get_no_solution(tmp_path: Path) -> None:
    plan = make_run_plan(tmp_path)
    _write_solution(plan.task.source_dir)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={"install_stop_hook": False}
                    )
                }
            )
        }
    )
    runtime = StagingRuntime(tmp_path)

    RSILoopAgentAdapter(CREDENTIALED, runtime=runtime).install_hooks(
        _hook_request(plan)
    )

    assert runtime.copies == []
    assert runtime.executions == []


class _NoDocker(Exception):
    pass


def _services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def reached(*_args: object, **_kwargs: object) -> None:
        raise _NoDocker

    class Inventory:
        list_devices = reached

    services = production.ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=Inventory(),
        rsi_loop_config=RSILoopConfig(),
    )
    monkeypatch.setattr(services, "_client", reached)
    return services


def _oracle_request(task: Path) -> RunRequest:
    return RunRequest(
        task_dir=task,
        agent_name="oracle",
        options=CompileOptions(agent_name="oracle"),
    )


@pytest.mark.parametrize("solution", ["missing", "directory", "symlink"])
def test_production_refuses_a_task_without_solve_sh_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, solution: str
) -> None:
    services = _services(tmp_path, monkeypatch)
    task = write_harbor_task(tmp_path)
    if solution == "directory":
        (task / "solution" / "solve.sh").mkdir(parents=True)
    elif solution == "symlink":
        (task / "solution").mkdir()
        (task / "elsewhere.sh").write_text("#!/bin/sh\n")
        (task / "solution" / "solve.sh").symlink_to(task / "elsewhere.sh")

    with pytest.raises(SetupError, match="runs solution/solve.sh"):
        services.run(_oracle_request(task))
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()


def test_production_refuses_a_model_access_judge_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = _services(tmp_path, monkeypatch)
    task = write_harbor_task(tmp_path)
    _write_solution(task)
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text() + "\n[metadata.rsi_harness.verifier]\nmodel_access = true\n"
    )

    with pytest.raises(SetupError, match="model_access.*oracle"):
        services.run(_oracle_request(task))
    assert not (tmp_path / "data").exists()


def test_production_refuses_agent_auth_for_the_oracle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = _services(tmp_path, monkeypatch)
    task = write_harbor_task(tmp_path)
    _write_solution(task)
    request = _oracle_request(task).model_copy(
        update={"agent_auth": AgentAuthSource.LOCAL}
    )

    with pytest.raises(SetupError, match="uses no credential"):
        services.run(request)


def test_production_lets_an_oracle_task_with_solve_sh_reach_runtime_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = _services(tmp_path, monkeypatch)
    task = write_harbor_task(tmp_path)
    _write_solution(task)

    # Past every static check: the first runtime call is GPU inventory.
    with pytest.raises(_NoDocker):
        services.run(_oracle_request(task))


def _launcher_env(tmp_path: Path, *, submit_status: int = 0) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    submit = bin_dir / "rsi-submit"
    submit.write_text(
        "#!/bin/sh\n"
        f'echo "submit cwd=$(pwd)" >> {tmp_path}/submits\n'
        f"exit {submit_status}\n"
    )
    submit.chmod(0o755)
    return {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}


def _solve(tmp_path: Path, body: str) -> Path:
    solution = tmp_path / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text(f"#!/bin/bash\n{body}\n")
    (solution / "solve.sh").chmod(0o755)
    return solution


@pytest.mark.parametrize(("count", "expected"), [(None, 1), ("2", 2)])
def test_launcher_solves_in_workdir_removes_solution_and_submits(
    tmp_path: Path, count: str | None, expected: int
) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    solution = _solve(
        tmp_path,
        'echo "$DEBIAN_FRONTEND" > answer\nls "$0" > /dev/null',
    )
    env = _launcher_env(tmp_path)
    if count is not None:
        env["RSI_ORACLE_SUBMISSIONS"] = count

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(workdir), str(solution)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (workdir / "answer").read_text() == "noninteractive\n"
    assert not solution.exists()
    assert (tmp_path / "submits").read_text().splitlines() == [
        f"submit cwd={workdir}"
    ] * expected
    assert "rsi-oracle: solve.sh exited 0" in completed.stdout
    assert f"rsi-oracle: submission {expected}/{expected}" in completed.stdout


def test_launcher_submits_after_a_failed_solve_and_returns_its_status(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    solution = _solve(tmp_path, "exit 7")

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(workdir), str(solution)],
        env={**_launcher_env(tmp_path, submit_status=3), "RSI_ORACLE_SUBMISSIONS": "2"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 7
    assert not solution.exists()
    # A failing rsi-submit does not stop the next submission.
    assert len((tmp_path / "submits").read_text().splitlines()) == 2
    assert "rsi-oracle: rsi-submit exited 3" in completed.stderr


def test_launcher_defers_distributed_solution_removal_to_native_submit(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    solution = _solve(tmp_path, "echo solved > answer")
    env = _launcher_env(tmp_path)
    # The native submit lifecycle checks idle remote workers before removal.
    submit = tmp_path / "bin/rsi-submit"
    submit.write_text(
        "#!/bin/sh\n"
        f"test -f '{solution}/solve.sh' || exit 4\n"
        f"rm -rf -- '{solution}'\n"
        "echo native-cleanup-before-judge\n"
    )
    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(workdir), str(solution)],
        env={**env, "RSI_ORACLE_SOLUTION_CLEANUP": "submit"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (workdir / "answer").read_text() == "solved\n"
    assert not solution.exists()
    assert "native-cleanup-before-judge" in completed.stdout
    assert "rsi-submit exited" not in completed.stderr


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory modes")
def test_launcher_empties_a_solution_it_cannot_unlink(tmp_path: Path) -> None:
    """A non-root exec user cannot unlink /solution from /: emptying is enough."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    parent = tmp_path / "root"
    parent.mkdir()
    solution = parent / "solution"
    solution.mkdir()
    (solution / "solve.sh").write_text("#!/bin/bash\ntouch solved\n")
    (solution / "solve.sh").chmod(0o755)
    (solution / "data").mkdir()
    (solution / "data" / "answer").write_text("42\n")
    parent.chmod(0o555)
    try:
        completed = subprocess.run(
            ["/bin/bash", str(LAUNCHER), str(workdir), str(solution)],
            env=_launcher_env(tmp_path),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        parent.chmod(0o755)

    assert completed.returncode == 0, completed.stderr
    assert (workdir / "solved").exists()
    assert solution.is_dir() and list(solution.iterdir()) == []
    assert f"rsi-oracle: emptied {solution}" in completed.stdout
    assert (tmp_path / "submits").exists()


def test_launcher_without_solve_sh_still_submits_and_fails(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(workdir), str(tmp_path / "missing")],
        env=_launcher_env(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 127
    assert (tmp_path / "submits").exists()


@pytest.mark.parametrize("count", ["0", "x", "01", "-1"])
def test_launcher_refuses_a_bad_submission_count(tmp_path: Path, count: str) -> None:
    workdir = tmp_path / "work"
    workdir.mkdir()
    solution = _solve(tmp_path, "touch ran")

    completed = subprocess.run(
        ["/bin/bash", str(LAUNCHER), str(workdir), str(solution)],
        env={**_launcher_env(tmp_path), "RSI_ORACLE_SUBMISSIONS": count},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert not (workdir / "ran").exists()
    assert not (tmp_path / "submits").exists()
