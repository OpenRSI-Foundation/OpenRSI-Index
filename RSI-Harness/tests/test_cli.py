from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rsi_harness.cluster.base import ClusterRunResult
from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentAuthSource,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    RunGPUPlan,
    RunRequest,
    RunResult,
    RunStatus,
)


@dataclass
class FakeServices:
    agents: tuple[str, ...] = ("claude-code", "codex")
    requests: list[RunRequest] = field(default_factory=list)
    recovered: list[str | None] = field(default_factory=list)
    cleaned: list[tuple[str, bool]] = field(default_factory=list)
    event_callback: Callable[[str, object], None] | None = None

    def available_agents(self) -> tuple[str, ...]:
        return self.agents

    def run(self, request: RunRequest) -> RunResult:
        self.requests.append(request)
        if self.event_callback is not None:
            devices = tuple(
                GPUDevice(index=index, uuid=f"GPU-{letter}", name="Test GPU")
                for index, letter in enumerate(("a", "b", "c", "d"))
            )
            self.event_callback(
                "gpu_plan",
                RunGPUPlan(
                    authorized_pool=GPUAllocation(devices=devices),
                    work=GPUAllocation(devices=devices[:2]),
                    judge=GPUAllocation(devices=devices[2:]),
                    judge_mode=JudgeGPUMode.DISJOINT,
                ),
            )
        return RunResult(
            run_id="run-123",
            status=RunStatus.COMPLETED,
            total_rounds=2,
            best_score=0.75,
            best_round="agent-2",
        )

    def recover(self, run_id: str | None) -> tuple[str, ...]:
        self.recovered.append(run_id)
        return ("run-123",) if run_id is None else (run_id,)

    def cleanup(self, run_id: str, *, delete_workspace: bool) -> None:
        self.cleaned.append((run_id, delete_workspace))


@pytest.fixture
def cli(monkeypatch, tmp_path: Path):
    from rsi_harness import cli as cli_module

    monkeypatch.chdir(tmp_path)
    services = FakeServices()

    def build(*, event_callback=None, **_):
        services.event_callback = event_callback
        return services

    monkeypatch.setattr(cli_module, "build_runtime_services", build)
    return cli_module, services


def test_run_preserves_every_cli_option_in_request(cli, tmp_path: Path) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()
    data = tmp_path / "runtime-data"
    logs = tmp_path / "runtime-logs"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            str(task),
            "--agent",
            "codex",
            "--gpus",
            "0,2",
            "--primary-reward",
            "accuracy",
            "--score-direction",
            "minimize",
            "--timeout",
            "91.5",
            "--max-submissions",
            "7",
            "--cooldown",
            "2.25",
            "--data-root",
            str(data),
            "--logs-root",
            str(logs),
            "--model",
            "gpt-test",
            "--reasoning-effort",
            "xhigh",
            "--agent-auth",
            "local",
            "--disable-stop-hook",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(services.requests) == 1
    request = services.requests[0]
    assert request.task_dir == task.resolve()
    assert request.agent_name == "codex"
    assert request.gpu_selectors == ("0", "2")
    assert request.options.agent_name == "codex"
    assert request.options.primary_reward == "accuracy"
    assert request.options.score_direction == "minimize"
    assert request.options.agent_timeout_seconds == 91.5
    assert request.options.max_submissions == 7
    assert request.options.cooldown_seconds == 2.25
    assert request.paths is not None
    assert request.paths.root.parent == data.resolve()
    assert request.paths.logs == logs.resolve()
    assert request.model == "gpt-test"
    assert request.reasoning_effort == "xhigh"
    assert request.agent_auth is AgentAuthSource.LOCAL
    assert request.options.disable_stop_hook is True
    assert "run-123" in result.output
    assert "completed" in result.output
    assert "agent-2" in result.output
    assert "0.75" in result.output


def test_cluster_run_routes_without_constructing_local_docker_services(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_module, local_services = cli
    task = tmp_path / "task"
    task.mkdir()
    requests = []

    class ClusterAdapter:
        def run(self, request):
            requests.append(request)
            return ClusterRunResult(
                run_id="cluster-123",
                status=RunStatus.COMPLETED,
                log_dir=(tmp_path / "cluster-logs").resolve(),
                job_ids=("701",),
            )

    monkeypatch.setattr(
        cli_module,
        "build_cluster_adapter",
        lambda name, event_callback=None: ClusterAdapter(),
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            str(task),
            "--cluster",
            "bluevela",
            "--agent",
            "codex",
            "--model",
            "gpt-test",
            "--reasoning-effort",
            "xhigh",
            "--agent-auth",
            "local",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert requests[0].task_dir == task.resolve()
    assert requests[0].model == "gpt-test"
    assert requests[0].dry_run is False
    assert local_services.requests == []
    assert "cluster-123" in result.output
    assert "701" in result.output


def test_cluster_run_rejects_local_gpu_selectors_before_adapter_creation(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_module, local_services = cli
    task = tmp_path / "task"
    task.mkdir()
    builds: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "build_cluster_adapter",
        lambda name, event_callback=None: builds.append(name),
    )

    result = CliRunner().invoke(
        cli_module.app,
        ["run", str(task), "--cluster", "bluevela", "--gpus", "0,1"],
    )

    assert result.exit_code == 2
    assert "--gpus is local-only" in result.output
    assert builds == []
    assert local_services.requests == []


def test_cluster_dry_run_prints_resolved_submissions(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_module, local_services = cli
    task = tmp_path / "task"
    task.mkdir()

    class ClusterAdapter:
        def __init__(self, callback) -> None:
            self.callback = callback

        def run(self, request):
            self.callback(
                "dry_run",
                {
                    "run_id": "planned-1",
                    "run_dir": "/proj/runs/planned-1",
                    "image": "/proj/images/task.sif",
                    "cache_hit": False,
                    "resources": {
                        "work_gpus": 2,
                        "verifier_gpus": 2,
                        "total_gpus": 4,
                        "cpu_slots": 8,
                        "memory_mb": 65536,
                        "build_walltime": "02:00",
                        "run_walltime": "02:15",
                    },
                    "build_argv": ("bsub", "-J", "build"),
                    "run_argv": (
                        "bsub",
                        "-gpu",
                        "num=4/task:mode=exclusive_process",
                    ),
                    "binds": ("/proj",),
                },
            )
            return ClusterRunResult(
                run_id="planned-1",
                status=RunStatus.PREPARING,
                log_dir=Path("/proj/logs"),
            )

    monkeypatch.setattr(
        cli_module,
        "build_cluster_adapter",
        lambda name, event_callback=None: ClusterAdapter(event_callback),
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            str(task),
            "--cluster",
            "bluevela",
            "--dry-run",
            "--model",
            "gpt-test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Work GPUs:        2" in result.output
    assert "Verifier GPUs:    2" in result.output
    assert "Total GPUs:       4" in result.output
    assert "num=4/task:mode=exclusive_process" in result.output
    assert local_services.requests == []


def test_cluster_dry_run_prints_profile_driven_multinode_geometry(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_module, _local_services = cli
    task = tmp_path / "task"
    task.mkdir()

    class ClusterAdapter:
        def __init__(self, callback) -> None:
            self.callback = callback

        def run(self, _request):
            self.callback(
                "dry_run",
                {
                    "run_id": "planned-multi",
                    "run_dir": "/gpfs/runs/planned-multi",
                    "image": "/gpfs/images/task.sif",
                    "cache_hit": True,
                    "resources": None,
                    "multi_node": {
                        "work": {"gpu_count": 12, "node_count": 3},
                        "verifier": {"gpu_count": 4, "node_count": 1},
                        "total_nodes": 4,
                        "gpus_per_node": 4,
                        "cpu_slots_per_node": 6,
                        "memory_mb_per_node": 131072,
                        "shared_workspace_mb": 204800,
                        "node_tmp_mb": 40960,
                    },
                    "pool_policy": "ordered Work prefix; ordered Judge suffix",
                    "build_argv": ("bsub", "-J", "build"),
                    "run_argv": ("bsub", "-n", "24", "run.sh"),
                    "binds": ("/gpfs",),
                    "assets": (
                        {
                            "phase": "work",
                            "path": "/rsi-data/train/manifest.json",
                            "ready": True,
                            "detail": "size=42",
                        },
                        {
                            "phase": "judge",
                            "path": "/rsi-data/paloma/manifest.json",
                            "ready": False,
                            "detail": "missing",
                        },
                    ),
                },
            )
            return ClusterRunResult(
                run_id="planned-multi",
                status=RunStatus.PREPARING,
                log_dir=Path("/gpfs/logs"),
            )

    monkeypatch.setattr(
        cli_module,
        "build_cluster_adapter",
        lambda _name, event_callback=None: ClusterAdapter(event_callback),
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            str(task),
            "--cluster",
            "bluevela",
            "--dry-run",
            "--model",
            "gpt-test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Work nodes:       3" in result.output
    assert "Judge nodes:      1" in result.output
    assert "GPUs per node:    4" in result.output
    assert "CPU/node:         6" in result.output
    assert "Shared workspace: 204800 MiB" in result.output
    assert "Node scratch:     40960 MiB" in result.output
    assert "Assets ready:     1/2" in result.output
    assert "Missing asset:    judge:/rsi-data/paloma/manifest.json" in result.output


def test_run_prints_gpu_plan_event_once_with_uuids_only(cli, tmp_path: Path) -> None:
    cli_module, _services = cli
    task = tmp_path / "task"
    task.mkdir()

    result = CliRunner().invoke(
        cli_module.app,
        ["run", str(task), "--agent", "codex", "--gpus", "0,1,2,3"],
    )

    assert result.exit_code == 0, result.output
    expected = (
        "GPU pool: GPU-a,GPU-b,GPU-c,GPU-d\n"
        "Work GPUs: GPU-a,GPU-b\n"
        "Judge GPUs: GPU-c,GPU-d\n"
        "Judge GPU mode: disjoint\n"
    )
    assert result.output.count(expected) == 1
    assert (
        result.output.count(
            "Preparing task images (the first run may take several minutes)..."
        )
        == 1
    )
    assert "Test GPU" not in result.output
    assert "index" not in result.output


def test_run_prints_none_for_empty_gpu_plan_judge_allocation(
    cli, tmp_path: Path
) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()

    def run(request: RunRequest) -> RunResult:
        services.requests.append(request)
        assert services.event_callback is not None
        services.event_callback(
            "gpu_plan",
            RunGPUPlan(
                authorized_pool=GPUAllocation(
                    devices=(GPUDevice(index=9, uuid="GPU-a", name="Test GPU"),)
                ),
                work=GPUAllocation(
                    devices=(GPUDevice(index=9, uuid="GPU-a", name="Test GPU"),)
                ),
                judge=GPUAllocation(),
                judge_mode=JudgeGPUMode.FREEZE_ONLY,
            ),
        )
        return RunResult(run_id="run-123", status=RunStatus.COMPLETED)

    services.run = run  # type: ignore[method-assign]
    result = CliRunner().invoke(
        cli_module.app, ["run", str(task), "--agent", "codex", "--gpus", "0"]
    )

    assert result.exit_code == 0, result.output
    assert "Judge GPUs: none" in result.output


def test_run_console_prints_lifecycle_progress(capsys) -> None:
    from rsi_harness.cli import _RunConsole

    console = _RunConsole()
    console("images_ready", None)
    console("run_started", "run-live")
    console("work_started", "container-id")
    console(
        "agent_started",
        {
            "name": "codex",
            "timeout_seconds": 60,
            "output_path": "/logs/runs/run-live/task-1/agent_output.txt",
        },
    )
    console("judge_started", {"round_id": "agent-1"})
    console("judge_exec_started", {"round_id": "agent-1"})
    console(
        "judge_finished",
        {
            "round_id": "agent-1",
            "status": "completed",
            "score": 1.0,
            "runtime_seconds": 2.5,
        },
    )
    console("agent_finished", {"exit_code": 0, "timed_out": False})

    output = capsys.readouterr().out
    assert "Task images ready" in output
    assert "Run ID: run-live" in output
    assert "Work container started" in output
    assert "Running codex (timeout=60s)" in output
    assert "Agent output: /logs/runs/run-live/task-1/agent_output.txt" in output
    assert "Judge agent-1: snapshotting Work" in output
    assert "Judge agent-1: running /tests/test.sh" in output
    assert "Judge agent-1 finished: completed score=1.0 runtime=2.50s" in output
    assert "Agent finished: exit_code=0 timed_out=False" in output


def test_run_console_never_prints_raw_process_output(capsys) -> None:
    """Codex's exec-trace stdout and the verifier's stdout are both noisy
    and already fully captured on disk (agent_output.txt / the persisted
    submission record); the live console never echoes either one."""
    from rsi_harness.cli import _RunConsole

    console = _RunConsole()
    console("agent_output", "hidden agent output\n")
    console("judge_output", "hidden judge output\n")

    assert capsys.readouterr().out == ""


def test_run_help_explains_ordered_gpu_pool() -> None:
    from rsi_harness import cli as cli_module

    result = CliRunner().invoke(cli_module.app, ["run", "--help"])

    assert result.exit_code == 0, result.output
    assert (
        "Ordered GPU pool authorized for this run; Work sees only its "
        "task-declared count"
        in re.sub(r"[│\s]+", " ", result.output)
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    (
        (("missing", "--agent", "codex", "--gpus", "0"), "task directory"),
        (("task", "--agent", "unknown", "--gpus", "0"), "unknown Agent"),
        (("task", "--agent", "codex", "--gpus", "0,,2"), "GPU selector"),
        (("task", "--agent", "codex", "--gpus", "0,0"), "GPU selector"),
    ),
)
def test_run_rejects_local_inputs_before_runtime_mutation(
    cli, tmp_path: Path, arguments: tuple[str, ...], message: str
) -> None:
    cli_module, services = cli
    (tmp_path / "task").mkdir()
    resolved = tuple(
        str(tmp_path / value) if value in {"task", "missing"} else value
        for value in arguments
    )

    result = CliRunner().invoke(cli_module.app, ["run", *resolved])

    assert result.exit_code == 2
    assert message in result.output
    assert "Traceback" not in result.output
    assert services.requests == []


def test_run_reports_invalid_tilde_path_as_concise_typed_setup_error(cli) -> None:
    cli_module, services = cli
    missing_user = "rsi-harness-definitely-no-such-user"

    result = CliRunner().invoke(
        cli_module.app,
        ["run", f"~{missing_user}/task", "--agent", "codex", "--gpus", "0"],
    )

    assert result.exit_code == 1
    assert "setup_error:" in result.output
    assert "normalize run paths" in result.output
    assert missing_user not in result.output
    assert "Traceback" not in result.output
    assert services.requests == []


def test_run_redacts_verbose_path_normalization_failure(
    cli, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli_module, services = cli
    secret = "PATH-NORMALIZATION-TOP-SECRET"
    original_expanduser = Path.expanduser

    def fail_secret_path(path: Path) -> Path:
        if str(path).startswith("~"):
            raise RuntimeError(f"credential={secret}")
        return original_expanduser(path)

    monkeypatch.setattr(Path, "expanduser", fail_secret_path)
    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            "~/task",
            "--agent",
            "codex",
            "--gpus",
            "0",
            "--verbose",
        ],
    )

    assert result.exit_code == 1
    assert "SetupError" in result.output
    assert "[REDACTED]" in result.output
    assert secret not in result.output
    assert "Traceback" not in result.output
    assert services.requests == []


def test_run_redacts_default_and_verbose_errors(cli, tmp_path: Path) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()

    def fail(_request: RunRequest) -> RunResult:
        raise RuntimeError("Authorization: Bearer TOP-SECRET")

    services.run = fail  # type: ignore[method-assign]
    concise = CliRunner().invoke(
        cli_module.app,
        ["run", str(task), "--agent", "codex", "--gpus", "0"],
    )
    verbose = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            str(task),
            "--agent",
            "codex",
            "--gpus",
            "0",
            "--verbose",
        ],
    )

    assert concise.exit_code != 0
    assert "RuntimeError" not in concise.output
    assert "TOP-SECRET" not in concise.output
    assert verbose.exit_code != 0
    assert "RuntimeError" in verbose.output
    assert "[REDACTED]" in verbose.output
    assert "TOP-SECRET" not in verbose.output
    assert "Traceback" not in verbose.output


def test_run_returns_nonzero_for_failed_terminal_result(cli, tmp_path: Path) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()

    services.run = lambda _request: RunResult(  # type: ignore[method-assign]
        run_id="failed-run", status=RunStatus.FAILED
    )
    result = CliRunner().invoke(
        cli_module.app,
        ["run", str(task), "--agent", "codex", "--gpus", "0"],
    )

    assert result.exit_code != 0
    assert "Status:" in result.output and "failed" in result.output
    assert "Traceback" not in result.output


def test_run_prints_concise_typed_public_error(cli, tmp_path: Path) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()

    def fail(_request):
        raise SetupError("unknown GPU selector '9'")

    services.run = fail  # type: ignore[method-assign]
    result = CliRunner().invoke(
        cli_module.app,
        ["run", str(task), "--agent", "codex", "--gpus", "9"],
    )

    assert result.exit_code != 0
    assert "setup_error: unknown GPU selector '9'" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("phase", ("construction", "registry"))
@pytest.mark.parametrize("verbose", (False, True))
def test_run_redacts_service_construction_and_registry_failures(
    cli, tmp_path: Path, monkeypatch, phase: str, verbose: bool
) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()
    secret = "BOUNDARY-TOP-SECRET"

    if phase == "construction":
        def fail_build(**_kwargs):
            raise RuntimeError(f"Authorization: Bearer {secret}")

        monkeypatch.setattr(cli_module, "build_runtime_services", fail_build)
    else:
        def fail_registry():
            raise RuntimeError(f"registry access_token={secret}")

        services.available_agents = fail_registry  # type: ignore[method-assign]

    arguments = ["run", str(task), "--agent", "codex", "--gpus", "0"]
    if verbose:
        arguments.append("--verbose")
    result = CliRunner().invoke(cli_module.app, arguments)

    assert result.exit_code != 0
    assert secret not in result.output
    assert "Traceback" not in result.output
    assert ("RuntimeError" in result.output) is verbose
    assert ("[REDACTED]" in result.output) is verbose
    if not verbose:
        assert "run failed; use --verbose" in result.output


@pytest.mark.parametrize(
    ("command", "arguments"),
    (
        ("recover", ("run-one",)),
        ("cleanup", ("run-one",)),
    ),
)
def test_recovery_commands_expand_home_before_service_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    arguments: tuple[str, ...],
) -> None:
    from rsi_harness import cli as cli_module

    home = tmp_path / "home"
    home.mkdir()
    captured: list[object] = []
    services = FakeServices()

    def build(*, roots):
        captured.append(roots)
        return services

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(cli_module, "build_runtime_services", build)

    result = CliRunner().invoke(
        cli_module.app,
        [
            command,
            *arguments,
            "--data-root",
            "~/runtime-data",
            "--logs-root",
            "~/runtime-logs",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == [
        cli_module.RuntimeRoots(
            data=(home / "runtime-data").resolve(),
            logs=(home / "runtime-logs").resolve(),
        )
    ]


def test_recover_supports_one_or_all_and_cleanup_requires_confirmation(
    cli,
) -> None:
    cli_module, services = cli
    runner = CliRunner()

    assert runner.invoke(cli_module.app, ["recover"]).exit_code == 0
    assert runner.invoke(cli_module.app, ["recover", "run-one"]).exit_code == 0
    cancelled = runner.invoke(
        cli_module.app,
        ["cleanup", "run-one", "--delete-workspace"],
        input="n\n",
    )
    confirmed = runner.invoke(
        cli_module.app,
        ["cleanup", "run-one", "--delete-workspace", "--yes"],
    )

    assert services.recovered == [None, "run-one"]
    assert cancelled.exit_code != 0
    assert services.cleaned == [("run-one", True)]
    assert confirmed.exit_code == 0


def test_visualize_runs_stock_rsi_loop_app_with_generated_task_metadata(
    cli, tmp_path: Path, monkeypatch
) -> None:
    cli_module, _services = cli
    calls: list[tuple[object, object, object, object]] = []
    stock_app = object()

    def create_app(runs_dir, *, tasks_dir=None):
        calls.append((runs_dir, tasks_dir, None, None))
        return stock_app

    def run(server_app, *, host, port):
        calls.append((server_app, None, host, port))

    monkeypatch.setattr("rsi_loop.visualizer.server.create_app", create_app)
    monkeypatch.setattr("uvicorn.run", run)
    data = tmp_path / "data"
    logs = tmp_path / "logs"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "visualize",
            "--data-root",
            str(data),
            "--logs-root",
            str(logs),
            "--host",
            "0.0.0.0",
            "--port",
            "8123",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == [
        (
            logs.resolve() / "runs",
            data.resolve() / "generated-tasks",
            None,
            None,
        ),
        (stock_app, None, "0.0.0.0", 8123),
    ]
