from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
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
    JudgeLLMAuth,
    JudgeLLMRequest,
    RunGPUPlan,
    RunRequest,
    RunResult,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.sandbox_ledger import PruneRow


@dataclass
class FakeServices:
    agents: tuple[str, ...] = ("claude-code", "codex")
    requests: list[RunRequest] = field(default_factory=list)
    recovered: list[str | None] = field(default_factory=list)
    cleaned: list[tuple[str, bool]] = field(default_factory=list)
    pruned: list[tuple[float | None, bool]] = field(default_factory=list)
    resume_plans: list[tuple[str, object, bool]] = field(default_factory=list)
    resumed: list[object] = field(default_factory=list)
    resume_refusal: Exception | None = None
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

    def plan_resume(self, run_id: str, grants, *, dry_run: bool = False):
        self.resume_plans.append((run_id, grants, dry_run))
        if self.resume_refusal is not None:
            raise self.resume_refusal
        return _resume_context(run_id, grants)

    def resume(self, context) -> RunResult:
        self.resumed.append(context)
        return RunResult(
            run_id=context.run_id,
            status=RunStatus.COMPLETED,
            total_rounds=5,
            best_score=0.9,
            best_round="agent-5",
        )

    def prune_images(self, *, older_than_seconds, dry_run, confirm):
        from rsi_harness.runtime.sandbox_ledger import KEPT, REMOVED, WOULD_REMOVE

        self.pruned.append((older_than_seconds, dry_run))
        rows = tuple(
            PruneRow(image, action, reason, size)
            for image, action, reason, size in (
                (_pulled("a"), WOULD_REMOVE, "", 3 * 1024**2),
                (_pulled("b"), KEPT, "used by container cccccccccccc", 0),
            )
        )
        if dry_run:
            return rows
        if not confirm(rows):
            return None
        return (replace(rows[0], action=REMOVED), rows[1])


def _resume_context(run_id: str, grants):
    from rsi_harness.models import ManagedWorkdirVolume, RootfsSnapshotMode
    from rsi_harness.runtime.resume import ResumeContext
    from tests.factories import make_run_plan

    plan = make_run_plan(Path("/tmp/rsi-cli-test"))
    return ResumeContext(
        run_id=run_id,
        attempt=2,
        request=RunRequest(task_dir=Path("/tmp/rsi-cli-test/task")),
        sandbox_policy=None,
        plan=plan,
        gpu_plan=plan.gpu_plan,
        reports=(),
        charged_rounds=3,
        last_allocated_at=None,
        agent_seconds_used=40.0,
        agent_seconds_left=80.0,
        max_submissions=5,
        cooldown_seconds=0.0,
        grants=grants,
        base_image_ref=f"rsi-harness-rootfs:retained-work-{'a' * 64}",
        base_image_id=None,
        base_image_round_id="salvage",
        workdir_volume=ManagedWorkdirVolume(
            name=f"rsi-harness-workdir-{'1' * 64}",
            run_id=run_id,
            task_id=plan.task.task_id,
            target=plan.workdir,
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        ),
        salvage_pending=True,
        message="",
    )


def _pulled(letter: str):
    from rsi_harness.runtime.sandbox_ledger import PulledImage

    when = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)
    return PulledImage(
        image_id="sha256:" + letter * 64,
        references=(f"public.ecr.aws/docker/library/busybox:{letter}",),
        registry="public.ecr.aws",
        run_id="run-1",
        last_run_id="run-1",
        first_pulled_at=when,
        last_used_at=when,
    )


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


@pytest.mark.parametrize(
    ("agent", "model", "effort"),
    [
        ("codex", "gpt-test", "xhigh"),
        ("codex", "gpt-6-astra", "max"),
        ("claude-code", "claude-opus-5-5", "max"),
    ],
)
def test_run_preserves_every_cli_option_in_request(
    cli, tmp_path: Path, agent: str, model: str, effort: str
) -> None:
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
            agent,
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
            model,
            "--reasoning-effort",
            effort,
            "--agent-auth",
            "local",
            "--disable-stop-hook",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(services.requests) == 1
    request = services.requests[0]
    assert request.task_dir == task.resolve()
    assert request.agent_name == agent
    assert request.gpu_selectors == ("0", "2")
    assert request.options.agent_name == agent
    assert request.options.primary_reward == "accuracy"
    assert request.options.score_direction == "minimize"
    assert request.options.agent_timeout_seconds == 91.5
    assert request.options.max_submissions == 7
    assert request.options.cooldown_seconds == 2.25
    assert request.paths is not None
    assert request.paths.root.parent == data.resolve()
    assert request.paths.logs == logs.resolve()
    assert request.model == model
    assert request.reasoning_effort == effort
    assert request.agent_auth is AgentAuthSource.LOCAL
    assert request.options.disable_stop_hook is True
    assert "run-123" in result.output
    assert "completed" in result.output
    assert "agent-2" in result.output
    assert "0.75" in result.output


@pytest.mark.parametrize(
    ("agent", "model", "effort"),
    [
        ("codex", "gpt-test", "xhigh"),
        ("codex", "gpt-6-astra", "max"),
        ("claude-code", "claude-opus-5-5", "max"),
    ],
)
def test_cluster_run_routes_without_constructing_local_docker_services(
    cli,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    model: str,
    effort: str,
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
            agent,
            "--model",
            model,
            "--reasoning-effort",
            effort,
            "--agent-auth",
            "local",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert requests[0].task_dir == task.resolve()
    assert requests[0].agent_name == agent
    assert requests[0].options.agent_name == agent
    assert requests[0].model == model
    assert requests[0].reasoning_effort == effort
    assert requests[0].agent_auth is AgentAuthSource.LOCAL
    assert requests[0].dry_run is False
    assert local_services.requests == []
    assert "cluster-123" in result.output
    assert "701" in result.output


def test_cluster_run_passes_judge_llm_options_and_prints_the_judge_llm(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--judge-llm-* reach the cluster adapter (where, never the credential),
    and its Judge LLM events print as on a local run."""
    cli_module, local_services = cli
    task = tmp_path / "task"
    task.mkdir()
    requests = []

    class ClusterAdapter:
        def __init__(self, event_callback):
            self.event_callback = event_callback

        def run(self, request):
            requests.append(request)
            self.event_callback(
                "judge_llm",
                {
                    "model": "gpt-judge",
                    "effort": "high",
                    "credential": "api",
                    "reused": False,
                    "expires_at": None,
                },
            )
            return ClusterRunResult(
                run_id="cluster-123",
                status=RunStatus.COMPLETED,
                log_dir=(tmp_path / "cluster-logs").resolve(),
                job_ids=("701",),
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
            "--judge-llm-auth",
            "api",
            "--judge-llm-api-key-env",
            "MY_JUDGE_KEY",
        ],
    )

    assert result.exit_code == 0, result.output
    assert requests[0].judge_llm == JudgeLLMRequest(
        auth=JudgeLLMAuth.API, api_key_env="MY_JUDGE_KEY"
    )
    assert local_services.requests == []
    assert "Judge LLM: gpt-judge (effort high) via --judge-llm-auth api" in (
        result.output
    )


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
    plain_output = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", result.output)
    assert (
        "Ordered GPU pool authorized for this run; Work sees only its "
        "task-declared count" in re.sub(r"[│\s]+", " ", plain_output)
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
        ("sandbox", ("prune-images", "--dry-run")),
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


def test_prune_images_dry_run_lists_candidates_and_what_it_would_free(cli) -> None:
    cli_module, services = cli
    result = CliRunner().invoke(
        cli_module.app,
        ["sandbox", "prune-images", "--dry-run", "--older-than", "7d"],
    )

    assert result.exit_code == 0, result.output
    assert services.pruned == [(7 * 86400.0, True)]
    lines = result.output.splitlines()
    assert lines[0].split()[:4] == ["IMAGE", "SIZE", "LAST", "USED"]
    assert lines[1].split()[:8] == [
        "a" * 12,
        "3.0",
        "MiB",
        "2026-09-30",
        "12:00",
        "UTC",
        "would",
        "remove",
    ]
    assert lines[1].endswith("public.ecr.aws/docker/library/busybox:a")
    assert "kept" in lines[2] and "used by container cccccccccccc" in lines[2]
    assert lines[3] == "Would free: 3145728 bytes (3.0 MiB)"


def test_prune_images_asks_before_removing_unless_yes(cli) -> None:
    cli_module, services = cli
    runner = CliRunner()

    cancelled = runner.invoke(cli_module.app, ["sandbox", "prune-images"], input="n\n")
    confirmed = runner.invoke(cli_module.app, ["sandbox", "prune-images"], input="y\n")
    unattended = runner.invoke(cli_module.app, ["sandbox", "prune-images", "--yes"])

    assert cancelled.exit_code == 1
    assert "Remove 1 pulled image(s)? [y/N]" in cancelled.output
    assert "Prune cancelled" in cancelled.output
    assert confirmed.exit_code == 0, confirmed.output
    assert "Freed: 3145728 bytes (3.0 MiB)" in confirmed.output
    # The plan before the prompt, then the result under its own header.
    plan, result = confirmed.output.split("Result:\n")
    assert "would remove" in plan and "removed" not in plan
    assert result.splitlines()[0].startswith("IMAGE")
    assert "removed" in result.splitlines()[1]
    assert unattended.exit_code == 0, unattended.output
    assert "[y/N]" not in unattended.output and "Result:" not in unattended.output
    assert "removed" in unattended.output.splitlines()[1]
    assert services.pruned == [(None, False)] * 3


def test_the_cli_loads_the_sandbox_runtime_only_for_prune_images(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, rsi_harness.cli; print(sorted({'docker', "
            "'rsi_harness.runtime.sandbox_ledger'} & set(sys.modules)))",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout == "[]\n"


@pytest.mark.parametrize("duration", ["7", "0d", "1y", "-3h", "1.5h", "d"])
def test_prune_images_rejects_a_malformed_duration(cli, duration) -> None:
    cli_module, services = cli
    result = CliRunner().invoke(
        cli_module.app, ["sandbox", "prune-images", "--older-than", duration]
    )
    assert result.exit_code == 2
    assert "--older-than" in result.output
    assert services.pruned == []


def test_prune_images_with_an_empty_ledger(cli, monkeypatch) -> None:
    cli_module, services = cli
    monkeypatch.setattr(services, "prune_images", lambda **_: (), raising=False)
    result = CliRunner().invoke(cli_module.app, ["sandbox", "prune-images"])
    assert result.exit_code == 0
    assert result.output == "No pulled images are recorded in the pull ledger.\n"


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


def test_run_passes_judge_llm_options_without_the_credential(
    cli, tmp_path: Path
) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()
    home = tmp_path / "codex-home"

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            str(task),
            "--judge-llm-auth",
            "codex-login",
            "--judge-llm-codex-home",
            str(home),
            "--judge-llm-proxy",
            "http://proxy.example:3128",
        ],
    )

    assert result.exit_code == 0, result.output
    request = services.requests[0].judge_llm
    assert request.auth is JudgeLLMAuth.CODEX_LOGIN
    assert request.codex_home == home.resolve()
    assert request.proxy == "http://proxy.example:3128"
    assert "Judge LLM auth:" in result.output and "codex-login" in result.output

    result = CliRunner().invoke(
        cli_module.app,
        [
            "run",
            str(task),
            "--judge-llm-auth",
            "api",
            "--judge-llm-api-key-env",
            "MY_JUDGE_KEY",
            "--judge-llm-base-url",
            "https://models.example/v1",
        ],
    )

    assert result.exit_code == 0, result.output
    request = services.requests[1].judge_llm
    assert request.auth is JudgeLLMAuth.API
    assert (request.api_key_env, request.base_url) == (
        "MY_JUDGE_KEY",
        "https://models.example/v1",
    )
    # Only where the key comes from is part of the request, never the key.
    assert "key" not in request.model_dump_json().replace("api_key", "")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--judge-llm-proxy", "http://p:1"], "need --judge-llm-auth"),
        (
            ["--judge-llm-auth", "api", "--judge-llm-codex-home", "/tmp"],
            "only to a codex-login",
        ),
        (
            ["--judge-llm-auth", "codex-login", "--judge-llm-api-key-env", "K"],
            "only to an api",
        ),
    ],
)
def test_run_rejects_inconsistent_judge_llm_options(
    cli, tmp_path: Path, arguments, message
) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()

    result = CliRunner().invoke(cli_module.app, ["run", str(task), *arguments])

    assert result.exit_code == 2
    assert message in result.output
    assert services.requests == []


def test_run_console_names_the_judge_llm_source(capsys) -> None:
    from rsi_harness.cli import _RunConsole

    console = _RunConsole()
    console(
        "judge_llm",
        {
            "model": "gpt-judge",
            "effort": "medium",
            "credential": "api",
            "reused": False,
        },
    )
    console(
        "judge_llm",
        {"model": "gpt-test", "effort": "high", "credential": "api", "reused": True},
    )

    output = capsys.readouterr().out
    assert "Judge LLM: gpt-judge (effort medium) via --judge-llm-auth api" in output
    assert "Judge LLM: gpt-test (effort high) via the Work Agent's credential" in output


def test_run_console_warns_when_a_judge_keeps_its_last_login(capsys) -> None:
    from rsi_harness.cli import _RunConsole

    _RunConsole()(
        "judge_login_kept",
        {
            "round_id": "agent-2",
            "reason": "the Work Agent's login file /home/agent/.codex/auth.json "
            "is missing",
        },
    )

    output = capsys.readouterr().out
    assert (
        "Warning: Judge agent-2: the Work Agent's current login was not used "
        "(the Work Agent's login file /home/agent/.codex/auth.json is missing); "
        "the Judge keeps its last good copy, which stops working when its access "
        "token expires."
    ) in output


def test_run_console_warns_when_the_operators_login_was_not_refreshed(
    capsys,
) -> None:
    from rsi_harness.cli import _RunConsole

    _RunConsole()(
        "judge_login_refresh_failed",
        {"round_id": "agent-3", "reason": "Codex could not refresh it: timed out"},
    )

    output = capsys.readouterr().out
    assert (
        "Warning: Judge agent-3: the Judge LLM's Codex login was not kept fresh "
        "(Codex could not refresh it: timed out); the Judge refuses calls its "
        "access token would not outlive."
    ) in output


def test_run_console_reports_login_expiry_and_unused_judge_llm_options(
    capsys,
) -> None:
    from rsi_harness.cli import _RunConsole

    console = _RunConsole()
    console(
        "judge_llm",
        {
            "model": "gpt-judge",
            "effort": "low",
            "credential": "codex-login",
            "reused": False,
            "expires_at": 4102444800.0,
        },
    )
    console("judge_llm_ignored", None)

    output = capsys.readouterr().out
    assert (
        "via --judge-llm-auth codex-login; its login's access token expires 2"
    ) in output
    assert "refreshed in place before a Judge round would outlive it" in output
    assert "task declares no judge_model; the --judge-llm-* options are ignored" in (
        output
    )


def _round(round_id: str, status: SubmissionStatus = SubmissionStatus.COMPLETED):
    return SubmissionReport(
        round_id=round_id,
        status=status,
        rewards={"reward": 1.0},
        score=1.0,
        exit_code=0 if status is SubmissionStatus.COMPLETED else 1,
        duration_seconds=12.25,
    )


def _oracle_result(**updates) -> RunResult:
    values = {
        "run_id": "oracle-run",
        "status": RunStatus.COMPLETED,
        "total_rounds": 1,
        "best_score": 1.0,
        "best_round": "agent-1",
        "reports": (_round("agent-1"),),
        "agent_exit_code": 0,
        "artifacts_dir": Path("/logs/runs/oracle-run/task"),
    }
    values.update(updates)
    return RunResult(**values)


def _run_oracle(cli, tmp_path: Path, result: RunResult, *extra: str):
    cli_module, services = cli
    services.agents = ("codex", "oracle")
    task = tmp_path / "task"
    task.mkdir(exist_ok=True)
    services.run = lambda _request: result  # type: ignore[method-assign]
    return CliRunner().invoke(
        cli_module.app, ["run", str(task), "--agent", "oracle", *extra]
    )


def test_oracle_run_passes_and_prints_where_to_read_it(cli, tmp_path: Path) -> None:
    result = _run_oracle(cli, tmp_path, _oracle_result())

    assert result.exit_code == 0, result.output
    assert "Artifacts:" in result.output
    assert "/logs/runs/oracle-run/task" in result.output
    assert "Agent exit code:  0" in result.output
    assert "agent-1: completed score=1.0 exit_code=0 runtime=12.2s" in result.output
    assert "Oracle check passed" in result.output


@pytest.mark.parametrize(
    ("updates", "extra", "reason"),
    [
        ({"agent_exit_code": 2}, (), "solve.sh exited 2"),
        ({"agent_exit_code": None}, (), "solve.sh returned no exit code"),
        (
            {
                "status": RunStatus.NO_VALID_SUBMISSION,
                "reports": (_round("agent-1", SubmissionStatus.VERIFIER_ERROR),),
            },
            (),
            "run status is no_valid_submission; agent-1 is verifier_error",
        ),
        (
            {},
            ("--max-submissions", "2"),
            "1 Judge round(s) ran, expected 2",
        ),
        (
            {
                "total_rounds": 2,
                "reports": (
                    _round("agent-1"),
                    _round("agent-2", SubmissionStatus.INFRASTRUCTURE_ERROR),
                ),
            },
            ("--max-submissions", "2"),
            "agent-2 is infrastructure_error",
        ),
    ],
)
def test_oracle_run_fails_unless_every_round_completed(
    cli, tmp_path: Path, updates, extra, reason
) -> None:
    result = _run_oracle(cli, tmp_path, _oracle_result(**updates), *extra)

    assert result.exit_code == 1, result.output
    assert f"Oracle check failed: {reason}" in result.output


def test_oracle_run_with_n_completed_rounds_passes(cli, tmp_path: Path) -> None:
    result = _run_oracle(
        cli,
        tmp_path,
        _oracle_result(total_rounds=2, reports=(_round("agent-1"), _round("agent-2"))),
        "--max-submissions",
        "2",
    )

    assert result.exit_code == 0, result.output


def test_model_agent_exit_code_ignores_its_own_exit_code(cli, tmp_path: Path) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()
    services.run = lambda _request: _oracle_result(  # type: ignore[method-assign]
        agent_exit_code=137, status=RunStatus.NO_VALID_SUBMISSION
    )

    result = CliRunner().invoke(cli_module.app, ["run", str(task), "--agent", "codex"])

    assert result.exit_code == 0, result.output
    assert "Oracle check" not in result.output


def test_oracle_is_refused_on_a_cluster(cli, tmp_path: Path, monkeypatch) -> None:
    cli_module, services = cli
    task = tmp_path / "task"
    task.mkdir()
    monkeypatch.setattr(
        cli_module,
        "build_cluster_adapter",
        lambda *_a, **_k: pytest.fail("cluster adapter built"),
    )

    result = CliRunner().invoke(
        cli_module.app,
        ["run", str(task), "--agent", "oracle", "--cluster", "bluevela"],
    )

    assert result.exit_code == 2
    assert "--agent oracle runs locally only" in result.output
    assert services.requests == []


def test_resume_dry_run_prints_the_next_attempt_and_changes_nothing(cli) -> None:
    cli_module, services = cli

    result = CliRunner().invoke(
        cli_module.app,
        [
            "resume",
            "run-1",
            "--add-time",
            "60",
            "--add-submissions",
            "2",
            "--cooldown",
            "5",
            "--gpus",
            "2,3",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    ((run_id, grants, dry_run),) = services.resume_plans
    assert (run_id, dry_run) == ("run-1", True)
    assert grants.add_time_seconds == 60.0
    assert grants.add_submissions == 2
    assert grants.cooldown_seconds == 5.0
    assert grants.gpu_selectors == ("2", "3")
    assert services.resumed == []
    assert "Resuming run run-1 as attempt 2" in result.output
    assert "runs first" in result.output
    assert "40s used, 80s left" in result.output
    assert "3 used, 2 left" in result.output
    assert "agent-4" in result.output
    assert f"retained-work-{'a' * 64}" in result.output
    assert f"rsi-harness-workdir-{'1' * 64}" in result.output
    assert "nothing was changed" in result.output


def test_resume_runs_the_planned_attempt_and_prints_its_result(cli) -> None:
    cli_module, services = cli

    result = CliRunner().invoke(cli_module.app, ["resume", "run-1"])

    assert result.exit_code == 0, result.output
    ((_, grants, dry_run),) = services.resume_plans
    assert not dry_run
    assert grants.add_time_seconds == 0.0
    assert grants.gpu_selectors is None
    (context,) = services.resumed
    assert context.run_id == "run-1"
    assert "Best round:" in result.output and "agent-5" in result.output


def test_resume_prints_a_refusal(cli) -> None:
    cli_module, services = cli
    services.resume_refusal = SetupError("run run-1 is still live")

    result = CliRunner().invoke(cli_module.app, ["resume", "run-1"])

    assert result.exit_code == 1
    assert "setup_error: run run-1 is still live" in result.output
    assert services.resumed == []


@pytest.mark.parametrize("option", ["--add-time", "--cooldown"])
def test_resume_refuses_an_infinite_grant(cli, option) -> None:
    cli_module, services = cli

    result = CliRunner().invoke(cli_module.app, ["resume", "run-1", option, "inf"])

    assert result.exit_code == 2
    assert "finite number" in result.output
    assert services.resume_plans == []


class _ResumingCluster:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.plans: list[tuple[str, object, bool]] = []
        self.resumed: list[object] = []

    def plan_resume(self, run_id, grants, *, dry_run=False):
        from types import SimpleNamespace

        self.plans.append((run_id, grants, dry_run))
        context = replace(
            _resume_context(run_id, grants),
            base_image_ref=None,
            base_image_round_id=None,
        )
        return SimpleNamespace(context=context, spec=SimpleNamespace(walltime="01:16"))

    def resume(self, plan):
        self.resumed.append(plan)
        return ClusterRunResult(
            run_id=plan.context.run_id,
            status=RunStatus.COMPLETED,
            log_dir=(self.tmp_path / "cluster-logs").resolve(),
            job_ids=("702",),
        )


@pytest.mark.parametrize("dry_run", [True, False])
def test_resume_on_a_cluster_plans_then_submits_the_attempt(
    cli, tmp_path: Path, monkeypatch, dry_run: bool
) -> None:
    cli_module, services = cli
    cluster = _ResumingCluster(tmp_path)
    names: list[str] = []
    monkeypatch.setattr(
        cli_module,
        "build_cluster_adapter",
        lambda name, event_callback=None: names.append(name) or cluster,
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "resume",
            "run-1",
            "--cluster",
            "bluevela",
            "--add-time",
            "60",
            *(["--dry-run"] if dry_run else []),
        ],
    )

    assert result.exit_code == 0, result.output
    assert names == ["bluevela"]
    ((run_id, grants, planned_dry),) = cluster.plans
    assert (run_id, grants.add_time_seconds, planned_dry) == ("run-1", 60.0, dry_run)
    assert services.resume_plans == []
    assert "Resuming run run-1 as attempt 2" in result.output
    assert "Walltime:" in result.output and "01:16" in result.output
    assert "Base image:" not in result.output
    if dry_run:
        assert cluster.resumed == []
        assert "nothing was changed" in result.output
    else:
        assert len(cluster.resumed) == 1
        assert "Job IDs:" in result.output and "702" in result.output


def test_resume_on_a_cluster_refuses_gpus(cli, monkeypatch) -> None:
    cli_module, _services = cli
    monkeypatch.setattr(
        cli_module,
        "build_cluster_adapter",
        lambda *_a, **_k: pytest.fail("cluster adapter built"),
    )

    result = CliRunner().invoke(
        cli_module.app, ["resume", "run-1", "--cluster", "bluevela", "--gpus", "0"]
    )

    assert result.exit_code == 2
    assert "--gpus is local-only" in result.output
