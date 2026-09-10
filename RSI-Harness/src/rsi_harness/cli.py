"""Command-line entry points for production RSI Harness runs."""

from __future__ import annotations

import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

import typer

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import build_cluster_adapter
from rsi_harness.errors import HarnessError, SetupError
from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    RunGPUPlan,
    RunPaths,
    RunRequest,
    RunResult,
    RunStatus,
)
from rsi_harness.runtime.redaction import redact_text

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


class RuntimeServicesPort(Protocol):
    def available_agents(self) -> tuple[str, ...]: ...

    def run(self, request: RunRequest) -> RunResult: ...

    def recover(self, run_id: str | None) -> tuple[str, ...]: ...

    def cleanup(self, run_id: str, *, delete_workspace: bool) -> None: ...


@dataclass(frozen=True, slots=True)
class RuntimeRoots:
    data: Path
    logs: Path


def build_runtime_services(
    *,
    roots: RuntimeRoots,
    event_callback: Callable[[str, object], None] | None = None,
) -> RuntimeServicesPort:
    """Build the concrete Docker/NVIDIA/RSI Loop production runtime."""
    from rsi_harness.runtime.production import ProductionRuntimeServices

    return ProductionRuntimeServices(
        data_root=roots.data,
        logs_root=roots.logs,
        event_callback=event_callback,
    )


def _selectors(value: str) -> tuple[str, ...]:
    raw = value.split(",")
    selectors = tuple(item.strip() for item in raw)
    if not selectors or any(not item for item in selectors):
        raise typer.BadParameter("GPU selector list contains an empty value")
    if len(selectors) != len(set(selectors)):
        raise typer.BadParameter("GPU selector list contains a duplicate value")
    return selectors


def _services(
    data_root: Path,
    logs_root: Path,
    *,
    event_callback: Callable[[str, object], None] | None = None,
) -> RuntimeServicesPort:
    roots = RuntimeRoots(
        data=data_root.expanduser().resolve(),
        logs=logs_root.expanduser().resolve(),
    )
    if event_callback is None:
        return build_runtime_services(roots=roots)
    return build_runtime_services(roots=roots, event_callback=event_callback)


@dataclass(frozen=True, slots=True)
class _RunConsole:
    def __call__(self, name: str, value: object) -> None:
        if name == "gpu_plan" and isinstance(value, RunGPUPlan):
            typer.echo(f"GPU pool: {','.join(value.authorized_pool.uuids)}")
            typer.echo(f"Work GPUs: {','.join(value.work.uuids)}")
            judge = ",".join(value.judge.uuids) or "none"
            typer.echo(f"Judge GPUs: {judge}")
            typer.echo(f"Judge GPU mode: {value.judge_mode.value}")
            typer.echo(
                "Preparing task images (the first run may take several minutes)..."
            )
        elif name == "images_ready":
            typer.echo("Task images ready; preparing Work container...")
        elif name == "run_started" and isinstance(value, str):
            typer.echo(f"Run ID: {value}")
        elif name == "work_started":
            typer.echo("Work container started; preparing Agent...")
        elif name == "agent_started":
            details = value if isinstance(value, Mapping) else {}
            agent = details.get("name", "Agent")
            timeout = details.get("timeout_seconds")
            suffix = "" if timeout is None else f" (timeout={timeout}s)"
            typer.echo(f"Running {agent}{suffix}...")
            output_path = details.get("output_path")
            if output_path is not None:
                typer.echo(f"Agent output: {output_path}")
        elif name == "agent_finished":
            details = value if isinstance(value, Mapping) else {}
            typer.echo(
                "Agent finished: "
                f"exit_code={details.get('exit_code')} "
                f"timed_out={details.get('timed_out', False)}"
            )
        elif name == "judge_started":
            round_id = self._round_id(value)
            typer.echo(f"Judge {round_id}: snapshotting Work...")
        elif name == "judge_exec_started":
            round_id = self._round_id(value)
            typer.echo(f"Judge {round_id}: running /tests/test.sh...")
        elif name == "judge_finished":
            details = value if isinstance(value, Mapping) else {}
            score = details.get("score")
            runtime = details.get("runtime_seconds")
            runtime_seconds = 0.0 if runtime is None else float(runtime)
            typer.echo(
                f"Judge {details.get('round_id', 'unknown')} finished: "
                f"{details.get('status', 'unknown')} "
                f"score={'-' if score is None else score} "
                f"runtime={runtime_seconds:.2f}s"
            )

    @staticmethod
    def _round_id(value: object) -> object:
        return (
            value.get("round_id", "unknown")
            if isinstance(value, Mapping)
            else "unknown"
        )


@dataclass(frozen=True, slots=True)
class _ClusterConsole:
    def __call__(self, name: str, value: object) -> None:
        details = value if isinstance(value, Mapping) else {}
        if name == "dry_run":
            resources = details.get("resources", {})
            resources = resources if isinstance(resources, Mapping) else {}
            multi_node = details.get("multi_node", {})
            multi_node = (
                multi_node if isinstance(multi_node, Mapping) else {}
            )
            typer.echo("Blue Vela dry-run:")
            typer.echo(_field("Run ID:", details.get("run_id")))
            typer.echo(_field("Run dir:", details.get("run_dir")))
            typer.echo(_field("Image:", details.get("image")))
            typer.echo(_field("Cache hit:", details.get("cache_hit")))
            if multi_node:
                work = multi_node.get("work", {})
                work = work if isinstance(work, Mapping) else {}
                verifier = multi_node.get("verifier", {})
                verifier = verifier if isinstance(verifier, Mapping) else {}
                typer.echo(_field("Work GPUs:", work.get("gpu_count")))
                typer.echo(_field("Judge GPUs:", verifier.get("gpu_count")))
                typer.echo(_field("Work nodes:", work.get("node_count")))
                typer.echo(_field("Judge nodes:", verifier.get("node_count")))
                typer.echo(_field("Total nodes:", multi_node.get("total_nodes")))
                typer.echo(
                    _field("GPUs per node:", multi_node.get("gpus_per_node"))
                )
                typer.echo(
                    _field("CPU/node:", multi_node.get("cpu_slots_per_node"))
                )
                typer.echo(
                    _field("Memory/node:", multi_node.get("memory_mb_per_node"))
                )
                typer.echo(
                    _field(
                        "Shared workspace:",
                        f"{multi_node.get('shared_workspace_mb')} MiB",
                    )
                )
                typer.echo(
                    _field(
                        "Node scratch:",
                        f"{multi_node.get('node_tmp_mb')} MiB",
                    )
                )
                typer.echo(_field("Pool policy:", details.get("pool_policy")))
            else:
                typer.echo(_field("Work GPUs:", resources.get("work_gpus")))
                typer.echo(
                    _field("Verifier GPUs:", resources.get("verifier_gpus"))
                )
                typer.echo(_field("Total GPUs:", resources.get("total_gpus")))
                typer.echo(_field("CPU slots:", resources.get("cpu_slots")))
                typer.echo(_field("Memory MB:", resources.get("memory_mb")))
            typer.echo("Build submission:")
            typer.echo(f"  {shlex.join(tuple(details.get('build_argv', ())))}")
            typer.echo("Run submission:")
            typer.echo(f"  {shlex.join(tuple(details.get('run_argv', ())))}")
            binds = tuple(details.get("binds", ()))
            typer.echo(_field("Binds:", ", ".join(str(item) for item in binds)))
            assets = tuple(details.get("assets", ()))
            if assets:
                ready = sum(
                    1
                    for item in assets
                    if isinstance(item, Mapping) and item.get("ready") is True
                )
                typer.echo(_field("Assets ready:", f"{ready}/{len(assets)}"))
                for item in assets:
                    if not isinstance(item, Mapping) or item.get("ready") is True:
                        continue
                    typer.echo(
                        _field(
                            "Missing asset:",
                            f"{item.get('phase')}:{item.get('path')}",
                        )
                    )
        elif name == "job_submitted":
            typer.echo(
                f"Submitted {details.get('stage')} job: {details.get('job_id')}"
            )
        elif name == "job_state":
            typer.echo(f"{details.get('stage')} job state: {details.get('state')}")


def _field(label: str, value: object) -> str:
    return f"  {label:<18}{value}"


def _fail(error: BaseException, *, verbose: bool) -> None:
    if verbose:
        detail = redact_text(f"{type(error).__name__}: {error}")
    elif isinstance(error, HarnessError):
        detail = f"{error.code.value}: {redact_text(str(error))}"
    else:
        detail = "run failed; use --verbose for diagnostic detail"
    typer.echo(f"Error: {detail}", err=True)
    raise typer.Exit(1)


@app.command("run")
def run_command(
    task_dir: Annotated[Path, typer.Argument()],
    agent: Annotated[str, typer.Option("--agent")] = "codex",
    gpus: Annotated[
        str,
        typer.Option(
            "--gpus",
            help=(
                "Ordered GPU pool authorized for this run; Work sees only its "
                "task-declared count. Omit for CPU-only Work and Judge"
            ),
        ),
    ] = "",
    primary_reward: Annotated[str | None, typer.Option("--primary-reward")] = None,
    score_direction: Annotated[
        Literal["maximize", "minimize"], typer.Option("--score-direction")
    ] = "maximize",
    timeout: Annotated[float | None, typer.Option("--timeout", min=0.0)] = None,
    max_submissions: Annotated[
        int | None, typer.Option("--max-submissions", min=1)
    ] = None,
    cooldown: Annotated[float, typer.Option("--cooldown", min=0.0)] = 0.0,
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(".rsi-harness"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    model: Annotated[str | None, typer.Option("--model")] = None,
    reasoning_effort: Annotated[
        str | None,
        typer.Option(
            "--reasoning-effort",
            help="Agent reasoning effort; validated by the selected Agent adapter",
        ),
    ] = None,
    agent_auth: Annotated[
        AgentAuthSource | None,
        typer.Option("--agent-auth", help="Explicit Agent credential source"),
    ] = None,
    disable_stop_hook: Annotated[
        bool,
        typer.Option(
            "--disable-stop-hook",
            help="Allow the Agent to stop naturally without the RSI Loop stop hook",
        ),
    ] = False,
    cluster: Annotated[
        str | None,
        typer.Option(
            "--cluster",
            help="Cluster name (for example bluevela) or a cluster profile TOML",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Resolve and print cluster submissions without creating or submitting",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    try:
        selectors = _selectors(gpus) if gpus else ()
    except typer.BadParameter as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from None

    try:
        task = task_dir.expanduser().resolve()
        roots = RuntimeRoots(
            data=data_root.expanduser().resolve(),
            logs=logs_root.expanduser().resolve(),
        )
    except Exception as error:
        _fail(
            SetupError(f"cannot normalize run paths: {error}"),
            verbose=verbose,
        )
        return
    if not task.is_dir():
        typer.echo(f"Error: task directory does not exist: {task}", err=True)
        raise typer.Exit(2)
    if cluster is not None and selectors:
        typer.echo(
            "Error: --gpus is local-only and cannot be used with --cluster",
            err=True,
        )
        raise typer.Exit(2)
    if dry_run and cluster is None:
        typer.echo("Error: --dry-run requires --cluster", err=True)
        raise typer.Exit(2)

    options = CompileOptions(
        agent_name=agent,
        primary_reward=primary_reward,
        score_direction=score_direction,
        agent_timeout_seconds=timeout,
        max_submissions=max_submissions,
        cooldown_seconds=cooldown,
        disable_stop_hook=disable_stop_hook,
    )
    if cluster is not None:
        cluster_request = ClusterRunRequest(
            task_dir=task,
            agent_name=agent,
            options=options,
            logs_root=roots.logs,
            model=model,
            reasoning_effort=reasoning_effort,
            agent_auth=agent_auth,
            dry_run=dry_run,
        )
        typer.echo(f"Running Agent {agent!r} on cluster {cluster!r}: {task.name}")
        try:
            adapter = build_cluster_adapter(
                cluster,
                event_callback=_ClusterConsole(),
            )
            started = time.monotonic()
            cluster_result = adapter.run(cluster_request)
        except BaseException as error:
            _fail(error, verbose=verbose)
            return
        if dry_run:
            typer.echo("\nDry-run complete; no jobs were submitted.")
            return
        elapsed = time.monotonic() - started
        typer.echo(f"\nCluster run completed in {elapsed:.1f}s")
        typer.echo(_field("Run ID:", cluster_result.run_id))
        typer.echo(_field("Status:", cluster_result.status.value))
        typer.echo(_field("Job IDs:", ", ".join(cluster_result.job_ids)))
        typer.echo(_field("Logs:", cluster_result.log_dir))
        if cluster_result.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
            raise typer.Exit(1)
        return

    try:
        services = _services(
            roots.data,
            roots.logs,
            event_callback=_RunConsole(),
        )
        available = services.available_agents()
    except BaseException as error:
        _fail(error, verbose=verbose)
        return
    if agent not in available:
        typer.echo(
            f"Error: unknown Agent {agent!r}; choose one of: {', '.join(available)}",
            err=True,
        )
        raise typer.Exit(2)
    pending_root = roots.data / "pending"
    request = RunRequest(
        task_dir=task,
        agent_name=agent,
        gpu_selectors=selectors,
        options=options,
        paths=RunPaths(
            root=pending_root,
            workspace=pending_root / "workspace",
            logs=roots.logs,
        ),
        model=model,
        reasoning_effort=reasoning_effort,
        agent_auth=agent_auth,
    )
    typer.echo(f"Running Agent {agent!r} on task: {task.name}")
    typer.echo(_field("Task dir:", task))
    typer.echo(_field("Timeout:", "task default" if timeout is None else f"{timeout}s"))
    if model:
        typer.echo(_field("Model:", model))
    if reasoning_effort:
        typer.echo(_field("Reasoning:", reasoning_effort))
    typer.echo(_field("GPU selectors:", ", ".join(selectors) if selectors else "auto"))
    if primary_reward:
        typer.echo(_field("Primary reward:", primary_reward))
    if max_submissions is not None:
        typer.echo(_field("Max submissions:", max_submissions))
    if cooldown:
        typer.echo(_field("Cooldown:", f"{cooldown}s"))
    if disable_stop_hook:
        typer.echo(_field("Stop hook:", "disabled"))
    if agent_auth is not None:
        typer.echo(_field("Agent auth:", agent_auth.value))
    typer.echo(_field("Data root:", roots.data))
    typer.echo(_field("Logs root:", roots.logs))
    typer.echo()

    start = time.monotonic()
    try:
        result = services.run(request)
    except BaseException as error:
        _fail(error, verbose=verbose)
        return
    elapsed = time.monotonic() - start

    typer.echo(f"\nRun completed in {elapsed:.1f}s")
    typer.echo(_field("Run ID:", result.run_id))
    typer.echo(_field("Status:", result.status.value))
    typer.echo(_field("Rounds:", result.total_rounds))
    typer.echo(
        _field("Best score:", "-" if result.best_score is None else result.best_score)
    )
    typer.echo(_field("Best round:", result.best_round or "-"))
    typer.echo(_field("Workspace:", roots.data / result.run_id / "workspace"))
    typer.echo(_field("Logs:", roots.logs / "runs" / result.run_id))
    if result.reports:
        typer.echo()
        typer.echo("Judge rounds:")
        for report in result.reports:
            score = "-" if report.score is None else report.score
            typer.echo(f"  {report.round_id}: {report.status.value} score={score}")
    if result.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
        raise typer.Exit(1)


@app.command("visualize")
def visualize_command(
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(".rsi-harness"),
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535)] = 8000,
) -> None:
    import uvicorn

    from rsi_loop.visualizer.server import create_app

    roots = RuntimeRoots(
        data=data_root.expanduser().resolve(),
        logs=logs_root.expanduser().resolve(),
    )
    visualizer = create_app(
        roots.logs / "runs", tasks_dir=roots.data / "generated-tasks"
    )
    uvicorn.run(visualizer, host=host, port=port)


@app.command("recover")
def recover_command(
    run_id: Annotated[str | None, typer.Argument()] = None,
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(".rsi-harness"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    try:
        recovered = _services(data_root, logs_root).recover(run_id)
    except BaseException as error:
        _fail(error, verbose=verbose)
        return
    for selected in recovered:
        typer.echo(f"recovered: {selected}")


@app.command("cleanup")
def cleanup_command(
    run_id: Annotated[str, typer.Argument()],
    delete_workspace: Annotated[bool, typer.Option("--delete-workspace")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(".rsi-harness"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    if delete_workspace and not yes:
        confirmed = typer.confirm(
            f"Delete the retained workspace for {run_id}?",
            default=False,
        )
        if not confirmed:
            typer.echo("Cleanup cancelled", err=True)
            raise typer.Exit(1)
    try:
        _services(data_root, logs_root).cleanup(
            run_id, delete_workspace=delete_workspace
        )
    except BaseException as error:
        _fail(error, verbose=verbose)
        return
    typer.echo(f"cleaned: {run_id}")


if __name__ == "__main__":
    app()
