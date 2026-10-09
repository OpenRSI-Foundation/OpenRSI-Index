"""Command-line entry points for production RSI Harness runs."""

from __future__ import annotations

import re
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

import typer

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import build_cluster_adapter
from rsi_harness.config import EngineConfig
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
from rsi_harness.runtime.sandbox_policy import load_sandbox_policy

if TYPE_CHECKING:
    # The ledger imports the docker SDK: loaded only by prune-images.
    from rsi_harness.runtime.sandbox_ledger import PruneRow

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
sandbox_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="Operator commands for managed sandboxes.",
)
app.add_typer(sandbox_app, name="sandbox")


class RuntimeServicesPort(Protocol):
    def available_agents(self) -> tuple[str, ...]: ...

    def run(self, request: RunRequest) -> RunResult: ...

    def recover(self, run_id: str | None) -> tuple[str, ...]: ...

    def cleanup(self, run_id: str, *, delete_workspace: bool) -> None: ...

    def prune_images(
        self,
        *,
        older_than_seconds: float | None,
        dry_run: bool,
        confirm: Callable[[tuple[PruneRow, ...]], bool],
    ) -> tuple[PruneRow, ...] | None: ...


@dataclass(frozen=True, slots=True)
class RuntimeRoots:
    data: Path
    logs: Path


def build_runtime_services(
    *,
    roots: RuntimeRoots,
    event_callback: Callable[[str, object], None] | None = None,
    engine_config: EngineConfig | None = None,
) -> RuntimeServicesPort:
    """Build the concrete Docker/NVIDIA/RSI Loop production runtime."""
    from rsi_harness.runtime.production import ProductionRuntimeServices

    return ProductionRuntimeServices(
        data_root=roots.data,
        logs_root=roots.logs,
        event_callback=event_callback,
        engine_config=engine_config,
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
    engine_config: EngineConfig | None = None,
) -> RuntimeServicesPort:
    roots = RuntimeRoots(
        data=data_root.expanduser().resolve(),
        logs=logs_root.expanduser().resolve(),
    )
    if engine_config is not None:
        return build_runtime_services(
            roots=roots,
            event_callback=event_callback,
            engine_config=engine_config,
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
        elif name == "sandbox_reserved" and isinstance(value, Mapping):
            typer.echo(
                f"Sandbox envelope: {value['cpus']} CPUs, {value['memory_mb']} MiB "
                "(parents + children + broker)"
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
            multi_node = multi_node if isinstance(multi_node, Mapping) else {}
            label = "Slurm" if details.get("adapter") == "slurm" else "Blue Vela"
            typer.echo(f"{label} dry-run:")
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
                typer.echo(_field("GPUs per node:", multi_node.get("gpus_per_node")))
                typer.echo(_field("CPU/node:", multi_node.get("cpu_slots_per_node")))
                typer.echo(_field("Memory/node:", multi_node.get("memory_mb_per_node")))
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
                typer.echo(_field("Verifier GPUs:", resources.get("verifier_gpus")))
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
            typer.echo(f"Submitted {details.get('stage')} job: {details.get('job_id')}")
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
    sandbox_policy: Annotated[
        Path | None,
        typer.Option(
            "--sandbox-policy", help="Operator-approved local CPU sandbox policy TOML"
        ),
    ] = None,
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
            help="Cluster name (bluevela or slurm) or a cluster profile TOML",
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
    cluster_policy = None
    if cluster is not None and sandbox_policy is not None:
        try:
            cluster_policy = load_sandbox_policy(sandbox_policy)
        except BaseException as error:
            _fail(error, verbose=verbose)
            return
        environments = cluster_policy.environments
        if environments is None or environments.host.backend != "e2b":
            typer.echo(
                "Error: --sandbox-policy with --cluster requires "
                '[environments.host] backend = "e2b"; Docker envs are local-only',
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
            sandbox_policy=cluster_policy,
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
        service_options = {}
        if sandbox_policy is not None:
            service_options["engine_config"] = EngineConfig(
                data_root=roots.data,
                logs_root=roots.logs,
                sandbox_policy=load_sandbox_policy(sandbox_policy),
            )
        services = _services(
            roots.data,
            roots.logs,
            event_callback=_RunConsole(),
            **service_options,
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


_CLUSTER_RECOVERY_HELP = (
    "Cluster name or profile TOML: kill the run's E2B sandboxes "
    "(after its scheduler job has ended)"
)


def _recover_cluster(cluster: str, run_id: str | None) -> tuple[str, ...]:
    from rsi_harness.cluster.bluevela.sandbox import recover_cluster_sandboxes
    from rsi_harness.cluster.config import load_cluster_profile

    profile = load_cluster_profile(cluster)
    return recover_cluster_sandboxes(profile.storage.run_root, run_id)


@app.command("recover")
def recover_command(
    run_id: Annotated[str | None, typer.Argument()] = None,
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(".rsi-harness"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    cluster: Annotated[
        str | None, typer.Option("--cluster", help=_CLUSTER_RECOVERY_HELP)
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    try:
        recovered = (
            _recover_cluster(cluster, run_id)
            if cluster is not None
            else _services(data_root, logs_root).recover(run_id)
        )
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
    cluster: Annotated[
        str | None, typer.Option("--cluster", help=_CLUSTER_RECOVERY_HELP)
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    if cluster is not None:
        if delete_workspace:
            typer.echo(
                "Error: --delete-workspace is local-only and cannot be used "
                "with --cluster",
                err=True,
            )
            raise typer.Exit(2)
        try:
            _recover_cluster(cluster, run_id)
        except BaseException as error:
            _fail(error, verbose=verbose)
            return
        typer.echo(f"cleaned: {run_id}")
        return
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


_DURATION = re.compile(r"([1-9][0-9]{0,8})([smhdw])")
_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _duration(value: str) -> float:
    match = _DURATION.fullmatch(value.strip())
    if match is None:
        raise typer.BadParameter("expected a duration such as 45m, 12h, 7d or 2w")
    return float(int(match.group(1)) * _DURATION_UNITS[match.group(2)])


def _size(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    size = float(value)
    for unit in ("KiB", "MiB", "GiB"):
        size /= 1024
        if size < 1024:
            break
    return f"{size:.1f} {unit}"


def _prune_table(rows: tuple[PruneRow, ...]) -> None:
    typer.echo(
        f"{'IMAGE':<12}  {'SIZE':>10}  {'LAST USED':<20}  {'ACTION':<12}  "
        f"{'REASON':<40}  REFERENCES"
    )
    for row in rows:
        image = row.image
        typer.echo(
            f"{image.image_id.removeprefix('sha256:')[:12]:<12}  "
            f"{_size(row.bytes) if row.bytes else '-':>10}  "
            f"{image.last_used_at.strftime('%Y-%m-%d %H:%M UTC'):<20}  "
            f"{row.action:<12}  {row.reason or '-':<40}  "
            f"{', '.join(image.references) or '-'}"
        )


@sandbox_app.command("prune-images")
def prune_images_command(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="List what would be removed; change nothing"),
    ] = False,
    older_than: Annotated[
        str | None,
        typer.Option(
            "--older-than",
            metavar="DURATION",
            help="Only images last used longer ago than this (45m, 12h, 7d, 2w)",
        ),
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", help="Remove without asking for confirmation")
    ] = False,
    data_root: Annotated[Path, typer.Option("--data-root")] = Path(".rsi-harness"),
    logs_root: Annotated[Path, typer.Option("--logs-root")] = Path("logs"),
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
) -> None:
    """Remove the images brokered pulls first brought to this host.

    Only images in the data root's pull ledger are candidates; images that
    were on the host before the sandbox first pulled them are never touched.
    Only this data root's run leases are checked: do not prune while runs
    under another data root use the same images.
    """
    from rsi_harness.runtime.sandbox_ledger import REMOVED, WOULD_REMOVE

    try:
        older_than_seconds = None if older_than is None else _duration(older_than)
    except typer.BadParameter as error:
        typer.echo(f"Error: --older-than: {error}", err=True)
        raise typer.Exit(2) from None

    asked = False

    def confirm(rows: tuple[PruneRow, ...]) -> bool:
        nonlocal asked
        if yes:
            return True
        asked = True
        _prune_table(rows)
        count = sum(1 for row in rows if row.action == WOULD_REMOVE)
        return typer.confirm(f"Remove {count} pulled image(s)?", default=False)

    try:
        rows = _services(data_root, logs_root).prune_images(
            older_than_seconds=older_than_seconds, dry_run=dry_run, confirm=confirm
        )
    except BaseException as error:
        _fail(error, verbose=verbose)
        return
    if rows is None:
        typer.echo("Prune cancelled", err=True)
        raise typer.Exit(1)
    if not rows:
        typer.echo("No pulled images are recorded in the pull ledger.")
        return
    if asked:
        typer.echo("Result:")
    _prune_table(rows)
    action = WOULD_REMOVE if dry_run else REMOVED
    freed = sum(row.bytes for row in rows if row.action == action)
    label = "Would free" if dry_run else "Freed"
    typer.echo(f"{label}: {freed} bytes ({_size(freed)})")


if __name__ == "__main__":
    app()
