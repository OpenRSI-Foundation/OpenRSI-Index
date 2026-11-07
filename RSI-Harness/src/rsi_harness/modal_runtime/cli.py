"""Lazy Modal CLI registration; Docker and cluster commands stay independent."""

from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, help="Native Work/Judge execution on Modal.")


@app.command("run")
def run(
    task: Path,
    config: Path = typer.Option(...),
    seconds: float = typer.Option(..., min=1),
    max_submissions: int = typer.Option(..., min=1),
    receipt: Path = typer.Option(...),
    agent: str = "codex",
    model: str | None = None,
    reasoning_effort: str | None = None,
):
    """Run a bounded task; retain native artifacts in the configured Volume."""
    from rsi_harness.models import CompileOptions

    from .cloud import launch
    from .config import ModalRunConfig

    if agent not in {"codex", "oracle"}:
        raise typer.BadParameter("agent must be codex or oracle")
    launch(
        task,
        ModalRunConfig.model_validate_json(config.read_text()),
        CompileOptions(
            agent_name=agent,
            agent_timeout_seconds=seconds,
            max_submissions=max_submissions,
            disable_stop_hook=agent == "oracle",
        ),
        model=model,
        reasoning_effort=reasoning_effort,
        receipt_path=receipt,
    )
