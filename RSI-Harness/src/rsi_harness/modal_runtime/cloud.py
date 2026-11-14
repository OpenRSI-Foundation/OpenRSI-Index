"""Operator launch and serial continuation; imports Modal only on use."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions
from rsi_harness.task.compiler import HarborTaskCompiler

from .config import ModalRunConfig, validate_task


def run_segment(payload):
    """Top-level cloud function; user code runs in a killable child process."""
    import modal

    config = ModalRunConfig.model_validate(payload["config"])
    volume = modal.Volume.from_name(
        config.volume_name, environment_name=config.environment_name
    )
    volume.reload()
    root = Path("/rsi-state/data") / payload["run_id"]
    root.mkdir(parents=True, exist_ok=True)
    request = root / "segment-input.json"
    request.write_text(json.dumps(payload))
    # No credentials in payloads, receipts, or retained request files.
    process = subprocess.Popen(
        [sys.executable, "-m", "rsi_harness.modal_runtime.worker", str(request)]
    )
    try:
        code = process.wait(
            timeout=config.segment_seconds + config.cleanup_seconds - 10
        )
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise SetupError("native cleanup exceeded its allowance; recovery required")
    finally:
        volume.commit()
    if code:
        raise SetupError(
            f"native worker failed with exit code {code}; inspect retained logs"
        )
    return json.loads((root / "segment-result.json").read_text())


def launch(task_dir, config, options, **kwargs):
    # Freeze uploads so edits in an operator checkout cannot change an experiment.
    with tempfile.TemporaryDirectory(prefix="rsi-modal-stage-") as temporary:
        staging = Path(temporary)
        source = Path(__file__).resolve().parents[2]
        shutil.copytree(
            source,
            staging / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copytree(Path(task_dir).resolve(), staging / "task", symlinks=True)
        return _launch(
            staging / "task", config, options, source=staging / "src", **kwargs
        )


def _launch(
    task_dir,
    config,
    options,
    *,
    model=None,
    reasoning_effort=None,
    receipt_path=None,
    source=None,
):
    # Set profile before importing Modal; do not mutate global operator config.
    os.environ["MODAL_PROFILE"] = config.profile
    import modal

    task_dir = Path(task_dir).resolve()
    task = HarborTaskCompiler().compile(task_dir, options)
    validate_task(task, config)
    if task.service.build_context != task_dir / "environment":
        raise SetupError("Modal requires a task-owned environment/Dockerfile build")
    image = (
        modal.Image.from_dockerfile(
            task_dir / "environment" / "Dockerfile",
            context_dir=task_dir / "environment",
        )
        .apt_install(
            "bubblewrap", "uidmap", "util-linux", "bash", "git", "curl", "xz-utils"
        )
        .run_commands(
            "printf 'root:0:65536\\n' > /etc/subuid",
            "printf 'root:0:65536\\n' > /etc/subgid",
        )
        .pip_install(
            "pydantic>=2.11.7,<3",
            "fastapi>=0.128,<1",
            "uvicorn>=0.38,<1",
            "docker>=7.1,<8",
            "jinja2",
            "pyyaml",
            "toml",
            "requests",
            "rich",
            "typer>=0.25,<1",
        )
        .run_commands("python -m pip install --no-deps harbor==0.21.0")
        .add_local_dir(
            source,
            "/opt/rsi-runtime",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc"],
        )
        .env({"PYTHONPATH": "/opt/rsi-runtime"})
        .run_commands("python -m rsi_harness.modal_runtime.bootstrap")
    )
    if options.agent_name == "codex":
        from rsi_loop.harness.agent.base import NODE_INSTALL_CMD
        from rsi_loop.harness.agent.versions import CODEX

        image = image.run_commands(
            NODE_INSTALL_CMD.replace("sudo -E ", "").replace("sudo ", ""),
            f"npm install -g @openai/codex@{CODEX}",
        )
    image = image.add_local_dir(task_dir, "/opt/rsi-task", copy=True)
    volume = modal.Volume.from_name(
        config.volume_name,
        environment_name=config.environment_name,
        create_if_missing=True,
    )
    volumes = {"/rsi-state": volume}
    for mount, name in config.asset_volumes.items():
        volumes[mount] = modal.Volume.from_name(
            name, environment_name=config.environment_name
        ).with_mount_options(read_only=True)
    secrets = (
        []
        if config.agent_secret_name is None
        else [
            modal.Secret.from_name(
                config.agent_secret_name,
                environment_name=config.environment_name,
                required_keys=["RSI_AGENT_API_KEY"],
            )
        ]
    )
    app = modal.App(config.app_name)
    function = app.function(
        image=image,
        gpu=config.gpu,
        cpu=(config.cpus, config.cpus),
        memory=(config.memory_mb, config.memory_mb),
        ephemeral_disk=config.disk_mb,
        timeout=config.segment_seconds + config.cleanup_seconds,
        max_containers=1,
        retries=0,
        single_use_containers=True,
        volumes=volumes,
        secrets=secrets,
        serialized=True,
    )(run_segment)
    run_id = uuid.uuid4().hex
    receipts = []
    with modal.enable_output(), app.run(environment_name=config.environment_name):
        payload = {
            "run_id": run_id,
            "config": config.model_dump(mode="json"),
            "options": options.model_dump(mode="json"),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "task_digest": task.source_digest,
            "image_id": image.object_id,
        }
        for segment in range(1, config.max_segments + 1):
            payload["expected_segment"] = segment
            receipt = function.remote(payload)
            receipts.append(receipt)
            if receipt_path:
                Path(receipt_path).write_text(
                    json.dumps(
                        {"run_id": run_id, "app_id": app.app_id, "segments": receipts},
                        indent=2,
                    )
                )
            if not receipt["continue"]:
                return receipts
    raise SetupError("segment cap reached before the run deadline")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--agent", choices=("codex", "oracle"), default="codex")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--max-submissions", type=int, required=True)
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    config = ModalRunConfig.model_validate_json(args.config.read_text())
    options = CompileOptions(
        agent_name=args.agent,
        agent_timeout_seconds=args.seconds,
        max_submissions=args.max_submissions,
        disable_stop_hook=args.agent == "oracle",
    )
    launch(
        args.task,
        config,
        options,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        receipt_path=args.receipt,
    )


if __name__ == "__main__":
    main()
