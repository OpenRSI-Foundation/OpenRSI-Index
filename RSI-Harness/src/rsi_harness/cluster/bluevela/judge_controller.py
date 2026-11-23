"""GPU-free verifier controller launched on the frozen Judge pool."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import socket
import subprocess
from pathlib import Path

from pydantic import Field, field_validator

from rsi_harness.cluster.bluevela.apptainer_environment import (
    isolated_apptainer_environment,
)
from rsi_harness.cluster.bluevela.multinode import RemoteWorkerTemplate
from rsi_harness.cluster.bluevela.preflight import is_singularity_ce
from rsi_harness.errors import InfrastructureError
from rsi_harness.models import PersistedModel
from rsi_harness.runtime.judge_binds import (
    BIND_RECORD_TARGET,
    bind_record_for,
    remove_bind_record,
)


class JudgeControllerControl(PersistedModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_id: str = Field(min_length=1)
    host: str = Field(min_length=1)
    worker: RemoteWorkerTemplate
    broker_root: Path
    command: tuple[str, ...] = Field(min_length=1)
    environment: dict[str, str]
    local_world_size: int = Field(gt=0)

    @field_validator("broker_root")
    @classmethod
    def _absolute_broker_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Judge broker root must be absolute")
        return value


def load_control(path: Path) -> JudgeControllerControl:
    try:
        return JudgeControllerControl.model_validate_json(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise InfrastructureError(
            f"invalid frozen Judge controller control {path}: {error}"
        ) from error


def controller_tmp_dir(control: JudgeControllerControl) -> Path:
    run_hash = hashlib.sha256(control.run_id.encode()).hexdigest()[:16]
    return (
        control.worker.temp_root
        / "rsi-harness"
        / run_hash
        / "judge-controller"
        / control.request_id
    )


def _environment_values(control: JudgeControllerControl) -> dict[str, str]:
    forbidden = {
        "CUDA_VISIBLE_DEVICES",
        "LSB_MCPU_HOSTS",
        "RSI_NODE_RANK",
        "RSI_MASTER_ADDR",
    }
    if forbidden.intersection(control.environment):
        raise InfrastructureError(
            "Judge controller environment contains cluster authority"
        )
    return {
        **control.worker.environment,
        **control.environment,
        # Apptainer applies PREPEND_PATH after the image's own environment.
        # This keeps the Harness-owned transparent launcher ahead of an
        # image-provided torchrun such as /opt/conda/bin/torchrun.
        "PREPEND_PATH": "/usr/local/bin",
        "RSI_MULTINODE_ROOT": "/run/rsi-harness/torchrun",
        "RSI_LOCAL_WORLD_SIZE": str(control.local_world_size),
        "TMPDIR": "/tmp",
        "TEMP": "/tmp",
        "TMP": "/tmp",
    }


def controller_binds(
    control: JudgeControllerControl, local_tmp: Path
) -> list[tuple[Path, Path, bool]]:
    """Every (source, target, read-only) bind of the controller's container."""
    return [
        *(
            (binding.source, binding.target, binding.read_only)
            for binding in control.worker.binds
        ),
        (control.broker_root, Path("/run/rsi-harness/torchrun"), False),
        (local_tmp, Path("/tmp"), False),
    ]


def build_apptainer_command(
    control: JudgeControllerControl,
    *,
    current_host: str,
    local_tmp: Path,
    bind_record: Path | None = None,
) -> tuple[str, ...]:
    host = current_host.split(".", 1)[0]
    if host != control.host:
        raise InfrastructureError(
            f"Judge controller host {host!r} differs from {control.host!r}"
        )
    expected_tmp = controller_tmp_dir(control)
    if Path(local_tmp).resolve() != expected_tmp.resolve():
        raise InfrastructureError("Judge controller tmp differs from authority")
    command = [str(control.worker.apptainer_binary), "exec"]
    if control.worker.network_mode == "no-network":
        command.extend(("--net", "--network", "none", "--hostname", "localhost"))
    command.extend(
        (
            "--containall",
            "--no-mount",
            "hostfs",
            "--no-umask",
            "--cleanenv",
            "--no-eval",
            "--writable-tmpfs",
            "--cwd",
            str(control.worker.container_workdir),
        )
    )
    for source, target, read_only in controller_binds(control, expected_tmp):
        suffix = ":ro" if read_only else ""
        command.extend(("--bind", f"{source}:{target}{suffix}"))
    if bind_record is not None:
        command.extend(("--bind", f"{bind_record}:{BIND_RECORD_TARGET}:ro"))
    _environment_values(control)
    command.extend((str(control.worker.sif_path), *control.command))
    if "--nv" in command or any(
        item.startswith("CUDA_VISIBLE_DEVICES=") for item in command
    ):
        raise InfrastructureError("Judge controller must not receive GPU selectors")
    return tuple(command)


def build_apptainer_environment(
    control: JudgeControllerControl,
) -> dict[str, str]:
    """Build the clean host environment used to inject Judge values."""

    return isolated_apptainer_environment(
        _environment_values(control),
        singularity_ce=is_singularity_ce(control.worker.apptainer_binary),
    )


def _run(control: JudgeControllerControl) -> int:
    local_tmp = controller_tmp_dir(control)
    local_tmp.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    local_tmp.mkdir(mode=0o700, exist_ok=False)
    environment = build_apptainer_environment(control)
    # On this node, in the namespace that starts the container: NFS device
    # numbers belong to one node.
    record = bind_record_for(
        (
            (source, target)
            for source, target, _ro in controller_binds(control, local_tmp)
        ),
        local_tmp.parent,
    )
    try:
        command = build_apptainer_command(
            control,
            current_host=socket.gethostname(),
            local_tmp=local_tmp,
            bind_record=record,
        )
    except BaseException:
        remove_bind_record(record)
        raise
    process: subprocess.Popen[bytes] | None = None

    def forward(signum: int, _frame: object) -> None:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signum)

    previous_term = signal.signal(signal.SIGTERM, forward)
    previous_int = signal.signal(signal.SIGINT, forward)
    try:
        process = subprocess.Popen(
            command,
            start_new_session=True,
            env=environment,
        )
        (local_tmp / "pid").write_text(str(process.pid))
        (local_tmp / "pid").chmod(0o600)
        return process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        remove_bind_record(record)
        shutil.rmtree(local_tmp, ignore_errors=True)


def _stop(control: JudgeControllerControl) -> int:
    local_tmp = controller_tmp_dir(control)
    pid_path = local_tmp / "pid"
    if not pid_path.is_file():
        return 0
    try:
        pid = int(pid_path.read_text())
    except (OSError, ValueError) as error:
        raise InfrastructureError(
            "invalid Judge controller process identity"
        ) from error
    os.killpg(pid, signal.SIGTERM)
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "stop"))
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = load_control(args.control)
    return _run(control) if args.action == "run" else _stop(control)


if __name__ == "__main__":
    raise SystemExit(_main())
