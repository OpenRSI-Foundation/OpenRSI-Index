"""Fixed-shape per-host Apptainer worker for Blue Vela torchrun requests."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

from rsi_harness.cluster.bluevela.apptainer_environment import (
    isolated_apptainer_environment,
)
from rsi_harness.cluster.bluevela.multinode import RemoteRequestControl
from rsi_harness.errors import InfrastructureError

_SIZE_FLAGS = {"--nnodes", "--nproc-per-node", "--nproc_per_node"}
_TORCHRUN_VALUE_FLAGS = {
    "--log-dir",
    "--log_dir",
    "--local-addr",
    "--local_addr",
    "--max-restarts",
    "--max_restarts",
    "--monitor-interval",
    "--monitor_interval",
    "--redirects",
    "--role",
    "--run-path",
    "--run_path",
    "--start-method",
    "--start_method",
    "--tee",
}


def load_control(path: Path) -> RemoteRequestControl:
    try:
        raw = Path(path).read_text()
        return RemoteRequestControl.model_validate_json(raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise InfrastructureError(
            f"invalid frozen remote worker control {path}: {error}"
        ) from error


def request_tmp_dir(
    control: RemoteRequestControl,
    *,
    pool_rank: int,
) -> Path:
    run_hash = __import__("hashlib").sha256(
        control.request.run_id.encode()
    ).hexdigest()[:16]
    return (
        control.worker.temp_root
        / "rsi-harness"
        / run_hash
        / control.phase
        / control.request.request_id
        / f"pool-{pool_rank}"
    )


def _task_torchrun_argv(argv: tuple[str, ...]) -> tuple[str, ...]:
    """Remove only task-supplied size flags before the training entrypoint."""
    result: list[str] = []
    offset = 0
    before_entrypoint = True
    while offset < len(argv):
        item = argv[offset]
        if not before_entrypoint:
            result.append(item)
            offset += 1
            continue
        if item == "--":
            result.extend(argv[offset:])
            break
        if not item.startswith("-"):
            before_entrypoint = False
            result.append(item)
            offset += 1
            continue
        name, separator, _inline = item.partition("=")
        if name in _SIZE_FLAGS:
            offset += 1 if separator else 2
            continue
        result.append(item)
        if not separator and name in _TORCHRUN_VALUE_FLAGS:
            if offset + 1 >= len(argv):
                raise InfrastructureError(
                    f"frozen torchrun option {name} has no value"
                )
            result.append(argv[offset + 1])
            offset += 2
            continue
        offset += 1
    if not result:
        raise InfrastructureError("frozen torchrun request has no entrypoint")
    return tuple(result)


def _rank_authority(
    control: RemoteRequestControl,
    *,
    request_id: str,
    pool_rank: int,
    node_rank: int,
    current_host: str,
):
    if request_id != control.request.request_id:
        raise InfrastructureError("remote worker request identity differs")
    if node_rank < 0 or node_rank >= len(control.lease.nodes):
        raise InfrastructureError("remote worker node rank is outside its lease")
    expected_pool_rank = control.lease.pool_ranks[node_rank]
    node = control.lease.nodes[node_rank]
    if pool_rank != expected_pool_rank:
        raise InfrastructureError("remote worker pool rank differs from its lease")
    host = current_host.split(".", 1)[0]
    if host != node.host:
        raise InfrastructureError(
            f"remote worker host {host!r} differs from frozen host {node.host!r}"
        )
    if len(control.rank_environment) != len(control.lease.nodes):
        raise InfrastructureError("remote worker rank environments are incomplete")
    environment = control.rank_environment[node_rank]
    if (
        environment.get("RSI_NODE_RANK") != str(node_rank)
        or environment.get("RSI_POOL_RANK") != str(pool_rank)
    ):
        raise InfrastructureError("remote worker rank environment differs")
    return node, environment


def build_apptainer_command(
    control: RemoteRequestControl,
    *,
    request_id: str,
    pool_rank: int,
    node_rank: int,
    current_host: str,
    local_tmp: Path,
) -> tuple[str, ...]:
    """Build the only container command a remote worker may execute."""
    node, rank_environment = _rank_authority(
        control,
        request_id=request_id,
        pool_rank=pool_rank,
        node_rank=node_rank,
        current_host=current_host,
    )
    expected_tmp = request_tmp_dir(control, pool_rank=pool_rank)
    if Path(local_tmp).resolve() != expected_tmp.resolve():
        raise InfrastructureError("remote worker local tmp differs from authority")
    workdir = control.request.working_directory
    # MultiNodeRequest already requires an absolute, traversal-free container
    # path.  Do not restrict it to the workspace: trusted task launchers may
    # invoke torchrun from an immutable image directory (for example a pinned
    # source checkout under /opt).  Apptainer's containall boundary and the
    # frozen bind list remain the filesystem authority for every remote rank.

    command = [str(control.worker.apptainer_binary), "exec"]
    # A remote torchrun rank must inherit the allocated host's routable
    # inter-node interface.  Apptainer's `--network none` removes that same
    # interface, so even the frozen RSI_MASTER_ADDR becomes unreachable.  The
    # requested policy remains in RemoteWorkerTemplate for audit; Blue Vela's
    # host-network limitation is contained to this multi-node-only worker.
    command.extend(
        (
            "--nv",
            "--containall",
            "--cleanenv",
            "--no-eval",
            "--writable-tmpfs",
            "--cwd",
            str(workdir),
        )
    )
    for binding in control.worker.binds:
        suffix = ":ro" if binding.read_only else ""
        command.extend(
            ("--bind", f"{binding.source}:{binding.target}{suffix}")
        )
    command.extend(("--bind", f"{expected_tmp}:/tmp"))
    command.extend(
        (
            str(control.worker.sif_path),
            str(control.worker.container_python),
            "-m",
            "torch.distributed.run",
            f"--nnodes={len(control.lease.nodes)}",
            f"--nproc-per-node={control.request.local_world_size}",
            f"--node-rank={node_rank}",
            f"--master-addr={rank_environment['RSI_MASTER_ADDR']}",
            f"--master-port={rank_environment['RSI_MASTER_PORT']}",
            *_task_torchrun_argv(control.request.argv),
        )
    )
    return tuple(command)


def build_apptainer_environment(
    control: RemoteRequestControl,
    *,
    request_id: str,
    pool_rank: int,
    node_rank: int,
    current_host: str,
) -> dict[str, str]:
    """Build the clean host environment used to inject container values."""

    node, rank_environment = _rank_authority(
        control,
        request_id=request_id,
        pool_rank=pool_rank,
        node_rank=node_rank,
        current_host=current_host,
    )
    values = {
        **control.worker.environment,
        **control.request.environment,
        **rank_environment,
        "CUDA_VISIBLE_DEVICES": ",".join(node.cuda_devices),
        "TMPDIR": "/tmp",
        "TEMP": "/tmp",
        "TMP": "/tmp",
    }
    return isolated_apptainer_environment(values)


def _pid_start_time(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[21]
    except (OSError, IndexError) as error:
        raise InfrastructureError("cannot attest remote worker process") from error


def _atomic_pid(path: Path, *, pid: int, start_time: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": pid, "start_time": start_time}, stream)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_worker(
    control: RemoteRequestControl,
    *,
    request_id: str,
    pool_rank: int,
    node_rank: int,
) -> int:
    local_tmp = request_tmp_dir(control, pool_rank=pool_rank)
    local_tmp.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    local_tmp.mkdir(mode=0o700, exist_ok=False)
    current_host = socket.gethostname()
    command = build_apptainer_command(
        control,
        request_id=request_id,
        pool_rank=pool_rank,
        node_rank=node_rank,
        current_host=current_host,
        local_tmp=local_tmp,
    )
    environment = build_apptainer_environment(
        control,
        request_id=request_id,
        pool_rank=pool_rank,
        node_rank=node_rank,
        current_host=current_host,
    )
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
        _atomic_pid(
            local_tmp / "process.json",
            pid=process.pid,
            start_time=_pid_start_time(process.pid),
        )
        return process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        shutil.rmtree(local_tmp, ignore_errors=True)


def stop_worker(
    control: RemoteRequestControl,
    *,
    request_id: str,
    pool_rank: int,
    node_rank: int,
) -> int:
    local_tmp = request_tmp_dir(control, pool_rank=pool_rank)
    _rank_authority(
        control,
        request_id=request_id,
        pool_rank=pool_rank,
        node_rank=node_rank,
        current_host=socket.gethostname(),
    )
    pid_path = local_tmp / "process.json"
    if not pid_path.is_file():
        return 0
    try:
        identity = json.loads(pid_path.read_text())
        pid = int(identity["pid"])
        start_time = str(identity["start_time"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise InfrastructureError("invalid remote worker process identity") from error
    if _pid_start_time(pid) != start_time:
        raise InfrastructureError("remote worker process identity is stale")
    os.killpg(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if Path(f"/proc/{pid}").exists():
        if _pid_start_time(pid) != start_time:
            raise InfrastructureError("remote worker process identity changed")
        os.killpg(pid, signal.SIGKILL)
    shutil.rmtree(local_tmp, ignore_errors=True)
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("run", "stop"))
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--pool-rank", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    args = parser.parse_args()
    control = load_control(args.control)
    function = run_worker if args.action == "run" else stop_worker
    return function(
        control,
        request_id=args.request,
        pool_rank=args.pool_rank,
        node_rank=args.node_rank,
    )


if __name__ == "__main__":
    raise SystemExit(_main())
