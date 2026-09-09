#!/usr/bin/env python3

"""Transparent ``torchrun`` front end for Blue Vela multi-node phases."""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

try:
    from rsi_harness.errors import SetupError
except ModuleNotFoundError:  # The installed in-container shim is standalone.
    class SetupError(RuntimeError):
        pass

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BLOCKED_PREFIXES = (
    "APPTAINER_",
    "LSB_",
    "LSF_",
    "SINGULARITY_",
)
_BLOCKED_NAMES = {
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "LOCAL_RANK",
    "LOGNAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OLDPWD",
    "PWD",
    "RANK",
    "RSI_LOCAL_WORLD_SIZE",
    "RSI_MASTER_ADDR",
    "RSI_MASTER_PORT",
    "RSI_NODE_RANK",
    "RSI_NUM_NODES",
    "RSI_POOL_DIGEST",
    "RSI_POOL_RANK",
    "RSI_POOL_SIZE",
    "RSI_MULTINODE_ROOT",
    "SHELL",
    "SSH_AUTH_SOCK",
    "USER",
    "WORLD_SIZE",
}
_SECRET_MARKERS = (
    "ANTHROPIC",
    "CREDENTIAL",
    "OPENAI",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
_TOPOLOGY_FLAGS = {
    "--master-addr",
    "--master_addr",
    "--master-port",
    "--master_port",
    "--rdzv-backend",
    "--rdzv_backend",
    "--rdzv-conf",
    "--rdzv_conf",
    "--rdzv-endpoint",
    "--rdzv_endpoint",
    "--rdzv-id",
    "--rdzv_id",
}
_NODE_RANK_FLAGS = {"--node-rank", "--node_rank"}
_VALUE_FLAGS = {
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


@dataclass(frozen=True)
class TorchrunInvocation:
    """Task-supplied torchrun intent after topology validation."""

    node_count: int
    local_world_size: int
    argv: tuple[str, ...]
    distributed: bool


@dataclass(frozen=True)
class BrokerAuthority:
    run_id: str
    pool_digest: str
    pool_size: int
    local_world_size: int


@dataclass(frozen=True)
class PublishedRequest:
    request_id: str


def _split_flag(value: str) -> tuple[str, str | None]:
    name, separator, inline = value.partition("=")
    return name, inline if separator else None


def _integer_value(
    argv: tuple[str, ...],
    offset: int,
    *,
    label: str,
) -> tuple[int, int]:
    name, inline = _split_flag(argv[offset])
    del name
    if inline is not None:
        raw = inline
        next_offset = offset + 1
    else:
        if offset + 1 >= len(argv):
            raise SetupError(f"torchrun {label} is missing a value")
        raw = argv[offset + 1]
        next_offset = offset + 2
    try:
        value = int(raw)
    except ValueError as error:
        raise SetupError(f"torchrun {label} must be a fixed integer") from error
    if value <= 0:
        raise SetupError(f"torchrun {label} must be positive")
    return value, next_offset


def parse_torchrun_invocation(
    argv: tuple[str, ...],
    *,
    expected_local_world_size: int,
) -> TorchrunInvocation:
    """Accept fixed task intent while rejecting cluster topology authority."""
    if expected_local_world_size <= 0:
        raise SetupError("expected local torchrun world size must be positive")

    node_count = 1
    local_world_size = expected_local_world_size
    entrypoint: str | None = None
    offset = 0
    while offset < len(argv):
        item = argv[offset]
        if item == "--":
            if offset + 1 < len(argv):
                entrypoint = argv[offset + 1]
            break
        if not item.startswith("-"):
            entrypoint = item
            break

        name, inline = _split_flag(item)
        if name in _NODE_RANK_FLAGS:
            raise SetupError("task torchrun arguments cannot set the node rank")
        if name in _TOPOLOGY_FLAGS or name == "--standalone":
            raise SetupError(
                "task torchrun arguments cannot set rendezvous topology"
            )
        if name == "--nnodes":
            node_count, offset = _integer_value(
                argv, offset, label="node count"
            )
            continue
        if name in {"--nproc-per-node", "--nproc_per_node"}:
            local_world_size, offset = _integer_value(
                argv, offset, label="local process count"
            )
            continue
        if inline is None and name in _VALUE_FLAGS:
            if offset + 1 >= len(argv):
                raise SetupError(f"torchrun option {name} is missing a value")
            offset += 2
            continue
        offset += 1

    if entrypoint is None:
        raise SetupError("torchrun requires a training entrypoint")
    if local_world_size != expected_local_world_size:
        raise SetupError(
            "torchrun local process count must match the allocated "
            f"{expected_local_world_size} GPUs per node"
        )
    return TorchrunInvocation(
        node_count=node_count,
        local_world_size=local_world_size,
        argv=argv,
        # The shim exists only inside an allocated multi-node phase.  Even a
        # one-node torchrun must lease one phase node so concurrent one-node
        # attempts cannot collide on the controller's GPUs.
        distributed=True,
    )


def forwarded_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Remove scheduler, container, identity, and secret-bearing host state."""
    forwarded: dict[str, str] = {}
    for name, value in environment.items():
        upper = name.upper()
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            continue
        if upper in _BLOCKED_NAMES or upper.startswith(_BLOCKED_PREFIXES):
            continue
        # TOKENIZER describes ordinary model configuration, not a credential.
        # Remove that word only for secret-name classification; a real secret
        # such as TOKENIZER_ACCESS_TOKEN still retains TOKEN and stays blocked.
        secret_name = upper.replace("TOKENIZER", "")
        if any(marker in secret_name for marker in _SECRET_MARKERS):
            continue
        forwarded[name] = value
    return forwarded


def _atomic_request(path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            if not payload.endswith("\n"):
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_request(
    root: Path,
    invocation: TorchrunInvocation,
    *,
    ready: Mapping[str, object],
    environment: Mapping[str, str],
    working_directory: str,
    request_id: str,
) -> PublishedRequest:
    """Publish closed task intent to only the injected current-phase broker."""
    try:
        if set(ready) != {
            "run_id",
            "pool_digest",
            "pool_size",
            "local_world_size",
        }:
            raise ValueError("unexpected authority fields")
        authority = BrokerAuthority(
            run_id=str(ready["run_id"]),
            pool_digest=str(ready["pool_digest"]),
            pool_size=int(ready["pool_size"]),
            local_world_size=int(ready["local_world_size"]),
        )
        if (
            not authority.run_id
            or re.fullmatch(r"[0-9a-f]{64}", authority.pool_digest) is None
            or authority.pool_size <= 0
            or authority.local_world_size <= 0
        ):
            raise ValueError("invalid authority values")
    except (KeyError, TypeError, ValueError) as error:
        raise SetupError("Blue Vela multi-node broker authority is invalid") from error
    if invocation.local_world_size != authority.local_world_size:
        raise SetupError("torchrun local process count differs from the phase pool")
    if not invocation.distributed:
        raise SetupError("one-node torchrun must use the local standard launcher")
    if invocation.node_count > authority.pool_size:
        raise SetupError(
            f"torchrun needs {invocation.node_count} nodes but the current "
            f"phase pool has {authority.pool_size}"
        )
    workdir = PurePosixPath(working_directory)
    if not workdir.is_absolute() or ".." in workdir.parts:
        raise SetupError("invalid multi-node torchrun working directory")
    if re.fullmatch(r"[0-9a-f]{32}", request_id) is None:
        raise SetupError("invalid multi-node torchrun request identity")
    payload = {
        "request_id": request_id,
        "run_id": authority.run_id,
        "pool_digest": authority.pool_digest,
        "node_count": invocation.node_count,
        "local_world_size": invocation.local_world_size,
        "argv": list(invocation.argv),
        "working_directory": str(workdir),
        "environment": forwarded_environment(environment),
    }
    request_path = Path(root) / "requests" / f"{request_id}.json"
    if not request_path.parent.is_dir():
        raise SetupError("Blue Vela multi-node broker request endpoint is missing")
    _atomic_request(
        request_path,
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
    )
    return PublishedRequest(request_id=request_id)


def wait_for_result(
    root: Path,
    *,
    request_id: str,
    node_count: int,
    output: BinaryIO,
    timeout_seconds: float | None = None,
    poll_interval: float = 0.2,
) -> int:
    """Wait for terminal evidence and emit bounded rank logs in rank order."""
    root = Path(root)
    result_path = root / "results" / f"{request_id}.json"
    heartbeat = root / "heartbeats" / request_id
    cancel = root / "cancel" / request_id
    descriptor = os.open(heartbeat, os.O_WRONLY | os.O_CREAT, 0o600)
    os.close(descriptor)
    started = time.monotonic()
    last_heartbeat = 0.0
    try:
        while not result_path.is_file():
            now = time.monotonic()
            if timeout_seconds is not None and now - started >= timeout_seconds:
                cancel.touch(mode=0o600, exist_ok=True)
                return 124
            if now - last_heartbeat >= 2:
                heartbeat.touch(exist_ok=True)
                last_heartbeat = now
            time.sleep(poll_interval)
        try:
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict):
                raise ValueError("result is not an object")
            ranks = result["ranks"]
            if not isinstance(ranks, list):
                raise ValueError("ranks are not an array")
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise SetupError("Blue Vela multi-node result is invalid") from error
        if result.get("request_id") != request_id:
            raise SetupError("Blue Vela multi-node result has the wrong identity")
        expected_ranks = tuple(range(node_count))
        try:
            actual_ranks = tuple(int(item["node_rank"]) for item in ranks)
            returncodes = tuple(int(item["returncode"]) for item in ranks)
        except (KeyError, TypeError, ValueError) as error:
            raise SetupError("Blue Vela multi-node rank result is invalid") from error
        if actual_ranks != expected_ranks:
            raise SetupError("Blue Vela multi-node result has incomplete ranks")
        for node_rank in expected_ranks:
            rank_output = (
                root
                / "results"
                / f"{request_id}.rank-{node_rank:04d}.out"
            )
            try:
                output.write(rank_output.read_bytes())
                output.flush()
            except OSError as error:
                raise SetupError(
                    f"Blue Vela multi-node rank {node_rank} output is missing"
                ) from error
        if result.get("cancelled") is True:
            return 143
        if result.get("timed_out") is True:
            return 124
        return next((code for code in returncodes if code != 0), 0)
    finally:
        heartbeat.unlink(missing_ok=True)


def run(
    argv: tuple[str, ...],
    *,
    environment: Mapping[str, str],
    working_directory: str,
    output: BinaryIO,
    execute: Callable[[str, tuple[str, ...]], object] = os.execv,
) -> int:
    """Execute local torchrun or transparently use the injected phase broker."""
    try:
        expected_local_world_size = int(environment["RSI_LOCAL_WORLD_SIZE"])
    except (KeyError, TypeError, ValueError) as error:
        raise SetupError(
            "transparent torchrun is missing RSI_LOCAL_WORLD_SIZE"
        ) from error
    invocation = parse_torchrun_invocation(
        argv,
        expected_local_world_size=expected_local_world_size,
    )
    if not invocation.distributed:
        execute(
            sys.executable,
            (
                sys.executable,
                "-m",
                "torch.distributed.run",
                *invocation.argv,
            ),
        )
        return 126

    raw_root = environment.get("RSI_MULTINODE_ROOT")
    if not raw_root:
        raise SetupError("transparent torchrun has no current-phase broker")
    root = Path(raw_root)
    try:
        ready = json.loads((root / "READY.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SetupError("Blue Vela multi-node broker is not ready") from error
    request = publish_request(
        root,
        invocation,
        ready=ready,
        environment=environment,
        working_directory=working_directory,
        request_id=secrets.token_hex(16),
    )
    cancel = root / "cancel" / request.request_id

    def request_cancel(_signum: int, _frame: object) -> None:
        cancel.touch(mode=0o600, exist_ok=True)

    previous_term = signal.signal(signal.SIGTERM, request_cancel)
    previous_int = signal.signal(signal.SIGINT, request_cancel)
    try:
        return wait_for_result(
            root,
            request_id=request.request_id,
            node_count=invocation.node_count,
            output=output,
        )
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def main() -> int:
    try:
        return run(
            tuple(sys.argv[1:]),
            environment=os.environ,
            working_directory=os.getcwd(),
            output=sys.stdout.buffer,
        )
    except Exception as error:
        sys.stderr.write(f"transparent Blue Vela torchrun failed: {error}\n")
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
