"""In-allocation node-pool authority for Blue Vela multi-node phases."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, field_validator

from rsi_harness.cluster.bluevela.allocation import AllocatedNode
from rsi_harness.errors import InfrastructureError
from rsi_harness.models import PersistedModel

RankRunner = Callable[
    [tuple[str, ...], threading.Event], tuple[int, bytes]
]

_REMOTE_OUTPUT_LIMIT = 16_000_000
_REMOTE_OUTPUT_READ_SIZE = 1024 * 1024
_OUTPUT_DRAIN_GRACE_SECONDS = 1.0
_OUTPUT_POLL_SECONDS = 0.05


def _valid_request_id(value: str) -> bool:
    return len(value) == 32 and all(
        character in "0123456789abcdef" for character in value
    )


class SubpoolLease(PersistedModel):
    """An immutable lease over ranks in one frozen phase pool."""

    lease_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    pool_ranks: tuple[int, ...]
    nodes: tuple[AllocatedNode, ...]


class MultiNodeRequest(PersistedModel):
    """Closed client intent; it deliberately has no host or phase fields."""

    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    run_id: str = Field(min_length=1)
    pool_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_count: int = Field(gt=0)
    local_world_size: int = Field(gt=0)
    argv: tuple[str, ...] = Field(min_length=1)
    working_directory: PurePosixPath
    environment: dict[str, str]

    @field_validator("working_directory")
    @classmethod
    def _absolute_working_directory(
        cls, value: PurePosixPath
    ) -> PurePosixPath:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("multi-node working directory must be absolute")
        return value


class RankResult(PersistedModel):
    pool_rank: int = Field(ge=0)
    node_rank: int = Field(ge=0)
    returncode: int


class MultiNodeResult(PersistedModel):
    request_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    ranks: tuple[RankResult, ...]
    cancelled: bool = False
    timed_out: bool = False


class BrokerReady(PersistedModel):
    run_id: str = Field(min_length=1)
    pool_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    pool_size: int = Field(gt=0)
    local_world_size: int = Field(gt=0)


class WorkerBind(PersistedModel):
    source: Path
    target: PurePosixPath
    read_only: bool = False

    @field_validator("source")
    @classmethod
    def _absolute_source(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("remote worker bind source must be absolute")
        return value

    @field_validator("target")
    @classmethod
    def _absolute_target(cls, value: PurePosixPath) -> PurePosixPath:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("remote worker bind target must be absolute")
        return value


class RemoteWorkerTemplate(PersistedModel):
    apptainer_binary: Path
    sif_path: Path
    sif_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    container_python: PurePosixPath
    container_workdir: PurePosixPath
    temp_root: Path
    network_mode: Literal["public", "no-network"]
    binds: tuple[WorkerBind, ...]
    environment: dict[str, str]

    @field_validator("apptainer_binary", "sif_path", "temp_root")
    @classmethod
    def _absolute_host_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("remote worker host paths must be absolute")
        return value

    @field_validator("container_python", "container_workdir")
    @classmethod
    def _absolute_container_path(
        cls, value: PurePosixPath
    ) -> PurePosixPath:
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError("remote worker container paths must be absolute")
        return value


class RemoteRequestControl(PersistedModel):
    phase: Literal["work", "verifier"]
    request: MultiNodeRequest
    lease: SubpoolLease
    rank_environment: tuple[dict[str, str], ...]
    worker: RemoteWorkerTemplate


class ReservedRequest(PersistedModel):
    lease: SubpoolLease
    control_path: Path
    commands: tuple[tuple[str, ...], ...]

    @field_validator("control_path")
    @classmethod
    def _absolute_control_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("broker control path must be absolute")
        return value


class SubpoolAllocator:
    """Allocate disjoint node subsets from one phase pool."""

    def __init__(self, nodes: tuple[AllocatedNode, ...]) -> None:
        if not nodes:
            raise InfrastructureError("a phase pool must contain at least one node")
        self._nodes = nodes
        self._leases: dict[str, tuple[int, ...]] = {}
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        return len(self._nodes)

    def acquire(self, node_count: int) -> SubpoolLease:
        if node_count <= 0:
            raise InfrastructureError("subpool node count must be positive")
        with self._lock:
            occupied = {
                rank for ranks in self._leases.values() for rank in ranks
            }
            available = tuple(
                rank for rank in range(len(self._nodes)) if rank not in occupied
            )
            if len(available) < node_count:
                raise InfrastructureError(
                    f"phase pool has {len(available)} free nodes; "
                    f"request needs {node_count}"
                )
            ranks = available[:node_count]
            lease_id = secrets.token_hex(16)
            self._leases[lease_id] = ranks
            return SubpoolLease(
                lease_id=lease_id,
                pool_ranks=ranks,
                nodes=tuple(self._nodes[rank] for rank in ranks),
            )

    def release(self, lease_id: str) -> None:
        with self._lock:
            if self._leases.pop(lease_id, None) is None:
                raise InfrastructureError(
                    f"unknown subpool lease {lease_id!r}"
                )


def _atomic_json(path: Path, value: PersistedModel | dict[str, object]) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = (
        value.model_dump(mode="json")
        if isinstance(value, PersistedModel)
        else value
    )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time_ns()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
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


def _atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time_ns()}")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
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


def time_ns() -> int:
    """Small seam for unique atomic-write names."""
    import time

    return time.time_ns()


class MultiNodeBroker:
    """Broker-owned authority for one already-frozen phase host pool."""

    def __init__(
        self,
        *,
        phase: Literal["work", "verifier"],
        run_id: str,
        nodes: tuple[AllocatedNode, ...],
        pool_digest: str,
        root: Path,
        authority_root: Path,
        local_world_size: int,
        worker_template: RemoteWorkerTemplate,
        remote_binary: str = "blaunch",
        remote_host_flag: str = "-z",
    ) -> None:
        if not run_id:
            raise InfrastructureError("multi-node broker run id is missing")
        if local_world_size <= 0:
            raise InfrastructureError("local world size must be positive")
        if not remote_binary or not remote_host_flag:
            raise InfrastructureError("remote launch profile is incomplete")
        self.phase = phase
        self.run_id = run_id
        self.nodes = nodes
        self.pool_digest = pool_digest
        self.root = Path(root).resolve()
        self.authority_root = Path(authority_root).resolve()
        self.local_world_size = local_world_size
        self.worker_template = worker_template
        self.remote_binary = remote_binary
        self.remote_host_flag = remote_host_flag
        self.allocator = SubpoolAllocator(nodes)
        self._active: dict[str, ReservedRequest] = {}
        self._lock = threading.RLock()
        self._initialized = False
        self._stop_event = threading.Event()
        self._server_thread: threading.Thread | None = None
        self._server_runner: RankRunner | None = None

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                raise InfrastructureError("multi-node broker is already initialized")
            self.root.mkdir(parents=True, mode=0o700, exist_ok=False)
            self.root.chmod(0o700)
            for name in ("cancel", "heartbeats", "requests", "results"):
                directory = self.root / name
                directory.mkdir(mode=0o700)
                directory.chmod(0o700)
            controls = self.authority_root / "requests"
            controls.mkdir(parents=True, mode=0o700, exist_ok=False)
            claimed = self.authority_root / "claimed"
            claimed.mkdir(mode=0o700)
            self.authority_root.chmod(0o700)
            controls.chmod(0o700)
            claimed.chmod(0o700)
            ready = BrokerReady(
                run_id=self.run_id,
                pool_digest=self.pool_digest,
                pool_size=len(self.nodes),
                local_world_size=self.local_world_size,
            )
            _atomic_json(self.root / "READY.json", ready)
            self._initialized = True

    def _validate(self, request: MultiNodeRequest) -> None:
        if not self._initialized:
            raise InfrastructureError("multi-node broker is not initialized")
        if request.run_id != self.run_id:
            raise InfrastructureError("multi-node request has the wrong run authority")
        if request.pool_digest != self.pool_digest:
            raise InfrastructureError("multi-node request has the wrong pool authority")
        if request.local_world_size != self.local_world_size:
            raise InfrastructureError(
                "multi-node request local world size differs from its phase pool"
            )
        from rsi_harness.cluster.bluevela.torchrun_shim import (
            forwarded_environment,
            parse_torchrun_invocation,
        )

        if forwarded_environment(request.environment) != request.environment:
            raise InfrastructureError(
                "multi-node request contains forbidden environment state"
            )
        try:
            invocation = parse_torchrun_invocation(
                request.argv,
                expected_local_world_size=self.local_world_size,
            )
        except Exception as error:
            raise InfrastructureError(
                f"multi-node request has invalid torchrun intent: {error}"
            ) from error
        if (
            not invocation.distributed
            or invocation.node_count != request.node_count
            or request.node_count > len(self.nodes)
        ):
            raise InfrastructureError(
                "multi-node request exceeds or differs from its phase pool"
            )

    def reserve(self, request: MultiNodeRequest) -> ReservedRequest:
        """Validate then atomically lease nodes and freeze worker authority."""
        self._validate(request)
        with self._lock:
            if request.request_id in self._active:
                raise InfrastructureError(
                    f"duplicate multi-node request {request.request_id}"
                )
            lease = self.allocator.acquire(request.node_count)
            environments = rank_environments(
                lease.nodes,
                pool_size=len(self.nodes),
                pool_ranks=lease.pool_ranks,
                local_world_size=self.local_world_size,
                request_id=request.request_id,
            )
            control = RemoteRequestControl(
                phase=self.phase,
                request=request,
                lease=lease,
                rank_environment=environments,
                worker=self.worker_template,
            )
            control_path = (
                self.authority_root / "requests" / f"{request.request_id}.json"
            )
            try:
                _atomic_json(control_path, control)
                commands = tuple(
                    (
                        self.remote_binary,
                        self.remote_host_flag,
                        node.host,
                        sys.executable,
                        "-m",
                        "rsi_harness.cluster.bluevela.remote_worker",
                        "run",
                        "--control",
                        str(control_path),
                        "--request",
                        request.request_id,
                        "--pool-rank",
                        str(pool_rank),
                        "--node-rank",
                        str(node_rank),
                    )
                    for node_rank, (pool_rank, node) in enumerate(
                        zip(lease.pool_ranks, lease.nodes, strict=True)
                    )
                )
                reserved = ReservedRequest(
                    lease=lease,
                    control_path=control_path,
                    commands=commands,
                )
                self._active[request.request_id] = reserved
                return reserved
            except Exception:
                control_path.unlink(missing_ok=True)
                self.allocator.release(lease.lease_id)
                raise

    def release(self, reserved: ReservedRequest) -> None:
        with self._lock:
            request_id = reserved.control_path.stem
            active = self._active.get(request_id)
            if active != reserved:
                raise InfrastructureError(
                    f"unknown active multi-node request {request_id!r}"
                )
            reserved.control_path.unlink(missing_ok=True)
            self.allocator.release(reserved.lease.lease_id)
            del self._active[request_id]

    def require_idle(self) -> None:
        with self._lock:
            if self._active:
                raise InfrastructureError(
                    "phase broker has active remote requests: "
                    + ", ".join(sorted(self._active))
                )

    def start(self, *, rank_runner: RankRunner | None = None) -> None:
        """Start the private request server for this phase."""
        with self._lock:
            if self._server_thread is not None:
                raise InfrastructureError("multi-node broker is already running")
            if not self._initialized:
                self.initialize()
            self._server_runner = rank_runner
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._serve,
                name=f"bluevela-{self.phase}-broker",
                daemon=True,
            )
            self._server_thread = thread
            thread.start()

    def stop(self) -> None:
        """Cancel exact active requests and stop accepting new work."""
        with self._lock:
            thread = self._server_thread
            if thread is None:
                return
            self._stop_event.set()
            active = tuple(self._active)
        for request_id in active:
            (self.root / "cancel" / request_id).touch(mode=0o600, exist_ok=True)
        thread.join(timeout=60)
        if thread.is_alive():
            raise InfrastructureError("multi-node broker did not stop")
        with self._lock:
            self._server_thread = None
        (self.root / "READY.json").unlink(missing_ok=True)

    def _serve(self) -> None:
        with ThreadPoolExecutor(
            max_workers=len(self.nodes),
            thread_name_prefix=f"{self.phase}-request",
        ) as executor:
            futures = set()
            while not self._stop_event.is_set():
                for request_path in sorted((self.root / "requests").glob("*.json")):
                    claimed = self.authority_root / "claimed" / request_path.name
                    try:
                        request_path.replace(claimed)
                    except FileNotFoundError:
                        continue
                    futures.add(executor.submit(self._handle_claimed, claimed))
                futures = {future for future in futures if not future.done()}
                time.sleep(0.01)
            for request_id in tuple(self._active):
                (self.root / "cancel" / request_id).touch(
                    mode=0o600, exist_ok=True
                )

    def _handle_claimed(self, path: Path) -> None:
        request_id = path.stem
        try:
            request = MultiNodeRequest.model_validate_json(path.read_text())
            if request.request_id != request_id:
                raise InfrastructureError(
                    "multi-node request filename differs from its identity"
                )
            self.execute(request, rank_runner=self._server_runner)
        except Exception:
            if _valid_request_id(request_id):
                failure = MultiNodeResult(request_id=request_id, ranks=())
                _atomic_json(self.root / "results" / f"{request_id}.json", failure)
        finally:
            path.unlink(missing_ok=True)

    def _default_rank_runner(
        self,
        command: tuple[str, ...],
        cancelled: threading.Event,
    ) -> tuple[int, bytes]:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        stdout_fd = process.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        captured = bytearray()
        truncated = False
        eof = False

        def drain_available() -> bool:
            """Drain bytes without requiring every descendant to close stdout."""
            nonlocal eof
            nonlocal truncated
            read_any = False
            while True:
                try:
                    chunk = os.read(stdout_fd, _REMOTE_OUTPUT_READ_SIZE)
                except BlockingIOError:
                    return read_any
                if not chunk:
                    eof = True
                    return read_any
                read_any = True
                if len(captured) < _REMOTE_OUTPUT_LIMIT:
                    remaining = _REMOTE_OUTPUT_LIMIT - len(captured)
                    captured.extend(chunk[:remaining])
                    truncated = truncated or len(chunk) > remaining
                else:
                    truncated = True

        cleanup_started = False
        while process.poll() is None:
            drain_available()
            if cancelled.is_set() and not cleanup_started:
                cleanup_started = True
                try:
                    subprocess.run(
                        stop_command(
                            command,
                            remote_binary=self.remote_binary,
                            remote_host_flag=self.remote_host_flag,
                        ),
                        check=False,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=20,
                    )
                except subprocess.TimeoutExpired:
                    pass
                process.terminate()
            time.sleep(_OUTPUT_POLL_SECONDS)
        deadline = time.monotonic() + _OUTPUT_DRAIN_GRACE_SECONDS
        while not eof:
            read_any = drain_available()
            if eof or time.monotonic() >= deadline:
                break
            if not read_any:
                time.sleep(_OUTPUT_POLL_SECONDS)
        process.stdout.close()
        if truncated:
            captured.extend(b"\n...[remote rank output truncated]...\n")
        assert process.returncode is not None
        return process.returncode, bytes(captured)

    def execute(
        self,
        request: MultiNodeRequest,
        *,
        rank_runner: RankRunner | None = None,
    ) -> MultiNodeResult:
        """Execute one validated request and publish terminal rank evidence."""
        runner = rank_runner or self._default_rank_runner
        reserved = self.reserve(request)
        cancelled = threading.Event()
        cancel_path = self.root / "cancel" / request.request_id
        results: list[RankResult] = []
        outputs: dict[int, bytes] = {}
        try:
            with ThreadPoolExecutor(
                max_workers=len(reserved.commands),
                thread_name_prefix=f"{self.phase}-rank",
            ) as executor:
                futures = {
                    node_rank: executor.submit(runner, command, cancelled)
                    for node_rank, command in enumerate(reserved.commands)
                }
                while not all(future.done() for future in futures.values()):
                    if cancel_path.exists():
                        cancelled.set()
                    time.sleep(0.01)
                for node_rank in sorted(futures):
                    pool_rank = reserved.lease.pool_ranks[node_rank]
                    try:
                        returncode, output = futures[node_rank].result()
                    except Exception as error:
                        cancelled.set()
                        returncode = 125
                        output = f"remote rank infrastructure error: {error}\n".encode()
                    results.append(
                        RankResult(
                            pool_rank=pool_rank,
                            node_rank=node_rank,
                            returncode=returncode,
                        )
                    )
                    outputs[node_rank] = output[:16_000_040]
            for node_rank, output in sorted(outputs.items()):
                _atomic_bytes(
                    self.root
                    / "results"
                    / f"{request.request_id}.rank-{node_rank:04d}.out",
                    output,
                )
            result = MultiNodeResult(
                request_id=request.request_id,
                ranks=tuple(results),
                cancelled=cancelled.is_set(),
            )
            _atomic_json(
                self.root / "results" / f"{request.request_id}.json",
                result,
            )
            return result
        finally:
            cancel_path.unlink(missing_ok=True)
            self.release(reserved)


def stop_command(
    command: tuple[str, ...],
    *,
    remote_binary: str = "blaunch",
    remote_host_flag: str = "-z",
) -> tuple[str, ...]:
    """Change only the frozen remote-worker action from run to stop."""
    if (
        len(command) < 7
        or command[:2] != (remote_binary, remote_host_flag)
        or command[5] != "rsi_harness.cluster.bluevela.remote_worker"
        or command[6] != "run"
    ):
        raise InfrastructureError("cannot derive cleanup from an unsafe rank command")
    return (*command[:6], "stop", *command[7:])


def rank_environments(
    selected: tuple[AllocatedNode, ...],
    *,
    pool_size: int,
    pool_ranks: tuple[int, ...],
    local_world_size: int,
    request_id: str,
) -> tuple[dict[str, str], ...]:
    """Build rank-local topology for only the leased phase subpool."""
    if not selected or len(selected) != len(pool_ranks):
        raise InfrastructureError(
            "selected nodes and phase pool ranks must be non-empty and aligned"
        )
    if pool_size <= 0 or local_world_size <= 0:
        raise InfrastructureError("pool and local world sizes must be positive")
    if len(set(pool_ranks)) != len(pool_ranks) or any(
        rank < 0 or rank >= pool_size for rank in pool_ranks
    ):
        raise InfrastructureError("phase pool ranks are invalid")

    master_addr = selected[0].ipv4
    port_offset = int(hashlib.sha256(request_id.encode()).hexdigest()[:8], 16)
    master_port = str(20_000 + port_offset % 30_000)
    node_count = len(selected)
    return tuple(
        {
            "RSI_NODE_RANK": str(node_rank),
            "RSI_POOL_RANK": str(pool_rank),
            "RSI_MASTER_ADDR": master_addr,
            "RSI_MASTER_PORT": master_port,
            "RSI_NUM_NODES": str(node_count),
            "RSI_POOL_SIZE": str(pool_size),
            "RSI_LOCAL_WORLD_SIZE": str(local_world_size),
        }
        for node_rank, pool_rank in enumerate(pool_ranks)
    )
