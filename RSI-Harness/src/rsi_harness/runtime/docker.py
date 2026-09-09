"""Safe, bounded adapter around the pinned Docker SDK surface."""

from __future__ import annotations

import codecs
import os
import shutil
import socket
import stat
import tarfile
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from docker.errors import APIError, NotFound
from docker.types import DeviceRequest, Mount

from rsi_harness.errors import (
    ContainerExecNotStartedError,
    InfrastructureError,
    SetupError,
)
from rsi_harness.models import (
    WORK_FEEDBACK_ROOT,
    AgentRunResult,
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerVolumeMount,
    ManagedNetwork,
    WorkQuiescence,
)
from rsi_harness.runtime.gpu import (
    NVIDIA_VISIBLE_DEVICES_ENV,
    NVIDIA_VISIBLE_DEVICES_VOID,
    nvidia_visible_devices_value,
)
from rsi_harness.runtime.local_auth import AgentAuthMaterial
from rsi_harness.runtime.mount_topology import validate_split_workdir_target
from rsi_harness.runtime.network import (
    NetworkPolicyEnforcer,
    NetworkPolicyLease,
    managed_bridge_interface,
)
from rsi_harness.runtime.redaction import redact_exact_values, redact_text
from rsi_harness.runtime.workdir_volume import DockerWorkdirVolumeBackend

_STAGING_TARGET = PurePosixPath("/run/rsi-harness/staging")
_DOCKER_SOCKET_TARGETS = frozenset(
    {PurePosixPath("/var/run/docker.sock"), PurePosixPath("/run/docker.sock")}
)
_MAX_INJECTED_TEST_ENTRIES = 100_000
_MAX_INJECTED_TEST_REGULAR_BYTES = 1_073_741_824


@dataclass(frozen=True, slots=True)
class DockerOutput:
    output: str
    output_truncated: bool


@dataclass(frozen=True, slots=True)
class _ArchiveEntry:
    name: str
    metadata: os.stat_result
    linkname: str | None = None


@dataclass(frozen=True, slots=True)
class _TestArchivePlan:
    root_descriptor: int
    root_metadata: os.stat_result
    entries: tuple[_ArchiveEntry, ...]


class _ChunkedReadStream:
    """Iterator that deliberately exposes neither read nor a pipe length."""

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    def __iter__(self) -> Iterable[bytes]:
        while chunk := self._reader.read(64 * 1024):
            yield chunk

    def close(self) -> None:
        self._reader.close()


class _BoundedBytes:
    def __init__(self, limit: int) -> None:
        if limit < 0:
            raise ValueError("output limit must be non-negative")
        self.limit = limit
        self.total = 0
        self._head_limit = (limit + 1) // 2
        self._tail_limit = limit - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._lock = threading.Lock()

    def append(self, chunk: bytes) -> None:
        with self._lock:
            self.total += len(chunk)
            needed = self._head_limit - len(self._head)
            if needed > 0:
                self._head.extend(chunk[:needed])
                chunk = chunk[needed:]
            if self._tail_limit and chunk:
                self._tail.extend(chunk)
                if len(self._tail) > self._tail_limit:
                    del self._tail[: len(self._tail) - self._tail_limit]

    @property
    def value(self) -> bytes:
        with self._lock:
            return bytes(self._head + self._tail)

    @property
    def truncated(self) -> bool:
        with self._lock:
            return self.total > self.limit


class _RedactedOutputWriter:
    """Stream redacted text to an artifact and/or a best-effort live sink."""

    def __init__(
        self,
        path: Path | None,
        secrets: Sequence[str],
        callback: Callable[[str], None] | None = None,
    ) -> None:
        self._stream = None
        if path is not None:
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            self._stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="")
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._pending = ""
        self._secrets = tuple(secrets)
        self._callback = callback

    def append(self, chunk: bytes) -> None:
        self._pending += self._decoder.decode(chunk)
        while (newline := self._pending.find("\n")) >= 0:
            self._write(self._pending[: newline + 1])
            self._pending = self._pending[newline + 1 :]
        if self._stream is not None:
            self._stream.flush()

    def finish(self) -> None:
        try:
            self._pending += self._decoder.decode(b"", final=True)
            if self._pending:
                self._write(self._pending)
                self._pending = ""
            if self._stream is not None:
                self._stream.flush()
                os.fsync(self._stream.fileno())
        finally:
            if self._stream is not None:
                self._stream.close()

    def abort(self) -> None:
        if self._stream is not None and not self._stream.closed:
            self._stream.close()

    def _write(self, value: str) -> None:
        safe = redact_text(redact_exact_values(value, self._secrets))
        if self._stream is not None:
            self._stream.write(safe)
        if self._callback is not None:
            try:
                self._callback(safe)
            except Exception:
                # Console/progress rendering is observational and must never
                # disturb Agent execution or durable artifact capture.
                pass


class _RawOutputWriter:
    """Atomically publish exact task-authored bytes after a durable drain."""

    def __init__(self, name: str, directory_descriptor: int) -> None:
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            os.close(directory_descriptor)
            raise OSError("unsafe output artifact name")
        self._directory_descriptor = directory_descriptor
        self._name = name
        self._temporary_name = f".{name}.{uuid.uuid4().hex}"
        self._lock = threading.Lock()
        self._closed = False
        self._abort_requested = threading.Event()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(
                self._temporary_name,
                flags,
                0o644,
                dir_fd=self._directory_descriptor,
            )
        except BaseException:
            os.close(self._directory_descriptor)
            raise
        try:
            os.fchmod(descriptor, 0o644)
            stream = os.fdopen(descriptor, "wb")
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            try:
                os.unlink(
                    self._temporary_name, dir_fd=self._directory_descriptor
                )
            except FileNotFoundError:
                pass
            os.close(self._directory_descriptor)
            self._directory_descriptor = -1
            raise
        self._stream = stream

    def append(self, chunk: bytes) -> None:
        with self._lock:
            if self._closed or self._abort_requested.is_set():
                raise OSError("complete output artifact is already closed")
            self._stream.write(chunk)

    def finish(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._abort_requested.is_set():
                self._discard_unpublished()
                return
            published = False
            try:
                self._stream.flush()
                if self._abort_requested.is_set():
                    self._discard_unpublished()
                    return
                os.fsync(self._stream.fileno())
                if self._abort_requested.is_set():
                    self._discard_unpublished()
                    return
                self._stream.close()
                os.replace(
                    self._temporary_name,
                    self._name,
                    src_dir_fd=self._directory_descriptor,
                    dst_dir_fd=self._directory_descriptor,
                )
                published = True
                if self._abort_requested.is_set():
                    os.unlink(self._name, dir_fd=self._directory_descriptor)
                    self._closed = True
                    return
                os.fsync(self._directory_descriptor)
                if self._abort_requested.is_set():
                    os.unlink(self._name, dir_fd=self._directory_descriptor)
                    self._closed = True
                    return
                self._closed = True
            except BaseException:
                if not self._stream.closed:
                    self._stream.close()
                target = self._name if published else self._temporary_name
                try:
                    os.unlink(target, dir_fd=self._directory_descriptor)
                except FileNotFoundError:
                    pass
                self._closed = True
                raise
            finally:
                self._close_directory()

    def abort(self) -> None:
        self._abort_requested.set()
        if not self._lock.acquire(blocking=False):
            return
        try:
            if self._closed:
                return
            self._discard_unpublished()
        finally:
            self._lock.release()

    def _discard_unpublished(self) -> None:
        if not self._stream.closed:
            self._stream.close()
        try:
            os.unlink(self._temporary_name, dir_fd=self._directory_descriptor)
        except FileNotFoundError:
            pass
        self._closed = True
        self._close_directory()

    def _close_directory(self) -> None:
        descriptor = self._directory_descriptor
        if descriptor < 0:
            return
        self._directory_descriptor = -1
        os.close(descriptor)


def _close_docker_stream(stream: object) -> None:
    response = getattr(stream, "_response", None)
    close = getattr(stream, "close", None)
    try:
        if close is not None:
            close()
    except Exception:
        pass
    finally:
        response_close = getattr(response, "close", None)
        try:
            if response_close is not None:
                response_close()
        except Exception:
            pass


class DockerContainerRuntime:
    """Create and operate labeled containers without Docker escape hatches."""

    def __init__(
        self,
        client: Any,
        *,
        run_id: str,
        task_id: str,
        role: str,
        task_source_dir: Path,
        allowed_mount_roots: Sequence[Path],
        network: ManagedNetwork | None = None,
        network_policy_enforcer: NetworkPolicyEnforcer | None = None,
        staging_dir: Path | None = None,
        exec_output_limit_bytes: int = 1_000_000,
        log_output_limit_bytes: int = 1_000_000,
        pause_timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.05,
        omit_gpu_device_requests_for_tests: bool = False,
        workdir_volume_references: Sequence[ContainerRef] = (),
        work_feedback_dir: Path | None = None,
    ) -> None:
        if role not in {"work", "judge", "helper"}:
            raise ValueError(f"unsupported container role {role!r}")
        self._client = client
        self._run_id = run_id
        self._task_id = task_id
        self._role = role
        self._task_source_dir = Path(task_source_dir).resolve(strict=True)
        self._allowed_mount_roots = tuple(
            Path(path).resolve(strict=True) for path in allowed_mount_roots
        )
        if not self._allowed_mount_roots:
            raise SetupError("at least one Engine-owned mount root is required")
        if network is not None and not isinstance(network, ManagedNetwork):
            raise SetupError("only an Engine-created network identity is accepted")
        self._network = network
        self._network_policy_enforcer = network_policy_enforcer
        if network is not None:
            self._validate_managed_network(network)
        self._staging_dir = Path(staging_dir).resolve() if staging_dir else None
        if self._staging_dir is not None:
            self._require_engine_owned(self._staging_dir)
        self._exec_output_limit_bytes = exec_output_limit_bytes
        self._log_output_limit_bytes = log_output_limit_bytes
        self._pause_timeout_seconds = pause_timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._omit_gpu_device_requests_for_tests = (
            omit_gpu_device_requests_for_tests
        )
        self._workdir_volume_references = tuple(workdir_volume_references)
        if not all(
            isinstance(reference, ContainerRef)
            for reference in self._workdir_volume_references
        ):
            raise SetupError("managed WORKDIR volume references must be typed")
        if work_feedback_dir is not None and role != "work":
            raise SetupError("feedback mounts are limited to Work")
        self._work_feedback_dir = (
            None if work_feedback_dir is None else Path(work_feedback_dir)
        )
        self._work_feedback_identity: tuple[int, int] | None = None
        if self._work_feedback_dir is not None:
            try:
                feedback_descriptor = self._open_engine_directory(
                    self._work_feedback_dir
                )
            except OSError as error:
                raise SetupError(
                    "Work feedback directory is missing, non-directory, or "
                    "symlinked"
                ) from error
            try:
                metadata = os.fstat(feedback_descriptor)
                self._work_feedback_identity = (
                    metadata.st_dev,
                    metadata.st_ino,
                )
            finally:
                os.close(feedback_descriptor)
        self._container_networks: dict[str, str] = {}
        self._policy_leases: dict[str, NetworkPolicyLease] = {}

    @property
    def recovery_labels(self) -> dict[str, str]:
        return {
            "rsi-harness.run-id": self._run_id,
            "rsi-harness.task-id": self._task_id,
            "rsi-harness.role": self._role,
        }

    def create(
        self, spec: ContainerSpec, *, planned_name: str | None = None
    ) -> ContainerRef:
        if len(spec.volume_mounts) > 1:
            raise SetupError("exactly one managed WORKDIR volume mount is allowed")
        for volume_mount in spec.volume_mounts:
            validate_split_workdir_target(volume_mount.target)
        mounts = list(spec.mounts)
        if self._staging_dir is not None:
            self._staging_dir.mkdir(parents=True, exist_ok=True)
            mounts.append(
                ContainerMount(
                    source=self._staging_dir,
                    target=_STAGING_TARGET,
                    read_only=True,
                )
            )
        typed_mounts = [self._mount(mount, workdir=spec.workdir) for mount in mounts]
        typed_mounts.extend(
            self._volume_mount(mount, workdir=spec.workdir)
            for mount in spec.volume_mounts
        )
        labels = dict(spec.labels)
        conflicts = labels.keys() & self.recovery_labels.keys()
        if conflicts:
            raise SetupError("runtime recovery labels cannot be overridden")
        labels.update(self.recovery_labels)
        device_requests: list[DeviceRequest] = []
        if (
            spec.gpu_allocation.devices
            and not self._omit_gpu_device_requests_for_tests
        ):
            device_requests.append(
                DeviceRequest(
                    driver="nvidia",
                    device_ids=list(spec.gpu_allocation.uuids),
                    capabilities=[["gpu"]],
                )
            )
        environment = dict(spec.environment)
        environment[NVIDIA_VISIBLE_DEVICES_ENV] = (
            NVIDIA_VISIBLE_DEVICES_VOID
            if self._omit_gpu_device_requests_for_tests
            else nvidia_visible_devices_value(spec.gpu_allocation)
        )
        kwargs: dict[str, object] = {
            "working_dir": str(spec.workdir) if spec.workdir is not None else None,
            "user": spec.user,
            "environment": environment,
            "mounts": typed_mounts,
            "device_requests": device_requests,
            "labels": labels,
            "privileged": False,
            "cap_drop": ["NET_RAW"],
            "detach": True,
        }
        if spec.shm_size is not None:
            kwargs["shm_size"] = spec.shm_size
        if spec.cpus is not None:
            kwargs["nano_cpus"] = int(spec.cpus * 1_000_000_000)
        if spec.memory_mb is not None:
            kwargs["mem_limit"] = f"{spec.memory_mb}m"
        if spec.tmpfs:
            kwargs["tmpfs"] = {
                str(tmpfs.target): tmpfs.options for tmpfs in spec.tmpfs
            }
        if planned_name is not None:
            if planned_name != self.planned_container_name(planned_name):
                raise SetupError("planned container name is not canonical")
            kwargs["name"] = planned_name
        if self._network is not None:
            kwargs["network"] = self._network.name
        else:
            kwargs["network_mode"] = "none"
        try:
            container = self._client.containers.create(
                spec.image, list(spec.command), **kwargs
            )
        except APIError as error:
            raise SetupError(
                f"failed to create {self._role.title()} container: {error}"
            ) from error
        ref = ContainerRef(container_id=container.id, role=self._role)
        if self._network is not None:
            self._container_networks[ref.container_id] = self._network.network_id
        return ref

    def attest_workdir_volume_mount(
        self, container: ContainerRef, mount: ContainerVolumeMount
    ) -> None:
        """Attest the post-create mount and exact role-aware reference set."""
        if container.role != self._role:
            raise InfrastructureError(
                "recovery_required: WORKDIR mount container role differs from runtime"
            )
        expected_references = self._workdir_volume_references + (container,)
        DockerWorkdirVolumeBackend(self._client).inspect(
            mount.volume,
            expected_references=expected_references,
        )
        try:
            docker_container = self._client.containers.get(container.container_id)
            docker_container.reload()
            mounts = docker_container.attrs["Mounts"]
            if not isinstance(mounts, (list, tuple)):
                raise TypeError("Docker Mounts is not a sequence")
            at_target = [
                candidate
                for candidate in mounts
                if isinstance(candidate, Mapping)
                and candidate.get("Destination") == str(mount.target)
            ]
            exact = (
                len(at_target) == 1
                and at_target[0].get("Type") == "volume"
                and at_target[0].get("Name") == mount.volume.name
                and at_target[0].get("Destination") == str(mount.target)
                and at_target[0].get("RW") is (not mount.read_only)
            )
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: managed WORKDIR mount attestation is "
                f"unproven: {error}"
            ) from error
        if not exact:
            raise InfrastructureError(
                "recovery_required: managed WORKDIR mount differs from exact authority"
            )

    def attest_work_feedback_mount(self, container: ContainerRef) -> None:
        """Attest Work's exact, immutable feedback bind before container start."""
        if (
            self._role != "work"
            or container.role != "work"
            or self._work_feedback_dir is None
            or self._work_feedback_identity is None
        ):
            raise InfrastructureError(
                "recovery_required: Work feedback mount attestation authority "
                "is unavailable"
            )
        try:
            directory_descriptor = self._open_engine_directory(
                self._work_feedback_dir
            )
            try:
                metadata = os.fstat(directory_descriptor)
            finally:
                os.close(directory_descriptor)
            if (metadata.st_dev, metadata.st_ino) != self._work_feedback_identity:
                raise RuntimeError("feedback source directory identity changed")
            docker_container = self._client.containers.get(container.container_id)
            docker_container.reload()
            mounts = docker_container.attrs["Mounts"]
            if not isinstance(mounts, (list, tuple)):
                raise TypeError("Docker Mounts is not a sequence")
            at_target = [
                candidate
                for candidate in mounts
                if isinstance(candidate, Mapping)
                and candidate.get("Destination") == str(WORK_FEEDBACK_ROOT)
            ]
            exact = (
                len(at_target) == 1
                and at_target[0].get("Type") == "bind"
                and at_target[0].get("Source")
                == str(self._work_feedback_dir)
                and at_target[0].get("RW") is False
            )
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: Work feedback mount attestation is "
                f"unproven: {error}"
            ) from error
        if not exact:
            raise InfrastructureError(
                "recovery_required: Work feedback mount attestation differs "
                "from exact read-only authority"
            )

    def start(self, container: ContainerRef) -> None:
        network_id = self._container_networks.get(container.container_id)
        lease = self._policy_leases.get(container.container_id)
        if network_id is None or lease is None:
            raise SetupError(
                "container start requires an installed network policy lease"
            )
        if self._network_policy_enforcer is None:
            raise SetupError("container start has no authoritative policy enforcer")
        self._network_policy_enforcer.attest(lease)
        try:
            self._container(container).start()
        except APIError as error:
            raise InfrastructureError(
                f"failed to start {self._role.title()} container: {error}"
            ) from error

    def install_network_policy(
        self, container: ContainerRef, lease: NetworkPolicyLease
    ) -> None:
        network_id = self._container_networks.get(container.container_id)
        managed = self._network
        enforcer = self._network_policy_enforcer
        if (
            network_id is None
            or managed is None
            or enforcer is None
            or container.role != self._role
            or lease.network_id != network_id
            or lease.network_name != managed.name
            or lease.rules.network_id != network_id
            or lease.rules.network_name != managed.name
            or lease.rules.container_id != container.container_id
            or lease.rules.role != self._role
        ):
            raise SetupError(
                "network policy lease does not match container, network, and role"
            )
        enforcer.attest(lease)
        self._policy_leases[container.container_id] = lease

    def pause(self, container: ContainerRef) -> None:
        try:
            docker_container = self._container(container)
            docker_container.pause()
        except APIError as error:
            raise InfrastructureError(
                f"failed to pause {self._role.title()} container "
                f"{container.container_id}: {error}"
            ) from error
        deadline = time.monotonic() + self._pause_timeout_seconds
        while True:
            try:
                docker_container.reload()
            except APIError as error:
                raise InfrastructureError(
                    f"failed to inspect {self._role.title()} container "
                    f"{container.container_id} after pause: {error}"
                ) from error
            if docker_container.attrs.get("State", {}).get("Paused") is True:
                return
            if time.monotonic() >= deadline:
                raise InfrastructureError("Docker did not report the container paused")
            time.sleep(self._poll_interval_seconds)

    def unpause(self, container: ContainerRef) -> None:
        try:
            docker_container = self._container(container)
            docker_container.reload()
        except APIError as error:
            raise InfrastructureError(
                f"failed to inspect {self._role.title()} container "
                f"{container.container_id} before unpause: {error}"
            ) from error
        if docker_container.attrs.get("State", {}).get("Paused") is True:
            try:
                docker_container.unpause()
            except APIError as error:
                raise InfrastructureError(
                    f"failed to unpause {self._role.title()} container "
                    f"{container.container_id}: {error}"
                ) from error

    def inspect_quiescence(
        self, container: ContainerRef
    ) -> WorkQuiescence | None:
        """Return exact paused/stopped state, or ``None`` when still running."""
        if self._role != "work" or container.role != "work":
            raise SetupError("quiescence inspection requires the exact Work runtime")
        try:
            docker_container = self._container(container)
            docker_container.reload()
            state = docker_container.attrs.get("State")
        except (APIError, AttributeError, KeyError, TypeError) as error:
            raise InfrastructureError(
                f"failed to inspect exact Work container "
                f"{container.container_id} quiescence"
            ) from error
        if not isinstance(state, dict):
            raise InfrastructureError("Docker returned malformed Work state")
        running = state.get("Running")
        paused = state.get("Paused")
        if running is True and paused is True:
            return WorkQuiescence.PAUSED
        if running is False and paused is False:
            return WorkQuiescence.STOPPED
        if running is True and paused is False:
            return None
        raise InfrastructureError("Docker returned ambiguous Work quiescence")

    def stop(self, container: ContainerRef) -> None:
        try:
            self._container(container).stop(timeout=10)
        except NotFound:
            return
        except APIError as error:
            raise InfrastructureError(
                f"failed to stop {self._role.title()} container "
                f"{container.container_id}: {error}"
            ) from error

    def is_stopped_or_gone(self, container: ContainerRef) -> bool:
        """Authoritatively prove a container is no longer executing."""
        try:
            docker_container = self._container(container)
            docker_container.reload()
        except NotFound:
            return True
        except APIError as error:
            raise InfrastructureError(
                f"failed to inspect {self._role.title()} container "
                f"{container.container_id} for containment: {error}"
            ) from error
        running = docker_container.attrs.get("State", {}).get("Running")
        if not isinstance(running, bool):
            raise InfrastructureError(
                f"ambiguous {self._role.title()} container containment state "
                f"for {container.container_id}"
            )
        return running is False

    def remove(self, container: ContainerRef) -> None:
        try:
            self._container(container).remove(force=True, v=True)
        except NotFound:
            return
        except APIError as error:
            raise InfrastructureError(
                f"failed to remove {self._role.title()} container "
                f"{container.container_id}: {error}"
            ) from error

    def logs(self, container: ContainerRef, *, tail: int = 1000) -> DockerOutput:
        output = _BoundedBytes(self._log_output_limit_bytes)
        try:
            stream = self._container(container).logs(
                stdout=True,
                stderr=True,
                timestamps=False,
                tail=tail,
                stream=True,
                follow=False,
            )
            chunks = (stream,) if isinstance(stream, bytes) else stream
            try:
                for chunk in chunks:
                    if chunk:
                        output.append(chunk)
            finally:
                _close_docker_stream(stream)
        except APIError as error:
            raise InfrastructureError(
                f"failed to read logs from {self._role.title()} container "
                f"{container.container_id}: {error}"
            ) from error
        return DockerOutput(
            output=output.value.decode("utf-8", errors="replace"),
            output_truncated=output.truncated,
        )

    def processes(self, container: ContainerRef) -> tuple[int, ...]:
        try:
            table = self._container(container).top(ps_args="-eo pid")
        except APIError as error:
            raise InfrastructureError(
                f"failed to list processes in {self._role.title()} container "
                f"{container.container_id}: {error}"
            ) from error
        try:
            column = table["Titles"].index("PID")
            return tuple(int(row[column]) for row in table["Processes"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise InfrastructureError(
                f"ambiguous Docker process listing: {error}"
            ) from error

    def exec(
        self,
        container: ContainerRef,
        command: str | tuple[str, ...] | list[str],
        *,
        timeout_seconds: float | None = None,
        user: str | None = None,
        environment: dict[str, str] | None = None,
        output_path: Path | None = None,
        output_redact_values: tuple[str, ...] = (),
        output_callback: Callable[[str], None] | None = None,
        redact_output: bool = True,
    ) -> AgentRunResult:
        output_writer: _RedactedOutputWriter | _RawOutputWriter | None = None
        resolved_output: Path | None = None
        try:
            if redact_output:
                if output_path is not None:
                    resolved_output = Path(output_path).resolve(strict=False)
                    self._require_engine_owned(resolved_output.parent)
                if resolved_output is not None or output_callback is not None:
                    output_writer = _RedactedOutputWriter(
                        resolved_output,
                        output_redact_values,
                        output_callback,
                    )
            elif output_path is not None:
                if output_callback is not None:
                    raise SetupError(
                        "raw output capture cannot be streamed to a live callback"
                    )
                output_candidate = Path(output_path)
                directory_descriptor = self._open_engine_directory(
                    output_candidate.parent
                )
                output_writer = _RawOutputWriter(
                    output_candidate.name, directory_descriptor
                )
            elif output_callback is not None:
                raise SetupError(
                    "raw output capture cannot be streamed to a live callback"
                )
        except OSError as error:
            raise InfrastructureError(
                f"failed to open complete {self._role.title()} output artifact"
            ) from error
        try:
            create = self._client.api.exec_create(
                container.container_id,
                command,
                stdout=True,
                stderr=True,
                stdin=False,
                tty=False,
                privileged=False,
                user="" if user is None else user,
                environment=environment,
            )
        except APIError as error:
            if output_writer is not None:
                output_writer.abort()
            raise ContainerExecNotStartedError(
                f"failed to create Docker exec in {self._role.title()} container "
                f"{container.container_id} for command {command!r}: {error}"
            ) from error
        exec_id = create["Id"]
        output = _BoundedBytes(self._exec_output_limit_bytes)
        try:
            stream: Iterable[bytes] = self._client.api.exec_start(
                exec_id, stream=True, demux=False
            )
        except APIError as error:
            if output_writer is not None:
                output_writer.abort()
            raise ContainerExecNotStartedError(
                f"failed to start Docker exec {exec_id} in "
                f"{self._role.title()} container {container.container_id} "
                f"for command {command!r}: {error}"
            ) from error
        stream_errors: list[BaseException] = []
        stream_drained = threading.Event()

        def drain() -> None:
            try:
                for chunk in stream:
                    if chunk:
                        output.append(chunk)
                        if output_writer is not None:
                            output_writer.append(chunk)
            except BaseException as error:  # surfaced on the caller thread below
                stream_errors.append(error)
                if output_writer is not None:
                    try:
                        output_writer.abort()
                    except BaseException as abort_error:
                        stream_errors.append(abort_error)
            else:
                stream_drained.set()
            finally:
                if output_writer is not None and not stream_errors:
                    try:
                        output_writer.finish()
                    except BaseException as error:
                        stream_errors.append(error)

        reader = threading.Thread(target=drain, daemon=True, name=f"exec-{exec_id}")
        reader.start()
        reader.join(timeout_seconds)
        capture_stalled = reader.is_alive() and stream_drained.is_set()
        timed_out = reader.is_alive() and not capture_stalled

        def abort_output_drain() -> None:
            if output_writer is not None:
                try:
                    output_writer.abort()
                except BaseException:
                    pass
            _close_docker_stream(stream)
            reader.join(self._pause_timeout_seconds)

        if capture_stalled:
            reader.join(self._pause_timeout_seconds)
            if reader.is_alive():
                if output_writer is not None:
                    output_writer.abort()
                _close_docker_stream(stream)
                raise InfrastructureError(
                    "recovery_required: complete Judge output capture did not "
                    "finish within the bounded persistence interval"
                )
        elif timed_out:
            try:
                docker_container = self._container(container)
                docker_container.stop(timeout=1)
            except APIError as error:
                abort_output_drain()
                raise InfrastructureError(
                    f"failed to stop {self._role.title()} container "
                    f"{container.container_id} after exec {exec_id} timed out: {error}"
                ) from error
            deadline = time.monotonic() + self._pause_timeout_seconds
            while True:
                try:
                    docker_container.reload()
                except APIError as error:
                    abort_output_drain()
                    raise InfrastructureError(
                        f"failed to confirm termination of timed-out exec {exec_id} "
                        f"in {self._role.title()} container "
                        f"{container.container_id}: {error}"
                    ) from error
                if docker_container.attrs.get("State", {}).get("Running") is False:
                    break
                if time.monotonic() >= deadline:
                    abort_output_drain()
                    raise InfrastructureError(
                        "timed-out Docker exec container did not stop"
                    )
                time.sleep(self._poll_interval_seconds)
            # Stopping the container lets Docker deliver bytes already emitted by
            # the verifier. Close the HTTP stream only after a bounded EOF grace.
            reader.join(self._pause_timeout_seconds)
            if reader.is_alive():
                if isinstance(output_writer, _RawOutputWriter):
                    output_writer.abort()
                    _close_docker_stream(stream)
                    reader.join(self._pause_timeout_seconds)
                    raise InfrastructureError(
                        "recovery_required: complete Judge output did not reach "
                        "natural EOF after verifier timeout containment"
                    )
                _close_docker_stream(stream)
                reader.join(self._pause_timeout_seconds)
            if reader.is_alive():
                abort_output_drain()
                raise InfrastructureError(
                    "recovery_required: timed-out Docker exec output drainer "
                    "did not terminate"
                )
        _close_docker_stream(stream)
        if stream_errors:
            message = (
                "recovery_required: complete Judge output capture failed"
                if isinstance(output_writer, _RawOutputWriter)
                else "Docker exec output stream failed"
            )
            raise InfrastructureError(
                f"{message}: {stream_errors[0]}"
            ) from stream_errors[0]
        try:
            inspect = {} if timed_out else self._client.api.exec_inspect(exec_id)
        except APIError as error:
            raise InfrastructureError(
                f"failed to inspect Docker exec {exec_id} in "
                f"{self._role.title()} container {container.container_id} "
                f"for command {command!r}: {error}"
            ) from error
        return AgentRunResult(
            exit_code=None if timed_out else inspect.get("ExitCode"),
            output=output.value.decode("utf-8", errors="replace"),
            output_truncated=output.truncated,
            full_output_captured=(output_path is not None),
            timed_out=timed_out,
        )

    def copy_to(
        self,
        container: ContainerRef,
        source: Path,
        target: PurePosixPath,
    ) -> None:
        if self._staging_dir is None:
            raise SetupError("copy_to requires an Engine-owned staging directory")
        if not target.is_absolute():
            raise SetupError("container copy destination must be absolute")
        if ".." in target.parts:
            raise SetupError("container copy destination contains lexical traversal")
        source_path = Path(source).resolve()
        self._staging_dir.mkdir(parents=True, exist_ok=True)
        staged_name = f"{uuid.uuid4().hex}-{source_path.name}"
        staged = self._staging_dir / staged_name
        try:
            shutil.copyfile(source_path, staged)
            parent = self.exec(
                container,
                ("/usr/bin/install", "-d", "-m", "0755", str(target.parent)),
                user="root",
            )
            if parent.exit_code != 0:
                raise InfrastructureError(
                    f"create container copy parent failed: {parent.output}"
                )
            result = self.exec(
                container,
                (
                    "/usr/bin/install",
                    "-m",
                    "0644",
                    f"{_STAGING_TARGET}/{staged_name}",
                    str(target),
                ),
                user="root",
            )
            if result.exit_code != 0:
                raise InfrastructureError(
                    f"copy into container failed: {result.output}"
                )
        finally:
            staged.unlink(missing_ok=True)

    def inject_agent_auth(
        self, container: ContainerRef, material: AgentAuthMaterial
    ) -> None:
        """Inject validated Agent credentials into exact Work-only tmpfs mounts."""

        if self._role != "work" or container.role != "work":
            raise SetupError("local Agent authentication is limited to Work")
        docker_container = self._container(container)
        try:
            docker_container.reload()
            attrs = docker_container.attrs
            host_tmpfs = (attrs.get("HostConfig") or {}).get("Tmpfs") or {}
            mounts = attrs.get("Mounts") or ()
        except (APIError, AttributeError, TypeError) as error:
            raise InfrastructureError(
                "failed to inspect Work Agent authentication tmpfs"
            ) from error
        for auth_mount in material.mounts:
            expected_target = str(auth_mount.tmpfs.target)
            if host_tmpfs.get(expected_target) != auth_mount.tmpfs.options:
                raise InfrastructureError(
                    "Work Agent authentication tmpfs is not installed exactly"
                )
            at_target = [
                mount
                for mount in mounts
                if isinstance(mount, Mapping)
                and mount.get("Destination") == expected_target
            ]
            reported_mount_is_exact = (
                len(at_target) == 1
                and at_target[0].get("Type") == "tmpfs"
                and at_target[0].get("RW") is True
            )
            active_mount_is_exact = reported_mount_is_exact
            if not at_target:
                mountinfo = self.exec(
                    container,
                    ("/bin/cat", "/proc/self/mountinfo"),
                    timeout_seconds=10.0,
                    user="root",
                )
                matches: list[tuple[set[str], str, set[str]]] = []
                if mountinfo.exit_code == 0:
                    for line in mountinfo.output.splitlines():
                        fields = line.split()
                        if "-" not in fields or len(fields) < 10:
                            continue
                        separator = fields.index("-")
                        if separator + 3 >= len(fields):
                            continue
                        if fields[4] == expected_target:
                            matches.append(
                                (
                                    set(fields[5].split(",")),
                                    fields[separator + 1],
                                    set(fields[separator + 3].split(",")),
                                )
                            )
                active_mount_is_exact = (
                    len(matches) == 1
                    and matches[0][1] == "tmpfs"
                    and {"rw", "nosuid", "nodev", "noexec"}
                    <= matches[0][0]
                    and "rw" in matches[0][2]
                    and matches[0][2] & {"mode=700", "mode=0700"}
                )
            if not active_mount_is_exact:
                raise InfrastructureError(
                    "Work Agent authentication tmpfs attestation failed"
                )

            hardened = self.exec(
                container,
                ("/bin/chmod", "0700", "--", expected_target),
                timeout_seconds=10.0,
                user="root",
            )
            if hardened.exit_code != 0:
                raise InfrastructureError(
                    "failed to secure Work Agent authentication tmpfs"
                )
            for auth_file in auth_mount.files:
                self._stream_agent_auth_file(
                    container,
                    target=auth_mount.tmpfs.target / auth_file.path,
                    content=auth_file.content,
                    mode=auth_file.mode,
                )
            attested = self.exec(
                container,
                (
                    "/bin/sh",
                    "-c",
                    'mode="$(/usr/bin/stat -c %a:%u:%g -- "$1")" && '
                    '[ "$mode" = "700:0:0" ]',
                    "rsi-agent-auth-directory",
                    expected_target,
                ),
                timeout_seconds=10.0,
                user="root",
            )
            if attested.exit_code != 0:
                raise InfrastructureError(
                    "Work Agent authentication tmpfs permissions differ"
                )

    def _stream_agent_auth_file(
        self,
        container: ContainerRef,
        *,
        target: PurePosixPath,
        content: bytes,
        mode: int,
    ) -> None:
        """Stream one secret from memory directly into an active tmpfs."""

        command = (
            "/bin/sh",
            "-c",
            'umask 077; /bin/cat > "$1" && /bin/chmod "$2" "$1"',
            "rsi-agent-auth",
            str(target),
            f"{mode:o}",
        )
        stream: Any | None = None
        response: Any | None = None
        try:
            created = self._client.api.exec_create(
                container.container_id,
                command,
                stdout=True,
                stderr=True,
                stdin=True,
                tty=False,
                privileged=False,
                user="root",
            )
            exec_id = created["Id"]
            stream = self._client.api.exec_start(
                exec_id,
                detach=False,
                tty=False,
                socket=True,
            )
            raw_socket = getattr(stream, "_sock", None)
            response = getattr(stream, "_response", None)
            if raw_socket is None or response is None:
                raise TypeError("Docker did not return an owned exec socket")
            raw_socket.settimeout(10.0)
            raw_socket.sendall(content)
            raw_socket.shutdown(socket.SHUT_WR)
            while stream.read(8192):
                pass
            inspection = self._client.api.exec_inspect(exec_id)
            if inspection.get("Running") is True or inspection.get("ExitCode") != 0:
                raise InfrastructureError(
                    "local Agent authentication write failed in Work"
                )
        except InfrastructureError:
            raise
        except (APIError, KeyError, OSError, TypeError) as error:
            raise InfrastructureError(
                "failed to stream local Agent authentication into Work"
            ) from error
        finally:
            if response is not None:
                try:
                    response.close()
                except Exception:
                    pass
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    def inject_directory(
        self,
        container: ContainerRef,
        source: Path,
        target: PurePosixPath,
    ) -> None:
        """Upload only the exact task tests tree through a bounded tar stream."""
        source_path = Path(source)
        if ".." in source_path.parts:
            raise SetupError("test injection source contains lexical traversal")
        if not target.is_absolute():
            raise SetupError("test injection target must be absolute")
        if ".." in target.parts:
            raise SetupError("test injection target contains lexical traversal")
        expected = self._task_source_dir / "tests"
        if (
            self._role != "judge"
            or source_path != expected
            or target != PurePosixPath("/tests")
        ):
            raise SetupError(
                "directory injection is limited to the exact task tests tree"
            )
        plan = self._collect_test_archive_entries(source_path)
        try:
            self._inject_test_archive(container, target, plan)
        finally:
            os.close(plan.root_descriptor)

    def _inject_test_archive(
        self,
        container: ContainerRef,
        target: PurePosixPath,
        plan: _TestArchivePlan,
    ) -> None:
        staging = PurePosixPath(
            f"/tmp/.rsi-harness-tests-{uuid.uuid4().hex}"
        )
        prepared = self.exec(
            container,
            ("/usr/bin/install", "-d", "-m", "0700", str(staging)),
            user="root",
        )
        if prepared.exit_code != 0:
            raise InfrastructureError(
                f"failed to prepare private tests staging: {prepared.output}"
            )
        reader_descriptor, writer_descriptor = os.pipe()
        reader = _ChunkedReadStream(
            os.fdopen(reader_descriptor, "rb", buffering=0)
        )
        writer = os.fdopen(writer_descriptor, "wb", buffering=0)
        producer_errors: list[BaseException] = []

        def produce() -> None:
            try:
                with writer:
                    with tarfile.open(
                        fileobj=writer,
                        mode="w|",
                        format=tarfile.PAX_FORMAT,
                    ) as archive:
                        self._require_unchanged_test_entry(
                            os.fstat(plan.root_descriptor),
                            plan.root_metadata,
                            "task tests root changed before upload",
                        )
                        metadata_by_name = {
                            entry.name: entry.metadata for entry in plan.entries
                        }
                        for entry in plan.entries:
                            self._add_test_archive_entry(
                                archive,
                                entry,
                                root_descriptor=plan.root_descriptor,
                                metadata_by_name=metadata_by_name,
                            )
                        self._require_unchanged_test_entry(
                            os.fstat(plan.root_descriptor),
                            plan.root_metadata,
                            "task tests root changed during upload",
                        )
            except BaseException as error:
                producer_errors.append(error)

        producer = threading.Thread(
            target=produce,
            daemon=True,
            name=f"tests-archive-{container.container_id}",
        )
        producer.start()
        upload_error: BaseException | None = None
        uploaded = False
        try:
            uploaded = bool(
                self._container(container).put_archive(str(staging), reader)
            )
        except APIError as error:
            upload_error = InfrastructureError(
                f"failed to inject tests into {self._role.title()} container "
                f"{container.container_id}: {error}"
            )
        finally:
            reader.close()
            producer.join()
        if upload_error is not None:
            raise upload_error
        if producer_errors:
            error = producer_errors[0]
            if isinstance(error, (InfrastructureError, SetupError)):
                raise error
            raise InfrastructureError(
                f"failed to stream bounded task tests archive: {error}"
            ) from error
        if not uploaded:
            raise InfrastructureError(
                f"Docker did not accept tests for {self._role.title()} container "
                f"{container.container_id}"
            )
        copied = self.exec(
            container,
            ("/bin/cp", "-a", f"{staging}/.", str(target)),
            user="root",
        )
        if copied.exit_code != 0:
            raise InfrastructureError(
                f"failed to populate private tests tmpfs: {copied.output}"
            )
        removed = self.exec(
            container,
            ("/bin/rm", "-rf", "--", str(staging)),
            user="root",
        )
        if removed.exit_code != 0:
            raise InfrastructureError(
                f"failed to remove private tests staging: {removed.output}"
            )

    def _collect_test_archive_entries(self, source: Path) -> _TestArchivePlan:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            root_descriptor = os.open(source, directory_flags)
            root_metadata = os.fstat(root_descriptor)
        except OSError as error:
            raise SetupError(f"cannot inspect task tests directory: {error}") from error
        if not stat.S_ISDIR(root_metadata.st_mode):
            os.close(root_descriptor)
            raise SetupError("exact task tests source must be a real directory")
        entries: list[_ArchiveEntry] = []
        regular_bytes = 0

        def visit(directory_descriptor: int, prefix: PurePosixPath) -> None:
            nonlocal regular_bytes
            try:
                with os.scandir(directory_descriptor) as iterator:
                    children = sorted(iterator, key=lambda child: child.name)
            except OSError as error:
                raise SetupError(
                    f"cannot enumerate task tests directory: {error}"
                ) from error
            for child in children:
                relative = prefix / child.name
                try:
                    metadata = child.stat(follow_symlinks=False)
                except OSError as error:
                    raise SetupError(
                        "cannot inspect task tests entry within pinned source "
                        f"authority: {relative}"
                    ) from error
                if relative.is_absolute() or ".." in relative.parts:
                    raise SetupError(
                        "task tests archive entry contains lexical traversal"
                    )
                linkname = None
                if stat.S_ISLNK(metadata.st_mode):
                    linkname = self._validated_test_symlink(
                        child.name, directory_descriptor
                    )
                entries.append(
                    _ArchiveEntry(
                        name=relative.as_posix(),
                        metadata=metadata,
                        linkname=linkname,
                    )
                )
                if len(entries) > _MAX_INJECTED_TEST_ENTRIES:
                    raise SetupError("task tests archive exceeds entry limit")
                if stat.S_ISREG(metadata.st_mode):
                    regular_bytes += metadata.st_size
                    if regular_bytes > _MAX_INJECTED_TEST_REGULAR_BYTES:
                        raise SetupError(
                            "task tests archive exceeds regular byte limit"
                        )
                elif stat.S_ISDIR(metadata.st_mode):
                    try:
                        child_descriptor = os.open(
                            child.name,
                            directory_flags,
                            dir_fd=directory_descriptor,
                        )
                    except OSError as error:
                        raise SetupError(
                            "task tests directory changed during traversal: "
                            f"{relative}: {error}"
                        ) from error
                    try:
                        self._require_unchanged_test_entry(
                            os.fstat(child_descriptor),
                            metadata,
                            "task tests directory changed during traversal: "
                            f"{relative}",
                        )
                        visit(child_descriptor, relative)
                    finally:
                        os.close(child_descriptor)
                elif not stat.S_ISLNK(metadata.st_mode):
                    raise SetupError(
                        f"unsupported task tests filesystem entry: {relative}"
                    )

        try:
            visit(root_descriptor, PurePosixPath())
            self._require_unchanged_test_entry(
                os.fstat(root_descriptor),
                root_metadata,
                "task tests root changed during traversal",
            )
        except BaseException:
            os.close(root_descriptor)
            raise
        return _TestArchivePlan(
            root_descriptor=root_descriptor,
            root_metadata=root_metadata,
            entries=tuple(entries),
        )

    @staticmethod
    def _validated_test_symlink(name: str, directory_descriptor: int) -> str:
        try:
            linkname = os.readlink(name, dir_fd=directory_descriptor)
        except OSError as error:
            raise SetupError(f"cannot read task tests symlink: {error}") from error
        linkpath = PurePosixPath(linkname)
        if linkpath.is_absolute() or ".." in linkpath.parts:
            raise SetupError("task tests symlink contains lexical traversal")
        return linkname

    @staticmethod
    def _add_test_archive_entry(
        archive: tarfile.TarFile,
        entry: _ArchiveEntry,
        *,
        root_descriptor: int,
        metadata_by_name: dict[str, os.stat_result],
    ) -> None:
        metadata = entry.metadata
        info = tarfile.TarInfo(entry.name)
        info.mode = stat.S_IMODE(metadata.st_mode)
        info.mtime = int(metadata.st_mtime)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        parent_descriptor = DockerContainerRuntime._open_test_archive_parent(
            root_descriptor,
            entry.name,
            metadata_by_name,
        )
        leaf = PurePosixPath(entry.name).name
        try:
            if stat.S_ISDIR(metadata.st_mode):
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                try:
                    descriptor = os.open(leaf, flags, dir_fd=parent_descriptor)
                except OSError as error:
                    raise SetupError(
                        f"task tests directory changed before upload: {entry.name}: "
                        f"{error}"
                    ) from error
                try:
                    DockerContainerRuntime._require_unchanged_test_entry(
                        os.fstat(descriptor),
                        metadata,
                        f"task tests directory changed before upload: {entry.name}",
                    )
                finally:
                    os.close(descriptor)
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
                return
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    before = os.stat(
                        leaf, dir_fd=parent_descriptor, follow_symlinks=False
                    )
                    linkname = os.readlink(leaf, dir_fd=parent_descriptor)
                    after = os.stat(
                        leaf, dir_fd=parent_descriptor, follow_symlinks=False
                    )
                except OSError as error:
                    raise SetupError(
                        f"task tests symlink changed before upload: {entry.name}: "
                        f"{error}"
                    ) from error
                DockerContainerRuntime._require_unchanged_test_entry(
                    before,
                    metadata,
                    f"task tests symlink changed before upload: {entry.name}",
                )
                DockerContainerRuntime._require_unchanged_test_entry(
                    after,
                    metadata,
                    f"task tests symlink changed during upload: {entry.name}",
                )
                if linkname != entry.linkname:
                    raise SetupError(
                        f"task tests symlink changed during upload: {entry.name}"
                    )
                info.type = tarfile.SYMTYPE
                info.linkname = linkname
                archive.addfile(info)
                return
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(leaf, flags, dir_fd=parent_descriptor)
            except OSError as error:
                raise SetupError(
                    f"cannot open task tests regular file {entry.name}: {error}"
                ) from error
            with os.fdopen(descriptor, "rb") as stream:
                before = os.fstat(stream.fileno())
                DockerContainerRuntime._require_unchanged_test_entry(
                    before,
                    metadata,
                    f"task tests regular file changed before upload: {entry.name}",
                )
                if not stat.S_ISREG(before.st_mode):
                    raise SetupError(
                        f"task tests regular file changed before upload: {entry.name}"
                    )
                info.size = before.st_size
                archive.addfile(info, stream)
                DockerContainerRuntime._require_unchanged_test_entry(
                    os.fstat(stream.fileno()),
                    metadata,
                    f"task tests regular file changed during upload: {entry.name}",
                )
        finally:
            os.close(parent_descriptor)

    @staticmethod
    def _open_test_archive_parent(
        root_descriptor: int,
        name: str,
        metadata_by_name: dict[str, os.stat_result],
    ) -> int:
        descriptor = os.dup(root_descriptor)
        prefix = PurePosixPath()
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            for part in PurePosixPath(name).parts[:-1]:
                prefix /= part
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                DockerContainerRuntime._require_unchanged_test_entry(
                    os.fstat(descriptor),
                    metadata_by_name[prefix.as_posix()],
                    f"task tests directory changed before upload: {prefix}",
                )
        except (OSError, KeyError) as error:
            os.close(descriptor)
            raise SetupError(
                f"cannot open pinned task tests parent for {name}: {error}"
            ) from error
        return descriptor

    @staticmethod
    def _require_unchanged_test_entry(
        actual: os.stat_result,
        expected: os.stat_result,
        message: str,
    ) -> None:
        def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
            )

        if identity(actual) != identity(expected):
            raise SetupError(message)

    def create_network(self, suffix: str, *, internal: bool) -> ManagedNetwork:
        name = self.planned_network_name(suffix)
        bridge_interface = managed_bridge_interface(name)
        try:
            network = self._client.networks.create(
                name,
                driver="bridge",
                internal=internal,
                check_duplicate=True,
                labels=self.recovery_labels,
                options={"com.docker.network.bridge.name": bridge_interface},
            )
        except APIError as error:
            discovered = self._discover_exact_networks(name, internal=internal)
            if discovered:
                network = discovered[0]
                try:
                    self.remove_network(network)
                except InfrastructureError as rollback_error:
                    raise InfrastructureError(
                        "recovery_required: failed Judge network creation may "
                        "have mutated and exact cleanup is unproven; "
                        f"planned_network={name}: {rollback_error}"
                    ) from error
                raise SetupError(
                    f"failed to create {self._role} network {name!r}: {error}; "
                    "the discovered mutation was removed and proven absent"
                ) from error
            raise SetupError(
                f"failed to create {self._role} network {name!r}: {error}; "
                "the planned network is proven absent"
            ) from error
        return ManagedNetwork(
            network_id=network.id,
            name=name,
            run_id=self._run_id,
            task_id=self._task_id,
            role=self._role,
            internal=internal,
        )

    def planned_network_name(self, suffix: str) -> str:
        safe_suffix = self._safe_network_component(suffix)
        if not safe_suffix:
            raise SetupError("network suffix must contain a safe identifier")
        return f"{self._network_name_prefix()}{safe_suffix}"

    def planned_container_name(self, name_or_suffix: str) -> str:
        safe = self._safe_network_component(name_or_suffix)
        if not safe:
            raise SetupError("container identity must contain a safe identifier")
        prefix = self._network_name_prefix()
        return safe if safe.startswith(prefix) else f"{prefix}{safe}"

    def remove_network(self, network_id: str | ManagedNetwork) -> None:
        resolved_id = (
            network_id.network_id
            if isinstance(network_id, ManagedNetwork)
            else network_id
        )
        try:
            network = self._client.networks.get(resolved_id)
        except NotFound:
            network = None
        except APIError as error:
            raise InfrastructureError(
                "recovery_required: cannot inspect exact network before removal "
                f"{resolved_id}: {error}"
            ) from error
        if network is not None:
            if isinstance(network_id, ManagedNetwork):
                self._attest_network(network, network_id)
            try:
                network.remove()
            except (APIError, OSError) as error:
                try:
                    self._client.networks.get(resolved_id)
                except NotFound:
                    pass
                except APIError as inspection_error:
                    raise InfrastructureError(
                        "recovery_required: exact network removal and follow-up "
                        f"inspection are ambiguous for {resolved_id}: "
                        f"{inspection_error}"
                    ) from error
                else:
                    raise InfrastructureError(
                        "recovery_required: exact network remains after failed "
                        f"removal {resolved_id}: {error}"
                    ) from error
        try:
            self._client.networks.get(resolved_id)
        except NotFound:
            pass
        except APIError as error:
            raise InfrastructureError(
                "recovery_required: exact network absence is unproven after "
                f"removal {resolved_id}: {error}"
            ) from error
        else:
            raise InfrastructureError(
                "recovery_required: exact network still exists after removal "
                f"{resolved_id}"
            )
        if isinstance(network_id, ManagedNetwork):
            discovered = self._discover_exact_networks(
                network_id.name, internal=network_id.internal
            )
            if discovered:
                raise InfrastructureError(
                    "recovery_required: planned Judge network remains after "
                    f"removal; planned_network={network_id.name}"
                )

    def _discover_exact_networks(
        self, name: str, *, internal: bool
    ) -> list[ManagedNetwork]:
        label_filters = [
            f"{key}={value}" for key, value in sorted(self.recovery_labels.items())
        ]
        try:
            candidates = self._client.networks.list(
                filters={"name": name, "label": label_filters}
            )
        except (APIError, AttributeError, TypeError) as error:
            raise InfrastructureError(
                "recovery_required: exact network discovery is unproven; "
                f"planned_network={name}: {error}"
            ) from error
        if not isinstance(candidates, list):
            raise InfrastructureError(
                "recovery_required: exact network discovery returned malformed "
                f"results; planned_network={name}"
            )
        exact: list[ManagedNetwork] = []
        for candidate in candidates:
            try:
                candidate.reload()
                attrs = candidate.attrs
                candidate_id = candidate.id
            except NotFound:
                continue
            except (APIError, AttributeError, KeyError, TypeError) as error:
                raise InfrastructureError(
                    "recovery_required: exact network discovery is unproven; "
                    f"planned_network={name}: {error}"
                ) from error
            if attrs.get("Name") != name:
                continue
            discovered = ManagedNetwork(
                network_id=candidate_id,
                name=name,
                run_id=self._run_id,
                task_id=self._task_id,
                role=self._role,
                internal=internal,
            )
            self._attest_network(candidate, discovered)
            exact.append(discovered)
        if len(exact) > 1:
            raise InfrastructureError(
                "recovery_required: multiple exact networks have planned "
                f"authority; planned_network={name}"
            )
        return exact

    def _attest_network(self, network: Any, expected: ManagedNetwork) -> None:
        attrs = network.attrs
        labels = attrs.get("Labels") or {}
        options = attrs.get("Options") or {}
        if (
            network.id != expected.network_id
            or attrs.get("Name") != expected.name
            or attrs.get("Driver") != "bridge"
            or attrs.get("Internal") is not expected.internal
            or labels != self.recovery_labels
            or options.get("com.docker.network.bridge.name")
            != managed_bridge_interface(expected.name)
        ):
            raise InfrastructureError(
                "recovery_required: discovered network does not attest exact "
                f"planned authority; planned_network={expected.name}"
            )

    def _container(self, container: ContainerRef) -> Any:
        return self._client.containers.get(container.container_id)

    def _mount(self, mount: ContainerMount, *, workdir: PurePosixPath | None) -> Mount:
        if ".." in mount.target.parts:
            raise SetupError("container mount target contains lexical traversal")
        if workdir is not None and mount.target == workdir:
            raise SetupError(
                "declared WORKDIR may only be accessed through a managed volume"
            )
        source = mount.source.resolve(strict=True)
        if not source.exists():
            raise SetupError(f"container mount source does not exist: {source}")
        if mount.target in _DOCKER_SOCKET_TARGETS or source.name == "docker.sock":
            raise SetupError("Docker socket mounts are forbidden")
        task = self._task_source_dir
        task_tests = task / "tests"
        exact_judge_tests = (
            self._role == "judge"
            and mount.target == PurePosixPath("/tests")
            and mount.read_only
            and task_tests.is_dir()
            and not task_tests.is_symlink()
            and source == task_tests.resolve(strict=True)
        )
        exact_work_feedback = (
            self._role == "work"
            and self._work_feedback_dir is not None
            and mount.target == WORK_FEEDBACK_ROOT
            and mount.read_only
            and source == self._work_feedback_dir
        )
        if mount.target == WORK_FEEDBACK_ROOT and not exact_work_feedback:
            raise SetupError(
                "Work feedback mount differs from exact read-only authority"
            )
        if (
            source == task or task in source.parents or source in task.parents
        ) and not exact_judge_tests:
            raise SetupError("task source or its ancestor cannot be mounted")
        if not exact_judge_tests:
            self._require_engine_owned(source)
        allowed_targets = {_STAGING_TARGET}
        if workdir is not None:
            allowed_targets.add(workdir)
        if self._role == "judge":
            allowed_targets.update(
                {PurePosixPath("/tests"), PurePosixPath("/logs/verifier")}
            )
        if exact_work_feedback:
            allowed_targets.add(WORK_FEEDBACK_ROOT)
        if mount.target not in allowed_targets:
            raise SetupError(
                f"mount target {mount.target} is not allowed for role {self._role}"
            )
        return Mount(
            source=str(source),
            target=str(mount.target),
            type="bind",
            read_only=mount.read_only,
        )

    def _volume_mount(
        self, mount: ContainerVolumeMount, *, workdir: PurePosixPath | None
    ) -> Mount:
        if self._role not in {"work", "judge"}:
            raise SetupError(
                "managed WORKDIR volumes are only valid for Work and Judge"
            )
        expected_read_only = self._role == "judge"
        if mount.read_only is not expected_read_only:
            raise SetupError("managed WORKDIR volume access differs from runtime role")
        if (
            workdir is None
            or mount.target != workdir
            or mount.target != mount.volume.target
        ):
            raise SetupError(
                "managed WORKDIR volume target differs from WORKDIR authority"
            )
        if self._role == "work" and self._workdir_volume_references:
            raise SetupError(
                "Work creation requires zero pre-existing WORKDIR volume references"
            )
        if self._role == "judge" and (
            len(self._workdir_volume_references) != 1
            or self._workdir_volume_references[0].role != "work"
        ):
            raise SetupError(
                "Judge creation requires the exact attested Work volume reference"
            )
        if mount.volume.name == "docker.sock" or mount.volume.name.endswith(".sock"):
            raise SetupError(
                "Docker socket-like managed WORKDIR volume names are forbidden"
            )
        expected_name = self._managed_workdir_volume_name()
        if mount.volume.name != expected_name:
            raise SetupError("managed WORKDIR volume authority is not canonical")
        try:
            self._client.volumes.get(mount.volume.name)
        except NotFound as error:
            raise SetupError(
                "cannot attest managed WORKDIR volume authority before container "
                f"create: {error}"
            ) from error
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume attestation is "
                f"ambiguous before container create: {error}"
            ) from error
        if (
            mount.volume.run_id != self._run_id
            or mount.volume.task_id != self._task_id
        ):
            raise SetupError("managed WORKDIR volume does not attest exact authority")
        DockerWorkdirVolumeBackend(self._client).inspect(
            mount.volume,
            expected_references=self._workdir_volume_references,
        )
        return Mount(
            source=mount.volume.name,
            target=str(mount.target),
            type="volume",
            read_only=mount.read_only,
        )

    def _managed_workdir_volume_name(self) -> str:
        return DockerWorkdirVolumeBackend(self._client).planned_name(
            run_id=self._run_id, task_id=self._task_id
        )

    def _require_engine_owned(self, source: Path) -> None:
        resolved = source.resolve(strict=source.exists())
        if not any(
            resolved == root or root in resolved.parents
            for root in self._allowed_mount_roots
        ):
            raise SetupError(
                f"mount source is outside explicit Engine-owned roots: {resolved}"
            )

    def _open_engine_directory(self, directory: Path) -> int:
        if not directory.is_absolute():
            raise SetupError("output artifact directory must be absolute")
        selected: tuple[Path, Path] | None = None
        for root in self._allowed_mount_roots:
            try:
                relative = directory.relative_to(root)
            except ValueError:
                continue
            selected = root, relative
            break
        if selected is None:
            raise SetupError(
                "output artifact directory is outside explicit Engine-owned roots"
            )
        root, relative = selected
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptors: list[int] = []
        try:
            descriptor = os.open(root, flags)
            descriptors.append(descriptor)
            for component in relative.parts:
                if component in {"", ".", ".."}:
                    raise SetupError(
                        "output artifact directory contains unsafe components"
                    )
                descriptor = os.open(component, flags, dir_fd=descriptor)
                descriptors.append(descriptor)
            result = os.dup(descriptors[-1])
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
        return result

    def _validate_managed_network(self, network: ManagedNetwork) -> None:
        expected_prefix = self._network_name_prefix()
        suffix = network.name.removeprefix(expected_prefix)
        if (
            network.run_id != self._run_id
            or network.task_id != self._task_id
            or network.role != self._role
            or not network.name.startswith(expected_prefix)
            or not suffix
            or self._safe_network_component(suffix) != suffix
            or network.name in {"bridge", "host", "none"}
            or network.name.startswith("container:")
        ):
            raise SetupError("Engine-created network identity does not match runtime")
        try:
            attrs = self._client.networks.get(network.network_id).attrs
        except (APIError, KeyError) as error:
            raise SetupError(
                f"cannot validate Engine-created network {network.network_id}: {error}"
            ) from error
        labels = attrs.get("Labels") or {}
        options = attrs.get("Options") or {}
        if (
            attrs.get("Name") != network.name
            or attrs.get("Driver") != "bridge"
            or attrs.get("Internal") is not network.internal
            or labels != self.recovery_labels
            or options.get("com.docker.network.bridge.name")
            != managed_bridge_interface(network.name)
        ):
            raise SetupError("network is unmanaged or has mismatched role labels")

    def _network_name_prefix(self) -> str:
        return (
            f"rsi-{self._safe_network_component(self._run_id)}-"
            f"{self._safe_network_component(self._task_id)}-{self._role}-"
        )

    @staticmethod
    def _safe_network_component(value: str) -> str:
        return "".join(
            character if character.isalnum() or character in "_.-" else "-"
            for character in value
        ).strip(".-")


__all__ = ["DockerContainerRuntime", "DockerOutput"]
