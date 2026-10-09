"""Narrow, inspected CPU child lifecycle. Never exposes Docker authority."""

from __future__ import annotations

import io
import json
import math
import select
import socket
import struct
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from docker.errors import NotFound
from docker.types import LogConfig, Ulimit

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations import sandbox_client as wire
from rsi_harness.integrations.sandbox_client import MAX_COMMAND_BYTES, MAX_OUTPUT_BYTES
from rsi_harness.runtime.sandbox_contracts import (
    IMAGE_ID,
    SandboxChildStopped,
    SandboxDownloadError,
    SandboxDownloadStopped,
    SandboxError,
    SandboxLease,
    SandboxProfile,
    SandboxResult,
    absolute_path,
    below,
)
from rsi_harness.runtime.sandbox_env_contracts import sandbox_object_labels
from rsi_harness.runtime.sandbox_env_docker import PausedKiller, terminate_container
from rsi_harness.runtime.sandbox_transfer import decode_bundle, encode_bundle

MIB = 1024**2


def sandbox_labels(lease: SandboxLease) -> dict[str, str]:
    # v1 children share the one label helper with env and builder objects.
    return sandbox_object_labels(lease.owner, "sandbox", {"sandbox-id": lease.child_id})


def attest_sandbox_identity(lease: SandboxLease, attrs: dict[str, Any]) -> None:
    identity = attrs.get("Id", "")
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or (lease.container_id is not None and identity != lease.container_id)
        or attrs.get("Name") != "/" + lease.planned_name
        or attrs.get("Image") != lease.image_id
        or (attrs.get("Config") or {}).get("Labels") != sandbox_labels(lease)
    ):
        raise InfrastructureError("sandbox exact container identity/ownership mismatch")


class SandboxDockerBackend:
    """Use a dedicated bounded-timeout Docker client in production composition.

    ``paused_killer`` (opt-in; production passes
    ``sandbox_env_docker.default_paused_killer(client.api)``) lets
    ``terminate`` kill a paused child: cgroup.kill as root, which proves zero
    execution after the freeze. Without it a paused child is refused.
    """

    def __init__(
        self, client: Any, *, paused_killer: PausedKiller | None = None
    ) -> None:
        self.client = client
        self._paused_killer = paused_killer
        self._profiles: dict[str, SandboxProfile] = {}
        self._environments: dict[str, dict[str, str]] = {}

    def preflight(self, profile: SandboxProfile) -> str:
        # Revalidate even callers using unchecked model_copy updates.
        profile = SandboxProfile.model_validate(profile.model_dump())
        info = self.client.info()
        for field in ("MemoryLimit", "SwapLimit", "PidsLimit", "CpuCfsQuota"):
            if info.get(field) is not True:
                raise SetupError(f"sandbox host cannot enforce {field}")
        security = info.get("SecurityOptions") or []
        if (
            info.get("OSType") != "linux"
            or str(info.get("CgroupVersion")) != "2"
            or "runc" not in (info.get("Runtimes") or {})
            or "name=apparmor" not in security
            or not any(option.startswith("name=seccomp,") for option in security)
        ):
            raise SetupError(
                "sandbox requires Linux cgroup v2, runc, AppArmor and seccomp"
            )
        try:
            image = self.client.images.get(profile.image)
        except NotFound as error:
            raise SetupError(
                f"sandbox profile {profile.name}: approved image is not cached"
            ) from error
        if not IMAGE_ID.fullmatch(image.id):
            raise SetupError("sandbox image did not resolve to an immutable ID")
        if IMAGE_ID.fullmatch(profile.image) and profile.image != image.id:
            raise SetupError("sandbox resolved image differs from approved image ID")
        config = image.attrs.get("Config") or {}
        if config.get("Volumes"):
            raise SetupError(
                "sandbox image declares anonymous volumes; rebuild without VOLUME"
            )
        environment = config.get("Env") or []
        if (
            len(environment) > 256
            or sum(len(value.encode()) for value in environment) > 65536
        ):
            raise SetupError(
                "sandbox image environment exceeds bounded metadata limits"
            )
        return image.id

    @staticmethod
    def _tmpfs(profile: SandboxProfile) -> dict[str, str]:
        return {
            path: f"rw,exec,nosuid,nodev,size={size * MIB},mode=1777"
            for path, size in profile.tmpfs_mb
            if path != "/dev/shm"
        }

    def create(self, lease: SandboxLease, profile: SandboxProfile) -> str:
        if (
            lease.cpus != profile.cpus
            or lease.memory_mb != profile.memory_mb
            or lease.reserved_lifetime_sec > profile.max_lifetime_sec
        ):
            raise InfrastructureError("sandbox profile exceeds durable reservation")
        image_id = self.preflight(profile)
        if image_id != lease.image_id:
            raise SetupError("sandbox lease image differs from approved image")
        image = self.client.images.get(image_id)
        environment = {
            item.split("=", 1)[0]: ""
            for item in (image.attrs.get("Config") or {}).get("Env", [])
        }
        environment.update(
            {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "TMPDIR": "/tmp",
                "LANG": "C.UTF-8",
                "NVIDIA_VISIBLE_DEVICES": "void",
            }
        )
        child = self.client.containers.create(
            image=image_id,
            name=lease.planned_name,
            labels=sandbox_labels(lease),
            command=["infinity"],
            entrypoint=["/bin/sleep"],
            user="0:0",
            working_dir=profile.workdir,
            environment=environment,
            network_mode="none",
            read_only=True,
            privileged=False,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true", "apparmor=docker-default"],
            runtime="runc",
            ipc_mode="private",
            cgroupns="private",
            nano_cpus=profile.cpus * 1_000_000_000,
            mem_limit=profile.memory_mb * MIB,
            memswap_limit=profile.memory_mb * MIB,
            pids_limit=profile.pids,
            ulimits=[Ulimit(name="nofile", soft=1024, hard=1024)],
            tmpfs=self._tmpfs(profile),
            shm_size=dict(profile.tmpfs_mb)["/dev/shm"] * MIB,
            log_config=LogConfig(type="none"),
            restart_policy={"Name": "no"},
            healthcheck={"test": ["NONE"]},
            detach=True,
        )
        self._profiles[child.id] = profile
        self._environments[child.id] = environment
        # No start here: actual identity must be durably journaled first.
        return child.id

    def _owned(self, lease: SandboxLease, *, missing_ok: bool = False) -> Any:
        if lease.container_id is None:
            raise InfrastructureError(
                "sandbox actual container identity is not recorded"
            )
        try:
            child = self.client.containers.get(lease.container_id)
            child.reload()
        except NotFound:
            if missing_ok:
                return None
            raise InfrastructureError("sandbox owned container is absent") from None
        attest_sandbox_identity(lease, child.attrs)
        return child

    def inspect(self, lease: SandboxLease) -> dict[str, Any]:
        return self._owned(lease).attrs

    def _attrs_or_none(self, lease: SandboxLease) -> dict[str, Any] | None:
        child = self._owned(lease, missing_ok=True)
        return None if child is None else child.attrs

    def _attest_configuration(self, lease: SandboxLease, child: Any) -> None:
        profile = self._profiles.get(child.id)
        if profile is None:
            raise InfrastructureError("sandbox start lacks approved profile authority")
        attrs = child.attrs
        host = attrs.get("HostConfig") or {}
        expected = {
            "ReadonlyRootfs": True,
            "NetworkMode": "none",
            "Privileged": False,
            "CapDrop": ["ALL"],
            "NanoCpus": profile.cpus * 1_000_000_000,
            "Memory": profile.memory_mb * MIB,
            "MemorySwap": profile.memory_mb * MIB,
            "PidsLimit": profile.pids,
            "Tmpfs": self._tmpfs(profile),
            "ShmSize": dict(profile.tmpfs_mb)["/dev/shm"] * MIB,
            "Runtime": "runc",
            "IpcMode": "private",
            "CgroupnsMode": "private",
            "SecurityOpt": ["no-new-privileges:true", "apparmor=docker-default"],
        }
        for key, value in expected.items():
            if host.get(key) != value:
                raise InfrastructureError(
                    f"sandbox Docker configuration mismatch: {key}"
                )
        for key in (
            "CapAdd",
            "Binds",
            "Devices",
            "DeviceRequests",
            "PortBindings",
            "PidMode",
            "UTSMode",
            "VolumesFrom",
        ):
            if host.get(key):
                raise InfrastructureError(f"sandbox forbidden Docker authority: {key}")
        if (
            (host.get("LogConfig") or {}).get("Type") != "none"
            or (host.get("RestartPolicy") or {}).get("Name") != "no"
            or host.get("Ulimits") != [{"Name": "nofile", "Soft": 1024, "Hard": 1024}]
        ):
            raise InfrastructureError("sandbox log/restart/open-file limits differ")
        config = attrs.get("Config") or {}
        expected_config = {
            "User": "0:0",
            "Entrypoint": ["/bin/sleep"],
            "Cmd": ["infinity"],
            "Healthcheck": {"Test": ["NONE"]},
            "WorkingDir": profile.workdir,
        }
        if any(config.get(key) != value for key, value in expected_config.items()):
            raise InfrastructureError("sandbox image startup configuration differs")
        environment = dict(item.split("=", 1) for item in config.get("Env", []))
        if environment != self._environments[child.id]:
            raise InfrastructureError("sandbox unexpected image environment")
        roots = dict(profile.tmpfs_mb)
        for mount in attrs.get("Mounts") or []:
            if mount.get("Type") != "tmpfs" or mount.get("Destination") not in roots:
                raise InfrastructureError(
                    "sandbox has an unauthorized filesystem mount"
                )
        networks = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
        if set(networks) - {"none"}:
            raise InfrastructureError("sandbox has an unauthorized network attachment")
        allowed_apparmor = (
            ("docker-default",)
            if (attrs.get("State") or {}).get("Running")
            else ("", "docker-default")
        )
        if attrs.get("AppArmorProfile") not in allowed_apparmor:
            raise InfrastructureError("sandbox AppArmor confinement is absent")

    def start(self, lease: SandboxLease) -> None:
        child = self._owned(lease)
        self._attest_configuration(lease, child)
        child.start()
        child = self._owned(lease)
        self._attest_configuration(lease, child)
        if not child.attrs["State"].get("Running"):
            raise InfrastructureError("sandbox idle command did not remain running")
        try:
            probe = self.execute(
                lease,
                ["/bin/sh", "-c", "exit 0"],
                self._profiles[lease.container_id].workdir,
                {},
                time.monotonic() + 5.0,
                1024,
            )
        except Exception as error:
            raise InfrastructureError(
                "sandbox approved image shell readiness failed"
            ) from error
        if probe.exit_code != 0:
            raise InfrastructureError(
                "sandbox approved image needs a working /bin/sh shell"
            )

    def pause(self, lease: SandboxLease) -> None:
        child = self._owned(lease)
        if not child.attrs["State"].get("Paused"):
            child.pause()
        if not self.inspect(lease)["State"].get("Paused"):
            raise InfrastructureError("sandbox pause could not be proven")

    def resume(self, lease: SandboxLease, *, admission=None) -> None:
        child = self._owned(lease)
        if child.attrs["State"].get("Paused"):
            with admission() if admission is not None else nullcontext():
                child.unpause()
        state = self.inspect(lease)["State"]
        if state.get("Paused") or not state.get("Running"):
            raise InfrastructureError("sandbox resume could not be proven")

    def terminate(self, lease: SandboxLease) -> None:
        child = self._owned(lease, missing_ok=True)
        if child is None:
            return
        state = child.attrs.get("State") or {}
        if state.get("Paused"):
            if self._paused_killer is None:
                # Moby kill/force-remove resumes paused tasks; no zero-execution
                # proof without the paused killer.
                raise InfrastructureError(
                    "sandbox is paused; safe termination unproven, recovery required"
                )
            terminate_container(
                self.client.api,
                child.id,
                inspect=lambda: self._attrs_or_none(lease),
                paused_killer=self._paused_killer,
            )
            return
        if state.get("Running"):
            child.kill(signal="SIGKILL")
        child = self._owned(lease, missing_ok=True)
        if child is not None and child.attrs["State"].get("Running"):
            raise InfrastructureError("sandbox termination could not be proven")

    def remove(self, lease: SandboxLease) -> None:
        child = self._owned(lease, missing_ok=True)
        if child is None:
            self._profiles.pop(lease.container_id, None)
            self._environments.pop(lease.container_id, None)
            return
        if child.attrs["State"].get("Paused"):
            raise InfrastructureError(
                "sandbox is paused; refusing implicit thaw during removal"
            )
        if child.attrs["State"].get("Running"):
            raise InfrastructureError("sandbox must be terminated before removal")
        child.remove(force=False, v=True)
        if self._owned(lease, missing_ok=True) is not None:
            raise InfrastructureError("sandbox removal could not be proven")
        self._profiles.pop(lease.container_id, None)
        self._environments.pop(lease.container_id, None)

    def _validate_command(self, lease, argv, cwd, env, deadline, output_limit):
        profile = self._profiles.get(lease.container_id)
        if profile is None:
            raise InfrastructureError("sandbox execution lacks profile authority")
        if (
            not isinstance(argv, (tuple, list))
            or not argv
            or any(type(value) is not str or "\x00" in value for value in argv)
        ):
            raise SandboxError(
                "invalid", "argv", "expected a nonempty string argument vector"
            )
        if not isinstance(env, dict) or any(
            type(key) is not str
            or not key
            or "=" in key
            or "\x00" in key
            or type(value) is not str
            or "\x00" in value
            for key, value in env.items()
        ):
            raise SandboxError(
                "invalid", "env", "expected string environment names and values"
            )
        if (
            sum(len(value.encode()) for value in argv)
            + sum(len(key.encode()) + len(value.encode()) for key, value in env.items())
            > MAX_COMMAND_BYTES
        ):
            raise SandboxError("quota", "argv/env", "command exceeds 64 KiB")
        try:
            absolute_path(cwd)
        except (ValueError, TypeError) as error:
            raise SandboxError(
                "invalid", "cwd", "expected normalized absolute scratch path"
            ) from error
        if not any(below(cwd, root) for root, _ in profile.tmpfs_mb):
            raise SandboxError("permission", "cwd", "outside approved scratch roots")
        if type(deadline) not in (float, int) or not math.isfinite(deadline):
            raise SandboxError(
                "invalid", "deadline", "expected finite absolute deadline"
            )
        if type(output_limit) is not int or output_limit <= 0:
            raise SandboxError(
                "invalid", "output_limit", "expected positive byte limit"
            )

    def _transfer_argv(self, lease, root, operation, byte_limit):
        if type(byte_limit) is not int or not 0 <= byte_limit <= wire.MAX_BUNDLE_BYTES:
            raise SandboxError("invalid", "byte_limit", "invalid transfer byte limit")
        profile = self._profiles.get(lease.container_id)
        if profile is None:
            raise InfrastructureError("sandbox transfer lacks profile authority")
        try:
            absolute_path(root)
        except (ValueError, TypeError) as error:
            raise SandboxError(
                "invalid", "root", "expected normalized absolute scratch root"
            ) from error
        if not any(below(root, allowed) for allowed, _ in profile.tmpfs_mb):
            raise SandboxError("permission", "root", "outside approved writable roots")
        protected = ("/usr", "/bin", "/lib", "/lib64", "/etc", "/opt")
        if any(
            below(path, base) or below(base, path)
            for path, _ in profile.tmpfs_mb
            for base in protected
        ):
            raise SandboxError(
                "unsupported",
                "profile.tmpfs_mb",
                "transfer helper needs immutable Python/runtime paths",
            )
        if self.inspect(lease)["State"].get("Paused"):
            raise SandboxError("busy", "state", "paused child cannot transfer files")
        source = Path(wire.__file__).read_text()
        script = (
            "import sys;scope={'__name__':'_rsi_transfer'};exec("
            + repr(source)
            + ",scope);scope['_transfer_main'](sys.argv[1],sys.argv[2],"
            + "None,int(sys.argv[3]))"
        )
        return [
            "python3",
            "-I",
            "-S",
            "-c",
            script,
            operation,
            root,
            str(byte_limit),
        ]

    def upload(
        self, lease, root, entries, deadline, *, byte_limit=wire.MAX_BUNDLE_BYTES
    ):
        payload = encode_bundle(entries, byte_limit=byte_limit)
        argv = self._transfer_argv(lease, root, "upload", byte_limit)
        cwd = self._profiles[lease.container_id].workdir
        self._validate_command(lease, argv, cwd, {}, deadline, 8192)
        result, stdout, stderr = self._run_exec(
            lease, argv, cwd, {}, deadline, 8192, payload
        )
        self._require_transfer_success(result, stderr)
        if stdout != b"ok\n":
            raise SandboxError(
                "unknown-outcome", "upload", "missing transfer acknowledgement"
            )

    def download(
        self, lease, root, paths, deadline, *, byte_limit=wire.MAX_BUNDLE_BYTES
    ):
        try:
            if (
                not isinstance(paths, (tuple, list))
                or not 1 <= len(paths) <= wire.MAX_ENTRIES
            ):
                raise SandboxError(
                    "invalid", "paths", "expected bounded nonempty selection"
                )
            try:
                for path in paths:
                    if path != ".":
                        wire.validate_name(path)
            except (ValueError, TypeError) as error:
                raise SandboxError("invalid", "paths", str(error)) from error
            control = json.dumps(paths, separators=(",", ":")).encode()
            if len(control) > wire.MAX_CONTROL_BYTES:
                raise SandboxError("quota", "paths", "download selection exceeds limit")
            argv = self._transfer_argv(lease, root, "download", byte_limit)
            cwd = self._profiles[lease.container_id].workdir
            self._validate_command(lease, argv, cwd, {}, deadline, wire.MAX_BODY_BYTES)
        except SandboxError as error:
            raise SandboxDownloadError(
                error.code,
                error.field,
                error.message,
                download_bytes=0,
                operation_started=False,
            ) from error
        # Sending bounded control on stdin keeps the helper source out of the
        # caller's metadata budget and avoids Linux's per-argument size limit.
        # Unknown outcomes from this call deliberately carry no refund evidence.
        result, stdout, stderr = self._run_exec(
            lease, argv, cwd, {}, deadline, wire.MAX_BODY_BYTES, control
        )
        try:
            self._require_transfer_success(result, stderr)
            return decode_bundle(stdout, byte_limit=byte_limit)
        except SandboxChildStopped as error:
            # A killed reader may have discarded buffered output before EOF.
            # Keep the full reservation rather than infer a partial transfer.
            raise SandboxDownloadStopped(
                error.code,
                error.field,
                error.message,
                download_bytes=byte_limit,
                operation_started=True,
            ) from error
        except SandboxError as error:
            raise SandboxDownloadError(
                error.code,
                error.field,
                error.message,
                download_bytes=self._observed_download_bytes(stdout, byte_limit),
                operation_started=True,
            ) from error

    @staticmethod
    def _observed_download_bytes(payload, byte_limit):
        """Count validated payload; conservatively charge any undecodable tail."""
        stream = io.BytesIO(payload)
        consumed = len(wire.MAGIC) if payload.startswith(wire.MAGIC) else 0
        transferred = 0
        try:
            for record in wire.iter_bundle(stream, max_bytes=byte_limit):
                transferred += len(record["data"])
                consumed = stream.tell()
        except (wire.ProtocolError, OSError, ValueError):
            transferred += len(payload) - consumed
        return min(byte_limit, transferred)

    @staticmethod
    def _require_transfer_success(result, stderr):
        if result.timed_out:
            raise SandboxChildStopped(
                "expired", "transfer", "transfer deadline expired; child terminated"
            )
        if result.output_limited or result.oom_killed:
            raise SandboxChildStopped(
                "quota",
                "transfer",
                "transfer resource limit exceeded; child terminated",
            )
        if result.exit_code != 0:
            raise SandboxError(
                "invalid",
                "transfer",
                stderr[:8192].decode("utf-8", "replace")
                or "child transfer helper failed",
            )

    def execute(self, lease, argv, cwd, env, deadline, output_limit):
        self._validate_command(lease, argv, cwd, env, deadline, output_limit)
        result, stdout, stderr = self._run_exec(
            lease,
            argv,
            cwd,
            env,
            deadline,
            min(output_limit, MAX_OUTPUT_BYTES),
        )
        return result.model_copy(
            update={
                "stdout": stdout.decode("utf-8", "replace"),
                "stderr": stderr.decode("utf-8", "replace"),
            }
        )

    def _run_exec(
        self, lease, argv, cwd, env, deadline, output_limit, input_bytes=None
    ):
        """Bound the entire start/read/wait, including daemon startup latency."""
        started = time.monotonic()
        done = threading.Event()
        cancel = threading.Event()
        excess = threading.Event()
        stdout, stderr = bytearray(), bytearray()
        errors = []
        exit_code = None

        def check_deadline():
            if cancel.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("sandbox execution deadline expired")

        def worker():
            nonlocal exit_code
            handle = None
            try:
                check_deadline()
                child = self._owned(lease)
                if child.attrs["State"].get("Paused"):
                    raise SandboxError("busy", "state", "paused child cannot execute")
                created = self.client.api.exec_create(
                    lease.container_id,
                    list(argv),
                    stdout=True,
                    stderr=True,
                    stdin=input_bytes is not None,
                    tty=False,
                    environment=env,
                    workdir=cwd,
                    user="0:0",
                )
                check_deadline()
                identity = created["Id"]
                handle = self.client.api.exec_start(identity, socket=True, tty=False)
                connection = getattr(handle, "_sock", handle)
                connection.setblocking(False)
                view = memoryview(input_bytes or b"")
                offset = 0
                input_error = None
                writing = input_bytes is not None
                poller = select.poll()
                if writing and not view:
                    connection.shutdown(socket.SHUT_WR)
                    writing = False

                def read(size, *, eof_ok=False):
                    nonlocal offset, writing, input_error
                    data = bytearray()
                    while len(data) < size:
                        check_deadline()
                        poller.register(
                            connection,
                            select.POLLIN | (select.POLLOUT if writing else 0),
                        )
                        events = poller.poll(
                            min(100, max(0, deadline - time.monotonic()) * 1000)
                        )
                        if not events:
                            continue
                        flags = events[0][1]
                        if flags & select.POLLNVAL:
                            raise OSError("sandbox transport descriptor is invalid")
                        # Drain output while sending input. A helper may
                        # reject the first record while Docker still accepts
                        # stdin, and a full-duplex peer must never deadlock us.
                        if writing and flags & select.POLLOUT:
                            try:
                                sent = connection.send(view[offset : offset + 65536])
                                if not sent:
                                    raise OSError("sandbox stdin disconnected")
                                offset += sent
                                if offset == len(view):
                                    connection.shutdown(socket.SHUT_WR)
                                    writing = False
                            except BlockingIOError:
                                pass
                            except OSError as error:
                                input_error = error
                                writing = False
                        if not flags & (
                            select.POLLIN | select.POLLHUP | select.POLLERR
                        ):
                            continue
                        try:
                            part = connection.recv(min(size - len(data), 65536))
                        except BlockingIOError:
                            continue
                        if not part:
                            if not data and eof_ok:
                                return None
                            raise OSError("truncated Docker exec frame")
                        data.extend(part)
                    return bytes(data)

                while True:
                    header = read(8, eof_ok=True)
                    if header is None:
                        break
                    stream, length = struct.unpack("!BxxxI", header)
                    if stream not in (1, 2):
                        raise OSError("invalid Docker exec stream")
                    target = stdout if stream == 1 else stderr
                    while length:
                        chunk = read(min(length, 65536))
                        room = output_limit - len(stdout) - len(stderr)
                        if stream == 2:
                            room = min(room, MAX_OUTPUT_BYTES - len(stderr))
                        target.extend(chunk[:room])
                        if len(chunk) > room:
                            excess.set()
                            return
                        length -= len(chunk)
                while True:
                    check_deadline()
                    state = self.client.api.exec_inspect(identity)
                    if not state.get("Running") and type(state.get("ExitCode")) is int:
                        exit_code = state["ExitCode"]
                        if exit_code == 0 and (input_error or offset != len(view)):
                            raise OSError("sandbox stdin was not fully delivered")
                        break
                    cancel.wait(0.01)
            except BaseException as error:
                errors.append(error)
            finally:
                try:
                    if handle is not None:
                        response = getattr(handle, "_response", None)
                        try:
                            if response is not None:
                                response.close()
                        finally:
                            handle.close()
                except BaseException as error:
                    errors.append(error)
                finally:
                    done.set()

        thread = threading.Thread(target=worker, name="rsi-sandbox-exec", daemon=True)
        thread.start()
        while not done.wait(max(0, min(0.02, deadline - time.monotonic()))):
            if time.monotonic() >= deadline or excess.is_set():
                break
        timed_out = time.monotonic() >= deadline
        output_limited = excess.is_set()
        if timed_out or output_limited:
            cancel.set()
            self.terminate(lease)
        # A timed-out transport is never declared reconciled while still mutating.
        thread.join(timeout=6.0)
        if thread.is_alive():
            raise SandboxError(
                "unknown-outcome",
                "exec",
                "Docker request still pending; retain child for recovery",
            )
        if errors and not (timed_out or output_limited):
            if isinstance(errors[0], SandboxError):
                raise errors[0]
            raise SandboxError(
                "unknown-outcome",
                "exec",
                "Docker execution failed; do not replay automatically",
            ) from errors[0]
        state = self.inspect(lease)["State"]
        oom_killed = bool(state.get("OOMKilled", False))
        if oom_killed:
            self.terminate(lease)
        result = SandboxResult(
            exit_code=None if timed_out or output_limited else exit_code,
            timed_out=timed_out,
            oom_killed=oom_killed,
            output_limited=output_limited,
            truncated=output_limited,
            duration_sec=max(0.0, time.monotonic() - started),
        )
        return result, bytes(stdout), bytes(stderr)
