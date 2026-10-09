"""Per-session BuildKit builders and the image builds they run (spec 4).

Each phase session (the Work session, or one Judge round) gets its own
BuildKit daemon in a broker-created "builder" container, created lazily on
the session's first build. The broker drives it only through Engine exec of
the image's own ``buildctl``; Work, Judge and children never see a Docker or
BuildKit socket. The finished image leaves as a docker-archive on the exec's
stdout, passes the streaming sanitizer (image_archive) into ``POST
/images/load``, and is handed back as a session-scoped ``i…`` handle.

The builder is the one documented exception to the service template (B2):
``CapAdd`` is exactly SYS_ADMIN and NET_ADMIN, and seccomp, AppArmor and the
masked system paths are unconfined, so BuildKit can sandbox RUN steps
itself. It is still runc, unprivileged, with no host bind, device or port,
on its own firewalled bridge (rule, bridge, container; teardown in reverse),
with cgroup limits from the grant and a fixed-size state filesystem
(sandbox_buildfs). The Engine API has no ``systempaths=unconfined`` security
option (the docker CLI translates it), so it is sent and attested as empty
``MaskedPaths`` and ``ReadonlyPaths``.

Journal order (S7): the builder lease is planned before any Docker call and
every created object is committed before the next step; a built image's
config digest is committed before the daemon can see the end of the load
stream (B8). A failed build, a vanished builder or a drifted one fails only
that build (``infrastructure``); only an unprovable removal or a journal
failure fails the run closed.
"""

from __future__ import annotations

import errno
import http.client
import io
import json
import logging
import math
import os
import re
import secrets
import select
import socket
import struct
import tarfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from docker.errors import APIError, DockerException, NotFound
from docker.utils.json_stream import json_stream

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.build_context import BuildInput
from rsi_harness.runtime.image_archive import ImageArchiveError, ImageArchiveSanitizer
from rsi_harness.runtime.sandbox_archive import CanonicalTarWriter
from rsi_harness.runtime.sandbox_contracts import (
    EnvBuildGrant,
    SandboxError,
    SandboxOwner,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    BUILD_ROLE,
    BUILDER_ROLE,
    LABEL_PREFIX,
    BuilderLease,
    SandboxImageLease,
    builder_container_name,
    builder_network_name,
    builder_rule_id,
    builder_volume_name,
    built_image_tag,
    sandbox_object_labels,
)
from rsi_harness.runtime.sandbox_env_docker import (
    DockerPausedKiller,
    image_preflight,
    terminate_container,
)
from rsi_harness.runtime.sandbox_images import (
    Image,
    Job,
    JobFailed,
    image_view,
)
from rsi_harness.runtime.sandbox_network import (
    SandboxNetworkBackend,
    plan_builder_network,
)

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
NANOSECONDS = 1_000_000_000
BUILDKIT_STATE_DIR = "/var/lib/buildkit"
BUILDKIT_CONFIG_DIR = "/etc/buildkit"
BUILDKIT_CONFIG = "buildkitd.toml"
# buildkitd always serves an OTLP trace collector and binds it into every
# RUN step at /dev/otel-grpc.sock while this file exists (v0.27.1 stats it
# per step, executor/oci/spec.go); it is pinned here and unlinked once the
# daemon is ready, so no BuildKit socket reaches a RUN step (S1).
BUILDKIT_TRACE_SOCKET = "/run/buildkit/otel-grpc.sock"
# Resolve through Docker's embedded DNS in the builder's netns; without it
# BuildKit silently uses 8.8.8.8 (VERIFIED).
BUILDKITD_TOML = (
    b'[dns]\n  nameservers = ["127.0.0.11"]\n'
    b'[otel]\n  socketPath = "' + BUILDKIT_TRACE_SOCKET.encode() + b'"\n'
)
BUILDER_ENTRYPOINT = ("buildkitd",)
BUILDER_CAP_ADD = ("CAP_SYS_ADMIN", "CAP_NET_ADMIN")
BUILDER_SECURITY_OPT = (
    "apparmor=unconfined",
    "seccomp=unconfined",
    "writable-cgroups=true",
)
BUILDER_NOFILE = 65536
BUILDER_LOG_CONFIG = {"Type": "none", "Config": {}}
READY_SEC = 10.0
CANCEL_GRACE_SEC = 10.0
EXEC_SEC = 60.0
LOAD_TIMEOUT_SEC = 600.0
# A closed load whose whole stream may be at the daemon waits this long for
# its answer, the only proof of what it registered: inside close_judge's
# DELETE_SEC (60 s), with time left to remove the image and the builder.
LOAD_ANSWER_GRACE_SEC = 30.0
# How long a load's send or read blocks before it looks for a close.
LOAD_POLL_SEC = 0.1
KILL_PROOF_SEC = 5.0
MEMORY_RESERVE_MB = 256
PIDS_RESERVE = 64
MAX_EXEC_OUTPUT = 64 * 1024
MAX_METADATA_BYTES = 64 * 1024
FLOOR_CHECK_SEC = 1.0
# A quiet load answers one line; more is not the daemon's answer.
MAX_LOAD_ANSWER = 64 * 1024
MAX_LOAD_RESPONSE = 2 * MAX_LOAD_ANSWER
_LOADED = re.compile(r"^Loaded image ID: (sha256:[0-9a-f]{64})\s*$")
_NO_SPACE = ("no space left on device", "enospc")
RECOVERY_REQUIRED = "recovery_required"
_EMPTY_HOST_FIELDS = (
    "Binds",
    "VolumesFrom",
    "Links",
    "Devices",
    "DeviceRequests",
    "DeviceCgroupRules",
    "PortBindings",
    "PidMode",
    "UTSMode",
    "UsernsMode",
    "Cgroup",
    "CgroupParent",
    "Sysctls",
    "Dns",
    "DnsSearch",
    "DnsOptions",
    "ExtraHosts",
    "Tmpfs",
    "VolumeDriver",
    "OomKillDisable",
    "AutoRemove",
    "CapDrop",
    "GroupAdd",
    "Isolation",
)

Commit = Callable[[BuilderLease], BuilderLease]


def _revise(model: Any, **changes: Any) -> Any:
    """A validated copy: every journal record keeps its invariants."""
    return type(model)(**{**dict(model), **changes})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _answered(error: Exception) -> bool:
    return isinstance(error, APIError) and error.status_code is not None


def _recovery(error: BaseException) -> bool:
    return isinstance(error, InfrastructureError) and str(error).startswith(
        RECOVERY_REQUIRED
    )


def _builder_env(image_env: Sequence[str]) -> list[str]:
    """Image Env with NVIDIA_* blanked and GPUs forced off (runc anyway)."""
    merged: dict[str, str | None] = {}
    for item in image_env:
        key, separator, value = item.partition("=")
        merged[key] = value if separator else None
    for key in merged:
        if key.upper().startswith("NVIDIA_"):
            merged[key] = ""
    merged["NVIDIA_VISIBLE_DEVICES"] = "void"
    return [key if value is None else f"{key}={value}" for key, value in merged.items()]


# -- plan ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuilderPlan:
    """Every name and the exact create body of one builder."""

    owner: SandboxOwner
    builder_id: str
    grant: EnvBuildGrant
    image_id: str
    env: tuple[str, ...]
    labels: Mapping[str, str]

    @property
    def container_name(self) -> str:
        return builder_container_name(self.builder_id)

    @property
    def volume_name(self) -> str:
        return builder_volume_name(self.builder_id)

    @property
    def network(self) -> Any:
        return plan_builder_network(self.owner, self.builder_id)

    @property
    def network_name(self) -> str:
        return builder_network_name(self.builder_id)

    @property
    def memory_bytes(self) -> int:
        return self.grant.memory_mb * MIB

    @property
    def build_memory_max(self) -> int:
        """``/buildkit`` memory.max: room for buildkitd when a RUN OOMs."""
        reserve = max(MEMORY_RESERVE_MB * MIB, self.memory_bytes // 10)
        # Whole MiB: the kernel rounds memory.max down to pages, and the
        # read-back must see exactly what was written.
        return (self.memory_bytes - reserve) // MIB * MIB

    @property
    def build_pids_max(self) -> int:
        return self.grant.pids - PIDS_RESERVE

    def command(self) -> list[str]:
        disk = self.grant.disk_mb
        keep = f"{int(disk * 0.3)},{int(disk * 0.2)},{int(disk * 0.8)}"
        return [
            "--config",
            f"{BUILDKIT_CONFIG_DIR}/{BUILDKIT_CONFIG}",
            "--oci-worker-net=host",
            "--oci-max-parallelism",
            str(self.grant.cpus),
            "--oci-worker-gc",
            "--oci-worker-gc-keepstorage",
            keep,
        ]

    def host_config(self) -> dict[str, Any]:
        return {
            "Runtime": "runc",
            "Privileged": False,
            "CapAdd": list(BUILDER_CAP_ADD),
            "SecurityOpt": list(BUILDER_SECURITY_OPT),
            # The Engine form of --security-opt systempaths=unconfined.
            "MaskedPaths": [],
            "ReadonlyPaths": [],
            "CgroupnsMode": "private",
            "IpcMode": "private",
            "NetworkMode": self.network_name,
            "Mounts": [
                {
                    "Type": "volume",
                    "Source": self.volume_name,
                    "Target": BUILDKIT_STATE_DIR,
                    "ReadOnly": False,
                }
            ],
            "NanoCpus": self.grant.cpus * NANOSECONDS,
            "Memory": self.memory_bytes,
            "MemorySwap": self.memory_bytes,
            "PidsLimit": self.grant.pids,
            "Ulimits": [
                {"Name": "nofile", "Soft": BUILDER_NOFILE, "Hard": BUILDER_NOFILE}
            ],
            "LogConfig": dict(BUILDER_LOG_CONFIG),
            # Rules are not persistent: a daemon-restarted builder would run
            # without its firewall.
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "PublishAllPorts": False,
            "ReadonlyRootfs": False,
        }

    def body(self) -> dict[str, Any]:
        return {
            "Image": self.image_id,
            "Entrypoint": list(BUILDER_ENTRYPOINT),
            "Cmd": self.command(),
            "Env": list(self.env),
            "User": "0:0",
            "Labels": dict(self.labels),
            "HostConfig": self.host_config(),
            "NetworkingConfig": {"EndpointsConfig": {self.network_name: {}}},
        }

    def lease(self) -> BuilderLease:
        return BuilderLease(
            owner=self.owner,
            builder_id=self.builder_id,
            container_name=self.container_name,
            volume_name=self.volume_name,
            network_name=self.network_name,
            rule_id=builder_rule_id(self.owner.run_id, self.builder_id),
            state_fs=self.grant.state_fs,
            cpus=self.grant.cpus,
            memory_mb=self.grant.memory_mb,
            disk_mb=self.grant.disk_mb,
            pending_mutation=True,
        )


def builder_labels(owner: SandboxOwner, builder_id: str) -> dict[str, str]:
    return sandbox_object_labels(owner, BUILDER_ROLE, {"sandbox-builder": builder_id})


def built_image_labels(owner: SandboxOwner, handle: str) -> dict[str, str]:
    """Forced on every built image through ``--opt label:`` (they override
    any Dockerfile LABEL, VERIFIED); env containers re-set every key."""
    return sandbox_object_labels(owner, BUILD_ROLE, {"sandbox-image": handle})


# -- exec streams ------------------------------------------------------------------


class ExecAborted(Exception):
    """The caller's check stopped reading an exec stream."""


class ExecStream(io.RawIOBase):
    """The stdout of one attached, non-tty exec as a readable file.

    Docker multiplexes stdout and stderr in 8-byte-header frames; stderr
    frames go to ``on_stderr`` as they arrive. ``check`` runs at least every
    ``tick`` seconds while the stream waits and may raise to stop it.
    """

    def __init__(
        self,
        handle: Any,
        *,
        on_stderr: Callable[[bytes], None],
        check: Callable[[], None],
        tick: float = 0.1,
    ) -> None:
        self._handle = handle
        self._socket = getattr(handle, "_sock", handle)
        self._socket.setblocking(False)
        self._on_stderr = on_stderr
        self._check = check
        self._tick = tick
        self._remaining = 0
        self._eof = False

    def readable(self) -> bool:
        return True

    def _recv(self, size: int) -> bytes:
        poller = select.poll()
        poller.register(self._socket, select.POLLIN | select.POLLHUP | select.POLLERR)
        while True:
            self._check()
            if not poller.poll(int(self._tick * 1000)):
                continue
            try:
                return self._socket.recv(size)
            except (BlockingIOError, InterruptedError):
                continue

    def _exact(self, size: int, *, eof_ok: bool = False) -> bytes | None:
        data = bytearray()
        while len(data) < size:
            part = self._recv(size - len(data))
            if not part:
                if not data and eof_ok:
                    return None
                raise OSError("truncated Docker exec frame")
            data.extend(part)
        return bytes(data)

    def _frame(self) -> bool:
        """Advance to the next stdout payload; False at end of stream."""
        while not self._remaining:
            header = self._exact(8, eof_ok=True)
            if header is None:
                self._eof = True
                return False
            stream, length = struct.unpack("!BxxxI", header)
            if stream == 1:
                self._remaining = length
                continue
            if stream != 2:
                raise OSError("invalid Docker exec stream")
            while length:
                chunk = self._exact(min(length, 65536))
                assert chunk is not None
                self._on_stderr(chunk)
                length -= len(chunk)
        return True

    def readinto(self, buffer) -> int:
        if self._eof or not self._frame():
            return 0
        size = min(len(buffer), self._remaining, 1 << 20)
        data = self._recv(size)
        if not data:
            raise OSError("truncated Docker exec frame")
        buffer[: len(data)] = data
        self._remaining -= len(data)
        return len(data)

    def peek_stdout(self) -> bool:
        """Wait for the first stdout byte; False when the exec ended first."""
        return not self._eof and self._frame()

    def drain(self) -> None:
        """Read to the end, discarding stdout."""
        while self.read(1 << 20):
            pass

    def close(self) -> None:
        try:
            response = getattr(self._handle, "_response", None)
            if response is not None:
                response.close()
        finally:
            try:
                self._handle.close()
            finally:
                super().close()


# -- backend -------------------------------------------------------------------------


class BuilderBackend:
    """Create, attest, drive and remove builder containers.

    ``statefs`` maps a grant's ``state_fs`` to its state filesystem
    (sandbox_buildfs); ``cgroup_root`` and ``proc_root`` are the host's, for
    the OOM counter and the ``/buildkit`` read-back (world-readable).
    ``load_connect`` opens a load's own connection to the daemon (default:
    the client's Unix socket).
    """

    def __init__(
        self,
        client: Any,
        network: SandboxNetworkBackend,
        statefs: Callable[[str], Any],
        *,
        cgroup_root: Path | None = Path("/sys/fs/cgroup"),
        proc_root: Path = Path("/proc"),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        ready_sec: float = READY_SEC,
        kill_proof_sec: float = KILL_PROOF_SEC,
        load_connect: Callable[[], socket.socket] | None = None,
    ) -> None:
        self._client = client
        self._api = client.api
        self._load_connect = load_connect
        self._network = network
        self._statefs = statefs
        self._cgroup_root = None if cgroup_root is None else Path(cgroup_root)
        self._proc_root = Path(proc_root)
        self._clock = clock
        self._sleep = sleep
        self._ready_sec = ready_sec
        self._kill_proof_sec = kill_proof_sec

    # -- plan ------------------------------------------------------------------

    def plan(
        self, owner: SandboxOwner, builder_id: str, grant: EnvBuildGrant
    ) -> BuilderPlan:
        try:
            attrs = self._api.inspect_image(grant.builder_image)
        except NotFound:
            raise SetupError("the approved builder image is no longer cached") from None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot inspect builder image: {error}"
            ) from error
        config = attrs.get("Config") or {}
        if attrs.get("Id") != grant.builder_image or config.get("Entrypoint") != list(
            BUILDER_ENTRYPOINT
        ):
            raise SetupError("the cached builder image changed since approval")
        image_labels = dict(config.get("Labels") or {})
        if any(key.startswith(LABEL_PREFIX) for key in image_labels):
            raise SetupError("the builder image carries a reserved label")
        return BuilderPlan(
            owner=owner,
            builder_id=builder_id,
            grant=grant,
            image_id=grant.builder_image,
            env=tuple(_builder_env(config.get("Env") or [])),
            labels={**image_labels, **builder_labels(owner, builder_id)},
        )

    # -- create ----------------------------------------------------------------

    def create(
        self,
        plan: BuilderPlan,
        lease: BuilderLease,
        commit: Commit,
        *,
        abort: Callable[[], bool] = lambda: False,
    ) -> BuilderLease:
        """Rule, bridge, state fs, container; configure, start and attest.

        ``lease`` is the committed ``plan.lease()``. A failure with a known
        outcome removes everything and raises SetupError; anything unproven
        raises InfrastructureError marked ``recovery_required``. ``abort``
        (the session ended) is checked before the container is made, before
        and after it starts and while it gets ready: a session end whose
        kill found no container yet still leaves no running builder.
        """
        if lease.state != "planned" or not lease.pending_mutation:
            raise InfrastructureError("builder create needs its pending plan")
        journal = _Journal(commit)

        def check() -> None:
            if abort():
                raise SetupError("the session ended while its builder was made")

        try:
            network_id = self._network.create(plan.network)
            lease = journal(_revise(lease, network_id=network_id))
            lease = self._statefs(lease.state_fs).create(lease, journal)
            check()
            container_id = self._create_container(plan)
            lease = journal(_revise(lease, container_id=container_id, state="created"))
            self.attest(plan, lease)
            self._configure(lease)
            check()
            self._start(lease)
            lease = journal(_revise(lease, state="running", pending_mutation=False))
            check()
            self.attest(plan, lease, started=True)
            # buildkitd moves itself into ``init`` and enables the subtree
            # controllers as it starts: only a ready daemon has done so.
            self._ready(lease, check)
            self._reserve(plan, lease, check)
            check()
            return lease
        except _JournalFailure as failure:
            raise failure.error from failure.error.__cause__
        except SetupError as error:
            return self._roll_back(lease, commit, error)
        except InfrastructureError as error:
            if _recovery(error):
                raise
            # An attestation or inspection failure: every object is where
            # the journal says, so the whole builder rolls back.
            return self._roll_back(lease, commit, error)

    def _roll_back(
        self, lease: BuilderLease, commit: Commit, error: Exception
    ) -> BuilderLease:
        try:
            self.destroy(lease, commit)
        except Exception as rollback_error:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder {lease.builder_id} rollback is "
                f"unproven: {rollback_error}"
            ) from error
        raise SetupError(
            f"builder {lease.builder_id} could not be made ready: {error}; every "
            "object is proven absent"
        ) from error

    def _create_container(self, plan: BuilderPlan) -> str:
        try:
            created = self._api.create_container_from_config(
                plan.body(), name=plan.container_name
            )
        except Exception as error:
            if _answered(error):
                raise SetupError(f"the Engine refused the builder: {error}") from error
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder {plan.builder_id} create outcome is "
                f"unknown: {error}"
            ) from error
        identity = (created or {}).get("Id")
        if not isinstance(identity, str) or len(identity) != 64:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder {plan.builder_id} create returned "
                "no container identity"
            )
        return identity

    def _configure(self, lease: BuilderLease) -> None:
        buffer = io.BytesIO()
        writer = CanonicalTarWriter(buffer, max_bytes=len(BUILDKITD_TOML))
        writer.add_file(
            (BUILDKIT_CONFIG,),
            io.BytesIO(BUILDKITD_TOML),
            len(BUILDKITD_TOML),
            mode=0o644,
            mtime=0,
        )
        writer.close()
        try:
            if not self._api.put_archive(
                lease.container_id, BUILDKIT_CONFIG_DIR, buffer.getvalue()
            ):
                raise SetupError("the builder configuration was not written")
        except APIError as error:
            raise SetupError(f"builder configuration failed: {error}") from error

    def _start(self, lease: BuilderLease) -> None:
        try:
            self._api.start(lease.container_id)
        except APIError as error:
            raise SetupError(f"the builder did not start: {error}") from error

    def _reserve(
        self,
        plan: BuilderPlan,
        lease: BuilderLease,
        check: Callable[[], None] = lambda: None,
    ) -> None:
        """``/buildkit`` limits below the builder's own, read back (B4)."""
        memory, pids = plan.build_memory_max, plan.build_pids_max
        # buildkitd (BUILDKIT_SETUP_CGROUPV2_ROOT) moves every process of the
        # builder's cgroup into ``init`` and delegates the controllers; an
        # exec landing in the root meanwhile makes that delegation fail
        # (EBUSY). Redo it idempotently, this shell first, then reserve.
        root = "/sys/fs/cgroup"
        script = (
            f"echo $$ > {root}/init/cgroup.procs"
            f" && for p in $(cat {root}/cgroup.procs);"
            f" do echo $p > {root}/init/cgroup.procs 2>/dev/null; done;"
            f" grep -qw memory {root}/cgroup.subtree_control"
            f" || sed 's/[a-z_]*/+&/g' {root}/cgroup.controllers"
            f" > {root}/cgroup.subtree_control;"
            " mkdir -p /sys/fs/cgroup/buildkit"
            f" && echo {memory} > /sys/fs/cgroup/buildkit/memory.max"
            f" && echo {pids} > /sys/fs/cgroup/buildkit/pids.max"
            " && cat /sys/fs/cgroup/buildkit/memory.max"
            " /sys/fs/cgroup/buildkit/pids.max"
        )
        # buildkitd enables the subtree controllers asynchronously; under
        # load that can trail its readiness, so the write is retried.
        deadline = self._clock() + self._ready_sec
        while True:
            check()
            code, output = self.run(lease, ["sh", "-c", script])
            if code == 0 and output.split() == [str(memory), str(pids)]:
                break
            if self._clock() >= deadline:
                raise SetupError(
                    "the builder /buildkit cgroup reserve did not hold: "
                    f"exit {code}: {output.strip()[:256]}"
                )
            self._sleep(0.2)
        scope = self.scope(lease)
        if scope is not None:
            try:
                limits = [
                    (scope / "buildkit" / name).read_text().strip()
                    for name in ("memory.max", "pids.max")
                ]
            except OSError as error:
                raise SetupError(
                    f"cannot read back the builder cgroup reserve: {error}"
                ) from error
            if limits != [str(memory), str(pids)]:
                raise SetupError(
                    f"the builder /buildkit cgroup reserve reads back {limits}"
                )

    def _ready(
        self, lease: BuilderLease, check: Callable[[], None] = lambda: None
    ) -> None:
        deadline = self._clock() + self._ready_sec
        while True:
            check()
            code, _ = self.run(lease, ["buildctl", "debug", "workers"], timeout=5.0)
            if code == 0:
                break
            if self._clock() >= deadline:
                raise SetupError("buildkitd did not become ready")
            self._sleep(0.2)
        # The collector keeps its (now unreachable) listener; a step whose
        # spec is generated after this sees no socket at all.
        code, output = self.run(lease, ["rm", "-f", BUILDKIT_TRACE_SOCKET])
        if code != 0:
            raise SetupError(
                f"cannot remove the builder trace socket: exit {code}: "
                f"{output.strip()[:256]}"
            )

    # -- attest ----------------------------------------------------------------

    def inspect(self, lease: BuilderLease) -> dict[str, Any] | None:
        try:
            return self._api.inspect_container(lease.container_id)
        except NotFound:
            return None
        except (APIError, OSError) as error:
            raise InfrastructureError(f"cannot inspect builder: {error}") from error

    def attest(
        self, plan: BuilderPlan, lease: BuilderLease, *, started: bool = False
    ) -> None:
        """Field by field against the plan; any drift is an InfrastructureError."""
        attrs = self.inspect(lease)
        name = plan.container_name
        if attrs is None:
            raise InfrastructureError(f"builder {name} is absent")

        def mismatch(field_name: str) -> InfrastructureError:
            return InfrastructureError(
                f"builder {name} configuration mismatch: {field_name}"
            )

        if (
            attrs.get("Id") != lease.container_id
            or attrs.get("Name") != "/" + name
            or attrs.get("Image") != plan.image_id
        ):
            raise mismatch("identity")
        config = attrs.get("Config") or {}
        for key, value in (
            ("Image", plan.image_id),
            ("Entrypoint", list(BUILDER_ENTRYPOINT)),
            ("Cmd", plan.command()),
            ("Env", list(plan.env)),
            ("User", "0:0"),
            ("Labels", dict(plan.labels)),
        ):
            if _canonical(config.get(key)) != _canonical(value):
                raise mismatch(f"Config.{key}")
        host = attrs.get("HostConfig") or {}
        for key, value in plan.host_config().items():
            if key == "Mounts":
                continue
            if _canonical(host.get(key)) != _canonical(value):
                raise mismatch(f"HostConfig.{key}")
        for key in _EMPTY_HOST_FIELDS:
            if host.get(key):
                raise mismatch(f"HostConfig.{key}")
        mounts = attrs.get("Mounts") or []
        if [
            (
                mount.get("Type"),
                mount.get("Name"),
                mount.get("Destination"),
                mount.get("RW"),
            )
            for mount in mounts
        ] != [("volume", plan.volume_name, BUILDKIT_STATE_DIR, True)]:
            raise mismatch("Mounts")
        networks = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
        if set(networks) != {plan.network_name}:
            raise mismatch("Networks")
        self._statefs(lease.state_fs).attest(lease)
        if lease.network_id is None:
            raise mismatch("network identity")
        self._network.attest(
            plan.network, lease.network_id, containers=(lease.container_id,)
        )
        if started:
            state = attrs.get("State") or {}
            if not state.get("Running") or state.get("Paused"):
                raise InfrastructureError(f"builder {name} is not running")
            if attrs.get("AppArmorProfile") != "unconfined":
                raise mismatch("AppArmorProfile")

    # -- exec ------------------------------------------------------------------

    def exec_stream(
        self,
        lease: BuilderLease,
        argv: Sequence[str],
        *,
        on_stderr: Callable[[bytes], None],
        check: Callable[[], None],
    ) -> tuple[str, ExecStream]:
        created = self._api.exec_create(
            lease.container_id,
            list(argv),
            stdout=True,
            stderr=True,
            stdin=False,
            tty=False,
            user="0:0",
        )
        handle = self._api.exec_start(created["Id"], socket=True, tty=False)
        return created["Id"], ExecStream(handle, on_stderr=on_stderr, check=check)

    def exec_result(self, exec_id: str, *, timeout: float = 5.0) -> int | None:
        """The exit code once the exec ended, or None if it still runs."""
        deadline = self._clock() + timeout
        while True:
            state = self._api.exec_inspect(exec_id)
            if not state.get("Running") and type(state.get("ExitCode")) is int:
                return state["ExitCode"]
            if self._clock() >= deadline:
                return None
            self._sleep(0.05)

    def run(
        self,
        lease: BuilderLease,
        argv: Sequence[str],
        *,
        timeout: float = EXEC_SEC,
    ) -> tuple[int, str]:
        """Run a short broker command in the builder; bounded output."""
        deadline = self._clock() + timeout
        output = bytearray()

        def keep(data: bytes) -> None:
            output.extend(data[: max(0, MAX_EXEC_OUTPUT - len(output))])

        def check() -> None:
            if self._clock() >= deadline:
                raise ExecAborted("builder command timed out")

        try:
            exec_id, stream = self.exec_stream(lease, argv, on_stderr=keep, check=check)
        except (APIError, OSError, DockerException) as error:
            raise InfrastructureError(f"builder exec failed: {error}") from error
        try:
            while True:
                data = stream.read(65536)
                if not data:
                    break
                keep(data)
        except (ExecAborted, OSError) as error:
            raise InfrastructureError(f"builder command failed: {error}") from error
        finally:
            stream.close()
        code = self.exec_result(exec_id)
        if code is None:
            raise InfrastructureError("builder command did not end")
        return code, output.decode("utf-8", "replace")

    def put_input(self, lease: BuilderLease, path: Path) -> None:
        with open(path, "rb") as data:
            try:
                ok = self._api.put_archive(lease.container_id, BUILDKIT_STATE_DIR, data)
            except (APIError, OSError, DockerException) as error:
                raise JobFailed(
                    "infrastructure", f"the build context was not staged: {error}"
                ) from error
        if not ok:
            raise JobFailed("infrastructure", "the build context was not staged")

    def read_file(self, lease: BuilderLease, path: str) -> bytes:
        try:
            stream, _ = self._api.get_archive(lease.container_id, path)
        except (APIError, OSError, DockerException) as error:
            raise JobFailed("infrastructure", f"cannot read {path}: {error}") from error
        data = bytearray()
        for chunk in stream:
            data.extend(chunk)
            if len(data) > MAX_METADATA_BYTES + 4096:
                raise JobFailed("infrastructure", f"{path} is too large")
        with tarfile.open(fileobj=io.BytesIO(bytes(data))) as archive:
            member = archive.next()
            if member is None or not member.isreg():
                raise JobFailed("infrastructure", f"{path} is not a file")
            reader = archive.extractfile(member)
            assert reader is not None
            return reader.read(MAX_METADATA_BYTES)

    def image_load(self, timeout: float, grace: float) -> ImageLoad:
        """A ``POST /images/load`` on a connection of its own."""
        connect = self._load_connect or engine_socket(self._api)
        return ImageLoad(
            connect, urlsplit(self._api._url("/images/load")).path, timeout, grace
        )

    # -- cgroup ----------------------------------------------------------------

    def scope(self, lease: BuilderLease) -> Path | None:
        """The builder's host cgroup (buildkitd moves itself to ``init``)."""
        if self._cgroup_root is None:
            return None
        attrs = self.inspect(lease)
        pid = ((attrs or {}).get("State") or {}).get("Pid")
        if type(pid) is not int or pid <= 0:
            return None
        try:
            lines = (self._proc_root / str(pid) / "cgroup").read_text().splitlines()
        except OSError:
            return None
        unified = [line[3:] for line in lines if line.startswith("0::")]
        if len(unified) != 1:
            return None
        path = unified[0].rstrip("/")
        if path.endswith("/init"):
            path = path[: -len("/init")]
        if (
            not path.startswith("/")
            or lease.container_id not in path.rsplit("/", 1)[-1]
        ):
            return None
        return self._cgroup_root / path.lstrip("/")

    def oom_kills(self, lease: BuilderLease) -> int | None:
        scope = self.scope(lease)
        if scope is None:
            return None
        try:
            text = (scope / "buildkit" / "memory.events").read_text()
        except OSError:
            return 0
        for line in text.splitlines():
            key, _, value = line.partition(" ")
            if key == "oom_kill" and value.strip().isdigit():
                return int(value)
        return 0

    # -- teardown --------------------------------------------------------------

    def alive(self, plan: BuilderPlan, lease: BuilderLease) -> bool:
        """A running builder that still attests exactly."""
        if lease.state != "running":
            return False
        try:
            self.attest(plan, lease, started=True)
        except InfrastructureError as error:
            LOGGER.warning("builder %s is unusable: %s", lease.builder_id, error)
            return False
        return True

    def _owned_attrs(self, lease: BuilderLease) -> tuple[str, dict[str, Any]] | None:
        """The container holding the builder's identity, found by journaled
        ID or planned name; foreign objects are never touched."""
        for lookup in (lease.container_id, lease.container_name):
            if lookup is None:
                continue
            try:
                attrs = self._api.inspect_container(lookup)
            except NotFound:
                continue
            except (APIError, OSError) as error:
                raise InfrastructureError(
                    f"{RECOVERY_REQUIRED}: cannot inspect builder: {error}"
                ) from error
            labels = {
                key: value
                for key, value in (
                    (attrs.get("Config") or {}).get("Labels") or {}
                ).items()
                if key.startswith(LABEL_PREFIX)
            }
            if (
                attrs.get("Name") != "/" + lease.container_name
                or labels != builder_labels(lease.owner, lease.builder_id)
                or (lease.container_id not in (None, attrs.get("Id")))
            ):
                raise InfrastructureError(
                    f"{RECOVERY_REQUIRED}: builder {lease.container_name} is not "
                    "owned by its planned identity"
                )
            return attrs["Id"], attrs
        return None

    def kill(self, lease: BuilderLease, *, timeout: float | None = None) -> None:
        """SIGKILL the builder and prove it stopped (no journal write)."""
        found = self._owned_attrs(lease)
        if found is None:
            return
        identity, _ = found

        def inspect() -> Mapping[str, Any] | None:
            try:
                return self._api.inspect_container(identity)
            except NotFound:
                return None

        terminate_container(
            self._api,
            identity,
            inspect=inspect,
            paused_killer=DockerPausedKiller(self._api),
            clock=self._clock,
            sleep=self._sleep,
            timeout=self._kill_proof_sec if timeout is None else timeout,
        )

    def destroy(self, lease: BuilderLease, commit: Commit) -> BuilderLease:
        """Container, state fs, bridge, rule; each proven absent, journaled."""
        if lease.state == "removed" and not lease.pending_mutation:
            return lease
        lease = commit(_revise(lease, pending_mutation=True))
        found = self._owned_attrs(lease)
        if found is not None:
            identity, _ = found
            if lease.container_id is None:
                lease = commit(_revise(lease, container_id=identity))
            self.kill(lease)
            if lease.state != "planned":
                lease = commit(_revise(lease, state="stopped"))
            try:
                self._api.remove_container(identity, v=True, force=False)
            except NotFound:
                pass
            except (APIError, OSError) as error:
                LOGGER.warning("builder removal reported: %s", error)
            if self._owned_attrs(lease) is not None:
                raise InfrastructureError(
                    f"{RECOVERY_REQUIRED}: builder {lease.container_name} remains "
                    "after removal"
                )
        lease = self._statefs(lease.state_fs).remove(lease, commit)
        # Every create call of a live broker was answered before a teardown,
        # so finding no bridge proves it absent (recovery settles first).
        self._network.remove(
            plan_builder_network(lease.owner, lease.builder_id), lease.network_id
        )
        return commit(
            _revise(
                lease,
                state="removed",
                network_id=None,
                loop_device=None,
                pending_mutation=False,
            )
        )


class _JournalFailure(Exception):
    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


class _Journal:
    """Marks journal failures so they are never mistaken for Docker ones:
    the run already failed closed and nothing may be rolled back."""

    def __init__(self, commit: Commit) -> None:
        self._commit = commit

    def __call__(self, lease: BuilderLease) -> BuilderLease:
        try:
            return self._commit(lease)
        except Exception as error:
            raise _JournalFailure(error) from error


# -- builds ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BuildRequest:
    """One admitted build: its staged input and every caller option.

    ``lease`` is the planned built-image record journaled at admission.
    """

    input: BuildInput
    input_path: Path
    target: str | None
    build_args: Mapping[str, str]
    labels: Mapping[str, str]
    no_cache: bool
    network: str
    fingerprint: str
    lease: SandboxImageLease


def build_argv(request: BuildRequest, forced_labels: Mapping[str, str]) -> list[str]:
    """The broker's fixed ``buildctl build`` (B6): only the generated local
    context and Dockerfile, allowlisted options, forced labels and one
    docker-archive output on stdout. Never ``--allow``, ``--secret``,
    ``--ssh``, cache import/export or a push; every caller value travels
    inside one ``--opt key=value`` argument."""
    root = f"{BUILDKIT_STATE_DIR}/{request.input.directory}"
    argv = [
        "/bin/sh",
        "-c",
        f'echo $$ > {root}/pid && exec buildctl "$@"',
        "buildctl",
        "build",
        "--progress=plain",
        "--frontend",
        "dockerfile.v0",
        "--local",
        f"context={root}/ctx",
        "--local",
        f"dockerfile={root}/df",
        "--opt",
        "filename=Dockerfile",
    ]
    if request.target is not None:
        argv += ["--opt", f"target={request.target}"]
    for key, value in sorted(request.build_args.items()):
        argv += ["--opt", f"build-arg:{key}={value}"]
    for key, value in sorted(request.labels.items()):
        argv += ["--opt", f"label:{key}={value}"]
    for key, value in sorted(forced_labels.items()):
        argv += ["--opt", f"label:{key}={value}"]
    if request.no_cache:
        argv += ["--opt", "no-cache="]
    if request.network == "none":
        argv += ["--opt", "force-network-mode=none"]
    argv += [
        "--opt",
        "platform=linux/amd64",
        "--metadata-file",
        f"{root}/meta.json",
        "--output",
        "type=docker,dest=-",
    ]
    return argv


@dataclass(eq=False)
class _SessionBuilder:
    """A session's builder and its build slots (max_concurrent_builds)."""

    session: Any
    slots: threading.Semaphore
    lock: threading.Lock = field(default_factory=threading.Lock)
    plan: BuilderPlan | None = None
    lease: BuilderLease | None = None
    # Killed or drifted: the next build removes it and makes a new one.
    broken: bool = False
    reaper: threading.Thread | None = None
    # Built images whose removal hit an rmi conflict, retried at session end.
    leaked: list[SandboxImageLease] = field(default_factory=list)
    # The image IDs among them whose load never answered: their absence
    # proves nothing, the daemon may still register them.
    unanswered: set[str] = field(default_factory=set)


@dataclass(eq=False)
class _Build:
    """Mutable state of one running build."""

    lease: SandboxImageLease
    tail: bytearray = field(default_factory=bytearray)
    aborted: bool = False
    failure: BaseException | None = None
    # The daemon may have got the whole load stream but never answered: it
    # may still register the image, so finding nothing now proves nothing.
    unanswered: bool = False


class BuildService:
    """Session builders and the build jobs that run in them.

    ``envs`` (SandboxEnvs) owns authority, quotas and the journal; the job
    table is ``envs.images`` (sandbox_images.JobRunner). A session's builder
    is made on its first build and kept for the whole session (the Work
    cache spans the Work session, each Judge round has its own); idle, it
    keeps running across a Work freeze because no caller code runs in it.
    """

    def __init__(
        self,
        envs: Any,
        backend: BuilderBackend,
        *,
        spool_root: Path,
        clock: Callable[[], float] = time.monotonic,
        load_timeout: float = LOAD_TIMEOUT_SEC,
        load_grace: float = LOAD_ANSWER_GRACE_SEC,
        cancel_grace: float = CANCEL_GRACE_SEC,
    ) -> None:
        self._envs = envs
        self._lock = envs._lock
        self._backend = backend
        self._clock = clock
        self._load_timeout = load_timeout
        self._load_grace = load_grace
        self._cancel_grace = cancel_grace
        self._spool = Path(spool_root) / "build"
        self._builders: dict[Any, _SessionBuilder] = {}

    @property
    def backend(self) -> BuilderBackend:
        return self._backend

    # -- admission helpers -----------------------------------------------------

    def input_path(self, job_id: str) -> Path:
        """A private spool file for one build's input tar."""
        self._spool.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._spool, 0o700)
        return self._spool / f"{job_id}.tar"

    def _record(self, session: Any) -> _SessionBuilder:
        """Under the broker lock."""
        record = self._builders.get(session.credentials)
        if record is None:
            record = _SessionBuilder(
                session,
                threading.Semaphore(session.env_grant.build.max_concurrent_builds),
            )
            self._builders[session.credentials] = record
        return record

    def retained(self, session: Any) -> bool:
        """Under the broker lock: the session's builder is not removed yet.

        A leaked image (an rmi conflict) runs nothing and never blocks the
        next session; its journal record keeps the reservation instead.
        """
        record = self._builders.get(session.credentials)
        return (
            record is not None
            and record.lease is not None
            and record.lease.state != "removed"
        )

    # -- journal -----------------------------------------------------------------

    def _commit(self, record: _SessionBuilder) -> Commit:
        def commit(lease: BuilderLease) -> BuilderLease:
            with self._lock:
                try:
                    self._envs._broker.journal.commit_builder(lease)
                except Exception as error:
                    raise self._envs._journal_failure(error) from error
                record.lease = lease
            return lease

        return commit

    def _commit_image(self, build: _Build, **changes: Any) -> SandboxImageLease:
        lease = _revise(build.lease, **changes)
        with self._lock:
            self._envs._commit_image(lease)
            build.lease = lease
        return lease

    # -- the job -------------------------------------------------------------------

    def run(self, job: Job) -> None:
        """The build job's thread body."""
        runner = self._envs.images
        request: BuildRequest = job.detail
        build = _Build(request.lease)
        state, error, result = "failed", None, None
        record: _SessionBuilder | None = None
        slot = False
        try:
            with self._lock:
                record = self._record(job.session)
            while not record.slots.acquire(timeout=0.1):
                if job.cancel.is_set():
                    raise JobFailed("canceled", "the build was canceled")
            slot = True
            runner.running(job)
            lease = self._ensure(job, record)
            result = self._build(job, record, lease, request, build)
            state = "succeeded"
        except Exception as failure:
            state, error = runner.outcome(job, failure)
        finally:
            try:
                request.input_path.unlink(missing_ok=True)
            except OSError as failure:
                LOGGER.warning("build input %s not removed: %s", job.job_id, failure)
            if state != "succeeded" and build.lease.state in ("planned", "loading"):
                # Nothing registered, or a load that failed after its digest
                # was journaled: remove whatever that digest names.
                try:
                    if build.lease.state == "loading":
                        self._discard(job, build)
                    else:
                        self._commit_image(build, state="removed")
                except InfrastructureError:
                    pass  # already failed closed
            runner.finish(job, state, error, result)
            try:
                if record is not None and slot:
                    self._clean(record, request, failed=state != "succeeded")
            finally:
                if slot and record is not None:
                    record.slots.release()

    def _ensure(self, job: Job, record: _SessionBuilder) -> BuilderLease:
        """The session's running builder, made (or remade) on demand."""
        with record.lock:
            if job.cancel.is_set():
                raise JobFailed("canceled", "the build was canceled")
            lease, plan = record.lease, record.plan
            if (
                lease is not None
                and plan is not None
                and lease.state != "removed"
                and not record.broken
                and self._backend.alive(plan, lease)
            ):
                return lease
            if lease is not None and lease.state != "removed":
                # Vanished, killed or drifted: only this builder is replaced.
                self._destroy(record)
            try:
                self._envs._admit_disk("build")
            except SandboxError as refusal:
                raise JobFailed("disk", refusal.message) from None
            session = job.session
            builder_id = "b" + secrets.token_hex(16)
            try:
                plan = self._backend.plan(
                    session.credentials.owner, builder_id, session.env_grant.build
                )
            except SetupError as refusal:
                raise JobFailed("infrastructure", str(refusal)) from None
            lease = plan.lease()
            with self._lock:
                if session.revoked:
                    raise JobFailed("canceled", "the session ended")
                try:
                    self._envs._broker.journal.plan_builder(lease)
                except Exception as failure:
                    raise self._envs._journal_failure(failure) from failure
                record.plan, record.lease, record.broken = plan, lease, False
            try:
                return self._backend.create(
                    plan,
                    lease,
                    self._commit(record),
                    abort=lambda: job.cancel.is_set() or session.revoked,
                )
            except SetupError as failure:
                if job.cancel.is_set() or session.revoked:
                    raise JobFailed("canceled", "the build was canceled") from None
                LOGGER.warning("builder %s failed: %s", builder_id, failure)
                raise JobFailed(
                    "infrastructure", "the builder could not be started"
                ) from None
            except InfrastructureError:
                if not self._envs._broker.recovery_required:
                    self._envs._broker._fail_closed()
                raise

    def _destroy(self, record: _SessionBuilder) -> None:
        """Under ``record.lock``: remove the builder with proof, or fail the
        run closed."""
        lease = record.lease
        if lease is None or lease.state == "removed":
            return
        try:
            self._backend.destroy(lease, self._commit(record))
        except Exception as error:
            self._envs._broker._fail_closed()
            raise InfrastructureError(
                "sandbox builder cleanup unresolved; recovery required"
            ) from error

    def _build(
        self,
        job: Job,
        record: _SessionBuilder,
        lease: BuilderLease,
        request: BuildRequest,
        build: _Build,
    ) -> dict[str, Any]:
        runner = self._envs.images
        self._backend.put_input(lease, request.input_path)
        oom_before = self._backend.oom_kills(lease)
        forced = built_image_labels(job.session.credentials.owner, job.handle)

        def on_stderr(data: bytes) -> None:
            build.tail.extend(data)
            del build.tail[:-8192]
            runner.log(job, data)

        def check() -> None:
            if job.cancel.is_set():
                build.aborted = True
                raise ExecAborted("the build was canceled")

        # A session end cancels its jobs before it kills the builder, so no
        # RUN step starts once that kill may have begun (close_judge).
        if job.cancel.is_set():
            raise JobFailed("canceled", "the build was canceled")
        try:
            exec_id, stream = self._backend.exec_stream(
                lease, build_argv(request, forced), on_stderr=on_stderr, check=check
            )
        except (APIError, OSError, DockerException) as error:
            record.broken = True
            raise JobFailed(
                "infrastructure", f"the build did not start: {error}"
            ) from error
        loaded: dict[str, Any] | None = None
        try:
            try:
                if stream.peek_stdout():
                    loaded = self._load(job, lease, stream, request, build)
                stream.drain()
            except Exception as failure:
                if not build.aborted:
                    raise
                build.failure = failure
        finally:
            stream.close()
        if build.aborted:
            self._interrupt(record, lease, request, exec_id)
            if loaded is not None:
                self._discard(job, build)
            raise JobFailed("canceled", "the build was canceled")
        code = self._backend.exec_result(exec_id, timeout=self._cancel_grace)
        if code != 0 or loaded is None:
            if loaded is not None:
                self._discard(job, build)
            raise self._failure(record, lease, code, oom_before, build)
        return self._finish(job, lease, request, build, loaded)

    def _interrupt(
        self,
        record: _SessionBuilder,
        lease: BuilderLease,
        request: BuildRequest,
        exec_id: str,
    ) -> None:
        """SIGTERM buildctl (it cancels the solve, VERIFIED); a build still
        running after the grace period kills the builder, which the next
        build replaces."""
        pidfile = f"{BUILDKIT_STATE_DIR}/{request.input.directory}/pid"
        try:
            self._backend.run(
                lease,
                ["sh", "-c", f"kill -TERM $(cat {pidfile}) 2>/dev/null; true"],
                timeout=5.0,
            )
            code = self._backend.exec_result(exec_id, timeout=self._cancel_grace)
        except Exception as error:
            LOGGER.warning("build cancel in %s failed: %s", lease.builder_id, error)
            code = None
        if code is None:
            record.broken = True
            try:
                self._backend.kill(lease)
            except Exception as error:
                LOGGER.error("builder %s kill failed: %s", lease.builder_id, error)
                self._envs._broker._fail_closed()

    def _failure(
        self,
        record: _SessionBuilder,
        lease: BuilderLease,
        code: int | None,
        oom_before: int | None,
        build: _Build,
    ) -> JobFailed:
        if build.failure is not None and isinstance(build.failure, JobFailed):
            return build.failure
        attrs = None
        try:
            attrs = self._backend.inspect(lease)
        except InfrastructureError:
            pass
        if attrs is None or not (attrs.get("State") or {}).get("Running"):
            record.broken = True
            return JobFailed("infrastructure", "the builder stopped during the build")
        oom_after = self._backend.oom_kills(lease)
        if oom_before is not None and oom_after is not None and oom_after > oom_before:
            return JobFailed("oom", "a build step ran out of memory")
        tail = build.tail.decode("utf-8", "replace")
        if any(needle in tail.lower() for needle in _NO_SPACE):
            return JobFailed("disk", "the builder state filesystem is full")
        lines = [line for line in tail.splitlines() if "error" in line.lower()]
        message = lines[-1].strip() if lines else f"buildctl exited {code}"
        return JobFailed("dockerfile", message[:1024])

    # -- load --------------------------------------------------------------------

    def _load(
        self,
        job: Job,
        lease: BuilderLease,
        stream: ExecStream,
        request: BuildRequest,
        build: _Build,
    ) -> dict[str, Any]:
        """Sanitize stdout into ``POST /images/load``; the config digest is
        journaled before the daemon receives the end of the stream."""
        runner = self._envs.images
        grant = job.session.env_grant.build
        with self._lock:
            cap = min(
                grant.max_image_mb * MIB,
                self._envs._remaining(job.session, "built_bytes"),
            )
        tag = built_image_tag(job.session.credentials.owner.run_id, job.handle)

        def on_config(image_id: str) -> None:
            self._commit_image(build, state="loading", image_id=image_id, tag=tag)

        sanitizer = ImageArchiveSanitizer(stream, max_bytes=cap, on_config=on_config)
        guard = _FloorGuard(sanitizer, self._envs, self._clock)
        load = self._backend.image_load(self._load_timeout, self._load_grace)

        def failed(error: Exception) -> BaseException:
            if load.final and not load.answered:
                # The daemon may still register the image after this.
                build.unanswered = True
            if guard.error is not None:
                return guard.error
            if isinstance(error, ExecAborted):
                return error
            if load.closed:
                return JobFailed("canceled", "the image load was interrupted")
            if build.unanswered:
                return JobFailed(
                    "infrastructure", f"the image load did not answer: {error}"
                )
            return _load_error(error)

        # A cancel, the job's deadline or the session's end cuts the stream,
        # or bounds the wait for the answer of a whole one, so close_judge
        # ends within DELETE_SEC; a closed load's image is then discarded.
        runner.hold(job, load.close)
        try:
            items = load.post(iter(guard))
        except Exception as error:
            raise failed(error) from None
        finally:
            runner.release(job, load.close)
        loaded = None
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("error") or item.get("errorDetail"):
                LOGGER.warning("image load refused: %s", item)
                raise JobFailed("infrastructure", "the daemon refused the built image")
            match = _LOADED.fullmatch(str(item.get("stream") or ""))
            if match is not None:
                loaded = match.group(1)
        if guard.error is not None:
            raise guard.error
        if sanitizer.result is None:
            raise JobFailed("infrastructure", "the image export did not complete")
        return {"loaded": loaded, "image": sanitizer.result}

    def _finish(
        self,
        job: Job,
        lease: BuilderLease,
        request: BuildRequest,
        build: _Build,
        loaded: dict[str, Any],
    ) -> dict[str, Any]:
        """Cross-check, charge, tag and bind the loaded image."""
        api = self._backend._api
        expected = build.lease.image_id
        try:
            metadata = json.loads(
                self._backend.read_file(
                    lease, f"{BUILDKIT_STATE_DIR}/{request.input.directory}/meta.json"
                )
            )
        except (JobFailed, ValueError):
            metadata = {}
        digest = metadata.get("containerimage.config.digest")
        if not (
            expected is not None
            and loaded["loaded"] == expected
            and loaded["image"].image_id == expected
            and digest == expected
        ):
            self._discard(job, build, also=loaded["loaded"])
            raise JobFailed("infrastructure", "the loaded image differs from the build")
        try:
            attrs = api.inspect_image(expected)
        except (APIError, OSError, DockerException) as error:
            self._discard(job, build)
            raise JobFailed(
                "infrastructure", f"cannot inspect the image: {error}"
            ) from error
        forced = built_image_labels(job.session.credentials.owner, job.handle)
        labels = (attrs.get("Config") or {}).get("Labels") or {}
        owned = {
            key: value for key, value in labels.items() if key.startswith(LABEL_PREFIX)
        }
        refusal = None
        if (
            attrs.get("Id") != expected
            or attrs.get("RepoTags") not in ([], None)
            or attrs.get("Os") != "linux"
            or attrs.get("Architecture") != "amd64"
            or owned != forced
        ):
            refusal = JobFailed("infrastructure", "the loaded image is not the build's")
        else:
            try:
                image_preflight(attrs, expected)
            except SandboxError as error:
                refusal = JobFailed("refused", error.message)
        size = attrs.get("Size") if type(attrs.get("Size")) is int else 0
        if refusal is None:
            with self._lock:
                try:
                    self._envs._charge(job.session, built_bytes=size)
                except SandboxError:
                    refusal = JobFailed(
                        "quota", "the image exceeds max_images_total_mb"
                    )
        if refusal is not None:
            self._discard(job, build)
            raise refusal
        try:
            repository, _, suffix = build.lease.tag.partition(":")
            api.tag(expected, repository, suffix)
            attrs = api.inspect_image(expected)
        except (APIError, OSError, DockerException) as error:
            with self._lock:
                self._envs._refund(job.session, built_bytes=size)
            self._discard(job, build)
            raise JobFailed(
                "infrastructure", f"the image was not tagged: {error}"
            ) from error
        lease_now = self._commit_image(build, state="present", bytes=size)
        image = Image(
            job.handle,
            job.session,
            "built",
            expected,
            build.lease.tag,
            size,
            attrs,
            lease_now,
            fingerprint=request.fingerprint,
        )
        with self._lock:
            if job.session.revoked:
                image = None
            else:
                self._envs.images.bind(image)
        if image is None:
            self.release(
                Image(
                    job.handle,
                    job.session,
                    "built",
                    expected,
                    "",
                    size,
                    attrs,
                    lease_now,
                )
            )
            raise JobFailed("canceled", "the session ended during the build")
        return {"image": image_view(attrs, job.handle)}

    def _discard(self, job: Job, build: _Build, *, also: str | None = None) -> None:
        """Remove a loaded image that is not kept; a conflict leaves it
        journaled as leaked. So does a load the daemon never answered: its
        digest stays journaled and is retried at session end (and swept by
        recovery), because the daemon may register it after this."""
        lease = build.lease
        identities = [lease.image_id] if lease.image_id is not None else []
        if also is not None and also not in identities:
            identities.append(also)
        if build.unanswered and lease.image_id is not None:
            # Once registered, removing it settles the load's outcome.
            build.unanswered = not self._present(lease.image_id)
        removed = all(
            self._remove_image(identity, lease.tag) for identity in identities
        )
        if build.unanswered:
            removed = False
        try:
            self._commit_image(build, state="removed" if removed else "leaked")
        except InfrastructureError:
            return
        if not removed:
            with self._lock:
                record = self._record(job.session)
                record.leaked.append(build.lease)
                if build.unanswered and lease.image_id is not None:
                    record.unanswered.add(lease.image_id)

    def _remove_image(self, image_id: str, tag: str | None) -> bool:
        """Untag, then a non-forced rmi; True once the ID is proven absent."""
        api = self._backend._api
        if tag is not None:
            try:
                attrs = api.inspect_image(tag)
                if attrs.get("Id") == image_id:
                    api.remove_image(tag, force=False)
            except NotFound:
                pass
            except (APIError, OSError, DockerException) as error:
                LOGGER.warning("untag of built image %s failed: %s", tag, error)
        try:
            api.remove_image(image_id, force=False)
        except NotFound:
            pass
        except (APIError, OSError, DockerException) as error:
            LOGGER.warning("built image %s removal failed: %s", image_id, error)
        try:
            api.inspect_image(image_id)
        except NotFound:
            return True
        except (APIError, OSError, DockerException):
            return False
        return False

    def release(self, image: Image) -> bool:
        """Remove a built image the session no longer binds (untag + rmi);
        refunds its bytes, or keeps it journaled as leaked on a conflict."""
        lease = image.lease
        removed = self._remove_image(image.image_id, lease.tag)
        with self._lock:
            self._envs._commit_image(
                lease.model_copy(update={"state": "removed" if removed else "leaked"})
            )
            if removed:
                self._envs._refund(image.session, built_bytes=image.bytes)
            else:
                self._record(image.session).leaked.append(
                    lease.model_copy(update={"state": "leaked"})
                )
        return removed

    def _clean(
        self, record: _SessionBuilder, request: BuildRequest, *, failed: bool
    ) -> None:
        """Drop the build's input and, after a failure, every unused snapshot
        (a failed RUN keeps its disk until pruned, VERIFIED)."""
        lease = record.lease
        if lease is None or lease.state != "running" or record.broken:
            return
        directory = f"{BUILDKIT_STATE_DIR}/{request.input.directory}"
        command = f"rm -rf {directory}"
        if failed:
            command += " && buildctl prune >/dev/null"
        try:
            self._backend.run(lease, ["sh", "-c", command])
        except Exception as error:
            LOGGER.warning("builder %s cleanup failed: %s", lease.builder_id, error)

    # -- sessions --------------------------------------------------------------------

    def kill_session(self, session: Any, deadline: float) -> list[BaseException]:
        """Stage one: SIGKILL the session's builder with proof by ``deadline``
        (monotonic); nothing of its builds executes afterwards."""
        with self._lock:
            record = self._builders.get(session.credentials)
            lease = None if record is None else record.lease
        if record is None or lease is None or lease.state == "removed":
            return []
        record.broken = True
        try:
            self._backend.kill(lease, timeout=max(0.5, deadline - time.monotonic()))
        except Exception as error:
            return [error]
        return []

    def remove_session(self, session: Any, deadline: float) -> list[BaseException]:
        """Stage two: build threads joined, built images removed, then the
        builder removed with proof by ``deadline`` (monotonic)."""
        errors: list[BaseException] = []
        with self._lock:
            jobs = [
                job
                for job in self._envs.images.jobs.values()
                if job.session is session and job.kind == "build"
            ]
            images = [
                image
                for image in self._envs.images.session_images(session)
                if image.kind == "built"
            ]
            for image in images:
                del self._envs.images.images[image.handle]
            record = self._builders.get(session.credentials)
        for job in jobs:
            if job.thread is not None and job.thread is not threading.current_thread():
                job.thread.join(max(0.0, deadline - time.monotonic()))
                if job.thread.is_alive():
                    errors.append(InfrastructureError("sandbox build did not end"))
        for image in images:
            try:
                self.release(image)
            except Exception as error:
                errors.append(error)
        if record is not None:
            errors.extend(self._retry_leaked(record))
            with self._lock:
                # sweep makes and starts a reaper under the lock.
                reaper = record.reaper
            if reaper is not None and reaper is not threading.current_thread():
                reaper.join(max(0.0, deadline - time.monotonic()))
            if not record.lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
                errors.append(InfrastructureError("sandbox builder is busy"))
            else:
                try:
                    self._destroy(record)
                except Exception as error:
                    errors.append(error)
                finally:
                    record.lock.release()
        return errors

    def _retry_leaked(self, record: _SessionBuilder) -> list[BaseException]:
        """Remove the session's leaked images again. One whose load never
        answered and that is absent stays leaked and fails the run closed:
        the daemon may register it later, and only recovery's sweep of the
        run's labels can find it then."""
        still: list[SandboxImageLease] = []
        errors: list[BaseException] = []
        for lease in record.leaked:
            if lease.image_id in record.unanswered and not self._present(
                lease.image_id
            ):
                still.append(lease)
                errors.append(
                    InfrastructureError(
                        f"built image {lease.handle} may still register "
                        "(its load never answered); recovery required"
                    )
                )
            elif lease.image_id is not None and self._remove_image(
                lease.image_id, lease.tag
            ):
                with self._lock:
                    self._envs._commit_image(
                        lease.model_copy(update={"state": "removed"})
                    )
                    self._envs._refund(record.session, built_bytes=lease.bytes)
            else:
                still.append(lease)
        record.leaked = still
        if still:
            LOGGER.warning(
                "built images remain after an rmi conflict or an unanswered load: %s",
                ", ".join(lease.handle for lease in still),
            )
        return errors

    def _present(self, image_id: str) -> bool:
        """False only when the daemon says the image is not there."""
        try:
            self._backend._api.inspect_image(image_id)
        except NotFound:
            return False
        except (APIError, OSError, DockerException):
            pass
        return True

    def sweep(self, dead: Callable[[Any], bool]) -> None:
        """Watchdog turn: a builder outlives neither its session's deadline
        nor its revocation; its removal runs on a background thread."""
        with self._lock:
            doomed = [
                record
                for record in self._builders.values()
                if record.lease is not None
                and record.lease.state != "removed"
                and record.reaper is None
                and dead(record.session)
                and not self._envs.images.pending(record.session)
            ]
            for record in doomed:

                def reap(record: _SessionBuilder = record) -> None:
                    try:
                        with record.lock:
                            self._destroy(record)
                    except Exception as error:
                        LOGGER.error("builder removal failed: %s", error)
                    finally:
                        with self._lock:
                            record.reaper = None

                record.reaper = threading.Thread(
                    target=reap, name="rsi-sandbox-builder-reaper", daemon=True
                )
                record.reaper.start()


class _FloorGuard:
    """The load body: the sanitizer's stream, with the Docker root's free
    space checked at least every second (B5). Its failure is kept, because
    requests may wrap an exception raised inside a request body."""

    def __init__(self, sanitizer: ImageArchiveSanitizer, envs: Any, clock) -> None:
        self._sanitizer = sanitizer
        self._envs = envs
        self._clock = clock
        self.error: JobFailed | None = None

    def __iter__(self):
        checked = self._clock()
        try:
            for chunk in self._sanitizer:
                if self._clock() - checked >= FLOOR_CHECK_SEC:
                    checked = self._clock()
                    try:
                        self._envs._admit_disk("build")
                    except SandboxError as refusal:
                        raise JobFailed("disk", refusal.message) from None
                yield chunk
        except ImageArchiveError as error:
            self.error = JobFailed(
                "quota" if error.kind == "quota" else "infrastructure", error.message
            )
            raise self.error from None
        except JobFailed as error:
            self.error = error
            raise
        except ExecAborted:
            raise
        except Exception as error:
            self.error = JobFailed(
                "infrastructure", f"the image export failed: {error}"
            )
            raise self.error from None


class LoadInterrupted(Exception):
    """The load was closed (a cancel, a deadline or the session's end)."""


class ImageLoad:
    """One ``POST /images/load`` on a connection of its own (B8).

    The docker SDK keeps a request's socket out of reach, so a load the
    daemon reads slowly or answers late could not be stopped and a Judge
    close could outlast DELETE_SEC. Here the body goes out chunked, and each
    send or read blocks at most LOAD_POLL_SEC before it looks for a
    ``close`` (from any thread; only this thread touches the socket).

    Until the stream's last chunk starts out, a close cuts the stream and
    the daemon refuses the truncated archive. From then on (``final``) the
    daemon may have the whole archive: it reads the archive's end before
    the chunked terminator and keeps loading after the client is gone
    (VERIFIED, Docker 29.2.1), so only its answer settles the outcome. A
    close then bounds the wait for it to ``grace``. A load that ends
    ``final`` without ``answered`` leaves the outcome unknown.
    """

    def __init__(
        self,
        connect: Callable[[], socket.socket],
        path: str,
        timeout: float,
        grace: float = LOAD_ANSWER_GRACE_SEC,
    ) -> None:
        self._connect = connect
        self._path = path
        self._timeout = timeout
        self._grace = grace
        self._lock = threading.Lock()
        self._deadline = math.inf
        self.closed = False
        self.final = False
        self.answered = False

    def close(self) -> None:
        with self._lock:
            if not self.closed:
                self.closed = True
                self._deadline = time.monotonic() + self._grace

    def post(self, body: Iterable[bytes]) -> list[Any]:
        """Send ``body`` and return the daemon's JSON answer items. Raises
        LoadInterrupted once closed, also after an answer (``answered``)."""
        self._check(time.monotonic())
        sock = self._connect()
        try:
            sock.settimeout(LOAD_POLL_SEC)
            self._send(
                sock,
                f"POST {self._path}?quiet=1 HTTP/1.1\r\nHost: docker\r\n"
                "Content-Type: application/x-tar\r\n"
                "Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode(),
            )
            chunks = iter(body)
            chunk = next(chunks, None)
            while chunk is not None:
                # One chunk ahead: the last one is known before it goes out.
                following = next(chunks, None)
                if following is None:
                    with self._lock:
                        self.final = True
                if chunk:
                    self._send(sock, b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                chunk = following
            with self._lock:
                self.final = True
            self._send(sock, b"0\r\n\r\n")
            response = self._receive(sock)
        finally:
            sock.close()
        status, text = self._answer(response)
        self.answered = True
        if self.closed:
            raise LoadInterrupted("the image load was closed")
        if status != 200:
            raise OSError(f"the daemon answered {status}: {text[:512]}")
        return list(json_stream([text]))

    def _check(self, idle_since: float) -> None:
        with self._lock:
            if self.closed and not self.final:
                raise LoadInterrupted("the image load was closed")
            deadline = self._deadline
        now = time.monotonic()
        if now >= deadline or now - idle_since >= self._timeout:
            raise TimeoutError("the image load timed out")

    def _send(self, sock: socket.socket, data: bytes) -> None:
        view, idle = memoryview(data), time.monotonic()
        while view:
            self._check(idle)
            try:
                sent = sock.send(view)
            except TimeoutError:
                continue
            view, idle = view[sent:], time.monotonic()

    def _receive(self, sock: socket.socket) -> bytes:
        """The whole response: the daemon closes the connection after it."""
        data, idle = bytearray(), time.monotonic()
        while True:
            self._check(idle)
            try:
                chunk = sock.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                return bytes(data)
            data += chunk
            idle = time.monotonic()
            if len(data) > MAX_LOAD_RESPONSE:
                raise OSError("the image load answer is too long")

    @staticmethod
    def _answer(data: bytes) -> tuple[int, str]:
        """The status and body of a whole HTTP response."""
        response = http.client.HTTPResponse(_Replay(data), method="POST")
        try:
            response.begin()
            body = response.read()
        except http.client.HTTPException as error:
            raise OSError(f"the image load answer is incomplete: {error!r}") from None
        finally:
            response.close()
        return response.status, body[:MAX_LOAD_ANSWER].decode("utf-8", "replace")


class _Replay:
    """A received response, as the socket http.client parses."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def makefile(self, mode: str) -> io.BytesIO:
        return io.BytesIO(self._data)


def engine_socket(api: Any) -> Callable[[], socket.socket]:
    """Connections to the daemon's Unix socket, the only transport loads use."""
    path = getattr(getattr(api, "_custom_adapter", None), "socket_path", None)
    if not isinstance(path, str):
        raise SetupError(
            "sandbox image builds need a unix:// Docker host (image loads use "
            "the daemon's Unix socket)"
        )

    def connect() -> socket.socket:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect(path)
        except BaseException:
            sock.close()
            raise
        return sock

    return connect


def _load_error(error: BaseException) -> JobFailed:
    if isinstance(error, JobFailed):
        return error
    if isinstance(error, ExecAborted):
        raise error
    if isinstance(error, OSError) and error.errno == errno.ENOSPC:
        return JobFailed("disk", "the host has no space for the image")
    return JobFailed("infrastructure", f"the image load failed: {error}")


def build_deadline(
    session: Any, timeout: float, grant: EnvBuildGrant, now: float
) -> float:
    """min(timeout_sec, max_build_sec, session remaining) (B4)."""
    deadlines = [now + timeout, now + grant.max_build_sec]
    if session.deadline is not None:
        deadlines.append(session.deadline)
    return min(deadlines) if deadlines else math.inf


__all__ = [
    "BUILDKITD_TOML",
    "BUILDKIT_STATE_DIR",
    "BUILDKIT_TRACE_SOCKET",
    "BuildRequest",
    "BuildService",
    "BuilderBackend",
    "BuilderPlan",
    "ExecStream",
    "build_argv",
    "build_deadline",
    "builder_labels",
    "built_image_labels",
]
