"""Brokered env services: one fixed low-level template, exact attestation.

Service containers are created only from an Engine create body the broker
writes itself (``create_container_from_config``); the high-level SDK would
silently drop endpoint aliases. The plan mirrors the daemon's create-time
merge of image and container config (Env, Labels, Entrypoint/Cmd, User,
WorkingDir, StopSignal, Healthcheck) and sends the merged values, so the
expected configuration is exact and every inspected field can be compared
before and after start.

Order (S5, S7): firewall rule, bridge, volumes, containers; teardown runs in
reverse. Each Docker mutation is preceded by a journal commit of what it may
create, and an object leaves the journal only after inspection proves it
absent. A failure whose rollback is proven is a SetupError; anything that
cannot be proven is an InfrastructureError marked ``recovery_required``, and
so is every teardown (terminate, destroy) failure that is not the journal's.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import posixpath
import time
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.sandbox_archive import ArchiveTarget
from rsi_harness.runtime.sandbox_contracts import (
    IMAGE_ID,
    SandboxError,
    SandboxOwner,
    below,
    swap_mb,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    ENV_NETWORK_ROLE,
    LABEL_PREFIX,
    MAX_ENV_VOLUME_LEASES,
    EnvDependency,
    EnvReason,
    EnvSpec,
    SandboxEnvLease,
    SandboxEnvServiceLease,
    SandboxEnvVolumeLease,
    env_container_labels,
    env_container_name,
    env_spec_digest,
    env_volume_labels,
    env_volume_name,
    sandbox_object_labels,
)
from rsi_harness.runtime.sandbox_exec import ExecTarget
from rsi_harness.runtime.sandbox_network import (
    SandboxNetworkBackend,
    SandboxNetworkPlan,
    env_lease_network_plan,
    plan_env_network,
)

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
NANOSECONDS = 1_000_000_000
# Same as the Work/Judge parents (docker.py); a spec can only drop more.
BASE_CAP_DROP = ("CAP_NET_RAW",)
APPARMOR = "apparmor=docker-default"
NO_NEW_PRIVILEGES = "no-new-privileges:true"
# Bounded diagnostics, and never the host journal (the daemon default).
LOG_CONFIG = {"Type": "json-file", "Config": {"max-size": "1m", "max-file": "1"}}
# Rules are not persistent: a daemon-restarted service would lose its firewall.
RESTART_POLICY = {"Name": "no"}
TMPFS_OPTIONS = "rw,nosuid,nodev,exec,size={size}"
POLL_INTERVAL_SEC = 0.25
STOP_PROOF_SEC = 10.0
HEALTH_TAIL_BYTES = 1024
LOG_TAIL_BYTES = 4096
MAX_IMAGE_ENV = 256
MAX_IMAGE_ENV_BYTES = 64 * 1024
_RESERVED_ROOTS = ("/proc", "/sys", "/dev", "/run/rsi-harness")
_HEALTH_TIMINGS = ("Interval", "Timeout", "StartPeriod", "StartInterval", "Retries")
_EMPTY_HOST_FIELDS = (
    "CapAdd",
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
    "Isolation",
    "VolumeDriver",
    "OomKillDisable",
    "AutoRemove",
)
_EXACT_HOST_FIELDS = (
    "Runtime",
    "Privileged",
    "CapDrop",
    "SecurityOpt",
    "Init",
    "IpcMode",
    "CgroupnsMode",
    "NanoCpus",
    "Memory",
    "MemorySwap",
    "PidsLimit",
    "Ulimits",
    "ShmSize",
    "NetworkMode",
    "LogConfig",
    "ReadonlyRootfs",
    "PublishAllPorts",
)

Commit = Callable[[SandboxEnvLease], SandboxEnvLease]
ServiceState = Literal["created", "running", "paused", "exited", "removed"]
RECOVERY_REQUIRED = "recovery_required"
# States an env may be failed from; later ones already belong to teardown.
_FAILABLE = ("created", "starting", "ready", "failed", "paused")
# Fixed start-failure details: Engine explanations can name host paths.
_START_FAILURES = (
    ("executable file not found", "entrypoint executable not found in the image"),
    ("no such file or directory", "entrypoint not found in the image"),
    ("permission denied", "entrypoint is not executable"),
    ("unable to find user", "user not found in the image"),
    ("unable to find group", "group not found in the image"),
)


def _recovery(error: BaseException) -> bool:
    return isinstance(error, InfrastructureError) and str(error).startswith(
        RECOVERY_REQUIRED
    )


def _canonical(value: Any) -> str:
    # JSON equality keeps bool/int distinct (True != 1) and dict order free.
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _revise(model: Any, **changes: Any) -> Any:
    """A validated copy: every journal record keeps its invariants."""
    return type(model)(**{**dict(model), **changes})


def _answered(error: Exception) -> bool:
    # Only an Engine HTTP answer ends a create; after a timeout dockerd may
    # still finish it.
    return isinstance(error, APIError) and error.status_code is not None


def _clean_dir(path: str | None) -> str:
    """dockerd's create-time ``filepath.Clean`` of an image's WorkingDir: the
    swebench images' ``/testbed/`` is created, and inspected, as
    ``/testbed``."""
    if not path:
        return ""
    cleaned = posixpath.normpath(path)
    # POSIX keeps a leading "//"; Go's Clean does not.
    return "/" + cleaned.lstrip("/") if cleaned.startswith("//") else cleaned


def _mismatch(name: str, field_name: str) -> InfrastructureError:
    return InfrastructureError(
        f"sandbox env service {name} configuration mismatch: {field_name}"
    )


def _start_failure(error: Exception) -> str:
    explanation = str(getattr(error, "explanation", None) or error).lower()
    for needle, detail in _START_FAILURES:
        if needle in explanation:
            return detail
    return "the daemon refused to start the service"


# -- plan ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnvVolumePlan:
    idx: int
    logical: str
    name: str
    labels: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class EnvServicePlan:
    """One service's exact create body and the attestation derived from it."""

    idx: int
    name: str
    container_name: str
    # The service's image handle (i<32hex>) and the ID it resolved to.
    image: str
    image_id: str
    config: Mapping[str, Any]
    network_mode: str
    # Endpoint aliases on the env bridge; None with network_mode none.
    aliases: tuple[str, ...] | None
    has_healthcheck: bool
    depends_on: Mapping[str, EnvDependency]
    # (volume name, target, read-write) exactly as inspect reports Mounts.
    mounts: tuple[tuple[str, str, bool], ...]
    implicit_volumes: tuple[str, ...]
    tmpfs: tuple[str, ...]

    def body(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.config))

    @property
    def host_config(self) -> Mapping[str, Any]:
        return self.config["HostConfig"]


@dataclass(frozen=True, slots=True)
class EnvPlan:
    owner: SandboxOwner
    env_id: str
    spec_sha256: str
    network_mode: Literal["public", "none", "allowlist"]
    network: SandboxNetworkPlan | None
    services: tuple[EnvServicePlan, ...]
    volumes: tuple[EnvVolumePlan, ...]
    order: tuple[str, ...]
    cpus_milli: int
    memory_mb: int
    # Swap of every service together (floor(memory_mb * swap_ratio) each).
    swap_mb: int
    disk_mb: int

    def service(self, name: str) -> EnvServicePlan:
        for service in self.services:
            if service.name == name:
                return service
        raise SandboxError("invalid", "service", f"unknown service {name}")

    @property
    def completion_targets(self) -> frozenset[str]:
        """Services allowed to exit(0): targets of completed_successfully."""
        return frozenset(
            target
            for service in self.services
            for target, dependency in service.depends_on.items()
            if dependency.condition == "completed_successfully"
        )

    @property
    def optional(self) -> frozenset[str]:
        """Services only ever depended on with ``required: false``."""
        incoming: dict[str, list[bool]] = {}
        for service in self.services:
            for target, dependency in service.depends_on.items():
                incoming.setdefault(target, []).append(dependency.required)
        return frozenset(
            target for target, required in incoming.items() if not any(required)
        )

    def lease(self, *, created_at: float, expires_at: float) -> SandboxEnvLease:
        """The journal record to commit before any Docker call."""
        return SandboxEnvLease(
            owner=self.owner,
            env_id=self.env_id,
            spec_sha256=self.spec_sha256,
            state="planned",
            created_at=created_at,
            expires_at=expires_at,
            network_mode=self.network_mode,
            network_name=None if self.network is None else self.network.name,
            rule_id=None if self.network is None else self.network.rule_id,
            cpus_milli=self.cpus_milli,
            memory_mb=self.memory_mb,
            disk_mb=self.disk_mb,
            services=tuple(
                SandboxEnvServiceLease(
                    idx=service.idx,
                    name=service.name,
                    planned_name=service.container_name,
                    image=service.image,
                    image_id=service.image_id,
                )
                for service in self.services
            ),
            volumes=tuple(
                SandboxEnvVolumeLease(
                    idx=volume.idx, logical=volume.logical, planned_name=volume.name
                )
                for volume in self.volumes
            ),
            pending_mutation=True,
        )


def image_preflight(attrs: Mapping[str, Any], image_id: str) -> None:
    """A service image must be the exact local ID with bounded metadata."""
    config = attrs.get("Config") or {}
    environment = config.get("Env") or []
    if attrs.get("Id") != image_id or not IMAGE_ID.fullmatch(image_id):
        raise SandboxError("invalid", "image", "image did not resolve to its exact ID")
    if attrs.get("Os") != "linux":
        raise SandboxError("unsupported", "image", "service images must be linux")
    if (
        len(environment) > MAX_IMAGE_ENV
        or sum(len(item.encode()) for item in environment) > MAX_IMAGE_ENV_BYTES
    ):
        raise SandboxError(
            "quota", "image", "image environment exceeds bounded metadata limits"
        )


def _service_env(image_env: Sequence[str], overlay: Mapping[str, str]) -> list[str]:
    """Image Env, then the spec overlay; NVIDIA_* blanked, GPUs forced off.

    The daemon re-appends every image key a create omits, so an image's
    NVIDIA_* keys are blanked in place rather than dropped; under runc they
    are inert either way.
    """
    merged: dict[str, str | None] = {}
    for item in image_env:
        key, separator, value = item.partition("=")
        merged[key] = value if separator else None
    merged.update(overlay)
    for key in merged:
        if key.upper().startswith("NVIDIA_"):
            merged[key] = ""
    merged["NVIDIA_VISIBLE_DEVICES"] = "void"
    return [key if value is None else f"{key}={value}" for key, value in merged.items()]


def _healthcheck_body(check: Mapping[str, Any] | None) -> dict[str, Any] | None:
    # Inspect omits zero timings; compare and send the same normalized form.
    if not check or not check.get("Test"):
        return None
    return {key: value for key, value in check.items() if value}


def _has_check(check: Mapping[str, Any] | None) -> bool:
    return bool(check and check.get("Test") and check["Test"][0] != "NONE")


def _effective_healthcheck(
    name: str, requested: Any, image: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """The daemon's merge: unset fields of the container check come from the
    image check (daemon/create.go merge)."""
    if requested is None:
        return _healthcheck_body(image)
    if requested == "image":
        if not _has_check(image):
            raise SandboxError(
                "invalid",
                f"spec.services.{name}.healthcheck",
                "image declares no healthcheck",
            )
        return _healthcheck_body(image)
    if requested == "none":
        user: dict[str, Any] = {"Test": ["NONE"]}
    else:
        user = {
            "Test": list(requested.test),
            "Interval": round(requested.interval_sec * NANOSECONDS),
            "Timeout": round(requested.timeout_sec * NANOSECONDS),
            "StartPeriod": round(requested.start_period_sec * NANOSECONDS),
            "StartInterval": round(requested.start_interval_sec * NANOSECONDS),
            "Retries": requested.retries,
        }
    if image:
        for key in _HEALTH_TIMINGS:
            if not user.get(key):
                user[key] = image.get(key, 0)
    return _healthcheck_body(user)


def _start_order(spec: EnvSpec) -> tuple[str, ...]:
    remaining = {
        name: set(service.depends_on) for name, service in spec.services.items()
    }
    order: list[str] = []
    while remaining:
        ready = sorted(name for name, needs in remaining.items() if not needs)
        if not ready:  # EnvSpec already refuses cycles
            raise SandboxError("invalid", "spec.services", "depends_on cycle")
        for name in ready:
            order.append(name)
            del remaining[name]
        for needs in remaining.values():
            needs.difference_update(ready)
    return tuple(order)


def _image_volumes(name: str, config: Mapping[str, Any]) -> tuple[str, ...]:
    paths = set()
    for raw in config.get("Volumes") or {}:
        path = posixpath.normpath(raw) if isinstance(raw, str) else ""
        if (
            not path.startswith("/")
            or path.startswith("//")
            or path == "/"
            or any(below(path, root) for root in _RESERVED_ROOTS)
        ):
            raise SandboxError(
                "unsupported",
                f"spec.services.{name}.image",
                "image declares a VOLUME at a reserved or invalid path",
            )
        paths.add(path)
    return tuple(sorted(paths))


def _image_labels(
    name: str, config: Mapping[str, Any], owned: Mapping[str, str]
) -> dict[str, str]:
    """Image labels overridden by the container's own exact set.

    Keys the container sets itself (owner keys, role, sandbox-env,
    sandbox-service, sandbox-image) are overridden, so an M8-built image
    labelled for its session is accepted; any other rsi-harness key is not.
    """
    labels = dict(config.get("Labels") or {})
    foreign = sorted(
        key for key in labels if key.startswith(LABEL_PREFIX) and key not in owned
    )
    if foreign:
        # The daemon copies image labels onto the container; a foreign
        # rsi-harness key would make it look like another sandbox object.
        raise SandboxError(
            "unsupported",
            f"spec.services.{name}.image",
            f"image carries reserved label {foreign[0]}",
        )
    return {**labels, **owned}


def plan_env(
    owner: SandboxOwner,
    env_id: str,
    spec: EnvSpec,
    images: Mapping[str, Mapping[str, Any]],
    *,
    default_pids: int,
    swap_ratio: float,
    no_new_privileges: bool = True,
    private_cidrs: Sequence[str] = (),
) -> EnvPlan:
    """Deterministic names, labels, create bodies and start order of an env.

    ``images`` maps each service's image handle to the preflighted inspect
    attributes of the image it resolves to in the session.
    ``swap_ratio`` is the grant's: each service may swap
    floor(memory_mb * swap_ratio) MiB on top of its memory.
    ``private_cidrs`` are the allowlist grant's (allowlist envs only).
    """
    network = plan_env_network(owner, env_id, spec, private_cidrs=private_cidrs)
    names = sorted(spec.services)
    volumes = [
        EnvVolumePlan(
            idx=index,
            logical=logical,
            name=env_volume_name(env_id, index),
            labels=env_volume_labels(owner, env_id),
        )
        for index, logical in enumerate(sorted(spec.volumes))
    ]
    by_logical = {volume.logical: volume for volume in volumes}
    image_configs = {}
    implicit: dict[str, dict[str, EnvVolumePlan]] = {}
    for index, name in enumerate(names):
        service = spec.services[name]
        attrs = images.get(service.image)
        if attrs is None:
            raise SandboxError(
                "permission", f"spec.services.{name}.image", "unknown image handle"
            )
        image_preflight(attrs, attrs.get("Id"))
        config = attrs.get("Config") or {}
        image_configs[name] = (attrs["Id"], config)
        covered = {mount.target for mount in service.mounts} | set(service.tmpfs)
        implicit[name] = {}
        # Exactly as the daemon: only an identical destination covers a VOLUME.
        for path in _image_volumes(name, config):
            if path in covered:
                continue
            volume = EnvVolumePlan(
                idx=len(volumes),
                logical=f"implicit:{index}:{hashlib.sha256(path.encode()).hexdigest()[:16]}",
                name=env_volume_name(env_id, len(volumes)),
                labels=env_volume_labels(owner, env_id),
            )
            volumes.append(volume)
            implicit[name][path] = volume
    if len(volumes) > MAX_ENV_VOLUME_LEASES:
        raise SandboxError(
            "quota",
            "spec.volumes",
            f"env needs more than {MAX_ENV_VOLUME_LEASES} volumes",
        )

    security = [APPARMOR] if not no_new_privileges else [NO_NEW_PRIVILEGES, APPARMOR]
    services = []
    for index, name in enumerate(names):
        service = spec.services[name]
        image_id, config = image_configs[name]
        labels = _image_labels(
            name, config, env_container_labels(owner, env_id, name, service.image)
        )
        if service.entrypoint is not None:
            entrypoint = list(service.entrypoint)
        else:
            entrypoint = config.get("Entrypoint")
        if service.command:
            command = list(service.command)
        elif service.entrypoint:
            command = None  # Docker: overriding the entrypoint clears Cmd
        else:
            command = config.get("Cmd")
        healthcheck = _effective_healthcheck(
            name, service.healthcheck, config.get("Healthcheck")
        )
        stop_signal = service.stop_signal or config.get("StopSignal") or None
        on_bridge = network is not None and service.network == "env"
        network_mode = network.name if on_bridge else "none"
        aliases = (name, *service.aliases) if on_bridge else None
        mounts = []
        mount_set = []
        for mount in service.mounts:
            volume = by_logical[mount.volume]
            spec_mount: dict[str, Any] = {
                "Type": "volume",
                "Source": volume.name,
                "Target": mount.target,
            }
            if mount.read_only:
                spec_mount["ReadOnly"] = True
            if spec.volumes[mount.volume].seeded:
                spec_mount["VolumeOptions"] = {"NoCopy": True}
            mounts.append(spec_mount)
            mount_set.append((volume.name, mount.target, not mount.read_only))
        for path, volume in implicit[name].items():
            mounts.append({"Type": "volume", "Source": volume.name, "Target": path})
            mount_set.append((volume.name, path, True))
        cap_drop = [
            *BASE_CAP_DROP,
            *sorted(set(service.cap_drop) - set(BASE_CAP_DROP)),
        ]
        swap = swap_mb(service.memory_mb, swap_ratio)
        host = {
            "Runtime": "runc",
            "Privileged": False,
            "CapDrop": cap_drop,
            "SecurityOpt": list(security),
            "Init": True,
            "IpcMode": "private",
            "CgroupnsMode": "private",
            "NanoCpus": round(service.cpus * 100) * (NANOSECONDS // 100),
            "Memory": service.memory_mb * MIB,
            # Memory plus swap; MemorySwappiness stays the kernel default.
            "MemorySwap": (service.memory_mb + swap) * MIB,
            "PidsLimit": service.pids if service.pids is not None else default_pids,
            "Ulimits": [
                {"Name": "nofile", "Soft": service.nofile, "Hard": service.nofile}
            ],
            "ShmSize": service.shm_mb * MIB,
            "Tmpfs": {
                path: TMPFS_OPTIONS.format(size=size * MIB)
                for path, size in service.tmpfs.items()
            },
            "Mounts": mounts,
            "NetworkMode": network_mode,
            "ExtraHosts": [f"{host}:{address}" for host, address in service.extra_hosts]
            or None,
            "LogConfig": LOG_CONFIG,
            "RestartPolicy": RESTART_POLICY,
            "ReadonlyRootfs": service.read_only,
            "GroupAdd": list(service.group_add) or None,
            "AutoRemove": False,
            "OomKillDisable": False,
            "PublishAllPorts": False,
        }
        body: dict[str, Any] = {
            "Image": image_id,
            "Entrypoint": entrypoint,
            "Cmd": command,
            "Env": _service_env(config.get("Env") or [], service.env),
            "User": service.user or config.get("User") or "",
            "WorkingDir": service.working_dir or _clean_dir(config.get("WorkingDir")),
            "Labels": labels,
            "Tty": service.tty,
            "OpenStdin": False,
            "AttachStdin": False,
            "AttachStdout": False,
            "AttachStderr": False,
            "StopTimeout": service.stop_grace_sec,
            "HostConfig": host,
        }
        if service.hostname is not None:
            body["Hostname"] = service.hostname
        if stop_signal is not None:
            body["StopSignal"] = stop_signal
        if healthcheck is not None:
            body["Healthcheck"] = healthcheck
        if aliases is not None:
            body["NetworkingConfig"] = {
                "EndpointsConfig": {network_mode: {"Aliases": list(aliases)}}
            }
        services.append(
            EnvServicePlan(
                idx=index,
                name=name,
                container_name=env_container_name(env_id, index),
                image=service.image,
                image_id=image_id,
                config=json.loads(json.dumps(body)),
                network_mode=network_mode,
                aliases=aliases,
                has_healthcheck=_has_check(healthcheck),
                depends_on=dict(service.depends_on),
                mounts=tuple(mount_set),
                implicit_volumes=tuple(implicit[name]),
                tmpfs=tuple(sorted(service.tmpfs)),
            )
        )
    return EnvPlan(
        owner=owner,
        env_id=env_id,
        spec_sha256=env_spec_digest(spec),
        network_mode=spec.network,
        network=network,
        services=tuple(services),
        volumes=tuple(volumes),
        order=_start_order(spec),
        cpus_milli=sum(
            round(service.cpus * 1000) for service in spec.services.values()
        ),
        memory_mb=sum(service.memory_mb for service in spec.services.values()),
        swap_mb=sum(
            swap_mb(service.memory_mb, swap_ratio) for service in spec.services.values()
        ),
        disk_mb=spec.disk_mb,
    )


# -- paused-safe termination ------------------------------------------------------

Settled = Callable[[], bool]


def _settled() -> bool:
    return True


class PausedKiller(Protocol):
    """Kill every task of a paused (frozen) container.

    ``kill`` only signals; it returns an extra proof that every task is
    gone (or None), which the caller polls with Docker's own state against
    one deadline, so a whole env can be signalled before any proof waits.
    """

    def kill(self, container_id: str, attrs: Mapping[str, Any]) -> Settled | None: ...


class DockerPausedKiller:
    """Non-root: ``docker kill`` on a paused container (VERIFIED on 29.2.1).

    Moby queues SIGKILL and then thaws the task, so zero execution after the
    freeze is not proven; only the root cgroup.kill path proves it.
    """

    def __init__(self, api: Any) -> None:
        self._api = api

    def kill(self, container_id: str, attrs: Mapping[str, Any]) -> None:
        self._api.kill(container_id, signal="SIGKILL")


class CgroupPausedKiller:
    """Root: ``cgroup.kill`` on the frozen scope; no task runs again.

    The scope is read from the frozen init's ``/proc/<pid>/cgroup`` and must
    name the container; a frozen PID cannot exit and be reused meanwhile.
    The returned proof is the scope's ``cgroup.events`` reporting
    ``populated 0`` (or the scope being gone).
    """

    def __init__(
        self,
        *,
        cgroup_root: Path = Path("/sys/fs/cgroup"),
        proc_root: Path = Path("/proc"),
    ) -> None:
        self._cgroup_root = Path(cgroup_root)
        self._proc_root = Path(proc_root)

    def scope(self, container_id: str, attrs: Mapping[str, Any]) -> Path:
        pid = (attrs.get("State") or {}).get("Pid")
        if type(pid) is not int or pid <= 0:
            raise InfrastructureError("paused sandbox container has no init PID")
        try:
            lines = (self._proc_root / str(pid) / "cgroup").read_text().splitlines()
        except OSError as error:
            raise InfrastructureError(
                f"cannot read paused sandbox cgroup: {error}"
            ) from error
        unified = [line[3:] for line in lines if line.startswith("0::")]
        if len(unified) != 1:
            raise InfrastructureError("paused sandbox is not in one cgroup v2 scope")
        path = unified[0]
        if (
            not path.startswith("/")
            or posixpath.normpath(path) != path
            or container_id not in path.rsplit("/", 1)[-1]
        ):
            raise InfrastructureError(
                "paused sandbox cgroup does not name its container"
            )
        return self._cgroup_root / path.lstrip("/")

    def kill(self, container_id: str, attrs: Mapping[str, Any]) -> Settled:
        scope = self.scope(container_id, attrs)
        try:
            descriptor = os.open(
                scope / "cgroup.kill", os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            try:
                os.write(descriptor, b"1")
            finally:
                os.close(descriptor)
        except OSError as error:
            raise InfrastructureError(
                f"cgroup.kill of paused sandbox failed: {error}"
            ) from error
        return lambda: self.emptied(scope)

    @staticmethod
    def emptied(scope: Path) -> bool:
        try:
            lines = (scope / "cgroup.events").read_text().splitlines()
        except FileNotFoundError:
            return True
        except OSError as error:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: cannot prove paused sandbox cgroup empty: "
                f"{error}"
            ) from error
        events = dict(line.split(" ", 1) for line in lines if " " in line)
        return events.get("populated") == "0"


def default_paused_killer(api: Any, *, euid: int | None = None) -> PausedKiller:
    """cgroup.kill as root (production); docker kill otherwise (dev, tests)."""
    if (os.geteuid() if euid is None else euid) == 0:
        return CgroupPausedKiller()
    return DockerPausedKiller(api)


def kill_container(
    api: Any,
    container_id: str,
    attrs: Mapping[str, Any],
    *,
    paused_killer: PausedKiller,
) -> Settled:
    """Send SIGKILL (cgroup.kill when paused); return the proof to poll.

    Any failure other than "not running" (409) is recovery_required.
    """
    state = attrs.get("State") or {}
    try:
        if state.get("Paused"):
            return paused_killer.kill(container_id, attrs) or _settled
        if state.get("Running"):
            api.kill(container_id, signal="SIGKILL")
    except Exception as error:
        # 409: it stopped by itself between inspect and kill.
        if isinstance(error, APIError) and error.status_code == 409:
            return _settled
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: sandbox container kill failed: {error}"
        ) from error
    return _settled


def await_stopped(
    inspect: Callable[[], Mapping[str, Any] | None],
    settled: Settled,
    *,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any] | None:
    """Poll until Docker reports the container not running and ``settled``."""
    while True:
        attrs = inspect()
        stopped = attrs is None or not (attrs.get("State") or {}).get("Running")
        if stopped and settled():
            return attrs
        if clock() >= deadline:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: sandbox container termination could not "
                "be proven"
            )
        sleep(0.05)


def terminate_container(
    api: Any,
    container_id: str,
    *,
    inspect: Callable[[], Mapping[str, Any] | None],
    paused_killer: PausedKiller,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = STOP_PROOF_SEC,
) -> Mapping[str, Any] | None:
    """SIGKILL a container, paused or not, and prove it is no longer running.

    ``inspect`` returns identity-attested attributes or None once the
    container is gone. v1 children use it through the opt-in
    ``SandboxDockerBackend(paused_killer=...)``. Returns the final attributes.
    """
    attrs = inspect()
    if attrs is None:
        return None
    state = attrs.get("State") or {}
    if not state.get("Paused") and not state.get("Running"):
        return attrs
    settled = kill_container(api, container_id, attrs, paused_killer=paused_killer)
    return await_stopped(
        inspect, settled, deadline=clock() + timeout, clock=clock, sleep=sleep
    )


# -- backend -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    state: ServiceState
    health: Literal["starting", "healthy", "unhealthy"] | None = None
    exit_code: int | None = None
    started_at: str | None = None


@dataclass(frozen=True, slots=True)
class EnvStartResult:
    lease: SandboxEnvLease
    state: Literal["ready", "failed"]
    reason: EnvReason | None = None
    service: str | None = None
    detail: str | None = None


@dataclass(slots=True)
class _Start:
    started: set[str] = field(default_factory=set)
    # Optional services that failed (or could not start); they are ignored.
    abandoned: set[str] = field(default_factory=set)
    observed: dict[str, ServiceStatus] = field(default_factory=dict)


def _status(attrs: Mapping[str, Any] | None) -> ServiceStatus:
    if attrs is None:
        return ServiceStatus("removed")
    state = attrs.get("State") or {}
    status = state.get("Status")
    started_at = state.get("StartedAt")
    if not isinstance(started_at, str) or started_at.startswith("0001-"):
        started_at = None
    if state.get("Paused"):
        kind: ServiceState = "paused"
    elif state.get("Running"):
        kind = "running"
    elif status == "created":
        kind = "created"
    else:
        kind = "exited"
    health = None
    if kind in ("running", "paused"):
        value = (state.get("Health") or {}).get("Status")
        if value in ("starting", "healthy", "unhealthy"):
            health = value
    exit_code = state.get("ExitCode") if kind == "exited" else None
    return ServiceStatus(
        kind,
        health,
        exit_code if type(exit_code) is int else None,
        started_at,
    )


def _tail(value: str | bytes, limit: int) -> str:
    data = value.encode("utf-8", "replace") if isinstance(value, str) else value
    return data[-limit:].decode("utf-8", "replace")


class SandboxEnvDockerBackend:
    """Create, attest, start, pause, stop and remove brokered envs.

    ``commit`` (supplied by the broker) durably writes an updated env lease
    and returns it; a journal failure propagates and no further mutation
    runs. Removal is idempotent and works from the lease alone.
    """

    def __init__(
        self,
        client: Any,
        network: SandboxNetworkBackend,
        *,
        no_new_privileges: bool = True,
        paused_killer: PausedKiller | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_interval: float = POLL_INTERVAL_SEC,
        stop_proof_sec: float = STOP_PROOF_SEC,
    ) -> None:
        self._client = client
        self._api = client.api
        self._network = network
        self._no_new_privileges = no_new_privileges
        self._paused_killer = paused_killer or default_paused_killer(client.api)
        self._clock = clock
        self._sleep = sleep
        self._poll = poll_interval
        self._stop_proof = stop_proof_sec

    # -- planning --------------------------------------------------------------

    def inspect_image(self, image_id: str) -> dict[str, Any]:
        try:
            attrs = self._api.inspect_image(image_id)
        except NotFound:
            raise SandboxError("invalid", "image", "image is not present") from None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot inspect sandbox env image: {error}"
            ) from error
        image_preflight(attrs, image_id)
        return attrs

    def plan(
        self,
        owner: SandboxOwner,
        env_id: str,
        spec: EnvSpec,
        images: Mapping[str, Mapping[str, Any]],
        *,
        default_pids: int,
        swap_ratio: float,
        private_cidrs: Sequence[str] = (),
    ) -> EnvPlan:
        return plan_env(
            owner,
            env_id,
            spec,
            images,
            default_pids=default_pids,
            swap_ratio=swap_ratio,
            no_new_privileges=self._no_new_privileges,
            private_cidrs=private_cidrs,
        )

    def allowlist_notes(self, plan: EnvPlan) -> tuple[str, ...]:
        """Allowlist entries of a created env that reach nothing."""
        if plan.network is None:
            return ()
        return self._network.allowlist_notes(plan.network)

    def refresh_network(self, plan: EnvPlan) -> bool:
        """Resolve an allowlist env's hostnames again (no Docker call);
        True when its firewall accepts changed."""
        if plan.network is None:
            return False
        return self._network.refresh(plan.network)

    @staticmethod
    def _require_plan(plan: EnvPlan, lease: SandboxEnvLease) -> None:
        expected = plan.lease(created_at=lease.created_at, expires_at=lease.expires_at)
        if (
            lease.owner != plan.owner
            or lease.env_id != plan.env_id
            or lease.spec_sha256 != plan.spec_sha256
            or lease.network_name != expected.network_name
            or [
                (item.name, item.planned_name, item.image, item.image_id)
                for item in lease.services
            ]
            != [
                (item.name, item.planned_name, item.image, item.image_id)
                for item in expected.services
            ]
            or [(item.logical, item.planned_name) for item in lease.volumes]
            != [(item.logical, item.planned_name) for item in expected.volumes]
        ):
            raise InfrastructureError("sandbox env plan differs from its journal")

    # -- identity --------------------------------------------------------------

    @staticmethod
    def _record(lease: SandboxEnvLease, service: str) -> SandboxEnvServiceLease:
        for record in lease.services:
            if record.name == service:
                return record
        raise SandboxError("invalid", "service", f"unknown service {service}")

    def _owned(
        self,
        lease: SandboxEnvLease,
        record: SandboxEnvServiceLease,
        *,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        reference = record.container_id or record.planned_name
        try:
            attrs = self._api.inspect_container(reference)
        except NotFound:
            if missing_ok:
                return None
            raise InfrastructureError(
                f"sandbox env service {record.name} is absent"
            ) from None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot inspect sandbox env service {record.name}: {error}"
            ) from error
        labels = (attrs.get("Config") or {}).get("Labels") or {}
        owned = {
            key: value for key, value in labels.items() if key.startswith(LABEL_PREFIX)
        }
        identity = attrs.get("Id")
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or (record.container_id is not None and identity != record.container_id)
            or attrs.get("Name") != "/" + record.planned_name
            or attrs.get("Image") != record.image_id
            or owned
            != env_container_labels(
                lease.owner, lease.env_id, record.name, record.image
            )
        ):
            raise InfrastructureError(
                f"recovery_required: sandbox env service {record.name} is not "
                "owned by its planned identity"
            )
        return attrs

    def _require(
        self, lease: SandboxEnvLease, record: SandboxEnvServiceLease
    ) -> dict[str, Any]:
        attrs = self._owned(lease, record)
        assert attrs is not None
        return attrs

    # -- attestation -------------------------------------------------------------

    def attest(self, plan: EnvPlan, lease: SandboxEnvLease) -> None:
        """Every object of a created env against the plan, exactly."""
        self._require_plan(plan, lease)
        for volume in plan.volumes:
            self._attest_volume(volume)
        self._attest_network(plan, lease)
        for service in plan.services:
            attrs = self._require(lease, lease.services[service.idx])
            self.attest_service(service, attrs, network_id=lease.network_id)

    def _attest_network(self, plan: EnvPlan, lease: SandboxEnvLease) -> None:
        if plan.network is None:
            return
        if lease.network_id is None:
            raise InfrastructureError("sandbox env bridge identity is not recorded")
        containers = [
            record.container_id for record in lease.services if record.container_id
        ]
        self._network.attest(plan.network, lease.network_id, containers=containers)

    def _attest_volume(self, volume: EnvVolumePlan) -> None:
        try:
            attrs = self._api.inspect_volume(volume.name)
        except NotFound:
            raise InfrastructureError(
                f"sandbox env volume {volume.name} is absent"
            ) from None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot inspect sandbox env volume {volume.name}: {error}"
            ) from error
        # A local volume with driver options can bind any host path.
        for key, matches in (
            ("Name", attrs.get("Name") == volume.name),
            ("Driver", attrs.get("Driver") == "local"),
            ("Options", not attrs.get("Options")),
            ("Labels", attrs.get("Labels") == volume.labels),
            ("Scope", attrs.get("Scope") == "local"),
        ):
            if not matches:
                raise InfrastructureError(
                    f"sandbox env volume {volume.name} configuration mismatch: {key}"
                )

    @staticmethod
    def attest_service(
        service: EnvServicePlan,
        attrs: Mapping[str, Any],
        *,
        network_id: str | None,
    ) -> None:
        """Pre- and post-start check of one container against its plan."""
        name = service.name
        host = attrs.get("HostConfig") or {}
        planned = service.host_config
        for key in _EXACT_HOST_FIELDS:
            if _canonical(host.get(key)) != _canonical(planned[key]):
                raise _mismatch(name, key)
        for key in _EMPTY_HOST_FIELDS:
            if host.get(key):
                raise _mismatch(name, key)
        # Never sent: null is the kernel default (0 would also read as empty).
        if host.get("MemorySwappiness") is not None:
            raise _mismatch(name, "MemorySwappiness")
        restart = host.get("RestartPolicy") or {}
        if restart.get("Name") != "no" or restart.get("MaximumRetryCount") not in (
            0,
            None,
        ):
            raise _mismatch(name, "RestartPolicy")
        # The daemon reports an empty list or map as null.
        for key in ("ExtraHosts", "GroupAdd", "Mounts", "Tmpfs"):
            if _canonical(host.get(key) or None) != _canonical(planned[key] or None):
                raise _mismatch(name, key)

        config = attrs.get("Config") or {}
        body = service.config
        running = bool((attrs.get("State") or {}).get("Running"))
        expected_config = {
            "Image": body["Image"],
            "Env": body["Env"],
            "Entrypoint": body["Entrypoint"] or None,
            "Cmd": body["Cmd"] or None,
            "User": body["User"],
            "WorkingDir": body["WorkingDir"],
            "Hostname": body.get("Hostname") or str(attrs.get("Id", ""))[:12],
            "StopSignal": body.get("StopSignal"),
            "StopTimeout": body["StopTimeout"],
            "Healthcheck": body.get("Healthcheck"),
            "Labels": body["Labels"],
            "Tty": body["Tty"],
        }
        actual_config = {
            "Image": config.get("Image"),
            "Env": config.get("Env"),
            "Entrypoint": config.get("Entrypoint") or None,
            "Cmd": config.get("Cmd") or None,
            "User": config.get("User") or "",
            "WorkingDir": config.get("WorkingDir") or "",
            "Hostname": config.get("Hostname"),
            "StopSignal": config.get("StopSignal") or None,
            "StopTimeout": config.get("StopTimeout"),
            "Healthcheck": _healthcheck_body(config.get("Healthcheck")),
            "Labels": config.get("Labels"),
            "Tty": config.get("Tty"),
        }
        for key, value in expected_config.items():
            if _canonical(actual_config[key]) != _canonical(value):
                raise _mismatch(name, key)
        if (
            config.get("OpenStdin")
            or config.get("AttachStdin")
            or config.get("Domainname")
        ):
            raise _mismatch(name, "Stdin")
        if attrs.get("Image") != service.image_id:
            raise _mismatch(name, "Image")

        mounts = []
        for mount in attrs.get("Mounts") or []:
            if (
                mount.get("Type") == "tmpfs"
                and mount.get("Destination") in service.tmpfs
            ):
                continue
            if mount.get("Type") != "volume" or mount.get("Driver") != "local":
                raise _mismatch(name, "Mounts")
            mounts.append(
                (mount.get("Name"), mount.get("Destination"), mount.get("RW") is True)
            )
        if sorted(mounts) != sorted(service.mounts):
            raise _mismatch(name, "Mounts")

        networks = (attrs.get("NetworkSettings") or {}).get("Networks") or {}
        if set(networks) != {service.network_mode}:
            raise _mismatch(name, "Networks")
        if service.aliases is not None:
            endpoint = networks[service.network_mode] or {}
            if list(endpoint.get("Aliases") or []) != list(service.aliases):
                raise _mismatch(name, "Aliases")
            attached = endpoint.get("NetworkID")
            if (running or attached) and attached != network_id:
                raise _mismatch(name, "NetworkID")
        apparmor = attrs.get("AppArmorProfile")
        if apparmor != "docker-default" and (running or apparmor != ""):
            raise _mismatch(name, "AppArmorProfile")

    # -- create ------------------------------------------------------------------

    def create(
        self, plan: EnvPlan, lease: SandboxEnvLease, commit: Commit
    ) -> SandboxEnvLease:
        """Journaled plan, then rule, bridge, volumes, containers; no start.

        ``lease`` is the committed ``plan.lease(...)``. Every container is
        attested once created. A failed step with a known outcome rolls the
        whole env back and raises SetupError; an unknown outcome retains
        everything for recovery.
        """
        self._require_plan(plan, lease)
        if lease.state != "planned" or not lease.pending_mutation:
            raise InfrastructureError("sandbox env create needs its pending plan")
        journal = _Journal(commit)
        try:
            if plan.network is not None:
                try:
                    network_id = self._network.create(plan.network)
                except SetupError as error:
                    # The network backend proved its bridge and rule absent.
                    raise _Definite(error) from error
                lease = journal(_revise(lease, network_id=network_id))
            for volume in plan.volumes:
                self._require_volume_absent(volume)
                lease = journal(_volume(lease, volume.idx, created=True))
                self._create_volume(volume)
                self._attest_volume(volume)
            for service in plan.services:
                container_id = self._create_container(service)
                lease = journal(
                    _service(
                        lease, service.idx, container_id=container_id, state="created"
                    )
                )
                self.attest_service(
                    service,
                    self._require(lease, lease.services[service.idx]),
                    network_id=lease.network_id,
                )
            self._attest_network(plan, lease)
            return journal(_revise(lease, state="created", pending_mutation=False))
        except _JournalFailure as failure:
            raise failure.error from failure.error.__cause__
        except InfrastructureError as error:
            if str(error).startswith("recovery_required"):
                raise
            return self._roll_back(lease, journal, error)
        except _Definite as definite:
            return self._roll_back(lease, journal, definite.error)

    def _roll_back(
        self, lease: SandboxEnvLease, journal: _Journal, error: Exception
    ) -> SandboxEnvLease:
        try:
            # Every create call so far was answered: nothing is in flight.
            self._destroy(lease, journal, reason="start_failed", pending=False)
        except _JournalFailure as failure:
            raise failure.error from error
        except Exception as rollback_error:
            raise InfrastructureError(
                f"recovery_required: partial sandbox env {lease.env_id} rollback "
                f"is unproven: {rollback_error}"
            ) from error
        raise SetupError(
            f"failed to create sandbox env {lease.env_id}: {error}; every object "
            "is proven absent"
        ) from error

    def _require_volume_absent(self, volume: EnvVolumePlan) -> None:
        try:
            self._api.inspect_volume(volume.name)
        except NotFound:
            return
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot prove sandbox env volume {volume.name} absent: {error}"
            ) from error
        raise _Definite(SetupError(f"sandbox env volume {volume.name} already exists"))

    def _create_volume(self, volume: EnvVolumePlan) -> None:
        try:
            # Never driver options: a local volume with options can bind a
            # host path (VERIFIED type=none,o=bind).
            self._api.create_volume(
                volume.name, driver="local", labels=dict(volume.labels)
            )
        except Exception as error:
            if _answered(error):
                raise _Definite(error) from error
            raise InfrastructureError(
                f"recovery_required: sandbox env volume {volume.name} create "
                f"outcome is unknown: {error}"
            ) from error

    def _create_container(self, service: EnvServicePlan) -> str:
        try:
            created = self._api.create_container_from_config(
                service.body(), name=service.container_name
            )
        except Exception as error:
            if _answered(error):
                raise _Definite(error) from error
            raise InfrastructureError(
                f"recovery_required: sandbox env service {service.name} create "
                f"outcome is unknown: {error}"
            ) from error
        identity = (created or {}).get("Id")
        if not isinstance(identity, str) or len(identity) != 64:
            raise InfrastructureError(
                f"recovery_required: sandbox env service {service.name} create "
                "returned no container identity"
            )
        return identity

    # -- start ---------------------------------------------------------------------

    def start(
        self,
        plan: EnvPlan,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        wait_timeout_sec: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> EnvStartResult:
        """Start in dependency order; ready means ``compose up --detach --wait``.

        Every service must run, and be healthy when it has a healthcheck;
        only a completed_successfully target may be exited(0); a service
        depended on only with ``required: false`` may fail. A failure is the
        caller's result (state ``failed``), not an unknown outcome; failed
        envs stay as they are until destroyed, and so do services already
        started when ``cancelled`` turns true. A configuration mismatch, or a
        start whose outcome is unknown, kills the env (``quarantined``).

        ``cancelled`` is checked again right before every start call. A
        frozen session must never start a service: the broker cancels and
        joins the starter before ``pause``, which refuses a starting env.
        """
        self._require_plan(plan, lease)
        if lease.state not in ("created", "starting"):
            raise SandboxError("busy", "env_id", f"env is {lease.state}")
        cancelled = cancelled or (lambda: False)
        if lease.state == "created":
            lease = commit(_revise(lease, state="starting"))
        self.inventory(lease, require_present=True)
        for volume in plan.volumes:
            self._attest_volume(volume)
        self._attest_network(plan, lease)
        deadline = self._clock() + wait_timeout_sec
        progress = _Start()
        completion, optional = plan.completion_targets, plan.optional
        while True:
            if cancelled():
                return self._finish(lease, commit, progress, "failed", "canceled")
            for name in sorted(progress.started - progress.abandoned):
                progress.observed[name] = _status(
                    self._owned(lease, self._record(lease, name), missing_ok=True)
                )
                failure = self._failure(plan.service(name), progress, completion)
                if failure is None:
                    continue
                if name in optional:
                    progress.abandoned.add(name)
                    continue
                return self._finish(lease, commit, progress, "failed", failure, name)
            for name in plan.order:
                if name in progress.started or name in progress.abandoned:
                    continue
                service = plan.service(name)
                verdict = self._dependencies(service, progress)
                if verdict == "wait":
                    continue
                if verdict == "failed":
                    if name in optional:
                        progress.abandoned.add(name)
                        continue
                    return self._finish(
                        lease, commit, progress, "failed", "start_failed", name
                    )
                if cancelled():
                    return self._finish(lease, commit, progress, "failed", "canceled")
                lease, reason, detail = self._start_one(lease, commit, service)
                if reason == "quarantined":
                    return self._quarantine(lease, commit, name, detail)
                if reason is not None:
                    if name in optional:
                        progress.abandoned.add(name)
                        continue
                    return self._finish(
                        lease, commit, progress, "failed", reason, name, detail
                    )
                progress.started.add(name)
                progress.observed[name] = _status(
                    self._require(lease, lease.services[service.idx])
                )
            if self._ready(plan, progress, completion):
                return self._finish(lease, commit, progress, "ready")
            if self._clock() >= deadline:
                return self._finish(lease, commit, progress, "failed", "wait_timeout")
            self._sleep(self._poll)

    @staticmethod
    def _failure(
        service: EnvServicePlan, progress: _Start, completion: Collection[str]
    ) -> EnvReason | None:
        status = progress.observed[service.name]
        if status.state == "removed":
            return "start_failed"
        if status.state == "exited":
            if service.name in completion and status.exit_code == 0:
                return None
            return "start_failed"
        if service.has_healthcheck and status.health == "unhealthy":
            return "unhealthy"
        return None

    @staticmethod
    def _dependencies(
        service: EnvServicePlan, progress: _Start
    ) -> Literal["ready", "wait", "failed"]:
        for target, dependency in service.depends_on.items():
            if target in progress.abandoned:
                if dependency.required:
                    return "failed"
                continue
            if target not in progress.started:
                return "wait"
            status = progress.observed[target]
            if dependency.condition == "healthy":
                if not (status.state == "running" and status.health == "healthy"):
                    return "wait"
            elif dependency.condition == "completed_successfully":
                if not (status.state == "exited" and status.exit_code == 0):
                    return "wait"
        return "ready"

    @staticmethod
    def _ready(plan: EnvPlan, progress: _Start, completion: Collection[str]) -> bool:
        for service in plan.services:
            if service.name in progress.abandoned:
                continue
            if service.name not in progress.started:
                return False
            status = progress.observed[service.name]
            if status.state == "exited" and service.name in completion:
                continue
            if status.state != "running":
                return False
            if service.has_healthcheck and status.health != "healthy":
                return False
        return True

    def _start_one(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        service: EnvServicePlan,
    ) -> tuple[SandboxEnvLease, EnvReason | None, str | None]:
        """Attest, start, attest again; else the failure reason and a fixed
        detail (Engine explanations stay in the host log)."""
        record = lease.services[service.idx]
        try:
            self.attest_service(
                service, self._require(lease, record), network_id=lease.network_id
            )
        except InfrastructureError as error:
            if _recovery(error):
                raise
            LOGGER.warning("sandbox env %s: %s", lease.env_id, error)
            return lease, "quarantined", "configuration mismatch"
        try:
            self._api.start(record.container_id)
        except Exception as error:
            return self._start_failed(lease, record, error)
        lease = commit(_service(lease, service.idx, state="running"))
        try:
            self.attest_service(
                service, self._require(lease, record), network_id=lease.network_id
            )
        except InfrastructureError as error:
            if _recovery(error):
                raise
            LOGGER.warning("sandbox env %s: %s", lease.env_id, error)
            return lease, "quarantined", "configuration mismatch"
        return lease, None, None

    def _start_failed(
        self,
        lease: SandboxEnvLease,
        record: SandboxEnvServiceLease,
        error: Exception,
    ) -> tuple[SandboxEnvLease, EnvReason, str]:
        """An answered refusal that left the service not running is the
        user's failure, whatever the status (Docker 29 answers an unknown
        ``user`` with 500). A start that may still run unattested, because it
        was never answered or the service runs anyway, quarantines the env."""
        LOGGER.warning(
            "sandbox env %s service %s start failed: %s",
            lease.env_id,
            record.name,
            error,
        )
        if _answered(error):
            try:
                attrs = self._owned(lease, record, missing_ok=True)
            except InfrastructureError as inspect_error:
                if _recovery(inspect_error):
                    raise
                attrs = None
            state = (attrs or {}).get("State") or {}
            if attrs is not None and not state.get("Running"):
                return lease, "start_failed", _start_failure(error)
        return lease, "quarantined", "service start outcome is unknown"

    def _finish(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        progress: _Start,
        state: Literal["ready", "failed"],
        reason: EnvReason | None = None,
        service: str | None = None,
        detail: str | None = None,
    ) -> EnvStartResult:
        for record in lease.services:
            status = progress.observed.get(record.name)
            if status is not None and status.state in ("running", "exited"):
                lease = _service(lease, record.idx, state=status.state)
        lease = commit(_revise(lease, state=state, reason=reason))
        return EnvStartResult(lease, state, reason, service, detail)

    def _quarantine(
        self, lease: SandboxEnvLease, commit: Commit, name: str, detail: str | None
    ) -> EnvStartResult:
        """A drifted or unattested container never keeps running."""
        lease = self.fail(lease, commit, "quarantined")
        return EnvStartResult(lease, "failed", "quarantined", name, detail)

    # -- runtime ---------------------------------------------------------------------

    def status(self, lease: SandboxEnvLease) -> dict[str, ServiceStatus]:
        return {
            record.name: (
                ServiceStatus("removed")
                if record.state in ("planned", "removed")
                else _status(self._owned(lease, record, missing_ok=True))
            )
            for record in lease.services
        }

    def diagnostics(self, lease: SandboxEnvLease, service: str) -> dict[str, str]:
        """Bounded health and log tails (1 KiB / 4 KiB) for env_status."""
        record = self._record(lease, service)
        attrs = self._owned(lease, record, missing_ok=True)
        if attrs is None:
            return {"health_tail": "", "log_tail": ""}
        log = ((attrs.get("State") or {}).get("Health") or {}).get("Log") or []
        health = log[-1].get("Output", "") if log else ""
        try:
            output = self._api.logs(
                attrs["Id"], stdout=True, stderr=True, tail=200, timestamps=False
            )
        except (APIError, OSError):
            output = b""
        return {
            "health_tail": _tail(health or "", HEALTH_TAIL_BYTES),
            "log_tail": _tail(output or b"", LOG_TAIL_BYTES),
        }

    def pause(self, lease: SandboxEnvLease, commit: Commit) -> SandboxEnvLease:
        """Freeze every running service (freeze_work); proven by inspection.

        A starting env is refused (``busy``): its starter could start one
        more service after the freeze. The broker cancels and joins it first.
        """
        if lease.state == "starting":
            raise SandboxError("busy", "env_id", "a starting env cannot pause")
        paused = []
        try:
            for record in lease.services:
                if record.state in ("planned", "removed"):
                    continue
                state = self._require(lease, record).get("State") or {}
                if state.get("Paused"):
                    paused.append(record.idx)
                    continue
                if not state.get("Running"):
                    continue
                try:
                    self._api.pause(record.container_id)
                except (APIError, OSError) as error:
                    failure: Exception | None = error
                else:
                    failure = None
                state = self._require(lease, record).get("State") or {}
                if not state.get("Running"):
                    continue  # it exited meanwhile: nothing left to freeze
                if not state.get("Paused"):
                    raise InfrastructureError(
                        f"sandbox env service {record.name} pause could not be proven"
                    ) from failure
                paused.append(record.idx)
        finally:
            if paused:
                for index in paused:
                    lease = _service(lease, index, state="paused")
                if lease.state == "ready":
                    lease = _revise(lease, state="paused")
                lease = commit(lease)
        return lease

    def resume(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        admission: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> SandboxEnvLease:
        resumed = []
        try:
            for record in lease.services:
                if record.state != "paused":
                    continue
                state = self._require(lease, record).get("State") or {}
                failure: Exception | None = None
                if state.get("Paused"):
                    with admission() if admission is not None else nullcontext():
                        try:
                            self._api.unpause(record.container_id)
                        except (APIError, OSError) as error:
                            failure = error
                state = self._require(lease, record).get("State") or {}
                if state.get("Paused") or not state.get("Running"):
                    raise InfrastructureError(
                        f"sandbox env service {record.name} resume could not be proven"
                    ) from failure
                resumed.append(record.idx)
        finally:
            if resumed:
                for index in resumed:
                    lease = _service(lease, index, state="running")
                if lease.state == "paused" and not any(
                    record.state == "paused" for record in lease.services
                ):
                    lease = _revise(lease, state="ready")
                lease = commit(lease)
        return lease

    def stop_service(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        service: str,
        *,
        timeout_sec: float,
    ) -> tuple[SandboxEnvLease, int | None]:
        """StopSignal, then SIGKILL after ``timeout_sec`` (≤60); the env stays."""
        if (
            type(timeout_sec) not in (int, float)
            or not math.isfinite(timeout_sec)
            or not 0 <= timeout_sec <= 60
        ):
            raise SandboxError("invalid", "timeout_sec", "expected 0..60 seconds")
        record = self._record(lease, service)
        if record.state in ("planned", "removed"):
            raise SandboxError("invalid", "service", f"service {service} is absent")
        state = self._require(lease, record).get("State") or {}
        if state.get("Paused"):
            # Moby thaws a paused container to deliver a stop signal.
            raise SandboxError("busy", "service", "a paused service cannot stop")
        if state.get("Running"):
            try:
                self._api.stop(record.container_id, timeout=math.ceil(timeout_sec))
            except (APIError, OSError) as error:
                # SIGKILL and the proof below settle it either way.
                LOGGER.warning(
                    "sandbox env %s service %s stop failed: %s",
                    lease.env_id,
                    service,
                    error,
                )
        status = _status(
            terminate_container(
                self._api,
                record.container_id,
                inspect=lambda: self._owned(lease, record, missing_ok=True),
                paused_killer=self._paused_killer,
                clock=self._clock,
                sleep=self._sleep,
                timeout=self._stop_proof,
            )
        )
        if status.state in ("exited", "created"):
            lease = commit(_service(lease, record.idx, state=status.state))
        return lease, status.exit_code

    def terminate(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        deadline: float | None = None,
    ) -> SandboxEnvLease:
        """SIGKILL every service (paused ones too) without removing anything.

        Every service is signalled before any proof waits, and all proofs
        share one ``deadline`` (a ``clock`` value, default now + 10 s), so
        close_judge can bound its kill phase. Failures are recovery_required.
        """
        with _teardown_guard(lease.env_id, "termination"):
            return self._terminate_all(lease, _Journal(commit), deadline)

    def fail(
        self, lease: SandboxEnvLease, commit: Commit, reason: EnvReason
    ) -> SandboxEnvLease:
        """Terminate every service and record ``failed(reason)``.

        For example ``disk_quota`` from the DiskWatchdog or ``quarantined``.
        Objects stay until destroy; an env that already failed keeps its
        first reason, but its services are still terminated.
        """
        if lease.state not in _FAILABLE:
            raise SandboxError("busy", "env_id", f"env is {lease.state}")
        lease = self.terminate(lease, commit)
        if lease.state == "failed" and lease.reason is not None:
            return lease
        return commit(_revise(lease, state="failed", reason=reason))

    def _terminate_all(
        self, lease: SandboxEnvLease, journal: _Journal, deadline: float | None
    ) -> SandboxEnvLease:
        if deadline is None:
            deadline = self._clock() + self._stop_proof
        signalled = []
        for record in lease.services:
            if record.state == "removed":
                continue
            attrs = self._owned(lease, record, missing_ok=True)
            if attrs is None:
                continue
            settled = kill_container(
                self._api, attrs["Id"], attrs, paused_killer=self._paused_killer
            )
            signalled.append((record, settled))
        for record, settled in signalled:
            attrs = await_stopped(
                lambda record=record: self._owned(lease, record, missing_ok=True),
                settled,
                deadline=deadline,
                clock=self._clock,
                sleep=self._sleep,
            )
            if attrs is not None and record.state != "planned":
                state = _status(attrs).state
                lease = journal(
                    _service(
                        lease,
                        record.idx,
                        state="created" if state == "created" else "exited",
                    )
                )
        return lease

    def _terminate(
        self, lease: SandboxEnvLease, record: SandboxEnvServiceLease
    ) -> Mapping[str, Any] | None:
        attrs = self._owned(lease, record, missing_ok=True)
        if attrs is None:
            return None
        return terminate_container(
            self._api,
            attrs["Id"],
            inspect=lambda: self._owned(lease, record, missing_ok=True),
            paused_killer=self._paused_killer,
            clock=self._clock,
            sleep=self._sleep,
            timeout=self._stop_proof,
        )

    def archive_target(self, lease: SandboxEnvLease, service: str) -> ArchiveTarget:
        """The attested container of ``service`` for copy and path-stat."""
        record = self._record(lease, service)
        if record.state in ("planned", "removed"):
            raise SandboxError("invalid", "service", f"service {service} is absent")
        attrs = self._require(lease, record)
        tmpfs = {"/dev/shm", *((attrs.get("HostConfig") or {}).get("Tmpfs") or {})}
        tmpfs.update(
            mount.get("Destination")
            for mount in attrs.get("Mounts") or []
            if mount.get("Type") == "tmpfs" and mount.get("Destination")
        )
        return ArchiveTarget(
            container_id=attrs["Id"],
            tmpfs=tuple(sorted(tmpfs)),
            inspect=lambda: self._require(lease, record),
        )

    def exec_target(self, lease: SandboxEnvLease, service: str) -> ExecTarget:
        """The attested container of ``service`` for exec (``ExecPump.start``).

        ``inspect`` returns None once the container is gone, so the pump can
        tell an exec killed with its service from one that exited.
        """
        record = self._record(lease, service)
        if record.state in ("planned", "removed"):
            raise SandboxError("invalid", "service", f"service {service} is absent")
        attrs = self._require(lease, record)
        return ExecTarget(
            container_id=attrs["Id"],
            inspect=lambda: self._owned(lease, record, missing_ok=True),
        )

    # -- teardown ------------------------------------------------------------------

    def destroy(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        reason: EnvReason | None = None,
    ) -> SandboxEnvLease:
        """Containers, then volumes, then bridge and rule; each proven absent.

        Idempotent. A container never journaled by ID is found by planned
        name and exact labels; a foreign object holding a planned name is
        never removed (fail closed). While ``pending_mutation`` is set, a
        create may still be in flight (``_in_flight``): finding nothing for
        it proves nothing, so teardown stops there with recovery_required
        and keeps every later object, the firewall rule last. Any other
        failure is recovery_required too.
        """
        with _teardown_guard(lease.env_id, "removal"):
            return self._destroy(
                lease, _Journal(commit), reason=reason, pending=lease.pending_mutation
            )

    def _destroy(
        self,
        lease: SandboxEnvLease,
        journal: _Journal,
        *,
        reason: EnvReason | None,
        pending: bool,
    ) -> SandboxEnvLease:
        if lease.state == "removed":
            return lease
        in_flight = _in_flight(lease) if pending else _InFlight()
        lease = journal(_revise(lease, state="stopping", reason=reason or lease.reason))
        for record in lease.services:
            if record.state == "removed":
                continue
            found = self._remove_service(lease, record)
            if not found and record.idx == in_flight.service:
                raise _pending_unknown(lease.env_id, f"service {record.name}")
            lease = journal(_service(lease, record.idx, state="removed"))
        self.inventory(lease, require_present=False)
        for volume in lease.volumes:
            if not volume.created:
                continue
            found = self._remove_volume(lease, volume)
            if not found and volume.idx == in_flight.volume:
                raise _pending_unknown(lease.env_id, f"volume {volume.planned_name}")
            lease = journal(_volume(lease, volume.idx, created=False))
        network = env_lease_network_plan(lease)
        if network is not None:
            # M2 keeps the rule when a pending bridge create finds nothing.
            self._network.remove(network, lease.network_id, pending=in_flight.network)
        if lease.network_id is not None or lease.pending_mutation:
            # Every create that could have been in flight is resolved now.
            lease = journal(_revise(lease, network_id=None, pending_mutation=False))
        if self.inventory(lease, require_present=False):
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: sandbox env {lease.env_id} gained objects "
                "during teardown"
            )
        return journal(_revise(lease, state="removed", pending_mutation=False))

    def _remove_service(
        self, lease: SandboxEnvLease, record: SandboxEnvServiceLease
    ) -> bool:
        """False when no container holds the service's identity."""
        attrs = self._terminate(lease, record)
        if attrs is None:
            return False
        self._remove_container(attrs["Id"], record.name)
        return True

    def _remove_container(self, container_id: str, name: str) -> None:
        failure: Exception | None = None
        try:
            # v=True: no anonymous volume may outlive its container.
            self._api.remove_container(container_id, v=True, force=False)
        except NotFound:
            return
        except (APIError, OSError) as error:
            failure = error
        try:
            self._api.inspect_container(container_id)
        except NotFound:
            return
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: sandbox env service {name} absence is "
                f"unproven: {error}"
            ) from error
        detail = f": {failure}" if failure is not None else ""
        raise InfrastructureError(
            f"recovery_required: sandbox env service {name} remains after "
            f"removal{detail}"
        ) from failure

    def _remove_volume(
        self, lease: SandboxEnvLease, volume: SandboxEnvVolumeLease
    ) -> bool:
        """False when the volume was already absent."""
        name = volume.planned_name
        try:
            attrs = self._api.inspect_volume(name)
        except NotFound:
            return False
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: cannot inspect sandbox env volume {name}: {error}"
            ) from error
        if attrs.get("Name") != name or attrs.get("Labels") != env_volume_labels(
            lease.owner, lease.env_id
        ):
            raise InfrastructureError(
                f"recovery_required: sandbox env volume {name} is not owned by "
                "its planned identity"
            )
        self._remove_volume_named(name)
        return True

    def _remove_volume_named(self, name: str) -> None:
        failure: Exception | None = None
        try:
            self._api.remove_volume(name, force=False)
        except NotFound:
            return
        except (APIError, OSError) as error:
            failure = error
        try:
            self._api.inspect_volume(name)
        except NotFound:
            return
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: sandbox env volume {name} absence is "
                f"unproven: {error}"
            ) from error
        detail = f": {failure}" if failure is not None else ""
        raise InfrastructureError(
            f"recovery_required: sandbox env volume {name} remains after "
            f"removal{detail}"
        ) from failure

    # -- inventory --------------------------------------------------------------------

    def inventory(
        self, lease: SandboxEnvLease, *, require_present: bool
    ) -> tuple[str, ...]:
        """Docker's label view of the env against its journal.

        Objects carrying ``rsi-harness.sandbox-env=<id>`` must be journaled.
        An unjournaled object with exactly the env's labels is removed and
        logged (returned by name); a partial label match fails closed. With
        ``require_present`` every journaled live object must also exist.
        """
        selector = {"label": f"{LABEL_PREFIX}sandbox-env={lease.env_id}"}
        owner, env_id = lease.owner, lease.env_id
        service_labels = {
            record.name: env_container_labels(owner, env_id, record.name, record.image)
            for record in lease.services
        }
        try:
            containers = self._api.containers(all=True, filters=selector)
            volumes = (self._api.volumes(filters=selector) or {}).get("Volumes") or []
            networks = self._api.networks(filters=selector)
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: sandbox env {env_id} inventory is unproven: "
                f"{error}"
            ) from error

        live = {
            record.planned_name: record
            for record in lease.services
            if record.state != "removed"
        }
        found_containers: set[str] = set()
        extra_containers = []
        for item in containers:
            names = item.get("Names") or []
            name = names[0].lstrip("/") if names else ""
            labels = _owned_labels(item.get("Labels"))
            record = live.get(name)
            if (
                record is not None
                and labels == service_labels[record.name]
                and record.container_id in (None, item.get("Id"))
            ):
                found_containers.add(name)
            elif labels in service_labels.values():
                extra_containers.append((item.get("Id"), name))
            else:
                raise _partial(env_id, "container", name)

        volume_labels = env_volume_labels(owner, env_id)
        journaled = {volume.planned_name for volume in lease.volumes if volume.created}
        found_volumes: set[str] = set()
        extra_volumes = []
        for item in volumes:
            name = item.get("Name", "")
            labels = _owned_labels(item.get("Labels"))
            if labels != volume_labels:
                raise _partial(env_id, "volume", name)
            if name in journaled:
                found_volumes.add(name)
            else:
                extra_volumes.append(name)

        network_plan = env_lease_network_plan(lease)
        network_labels = sandbox_object_labels(
            owner, ENV_NETWORK_ROLE, {"sandbox-env": env_id}
        )
        found_network = False
        extra_networks = []
        for item in networks:
            name = item.get("Name", "")
            if _owned_labels(item.get("Labels")) != network_labels:
                raise _partial(env_id, "network", name)
            if (
                network_plan is not None
                and lease.network_id is not None
                and item.get("Id") == lease.network_id
            ):
                found_network = True
            elif network_plan is not None and lease.network_id is None:
                # Created but unjournaled: removal by planned name handles it.
                found_network = name == network_plan.name
                if not found_network:
                    extra_networks.append((item.get("Id"), name))
            else:
                extra_networks.append((item.get("Id"), name))

        if require_present:
            missing = [
                record.name
                for record in live.values()
                if record.state != "planned"
                and record.planned_name not in found_containers
            ]
            missing += sorted(journaled - found_volumes)
            if lease.network_id is not None and not found_network:
                missing.append(network_plan.name if network_plan else "network")
            if missing:
                raise InfrastructureError(
                    f"sandbox env {env_id} inventory differs from its journal: "
                    f"missing {', '.join(missing)}"
                )

        removed = []
        for container_id, name in extra_containers:
            LOGGER.warning("removing unjournaled sandbox env container %s", name)
            terminate_container(
                self._api,
                container_id,
                inspect=lambda container_id=container_id: self._inspect_or_none(
                    container_id
                ),
                paused_killer=self._paused_killer,
                clock=self._clock,
                sleep=self._sleep,
                timeout=self._stop_proof,
            )
            self._remove_container(container_id, name)
            removed.append(name)
        for name in extra_volumes:
            LOGGER.warning("removing unjournaled sandbox env volume %s", name)
            self._remove_volume_named(name)
            removed.append(name)
        for network_id, name in extra_networks:
            LOGGER.warning("removing unjournaled sandbox env network %s", name)
            self._remove_network(network_id, name)
            removed.append(name)
        return tuple(removed)

    def _inspect_or_none(self, container_id: str) -> dict[str, Any] | None:
        try:
            return self._api.inspect_container(container_id)
        except NotFound:
            return None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: cannot inspect sandbox env object: {error}"
            ) from error

    def _remove_network(self, network_id: str, name: str) -> None:
        try:
            self._api.remove_network(network_id)
        except NotFound:
            return
        except (APIError, OSError):
            pass
        try:
            self._api.inspect_network(network_id)
        except NotFound:
            return
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: sandbox env network {name} absence is "
                f"unproven: {error}"
            ) from error
        raise InfrastructureError(
            f"recovery_required: sandbox env network {name} remains after removal"
        )


# -- journal helpers -----------------------------------------------------------------


class _Definite(Exception):
    """A failed step whose outcome is known: nothing more was created."""

    def __init__(self, error: Exception) -> None:
        super().__init__(str(error))
        self.error = error


class _JournalFailure(Exception):
    def __init__(self, error: BaseException) -> None:
        super().__init__(str(error))
        self.error = error


class _Journal:
    """Separates journal failures (never rolled back) from Docker failures."""

    def __init__(self, commit: Commit) -> None:
        self._commit = commit

    def __call__(self, lease: SandboxEnvLease) -> SandboxEnvLease:
        return self.commit(lease)

    def commit(self, lease: SandboxEnvLease) -> SandboxEnvLease:
        try:
            return self._commit(lease)
        except _JournalFailure:
            raise
        except Exception as error:
            raise _JournalFailure(error) from error


@contextmanager
def _teardown_guard(env_id: str, action: str) -> Iterator[None]:
    """A teardown that cannot be proven is recovery_required (S8); journal
    failures and caller errors pass through unchanged."""
    try:
        yield
    except _JournalFailure as failure:
        raise failure.error from failure.error.__cause__
    except SandboxError:
        raise
    except Exception as error:
        if _recovery(error):
            raise
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: sandbox env {env_id} {action} is unproven: {error}"
        ) from error


@dataclass(frozen=True, slots=True)
class _InFlight:
    """The create calls of a pending lease whose outcome the journal lacks.

    create() makes one call at a time: bridge, volumes by index, containers
    by index. A bridge ID is journaled after its create, a volume's mark
    before it and a container ID after it. So only the bridge, or else the
    last marked volume and the first container without an ID can be in
    flight; both of the latter while every volume is marked and no ID is
    journaled, since the journal cannot tell whether the last volume
    create returned.
    """

    network: bool = False
    volume: int | None = None
    service: int | None = None


def _in_flight(lease: SandboxEnvLease) -> _InFlight:
    if lease.network_name is not None and lease.network_id is None:
        return _InFlight(network=True)
    marked = [volume.idx for volume in lease.volumes if volume.created]
    unjournaled = [record for record in lease.services if record.container_id is None]
    service = None
    # A first unjournaled record already removed was found and resolved.
    if (
        all(volume.created for volume in lease.volumes)
        and unjournaled
        and unjournaled[0].state == "planned"
    ):
        service = unjournaled[0].idx
    containers = any(record.container_id is not None for record in lease.services)
    volume = marked[-1] if marked and not containers else None
    return _InFlight(volume=volume, service=service)


def _pending_unknown(env_id: str, what: str) -> InfrastructureError:
    return InfrastructureError(
        f"{RECOVERY_REQUIRED}: sandbox env {env_id} {what} pending create outcome "
        "is unknown; later objects are retained"
    )


def _service(lease: SandboxEnvLease, index: int, **changes: Any) -> SandboxEnvLease:
    services = list(lease.services)
    services[index] = _revise(services[index], **changes)
    return _revise(lease, services=tuple(services))


def _volume(lease: SandboxEnvLease, index: int, **changes: Any) -> SandboxEnvLease:
    volumes = list(lease.volumes)
    volumes[index] = _revise(volumes[index], **changes)
    return _revise(lease, volumes=tuple(volumes))


def _owned_labels(labels: Mapping[str, str] | None) -> dict[str, str]:
    return {
        key: value
        for key, value in (labels or {}).items()
        if key.startswith(LABEL_PREFIX)
    }


def _partial(env_id: str, kind: str, name: str) -> InfrastructureError:
    return InfrastructureError(
        f"recovery_required: sandbox env {env_id} {kind} {name} matches only "
        "part of its label identity"
    )


__all__ = [
    "RECOVERY_REQUIRED",
    "CgroupPausedKiller",
    "DockerPausedKiller",
    "EnvPlan",
    "EnvServicePlan",
    "EnvStartResult",
    "EnvVolumePlan",
    "PausedKiller",
    "SandboxEnvDockerBackend",
    "ServiceStatus",
    "await_stopped",
    "default_paused_killer",
    "image_preflight",
    "kill_container",
    "plan_env",
    "terminate_container",
]
