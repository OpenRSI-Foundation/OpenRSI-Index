"""Immutable brokered-environment contracts: EnvSpec and schema-6 journal records.

Everything here is pure data: no Docker, network or broker state. Names are
derived deterministically from run and handle identity so recovery can rebuild
the whole cleanup plan from the lease alone.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Literal, Self, TypeVar

from pydantic import (
    AfterValidator,
    Field,
    ValidationError,
    WrapSerializer,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from rsi_harness.runtime.sandbox_contracts import (
    MAX_ALLOWLIST_ENTRIES,
    MAX_ENV_SERVICES,
    EnvNetwork,
    NonNegativeInt,
    PositiveInt,
    SandboxError,
    SandboxModel,
    SandboxOwner,
    Seconds,
    absolute_path,
    below,
    parse_allow_entry,
)

# Same bound as a v1 exec (sandbox_client.MAX_COMMAND_BYTES): argv plus env.
ENV_COMMAND_BYTES = 64 * 1024
ENV_QUOTA_ERROR = "sandbox_quota"
ENV_NOFILE_CAP = 1024 * 1024
MAX_ENV_VOLUMES = 8
# Declared volumes plus image-declared VOLUME paths (implicit volumes).
MAX_ENV_VOLUME_LEASES = 64
BUILT_IMAGE_REPOSITORY = "rsi-sbx-img"
_RESERVED_MOUNT_ROOTS = ("/proc", "/sys", "/dev", "/run/rsi-harness")
LABEL_PREFIX = "rsi-harness."
# rsi-harness.role of every brokered object (spec 3.2).
ENV_ROLE = "sandbox-env"
ENV_NETWORK_ROLE = "sandbox-env-net"
ENV_VOLUME_ROLE = "sandbox-env-vol"
BUILDER_ROLE = "sandbox-builder"
BUILDER_NETWORK_ROLE = "sandbox-builder-net"
BUILDER_VOLUME_ROLE = "sandbox-builder-vol"
BUILD_ROLE = "sandbox-build"

EnvId = Annotated[str, Field(pattern=r"^e[0-9a-f]{32}$")]
ImageHandle = Annotated[str, Field(pattern=r"^i[0-9a-f]{32}$")]
BuilderId = Annotated[str, Field(pattern=r"^b[0-9a-f]{32}$")]
ImageId = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
ObjectId = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
E2BSandboxId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ServiceName = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}$")]
VolumeName = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,62}$")]
HostAlias = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,252}$")]
DnsLabel = Annotated[
    str, Field(pattern=r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$")
]
UserName = Annotated[
    str, Field(pattern=r"^[A-Za-z0-9_.-]{1,32}(?::[A-Za-z0-9_.-]{1,32})?$")
]
GroupName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,32}$")]
Capability = Annotated[str, Field(pattern=r"^CAP_[A-Z_]{1,32}$")]
SignalName = Annotated[
    str, Field(pattern=r"^SIG(?:[A-Z]{2,8}[0-9]?|RTMIN\+[0-9]{1,2}|RTMAX-[0-9]{1,2})$")
]
EnvKey = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^[^=\x00]+$")]
Text = Annotated[str, Field(pattern=r"^[^\x00]*$")]
HealthSeconds = Annotated[
    float, Field(strict=True, ge=0, le=86400, allow_inf_nan=False)
]
K = TypeVar("K")
V = TypeVar("V")


class FrozenMap(Mapping[K, V]):
    """Read-only, hashable mapping so validated specs stay deeply immutable."""

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[K, V] | None = None) -> None:
        self._items = dict(items or {})

    def __getitem__(self, key: K) -> V:
        return self._items[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(frozenset(self._items.items()))

    def __repr__(self) -> str:
        return f"FrozenMap({self._items!r})"


# Validated as a JSON object, then frozen; serialized back as a plain object.
# Construct with a plain dict (or model_validate_json), never a FrozenMap.
FrozenDict = Annotated[
    dict[K, V],
    AfterValidator(FrozenMap),
    WrapSerializer(lambda value, handler: handler(dict(value))),
]


def short_identity(identity: str) -> str:
    """The 16 hex digits every Docker name of an env or builder carries."""
    return identity[1:17]


def env_container_name(env_id: str, index: int) -> str:
    return f"rsi-sbx-{short_identity(env_id)}-{index}"


def env_volume_name(env_id: str, index: int) -> str:
    return f"rsi-sbvol-{short_identity(env_id)}-{index}"


def env_network_name(env_id: str) -> str:
    return f"rsi-sbnet-{short_identity(env_id)}"


def env_rule_id(run_id: str, env_id: str) -> str:
    return f"rsi-{run_id}-sbx-{short_identity(env_id)}"


def builder_container_name(builder_id: str) -> str:
    return f"rsi-sbb-{short_identity(builder_id)}"


def builder_volume_name(builder_id: str) -> str:
    return f"rsi-sbbvol-{short_identity(builder_id)}"


def builder_network_name(builder_id: str) -> str:
    return f"rsi-sbbnet-{short_identity(builder_id)}"


def builder_rule_id(run_id: str, builder_id: str) -> str:
    return f"rsi-{run_id}-sbb-{short_identity(builder_id)}"


def builder_loop_file(data_root: Path, run_id: str, builder_id: str) -> Path:
    """``<data_root>/<run_id>/sb/build/<b16>.img``; never stored in the lease.

    Recovery derives it from its own managed data root, so a journal record
    can never point loop detach or file removal outside that root.
    """
    return (
        Path(data_root) / run_id / "sb" / "build" / f"{short_identity(builder_id)}.img"
    )


def sandbox_spool_root(data_root: Path, run_id: str) -> Path:
    """``<data_root>/<run_id>/sb/spool``: stages and exec output (spec 3.2).

    Nothing journals it; recovery derives it from its managed data root.
    Production's bind_sandbox_lease builds the broker's spool root with this
    helper, so recovery removes the very path production writes. The
    builders' loop files (``sb/build``) go with the builders.
    """
    return Path(data_root) / run_id / "sb" / "spool"


def built_image_tag(run_id: str, handle: str) -> str:
    run = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    return f"{BUILT_IMAGE_REPOSITORY}:{run}-{handle[1:]}"


def sandbox_object_labels(
    owner: SandboxOwner, role: str, handles: Mapping[str, str]
) -> dict[str, str]:
    """The one exact label set of a brokered object (v1 child, env or builder).

    ``handles`` names the object's own handles without the prefix, e.g.
    ``{"sandbox-env": env_id, "sandbox-service": "db"}``. Attestation and
    recovery require equality, never a superset.
    """
    labels = {
        f"{LABEL_PREFIX}run-id": owner.run_id,
        f"{LABEL_PREFIX}task-id": owner.task_id,
        f"{LABEL_PREFIX}role": role,
        f"{LABEL_PREFIX}sandbox-phase": owner.phase,
    }
    if owner.round_id is not None:
        labels[f"{LABEL_PREFIX}round-id"] = owner.round_id
    for key, value in handles.items():
        labels[f"{LABEL_PREFIX}{key}"] = value
    return labels


def env_container_labels(
    owner: SandboxOwner, env_id: str, service: str, image: str
) -> dict[str, str]:
    """``image`` is the service's image handle.

    The daemon copies image labels onto a container. A built image carries
    ``sandbox_object_labels(owner, BUILD_ROLE, {"sandbox-image": handle})``
    (M8), and every one of those keys is set here again, so a container of
    the same session overrides each of them and its label set stays exact.
    """
    return sandbox_object_labels(
        owner,
        ENV_ROLE,
        {"sandbox-env": env_id, "sandbox-service": service, "sandbox-image": image},
    )


def env_volume_labels(owner: SandboxOwner, env_id: str) -> dict[str, str]:
    return sandbox_object_labels(owner, ENV_VOLUME_ROLE, {"sandbox-env": env_id})


def _mount_target(value: str) -> str:
    # Volume mount targets share the tmpfs restriction: nothing may shadow
    # kernel filesystems, device nodes or the broker endpoint.
    absolute_path(value)
    if value == "/" or any(below(value, root) for root in _RESERVED_MOUNT_ROOTS):
        raise ValueError(f"reserved mount target {value}")
    return value


def _unique(field: str, values: tuple) -> tuple:
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {field}")
    return values


class EnvVolumeSpec(SandboxModel):
    # Seeded volumes are mounted with NoCopy so image content never shadows seeds.
    seeded: bool = False


class EnvMount(SandboxModel):
    volume: VolumeName
    target: str
    read_only: bool = False

    _target = field_validator("target")(_mount_target)


class EnvHealthcheck(SandboxModel):
    test: tuple[Text, ...] = Field(min_length=2, max_length=4096)
    interval_sec: HealthSeconds = 30.0
    timeout_sec: HealthSeconds = 30.0
    start_period_sec: HealthSeconds = 0.0
    start_interval_sec: HealthSeconds = 5.0
    retries: Annotated[int, Field(strict=True, ge=0, le=100)] = 3

    @field_validator("test")
    @classmethod
    def docker_test_form(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value[0] == "CMD" or (value[0] == "CMD-SHELL" and len(value) == 2):
            return value
        raise ValueError('test must be ["CMD", arg, ...] or ["CMD-SHELL", command]')


class EnvDependency(SandboxModel):
    condition: Literal["started", "healthy", "completed_successfully"] = "started"
    required: bool = True


class ServiceSpec(SandboxModel):
    """One service container; it can only narrow the broker's fixed template."""

    image: ImageHandle
    entrypoint: tuple[Text, ...] | None = None
    command: tuple[Text, ...] | None = None
    env: FrozenDict[EnvKey, Text] = Field(default_factory=FrozenMap, max_length=256)
    working_dir: str | None = None
    user: UserName | None = None
    group_add: tuple[GroupName, ...] = Field(default=(), max_length=16)
    hostname: DnsLabel | None = None
    aliases: tuple[HostAlias, ...] = Field(default=(), max_length=16)
    extra_hosts: tuple[tuple[HostAlias, str], ...] = Field(default=(), max_length=64)
    network: Literal["env", "none"] = "env"
    read_only: bool = False
    tty: bool = False
    cap_drop: tuple[Capability, ...] = Field(default=(), max_length=64)
    cpus: Annotated[float, Field(strict=True, gt=0, le=1024, allow_inf_nan=False)]
    memory_mb: Annotated[int, Field(strict=True, ge=6)]
    # None means the grant's pids_per_container; every value stays finite.
    pids: Annotated[int, Field(strict=True, ge=16)] | None = None
    nofile: Annotated[int, Field(strict=True, ge=64, le=ENV_NOFILE_CAP)] = 65536
    shm_mb: PositiveInt = 64
    tmpfs: FrozenDict[str, PositiveInt] = Field(
        default_factory=FrozenMap, max_length=16
    )
    mounts: tuple[EnvMount, ...] = Field(default=(), max_length=16)
    healthcheck: EnvHealthcheck | Literal["image", "none"] | None = None
    depends_on: FrozenDict[ServiceName, EnvDependency] = Field(
        default_factory=FrozenMap, max_length=MAX_ENV_SERVICES - 1
    )
    stop_signal: SignalName | None = None
    # Whole seconds: the Engine's StopTimeout is an integer.
    stop_grace_sec: Annotated[int, Field(strict=True, ge=0, le=30)] = 10

    @field_validator("working_dir")
    @classmethod
    def absolute_working_dir(cls, value: str | None) -> str | None:
        return None if value is None else absolute_path(value)

    @field_validator("cpus")
    @classmethod
    def centi_cpus(cls, value: float) -> float:
        if abs(round(value * 100) - value * 100) > 1e-6:
            raise ValueError("cpus must be a multiple of 0.01")
        return value

    @field_validator("env")
    @classmethod
    def no_device_authority(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        for key in value:
            # The broker forces NVIDIA_VISIBLE_DEVICES=void; never let a spec
            # re-open GPU/driver capabilities through the nvidia runtime hooks.
            if key.upper().startswith("NVIDIA_"):
                raise ValueError(f"environment key {key} is reserved")
        return value

    @field_validator("extra_hosts")
    @classmethod
    def literal_addresses(
        cls, value: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, str], ...]:
        for host, address in value:
            try:
                ipaddress.ip_address(address)
            except ValueError:
                raise ValueError(
                    f"extra_hosts.{host}: expected an IP literal (no host-gateway)"
                ) from None
            if "%" in address:
                raise ValueError(f"extra_hosts.{host}: scoped addresses are refused")
        return value

    @field_validator("group_add", "aliases", "cap_drop")
    @classmethod
    def unique_items(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        return _unique(info.field_name, value)

    @model_validator(mode="after")
    def bounded_layout(self) -> Self:
        argv = (self.entrypoint or ()) + (self.command or ())
        size = sum(len(item.encode()) for item in argv) + sum(
            len(key.encode()) + len(value.encode()) for key, value in self.env.items()
        )
        if size > ENV_COMMAND_BYTES:
            # A quota, as for a v1 exec command, not a malformed spec.
            raise PydanticCustomError(ENV_QUOTA_ERROR, "argv and env exceed 64 KiB")
        if self.network == "none" and self.aliases:
            raise ValueError("a service without a network cannot have aliases")
        roots: list[str] = []
        for path in (*self.tmpfs, *(mount.target for mount in self.mounts)):
            _mount_target(path)
            if any(below(path, other) or below(other, path) for other in roots):
                raise ValueError(f"duplicate or overlapping mount target {path}")
            roots.append(path)
        if sum(self.tmpfs.values()) + self.shm_mb > self.memory_mb:
            raise ValueError("tmpfs and shm_mb together exceed memory_mb")
        return self


class EnvSpec(SandboxModel):
    """Caller-built environment; it can express no host authority at all."""

    version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    network: EnvNetwork
    # Destinations of an allowlist env (sandbox_contracts.parse_allow_entry),
    # normalized; only with network allowlist.
    allowlist: tuple[str, ...] = Field(default=(), max_length=MAX_ALLOWLIST_ENTRIES)
    lifetime_sec: Seconds | None = None
    disk_mb: PositiveInt
    volumes: FrozenDict[VolumeName, EnvVolumeSpec] = Field(
        default_factory=FrozenMap, max_length=MAX_ENV_VOLUMES
    )
    services: FrozenDict[ServiceName, ServiceSpec] = Field(
        min_length=1, max_length=MAX_ENV_SERVICES
    )

    @field_validator("allowlist")
    @classmethod
    def normalized_entries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        entries = tuple(parse_allow_entry(item).text for item in value)
        return _unique("allowlist", entries)

    @model_validator(mode="after")
    def consistent_graph(self) -> Self:
        if self.allowlist and self.network != "allowlist":
            raise ValueError("allowlist entries require network allowlist")
        owners: dict[str, str] = {}
        for name, service in self.services.items():
            for mount in service.mounts:
                if mount.volume not in self.volumes:
                    raise ValueError(f"{name}: undeclared volume {mount.volume}")
            for alias in (name, *service.aliases):
                owner = owners.setdefault(alias.lower(), name)
                if owner != name:
                    raise ValueError(f"{name}: alias {alias} already names {owner}")
            for target, dependency in service.depends_on.items():
                if target == name or target not in self.services:
                    raise ValueError(f"{name}: invalid dependency {target}")
                check = self.services[target].healthcheck
                if dependency.condition == "healthy" and check in (None, "none"):
                    raise ValueError(
                        f"{name}: healthy dependency {target} declares no healthcheck"
                    )
        visiting: set[str] = set()
        done: set[str] = set()

        def visit(name: str) -> None:
            if name in done:
                return
            if name in visiting:
                raise ValueError(f"depends_on cycle through {name}")
            visiting.add(name)
            for target in self.services[name].depends_on:
                visit(target)
            visiting.discard(name)
            done.add(name)

        for name in self.services:
            visit(name)
        return self


def _invalid(error: ValueError) -> SandboxError:
    # Never echo input values: env values and argv may carry task secrets.
    if isinstance(error, ValidationError):
        errors = error.errors(include_input=False, include_url=False)
        # A malformed spec outranks an oversized one, as in v1 request checks.
        first = next(
            (item for item in errors if item["type"] != ENV_QUOTA_ERROR), errors[0]
        )
        code = "quota" if first["type"] == ENV_QUOTA_ERROR else "invalid"
        field = ".".join(["spec", *(str(part) for part in first["loc"])])
        return SandboxError(code, field[:256], first["msg"][:1024])
    return SandboxError("invalid", "spec", "expected a JSON EnvSpec object")


def parse_env_spec(raw: object) -> EnvSpec:
    """Validate an untrusted wire EnvSpec; JSON mode keeps arrays strict.

    Only the spec itself is checked here. Fitting it to the phase grant and
    the live quota (permission/quota errors) is broker admission.
    """
    try:
        return EnvSpec.model_validate_json(json.dumps(raw, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise _invalid(error) from None


def env_spec_digest(spec: EnvSpec) -> str:
    """Journal identity of a spec; the lease stores this, never env values.

    An empty ``allowlist`` is left out, so specs without one keep the
    digest they had before the field existed."""
    canonical = json.dumps(
        spec.model_dump(mode="json", exclude=None if spec.allowlist else {"allowlist"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


EnvState = Literal[
    "planned",
    "created",
    "starting",
    "ready",
    "failed",
    "paused",
    "stopping",
    "removed",
    "recovery-required",
]
EnvReason = Literal[
    "start_failed",
    "unhealthy",
    "wait_timeout",
    "disk_quota",
    "expired",
    "quarantined",
    "canceled",
]


class SandboxEnvServiceLease(SandboxModel):
    idx: Annotated[int, Field(strict=True, ge=0, lt=MAX_ENV_SERVICES)]
    name: ServiceName
    planned_name: str
    # The service's image handle: recovery rebuilds the exact container
    # labels (rsi-harness.sandbox-image) from the lease alone.
    image: ImageHandle
    image_id: ImageId
    container_id: ObjectId | None = None
    # E2B envs (SandboxEnvLease.backend "e2b"): the sandbox, never a container.
    sandbox_id: E2BSandboxId | None = None
    state: Literal["planned", "created", "running", "paused", "exited", "removed"] = (
        "planned"
    )

    @model_validator(mode="after")
    def actual_identity(self) -> Self:
        if self.state in ("created", "running", "paused", "exited") and (
            self.container_id is None and self.sandbox_id is None
        ):
            raise ValueError("active service state requires actual container identity")
        if self.container_id is not None and self.sandbox_id is not None:
            raise ValueError("a service is a container or an E2B sandbox, not both")
        return self


class SandboxEnvVolumeLease(SandboxModel):
    idx: Annotated[int, Field(strict=True, ge=0, lt=MAX_ENV_VOLUME_LEASES)]
    # Declared volume name, or implicit:<service idx>:<sha256(path)[:16]> for
    # an image VOLUME that no mount or tmpfs covers.
    logical: Annotated[
        str,
        Field(pattern=r"^(?:[a-z0-9][a-z0-9_.-]{0,62}|implicit:[0-7]:[0-9a-f]{16})$"),
    ]
    planned_name: str
    # True from the create attempt until removal is proven: "may exist".
    created: bool = False


class SandboxEnvLease(SandboxModel):
    """Journal authority for one env; written before any Docker mutation.

    ``removed`` is proof, not history: every object without its own state
    flag must be cleared first (volume ``created``, ``network_id``). Service
    container IDs stay as evidence behind their per-service state.
    """

    owner: SandboxOwner
    env_id: EnvId
    spec_sha256: Digest
    # Where the services run; "e2b" envs have no bridge, rule or volume.
    backend: Literal["docker", "e2b"] = "docker"
    state: EnvState = "planned"
    reason: EnvReason | None = None
    created_at: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    expires_at: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    network_mode: EnvNetwork
    network_name: str | None = None
    network_id: ObjectId | None = None
    rule_id: str | None = None
    cpus_milli: PositiveInt
    memory_mb: PositiveInt
    disk_mb: PositiveInt
    services: tuple[SandboxEnvServiceLease, ...] = Field(
        min_length=1, max_length=MAX_ENV_SERVICES
    )
    volumes: tuple[SandboxEnvVolumeLease, ...] = Field(
        default=(), max_length=MAX_ENV_VOLUME_LEASES
    )
    pending_mutation: bool = False

    @model_validator(mode="after")
    def deterministic_identity(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must follow created_at")
        # Firewall rules exist exactly when a bridge does; both derive from
        # (run, env) so recovery never trusts a stored name it cannot recompute.
        if (self.network_name is None) != (self.rule_id is None):
            raise ValueError("env bridge and firewall rule must be planned together")
        if self.network_name is not None and (
            self.network_name != env_network_name(self.env_id)
            or self.rule_id != env_rule_id(self.owner.run_id, self.env_id)
        ):
            raise ValueError("env network and rule must match env identity")
        if self.network_id is not None and self.network_name is None:
            raise ValueError("env network identity requires a planned bridge")
        if self.backend == "e2b":
            if self.network_name is not None or self.volumes:
                raise ValueError("an e2b env has no bridge, rule or volume")
            if any(service.container_id is not None for service in self.services):
                raise ValueError("an e2b env service is an E2B sandbox")
        # A "none" env gets an internal bridge only when services must talk;
        # that choice is the broker's, so either shape is recoverable here.
        elif self.network_mode != "none" and self.network_name is None:
            raise ValueError(
                f"{self.network_mode} env requires its private firewalled bridge"
            )
        names = [service.name for service in self.services]
        if names != sorted(set(names)):
            raise ValueError("env services must be unique and ordered by name")
        for index, service in enumerate(self.services):
            if service.idx != index or service.planned_name != env_container_name(
                self.env_id, index
            ):
                raise ValueError("service planned name must match env identity")
        for index, volume in enumerate(self.volumes):
            if volume.idx != index or volume.planned_name != env_volume_name(
                self.env_id, index
            ):
                raise ValueError("volume planned name must match env identity")
        _unique("env volume", tuple(volume.logical for volume in self.volumes))
        if self.state in ("created", "starting", "ready", "paused") and any(
            service.container_id is None and service.sandbox_id is None
            for service in self.services
        ):
            raise ValueError("active env state requires every container identity")
        if self.reason is not None and self.state not in (
            "failed",
            "stopping",
            "removed",
            "recovery-required",
        ):
            raise ValueError("only an ending env can carry a reason")
        if self.state == "removed" and (
            self.pending_mutation
            or any(service.state != "removed" for service in self.services)
            or any(volume.created for volume in self.volumes)
            or self.network_id is not None
        ):
            raise ValueError(
                "removed env cannot retain services, volumes, network "
                "or pending mutation"
            )
        return self


ImageLeaseState = Literal["planned", "loading", "present", "removed", "leaked"]


class SandboxImageLease(SandboxModel):
    """Built images are run-owned; pulled images are a cache and never removed.

    A pulled record is a ledger entry only: it never holds the reservation or
    blocks recovery, ``removed`` means its handle was unbound (the image
    stays), and it stores no caller-chosen reference (free-form refs do not
    survive lease redaction). A built image holds the reservation until
    ``removed``; ``leaked`` records an rmi conflict still to be retried.
    """

    owner: SandboxOwner
    handle: ImageHandle
    kind: Literal["pulled", "built"]
    image_id: ImageId | None = None
    # Built images only: always built_image_tag(run, handle).
    tag: str | None = None
    pre_existing: bool = False
    state: ImageLeaseState = "planned"
    bytes: NonNegativeInt = 0

    @model_validator(mode="after")
    def ownership(self) -> Self:
        if self.kind == "built":
            if self.pre_existing:
                raise ValueError("a built image cannot be pre-existing")
            if self.tag is not None and self.tag != built_image_tag(
                self.owner.run_id, self.handle
            ):
                raise ValueError("built image tag must match run and handle")
        else:
            if self.state in ("loading", "leaked"):
                raise ValueError("pulled images are never loaded or removed")
            if self.tag is not None:
                raise ValueError("pulled image records store no reference")
        if self.state in ("present", "leaked") and self.image_id is None:
            raise ValueError("present or leaked image requires actual image identity")
        return self


class BuilderLease(SandboxModel):
    """The per-session BuildKit builder; the documented G4 exception.

    A loop-ext4 builder's backing file is builder_loop_file(data_root, run,
    builder), derived by recovery and never stored. ``removed`` is proof: the
    attached loop device and the bridge identity are cleared first.
    """

    owner: SandboxOwner
    builder_id: BuilderId
    container_name: str
    volume_name: str
    network_name: str
    rule_id: str
    state_fs: Literal["loop-ext4", "tmpfs"]
    loop_device: Annotated[str, Field(pattern=r"^/dev/loop[0-9]{1,6}$")] | None = None
    network_id: ObjectId | None = None
    container_id: ObjectId | None = None
    state: Literal["planned", "created", "running", "stopped", "removed"] = "planned"
    cpus: PositiveInt
    memory_mb: PositiveInt
    disk_mb: PositiveInt
    pending_mutation: bool = False

    @model_validator(mode="after")
    def deterministic_identity(self) -> Self:
        run_id, builder_id = self.owner.run_id, self.builder_id
        if (
            self.container_name,
            self.volume_name,
            self.network_name,
            self.rule_id,
        ) != (
            builder_container_name(builder_id),
            builder_volume_name(builder_id),
            builder_network_name(builder_id),
            builder_rule_id(run_id, builder_id),
        ):
            raise ValueError("builder planned names must match builder identity")
        if self.state_fs == "tmpfs" and self.loop_device is not None:
            raise ValueError("tmpfs builder state cannot own a loop device")
        if self.state in ("created", "running", "stopped") and (
            self.container_id is None
        ):
            raise ValueError("active builder state requires actual container identity")
        if self.state == "removed" and (
            self.pending_mutation
            or self.loop_device is not None
            or self.network_id is not None
        ):
            raise ValueError(
                "removed builder cannot retain loop device, network or pending mutation"
            )
        return self
