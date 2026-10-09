"""Immutable managed-sandbox contracts. No dependency on the parent models."""

from __future__ import annotations

import fnmatch
import ipaddress
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
Seconds = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
# Swap of an env service as a fraction of its memory: 1 is Docker's default
# (and stock Harbor's), memory plus as much swap; 0 disables swap.
SwapRatio = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")]
IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
IMMUTABLE_IMAGE = re.compile(r"^(?:sha256:|[^\s@]+@sha256:)[0-9a-f]{64}$")
# Concurrent transfer envelope: 20 retained wire bodies (48 MiB each),
# 4 active workers (256 MiB transient allowance each), and 64 MiB for
# framing/control/transport overhead. Policy adds quota-derived metadata.
BROKER_HEADROOM_MB = 20 * 48 + 4 * 256 + 64
# Environment metadata (env_metadata_mb) grows with the grant instead of
# max_operations like v1 replay records. Per phase session (Work, and one
# Judge round) it allows 64 KiB per live env, 512 KiB per live container,
# running exec or image handle (64 KiB argv/env or image Env, widened in
# memory, as for a v1 child) and 8 MiB per running job (1 MiB inline
# Dockerfile plus arguments, widened), up to the phase's live limits; plus
# 128 KiB per long-poll waiter of the host. Removed envs and released images
# leave the journal, and full request replay records are kept for live
# objects only. An ended object's request_ids shrink to tombstones (a
# fingerprint digest plus the compact result or bounded error, about 1 KiB
# each), at most 4096 per session (sandbox_envs.MAX_TOMBSTONES): 4 MiB per
# phase. At least ENV_METADATA_MIN_MB, the allowance before the live-count
# bounds were raised, so small grants reserve what they did; at most
# 2 * (128*64 KiB + (512 + 512 + 256)*512 KiB + 4*8 MiB + 4 MiB)
# + 1024*128 KiB = 1496 MiB at the bounds below (ENV_METADATA_MAX_MB).
ENV_METADATA_MIN_MB = 256
ENV_METADATA_MAX_MB = 1496
# Live-count bounds of one phase session. The operator's policy picks the
# actual values; these only keep every per-object cost finite (bridges,
# firewall jumps, journal records, exec threads and descriptors grow with
# them; docs/sandbox-operator-guide.md, "Many envs at once"). Live containers and
# running execs allow four per env on average, as the bounds before did.
MAX_ENVS_LIVE = 128
MAX_CONTAINERS_LIVE = 512
MAX_EXECS_RUNNING = 512
MAX_JOBS_RUNNING = 4
# One long-poll per running exec of Work and of a Judge round.
MAX_WAITERS = 1024
# Live image handles (plus pending pulls and builds) of a session: at least
# 64, as before the live-count bounds were raised, and two per live env so
# that every env can run an image of its own (EnvLimits.max_image_handles).
MIN_IMAGE_HANDLES = 64
MAX_IMAGE_HANDLES = 2 * MAX_ENVS_LIVE
# Service containers are named rsi-sbx-<e16>-<idx> with idx 0..7.
MAX_ENV_SERVICES = 8
EnvNetwork = Literal["public", "none", "allowlist"]
EnvNetworks = Annotated[tuple[EnvNetwork, ...], Field(min_length=1, max_length=3)]
# Builders pull bases through their own bridge, so a build is public or none.
BuildNetwork = Literal["public", "none"]
BuildNetworks = Annotated[tuple[BuildNetwork, ...], Field(min_length=1, max_length=2)]
# Entries of one allowlist env (EnvSpec.allowlist), whatever the grant says.
MAX_ALLOWLIST_ENTRIES = 64
# The private ranges an operator may open to allowlist envs; link-local
# (metadata), loopback and multicast never are.
ALLOWLIST_PRIVATE_V4 = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "100.64.0.0/10", "172.16.0.0/12", "192.168.0.0/16")
)
_ALLOW_HOST = re.compile(
    r"^(?=.{1,253}$)[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$"
)
_ALLOW_PATTERN = re.compile(r"^(?:\*\.)?[a-z0-9*](?:[a-z0-9.*-]{0,251}[a-z0-9])?$")
Registry = Annotated[
    str, Field(pattern=r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?(?::[0-9]{1,5})?$")
]
Frontend = Annotated[
    str, Field(max_length=255, pattern=r"^[a-z0-9]+(?:[._:/-][a-z0-9]+)*$")
]


class SandboxModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


@dataclass(frozen=True, slots=True)
class AllowEntry:
    """One parsed allowlist entry: a hostname or an IPv4 address/CIDR
    (``network``), optionally limited to one TCP ``port``."""

    host: str
    network: ipaddress.IPv4Network | None
    port: int | None

    @property
    def text(self) -> str:
        return self.host if self.port is None else f"{self.host}:{self.port}"


def parse_allow_entry(value: str) -> AllowEntry:
    """``host``, ``a.b.c.d`` or ``a.b.c.d/n``, each optionally ``:port``.

    Hostnames are exact (no wildcards: the broker resolves each one) and
    normalized to lower case without a trailing dot; IPv6 is refused (env
    bridges have none). Raises ValueError.
    """
    if not isinstance(value, str) or not value or len(value) > 260:
        raise ValueError("allowlist entry must be a non-empty string")
    text = value.strip().lower()
    port: int | None = None
    if text.count(":") == 1:
        text, _, raw_port = text.partition(":")
        if not raw_port.isdigit() or not 0 < int(raw_port) < 65536:
            raise ValueError(f"allowlist entry {value!r}: port must be 1-65535")
        port = int(raw_port)
    elif ":" in text:
        raise ValueError(f"allowlist entry {value!r}: IPv6 is unsupported")
    text = text.rstrip(".")
    network: ipaddress.IPv4Network | None = None
    if re.fullmatch(r"[0-9./]+", text):
        try:
            network = ipaddress.IPv4Network(text, strict=True)
        except ValueError as error:
            raise ValueError(f"allowlist entry {value!r}: {error}") from None
        text = str(network.network_address) if network.prefixlen == 32 else str(network)
    elif _ALLOW_HOST.fullmatch(text) is None:
        raise ValueError(
            f"allowlist entry {value!r}: expected a hostname, an IPv4 address or "
            "CIDR, optionally with :port (wildcards are unsupported)"
        )
    return AllowEntry(host=text, network=network, port=port)


def allow_pattern_matches(pattern: str, entry: AllowEntry) -> bool:
    """An operator pattern: a hostname glob for hostnames, a CIDR for IPs."""
    if entry.network is None:
        return "/" not in pattern and fnmatch.fnmatchcase(entry.host, pattern)
    try:
        bound = ipaddress.IPv4Network(pattern, strict=False)
    except ValueError:
        return False
    return entry.network.subnet_of(bound)


def absolute_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or value.startswith("//")
        or ".." in path.parts
        or "\x00" in value
        or len(value.encode()) > 4096
    ):
        raise ValueError("expected a normalized absolute path (maximum 4096 bytes)")
    return value


def below(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


class SandboxProfile(SandboxModel):
    name: Name
    image: str
    cpus: PositiveInt
    memory_mb: PositiveInt
    pids: PositiveInt
    max_lifetime_sec: Seconds
    workdir: str
    tmpfs_mb: tuple[tuple[str, PositiveInt], ...]

    @field_validator("image")
    @classmethod
    def immutable_image(cls, value: str) -> str:
        if not IMMUTABLE_IMAGE.fullmatch(value):
            raise ValueError("image must be an immutable sha256 ID or digest reference")
        return value

    @model_validator(mode="after")
    def writable_layout(self) -> Self:
        absolute_path(self.workdir)
        roots: list[str] = []
        for path, _size in self.tmpfs_mb:
            absolute_path(path)
            if (
                path == "/"
                or any(
                    below(path, denied) or below(denied, path)
                    for denied in ("/proc", "/sys", "/run/rsi-harness/sandbox")
                )
                or (below(path, "/dev") and path != "/dev/shm")
            ):
                raise ValueError(f"tmpfs_mb: reserved writable path {path}")
            if any(below(path, other) or below(other, path) for other in roots):
                raise ValueError(f"tmpfs_mb: duplicate or overlapping path {path}")
            roots.append(path)
        if "/tmp" not in roots or "/dev/shm" not in roots:
            raise ValueError("tmpfs_mb must explicitly size /tmp and /dev/shm")
        if not any(below(self.workdir, root) for root in roots):
            raise ValueError("workdir must be within an approved tmpfs_mb root")
        return self


class SandboxLimits(SandboxModel):
    max_live: PositiveInt
    max_created: PositiveInt
    max_operations: PositiveInt
    max_cpus: PositiveInt
    max_memory_mb: PositiveInt
    max_lifetime_sec: Seconds
    max_upload_bytes: PositiveInt
    max_download_bytes: PositiveInt
    max_log_bytes: PositiveInt

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.max_live > self.max_created:
            raise ValueError("max_live cannot exceed max_created")
        return self


class SandboxPhaseGrant(SandboxModel):
    profiles: tuple[Name, ...] = Field(min_length=1, max_length=64)
    limits: SandboxLimits

    @field_validator("profiles")
    @classmethod
    def unique_profiles(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("duplicate phase profiles")
        return value


def _consistent_profiles(
    profiles: tuple[SandboxProfile, ...],
    work: SandboxPhaseGrant | None,
    judge: SandboxPhaseGrant | None,
) -> None:
    named = {profile.name: profile for profile in profiles}
    if len(named) != len(profiles):
        raise ValueError("profiles contain duplicate names")
    if work is None and judge is None:
        raise ValueError("sandbox must grant work or judge")
    for phase_name, phase in (("work", work), ("judge", judge)):
        if phase is None:
            continue
        for name in phase.profiles:
            if name not in named:
                raise ValueError(f"{phase_name}.profiles: unknown profile {name}")
            profile = named[name]
            for field, limit in (
                ("cpus", "max_cpus"),
                ("memory_mb", "max_memory_mb"),
                ("max_lifetime_sec", "max_lifetime_sec"),
            ):
                if getattr(profile, field) > getattr(phase.limits, limit):
                    raise ValueError(
                        f"profiles.{name}.{field} exceeds {phase_name}.limits.{limit}"
                    )


class SandboxTask(SandboxModel):
    version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    profiles: tuple[SandboxProfile, ...] = Field(min_length=1, max_length=64)
    work: SandboxPhaseGrant | None = None
    judge: SandboxPhaseGrant | None = None

    @model_validator(mode="after")
    def consistent_profiles(self) -> Self:
        _consistent_profiles(self.profiles, self.work, self.judge)
        return self


class EnvLimits(SandboxModel):
    """Per-session ceilings of one phase: the Work run or one Judge round."""

    max_envs_live: Annotated[int, Field(strict=True, gt=0, le=MAX_ENVS_LIVE)]
    max_envs_created: PositiveInt
    max_services_per_env: Annotated[int, Field(strict=True, gt=0, le=MAX_ENV_SERVICES)]
    max_containers_live: Annotated[
        int, Field(strict=True, gt=0, le=MAX_CONTAINERS_LIVE)
    ]
    max_cpus_live: PositiveInt
    max_memory_mb_live: PositiveInt
    max_disk_mb_live: PositiveInt  # soft: enforced by watchdog, not a quota fs
    cpus_per_container: PositiveInt
    memory_mb_per_container: PositiveInt
    pids_per_container: PositiveInt
    disk_mb_per_container: PositiveInt
    max_env_lifetime_sec: Seconds
    max_wait_timeout_sec: Seconds
    max_execs_running: Annotated[int, Field(strict=True, gt=0, le=MAX_EXECS_RUNNING)]
    max_exec_output_bytes: PositiveInt
    max_jobs_running: Annotated[int, Field(strict=True, gt=0, le=MAX_JOBS_RUNNING)]
    max_pull_mb: PositiveInt
    max_log_bytes: PositiveInt
    max_upload_bytes: PositiveInt
    max_download_bytes: PositiveInt
    max_operations: PositiveInt
    swap_ratio: SwapRatio = 1.0

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.max_envs_live > self.max_envs_created:
            raise ValueError("max_envs_live cannot exceed max_envs_created")
        return self

    @property
    def max_swap_mb_live(self) -> int:
        """Swap of all live services: each has floor(memory * swap_ratio)."""
        return swap_mb(self.max_memory_mb_live, self.swap_ratio)

    @property
    def max_image_handles(self) -> int:
        """Live image handles plus pending jobs of one session."""
        return max(MIN_IMAGE_HANDLES, 2 * self.max_envs_live)


def env_metadata_mb(phases: Iterable[EnvLimits], waiters: int) -> int:
    """Broker memory for the metadata of ``phases`` (ENV_METADATA_MIN_MB)."""
    kib = 128 * waiters
    for phase in phases:
        kib += (
            64 * phase.max_envs_live
            + 512
            * (
                phase.max_containers_live
                + phase.max_execs_running
                + phase.max_image_handles
            )
            + 8 * 1024 * phase.max_jobs_running
            + 4 * 1024
        )
    return max(ENV_METADATA_MIN_MB, math.ceil(kib / 1024))


def swap_mb(memory_mb: int, swap_ratio: float) -> int:
    return math.floor(memory_mb * swap_ratio)


class EnvLimitsRequest(SandboxModel):
    """Task-side tightening; an omitted limit takes the operator's value."""

    max_envs_live: PositiveInt | None = None
    max_envs_created: PositiveInt | None = None
    max_services_per_env: PositiveInt | None = None
    max_containers_live: PositiveInt | None = None
    max_cpus_live: PositiveInt | None = None
    max_memory_mb_live: PositiveInt | None = None
    max_disk_mb_live: PositiveInt | None = None
    cpus_per_container: PositiveInt | None = None
    memory_mb_per_container: PositiveInt | None = None
    pids_per_container: PositiveInt | None = None
    disk_mb_per_container: PositiveInt | None = None
    max_env_lifetime_sec: Seconds | None = None
    max_wait_timeout_sec: Seconds | None = None
    max_execs_running: PositiveInt | None = None
    max_exec_output_bytes: PositiveInt | None = None
    max_jobs_running: PositiveInt | None = None
    max_pull_mb: PositiveInt | None = None
    max_log_bytes: PositiveInt | None = None
    max_upload_bytes: PositiveInt | None = None
    max_download_bytes: PositiveInt | None = None
    max_operations: PositiveInt | None = None
    swap_ratio: SwapRatio | None = None


def _unique_items(value: tuple[str, ...] | None) -> tuple[str, ...] | None:
    if value is not None and len(set(value)) != len(value):
        raise ValueError("duplicate entries")
    return value


class EnvBuildGrant(SandboxModel):
    """Build approval; granting it accepts the documented builder exception."""

    builder_image: str
    network: BuildNetworks
    cpus: PositiveInt
    memory_mb: Annotated[int, Field(strict=True, ge=1024)]
    pids: Annotated[int, Field(strict=True, ge=512)]
    disk_mb: PositiveInt
    state_fs: Literal["loop-ext4", "tmpfs"] = "loop-ext4"
    max_builds: PositiveInt
    max_concurrent_builds: Annotated[int, Field(strict=True, ge=1, le=4)] = 1
    max_build_sec: Seconds
    max_context_mb: PositiveInt
    max_image_mb: PositiveInt
    max_images_total_mb: PositiveInt
    syntax_frontends: tuple[Frontend, ...] = Field(default=(), max_length=16)

    _unique = field_validator("network", "syntax_frontends")(_unique_items)

    @field_validator("builder_image")
    @classmethod
    def immutable_image(cls, value: str) -> str:
        if not IMMUTABLE_IMAGE.fullmatch(value):
            raise ValueError(
                "builder_image must be an immutable sha256 ID or digest reference"
            )
        return value

    @model_validator(mode="after")
    def bounded_state(self) -> Self:
        # tmpfs state is test-only and is charged to the builder's memory.
        if self.state_fs == "tmpfs" and 2 * self.disk_mb > self.memory_mb:
            raise ValueError("tmpfs builder state requires disk_mb <= memory_mb/2")
        return self


class EnvAllowlistGrant(SandboxModel):
    """Operator bounds of ``network = "allowlist"`` envs (the second key).

    ``patterns`` (empty: any entry) are hostname globs (``*.pypi.org``) a
    hostname entry must match, or CIDRs an IP entry must lie within.
    ``private_cidrs`` are the only private ranges entries may reach; other
    private, CGNAT, link-local (metadata), loopback and multicast addresses
    stay blocked even when listed. Hostnames are resolved again every
    ``refresh_sec``.
    """

    max_entries: Annotated[int, Field(strict=True, gt=0, le=MAX_ALLOWLIST_ENTRIES)] = 16
    patterns: tuple[str, ...] = Field(default=(), max_length=64)
    private_cidrs: tuple[str, ...] = Field(default=(), max_length=16)
    refresh_sec: Annotated[
        float, Field(strict=True, ge=5, le=3600, allow_inf_nan=False)
    ] = 60.0

    _unique = field_validator("patterns", "private_cidrs")(_unique_items)

    @field_validator("patterns")
    @classmethod
    def valid_patterns(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in value:
            if "/" in pattern or re.fullmatch(r"[0-9.]+", pattern):
                ipaddress.IPv4Network(pattern, strict=True)
            elif _ALLOW_PATTERN.fullmatch(pattern) is None:
                raise ValueError(
                    f"patterns: {pattern!r} is not a hostname glob or CIDR"
                )
        return value

    @field_validator("private_cidrs")
    @classmethod
    def private_only(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for cidr in value:
            network = ipaddress.IPv4Network(cidr, strict=True)
            if not any(network.subnet_of(scope) for scope in ALLOWLIST_PRIVATE_V4):
                raise ValueError(
                    f"private_cidrs: {cidr} is not inside 10/8, 100.64/10, "
                    "172.16/12 or 192.168/16"
                )
        return value


class EnvGrant(EnvLimits):
    """One phase's operator grant; limits are ceilings, never clamped."""

    network: EnvNetworks
    pull: bool = False
    registries: tuple[Registry, ...] = Field(default=(), max_length=64)
    build: EnvBuildGrant | None = None
    # Present exactly when ``network`` includes "allowlist".
    allowlist: EnvAllowlistGrant | None = None

    _unique = field_validator("network", "registries")(_unique_items)

    @model_validator(mode="after")
    def registries_for_pull(self) -> Self:
        if self.pull and not self.registries:
            raise ValueError("pull requires at least one approved registry")
        if ("allowlist" in self.network) != (self.allowlist is not None):
            raise ValueError("network allowlist and the allowlist table go together")
        return self


class EnvToolFile(SandboxModel):
    """An operator-built static binary for env services that lack the tool
    (``[environments.host.tmux]``). The broker reads ``path`` itself and
    copies it only while its SHA-256 is ``sha256``."""

    path: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @field_validator("path")
    @classmethod
    def host_path(cls, value: str) -> str:
        return absolute_path(value)


class EnvE2BHost(SandboxModel):
    """``[environments.host.e2b]``: where the broker runs env services when
    ``backend = "e2b"``. Only the broker reads the API key, from one of
    ``api_key_file`` (an absolute path) or ``api_key_env`` (a variable
    name); this record names where the key is, never the key itself, so it
    may be persisted in the run plan and the resource lease."""

    # None: the SDK default (E2B Cloud, e2b.app).
    domain: (
        Annotated[str, Field(pattern=r"^[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?$")]
        | None
    ) = None
    api_key_file: str | None = None
    api_key_env: (
        Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")] | None
    ) = None
    template_prefix: Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,19}$")] = "rsi"
    # An HTTP(S) or SOCKS5 proxy for every E2B call, e.g. from cluster
    # compute nodes; no credentials (it is persisted with the plan).
    proxy: (
        Annotated[str, Field(pattern=r"^(https?|socks5h?)://[^@/\s]+/?$")] | None
    ) = None
    # The account's maximum sandbox length (E2B refuses a longer timeout and
    # kills a sandbox that has run this long).
    max_sandbox_hours: Annotated[int, Field(strict=True, ge=1, le=24)] = 1

    @field_validator("api_key_file")
    @classmethod
    def host_path(cls, value: str | None) -> str | None:
        return None if value is None else absolute_path(value)

    @model_validator(mode="after")
    def one_key_source(self) -> Self:
        if (self.api_key_file is None) == (self.api_key_env is None):
            raise ValueError("set exactly one of api_key_file and api_key_env")
        return self


class EnvHostPolicy(SandboxModel):
    pool_disk_mb: PositiveInt
    disk_floor_mb: PositiveInt
    disk_hard_floor_mb: PositiveInt
    request_slots_active: Annotated[int, Field(strict=True, ge=1, le=64)] = 8
    request_slots_queued: Annotated[int, Field(strict=True, ge=0, le=256)] = 32
    waiters: Annotated[int, Field(strict=True, ge=1, le=MAX_WAITERS)] = 64
    no_new_privileges: bool = True
    # The first key of tool_install: a static tmux the broker may copy into
    # an env service whose image has none (offline terminus-2).
    tmux: EnvToolFile | None = None
    # Where env services run: local Docker, or E2B sandboxes (sandbox_e2b).
    backend: Literal["docker", "e2b"] = "docker"
    e2b: EnvE2BHost | None = None

    @model_validator(mode="after")
    def ordered_floors(self) -> Self:
        if self.disk_hard_floor_mb > self.disk_floor_mb:
            raise ValueError("disk_hard_floor_mb cannot exceed disk_floor_mb")
        if (self.backend == "e2b") != (self.e2b is not None):
            raise ValueError('backend = "e2b" and the e2b table go together')
        return self


class EnvRunLimits(SandboxModel):
    """Cumulative ceilings over Work plus every Judge round of one run."""

    max_envs_created: PositiveInt
    max_operations: PositiveInt
    max_builds: PositiveInt
    max_pull_mb: PositiveInt
    max_upload_bytes: PositiveInt
    max_download_bytes: PositiveInt
    max_log_bytes: PositiveInt


class EnvPolicy(SandboxModel):
    """The operator's one-time ``[environments]`` approval for a run."""

    host: EnvHostPolicy
    work: EnvGrant | None = None
    judge: EnvGrant | None = None
    run_limits: EnvRunLimits

    @model_validator(mode="after")
    def within_run_limits(self) -> Self:
        if self.work is None and self.judge is None:
            raise ValueError("environments must grant work or judge")
        shared = [
            name for name in EnvRunLimits.model_fields if name in EnvLimits.model_fields
        ]
        for phase_name in ("work", "judge"):
            phase = getattr(self, phase_name)
            if phase is None:
                continue
            for field in shared:
                if getattr(phase, field) > getattr(self.run_limits, field):
                    raise ValueError(f"{phase_name}.{field} exceeds run_limits.{field}")
            if phase.build is not None and (
                phase.build.max_builds > self.run_limits.max_builds
            ):
                raise ValueError(
                    f"{phase_name}.build.max_builds exceeds run_limits.max_builds"
                )
        return self


class EnvPhaseRequest(SandboxModel):
    """Task intent for one phase; it never grants anything by itself."""

    network: EnvNetworks
    pull: bool = False
    build: bool = False
    # Networks for image builds; omitted means the requested env networks
    # other than allowlist (builds are public or none).
    build_network: BuildNetworks | None = None
    limits: EnvLimitsRequest = EnvLimitsRequest()

    _unique = field_validator("network", "build_network")(_unique_items)

    @model_validator(mode="after")
    def build_network_needs_build(self) -> Self:
        if self.build_network is not None and not self.build:
            raise ValueError("build_network requires build = true")
        return self


class EnvRequest(SandboxModel):
    work: EnvPhaseRequest | None = None
    judge: EnvPhaseRequest | None = None

    @model_validator(mode="after")
    def some_phase(self) -> Self:
        if self.work is None and self.judge is None:
            raise ValueError("environments must request work or judge")
        return self


class SandboxEnvTask(SandboxModel):
    """Task metadata version 2: brokered environments instead of profiles."""

    version: Annotated[int, Field(strict=True, ge=2, le=2)]
    environments: EnvRequest


class SandboxPolicy(SandboxModel):
    """Operator policy: v1 profile approvals and/or v2 environment grants."""

    version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    profiles: tuple[SandboxProfile, ...] = Field(default=(), max_length=64)
    work: SandboxPhaseGrant | None = None
    judge: SandboxPhaseGrant | None = None
    run_limits: SandboxLimits | None = None
    pool_cpus: PositiveInt
    pool_memory_mb: PositiveInt
    environments: EnvPolicy | None = None

    @model_validator(mode="after")
    def approved_capabilities(self) -> Self:
        profile_approval = (
            self.profiles or self.work or self.judge or self.run_limits is not None
        )
        if profile_approval:
            if not self.profiles or self.run_limits is None:
                raise ValueError("profile approval requires profiles and run_limits")
            _consistent_profiles(self.profiles, self.work, self.judge)
        elif self.environments is None:
            raise ValueError("sandbox policy must approve profiles or environments")
        return self


class SandboxGrant(SandboxTask):
    run_limits: SandboxLimits
    reserved_cpus: PositiveInt
    reserved_memory_mb: PositiveInt
    pool_cpus: PositiveInt
    pool_memory_mb: PositiveInt

    @field_validator("profiles")
    @classmethod
    def resolved_images(
        cls, profiles: tuple[SandboxProfile, ...]
    ) -> tuple[SandboxProfile, ...]:
        if any(not IMAGE_ID.fullmatch(profile.image) for profile in profiles):
            raise ValueError("resolved profiles require exact Docker image IDs")
        return profiles


class SandboxEnvGrant(SandboxModel):
    """Effective v2 grant persisted in RunPlan.sandbox: request within policy."""

    version: Annotated[int, Field(strict=True, ge=2, le=2)] = 2
    environments: EnvPolicy
    reserved_cpus: PositiveInt
    reserved_memory_mb: PositiveInt
    reserved_disk_mb: PositiveInt
    pool_cpus: PositiveInt
    pool_memory_mb: PositiveInt

    @property
    def pool_disk_mb(self) -> int:
        return self.environments.host.pool_disk_mb

    @model_validator(mode="after")
    def resolved_builders(self) -> Self:
        for phase in (self.environments.work, self.environments.judge):
            if (
                phase is not None
                and phase.build is not None
                and (not IMAGE_ID.fullmatch(phase.build.builder_image))
            ):
                raise ValueError("resolved builder_image requires an exact image ID")
        return self


class SandboxOwner(SandboxModel):
    run_id: Name
    task_id: Name
    phase: Literal["work", "judge"]
    round_id: Name | None = None

    @model_validator(mode="after")
    def phase_round(self) -> Self:
        if (self.phase == "judge") != (self.round_id is not None):
            raise ValueError("only Judge ownership must include round_id")
        return self


SandboxState = Literal[
    "planned", "running", "paused", "expired", "stopped", "removed", "recovery-required"
]


class SandboxLease(SandboxModel):
    owner: SandboxOwner
    child_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    planned_name: Name
    image_id: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    container_id: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    state: SandboxState = "planned"
    reserved_lifetime_sec: Seconds
    cpus: PositiveInt
    memory_mb: PositiveInt
    created_at: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    expires_at: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    pending_mutation: bool = False

    @model_validator(mode="after")
    def lifetime_and_identity(self) -> Self:
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must follow created_at")
        if self.planned_name != f"rsi-sandbox-{self.child_id}":
            raise ValueError("planned_name must match child identity")
        if self.state in ("running", "paused", "stopped") and self.container_id is None:
            raise ValueError("active child state requires actual container identity")
        if self.state == "removed" and self.pending_mutation:
            raise ValueError("removed child cannot have pending mutation")
        return self


class SandboxReservation(SandboxModel):
    cpus: PositiveInt
    memory_mb: PositiveInt
    pool_cpus: PositiveInt
    pool_memory_mb: PositiveInt
    # Environment grants also reserve builder/env disk; profile grants use none.
    disk_mb: NonNegativeInt = 0
    pool_disk_mb: PositiveInt | None = None

    @model_validator(mode="after")
    def disk_pool(self) -> Self:
        if self.disk_mb and self.pool_disk_mb is None:
            raise ValueError("disk reservation requires its pool capacity")
        return self


class SandboxError(Exception):
    """Structured public operation error, never a reward or Docker authority."""

    def __init__(self, code: str, field: str, message: str):
        super().__init__(f"{code}: {field}: {message}")
        self.code = code
        self.field = field
        self.message = message


class SandboxChildStopped(SandboxError):
    """Internal adapter evidence: this failure already stopped the owned child."""


class SandboxDownloadError(SandboxError):
    """Host-observed accounting for a reconciled, failed download operation."""

    def __init__(
        self,
        code: str,
        field: str,
        message: str,
        *,
        download_bytes: int,
        operation_started: bool,
    ):
        super().__init__(code, field, message)
        if type(download_bytes) is not int or download_bytes < 0:
            raise ValueError("download_bytes must be a nonnegative integer")
        if type(operation_started) is not bool:
            raise ValueError("operation_started must be a boolean")
        if not operation_started and download_bytes:
            raise ValueError("an unstarted download cannot have transferred bytes")
        self.download_bytes = download_bytes
        self.operation_started = operation_started


class SandboxDownloadStopped(SandboxDownloadError, SandboxChildStopped):
    """Reconciled download failure with proven whole-child containment."""


class SandboxBundleEntry(SandboxModel):
    path: str
    kind: Literal["file", "directory"]
    mode: Annotated[int, Field(strict=True, ge=0, le=0o777)]
    data: bytes = b""


class SandboxResult(SandboxModel):
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    oom_killed: bool = False
    output_limited: bool = False
    truncated: bool = False
    duration_sec: Annotated[float, Field(ge=0, allow_inf_nan=False)]
