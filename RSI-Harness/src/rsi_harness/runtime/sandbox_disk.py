"""Soft env disk limits and host watermarks (spec 5 DiskWatchdog, S6).

overlay2 on ext4 has no per-container quota, so env disk is measured, not
enforced by the filesystem. Only the Engine measures it (the overlay upper
directories are root-only): container writable layers through
``/containers/json?size=1`` (SizeRw), labelled env volumes through
``/system/df?type=volume``, and the host through ``statvfs`` of the Docker
root. An env over its ``disk_mb`` fails with ``disk_quota``
(``SandboxEnvDockerBackend.fail``) and is reclaimed after a grace period;
below the hard floor the largest env goes first. The overshoot is bounded by
the write rate times the container poll interval.

Deviation from spec 5 (one ``GET /containers/{id}/json?size=1`` per
service): one label-filtered list call per poll measures every env service
of the run. Labels are immutable after create and attested exactly before
and after start, so the list sees exactly the journaled services; an object
the Engine cannot size is reported in ``DiskVerdict.unmeasured`` and keeps
its last sample instead of counting as free.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_env_contracts import (
    ENV_ROLE,
    ENV_VOLUME_ROLE,
    LABEL_PREFIX,
)

MIB = 1024**2
CONTAINER_POLL_SEC = 10.0
VOLUME_POLL_SEC = 60.0
HOST_POLL_SEC = 2.0
DISK_QUOTA_GRACE_SEC = 60.0
_ENV_LABEL = f"{LABEL_PREFIX}sandbox-env"
_SERVICE_LABEL = f"{LABEL_PREFIX}sandbox-service"


@dataclass(frozen=True, slots=True)
class EnvDiskBudget:
    """One live env's soft limits: the env total and, optionally, per service."""

    env_id: str
    disk_mb: int
    container_mb: int | None = None


@dataclass(frozen=True, slots=True)
class EnvDiskUsage:
    """Last measured bytes: service writable layers and env volumes."""

    env_id: str
    containers: Mapping[str, int] = field(default_factory=dict)
    volumes: Mapping[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(self.containers.values()) + sum(self.volumes.values())

    def service_mb(self, service: str) -> int:
        return -(-self.containers.get(service, 0) // MIB)


@dataclass(frozen=True, slots=True)
class DiskVerdict:
    """What the broker must do after one watchdog poll.

    ``over_quota``: envs past their soft limit (fail them with disk_quota
    and stop their services). ``reclaim``: envs over quota for the grace
    period (remove them). ``hard_floor``: envs to remove now, largest first,
    because the host is below ``disk_hard_floor_mb``. ``refuse_new``: the
    host is below ``disk_floor_mb`` (refuse env_create, pull, build, load).
    ``unmeasured``: envs with an object the Engine could not size; their
    usage holds its last sample (0 if none) until it can.
    """

    usage: Mapping[str, EnvDiskUsage]
    over_quota: tuple[str, ...] = ()
    reclaim: tuple[str, ...] = ()
    hard_floor: tuple[str, ...] = ()
    host_free_mb: int | None = None
    refuse_new: bool = False
    unmeasured: tuple[str, ...] = ()


# {env_id: {service or volume: bytes}}; None: the Engine could not size it.
Sizes = Mapping[str, Mapping[str, int | None]]


class DiskProbe(Protocol):
    def container_sizes(self, run_id: str) -> Sizes: ...

    def volume_sizes(self, run_id: str) -> Sizes: ...

    def host_free_bytes(self) -> int: ...


def _run_filters(run_id: str, role: str) -> dict[str, list[str]]:
    return {"label": [f"{LABEL_PREFIX}run-id={run_id}", f"{LABEL_PREFIX}role={role}"]}


def _measured(size: Any) -> int | None:
    return size if type(size) is int and size >= 0 else None


class DockerDiskProbe:
    """Engine-side measurements; works as a docker-group user and as root."""

    def __init__(
        self,
        client: Any,
        *,
        docker_root: str | Path,
        statvfs: Callable[[str | Path], Any] = os.statvfs,
    ) -> None:
        self._api = client.api
        self._root = docker_root
        self._statvfs = statvfs

    def container_sizes(self, run_id: str) -> dict[str, dict[str, int | None]]:
        """``{env_id: {service: SizeRw}}`` for every env service of the run."""
        try:
            listed = self._api.containers(
                all=True, size=True, filters=_run_filters(run_id, ENV_ROLE)
            )
        except Exception as error:
            raise InfrastructureError(
                f"cannot measure sandbox env container disk: {error}"
            ) from error
        sizes: dict[str, dict[str, int | None]] = {}
        for item in listed:
            labels = item.get("Labels") or {}
            env_id, service = labels.get(_ENV_LABEL), labels.get(_SERVICE_LABEL)
            if env_id is None or service is None:
                continue
            # The Engine omits a zero SizeRw (omitempty); anything else odd
            # is unknown.
            sizes.setdefault(env_id, {})[service] = _measured(item.get("SizeRw", 0))
        return sizes

    def volume_sizes(self, run_id: str) -> dict[str, dict[str, int | None]]:
        """``{env_id: {volume: bytes}}`` from the volume-only disk usage view."""
        try:
            response = self._api._get(
                self._api._url("/system/df"), params={"type": "volume"}
            )
            usage = self._api._result(response, True)
        except Exception as error:
            raise InfrastructureError(
                f"cannot measure sandbox env volume disk: {error}"
            ) from error
        sizes: dict[str, dict[str, int | None]] = {}
        for volume in usage.get("Volumes") or ():
            labels = volume.get("Labels") or {}
            if (
                labels.get(f"{LABEL_PREFIX}run-id") != run_id
                or labels.get(f"{LABEL_PREFIX}role") != ENV_VOLUME_ROLE
                or labels.get(_ENV_LABEL) is None
            ):
                continue
            # -1 means "not computed": unknown, never free.
            sizes.setdefault(labels[_ENV_LABEL], {})[volume.get("Name", "")] = (
                _measured((volume.get("UsageData") or {}).get("Size"))
            )
        return sizes

    def host_free_bytes(self) -> int:
        try:
            result = self._statvfs(self._root)
        except OSError as error:
            raise InfrastructureError(
                f"cannot measure Docker root free space: {error}"
            ) from error
        return result.f_bavail * result.f_frsize


class DiskWatchdog:
    """Cadenced disk measurement and the soft-limit and watermark decisions.

    ``poll`` is cheap to call often (for example from the 0.1 s deadline
    sweep): each measurement runs only when its interval has elapsed.
    """

    def __init__(
        self,
        probe: DiskProbe,
        *,
        run_id: str,
        floor_mb: int,
        hard_floor_mb: int,
        clock: Callable[[], float] = time.monotonic,
        container_interval: float = CONTAINER_POLL_SEC,
        volume_interval: float = VOLUME_POLL_SEC,
        host_interval: float = HOST_POLL_SEC,
        grace_sec: float = DISK_QUOTA_GRACE_SEC,
    ) -> None:
        if hard_floor_mb > floor_mb:
            raise ValueError("disk_hard_floor_mb cannot exceed disk_floor_mb")
        self._probe = probe
        self._run_id = run_id
        self._floor = floor_mb * MIB
        self._hard_floor = hard_floor_mb * MIB
        self._clock = clock
        self._intervals = {
            "containers": container_interval,
            "volumes": volume_interval,
            "host": host_interval,
        }
        self._grace = grace_sec
        self._measured: dict[str, float] = {}
        self._containers: Mapping[str, Mapping[str, int]] = {}
        self._volumes: Mapping[str, Mapping[str, int]] = {}
        self._unmeasured: dict[str, frozenset[str]] = {}
        self._free: int | None = None
        self._host_sample = 0
        self._killed_sample = -1
        self._exceeded: dict[str, float] = {}
        self._floored: set[str] = set()

    def _due(self, kind: str, now: float) -> bool:
        # Stamped only after a successful probe: a failure retries next poll.
        last = self._measured.get(kind)
        return last is None or now - last >= self._intervals[kind]

    def _refresh_host(self, now: float, *, force: bool = False) -> None:
        if not force and not self._due("host", now):
            return
        self._free = self._probe.host_free_bytes()
        self._measured["host"] = now
        self._host_sample += 1

    def _merge(
        self, kind: str, previous: Mapping[str, Mapping[str, int]], fresh: Sizes
    ) -> dict[str, dict[str, int]]:
        """Fresh sizes; an unsized object keeps its last sample (or 0)."""
        merged: dict[str, dict[str, int]] = {}
        unmeasured = set()
        for env_id, sizes in fresh.items():
            last = previous.get(env_id, {})
            merged[env_id] = {}
            for name, size in sizes.items():
                if size is None:
                    unmeasured.add(env_id)
                    size = last.get(name, 0)
                merged[env_id][name] = size
        self._unmeasured[kind] = frozenset(unmeasured)
        return merged

    def usage(self, env_id: str) -> EnvDiskUsage:
        return EnvDiskUsage(
            env_id,
            dict(self._containers.get(env_id, {})),
            dict(self._volumes.get(env_id, {})),
        )

    def poll(self, budgets: Iterable[EnvDiskBudget]) -> DiskVerdict:
        now = self._clock()
        budgets = {budget.env_id: budget for budget in budgets}
        if self._due("containers", now):
            self._containers = self._merge(
                "containers",
                self._containers,
                self._probe.container_sizes(self._run_id),
            )
            self._measured["containers"] = now
        if self._due("volumes", now):
            self._volumes = self._merge(
                "volumes", self._volumes, self._probe.volume_sizes(self._run_id)
            )
            self._measured["volumes"] = now
        self._refresh_host(now)
        for env_id in tuple(self._exceeded):
            if env_id not in budgets:
                del self._exceeded[env_id]
        self._floored &= set(budgets)

        usage = {env_id: self.usage(env_id) for env_id in budgets}
        for env_id, budget in budgets.items():
            measured = usage[env_id]
            over = measured.total > budget.disk_mb * MIB or (
                budget.container_mb is not None
                and any(
                    size > budget.container_mb * MIB
                    for size in measured.containers.values()
                )
            )
            if over:
                self._exceeded.setdefault(env_id, now)
        over_quota = tuple(sorted(self._exceeded))
        reclaim = tuple(
            env_id
            for env_id in over_quota
            if now - self._exceeded[env_id] >= self._grace
        )
        hard_floor: tuple[str, ...] = ()
        # One victim per fresh host sample: its removal must show in statvfs
        # before a second env is judged necessary.
        if (
            self._free is not None
            and self._free < self._hard_floor
            and self._host_sample != self._killed_sample
        ):
            candidates = sorted(
                (env_id for env_id in usage if env_id not in self._floored),
                key=lambda env_id: (-usage[env_id].total, env_id),
            )
            if candidates:
                self._floored.add(candidates[0])
                self._killed_sample = self._host_sample
                hard_floor = (candidates[0],)
        return DiskVerdict(
            usage=usage,
            over_quota=over_quota,
            reclaim=reclaim,
            hard_floor=hard_floor,
            host_free_mb=None if self._free is None else self._free // MIB,
            refuse_new=self._free is not None and self._free < self._floor,
            unmeasured=tuple(
                sorted(set().union(*self._unmeasured.values()) & set(budgets))
            ),
        )

    def admit(self, requested_mb: int = 0, *, field: str = "disk_mb") -> None:
        """Admission (env_create, pull, build, load): free >= floor + request.

        Spec 3.5 refuses only below ``disk_floor_mb`` (``requested_mb=0``);
        a caller may also keep a request's own size above the floor.
        """
        now = self._clock()
        self._refresh_host(now, force=self._free is None)
        assert self._free is not None
        if self._free < self._floor + requested_mb * MIB:
            raise SandboxError(
                "quota",
                field,
                f"host disk free {self._free // MIB} MiB is below the "
                f"{self._floor // MIB} MiB floor plus {requested_mb} MiB requested",
            )

    def forget(self, env_id: str) -> None:
        """Drop state of a removed env."""
        self._exceeded.pop(env_id, None)
        self._floored.discard(env_id)


__all__ = [
    "CONTAINER_POLL_SEC",
    "DISK_QUOTA_GRACE_SEC",
    "HOST_POLL_SEC",
    "VOLUME_POLL_SEC",
    "DiskVerdict",
    "DiskWatchdog",
    "DockerDiskProbe",
    "EnvDiskBudget",
    "EnvDiskUsage",
]
