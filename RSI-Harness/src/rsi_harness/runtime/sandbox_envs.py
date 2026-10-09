"""Brokered environments, execs, stages and images (the v2 wire ops).

``SandboxBroker`` delegates every v2 operation here; this module owns no
authority of its own. Sessions, credentials, deadlines, the journal, the
request replay records and the global fail-closed switch stay the broker's,
and every call goes through the broker's ``_authenticate``, so a handle
resolves only in the session that created it (S2): a Work token never sees a
Judge env, and each Judge round is a new session that never sees the last.

Every lease mutation is journaled before its Docker call (S7): ``env_create``
commits the whole plan first, and ``SandboxEnvDockerBackend`` commits each
step through ``_commit``. Execs, copies and stages create no host resource
and are never journaled. Failure isolation (S8): an env whose outcome is
unknown, or whose container drifted, is quarantined (every service killed,
the env kept failed until destroyed); only a journal failure or a removal
that cannot be proven fails the whole run closed.

Quotas are charged per phase session and, for the cumulative ones, per run
(``EnvRunLimits``). Live quotas (envs, containers, CPUs, memory, disk,
running execs and jobs) are refunded when their object ends; cumulative ones
(operations, created envs, bytes) never are.

Image jobs (pulls and builds) and the image handles they bind run in
``sandbox_images.JobRunner``; builds, their per-session builder and built
images in ``sandbox_build.BuildService``. Admission (validation, replay
records, quotas and the planned journal record) stays here, under the
broker lock, like every other operation.

Allowlist envs: the grant bounds the entries at admission; the watchdog
turn starts a refresh thread that resolves each env's hostnames again every
``refresh_sec`` and replaces its allow chain (sandbox_network). A failed
replacement quarantines that env.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi_harness.errors import (
    InfrastructureError,
    RetryableSubmissionError,
    SetupError,
)
from rsi_harness.runtime.build_context import (
    build_fingerprint,
    prepare_build_input,
)
from rsi_harness.runtime.sandbox_archive import (
    STAGE_ID,
    ArchiveTransfer,
    StageStore,
)
from rsi_harness.runtime.sandbox_build import (
    BuilderBackend,
    BuildRequest,
    BuildService,
    build_deadline,
)
from rsi_harness.runtime.sandbox_buildfs import state_fs
from rsi_harness.runtime.sandbox_contracts import (
    EnvRunLimits,
    SandboxError,
    allow_pattern_matches,
    parse_allow_entry,
)
from rsi_harness.runtime.sandbox_disk import (
    DiskWatchdog,
    DockerDiskProbe,
    EnvDiskBudget,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    LABEL_PREFIX,
    SandboxEnvLease,
    SandboxImageLease,
    env_spec_digest,
    parse_env_spec,
)
from rsi_harness.runtime.sandbox_env_docker import (
    RECOVERY_REQUIRED,
    SandboxEnvDockerBackend,
)
from rsi_harness.runtime.sandbox_exec import (
    START_CONFIRM_SEC,
    ExecPump,
    ExecSummary,
    ProcessTable,
    parse_exec_request,
)
from rsi_harness.runtime.sandbox_images import (
    ENDED_JOB,
    PULL_POLICIES,
    DockerImagePuller,
    JobRunner,
    image_view,
    job_deadline,
    pull_reference,
    registry_of,
)
from rsi_harness.runtime.sandbox_ledger import PullLedger, pull_ledger_root
from rsi_harness.runtime.sandbox_network import (
    SandboxNetworkBackend,
    allow_destination,
)
from rsi_harness.runtime.sandbox_tools import (
    TOOL_DIR,
    TOOLS,
    read_tool,
    tool_file,
    tool_tar,
)

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
MAX_WAIT_SEC = 30.0
# close_judge: every Judge execution ends within KILL_SEC, and every Judge
# object is proven removed within DELETE_SEC, before Work may resume. A
# session's end adds time per live service container and per env: the daemon
# kills only so many containers a second, and every removal step rewrites the
# journal under the broker lock (a cancelled run's 128 public single-service
# envs took 10 s to kill and 46 s to remove with the fake firewall; the real
# one adds its iptables calls per env). Both bounds grow with the session,
# finitely: 57 s and 188 s at the hard caps.
KILL_SEC = 6.0
DELETE_SEC = 60.0
KILL_SEC_PER_CONTAINER = 0.1
DELETE_SEC_PER_ENV = 1.0
# An ending session's envs are killed, then removed, on up to this many
# threads at once: one at a time, 128 live envs would not end within KILL_SEC.
# Within docker-py's 10 pooled connections per client.
TEARDOWN_THREADS = 8
# Replay tombstones kept per session once their object ended (spec 3.1);
# bounded because max_operations may be a million (env_metadata_mb).
MAX_TOMBSTONES = 4096
ENV_ID = re.compile(r"^e[0-9a-f]{32}$")
# Build options (spec 4 B6): build-arg names, and the target stage.
BUILD_ARG = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
BUILD_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MAX_BUILD_OPTIONS = 64
MAX_BUILD_VALUE = 4096
_DENIED_BUILD_ARGS = ("BUILDKIT_", "BUILDX_")
# usage key -> (EnvLimits field, scale). Live keys are refunded; the rest are
# cumulative, and those named in EnvRunLimits are also charged per run. Keys
# of _BUILD_LIMITS are bounded by the phase's build grant instead.
_LIMITS = {
    "envs_live": ("max_envs_live", 1),
    "containers_live": ("max_containers_live", 1),
    "cpus_milli_live": ("max_cpus_live", 1000),
    "memory_mb_live": ("max_memory_mb_live", 1),
    # Derived bound (EnvLimits.max_swap_mb_live): floor(ratio * memory).
    "swap_mb_live": ("max_swap_mb_live", 1),
    "disk_mb_live": ("max_disk_mb_live", 1),
    "execs_running": ("max_execs_running", 1),
    "jobs_running": ("max_jobs_running", 1),
    "envs_created": ("max_envs_created", 1),
    "operations": ("max_operations", 1),
    "pull_bytes": ("max_pull_mb", MIB),
    "upload_bytes": ("max_upload_bytes", 1),
    "download_bytes": ("max_download_bytes", 1),
    "log_bytes": ("max_log_bytes", 1),
}
_BUILD_LIMITS = {
    "builds": ("max_builds", 1),
    "built_bytes": ("max_images_total_mb", MIB),
}
LIVE_USAGE = (
    "envs_live",
    "containers_live",
    "cpus_milli_live",
    "memory_mb_live",
    "swap_mb_live",
    "disk_mb_live",
    "execs_running",
    "jobs_running",
    "built_bytes",
)
_TRANSFERABLE_ENVS = ("created", "starting", "ready", "failed")


def _seconds(value: object, field_name: str, *, ceiling: float) -> float:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= ceiling
    ):
        raise SandboxError("invalid", field_name, f"expected 0..{ceiling:g} seconds")
    return float(value)


def _handle(value: object, pattern: re.Pattern[str], field_name: str) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise SandboxError(
            "permission", field_name, "handle is not owned by this session"
        )
    return value


def _recovery(error: BaseException) -> bool:
    return isinstance(error, InfrastructureError) and str(error).startswith(
        RECOVERY_REQUIRED
    )


def _digest(fingerprint: tuple) -> str:
    return hashlib.sha256(repr(fingerprint).encode()).hexdigest()


def _text(value: object, field_name: str, limit: int) -> str:
    if type(value) is not str or "\x00" in value or len(value.encode()) > limit:
        raise SandboxError(
            "invalid", field_name, f"expected text of at most {limit} bytes"
        )
    return value


def _build_options(
    dockerfile: object,
    dockerfile_inline: object,
    target: object,
    build_args: object,
    labels: object,
    no_cache: object,
) -> dict[str, Any]:
    """The caller's build options, checked against the allowlist (B6).

    Build args are ``[A-Za-z_][A-Za-z0-9_]*`` names with at most 4 KiB
    values and never ``BUILDKIT_*``/``BUILDX_*`` (they steer the frontend,
    e.g. BUILDKIT_SYNTAX); labels never use the ``rsi-harness.`` prefix,
    which the broker forces on every built image.
    """
    if dockerfile is not None:
        _text(dockerfile, "dockerfile", 4096)
    if dockerfile_inline is not None:
        _text(dockerfile_inline, "dockerfile_inline", MIB)
    if target is not None and (
        type(target) is not str or BUILD_TARGET.fullmatch(target) is None
    ):
        raise SandboxError("invalid", "target", "expected a build stage name")
    if type(no_cache) is not bool:
        raise SandboxError("invalid", "no_cache", "expected a boolean")
    options: dict[str, Any] = {
        "dockerfile": dockerfile,
        "dockerfile_inline": dockerfile_inline,
        "target": target,
        "no_cache": no_cache,
    }
    for field_name, value in (("build_args", build_args), ("labels", labels)):
        if not isinstance(value, dict) or len(value) > MAX_BUILD_OPTIONS:
            raise SandboxError(
                "invalid", field_name, f"expected at most {MAX_BUILD_OPTIONS} entries"
            )
        checked = {}
        for key, item in value.items():
            _text(item, f"{field_name}.{key}"[:256], MAX_BUILD_VALUE)
            if field_name == "build_args":
                if type(key) is not str or BUILD_ARG.fullmatch(key) is None:
                    raise SandboxError("invalid", field_name, "invalid build-arg name")
                if key.upper().startswith(_DENIED_BUILD_ARGS):
                    raise SandboxError(
                        "permission", field_name, f"build-arg {key} is reserved"
                    )
            else:
                _text(key, field_name, 256)
                if not key or "=" in key:
                    raise SandboxError("invalid", field_name, "invalid label key")
                if key.lower().startswith(LABEL_PREFIX):
                    raise SandboxError(
                        "permission", field_name, f"label {key[:128]} is reserved"
                    )
            checked[key] = item
        options[field_name] = dict(sorted(checked.items()))
    return options


# -- runtime -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EnvRuntime:
    """The Docker-facing parts the broker delegates to (all injectable).

    ``pump`` builds the exec engine given the broker's settlement callback;
    ``transfer`` builds one archive transfer per session stage store;
    ``builder`` (a sandbox_build.BuilderBackend) runs image builds, and
    without it ``image_build`` is unsupported; ``ledger`` (a
    sandbox_ledger.PullLedger) records pulls for ``prune-images``.
    ``backend_name`` names where services run: the Docker composition
    below, or ``sandbox_e2b.e2b_env_runtime``.
    """

    backend: Any
    images: Any
    transfer: Callable[[StageStore], Any]
    pump: Callable[[Callable[[ExecSummary], None]], Any]
    spool_root: Path
    disk: Any | None = None
    table: Any | None = None
    # Host floor of the spool filesystem (stages); raises SandboxError quota.
    spool_admit: Callable[[str], None] | None = None
    builder: Any | None = None
    ledger: Any | None = None
    backend_name: str = "docker"


def spool_floor(root: Path, floor_mb: int) -> Callable[[str], None]:
    """Admission for stage writes: the spool filesystem keeps ``floor_mb``
    free, as DiskWatchdog.admit keeps it for the Docker root."""

    def admit(field_name: str) -> None:
        existing = root
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        try:
            result = os.statvfs(existing)
        except OSError as error:
            raise SandboxError(
                "infrastructure", field_name, "sandbox spool is unavailable"
            ) from error
        free = result.f_bavail * result.f_frsize
        if free < floor_mb * MIB:
            raise SandboxError(
                "quota",
                field_name,
                f"spool disk free {free // MIB} MiB is below the {floor_mb} MiB floor",
            )

    return admit


def _each_env(
    envs: Sequence[Any], action: Callable[[Any], None]
) -> list[BaseException]:
    """``action`` for every env, on up to TEARDOWN_THREADS threads; the
    errors it raised. Each action bounds its own Docker calls."""
    if len(envs) <= 1:
        errors: list[BaseException] = []
        for env in envs:
            try:
                action(env)
            except Exception as error:
                errors.append(error)
        return errors
    with ThreadPoolExecutor(
        min(TEARDOWN_THREADS, len(envs)), thread_name_prefix="rsi-sandbox-env-close"
    ) as pool:
        futures = [pool.submit(action, env) for env in envs]
    return [error for future in futures if (error := future.exception()) is not None]


def docker_env_runtime(
    client: Any,
    firewall: Any,
    *,
    run_id: str,
    spool_root: Path,
    docker_root: str | Path,
    host: Any,
    clock: Callable[[], float] = time.monotonic,
    engine_destinations: tuple[str, ...] = (),
    paused_killer: Any = None,
    exec_killer: Any = None,
    data_root: Path | None = None,
    state_fs_runner: Any = None,
) -> EnvRuntime:
    """The production composition over one bounded-timeout Docker client.

    ``data_root`` holds loop-ext4 builder files (``<data_root>/<run>/sb/
    build``) and the host pull ledger; without it only tmpfs builder state
    (tests) is available and no pull is recorded.
    """
    table = ProcessTable()
    network = SandboxNetworkBackend(
        client, firewall, engine_destinations=engine_destinations
    )

    def statefs(kind: str) -> Any:
        if kind == "loop-ext4" and data_root is None:
            raise SetupError("loop-ext4 builder state needs the managed data root")
        options = {} if state_fs_runner is None else {"runner": state_fs_runner}
        return state_fs(kind, client.api, data_root or Path("/nonexistent"), **options)

    return EnvRuntime(
        backend=SandboxEnvDockerBackend(
            client,
            network,
            no_new_privileges=host.no_new_privileges,
            paused_killer=paused_killer,
        ),
        images=DockerImagePuller(client.api),
        transfer=lambda stages: ArchiveTransfer(client, stages),
        pump=lambda on_finish: ExecPump(
            client.api,
            Path(spool_root),
            killer=exec_killer,
            table=table,
            on_finish=on_finish,
            clock=clock,
            start_confirm_sec=START_CONFIRM_SEC,
        ),
        spool_root=Path(spool_root),
        disk=DiskWatchdog(
            DockerDiskProbe(client, docker_root=docker_root),
            run_id=run_id,
            floor_mb=host.disk_floor_mb,
            hard_floor_mb=host.disk_hard_floor_mb,
            clock=clock,
        ),
        table=table,
        spool_admit=spool_floor(Path(spool_root), host.disk_floor_mb),
        builder=BuilderBackend(client, network, statefs),
        ledger=None if data_root is None else PullLedger(pull_ledger_root(data_root)),
    )


# -- records -------------------------------------------------------------------


@dataclass(eq=False)
class _Env:
    env_id: str
    session: Any
    plan: Any
    lease: SandboxEnvLease
    deadline: float
    charges: dict[str, int]
    images: tuple[str, ...]
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel: threading.Event = field(default_factory=threading.Event)
    # Set once every service is SIGKILLed with proof (by a removal, a
    # quarantine or a disk kill): close_judge's kill stage is then met even
    # while the removal itself still runs.
    killed: threading.Event = field(default_factory=threading.Event)
    starter: threading.Thread | None = None
    reaper: threading.Thread | None = None
    # The session is ending: no new background action may start.
    closing: bool = False
    quarantined: bool = False
    # copy/stop/path-stat calls in flight (freeze refuses them as Retryable).
    uses: int = 0
    generation: int = 0
    failure: dict[str, Any] | None = None
    # (container ID, cgroup scope) and last oom_kill count per service.
    scopes: dict[str, tuple[str, Any]] = field(default_factory=dict)
    ooms: dict[str, int] = field(default_factory=dict)
    # Broker clock of the last allowlist resolution (allowlist envs only).
    refreshed_at: float = 0.0


@dataclass(eq=False)
class _Spool:
    stages: StageStore
    transfer: Any
    # copy_out stages whose bytes download_bytes already paid, by stage ID.
    prepaid: dict[str, int] = field(default_factory=dict)


class SandboxEnvs:
    """The broker's v2 half. Lock order: an env's ``lock`` is only ever taken
    without the broker lock held (its holders commit through the broker lock);
    the exec pump never calls back into the broker while holding its own."""

    def __init__(self, broker: Any, runtime: EnvRuntime) -> None:
        self._broker = broker
        self._lock = broker._lock
        self._clock = broker.clock
        self._runtime = runtime
        self._backend = runtime.backend
        self._disk = runtime.disk
        self._disk_lock = threading.Lock()
        self._disk_thread: threading.Thread | None = None
        self._refresh_thread: threading.Thread | None = None
        # Removed envs whose DiskWatchdog state the disk thread drops next.
        self._disk_forget: list[str] = []
        self._spool_admit = runtime.spool_admit
        self._table = runtime.table or ProcessTable()
        self._spool_root = Path(runtime.spool_root)
        self._envs: dict[str, _Env] = {}
        self._spools: dict[Any, _Spool] = {}
        self._sessions: dict[Any, Any] = {}
        self._run_usage: dict[str, int] = {}
        self._counter = itertools.count(1)
        self._closed = False
        self.builds = (
            None
            if runtime.builder is None
            else BuildService(
                self, runtime.builder, spool_root=self._spool_root, clock=self._clock
            )
        )
        self.images = JobRunner(self, runtime.images, self.builds, runtime.ledger)
        self.pump = runtime.pump(self._exec_finished)

    # -- authority -------------------------------------------------------------

    def _session(self, credential: str, *, mutation: bool = False) -> Any:
        """Under the broker lock: an authenticated session with an env grant."""
        session = self._broker._authenticate(credential, mutation=mutation)
        if session.env_grant is None:
            raise SandboxError("permission", "phase", "phase has no environment grant")
        self._sessions.setdefault(session.credentials, session)
        return session

    def _owned_env(self, session: Any, env_id: object) -> _Env:
        env = self._envs.get(env_id) if type(env_id) is str else None
        if env is None or env.session is not session:
            raise SandboxError(
                "permission", "env_id", "handle is not owned by this session"
            )
        return env

    def _owned_image(self, session: Any, handle: object, field_name: str) -> Any:
        return self.images.owned_image(session, handle, field_name)

    def _usable(self, env: _Env) -> None:
        state = env.lease.state
        if state in ("stopping", "removed"):
            raise SandboxError("invalid", "env_id", "env is removed or being removed")
        if env.quarantined or state == "recovery-required":
            raise SandboxError("busy", "env_id", "env is quarantined")
        if state == "planned" or env.reaper is not None:
            raise SandboxError("busy", "env_id", "env is being created or changed")

    # -- accounting ------------------------------------------------------------

    def _limit(self, session: Any, name: str) -> tuple[int, int | None]:
        """(phase ceiling, run ceiling or None) of a usage key, scaled."""
        if name in _BUILD_LIMITS:
            limit, scale = _BUILD_LIMITS[name]
            build = session.env_grant.build
            phase = 0 if build is None else getattr(build, limit) * scale
        else:
            limit, scale = _LIMITS[name]
            phase = getattr(session.env_grant, limit) * scale
        if limit not in EnvRunLimits.model_fields:
            return phase, None
        run_limits = self._broker.grant.environments.run_limits
        return phase, getattr(run_limits, limit) * scale

    def _charge(self, session: Any, **charges: int) -> None:
        for name, value in charges.items():
            phase, run = self._limit(session, name)
            if session.env_usage.get(name, 0) + value > phase or (
                run is not None and self._run_usage.get(name, 0) + value > run
            ):
                limit = (_BUILD_LIMITS.get(name) or _LIMITS[name])[0]
                raise SandboxError("quota", limit, "phase or run budget exhausted")
        for usage in (session.env_usage, self._run_usage):
            for name, value in charges.items():
                usage[name] = usage.get(name, 0) + value

    def _force(self, session: Any, **charges: int) -> None:
        """Charge bytes already spent even past the budget: nothing more fits."""
        for usage in (session.env_usage, self._run_usage):
            for name, value in charges.items():
                usage[name] = usage.get(name, 0) + value

    def _refund(self, session: Any, **charges: int) -> None:
        for usage in (session.env_usage, self._run_usage):
            for name, value in charges.items():
                usage[name] = usage.get(name, 0) - value

    def _remaining(self, session: Any, name: str) -> int:
        phase, run = self._limit(session, name)
        remaining = phase - session.env_usage.get(name, 0)
        if run is not None:
            remaining = min(remaining, run - self._run_usage.get(name, 0))
        return max(0, remaining)

    def _record(self, session: Any, request_id: object, fingerprint: tuple) -> Any:
        """The live record or tombstone of ``request_id`` (spec 3.1): the same
        fingerprint replays, a different one is ``invalid``."""
        record = self._broker._request(session, request_id, fingerprint)
        tombstone = session.tombstones.get(request_id)
        if record is not None or tombstone is None:
            return record
        digest, result, error = tombstone
        if digest != _digest(fingerprint):
            raise SandboxError(
                "invalid", "request_id", "conflicting idempotency key reuse"
            )
        if error is not None:
            raise SandboxError(*error)
        from rsi_harness.runtime.sandbox import _Request

        return _Request(fingerprint, result=json.loads(result), complete=True)

    def _new_record(
        self, session: Any, request_id: str, fingerprint: tuple, handle: str | None
    ) -> Any:
        from rsi_harness.runtime.sandbox import _Request

        record = _Request(fingerprint, handle=handle)
        session.requests[request_id] = record
        return record

    def _forget_requests(self, session: Any, handle: str) -> None:
        """Records of an ended object shrink to tombstones (fingerprint digest
        and compact result): its request_ids still replay or conflict. Only
        the newest MAX_TOMBSTONES are kept; an older request_id reused after
        its object ended counts as new."""
        for key in [
            key
            for key, record in session.requests.items()
            if getattr(record, "handle", None) == handle and record.complete
        ]:
            record = session.requests.pop(key)
            session.tombstones.pop(key, None)
            session.tombstones[key] = (
                _digest(record.fingerprint),
                json.dumps(record.result, separators=(",", ":")),
                record.error,
            )
        while len(session.tombstones) > MAX_TOMBSTONES:
            del session.tombstones[next(iter(session.tombstones))]

    @staticmethod
    def _cache(record: Any, error: BaseException) -> None:
        from rsi_harness.runtime.sandbox import _cache_error

        record.error = _cache_error(error)

    def _touch(self, item: Any) -> None:
        item.generation = next(self._counter)

    def _admit_disk(self, field_name: str) -> None:
        if self._disk is None:
            return
        with self._disk_lock:
            self._disk.admit(field=field_name)

    def _admit_spool(self, field_name: str) -> None:
        if self._spool_admit is not None:
            self._spool_admit(field_name)

    # -- journal -----------------------------------------------------------------

    def _journal_failure(self, error: BaseException) -> InfrastructureError:
        self._broker._fail_closed()
        failure = InfrastructureError("sandbox journal unavailable; recovery required")
        failure.__cause__ = error
        return failure

    def _commit(self, env: _Env) -> Callable[[SandboxEnvLease], SandboxEnvLease]:
        def commit(lease: SandboxEnvLease) -> SandboxEnvLease:
            with self._lock:
                try:
                    self._broker.journal.commit_env(lease)
                except Exception as error:
                    raise self._journal_failure(error) from error
                env.lease = lease
                self._touch(env)
            return lease

        return commit

    def _commit_image(self, lease: SandboxImageLease) -> None:
        try:
            self._broker.journal.commit_image(lease)
        except Exception as error:
            raise self._journal_failure(error) from error

    # -- capabilities --------------------------------------------------------------

    def capabilities(self, session: Any) -> dict[str, Any] | None:
        grant = session.env_grant
        if grant is None:
            return None
        usage = session.env_usage
        remaining = None
        if session.deadline is not None:
            remaining = max(0.0, session.deadline - self._clock())
        build = None
        if grant.build is not None and self.builds is not None:
            build = {
                "network": list(grant.build.network),
                "max_build_sec": grant.build.max_build_sec,
                "max_context_mb": grant.build.max_context_mb,
                "max_image_mb": grant.build.max_image_mb,
                "remaining_builds": self._remaining(session, "builds"),
            }
        backend = self._runtime.backend_name
        return {
            # Docker answers keep their v2 shape; any other backend names itself.
            **({} if backend == "docker" else {"backend": backend}),
            "network": list(grant.network),
            "pull": grant.pull,
            "registries": list(grant.registries),
            "build": build,
            "allowlist": None
            if grant.allowlist is None
            else grant.allowlist.model_dump(mode="json"),
            # Operator tools tool_install may copy (the policy's first key).
            "tools": [
                name
                for name in TOOLS
                if getattr(self._broker.host_policy, name, None) is not None
            ],
            "limits": {
                **grant.model_dump(
                    mode="json",
                    exclude={"network", "pull", "registries", "build", "allowlist"},
                ),
                "max_swap_mb_live": grant.max_swap_mb_live,
            },
            "usage": {
                "envs_live": usage.get("envs_live", 0),
                "containers_live": usage.get("containers_live", 0),
                "cpus_live": usage.get("cpus_milli_live", 0) / 1000,
                "memory_mb_live": usage.get("memory_mb_live", 0),
                "swap_mb_live": usage.get("swap_mb_live", 0),
                "disk_mb_live": usage.get("disk_mb_live", 0),
                "execs_running": usage.get("execs_running", 0),
                "jobs_running": usage.get("jobs_running", 0),
                "envs_created": usage.get("envs_created", 0),
                "operations": usage.get("operations", 0),
                "upload_bytes": usage.get("upload_bytes", 0),
                "download_bytes": usage.get("download_bytes", 0),
                "log_bytes": usage.get("log_bytes", 0),
                "image_bytes": usage.get("pull_bytes", 0) + usage.get("built_bytes", 0),
            },
            "session_remaining_sec": remaining,
        }

    # -- envs ------------------------------------------------------------------------

    def _admit_spec(self, session: Any, spec: Any) -> None:
        grant = session.env_grant
        if spec.network not in grant.network:
            raise SandboxError(
                "permission", "spec.network", "network mode not granted to phase"
            )
        if spec.network == "allowlist":
            self._admit_allowlist(grant.allowlist, spec.allowlist)
        if len(spec.services) > grant.max_services_per_env:
            raise SandboxError(
                "quota", "spec.services", "more services than max_services_per_env"
            )
        for name, service in spec.services.items():
            for value, limit, label in (
                (service.cpus, grant.cpus_per_container, "cpus"),
                (service.memory_mb, grant.memory_mb_per_container, "memory_mb"),
                (service.pids or 0, grant.pids_per_container, "pids"),
            ):
                if value > limit:
                    raise SandboxError(
                        "quota",
                        f"spec.services.{name}.{label}",
                        "exceeds the per-container grant",
                    )
        if (
            spec.lifetime_sec is not None
            and spec.lifetime_sec > grant.max_env_lifetime_sec
        ):
            raise SandboxError(
                "quota", "spec.lifetime_sec", "exceeds max_env_lifetime_sec"
            )

    @staticmethod
    def _admit_allowlist(bounds: Any, entries: tuple[str, ...]) -> None:
        """The operator's allowlist bounds; never an entry's value in errors."""
        if len(entries) > bounds.max_entries:
            raise SandboxError(
                "quota", "spec.allowlist", "more entries than allowlist max_entries"
            )
        for index, text in enumerate(entries):
            entry = parse_allow_entry(text)
            where = f"spec.allowlist.{index}"
            if bounds.patterns and not any(
                allow_pattern_matches(pattern, entry) for pattern in bounds.patterns
            ):
                raise SandboxError(
                    "permission", where, "entry matches no approved allowlist pattern"
                )
            if entry.network is not None and not allow_destination(
                entry.network, bounds.private_cidrs
            ):
                raise SandboxError(
                    "permission",
                    where,
                    "entry reaches a blocked range (private, CGNAT, metadata, "
                    "loopback or multicast) the operator did not approve",
                )

    def _deadline(self, session: Any, lifetime: float) -> float:
        deadlines = [self._clock() + lifetime]
        if session.deadline is not None:
            deadlines.append(session.deadline)
        run = self._broker._run_deadline
        if session.credentials.owner.phase == "work" and run is not None:
            deadlines.append(run)
        return min(deadlines)

    def env_create(self, credential: str, spec: object, request_id: str) -> dict:
        parsed = parse_env_spec(spec)
        fingerprint = ("env_create", env_spec_digest(parsed))
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
        # Host floor (statvfs) before any reservation, outside the broker lock.
        self._admit_disk("disk_mb")
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            grant = session.env_grant
            self._admit_spec(session, parsed)
            handles = tuple(sorted({s.image for s in parsed.services.values()}))
            attrs = {
                handle: self._owned_image(session, handle, "spec.services.image").attrs
                for handle in handles
            }
            env_id = "e" + secrets.token_hex(16)
            options = {}
            if parsed.network == "allowlist":
                options["private_cidrs"] = grant.allowlist.private_cidrs
            plan = self._backend.plan(
                session.credentials.owner,
                env_id,
                parsed,
                attrs,
                default_pids=grant.pids_per_container,
                swap_ratio=grant.swap_ratio,
                **options,
            )
            lifetime = parsed.lifetime_sec or grant.max_env_lifetime_sec
            now = self._clock()
            deadline = self._deadline(session, lifetime)
            if deadline <= now:
                raise SandboxError("expired", "session", "owner deadline expired")
            charges = dict(
                envs_live=1,
                containers_live=len(plan.services),
                cpus_milli_live=plan.cpus_milli,
                memory_mb_live=plan.memory_mb,
                swap_mb_live=plan.swap_mb,
                disk_mb_live=plan.disk_mb,
            )
            cumulative = dict(operations=1, envs_created=1)
            self._charge(session, **charges, **cumulative)
            wall = time.time()
            lease = plan.lease(created_at=wall, expires_at=wall + (deadline - now))
            try:
                self._broker.journal.plan_env(lease)
            except Exception as error:
                self._refund(session, **charges, **cumulative)
                raise self._journal_failure(error) from error
            env = _Env(env_id, session, plan, lease, deadline, charges, handles)
            env.lock.acquire()
            self._envs[env_id] = env
            self._touch(env)
            record = self._new_record(session, request_id, fingerprint, env_id)
        try:
            try:
                lease = self._backend.create(plan, lease, self._commit(env))
            except SetupError as error:
                # Rolled back and proven absent: the env is gone, not unknown.
                LOGGER.warning("sandbox env %s create failed: %s", env_id, error)
                with self._lock:
                    self._removed(env)
                raise SandboxError(
                    "infrastructure",
                    "spec",
                    "the Engine refused the env; it was rolled back, nothing remains",
                ) from None
            except InfrastructureError as error:
                if self._broker.recovery_required:
                    raise
                # Unknown outcome: only this env, unless its removal is unproven.
                LOGGER.warning("sandbox env %s create unresolved: %s", env_id, error)
                env.quarantined = True
                self._destroy_locked(env, reason="quarantined")
                raise SandboxError(
                    "infrastructure",
                    "spec",
                    "env creation did not complete; it was removed",
                ) from None
            notes = []
            if plan.network_mode == "allowlist":
                env.refreshed_at = self._clock()
                notes = list(self._backend.allowlist_notes(plan))
            result = {
                "env_id": env_id,
                "state": "created",
                "network": None if plan.network is None else plan.network.name,
                "services": {
                    service.name: {
                        "state": "created",
                        "implicit_volumes": list(service.implicit_volumes),
                    }
                    for service in plan.services
                },
                "notes": notes,
                "expires_in_sec": max(0.0, env.deadline - self._clock()),
            }
            record.result = result
            return result
        except BaseException as error:
            self._cache(record, error)
            raise
        finally:
            env.lock.release()
            with self._lock:
                record.complete = True

    def env_start(
        self, credential: str, env_id: str, wait_timeout_sec: object, request_id: str
    ) -> dict:
        if (
            type(wait_timeout_sec) not in (int, float)
            or not math.isfinite(wait_timeout_sec)
            or wait_timeout_sec <= 0
        ):
            raise SandboxError(
                "invalid", "wait_timeout_sec", "expected finite positive seconds"
            )
        timeout = float(wait_timeout_sec)
        with self._lock:
            session = self._session(credential, mutation=True)
            env = self._owned_env(session, env_id)
            fingerprint = ("env_start", env_id, timeout)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            self._usable(env)
            if env.lease.state != "created" or env.starter is not None:
                raise SandboxError("busy", "env_id", "env was already started")
            if timeout > session.env_grant.max_wait_timeout_sec:
                raise SandboxError(
                    "quota", "wait_timeout_sec", "exceeds max_wait_timeout_sec"
                )
            if timeout > env.deadline - self._clock():
                raise SandboxError(
                    "quota", "wait_timeout_sec", "exceeds the env's remaining lifetime"
                )
            self._charge(session, operations=1)
            env.starter = threading.Thread(
                target=self._run_start,
                args=(env, timeout),
                name="rsi-sandbox-env-start",
                daemon=True,
            )
            record = self._new_record(session, request_id, fingerprint, env_id)
            record.result = {"state": "starting"}
            record.complete = True
            self._touch(env)
            env.starter.start()
        return {"state": "starting"}

    def _run_start(self, env: _Env, timeout: float) -> None:
        try:
            with env.lock:
                try:
                    result = self._backend.start(
                        env.plan,
                        env.lease,
                        self._commit(env),
                        wait_timeout_sec=timeout,
                        cancelled=env.cancel.is_set,
                    )
                except InfrastructureError as error:
                    if not self._broker.recovery_required:
                        self._quarantine_locked(env, error)
                    return
                except SandboxError as error:
                    LOGGER.warning(
                        "sandbox env %s start refused: %s", env.env_id, error
                    )
                    return
                if result.state == "failed":
                    env.failure = {"service": result.service, "detail": result.detail}
                    if result.reason == "quarantined":
                        env.quarantined = True
                        self.pump.env_stopped(env.env_id)
        except Exception:
            LOGGER.exception("sandbox env %s starter failed", env.env_id)
        finally:
            with self._lock:
                env.starter = None
                self._touch(env)

    def _quarantine_locked(self, env: _Env, error: BaseException) -> None:
        """Kill a drifted or unresolved env; the env lock is held.

        Only this env stops (S8). When even its termination cannot be proven,
        the whole run fails closed.
        """
        LOGGER.warning("sandbox env %s quarantined: %s", env.env_id, error)
        with self._lock:
            env.quarantined = True
            self._touch(env)
        self.pump.env_stopped(env.env_id)
        if env.lease.state in ("stopping", "removed"):
            return
        try:
            self._backend.fail(env.lease, self._commit(env), "quarantined")
            env.killed.set()
        except SandboxError:
            pass  # already ending
        except Exception as failure:
            LOGGER.error("sandbox env %s cannot be contained: %s", env.env_id, failure)
            self._broker._fail_closed()

    def _quarantine(self, env: _Env, error: BaseException) -> None:
        if self._broker.recovery_required:
            return
        with env.lock:
            self._quarantine_locked(env, error)

    def _view_state(self, env: _Env) -> str:
        state = env.lease.state
        if state == "planned":
            return "created"
        if state == "recovery-required":
            return "failed"
        if state == "created" and env.starter is not None:
            return "starting"
        return state

    def _view_reason(self, env: _Env) -> str | None:
        if env.lease.reason is not None:
            return env.lease.reason
        if env.quarantined:
            return "quarantined"
        return None

    def _oom(self, env: _Env, name: str, container_id: str | None) -> int:
        if container_id is None:
            return env.ooms.get(name, 0)
        cached = env.scopes.get(name)
        if cached is None or cached[0] != container_id:
            try:
                attrs = self._backend.exec_target(env.lease, name).inspect() or {}
                pid = (attrs.get("State") or {}).get("Pid")
                cached = (container_id, self._table.scope(container_id, pid))
            except Exception:
                return env.ooms.get(name, 0)
            env.scopes[name] = cached
        count = self._table.oom_kills(cached[1])
        if count is not None:
            env.ooms[name] = count
        return env.ooms.get(name, 0)

    def env_status(self, credential: str, env_id: str, wait_sec: object = 0) -> dict:
        _seconds(wait_sec, "wait_sec", ceiling=MAX_WAIT_SEC)
        with self._lock:
            session = self._session(credential)
            env = self._owned_env(session, env_id)
            lease = env.lease
            state, reason = self._view_state(env), self._view_reason(env)
            remaining = max(0.0, env.deadline - self._clock())
        services: dict[str, Any] = {}
        if lease.state not in ("planned", "removed"):
            try:
                statuses = self._backend.status(lease)
            except InfrastructureError as error:
                if _recovery(error):
                    self._background(
                        env, lambda error=error: self._quarantine(env, error)
                    )
                raise SandboxError(
                    "infrastructure", "env_id", "env state could not be read"
                ) from None
        else:
            statuses = {}
        # Lock-free: DiskWatchdog replaces its size mappings whole, so a
        # status never waits for a disk poll in progress.
        usage = None if self._disk is None else self._disk.usage(env_id)
        for record in lease.services:
            status = statuses.get(record.name)
            if status is None:
                services[record.name] = {
                    "state": "removed" if lease.state == "removed" else "created",
                    "health": None,
                    "exit_code": None,
                    "oom_kills": env.ooms.get(record.name, 0),
                    "disk_mb_used": 0,
                    "started_at": None,
                    "diagnostics": None,
                }
                continue
            diagnostics = None
            if (
                state == "failed"
                or status.state == "exited"
                or (status.health == "unhealthy")
            ):
                try:
                    diagnostics = self._backend.diagnostics(lease, record.name)
                except Exception:
                    diagnostics = None
            services[record.name] = {
                "state": status.state,
                "health": status.health,
                "exit_code": status.exit_code,
                "oom_kills": self._oom(
                    env,
                    record.name,
                    record.container_id if status.state == "running" else None,
                ),
                "disk_mb_used": 0 if usage is None else usage.service_mb(record.name),
                "started_at": status.started_at,
                "diagnostics": diagnostics,
            }
        return {
            "env_id": env_id,
            "state": state,
            "reason": reason,
            "remaining_sec": remaining,
            "services": services,
        }

    def env_stop_service(
        self,
        credential: str,
        env_id: str,
        service: object,
        timeout_sec: object,
        request_id: str,
    ) -> dict:
        timeout = _seconds(timeout_sec, "timeout_sec", ceiling=60.0)
        with self._lock:
            session = self._session(credential, mutation=True)
            env = self._owned_env(session, env_id)
            fingerprint = ("env_stop_service", env_id, service, timeout)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            self._usable(env)
            if env.starter is not None or env.lease.state == "starting":
                raise SandboxError("busy", "env_id", "env is starting")
            self._charge(session, operations=1)
            env.uses += 1
            record = self._new_record(session, request_id, fingerprint, env_id)
        try:
            with env.lock:
                try:
                    _, code = self._backend.stop_service(
                        env.lease, self._commit(env), service, timeout_sec=timeout
                    )
                except InfrastructureError as error:
                    if self._broker.recovery_required:
                        raise
                    if not _recovery(error):
                        # A transient inspect failure: nothing to quarantine.
                        raise SandboxError(
                            "infrastructure",
                            "service",
                            "service could not be inspected",
                        ) from None
                    self._quarantine_locked(env, error)
                    raise SandboxError(
                        "infrastructure", "service", "stop unresolved; env quarantined"
                    ) from None
            self.pump.env_stopped(env_id, service=service)
            record.result = {"state": "exited", "exit_code": code}
            return record.result
        except BaseException as error:
            self._cache(record, error)
            raise
        finally:
            with self._lock:
                env.uses -= 1
                record.complete = True
                self._touch(env)

    def env_destroy(self, credential: str, env_id: str) -> dict:
        with self._lock:
            session = self._session(credential, mutation=True)
            env = self._owned_env(session, env_id)
            if env.lease.state == "removed":
                return {"state": "removed"}
            self._charge(session, operations=1)
        self._destroy(env, reason=None)
        return {"state": "removed"}

    def env_list(self, credential: str) -> dict:
        with self._lock:
            session = self._session(credential)
            return {
                "envs": [
                    {
                        "env_id": env.env_id,
                        "state": self._view_state(env),
                        "services": {
                            record.name: record.state for record in env.lease.services
                        },
                    }
                    for env in self._envs.values()
                    if env.session is session and env.lease.state != "removed"
                ],
                "jobs": [
                    {"job_id": job.job_id, "kind": job.kind, "state": job.state}
                    for job in self.images.jobs.values()
                    if job.session is session
                ],
            }

    # -- teardown ------------------------------------------------------------------

    def _join_starter(self, env: _Env, timeout: float) -> None:
        env.cancel.set()
        starter = env.starter
        if starter is not None and starter is not threading.current_thread():
            starter.join(max(0.0, timeout))

    def _destroy(
        self, env: _Env, *, reason: str | None, timeout: float = DELETE_SEC
    ) -> None:
        """Preempt starts and execs, then remove every object with proof."""
        end = time.monotonic() + timeout
        self._join_starter(env, timeout)
        self.pump.env_stopped(env.env_id, release=True)
        if not env.lock.acquire(timeout=max(0.0, end - time.monotonic())):
            self._broker._fail_closed()
            raise InfrastructureError(
                f"sandbox env {env.env_id} teardown is blocked; recovery required"
            )
        try:
            self._destroy_locked(env, reason=reason)
        finally:
            env.lock.release()

    def _destroy_locked(self, env: _Env, *, reason: str | None) -> None:
        if env.lease.state == "removed":
            with self._lock:
                self._removed(env)
            return
        self.pump.env_stopped(env.env_id, release=True)
        try:
            # Every service is killed with proof before anything is removed,
            # so close_judge's kill stage can rely on a removal still running.
            # The proof comes before the journal: when many envs end at once
            # (a cancelled run), their journal writes queue on the broker lock.
            self._backend.terminate(env.lease, lambda lease: lease)
            env.killed.set()
            self._backend.terminate(env.lease, self._commit(env))
            self._backend.destroy(env.lease, self._commit(env), reason=reason)
        except Exception as error:
            # Unprovable removal (or a journal failure): the run fails closed.
            self._broker._fail_closed()
            raise InfrastructureError(
                "sandbox env cleanup unresolved; recovery required"
            ) from error
        with self._lock:
            self._removed(env)

    def _removed(self, env: _Env) -> None:
        """Under the broker lock: refund live usage once removal is proven."""
        if env.plan is None:
            return
        self._refund(env.session, **env.charges)
        env.plan = None
        env.scopes.clear()
        self._forget_requests(env.session, env.env_id)
        # Never a Docker probe under the broker lock: the disk thread drops
        # the env's watchdog state on its next turn.
        self._disk_forget.append(env.env_id)
        self._touch(env)

    def _background(self, env: _Env, action: Callable[[], None]) -> None:
        with self._lock:
            if env.reaper is not None or env.closing or self._closed:
                return

            def run() -> None:
                try:
                    action()
                except Exception:
                    LOGGER.exception("sandbox env %s background action", env.env_id)
                finally:
                    with self._lock:
                        env.reaper = None
                        self._touch(env)

            env.reaper = threading.Thread(
                target=run, name="rsi-sandbox-env-reaper", daemon=True
            )
            try:
                env.reaper.start()
            except BaseException:
                env.reaper = None  # nothing began; a later turn retries
                raise

    # -- execs -----------------------------------------------------------------------

    def exec_start(
        self,
        credential: str,
        env_id: str,
        service: object,
        argv: object,
        cwd: object,
        env: object,
        user: object,
        timeout_sec: object,
        merge_stderr: object,
        request_id: str,
    ) -> dict:
        request = parse_exec_request(
            {
                "service": service,
                "argv": argv,
                "cwd": cwd,
                "env": env,
                "user": user,
                "timeout_sec": timeout_sec,
                "merge_stderr": merge_stderr,
            }
        )
        fingerprint = (
            "exec_start",
            env_id,
            json.dumps(request.model_dump(mode="json"), sort_keys=True),
        )
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            box = self._owned_env(session, env_id)
            self._usable(box)
            streams = 1 if request.merge_stderr else 2
            limit = min(
                session.env_grant.max_exec_output_bytes,
                self._remaining(session, "log_bytes") // streams,
            )
            if limit <= 0:
                raise SandboxError(
                    "quota", "max_log_bytes", "retained output budget exhausted"
                )
            reservation = dict(execs_running=1, log_bytes=limit * streams)
            self._charge(session, operations=1, **reservation)
            record = self._new_record(session, request_id, fingerprint, env_id)
            lease, deadline = box.lease, box.deadline
            # Until the process has started, freeze must wait (freeze_check):
            # runc refuses to start an exec in a paused container.
            box.uses += 1
        try:
            try:
                target = self._backend.exec_target(lease, request.service)
                exec_id = self.pump.start(
                    target,
                    request,
                    session=session.credentials,
                    env_id=env_id,
                    request_id=request_id,
                    output_limit=limit,
                    deadline=deadline,
                )
            except BaseException:
                with self._lock:
                    self._refund(session, **reservation)
                raise
            record.result = {"exec_id": exec_id}
            return record.result
        except InfrastructureError as error:
            if self._broker.recovery_required:
                self._cache(record, error)
                raise
            if _recovery(error):
                self._background(box, lambda error=error: self._quarantine(box, error))
                failure = SandboxError(
                    "infrastructure",
                    "env_id",
                    "env service identity changed; quarantined",
                )
            else:
                # A transient inspect failure mutated nothing: the env stays.
                failure = SandboxError(
                    "infrastructure", "env_id", "env service could not be inspected"
                )
            self._cache(record, failure)
            raise failure from None
        except BaseException as error:
            self._cache(record, error)
            raise
        finally:
            with self._lock:
                box.uses -= 1
                record.complete = True

    def _exec_finished(self, summary: ExecSummary) -> None:
        """Pump settlement: the running slot and the unused output reservation."""
        with self._lock:
            session = self._sessions.get(summary.session)
            if session is None:
                return
            streams = 1 if summary.merge_stderr else 2
            used = summary.stdout_total + summary.stderr_total
            unused = max(0, summary.output_limit * streams - used)
            self._refund(session, execs_running=1, log_bytes=unused)

    def exec_wait(
        self,
        credential: str,
        exec_id: str,
        stdout_offset: int,
        stderr_offset: int,
        wait_sec: object,
        max_bytes: int,
    ) -> dict:
        """The short read after the server's long-poll (never blocks here)."""
        _seconds(wait_sec, "wait_sec", ceiling=MAX_WAIT_SEC)
        with self._lock:
            session = self._session(credential)
        return self.pump.read(
            exec_id,
            session=session.credentials,
            stdout_offset=stdout_offset,
            stderr_offset=stderr_offset,
            max_bytes=max_bytes,
        )

    def exec_kill(
        self, credential: str, exec_id: str, signal: object, scope: object
    ) -> dict:
        with self._lock:
            session = self._session(credential, mutation=True)
            self._charge(session, operations=1)
        return self.pump.kill(
            exec_id, session=session.credentials, signal=signal, scope=scope
        )

    # -- stages and copies -----------------------------------------------------------

    def _spool(self, session: Any) -> _Spool:
        """Under the broker lock: the session's private stage spool."""
        spool = self._spools.get(session.credentials)
        if spool is None:
            owner = session.credentials.owner
            name = owner.phase if owner.round_id is None else f"judge-{owner.round_id}"
            root = self._spool_root / "stage" / f"{name}-{secrets.token_hex(4)}"
            stages = StageStore(root, clock=self._clock)
            spool = _Spool(stages, self._runtime.transfer(stages))
            self._spools[session.credentials] = spool
        return spool

    def stage_put(
        self,
        credential: str,
        stage_id: object,
        offset: object,
        final: object,
        sha256: object,
        request_id: str,
        payload: bytes,
    ) -> dict:
        digest = hashlib.sha256(payload).hexdigest()
        fingerprint = ("stage_put", stage_id, offset, final, sha256, digest)
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
        self._admit_spool("stage")
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            self._charge(session, operations=1, upload_bytes=len(payload))
            spool = self._spool(session)
            record = self._new_record(session, request_id, fingerprint, None)
            ceiling = session.env_grant.max_upload_bytes
        try:
            result = spool.stages.put(
                stage_id,
                offset,
                payload,
                final=final,
                sha256=sha256,
                max_bytes=ceiling,
            )
            record.handle = result["stage_id"]
            record.result = result
            return result
        except BaseException as error:
            self._cache(record, error)
            raise
        finally:
            with self._lock:
                record.complete = True

    def stage_get(
        self, credential: str, stage_id: object, offset: object, length: object
    ) -> bytes:
        """A copy_out stage was paid in full when it was made (so re-reading it
        costs nothing more); any other stage charges the bytes read."""
        with self._lock:
            session = self._session(credential)
            spool = self._spool(session)
            if type(length) is not int or length < 0:
                raise SandboxError("invalid", "length", "length must be 0..16 MiB")
            if type(stage_id) is str and stage_id in spool.prepaid:
                self._charge(session, operations=1)
                cap = 0
            else:
                cap = min(length, self._remaining(session, "download_bytes"))
                if length and not cap:
                    raise SandboxError(
                        "quota", "max_download_bytes", "download budget exhausted"
                    )
                self._charge(session, operations=1, download_bytes=cap)
        try:
            data = spool.stages.read(stage_id, offset, cap or length)
        except BaseException:
            with self._lock:
                self._refund(session, download_bytes=cap)
            raise
        if cap:
            with self._lock:
                self._refund(session, download_bytes=cap - len(data))
        return data

    def _transfer_env(self, session: Any, env_id: object) -> _Env:
        env = self._owned_env(session, env_id)
        self._usable(env)
        if env.lease.state not in _TRANSFERABLE_ENVS:
            raise SandboxError("busy", "env_id", f"env is {env.lease.state}")
        return env

    def _archive(self, env: _Env, service: object) -> Any:
        try:
            return self._backend.archive_target(env.lease, service)
        except InfrastructureError as error:
            if _recovery(error):
                self._background(env, lambda error=error: self._quarantine(env, error))
                raise SandboxError(
                    "infrastructure", "service", "service identity changed; quarantined"
                ) from None
            raise SandboxError(
                "infrastructure", "service", "service could not be inspected"
            ) from None

    def copy_in(
        self,
        credential: str,
        env_id: str,
        service: object,
        dest_dir: object,
        stage_id: object,
        request_id: str,
    ) -> dict:
        fingerprint = ("copy_in", env_id, service, dest_dir, stage_id)
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            env = self._transfer_env(session, env_id)
            self._charge(session, operations=1)
            spool = self._spool(session)
            env.uses += 1
            record = self._new_record(session, request_id, fingerprint, env_id)
        try:
            result = spool.transfer.copy_in(
                self._archive(env, service), dest_dir, stage_id
            )
            record.result = result
            return result
        except BaseException as error:
            self._cache(record, error)
            raise
        finally:
            with self._lock:
                env.uses -= 1
                record.complete = True
                if type(stage_id) is str:
                    self._forget_requests(session, stage_id)
                    spool.prepaid.pop(stage_id, None)

    def copy_out(
        self,
        credential: str,
        env_id: str,
        service: object,
        path: object,
        max_bytes: object,
        exclude: object,
    ) -> dict:
        """The stage is charged to download_bytes when it is made, not when it
        is read: unread stages can never exceed the download budget on the
        spool (S6)."""
        if type(max_bytes) is not int or max_bytes < 0:
            raise SandboxError("invalid", "max_bytes", "expected a nonnegative limit")
        with self._lock:
            session = self._session(credential, mutation=True)
            self._transfer_env(session, env_id)
        self._admit_spool("path")
        with self._lock:
            session = self._session(credential, mutation=True)
            env = self._transfer_env(session, env_id)
            limit = min(max_bytes, self._remaining(session, "download_bytes"))
            self._charge(session, operations=1, download_bytes=limit)
            spool = self._spool(session)
            env.uses += 1
        try:
            result = spool.transfer.copy_out(
                self._archive(env, service), path, max_bytes=limit, exclude=exclude
            )
            size = spool.stages.size(result["stage_id"])
        except BaseException:
            with self._lock:
                self._refund(session, download_bytes=limit)
            raise
        finally:
            with self._lock:
                env.uses -= 1
        with self._lock:
            # Settle the reservation on the stage's tar size (headers too).
            self._refund(session, download_bytes=limit)
            try:
                self._charge(session, download_bytes=size)
            except SandboxError:
                refused = True
            else:
                refused = False
                spool.prepaid[result["stage_id"]] = size
        if refused:
            spool.stages.discard(result["stage_id"])
            raise SandboxError(
                "quota", "max_download_bytes", "the archive exceeds the download budget"
            )
        return result

    def path_stat(
        self,
        credential: str,
        env_id: str,
        service: object,
        path: object,
        follow: object,
    ) -> dict:
        with self._lock:
            session = self._session(credential, mutation=True)
            env = self._owned_env(session, env_id)
            self._usable(env)
            self._charge(session, operations=1)
            spool = self._spool(session)
            env.uses += 1
        try:
            return spool.transfer.path_stat(
                self._archive(env, service), path, follow=follow
            )
        finally:
            with self._lock:
                env.uses -= 1

    def tool_install(
        self, credential: str, env_id: str, service: object, tool: object
    ) -> dict:
        """Copy the operator's pinned ``tool`` into ``service`` as
        /usr/local/bin/<tool> (root, 0755) unless that path exists.

        Two keys: the operator's policy names the file and its SHA-256
        (``[environments.host.tmux]``), the caller asks for an env of its
        own. The broker reads the file and stages the bytes it hashed for
        the ordinary archive copy: a mismatch copies nothing. Whether the
        tool is already on the image's PATH (``command -v``) is the caller's
        check; an existing file at the path is never replaced, so a repeat
        answers ``installed: false``.
        """
        source = tool_file(self._broker.host_policy, tool)
        path = f"{TOOL_DIR}/{tool}"
        with self._lock:
            session = self._session(credential, mutation=True)
            self._transfer_env(session, env_id)
        self._admit_spool("tool")
        with self._lock:
            session = self._session(credential, mutation=True)
            env = self._transfer_env(session, env_id)
            self._charge(session, operations=1)
            spool = self._spool(session)
            env.uses += 1
        stage_id = None
        try:
            target = self._archive(env, service)
            if spool.transfer.path_stat(target, path, follow=False)["exists"]:
                return {"tool": tool, "path": path, "installed": False}
            archive = tool_tar(tool, read_tool(source))
            # The operator's bytes, not the caller's: no upload_bytes.
            stage_id = spool.stages.put(
                None,
                0,
                archive,
                final=True,
                sha256=hashlib.sha256(archive).hexdigest(),
                max_bytes=len(archive),
            )["stage_id"]
            spool.transfer.copy_in(target, TOOL_DIR, stage_id)
            return {"tool": tool, "path": path, "installed": True}
        finally:
            if stage_id is not None:
                spool.stages.discard(stage_id)  # a copy_in refused before use
            with self._lock:
                env.uses -= 1

    # -- images and jobs ---------------------------------------------------------------

    def image_pull(
        self, credential: str, ref: object, policy: object, request_id: str
    ) -> dict:
        if policy not in PULL_POLICIES:
            raise SandboxError("invalid", "policy", "expected missing or always")
        repository, tag = pull_reference(ref)
        fingerprint = ("image_pull", repository, tag, policy)
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            grant = session.env_grant
            if not grant.pull:
                raise SandboxError("permission", "ref", "image pull not granted")
            if registry_of(repository) not in grant.registries:
                raise SandboxError("permission", "ref", "registry not approved")
        self._admit_disk("ref")
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            self.images.admit_handle(session)
            # A "missing" pull of an image already on the host downloads
            # nothing and spends no budget, so it is admitted even when the
            # budget is spent; the job refuses one that would download.
            if policy == "always" and not self._remaining(session, "pull_bytes"):
                raise SandboxError("quota", "max_pull_mb", "pull budget exhausted")
            self._charge(session, operations=1, jobs_running=1)
            handle = "i" + secrets.token_hex(16)
            lease = SandboxImageLease(
                owner=session.credentials.owner, handle=handle, kind="pulled"
            )
            try:
                self._broker.journal.plan_image(lease)
            except Exception as error:
                self._refund(session, jobs_running=1)
                raise self._journal_failure(error) from error
            job = self.images.submit(
                session,
                "pull",
                handle,
                job_deadline(session),
                (repository, tag, policy, lease),
                self.images.run_pull,
            )
            record = self._new_record(session, request_id, fingerprint, job.job_id)
            record.result = {"job_id": job.job_id}
            record.complete = True
        return {"job_id": job.job_id}

    def image_build(
        self,
        credential: str,
        stage_id: object,
        dockerfile: object,
        dockerfile_inline: object,
        target: object,
        build_args: object,
        labels: object,
        no_cache: object,
        network: object,
        timeout_sec: object,
        request_id: str,
    ) -> dict:
        """Admit one build (spec 4): the stage is consumed into the builder
        input now, so a malformed context, Dockerfile or directive is this
        call's error; the build itself is a job. An identical build (B9)
        whose image this session still binds is answered by that image."""
        options = _build_options(
            dockerfile, dockerfile_inline, target, build_args, labels, no_cache
        )
        if type(stage_id) is not str or STAGE_ID.fullmatch(stage_id) is None:
            raise SandboxError("invalid", "stage_id", "expected an s<32hex> stage")
        if network not in ("public", "none"):
            raise SandboxError("invalid", "network", "expected public or none")
        if (
            type(timeout_sec) not in (int, float)
            or not math.isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise SandboxError(
                "invalid", "timeout_sec", "expected finite positive seconds"
            )
        canonical = json.dumps(options, sort_keys=True)
        fingerprint = (
            "image_build",
            stage_id,
            hashlib.sha256(canonical.encode()).hexdigest(),
            network,
            float(timeout_sec),
        )
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            build = session.env_grant.build
            if self.builds is None and self._runtime.backend_name != "docker":
                raise SandboxError(
                    "unsupported",
                    "build",
                    f"image_build is unsupported on the {self._runtime.backend_name} "
                    "backend",
                )
            if build is None or self.builds is None:
                raise SandboxError("permission", "build", "image build not granted")
            if network not in build.network:
                raise SandboxError(
                    "permission", "network", "build network not granted to phase"
                )
            if not self._remaining(session, "builds"):
                raise SandboxError("quota", "max_builds", "build budget exhausted")
            self.images.admit_handle(session)
        self._admit_spool("stage_id")
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._record(session, request_id, fingerprint)
            if record is not None:
                return record.result
            self._charge(session, operations=1)
            spool = self._spool(session)
            record = self._new_record(session, request_id, fingerprint, None)
        directory = secrets.token_hex(16)
        path: Path | None = None
        try:
            path = self.builds.input_path(directory)
            with spool.stages.consume(stage_id) as (stage, _summary):
                descriptor = os.open(
                    path,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                    0o600,
                )
                with open(descriptor, "w+b") as dest:
                    staged = prepare_build_input(
                        stage,
                        dest,
                        job_id=directory,
                        dockerfile=options["dockerfile"],
                        dockerfile_inline=options["dockerfile_inline"],
                        max_bytes=build.max_context_mb * MIB,
                        syntax_frontends=build.syntax_frontends,
                    )
            key = build_fingerprint(
                staged.digest,
                {
                    key: value
                    for key, value in options.items()
                    if key not in ("dockerfile", "dockerfile_inline")
                }
                | {"network": network},
            )
            with self._lock:
                session = self._session(credential, mutation=True)
                existing = self.images.find_built(session, key)
                if existing is not None:
                    path.unlink(missing_ok=True)
                    job = self.images.ended(
                        session,
                        "build",
                        existing.handle,
                        {"image": image_view(existing.attrs, existing.handle)},
                    )
                else:
                    job = self._submit_build(
                        session, staged, path, options, network, key, timeout_sec
                    )
                record.handle = job.job_id
                record.result = {"job_id": job.job_id}
            return record.result
        except BaseException as error:
            if path is not None:
                path.unlink(missing_ok=True)
            self._cache(record, error)
            raise
        finally:
            with self._lock:
                record.complete = True
                self._forget_requests(session, stage_id)
                spool.prepaid.pop(stage_id, None)

    def _submit_build(
        self,
        session: Any,
        staged: Any,
        path: Path,
        options: Mapping[str, Any],
        network: str,
        fingerprint: str,
        timeout: float,
    ) -> Any:
        """Under the broker lock: charge, journal the planned image, start."""
        grant = session.env_grant.build
        self.images.admit_handle(session)
        self._charge(session, builds=1, jobs_running=1)
        handle = "i" + secrets.token_hex(16)
        lease = SandboxImageLease(
            owner=session.credentials.owner,
            handle=handle,
            kind="built",
            tag=None,
        )
        try:
            self._broker.journal.plan_image(lease)
        except Exception as error:
            self._refund(session, jobs_running=1)
            raise self._journal_failure(error) from error
        request = BuildRequest(
            input=staged,
            input_path=path,
            target=options["target"],
            build_args=dict(options["build_args"]),
            labels=dict(options["labels"]),
            no_cache=options["no_cache"],
            network=network,
            fingerprint=fingerprint,
            lease=lease,
        )
        deadline = build_deadline(session, float(timeout), grant, self._clock())
        return self.images.submit(
            session, "build", handle, deadline, request, self.builds.run
        )

    def job_wait(
        self, credential: str, job_id: str, log_offset: object, wait_sec: object
    ) -> dict:
        _seconds(wait_sec, "wait_sec", ceiling=MAX_WAIT_SEC)
        with self._lock:
            session = self._session(credential)
            return self.images.view(session, job_id, log_offset)

    def job_cancel(self, credential: str, job_id: str) -> dict:
        with self._lock:
            session = self._session(credential, mutation=True)
            job = self.images.owned_job(session, job_id)
            self._charge(session, operations=1)
        return {"state": self.images.cancel(job)}

    def image_list(self, credential: str) -> dict:
        with self._lock:
            session = self._session(credential)
            return {"images": self.images.listing(session)}

    def image_in_use(self, image: Any) -> bool:
        return any(
            env.session is image.session
            and env.lease.state != "removed"
            and image.handle in env.images
            for env in self._envs.values()
        )

    def image_release(self, credential: str, image: object) -> dict:
        """Pulled: the handle is unbound (the image stays, S9). Built: the
        handle is unbound first, so no env can take it, then the image is
        untagged and removed; an rmi conflict keeps it journaled as leaked
        until the session ends."""
        with self._lock:
            session = self._session(credential, mutation=True)
            record = self._owned_image(session, image, "image")
            if self.image_in_use(record):
                raise SandboxError(
                    "busy", "image", "a live env of this session uses it"
                )
            self._charge(session, operations=1)
            if record.kind == "pulled":
                self.images.unbind_pulled(record)
                return {"ok": True}
            del self.images.images[record.handle]
        self.builds.release(record)
        return {"ok": True}

    # -- long-poll ---------------------------------------------------------------------

    def wait_condition(
        self, credential: str, operation: str, metadata: Mapping[str, Any]
    ) -> Callable[[], bool] | None:
        """None: answer now. Otherwise a lock-free check the server polls
        outside any worker slot until it turns true or the wait ends.

        exec_wait waits while the exec runs with nothing unread, job_wait
        while the job runs with nothing unread, env_status while the env is
        still changing (for example starting).
        """
        wait = _seconds(metadata.get("wait_sec"), "wait_sec", ceiling=MAX_WAIT_SEC)
        if not wait:
            return None
        with self._lock:
            session = self._session(credential)
            if operation == "exec_wait":
                exec_id = metadata.get("exec_id")
                if not self.pump.idle(
                    exec_id,
                    session=session.credentials,
                    stdout_offset=metadata.get("stdout_offset"),
                    stderr_offset=metadata.get("stderr_offset"),
                ):
                    return None
                start = self.pump.generation(exec_id)
                return lambda: session.revoked or self.pump.generation(exec_id) != start
            if operation == "job_wait":
                job = self.images.owned_job(session, metadata.get("job_id"))
                offset = metadata.get("log_offset")
                if job.state in ENDED_JOB or offset != len(job.log):
                    return None
                item = job
            elif operation == "env_status":
                item = self._owned_env(session, metadata.get("env_id"))
                # Only a changing env is waited on: created, starting, being
                # quarantined or removed. A settled one answers at once.
                if (
                    self._view_state(item) not in ("created", "starting", "stopping")
                    and item.reaper is None
                ):
                    return None
            else:
                return None
            start = item.generation
        return lambda: session.revoked or item.generation != start

    # -- phase barriers ----------------------------------------------------------------

    def _family(self, session: Any) -> list[_Env]:
        return [
            env
            for env in self._envs.values()
            if env.session is session and env.lease.state != "removed"
        ]

    def retained(self, session: Any) -> bool:
        with self._lock:
            return bool(self._family(session)) or (
                self.builds is not None and self.builds.retained(session)
            )

    def freeze_check(self, session: Any) -> None:
        """Under the broker lock, before Work freezes (S6). An idle Work
        builder keeps running: it is broker infrastructure without caller
        code, and a build in flight is a pending job here."""
        busy = self.images.pending(session) + [
            env
            for env in self._family(session)
            if env.starter is not None
            or env.uses
            or env.reaper is not None
            or env.lease.state in ("planned", "starting")
        ]
        if busy:
            raise RetryableSubmissionError(
                "sandbox environment job busy; retry submission after it completes"
            )

    def freeze(self, session: Any) -> None:
        """Pause every Work env; exec clocks stop until ``thaw``.

        An env that cannot be paused is quarantined (killed) instead: either
        way nothing of it executes while Judge runs. Only an env that can be
        neither paused nor killed propagates, for the broker to fail closed.
        """
        self.pump.freeze(session.credentials)
        with self._lock:
            family = self._family(session)
        for env in family:
            with env.lock:
                if env.lease.state in ("stopping", "removed"):
                    continue
                try:
                    self._backend.pause(env.lease, self._commit(env))
                except Exception as error:
                    if self._broker.recovery_required:
                        raise
                    self._quarantine_locked(env, error)
                    if self._broker.recovery_required:
                        raise InfrastructureError(
                            "sandbox env freeze failed; recovery required"
                        ) from error

    def thaw(self, session: Any, admission: Callable[[], Any]) -> None:
        """Resume every paused Work env; one that expired meanwhile is removed
        instead, never thawed."""
        with self._lock:
            family = self._family(session)
        now = self._clock()
        for env in family:
            if now >= env.deadline:
                self._destroy(env, reason="expired")
                continue
            with env.lock:
                if not any(record.state == "paused" for record in env.lease.services):
                    continue
                self._backend.resume(env.lease, self._commit(env), admission=admission)
        self.pump.thaw(session.credentials)

    def contain(self, session: Any) -> list[BaseException]:
        self.pump.freeze(session.credentials)
        with self._lock:
            family = self._family(session)
        errors: list[BaseException] = []
        for env in family:
            try:
                if not env.lock.acquire(timeout=KILL_SEC):
                    raise InfrastructureError("sandbox env busy; cannot contain it")
                try:
                    self._backend.pause(env.lease, self._commit(env))
                finally:
                    env.lock.release()
            except Exception as error:
                errors.append(error)
        return errors

    def repause(self, session: Any) -> None:
        """Best effort after a failed resume; the broker already failed closed."""
        self.pump.freeze(session.credentials)
        with self._lock:
            family = self._family(session)
        for env in family:
            try:
                if env.lock.acquire(timeout=KILL_SEC):
                    try:
                        self._backend.pause(env.lease, self._commit(env))
                    finally:
                        env.lock.release()
            except Exception:
                pass

    # -- sweep -------------------------------------------------------------------------

    def _dead(self, env: _Env, now: float) -> bool:
        return now >= env.deadline or self._session_dead(env.session, now)

    def _session_dead(self, session: Any, now: float) -> bool:
        broker = self._broker
        return (
            session.revoked
            or broker._cancelled
            or broker._cancel_requested.is_set()
            or (session.deadline is not None and now >= session.deadline)
            or (
                session.credentials.owner.phase == "work"
                and broker._run_deadline is not None
                and now >= broker._run_deadline
            )
        )

    @staticmethod
    def _frozen(env: _Env) -> bool:
        """Paused by freeze_work: it executes nothing, and resume or close()
        ends it. A fail-closed broker freezes every session without pausing
        anything, so that alone never retains an env (as v1's
        ``_retired_paused``)."""
        return env.session.frozen and (
            env.lease.state == "paused"
            or any(record.state == "paused" for record in env.lease.services)
        )

    def sweep(self) -> None:
        """Watchdog turn: expiry, job deadlines and stage TTLs; the disk
        watchdog polls on its own thread so a slow daemon never delays
        deadlines. Housekeeping never fails the run closed: its errors are
        logged and the next turn retries. Journal failures and unprovable
        removals fail closed where they happen (in the reapers)."""
        now = self._clock()
        with self._lock:
            if self._closed:
                return
            expired = [
                env
                for env in self._envs.values()
                if env.lease.state != "removed"
                and env.reaper is None
                and self._dead(env, now)
                and not self._frozen(env)
            ]
            late = self.images.expire(now)
            spools = list(self._spools.values())
            stale = [env for env in self._envs.values() if self._refresh_due(env, now)]
            for env in stale:
                env.refreshed_at = now
        for job in late:
            self.images.close(job)
        if self.builds is not None:
            self._housekeep(
                "builder expiry",
                lambda: self.builds.sweep(
                    lambda session: self._session_dead(session, now)
                ),
            )
        reason = "canceled" if self._broker._cancelled else "expired"
        for env in expired:
            self._housekeep(
                "env expiry",
                lambda env=env: self._background(
                    env, lambda: self._destroy(env, reason=reason)
                ),
            )
        for spool in spools:
            self._housekeep(
                "stage sweep", lambda spool=spool: self._sweep_stages(spool)
            )
        if self._disk is not None:
            self._housekeep("disk watchdog", self._start_disk_sweep)
        if stale:
            self._housekeep("allowlist refresh", lambda: self._start_refresh(stale))

    @staticmethod
    def _housekeep(what: str, action: Callable[[], None]) -> None:
        try:
            action()
        except Exception as error:
            LOGGER.warning("sandbox env %s failed; retrying: %s", what, error)

    def _sweep_stages(self, spool: _Spool) -> None:
        stale = spool.stages.sweep()
        if stale:
            with self._lock:
                for stage_id in stale:
                    spool.prepaid.pop(stage_id, None)

    def _start_disk_sweep(self) -> None:
        with self._lock:
            if self._closed or (
                self._disk_thread is not None and self._disk_thread.is_alive()
            ):
                return
            self._disk_thread = threading.Thread(
                target=self._disk_sweep, name="rsi-sandbox-env-disk", daemon=True
            )
            self._disk_thread.start()

    def _disk_sweep(self) -> None:
        """One disk watchdog turn; DiskWatchdog paces its own probes."""
        with self._lock:
            forget, self._disk_forget = self._disk_forget, []
            live = [
                env
                for env in self._envs.values()
                if env.lease.state not in ("planned", "removed")
            ]
            budgets = [
                EnvDiskBudget(
                    env.env_id,
                    env.lease.disk_mb,
                    env.session.env_grant.disk_mb_per_container,
                )
                for env in live
            ]
        try:
            with self._disk_lock:
                for env_id in forget:
                    self._disk.forget(env_id)
                if not live:
                    return
                verdict = self._disk.poll(budgets)
        except Exception as error:
            # A daemon hiccup must not fail the run closed; the next turn retries.
            LOGGER.warning("sandbox env disk poll failed: %s", error)
            return
        by_id = {env.env_id: env for env in live}
        doomed = set(verdict.reclaim) | set(verdict.hard_floor)
        for env_id in doomed:
            env = by_id.get(env_id)
            if env is not None and not self._frozen(env):
                self._housekeep(
                    "disk reclaim",
                    lambda env=env: self._background(
                        env, lambda: self._destroy(env, reason="disk_quota")
                    ),
                )
        for env_id in verdict.over_quota:
            env = by_id.get(env_id)
            if (
                env is None
                or env_id in doomed
                or self._frozen(env)
                or env.lease.reason == "disk_quota"
            ):
                continue
            self._housekeep(
                "disk kill",
                lambda env=env: self._background(env, lambda: self._fail_disk(env)),
            )

    # -- allowlist refresh -----------------------------------------------------------

    def _refresh_due(self, env: _Env, now: float) -> bool:
        """Under the broker lock: a live allowlist env past its refresh_sec."""
        return (
            env.plan is not None
            and env.plan.network_mode == "allowlist"
            and env.lease.state in ("created", "starting", "ready", "paused", "failed")
            and env.reaper is None
            and not (env.closing or env.quarantined or self._frozen(env))
            and now - env.refreshed_at >= env.session.env_grant.allowlist.refresh_sec
        )

    def _start_refresh(self, envs: list[_Env]) -> None:
        """One thread resolves (DNS may block for seconds, never the
        watchdog); a turn still running skips this one."""
        with self._lock:
            if self._closed or (
                self._refresh_thread is not None and self._refresh_thread.is_alive()
            ):
                return
            self._refresh_thread = threading.Thread(
                target=lambda: [self._refresh(env) for env in envs],
                name="rsi-sandbox-env-allowlist",
                daemon=True,
            )
            self._refresh_thread.start()

    def _refresh(self, env: _Env) -> None:
        with self._lock:
            plan = env.plan
            if self._closed or plan is None or env.reaper is not None:
                return
        try:
            if self._backend.refresh_network(plan):
                LOGGER.info("sandbox env %s allowlist accepts changed", env.env_id)
            return
        except Exception as error:
            failure = error
        with self._lock:
            ended = env.plan is None or env.reaper is not None or env.closing
        if not ended:
            # Never opened beyond the old and new accepts, but no longer the
            # attested set: stop the env (S8).
            self._housekeep(
                "allowlist quarantine",
                lambda: self._background(env, lambda: self._quarantine(env, failure)),
            )

    def join_refresh(self, timeout: float | None = None) -> None:
        """Wait for an allowlist refresh turn in progress (tests)."""
        thread = self._refresh_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def join_disk(self, timeout: float | None = None) -> None:
        """Wait for a disk watchdog turn in progress (tests, close)."""
        thread = self._disk_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _fail_disk(self, env: _Env) -> None:
        self._join_starter(env, KILL_SEC)
        with env.lock:
            if env.lease.state not in ("created", "starting", "ready", "failed"):
                return
            self.pump.env_stopped(env.env_id)
            try:
                self._backend.fail(env.lease, self._commit(env), "disk_quota")
                env.killed.set()
            except SandboxError:
                pass
            except Exception as error:
                if not self._broker.recovery_required:
                    LOGGER.error(
                        "sandbox env %s disk kill failed: %s", env.env_id, error
                    )
                    self._broker._fail_closed()

    # -- session end -----------------------------------------------------------------

    def kill_session(self, session: Any) -> float:
        """Stage one of a session's end: nothing of it executes after KILL_SEC
        plus KILL_SEC_PER_CONTAINER for each of its service containers.

        Starters, pulls and builds are cancelled, execs end, the builder and
        every service are SIGKILLed (paused ones through the paused killer)
        and proven stopped, without touching the journal. Returns the
        deadline for ``remove_session`` (DELETE_SEC plus DELETE_SEC_PER_ENV
        for each env). Anything unproven fails the run closed.
        """
        begin = time.monotonic()
        with self._lock:
            family = self._family(session)
            for env in family:
                env.closing = True
        containers = sum(len(env.lease.services) for env in family)
        kill_end = begin + KILL_SEC + KILL_SEC_PER_CONTAINER * containers
        self.images.cancel_session(session)
        errors: list[BaseException] = []
        if self.builds is not None:
            # No RUN step executes once its builder is proven stopped.
            errors.extend(self.builds.kill_session(session, kill_end))
        for env in family:
            self._join_starter(env, kill_end - time.monotonic())
            # A removal, quarantine or disk kill the watchdog already began
            # kills before it removes: its kill proof is enough here, and
            # remove_session waits for the rest within DELETE_SEC.
            reaper = env.reaper
            if reaper is not None and reaper is not threading.current_thread():
                while (
                    reaper.is_alive()
                    and not env.killed.is_set()
                    and time.monotonic() < kill_end
                ):
                    env.killed.wait(0.01)
            self.pump.env_stopped(env.env_id)

        def kill(env: _Env) -> None:
            starter, reaper = env.starter, env.reaper
            if starter is not None and starter.is_alive():
                raise InfrastructureError("sandbox env start did not settle")
            if reaper is not None and reaper.is_alive():
                if not env.killed.is_set():
                    raise InfrastructureError("sandbox env did not settle")
                return
            if env.lease.state == "removed":
                return
            remaining = max(0.0, kill_end - time.monotonic())
            self._backend.terminate(
                env.lease, lambda lease: lease, deadline=self._clock() + remaining
            )
            env.killed.set()

        errors.extend(_each_env(family, kill))
        if errors:
            self._broker._fail_closed()
            raise InfrastructureError(
                "sandbox session execution could not be ended; recovery required"
            ) from errors[0]
        return begin + DELETE_SEC + DELETE_SEC_PER_ENV * len(family)

    def remove_session(self, session: Any, deadline: float) -> None:
        """Stage two: every env removed with proof by ``deadline`` (monotonic),
        then built images and the builder, pulled handles unbound, stages and
        exec records dropped."""
        with self._lock:
            family = self._family(session)
            jobs = [
                job
                for job in self.images.jobs.values()
                if job.session is session and job.kind == "pull"
            ]
        # remove() may run on a pool thread: never join the caller itself.
        caller = threading.current_thread()

        def remove(env: _Env) -> None:
            reaper = env.reaper
            if reaper is not None and reaper is not caller:
                reaper.join(max(0.0, deadline - time.monotonic()))
            self._destroy(
                env, reason="canceled", timeout=max(0.0, deadline - time.monotonic())
            )

        errors = _each_env(family, remove)
        if self.builds is not None:
            # After the envs: none of them may still run a built image.
            errors.extend(self.builds.remove_session(session, deadline))
        for job in jobs:
            if job.thread is not None and job.thread is not threading.current_thread():
                # A pull executes nothing; it only must not bind a handle late.
                job.thread.join(max(0.0, min(KILL_SEC, deadline - time.monotonic())))
        with self._lock:
            for image in self.images.session_images(session):
                try:
                    self.images.unbind_pulled(image)
                except Exception as error:
                    errors.append(error)
            spool = self._spools.pop(session.credentials, None)
        self.pump.close_session(session.credentials)
        if spool is not None:
            spool.stages.close()
        if errors or time.monotonic() > deadline:
            self._broker._fail_closed()
            raise InfrastructureError(
                "sandbox session cleanup requires recovery"
            ) from (errors[0] if errors else None)

    def close_session(self, session: Any) -> None:
        self.remove_session(session, self.kill_session(session))

    def close(self) -> None:
        with self._lock:
            sessions = []
            for env in self._envs.values():
                if env.session not in sessions:
                    sessions.append(env.session)
            for session in self._sessions.values():
                if session not in sessions:
                    sessions.append(session)
        errors: list[BaseException] = []
        for session in sessions:
            try:
                self.close_session(session)
            except Exception as error:
                errors.append(error)
        with self._lock:
            self._closed = True
            spools = list(self._spools.values())
            self._spools.clear()
        for spool in spools:
            spool.stages.close()
        self.pump.close()
        if errors:
            raise InfrastructureError(
                "sandbox environment cleanup requires recovery"
            ) from errors[0]


__all__ = [
    "DELETE_SEC",
    "DELETE_SEC_PER_ENV",
    "KILL_SEC",
    "KILL_SEC_PER_CONTAINER",
    "EnvRuntime",
    "SandboxEnvs",
    "docker_env_runtime",
]
