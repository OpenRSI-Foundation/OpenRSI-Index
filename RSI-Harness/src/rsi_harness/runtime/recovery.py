"""Durable run resource leases, crash recovery, and contained cleanup."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from pydantic import Field, field_validator, model_validator

from rsi_harness.errors import InfrastructureError
from rsi_harness.models import (
    ManagedWorkdirVolume,
    PersistedModel,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunStatus,
)
from rsi_harness.runtime.durable import durable_mkdir, fsync_directory
from rsi_harness.runtime.image_authority import is_immutable_image_ref
from rsi_harness.runtime.redaction import redact_structure, redact_text
from rsi_harness.runtime.sandbox_contracts import (
    MAX_ENV_SERVICES,
    EnvE2BHost,
    SandboxLease,
    SandboxOwner,
    SandboxReservation,
)
from rsi_harness.runtime.sandbox_docker import attest_sandbox_identity, sandbox_labels
from rsi_harness.runtime.sandbox_env_contracts import (
    BUILD_ROLE,
    BUILDER_NETWORK_ROLE,
    BUILDER_ROLE,
    BUILDER_VOLUME_ROLE,
    BUILT_IMAGE_REPOSITORY,
    ENV_NETWORK_ROLE,
    ENV_ROLE,
    ENV_VOLUME_ROLE,
    LABEL_PREFIX,
    MAX_ENV_VOLUME_LEASES,
    BuilderLease,
    SandboxEnvLease,
    SandboxEnvServiceLease,
    SandboxImageLease,
    builder_container_name,
    builder_loop_file,
    builder_network_name,
    builder_rule_id,
    builder_volume_name,
    built_image_tag,
    env_container_labels,
    env_container_name,
    env_network_name,
    env_rule_id,
    env_volume_labels,
    env_volume_name,
    sandbox_object_labels,
    sandbox_spool_root,
    short_identity,
)
from rsi_harness.runtime.sandbox_env_docker import _in_flight, _InFlight
from rsi_harness.runtime.sandbox_lifecycle import ENDPOINT_MODULES
from rsi_harness.runtime.workdir_volume import (
    attest_normalized_workdir_volume_state,
    managed_workdir_volume_labels,
)

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_RUN_LABEL = "rsi-harness.run-id"
_TASK_LABEL = "rsi-harness.task-id"
_ROLE_LABEL = "rsi-harness.role"
_ROUND_LABEL = "rsi-harness.round-id"
_SOURCE_CONTAINER_LABEL = "rsi-harness.source-container-id"
_ROUND_IMAGE_ROLE = "rootfs-snapshot"
_RETAINED_IMAGE_ROLE = "retained-work-rootfs"
_RETAINED_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RETAINED_IMAGE_REF = re.compile(r"rsi-harness-rootfs:retained-work-[0-9a-f]{64}\Z")
_JUDGE_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_JUDGE_IMAGE_REF = re.compile(r"rsi-harness-rootfs:judge-round-[0-9a-f]{64}\Z")
_PHASE_LABEL = "rsi-harness.sandbox-phase"
_ENV_LABEL = "rsi-harness.sandbox-env"
_ENV_ID = re.compile(r"e[0-9a-f]{32}\Z")
_BUILDER_LABEL = "rsi-harness.sandbox-builder"
_BUILDER_ID = re.compile(r"b[0-9a-f]{32}\Z")
_LOOP_DEVICE = re.compile(r"/dev/loop[0-9]{1,6}\Z")
_LOOP_FILE = re.compile(r"[0-9a-f]{16}\.img\Z")
# A phase endpoint directory under ``sb`` (SandboxLifecycle._prepare).
_ENDPOINT_DIRECTORY = re.compile(r"[0-9a-f]{8}\Z")
_DOCKER_ID = re.compile(r"[0-9a-f]{64}\Z")
LOGGER = logging.getLogger(__name__)


def _absolute_optional_path(value: Path | None) -> Path | None:
    if value is not None and not value.is_absolute():
        raise ValueError("lease paths must be absolute")
    return value


class RetainedImageRollbackAuthority(PersistedModel):
    """Exact acquired image authority awaiting rollback/recovery."""

    image_id: str
    image_ref: str

    @field_validator("image_id")
    @classmethod
    def valid_image_id(cls, value: str) -> str:
        if _RETAINED_IMAGE_ID.fullmatch(value) is None:
            raise ValueError("invalid retained Work rollback image ID")
        return value

    @field_validator("image_ref")
    @classmethod
    def valid_image_ref(cls, value: str) -> str:
        if _RETAINED_IMAGE_REF.fullmatch(value) is None:
            raise ValueError("invalid retained Work rollback image reference")
        return value


class WorkdirVolumeResourceLease(PersistedModel):
    """Durable planned, actual, and mounted WORKDIR-volume authority."""

    planned_name: str | None = None
    planned_target: PurePosixPath | None = None
    planned_snapshot_mode: RootfsSnapshotMode | None = None
    planned_freshness_nonce: str | None = None
    actual: ManagedWorkdirVolume | None = None
    rollback: ManagedWorkdirVolume | None = None
    mounted: bool = False

    @field_validator("planned_name")
    @classmethod
    def safe_planned_name(cls, value: str | None) -> str | None:
        if value is not None and _SAFE_ID.fullmatch(value) is None:
            raise ValueError("unsafe planned WORKDIR volume name")
        return value

    @field_validator("planned_target", mode="after")
    @classmethod
    def valid_planned_target(cls, value: PurePosixPath | None) -> PurePosixPath | None:
        if value is None:
            return None
        if not value.is_absolute():
            raise ValueError("planned WORKDIR volume target must be absolute")
        if value == PurePosixPath("/"):
            raise ValueError("planned WORKDIR volume requires a non-root target")
        if ".." in value.parts:
            raise ValueError("planned WORKDIR volume target contains lexical traversal")
        return value

    @field_validator("planned_snapshot_mode")
    @classmethod
    def valid_planned_snapshot_mode(
        cls, value: RootfsSnapshotMode | None
    ) -> RootfsSnapshotMode | None:
        if value is not None and value is not RootfsSnapshotMode.SPLIT_WORKDIR:
            raise ValueError("planned WORKDIR volume mode must be split-workdir")
        return value

    @field_validator("planned_freshness_nonce")
    @classmethod
    def valid_planned_freshness_nonce(cls, value: str | None) -> str | None:
        if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(
                "planned WORKDIR volume freshness nonce must be "
                "64 lowercase hex characters"
            )
        return value

    @model_validator(mode="after")
    def consistent_authority(self) -> WorkdirVolumeResourceLease:
        planned_fields = (
            self.planned_name,
            self.planned_target,
            self.planned_snapshot_mode,
            self.planned_freshness_nonce,
        )
        if any(value is not None for value in planned_fields) and not all(
            value is not None for value in planned_fields
        ):
            raise ValueError(
                "planned name, target, mode, and freshness nonce for WORKDIR volume "
                "must appear together"
            )
        if self.actual is not None:
            if self.planned_name is None:
                raise ValueError("actual WORKDIR volume requires a planned name")
            if self.actual.name != self.planned_name:
                raise ValueError("actual WORKDIR volume differs from planned name")
            if self.actual.target != self.planned_target:
                raise ValueError(
                    "actual WORKDIR volume target differs from planned target"
                )
            if self.actual.snapshot_mode != self.planned_snapshot_mode:
                raise ValueError("actual WORKDIR volume mode differs from planned mode")
            if self.actual.freshness_nonce != self.planned_freshness_nonce:
                raise ValueError(
                    "actual WORKDIR volume freshness differs from planned authority"
                )
        if self.rollback is not None:
            if self.planned_name is None:
                raise ValueError("rollback WORKDIR volume requires planned authority")
            if self.actual is not None or self.mounted:
                raise ValueError(
                    "trusted and rollback WORKDIR volume authority cannot coexist"
                )
        if self.mounted and self.actual is None:
            raise ValueError("mounted WORKDIR volume requires actual authority")
        return self


class WorkResourceLease(PersistedModel):
    """All recoverable identities owned by the persistent Work runtime."""

    planned_container: str | None = None
    container_id: str | None = None
    planned_network: str | None = None
    network_id: str | None = None
    network_name: str | None = None
    planned_policy_rule_id: str | None = None
    policy_rule_id: str | None = None
    planned_retained_image_ref: str | None = None
    retained_image_id: str | None = None
    retained_image_ref: str | None = None
    retained_image_rollback: RetainedImageRollbackAuthority | None = None
    planned_quiescence: Literal["pause-if-running"] | None = None
    paused: bool = False
    stopped: bool = False
    workdir_volume: WorkdirVolumeResourceLease = Field(
        default_factory=WorkdirVolumeResourceLease
    )

    @field_validator("planned_retained_image_ref", "retained_image_ref")
    @classmethod
    def valid_retained_image_ref(cls, value: str | None) -> str | None:
        if value is not None and _RETAINED_IMAGE_REF.fullmatch(value) is None:
            raise ValueError("invalid retained Work image reference")
        return value

    @field_validator("retained_image_id")
    @classmethod
    def valid_retained_image_id(cls, value: str | None) -> str | None:
        if value is not None and _RETAINED_IMAGE_ID.fullmatch(value) is None:
            raise ValueError("invalid retained Work image ID")
        return value

    @model_validator(mode="after")
    def consistent_retained_image_authority(self) -> WorkResourceLease:
        if self.paused and self.stopped:
            raise ValueError("Work cannot be both paused and stopped")
        if (self.retained_image_id is None) != (self.retained_image_ref is None):
            raise ValueError(
                "retained Work image ID and reference must appear together"
            )
        if self.retained_image_ref is not None:
            if self.planned_retained_image_ref is None:
                raise ValueError("retained Work image requires a planned reference")
            if self.retained_image_ref != self.planned_retained_image_ref:
                raise ValueError("retained Work image reference differs from planned")
        if self.retained_image_rollback is not None:
            if self.planned_retained_image_ref is None:
                raise ValueError(
                    "retained Work rollback authority requires a planned reference"
                )
            if self.retained_image_ref is not None:
                raise ValueError(
                    "trusted and rollback retained Work authority cannot coexist"
                )
        return self


class JudgeResourceLease(PersistedModel):
    """All recoverable identities owned by the current Judge round."""

    round_id: str | None = None
    planned_container: str | None = None
    container_id: str | None = None
    planned_network: str | None = None
    network_id: str | None = None
    network_name: str | None = None
    planned_policy_rule_id: str | None = None
    policy_rule_id: str | None = None
    planned_snapshot: str | None = None
    planned_snapshot_ref: str | None = None
    snapshot_lease_id: str | None = None
    snapshot_image_id: str | None = None
    snapshot_image_ref: str | None = None
    snapshot_source_container_id: str | None = None
    snapshot_merged_path: Path | None = None
    snapshot_process_id: int | None = None

    _absolute_snapshot_path = field_validator("snapshot_merged_path", mode="after")(
        _absolute_optional_path
    )

    @field_validator("planned_snapshot_ref", "snapshot_image_ref")
    @classmethod
    def valid_judge_snapshot_ref(cls, value: str | None) -> str | None:
        if value is not None and _JUDGE_IMAGE_REF.fullmatch(value) is None:
            raise ValueError("invalid Judge snapshot image reference")
        return value

    @field_validator("snapshot_image_id")
    @classmethod
    def valid_judge_snapshot_id(cls, value: str | None) -> str | None:
        if value is not None and _JUDGE_IMAGE_ID.fullmatch(value) is None:
            raise ValueError("invalid Judge snapshot image ID")
        return value

    @field_validator("snapshot_source_container_id")
    @classmethod
    def valid_snapshot_source_container(cls, value: str | None) -> str | None:
        if value is not None and _SAFE_ID.fullmatch(value) is None:
            raise ValueError("invalid Judge snapshot source Work container ID")
        return value

    @model_validator(mode="after")
    def consistent_judge_snapshot_authority(self) -> JudgeResourceLease:
        has_image_authority = self.planned_snapshot_ref is not None or any(
            value is not None
            for value in (
                self.snapshot_image_id,
                self.snapshot_image_ref,
                self.snapshot_source_container_id,
            )
        )
        has_path_authority = any(
            value is not None
            for value in (self.snapshot_merged_path, self.snapshot_process_id)
        )
        if has_image_authority and has_path_authority:
            raise ValueError("Judge snapshot path and image authority cannot be mixed")
        actual = (
            self.snapshot_image_id,
            self.snapshot_image_ref,
            self.snapshot_source_container_id,
        )
        if any(value is not None for value in actual) and not all(
            value is not None for value in actual
        ):
            raise ValueError(
                "Judge snapshot image ID, reference, and source must appear together"
            )
        if self.snapshot_image_ref is not None:
            if self.planned_snapshot_ref is None:
                raise ValueError(
                    "actual Judge snapshot authority requires a planned reference"
                )
            if self.snapshot_image_ref != self.planned_snapshot_ref:
                raise ValueError("actual Judge snapshot reference differs from planned")
            if self.snapshot_lease_id is None:
                raise ValueError(
                    "actual Judge snapshot authority requires its lease ID"
                )
        return self


class SnapshotRecoveryAuthority(PersistedModel):
    """Authoritative active snapshot manifest/layer/process discovery result."""

    run_id: str
    lease_id: str
    round_id: str
    merged_path: Path
    process_id: int | None = None
    manifest_requires_recovery: bool
    layers_present: bool
    process_alive: bool

    _absolute_merged_path = field_validator("merged_path", mode="after")(
        _absolute_optional_path
    )


class ResourceLease(PersistedModel):
    """Secret-free recovery authority for every runtime resource in one run."""

    schema_version: int = Field(default=6, strict=True)
    run_id: str
    task_id: str
    coordinator_pid: int
    coordinator_started_at: float
    phase: str
    phase_history: tuple[str, ...] = (RunStatus.PREPARING.value,)
    status: RunStatus | None = None
    workspace_path: Path | None = None
    cleanup_image_ref: str | None = None
    rootfs_snapshot_mode: RootfsSnapshotMode = RootfsSnapshotMode.FULL_ROOTFS
    gpu_plan: RunGPUPlan | None = None
    work: WorkResourceLease = Field(default_factory=WorkResourceLease)
    judge: JudgeResourceLease = Field(default_factory=JudgeResourceLease)
    sandboxes: tuple[SandboxLease, ...] = ()
    sandbox_reservation: SandboxReservation | None = None
    sandbox_envs: tuple[SandboxEnvLease, ...] = ()
    sandbox_images: tuple[SandboxImageLease, ...] = ()
    sandbox_builders: tuple[BuilderLease, ...] = ()
    # Set with an environment grant's reservation and never cleared: every
    # later recovery still sweeps the run's exact labels for late creates.
    sandbox_env_authority: bool = False
    # Set when the run's envs are E2B sandboxes: where they are, and where
    # recovery reads the API key (never the key itself).
    sandbox_e2b: EnvE2BHost | None = None
    recovery_required: bool = False
    error: str | None = None

    @field_validator("run_id", "task_id")
    @classmethod
    def safe_identity(cls, value: str) -> str:
        if _SAFE_ID.fullmatch(value) is None:
            raise ValueError("lease identifiers must be safe path components")
        return value

    @field_validator("workspace_path", mode="after")
    @classmethod
    def absolute_optional_path(cls, value: Path | None) -> Path | None:
        return _absolute_optional_path(value)

    @field_validator("cleanup_image_ref")
    @classmethod
    def immutable_cleanup_image(cls, value: str | None) -> str | None:
        if value is not None and not is_immutable_image_ref(value):
            raise ValueError("cleanup image authority must be immutable")
        return value

    @model_validator(mode="after")
    def current_recovery_schema_and_sources(self) -> ResourceLease:
        if self.schema_version != 6:
            raise ValueError("unsupported resource lease schema version")
        seen: set[str] = set()
        for child in self.sandboxes:
            if child.owner.run_id != self.run_id or child.owner.task_id != self.task_id:
                raise ValueError("sandbox owner identity differs from resource lease")
            if child.child_id in seen:
                raise ValueError("duplicate sandbox child identity")
            seen.add(child.child_id)
        # Env and builder Docker names carry only 16 hex digits of identity;
        # one journal record must own each derived container/network/rule.
        for kind, records, identity in (
            ("environment", self.sandbox_envs, lambda env: short_identity(env.env_id)),
            ("image", self.sandbox_images, lambda image: image.handle),
            (
                "builder",
                self.sandbox_builders,
                lambda builder: short_identity(builder.builder_id),
            ),
        ):
            identities: set[str] = set()
            for record in records:
                if (record.owner.run_id, record.owner.task_id) != (
                    self.run_id,
                    self.task_id,
                ):
                    raise ValueError(
                        f"sandbox {kind} owner identity differs from resource lease"
                    )
                if identity(record) in identities:
                    raise ValueError(f"duplicate sandbox {kind} identity")
                identities.add(identity(record))
        volume = self.work.workdir_volume
        if self.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS and (
            volume.planned_name is not None
            or volume.planned_target is not None
            or volume.planned_snapshot_mode is not None
            or volume.planned_freshness_nonce is not None
            or volume.actual is not None
            or volume.rollback is not None
            or volume.mounted
        ):
            raise ValueError("full-rootfs lease cannot carry WORKDIR volume authority")
        if volume.actual is not None:
            if volume.actual.run_id != self.run_id:
                raise ValueError("WORKDIR volume run identity differs from lease")
            if volume.actual.task_id != self.task_id:
                raise ValueError("WORKDIR volume task identity differs from lease")
            if volume.actual.snapshot_mode is not self.rootfs_snapshot_mode:
                raise ValueError("WORKDIR volume mode differs from lease")
        source = self.judge.snapshot_source_container_id
        if (
            source is not None
            and self.work.container_id is not None
            and source != self.work.container_id
        ):
            raise ValueError(
                "Judge snapshot source differs from durable Work container"
            )
        return self


def retained_sandbox_resources(lease: ResourceLease) -> tuple[str, ...]:
    """Environment, image and builder records whose cleanup is not proven.

    Pool release needs every env, built image and builder removed (spec 5).
    A leaked built image (rmi conflict) still occupies disk counted in the
    reservation, so it is retained too. Pulled images are a cache outside the
    pool that is never removed, so they never hold the reservation.
    """
    return (
        *(
            f"env {env.env_id}"
            for env in lease.sandbox_envs
            if env.state != "removed" or env.pending_mutation
        ),
        *(
            f"image {image.handle}"
            for image in lease.sandbox_images
            if image.kind == "built" and image.state != "removed"
        ),
        *(
            f"builder {builder.builder_id}"
            for builder in lease.sandbox_builders
            if builder.state != "removed" or builder.pending_mutation
        ),
    )


class RecoveryBackend(Protocol):
    """Authoritative production inspection/mutation port used by recovery."""

    def list_containers(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]: ...

    def inspect_container(self, container_id: str) -> Mapping[str, Any] | None: ...

    def stop_container(self, container_id: str) -> None: ...

    def remove_container(self, container_id: str) -> None: ...

    def terminate_sandbox(self, child: SandboxLease) -> None: ...

    def remove_sandbox(self, child: SandboxLease) -> None: ...

    def kill_sandbox_container(self, container_id: str) -> None: ...

    def remove_sandbox_container(self, container_id: str) -> None: ...

    def recover_sandbox_network(
        self, env: SandboxEnvLease | BuilderLease, *, settled: bool = False
    ) -> tuple[str, ...]: ...

    def remove_builder_loop(self, path: Path) -> None: ...

    def coordinator_alive(self, pid: int, started_at: float) -> bool: ...

    def settle_sandbox_creates(self, rule_ids: tuple[str, ...]) -> bool: ...

    def remove_sandbox_spool(self, path: Path) -> None: ...

    def list_images(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]: ...

    def list_tagged_images(
        self, prefix: str
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]: ...

    def inspect_image(self, image_id: str) -> Mapping[str, Any] | None: ...

    def image_in_use(self, image_id: str) -> bool: ...

    def remove_image(self, image_id: str) -> None: ...

    def list_volumes(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]: ...

    def inspect_volume(self, name: str) -> Mapping[str, Any] | None: ...

    def volume_in_use(self, name: str) -> bool: ...

    def remove_volume(self, name: str) -> None: ...

    def list_networks(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]: ...

    def network_in_use(self, network_id: str) -> bool: ...

    def remove_network(self, network_id: str) -> None: ...

    def is_mounted(self, path: Path) -> bool: ...

    def discover_snapshot_leases(
        self,
        *,
        run_id: str,
        round_id: str | None,
        lease_id: str | None,
    ) -> tuple[SnapshotRecoveryAuthority, ...]: ...

    def release_snapshot(self, authority: SnapshotRecoveryAuthority) -> None: ...

    def unpause_container(self, container_id: str) -> None: ...

    def remove_policy(self, rule_id: str) -> None: ...

    def policy_exists(self, rule_id: str) -> bool: ...

    def workspace_is_mounted(self, workspace: Path) -> bool: ...

    def delete_workspace(
        self,
        workspace: Path,
        *,
        image_ref: str,
        run_id: str,
        task_id: str,
    ) -> None: ...


class LeaseStore:
    """Atomically persist leases and serialize per-run lifecycle operations."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def path_for(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.root / f"{run_id}.json"

    def write(self, lease: ResourceLease) -> None:
        # model_copy(update=...) deliberately skips Pydantic validation.  The
        # durable boundary must not accept forged root-helper authority.
        lease = ResourceLease.model_validate(lease.model_dump())
        durable_mkdir(self.root)
        path = self.path_for(lease.run_id)
        payload = redact_structure(lease.model_dump(mode="json"))
        self._reject_secret_keys(payload)
        # Redaction may rewrite a string into an invalid value; refuse before
        # replacing the last readable lease rather than persist one that
        # recovery and admission can no longer read.
        ResourceLease.model_validate_json(json.dumps(payload))
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(raw_temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(
                    payload,
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

    def read(self, run_id: str) -> ResourceLease | None:
        path = self.path_for(run_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        if type(raw.get("schema_version")) is int and raw["schema_version"] == 4:
            if raw.get("sandboxes") or raw.get("sandbox_reservation") is not None:
                raise ValueError("schema 4 cannot carry sandbox authority")
            raw["schema_version"] = 5
            raw["sandboxes"] = []
            raw["sandbox_reservation"] = None
        if type(raw.get("schema_version")) is int and raw["schema_version"] == 5:
            if (
                raw.get("sandbox_envs")
                or raw.get("sandbox_images")
                or raw.get("sandbox_builders")
            ):
                raise ValueError("schema 5 cannot carry sandbox environment authority")
            raw["schema_version"] = 6
            raw["sandbox_envs"] = []
            raw["sandbox_images"] = []
            raw["sandbox_builders"] = []
        # JSON validation preserves strict nested tuple/scalar contracts.
        lease = ResourceLease.model_validate_json(json.dumps(raw))
        if lease.run_id != run_id:
            raise ValueError("lease run identity differs from filename")
        return lease

    def list_run_ids(self) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        return tuple(sorted(path.stem for path in self.root.glob("*.json")))

    @contextmanager
    def lock(self, run_id: str, *, blocking: bool = True) -> Iterator[None]:
        self._validate_run_id(run_id)
        durable_mkdir(self.root)
        lock_path = self.root / f"{run_id}.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(descriptor, operation)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if _SAFE_ID.fullmatch(run_id) is None:
            raise ValueError("run ID must be one safe path component")

    @classmethod
    def _reject_secret_keys(cls, value: object) -> None:
        forbidden = ("secret", "token", "authorization", "header", "credential")
        if isinstance(value, Mapping):
            for key, child in value.items():
                if any(word in str(key).casefold() for word in forbidden):
                    raise ValueError(f"secret-bearing lease key rejected: {key}")
                cls._reject_secret_keys(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                cls._reject_secret_keys(child)


class RecoveryManager:
    """Recover labeled resources without trusting IDs in stale lease JSON."""

    def __init__(
        self,
        *,
        store: LeaseStore,
        backend: RecoveryBackend,
        managed_root: Path,
        e2b_client: Callable[[EnvE2BHost], Any] | None = None,
    ) -> None:
        self._store = store
        self._backend = backend
        self._managed_root = Path(managed_root).resolve()
        # [environments.host.e2b] -> a client of the run's E2B sandboxes.
        self._e2b_client = e2b_client

    def recover(self, run_id: str | None = None) -> tuple[str, ...]:
        run_ids = (run_id,) if run_id is not None else self._store.list_run_ids()
        recovered: list[str] = []
        for selected in run_ids:
            with self._store.lock(selected):
                lease = self._store.read(selected)
                if lease is None:
                    continue
                mark_interrupted = lease.status is None or lease.recovery_required
                self._recover_locked(lease, mark_interrupted=mark_interrupted)
                recovered.append(selected)
        return tuple(recovered)

    def recover_e2b(self, run_id: str) -> bool:
        """Kill only the run's E2B sandboxes (a cluster run, whose Work and
        Judge were scheduler-job processes). False when it has none.
        A run whose Engine still holds its lease is refused, not awaited."""
        lock = self._store.lock(run_id, blocking=False)
        try:
            lock.__enter__()
        except BlockingIOError:
            raise RuntimeError(
                f"run {run_id} is still live (its Engine holds the lease)"
            ) from None
        try:
            lease = self._store.read(run_id)
            if lease is None or lease.sandbox_e2b is None:
                return False
            self._recover_e2b_envs(lease)
            return True
        finally:
            lock.__exit__(None, None, None)

    def cleanup(self, run_id: str, delete_workspace: bool = False) -> None:
        with self._store.lock(run_id):
            lease = self._store.read(run_id)
            if lease is None:
                return
            self._recover_locked(
                lease,
                mark_interrupted=False,
                delete_retained_image=delete_workspace,
            )
            current = self._require_lease(run_id)
            if not delete_workspace:
                return
            if self._find_containers(current, "helper"):
                self._fail_closed(current, "workspace deletion blocked by live helper")
            if self._find_container(current, "judge") is not None:
                self._fail_closed(current, "workspace deletion blocked by live Judge")
            if self._find_container(current, "work") is not None:
                self._fail_closed(current, "workspace deletion blocked by live Work")
            for role in ("judge", "work"):
                if self._find_networks(current, role):
                    self._fail_closed(
                        current,
                        f"workspace deletion blocked by a live {role} network",
                    )
            if any(
                rule_id is not None
                for rule_id in (
                    current.judge.planned_policy_rule_id,
                    current.judge.policy_rule_id,
                    current.work.planned_policy_rule_id,
                    current.work.policy_rule_id,
                )
            ):
                self._fail_closed(
                    current, "workspace deletion blocked by firewall authority"
                )
            if current.workspace_path is None:
                return
            workspace = self._validated_workspace(current)
            if self._backend_call(
                current,
                "workspace mount inspection",
                lambda: self._backend.workspace_is_mounted(workspace),
            ):
                self._fail_closed(current, "workspace deletion blocked by a live mount")
            if workspace.exists():
                if current.cleanup_image_ref is None:
                    raise RuntimeError(
                        "workspace cleanup image authority is unavailable"
                    )
                try:
                    self._backend.delete_workspace(
                        workspace,
                        image_ref=current.cleanup_image_ref,
                        run_id=current.run_id,
                        task_id=current.task_id,
                    )
                except Exception as delete_error:
                    self._contain_helpers_after_delete_failure(
                        current, delete_error=delete_error
                    )

    def _recover_locked(
        self,
        lease: ResourceLease,
        *,
        mark_interrupted: bool,
        delete_retained_image: bool = False,
    ) -> None:
        # Children, then envs, reconcile independently: a failed child never
        # leaves an env executing (S8). Parents are contained only after both
        # ran, if either failed. A run without an environment grant has no
        # env authority and its recovery is unchanged.
        sandbox_errors: list[Exception] = []
        for reconcile in (self._recover_sandboxes, self._recover_sandbox_envs):
            try:
                lease = reconcile(lease)
            except Exception as error:
                sandbox_errors.append(error)
                lease = self._require_lease(lease.run_id)
        if sandbox_errors:
            containment_errors = self._contain_parents_after_sandbox_failure(lease)
            self._fail_closed(
                self._require_lease(lease.run_id),
                "; ".join(str(e) for e in [*sandbox_errors, *containment_errors]),
                cause=sandbox_errors[0],
            )
        # A crashed initialization/cleanup helper is root and still has the
        # workspace bind.  Contain every exact-labeled helper before touching
        # snapshots, Work, or starting another cleanup helper.
        helper_errors: list[Exception] = []
        for helper in self._find_containers(lease, "helper"):
            try:
                self._stop_and_remove_helper(lease, *helper)
            except Exception as error:
                helper_errors.append(error)
        if helper_errors:
            detail = "; ".join(str(error) for error in helper_errors)
            self._fail_closed(
                lease,
                "cleanup helper containment/removal failed for one or more "
                f"containers: {detail}",
                cause=helper_errors[0],
            )

        # Anti-cheat ordering: contain Judge before optional Work discovery.
        judge_errors: list[Exception] = []
        for judge in self._find_containers(lease, "judge"):
            try:
                self._stop_and_remove_judge(lease, *judge)
            except Exception as error:
                judge_errors.append(error)
        if judge_errors:
            detail = "; ".join(str(error) for error in judge_errors)
            self._fail_closed(
                lease,
                "Judge containment/removal failed for one or more containers: "
                f"{detail}",
                cause=judge_errors[0],
            )
        lease = self._update_judge(lease, container_id=None)

        # A Judge policy or network may still reference or expose the round.
        # Prove both absent before releasing either legacy or image snapshots.
        lease = self._recover_policies(lease, "judge")
        lease = self._recover_networks(lease, "judge")
        lease = self._recover_snapshots(lease)
        lease = self._recover_round_images(lease)

        work = self._find_container(lease, "work")
        retained_source_container_id = (
            work[0] if work is not None else lease.work.container_id
        )
        if work is not None:
            work_id, state = work
            if bool(state.get("paused", False)):
                self._backend_call(
                    lease,
                    "Work unpause",
                    lambda: self._backend.unpause_container(work_id),
                )
            inspected = self._backend_call(
                lease,
                "Work inspection",
                lambda: self._backend.inspect_container(work_id),
            )
            if inspected is not None and bool(inspected.get("running", False)):
                self._backend_call(
                    lease,
                    "Work stop",
                    lambda: self._backend.stop_container(work_id),
                )
            inspected = self._backend_call(
                lease,
                "Work post-stop inspection",
                lambda: self._backend.inspect_container(work_id),
            )
            if inspected is not None and bool(inspected.get("running", False)):
                self._fail_closed(lease, "Work termination cannot be proven")
            self._backend_call(
                lease,
                "Work removal",
                lambda: self._backend.remove_container(work_id),
            )
            if (
                self._backend_call(
                    lease,
                    "Work post-removal inspection",
                    lambda: self._backend.inspect_container(work_id),
                )
                is not None
            ):
                self._fail_closed(lease, "Work removal cannot be proven")
        lease = self._update_work(
            lease,
            container_id=None,
            planned_quiescence=None,
            paused=False,
            stopped=False,
        )
        lease = self._recover_policies(lease, "work")
        lease = self._recover_networks(lease, "work")
        if delete_retained_image:
            lease = self._recover_workdir_volume(
                lease, delete=False, preflight_only=True
            )
        lease = self._recover_retained_images(
            lease,
            delete=delete_retained_image,
            source_container_id=retained_source_container_id,
        )
        lease = self._recover_workdir_volume(lease, delete=delete_retained_image)
        # No Work or Judge binds an endpoint any more.
        self._remove_sandbox_root(lease)

        phase = "interrupted" if mark_interrupted else lease.phase
        status = RunStatus.CANCELLED if mark_interrupted else lease.status
        history = lease.phase_history
        if mark_interrupted and history[-1:] != (phase,):
            history += (phase,)
        final = lease.model_copy(
            update={
                "phase": phase,
                "phase_history": history,
                "status": status,
                "work": lease.work.model_copy(update={"planned_container": None}),
                "judge": lease.judge.model_copy(
                    update={"planned_container": None, "round_id": None}
                ),
                "recovery_required": False,
                "error": None,
                "sandbox_reservation": None,
            }
        )
        self._store.write(final)

    def _contain_parents_after_sandbox_failure(
        self, lease: ResourceLease
    ) -> list[Exception]:
        """Stop independent running parents; never thaw or release dependencies."""
        errors: list[Exception] = []
        discover = getattr(
            self._backend, "list_container_candidates", self._backend.list_containers
        )
        for role in ("helper", "judge", "work"):
            required = self._labels(lease, role)
            try:
                parents = discover(labels=required)
            except Exception as error:
                errors.append(error)
                continue
            for identity, listed in parents:
                try:
                    if not all(
                        listed.get("labels", {}).get(key) == value
                        for key, value in required.items()
                    ):
                        continue
                    # One failed inspection must not hide later same-role peers.
                    state = self._backend.inspect_container(identity)
                    if state is None:
                        continue
                    labels = state.get("labels", {})
                    if not all(
                        labels.get(key) == value for key, value in required.items()
                    ):
                        raise InfrastructureError(
                            f"{role} containment identity changed"
                        )
                    if not isinstance(state.get("running"), bool) or not isinstance(
                        state.get("paused"), bool
                    ):
                        raise InfrastructureError(
                            f"{role} containment state is ambiguous"
                        )
                    if state["paused"] or not state["running"]:
                        continue
                    self._backend.stop_container(identity)
                    stopped = self._backend.inspect_container(identity)
                    if stopped is not None and stopped.get("running") is not False:
                        raise InfrastructureError(
                            f"{role} containment cannot be proven"
                        )
                except Exception as error:
                    errors.append(error)
        return errors

    def _recover_sandboxes(self, lease: ResourceLease) -> ResourceLease:
        """Reconcile every child independently before releasing dependencies."""
        errors: list[Exception] = []
        for original in lease.sandboxes:
            try:
                lease = self._recover_sandbox(lease, original)
            except Exception as error:
                errors.append(error)
                # A failed late create may already have persisted its actual ID.
                lease = self._require_lease(lease.run_id)
        if errors:
            self._fail_closed(
                lease,
                "sandbox reconciliation failed: " + "; ".join(str(e) for e in errors),
                cause=errors[0],
            )
        return lease

    def _recover_sandbox(
        self, lease: ResourceLease, original: SandboxLease
    ) -> ResourceLease:
        candidates: dict[str, Mapping[str, Any]] = {}
        required = sandbox_labels(original)
        listed = self._backend_call(
            lease,
            "sandbox discovery",
            lambda: self._backend.list_containers(labels=required),
        )
        for identity, state in listed:
            if state.get("labels") == required:
                candidates[identity] = state
        for lookup in (original.container_id, original.planned_name):
            if lookup is None:
                continue
            state = self._backend_call(
                lease,
                "sandbox identity inspection",
                lambda lookup=lookup: self._backend.inspect_container(lookup),
            )
            if state is not None:
                identity = state.get("id")
                if not isinstance(identity, str):
                    self._fail_closed(lease, "sandbox inspection lacks actual identity")
                candidates[identity] = state
        if len(candidates) > 1:
            self._fail_closed(lease, "sandbox ambiguous planned/actual identity")
        child = original
        for identity in candidates:
            state = self._backend_call(
                lease,
                "sandbox fresh ownership inspection",
                lambda: self._backend.inspect_container(identity),
            )
            if state is None:
                self._fail_closed(lease, "sandbox disappeared during reconciliation")
            try:
                attest_sandbox_identity(
                    original,
                    {
                        "Id": state.get("id"),
                        "Name": "/" + str(state.get("name")),
                        "Image": state.get("image_id"),
                        "Config": {"Labels": state.get("labels")},
                    },
                )
            except InfrastructureError as error:
                self._fail_closed(lease, str(error), cause=error)
            child = original.model_copy(update={"container_id": identity})
            # Persist a late create's discovered identity before mutation.
            lease = lease.model_copy(
                update={
                    "sandboxes": tuple(
                        child if item.child_id == child.child_id else item
                        for item in lease.sandboxes
                    )
                }
            )
            if identity != original.container_id:
                self._store.write(lease)
            self._backend_call(
                lease,
                "sandbox termination",
                lambda: self._backend.terminate_sandbox(child),
            )
            stopped = self._backend_call(
                lease,
                "sandbox post-stop inspection",
                lambda: self._backend.inspect_container(identity),
            )
            if stopped is not None and (
                stopped.get("running") or stopped.get("paused")
            ):
                self._fail_closed(lease, "sandbox termination cannot be proven")
            # Containing an already-durable identity must survive a storage
            # outage. Still require writable authority before deletion/release.
            if identity == original.container_id:
                self._store.write(lease)
            self._backend_call(
                lease,
                "sandbox removal",
                lambda: self._backend.remove_sandbox(child),
            )
            if (
                self._backend_call(
                    lease,
                    "sandbox post-removal inspection",
                    lambda: self._backend.inspect_container(identity),
                )
                is not None
            ):
                self._fail_closed(lease, "sandbox removal cannot be proven")
        if not candidates and original.pending_mutation:
            self._fail_closed(
                lease,
                "sandbox pending mutation outcome is unknown; reservation retained",
            )
        removed = child.model_copy(
            update={"state": "removed", "pending_mutation": False}
        )
        lease = lease.model_copy(
            update={
                "sandboxes": tuple(
                    removed if item.child_id == child.child_id else item
                    for item in lease.sandboxes
                )
            }
        )
        self._store.write(lease)
        return lease

    # -- brokered environments (schema 6) ------------------------------------------

    def _recover_sandbox_envs(self, lease: ResourceLease) -> ResourceLease:
        """Converge every journaled env to proven absence (spec 5, M6).

        Only a run that had an environment grant has env authority; any other
        run returns untouched. Every service of every env is SIGKILLed with
        proof first (Judge before Work; a paused one through the paused
        killer), then each env is removed in teardown order: containers,
        volumes, then bridge and firewall rule. Objects are found by planned
        name and exact labels and each leaves the journal only once proven
        absent; a removed env leaves the lease.

        A pending create has at most one call whose outcome the journal lacks
        (``_in_flight``). Finding its object resolves it. Finding nothing
        counts as absence only once that create has settled: the coordinator
        that held the broker is gone, no firewall command for the env's rule
        still runs, and after a settle delay the env is searched again (spec
        5 resolution for M6). Otherwise only that env fails closed, as a
        pending v1 child does. Every recovery of a run with env authority
        then sweeps the run's exact labels for orphans, pulled handles are
        released (the cached image stays, S9) and the run's stage spool is
        removed.
        """
        reservation = lease.sandbox_reservation
        if not (
            lease.sandbox_env_authority
            or lease.sandbox_envs
            or lease.sandbox_images
            or lease.sandbox_builders
            or (reservation is not None and reservation.pool_disk_mb is not None)
        ):
            return lease
        if not lease.sandbox_env_authority:
            # Durable before any mutation, so the sweep outlives the records.
            lease = lease.model_copy(update={"sandbox_env_authority": True})
            self._store.write(lease)
        if lease.sandbox_e2b is not None:
            lease = self._recover_e2b_envs(lease)
        errors: list[Exception] = []
        # From the durable journal, before containment journals any late ID.
        in_flight = {
            env.env_id: _in_flight(env)
            for env in lease.sandbox_envs
            if env.pending_mutation and env.state != "removed"
        }
        contained: dict[str, dict[int, str]] = {}
        order = sorted(lease.sandbox_envs, key=lambda env: env.owner.phase != "judge")
        for original in order:
            if original.state == "removed":
                continue
            try:
                lease, contained[original.env_id] = self._contain_sandbox_env(
                    lease, original.env_id
                )
            except Exception as error:
                errors.append(error)
                lease = self._require_lease(lease.run_id)
        unresolved: list[SandboxEnvLease] = []
        for original in order:
            if original.env_id not in contained or original.env_id not in in_flight:
                continue
            try:
                env = self._env_record(lease, original.env_id)
                if not self._in_flight_found(
                    lease, env, in_flight[env.env_id], contained[env.env_id]
                ):
                    unresolved.append(env)
            except Exception as error:
                errors.append(error)
                lease = self._require_lease(lease.run_id)
                del contained[original.env_id]
        settled: set[str] = set()
        if unresolved:
            try:
                self._settle_sandbox_creates(lease, unresolved, in_flight)
            except Exception as error:
                errors.append(error)
                lease = self._require_lease(lease.run_id)
            else:
                for env in unresolved:
                    try:
                        # A create that landed meanwhile is found and contained.
                        lease, contained[env.env_id] = self._contain_sandbox_env(
                            lease, env.env_id
                        )
                        settled.add(env.env_id)
                    except Exception as error:
                        errors.append(error)
                        lease = self._require_lease(lease.run_id)
            for env in unresolved:
                if env.env_id not in settled:
                    contained.pop(env.env_id, None)
        for original in order:
            try:
                env = self._env_record(lease, original.env_id)
                if env.state == "removed":
                    # Proof already journaled: only compaction remains.
                    lease = self._commit_env(lease, env)
                    continue
                if env.env_id not in contained:
                    continue
                lease = self._remove_sandbox_env(
                    lease,
                    env,
                    contained[env.env_id],
                    settled=env.env_id in settled,
                )
            except Exception as error:
                errors.append(error)
                lease = self._require_lease(lease.run_id)
        if errors:
            self._fail_closed(
                lease,
                "sandbox environment reconciliation failed: "
                + "; ".join(str(e) for e in errors),
                cause=errors[0],
            )
        lease = self._sweep_sandbox_env_orphans(lease)
        pulled = [image for image in lease.sandbox_images if image.kind == "pulled"]
        if pulled:
            # A pulled record is a ledger entry only: the handle is unbound
            # and the image stays cached, whoever pulled it first (S9). Its
            # record leaves the lease as image_release's does (M5).
            for image in pulled:
                LOGGER.info(
                    "releasing pulled sandbox image %s (%s, pre-existing: %s); "
                    "the image stays cached",
                    image.handle,
                    image.image_id,
                    image.pre_existing,
                )
            lease = lease.model_copy(
                update={
                    "sandbox_images": tuple(
                        image
                        for image in lease.sandbox_images
                        if image.kind != "pulled"
                    )
                }
            )
            self._store.write(lease)
        self._remove_sandbox_spool(lease)
        lease = self._recover_sandbox_builders(lease)
        lease = self._recover_built_images(lease)
        retained = retained_sandbox_resources(lease)
        if retained:
            self._fail_closed(
                lease,
                "sandbox environment reconciliation left " + ", ".join(retained[:8]),
            )
        return lease

    def _recover_e2b_envs(self, lease: ResourceLease) -> ResourceLease:
        """E2B envs: kill every sandbox carrying the run's metadata, running
        or paused (a paused one never expires on E2B), prove none is left,
        then journal each env removed. Templates stay: a shared cache."""
        from rsi_harness.runtime.sandbox_e2b import e2b_client, kill_run_sandboxes

        settings = lease.sandbox_e2b
        assert settings is not None
        try:
            client = (self._e2b_client or e2b_client)(settings)
            try:
                killed = kill_run_sandboxes(client, lease.run_id)
            finally:
                client.close()
        except Exception as error:
            self._fail_closed(
                lease, f"e2b sandbox cleanup failed: {error}", cause=error
            )
        if killed:
            LOGGER.warning(
                "killed %d e2b sandboxes of run %s", len(killed), lease.run_id
            )
        for env in lease.sandbox_envs:
            if env.backend != "e2b" or env.state == "removed":
                continue
            removed = env.model_copy(
                update={
                    "state": "removed",
                    "pending_mutation": False,
                    "services": tuple(
                        service.model_copy(update={"state": "removed"})
                        for service in env.services
                    ),
                }
            )
            lease = self._commit_env(lease, removed)
        return lease

    # -- builders and built images (M8) --------------------------------------------

    def _recover_sandbox_builders(self, lease: ResourceLease) -> ResourceLease:
        """Converge every journaled builder to proven absence (spec 4).

        Per builder, in order: the container (killed, then removed), the
        state volume, the loop devices of its file (``losetup -j`` finds one
        the journal missed) and the file, then the bridge and its rule. A
        builder whose create was in flight is settled first, as a pending
        env create is: only then does finding nothing prove absence. The
        run's exact builder labels are swept for late creates, and the
        run's ``sb/build`` directory goes last.
        """
        errors: list[Exception] = []
        order = sorted(
            lease.sandbox_builders, key=lambda builder: builder.owner.phase != "judge"
        )
        pending = [
            builder
            for builder in order
            if builder.pending_mutation and builder.state == "planned"
        ]
        if pending:
            names = ", ".join(builder.builder_id for builder in pending)
            if self._backend_call(
                lease,
                "sandbox coordinator liveness inspection",
                lambda: self._backend.coordinator_alive(
                    lease.coordinator_pid, lease.coordinator_started_at
                ),
            ):
                self._fail_closed(
                    lease,
                    f"sandbox builder {names} pending create outcome is unknown "
                    "while its coordinator may still run",
                )
            rule_ids = tuple(
                builder.rule_id for builder in pending if builder.network_id is None
            )
            if not self._backend_call(
                lease,
                "sandbox builder pending create settlement",
                lambda: self._backend.settle_sandbox_creates(rule_ids),
            ):
                self._fail_closed(
                    lease,
                    f"sandbox builder {names} pending create has not settled",
                )
        for original in order:
            try:
                lease = self._remove_sandbox_builder(lease, original.builder_id)
            except Exception as error:
                errors.append(error)
                lease = self._require_lease(lease.run_id)
        if errors:
            self._fail_closed(
                lease,
                "sandbox builder reconciliation failed: "
                + "; ".join(str(e) for e in errors),
                cause=errors[0],
            )
        lease = self._sweep_sandbox_builder_orphans(lease)
        self._remove_sandbox_build_dir(lease)
        return lease

    def _remove_sandbox_builder(
        self, lease: ResourceLease, builder_id: str
    ) -> ResourceLease:
        builder = self._builder_record(lease, builder_id)
        if builder.state == "removed" and not builder.pending_mutation:
            return self._commit_builder(lease, builder)
        # A planned builder's creates were settled before any removal (or
        # its objects were found by label): finding nothing proves absence.
        settled = builder.state == "planned"
        builder = builder.model_copy(update={"pending_mutation": True})
        lease = self._commit_builder(lease, builder)
        identity = self._find_builder_container(lease, builder)
        if identity is not None:
            if builder.container_id is None:
                builder = builder.model_copy(update={"container_id": identity})
                lease = self._commit_builder(lease, builder)
            self._backend_call(
                lease,
                "sandbox builder kill",
                lambda: self._backend.kill_sandbox_container(identity),
            )
            self._backend_call(
                lease,
                "sandbox builder removal",
                lambda: self._backend.remove_sandbox_container(identity),
            )
        for lookup in (identity, builder.container_name):
            if lookup is not None and (
                self._backend_call(
                    lease,
                    "sandbox builder post-removal inspection",
                    lambda lookup=lookup: self._backend.inspect_container(lookup),
                )
                is not None
            ):
                self._fail_closed(
                    lease,
                    f"sandbox builder {builder.container_name} removal cannot be "
                    "proven",
                )
        state = self._backend_call(
            lease,
            "sandbox builder volume inspection",
            lambda: self._backend.inspect_volume(builder.volume_name),
        )
        if state is not None:
            self._attest_builder_volume(lease, builder, state)
            self._remove_env_volume_proven(lease, builder.volume_name)
        if builder.state_fs == "loop-ext4":
            path = self._builder_loop_path(lease, builder_id)
            self._backend_call(
                lease,
                "sandbox builder loop device and file removal",
                lambda: self._backend.remove_builder_loop(path),
            )
            if os.path.lexists(path):
                self._fail_closed(
                    lease, f"sandbox builder loop file {path} removal is unproven"
                )
        # The bridge by ID or planned name and exact labels, then its rule.
        self._backend_call(
            lease,
            "sandbox builder network recovery",
            lambda: self._backend.recover_sandbox_network(builder, settled=settled),
        )
        return self._commit_builder(
            lease,
            builder.model_copy(
                update={
                    "state": "removed",
                    "network_id": None,
                    "loop_device": None,
                    "pending_mutation": False,
                }
            ),
        )

    def _find_builder_container(
        self, lease: ResourceLease, builder: BuilderLease
    ) -> str | None:
        """The one container holding a builder's identity, or None."""
        required = sandbox_object_labels(
            builder.owner, BUILDER_ROLE, {"sandbox-builder": builder.builder_id}
        )
        candidates: dict[str, Mapping[str, Any]] = {}
        listed = self._backend_call(
            lease,
            "sandbox builder discovery",
            lambda: self._backend.list_containers(labels=required),
        )
        for identity, state in listed:
            if self._owned_labels(state.get("labels")) == required:
                candidates[str(identity)] = state
        for lookup in (builder.container_id, builder.container_name):
            if lookup is None:
                continue
            state = self._backend_call(
                lease,
                "sandbox builder identity inspection",
                lambda lookup=lookup: self._backend.inspect_container(lookup),
            )
            if state is not None:
                candidates[str(state.get("id"))] = state
        if len(candidates) > 1:
            self._fail_closed(
                lease,
                f"sandbox builder {builder.builder_id} has ambiguous identity",
            )
        for identity, state in candidates.items():
            if not (
                _DOCKER_ID.fullmatch(identity) is not None
                and state.get("id") == identity
                and builder.container_id in (None, identity)
                and state.get("name") == builder.container_name
                and self._owned_labels(state.get("labels")) == required
            ):
                # Never remove an object only because its name looks like ours.
                self._fail_closed(
                    lease,
                    f"sandbox builder {builder.container_name} is not owned by its "
                    "planned identity",
                )
            return identity
        return None

    def _attest_builder_volume(
        self,
        lease: ResourceLease,
        builder: BuilderLease,
        state: Mapping[str, Any],
    ) -> None:
        """Only the builder's volume: exact labels, local, its state fs."""
        options = dict(state.get("options") or {})
        if builder.state_fs == "tmpfs":
            expected = {
                "type": "tmpfs",
                "device": "tmpfs",
                "o": f"size={builder.disk_mb}m",
            }
            owned = options == expected
        else:
            device = options.get("device")
            owned = (
                set(options) == {"type", "device"}
                and options.get("type") == "ext4"
                and isinstance(device, str)
                and _LOOP_DEVICE.fullmatch(device) is not None
                and builder.loop_device in (None, device)
            )
        if not (
            owned
            and state.get("name") == builder.volume_name
            and state.get("driver") == "local"
            and state.get("scope") == "local"
            and state.get("labels")
            == sandbox_object_labels(
                builder.owner,
                BUILDER_VOLUME_ROLE,
                {"sandbox-builder": builder.builder_id},
            )
        ):
            self._fail_closed(
                lease,
                f"sandbox builder volume {builder.volume_name} is not owned by its "
                "planned identity",
            )

    def _builder_loop_path(self, lease: ResourceLease, builder_id: str) -> Path:
        path = builder_loop_file(self._managed_root, lease.run_id, builder_id)
        current = self._managed_root
        for component in path.parent.relative_to(self._managed_root).parts:
            current /= component
            if current.is_symlink():
                self._fail_closed(
                    lease, "sandbox build directory is not the managed run directory"
                )
        return path

    def _sweep_sandbox_builder_orphans(self, lease: ResourceLease) -> ResourceLease:
        """Remove exact-labeled builder objects of this run the journal lacks
        (a create that landed after its builder's removal was proven): each
        becomes a removed-builder record's worth of cleanup."""
        found: dict[str, SandboxOwner] = {}
        for role, discover in (
            (BUILDER_ROLE, self._backend.list_containers),
            (BUILDER_VOLUME_ROLE, self._backend.list_volumes),
            (BUILDER_NETWORK_ROLE, self._backend.list_networks),
        ):
            for _, state in self._env_orphans(lease, role, discover):
                owner, builder_id = self._builder_label_owner(
                    lease, state.get("labels")
                )
                found.setdefault(builder_id, owner)
        known = {builder.builder_id for builder in lease.sandbox_builders}
        for builder_id, owner in found.items():
            if builder_id in known:
                continue
            LOGGER.warning("removing unjournaled sandbox builder %s", builder_id)
            volume = self._backend_call(
                lease,
                "sandbox builder orphan volume inspection",
                lambda builder_id=builder_id: self._backend.inspect_volume(
                    builder_volume_name(builder_id)
                ),
            )
            state_fs = "tmpfs"
            disk_mb = 1
            options = dict((volume or {}).get("options") or {})
            if options.get("type") == "ext4":
                state_fs = "loop-ext4"
            elif options.get("type") == "tmpfs":
                size = str(options.get("o", "")).removeprefix("size=")
                if size.endswith("m") and size[:-1].isdigit() and int(size[:-1]):
                    disk_mb = int(size[:-1])
            orphan = BuilderLease(
                owner=owner,
                builder_id=builder_id,
                container_name=builder_container_name(builder_id),
                volume_name=builder_volume_name(builder_id),
                network_name=builder_network_name(builder_id),
                rule_id=builder_rule_id(lease.run_id, builder_id),
                state_fs=state_fs,
                cpus=1,
                memory_mb=1,
                disk_mb=disk_mb,
                pending_mutation=True,
            )
            lease = lease.model_copy(
                update={"sandbox_builders": lease.sandbox_builders + (orphan,)}
            )
            self._store.write(lease)
            lease = self._remove_sandbox_builder(lease, builder_id)
        for role, discover in (
            (BUILDER_ROLE, self._backend.list_containers),
            (BUILDER_VOLUME_ROLE, self._backend.list_volumes),
            (BUILDER_NETWORK_ROLE, self._backend.list_networks),
        ):
            if self._env_orphans(lease, role, discover):
                self._fail_closed(
                    lease, f"sandbox builder orphan {role} absence is unproven"
                )
        return lease

    def _builder_label_owner(
        self, lease: ResourceLease, labels: object
    ) -> tuple[SandboxOwner, str]:
        labels = labels if isinstance(labels, Mapping) else {}
        builder_id = labels.get(_BUILDER_LABEL)
        try:
            owner = SandboxOwner(
                run_id=labels.get(_RUN_LABEL),
                task_id=labels.get(_TASK_LABEL),
                phase=labels.get(_PHASE_LABEL),
                round_id=labels.get(_ROUND_LABEL),
            )
        except ValueError:
            owner = None
        if (
            owner is None
            or (owner.run_id, owner.task_id) != (lease.run_id, lease.task_id)
            or not isinstance(builder_id, str)
            or _BUILDER_ID.fullmatch(builder_id) is None
        ):
            self._fail_closed(
                lease,
                "sandbox builder object matches only part of its label identity",
            )
        return owner, builder_id

    def _remove_sandbox_build_dir(self, lease: ResourceLease) -> None:
        """``<data_root>/<run>/sb/build``: the run's builder loop files.

        Only once no builder record remains; a file a crash left behind is
        detached from every loop device before it is removed.
        """
        if lease.sandbox_builders:
            return
        directory = builder_loop_file(
            self._managed_root, lease.run_id, "b" + "0" * 32
        ).parent
        current = self._managed_root
        for component in directory.relative_to(self._managed_root).parts:
            current /= component
            if current.is_symlink():
                self._fail_closed(
                    lease, "sandbox build directory is not the managed run directory"
                )
        if not directory.is_dir():
            return
        for entry in sorted(directory.iterdir()):
            if (
                entry.is_symlink()
                or not entry.is_file()
                or not _LOOP_FILE.fullmatch(entry.name)
            ):
                self._fail_closed(
                    lease, f"sandbox build directory holds a foreign entry {entry.name}"
                )
            self._backend_call(
                lease,
                "sandbox builder orphan loop file removal",
                lambda entry=entry: self._backend.remove_builder_loop(entry),
            )
        try:
            directory.rmdir()
        except OSError as error:
            self._fail_closed(
                lease, f"sandbox build directory removal failed: {error}", cause=error
            )

    def _recover_built_images(self, lease: ResourceLease) -> ResourceLease:
        """Remove every built image of the run, then sweep for the rest.

        A record with a journaled ID (loading, present, leaked) is untagged
        and removed without force by that ID, labelled or not (spec steps 7
        and 8: a dangling image of a journaled config digest is that ID);
        the digest was journaled before the load stream ended, so a loading
        record without an ID never reached the daemon (B8). Then every image
        carrying the run's forced build labels or a tag with the run's
        ``rsi-sbx-img`` prefix is removed, whatever the journal says (step
        9). Every recovery with environment authority sweeps, so an image a
        load registered after its record was settled is still found.
        """
        errors: list[Exception] = []
        for image in lease.sandbox_images:
            if image.kind != "built":
                continue
            try:
                if image.image_id is not None and image.state != "removed":
                    self._remove_built_image(lease, image.image_id, image.tag)
                lease = self._commit_image(
                    lease, image.model_copy(update={"state": "removed"})
                )
            except Exception as error:
                errors.append(error)
                lease = self._require_lease(lease.run_id)
        if errors:
            self._fail_closed(
                lease,
                "sandbox built image reconciliation failed: "
                + "; ".join(str(e) for e in errors),
                cause=errors[0],
            )
        required = {_RUN_LABEL: lease.run_id, _ROLE_LABEL: BUILD_ROLE}
        prefix = built_image_tag(lease.run_id, "i" + "0" * 32).rsplit("-", 1)[0] + "-"
        found: dict[str, tuple[str, ...]] = {}
        for identity, state in self._backend_call(
            lease,
            "sandbox built image label sweep",
            lambda: self._backend.list_images(labels=required),
        ):
            labels = state.get("labels") or {}
            if all(labels.get(key) == value for key, value in required.items()):
                found[str(identity)] = tuple(state.get("repo_tags") or ())
        for identity, state in self._backend_call(
            lease,
            "sandbox built image tag sweep",
            lambda: self._backend.list_tagged_images(prefix),
        ):
            tags = tuple(state.get("repo_tags") or ())
            if any(tag.startswith(prefix) for tag in tags):
                found[str(identity)] = tags
        for identity, tags in found.items():
            if any(not tag.startswith(prefix) for tag in tags):
                self._fail_closed(
                    lease,
                    f"sandbox built image {identity} carries a foreign tag",
                )
            LOGGER.warning("removing unjournaled sandbox built image %s", identity)
            for tag in tags:
                self._remove_built_image(lease, identity, tag)
            self._remove_built_image(lease, identity, None)
        return lease

    def _remove_built_image(
        self, lease: ResourceLease, image_id: str, tag: str | None
    ) -> None:
        """Untag (only a tag naming this ID), then a non-forced rmi; proven.

        An image a container still uses is left as it is and fails the
        recovery closed: its record (and so the run's reservation, which
        pool release needs every built image removed for) is kept until a
        later recovery removes it. A live broker records such an rmi
        conflict as ``leaked`` and retries it at session end instead.
        """
        state = self._backend_call(
            lease,
            "sandbox built image inspection",
            lambda: self._backend.inspect_image(image_id),
        )
        if state is not None and self._backend_call(
            lease,
            "sandbox built image reference inspection",
            lambda: self._backend.image_in_use(image_id),
        ):
            self._fail_closed(
                lease, f"sandbox built image {image_id} is still used (leaked)"
            )
        if tag is not None:
            if not tag.startswith(BUILT_IMAGE_REPOSITORY + ":"):
                self._fail_closed(lease, f"sandbox built image tag {tag} is not ours")
            tagged = self._backend_call(
                lease,
                "sandbox built image tag inspection",
                lambda: self._backend.inspect_image(tag),
            )
            if tagged is not None and tagged.get("id") == image_id:
                self._backend_call(
                    lease,
                    "sandbox built image untag",
                    lambda: self._backend.remove_image(tag),
                )
        if state is None:
            return
        self._backend_call(
            lease,
            "sandbox built image removal",
            lambda: self._backend.remove_image(image_id),
        )
        if (
            self._backend_call(
                lease,
                "sandbox built image post-removal inspection",
                lambda: self._backend.inspect_image(image_id),
            )
            is not None
        ):
            self._fail_closed(
                lease, f"sandbox built image {image_id} removal is unproven"
            )

    @staticmethod
    def _builder_record(lease: ResourceLease, builder_id: str) -> BuilderLease:
        for builder in lease.sandbox_builders:
            if builder.builder_id == builder_id:
                return builder
        raise RuntimeError(
            f"sandbox builder {builder_id} left the journal during recovery"
        )

    def _commit_builder(
        self, lease: ResourceLease, builder: BuilderLease
    ) -> ResourceLease:
        """Replace a builder record; a proven-removed one leaves the lease
        (SandboxJournal.commit_builder compacts the same way)."""
        removed = builder.state == "removed" and not builder.pending_mutation
        updated = lease.model_copy(
            update={
                "sandbox_builders": tuple(
                    builder if item.builder_id == builder.builder_id else item
                    for item in lease.sandbox_builders
                    if not (removed and item.builder_id == builder.builder_id)
                )
            }
        )
        self._store.write(updated)
        return updated

    def _commit_image(
        self, lease: ResourceLease, image: SandboxImageLease
    ) -> ResourceLease:
        removed = image.state == "removed"
        updated = lease.model_copy(
            update={
                "sandbox_images": tuple(
                    image if item.handle == image.handle else item
                    for item in lease.sandbox_images
                    if not (removed and item.handle == image.handle)
                )
            }
        )
        self._store.write(updated)
        return updated

    def _in_flight_found(
        self,
        lease: ResourceLease,
        env: SandboxEnvLease,
        flight: _InFlight,
        identities: Mapping[int, str],
    ) -> bool:
        """Whether the object of the one create in flight exists.

        Calls run one at a time, so its object existing proves every earlier
        call returned and no later one was made. With a volume and a service
        both possible, only the container proves which call was last.
        """
        if flight.network:
            required = sandbox_object_labels(
                env.owner, ENV_NETWORK_ROLE, {"sandbox-env": env.env_id}
            )
            listed = self._backend_call(
                lease,
                "sandbox env in-flight bridge discovery",
                lambda: self._backend.list_networks(labels=required),
            )
            return any(
                state.get("Name") == env.network_name
                and state.get("labels") == required
                for _, state in listed
            )
        if flight.service is not None:
            return flight.service in identities
        if flight.volume is not None:
            name = env.volumes[flight.volume].planned_name
            return (
                self._backend_call(
                    lease,
                    "sandbox env in-flight volume inspection",
                    lambda: self._backend.inspect_volume(name),
                )
                is not None
            )
        return True

    def _settle_sandbox_creates(
        self,
        lease: ResourceLease,
        envs: Sequence[SandboxEnvLease],
        in_flight: Mapping[str, _InFlight],
    ) -> None:
        """Prove that no create of these pending envs can still land.

        The broker that sent it must be gone, so it sends nothing more; the
        backend then waits for what the daemon already received and proves
        that no firewall command for a rule being installed still runs.
        """
        names = ", ".join(env.env_id for env in envs)
        if self._backend_call(
            lease,
            "sandbox coordinator liveness inspection",
            lambda: self._backend.coordinator_alive(
                lease.coordinator_pid, lease.coordinator_started_at
            ),
        ):
            self._fail_closed(
                lease,
                f"sandbox env {names} pending create outcome is unknown while "
                "its coordinator may still run; its objects are retained",
            )
        rule_ids = tuple(
            env.rule_id
            for env in envs
            if in_flight[env.env_id].network and env.rule_id is not None
        )
        if not self._backend_call(
            lease,
            "sandbox env pending create settlement",
            lambda: self._backend.settle_sandbox_creates(rule_ids),
        ):
            self._fail_closed(
                lease,
                f"sandbox env {names} pending create has not settled: a firewall "
                "command for it may still run; its objects are retained",
            )

    def _contain_sandbox_env(
        self, lease: ResourceLease, env_id: str
    ) -> tuple[ResourceLease, dict[int, str]]:
        """SIGKILL every service found with proof; returns their identities.

        Each service is contained on its own: one ambiguous or foreign
        service never leaves a provably owned sibling running.
        """
        env = self._env_record(lease, env_id)
        identities: dict[int, str] = {}
        errors: list[Exception] = []
        for record in env.services:
            if record.state == "removed":
                continue
            try:
                lease, identity = self._contain_env_service(lease, env_id, record)
            except Exception as error:
                errors.append(error)
                lease = self._require_lease(lease.run_id)
                continue
            if identity is not None:
                identities[record.idx] = identity
        if errors:
            self._fail_closed(
                lease,
                f"sandbox env {env_id} containment failed: "
                + "; ".join(str(e) for e in errors),
                cause=errors[0],
            )
        return lease, identities

    def _contain_env_service(
        self, lease: ResourceLease, env_id: str, record: SandboxEnvServiceLease
    ) -> tuple[ResourceLease, str | None]:
        env = self._env_record(lease, env_id)
        identity = self._find_env_service(lease, env, record)
        if identity is None:
            return lease, None
        if record.container_id is None:
            # Persist a late create's discovered identity before mutation.
            lease = self._commit_env(
                lease, self._env_service(env, record.idx, container_id=identity)
            )
        self._backend_call(
            lease,
            "sandbox env service kill",
            lambda: self._backend.kill_sandbox_container(identity),
        )
        stopped = self._backend_call(
            lease,
            "sandbox env service post-kill inspection",
            lambda: self._backend.inspect_container(identity),
        )
        if stopped is not None and (stopped.get("running") or stopped.get("paused")):
            self._fail_closed(
                lease,
                f"sandbox env {env_id} service {record.name} termination "
                "cannot be proven",
            )
        return lease, identity

    def _find_env_service(
        self,
        lease: ResourceLease,
        env: SandboxEnvLease,
        record: SandboxEnvServiceLease,
    ) -> str | None:
        """The one container holding a service's identity, or None."""
        required = env_container_labels(
            env.owner, env.env_id, record.name, record.image
        )
        candidates: dict[str, Mapping[str, Any]] = {}
        listed = self._backend_call(
            lease,
            "sandbox env service discovery",
            lambda: self._backend.list_containers(labels=required),
        )
        for identity, state in listed:
            labels = state.get("labels") or {}
            if all(labels.get(key) == value for key, value in required.items()):
                candidates[str(identity)] = state
        for lookup in (record.container_id, record.planned_name):
            if lookup is None:
                continue
            state = self._backend_call(
                lease,
                "sandbox env service identity inspection",
                lambda lookup=lookup: self._backend.inspect_container(lookup),
            )
            if state is None:
                continue
            identity = state.get("id")
            if not isinstance(identity, str):
                self._fail_closed(
                    lease, "sandbox env service inspection lacks actual identity"
                )
            candidates[identity] = state
        if len(candidates) > 1:
            self._fail_closed(
                lease,
                f"sandbox env {env.env_id} service {record.name} has ambiguous "
                "planned/actual identity",
            )
        for identity in candidates:
            state = self._backend_call(
                lease,
                "sandbox env service ownership inspection",
                lambda: self._backend.inspect_container(identity),
            )
            if state is None:
                self._fail_closed(
                    lease,
                    f"sandbox env {env.env_id} service {record.name} disappeared "
                    "during reconciliation",
                )
            if not (
                _DOCKER_ID.fullmatch(identity) is not None
                and state.get("id") == identity
                and (record.container_id in (None, identity))
                and state.get("name") == record.planned_name
                and state.get("image_id") == record.image_id
                and self._owned_labels(state.get("labels")) == required
            ):
                # Never remove an object only because its name looks like ours.
                self._fail_closed(
                    lease,
                    f"sandbox env {env.env_id} service {record.name} is not "
                    "owned by its planned identity",
                )
            return identity
        return None

    def _remove_sandbox_env(
        self,
        lease: ResourceLease,
        env: SandboxEnvLease,
        identities: Mapping[int, str],
        *,
        settled: bool,
    ) -> ResourceLease:
        """Containers, volumes, then bridge and rule; each proven absent.

        ``pending_mutation`` stays journaled until the env is removed, so a
        crash here never loses the pending rule; ``settled`` says recovery
        proved the one create in flight settled.
        """
        env = env.model_copy(update={"state": "stopping"})
        lease = self._commit_env(lease, env)
        for record in env.services:
            if record.state == "removed":
                continue
            identity = identities.get(record.idx)
            if identity is not None:
                self._backend_call(
                    lease,
                    "sandbox env service removal",
                    lambda: self._backend.remove_sandbox_container(identity),
                )
            for lookup in (identity, record.planned_name):
                if lookup is not None and (
                    self._backend_call(
                        lease,
                        "sandbox env service post-removal inspection",
                        lambda lookup=lookup: self._backend.inspect_container(lookup),
                    )
                    is not None
                ):
                    self._fail_closed(
                        lease,
                        f"sandbox env {env.env_id} service {record.name} "
                        "removal cannot be proven",
                    )
            env = self._env_service(env, record.idx, state="removed")
            lease = self._commit_env(lease, env)
        for volume in env.volumes:
            if not volume.created:
                continue
            self._remove_env_volume(lease, env, volume.planned_name)
            volumes = list(env.volumes)
            volumes[volume.idx] = volume.model_copy(update={"created": False})
            env = env.model_copy(update={"volumes": tuple(volumes)})
            lease = self._commit_env(lease, env)
        if env.network_name is not None:
            # M2: the bridge by ID or planned name and exact labels, then its
            # rule; both proven absent. Unless settled, M2 keeps the rule of
            # a pending bridge create that finds nothing.
            self._backend_call(
                lease,
                "sandbox env network recovery",
                lambda: self._backend.recover_sandbox_network(env, settled=settled),
            )
        return self._commit_env(
            lease,
            env.model_copy(
                update={
                    "network_id": None,
                    "state": "removed",
                    "pending_mutation": False,
                }
            ),
        )

    def _remove_env_volume(
        self, lease: ResourceLease, env: SandboxEnvLease, name: str
    ) -> None:
        state = self._backend_call(
            lease,
            "sandbox env volume inspection",
            lambda: self._backend.inspect_volume(name),
        )
        if state is None:
            return
        self._attest_env_volume(
            lease, name, state, env_volume_labels(env.owner, env.env_id)
        )
        self._remove_env_volume_proven(lease, name)

    def _attest_env_volume(
        self,
        lease: ResourceLease,
        name: str,
        state: Mapping[str, Any],
        required: Mapping[str, str],
    ) -> None:
        """Only a broker-made volume: exact labels, local, no driver options."""
        if not (
            isinstance(state, Mapping)
            and state.get("name") == name
            and state.get("labels") == required
            and state.get("driver") == "local"
            and not state.get("options")
            and state.get("scope") == "local"
        ):
            self._fail_closed(
                lease, f"sandbox env volume {name} is not owned by its planned identity"
            )

    def _remove_env_volume_proven(self, lease: ResourceLease, name: str) -> None:
        if self._backend_call(
            lease,
            "sandbox env volume reference inspection",
            lambda: self._backend.volume_in_use(name),
        ):
            self._fail_closed(lease, f"sandbox env volume {name} remains in use")
        self._backend_call(
            lease,
            "sandbox env volume removal",
            lambda: self._backend.remove_volume(name),
        )
        if (
            self._backend_call(
                lease,
                "sandbox env volume post-removal inspection",
                lambda: self._backend.inspect_volume(name),
            )
            is not None
        ):
            self._fail_closed(lease, f"sandbox env volume {name} removal is unproven")

    def _sweep_sandbox_env_orphans(self, lease: ResourceLease) -> ResourceLease:
        """Remove exact-labeled env objects of this run the journal lacks.

        Such an object is a create that finished after its env's removal was
        proven. Only this run's labels are queried; an object that matches
        them only in part is not provably ours and fails closed.
        """
        containers = self._env_orphans(lease, ENV_ROLE, self._backend.list_containers)
        for identity, _ in containers:
            state = self._backend_call(
                lease,
                "sandbox env orphan inspection",
                lambda identity=identity: self._backend.inspect_container(identity),
            )
            if state is None:
                continue
            owner, env_id = self._env_label_owner(lease, state.get("labels"))
            labels = self._owned_labels(state.get("labels"))
            name = str(state.get("name"))
            if not (
                labels
                == env_container_labels(
                    owner,
                    env_id,
                    str(labels.get(f"{LABEL_PREFIX}sandbox-service")),
                    str(labels.get(f"{LABEL_PREFIX}sandbox-image")),
                )
                and state.get("id") == identity
                and name
                in {env_container_name(env_id, idx) for idx in range(MAX_ENV_SERVICES)}
            ):
                self._fail_closed(
                    lease, f"sandbox env orphan container {name} is not provably owned"
                )
            LOGGER.warning("removing unjournaled sandbox env container %s", name)
            self._backend_call(
                lease,
                "sandbox env orphan kill",
                lambda identity=identity: self._backend.kill_sandbox_container(
                    identity
                ),
            )
            self._backend_call(
                lease,
                "sandbox env orphan removal",
                lambda identity=identity: self._backend.remove_sandbox_container(
                    identity
                ),
            )
            if (
                self._backend_call(
                    lease,
                    "sandbox env orphan post-removal inspection",
                    lambda identity=identity: self._backend.inspect_container(identity),
                )
                is not None
            ):
                self._fail_closed(
                    lease, f"sandbox env orphan container {name} removal is unproven"
                )
        volumes = self._env_orphans(lease, ENV_VOLUME_ROLE, self._backend.list_volumes)
        for name, state in volumes:
            owner, env_id = self._env_label_owner(lease, state.get("labels"))
            if name not in {
                env_volume_name(env_id, idx) for idx in range(MAX_ENV_VOLUME_LEASES)
            }:
                self._fail_closed(
                    lease, f"sandbox env orphan volume {name} is not provably owned"
                )
            self._attest_env_volume(
                lease, name, state, env_volume_labels(owner, env_id)
            )
            LOGGER.warning("removing unjournaled sandbox env volume %s", name)
            self._remove_env_volume_proven(lease, name)
        networks = self._env_orphans(
            lease, ENV_NETWORK_ROLE, self._backend.list_networks
        )
        for network_id, state in networks:
            owner, env_id = self._env_label_owner(lease, state.get("labels"))
            name = env_network_name(env_id)
            if (
                state.get("labels")
                != sandbox_object_labels(
                    owner, ENV_NETWORK_ROLE, {"sandbox-env": env_id}
                )
                or state.get("Name") != name
            ):
                self._fail_closed(
                    lease,
                    f"sandbox env orphan network {network_id} is not provably owned",
                )
            LOGGER.warning("removing unjournaled sandbox env network %s", name)
            if self._backend_call(
                lease,
                "sandbox env orphan network use inspection",
                lambda network_id=network_id: self._backend.network_in_use(network_id),
            ):
                self._fail_closed(lease, f"sandbox env orphan network {name} is in use")
            self._backend_call(
                lease,
                "sandbox env orphan network removal",
                lambda network_id=network_id: self._backend.remove_network(network_id),
            )
            # Bridge before rule, as in M2; the rule derives from (run, env).
            rule_id = env_rule_id(lease.run_id, env_id)
            if self._backend_call(
                lease,
                "sandbox env orphan rule inspection",
                lambda: self._backend.policy_exists(rule_id),
            ):
                self._backend_call(
                    lease,
                    "sandbox env orphan rule removal",
                    lambda: self._backend.remove_policy(rule_id),
                )
            if self._backend_call(
                lease,
                "sandbox env orphan rule post-removal inspection",
                lambda: self._backend.policy_exists(rule_id),
            ):
                self._fail_closed(
                    lease, f"sandbox env orphan rule {rule_id} removal is unproven"
                )
        for role, discover in (
            (ENV_ROLE, self._backend.list_containers),
            (ENV_VOLUME_ROLE, self._backend.list_volumes),
            (ENV_NETWORK_ROLE, self._backend.list_networks),
        ):
            if self._env_orphans(lease, role, discover):
                self._fail_closed(
                    lease, f"sandbox env orphan {role} absence is unproven"
                )
        return lease

    def _env_orphans(
        self,
        lease: ResourceLease,
        role: str,
        discover: Callable[..., Any],
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        required = {_RUN_LABEL: lease.run_id, _ROLE_LABEL: role}
        listed = self._backend_call(
            lease, f"{role} orphan query", lambda: discover(labels=required)
        )
        return tuple(
            (str(identity), state)
            for identity, state in listed
            if all(
                (state.get("labels") or {}).get(key) == value
                for key, value in required.items()
            )
        )

    def _env_label_owner(
        self, lease: ResourceLease, labels: object
    ) -> tuple[SandboxOwner, str]:
        """The run's owner and env named by an object's labels, or fail closed."""
        labels = labels if isinstance(labels, Mapping) else {}
        env_id = labels.get(_ENV_LABEL)
        try:
            owner = SandboxOwner(
                run_id=labels.get(_RUN_LABEL),
                task_id=labels.get(_TASK_LABEL),
                phase=labels.get(_PHASE_LABEL),
                round_id=labels.get(_ROUND_LABEL),
            )
        except ValueError:
            owner = None
        if (
            owner is None
            or (owner.run_id, owner.task_id) != (lease.run_id, lease.task_id)
            or not isinstance(env_id, str)
            or _ENV_ID.fullmatch(env_id) is None
        ):
            self._fail_closed(
                lease, "sandbox env object matches only part of its label identity"
            )
        return owner, env_id

    def _remove_sandbox_spool(self, lease: ResourceLease) -> None:
        """``<data_root>/<run>/sb/spool``: stages and exec output of the run."""
        spool = sandbox_spool_root(self._managed_root, lease.run_id)
        current = self._managed_root
        for component in spool.relative_to(self._managed_root).parts:
            current /= component
            if current.is_symlink():
                self._fail_closed(lease, "sandbox spool is not the managed run spool")
        if not os.path.lexists(spool):
            return
        self._backend_call(
            lease,
            "sandbox spool removal",
            lambda: self._backend.remove_sandbox_spool(spool),
        )
        if os.path.lexists(spool):
            self._fail_closed(lease, "sandbox spool removal is unproven")

    def _remove_sandbox_root(self, lease: ResourceLease) -> None:
        """``<data_root>/<run>/sb``, which a closed run never leaves (spec A5).

        After a crash it still holds the phase endpoints SandboxLifecycle
        made, ``<8 hex>/{s, rsi-sandbox, py/<endpoint modules>}``, for a v1
        profile run as for one with env authority; the spool and
        ``sb/build`` already went with the env and builder recovery.
        Every entry is checked before anything is removed: only those names,
        the socket ``s``, regular files and real directories. A symlink or
        any other entry fails closed with nothing removed; nothing is ever
        followed. A run without sandboxes has no ``sb``.
        """
        root = self._managed_root / lease.run_id / "sb"
        current = self._managed_root
        for component in root.relative_to(self._managed_root).parts:
            current /= component
            if current.is_symlink():
                self._fail_closed(
                    lease, "sandbox root is not the managed run directory"
                )
        if not os.path.lexists(root):
            return
        if not stat.S_ISDIR(os.lstat(root).st_mode):
            self._fail_closed(lease, "sandbox root is not the managed run directory")
        removals: list[Path] = []
        for endpoint in sorted(root.iterdir()):
            removals.extend(self._endpoint_removals(lease, endpoint))
        try:
            for path in removals:
                if stat.S_ISDIR(os.lstat(path).st_mode):
                    path.rmdir()
                else:
                    path.unlink()
            root.rmdir()
        except OSError as error:
            self._fail_closed(
                lease, f"sandbox root removal failed: {error}", cause=error
            )

    def _endpoint_removals(self, lease: ResourceLease, endpoint: Path) -> list[Path]:
        """The paths of one endpoint directory, deepest first; anything but
        exactly its known content fails closed."""

        def kind(path: Path) -> int:
            return stat.S_IFMT(os.lstat(path).st_mode)

        if (
            _ENDPOINT_DIRECTORY.fullmatch(endpoint.name) is None
            or kind(endpoint) != stat.S_IFDIR
        ):
            self._fail_closed(
                lease, f"sandbox root holds a foreign entry {endpoint.name}"
            )
        files: list[Path] = []
        directories: list[Path] = []
        for entry in sorted(endpoint.iterdir()):
            found = kind(entry)
            if (entry.name, found) in (
                ("s", stat.S_IFSOCK),
                ("rsi-sandbox", stat.S_IFREG),
            ):
                files.append(entry)
            elif (entry.name, found) == ("py", stat.S_IFDIR):
                for module in sorted(entry.iterdir()):
                    if module.name not in ENDPOINT_MODULES or kind(module) != (
                        stat.S_IFREG
                    ):
                        self._fail_closed(
                            lease,
                            "sandbox endpoint holds a foreign entry "
                            f"{endpoint.name}/py/{module.name}",
                        )
                    files.append(module)
                directories.append(entry)
            else:
                self._fail_closed(
                    lease,
                    f"sandbox endpoint holds a foreign entry {endpoint.name}/"
                    f"{entry.name}",
                )
        return [*files, *directories, endpoint]

    @staticmethod
    def _owned_labels(labels: object) -> dict[str, str]:
        if not isinstance(labels, Mapping):
            return {}
        return {
            key: value for key, value in labels.items() if key.startswith(LABEL_PREFIX)
        }

    @staticmethod
    def _env_record(lease: ResourceLease, env_id: str) -> SandboxEnvLease:
        for env in lease.sandbox_envs:
            if env.env_id == env_id:
                return env
        raise RuntimeError(f"sandbox env {env_id} left the journal during recovery")

    @staticmethod
    def _env_service(
        env: SandboxEnvLease, index: int, **updates: object
    ) -> SandboxEnvLease:
        services = list(env.services)
        services[index] = services[index].model_copy(update=updates)
        return env.model_copy(update={"services": tuple(services)})

    def _commit_env(self, lease: ResourceLease, env: SandboxEnvLease) -> ResourceLease:
        """Replace an env record; a proven-removed env leaves the lease
        (SandboxJournal.commit_env compacts the same way)."""
        removed = env.state == "removed" and not env.pending_mutation
        updated = lease.model_copy(
            update={
                "sandbox_envs": tuple(
                    env if item.env_id == env.env_id else item
                    for item in lease.sandbox_envs
                    if not (removed and item.env_id == env.env_id)
                )
            }
        )
        self._store.write(updated)
        return updated

    def _stop_and_remove_judge(
        self,
        lease: ResourceLease,
        judge_id: str,
        state: Mapping[str, Any],
    ) -> None:
        if bool(state.get("running", False)):
            self._backend_call(
                lease,
                "Judge stop",
                lambda: self._backend.stop_container(judge_id),
            )
        inspected = self._backend_call(
            lease,
            "Judge inspection",
            lambda: self._backend.inspect_container(judge_id),
        )
        if inspected is not None and bool(inspected.get("running", False)):
            self._fail_closed(lease, "Judge containment cannot be proven")
        self._backend_call(
            lease,
            "Judge removal",
            lambda: self._backend.remove_container(judge_id),
        )
        if (
            self._backend_call(
                lease,
                "Judge post-removal inspection",
                lambda: self._backend.inspect_container(judge_id),
            )
            is not None
        ):
            self._fail_closed(lease, "Judge removal cannot be proven")

    def _stop_and_remove_helper(
        self,
        lease: ResourceLease,
        helper_id: str,
        state: Mapping[str, Any],
    ) -> None:
        if bool(state.get("running", False)):
            self._backend_call(
                lease,
                "cleanup helper stop",
                lambda: self._backend.stop_container(helper_id),
            )
        inspected = self._backend_call(
            lease,
            "cleanup helper inspection",
            lambda: self._backend.inspect_container(helper_id),
        )
        if inspected is not None and bool(inspected.get("running", False)):
            self._fail_closed(lease, "cleanup helper containment cannot be proven")
        self._backend_call(
            lease,
            "cleanup helper removal",
            lambda: self._backend.remove_container(helper_id),
        )
        if (
            self._backend_call(
                lease,
                "cleanup helper post-removal inspection",
                lambda: self._backend.inspect_container(helper_id),
            )
            is not None
        ):
            self._fail_closed(lease, "cleanup helper removal cannot be proven")

    def _contain_helpers_after_delete_failure(
        self,
        lease: ResourceLease,
        *,
        delete_error: Exception,
    ) -> None:
        """Contain a helper that may have been created before its run failed."""

        containment_errors: list[Exception] = []
        try:
            helpers = self._find_containers(lease, "helper")
        except Exception as error:
            helpers = ()
            containment_errors.append(error)
        for helper in helpers:
            try:
                self._stop_and_remove_helper(lease, *helper)
            except Exception as error:
                containment_errors.append(error)
        try:
            remaining = self._find_containers(lease, "helper")
        except Exception as error:
            remaining = ()
            containment_errors.append(error)
        if remaining:
            containment_errors.append(
                RuntimeError(
                    "cleanup helper absence cannot be proven after workspace "
                    "deletion failure"
                )
            )

        detail = f"managed workspace deletion failed: {delete_error}"
        if containment_errors:
            failures = "; ".join(str(error) for error in containment_errors)
            detail += f"; cleanup helper containment is unproven: {failures}"
        self._fail_closed(lease, detail, cause=delete_error)

    def _recover_policies(
        self, lease: ResourceLease, role: Literal["judge", "work"]
    ) -> ResourceLease:
        owned = lease.judge if role == "judge" else lease.work
        rule_ids = tuple(
            dict.fromkeys(
                rule_id
                for rule_id in (
                    owned.policy_rule_id,
                    owned.planned_policy_rule_id,
                )
                if rule_id is not None
            )
        )
        for rule_id in rule_ids:
            if self._backend_call(
                lease,
                f"{role} firewall policy inspection",
                lambda rule_id=rule_id: self._backend.policy_exists(rule_id),
            ):
                self._backend_call(
                    lease,
                    f"{role} firewall policy removal",
                    lambda rule_id=rule_id: self._backend.remove_policy(rule_id),
                )
            if self._backend_call(
                lease,
                f"{role} firewall policy post-removal inspection",
                lambda rule_id=rule_id: self._backend.policy_exists(rule_id),
            ):
                self._fail_closed(lease, f"{role} firewall policy removal is unproven")
        if not rule_ids:
            return lease
        if role == "judge":
            return self._update_judge(
                lease, planned_policy_rule_id=None, policy_rule_id=None
            )
        return self._update_work(
            lease, planned_policy_rule_id=None, policy_rule_id=None
        )

    def _recover_networks(
        self, lease: ResourceLease, role: Literal["judge", "work"]
    ) -> ResourceLease:
        for network_id, _ in self._find_networks(lease, role):
            if self._backend_call(
                lease,
                f"{role} network use inspection",
                lambda network_id=network_id: self._backend.network_in_use(network_id),
            ):
                self._fail_closed(lease, f"{role} network remains in use")
            self._backend_call(
                lease,
                f"{role} network removal",
                lambda network_id=network_id: self._backend.remove_network(network_id),
            )
        if self._find_networks(lease, role):
            self._fail_closed(lease, f"{role} network removal is unproven")
        updates = {
            "planned_network": None,
            "network_id": None,
            "network_name": None,
        }
        if role == "judge":
            return self._update_judge(lease, **updates)
        return self._update_work(lease, **updates)

    def _recover_snapshots(self, lease: ResourceLease) -> ResourceLease:
        judge = lease.judge
        has_path_authority = any(
            value is not None
            for value in (
                judge.snapshot_merged_path,
                judge.snapshot_process_id,
            )
        )
        legacy_planned_only = (
            judge.planned_snapshot is not None and judge.planned_snapshot_ref is None
        )
        if not has_path_authority and not legacy_planned_only:
            return lease
        discovered = self._backend_call(
            lease,
            "snapshot authority discovery",
            lambda: self._backend.discover_snapshot_leases(
                run_id=lease.run_id,
                round_id=judge.round_id,
                lease_id=judge.snapshot_lease_id,
            ),
        )
        if len(discovered) > 1:
            self._fail_closed(lease, "ambiguous active snapshot authority")
        for authority in discovered:
            self._backend_call(
                lease,
                "snapshot release",
                lambda authority=authority: self._backend.release_snapshot(authority),
            )
            if self._backend_call(
                lease,
                "snapshot mount inspection",
                lambda authority=authority: self._backend.is_mounted(
                    authority.merged_path
                ),
            ):
                self._fail_closed(lease, "snapshot containment release is unproven")
        remaining = self._backend_call(
            lease,
            "snapshot post-release discovery",
            lambda: self._backend.discover_snapshot_leases(
                run_id=lease.run_id,
                round_id=judge.round_id,
                lease_id=judge.snapshot_lease_id,
            ),
        )
        if remaining:
            self._fail_closed(lease, "snapshot durable authority remains")
        if judge.snapshot_merged_path is not None and self._backend_call(
            lease,
            "planned snapshot mount inspection",
            lambda: self._backend.is_mounted(judge.snapshot_merged_path),
        ):
            self._fail_closed(lease, "snapshot mount remains without a manifest")
        return self._update_judge(
            lease,
            planned_snapshot=None,
            snapshot_lease_id=None,
            snapshot_merged_path=None,
            snapshot_process_id=None,
        )

    def _recover_round_images(self, lease: ResourceLease) -> ResourceLease:
        judge = lease.judge
        has_authority = any(
            value is not None
            for value in (
                judge.planned_snapshot_ref,
                judge.snapshot_image_id,
                judge.snapshot_image_ref,
                judge.snapshot_source_container_id,
            )
        )
        if not has_authority:
            return lease
        if judge.round_id is None or judge.planned_snapshot_ref is None:
            self._fail_closed(lease, "Judge snapshot image authority is incomplete")
        source_container_id = judge.snapshot_source_container_id
        if source_container_id is None:
            live_work = self._find_container(lease, "work")
            source_container_id = (
                live_work[0] if live_work is not None else lease.work.container_id
            )
        required = self._image_labels(
            lease,
            role=_ROUND_IMAGE_ROLE,
            round_id=judge.round_id,
        )
        candidates = self._find_images(
            lease,
            required=required,
            source_container_id=source_container_id,
        )
        expected_id = judge.snapshot_image_id
        expected_ref = judge.snapshot_image_ref or judge.planned_snapshot_ref
        if expected_id is not None:
            direct = self._backend_call(
                lease,
                "Judge snapshot image inspection",
                lambda: self._backend.inspect_image(expected_id),
            )
            if direct is None:
                if candidates:
                    self._fail_closed(
                        lease,
                        "Judge snapshot image ID differs from exact-labeled images",
                    )
                return self._clear_round_image_authority(lease)
            self._attest_image_state(
                lease,
                image_id=expected_id,
                state=direct,
                required=required,
                source_container_id=source_container_id,
                expected_ref=expected_ref,
            )
            if expected_id not in dict(candidates):
                self._fail_closed(
                    lease,
                    "Judge snapshot image listing omitted durable image authority",
                )
        elif candidates and not any(
            expected_ref in state["repo_tags"] for _, state in candidates
        ):
            self._fail_closed(
                lease,
                "planned Judge snapshot reference does not match discovered image",
            )
        self._remove_images(
            lease,
            candidates=candidates,
            required=required,
            source_container_id=source_container_id,
            purpose="Judge snapshot",
        )
        return self._clear_round_image_authority(lease)

    def _clear_round_image_authority(self, lease: ResourceLease) -> ResourceLease:
        return self._update_judge(
            lease,
            planned_snapshot=None,
            planned_snapshot_ref=None,
            snapshot_lease_id=None,
            snapshot_image_id=None,
            snapshot_image_ref=None,
            snapshot_source_container_id=None,
        )

    def _recover_retained_images(
        self,
        lease: ResourceLease,
        *,
        delete: bool,
        source_container_id: str | None,
    ) -> ResourceLease:
        work = lease.work
        rollback = work.retained_image_rollback
        has_authority = any(
            value is not None
            for value in (
                work.planned_retained_image_ref,
                work.retained_image_id,
                work.retained_image_ref,
                rollback,
            )
        )
        if not has_authority:
            return lease
        if work.planned_retained_image_ref is None:
            self._fail_closed(lease, "retained Work image authority is incomplete")
        required = self._image_labels(
            lease,
            role=_RETAINED_IMAGE_ROLE,
            round_id="final",
        )
        candidates = self._find_images(
            lease,
            required=required,
            source_container_id=source_container_id,
        )
        candidate_count = len(candidates)
        if candidate_count > 1:
            self._fail_closed(lease, "ambiguous retained Work image authority")

        expected_id = (
            rollback.image_id if rollback is not None else work.retained_image_id
        )
        expected_ref = (
            rollback.image_ref
            if rollback is not None
            else work.retained_image_ref or work.planned_retained_image_ref
        )
        direct_authority = expected_id or expected_ref
        if direct_authority is None:
            self._fail_closed(lease, "retained Work image authority is incomplete")
        direct = self._backend_call(
            lease,
            "retained Work image inspection",
            lambda: self._backend.inspect_image(direct_authority),
        )

        if candidate_count == 0:
            if direct is not None:
                direct_id = direct.get("id")
                if not isinstance(direct_id, str):
                    self._fail_closed(lease, "rootfs image attestation failed")
                self._attest_image_state(
                    lease,
                    image_id=direct_id,
                    state=direct,
                    required=required,
                    source_container_id=source_container_id,
                    expected_ref=expected_ref,
                )
                self._fail_closed(
                    lease,
                    "retained Work image listing omitted durable image authority",
                )
            return self._clear_retained_image_authority(lease)

        # candidate_count == 1: preflight the sole image before preserving or
        # removing it; removal performs its own post-removal absence proof.
        candidate_id, _ = candidates[0]
        if direct is None:
            self._fail_closed(
                lease,
                "retained Work image authority differs from exact-labeled image",
            )
        direct_id = direct.get("id")
        if not isinstance(direct_id, str):
            self._fail_closed(lease, "rootfs image attestation failed")
        self._attest_image_state(
            lease,
            image_id=direct_id,
            state=direct,
            required=required,
            source_container_id=source_container_id,
            expected_ref=expected_ref,
        )
        if direct_id != candidate_id or (
            expected_id is not None and direct_id != expected_id
        ):
            self._fail_closed(
                lease,
                "retained Work image authority differs from exact-labeled image",
            )

        must_remove = delete or rollback is not None
        if must_remove:
            self._remove_images(
                lease,
                candidates=candidates,
                required=required,
                source_container_id=source_container_id,
                purpose="retained Work",
            )
            return self._clear_retained_image_authority(lease)
        for image_id, _ in candidates:
            if self._backend_call(
                lease,
                "retained Work image reference inspection",
                lambda image_id=image_id: self._backend.image_in_use(image_id),
            ):
                self._fail_closed(lease, "retained Work image remains referenced")
        if expected_id is None:
            return self._update_work(
                lease,
                retained_image_id=candidate_id,
                retained_image_ref=work.planned_retained_image_ref,
            )
        return lease

    def _clear_retained_image_authority(self, lease: ResourceLease) -> ResourceLease:
        return self._update_work(
            lease,
            planned_retained_image_ref=None,
            retained_image_id=None,
            retained_image_ref=None,
            retained_image_rollback=None,
        )

    def _recover_workdir_volume(
        self,
        lease: ResourceLease,
        *,
        delete: bool,
        preflight_only: bool = False,
    ) -> ResourceLease:
        if lease.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS:
            return lease
        owned = lease.work.workdir_volume
        has_authority = any(
            value is not None
            for value in (
                owned.planned_name,
                owned.planned_target,
                owned.planned_snapshot_mode,
                owned.planned_freshness_nonce,
                owned.actual,
                owned.rollback,
            )
        )
        if not has_authority:
            return lease
        if (
            owned.planned_name is None
            or owned.planned_target is None
            or owned.planned_snapshot_mode is None
            or owned.planned_freshness_nonce is None
        ):
            self._fail_closed(lease, "WORKDIR volume authority is incomplete")
        authority = owned.rollback or owned.actual
        if authority is None:
            authority = ManagedWorkdirVolume(
                name=owned.planned_name,
                run_id=lease.run_id,
                task_id=lease.task_id,
                target=owned.planned_target,
                snapshot_mode=owned.planned_snapshot_mode,
                freshness_nonce=owned.planned_freshness_nonce,
            )
        required = self._volume_labels(authority)
        direct = self._backend_call(
            lease,
            "WORKDIR volume direct-name inspection",
            lambda: self._backend.inspect_volume(authority.name),
        )
        labeled = self._find_volumes(lease, required=required, expected=authority)
        must_remove = delete or owned.rollback is not None or owned.actual is None

        if direct is None and not labeled:
            if preflight_only:
                if owned.actual is not None:
                    self._fail_closed(
                        lease,
                        "final WORKDIR volume durable authority is absent",
                    )
                return lease
            if owned.actual is not None and not must_remove:
                self._fail_closed(
                    lease, "final WORKDIR volume durable authority is absent"
                )
            return self._clear_workdir_volume_authority(lease)
        if direct is None or len(labeled) != 1:
            self._fail_closed(
                lease,
                "direct-name and exact-labeled WORKDIR volume authority differ",
            )
        self._attest_volume_state(
            lease, name=authority.name, state=direct, expected=authority
        )
        labeled_name, labeled_state = labeled[0]
        if labeled_name != authority.name or direct != labeled_state:
            self._fail_closed(
                lease,
                "direct-name and exact-labeled WORKDIR volume authority differ",
            )
        if self._backend_call(
            lease,
            "WORKDIR volume container reference inspection",
            lambda: self._backend.volume_in_use(authority.name),
        ):
            self._fail_closed(lease, "WORKDIR volume remains referenced")
        if preflight_only:
            return lease
        if not must_remove:
            return lease

        self._backend_call(
            lease,
            "WORKDIR volume removal",
            lambda: self._backend.remove_volume(authority.name),
        )
        if (
            self._backend_call(
                lease,
                "WORKDIR volume post-removal direct-name inspection",
                lambda: self._backend.inspect_volume(authority.name),
            )
            is not None
        ):
            self._fail_closed(lease, "WORKDIR volume removal is unproven")
        if self._find_volumes(lease, required=required, expected=authority):
            self._fail_closed(
                lease,
                "WORKDIR volume exact-labeled absence is unproven",
            )
        return self._clear_workdir_volume_authority(lease)

    def _find_volumes(
        self,
        lease: ResourceLease,
        *,
        required: Mapping[str, str],
        expected: ManagedWorkdirVolume,
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        listed = self._backend_call(
            lease,
            "WORKDIR volume exact-label query",
            lambda: self._backend.list_volumes(labels=required),
        )
        found: list[tuple[str, Mapping[str, Any]]] = []
        seen: set[str] = set()
        for raw_name, state in listed:
            name = str(raw_name)
            if name in seen:
                self._fail_closed(lease, "duplicate WORKDIR volume query identity")
            seen.add(name)
            self._attest_volume_state(lease, name=name, state=state, expected=expected)
            found.append((name, state))
        if len(found) > 1:
            self._fail_closed(lease, "ambiguous WORKDIR volume authority")
        return tuple(found)

    def _attest_volume_state(
        self,
        lease: ResourceLease,
        *,
        name: str,
        state: Mapping[str, Any],
        expected: ManagedWorkdirVolume,
    ) -> None:
        if not isinstance(state, Mapping):
            self._fail_closed(lease, "WORKDIR volume attestation failed")
        if name != expected.name:
            self._fail_closed(lease, "WORKDIR volume attestation failed")
        try:
            attest_normalized_workdir_volume_state(
                state, expected, expected_references=()
            )
        except InfrastructureError as error:
            self._fail_closed(lease, "WORKDIR volume attestation failed", cause=error)

    @staticmethod
    def _volume_labels(volume: ManagedWorkdirVolume) -> dict[str, str]:
        return managed_workdir_volume_labels(volume)

    def _clear_workdir_volume_authority(self, lease: ResourceLease) -> ResourceLease:
        return self._update_work(lease, workdir_volume=WorkdirVolumeResourceLease())

    def _find_images(
        self,
        lease: ResourceLease,
        *,
        required: Mapping[str, str],
        source_container_id: str | None,
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        listed = self._backend_call(
            lease,
            "image query",
            lambda: self._backend.list_images(labels=required),
        )
        found: list[tuple[str, Mapping[str, Any]]] = []
        seen: set[str] = set()
        for raw_id, state in listed:
            image_id = str(raw_id)
            if image_id in seen:
                self._fail_closed(lease, "duplicate image query identity")
            seen.add(image_id)
            self._attest_image_state(
                lease,
                image_id=image_id,
                state=state,
                required=required,
                source_container_id=source_container_id,
            )
            inspected = self._backend_call(
                lease,
                "image inspection",
                lambda image_id=image_id: self._backend.inspect_image(image_id),
            )
            if inspected is None:
                continue
            self._attest_image_state(
                lease,
                image_id=image_id,
                state=inspected,
                required=required,
                source_container_id=source_container_id,
            )
            if state.get("labels") != inspected.get("labels") or state.get(
                "repo_tags"
            ) != inspected.get("repo_tags"):
                self._fail_closed(
                    lease, "image listing and inspection authority differ"
                )
            found.append((image_id, inspected))
        return tuple(found)

    def _attest_image_state(
        self,
        lease: ResourceLease,
        *,
        image_id: str,
        state: Mapping[str, Any],
        required: Mapping[str, str],
        source_container_id: str | None,
        expected_ref: str | None = None,
    ) -> None:
        labels = state.get("labels")
        repo_tags = state.get("repo_tags")
        source = (
            labels.get(_SOURCE_CONTAINER_LABEL) if isinstance(labels, Mapping) else None
        )
        valid = (
            _JUDGE_IMAGE_ID.fullmatch(image_id) is not None
            and state.get("id") == image_id
            and isinstance(labels, Mapping)
            and all(labels.get(key) == value for key, value in required.items())
            and isinstance(source, str)
            and _SAFE_ID.fullmatch(source) is not None
            and (source_container_id is None or source == source_container_id)
            and isinstance(repo_tags, (tuple, list))
            and all(isinstance(tag, str) for tag in repo_tags)
            and (expected_ref is None or expected_ref in repo_tags)
        )
        if not valid:
            self._fail_closed(lease, "rootfs image attestation failed")

    def _remove_images(
        self,
        lease: ResourceLease,
        *,
        candidates: tuple[tuple[str, Mapping[str, Any]], ...],
        required: Mapping[str, str],
        source_container_id: str | None,
        purpose: str,
    ) -> None:
        for image_id, _ in candidates:
            if self._backend_call(
                lease,
                f"{purpose} image reference inspection",
                lambda image_id=image_id: self._backend.image_in_use(image_id),
            ):
                self._fail_closed(lease, f"{purpose} image remains referenced")
        for image_id, _ in candidates:
            self._backend_call(
                lease,
                f"{purpose} image removal",
                lambda image_id=image_id: self._backend.remove_image(image_id),
            )
            if (
                self._backend_call(
                    lease,
                    f"{purpose} image absence query",
                    lambda image_id=image_id: self._backend.inspect_image(image_id),
                )
                is not None
            ):
                self._fail_closed(lease, f"{purpose} image removal is unproven")
        if self._find_images(
            lease,
            required=required,
            source_container_id=source_container_id,
        ):
            self._fail_closed(
                lease, f"{purpose} exact-labeled image absence is unproven"
            )

    @staticmethod
    def _image_labels(
        lease: ResourceLease, *, role: str, round_id: str
    ) -> dict[str, str]:
        return {
            _RUN_LABEL: lease.run_id,
            _TASK_LABEL: lease.task_id,
            _ROLE_LABEL: role,
            _ROUND_LABEL: round_id,
        }

    def _find_container(
        self, lease: ResourceLease, role: str
    ) -> tuple[str, Mapping[str, Any]] | None:
        matched = self._find_containers(lease, role)
        if len(matched) > 1:
            self._fail_closed(
                lease,
                f"ambiguous labeled {role} containers for {lease.run_id}",
            )
        return matched[0] if matched else None

    def _find_containers(
        self, lease: ResourceLease, role: str
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        required = self._labels(lease, role)
        matched: list[tuple[str, Mapping[str, Any]]] = []
        containers = self._backend_call(
            lease,
            f"{role} container listing",
            lambda: self._backend.list_containers(labels=required),
        )
        for raw_id, state in containers:
            labels = state.get("labels", {})
            if all(labels.get(key) == value for key, value in required.items()):
                container_id = str(raw_id)
                inspected = self._backend_call(
                    lease,
                    f"{role} container inspection",
                    lambda container_id=container_id: self._backend.inspect_container(
                        container_id
                    ),
                )
                if inspected is not None:
                    matched.append((container_id, inspected))
        return tuple(matched)

    def _find_networks(
        self, lease: ResourceLease, role: str
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]:
        required = self._labels(lease, role)
        networks = self._backend_call(
            lease,
            f"{role} network listing",
            lambda: self._backend.list_networks(labels=required),
        )
        return tuple(
            (str(network_id), state)
            for network_id, state in networks
            if all(
                state.get("labels", {}).get(key) == value
                for key, value in required.items()
            )
        )

    @staticmethod
    def _labels(lease: ResourceLease, role: str) -> dict[str, str]:
        return {
            _RUN_LABEL: lease.run_id,
            _TASK_LABEL: lease.task_id,
            _ROLE_LABEL: role,
        }

    def _update_work(self, lease: ResourceLease, **updates: object) -> ResourceLease:
        updated = lease.model_copy(
            update={"work": lease.work.model_copy(update=updates)}
        )
        self._store.write(updated)
        return updated

    def _update_judge(self, lease: ResourceLease, **updates: object) -> ResourceLease:
        updated = lease.model_copy(
            update={"judge": lease.judge.model_copy(update=updates)}
        )
        self._store.write(updated)
        return updated

    def _backend_call(
        self,
        lease: ResourceLease,
        operation: str,
        callback: Callable[[], Any],
    ) -> Any:
        try:
            return callback()
        except Exception as error:
            self._fail_closed(
                lease,
                f"{operation} failed: {error}",
                cause=error,
            )

    def _fail_closed(
        self,
        lease: ResourceLease,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        safe_message = redact_text(message)
        retained = lease.model_copy(
            update={
                "recovery_required": True,
                "error": safe_message,
            }
        )
        self._store.write(retained)
        error = RuntimeError(f"{safe_message}; runtime authority remains retained")
        if cause is None:
            raise error
        raise error from cause

    def _require_lease(self, run_id: str) -> ResourceLease:
        lease = self._store.read(run_id)
        if lease is None:
            raise RuntimeError(f"lease disappeared for {run_id}")
        return lease

    def _validated_workspace(self, lease: ResourceLease) -> Path:
        if lease.workspace_path is None:
            raise ValueError("lease has no managed workspace path")
        lexical = lease.workspace_path
        expected = self._managed_root / lease.run_id / "workspace"
        if lexical != expected or lexical.is_symlink():
            raise ValueError("workspace is not the exact managed run workspace")
        resolved = lexical.resolve(strict=False)
        expected_resolved = expected.resolve(strict=False)
        if (
            resolved != expected_resolved
            or resolved == self._managed_root
            or self._managed_root not in resolved.parents
        ):
            raise ValueError("workspace is not the exact managed run workspace")
        current = self._managed_root
        for component in lexical.relative_to(self._managed_root).parts:
            current /= component
            if current.is_symlink():
                raise ValueError("workspace is not the exact managed run workspace")
        return lexical


__all__ = [
    "JudgeResourceLease",
    "LeaseStore",
    "RecoveryBackend",
    "RecoveryManager",
    "RetainedImageRollbackAuthority",
    "ResourceLease",
    "SnapshotRecoveryAuthority",
    "WorkdirVolumeResourceLease",
    "WorkResourceLease",
]
