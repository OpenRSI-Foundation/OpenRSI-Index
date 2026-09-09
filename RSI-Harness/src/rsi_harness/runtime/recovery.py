"""Durable run resource leases, crash recovery, and contained cleanup."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
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
_RETAINED_IMAGE_REF = re.compile(
    r"rsi-harness-rootfs:retained-work-[0-9a-f]{64}\Z"
)
_JUDGE_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_JUDGE_IMAGE_REF = re.compile(
    r"rsi-harness-rootfs:judge-round-[0-9a-f]{64}\Z"
)


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
    def valid_planned_target(
        cls, value: PurePosixPath | None
    ) -> PurePosixPath | None:
        if value is None:
            return None
        if not value.is_absolute():
            raise ValueError("planned WORKDIR volume target must be absolute")
        if value == PurePosixPath("/"):
            raise ValueError(
                "planned WORKDIR volume requires a non-root target"
            )
        if ".." in value.parts:
            raise ValueError(
                "planned WORKDIR volume target contains lexical traversal"
            )
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
                raise ValueError(
                    "actual WORKDIR volume mode differs from planned mode"
                )
            if self.actual.freshness_nonce != self.planned_freshness_nonce:
                raise ValueError(
                    "actual WORKDIR volume freshness differs from planned authority"
                )
        if self.rollback is not None:
            if self.planned_name is None:
                raise ValueError(
                    "rollback WORKDIR volume requires planned authority"
                )
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

    _absolute_snapshot_path = field_validator(
        "snapshot_merged_path", mode="after"
    )(_absolute_optional_path)

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
            raise ValueError(
                "Judge snapshot path and image authority cannot be mixed"
            )
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
                raise ValueError(
                    "actual Judge snapshot reference differs from planned"
                )
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

    schema_version: int = 4
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
        if self.schema_version != 4:
            raise ValueError("unsupported resource lease schema version")
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


class RecoveryBackend(Protocol):
    """Authoritative production inspection/mutation port used by recovery."""

    def list_containers(
        self, *, labels: Mapping[str, str]
    ) -> tuple[tuple[str, Mapping[str, Any]], ...]: ...

    def inspect_container(self, container_id: str) -> Mapping[str, Any] | None: ...

    def stop_container(self, container_id: str) -> None: ...

    def remove_container(self, container_id: str) -> None: ...

    def list_images(
        self, *, labels: Mapping[str, str]
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
        return ResourceLease.model_validate_json(path.read_text())

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
    ) -> None:
        self._store = store
        self._backend = backend
        self._managed_root = Path(managed_root).resolve()

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
                self._fail_closed(
                    current, "workspace deletion blocked by live helper"
                )
            if self._find_container(current, "judge") is not None:
                self._fail_closed(
                    current, "workspace deletion blocked by live Judge"
                )
            if self._find_container(current, "work") is not None:
                self._fail_closed(
                    current, "workspace deletion blocked by live Work"
                )
            for role in ("judge", "work"):
                if self._find_networks(current, role):
                    self._fail_closed(
                        current,
                        "workspace deletion blocked by a live "
                        f"{role} network",
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
                self._fail_closed(
                    current, "workspace deletion blocked by a live mount"
                )
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
            if self._backend_call(
                lease,
                "Work post-removal inspection",
                lambda: self._backend.inspect_container(work_id),
            ) is not None:
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
        lease = self._recover_workdir_volume(
            lease, delete=delete_retained_image
        )

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
            }
        )
        self._store.write(final)

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
        if self._backend_call(
            lease,
            "Judge post-removal inspection",
            lambda: self._backend.inspect_container(judge_id),
        ) is not None:
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
            self._fail_closed(
                lease, "cleanup helper containment cannot be proven"
            )
        self._backend_call(
            lease,
            "cleanup helper removal",
            lambda: self._backend.remove_container(helper_id),
        )
        if self._backend_call(
            lease,
            "cleanup helper post-removal inspection",
            lambda: self._backend.inspect_container(helper_id),
        ) is not None:
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
                self._fail_closed(
                    lease, f"{role} firewall policy removal is unproven"
                )
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
                lambda network_id=network_id: self._backend.network_in_use(
                    network_id
                ),
            ):
                self._fail_closed(lease, f"{role} network remains in use")
            self._backend_call(
                lease,
                f"{role} network removal",
                lambda network_id=network_id: self._backend.remove_network(
                    network_id
                ),
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
            judge.planned_snapshot is not None
            and judge.planned_snapshot_ref is None
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
                lambda authority=authority: self._backend.release_snapshot(
                    authority
                ),
            )
            if self._backend_call(
                lease,
                "snapshot mount inspection",
                lambda authority=authority: self._backend.is_mounted(
                    authority.merged_path
                ),
            ):
                self._fail_closed(
                    lease, "snapshot containment release is unproven"
                )
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
        if (
            judge.snapshot_merged_path is not None
            and self._backend_call(
                lease,
                "planned snapshot mount inspection",
                lambda: self._backend.is_mounted(judge.snapshot_merged_path),
            )
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
                self._fail_closed(
                    lease, "retained Work image remains referenced"
                )
        if expected_id is None:
            return self._update_work(
                lease,
                retained_image_id=candidate_id,
                retained_image_ref=work.planned_retained_image_ref,
            )
        return lease

    def _clear_retained_image_authority(
        self, lease: ResourceLease
    ) -> ResourceLease:
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
        labeled = self._find_volumes(
            lease, required=required, expected=authority
        )
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
        if self._backend_call(
            lease,
            "WORKDIR volume post-removal direct-name inspection",
            lambda: self._backend.inspect_volume(authority.name),
        ) is not None:
            self._fail_closed(lease, "WORKDIR volume removal is unproven")
        if self._find_volumes(
            lease, required=required, expected=authority
        ):
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
            self._attest_volume_state(
                lease, name=name, state=state, expected=expected
            )
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
            self._fail_closed(
                lease, "WORKDIR volume attestation failed", cause=error
            )

    @staticmethod
    def _volume_labels(volume: ManagedWorkdirVolume) -> dict[str, str]:
        return managed_workdir_volume_labels(volume)

    def _clear_workdir_volume_authority(
        self, lease: ResourceLease
    ) -> ResourceLease:
        return self._update_work(
            lease, workdir_volume=WorkdirVolumeResourceLease()
        )

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
            if (
                state.get("labels") != inspected.get("labels")
                or state.get("repo_tags") != inspected.get("repo_tags")
            ):
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
            labels.get(_SOURCE_CONTAINER_LABEL)
            if isinstance(labels, Mapping)
            else None
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
            if self._backend_call(
                lease,
                f"{purpose} image absence query",
                lambda image_id=image_id: self._backend.inspect_image(image_id),
            ) is not None:
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
        error = RuntimeError(
            f"{safe_message}; runtime authority remains retained"
        )
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
