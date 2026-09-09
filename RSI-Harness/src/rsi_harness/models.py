"""Immutable domain contracts shared by the compiler and runtime."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import core_schema

VERIFIER_OUTPUT_TRUNCATION_MARKER = "\n...[verifier output truncated]...\n"
WORK_FEEDBACK_ROOT = PurePosixPath("/run/rsi-harness/feedback")
MIN_VERIFIER_OUTPUT_LIMIT_BYTES = len(
    VERIFIER_OUTPUT_TRUNCATION_MARKER.encode("utf-8")
)


class PersistedModel(BaseModel):
    """Base class for data written to plans, reports, and resource leases."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class FrozenRewardMap(Mapping[str, float]):
    """An immutable reward mapping that serializes as a JSON object."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, float] | None = None) -> None:
        self._values = MappingProxyType(dict(values or {}))

    def __getitem__(self, key: str) -> float:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        dictionary = core_schema.dict_schema(
            keys_schema=core_schema.str_schema(),
            values_schema=core_schema.float_schema(),
        )
        return core_schema.no_info_after_validator_function(
            cls,
            dictionary,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: dict(value.items()), return_schema=dictionary
            ),
        )


def _absolute_path(value: Path) -> Path:
    if not value.is_absolute():
        raise ValueError("persisted paths must be absolute")
    return value


def _runtime_path(value: Path | str) -> Path:
    return _absolute_path(Path(value))


def _workdir(value: PurePosixPath | str) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if (
        raw in {"", "."}
        or ".." in raw.split("/")
        or not path.is_absolute()
        or (len(path.parts) <= 1 and raw != "/")
    ):
        raise ValueError("workdir must be an absolute POSIX path")
    return path


def _declared_workdir(value: PurePosixPath | str) -> PurePosixPath:
    if str(value) in {"", ".", "/"}:
        return PurePosixPath("/")
    return _workdir(value)


def _shm_size(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"[1-9][0-9]*(?:b|k|kb|m|mb|g|gb)?", normalized):
        raise ValueError(
            "shm_size must be a positive integer with an optional size unit"
        )
    return normalized


class RunStatus(StrEnum):
    PREPARING = "preparing"
    AGENT_RUNNING = "agent_running"
    SNAPSHOTTING = "snapshotting"
    JUDGING = "judging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    NO_VALID_SUBMISSION = "no_valid_submission"


class SubmissionStatus(StrEnum):
    COMPLETED = "completed"
    SUBMISSION_ERROR = "submission_error"
    VERIFIER_ERROR = "verifier_error"
    VERIFIER_TIMEOUT = "verifier_timeout"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class RootfsSnapshotMode(StrEnum):
    FULL_ROOTFS = "full-rootfs"
    SPLIT_WORKDIR = "split-workdir"


class AgentAuthSource(StrEnum):
    """Explicit source selected for Agent credentials."""

    LOCAL = "local"


class CompileOptions(PersistedModel):
    agent_name: str = "codex"
    primary_reward: str | None = None
    score_direction: Literal["maximize", "minimize"] = "maximize"
    agent_timeout_seconds: float | None = None
    max_submissions: int | None = None
    cooldown_seconds: float = 0.0
    disable_stop_hook: bool = False


class RunPaths(PersistedModel):
    root: Path
    workspace: Path
    logs: Path

    _absolute_paths = field_validator("root", "workspace", "logs", mode="after")(
        _absolute_path
    )


class RunRequest(PersistedModel):
    task_dir: Path
    agent_name: str = "codex"
    gpu_selectors: tuple[str, ...] = ()
    options: CompileOptions = Field(default_factory=CompileOptions)
    paths: RunPaths | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    agent_auth: AgentAuthSource | None = None

    _absolute_task_dir = field_validator("task_dir", mode="after")(_absolute_path)


class MainServiceConfig(PersistedModel):
    build_context: Path | None = None
    image: str | None = None
    workdir: PurePosixPath | None = None
    user: str | None = None
    environment: tuple[tuple[str, str], ...] = ()
    shm_size: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    build_timeout_seconds: float = 600.0
    network_mode: Literal["public", "no-network", "allowlist"] = "no-network"
    gpu_requirement: GPURequirement | None = None

    _absolute_build_context = field_validator("build_context", mode="after")(
        lambda value: _absolute_path(value) if value is not None else value
    )
    _valid_workdir = field_validator("workdir", mode="before")(
        lambda value: _declared_workdir(value) if value is not None else value
    )
    _valid_shm_size = field_validator("shm_size", mode="after")(
        lambda value: _shm_size(value) if value is not None else value
    )


class GPURequirement(PersistedModel):
    count: int | Literal["all"] = 0
    name: str | None = None

    @field_validator("count")
    @classmethod
    def require_non_negative_count(cls, value: int | str) -> int | str:
        if isinstance(value, int) and value < 0:
            raise ValueError("GPU count must be non-negative")
        return value


class GPUDevice(PersistedModel):
    index: int
    uuid: str
    name: str


class GPUAllocation(PersistedModel):
    devices: tuple[GPUDevice, ...] = ()

    @model_validator(mode="after")
    def require_unique_uuids(self) -> GPUAllocation:
        uuids = tuple(device.uuid for device in self.devices)
        if len(uuids) != len(set(uuids)):
            raise ValueError("duplicate physical GPU UUID")
        return self

    @property
    def uuids(self) -> tuple[str, ...]:
        return tuple(device.uuid for device in self.devices)


class JudgeGPUMode(StrEnum):
    FREEZE_ONLY = "freeze-only"
    DISJOINT = "disjoint"
    RELEASE_ALL = "release-all"


class RunGPUPlan(PersistedModel):
    authorized_pool: GPUAllocation
    work: GPUAllocation
    judge: GPUAllocation
    judge_mode: JudgeGPUMode

    @model_validator(mode="after")
    def require_consistent_phase_allocations(self) -> RunGPUPlan:
        pool = set(self.authorized_pool.uuids)
        work = set(self.work.uuids)
        judge = set(self.judge.uuids)
        if not work <= pool or not judge <= pool:
            raise ValueError("phase GPU allocation escapes authorized pool")
        if self.judge_mode is JudgeGPUMode.FREEZE_ONLY and judge:
            raise ValueError("freeze-only Judge cannot receive GPUs")
        if self.judge_mode is JudgeGPUMode.DISJOINT and (
            not judge or work & judge
        ):
            raise ValueError("disjoint Judge allocation must be nonempty and disjoint")
        if self.judge_mode is JudgeGPUMode.RELEASE_ALL and (
            not judge or not work & judge
        ):
            raise ValueError("release-all requires an overlapping Judge allocation")
        return self


class ImagePlan(PersistedModel):
    base_ref: str
    work_ref: str
    judge_ref: str
    workdir: PurePosixPath
    rootfs_snapshot_mode: RootfsSnapshotMode
    work_user: str | None = None
    judge_user: str | None = None
    base_digest: str | None = None
    work_digest: str | None = None
    judge_digest: str | None = None

    _valid_workdir = field_validator("workdir", mode="before")(_workdir)


class NetworkPolicy(PersistedModel):
    mode: Literal["public", "no-network", "allowlist"] = "no-network"
    allowlist: tuple[str, ...] = ()

    @field_validator("allowlist", mode="before")
    @classmethod
    def normalize_allowlist(cls, value: object) -> tuple[str, ...]:
        entries = tuple(str(entry).strip().lower() for entry in value or ())
        if not all(entries):
            raise ValueError("network allowlist entries must not be empty")
        return tuple(dict.fromkeys(entries))

    @model_validator(mode="after")
    def require_allowlist_entries(self) -> NetworkPolicy:
        if self.mode == "allowlist" and not self.allowlist:
            raise ValueError("allowlist network policy requires at least one entry")
        return self


class VerifierPlan(PersistedModel):
    command: tuple[str, ...]
    gpu_count: Annotated[int, Field(strict=True, ge=0)] = 0
    timeout_seconds: float = 30.0
    user: str | None = None
    environment: tuple[tuple[str, str], ...] = ()
    secret_env_names: tuple[str, ...] = ()
    primary_reward: str | None = None
    output_limit_bytes: int = 1_000_000
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)

    @field_validator("output_limit_bytes")
    @classmethod
    def require_explicit_truncation_marker(cls, value: int) -> int:
        if value < MIN_VERIFIER_OUTPUT_LIMIT_BYTES:
            raise ValueError(
                "verifier output limit must fit the explicit truncation marker"
            )
        return value


class AgentPlan(PersistedModel):
    name: str
    model: str | None = None
    reasoning_effort: str | None = None
    timeout_seconds: float = 60.0
    user: str | None = None
    environment: tuple[tuple[str, str], ...] = ()
    secret_env_names: tuple[str, ...] = ()
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    install_stop_hook: bool = True


class AssetRequirement(PersistedModel):
    """Portable container-path readiness contract for cluster-staged data."""

    path: PurePosixPath
    phase: Literal["work", "judge"]
    kind: Literal["file", "directory"] = "file"
    min_bytes: int = Field(default=0, ge=0)
    min_entries: int = Field(default=0, ge=0)

    @field_validator("path", mode="before")
    @classmethod
    def require_absolute_non_root_path(
        cls, value: PurePosixPath | str
    ) -> PurePosixPath:
        path = PurePosixPath(value)
        if not path.is_absolute() or path == PurePosixPath("/") or ".." in path.parts:
            raise ValueError("asset path must be an absolute non-root container path")
        return path


class TaskDefinition(PersistedModel):
    task_id: str
    instruction: str = ""
    source_dir: Path
    source_digest: str
    instruction_digest: str
    tests_digest: str
    workdir: PurePosixPath | None = None
    service: MainServiceConfig
    gpu_requirement: GPURequirement
    verifier: VerifierPlan
    agent: AgentPlan
    assets: tuple[AssetRequirement, ...] = ()
    require_disjoint_phase_nodes: bool = False
    environment_digest: str | None = None
    score_direction: Literal["maximize", "minimize"] = "maximize"

    _absolute_source_dir = field_validator("source_dir", mode="after")(_absolute_path)
    _valid_workdir = field_validator("workdir", mode="before")(
        lambda value: _declared_workdir(value) if value is not None else None
    )


class RunPlan(PersistedModel):
    schema_version: int
    task: TaskDefinition
    workdir: PurePosixPath
    rootfs_snapshot_mode: RootfsSnapshotMode
    images: ImagePlan
    gpu_plan: RunGPUPlan
    paths: RunPaths
    snapshot_kind: str

    _valid_workdir = field_validator("workdir", mode="before")(_workdir)


class ContainerMount(PersistedModel):
    source: Path
    target: PurePosixPath
    read_only: bool = False

    _absolute_source = field_validator("source", mode="after")(_absolute_path)

    @field_validator("target", mode="after")
    @classmethod
    def require_absolute_target(cls, value: PurePosixPath) -> PurePosixPath:
        if not value.is_absolute():
            raise ValueError("container mount target must be absolute")
        return value


class ManagedWorkdirVolume(PersistedModel):
    """The exact Docker-owned volume authority for one task workdir."""

    name: str
    driver: Literal["local"] = "local"
    run_id: str
    task_id: str
    target: PurePosixPath
    snapshot_mode: Literal[RootfsSnapshotMode.SPLIT_WORKDIR]
    freshness_nonce: str

    @field_validator("freshness_nonce")
    @classmethod
    def valid_freshness_nonce(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(
                "managed WORKDIR volume freshness nonce must be "
                "64 lowercase hex characters"
            )
        return value

    @model_validator(mode="after")
    def valid_identity(self) -> ManagedWorkdirVolume:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError("unsafe managed WORKDIR volume name")
        if self.target == PurePosixPath("/"):
            raise ValueError("managed WORKDIR volume requires a non-root target")
        _workdir(self.target)
        return self


class ContainerVolumeMount(PersistedModel):
    """A role-specific mount of the managed WORKDIR volume authority."""

    volume: ManagedWorkdirVolume
    target: PurePosixPath
    read_only: bool

    @model_validator(mode="after")
    def exact_target(self) -> ContainerVolumeMount:
        if self.target != self.volume.target:
            raise ValueError("volume mount target differs from managed authority")
        return self


@dataclass(frozen=True, slots=True)
class ContainerTmpfs:
    target: PurePosixPath
    options: str


@dataclass(frozen=True, slots=True)
class ContainerSpec:
    """Runtime container input; it may carry resolved environment values."""

    image: str
    command: tuple[str, ...] = ()
    workdir: PurePosixPath | None = None
    user: str | None = None
    environment: tuple[tuple[str, str], ...] = ()
    mounts: tuple[ContainerMount, ...] = ()
    volume_mounts: tuple[ContainerVolumeMount, ...] = ()
    tmpfs: tuple[ContainerTmpfs, ...] = ()
    gpu_allocation: GPUAllocation = GPUAllocation()
    labels: tuple[tuple[str, str], ...] = ()
    shm_size: str | None = None
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None

    def __post_init__(self) -> None:
        if self.workdir is not None:
            object.__setattr__(self, "workdir", _workdir(self.workdir))
        if self.shm_size is not None:
            object.__setattr__(self, "shm_size", _shm_size(self.shm_size))


class ContainerRef(PersistedModel):
    container_id: str
    role: Literal["work", "judge", "helper"]


class WorkQuiescence(StrEnum):
    """Authoritative non-running state of a Work container."""

    PAUSED = "paused"
    STOPPED = "stopped"


class RootfsSnapshotLease(PersistedModel):
    """Attested Docker image authority for one immutable Work rootfs state."""

    lease_id: str
    purpose: Literal["judge-round", "retained-work"]
    run_id: str
    task_id: str
    round_id: str
    source_container_id: str
    image_id: str
    image_ref: str


@dataclass(frozen=True, slots=True)
class ManagedNetwork:
    """Docker network identity created and labeled by the Engine."""

    network_id: str
    name: str
    run_id: str
    task_id: str
    role: Literal["work", "judge", "helper"]
    internal: bool


class SnapshotCapabilities(PersistedModel):
    backend: str
    atomic: bool
    immutable: bool
    copy_on_write: bool
    cleanup: bool
    reason: str | None = None


class SnapshotLease(PersistedModel):
    lease_id: str
    lower_dir: Path
    upper_dir: Path
    work_dir: Path
    merged_dir: Path
    backend: str
    process_id: int | None = None

    _absolute_paths = field_validator(
        "lower_dir", "upper_dir", "work_dir", "merged_dir", mode="after"
    )(_absolute_path)


@dataclass(frozen=True, slots=True)
class PreparedAgent:
    """Runtime agent command and resolved environment values."""

    agent_name: str
    command: tuple[str, ...]
    prompt_path: Path
    environment: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_path", _runtime_path(self.prompt_path))


@dataclass(frozen=True, slots=True)
class AgentPrepareRequest:
    run_plan: RunPlan
    prompt_path: Path
    max_submissions: int | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_path", _runtime_path(self.prompt_path))


@dataclass(frozen=True, slots=True)
class AgentHookRequest:
    run_plan: RunPlan
    container: ContainerRef
    submit_url: str
    token: str


@dataclass(frozen=True, slots=True)
class AgentRunRequest:
    prepared: PreparedAgent
    container: ContainerRef
    timeout_seconds: float | None = None
    output_path: Path | None = None
    output_redact_values: tuple[str, ...] = ()
    output_callback: Callable[[str], None] | None = None

    def __post_init__(self) -> None:
        if self.output_path is not None:
            object.__setattr__(self, "output_path", _runtime_path(self.output_path))
        object.__setattr__(
            self,
            "output_redact_values",
            tuple(str(value) for value in self.output_redact_values if value),
        )


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    exit_code: int | None
    output: str = ""
    output_truncated: bool = False
    full_output_captured: bool = False
    timed_out: bool = False
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationRequest:
    run_plan: RunPlan
    work_container: ContainerRef
    round_id: str
    verifier_logs: Path
    verifier_output: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "verifier_logs", _runtime_path(self.verifier_logs))
        object.__setattr__(
            self, "verifier_output", _runtime_path(self.verifier_output)
        )


class RewardResult(PersistedModel):
    rewards: FrozenRewardMap = Field(default_factory=FrozenRewardMap)
    score: float | None = None
    error: str | None = None


class SubmissionReport(PersistedModel):
    round_id: str
    status: SubmissionStatus
    rewards: FrozenRewardMap = Field(default_factory=FrozenRewardMap)
    score: float | None = None
    output: str = ""
    exit_code: int | None = None
    timed_out: bool = False
    verifier_output_required: bool = False
    full_output_captured: bool = False
    duration_seconds: float | None = None
    error: str | None = None


class RunResult(PersistedModel):
    run_id: str
    status: RunStatus
    total_rounds: int = 0
    best_score: float | None = None
    best_round: str | None = None
    best_rewards: FrozenRewardMap = Field(default_factory=FrozenRewardMap)
    reports: tuple[SubmissionReport, ...] = ()
