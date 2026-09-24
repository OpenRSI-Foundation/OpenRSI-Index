"""Strict, portable cluster profile loading."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Self

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from rsi_harness.errors import SetupError

_ENV_REFERENCE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_SCHEDULER_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")


class _ProfileModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SchedulerProfile(_ProfileModel):
    kind: Literal["lsf"]
    submit_binary: str = "bsub"
    status_binary: str = "bjobs"
    cancel_binary: str = "bkill"
    remote_binary: str = "blaunch"
    remote_host_flag: str = "-z"
    remote_args: tuple[str, ...] = ()
    queue: str
    group: str
    exclusive: bool = True
    excluded_hosts: tuple[str, ...] = ()
    poll_seconds: float = Field(default=10.0, gt=0.0)

    @field_validator(
        "submit_binary",
        "status_binary",
        "cancel_binary",
        "remote_binary",
        "remote_host_flag",
    )
    @classmethod
    def _nonempty_binary_argument(cls, value: str) -> str:
        if not value or "\0" in value:
            raise ValueError("scheduler command values must be non-empty")
        return value

    @field_validator("remote_args")
    @classmethod
    def _safe_remote_args(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or "\0" in item for item in value):
            raise ValueError("scheduler remote_args must be non-empty arguments")
        return value

    @field_validator("excluded_hosts")
    @classmethod
    def _safe_excluded_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("scheduler excluded_hosts contains a duplicate")
        if any(_SCHEDULER_HOST.fullmatch(host) is None for host in value):
            raise ValueError("scheduler excluded_hosts contains an unsafe host")
        return value


class SlurmSchedulerProfile(SchedulerProfile):
    kind: Literal["slurm"]
    submit_binary: str = "sbatch"
    status_binary: str = "squeue"
    cancel_binary: str = "scancel"
    accounting_binary: str = "sacct"
    control_binary: str = "scontrol"
    remote_binary: str = "srun"
    remote_host_flag: str = "--nodelist"
    # Controllers and cleanup steps must coexist with their rank workers.
    remote_args: tuple[str, ...] = (
        "--overlap", "--exact", "--nodes=1", "--ntasks=1", "--cpu-bind=none",
        "--export=ALL",
    )
    queue: str = Field(validation_alias=AliasChoices("partition", "queue"))
    group: str = Field(
        default="", validation_alias=AliasChoices("account", "group")
    )
    qos: str | None = None
    constraint: str | None = None

    @field_validator("accounting_binary", "control_binary")
    @classmethod
    def _nonempty_slurm_binary(cls, value: str) -> str:
        return cls._nonempty_binary_argument(value)


class StorageProfile(_ProfileModel):
    run_root: Path
    image_cache: Path
    logs_root: Path
    hf_home: Path
    hf_datasets_cache: Path

    @field_validator(
        "run_root", "image_cache", "logs_root", "hf_home", "hf_datasets_cache"
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("cluster storage paths must be absolute")
        return value


class BuilderProfile(_ProfileModel):
    binary: Path
    cpu_slots: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    walltime: str
    temp_root: Path = Path("/tmp")
    min_tmp_mb: int = Field(gt=0)
    rootless_apt_sandbox: bool = False
    faked_binary: Path | None = None
    fakeroot_library: Path | None = None

    @model_validator(mode="after")
    def _paired_fakeroot_tools(self) -> Self:
        if (self.faked_binary is None) != (self.fakeroot_library is None):
            raise ValueError(
                "builder faked_binary and fakeroot_library must be set together"
            )
        return self


class ApptainerBindProfile(_ProfileModel):
    """One site-owned bind, scoped to a single execution phase."""

    source: Path
    target: PurePosixPath
    read_only: bool = True

    @field_validator("source")
    @classmethod
    def _absolute_source(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("Apptainer bind sources must be absolute")
        return value

    @field_validator("target")
    @classmethod
    def _absolute_target(cls, value: PurePosixPath) -> PurePosixPath:
        if (
            not value.is_absolute()
            or value == PurePosixPath("/")
            or ".." in value.parts
        ):
            raise ValueError(
                "Apptainer bind targets must be absolute non-root paths"
            )
        return value


class ApptainerProfile(_ProfileModel):
    binary: Path
    dns_bind: Path = Path("/etc/resolv.conf")
    # Existing site profiles retain their mount behavior unless they opt in.
    mount_policy: Literal["legacy", "scoped"] = "legacy"
    extra_binds: tuple[Path, ...] = ()
    work_binds: tuple[ApptainerBindProfile, ...] = ()
    judge_binds: tuple[ApptainerBindProfile, ...] = ()
    # Optional site-owned root. A task-specific child, when present, is bound
    # read-only at /run-contract for Judge only.
    judge_authority_root: Path | None = None
    build_args: tuple[str, ...] = ()
    workspace_target: PurePosixPath = PurePosixPath("/testbed")
    temp_root: Path = Path("/tmp")
    container_python: PurePosixPath = PurePosixPath("/usr/bin/python3")
    rdma_binds: tuple[Path, ...] = ()
    environment: dict[str, str] = Field(default_factory=dict)
    work_environment: dict[str, str] = Field(default_factory=dict)
    judge_environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("environment", "work_environment", "judge_environment")
    @classmethod
    def _runtime_environment(cls, value: dict[str, str]) -> dict[str, str]:
        for name, item in value.items():
            if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name) is None:
                raise ValueError(
                    f"Apptainer environment name is invalid: {name!r}"
                )
            if "\0" in item:
                raise ValueError(
                    f"Apptainer environment value contains NUL: {name}"
                )
        return value

    @field_validator("workspace_target", "container_python")
    @classmethod
    def _container_path(cls, value: PurePosixPath) -> PurePosixPath:
        if (
            not value.is_absolute()
            or value == PurePosixPath("/")
            or ".." in value.parts
        ):
            raise ValueError(
                "Apptainer container paths must be absolute non-root paths"
            )
        return value

    @model_validator(mode="after")
    def _scoped_binds_only(self) -> Self:
        if self.mount_policy == "scoped" and self.extra_binds:
            raise ValueError(
                "scoped mount policy forbids extra_binds; use work_binds and "
                "judge_binds with explicit targets and read_only settings"
            )
        return self

    @field_validator("extra_binds", "rdma_binds")
    @classmethod
    def _absolute_host_binds(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if any(not path.is_absolute() for path in value):
            raise ValueError("Apptainer host bind paths must be absolute")
        return value

    @field_validator("judge_authority_root")
    @classmethod
    def _absolute_authority_root(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("Judge authority root must be absolute")
        return value


class ResourceProfile(_ProfileModel):
    min_cpu_slots: int = Field(gt=0)
    min_memory_mb: int = Field(gt=0)
    min_runtime_tmp_mb: int = Field(default=0, ge=0)
    gpus_per_node: int = Field(gt=0)
    walltime_margin_seconds: int = Field(ge=0)
    all_gpus_override: int | None = Field(default=None, gt=0)


class ClusterProfile(_ProfileModel):
    name: str
    adapter: Literal["bluevela", "slurm"]
    owner: str
    scheduler: SchedulerProfile | SlurmSchedulerProfile = Field(discriminator="kind")
    storage: StorageProfile
    builder: BuilderProfile
    apptainer: ApptainerProfile
    resources: ResourceProfile

    @model_validator(mode="after")
    def _matching_scheduler(self) -> Self:
        expected = "lsf" if self.adapter == "bluevela" else "slurm"
        if self.scheduler.kind != expected:
            raise ValueError(f"{self.adapter} adapter requires {expected} scheduler")
        return self


def _expand(value: Any, environ: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in environ:
                raise SetupError(
                    f"cluster profile references missing environment variable {name}"
                )
            return environ[name]

        return _ENV_REFERENCE.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item, environ) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item, environ) for key, item in value.items()}
    return value


def _profile_bytes(name_or_path: str | Path) -> bytes:
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.read_bytes()
    if str(name_or_path) in {"bluevela", "slurm"}:
        return (
            resources.files(f"rsi_harness.cluster.{name_or_path}")
            .joinpath("profile.toml")
            .read_bytes()
        )
    raise SetupError(f"unknown cluster or profile path: {name_or_path}")


def load_cluster_profile(
    name_or_path: str | Path,
    environ: Mapping[str, str] | None = None,
) -> ClusterProfile:
    """Load a packaged cluster name or an explicit TOML profile."""
    try:
        raw = tomllib.loads(_profile_bytes(name_or_path).decode("utf-8"))
        expanded = _expand(raw, os.environ if environ is None else environ)
        return ClusterProfile.model_validate(expanded)
    except SetupError:
        raise
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValidationError) as error:
        raise SetupError(f"invalid cluster profile {name_or_path}: {error}") from error
