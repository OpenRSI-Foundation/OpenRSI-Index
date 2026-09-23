"""Shared resource and result contracts for batch schedulers.

GPU and memory requests are per host for multi-host jobs. Walltime is HH:MM.
Queue/group map to the scheduler's partition/account on Slurm.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator

from rsi_harness.models import PersistedModel

_SAFE_JOB_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")


class JobSpec(PersistedModel):
    name: str
    queue: str
    group: str
    cpu_slots: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    walltime: str
    stdout_path: Path
    stderr_path: Path
    script_path: Path
    gpu_count: int = Field(default=0, ge=0)
    local_tmp_mb: int = Field(default=0, ge=0)
    one_host: bool = True
    hosts: int = Field(default=1, gt=0)
    slots_per_host: int | None = Field(default=None, gt=0)
    memory_per_host: bool = False
    exclusive: bool = False
    excluded_hosts: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def _safe_name(cls, value: str) -> str:
        if not _SAFE_JOB_NAME.fullmatch(value):
            raise ValueError("scheduler job name contains unsupported characters")
        return value

    @field_validator("stdout_path", "stderr_path", "script_path")
    @classmethod
    def _absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("scheduler paths must be absolute")
        return value

    @field_validator("excluded_hosts")
    @classmethod
    def _safe_excluded_hosts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("excluded_hosts contains a duplicate")
        if any(_SAFE_HOST.fullmatch(host) is None for host in value):
            raise ValueError("excluded_hosts contains an unsafe host")
        return value

    @model_validator(mode="after")
    def _host_shape(self) -> Self:
        if self.one_host:
            if self.hosts != 1 or self.slots_per_host is not None:
                raise ValueError(
                    "one-host scheduler specs require hosts=1 without slots_per_host"
                )
            if self.memory_per_host:
                raise ValueError("per-host memory requires a multi-host scheduler spec")
            return self
        if self.hosts <= 1 or self.slots_per_host is None:
            raise ValueError(
                "multi-host scheduler specs require hosts>1 and slots_per_host"
            )
        if self.cpu_slots != self.hosts * self.slots_per_host:
            raise ValueError("multi-host cpu_slots must equal hosts * slots_per_host")
        return self


class JobResult(PersistedModel):
    job_id: str
    state: str
    exit_code: int
