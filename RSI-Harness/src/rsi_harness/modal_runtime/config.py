"""Operator-owned configuration; never read from candidate artifacts."""

from __future__ import annotations

import math
import re
from pathlib import Path, PurePosixPath

from pydantic import Field, model_validator

from rsi_harness.errors import SetupError
from rsi_harness.models import PersistedModel, TaskDefinition


class AssetBind(PersistedModel):
    source: Path
    target: PurePosixPath
    phase: str

    @model_validator(mode="after")
    def validate_bind(self):
        if self.phase not in {"work", "judge"}:
            raise ValueError("asset phase must be work or judge")
        if not self.source.is_absolute() or not self.target.is_absolute():
            raise ValueError("asset paths must be absolute")
        if ".." in self.target.parts or self.target == PurePosixPath("/"):
            raise ValueError("invalid asset target")
        return self


class ModalRunConfig(PersistedModel):
    app_name: str = "openrsi-native"
    environment_name: str
    profile: str
    gpu: str | None = "H100"
    cpus: float = Field(default=16, gt=0)
    memory_mb: int = Field(default=131072, gt=0)
    disk_mb: int | None = Field(default=None, ge=524288, le=3145728)
    segment_seconds: int = Field(default=23 * 3600, ge=30, le=23 * 3600)
    cleanup_seconds: int = Field(default=120, ge=30, le=600)
    volume_name: str
    max_segments: int = Field(default=3, ge=1, le=20)
    max_compute_seconds: int = Field(ge=60, le=7 * 86400)
    assets: tuple[AssetBind, ...] = ()
    provider_urls: tuple[str, ...] = ()
    agent_secret_name: str | None = None
    asset_volumes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self):
        for value in (self.app_name, self.environment_name, self.volume_name):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
                raise ValueError("invalid Modal resource name")
        if self.gpu is not None and (":" in self.gpu or "," in self.gpu):
            raise ValueError("this runner reserves at most one GPU")
        if self.segment_seconds + self.cleanup_seconds > 86400:
            raise ValueError("segment plus cleanup exceeds Modal's limit")
        return self


def validate_task(task: TaskDefinition, config: ModalRunConfig) -> None:
    """Reject unsupported contracts before creating a paid allocation."""
    for duration in (task.agent.timeout_seconds, task.verifier.timeout_seconds):
        if not math.isfinite(duration) or duration <= 0:
            raise SetupError("all task time limits must be finite and positive")
    if (
        min(task.agent.timeout_seconds, config.segment_seconds) + config.cleanup_seconds
        > config.max_compute_seconds
    ):
        raise SetupError("compute ceiling is too small for the first segment")
    if task.service.cpus and task.service.cpus > config.cpus:
        raise SetupError("allocation has fewer CPUs than the task requires")
    if task.service.memory_mb and task.service.memory_mb > config.memory_mb:
        raise SetupError("allocation has less memory than the task requires")
    if task.workdir != PurePosixPath("/workspace"):
        raise SetupError("Modal native currently requires WORKDIR /workspace")
    if not task.verifier.clean_image or task.verifier.visibility != "hidden":
        raise SetupError("Modal native requires a clean, hidden Judge")
    if task.require_disjoint_phase_nodes or task.sandbox is not None:
        raise SetupError("Modal native supports one node without child sandboxes")
    if task.agent.name not in {"codex", "oracle"}:
        raise SetupError("Modal native currently supports Codex and Oracle")
    if task.verifier.model_access or task.verifier.judge_model:
        raise SetupError("Judge provider calls are unsupported; use local inference")
    if any(n.mode != "no-network" for n in (task.agent.network, task.verifier.network)):
        raise SetupError("Modal native requires no-network Work and Judge")
    if task.gpu_requirement.count != 0 or task.verifier.gpu_count not in {0, 1}:
        raise SetupError("Work uses zero GPUs; the local-model Judge uses at most one")
    if task.verifier.gpu_count and config.gpu is None:
        raise SetupError("task requires a GPU allocation")
    if task.service.user not in {None, "root", "0"}:
        raise SetupError("task image must permit the trusted root controller")
    if task.agent.user is not None or task.verifier.user not in {None, "root", "0"}:
        raise SetupError("phase identities are assigned by the Modal runner")
    if task.agent.name == "codex" and not config.provider_urls:
        raise SetupError("Codex needs explicitly pinned provider URLs")
    if task.agent.name == "codex" and not config.agent_secret_name:
        raise SetupError("Codex needs an explicitly named Modal API-key Secret")
    if task.agent.name == "oracle" and config.provider_urls:
        raise SetupError("Oracle must not receive provider access")
    for mount, name in config.asset_volumes.items():
        path = PurePosixPath(mount)
        if (
            not path.is_relative_to("/asset-store")
            or path == PurePosixPath("/asset-store")
            or ".." in path.parts
        ):
            raise SetupError("asset volumes must mount below /asset-store")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise SetupError("invalid asset volume name")
    forbidden = [
        PurePosixPath(p)
        for p in (
            "/usr",
            "/bin",
            "/lib",
            "/lib64",
            "/etc",
            "/proc",
            "/dev",
            "/sys",
            "/tmp",
            "/run",
            "/home",
            "/root",
            "/tests",
            "/logs",
            "/workspace",
            "/opt/rsi-isolate",
        )
    ]
    for bind in config.assets:
        if ".." in bind.source.parts or not any(
            bind.source.is_relative_to(Path(mount)) for mount in config.asset_volumes
        ):
            raise SetupError("asset sources must be inside an explicit asset volume")
        if any(
            bind.target == p or bind.target in p.parents or p in bind.target.parents
            for p in forbidden
        ):
            raise SetupError(f"asset overlaps a runtime-owned target: {bind.target}")
