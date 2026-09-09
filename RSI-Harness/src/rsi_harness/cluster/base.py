"""Small contracts shared by cluster adapters and the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import Field

from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    PersistedModel,
    RunStatus,
)


class ClusterRunRequest(PersistedModel):
    task_dir: Path
    agent_name: str = "codex"
    options: CompileOptions = Field(default_factory=CompileOptions)
    logs_root: Path
    model: str | None = None
    reasoning_effort: str | None = None
    agent_auth: AgentAuthSource | None = None
    dry_run: bool = False


class ClusterRunResult(PersistedModel):
    run_id: str
    status: RunStatus
    log_dir: Path
    job_ids: tuple[str, ...] = ()


class ClusterAdapter(Protocol):
    def run(self, request: ClusterRunRequest) -> ClusterRunResult: ...
