"""Small contracts shared by cluster adapters and the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import Field

from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    JudgeLLMRequest,
    PersistedModel,
    RunStatus,
)
from rsi_harness.runtime.sandbox_contracts import SandboxPolicy

if TYPE_CHECKING:
    from rsi_harness.cluster.bluevela.adapter import ClusterResumePlan
    from rsi_harness.runtime.recovery import AttemptGrants


class ClusterRunRequest(PersistedModel):
    task_dir: Path
    agent_name: str = "codex"
    options: CompileOptions = Field(default_factory=CompileOptions)
    logs_root: Path
    model: str | None = None
    reasoning_effort: str | None = None
    agent_auth: AgentAuthSource | None = None
    # Where a judge_model task's credential comes from; never the credential.
    judge_llm: JudgeLLMRequest | None = None
    dry_run: bool = False
    # Only an E2B env policy ([environments.host] backend = "e2b").
    sandbox_policy: SandboxPolicy | None = None


class ClusterRunResult(PersistedModel):
    run_id: str
    status: RunStatus
    log_dir: Path
    job_ids: tuple[str, ...] = ()


class ClusterAdapter(Protocol):
    def run(self, request: ClusterRunRequest) -> ClusterRunResult: ...

    def plan_resume(
        self, run_id: str, grants: AttemptGrants, *, dry_run: bool = False
    ) -> ClusterResumePlan: ...

    def resume(self, plan: ClusterResumePlan) -> ClusterRunResult: ...
