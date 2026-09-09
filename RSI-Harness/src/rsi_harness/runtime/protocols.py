"""Narrow runtime ports consumed by the coordinator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    ContainerSpec,
    EvaluationRequest,
    PreparedAgent,
    SnapshotCapabilities,
    SnapshotLease,
    SubmissionReport,
)


class ContainerRuntime(Protocol):
    def create(self, spec: ContainerSpec) -> ContainerRef: ...

    def start(self, container: ContainerRef) -> None: ...

    def pause(self, container: ContainerRef) -> None: ...

    def unpause(self, container: ContainerRef) -> None: ...

    def remove(self, container: ContainerRef) -> None: ...


class SnapshotBackend(Protocol):
    def probe(self, workspace_root: Path) -> SnapshotCapabilities: ...

    def acquire(
        self, workspace: Path, *, run_id: str, round_id: str
    ) -> SnapshotLease: ...

    def release(self, lease: SnapshotLease) -> None: ...


class AgentAdapter(Protocol):
    def available_agents(self) -> tuple[str, ...]: ...

    def prepare(self, request: AgentPrepareRequest) -> PreparedAgent: ...

    def install_hooks(self, request: AgentHookRequest) -> None: ...

    def run(self, request: AgentRunRequest) -> AgentRunResult: ...


class Evaluator(Protocol):
    def evaluate(self, request: EvaluationRequest) -> SubmissionReport: ...


class ArtifactSink(Protocol):
    def start(self) -> None: ...

    def record_submission(self, report: SubmissionReport) -> None: ...

    def finalize(self) -> None: ...


class Clock(Protocol):
    def now(self) -> datetime: ...
