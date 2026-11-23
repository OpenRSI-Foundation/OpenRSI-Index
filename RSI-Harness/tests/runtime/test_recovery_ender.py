"""A round that requires recovery keeps Work paused: the Agent is ended at
once instead of waiting out its time."""

from __future__ import annotations

import threading
from pathlib import Path

from rsi_harness.models import (
    AgentRunResult,
    RunRequest,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.coordinator import RunCoordinator
from rsi_harness.runtime.recovery import LeaseStore
from tests.runtime.test_coordinator import Clock, ScriptedBackend

RECOVERY = SubmissionReport(
    round_id="agent-1",
    status=SubmissionStatus.INFRASTRUCTURE_ERROR,
    error="recovery_required: Judge removal failed",
)


class PausedAgentBackend(ScriptedBackend):
    """The Agent submits once, then runs on only while Work is not held:
    a held Work keeps it paused until its exec is ended."""

    def __init__(self, tmp_path: Path, *, fail_end: bool = False) -> None:
        super().__init__(tmp_path)
        self.fail_end = fail_end
        self.ended = threading.Event()

    def run_agent(self, prepared, work, timeout):
        del prepared, timeout
        self.events.append(("agent_start", work.container_id))
        assert self.service is not None
        self.service.submit()
        if self.ended.wait(0.5 if self.fail_end else 5):
            self.events.append(("agent_end", work.container_id))
            return AgentRunResult(exit_code=137)
        self.events.append(("agent_timeout", work.container_id))
        return AgentRunResult(exit_code=None, timed_out=True)

    def end_agent(self, work):
        self.events.append(("agent_end_requested", work.container_id))
        if self.fail_end:
            raise RuntimeError("Work kill failed")
        self.ended.set()


def _run(backend: ScriptedBackend, tmp_path: Path):
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
    ).run(RunRequest(task_dir=tmp_path))
    return result, store


def _names(backend: ScriptedBackend) -> list[str]:
    return [name for name, _ in backend.events]


def test_a_round_that_requires_recovery_ends_the_agent_at_once(tmp_path) -> None:
    backend = PausedAgentBackend(tmp_path)
    backend.reports[0] = RECOVERY

    result, store = _run(backend, tmp_path)

    assert result.status is RunStatus.FAILED
    assert result.total_rounds == 1
    assert result.agent_exit_code == 137
    assert backend.artifacts.finalized["timed_out"] is False
    names = _names(backend)
    assert names.index("judge_create") < names.index("agent_end_requested")
    assert names.index("agent_end_requested") < names.index("agent_end")
    assert ("agent_ended_after_recovery_required", None) in backend.events
    # Everything else is kept for recover, as before.
    assert "work_remove" not in names and "network_remove" not in names
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.judge.container_id == "judge:agent-1"


def test_a_completed_round_leaves_the_agent_running(tmp_path) -> None:
    backend = PausedAgentBackend(tmp_path, fail_end=True)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    names = _names(backend)
    assert "agent_end_requested" not in names
    assert "agent_timeout" in names


def test_an_agent_that_cannot_be_ended_runs_out_its_time(tmp_path) -> None:
    backend = PausedAgentBackend(tmp_path, fail_end=True)
    backend.reports[0] = RECOVERY

    result, store = _run(backend, tmp_path)

    assert result.status is RunStatus.FAILED
    assert backend.artifacts.finalized["timed_out"] is True
    assert "agent_timeout" in _names(backend)
    failed = [
        value
        for name, value in backend.events
        if name == "agent_end_after_recovery_required_failed"
    ]
    assert len(failed) == 1 and "Work kill failed" in str(failed[0])
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
