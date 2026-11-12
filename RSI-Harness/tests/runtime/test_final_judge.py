"""A run with a submission budget of 1: one final judge per attempt."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentRunResult,
    RunRequest,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
    scored_reports,
)
from rsi_harness.runtime import coordinator as coordinator_module
from rsi_harness.runtime.coordinator import (
    RunCoordinator,
    StartedSubmissionServer,
    aggregate_result,
)
from rsi_harness.runtime.recovery import AttemptGrants, LeaseStore
from rsi_harness.runtime.submissions import JudgeEndpoint, SubmissionService
from tests.runtime.test_coordinator import (
    Clock,
    ResumingBackend,
    ScriptedBackend,
    Server,
    _resume_context,
)


def _final_plan(plan, budget: int | None = 1):
    verifier = plan.task.verifier.model_copy(update={"max_submissions": budget})
    return plan.model_copy(
        update={"task": plan.task.model_copy(update={"verifier": verifier})}
    )


class FinalBackend(ScriptedBackend):
    """A real SubmissionService; the Agent's behaviour is ``agent``."""

    def __init__(self, tmp_path: Path, agent, *, budget: int | None = 1) -> None:
        super().__init__(tmp_path)
        self.plan = _final_plan(self.plan, budget)
        self.agent = agent
        self.ended = threading.Event()
        self.submit_outputs: list[object] = []
        self.evaluated: list[tuple[str, bool]] = []

    def start_server(self, evaluator, artifacts, clock):
        self.events.append(("server_start", None))
        self.service = SubmissionService(
            evaluator=evaluator, artifact_writer=artifacts, clock=clock
        )
        return StartedSubmissionServer(
            service=self.service,
            owner=_DrainingServer(self.events, self.service),
            endpoint=JudgeEndpoint(
                url="http://172.30.0.1:9020", bind_host="127.0.0.1", port=9020
            ),
        )

    def submit(self) -> None:
        assert self.service is not None and self.submit_token is not None
        try:
            self.submit_outputs.append(self.service.submit(self.submit_token))
        except Exception as error:
            self.submit_outputs.append(error)

    def run_agent(self, prepared, work, timeout):
        del prepared, timeout
        self.events.append(("agent_start", work.container_id))
        result = self.agent(self)
        self.events.append(("agent_end", work.container_id))
        return result

    def end_agent(self, work):
        self.events.append(("agent_end_requested", work.container_id))
        self.ended.set()

    def evaluate_submission(self, request, *, lifecycle_observer):
        self.evaluated.append((request.round_id, request.final))
        if self.store is not None:
            lease = self.store.read("run-1")
            assert lease is not None
            self.lease_at_judge = lease
        return super().evaluate_submission(
            request, lifecycle_observer=lifecycle_observer
        )


class _DrainingServer(Server):
    def __init__(self, events, service) -> None:
        super().__init__(events)
        self.service = service

    def stop(self) -> None:
        self.service.close(timeout_seconds=5)
        super().stop()


def _run(backend: FinalBackend, tmp_path: Path, **options):
    store = LeaseStore(tmp_path / "leases")
    backend.store = store
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        **options,
    ).run(RunRequest(task_dir=tmp_path))
    return result, store


def _names(backend: ScriptedBackend) -> list[str]:
    return [name for name, _ in backend.events]


@pytest.mark.parametrize(
    "ending",
    [
        AgentRunResult(exit_code=None, timed_out=True),
        AgentRunResult(exit_code=0),
        AgentRunResult(exit_code=1, provider_error="rate_limited", provider_retries=8),
    ],
    ids=["timeout", "clean-exit", "provider-stop"],
)
def test_an_agent_that_never_submits_is_judged_once_on_stopped_work(
    tmp_path, ending
) -> None:
    backend = FinalBackend(tmp_path, lambda _backend: ending)

    result, store = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert result.total_rounds == 1
    assert result.best_round == "agent-1"
    assert backend.evaluated == [("agent-1", True)]
    names = _names(backend)
    # Work stops before the round, and the Agent's time is closed then.
    assert names.index("agent_end") < names.index("agent_stop")
    assert names.index("agent_stop") < names.index("judge_create")
    assert names.count("agent_stop") == 1
    assert names.index("agent_stop") < names.index("final_judge_started")
    assert "agent_end_requested" not in names
    (attempt,) = backend.lease_at_judge.attempts
    assert attempt.ended_at is not None
    lease = store.read("run-1")
    assert lease is not None and lease.status is RunStatus.COMPLETED
    assert backend.artifacts.finalized["timed_out"] is ending.timed_out


def test_the_final_round_runs_outside_the_agent_budget(tmp_path) -> None:
    class SlowJudge(FinalBackend):
        def evaluate_submission(self, request, *, lifecycle_observer):
            time.sleep(0.5)
            return super().evaluate_submission(
                request, lifecycle_observer=lifecycle_observer
            )

    backend = SlowJudge(
        tmp_path, lambda _backend: AgentRunResult(exit_code=None, timed_out=True)
    )

    result, store = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert backend.artifacts.finalized["agent_runtime_seconds"] < 0.4
    assert backend.artifacts.finalized["runtime_seconds"] >= 0.5
    (attempt,) = store.read("run-1").attempts
    assert attempt.agent_seconds is not None and attempt.agent_seconds < 0.4


def test_an_agent_that_crashes_early_is_not_judged(tmp_path) -> None:
    backend = FinalBackend(tmp_path, lambda _backend: AgentRunResult(exit_code=3))

    result, store = _run(backend, tmp_path)

    assert result.status is RunStatus.FAILED
    assert result.total_rounds == 0
    assert backend.evaluated == []
    assert any(
        "exited with code 3 before its time ended" in error
        for error in backend.artifacts.engine_errors
    )
    lease = store.read("run-1")
    assert lease is not None and lease.status is RunStatus.FAILED
    # Closed, so `resume` can continue it with the time left.
    assert all(entry.ended_at is not None for entry in lease.attempts)


def test_a_cancelled_agent_is_not_judged(tmp_path) -> None:
    backend = FinalBackend(
        tmp_path, lambda _backend: AgentRunResult(exit_code=None, cancelled=True)
    )

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.CANCELLED
    assert backend.evaluated == []


def test_the_agents_one_rsi_submit_is_the_round_and_ends_the_agent(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(coordinator_module, "FINAL_FEEDBACK_GRACE_SECONDS", 0.2)

    def agent(backend: FinalBackend) -> AgentRunResult:
        backend.submit()
        # It keeps working, and calls again, until the Harness ends it.
        backend.submit()
        assert backend.ended.wait(5)
        return AgentRunResult(exit_code=137)

    backend = FinalBackend(tmp_path, agent)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert result.total_rounds == 1
    assert backend.evaluated == [("agent-1", True)]
    first, second = backend.submit_outputs
    assert isinstance(first, str)
    assert first.endswith("This was the run's one final judge; the run ends now.\n")
    assert "remaining" not in first and "Full verifier output" not in first
    assert "can be called once in this run" in str(second)
    names = _names(backend)
    assert names.index("judge_create") < names.index("agent_end_requested")
    assert names.index("agent_end_requested") < names.index("agent_end")
    assert ("agent_ended_after_final_judge", None) in backend.events


def test_an_agent_that_exits_after_its_submit_is_not_judged_again(tmp_path) -> None:
    def agent(backend: FinalBackend) -> AgentRunResult:
        backend.submit()
        return AgentRunResult(exit_code=0)

    backend = FinalBackend(tmp_path, agent)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert backend.evaluated == [("agent-1", True)]
    assert "agent_end_requested" not in _names(backend)


def test_a_round_in_flight_at_the_timeout_is_the_attempts_round(tmp_path) -> None:
    """The Agent's own late rsi-submit is drained and counted; no second round."""
    release = threading.Event()

    class SlowJudge(FinalBackend):
        def evaluate_submission(self, request, *, lifecycle_observer):
            release.wait(5)
            return super().evaluate_submission(
                request, lifecycle_observer=lifecycle_observer
            )

    def agent(backend: FinalBackend) -> AgentRunResult:
        threading.Thread(target=backend.submit, daemon=True).start()
        deadline = time.monotonic() + 5
        while not backend.service.attempt_has_round(0):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        threading.Timer(0.3, release.set).start()
        return AgentRunResult(exit_code=None, timed_out=True)

    backend = SlowJudge(tmp_path, agent)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert result.total_rounds == 1
    assert backend.evaluated == [("agent-1", True)]


def test_a_round_in_flight_at_the_timeout_that_is_given_back_is_judged_again(
    tmp_path,
) -> None:
    """The Agent's late rsi-submit meets a busy Work and is refunded: the
    Harness waits for it, then judges the attempt itself."""
    from rsi_harness.errors import RetryableSubmissionError

    release = threading.Event()

    class BusyJudge(FinalBackend):
        def evaluate_submission(self, request, *, lifecycle_observer):
            if not self.evaluated:
                self.evaluated.append((request.round_id, request.final))
                release.wait(5)
                raise RetryableSubmissionError("release all Work GPU processes")
            return super().evaluate_submission(
                request, lifecycle_observer=lifecycle_observer
            )

    def agent(backend: FinalBackend) -> AgentRunResult:
        threading.Thread(target=backend.submit, daemon=True).start()
        deadline = time.monotonic() + 5
        while not backend.service.attempt_has_round(0):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        threading.Timer(0.3, release.set).start()
        return AgentRunResult(exit_code=None, timed_out=True)

    backend = BusyJudge(tmp_path, agent)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert result.total_rounds == 1
    assert backend.evaluated == [("agent-1", True), ("agent-1", True)]
    assert ("final_judge_started", None) in backend.events
    assert backend.artifacts.final_rounds == [("agent-1", "harness")]


def test_an_agent_that_fails_after_its_submit_keeps_its_round(tmp_path) -> None:
    """Its exit code after the one round does not matter: no early end."""

    def agent(backend: FinalBackend) -> AgentRunResult:
        backend.submit()
        return AgentRunResult(exit_code=1)

    backend = FinalBackend(tmp_path, agent)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert result.total_rounds == 1
    assert backend.evaluated == [("agent-1", True)]
    assert backend.artifacts.engine_errors == []
    assert backend.artifacts.final_rounds == [("agent-1", "agent")]
    assert backend.artifacts.finalized["first_submission_seconds"] is not None


def test_the_agent_is_ended_only_after_its_feedback_grace(
    tmp_path, monkeypatch
) -> None:
    # Long enough for rsi-submit to print the result into the session.
    assert coordinator_module.FINAL_FEEDBACK_GRACE_SECONDS == 10.0
    grace = 0.4
    monkeypatch.setattr(coordinator_module, "FINAL_FEEDBACK_GRACE_SECONDS", grace)
    returned: list[float] = []

    def agent(backend: FinalBackend) -> AgentRunResult:
        backend.submit()
        returned.append(time.monotonic())
        assert backend.ended.wait(5)
        return AgentRunResult(exit_code=137)

    class Timed(FinalBackend):
        def end_agent(self, work):
            self.ended_at = time.monotonic()
            super().end_agent(work)

    backend = Timed(tmp_path, agent)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert backend.ended_at - returned[0] >= grace - 0.05


def test_the_harnesss_round_is_not_a_first_submission(tmp_path) -> None:
    backend = FinalBackend(tmp_path, lambda _backend: AgentRunResult(exit_code=0))

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert backend.artifacts.finalized["first_submission_seconds"] is None
    assert backend.artifacts.final_rounds == [("agent-1", "harness")]
    assert ("final_judge_planned", None) in backend.events


def test_an_early_crash_says_why_and_that_it_can_be_resumed(tmp_path) -> None:
    backend = FinalBackend(tmp_path, lambda _backend: AgentRunResult(exit_code=137))

    _run(backend, tmp_path)

    (message,) = [
        value for name, value in backend.events if name == "agent_ended_early"
    ]
    assert "exited with code 137" in message and "resume the run" in message


def test_other_budgets_announce_no_final_judge(tmp_path) -> None:
    backend = FinalBackend(
        tmp_path, lambda _backend: AgentRunResult(exit_code=0), budget=2
    )

    _run(backend, tmp_path)

    assert "final_judge_planned" not in _names(backend)
    assert backend.artifacts.final_rounds == []


@pytest.mark.parametrize(
    "status", [SubmissionStatus.VERIFIER_ERROR, SubmissionStatus.VERIFIER_TIMEOUT]
)
def test_a_final_round_without_a_reward_is_no_valid_submission(
    tmp_path, status
) -> None:
    backend = FinalBackend(tmp_path, lambda _backend: AgentRunResult(exit_code=0))
    backend.reports[0] = backend.reports[0].model_copy(
        update={"status": status, "score": None, "rewards": {}, "error": "no reward"}
    )

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.NO_VALID_SUBMISSION
    assert result.best_score is None


def test_a_busy_final_round_is_tried_again(tmp_path, monkeypatch) -> None:
    from rsi_harness.errors import RetryableSubmissionError

    monkeypatch.setattr(coordinator_module, "FINAL_ROUND_RETRY_SECONDS", 0.01)

    class BusyOnce(FinalBackend):
        busy = True

        def evaluate_submission(self, request, *, lifecycle_observer):
            if self.busy:
                self.busy = False
                raise RetryableSubmissionError("active remote Work must finish")
            return super().evaluate_submission(
                request, lifecycle_observer=lifecycle_observer
            )

    backend = BusyOnce(tmp_path, lambda _backend: AgentRunResult(exit_code=0))

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert backend.evaluated == [("agent-1", True)]


def test_other_budgets_keep_every_round_and_the_best_score(tmp_path) -> None:
    """A budget of 2: no Harness round, no end, best-of as before."""

    def agent(backend: FinalBackend) -> AgentRunResult:
        backend.submit()
        backend.submit()
        backend.submit()
        return AgentRunResult(exit_code=None, timed_out=True)

    backend = FinalBackend(tmp_path, agent, budget=2)

    result, _ = _run(backend, tmp_path)

    assert result.status is RunStatus.COMPLETED
    assert backend.evaluated == [("agent-1", False), ("agent-2", False)]
    assert result.best_round == "agent-2"
    assert "budget is exhausted" in str(backend.submit_outputs[2])
    assert "remaining: 0" in backend.submit_outputs[1]
    assert backend.service.final_round_done.is_set() is False
    assert "agent_end_requested" not in _names(backend)


def test_a_task_budget_reaches_the_service_and_the_agent(tmp_path) -> None:
    backend = FinalBackend(
        tmp_path, lambda _backend: AgentRunResult(exit_code=0), budget=3
    )

    _run(backend, tmp_path)

    assert backend.prepared_max_submissions == 3


def _resumed(tmp_path: Path, backend: FinalBackend, **updates):
    store = LeaseStore(tmp_path / "leases")
    first = FinalBackend(tmp_path, lambda _backend: AgentRunResult(exit_code=0))
    first.store = store
    assert (
        RunCoordinator(
            backend=first,
            lease_store=store,
            run_id_factory=lambda: "run-1",
            clock=Clock(),
        )
        .run(RunRequest(task_dir=tmp_path))
        .status
        is RunStatus.COMPLETED
    )
    context = _resume_context(
        tmp_path,
        store,
        backend,
        reports=(
            SubmissionReport(
                round_id="agent-1",
                status=SubmissionStatus.COMPLETED,
                rewards={"reward": 2.0},
                score=2.0,
            ),
        ),
        charged_rounds=1,
        max_submissions=2,
        grants=AttemptGrants(add_time_seconds=5.0),
        **updates,
    )
    return store, context


class ResumedFinalBackend(FinalBackend, ResumingBackend):
    def run_agent(self, prepared, work, timeout):
        self.agent_timeout = timeout
        return FinalBackend.run_agent(self, prepared, work, timeout)


def test_a_resumed_attempt_ends_with_its_own_round_which_is_the_score(
    tmp_path,
) -> None:
    backend = ResumedFinalBackend(
        tmp_path, lambda _backend: AgentRunResult(exit_code=None, timed_out=True)
    )
    store, context = _resumed(tmp_path, backend)

    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        resume=context,
    ).run(RunRequest(task_dir=tmp_path))

    # agent-1 scored 2.0; the newest round, agent-2, scores 0.25 and wins.
    assert backend.evaluated == [("agent-2", True)]
    assert result.status is RunStatus.COMPLETED
    assert result.best_round == "agent-2"
    assert result.best_score == 0.25
    assert result.total_rounds == 2
    assert backend.service.final_round_done.is_set()


def test_a_resumed_attempt_keeps_the_runs_own_budget(tmp_path) -> None:
    """The saved plan decides; a recompiled plan that differs is refused."""
    backend = ResumedFinalBackend(
        tmp_path, lambda _backend: AgentRunResult(exit_code=0)
    )
    store, context = _resumed(tmp_path, backend)
    context = replace(context, plan=_final_plan(context.plan, None))

    with pytest.raises(SetupError, match="submission budget differs"):
        RunCoordinator(
            backend=backend,
            lease_store=store,
            run_id_factory=lambda: "run-1",
            clock=Clock(),
            resume=context,
        ).run(RunRequest(task_dir=tmp_path))

    assert backend.evaluated == []
    # The run is left as it ended, to be resumed again.
    assert store.read("run-1").attempt == 1


def test_scored_reports_take_the_newest_round_only_with_one_final_judge() -> None:
    reports = tuple(
        SubmissionReport(
            round_id=f"agent-{number}",
            status=SubmissionStatus.COMPLETED,
            rewards={"reward": score},
            score=score,
        )
        for number, score in ((1, 0.9), (10, 0.1), (2, 0.5))
    )

    assert scored_reports(reports, False) == reports
    assert scored_reports(reports, True) == (reports[1],)
    assert scored_reports((), True) == ()
    best = aggregate_result(
        run_id="run-1",
        reports=reports,
        primary_reward=None,
        direction="maximize",
        final_judge=True,
    )
    assert (best.best_round, best.best_score, best.total_rounds) == ("agent-10", 0.1, 3)
    assert (
        aggregate_result(
            run_id="run-1", reports=reports, primary_reward=None, direction="maximize"
        ).best_round
        == "agent-1"
    )


def test_the_production_backend_ends_an_agent_through_its_ender() -> None:
    from dataclasses import MISSING, fields

    from rsi_harness.models import ContainerRef
    from rsi_harness.runtime.coordinator import ProductionCoordinatorBackend

    required = {
        field.name: None
        for field in fields(ProductionCoordinatorBackend)
        if field.default is MISSING and field.default_factory is MISSING
    }
    work = ContainerRef(container_id="work-1", role="work")
    ended: list[ContainerRef] = []

    ProductionCoordinatorBackend(**required, agent_ender=ended.append).end_agent(work)

    assert ended == [work]
    with pytest.raises(SetupError, match="cannot end an Agent early"):
        ProductionCoordinatorBackend(**required).end_agent(work)
