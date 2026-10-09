"""Coordinator retires Work before draining and never fabricates resume."""

import threading
from types import SimpleNamespace

import pytest

from rsi_harness.errors import InfrastructureError, RetryableSubmissionError
from rsi_harness.models import (
    AgentRunResult,
    RunRequest,
    RunStatus,
    SubmissionStatus,
)
from rsi_harness.runtime.coordinator import (
    CoordinatorState,
    RunCoordinator,
    StartedSubmissionServer,
    _RoundEvaluator,
)
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_lifecycle import NullSandboxLifecycle
from rsi_harness.runtime.submissions import JudgeEndpoint, SubmissionService
from tests.runtime.test_coordinator import Clock, ScriptedBackend


class Lifecycle(NullSandboxLifecycle):
    enabled = True

    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail
        self.can_resume = True
        self.work_ended_normally = False
        self.work_ended = threading.Event()

    def cancel_work(self):
        self.events.append(("sandbox_cancel_work", None))
        self.can_resume = False
        self.work_ended_normally = True
        self.work_ended.set()

    def cancel_run(self):
        self.events.append(("sandbox_cancel", None))
        self.can_resume = False
        self.work_ended_normally = False
        self.work_ended.set()

    def close(self):
        self.events.append(("sandbox_close", None))
        if self.fail:
            raise InfrastructureError("sandbox child cleanup recovery_required")

    def release_resources(self):
        self.events.append(("sandbox_release", None))


def test_work_retired_before_submission_drain_and_children_close_before_parents(
    tmp_path,
):
    backend = ScriptedBackend(tmp_path)
    lifecycle = Lifecycle(backend.events)
    result = RunCoordinator(
        backend=backend,
        lease_store=LeaseStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        sandbox_lifecycle=lifecycle,
    ).run(RunRequest(task_dir=tmp_path))
    assert result.status == RunStatus.COMPLETED
    names = [name for name, _ in backend.events]
    assert "sandbox_cancel" not in names
    assert names.index("sandbox_cancel_work") < names.index("server_stop")
    # close() cancels the whole run only after accepted rounds have drained.
    assert names.index("server_stop") < names.index("sandbox_close")
    assert names.index("sandbox_close") < names.index("work_remove")
    assert names.index("work_remove") < names.index("sandbox_release")


class _EndedWorkObserver:
    """A Judge whose Work ended mid-round leaves the parent paused."""

    def __init__(self, observer):
        self.observer = observer

    def resource_event(self, name, **values):
        if name != "work_unpaused":
            self.observer.resource_event(name, **values)


def _serve_through_submission_service(backend, rounds=()):
    """Serve rounds through the real SubmissionService.

    The drain joins accepted rounds before acceptance closes, so a queued
    round reaches the evaluator, as when a round ends just before the drain.
    """

    def start_server(evaluator, artifacts, clock):
        backend.events.append(("server_start", None))
        backend.service = SubmissionService(
            evaluator=evaluator, artifact_writer=artifacts, clock=clock
        )

        def drain():
            for accepted in rounds:
                accepted.join(10)
            backend.events.append(("server_stop", None))
            backend.service.close()

        return StartedSubmissionServer(
            service=backend.service,
            owner=SimpleNamespace(stop=drain),
            endpoint=JudgeEndpoint(
                url="http://172.30.0.1:9020", bind_host="127.0.0.1", port=9020
            ),
        )

    backend.start_server = start_server


@pytest.mark.parametrize("timed_out", [True, False])
def test_round_in_flight_when_work_ends_keeps_reward_without_recovery(
    tmp_path, timed_out
):
    """The agent's deadline passes while an accepted round is still judging.

    A submission queued behind that round is never judged: the ended Work
    stays paused, and a second round would try to pause it again.
    """
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    lifecycle = Lifecycle(backend.events)
    judging, rounds, judged, rejected = threading.Event(), [], [], []
    evaluate = backend.evaluate_submission

    def judge_across_work_end(request, *, lifecycle_observer):
        judged.append(request.round_id)
        judging.set()
        assert lifecycle.work_ended.wait(5)
        if lifecycle.work_ended_normally:
            # The Judge contract once Work ended normally (judge.py).
            lifecycle_observer.resource_event("submission_closed")
        else:
            # The Judge contract for a revoked family (judge.py).
            backend.reports[0] = backend.reports[0].model_copy(
                update={
                    "status": SubmissionStatus.INFRASTRUCTURE_ERROR,
                    "score": None,
                    "error": "recovery_required: sandbox Work phase cancelled",
                }
            )
        return evaluate(
            request, lifecycle_observer=_EndedWorkObserver(lifecycle_observer)
        )

    def submit():
        try:
            backend.service.submit(backend.submit_token)
        except Exception as error:
            rejected.append(str(error))

    def agent_ends_mid_round(prepared, work, timeout):
        del prepared, timeout
        backend.events.append(("agent_start", work.container_id))
        rounds.extend(threading.Thread(target=submit) for _ in range(2))
        rounds[0].start()
        assert judging.wait(5)
        rounds[1].start()
        return AgentRunResult(exit_code=None if timed_out else 0, timed_out=timed_out)

    backend.evaluate_submission = judge_across_work_end
    backend.run_agent = agent_ends_mid_round
    _serve_through_submission_service(backend, rounds)
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        sandbox_lifecycle=lifecycle,
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert [report.status for report in result.reports] == [SubmissionStatus.COMPLETED]
    assert result.best_score == 0.25
    # Reaching the Agent budget is the normal end of a run: the outcome
    # follows the accepted round either way.
    assert result.status == RunStatus.COMPLETED
    assert judged == ["agent-1"]
    assert len(rejected) == 1 and "submissions are closed" in rejected[0]
    assert not store.read("run-1").recovery_required
    assert coordinator.phase_history[-3:] == (
        RunStatus.JUDGING,
        RunStatus.AGENT_RUNNING,
        result.status,
    )
    names = [name for name, _ in backend.events]
    assert "work_resume" not in names
    assert names.index("sandbox_cancel_work") < names.index("judge_create")
    assert names.index("judge_create") < names.index("server_stop")
    assert names.index("server_stop") < names.index("sandbox_close")
    assert names.index("sandbox_close") < names.index("work_remove")


def test_rejected_round_after_work_ends_closes_round_without_resume(tmp_path):
    """A retry rejected once Work ended must not strand the round phase."""
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    lifecycle = Lifecycle(backend.events)
    judged = []

    def reject_after_work_end(request, *, lifecycle_observer):
        judged.append(request.round_id)
        for name in ("work_pause_planned", "work_paused"):
            lifecycle_observer.resource_event(
                name, work_container_id=request.work_container.container_id
            )
        # Work's deadline passes while its GPUs are still busy; the Judge
        # leaves Work paused and closes the stream (judge.py).
        lifecycle.can_resume, lifecycle.work_ended_normally = False, True
        lifecycle_observer.resource_event("submission_closed")
        raise RetryableSubmissionError(
            "release all Work GPU processes before retrying: PID 42"
        )

    def agent_retries_after_work_end(prepared, work, timeout):
        del prepared, work, timeout
        with pytest.raises(RetryableSubmissionError):
            backend.service.submit(backend.submit_token)
        with pytest.raises(InfrastructureError, match="submissions are closed"):
            backend.service.submit(backend.submit_token)
        return AgentRunResult(exit_code=0)

    backend.evaluate_submission = reject_after_work_end
    backend.run_agent = agent_retries_after_work_end
    _serve_through_submission_service(backend)
    coordinator = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        sandbox_lifecycle=lifecycle,
    )

    result = coordinator.run(RunRequest(task_dir=tmp_path))

    assert result.status == RunStatus.NO_VALID_SUBMISSION
    assert not result.reports
    assert judged == ["agent-1"]
    assert not store.read("run-1").recovery_required
    assert coordinator.phase_history[-3:] == (
        RunStatus.SNAPSHOTTING,
        RunStatus.AGENT_RUNNING,
        RunStatus.NO_VALID_SUBMISSION,
    )
    assert "work_resume" not in [name for name, _ in backend.events]


def test_user_cancellation_still_revokes_whole_run_before_drain(tmp_path):
    backend = ScriptedBackend(tmp_path)

    def interrupt(prepared, work, timeout):
        del prepared, timeout
        backend.events.append(("agent_start", work.container_id))
        raise KeyboardInterrupt

    backend.run_agent = interrupt
    result = RunCoordinator(
        backend=backend,
        lease_store=LeaseStore(tmp_path / "leases"),
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        sandbox_lifecycle=Lifecycle(backend.events),
    ).run(RunRequest(task_dir=tmp_path))
    assert result.status == RunStatus.CANCELLED
    names = [name for name, _ in backend.events]
    assert "sandbox_cancel_work" not in names
    assert names.index("sandbox_cancel") < names.index("server_stop")


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_sandbox_revocation_still_contains_work(tmp_path, cancelled):
    """Failed Work retirement and failed user cancellation both fail closed."""
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    lifecycle = Lifecycle(backend.events)

    def revocation_failure():
        raise InfrastructureError("sandbox revocation failed")

    if cancelled:

        def interrupt(prepared, work, timeout):
            del prepared, timeout
            backend.events.append(("agent_start", work.container_id))
            raise KeyboardInterrupt

        backend.run_agent = interrupt
        lifecycle.cancel_run = revocation_failure
    else:
        lifecycle.cancel_work = revocation_failure
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        sandbox_lifecycle=lifecycle,
    ).run(RunRequest(task_dir=tmp_path))
    assert result.status == (RunStatus.CANCELLED if cancelled else RunStatus.FAILED)
    retained = store.read("run-1")
    assert retained.recovery_required
    assert "sandbox revocation failed" in retained.error
    assert retained.work.paused
    assert retained.work.container_id == "work-1"
    assert not any(
        name in {"work_remove", "network_remove", "sandbox_release", "work_resume"}
        for name, _ in backend.events
    )


def test_child_cleanup_failure_retains_parent_authority(tmp_path):
    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        sandbox_lifecycle=Lifecycle(backend.events, True),
    ).run(RunRequest(task_dir=tmp_path))
    assert result.status == RunStatus.FAILED
    assert store.read("run-1").recovery_required
    assert store.read("run-1").work.paused
    assert not any(
        name in {"work_remove", "sandbox_release"} for name, _ in backend.events
    )


@pytest.mark.parametrize("failure", ["child", "drain", "stop"])
def test_failed_cleanup_still_quiesces_work_without_releasing_authority(
    tmp_path, failure
):
    backend = ScriptedBackend(tmp_path)
    backend.fail_server_stop = failure == "drain"
    backend.fail_stop_agent = failure == "stop"
    store = LeaseStore(tmp_path / "leases")
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        run_id_factory=lambda: "run-1",
        clock=Clock(),
        sandbox_lifecycle=Lifecycle(backend.events, failure == "child"),
    ).run(RunRequest(task_dir=tmp_path))
    assert result.status == RunStatus.FAILED
    retained = store.read("run-1")
    assert retained.recovery_required
    assert retained.work.paused
    assert retained.work.planned_quiescence is None
    assert retained.work.container_id == "work-1"
    assert not any(
        name in {"work_remove", "network_remove", "sandbox_release", "work_resume"}
        for name, _ in backend.events
    )


@pytest.mark.parametrize("failure", ["child", "cancel", "drain", "none"])
def test_persistent_lease_write_failure_still_contains_durable_work(tmp_path, failure):
    class FailingStore(LeaseStore):
        failed = False

        def write(self, lease):
            if self.failed:
                raise OSError("lease storage unavailable")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    store = FailingStore(tmp_path / "leases")
    run_agent = backend.run_agent

    def run_then_lose_storage(*args):
        result = run_agent(*args)
        store.failed = True
        return result

    backend.run_agent = run_then_lose_storage
    backend.fail_server_stop = failure == "drain"
    lifecycle = Lifecycle(backend.events, failure == "child")
    if failure == "cancel":

        def cancel_failure():
            raise InfrastructureError("sandbox revocation failed")

        lifecycle.cancel_work = cancel_failure
    with pytest.raises(OSError, match="lease storage unavailable"):
        RunCoordinator(
            backend=backend,
            lease_store=store,
            run_id_factory=lambda: "run-1",
            clock=Clock(),
            sandbox_lifecycle=lifecycle,
        ).run(RunRequest(task_dir=tmp_path))
    names = [name for name, _ in backend.events]
    assert "work_pause" in names
    assert not {
        "work_remove",
        "network_remove",
        "sandbox_release",
        "work_resume",
    } & set(names)
    retained = store.read("run-1")
    assert retained.work.container_id == "work-1"
    assert retained.work.network_id == "network-work-1"
    # The failed write must not fabricate durable observations of the stop.
    assert not retained.work.paused
    assert not retained.work.stopped


def test_storage_failure_does_not_bypass_undurable_work_identity(tmp_path):
    class FailingStore(LeaseStore):
        failed = False

        def write(self, lease):
            if self.failed:
                raise OSError("actual identity unavailable")
            super().write(lease)

    backend = ScriptedBackend(tmp_path)
    store = FailingStore(tmp_path / "leases")
    create_work = backend.create_work

    def create_then_lose_storage(*args):
        result = create_work(*args)
        store.failed = True
        return result

    backend.create_work = create_then_lose_storage
    with pytest.raises(OSError, match="actual identity unavailable"):
        RunCoordinator(
            backend=backend,
            lease_store=store,
            run_id_factory=lambda: "run-1",
            clock=Clock(),
            sandbox_lifecycle=Lifecycle(backend.events, True),
        ).run(RunRequest(task_dir=tmp_path))
    retained = store.read("run-1")
    assert retained.work.planned_container is not None
    assert retained.work.container_id is None
    assert not any(
        name in {"work_pause", "work_stopped", "work_remove", "sandbox_release"}
        for name, _ in backend.events
    )


@pytest.mark.parametrize("raises", [False, True])
def test_cancelled_family_disables_both_round_evaluator_resume_fallbacks(
    tmp_path, raises
):
    backend = ScriptedBackend(tmp_path)
    state = CoordinatorState(run_id="run-1")
    for status in (
        RunStatus.PREPARING,
        RunStatus.AGENT_RUNNING,
        RunStatus.SNAPSHOTTING,
        RunStatus.JUDGING,
    ):
        if state.status != status:
            state.transition(status)
    lifecycle = Lifecycle(backend.events)
    lifecycle.can_resume = False

    def evaluate(request, *, lifecycle_observer):
        if raises:
            raise RuntimeError("finished while caller cancelled")
        return backend.reports[0]

    backend.evaluate_submission = evaluate
    evaluator = _RoundEvaluator(
        backend=backend,
        state=state,
        transition=state.transition,
        persist_recovery=lambda _: None,
        update_work=lambda **kw: None,
        update_judge=lambda **kw: None,
        sandbox_lifecycle=lifecycle,
    )
    # Only the fallback reads work_container; no unrelated fixture setup needed.
    from types import SimpleNamespace

    request = SimpleNamespace(work_container=SimpleNamespace(container_id="work"))
    if raises:
        with pytest.raises(RuntimeError):
            evaluator.evaluate(request)
    else:
        evaluator.evaluate(request)
    assert state.status == RunStatus.JUDGING
    assert not any(name == "work_resume" for name, _ in backend.events)
