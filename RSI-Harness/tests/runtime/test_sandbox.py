"""Phase ownership and accounting under real journal/concurrent broker calls."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rsi_harness.errors import InfrastructureError, RetryableSubmissionError
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_contracts import (
    SandboxBundleEntry,
    SandboxChildStopped,
    SandboxError,
    SandboxOwner,
)
from tests.runtime.test_sandbox_budget import authority
from tests.sandbox_helpers import FakeClock, FakeSandboxBackend, make_sandbox_grant


@pytest.fixture
def kit(tmp_path):
    from rsi_harness.runtime.sandbox import SandboxBroker

    clock, backend = FakeClock(), FakeSandboxBackend()
    journal = SandboxJournal(authority(LeaseStore(tmp_path)))
    broker = SandboxBroker(make_sandbox_grant(), backend, journal, clock)
    return broker, backend, clock


@pytest.fixture
def work(kit):
    return kit[0].open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), 1000
    )


def new_judge(broker, round_id="r1"):
    return broker.open_judge(
        SandboxOwner(run_id="run-1", task_id="task", phase="judge", round_id=round_id),
        500,
    )


def create(broker, session, request="one", lifetime=30):
    return broker.create(session.credential, "offline", lifetime, request)


def test_create_retry_does_not_charge_twice(kit, work):
    broker, backend, _ = kit
    child = create(broker, work)
    assert create(broker, work) == child
    assert len(broker.journal.snapshot()) == 1
    assert [operation for operation, _ in backend.events] == ["create", "start"]
    assert work.credential not in repr(broker.journal.snapshot())
    assert work.credential not in repr(work)


def test_conflicting_idempotency_key_is_rejected(kit, work):
    child = create(kit[0], work)
    with pytest.raises(SandboxError, match="invalid.*request_id"):
        create(kit[0], work, lifetime=29)
    assert kit[0].status(work.credential, child.child_id)["state"] == "running"


def test_failed_upload_replays_diagnostic_without_retaining_exception_frames(
    kit, work, monkeypatch
):
    broker, backend, _ = kit
    child = create(broker, work)
    entry = SandboxBundleEntry(path="x", kind="file", mode=0o644, data=b"payload")

    def reject(*args, **kwargs):
        raise SandboxError("invalid", "bundle", "file changed")

    monkeypatch.setattr(backend, "upload", reject)
    failures = []
    for _ in range(2):
        with pytest.raises(SandboxError, match="file changed") as failure:
            broker.upload(
                work.credential,
                child.child_id,
                "/workspace",
                (entry,),
                "failed-upload",
                10,
            )
        failures.append(failure.value)
    assert failures[0] is not failures[1]
    session = broker._sessions["work"]
    assert not isinstance(session.requests["failed-upload"].error, BaseException)


def test_judge_cannot_use_work_handle_and_stale_token_is_revoked(kit, work):
    broker, _, _ = kit
    child = create(broker, work)
    broker.freeze_work()
    judge = new_judge(broker)
    with pytest.raises(SandboxError, match="permission"):
        broker.status(judge.credential, child.child_id)
    with pytest.raises(SandboxError, match="busy"):
        create(broker, work, "blocked")
    judge_child = create(broker, judge)
    broker.close_judge()
    assert (
        next(
            c for c in broker.journal.snapshot() if c.child_id == judge_child.child_id
        ).state
        == "removed"
    )
    with pytest.raises(SandboxError, match="permission"):
        broker.status(judge.credential, judge_child.child_id)
    broker.resume_work()
    assert broker.status(work.credential, child.child_id)["state"] == "running"


def test_destroy_is_owned_idempotent_and_creation_count_not_refunded(kit, work):
    broker, backend, _ = kit
    for index in range(8):
        child = create(broker, work, str(index))
        broker.destroy(work.credential, child.child_id)
        broker.destroy(work.credential, child.child_id)
    with pytest.raises(SandboxError, match="quota.*max_created"):
        create(broker, work, "nine")
    assert not backend.states
    with pytest.raises(SandboxError, match="permission"):
        broker.destroy(work.credential, "0" * 32)


@pytest.mark.parametrize("lifetime", [True, "10", 0, -1, float("inf"), 121])
def test_invalid_create_is_rejected_before_journal_or_docker(kit, work, lifetime):
    broker, backend, _ = kit
    with pytest.raises(SandboxError):
        create(broker, work, lifetime=lifetime)
    assert not backend.events
    assert broker.journal.snapshot() == ()


def test_upload_retries_are_idempotent_and_conflicting_data_is_rejected(kit, work):
    broker, backend, _ = kit
    child = create(broker, work)
    entry = SandboxBundleEntry(path="x", kind="file", mode=0o644, data=b"ok")
    for _ in range(2):
        broker.upload(
            work.credential, child.child_id, "/workspace", (entry,), "upload-1", 10
        )
    with pytest.raises(SandboxError, match="invalid.*request_id"):
        broker.upload(
            work.credential,
            child.child_id,
            "/workspace",
            (entry.model_copy(update={"data": b"no"}),),
            "upload-1",
            10,
        )
    assert [op for op, _ in backend.events].count("upload") == 1


@pytest.mark.parametrize("operation", ["upload", "download"])
def test_proven_stopped_transfer_is_removed_by_next_freeze(
    kit, work, monkeypatch, operation
):
    broker, backend, _ = kit
    child = create(broker, work)

    def stopped(*args, **kwargs):
        raise SandboxChildStopped("quota", "transfer", "transfer limit stopped child")

    monkeypatch.setattr(backend, operation, stopped)
    with pytest.raises(SandboxChildStopped):
        if operation == "upload":
            broker.upload(
                work.credential,
                child.child_id,
                "/workspace",
                (),
                "stopped-upload",
                10,
            )
        else:
            broker.download(work.credential, child.child_id, "/workspace", ["."], 10)

    assert broker.journal.snapshot()[0].state == "stopped"
    broker.freeze_work()
    assert child.child_id not in backend.states
    assert broker.recovery_required is False
    assert not any(event == ("pause", child.child_id) for event in backend.events)


@pytest.mark.parametrize("operation", ["upload", "download"])
def test_invalid_transfer_keeps_child_running(kit, work, monkeypatch, operation):
    broker, backend, _ = kit
    child = create(broker, work)

    def invalid(*args, **kwargs):
        raise SandboxError("invalid", "transfer", "bad selection")

    monkeypatch.setattr(backend, operation, invalid)
    with pytest.raises(SandboxError, match="invalid"):
        if operation == "upload":
            broker.upload(
                work.credential,
                child.child_id,
                "/workspace",
                (),
                "invalid-upload",
                10,
            )
        else:
            broker.download(work.credential, child.child_id, "/workspace", ["."], 10)

    assert broker.journal.snapshot()[0].state == "running"
    broker.freeze_work()
    assert backend.states[child.child_id]["Paused"] is True
    assert broker.recovery_required is False


def test_exec_deadline_is_minimum_of_child_and_owner_deadlines(kit, work):
    broker, backend, clock = kit
    child = create(broker, work)
    clock.now = 110
    result = broker.execute(
        work.credential, child.child_id, ["false"], "/workspace", {}, 100
    )
    assert backend.last_deadline == 130
    assert result.exit_code == 3
    assert result.stdout == "out" and result.stderr == "err"


@pytest.mark.parametrize(
    "operation", ["create", "execute", "upload", "download", "destroy"]
)
def test_busy_work_operation_rejects_submit_before_any_pause(kit, work, operation):
    broker, backend, _ = kit
    child = create(broker, work)
    entered, release = threading.Event(), threading.Event()
    backend.hooks["terminate" if operation == "destroy" else operation] = lambda _: (
        entered.set(),
        release.wait(3),
    )
    calls = {
        "create": lambda: create(broker, work, "two"),
        "execute": lambda: broker.execute(
            work.credential, child.child_id, ["true"], "/workspace", {}, 5
        ),
        "upload": lambda: broker.upload(
            work.credential, child.child_id, "/workspace", (), "up", 5
        ),
        "download": lambda: broker.download(
            work.credential, child.child_id, "/workspace", ["."], 5
        ),
        "destroy": lambda: broker.destroy(work.credential, child.child_id),
    }
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(calls[operation])
        try:
            assert entered.wait(2)
            with pytest.raises(RetryableSubmissionError, match="sandbox.*busy"):
                broker.freeze_work()
            assert not any(op == "pause" for op, _ in backend.events)
        finally:
            release.set()
        future.result()
    broker.freeze_work()


def test_simultaneous_create_reserves_live_slot_before_docker_returns(kit, work):
    broker, backend, _ = kit
    entered, release = threading.Event(), threading.Event()
    create(broker, work)
    backend.hooks["create"] = lambda _: (entered.set(), release.wait(3))
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(create, broker, work, "two")
        try:
            assert entered.wait(2)
            with pytest.raises(SandboxError, match="quota.*max_live"):
                create(broker, work, "three")
            assert len(broker.journal.snapshot()) == 2
        finally:
            release.set()
        future.result()


def test_cancel_late_create_never_starts_or_returns_a_handle(kit, work):
    broker, backend, _ = kit
    entered, release = threading.Event(), threading.Event()
    backend.hooks["create"] = lambda _: (entered.set(), release.wait(3))
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(create, broker, work)
        try:
            assert entered.wait(2)
            broker.cancel_run()
            assert broker.journal.snapshot()[0].pending_mutation
        finally:
            release.set()
        with pytest.raises(SandboxError, match="expired"):
            future.result()
    broker.close()
    assert not backend.states
    assert not any(op == "start" for op, _ in backend.events)
    assert broker.journal.snapshot()[0].state == "removed"


def test_paused_expiry_never_resumes_candidate_and_retains_authority(kit, work):
    broker, backend, clock = kit
    backend.paused_killer = False
    child = create(broker, work)
    broker.freeze_work()
    clock.now = 131
    broker.sweep_expired()
    assert broker.status(work.credential, child.child_id)["expired"] is True
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.resume_work()
    assert backend.states[child.child_id]["Paused"]
    assert not any(op == "resume" for op, _ in backend.events)
    assert broker.journal.snapshot()[0].state != "removed"


def test_paused_expiry_is_killed_without_thaw_by_the_paused_killer(kit, work):
    """With the paused killer (production) an expired paused child is removed."""
    broker, backend, clock = kit
    child = create(broker, work)
    broker.freeze_work()
    clock.now = 131
    broker.sweep_expired()
    assert child.child_id not in backend.states
    assert not any(op == "resume" for op, _ in backend.events)
    assert broker.journal.snapshot()[0].state == "removed"
    assert not broker.recovery_required


@pytest.mark.parametrize("interruption", ["cancel", "child_deadline", "run_deadline"])
def test_resume_rechecks_authority_after_pending_journal_write(
    kit, work, monkeypatch, interruption
):
    broker, backend, clock = kit
    child = create(broker, work)
    broker.freeze_work()
    original = broker._state
    injected = False

    def delayed_journal(record, state, *, pending=False):
        nonlocal injected
        original(record, state, pending=pending)
        if pending and record.inflight == "resume" and not injected:
            injected = True
            if interruption == "cancel":
                broker.cancel_run()
            elif interruption == "child_deadline":
                clock.now = 131
            else:
                clock.now = 1001

    monkeypatch.setattr(broker, "_state", delayed_journal)
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.resume_work()
    assert injected
    assert not any(op == "resume" for op, _ in backend.events)
    assert backend.states[child.child_id]["Paused"]


def test_expiry_during_admitted_resume_cannot_reopen_work(kit, work):
    broker, backend, clock = kit
    child = create(broker, work)
    broker.freeze_work()
    backend.hooks["resume"] = lambda _: setattr(clock, "now", 131)
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.resume_work()
    assert broker._sessions["work"].frozen
    assert backend.states[child.child_id]["Paused"]


def test_cancel_during_admitted_resume_denies_next_child_and_reopen(kit, work):
    broker, backend, _ = kit
    first = create(broker, work, "first")
    create(broker, work, "second")
    broker.freeze_work()
    entered, release = threading.Event(), threading.Event()
    backend.hooks["resume"] = lambda _: (entered.set(), release.wait(3))
    with ThreadPoolExecutor(2) as pool:
        resuming = pool.submit(broker.resume_work)
        try:
            assert entered.wait(2)
            cancelling = pool.submit(broker.cancel_run)
            assert broker._cancel_requested.wait(2)
            # An already-admitted daemon request precedes revocation. The
            # cancellation request is nevertheless visible before it returns.
            assert not broker._cancelled
        finally:
            release.set()
        with pytest.raises(InfrastructureError, match="recovery"):
            resuming.result(timeout=3)
        cancelling.result(timeout=3)
    assert [identity for op, identity in backend.events if op == "resume"] == [
        first.child_id
    ]
    assert not broker.can_resume
    assert broker._sessions["work"].frozen


@pytest.mark.parametrize("started,observed", [(False, 0), (True, 0), (True, 3)])
def test_failed_download_charges_only_known_transfer_usage(
    kit, work, monkeypatch, started, observed
):
    from rsi_harness.runtime.sandbox_contracts import SandboxDownloadError

    broker, backend, _ = kit
    child = create(broker, work)
    initial_operations = broker._usage["max_operations"]

    def failed(*args, **kwargs):
        raise SandboxDownloadError(
            "invalid", "paths", "missing file", download_bytes=observed,
            operation_started=started,
        )

    monkeypatch.setattr(backend, "download", failed)
    with pytest.raises(SandboxDownloadError):
        broker.download(work.credential, child.child_id, "/workspace", ["missing"], 1)
    for usage in (broker._usage, broker._sessions["work"].usage):
        assert usage["max_download_bytes"] == observed
        assert usage["max_operations"] == initial_operations + int(started)
    monkeypatch.setattr(backend, "download", lambda *a, **kw: (
        SandboxBundleEntry(path="out", kind="file", mode=0o644, data=b"ok"),
    ))
    result = broker.download(
        work.credential, child.child_id, "/workspace", ["out"], 1
    )
    assert result[0].data == b"ok"
    assert broker._usage["max_download_bytes"] == observed + 2


def test_unknown_download_failure_retains_reservation_and_requires_recovery(
    kit, work, monkeypatch
):
    from rsi_harness.integrations.sandbox_client import MAX_BUNDLE_BYTES

    broker, backend, _ = kit
    child = create(broker, work)

    def unknown(*args, **kwargs):
        raise SandboxError("unknown-outcome", "download", "daemon response lost")

    monkeypatch.setattr(backend, "download", unknown)
    with pytest.raises(SandboxError):
        broker.download(work.credential, child.child_id, "/workspace", ["out"], 1)
    assert broker._usage["max_download_bytes"] == MAX_BUNDLE_BYTES
    assert broker.recovery_required


def test_watchdog_terminates_without_waiting_for_operation_worker(kit, work):
    broker, backend, clock = kit
    child = create(broker, work)
    entered, stopped = threading.Event(), threading.Event()
    backend.hooks["execute"] = lambda _: (entered.set(), stopped.wait(3))
    backend.hooks["terminate"] = lambda _: stopped.set()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            broker.execute,
            work.credential,
            child.child_id,
            ["sleep"],
            "/workspace",
            {},
            100,
        )
        assert entered.wait(2)
        clock.now = 131
        broker.sweep_expired()
        assert stopped.is_set()
        with pytest.raises(SandboxError, match="expired"):
            future.result()
    broker.close()
    assert not backend.states


def test_durable_identity_failure_prevents_start_and_retains_reservation(
    kit, work, monkeypatch
):
    broker, backend, _ = kit
    monkeypatch.setattr(
        broker.journal,
        "record_container",
        lambda *a: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(InfrastructureError, match="recovery"):
        create(broker, work)
    assert not any(op == "start" for op, _ in backend.events)
    assert broker.journal.snapshot()[0].pending_mutation


def test_inactive_session_cannot_create_or_reset_deadline(kit):
    broker, _, _ = kit
    session = broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
    )
    with pytest.raises(SandboxError, match="busy"):
        create(broker, session)
    broker.activate_work(120)
    with pytest.raises(InfrastructureError, match="deadline"):
        broker.activate_work(500)
    child = create(broker, session)
    broker.execute(session.credential, child.child_id, ["true"], "/workspace", {}, 100)
    assert kit[1].last_deadline == 120


def test_judge_round_keeps_own_deadline_after_work_deadline(kit):
    """Judge 30 s before Work's end keeps its 900 s verifier deadline."""
    broker, backend, clock = kit
    work = broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
    )
    broker.activate_work(1000)
    broker.freeze_work()
    clock.now = 970
    judge = broker.open_judge(
        SandboxOwner(run_id="run-1", task_id="task", phase="judge", round_id="r1"),
        None,
    )
    broker.activate_judge(970 + 900)
    child = create(broker, judge, lifetime=120)

    clock.now = 1001
    broker.sweep_expired()
    assert child.child_id in backend.states
    assert broker.status(judge.credential, child.child_id)["expired"] is False
    assert create(broker, judge, "late").state == "running"
    broker.execute(judge.credential, child.child_id, ["true"], "/workspace", {}, 60)
    assert backend.last_deadline == 1061
    with pytest.raises(SandboxError, match="expired"):
        broker.capabilities(work.credential)
    with pytest.raises(SandboxError, match="expired"):
        create(broker, work, "work-late")
    assert broker.can_resume is False
    with pytest.raises(InfrastructureError, match="reopen"):
        broker.reopen_work()

    # Only the Judge's own deadline ends its round.
    clock.now = 1870
    with pytest.raises(SandboxError, match="expired"):
        create(broker, judge, "too-late")
    broker.sweep_expired()
    assert not backend.states
    assert all(lease.state == "removed" for lease in broker.journal.snapshot())
    broker.close_judge()
    with pytest.raises(SandboxError, match="expired"):
        broker.open_session(
            SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
        )
    broker.open_judge(
        SandboxOwner(run_id="run-1", task_id="task", phase="judge", round_id="r2"),
        5000,
    )
    assert broker._sessions["judge"].deadline == 5000


@pytest.mark.parametrize("end", ["cancel_work", "deadline"])
def test_work_end_leaves_paused_family_while_judge_round_continues(kit, end):
    """The watchdog must not fail the run closed over a paused Work child."""
    broker, backend, clock = kit
    work = broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), None
    )
    broker.activate_work(1000)
    clock.now = 950
    work_child = create(broker, work, "work-child", lifetime=120)
    broker.freeze_work()
    clock.now = 970
    judge = broker.open_judge(
        SandboxOwner(run_id="run-1", task_id="task", phase="judge", round_id="r1"),
        None,
    )
    broker.activate_judge(970 + 900)

    if end == "cancel_work":
        broker.cancel_work()
    else:
        clock.now = 1001
    broker.sweep_expired()

    judge_child = create(broker, judge, "judge-child")
    broker.execute(
        judge.credential, judge_child.child_id, ["true"], "/workspace", {}, 5
    )
    with pytest.raises(
        SandboxError, match="permission" if end == "cancel_work" else "expired"
    ):
        broker.status(work.credential, work_child.child_id)
    # Work's end removes nothing and is not a run cancellation.
    assert not any(op in ("terminate", "remove") for op, _ in backend.events)
    assert backend.states[work_child.child_id]["Paused"]
    assert not broker._cancelled and not broker.recovery_required
    assert broker.can_resume is False
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.resume_work()
    broker.close_judge()
    assert judge_child.child_id not in backend.states
    assert not broker.recovery_required

    # The paused killer ends the paused child without a thaw, so close()
    # removes it instead of failing closed (M0's gap, closed in M5).
    broker.close()
    assert not broker.recovery_required
    assert work_child.child_id not in backend.states
    assert not any(op == "resume" for op, _ in backend.events)
    assert (
        next(
            lease
            for lease in broker.journal.snapshot()
            if lease.child_id == work_child.child_id
        ).state
        == "removed"
    )


def test_judge_only_grant_run_deadline_is_immutable_but_judge_keeps_its_own(
    tmp_path,
):
    from rsi_harness.runtime.sandbox import SandboxBroker

    clock, backend = FakeClock(), FakeSandboxBackend()
    journal = SandboxJournal(authority(LeaseStore(tmp_path)))
    grant = make_sandbox_grant().model_copy(update={"work": None})
    broker = SandboxBroker(grant, backend, journal, clock)

    broker.activate_work(105)
    with pytest.raises(InfrastructureError, match="deadline"):
        broker.activate_work(500)
    judge = broker.open_judge(
        SandboxOwner(run_id="run-1", task_id="task", phase="judge", round_id="r1"),
        None,
    )
    broker.activate_judge(200)
    child = create(broker, judge, lifetime=120)
    broker.execute(judge.credential, child.child_id, ["true"], "/workspace", {}, 100)
    assert backend.last_deadline == 200

    clock.now = 106
    broker.execute(judge.credential, child.child_id, ["true"], "/workspace", {}, 10)
    assert backend.last_deadline == 116
    assert broker.can_resume is False


def test_judge_only_grant_cannot_activate_run_after_cancellation(tmp_path):
    from rsi_harness.runtime.sandbox import SandboxBroker

    clock, backend = FakeClock(), FakeSandboxBackend()
    journal = SandboxJournal(authority(LeaseStore(tmp_path)))
    grant = make_sandbox_grant().model_copy(update={"work": None})
    broker = SandboxBroker(grant, backend, journal, clock)

    broker.cancel_run()
    with pytest.raises(InfrastructureError, match="revoked"):
        broker.activate_work(105)


def test_partial_freeze_failure_retains_family_and_blocks_new_operations(kit, work):
    broker, backend, _ = kit
    child = create(broker, work)
    create(broker, work, "two")

    def fail_second(lease):
        if lease.child_id != child.child_id:
            raise OSError("pause failed")

    backend.hooks["pause"] = fail_second
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.freeze_work()
    with pytest.raises(SandboxError):
        create(broker, work, "three")
    assert broker.recovery_required


def test_closed_judge_budget_resets_per_round_but_not_run(kit, work):
    broker, _, _ = kit
    broker.freeze_work()
    for index in range(8):
        session = new_judge(broker, f"r{index}")
        create(broker, session)
        broker.close_judge()
    session = new_judge(broker, "r9")
    with pytest.raises(SandboxError, match="quota.*max_created"):
        create(broker, session)


def test_small_exec_output_charges_actual_bytes_not_per_call_ceiling(kit, work):
    broker, _, _ = kit
    child = create(broker, work)
    for _ in range(12):
        result = broker.execute(
            work.credential, child.child_id, ["true"], "/workspace", {}, 5
        )
        assert result.stdout == "out"


def test_expired_running_child_is_removed_before_freeze_not_paused(kit, work):
    broker, backend, clock = kit
    child = create(broker, work)
    clock.now = 131
    broker.freeze_work()
    assert child.child_id not in backend.states
    assert not any(op == "pause" for op, _ in backend.events)
    broker.resume_work()


def test_journal_failure_before_exec_blocks_admission_without_execution(
    kit, work, monkeypatch
):
    broker, backend, _ = kit
    child = create(broker, work)
    monkeypatch.setattr(
        broker.journal,
        "set_state",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.execute(work.credential, child.child_id, ["true"], "/workspace", {}, 5)
    assert broker.recovery_required
    assert not any(op == "execute" for op, _ in backend.events)


def test_unknown_exec_mutation_cannot_be_declared_removed(kit, work):
    broker, backend, _ = kit
    child = create(broker, work)
    backend.hooks["execute"] = lambda _: (_ for _ in ()).throw(
        SandboxError("unknown-outcome", "exec", "reader still alive")
    )
    with pytest.raises(SandboxError, match="unknown-outcome"):
        broker.execute(work.credential, child.child_id, ["true"], "/workspace", {}, 5)
    assert broker.recovery_required
    assert broker.journal.snapshot()[0].pending_mutation
    broker.cancel_run()
    assert child.child_id in backend.states
    assert not any(op == "remove" for op, _ in backend.events)


def test_cancel_during_resume_cannot_resume_later_children(kit, work):
    broker, backend, _ = kit
    first = create(broker, work)
    second = create(broker, work, "two")
    broker.freeze_work()
    backend.hooks["resume"] = lambda _: broker.cancel_run()
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.resume_work()
    assert ("resume", second.child_id) not in backend.events
    assert backend.states[first.child_id]["Paused"]


def test_failed_first_pause_still_contains_remaining_family(kit, work):
    broker, backend, _ = kit
    first = create(broker, work)
    second = create(broker, work, "two")

    def fail_first(lease):
        if lease.child_id == first.child_id:
            raise OSError("first pause failed")

    backend.hooks["pause"] = fail_first
    with pytest.raises(InfrastructureError, match="recovery"):
        broker.freeze_work()
    assert backend.states[second.child_id]["Paused"]


def test_real_watchdog_loop_expires_idle_child_without_requests(kit, work):
    broker, backend, clock = kit
    child = create(broker, work)
    removed = threading.Event()
    backend.hooks["remove"] = lambda _: removed.set()
    broker.start()
    try:
        clock.now = 131
        assert removed.wait(2)
    finally:
        broker.close()
    assert child.child_id not in backend.states
