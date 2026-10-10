"""Durable reservations and a single writer survive concurrency and crashes."""

import json
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.recovery import LeaseStore, ResourceLease
from tests.sandbox_helpers import make_profile, make_sandbox_grant


def make_lease(run_id="run-1"):
    return ResourceLease(
        run_id=run_id,
        task_id="task",
        coordinator_pid=1,
        coordinator_started_at=1.0,
        phase="preparing",
    )


def authority(store, run_id="run-1"):
    current = store.read(run_id)
    if current is None:
        current = make_lease(run_id)
        store.write(current)
    lock = threading.RLock()

    def mutate(transform):
        nonlocal current
        with lock:
            updated = transform(current)
            if updated is not current:
                store.write(updated)
                current = updated
            return current

    return mutate


def make_child(run_id="run-1", child_id="a" * 32, **updates):
    from rsi_harness.runtime.sandbox_contracts import SandboxLease, SandboxOwner

    values = dict(
        owner=SandboxOwner(run_id=run_id, task_id="task", phase="work"),
        child_id=child_id,
        planned_name=f"rsi-sandbox-{child_id}",
        image_id=make_profile().image,
        state="planned",
        reserved_lifetime_sec=30,
        cpus=1,
        memory_mb=256,
        created_at=100.0,
        expires_at=130.0,
        pending_mutation=True,
    )
    values.update(updates)
    return SandboxLease(**values)


def test_v4_is_read_as_child_free_without_rewriting_file(tmp_path):
    store = LeaseStore(tmp_path)
    raw = make_lease("old").model_dump(mode="json")
    raw["schema_version"] = 4
    raw.pop("sandboxes", None)
    raw.pop("sandbox_reservation", None)
    path = store.path_for("old")
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    lease = store.read("old")
    assert lease.schema_version == 6
    assert lease.sandboxes == ()
    assert lease.sandbox_reservation is None
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [3, 7, True, "5"])
def test_unknown_or_coerced_schema_cannot_hide_authority(tmp_path, version):
    store = LeaseStore(tmp_path)
    raw = make_lease().model_dump(mode="json")
    raw["schema_version"] = version
    store.path_for("run-1").write_text(json.dumps(raw))
    with pytest.raises(ValidationError):
        store.read("run-1")


def test_v4_with_new_child_authority_is_not_silently_downgraded(tmp_path):
    store = LeaseStore(tmp_path)
    raw = make_lease().model_dump(mode="json")
    raw.update(schema_version=4, sandboxes=[{"child_id": "hidden"}])
    store.path_for("run-1").write_text(json.dumps(raw))
    with pytest.raises((ValidationError, ValueError)):
        store.read("run-1")


def test_store_rejects_child_from_another_run(tmp_path):
    forged = make_lease().model_copy(update={"sandboxes": (make_child("other"),)})
    with pytest.raises(ValidationError, match="identity"):
        LeaseStore(tmp_path).write(forged)


def test_owner_judge_round_is_mandatory_and_work_has_no_round():
    from rsi_harness.runtime.sandbox_contracts import SandboxOwner

    with pytest.raises(ValidationError):
        SandboxOwner(run_id="run", task_id="task", phase="judge")
    with pytest.raises(ValidationError):
        SandboxOwner(run_id="run", task_id="task", phase="work", round_id="agent-1")


@pytest.fixture
def journal(tmp_path):
    from rsi_harness.runtime.sandbox_budget import SandboxJournal

    store = LeaseStore(tmp_path)
    mutate = authority(store)
    return SandboxJournal(mutate), store, mutate


def test_journal_persists_planned_and_actual_identity_without_second_writer(journal):
    journal, store, _ = journal
    child = make_child()
    journal.plan(child)
    assert store.read("run-1").sandboxes == (child,)
    journal.record_container(child.child_id, "b" * 64)
    journal.set_state(child.child_id, "running", pending_mutation=False)
    recorded = store.read("run-1").sandboxes[0]
    assert recorded.container_id == "b" * 64
    assert recorded.state == "running"
    assert recorded.pending_mutation is False
    journal.mark_removed(child.child_id)
    assert journal.snapshot()[0].state == "removed"
    assert journal.snapshot()[0].container_id == "b" * 64  # retained evidence


def test_journal_refuses_duplicate_identity_and_id_rebinding(journal):
    journal, _, _ = journal
    child = make_child()
    journal.plan(child)
    with pytest.raises(InfrastructureError):
        journal.plan(child)
    journal.record_container(child.child_id, "b" * 64)
    with pytest.raises(InfrastructureError):
        journal.record_container(child.child_id, "c" * 64)


def test_parent_and_children_share_atomic_mutator(journal):
    journal, store, mutate = journal
    start = threading.Barrier(3)

    def parent():
        start.wait()
        for index in range(20):
            mutate(
                lambda lease: lease.model_copy(
                    update={"phase_history": lease.phase_history + (f"phase-{index}",)}
                )
            )

    def children():
        start.wait()
        for index in range(20):
            journal.plan(make_child(child_id=f"{index:032x}"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(parent)
        second = executor.submit(children)
        start.wait()
        first.result(timeout=5)
        second.result(timeout=5)
    result = store.read("run-1")
    assert len(result.sandboxes) == 20
    assert len(result.phase_history) == 21


def test_snapshot_is_read_only_and_write_failure_keeps_previous_authority(
    journal, monkeypatch
):
    journal, store, _ = journal
    before = store.path_for("run-1").read_bytes()
    monkeypatch.setattr(
        store, "write", lambda lease: (_ for _ in ()).throw(OSError("disk failed"))
    )
    assert journal.snapshot() == ()
    with pytest.raises(OSError):
        journal.plan(make_child())
    assert journal.snapshot() == ()
    assert store.path_for("run-1").read_bytes() == before


def _reserve_process(root, run_id, barrier, results):
    from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool

    store = LeaseStore(root)
    mutate = authority(store, run_id)
    barrier.wait(timeout=5)
    try:
        SandboxAdmissionPool(store).reserve_run(run_id, make_sandbox_grant(), mutate)
    except SetupError:
        results.put("denied")
    else:
        results.put("reserved")


def test_two_processes_cannot_overbook_or_reclaim_dead_process(tmp_path):
    from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_reserve_process, args=(tmp_path, f"run-{i}", barrier, results)
        )
        for i in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0
    assert sorted([results.get(timeout=2), results.get(timeout=2)]) == [
        "denied",
        "reserved",
    ]
    store = LeaseStore(tmp_path)
    mutate = authority(store, "later")
    with pytest.raises(SetupError, match="pool"):
        SandboxAdmissionPool(store).reserve_run("later", make_sandbox_grant(), mutate)


def test_release_needs_removed_children_and_no_pending_mutations(journal):
    from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool

    journal, store, mutate = journal
    pool = SandboxAdmissionPool(store)
    pool.reserve_run("run-1", make_sandbox_grant(), mutate)
    child = make_child()
    journal.plan(child)
    with pytest.raises(InfrastructureError):
        pool.release_run("run-1", mutate)
    assert store.read("run-1").sandbox_reservation is not None
    journal.mark_removed(child.child_id)
    pool.release_run("run-1", mutate)
    assert store.read("run-1").sandbox_reservation is None
    # A profile grant never gives env authority to reconcile.
    assert not store.read("run-1").sandbox_env_authority
    pool.release_run("run-1", mutate)


def test_corrupt_other_lease_blocks_admission_without_changes(journal):
    from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool

    _, store, mutate = journal
    store.path_for("other").write_text("broken json")
    before = store.path_for("run-1").read_bytes()
    with pytest.raises(InfrastructureError, match="other"):
        SandboxAdmissionPool(store).reserve_run("run-1", make_sandbox_grant(), mutate)
    assert store.path_for("run-1").read_bytes() == before


def test_reservation_is_idempotent_but_cannot_change_envelope(journal):
    from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool

    _, store, mutate = journal
    pool = SandboxAdmissionPool(store)
    grant = make_sandbox_grant()
    pool.reserve_run("run-1", grant, mutate)
    pool.reserve_run("run-1", grant, mutate)
    with pytest.raises(InfrastructureError):
        pool.reserve_run("run-1", grant.model_copy(update={"reserved_cpus": 7}), mutate)
    assert store.read("run-1").sandbox_reservation.cpus == 6


def test_coordinator_exposes_one_authority_and_preserves_child_updates(tmp_path):
    from rsi_harness.models import RunRequest, RunStatus
    from rsi_harness.runtime.coordinator import RunCoordinator
    from rsi_harness.runtime.sandbox_budget import SandboxJournal
    from tests.runtime.test_coordinator import ScriptedBackend

    backend = ScriptedBackend(tmp_path)
    store = LeaseStore(tmp_path / "leases")
    mutations = []
    original_agent = backend.run_agent

    def agent(*args, **kwargs):
        journal = SandboxJournal(mutations[0])

        def record_children():
            for index in range(20):
                child = make_child(child_id=f"{index:032x}")
                child = child.model_copy(
                    update={
                        "owner": child.owner.model_copy(
                            update={"task_id": backend.plan.task.task_id}
                        )
                    }
                )
                journal.plan(child)
                journal.mark_removed(child.child_id)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(record_children)
            result = original_agent(*args, **kwargs)
            future.result(timeout=10)
        return result

    backend.run_agent = agent
    result = RunCoordinator(
        backend=backend,
        lease_store=store,
        clock=object(),
        run_id_factory=lambda: "run-1",
        on_lease_ready=mutations.append,
    ).run(RunRequest(task_dir=backend.plan.task.source_dir))
    assert result.status == RunStatus.COMPLETED
    assert len(mutations) == 1
    lease = store.read("run-1")
    assert len(lease.sandboxes) == 20
    assert all(child.state == "removed" for child in lease.sandboxes)
    assert lease.status == RunStatus.COMPLETED
    assert lease.work.container_id is None
