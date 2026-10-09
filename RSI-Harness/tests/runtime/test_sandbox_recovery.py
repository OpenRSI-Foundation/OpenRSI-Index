"""Child recovery never borrows a parent's looser identity or unpause path."""

from types import SimpleNamespace

import pytest

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.production import ProductionRecoveryBackend
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager
from rsi_harness.runtime.sandbox_contracts import SandboxOwner
from rsi_harness.runtime.sandbox_docker import sandbox_labels
from tests.runtime.test_recovery import RecoveryBackend
from tests.runtime.test_recovery import lease as parent_lease
from tests.runtime.test_sandbox_budget import make_child


class ChildRecoveryBackend(RecoveryBackend):
    def inspect_container(self, identity):
        for container_id, state in self.containers.items():
            if identity in (container_id, state.get("name")):
                return state
        return None

    def terminate_sandbox(self, child):
        self.events.append(("sandbox_terminate", child.container_id))
        state = self.containers[child.container_id]
        if state["paused"]:
            raise InfrastructureError("paused sandbox cannot be safely terminated")
        state["running"] = False

    def remove_sandbox(self, child):
        self.remove_container(child.container_id)


def setup_recovery(tmp_path, *, paused=False, actual=True):
    original = parent_lease(tmp_path)
    child = make_child(container_id="b" * 64 if actual else None)
    child = child.model_copy(
        update={"owner": child.owner.model_copy(update={"task_id": original.task_id})}
    )
    original = original.model_copy(update={"sandboxes": (child,)})
    store = LeaseStore(tmp_path / "leases")
    store.write(original)
    backend = ChildRecoveryBackend()
    backend.containers["b" * 64] = {
        "id": "b" * 64,
        "name": child.planned_name,
        "image_id": child.image_id,
        "labels": sandbox_labels(child),
        "running": True,
        "paused": paused,
    }
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )
    return manager, store, backend, child


@pytest.mark.parametrize("actual", [True, False])
def test_child_reclaimed_before_any_parent_including_planned_create(tmp_path, actual):
    manager, store, backend, child = setup_recovery(tmp_path, actual=actual)
    unrelated = {
        "labels": {"rsi-harness.run-id": "other"},
        "running": True,
        "paused": False,
    }
    backend.containers["unrelated"] = unrelated.copy()
    manager.recover("run-1")
    assert "b" * 64 not in backend.containers
    events = [name for name, _ in backend.events]
    assert events.index("sandbox_terminate") < events.index("stop")
    recovered = store.read("run-1").sandboxes[0]
    assert recovered.child_id == child.child_id
    assert recovered.state == "removed"
    assert not recovered.pending_mutation
    assert backend.containers["unrelated"] == unrelated


def test_paused_child_failure_prevents_parent_cleanup_and_keeps_authority(tmp_path):
    manager, store, backend, _ = setup_recovery(tmp_path, paused=True)
    with pytest.raises(RuntimeError, match="paused"):
        manager.recover("run-1")
    assert "work-real" in backend.containers
    assert "judge-real" in backend.containers
    assert not backend.containers["judge-real"]["running"]
    assert backend.containers["work-real"]["paused"]
    assert not any(name in ("unpause", "remove") for name, _ in backend.events)
    assert store.read("run-1").recovery_required
    assert store.read("run-1").sandboxes[0].state != "removed"


@pytest.mark.parametrize("actual", [True, False])
def test_persistent_lease_write_failure_contains_only_durable_child(
    tmp_path, monkeypatch, actual
):
    manager, store, backend, child = setup_recovery(tmp_path, actual=actual)
    retained = store.read("run-1")

    def failed_write(lease):
        raise OSError("lease storage unavailable")

    monkeypatch.setattr(store, "write", failed_write)
    with pytest.raises(OSError, match="lease storage unavailable"):
        manager.recover("run-1")
    assert backend.containers["b" * 64]["running"] is not actual
    assert (("sandbox_terminate", child.container_id) in backend.events) is actual
    assert not any(name in {"remove", "unpause"} for name, _ in backend.events)
    assert not backend.containers["judge-real"]["running"]
    assert backend.containers["work-real"]["paused"]
    assert store.read("run-1") == retained
    assert backend.networks and backend.snapshots


@pytest.mark.parametrize("failure", ["paused", "identity", "remove"])
def test_one_bad_child_does_not_skip_other_child_containment(tmp_path, failure):
    manager, store, backend, first = setup_recovery(
        tmp_path, paused=failure == "paused"
    )
    second = make_child(
        child_id="c" * 32,
        container_id="d" * 64,
        owner=SandboxOwner(
            run_id=first.owner.run_id,
            task_id=first.owner.task_id,
            phase="judge",
            round_id="round-1",
        ),
        state="running",
        pending_mutation=False,
    )
    store.write(store.read("run-1").model_copy(update={"sandboxes": (first, second)}))
    backend.containers[second.container_id] = {
        "id": second.container_id,
        "name": second.planned_name,
        "image_id": second.image_id,
        "labels": sandbox_labels(second),
        "running": True,
        "paused": False,
    }
    if failure == "identity":
        backend.containers[first.container_id]["image_id"] = "sha256:" + "f" * 64
    if failure == "remove":
        remove = backend.remove_sandbox

        def fail_first(child):
            if child.child_id == first.child_id:
                raise RuntimeError("first removal unavailable")
            remove(child)

        backend.remove_sandbox = fail_first
    for _ in range(2):
        with pytest.raises(RuntimeError, match="sandbox"):
            manager.recover("run-1")
        assert second.container_id not in backend.containers
        retained = store.read("run-1")
        assert retained.recovery_required
        assert retained.sandboxes[0].state != "removed"
        assert retained.sandboxes[1].state == "removed"
        assert not backend.containers["judge-real"]["running"]
        assert backend.containers["work-real"]["paused"]
        assert backend.snapshots
        assert backend.networks
    if failure in {"paused", "identity"}:
        assert backend.containers[first.container_id]["running"]


def test_parent_stop_failure_does_not_skip_other_parent_containment(tmp_path):
    manager, store, backend, _ = setup_recovery(tmp_path, paused=True)
    backend.containers["work-real"]["paused"] = False
    stop = backend.stop_container

    def fail_judge(identity):
        if identity == "judge-real":
            raise RuntimeError("Judge stop unavailable")
        stop(identity)

    backend.stop_container = fail_judge
    with pytest.raises(RuntimeError, match="Judge stop unavailable"):
        manager.recover("run-1")
    assert backend.containers["judge-real"]["running"]
    assert not backend.containers["work-real"]["running"]
    assert store.read("run-1").recovery_required
    assert backend.networks and backend.snapshots


def test_parent_identity_drift_is_not_containment_authority(tmp_path):
    manager, store, backend, _ = setup_recovery(tmp_path, paused=True)
    inspect = backend.inspect_container

    def replaced_judge(identity):
        state = inspect(identity)
        if identity == "judge-real":
            return {**state, "labels": {"rsi-harness.run-id": "other"}}
        return state

    backend.inspect_container = replaced_judge
    with pytest.raises(RuntimeError, match="identity changed"):
        manager.recover("run-1")
    assert backend.containers["judge-real"]["running"]
    assert not any(name == "stop" for name, _ in backend.events)
    assert store.read("run-1").recovery_required


@pytest.mark.parametrize("role", ["judge", "helper"])
def test_parent_inspection_failure_does_not_skip_same_role_sibling(tmp_path, role):
    manager, store, backend, _ = setup_recovery(tmp_path, paused=True)
    labels = {
        "rsi-harness.run-id": "run-1",
        "rsi-harness.task-id": "task-1",
        "rsi-harness.role": role,
    }
    for name in ("uninspectable", "sibling"):
        backend.containers[name] = {"labels": labels, "running": True, "paused": False}
    inspect = backend.inspect_container

    def fail_one(identity):
        if identity == "uninspectable":
            raise RuntimeError("one parent inspection unavailable")
        return inspect(identity)

    backend.inspect_container = fail_one
    with pytest.raises(RuntimeError, match="inspection unavailable"):
        manager.recover("run-1")
    assert backend.containers["uninspectable"]["running"]
    assert not backend.containers["sibling"]["running"]
    assert backend.containers["work-real"]["paused"]
    assert store.read("run-1").recovery_required
    assert backend.networks and backend.snapshots


def test_production_parent_discovery_does_not_inspect_whole_collection(tmp_path):
    manager, store, backend, _ = setup_recovery(tmp_path, paused=True)
    backend.containers["judge-second"] = dict(backend.containers["judge-real"])
    listed = backend.list_containers
    inspect = backend.inspect_container

    def list_api(*, all, filters):
        assert all is True
        labels = dict(item.split("=", 1) for item in filters["label"])
        return [
            {"Id": identity, "Labels": state["labels"]}
            for identity, state in listed(labels=labels)
        ]

    def fail_judge(identity):
        if identity == "judge-real":
            raise RuntimeError("one Docker inspection unavailable")
        return inspect(identity)

    def eager_list(**kwargs):
        raise RuntimeError("collection list tried inspecting the broken container")

    production = ProductionRecoveryBackend(
        SimpleNamespace(
            api=SimpleNamespace(containers=list_api),
            containers=SimpleNamespace(list=eager_list),
        ),
        object(),
        object(),
    )
    backend.list_container_candidates = production.list_container_candidates
    backend.inspect_container = fail_judge
    with pytest.raises(RuntimeError, match="inspection unavailable"):
        manager.recover("run-1")
    assert not backend.containers["judge-second"]["running"]
    assert backend.containers["judge-real"]["running"]
    assert store.read("run-1").recovery_required


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other-name"),
        ("image_id", "sha256:" + "c" * 64),
        ("labels", {"rsi-harness.run-id": "other"}),
    ],
)
def test_fresh_identity_drift_is_not_deletion_authority(tmp_path, field, value):
    manager, store, backend, _ = setup_recovery(tmp_path)
    backend.containers["b" * 64][field] = value
    with pytest.raises(RuntimeError, match="sandbox"):
        manager.recover("run-1")
    assert "b" * 64 in backend.containers
    assert not any(
        name in ("remove", "sandbox_terminate") for name, _ in backend.events
    )
    assert not backend.containers["judge-real"]["running"]
    assert store.read("run-1").recovery_required


def test_absent_pending_create_is_not_proof_it_will_never_complete(tmp_path):
    manager, store, backend, _ = setup_recovery(tmp_path, actual=False)
    del backend.containers["b" * 64]
    with pytest.raises(RuntimeError, match="pending"):
        manager.recover("run-1")
    assert "work-real" in backend.containers
    assert store.read("run-1").sandboxes[0].pending_mutation
