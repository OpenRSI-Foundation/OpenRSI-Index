"""Real Docker regressions for independent parent and child containment."""

import tempfile
import time
from pathlib import Path
from uuid import uuid4

import pytest

from rsi_harness.models import AgentRunResult, ContainerRef, RunRequest, RunStatus
from rsi_harness.runtime.coordinator import RunCoordinator
from rsi_harness.runtime.recovery import LeaseStore
from tests.integration.sandbox_support import require_sandbox_authority
from tests.runtime.test_coordinator import Clock, ScriptedBackend

pytestmark = pytest.mark.integration


class PersistentWriteFailureStore(LeaseStore):
    """Lose durable writes only when an individual test explicitly arms it."""

    failed = False

    def write(self, lease):
        if self.failed:
            raise OSError("lease storage unavailable")
        super().write(lease)


def _remove_attested_parent(
    client,
    *,
    identity,
    name,
    image_id,
    labels,
):
    """Remove one exact fixture after a fresh full-identity inspection."""
    from docker.errors import NotFound

    try:
        parent = client.containers.get(identity)
    except NotFound:
        return
    parent.reload()
    assert parent.id == identity
    assert parent.attrs.get("Name") == "/" + name
    assert parent.attrs.get("Image") == image_id
    assert (parent.attrs.get("Config") or {}).get("Labels") == labels
    state = parent.attrs.get("State") or {}
    if state.get("Paused"):
        parent.unpause()
        parent.reload()
    if parent.attrs["State"].get("Running"):
        parent.kill(signal="SIGKILL")
        parent.reload()
    assert parent.attrs["State"].get("Running") is False
    assert parent.attrs["State"].get("Paused") is False
    parent.remove(force=False, v=True)


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_failed_child_cleanup_contains_real_parent(tmp_path, cleanup_failure):
    client, image = require_sandbox_authority()
    parent = None
    broker = None
    transport_backend = None
    original_remove = None
    temporary = tempfile.TemporaryDirectory(prefix="rsi-audit2-parent-")
    identity = "audit2-parent-cleanup-" + uuid4().hex
    try:
        parent = client.containers.create(
            image.id,
            entrypoint=["python3"],
            command=[
                "-c",
                "import time,signal;from pathlib import Path;"
                "signal.alarm(30);p=Path('/workspace/tick');"
                "[(p.write_text(str(i)),time.sleep(.03)) for i in range(900)]",
            ],
            network_mode="none",
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            runtime="runc",
            mem_limit="128m",
            memswap_limit="128m",
            pids_limit=16,
            nano_cpus=1_000_000_000,
            tmpfs={"/workspace": "rw,exec,nosuid,nodev,size=8m"},
            labels={
                "rsi-harness.run-id": identity,
                "rsi-harness.role": "audit2-parent",
            },
            environment={"NVIDIA_VISIBLE_DEVICES": "void"},
        )

        class Backend(ScriptedBackend):
            def create_work(self, *args):
                self.events.append(("work_create", parent.id))
                return ContainerRef(container_id=parent.id, role="work")

            def start_work(self, work):
                super().start_work(work)
                parent.start()

            def run_agent(self, prepared, work, timeout):
                self.events.append(("agent_start", work.container_id))
                lifecycle.activate_work(time.monotonic() + 25)
                token = lifecycle.prepare_work().environment["RSI_SANDBOX_TOKEN"]
                broker.create(token, "offline", 20, "cleanup-child")
                deadline = time.monotonic() + 4
                while time.monotonic() < deadline:
                    if (
                        parent.exec_run(["test", "-f", "/workspace/tick"]).exit_code
                        == 0
                    ):
                        return AgentRunResult(exit_code=0)
                    time.sleep(0.02)
                raise AssertionError("parent fixture did not start")

            def stop_agent(self, work):
                super().stop_agent(work)
                parent.stop(timeout=1)

            def quiesce_work(self, work):
                parent.reload()
                self.work_quiescence = (
                    "stopped" if not parent.attrs["State"]["Running"] else "paused"
                )
                if self.work_quiescence == "paused":
                    parent.pause()
                return super().quiesce_work(work)

        backend = Backend(tmp_path)
        store = LeaseStore(tmp_path / "leases")
        from rsi_harness.runtime.sandbox import SandboxBroker
        from rsi_harness.runtime.sandbox_budget import SandboxJournal
        from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend
        from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
        from tests.integration.sandbox_support import sandbox_grant

        lifecycle = SandboxLifecycle()
        transport_backend = SandboxDockerBackend(client)
        original_remove = transport_backend.remove
        if cleanup_failure:

            def fail_remove(lease):
                raise RuntimeError("injected transient child Docker remove failure")

            transport_backend.remove = fail_remove

        def on_ready(mutate):
            nonlocal broker
            broker = SandboxBroker(
                sandbox_grant(image.id), transport_backend, SandboxJournal(mutate)
            )
            lifecycle.configure(
                broker, Path(temporary.name) / "sb", "run-1", backend.plan.task.task_id
            )
            lifecycle.prepare_work()

        result = RunCoordinator(
            backend=backend,
            lease_store=store,
            run_id_factory=lambda: "run-1",
            clock=Clock(),
            sandbox_lifecycle=lifecycle,
            on_lease_ready=on_ready,
        ).run(RunRequest(task_dir=tmp_path))
        parent.reload()
        state = parent.attrs["State"]
        assert result.status == (
            RunStatus.FAILED if cleanup_failure else RunStatus.NO_VALID_SUBMISSION
        )
        assert not state["Running"] or state["Paused"], (
            "failed child cleanup skipped parent containment"
        )
    finally:
        if broker is not None:
            transport_backend.remove = original_remove
            broker.close()
            lifecycle.close()
            for lease in broker.journal.snapshot():
                assert lease.state == "removed"
        if parent is not None:
            parent.remove(force=True, v=True)
        client.close()
        temporary.cleanup()


def test_persistent_write_failure_still_quiesces_durable_real_work(tmp_path):
    """A failed child close cannot leave already-durable Work executing."""
    from docker.errors import NotFound

    from rsi_harness.runtime.sandbox import SandboxBroker
    from rsi_harness.runtime.sandbox_budget import SandboxJournal
    from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend
    from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
    from tests.integration.sandbox_support import sandbox_grant

    client, image = require_sandbox_authority()
    parent = None
    broker = None
    lifecycle = SandboxLifecycle()
    store = PersistentWriteFailureStore(tmp_path / "leases")
    temporary = tempfile.TemporaryDirectory(prefix="rsi-audit2-storage-")
    # ScriptedBackend's complete split-WORKDIR fixture is intentionally bound to
    # run-1; keep that production-shaped authority and use a unique Docker label.
    run_id = "run-1"
    docker_run_label = "audit2-storage-coordinator-" + uuid4().hex
    task_id = "task-1"
    parent_name = "rsi-audit2-storage-work-" + uuid4().hex
    parent_labels = {
        "rsi-harness.run-id": docker_run_label,
        "rsi-harness.task-id": task_id,
        "rsi-harness.role": "work",
    }
    durable = None
    try:
        parent = client.containers.create(
            image.id,
            name=parent_name,
            entrypoint=["python3"],
            command=[
                "-c",
                "import time,signal;from pathlib import Path;"
                "signal.alarm(30);p=Path('/workspace/tick');"
                "[(p.write_text(str(i)),time.sleep(.03)) for i in range(900)]",
            ],
            network_mode="none",
            read_only=True,
            privileged=False,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            runtime="runc",
            mem_limit="128m",
            memswap_limit="128m",
            pids_limit=16,
            nano_cpus=1_000_000_000,
            tmpfs={"/workspace": "rw,exec,nosuid,nodev,size=8m"},
            labels=parent_labels,
            environment={"NVIDIA_VISIBLE_DEVICES": "void"},
        )

        class Backend(ScriptedBackend):
            def create_work(self, *args):
                self.events.append(("work_create", parent.id))
                return ContainerRef(container_id=parent.id, role="work")

            def start_work(self, work):
                super().start_work(work)
                parent.start()

            def run_agent(self, prepared, work, timeout):
                nonlocal durable
                self.events.append(("agent_start", work.container_id))
                lifecycle.activate_work(time.monotonic() + 25)
                token = lifecycle.prepare_work().environment["RSI_SANDBOX_TOKEN"]
                child = broker.create(token, "offline", 20, "storage-child")
                deadline = time.monotonic() + 4
                while parent.exec_run(["test", "-f", "/workspace/tick"]).exit_code:
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                durable = store.read(run_id)
                assert durable is not None
                assert durable.work.container_id == parent.id
                assert any(
                    lease.container_id == child.container_id
                    for lease in durable.sandboxes
                )
                # Docker control remains healthy; only future durable writes fail.
                store.failed = True
                return AgentRunResult(exit_code=0)

            def quiesce_work(self, work):
                parent.reload()
                if parent.attrs["State"]["Running"]:
                    parent.pause()
                    self.work_quiescence = "paused"
                else:
                    self.work_quiescence = "stopped"
                self.events.append(("work_pause", work.container_id))
                return self.work_quiescence

        backend = Backend(tmp_path)
        transport = SandboxDockerBackend(client)

        def on_ready(mutate):
            nonlocal broker
            broker = SandboxBroker(
                sandbox_grant(image.id), transport, SandboxJournal(mutate)
            )
            lifecycle.configure(
                broker,
                Path(temporary.name) / "sb",
                run_id,
                backend.plan.task.task_id,
            )
            lifecycle.prepare_work()

        with pytest.raises(OSError, match="lease storage unavailable"):
            RunCoordinator(
                backend=backend,
                lease_store=store,
                run_id_factory=lambda: run_id,
                clock=Clock(),
                sandbox_lifecycle=lifecycle,
                on_lease_ready=on_ready,
            ).run(RunRequest(task_dir=tmp_path))

        parent.reload()
        assert parent.attrs["State"]["Running"] is True
        assert parent.attrs["State"]["Paused"] is True
        assert store.read(run_id) == durable
        assert durable.work.container_id == parent.id
        assert durable.sandboxes[0].container_id is not None
        # Docker removal completed before the journal write failed. The stale
        # durable child authority must remain available for idempotent recovery.
        with pytest.raises(NotFound):
            client.containers.get(durable.sandboxes[0].container_id)
        assert durable.sandboxes[0].state == "running"
        names = [name for name, _ in backend.events]
        assert "work_pause" in names
        assert not {
            "work_remove",
            "network_remove",
            "work_resume",
        } & set(names)
    finally:
        # Restore the normal storage path before fixture cleanup. Cleanup itself
        # uses production attestation and only exact IDs created above.
        store.failed = False
        if broker is not None:
            broker.close()
            lifecycle.close()
            for lease in broker.journal.snapshot():
                assert lease.state == "removed"
        if parent is not None:
            _remove_attested_parent(
                client,
                identity=parent.id,
                name=parent_name,
                image_id=image.id,
                labels=parent_labels,
            )
        assert (
            client.containers.list(
                all=True,
                filters={"label": f"rsi-harness.run-id={docker_run_label}"},
            )
            == []
        )
        client.close()
        temporary.cleanup()


def test_recovery_paused_first_still_removes_running_judge_child(tmp_path):
    """Reconstitute a crashed run; no watchdog is running in this fixture."""
    from docker.errors import NotFound

    from rsi_harness.runtime.production import ProductionRecoveryBackend
    from rsi_harness.runtime.recovery import RecoveryManager
    from rsi_harness.runtime.sandbox_contracts import SandboxOwner
    from rsi_harness.runtime.sandbox_docker import (
        attest_sandbox_identity,
        sandbox_labels,
    )
    from tests.integration.sandbox_support import (
        EmptyFirewallRecovery,
        EmptySnapshotRecovery,
        create_broker,
        require_sandbox_authority,
    )

    client, image = require_sandbox_authority()
    run_id = "audit2-recover-" + uuid4().hex
    store = LeaseStore(tmp_path / "leases")
    broker = create_broker(
        client,
        store,
        run_id=run_id,
        task_id="audit2-recovery",
        image_id=image.id,
    )
    broker.grant = broker.grant.model_copy(update={"judge": broker.grant.work})
    assert broker._watchdog is None
    try:
        work = broker.open_session(
            SandboxOwner(run_id=run_id, task_id="audit2-recovery", phase="work"),
            time.monotonic() + 25,
        )
        work_child = broker.create(work.credential, "offline", 20, "work")
        broker.freeze_work()
        judge = broker.open_judge(
            SandboxOwner(
                run_id=run_id,
                task_id="audit2-recovery",
                phase="judge",
                round_id="r1",
            ),
            time.monotonic() + 25,
        )
        judge_child = broker.create(judge.credential, "offline", 20, "judge")
        heartbeat = (
            "python3 -c 'import time;from pathlib import Path;"
            'p=Path("/workspace/audit-heartbeat");'
            "[(p.write_text(str(i)),time.sleep(.03)) for i in range(300)]' "
            "</dev/null >/tmp/audit-heartbeat.log 2>&1 &"
        )
        result = broker.execute(
            judge.credential,
            judge_child.child_id,
            ["/bin/sh", "-c", heartbeat],
            "/workspace",
            {},
            3,
        )
        assert result.exit_code == 0
        second = client.containers.get(judge_child.container_id)
        ready_deadline = time.monotonic() + 3
        while second.exec_run(["test", "-f", "/workspace/audit-heartbeat"]).exit_code:
            assert time.monotonic() < ready_deadline
            time.sleep(0.02)
        manager = RecoveryManager(
            store=store,
            backend=ProductionRecoveryBackend(
                client,
                EmptySnapshotRecovery(),
                EmptyFirewallRecovery(),
            ),
            managed_root=tmp_path / "managed",
        )

        for _ in range(2):
            with pytest.raises(RuntimeError, match="paused"):
                manager.recover(run_id)
            first = client.containers.get(work_child.container_id)
            assert first.attrs["State"]["Paused"] is True
            with pytest.raises(NotFound):
                client.containers.get(judge_child.container_id)
            retained = store.read(run_id)
            assert retained.recovery_required
            assert retained.sandboxes[0].state != "removed"
            assert retained.sandboxes[1].state == "removed"
    finally:
        # Only exact fixtures in this run, after fresh full ownership attestation.
        # Work contains only the trusted idle sleep and readiness probe.
        for lease in broker.journal.snapshot():
            for child in client.containers.list(
                all=True,
                filters={
                    "label": [f"{k}={v}" for k, v in sandbox_labels(lease).items()]
                },
            ):
                child.reload()
                attest_sandbox_identity(lease, child.attrs)
                if child.attrs["State"]["Paused"]:
                    child.unpause()
                child.reload()
                if child.attrs["State"]["Running"]:
                    child.kill(signal="SIGKILL")
                child.remove(force=False, v=True)
        assert (
            client.containers.list(
                all=True,
                filters={"label": f"rsi-harness.run-id={run_id}"},
            )
            == []
        )
        client.close()


def test_recovery_write_failure_contains_durable_real_child_and_judge(tmp_path):
    """Storage loss permits containment, but not deletion or authority release."""
    from rsi_harness.runtime.production import ProductionRecoveryBackend
    from rsi_harness.runtime.recovery import RecoveryManager
    from rsi_harness.runtime.sandbox_contracts import SandboxOwner, SandboxReservation
    from rsi_harness.runtime.sandbox_docker import sandbox_labels
    from tests.integration.sandbox_support import (
        EmptyFirewallRecovery,
        EmptySnapshotRecovery,
        create_broker,
    )

    client, image = require_sandbox_authority()
    run_id = "audit2-storage-recovery-" + uuid4().hex
    task_id = "audit2-storage-recovery"
    store = PersistentWriteFailureStore(tmp_path / "leases")
    broker = create_broker(
        client,
        store,
        run_id=run_id,
        task_id=task_id,
        image_id=image.id,
    )
    judge = None
    judge_name = "rsi-audit2-storage-judge-" + uuid4().hex
    judge_labels = {
        "rsi-harness.run-id": run_id,
        "rsi-harness.task-id": task_id,
        "rsi-harness.role": "judge",
    }
    child_lease = None
    durable = None
    try:
        work = broker.open_session(
            SandboxOwner(run_id=run_id, task_id=task_id, phase="work"),
            time.monotonic() + 25,
        )
        child_lease = broker.create(
            work.credential,
            "offline",
            20,
            "storage-recovery-child",
        )
        judge = client.containers.create(
            image.id,
            name=judge_name,
            entrypoint=["/bin/sleep"],
            command=["30"],
            network_mode="none",
            read_only=True,
            privileged=False,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            runtime="runc",
            mem_limit="128m",
            memswap_limit="128m",
            pids_limit=16,
            nano_cpus=1_000_000_000,
            labels=judge_labels,
            environment={"NVIDIA_VISIBLE_DEVICES": "void"},
        )
        judge.start()
        durable = store.read(run_id)
        assert durable is not None
        durable = durable.model_copy(
            update={
                "sandbox_reservation": SandboxReservation(
                    cpus=1,
                    memory_mb=128,
                    pool_cpus=8,
                    pool_memory_mb=4096,
                ),
                "work": durable.work.model_copy(
                    update={
                        "planned_network": run_id + "-work-network",
                        "network_id": run_id + "-work-network-id",
                        "network_name": run_id + "-work-network",
                    }
                ),
                "judge": durable.judge.model_copy(
                    update={
                        "round_id": "r1",
                        "planned_container": judge_name,
                        "container_id": judge.id,
                        "planned_snapshot": "snapshot:" + run_id + ":r1",
                        "snapshot_lease_id": run_id + "-snapshot",
                    }
                ),
            }
        )
        store.write(durable)
        store.failed = True

        manager = RecoveryManager(
            store=store,
            backend=ProductionRecoveryBackend(
                client,
                EmptySnapshotRecovery(),
                EmptyFirewallRecovery(),
            ),
            managed_root=tmp_path / "managed",
        )
        with pytest.raises(OSError, match="lease storage unavailable"):
            manager.recover(run_id)

        child = client.containers.get(child_lease.container_id)
        child.reload()
        judge.reload()
        assert child.attrs["State"]["Running"] is False
        assert child.attrs["State"]["Paused"] is False
        assert judge.attrs["State"]["Running"] is False
        assert store.read(run_id) == durable
        assert durable.sandboxes[0].container_id == child.id
        assert durable.sandboxes[0].state == "running"
        assert durable.sandbox_reservation is not None
        assert durable.work.network_id == run_id + "-work-network-id"
        assert durable.judge.container_id == judge.id
        assert durable.judge.snapshot_lease_id == run_id + "-snapshot"
    finally:
        store.failed = False
        if broker is not None:
            broker.close()
            for lease in broker.journal.snapshot():
                assert lease.state == "removed"
        if judge is not None:
            _remove_attested_parent(
                client,
                identity=judge.id,
                name=judge_name,
                image_id=image.id,
                labels=judge_labels,
            )
        # Every sandbox cleanup path used its exact durable child authority.
        if child_lease is not None:
            assert (
                client.containers.list(
                    all=True,
                    filters={
                        "label": [
                            f"{key}={value}"
                            for key, value in sandbox_labels(child_lease).items()
                        ]
                    },
                )
                == []
            )
        assert (
            client.containers.list(
                all=True,
                filters={"label": f"rsi-harness.run-id={run_id}"},
            )
            == []
        )
        client.close()
