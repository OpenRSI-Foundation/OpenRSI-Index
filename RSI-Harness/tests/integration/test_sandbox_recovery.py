"""Real Docker acceptance for late-create containment and crash recovery."""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest

from rsi_harness.runtime.production import ProductionRecoveryBackend
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager
from rsi_harness.runtime.sandbox_contracts import SandboxError, SandboxOwner
from tests.integration.sandbox_support import (
    EmptyFirewallRecovery,
    EmptySnapshotRecovery,
    assert_no_sandbox_resources,
    create_broker,
    owned_resources,
    remove_exact_containers,
    require_sandbox_authority,
)

pytestmark = pytest.mark.integration


def _create_child_then_wait_for_termination(
    lease_root: str,
    image_id: str,
    run_id: str,
    task_id: str,
    connection,
) -> None:
    import docker

    client = docker.from_env(timeout=5)
    broker = create_broker(
        client,
        LeaseStore(Path(lease_root)),
        run_id=run_id,
        task_id=task_id,
        image_id=image_id,
        coordinator_pid=os.getpid(),
    )
    credentials = broker.open_session(
        SandboxOwner(run_id=run_id, task_id=task_id, phase="work"),
        deadline=time.monotonic() + 25.0,
    )
    child = broker.create(credentials.credential, "offline", 20.0, "create-1")
    connection.send((child.child_id, child.container_id, os.getpid()))
    connection.close()
    while True:
        time.sleep(60)


def test_real_delayed_create_cancel_contains_child_without_start(tmp_path: Path):
    client, image = require_sandbox_authority()
    run_id = f"sandbox-late-{uuid4().hex}"
    task_id = "sandbox-late-create"
    store = LeaseStore(tmp_path / "leases")
    broker = create_broker(
        client,
        store,
        run_id=run_id,
        task_id=task_id,
        image_id=image.id,
    )
    credentials = broker.open_session(
        SandboxOwner(run_id=run_id, task_id=task_id, phase="work"),
        deadline=time.monotonic() + 25.0,
    )
    backend = broker.backend
    original_create = backend.create
    original_start = backend.start
    daemon_created = multiprocessing.Event()
    release_response = multiprocessing.Event()
    start_calls: list[str] = []
    exact_ids: set[str] = set()

    def gated_create(lease, profile):
        identity = original_create(lease, profile)
        exact_ids.add(identity)
        daemon_created.set()
        assert release_response.wait(timeout=10)
        return identity

    def record_start(lease):
        start_calls.append(lease.container_id)
        return original_start(lease)

    backend.create = gated_create
    backend.start = record_start
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            creating = executor.submit(
                broker.create,
                credentials.credential,
                "offline",
                20.0,
                "create-1",
            )
            assert daemon_created.wait(timeout=10)
            resources_during_gate = owned_resources(client, run_id)
            assert len(resources_during_gate["containers"]) == 1
            late_identity = resources_during_gate["containers"][0]
            exact_ids.add(late_identity)
            container = client.containers.get(late_identity)
            container.reload()
            assert container.attrs["State"]["Running"] is False
            planned = store.read(run_id)
            assert planned is not None
            assert planned.sandboxes[0].container_id is None
            assert planned.sandboxes[0].pending_mutation is True

            broker.cancel_run()
            release_response.set()
            with pytest.raises(SandboxError, match="expired.*creation"):
                creating.result(timeout=10)

        assert start_calls == []
        assert_no_sandbox_resources(client, store, run_id)
    finally:
        release_response.set()
        try:
            broker.close()
        finally:
            remove_exact_containers(client, exact_ids)
            client.close()


def test_real_recovery_removes_only_crashed_run_child(tmp_path: Path):
    client, image = require_sandbox_authority()
    suffix = uuid4().hex
    crashed_run = f"sandbox-crashed-{suffix}"
    unrelated_run = f"sandbox-unrelated-{suffix}"
    task_id = "sandbox-recovery"
    store = LeaseStore(tmp_path / "leases")
    managed_root = tmp_path / "managed"
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    coordinator = context.Process(
        target=_create_child_then_wait_for_termination,
        args=(str(store.root), image.id, crashed_run, task_id, sender),
    )
    unrelated_broker = create_broker(
        client,
        store,
        run_id=unrelated_run,
        task_id=task_id,
        image_id=image.id,
    )
    unrelated_credentials = unrelated_broker.open_session(
        SandboxOwner(run_id=unrelated_run, task_id=task_id, phase="work"),
        deadline=time.monotonic() + 25.0,
    )
    unrelated_child = unrelated_broker.create(
        unrelated_credentials.credential,
        "offline",
        20.0,
        "create-1",
    )
    exact_ids = {unrelated_child.container_id}
    coordinator.start()
    sender.close()
    crashed_identity = None
    try:
        assert receiver.poll(20), (
            "crash fixture did not durably create a child; "
            f"coordinator exitcode={coordinator.exitcode}"
        )
        child_id, crashed_identity, reported_pid = receiver.recv()
        exact_ids.add(crashed_identity)
        assert reported_pid == coordinator.pid
        durable_before = store.read(crashed_run)
        assert durable_before is not None
        assert durable_before.sandboxes[0].child_id == child_id
        assert durable_before.sandboxes[0].container_id == crashed_identity
        assert durable_before.sandboxes[0].state == "running"
        assert durable_before.coordinator_pid == reported_pid

        unrelated_before = owned_resources(client, unrelated_run)
        unrelated_lease_before = store.path_for(unrelated_run).read_bytes()
        unrelated_container = client.containers.get(unrelated_child.container_id)
        unrelated_container.reload()
        assert unrelated_container.attrs["State"]["Running"] is True

        coordinator.terminate()
        coordinator.join(timeout=10)
        assert coordinator.exitcode is not None

        recovered = RecoveryManager(
            store=store,
            backend=ProductionRecoveryBackend(
                client,
                EmptySnapshotRecovery(),
                EmptyFirewallRecovery(),
            ),
            managed_root=managed_root,
        ).recover(crashed_run)

        assert recovered == (crashed_run,)
        assert_no_sandbox_resources(client, store, crashed_run)
        assert owned_resources(client, unrelated_run) == unrelated_before
        assert store.path_for(unrelated_run).read_bytes() == unrelated_lease_before
        unrelated_container.reload()
        assert unrelated_container.attrs["State"]["Running"] is True
    finally:
        receiver.close()
        if coordinator.is_alive():
            coordinator.terminate()
            coordinator.join(timeout=10)
        try:
            unrelated_broker.close()
        finally:
            remove_exact_containers(
                client, {identity for identity in exact_ids if identity}
            )
            client.close()
