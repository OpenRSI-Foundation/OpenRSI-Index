"""Real Docker: kill -9 the process holding the broker, then recover.

A spawned child plays the coordinator: it holds the run's lease lock, owns a
production SandboxBroker with brokered envs (only the iptables half is
faked) behind the production lifecycle's Work endpoint, and stops at a gate
in the middle of env_create or image_pull. The test SIGKILLs it and runs
RecoveryManager with ProductionRecoveryBackend: nothing labelled and no
``<data_root>/<run>/sb`` (spec A5) remain.
Dying before a create is sent leaves a pending create that finds nothing,
which converges only through the production settlement.
Real firewall rules and cgroup.kill of a paused env need root and belong to
the operator check (spec 8, items 5 and 6), which runs this file as root in
root mode (RSI_SANDBOX_ROOT_MODE=1: the real firewall; a paused env is then
recovered through cgroup.kill, with no write after the freeze).
"""

from __future__ import annotations

import hashlib
import io
import multiprocessing
import os
import shutil
import signal
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException

from rsi_harness.runtime.production import ProductionRecoveryBackend
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager, ResourceLease
from rsi_harness.runtime.sandbox_contracts import SandboxOwner
from rsi_harness.runtime.sandbox_env_contracts import (
    ENV_NETWORK_ROLE,
    env_container_labels,
    env_container_name,
    env_network_name,
    env_volume_labels,
    env_volume_name,
    sandbox_object_labels,
    sandbox_spool_root,
)
from rsi_harness.runtime.sandbox_env_docker import (
    CgroupPausedKiller,
    default_paused_killer,
)
from tests.integration.sandbox_support import (
    EmptySnapshotRecovery,
    assert_no_rules,
    sandbox_firewall,
)
from tests.integration.test_sandbox_env_docker import BUSYBOX, remove_labelled

pytestmark = pytest.mark.integration

TASK_ID = "sandbox-env-recovery"
TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
"""


def _service(handle, **values):
    return {
        "image": handle,
        "command": ["sh", "-c", "sleep 600"],
        "cpus": 0.5,
        "memory_mb": 64,
        "pids": 64,
        **values,
    }


# The paused gate's service: the host's uptime (busybox date has no %N)
# every 50 ms, so any run after the freeze would show in the file.
SENTINEL = "while :; do cut -d' ' -f1 /proc/uptime >> /sentinel; sleep 0.05; done"


def uptime() -> float:
    return float(Path("/proc/uptime").read_text().split()[0])


def _three_services(handle):
    return {
        "version": 1,
        "network": "public",
        "lifetime_sec": 600,
        "disk_mb": 128,
        "volumes": {"data": {"seeded": False}},
        "services": {
            "a": _service(
                handle,
                mounts=[{"volume": "data", "target": "/data", "read_only": False}],
            ),
            "b": _service(handle),
            "c": _service(handle),
        },
    }


def _stage(broker, credential):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("seed.txt")
        info.size = 5
        archive.addfile(info, io.BytesIO(b"seed\n"))
    data = buffer.getvalue()
    return broker.stage_put(
        credential,
        None,
        0,
        True,
        hashlib.sha256(data).hexdigest(),
        "stage-1",
        data,
    )


def _pulled(broker, credential, request_id):
    job_id = broker.image_pull(credential, BUSYBOX, "missing", request_id)["job_id"]
    deadline = time.monotonic() + 60
    while True:
        view = broker.job_wait(credential, job_id, 0, 0)
        if view["state"] not in ("queued", "running"):
            break
        assert time.monotonic() < deadline, view
        time.sleep(0.05)
    assert view["state"] == "succeeded", view
    return view["result"]["image"]["handle"]


def _hold_broker_until_killed(
    root: str, managed: str, run_id: str, gate: str, connection
) -> None:
    """The coordinator: one broker, stopped forever at ``gate``; ``managed``
    is the data root, short enough for the endpoint socket path."""
    from rsi_harness.runtime.sandbox import SandboxBroker
    from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool, SandboxJournal
    from rsi_harness.runtime.sandbox_envs import docker_env_runtime
    from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
    from tests.sandbox_helpers import (
        FakeSandboxBackend,
        make_env_grant,
        make_env_task,
    )

    base = Path(root)
    client = docker.from_env(timeout=60)
    store = LeaseStore(base / "leases")

    def stop_here(what):
        connection.send((what, os.getpid()))
        connection.close()
        threading.Event().wait()

    with store.lock(run_id):
        current = ResourceLease(
            run_id=run_id,
            task_id=TASK_ID,
            coordinator_pid=os.getpid(),
            coordinator_started_at=time.time(),
            phase="agent_running",
        )
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

        grant = make_env_grant(base, make_env_task(TASK))
        SandboxAdmissionPool(store).reserve_run(run_id, grant, mutate)
        # As production's prepare_plan: the run-owned root, then the broker.
        (Path(managed) / run_id / "sb").mkdir(mode=0o700, parents=True)
        runtime = docker_env_runtime(
            client,
            sandbox_firewall(client),
            run_id=run_id,
            spool_root=sandbox_spool_root(Path(managed), run_id),
            docker_root=client.info()["DockerRootDir"],
            host=grant.environments.host,
        )
        broker = SandboxBroker(
            grant,
            FakeSandboxBackend(),
            SandboxJournal(mutate),
            time.monotonic,
            envs=runtime,
        )
        lifecycle = SandboxLifecycle()
        lifecycle.configure(broker, Path(managed) / run_id / "sb", run_id, TASK_ID)
        endpoint = lifecycle.prepare_work()
        credential = endpoint.environment["RSI_SANDBOX_TOKEN"]
        lifecycle.activate_work(time.monotonic() + 600)
        handle = _pulled(broker, credential, "pull-1")
        _stage(broker, credential)

        if gate == "pull":
            # A ready env keeps running while a second pull is cut short.
            spec = {
                "version": 1,
                "network": "none",
                "lifetime_sec": 600,
                "disk_mb": 128,
                "services": {"main": _service(handle)},
            }
            env_id = broker.env_create(credential, spec, "env-1")["env_id"]
            broker.env_start(credential, env_id, 60, "start-1")
            deadline = time.monotonic() + 60
            while broker.env_status(credential, env_id)["state"] != "ready":
                assert time.monotonic() < deadline
                time.sleep(0.1)
            runtime.images.pull = lambda *args, **kwargs: stop_here("pull")
            broker.image_pull(credential, BUSYBOX, "always", "pull-2")
            threading.Event().wait()
        if gate == "paused":
            # Submit froze the whole Work group; the coordinator dies frozen.
            spec = {
                "version": 1,
                "network": "none",
                "lifetime_sec": 600,
                "disk_mb": 128,
                "services": {"main": _service(handle, command=["sh", "-c", SENTINEL])},
            }
            env_id = broker.env_create(credential, spec, "env-1")["env_id"]
            broker.env_start(credential, env_id, 60, "start-1")
            deadline = time.monotonic() + 60
            while broker.env_status(credential, env_id)["state"] != "ready":
                assert time.monotonic() < deadline
                time.sleep(0.1)
            time.sleep(0.5)
            broker.freeze_work()
            stop_here("paused")
        if gate == "bridge":
            # client.networks is a fresh collection per access; its API is not.
            create_network = client.api.create_network

            def gated_network(*args, **kwargs):
                create_network(*args, **kwargs)
                stop_here("bridge")

            client.api.create_network = gated_network
        if gate == "container":
            create_container = client.api.create_container_from_config

            def gated_container(config, name=None):
                create_container(config, name=name)
                stop_here("container")

            client.api.create_container_from_config = gated_container
        if gate == "before-container":
            client.api.create_container_from_config = lambda config, name=None: (
                stop_here("before-container")
            )
        broker.env_create(credential, _three_services(handle), "env-1")
        connection.send(("create finished", os.getpid()))


def _late_env(client, run_id, image):
    """Objects of an env whose create finished after its journal record left:
    exact labels of the run, but no record names them."""
    env_id = "e" + uuid.uuid4().hex
    owner = SandboxOwner(run_id=run_id, task_id=TASK_ID, phase="work")
    client.volumes.create(
        name=env_volume_name(env_id, 0),
        driver="local",
        labels=env_volume_labels(owner, env_id),
    )
    client.networks.create(
        env_network_name(env_id),
        driver="bridge",
        internal=True,
        labels=sandbox_object_labels(owner, ENV_NETWORK_ROLE, {"sandbox-env": env_id}),
    )
    client.containers.create(
        BUSYBOX,
        ["sleep", "600"],
        name=env_container_name(env_id, 0),
        labels=env_container_labels(owner, env_id, "main", image),
        runtime="runc",
        network_mode="none",
    )


class SentinelKiller:
    """The recovery's own paused killer (cgroup.kill as root), which also
    reads the frozen service's /sentinel once its proof holds: every task
    is gone, the container not yet removed."""

    def __init__(self, client):
        self.client = client
        self.inner = default_paused_killer(client.api)
        self.written = {}

    def kill(self, container_id, attrs):
        settled = self.inner.kill(container_id, attrs)

        def proof():
            if settled is not None and not settled():
                return False
            if container_id not in self.written:
                stream, _ = self.client.api.get_archive(container_id, "/sentinel")
                data = io.BytesIO(b"".join(stream))
                with tarfile.open(fileobj=data) as archive:
                    self.written[container_id] = archive.extractfile("sentinel").read()
            return True

        return proof


class SettlementRecording(ProductionRecoveryBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.settlements = []

    def settle_sandbox_creates(self, rule_ids):
        self.settlements.append(rule_ids)
        return super().settle_sandbox_creates(rule_ids)


def _labelled(client, run_id):
    filters = {"label": f"rsi-harness.run-id={run_id}"}
    return (
        client.containers.list(all=True, filters=filters),
        client.volumes.list(filters=filters),
        client.networks.list(filters=filters),
    )


@pytest.fixture
def client():
    try:
        client = docker.from_env(timeout=60)
        client.ping()
        client.images.get(BUSYBOX)
    except (DockerException, OSError) as error:
        message = f"Docker/{BUSYBOX} capability unavailable: {error}"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    try:
        yield client
    finally:
        client.close()


@pytest.mark.parametrize(
    "gate", ["bridge", "container", "before-container", "pull", "paused"]
)
def test_kill_9_mid_operation_then_recover_leaves_nothing_labelled(
    client, tmp_path, gate
):
    run_id = f"m6-recover-{uuid.uuid4().hex[:12]}"
    busybox = client.images.get(BUSYBOX).id
    store = LeaseStore(tmp_path / "leases")
    # A short data root: the endpoint socket path must fit 107 bytes.
    managed = Path(tempfile.mkdtemp(prefix="rsi-m6-"))
    spool = sandbox_spool_root(managed, run_id)
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    coordinator = context.Process(
        target=_hold_broker_until_killed,
        args=(str(tmp_path), str(managed), run_id, gate, sender),
    )
    coordinator.start()
    sender.close()
    try:
        assert receiver.poll(120), f"no gate reached; exitcode={coordinator.exitcode}"
        reached, pid = receiver.recv()
        frozen_by = uptime()
        assert (reached, pid) == (gate, coordinator.pid)
        os.kill(coordinator.pid, signal.SIGKILL)
        coordinator.join(10)
        assert coordinator.exitcode == -signal.SIGKILL

        crashed = store.read(run_id)
        containers, volumes, networks = _labelled(client, run_id)
        assert spool.is_dir()
        [work] = [path for path in spool.parent.iterdir() if path != spool]
        assert (work / "s").is_socket()
        assert crashed.sandbox_reservation is not None
        if gate == "paused":
            (env,) = crashed.sandbox_envs
            assert env.state == "paused"
            assert [item.status for item in containers] == ["paused"]
        elif gate == "pull":
            assert [image.state for image in crashed.sandbox_images] == [
                "present",
                "planned",
            ]
            (env,) = crashed.sandbox_envs
            assert env.state == "ready"
            assert [item.status for item in containers] == ["running"]
        else:
            (env,) = crashed.sandbox_envs
            assert (env.state, env.pending_mutation) == ("planned", True)
            assert len(networks) == 1
            if gate in ("container", "before-container"):
                # One of three created (or none), none journaled; volume and
                # bridge are.
                assert len(containers) == (gate == "container")
                assert env.volumes[0].created
                assert env.network_id == networks[0].id
                assert all(item.container_id is None for item in env.services)
            else:
                # The bridge exists, but its create never answered.
                assert env.network_id is None
                assert (containers, volumes) == ([], [])
                _late_env(client, run_id, env.services[0].image)

        rule_ids = [env.rule_id for env in crashed.sandbox_envs if env.rule_id]
        firewall = sandbox_firewall(client)
        killer = SentinelKiller(client)
        backend = SettlementRecording(
            client, EmptySnapshotRecovery(), firewall, paused_killer=killer
        )
        recovered = RecoveryManager(
            store=store, backend=backend, managed_root=managed
        ).recover(run_id)

        assert recovered == (run_id,)
        assert not spool.parent.exists()
        # Only a pending create that found nothing waits for it to settle.
        assert backend.settlements == ([()] if gate == "before-container" else [])
        assert _labelled(client, run_id) == ([], [], [])
        final = store.read(run_id)
        assert (final.sandbox_envs, final.sandbox_images) == ((), ())
        assert not final.recovery_required
        assert final.sandbox_reservation is None
        assert not spool.exists()
        assert_no_rules(firewall, run_id, rule_ids=rule_ids)
        if gate == "paused":
            # The frozen service was killed where it stood, never thawed: as
            # root (cgroup.kill) no write follows the freeze. Non-root's
            # docker kill thaws the task first, so only the kill is proven.
            [written] = [raw.split() for raw in killer.written.values()]
            assert written
            if isinstance(killer.inner, CgroupPausedKiller):
                assert float(written[-1]) <= frozen_by
        # A pulled image is the host's cache: never removed (S9).
        assert client.images.get(BUSYBOX).id == busybox
    finally:
        receiver.close()
        if coordinator.is_alive():
            coordinator.kill()
            coordinator.join(10)
        # Never leave an object behind, whatever failed above.
        remove_labelled(client, {"label": f"rsi-harness.run-id={run_id}"})
        shutil.rmtree(managed, ignore_errors=True)
