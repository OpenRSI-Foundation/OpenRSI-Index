"""Real Docker probes for the effective managed-sandbox resource envelope."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest

from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_contracts import SandboxOwner
from tests.integration.sandbox_support import (
    assert_no_sandbox_resources,
    create_broker,
    remove_exact_containers,
    require_sandbox_authority,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def live_sandbox(tmp_path: Path):
    client, image = require_sandbox_authority()
    run_id = f"sandbox-resources-{uuid4().hex}"
    task_id = "sandbox-resources"
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
    child = broker.create(credentials.credential, "offline", 20.0, "create-1")
    exact_ids = {child.container_id}
    try:
        yield client, broker, credentials, child, store, run_id, exact_ids
    finally:
        try:
            broker.close()
        finally:
            remove_exact_containers(client, {item for item in exact_ids if item})
            client.close()


def test_effective_cgroup_and_tmpfs_limits_match_durable_profile(live_sandbox):
    client, broker, credentials, child, store, run_id, _exact_ids = live_sandbox
    attrs = client.containers.get(child.container_id).attrs
    host = attrs["HostConfig"]

    result = broker.execute(
        credentials.credential,
        child.child_id,
        [
            "python3",
            "-c",
            """
import json, os
def text(path):
    with open(path, encoding="utf-8") as source:
        return source.read().strip()
def size(path):
    value = os.statvfs(path)
    return value.f_frsize * value.f_blocks
print(json.dumps({
    "cpu": text("/sys/fs/cgroup/cpu.max"),
    "memory": text("/sys/fs/cgroup/memory.max"),
    "pids": text("/sys/fs/cgroup/pids.max"),
    "workspace": size("/workspace"),
    "tmp": size("/tmp"),
    "shm": size("/dev/shm"),
}))
""",
        ],
        "/workspace",
        {},
        5.0,
    )

    assert result.exit_code == 0, result.stderr
    effective = json.loads(result.stdout)
    quota, period = (int(value) for value in effective["cpu"].split())
    assert quota / period == 1
    assert effective["memory"] == str(128 * 1024**2)
    assert effective["pids"] == "16"
    assert effective["workspace"] == 16 * 1024**2
    assert effective["tmp"] == 8 * 1024**2
    assert effective["shm"] == 8 * 1024**2
    assert host["NanoCpus"] == 1_000_000_000
    assert host["Memory"] == host["MemorySwap"] == 128 * 1024**2
    assert host["PidsLimit"] == 16
    assert host["ReadonlyRootfs"] is True

    broker.destroy(credentials.credential, child.child_id)
    assert_no_sandbox_resources(client, store, run_id)


def test_real_memory_excess_kills_allocator_and_records_cgroup_oom(live_sandbox):
    """Removing the memory cap must not let a bounded 192 MiB allocation pass."""
    client, broker, credentials, child, store, run_id, _exact_ids = live_sandbox
    result = broker.execute(
        credentials.credential,
        child.child_id,
        [
            "python3",
            "-c",
            """
import json, subprocess, sys
from pathlib import Path
def events():
    return {
        key: int(value)
        for key, value in (
            line.split()
            for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        )
    }
before = events()
allocator = subprocess.run(
    [sys.executable, "-c", "chunks = [bytearray(1024 * 1024) for _ in range(192)]"],
    timeout=5,
)
print(json.dumps({
    "returncode": allocator.returncode,
    "before": before,
    "after": events(),
}))
""",
        ],
        "/workspace",
        {},
        8.0,
    )

    assert result.exit_code == 0, result.stderr
    assert not result.timed_out
    evidence = json.loads(result.stdout)
    assert evidence["returncode"] == -9
    assert evidence["after"]["oom"] > evidence["before"]["oom"]
    assert evidence["after"]["oom_kill"] > evidence["before"]["oom_kill"]

    broker.destroy(credentials.credential, child.child_id)
    assert_no_sandbox_resources(client, store, run_id)


def test_real_pid_excess_denies_fork_and_records_cgroup_limit(live_sandbox):
    """Removing the PID cap must not allow all 24 bounded fork attempts."""
    client, broker, credentials, child, store, run_id, _exact_ids = live_sandbox
    result = broker.execute(
        credentials.credential,
        child.child_id,
        [
            "python3",
            "-c",
            """
import json, os
from pathlib import Path
def limit_events():
    return dict(
        line.split()
        for line in Path("/sys/fs/cgroup/pids.events").read_text().splitlines()
    )["max"]
before = int(limit_events())
read_fd, write_fd = os.pipe()
children = []
denied_errno = None
try:
    for _ in range(24):
        try:
            pid = os.fork()
        except OSError as error:
            denied_errno = error.errno
            break
        if pid == 0:
            os.close(write_fd)
            os.read(read_fd, 1)
            os._exit(0)
        children.append(pid)
    current = int(Path("/sys/fs/cgroup/pids.current").read_text())
    after = int(limit_events())
finally:
    os.close(write_fd)
    os.close(read_fd)
    for pid in children:
        os.waitpid(pid, 0)
print(json.dumps({
    "created": len(children),
    "errno": denied_errno,
    "current": current,
    "before": before,
    "after": after,
}))
""",
        ],
        "/workspace",
        {},
        8.0,
    )

    assert result.exit_code == 0, result.stderr
    assert not result.timed_out
    evidence = json.loads(result.stdout)
    assert 0 < evidence["created"] < 24
    assert evidence["errno"] == 11  # Linux EAGAIN from the cgroup PID ceiling.
    assert evidence["current"] == 16
    assert evidence["after"] > evidence["before"]

    broker.destroy(credentials.credential, child.child_id)
    assert_no_sandbox_resources(client, store, run_id)


def test_real_child_cannot_reach_host_peer_gpu_or_host_sockets_and_scratch_is_bounded(
    live_sandbox,
):
    client, broker, credentials, child, store, run_id, exact_ids = live_sandbox
    labels = {
        "rsi-harness.run-id": run_id,
        "rsi-harness.task-id": "sandbox-resources",
        "rsi-harness.role": "fixture-peer",
    }
    peer = client.containers.create(
        image=client.images.get("python:3.12-slim-bookworm").id,
        command=["python3", "-m", "http.server", "39123"],
        network_mode="bridge",
        labels=labels,
        detach=True,
    )
    exact_ids.add(peer.id)
    peer.start()
    peer.reload()
    bridge = peer.attrs["NetworkSettings"]["Networks"]["bridge"]
    peer_ip = bridge["IPAddress"]
    host_ip = bridge["Gateway"]
    peer_deadline = time.monotonic() + 5.0
    while True:
        try:
            with socket.create_connection((peer_ip, 39123), timeout=0.25):
                break
        except OSError:
            if time.monotonic() >= peer_deadline:
                pytest.fail("controlled peer did not become reachable from the host")
            time.sleep(0.05)

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", 0))
    listener.listen()
    listener.settimeout(0.1)
    stop_listener = threading.Event()

    def accept_until_stopped() -> None:
        while not stop_listener.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if stop_listener.is_set():
                    return
                raise
            connection.close()

    listener_thread = threading.Thread(target=accept_until_stopped, daemon=True)
    listener_thread.start()
    with socket.create_connection((host_ip, listener.getsockname()[1]), timeout=1):
        pass
    try:
        result = broker.execute(
            credentials.credential,
            child.child_id,
            [
                "python3",
                "-c",
                """
import glob, json, os, socket, sys
def reachable(host, port):
    connection = socket.socket()
    connection.settimeout(0.25)
    try:
        connection.connect((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        connection.close()
root_write = True
try:
    with open("/sandbox-root-write", "w", encoding="utf-8") as target:
        target.write("forbidden")
except OSError:
    root_write = False
written = 0
scratch_limited = False
try:
    with open("/workspace/fill", "wb") as target:
        for _ in range(20):
            target.write(b"x" * 1024 * 1024)
            target.flush()
            written += 1024 * 1024
except OSError:
    scratch_limited = True
finally:
    try:
        os.unlink("/workspace/fill")
    except FileNotFoundError:
        pass
print(json.dumps({
    "peer": reachable(sys.argv[1], 39123),
    "host": reachable(sys.argv[2], sys.argv[3]),
    "root_write": root_write,
    "scratch_limited": scratch_limited,
    "written": written,
    "gpu": glob.glob("/dev/nvidia*"),
    "docker_socket": (
        os.path.exists("/var/run/docker.sock")
        or os.path.exists("/run/docker.sock")
    ),
    "broker_socket": os.path.exists("/run/rsi-harness/sandbox"),
    "nvidia_env": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
}))
""",
                peer_ip,
                host_ip,
                str(listener.getsockname()[1]),
            ],
            "/workspace",
            {},
            8.0,
        )
    finally:
        stop_listener.set()
        listener.close()
        listener_thread.join(timeout=1)
        peer.remove(force=True, v=True)
        exact_ids.discard(peer.id)

    assert result.exit_code == 0, result.stderr
    confinement = json.loads(result.stdout)
    written = confinement.pop("written")
    assert 15 * 1024**2 <= written <= 16 * 1024**2
    assert confinement == {
        "peer": False,
        "host": False,
        "root_write": False,
        "scratch_limited": True,
        "gpu": [],
        "docker_socket": False,
        "broker_socket": False,
        "nvidia_env": "void",
    }
    attrs = client.containers.get(child.container_id).attrs
    host = attrs["HostConfig"]
    assert host.get("Devices") in (None, [])
    assert host.get("DeviceRequests") in (None, [])
    assert host.get("Binds") in (None, [])
    assert host.get("VolumesFrom") in (None, [])
    assert host["NetworkMode"] == "none"
    assert all(
        mount.get("Type") == "tmpfs"
        and mount.get("Destination") in {"/workspace", "/tmp", "/dev/shm"}
        for mount in attrs.get("Mounts", ())
    )

    broker.destroy(credentials.credential, child.child_id)
    assert_no_sandbox_resources(client, store, run_id)
