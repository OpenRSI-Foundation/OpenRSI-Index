"""Small real-Docker confinement probe using a cached image, no GPU/firewall."""

import os
import time
import uuid

import docker
import pytest

from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend
from tests.runtime.test_sandbox_budget import make_child
from tests.sandbox_helpers import make_profile

pytestmark = pytest.mark.integration


@pytest.fixture
def live_child():
    client = docker.from_env(timeout=5)
    image = client.images.get("python:3.12-slim-bookworm")
    profile = make_profile().model_copy(update={"image": image.id})
    lease = make_child(child_id=uuid.uuid4().hex, image_id=image.id)
    backend = SandboxDockerBackend(client)
    identity = backend.create(lease, profile)
    lease = lease.model_copy(update={"container_id": identity})
    try:
        backend.start(lease)
        yield backend, lease
    finally:
        backend.terminate(lease)
        backend.remove(lease)
        client.close()


def test_real_exec_keeps_state_separates_stderr_and_preserves_nonzero(live_child):
    backend, lease = live_child
    first = backend.execute(
        lease,
        ["/bin/sh", "-c", "echo saved > marker; printf out; printf err >&2; exit 3"],
        "/workspace",
        {},
        time.monotonic() + 5,
        1024,
    )
    assert first.exit_code == 3
    assert first.stdout == "out"
    assert first.stderr == "err"
    assert not first.timed_out
    assert not first.oom_killed
    second = backend.execute(
        lease, ["/bin/cat", "marker"], "/workspace", {}, time.monotonic() + 5, 1024
    )
    assert second.stdout == "saved\n"
    ordinary_137 = backend.execute(
        lease,
        ["/bin/sh", "-c", "exit 137"],
        "/workspace",
        {},
        time.monotonic() + 5,
        1024,
    )
    assert ordinary_137.exit_code == 137
    assert not ordinary_137.oom_killed


def test_real_exec_timeout_terminates_descendant_holding_stdout(live_child):
    backend, lease = live_child
    start = time.monotonic()
    result = backend.execute(
        lease,
        ["/bin/sh", "-c", "sleep 60 & exit 0"],
        "/workspace",
        {},
        start + 0.3,
        1024,
    )
    assert result.timed_out
    assert time.monotonic() - start < 8
    assert not backend.inspect(lease)["State"]["Running"]


def test_real_exec_output_excess_stops_entire_child(live_child):
    backend, lease = live_child
    result = backend.execute(
        lease,
        ["python3", "-c", "import os; os.write(1,b'x'*1000000)"],
        "/workspace",
        {},
        time.monotonic() + 5,
        128,
    )
    assert result.output_limited
    assert result.truncated
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= 128
    assert not backend.inspect(lease)["State"]["Running"]


def test_real_transfer_uses_live_tmpfs_not_docker_archive_view(live_child):
    from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry

    backend, lease = live_child
    backend.execute(
        lease,
        ["/bin/sh", "-c", "printf live > marker; mkdir empty"],
        "/workspace",
        {},
        time.monotonic() + 5,
        1024,
    )
    records = backend.download(
        lease, "/workspace", ("marker", "empty"), time.monotonic() + 5
    )
    assert {record.path: record.data for record in records} == {
        "marker": b"live",
        "empty": b"",
    }
    backend.upload(
        lease,
        "/workspace",
        (
            SandboxBundleEntry(
                path="uploaded", kind="file", mode=0o644, data=b"actual tmpfs"
            ),
        ),
        time.monotonic() + 5,
    )
    result = backend.execute(
        lease,
        ["/bin/cat", "/workspace/uploaded"],
        "/workspace",
        {},
        time.monotonic() + 5,
        1024,
    )
    assert result.stdout == "actual tmpfs"


def test_transfer_never_unpauses_child_or_follows_candidate_symlinks(live_child):
    from rsi_harness.runtime.sandbox_contracts import SandboxError

    backend, lease = live_child
    backend.execute(
        lease,
        ["/bin/ln", "-s", "/etc/passwd", "/workspace/link"],
        "/workspace",
        {},
        time.monotonic() + 5,
        1024,
    )
    with pytest.raises(SandboxError):
        backend.download(lease, "/workspace", ("link",), time.monotonic() + 5)
    backend.pause(lease)
    try:
        with pytest.raises(SandboxError, match="paused"):
            backend.download(lease, "/workspace", (".",), time.monotonic() + 5)
        assert backend.inspect(lease)["State"]["Paused"]
    finally:
        backend.resume(lease)


def test_real_scratch_allows_uploaded_executable_scripts(live_child):
    from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry

    backend, lease = live_child
    backend.upload(
        lease,
        "/workspace",
        (
            SandboxBundleEntry(
                path="probe.sh",
                kind="file",
                mode=0o755,
                data=b"#!/bin/sh\nprintf executable",
            ),
        ),
        time.monotonic() + 5,
    )
    result = backend.execute(
        lease, ["/workspace/probe.sh"], "/workspace", {}, time.monotonic() + 5, 1024
    )
    assert result.exit_code == 0
    assert result.stdout == "executable"


def test_real_cpu_child_lifecycle_and_readonly_generated_files():
    try:
        client = docker.from_env(timeout=5)
        client.ping()
        image = client.images.get("python:3.12-slim-bookworm")
    except docker.errors.DockerException as error:
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(f"required Docker/cached fixture image unavailable: {error}")
        pytest.skip(f"Docker/cached fixture image unavailable: {error}")
    profile = make_profile().model_copy(update={"image": image.id})
    lease = make_child(child_id=uuid.uuid4().hex, image_id=image.id)
    backend = SandboxDockerBackend(client)
    identity = backend.create(lease, profile)
    lease = lease.model_copy(update={"container_id": identity})
    try:
        backend.start(lease)
        attrs = backend.inspect(lease)
        assert attrs["HostConfig"]["NetworkMode"] == "none"
        assert attrs["HostConfig"]["ReadonlyRootfs"] is True
        result = client.containers.get(identity).exec_run(
            [
                "/bin/sh",
                "-c",
                "for f in /etc/hosts /etc/hostname /etc/resolv.conf; "
                'do if echo sandbox-probe >> "$f" 2>/dev/null; then exit 1; fi; done; '
                "test ! -e /var/run/docker.sock && test ! -e /dev/nvidia0",
            ]
        )
        assert result.exit_code == 0, result.output
        backend.pause(lease)
        assert backend.inspect(lease)["State"]["Paused"]
        backend.resume(lease)
    finally:
        # This fixture runs only trusted probes and idle sleep; it owns this ID.
        try:
            if backend.inspect(lease)["State"].get("Paused"):
                backend.resume(lease)
            backend.terminate(lease)
            backend.remove(lease)
        finally:
            client.close()
