"""Owned endpoint lifetime and the parent/child resume admission boundary."""

import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from rsi_harness.runtime.sandbox_contracts import SandboxError
from tests.runtime.test_sandbox import kit as kit
from tests.runtime.test_sandbox_server import _connect, _wait_for, _wire_request


@pytest.fixture
def lifecycle(kit):
    from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle

    with tempfile.TemporaryDirectory(prefix="rsi-phase-") as root:
        lifecycle = SandboxLifecycle()
        lifecycle.configure(kit[0], Path(root), "run-1", "task")
        try:
            yield lifecycle
        finally:
            # These are trusted fake children; restore any intentionally paused fixture.
            for state in kit[1].states.values():
                state["Paused"] = False
            lifecycle.close()


def test_work_endpoint_inactive_until_parent_exec_and_credential_not_file(
    lifecycle, kit
):
    endpoint = lifecycle.prepare_work()
    assert endpoint.mount.read_only
    assert endpoint.directory.stat().st_mode & 0o777 == 0o755
    assert (endpoint.directory / "s").stat().st_mode & 0o777 == 0o666
    assert lifecycle.root.stat().st_mode & 0o777 == 0o700
    assert endpoint.mount.target.as_posix() == "/run/rsi-harness/sandbox"
    token = endpoint.environment["RSI_SANDBOX_TOKEN"]
    assert token.encode() not in (endpoint.directory / "rsi-sandbox").read_bytes()
    assert {p.name for p in endpoint.directory.iterdir()} == {"s", "rsi-sandbox", "py"}
    with pytest.raises(SandboxError, match="busy"):
        kit[0].create(token, "offline", 30, "one")
    lifecycle.activate_work(1000)
    assert kit[0].create(token, "offline", 30, "one").state == "running"


def test_resume_keeps_admission_closed_until_parent_is_running(lifecycle, kit):
    endpoint = lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    token = endpoint.environment["RSI_SANDBOX_TOKEN"]
    child = kit[0].create(token, "offline", 30, "one")
    lifecycle.freeze_work()
    lifecycle.resume_work()
    assert not kit[1].states[child.child_id]["Paused"]
    with pytest.raises(SandboxError, match="busy"):
        kit[0].create(token, "offline", 30, "two")
    lifecycle.reopen_work()
    kit[0].create(token, "offline", 30, "two")


def test_each_judge_endpoint_and_token_are_fresh_and_removed(lifecycle, kit):
    lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    lifecycle.freeze_work()
    first = lifecycle.prepare_judge("r1")
    assert lifecycle.prepare_judge("r1") is first
    lifecycle.activate_judge(500)
    token = first.environment["RSI_SANDBOX_TOKEN"]
    kit[0].create(token, "offline", 30, "one")
    lifecycle.close_judge()
    assert not first.directory.exists()
    second = lifecycle.prepare_judge("r2")
    assert first.directory != second.directory
    assert token != second.environment["RSI_SANDBOX_TOKEN"]
    with pytest.raises(SandboxError, match="permission"):
        kit[0].capabilities(token)


def test_cancel_closes_unbound_lifecycle_and_forbids_resume():
    from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle

    lifecycle = SandboxLifecycle()
    lifecycle.cancel_run()
    assert not lifecycle.can_resume
    lifecycle.close()


@pytest.mark.parametrize("interruption", ["cancel", "deadline"])
def test_parent_resume_checks_after_daemon_inspection(
    lifecycle, kit, tmp_path, interruption
):
    from rsi_harness.errors import InfrastructureError
    from rsi_harness.models import ContainerRef
    from tests.fakes import FakeDockerClient, FakeDockerContainer
    from tests.runtime.test_docker import make_runtime

    lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    lifecycle.freeze_work()
    lifecycle.resume_work()
    client = FakeDockerClient()
    parent = FakeDockerContainer("work", pause_states=(True,))
    client.containers.by_id["work"] = parent
    original = parent.reload

    def delayed_inspection():
        original()
        if interruption == "cancel":
            lifecycle.cancel_run()
        else:
            kit[2].now = 1001

    parent.reload = delayed_inspection
    runtime = make_runtime(client, tmp_path)
    with pytest.raises(InfrastructureError, match="recovery"):
        lifecycle.resume_parent(runtime, ContainerRef(container_id="work", role="work"))
    assert "unpause" not in parent.events


def test_parent_resume_uses_bounded_control_transport(lifecycle, tmp_path):
    from rsi_harness.models import ContainerRef
    from tests.fakes import FakeDockerClient, FakeDockerContainer
    from tests.runtime.test_docker import make_runtime

    lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    lifecycle.freeze_work()
    slow, bounded = FakeDockerClient(), FakeDockerClient()
    bounded_parent = FakeDockerContainer("work", pause_states=(True,))
    bounded.containers.by_id["work"] = bounded_parent
    lifecycle.own_transport(bounded)
    runtime = make_runtime(slow, tmp_path)
    lifecycle.resume_parent(runtime, ContainerRef(container_id="work", role="work"))
    assert "unpause" in bounded_parent.events
    assert "work" not in slow.containers.by_id


@pytest.mark.parametrize("read_responses", [False, True])
def test_immediate_judge_handoff_reclaims_completed_work_transports(
    lifecycle, kit, monkeypatch, read_responses
):
    from rsi_harness.integrations.sandbox_client import ProtocolError, SandboxClient
    from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry

    work = lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    token = work.environment["RSI_SANDBOX_TOKEN"]
    client = SandboxClient(work.server.path, token)
    child = kit[0].create(token, "offline", 30, "download")
    monkeypatch.setattr(
        kit[1],
        "download",
        lambda *args, **kwargs: (
            SandboxBundleEntry(
                path="out", kind="file", mode=0o644, data=b"x" * 1024**2
            ),
        ),
    )
    metadata = {
        "child_id": child.child_id,
        "root": "/workspace",
        "paths": ["out"],
        "timeout_sec": 1,
    }
    connections = []
    try:
        for count in range(20):
            if read_responses:
                assert len(client._request("download", metadata)[0]["data"]) == 1024**2
            else:
                connection = _connect(work.server)
                connections.append(connection)
                connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                connection.sendall(_wire_request("download", token, metadata))
                _wait_for(
                    lambda: (
                        sum(
                            p.transport.get_write_buffer_size() > 0
                            for p in work.server._server.server_state.connections
                        )
                        == count + 1
                    )
                )
        slots = lifecycle._slots
        expected_places = 20 if read_responses else 0
        _wait_for(
            lambda: (
                slots._places._value == expected_places and slots._active._value == 4
            )
        )
        assert kit[0].status(token, child.child_id)["inflight"] is None
        if not read_responses:
            assert slots._connections._value == 0
        started = time.monotonic()
        lifecycle.freeze_work()
        assert slots._places._value == 20
        judge = lifecycle.prepare_judge("r1")
        lifecycle.activate_judge(500)
        judge_client = SandboxClient(
            judge.server.path, judge.environment["RSI_SANDBOX_TOKEN"]
        )
        assert judge_client.capabilities()["version"] == 1
        assert time.monotonic() - started < 2  # No five-second flush wait.
        # Late Work peers cannot reclaim Judge's connection reservation.
        for _ in range(20):
            with _connect(work.server) as late:
                assert late.recv(1) == b""
        assert judge_client.capabilities()["version"] == 1
        lifecycle.close_judge()
        lifecycle.resume_work()
        with pytest.raises(ProtocolError):
            client.capabilities()
        lifecycle.reopen_work()
        assert client.capabilities()["version"] == 1
    finally:
        for connection in connections:
            connection.close()


def test_judge_only_lifecycle_needs_no_work_transport(lifecycle, kit):
    from rsi_harness.integrations.sandbox_client import SandboxClient

    kit[0].grant = kit[0].grant.model_copy(update={"work": None})
    assert lifecycle.prepare_work() is None
    lifecycle.activate_work(1000)
    lifecycle.freeze_work()
    judge = lifecycle.prepare_judge("r1")
    lifecycle.activate_judge(500)
    assert (
        SandboxClient(
            judge.server.path, judge.environment["RSI_SANDBOX_TOKEN"]
        ).capabilities()["version"]
        == 1
    )
    lifecycle.close_judge()
    lifecycle.resume_work()
    lifecycle.reopen_work()


def test_busy_freeze_preserves_work_transport_and_mutation_ownership(lifecycle, kit):
    from rsi_harness.errors import RetryableSubmissionError
    from rsi_harness.integrations.sandbox_client import SandboxClient

    work = lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    token = work.environment["RSI_SANDBOX_TOKEN"]
    child = kit[0].create(token, "offline", 30, "busy")
    entered, release = threading.Event(), threading.Event()

    def block(lease):
        entered.set()
        assert release.wait(2)

    kit[1].hooks["execute"] = block
    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(
            kit[0].execute, token, child.child_id, ["true"], "/workspace", {}, 10
        )
        try:
            assert entered.wait(1)
            with pytest.raises(RetryableSubmissionError):
                lifecycle.freeze_work()
            assert SandboxClient(work.server.path, token).capabilities()["version"] == 1
            assert kit[0].status(token, child.child_id)["inflight"] == "execute"
        finally:
            release.set()
        pending.result()
    lifecycle.freeze_work()
    assert not work.server._server.server_state.connections


@pytest.mark.parametrize("failure", ["cancel", "deadline", "pause"])
def test_failed_phase_transition_cannot_reopen_work(lifecycle, kit, failure):
    from rsi_harness.errors import InfrastructureError
    from rsi_harness.integrations.sandbox_client import ProtocolError, SandboxClient

    endpoint = lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    token = endpoint.environment["RSI_SANDBOX_TOKEN"]
    kit[0].create(token, "offline", 30, "one")
    if failure == "pause":
        failed = False

        def reject(lease):
            nonlocal failed
            if not failed:
                failed = True
                raise InfrastructureError("pause failed")

        kit[1].hooks["pause"] = reject
        with pytest.raises(InfrastructureError):
            lifecycle.freeze_work()
    else:
        lifecycle.freeze_work()
        if failure == "cancel":
            for state in kit[1].states.values():
                state["Paused"] = False
            lifecycle.cancel_run()
        else:
            kit[2].now = 1001
    with pytest.raises(InfrastructureError):
        lifecycle.reopen_work()
    with pytest.raises(ProtocolError):
        SandboxClient(endpoint.server.path, token).create("offline", 30, "denied")


@pytest.mark.parametrize("end", ["deadline", "cancel_work", "cancel_run", "recovery"])
def test_only_expired_or_retired_work_ends_normally(lifecycle, kit, end):
    from rsi_harness.integrations.sandbox_client import ProtocolError, SandboxClient

    endpoint = lifecycle.prepare_work()
    lifecycle.activate_work(1000)
    client = SandboxClient(
        endpoint.server.path, endpoint.environment["RSI_SANDBOX_TOKEN"]
    )
    assert client.capabilities()["version"] == 1
    assert not lifecycle.work_ended_normally
    if end == "cancel_work":
        lifecycle.cancel_work()
    else:
        kit[2].now = 1000
    if end == "cancel_run":
        lifecycle.cancel_run()
    elif end == "recovery":
        kit[0]._fail_closed()
    assert not lifecycle.can_resume
    assert lifecycle.work_ended_normally is (end in ("deadline", "cancel_work"))
    # Every end, including retirement before the deadline, revokes the Work
    # credential at the broker itself, not only in the lifecycle's view.
    with pytest.raises(ProtocolError):
        client.capabilities()


def test_retired_work_cannot_resume_while_judge_endpoint_serves(lifecycle, kit):
    """Retirement must hold even for a Judge-only grant with no Work session."""
    from rsi_harness.integrations.sandbox_client import SandboxClient

    kit[0].grant = kit[0].grant.model_copy(update={"work": None})
    assert lifecycle.prepare_work() is None
    lifecycle.activate_work(1000)
    lifecycle.freeze_work()
    judge = lifecycle.prepare_judge("r1")
    lifecycle.activate_judge(500)
    lifecycle.cancel_work()
    assert not lifecycle.can_resume
    assert lifecycle.work_ended_normally
    client = SandboxClient(judge.server.path, judge.environment["RSI_SANDBOX_TOKEN"])
    assert client.capabilities()["version"] == 1
    lifecycle.close_judge()


ENDPOINT_TREE = {
    "s",
    "rsi-sandbox",
    "py",
    "py/rsi_sandbox_client.py",
    "py/rsi_sandbox_compose.py",
    "py/rsi_sandbox_harbor.py",
}


def _tree(directory):
    return {
        str(path.relative_to(directory)): path.lstat() for path in directory.rglob("*")
    }


@pytest.mark.parametrize("phase", ["work", "judge"])
def test_endpoint_holds_exactly_the_socket_cli_and_plugin_modules(
    lifecycle, kit, phase
):
    """spec 6: {s, rsi-sandbox, py/*}, bound read-only; no file is writable
    by the parent UID or carries the credential."""
    from rsi_harness.integrations import sandbox_client
    from rsi_harness.runtime.sandbox_lifecycle import ENDPOINT_MODULES

    if phase == "work":
        endpoint = lifecycle.prepare_work()
    else:
        lifecycle.prepare_work()
        lifecycle.activate_work(1000)
        lifecycle.freeze_work()
        endpoint = lifecycle.prepare_judge("r1")
    tree = _tree(endpoint.directory)
    assert set(tree) == ENDPOINT_TREE
    assert endpoint.mount.read_only
    assert endpoint.mount.source == endpoint.directory
    assert endpoint.environment["RSI_SANDBOX_PYTHONPATH"] == (
        "/run/rsi-harness/sandbox/py"
    )
    token = endpoint.environment["RSI_SANDBOX_TOKEN"].encode()
    for name, info in tree.items():
        if name == "s":
            continue
        assert info.st_uid == os.getuid()
        assert not info.st_mode & 0o022, name
        if name != "py" and name != "rsi-sandbox":
            assert info.st_mode & 0o777 == 0o644, name
        assert token not in (
            b"" if name == "py" else (endpoint.directory / name).read_bytes()
        )
    assert tree["py"].st_mode & 0o777 == 0o755
    assert tree["rsi-sandbox"].st_mode & 0o777 == 0o755
    source = Path(sandbox_client.__file__)
    for name, module in ENDPOINT_MODULES.items():
        assert (endpoint.directory / "py" / name).read_bytes() == (
            source.with_name(module).read_bytes()
        )
    assert (endpoint.directory / "rsi-sandbox").read_bytes() == source.read_bytes()
    directory = endpoint.directory
    if phase == "judge":
        lifecycle.close_judge()
    else:
        lifecycle.close()
    assert not directory.exists()


def test_injected_plugin_imports_its_siblings_without_the_harness(lifecycle):
    """Inside Work or Judge only the endpoint modules and Harbor exist."""
    endpoint = lifecycle.prepare_work()
    script = (
        "import sys\n"
        "sys.modules['rsi_harness'] = None\n"
        "import rsi_sandbox_harbor as plugin\n"
        "assert plugin.wire.__name__ == 'rsi_sandbox_client'\n"
        "assert plugin.compose.__name__ == 'rsi_sandbox_compose'\n"
        "assert plugin.ManagedSandboxEnvironment.type() == 'rsi-managed-sandbox'\n"
    )
    completed = subprocess.run(
        # -B: the real bind is read-only; never leave __pycache__ behind.
        [sys.executable, "-B", "-c", script],
        cwd=endpoint.directory,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "PYTHONPATH": str(endpoint.directory / "py"),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


class _ClosingBroker:
    """Only what configure() and close() call on a broker."""

    recovery_required = False

    def start(self):
        pass

    def cancel_run(self):
        pass

    def close(self):
        pass


@pytest.mark.parametrize("left", [None, "empty-build", "build", "recovery"])
def test_clean_close_removes_the_spool_and_sb(tmp_path, left):
    """A5: a cleanly closed run leaves no <run>/sb (an emptied build tree
    included); a loop file left in sb/build and a broker that needs
    recovery keep what recovery must still converge."""
    from rsi_harness.runtime.sandbox_env_contracts import sandbox_spool_root
    from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle

    broker = _ClosingBroker()
    sb = tmp_path / "run-1" / "sb"
    lifecycle = SandboxLifecycle()
    lifecycle.configure(broker, sb, "run-1", "task")
    spool = sandbox_spool_root(tmp_path, "run-1")
    # What a closed env broker leaves: emptied stage and exec directories.
    (spool / "stage" / "work-0a1b").mkdir(mode=0o700, parents=True)
    (spool / "x").mkdir(mode=0o700)
    (spool / "stage" / "work-0a1b" / "leftover").write_bytes(b"")
    if left in ("build", "empty-build"):
        (sb / "build").mkdir()
    if left == "build":
        (sb / "build" / ("0" * 16 + ".img")).write_bytes(b"")
    if left == "recovery":
        broker.recovery_required = True
    lifecycle.close()
    if left in (None, "empty-build"):
        assert not sb.exists()
    elif left == "build":
        assert sorted(path.name for path in sb.iterdir()) == ["build"]
    else:
        assert sorted(path.name for path in sb.iterdir()) == ["spool"]
    lifecycle.close()  # idempotent


def test_a_symlinked_spool_is_never_followed(tmp_path):
    from rsi_harness.errors import InfrastructureError
    from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle

    outside = tmp_path / "outside"
    (outside / "keep").mkdir(parents=True)
    sb = tmp_path / "run-1" / "sb"
    lifecycle = SandboxLifecycle()
    lifecycle.configure(_ClosingBroker(), sb, "run-1", "task")
    (sb / "spool").symlink_to(outside)
    with pytest.raises(InfrastructureError, match="recovery required"):
        lifecycle.close()
    assert (outside / "keep").is_dir()
