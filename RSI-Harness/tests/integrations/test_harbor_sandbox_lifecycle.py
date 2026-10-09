"""Cancellation must retain authority over thread-backed Harbor children."""

import asyncio
import threading

import pytest

from rsi_harness.integrations import harbor_sandbox
from rsi_harness.integrations.sandbox_client import ProtocolError
from tests.integrations.test_harbor_sandbox import (
    FakeClient,
    _environment,
    _offline_policy,
)


class OwnedClient(FakeClient):
    """The external service owns a child even before its response is delivered."""

    def __init__(self):
        super().__init__()
        self.children = set()
        self.requests = {}
        self.request_ids = []
        self.created = threading.Event()
        self.release_create = threading.Event()
        self.release_create.set()
        self.destroying = threading.Event()
        self.release_destroy = threading.Event()
        self.release_destroy.set()
        self.destroy_error = None
        self.lose_create_response = False
        self.destroys = 0

    def create(self, profile, lifetime_sec, request_id=None):
        self.request_ids.append(request_id)
        key = request_id if request_id is not None else str(len(self.request_ids))
        if key not in self.requests:
            self.requests[key] = f"child-{len(self.requests)}"
            self.children.add(self.requests[key])
        self.created.set()
        assert self.release_create.wait(3), "create barrier timed out"
        if self.lose_create_response:
            self.lose_create_response = False
            raise ProtocolError("response lost", code="unknown-outcome")
        return self.requests[key]

    def destroy(self, handle):
        self.destroys += 1
        self.destroying.set()
        assert self.release_destroy.wait(3), "destroy barrier timed out"
        if self.destroy_error:
            error, self.destroy_error = self.destroy_error, None
            raise error
        self.children.discard(handle)


@pytest.mark.asyncio
async def test_cancelled_create_is_retained_and_reconciled_by_stop(tmp_path):
    client = OwnedClient()
    client.release_create.clear()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    starting = asyncio.create_task(env.start(force_build=False))
    try:
        assert await asyncio.to_thread(client.created.wait, 2)
        starting.cancel()
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(starting, 1)
    finally:
        client.release_create.set()
    await env.stop(delete=True)
    assert not client.children
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_stop_during_start_blocks_another_start_and_removes_late_child(tmp_path):
    client = OwnedClient()
    client.release_create.clear()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    starting = asyncio.create_task(env.start(force_build=False))
    stopping = None
    try:
        assert await asyncio.to_thread(client.created.wait, 2)
        stopping = asyncio.create_task(env.stop(delete=True))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(env.start(force_build=False), 0.1)
    finally:
        client.release_create.set()
        if stopping is not None:
            await stopping
        try:
            await starting
        except RuntimeError:
            pass
    assert not client.children
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_failed_destroy_preserves_handle_for_stop_retry(tmp_path):
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    client.destroy_error = ProtocolError(
        "lost destroy response", code="unknown-outcome"
    )
    with pytest.raises(harbor_sandbox.SandboxUnknownOutcomeError):
        await env.stop(delete=True)
    assert env._child_id in client.children
    await env.stop(delete=True)
    assert not client.children
    assert env._child_id is None


@pytest.mark.asyncio
async def test_cancelled_stop_retains_shared_bounded_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(harbor_sandbox, "_LIFECYCLE_WAIT_SEC", 0.05, raising=False)
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    client.release_destroy.clear()
    stopping = asyncio.create_task(env.stop(delete=True))
    try:
        assert await asyncio.to_thread(client.destroying.wait, 2)
        stopping.cancel()
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping
        assert env._child_id in client.children
        with pytest.raises(harbor_sandbox.SandboxUnknownOutcomeError, match="pending"):
            await env.stop(delete=True)
    finally:
        client.release_destroy.set()
    await env.stop(delete=True)
    assert not client.children
    assert client.destroys == 1


@pytest.mark.asyncio
async def test_lost_create_response_is_reconciled_using_same_request_id(tmp_path):
    client = OwnedClient()
    client.lose_create_response = True
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    with pytest.raises(ProtocolError, match="response lost"):
        await env.start(force_build=False)
    await env.stop(delete=True)
    assert not client.children
    assert len(client.requests) == 1
    assert len(client.request_ids) == 2
    assert client.request_ids[0] is not None
    assert client.request_ids[0] == client.request_ids[1]


@pytest.mark.asyncio
async def test_denied_reconciliation_does_not_forget_unknown_create(tmp_path):
    class DeniedReplay(OwnedClient):
        def create(self, *args, **kwargs):
            result = super().create(*args, **kwargs)
            if len(self.request_ids) == 2:
                raise ProtocolError("reconciliation denied", code="permission")
            return result

    client = DeniedReplay()
    client.lose_create_response = True
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    with pytest.raises(ProtocolError, match="reconciliation denied"):
        await env.start(force_build=False)
    await env.stop(delete=True)
    assert not client.children
    assert len(set(client.request_ids)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["exec", "upload_file", "upload_dir", "download_file", "download_dir"]
)
async def test_cancelled_operation_is_drained_before_owned_cleanup(
    tmp_path, monkeypatch, operation
):
    """Dropping a cancelled thread would destroy a child while it is busy."""
    monkeypatch.setattr(harbor_sandbox, "_LIFECYCLE_WAIT_SEC", 0.03)
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    entered, release = threading.Event(), threading.Event()
    service_method = {
        "exec": "execute",
        "upload_file": "upload",
        "upload_dir": "upload",
        "download_file": "download",
        "download_dir": "download",
    }[operation]
    actual = getattr(client, service_method)
    operation_calls = []

    def blocked(*args, **kwargs):
        operation_calls.append(args)
        entered.set()
        assert release.wait(3), "operation barrier timed out"
        return actual(*args, **kwargs)

    monkeypatch.setattr(client, service_method, blocked)
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("payload")
    arguments = {
        "exec": ("sleep 1",),
        "upload_file": (source / "file", "/tests/file"),
        "upload_dir": (source, "/tests"),
        "download_file": ("/tests/file", tmp_path / "download"),
        "download_dir": ("/tests", tmp_path / "download"),
    }
    command = asyncio.create_task(getattr(env, operation)(*arguments[operation]))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command
        with pytest.raises(harbor_sandbox.SandboxUnknownOutcomeError, match="pending"):
            await env.stop(delete=True)
        assert client.destroys == 0
        assert client.children
        release.set()
        # No second stop is needed: Harbor marks the environment stopped even
        # when the caller's bounded cleanup wait has elapsed.
        await asyncio.wait_for(asyncio.shield(env._cleanup_task), 2)
        assert not client.children
        assert len(operation_calls) == 1
        await env.stop(delete=True)
        assert client.destroys == 1
    finally:
        release.set()
        await asyncio.gather(command, return_exceptions=True)
        await env.stop(delete=True)


@pytest.mark.asyncio
async def test_stop_closes_admission_while_retaining_child(tmp_path):
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    client.release_destroy.clear()
    stopping = asyncio.create_task(env.stop(delete=True))
    try:
        assert await asyncio.to_thread(client.destroying.wait, 2)
        with pytest.raises(RuntimeError, match="stop"):
            await env.exec("must not execute")
    finally:
        client.release_destroy.set()
        await stopping


def test_completed_cancelled_startup_can_reconcile_after_loop_shutdown(tmp_path):
    client = OwnedClient()
    client.release_create.clear()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    timer = threading.Timer(0.25, client.release_create.set)
    timer.start()

    async def first_loop():
        starting = asyncio.create_task(env.start(force_build=False))
        assert await asyncio.to_thread(client.created.wait, 2)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting

    try:
        asyncio.run(first_loop())
        assert client.children
        assert env._startup_task.cancelled()
        asyncio.run(env.stop(delete=True))
        asyncio.run(env.stop(delete=True))
        assert not client.children
        assert len(client.requests) == 1
        assert len(set(client.request_ids)) == 1
        assert client.destroys == 1
    finally:
        client.release_create.set()
        timer.join(2)
        for child in tuple(client.children):
            client.destroy(child)


@pytest.mark.asyncio
async def test_busy_destroy_is_reconciled_without_second_stop(tmp_path):
    """A terminal client error does not prove the broker finished the request."""
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    client.destroy_error = ProtocolError("operation still pending", code="busy")
    await env.stop(delete=True)
    assert not client.children
    assert client.destroys == 2


@pytest.mark.asyncio
async def test_completed_failed_operation_does_not_poison_stop(tmp_path, monkeypatch):
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)

    def failed(*args, **kwargs):
        raise ProtocolError("invalid command", code="invalid")

    monkeypatch.setattr(client, "execute", failed)
    with pytest.raises(ProtocolError, match="invalid command"):
        await env.exec("invalid command")
    await env.stop(delete=True)
    await env.stop(delete=True)
    assert not client.children
    assert client.destroys == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["execute", "upload"])
async def test_stop_during_startup_operation_waits_without_deadlock(
    tmp_path, monkeypatch, method
):
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    (env.environment_dir / "input").write_text("startup input")
    entered, release = threading.Event(), threading.Event()
    actual = getattr(client, method)

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(3), "startup operation barrier timed out"
        return actual(*args, **kwargs)

    monkeypatch.setattr(client, method, blocked)
    starting = asyncio.create_task(env.start(force_build=False))
    stopping = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        stopping = asyncio.create_task(env.stop(delete=True))
        await asyncio.sleep(0)
        assert client.destroys == 0
        release.set()
        with pytest.raises(
            harbor_sandbox.ManagedSandboxError, match="stopped during startup"
        ):
            await asyncio.wait_for(starting, 2)
        await asyncio.wait_for(stopping, 2)
        assert not client.children
    finally:
        release.set()
        await asyncio.gather(
            starting, *([stopping] if stopping else []), return_exceptions=True
        )
        await env.stop(delete=True)


@pytest.mark.asyncio
async def test_cancelled_failing_operation_does_not_skip_cleanup(tmp_path, monkeypatch):
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    entered, release = threading.Event(), threading.Event()

    def failed(*args, **kwargs):
        entered.set()
        assert release.wait(3), "failed operation barrier timed out"
        raise ProtocolError("lost response", code="unknown-outcome")

    monkeypatch.setattr(client, "execute", failed)
    command = asyncio.create_task(env.exec("failed operation"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        command.cancel()
        with pytest.raises(asyncio.CancelledError):
            await command
        stopping = asyncio.create_task(env.stop(delete=True))
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(stopping, 2)
        assert not client.children
        await env.stop(delete=True)
        assert client.destroys == 1
    finally:
        release.set()
        await asyncio.gather(command, return_exceptions=True)
        await env.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload_file", "upload_dir"])
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_upload_preparation_owns_generation_until_cleanup(
    tmp_path, monkeypatch, operation, cancel_waiter
):
    """Preparation must not resume into a replacement sandbox after stop."""
    monkeypatch.setattr(harbor_sandbox, "_LIFECYCLE_WAIT_SEC", 0.03)
    client = OwnedClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    original_child = env._child_id
    source = tmp_path / "payload"
    source.mkdir()
    (source / "file").write_text("original generation")
    entered, release = threading.Event(), threading.Event()
    iterator = (
        "iter_selected_records" if operation == "upload_file" else "iter_local_records"
    )
    actual = getattr(harbor_sandbox, iterator)

    def preparing(*args):
        entered.set()
        assert release.wait(3), "upload preparation barrier timed out"
        return actual(*args)

    monkeypatch.setattr(harbor_sandbox, iterator, preparing)
    source_path = source / "file" if operation == "upload_file" else source
    target_path = "/tests/file" if operation == "upload_file" else "/tests"
    upload = asyncio.create_task(getattr(env, operation)(source_path, target_path))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        if cancel_waiter:
            upload.cancel()
            with pytest.raises(asyncio.CancelledError):
                await upload
        with pytest.raises(harbor_sandbox.SandboxUnknownOutcomeError, match="pending"):
            await env.stop(delete=True)
        with pytest.raises(RuntimeError, match="cleanup is pending"):
            await env.start(force_build=False)
        assert client.children == {original_child}
        assert client.destroys == 0
        release.set()
        await asyncio.gather(upload, return_exceptions=cancel_waiter)
        await asyncio.wait_for(asyncio.shield(env._cleanup_task), 2)
        assert not client.children
        await env.start(force_build=False)
        assert env._child_id != original_child
        uploads = [call for call in client.calls if call[0] == "upload"]
        assert len(uploads) == 1
        assert uploads[0][1] == original_child
    finally:
        release.set()
        await asyncio.gather(upload, return_exceptions=True)
        await env.stop(delete=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["download_file", "download_dir"])
async def test_download_materialization_finishes_before_stop_and_restart(
    tmp_path, monkeypatch, operation
):
    monkeypatch.setattr(harbor_sandbox, "_LIFECYCLE_WAIT_SEC", 0.03)
    client = OwnedClient()
    client.download_result = (
        {"kind": "file", "path": "file", "mode": 0o600, "data": b"old generation"},
    )
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(force_build=False)
    entered, release = threading.Event(), threading.Event()
    actual = harbor_sandbox.write_local_records

    def writing(*args):
        entered.set()
        assert release.wait(3), "download materialization barrier timed out"
        return actual(*args)

    monkeypatch.setattr(harbor_sandbox, "write_local_records", writing)
    target = tmp_path / "download"
    remote = "/tests/file" if operation == "download_file" else "/tests"
    download = asyncio.create_task(getattr(env, operation)(remote, target))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        download.cancel()
        with pytest.raises(asyncio.CancelledError):
            await download
        with pytest.raises(harbor_sandbox.SandboxUnknownOutcomeError, match="pending"):
            await env.stop(delete=True)
        with pytest.raises(RuntimeError, match="cleanup is pending"):
            await env.start(force_build=False)
        release.set()
        await asyncio.wait_for(asyncio.shield(env._cleanup_task), 2)
        destination = target if operation == "download_file" else target / "file"
        assert destination.read_bytes() == b"old generation"
        assert not client.children
    finally:
        release.set()
        await asyncio.gather(download, return_exceptions=True)
        await env.stop(delete=True)
