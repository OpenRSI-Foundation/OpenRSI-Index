"""The wire accepts finite operations, never Docker authority."""

import json
import socket
import stat
import struct
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rsi_harness.runtime.sandbox_contracts import SandboxError
from tests.runtime.test_sandbox import kit as kit
from tests.runtime.test_sandbox import work as work


@pytest.fixture
def client(kit, work):
    from rsi_harness.runtime.sandbox_server import create_sandbox_app

    with TestClient(create_sandbox_app(kit[0], work.owner)) as client:
        client.headers["Authorization"] = "Bearer " + work.credential
        yield client


def test_protocol_never_accepts_host_mounts(client, kit):
    response = client.post(
        "/v1/create",
        json={
            "profile": "offline",
            "lifetime_sec": 10,
            "request_id": "bad",
            "mounts": [{"source": "/", "target": "/host"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid"
    assert kit[0].journal.snapshot() == ()


def test_response_exposes_opaque_handle_not_docker_identity(client):
    response = client.post(
        "/v1/create",
        json={"profile": "offline", "lifetime_sec": 10, "request_id": "one"},
    )
    assert response.status_code == 200
    assert set(response.json()) == {"child_id"}
    assert len(response.json()["child_id"]) == 32


@pytest.mark.parametrize("body", [b"{}" * 70000, b'{"x":1,"x":2}', b"[" * 5000])
def test_bounded_control_json_rejects_overflow_duplicate_and_deep_input(client, body):
    response = client.post("/v1/create", content=body)
    assert response.status_code in (400, 413)


def test_wrong_and_expired_credentials_are_denied_before_decoding(client, kit):
    response = client.post(
        "/v1/upload", content=b"hostile", headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 401
    kit[2].now = 1001
    response = client.post("/v1/capabilities", json={})
    assert response.status_code in (401, 410)


def test_upload_metadata_length_checked_before_large_decode(client):
    response = client.post("/v1/upload", content=struct.pack("!I", 2**31))
    assert response.status_code == 413


@pytest.mark.parametrize("path", ["/v1/docker", "/v2/create", "/v1/create?host=1"])
def test_unsupported_routes_and_query_parameters_cannot_reach_backend(
    client, kit, path
):
    response = client.post(path, json={})
    assert response.status_code == 400
    assert response.json()["error"]["code"] in ("unsupported", "invalid")
    assert not kit[1].events


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/v1/containers/json"),
        ("GET", "/v1/containers/json"),
        ("POST", "/v1/containers/create"),
        ("GET", "/v1/capabilities"),
        ("DELETE", "/v1/create"),
        ("POST", "/v1/"),
        ("GET", "/v2/images/json"),
        ("PUT", "/v2/a/b/c"),
    ],
)
def test_engine_style_routes_are_unsupported_without_reading_the_body(
    client, kit, method, path
):
    """Spec A6: no Engine API. Anything but ``POST /v1/<operation>`` under
    the API prefixes answers 400 unsupported, unauthenticated and unread."""

    def body():
        raise AssertionError("the body was read")
        yield b""

    response = client.request(
        method, path, content=body(), headers={"Authorization": "Bearer nope"}
    )
    assert response.status_code == 400
    assert response.json()["error"] == {
        "code": "unsupported",
        "field": "operation",
        "message": "unknown sandbox operation",
    }
    assert not kit[1].events


def test_exec_unknown_outcome_is_structured_not_replayed(client, kit):
    handle = client.post(
        "/v1/create",
        json={"profile": "offline", "lifetime_sec": 10, "request_id": "one"},
    ).json()["child_id"]

    def unknown(_):
        raise SandboxError("unknown-outcome", "exec", "response lost")

    kit[1].hooks["execute"] = unknown
    response = client.post(
        "/v1/exec",
        json={
            "child_id": handle,
            "argv": ["true"],
            "cwd": "/workspace",
            "env": {},
            "timeout_sec": 1,
        },
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "unknown-outcome"
    assert [op for op, _ in kit[1].events].count("execute") == 1


def test_socket_path_limit_fails_before_binding(kit, work, tmp_path):
    from rsi_harness.errors import SetupError
    from rsi_harness.runtime.sandbox_server import SandboxServer

    with pytest.raises(SetupError, match="socket.*path"):
        SandboxServer(kit[0], tmp_path / ("x" * 110) / "s", work.owner)


def test_request_capacity_does_not_block_broker_cancellation(kit, work, monkeypatch):
    from rsi_harness.runtime.sandbox_server import (
        SandboxRequestSlots,
        create_sandbox_app,
    )

    broker = kit[0]
    gate = SandboxRequestSlots(active=1, queued=1)
    entered, release = threading.Event(), threading.Event()
    original = broker.capabilities

    def slow(credential):
        entered.set()
        assert release.wait(3)
        return original(credential)

    monkeypatch.setattr(broker, "capabilities", slow)
    with TestClient(create_sandbox_app(broker, work.owner, slots=gate)) as client:
        headers = {"Authorization": "Bearer " + work.credential}
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(
                client.post, "/v1/capabilities", json={}, headers=headers
            )
            assert entered.wait(2)
            # Reserve the only queued place without an HTTP timing race.
            assert gate.enter()
            try:
                rejected = client.post("/v1/capabilities", json={}, headers=headers)
                assert rejected.status_code == 409
                broker.cancel_run()
            finally:
                gate.leave()
                release.set()
            first.result()


def test_upload_and_download_use_binary_bundle(client, kit):
    from rsi_harness.integrations import sandbox_client as wire

    handle = client.post(
        "/v1/create",
        json={"profile": "offline", "lifetime_sec": 10, "request_id": "one"},
    ).json()["child_id"]
    metadata = json.dumps(
        {"child_id": handle, "root": "/workspace", "request_id": "up", "timeout_sec": 5}
    ).encode()
    body = struct.pack("!I", len(metadata)) + metadata + wire.encode_records([])
    assert client.post("/v1/upload", content=body).status_code == 200
    response = client.post(
        "/v1/download",
        json={
            "child_id": handle,
            "root": "/workspace",
            "paths": ["."],
            "timeout_sec": 5,
        },
    )
    assert response.status_code == 200
    assert response.content == b"RSIBNDL1\n\0\0\0\0"


def test_live_upload_retains_only_records_and_backend_stdin(
    live_servers, kit, work, monkeypatch
):
    """Dead HTTP/decoded/hash input copies must not span a blocked backend call."""
    import asyncio
    import sys
    from types import SimpleNamespace

    from rsi_harness.integrations.sandbox_client import SandboxClient
    from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry, SandboxResult
    from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend

    broker, backend, _ = kit
    server = live_servers()
    child = broker.create(work.credential, "offline", 30, "memory")
    payload_size = 2 * 1024**2
    entered, release = threading.Event(), threading.Event()
    executing = []
    real_backend = SandboxDockerBackend(None)
    real_backend._profiles[child.container_id] = SimpleNamespace(workdir="/workspace")
    monkeypatch.setattr(real_backend, "_transfer_argv", lambda *args: ["fixture"])
    monkeypatch.setattr(real_backend, "_validate_command", lambda *args: None)

    def blocked_exec(*args):
        executing.append(threading.get_ident())
        entered.set()
        assert release.wait(3)
        return SandboxResult(exit_code=0, duration_sec=0), b"ok\n", b""

    monkeypatch.setattr(real_backend, "_run_exec", blocked_exec)
    monkeypatch.setattr(backend, "upload", real_backend.upload)
    client = SandboxClient(server.path, work.credential)
    client._capabilities = {"version": 1}
    buffers = {}

    def retain(value):
        if isinstance(value, (bytes, bytearray)) and len(value) >= payload_size:
            buffers[id(value)] = len(value)
        elif isinstance(value, (tuple, list)):
            for entry in value:
                if isinstance(entry, SandboxBundleEntry):
                    retain(entry.data)

    try:
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(
                client.upload,
                child.child_id,
                "/workspace",
                (
                    {
                        "path": "in",
                        "kind": "file",
                        "mode": 0o644,
                        "data": b"x" * payload_size,
                    },
                ),
            )
            assert entered.wait(2)
            frame = sys._current_frames()[executing[0]]
            while frame is not None:
                if "/rsi_harness/" in frame.f_code.co_filename:
                    for value in frame.f_locals.values():
                        retain(value)
                frame = frame.f_back

            async def inspect_waiters():
                for task in asyncio.all_tasks():
                    coroutine = task.get_coro()
                    while coroutine is not None:
                        frame = getattr(coroutine, "cr_frame", None)
                        if (
                            frame is not None
                            and "/rsi_harness/" in frame.f_code.co_filename
                        ):
                            for value in frame.f_locals.values():
                                retain(value)
                        coroutine = getattr(coroutine, "cr_await", None)

            loop = next(iter(server._connections)).loop
            asyncio.run_coroutine_threadsafe(inspect_waiters(), loop).result(1)
            release.set()
            result.result(2)
    finally:
        release.set()
    assert sum(buffers.values()) <= 2 * payload_size + 128 * 1024, buffers


def test_queued_request_has_finite_wait(client, kit, monkeypatch):
    import asyncio

    from rsi_harness.runtime import sandbox_server

    async def scenario():
        slots = sandbox_server.SandboxRequestSlots(active=1, queued=1)
        assert slots._active.acquire(blocking=False)
        try:
            with pytest.raises(SandboxError, match="busy"):
                await slots.wait_active(timeout=0.01)
        finally:
            slots._active.release()

    asyncio.run(scenario())


@pytest.fixture
def live_servers(kit, work):
    from rsi_harness.runtime.sandbox_server import SandboxServer

    servers = []
    with tempfile.TemporaryDirectory(prefix="rsi-wire-") as directory:

        def start(*, mode=0o700, **kwargs):
            root = Path(directory) / str(len(servers))
            root.mkdir(mode=0o700)
            root.chmod(mode)
            server = SandboxServer(kit[0], root / "s", work.owner, **kwargs)
            servers.append(server)
            server.start()
            return server

        try:
            yield start
        finally:
            for server in servers:
                server.stop()


def _connect(server):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.settimeout(1)
    connection.connect(str(server.path))
    return connection


def _wait_for(predicate):
    deadline = time.monotonic() + 1
    while not predicate():
        assert time.monotonic() < deadline
        time.sleep(0.005)


def test_idle_connections_are_bounded_across_both_phase_endpoints(live_servers):
    """Moving a connection to Judge must not bypass Work's run-wide cap."""
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    slots = SandboxRequestSlots(active=1, queued=1)
    work_server = live_servers(slots=slots)
    judge_server = live_servers(slots=slots)
    with _connect(work_server) as first, _connect(work_server) as second:
        _wait_for(lambda: len(work_server._server.server_state.connections) == 2)
        with _connect(judge_server) as excess:
            assert excess.recv(1) == b""
        first.close()
        _wait_for(lambda: len(work_server._server.server_state.connections) == 1)
        with _connect(judge_server) as admitted:
            _wait_for(lambda: len(judge_server._server.server_state.connections) == 1)
            admitted.settimeout(0.02)
            with pytest.raises(TimeoutError):
                admitted.recv(1)
        assert second.fileno() >= 0


@pytest.mark.parametrize(
    "prefix", [b"", b"POST /v1/capabilities HTTP/1.1\r\nAuthorization:"]
)
def test_idle_and_partial_headers_expire_before_asgi(live_servers, prefix):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    slots = SandboxRequestSlots(active=1, queued=0, header_timeout=0.05)
    server = live_servers(slots=slots)
    with _connect(server) as connection:
        if prefix:
            connection.sendall(prefix)
        assert connection.recv(1) == b""
    _wait_for(lambda: not server._server.server_state.connections)


def test_header_progress_does_not_reset_absolute_header_deadline(live_servers):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    server = live_servers(
        slots=SandboxRequestSlots(active=1, queued=0, header_timeout=0.08)
    )
    with _connect(server) as connection:
        connection.sendall(b"POST /v1/capabilities HTTP/1.1\r\nX-Slow: ")
        started = time.monotonic()
        while True:
            connection.settimeout(0.02)
            try:
                assert connection.recv(1) == b""
                break
            except ConnectionResetError:
                break
            except TimeoutError:
                assert time.monotonic() - started < 0.5
                try:
                    connection.sendall(b"x")
                except (BrokenPipeError, ConnectionResetError):
                    break


def test_reused_connection_gets_a_new_bounded_header_window(live_servers, work):
    from rsi_harness.integrations.sandbox_client import UnixHTTPConnection
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    server = live_servers(
        slots=SandboxRequestSlots(active=1, queued=0, header_timeout=0.05)
    )
    connection = UnixHTTPConnection(server.path, timeout=1)
    try:
        connection.request(
            "POST",
            "/v1/capabilities",
            "{}",
            headers={"Authorization": "Bearer " + work.credential},
        )
        assert connection.getresponse().read()
        connection.sock.sendall(b"POST /v1/capabilities HTTP/1.1\r\nX-Slow:")
        assert connection.sock.recv(1) == b""
    finally:
        connection.close()


@pytest.mark.parametrize("parent_users, mode", [(False, 0o600), (True, 0o666)])
def test_socket_permission_allows_explicit_parent_user_mode(
    live_servers, parent_users, mode
):
    server = live_servers(parent_users=parent_users)
    assert stat.S_IMODE(server.path.stat().st_mode) == mode


@pytest.mark.parametrize(
    "parent_users, mode", [(False, 0o755), (False, 0o777), (True, 0o777)]
)
def test_endpoint_directory_mode_is_validated_before_binding(
    live_servers, parent_users, mode
):
    from rsi_harness.errors import SetupError

    with pytest.raises(SetupError, match="endpoint directory"):
        live_servers(parent_users=parent_users, mode=mode)


def test_endpoint_directory_must_be_owned_by_engine(live_servers, monkeypatch):
    from rsi_harness.errors import SetupError
    from rsi_harness.runtime import sandbox_server

    engine_uid = sandbox_server.os.getuid()
    monkeypatch.setattr(sandbox_server.os, "getuid", lambda: engine_uid + 1)
    with pytest.raises(SetupError, match="endpoint directory"):
        live_servers()


def test_complete_headers_disable_header_timer_during_operation(
    live_servers, kit, work, monkeypatch
):
    from rsi_harness.integrations.sandbox_client import SandboxClient
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    server = live_servers(
        slots=SandboxRequestSlots(active=1, queued=0, header_timeout=0.03)
    )
    capabilities = kit[0].capabilities

    def slow(credential):
        time.sleep(0.08)
        return capabilities(credential)

    monkeypatch.setattr(kit[0], "capabilities", slow)
    assert SandboxClient(server.path, work.credential).capabilities()["version"] == 1


def test_shutdown_releases_shared_connection_capacity(live_servers):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    slots = SandboxRequestSlots(active=1, queued=0)
    work_server = live_servers(slots=slots)
    judge_server = live_servers(slots=slots)
    with _connect(work_server) as work:
        _wait_for(lambda: len(work_server._server.server_state.connections) == 1)
        work_server.stop()
        assert work.recv(1) == b""
        with _connect(judge_server):
            _wait_for(lambda: len(judge_server._server.server_state.connections) == 1)


def _wire_request(operation, credential, metadata, *, close=False):
    body = json.dumps(metadata).encode()
    return (
        f"POST /v1/{operation} HTTP/1.1\r\nHost: localhost\r\n"
        f"Authorization: Bearer {credential}\r\nContent-Length: {len(body)}\r\n"
        + ("Connection: close\r\n" if close else "")
        + "\r\n"
    ).encode() + body


def _download_request(kit, work, monkeypatch):
    from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry

    monkeypatch.setattr(
        kit[1],
        "download",
        lambda *args, **kwargs: (
            SandboxBundleEntry(
                path="out", kind="file", mode=0o644, data=b"x" * 1024**2
            ),
        ),
    )
    child = kit[0].create(work.credential, "offline", 30, "slow-reader")
    return child, {
        "child_id": child.child_id,
        "root": "/workspace",
        "paths": ["out"],
        "timeout_sec": 1,
    }


def test_download_transport_backlog_is_chunk_bounded_and_body_is_complete(
    live_servers, kit, work, monkeypatch
):
    import http.client
    import io

    from rsi_harness.integrations import sandbox_client as wire

    server = live_servers()
    _, metadata = _download_request(kit, work, monkeypatch)
    with _connect(server) as peer:
        peer.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        peer.sendall(_wire_request("download", work.credential, metadata))
        _wait_for(
            lambda: any(
                p.transport.get_write_buffer_size() for p in server._connections
            )
        )
        assert server._slots._places._value == 19
        backlog = sum(p.transport.get_write_buffer_size() for p in server._connections)
        response = http.client.HTTPResponse(peer)
        response.begin()
        records = tuple(wire.iter_bundle(io.BytesIO(response.read())))
    assert backlog <= 3 * 65536
    assert len(records) == 1 and records[0]["data"] == b"x" * 1024**2
    _wait_for(lambda: server._slots._places._value == 20)


@pytest.mark.parametrize("cancel_when", ["backend", "response"])
def test_cancelled_asgi_request_owns_worker_and_discards_undelivered_body(
    live_servers, kit, work, monkeypatch, cancel_when
):
    import asyncio

    from rsi_harness.runtime import sandbox_server
    from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry

    slots = sandbox_server.SandboxRequestSlots(active=1, queued=0)
    server = live_servers(slots=slots)
    child = kit[0].create(work.credential, "offline", 30, "cancel-response")
    reached, release = threading.Event(), threading.Event()
    responses = []
    original_init = sandbox_server._BundleResponse.__init__

    def capture(response, *args, **kwargs):
        original_init(response, *args, **kwargs)
        responses.append(response)

    monkeypatch.setattr(sandbox_server._BundleResponse, "__init__", capture)

    def download(*args, **kwargs):
        if cancel_when == "backend":
            reached.set()
            assert release.wait(3)
        return (
            SandboxBundleEntry(
                path="out", kind="file", mode=0o644, data=b"x" * 1024**2
            ),
        )

    async def undelivered(response, scope, receive, send):
        reached.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(kit[1], "download", download)
    if cancel_when == "response":
        monkeypatch.setattr(sandbox_server._BundleResponse, "__call__", undelivered)
    try:
        with _connect(server) as peer:
            peer.sendall(
                _wire_request(
                    "download",
                    work.credential,
                    {
                        "child_id": child.child_id,
                        "root": "/workspace",
                        "paths": ["out"],
                        "timeout_sec": 5,
                    },
                )
            )
            assert reached.wait(2)
            assert slots._places._value == 0
            loop = next(iter(server._connections)).loop
            for task in tuple(server._server.server_state.tasks):
                loop.call_soon_threadsafe(task.cancel)
            _wait_for(lambda: not server._server.server_state.tasks)
            if cancel_when == "backend":
                assert slots._places._value == 0
                assert slots._active._value == 0
            release.set()
            _wait_for(lambda: slots._places._value == 1)
            assert slots._active._value == 1
            assert len(responses) == 1
            assert responses[0].body == b""
    finally:
        release.set()


@pytest.mark.parametrize("started", [False, True])
def test_cancelled_executor_waiter_returns_only_unstarted_worker_capacity(
    live_servers, kit, work, monkeypatch, started
):
    import asyncio
    from types import FunctionType

    from rsi_harness.integrations import sandbox_client as wire
    from rsi_harness.runtime import sandbox_server

    slots = sandbox_server.SandboxRequestSlots(active=1, queued=0)
    server = live_servers(slots=slots)
    child = kit[0].create(work.credential, "offline", 30, "cancel-executor")
    reached, release = threading.Event(), threading.Event()
    workers = []
    queued_calls = []
    backend_calls = []
    original = asyncio.to_thread

    async def observe(*args, **kwargs):
        workers.append(asyncio.current_task())
        if not started:
            queued_calls.append((args, kwargs))
            reached.set()
            await asyncio.Event().wait()
        return await original(*args, **kwargs)

    def upload(*args, **kwargs):
        backend_calls.append(1)
        reached.set()
        assert release.wait(3)

    def retained_payload(function):
        pending, seen, total = [function], set(), 0
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, (bytes, bytearray)):
                total += len(value)
            elif isinstance(value, FunctionType):
                pending.extend(cell.cell_contents for cell in (value.__closure__ or ()))
        return total

    monkeypatch.setattr(asyncio, "to_thread", observe)
    monkeypatch.setattr(kit[1], "upload", upload)
    metadata = json.dumps(
        {
            "child_id": child.child_id,
            "root": "/workspace",
            "request_id": "upload",
            "timeout_sec": 5,
        }
    ).encode()
    body = (
        struct.pack("!I", len(metadata))
        + metadata
        + wire.encode_records(
            (
                {
                    "path": "in",
                    "kind": "file",
                    "mode": 0o644,
                    "data": b"x" * (2 * 1024**2),
                },
            )
        )
    )
    try:
        with _connect(server) as peer:
            peer.sendall(
                (
                    "POST /v1/upload HTTP/1.1\r\nHost: localhost\r\n"
                    f"Authorization: Bearer {work.credential}\r\n"
                    f"Content-Length: {len(body)}\r\n\r\n"
                ).encode()
                + body
            )
            assert reached.wait(2)
            if not started:
                assert retained_payload(queued_calls[0][0][0]) >= 2 * 1024**2
            workers[0].get_loop().call_soon_threadsafe(workers[0].cancel)
            _wait_for(lambda: not server._server.server_state.tasks)
            if started:
                assert slots._places._value == 0
                assert slots._active._value == 0
                release.set()
            _wait_for(lambda: slots._places._value == 1)
            assert slots._active._value == 1
            if not started:
                # Even if an executor invokes the queued callable after the
                # waiter was cancelled, it must not reacquire mutation authority.
                args, kwargs = queued_calls[0]
                assert retained_payload(args[0]) < 128 * 1024
                args[0](*args[1:], **kwargs)
                assert slots._active._value == 1
            assert len(backend_calls) == int(started)
    finally:
        release.set()


def test_default_nonreaders_cannot_retain_all_work_and_judge_connections(
    live_servers, kit, work, monkeypatch
):
    from rsi_harness.integrations.sandbox_client import SandboxClient
    from rsi_harness.runtime.sandbox_contracts import SandboxOwner
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots, SandboxServer

    slots = SandboxRequestSlots()
    server = live_servers(slots=slots)
    _, metadata = _download_request(kit, work, monkeypatch)
    request = _wire_request("download", work.credential, metadata)
    connections = []
    judge_server = None
    try:
        for count in range(20):
            connection = _connect(server)
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            connections.append(connection)
            connection.sendall(request)
            # Serialize completions, ensuring every socket has a large response.
            _wait_for(
                lambda: (
                    sum(
                        p.transport.get_write_buffer_size() > 0
                        for p in server._server.server_state.connections
                    )
                    == count + 1
                )
            )
        deadline = time.monotonic() + 6
        while server._server.server_state.connections and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not server._server.server_state.connections

        kit[0].freeze_work()
        judge = kit[0].open_judge(
            SandboxOwner(
                run_id="run-1", task_id="task", phase="judge", round_id="round-1"
            ),
            500,
        )
        judge_server = SandboxServer(
            kit[0], server.path.parent / "judge", judge.owner, slots=slots
        )
        judge_server.start()
        assert (
            SandboxClient(judge_server.path, judge.credential).capabilities()["version"]
            == 1
        )
    finally:
        for connection in connections:
            connection.close()
        if judge_server is not None:
            judge_server.stop()


@pytest.mark.parametrize("close", [False, True])
def test_flush_deadline_aborts_even_when_peer_dribbles_or_close_is_pending(
    live_servers, kit, work, monkeypatch, close
):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    slots = SandboxRequestSlots(header_timeout=10, flush_timeout=0.12)
    server = live_servers(slots=slots)
    _, metadata = _download_request(kit, work, monkeypatch)
    with _connect(server) as connection:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        connection.sendall(
            _wire_request("download", work.credential, metadata, close=close)
        )
        _wait_for(
            lambda: any(
                p.transport.get_write_buffer_size() > 0
                for p in server._server.server_state.connections
            )
        )
        deadline = time.monotonic() + 0.5
        while server._server.server_state.connections and time.monotonic() < deadline:
            # Progress is deliberately insufficient to drain the pending response.
            connection.recv(1)
            time.sleep(0.02)
        assert not server._server.server_state.connections


@pytest.mark.parametrize("abort", ["flush", "suspend"])
def test_transport_abort_preserves_pipelined_mutation_worker_and_request_slot(
    live_servers, kit, work, monkeypatch, abort
):
    from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    slots = SandboxRequestSlots(
        active=1,
        queued=1,
        header_timeout=10,
        flush_timeout=0.12 if abort == "flush" else 5,
    )
    server = live_servers(slots=slots)
    child, metadata = _download_request(kit, work, monkeypatch)
    # A final bounded chunk can leave a short unread tail while the next
    # pipelined mutation starts. Larger bodies now correctly backpressure.
    monkeypatch.setattr(
        kit[1],
        "download",
        lambda *args, **kwargs: (
            SandboxBundleEntry(
                path="out", kind="file", mode=0o644, data=b"x" * (256 * 1024)
            ),
        ),
    )
    started, release = threading.Event(), threading.Event()

    def block(lease):
        started.set()
        assert release.wait(2), "mutation barrier timed out"

    kit[1].hooks["execute"] = block
    try:
        with _connect(server) as connection:
            connection.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            connection.sendall(
                _wire_request("download", work.credential, metadata)
                + _wire_request(
                    "exec",
                    work.credential,
                    {
                        "child_id": child.child_id,
                        "argv": ["true"],
                        "cwd": "/workspace",
                        "env": {},
                        "timeout_sec": 10,
                    },
                )
            )
            assert started.wait(1)
            if abort == "suspend":
                server.suspend()
            _wait_for(lambda: not server._server.server_state.connections)
            assert (
                kit[0].status(work.credential, child.child_id)["inflight"] == "execute"
            )
            assert not slots._active.acquire(blocking=False)
            assert slots._places._value == 1
            assert slots._connections._value == 2
    finally:
        release.set()
    _wait_for(
        lambda: kit[0].status(work.credential, child.child_id)["inflight"] is None
    )
    assert slots._active.acquire(blocking=False)
    slots._active.release()
    _wait_for(lambda: slots._places._value == 2)
    assert [operation for operation, _ in kit[1].events].count("execute") == 1


def test_suspension_and_close_race_release_capacity_and_forbid_reopen(live_servers):
    from rsi_harness.errors import InfrastructureError
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots

    slots = SandboxRequestSlots(active=1, queued=0)
    server = live_servers(slots=slots)
    with _connect(server) as connection:
        _wait_for(lambda: slots._connections._value == 0)
        barrier = threading.Barrier(2)

        def suspend():
            barrier.wait()
            server.suspend()

        def stop():
            barrier.wait()
            server.stop()

        with ThreadPoolExecutor(2) as pool:
            suspended, stopped = pool.submit(suspend), pool.submit(stop)
            suspended.result(timeout=2)
            stopped.result(timeout=2)
        assert connection.recv(1) == b""
    assert slots._connections._value == 1
    with pytest.raises(InfrastructureError, match="stopped"):
        server.resume()
    # Restart binds the same endpoint without inheriting the suspension state.
    server.start()
    with _connect(server):
        _wait_for(lambda: slots._connections._value == 0)


def test_stop_keeps_shutdown_intent_when_transport_drain_times_out(
    live_servers, kit, work, monkeypatch
):
    from rsi_harness.errors import InfrastructureError

    broker = kit[0]
    authenticating = threading.Event()
    authenticate = broker.authenticate

    def observed_authenticate(*args):
        authenticating.set()
        return authenticate(*args)

    monkeypatch.setattr(broker, "authenticate", observed_authenticate)
    server = live_servers()
    owned_server, owned_thread, owned_socket = (
        server._server,
        server._thread,
        server._socket,
    )
    identity = server._identity
    with _connect(server) as peer, ThreadPoolExecutor(1) as pool:
        _wait_for(lambda: len(server._connections) == 1)
        # The event loop waits on the real broker lock, as it can during a slow
        # journal operation. Its queued transport abort cannot run in that time.
        with broker._lock:
            peer.sendall(_wire_request("capabilities", work.credential, {}))
            assert authenticating.wait(1)
            stopping = pool.submit(server.stop)
            with pytest.raises(InfrastructureError, match="suspension pending"):
                stopping.result(timeout=6)
            assert owned_server.should_exit
            assert server._server is owned_server
            assert server._thread is owned_thread
            assert server._socket is owned_socket
            assert server._identity == identity
            assert server.path.exists()
        # Clearing the stall must terminate the listener without a second stop.
        owned_thread.join(timeout=2)
        assert not owned_thread.is_alive()
        assert server._slots._connections._value == 20
        assert server._slots._places._value == 20
        assert server._slots._active._value == 4


def test_stop_keeps_shutdown_intent_when_abort_scheduling_fails(live_servers):
    import asyncio

    server = live_servers()
    owned_thread = server._thread
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()
    with _connect(server):
        _wait_for(lambda: len(server._connections) == 1)
        protocol = next(iter(server._connections))
        live_loop = protocol.loop
        try:
            # Exercise a real closed-loop scheduling error without killing the
            # live test server's loop, so eventual shutdown remains observable.
            protocol.loop = closed_loop
            with pytest.raises(RuntimeError, match="closed"):
                server.stop()
            assert server._server.should_exit
            assert server._thread is owned_thread
            assert server.path.exists()
        finally:
            protocol.loop = live_loop
        owned_thread.join(timeout=2)
        assert not owned_thread.is_alive()
        assert server._slots._connections._value == 20


def test_stop_is_safe_before_start_and_after_failed_startup(kit, work, monkeypatch):
    from rsi_harness.errors import InfrastructureError
    from rsi_harness.runtime import sandbox_server

    def fail_startup(server, **kwargs):
        raise RuntimeError("server startup failed")

    with tempfile.TemporaryDirectory(prefix="rsi-start-stop-") as root:
        server = sandbox_server.SandboxServer(kit[0], Path(root) / "s", work.owner)
        server.stop()
        assert not server.path.exists()
        monkeypatch.setattr(sandbox_server._ReadyServer, "run", fail_startup)
        with pytest.raises(InfrastructureError, match="startup failed"):
            server.start()
        assert not server.path.exists()
        assert server._thread is None
        assert server._server is None
        assert server._socket is None
        server.stop()
