"""Phase-scoped Unix HTTP service, independent of submission's round lock."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import struct
import threading
from functools import partial
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from uvicorn.protocols.http.h11_impl import H11Protocol

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations import sandbox_client as wire
from rsi_harness.runtime.sandbox_archive import MAX_STAGE_FRAME
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_transfer import decode_bundle, encode_bundle
from rsi_harness.runtime.submissions import _ReadyServer

SANDBOX_TARGET = "/run/rsi-harness/sandbox"
LONG_POLL_TICK_SEC = 0.05


class _BundleResponse(Response):
    """Keep socket backlog bounded while retaining only one encoded bundle."""

    async def __call__(self, scope, receive, send):
        view = memoryview(self.body)
        try:
            await send(
                {
                    "type": "http.response.start",
                    "status": self.status_code,
                    "headers": self.raw_headers,
                }
            )
            for offset in range(0, max(1, len(view)), 65536):
                await send(
                    {
                        "type": "http.response.body",
                        "body": bytes(view[offset : offset + 65536]),
                        "more_body": offset + 65536 < len(view),
                    }
                )
            if self.background is not None:
                await self.background()
        finally:
            # A cancelled ASGI task or traceback must not retain the full body.
            view.release()
            self.body = b""


class _RequestPlace:
    """One place owns queue, real worker, and response lifetimes together."""

    def __init__(self, slots):
        self.slots = slots
        self.lock = threading.Lock()
        self.worker_pending = False
        self.worker_started = False
        self.discard_pending_inputs = None
        self.request_done = False
        self.released = False
        self.response = None

    def start_worker(self, discard_pending_inputs):
        with self.lock:
            self.worker_pending = True
            self.discard_pending_inputs = discard_pending_inputs

    def claim_worker(self):
        with self.lock:
            if not self.worker_pending:
                return False
            self.worker_started = True
            self.discard_pending_inputs = None
            return True

    def finish_waiter(self, task):
        if task.cancelled():
            with self.lock:
                if self.worker_pending and not self.worker_started:
                    # Cancellation won the claim race. A queued executor call
                    # that later starts must return without performing work.
                    self.discard_pending_inputs()
                    self.discard_pending_inputs = None
                    self.worker_pending = False
                    self.slots._active.release()
                    self._release()

    def deliver(self, response):
        with self.lock:
            if self.request_done:
                response.body = b""
            else:
                self.response = response

    def finish_worker(self):
        with self.lock:
            self.worker_pending = False
            self._release()

    def finish_request(self, task):
        with self.lock:
            self.request_done = True
            if self.response is not None:
                self.response.body = b""
                self.response = None
            self._release()

    def _release(self):
        if self.request_done and not self.worker_pending and not self.released:
            self.released = True
            self.slots.leave()


class SandboxRequestSlots:
    """One shared gate across a run's separate Work/Judge server threads.

    ``waiters`` long-polls (exec_wait, job_wait, env_status) wait with no
    request place, worker thread or active slot, each holding at most one
    control body; they read that body in a request place, and take a place
    again only for the short read afterwards. A long-poll that needs no
    wait keeps its request place and never takes a waiter place.
    """

    def __init__(
        self, active=4, queued=16, *, waiters=0, header_timeout=5, flush_timeout=5
    ):
        self._places = threading.BoundedSemaphore(active + queued)
        self._active = threading.BoundedSemaphore(active)
        self._waiters = threading.BoundedSemaphore(waiters) if waiters else None
        # Requests cannot gate clients that have not sent complete headers yet.
        # This separate connection authority spans both phase server threads.
        self._connections = threading.BoundedSemaphore(active + queued + waiters)
        self.header_timeout = header_timeout
        self.flush_timeout = flush_timeout

    @classmethod
    def from_policy(cls, host):
        """Sizes from ``[environments.host]``; profile grants keep the defaults."""
        if host is None:
            return cls()
        return cls(
            host.request_slots_active,
            host.request_slots_queued,
            waiters=host.waiters,
        )

    @property
    def long_polls(self):
        return self._waiters is not None

    def enter(self):
        return self._places.acquire(blocking=False)

    def leave(self):
        self._places.release()

    def enter_waiter(self):
        return self._waiters.acquire(blocking=False)

    def leave_waiter(self):
        self._waiters.release()

    async def wait_active(self, timeout=10):
        try:
            async with asyncio.timeout(timeout):
                while not self._active.acquire(blocking=False):
                    await asyncio.sleep(0.01)
        except TimeoutError as error:
            raise SandboxError(
                "busy", "requests", "sandbox queue wait expired"
            ) from error


class _BoundedH11Protocol(H11Protocol):
    """Bound admission, header reads and pending response writes independently."""

    def __init__(self, *args, slots, endpoint, **kwargs):
        self._slots = slots
        self._endpoint = endpoint
        self._connection_admitted = False
        self._header_timer = None
        self._flush_timer = None
        super().__init__(*args, **kwargs)

    def _cancel_header_timer(self):
        if self._header_timer is not None:
            self._header_timer.cancel()
            self._header_timer = None

    def _start_headers(self):
        self._cancel_header_timer()
        self._header_timer = self.loop.call_later(
            self._slots.header_timeout, self.transport.abort
        )

    def _cancel_flush_timer(self):
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

    def _flush_expired(self):
        self._flush_timer = None
        if self.transport.get_write_buffer_size():
            # close() waits for queued bytes and can retain a nonreader forever.
            self.transport.abort()

    def _bound_flush(self):
        if self._flush_timer is None and self.transport.get_write_buffer_size():
            self._flush_timer = self.loop.call_later(
                self._slots.flush_timeout, self._flush_expired
            )

    def pause_writing(self):
        super().pause_writing()
        self._bound_flush()

    def resume_writing(self):
        super().resume_writing()
        # Falling below the low-water mark is progress, not necessarily drain.
        # A slow reader must not restart the absolute pending-write deadline.
        if not self.transport.get_write_buffer_size():
            self._cancel_flush_timer()

    def connection_made(self, transport):
        # Registration and phase suspension share one authority: a late Work
        # peer must never consume the connection capacity reclaimed for Judge.
        with self._endpoint._admission:
            if not self._endpoint._accepting or not self._slots._connections.acquire(
                blocking=False
            ):
                transport.abort()
                return
            self._connection_admitted = True
            super().connection_made(transport)
            self._endpoint._connections.add(self)
            self._start_headers()

    def data_received(self, data):
        if self._connection_admitted:
            super().data_received(data)

    def handle_events(self):
        super().handle_events()
        if self.cycle is not None and not self.cycle.response_complete:
            self._cancel_header_timer()

    def on_response_complete(self):
        # ASGI completion only means bytes were queued. Retain this independent
        # timer across graceful close and any pipelined request/operation.
        self._bound_flush()
        if not self.transport.is_closing():
            self._start_headers()
        # This may immediately consume a pipelined request. handle_events then
        # cancels the new header timer once that request's headers are complete.
        super().on_response_complete()

    def connection_lost(self, exc):
        self._cancel_header_timer()
        self._cancel_flush_timer()
        if self._connection_admitted:
            self._connection_admitted = False
            try:
                super().connection_lost(exc)
            finally:
                self._slots._connections.release()
                with self._endpoint._admission:
                    self._endpoint._connections.discard(self)
                    self._endpoint._admission.notify_all()


_HTTP_METHODS = ("GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS")


def _error(error):
    status = {
        "permission": 401,
        "unsupported": 400,
        "invalid": 400,
        "busy": 409,
        "quota": 413,
        "expired": 410,
        "unknown-outcome": 502,
        "infrastructure": 503,
    }.get(error.code, 503)
    return JSONResponse(
        {
            "error": {
                "code": error.code,
                "field": error.field,
                "message": error.message[:4096],
            }
        },
        status_code=status,
    )


def _metadata(body):
    if len(body) > wire.MAX_CONTROL_BYTES:
        raise SandboxError("quota", "body", "control JSON exceeds 128 KiB")
    try:
        value = json.loads(body, object_pairs_hook=wire._unique_json)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise SandboxError("invalid", "body", "invalid control JSON") from error
    if not isinstance(value, dict):
        raise SandboxError("invalid", "body", "expected an object")
    return value


_FIELDS = {
    "capabilities": set(),
    "create": {"profile", "lifetime_sec", "request_id"},
    "exec": {"child_id", "argv", "cwd", "env", "timeout_sec"},
    "upload": {"child_id", "root", "request_id", "timeout_sec"},
    "download": {"child_id", "root", "paths", "timeout_sec"},
    "status": {"child_id"},
    "destroy": {"child_id"},
}
# Environment operations (v2); the v1 table above is unchanged.
_V2_FIELDS = {
    "stage_put": {"stage_id", "offset", "final", "sha256", "request_id"},
    "stage_get": {"stage_id", "offset", "length"},
    "image_pull": {"ref", "policy", "request_id"},
    "image_build": {
        "stage_id",
        "dockerfile",
        "dockerfile_inline",
        "target",
        "build_args",
        "labels",
        "no_cache",
        "network",
        "timeout_sec",
        "request_id",
    },
    "job_wait": {"job_id", "log_offset", "wait_sec"},
    "job_cancel": {"job_id"},
    "image_list": set(),
    "image_release": {"image"},
    "env_create": {"spec", "request_id"},
    "env_start": {"env_id", "wait_timeout_sec", "request_id"},
    "env_status": {"env_id", "wait_sec"},
    "env_stop_service": {"env_id", "service", "timeout_sec", "request_id"},
    "env_destroy": {"env_id"},
    "env_list": set(),
    "exec_start": {
        "env_id",
        "service",
        "argv",
        "cwd",
        "env",
        "user",
        "timeout_sec",
        "merge_stderr",
        "request_id",
    },
    "exec_wait": {"exec_id", "stdout_offset", "stderr_offset", "wait_sec", "max_bytes"},
    "exec_kill": {"exec_id", "signal", "scope"},
    "copy_in": {"env_id", "service", "dest_dir", "stage_id", "request_id"},
    "copy_out": {"env_id", "service", "path", "max_bytes", "exclude"},
    "path_stat": {"env_id", "service", "path", "follow"},
    "tool_install": {"env_id", "service", "tool"},
}
_ALL_FIELDS = {**_FIELDS, **_V2_FIELDS}
_LONG_POLL = frozenset({"exec_wait", "job_wait", "env_status"})
# Binary requests: !I meta_len | meta JSON | payload.
_BODY_CAPS = {
    "upload": wire.MAX_BODY_BYTES,
    "stage_put": 4 + wire.MAX_CONTROL_BYTES + MAX_STAGE_FRAME,
}


def _framed(operation, body):
    if len(body) < 4:
        raise SandboxError("invalid", "body", f"truncated {operation} header")
    size = struct.unpack("!I", body[:4])[0]
    if size > wire.MAX_CONTROL_BYTES:
        raise SandboxError("quota", "body", f"{operation} metadata exceeds limit")
    if 4 + size > len(body):
        raise SandboxError("invalid", "body", f"truncated {operation} metadata")
    return _metadata(body[4 : 4 + size]), bytes(memoryview(body)[4 + size :])


def _exact_fields(operation, metadata):
    if set(metadata) != _ALL_FIELDS[operation]:
        raise SandboxError(
            "invalid",
            "fields",
            "expected exactly: " + ", ".join(sorted(_ALL_FIELDS[operation])),
        )


async def _read_body(request, cap):
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdecimal() or int(declared) > cap):
        raise SandboxError("quota", "body", "request body exceeds limit")
    body = bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                if len(body) + len(chunk) > cap:
                    raise SandboxError("quota", "body", "request body exceeds limit")
                body.extend(chunk)
    except TimeoutError as error:
        raise SandboxError(
            "expired", "body", "request transfer exceeded 10 seconds"
        ) from error
    return body


async def _long_poll(changed, wait_sec):
    """Wait in the event loop, polling lock-free generations every 50 ms."""
    loop = asyncio.get_running_loop()
    end = loop.time() + wait_sec
    while loop.time() < end and not changed():
        await asyncio.sleep(LONG_POLL_TICK_SEC)


def create_sandbox_app(broker, owner, *, slots=None):
    slots = slots or SandboxRequestSlots()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/v1/{operation}")
    async def dispatch(operation: str, request: Request):
        active = dispatched = False
        try:
            if operation not in _ALL_FIELDS:
                return _error(
                    SandboxError(
                        "unsupported", "operation", "unknown sandbox operation"
                    )
                )
            if request.query_params:
                raise SandboxError(
                    "invalid", "query", "query parameters are not supported"
                )
            authorization = request.headers.get("authorization", "")
            if not authorization.startswith("Bearer "):
                raise SandboxError(
                    "permission", "credential", "missing bearer credential"
                )
            credential = authorization[7:]
            broker.authenticate(credential, owner)
            if operation != "capabilities" and not broker.supports(
                owner.phase, 2 if operation in _V2_FIELDS else 1
            ):
                # Before any body read: an ungranted operation never allocates.
                raise SandboxError(
                    "permission", "operation", "operation not granted to phase"
                )
            metadata = None
            held = False
            if operation in _LONG_POLL and slots.long_polls:
                # Only the body tells whether this is a wait at all, and any
                # body is read in a request place (a full request queue is
                # busy). An answer due now (wait_sec 0, a settled env) keeps
                # that place and never takes a waiter place, nor is refused
                # for lack of one; a real wait holds a waiter place only.
                if not slots.enter():
                    raise SandboxError(
                        "busy", "requests", "sandbox request queue is full"
                    )
                held = True
                try:
                    metadata = _metadata(
                        await _read_body(request, wire.MAX_CONTROL_BYTES)
                    )
                    _exact_fields(operation, metadata)
                    changed = broker.wait_condition(credential, operation, metadata)
                    if changed is not None:
                        slots.leave()
                        held = False
                        if not slots.enter_waiter():
                            raise SandboxError(
                                "busy",
                                "requests",
                                "sandbox long-poll capacity is full",
                            )
                        try:
                            await _long_poll(changed, metadata["wait_sec"])
                        finally:
                            slots.leave_waiter()
                except BaseException:
                    if held:
                        slots.leave()
                    raise
            if not held and not slots.enter():
                raise SandboxError("busy", "requests", "sandbox request queue is full")
            place = _RequestPlace(slots)
            asyncio.current_task().add_done_callback(place.finish_request)
            # Queue without blocking the event loop or allocating request bodies.
            await slots.wait_active()
            active = True
            broker.authenticate(credential, owner)
            payload = None
            if metadata is None:
                body = await _read_body(
                    request, _BODY_CAPS.get(operation, wire.MAX_CONTROL_BYTES)
                )
                if operation in _BODY_CAPS:
                    metadata, payload = _framed(operation, body)
                else:
                    metadata = _metadata(body)
                # A blocked backend owns decoded records, not the raw HTTP body.
                # Release it before dispatch.
                del body
                _exact_fields(operation, metadata)

            def perform():
                nonlocal payload
                try:
                    broker.authenticate(credential, owner)
                    if operation == "upload":
                        entries = decode_bundle(payload)
                        payload = None
                        broker.upload(credential, entries=entries, **metadata)
                        return JSONResponse({"ok": True})
                    if operation == "download":
                        entries = broker.download(credential, **metadata)
                        return _BundleResponse(
                            encode_bundle(entries),
                            media_type="application/octet-stream",
                        )
                    if operation == "stage_put":
                        data, payload = payload, None
                        return JSONResponse(
                            broker.stage_put(credential, payload=data, **metadata)
                        )
                    if operation == "stage_get":
                        return _BundleResponse(
                            broker.stage_get(credential, **metadata),
                            media_type="application/octet-stream",
                        )
                    method = (
                        broker.execute
                        if operation == "exec"
                        else getattr(broker, operation)
                    )
                    result = method(credential, **metadata)
                    if operation == "create":
                        result = {"child_id": result.child_id}
                    elif operation == "exec":
                        result = result.model_dump(mode="json")
                    elif operation == "destroy":
                        result = {"ok": True}
                    return JSONResponse(result)
                except SandboxError as error:
                    return _error(error)
                except (TypeError, ValueError) as error:
                    return _error(SandboxError("invalid", "request", str(error)[:4096]))
                except Exception:
                    return _error(
                        SandboxError(
                            "infrastructure",
                            "operation",
                            "sandbox operation failed; recovery may be required",
                        )
                    )

            def run():
                if not place.claim_worker():
                    return
                try:
                    # The executor Future never owns a large response. Its
                    # actual thread hands ownership to the request place, or
                    # discards the body if the ASGI task already ended.
                    place.deliver(perform())
                finally:
                    # perform's large decoded/serialization locals are gone
                    # before another worker may claim transient-memory space.
                    slots._active.release()
                    place.finish_worker()

            def discard_pending_inputs():
                nonlocal payload, metadata
                payload = None
                metadata = None

            place.start_worker(discard_pending_inputs)
            dispatched = True
            # A disconnect cannot free a slot while its mutation still runs.
            worker = asyncio.create_task(asyncio.to_thread(run))
            worker.add_done_callback(place.finish_waiter)
            await asyncio.shield(worker)
            return place.response
        except SandboxError as error:
            return _error(error)
        finally:
            if not dispatched:
                if active:
                    slots._active.release()

    async def unsupported():
        # Every other method or path under the API prefixes, Engine-style
        # ones included (spec A6: /v1/containers/json), is unsupported; the
        # request is not authenticated and its body never read.
        return _error(
            SandboxError("unsupported", "operation", "unknown sandbox operation")
        )

    for path in ("/v1/{path:path}", "/v2/{path:path}"):
        app.add_api_route(path, unsupported, methods=list(_HTTP_METHODS))
    return app


class SandboxServer:
    def __init__(self, broker, socket_path, owner, *, slots=None, parent_users=False):
        self.path = Path(socket_path)
        if not self.path.is_absolute() or len(os.fsencode(self.path)) > 107:
            raise SetupError(
                "sandbox socket path must be absolute and at most 107 bytes; "
                "choose a shorter data root"
            )
        self._slots = slots or SandboxRequestSlots()
        self._parent_users = parent_users
        self.app = create_sandbox_app(broker, owner, slots=self._slots)
        self._thread = self._server = self._socket = None
        self._identity = None
        self._error = None
        self._control = threading.RLock()
        self._admission = threading.Condition()
        self._accepting = True
        self._connections = set()

    def start(self):
        with self._control:
            self._start()

    def _start(self):
        if self._thread is not None:
            raise InfrastructureError("sandbox server already started")
        with self._admission:
            self._accepting = True
        with wire._directory(self.path.parent) as descriptor:
            metadata = os.fstat(descriptor)
            allowed_modes = {0o700, 0o755} if self._parent_users else {0o700}
            if (
                metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) not in allowed_modes
            ):
                raise SetupError(
                    "sandbox endpoint directory is not Engine-owned "
                    "with safe permissions"
                )
        bound = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            bound.bind(str(self.path))
            os.chmod(self.path, 0o666 if self._parent_users else 0o600)
            info = self.path.lstat()
            self._identity = (info.st_dev, info.st_ino)
            bound.listen(20)
            ready = threading.Event()
            server = _ReadyServer(
                uvicorn.Config(
                    self.app,
                    access_log=False,
                    log_level="warning",
                    timeout_keep_alive=1,
                    timeout_graceful_shutdown=6,
                    h11_max_incomplete_event_size=16384,
                    http=partial(_BoundedH11Protocol, slots=self._slots, endpoint=self),
                    ws="none",
                ),
                ready,
            )

            def run():
                try:
                    server.run(sockets=[bound])
                except BaseException as error:
                    self._error = error
                    ready.set()

            self._socket, self._server = bound, server
            self._thread = threading.Thread(
                target=run, name="rsi-sandbox-http", daemon=True
            )
            self._thread.start()
            if not ready.wait(5) or self._error is not None or not server.started:
                self.stop()
                raise InfrastructureError("sandbox socket server startup failed")
        except BaseException:
            bound.close()
            raise

    def suspend(self):
        """Reclaim this phase's transports without cancelling its workers."""
        with self._control, self._admission:
            self._accepting = False
            loops = {connection.loop for connection in self._connections}
            if self._server is not None:
                loops.update(
                    task.get_loop() for task in self._server.server_state.tasks
                )
            for connection in tuple(self._connections):
                connection.loop.call_soon_threadsafe(connection.transport.abort)
            pending_drains = [len(loops)]

            async def drain_requests():
                # These are only Uvicorn's ASGI request tasks. Shielded workers
                # and their real executor threads retain independent ownership.
                tasks = tuple(self._server.server_state.tasks)
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                with self._admission:
                    pending_drains[0] -= 1
                    self._admission.notify_all()

            for loop in loops:
                asyncio.run_coroutine_threadsafe(drain_requests(), loop)
            # abort is asynchronous. Do not admit Judge until connection_lost
            # and completed ASGI bodies return their places. Pending mutations
            # keep their worker/request places until their actual thread ends.
            if not self._admission.wait_for(
                lambda: not self._connections and not pending_drains[0], timeout=5
            ):
                raise InfrastructureError(
                    "sandbox transport suspension pending; recovery required"
                )

    def resume(self):
        with self._control, self._admission:
            if self._server is None or self._server.should_exit:
                raise InfrastructureError("sandbox endpoint is stopped")
            self._accepting = True

    def stop(self):
        with self._control:
            # Record intent before scheduling/draining transports: even when
            # that fails, a recovering loop must still terminate its listener.
            if self._server is not None:
                self._server.should_exit = True
            self.suspend()
            self._stop()

    def _stop(self):
        if self._thread is not None:
            self._thread.join(7)
            if self._thread.is_alive():
                raise InfrastructureError(
                    "sandbox server shutdown pending; recovery required"
                )
        if self._socket is not None:
            self._socket.close()
        if self._identity is not None:
            try:
                info = self.path.lstat()
            except FileNotFoundError:
                pass
            else:
                if (
                    not stat.S_ISSOCK(info.st_mode)
                    or (info.st_dev, info.st_ino) != self._identity
                ):
                    raise InfrastructureError(
                        "sandbox socket identity changed; refusing cleanup"
                    )
                self.path.unlink()
        self._thread = self._server = self._socket = None
