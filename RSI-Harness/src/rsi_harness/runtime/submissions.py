"""Token-scoped synchronous submissions served by an in-process Uvicorn app."""

from __future__ import annotations

import hashlib
import json
import secrets
import socket
import threading
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict

from rsi_harness.errors import RetryableSubmissionError
from rsi_harness.models import (
    ContainerRef,
    EvaluationRequest,
    RunPlan,
    SubmissionReport,
)
from rsi_harness.runtime.protocols import Clock, Evaluator


class EmptySubmission(BaseModel):
    """The submission request deliberately has no data-bearing fields."""

    model_config = ConfigDict(extra="forbid")


class SubmissionSessionState(BaseModel):
    """Persistable control state; bearer material is represented only by a hash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    token_hash: str
    rounds_allocated: int = 0
    last_allocated_at: datetime | None = None


class JudgeEndpoint(BaseModel):
    """The Work-reachable address of one embedded submission server."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    bind_host: str
    port: int


class SubmissionClosedError(RuntimeError):
    """A submission was rejected by lifecycle or run budget authority."""


class SubmissionStartupTimeout(TimeoutError):
    """The embedded server did not become ready within its startup bound."""


class SubmissionShutdownTimeout(TimeoutError):
    """The current round or embedded server did not stop within its bound."""


class SubmissionService:
    """Own one run's authentication, round allocation, and synchronous history."""

    def __init__(
        self,
        *,
        evaluator: Evaluator,
        artifact_writer: object,
        clock: Clock,
    ) -> None:
        self._evaluator = evaluator
        self._artifact_writer = artifact_writer
        self._clock = clock
        self._session: SubmissionSessionState | None = None
        self._run_plan: RunPlan | None = None
        self._work_container: ContainerRef | None = None
        self._max_submissions: int | None = None
        self._cooldown_seconds = 0.0
        self._reports: list[SubmissionReport] = []
        self._round_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._accepting = True

    def register(
        self,
        *,
        run_id: str,
        run_plan: RunPlan,
        work_container: ContainerRef,
        max_submissions: int | None = None,
        cooldown_seconds: float = 0.0,
    ) -> str:
        """Register exactly one run and return its runtime-only bearer token."""
        if not run_id:
            raise ValueError("run_id must not be empty")
        if max_submissions is not None and max_submissions < 0:
            raise ValueError("max_submissions must be non-negative")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        with self._state_lock:
            if self._session is not None:
                raise RuntimeError("a run is already registered")
            if not self._accepting:
                raise SubmissionClosedError("submission service is closed")
            token = secrets.token_urlsafe(32)
            self._session = SubmissionSessionState(
                run_id=run_id,
                token_hash=_token_hash(token),
            )
            self._run_plan = run_plan
            self._work_container = work_container
            self._max_submissions = max_submissions
            self._cooldown_seconds = cooldown_seconds
            return token

    @property
    def session_state(self) -> SubmissionSessionState:
        with self._state_lock:
            if self._session is None:
                raise RuntimeError("no run is registered")
            return self._session

    @property
    def reports(self) -> tuple[SubmissionReport, ...]:
        """Return a stable history snapshot after close/drain has completed."""
        with self._round_lock:
            return tuple(self._reports)

    def submit(self, token: str) -> str:
        """Synchronously evaluate and persist one serialized Agent round."""
        self._authenticate(token)
        with self._state_lock:
            if not self._accepting:
                raise SubmissionClosedError("submission service is closed")
        with self._round_lock:
            with self._state_lock:
                if not self._accepting:
                    raise SubmissionClosedError("submission service is closed")
                session = self._require_session()
                previous_session = session
                now = self._clock.now()
                if (
                    self._max_submissions is not None
                    and session.rounds_allocated >= self._max_submissions
                ):
                    raise SubmissionClosedError("submission budget is exhausted")
                if session.last_allocated_at is not None:
                    elapsed = (now - session.last_allocated_at).total_seconds()
                    if elapsed < self._cooldown_seconds:
                        remaining = self._cooldown_seconds - elapsed
                        raise SubmissionClosedError(
                            f"submission cooldown has {remaining:.3f} seconds remaining"
                        )
                round_number = session.rounds_allocated + 1
                round_id = f"agent-{round_number}"
                self._session = session.model_copy(
                    update={
                        "rounds_allocated": round_number,
                        "last_allocated_at": now,
                    }
                )
                plan = self._require_plan()
                work = self._require_work_container()
                remaining_budget = self._remaining_budget(round_number)

            try:
                report = self._evaluator.evaluate(
                    EvaluationRequest(
                        run_plan=plan,
                        work_container=work,
                        round_id=round_id,
                        verifier_logs=_verifier_logs(plan, session.run_id, round_id),
                        verifier_output=_verifier_output(
                            plan, session.run_id, round_id
                        ),
                    )
                )
            except RetryableSubmissionError:
                with self._state_lock:
                    self._session = previous_session
                raise
            except BaseException:
                if bool(getattr(self._evaluator, "recovery_required", False)) or bool(
                    getattr(self._evaluator, "submission_closed", False)
                ):
                    with self._state_lock:
                        self._accepting = False
                raise
            if report.round_id != round_id:
                report = report.model_copy(update={"round_id": round_id})
            self._reports.append(report)
            if (
                report.error is not None
                and "recovery_required" in report.error
                and report.status.value == "infrastructure_error"
            ):
                with self._state_lock:
                    self._accepting = False
            return _render_feedback(report, remaining_budget=remaining_budget)

    def history(self, token: str) -> str:
        """Return only the registered run's completed submission feedback."""
        self._authenticate(token)
        with self._round_lock:
            if not self._reports:
                return "No submissions yet.\n"
            return "\n".join(
                _render_feedback(
                    report,
                    remaining_budget=self._remaining_budget(index),
                )
                for index, report in enumerate(self._reports, start=1)
            )

    def close(self, *, timeout_seconds: float = 10.0) -> None:
        """Reject new work, then drain at most one active serialized round."""
        if timeout_seconds < 0:
            raise ValueError("timeout_seconds must be non-negative")
        with self._state_lock:
            self._accepting = False
        acquired = self._round_lock.acquire(timeout=timeout_seconds)
        if not acquired:
            raise SubmissionShutdownTimeout(
                "submission round did not drain before shutdown timeout"
            )
        self._round_lock.release()

    def _authenticate(self, token: str) -> None:
        candidate_hash = _token_hash(token)
        with self._state_lock:
            expected_hash = self._require_session().token_hash
        if not secrets.compare_digest(candidate_hash, expected_hash):
            raise PermissionError("invalid submission token")

    def _remaining_budget(self, allocated: int) -> int | None:
        if self._max_submissions is None:
            return None
        return max(0, self._max_submissions - allocated)

    def _require_session(self) -> SubmissionSessionState:
        if self._session is None:
            raise RuntimeError("no run is registered")
        return self._session

    def _require_plan(self) -> RunPlan:
        if self._run_plan is None:
            raise RuntimeError("no run is registered")
        return self._run_plan

    def _require_work_container(self) -> ContainerRef:
        if self._work_container is None:
            raise RuntimeError("no run is registered")
        return self._work_container


def create_submission_app(service: SubmissionService) -> FastAPI:
    """Build the narrow FastAPI surface owned by an EmbeddedJudgeServer."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    async def require_control_only(request: Request) -> None:
        if request.query_params:
            raise HTTPException(
                status_code=422, detail="query parameters are not allowed"
            )
        if await request.body():
            raise HTTPException(status_code=422, detail="submission body must be empty")

    @app.post("/api/v1/submit", response_class=PlainTextResponse)
    def submit(
        payload: EmptySubmission = Body(default_factory=EmptySubmission),
        authorization: str | None = Header(default=None),
        control_only: None = Depends(require_control_only),
    ) -> str:
        del payload, control_only
        token = _bearer_token(authorization)
        try:
            return service.submit(token)
        except PermissionError as error:
            raise HTTPException(
                status_code=401,
                detail="invalid submission token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from error
        except SubmissionClosedError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error
        except RetryableSubmissionError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/v1/history", response_class=PlainTextResponse)
    def history(
        authorization: str | None = Header(default=None),
        control_only: None = Depends(require_control_only),
    ) -> str:
        del control_only
        token = _bearer_token(authorization)
        try:
            return service.history(token)
        except PermissionError as error:
            raise HTTPException(
                status_code=401,
                detail="invalid submission token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from error

    return app


class _ReadyServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, ready: threading.Event) -> None:
        super().__init__(config)
        self._ready = ready

    async def startup(self, sockets: list[socket.socket] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            self._ready.set()


class EmbeddedJudgeServer:
    """Own a pre-bound Uvicorn server and its bounded in-process lifecycle."""

    def __init__(
        self,
        service: SubmissionService,
        *,
        bind_host: str,
        port: int,
        bridge_gateway: str,
        startup_timeout_seconds: float = 5.0,
        shutdown_timeout_seconds: float = 10.0,
    ) -> None:
        if not bind_host or not bridge_gateway:
            raise ValueError("bind_host and bridge_gateway must not be empty")
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if startup_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("server timeouts must be positive")
        self._service = service
        self._bind_host = bind_host
        self._port = port
        self._bridge_gateway = bridge_gateway
        self._startup_timeout = startup_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._ready = threading.Event()
        self._server: _ReadyServer | None = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None
        self._thread_error: BaseException | None = None
        self._endpoint: JudgeEndpoint | None = None

    def start(self) -> JudgeEndpoint:
        if self._thread is not None:
            raise RuntimeError("embedded Judge server is already started")
        try:
            config = uvicorn.Config(
                create_submission_app(self._service),
                host=self._bind_host,
                port=self._port,
                access_log=False,
                log_level="warning",
                timeout_graceful_shutdown=self._shutdown_timeout,
            )
        except SystemExit as error:
            address = f"{_url_host(self._bind_host)}:{self._port}"
            raise SubmissionStartupTimeout(
                f"embedded Judge server configuration failed for {address}"
            ) from error
        bound_socket = _bind_server_socket(self._bind_host, self._port)
        actual_port = int(bound_socket.getsockname()[1])
        try:
            server = _ReadyServer(config, self._ready)
        except SystemExit as error:
            bound_socket.close()
            address = f"{_url_host(self._bind_host)}:{actual_port}"
            raise SubmissionStartupTimeout(
                f"embedded Judge server startup failed for {address}"
            ) from error
        self._socket = bound_socket
        self._server = server
        self._thread = threading.Thread(
            target=self._run_server,
            args=(server, bound_socket),
            name="rsi-harness-submissions",
            daemon=False,
        )
        self._thread.start()
        ready = self._ready.wait(timeout=self._startup_timeout)
        if not ready or self._thread_error is not None or not server.started:
            server.should_exit = True
            self._thread.join(timeout=self._startup_timeout)
            bound_socket.close()
            detail = (
                "embedded Judge server failed during startup"
                if self._thread_error is not None
                else "embedded Judge server readiness timed out"
            )
            raise SubmissionStartupTimeout(detail) from self._thread_error
        gateway = _url_host(self._bridge_gateway)
        self._endpoint = JudgeEndpoint(
            url=f"http://{gateway}:{actual_port}",
            bind_host=self._bind_host,
            port=actual_port,
        )
        return self._endpoint

    def stop(self) -> None:
        thread = self._thread
        server = self._server
        if thread is None or server is None:
            return
        drain_error: SubmissionShutdownTimeout | None = None
        try:
            self._service.close(timeout_seconds=self._shutdown_timeout)
        except SubmissionShutdownTimeout as error:
            drain_error = error
        server.should_exit = True
        thread.join(timeout=self._shutdown_timeout + 1.0)
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=1.0)
        self._thread = None
        self._server = None
        self._socket = None
        if thread.is_alive():
            raise SubmissionShutdownTimeout(
                "embedded Judge server thread did not stop before timeout"
            )
        if self._thread_error is not None:
            raise RuntimeError("embedded Judge server failed") from self._thread_error
        if drain_error is not None:
            raise drain_error

    def _run_server(self, server: _ReadyServer, bound_socket: socket.socket) -> None:
        try:
            server.run(sockets=[bound_socket])
        except BaseException as error:
            self._thread_error = error
            self._ready.set()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _bearer_token(authorization: str | None) -> str:
    if authorization is None:
        raise HTTPException(
            status_code=401,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    scheme, separator, token = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


def _verifier_logs(plan: RunPlan, run_id: str, round_id: str) -> Path:
    return plan.paths.logs / "runs" / run_id / plan.task.task_id / "verifier" / round_id


def _verifier_output(plan: RunPlan, run_id: str, round_id: str) -> Path:
    return (
        plan.paths.logs
        / "runs"
        / run_id
        / plan.task.task_id
        / "feedback"
        / f"{round_id}.log"
    )


def _render_feedback(report: SubmissionReport, *, remaining_budget: int | None) -> str:
    reward = (
        None
        if report.error is not None and not report.rewards
        else dict(report.rewards)
    )
    footer = [
        "--- rsi-harness submission ---",
        f"round: {report.round_id}",
        f"status: {report.status.value}",
        "reward: "
        + json.dumps(reward, sort_keys=True, ensure_ascii=False, allow_nan=False),
    ]
    if report.score is not None:
        footer.append(f"score: {report.score}")
    footer.extend(
        (
            f"exit_code: {_json_scalar(report.exit_code)}",
            f"timed_out: {_json_scalar(report.timed_out)}",
            f"duration_seconds: {_json_scalar(report.duration_seconds)}",
            "remaining: unlimited"
            if remaining_budget is None
            else f"remaining: {remaining_budget}",
        )
    )
    if report.error is not None:
        footer.append(f"error: {report.error}")
    output_path = f"/run/rsi-harness/feedback/{report.round_id}.log"
    return f"Full verifier output: {output_path}\n" + "\n".join(footer) + "\n"


def _json_scalar(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _url_host(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def _bind_server_socket(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    bound_socket = socket.socket(family=family)
    bound_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        bound_socket.bind((host, port))
    except OSError as error:
        bound_socket.close()
        address = f"{_url_host(host)}:{port}"
        raise SubmissionStartupTimeout(
            f"embedded Judge server could not bind {address}: {error}"
        ) from error
    bound_socket.set_inheritable(True)
    return bound_socket


__all__ = [
    "EmbeddedJudgeServer",
    "EmptySubmission",
    "JudgeEndpoint",
    "SubmissionClosedError",
    "SubmissionService",
    "SubmissionSessionState",
    "SubmissionShutdownTimeout",
    "SubmissionStartupTimeout",
    "create_submission_app",
]
