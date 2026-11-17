"""The intake queue: accept dispatches, fetch and verify bundles, run tasks,
report status back to the Brewer."""

from __future__ import annotations

import datetime as dt
import json
import queue
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import requests

from rsi_harness.intake.bundle import (
    BundleError,
    BundleTooLarge,
    copy_stream_checked,
    reset_dir,
    unpack_bundle,
    verify_unpacked,
)
from rsi_harness.intake.executor import Executor
from rsi_harness.intake.protocol import (
    MAX_DETAIL_CHARS,
    ExecutionMode,
    IntakeDispatch,
    IntakeStatus,
    sign_webhook,
)
from rsi_harness.intake.staging import canonical_task_id, stage_task
from rsi_harness.intake.store import IntakeRecord, IntakeStore

DEFAULT_MAX_BUNDLE_BYTES = 512 * 1024 * 1024
WEBHOOK_PATH = "/api/v1/webhooks/ale-status"
MAX_REDIRECTS = 5
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class IntakeValidationError(Exception):
    """HTTP 422."""


class IntakeConflictError(Exception):
    """HTTP 409."""


class IntakeNotFoundError(Exception):
    """HTTP 404."""


@dataclass
class IntakeSettings:
    secret: str = ""
    token: str = ""
    webhook_url: str = ""
    webhook_secret: str = ""
    allowed_origins: tuple[str, ...] = ()
    state_dir: Path = Path(".rsi-harness/intake")
    default_mode: ExecutionMode = ExecutionMode.RUN
    concurrency: int = 1
    max_skew_sec: int = 300
    max_bundle_bytes: int = DEFAULT_MAX_BUNDLE_BYTES
    webhook_attempts: int = 3
    webhook_backoff_sec: float = 1.0
    download_timeout_sec: float = 300.0
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def query_credential(self) -> str:
        return self.token or self.secret

    @property
    def effective_webhook_secret(self) -> str:
        return self.webhook_secret or self.secret or self.token


def now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _origin(url: str) -> str:
    parts = urlsplit(url.strip())
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}"


def _safe_name(raw: str, fallback: str) -> str:
    name = _SAFE_NAME.sub("_", Path(raw or "").name).strip("._")
    return name[:120] or fallback


class IntakeService:
    def __init__(
        self,
        settings: IntakeSettings,
        executor: Executor,
        *,
        http: requests.Session | None = None,
    ) -> None:
        self.settings = settings
        self.executor = executor
        self.http = http or requests.Session()
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir = settings.state_dir / "tasks"
        self.store = IntakeStore(settings.state_dir / "intake.db")
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._submit_lock = threading.Lock()
        self._stopping = threading.Event()

    # --- lifecycle ---

    def start(self) -> None:
        if self._workers:
            return
        for record in self.store.unfinished():
            self._queue.put(record.id)
        for index in range(max(1, self.settings.concurrency)):
            worker = threading.Thread(
                target=self._worker_loop, name=f"rsi-intake-{index}", daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def stop(self, timeout: float | None = 5.0) -> None:
        """Stop taking work; close the store only once every worker is out.

        A worker still inside a long run keeps the store open (its thread is
        a daemon); its record stays RUNNING and is queued again on restart.
        """
        self._stopping.set()
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout)
        alive = [worker for worker in self._workers if worker.is_alive()]
        self._workers.clear()
        if not alive:
            self.store.close()

    def wait_idle(self) -> None:
        """Block until every queued task is processed (tests, tooling)."""
        self._queue.join()

    # --- URLs ---

    def _check_url(self, url: str, what: str) -> str:
        parts = urlsplit(url.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise IntakeValidationError(f"{what} must be an http(s) URL")
        allowed = {_origin(item) for item in self.settings.allowed_origins}
        if allowed and _origin(url) not in allowed:
            raise IntakeValidationError(f"{what} origin is not in the allowed origins")
        return url.strip()

    def _webhook_url(self, dispatch: IntakeDispatch) -> str | None:
        """The dispatch's own URL, else the configured one, else ``auto``.

        A dispatch's ``status_webhook_url`` overrides the server default
        (spec 4.2); it must still pass the allowed origins.
        """
        if dispatch.status_webhook_url and dispatch.status_webhook_url.strip():
            return self._check_url(dispatch.status_webhook_url, "status_webhook_url")
        configured = self.settings.webhook_url.strip()
        if configured and configured.lower() != "auto":
            return configured
        if configured.lower() == "auto" and dispatch.artifact.download_url:
            parts = urlsplit(dispatch.artifact.download_url.strip())
            if (
                "/api/v1/outbox/bundles/" in parts.path
                and parts.scheme
                and parts.netloc
            ):
                return f"{parts.scheme}://{parts.netloc}{WEBHOOK_PATH}"
        return None

    # --- API operations ---

    def submit(
        self, dispatch: IntakeDispatch, uploaded_bundle: Path | None = None
    ) -> tuple[IntakeRecord, bool]:
        """Record a dispatch; ``(record, created)``. Idempotent on its key."""
        expected_sha = dispatch.artifact.sha256
        with self._submit_lock:
            existing = self.store.get_by_idempotency_key(dispatch.idempotency_key)
            if existing is not None:
                if uploaded_bundle is not None:
                    uploaded_bundle.unlink(missing_ok=True)
                if existing.artifact_sha256 != expected_sha:
                    raise IntakeConflictError(
                        "idempotency_key was already used for a different "
                        "artifact.sha256"
                    )
                return existing, False

            if uploaded_bundle is None:
                if not dispatch.artifact.download_url.strip():
                    raise IntakeValidationError(
                        "artifact.download_url is required when no bundle is uploaded"
                    )
                self._check_url(dispatch.artifact.download_url, "artifact.download_url")
            webhook_url = self._webhook_url(dispatch)

            intake_id = uuid.uuid4().hex
            bundle_path: str | None = None
            if uploaded_bundle is not None:
                target = (
                    self.tasks_dir
                    / intake_id
                    / "bundle"
                    / _safe_name(dispatch.artifact.name, "bundle")
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(uploaded_bundle, target)
                bundle_path = str(target)

            stamp = now_iso()
            record = IntakeRecord(
                id=intake_id,
                dispatch_id=dispatch.dispatch_id,
                idempotency_key=dispatch.idempotency_key,
                task_id=canonical_task_id(
                    dispatch.task.task_name,
                    task_repo=dispatch.task.task_repo,
                    domain=dispatch.task.domain,
                ),
                status=IntakeStatus.ACKNOWLEDGED,
                execution_mode=dispatch.execution_mode or self.settings.default_mode,
                artifact_sha256=expected_sha,
                request=dispatch,
                bundle_path=bundle_path,
                webhook_url=webhook_url,
                detail="acknowledged",
                received_at=stamp,
                updated_at=stamp,
            )
            self.store.insert(record)
        self._queue.put(record.id)
        return record, True

    def get(self, ref: str) -> IntakeRecord:
        record = self.store.get(ref)
        if record is None:
            raise IntakeNotFoundError(f"unknown intake task {ref!r}")
        return record

    def list(self, *, status: str | None, limit: int) -> list[IntakeRecord]:
        return self.store.list(status=status, limit=limit)

    def retry(self, ref: str) -> IntakeRecord:
        with self._submit_lock:
            record = self.get(ref)
            if record.status is not IntakeStatus.FAILED:
                raise IntakeConflictError(
                    f"intake task {record.id} cannot be retried from status "
                    f"{record.status.value!r}"
                )
            record.status = IntakeStatus.ACKNOWLEDGED
            record.detail = "re-queued for retry"
            record.completed_at = None
            record.updated_at = now_iso()
            self.store.put(record)
        self._queue.put(record.id)
        return record

    # --- worker ---

    def _worker_loop(self) -> None:
        while True:
            intake_id = self._queue.get()
            try:
                if intake_id is None:
                    return
                if self._stopping.is_set():
                    continue  # left queued in the store; resumed on restart
                self._process(intake_id)
            finally:
                self._queue.task_done()

    def _save(self, record: IntakeRecord, status: IntakeStatus, detail: str) -> None:
        record.status = status
        record.detail = detail[:MAX_DETAIL_CHARS]
        record.updated_at = now_iso()
        if status in (IntakeStatus.COMPLETED, IntakeStatus.FAILED):
            record.completed_at = record.updated_at
        self.store.put(record)

    def _download(self, record: IntakeRecord, task_root: Path) -> Path:
        """Fetch the bundle; every redirect hop must pass the allowed origins.

        ``Accept-Encoding: identity`` keeps the bytes on disk the file the
        Brewer hashed (no transparent gzip of a ``.tar.gz``).
        """
        artifact = record.request.artifact
        url = self._check_url(artifact.download_url, "artifact.download_url")
        target = task_root / "bundle" / _safe_name(artifact.name, "bundle")
        try:
            for _ in range(MAX_REDIRECTS + 1):
                headers = {"Accept-Encoding": "identity"}
                if "signature=" not in urlsplit(url).query and self.settings.token:
                    headers["Authorization"] = f"Bearer {self.settings.token}"
                with self.http.get(
                    url,
                    headers=headers,
                    stream=True,
                    allow_redirects=False,
                    timeout=(15, self.settings.download_timeout_sec),
                ) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        location = response.headers.get("Location", "")
                        url = self._check_url(urljoin(url, location), "bundle redirect")
                        continue
                    if response.status_code != 200:
                        raise BundleError(
                            "bundle download failed with HTTP "
                            f"{response.status_code}: {response.text[:300]}"
                        )
                    digest, _ = copy_stream_checked(
                        response.raw, target, max_bytes=self.settings.max_bundle_bytes
                    )
                    break
            else:
                raise BundleError(
                    f"bundle download redirected more than {MAX_REDIRECTS} times"
                )
        except BundleError:
            target.unlink(missing_ok=True)
            raise
        except IntakeValidationError as error:
            raise BundleError(str(error)) from error
        except Exception as error:  # requests and urllib3 stream errors alike
            target.unlink(missing_ok=True)
            raise BundleError(f"bundle download failed: {error!r}") from error
        if digest != record.artifact_sha256:
            target.unlink(missing_ok=True)
            raise BundleError(
                f"downloaded bundle sha256 {digest} does not match artifact.sha256"
            )
        return target

    def _process(self, intake_id: str) -> None:
        record = self.store.get(intake_id)
        if record is None or record.status not in (
            IntakeStatus.ACKNOWLEDGED,
            IntakeStatus.RUNNING,
        ):
            return
        record.attempt_count += 1
        record.started_at = now_iso()
        self._save(record, IntakeStatus.RUNNING, "running")
        self._emit_webhook(record)

        task_root = self.tasks_dir / record.id
        attempt_dir = task_root / f"attempt-{record.attempt_count}"
        try:
            bundle = Path(record.bundle_path) if record.bundle_path else None
            if bundle is None or not bundle.is_file():
                bundle = self._download(record, task_root)
                record.bundle_path = str(bundle)
                self.store.put(record)
            reset_dir(attempt_dir)
            unpack_bundle(bundle, attempt_dir / "unpacked")
            verified = verify_unpacked(attempt_dir / "unpacked")
            info = record.request.task
            staged = stage_task(
                verified.task_dir,
                attempt_dir / "task",
                fallback_name=info.task_name,
                task_repo=info.task_repo,
                domain=info.domain,
            )
            record.task_id = staged.task_id
            base = {
                "task_id": staged.task_id,
                "sealed": verified.sealed,
                "task_digest": verified.task_digest,
                "adapted": staged.adapted,
                "adaptation_notices": list(staged.notices),
                "mode": record.execution_mode.value,
            }
            record.result = base
            self.store.put(record)
            outcome = self.executor.execute(record.execution_mode, staged, attempt_dir)
            record.result = {**base, **outcome.result}
            status = IntakeStatus.COMPLETED if outcome.ok else IntakeStatus.FAILED
            self._save(record, status, outcome.detail)
        except BundleError as error:
            self._save(record, IntakeStatus.FAILED, str(error))
        except OSError as error:
            self._save(
                record,
                IntakeStatus.FAILED,
                f"bundle could not be unpacked or staged: {error}",
            )
        except Exception as error:  # never let one task stop the worker
            self._save(record, IntakeStatus.FAILED, f"internal error: {error!r}")
        self._emit_webhook(record)

    # --- webhook ---

    def _emit_webhook(self, record: IntakeRecord) -> None:
        if not record.webhook_url:
            return
        payload = {
            "dispatch_id": record.dispatch_id,
            "idempotency_key": record.idempotency_key,
            "status": record.status.value,
            "ale_task_id": record.task_id,
            "detail": record.detail[:MAX_DETAIL_CHARS],
            "response_body": record.view(),
        }
        body = json.dumps(payload, separators=(",", ":"), default=str).encode()
        headers = {"Content-Type": "application/json"}
        secret = self.settings.effective_webhook_secret
        if secret:
            headers["X-ALE-Signature"] = sign_webhook(secret, body)
        error = ""
        for attempt in range(max(1, self.settings.webhook_attempts)):
            try:
                response = self.http.post(
                    record.webhook_url, data=body, headers=headers, timeout=15
                )
                if 200 <= response.status_code < 300:
                    error = ""
                    break
                error = f"HTTP {response.status_code}: {response.text[:300]}"
            except requests.RequestException as exc:
                error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < self.settings.webhook_attempts:
                time.sleep(self.settings.webhook_backoff_sec * (2**attempt))
        # Update only the webhook fields: a retry() may have changed the
        # record while this delivery was in flight.
        if error:
            record.webhook_last_error = error[:1000]
        else:
            record.webhook_delivered_at = now_iso()
            record.webhook_last_error = ""
        self.store.update_webhook(
            record.id, record.webhook_delivered_at, record.webhook_last_error
        )


__all__ = [
    "BundleTooLarge",
    "IntakeConflictError",
    "IntakeNotFoundError",
    "IntakeService",
    "IntakeSettings",
    "IntakeValidationError",
    "now_iso",
]
