"""HTTP routes of the Runtime Task Intake API (``ale.task.intake.v1``)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import FormData, Headers, UploadFile

from rsi_harness.intake.bundle import BundleTooLarge, copy_stream_checked
from rsi_harness.intake.protocol import (
    HEADER_DISPATCH_ID,
    HEADER_IDEMPOTENCY_KEY,
    HEADER_SIGNATURE,
    HEADER_TIMESTAMP,
    PROTOCOL,
    SERVICE_NAME,
    IntakeDispatch,
    IntakeStatus,
    bearer_matches,
    verify_dispatch_signature,
)
from rsi_harness.intake.service import (
    IntakeConflictError,
    IntakeNotFoundError,
    IntakeService,
    IntakeValidationError,
)

MAX_METADATA_BYTES = 4 * 1024 * 1024


def _version() -> str:
    try:
        return version("rsi-harness")
    except PackageNotFoundError:
        return "0"


def _error(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def _validation_detail(error: ValidationError) -> str:
    first = error.errors()[0]
    where = ".".join(str(part) for part in first.get("loc", ())) or "payload"
    return f"invalid dispatch: {where}: {first.get('msg', 'invalid value')}"


class _Rejected(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


def create_app(service: IntakeService) -> FastAPI:
    settings = service.settings

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service.start()
        try:
            yield
        finally:
            await run_in_threadpool(service.stop)

    app = FastAPI(
        title="RSI Runtime Task Intake",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "protocol": PROTOCOL,
            "version": _version(),
        }

    app.get("/healthz")(health)
    app.get("/v1/health")(health)

    def _require_query_bearer(request: Request) -> JSONResponse | None:
        if not bearer_matches(
            request.headers.get("authorization"), settings.query_credential
        ):
            return _error(401, "missing or invalid bearer token")
        return None

    def _save_upload(upload: UploadFile) -> tuple[Path, str]:
        target = settings.state_dir / "uploads" / uuid.uuid4().hex
        upload.file.seek(0)
        try:
            digest, _ = copy_stream_checked(
                upload.file, target, max_bytes=settings.max_bundle_bytes
            )
        except BundleTooLarge as error:
            raise _Rejected(413, "uploaded bundle exceeds size limit") from error
        return target, digest

    def _declared_length(headers: Headers, limit: int, what: str) -> int:
        raw = headers.get("content-length")
        if raw is None:
            raise _Rejected(411, f"{what} needs a Content-Length")
        if not raw.isdigit():
            raise _Rejected(422, "invalid Content-Length")
        if int(raw) > limit:
            raise _Rejected(413, f"{what} is too large")
        return int(raw)

    async def _read_capped(request: Request, limit: int) -> bytes:
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > limit:
                raise _Rejected(413, "dispatch metadata is too large")
        return bytes(body)

    @app.post("/v1/intake/tasks")
    async def submit(request: Request) -> JSONResponse:
        headers = request.headers
        if settings.token and not bearer_matches(
            headers.get("authorization"), settings.token
        ):
            return _error(401, "missing or invalid bearer token")

        content_type = headers.get("content-type", "").lower()
        upload: Path | None = None
        forms: list[FormData] = []
        try:
            if content_type.startswith("application/json"):
                raw_length = headers.get("content-length")
                if raw_length is not None:
                    _declared_length(headers, MAX_METADATA_BYTES, "dispatch metadata")
                # Chunked bodies have no length: read with a hard cap.
                metadata = await _read_capped(request, MAX_METADATA_BYTES)
                upload_sha = None
            elif content_type.startswith("multipart/form-data"):
                # Refuse oversized uploads before Starlette spools any of it.
                _declared_length(
                    headers,
                    settings.max_bundle_bytes + MAX_METADATA_BYTES,
                    "multipart dispatch",
                )
                form = await request.form(max_files=1, max_fields=16)
                forms.append(form)
                part = form.get("metadata")
                if isinstance(part, UploadFile):
                    metadata = await part.read()
                elif isinstance(part, str):
                    metadata = part.encode("utf-8")
                else:
                    raise _Rejected(422, "multipart dispatch needs a 'metadata' part")
                bundle = form.get("bundle")
                if not isinstance(bundle, UploadFile):
                    raise _Rejected(
                        422, "multipart dispatch needs a 'bundle' file part"
                    )
                upload, upload_sha = await run_in_threadpool(_save_upload, bundle)
            else:
                raise _Rejected(
                    422, "Content-Type must be application/json or multipart/form-data"
                )
            if len(metadata) > MAX_METADATA_BYTES:
                raise _Rejected(413, "dispatch metadata is too large")

            if settings.secret:
                ok, reason = verify_dispatch_signature(
                    secret=settings.secret,
                    signature_header=headers.get(HEADER_SIGNATURE, ""),
                    timestamp_header=headers.get(HEADER_TIMESTAMP, ""),
                    idempotency_key=headers.get(HEADER_IDEMPOTENCY_KEY, ""),
                    metadata_bytes=metadata,
                    max_skew_sec=settings.max_skew_sec,
                )
                if not ok:
                    raise _Rejected(401, reason)

            try:
                dispatch = IntakeDispatch.model_validate(json.loads(metadata))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise _Rejected(422, f"dispatch metadata is not valid JSON: {error}")
            except ValidationError as error:
                raise _Rejected(422, _validation_detail(error))

            header_dispatch = headers.get(HEADER_DISPATCH_ID, "").strip()
            if header_dispatch and header_dispatch != dispatch.dispatch_id:
                raise _Rejected(
                    422, f"{HEADER_DISPATCH_ID} header disagrees with dispatch_id"
                )
            header_key = headers.get(HEADER_IDEMPOTENCY_KEY, "").strip()
            if header_key and header_key != dispatch.idempotency_key:
                raise _Rejected(
                    422,
                    f"{HEADER_IDEMPOTENCY_KEY} header disagrees with idempotency_key",
                )
            if upload is not None and upload_sha != dispatch.artifact.sha256:
                raise _Rejected(
                    422, "uploaded bundle sha256 does not match artifact.sha256"
                )

            try:
                record, created = await run_in_threadpool(
                    service.submit, dispatch, upload
                )
            except IntakeConflictError as error:
                raise _Rejected(409, str(error))
            except IntakeValidationError as error:
                raise _Rejected(422, str(error))
            upload = None  # the service owns (or removed) the file now
            return JSONResponse(
                record.receipt(created=created), status_code=202 if created else 200
            )
        except _Rejected as rejected:
            return _error(rejected.status, rejected.detail)
        finally:
            if upload is not None:
                upload.unlink(missing_ok=True)
            for form in forms:
                await form.close()

    @app.get("/v1/intake/tasks")
    def list_tasks(
        request: Request,
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> JSONResponse:
        if denied := _require_query_bearer(request):
            return denied
        if status and status not in {item.value for item in IntakeStatus}:
            return _error(422, f"unknown status {status!r}")
        records = service.list(status=status, limit=limit)
        return JSONResponse({"tasks": [record.view() for record in records]})

    @app.get("/v1/intake/tasks/{ref}")
    def get_task(request: Request, ref: str) -> JSONResponse:
        if denied := _require_query_bearer(request):
            return denied
        try:
            return JSONResponse(service.get(ref).view())
        except IntakeNotFoundError as error:
            return _error(404, str(error))

    @app.post("/v1/intake/tasks/{ref}/retry")
    def retry_task(request: Request, ref: str) -> JSONResponse:
        if denied := _require_query_bearer(request):
            return denied
        try:
            record = service.retry(ref)
        except IntakeNotFoundError as error:
            return _error(404, str(error))
        except IntakeConflictError as error:
            return _error(409, str(error))
        return JSONResponse(record.view(), status_code=202)

    return app
