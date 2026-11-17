"""Wire contract of the Runtime Task Intake API (``ale.task.intake.v1``).

An upstream task brewer (``ale-brew-science``) pushes reviewed task bundles to
an execution runtime over this contract. This module holds the RSI-Harness
implementation: the dispatch, receipt and status models, the ``X-ALE-*``
headers, the dispatch HMAC check and the webhook signature.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL = "ale.task.intake.v1"
RECEIPT_SCHEMA = "ale.task.intake.receipt.v1"
STATUS_SCHEMA = "ale.task.intake.status.v1"
SERVICE_NAME = "rsi-intake"
RUNTIME_NAME = "rsi"
DEFAULT_PORT = 7472

HEADER_DISPATCH_ID = "X-ALE-Dispatch-Id"
HEADER_IDEMPOTENCY_KEY = "X-ALE-Idempotency-Key"
HEADER_TIMESTAMP = "X-ALE-Timestamp"
HEADER_SIGNATURE = "X-ALE-Signature"

# The Brewer stores at most this much of a webhook's detail.
MAX_DETAIL_CHARS = 4000

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class ExecutionMode(StrEnum):
    """What the runtime does with an accepted task.

    lint      compile the staged task with the Harness task compiler
    prepare   lint, then a cluster dry run when a cluster is configured
    validate  run the task's own solution (the oracle Agent) and check it scores
    run       validate, then a real Agent run as OpenRSI-X runs its own tasks
    """

    LINT = "lint"
    PREPARE = "prepare"
    VALIDATE = "validate"
    RUN = "run"


class IntakeStatus(StrEnum):
    """The only status values the Brewer webhook accepts."""

    ACKNOWLEDGED = "acknowledged"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def normalize_sha256(value: str) -> str:
    """64 lowercase hex digits; a ``sha256:`` prefix is accepted and dropped."""
    cleaned = value.strip().lower()
    if cleaned.startswith("sha256:"):
        cleaned = cleaned[len("sha256:") :]
    if not _SHA256_HEX.fullmatch(cleaned):
        raise ValueError("must be 64 hex digits (optionally prefixed sha256:)")
    return cleaned


class IntakeTaskInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_name: str = Field(min_length=1, max_length=255)
    task_repo: str = ""
    domain: str = ""
    title: str = ""
    summary: str = ""
    submission_id: str = ""
    job_id: str = ""
    source_type: str = ""
    source_skill: str = ""
    source_ref: str = ""


class IntakeArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = ""
    name: str = ""
    sha256: str
    size_bytes: int | None = Field(default=None, ge=0)
    download_url: str = ""
    offline_report: dict[str, Any] | None = None
    tree_manifest: dict[str, Any] | None = None

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        return normalize_sha256(value)


class IntakeDispatch(BaseModel):
    """The Brewer's dispatch payload (``ale_outbox.build_ale_dispatch_payload``)."""

    model_config = ConfigDict(extra="allow")

    schema_version: str
    dispatch_id: str = Field(min_length=1, max_length=255)
    idempotency_key: str = Field(min_length=1, max_length=255)
    submitted_at: str = ""
    task: IntakeTaskInfo
    author: dict[str, Any] = Field(default_factory=dict)
    review: dict[str, Any] = Field(default_factory=dict)
    artifact: IntakeArtifact
    status_webhook_url: str | None = None
    execution_mode: ExecutionMode | None = None

    @field_validator("schema_version")
    @classmethod
    def _protocol(cls, value: str) -> str:
        if value != PROTOCOL:
            raise ValueError(f"schema_version must be {PROTOCOL!r}")
        return value


def verify_dispatch_signature(
    *,
    secret: str,
    signature_header: str,
    timestamp_header: str,
    idempotency_key: str,
    metadata_bytes: bytes,
    max_skew_sec: int = 300,
    now: float | None = None,
) -> tuple[bool, str]:
    """Check the Brewer's ``X-ALE-Signature`` over the raw metadata bytes.

    ``sha256=HMAC_SHA256(secret, f"{ts}.{idempotency_key}.".encode() + bytes)``.
    """
    if not secret:
        return False, "intake secret not configured"
    if not idempotency_key.strip():
        return False, f"missing {HEADER_IDEMPOTENCY_KEY}"
    try:
        ts = int(timestamp_header.strip())
    except ValueError:
        return False, f"invalid {HEADER_TIMESTAMP}"
    current = time.time() if now is None else now
    if max_skew_sec > 0 and abs(current - ts) > max_skew_sec:
        return False, f"stale {HEADER_TIMESTAMP}"
    raw = signature_header.strip()
    provided = (
        raw[7:].strip().lower() if raw.lower().startswith("sha256=") else raw.lower()
    )
    message = (
        f"{timestamp_header.strip()}.{idempotency_key.strip()}.".encode()
        + metadata_bytes
    )
    expected = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    if not provided or not hmac.compare_digest(provided, expected):
        return False, f"invalid {HEADER_SIGNATURE}"
    return True, ""


def sign_dispatch(
    secret: str, timestamp: str, idempotency_key: str, body: bytes
) -> str:
    """The Brewer's side of :func:`verify_dispatch_signature` (tests, tooling)."""
    message = f"{timestamp}.{idempotency_key}.".encode() + body
    return "sha256=" + hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def sign_webhook(secret: str, raw_json_bytes: bytes) -> str:
    digest = hmac.new(secret.encode(), raw_json_bytes, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def bearer_matches(authorization: str | None, credential: str) -> bool:
    if not credential or not authorization:
        return False
    scheme, _, value = authorization.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(value.strip().encode(), credential.encode())
