"""Bounded GitHub retries; safe metadata lets a durable controller resume later.

Keep this stdlib-only file identical to the public checks/github_retry.py copy.
Operation names must be static labels, never URLs, arguments, or user content.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import math
import re
import sys
import ssl
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from typing import TextIO, TypeVar

SHORT_DELAYS = (1, 5, 10, 30, 60)
LONG_DELAYS = (600, 1200, 3600, 7200)
RECOVERY_WINDOW_SECONDS = 86400
REQUEST_TIMEOUT_SECONDS = 30
DIAGNOSTIC_PREFIX = "RSI_GITHUB_RETRYABLE"
_OPERATION = re.compile(r"[a-z][a-z0-9_-]*(?:\.[a-z][a-z0-9_-]*){1,7}\Z")
_T = TypeVar("_T")
_ACTIVE_RETRY = ContextVar("github_short_retry_active", default=False)


def _number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


class GitHubTransientError(RuntimeError):
    """No request/response content is retained by this recovery signal."""

    def __init__(
        self,
        status: int | None = None,
        *,
        operation: str = "github.request",
        retry_after: float | None = None,
        first_failure_at: float | None = None,
        refresh_credentials: bool = False,
    ) -> None:
        self.status = status if type(status) is int and 100 <= status <= 599 else None
        self.operation = operation if isinstance(operation, str) and _OPERATION.fullmatch(operation) else "github.request"
        self.retry_after = _number(retry_after)
        self.first_failure_at = _number(first_failure_at)
        self.refresh_credentials = bool(refresh_credentials)
        super().__init__("GitHub operation temporarily unavailable")


def _wait_seconds(headers: Mapping[str, object], now: float) -> float | None:
    values = {str(key).lower(): value for key, value in headers.items()}
    waits = []
    retry_after = values.get("retry-after")
    numeric = _number(retry_after)
    if numeric is not None:
        waits.append(numeric)
    elif isinstance(retry_after, str):
        try:
            timestamp = email.utils.parsedate_to_datetime(retry_after).timestamp()
            waits.append(max(0, timestamp - now))
        except (ValueError, TypeError, OverflowError):
            pass
    if str(values.get("x-ratelimit-remaining")) == "0":
        reset = _number(values.get("x-ratelimit-reset"))
        if reset is not None:
            waits.append(max(0, reset - now))
    return max(waits) if waits else None


def transient_http_error(
    status: int,
    headers: Mapping[str, object] | None,
    body: bytes | str | object,
    *,
    operation: str,
    clock: Callable[[], float] | None = None,
) -> GitHubTransientError | None:
    """Classify explicit outages/rate limits, never a blanket 401/403/404."""
    now = (clock or time.time)()
    headers = headers or {}
    values = {str(key).lower(): value for key, value in headers.items()}
    try:
        payload = json.loads(body) if isinstance(body, (bytes, str)) else body
    except (ValueError, UnicodeError):
        payload = None
    message = payload.get("message", "") if isinstance(payload, dict) else ""
    message = message.casefold() if isinstance(message, str) else ""
    rate_limited = (
        "retry-after" in values
        or str(values.get("x-ratelimit-remaining")) == "0"
        or any(text in message for text in ("rate limit", "secondary rate", "abuse detection"))
    )
    expired = status in (401, 403) and any(
        text in message for text in ("token expired", "token has expired", "expired token", "expiration time")
    )
    graphql_transient = False
    if status == 200 and isinstance(payload, dict) and isinstance(payload.get("errors"), list):
        errors = payload["errors"]
        # Mixed authorization/schema and transient errors are not safe to retry.
        types = []
        for error in errors:
            if not isinstance(error, dict):
                types.append(None)
                continue
            extension = error.get("extensions")
            types.append(error.get("type") or (extension.get("code") if isinstance(extension, dict) else None))
        graphql_transient = bool(types) and all(
            item in {"RATE_LIMITED", "INTERNAL", "INTERNAL_SERVER_ERROR", "SERVICE_UNAVAILABLE"}
            for item in types
        )
    if type(status) is not int or not (
        500 <= status <= 599 or status in (408, 429)
        or status == 403 and rate_limited or expired or graphql_transient
    ):
        return None
    return GitHubTransientError(
        status, operation=operation, retry_after=_wait_seconds(headers, now),
        refresh_credentials=expired,
    )


def transport_error(error: BaseException, *, operation: str) -> GitHubTransientError | None:
    if isinstance(error, GitHubTransientError):
        return error
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, ssl.SSLCertVerificationError):
        return None
    if isinstance(error, (OSError, urllib.error.URLError, http.client.IncompleteRead, http.client.RemoteDisconnected)) and not isinstance(error, urllib.error.HTTPError):
        return GitHubTransientError(operation=operation)
    return None


def retry_call(
    call: Callable[[], _T],
    *,
    delays: Sequence[float] = SHORT_DELAYS,
    sleep: Callable[[float], object] | None = None,
    clock: Callable[[], float] | None = None,
) -> _T:
    """Short retries only. Long waits/credential refresh are controller work."""
    if _ACTIVE_RETRY.get():
        return call()
    token = _ACTIVE_RETRY.set(True)
    try:
        return _retry_call(call, delays=delays, sleep=sleep, clock=clock)
    finally:
        _ACTIVE_RETRY.reset(token)


def _retry_call(
    call: Callable[[], _T], *, delays: Sequence[float],
    sleep: Callable[[float], object] | None, clock: Callable[[], float] | None,
) -> _T:
    sleep = sleep or time.sleep
    clock = clock or time.time
    first_failure_at = None
    for attempt in range(len(delays) + 1):
        try:
            return call()
        except GitHubTransientError as error:
            first_failure_at = min(
                value for value in (first_failure_at, error.first_failure_at, clock())
                if value is not None
            )
            error.first_failure_at = first_failure_at
            if attempt == len(delays) or error.refresh_credentials:
                raise
            wait = max(delays[attempt], error.retry_after or 0)
            if wait > max(SHORT_DELAYS) or clock() + wait >= first_failure_at + RECOVERY_WINDOW_SECONDS:
                raise
            sleep(wait)
    raise AssertionError("unreachable retry state")


def emit_retry_diagnostic(error: GitHubTransientError, *, stream: TextIO | None = None) -> None:
    value = {
        "schema": 1,
        "status": error.status,
        "operation": error.operation,
        "first_failure_at": error.first_failure_at if error.first_failure_at is not None else time.time(),
        "retry_after": error.retry_after,
        "refresh_credentials": error.refresh_credentials,
    }
    (stream or sys.stderr).write(DIAGNOSTIC_PREFIX + " " + json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
