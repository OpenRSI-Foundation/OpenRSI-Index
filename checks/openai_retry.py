"""Bounded OpenAI retries, separate from durable GitHub outage recovery."""

from __future__ import annotations

import asyncio
import email.utils
import math
import random
import re
import sys
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIConnectionError, APIStatusError

INITIAL_DELAY_SECONDS = 5
MAX_DELAY_SECONDS = 300
MAX_RETRIES = 12
RETRY_WINDOW_SECONDS = 1800
_PERMANENT_QUOTA_CODES = {
    "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
    "usage_limit_reached",
}
_T = TypeVar("_T")


def _seconds(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _server_delay(error: APIStatusError) -> float:
    headers = error.response.headers
    waits = [0.0]
    milliseconds = _seconds(headers.get("retry-after-ms"))
    if milliseconds is not None:
        waits.append(milliseconds / 1000)
    value = headers.get("retry-after")
    seconds = _seconds(value)
    if seconds is not None:
        waits.append(seconds)
    elif value:
        try:
            waits.append(max(0, email.utils.parsedate_to_datetime(value).timestamp() - time.time()))
        except (TypeError, ValueError, OverflowError):
            pass
    # Some token-limit responses supply the cooldown only in the error message.
    hint = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s\b", error.message, re.I)
    if hint:
        seconds = _seconds(hint.group(1))
        if seconds is not None:
            waits.append(seconds)
    return max(waits)


def _retry_wait(error: APIConnectionError | APIStatusError, attempt: int, deadline: float) -> float | None:
    status = getattr(error, "status_code", None)
    if isinstance(error, APIStatusError):
        # Quota/billing errors also use 429, but waiting cannot repair them.
        if status == 429:
            if error.code in _PERMANENT_QUOTA_CODES or error.type in _PERMANENT_QUOTA_CODES:
                return None
        elif status not in (408, 409) and not 500 <= status <= 599:
            return None
    if attempt >= MAX_RETRIES:
        return None
    backoff = min(INITIAL_DELAY_SECONDS * 2**attempt, MAX_DELAY_SECONDS)
    server_delay = _server_delay(error) if isinstance(error, APIStatusError) else 0
    wait = max(backoff, server_delay) + random.uniform(0, 1)
    if time.monotonic() + wait >= deadline:
        return None
    print(
        f"OpenAI temporary error (status={status}, code={getattr(error, 'code', None)}); "
        f"retry {attempt + 1}/{MAX_RETRIES} in {wait:.2f}s.",
        file=sys.stderr, flush=True,
    )
    return wait


def retry_openai(call: Callable[[], _T]) -> _T:
    """Retry transient API errors without changing the request or its result."""
    deadline = time.monotonic() + RETRY_WINDOW_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        try:
            return call()
        except (APIConnectionError, APIStatusError) as error:
            wait = _retry_wait(error, attempt, deadline)
            if wait is None:
                raise
            time.sleep(wait)
            if time.monotonic() >= deadline:
                raise
    raise AssertionError("unreachable retry state")


async def async_retry_openai(call: Callable[[], Awaitable[_T]]) -> _T:
    """Async equivalent; never block the event loop during backoff."""
    deadline = time.monotonic() + RETRY_WINDOW_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await call()
        except (APIConnectionError, APIStatusError) as error:
            wait = _retry_wait(error, attempt, deadline)
            if wait is None:
                raise
            await asyncio.sleep(wait)
            if time.monotonic() >= deadline:
                raise
    raise AssertionError("unreachable retry state")
