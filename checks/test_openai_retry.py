import asyncio
from email.utils import formatdate
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from checks import openai_retry


def status_error(status=429, code=None, headers=None, message="Rate limited", error_type=None):
    response = httpx.Response(
        status, headers=headers, request=httpx.Request("POST", "https://api.openai.com/v1/responses")
    )
    return APIStatusError(
        message, response=response, body={"code": code, "type": error_type, "message": message}
    )


@pytest.fixture(params=["sync", "async"])
def run_retry(request):
    def run(call):
        if request.param == "sync":
            return openai_retry.retry_openai(call)

        async def async_call():
            return call()

        return asyncio.run(openai_retry.async_retry_openai(async_call))

    return run


@pytest.fixture
def clock(monkeypatch):
    state = SimpleNamespace(now=0.0, waits=[], oversleep=0.0)

    def sleep(delay):
        state.waits.append(delay)
        state.now += delay + state.oversleep

    async def async_sleep(delay):
        sleep(delay)

    monkeypatch.setattr(openai_retry.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(openai_retry.time, "time", lambda: 1_700_000_000 + state.now)
    monkeypatch.setattr(openai_retry.time, "sleep", sleep)
    monkeypatch.setattr(openai_retry.asyncio, "sleep", async_sleep)
    monkeypatch.setattr(openai_retry.random, "uniform", lambda low, high: 0.25)
    return state


@pytest.mark.parametrize(
    "error",
    [status_error(status) for status in (408, 409, 429, 500, 502, 503, 504)]
    + [status_error(code=code) for code in ("rate_limit_exceeded", "unknown_error")]
    + [kind(request=httpx.Request("POST", "https://api.openai.com"))
       for kind in (APIConnectionError, APITimeoutError)],
)
def test_transient_failure_recovers(run_retry, clock, error):
    outcomes = iter([error, "response"])

    def call():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    assert run_retry(call) == "response"
    assert clock.waits == [5.25]


def test_backoff_doubles_and_caps_with_jitter(run_retry, clock):
    attempts = []

    def call():
        attempts.append(clock.now)
        if len(attempts) <= 8:
            raise status_error()
        return "response"

    assert run_retry(call) == "response"
    assert clock.waits == [5.25, 10.25, 20.25, 40.25, 80.25, 160.25, 300.25, 300.25]
    assert len(attempts) == 9


@pytest.mark.parametrize(
    "error",
    [status_error(status) for status in (400, 401, 403)]
    + [status_error(**{field: code})
       for field in ("code", "error_type")
       for code in ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "usage_limit_reached")]
    + [ValueError("Invalid local input")],
)
def test_permanent_failure_is_raised_without_retry(run_retry, clock, error):
    attempts = []

    def call():
        attempts.append(clock.now)
        raise error

    with pytest.raises(type(error)) as caught:
        run_retry(call)
    assert caught.value is error
    assert len(attempts) == 1
    assert clock.waits == []


@pytest.mark.parametrize(
    "headers,message,minimum",
    [
        ({"retry-after": "12.5"}, "Rate limited", 12.5),
        ({"retry-after-ms": "10798"}, "Rate limited", 10.798),
        ({"retry-after": formatdate(1_700_000_090, usegmt=True)}, "Rate limited", 90),
        ({"retry-after": "600"}, "Rate limited", 600),
        ({"retry-after": "invalid"}, "Please try again in 10.798s.", 10.798),
        ({}, "Please try again in 10.798s.", 10.798),
    ],
)
def test_server_hint_is_a_minimum_wait(run_retry, clock, headers, message, minimum):
    errors = [status_error(headers=headers, message=message)]

    def call():
        if errors:
            raise errors.pop()
        return "response"

    assert run_retry(call) == "response"
    assert len(clock.waits) == 1
    assert minimum <= clock.waits[0] <= minimum + 1


def test_retry_count_is_bounded(run_retry, clock, monkeypatch):
    monkeypatch.setattr(openai_retry, "RETRY_WINDOW_SECONDS", 10_000)
    error, attempts = status_error(), []

    def call():
        attempts.append(clock.now)
        raise error

    with pytest.raises(APIStatusError) as caught:
        run_retry(call)
    assert caught.value is error
    assert len(attempts) == 13
    assert len(clock.waits) == 12


@pytest.mark.parametrize("elapsed,oversleep,waits", [(1795, 0, []), (0, 1800, [5])])
def test_deadline_prevents_wait_or_retry_after_oversleep(
    run_retry, clock, monkeypatch, elapsed, oversleep, waits
):
    monkeypatch.setattr(openai_retry.random, "uniform", lambda low, high: 0)
    clock.oversleep = oversleep
    error, attempts = status_error(), []

    def call():
        attempts.append(clock.now)
        clock.now += elapsed
        raise error

    with pytest.raises(APIStatusError) as caught:
        run_retry(call)
    assert caught.value is error
    assert len(attempts) == 1
    assert clock.waits == waits
