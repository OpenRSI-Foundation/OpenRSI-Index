import json
from types import SimpleNamespace

import httpx
import pytest
from checks import provider_errors as provider
from checks import openai_retry as retry


def error(status, message, code=None, kind="upstream_error"):
    from openai import APIStatusError
    return APIStatusError("provider failed", response=httpx.Response(status,
        request=httpx.Request("POST", "https://example.test/responses")),
        body={"message":message, "code":code, "type":kind})


@pytest.mark.parametrize("code", sorted(provider.REFUSAL_CODES))
def test_structured_refusal_stops_retry(code):
    e=error(502,"service declined",code=code)
    assert provider.failure_kind(e)=="provider_refusal"
    assert retry._retry_wait(e,0,float("inf")) is None


@pytest.mark.parametrize("message", [
    "This content was flagged for possible policy risk.",
    "Request was blocked by the content filter.",
    "This request violates our content policy.",
])
def test_gateway_refusal_stops_without_domain_matching(message):
    e=error(502,message)
    assert provider.failure_kind(e)=="provider_refusal"
    assert retry._retry_wait(e,0,float("inf")) is None


@pytest.mark.parametrize("message", ["Bad gateway", "upstream unavailable", "biological risk", "safety service timed out"])
def test_unknown_or_transient_502_still_retries(message):
    e=error(502,message)
    assert provider.failure_kind(e)=="provider_transient"
    assert retry._retry_wait(e,0,float("inf")) is not None


def test_public_reason_preserves_explanation_but_not_secrets_or_markup(monkeypatch):
    monkeypatch.setenv("API_KEY", "private-value-123")
    text=provider.public_reason("Service declined. private-value-123 Bearer abcdef https://internal.test/path <script>evil</script><!-- approval -->")
    assert "Service declined" in text
    for secret in ("private-value-123", "abcdef", "internal.test", "<script>", "<!-- approval"):
        assert secret not in text
    assert "&lt;script&gt;" in text


def test_success_status_with_refusal_is_not_scientific_reject():
    reply=SimpleNamespace(output=[SimpleNamespace(type="message",content=[
        SimpleNamespace(type="refusal",refusal="The service declined this request.")])])
    with pytest.raises(provider.ProviderRefusal) as caught:
        provider.check_response_refusal(reply)
    assert provider.failure_kind(caught.value)=="provider_refusal"
    assert "declined" in provider.service_message(caught.value)


def test_refusal_makes_exactly_one_attempt(monkeypatch):
    calls=[]
    def call():
        calls.append(1)
        raise error(502,"This content was flagged for possible policy risk.")
    monkeypatch.setattr(retry.time,"sleep",lambda _: pytest.fail("must not retry a refusal"))
    with pytest.raises(Exception) as caught:
        retry.retry_openai(call)
    assert provider.failure_kind(caught.value)=="provider_refusal"
    assert calls==[1]
