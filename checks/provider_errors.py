"""Shared review-provider diagnostics; keep Skills and Index copies identical."""
import html
import os
import re

REFUSAL_CODES = {"content_policy_violation", "safety_violation", "content_filter", "refusal", "request_refused", "policy_violation"}
QUOTA_CODES = {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "usage_limit_reached"}


def service_message(error):
    body = getattr(error, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    message = body.get("message", "") if isinstance(body, dict) else ""
    return message if isinstance(message, str) else ""


def failure_kind(error):
    code, kind = getattr(error, "code", None), getattr(error, "type", None)
    message = service_message(error).lower()
    # Compatibility for gateways that discard the structured refusal code.
    # Inspect only the provider's error message, never proposal or model output.
    explicit_refusal = any(phrase in message for phrase in (
        "this content was flagged", "request was blocked by", "request was rejected by the safety",
        "request violates our content policy", "request violates the content policy"))
    if code in REFUSAL_CODES or kind in REFUSAL_CODES or explicit_refusal:
        return "provider_refusal"
    status = getattr(error, "status_code", None)
    if status in (401, 403) or "upstream authentication failed" in message:
        return "provider_authentication"
    if code in QUOTA_CODES or kind in QUOTA_CODES:
        return "provider_quota"
    if status in (408, 409, 429) or isinstance(status, int) and 500 <= status <= 599:
        return "provider_transient"
    return "provider"


def redact_message(message):
    for name, value in os.environ.items():
        if len(value) >= 8 and re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|PRIVATE_KEY", name, re.I):
            message = message.replace(value, "[redacted]")
    message = re.sub(r"(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", "[redacted]", message)
    message = re.sub(r"(?i)\b(?:Bearer\s+\S+|(?:sk-|gh[pousr]_|github_pat_)[A-Za-z0-9_-]+)", "[redacted]", message)
    message = re.sub(r"(?i)(?:api[_-]?key|token|password|secret|authorization)\s*[=:]\s*[^\s,;]+", "[redacted]", message)
    message = re.sub(r"https?://[^\s<>]+", "[URL omitted]", message)
    message = re.sub(r"(?<![\w])/(?:[^\s<>]+)", "[path omitted]", message)
    message = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b", "[address omitted]", message)
    return "".join(c for c in message if c in "\n\t" or ord(c) >= 32).strip()


def public_reason(message):
    # Escaping prevents provider text from injecting workflow markers or markup.
    return "<pre>" + html.escape(redact_message(message)) + "</pre>" if message else ""


class ProviderRefusal(Exception):
    """A completed API response can carry a refusal rather than a review."""
    code = "refusal"
    type = "refusal"
    status_code = None

    def __init__(self, message):
        super().__init__("The provider declined the review")
        self.body = {"message": message}


def check_response_refusal(response):
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", None) or []:
                if getattr(block, "type", None) == "refusal":
                    raise ProviderRefusal(getattr(block, "refusal", "The provider declined the review."))
