"""Shared redaction for Engine-authored diagnostics and durable metadata."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

_AUTHORIZATION = re.compile(
    r"(?im)(?P<open>[\"']?)(?P<key>\bauthorization)(?P=open)"
    r"(?P<separator>\s*[:=]\s*)[^\r\n]*"
)
_SCHEME = re.compile(r"(?i)\b(bearer|basic|digest)\s+[^\s,;\"']+")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?P<open>[\"']?)"
    r"(?P<key>"
    r"x[-_]?api[-_]?key|api[-_]?key|access[-_]?token|refresh[-_]?token|"
    r"token|secret|credential|password"
    r")(?P=open)"
    r"(?P<separator>\s*[:=]\s*|\s+)"
    r"(?P<value>\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;&}\r\n]+)"
)
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "xapikey",
        "apikey",
        "accesstoken",
        "refreshtoken",
        "token",
        "secret",
        "credential",
        "password",
    }
)


def _normalized_key(value: object) -> str:
    return "".join(
        character
        for character in str(value).casefold()
        if character.isalnum()
    )


def redact_text(value: str) -> str:
    """Consume complete headers and query/form-like credential values."""
    value = _AUTHORIZATION.sub(
        r"\g<key>\g<separator>[REDACTED]", value
    )
    value = _SCHEME.sub(r"\1 [REDACTED]", value)
    return _SENSITIVE_ASSIGNMENT.sub(
        lambda match: (
            f"{match.group('key')}{match.group('separator')}[REDACTED]"
        ),
        value,
    )


def redact_exact_values(value: str, secrets: Iterable[str]) -> str:
    """Redact non-empty runtime secret values, longest first."""

    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return value


def redact_structure(value: object) -> object:
    """Redact nested Engine metadata without touching task-authored artifacts."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if _normalized_key(key) in _SENSITIVE_KEYS
                else redact_structure(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_structure(child) for child in value)
    if isinstance(value, list):
        return [redact_structure(child) for child in value]
    return value


__all__ = ["redact_exact_values", "redact_structure", "redact_text"]
