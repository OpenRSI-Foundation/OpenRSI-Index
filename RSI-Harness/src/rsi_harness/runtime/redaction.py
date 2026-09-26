"""Shared redaction for Engine-authored diagnostics and durable metadata."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping

_QUOTED_VALUE = r'"(?:\\.|[^"\\\r\n])*"' + r"|'(?:\\.|[^'\\\r\n])*'"
_AUTHORIZATION = re.compile(
    r"""(?i)(?<![\w-])(?P<prefix>["']?authorization["']?[ \t]*[:=][ \t]*)"""
    rf"""(?P<value>{_QUOTED_VALUE}|(?:(?!\\[nr])[^\r\n"'{{}}])+)"""
)
_ASSIGNMENT = re.compile(
    r"""(?<![\w-])(?P<prefix>(?P<quote>["']?)(?P<key>[\w-]+)(?P=quote)"""
    r"""[ \t]*[:=][ \t]*)"""
)
_VALUE = re.compile(
    _QUOTED_VALUE + r"""|\[REDACTED\]|(?:(?!\\[nr])[^\s,;&}\]"'])+"""
)
_URL_TOKEN = re.compile(
    r"(?i)(?P<prefix>[?&]token=)(?P<value>(?:(?!\\[nr])[^\s&#\"'])+)"
)
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "xapikey",
        "apikey",
        "accesstoken",
        "refreshtoken",
        "idtoken",
        "secret",
        "credential",
        "credentials",
        "password",
        "passwd",
        "hftoken",
        "ghtoken",
        "githubtoken",
        "gitlabtoken",
        "npmtoken",
        "pypitoken",
        "huggingfacetoken",
        "huggingfacehubtoken",
    }
)
_CREDENTIAL_SUFFIX = re.compile(
    r"(?:^|_)(?:(?:API|ACCESS|SECRET|PRIVATE)_?KEY(?:_ID)?|"
    r"(?:ACCESS|REFRESH|AUTH|SESSION|ID)_TOKEN|SECRET|PASSWORD|PASSWD)$"
)
_SETTING_WORDS = frozenset({"DISABLE", "ENABLE", "REQUIRE", "USE", "HAS", "NO"})


def _normalized_key(value: object) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def is_credential_name(name: object) -> bool:
    """Recognize credential fields, not arbitrary token/configuration names."""
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name)).upper().replace("-", "_")
    if _SETTING_WORDS.intersection(words.split("_")):
        return False
    return (
        _normalized_key(name) in _SENSITIVE_KEYS
        or _CREDENTIAL_SUFFIX.search(words) is not None
    )


def _masked_assignment(match: re.Match[str]) -> str:
    original = match["value"]
    quote = original[0] if original.startswith(('"', "'")) else ""
    return f"{match['prefix']}{quote}[REDACTED]{quote}"


def _redact_line(value: str) -> str:
    value = _AUTHORIZATION.sub(_masked_assignment, value)
    parts: list[str] = []
    end = 0
    for match in _ASSIGNMENT.finditer(value):
        if match.start() < end or not is_credential_name(match["key"]):
            continue
        field = _VALUE.match(value, match.end())
        if field is None:
            continue
        quote = field[0][0] if field[0].startswith(('"', "'")) else ""
        parts.extend((value[end:match.end()], f"{quote}[REDACTED]{quote}"))
        end = field.end()
    parts.append(value[end:])
    return _URL_TOKEN.sub(_masked_assignment, "".join(parts))


def redact_text(value: str) -> str:
    """Redact credentials without consuming prose, code or JSON delimiters."""
    if value.lstrip().startswith(("{", "[")):
        try:
            payload = json.loads(value)
        except ValueError:
            pass
        else:
            safe = redact_structure(payload)
            if safe == payload:
                return value
            prefix = value[: len(value) - len(value.lstrip())]
            suffix = value[len(value.rstrip()) :]
            return prefix + json.dumps(safe, ensure_ascii=False) + suffix
    lines = value.splitlines(keepends=True)
    if len(lines) > 1:
        return "".join(redact_text(line) for line in lines)
    return _redact_line(value)


def redact_exact_values(value: str, secrets: Iterable[str]) -> str:
    """Redact non-empty runtime secret values, longest first."""

    variants = {
        variant
        for item in secrets
        if item
        for variant in (
            item,
            json.dumps(item)[1:-1],
            json.dumps(item, ensure_ascii=False)[1:-1],
        )
    }
    for secret in sorted(variants, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return value


def redact_structure(value: object) -> object:
    """Redact nested Engine metadata without touching task-authored artifacts."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]" if is_credential_name(key) else redact_structure(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_structure(child) for child in value)
    if isinstance(value, list):
        return [redact_structure(child) for child in value]
    return value


__all__ = ["redact_exact_values", "redact_structure", "redact_text"]
