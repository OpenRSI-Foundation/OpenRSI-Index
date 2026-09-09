"""Resolve Harbor environment templates from runtime-only value sources."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

# Match Harbor's full-value template grammar exactly.  Environment lookup is
# mapping based, so names need not be valid shell identifiers.
_TEMPLATE = re.compile(r"^\$\{([^}:]+)(?::-(.*))?\}$")


class MissingRuntimeEnvironmentError(ValueError):
    """A required runtime template name has no value or default."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name


def runtime_template_name(value: str) -> str | None:
    match = _TEMPLATE.fullmatch(value)
    return None if match is None else match.group(1)


def resolve_runtime_environment(
    environment: Sequence[tuple[str, str]] | Mapping[str, str],
    source: Mapping[str, str],
) -> dict[str, str]:
    """Resolve full-value ``${NAME}``/``${NAME:-default}`` templates."""

    items = environment.items() if isinstance(environment, Mapping) else environment
    resolved: dict[str, str] = {}
    for key, value in items:
        match = _TEMPLATE.fullmatch(value)
        if match is None:
            resolved[key] = value
            continue
        name, default = match.groups()
        if name in source:
            resolved[key] = source[name]
        elif default is not None:
            resolved[key] = default
        else:
            raise MissingRuntimeEnvironmentError(name)
    return resolved


__all__ = [
    "MissingRuntimeEnvironmentError",
    "resolve_runtime_environment",
    "runtime_template_name",
]
