"""Isolated, argv-safe environment injection for Blue Vela Apptainer."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from rsi_harness.errors import InfrastructureError

_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_APPTAINER_CONTROL_PREFIXES = (
    "APPTAINER",
    "SINGULARITY",
)


def isolated_apptainer_environment(
    values: Mapping[str, str],
    host_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Inject container values without exposing them to CLI CSV parsing."""

    source = os.environ if host_environment is None else host_environment
    environment = {
        name: value
        for name, value in source.items()
        if not name.startswith(_APPTAINER_CONTROL_PREFIXES)
    }
    for name, value in values.items():
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise InfrastructureError(
                f"invalid Apptainer environment name: {name!r}"
            )
        if not isinstance(value, str) or "\x00" in value:
            raise InfrastructureError(
                f"invalid Apptainer environment value: {name}"
            )
        environment[f"APPTAINERENV_{name}"] = value
    return environment


__all__ = ["isolated_apptainer_environment"]
