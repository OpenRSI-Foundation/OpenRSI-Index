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
# Operator settings that add no bind and no container variable.
OPERATOR_APPTAINER_NAMES = ("APPTAINER_TMPDIR", "APPTAINER_CACHEDIR")
# SingularityCE's names for them, which it alone reads.
OPERATOR_SINGULARITY_NAMES = ("SINGULARITY_TMPDIR", "SINGULARITY_CACHEDIR")


def isolated_apptainer_environment(
    values: Mapping[str, str],
    host_environment: Mapping[str, str] | None = None,
    *,
    singularity_ce: bool = False,
) -> dict[str, str]:
    """Inject container values without exposing them to CLI CSV parsing:
    as APPTAINERENV_*, or SINGULARITYENV_* for SingularityCE, which then
    also gets the operator's APPTAINER_TMPDIR and APPTAINER_CACHEDIR under
    its own names where those are not set."""

    source = os.environ if host_environment is None else host_environment
    kept = OPERATOR_SINGULARITY_NAMES if singularity_ce else OPERATOR_APPTAINER_NAMES
    environment = {
        name: value
        for name, value in source.items()
        if not name.startswith(_APPTAINER_CONTROL_PREFIXES) or name in kept
    }
    if singularity_ce:
        for apptainer, singularity in zip(
            OPERATOR_APPTAINER_NAMES, OPERATOR_SINGULARITY_NAMES, strict=True
        ):
            if apptainer in source and singularity not in environment:
                environment[singularity] = source[apptainer]
    prefix = "SINGULARITYENV_" if singularity_ce else "APPTAINERENV_"
    for name, value in values.items():
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise InfrastructureError(f"invalid Apptainer environment name: {name!r}")
        if not isinstance(value, str) or "\x00" in value:
            raise InfrastructureError(f"invalid Apptainer environment value: {name}")
        environment[f"{prefix}{name}"] = value
    return environment


__all__ = [
    "OPERATOR_APPTAINER_NAMES",
    "OPERATOR_SINGULARITY_NAMES",
    "isolated_apptainer_environment",
]
