"""Checks rendered into scheduler jobs, never evaluated on the submit host."""

import shlex
from pathlib import Path


def apptainer_preflight(binary: Path) -> str:
    """Check the configured executable where the build or Engine will run.

    This is an availability check, not a substitute for image, namespace,
    device, or Judge isolation checks during actual container startup.
    """
    return f"""apptainer={shlex.quote(str(binary))}
if [[ ! -f "$apptainer" || ! -x "$apptainer" ]]; then
  echo "compute-node preflight: Apptainer is missing or not executable: $apptainer" >&2
  exit 1
fi
"""
