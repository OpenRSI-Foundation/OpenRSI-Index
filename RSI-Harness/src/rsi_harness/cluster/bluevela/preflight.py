"""Checks of the configured Apptainer: rendered into scheduler jobs, and
repeated on the submit host when it sees the same file."""

import os
import shlex
import subprocess
from pathlib import Path

from rsi_harness.errors import SetupError

SINGULARITY_CE_UNSUPPORTED = (
    "SingularityCE is not supported; install an unprivileged Apptainer "
    "(upstream install-unprivileged.sh) under your storage and set "
    "apptainer.binary to it"
)


def is_singularity_ce(binary: Path) -> bool:
    """Whether ``binary`` is SingularityCE (or Singularity before it) by what
    ``--version`` says, else by a resolved name of exactly ``singularity``.
    An Apptainer installed or wrapped as ``singularity`` says ``apptainer
    version``, so it is Apptainer. ``apptainer_preflight`` applies the same
    rule on the node."""
    try:
        completed = subprocess.run(
            (str(binary), "--version"),
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
        answer = completed.stdout if completed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        answer = None
    if answer is None:
        return binary.resolve().name == "singularity"
    return answer.strip().lower().startswith("singularity")


def refuse_singularity_ce(binary: Path) -> None:
    """Refuse before any job a SingularityCE the submit host sees at the
    profile's path; a path missing here is left to the jobs' check."""
    if binary.is_file() and os.access(binary, os.X_OK) and is_singularity_ce(binary):
        raise SetupError(f"apptainer.binary {binary}: {SINGULARITY_CE_UNSUPPORTED}")


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
singularity_ce=0
if apptainer_says="$(timeout 30 "$apptainer" --version 2>/dev/null)"; then
  apptainer_says="${{apptainer_says#"${{apptainer_says%%[![:space:]]*}}"}}"
  if [[ ${{apptainer_says,,}} == singularity* ]]; then singularity_ce=1; fi
elif [[ $(basename -- "$(readlink -f -- "$apptainer")") == singularity ]]; then
  singularity_ce=1
fi
if (( singularity_ce )); then
  echo "compute-node preflight: $apptainer:" \\
    {shlex.quote(SINGULARITY_CE_UNSUPPORTED)} >&2
  exit 1
fi
"""


def node_tmp_assignment(name: str, default: Path, variable: str | None) -> str:
    """A job script line: ``name`` is the node-local scratch the job's
    ``variable`` names where the scheduler set it, else ``default``."""
    if variable is None:
        return f"{name}={shlex.quote(str(default))}"
    fallback = shlex.quote(str(default))
    return f'{name}="${{{variable}:-}}"\n[[ -n "${name}" ]] || {name}={fallback}'
