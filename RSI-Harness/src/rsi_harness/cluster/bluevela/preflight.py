"""Checks of the configured Apptainer: rendered into scheduler jobs, and
repeated on the submit host when it sees the same file."""

import os
import shlex
import subprocess
from pathlib import Path

from rsi_harness.errors import SetupError

SETUID_SECCOMP = (
    'a setuid SingularityCE cannot run with network_isolation = "seccomp" '
    "(its filter sets no_new_privs, which disables setuid): use "
    '"namespace", or a SingularityCE or Apptainer installed without setuid'
)
SETUID_USER_NAMESPACE = (
    "a setuid SingularityCE cannot run inside a user namespace, where its "
    "setuid starter gets no privileges: use a SingularityCE or Apptainer "
    "installed without setuid"
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


def build_config(binary: Path) -> dict[str, str]:
    """What ``binary buildcfg`` reports (``SINGULARITY_SUID_INSTALL=1``, for
    example), or nothing when it cannot answer."""
    try:
        completed = subprocess.run(
            (str(binary), "buildcfg"),
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode:
        return {}
    pairs = (line.partition("=") for line in completed.stdout.splitlines())
    return {key.strip(): value.strip() for key, found, value in pairs if found}


def runtime_conf(binary: Path, config: dict[str, str] | None = None) -> Path | None:
    """The configuration file ``binary`` reports, or None."""
    config = build_config(binary) if config is None else config
    value = config.get("SINGULARITY_CONF_FILE") or config.get("APPTAINER_CONF_FILE")
    return Path(value) if value else None


def runs_setuid(binary: Path) -> bool:
    """Whether a SingularityCE at ``binary`` starts containers setuid: a
    setuid install whose configuration does not say ``allow setuid = no``.
    An install that cannot tell counts as not setuid; the job's isolation
    checks still prove what it does."""
    config = build_config(binary)
    if config.get("SINGULARITY_SUID_INSTALL") != "1":
        return False
    conf = runtime_conf(binary, config)
    try:
        text = "" if conf is None else conf.read_text(errors="replace")
    except OSError:
        text = ""
    allowed = "yes"
    for line in text.splitlines():
        key, found, value = line.partition("=")
        if found and key.strip() == "allow setuid":
            allowed = value.strip().lower()
    return allowed != "no"


def job_runtime_is_singularity_ce() -> bool:
    """In a job's Engine: what its driver's ``apptainer_preflight`` found."""
    return os.environ.get("RSI_HARNESS_SINGULARITY_CE") == "1"


def singularity_ce_conflict(
    binary: Path, *, network_isolation: str, user_namespace: bool
) -> str | None:
    """Why the SingularityCE at ``binary`` cannot give the profile's
    isolation, or None (also for Apptainer)."""
    if not is_singularity_ce(binary) or not runs_setuid(binary):
        return None
    if network_isolation == "seccomp":
        return SETUID_SECCOMP
    if user_namespace:
        return SETUID_USER_NAMESPACE
    return None


def refuse_singularity_ce_conflict(binary: Path, *, network_isolation: str) -> None:
    """Refuse before any job a SingularityCE the submit host sees at the
    profile's path that cannot isolate as the profile asks; a path missing
    here is left to the job's check."""
    if not (binary.is_file() and os.access(binary, os.X_OK)):
        return
    conflict = singularity_ce_conflict(
        binary, network_isolation=network_isolation, user_namespace=False
    )
    if conflict is not None:
        raise SetupError(f"apptainer.binary {binary}: {conflict}")


def apptainer_preflight(binary: Path) -> str:
    """Check the configured executable where the build or Engine will run,
    and set ``singularity_ce`` (1 or 0) for the rest of the script by
    ``is_singularity_ce``'s rule.

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
# SingularityCE reads container variables only as SINGULARITYENV_*. The
# Engine reads the answer from RSI_HARNESS_SINGULARITY_CE.
if (( singularity_ce )); then
  container_env=SINGULARITYENV_
else
  container_env=APPTAINERENV_
fi
export RSI_HARNESS_SINGULARITY_CE="$singularity_ce"
"""


def node_tmp_assignment(name: str, default: Path, variable: str | None) -> str:
    """A job script line: ``name`` is the node-local scratch the job's
    ``variable`` names where the scheduler set it, else ``default``."""
    if variable is None:
        return f"{name}={shlex.quote(str(default))}"
    fallback = shlex.quote(str(default))
    return f'{name}="${{{variable}:-}}"\n[[ -n "${name}" ]] || {name}={fallback}'
