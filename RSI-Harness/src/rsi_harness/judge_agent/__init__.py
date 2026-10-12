"""``rsi-agent``: the run's own Agent, callable from a model-access Judge.

The program itself (``rsi_agent.py`` and the ``rsi-agent`` launcher) runs
inside the Judge and imports nothing from RSI Harness. This module holds the
host side: where it is mounted, and installing a copy for one run.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path, PurePosixPath

from rsi_harness.models import ContainerTmpfs

# The host directory, below the run's artifacts, holding the copy for a run.
JUDGE_AGENT_DIRECTORY = "judge-agent"
JUDGE_AGENT_PROGRAM = "rsi_agent.py"
JUDGE_AGENT_LAUNCHER = "rsi-agent"
# In the Judge: the program, the launcher on PATH, and its state.
JUDGE_AGENT_CODE_TARGET = PurePosixPath("/opt/rsi-agent")
JUDGE_AGENT_LAUNCHER_TARGET = PurePosixPath("/usr/local/bin/rsi-agent")
JUDGE_AGENT_STATE_ROOT = PurePosixPath("/run/rsi-agent")
JUDGE_AGENT_PRIVATE = JUDGE_AGENT_STATE_ROOT / "private"
JUDGE_AGENT_LAUNCH_SPEC = JUDGE_AGENT_PRIVATE / "launch.json"
# The Agents' unprivileged users in a root Judge: each concurrent call takes
# its own uid (and gid of the same number) from this range, which no passwd or
# group entry in the image may use.
JUDGE_AGENT_UID = 61000
JUDGE_AGENT_SLOTS = 16
JUDGE_AGENT_UIDS = range(JUDGE_AGENT_UID, JUDGE_AGENT_UID + JUDGE_AGENT_SLOTS)
JUDGE_AGENT_STATE_TMPFS = ContainerTmpfs(
    target=JUDGE_AGENT_STATE_ROOT,
    options="rw,nosuid,nodev,noexec,mode=0711",
)
JUDGE_AGENT_PRIVATE_TMPFS = ContainerTmpfs(
    target=JUDGE_AGENT_PRIVATE,
    options="rw,nosuid,nodev,noexec,mode=0700",
)
JUDGE_AGENT_TMPFS = (JUDGE_AGENT_STATE_TMPFS, JUDGE_AGENT_PRIVATE_TMPFS)
JUDGE_AGENT_TARGETS = frozenset(
    {
        JUDGE_AGENT_CODE_TARGET,
        JUDGE_AGENT_LAUNCHER_TARGET,
        JUDGE_AGENT_STATE_ROOT,
        JUDGE_AGENT_PRIVATE,
    }
)
_FILES = ((JUDGE_AGENT_PROGRAM, 0o644), (JUDGE_AGENT_LAUNCHER, 0o755))


def judge_agent_source(name: str) -> bytes:
    """The packaged bytes of the program or the launcher."""
    return resources.files(__name__).joinpath(name).read_bytes()


def install_judge_agent(
    run_root: Path, *, owner: tuple[int, int] | None = None
) -> Path:
    """Copy ``rsi-agent`` below ``run_root`` once and return the directory.

    Idempotent: a later round reuses the copy after checking its bytes and
    modes, and a differing or symlinked copy is replaced.
    """
    directory = run_root / JUDGE_AGENT_DIRECTORY
    if directory.is_symlink():
        directory.unlink()
    directory.mkdir(mode=0o755, exist_ok=True)
    os.chmod(directory, 0o755)
    if owner is not None:
        os.chown(directory, *owner, follow_symlinks=False)
    for name, mode in _FILES:
        content = judge_agent_source(name)
        target = directory / name
        try:
            current = target.lstat()
        except FileNotFoundError:
            current = None
        if (
            current is not None
            and not target.is_symlink()
            and target.is_file()
            and current.st_mode & 0o777 == mode
            and target.read_bytes() == content
        ):
            continue
        temporary = directory / f".{name}.tmp"
        temporary.unlink(missing_ok=True)
        temporary.write_bytes(content)
        os.chmod(temporary, mode)
        if owner is not None:
            os.chown(temporary, *owner, follow_symlinks=False)
        os.replace(temporary, target)
    return directory


__all__ = [
    "JUDGE_AGENT_CODE_TARGET",
    "JUDGE_AGENT_DIRECTORY",
    "JUDGE_AGENT_LAUNCHER",
    "JUDGE_AGENT_LAUNCHER_TARGET",
    "JUDGE_AGENT_LAUNCH_SPEC",
    "JUDGE_AGENT_PRIVATE",
    "JUDGE_AGENT_PRIVATE_TMPFS",
    "JUDGE_AGENT_PROGRAM",
    "JUDGE_AGENT_STATE_ROOT",
    "JUDGE_AGENT_STATE_TMPFS",
    "JUDGE_AGENT_TARGETS",
    "JUDGE_AGENT_TMPFS",
    "JUDGE_AGENT_SLOTS",
    "JUDGE_AGENT_UID",
    "JUDGE_AGENT_UIDS",
    "install_judge_agent",
    "judge_agent_source",
]
