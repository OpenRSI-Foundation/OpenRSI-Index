"""Programs RSI Harness mounts into a Judge.

``rsi-agent`` makes model calls (the run's Agent, or the Judge LLM) in a
Judge that declares them; ``rsi-isolate`` runs one command isolated, in
every Judge. Each (``rsi_agent.py`` or ``rsi_isolate.py``, the isolation
they share in ``judge_isolation.py``, and a launcher) runs inside the Judge
and imports nothing from RSI Harness. This module holds the host side: where
they are mounted, and installing a copy for one run.
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
# ``rsi-agent llm`` with a Codex login: the pinned Codex CLI's native binary,
# read-only, from a host cache below the data root (one per version and
# image architecture). Never part of the Work image.
JUDGE_CODEX_TARGET = PurePosixPath("/opt/rsi-agent-codex")
JUDGE_CODEX_BINARY = JUDGE_CODEX_TARGET / "codex"
JUDGE_CODEX_CACHE = "judge-codex"
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
        JUDGE_CODEX_TARGET,
    }
)
# What rsi-agent and rsi-isolate share.
JUDGE_ISOLATION_MODULE = "judge_isolation.py"
# rsi-isolate: the host directory below the run's artifacts, and in every
# Judge the program and the launcher on PATH.
JUDGE_ISOLATE_DIRECTORY = "judge-isolate"
JUDGE_ISOLATE_PROGRAM = "rsi_isolate.py"
JUDGE_ISOLATE_LAUNCHER = "rsi-isolate"
JUDGE_ISOLATE_CODE_TARGET = PurePosixPath("/opt/rsi-isolate")
JUDGE_ISOLATE_LAUNCHER_TARGET = PurePosixPath("/usr/local/bin/rsi-isolate")
# rsi-isolate's unprivileged users in a root Judge, one per concurrent call
# (gid of the same number); disjoint from rsi-agent's. An image that uses
# one of them gets Landlock instead.
JUDGE_ISOLATE_UID = 62000
JUDGE_ISOLATE_SLOTS = 64
JUDGE_ISOLATE_UIDS = range(JUDGE_ISOLATE_UID, JUDGE_ISOLATE_UID + JUDGE_ISOLATE_SLOTS)
_AGENT_FILES = (
    (JUDGE_AGENT_PROGRAM, 0o644),
    (JUDGE_ISOLATION_MODULE, 0o644),
    (JUDGE_AGENT_LAUNCHER, 0o755),
)
_ISOLATE_FILES = (
    (JUDGE_ISOLATE_PROGRAM, 0o644),
    (JUDGE_ISOLATION_MODULE, 0o644),
    (JUDGE_ISOLATE_LAUNCHER, 0o755),
)


def judge_agent_source(name: str) -> bytes:
    """The packaged bytes of a program, the shared module or a launcher."""
    return resources.files(__name__).joinpath(name).read_bytes()


def install_judge_agent(
    run_root: Path, *, owner: tuple[int, int] | None = None
) -> Path:
    """Copy ``rsi-agent`` below ``run_root`` once and return the directory."""
    return _install(run_root / JUDGE_AGENT_DIRECTORY, _AGENT_FILES, owner)


def install_judge_isolate(
    run_root: Path, *, owner: tuple[int, int] | None = None
) -> Path:
    """Copy ``rsi-isolate`` below ``run_root`` once and return the directory."""
    return _install(run_root / JUDGE_ISOLATE_DIRECTORY, _ISOLATE_FILES, owner)


def _install(directory: Path, files, owner: tuple[int, int] | None) -> Path:
    """Idempotent: a later round reuses the copy after checking its bytes and
    modes, and a differing or symlinked copy is replaced."""
    if directory.is_symlink():
        directory.unlink()
    directory.mkdir(mode=0o755, exist_ok=True)
    os.chmod(directory, 0o755)
    if owner is not None:
        os.chown(directory, *owner, follow_symlinks=False)
    for name, mode in files:
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
    "JUDGE_CODEX_BINARY",
    "JUDGE_CODEX_CACHE",
    "JUDGE_CODEX_TARGET",
    "JUDGE_ISOLATE_CODE_TARGET",
    "JUDGE_ISOLATE_DIRECTORY",
    "JUDGE_ISOLATE_LAUNCHER",
    "JUDGE_ISOLATE_LAUNCHER_TARGET",
    "JUDGE_ISOLATE_PROGRAM",
    "JUDGE_ISOLATE_SLOTS",
    "JUDGE_ISOLATE_UID",
    "JUDGE_ISOLATE_UIDS",
    "JUDGE_ISOLATION_MODULE",
    "install_judge_agent",
    "install_judge_isolate",
    "judge_agent_source",
]
