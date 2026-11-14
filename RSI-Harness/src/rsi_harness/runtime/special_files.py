"""FIFOs, sockets and devices in a paused Work's WORKDIR.

A Judge that reads Work's paused WORKDIR in place, read-only (a split-WORKDIR
Judge on Docker, every cluster Judge), can still write into a FIFO, connect to
a socket or open a device there: read-only stops changes to the files, not
data passed through them. If Work holds the other end, it reads what the
Judge's code sent once it resumes, such as the hidden tests. So before such a
round, while Work is paused, the Harness looks for these files and refuses
the round, not counted, while any exist: on Docker a helper walks the volume
(DockerWorkdirVolumeBackend.special_files), on a cluster the controller walks
the workspace (find_special_files).
"""

from __future__ import annotations

import os
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

# The most paths a refusal lists; the walk stops once it has found these.
SPECIAL_FILE_LIMIT = 5
# The most entries a cluster walk reads: a Work that wakes during the walk
# cannot keep it going. A larger workspace is refused, not counted.
SCAN_ENTRY_LIMIT = 10_000_000

_KINDS = (
    (stat.S_ISFIFO, "FIFO"),
    (stat.S_ISSOCK, "socket"),
    (stat.S_ISCHR, "character device"),
    (stat.S_ISBLK, "block device"),
)
# The most characters of a path a refusal shows; a longer one keeps its end.
_SHOWN_LIMIT = 200
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class SpecialFileScanError(Exception):
    """The walk could not check every entry; ``path`` is below the root."""

    def __init__(self, path: PurePosixPath, reason: str) -> None:
        super().__init__(f"{path}: {reason}")
        self.path = path
        self.reason = reason


def find_special_files(
    root: Path,
    *,
    limit: int = SPECIAL_FILE_LIMIT,
    entry_limit: int = SCAN_ENTRY_LIMIT,
) -> list[tuple[PurePosixPath, str]]:
    """Up to ``limit`` (path below ``root``, kind) pairs, one per FIFO,
    socket or device. Follows no symlink and enters every directory, even
    one on another file system or subvolume: the Judge's bind shows those
    too. Opens each directory relative to its parent and climbs back by
    "..", so a tree of any depth needs one descriptor and no long path.
    Raises SpecialFileScanError for an entry it cannot read, which could
    hide one, or past ``entry_limit`` entries."""
    found: list[tuple[PurePosixPath, str]] = []
    entries = 0
    # One per directory on the way down: its path, identity, and the
    # subdirectories still to visit.
    stack: list[tuple[PurePosixPath, tuple[int, int], list[str]]] = []
    path = PurePosixPath()
    descriptor = _open(path, str(root), None)
    try:
        while True:
            subdirectories: list[str] = []
            try:
                with os.scandir(descriptor) as listing:
                    for entry in listing:
                        entries += 1
                        if entries > entry_limit:
                            raise SpecialFileScanError(
                                PurePosixPath(), f"more than {entry_limit} entries"
                            )
                        kind = _kind(path / entry.name, entry)
                        if kind == "directory":
                            subdirectories.append(entry.name)
                        elif kind is not None:
                            found.append((path / entry.name, kind))
                            if len(found) >= limit:
                                return found
            except OSError as error:
                raise SpecialFileScanError(path, error.strerror or str(error)) from None
            identity = _identity(descriptor, path)
            stack.append((path, identity, subdirectories))
            while stack and not stack[-1][2]:
                stack.pop()
                if stack:
                    path = stack[-1][0]
                    descriptor = _move(descriptor, path, "..")
                    if _identity(descriptor, path) != stack[-1][1]:
                        raise SpecialFileScanError(path, "changed during the walk")
            if not stack:
                return found
            name = stack[-1][2].pop()
            path = stack[-1][0] / name
            descriptor = _move(descriptor, path, name)
    finally:
        os.close(descriptor)


def _kind(path: PurePosixPath, entry: os.DirEntry[str]) -> str | None:
    """None for a file or symlink, "directory", or the special kind."""
    try:
        if entry.is_dir(follow_symlinks=False):
            return "directory"
        if entry.is_file(follow_symlinks=False) or entry.is_symlink():
            return None
        mode = entry.stat(follow_symlinks=False).st_mode
    except OSError as error:
        raise SpecialFileScanError(path, error.strerror or str(error)) from None
    return next((kind for test, kind in _KINDS if test(mode)), "special file")


def _open(path: PurePosixPath, name: str, parent: int | None) -> int:
    try:
        return os.open(name, _DIRECTORY, dir_fd=parent)
    except OSError as error:
        raise SpecialFileScanError(path, error.strerror or str(error)) from None


def _move(descriptor: int, path: PurePosixPath, name: str) -> int:
    """Open ``name`` relative to ``descriptor``; close ``descriptor`` only
    once that worked, so the caller still owns it on failure."""
    opened = _open(path, name, descriptor)
    os.close(descriptor)
    return opened


def _identity(descriptor: int, path: PurePosixPath) -> tuple[int, int]:
    try:
        result = os.fstat(descriptor)
    except OSError as error:
        raise SpecialFileScanError(path, error.strerror or str(error)) from None
    return result.st_dev, result.st_ino


def _shown(path: PurePosixPath) -> str:
    text = str(path)
    if len(text) > _SHOWN_LIMIT:
        text = "..." + text[-_SHOWN_LIMIT:]
    return text if text.isprintable() else repr(text)


def special_files_refusal(
    found: Sequence[tuple[PurePosixPath, str]], workdir: PurePosixPath
) -> str:
    """The rsi-submit message for a round refused over ``found``."""
    listed = ", ".join(f"{_shown(workdir / path)} ({kind})" for path, kind in found)
    more = " and maybe more" if len(found) >= SPECIAL_FILE_LIMIT else ""
    return (
        "submission refused and not counted: the Judge reads WORKDIR in place, "
        f"so it must hold no FIFO, socket or device: {listed}{more}. "
        "Remove them, then submit again."
    )


def scan_failure_refusal(error: SpecialFileScanError, workdir: PurePosixPath) -> str:
    """The rsi-submit message for a round refused because the walk failed."""
    return (
        "submission refused and not counted: the Judge reads WORKDIR in place, "
        "and the Harness could not check it for FIFOs, sockets and devices: "
        f"{_shown(workdir / error.path)}: {error.reason}. "
        "Fix that, then submit again."
    )
