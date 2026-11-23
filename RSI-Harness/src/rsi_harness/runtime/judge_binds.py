"""The record of a Judge's NFS binds (judge_isolation.BIND_RECORD).

NFS shows "/" as the root of every mount in mountinfo, so inside a Judge
rsi-isolate and rsi-agent cannot tell binds of different directories of one
NFS file system from the hidden /logs/verifier. Host paths do not tell them
apart either: a server can show one directory at several paths of one file
system (a bind it exports without its own fsid), with no mount between them.

So the Harness walks the tree of each bind it makes from NFS, in the mount
namespace that starts the container, right before it starts, and collects
every inode it reaches on that file system. Two binds whose inodes do not
meet show no file of each other; for each bind the record lists those
others ("disjoint"), with the device and inode of the bind's top, which the
Judge checks against the mount it sees. A tree the Harness cannot read
fully, or that is too large to walk in time, proves nothing.

The record is bound read-only at judge_isolation.BIND_RECORD_DIR from
storage that is not NFS. A Judge with no bind from NFS gets no record and
no extra mount.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from rsi_harness.judge_agent.judge_isolation import (
    BIND_RECORD_DIR,
    NFS_TYPES,
    device_of,
    mount_table,
)

BIND_RECORD_TARGET = PurePosixPath(BIND_RECORD_DIR)
BIND_RECORD_NAME = "binds.json"
# The entries the walk of one bind may visit, and for how long, and how
# long all walks of one Judge may take: a tree beyond them proves nothing,
# and its bind stays a mount of what the Judge hides. Each bind has its own
# bound, so a large tree (a workspace, a site cache) never leaves a small one
# unproven.
WALK_ENTRIES = 1_000_000
WALK_SECONDS = 20.0
WALK_TOTAL_SECONDS = 60.0
# Where the Judge's hidden paths lie: rsi-isolate's /logs/verifier,
# rsi-agent's /tests, and the science Judge library's /logs, /run, /home,
# /root, /var/tmp, /tmp and WORKDIR (any target). Any pair of binds can be
# asked about, so every pair is recorded. The walks share one deadline, so
# the order matters: first the Harness's own binds that every rsi-isolate
# and rsi-agent call needs set apart (small, and walked first before any
# other was), then other binds below the hidden roots, then the rest under
# /opt and /usr, then the rest (a workspace, a site cache), which the budget
# may leave unproven.
OWN_TARGETS = (
    "/logs/verifier",
    "/tests",
    "/logs/agent",
    "/opt/rsi-isolate",
    "/usr/local/bin/rsi-isolate",
    "/opt/rsi-agent",
    "/usr/local/bin/rsi-agent",
)
FIRST_TARGETS = ("/logs", "/tests", "/run", "/home", "/root", "/var/tmp", "/tmp")
THEN_TARGETS = ("/opt", "/usr")


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


class _Budget:
    def __init__(self, entries: int, seconds: float, deadline: float) -> None:
        self.entries = entries
        self.deadline = min(time.monotonic() + seconds, deadline)

    def spend(self, count: int = 1) -> bool:
        self.entries -= count
        return self.entries >= 0 and time.monotonic() < self.deadline


def _reach(source: str, status: os.stat_result, budget: _Budget) -> set[int] | None:
    """Every inode the tree of ``source`` reaches (its own, and for each
    entry of each directory both the inode the directory lists and the one
    its lstat gives), or None when the tree cannot be walked fully or
    reaches another file system (a mount in it, which the Judge would see
    as its own mount the record says nothing about)."""
    if not stat.S_ISDIR(status.st_mode):
        return {status.st_ino}
    seen = {status.st_ino}
    stack = [(source, status)]
    try:
        while stack:
            path, expected = stack.pop()
            # Open what is listed, and check it is what was lstat'ed: an
            # automount the open triggers is another file system.
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
                    return None
                with os.scandir(descriptor) as entries:
                    for entry in entries:
                        if not budget.spend():
                            return None
                        below = os.lstat(entry.name, dir_fd=descriptor)
                        if below.st_dev != status.st_dev:
                            return None
                        seen.add(entry.inode())
                        seen.add(below.st_ino)
                        if stat.S_ISDIR(below.st_mode):
                            stack.append((os.path.join(path, entry.name), below))
            finally:
                os.close(descriptor)
    except (OSError, ValueError):
        return None
    return seen


def nfs_bind_record(
    binds: Iterable[tuple[Path | str, PurePosixPath | str]],
    mountinfo: str | None = None,
    *,
    entries: int = WALK_ENTRIES,
    seconds: float = WALK_SECONDS,
    total_seconds: float = WALK_TOTAL_SECONDS,
) -> dict | None:
    """The record of the ``(source, target)`` binds whose source is on NFS:
    for each target, the device and inode of the source, and the recorded
    targets whose trees share no inode with its own. None when no source is
    on NFS.

    A source whose device is not that of its holding mount (a race with a
    remount, or a namespace that differs) is left out, so the Judge keeps
    treating it as it always has."""
    table = mount_table(mountinfo)
    found: dict[str, tuple[str, os.stat_result]] = {}
    for source, target in binds:
        try:
            real = os.path.realpath(source)
            status = os.stat(real)
        except (OSError, ValueError):
            continue
        holders = [point for point in table if _within(real, point)]
        if not holders:
            continue
        device, _root, _options, kind = table[max(holders, key=len)]
        if kind not in NFS_TYPES or device_of(status) != device:
            continue
        found[os.path.normpath(str(target))] = (real, status)
    if not found:
        return None
    # Only a file system with two recorded binds has a pair to set apart.
    shared = {
        device
        for device in {status.st_dev for _real, status in found.values()}
        if sum(status.st_dev == device for _real, status in found.values()) > 1
    }

    def rank(target: str) -> tuple[int, int]:
        if target in OWN_TARGETS:
            return (0, OWN_TARGETS.index(target))
        if any(_within(target, root) for root in FIRST_TARGETS):
            return (1, 0)
        return (2, 0) if any(_within(target, root) for root in THEN_TARGETS) else (3, 0)

    order = sorted(
        (target for target in found if found[target][1].st_dev in shared), key=rank
    )
    deadline = time.monotonic() + total_seconds
    reach = {
        target: _reach(*found[target], _Budget(entries, seconds, deadline))
        for target in order
    }

    def apart(left: str, right: str) -> bool:
        one, two = reach.get(left), reach.get(right)
        return (
            found[left][1].st_dev == found[right][1].st_dev
            and one is not None
            and two is not None
            and one.isdisjoint(two)
        )

    record: dict[str, dict[str, object]] = {
        target: {
            "device": device_of(status),
            "inode": status.st_ino,
            "disjoint": sorted(
                other for other in found if other != target and apart(target, other)
            ),
        }
        for target, (_real, status) in found.items()
    }
    return {"version": 2, "binds": record}


def _local(directory: Path) -> bool:
    """Whether ``directory`` is on a file system other than NFS."""
    try:
        real = os.path.realpath(directory)
        table = mount_table()
    except OSError:
        return False
    holders = [point for point in table if _within(real, point)]
    return bool(holders) and table[max(holders, key=len)][3] not in NFS_TYPES


def write_bind_record(record: dict, parent: Path) -> Path:
    """A new directory below ``parent`` (which must not be on NFS) holding
    the record, read-only for everyone, to bind at BIND_RECORD_DIR. The
    caller removes it with remove_bind_record once the Judge is gone."""
    parent = Path(parent)
    parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not _local(parent):
        raise OSError(f"{parent} is on NFS or cannot be checked")
    directory = Path(tempfile.mkdtemp(prefix="judge-binds-", dir=parent))
    try:
        path = directory / BIND_RECORD_NAME
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
        os.chmod(path, 0o444)
        os.chmod(directory, 0o755)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return directory


def remove_bind_record(directory: Path | None) -> None:
    if directory is not None:
        shutil.rmtree(directory, ignore_errors=True)


def bind_record_for(
    binds: Iterable[tuple[Path | str, PurePosixPath | str]],
    parent: Path,
) -> Path | None:
    """The record directory for these binds, written below ``parent``, or
    None when no source is on NFS or the record cannot be written; without
    one the Judge treats NFS binds as it always has."""
    record = nfs_bind_record(binds)
    if record is None:
        return None
    try:
        return write_bind_record(record, parent)
    except OSError as error:
        print(f"[rsi-engine] judge_bind_record: {error}", flush=True)
        return None
