"""Judge workspace copies for the cluster runtimes.

Each Judge round reads a copy of Work's paused workspace that shares no inode
with Work: every file in it is one the Harness created under the rounds
directory, which Work never sees. A file Work has not changed since the
previous round is hard-linked from that round's copy, kept as ``.retained``
until the next round; everything else is copied. The Judge binds its copy
read-only, so the retained copy holds exactly what the Harness wrote.

A lease belongs to one acquisition in the controller process. Nothing is
persisted or adopted across processes: a resumed run starts with a full copy.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import os
import secrets
import shutil
import stat
import struct
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from rsi_harness.errors import InfrastructureError

RETAINED = ".retained"
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# Entries that vanished or changed type during the walk, or that the Harness
# cannot open: left out of the copy, never an error.
_OMIT = frozenset(
    {errno.EACCES, errno.EPERM, errno.ELOOP, errno.ENOENT, errno.ENXIO, errno.ENOTDIR}
)
# The file system cannot hard-link here: copy every file for the rest of the run.
_NO_LINK = frozenset({errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK, errno.EXDEV})
_OMITTED_SHOWN = 5
# Deeper directories are left out: each level holds three descriptors and two
# stack frames, which must stay within common limits (1024 and 1000).
MAX_DEPTH = 200
# tmpfs and ramfs never write pages back, so a write through a mapping made
# earlier does not fault again and leaves ctime as it was.
_NO_WRITEBACK = frozenset({0x01021994, 0x858458F6})

# A manifest entry, packed to keep the controller's memory small: Work's file
# when the Harness read it (device, inode, size, mtime, ctime, mode), then the
# Harness inode holding that content and its size.
_KEY = struct.Struct("<QQQqqQ")
_COPY = struct.Struct("<QQ")


def _work_key(found: os.stat_result) -> bytes:
    return _KEY.pack(
        found.st_dev,
        found.st_ino,
        found.st_size,
        found.st_mtime_ns,
        found.st_ctime_ns,
        found.st_mode,
    )


@dataclass(frozen=True)
class DirectorySnapshot:
    root: Path
    name: str
    identity: tuple[int, int]

    @property
    def workspace(self) -> Path:
        return self.root / self.name / "workspace"

    def release(self) -> None:
        """Remove only this acquisition; missing contents make retries safe.

        Device/inode identity is checked in this process, not persisted across
        nodes. The run controller owns the parent directory; neither Work nor
        Judge may modify it.
        """
        root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                fd = os.open(
                    self.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
            except FileNotFoundError:
                return
            try:
                found = os.fstat(fd)
                if (found.st_dev, found.st_ino) != self.identity:
                    raise InfrastructureError("snapshot directory identity changed")
                try:
                    workspace_fd = _open_copied("workspace", fd)
                except FileNotFoundError:
                    pass
                else:
                    try:
                        if os.fstat(workspace_fd).st_dev != found.st_dev:
                            raise InfrastructureError("snapshot workspace is mounted")
                        _writable_directories(workspace_fd)
                    finally:
                        os.close(workspace_fd)
                    shutil.rmtree("workspace", dir_fd=fd)
                os.fsync(fd)
            finally:
                os.close(fd)
            # An unexpected sibling prevents success; never recursively remove
            # anything beyond the workspace acquired above.
            os.rmdir(self.name, dir_fd=root_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)


@dataclass(frozen=True)
class _Base:
    identity: tuple[int, int]  # rounds/.retained
    manifest: dict[str, bytes]  # path -> entry (see _KEY)
    j0_ns: int  # no Judge could change the copy before this ctime


class _BaseChanged(Exception):
    """A retained file differs from what the Harness wrote."""


@dataclass
class _Walk:
    t0_ns: int  # a file whose ctime is at or after this is not recorded
    base: _Base | None
    source_dev: int
    links_ok: bool
    manifest: dict[str, bytes] = field(default_factory=dict)
    omitted: list[str] = field(default_factory=list)


class JudgeSnapshots:
    """The run's Judge copies under ``rounds`` (see the module docstring)."""

    def __init__(self, rounds: Path) -> None:
        self.rounds = rounds
        self.links_ok = True
        self._base: _Base | None = None
        self._pending: tuple[DirectorySnapshot, _Base | None] | None = None

    def clock(self) -> int:
        """The file system time under ``rounds``. Read it just before the
        pause signal and pass it to ``acquire`` as ``since``."""
        self.rounds.mkdir(parents=True, exist_ok=True, mode=0o700)
        root_fd = os.open(self.rounds, _DIR_FLAGS)
        try:
            return _fs_now_ns(root_fd)
        finally:
            os.close(root_fd)

    def acquire(
        self, source: Path, name: str, *, since: int | None = None
    ) -> DirectorySnapshot:
        """Copy ``source``, with Work paused, to ``rounds/<name>/workspace``.

        ``since`` is ``clock()`` read before Work was sent its pause signal.
        Pass it only when the pause waits until every Work thread has
        stopped: a write sets ctime once, when it starts, so only a write
        that started before ``since`` (and has returned by then) may leave a
        copy that is reused. A later write, even by a Work that woke itself,
        has a ctime of at least ``since`` and is copied again next round.
        Without ``since``, or when ``can_link`` is False, every file is
        copied and nothing is retained.
        """
        try:
            return self._acquire(source, name, since=since)
        except BaseException:
            self.drop()
            raise

    def _acquire(
        self, source: Path, name: str, *, since: int | None
    ) -> DirectorySnapshot:
        if not name or name.startswith(".") or any(c in name for c in "/\\\0"):
            raise InfrastructureError("invalid snapshot round identity")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise InfrastructureError("symlink-safe directory removal unavailable")
        root = self.rounds
        if source.resolve().is_relative_to(
            root.resolve()
        ) or root.resolve().is_relative_to(source.resolve()):
            raise InfrastructureError("snapshot root and workspace overlap")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        base, self._base, self._pending = self._base, None, None
        root_fd = os.open(root, _DIR_FLAGS)
        try:
            os.mkdir(name, mode=0o700, dir_fd=root_fd)
            found = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            lease = DirectorySnapshot(root, name, (found.st_dev, found.st_ino))
            try:
                source_fd = os.open(source, _DIR_FLAGS)
            except PermissionError:
                # Work made its WORKDIR root unreadable (chmod 000, 300...):
                # left out like any directory the Harness cannot open.
                reuse = False
                walk = _copy_closed_root(source, root_fd, lease, self.links_ok)
            else:
                try:
                    reuse = since is not None and _can_link(source_fd, root_fd)
                    walk = self._copy(
                        source_fd, root_fd, lease, base, since if reuse else None
                    )
                finally:
                    os.close(source_fd)
            _remove_retained_at(root_fd)
            if walk.omitted:
                shown = " ".join(_shown(path) for path in walk.omitted[:_OMITTED_SHOWN])
                more = len(walk.omitted) - _OMITTED_SHOWN
                print(
                    f"[rsi-engine] judge_snapshot_omitted: {name} {shown}"
                    + (f" (+{more})" if more > 0 else ""),
                    flush=True,
                )
            pending = None
            if reuse and walk.links_ok:
                # After every Harness link and unlink of these inodes: a later
                # change to one has a ctime of at least j0. The identity is
                # filled in when the copy is retained.
                pending = _Base((0, 0), walk.manifest, _tick(root_fd))
        finally:
            os.close(root_fd)
        self._pending = (lease, pending)
        return lease

    def _copy(
        self,
        source_fd: int,
        root_fd: int,
        lease: DirectorySnapshot,
        base: _Base | None,
        since: int | None,
    ) -> _Walk:
        """Copy the workspace; with ``since``, link from ``base`` and record
        what the next round may link."""
        source_dev = os.fstat(source_fd).st_dev
        retained = _open_retained(root_fd, base)
        if base is not None and retained is None:
            base = None
            _log_base_changed(lease.name)
        if since is None or not self.links_ok:
            base = None
        if base is None and retained is not None:
            # Nothing to link from: remove it first, so it does not take
            # space while the round copies.
            os.close(retained)
            retained = None
            _remove_retained_at(root_fd)
        try:
            if retained is not None:
                # The walk removes what it will not link before copying.
                _writable(retained, source_dev)
            record = since is not None and self.links_ok
            if record:
                # Writes back dirty pages and write-protects their mappings,
                # so Work's next write through a mapping updates ctime. Not
                # needed when nothing will be reused.
                _syncfs(source_fd)
            # 0: no ctime is below it, so nothing is recorded.
            walk = _Walk(since if record else 0, base, source_dev, self.links_ok)
            if not _copy_workspace(source_fd, root_fd, lease, retained, walk):
                # Copy everything, so which Work files the Harness reads does
                # not depend on what changed in the retained copy.
                _log_base_changed(lease.name)
                if retained is not None:
                    os.close(retained)
                    retained = None
                    _remove_retained_at(root_fd)
                walk = _Walk(walk.t0_ns, None, source_dev, self.links_ok)
                _copy_workspace(source_fd, root_fd, lease, None, walk)
        finally:
            if retained is not None:
                os.close(retained)
        self.links_ok = walk.links_ok
        return walk

    def retain(self, lease: DirectorySnapshot) -> None:
        """Keep the finished round's copy as the next round's base, or remove
        it when nothing can be linked from it. Called after the Judge has stopped
        and the round's artifacts are committed."""
        pending, self._pending, self._base = self._pending, None, None
        if pending is None or pending[0] != lease:
            raise InfrastructureError("snapshot was not acquired for this round")
        base = pending[1]
        if base is None or not self.links_ok:
            lease.release()
            return
        root_fd = os.open(lease.root, _DIR_FLAGS)
        try:
            fd = os.open(lease.name, _DIR_FLAGS, dir_fd=root_fd)
            try:
                found = os.fstat(fd)
                if (found.st_dev, found.st_ino) != lease.identity:
                    raise InfrastructureError("snapshot directory identity changed")
                workspace_fd = _open_copied("workspace", fd)
                try:
                    kept = os.fstat(workspace_fd)
                    if kept.st_dev != found.st_dev:
                        raise InfrastructureError("snapshot workspace is mounted")
                    # Moving a directory to another parent needs write
                    # permission on it; its copied mode is not read again.
                    os.fchmod(workspace_fd, stat.S_IMODE(kept.st_mode) | 0o700)
                finally:
                    os.close(workspace_fd)
                try:
                    os.stat(RETAINED, dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise InfrastructureError("a retained Judge copy already exists")
                os.rename("workspace", RETAINED, src_dir_fd=fd, dst_dir_fd=root_fd)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.rmdir(lease.name, dir_fd=root_fd)
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        self._base = replace(base, identity=(kept.st_dev, kept.st_ino))

    def drop(self) -> None:
        """Forget the retained copy: the next round copies everything."""
        self._base = None
        self._pending = None

    def close(self) -> None:
        """Remove the retained copy at the end of the run."""
        self.drop()
        remove_retained(self.rounds)


def can_link(source: Path, rounds: Path) -> bool:
    """Whether a copy of ``source`` under ``rounds`` can link files from the
    previous round's copy (same file system, not tmpfs or ramfs)."""
    source_fd = os.open(source, _DIR_FLAGS)
    try:
        root_fd = os.open(rounds, _DIR_FLAGS)
        try:
            return _can_link(source_fd, root_fd)
        finally:
            os.close(root_fd)
    finally:
        os.close(source_fd)


def _can_link(source_fd: int, root_fd: int) -> bool:
    return os.fstat(source_fd).st_dev == os.fstat(
        root_fd
    ).st_dev and _mapped_writes_change_ctime(source_fd)


def remove_retained(rounds: Path) -> None:
    """Remove ``rounds/.retained``; a missing one is fine."""
    try:
        root_fd = os.open(rounds, _DIR_FLAGS)
    except FileNotFoundError:
        return
    try:
        _remove_retained_at(root_fd)
    finally:
        os.close(root_fd)


def _shown(path: str) -> str:
    """Work names it: one short line, whatever it contains."""
    escaped = path.encode("unicode_escape").decode()
    return escaped if len(escaped) <= 200 else f"{escaped[:200]}..."


def _log_base_changed(name: str) -> None:
    print(f"[rsi-engine] judge_snapshot_base_changed: {name}", flush=True)


def _open_retained(root_fd: int, base: _Base | None) -> int | None:
    """An fd of ``.retained``, never through a symlink; None if missing."""
    try:
        fd = _open_copied(RETAINED, root_fd)
    except FileNotFoundError:
        return None
    except OSError as error:
        if error.errno in (errno.ELOOP, errno.ENOTDIR):
            raise InfrastructureError(
                "retained Judge copy is not a directory"
            ) from error
        raise
    try:
        found = os.fstat(fd)
        if found.st_dev != os.fstat(root_fd).st_dev:
            raise InfrastructureError("retained Judge copy is mounted")
        if base is not None and (found.st_dev, found.st_ino) != base.identity:
            raise InfrastructureError("retained Judge copy identity changed")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _remove_retained_at(root_fd: int) -> None:
    fd = _open_retained(root_fd, None)
    if fd is None:
        return
    try:
        _writable_directories(fd)
    finally:
        os.close(fd)
    shutil.rmtree(RETAINED, dir_fd=root_fd)
    os.fsync(root_fd)


def _copy_workspace(
    source: int,
    root_fd: int,
    lease: DirectorySnapshot,
    retained: int | None,
    walk: _Walk,
) -> bool:
    """Build ``<round>/workspace``; False, leaving nothing, if a retained
    file changed."""
    round_fd = os.open(lease.name, _DIR_FLAGS, dir_fd=root_fd)
    try:
        os.mkdir("workspace", mode=0o700, dir_fd=round_fd)
        target = os.open("workspace", _DIR_FLAGS, dir_fd=round_fd)
        try:
            try:
                _copy_tree(source, target, retained, "", walk)
                _copy_dir_attributes(source, target)
            except _BaseChanged:
                _writable_directories(target)
                changed = True
            else:
                changed = False
        finally:
            os.close(target)
        if changed:
            shutil.rmtree("workspace", dir_fd=round_fd)
        return not changed
    finally:
        os.close(round_fd)


def _copy_closed_root(
    source: Path, root_fd: int, lease: DirectorySnapshot, links_ok: bool
) -> _Walk:
    """An empty ``<round>/workspace`` with the root's mode and times, for a
    root the Harness cannot open. Nothing is recorded for reuse; the caller
    removes any retained copy."""
    found = os.stat(source, follow_symlinks=False)
    round_fd = os.open(lease.name, _DIR_FLAGS, dir_fd=root_fd)
    try:
        os.mkdir("workspace", mode=0o700, dir_fd=round_fd)
        target = os.open("workspace", _DIR_FLAGS, dir_fd=round_fd)
        try:
            os.fchmod(target, stat.S_IMODE(found.st_mode))
            os.utime(target, ns=(found.st_atime_ns, found.st_mtime_ns))
        finally:
            os.close(target)
    finally:
        os.close(round_fd)
    walk = _Walk(0, None, found.st_dev, links_ok)
    walk.omitted.append(".")
    return walk


def _copy_tree(
    source: int,
    target: int,
    base: int | None,
    prefix: str,
    walk: _Walk,
    depth: int = 0,
) -> None:
    names = sorted(os.listdir(source))
    if base is not None:
        # Free retained files this round will not link before copying, so a
        # round holds about one copy of the workspace, not two.
        present = set(names)
        for name in os.listdir(base):
            if name not in present:
                _discard(base, name)
    for name in names:
        path = f"{prefix}{name}"
        try:
            found = os.stat(name, dir_fd=source, follow_symlinks=False)
        except OSError as error:
            if error.errno not in _OMIT:
                raise
            _omit(walk, path, base, name)
            continue
        mode = found.st_mode
        if stat.S_ISDIR(mode):
            _copy_directory(source, target, base, name, path, walk, depth + 1)
        elif stat.S_ISLNK(mode):
            _discard(base, name)
            try:
                link = os.readlink(name, dir_fd=source)
            except OSError as error:
                if error.errno not in _OMIT | {errno.EINVAL}:
                    raise
                walk.omitted.append(path)
                continue
            os.symlink(link, name, dir_fd=target)
            os.utime(
                name,
                ns=(found.st_atime_ns, found.st_mtime_ns),
                dir_fd=target,
                follow_symlinks=False,
            )
        elif stat.S_ISREG(mode):
            if not _link_from_base(target, base, name, path, found, walk):
                _discard(base, name)
                _copy_file(source, target, name, path, walk)
        else:
            _omit(walk, path, base, name)  # FIFO, socket or device


def _omit(walk: _Walk, path: str, base: int | None, name: str) -> None:
    walk.omitted.append(path)
    _discard(base, name)


def _discard(base: int | None, name: str) -> None:
    """Remove ``name`` from a retained directory the walk made writable; a
    missing one is fine."""
    if base is None:
        return
    try:
        found = os.stat(name, dir_fd=base, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(found.st_mode):
        os.unlink(name, dir_fd=base)
        return
    fd = _open_copied(name, base)
    try:
        _writable_directories(fd)
    finally:
        os.close(fd)
    shutil.rmtree(name, dir_fd=base)


def _writable(directory: int, dev: int) -> None:
    """Let the walk remove entries from a retained directory."""
    found = os.fstat(directory)
    if found.st_dev != dev:
        raise InfrastructureError("snapshot contains a mounted filesystem")
    os.fchmod(directory, stat.S_IMODE(found.st_mode) | 0o700)


def _copy_directory(
    source: int,
    target: int,
    base: int | None,
    name: str,
    path: str,
    walk: _Walk,
    depth: int,
) -> None:
    if depth > MAX_DEPTH:
        _omit(walk, path, base, name)
        return
    try:
        child = os.open(name, _DIR_FLAGS, dir_fd=source)
    except OSError as error:
        if error.errno not in _OMIT:
            raise
        _omit(walk, path, base, name)
        return
    try:
        os.mkdir(name, mode=0o700, dir_fd=target)
        copied = os.open(name, _DIR_FLAGS, dir_fd=target)
        try:
            child_base = None
            if base is not None:
                # Missing or not a directory: any unchanged Work file below
                # finds no retained copy, which _link_from_base reports.
                try:
                    child_base = _open_copied(name, base)
                except OSError:
                    _discard(base, name)
                else:
                    try:
                        _writable(child_base, walk.source_dev)
                    except BaseException:
                        os.close(child_base)
                        raise
            try:
                _copy_tree(child, copied, child_base, f"{path}/", walk, depth)
            finally:
                if child_base is not None:
                    os.close(child_base)
            _copy_dir_attributes(child, copied)
        finally:
            os.close(copied)
    finally:
        os.close(child)


def _copy_dir_attributes(source: int, target: int) -> None:
    found = os.fstat(source)
    _copy_xattrs(source, target)
    os.fchmod(target, stat.S_IMODE(found.st_mode))
    os.utime(target, ns=(found.st_atime_ns, found.st_mtime_ns))


def _link_from_base(
    target: int,
    base: int | None,
    name: str,
    path: str,
    found: os.stat_result,
    walk: _Walk,
) -> bool:
    """Hard-link a file Work has not changed from the retained copy."""
    if walk.base is None or found.st_dev != walk.source_dev:
        return False
    # Each path is visited once: the old manifest shrinks as the new grows.
    entry = walk.base.manifest.pop(path, None)
    if entry is None or entry[: _KEY.size] != _work_key(found):
        return False
    # The retained file must be exactly the one the Harness wrote, unchanged
    # since before its Judge could start (ctime cannot be set back).
    if base is None:
        raise _BaseChanged
    try:
        held = os.stat(name, dir_fd=base, follow_symlinks=False)
    except OSError as error:
        raise _BaseChanged from error
    if (
        not stat.S_ISREG(held.st_mode)
        or (held.st_ino, held.st_size) != _COPY.unpack_from(entry, _KEY.size)
        or held.st_ctime_ns >= walk.base.j0_ns
    ):
        raise _BaseChanged
    if not walk.links_ok:
        return False
    try:
        os.link(name, name, src_dir_fd=base, dst_dir_fd=target, follow_symlinks=False)
    except OSError as error:
        if error.errno not in _NO_LINK:
            raise
        walk.links_ok = False
        return False
    walk.manifest[path] = entry
    return True


def _copy_file(source: int, target: int, name: str, path: str, walk: _Walk) -> None:
    try:
        # O_NONBLOCK: a file swapped for a FIFO cannot block the open.
        read_fd = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=source,
        )
    except OSError as error:
        if error.errno not in _OMIT:
            raise
        walk.omitted.append(path)
        return
    try:
        before = os.fstat(read_fd)
        if not stat.S_ISREG(before.st_mode):
            walk.omitted.append(path)
            return
        write_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=target,
        )
        try:
            size = _copy_content(read_fd, write_fd, before.st_size)
            after = os.fstat(read_fd)
            _copy_xattrs(read_fd, write_fd)
            os.fchmod(write_fd, stat.S_IMODE(after.st_mode))
            os.utime(write_fd, ns=(after.st_atime_ns, after.st_mtime_ns))
            copied = os.fstat(write_fd)
        finally:
            os.close(write_fd)
    finally:
        os.close(read_fd)
    # Written after the pause signal or during the copy: copied again next
    # round.
    if (
        _work_key(after) == _work_key(before)
        and after.st_ctime_ns < walk.t0_ns
        and size == after.st_size
    ):
        walk.manifest[path] = _work_key(after) + _COPY.pack(copied.st_ino, size)


def _copy_content(read_fd: int, write_fd: int, size: int) -> int:
    """Up to ``size`` bytes: a file that grows meanwhile is not reused, so
    the copy need not follow it."""
    offset = 0
    try:
        while offset < size:
            sent = os.sendfile(write_fd, read_fd, offset, min(size - offset, 1 << 30))
            if sent == 0:
                return offset
            offset += sent
        return offset
    except OSError as error:
        if offset or error.errno not in (errno.EINVAL, errno.ENOSYS):
            raise
    while offset < size:
        chunk = os.pread(read_fd, min(size - offset, 1 << 20), offset)
        if not chunk:
            break
        offset += os.write(write_fd, chunk)
    return offset


def _copy_xattrs(source: int, target: int) -> None:
    """User extended attributes, best effort, as copytree's copystat."""
    try:
        names = os.listxattr(source)
    except OSError:
        return
    for attribute in names:
        if attribute.startswith("user."):
            with contextlib.suppress(OSError):
                os.setxattr(target, attribute, os.getxattr(source, attribute))


def _fs_now_ns(directory: int) -> int:
    """The file system's own clock: the ctime of a fresh marker file."""
    marker = f".clock-{secrets.token_hex(8)}"
    fd = os.open(
        marker,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        0o600,
        dir_fd=directory,
    )
    try:
        return os.fstat(fd).st_ctime_ns
    finally:
        os.close(fd)
        os.unlink(marker, dir_fd=directory)


def _tick(directory: int) -> int:
    """A file system time later than every change made so far: timestamps
    are coarse, so wait until the clock moves past a first marker."""
    first = _fs_now_ns(directory)
    while True:
        now = _fs_now_ns(directory)
        if now > first:
            return now
        time.sleep(0.001)


def _syncfs(fd: int) -> None:
    try:
        syncfs = ctypes.CDLL(None, use_errno=True).syncfs
    except (OSError, AttributeError):
        os.sync()
        return
    if syncfs(fd) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _mapped_writes_change_ctime(fd: int) -> bool:
    """False on a file system where reuse could miss a write through a
    mapping (see _NO_WRITEBACK), or when that cannot be told."""
    buffer = ctypes.create_string_buffer(256)  # struct statfs, f_type first
    try:
        fstatfs = ctypes.CDLL(None, use_errno=True).fstatfs
    except (OSError, AttributeError):
        return False
    if fstatfs(fd, buffer) != 0:
        return False
    return ctypes.c_long.from_buffer(buffer).value not in _NO_WRITEBACK


def _open_copied(name: str, parent: int) -> int:
    """Open a directory the Harness copied, first giving it owner rwx when
    the mode copied from Work (which Work can change mid-walk) shuts the
    owner out, so no mode stops its removal. Only the Harness changes
    entries under the rounds directory, so between the check and the chmod
    the name cannot turn into a symlink."""
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent)
    except PermissionError:
        found = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISDIR(found.st_mode):
            raise
        os.chmod(name, stat.S_IMODE(found.st_mode) | 0o700, dir_fd=parent)
    return os.open(name, _DIR_FLAGS, dir_fd=parent)


def _writable_directories(fd: int) -> None:
    """Make copied directories removable without changing linked files."""
    found = os.fstat(fd)
    os.fchmod(fd, stat.S_IMODE(found.st_mode) | 0o700)
    for name in os.listdir(fd):
        child = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISDIR(child.st_mode):
            continue
        child_fd = _open_copied(name, fd)
        try:
            if os.fstat(child_fd).st_dev != found.st_dev:
                raise InfrastructureError("snapshot contains a mounted filesystem")
            _writable_directories(child_fd)
        finally:
            os.close(child_fd)
