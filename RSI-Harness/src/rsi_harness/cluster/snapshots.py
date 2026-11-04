"""Disposable directory snapshots for the shared Slurm/LSF runtime.

A lease belongs to one acquisition in the controller process. Release touches
only that directory, after its Judge has stopped. No scan, retention policy,
archive, or adoption of directories from previous runs is involved.
"""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from rsi_harness.errors import InfrastructureError


@dataclass(frozen=True)
class DirectorySnapshot:
    root: Path
    name: str
    identity: tuple[int, int]

    @property
    def workspace(self) -> Path:
        return self.root / self.name / "workspace"

    @classmethod
    def acquire(cls, source: Path, root: Path, name: str) -> DirectorySnapshot:
        if not name or name in (".", "..") or any(c in name for c in "/\\\0"):
            raise InfrastructureError("invalid snapshot round identity")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise InfrastructureError("symlink-safe directory removal unavailable")
        if source.resolve().is_relative_to(
            root.resolve()
        ) or root.resolve().is_relative_to(source.resolve()):
            raise InfrastructureError("snapshot root and workspace overlap")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.mkdir(name, mode=0o700, dir_fd=fd)
            found = os.stat(name, dir_fd=fd, follow_symlinks=False)
            lease = cls(root, name, (found.st_dev, found.st_ino))
        finally:
            os.close(fd)
        # A partial copy is retained on infrastructure failure. It never has
        # a released event and is not adopted by a future round.
        shutil.copytree(source, lease.workspace, symlinks=True)
        return lease

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
                    workspace_fd = os.open(
                        "workspace",
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=fd,
                    )
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


def _writable_directories(fd: int) -> None:
    """Make copied directories removable without changing linked files."""
    found = os.fstat(fd)
    os.fchmod(fd, stat.S_IMODE(found.st_mode) | 0o700)
    for name in os.listdir(fd):
        child = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if not stat.S_ISDIR(child.st_mode):
            continue
        child_fd = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
        )
        try:
            if os.fstat(child_fd).st_dev != found.st_dev:
                raise InfrastructureError("snapshot contains a mounted filesystem")
            _writable_directories(child_fd)
        finally:
            os.close(child_fd)
