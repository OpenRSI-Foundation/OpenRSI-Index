"""Small durable filesystem primitives for persisted runtime authority."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_directory(path: Path) -> None:
    """Durably flush one existing directory entry set."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path: Path, *, mode: int = 0o700) -> None:
    """Create each missing component and fsync both child and parent."""
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("durable directory path must be absolute")
    missing: list[Path] = []
    current = target
    while not current.exists():
        if current.is_symlink():
            raise OSError(f"durable directory component is a symlink: {current}")
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    if current.is_symlink() or not current.is_dir():
        raise OSError(f"durable directory ancestor is unsafe: {current}")
    for directory in reversed(missing):
        parent = directory.parent
        try:
            os.mkdir(directory, mode)
        except FileExistsError:
            if directory.is_symlink() or not directory.is_dir():
                raise
        fsync_directory(parent)
        fsync_directory(directory)


__all__ = ["durable_mkdir", "fsync_directory"]
