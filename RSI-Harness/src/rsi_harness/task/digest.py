"""Contained deterministic directory digests."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from rsi_harness.errors import UnsupportedTaskError


def _add_field(digest: object, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def hash_tree(root: Path, *, excluded_roots: tuple[str, ...] = ()) -> str:
    """Hash a tree without following links or depending on directory order."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise UnsupportedTaskError(f"digest root is not a directory: {root}")
    excluded = frozenset(excluded_roots)
    if any(Path(name).name != name or name in {"", ".", ".."} for name in excluded):
        raise ValueError("excluded roots must be single top-level names")

    entries = _collect_entries(root, excluded)
    digest = hashlib.sha256()
    for path in entries:
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"directory"
            payload = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"file"
            payload = _read_regular_file(path)
        elif stat.S_ISLNK(metadata.st_mode):
            kind = b"symlink"
            _require_contained_symlink(path, root)
            payload = os.readlink(path).encode()
        else:
            raise UnsupportedTaskError(
                f"unsupported filesystem entry in task: {relative}"
            )
        _add_field(digest, relative.encode())
        _add_field(digest, str(mode).encode())
        _add_field(digest, kind)
        _add_field(digest, payload)
    return digest.hexdigest()


def _collect_entries(root: Path, excluded: frozenset[str]) -> list[Path]:
    collected: list[Path] = []

    def visit(directory: Path) -> None:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root)
            if len(relative.parts) == 1 and child.name in excluded:
                continue
            collected.append(path)
            if child.is_dir(follow_symlinks=False):
                visit(path)

    visit(root)
    return sorted(collected, key=lambda path: path.relative_to(root).as_posix())


def _require_contained_symlink(path: Path, root: Path) -> None:
    try:
        target = path.resolve(strict=False)
        target.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        raise UnsupportedTaskError(
            f"task symlink resolves outside the task root: {path.relative_to(root)}"
        ) from error


def _read_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise UnsupportedTaskError(
            f"unable to hash task file {path}: {error}"
        ) from error
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()
