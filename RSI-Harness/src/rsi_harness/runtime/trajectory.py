"""Read-only copy of the Agent trajectory for one Judge round.

Every Judge round receives the run's harness-captured Agent output, frozen at
the moment Work was paused for that submission, at ``/logs/agent``. The copy is
taken from this run's artifact directory only, so it never contains another
run's data or anything the Agent did after the submission. Each source file is
bounded: when it exceeds the limit only its most recent bytes are copied, and
``manifest.json`` records the exact byte range, digest, and the source size at
the previous round so a Judge procedure can read only what is new.

The copied bytes are a range of the append-only source file, so after the round
only the manifest is kept; the range can be reproduced from the run artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from rsi_harness.errors import InfrastructureError
from rsi_harness.models import JUDGE_TRAJECTORY_ROOT

TRAJECTORY_DIRECTORY = "judge-trajectory"
TRAJECTORY_MANIFEST = "manifest.json"
TRAJECTORY_SCHEMA_VERSION = 1
TRAJECTORY_SOURCES = ("agent_prompt.md", "agent_output.txt")
TRAJECTORY_FILE_LIMIT_BYTES = 256 * 1024 * 1024
_ROUND_ID = re.compile(r"agent-([1-9][0-9]*)\Z")
_CHUNK_BYTES = 1024 * 1024
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def capture_round_trajectory(
    run_root: Path,
    round_id: str,
    *,
    sources: Sequence[str] = TRAJECTORY_SOURCES,
    limit_bytes: int = TRAJECTORY_FILE_LIMIT_BYTES,
    owner: tuple[int, int] | None = None,
) -> Path:
    """Copy this run's trajectory for ``round_id`` and return the directory.

    Call it while Work is paused. Missing sources are omitted from the
    manifest; a symlinked or non-regular source fails the round.
    """
    number = _round_number(round_id)
    if limit_bytes < 0:
        raise ValueError("trajectory limit must be non-negative")
    parent = run_root / TRAJECTORY_DIRECTORY
    _require_real_directory(run_root)
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    else:
        _set_owner_and_mode(parent, owner, 0o700)
    _require_real_directory(parent)
    target = parent / round_id
    try:
        target.mkdir(mode=0o700)
    except FileExistsError as error:
        raise InfrastructureError(
            f"Judge trajectory for {round_id} already exists"
        ) from error
    previous_round, previous_sizes = _previous_round(parent, number)
    files = []
    for name in sources:
        relative = _relative_source(name)
        entry = _copy_tail(run_root, relative, target, limit_bytes, owner)
        if entry is None:
            continue
        entry["previous_size"] = previous_sizes.get(str(relative))
        files.append(entry)
    manifest = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "round_id": round_id,
        "previous_round_id": previous_round,
        "limit_bytes": limit_bytes,
        "files": files,
    }
    _write_manifest(target / TRAJECTORY_MANIFEST, manifest, owner)
    _publish_directory(target, owner)
    return target


def release_round_trajectory(directory: Path) -> None:
    """Remove a finished round's copied bytes, keeping its manifest.

    Best effort and never raises: it runs during Judge cleanup, and a leftover
    copy only costs disk space.
    """
    try:
        manifest = json.loads((directory / TRAJECTORY_MANIFEST).read_text())
        names = [_relative_source(str(entry["path"])) for entry in manifest["files"]]
    except (OSError, ValueError, KeyError, TypeError):
        return
    for relative in names:
        try:
            (directory / relative).unlink(missing_ok=True)
        except OSError:
            continue
        for parent in relative.parents[:-1]:
            try:
                (directory / parent).rmdir()
            except OSError:
                break


def _round_number(round_id: str) -> int:
    match = _ROUND_ID.fullmatch(round_id)
    if match is None:
        raise InfrastructureError("Judge round identity is invalid")
    return int(match.group(1))


def _relative_source(name: str) -> PurePosixPath:
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.name == TRAJECTORY_MANIFEST
    ):
        raise ValueError(f"trajectory source is not a safe relative path: {name}")
    return relative


def _require_real_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise InfrastructureError(
            f"Judge trajectory directory is missing: {path}"
        ) from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise InfrastructureError(
            f"Judge trajectory directory is not a real directory: {path}"
        )


def _previous_round(parent: Path, number: int) -> tuple[str | None, dict[str, int]]:
    """Source sizes recorded by the latest earlier round of this run."""
    candidates = []
    for child in parent.iterdir():
        match = _ROUND_ID.fullmatch(child.name)
        if match is not None and int(match.group(1)) < number:
            candidates.append((int(match.group(1)), child))
    for _number, child in sorted(candidates, reverse=True):
        manifest_path = child / TRAJECTORY_MANIFEST
        if manifest_path.is_symlink():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
            sizes = {
                str(entry["path"]): int(entry["source_size"])
                for entry in manifest["files"]
            }
        except (OSError, ValueError, KeyError, TypeError):
            continue
        return child.name, sizes
    return None, {}


def _open_source(run_root: Path, relative: PurePosixPath) -> int | None:
    descriptors: list[int] = []
    try:
        descriptors.append(os.open(run_root, _DIRECTORY_FLAGS))
        for component in relative.parts[:-1]:
            descriptors.append(
                os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptors[-1])
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        return os.open(relative.name, flags, dir_fd=descriptors[-1])
    except FileNotFoundError:
        return None
    except OSError as error:
        raise InfrastructureError(
            f"Judge trajectory source {relative} is not a regular file: {error}"
        ) from error
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _copy_tail(
    run_root: Path,
    relative: PurePosixPath,
    target: Path,
    limit_bytes: int,
    owner: tuple[int, int] | None,
) -> dict[str, object] | None:
    source = _open_source(run_root, relative)
    if source is None:
        return None
    try:
        metadata = os.fstat(source)
        if not stat.S_ISREG(metadata.st_mode):
            raise InfrastructureError(
                f"Judge trajectory source {relative} is not a regular file"
            )
        # The size at capture bounds the copy, so bytes appended later are
        # never included even if the source keeps growing.
        size = metadata.st_size
        offset = max(0, size - limit_bytes)
        destination = target / relative
        for parent in reversed(relative.parents[:-1]):
            directory = target / parent
            directory.mkdir(mode=0o755, exist_ok=True)
            _set_owner_and_mode(directory, owner, 0o755)
        digest = hashlib.sha256()
        copied = 0
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        output = os.open(destination, flags, 0o644)
        try:
            while copied < size - offset:
                chunk = os.pread(
                    source,
                    min(_CHUNK_BYTES, size - offset - copied),
                    offset + copied,
                )
                if not chunk:
                    break
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(output, view)
                    view = view[written:]
                copied += len(chunk)
            os.fchmod(output, 0o644)
            if owner is not None:
                os.fchown(output, *owner)
            os.fsync(output)
        finally:
            os.close(output)
    finally:
        os.close(source)
    return {
        "path": str(relative),
        "source_size": size,
        "offset": offset,
        "bytes": copied,
        "sha256": digest.hexdigest(),
        "truncated": offset > 0,
    }


def _write_manifest(
    path: Path, manifest: dict[str, object], owner: tuple[int, int] | None
) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        os.write(descriptor, payload.encode())
        os.fchmod(descriptor, 0o644)
        if owner is not None:
            os.fchown(descriptor, *owner)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(directory: Path, owner: tuple[int, int] | None) -> None:
    # The round directory becomes readable by any Judge user only once it is
    # complete; the private parent keeps it from other host users.
    _set_owner_and_mode(directory, owner, 0o755)
    descriptor = os.open(directory, _DIRECTORY_FLAGS)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _set_owner_and_mode(path: Path, owner: tuple[int, int] | None, mode: int) -> None:
    if owner is not None:
        os.chown(path, *owner, follow_symlinks=False)
    path.chmod(mode)


__all__ = [
    "JUDGE_TRAJECTORY_ROOT",
    "TRAJECTORY_DIRECTORY",
    "TRAJECTORY_FILE_LIMIT_BYTES",
    "TRAJECTORY_MANIFEST",
    "TRAJECTORY_SOURCES",
    "capture_round_trajectory",
    "release_round_trajectory",
]
