"""Unpack an intake bundle safely and verify a Brewer seal.

A bundle is either a Brewer *sealed* bundle (``artifact.json``,
``offline-report.json``, ``task/`` at its root or under one wrapping
directory) or a plain task folder holding ``task.toml``.

Seal format: ``artifact.json["files"]`` lists every entry under ``task/``
in ``sorted(rglob)`` order as ``{path, kind, mode, size, digest}``; ``mode``
is the permission bits, directories have size 0 and the digest of empty
bytes, and digests are SHA-256, written ``sha256:<hex>`` (bare hex in any
case is accepted). ``task_digest`` is the SHA-256 of that list as
``json.dumps(..., sort_keys=True)``; ``offline_report_digest`` is the
SHA-256 of ``offline-report.json``, whose ``task_digest`` must match and
whose ``checks`` must hold no ``failed`` status.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

MAX_FILES = 2000
MAX_UNPACKED_BYTES = 512 * 1024 * 1024
SEALED_NAMES = frozenset({"artifact.json", "offline-report.json", "task"})
_CHUNK = 1024 * 1024


class BundleError(Exception):
    """The bundle is unsafe, malformed or does not match its seal."""


class BundleTooLarge(BundleError):
    """The bundle is over the server's size limit (HTTP 413)."""


@dataclass(frozen=True)
class VerifiedBundle:
    task_dir: Path
    sealed: bool
    task_digest: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- unpacking ---


def _member_path(raw_name: str) -> PurePosixPath | None:
    """The safe relative path of an archive member; None for the root itself."""
    if "\\" in raw_name:
        raise BundleError(f"archive entry uses a backslash: {raw_name!r}")
    if raw_name.startswith("/"):
        raise BundleError(f"archive entry is absolute: {raw_name!r}")
    parts = [part for part in raw_name.split("/") if part not in ("", ".")]
    if ".." in parts:
        raise BundleError(f"archive entry escapes the bundle: {raw_name!r}")
    return PurePosixPath(*parts) if parts else None


class _Extractor:
    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.files = 0
        self.total = 0
        self.dir_modes: dict[Path, int] = {}

    def _target(self, rel: PurePosixPath) -> Path:
        target = self.destination.joinpath(*rel.parts)
        if not target.resolve().is_relative_to(self.destination.resolve()):
            raise BundleError(f"archive entry escapes the bundle: {rel}")
        return target

    def _count(self) -> None:
        self.files += 1
        if self.files > MAX_FILES:
            raise BundleError(f"bundle has more than {MAX_FILES} entries")

    def directory(self, rel: PurePosixPath, mode: int) -> None:
        self._count()
        target = self._target(rel)
        if target.exists() and not target.is_dir():
            raise BundleError(f"archive entry is both a file and a directory: {rel}")
        target.mkdir(parents=True, exist_ok=True)
        self.dir_modes[target] = mode

    def file(self, rel: PurePosixPath, mode: int, source: BinaryIO) -> None:
        self._count()
        target = self._target(rel)
        if target.exists():
            raise BundleError(f"archive repeats an entry: {rel}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out:
            for chunk in iter(lambda: source.read(_CHUNK), b""):
                self.total += len(chunk)
                if self.total > MAX_UNPACKED_BYTES:
                    raise BundleError(
                        f"bundle unpacks to more than {MAX_UNPACKED_BYTES} bytes"
                    )
                out.write(chunk)
        target.chmod(mode)

    def finish(self) -> None:
        # Children first; the owner always keeps rwx so the tree stays readable.
        for path in sorted(self.dir_modes, key=lambda p: len(p.parts), reverse=True):
            path.chmod(self.dir_modes[path] | 0o700)


def _extract_zip(archive: Path, extractor: _Extractor) -> None:
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            rel = _member_path(info.filename)
            unix_mode = info.external_attr >> 16
            if rel is None:
                continue
            if unix_mode and not (stat.S_ISREG(unix_mode) or stat.S_ISDIR(unix_mode)):
                raise BundleError(f"archive entry is a link or special file: {rel}")
            perms = stat.S_IMODE(unix_mode) & 0o777
            if info.is_dir():
                extractor.directory(rel, perms or 0o755)
            else:
                with zf.open(info) as source:
                    extractor.file(rel, perms or 0o644, source)


def _extract_tar(tf: tarfile.TarFile, extractor: _Extractor) -> None:
    for member in tf:
        rel = _member_path(member.name)
        if rel is None:
            continue
        if member.isdir():
            extractor.directory(rel, (member.mode & 0o777) or 0o755)
        elif member.isreg():
            source = tf.extractfile(member)
            if source is None:
                raise BundleError(f"archive entry has no data: {rel}")
            with source:
                extractor.file(rel, (member.mode & 0o777) or 0o644, source)
        else:
            raise BundleError(f"archive entry is a link or special file: {rel}")


def _open_zstd(archive: Path) -> BinaryIO:
    try:
        import zstandard  # type: ignore[import-not-found]
    except ImportError as error:
        raise BundleError(
            ".tar.zst bundles need the 'zstandard' package on the intake server"
        ) from error
    handle = archive.open("rb")
    return zstandard.ZstdDecompressor().stream_reader(handle, closefd=True)


def unpack_bundle(archive: Path, destination: Path) -> None:
    """Unpack a .zip, .tar, .tar.gz/.tgz or .tar.zst bundle into ``destination``."""
    destination.mkdir(parents=True, exist_ok=False)
    extractor = _Extractor(destination)
    with archive.open("rb") as probe:
        magic = probe.read(4)
    try:
        if magic.startswith(b"PK"):
            _extract_zip(archive, extractor)
        elif magic == b"\x28\xb5\x2f\xfd":
            with _open_zstd(archive) as stream:
                with tarfile.open(fileobj=stream, mode="r|") as tf:
                    _extract_tar(tf, extractor)
        else:
            with tarfile.open(archive, mode="r:*") as tf:
                _extract_tar(tf, extractor)
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as error:
        raise BundleError(f"bundle is not a readable archive: {error}") from error
    extractor.finish()


# --- seal verification ---


def _digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _normalized(entries: list[object]) -> list[object]:
    out: list[object] = []
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("digest"), str):
            digest = entry["digest"].strip().lower()
            entry = {**entry, "digest": digest.removeprefix("sha256:")}
        out.append(entry)
    return out


def _same_digest(observed: str, recorded: str) -> bool:
    """``sha256:<hex>`` and bare hex name the same digest, in any case."""

    def bare(value: str) -> str:
        value = value.strip().lower()
        return value[len("sha256:") :] if value.startswith("sha256:") else value

    return bare(observed) == bare(recorded)


def _tree_entries(task_dir: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for item in sorted(task_dir.rglob("*")):
        st = item.lstat()
        is_dir = stat.S_ISDIR(st.st_mode)
        if not (is_dir or stat.S_ISREG(st.st_mode)):
            raise BundleError(f"task tree holds a link or special file: {item.name}")
        entries.append(
            {
                "path": item.relative_to(task_dir).as_posix(),
                "kind": "directory" if is_dir else "file",
                "mode": stat.S_IMODE(st.st_mode) & 0o777,
                "size": 0 if is_dir else st.st_size,
                "digest": _digest_bytes(b"")
                if is_dir
                else "sha256:" + sha256_file(item),
            }
        )
    return entries


def tree_digest(entries: list[dict[str, object]]) -> str:
    return _digest_bytes(json.dumps(entries, sort_keys=True).encode())


def seal_task_tree(task_dir: Path) -> tuple[list[dict[str, object]], str]:
    """The ``files`` list and ``task_digest`` a Brewer seal records for a tree."""
    entries = _tree_entries(task_dir)
    return entries, tree_digest(entries)


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BundleError(f"{path.name} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise BundleError(f"{path.name} must hold a JSON object")
    return value


# Archive litter some tools add next to the real content.
IGNORED_NAMES = frozenset({"__MACOSX", ".DS_Store"})


def _names(directory: Path) -> set[str]:
    return {entry.name for entry in directory.iterdir()} - IGNORED_NAMES


def _locate_root(root: Path) -> tuple[Path, bool]:
    for candidate in (root, *_single_child(root)):
        names = _names(candidate)
        if "artifact.json" in names:
            # Anything carrying a seal must be exactly a sealed bundle: never
            # fall back to running it unsealed.
            if names != SEALED_NAMES:
                extra = ", ".join(sorted(names - SEALED_NAMES)) or "-"
                missing = ", ".join(sorted(SEALED_NAMES - names)) or "-"
                raise BundleError(
                    "sealed bundle root must hold exactly artifact.json, "
                    f"offline-report.json and task/ (extra: {extra}; "
                    f"missing: {missing})"
                )
            return candidate, True
        if "task.toml" in names:
            return candidate, False
    raise BundleError(
        "bundle must be a sealed artifact (artifact.json, offline-report.json, task/) "
        "or a task folder holding task.toml"
    )


def _single_child(root: Path) -> tuple[Path, ...]:
    entries = [entry for entry in root.iterdir() if entry.name not in IGNORED_NAMES]
    if len(entries) == 1 and entries[0].is_dir() and not entries[0].is_symlink():
        return (entries[0],)
    return ()


def _reject_links(root: Path) -> None:
    for item in root.rglob("*"):
        mode = item.lstat().st_mode
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise BundleError(f"task tree holds a link or special file: {item.name}")


def _verify_seal(root: Path) -> str:
    artifact = _read_json(root / "artifact.json")
    report_path = root / "offline-report.json"
    report = _read_json(report_path)
    task_digest = artifact.get("task_digest")
    report_digest = artifact.get("offline_report_digest")
    files = artifact.get("files")
    if not isinstance(task_digest, str) or not isinstance(report_digest, str):
        raise BundleError("artifact.json needs task_digest and offline_report_digest")
    if not isinstance(files, list):
        raise BundleError("artifact.json needs a files list")
    if not _same_digest(_digest_bytes(report_path.read_bytes()), report_digest):
        raise BundleError("offline-report.json does not match artifact.json")
    report_task_digest = report.get("task_digest")
    if not isinstance(report_task_digest, str) or not _same_digest(
        report_task_digest, task_digest
    ):
        raise BundleError(
            "offline-report.json task_digest does not match artifact.json"
        )
    checks = report.get("checks") or []
    if not isinstance(checks, list):
        raise BundleError("offline-report.json checks must be a list")
    failed = [
        str(check.get("id") or check.get("name") or "?")
        for check in checks
        if isinstance(check, dict) and check.get("status") == "failed"
    ]
    if failed or report.get("passed") is False:
        raise BundleError(
            f"offline report has failed checks: {', '.join(failed) or '-'}"
        )
    task_dir = root / "task"
    if task_dir.is_symlink() or not task_dir.is_dir():
        raise BundleError("sealed bundle task/ is missing or unsafe")
    observed, _ = seal_task_tree(task_dir)
    # Compare entries with digests normalised, and hash the list as the
    # Brewer recorded it, so either digest spelling verifies.
    if _normalized(observed) != _normalized(files) or not _same_digest(
        tree_digest(files), task_digest
    ):
        raise BundleError(
            "sealed task files, modes or digests do not match artifact.json"
        )
    return task_digest


def verify_unpacked(root: Path) -> VerifiedBundle:
    bundle_root, sealed = _locate_root(root)
    if not sealed:
        _reject_links(bundle_root)
        return VerifiedBundle(task_dir=bundle_root, sealed=False, task_digest=None)
    digest = _verify_seal(bundle_root)
    return VerifiedBundle(
        task_dir=bundle_root / "task", sealed=True, task_digest=digest
    )


def copy_stream_checked(
    source: BinaryIO, destination: Path, *, max_bytes: int
) -> tuple[str, int]:
    """Copy ``source`` to ``destination``; return its sha256 hex and size."""
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as out:
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                destination.unlink(missing_ok=True)
                raise BundleTooLarge(f"bundle exceeds {max_bytes} bytes")
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest(), size


def reset_dir(path: Path) -> None:
    if path.exists():
        for root, dirs, _ in os.walk(path):
            for name in dirs:
                (Path(root) / name).chmod(0o755)
        shutil.rmtree(path)


__all__ = [
    "BundleError",
    "BundleTooLarge",
    "VerifiedBundle",
    "copy_stream_checked",
    "reset_dir",
    "seal_task_tree",
    "sha256_file",
    "tree_digest",
    "unpack_bundle",
    "verify_unpacked",
]
