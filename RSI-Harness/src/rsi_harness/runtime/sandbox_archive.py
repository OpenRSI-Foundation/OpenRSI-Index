"""Host-side archive transfer for env services: canonical tar, stages, path-stat.

Copies go only through the Engine archive API (HEAD/GET/PUT archive); no code
ever runs in a child, so images without python3 or a shell work alike. Every
tar the broker hands to Docker or back to a caller is canonical: regular
files, directories and symlinks only, normalized relative names, mode & 0o777,
uid = gid = 0 and no extended attributes. Symlinks are only ever resolved by
the daemon inside the container's scope.

The daemon extracts into the container rootfs without its tmpfs mounts and
happily writes into a paused container, so both are refused here rather than
silently writing where the service can never see it. Every tar is read as a
stream with bounded headers (``BoundedTarInfo``); nothing keeps per-member
state beyond fixed-size keys.
"""

from __future__ import annotations

import base64
import binascii
import fnmatch
import hashlib
import io
import json
import logging
import os
import posixpath
import re
import secrets
import stat as stat_module
import tarfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from docker.errors import APIError

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.sandbox_contracts import SandboxError, absolute_path, below

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
MAX_STAGE_FRAME = 16 * MIB
STAGE_TTL_SEC = 600.0
MAX_ARCHIVE_ENTRIES = 100_000
MAX_ARCHIVE_DEPTH = 64
MAX_NAME_BYTES = 4096
MAX_LINK_TARGET = 4096
MAX_EXCLUDES = 64
# A PAX or GNU long-name body holds at most a 4 KiB path and link target plus
# a few records; tarfile would otherwise buffer any claimed size whole.
MAX_EXTENDED_HEADER = 64 * 1024
MAX_EXTENDED_CHAIN = 8
# Existing directories one copy_in may resolve (one path-stat each).
MAX_RESOLVED_DIRS = 16_384
STAGE_ID = re.compile(r"^s[0-9a-f]{32}$")
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
PATH_STAT_HEADER = "X-Docker-Container-Path-Stat"
# Kernel filesystems and the container's /dev tmpfs are never copy targets.
SPECIAL_ROOTS = ("/dev", "/proc", "/sys")
# Service states whose filesystem the daemon can archive without a thaw.
TRANSFERABLE_STATES = ("created", "running", "exited")
_MTIME_MAX = 8**11 - 1  # the ustar field; larger times would need a PAX record

# Go io/fs.FileMode bits, as the daemon encodes path-stat modes.
_MODE_DIR = 1 << 31
_MODE_SYMLINK = 1 << 27
_MODE_TYPE = (
    _MODE_DIR
    | _MODE_SYMLINK
    | (1 << 26)  # device
    | (1 << 25)  # named pipe
    | (1 << 24)  # socket
    | (1 << 21)  # char device
    | (1 << 19)  # irregular
)
_MODE_SETUID = 1 << 23
_MODE_SETGID = 1 << 22
_MODE_STICKY = 1 << 20
_STAT_FIELDS = frozenset({"name", "size", "mode", "mtime", "linkTarget"})
_EXTENDED_TYPES = frozenset(
    {
        tarfile.XHDTYPE,
        tarfile.XGLTYPE,
        tarfile.SOLARIS_XHDTYPE,
        tarfile.GNUTYPE_LONGNAME,
        tarfile.GNUTYPE_LONGLINK,
    }
)
_CHAIN = "_rsi_extended_chain"
_GLOBAL = "_rsi_global_bytes"


# -- path-stat (lifted from archive/vm-evaluation native_copy_gate.py) --------


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate path-stat field")
        result[key] = value
    return result


def _no_constant(name: str) -> Any:
    raise ValueError(f"non-finite path-stat value {name}")


def decode_path_stat(value: object) -> dict[str, Any]:
    """Strictly decode the base64 JSON ``X-Docker-Container-Path-Stat`` header."""
    if type(value) is not str or not value or len(value) > 16384:
        raise SandboxError("infrastructure", "path", "archive path stat bound differs")
    try:
        raw = base64.b64decode(value, validate=True)
        if not 0 < len(raw) <= 12288:
            raise ValueError("path stat size")
        result = json.loads(
            raw, object_pairs_hook=_unique_pairs, parse_constant=_no_constant
        )
    except (ValueError, binascii.Error):
        raise SandboxError(
            "infrastructure", "path", "archive path stat encoding differs"
        ) from None
    if (
        type(result) is not dict
        or set(result) != _STAT_FIELDS
        or any(
            type(result[key]) is not str
            or len(result[key].encode()) > 4096
            or "\x00" in result[key]
            for key in ("name", "mtime", "linkTarget")
        )
        or type(result["size"]) is not int
        or not 0 <= result["size"] < 2**63
        or type(result["mode"]) is not int
        or not 0 <= result["mode"] < 2**32
    ):
        raise SandboxError("infrastructure", "path", "archive path stat schema differs")
    return result


def stat_kind(stat: Mapping[str, Any]) -> str:
    """``file``, ``dir``, ``symlink`` or ``other`` from a decoded path stat."""
    kind = stat["mode"] & _MODE_TYPE
    return {0: "file", _MODE_DIR: "dir", _MODE_SYMLINK: "symlink"}.get(kind, "other")


def path_stat_view(stat: Mapping[str, Any] | None) -> dict[str, Any]:
    """The wire ``path_stat`` result; ``mode`` uses Unix permission bits."""
    if stat is None:
        return {
            "exists": False,
            "kind": None,
            "size": None,
            "mode": None,
            "mtime": None,
            "link_target": None,
        }
    mode = stat["mode"]
    permissions = (
        (mode & 0o777)
        | (stat_module.S_ISUID if mode & _MODE_SETUID else 0)
        | (stat_module.S_ISGID if mode & _MODE_SETGID else 0)
        | (stat_module.S_ISVTX if mode & _MODE_STICKY else 0)
    )
    return {
        "exists": True,
        "kind": stat_kind(stat),
        "size": stat["size"],
        "mode": permissions,
        "mtime": stat["mtime"],
        "link_target": stat["linkTarget"] or None,
    }


# -- bounded tar reading ------------------------------------------------------


class BoundedTarInfo(tarfile.TarInfo):
    """Refuses unbounded headers before tarfile buffers them.

    CPython reads a PAX or GNU long-name body whole (and a GNU sparse map
    block by block) before any caller sees the member, so an untrusted
    stream could make it allocate whatever size a header claims. Each
    extended header is capped at 64 KiB, at most 8 precede one member,
    global PAX headers share one 64 KiB budget, and sparse members are
    refused outright.
    """

    def _proc_member(self, archive: tarfile.TarFile) -> tarfile.TarInfo:
        if self.type in _EXTENDED_TYPES:
            chain = archive.__dict__.get(_CHAIN, 0) + 1
            total = archive.__dict__.get(_GLOBAL, 0)
            if self.type == tarfile.XGLTYPE:
                total += self.size
            if (
                self.size > MAX_EXTENDED_HEADER
                or chain > MAX_EXTENDED_CHAIN
                or total > MAX_EXTENDED_HEADER
            ):
                raise tarfile.HeaderError("extended tar header exceeds its bound")
            archive.__dict__[_CHAIN] = chain
            archive.__dict__[_GLOBAL] = total
        else:
            archive.__dict__[_CHAIN] = 0
            if self.type == tarfile.GNUTYPE_SPARSE:
                raise _sparse()
        return super()._proc_member(archive)

    # PAX-described GNU sparse maps, versions 0.0, 0.1 and 1.0 (the last one
    # reads its map from the stream without a bound).
    def _proc_gnusparse_00(self, next: tarfile.TarInfo, raw_headers: Any) -> None:
        raise _sparse()

    def _proc_gnusparse_01(self, next: tarfile.TarInfo, pax_headers: Any) -> None:
        raise _sparse()

    def _proc_gnusparse_10(
        self, next: tarfile.TarInfo, pax_headers: Any, archive: tarfile.TarFile
    ) -> None:
        raise _sparse()


def _sparse() -> tarfile.HeaderError:
    return tarfile.HeaderError("sparse tar members are not accepted")


class BoundedTarFile(tarfile.TarFile):
    """Forward-only reading with bounded headers and no member cache.

    TarFile keeps every TarInfo it reads, even in ``r|`` mode before Python
    3.13's ``stream=True``; a forward-only reader never needs them.
    """

    tarinfo = BoundedTarInfo

    def next(self) -> tarfile.TarInfo | None:
        member = super().next()
        self.members.clear()
        return member


def read_tar(source: IO[bytes], *, errors: str = "strict") -> tarfile.TarFile:
    """Open an untrusted tar stream for one forward pass."""
    return BoundedTarFile.open(
        fileobj=source, mode="r|", encoding="utf-8", errors=errors
    )


# -- canonical tar ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArchiveSummary:
    entries: int
    bytes: int
    skipped: int = 0


class _Rejected(ValueError):
    """A member the canonical form cannot express (not a quota)."""


def entry_parts(name: object) -> tuple[str, ...]:
    """Normalized relative components of a member name; ``()`` is the root."""
    if type(name) is not str or "\x00" in name:
        raise _Rejected("entry name is not a string")
    try:
        encoded = name.encode("utf-8")
    except UnicodeEncodeError:
        raise _Rejected("entry name is not UTF-8") from None
    if len(encoded) > MAX_NAME_BYTES or name.startswith("/"):
        raise _Rejected("entry name must be a bounded relative path")
    parts = tuple(part for part in name.split("/") if part not in ("", "."))
    if ".." in parts:
        raise _Rejected("entry name must not contain '..'")
    if len(parts) > MAX_ARCHIVE_DEPTH:
        raise _Rejected(f"entry deeper than {MAX_ARCHIVE_DEPTH} components")
    return parts


def _key(parts: Sequence[str]) -> bytes:
    # Fixed-size identities: 100000 entries stay within about 35 MiB.
    return hashlib.blake2b("/".join(parts).encode(), digest_size=16).digest()


def _link_target(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or len(value.encode("utf-8", "surrogateescape")) > MAX_LINK_TARGET
    ):
        raise _Rejected("symlink target must be 1..4096 bytes")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _Rejected("symlink target is not UTF-8") from None
    return value


class CanonicalTarWriter:
    """Stream validated members into one canonical tar with bounded state.

    Rejected layouts: duplicates (other than repeated directories), entries
    below a file or symlink of the same archive, and anything beyond the
    entry or byte bounds. ``fileobj`` must be a real, seekable file so a
    hardlink can later be re-emitted from its already written data.
    """

    def __init__(
        self,
        fileobj: IO[bytes],
        *,
        max_bytes: int,
        max_entries: int = MAX_ARCHIVE_ENTRIES,
    ) -> None:
        self._file = fileobj
        self._max_bytes = max_bytes
        self._max_entries = max_entries
        self._kinds: dict[bytes, str] = {}
        self._data: dict[bytes, tuple[int, int]] = {}
        self._archive = tarfile.open(
            fileobj=fileobj, mode="w", format=tarfile.PAX_FORMAT, encoding="utf-8"
        )
        self.entries = 0
        self.bytes = 0

    def _admit(self, parts: tuple[str, ...], kind: str) -> bool:
        """False for a repeated directory, which is written only once."""
        if not parts:
            raise _Rejected("only a directory can be the archive root")
        for depth in range(1, len(parts)):
            parent = self._kinds.get(_key(parts[:depth]))
            if parent is not None and parent != "dir":
                raise _Rejected("entry below a file or symlink of the same archive")
        existing = self._kinds.get(_key(parts))
        if existing is not None:
            if existing == kind == "dir":
                return False
            raise _Rejected("duplicate archive entry")
        if self.entries >= self._max_entries:
            raise SandboxError(
                "quota", "entries", f"archive exceeds {self._max_entries} entries"
            )
        return True

    def _info(self, parts: tuple[str, ...], kind: bytes, mode: int, mtime: Any):
        info = tarfile.TarInfo("/".join(parts))
        if self.entries == 0:
            # The daemon sniffs put_archive bodies for compression magic; a
            # leading PAX header keeps a name like "BZh..." from matching.
            info.pax_headers = {"path": info.name}
        info.type = kind
        info.mode = int(mode) & 0o777
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        try:
            info.mtime = min(max(int(mtime), 0), _MTIME_MAX)
        except (TypeError, ValueError, OverflowError):
            info.mtime = 0
        return info

    def _add(self, info: tarfile.TarInfo, source: IO[bytes] | None = None) -> None:
        self._archive.addfile(info, source)
        # TarFile keeps every written TarInfo; only fixed-size keys stay.
        self._archive.members.clear()

    def add_dir(self, parts: tuple[str, ...], *, mode: int, mtime: Any) -> None:
        if not self._admit(parts, "dir"):
            return
        self._add(self._info(parts, tarfile.DIRTYPE, mode, mtime))
        self._kinds[_key(parts)] = "dir"
        self.entries += 1

    def add_symlink(
        self, parts: tuple[str, ...], target: str, *, mode: int, mtime: Any
    ) -> None:
        target = _link_target(target)
        self._admit(parts, "symlink")
        info = self._info(parts, tarfile.SYMTYPE, mode, mtime)
        info.linkname = target
        self._add(info)
        self._kinds[_key(parts)] = "symlink"
        self.entries += 1

    def add_file(
        self,
        parts: tuple[str, ...],
        source: IO[bytes],
        size: int,
        *,
        mode: int,
        mtime: Any,
    ) -> None:
        if type(size) is not int or size < 0:
            raise _Rejected("regular file size is invalid")
        self._admit(parts, "file")
        if self.bytes + size > self._max_bytes:
            raise SandboxError(
                "quota", "bytes", f"archive exceeds {self._max_bytes} content bytes"
            )
        info = self._info(parts, tarfile.REGTYPE, mode, mtime)
        info.size = size
        self._add(info, source)
        blocks = -(-size // tarfile.BLOCKSIZE) * tarfile.BLOCKSIZE
        self._data[_key(parts)] = (self._archive.offset - blocks, size)
        self._kinds[_key(parts)] = "file"
        self.entries += 1
        self.bytes += size

    def add_hardlink_copy(
        self,
        parts: tuple[str, ...],
        target: tuple[str, ...],
        *,
        mode: int,
        mtime: Any,
    ) -> bool:
        """Re-emit a hardlink as a regular file; False if its data is unknown."""
        known = self._data.get(_key(target))
        if known is None:
            return False
        offset, size = known
        self._file.flush()
        self.add_file(
            parts,
            _RegionReader(self._file.fileno(), offset, size),
            size,
            mode=mode,
            mtime=mtime,
        )
        return True

    def close(self) -> ArchiveSummary:
        self._archive.close()
        self._file.flush()
        return ArchiveSummary(entries=self.entries, bytes=self.bytes)


class _RegionReader(io.RawIOBase):
    def __init__(self, descriptor: int, offset: int, size: int) -> None:
        self._descriptor = descriptor
        self._offset = offset
        self._remaining = size

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self._remaining)
        if count == 0:
            return 0
        data = os.pread(self._descriptor, count, self._offset)
        if not data:
            raise OSError("canonical archive region is truncated")
        buffer[: len(data)] = data
        self._offset += len(data)
        self._remaining -= len(data)
        return len(data)


def canonicalize_tar(
    source: IO[bytes], dest: IO[bytes], *, max_bytes: int
) -> ArchiveSummary:
    """Validate an uploaded tar stream and write its canonical form.

    Accepts only regular files, directories and symlinks with relative names
    free of ``..``; hardlinks, devices, FIFOs and absolute names are invalid.
    A root ``.`` directory entry is dropped: modes of existing destination
    directories are never changed by a copy.
    """
    writer = CanonicalTarWriter(dest, max_bytes=max_bytes)
    try:
        with read_tar(source) as archive:
            for member in archive:
                parts = entry_parts(member.name)
                if member.isdir():
                    if parts:
                        writer.add_dir(parts, mode=member.mode, mtime=member.mtime)
                elif member.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    reader = archive.extractfile(member)
                    assert reader is not None
                    writer.add_file(
                        parts, reader, member.size, mode=member.mode, mtime=member.mtime
                    )
                elif member.issym():
                    writer.add_symlink(
                        parts, member.linkname, mode=member.mode, mtime=member.mtime
                    )
                else:
                    raise _Rejected(
                        "only regular files, directories and symlinks are accepted"
                    )
    except SandboxError:
        raise
    except (tarfile.TarError, UnicodeError, ValueError, OSError, EOFError) as error:
        raise SandboxError("invalid", "stage", f"invalid tar stream: {error}") from None
    return writer.close()


def directory_tar(parts: Sequence[str], mode: int = 0o755) -> bytes:
    """Canonical tar creating ``parts`` as nested directories (root-owned)."""
    buffer = io.BytesIO()
    writer = CanonicalTarWriter(buffer, max_bytes=0)
    for depth in range(1, len(parts) + 1):
        writer.add_dir(tuple(parts[:depth]), mode=mode, mtime=time.time())
    writer.close()
    return buffer.getvalue()


# -- stages -------------------------------------------------------------------


@dataclass(slots=True)
class _Stage:
    ready: bool
    size: int
    last_used: float
    digest: Any = None
    entries: int | None = None
    content_bytes: int = 0
    busy: bool = False


class _ResultStage:
    def __init__(self, stage_id: str, file: IO[bytes]) -> None:
        self.stage_id = stage_id
        self.file = file
        self.summary: ArchiveSummary | None = None


class StageStore:
    """One session's spool of staged tars: a 0700 directory of 0600 files.

    Uploads arrive in sequential frames of at most 16 MiB and become a
    canonical tar on the final frame, after the whole-body SHA-256 matches.
    A stage is consumed at most once and removed after ``ttl_sec`` unused.
    Byte accounting (max_upload_bytes) stays with the broker.
    """

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], float] = time.monotonic,
        ttl_sec: float = STAGE_TTL_SEC,
    ) -> None:
        root = Path(root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._dir = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(self._dir)
        if info.st_uid != os.geteuid():
            os.close(self._dir)
            raise InfrastructureError("sandbox stage spool is not broker-owned")
        os.fchmod(self._dir, 0o700)
        self._root = root
        self._clock = clock
        self._ttl = ttl_sec
        self._lock = threading.Lock()
        self._stages: dict[str, _Stage] = {}

    def _open(self, name: str, flags: int) -> int:
        return os.open(
            name,
            flags | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=self._dir,
        )

    def _unlink(self, name: str) -> None:
        try:
            os.unlink(name, dir_fd=self._dir)
        except FileNotFoundError:
            pass

    def _drop(self, stage_id: str) -> None:
        self._unlink(f"{stage_id}.part")
        self._unlink(f"{stage_id}.tar")

    def _record(self, stage_id: object, *, ready: bool) -> _Stage:
        if type(stage_id) is not str or STAGE_ID.fullmatch(stage_id) is None:
            raise SandboxError("invalid", "stage_id", "expected an s<32hex> stage")
        record = self._stages.get(stage_id)
        if record is None:
            raise SandboxError("invalid", "stage_id", "unknown or expired stage")
        if record.busy:
            raise SandboxError("busy", "stage_id", "stage is in use")
        if record.ready != ready:
            state = "is not final yet" if ready else "is already final"
            raise SandboxError("invalid", "stage_id", f"stage {state}")
        return record

    def put(
        self,
        stage_id: str | None,
        offset: int,
        data: bytes,
        *,
        final: bool,
        sha256: str | None,
        max_bytes: int,
    ) -> dict[str, Any]:
        """Append one frame; the final frame canonicalizes the whole stage."""
        if type(data) is not bytes:
            raise SandboxError("invalid", "payload", "expected binary stage data")
        if len(data) > MAX_STAGE_FRAME:
            raise SandboxError("quota", "payload", "stage frame exceeds 16 MiB")
        if type(offset) is not int or offset < 0:
            raise SandboxError("invalid", "offset", "expected a nonnegative offset")
        if type(final) is not bool:
            raise SandboxError("invalid", "final", "expected a boolean")
        if final and (type(sha256) is not str or not SHA256_HEX.fullmatch(sha256)):
            raise SandboxError("invalid", "sha256", "final frame needs the body hash")
        if not final and sha256 is not None:
            raise SandboxError("invalid", "sha256", "only the final frame has a hash")
        with self._lock:
            if stage_id is None:
                if offset != 0:
                    raise SandboxError("invalid", "offset", "a new stage starts at 0")
                if len(data) > max_bytes:
                    raise SandboxError(
                        "quota", "stage", "stage exceeds the upload budget"
                    )
                stage_id = "s" + secrets.token_hex(16)
                descriptor = self._open(
                    f"{stage_id}.part", os.O_WRONLY | os.O_CREAT | os.O_EXCL
                )
                os.close(descriptor)
                record = _Stage(
                    ready=False,
                    size=0,
                    last_used=self._clock(),
                    digest=hashlib.sha256(),
                )
                self._stages[stage_id] = record
            else:
                record = self._record(stage_id, ready=False)
                if offset != record.size:
                    raise SandboxError(
                        "invalid", "offset", f"expected offset {record.size}"
                    )
            if record.size + len(data) > max_bytes:
                raise SandboxError("quota", "stage", "stage exceeds the upload budget")
            record.busy = True
        try:
            descriptor = self._open(f"{stage_id}.part", os.O_WRONLY)
            try:
                written = os.pwrite(descriptor, data, offset)
                if written != len(data):
                    raise OSError("short stage write")
            finally:
                os.close(descriptor)
            record.digest.update(data)
            record.size += len(data)
            if final:
                self._finish(stage_id, record, sha256)
        except BaseException:
            with self._lock:
                self._stages.pop(stage_id, None)
                self._drop(stage_id)
            raise
        with self._lock:
            record.busy = False
            record.last_used = self._clock()
        return {
            "stage_id": stage_id,
            "bytes": record.size,
            "entries": record.entries if final else None,
        }

    def _finish(self, stage_id: str, record: _Stage, sha256: str | None) -> None:
        if record.digest.hexdigest() != sha256:
            raise SandboxError("invalid", "sha256", "stage body hash differs")
        source = self._open(f"{stage_id}.part", os.O_RDONLY)
        target = self._open(f"{stage_id}.tar", os.O_RDWR | os.O_CREAT | os.O_EXCL)
        with open(source, "rb") as raw, open(target, "w+b") as canonical:
            summary = canonicalize_tar(raw, canonical, max_bytes=record.size)
        self._unlink(f"{stage_id}.part")
        record.ready = True
        record.digest = None
        record.entries = summary.entries
        record.content_bytes = summary.bytes

    def read(self, stage_id: str, offset: int, length: int) -> bytes:
        """``stage_get``: a bounded slice of a final stage."""
        if type(offset) is not int or offset < 0:
            raise SandboxError("invalid", "offset", "expected a nonnegative offset")
        if type(length) is not int or not 0 <= length <= MAX_STAGE_FRAME:
            raise SandboxError("invalid", "length", "length must be 0..16 MiB")
        with self._lock:
            record = self._record(stage_id, ready=True)
            if offset > record.size:
                raise SandboxError("invalid", "offset", "offset beyond the stage")
            record.last_used = self._clock()
            descriptor = self._open(f"{stage_id}.tar", os.O_RDONLY)
        try:
            return os.pread(descriptor, length, offset)
        finally:
            os.close(descriptor)

    def size(self, stage_id: str) -> int:
        with self._lock:
            return self._record(stage_id, ready=True).size

    @contextmanager
    def consume(self, stage_id: str) -> Iterator[tuple[IO[bytes], ArchiveSummary]]:
        """Hand out a final stage exactly once; it is deleted afterwards."""
        with self._lock:
            record = self._record(stage_id, ready=True)
            del self._stages[stage_id]
            descriptor = self._open(f"{stage_id}.tar", os.O_RDONLY)
        summary = ArchiveSummary(
            entries=record.entries or 0, bytes=record.content_bytes
        )
        try:
            with open(descriptor, "rb") as file:
                yield file, summary
        finally:
            self._drop(stage_id)

    @contextmanager
    def create_result(self) -> Iterator[_ResultStage]:
        """A new stage written by the broker (``copy_out``); kept on success."""
        stage_id = "s" + secrets.token_hex(16)
        with self._lock:
            descriptor = self._open(
                f"{stage_id}.tar", os.O_RDWR | os.O_CREAT | os.O_EXCL
            )
        result = _ResultStage(stage_id, open(descriptor, "w+b"))
        try:
            yield result
            result.file.flush()
            size = os.fstat(result.file.fileno()).st_size
            summary = result.summary or ArchiveSummary(entries=0, bytes=0)
        except BaseException:
            result.file.close()
            self._drop(stage_id)
            raise
        result.file.close()
        with self._lock:
            self._stages[stage_id] = _Stage(
                ready=True,
                size=size,
                last_used=self._clock(),
                entries=summary.entries,
                content_bytes=summary.bytes,
            )

    def discard(self, stage_id: str) -> None:
        with self._lock:
            record = self._stages.get(stage_id)
            if record is None or record.busy:
                return
            del self._stages[stage_id]
            self._drop(stage_id)

    def sweep(self) -> tuple[str, ...]:
        """Remove stages unused for ``ttl_sec``; returns their IDs."""
        now = self._clock()
        with self._lock:
            stale = tuple(
                stage_id
                for stage_id, record in self._stages.items()
                if not record.busy and now - record.last_used >= self._ttl
            )
            for stage_id in stale:
                del self._stages[stage_id]
                self._drop(stage_id)
        return stale

    def close(self) -> None:
        with self._lock:
            for stage_id in tuple(self._stages):
                self._drop(stage_id)
            self._stages.clear()
            os.close(self._dir)


# -- archive transfer ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArchiveTarget:
    """One owned service container as the archive API sees it.

    ``tmpfs`` lists the service's tmpfs mount roots (including /dev/shm);
    ``inspect`` returns identity-attested live inspect attributes.
    """

    container_id: str
    tmpfs: tuple[str, ...]
    inspect: Callable[[], Mapping[str, Any]]

    def shadowed(self, path: str) -> bool:
        return any(below(path, root) for root in (*SPECIAL_ROOTS, *self.tmpfs))

    def refuse_shadowed(self, path: str, field: str) -> None:
        # The daemon writes and reads the layer under a tmpfs, never the tmpfs.
        if self.shadowed(path):
            raise SandboxError(
                "unsupported", field, "tmpfs and kernel filesystems are not copyable"
            )

    def shadow_roots_below(self, path: str) -> tuple[tuple[str, ...], ...]:
        """Relative components of every shadowing root strictly below ``path``."""
        roots = []
        for root in (*SPECIAL_ROOTS, *self.tmpfs):
            if root != path and below(root, path):
                relative = posixpath.relpath(root, path)
                roots.append(tuple(relative.split("/")))
        return tuple(roots)


def _under(parts: Sequence[str], roots: Sequence[Sequence[str]]) -> bool:
    return any(tuple(parts[: len(root)]) == tuple(root) for root in roots)


class _Dir:
    """One directory prefix of a staged archive as the daemon resolves it."""

    __slots__ = ("children", "exists", "path")

    def __init__(self, path: str, exists: bool) -> None:
        self.path = path
        self.exists = exists
        self.children: dict[str, _Dir] = {}


def _reaches_tmpfs() -> SandboxError:
    return SandboxError(
        "unsupported", "dest_dir", "archive entries reach a tmpfs mount"
    )


def _exclusions(patterns: object) -> tuple[re.Pattern[str], ...]:
    """GNU tar ``--exclude`` semantics: unanchored, wildcards match '/'."""
    if not isinstance(patterns, (list, tuple)) or len(patterns) > MAX_EXCLUDES:
        raise SandboxError("invalid", "exclude", "expected at most 64 glob patterns")
    compiled = []
    for pattern in patterns:
        if (
            type(pattern) is not str
            or not pattern
            or "\x00" in pattern
            or len(pattern.encode()) > MAX_NAME_BYTES
        ):
            raise SandboxError("invalid", "exclude", "expected bounded glob strings")
        while pattern.startswith("./"):
            pattern = pattern[2:]
        compiled.append(re.compile(r"(?s:.*/)?" + fnmatch.translate(pattern)))
    return tuple(compiled)


def _excluded(relative: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> bool:
    name = "/".join(relative)
    return any(pattern.fullmatch(name) for pattern in patterns)


# Fixed refusal texts: Engine explanations can name host paths or IDs.
_REFUSALS = {
    403: "the target is read-only",
    409: "the service state does not allow it",
}


def _archive_error(error: Exception, field: str, action: str) -> SandboxError:
    status = getattr(error, "status_code", None)
    if isinstance(error, APIError) and status is not None and 400 <= status < 500:
        explanation = getattr(error, "explanation", None) or str(error)
        LOGGER.info("sandbox %s (%s): %s", action, status, explanation)
        refusal = _REFUSALS.get(status, "the daemon refused the request")
        return SandboxError("invalid", field, f"{action}: {refusal}")
    return SandboxError("unknown-outcome", field, f"{action} did not complete")


class ArchiveTransfer:
    """``copy_in``/``copy_out``/``path_stat`` through the Engine archive API."""

    def __init__(self, client: Any, stages: StageStore) -> None:
        self._api = client.api
        self._stages = stages

    def _url(self, container_id: str) -> str:
        return self._api._url("/containers/{0}/archive", container_id)

    @staticmethod
    def _require_transferable(target: ArchiveTarget) -> None:
        state = target.inspect().get("State") or {}
        if state.get("Paused"):
            raise SandboxError("busy", "service", "a paused service cannot transfer")
        status = state.get("Status")
        if status not in TRANSFERABLE_STATES:
            raise SandboxError("busy", "service", f"a {status} service cannot transfer")

    def _head(self, container_id: str, path: str) -> dict[str, Any] | None:
        try:
            response = self._api.head(
                self._url(container_id),
                params={"path": path},
                **self._api._set_request_timeout({}),
            )
        except Exception as error:
            raise SandboxError(
                "unknown-outcome", "path", "archive path stat did not complete"
            ) from error
        if response.status_code == 404:
            return None
        try:
            self._api._raise_for_status(response)
        except APIError as error:
            raise _archive_error(error, "path", "archive path stat failed") from None
        return decode_path_stat(response.headers.get(PATH_STAT_HEADER))

    def _resolve(self, target: ArchiveTarget, path: str) -> tuple[str, tuple[str, ...]]:
        """(deepest existing directory, missing components), in container scope.

        Each existing symlink component is replaced by the daemon's fully
        evaluated target, so tmpfs checks see the path the daemon writes.
        """
        current = "/"
        parts = tuple(part for part in path.split("/") if part)
        for index, part in enumerate(parts):
            candidate = posixpath.join(current, part)
            found = self._head(target.container_id, candidate)
            if found is None:
                return current, parts[index:]
            if stat_kind(found) == "symlink":
                link = found["linkTarget"]
                try:
                    absolute_path(link)
                except ValueError:
                    raise SandboxError(
                        "unsupported", "path", "symlink target is not canonical"
                    ) from None
                candidate = link
                found = self._head(target.container_id, candidate)
                if found is None:
                    dangling = posixpath.join(current, part)
                    raise SandboxError("invalid", "path", f"{dangling} is dangling")
            if stat_kind(found) != "dir":
                raise SandboxError("invalid", "path", f"{candidate} is not a directory")
            current = candidate
        return current, ()

    def path_stat(
        self, target: ArchiveTarget, path: str, *, follow: bool
    ) -> dict[str, Any]:
        _require_path(path, "path")
        if type(follow) is not bool:
            raise SandboxError("invalid", "follow", "expected a boolean")
        target.inspect()
        found = self._head(target.container_id, path)
        if (
            found is not None
            and follow
            and stat_kind(found) == "symlink"
            and found["linkTarget"]
        ):
            # The daemon evaluates every link in scope; the target is final.
            found = self._head(target.container_id, found["linkTarget"])
        return path_stat_view(found)

    def copy_in(
        self, target: ArchiveTarget, dest_dir: str, stage_id: str
    ) -> dict[str, int]:
        """Extract a consumed stage into ``dest_dir``, creating missing
        ancestors (0755 root) first.

        ``dest_dir`` is refused as written (the layer under a tmpfs may hold
        a symlink out of it) and as the daemon resolves it; so is every
        entry path, resolved through symlinks already in the container.
        """
        _require_path(dest_dir, "dest_dir")
        self._require_transferable(target)
        target.refuse_shadowed(dest_dir, "dest_dir")
        existing, missing = self._resolve(target, dest_dir)
        resolved = posixpath.join(existing, *missing)
        target.refuse_shadowed(resolved, "dest_dir")
        with self._stages.consume(stage_id) as (stage, summary):
            self._refuse_shadowed_entries(target, resolved, not missing, stage)
            stage.seek(0)
            if missing:
                self._put(target, existing, directory_tar(missing), "dest_dir")
            self._put(target, resolved, stage, "dest_dir")
        return {"entries": summary.entries, "bytes": summary.bytes}

    def _refuse_shadowed_entries(
        self, target: ArchiveTarget, dest: str, exists: bool, stage: IO[bytes]
    ) -> None:
        """Refuse entries the daemon would write under a tmpfs or /dev.

        The daemon follows symlinks that already exist in the container at
        every intermediate component of an entry (a leaf is replaced, never
        followed). Components are resolved top-down through path-stat, once
        per directory; nothing below a missing one exists yet, so there only
        the literal path matters. A symlink swapped in while the copy runs
        is not seen (the service's own race, as for ``dest_dir``).
        """
        root = _Dir(dest, exists)
        resolved = 0
        with read_tar(stage) as archive:
            for member in archive:
                parts = entry_parts(member.name)
                node = root
                for depth, name in enumerate(parts):
                    if not node.exists or depth == len(parts) - 1:
                        # A shadowed prefix shadows the whole path, so one
                        # literal check covers the rest of the entry.
                        path = posixpath.join(node.path, *parts[depth:])
                        if target.shadowed(path):
                            raise _reaches_tmpfs()
                        break
                    child = node.children.get(name)
                    if child is None:
                        resolved += 1
                        if resolved > MAX_RESOLVED_DIRS:
                            raise SandboxError(
                                "quota",
                                "dest_dir",
                                f"copy_in resolves more than {MAX_RESOLVED_DIRS} "
                                "existing directories",
                            )
                        path = posixpath.join(node.path, name)
                        if target.shadowed(path):
                            raise _reaches_tmpfs()
                        child = self._probe_dir(target, path)
                        node.children[name] = child
                    node = child

    def _probe_dir(self, target: ArchiveTarget, path: str) -> _Dir:
        """``path`` as the daemon resolves it, and whether a directory is there.

        A link into a tmpfs is refused by the next component's check, which
        always follows: only intermediate components are probed.
        """
        found = self._head(target.container_id, path)
        if found is None:
            return _Dir(path, False)
        if stat_kind(found) == "symlink":
            link = found["linkTarget"]
            try:
                absolute_path(link)
            except ValueError:
                raise SandboxError(
                    "unsupported", "path", "symlink target is not canonical"
                ) from None
            path, found = link, self._head(target.container_id, link)
        # Below a file or a dangling link the daemon's own mkdir fails.
        return _Dir(path, found is not None and stat_kind(found) == "dir")

    def _put(self, target: ArchiveTarget, path: str, data: Any, field: str) -> None:
        try:
            self._api.put_archive(target.container_id, path, data)
        except Exception as error:
            raise _archive_error(error, field, "archive upload failed") from error

    def copy_out(
        self,
        target: ArchiveTarget,
        path: str,
        *,
        max_bytes: int,
        exclude: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Re-emit ``path`` as a canonical tar stage rooted at its basename.

        Entries under a tmpfs root below ``path``, excluded entries,
        special files and hardlinks to unknown data are counted in
        ``skipped``; hardlinks to copied files become regular files.
        """
        _require_path(path, "path")
        if path == "/":
            raise SandboxError("invalid", "path", "copy_out needs a named path")
        if type(max_bytes) is not int or max_bytes < 0:
            raise SandboxError("invalid", "max_bytes", "expected a nonnegative limit")
        patterns = _exclusions(exclude)
        self._require_transferable(target)
        target.refuse_shadowed(path, "path")
        parent, name = posixpath.split(path)
        existing, missing = self._resolve(target, parent)
        if missing:
            raise SandboxError("invalid", "path", f"{path} does not exist")
        resolved = posixpath.join(existing, name)
        target.refuse_shadowed(resolved, "path")
        shadows = target.shadow_roots_below(resolved)
        try:
            # Defensive: ask for an identity body; decode_content below would
            # also undo a gzip encoding.
            response = self._api._get(
                self._url(target.container_id),
                params={"path": resolved},
                headers={"Accept-Encoding": "identity"},
                stream=True,
            )
        except Exception as error:
            raise SandboxError(
                "unknown-outcome", "path", "archive download did not start"
            ) from error
        try:
            if response.status_code == 404:
                raise SandboxError("invalid", "path", f"{path} does not exist")
            try:
                self._api._raise_for_status(response)
            except APIError as error:
                raise _archive_error(error, "path", "archive download failed") from None
            decode_path_stat(response.headers.get(PATH_STAT_HEADER))
            response.raw.decode_content = True
            with self._stages.create_result() as result:
                result.summary = self._reemit(
                    response.raw, result.file, max_bytes, patterns, shadows
                )
        finally:
            response.close()
        summary = result.summary
        return {
            "stage_id": result.stage_id,
            "bytes": summary.bytes,
            "entries": summary.entries,
            "skipped": summary.skipped,
        }

    @staticmethod
    def _reemit(
        source: IO[bytes],
        dest: IO[bytes],
        max_bytes: int,
        patterns: Sequence[re.Pattern[str]],
        shadows: Sequence[Sequence[str]],
    ) -> ArchiveSummary:
        writer = CanonicalTarWriter(dest, max_bytes=max_bytes)
        skipped = 0
        dropped: list[tuple[str, ...]] = []
        try:
            with read_tar(source, errors="surrogateescape") as archive:
                for member in archive:
                    try:
                        parts = entry_parts(member.name)
                    except _Rejected:
                        skipped += 1
                        continue
                    relative = parts[1:]
                    if relative and (
                        _under(relative, shadows)
                        or _under(parts, dropped)
                        or _excluded(relative, patterns)
                    ):
                        skipped += 1
                        if member.isdir():
                            dropped.append(parts)
                        continue
                    try:
                        if member.isdir():
                            writer.add_dir(parts, mode=member.mode, mtime=member.mtime)
                        elif member.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                            reader = archive.extractfile(member)
                            assert reader is not None
                            writer.add_file(
                                parts,
                                reader,
                                member.size,
                                mode=member.mode,
                                mtime=member.mtime,
                            )
                        elif member.issym():
                            writer.add_symlink(
                                parts,
                                member.linkname,
                                mode=member.mode,
                                mtime=member.mtime,
                            )
                        elif not (
                            member.islnk()
                            and writer.add_hardlink_copy(
                                parts,
                                entry_parts(member.linkname),
                                mode=member.mode,
                                mtime=member.mtime,
                            )
                        ):
                            skipped += 1
                    except _Rejected:
                        skipped += 1
                        if member.isdir():
                            dropped.append(parts)
        except SandboxError:
            raise
        except (tarfile.TarError, UnicodeError, ValueError, OSError, EOFError) as error:
            raise SandboxError(
                "unknown-outcome", "path", f"archive stream failed: {error}"
            ) from None
        summary = writer.close()
        return ArchiveSummary(summary.entries, summary.bytes, skipped)


def _require_path(value: object, field: str) -> str:
    try:
        return absolute_path(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise SandboxError(
            "invalid", field, "expected a normalized absolute path"
        ) from None


__all__ = [
    "MAX_EXTENDED_HEADER",
    "MAX_STAGE_FRAME",
    "PATH_STAT_HEADER",
    "ArchiveSummary",
    "ArchiveTarget",
    "ArchiveTransfer",
    "BoundedTarFile",
    "BoundedTarInfo",
    "CanonicalTarWriter",
    "StageStore",
    "canonicalize_tar",
    "decode_path_stat",
    "directory_tar",
    "entry_parts",
    "path_stat_view",
    "read_tar",
    "stat_kind",
]
