"""Streaming sanitizer for a built image's docker-archive export (spec 4 B8).

BuildKit's ``--output type=docker,dest=-`` writes an OCI layout plus the
legacy ``manifest.json`` that ``POST /images/load`` reads: ``blobs/``,
``blobs/sha256/<64hex>`` (config, manifest and layers), ``index.json``,
``manifest.json`` and ``oci-layout`` (VERIFIED on BuildKit v0.27.1 with
Docker 29.2.1). The builder runs the caller's build, so its output is not
trusted: only those members pass, every blob must hash to its name, the
manifest must name exactly one image whose config and layers are present,
``RepoTags`` is forced to null (a tag would retag host images such as
``ubuntu:24.04``) and index ref-name annotations are dropped.

The sanitizer re-emits every member itself (normalized headers, content
streamed in bounded chunks) and holds back only the tar's end-of-archive
blocks: all checks, and the ``on_config`` callback that journals the image
ID, run before the daemon can see the end of the stream, so an image the
daemon registers is always one the journal already names.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import IO, Any

from rsi_harness.runtime.sandbox_archive import read_tar

MIB = 1024**2
# index.json, manifest.json and oci-layout are small; bound them in memory.
MAX_JSON_BYTES = MIB
MAX_BLOBS = 1024
CHUNK = MIB
_BLOB = re.compile(r"^blobs/sha256/([0-9a-f]{64})$")
_DIRS = ("blobs", "blobs/sha256")
_JSON = ("index.json", "manifest.json", "oci-layout")
_REF_ANNOTATIONS = ("io.containerd.image.name", "org.opencontainers.image.ref.name")
_MANIFEST_KEYS = frozenset({"Config", "RepoTags", "Layers"})
_EOF = b"\0" * (2 * tarfile.BLOCKSIZE)


class ImageArchiveError(Exception):
    """``kind`` is ``quota`` (the export exceeds its byte cap) or ``invalid``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True, slots=True)
class SanitizedImage:
    image_id: str
    layers: tuple[str, ...]
    bytes: int


def _invalid(message: str) -> ImageArchiveError:
    return ImageArchiveError("invalid", message)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(name: str, data: bytes) -> Any:
    try:
        return json.loads(data, object_pairs_hook=_unique_pairs)
    except (ValueError, RecursionError, UnicodeError):
        raise _invalid(f"{name} is not valid JSON") from None


def _blob(path: object, field: str) -> str:
    match = _BLOB.fullmatch(path) if isinstance(path, str) else None
    if match is None:
        raise _invalid(f"manifest {field} must name a blobs/sha256 member")
    return match.group(1)


def _manifest(data: bytes) -> tuple[bytes, str, tuple[str, ...]]:
    value = _json("manifest.json", data)
    if not isinstance(value, list) or len(value) != 1:
        raise _invalid("manifest.json must describe exactly one image")
    entry = value[0]
    if not isinstance(entry, dict) or set(entry) - _MANIFEST_KEYS:
        raise _invalid("manifest.json holds unsupported fields")
    config = _blob(entry.get("Config"), "Config")
    layers = entry.get("Layers")
    if not isinstance(layers, list) or len(layers) > MAX_BLOBS:
        raise _invalid("manifest.json Layers must be a bounded list")
    digests = tuple(_blob(layer, "Layers") for layer in layers)
    rewritten = [
        {
            "Config": entry["Config"],
            # Never a tag: the broker tags by ID once the load is verified.
            "RepoTags": None,
            "Layers": list(layers),
        }
    ]
    return json.dumps(rewritten, separators=(",", ":")).encode(), config, digests


def _index(data: bytes) -> bytes:
    value = _json("index.json", data)
    if not isinstance(value, dict) or not isinstance(value.get("manifests"), list):
        raise _invalid("index.json must be an OCI image index")
    for descriptor in value["manifests"]:
        if not isinstance(descriptor, dict):
            raise _invalid("index.json manifests must be descriptors")
        annotations = descriptor.get("annotations")
        if annotations is None:
            continue
        if not isinstance(annotations, dict):
            raise _invalid("index.json annotations must be an object")
        for key in _REF_ANNOTATIONS:
            annotations.pop(key, None)
    return json.dumps(value, separators=(",", ":")).encode()


def _header(name: str, kind: bytes, size: int, mode: int) -> bytes:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = size
    info.mode = mode
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info.tobuf(tarfile.USTAR_FORMAT, "utf-8", "strict")


def _padding(size: int) -> bytes:
    return b"\0" * (-size % tarfile.BLOCKSIZE)


class ImageArchiveSanitizer:
    """Iterate to read ``source`` and yield the sanitized archive's bytes.

    ``max_bytes`` caps the member content read from ``source``; the export
    is refused as soon as a member header announces more, before its data
    is read. ``on_config`` receives ``sha256:<config digest>`` (the ID the
    daemon's load will report) after every check passed and before the
    end-of-archive blocks are yielded; if it raises, those blocks never
    are. ``result`` is set once the whole archive was yielded.
    """

    def __init__(
        self,
        source: IO[bytes],
        *,
        max_bytes: int,
        on_config: Callable[[str], None],
    ) -> None:
        self._source = source
        self._max_bytes = max_bytes
        self._on_config = on_config
        self.result: SanitizedImage | None = None

    def __iter__(self) -> Iterator[bytes]:
        seen: set[str] = set()
        blobs: set[str] = set()
        manifest: tuple[str, tuple[str, ...]] | None = None
        total = 0
        try:
            archive = read_tar(self._source)
        except (tarfile.TarError, OSError, EOFError, UnicodeError) as error:
            raise _invalid(f"the export is not a tar stream: {error}") from None
        with archive:
            try:
                for member in archive:
                    name = member.name.rstrip("/")
                    if name in seen:
                        raise _invalid(f"duplicate archive member {name}")
                    seen.add(name)
                    if name in _DIRS:
                        if not member.isdir():
                            raise _invalid(f"{name} must be a directory")
                        yield _header(name + "/", tarfile.DIRTYPE, 0, 0o755)
                        continue
                    blob = _BLOB.fullmatch(name)
                    if (blob is None and name not in _JSON) or member.type not in (
                        tarfile.REGTYPE,
                        tarfile.AREGTYPE,
                    ):
                        raise _invalid(f"unexpected archive member {name[:256]}")
                    if total + member.size > self._max_bytes:
                        raise ImageArchiveError(
                            "quota",
                            f"the image export exceeds {self._max_bytes} bytes",
                        )
                    total += member.size
                    reader = archive.extractfile(member)
                    if reader is None:
                        raise _invalid(f"archive member {name} has no data")
                    if blob is not None:
                        if len(blobs) >= MAX_BLOBS:
                            raise _invalid("the export holds too many blobs")
                        yield from self._blob(name, blob.group(1), reader, member.size)
                        blobs.add(blob.group(1))
                        continue
                    if member.size > MAX_JSON_BYTES:
                        raise _invalid(f"{name} exceeds {MAX_JSON_BYTES} bytes")
                    data = reader.read(member.size)
                    if len(data) != member.size:
                        raise _invalid(f"{name} is truncated")
                    if name == "manifest.json":
                        data, config, layers = _manifest(data)
                        manifest = (config, layers)
                    elif name == "index.json":
                        data = _index(data)
                    elif not isinstance(_json(name, data), dict):
                        raise _invalid("oci-layout must be a JSON object")
                    yield _header(name, tarfile.REGTYPE, len(data), 0o644)
                    yield data + _padding(len(data))
            except ImageArchiveError:
                raise
            except (tarfile.TarError, OSError, EOFError, UnicodeError) as error:
                raise _invalid(f"invalid export tar stream: {error}") from None
        if manifest is None:
            raise _invalid("the export has no manifest.json")
        config, layers = manifest
        missing = [digest for digest in (config, *layers) if digest not in blobs]
        if missing:
            raise _invalid(f"the export lacks blob sha256:{missing[0]}")
        image_id = "sha256:" + config
        # Journal first; only then may the daemon see the end of the stream.
        self._on_config(image_id)
        self.result = SanitizedImage(
            image_id, tuple("sha256:" + layer for layer in layers), total
        )
        yield _EOF

    @staticmethod
    def _blob(name: str, digest: str, reader: IO[bytes], size: int) -> Iterator[bytes]:
        yield _header(name, tarfile.REGTYPE, size, 0o444)
        hasher = hashlib.sha256()
        remaining = size
        while remaining:
            chunk = reader.read(min(CHUNK, remaining))
            if not chunk:
                raise _invalid(f"{name} is truncated")
            hasher.update(chunk)
            remaining -= len(chunk)
            yield chunk
        if hasher.hexdigest() != digest:
            raise _invalid(f"{name} does not match its digest")
        yield _padding(size)


__all__ = [
    "ImageArchiveError",
    "ImageArchiveSanitizer",
    "SanitizedImage",
]
