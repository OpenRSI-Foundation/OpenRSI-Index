"""CPU-only publication checks. Task code and evaluator scripts are never executed."""

from __future__ import annotations

import argparse
import ast
import bz2
import contextlib
import gzip
import hashlib
import io
import json
import lzma
import math
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import tomllib

MAX_STORED = 100 * 1024**2
MAX_MEMBER = 1024**3
MAX_EXPANDED = 8 * 1024**3
MAX_ENTRIES = 100_000
MAX_DEPTH = 4
# Seconds; inside the 20-minute job timeout so a hung scan fails this step.
SCAN_TIMEOUT = 620
CHUNK = 1024**2
# Per member, PAX/GNU long-name records and sparse maps; real ones are a few KiB.
MAX_TAR_HEADER = 1024**2
# Collection-level documentation may sit beside packages; any other file there
# would escape package checks, so it is rejected.
COLLECTIONS = {
    "rsi-tasks",
    "rsi-tasks/signature-tasks",
    "rsi-logs",
    "rsi-logs/signature-tasks",
}
# The Harness runs a task's verifier and reference solution; other task Python
# (for example deliberately broken starter code) only runs inside the task image.
HARNESS_RUN_DIRS = ("tests", "solution")
# Gitleaks' default config exempts paths such as *.bin, *.pdf, images, lockfiles
# and node_modules/ from every rule. Staged copies therefore get neutral names;
# only the suffixes that enable path-specific default rules are kept.
PATH_RULE_SUFFIXES = (
    ".yaml",
    ".yml",
    ".tf",
    ".hcl",
    ".php",
    ".p12",
    ".pfx",
    "nuget.config",
)
# Logs mode reads a separate Logs repository whose root is mounted at this
# Index path; every Logs path p is checked as rsi-logs/<p>.
LOGS_MOUNT = "rsi-logs"
# Root files of the Logs repository that are repository infrastructure, not logs:
# they are secret-scanned only.
LOGS_INFRASTRUCTURE = ("README.md", "LICENSE", ".gitattributes")
# The Logs repository stores raw files (large ones through LFS/Xet), so new or
# modified compressed wrappers and lossless chunk parts are rejected there.
PACKED_SUFFIXES = (
    ".gz",
    ".tgz",
    ".xz",
    ".txz",
    ".bz2",
    ".tbz2",
    ".zst",
    ".zstd",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".lz4",
    ".lzma",
)
# Compression signatures, recognized after renaming. ZIP is matched by suffix
# only: NumPy, PyTorch and office files are ZIP containers, not wrappers.
PACKED_SIGNATURES = (
    b"\x1f\x8b",
    b"\xfd7zXZ\0",
    b"BZh1",
    b"BZh2",
    b"BZh3",
    b"BZh4",
    b"BZh5",
    b"BZh6",
    b"BZh7",
    b"BZh8",
    b"BZh9",
    b"\x28\xb5\x2f\xfd",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!",
)
# A Git LFS pointer is a small text file starting with this line. In a checkout
# whose LFS content was not downloaded, published files are such pointers.
LFS_POINTER = b"version https://git-lfs.github.com/spec/"


class CheckError(ValueError):
    pass


class MemberError(CheckError):
    """A CheckError already labelled with the published file (and member) it concerns."""


def annotate(level: str, message: str):
    # Escape Actions command data. File names are untrusted publication input.
    message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::{level}::{message}")


def safe_relative(name: str) -> Path:
    p = PurePosixPath(name)
    if (
        not name
        or p.is_absolute()
        or ".." in p.parts
        or "\\" in name
        or any(ord(c) < 32 for c in name)
    ):
        raise CheckError("unsafe relative path")
    return Path(*p.parts)


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
    )


def tracked_files(repo: Path, ref: str) -> set[str]:
    return set(
        git(repo, "ls-tree", "-r", "--name-only", "-z", ref).decode().split("\0")
    ) - {""}


def task_root(path: str, roots: set[str]) -> str | None:
    return next(
        (r for r in sorted(roots, key=len, reverse=True) if path.startswith(r + "/")),
        None,
    )


def collection_doc(path: str) -> bool:
    p = PurePosixPath(path)
    return str(p.parent) in COLLECTIONS and p.suffix.lower() == ".md"


def log_root(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[0] != "rsi-logs":
        return None
    n = 3 if parts[1] == "signature-tasks" else 2
    return "/".join(parts[:n]) if len(parts) > n else None


def load_json(path: Path):
    return strict_json(path.read_text(encoding="utf-8"))


def strict_json(text):
    def reject_constant(_):
        raise CheckError("JSON contains a non-finite number")

    def unique_keys(items):
        out = {}
        for key, value in items:
            if key in out:
                raise CheckError("duplicate JSON object key")
            out[key] = value
        return out

    try:

        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise CheckError("non-finite JSON number")
            return number

        return json.loads(
            text,
            parse_constant=reject_constant,
            parse_float=finite_float,
            object_pairs_hook=unique_keys,
        )
    except (ValueError, UnicodeError) as exc:
        # Do not print JSON excerpts: a malformed record could contain a secret.
        raise CheckError("invalid JSON syntax or encoding") from exc


def positive_number(value, label: str):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise CheckError(f"{label} must be a positive finite number")


def check_task(repo: Path, root: str, historical: bool, compile_task: bool):
    task = repo / root
    if not (task / "task.toml").is_file():
        raise CheckError(
            "missing task.toml while other task files remain; restore it or remove the whole task"
        )
    try:
        config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, UnicodeError) as exc:
        # No parser message: it could quote part of a malformed value.
        raise CheckError("task.toml: invalid TOML syntax or encoding") from exc
    if not (task / "instruction.md").is_file():
        raise CheckError("missing instruction.md")
    if not (task / "instruction.md").read_text(encoding="utf-8").strip():
        raise CheckError("empty instruction.md")
    for section in ("agent", "verifier"):
        if "timeout_sec" in config.get(section, {}):
            positive_number(config[section]["timeout_sec"], f"{section}.timeout_sec")
    if "build_timeout_sec" in config.get("environment", {}):
        positive_number(
            config["environment"]["build_timeout_sec"], "environment.build_timeout_sec"
        )
    for section in (
        config.get("environment", {}),
        config.get("metadata", {}).get("rsi_harness", {}).get("verifier", {}),
    ):
        if "gpus" in section and (
            type(section["gpus"]) is not int or section["gpus"] < 0
        ):
            raise CheckError("GPU count must be a non-negative integer")
    for field in ("name", "description"):
        if (
            not isinstance(config.get("task", {}).get(field), str)
            or not config["task"][field].strip()
        ):
            raise CheckError(f"task.{field} must be a non-empty string")
    if historical:
        print(
            f"NOTICE {root}: historical archive; no current-Harness compatibility claim"
        )
        return
    if config.get("schema_version") != "1.4":
        raise CheckError("new runnable tasks must use supported Harbor schema 1.4")
    if not (task / "tests/test.sh").is_file():
        raise CheckError("missing tests/test.sh")
    if not any(
        (task / p).is_file()
        for p in ("environment/Dockerfile", "environment/docker-compose.yaml")
    ) and not config.get("environment", {}).get("docker_image"):
        raise CheckError("missing environment build/image declaration")
    if compile_task:
        # Import the separately installed, BASE-revision Harness. This only parses,
        # hashes and compiles configuration; never builds an image or runs a task.
        from rsi_harness.models import CompileOptions
        from rsi_harness.task.compiler import HarborTaskCompiler

        try:
            HarborTaskCompiler().compile(task, CompileOptions())
        except Exception as exc:
            raise CheckError(
                f"Harness compilation failed ({type(exc).__name__}); inspect task configuration locally"
            ) from exc


def validate_state(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("submissions"), list):
        raise CheckError("evolve_state requires a submissions array")
    seen = {}
    for record in doc["submissions"]:
        if not isinstance(record, dict) or not isinstance(record.get("round"), str):
            raise CheckError("submission requires a round identifier")
        if record["round"] in seen:
            raise CheckError("duplicate submission round")
        score = record.get("score")
        if score is not None and (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
        ):
            raise CheckError("score must be finite numeric or null")
        seen[record["round"]] = score
    best = doc.get("best_score")
    if best is not None:
        if (
            isinstance(best, bool)
            or not isinstance(best, (int, float))
            or not math.isfinite(best)
        ):
            raise CheckError("best_score must be finite numeric or null")
        if doc.get("best_round") not in seen or seen[doc["best_round"]] != best:
            raise CheckError("best_round/best_score contradict recorded submissions")
    # No guessed score direction, required final score, status whitelist or
    # contiguous-round requirement: rejected/incomplete runs and gaps are evidence.


def digest(stream):
    h, size, lines = hashlib.sha256(), 0, 0
    while block := stream.read(CHUNK):
        size += len(block)
        if size > MAX_MEMBER:
            raise CheckError("expanded member exceeds 1 GiB; split the archive")
        h.update(block)
        lines += block.count(b"\n")
    return h.hexdigest(), size, lines


def verify_file(parent: Path, entry: dict, tracked: set[str], repo: Path):
    if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
        raise CheckError("manifest entry requires a path")
    p = parent / safe_relative(entry["path"])
    try:
        verify_entry(p, entry, tracked, repo)
    except CheckError as exc:
        raise CheckError(f"{entry['path']}: {exc}") from exc


def verify_entry(p: Path, entry: dict, tracked: set[str], repo: Path):
    if (
        p.relative_to(repo).as_posix() not in tracked
        or not p.is_file()
        or p.is_symlink()
    ):
        raise CheckError("manifest references missing/untracked/linked file")
    with p.open("rb") as f:
        stored = sha, size, lines = digest(f)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
        or sha != entry["sha256"]
    ):
        raise CheckError("manifest SHA-256 mismatch")
    if entry.get("bytes", size) != size:
        raise CheckError("manifest byte count mismatch")
    if entry.get("compression") == "gzip":
        with gzip.open(p, "rb") as f:
            sha, size, lines = digest(f)
        for field, value in [
            ("uncompressed_sha256", sha),
            ("uncompressed_bytes", size),
            ("line_count", lines),
        ]:
            if field in entry and entry[field] != value:
                raise CheckError(f"manifest {field} mismatch")
    elif "uncompressed_sha256" in entry:
        if (
            entry["uncompressed_sha256"] != sha
            or entry.get("uncompressed_bytes", size) != size
        ):
            raise CheckError("manifest uncompressed hash/size mismatch")
        if "line_count" in entry and entry["line_count"] != lines:
            raise CheckError("manifest line_count mismatch")
    return stored


ARTIFACT_FIELDS = (
    "compressed_size",
    "sanitized_uncompressed_sha256",
    "sanitized_uncompressed_size",
    "sanitized_uncompressed_line_feeds",
)
# The task runner's split_run_log.py writes exactly these keys. Tasks may ship
# their own data manifests, so only this shape counts as a chunk manifest there.
RUNNER_CHUNK_FIELDS = {
    "version",
    "source_name",
    "source_size",
    "source_sha256",
    "chunk_size",
    "parts",
}


def hex_digest(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def count(value) -> bool:
    return type(value) is int and value >= 0


def verify_restored(path: Path, doc: dict, tracked: set[str], repo: Path):
    """Version-2 manifest: one raw file, optionally restored from a compressed or
    chunked publication. Only the raw file is published, so the description of
    the former publication is checked for shape, not against bytes."""
    missing = [f for f in ("artifact", "size", "sha256", "line_feeds") if f not in doc]
    if missing:
        raise CheckError(f"version 2 manifest lacks {', '.join(missing)}")
    artifact = doc["artifact"]
    if not isinstance(artifact, str) or "/" in artifact:
        raise CheckError(
            "version 2 manifest artifact must be a file name in its directory"
        )
    safe_relative(artifact)
    expected = artifact.split(".", 1)[0] + ".manifest.json"
    if path.name != expected:
        raise CheckError(f"version 2 manifest for {artifact} must be named {expected}")
    if not count(doc["size"]) or not count(doc["line_feeds"]):
        raise CheckError("version 2 manifest size and line_feeds must be integers")
    if not hex_digest(doc["sha256"]):
        raise CheckError("version 2 manifest sha256 must be a SHA-256 hex digest")
    try:
        _, _, lines = verify_entry(
            path.parent / safe_relative(artifact),
            {"sha256": doc["sha256"], "bytes": doc["size"]},
            tracked,
            repo,
        )
        if lines != doc["line_feeds"]:
            raise CheckError("manifest line_feeds mismatch")
    except CheckError as exc:
        raise CheckError(f"{artifact}: {exc}") from exc
    source = doc.get("converted_from")
    if source is None:
        return
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("artifact"), str)
        or not isinstance(source.get("compression"), str)
        or not count(source.get("compressed_size"))
        or not hex_digest(source.get("compressed_sha256"))
        or not isinstance(source.get("original_manifest", {}), dict)
    ):
        raise CheckError(
            "converted_from requires artifact, compression, compressed_size and "
            "compressed_sha256; original_manifest must be an object"
        )
    if "parts" in source and (
        not isinstance(source["parts"], list)
        or not source["parts"]
        or not all(
            isinstance(e, dict)
            and isinstance(e.get("path"), str)
            and count(e.get("bytes"))
            and hex_digest(e.get("sha256"))
            for e in source["parts"]
        )
    ):
        raise CheckError("converted_from parts require path, bytes and sha256")


def verify_manifest(path: Path, doc, tracked: set[str], repo: Path):
    if not isinstance(doc, dict):
        raise CheckError("manifest must be an object")
    if doc.get("version") == 2 and type(doc["version"]) is int and "artifact" in doc:
        verify_restored(path, doc, tracked, repo)
        return None
    entries = doc.get("files")
    if "files" in doc:
        if not isinstance(entries, list) or not all(
            isinstance(e, dict) and isinstance(e.get("path"), str) and "sha256" in e
            for e in entries
        ):
            raise CheckError("manifest files require path and sha256 per entry")
        names = [e["path"] for e in entries]
        if len(names) != len(set(names)):
            raise CheckError("duplicate manifest file path")
        for entry in entries:
            verify_file(path.parent, entry, tracked, repo)
    if "artifact" in doc and "compressed_sha256" in doc:
        missing = [field for field in ARTIFACT_FIELDS if field not in doc]
        if missing:
            raise CheckError(f"compressed artifact manifest lacks {', '.join(missing)}")
        verify_file(
            path.parent,
            {
                "path": doc["artifact"],
                "sha256": doc["compressed_sha256"],
                "bytes": doc["compressed_size"],
                "compression": "gzip",
                "uncompressed_sha256": doc["sanitized_uncompressed_sha256"],
                "uncompressed_bytes": doc["sanitized_uncompressed_size"],
                "line_count": doc["sanitized_uncompressed_line_feeds"],
            },
            tracked,
            repo,
        )
    if "parts" in doc:
        if not isinstance(doc["parts"], list) or not doc["parts"]:
            raise CheckError("chunk manifest parts must be a non-empty array")
        if not all(
            isinstance(e, dict) and isinstance(e.get("path", e.get("file")), str)
            for e in doc["parts"]
        ):
            raise CheckError("chunk manifest parts require a path or file per entry")
        # Supported chunk-manifest schema: ordered parts, bytes/hash per part,
        # plus whole-stream bytes/hash. Raw historical formats remain raw records.
        parts = [
            dict(
                e,
                path=e.get("path", e.get("file")),
                bytes=e.get("bytes", e.get("size")),
            )
            for e in doc["parts"]
        ]
        if len({e["path"] for e in parts}) != len(parts):
            raise CheckError("duplicate chunk path")
        h, total = hashlib.sha256(), 0
        for entry in parts:
            verify_file(path.parent, entry, tracked, repo)
            with (path.parent / safe_relative(entry["path"])).open("rb") as f:
                while block := f.read(CHUNK):
                    h.update(block)
                    total += len(block)
                    if total > MAX_EXPANDED:
                        raise CheckError("chunk stream exceeds the scan budget")
        expected_sha, expected_size = (
            doc.get("sha256", doc.get("source_sha256")),
            doc.get("bytes", doc.get("source_size")),
        )
        if h.hexdigest() != expected_sha or total != expected_size:
            raise CheckError("chunk manifest whole-stream checksum/size mismatch")
        return parts
    return None


class TarHeaderBound:
    """Tar stream wrapper that caps header bytes before tarfile reads them.

    tarfile reads a whole PAX or GNU long-name record (and chains of them) into
    memory before it yields the member, so a tiny compressed archive could
    otherwise exhaust memory instead of failing a bound. Member payloads are
    read in bounded blocks and are exempt while ``payload`` is set.
    """

    def __init__(self, stream):
        self.stream, self.header, self.payload = stream, 0, False

    def read(self, size):
        if not self.payload:
            self.header += size
            if self.header > MAX_TAR_HEADER:
                raise CheckError("tar header records exceed 1 MiB for one member")
        return self.stream.read(size)

    def seek(self, *args):
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()


def tar_layer(raw, header: bytes):
    """The decompressed tar stream, so its headers pass through TarHeaderBound."""
    if header.startswith(b"\x1f\x8b"):
        return gzip.GzipFile(fileobj=raw, mode="rb")
    if header.startswith(b"\xfd7zXZ\0"):
        return lzma.LZMAFile(raw)
    if header.startswith(b"BZh"):
        return bz2.BZ2File(raw)
    return raw


class PartStream:
    """Reassemble ordered chunks for scanning, including secrets across boundaries."""

    def __init__(self, paths):
        self.paths, self.current = iter(paths), None

    def read(self, size):
        out = bytearray()
        while len(out) < size:
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    break
            block = self.current.read(size - len(out))
            out.extend(block)
            if not block:
                self.current.close()
                self.current = None
        return bytes(out)

    def close(self):
        if self.current:
            self.current.close()


# What the standard-library readers raise on malformed input; reported by type
# only, since their messages can quote publication content.
ARCHIVE_ERRORS = (
    ValueError,
    OSError,
    EOFError,
    RecursionError,
    NotImplementedError,
    lzma.LZMAError,
    zlib.error,
    tarfile.TarError,
    zipfile.BadZipFile,
)


def staged_name(name: str) -> str:
    base = PurePosixPath(name).name.lower()
    tail = next((s for s in PATH_RULE_SUFFIXES if base.endswith(s)), "")
    return "member" + (tail if tail.startswith(".") else "." + tail if tail else "")


class ScanStage:
    """Bounded archive traversal; no archive paths or executable modes are extracted.

    Each staged copy lives under its own index directory with a neutral name, and
    ``labels`` maps that index back to the published file, written as
    ``file!member`` for archive members, so every warning names a file a reviewer
    can open.
    """

    def __init__(self, destination: Path):
        self.destination, self.total, self.count = destination, 0, 0
        self.labels: dict[str, str] = {}

    def entry(self):
        self.count += 1
        if self.count > MAX_ENTRIES:
            raise CheckError("more than 100000 scan members")

    def charge(self, size: int):
        self.total += size
        if self.total > MAX_EXPANDED:
            raise CheckError(
                "scan expansion budget exceeded; split publication (no files silently skipped)"
            )

    def add(
        self, stream, name: str, depth=0, structured=True, label=None, expand=True
    ):
        label = name if label is None else label
        try:
            self._add(stream, name, depth, structured, label, expand)
        except MemberError:
            raise
        except CheckError as exc:
            raise MemberError(f"{label}: {exc}") from exc
        except ARCHIVE_ERRORS as exc:
            raise MemberError(
                f"{label}: archive/structured-data validation failed ({type(exc).__name__})"
            ) from exc

    def _add(
        self, stream, name: str, depth: int, structured: bool, label: str, expand: bool
    ):
        safe_relative(name)
        if depth > MAX_DEPTH:
            raise CheckError(
                "nested archive exceeds four levels; cannot certify full scan"
            )
        self.entry()
        index = f"{self.count:06d}"
        self.labels[index] = label
        target = self.destination / index / staged_name(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with target.open("wb") as out:
            while block := stream.read(CHUNK):
                size += len(block)
                if size > MAX_MEMBER:
                    raise CheckError(
                        "scan expansion budget exceeded; split publication (no files silently skipped)"
                    )
                self.charge(len(block))
                out.write(block)
        if not expand:
            # Only checksum-verified chunks may be copied opaquely. Their
            # complete reassembled stream is expanded and audited below.
            return
        lower = name.lower()
        with target.open("rb") as f:
            header = f.read(512)
        if (
            lower.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz"))
            or header[257:262] == b"ustar"
        ):
            with target.open("rb") as raw, tar_layer(raw, header) as layer:
                bounded = TarHeaderBound(layer)
                with tarfile.open(fileobj=bounded, mode="r:") as archive:
                    # First pass reads headers only (payloads are skipped), so a
                    # source link can be checked against every regular file.
                    members, position, declared = [], 0, 0
                    for member in archive:
                        bounded.header = 0
                        # Header blocks (including PAX/long-name records) and
                        # directory entries are expanded and held in memory too, so
                        # they count toward the bounds even though nothing is staged.
                        payload = member.size if member.isfile() else 0
                        self.charge(max(archive.offset - position - payload, 0))
                        position = archive.offset
                        safe_relative(member.name)
                        if self.count + len(members) >= MAX_ENTRIES:
                            raise CheckError("more than 100000 archive entries")
                        if not (
                            member.isfile()
                            or member.isdir()
                            or (not structured and member.issym())
                        ):
                            raise CheckError(
                                f"archive contains a link/device/non-regular entry ({member.name})"
                            )
                        if member.size > MAX_MEMBER:
                            raise CheckError(
                                f"archive member exceeds 1 GiB ({member.name})"
                            )
                        # Fail before decompressing payloads that cannot fit.
                        declared += payload
                        if self.total + declared > MAX_EXPANDED:
                            raise CheckError(
                                "scan expansion budget exceeded; split publication (no files silently skipped)"
                            )
                        members.append(member)
                    # Compare normalised paths: `tar -C dir .` names members
                    # "./x", while the computed link target drops the "./".
                    regular_names = {
                        PurePosixPath(m.name).as_posix() for m in members if m.isfile()
                    }
                    bounded.payload = True
                    for member in members:
                        if member.isdir():
                            self.entry()
                            continue
                        if member.issym():
                            # Upstream task sources may alias CLAUDE.md to AGENTS.md.
                            # Never create/follow links. Permit only a safe relative
                            # declaration whose target is a regular file in this tar;
                            # all regular target bytes are independently scanned.
                            relative = safe_relative(member.linkname)
                            link_target = (
                                PurePosixPath(member.name).parent / relative.as_posix()
                            ).as_posix()
                            if link_target not in regular_names:
                                raise CheckError(
                                    "source archive link must name a regular file "
                                    f"in the same archive ({member.name})"
                                )
                            self.add(
                                io.BytesIO(member.linkname.encode()),
                                member.name + ".symlink-target.txt",
                                depth + 1,
                                False,
                                f"{label}!{member.name}",
                            )
                            continue
                        with archive.extractfile(member) as child:
                            self.add(
                                child,
                                member.name,
                                depth + 1,
                                structured,
                                f"{label}!{member.name}",
                            )
        elif lower.endswith(".zip") or header.startswith(
            (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
        ):
            # ZipFile parses the whole central directory into memory first, so
            # the declared entry count is checked before it is built.
            with target.open("rb") as raw:
                end = zipfile._EndRecData(raw)
            if end and self.count + end[zipfile._ECD_ENTRIES_TOTAL] > MAX_ENTRIES:
                raise CheckError("more than 100000 scan members")
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    safe_relative(member.filename)
                    if member.is_dir():
                        self.entry()
                        continue
                    if (
                        member.flag_bits & 1
                        or (member.external_attr >> 16) & 0o170000 == 0o120000
                    ):
                        raise CheckError(
                            f"encrypted or linked ZIP entry cannot be audited ({member.filename})"
                        )
                    if member.file_size > MAX_MEMBER:
                        raise CheckError(
                            f"ZIP member exceeds 1 GiB ({member.filename})"
                        )
                    with archive.open(member) as child:
                        self.add(
                            child,
                            member.filename,
                            depth + 1,
                            structured,
                            f"{label}!{member.filename}",
                        )
        elif lower.endswith((".gz", ".xz", ".bz2")) or header.startswith(
            (b"\x1f\x8b", b"\xfd7zXZ\0", b"BZh")
        ):
            if lower.endswith(".gz") or header.startswith(b"\x1f\x8b"):
                opener, suffix = gzip.open, ".gz"
            elif lower.endswith(".xz") or header.startswith(b"\xfd7zXZ\0"):
                opener, suffix = lzma.open, ".xz"
            else:
                opener, suffix = bz2.open, ".bz2"
            child_name = (
                name[: -len(suffix)] if lower.endswith(suffix) else name + ".expanded"
            )
            with opener(target, "rb") as child:
                self.add(
                    child,
                    child_name,
                    depth + 1,
                    structured,
                    f"{label}!{PurePosixPath(child_name).name}",
                )
        elif lower.endswith((".7z", ".rar", ".zst", ".zstd")) or header.startswith(
            (b"7z\xbc\xaf\x27\x1c", b"Rar!", b"\x28\xb5\x2f\xfd")
        ):
            raise CheckError(
                "unsupported compressed archive; use gzip/tar/zip/xz/bzip2 for an auditable publication"
            )
        # Validate decompressed structured records too, not just their wrapper.
        elif structured and lower.endswith(".json"):
            load_json(target)
        elif structured and lower.endswith(".jsonl"):
            with target.open(encoding="utf-8") as f:
                for number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        strict_json(line)
                    except (ValueError, UnicodeError) as exc:
                        raise CheckError(f"invalid JSONL at line {number}") from exc

    def scan(self, binary: Path, config: Path):
        report = self.destination.parent / "gitleaks.json"
        # A stale report must never stand in for this run's result.
        report.unlink(missing_ok=True)
        # The PR cannot suppress secrets with gitleaks:allow or .gitleaksignore.
        result = subprocess.run(
            [
                str(binary.resolve()),
                "dir",
                str(self.destination),
                "--config",
                str(config.resolve()),
                "--gitleaks-ignore-path",
                str(self.destination.parent / "no-ignore"),
                "--ignore-gitleaks-allow",
                "--max-decode-depth",
                "5",
                "--max-target-megabytes",
                "0",
                "--redact=100",
                "--report-format",
                "json",
                "--report-path",
                str(report),
                "--no-banner",
                "--log-level",
                "error",
            ],
            capture_output=True,
            # An external timeout must fail the check, never leave a partially
            # scanned report looking successful after an internal cancellation.
            timeout=SCAN_TIMEOUT,
            check=False,
        )
        if result.returncode not in (0, 1) or not report.is_file():
            raise CheckError(
                "secret scanner failed; raw diagnostics withheld to protect log contents"
            )
        findings = load_json(report)
        if findings:
            warnings = set()
            for item in findings:
                index = Path(item["File"]).relative_to(self.destination).parts[0]
                warnings.add((self.labels[index], item["RuleID"], item["StartLine"]))
            # Paths/rule IDs/line numbers only. Never print or upload matched text.
            # Every distinct warning is printed: an advisory scan only helps if a
            # reviewer can see and locate all of its findings.
            for label, rule, line in sorted(warnings):
                annotate(
                    "warning", f"Possible credential: {label}; rule={rule}, line={line}"
                )
            print(
                f"NOTICE {len(warnings)} possible credentials (advisory only); "
                "review and redact confirmed credentials before publication"
            )
            return len(warnings)
        if result.returncode != 0:
            raise CheckError("secret scanner exited unsuccessfully")
        return 0


def runner_chunk_manifest(path: Path):
    """The document if it is exactly the runner's chunk format, else None."""
    try:
        doc = load_json(path)
    except CheckError:
        # Task data is not held to strict JSON; such a file is not a chunk manifest.
        return None
    return doc if isinstance(doc, dict) and set(doc) == RUNNER_CHUNK_FIELDS else None


def check_python(name: str, path: Path, root: str):
    try:
        # Bytes honour PEP 263 coding declarations; nothing is imported or run.
        ast.parse(path.read_bytes(), filename=name)
    # CPython's parser raises RecursionError (or MemoryError) on deeply nested
    # or very long expressions; treat them as unparsable, not as checker crashes.
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        where = PurePosixPath(name).relative_to(root).parts[0]
        if where in HARNESS_RUN_DIRS:
            raise CheckError(
                f"{name}: invalid Python syntax (not executed); the Harness runs {where}/"
            ) from exc
        # Starter code may be deliberately broken or target the task image's
        # Python rather than this job's, so it informs reviewers only.
        annotate(
            "warning",
            f"{name}: Python does not parse under Python "
            f"{sys.version_info.major}.{sys.version_info.minor}; "
            "not blocking because the Harness does not run it",
        )


def packed_reason(name: str, path: Path) -> str | None:
    """Why a Logs-repository file is not a raw file, or None."""
    p = PurePosixPath(name)
    if any(part.endswith(".parts") for part in p.parts[:-1]):
        return "lossless chunk part"
    lower = p.name.lower()
    if lower.endswith(PACKED_SUFFIXES) or ".tar." in lower:
        return "compressed or archived file"
    with path.open("rb") as f:
        header = f.read(512)
    if header.startswith(PACKED_SIGNATURES) or header[257:262] == b"ustar":
        return "compressed or archived file (by content)"
    return None


def lfs_pointer(path: Path) -> bool:
    if path.stat().st_size >= 1024:
        return False
    with path.open("rb") as f:
        return f.read(len(LFS_POINTER)) == LFS_POINTER


class Revisions:
    """The checked revisions and paths of one run.

    In Logs mode ``prefix`` is ``rsi-logs/``: every Logs-repository path is
    handled under that name, so log rules, messages and labels are the same as
    for the Index tree. ``local`` maps such a name back to the checkout.
    """

    def __init__(self, repo: Path, base_ref, head_ref: str, full: bool, mount=None):
        self.repo = repo.resolve()
        self.prefix = mount + "/" if mount else ""
        current = git(self.repo, "rev-parse", "HEAD").decode().strip()
        try:
            head = (
                git(self.repo, "rev-parse", "--verify", head_ref + "^{commit}")
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError as exc:
            raise CheckError("head revision not found") from exc
        if current != head:
            raise CheckError("checkout must match the checked PR head")
        if base_ref is None:
            if not full:
                raise CheckError("--base-ref is required unless --full is given")
            base = head
        else:
            try:
                base = git(self.repo, "merge-base", base_ref, head).decode().strip()
            except subprocess.CalledProcessError as exc:
                raise CheckError(
                    "base revision is missing or shares no history with the head; "
                    "audit the head with --full"
                ) from exc
        self.head, self.base = head, base
        # Repository paths as stored; manifests are verified against these.
        self.stored = tracked_files(self.repo, head)
        self.files = {self.prefix + p for p in self.stored}
        self.old_files = {self.prefix + p for p in tracked_files(self.repo, base)}
        self.changed = (
            self.files | self.old_files
            if full
            else {
                self.prefix + p
                for p in git(
                    self.repo, "diff", "--name-only", "--no-renames", "-z", base, head
                )
                .decode()
                .split("\0")
                if p
            }
        )

    def local(self, name: str) -> Path:
        return self.repo / name[len(self.prefix) :]


def lfs_includes(revisions: Revisions) -> list[str]:
    """Git LFS include patterns for what an incremental Logs-mode check reads."""
    patterns = set()
    for p in revisions.changed:
        # A deletion still makes the rest of its log root part of the check.
        if root := log_root(p):
            patterns.add(root[len(revisions.prefix) :] + "/**")
        elif p in revisions.files:
            patterns.add(p[len(revisions.prefix) :])
    return sorted(patterns)


def check(
    repo: Path,
    base_ref: str | None,
    head_ref: str,
    policy: Path,
    compile_task=False,
    gitleaks: Path | None = None,
    secret_config: Path | None = None,
    full=False,
    mount: str | None = None,
    task_repo: Path | None = None,
    task_ref: str | None = None,
):
    logs_mode = mount is not None
    if logs_mode and (mount != LOGS_MOUNT or task_repo is None or task_ref is None):
        raise CheckError(
            f"Logs mode needs --mount {LOGS_MOUNT}, --task-repo and --task-ref"
        )
    if not logs_mode and (task_repo is not None or task_ref is not None):
        raise CheckError("--task-repo and --task-ref are only used with --mount")
    revisions = Revisions(repo, base_ref, head_ref, full, mount)
    repo, files, old_files, changed = (
        revisions.repo,
        revisions.files,
        revisions.old_files,
        revisions.changed,
    )
    config = tomllib.loads(policy.read_text())
    if logs_mode:
        # Tasks come from the task repository (OpenRSI-Index main in CI), never
        # from the Logs repository under review.
        try:
            task_files = set(
                git(
                    task_repo.resolve(),
                    "ls-tree",
                    "-r",
                    "--name-only",
                    "-z",
                    task_ref,
                    "--",
                    "rsi-tasks",
                )
                .decode()
                .split("\0")
            ) - {""}
        except subprocess.CalledProcessError as exc:
            raise CheckError("task repository revision not found") from exc
        infrastructure = {revisions.prefix + n for n in LOGS_INFRASTRUCTURE}
        print(
            f"NOTICE Logs mode: Logs repository paths are shown as {LOGS_MOUNT}/<path>; "
            "tasks are read from the task repository"
        )
    else:
        task_files, infrastructure = files | old_files, set()
    roots = {
        str(PurePosixPath(p).parent)
        for p in task_files
        if p.startswith("rsi-tasks/") and p.endswith("/task.toml")
    }
    tasks = {r for p in changed if (r := task_root(p, roots))}
    logs = {r for p in changed if (r := log_root(p))}
    docs = set()
    for p in sorted(changed):
        safe_relative(p)
        # A file outside any package escapes every package check, so only
        # collection documentation may sit there; it is still secret-scanned.
        # Removing such a file needs no content check.
        if (
            p not in files
            or not p.startswith(("rsi-tasks/", "rsi-logs/"))
            or task_root(p, roots)
            or log_root(p)
        ):
            continue
        if collection_doc(p) or p in infrastructure:
            docs.add(p)
        elif p.startswith("rsi-tasks/"):
            raise CheckError(f"{p}: task files have no enclosing task.toml")
        elif logs_mode:
            raise CheckError(
                f"{p}: log files must be inside <task>/ or signature-tasks/<task>/ "
                "of the Logs repository; only .md documentation, README.md, LICENSE "
                "and .gitattributes may sit outside them"
            )
        else:
            raise CheckError(
                f"{p}: log files must be inside rsi-logs/<task>/ or "
                "rsi-logs/signature-tasks/<task>/; only .md documentation may sit beside them"
            )
    selected = {p for p in files if any(p.startswith(r + "/") for r in tasks | logs)}
    published = selected | docs
    for p in sorted(published):
        f = revisions.local(p)
        if f.is_symlink() or not f.is_file():
            raise CheckError(f"{p}: publication must use regular files, not symlinks")
        if logs_mode:
            # A skipped LFS download must never pass as a clean (tiny) file.
            if lfs_pointer(f):
                raise CheckError(
                    f"{p}: Git LFS pointer, not content; download LFS objects "
                    "(git lfs pull) before checking"
                )
            if p in changed and (reason := packed_reason(p, f)):
                raise CheckError(
                    f"{p}: {reason}; the Logs repository stores raw files only "
                    "(large files go through LFS/Xet), so upload the uncompressed, "
                    "unsplit file instead"
                )
        elif f.stat().st_size >= MAX_STORED:
            raise CheckError(
                f"{p}: stored file must be below GitHub 100 MiB limit; use lossless chunks"
            )
    for root in sorted(tasks):
        if not any(p.startswith(root + "/") for p in files):
            print(f"PASS {root}: task removed")
            continue
        historical = (
            root.removeprefix("rsi-tasks/") in config.get("historical_tasks", {})
            and root + "/task.toml" in old_files
        )
        try:
            check_task(repo, root, historical, compile_task)
        except CheckError as exc:
            raise CheckError(f"{root}: {exc}") from exc
        except (OSError, UnicodeError) as exc:
            raise CheckError(
                f"{root}: unreadable task file ({type(exc).__name__})"
            ) from exc
        print(
            f"PASS {root}: task configuration"
            + (
                " and CPU Harness compilation"
                if compile_task and not historical
                else ""
            )
        )
    for root in sorted(logs):
        if not any(p.startswith(root + "/") for p in files):
            print(f"PASS {root}: log archive removed")
            continue
        slug = root.removeprefix("rsi-logs/")
        matched = (
            "rsi-tasks/"
            + config.get("log_task_aliases", {}).get(slug, slug)
            + "/task.toml"
        )
        if matched not in task_files:
            if logs_mode:
                raise CheckError(
                    f"{root}: corresponding public task.toml not found in the task "
                    "repository; merge the task into OpenRSI-Index first (or declare "
                    "a reviewed alias)"
                )
            raise CheckError(
                f"{root}: corresponding public task.toml not found (declare a reviewed alias if needed)"
            )
        new_root = not any(p.startswith(root + "/") for p in old_files)
        if new_root and root + "/README.md" not in files:
            raise CheckError(
                f"{root}: new log archive needs README.md describing source and recorded protocol"
            )
    chunk_manifests, chunk_parts = [], set()
    for name in sorted(selected):
        p = revisions.local(name)
        try:
            # Validate new/modified structured records; existing historical
            # contents are not retroactively coerced to a different data schema.
            if name in changed and p.suffix == ".json":
                doc = load_json(p)
                if p.name in ("evolve_state.json", "e_state.json"):
                    validate_state(doc)
            # Log archives carry publication/chunk manifests. Tasks may ship
            # their own data manifests (verifier assets) with a different schema,
            # so in a task only a runner-format chunk manifest is verified.
            log = name.startswith("rsi-logs/")
            if (
                log
                and p.suffix == ".json"
                and (p.name.endswith("manifest.json") or p.name == "publication.json")
            ):
                doc = load_json(p)
            elif not log and p.name == "manifest.json":
                doc = runner_chunk_manifest(p)
            else:
                doc = None
            parts = (
                None if doc is None else verify_manifest(p, doc, revisions.stored, repo)
            )
            if parts is not None:
                if logs_mode and name in changed:
                    raise CheckError(
                        "chunk manifest; the Logs repository stores raw files only, "
                        "so upload the reassembled file instead"
                    )
                source = doc.get("source_name", "reassembled-transcript.txt")
                if not isinstance(source, str):
                    raise CheckError("chunk manifest source_name must be a string")
                safe_relative(source)
                paths = [p.parent / safe_relative(e["path"]) for e in parts]
                chunk_manifests.append((name, source, paths, log))
                chunk_parts.update(
                    revisions.prefix + x.relative_to(repo).as_posix() for x in paths
                )
        except CheckError as exc:
            raise CheckError(f"{name}: {exc}") from exc
        except (OSError, UnicodeError) as exc:
            raise CheckError(f"{name}: unreadable file ({type(exc).__name__})") from exc
        if name in changed and name.startswith("rsi-tasks/") and p.suffix == ".py":
            check_python(name, p, task_root(name, roots))
    if published:
        with tempfile.TemporaryDirectory(prefix="rsi-publication-") as temporary:
            stage = ScanStage(Path(temporary) / "scan")
            for name in sorted(published):
                with revisions.local(name).open("rb") as f:
                    stage.add(
                        f,
                        name,
                        structured=name.startswith("rsi-logs/"),
                        # A lossless chunk of a compressed or tar stream is
                        # truncated on its own: verified parts are scanned as raw
                        # bytes only, and the reassembled stream is expanded below.
                        expand=name not in chunk_parts,
                    )
            for name, source, paths, log in chunk_manifests:
                with contextlib.closing(PartStream(paths)) as stream:
                    stage.add(stream, source, structured=log, label=f"{name}!{source}")
            if gitleaks:
                if secret_config is None:
                    raise CheckError(
                        "secret scanner requires an explicit trusted policy"
                    )
                stage.scan(gitleaks, secret_config)
            else:
                print(
                    "NOTICE secret scanning not requested locally; mandatory in PR CI"
                )
            print(
                f"PASS {len(published)} published files; {stage.count} scan members; {stage.total} expanded bytes"
            )
    else:
        print("PASS no task/log changes; CPU publication check not applicable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--base-ref", help="Required unless --full; the diff starts at the merge base"
    )
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).with_name("publication-policy.toml"),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gitleaks", type=Path)
    parser.add_argument("--secret-config", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Audit every published package, not just changes (used by manual workflow runs)",
    )
    logs = parser.add_argument_group(
        "Logs mode",
        "Check a separate Logs repository checkout (--repo) whose root is mounted "
        "at rsi-logs/; tasks are read from --task-ref of --task-repo",
    )
    logs.add_argument("--mount", choices=[LOGS_MOUNT])
    logs.add_argument("--task-repo", type=Path)
    logs.add_argument("--task-ref")
    logs.add_argument(
        "--lfs-include",
        action="store_true",
        help="Print the Git LFS include patterns an incremental check reads, one "
        "per line, and exit",
    )
    args = parser.parse_args()
    try:
        if args.lfs_include:
            if args.mount is None:
                raise CheckError("--lfs-include is only used with --mount")
            for pattern in lfs_includes(
                Revisions(args.repo, args.base_ref, args.head_ref, args.full, args.mount)
            ):
                print(pattern)
            return 0
        check(
            args.repo,
            args.base_ref,
            args.head_ref,
            args.policy,
            args.compile,
            args.gitleaks,
            args.secret_config,
            args.full,
            args.mount,
            args.task_repo,
            args.task_ref,
        )
    except Exception as exc:  # noqa: BLE001 -- never leak parser/tool excerpts at CLI boundary
        # No exception contents beyond our controlled errors: TOML/JSON or tool
        # parsers can include credentials from malformed input in diagnostics.
        print(
            "FAIL " + (str(exc) if isinstance(exc, CheckError) else type(exc).__name__)
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
