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
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import tomllib

MAX_STORED = 100 * 1024**2
MAX_MEMBER = 1024**3
MAX_EXPANDED = 8 * 1024**3
MAX_ENTRIES = 100_000
MAX_DEPTH = 4
CHUNK = 1024**2


class CheckError(ValueError):
    pass


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


def log_root(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[0] != "rsi-logs":
        return None
    n = 3 if parts[1] == "signature-tasks" else 2
    return "/".join(parts[:n]) if len(parts) > n else None


def load_json(path: Path):
    return strict_json(path.read_text())


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
    config = tomllib.loads((task / "task.toml").read_text())
    for required in ("instruction.md", "task.toml"):
        if not (task / required).is_file():
            raise CheckError(f"missing {required}")
    if not (task / "instruction.md").read_text().strip():
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
    if (
        p.relative_to(repo).as_posix() not in tracked
        or not p.is_file()
        or p.is_symlink()
    ):
        raise CheckError("manifest references missing/untracked/linked file")
    with p.open("rb") as f:
        sha, size, lines = digest(f)
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


def verify_manifest(path: Path, doc, tracked: set[str], repo: Path):
    if not isinstance(doc, dict):
        raise CheckError("manifest must be an object")
    entries = doc.get("files")
    if "files" in doc:
        if not isinstance(entries, list) or not all(
            isinstance(e, dict) and "path" in e and "sha256" in e for e in entries
        ):
            raise CheckError("manifest files require path and sha256 per entry")
        names = [e["path"] for e in entries]
        if len(names) != len(set(names)):
            raise CheckError("duplicate manifest file path")
        for entry in entries:
            verify_file(path.parent, entry, tracked, repo)
    if "artifact" in doc and "compressed_sha256" in doc:
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


class ScanStage:
    """Bounded archive traversal; no archive paths or executable modes are extracted."""

    def __init__(self, destination: Path):
        self.destination, self.total, self.count = destination, 0, 0

    def add(self, stream, name: str, depth=0, structured=True):
        safe_relative(name)
        if depth > MAX_DEPTH:
            raise CheckError(
                "nested archive exceeds four levels; cannot certify full scan"
            )
        self.count += 1
        if self.count > MAX_ENTRIES:
            raise CheckError("more than 100000 scan members")
        target = self.destination / f"{self.count:06d}" / safe_relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with target.open("wb") as out:
            while block := stream.read(CHUNK):
                size += len(block)
                self.total += len(block)
                if size > MAX_MEMBER or self.total > MAX_EXPANDED:
                    raise CheckError(
                        "scan expansion budget exceeded; split publication (no files silently skipped)"
                    )
                out.write(block)
        lower = name.lower()
        with target.open("rb") as f:
            header = f.read(512)
        if (
            lower.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz"))
            or header[257:262] == b"ustar"
        ):
            with tarfile.open(target, "r:*") as archive:
                members = []
                for member in archive:
                    safe_relative(member.name)
                    if self.count + len(members) >= MAX_ENTRIES:
                        raise CheckError("more than 100000 archive entries")
                    members.append(member)
                regular_names = {m.name for m in members if m.isfile()}
                for member in members:
                    if member.isdir():
                        continue
                    if not structured and member.issym():
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
                                "source archive link must name a regular file in the same archive"
                            )
                        self.add(
                            io.BytesIO(member.linkname.encode()),
                            member.name + ".symlink-target.txt",
                            depth + 1,
                            structured=False,
                        )
                        continue
                    if not member.isfile():
                        raise CheckError(
                            "archive contains a link/device/non-regular entry"
                        )
                    if member.size > MAX_MEMBER:
                        raise CheckError("archive member exceeds 1 GiB")
                    with archive.extractfile(member) as child:
                        self.add(child, member.name, depth + 1, structured)
        elif lower.endswith(".zip") or header.startswith(
            (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
        ):
            with zipfile.ZipFile(target) as archive:
                for member in archive.infolist():
                    safe_relative(member.filename)
                    if member.is_dir():
                        continue
                    if (
                        member.flag_bits & 1
                        or (member.external_attr >> 16) & 0o170000 == 0o120000
                    ):
                        raise CheckError(
                            "encrypted or linked ZIP entry cannot be audited"
                        )
                    if member.file_size > MAX_MEMBER:
                        raise CheckError("ZIP member exceeds 1 GiB")
                    with archive.open(member) as child:
                        self.add(child, member.filename, depth + 1, structured)
        elif lower.endswith(".gz") or header.startswith(b"\x1f\x8b"):
            with gzip.open(target, "rb") as child:
                self.add(
                    child,
                    name[:-3] if lower.endswith(".gz") else name + ".expanded",
                    depth + 1,
                    structured,
                )
        elif lower.endswith(".xz") or header.startswith(b"\xfd7zXZ\0"):
            with lzma.open(target, "rb") as child:
                self.add(
                    child,
                    name[:-3] if lower.endswith(".xz") else name + ".expanded",
                    depth + 1,
                    structured,
                )
        elif lower.endswith(".bz2") or header.startswith(b"BZh"):
            with bz2.open(target, "rb") as child:
                self.add(
                    child,
                    name[:-4] if lower.endswith(".bz2") else name + ".expanded",
                    depth + 1,
                    structured,
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
            with target.open() as f:
                for number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        strict_json(line)
                    except (ValueError, UnicodeError) as exc:
                        raise CheckError(f"invalid JSONL at line {number}") from exc

    def scan(self, binary: Path, config: Path):
        report = self.destination.parent / "gitleaks.json"
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
            timeout=620,
            check=False,
        )
        if result.returncode not in (0, 1) or not report.is_file():
            raise CheckError(
                "secret scanner failed; raw diagnostics withheld to protect log contents"
            )
        findings = load_json(report)
        if findings:
            # Paths/rule IDs/line numbers only. Never print or upload matched text.
            for item in findings[:30]:
                rel = Path(item["File"]).relative_to(self.destination).as_posix()
                message = (
                    f"Possible credential: {rel}; "
                    f"rule={item['RuleID']}, line={item['StartLine']}"
                )
                # Escape Actions command data, even though matched content is
                # never included. File names are untrusted publication input.
                message = (
                    message.replace("%", "%25")
                    .replace("\r", "%0D")
                    .replace("\n", "%0A")
                )
                print("::warning::" + message)
            print(
                f"NOTICE {len(findings)} possible credentials (advisory only); "
                "review and redact confirmed credentials before publication"
            )
            return len(findings)
        if result.returncode != 0:
            raise CheckError("secret scanner exited unsuccessfully")
        return 0


def check(
    repo: Path,
    base_ref: str,
    head_ref: str,
    policy: Path,
    compile_task=False,
    gitleaks: Path | None = None,
    secret_config: Path | None = None,
    full=False,
):
    repo = repo.resolve()
    current = git(repo, "rev-parse", "HEAD").decode().strip()
    head = git(repo, "rev-parse", "--verify", head_ref + "^{commit}").decode().strip()
    if current != head:
        raise CheckError("checkout must match the checked PR head")
    base = git(repo, "merge-base", base_ref, head).decode().strip()
    files, old_files = tracked_files(repo, head), tracked_files(repo, base)
    changed = (
        files | old_files
        if full
        else set(
            git(repo, "diff", "--name-only", "--no-renames", "-z", base, head)
            .decode()
            .split("\0")
        )
        - {""}
    )
    config = tomllib.loads(policy.read_text())
    roots = {
        str(PurePosixPath(p).parent)
        for p in files | old_files
        if p.startswith("rsi-tasks/") and p.endswith("/task.toml")
    }
    tasks = {r for p in changed if (r := task_root(p, roots))}
    logs = {r for p in changed if (r := log_root(p))}
    for p in changed:
        safe_relative(p)
        if (
            p.startswith("rsi-tasks/")
            and task_root(p, roots) is None
            and not p.endswith("/README.md")
            and p != "rsi-tasks/README.md"
        ):
            raise CheckError(f"{p}: task files have no enclosing task.toml")
    selected = {p for p in files if any(p.startswith(r + "/") for r in tasks | logs)}
    for p in sorted(selected):
        f = repo / p
        if f.is_symlink() or not f.is_file():
            raise CheckError(f"{p}: publication must use regular files, not symlinks")
        if f.stat().st_size >= MAX_STORED:
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
        check_task(repo, root, historical, compile_task)
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
        if matched not in files:
            raise CheckError(
                f"{root}: corresponding public task.toml not found (declare a reviewed alias if needed)"
            )
        new_root = not any(p.startswith(root + "/") for p in old_files)
        if new_root and root + "/README.md" not in files:
            raise CheckError(
                f"{root}: new log archive needs README.md describing source and recorded protocol"
            )
    chunk_manifests = []
    for name in sorted(selected):
        p = repo / name
        # Validate new/modified structured records; existing historical contents
        # are not retroactively coerced to a different data schema.
        if name in changed and p.suffix == ".json":
            doc = load_json(p)
            if p.name in ("evolve_state.json", "e_state.json"):
                validate_state(doc)
        if p.suffix == ".json" and (
            p.name.endswith("manifest.json") or p.name == "publication.json"
        ):
            doc = load_json(p)
            parts = verify_manifest(p, doc, files, repo)
            if parts is not None:
                chunk_manifests.append((p, doc, parts))
        if name in changed and name.startswith("rsi-tasks/") and p.suffix == ".py":
            try:
                ast.parse(p.read_text())
            except (SyntaxError, UnicodeError) as exc:
                raise CheckError(
                    f"{name}: invalid Python syntax (not executed)"
                ) from exc
    if selected:
        with tempfile.TemporaryDirectory(prefix="rsi-publication-") as temporary:
            stage = ScanStage(Path(temporary) / "scan")
            for name in sorted(selected):
                try:
                    with (repo / name).open("rb") as f:
                        stage.add(f, name, structured=name.startswith("rsi-logs/"))
                except (
                    ValueError,
                    OSError,
                    EOFError,
                    tarfile.TarError,
                    zipfile.BadZipFile,
                ) as exc:
                    raise CheckError(
                        f"{name}: archive/structured-data validation failed ({type(exc).__name__})"
                    ) from exc
            for p, doc, parts in chunk_manifests:
                paths = [p.parent / safe_relative(e["path"]) for e in parts]
                with contextlib.closing(PartStream(paths)) as stream:
                    stage.add(
                        stream, doc.get("source_name", "reassembled-transcript.txt")
                    )
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
                f"PASS {len(selected)} published files; {stage.count} scan members; {stage.total} expanded bytes"
            )
    else:
        print("PASS no task/log changes; CPU publication check not applicable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-ref", required=True)
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
        help="Explicit local audit of all published packages",
    )
    args = parser.parse_args()
    try:
        check(
            args.repo,
            args.base_ref,
            args.head_ref,
            args.policy,
            args.compile,
            args.gitleaks,
            args.secret_config,
            args.full,
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
