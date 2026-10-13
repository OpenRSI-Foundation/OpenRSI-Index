"""Publish research logs to the OpenRSI-Index Logs dataset on the Hugging Face Hub.

    publish_logs.py SOURCE --task TASK [--dest PATH] [--create-pr] [--dry-run]
    publish_logs.py --check-pr N

SOURCE is a local run directory or log tree; it is never modified. Its files are
copied into a staging directory at ``--dest`` (default: the task's log root) and
normalized the same way the history was converted for the Hub:

- lossless chunk directories (a ``manifest.json`` listing ordered ``parts``, in
  the task runner's ``file``/``size`` + ``source_*`` format or the
  ``path``/``bytes`` + ``sha256``/``bytes`` format) are reassembled after every
  part and the whole stream are verified;
- ``<x>.gz`` files (including reassembled ones) are decompressed to ``<x>``,
  checked against any original manifest;
- each converted file gets a version-2 manifest
  ``<name-before-first-dot>.manifest.json`` that records the raw file and what
  it was converted from (the original manifest is kept inside it), and listing
  manifests (``publication.json``, ``*manifest.json``) list the raw file in
  place of what it replaced, followed by its version-2 manifest, byte for byte
  as the history conversion wrote them (only the manifests' ``note`` differs);
- any other compressed or archived file is refused: decompress it yourself.

The staged tree is then committed to a temporary Git repository and checked with
the publication checker in Logs mode (``--full`` over the staged files, tasks
from the local OpenRSI-Index checkout). A new log root also needs a README.md,
either staged or already on the Hub. Only if everything passes is the staged
tree uploaded with ``huggingface_hub`` (``upload_folder``) to ``main`` or, with
``--create-pr``, as a Hub pull request. The Hub token comes from the standard
Hugging Face login or the ``HF_TOKEN`` environment variable and is never printed.

``--check-pr N`` fetches ``refs/pr/N`` of the Logs repository anonymously,
downloads the LFS content of the affected log roots and runs the same checker
incrementally against the Hub's ``main``: the check maintainers use before
merging a Hub pull request.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path, PurePosixPath

import tomllib

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "check_publication", HERE / "check_publication.py"
)
pub = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pub)

# The Logs dataset repository on the Hugging Face Hub. Workflows read it from
# here (--print-repo); change it only here.
LOGS_REPO = "OpenRSI-Foundation/OpenRSI-Index-Logs"
# Uploads are made with exactly this huggingface_hub release.
HUB_VERSION = "2.0.0"
HUB_URL = "https://huggingface.co"
# The note written by the one-time conversion of OpenRSI-Index's rsi-logs/.
HISTORY_NOTE = (
    "Restored from the gzip publication in OpenRSI-Index; "
    "the Hub stores large files natively."
)
# The top-level "compression" description a listing manifest gets when it named
# a gzip or chunked file that was converted (the history conversion's wording).
RAW_COMPRESSION = (
    "Complete sanitized transcripts stored as raw files; the gzip and chunked-gzip "
    "copies first published in OpenRSI-Index were restored byte-exactly and each "
    "conversion is recorded in an adjacent .manifest.json; no truncation"
)
CHUNK = 1024**2
REPO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
SLUG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)*")
GZIP_MAGIC = b"\x1f\x8b"
# Git for the temporary repositories: no user/system configuration (hooks, LFS
# filters, signing), so a contributor's setup cannot change what is checked.
GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


class PublishError(Exception):
    pass


def digest(path: Path):
    """SHA-256, size and line-feed count of a file."""
    h, size, lines = hashlib.sha256(), 0, 0
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
            size += len(block)
            lines += block.count(b"\n")
    return h.hexdigest(), size, lines


def dump_json(doc, ascii_only=False) -> bytes:
    """The history conversion's serialization: two-space indent, final newline."""
    return (json.dumps(doc, indent=2, ensure_ascii=ascii_only) + "\n").encode("utf-8")


def json_style(data: bytes, doc):
    """The ensure_ascii setting that re-serializes ``data`` byte for byte, or
    None when neither does (the file is then written in the default style)."""
    for ascii_only in (False, True):
        if dump_json(doc, ascii_only) == data:
            return ascii_only
    return None


def run_git(repo: Path, *args: str, capture=False):
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        env={**os.environ, **GIT_ENV},
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PublishError(f"git {args[0]} failed: {result.stderr.strip()[-500:]}")
    return result.stdout.strip() if capture else None


# Staging and normalization


def copy_source(source: Path, target: Path):
    if not source.is_dir():
        raise PublishError(f"{source}: SOURCE must be a directory")
    if target.exists():
        raise PublishError(f"{target}: staging target already exists")
    count = 0
    for path in sorted(source.rglob("*")):
        rel = path.relative_to(source)
        if ".git" in rel.parts:
            raise PublishError(
                f"{rel.as_posix()}: Git metadata is not published; pass the log files only"
            )
        if path.is_symlink():
            raise PublishError(
                f"{rel.as_posix()}: symbolic link; replace it with the file it names"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise PublishError(f"{rel.as_posix()}: not a regular file")
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out)
        count += 1
    if not count:
        raise PublishError(f"{source}: no files to publish")


def load_doc(path: Path, rel: str):
    try:
        return pub.load_json(path)
    except (pub.CheckError, OSError, UnicodeError) as exc:
        raise PublishError(f"{rel}: invalid JSON") from exc


def chunk_parts(doc: dict, rel: str):
    parts = []
    for entry in doc["parts"]:
        if not isinstance(entry, dict):
            raise PublishError(f"{rel}: chunk parts must be objects")
        path = entry.get("path", entry.get("file"))
        size = entry.get("bytes", entry.get("size"))
        sha = entry.get("sha256")
        if (
            not isinstance(path, str)
            or "/" in path
            or path in ("", ".", "..")
            or type(size) is not int
            or not pub.hex_digest(sha)
        ):
            raise PublishError(
                f"{rel}: each chunk part needs a file name, byte count and SHA-256"
            )
        parts.append({"path": path, "bytes": size, "sha256": sha})
    if not parts or len({p["path"] for p in parts}) != len(parts):
        raise PublishError(f"{rel}: chunk parts must be a non-empty list of distinct files")
    return parts


class Normalizer:
    """Converts a staged tree in place. Paths are relative to the tree root,
    which mirrors the dataset layout."""

    def __init__(self, tree: Path, note: str | None = None):
        self.tree, self.note = tree, note
        # Removed file -> (sha256, size) and -> the raw file replacing it.
        self.removed: dict[str, tuple[str, int]] = {}
        self.replaced_by: dict[str, str] = {}
        # Raw file -> its digest; and the v2 manifests written.
        self.raw: dict[str, tuple[str, int, int]] = {}
        self.written: set[str] = set()
        self.manifest_of: dict[str, str] = {}
        self.modified: set[str] = set()
        self.converted: list[str] = []

    def rel(self, path: Path) -> str:
        return path.relative_to(self.tree).as_posix()

    def run(self):
        chunks = self.reassemble()
        for path in sorted(p for p in self.tree.rglob("*") if p.is_file()):
            with path.open("rb") as f:
                gz = f.read(2) == GZIP_MAGIC
            if path.name.lower().endswith(".gz"):
                self.gunzip(path, chunks.pop(self.rel(path), None))
            elif gz:
                raise PublishError(
                    f"{self.rel(path)}: gzip content without a .gz name; rename it "
                    "to <name>.gz or decompress it yourself"
                )
        # Chunked files that were not compressed.
        for rel, chunk in sorted(chunks.items()):
            sha, size, lines = self.raw[rel] = digest(self.tree / rel)
            self.restored(self.tree / rel, chunk["artifact"], "none", sha, size, chunk)
        for path in sorted(p for p in self.tree.rglob("*") if p.is_file()):
            rel = self.rel(path)
            if reason := pub.packed_reason("rsi-logs/" + rel, path):
                raise PublishError(
                    f"{rel}: {reason} in a format this tool does not convert; the Logs "
                    "repository stores raw files only, so decompress or unpack it "
                    "yourself and publish the raw files"
                )
        self.update_listings()
        return self.converted

    def remember_removed(self, path: Path, raw: Path):
        sha, size, _ = digest(path)
        self.removed[self.rel(path)] = (sha, size)
        self.replaced_by[self.rel(path)] = self.rel(raw)

    def reassemble(self):
        """Reassemble chunk directories; returns their records by output file."""
        chunks = {}
        manifests = {}
        for path in sorted(self.tree.rglob("*manifest.json")):
            rel = self.rel(path)
            if not path.is_file():
                continue
            doc = load_doc(path, rel)
            if isinstance(doc, dict) and isinstance(doc.get("parts"), list):
                manifests[path.parent] = (path, doc)
        for directory in sorted(self.tree.rglob("*.parts")):
            if directory.is_dir() and directory not in manifests:
                raise PublishError(
                    f"{self.rel(directory)}: chunk directory without a chunk manifest"
                )
        for directory, (path, doc) in sorted(manifests.items()):
            rel = self.rel(path)
            if path.name != "manifest.json":
                raise PublishError(
                    f"{rel}: chunk manifests must be named manifest.json inside the chunk directory"
                )
            parts = chunk_parts(doc, rel)
            whole_sha = doc.get("sha256", doc.get("source_sha256"))
            whole_size = doc.get("bytes", doc.get("source_size"))
            if not pub.hex_digest(whole_sha) or type(whole_size) is not int:
                raise PublishError(f"{rel}: chunk manifest lacks the whole-stream SHA-256/size")
            name = doc.get("source_name") or directory.name.removesuffix(".parts")
            if (
                not isinstance(name, str)
                or "/" in name
                or name in ("", ".", "..")
                or name == directory.name
            ):
                raise PublishError(f"{rel}: chunk source_name must be a file name")
            out = directory.parent / name
            if out.exists():
                raise PublishError(
                    f"{self.rel(out)}: exists beside its chunk directory; keep one of them"
                )
            listed = {p["path"] for p in parts} | {"manifest.json", ".gitattributes"}
            for extra in sorted(directory.rglob("*")):
                if extra.is_dir() or extra.relative_to(directory).as_posix() in listed:
                    continue
                raise PublishError(
                    f"{self.rel(extra)}: not listed by the chunk manifest {rel}"
                )
            h, total = hashlib.sha256(), 0
            with out.open("wb") as joined:
                for part in parts:
                    part_path = directory / part["path"]
                    if not part_path.is_file():
                        raise PublishError(f"{rel}: missing part {part['path']}")
                    sha, size, _ = digest(part_path)
                    if (sha, size) != (part["sha256"], part["bytes"]):
                        raise PublishError(
                            f"{rel}: part {part['path']} does not match its SHA-256/size"
                        )
                    with part_path.open("rb") as f:
                        while block := f.read(CHUNK):
                            h.update(block)
                            total += len(block)
                            joined.write(block)
            if (h.hexdigest(), total) != (whole_sha, whole_size):
                raise PublishError(
                    f"{rel}: reassembled stream does not match the manifest SHA-256/size"
                )
            for item in sorted(directory.rglob("*")):
                if item.is_file():
                    self.remember_removed(item, out)
            shutil.rmtree(directory)
            chunks[self.rel(out)] = {
                "artifact": directory.name,
                "parts": parts,
                "manifest": doc,
                "manifest_rel": rel,
                "sha256": whole_sha,
                "bytes": whole_size,
            }
        return chunks

    def gunzip(self, path: Path, chunk):
        rel = self.rel(path)
        raw = path.with_name(path.name[: -len(".gz")])
        if raw.exists():
            raise PublishError(f"{rel}: {self.rel(raw)} also exists; keep one of them")
        try:
            with gzip.open(path, "rb") as source, raw.open("wb") as out:
                shutil.copyfileobj(source, out, CHUNK)
        except (OSError, EOFError, zlib.error) as exc:
            raw.unlink(missing_ok=True)
            raise PublishError(f"{rel}: not a valid gzip file ({type(exc).__name__})") from exc
        csha, csize, _ = digest(path)
        sha, size, lines = self.raw[self.rel(raw)] = digest(raw)
        if chunk is None:
            self.remember_removed(path, raw)
        else:
            # The gzip itself was produced by reassembly; its parts are the
            # removed files, already mapped to the reassembled name.
            for old, new in list(self.replaced_by.items()):
                if new == rel:
                    self.replaced_by[old] = self.rel(raw)
            for field, value in (
                ("uncompressed_sha256", sha),
                ("uncompressed_bytes", size),
                ("line_count", lines),
            ):
                if field in chunk["manifest"] and chunk["manifest"][field] != value:
                    raise PublishError(
                        f"{chunk['manifest_rel']}: {field} does not match the decompressed file"
                    )
        path.unlink()
        self.restored(raw, chunk["artifact"] if chunk else path.name, "gzip", csha, csize, chunk)

    def restored(self, raw: Path, old_name, compression, csha, csize, chunk):
        sha, size, lines = self.raw[self.rel(raw)]
        stem = raw.name.split(".", 1)[0]
        if not stem:
            raise PublishError(f"{self.rel(raw)}: a converted file name cannot start with '.'")
        target = raw.with_name(stem + ".manifest.json")
        target_rel = self.rel(target)
        original = chunk["manifest"] if chunk else None
        if target_rel in self.written:
            raise PublishError(
                f"{target_rel}: two converted files in one directory share the name "
                f"{stem}; rename one of them"
            )
        if target.exists():
            doc = load_doc(target, target_rel)
            gz_name = old_name if compression == "gzip" and not chunk else raw.name + ".gz"
            if not (isinstance(doc, dict) and doc.get("artifact") in (gz_name, old_name)):
                raise PublishError(
                    f"{target_rel}: exists and does not describe {old_name}; the "
                    f"version 2 manifest for {raw.name} must be written there"
                )
            for field, value in (
                ("compressed_sha256", csha),
                ("compressed_size", csize),
                ("sanitized_uncompressed_sha256", sha),
                ("sanitized_uncompressed_size", size),
                ("sanitized_uncompressed_line_feeds", lines),
            ):
                if field in doc and doc[field] != value:
                    raise PublishError(f"{target_rel}: {field} does not match the restored file")
            if original is None:
                original = doc
            # Listing manifests that name the replaced manifest are updated too.
            self.modified.add(target_rel)
        source = {
            "artifact": old_name,
            "compression": compression,
            "compressed_size": csize,
            "compressed_sha256": csha,
        }
        if chunk:
            source["parts"] = chunk["parts"]
        if original is not None:
            source["original_manifest"] = original
        label = "chunked" if compression == "none" else compression
        target.write_bytes(
            dump_json(
                {
                    "version": 2,
                    "artifact": raw.name,
                    "size": size,
                    "sha256": sha,
                    "line_feeds": lines,
                    "converted_from": source,
                    "note": self.note
                    or f"Restored from a {label} upload; the Hub stores large files natively.",
                }
            )
        )
        self.written.add(target_rel)
        self.manifest_of[self.rel(raw)] = target_rel
        self.converted.append(f"{old_name} -> {self.rel(raw)} ({size} bytes)")

    def listing(self, path: Path):
        """(doc, ensure_ascii) of a listing manifest, or None."""
        if not (path.name == "publication.json" or path.name.endswith("manifest.json")):
            return None
        if self.rel(path) in self.written:
            return None
        doc = load_doc(path, self.rel(path))
        if not isinstance(doc, dict) or not isinstance(doc.get("files"), list):
            return None
        return doc, json_style(path.read_bytes(), doc)

    def update_listings(self):
        """Point listing manifests at raw files; repeat while a rewritten
        manifest is itself listed by another manifest."""
        for _ in range(10):
            changed = False
            for path in sorted(self.tree.rglob("*.json")):
                loaded = self.listing(path)
                if loaded is None:
                    continue
                doc, ascii_only = loaded
                files, converted = self.update_files(path, doc["files"])
                if files is None:
                    continue
                doc = dict(doc, files=files)
                described = doc.get("compression")
                if converted and isinstance(described, str) and "gzip" in described:
                    doc["compression"] = RAW_COMPRESSION
                # The original style where it re-serializes byte for byte.
                path.write_bytes(dump_json(doc, bool(ascii_only)))
                self.modified.add(self.rel(path))
                changed = True
            if not changed:
                return
        raise PublishError("listing manifests reference each other in a cycle")

    def target(self, manifest: Path, entry):
        """The tree path a listing entry names, or None."""
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return None
        try:
            target = (manifest.parent / pub.safe_relative(entry["path"])).resolve()
            return target.relative_to(self.tree.resolve()).as_posix()
        except (pub.CheckError, ValueError):
            return None

    def update_files(self, manifest: Path, files: list):
        """A listing's new ``files`` (None if unchanged) and whether it named a
        converted file. Exactly as the history conversion: a converted file is
        listed once (a chunk directory's entries collapse into its first), by
        the first entry's fields with the raw file's values, followed by its
        version 2 manifest ({path, bytes, sha256}); an older manifest it
        replaces is dropped where it was listed. When sibling entries record
        uncompressed digests, the raw file's are recorded too."""
        targets = [self.target(manifest, entry) for entry in files]
        converted = {self.replaced_by[t] for t in targets if t in self.replaced_by}
        relisted = {self.manifest_of[raw] for raw in converted}
        backfill = any(isinstance(e, dict) and "uncompressed_sha256" in e for e in files)
        entries, seen, done, dirty = [], set(), set(), False

        def add(entry):
            key = entry.get("path") if isinstance(entry, dict) else None
            if key is not None and key in seen:
                return True
            seen.add(key)
            entries.append(entry)
            return False

        for entry, rel in zip(files, targets):
            if rel in relisted:
                dirty = True
                continue
            if rel in self.replaced_by:
                dirty = True
                raw = self.check_replaced(manifest, entry, rel)
                if raw in done:
                    continue
                done.add(raw)
                add(self.raw_entry(manifest, entry, raw, backfill))
                listed = self.tree / self.manifest_of[raw]
                sha, size, _ = digest(listed)
                add({"path": self.relative(manifest, listed), "bytes": size, "sha256": sha})
                continue
            new = self.refreshed(entry, rel)
            dirty |= new is not entry
            dirty |= add(new)
        return (entries if dirty else None), bool(converted)

    def relative(self, manifest: Path, path: Path) -> str:
        return PurePosixPath(os.path.relpath(path, manifest.parent)).as_posix()

    def check_replaced(self, manifest: Path, entry: dict, rel: str) -> str:
        """Check a listed removed file against its entry; returns its raw file."""
        where = f"{self.rel(manifest)}: {entry['path']}"
        old_sha, old_size = self.removed[rel]
        if entry.get("sha256", old_sha) != old_sha or entry.get("bytes", old_size) != old_size:
            raise PublishError(f"{where}: listed SHA-256/size do not match the file")
        raw = self.replaced_by[rel]
        sha, size, lines = self.raw[raw]
        if entry.get("compression") == "gzip":
            for field, value in (
                ("uncompressed_sha256", sha),
                ("uncompressed_bytes", size),
                ("line_count", lines),
            ):
                if field in entry and entry[field] != value:
                    raise PublishError(f"{where}: listed {field} does not match the restored file")
        return raw

    def raw_entry(self, manifest: Path, entry: dict, raw: str, backfill: bool) -> dict:
        sha, size, lines = self.raw[raw]
        values = {
            "path": self.relative(manifest, self.tree / raw),
            "bytes": size,
            "sha256": sha,
            "uncompressed_bytes": size,
            "uncompressed_sha256": sha,
            "line_count": lines,
            "compression": None,
        }
        new = {key: values.get(key, value) for key, value in entry.items()}
        if backfill and "uncompressed_sha256" not in new:
            new.update(
                uncompressed_bytes=size, uncompressed_sha256=sha, line_count=lines, compression=None
            )
        return new

    def refreshed(self, entry, rel):
        """An entry for a rewritten file, with its new digest."""
        if rel not in self.modified or "sha256" not in entry:
            return entry
        sha, size, lines = digest(self.tree / rel)
        if entry["sha256"] == sha and entry.get("bytes", size) == size:
            return entry
        new = dict(entry, sha256=sha)
        if "bytes" in new:
            new["bytes"] = size
        for field, value in (
            ("uncompressed_sha256", sha),
            ("uncompressed_bytes", size),
            ("line_count", lines),
        ):
            if field in new:
                new[field] = value
        return new


def normalize(tree: Path, note: str | None = None) -> list[str]:
    """Normalize a staged tree in place; returns a description per conversion."""
    return Normalizer(tree, note).run()


# Destination and checks


def log_roots_for(task: str, policy: dict) -> list[str]:
    aliases = [
        log for log, target in policy.get("log_task_aliases", {}).items() if target == task
    ]
    return aliases or [task]


def resolve_dest(task: str, dest: str | None, policy: dict) -> tuple[str, str]:
    """(dest, log root) for a task, validating an explicit --dest."""
    if not SLUG.fullmatch(task):
        raise PublishError("--task must be a task path under rsi-tasks/, e.g. demo-task")
    roots = log_roots_for(task, policy)
    allowed = set(roots) | {task}
    if dest is None:
        return roots[0], roots[0]
    dest = dest.strip("/")
    if not SLUG.fullmatch(dest):
        raise PublishError("--dest must be a relative dataset path")
    root = pub.log_root(f"rsi-logs/{dest}/file")
    root = root.removeprefix("rsi-logs/") if root else None
    if root not in allowed:
        raise PublishError(
            f"--dest {dest} is outside the task's log root ({' or '.join(sorted(allowed))})"
        )
    return dest, root


def hub_api():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise PublishError(
            f"install the Hugging Face client: pip install huggingface_hub=={HUB_VERSION}"
        ) from exc
    if huggingface_hub.__version__ != HUB_VERSION:
        raise PublishError(
            f"huggingface_hub {huggingface_hub.__version__} is installed; this tool uses "
            f"{HUB_VERSION}: pip install huggingface_hub=={HUB_VERSION}"
        )
    return huggingface_hub.HfApi()


def remote_has_file(repo: str, path: str) -> bool:
    api = hub_api()
    try:
        return api.file_exists(repo, path, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001 -- report the type; never a response body
        raise PublishError(
            f"could not ask the Hub whether {path} exists ({type(exc).__name__})"
        ) from exc


def commit_tree(tree: Path):
    run_git(tree, "init", "-q")
    run_git(tree, "add", "-A")
    run_git(
        tree,
        "-c",
        "user.name=publish_logs",
        "-c",
        "user.email=publish-logs@localhost",
        "commit",
        "-qm",
        "staged logs",
    )


def check_staged(tree: Path, index: Path, task_ref: str, gitleaks: Path | None):
    commit_tree(tree)
    pub.check(
        tree,
        None,
        "HEAD",
        HERE / "publication-policy.toml",
        gitleaks=gitleaks,
        secret_config=HERE / "publication-gitleaks.toml" if gitleaks else None,
        full=True,
        mount=pub.LOGS_MOUNT,
        task_repo=index,
        task_ref=task_ref,
    )


def upload(tree: Path, repo: str, create_pr: bool, message: str):
    info = hub_api().upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=tree,
        path_in_repo="",
        commit_message=message,
        create_pr=create_pr,
        ignore_patterns=[".git", ".git/*"],
    )
    return getattr(info, "pr_url", None) or getattr(info, "commit_url", None) or str(info)


def publish(args) -> int:
    policy = tomllib.loads((HERE / "publication-policy.toml").read_text())
    dest, root = resolve_dest(args.task, args.dest, policy)
    with Staging(args.staging) as staging:
        tree = staging / "tree"
        copy_source(args.source.resolve(), tree / dest)
        print(f"Staged {args.source} at {dest}/ in {tree}")
        for line in normalize(tree, args.note):
            print(f"CONVERTED {line}")
        if not (tree / root / "README.md").is_file() and not remote_has_file(
            args.repo, f"{root}/README.md"
        ):
            raise PublishError(
                f"{root}: new log archive needs README.md describing source and "
                "recorded protocol; add it to SOURCE"
            )
        check_staged(tree, args.index.resolve(), args.task_ref, args.gitleaks)
        if args.dry_run:
            print("DRY RUN: checks passed; nothing uploaded")
            return 0
        message = args.message or f"Publish logs for {args.task} ({dest})"
        url = upload(tree, args.repo, args.create_pr, message)
        print(f"UPLOADED {url}")
    return 0


class Staging:
    """A kept --staging directory, or a temporary one."""

    def __init__(self, path: Path | None):
        self.path, self.temporary = path, None

    def __enter__(self) -> Path:
        if self.path is None:
            self.temporary = tempfile.TemporaryDirectory(prefix="publish-logs-")
            return Path(self.temporary.name)
        if self.path.exists() and any(self.path.iterdir()):
            raise PublishError(f"{self.path}: --staging must be empty or absent")
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path.resolve()

    def __exit__(self, *exc):
        if self.temporary:
            self.temporary.cleanup()
        return False


def hub_git_url(repo: str) -> str:
    return f"{HUB_URL}/datasets/{repo}"


# Attributes a submitted tree may not apply to the checkout: conversions would
# make the checked bytes differ from the blobs the Hub serves (and a re-encoded
# file could hide a credential from the scanner). LFS (filter=lfs) still applies.
NEUTRAL_ATTRIBUTES = "* -text -eol -ident -working-tree-encoding\n"


def neutralize_attributes(logs: Path, repo: str):
    """Check out the published blobs byte for byte, with LFS objects from the Hub."""
    run_git(logs, "config", "core.autocrlf", "false")
    info = logs / ".git" / "info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "attributes").write_text(NEUTRAL_ATTRIBUTES)
    # Local configuration outranks a submitted .lfsconfig.
    run_git(logs, "config", "lfs.url", hub_git_url(repo) + ".git/info/lfs")


def check_pr(args) -> int:
    with Staging(args.staging) as staging:
        logs = staging / "logs"
        logs.mkdir()
        run_git(logs, "init", "-q")
        # Anonymous read: no credential helper; LFS content is fetched below.
        run_git(logs, "config", "credential.helper", "")
        run_git(logs, "lfs", "install", "--local", "--skip-smudge")
        neutralize_attributes(logs, args.repo)
        run_git(logs, "remote", "add", "origin", hub_git_url(args.repo))
        run_git(
            logs,
            "fetch",
            "--no-tags",
            "--quiet",
            "origin",
            "+refs/heads/main:refs/remotes/origin/main",
            f"+refs/pr/{args.check_pr}:refs/remotes/origin/pr",
        )
        run_git(logs, "checkout", "-q", "--detach", "refs/remotes/origin/pr")
        try:
            base = run_git(logs, "merge-base", "refs/remotes/origin/main", "HEAD", capture=True)
        except PublishError:
            base = None
        head = run_git(logs, "rev-parse", "HEAD", capture=True)
        print(
            f"Checking {args.repo} refs/pr/{args.check_pr} at {head} "
            + (f"against main (merge base {base})" if base else "in full (no merge base with main)")
        )
        revisions = pub.Revisions(logs, base, "HEAD", base is None, pub.LOGS_MOUNT)
        if base is None:
            run_git(logs, "lfs", "pull")
        for pattern in [] if base is None else pub.lfs_includes(revisions):
            run_git(logs, "lfs", "pull", f"--include={pattern}", "--exclude=")
        pub.check(
            logs,
            base,
            "HEAD",
            HERE / "publication-policy.toml",
            gitleaks=args.gitleaks,
            secret_config=HERE / "publication-gitleaks.toml" if args.gitleaks else None,
            full=base is None,
            mount=pub.LOGS_MOUNT,
            task_repo=args.index.resolve(),
            task_ref=args.task_ref,
        )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", nargs="?", type=Path, help="Run directory or log tree")
    parser.add_argument("--task", help="Task path under rsi-tasks/, e.g. signature-tasks/x")
    parser.add_argument(
        "--dest", help="Path inside the dataset (default: the task's log root)"
    )
    parser.add_argument("--create-pr", action="store_true", help="Upload as a Hub pull request")
    parser.add_argument("--dry-run", action="store_true", help="Stage and check; do not upload")
    parser.add_argument("--repo", default=LOGS_REPO, help="Logs dataset repository id")
    parser.add_argument("--check-pr", type=int, metavar="N", help="Check Hub pull request N")
    parser.add_argument(
        "--index",
        type=Path,
        default=HERE.parent,
        help="OpenRSI-Index checkout whose tasks the logs must match (default: this one)",
    )
    parser.add_argument("--task-ref", default="HEAD", help="Index revision (default: HEAD)")
    parser.add_argument(
        "--gitleaks",
        type=Path,
        default=os.environ.get("PUBLICATION_GITLEAKS_BIN") or None,
        help="gitleaks binary for the secret scan (default: $PUBLICATION_GITLEAKS_BIN)",
    )
    parser.add_argument("--staging", type=Path, help="Keep the staged tree in this empty directory")
    parser.add_argument("--message", help="Commit message for the upload")
    parser.add_argument("--note", help=argparse.SUPPRESS)
    parser.add_argument("--print-repo", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.print_repo:
        print(LOGS_REPO)
        return 0
    try:
        if not REPO_ID.fullmatch(args.repo):
            raise PublishError("--repo must be an owner/name repository id")
        if args.check_pr is not None:
            if args.source or args.task or args.create_pr:
                raise PublishError("--check-pr takes no SOURCE, --task or --create-pr")
            if args.check_pr < 0:
                raise PublishError("--check-pr needs a pull request number")
            return check_pr(args)
        if args.source is None or args.task is None:
            parser.error("SOURCE and --task are required (or use --check-pr N)")
        if args.gitleaks is None:
            print("NOTICE no --gitleaks: the secret scan is skipped here; the Hub audit runs it")
        return publish(args)
    except pub.CheckError as exc:
        print(f"FAIL {exc}")
    except PublishError as exc:
        print(f"ERROR {exc}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
