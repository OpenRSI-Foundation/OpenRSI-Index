"""Tests for publish_logs.py: synthetic data only; uploads and Hub calls are mocked
and nothing reaches the network."""

from __future__ import annotations

import bz2
import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest
import zipfile
import zlib
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("publish_logs", HERE / "publish_logs.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
GITLEAKS = os.environ.get("PUBLICATION_GITLEAKS_BIN")
REQUIRE_INTEGRATION = os.environ.get("PUBLICATION_REQUIRE_INTEGRATION") == "1"
NOTE = tool.HISTORY_NOTE


def need(test: unittest.TestCase, available, reason: str):
    if not available:
        if REQUIRE_INTEGRATION:
            test.fail(reason + " (required by PUBLICATION_REQUIRE_INTEGRATION)")
        test.skipTest(reason)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def init_repo(repo: Path):
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Publish test")
    git(repo, "config", "user.email", "publish-test@example.invalid")


def commit(repo: Path) -> str:
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "fixture", "--allow-empty")
    return git(repo, "rev-parse", "HEAD")


RAW = b"".join(f"line {n}: tool output\n".encode() for n in range(2000))


def stored_gzip(data: bytes) -> bytes:
    """A gzip stream of stored (uncompressed) deflate blocks with mtime 0: the
    same bytes under every zlib build, so frozen digests stay valid."""
    out = bytearray(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff")
    blocks = [data[i : i + 65535] for i in range(0, len(data), 65535)] or [b""]
    for number, block in enumerate(blocks, 1):
        out.append(1 if number == len(blocks) else 0)
        out += len(block).to_bytes(2, "little") + (0xFFFF ^ len(block)).to_bytes(2, "little")
        out += block
    out += zlib.crc32(data).to_bytes(4, "little") + (len(data) & 0xFFFFFFFF).to_bytes(4, "little")
    return bytes(out)


def transcript(label: str, lines: int) -> bytes:
    return b"".join(f"[{label}] event {n}: tool call and output\n".encode() for n in range(lines))


def dumps(doc) -> bytes:
    return (json.dumps(doc, indent=2) + "\n").encode()


SGLANG_RUNS = (
    "d154-astra-xhigh-650175-20261001T230314",
    "d154-rolling-astra-xhigh-655820-20261002T120036",
    "omni-resume3-669485-20261004T141956Z-resume3",
)


def history_fixture(root: Path) -> dict[str, bytes]:
    """Miniatures of the three log roots OpenRSI-Index published compressed
    (rsi-logs/ at e00f0a9a): the same paths, manifest formats, key order,
    listing order and field sets, with small transcripts. Returns path -> bytes."""
    files: dict[str, bytes] = {}

    def put(path, data):
        files[path] = data if isinstance(data, bytes) else data.encode()

    # ace-playbook-repair and kev-decision-architecture: a gzip transcript and
    # its version 1 manifest, no listing manifest.
    for task, lines in (("ace-playbook-repair", 3000), ("kev-decision-architecture", 1700)):
        raw = transcript(task, lines)
        packed = stored_gzip(raw)
        run = f"{task}/codex-gpt-5.6-sol"
        put(f"{task}/README.md", f"# {task} research logs\n")
        put(f"{run}/README.md", "Codex run.\n")
        put(f"{run}/agent_output.txt.gz", packed)
        put(
            f"{run}/agent_output.manifest.json",
            dumps(
                {
                    "version": 1,
                    "artifact": "agent_output.txt.gz",
                    "encoding": "utf-8",
                    "compression": "gzip",
                    "gzip_mtime": 0,
                    "restored_name": "agent_output.txt",
                    "sanitized_uncompressed_size": len(raw),
                    "sanitized_uncompressed_sha256": sha(raw),
                    "sanitized_uncompressed_line_feeds": raw.count(b"\n"),
                    "compressed_size": len(packed),
                    "compressed_sha256": sha(packed),
                }
            ),
        )
        put(f"{run}/final_result.json", dumps({"best_score": 0.5}))
        put(f"{task}/claude-code-claude-opus-5/agent_output.txt", transcript("claude", 40))

    # sglang-omni-tts-runtime: two gzip transcripts and one chunked gzip, all
    # named by publication.json, whose other entries record uncompressed digests.
    task = "sglang-omni-tts-runtime"
    entries = []

    def plain(rel, data):
        put(f"{task}/{rel}", data)
        data = files[f"{task}/{rel}"]
        entries.append(
            {
                "path": rel,
                "bytes": len(data),
                "sha256": sha(data),
                "uncompressed_bytes": len(data),
                "uncompressed_sha256": sha(data),
                "line_count": data.count(b"\n"),
                "compression": None,
            }
        )

    for number, run_id in enumerate(SGLANG_RUNS):
        run = f"codex-gpt-6-astra/runs/{run_id}"
        inner = f"{run}/sglang-omni-tts-runtime"
        raw = transcript(run_id, 900 + 700 * number)
        packed = stored_gzip(raw)
        if number < 2:
            put(f"{task}/{inner}/agent_output.txt.gz", packed)
            entries.append(
                {
                    "path": f"{inner}/agent_output.txt.gz",
                    "bytes": len(packed),
                    "sha256": sha(packed),
                    "uncompressed_bytes": len(raw),
                    "uncompressed_sha256": sha(raw),
                    "line_count": raw.count(b"\n"),
                    "compression": "gzip",
                }
            )
        else:
            plain(f"{run}/research-summary.json", dumps({"best_round": "agent-4"}))
            half = 40000
            parts = []
            for index, block in enumerate((packed[:half], packed[half:]), 1):
                name = f"part-{index:04d}.bin"
                put(f"{task}/{inner}/agent_output.txt.gz.parts/{name}", block)
                parts.append({"path": name, "bytes": len(block), "sha256": sha(block)})
            chunk = dumps(
                {
                    "schema_version": 1,
                    "source_name": "agent_output.txt.gz",
                    "sha256": sha(packed),
                    "bytes": len(packed),
                    "uncompressed_sha256": sha(raw),
                    "uncompressed_bytes": len(raw),
                    "line_count": raw.count(b"\n"),
                    "compression": "gzip",
                    "parts": parts,
                    "note": "Concatenate every part in manifest order to restore the "
                    "complete gzip; no bytes were truncated.",
                }
            )
            put(f"{task}/{inner}/agent_output.txt.gz.parts/manifest.json", chunk)
            entries += [dict(part, path=f"{inner}/agent_output.txt.gz.parts/{part['path']}") for part in parts]
            entries.append(
                {
                    "path": f"{inner}/agent_output.txt.gz.parts/manifest.json",
                    "bytes": len(chunk),
                    "sha256": sha(chunk),
                }
            )
        plain(f"{inner}/agent_prompt.md", f"# Task prompt for {run_id}\n")
        plain(f"{inner}/evolve_state.json", dumps({"rounds": number + 1}))
        plain(f"{run}/source-publication.json", dumps({"phase": "final", "run_id": run_id, "version": 1}))
    put(f"{task}/README.md", "# SGLang Omni TTS runtime research logs\n")
    put(f"{task}/measurements.json", dumps({"measurements": []}))
    put(
        f"{task}/publication.json",
        dumps(
            {
                "schema_version": 1,
                "derived_evidence": [
                    {"path": "measurements.json", "sha256": sha(files[f"{task}/measurements.json"]), "description": "Judge measurements"}
                ],
                "source_repository": "OpenRSI-Foundation/sglang-omni-tts-runtime-d154",
                "source_commit": "c819dc10aaf7dab0c2d0ed5eea5499e449a6a9d5",
                "agent": "codex",
                "model": "gpt-6-astra",
                "reasoning_effort": "xhigh",
                "runs": [{"run_id": run_id, "best_score": None} for run_id in SGLANG_RUNS],
                "files": entries,
                "redaction_counts": {"provider-key": 0, "home": 67},
                "compression": "Complete sanitized gzip streams; oversized streams are losslessly "
                "chunked with reconstruction manifests; no truncation",
                "secret_scan": "Gitleaks 8.30.1; full uncompressed logs; contextual false positives reviewed",
                "source_file_hashes": {"logs/runs/x/.gitignore": {"bytes": 0, "sha256": sha(b"")}},
            }
        ),
    )
    for path, data in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return files


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.src = self.root / "src"
        self.src.mkdir()
        self.tree = self.root / "tree"

    def write(self, path, content, root=None):
        p = (root or self.src) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        return p

    def stage(self, dest="demo-task", note=NOTE):
        tool.copy_source(self.src, self.tree / dest)
        return tool.normalize(self.tree, note)

    def staged(self, path):
        return self.tree / "demo-task" / path

    def assertRefused(self, pattern):
        with self.assertRaisesRegex(tool.PublishError, pattern):
            self.stage()

    def split_runner(self, name, data, chunk_size):
        """The task runner's split_run_log.py format."""
        directory = name + ".parts"
        parts = []
        for number, start in enumerate(range(0, len(data), chunk_size), 1):
            block = data[start : start + chunk_size]
            part = f"part-{number:06d}.bin"
            self.write(f"{directory}/{part}", block)
            parts.append({"file": part, "size": len(block), "sha256": sha(block)})
        self.write(directory + "/.gitattributes", "part-*.bin -text -filter\n")
        manifest = {
            "version": 1,
            "source_name": Path(name).name,
            "source_size": len(data),
            "source_sha256": sha(data),
            "chunk_size": chunk_size,
            "parts": parts,
        }
        self.write(directory + "/manifest.json", json.dumps(manifest))
        return directory, manifest

    def split_legacy(self, name, packed, raw):
        """The path/bytes + sha256/bytes chunk format (as in published history)."""
        directory = name + ".parts"
        half = len(packed) // 2
        parts = []
        for number, block in enumerate((packed[:half], packed[half:]), 1):
            part = f"part-{number:04d}.bin"
            self.write(f"{directory}/{part}", block)
            parts.append({"path": part, "bytes": len(block), "sha256": sha(block)})
        manifest = {
            "schema_version": 1,
            "source_name": Path(name).name,
            "sha256": sha(packed),
            "bytes": len(packed),
            "uncompressed_sha256": sha(raw),
            "uncompressed_bytes": len(raw),
            "line_count": raw.count(b"\n"),
            "compression": "gzip",
            "parts": parts,
        }
        self.write(directory + "/manifest.json", json.dumps(manifest))
        return directory, manifest


class NormalizeTests(Base):
    def test_gzip_with_original_manifest(self):
        packed = gzip.compress(RAW, mtime=0)
        self.write("m/agent_output.txt.gz", packed)
        original = {
            "version": 1,
            "artifact": "agent_output.txt.gz",
            "compression": "gzip",
            "sanitized_uncompressed_size": len(RAW),
            "sanitized_uncompressed_sha256": sha(RAW),
            "sanitized_uncompressed_line_feeds": 2000,
            "compressed_size": len(packed),
            "compressed_sha256": sha(packed),
        }
        self.write("m/agent_output.manifest.json", json.dumps(original))
        converted = self.stage()
        self.assertEqual(len(converted), 1)
        self.assertEqual(self.staged("m/agent_output.txt").read_bytes(), RAW)
        self.assertFalse(self.staged("m/agent_output.txt.gz").exists())
        doc = json.loads(self.staged("m/agent_output.manifest.json").read_text())
        self.assertEqual(
            doc,
            {
                "version": 2,
                "artifact": "agent_output.txt",
                "size": len(RAW),
                "sha256": sha(RAW),
                "line_feeds": 2000,
                "converted_from": {
                    "artifact": "agent_output.txt.gz",
                    "compression": "gzip",
                    "compressed_size": len(packed),
                    "compressed_sha256": sha(packed),
                    "original_manifest": original,
                },
                "note": NOTE,
            },
        )

    def test_original_manifest_must_match(self):
        packed = gzip.compress(RAW)
        self.write("m/agent_output.txt.gz", packed)
        self.write(
            "m/agent_output.manifest.json",
            json.dumps(
                {
                    "artifact": "agent_output.txt.gz",
                    "sanitized_uncompressed_line_feeds": 1999,
                    "compressed_sha256": sha(packed),
                }
            ),
        )
        self.assertRefused("sanitized_uncompressed_line_feeds does not match")

    def test_unrelated_file_at_the_manifest_path_is_kept_safe(self):
        self.write("m/agent_output.txt.gz", gzip.compress(RAW))
        self.write("m/agent_output.manifest.json", json.dumps({"files": []}))
        self.assertRefused("exists and does not describe agent_output.txt.gz")

    def test_runner_chunks_of_a_gzip_are_reassembled_and_decompressed(self):
        packed = gzip.compress(RAW, mtime=0)
        directory, manifest = self.split_runner("m/trace.jsonl.gz", packed, 1000)
        self.stage()
        self.assertFalse(self.staged(directory).exists())
        self.assertEqual(self.staged("m/trace.jsonl").read_bytes(), RAW)
        doc = json.loads(self.staged("m/trace.manifest.json").read_text())
        source = doc["converted_from"]
        self.assertEqual(source["artifact"], "trace.jsonl.gz.parts")
        self.assertEqual(source["compression"], "gzip")
        self.assertEqual(source["compressed_sha256"], sha(packed))
        self.assertEqual(
            source["parts"],
            [{"path": p["file"], "bytes": p["size"], "sha256": p["sha256"]} for p in manifest["parts"]],
        )
        self.assertEqual(source["original_manifest"], manifest)

    def test_legacy_chunks_and_listing_manifests_point_at_the_raw_file(self):
        packed = gzip.compress(RAW, mtime=0)
        directory, manifest = self.split_legacy("run/agent_output.txt.gz", packed, RAW)
        self.write("run/notes.md", "notes\n")
        files = [
            {"path": "run/notes.md", "bytes": 6, "sha256": sha(b"notes\n")},
        ]
        for part in manifest["parts"]:
            files.append({"path": f"{directory}/{part['path']}", "bytes": part["bytes"], "sha256": part["sha256"]})
        chunk_manifest = (self.src / directory / "manifest.json").read_bytes()
        files.append({"path": f"{directory}/manifest.json", "bytes": len(chunk_manifest), "sha256": sha(chunk_manifest)})
        publication = json.dumps({"schema_version": 1, "files": files}).encode()
        self.write("publication.json", publication)
        self.write(
            "release-manifest.json",
            json.dumps({"files": [{"path": "publication.json", "bytes": len(publication), "sha256": sha(publication)}]}),
        )
        self.stage()
        doc = json.loads(self.staged("publication.json").read_text())
        v2_bytes = self.staged("run/agent_output.manifest.json").read_bytes()
        self.assertEqual(
            doc["files"],
            [
                files[0],
                {"path": "run/agent_output.txt", "bytes": len(RAW), "sha256": sha(RAW)},
                {"path": "run/agent_output.manifest.json", "bytes": len(v2_bytes), "sha256": sha(v2_bytes)},
            ],
        )
        release = json.loads(self.staged("release-manifest.json").read_text())
        rewritten = self.staged("publication.json").read_bytes()
        self.assertEqual(release["files"][0]["sha256"], sha(rewritten))
        self.assertEqual(release["files"][0]["bytes"], len(rewritten))
        v2 = json.loads(self.staged("run/agent_output.manifest.json").read_text())
        self.assertEqual(v2["converted_from"]["original_manifest"], manifest)
        self.assertEqual(v2["line_feeds"], 2000)

    def test_gzip_listing_entry_becomes_raw(self):
        packed = gzip.compress(RAW, mtime=0)
        self.write("m/out.txt.gz", packed)
        entry = {
            "path": "m/out.txt.gz",
            "bytes": len(packed),
            "sha256": sha(packed),
            "uncompressed_bytes": len(RAW),
            "uncompressed_sha256": sha(RAW),
            "line_count": 2000,
            "compression": "gzip",
        }
        self.write("publication.json", json.dumps({"files": [entry]}))
        self.stage()
        new, listed = json.loads(self.staged("publication.json").read_text())["files"]
        v2_bytes = self.staged("m/out.manifest.json").read_bytes()
        self.assertEqual(
            listed, {"path": "m/out.manifest.json", "bytes": len(v2_bytes), "sha256": sha(v2_bytes)}
        )
        self.assertEqual(
            new,
            {
                "path": "m/out.txt",
                "bytes": len(RAW),
                "sha256": sha(RAW),
                "uncompressed_bytes": len(RAW),
                "uncompressed_sha256": sha(RAW),
                "line_count": 2000,
                "compression": None,
            },
        )

    def test_replaced_manifest_is_listed_after_its_raw_file(self):
        packed = gzip.compress(RAW, mtime=0)
        self.write("m/agent_output.txt.gz", packed)
        v1 = json.dumps(
            {"version": 1, "artifact": "agent_output.txt.gz", "compressed_sha256": sha(packed)}
        ).encode()
        self.write("m/agent_output.manifest.json", v1)
        self.write("m/notes.md", "notes\n")
        listed = [
            {"path": "agent_output.manifest.json", "bytes": len(v1), "sha256": sha(v1)},
            {"path": "notes.md", "sha256": sha(b"notes\n")},
            {"path": "agent_output.txt.gz", "sha256": sha(packed), "extra": "kept"},
        ]
        self.write("m/release-manifest.json", json.dumps({"compression": "gzip", "files": listed}, ensure_ascii=False))
        self.stage()
        v2 = self.staged("m/agent_output.manifest.json").read_bytes()
        doc = json.loads(self.staged("m/release-manifest.json").read_text())
        self.assertEqual(
            doc["files"],
            [
                listed[1],
                {"path": "agent_output.txt", "sha256": sha(RAW), "extra": "kept"},
                {"path": "agent_output.manifest.json", "bytes": len(v2), "sha256": sha(v2)},
            ],
        )
        self.assertEqual(doc["compression"], tool.RAW_COMPRESSION)

    def test_listing_keeps_its_ascii_escaping(self):
        packed = gzip.compress(RAW, mtime=0)
        self.write("m/out.txt.gz", packed)
        for ascii_only in (True, False):
            with self.subTest(ascii_only=ascii_only):
                shutil.rmtree(self.tree, ignore_errors=True)
                doc = {"title": "Café run", "files": [{"path": "m/out.txt.gz", "sha256": sha(packed)}]}
                self.write("publication.json", json.dumps(doc, indent=2, ensure_ascii=ascii_only) + "\n")
                self.stage()
                data = self.staged("publication.json").read_bytes()
                self.assertEqual(b"\\u00e9" in data, ascii_only)
                self.assertEqual(json.loads(data)["title"], "Café run")

    def test_listing_with_wrong_checksum_is_refused(self):
        packed = gzip.compress(RAW)
        self.write("m/out.txt.gz", packed)
        self.write("publication.json", json.dumps({"files": [{"path": "m/out.txt.gz", "sha256": "0" * 64}]}))
        self.assertRefused("listed SHA-256/size do not match")

    def test_chunked_raw_file(self):
        self.split_runner("m/run.log", RAW, 4096)
        self.stage(note=None)
        self.assertEqual(self.staged("m/run.log").read_bytes(), RAW)
        doc = json.loads(self.staged("m/run.manifest.json").read_text())
        self.assertEqual(doc["converted_from"]["compression"], "none")
        self.assertEqual(doc["converted_from"]["compressed_sha256"], sha(RAW))
        self.assertIn("chunked upload", doc["note"])

    def test_unconverted_formats_are_refused(self):
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode="w") as archive:
            info = tarfile.TarInfo("a.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"a"))
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, "w") as archive:
            archive.writestr("a.txt", "a")
        cases = {
            "m/out.txt.xz": (lzma.compress(RAW), "compressed or archived"),
            "m/out.txt.bz2": (bz2.compress(RAW), "compressed or archived"),
            "m/out.zst": (b"\x28\xb5\x2f\xfd" + RAW, "compressed or archived"),
            "m/bundle.zip": (zipped.getvalue(), "compressed or archived"),
            "m/bundle.tar": (tar.getvalue(), "compressed or archived"),
            "m/workspace": (tar.getvalue(), "by content"),
            "m/out.txt": (gzip.compress(RAW), "gzip content without a .gz name"),
            "m/broken.txt.gz": (gzip.compress(RAW)[:-20], "not a valid gzip file"),
        }
        for name, (content, message) in cases.items():
            with self.subTest(name=name):
                shutil.rmtree(self.src)
                shutil.rmtree(self.tree, ignore_errors=True)
                self.write(name, content)
                self.assertRefused(message)

    def test_tar_gz_is_refused_after_decompression(self):
        tar = io.BytesIO()
        with tarfile.open(fileobj=tar, mode="w:gz") as archive:
            info = tarfile.TarInfo("a.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"a"))
        self.write("m/bundle.tar.gz", tar.getvalue())
        self.assertRefused("demo-task/m/bundle.tar: compressed or archived")

    def test_chunk_problems_are_refused(self):
        packed = gzip.compress(RAW)
        cases = {
            "part mismatch": lambda d: (self.src / d / "part-000001.bin").write_bytes(b"x"),
            "extra file": lambda d: self.write(f"{d}/stray.bin", "x"),
            "missing part": lambda d: (self.src / d / "part-000002.bin").unlink(),
            "raw beside chunks": lambda d: self.write("m/trace.jsonl.gz", packed),
        }
        messages = {
            "part mismatch": "part part-000001.bin does not match",
            "extra file": "stray.bin: not listed by the chunk manifest",
            "missing part": "missing part part-000002.bin",
            "raw beside chunks": "exists beside its chunk directory",
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                shutil.rmtree(self.src)
                shutil.rmtree(self.tree, ignore_errors=True)
                self.src.mkdir()
                directory, _ = self.split_runner("m/trace.jsonl.gz", packed, 1000)
                mutate(directory)
                self.assertRefused(messages[label])
        shutil.rmtree(self.src)
        shutil.rmtree(self.tree, ignore_errors=True)
        self.write("m/x.txt.parts/part-1.bin", "x")
        self.assertRefused("chunk directory without a chunk manifest")

    def test_name_collisions_are_refused(self):
        self.write("m/out.txt.gz", gzip.compress(RAW))
        self.write("m/out.txt", "other\n")
        self.assertRefused("out.txt also exists")
        shutil.rmtree(self.src)
        shutil.rmtree(self.tree)
        self.write("m/out.txt.gz", gzip.compress(RAW))
        self.write("m/out.jsonl.gz", gzip.compress(b"{}\n"))
        self.assertRefused("share the name out")

    def test_links_and_git_metadata_are_refused(self):
        self.write("a.txt", "a\n")
        (self.src / "link").symlink_to("a.txt")
        with self.assertRaisesRegex(tool.PublishError, "link: symbolic link"):
            self.stage()
        (self.src / "link").unlink()
        self.write(".git/config", "x")
        shutil.rmtree(self.tree, ignore_errors=True)
        with self.assertRaisesRegex(tool.PublishError, "Git metadata"):
            self.stage()

    def test_raw_input_is_unchanged_and_normalizing_is_idempotent(self):
        self.write("README.md", "Run\n")
        self.write("m/agent_output.txt", RAW)
        self.write("m/state.json", '{"submissions": []}\n')
        self.write("m/arrays.npz", b"PK\x03\x04" + b"\0" * 30)
        self.assertEqual(self.stage(), [])
        self.assertEqual(snapshot(self.tree / "demo-task"), snapshot(self.src))
        # A converted tree is raw: converting it again changes nothing.
        shutil.rmtree(self.src)
        shutil.rmtree(self.tree)
        self.write("m/out.txt.gz", gzip.compress(RAW))
        self.split_runner("m/run.log", RAW, 4096)
        self.stage()
        once = snapshot(self.tree)
        again = self.root / "again"
        tool.copy_source(self.tree, again)
        self.assertEqual(tool.normalize(again, NOTE), [])
        self.assertEqual(snapshot(again), once)

    def test_source_is_never_modified(self):
        self.write("m/out.txt.gz", gzip.compress(RAW))
        self.split_runner("m/run.log", RAW, 4096)
        before = snapshot(self.src)
        self.stage()
        self.assertEqual(snapshot(self.src), before)


# SHA-256 of every file the one-off conversion of OpenRSI-Index's rsi-logs/
# history (the Logs repository's initial import) wrote for history_fixture().
HISTORY_DIGESTS = {
    "ace-playbook-repair/README.md":
        "409ac9b78128af79676a5be263bdecd664acead498da02449e9b7b1f47024c2a",
    "ace-playbook-repair/claude-code-claude-opus-5/agent_output.txt":
        "480c91c9e8aa48f1d7b8b7330b55379f3420b9cc3a2de958c709d6d04ddbd2a7",
    "ace-playbook-repair/codex-gpt-5.6-sol/README.md":
        "a3bd9ffa50df22dfb79ade531a5c40b61a5ead73e5955e1fae6586cd496f04aa",
    "ace-playbook-repair/codex-gpt-5.6-sol/agent_output.manifest.json":
        "d171612735993b88776145e5259e46f19a5438f5972bfda942a25c7fe94a50bf",
    "ace-playbook-repair/codex-gpt-5.6-sol/agent_output.txt":
        "bc0a73f3c908b5765a68b6e479beae78c219a7427f88de05df0596704aaa80c4",
    "ace-playbook-repair/codex-gpt-5.6-sol/final_result.json":
        "1803ba26a6da37d0003ce0f9ca34cfd700a79a2d933915d57c219ddc12a1f476",
    "kev-decision-architecture/README.md":
        "d464befe41cfdf758a7113754277389da36e8f8780f5701d433c34460d3cfc6e",
    "kev-decision-architecture/claude-code-claude-opus-5/agent_output.txt":
        "480c91c9e8aa48f1d7b8b7330b55379f3420b9cc3a2de958c709d6d04ddbd2a7",
    "kev-decision-architecture/codex-gpt-5.6-sol/README.md":
        "a3bd9ffa50df22dfb79ade531a5c40b61a5ead73e5955e1fae6586cd496f04aa",
    "kev-decision-architecture/codex-gpt-5.6-sol/agent_output.manifest.json":
        "37049b47abdcc0792e99dd41cde68e009e4ed1559284fdcf4f3d9b0f9b627ae8",
    "kev-decision-architecture/codex-gpt-5.6-sol/agent_output.txt":
        "330a58959ee0e1fa3cb58ddc834d12ee6e4ad7608b7fbe8883852f8aff90cc20",
    "kev-decision-architecture/codex-gpt-5.6-sol/final_result.json":
        "1803ba26a6da37d0003ce0f9ca34cfd700a79a2d933915d57c219ddc12a1f476",
    "sglang-omni-tts-runtime/README.md":
        "3de2aaa33ac32d24d4a42d5ddea1816fe15664c92b91377c8920312dad1ad1c5",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-astra-xhigh-650175-20261001T230314/sglang-omni-tts-runtime/agent_output.manifest.json":
        "f6f1f566a60c3c676eb35f301b039af50fd305cb86ddbffd95c9455ea2a201bd",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-astra-xhigh-650175-20261001T230314/sglang-omni-tts-runtime/agent_output.txt":
        "6641cc7d72b1e2eb18d4babf620c75185d2c8d89737e2cc2a911255b42dcf7a8",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-astra-xhigh-650175-20261001T230314/sglang-omni-tts-runtime/agent_prompt.md":
        "efb9e53ba9b3fdc578e997cdb0c789cf10a4b55ac001c39da207cece4e749b25",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-astra-xhigh-650175-20261001T230314/sglang-omni-tts-runtime/evolve_state.json":
        "93b8aa3ce1c558b39119d7892821add59e1c3453534375fffa9ec70c12d51e93",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-astra-xhigh-650175-20261001T230314/source-publication.json":
        "a13af863e4c2766f1e18d398f39d1ed1ea3eb8106db57030bdb02165b1961f62",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-rolling-astra-xhigh-655820-20261002T120036/sglang-omni-tts-runtime/agent_output.manifest.json":
        "75ee854e75a969fd7ab1dd9222aa89341587635cf06c4cf8623c5fc380f75df5",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-rolling-astra-xhigh-655820-20261002T120036/sglang-omni-tts-runtime/agent_output.txt":
        "66641cf29573b3c52ebb64ac850e3dbe92c33070e772fe409ac971720e1da127",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-rolling-astra-xhigh-655820-20261002T120036/sglang-omni-tts-runtime/agent_prompt.md":
        "c4b7d7119e61c7203df4afbdedd40b46409d0e4d4fc5dcbc8ed87ce651bd037f",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-rolling-astra-xhigh-655820-20261002T120036/sglang-omni-tts-runtime/evolve_state.json":
        "2215106aa1551b9390beb49d328b526f563797aace7d7705034a6b0c62ea4757",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/d154-rolling-astra-xhigh-655820-20261002T120036/source-publication.json":
        "ebad59f1dd972195d8377c4c38ac15bcd2980fff64685f6854a2049c180d769d",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/omni-resume3-669485-20261004T141956Z-resume3/research-summary.json":
        "1f9034a1a362ecaefa5e3d7275aa4e13b73acc93ff04ee67d0d9466d4a5f4527",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/omni-resume3-669485-20261004T141956Z-resume3/sglang-omni-tts-runtime/agent_output.manifest.json":
        "8e62b99a69f287d163a8bc510a0ee7d5a670379e1238f5e9c6b602a8757849a7",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/omni-resume3-669485-20261004T141956Z-resume3/sglang-omni-tts-runtime/agent_output.txt":
        "223e6530bab4d30d38468a62d61dead3cd81a98dde35b1c48355040ee963fde8",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/omni-resume3-669485-20261004T141956Z-resume3/sglang-omni-tts-runtime/agent_prompt.md":
        "ae779dd6b108a919e3c5bb7734abefafdd2a1367f5ef0fbb6ea3a8ab74539a67",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/omni-resume3-669485-20261004T141956Z-resume3/sglang-omni-tts-runtime/evolve_state.json":
        "d377939256b4cd8d0c31dd546f681fd149798d35bf0db619e6ea40e280bf1b46",
    "sglang-omni-tts-runtime/codex-gpt-6-astra/runs/omni-resume3-669485-20261004T141956Z-resume3/source-publication.json":
        "715dc8c6cd4c70311d4fb13cad66299d36efb147843abfed24af12afdcd02a8e",
    "sglang-omni-tts-runtime/measurements.json":
        "101f5a74238a91140b075b3895e4fbf5058f14f218b9398aefaef1d7e8b0175c",
    "sglang-omni-tts-runtime/publication.json":
        "b9110a5d10c63ace61107fc41ee167baeb3d719a184dc4ac7e5427d1d8e794ba",
}


class HistoryConversionTests(unittest.TestCase):
    """Uploading a log root that was published compressed in OpenRSI-Index must
    reproduce the Logs repository's converted history byte for byte, so a
    re-upload never rewrites it."""

    def test_normalizing_the_history_trees_matches_the_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            history_fixture(src)
            got = {}
            for task in sorted(p.name for p in src.iterdir()):
                # Each log root is staged and normalized on its own, as uploaded.
                tree = Path(tmp) / f"tree-{task}"
                tool.copy_source(src / task, tree / task)
                tool.normalize(tree, NOTE)
                got.update({path: sha(data) for path, data in snapshot(tree).items()})
            publication = (Path(tmp) / "tree-sglang-omni-tts-runtime/sglang-omni-tts-runtime/publication.json").read_bytes()
        mismatched = sorted(p for p in set(got) | set(HISTORY_DIGESTS) if got.get(p) != HISTORY_DIGESTS.get(p))
        self.assertEqual(mismatched, [], "files differing from the history conversion")
        doc = json.loads(publication)
        self.assertEqual(publication, dumps(doc))
        self.assertEqual(doc["compression"], tool.RAW_COMPRESSION)
        listed = [e for e in doc["files"] if "agent_output" in e["path"]]
        resume3 = "codex-gpt-6-astra/runs/" + SGLANG_RUNS[2] + "/sglang-omni-tts-runtime/"
        self.assertEqual(len(listed), 6)
        for raw, manifest in zip(listed[::2], listed[1::2]):
            # Each raw file is followed by its version 2 manifest and carries
            # the uncompressed fields its sibling entries record.
            self.assertTrue(raw["path"].endswith("/agent_output.txt"))
            self.assertEqual(manifest["path"], raw["path"].removesuffix(".txt") + ".manifest.json")
            self.assertEqual(list(manifest), ["path", "bytes", "sha256"])
            self.assertEqual(
                list(raw),
                ["path", "bytes", "sha256", "uncompressed_bytes", "uncompressed_sha256", "line_count", "compression"],
            )
            self.assertEqual(raw["uncompressed_sha256"], raw["sha256"])
            self.assertIsNone(raw["compression"])
        self.assertEqual(listed[4]["path"], resume3 + "agent_output.txt")


class DestinationTests(unittest.TestCase):
    POLICY = {"log_task_aliases": {"signature-tasks/gpic-10m": "signature-tasks/gpic_generation"}}

    def test_default_and_alias(self):
        self.assertEqual(tool.resolve_dest("demo-task", None, self.POLICY), ("demo-task", "demo-task"))
        self.assertEqual(
            tool.resolve_dest("signature-tasks/gpic_generation", None, self.POLICY),
            ("signature-tasks/gpic-10m", "signature-tasks/gpic-10m"),
        )
        self.assertEqual(
            tool.resolve_dest("demo-task", "demo-task/codex/run-1/", self.POLICY),
            ("demo-task/codex/run-1", "demo-task"),
        )

    def test_outside_the_log_root_refused(self):
        for task, dest in (
            ("demo-task", "other-task/run"),
            ("demo-task", "demo-task/../other"),
            ("signature-tasks/x", "signature-tasks"),
            ("demo-task", "/abs"),
        ):
            with self.subTest(dest=dest), self.assertRaises(tool.PublishError):
                tool.resolve_dest(task, dest, self.POLICY)
        with self.assertRaisesRegex(tool.PublishError, "--task must be"):
            tool.resolve_dest("../x", None, self.POLICY)


class MainTests(Base):
    def setUp(self):
        super().setUp()
        self.index = self.root / "index"
        init_repo(self.index)
        self.write("rsi-tasks/demo-task/task.toml", "x\n", root=self.index)
        commit(self.index)
        self.write("README.md", "Recorded run\n")
        self.write("m/agent_output.txt.gz", gzip.compress(RAW))

    def main(self, *args, remote=True, upload_url="https://example.invalid/pr/1"):
        if "--dry-run" not in args and "--print-repo" not in args and "--gitleaks" not in args:
            # Uploads must run the secret scan.
            need(self, GITLEAKS, "set PUBLICATION_GITLEAKS_BIN for upload tests")
            args = (*args, "--gitleaks", GITLEAKS)
        output = io.StringIO()
        with (
            patch.object(tool, "remote_root_has_files", return_value=remote) as remote_mock,
            patch.object(tool, "upload", return_value=upload_url) as upload_mock,
            contextlib.redirect_stdout(output),
        ):
            code = tool.main([*args, "--index", str(self.index)])
        return code, output.getvalue(), upload_mock, remote_mock

    def test_dry_run_checks_and_uploads_nothing(self):
        before = snapshot(self.src)
        code, out, upload, remote = self.main(str(self.src), "--task", "demo-task", "--dry-run")
        self.assertEqual(code, 0, out)
        self.assertIn("CONVERTED agent_output.txt.gz -> demo-task/m/agent_output.txt", out)
        self.assertIn("NOTICE Logs mode", out)
        self.assertIn("DRY RUN", out)
        upload.assert_not_called()
        remote.assert_not_called()
        self.assertEqual(snapshot(self.src), before)

    def test_upload_as_pull_request(self):
        staging = self.root / "staging"
        code, out, upload, _ = self.main(
            str(self.src), "--task", "demo-task", "--create-pr", "--staging", str(staging)
        )
        self.assertEqual(code, 0, out)
        upload.assert_called_once()
        tree, repo, create_pr, message = upload.call_args.args
        self.assertEqual((repo, create_pr), (tool.LOGS_REPO, True))
        self.assertIn("demo-task", message)
        self.assertEqual(
            sorted(snapshot(tree)),
            ["demo-task/README.md", "demo-task/m/agent_output.manifest.json", "demo-task/m/agent_output.txt"],
        )
        self.assertIn("UPLOADED https://example.invalid/pr/1", out)

    def test_failed_check_blocks_the_upload(self):
        code, out, upload, _ = self.main(str(self.src), "--task", "missing-task")
        self.assertEqual(code, 1)
        self.assertIn("FAIL rsi-logs/missing-task: corresponding public task.toml not found", out)
        upload.assert_not_called()
        self.write("m/bad.jsonl", "{bad\n")
        code, out, upload, _ = self.main(str(self.src), "--task", "demo-task")
        self.assertEqual(code, 1)
        self.assertIn("invalid JSONL at line 1", out)
        upload.assert_not_called()

    def test_refused_input_blocks_the_upload(self):
        self.write("m/x.xz", lzma.compress(RAW))
        code, out, upload, _ = self.main(str(self.src), "--task", "demo-task")
        self.assertEqual(code, 1)
        self.assertIn("ERROR demo-task/m/x.xz", out)
        upload.assert_not_called()

    def test_only_a_log_root_new_to_the_hub_needs_a_readme(self):
        (self.src / "README.md").unlink()
        # The root has no files on the Hub yet: a new archive must bring a README.
        for dest, fix in (
            (None, "add it to SOURCE"),
            (
                "demo-task/codex/run-1",
                "publish demo-task/README.md first (SOURCE is staged at demo-task/codex/run-1/)",
            ),
        ):
            with self.subTest(dest=dest):
                args = (str(self.src), "--task", "demo-task", "--dry-run")
                args += ("--dest", dest) if dest else ()
                code, out, upload, remote = self.main(*args, remote=False)
                self.assertEqual(code, 1, out)
                self.assertIn(
                    "ERROR demo-task: new log archive (no files under demo-task/ on the Hub "
                    "yet) needs README.md describing source and recorded protocol; " + fix,
                    out,
                )
                remote.assert_called_once_with(tool.LOGS_REPO, "demo-task")
                upload.assert_not_called()
        # The root already has files on the Hub (with or without a README there),
        # as the checker accepts for an archive that is not newly added.
        for args in (
            (str(self.src), "--task", "demo-task", "--dry-run"),
            (str(self.src), "--task", "demo-task", "--dest", "demo-task/codex/run-1"),
        ):
            with self.subTest(args=args):
                code, out, upload, remote = self.main(*args, remote=True)
                self.assertEqual(code, 0, out)
                remote.assert_called_once_with(tool.LOGS_REPO, "demo-task")
                self.assertEqual(upload.call_count, 0 if "--dry-run" in args else 1)

    def test_cli_arguments(self):
        code, out, _, _ = self.main("--print-repo")
        self.assertEqual((code, out), (0, tool.LOGS_REPO + "\n"))
        code, out, _, _ = self.main(str(self.src), "--check-pr", "3")
        self.assertEqual(code, 1)
        self.assertIn("--check-pr takes no SOURCE", out)
        code, out, _, _ = self.main(str(self.src), "--task", "demo-task", "--repo", "bad id")
        self.assertIn("owner/name", out)
        staging = self.root / "busy"
        staging.mkdir()
        (staging / "x").write_text("x")
        code, out, _, _ = self.main(str(self.src), "--task", "demo-task", "--staging", str(staging))
        self.assertIn("--staging must be empty", out)

    def test_uploads_and_pull_request_checks_require_the_secret_scan(self):
        output = io.StringIO()
        with (
            patch.object(tool, "upload") as upload,
            patch.object(tool, "check_pr") as check_pr,
            contextlib.redirect_stdout(output),
            patch.dict(os.environ, {"PUBLICATION_GITLEAKS_BIN": ""}),
        ):
            for argv in (
                [str(self.src), "--task", "demo-task"],
                [str(self.src), "--task", "demo-task", "--create-pr"],
                ["--check-pr", "1"],
                ["--check-pr", "1", "--dry-run"],
            ):
                self.assertEqual(tool.main([*argv, "--index", str(self.index)]), 1, argv)
        self.assertEqual(output.getvalue().count("ERROR the secret scan needs gitleaks"), 4)
        upload.assert_not_called()
        check_pr.assert_not_called()

    def test_secret_scan_with_gitleaks(self):
        need(self, GITLEAKS, "set PUBLICATION_GITLEAKS_BIN for scanner tests")
        token = ("gh" + "p_") + ("a1b2c3d4e5f6g7h8i9j0" * 2)[:36]
        self.write("m/env.txt", f'GITHUB_TOKEN="{token}"\n')
        code, out, _, _ = self.main(
            str(self.src), "--task", "demo-task", "--dry-run", "--gitleaks", GITLEAKS
        )
        self.assertEqual(code, 0, out)
        self.assertIn("Possible credential: rsi-logs/demo-task/m/env.txt;", out)
        self.assertNotIn(token, out)


class HubClientTests(unittest.TestCase):
    def test_pinned_version_required(self):
        fake = types.ModuleType("huggingface_hub")
        fake.__version__ = "0.0.1"
        with patch.dict(sys.modules, {"huggingface_hub": fake}):
            with self.assertRaisesRegex(tool.PublishError, f"huggingface_hub=={tool.HUB_VERSION}"):
                tool.hub_api()

    def test_upload_call(self):
        calls = []

        class FakeApi:
            def upload_folder(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(pr_url="https://example.invalid/pr/2", commit_url="c")

        with patch.object(tool, "hub_api", return_value=FakeApi()):
            url = tool.upload(Path("/staged"), "o/r", True, "msg")
        self.assertEqual(url, "https://example.invalid/pr/2")
        (kwargs,) = calls
        self.assertEqual(kwargs["repo_type"], "dataset")
        self.assertEqual(kwargs["path_in_repo"], "")
        self.assertTrue(kwargs["create_pr"])
        self.assertIn(".git", kwargs["ignore_patterns"])
        self.assertNotIn("token", kwargs)

    @staticmethod
    def fake_hub(list_repo_tree):
        class EntryNotFoundError(Exception):
            pass

        class FakeApi:
            def list_repo_tree(self, *args, **kwargs):
                return list_repo_tree(EntryNotFoundError, *args, **kwargs)

        return types.SimpleNamespace(
            HfApi=FakeApi, errors=types.SimpleNamespace(EntryNotFoundError=EntryNotFoundError)
        )

    def test_log_root_presence_on_the_hub(self):
        calls = []

        def tree(missing, repo, path, **kwargs):
            calls.append((repo, path, kwargs))
            if path == "absent":
                raise missing("404")

            def entries():
                yield from {"listed": ["listed/run"], "empty": []}[path]

            return entries()

        with patch.object(tool, "hub", return_value=self.fake_hub(tree)):
            self.assertTrue(tool.remote_root_has_files("o/r", "listed"))
            self.assertFalse(tool.remote_root_has_files("o/r", "empty"))
            self.assertFalse(tool.remote_root_has_files("o/r", "absent"))
        # Anonymous unless the user logged in: no token is passed explicitly.
        self.assertEqual(calls[0], ("o/r", "listed", {"repo_type": "dataset"}))

    def test_hub_errors_are_reported_by_type(self):
        def tree(missing, *args, **kwargs):
            raise RuntimeError("response body")

        with patch.object(tool, "hub", return_value=self.fake_hub(tree)):
            with self.assertRaisesRegex(tool.PublishError, r"x/ has files \(RuntimeError\)$"):
                tool.remote_root_has_files("o/r", "x")


class CheckPrTests(unittest.TestCase):
    """--check-pr against a local repository standing in for the Hub."""

    def setUp(self):
        need(self, shutil.which("git-lfs"), "git-lfs is not installed")
        need(self, GITLEAKS, "set PUBLICATION_GITLEAKS_BIN for --check-pr tests")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.index = root / "index"
        init_repo(self.index)
        (self.index / "rsi-tasks/demo-task").mkdir(parents=True)
        (self.index / "rsi-tasks/demo-task/task.toml").write_text("x\n")
        commit(self.index)
        self.remote = root / "remote"
        init_repo(self.remote)
        (self.remote / "demo-task").mkdir()
        (self.remote / "demo-task/README.md").write_text("Run\n")
        commit(self.remote)
        git(self.remote, "checkout", "-qb", "pr1")
        (self.remote / "demo-task/run.log").write_text("ok\n")
        git(self.remote, "update-ref", "refs/pr/1", commit(self.remote))
        git(self.remote, "checkout", "-qb", "pr2", "main")
        (self.remote / "missing-task").mkdir()
        (self.remote / "missing-task/README.md").write_text("x\n")
        git(self.remote, "update-ref", "refs/pr/2", commit(self.remote))
        git(self.remote, "checkout", "-q", "main")

    def check(self, number):
        output = io.StringIO()
        with (
            patch.object(tool, "hub_git_url", return_value=str(self.remote)),
            contextlib.redirect_stdout(output),
        ):
            code = tool.main(
                ["--check-pr", str(number), "--index", str(self.index), "--gitleaks", GITLEAKS]
            )
        return code, output.getvalue()

    def test_pull_requests_are_checked_against_main(self):
        code, out = self.check(1)
        self.assertEqual(code, 0, out)
        self.assertIn("against main", out)
        self.assertIn("PASS 2 published files", out)
        code, out = self.check(2)
        self.assertEqual(code, 1)
        self.assertIn("FAIL rsi-logs/missing-task: corresponding public task.toml", out)
        code, out = self.check(9)
        self.assertEqual(code, 1)
        self.assertIn("ERROR git fetch failed", out)


    def test_checkout_ignores_submitted_conversion_attributes(self):
        git(self.remote, "checkout", "-qb", "pr3", "main")
        log = self.remote / "demo-task/run.log"
        log.write_text("one\ntwo\n")
        (self.remote / ".gitattributes").write_text("demo-task/run.log text eol=crlf\n")
        git(self.remote, "update-ref", "refs/pr/3", commit(self.remote))
        git(self.remote, "checkout", "-q", "main")
        staging = Path(self.tmp.name) / "staging"
        output = io.StringIO()
        with (
            patch.object(tool, "hub_git_url", return_value=str(self.remote)),
            contextlib.redirect_stdout(output),
        ):
            code = tool.main(
                [
                    "--check-pr", "3", "--index", str(self.index), "--staging", str(staging),
                    "--gitleaks", GITLEAKS,
                ]
            )
        out = output.getvalue()
        self.assertEqual(code, 1, out)
        self.assertIn(
            "FAIL rsi-logs/.gitattributes: line 1: only '<pattern> filter=lfs diff=lfs "
            "merge=lfs -text' rules are allowed",
            out,
        )
        # The checked file is the blob, not the CRLF conversion it asks for.
        self.assertEqual((staging / "logs/demo-task/run.log").read_bytes(), b"one\ntwo\n")
        self.assertEqual(
            (staging / "logs/.git/info/attributes").read_text(), tool.NEUTRAL_ATTRIBUTES
        )
        self.assertEqual(
            git(staging / "logs", "config", "lfs.url"), str(self.remote) + ".git/info/lfs"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
