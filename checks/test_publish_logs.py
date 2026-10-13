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
        self.assertEqual(
            doc["files"],
            [
                files[0],
                {"path": "run/agent_output.txt", "bytes": len(RAW), "sha256": sha(RAW)},
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
        (new,) = json.loads(self.staged("publication.json").read_text())["files"]
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
        output = io.StringIO()
        with (
            patch.object(tool, "remote_has_file", return_value=remote) as remote_mock,
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

    def test_new_log_root_needs_a_readme_here_or_on_the_hub(self):
        (self.src / "README.md").unlink()
        args = (str(self.src), "--task", "demo-task", "--dest", "demo-task/codex/run-1", "--dry-run")
        code, out, _, remote = self.main(*args, remote=False)
        self.assertEqual(code, 1)
        self.assertIn("new log archive needs README.md", out)
        remote.assert_called_once_with(tool.LOGS_REPO, "demo-task/README.md")
        code, out, _, _ = self.main(*args, remote=True)
        self.assertEqual(code, 0, out)

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

    def test_hub_errors_are_reported_by_type(self):
        class FakeApi:
            def file_exists(self, *args, **kwargs):
                raise RuntimeError("response body")

        with patch.object(tool, "hub_api", return_value=FakeApi()):
            with self.assertRaisesRegex(tool.PublishError, r"\(RuntimeError\)$"):
                tool.remote_has_file("o/r", "x/README.md")


class CheckPrTests(unittest.TestCase):
    """--check-pr against a local repository standing in for the Hub."""

    def setUp(self):
        need(self, shutil.which("git-lfs"), "git-lfs is not installed")
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
            code = tool.main(["--check-pr", str(number), "--index", str(self.index)])
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
