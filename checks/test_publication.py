"""Publication-check tests: synthetic data only, no task execution or GPU access."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "publication", HERE / "check_publication.py"
)
pub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)


class PrimitiveTests(unittest.TestCase):
    def test_long_budgets_are_valid(self):
        for value in (86400, 237600, 360000):
            pub.positive_number(value, "timeout")

    def test_invalid_budgets_fail(self):
        for value in (0, -1, True, "86400", float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(pub.CheckError):
                pub.positive_number(value, "timeout")

    def test_signature_task_and_log_detection(self):
        root = "rsi-tasks/signature-tasks/my-task"
        self.assertEqual(pub.task_root(root + "/tests/test.sh", {root}), root)
        self.assertEqual(
            pub.log_root("rsi-logs/signature-tasks/my-task/model/evolve_state.json"),
            "rsi-logs/signature-tasks/my-task",
        )

    def test_unsafe_relative_paths_fail(self):
        for path in (
            "../outside",
            "/absolute",
            "one/../../outside",
            "name\nsecret",
            "one\\two",
        ):
            with self.subTest(path=path), self.assertRaises(pub.CheckError):
                pub.safe_relative(path)

    def test_low_negative_and_unscored_results_are_evidence(self):
        pub.validate_state(
            {
                "best_round": "agent-7",
                "best_score": 0.3,
                "submissions": [
                    {"round": "agent-1", "score": None, "status": "verifier_error"},
                    {"round": "agent-4", "score": -2.7, "status": "rejected"},
                    {"round": "agent-7", "score": 0.3, "status": "completed"},
                ],
            }
        )
        pub.validate_state({"best_score": None, "submissions": []})

    def test_best_score_consistency_not_universal_threshold(self):
        with self.assertRaises(pub.CheckError):
            pub.validate_state(
                {
                    "best_round": "agent-1",
                    "best_score": 1,
                    "submissions": [{"round": "agent-1", "score": 0.3}],
                }
            )

    def test_duplicate_round_rejected(self):
        with self.assertRaises(pub.CheckError):
            pub.validate_state({"submissions": [{"round": "a"}, {"round": "a"}]})

    def test_strict_json_no_duplicate_or_nonfinite(self):
        for text in (
            '{"x":1,"x":2}',
            '{"x":NaN}',
            '{"x":Infinity}',
            '{"x":1e999}',
            "{bad}",
        ):
            with self.subTest(text=text), self.assertRaises(pub.CheckError):
                pub.strict_json(text)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = pub.ScanStage(self.root / "scan")

    def test_gzip_decompressed_without_truncation(self):
        content = b"line\n" * 1000
        self.stage.add(io.BytesIO(gzip.compress(content)), "agent_output.txt.gz")
        self.assertTrue(
            any(
                p.read_bytes() == content
                for p in (self.root / "scan").rglob("agent_output.txt")
            )
        )

    def test_compressed_payload_detected_despite_renaming(self):
        content = b"not a credential\n"
        self.stage.add(io.BytesIO(gzip.compress(content)), "payload.bin")
        self.assertEqual(self.stage.count, 2)

    def test_corrupt_gzip_fails(self):
        with self.assertRaises((OSError, EOFError)):
            self.stage.add(io.BytesIO(gzip.compress(b"hello")[:-4]), "broken.gz")

    def test_tar_traversal_fails(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            item = tarfile.TarInfo("../escape.txt")
            item.size = 1
            archive.addfile(item, io.BytesIO(b"x"))
        with self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(data.getvalue()), "payload.tar")

    def test_tar_symlink_fails(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            item = tarfile.TarInfo("link")
            item.type, item.linkname = tarfile.SYMTYPE, "/outside"
            archive.addfile(item)
        with self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(data.getvalue()), "payload.tar")

    def test_nested_zip_gzip_scanned(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("log.txt.gz", gzip.compress(b"whole transcript"))
        self.stage.add(io.BytesIO(data.getvalue()), "outer.zip")
        self.assertEqual(self.stage.count, 3)

    def test_archive_bomb_fails_instead_of_skipping(self):
        with patch.object(pub, "MAX_MEMBER", 64), self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(gzip.compress(b"x" * 1000)), "bomb.gz")

    def test_unsupported_compression_fails(self):
        with self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(b"Rar!test"), "blob.bin")

    def test_bad_decompressed_json_fails(self):
        with self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(gzip.compress(b'{"x":NaN}')), "result.json.gz")

    def test_invalid_jsonl_fails(self):
        with self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(b'{"x":1}\n{"x":1e999}\n'), "events.jsonl")


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Publication test")
        self.git("config", "user.email", "publication-test@example.invalid")
        (self.repo / "README.md").write_text("Test repository\n")
        self.base = self.commit()

    def git(self, *args):
        return (
            subprocess.check_output(
                ["git", "-C", str(self.repo), *args], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "synthetic fixture")
        return self.git("rev-parse", "HEAD")

    def write(self, path, content):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        return p

    def task(self, slug="demo-task"):
        root = "rsi-tasks/" + slug
        self.write(
            root + "/task.toml",
            'schema_version = "1.4"\n[task]\nname="rsi/demo-task"\ndescription="Synthetic task"\n[agent]\ntimeout_sec=86400\n[environment]\ngpus=0\n',
        )
        self.write(root + "/instruction.md", "Task instructions\n")
        self.write(
            root + "/environment/Dockerfile", "FROM ubuntu:24.04\nWORKDIR /workspace\n"
        )
        self.write(root + "/tests/test.sh", "#!/bin/bash\nexit 0\n")
        return root

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            pub.check(self.repo, self.base, "HEAD", HERE / "publication-policy.toml")

    def test_unrelated_pr_explicitly_passes(self):
        self.write("README.md", "Changed prose\n")
        self.commit()
        self.check()

    def test_new_task_24h_passes(self):
        self.task()
        self.commit()
        self.check()

    def test_nested_signature_task_passes(self):
        self.task("signature-tasks/new-signature")
        self.commit()
        self.check()

    def test_missing_instruction_fails(self):
        root = self.task()
        (self.repo / root / "instruction.md").unlink()
        self.commit()
        with self.assertRaises((pub.CheckError, OSError)):
            self.check()

    def test_task_code_is_not_executed(self):
        root = self.task()
        self.write(
            root + "/tests/evaluate.py", 'raise RuntimeError("must not execute")\n'
        )
        self.write(root + "/tests/test.sh", "touch /definitely-not-to-be-executed\n")
        self.commit()
        self.check()

    def test_new_log_requires_task(self):
        self.write("rsi-logs/missing-task/README.md", "Archived run\n")
        self.commit()
        with self.assertRaises(pub.CheckError):
            self.check()

    def test_failed_only_log_is_valid(self):
        self.task()
        self.write(
            "rsi-logs/demo-task/README.md", "One recorded model, incomplete run\n"
        )
        self.write(
            "rsi-logs/demo-task/model/evolve_state.json",
            json.dumps(
                {
                    "best_score": None,
                    "submissions": [
                        {"round": "agent-12", "score": None, "status": "verifier_error"}
                    ],
                }
            ),
        )
        self.commit()
        self.check()

    def test_deleted_required_file_detected(self):
        root = self.task()
        self.base = self.commit()
        (self.repo / root / "tests/test.sh").unlink()
        self.commit()
        with self.assertRaises(pub.CheckError):
            self.check()

    def test_whole_task_deletion_allowed(self):
        root = self.task()
        self.base = self.commit()
        for p in (self.repo / root).rglob("*"):
            if p.is_file():
                p.unlink()
        self.commit()
        self.check()

    def test_new_task_cannot_claim_historical_exemption(self):
        root = self.task("signature-tasks/gpic_generation")
        (self.repo / root / "tests/test.sh").unlink()
        self.commit()
        with self.assertRaises(pub.CheckError):
            self.check()

    def test_symlink_rejected(self):
        root = self.task()
        (self.repo / root / "link").symlink_to("/etc/passwd")
        self.commit()
        with self.assertRaises(pub.CheckError):
            self.check()

    def manifest(self, payload=b"whole sanitized transcript\n"):
        p = self.write("rsi-logs/demo-task/model/log.txt", payload)
        entry = {
            "path": "model/log.txt",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        m = self.write(
            "rsi-logs/demo-task/publication.json", json.dumps({"files": [entry]})
        )
        return p, m

    def test_file_manifest_checksums_verified(self):
        self.task()
        self.write("rsi-logs/demo-task/README.md", "Recorded run\n")
        self.manifest()
        self.commit()
        self.check()

    def test_changed_file_requires_manifest_update(self):
        self.task()
        self.write("rsi-logs/demo-task/README.md", "Recorded run\n")
        p, _ = self.manifest()
        self.base = self.commit()
        p.write_text("Changed log\n")
        self.commit()
        with self.assertRaises(pub.CheckError):
            self.check()

    def test_missing_manifest_member_fails(self):
        p, m = self.manifest()
        p.unlink()
        doc = pub.load_json(m)
        with self.assertRaises(pub.CheckError):
            pub.verify_manifest(
                m, doc, {m.relative_to(self.repo).as_posix()}, self.repo
            )

    def test_malformed_chunk_manifest_cannot_skip_validation(self):
        manifest = self.repo / "manifest.json"
        for parts in ({}, "not an array", None, []):
            with self.subTest(parts=parts), self.assertRaises(pub.CheckError):
                pub.verify_manifest(manifest, {"parts": parts}, set(), self.repo)

    def test_legacy_chunk_manifest_and_boundary_reassembly(self):
        content = b"first line\nsecond line\n"
        parts = []
        paths = []
        for number, block in enumerate((content[:7], content[7:])):
            p = self.write(f"rsi-logs/demo-task/model/part-{number}.bin", block)
            paths.append(p)
            parts.append(
                {
                    "file": p.name,
                    "size": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                }
            )
        m = self.write(
            "rsi-logs/demo-task/model/manifest.json",
            json.dumps(
                {
                    "source_name": "agent_output.txt",
                    "source_size": len(content),
                    "source_sha256": hashlib.sha256(content).hexdigest(),
                    "parts": parts,
                }
            ),
        )
        tracked = {p.relative_to(self.repo).as_posix() for p in paths}
        pub.verify_manifest(m, pub.load_json(m), tracked, self.repo)
        with contextlib.closing(pub.PartStream(paths)) as stream:
            self.assertEqual(stream.read(100), content)
            self.assertEqual(stream.read(100), b"")

    def test_compressed_manifest_verified(self):
        raw = b"line one\nline two\n"
        packed = gzip.compress(raw, mtime=0)
        p = self.write("rsi-logs/demo-task/agent_output.txt.gz", packed)
        m = self.write(
            "rsi-logs/demo-task/agent_output.manifest.json",
            json.dumps(
                {
                    "artifact": p.name,
                    "compressed_size": len(packed),
                    "compressed_sha256": hashlib.sha256(packed).hexdigest(),
                    "sanitized_uncompressed_size": len(raw),
                    "sanitized_uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
                    "sanitized_uncompressed_line_feeds": 2,
                }
            ),
        )
        pub.verify_manifest(
            m, pub.load_json(m), {p.relative_to(self.repo).as_posix()}, self.repo
        )


class WorkflowTests(unittest.TestCase):
    def test_cpu_only_and_base_policy_enforcement(self):
        text = (
            HERE.parent / ".github/workflows/task-log-publication-checks.yml"
        ).read_text()
        self.assertIn("pull_request:", text)
        self.assertNotIn("pull_request_target:", text)
        self.assertNotIn("secrets.", text)
        self.assertIn("contents: read", text)
        self.assertIn("persist-credentials: false", text)
        self.assertIn("trusted-base/RSI-Harness", text)
        self.assertIn("--frozen --no-dev", text)
        self.assertIn("--gitleaks", text)
        self.assertIn("--compile", text)
        for forbidden in (
            "harbor run",
            "docker build",
            "nvidia-smi",
            "pytest -m gpu",
            "workflow_run:",
        ):
            self.assertNotIn(forbidden, text)

    def test_obsolete_evaluation_workflows_removed(self):
        workflows = HERE.parent / ".github/workflows"
        for name in (
            "auto-trials-on-review-request.yml",
            "run-trials.yml",
            "run-cheat-trials.yml",
            "validate-task.yml",
            "fortify-task.yml",
            "review.yml",
        ):
            self.assertFalse((workflows / name).exists(), name)
        for name in (
            "discussion-review.yml",
            "discussion-task-dispatch.yml",
            "harness-publication-checks.yml",
        ):
            self.assertTrue((workflows / name).exists(), name)


@unittest.skipUnless(
    os.environ.get("PUBLICATION_GITLEAKS_BIN"),
    "set PUBLICATION_GITLEAKS_BIN for scanner integration tests",
)
class SecretScannerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = pub.ScanStage(self.root / "scan")
        self.binary = Path(os.environ["PUBLICATION_GITLEAKS_BIN"])
        # Synthetic, deliberately invalid provider-shaped test value; assembled
        # at runtime so the test source is not itself a token-looking artifact.
        self.fake = ("gh" + "p_") + ("a1b2c3d4e5f6g7h8i9j0" * 2)[:36]

    def scan(self):
        return self.stage.scan(self.binary, HERE / "publication-gitleaks.toml")

    def test_benign_sorting_code_and_hashes_pass(self):
        self.stage.add(
            io.BytesIO(
                b"partner_keys = tl.flip(keys, log2_num_keys - log2_run_keys + step)\n"
            ),
            "code.py",
        )
        self.stage.add(
            io.BytesIO(b'content_key_sequence_sha256: "' + b"a" * 64 + b'"\n'),
            "inventory.txt",
        )
        self.stage.add(
            io.BytesIO(
                b'"added_tokens.json": "8bc892fbf3933cf0c00794b0f0fa638ca22429db58d427f71ea953accabbf1e7",\n'
            ),
            "model_inventory.py",
        )
        self.assertEqual(self.scan(), 0)

    def test_secret_inside_gzip_detected_without_printing_it(self):
        payload = ('github_token = "' + self.fake + '" # gitleaks:allow\n').encode()
        self.stage.add(io.BytesIO(gzip.compress(payload)), "log.txt.gz")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertGreater(self.scan(), 0)
        self.assertIn("::warning::Possible credential:", output.getvalue())
        self.assertIn("advisory only", output.getvalue())
        self.assertNotIn(self.fake, output.getvalue())

    def test_secret_across_chunk_boundary_detected(self):
        payload = ('github_token = "' + self.fake + '"\n').encode()
        paths = []
        for number, block in enumerate((payload[:20], payload[20:])):
            p = self.root / f"part-{number}.bin"
            p.write_bytes(block)
            paths.append(p)
        with contextlib.closing(pub.PartStream(paths)) as stream:
            self.stage.add(stream, "reassembled.txt")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertGreater(self.scan(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
