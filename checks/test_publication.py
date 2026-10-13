"""Publication-check tests: synthetic data only, no task execution or GPU access.

Scanner and compiler integration tests skip when gitleaks
(PUBLICATION_GITLEAKS_BIN) or the Harness are unavailable, unless
PUBLICATION_REQUIRE_INTEGRATION=1 makes a missing dependency a failure, as in CI.
"""

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
import re
import shutil
import struct
import subprocess
import sys
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

WORKFLOW = HERE.parent / ".github/workflows/task-log-publication-checks.yml"
GITLEAKS = os.environ.get("PUBLICATION_GITLEAKS_BIN")
HARNESS = importlib.util.find_spec("rsi_harness") is not None
REQUIRE_INTEGRATION = os.environ.get("PUBLICATION_REQUIRE_INTEGRATION") == "1"


def need(test: unittest.TestCase, available, reason: str):
    if not available:
        if REQUIRE_INTEGRATION:
            test.fail(reason + " (required by PUBLICATION_REQUIRE_INTEGRATION)")
        test.skipTest(reason)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_token() -> str:
    # Synthetic, deliberately invalid provider-shaped test value; assembled at
    # runtime so the test source is not itself a token-looking artifact.
    return ("gh" + "p_") + ("a1b2c3d4e5f6g7h8i9j0" * 2)[:36]


def tar_bytes(entries, compression=""):
    """entries: (name, bytes) for files, (name, None) for directories."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:" + compression) as archive:
        for name, content in entries:
            item = tarfile.TarInfo(name)
            if content is None:
                item.type = tarfile.DIRTYPE
                archive.addfile(item)
            else:
                item.size = len(content)
                archive.addfile(item, io.BytesIO(content))
    return data.getvalue()


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
        nested = {"rsi-tasks/a", "rsi-tasks/a/b"}
        self.assertEqual(
            pub.task_root("rsi-tasks/a/b/task.toml", nested), "rsi-tasks/a/b"
        )
        for loose in ("rsi-logs/run.jsonl", "rsi-logs/signature-tasks/x.json"):
            self.assertIsNone(pub.log_root(loose))

    def test_collection_documentation(self):
        for path in (
            "rsi-logs/SANITIZATION.md",
            "rsi-logs/signature-tasks/README.md",
            "rsi-tasks/README.md",
            "rsi-tasks/signature-tasks/NOTES.MD",
        ):
            self.assertTrue(pub.collection_doc(path), path)
        for path in (
            "rsi-logs/run.jsonl",
            "rsi-logs/README.txt",
            "rsi-tasks/stray/README.md",
            "README.md",
            "docs/rsi-logs.md",
        ):
            self.assertFalse(pub.collection_doc(path), path)

    def test_staged_names_avoid_scanner_path_exemptions(self):
        for name in ("a.bin", "x/node_modules/a.txt", "doc.pdf", "package-lock.json"):
            self.assertEqual(pub.staged_name(name), "member")
        self.assertEqual(pub.staged_name("deploy/Secret.YAML"), "member.yaml")
        self.assertEqual(pub.staged_name("cfg/nuget.config"), "member.nuget.config")
        self.assertEqual(pub.staged_name("cert.p12"), "member.p12")

    def test_unsafe_relative_paths_fail(self):
        for path in (
            "../outside",
            "/absolute",
            "one/../../outside",
            "name\nsecret",
            "one\\two",
            "",
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
        for doc in (
            {
                "best_round": "agent-1",
                "best_score": 1,
                "submissions": [{"round": "agent-1", "score": 0.3}],
            },
            {
                "best_round": "agent-9",
                "best_score": 0.3,
                "submissions": [{"round": "agent-1", "score": 0.3}],
            },
        ):
            with self.subTest(doc=doc), self.assertRaises(pub.CheckError):
                pub.validate_state(doc)

    def test_duplicate_round_rejected(self):
        with self.assertRaises(pub.CheckError):
            pub.validate_state({"submissions": [{"round": "a"}, {"round": "a"}]})

    def test_malformed_evolution_index_rejected(self):
        for doc in (
            [],
            {},
            {"submissions": {}},
            {"submissions": ["agent-1"]},
            {"submissions": [{"round": 1}]},
            {"submissions": [{"round": "a", "score": True}]},
            {"submissions": [{"round": "a", "score": float("inf")}]},
            {"submissions": [{"round": "a", "score": "0.5"}]},
            {"submissions": [], "best_score": float("nan")},
            {"submissions": [], "best_score": False},
        ):
            with self.subTest(doc=doc), self.assertRaises(pub.CheckError):
                pub.validate_state(doc)

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

    def staged(self, label):
        index = next(i for i, value in self.stage.labels.items() if value == label)
        return next((self.stage.destination / index).iterdir()).read_bytes()

    def test_gzip_decompressed_without_truncation(self):
        content = b"line\n" * 1000
        self.stage.add(io.BytesIO(gzip.compress(content)), "agent_output.txt.gz")
        self.assertEqual(self.staged("agent_output.txt.gz!agent_output.txt"), content)

    def test_xz_and_bzip2_decompressed(self):
        content = b"whole transcript\n"
        self.stage.add(io.BytesIO(lzma.compress(content)), "a.log.xz")
        self.stage.add(io.BytesIO(bz2.compress(content)), "b.log.bz2")
        self.assertEqual(self.staged("a.log.xz!a.log"), content)
        self.assertEqual(self.staged("b.log.bz2!b.log"), content)

    def test_compressed_payload_detected_despite_renaming(self):
        content = b"not a credential\n"
        self.stage.add(io.BytesIO(gzip.compress(content)), "payload.bin")
        self.assertEqual(self.stage.count, 2)
        self.assertEqual(self.staged("payload.bin!payload.bin.expanded"), content)

    def test_renamed_xz_and_bzip2_detected(self):
        content = b"not a credential\n"
        self.stage.add(io.BytesIO(lzma.compress(content)), "a.bin")
        self.stage.add(io.BytesIO(bz2.compress(content)), "b.bin")
        self.assertEqual(self.staged("a.bin!a.bin.expanded"), content)
        self.assertEqual(self.staged("b.bin!b.bin.expanded"), content)

    def test_renamed_zip_detected(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("events.jsonl", b'{"x": 1}\n' * 50)
        self.stage.add(io.BytesIO(data.getvalue()), "bundle.bin")
        self.assertEqual(self.staged("bundle.bin!events.jsonl"), b'{"x": 1}\n' * 50)

    def test_tar_inside_renamed_gzip_detected(self):
        packed = gzip.compress(tar_bytes([("inner.txt", b"x\n")]))
        self.stage.add(io.BytesIO(packed), "payload.bin")
        self.assertEqual(
            self.staged("payload.bin!payload.bin.expanded!inner.txt"), b"x\n"
        )

    def test_corrupt_gzip_fails_naming_the_file(self):
        with self.assertRaisesRegex(pub.CheckError, r"^broken\.gz!broken: .*EOFError"):
            self.stage.add(io.BytesIO(gzip.compress(b"hello")[:-4]), "broken.gz")

    def test_tar_traversal_fails(self):
        data = tar_bytes([("../escape.txt", b"x")])
        with self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(data), "payload.tar")

    def test_tar_directory_traversal_fails(self):
        data = tar_bytes([("../escape", None)])
        with self.assertRaisesRegex(pub.CheckError, "unsafe relative path"):
            self.stage.add(io.BytesIO(data), "payload.tar")

    def test_zip_directory_traversal_fails(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("../escape/", b"")
        with self.assertRaisesRegex(pub.CheckError, "unsafe relative path"):
            self.stage.add(io.BytesIO(data.getvalue()), "payload.zip")

    def test_tar_symlink_fails(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            item = tarfile.TarInfo("link")
            item.type, item.linkname = tarfile.SYMTYPE, "/outside"
            archive.addfile(item)
        with self.assertRaisesRegex(pub.CheckError, "payload.tar: .*non-regular"):
            self.stage.add(io.BytesIO(data.getvalue()), "payload.tar")

    def test_tar_devices_and_fifos_fail_naming_the_file(self):
        for kind in (tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE):
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode="w") as archive:
                item = tarfile.TarInfo("dev0")
                item.type = kind
                archive.addfile(item)
            for structured in (True, False):
                with (
                    self.subTest(kind=kind, structured=structured),
                    self.assertRaisesRegex(
                        pub.CheckError,
                        r"^payload\.tar: archive contains a link/device/"
                        r"non-regular entry \(dev0\)",
                    ),
                ):
                    self.stage.add(
                        io.BytesIO(data.getvalue()), "payload.tar", structured=structured
                    )

    def source_tar(self, link="AGENTS.md", kind=tarfile.SYMTYPE):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            regular = tarfile.TarInfo("source/AGENTS.md")
            regular.size = 12
            archive.addfile(regular, io.BytesIO(b"source notes"))
            item = tarfile.TarInfo("source/CLAUDE.md")
            item.type, item.linkname = kind, link
            archive.addfile(item)
        return io.BytesIO(data.getvalue())

    def test_safe_source_link_is_inert_and_target_is_scanned(self):
        self.stage.add(self.source_tar(), "source.tar", structured=False)
        files = list((self.root / "scan").rglob("*"))
        self.assertFalse(any(p.is_symlink() for p in files))
        self.assertEqual(self.staged("source.tar!source/AGENTS.md"), b"source notes")
        self.assertEqual(self.staged("source.tar!source/CLAUDE.md"), b"AGENTS.md")

    def test_dot_prefixed_source_link_accepted(self):
        # `tar -czf src.tar.gz -C dir .` prefixes every member name with "./".
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz") as archive:
            for name in ("./", "./s"):
                item = tarfile.TarInfo(name)
                item.type = tarfile.DIRTYPE
                archive.addfile(item)
            regular = tarfile.TarInfo("./s/AGENTS.md")
            regular.size = 5
            archive.addfile(regular, io.BytesIO(b"notes"))
            item = tarfile.TarInfo("./s/CLAUDE.md")
            item.type, item.linkname = tarfile.SYMTYPE, "AGENTS.md"
            archive.addfile(item)
        self.stage.add(io.BytesIO(data.getvalue()), "src.tar.gz", structured=False)
        self.assertEqual(self.staged("src.tar.gz!./s/AGENTS.md"), b"notes")
        self.assertEqual(self.staged("src.tar.gz!./s/CLAUDE.md"), b"AGENTS.md")

    def test_source_link_may_precede_its_target(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            item = tarfile.TarInfo("CLAUDE.md")
            item.type, item.linkname = tarfile.SYMTYPE, "AGENTS.md"
            archive.addfile(item)
            regular = tarfile.TarInfo("AGENTS.md")
            regular.size = 5
            archive.addfile(regular, io.BytesIO(b"notes"))
        self.stage.add(io.BytesIO(data.getvalue()), "src.tar.gz", structured=False)
        self.assertEqual(self.staged("src.tar.gz!AGENTS.md"), b"notes")

    def test_safe_link_in_log_archive_still_rejected(self):
        with self.assertRaises(pub.CheckError):
            self.stage.add(self.source_tar(), "log.tar")

    def test_unsafe_missing_and_chained_source_links_rejected(self):
        for link in ("/outside", "../../outside", "missing.md", "CLAUDE.md"):
            with self.subTest(link=link), self.assertRaises(pub.CheckError):
                self.stage.add(self.source_tar(link), "source.tar", structured=False)
        with self.assertRaises(pub.CheckError):
            self.stage.add(
                self.source_tar("source/AGENTS.md", tarfile.LNKTYPE),
                "source.tar",
                structured=False,
            )

    def test_tar_directory_entries_are_bounded(self):
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as archive:
            for number in range(5):
                item = tarfile.TarInfo(f"dir-{number}")
                item.type = tarfile.DIRTYPE
                archive.addfile(item)
        with patch.object(pub, "MAX_ENTRIES", 4), self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(data.getvalue()), "source.tar", structured=False)
    def test_oversized_tar_member_fails(self):
        # Compressed, so only the member and not its wrapper exceeds the bound.
        data = tar_bytes([("big.txt", b"x" * 100_000)], "gz")
        with (
            patch.object(pub, "MAX_MEMBER", 50_000),
            self.assertRaisesRegex(
                pub.CheckError, r"archive member exceeds 1 GiB \(big.txt\)"
            ),
        ):
            self.stage.add(io.BytesIO(data), "payload.tar.gz")

    def test_tar_payloads_checked_against_budget_before_extraction(self):
        data = tar_bytes([("a.txt", b"a" * 30_000), ("b.txt", b"b" * 30_000)], "gz")
        with (
            patch.object(pub, "MAX_EXPANDED", len(data) + 50_000),
            self.assertRaisesRegex(pub.CheckError, "budget"),
        ):
            self.stage.add(io.BytesIO(data), "payload.tar.gz")
        # The declared sizes fail the archive before any member is staged.
        self.assertEqual(list(self.stage.labels.values()), ["payload.tar.gz"])

    def test_tar_entry_count_checked_before_members_are_staged(self):
        entries = [("a.txt", b"x")] + [(f"d{n}", None) for n in range(5)]
        with (
            patch.object(pub, "MAX_ENTRIES", 4),
            self.assertRaisesRegex(pub.CheckError, "100000 archive entries"),
        ):
            self.stage.add(io.BytesIO(tar_bytes(entries)), "x.tar")
        self.assertEqual(list(self.stage.labels.values()), ["x.tar"])

    def test_nested_zip_gzip_scanned(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("log.txt.gz", gzip.compress(b"whole transcript"))
        self.stage.add(io.BytesIO(data.getvalue()), "outer.zip")
        self.assertEqual(self.stage.count, 3)
        self.assertEqual(
            self.staged("outer.zip!log.txt.gz!log.txt"), b"whole transcript"
        )

    def zip_with(self, mutate):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            info = zipfile.ZipInfo("entry.txt")
            mutate(info)
            archive.writestr(info, b"payload")
        return data.getvalue()

    def test_encrypted_zip_fails(self):
        raw = bytearray(self.zip_with(lambda info: None))
        # Set the "encrypted" general-purpose flag in the central directory.
        central = raw.index(b"PK\x01\x02")
        (flags,) = struct.unpack_from("<H", raw, central + 8)
        struct.pack_into("<H", raw, central + 8, flags | 1)
        with self.assertRaisesRegex(pub.CheckError, "encrypted or linked"):
            self.stage.add(io.BytesIO(bytes(raw)), "secret.zip")

    def test_zip_symlink_fails(self):
        def symlink(info):
            info.external_attr = 0o120777 << 16

        with self.assertRaisesRegex(pub.CheckError, "encrypted or linked"):
            self.stage.add(io.BytesIO(self.zip_with(symlink)), "links.zip")

    def test_oversized_zip_member_fails(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("big.txt", b"x" * 100_000)
        with (
            patch.object(pub, "MAX_MEMBER", 50_000),
            self.assertRaisesRegex(
                pub.CheckError, r"ZIP member exceeds 1 GiB \(big.txt\)"
            ),
        ):
            self.stage.add(io.BytesIO(data.getvalue()), "big.zip")

    def test_archive_bomb_fails_instead_of_skipping(self):
        with patch.object(pub, "MAX_MEMBER", 64), self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(gzip.compress(b"x" * 1000)), "bomb.gz")

    def test_aggregate_budget_fails(self):
        with patch.object(pub, "MAX_EXPANDED", 1500), self.assertRaises(pub.CheckError):
            for number in range(3):
                self.stage.add(io.BytesIO(b"x" * 600), f"part{number}.txt")

    def test_nesting_depth_bound(self):
        def nested(levels):
            data = b"inner\n"
            for _ in range(levels):
                data = gzip.compress(data)
            return data

        self.stage.add(io.BytesIO(nested(4)), "ok" + ".gz" * 4)
        with self.assertRaisesRegex(pub.CheckError, "four levels"):
            self.stage.add(io.BytesIO(nested(5)), "deep" + ".gz" * 5)

    def test_member_bound_fails(self):
        with patch.object(pub, "MAX_ENTRIES", 3), self.assertRaises(pub.CheckError):
            for number in range(4):
                self.stage.add(io.BytesIO(b"x"), f"f{number}.txt")

    def test_tar_directory_entries_stay_counted_after_their_archive(self):
        # Archive plus three directories: four of five entries remain in use.
        self.stage.add(io.BytesIO(tar_bytes([(f"d{n}", None) for n in range(3)])), "dirs.tar")
        with patch.object(pub, "MAX_ENTRIES", 5):
            self.stage.add(io.BytesIO(b"x"), "fifth.txt")
            with self.assertRaisesRegex(pub.CheckError, "100000 scan members"):
                self.stage.add(io.BytesIO(b"x"), "sixth.txt")

    def test_zip_directory_entries_count_toward_member_bound(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for number in range(10):
                archive.writestr(f"d{number}/", b"")
        # Understate the end record's entry count, so only counting the
        # directory entries themselves enforces the bound.
        raw = bytearray(data.getvalue())
        end = raw.rindex(b"PK\x05\x06")
        struct.pack_into("<HH", raw, end + 8, 1, 1)
        with (
            patch.object(pub, "MAX_ENTRIES", 5),
            self.assertRaisesRegex(pub.CheckError, "100000 scan members"),
        ):
            self.stage.add(io.BytesIO(bytes(raw)), "dirs.zip")

    def test_tar_header_bytes_count_toward_scan_budget(self):
        # 200 directory headers expand to ~100 KiB from a tiny compressed file.
        packed = tar_bytes([(f"d{number}", None) for number in range(200)], "gz")
        self.assertLess(len(packed), 20_000)
        with (
            patch.object(pub, "MAX_EXPANDED", 50_000),
            self.assertRaisesRegex(pub.CheckError, "budget"),
        ):
            self.stage.add(io.BytesIO(packed), "headers.tar.gz")

    def test_tar_header_records_are_bounded_before_reading(self):
        # Ordinary long names use PAX records too and pass. The cap is per
        # member: many headers or a large payload together exceed it and pass.
        long_name = "d/" * 150 + "events.txt"
        entries = [(f"dir{number}", None) for number in range(2100)]
        entries += [(long_name, b"x\n"), ("big.txt", b"y" * (2 * 1024**2))]
        self.stage.add(io.BytesIO(tar_bytes(entries, "gz")), "ok.tgz")
        self.assertEqual(self.staged(f"ok.tgz!{long_name}"), b"x\n")
        self.assertEqual(len(self.staged("ok.tgz!big.txt")), 2 * 1024**2)
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
            item = tarfile.TarInfo("small.txt")
            item.size = 3
            item.pax_headers = {"comment": "x" * (2 * 1024**2)}
            tar.addfile(item, io.BytesIO(b"abc"))
        self.assertLess(len(data.getvalue()), 50_000)
        with self.assertRaisesRegex(
            pub.CheckError, r"^big\.tar\.gz: tar header records exceed 1 MiB"
        ):
            self.stage.add(io.BytesIO(data.getvalue()), "big.tar.gz")

    def test_zip_entry_count_checked_before_its_directory_is_read(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            for number in range(10):
                archive.writestr(f"f{number}.txt", b"x")
        with (
            patch.object(pub, "MAX_ENTRIES", 5),
            patch.object(pub.zipfile, "ZipFile", side_effect=AssertionError),
            self.assertRaisesRegex(pub.CheckError, "100000 scan members"),
        ):
            self.stage.add(io.BytesIO(data.getvalue()), "many.zip")

    def test_unsupported_compression_fails(self):
        # Recognised by signature even when renamed.
        for magic in (b"Rar!", b"7z\xbc\xaf\x27\x1c", b"\x28\xb5\x2f\xfd"):
            with (
                self.subTest(magic=magic),
                self.assertRaisesRegex(pub.CheckError, "unsupported compressed"),
            ):
                self.stage.add(io.BytesIO(magic + b"test"), "blob.bin")

    def test_bad_decompressed_json_fails(self):
        with self.assertRaises(pub.CheckError):
            self.stage.add(io.BytesIO(gzip.compress(b'{"x":NaN}')), "result.json.gz")

    def test_invalid_jsonl_fails(self):
        with self.assertRaisesRegex(
            pub.CheckError, "events.jsonl: invalid JSONL at line 2"
        ):
            self.stage.add(io.BytesIO(b'{"x":1}\n{"x":1e999}\n'), "events.jsonl")

    def test_blank_jsonl_lines_allowed(self):
        self.stage.add(io.BytesIO(b'{"x": 1}\n\n  \n{"x": 2}\n'), "events.jsonl")

    def test_task_data_is_not_held_to_log_record_rules(self):
        self.stage.add(io.BytesIO(b"{not json"), "fixture.json", structured=False)
        self.stage.add(io.BytesIO(b"{not json\n"), "fixture.jsonl", structured=False)


FAKE_SCANNER = r"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
mode = os.environ["FAKE_SCANNER_MODE"]
if mode == "crash":
    sys.exit(2)
if mode == "hang":
    import time
    time.sleep(60)
report = Path(args[args.index("--report-path") + 1])
findings = []
if mode in ("findings", "crash-with-report"):
    for path in sorted(Path(args[1]).rglob("*")):
        if path.is_file():
            # Duplicate each finding, as a scanner does for repeated matches.
            findings += [{"File": str(path), "RuleID": "fake-rule", "StartLine": 3}] * 2
if mode != "no-report":
    report.write_text(json.dumps(findings))
if mode == "crash-with-report":
    sys.exit(2)
sys.exit(1 if findings or mode == "empty-failure" else 0)
"""


class ScannerReportTests(unittest.TestCase):
    """Scanner result handling, with a stand-in for the scanner binary."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = pub.ScanStage(self.root / "scan")
        self.binary = self.root / "fake-gitleaks"
        self.binary.write_text(f"#!{sys.executable} -I\n{FAKE_SCANNER}")
        self.binary.chmod(0o755)

    def scan(self, mode):
        output = io.StringIO()
        with (
            patch.dict(os.environ, FAKE_SCANNER_MODE=mode),
            contextlib.redirect_stdout(output),
        ):
            count = self.stage.scan(self.binary, HERE / "publication-gitleaks.toml")
        return count, output.getvalue()

    def test_scanner_failures_are_errors(self):
        self.stage.add(io.BytesIO(b"text\n"), "a.txt")
        for mode in ("crash", "crash-with-report", "no-report", "empty-failure"):
            with self.subTest(mode=mode), self.assertRaises(pub.CheckError):
                self.scan(mode)

    def test_hung_scanner_fails(self):
        self.stage.add(io.BytesIO(b"text\n"), "a.txt")
        with (
            patch.object(pub, "SCAN_TIMEOUT", 1),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            self.scan("hang")

    def test_clean_scan(self):
        self.stage.add(io.BytesIO(b"text\n"), "a.txt")
        self.assertEqual(self.scan("clean"), (0, ""))

    def test_every_warning_printed_once_with_published_path(self):
        for number in range(40):
            self.stage.add(io.BytesIO(b"text\n"), f"rsi-logs/demo/log{number:02d}.txt")
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, "w") as archive:
            archive.writestr("cfg/a.yaml", b"key: value\n")
        self.stage.add(io.BytesIO(zipped.getvalue()), "rsi-logs/demo/bundle.zip")
        count, output = self.scan("findings")
        warnings = [
            line for line in output.splitlines() if line.startswith("::warning::")
        ]
        self.assertEqual(count, 42)
        self.assertEqual(len(warnings), 42)
        self.assertIn("NOTICE 42 possible credentials", output)
        self.assertIn(
            "::warning::Possible credential: rsi-logs/demo/log39.txt; rule=fake-rule, line=3",
            warnings,
        )
        self.assertIn(
            "::warning::Possible credential: rsi-logs/demo/bundle.zip!cfg/a.yaml; "
            "rule=fake-rule, line=3",
            warnings,
        )

    def test_warning_text_cannot_inject_workflow_commands(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pub.annotate("warning", "a\n::error::b%")
        self.assertEqual(output.getvalue(), "::warning::a%0A::error::b%25\n")


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
        self.git("commit", "-qm", "synthetic fixture", "--allow-empty")
        return self.git("rev-parse", "HEAD")

    def write(self, path, content):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        return p

    def task(self, slug="demo-task", toml=None):
        root = "rsi-tasks/" + slug
        self.write(
            root + "/task.toml",
            toml
            or 'schema_version = "1.4"\n[task]\nname="rsi/demo-task"\ndescription="Synthetic task"\n[agent]\ntimeout_sec=86400\n[environment]\ngpus=0\n',
        )
        self.write(root + "/instruction.md", "Task instructions\n")
        self.write(
            root + "/environment/Dockerfile", "FROM ubuntu:24.04\nWORKDIR /workspace\n"
        )
        self.write(root + "/tests/test.sh", "#!/bin/bash\nexit 0\n")
        return root

    def log(self, slug="demo-task"):
        root = "rsi-logs/" + slug
        self.write(root + "/README.md", "Recorded run\n")
        return root

    def check(self, **options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pub.check(
                self.repo,
                self.base,
                "HEAD",
                HERE / "publication-policy.toml",
                **options,
            )
        return output.getvalue()

    def assertFails(self, pattern="", **options):
        with self.assertRaisesRegex(pub.CheckError, pattern):
            self.check(**options)

    def split(self, path, chunk_size):
        """Lossless chunks in the task runner's format (parts plus manifest.json)."""
        data = (self.repo / path).read_bytes()
        (self.repo / path).unlink()
        directory = path + ".parts"
        parts = []
        for number, start in enumerate(range(0, len(data), chunk_size), 1):
            block = data[start : start + chunk_size]
            name = f"part-{number:06d}.bin"
            self.write(f"{directory}/{name}", block)
            parts.append({"file": name, "size": len(block), "sha256": sha(block)})
        self.write(directory + "/.gitattributes", "part-*.bin -text -filter\n")
        manifest = {
            "version": 1,
            "source_name": Path(path).name,
            "source_size": len(data),
            "source_sha256": sha(data),
            "chunk_size": chunk_size,
            "parts": parts,
        }
        self.write(directory + "/manifest.json", json.dumps(manifest, indent=2))
        return directory, manifest

    # Repository-level rules

    def test_unrelated_pr_explicitly_passes(self):
        self.write("README.md", "Changed prose\n")
        self.commit()
        self.assertIn("not applicable", self.check())

    def test_unsafe_changed_path_fails(self):
        self.write("notes\nname.md", "x\n")
        self.commit()
        self.assertFails("unsafe relative path")

    def test_checkout_must_match_head(self):
        self.task()
        self.commit()
        with self.assertRaisesRegex(pub.CheckError, "checkout must match"):
            with contextlib.redirect_stdout(io.StringIO()):
                pub.check(
                    self.repo, self.base, self.base, HERE / "publication-policy.toml"
                )

    def test_diff_starts_at_the_merge_base(self):
        root = self.task()
        branch_point = self.commit()
        self.write(root + "/instruction.md", "Clarified instructions\n")
        head = self.commit()
        # The base branch moves on after the PR branched, with a package the PR
        # never touched; it must be neither checked nor reported for the PR.
        self.git("checkout", "-q", branch_point)
        other = self.task("other-task")
        self.write(other + "/instruction.md", "")
        self.base = self.commit()
        self.git("checkout", "-q", head)
        output = self.check()
        self.assertIn(f"PASS {root}: task configuration", output)
        self.assertNotIn(other, output)

    def test_full_audit_checks_unchanged_packages(self):
        root = self.task()
        (self.repo / root / "instruction.md").write_text("")
        self.base = self.commit()
        self.assertIn("not applicable", self.check())
        self.assertFails("demo-task: empty instruction.md", full=True)

    def test_symlink_rejected(self):
        root = self.task()
        (self.repo / root / "link").symlink_to("/etc/passwd")
        self.commit()
        self.assertFails("link: publication must use regular files")

    def test_stored_file_size_limit(self):
        root = self.task()
        self.write(root + "/environment/data.bin", b"x" * 999)
        self.commit()
        with patch.object(pub, "MAX_STORED", 1000):
            self.check()
        self.write(root + "/environment/data.bin", b"x" * 1000)
        self.commit()
        with patch.object(pub, "MAX_STORED", 1000):
            self.assertFails("data.bin: stored file must be below")
        (self.repo / root / "environment/data.bin").unlink()
        self.write("rsi-logs/NOTES.md", b"x" * 1000)
        self.commit()
        with patch.object(pub, "MAX_STORED", 1000):
            self.assertFails("^rsi-logs/NOTES.md: stored file must be below")

    # Task rules

    def test_new_task_24h_passes(self):
        root = self.task()
        self.commit()
        self.assertIn(f"PASS {root}: task configuration\n", self.check())

    def test_nested_signature_task_passes(self):
        self.task("signature-tasks/new-signature")
        self.commit()
        self.check()

    def test_missing_instruction_fails(self):
        root = self.task()
        (self.repo / root / "instruction.md").unlink()
        self.commit()
        self.assertFails(f"^{root}: missing instruction.md$")

    def test_empty_instruction_fails(self):
        root = self.task()
        self.write(root + "/instruction.md", " \n")
        self.commit()
        self.assertFails(f"^{root}: empty instruction.md$")

    def test_removed_task_toml_gives_clear_error(self):
        root = self.task()
        self.base = self.commit()
        (self.repo / root / "task.toml").unlink()
        self.commit()
        self.assertFails(f"^{root}: missing task.toml")

    def test_invalid_toml_names_task(self):
        root = self.task()
        self.write(root + "/task.toml", "schema_version = \n")
        self.commit()
        self.assertFails(f"^{root}: task.toml: invalid TOML")

    def test_task_metadata_rules(self):
        good = 'schema_version = "1.4"\n[task]\nname="n"\ndescription="d"\n'
        cases = {
            "agent.timeout_sec": good + "[agent]\ntimeout_sec=0\n",
            "verifier.timeout_sec": good + "[verifier]\ntimeout_sec=-5\n",
            "build_timeout_sec": good + "[environment]\nbuild_timeout_sec=0\n",
            "GPU count": good + "[environment]\ngpus=1.5\n",
            "GPU count ": good + "[environment]\ngpus=true\n",
            "GPU count  ": good + "[metadata.rsi_harness.verifier]\ngpus=-1\n",
            "task.name": 'schema_version = "1.4"\n[task]\nname=" "\ndescription="d"\n',
            "task.description": 'schema_version = "1.4"\n[task]\nname="n"\n',
            "schema 1.4": '[task]\nname="n"\ndescription="d"\n',
        }
        for message, toml in cases.items():
            with self.subTest(message=message):
                self.task(toml=toml)
                self.commit()
                self.assertFails(message.strip())

    def test_runnable_task_requires_verifier_and_environment(self):
        root = self.task()
        (self.repo / root / "tests/test.sh").unlink()
        self.commit()
        self.assertFails("missing tests/test.sh")
        self.write(root + "/tests/test.sh", "exit 0\n")
        (self.repo / root / "environment/Dockerfile").unlink()
        self.commit()
        self.assertFails("missing environment build/image declaration")

    def test_environment_alternatives_pass(self):
        root = self.task()
        (self.repo / root / "environment/Dockerfile").unlink()
        self.write(root + "/environment/docker-compose.yaml", "services: {}\n")
        self.commit()
        self.check()
        (self.repo / root / "environment/docker-compose.yaml").unlink()
        toml = (self.repo / root / "task.toml").read_text()
        self.write(root + "/task.toml", toml + 'docker_image = "ubuntu:24.04"\n')
        self.commit()
        self.check()

    def test_task_code_is_not_executed(self):
        root = self.task()
        marker = self.repo / "executed"
        self.write(
            root + "/tests/evaluate.py",
            f"open({str(marker)!r}, 'w').close()\nraise RuntimeError('must not execute')\n",
        )
        self.write(root + "/tests/test.sh", f"touch {marker}\n")
        self.commit()
        self.check()
        self.assertFalse(marker.exists())

    def test_python_run_by_harness_must_parse(self):
        for directory in ("tests", "solution"):
            with self.subTest(directory=directory):
                root = self.task()
                self.write(f"{root}/{directory}/broken.py", "def broken(:\n")
                self.commit()
                self.assertFails(f"{directory}/broken.py: invalid Python syntax")
                (self.repo / root / directory / "broken.py").unlink()
                self.commit()

    def test_other_task_python_syntax_is_a_warning(self):
        root = self.task()
        self.write(root + "/environment/starter/buggy.py", "def broken(:\n")
        self.commit()
        output = self.check()
        self.assertIn("::warning::" + root + "/environment/starter/buggy.py", output)
        self.assertIn("not blocking", output)

    def test_python_too_deep_for_the_parser_follows_syntax_rules(self):
        # CPython raises RecursionError, not SyntaxError, on long expression chains.
        source = "x = " + " + ".join(["1"] * 20000) + "\n"
        root = self.task()
        self.write(root + "/environment/gen_table.py", source)
        self.commit()
        output = self.check()
        self.assertIn("::warning::" + root + "/environment/gen_table.py", output)
        self.write(root + "/tests/gen_table.py", source)
        self.commit()
        self.assertFails("tests/gen_table.py: invalid Python syntax")

    def test_python_coding_declaration_honoured(self):
        root = self.task()
        self.write(
            root + "/tests/latin.py",
            "# -*- coding: latin-1 -*-\nname = 'caf\xe9'\n".encode("latin-1"),
        )
        self.commit()
        self.assertNotIn("::warning::", self.check())

    def test_unchanged_task_python_is_not_rechecked(self):
        root = self.task()
        self.write(root + "/tests/old.py", "print 'python 2'\n")
        self.base = self.commit()
        self.write(root + "/instruction.md", "Clarified instructions\n")
        self.commit()
        self.assertIn(f"PASS {root}: task configuration", self.check())

    def test_log_python_is_not_syntax_checked(self):
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/m/agent_attempt.py", "def broken(:\n")
        self.commit()
        self.assertNotIn("Python", self.check())

    def test_unreadable_task_file_names_task(self):
        root = self.task()
        self.write(root + "/instruction.md", b"\xff\xfe not UTF-8\n")
        self.commit()
        self.assertFails(f"^{root}: unreadable task file \\(UnicodeDecodeError\\)$")

    def test_whole_task_deletion_allowed(self):
        root = self.task()
        self.base = self.commit()
        for p in (self.repo / root).rglob("*"):
            if p.is_file():
                p.unlink()
        self.commit()
        self.assertIn(f"PASS {root}: task removed", self.check())

    def test_deleted_required_file_detected(self):
        root = self.task()
        self.base = self.commit()
        (self.repo / root / "tests/test.sh").unlink()
        self.commit()
        self.assertFails("missing tests/test.sh")

    def test_required_file_moved_out_of_task_detected(self):
        root = self.task()
        self.base = self.commit()
        (self.repo / "docs").mkdir()
        self.git("mv", root + "/tests/test.sh", "docs/test.sh")
        self.commit()
        self.assertFails(f"^{root}: missing tests/test.sh$")

    def test_existing_historical_archive_keeps_exemption(self):
        root = self.task(
            "signature-tasks/gpic_generation",
            toml='schema_version = "1.3"\n[task]\nname="g"\ndescription="d"\n',
        )
        (self.repo / root / "tests/test.sh").unlink()
        self.base = self.commit()
        self.write(root + "/instruction.md", "Clarified instructions\n")
        self.commit()
        self.assertIn("historical archive", self.check())

    def test_new_task_cannot_claim_historical_exemption(self):
        root = self.task("signature-tasks/gpic_generation")
        (self.repo / root / "tests/test.sh").unlink()
        self.commit()
        self.assertFails("missing tests/test.sh")

    def test_task_files_need_enclosing_task(self):
        for path in ("rsi-tasks/stray/data.csv", "rsi-tasks/stray/README.md"):
            with self.subTest(path=path):
                self.write(path, "x\n")
                self.commit()
                self.assertFails("no enclosing task.toml")
                (self.repo / path).unlink()
                self.commit()

    def test_task_data_files_are_not_log_records(self):
        root = self.task()
        self.write(root + "/environment/fixture.jsonl", "{not a record\n")
        self.commit()
        self.check()

    def test_task_owned_data_manifest_is_not_a_publication_manifest(self):
        root = self.task()
        self.write(
            root + "/tests/assets/manifest.json",
            json.dumps({"files": {"data.zip": "0" * 64}}),
        )
        self.write(
            root + "/tests/assets/upstream-manifest.json",
            json.dumps({"files": {"x": {"sha256": "1" * 64}}, "parent_files": {}}),
        )
        # Not the runner's exact chunk format, so not a chunk manifest either.
        self.write(
            root + "/tests/assets/shards/manifest.json",
            json.dumps({"parts": ["shard-1.npy"], "dtype": "float32"}),
        )
        self.commit()
        self.check(full=True)
        # Task data need not be strict JSON; an unchanged file is not reparsed.
        self.write(root + "/tests/assets/broken/manifest.json", "{not JSON")
        self.base = self.commit()
        self.write(root + "/instruction.md", "Clarified instructions\n")
        self.commit()
        self.check()

    def test_split_task_data_is_reassembled(self):
        root = self.task()
        fixture = b"{deliberately not JSON\n" + b"".join(
            sha(b"%d" % n).encode() + b"\n" for n in range(200)
        )
        self.write(
            root + "/tests/assets/data.tar.gz",
            tar_bytes([("data", None), ("data/fixture.jsonl", fixture)], "gz"),
        )
        directory, manifest = self.split(root + "/tests/assets/data.tar.gz", 1024)
        self.assertGreater(len(manifest["parts"]), 1)
        self.commit()
        # A chunk on its own is a truncated gzip stream and would fail; the
        # fixture inside is task data, not a log record.
        self.assertIn(f"PASS {root}: task configuration", self.check())

    # Collection-level files

    def test_loose_log_data_files_fail(self):
        self.task()
        for path in ("rsi-logs/run.jsonl", "rsi-logs/signature-tasks/x.json"):
            with self.subTest(path=path):
                self.write(path, '{"x": 1}\n')
                self.commit()
                self.assertFails(f"^{path}: log files must be inside")
                (self.repo / path).unlink()
                self.commit()

    def test_collection_documentation_passes_and_is_scanned(self):
        self.write("rsi-logs/NOTES.md", "Notes\n")
        self.write("rsi-logs/signature-tasks/README.md", "Signature logs\n")
        self.write("rsi-tasks/README.md", "Tasks\n")
        self.commit()
        self.assertIn("PASS 3 published files", self.check())

    def test_removing_a_loose_file_passes(self):
        self.write("rsi-logs/run.jsonl", "{}\n")
        self.base = self.commit()
        (self.repo / "rsi-logs/run.jsonl").unlink()
        self.commit()
        self.assertIn("not applicable", self.check())

    # Log rules

    def test_new_log_requires_task(self):
        self.write("rsi-logs/missing-task/README.md", "Archived run\n")
        self.commit()
        self.assertFails("rsi-logs/missing-task: corresponding public task.toml")

    def test_historical_log_alias(self):
        self.task("signature-tasks/gpic_generation")
        self.log("signature-tasks/gpic-10m-autoresearch")
        self.commit()
        self.check()

    def test_new_log_requires_readme(self):
        self.task()
        self.write("rsi-logs/demo-task/model/agent_output.txt", "output\n")
        self.commit()
        self.assertFails("rsi-logs/demo-task: new log archive needs README.md")

    def test_existing_log_without_readme_can_be_extended(self):
        self.task()
        self.write("rsi-logs/demo-task/model/agent_output.txt", "output\n")
        self.base = self.commit()
        self.write("rsi-logs/demo-task/model2/agent_output.txt", "output\n")
        self.commit()
        self.check()

    def test_log_removal_passes(self):
        self.task()
        self.log()
        self.base = self.commit()
        (self.repo / "rsi-logs/demo-task/README.md").unlink()
        self.commit()
        self.assertIn("PASS rsi-logs/demo-task: log archive removed", self.check())

    def test_failed_only_log_is_valid(self):
        self.task()
        self.log()
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

    def test_changed_evolution_index_validated(self):
        self.task()
        self.log()
        for name in ("evolve_state.json", "e_state.json"):
            with self.subTest(name=name):
                path = "rsi-logs/demo-task/model/" + name
                self.write(path, json.dumps({"submissions": [{"round": "a"}] * 2}))
                self.commit()
                self.assertFails(f"^{path}: duplicate submission round")
                (self.repo / path).unlink()
                self.commit()

    def test_unchanged_evolution_index_not_coerced(self):
        self.task()
        self.log()
        self.write(
            "rsi-logs/demo-task/model/evolve_state.json",
            json.dumps({"submissions": [{"round": "a"}] * 2}),
        )
        self.base = self.commit()
        self.write("rsi-logs/demo-task/README.md", "Clarified README\n")
        self.commit()
        self.check()

    def test_invalid_log_json_names_file(self):
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/model/result.json", '{"x": NaN}')
        self.commit()
        self.assertFails("^rsi-logs/demo-task/model/result.json: ")

    def test_invalid_log_jsonl_names_file(self):
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/model/events.jsonl", '{"x": 1}\n{bad\n')
        self.commit()
        self.assertFails("events.jsonl: invalid JSONL at line 2")

    # Log manifests

    def manifest(self, payload=b"whole sanitized transcript\n", **extra):
        p = self.write("rsi-logs/demo-task/model/log.txt", payload)
        entry = {
            "path": "model/log.txt",
            "bytes": len(payload),
            "sha256": sha(payload),
            **extra,
        }
        m = self.write(
            "rsi-logs/demo-task/publication.json", json.dumps({"files": [entry]})
        )
        return p, m

    def test_file_manifest_checksums_verified(self):
        self.task()
        self.log()
        self.manifest()
        self.commit()
        self.check()

    def test_changed_file_requires_manifest_update(self):
        self.task()
        self.log()
        p, _ = self.manifest()
        self.base = self.commit()
        p.write_text("Changed log\n")
        self.commit()
        self.assertFails(
            "^rsi-logs/demo-task/publication.json: model/log.txt: manifest SHA-256 mismatch"
        )

    def test_manifest_entry_rules(self):
        payload = b"one\ntwo\n"
        cases = {
            "byte count": {"bytes": 1},
            "uncompressed hash/size": {"uncompressed_sha256": "0" * 64},
            "line_count": {"uncompressed_sha256": sha(payload), "line_count": 7},
        }
        self.task()
        self.log()
        for message, extra in cases.items():
            with self.subTest(message=message):
                self.manifest(payload)
                m = self.repo / "rsi-logs/demo-task/publication.json"
                doc = json.loads(m.read_text())
                doc["files"][0].update(extra)
                m.write_text(json.dumps(doc))
                self.commit()
                self.assertFails(message)

    def test_gzip_manifest_entry_verified(self):
        raw = b"line one\nline two\n"
        packed = gzip.compress(raw, mtime=0)
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/model/out.txt.gz", packed)
        entry = {
            "path": "model/out.txt.gz",
            "sha256": sha(packed),
            "compression": "gzip",
            "uncompressed_sha256": sha(raw),
            "uncompressed_bytes": len(raw),
            "line_count": 2,
        }
        manifest = "rsi-logs/demo-task/publication.json"
        for field, wrong in (
            (None, None),
            ("uncompressed_sha256", "0" * 64),
            ("uncompressed_bytes", 1),
            ("line_count", 9),
        ):
            with self.subTest(field=field):
                self.write(
                    manifest,
                    json.dumps(
                        {"files": [dict(entry, **({field: wrong} if field else {}))]}
                    ),
                )
                self.commit()
                if field:
                    self.assertFails(f"manifest {field} mismatch")
                else:
                    self.check()

    def test_manifest_structure_rules(self):
        self.task()
        self.log()
        p, m = self.manifest()
        entry = json.loads(m.read_text())["files"][0]
        for message, doc in {
            "duplicate manifest file path": {"files": [entry, entry]},
            "require path and sha256": {"files": [{"path": "model/log.txt"}]},
            "missing/untracked": {"files": [dict(entry, path="model/absent.txt")]},
            "must be an object": [entry],
        }.items():
            with self.subTest(message=message):
                m.write_text(json.dumps(doc))
                self.commit()
                self.assertFails(message)

    def test_manifest_gzip_expansion_is_bounded(self):
        raw = b"x" * 5000
        packed = gzip.compress(raw)
        p = self.write("rsi-logs/demo-task/model/out.txt.gz", packed)
        m = self.write(
            "rsi-logs/demo-task/publication.json",
            json.dumps(
                {
                    "files": [
                        {
                            "path": "model/out.txt.gz",
                            "sha256": sha(packed),
                            "compression": "gzip",
                        }
                    ]
                }
            ),
        )
        tracked = {p.relative_to(self.repo).as_posix()}
        with (
            patch.object(pub, "MAX_MEMBER", 1000),
            self.assertRaisesRegex(pub.CheckError, "expanded member exceeds"),
        ):
            pub.verify_manifest(m, pub.load_json(m), tracked, self.repo)

    def test_untracked_manifest_member_fails(self):
        p, m = self.manifest()
        with self.assertRaisesRegex(pub.CheckError, "missing/untracked"):
            pub.verify_manifest(m, pub.load_json(m), set(), self.repo)

    def test_missing_manifest_member_fails(self):
        p, m = self.manifest()
        p.unlink()
        doc = pub.load_json(m)
        with self.assertRaises(pub.CheckError):
            pub.verify_manifest(
                m, doc, {m.relative_to(self.repo).as_posix()}, self.repo
            )

    def test_compressed_artifact_manifest_verified(self):
        raw = b"line one\nline two\n"
        packed = gzip.compress(raw, mtime=0)
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/agent_output.txt.gz", packed)
        doc = {
            "artifact": "agent_output.txt.gz",
            "compressed_size": len(packed),
            "compressed_sha256": sha(packed),
            "sanitized_uncompressed_size": len(raw),
            "sanitized_uncompressed_sha256": sha(raw),
            "sanitized_uncompressed_line_feeds": 2,
        }
        manifest = "rsi-logs/demo-task/agent_output.manifest.json"
        self.write(manifest, json.dumps(doc))
        self.commit()
        self.check()
        self.write(manifest, json.dumps(dict(doc, sanitized_uncompressed_line_feeds=3)))
        self.commit()
        self.assertFails(
            f"^{manifest}: agent_output.txt.gz: manifest line_count mismatch"
        )

    def test_malformed_manifests_fail_naming_the_manifest(self):
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/m/a.gz", gzip.compress(b"x\n"))
        manifest = "rsi-logs/demo-task/m/manifest.json"
        for message, doc in {
            "compressed artifact manifest lacks compressed_size": {
                "artifact": "a.gz",
                "compressed_sha256": "0" * 64,
            },
            "chunk manifest parts require a path": {"parts": ["part-1.bin"]},
            "chunk manifest parts require a path ": {"parts": [{"path": [1]}]},
            "manifest files require path and sha256": {
                "files": [{"path": [1], "sha256": ""}]
            },
        }.items():
            with self.subTest(message=message):
                self.write(manifest, json.dumps(doc))
                self.commit()
                self.assertFails(f"^{manifest}: {message.strip()}")

    def test_malformed_chunk_manifest_cannot_skip_validation(self):
        manifest = self.repo / "manifest.json"
        for parts in ({}, "not an array", None, []):
            with (
                self.subTest(parts=parts),
                self.assertRaisesRegex(pub.CheckError, "non-empty array"),
            ):
                pub.verify_manifest(manifest, {"parts": parts}, set(), self.repo)

    def test_legacy_chunk_manifest_and_boundary_reassembly(self):
        content = b"first line\nsecond line\n"
        parts = []
        paths = []
        for number, block in enumerate((content[:7], content[7:])):
            p = self.write(f"rsi-logs/demo-task/model/part-{number}.bin", block)
            paths.append(p)
            parts.append({"file": p.name, "size": len(block), "sha256": sha(block)})
        m = self.write(
            "rsi-logs/demo-task/model/manifest.json",
            json.dumps(
                {
                    "source_name": "agent_output.txt",
                    "source_size": len(content),
                    "source_sha256": sha(content),
                    "parts": parts,
                }
            ),
        )
        tracked = {p.relative_to(self.repo).as_posix() for p in paths}
        pub.verify_manifest(m, pub.load_json(m), tracked, self.repo)
        with contextlib.closing(pub.PartStream(paths)) as stream:
            self.assertEqual(stream.read(100), content)
            self.assertEqual(stream.read(100), b"")

    def test_gzip_chunks_validated_only_after_complete_reassembly(self):
        self.task()
        self.write("rsi-logs/demo-task/README.md", "Recorded run\n")
        content = b"complete sanitized transcript\n" * 100
        compressed = gzip.compress(content, mtime=0)
        parts = []
        for number, block in enumerate((compressed[:12], compressed[12:])):
            name = f"part-{number}.bin"
            self.write(f"rsi-logs/demo-task/model/{name}", block)
            parts.append(
                {
                    "path": name,
                    "bytes": len(block),
                    "sha256": hashlib.sha256(block).hexdigest(),
                }
            )
        self.write(
            "rsi-logs/demo-task/model/manifest.json",
            json.dumps(
                {
                    "source_name": "agent_output.txt.gz",
                    "bytes": len(compressed),
                    "sha256": hashlib.sha256(compressed).hexdigest(),
                    "parts": parts,
                }
            ),
        )
        self.commit()
        self.check()

    def test_current_chunk_manifest_format(self):
        content = b"first line\nsecond line\n"
        p = self.write("rsi-logs/demo-task/model/part-1.bin", content)
        m = self.write(
            "rsi-logs/demo-task/model/manifest.json",
            json.dumps(
                {
                    "sha256": sha(content),
                    "bytes": len(content),
                    "parts": [
                        {"path": p.name, "bytes": len(content), "sha256": sha(content)}
                    ],
                }
            ),
        )
        tracked = {p.relative_to(self.repo).as_posix()}
        self.assertEqual(
            len(pub.verify_manifest(m, pub.load_json(m), tracked, self.repo)), 1
        )

    def test_chunk_checksum_mismatches_fail(self):
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/m/agent_output.txt", b"x" * 300)
        directory, manifest = self.split("rsi-logs/demo-task/m/agent_output.txt", 100)
        for message, doc in {
            "whole-stream checksum/size": dict(manifest, source_sha256="0" * 64),
            "whole-stream checksum/size ": dict(manifest, source_size=299),
            "part-000002.bin: manifest SHA-256": dict(
                manifest,
                parts=[
                    manifest["parts"][0],
                    dict(manifest["parts"][1], sha256="0" * 64),
                    manifest["parts"][2],
                ],
            ),
            "duplicate chunk path": dict(manifest, parts=[manifest["parts"][0]] * 2),
        }.items():
            with self.subTest(message=message):
                self.write(directory + "/manifest.json", json.dumps(doc))
                self.commit()
                self.assertFails(f"{directory}/manifest.json: .*{message.strip()}")

    def test_split_compressed_jsonl_log_is_reassembled_before_decompression(self):
        self.task()
        self.log()
        lines = b"".join(
            json.dumps({"step": n, "text": sha(str(n).encode())}).encode() + b"\n"
            for n in range(200)
        )
        self.write("rsi-logs/demo-task/m/trajectory.jsonl.gz", gzip.compress(lines))
        directory, manifest = self.split(
            "rsi-logs/demo-task/m/trajectory.jsonl.gz", 4096
        )
        self.assertGreater(len(manifest["parts"]), 1)
        self.assertTrue(
            (self.repo / directory / "part-000001.bin")
            .read_bytes()
            .startswith(b"\x1f\x8b")
        )
        self.commit()
        output = self.check()
        self.assertIn("PASS", output.splitlines()[-1])

    def test_split_tar_gz_log_is_reassembled_before_extraction(self):
        self.task()
        self.log()
        records = b"".join(
            b'{"n": %d, "pad": "%s"}\n' % (n, sha(b"%d" % n).encode())
            for n in range(300)
        )
        self.write(
            "rsi-logs/demo-task/m/run.tar.gz",
            tar_bytes([("run", None), ("run/events.jsonl", records)], "gz"),
        )
        directory, manifest = self.split("rsi-logs/demo-task/m/run.tar.gz", 4096)
        self.assertGreater(len(manifest["parts"]), 1)
        self.commit()
        self.check()

    def test_reassembled_stream_contents_are_validated(self):
        self.task()
        self.log()
        lines = b'{"ok": 1}\n' * 2000 + b"{bad\n"
        self.write("rsi-logs/demo-task/m/trajectory.jsonl.gz", gzip.compress(lines))
        directory, _ = self.split("rsi-logs/demo-task/m/trajectory.jsonl.gz", 64)
        self.commit()
        self.assertFails(
            f"^{directory}/manifest.json!trajectory.jsonl.gz!trajectory.jsonl: "
            "invalid JSONL at line 2001"
        )

    def test_unsafe_chunk_source_name_fails(self):
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/m/out.txt", b"x" * 10)
        directory, manifest = self.split("rsi-logs/demo-task/m/out.txt", 5)
        for source, message in (
            ("../x", "unsafe relative path"),
            (7, "must be a string"),
        ):
            with self.subTest(source=source):
                self.write(
                    directory + "/manifest.json",
                    json.dumps(dict(manifest, source_name=source)),
                )
                self.commit()
                self.assertFails(f"^{directory}/manifest.json: .*{message}")

    # Manifest v2 and full audits in the Index tree

    def test_v2_manifest_verified_in_the_index_tree(self):
        raw = b"a\nb\nc\n"
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/m/agent_output.txt", raw)
        manifest = "rsi-logs/demo-task/m/agent_output.manifest.json"
        self.write(manifest, json.dumps(v2_manifest(raw)))
        self.commit()
        self.check()
        self.write(manifest, json.dumps(v2_manifest(raw, line_feeds=2)))
        self.commit()
        self.assertFails(f"^{manifest}: agent_output.txt: manifest line_feeds mismatch")

    def test_index_mode_keeps_compressed_and_chunked_logs(self):
        self.task()
        self.log()
        self.write("rsi-logs/demo-task/m/out.txt.gz", gzip.compress(b"x\n"))
        self.commit()
        self.check()

    def test_full_audit_without_base(self):
        self.task()
        self.commit()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pub.check(self.repo, None, "HEAD", HERE / "publication-policy.toml", full=True)
        self.assertIn("PASS rsi-tasks/demo-task: task configuration", output.getvalue())

    def test_missing_base_names_the_remedy(self):
        self.task()
        self.commit()
        with self.assertRaisesRegex(pub.CheckError, "audit the head with --full"):
            with contextlib.redirect_stdout(io.StringIO()):
                pub.check(self.repo, "f" * 40, "HEAD", HERE / "publication-policy.toml")

    # CI path: compiler and scanner integration

    def test_harness_compilation(self):
        need(self, HARNESS, "the Harness compiler is not installed")
        root = self.task()
        self.commit()
        self.assertIn(
            f"PASS {root}: task configuration and CPU Harness compilation",
            self.check(compile_task=True),
        )
        toml = (self.repo / root / "task.toml").read_text()
        self.write(root + "/task.toml", toml + 'cpus = "lots"\n')
        self.commit()
        self.check()
        self.assertFails(f"^{root}: Harness compilation failed", compile_task=True)

    def test_scanner_requires_trusted_policy(self):
        self.task()
        self.commit()
        self.assertFails("explicit trusted policy", gitleaks=Path("/bin/false"))

    def test_ci_invocation_warns_with_published_paths(self):
        need(self, GITLEAKS, "set PUBLICATION_GITLEAKS_BIN for scanner tests")
        need(self, HARNESS, "the Harness compiler is not installed")
        token = fake_token()
        self.task()
        self.log()
        self.write("rsi-tasks/README.md", f'token = "{token}"\n')
        self.write("rsi-logs/NOTES.md", f'github_token = "{token}"\n')
        line = json.dumps({"tool_output": f'export GITHUB_TOKEN="{token}"'}) + "\n"
        filler = b"".join(
            json.dumps({"n": n, "pad": sha(str(n).encode())}).encode() + b"\n"
            for n in range(100)
        )
        self.write(
            "rsi-logs/demo-task/m/trajectory.jsonl.gz",
            gzip.compress(filler + line.encode() + filler),
        )
        directory, _ = self.split("rsi-logs/demo-task/m/trajectory.jsonl.gz", 1024)
        self.commit()
        output = self.check(
            compile_task=True,
            gitleaks=Path(GITLEAKS),
            secret_config=HERE / "publication-gitleaks.toml",
        )
        self.assertNotIn(token, output)
        for label in (
            "rsi-tasks/README.md",
            "rsi-logs/NOTES.md",
            f"{directory}/manifest.json!trajectory.jsonl.gz!trajectory.jsonl",
        ):
            self.assertIn(f"::warning::Possible credential: {label};", output)


def v2_manifest(raw: bytes, artifact="agent_output.txt", **extra):
    doc = {
        "version": 2,
        "artifact": artifact,
        "size": len(raw),
        "sha256": sha(raw),
        "line_feeds": raw.count(b"\n"),
        "converted_from": {
            "artifact": artifact + ".gz",
            "compression": "gzip",
            "compressed_size": 123,
            "compressed_sha256": "c" * 64,
            "original_manifest": {"version": 1, "artifact": artifact + ".gz"},
        },
        "note": "Restored from the gzip publication in OpenRSI-Index; the Hub stores large files natively.",
    }
    doc.update(extra)
    return doc


LFS_POINTER_TEXT = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:" + "d" * 64 + "\n"
    "size 52428800\n"
)


class LogsModeTests(unittest.TestCase):
    """Two repositories: an Index checkout with tasks, and a Logs repository
    whose root is checked as rsi-logs/."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.index = Path(self.tmp.name) / "index"
        self.repo = Path(self.tmp.name) / "logs"
        for repo in (self.index, self.repo):
            repo.mkdir()
            self.git(repo, "init", "-q")
            self.git(repo, "config", "user.name", "Publication test")
            self.git(repo, "config", "user.email", "publication-test@example.invalid")
        self.write(self.index, "README.md", "Index\n")
        self.write(
            self.index,
            "rsi-tasks/demo-task/task.toml",
            'schema_version = "1.4"\n[task]\nname="rsi/demo"\ndescription="Synthetic"\n',
        )
        self.write(self.index, "rsi-tasks/demo-task/instruction.md", "Do it\n")
        self.write(
            self.index, "rsi-tasks/signature-tasks/gpic_generation/task.toml", "x\n"
        )
        self.task_ref = self.commit(self.index)
        self.write(self.repo, "README.md", "---\nlicense: other\n---\n# Logs\n")
        self.write(self.repo, "LICENSE", "License text\n")
        self.write(self.repo, ".gitattributes", "*.bin filter=lfs diff=lfs merge=lfs -text\n")
        self.base = self.commit(self.repo)

    def git(self, repo, *args):
        return (
            subprocess.check_output(
                ["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )

    def commit(self, repo=None):
        repo = repo or self.repo
        self.git(repo, "add", "-A")
        self.git(repo, "commit", "-qm", "synthetic fixture", "--allow-empty")
        return self.git(repo, "rev-parse", "HEAD")

    def write(self, repo, path, content):
        p = repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        return p

    def log(self, slug="demo-task"):
        self.write(self.repo, slug + "/README.md", "Recorded run\n")
        return slug

    def check(self, base=..., **options):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            pub.check(
                self.repo,
                self.base if base is ... else base,
                "HEAD",
                HERE / "publication-policy.toml",
                mount="rsi-logs",
                task_repo=self.index,
                task_ref=options.pop("task_ref", self.task_ref),
                **options,
            )
        return output.getvalue()

    def assertFails(self, pattern="", **options):
        with self.assertRaisesRegex(pub.CheckError, pattern):
            self.check(**options)

    # Layout and task correspondence

    def test_log_root_checked_against_index_task(self):
        self.log()
        self.write(self.repo, "demo-task/model/events.jsonl", '{"x": 1}\n')
        self.commit()
        output = self.check()
        self.assertIn("NOTICE Logs mode", output)
        self.assertIn("PASS 2 published files", output)

    def test_log_without_index_task_fails(self):
        self.log("missing-task")
        self.commit()
        self.assertFails(
            "^rsi-logs/missing-task: corresponding public task.toml not found in "
            "the task repository; merge the task into OpenRSI-Index first"
        )

    def test_tasks_come_from_the_task_ref_only(self):
        self.git(self.index, "checkout", "-qb", "proposal")
        self.write(self.index, "rsi-tasks/new-task/task.toml", "x\n")
        proposal = self.commit(self.index)
        self.log("new-task")
        self.commit()
        self.check(task_ref=proposal)
        self.assertFails("rsi-logs/new-task: corresponding public task.toml")
        # A task.toml inside the Logs repository never stands in for one.
        self.write(self.repo, "rsi-tasks/new-task/task.toml", "x\n")
        self.commit()
        self.assertFails("corresponding public task.toml")

    def test_unknown_task_ref_fails(self):
        self.log()
        self.commit()
        self.assertFails("task repository revision not found", task_ref="no-such-ref")

    def test_historical_alias_and_signature_layout(self):
        self.log("signature-tasks/gpic-10m-autoresearch")
        self.write(self.repo, "signature-tasks/README.md", "Signature logs\n")
        self.commit()
        output = self.check()
        self.assertIn("PASS 2 published files", output)

    def test_new_log_requires_readme(self):
        self.write(self.repo, "demo-task/model/agent_output.txt", "output\n")
        self.commit()
        self.assertFails("rsi-logs/demo-task: new log archive needs README.md")

    def test_root_infrastructure_is_scanned_only(self):
        self.write(self.repo, ".gitattributes", "*.txt filter=lfs diff=lfs merge=lfs -text\n")
        self.write(self.repo, "LICENSE", "Changed license\n")
        self.write(self.repo, "README.md", "Changed card\n")
        self.write(self.repo, "SANITIZATION.md", "Notes\n")
        self.commit()
        self.assertIn("PASS 4 published files", self.check())

    def test_other_root_files_fail(self):
        for name in ("data.json", "notes.txt"):
            with self.subTest(name=name):
                self.git(self.repo, "reset", "-q", "--hard", self.base)
                self.write(self.repo, name, "{}\n")
                self.commit()
                self.assertFails(
                    f"^rsi-logs/{re.escape(name)}: log files must be inside <task>/ "
                    "or signature-tasks/<task>/ of the Logs repository"
                )

    def test_unknown_top_level_directory_is_a_log_root_without_task(self):
        self.write(self.repo, ".github/workflows/x.yml", "on: push\n")
        self.commit()
        self.assertFails("^rsi-logs/.github: corresponding public task.toml not found")

    def test_unrelated_change_and_removal_pass(self):
        self.log()
        self.base = self.commit()
        self.assertIn("not applicable", self.check())
        (self.repo / "demo-task/README.md").unlink()
        self.commit()
        self.assertIn("PASS rsi-logs/demo-task: log archive removed", self.check())

    def test_full_audit_needs_no_base(self):
        self.write(self.repo, "missing-task/README.md", "x\n")
        self.base = self.commit()
        self.assertIn("not applicable", self.check())
        self.assertFails("rsi-logs/missing-task: corresponding", base=None, full=True)
        with self.assertRaisesRegex(pub.CheckError, "--base-ref is required"):
            self.check(base=None)

    def test_missing_or_unrelated_base_asks_for_full(self):
        self.log()
        self.commit()
        self.assertFails("audit the head with --full", base="f" * 40)
        self.git(self.repo, "checkout", "-q", "--orphan", "rewritten")
        self.commit()
        self.assertFails("shares no history", base=self.base)
        self.check(base=None, full=True)

    def test_mode_options_must_be_complete(self):
        with self.assertRaisesRegex(pub.CheckError, "Logs mode needs"):
            pub.check(
                self.repo, self.base, "HEAD", HERE / "publication-policy.toml",
                mount="rsi-logs",
            )
        with self.assertRaisesRegex(pub.CheckError, "only used with --mount"):
            pub.check(
                self.repo, self.base, "HEAD", HERE / "publication-policy.toml",
                task_repo=self.index, task_ref="HEAD",
            )

    # Raw files only

    def test_new_compressed_files_and_chunks_rejected(self):
        raw = b"line\n" * 100
        cases = {
            "demo-task/m/agent_output.txt.gz": gzip.compress(raw),
            "demo-task/m/trace.jsonl.xz": lzma.compress(raw),
            "demo-task/m/trace.bz2": bz2.compress(raw),
            "demo-task/m/trace.zst": b"\x28\xb5\x2f\xfd" + raw,
            "demo-task/m/bundle.zip": b"PK\x05\x06" + b"\0" * 18,
            "demo-task/m/bundle.tar": tar_bytes([("a.txt", raw)]),
            "demo-task/m/bundle.tar.lz4": raw,
            "demo-task/m/agent_output.txt": gzip.compress(raw),
            "demo-task/m/workspace": tar_bytes([("a.txt", raw)]),
            "demo-task/m/agent_output.txt.gz.parts/part-0001.bin": raw,
        }
        for name, content in cases.items():
            with self.subTest(name=name):
                self.git(self.repo, "reset", "-q", "--hard", self.base)
                self.git(self.repo, "clean", "-qfdx")
                self.log()
                self.write(self.repo, name, content)
                self.commit()
                self.assertFails(
                    f"^rsi-logs/{re.escape(name)}: .*stores raw files only .*"
                    "upload the uncompressed, unsplit file"
                )

    def test_chunk_manifest_rejected(self):
        self.log()
        payload = b"whole transcript\n"
        self.write(self.repo, "demo-task/m/out.txt.chunks/p1", payload)
        manifest = {
            "sha256": sha(payload),
            "bytes": len(payload),
            "parts": [{"path": "p1", "bytes": len(payload), "sha256": sha(payload)}],
        }
        self.write(
            self.repo, "demo-task/m/out.txt.chunks/manifest.json", json.dumps(manifest)
        )
        self.commit()
        self.assertFails(
            "^rsi-logs/demo-task/m/out.txt.chunks/manifest.json: chunk manifest; "
            "the Logs repository stores raw files only"
        )

    def test_zip_containers_with_their_own_suffix_are_raw(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("arr.npy", b"\x93NUMPY")
        self.log()
        self.write(self.repo, "demo-task/m/arrays.npz", data.getvalue())
        self.commit()
        self.check()

    def test_unchanged_compressed_file_is_not_rejected_incrementally(self):
        self.log()
        self.write(self.repo, "demo-task/m/old.txt.gz", gzip.compress(b"old\n"))
        self.base = self.commit()
        self.write(self.repo, "demo-task/m/new.txt", "new\n")
        self.commit()
        self.check()
        self.assertFails("old.txt.gz: compressed", base=None, full=True)

    def test_large_raw_file_has_no_stored_size_limit(self):
        self.log()
        self.write(self.repo, "demo-task/m/agent_output.txt", b"x\n" * 10)
        self.commit()
        with patch.object(pub, "MAX_STORED", 4):
            self.check()

    def test_lfs_pointer_is_an_error(self):
        self.log()
        self.write(self.repo, "demo-task/m/agent_output.txt", LFS_POINTER_TEXT)
        self.commit()
        self.assertFails(
            "^rsi-logs/demo-task/m/agent_output.txt: Git LFS pointer, not content; "
            "download LFS objects"
        )
        # Infrastructure files must be downloaded as well.
        self.write(self.repo, "demo-task/m/agent_output.txt", "content\n")
        self.write(self.repo, "LICENSE", LFS_POINTER_TEXT)
        self.commit()
        self.assertFails("^rsi-logs/LICENSE: Git LFS pointer")

    def test_gitattributes_only_at_root_and_only_for_lfs(self):
        self.log()
        self.write(self.repo, "demo-task/m/b.txt", "b\n")
        self.commit()
        self.check()
        # A nested .gitattributes could re-encode the checkout of a log file.
        self.write(self.repo, "demo-task/.gitattributes", "m/b.txt working-tree-encoding=UTF-16LE\n")
        self.commit()
        self.assertFails(
            "^rsi-logs/demo-task/.gitattributes: .gitattributes is only allowed at "
            "the Logs repository root"
        )
        (self.repo / "demo-task/.gitattributes").unlink()
        self.commit()
        self.check()
        lfs = "*.bin filter=lfs diff=lfs merge=lfs -text\n"
        self.write(self.repo, ".gitattributes", "# Large files\n\n" + lfs + "*.parquet -text merge=lfs diff=lfs filter=lfs\n")
        self.commit()
        self.check()
        for label, rule in {
            "encoding": "*.txt working-tree-encoding=UTF-16LE",
            "eol": "*.txt eol=crlf",
            "text": "* text",
            "ident": "*.txt ident",
            "extra": "*.bin filter=lfs diff=lfs merge=lfs -text text",
            "macro": "[attr]lfs filter=lfs diff=lfs merge=lfs -text",
            "partial": "*.bin filter=lfs",
        }.items():
            with self.subTest(label=label):
                self.write(self.repo, ".gitattributes", lfs + rule + "\n")
                self.commit()
                self.assertFails(
                    "^rsi-logs/.gitattributes: line 2: only '<pattern> filter=lfs "
                    "diff=lfs merge=lfs -text' rules are allowed"
                )
        # A --full audit applies the same rule to an unchanged tree.
        self.base = self.commit()
        self.assertFails("^rsi-logs/.gitattributes: line 2", base=None, full=True)

    def test_lfs_includes_cover_what_the_check_reads(self):
        self.log()
        self.log("signature-tasks/gpic-10m-autoresearch")
        self.write(self.repo, "demo-task/m/a.txt", "a\n")
        self.base = self.commit()
        (self.repo / "demo-task/m/a.txt").unlink()
        self.write(self.repo, "signature-tasks/gpic-10m-autoresearch/m/b.txt", "b\n")
        self.write(self.repo, "signature-tasks/README.md", "docs\n")
        self.write(self.repo, "LICENSE", "new\n")
        self.commit()
        revisions = pub.Revisions(self.repo, self.base, "HEAD", False, "rsi-logs")
        self.assertEqual(
            pub.lfs_includes(revisions),
            [
                "LICENSE",
                "demo-task/**",
                "signature-tasks/README.md",
                "signature-tasks/gpic-10m-autoresearch/**",
            ],
        )

    # Manifests

    def test_v2_manifest_verified(self):
        raw = b"line one\nline two\n"
        self.log()
        self.write(self.repo, "demo-task/m/agent_output.txt", raw)
        manifest = "demo-task/m/agent_output.manifest.json"
        self.write(self.repo, manifest, json.dumps(v2_manifest(raw)))
        self.commit()
        self.check()
        bad = {
            "size": (dict(size=len(raw) + 1), "agent_output.txt: manifest byte count mismatch"),
            "sha": (dict(sha256="0" * 64), "agent_output.txt: manifest SHA-256 mismatch"),
            "lines": (dict(line_feeds=3), "agent_output.txt: manifest line_feeds mismatch"),
            "type": (dict(size="18"), "size and line_feeds must be integers"),
            "digest": (dict(sha256="ABC"), "sha256 must be a SHA-256 hex digest"),
            "path": (dict(artifact="../x.txt"), "artifact must be a file name"),
            "source": (
                dict(converted_from={"artifact": "a.gz", "compression": "gzip"}),
                "converted_from requires",
            ),
            "parts": (
                dict(
                    converted_from=dict(
                        v2_manifest(raw)["converted_from"], parts=[{"path": "p"}]
                    )
                ),
                "converted_from parts require",
            ),
        }
        for label, (change, message) in bad.items():
            with self.subTest(label=label):
                self.write(self.repo, manifest, json.dumps(v2_manifest(raw, **change)))
                self.commit()
                self.assertFails(f"^rsi-logs/{manifest}: .*{re.escape(message)}")
        doc = v2_manifest(raw)
        del doc["line_feeds"]
        self.write(self.repo, manifest, json.dumps(doc))
        self.commit()
        self.assertFails("version 2 manifest lacks line_feeds")

    def test_v2_manifest_from_chunks_and_name_rule(self):
        raw = b"chunked once\n"
        self.log()
        self.write(self.repo, "demo-task/m/agent_output.txt", raw)
        source = dict(
            v2_manifest(raw)["converted_from"],
            artifact="agent_output.txt.gz.parts",
            parts=[{"path": "part-0001.bin", "bytes": 3, "sha256": "e" * 64}],
            original_manifest={"schema_version": 1, "parts": [{"path": "part-0001.bin"}]},
        )
        manifest = "demo-task/m/agent_output.manifest.json"
        self.write(self.repo, manifest, json.dumps(v2_manifest(raw, converted_from=source)))
        self.commit()
        self.check()
        (self.repo / manifest).rename(self.repo / "demo-task/m/other.manifest.json")
        self.commit()
        self.assertFails(
            "other.manifest.json: version 2 manifest for agent_output.txt must be "
            "named agent_output.manifest.json"
        )

    def test_v2_manifest_needs_its_tracked_artifact(self):
        raw = b"x\n"
        self.log()
        self.write(
            self.repo,
            "demo-task/m/agent_output.manifest.json",
            json.dumps(v2_manifest(raw)),
        )
        self.commit()
        self.assertFails("agent_output.txt: manifest references missing")

    def test_misnamed_v2_fields_are_not_skipped(self):
        raw = b"line one\nline two\n"
        self.log()
        self.write(self.repo, "demo-task/m/agent_output.txt", raw)
        manifest = "demo-task/m/agent_output.manifest.json"
        doc = v2_manifest(raw, size=999, sha256="0" * 64, line_feeds=7)
        doc["artifcat"] = doc.pop("artifact")
        self.write(self.repo, manifest, json.dumps(doc))
        self.commit()
        self.assertFails(f"^rsi-logs/{manifest}: version 2 manifest lacks artifact")
        # A version typo is not a pass either: in the Logs repository a
        # *.manifest.json must be a version 2 manifest.
        self.write(self.repo, manifest, json.dumps(dict(v2_manifest(raw), version="2")))
        self.commit()
        self.assertFails(
            f"^rsi-logs/{manifest}: a \\*.manifest.json in the Logs repository must "
            "be a version 2 manifest"
        )
        self.write(self.repo, manifest, json.dumps(v2_manifest(raw)))
        self.commit()
        self.check()

    def test_publication_manifest_lists_raw_files(self):
        raw = b"restored\n"
        self.log()
        self.write(self.repo, "demo-task/m/agent_output.txt", raw)
        entry = {
            "path": "m/agent_output.txt",
            "bytes": len(raw),
            "sha256": sha(raw),
            "uncompressed_bytes": len(raw),
            "uncompressed_sha256": sha(raw),
            "line_count": 1,
            "compression": None,
        }
        self.write(self.repo, "demo-task/publication.json", json.dumps({"files": [entry]}))
        self.commit()
        self.check()
        self.write(
            self.repo,
            "demo-task/publication.json",
            json.dumps({"files": [dict(entry, path="m/agent_output.txt.gz")]}),
        )
        self.commit()
        self.assertFails("publication.json: m/agent_output.txt.gz: manifest references missing")

    def test_structured_records_still_validated(self):
        self.log()
        self.write(self.repo, "demo-task/m/events.jsonl", '{"x": 1}\n{bad\n')
        self.commit()
        self.assertFails("^rsi-logs/demo-task/m/events.jsonl: invalid JSONL at line 2")

    def test_cli_logs_mode_and_lfs_include(self):
        self.log()
        self.commit()
        script = str(HERE / "check_publication.py")
        common = ["--repo", str(self.repo), "--mount", "rsi-logs", "--base-ref", self.base]
        result = subprocess.run(
            [sys.executable, "-I", script, *common, "--lfs-include"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(result.stdout, "demo-task/**\n")
        result = subprocess.run(
            [
                sys.executable, "-I", script, *common,
                "--task-repo", str(self.index), "--task-ref", self.task_ref,
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASS 1 published files", result.stdout)
        result = subprocess.run(
            [sys.executable, "-I", script, "--repo", str(self.repo), "--mount", "rsi-logs",
             "--task-repo", str(self.index), "--task-ref", self.task_ref],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL --base-ref is required unless --full", result.stdout)

    def test_ci_invocation_scans_with_published_paths(self):
        need(self, GITLEAKS, "set PUBLICATION_GITLEAKS_BIN for scanner tests")
        token = fake_token()
        self.log()
        self.write(self.repo, "LICENSE", f'github_token = "{token}"\n')
        line = json.dumps({"tool_output": f'export GITHUB_TOKEN="{token}"'}) + "\n"
        self.write(self.repo, "demo-task/m/trajectory.jsonl", line)
        self.commit()
        output = self.check(
            gitleaks=Path(GITLEAKS), secret_config=HERE / "publication-gitleaks.toml"
        )
        self.assertNotIn(token, output)
        for label in ("rsi-logs/LICENSE", "rsi-logs/demo-task/m/trajectory.jsonl"):
            self.assertIn(f"::warning::Possible credential: {label};", output)


class WorkflowTests(unittest.TestCase):
    def workflow(self):
        try:
            import yaml
        except ImportError:
            need(self, False, "PyYAML is not installed")
        return yaml.safe_load(WORKFLOW.read_text())

    def test_cpu_only_without_secrets(self):
        text = WORKFLOW.read_text()
        self.assertNotIn("pull_request_target:", text)
        self.assertNotIn("secrets.", text)
        self.assertIn("contents: read", text)
        self.assertNotIn("persist-credentials: true", text)
        self.assertEqual(text.count("persist-credentials: false"), 3)
        for forbidden in (
            "harbor run",
            "docker build",
            "nvidia-smi",
            "pytest -m gpu",
            "workflow_run:",
        ):
            self.assertNotIn(forbidden, text)

    def test_triggers_cover_exactly_what_can_change_the_result(self):
        on = self.workflow()[True]
        watched = [
            "rsi-tasks/**",
            "checks/check_publication.py",
            "checks/test_publication.py",
            "checks/publication-*.toml",
            "RSI-Harness/pyproject.toml",
            "RSI-Harness/uv.lock",
            "RSI-Harness/src/**",
            ".github/workflows/task-log-publication-checks.yml",
            ".github/workflows/hf-logs-publication-checks.yml",
            ".github/workflows/history-guard.yml",
            "checks/publish_logs.py",
            "checks/test_publish_logs.py",
        ]
        self.assertEqual(on["pull_request"]["paths"], watched)
        self.assertEqual(on["push"]["paths"], watched)
        self.assertEqual(on["push"]["branches"], ["main"])
        self.assertIn("workflow_dispatch", on)

    def test_enforcement_runs_only_base_code(self):
        job = self.workflow()["jobs"]["publication"]
        self.assertEqual(job["name"], "Task and log publication (CPU only)")
        runs = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertNotIn("proposal/checks", runs)
        self.assertNotIn("test_publication", runs)
        validate = job["steps"][-1]
        self.assertEqual(validate["working-directory"], "trusted-base/RSI-Harness")
        run = validate["run"]
        self.assertIn('PUBLICATION_CHECKS="$GITHUB_WORKSPACE/trusted-base/checks"', run)
        self.assertEqual(run.count("PUBLICATION_CHECKS="), 1)
        self.assertIn("exit 1", run)
        for flag in ("--frozen --no-dev", "--compile", "--gitleaks", "--secret-config"):
            self.assertIn(flag, run)

    def test_only_superseded_pr_runs_are_cancelled(self):
        group = self.workflow()["concurrency"]["group"]
        # Push and manual runs fall back to their unique run id.
        self.assertIn("github.event.pull_request.number || github.run_id", group)
        self.assertIn("github.event_name", group)

    def test_enforcement_checks_out_base_code_and_reads_proposal(self):
        steps = self.workflow()["jobs"]["publication"]["steps"]
        checkouts = {
            step["with"]["path"]: step["with"]
            for step in steps
            if step.get("uses", "").startswith("actions/checkout@")
        }
        self.assertEqual(set(checkouts), {"proposal", "trusted-base"})
        base = checkouts["trusted-base"]
        self.assertEqual(
            base["ref"], "${{ github.event.pull_request.base.sha || github.sha }}"
        )
        self.assertEqual(base["sparse-checkout"].split(), ["RSI-Harness", "checks"])
        proposal = checkouts["proposal"]
        self.assertEqual(
            proposal["ref"], "${{ github.event.pull_request.head.sha || github.sha }}"
        )
        self.assertEqual(proposal["fetch-depth"], 0)
        # Pushes to main diff against the previous tip, not against themselves.
        env = steps[-1]["env"]
        self.assertEqual(
            env["BASE_REF"],
            "${{ github.event.pull_request.base.sha || github.event.before || github.sha }}",
        )
        self.assertEqual(
            env["HEAD_REF"], "${{ github.event.pull_request.head.sha || github.sha }}"
        )
        run = steps[-1]["run"]
        self.assertIn('--repo "$GITHUB_WORKSPACE/proposal"', run)
        self.assertIn('python -I "$PUBLICATION_CHECKS/check_publication.py"', run)
        self.assertIn("set -euo pipefail", run)

    def test_scanner_is_checksum_pinned_and_jobs_time_out(self):
        jobs = self.workflow()["jobs"]
        self.assertEqual(jobs["publication"]["timeout-minutes"], 20)
        self.assertEqual(jobs["checker-tests"]["timeout-minutes"], 15)
        for name, job in jobs.items():
            installs = [
                step["run"]
                for step in job["steps"]
                if "gitleaks" in step.get("run", "") and "curl" in step["run"]
            ]
            self.assertEqual(len(installs), 1, name)
            self.assertIn("set -euo pipefail", installs[0])
            self.assertIn(
                "'551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb' \\\n"
                '  "$SCANNER_DIR/gitleaks.tar.gz" | sha256sum --check --strict\n',
                installs[0],
            )

    def test_manual_run_audits_everything(self):
        run = self.workflow()["jobs"]["publication"]["steps"][-1]["run"]
        self.assertIn('if [ "$GITHUB_EVENT_NAME" = "workflow_dispatch" ]', run)
        self.assertIn("mode=(--full)", run)
        self.assertIn('"${mode[@]}"', run)

    def test_forced_push_or_missing_base_falls_back_to_full(self):
        step = self.workflow()["jobs"]["publication"]["steps"][-1]
        self.assertEqual(step["env"]["FORCED"], "${{ github.event.forced || false }}")
        run = step["run"]
        self.assertIn('[ "$FORCED" = "true" ]', run)
        self.assertIn(
            '! git -C "$GITHUB_WORKSPACE/proposal" cat-file -e "${BASE_REF}^{commit}"', run
        )
        # The checker's merge base must exist, so a full audit diffs the head
        # against itself rather than against an unreachable base.
        self.assertEqual(run.count('BASE_REF="$HEAD_REF"'), 2)
        self.assertEqual(run.count("mode=(--full)"), 2)

    def test_proposed_checker_tests_run_in_their_own_job(self):
        jobs = self.workflow()["jobs"]
        tests = jobs["checker-tests"]
        self.assertNotIn("needs", jobs["publication"])
        run = tests["steps"][-1]
        self.assertIn("test_publication.py", run["run"])
        self.assertEqual(run["env"]["PUBLICATION_REQUIRE_INTEGRATION"], "1")
        self.assertIn("PUBLICATION_GITLEAKS_BIN", run["env"])
        for step in tests["steps"]:
            self.assertNotIn("trusted-base", json.dumps(step))

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


WORKFLOWS = HERE.parent / ".github/workflows"
HF_WORKFLOW = WORKFLOWS / "hf-logs-publication-checks.yml"
GUARD_WORKFLOW = WORKFLOWS / "history-guard.yml"
_SPEC_TOOL = importlib.util.spec_from_file_location("publish_logs", HERE / "publish_logs.py")
publish_logs = importlib.util.module_from_spec(_SPEC_TOOL)
_SPEC_TOOL.loader.exec_module(publish_logs)
LOGS_REPO = publish_logs.LOGS_REPO


def load_workflow(test: unittest.TestCase, path: Path):
    try:
        import yaml
    except ImportError:
        need(test, False, "PyYAML is not installed")
    return yaml.safe_load(path.read_text())


def named_step(job, name):
    (step,) = [s for s in job["steps"] if s.get("name") == name]
    return step


def run_step(script: str, cwd: Path, env: dict):
    """Run a workflow step the way GitHub's bash shell does."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step.sh"
        path.write_text(script)
        output = Path(tmp) / "output"
        output.touch()
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(path)],
            cwd=cwd,
            env={
                "PATH": os.environ["PATH"],
                "HOME": os.environ.get("HOME", tmp),
                "GITHUB_OUTPUT": str(output),
                **env,
            },
            capture_output=True,
            text=True,
        )
        values = dict(
            line.split("=", 1) for line in output.read_text().splitlines() if line
        )
    return result, values


class HfLogsAuditWorkflowTests(unittest.TestCase):
    WATCHED = [
        "checks/check_publication.py",
        "checks/publication-*.toml",
        "checks/publish_logs.py",
        "rsi-tasks/**",
        ".github/workflows/hf-logs-publication-checks.yml",
    ]

    def setUp(self):
        self.wf = load_workflow(self, HF_WORKFLOW)
        self.job = self.wf["jobs"]["audit"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def step(self, name):
        return named_step(self.job, name)

    def test_triggers(self):
        on = self.wf[True]
        self.assertEqual(set(on), {"push", "pull_request", "workflow_dispatch"})
        self.assertEqual(on["push"]["branches"], ["main"])
        self.assertEqual(on["push"]["paths"], self.WATCHED)
        self.assertEqual(on["pull_request"]["paths"], self.WATCHED)
        inputs = on["workflow_dispatch"]["inputs"]
        self.assertEqual(set(inputs), {"revision"})
        self.assertEqual(inputs["revision"]["default"], "main")

    def test_read_only_without_secrets_or_inline_expressions(self):
        text = HF_WORKFLOW.read_text()
        self.assertEqual(self.wf["permissions"], {"contents": "read"})
        self.assertNotIn("secrets.", text)
        self.assertNotIn("pull_request_target", text)
        self.assertNotIn("persist-credentials: true", text)
        self.assertEqual(text.count("persist-credentials: false"), 2)
        for step in self.job["steps"]:
            self.assertNotIn("${{", step.get("run", ""), step.get("name"))

    def test_pinned_runners(self):
        self.assertEqual(self.job["runs-on"], "ubuntu-24.04")
        guard = load_workflow(self, GUARD_WORKFLOW)
        self.assertEqual(guard["jobs"]["history-guard"]["runs-on"], "ubuntu-24.04")

    def test_repository_id_comes_from_the_upload_tool(self):
        self.assertNotIn("OpenRSI-Index-Logs", HF_WORKFLOW.read_text())
        self.assertIn("publish_logs.py\" --print-repo", self.step("Locate the Logs repository")["run"])

    def test_trusted_checker_and_evaluated_task_list(self):
        checkouts = {
            s["with"]["path"]: s["with"]
            for s in self.job["steps"]
            if s.get("uses", "").startswith("actions/checkout@")
        }
        self.assertEqual(set(checkouts), {"index", "trusted"})
        self.assertEqual(
            checkouts["trusted"]["ref"],
            "${{ github.event.pull_request.base.sha || github.sha }}",
        )
        self.assertEqual(checkouts["trusted"]["sparse-checkout"], "checks")
        self.assertEqual(
            checkouts["index"]["ref"],
            "${{ github.event.pull_request.head.sha || github.sha }}",
        )
        self.assertEqual(checkouts["index"]["filter"], "blob:none")
        run = self.step("Audit every published log (--full)")["run"]
        self.assertIn('checks="$GITHUB_WORKSPACE/trusted/checks"', run)
        for flag in (
            '--repo "$GITHUB_WORKSPACE/logs" --mount rsi-logs --full',
            '--task-repo "$GITHUB_WORKSPACE/index" --task-ref HEAD',
            '--policy "$checks/publication-policy.toml"',
            '--secret-config "$checks/publication-gitleaks.toml"',
            '--gitleaks "$SCANNER_DIR/gitleaks"',
            'exit "$status"',
        ):
            self.assertIn(flag, run)
        self.assertNotIn("index/checks", HF_WORKFLOW.read_text())

    def test_scanner_install_matches_the_task_workflow(self):
        task_log = load_workflow(self, WORKFLOW)
        name = "Install checksum-pinned secret scanner"
        self.assertEqual(self.step(name), named_step(task_log["jobs"]["publication"], name))
        self.assertEqual(
            {s["uses"] for s in self.job["steps"] if "uses" in s},
            {named_step(task_log["jobs"]["publication"], "Checkout submitted publication data")["uses"]},
        )

    def locate(self, code="200", revision="main", tool=True):
        checks = self.root / "trusted/checks"
        checks.mkdir(parents=True, exist_ok=True)
        for name in ("check_publication.py", "publication-policy.toml", "publication-gitleaks.toml"):
            shutil.copy(HERE / name, checks / name)
        if tool:
            shutil.copy(HERE / "publish_logs.py", checks / "publish_logs.py")
        else:
            (checks / "publish_logs.py").unlink(missing_ok=True)
        bin_dir = self.root / "bin"
        bin_dir.mkdir(exist_ok=True)
        curl = bin_dir / "curl"
        curl.write_text('#!/bin/sh\nprintf %s "$FAKE_HTTP_CODE"\n')
        curl.chmod(0o755)
        return run_step(
            self.step("Locate the Logs repository")["run"],
            self.root,
            {
                "GITHUB_WORKSPACE": str(self.root),
                "REVISION": revision,
                "FAKE_HTTP_CODE": code,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
            },
        )

    def test_locate_outcomes(self):
        result, out = self.locate()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(out, {"available": "true", "repo": LOGS_REPO, "revision": "main"})
        for code in ("401", "404"):
            result, out = self.locate(code)
            self.assertEqual((result.returncode, out), (0, {"available": "false"}))
            self.assertIn("is not published", result.stdout)
        result, out = self.locate("503")
        self.assertEqual(result.returncode, 1)
        self.assertIn("audit did not run", result.stdout)
        result, out = self.locate(tool=False)
        self.assertEqual((result.returncode, out), (0, {"available": "false"}))
        self.assertIn("predates Logs repository audits", result.stdout)
        for revision in ("refs/pr/7", "a" * 40):
            self.assertEqual(self.locate(revision=revision)[1]["revision"], revision)
        for revision in ("dev", "refs/heads/main;id", "A" * 40, "refs/pr/x"):
            result, out = self.locate(revision=revision)
            self.assertEqual(result.returncode, 1, revision)
            self.assertNotIn("available", out)

    def test_fetch_checks_out_blobs_not_attribute_conversions(self):
        need(self, shutil.which("git-lfs"), "git-lfs is not installed")
        remote = self.root / "remote"
        remote.mkdir()

        def git(*args, cwd=remote):
            return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()

        git("init", "-q", "-b", "main")
        git("config", "user.name", "t")
        git("config", "user.email", "t@example.invalid")
        blobs = {
            "secret.txt": "token=example-value\n",
            "crlf.txt": "one\ntwo\n",
            ".gitattributes": (
                "secret.txt working-tree-encoding=UTF-16LE\n"
                "crlf.txt text eol=crlf\n"
            ),
        }
        for name, content in blobs.items():
            (remote / name).write_text(content)
            # Store the bytes as they are, bypassing the attributes being added.
            oid = git("hash-object", "-w", "--no-filters", name)
            git("update-index", "--add", "--cacheinfo", f"100644,{oid},{name}")
        git("commit", "-qm", "attributes")
        sha = git("rev-parse", "HEAD")
        # A plain clone applies the in-tree attributes.
        subprocess.run(["git", "clone", "-q", str(remote), str(self.root / "plain")], check=True)
        self.assertNotEqual((self.root / "plain/secret.txt").read_bytes(), blobs["secret.txt"].encode())
        self.assertNotEqual((self.root / "plain/crlf.txt").read_bytes(), blobs["crlf.txt"].encode())
        run = self.step("Fetch the Logs revision with its LFS content (anonymous)")["run"]
        run = run.replace('"https://huggingface.co/datasets/$LOGS_REPO"', '"$LOCAL_REMOTE"')
        work = self.root / "work"
        work.mkdir()
        result, out = run_step(
            run, work, {"LOGS_REPO": "x/y", "LOCAL_REMOTE": str(remote), "REVISION": "main"}
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(out, {"sha": sha})
        for name, content in blobs.items():
            self.assertEqual((work / "logs" / name).read_bytes(), content.encode(), name)
        self.assertEqual(
            git("config", "lfs.url", cwd=work / "logs"),
            "https://huggingface.co/datasets/x/y.git/info/lfs",
        )

    def test_fetch_step_against_a_local_logs_repository(self):
        need(self, shutil.which("git-lfs"), "git-lfs is not installed")
        remote = self.root / "remote"
        remote.mkdir()

        def git(*args):
            return subprocess.check_output(["git", "-C", str(remote), *args], text=True).strip()

        git("init", "-q", "-b", "main")
        git("config", "user.name", "t")
        git("config", "user.email", "t@example.invalid")
        (remote / "README.md").write_text("x\n")
        git("add", "-A")
        git("commit", "-qm", "one")
        main = git("rev-parse", "HEAD")
        git("checkout", "-qb", "pr")
        (remote / "a.txt").write_text("a\n")
        git("add", "-A")
        git("commit", "-qm", "pr")
        pr = git("rev-parse", "HEAD")
        git("update-ref", "refs/pr/5", pr)
        git("checkout", "-q", "main")
        run = self.step("Fetch the Logs revision with its LFS content (anonymous)")["run"]
        self.assertIn('"https://huggingface.co/datasets/$LOGS_REPO"', run)
        run = run.replace('"https://huggingface.co/datasets/$LOGS_REPO"', '"$LOCAL_REMOTE"')

        def fetch(case, revision):
            work = self.root / case
            work.mkdir()
            return run_step(
                run, work, {"LOGS_REPO": "x/y", "LOCAL_REMOTE": str(remote), "REVISION": revision}
            )

        for case, revision, expected in (
            ("main", "main", main),
            ("pr", "refs/pr/5", pr),
            ("sha", pr, pr),
        ):
            result, out = fetch(case, revision)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(out, {"sha": expected})
            head = subprocess.check_output(
                ["git", "-C", str(self.root / case / "logs"), "rev-parse", "HEAD"], text=True
            ).strip()
            self.assertEqual(head, expected)
        result, out = fetch("unknown", "c" * 40)
        self.assertEqual(result.returncode, 1)
        message = f"Commit {'c' * 40} is not on main or a pull request of x/y."
        self.assertIn(f"::error::{message}", result.stdout)
        self.assertEqual(out, {"error": message})
        # A pull request that does not exist: an annotation, not just git's fatal.
        result, out = fetch("missing-pr", "refs/pr/9")
        self.assertEqual(result.returncode, 1)
        message = "Could not fetch refs/pr/9 from x/y (no such pull request or commit?)"
        self.assertIn(f"::error::{message}\n", result.stdout)
        self.assertEqual(out, {"error": message})

    def test_job_summary(self):
        output = self.root / "logs-audit.txt"
        output.write_text("::warning::Possible credential: a```b; rule=x, line=1\nFAIL done\n")
        step = self.step("Job summary")
        self.assertIn("always()", step["if"])
        summary = self.root / "summary.md"
        result, _ = run_step(
            step["run"],
            self.root,
            {
                "RUNNER_TEMP": str(self.root),
                "GITHUB_STEP_SUMMARY": str(summary),
                "OUTCOME": "failure",
                "LOGS_REPO": "o/r",
                "REVISION": "main",
                "SHA": "a" * 40,
                "INDEX_SHA": "b" * 40,
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = summary.read_text()
        self.assertIn("### Logs repository audit: failure", text)
        self.assertIn("WARNING: Possible credential", text)
        self.assertNotIn("::warning::", text)
        self.assertIn("````text\n", text)
        self.assertIn("FAIL done", text)
        self.assertNotIn("Not audited", text)

    def test_job_summary_explains_a_failed_fetch(self):
        step = self.step("Job summary")
        self.assertEqual(step["env"]["FETCH_OUTCOME"], "${{ steps.fetch.outcome }}")
        self.assertEqual(step["env"]["FETCH_ERROR"], "${{ steps.fetch.outputs.error }}")
        base = {
            "RUNNER_TEMP": str(self.root),
            "OUTCOME": "skipped",
            "LOGS_REPO": "o/r",
            "REVISION": "refs/pr/9",
            "SHA": "",
            "INDEX_SHA": "b" * 40,
            "FETCH_OUTCOME": "failure",
        }
        for case, error, reason in (
            (
                "annotated",
                "Could not fetch refs/pr/9 from o/r (no such pull request or commit?)",
                "Could not fetch refs/pr/9 from o/r (no such pull request or commit?)",
            ),
            ("unannotated", "", "the fetch step failed; see its log."),
        ):
            summary = self.root / f"{case}.md"
            result, _ = run_step(
                step["run"],
                self.root,
                {**base, "FETCH_ERROR": error, "GITHUB_STEP_SUMMARY": str(summary)},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            text = summary.read_text()
            self.assertIn("### Logs repository audit: failure", text)
            self.assertIn("at `refs/pr/9` (`not fetched`)", text)
            self.assertIn(f"- Not audited: {reason}", text)
            self.assertNotIn("```", text)


def tomllib_loads(text):
    import tomllib

    return tomllib.loads(text)


class HistoryGuardTests(unittest.TestCase):
    STEP = "Reject unrelated history and removed paths"

    def setUp(self):
        self.wf = load_workflow(self, GUARD_WORKFLOW)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        self.git("init", "-q")
        self.git("config", "user.name", "Guard test")
        self.git("config", "user.email", "guard-test@example.invalid")
        self.policy(["rsi-logs/", "rsi-tasks/demo/tests/assets/bright/"])
        self.write("README.md", "x\n")
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    def write(self, path, text):
        p = self.repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def policy(self, prefixes):
        listed = ", ".join(json.dumps(p) for p in prefixes)
        self.write(
            "checks/publication-policy.toml",
            f"[historical_tasks]\n\n[history_guard]\nforbidden_prefixes = [{listed}]\n",
        )

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture", "--allow-empty")
        return self.git("rev-parse", "HEAD")

    def guard(self, head=None):
        step = named_step(self.wf["jobs"]["history-guard"], self.STEP)
        return run_step(
            step["run"],
            self.repo,
            {"BASE_SHA": self.base, "HEAD_SHA": head or self.git("rev-parse", "HEAD")},
        )[0]

    def test_workflow_runs_on_every_pull_request(self):
        self.assertEqual(self.wf[True], {"pull_request": None})
        job = self.wf["jobs"]["history-guard"]
        self.assertEqual(job["name"], "history-guard")
        self.assertEqual(self.wf["permissions"], {"contents": "read"})
        (checkout,) = [s for s in job["steps"] if "uses" in s]
        self.assertEqual(checkout["with"]["fetch-depth"], 0)
        self.assertFalse(checkout["with"]["persist-credentials"])
        for step in job["steps"]:
            self.assertNotIn("${{", step.get("run", ""))

    def test_policy_lists_the_removed_assets(self):
        policy = tomllib_loads((HERE / "publication-policy.toml").read_text())
        self.assertEqual(
            policy["history_guard"]["forbidden_prefixes"],
            [
                "rsi-logs/",
                "rsi-tasks/reasonir-difficulty-curriculum/tests/assets/bright/",
            ],
        )

    def test_related_pr_passes(self):
        self.write("rsi-tasks/demo/tests/assets/manifest.json", "{}\n")
        self.write("rsi-logs-notes.md", "not under the prefix\n")
        self.commit()
        result = self.guard()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_added_then_deleted_file_still_fails(self):
        self.write("rsi-tasks/demo/tests/assets/bright/q.parquet", "data\n")
        self.commit()
        self.git("rm", "-rq", "rsi-tasks/demo/tests/assets/bright")
        self.commit()
        result = self.guard()
        self.assertEqual(result.returncode, 1)
        self.assertIn("::error::This PR's commits add 1 file(s)", result.stdout)
        self.assertIn("  removed path: rsi-tasks/demo/tests/assets/bright/q.parquet", result.stdout)

    def test_files_arriving_through_a_merge_fail(self):
        self.git("checkout", "-qb", "old")
        self.write("rsi-logs/task/run.log", "x\n")
        self.commit()
        self.git("checkout", "-q", "-")
        self.git("checkout", "-qb", "feature")
        self.write("feature.md", "x\n")
        self.commit()
        self.git("merge", "-q", "--no-edit", "old")
        self.assertEqual(self.guard().returncode, 1)

    def test_unrelated_history_fails(self):
        self.git("checkout", "-q", "--orphan", "rewritten")
        self.write("other.md", "x\n")
        result = self.guard(self.commit())
        self.assertEqual(result.returncode, 1)
        self.assertIn("shares no history with its base", result.stdout)

    def test_prefixes_come_from_the_base_revision(self):
        self.policy([])
        self.write("rsi-logs/task/run.log", "x\n")
        self.commit()
        self.assertEqual(self.guard().returncode, 1)

    def test_base_without_a_list_passes(self):
        self.write("checks/publication-policy.toml", "[historical_tasks]\n")
        self.base = self.commit()
        self.write("rsi-logs/task/run.log", "x\n")
        self.commit()
        result = self.guard()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("lists no removed path prefixes", result.stdout)

    def test_invalid_base_policy_fails(self):
        self.write("checks/publication-policy.toml", "[history_guard]\nforbidden_prefixes = [1]\n")
        self.base = self.commit()
        self.write("x.md", "x\n")
        self.commit()
        self.assertNotEqual(self.guard().returncode, 0)


class SecretScannerTests(unittest.TestCase):
    def setUp(self):
        need(
            self, GITLEAKS, "set PUBLICATION_GITLEAKS_BIN for scanner integration tests"
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stage = pub.ScanStage(self.root / "scan")
        self.binary = Path(GITLEAKS)
        self.fake = fake_token()

    def scan(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            count = self.stage.scan(self.binary, HERE / "publication-gitleaks.toml")
        self.assertNotIn(self.fake, output.getvalue())
        return count, output.getvalue()

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
        self.assertEqual(self.scan()[0], 0)

    def test_secret_inside_gzip_detected_without_printing_it(self):
        payload = ('github_token = "' + self.fake + '" # gitleaks:allow\n').encode()
        self.stage.add(io.BytesIO(gzip.compress(payload)), "log.txt.gz")
        count, output = self.scan()
        self.assertGreater(count, 0)
        self.assertIn("::warning::Possible credential: log.txt.gz!log.txt;", output)
        self.assertIn("advisory only", output)

    def test_secret_across_chunk_boundary_detected(self):
        payload = ('github_token = "' + self.fake + '"\n').encode()
        paths = []
        for number, block in enumerate((payload[:20], payload[20:])):
            p = self.root / f"part-{number}.bin"
            p.write_bytes(block)
            paths.append(p)
        with contextlib.closing(pub.PartStream(paths)) as stream:
            self.stage.add(stream, "reassembled.txt")
        self.assertGreater(self.scan()[0], 0)

    def test_default_path_exemptions_do_not_hide_published_files(self):
        names = (
            "rsi-logs/demo/m/part-000001.bin",
            "rsi-logs/demo/summary.pdf",
            "rsi-logs/demo/plot.png",
            "rsi-tasks/demo/environment/node_modules/pkg/a.txt",
            "rsi-tasks/demo/environment/.venv/lib/python3.12/site-packages/x/a.py",
            "rsi-tasks/demo/environment/package-lock.json",
            "rsi-tasks/demo/environment/go.sum",
        )
        for name in names:
            self.stage.add(
                io.BytesIO(f'github_token = "{self.fake}"\n'.encode()),
                name,
                structured=False,
            )
        _, output = self.scan()
        for name in names:
            self.assertIn(f"::warning::Possible credential: {name};", output)

    def test_path_specific_rules_still_apply(self):
        manifest = (
            "apiVersion: v1\nkind: Secret\nmetadata:\n  name: demo\n"
            "data:\n  password: c3VwZXJzZWNyZXRwYXNzd29yZDEyMw==\n"
        ).encode()
        self.stage.add(io.BytesIO(manifest), "deploy/secret.yaml")
        self.stage.add(io.BytesIO(manifest), "deploy/secret.txt")
        _, output = self.scan()
        self.assertIn("deploy/secret.yaml; rule=kubernetes-secret-yaml", output)
        self.assertNotIn("deploy/secret.txt; rule=kubernetes-secret-yaml", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
