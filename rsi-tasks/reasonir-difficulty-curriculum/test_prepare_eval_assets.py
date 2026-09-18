"""CPU-only checks for public asset acquisition, outside the Judge test tree."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_eval_assets as preparer


REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
REMOTE_FILES = {
    **{
        f"beir/{name}.zip": (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        )
        for name in ("arguana", "fiqa", "nfcorpus", "scifact")
    },
    **{
        f"bright/documents/{name}.parquet": (
            f"https://huggingface.co/datasets/xlangai/BRIGHT/resolve/{REVISION}/"
            f"documents/{name}-00000-of-00001.parquet"
        )
        for name in ("earth_science", "psychology")
    },
}


class PrepareAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reasonir-public-assets-test-")
        self.addCleanup(self.temp.cleanup)
        self.task = Path(self.temp.name)
        self.assets = self.task / "tests/assets"
        (self.assets / "bright/documents").mkdir(parents=True)
        self.expected = {
            relative: f"fixed fixture: {relative}\n".encode()
            for relative in REMOTE_FILES
        }
        files = {
            relative: hashlib.sha256(data).hexdigest()
            for relative, data in self.expected.items()
        }
        for index in range(24):
            relative = f"bright/fixture-{index}.parquet"
            data = f"unchanged fixture {index}\n".encode()
            (self.assets / relative).write_bytes(data)
            files[relative] = hashlib.sha256(data).hexdigest()
        (self.assets / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "bright_revision": REVISION, "files": files,
        }))
        self.responses = {
            REMOTE_FILES[relative]: data for relative, data in self.expected.items()
        }

    def response(self, request, timeout):
        return io.BytesIO(self.responses[request.full_url])

    def test_fresh_clone_downloads_six_exact_payloads_and_verifies_all_thirty(self):
        # Removing BRIGHT download support must fail this consumer-visible check.
        with patch.object(preparer, "urlopen", side_effect=self.response):
            try:
                preparer.prepare(self.task)
            except preparer.AssetError as exc:
                self.fail(f"fresh clone cannot prepare all required data: {exc}")
        for relative, expected in self.expected.items():
            self.assertEqual((self.assets / relative).read_bytes(), expected)
        self.assertEqual(len(list(self.assets.rglob("*.parquet"))), 26)
        self.assertEqual(len(list(self.assets.rglob("*.zip"))), 4)
        # All correct existing files must work without network or rewriting.
        before = {r: (self.assets / r).stat().st_mtime_ns for r in self.expected}
        with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
            preparer.prepare(self.task)
        self.assertEqual(before, {r: (self.assets / r).stat().st_mtime_ns for r in self.expected})

    def test_bad_bright_download_installs_none_of_the_missing_files(self):
        self.responses[REMOTE_FILES["bright/documents/psychology.parquet"]] = b"corrupt"
        with patch.object(preparer, "urlopen", side_effect=self.response):
            with self.assertRaisesRegex(preparer.AssetError, "sha256 mismatch"):
                preparer.prepare(self.task)
        for relative in self.expected:
            self.assertFalse((self.assets / relative).exists())

    def test_wrong_existing_bright_file_is_not_overwritten(self):
        path = self.assets / "bright/documents/earth_science.parquet"
        path.write_bytes(b"must not overwrite")
        with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(preparer.AssetError, "sha256 mismatch"):
                preparer.prepare(self.task)
        self.assertEqual(path.read_bytes(), b"must not overwrite")

    def test_symlinked_bright_parent_is_rejected(self):
        parent = self.assets / "bright/documents"
        parent.rmdir()
        outside = self.task / "outside"
        outside.mkdir()
        parent.symlink_to(outside, target_is_directory=True)
        with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(preparer.AssetError, "symlink|real directory"):
                preparer.prepare(self.task)
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
