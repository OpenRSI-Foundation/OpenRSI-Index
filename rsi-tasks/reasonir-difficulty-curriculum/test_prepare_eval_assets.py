"""CPU-only checks for public asset acquisition, outside the Judge test tree."""

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_eval_assets as preparer


REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT = f"https://huggingface.co/datasets/xlangai/BRIGHT/resolve/{REVISION}"
REMOTE_FILES = {
    **{
        f"beir/{name}.zip": (
            f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
        )
        for name in ("arguana", "fiqa", "nfcorpus", "scifact")
    },
    **{
        f"bright/documents/{name}.parquet": f"{BRIGHT}/documents/{name}-00000-of-00001.parquet"
        for name in (
            "aops", "biology", "earth_science", "economics", "pony", "psychology",
            "robotics", "stackoverflow", "sustainable_living", "theoremqa_theorems",
        )
    },
    **{
        f"bright/examples/{name}.parquet": f"{BRIGHT}/examples/{name}-00000-of-00001.parquet"
        for name in (
            "aops", "biology", "earth_science", "economics", "leetcode", "pony",
            "psychology", "robotics", "stackoverflow", "sustainable_living",
            "theoremqa_questions", "theoremqa_theorems",
        )
    },
}
LEETCODE_URL = f"{BRIGHT}/documents/leetcode-00000-of-00001.parquet"
LEETCODE_SHARDS = tuple(f"bright/documents/leetcode-{index:05d}.parquet" for index in range(4))


def pyarrow_version():
    try:
        import pyarrow
    except ImportError:
        return None
    return pyarrow.__version__


class PrepareAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="reasonir-public-assets-test-")
        self.addCleanup(self.temp.cleanup)
        self.task = Path(self.temp.name)
        self.assets = self.task / "tests/assets"
        self.assets.mkdir(parents=True)
        self.expected = {
            relative: f"fixed fixture: {relative}\n".encode()
            for relative in (*REMOTE_FILES, *LEETCODE_SHARDS)
        }
        self.leetcode_source = b"fixed upstream leetcode fixture\n"
        files = {
            relative: hashlib.sha256(data).hexdigest()
            for relative, data in self.expected.items()
        }
        (self.assets / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "bright_revision": REVISION, "files": files,
            "leetcode_sharding": {
                "source_sha256": hashlib.sha256(self.leetcode_source).hexdigest(),
            },
        }))
        self.responses = {
            REMOTE_FILES[relative]: self.expected[relative] for relative in REMOTE_FILES
        }
        self.responses[LEETCODE_URL] = self.leetcode_source
        self.requests = []
        # The real shard step needs pyarrow==18.1.0; these tests check the control flow.
        patcher = patch.object(preparer, "require_pyarrow", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(preparer, "shard_leetcode", side_effect=self.fake_shard)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(preparer.time, "sleep")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.failures = {}

    def response(self, request, timeout):
        self.requests.append(request.full_url)
        if self.failures.get(request.full_url, 0):
            self.failures[request.full_url] -= 1
            raise OSError("simulated connection reset")
        return io.BytesIO(self.responses[request.full_url])

    def fake_shard(self, source, destinations):
        self.assertEqual(source.read_bytes(), self.leetcode_source)
        self.assertEqual(len(destinations), len(LEETCODE_SHARDS))
        for relative, destination in zip(LEETCODE_SHARDS, destinations):
            destination.write_bytes(self.expected[relative])

    def installed(self):
        return {
            p.relative_to(self.assets).as_posix()
            for p in self.assets.rglob("*") if p.is_file() and p.name != "manifest.json"
        }

    def test_real_manifest_locks_exactly_the_downloadable_inventory(self):
        _, files, source = preparer.load_manifest(preparer.TASK_ROOT)
        self.assertEqual(set(files), set(REMOTE_FILES) | set(LEETCODE_SHARDS))
        self.assertEqual(
            preparer.BRIGHT_DOWNLOADS,
            {r: u for r, u in REMOTE_FILES.items() if r.startswith("bright/")},
        )
        self.assertEqual(preparer.LEETCODE_SOURCE_URL, LEETCODE_URL)
        self.assertEqual(source, "a31aa55116b5ea9e31cd5ca6117a5822a9ba64d1e63dda463b0908859cbba240")

    def test_fresh_clone_downloads_everything_and_verifies_all_thirty(self):
        with patch.object(preparer, "urlopen", side_effect=self.response):
            try:
                preparer.prepare(self.task)
            except preparer.AssetError as exc:
                self.fail(f"fresh clone cannot prepare all required data: {exc}")
        self.assertEqual(sorted(self.requests), sorted([*REMOTE_FILES.values(), LEETCODE_URL]))
        for relative, expected in self.expected.items():
            self.assertEqual((self.assets / relative).read_bytes(), expected)
        self.assertEqual(len(list(self.assets.rglob("*.parquet"))), 26)
        self.assertEqual(len(list(self.assets.rglob("*.zip"))), 4)
        self.assertEqual(sorted(p.name for p in self.assets.iterdir()),
                         ["beir", "bright", "manifest.json"])
        # All correct existing files must work without network or rewriting.
        before = {r: (self.assets / r).stat().st_mtime_ns for r in self.expected}
        with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
            preparer.prepare(self.task)
        self.assertEqual(before, {r: (self.assets / r).stat().st_mtime_ns for r in self.expected})

    def test_one_missing_leetcode_shard_is_rebuilt_without_touching_the_others(self):
        with patch.object(preparer, "urlopen", side_effect=self.response):
            preparer.prepare(self.task)
        (self.assets / LEETCODE_SHARDS[2]).unlink()
        before = {r: (self.assets / r).stat().st_mtime_ns for r in LEETCODE_SHARDS if r != LEETCODE_SHARDS[2]}
        self.requests.clear()
        with patch.object(preparer, "urlopen", side_effect=self.response):
            preparer.prepare(self.task)
        self.assertEqual(self.requests, [LEETCODE_URL])
        self.assertEqual((self.assets / LEETCODE_SHARDS[2]).read_bytes(), self.expected[LEETCODE_SHARDS[2]])
        self.assertEqual(before, {r: (self.assets / r).stat().st_mtime_ns for r in before})

    def test_bad_download_is_never_installed_and_verified_files_are_kept(self):
        bad = "bright/examples/psychology.parquet"
        self.responses[REMOTE_FILES[bad]] = b"corrupt"
        with patch.object(preparer, "urlopen", side_effect=self.response):
            with self.assertRaisesRegex(preparer.AssetError, "sha256 mismatch"):
                preparer.prepare(self.task)
        order = list(REMOTE_FILES)
        kept = set(order[:order.index(bad)])
        self.assertEqual(self.installed(), kept)
        for relative in kept:
            self.assertEqual((self.assets / relative).read_bytes(), self.expected[relative])
        self.assertEqual(list(self.task.glob(".prepare-eval-assets-*")), [])
        # A rerun fetches only what is still missing.
        self.responses[REMOTE_FILES[bad]] = self.expected[bad]
        self.requests.clear()
        with patch.object(preparer, "urlopen", side_effect=self.response):
            preparer.prepare(self.task)
        self.assertEqual(
            sorted(self.requests),
            sorted([REMOTE_FILES[r] for r in order if r not in kept] + [LEETCODE_URL]),
        )
        self.assertEqual(self.installed(), set(self.expected))

    def test_transient_errors_are_retried_and_persistent_ones_keep_progress(self):
        flaky = REMOTE_FILES["bright/documents/pony.parquet"]
        self.failures[flaky] = preparer.DOWNLOAD_ATTEMPTS - 1
        with patch.object(preparer, "urlopen", side_effect=self.response):
            preparer.prepare(self.task)
        self.assertEqual(self.requests.count(flaky), preparer.DOWNLOAD_ATTEMPTS)
        self.assertEqual(self.installed(), set(self.expected))

        for relative in ("beir/fiqa.zip", "bright/documents/pony.parquet"):
            (self.assets / relative).unlink()
        self.failures[flaky] = preparer.DOWNLOAD_ATTEMPTS
        with patch.object(preparer, "urlopen", side_effect=self.response):
            with self.assertRaisesRegex(preparer.AssetError, "after 3 attempts.*rerun to resume"):
                preparer.prepare(self.task)
        self.assertEqual(self.installed(), set(self.expected) - {"bright/documents/pony.parquet"})

    def test_staging_left_by_a_killed_run_is_removed_and_ignored(self):
        stale = self.task / ".prepare-eval-assets-killed/beir"
        stale.mkdir(parents=True)
        (stale / "fiqa.zip").write_bytes(b"partial")
        with patch.object(preparer, "urlopen", side_effect=self.response):
            preparer.prepare(self.task)
        self.assertFalse(stale.parent.exists())
        self.assertEqual(self.installed(), set(self.expected))

    def test_unlisted_file_in_assets_names_the_fix(self):
        extra = self.assets / ".prepare-eval-assets-old/beir/fiqa.zip"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"partial")
        with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(preparer.AssetError, "remove files that the manifest does not list"):
                preparer.prepare(self.task)

    def test_concurrent_run_is_refused(self):
        descriptor = os.open(self.task, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
                with self.assertRaisesRegex(preparer.AssetError, "another prepare_eval_assets.py run"):
                    preparer.prepare(self.task)
        finally:
            os.close(descriptor)
        self.assertEqual(sorted(p.name for p in self.task.iterdir()), ["tests"])

    def test_bad_leetcode_source_or_shard_installs_no_shard(self):
        for corrupt in ("source", "shard"):
            with self.subTest(corrupt=corrupt):
                if corrupt == "source":
                    self.responses[LEETCODE_URL] = b"corrupt"
                else:
                    self.responses[LEETCODE_URL] = self.leetcode_source
                    self.expected[LEETCODE_SHARDS[3]] = b"different bytes"
                with patch.object(preparer, "urlopen", side_effect=self.response):
                    with self.assertRaisesRegex(preparer.AssetError, "sha256 mismatch"):
                        preparer.prepare(self.task)
                self.assertEqual(self.installed(), set(REMOTE_FILES))

    def test_wrong_pyarrow_fails_before_any_download(self):
        with patch.object(preparer, "require_pyarrow", side_effect=preparer.AssetError("pyarrow")):
            with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
                with self.assertRaisesRegex(preparer.AssetError, "pyarrow"):
                    preparer.prepare(self.task)

    def test_wrong_existing_bright_file_is_not_overwritten(self):
        path = self.assets / "bright/documents/earth_science.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"must not overwrite")
        with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(preparer.AssetError,
                                        "sha256 mismatch.*delete it and rerun"):
                preparer.prepare(self.task)
        self.assertEqual(path.read_bytes(), b"must not overwrite")

    def test_symlinked_bright_parent_is_rejected(self):
        outside = self.task / "outside"
        outside.mkdir()
        (self.assets / "bright").mkdir()
        (self.assets / "bright/examples").symlink_to(outside, target_is_directory=True)
        with patch.object(preparer, "urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaisesRegex(preparer.AssetError, "symlink|real directory"):
                preparer.prepare(self.task)
        self.assertEqual(list(outside.iterdir()), [])


@unittest.skipUnless(pyarrow_version() == preparer.PYARROW_VERSION,
                     f"needs pyarrow=={preparer.PYARROW_VERSION} (run with `uv run`)")
class LeetCodeShardTest(unittest.TestCase):
    def test_shards_are_contiguous_unchanged_zstd_slices(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory(prefix="reasonir-leetcode-shard-test-") as raw:
            root = Path(raw)
            table = pa.table({"id": [str(i) for i in range(10)],
                              "content": [f"doc {i}" for i in range(10)]})
            pq.write_table(table, root / "source.parquet", compression="snappy")
            shards = [root / f"shard-{index}.parquet" for index in range(4)]
            preparer.shard_leetcode(root / "source.parquet", shards)
            tables = [pq.read_table(shard) for shard in shards]
            self.assertEqual([t.num_rows for t in tables], [3, 3, 3, 1])
            self.assertTrue(pa.concat_tables(tables).equals(table))
            metadata = pq.ParquetFile(shards[0]).metadata
            self.assertEqual(metadata.row_group(0).column(0).compression, "ZSTD")
            self.assertEqual(metadata.created_by, "parquet-cpp-arrow version 18.1.0")


if __name__ == "__main__":
    unittest.main()
