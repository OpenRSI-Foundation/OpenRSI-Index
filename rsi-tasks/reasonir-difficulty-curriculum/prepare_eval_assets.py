#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9,<3.14"
# dependencies = ["pyarrow==18.1.0"]
# ///
"""Fetch every locked evaluation asset from its upstream source and verify the inventory.

None of the 30 payloads in tests/assets/manifest.json is stored in Git. The four BEIR
archives come from the official BEIR endpoint. Twenty-two BRIGHT Parquet files are
byte-identical to the pinned Hugging Face revision. The four LeetCode document files are
the pinned upstream LeetCode document Parquet split into four contiguous row slices with
PyArrow 18.1.0, exactly as recorded under ``leetcode_sharding`` in the manifest.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from urllib.request import Request, urlopen


TASK_ROOT = Path(__file__).resolve().parent
UPSTREAM_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASETS = ("arguana", "fiqa", "nfcorpus", "scifact")
ARCHIVES = tuple(f"beir/{name}.zip" for name in DATASETS)
BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT_BASE = f"https://huggingface.co/datasets/xlangai/BRIGHT/resolve/{BRIGHT_REVISION}"
BRIGHT_DOCUMENTS = (
    "aops", "biology", "earth_science", "economics", "pony", "psychology",
    "robotics", "stackoverflow", "sustainable_living", "theoremqa_theorems",
)
BRIGHT_EXAMPLES = (
    "aops", "biology", "earth_science", "economics", "leetcode", "pony", "psychology",
    "robotics", "stackoverflow", "sustainable_living", "theoremqa_questions",
    "theoremqa_theorems",
)
BRIGHT_DOWNLOADS = {
    **{
        f"bright/documents/{name}.parquet": f"{BRIGHT_BASE}/documents/{name}-00000-of-00001.parquet"
        for name in BRIGHT_DOCUMENTS
    },
    **{
        f"bright/examples/{name}.parquet": f"{BRIGHT_BASE}/examples/{name}-00000-of-00001.parquet"
        for name in BRIGHT_EXAMPLES
    },
}
LEETCODE_SOURCE_URL = f"{BRIGHT_BASE}/documents/leetcode-00000-of-00001.parquet"
LEETCODE_SHARDS = tuple(f"bright/documents/leetcode-{index:05d}.parquet" for index in range(4))
LEETCODE_ROW_GROUP_SIZE = 4096
PYARROW_VERSION = "18.1.0"
ASSET_DIRECTORIES = ("beir", "bright", "bright/documents", "bright/examples")
BLOCK_SIZE = 1024 * 1024
DOWNLOAD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 5
STAGING_PREFIX = ".prepare-eval-assets-"


class AssetError(RuntimeError):
    """The locked evaluation assets cannot be prepared safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def load_manifest(task_root: Path) -> tuple[Path, dict[str, str], str]:
    manifest_path = task_root / "tests/assets/manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AssetError(f"asset manifest must be a regular file: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"cannot read asset manifest {manifest_path}: {exc}") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    schema_version = manifest.get("schema_version") if isinstance(manifest, dict) else None
    if schema_version != 1 or not isinstance(files, dict) or len(files) != 30:
        raise AssetError("asset manifest must be schema version 1 with exactly 30 files")
    if manifest.get("bright_revision") != BRIGHT_REVISION:
        raise AssetError("asset manifest BRIGHT revision differs from the pinned download source")
    sharding = manifest.get("leetcode_sharding")
    leetcode_source = sharding.get("source_sha256") if isinstance(sharding, dict) else None
    if not is_sha256(leetcode_source):
        raise AssetError("asset manifest must lock the LeetCode source SHA-256")
    checked: dict[str, str] = {}
    for relative, expected in files.items():
        path = Path(relative) if isinstance(relative, str) else Path(".")
        if (not isinstance(relative, str) or path.is_absolute() or ".." in path.parts
                or path.as_posix() != relative):
            raise AssetError(f"invalid asset manifest path: {relative!r}")
        if not is_sha256(expected):
            raise AssetError(f"invalid SHA-256 in asset manifest for {relative!r}")
        checked[relative] = expected
    required = set(ARCHIVES) | set(BRIGHT_DOWNLOADS) | set(LEETCODE_SHARDS)
    if set(checked) != required:
        raise AssetError(
            "asset manifest differs from the downloadable inventory: "
            f"missing={sorted(required - set(checked))} extra={sorted(set(checked) - required)}"
        )
    return manifest_path.parent, checked, leetcode_source


def verify_inventory(
    asset_root: Path,
    files: dict[str, str],
    *,
    allowed_missing: frozenset[str] = frozenset(),
) -> None:
    actual: set[str] = set()
    for path in asset_root.rglob("*"):
        relative = path.relative_to(asset_root).as_posix()
        if path.is_symlink():
            raise AssetError(f"symlink is not allowed in locked assets: {relative}")
        if path.is_file() and relative != "manifest.json":
            actual.add(relative)
    expected = set(files)
    missing = expected - actual
    added = actual - expected
    unexpected_missing = missing - set(allowed_missing)
    if unexpected_missing or added:
        hint = "; remove files that the manifest does not list" if added else ""
        raise AssetError(
            "asset inventory mismatch: "
            f"missing={sorted(unexpected_missing)} added={sorted(added)}{hint}"
        )
    for relative in sorted(actual & expected):
        path = asset_root / relative
        observed = sha256(path)
        if observed != files[relative]:
            raise AssetError(
                f"sha256 mismatch for {relative}: expected {files[relative]}, "
                f"observed {observed}; delete it and rerun to download a fresh copy"
            )


def download(url: str, destination: Path, expected: str) -> None:
    """Fetch ``url`` into a new file, retrying transfer errors, then check its SHA-256."""
    request = Request(url, headers={"User-Agent": "OpenRSI-Index-eval-asset-preparer/1"})
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        digest = hashlib.sha256()
        try:
            with urlopen(request, timeout=120) as response, destination.open("xb") as output:
                while True:
                    block = response.read(BLOCK_SIZE)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
            break
        except Exception as exc:
            destination.unlink(missing_ok=True)
            if attempt == DOWNLOAD_ATTEMPTS:
                raise AssetError(
                    f"download failed for {url} after {attempt} attempts: {exc}; "
                    "rerun to resume, verified files are kept"
                ) from exc
            print(f"retrying {url} after error: {exc}")
            time.sleep(RETRY_DELAY_SECONDS * attempt)
    observed = digest.hexdigest()
    if observed != expected:
        raise AssetError(
            f"sha256 mismatch for downloaded {destination.name}: "
            f"expected {expected}, observed {observed}"
        )


def require_pyarrow():
    """Return pyarrow.parquet at the exact version that produced the locked shards."""
    try:
        import pyarrow
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise AssetError(
            f"pyarrow=={PYARROW_VERSION} is required to rebuild the LeetCode shards; "
            "run this script with `uv run` or install that exact version"
        ) from exc
    if pyarrow.__version__ != PYARROW_VERSION:
        raise AssetError(
            f"pyarrow=={PYARROW_VERSION} is required to rebuild the LeetCode shards byte for "
            f"byte, found {pyarrow.__version__}; run this script with `uv run`"
        )
    return parquet


def shard_leetcode(source: Path, destinations: list[Path]) -> None:
    """Split the upstream file into contiguous ceil(rows/4) slices, unchanged records."""
    parquet = require_pyarrow()
    table = parquet.read_table(source)
    rows = math.ceil(table.num_rows / len(destinations))
    for index, destination in enumerate(destinations):
        parquet.write_table(
            table.slice(index * rows, rows), destination,
            compression="zstd", row_group_size=LEETCODE_ROW_GROUP_SIZE,
        )


def ensure_directory(path: Path, label: str) -> None:
    if not path.exists() and not path.is_symlink():
        try:
            path.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise AssetError(f"cannot create {label} {path}: {exc}") from exc
    if path.is_symlink() or not path.is_dir():
        raise AssetError(f"{label} must be a real directory: {path}")


def install(staged: Path, asset_root: Path, relative: str) -> None:
    """Move one verified file into place without replacing anything that appeared meanwhile."""
    target = asset_root / relative
    if target.exists() or target.is_symlink():
        raise AssetError(f"refusing to overwrite path created during download: {relative}")
    try:
        os.replace(staged, target)
    except OSError as exc:
        raise AssetError(f"cannot install {relative}: {exc}") from exc
    print(f"installed {relative}")


@contextmanager
def preparation_lock(task_root: Path):
    """Serialize preparation and remove staging left behind by a killed earlier run."""
    # Lock the task folder itself so no lock file is left behind.
    descriptor = os.open(task_root, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise AssetError(f"another prepare_eval_assets.py run is using {task_root}") from exc
        for stale in task_root.glob(f"{STAGING_PREFIX}*"):
            if stale.is_dir() and not stale.is_symlink():
                print(f"removing interrupted staging {stale.name}")
                shutil.rmtree(stale)
        yield
    finally:
        os.close(descriptor)


def prepare(task_root: Path = TASK_ROOT, *, base_url: str = UPSTREAM_BASE) -> None:
    task_root = Path(task_root).resolve()
    tests_root = task_root / "tests"
    expected_asset_root = tests_root / "assets"
    for label, directory in (("tests root", tests_root), ("asset root", expected_asset_root)):
        if directory.is_symlink() or not directory.is_dir():
            raise AssetError(f"{label} must be an existing real directory: {directory}")
    asset_root, files, leetcode_source = load_manifest(task_root)
    if asset_root != expected_asset_root:
        raise AssetError(f"unexpected asset root: {asset_root}")
    with preparation_lock(task_root):
        prepare_locked(task_root, asset_root, files, leetcode_source, base_url)
    verify_inventory(asset_root, files)
    print(f"verified {len(files)} locked evaluation assets")


def prepare_locked(
    task_root: Path,
    asset_root: Path,
    files: dict[str, str],
    leetcode_source: str,
    base_url: str,
) -> None:
    for relative in ASSET_DIRECTORIES:
        ensure_directory(asset_root / relative, "asset destination")

    downloads = {
        **{f"beir/{name}.zip": f"{base_url.rstrip('/')}/{name}.zip" for name in DATASETS},
        **BRIGHT_DOWNLOADS,
    }

    missing: list[str] = []
    for relative in (*downloads, *LEETCODE_SHARDS):
        target = asset_root / relative
        if target.is_symlink():
            raise AssetError(f"refusing to overwrite symlink: {relative}")
        if target.exists():
            if not target.is_file():
                raise AssetError(f"refusing to overwrite non-file: {relative}")
        else:
            missing.append(relative)

    verify_inventory(asset_root, files, allowed_missing=frozenset(missing))
    if not missing:
        return
    rebuild_leetcode = any(relative in LEETCODE_SHARDS for relative in missing)
    if rebuild_leetcode:
        require_pyarrow()  # Fail before any network traffic.

    # Stage beside tests/assets (same filesystem, outside the hashed inventory). Each file
    # is installed as soon as it matches the manifest, so a failed run keeps its progress.
    with tempfile.TemporaryDirectory(prefix=STAGING_PREFIX, dir=task_root) as raw:
        staging = Path(raw)
        for relative in missing:
            if relative not in downloads:
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = downloads[relative]
            print(f"downloading {url}")
            download(url, destination, files[relative])
            install(destination, asset_root, relative)
        if rebuild_leetcode:
            source = staging / "leetcode-source.parquet"
            print(f"downloading {LEETCODE_SOURCE_URL}")
            download(LEETCODE_SOURCE_URL, source, leetcode_source)
            shards = [staging / relative for relative in LEETCODE_SHARDS]
            shards[0].parent.mkdir(parents=True, exist_ok=True)
            shard_leetcode(source, shards)
            source.unlink()
            # The four shards are one rebuild; install them only if all of them match.
            for relative, shard in zip(LEETCODE_SHARDS, shards):
                observed = sha256(shard)
                if observed != files[relative]:
                    raise AssetError(
                        f"sha256 mismatch for rebuilt {relative}: "
                        f"expected {files[relative]}, observed {observed}; "
                        f"the rebuild needs pyarrow=={PYARROW_VERSION} and was confirmed "
                        "on Linux x86_64"
                    )
            for relative, shard in zip(LEETCODE_SHARDS, shards):
                if relative in missing:
                    install(shard, asset_root, relative)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download four BEIR archives and the pinned BRIGHT Parquet files, rebuild the four "
            "LeetCode document shards, and verify all 30 locked assets after you review "
            "THIRD_PARTY.md and the upstream dataset terms."
        )
    )
    parser.add_argument(
        "--accept-dataset-terms",
        action="store_true",
        help=(
            "confirm that you reviewed THIRD_PARTY.md and the dataset terms linked "
            "there before downloading from the official BEIR and Hugging Face sources"
        ),
    )
    args = parser.parse_args()
    if not args.accept_dataset_terms:
        parser.error(
            "explicit --accept-dataset-terms is required; read THIRD_PARTY.md and "
            "the upstream terms before downloading from the official BEIR and Hugging Face sources"
        )
    try:
        prepare()
    except AssetError as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
