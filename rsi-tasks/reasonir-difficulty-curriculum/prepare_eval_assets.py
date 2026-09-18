#!/usr/bin/env python3
"""Fetch six upstream evaluation assets and verify the complete locked inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen


TASK_ROOT = Path(__file__).resolve().parent
UPSTREAM_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"
DATASETS = ("arguana", "fiqa", "nfcorpus", "scifact")
ARCHIVES = tuple(f"beir/{name}.zip" for name in DATASETS)
BRIGHT_REVISION = "3066d29c9651a576c8aba4832d249807b181ecae"
BRIGHT_DOWNLOADS = {
    f"bright/documents/{name}.parquet": (
        "https://huggingface.co/datasets/xlangai/BRIGHT/resolve/"
        f"{BRIGHT_REVISION}/documents/{name}-00000-of-00001.parquet"
    )
    for name in ("earth_science", "psychology")
}
BLOCK_SIZE = 1024 * 1024


class AssetError(RuntimeError):
    """The locked evaluation assets cannot be prepared safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(task_root: Path) -> tuple[Path, dict[str, str]]:
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
    checked: dict[str, str] = {}
    for relative, expected in files.items():
        path = Path(relative) if isinstance(relative, str) else Path(".")
        if (not isinstance(relative, str) or path.is_absolute() or ".." in path.parts
                or path.as_posix() != relative):
            raise AssetError(f"invalid asset manifest path: {relative!r}")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise AssetError(f"invalid SHA-256 in asset manifest for {relative!r}")
        checked[relative] = expected
    missing_locks = sorted((set(ARCHIVES) | set(BRIGHT_DOWNLOADS)) - set(checked))
    if missing_locks:
        raise AssetError(f"asset manifest is missing required download locks: {missing_locks}")
    return manifest_path.parent, checked


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
        raise AssetError(
            "asset inventory mismatch: "
            f"missing={sorted(unexpected_missing)} added={sorted(added)}"
        )
    for relative in sorted(actual & expected):
        path = asset_root / relative
        observed = sha256(path)
        if observed != files[relative]:
            raise AssetError(
                f"sha256 mismatch for {relative}: expected {files[relative]}, "
                f"observed {observed}"
            )


def download(url: str, destination: Path, expected: str) -> None:
    request = Request(url, headers={"User-Agent": "OpenRSI-Index-eval-asset-preparer/1"})
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=120) as response, destination.open("xb") as output:
            while True:
                block = response.read(BLOCK_SIZE)
                if not block:
                    break
                output.write(block)
                digest.update(block)
    except Exception as exc:
        raise AssetError(f"download failed for {url}: {exc}") from exc
    observed = digest.hexdigest()
    if observed != expected:
        raise AssetError(
            f"sha256 mismatch for downloaded {destination.name}: "
            f"expected {expected}, observed {observed}"
        )


def prepare(task_root: Path = TASK_ROOT, *, base_url: str = UPSTREAM_BASE) -> None:
    task_root = Path(task_root).resolve()
    tests_root = task_root / "tests"
    expected_asset_root = tests_root / "assets"
    for label, directory in (("tests root", tests_root), ("asset root", expected_asset_root)):
        if directory.is_symlink() or not directory.is_dir():
            raise AssetError(f"{label} must be an existing real directory: {directory}")
    asset_root, files = load_manifest(task_root)
    if asset_root != expected_asset_root:
        raise AssetError(f"unexpected asset root: {asset_root}")
    beir_root = asset_root / "beir"
    if not beir_root.exists() and not beir_root.is_symlink():
        try:
            beir_root.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise AssetError(f"cannot create BEIR destination {beir_root}: {exc}") from exc
    if beir_root.is_symlink() or not beir_root.is_dir():
        raise AssetError(f"BEIR destination must be a real directory: {beir_root}")
    for relative in ("bright", "bright/documents"):
        directory = asset_root / relative
        if directory.is_symlink() or not directory.is_dir():
            raise AssetError(f"BRIGHT destination must be an existing real directory: {directory}")

    downloads = {
        **{f"beir/{name}.zip": f"{base_url.rstrip('/')}/{name}.zip" for name in DATASETS},
        **BRIGHT_DOWNLOADS,
    }

    missing: list[str] = []
    for relative in downloads:
        target = asset_root / relative
        if target.is_symlink():
            raise AssetError(f"refusing to overwrite symlink: {relative}")
        if target.exists():
            if not target.is_file():
                raise AssetError(f"refusing to overwrite non-file: {relative}")
        else:
            missing.append(relative)

    verify_inventory(asset_root, files, allowed_missing=frozenset(missing))
    if missing:
        staged: dict[str, Path] = {}
        with tempfile.TemporaryDirectory(prefix=".prepare-eval-assets-", dir=beir_root) as raw:
            staging = Path(raw)
            for relative in missing:
                destination = staging / Path(relative).name
                url = downloads[relative]
                print(f"downloading {url}")
                download(url, destination, files[relative])
                staged[relative] = destination

            # Recheck every destination before installing any staged file.
            for relative in missing:
                target = asset_root / relative
                if target.exists() or target.is_symlink():
                    raise AssetError(f"refusing to overwrite path created during download: {relative}")
            for relative in missing:
                os.replace(staged[relative], asset_root / relative)
                print(f"installed {relative}")

    verify_inventory(asset_root, files)
    print(f"verified {len(files)} locked evaluation assets")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download four BEIR archives and two pinned BRIGHT Parquet files after you review "
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
