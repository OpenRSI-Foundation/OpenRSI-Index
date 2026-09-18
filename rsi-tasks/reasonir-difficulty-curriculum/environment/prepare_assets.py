#!/usr/bin/env python3
"""Build immutable model and train-only assets from the locked public pins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download
import pyarrow as pa
import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"sha256 mismatch for {path}: expected {expected}, observed {actual}")


def download_file(section: dict, name: str, destination: Path) -> Path:
    cached = Path(hf_hub_download(repo_id=section["repo_id"], repo_type=section["repo_type"], revision=section["revision"], filename=name))
    require_hash(cached, section["files"][name])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cached, destination)
    require_hash(destination, section["files"][name])
    return destination


def pair(value, context: str) -> dict[str, str]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(isinstance(item, str) for item in value):
        raise RuntimeError(f"invalid instruction/text pair in {context}")
    return {"instruction": value[0], "text": value[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--reasonir-source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise RuntimeError("unsupported asset lock")
    base = args.destination / "base" / "ReasonIR-8B"
    snapshot_download(
        repo_id=lock["model"]["repo_id"],
        repo_type="model",
        revision=lock["model"]["revision"],
        local_dir=base,
        allow_patterns=sorted(lock["model"]["files"]),
    )
    for name, expected in lock["model"]["files"].items():
        require_hash(base / name, expected)

    train = args.destination / "train"
    pool_source = train / "upstream"
    pool_source.mkdir(parents=True)
    hq = download_file(lock["training"], "hq/train-00000-of-00001.parquet", pool_source / "hq.parquet")
    vl = download_file(lock["training"], "vl/train-00000-of-00001.parquet", pool_source / "vl.parquet")
    hq_rows = pq.read_table(hq).to_pylist()
    wanted: set[str] = set()
    for index, row in enumerate(hq_rows):
        for positive in row["pos"]:
            parsed = pair(positive, f"hq[{index}].pos")
            if not parsed["text"]:
                raise RuntimeError(f"empty HQ positive ID at row {index}")
            wanted.add(parsed["text"])

    resolved: dict[str, str] = {}
    origins: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="reasonir-bright-") as temporary:
        scratch = Path(temporary)
        for name, expected in lock["bright_documents"]["files"].items():
            local = download_file(lock["bright_documents"], name, scratch / Path(name).name)
            table = pq.read_table(local, columns=["id", "content"])
            for row in table.to_pylist():
                identifier = row["id"]
                if identifier not in wanted:
                    continue
                content = row["content"]
                if identifier in resolved:
                    raise RuntimeError(f"conflicting or duplicate HQ positive ID {identifier!r}: {origins[identifier]} and {name}")
                resolved[identifier] = content
                origins[identifier] = name
    missing = sorted(wanted - set(resolved))
    if missing:
        raise RuntimeError(f"missing {len(missing)} exact HQ positive IDs; first IDs: {missing[:20]}")

    normalized = []
    for source, rows in (("hq", hq_rows), ("vl", pq.read_table(vl).to_pylist())):
        for index, row in enumerate(rows):
            query = pair(row["query"], f"{source}[{index}].query")
            positives = [pair(item, f"{source}[{index}].pos") for item in row["pos"]]
            negatives = [pair(item, f"{source}[{index}].neg") for item in row["neg"]]
            if source == "hq":
                for positive in positives:
                    positive["text"] = resolved[positive["text"]]
            normalized.append({
                "row_id": f"{source}:{index:06d}",
                "source": source,
                "query_instruction": query["instruction"],
                "query": query["text"],
                "positive": positives,
                "negative": negatives,
            })
    pool = train / "pool.parquet"
    pq.write_table(pa.Table.from_pylist(normalized), pool, compression="zstd", use_dictionary=True, row_group_size=4096)
    resolution = train / "hq-positive-resolution.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": key, "content": resolved[key], "source_blob": origins[key]} for key in sorted(resolved)]), resolution, compression="zstd", use_dictionary=True)
    manifest = {
        "schema_version": 1,
        "reasonir_data_revision": lock["training"]["revision"],
        "bright_revision": lock["bright_documents"]["revision"],
        "hq_rows": len(hq_rows),
        "vl_rows": len(normalized) - len(hq_rows),
        "resolved_unique_hq_positive_ids": len(resolved),
        "files": {"pool.parquet": sha256(pool), "hq-positive-resolution.parquet": sha256(resolution)},
    }
    (train / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(pool_source)

    zero = args.destination / "base" / "zero-adapter"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(args.destination / "runtime")
    subprocess.run([sys.executable, str(args.destination / "runtime" / "create_zero_adapter.py"), "--output", str(zero)], check=True, env=environment)
    for directory in [base, train, args.destination / "runtime"]:
        for path in directory.rglob("*"):
            if path.is_dir():
                path.chmod(0o555)
            elif path.is_file():
                path.chmod(0o444)


if __name__ == "__main__":
    main()
