#!/usr/bin/env python3
"""Compile bounded declarative difficulty metadata into the fixed 64k schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

UPDATES = 1000
EXPOSURES_PER_UPDATE = 64


def fail(message: str) -> None:
    raise SystemExit(f"policy_invalid: {message}")


def load_scores(policy_path: Path, policy: dict, known: set[str]) -> dict[str, dict]:
    name = policy.get("scores_file")
    if not name:
        return {}
    path = (policy_path.parent / name).resolve()
    if policy_path.parent.resolve() not in path.parents:
        fail("scores_file escapes /workspace/policy")
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        fail("scores_file is missing or exceeds 256 MiB")
    values: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) - {"row_id", "difficulty", "weight", "bucket", "include"}:
                    raise ValueError("unknown score fields")
                row_id = row["row_id"]
                difficulty = float(row.get("difficulty", 0.0))
                weight = float(row.get("weight", 1.0))
                bucket = int(row.get("bucket", 0))
                include = bool(row.get("include", True))
            except Exception as exc:
                fail(f"scores_file line {line_number}: {exc}")
            if row_id not in known or row_id in values:
                fail(f"scores_file line {line_number}: unknown or duplicate row_id")
            if not np.isfinite(difficulty) or not np.isfinite(weight) or weight < 0 or not 0 <= bucket <= 99:
                fail(f"scores_file line {line_number}: invalid difficulty/weight/bucket")
            values[row_id] = {"difficulty": difficulty, "weight": weight, "bucket": bucket, "include": include}
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--pool", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.policy.read_bytes()
    if len(raw) > 1024 * 1024:
        fail("policy.json exceeds 1 MiB")
    policy = json.loads(raw)
    required = {"version", "name", "method", "include_sources", "replacement", "seed"}
    allowed = required | {"scores_file", "curriculum_phases"}
    if set(policy) - allowed or not required <= set(policy):
        fail("policy has missing or unknown fields")
    if policy["version"] != 1 or policy["seed"] != 42 or policy["replacement"] is not True:
        fail("version, seed, or replacement differs from the fixed contract")
    if not isinstance(policy["name"], str) or not 1 <= len(policy["name"]) <= 80 or not all(character.isalnum() or character in "._-" for character in policy["name"]):
        fail("name must be 1-80 safe characters")
    if policy["method"] not in {"uniform", "weighted", "easy-to-hard", "hard-to-easy", "bucket-curriculum"}:
        fail("unsupported method")
    table = pq.read_table(args.pool, columns=["row_id", "source"])
    ids = table.column("row_id").to_pylist()
    sources = table.column("source").to_pylist()
    if len(ids) != len(set(ids)):
        fail("fixed pool contains duplicate row identities")
    include_sources = set(policy["include_sources"])
    if not include_sources or not include_sources <= {"hq", "vl"}:
        fail("include_sources must be a nonempty subset of hq/vl")
    scores = load_scores(args.policy, policy, set(ids))
    difficulty = np.array([scores.get(x, {}).get("difficulty", 0.0) for x in ids], dtype=np.float64)
    weight = np.array([scores.get(x, {}).get("weight", 1.0) for x in ids], dtype=np.float64)
    bucket = np.array([scores.get(x, {}).get("bucket", 0) for x in ids], dtype=np.int64)
    include = np.array([sources[i] in include_sources and scores.get(ids[i], {}).get("include", True) for i in range(len(ids))])
    rng = np.random.default_rng(42)
    schedule: list[int] = []
    phases = policy.get("curriculum_phases") or [{"until_update": UPDATES, "min_bucket": 0, "max_bucket": 99}]
    if not isinstance(phases, list) or not 1 <= len(phases) <= UPDATES or any(not isinstance(phase, dict) or set(phase) != {"until_update", "min_bucket", "max_bucket"} for phase in phases):
        fail("curriculum phases have invalid structure")
    if phases[-1].get("until_update") != UPDATES:
        fail("last curriculum phase must end at update 1000")
    previous = 0
    for phase in phases:
        end = int(phase["until_update"])
        lo, hi = int(phase["min_bucket"]), int(phase["max_bucket"])
        if not previous < end <= UPDATES or not 0 <= lo <= hi <= 99:
            fail("curriculum phases must be increasing with valid bucket bounds")
        mask = include & (bucket >= lo) & (bucket <= hi)
        indexes = np.flatnonzero(mask)
        if not len(indexes):
            fail("a curriculum phase selects no rows")
        probabilities = np.ones(len(indexes), dtype=np.float64) if policy["method"] == "uniform" else weight[indexes].copy()
        if policy["method"] == "easy-to-hard":
            probabilities *= np.exp(-difficulty[indexes])
        elif policy["method"] == "hard-to-easy":
            probabilities *= np.exp(difficulty[indexes])
        if not np.isfinite(probabilities).all() or probabilities.sum() <= 0:
            fail("a curriculum phase has no finite positive sampling mass")
        probabilities /= probabilities.sum()
        count = (end - previous) * EXPOSURES_PER_UPDATE
        schedule.extend(rng.choice(indexes, size=count, replace=True, p=probabilities).tolist())
        previous = end
    if len(schedule) != UPDATES * EXPOSURES_PER_UPDATE:
        fail("compiled schedule has wrong exposure count")
    output = {
        "version": 1,
        "policy_sha256": hashlib.sha256(raw).hexdigest(),
        "pool_rows": len(ids),
        "updates": UPDATES,
        "exposures_per_update": EXPOSURES_PER_UPDATE,
        "schedule": schedule,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, separators=(",", ":")), encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
