#!/usr/bin/env python3
"""Atomically select one complete fixed-trainer trial for submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()
    trial = args.trial.resolve()
    trials = Path("/workspace/trials").resolve()
    if trials not in trial.parents or not (trial / "COMPLETE").is_file():
        raise SystemExit("trial is outside /workspace/trials or is not complete")
    manifest = json.loads((trial / "manifest.json").read_text(encoding="utf-8"))
    staging = Path("/workspace/submission.next")
    destination = Path("/workspace/submission")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    shutil.copytree(trial / "adapter", staging / "adapter")
    (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    previous = Path("/workspace/submission.previous")
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.replace(previous)
    staging.replace(destination)
    if previous.exists():
        shutil.rmtree(previous)


if __name__ == "__main__":
    main()
