#!/usr/bin/env python3
"""Compile one policy and launch the only supported candidate-producing run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

RUNTIME = Path("/opt/reasonir-task/runtime")
POOL = Path("/opt/reasonir-task/train/pool.parquet")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--trial", required=True, type=Path)
    args = parser.parse_args()
    policy = args.policy.resolve()
    trial = args.trial.resolve()
    if Path("/workspace/policy").resolve() not in policy.parents or Path("/workspace/trials").resolve() not in trial.parents:
        raise SystemExit("policy/trial must stay under their candidate-owned roots")
    if trial.exists():
        if (trial / "COMPLETE").exists():
            print(f"trial already complete: {trial}")
            return
        if not (trial / "policy.json").is_file():
            raise SystemExit("existing incomplete trial has no fixed policy copy")
    else:
        trial.mkdir(parents=True)
        shutil.copyfile(policy, trial / "policy.json")
        policy_document = json.loads(policy.read_text(encoding="utf-8"))
        if policy_document.get("scores_file"):
            score_source = (policy.parent / policy_document["scores_file"]).resolve()
            if policy.parent.resolve() not in score_source.parents or not score_source.is_file():
                raise SystemExit("scores_file escapes /workspace/policy or is missing")
            score_destination = trial / policy_document["scores_file"]
            score_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(score_source, score_destination)
    subprocess.run(
        [sys.executable, str(RUNTIME / "compile_policy.py"), "--policy", str(trial / "policy.json"), "--pool", str(POOL), "--output", str(trial / "schedule.json")],
        check=True,
    )
    environment = os.environ.copy()
    environment.update({"PYTHONPATH": str(RUNTIME), "PYTHONHASHSEED": "42", "NCCL_ASYNC_ERROR_HANDLING": "1"})
    subprocess.run(
        ["torchrun", "--standalone", "--nproc-per-node=4", str(RUNTIME / "train_trial.py"), "--schedule", str(trial / "schedule.json"), "--pool", str(POOL), "--trial", str(trial)],
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
