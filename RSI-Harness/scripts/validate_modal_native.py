"""Opt-in bounded cloud acceptance. All scores are synthetic fixture values."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from rsi_harness.modal_runtime.cloud import launch
from rsi_harness.modal_runtime.config import ModalRunConfig
from rsi_harness.models import CompileOptions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--case",
        choices=("two-rounds", "gpu", "resume", "verifier-error"),
        required=True,
    )
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    config = ModalRunConfig.model_validate_json(args.config.read_text())
    fixture = Path(__file__).resolve().parents[1] / "tests/fixtures/tasks/modal-native"
    with tempfile.TemporaryDirectory(prefix="rsi-modal-acceptance-") as temporary:
        task = Path(temporary) / "task"
        shutil.copytree(fixture, task)
        verify = task / "tests/verify.py"
        if args.case == "gpu":
            if config.gpu is None:
                parser.error("GPU acceptance requires a GPU in config")
            manifest = task / "task.toml"
            manifest.write_text(
                manifest.read_text().replace(
                    "[metadata.rsi_harness.verifier]\ngpus = 0",
                    "[metadata.rsi_harness.verifier]\ngpus = 1",
                )
            )
            verify.write_text(
                verify.read_text() + "\nprint(subprocess.check_output(['nvidia-smi',"
                "'--query-gpu=uuid,name','--format=csv,noheader'], text=True))\n"
                "subprocess.run(['python3','/tests/cuda_probe.py'], check=True)\n"
            )
        elif args.case == "verifier-error":
            verify.write_text(verify.read_text() + "\nraise SystemExit(7)\n")
        elif args.case == "resume":
            if config.segment_seconds != 30:
                parser.error("resume acceptance requires segment_seconds=30")
            (task / "solution/solve.sh").write_text("""#!/bin/bash
set -euo pipefail
if [ ! -f /workspace/checkpoint ]; then
  printf 0 > /workspace/value
  rsi-submit
  printf retained > /workspace/checkpoint
  printf retained > /home/agent/checkpoint
  sleep 90
else
  test "$(cat /home/agent/checkpoint)" = retained
  printf 1 > /workspace/value
  rsi-submit
fi
""")
        receipts = launch(
            task,
            config,
            CompileOptions(
                agent_name="oracle",
                agent_timeout_seconds=120,
                max_submissions=2,
                disable_stop_hook=True,
            ),
            receipt_path=args.receipt,
        )
    result = receipts[-1]["result"]
    assert result["total_rounds"] == 2, result
    if args.case == "verifier-error":
        assert result["best_score"] is None, result
        assert all(
            r["score"] is None and r["status"] == "verifier_error"
            for r in result["reports"]
        ), result
    else:
        assert [r["score"] for r in result["reports"]] == [0.0, 1.0], result
        assert result["best_round"] == "agent-2", result
    if args.case == "resume":
        assert len(receipts) == 2, receipts
        assert len({r["run_id"] for r in receipts}) == 1
        assert len({r["deadline"] for r in receipts}) == 1
    print(
        json.dumps(
            {
                "case": args.case,
                "passed": True,
                "run_id": receipts[-1]["run_id"],
                "synthetic_fixture": True,
            }
        )
    )


if __name__ == "__main__":
    main()
