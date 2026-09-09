#!/usr/bin/env bash
set -uo pipefail
# Container defaults; the GPIC_* overrides exist for containerless
# infrastructure smoke testing only (the harness never sets them).
LOG_DIR="${GPIC_VERIFIER_LOG_DIR:-/logs/verifier}"
TESTS_DIR="${GPIC_TESTS_DIR:-/tests}"
POLICY_PY="${GPIC_POLICY_PYTHON:-/opt/policy-venv/bin/python}"
CLEAN_ROOT="${GPIC_CLEAN_ROOT:-/opt/project}"
CANDIDATE_ROOT="${GPIC_CANDIDATE_ROOT:-/app/project}"
OUTPUT_ROOT="${GPIC_OUTPUT_ROOT:-/app/output}"
mkdir -p "$LOG_DIR"

zero() {
  LOG_DIR="$LOG_DIR" python3 - "$1" <<'PY'
import json, os, sys
log_dir = os.environ["LOG_DIR"]
result = {"reward": 0.0, "policy_gate": 0.0}
failure = {"reason": sys.argv[1]}
open(f"{log_dir}/reward.json", "w").write(json.dumps(result, indent=2) + "\n")
open(f"{log_dir}/failure.json", "w").write(json.dumps(failure, indent=2) + "\n")
open(f"{log_dir}/reward.txt", "w").write("0.0\n")
PY
}

if ! "$POLICY_PY" "$TESTS_DIR/policy_check.py" \
    --policy "$TESTS_DIR/policy.yaml" --clean "$CLEAN_ROOT" \
    --candidate "$CANDIDATE_ROOT" --output-dir "$OUTPUT_ROOT" \
    --report "$LOG_DIR/policy_check.json"; then
  zero reward_integrity_gate
  exit 0
fi
# GPIC_VERIFIER_SMOKE=1 restricts evaluate.py to submission validation, FD on
# the first 1024 images, and a 64-key spot regeneration — infrastructure
# debugging only; score.py refuses smoke metrics.
if ! python3 "$TESTS_DIR/evaluate.py"; then
  zero frozen_evaluation_failed
  exit 0
fi
cp "$OUTPUT_ROOT/verifier-metrics.json" "$LOG_DIR/metrics.json"
python3 "$TESTS_DIR/score.py"
