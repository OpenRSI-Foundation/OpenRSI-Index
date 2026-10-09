#!/bin/bash
# A2, A3: the Judge's own Harbor commands (the sample's run_suites.sh) in
# Work before submitting, over $RSI_HARBOR_SUITES.
set -uo pipefail
root=$HOME/rsi-acceptance
mkdir -p "$root" && tar -xf "$RSI_ACCEPTANCE_FILES" -C "$root"
# shellcheck source=lib.sh
source "$root/work/lib.sh"
bash "$root/tests/run_suites.sh" "$root/jobs" "$root/reward.json" \
    "$root/summary.json"
note "work-reward $(cat "$root/reward.json")"
python3 - "$root/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1]))
for suite, report in summary.get("suites", {}).items():
    oracle, nop = report["oracle"], report["nop"]
    print(f"RSI-ACCEPTANCE work-suite {suite} oracle={oracle['ok']} nop={nop['ok']}")
PY
submit
