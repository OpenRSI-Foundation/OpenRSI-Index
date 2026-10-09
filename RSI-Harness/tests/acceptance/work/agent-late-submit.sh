#!/bin/bash
# A8: submit about 60 s before the Work deadline, so the Judge round (its own
# verifier timeout) runs across the Work deadline and must still finish with
# a reward. The Harness starts the Work deadline ($WORK_TIMEOUT_SEC, the
# run's --timeout) as it starts this script: the first thing read here.
started=$(date +%s)
set -uo pipefail
root=$HOME/rsi-acceptance
mkdir -p "$root" && tar -xf "$RSI_ACCEPTANCE_FILES" -C "$root"
# shellcheck source=lib.sh
source "$root/work/lib.sh"
note "work-deadline $((started + WORK_TIMEOUT_SEC))"
sleep "$(( WORK_TIMEOUT_SEC - 60 - ($(date +%s) - started) ))"
note "late-submit $(date +%s)"
submit
sleep 3600
