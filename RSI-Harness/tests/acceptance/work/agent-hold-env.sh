#!/bin/bash
# A7 (env_create, paused): bring up a two-service env and keep it; with
# HOLD_SUBMIT=1 also submit, so a Judge round runs while the env is paused.
# The operator script kills the harness at the chosen moment.
set -uo pipefail
root=$HOME/rsi-acceptance
mkdir -p "$root" && tar -xf "$RSI_ACCEPTANCE_FILES" -C "$root"
# shellcheck source=lib.sh
source "$root/work/lib.sh"
rsi-sandbox compose -f "$root/work/hold-compose.yaml" --network none up
note "hold-up $?"
if [[ ${HOLD_SUBMIT:-0} == 1 ]]; then
    submit
fi
sleep 3600
