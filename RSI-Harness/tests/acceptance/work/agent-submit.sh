#!/bin/bash
# A1, A4, A6: the Work half of A6 (unless RSI_ACCEPTANCE_PROBE=0), then one
# submission; the Judge's fixed procedure (/tests/test.sh) runs the suites
# the operator selected.
set -uo pipefail
root=$HOME/rsi-acceptance
mkdir -p "$root" && tar -xf "$RSI_ACCEPTANCE_FILES" -C "$root"
# shellcheck source=lib.sh
source "$root/work/lib.sh"
if [[ ${RSI_ACCEPTANCE_PROBE:-1} == 1 ]]; then
    probe_work
fi
submit
