#!/bin/bash
# A7 (build, load): build fix-git from its Dockerfile through the Work
# builder; the operator script kills the harness mid-build or mid-load.
set -uo pipefail
root=$HOME/rsi-acceptance
mkdir -p "$root" && tar -xf "$RSI_ACCEPTANCE_FILES" -C "$root"
# shellcheck source=lib.sh
source "$root/work/lib.sh"
rsi-sandbox build "$root/tests/fix-git-build/environment" --timeout 1800
note "build-exit $?"
sleep 3600
