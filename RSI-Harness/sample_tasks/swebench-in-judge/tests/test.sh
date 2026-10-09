#!/bin/bash
# The Judge's fixed procedure; the sample's own logic, not the Harness's.
# Harbor runs oracle and nop over the SWE-bench Verified tasks shipped in
# /tests (see run_tasks.sh), every task environment created by the host
# broker without network, and the reward is 1 only if each oracle trial
# scored 1 and each nop trial 0.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p /logs/verifier
bash "$here/run_tasks.sh" /logs/verifier/harbor-jobs \
    /logs/verifier/reward.json /logs/verifier/harbor-summary.json
