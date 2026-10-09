#!/bin/bash
# Run Harbor's oracle and nop agents over this sample's SWE-bench Verified
# tasks (swebench-verified/), every task environment created by the host
# broker without network through the rsi_sandbox_harbor plugin, then score
# the jobs with harbor_reward.py. The Judge's fixed procedure (test.sh)
# runs this, and so can a Work agent before it submits.
#
# Usage: run_tasks.sh JOBS_DIR REWARD_JSON SUMMARY_JSON
#   RSI_SWEBENCH_TASKS        optional comma list of task names (default all)
#   RSI_SWEBENCH_CONCURRENCY  concurrent trials per job (default 3)
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
jobs=${1:?jobs directory} reward=${2:?reward file} summary=${3:?summary file}
: "${RSI_SANDBOX_PYTHONPATH:?no sandbox endpoint: this sample needs an environment grant}"
export HARBOR_TELEMETRY=0
suite=$here/swebench-verified
mkdir -p "$jobs"

filters=() score=()
if [[ -n ${RSI_SWEBENCH_TASKS:-} ]]; then
    IFS=, read -ra tasks <<< "$RSI_SWEBENCH_TASKS"
    for task in "${tasks[@]}"; do
        filters+=(--include-task-name "$task")
    done
    score+=(--only "swebench=$RSI_SWEBENCH_TASKS")
fi
for agent in oracle nop; do
    PYTHONPATH=$RSI_SANDBOX_PYTHONPATH harbor run \
        --env rsi_sandbox_harbor:ManagedSandboxEnvironment \
        --agent "$agent" --path "$suite" "${filters[@]}" \
        --jobs-dir "$jobs" --job-name "swebench-$agent" \
        --n-concurrent "${RSI_SWEBENCH_CONCURRENCY:-3}" --quiet --yes
    echo "harbor run swebench $agent: exit $?"
done
python3 "$here/harbor_reward.py" --jobs "$jobs" --suite "swebench=$suite" \
    "${score[@]}" --reward-json "$reward" --summary "$summary"
