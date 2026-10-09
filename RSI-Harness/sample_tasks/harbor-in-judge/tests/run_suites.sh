#!/bin/bash
# Run Harbor's oracle and nop agents over this sample's suites, every task
# environment created by the host broker through the rsi_sandbox_harbor
# plugin, then score the jobs with harbor_reward.py. The Judge's fixed
# procedure (test.sh) runs this, and so can a Work agent before it submits.
#
# Usage: run_suites.sh JOBS_DIR REWARD_JSON SUMMARY_JSON
#   RSI_HARBOR_SUITES       comma list of: tb2 (the TB2 prebuilt subset),
#                           compose (a Compose sidecar), build (fix-git
#                           built from its Dockerfile); default tb2
#   RSI_HARBOR_TB2_TASKS    optional comma list of TB2 task names
#   RSI_HARBOR_CONCURRENCY  concurrent trials per job (default 2)
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
jobs=${1:?jobs directory} reward=${2:?reward file} summary=${3:?summary file}
: "${RSI_SANDBOX_PYTHONPATH:?no sandbox endpoint: this sample needs an environment grant}"
export HARBOR_TELEMETRY=0
mkdir -p "$jobs"

score=()
IFS=, read -ra suites <<< "${RSI_HARBOR_SUITES:-tb2}"
for suite in "${suites[@]}"; do
    case $suite in
        tb2) path=$here/tb2-subset ;;
        compose) path=$here/compose-sidecar ;;
        build) path=$here/fix-git-build ;;
        *) echo "unknown suite: $suite" >&2; exit 2 ;;
    esac
    filters=()
    if [[ $suite == tb2 && -n ${RSI_HARBOR_TB2_TASKS:-} ]]; then
        IFS=, read -ra tasks <<< "$RSI_HARBOR_TB2_TASKS"
        for task in "${tasks[@]}"; do
            filters+=(--include-task-name "$task")
        done
        score+=(--only "tb2=$RSI_HARBOR_TB2_TASKS")
    fi
    score+=(--suite "$suite=$path")
    for agent in oracle nop; do
        PYTHONPATH=$RSI_SANDBOX_PYTHONPATH harbor run \
            --env rsi_sandbox_harbor:ManagedSandboxEnvironment \
            --agent "$agent" --path "$path" "${filters[@]}" \
            --jobs-dir "$jobs" --job-name "$suite-$agent" \
            --n-concurrent "${RSI_HARBOR_CONCURRENCY:-2}" --quiet --yes
        echo "harbor run $suite $agent: exit $?"
    done
done
python3 "$here/harbor_reward.py" --jobs "$jobs" "${score[@]}" \
    --reward-json "$reward" --summary "$summary"
