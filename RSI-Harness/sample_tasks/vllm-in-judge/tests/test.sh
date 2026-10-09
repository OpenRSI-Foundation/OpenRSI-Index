#!/bin/bash
# The Judge's fixed procedure; the sample's own logic, not the Harness's.
# 1. vLLM serves the checkpoint Work left in its WORKDIR (a read-only
#    snapshot here) on the Judge's GPU, on loopback only, and must become
#    healthy;
# 2. metering_proxy.py meters vLLM on another loopback port: every request
#    made under /t/<trial>/v1 is one line of usage.jsonl for that trial;
# 3. Harbor runs the terminus-2 agent, its model being that endpoint, over
#    the tasks in /tests/harbor-tasks, every task environment created by the
#    host broker through the rsi_sandbox_harbor plugin. Agent kwargs are a
#    job's, not a trial's, so each task is a Harbor job of its own (one
#    trial) whose api_base is the task's base path at the proxy; up to
#    RSI_VLLM_CONCURRENCY jobs run at once;
# 4. demo_report.py writes the reward (the fraction of trials solved) and
#    vllm-demo-summary.json: vLLM's GPU and model, the completions it
#    served, each trial's outcome, environment and metered tokens, accuracy
#    against tokens, and any infrastructure error.
# Harbor gets what is left of the round's budget less a reserve, so the
# report is written even when Harbor hangs.
#
#   RSI_VLLM_CHECKPOINT   the checkpoint (default <WORKDIR>/checkpoint)
#   RSI_VLLM_TASKS        optional comma list of task names to run
#   RSI_VLLM_MAX_TURNS    terminus-2 turns per trial (default 12)
#   RSI_VLLM_CONCURRENCY  concurrent trials (default 2)
#   RSI_VLLM_OUT          where the evidence goes (default /logs/verifier)
#   RSI_VLLM_PORT         vLLM's loopback port (default 8000)
#   RSI_VLLM_PROXY_PORT   the metering proxy's (default RSI_VLLM_PORT + 1)
#   RSI_VLLM_BUDGET_SEC   the round's budget (default 3600, the verifier's
#                         timeout_sec in task.toml)
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
out=${RSI_VLLM_OUT:-/logs/verifier}
checkpoint=${RSI_VLLM_CHECKPOINT:-/workspace/checkpoint}
model=rsi-checkpoint
port=${RSI_VLLM_PORT:-8000}
budget=${RSI_VLLM_BUDGET_SEC:-3600}
# For stopping Harbor and vLLM, the metrics and the report.
reserve=300
plugin=rsi_sandbox_harbor:ManagedSandboxEnvironment
base=http://127.0.0.1:$port
proxy_port=${RSI_VLLM_PROXY_PORT:-$((port + 1))}
proxy=http://127.0.0.1:$proxy_port
evidence=$out/vllm
jobs=$out/harbor-jobs
report=(python3 "$here/demo_report.py")
mkdir -p "$evidence" "$jobs"
: "${RSI_SANDBOX_PYTHONPATH:?no sandbox endpoint: this sample needs an environment grant}"
# The Judge has no network: nothing may try the Hub, the cost map or telemetry.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 HARBOR_TELEMETRY=0 LITELLM_LOCAL_MODEL_COST_MAP=True \
    HOSTED_VLLM_API_KEY=unused

gpus() {
    nvidia-smi --query-gpu=index,uuid,name,memory.used \
        --format=csv,noheader,nounits > "$1" 2>> "$evidence/nvidia-smi.err"
}

# Running, not a zombie waiting to be reaped.
alive() {
    local stat
    stat=$(cat "/proc/$1/stat" 2> /dev/null) || return 1
    stat=${stat##*) }
    [[ ${stat:0:1} != [ZX] ]]
}

vllm_pid=
proxy_pid=
stop_vllm() {
    if [[ -n $proxy_pid ]]; then
        kill -TERM "$proxy_pid" 2> /dev/null
        wait "$proxy_pid" 2> /dev/null
        proxy_pid=
    fi
    [[ -n $vllm_pid ]] || return 0
    # Its own session: the API server and the engine process go together.
    kill -TERM -- "-$vllm_pid" 2> /dev/null
    for _ in $(seq 60); do
        alive "$vllm_pid" || break
        sleep 1
    done
    kill -KILL -- "-$vllm_pid" 2> /dev/null
    wait "$vllm_pid" 2> /dev/null
    vllm_pid=
}
trap stop_vllm EXIT

ls -la "$checkpoint" > "$evidence/checkpoint.txt" 2>&1
gpus "$evidence/gpu-before.csv"
setsid vllm serve "$checkpoint" --served-model-name "$model" \
    --host 127.0.0.1 --port "$port" --max-model-len 32768 \
    --gpu-memory-utilization 0.5 --enforce-eager \
    > "$evidence/vllm.log" 2>&1 &
vllm_pid=$!
harbor_exit=
if "${report[@]}" wait --base "$base" --pid "$vllm_pid" --timeout 900; then
    touch "$evidence/healthy"
    gpus "$evidence/gpu-after.csv"
    "${report[@]}" fetch --base "$base" --path /v1/models --out "$evidence/models.json"
    python3 "$here/metering_proxy.py" --upstream "$base" --port "$proxy_port" \
        --log "$out/usage.jsonl" > "$evidence/proxy.log" 2>&1 &
    proxy_pid=$!
    "${report[@]}" wait --base "$proxy" --pid "$proxy_pid" --timeout 30 \
        --path /metering/health || harbor_exit=1
    tasks=()
    while read -r task key; do
        tasks+=("$task $key")
    done < <("${report[@]}" tasks --tasks "$here/harbor-tasks" --only "${RSI_VLLM_TASKS:-}")
    info='{"max_input_tokens": 28672, "max_output_tokens": 4096,'
    info+=' "input_cost_per_token": 0.0, "output_cost_per_token": 0.0}'
    pids=()
    for entry in "${tasks[@]}"; do
        [[ -z $harbor_exit ]] || break
        task=${entry% *} key=${entry##* }
        # At most RSI_VLLM_CONCURRENCY jobs at once.
        while :; do
            running=0
            for pid in "${pids[@]}"; do
                alive "$pid" && running=$((running + 1))
            done
            ((running < ${RSI_VLLM_CONCURRENCY:-2})) && break
            sleep 1
        done
        limit=$((budget - SECONDS - reserve))
        ((limit > 10)) || limit=10
        echo "harbor run terminus-2 $task: limit ${limit}s"
        PYTHONPATH=$RSI_SANDBOX_PYTHONPATH timeout --kill-after=30 "${limit}s" harbor run \
            --env "$plugin" \
            --agent terminus-2 --model "hosted_vllm/$model" \
            --ak "api_base=$proxy/t/$key/v1" --ak "model_info=$info" \
            --ak "max_turns=${RSI_VLLM_MAX_TURNS:-12}" \
            --ak suppress_max_turns_warning=true \
            --ak record_terminal_session=false --ak temperature=0.0 \
            --path "$here/harbor-tasks" --include-task-name "$task" \
            --jobs-dir "$jobs/terminus-2" --job-name "$key" \
            --n-concurrent 1 --quiet --yes &
        pids+=("$!")
    done
    # The worst exit of the jobs: a time limit's 124 or 137 over a failure.
    for index in "${!pids[@]}"; do
        wait "${pids[$index]}"
        status=$?
        echo "harbor run terminus-2 ${tasks[$index]% *}: exit $status"
        ((status > ${harbor_exit:-0})) && harbor_exit=$status
    done
    harbor_exit=${harbor_exit:-0}
    echo "harbor run terminus-2: exit $harbor_exit"
    "${report[@]}" fetch --base "$base" --path /metrics --out "$evidence/metrics.txt"
fi
stop_vllm
"${report[@]}" report --evidence "$evidence" --jobs "$jobs" --job terminus-2 \
    --tasks "$here/harbor-tasks" --only "${RSI_VLLM_TASKS:-}" \
    --checkpoint "$checkpoint" --environment "$plugin" \
    ${harbor_exit:+--harbor-exit "$harbor_exit"} --usage "$out/usage.jsonl" \
    --reward-json "$out/reward.json" --summary "$out/vllm-demo-summary.json"
