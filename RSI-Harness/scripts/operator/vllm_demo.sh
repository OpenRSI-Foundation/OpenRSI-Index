#!/usr/bin/env bash
# The vLLM-in-Judge demo: the reference scenario of the managed Docker
# capability, run end to end through `rsi-harness run` as root.
#
# Work (sample_tasks/vllm-in-judge, its scripted agent work/agent.sh, no
# model CLI; see tests/acceptance/scripted_cli.py) places a small existing
# checkpoint from the Hugging Face Hub into its WORKDIR and submits. The
# Judge, derived from Work, runs its fixed procedure /tests/test.sh: vLLM
# serves the read-only checkpoint on the Judge's GPU, and Harbor runs the
# terminus-2 agent against it over the tasks in /tests/harbor-tasks, every
# task environment created by the host broker. Then the A5 leftover audit,
# `rsi-harness cleanup` and the strict audit, and a PASS/FAIL table:
#
#   V0 Work      the checkpoint was downloaded and the submission accepted
#   V1 vLLM      vLLM became healthy and served the WORKDIR checkpoint on a
#                GPU the Harness gave the Judge
#   V2 requests  the completions vLLM served (its metrics, aborted ones
#                apart), and every trial's agent asked the model
#   V3 trials    every trial ran to completion in a broker-created
#                environment (the plugin's), without an infrastructure error
#                (the score itself does not matter)
#   A5 vllm      nothing of the run left, before and after cleanup
#
#   sudo scripts/operator/vllm_demo.sh [--dry-run] [--gpus 4] [--policy FILE]
#       [--model REPO [--revision SHA]] [--tasks rsi/hello-file,regex-log]
#       [--scratch DIR] [--keep]
#
# --gpus is the run's GPU pool (the Judge takes one; Work needs none),
# --model/--revision the checkpoint Work downloads (default
# Qwen/Qwen2.5-1.5B-Instruct at a pinned commit; another model's default
# revision is main), --tasks narrows the
# Judge's task list (default: all of them). The script only ever acts on
# the run it started and reads nothing else. --dry-run (any user) prints the
# plan and every command, runs none. The scratch directory is new and
# root's: by default `mktemp -d`, and a --scratch DIR must not exist yet
# (its parent must). Preflight refuses a --gpus device that already holds
# more than 1024 MiB, and less free space under Docker's root than the run
# needs (20 GiB, 15 GiB while the sample's image layers are cached).
#
# Interrupted (Ctrl-C, TERM), the script stops its `rsi-harness run` (TERM)
# and prints the run's recovery: `rsi-harness recover RUN --data-root
# <scratch>/vllm/data --logs-root <scratch>/vllm/logs`, then `rsi-harness
# cleanup RUN --delete-workspace --yes` with the same roots.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${RSI_PYTHON:-$REPO/.venv/bin/python}
HARNESS=${RSI_HARNESS:-$REPO/.venv/bin/rsi-harness}
SAMPLE=$REPO/sample_tasks/vllm-in-judge
POLICY=$REPO/sample_tasks/vllm-in-judge/operator-policy.toml
GPUS=4
MODEL=Qwen/Qwen2.5-1.5B-Instruct
REVISION=
TASKS=
SCRATCH=
DRY_RUN=0
KEEP=0
IMAGES=(python:3.13-slim-bookworm alexgshaw/regex-log:20251031)
# The sample's Work image as tests/integration/test_vllm_in_judge_sample.py
# builds it: its layers are the Harness's Base image cache.
WORK_IMAGE=rsi-sample-vllm-in-judge:work
GPU_BUSY_MIB=1024

usage() {
    sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while (($#)); do
    case $1 in
        --dry-run) DRY_RUN=1 ;;
        --keep) KEEP=1 ;;
        --gpus) GPUS=${2:?}; shift ;;
        --policy) POLICY=${2:?}; shift ;;
        --model) MODEL=${2:?}; shift ;;
        --revision) REVISION=${2:?}; shift ;;
        --tasks) TASKS=${2:?}; shift ;;
        --scratch) SCRATCH=${2:?}; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
    shift
done
# Root never writes bytecode into the checkout.
export PYTHONDONTWRITEBYTECODE=1

announce() {
    printf '\n== %s\n' "$*"
}

show() {
    printf '   $'
    printf ' %q' "$@"
    printf '\n'
}

preflight() {
    announce "preflight: root, cached images, GPUs $GPUS, disk"
    if ((!DRY_RUN)) && [[ $(id -u) != 0 ]]; then
        echo "run as root (sudo $0); --dry-run works as any user" >&2
        exit 2
    fi
    [[ -x $PY && -x $HARNESS ]] || { echo "missing $PY or $HARNESS" >&2; exit 2; }
    [[ -f $POLICY ]] || { echo "missing policy $POLICY" >&2; exit 2; }
    local missing=0 image gpu
    # Short: <scratch>/vllm/data/<run>/sb/<8 hex>/s must fit 107 bytes.
    local scratch=${SCRATCH:-/var/tmp/rsi-vllm-XXXXXX}
    local socket=$scratch/vllm/data/$(printf '%032d' 0)/sb/00000000/s
    if ((${#socket} > 107)); then
        echo "   --scratch $scratch is too long for the endpoint socket path"
        missing=1
    fi
    for image in "${IMAGES[@]}"; do
        if ! docker image inspect "$image" > /dev/null 2>&1; then
            echo "   not cached: $image (the operator pre-pulls it)"
            missing=1
        fi
    done
    local used
    for gpu in ${GPUS//,/ }; do
        if ! nvidia-smi --id="$gpu" --query-gpu=index,uuid,memory.used \
            --format=csv,noheader 2> /dev/null | sed 's/^/   GPU /'; then
            echo "   no GPU $gpu"
            missing=1
            continue
        fi
        used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used \
            --format=csv,noheader,nounits 2> /dev/null | tr -d ' ')
        if ! [[ $used =~ ^[0-9]+$ ]] || ((used > GPU_BUSY_MIB)); then
            echo "   GPU $gpu is busy (${used:-?} MiB used, over $GPU_BUSY_MIB MiB)"
            missing=1
        fi
    done
    local root need=20 free
    root=$(docker info --format '{{.DockerRootDir}}' 2> /dev/null) || root=
    docker image inspect "$WORK_IMAGE" > /dev/null 2>&1 && need=15
    free=$(df --output=avail -BG -- "${root:-/}" 2> /dev/null | tail -1 | tr -dc 0-9)
    echo "   ${free:-?} GiB free under the Docker root ${root:-(unknown)}; need $need GiB"
    if [[ -z $root || -z $free ]] || ((free < need)); then
        echo "   not enough disk for the Work image, the checkpoint and the policy's floor"
        missing=1
    fi
    if ((missing)) && ((!DRY_RUN)); then
        echo "preflight failed" >&2
        exit 2
    fi
}

# The scratch directory: created here, by root, and never one that already
# existed (another user could have planted links in it).
make_scratch() {
    if [[ -z $SCRATCH ]]; then
        SCRATCH=$(mktemp -d /var/tmp/rsi-vllm-XXXXXX) || exit 2
    elif ! mkdir -m 0700 -- "$SCRATCH"; then
        echo "--scratch $SCRATCH must not exist yet (its parent must)" >&2
        exit 2
    fi
    if [[ -L $SCRATCH || $(stat -c %u -- "$SCRATCH") != 0 ]]; then
        echo "scratch $SCRATCH is not root's own directory" >&2
        exit 2
    fi
    chmod 0700 -- "$SCRATCH"
}

# with_run RUN_ID WORD...: the words with the placeholder RUN_ID replaced.
with_run() {
    local run_id=$1 word
    shift
    for word in "$@"; do
        [[ $word == RUN_ID ]] && word=$run_id
        printf '%s\0' "$word"
    done
}

row() {
    printf '%s|%s|%s\n' "$1" "$2" "$3" | tee -a "$SCRATCH/rows" | sed 's/^/   /'
}

RUN_PID=
WATCH_PID=
RECOVER=()
CLEANUP=()
# Ctrl-C or TERM: a background job ignores SIGINT, so the run would go on
# alone. Stop it, then print how to recover and clean up what it left.
interrupted() {
    trap - INT TERM
    echo "interrupted: stopping rsi-harness run${RUN_PID:+ (pid $RUN_PID)}" >&2
    if [[ -n $RUN_PID ]]; then
        kill -TERM "$RUN_PID" 2> /dev/null
        wait "$RUN_PID" 2> /dev/null
    fi
    [[ -n $WATCH_PID ]] && wait "$WATCH_PID" 2> /dev/null
    local run_id=
    if ((${#RECOVER[@]})); then
        run_id=$(cd "$REPO" && "$PY" -m tests.acceptance.audit run-id \
            --data-root "$SCRATCH/vllm/data" 2> /dev/null)
        echo "recover the run, then clean it up:" >&2
        show "${RECOVER[@]/#RUN_ID/${run_id:-RUN_ID}}" >&2
        show "${CLEANUP[@]/#RUN_ID/${run_id:-RUN_ID}}" >&2
    fi
    exit 130
}

# Every command is built once, shown, and (without --dry-run) run as shown,
# RUN_ID and PID standing for what only the run itself tells.
demo() {
    local dir=$SCRATCH/vllm
    local run=(env "RSI_VLLM_TASKS=$TASKS"
        "$PY" -m tests.acceptance.scripted_cli --work-script "$SAMPLE/work/agent.sh"
        --work-env "RSI_VLLM_MODEL=$MODEL" --work-env "RSI_VLLM_REVISION=$REVISION"
        -- run "$SAMPLE" --agent codex --sandbox-policy "$POLICY" --gpus "$GPUS"
        --data-root "$dir/data" --logs-root "$dir/logs")
    local watch=("$PY" -m tests.acceptance.audit watch --data-root "$dir/data"
        --pid PID --out "$dir/watch.json")
    local audit=("$PY" -m tests.acceptance.audit leftovers --data-root "$dir/data"
        --observed "$dir/watch.json" --run-id RUN_ID)
    local cleanup=("$HARNESS" cleanup RUN_ID --delete-workspace --yes
        --data-root "$dir/data" --logs-root "$dir/logs")
    RECOVER=("$HARNESS" recover RUN_ID --data-root "$dir/data"
        --logs-root "$dir/logs")
    CLEANUP=("${cleanup[@]}")
    local verdict=("$PY" -m tests.acceptance.audit verdict vllm --dir "$dir")
    announce "[vllm] Work places $MODEL@${REVISION:-default}; the Judge serves it on GPU pool $GPUS; tasks ${TASKS:-all}"
    show "${run[@]}"
    show "${watch[@]}"
    show "${audit[@]}"
    if ((!KEEP)); then
        show "${cleanup[@]}"
        show "${audit[@]}" --strict
    fi
    show "${verdict[@]}"
    if ((DRY_RUN)); then
        return
    fi
    mkdir -m 0700 "$dir" || { row vllm FAIL "cannot create $dir"; return; }
    trap interrupted INT TERM
    (cd "$REPO" && exec "${run[@]}") > "$dir/run.log" 2>&1 &
    local pid=$!
    RUN_PID=$pid
    watch[7]=$pid
    (cd "$REPO" && exec "${watch[@]}") > "$dir/watch.log" 2>&1 &
    local watcher=$!
    WATCH_PID=$watcher
    wait "$pid"
    echo $? > "$dir/run.rc"
    wait "$watcher"
    trap - INT TERM
    RUN_PID=
    WATCH_PID=
    local run_id
    run_id=$(cd "$REPO" && "$PY" -m tests.acceptance.audit run-id --data-root "$dir/data")
    if [[ -z $run_id ]]; then
        row vllm FAIL "no run found under $dir/data (rsi-harness exit $(cat "$dir/run.rc"))"
        return
    fi
    echo "   run $run_id: rsi-harness exit $(cat "$dir/run.rc")"
    local -a command
    mapfile -d '' command < <(with_run "$run_id" "${audit[@]}")
    (cd "$REPO" && "${command[@]}") > "$dir/leftovers.json"
    if ((!KEEP)); then
        # Free the host (this run's retained Work image, WORKDIR volume with
        # the checkpoint, and workspace), then nothing of the run may be left.
        mapfile -d '' command < <(with_run "$run_id" "${cleanup[@]}")
        "${command[@]}" > "$dir/cleanup.log" 2>&1
        mapfile -d '' command < <(with_run "$run_id" "${audit[@]}" --strict)
        (cd "$REPO" && "${command[@]}") > "$dir/leftovers-after-cleanup.json"
    fi
    local rows
    if ! rows=$(cd "$REPO" && "${verdict[@]}" 2> "$dir/verdict.log") || [[ -z $rows ]]; then
        row vllm FAIL "no verdict (see $dir/verdict.log)"
        return
    fi
    printf '%s\n' "$rows" | tee -a "$SCRATCH/rows" | sed 's/^/   /'
}

preflight
if ((DRY_RUN)); then
    SCRATCH=${SCRATCH:-/var/tmp/rsi-vllm-XXXXXX}
else
    make_scratch
    : > "$SCRATCH/rows"
fi
announce "scratch $SCRATCH (new, root 0700); sample $SAMPLE; policy $POLICY"
demo
if ((DRY_RUN)); then
    announce "dry run: nothing was run"
    exit 0
fi
announce "result"
printf '%-12s %-6s %s\n' CHECK RESULT DETAIL
failed=0
rows=0
while IFS='|' read -r check result detail; do
    printf '%-12s %-6s %s\n' "$check" "$result" "$detail"
    [[ $result == PASS ]] || failed=1
    rows=$((rows + 1))
done < "$SCRATCH/rows"
if ((rows == 0)); then
    echo "no check ran" >&2
    failed=1
fi
echo "logs (the Judge's vllm-demo-summary.json, vllm/ evidence, harbor-jobs/), watch record and audits: $SCRATCH"
exit "$failed"
