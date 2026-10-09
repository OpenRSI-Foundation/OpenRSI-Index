#!/usr/bin/env bash
# Objective acceptance A1-A8 of the managed Docker capability (spec 7, M9).
#
# Runs `rsi-harness run` and `rsi-harness recover` as root on
# sample_tasks/harbor-in-judge with the operator policy, then prints a
# PASS/FAIL table including the A5 leftover audit and the A6 socket audit.
# The CLI has no scripted agent (agents are not Harness work), so each run
# goes through tests/acceptance/scripted_cli.py: the real `rsi-harness` app
# with the codex model CLI replaced by a bash script (tests/acceptance/work).
#
#   sudo scripts/operator/sandbox_acceptance.sh [--dry-run] [--only a1,a4,...]
#       [--policy FILE] [--scratch DIR] [--keep]
#
#   a1           A1 TB2 subset (6 tasks, oracle 1 / nop 0) in a real Judge, and
#                A6: every Work/Judge/env/builder container inspected, the
#                sockets the Judge, children and a RUN step see
#   a2           A2 the same subset run by a scripted agent in Work before it
#                submits, and A3 the Compose sidecar in Work and in the Judge
#   a4           A4 fix-git built from its Dockerfile in the Judge
#   a7-env-create, a7-build, a7-load, a7-paused
#                A7 kill -9 of the harness at that moment, then recover
#   a8           A8 a Judge activated 60 s before the Work deadline
#                (verifier 900 s) finishes with a reward
#   swebench     opt-in (only with --only): sample_tasks/swebench-in-judge
#                with its own policy, three SWE-bench Verified tasks offline
#                (oracle 1 / nop 0) from the images of
#                sample_tasks/swebench-in-judge/images.manifest, pre-pulled
# Every scenario ends with its A5 audit. Each run uses its own data root
# under the scratch directory; the script only ever acts on the runs it
# started (their labels, rules, bridges and loop files) and reads nothing
# else. --dry-run (any user) prints the plan and every command, runs none.
# The scratch directory is new and root's: by default `mktemp -d`, and a
# --scratch DIR must not exist yet (its parent must).
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PY=${RSI_PYTHON:-$REPO/.venv/bin/python}
HARNESS=${RSI_HARNESS:-$REPO/.venv/bin/rsi-harness}
SAMPLE=$REPO/sample_tasks/harbor-in-judge
WORK=$REPO/tests/acceptance/work
POLICY=$REPO/sample_tasks/harbor-in-judge/operator-policy.toml
SWEBENCH=$REPO/sample_tasks/swebench-in-judge
SWEBENCH_POLICY=$REPO/sample_tasks/swebench-in-judge/operator-policy.toml
SWEBENCH_MANIFEST=$REPO/sample_tasks/swebench-in-judge/images.manifest
SCRATCH=
DRY_RUN=0
KEEP=0
ONLY=
ALL=(a1 a2 a4 a7-env-create a7-build a7-load a7-paused a8)
# Run only when --only names them.
OPT_IN=(swebench)
IMAGES=(
    alexgshaw/fix-git:20251031 alexgshaw/regex-log:20251031
    alexgshaw/adaptive-rejection-sampler:20251031
    alexgshaw/kv-store-grpc:20251031 alexgshaw/nginx-request-logging:20251031
    alexgshaw/git-multibranch:20251031 python:3.13-slim-bookworm
    redis:7-alpine alpine:3.21 moby/buildkit:v0.27.1
)

usage() {
    sed -n '2,/^set -uo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

while (($#)); do
    case $1 in
        --dry-run) DRY_RUN=1 ;;
        --keep) KEEP=1 ;;
        --only) ONLY=${2:?}; shift ;;
        --policy) POLICY=${2:?}; shift ;;
        --scratch) SCRATCH=${2:?}; shift ;;
        -h|--help) usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
    shift
done
for name in ${ONLY//,/ }; do
    [[ " ${ALL[*]} ${OPT_IN[*]} " == *" $name "* ]] || {
        echo "unknown scenario in --only: $name (one of: ${ALL[*]} ${OPT_IN[*]})" >&2
        exit 2
    }
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

# Run (or, with --dry-run, only show) one command.
act() {
    show "$@"
    ((DRY_RUN)) || "$@"
}

selected() {
    if [[ -z $ONLY ]]; then
        [[ " ${ALL[*]} " == *" $1 "* ]]
    else
        [[ ,$ONLY, == *,$1,* ]]
    fi
}

# Whether a scenario of the harbor-in-judge sample is selected.
harbor_selected() {
    local name
    for name in "${ALL[@]}"; do
        selected "$name" && return 0
    done
    return 1
}

preflight() {
    announce "preflight: root, cached images, builder digest, loop devices, disk"
    if ((!DRY_RUN)) && [[ $(id -u) != 0 ]]; then
        echo "run as root (sudo $0); --dry-run works as any user" >&2
        exit 2
    fi
    [[ -x $PY && -x $HARNESS ]] || { echo "missing $PY or $HARNESS" >&2; exit 2; }
    [[ -f $POLICY ]] || { echo "missing policy $POLICY" >&2; exit 2; }
    local missing=0 image
    # Short: <scratch>/<scenario>/data/<run>/sb/<8 hex>/s must fit 107 bytes.
    local scratch=${SCRATCH:-/var/tmp/rsi-acc-XXXXXX}
    local socket=$scratch/a7-env-create/data/$(printf '%032d' 0)/sb/00000000/s
    if ((${#socket} > 107)); then
        echo "   --scratch $scratch is too long for the endpoint socket path"
        missing=1
    fi
    local images=()
    harbor_selected && images+=("${IMAGES[@]}")
    if selected swebench; then
        [[ -f $SWEBENCH_POLICY ]] || { echo "missing policy $SWEBENCH_POLICY" >&2; exit 2; }
        mapfile -t -O "${#images[@]}" images < <(grep -v '^#' "$SWEBENCH_MANIFEST" | grep .)
    fi
    for image in "${images[@]}"; do
        if ! docker image inspect "$image" > /dev/null 2>&1; then
            echo "   not cached: $image (spec 8 step 2: the operator pre-pulls it)"
            missing=1
        fi
    done
    local digest
    if harbor_selected; then
        digest=$(docker image inspect --format '{{index .RepoDigests 0}}' \
            moby/buildkit:v0.27.1 2> /dev/null || true)
        if [[ -n $digest ]] && ! grep -qF "\"$digest\"" "$POLICY"; then
            echo "   the policy's builder_image is not the cached $digest"
            missing=1
        fi
        [[ -e /dev/loop-control ]] || { echo "   no /dev/loop-control (modprobe loop)"; missing=1; }
    fi
    df -h / | sed 's/^/   /'
    if ((missing)) && ((!DRY_RUN)); then
        echo "preflight failed" >&2
        exit 2
    fi
}

# The scratch directory: created here, by root, and never one that already
# existed (another user could have planted links in it).
make_scratch() {
    if [[ -z $SCRATCH ]]; then
        SCRATCH=$(mktemp -d /var/tmp/rsi-acc-XXXXXX) || exit 2
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

# row CHECK PASS|FAIL DETAIL: one line of the result table.
row() {
    printf '%s|%s|%s\n' "$1" "$2" "$3" | tee -a "$SCRATCH/rows" | sed 's/^/   /'
}

# scenario NAME AGENT KILL_AT TASK TIMEOUT JUDGE_SUITES JUDGE_TB2_TASKS [WORK_ENV...]
# Every command is built once, shown, and (without --dry-run) run as shown,
# RUN_ID and PID standing for what only the run itself tells.
# SCENARIO_SAMPLE and SCENARIO_POLICY name another sample (its tests go to
# Work) and policy than harbor-in-judge's.
scenario() {
    local name=$1 agent=$2 kill_at=$3 task=$4 timeout=$5 suites=$6 tb2=$7
    shift 7
    local dir=$SCRATCH/$name
    local sample=${SCENARIO_SAMPLE:-$SAMPLE} policy=${SCENARIO_POLICY:-$POLICY}
    local run=(env "RSI_HARBOR_SUITES=$suites" "RSI_HARBOR_TB2_TASKS=$tb2"
        "$PY" -m tests.acceptance.scripted_cli --work-script "$WORK/$agent"
        --work-files "$sample/tests" --work-files "$WORK")
    local item
    for item in "$@"; do
        run+=(--work-env "$item")
    done
    run+=(-- run "$task" --agent codex --sandbox-policy "$policy"
        --data-root "$dir/data" --logs-root "$dir/logs")
    [[ -n $timeout ]] && run+=(--timeout "$timeout")
    local watch=("$PY" -m tests.acceptance.audit watch --data-root "$dir/data"
        --pid PID --out "$dir/watch.json")
    [[ -n $kill_at ]] && watch+=(--kill-at "$kill_at")
    local recover=("$HARNESS" recover RUN_ID --data-root "$dir/data"
        --logs-root "$dir/logs")
    local audit=("$PY" -m tests.acceptance.audit leftovers --data-root "$dir/data"
        --observed "$dir/watch.json" --run-id RUN_ID)
    local cleanup=("$HARNESS" cleanup RUN_ID --delete-workspace --yes
        --data-root "$dir/data" --logs-root "$dir/logs")
    local verdict=("$PY" -m tests.acceptance.audit verdict "$name" --dir "$dir")
    announce "[$name] Judge suites=$suites${tb2:+ ($tb2)}; Work agent $agent${kill_at:+; kill -9 at $kill_at}"
    show "${run[@]}"
    show "${watch[@]}"
    [[ -n $kill_at ]] && show "${recover[@]}"
    show "${audit[@]}"
    if ((!KEEP)); then
        show "${cleanup[@]}"
        show "${audit[@]}" --strict
    fi
    show "${verdict[@]}"
    if ((DRY_RUN)); then
        return
    fi
    mkdir -m 0700 "$dir" || { row "$name" FAIL "cannot create $dir"; return; }
    (cd "$REPO" && exec "${run[@]}") > "$dir/run.log" 2>&1 &
    local pid=$!
    watch[7]=$pid
    (cd "$REPO" && exec "${watch[@]}") > "$dir/watch.log" 2>&1 &
    local watcher=$!
    wait "$pid"
    echo $? > "$dir/run.rc"
    wait "$watcher"
    local run_id
    run_id=$(cd "$REPO" && "$PY" -m tests.acceptance.audit run-id --data-root "$dir/data")
    if [[ -z $run_id ]]; then
        row "$name" FAIL "no run found under $dir/data (rsi-harness exit $(cat "$dir/run.rc"))"
        return
    fi
    echo "   run $run_id: rsi-harness exit $(cat "$dir/run.rc")"
    local -a command
    if [[ -n $kill_at ]]; then
        mapfile -d '' command < <(with_run "$run_id" "${recover[@]}")
        "${command[@]}" > "$dir/recover.log" 2>&1
        echo $? > "$dir/recover.rc"
    fi
    mapfile -d '' command < <(with_run "$run_id" "${audit[@]}")
    (cd "$REPO" && "${command[@]}") > "$dir/leftovers.json"
    if ((!KEEP)); then
        # Free the host (this run's retained Work image, WORKDIR volume and
        # workspace), then nothing labelled with the run may be left.
        mapfile -d '' command < <(with_run "$run_id" "${cleanup[@]}")
        "${command[@]}" > "$dir/cleanup.log" 2>&1
        mapfile -d '' command < <(with_run "$run_id" "${audit[@]}" --strict)
        (cd "$REPO" && "${command[@]}") > "$dir/leftovers-after-cleanup.json"
    fi
    local rows
    if ! rows=$(cd "$REPO" && "${verdict[@]}" 2> "$dir/verdict.log") || [[ -z $rows ]]; then
        row "$name" FAIL "no verdict (see $dir/verdict.log)"
        return
    fi
    printf '%s\n' "$rows" | tee -a "$SCRATCH/rows" | sed 's/^/   /'
}

a8_task() {
    announce "[a8] a copy of the sample whose Judge has spec A8's 900 s verifier and starts its suites 90 s in, after the Work deadline"
    local copy=$SCRATCH/a8-task
    act cp -a "$SAMPLE" "$copy"
    act sed -i '/^\[verifier\]$/,/^\[/ s/^timeout_sec = 3600$/timeout_sec = 900/' \
        "$copy/task.toml"
    # The round must outlast the Work deadline (submitted 60 s before it):
    # every Judge environment is then created after Work's time is up.
    act sed -i 's/^set -uo pipefail$/set -uo pipefail\nsleep 90/' \
        "$copy/tests/test.sh"
    ((DRY_RUN)) || grep -A1 '^\[verifier\]$' "$copy/task.toml" | sed 's/^/   /'
}

preflight
if ((DRY_RUN)); then
    SCRATCH=${SCRATCH:-/var/tmp/rsi-acc-XXXXXX}
else
    make_scratch
    : > "$SCRATCH/rows"
fi
announce "scratch $SCRATCH (new, root 0700); sample $SAMPLE; policy $POLICY"
for name in "${ALL[@]}" "${OPT_IN[@]}"; do
    selected "$name" || continue
    case $name in
        a1) scenario a1 agent-submit.sh "" "$SAMPLE" "" tb2 "" ;;
        a2) scenario a2 agent-work-suites.sh "" "$SAMPLE" "" compose "" \
                RSI_HARBOR_SUITES=tb2,compose ;;
        a4) scenario a4 agent-submit.sh "" "$SAMPLE" "" build "" \
                RSI_ACCEPTANCE_PROBE=0 ;;
        a7-env-create) scenario a7-env-create agent-hold-env.sh env_create \
                "$SAMPLE" "" compose "" ;;
        a7-build) scenario a7-build agent-build-hold.sh build "$SAMPLE" "" \
                compose "" ;;
        a7-load) scenario a7-load agent-build-hold.sh load "$SAMPLE" "" \
                compose "" ;;
        a7-paused) scenario a7-paused agent-hold-env.sh paused "$SAMPLE" "" \
                compose "" HOLD_SUBMIT=1 ;;
        a8) a8_task
            scenario a8 agent-late-submit.sh "" "$SCRATCH/a8-task" 300 tb2 \
                fix-git,regex-log WORK_TIMEOUT_SEC=300 RSI_ACCEPTANCE_PROBE=0 ;;
        swebench) SCENARIO_SAMPLE=$SWEBENCH SCENARIO_POLICY=$SWEBENCH_POLICY \
                scenario swebench agent-submit.sh "" "$SWEBENCH" "" swebench "" \
                RSI_ACCEPTANCE_PROBE=0 ;;
    esac
done

if ((DRY_RUN)); then
    announce "dry run: nothing was run"
    exit 0
fi
announce "result"
printf '%-18s %-6s %s\n' CHECK RESULT DETAIL
failed=0
rows=0
while IFS='|' read -r check result detail; do
    printf '%-18s %-6s %s\n' "$check" "$result" "$detail"
    [[ $result == PASS ]] || failed=1
    rows=$((rows + 1))
done < "$SCRATCH/rows"
if ((rows == 0)); then
    echo "no check ran" >&2
    failed=1
fi
echo "logs, watch records and audits: $SCRATCH"
exit "$failed"
