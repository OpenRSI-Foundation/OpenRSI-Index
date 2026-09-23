#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Run E0-E5 concurrently for one frozen optimizer candidate (216 GPUs total).

Usage:
  run_optimizer_candidate_ladder.sh --candidate-id ID --hypothesis-file FILE [--dry-run]

The six attempts are named ID-e0 through ID-e5. Each rung is submitted through
the trusted <CLUSTER> Slurm proxy. The agent must not invoke Slurm directly.
EOF
}

candidate_id=
hypothesis_file=
dry_run=0
while (( $# > 0 )); do
    case "$1" in
        --candidate-id) candidate_id=${2:-}; shift 2 ;;
        --hypothesis-file) hypothesis_file=${2:-}; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'error: unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ ${candidate_id} =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$ ]] || {
    printf 'error: --candidate-id must contain 1-81 safe identifier characters\n' >&2
    exit 2
}
[[ -s ${hypothesis_file} ]] || {
    printf 'error: --hypothesis-file must name a non-empty file\n' >&2
    exit 2
}
if (( dry_run == 1 )); then
    printf 'parallel candidate: %s\n' "${candidate_id}"
    printf 'scheduler: trusted <CLUSTER> Slurm proxy\n'
    printf 'rung resources: E0-E2=1 node each, E3-E4=4 nodes each, E5=16 nodes\n'
    for scale in E0 E1 E2 E3 E4 E5; do
        rung=${scale,,}
        "$(dirname -- "${BASH_SOURCE[0]}")/run_optimizer_scaling_ladder.sh" \
            --dry-run \
            --scale "${scale}" \
            --attempt-id "${candidate_id}-${rung}" \
            --hypothesis-file "${hypothesis_file}"
    done
    exit 0
fi

deadline_file=/run-contract/RESEARCH_DEADLINE_UTC
minimum_parallel_ladder_seconds=36000
[[ -r ${deadline_file} ]] || {
    printf 'error: trusted research deadline is missing: %s\n' "${deadline_file}" >&2
    exit 2
}
deadline_epoch=$(date -u -d "$(tr -d '[:space:]' <"${deadline_file}")" +%s)
remaining_seconds=$((deadline_epoch - $(date -u +%s)))
(( remaining_seconds >= minimum_parallel_ladder_seconds )) || {
    printf 'error: only %s seconds remain; a parallel E0-E5 ladder requires at least %s\n' \
        "${remaining_seconds}" "${minimum_parallel_ladder_seconds}" >&2
    exit 2
}

task_tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
frozen_source_hash=$(python "${task_tools}/optimizer_scaling_task.py" hash --project /app/project)
log_root=/app/output/launcher-logs/${candidate_id}
[[ ! -e ${log_root} ]] || {
    printf 'error: candidate launcher log directory already exists: %s\n' "${log_root}" >&2
    exit 2
}
mkdir -p -- "${log_root}"

scales=(E0 E1 E2 E3 E4 E5)
pids=()
cleanup() {
    status=$?
    trap - EXIT INT TERM
    for pid in "${pids[@]:-}"; do
        kill -TERM "${pid}" 2>/dev/null || true
    done
    wait "${pids[@]:-}" 2>/dev/null || true
    exit "${status}"
}
trap cleanup EXIT INT TERM

for scale in "${scales[@]}"; do
    rung=${scale,,}
    "${task_tools}/run_optimizer_scaling_ladder.sh" \
        --scale "${scale}" \
        --attempt-id "${candidate_id}-${rung}" \
        --hypothesis-file "${hypothesis_file}" \
        >"${log_root}/${scale}.log" 2>&1 &
    pids+=("$!")
done

# Fail before meaningful GPU work if concurrent editing caused any rung to
# snapshot a different candidate. Each prepare writes its contract before the
# shared launcher starts model initialization.
contracts_ready=0
for _ in {1..120}; do
    contracts_ready=1
    for scale in "${scales[@]}"; do
        rung=${scale,,}
        if [[ ! -s /app/output/attempts/${candidate_id}-${rung}/run_contract.json ]]; then
            contracts_ready=0
            break
        fi
    done
    (( contracts_ready == 1 )) && break
    sleep 1
done
(( contracts_ready == 1 )) || {
    printf 'error: not all six run contracts appeared within 120 seconds\n' >&2
    exit 1
}
for scale in "${scales[@]}"; do
    rung=${scale,,}
    contract_hash=$(python -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["source_inventory_sha256"])' \
        "/app/output/attempts/${candidate_id}-${rung}/run_contract.json")
    [[ ${contract_hash} == "${frozen_source_hash}" ]] || {
        printf 'error: %s captured source %s, expected %s\n' \
            "${scale}" "${contract_hash}" "${frozen_source_hash}" >&2
        exit 1
    }
done

failed=()
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        failed+=("${scales[index]}")
    fi
done
pids=()
trap - EXIT INT TERM

if (( ${#failed[@]} > 0 )); then
    printf 'candidate %s failed rungs:' "${candidate_id}" >&2
    printf ' %s' "${failed[@]}" >&2
    printf '\nlogs: %s\n' "${log_root}" >&2
    exit 1
fi
printf 'candidate %s completed E0-E5; logs: %s\n' "${candidate_id}" "${log_root}"
