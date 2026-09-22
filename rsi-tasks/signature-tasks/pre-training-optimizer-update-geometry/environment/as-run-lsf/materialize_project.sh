#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Materialize the isolated optimizer-encoding project tree.

Usage:
  materialize_project.sh SOURCE_REPO DESTINATION PROJECT_OVERLAY BRIDGE_COMMIT MCORE_COMMIT

SOURCE_REPO is read only. Its current working tree, including uncommitted files
and deletions, is snapshotted. DESTINATION must not already exist.
EOF
}

(( $# == 5 )) || { usage >&2; exit 2; }
source_repo=$1
destination=$2
project_overlay=$3
bridge_commit=$4
mcore_commit=$5

[[ -d ${source_repo} && -f ${source_repo}/pyproject.toml ]] || {
    printf 'error: invalid Megatron-Bridge source: %s\n' "${source_repo}" >&2
    exit 2
}
[[ -d ${source_repo}/3rdparty/Megatron-LM ]] || {
    printf 'error: Megatron-LM source is missing below %s\n' "${source_repo}" >&2
    exit 2
}
[[ -d ${project_overlay}/examples/training/optimizer ]] || {
    printf 'error: optimizer project overlay is missing: %s\n' "${project_overlay}" >&2
    exit 2
}
[[ ${destination} == /* && ${destination} != / && ${destination} != "${HOME:-}" ]] || {
    printf 'error: destination must be a safe absolute path: %s\n' "${destination}" >&2
    exit 2
}
[[ ! -e ${destination} ]] || {
    printf 'error: destination already exists: %s\n' "${destination}" >&2
    exit 2
}

git -C "${source_repo}" cat-file -e "${bridge_commit}^{commit}"
git -C "${source_repo}/3rdparty/Megatron-LM" cat-file -e "${mcore_commit}^{commit}"
[[ $(git -C "${source_repo}" rev-parse HEAD) == "${bridge_commit}" ]] || {
    printf 'error: Bridge HEAD does not match the locked base commit\n' >&2
    exit 2
}
[[ $(git -C "${source_repo}/3rdparty/Megatron-LM" rev-parse HEAD) == "${mcore_commit}" ]] || {
    printf 'error: MCore HEAD does not match the locked commit\n' >&2
    exit 2
}

source_positional_dir=
for candidate in "${source_repo}"/examples/training/*; do
    [[ -d ${candidate} \
        && -f ${candidate}/runtime/pretrain_gpt_marin_adamh.py \
        && -f ${candidate}/runtime/longppl_runner.py ]] || continue
    [[ -z ${source_positional_dir} ]] || {
        printf 'error: multiple positional training sources found below %s\n' \
            "${source_repo}/examples/training" >&2
        exit 2
    }
    source_positional_dir=${candidate}
done
[[ -n ${source_positional_dir} ]] || {
    printf 'error: positional training source is missing below %s\n' \
        "${source_repo}/examples/training" >&2
    exit 2
}
source_positional_name=${source_positional_dir##*/}

mkdir -p -- "${destination}"

# Snapshot the complete current checkout so task-specific tracked and untracked
# source is available to the agent. Exclude local development artifacts and the
# complete training-example tree; the optimizer experiment is copied back
# explicitly below.
tar -C "${source_repo}" \
    --exclude='./.git' \
    --exclude='./.venv' \
    --exclude='./.pytest_cache' \
    --exclude='./.ruff_cache' \
    --exclude='./3rdparty/Megatron-LM/.git' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    --exclude='./examples/training' \
    -cf - . | tar -xf - -C "${destination}"

# Reuse the already validated Marin/Paloma experiment support files, then
# rename that isolated experiment directory for this optimizer task.  The
# task-local overlay below replaces its candidate and trusted entrypoint.
mkdir -p -- "${destination}/examples/training"
tar -C "${source_repo}" \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    -cf - "examples/training/${source_positional_name}" | tar -xf - -C "${destination}"
mv -- "${destination}/examples/training/${source_positional_name}" \
    "${destination}/examples/training/optimizer"

# Use the current baseline <CLUSTER> launcher, which contains the validated
# Slurm + Apptainer training path, then restore the task-specific trusted
# entrypoint below.
cp -- "${source_repo}/examples/training/baseline/run_marin_adamh.sh" \
    "${destination}/examples/training/optimizer/run_marin_adamh.sh"

cp -- "${source_repo}/examples/training/baseline/run_marin_<CLUSTER>.sh" \
    "${destination}/examples/training/optimizer/run_marin_<CLUSTER>.sh"

rm -f -- \
    "${destination}/examples/training/optimizer/run_marin_adamh_bluevela_lsf.sh"

# Task-local locked runtime files override their development-checkout versions.
tar -C "${project_overlay}" --exclude '__pycache__' --exclude '*.pyc' -cf - . | \
    tar -xf - -C "${destination}"

python3 - "${destination}/examples/training/optimizer/run_marin_<CLUSTER>.sh" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
source = path.read_text()
replacements = {
    '    : "${PROFILE:?profile script must set PROFILE}"\n':
        '    : "${PROFILE:?profile script must set PROFILE}"\n'
        '    : "${PRETRAIN_RUNTIME:?profile script must set PRETRAIN_RUNTIME}"\n',
    '    pretrain_entrypoint="${repo_dir}/3rdparty/Megatron-LM/pretrain_gpt.py"\n'
    '    if [[ $MARIN_ADAMH == 1 ]]; then\n'
    '        pretrain_entrypoint="${repo_dir}/runtime/pretrain_gpt_marin_adamh.py"\n'
    '    fi\n':
        '    pretrain_entrypoint="${repo_dir}/${PRETRAIN_RUNTIME}"\n',
    '        "LONGPPL_ENABLED=$LONGPPL_ENABLED"\n':
        '        "LONGPPL_ENABLED=$LONGPPL_ENABLED"\n'
        '        "PRETRAIN_RUNTIME=$PRETRAIN_RUNTIME"\n',
    '    if [[ $MARIN_ADAMH == 1 && ! -f ${repo_dir}/runtime/pretrain_gpt_marin_adamh.py ]]; then\n'
    '        die "Marin AdamH runtime is missing from the shared repository"\n'
    '    fi\n':
        '    [[ -f ${repo_dir}/${PRETRAIN_RUNTIME} ]] \\\n'
        '        || die "trusted pretraining runtime is missing: ${repo_dir}/${PRETRAIN_RUNTIME}"\n',
    '        SUCCESS_MARKER EVAL_ONLY LM_EVAL_ENABLED LONGPPL_ENABLED\n':
        '        SUCCESS_MARKER EVAL_ONLY LM_EVAL_ENABLED LONGPPL_ENABLED PRETRAIN_RUNTIME\n',
    '    local pretrain_runtime=runtime/pretrain_gpt_lm_eval.py\n'
    '    if [[ $MARIN_ADAMH == 1 ]]; then\n'
    '        pretrain_runtime=runtime/pretrain_gpt_marin_adamh.py\n'
    '    fi\n':
        '    local pretrain_runtime=$PRETRAIN_RUNTIME\n',
    '        if [[ ( $cmdline == *runtime/pretrain_gpt_lm_eval.py* \\\n'
    '                || $cmdline == *runtime/pretrain_gpt_marin_adamh.py* ) \\\n'
    '            && $cmdline == *"$target_run_dir/checkpoints"* ]]; then\n':
        '        if [[ $cmdline == *"$PRETRAIN_RUNTIME"* \\\n'
        '            && $cmdline == *"$target_run_dir/checkpoints"* ]]; then\n',
    'examples/training/baseline/data_generation/validate_adamh_v6_blend.py':
        'examples/training/optimizer/data_generation/validate_adamh_v6_blend.py',
    '    container_script="$container_repo_dir/examples/training/baseline"\n':
        '    container_script="$container_repo_dir/examples/training/optimizer"\n',
}
for old, new in replacements.items():
    count = source.count(old)
    if count < 1:
        raise SystemExit(f"error: missing trusted launcher patch anchor in {path}: {old!r}")
    source = source.replace(old, new)
path.write_text(source)
PY

[[ -f ${destination}/examples/training/optimizer/README.md ]] || {
    printf 'error: materialized project is missing the optimizer README\n' >&2
    exit 2
}
[[ -f ${destination}/examples/training/optimizer/runtime/pretrain_gpt_marin_adamh.py ]] || {
    printf 'error: materialized project is missing the optimizer runtime\n' >&2
    exit 2
}
[[ -f ${destination}/examples/training/optimizer/runtime/optimizer_candidate.py ]] || {
    printf 'error: materialized project is missing the candidate hook\n' >&2
    exit 2
}
unexpected_examples=$(find "${destination}/examples/training" -mindepth 1 -maxdepth 1 \
    ! -name optimizer -printf '%f\n')
[[ -z ${unexpected_examples} ]] || {
    printf 'error: non-optimizer training examples leaked into task project:\n%s\n' \
        "${unexpected_examples}" >&2
    exit 2
}

printf 'materialized optimizer project: %s\n' "${destination}"
printf 'source mode:   current working-tree snapshot\n'
printf 'Bridge commit: %s\n' "${bridge_commit}"
printf 'MCore commit:  %s\n' "${mcore_commit}"
