#!/usr/bin/env bash
# Launch translation for the gpic-generation task — the single-node analog
# of dinov3's launch_dinov3_4x8.sh. This is task infrastructure, not an
# optimization variable.
#
# Usage:  launch_gpic_1x8.sh <config.yaml> [extra LightningCLI args...]
#
# Contract: released topology is 1 node x 8 GPUs, exactly. The trainer is
# LightningCLI (DDP spawned internally), so no torchrun rendezvous is
# needed. The launcher always pins the staged shard directories so the
# training data path is not an agent-controlled variable, and always keeps
# logging offline.
#
# GPIC_SMOKE=1 adds resource-only overrides (max_steps, tiny shard pattern,
# small batch) for infrastructure debugging; smoke runs are not comparable
# to the sealed baseline and never count as scientific evidence.
#
# Known repository pitfalls handled by task infrastructure:
#   - main.py's broken src.lightning_*_keshik imports are patched to the
#     shipped src.lightning_*_gpic modules at image build (both /opt and
#     /app copies), so the fix never appears as an agent diff;
#   - the shipped sampling config sets guidance 4.0 — the task contract
#     fixes guidance to 1.0; when using the pinned sampling config, pass
#     --model.diffusion_sampler.init_args.guidance 1.0 (the frozen verifier
#     and policy RH-006 enforce pure conditional sampling regardless).
set -euo pipefail

CONFIG="${1:?usage: launch_gpic_1x8.sh <config.yaml> [args...]}"
shift

REPO_ROOT="${GPIC_REPO_ROOT:-/app/project/gpic/baselines/PixelGen}"
TRAIN_DIR="${GPIC_TRAIN_DIR:-/datasets/gpic/train}"
VAL_DIR="${GPIC_VAL_DIR:-/datasets/gpic/val}"
DEVICES="${GPIC_TOPOLOGY_GPUS:-8}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1

SMOKE_ARGS=()
if [ "${GPIC_SMOKE:-0}" = "1" ]; then
  # >= one shard PER RANK: webdataset split_by_node hands whole shards to
  # ranks, so fewer shards than ranks leaves some ranks with empty streams
  # and DDP deadlocks in the first all-reduce (observed). Validation is
  # skipped for the same reason (one val shard cannot feed 8 ranks).
  SMOKE_ARGS+=(
    --trainer.max_steps "${GPIC_SMOKE_STEPS:-20}"
    --trainer.limit_val_batches 0
    --trainer.num_sanity_val_steps 0
    --data.train_dataset.init_args.tar_pattern "gpic_train_{00000..00007}.tar"
  )
  echo "[launch_gpic_1x8] SMOKE MODE: infrastructure debugging only" >&2
fi

# torchrun (external process creation) rather than Lightning's internal
# subprocess launcher: inside containers the internal launcher can hang at
# "MEMBER: 1/8" waiting for child ranks that never come up (observed on
# Slurm+pyxis). With LOCAL_RANK set by torchrun, Lightning attaches to the
# pre-created processes.
cd "${REPO_ROOT}"
exec torchrun --standalone --nproc_per_node "${DEVICES}" main.py fit \
  --config "${CONFIG}" \
  --trainer.devices "${DEVICES}" \
  --trainer.num_nodes 1 \
  --data.train_dataset.init_args.tar_dir "${TRAIN_DIR}" \
  --data.val_dataset.init_args.tar_dir "${VAL_DIR}" \
  "${SMOKE_ARGS[@]}" \
  "$@"
