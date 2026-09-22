#!/usr/bin/env bash

# Locked six-scale scientific profiles for optimizer-encoding transfer.
# The candidate selects OPTIMIZER_SCALE only. Every other value is trusted.

set -euo pipefail

: "${OPTIMIZER_SCALE:?set OPTIMIZER_SCALE to E0, E1, E2, E3, E4, or E5}"

OPTIMIZER_ASSET_ROOT=${OPTIMIZER_ASSET_ROOT:-/data19/<USER>/marin-adamh}
OPTIMIZER_PROJECT=/app/project
OPTIMIZER_OUTPUT_ROOT=/app/output
OPTIMIZER_HOST_OUTPUT_FILE=/run-contract/HOST_OUTPUT_ROOT

export TOKENIZER_PROFILE=llama3
export DATA_TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
export DATA_TOKENIZER_REVISION=d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
export LM_EVAL_TOKENIZER_MODEL=meta-llama/Llama-3.1-8B
export LM_EVAL_TOKENIZER_REVISION=d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
export VOCAB_SIZE=128256
export EOD_ID=128001

export MODEL_ROTARY_BASE=500000
export MODEL_TIE_EMBEDDINGS=0
export MAKE_VOCAB_SIZE_DIVISIBLE_BY=1
export SEQUENCE_LENGTH=4096
export SEED=0

export TENSOR_MODEL_PARALLEL_SIZE=1
export PIPELINE_MODEL_PARALLEL_SIZE=1
export SEQUENCE_PARALLEL=0
export DECODER_FIRST_PIPELINE_NUM_LAYERS=0
export DECODER_LAST_PIPELINE_NUM_LAYERS=0
export MARIN_ADAMH=1
export MARIN_ADAMH_DATA=1
export MARIN_FULL_RECOMPUTE=0
export MARIN_GRADIENT_ACCUMULATION_FUSION=1
export MARIN_ADAMH_BETA1=0.9
export MARIN_ADAMH_MAX_GRAD_NORM=0.1

case "${OPTIMIZER_SCALE}" in
    E0)
        export SCALE_KEY=9e18-p550m-t2p904b-d1152-l12
        export SOURCE_PARAMETER_COUNT=550337664
        export MODEL_NUM_LAYERS=12 MODEL_HIDDEN_SIZE=1152 MODEL_FFN_HIDDEN_SIZE=4608
        export MODEL_NUM_ATTENTION_HEADS=9 MODEL_NUM_QUERY_GROUPS=9
        export TARGET_TOKENS=2904358912 TRAIN_ITERS=44317 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SIZE=2
        export LR_WARMUP_ITERS=4431 LR_WSD_DECAY_ITERS=8863
        export MARIN_ADAMH_LR=0.003011461785505323
        export MARIN_ADAM_LR=0.000304311605183897
        export MARIN_ADAMH_BETA2=0.9999 MARIN_ADAMH_EPSILON=3.988017477238883e-08
        export ACTIVATION_OFFLOAD_FRACTION=1.0
        export LSF_SLOTS=8 LSF_WALLTIME=08:00 LSF_MEMORY=32768
        export DATASET_SUBSET=adamh-v6-nine-component
        export DATASET_REPO=marin-community/adamh-scaling-v6
        export DATASET_REVISION=9e946e47cfa74efe43cca1b74360c27f64dd1226
        export DATASET_TOKENS=12143955642 DATA_PREFIX_BASENAME=adamh_v6 SHARD_COUNT=211
        export DATA_CACHE_NAMESPACE=marin-adamh-v6-llama3
        export MARIN_PRETOKENIZED_DATA_SUBDIR=adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B
        ;;
    E1)
        export SCALE_KEY=1p8e19-p837m-t3p613b-d1408-l15
        export SOURCE_PARAMETER_COUNT=837007744
        export MODEL_NUM_LAYERS=15 MODEL_HIDDEN_SIZE=1408 MODEL_FFN_HIDDEN_SIZE=5632
        export MODEL_NUM_ATTENTION_HEADS=11 MODEL_NUM_QUERY_GROUPS=11
        export TARGET_TOKENS=3612672000 TRAIN_ITERS=55125 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SIZE=2
        export LR_WARMUP_ITERS=5512 LR_WSD_DECAY_ITERS=11025
        export MARIN_ADAMH_LR=0.0028206162035338996
        export MARIN_ADAM_LR=0.0002728535329687243
        export MARIN_ADAMH_BETA2=0.9999 MARIN_ADAMH_EPSILON=4.4478075354043814e-08
        export ACTIVATION_OFFLOAD_FRACTION=0
        export LSF_SLOTS=8 LSF_WALLTIME=08:00 LSF_MEMORY=32768
        export DATASET_SUBSET=adamh-v6-nine-component
        export DATASET_REPO=marin-community/adamh-scaling-v6
        export DATASET_REVISION=9e946e47cfa74efe43cca1b74360c27f64dd1226
        export DATASET_TOKENS=12143955642 DATA_PREFIX_BASENAME=adamh_v6 SHARD_COUNT=211
        export DATA_CACHE_NAMESPACE=marin-adamh-v6-llama3
        export MARIN_PRETOKENIZED_DATA_SUBDIR=adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B
        ;;
    E2)
        export SCALE_KEY=3e19-p998m-t4p983b-d1536-l16
        export SOURCE_PARAMETER_COUNT=998036992
        export MODEL_NUM_LAYERS=16 MODEL_HIDDEN_SIZE=1536 MODEL_FFN_HIDDEN_SIZE=6144
        export MODEL_NUM_ATTENTION_HEADS=12 MODEL_NUM_QUERY_GROUPS=12
        export TARGET_TOKENS=4982571008 TRAIN_ITERS=38014 GLOBAL_BATCH_SIZE=32 MICRO_BATCH_SIZE=4
        export LR_WARMUP_ITERS=3801 LR_WSD_DECAY_ITERS=7602
        export MARIN_ADAMH_LR=0.0036221923183594514
        export MARIN_ADAM_LR=0.00032857316983952165
        export MARIN_ADAMH_BETA2=0.9999 MARIN_ADAMH_EPSILON=3.6935456434033684e-08
        export ACTIVATION_OFFLOAD_FRACTION=0
        export LSF_SLOTS=8 LSF_WALLTIME=08:00 LSF_MEMORY=32768
        export DATASET_SUBSET=adamh-v6-nine-component
        export DATASET_REPO=marin-community/adamh-scaling-v6
        export DATASET_REVISION=9e946e47cfa74efe43cca1b74360c27f64dd1226
        export DATASET_TOKENS=12143955642 DATA_PREFIX_BASENAME=adamh_v6 SHARD_COUNT=211
        export DATA_CACHE_NAMESPACE=marin-adamh-v6-llama3
        export MARIN_PRETOKENIZED_DATA_SUBDIR=adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B
        ;;
    E3)
        export SCALE_KEY=9e19-p1385m-t10p560b-d1792-l18
        export SOURCE_PARAMETER_COUNT=1384584448
        export MODEL_NUM_LAYERS=18 MODEL_HIDDEN_SIZE=1792 MODEL_FFN_HIDDEN_SIZE=7168
        export MODEL_NUM_ATTENTION_HEADS=14 MODEL_NUM_QUERY_GROUPS=14
        export TARGET_TOKENS=10559946752 TRAIN_ITERS=40283 GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=2
        export LR_WARMUP_ITERS=4028 LR_WSD_DECAY_ITERS=8056
        export MARIN_ADAMH_LR=0.004089065182552466
        export MARIN_ADAM_LR=0.0003191853718412673
        export MARIN_ADAMH_BETA2=0.9999 MARIN_ADAMH_EPSILON=3.8021792571481954e-08
        export ACTIVATION_OFFLOAD_FRACTION=0
        export LSF_SLOTS=32 LSF_WALLTIME=08:00 LSF_MEMORY=32768
        export DATASET_SUBSET=adamh-v6-nine-component
        export DATASET_REPO=marin-community/adamh-scaling-v6
        export DATASET_REVISION=9e946e47cfa74efe43cca1b74360c27f64dd1226
        export DATASET_TOKENS=12143955642 DATA_PREFIX_BASENAME=adamh_v6 SHARD_COUNT=211
        export DATA_CACHE_NAMESPACE=marin-adamh-v6-llama3
        export MARIN_PRETOKENIZED_DATA_SUBDIR=adamh-v6-pretokenized-12b/meta-llama--Llama-3.1-8B
        ;;
    E4)
        export SCALE_KEY=1p8e20-p1935m-t14p805b-d2048-l21
        export SOURCE_PARAMETER_COUNT=1934716160
        export MODEL_NUM_LAYERS=21 MODEL_HIDDEN_SIZE=2048 MODEL_FFN_HIDDEN_SIZE=8192
        export MODEL_NUM_ATTENTION_HEADS=16 MODEL_NUM_QUERY_GROUPS=16
        export TARGET_TOKENS=14805106688 TRAIN_ITERS=56477 GLOBAL_BATCH_SIZE=64 MICRO_BATCH_SIZE=2
        export LR_WARMUP_ITERS=5647 LR_WSD_DECAY_ITERS=11295
        export MARIN_ADAMH_LR=0.003694869377544804
        export MARIN_ADAM_LR=0.0002695678364744162
        export MARIN_ADAMH_BETA2=0.9999 MARIN_ADAMH_EPSILON=4.502020774704621e-08
        export ACTIVATION_OFFLOAD_FRACTION=0
        export LSF_SLOTS=32 LSF_WALLTIME=12:00 LSF_MEMORY=65536
        export DATASET_SUBSET=adamh-v6-nine-component-view-1p8e20
        export DATASET_REPO=nvidia/Nemotron-CC+bigcode/starcoderdata+EleutherAI/proof-pile-2
        export DATASET_REVISION=locked-source-pins-in-manifest
        export DATASET_TOKENS=17322604062 DATA_PREFIX_BASENAME=adamh_v6_1p8e20_view SHARD_COUNT=271
        export DATA_CACHE_NAMESPACE=marin-adamh-v6-view-1p8e20-llama3
        export MARIN_PRETOKENIZED_DATA_SUBDIR=adamh-v6-training-views/1p8e20-t14p805b/meta-llama--Llama-3.1-8B
        ;;
    E5)
        export SCALE_KEY=3e20-p2545m-t18p617b-d2304-l23
        export SOURCE_PARAMETER_COUNT=2544614912
        export MODEL_NUM_LAYERS=23 MODEL_HIDDEN_SIZE=2304 MODEL_FFN_HIDDEN_SIZE=9216
        export MODEL_NUM_ATTENTION_HEADS=18 MODEL_NUM_QUERY_GROUPS=18
        export TARGET_TOKENS=18617466880 TRAIN_ITERS=35510 GLOBAL_BATCH_SIZE=128 MICRO_BATCH_SIZE=1
        export LR_WARMUP_ITERS=3551 LR_WSD_DECAY_ITERS=7102
        export MARIN_ADAMH_LR=0.004878221521523273
        export MARIN_ADAM_LR=0.00033996075743610645
        export MARIN_ADAMH_BETA2=0.99980001 MARIN_ADAMH_EPSILON=3.5698237912888785e-08
        export ACTIVATION_OFFLOAD_FRACTION=0
        export LSF_SLOTS=128 LSF_WALLTIME=08:00 LSF_MEMORY=65536
        export DATASET_SUBSET=adamh-v6-nine-component-view-3e20
        export DATASET_REPO=nvidia/Nemotron-CC+bigcode/starcoderdata+EleutherAI/proof-pile-2
        export DATASET_REVISION=locked-source-pins-in-manifest
        export DATASET_TOKENS=21674496950 DATA_PREFIX_BASENAME=adamh_v6_3e20_view SHARD_COUNT=339
        export DATA_CACHE_NAMESPACE=marin-adamh-v6-view-3e20-llama3
        export MARIN_PRETOKENIZED_DATA_SUBDIR=adamh-v6-training-views/3e20-t18p617b/meta-llama--Llama-3.1-8B
        ;;
    *)
        printf 'error: unknown OPTIMIZER_SCALE: %s\n' "${OPTIMIZER_SCALE}" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

export GPU_COUNT=${GPU_COUNT:-${LSF_SLOTS}}
export EXPECTED_NODES=$((GPU_COUNT / 8))
export GPUS_PER_NODE=8
export LSF_SPAN='span[ptile=8]'
export PROFILE="optimizer-scaling-${OPTIMIZER_SCALE}-${SCALE_KEY}"
export PROFILE_SCRIPT=/task-tools/optimizer_scaling_ladder_profile.sh

export EVAL_INTERVAL=5000
export EVAL_ITERS=0
export SAVE_INTERVAL=5000
export LM_EVAL_ENABLED=1
export LM_EVAL_INTERVAL=$EVAL_INTERVAL
export LM_EVAL_TASKS=marin_paloma
export LM_EVAL_FINAL_ONLY_TASKS=
export LM_EVAL_SEQUENCE_LENGTH=4096
export LM_EVAL_BATCH_SIZE=1
export LM_EVAL_BOOTSTRAP_ITERS=0
export LONGPPL_ENABLED=0

export TOKENIZER_DATA_ROOT="${OPTIMIZER_ASSET_ROOT}/data/${MARIN_PRETOKENIZED_DATA_SUBDIR}"
export DATA_DIR=$TOKENIZER_DATA_ROOT
export DATA_MANIFEST="${TOKENIZER_DATA_ROOT}/manifest.json"
export DATA_BLEND_MANIFEST="${TOKENIZER_DATA_ROOT}/blend.json"
export PALOMA_DATA_DIR="${OPTIMIZER_ASSET_ROOT}/data/paloma-rev-65cd6fc"
export ALLOW_DATA_REUSE=0
export USE_MOCK_DATA=0

export SUCCESS_MARKER=SUCCESS
export EXPECT_FINAL_CHECKPOINT=$((1 - ${EVAL_ONLY:-0}))
export PRETRAIN_RUNTIME=examples/training/optimizer/runtime/pretrain_gpt_marin_adamh.py
export RUN_ROOT="${OPTIMIZER_ASSET_ROOT}"
export CACHE_DIR="${OPTIMIZER_ASSET_ROOT}/cache/marin-adamh-baseline"
export CONTAINER_IMAGE=${OPTIMIZER_SCALING_CONTAINER_IMAGE:-"${OPTIMIZER_ASSET_ROOT}/images/optimizer-update-geometry-environment-accefd7b3a44-r1.sif"}
export APPTAINER=${APPTAINER:-/usr/bin/apptainer}
export WANDB_MODE=offline
export WANDB_PROJECT=optimizer-update-geometry
export WANDB_RUN_NAME=${OPTIMIZER_WANDB_RUN_NAME:-"optimizer-scaling-${OPTIMIZER_SCALE}-candidate"}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# The Spectrum-X plugin can intermittently deadlock its control-plane
# rendezvous at the 128-rank E5 scale even when every HCA is ACTIVE.  Keep E5
# on NCCL's built-in IB transport (still RDMA/GDR) and disable CollNet/SHARP;
# smaller rungs retain the faster Spectrum-X path that is stable for them.
if [[ ${OPTIMIZER_SCALE} == E5 ]]; then
    export NCCL_NET=IB
    export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
    export NCCL_COLLNET_ENABLE=0
    # The immutable candidate launcher forwards an allowlisted environment into
    # Apptainer and predates NCCL_COLLNET_ENABLE.  The APPTAINERENV_ spelling is
    # the trusted infrastructure override that makes the setting reach workers
    # without changing the candidate source hash.
    export APPTAINERENV_NCCL_COLLNET_ENABLE=0
    export NCCL_DEBUG=INFO
fi

for required_path in \
    "${DATA_MANIFEST}" \
    "${DATA_BLEND_MANIFEST}" \
    "${PALOMA_DATA_DIR}" \
    "${CACHE_DIR}/huggingface/hub/models--meta-llama--Llama-3.1-8B"; do
    [[ -e ${required_path} ]] || {
        printf 'error: required staged asset is missing for %s: %s\n' "${OPTIMIZER_SCALE}" "${required_path}" >&2
        return 1 2>/dev/null || exit 1
    }
done

export OPTIMIZER_ASSET_ROOT OPTIMIZER_PROJECT OPTIMIZER_OUTPUT_ROOT OPTIMIZER_HOST_OUTPUT_FILE
