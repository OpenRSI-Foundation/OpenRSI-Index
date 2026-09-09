# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ARG GPIC_COMMIT=afa82daab73cfeab4ef9fbd174ff62b68bffc456
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned scientific source. /opt/project is the pristine tree the verifier
# diffs against; /app/project is the agent's writable copy. The main.py
# import fix (src.lightning_*_keshik -> src.lightning_*_gpic) is task
# infrastructure applied to BOTH copies so it never appears as an agent diff.
COPY patches/pixelgen-main-import.patch /tmp/pixelgen-main-import.patch
RUN git clone https://github.com/keshik6/gpic.git /opt/project/gpic \
    && git -C /opt/project/gpic checkout "${GPIC_COMMIT}" \
    && test "$(git -C /opt/project/gpic rev-parse HEAD)" = "${GPIC_COMMIT}" \
    && git -C /opt/project/gpic apply /tmp/pixelgen-main-import.patch \
    && python -c "import ast; s = open('/opt/project/gpic/baselines/PixelGen/main.py').read(); \
ast.parse(s); assert 'lightning_data_gpic' in s and 'keshik' not in s" \
    && chmod -R a+rX /opt/project

# PixelGen requirements (torch==2.7.1 comes from the base image; gradio is
# demo-only and skipped) + the pinned gpic_eval toolkit (DINOv2-only install;
# no tensorflow).
RUN python -m pip install \
        lightning omegaconf diffusers transformers \
        'jsonargparse[signatures]==4.41.0' timm accelerate wandb webdataset \
        tensorboard einops lpips pyarrow scipy scikit-learn tqdm \
    && python -m pip install /opt/project/gpic/gpic_eval \
    && python -c "import torch, lightning, gpic_eval, webdataset; \
v = torch.__version__.split('+')[0]; \
assert tuple(map(int, v.split('.')[:2])) >= (2, 7), v"

# Staged pretrained allowlist: the Qwen3-1.7B text encoder is the ONLY
# external image-adjacent checkpoint in the agent image besides the
# screening cache below (policy RH-005).
# BUILD NOTE: pass the HF token as a build secret:
#   docker build --secret id=hf_token,env=HF_TOKEN ...
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-1.7B', local_dir='/opt/hf/Qwen3-1.7B', \
token=os.environ.get('HF_TOKEN') or None); print('Qwen3-1.7B staged')"

# torch-hub DINOv2 cache for the screening tool ONLY (val-split FD proxy via
# /task-tools/gpic_local_eval.py). Policy RH-004 bans every other use; the
# sealed test reference lives only in the tests image. (Author decision:
# stage-for-screening; alternative: keep DINO out and screen with Inception-FID.)
ENV TORCH_HOME=/opt/torch-hub
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')" \
    && chmod -R a+rX /opt/torch-hub /opt/hf

COPY --chmod=755 task-tools/gpic_async_run.py /task-tools/gpic_async_run.py
COPY --chmod=755 task-tools/gpic_task_tool.py /task-tools/gpic_task_tool.py
COPY --chmod=755 task-tools/gpic_local_eval.py /task-tools/gpic_local_eval.py
COPY --chmod=755 launch_gpic_1x8.sh /task-tools/launch_gpic_1x8.sh

# Contract defaults; the harness may override via task.toml [environment.env].
ENV GPIC_GPU_HOURS_CAP=1000.0 \
    GPIC_TOPOLOGY_GPUS=8 \
    GPIC_IMAGES_CEILING=100000000 \
    GPIC_TRAIN_DIR=/datasets/gpic/train \
    GPIC_VAL_DIR=/datasets/gpic/val \
    GPIC_EVAL_JSONL=/datasets/gpic/gpic_eval_50k.jsonl \
    GPIC_VAL_STATS=/datasets/gpic/val_stats.npz \
    GPIC_PRETRAINED_ALLOWLIST=Qwen/Qwen3-1.7B \
    GPIC_QWEN_DIR=/opt/hf/Qwen3-1.7B \
    HF_HUB_OFFLINE=1

RUN mkdir -p /app/output /logs/artifacts \
    && cp -a /opt/project /app/project \
    && chmod -R a+rwX /app/project /app/output
WORKDIR /app/project/gpic/baselines/PixelGen
CMD ["bash"]
