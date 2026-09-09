# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ARG GPIC_COMMIT=afa82daab73cfeab4ef9fbd174ff62b68bffc456
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Pristine pinned tree: the policy gate diffs the candidate workspace
# against this. The same infrastructure patch as the environment image is
# applied so the fix never appears as an agent diff.
COPY pixelgen-main-import.patch /tmp/pixelgen-main-import.patch
RUN git clone https://github.com/keshik6/gpic.git /opt/project/gpic \
    && git -C /opt/project/gpic checkout "${GPIC_COMMIT}" \
    && test "$(git -C /opt/project/gpic rev-parse HEAD)" = "${GPIC_COMMIT}" \
    && git -C /opt/project/gpic apply /tmp/pixelgen-main-import.patch \
    && chmod -R a+rX /opt/project

# The verifier both scores images with the pinned gpic_eval AND executes the
# agent's generate.py for the spot gate, so the interpreter stack matches
# the environment image's pip section exactly.
RUN python -m pip install \
        lightning omegaconf diffusers transformers \
        'jsonargparse[signatures]==4.41.0' timm accelerate wandb webdataset \
        tensorboard einops lpips pyarrow scipy scikit-learn tqdm pyyaml \
    && python -m pip install /opt/project/gpic/gpic_eval \
    && python3 -m venv /opt/policy-venv \
    && /opt/policy-venv/bin/pip install --no-cache-dir litellm pyyaml

# The verifier runs the agent's generate.py for the spot gate, so it needs
# the SAME allowlisted text encoder the agent used. Stage Qwen3-1.7B
# identically to the environment image (build secret: hf_token).
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN=$(cat /run/secrets/hf_token 2>/dev/null || true) \
    python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-1.7B', local_dir='/opt/hf/Qwen3-1.7B', \
token=os.environ.get('HF_TOKEN') or None); print('Qwen3-1.7B staged')"
ENV GPIC_QWEN_DIR=/opt/hf/Qwen3-1.7B

# Frozen metric assets, verifier-only: DINOv2 hub weights, the sealed test
# reference statistics, and the verifier-owned copy of the frozen caption
# set (sha256-matched to the agent-visible copy at staging).
# BUILD NOTE: place test_stats.npz and gpic_eval_50k.jsonl next to this
# Dockerfile before building (produced by environment/stage_data.sh with a
# gated-HF token); they are baked in so no agent-writable path is trusted.
ENV TORCH_HOME=/opt/torch-hub
RUN python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')"
COPY test_stats.npz /opt/gpic-ref/test_stats.npz
COPY gpic_eval_50k.jsonl /opt/gpic-ref/gpic_eval_50k.jsonl
RUN chmod -R a+rX /opt/gpic-ref /opt/torch-hub

COPY policy.yaml /tests/policy.yaml
COPY policy_check.py evaluate.py score.py /tests/
COPY --chmod=755 test.sh /tests/test.sh
RUN mkdir -p /app/project /app/output /logs/verifier
CMD ["bash"]
