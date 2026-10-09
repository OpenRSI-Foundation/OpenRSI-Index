# The Work image of the vLLM-in-Judge demo. The Judge is derived from Work,
# so everything its fixed procedure needs is installed here: vLLM (in its own
# virtual environment, so its pins never meet Harbor's), Harbor 0.21.0 and
# the Hugging Face CLI (`hf`, from vLLM's environment) the Work agent fetches
# the checkpoint with. There is no Docker client or socket: every Harbor
# environment is created by the host broker behind the injected endpoint,
# through the rsi_sandbox_harbor plugin found on $RSI_SANDBOX_PYTHONPATH.
# The base is pinned by digest, so the build stays cached when the tag moves.
FROM python:3.13-slim-bookworm@sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26
# Triton (vLLM's kernels) compiles its small launcher with the C compiler.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# vLLM 0.11.0 runs on torch 2.8.0 built for CUDA 12.8 (NVIDIA driver >= 570).
# Its dependencies are resolved as of its release (uv --exclude-newer): its
# open ranges would otherwise take today's transformers, which it predates.
RUN pip install --no-cache-dir uv==0.12.21 \
    && python3 -m venv /opt/vllm \
    && uv pip install --no-cache --python /opt/vllm/bin/python \
        --exclude-newer 2025-10-05T00:00:00Z vllm==0.11.0 \
    && pip uninstall -y uv \
    && ln -s /opt/vllm/bin/vllm /opt/vllm/bin/hf /usr/local/bin/
# Harbor's own dependencies (litellm, openai and the rest) resolved as of
# the day this image was verified, so a rebuild keeps the hosted_vllm path
# terminus-2 takes to vLLM.
RUN pip install --no-cache-dir uv==0.12.21 \
    && uv pip install --no-cache --system \
        --exclude-newer 2026-10-01T00:00:00Z harbor==0.21.0 \
    && pip uninstall -y uv
WORKDIR /workspace
