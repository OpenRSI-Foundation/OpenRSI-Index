FROM nvidia/cuda:12.9.1-devel-ubuntu22.04@sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8

ENV DEBIAN_FRONTEND=noninteractive MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 MAKEFLAGS=-j8
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev build-essential cmake ninja-build git ca-certificates curl libffi-dev libssl-dev libseccomp2 libgl1 libglib2.0-0 swig autoconf automake bison flex libncurses-dev && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY make-bounded.sh /usr/local/bin/make
RUN chmod 755 /usr/local/bin/make
COPY requirements.lock /opt/build/requirements.lock
RUN python -m pip install --no-cache-dir pip==25.2 setuptools==78.1.1 wheel==0.45.1 cffi==2.0.0 pycparser==2.23 && python -m pip install --no-cache-dir --no-build-isolation --only-binary=textworld -r /opt/build/requirements.lock && python -m pip check
COPY vendor /opt/build/vendor
COPY model_identity.json prepare_assets.py /opt/build/
RUN python /opt/build/prepare_assets.py && rm -rf /opt/build/vendor
COPY patches.json backend_patches.json apply_backend_patches.py /opt/build/
RUN python /opt/build/prepare_assets.py --apply-patches-only && python /opt/build/apply_backend_patches.py
COPY support /opt/opentinker_task
COPY dev_manifest.json /opt/opentinker_task/dev_manifest.json
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp/cache PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false ALFWORLD_DATA=/opt/data/alfworld VLLM_NO_USAGE_STATS=1 VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN DO_NOT_TRACK=1 RAY_USAGE_STATS_ENABLED=0 WANDB_MODE=disabled
# Last filesystem initialization: only these two files exist in the pristine WORKDIR.
RUN mkdir -p /workspace/candidate && cp /opt/opentinker_task/reference/policy.py /workspace/candidate/policy.py && cp /opt/opentinker_task/reference/source_manifest.json /workspace/candidate/source_manifest.json && python -c 'from pathlib import Path; p=Path("/workspace"); assert sorted(str(x.relative_to(p)) for x in p.rglob("*")) == ["candidate", "candidate/policy.py", "candidate/source_manifest.json"]'
WORKDIR /workspace
