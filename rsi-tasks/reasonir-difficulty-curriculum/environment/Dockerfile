FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel@sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git git-lfs build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /tmp/reasonir-requirements.lock
RUN python -m pip install --no-cache-dir --requirement /tmp/reasonir-requirements.lock \
    && rm /tmp/reasonir-requirements.lock

RUN git clone https://github.com/facebookresearch/ReasonIR.git /opt/reasonir-src \
    && git -C /opt/reasonir-src checkout --detach 0aac96269e455965949df16520fab72da68ffc22 \
    && test "$(git -C /opt/reasonir-src rev-parse HEAD)" = "0aac96269e455965949df16520fab72da68ffc22" \
    && git clone https://github.com/ContextualAI/gritlm.git /opt/gritlm-src \
    && git -C /opt/gritlm-src checkout --detach 971068105a8508bca421841c59fddba7f6596402 \
    && test "$(git -C /opt/gritlm-src rev-parse HEAD)" = "971068105a8508bca421841c59fddba7f6596402" \
    && git clone https://github.com/beir-cellar/beir.git /opt/beir-src \
    && git -C /opt/beir-src checkout --detach 6ef8c9097ebfb203ad360bd64e0cfb93e64f4a44 \
    && test "$(git -C /opt/beir-src rev-parse HEAD)" = "6ef8c9097ebfb203ad360bd64e0cfb93e64f4a44" \
    && rm -rf /opt/reasonir-src/.git /opt/gritlm-src/.git /opt/beir-src/.git

COPY asset-lock.json prepare_assets.py /opt/reasonir-task/build/
COPY runtime /opt/reasonir-task/runtime
RUN TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
    python /opt/reasonir-task/build/prepare_assets.py \
      --lock /opt/reasonir-task/build/asset-lock.json \
      --reasonir-source /opt/reasonir-src \
      --destination /opt/reasonir-task \
    && rm -rf /root/.cache /tmp/* /opt/reasonir-task/build \
    && install -m 0555 /opt/reasonir-task/runtime/validate_candidate.py /usr/local/bin/validate-reasonir-candidate

RUN mkdir -p /workspace \
    && cp -a /opt/reasonir-src /workspace/reasonir \
    && mkdir -p /workspace/policy /workspace/trials /workspace/submission \
    && cp /opt/reasonir-task/runtime/default-policy.json /workspace/policy/policy.json \
    && find /workspace -type d -exec chmod 0755 {} + \
    && find /workspace/reasonir -type f -exec chmod 0444 {} + \
    && chmod 0644 /workspace/policy/policy.json \
    && python /opt/reasonir-task/runtime/starting_state.py create --root /workspace --output /opt/reasonir-task/runtime/workspace-tree.json \
    && python /opt/reasonir-task/runtime/starting_state.py check --root /workspace --manifest /opt/reasonir-task/runtime/workspace-tree.json

WORKDIR /workspace

# The pinned base ships Ninja 1.11.1.1 with Tag headers after a blank line.
# Repair metadata only; preserve its version/executable and all model layers.
COPY fix_ninja_metadata.py /tmp/fix_ninja_metadata.py
RUN python /tmp/fix_ninja_metadata.py \
    && python -m pip check \
    && rm /tmp/fix_ninja_metadata.py
