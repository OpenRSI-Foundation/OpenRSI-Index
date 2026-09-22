# The immutable official Python image provides Python 3.12 on Debian Bookworm.
FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e AS upstream-source
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git init /tmp/kev-source \
    && git -C /tmp/kev-source remote add origin https://github.com/jaredpalmer/kev.git \
    && git -C /tmp/kev-source fetch --depth=1 origin c096660c8da20a80ce7c61c63d224960f497623a \
    && git -C /tmp/kev-source checkout --detach FETCH_HEAD \
    && test "$(git -C /tmp/kev-source rev-parse HEAD)" = c096660c8da20a80ce7c61c63d224960f497623a
COPY prepare_source.py /tmp/prepare_source.py
RUN python /tmp/prepare_source.py /tmp/kev-source /opt/kev-reference

FROM python:3.12-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates git libgomp1 util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.10.3
COPY --from=upstream-source /opt/kev-reference /opt/kev-reference
ENV UV_PROJECT_ENVIRONMENT=/opt/kev-venv UV_LINK_MODE=copy
# Exact upstream lock: torch 2.8.0, transformers 5.17.0, peft 0.21.0.
# Do not install the reference as a package: candidate-local kev/ is authoritative.
RUN uv sync --frozen --no-dev --no-install-project --project /opt/kev-reference --python /usr/local/bin/python
ENV PATH="/opt/kev-venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
COPY reference/ /opt/kev-reference/
COPY prepare_assets.py /opt/kev-build/prepare_assets.py
COPY public/ /opt/kev-assets/public/
COPY runtime/ /opt/kev-runtime/
COPY bin/validate-candidate /usr/local/bin/validate-candidate
RUN umask 022 \
    && PYTHONPATH=/opt/kev-reference python /opt/kev-build/prepare_assets.py \
    && python /opt/kev-reference/restore.py \
    && chmod 755 /usr/local/bin/validate-candidate
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
# Work remains root under the Harness contract; Judge drops privileges only for
# the candidate inference subprocess. Judge-only inputs are absent from Work.
WORKDIR /workspace
CMD ["sleep", "infinity"]
