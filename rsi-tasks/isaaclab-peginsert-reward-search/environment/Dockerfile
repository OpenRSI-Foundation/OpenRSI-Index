FROM python:3.11.16-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/peginsert_public \
    OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    ISAACSIM_PATH=/usr/local/lib/python3.11/site-packages/isaacsim \
    ISAAC_NUCLEUS_DIR=/opt/peginsert_assets/Isaac

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git git-lfs libatomic1 libegl1 libgl1 libglib2.0-0 libglu1-mesa libgomp1 libsm6 libx11-6 libxext6 libxrender1 libxt6 libxi6 libxrandr2 libxinerama1 libxcursor1 libfontconfig1 libfreetype6 libvulkan1 wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock /opt/peginsert_build/requirements.lock
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation \
        --index-url=https://pypi.org/simple \
        --extra-index-url=https://pypi.nvidia.com \
        --extra-index-url=https://download.pytorch.org/whl/cu128 \
        -r /opt/peginsert_build/requirements.lock

RUN git clone https://github.com/isaac-sim/IsaacLab.git /opt/isaaclab \
    && cd /opt/isaaclab \
    && git checkout b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8 \
    && test "$(git rev-parse HEAD)" = b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8 \
    && git clone https://github.com/isaac-sim/rl_games.git /opt/rl_games \
    && cd /opt/rl_games \
    && git checkout 6b3534f29568158e9e29ec8bf83cc88fce5f0cae \
    && test "$(git rev-parse HEAD)" = 6b3534f29568158e9e29ec8bf83cc88fce5f0cae \
    && python -m pip install --no-cache-dir --no-deps --index-url=https://pypi.org/simple poetry-core==2.4.1 \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation -e /opt/isaaclab/source/isaaclab -e /opt/isaaclab/source/isaaclab_assets -e /opt/isaaclab/source/isaaclab_tasks -e /opt/isaaclab/source/isaaclab_rl \
    && python -m pip install --no-cache-dir --no-deps --no-build-isolation /opt/rl_games \
    && python -m pip uninstall -y poetry-core \
    && git -C /opt/isaaclab clean -ffdx \
    && test -z "$(git -C /opt/isaaclab status --porcelain=v1 --untracked-files=all)" \
    && git -C /opt/rl_games clean -ffdx \
    && test -z "$(git -C /opt/rl_games status --porcelain=v1 --untracked-files=all)"

COPY prepare_assets.py convert_reference.py runtime_manifest.py /opt/peginsert_build/
# The USD bindings (pxr) ship inside pip Isaac Sim's omni.usd.libs extension; expose them to the
# asset mirror without launching Kit (no GPU at build time).
RUN set -- /usr/local/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-* \
    && test "$#" -eq 1 && test -d "$1/pxr" \
    && PYTHONPATH="$1:/opt/peginsert_public" LD_LIBRARY_PATH="$1/bin" python /opt/peginsert_build/prepare_assets.py --output /opt/peginsert_assets \
    && mkdir -p /opt/peginsert_reference/source \
    && curl -fL --retry 5 -o /opt/peginsert_reference/source/checkpoint.pth https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/IsaacLab/PretrainedCheckpoints/rl_games/Isaac-Factory-PegInsert-Direct-v0/checkpoint.pth \
    && python /opt/peginsert_build/convert_reference.py /opt/peginsert_reference/source/checkpoint.pth /opt/peginsert_reference/converted \
    && rmdir /opt/peginsert_reference/source

RUN mkdir -p /opt/peginsert_public
COPY baseline_reward.json fixed_env.json reward_graph.py driver_common.py work_train.py /opt/peginsert_public/
RUN chmod 0555 /opt/peginsert_public/reward_graph.py /opt/peginsert_public/work_train.py \
    && ln -s /opt/peginsert_public/reward_graph.py /usr/local/bin/validate-reward \
    && ln -s /opt/peginsert_public/work_train.py /usr/local/bin/peginsert-develop \
    && python /opt/peginsert_build/runtime_manifest.py --lock /opt/peginsert_build/requirements.lock --output /opt/peginsert_public/runtime_manifest.json \
    && chmod 0555 /opt/isaaclab /opt/rl_games /opt/peginsert_assets /opt/peginsert_reference /opt/peginsert_public

RUN mkdir -p /workspace/research \
    && install -m 0644 /opt/peginsert_public/baseline_reward.json /workspace/reward.json \
    && find /workspace -xdev -type f ! -path /workspace/reward.json -delete \
    && find /workspace -xdev -type l -delete \
    && find /workspace -xdev -mindepth 1 -maxdepth 1 ! -name reward.json ! -name research -exec false {} + \
    && test -d /workspace/research \
    && test -z "$(find /workspace/research -mindepth 1 -print -quit)"

WORKDIR /workspace
