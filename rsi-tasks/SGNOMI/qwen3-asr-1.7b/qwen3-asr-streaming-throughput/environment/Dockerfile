FROM nvidia/cuda:13.0.0-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev git ca-certificates build-essential cmake ninja-build pkg-config autoconf automake libtool libnuma-dev libibverbs-dev librdmacm-dev libsndfile1 ffmpeg sox libsox-dev libfst-dev patchelf && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/openucx/ucx.git /tmp/ucx && git -C /tmp/ucx checkout d8e50df6651b9ea5b76f23aee0aefbf053a4137a && test "$(git -C /tmp/ucx rev-parse HEAD)" = d8e50df6651b9ea5b76f23aee0aefbf053a4137a && cd /tmp/ucx && ./autogen.sh && ./contrib/configure-release-mt --enable-shared --disable-static --disable-doxygen-doc --enable-optimizations --enable-cma --enable-devel-headers --with-cuda=/usr/local/cuda --with-verbs --with-dm --prefix=/usr/local && make -j8 && make install-strip && ldconfig && rm -rf /tmp/ucx
RUN python3.12 -m venv /opt/asr-venv
ENV PATH=/opt/asr-venv/bin:$PATH VIRTUAL_ENV=/opt/asr-venv
COPY requirements.lock source-requirements.lock /opt/asr-reference/
RUN python -m pip install --no-cache-dir --index-url=https://pypi.org/simple --require-hashes --only-binary=:all: --no-deps -r /opt/asr-reference/requirements.lock
RUN python -m pip install --no-cache-dir --index-url=https://pypi.org/simple --require-hashes --no-build-isolation --no-deps -r /opt/asr-reference/source-requirements.lock
COPY source.tar.gz source-manifest.json model-manifest.json initialize.py restore.py candidate.yaml /opt/asr-reference/
RUN python -I -B /opt/asr-reference/initialize.py prepare && python -m pip install --no-deps --no-build-isolation /opt/asr-reference/source
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ENV LD_LIBRARY_PATH=/opt/asr-venv/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/lib:/usr/local/cuda/lib64
RUN python -I -B /opt/asr-reference/initialize.py close
WORKDIR /workspace
