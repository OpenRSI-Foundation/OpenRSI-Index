FROM nvcr.io/nvidia/pytorch:26.06-py3@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1

RUN python3 -m venv --system-site-packages /opt/venv \
    && mkdir -p /workspace /task-tools /logs/verifier

COPY smoke.py /task-tools/smoke.py
COPY judge.py /task-tools/judge.py
WORKDIR /workspace
