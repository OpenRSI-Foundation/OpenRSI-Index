# The task image; Work adds the Agent CLI and each Judge starts from that
# clean Work image, so Harbor installed here is in both. There is no Docker
# client or socket. Every Harbor environment is created by the host broker
# behind the injected endpoint, through the rsi_sandbox_harbor plugin found
# on $RSI_SANDBOX_PYTHONPATH.
FROM python:3.13-slim-bookworm
RUN pip install --no-cache-dir harbor==0.21.0
WORKDIR /workspace
