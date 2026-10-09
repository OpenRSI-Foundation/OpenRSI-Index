FROM python:3.12-slim-bookworm
RUN mkdir -p /workspace /opt/fixture/bin && ln -s /usr/local/bin/python3 /opt/fixture/bin/task-python
ENV PATH="/opt/fixture/bin:${PATH}"
WORKDIR /workspace
