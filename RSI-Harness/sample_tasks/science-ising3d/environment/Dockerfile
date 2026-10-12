# The task image. Work starts from it plus the Agent CLI, and every public
# Judge starts from that same clean image, read-only, so nothing the Agent
# installs in Work reaches the Judge. The rubrics and the Judge are not copied
# here; Work reads them read-only at /tests. Python 3.11, standard library only.
FROM python@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b
ENV PYTHONDONTWRITEBYTECODE=1

COPY starter/ /opt/science/starter/
COPY starter/ /workspace/candidate/

WORKDIR /workspace
