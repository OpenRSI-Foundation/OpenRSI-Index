"""The E2B env backend against real E2B (opt-in: RSI_E2B_KEY_FILE).

Set ``RSI_E2B_KEY_FILE`` to the operator's key file (the key is read only
by the backend). One template is built from python:3.13-slim-bookworm (a
shared cache, kept), four 1-vCPU sandboxes run for about a minute each, and
every sandbox of a test's run is killed and proven gone by metadata at the
end.

A Dockerfile build adds one template on that image (kept, too), and a
small one on a mirror.gcr.io image is built again without E2B's cache;
with ``RSI_E2B_PRIVATE_IMAGE``, one more builds on that image.

A private pull also needs ``RSI_E2B_REGISTRY_AUTH_FILE`` (a Docker-style
auth file) and ``RSI_E2B_PRIVATE_IMAGE``: an image in that registry with a
WORKDIR, a USER and ``ENTRYPOINT ["env", "FROM_ENTRYPOINT=1"]``.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass

import pytest

from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_contracts import SandboxOwner
from rsi_harness.runtime.sandbox_e2b import e2b_client
from rsi_harness.runtime.sandbox_images import pull_reference
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from rsi_harness.runtime.sandbox_remote import (
    kill_run_sandboxes,
    remote_env_runtime,
    run_metadata,
)
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_e2b import (
    TASK,
    build_view,
    e2b_policy_text,
    pull,
    pull_view,
    ready,
    run,
    single,
    start_exec,
    wait_exec,
)
from tests.sandbox_helpers import FakeSandboxBackend, load_policy_text, make_env_task

KEY_FILE = os.environ.get("RSI_E2B_KEY_FILE")
AUTH_FILE = os.environ.get("RSI_E2B_REGISTRY_AUTH_FILE")
PRIVATE_IMAGE = os.environ.get("RSI_E2B_PRIVATE_IMAGE")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not KEY_FILE, reason="set RSI_E2B_KEY_FILE for live E2B"),
]
IMAGE = (
    "docker.io/library/python@"
    "sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26"
)
# E2B refused uncached pulls from Docker Hub (access denied), so the
# no_cache build uses another registry.
FRESH_BASE = "mirror.gcr.io/library/python:3.13-slim-bookworm"
# The image's own PATH (docker inspect), without E2B's build-time suffix.
IMAGE_PATH = (
    "/usr/local/bin:/usr/local/sbin:" + "/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


@dataclass
class LiveKit:
    broker: SandboxBroker
    client: object

    @property
    def envs(self):
        return self.broker.envs


@pytest.fixture
def live(tmp_path):
    run_id = "e2b-live-" + secrets.token_hex(4)
    e2b = f"api_key_file = {KEY_FILE!r}"
    if AUTH_FILE:
        e2b += f"\nregistry_auth_file = {AUTH_FILE!r}"
    text = e2b_policy_text(e2b=e2b, allowlist={})
    domains = ["docker.io", "mirror.gcr.io"]
    if PRIVATE_IMAGE:
        domains.append(pull_reference(PRIVATE_IMAGE)[0].partition("/")[0])
    text = text.replace(
        'registries = ["docker.io"]', f"registries = {json.dumps(domains)}"
    )
    # The smallest shape: one vCPU and 1 GiB per sandbox.
    text = text.replace("cpus_per_container = 4\n", "cpus_per_container = 1\n")
    text = text.replace(
        "memory_mb_per_container = 8192\n", "memory_mb_per_container = 1024\n"
    )
    grant = resolve_env_grant(
        make_env_task(
            TASK.replace(
                'network = ["public", "none"]',
                'network = ["public", "none", "allowlist"]',
                1,
            ).replace("pull = true", "pull = true\nbuild = true", 1)
        ),
        load_policy_text(tmp_path, text),
        builder_images={},
        parent_cpus=1,
        parent_memory_mb=256,
    )
    client = e2b_client(grant.environments.host.e2b)
    runtime = remote_env_runtime(
        client, spool_root=tmp_path / "spool", host=grant.environments.host
    )
    store = LeaseStore(tmp_path / "leases")
    broker = SandboxBroker(
        grant,
        FakeSandboxBackend(),
        SandboxJournal(authority(store, run_id)),
        time.monotonic,
        envs=runtime,
    )
    kit = LiveKit(broker, client)
    session = broker.open_session(
        SandboxOwner(run_id=run_id, task_id="task", phase="work"), None
    )
    broker.activate_work(time.monotonic() + 600)
    try:
        yield kit, session, run_id
    finally:
        try:
            broker.close()
        finally:
            kill_run_sandboxes(client, run_id)
            assert client.list(run_metadata(run_id)) == []


def test_a_live_env_runs_execs_survives_a_freeze_and_is_killed(live):
    kit, work, run_id = live
    handle = pull(kit, work, ref=IMAGE)
    env_id = ready(kit, work, handle, spec=single(handle, network="none"))
    sandbox_id = kit.envs._envs[env_id].lease.services[0].sandbox_id
    view = run(
        kit, work, env_id, ["sh", "-c", 'echo "$PATH|$PYTHON_VERSION|$E2B_SANDBOX_ID"']
    )
    path, version, own_id = view["stdout"].decode().strip().split("|")
    assert (path, own_id) == (IMAGE_PATH, sandbox_id) and version.startswith("3.13")
    # E2B's egress firewall is a TCP proxy: a connect() may complete, but
    # no byte reaches the internet (Docker's "none" refuses the connect).
    probe = (
        "import socket, urllib.request\n"
        "try:\n"
        "    socket.getaddrinfo('example.com', 443)\n"
        "    print('dns')\n"
        "except OSError:\n"
        "    pass\n"
        "try:\n"
        "    urllib.request.urlopen('http://1.1.1.1/', timeout=8).read(1)\n"
        "    print('http')\n"
        "except OSError:\n"
        "    pass\n"
    )
    blocked = wait_exec(
        kit,
        work,
        start_exec(kit, work, env_id, ["python3", "-c", probe], "net"),
        timeout=60,
    )
    assert (blocked["exit_code"], blocked["stdout"]) == (0, b"")

    timed = run(kit, work, env_id, ["sleep", "30"], "slow", timeout_sec=1)
    assert (timed["state"], timed["exit_code"]) == ("timed_out", 143)
    # A pause cuts envd's streams: the exec and the service are followed again.
    exec_id = start_exec(
        kit, work, env_id, ["sh", "-c", "echo one; sleep 3; echo two"], "across"
    )
    wait_exec(kit, work, exec_id, until=lambda view: view["stdout"])
    kit.broker.freeze_work()
    assert kit.client.state(sandbox_id) == "paused"
    kit.broker.resume_work()
    across = wait_exec(kit, work, exec_id, timeout=60)
    assert (across["state"], across["exit_code"]) == ("exited", 0)
    assert across["stdout"] == b"one\ntwo\n"
    stopped = kit.broker.env_stop_service(work.credential, env_id, "main", 5, "stop")
    assert stopped == {"state": "exited", "exit_code": 143}

    kit.broker.close()
    assert kit.client.state(sandbox_id) is None
    assert kit.client.list(run_metadata(run_id)) == []


# Each check prints its name when it gets an answer. Only the listed IP
# answers, whatever TLS SNI or HTTP Host is sent, and nothing resolves.
ALLOW_PROBE = """\
import socket, ssl

def ask(addr, port, host, tls):
    with socket.create_connection((addr, port), timeout=8) as conn:
        if tls:
            context = ssl._create_unverified_context()
            conn = context.wrap_socket(conn, server_hostname=host)
        conn.sendall(f"HEAD / HTTP/1.1\\r\\nHost: {host}\\r\\n\\r\\n".encode())
        return conn.recv(1)

checks = {
    "1.1.1.1": lambda: ask("1.1.1.1", 80, "1.1.1.1", False),
    "host": lambda: ask("1.0.0.1", 80, "1.1.1.1", False),
    "sni": lambda: ask("1.0.0.1", 443, "one.one.one.one", True),
    "dns": lambda: socket.getaddrinfo("example.com", 443),
}
for name, check in checks.items():
    try:
        if check():
            print(name)
    except OSError:
        pass
"""


def test_a_live_allowlist_env_reaches_only_its_entries(live):
    kit, work, _ = live
    handle = pull(kit, work, ref=IMAGE)
    spec = single(handle, network="allowlist")
    spec["allowlist"] = ["1.1.1.1"]
    env_id = ready(kit, work, handle, spec=spec)

    def reached(request_id):
        argv = ["python3", "-c", ALLOW_PROBE]
        view = wait_exec(
            kit, work, start_exec(kit, work, env_id, argv, request_id), timeout=120
        )
        assert view["exit_code"] == 0, view
        return view["stdout"].decode().split()

    assert reached("net") == ["1.1.1.1"]
    # The rules hold across a pause.
    kit.broker.freeze_work()
    kit.broker.resume_work()
    assert reached("net-again") == ["1.1.1.1"]


# Each check that gets an answer prints its name and whether it is
# example.com's page. A hostname entry is matched by TLS SNI and HTTP Host,
# and E2B connects to that host whatever address the sandbox dials, so a
# spoof still gets only example.com (its certificate verifies).
HOST_PROBE = """\
import socket, ssl

def get(addr, port, host, tls):
    with socket.create_connection((addr, port), timeout=8) as conn:
        if tls:
            context = ssl.create_default_context()
            conn = context.wrap_socket(conn, server_hostname=host)
        request = f"GET / HTTP/1.1\\r\\nHost: {host}\\r\\nConnection: close\\r\\n\\r\\n"
        conn.sendall(request.encode())
        body = b""
        while chunk := conn.recv(65536):
            body += chunk
        return b"<title>Example Domain</title>" in body

checks = {
    "https": lambda: get("example.com", 443, "example.com", True),
    "http": lambda: get("example.com", 80, "example.com", False),
    "other": lambda: get("example.org", 443, "example.org", True),
    "sni": lambda: get("9.9.9.9", 443, "example.com", True),
    "host": lambda: get("9.9.9.9", 80, "example.com", False),
    "unlisted": lambda: get("9.9.9.9", 443, "dns.quad9.net", True) or True,
}
for name, check in checks.items():
    try:
        print(f"{name}={check()}")
    except OSError:
        pass
"""


def test_a_live_hostname_allowlist_reaches_only_that_host(live):
    kit, work, _ = live
    handle = pull(kit, work, ref=IMAGE)
    spec = single(handle, network="allowlist")
    spec["allowlist"] = ["example.com:443", "example.com:80"]
    env_id = ready(kit, work, handle, spec=spec)
    argv = ["python3", "-c", HOST_PROBE]
    view = wait_exec(
        kit, work, start_exec(kit, work, env_id, argv, "host"), timeout=120
    )
    assert view["exit_code"] == 0, view
    assert view["stdout"].decode().split() == [
        "https=True",
        "http=True",
        "sni=True",
        "host=True",
    ]


@pytest.mark.skipif(
    not (AUTH_FILE and PRIVATE_IMAGE),
    reason="set RSI_E2B_REGISTRY_AUTH_FILE and RSI_E2B_PRIVATE_IMAGE",
)
def test_a_live_private_image_builds_with_its_config(live):
    kit, work, _ = live
    view = pull_view(kit, work, ref=PRIVATE_IMAGE)
    assert view["state"] == "succeeded", view
    image = view["result"]["image"]
    assert image["entrypoint"] == ["env", "FROM_ENTRYPOINT=1"], image
    assert image["workdir"] and image["user"], image
    env_id = ready(kit, work, image["handle"])
    view = run(kit, work, env_id, ["sh", "-c", "pwd; id -un"])
    assert view["stdout"].decode().split() == [
        image["workdir"].rstrip("/") or "/",
        image["user"].partition(":")[0],
    ], view
    # The service ran under the image's entrypoint (env execs its command).
    found = run(kit, work, env_id, ["sh", "-c", "cat /proc/[0-9]*/environ"], "environ")
    assert b"FROM_ENTRYPOINT=1" in found["stdout"], found


BUILD_CONTEXT = {
    "Dockerfile": f"""ARG BASE={IMAGE}
FROM $BASE
ARG GREETING=hi
ENV APP_HOME=/srv/app
WORKDIR $APP_HOME
COPY data.txt ./
COPY --chown=nobody:nogroup --chmod=0750 tool.sh /opt/bin/
COPY app/ ./app/
COPY . sub
COPY *.txt /opt/txt/
COPY --chown=65534:65534 data.txt /opt/num/
RUN echo "$GREETING" > greeting && id -un > built-by
USER nobody
WORKDIR work
RUN id -un > who
CMD ["sleep", "300"]
""".encode(),
    "data.txt": b"x\n",
    "tool.sh": b"#!/bin/sh\necho tool\n",
    "app/run.sh": b"#!/bin/sh\n",
    "notes.txt": b"n\n",
    "secret.txt": b"s\n",
    ".dockerignore": b"secret.txt\n",
}


def test_a_live_dockerfile_build_runs_its_steps_and_is_reused(live):
    kit, work, _ = live
    view = build_view(
        kit,
        work,
        BUILD_CONTEXT,
        wait=600,
        timeout_sec=600,
        build_args={"GREETING": "yo"},
    )
    assert view["state"] == "succeeded", view
    image = view["result"]["image"]
    assert (image["workdir"], image["user"]) == ("/srv/app/work", "nobody"), image
    env_id = ready(kit, work, image["handle"], spec=single(image["handle"]))
    script = (
        "pwd; id -un; cat who ../greeting ../built-by; "
        'echo "${GREETING:-none} $APP_HOME"; '
        "stat -c '%U:%G %a' /opt/bin/tool.sh ../data.txt .; "
        "ls -A ../app ../sub /opt/txt | tr '\\n' ' '; echo; "
        "stat -c '%U' ../sub ../sub/app/run.sh /opt/txt/notes.txt | tr '\\n' ' '; "
        "echo; "
        "stat -c '%u:%g' /opt/num/data.txt"
    )
    found = run(kit, work, env_id, ["sh", "-c", script])
    assert found["exit_code"] == 0, found
    # COPY keeps the context's modes, owned by root without --chown; the
    # WORKDIR made after USER is that user's (BuildKit's behavior).
    assert found["stdout"].decode().splitlines() == [
        "/srv/app/work",
        "nobody",
        "nobody",
        "yo",
        "root",
        "none /srv/app",
        "nobody:nogroup 750",
        "root:root 640",
        "nobody:nogroup 755",
        # A directory's contents are copied, .dockerignore applies to every
        # COPY, and a numeric --chown is kept.
        "../app: run.sh  ../sub: .dockerignore Dockerfile app data.txt notes.txt "
        "tool.sh  /opt/txt: data.txt notes.txt ",
        "root root root ",
        "65534:65534",
    ], found
    # Labels are not sent to E2B: a new handle on the same template.
    again = build_view(
        kit,
        work,
        BUILD_CONTEXT,
        "again",
        build_args={"GREETING": "yo"},
        labels={"a": "b"},
    )
    assert again["state"] == "succeeded", again
    assert again["result"]["image"]["handle"] != image["handle"]
    assert "building template" not in again["log"], again
    # no_cache builds a template E2B already has again.
    fresh = {"Dockerfile": f"FROM {FRESH_BASE}\nRUN true\n".encode()}
    for request_id, no_cache in (("cached", False), ("fresh", True)):
        view = build_view(
            kit, work, fresh, request_id, wait=600, timeout_sec=600, no_cache=no_cache
        )
        assert view["state"] == "succeeded", view
    assert "building template" in view["log"], view


@pytest.mark.skipif(
    not (AUTH_FILE and PRIVATE_IMAGE),
    reason="set RSI_E2B_REGISTRY_AUTH_FILE and RSI_E2B_PRIVATE_IMAGE",
)
def test_a_live_dockerfile_build_from_a_private_image(live):
    kit, work, _ = live
    files = {
        "Dockerfile": f"FROM {PRIVATE_IMAGE}\nCOPY data.txt /opt/\n".encode(),
        "data.txt": b"x\n",
    }
    view = build_view(kit, work, files, wait=600, timeout_sec=600)
    assert view["state"] == "succeeded", view
    env_id = ready(kit, work, view["result"]["image"]["handle"])
    assert run(kit, work, env_id, ["cat", "/opt/data.txt"])["stdout"] == b"x\n"
