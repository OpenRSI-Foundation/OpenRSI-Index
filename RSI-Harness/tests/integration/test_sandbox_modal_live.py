"""The Modal env backend against real Modal (opt-in: RSI_MODAL_TOKEN_FILE).

Set ``RSI_MODAL_TOKEN_FILE`` to a TOML file with the operator's
``token_id`` and ``token_secret`` (read only by the backend), and
optionally ``RSI_MODAL_PROXY`` (the policy's ``proxy``). The sandboxes
belong to the app ``rsi-harness-live`` (made when missing, kept); a few
1-CPU sandboxes run for about a minute each, and every sandbox of a test's
run is killed and proven gone by its tags at the end. Images are Modal's
content cache: nothing to clean.
"""

from __future__ import annotations

import os
import secrets
import time
from dataclasses import dataclass

import pytest

from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_contracts import SandboxOwner
from rsi_harness.runtime.sandbox_modal import modal_client
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from rsi_harness.runtime.sandbox_remote import (
    kill_run_sandboxes,
    remote_env_runtime,
    run_in,
    run_metadata,
)
from tests.integration.test_sandbox_e2b_live import ALLOW_PROBE, IMAGE
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_e2b import (
    TASK,
    build_view,
    e2b_policy_text,
    pull,
    ready,
    run,
    single,
    start_exec,
    wait_exec,
)
from tests.sandbox_helpers import FakeSandboxBackend, load_policy_text, make_env_task

TOKEN_FILE = os.environ.get("RSI_MODAL_TOKEN_FILE")
PROXY = os.environ.get("RSI_MODAL_PROXY")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TOKEN_FILE, reason="set RSI_MODAL_TOKEN_FILE for Modal"),
]
# Each process's pid and state letter, from /proc.
STATES = (
    'for f in /proc/[0-9]*/stat; do read -r s < "$f" 2>/dev/null || continue; '
    'p=${f#/proc/}; s=${s##*) }; echo "${p%/stat} ${s%% *}"; done'
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
    run_id = "modal-live-" + secrets.token_hex(4)
    table = f'api_key_file = {TOKEN_FILE!r}\napp_name = "rsi-harness-live"'
    if PROXY:
        table += f"\nproxy = {PROXY!r}"
    text = e2b_policy_text(backend="modal", modal=table, allowlist={})
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
    client = modal_client(grant.environments.host.modal)
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
    session = broker.open_session(
        SandboxOwner(run_id=run_id, task_id="task", phase="work"), None
    )
    broker.activate_work(time.monotonic() + 600)
    try:
        yield LiveKit(broker, client), session, run_id
    finally:
        try:
            broker.close()
        finally:
            kill_run_sandboxes(client, run_id)
            assert client.list(run_metadata(run_id)) == []


def states(kit, sandbox_id):
    code, out, _ = run_in(kit.client, sandbox_id, ["sh", "-c", STATES])
    assert code == 0
    return dict(line.split() for line in out.decode().splitlines())


def test_a_live_env_runs_execs_survives_a_freeze_and_is_killed(live):
    import modal.config

    kit, work, run_id = live
    # The default sandbox path (V2): ~/.modal.toml and MODAL_* cannot change it.
    assert modal.config.config.get("sandbox_v2") is True
    handle = pull(kit, work, ref=IMAGE)
    env_id = ready(kit, work, handle, spec=single(handle, network="none"))
    sandbox_id = kit.envs._envs[env_id].lease.services[0].sandbox_id
    view = run(kit, work, env_id, ["sh", "-c", 'echo "$PYTHON_VERSION"; id -u'])
    version, uid = view["stdout"].decode().split()
    assert version.startswith("3.13") and uid == "0"
    # The command leads its group; a signal death is 128 + N.
    script = "echo $$; cut -d' ' -f5 /proc/$$/stat"
    pid, group = run(kit, work, env_id, ["sh", "-c", script], "group")["stdout"].split()
    assert pid == group
    killed = run(kit, work, env_id, ["sh", "-c", "kill -TERM $$"], "term")
    assert killed["exit_code"] == 143
    # A non-root user through setpriv, with its HOME.
    nobody = run(
        kit, work, env_id, ["sh", "-c", 'id -un; echo "$HOME"'], "user", user="nobody"
    )
    assert nobody["stdout"].decode().split() == ["nobody", "/nonexistent"], nobody
    blocked = run(
        kit,
        work,
        env_id,
        [
            "python3",
            "-c",
            "import urllib.request as u; u.urlopen('http://1.1.1.1/', timeout=8)",
        ],
        "net",
    )
    assert blocked["exit_code"] != 0

    # A freeze stops every process but the sandbox's own; it is reported
    # paused and new commands still run; the thaw continues them.
    service = kit.envs._backend.box(sandbox_id).service.process.pid
    kit.broker.freeze_work()
    frozen = states(kit, sandbox_id)
    assert frozen[str(service)] == "T", frozen
    status = kit.broker.env_status(work.credential, env_id)
    assert status["services"]["main"]["state"] == "paused", status
    kit.broker.resume_work()
    assert states(kit, sandbox_id)[str(service)] in "SR"
    assert run(kit, work, env_id, ["true"], "after")["exit_code"] == 0

    started = time.monotonic()
    kit.broker.close()
    assert time.monotonic() - started < 30
    assert kit.client.state(sandbox_id) is None
    assert kit.client.list(run_metadata(run_id)) == []


def test_a_live_allowlist_env_reaches_only_its_entries(live):
    kit, work, _ = live
    handle = pull(kit, work, ref=IMAGE)
    spec = single(handle, network="allowlist")
    spec["allowlist"] = ["1.1.1.1"]
    env_id = ready(kit, work, handle, spec=spec)
    argv = ["python3", "-c", ALLOW_PROBE]
    view = wait_exec(kit, work, start_exec(kit, work, env_id, argv, "net"), timeout=120)
    assert view["exit_code"] == 0, view
    # Only the IP answers (whether DNS resolves is Modal's to say).
    reached = [name for name in view["stdout"].decode().split() if name != "dns"]
    assert reached == ["1.1.1.1"], view


HOST_PROBE = """\
import socket, ssl

def get(host, port, tls):
    with socket.create_connection((host, port), timeout=8) as conn:
        if tls:
            conn = ssl.create_default_context().wrap_socket(conn, server_hostname=host)
        conn.sendall(f"HEAD / HTTP/1.1\\r\\nHost: {host}\\r\\n\\r\\n".encode())
        return conn.recv(1)

for name, port, tls in (
    ("example.com", 443, True),
    ("example.com", 80, False),
    ("example.org", 443, True),
):
    try:
        if get(name, port, tls):
            print(f"{name}:{port}")
    except OSError:
        pass
"""


def test_a_live_hostname_allowlist_reaches_only_that_host_on_443(live):
    kit, work, _ = live
    handle = pull(kit, work, ref=IMAGE)
    spec = single(handle, network="allowlist")
    spec["allowlist"] = ["example.com"]
    env_id = ready(kit, work, handle, spec=spec)
    argv = ["python3", "-c", HOST_PROBE]
    view = wait_exec(
        kit, work, start_exec(kit, work, env_id, argv, "host"), timeout=120
    )
    assert view["exit_code"] == 0, view
    assert view["stdout"].decode().split() == ["example.com:443"], view


BUILD_CONTEXT = {
    "Dockerfile": f"""ARG BASE={IMAGE}
FROM $BASE
ARG BASE
ARG GREETING=hi
ENV APP_HOME=/srv/app
WORKDIR $APP_HOME
COPY data.txt ./
COPY --chown=nobody --chmod=0750 tool.sh /opt/bin/
RUN echo "$GREETING $BASE" > greeting && id -un > built-by
USER 65534
RUN id -u > /tmp/who
USER nobody
CMD ["sleep", "300"]
""".encode(),
    "data.txt": b"x\n",
    "tool.sh": b"#!/bin/sh\necho tool\n",
}


def test_a_live_dockerfile_build_runs_its_lines_and_is_reused(live):
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
    assert (image["workdir"], image["user"]) == ("/srv/app", "nobody"), image
    env_id = ready(kit, work, image["handle"], spec=single(image["handle"]))
    script = (
        "pwd; id -un; cat greeting built-by /tmp/who; "
        "stat -c '%U %a' /opt/bin/tool.sh; echo $(ls -A)"
    )
    found = run(kit, work, env_id, ["sh", "-c", script])
    assert found["exit_code"] == 0, found
    assert found["stdout"].decode().splitlines() == [
        "/srv/app",
        "nobody",
        f"yo {IMAGE}",
        "root",
        "65534",
        "nobody 750",
        "built-by data.txt greeting",
    ], found
    again = build_view(
        kit, work, BUILD_CONTEXT, "again", wait=600, build_args={"GREETING": "yo"}
    )
    assert again["state"] == "succeeded", again
    assert "building template" not in again["log"], again
