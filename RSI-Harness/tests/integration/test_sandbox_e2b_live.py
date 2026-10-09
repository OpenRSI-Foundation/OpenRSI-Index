"""The E2B env backend against real E2B (opt-in: RSI_E2B_KEY_FILE).

Set ``RSI_E2B_KEY_FILE`` to the operator's key file (the key is read only
by the backend). One template is built from python:3.13-slim-bookworm (a
shared cache, kept), two 1-vCPU sandboxes run for about a minute, and every
sandbox of the test's run is killed and proven gone by metadata at the end.
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
from rsi_harness.runtime.sandbox_e2b import (
    e2b_client,
    e2b_env_runtime,
    kill_run_sandboxes,
    run_metadata,
)
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_e2b import (
    TASK,
    e2b_policy_text,
    pull,
    ready,
    run,
    single,
    start_exec,
    wait_exec,
)
from tests.sandbox_helpers import FakeSandboxBackend, load_policy_text, make_env_task

KEY_FILE = os.environ.get("RSI_E2B_KEY_FILE")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not KEY_FILE, reason="set RSI_E2B_KEY_FILE for live E2B"),
]
IMAGE = (
    "docker.io/library/python@"
    "sha256:2325bb286ec344af3e5898cc224b5844e2707ac6e26b1632516fd3edc84a5e26"
)
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
    text = e2b_policy_text(e2b=f"api_key_file = {KEY_FILE!r}")
    # The smallest shape: one vCPU and 1 GiB per sandbox.
    text = text.replace("cpus_per_container = 4\n", "cpus_per_container = 1\n")
    text = text.replace(
        "memory_mb_per_container = 8192\n", "memory_mb_per_container = 1024\n"
    )
    grant = resolve_env_grant(
        make_env_task(TASK),
        load_policy_text(tmp_path, text),
        builder_images={},
        parent_cpus=1,
        parent_memory_mb=256,
    )
    client = e2b_client(grant.environments.host.e2b)
    runtime = e2b_env_runtime(
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
