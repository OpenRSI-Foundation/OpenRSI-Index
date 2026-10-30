"""The Daytona env backend against real Daytona (opt-in: RSI_DAYTONA_KEY_FILE).

Set ``RSI_DAYTONA_KEY_FILE`` to a file holding the operator's API key (read
only by the backend), and optionally ``RSI_DAYTONA_API_URL`` (a self-hosted
Daytona), ``RSI_DAYTONA_TARGET`` (default ``us``) and ``RSI_DAYTONA_PROXY``
(the policy's ``proxy``). A few 1-core sandboxes run for about a minute
each; every sandbox of a test's run is killed and proven gone by its labels
at the end.
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
from rsi_harness.runtime.sandbox_daytona import daytona_client
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from rsi_harness.runtime.sandbox_remote import (
    kill_run_sandboxes,
    remote_env_runtime,
    run_in,
    run_metadata,
)
from tests.integration.test_sandbox_e2b_live import ALLOW_PROBE, IMAGE
from tests.integration.test_sandbox_modal_live import HOST_PROBE, STATES
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_e2b import (
    TASK,
    e2b_policy_text,
    ready,
    run,
    single,
    start_exec,
    wait_exec,
)
from tests.sandbox_helpers import FakeSandboxBackend, load_policy_text, make_env_task

KEY_FILE = os.environ.get("RSI_DAYTONA_KEY_FILE")
API_URL = os.environ.get("RSI_DAYTONA_API_URL")
TARGET = os.environ.get("RSI_DAYTONA_TARGET", "us")
PROXY = os.environ.get("RSI_DAYTONA_PROXY")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not KEY_FILE, reason="set RSI_DAYTONA_KEY_FILE for Daytona"),
]


@dataclass
class LiveKit:
    broker: SandboxBroker
    client: object

    @property
    def envs(self):
        return self.broker.envs


@pytest.fixture
def live(tmp_path):
    run_id = "daytona-live-" + secrets.token_hex(4)
    table = (
        f'api_key_file = {KEY_FILE!r}\ntarget = "{TARGET}"\ntemplate_prefix = "rsilive"'
    )
    if API_URL:
        table += f"\napi_url = {API_URL!r}"
    if PROXY:
        table += f"\nproxy = {PROXY!r}"
    text = e2b_policy_text(backend="daytona", daytona=table, allowlist={})
    for old, new in (
        ("cpus_per_container = 4\n", "cpus_per_container = 1\n"),
        ("memory_mb_per_container = 8192\n", "memory_mb_per_container = 1024\n"),
        ("disk_mb_per_container = 10240\n", "disk_mb_per_container = 3072\n"),
    ):
        text = text.replace(old, new)
    grant = resolve_env_grant(
        make_env_task(
            TASK.replace(
                'network = ["public", "none"]',
                'network = ["public", "none", "allowlist"]',
                1,
            )
        ),
        load_policy_text(tmp_path, text),
        builder_images={},
        parent_cpus=1,
        parent_memory_mb=256,
    )
    client = daytona_client(grant.environments.host.daytona)
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
            kill_run_sandboxes(client, run_id, wait_sec=60)
            assert client.list(run_metadata(run_id)) == []
        finally:
            client.close()


def pull(kit, session, request_id="pull", ref=IMAGE):
    """As test_sandbox_e2b's, but waiting out Daytona's first build of the
    image."""
    job_id = kit.broker.image_pull(session.credential, ref, "missing", request_id)[
        "job_id"
    ]
    kit.envs.images.jobs[job_id].thread.join(600)
    view = kit.broker.job_wait(session.credential, job_id, 0, 0)
    assert view["state"] == "succeeded", view
    return view["result"]["image"]["handle"]


def skip_unless_open(kit, work, handle):
    """An organization whose own network policy replaces the sandbox's
    answers every outside request through its proxy (403); its allowlists
    cannot be tested."""
    env_id = ready(kit, work, handle, "open", spec=single(handle))
    probe = "import urllib.request as u; u.urlopen('http://1.1.1.1/', timeout=8)"
    view = run(kit, work, env_id, ["python3", "-c", probe], "open-probe")
    if b"403" in view.get("stderr", b""):
        pytest.skip("the organization's network policy replaces the sandbox's")


def states(kit, sandbox_id):
    code, out, _ = run_in(kit.client, sandbox_id, ["sh", "-c", STATES])
    assert code == 0
    return dict(line.split() for line in out.decode().splitlines())


def test_a_live_env_runs_execs_survives_a_freeze_and_is_killed(live):
    kit, work, run_id = live
    handle = pull(kit, work, ref=IMAGE)
    env_id = ready(kit, work, handle, spec=single(handle, network="none"))
    sandbox_id = kit.envs._envs[env_id].lease.services[0].sandbox_id
    # Its TTL and auto-stop are the timeout, in minutes.
    box = kit.client._call(kit.client._get(sandbox_id))
    assert 11 <= box.auto_stop_interval <= 12, box.auto_stop_interval
    view = run(kit, work, env_id, ["sh", "-c", 'echo "$PYTHON_VERSION"; id -u'])
    version, uid = view["stdout"].decode().split()
    assert version.startswith("3.13") and uid == "0"
    # The command leads its group; a signal death is 128 + N.
    script = "echo $$; cut -d' ' -f5 /proc/$$/stat"
    pid, group = run(kit, work, env_id, ["sh", "-c", script], "group")["stdout"].split()
    assert pid == group
    killed = run(kit, work, env_id, ["sh", "-c", "kill -TERM $$"], "term")
    assert killed["exit_code"] == 143
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

    # A freeze stops every process but the daemon's and its own session's;
    # it is reported paused and new commands still run; the thaw continues.
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
    skip_unless_open(kit, work, handle)
    spec = single(handle, network="allowlist")
    spec["allowlist"] = ["1.1.1.1"]
    env_id = ready(kit, work, handle, spec=spec)
    argv = ["python3", "-c", ALLOW_PROBE]
    view = wait_exec(kit, work, start_exec(kit, work, env_id, argv, "net"), timeout=120)
    assert view["exit_code"] == 0, view
    reached = [name for name in view["stdout"].decode().split() if name != "dns"]
    assert reached == ["1.1.1.1"], view


def test_a_live_hostname_allowlist_reaches_only_that_host(live):
    kit, work, _ = live
    handle = pull(kit, work, ref=IMAGE)
    skip_unless_open(kit, work, handle)
    spec = single(handle, network="allowlist")
    spec["allowlist"] = ["example.com"]
    env_id = ready(kit, work, handle, spec=spec)
    argv = ["python3", "-c", HOST_PROBE]
    view = wait_exec(
        kit, work, start_exec(kit, work, env_id, argv, "host"), timeout=120
    )
    assert view["exit_code"] == 0, view
    reached = view["stdout"].decode().split()
    assert "example.com:443" in reached and "example.org:443" not in reached, view
