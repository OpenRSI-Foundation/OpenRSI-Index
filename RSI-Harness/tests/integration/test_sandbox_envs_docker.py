"""Real brokered envs through the phase sockets as a non-root user.

The broker, lifecycle, server and client are the production ones; only the
iptables half is faked. Real firewall blocking and cgroup.kill of a paused
env need root and belong to the operator check (spec 8), which runs this
file as root in root mode (RSI_SANDBOX_ROOT_MODE=1: the real firewall, and
the backend's paused killer is then cgroup.kill).
"""

import io
import os
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException

from rsi_harness.integrations.sandbox_client import SandboxClient
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_envs import docker_env_runtime
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from tests.integration.sandbox_support import assert_no_rules, sandbox_firewall
from tests.integration.test_sandbox_env_docker import BUSYBOX, remove_labelled
from tests.runtime.test_sandbox_budget import authority
from tests.sandbox_helpers import FakeSandboxBackend, make_env_grant, make_env_task

pytestmark = pytest.mark.integration

MIB = 1024**2
TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
[metadata.rsi_harness.sandbox.environments.judge]
network = ["public", "none"]
pull = true
"""


class Run:
    def __init__(self, client, run_id, lifecycle, broker, firewall):
        self.client = client
        self.run_id = run_id
        self.lifecycle = lifecycle
        self.broker = broker
        self.firewall = firewall

    def endpoint(self, phase, round_id=None):
        if phase == "work":
            endpoint = self.lifecycle.prepare_work()
            self.lifecycle.activate_work(time.monotonic() + 900)
        else:
            endpoint = self.lifecycle.prepare_judge(round_id)
            self.lifecycle.activate_judge(time.monotonic() + 900)
        return SandboxClient(
            endpoint.directory / "s", endpoint.environment["RSI_SANDBOX_TOKEN"]
        )

    def labelled(self, **labels):
        filters = {
            "label": [f"rsi-harness.run-id={self.run_id}"]
            + [f"rsi-harness.{key}={value}" for key, value in labels.items()]
        }
        return (
            self.client.containers.list(all=True, filters=filters),
            self.client.volumes.list(filters=filters),
            self.client.networks.list(filters=filters),
        )


@pytest.fixture
def run(tmp_path):
    try:
        client = docker.from_env(timeout=60)
        client.ping()
        client.images.get(BUSYBOX)
        docker_root = client.info()["DockerRootDir"]
    except (DockerException, OSError, KeyError) as error:
        message = f"Docker/{BUSYBOX} capability unavailable: {error}"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    run_id = f"m5-envs-{uuid.uuid4().hex[:12]}"
    grant = make_env_grant(tmp_path, make_env_task(TASK))
    firewall = sandbox_firewall(client)
    spool = tmp_path / "sb" / "spool"
    runtime = docker_env_runtime(
        client,
        firewall,
        run_id=run_id,
        spool_root=spool,
        docker_root=docker_root,
        host=grant.environments.host,
    )
    journal = SandboxJournal(authority(LeaseStore(tmp_path / "leases"), run_id))
    broker = SandboxBroker(
        grant, FakeSandboxBackend(), journal, time.monotonic, envs=runtime
    )
    # A short root: the endpoint socket path must fit 107 bytes.
    root = Path(tempfile.mkdtemp(prefix="rsi-m5-"))
    lifecycle = SandboxLifecycle()
    lifecycle.configure(broker, root / "sb", run_id, "task")
    try:
        yield Run(client, run_id, lifecycle, broker, firewall)
    finally:
        try:
            lifecycle.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)
            filters = {"label": f"rsi-harness.run-id={run_id}"}
            errors = remove_labelled(client, filters)
            leftovers = (
                client.containers.list(all=True, filters=filters),
                client.volumes.list(filters=filters),
                client.networks.list(filters=filters),
            )
            client.close()
            assert (errors, leftovers) == ([], ([], [], []))
            assert not broker.recovery_required
            assert_no_rules(firewall, run_id)
            assert journal.envs() == ()
            assert not (spool / "x").exists()


def pulled(client):
    job_id = client.image_pull(BUSYBOX, "missing")
    view = client.follow_job(job_id)
    assert view["state"] == "succeeded", view
    return view["result"]["image"]["handle"]


def service(handle, command="sleep 600", **values):
    return {
        "image": handle,
        "command": ["sh", "-c", command],
        "cpus": 0.5,
        "memory_mb": 128,
        "pids": 256,
        **values,
    }


def env_spec(handle, network="none", **services):
    return {
        "version": 1,
        "network": network,
        "lifetime_sec": 600,
        "disk_mb": 256,
        "services": services or {"main": service(handle)},
    }


def up(client, spec):
    env_id = client.env_create(spec)["env_id"]
    client.env_start(env_id, 60)
    status = client.wait_env(env_id, 90)
    assert status["state"] == "ready", status
    return env_id


def run_exec(client, env_id, argv, **fields):
    exec_id = client.exec_start(env_id, "main", argv, **fields)
    out = []
    final = client.follow_exec(exec_id, lambda stream, data: out.append((stream, data)))
    stdout = b"".join(data for stream, data in out if stream == "stdout")
    return final, stdout


def tar_of(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def test_real_env_lifecycle_exec_and_copies_over_the_work_socket(run):
    client = run.endpoint("work")
    handle = pulled(client)
    env_id = up(client, env_spec(handle))

    final, stdout = run_exec(client, env_id, ["sh", "-c", "echo hello; exit 3"])
    assert (final["state"], final["exit_code"], stdout) == ("exited", 3, b"hello\n")

    staged = client.upload_stage(tar_of({"seed.txt": b"seeded\n"}))
    assert client.copy_in(env_id, "main", "/app/in", staged["stage_id"]) == {
        "entries": 1,
        "bytes": 7,
    }
    _, stdout = run_exec(client, env_id, ["cat", "/app/in/seed.txt"])
    assert stdout == b"seeded\n"
    assert client.path_stat(env_id, "main", "/app/in")["kind"] == "dir"
    out = client.copy_out(env_id, "main", "/app/in", 1 << 20)
    body = io.BytesIO()
    client.download_stage(out["stage_id"], body)
    with tarfile.open(fileobj=io.BytesIO(body.getvalue())) as archive:
        assert archive.extractfile("in/seed.txt").read() == b"seeded\n"

    final = client.env_stop_service(env_id, "main", 1)
    assert final["state"] == "exited"
    status = client.env_status(env_id)
    assert status["services"]["main"]["state"] == "exited"
    assert client.env_destroy(env_id) == {"state": "removed"}
    assert run.labelled(**{"sandbox-env": env_id}) == ([], [], [])
    assert client.image_release(handle) == {"ok": True}


def test_status_answers_within_a_second_while_long_polls_and_copies_run(run):
    client = run.endpoint("work")
    env_id = up(client, env_spec(pulled(client)))
    execs = [
        client.exec_start(env_id, "main", ["sleep", "30"], timeout_sec=60)
        for _ in range(8)
    ]
    payload = tar_of({"blob.bin": os.urandom(4 * MIB)})
    stop = threading.Event()
    copies = []

    def long_poll(exec_id):
        return client.exec_wait(exec_id, wait_sec=25)["state"]

    def copy_loop(index):
        while not stop.is_set():
            staged = client.upload_stage(payload)
            copies.append(
                client.copy_in(env_id, "main", f"/copies/{index}", staged["stage_id"])
            )

    latencies = []
    with ThreadPoolExecutor(16) as pool:
        waits = [pool.submit(long_poll, exec_id) for exec_id in execs]
        loops = [pool.submit(copy_loop, index) for index in range(8)]
        time.sleep(1.0)
        try:
            for _ in range(6):
                began = time.monotonic()
                assert client.env_status(env_id)["state"] == "ready"
                latencies.append(time.monotonic() - began)
                time.sleep(0.3)
        finally:
            stop.set()
            for loop in loops:
                loop.result(60)
        # Every long-poll is still waiting; none held a worker slot.
        assert not any(wait.done() for wait in waits)
        for exec_id in execs:
            client.exec_kill(exec_id, "KILL")
        assert {wait.result(30) for wait in waits} <= {"killed", "running"}
    assert max(latencies) < 1.0, latencies
    assert len(copies) >= 8
    client.env_destroy(env_id)


def test_close_judge_removes_the_round_before_work_resumes(run):
    work = run.endpoint("work")
    work_env = up(work, env_spec(pulled(work)))
    held = work.exec_start(work_env, "main", ["sleep", "300"], timeout_sec=120)
    run.lifecycle.freeze_work()
    container = run.labelled(**{"sandbox-env": work_env})[0][0]
    container.reload()
    assert container.attrs["State"]["Paused"]

    judge = run.endpoint("judge", "round-1")
    handle = pulled(judge)
    judge_env = up(
        judge,
        env_spec(
            handle,
            "public",
            main=service(handle, depends_on={"db": {"condition": "started"}}),
            db=service(handle, aliases=["kvstore"]),
        ),
    )
    judge.exec_start(judge_env, "main", ["sleep", "600"])
    assert run.labelled(**{"round-id": "round-1"}) != ([], [], [])
    assert len(run.firewall.installed) == 1

    began = time.monotonic()
    run.lifecycle.close_judge()
    assert time.monotonic() - began < 30
    assert run.labelled(**{"round-id": "round-1"}) == ([], [], [])
    assert run.firewall.installed == {}
    assert not run.broker.recovery_required

    run.lifecycle.resume_work()
    run.lifecycle.reopen_work()
    container.reload()
    assert container.attrs["State"]["Running"]
    assert not container.attrs["State"]["Paused"]
    assert work.exec_wait(held, wait_sec=0)["state"] == "running"
    work.exec_kill(held, "KILL")
    assert work.follow_exec(held)["state"] == "killed"
    work.env_destroy(work_env)


def test_run_close_removes_a_paused_work_env_without_recovery(run):
    work = run.endpoint("work")
    env_id = up(work, env_spec(pulled(work)))
    run.lifecycle.freeze_work()
    run.lifecycle.close()
    assert not run.broker.recovery_required
    assert run.labelled(**{"sandbox-env": env_id}) == ([], [], [])


def test_a_cached_image_pulls_by_digest_after_the_budget_is_spent(run):
    """A pre-pulled image referenced by digest costs no pull budget: with the
    run's max_pull_mb spent it still binds and runs; a digest not on the
    host is refused before the daemon contacts any registry."""
    from rsi_harness.integrations.sandbox_client import ProtocolError

    [digest] = [
        entry
        for entry in run.client.images.get(BUSYBOX).attrs["RepoDigests"]
        if entry.startswith("busybox@sha256:")
    ]
    work = run.endpoint("work")
    run.broker.envs._run_usage["pull_bytes"] = 40960 * MIB  # run_limits.max_pull_mb
    logs = []
    view = work.follow_job(work.image_pull(digest, "missing"), logs.append)
    assert view["state"] == "succeeded", view
    handle = view["result"]["image"]["handle"]
    assert "pulling" not in "".join(logs)
    assert work.capabilities()["environments"]["usage"]["image_bytes"] == 0
    env_id = up(work, env_spec(handle))
    _, stdout = run_exec(work, env_id, ["echo", "cached"])
    assert stdout == b"cached\n"
    assert work.env_destroy(env_id) == {"state": "removed"}

    absent = "busybox@sha256:" + "0" * 64
    view = work.follow_job(work.image_pull(absent, "missing"), logs.append)
    assert (view["state"], view["error"]["kind"]) == ("failed", "quota")
    assert "pulling" not in "".join(logs)
    with pytest.raises(ProtocolError) as caught:
        work.image_pull(digest, "always")
    assert (caught.value.code, caught.value.field) == ("quota", "max_pull_mb")
    assert work.image_release(handle) == {"ok": True}
    assert BUSYBOX in {
        tag for image in run.client.images.list() for tag in image.tags
    }  # the cached image stays on the host
