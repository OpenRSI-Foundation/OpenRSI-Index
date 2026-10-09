"""Environment ops over the real Unix transport, client and copied CLI."""

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.integrations.sandbox_client import (
    ProtocolError,
    SandboxClient,
    _versions,
)
from tests.runtime.test_sandbox_envs import (
    BUSYBOX,
    finish,
    join_starter,
    open_work,
    single,
    tar_of,
)
from tests.runtime.test_sandbox_envs import (
    kit as kit,
)

V1_FIELDS = {
    "capabilities": set(),
    "create": {"profile", "lifetime_sec", "request_id"},
    "exec": {"child_id", "argv", "cwd", "env", "timeout_sec"},
    "upload": {"child_id", "root", "request_id", "timeout_sec"},
    "download": {"child_id", "root", "paths", "timeout_sec"},
    "status": {"child_id"},
    "destroy": {"child_id"},
}


@pytest.fixture
def served(kit):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots, SandboxServer

    work = open_work(kit)
    with tempfile.TemporaryDirectory(prefix="rsi-v2-") as root:
        path = Path(root) / "s"
        slots = SandboxRequestSlots(1, 0, waiters=4)
        server = SandboxServer(kit.broker, path, work.owner, slots=slots)
        server.start()
        try:
            yield kit, work, path
        finally:
            server.stop()


def pulled(client, kit):
    job_id = client.image_pull(BUSYBOX)
    kit.envs.images.jobs[job_id].thread.join(5)
    view = client.follow_job(job_id)
    assert view["state"] == "succeeded", view
    return view["result"]["image"]["handle"]


def test_v1_fields_are_byte_identical_and_v2_fields_are_separate():
    from rsi_harness.runtime import sandbox_server

    assert sandbox_server._FIELDS == V1_FIELDS
    assert not set(sandbox_server._V2_FIELDS) & set(V1_FIELDS)
    assert len(sandbox_server._V2_FIELDS) == 21  # tool_install is the 21st


@pytest.mark.parametrize(
    ("answer", "versions"),
    [
        ({"version": 1}, {1}),
        ({"version": 1, "versions": [1, 2], "environments": {}}, {1, 2}),
        ({"version": 2, "versions": [2], "environments": {}}, {2}),
        # Only a non-null grant or environments offers that version's ops.
        ({"version": 1, "versions": [1, 2], "grant": {}, "environments": None}, {1}),
        ({"version": 1, "versions": [1, 2], "grant": None, "environments": {}}, {2}),
        ({"version": 99, "versions": [1, 2]}, set()),
        ({"version": 1, "versions": [1, "2"]}, set()),
        ({"version": 3, "versions": [3]}, set()),
        ([], set()),
    ],
)
def test_client_accepts_versions_it_shares_with_the_server(answer, versions):
    assert _versions(answer) == versions


def test_real_capabilities_offer_exactly_the_granted_version(kit, tmp_path):
    from rsi_harness.runtime.recovery import LeaseStore
    from rsi_harness.runtime.sandbox import SandboxBroker
    from rsi_harness.runtime.sandbox_budget import SandboxJournal
    from rsi_harness.runtime.sandbox_contracts import SandboxOwner
    from tests.runtime.test_sandbox_budget import authority
    from tests.sandbox_helpers import (
        FakeClock,
        FakeSandboxBackend,
        make_sandbox_grant,
    )

    work = open_work(kit)
    assert _versions(kit.broker.capabilities(work.credential)) == {2}
    journal = SandboxJournal(authority(LeaseStore(tmp_path / "profile")))
    broker = SandboxBroker(
        make_sandbox_grant(), FakeSandboxBackend(), journal, FakeClock()
    )
    session = broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), 1000
    )
    answer = broker.capabilities(session.credential)
    assert answer["versions"] == [1, 2]  # every version the broker speaks
    client = SandboxClient("/nonexistent", "token")
    client._capabilities = answer
    # Refused by the handshake, before any request reaches a server.
    with pytest.raises(ProtocolError, match="version 2") as caught:
        client.env_list()
    assert caught.value.code == "unsupported"


def test_client_refuses_operations_of_a_version_the_server_lacks():
    client = SandboxClient("/nonexistent", "token")
    client._capabilities = {"version": 1}
    with pytest.raises(ProtocolError, match="version 2") as caught:
        client.env_list()
    assert caught.value.code == "unsupported"
    client._capabilities = {"version": 2, "versions": [2]}
    with pytest.raises(ProtocolError, match="version 1"):
        client.create("offline", 10)


def test_policy_sizes_every_request_gate(tmp_path):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots
    from tests.sandbox_helpers import make_env_grant, make_env_task

    host = make_env_grant(tmp_path, make_env_task()).environments.host.model_copy(
        update={"request_slots_active": 3, "request_slots_queued": 5, "waiters": 7}
    )
    slots = SandboxRequestSlots.from_policy(host)
    assert slots._active._initial_value == 3
    assert slots._places._initial_value == 3 + 5
    assert slots._waiters._initial_value == 7
    assert slots._connections._initial_value == 3 + 5 + 7
    default = SandboxRequestSlots.from_policy(None)
    assert (default._active._initial_value, default.long_polls) == (4, False)


def test_long_polls_beyond_the_waiter_cap_are_busy(kit):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots, SandboxServer

    work = open_work(kit)
    with tempfile.TemporaryDirectory(prefix="rsi-v2-") as root:
        path = Path(root) / "s"
        server = SandboxServer(
            kit.broker, path, work.owner, slots=SandboxRequestSlots(1, 1, waiters=1)
        )
        server.start()
        try:
            client = SandboxClient(path, work.credential)
            env_id = client.env_create(single(pulled(client, kit)))["env_id"]
            client.env_start(env_id, 30)
            join_starter(kit, env_id)
            exec_id = client.exec_start(env_id, "main", ["sh", "-c", "work"])
            docker = kit.api.last()
            docker.pid = 500
            kit.host.add(docker.pid, start=5500)
            kit.io()
            waiter = threading.Thread(
                target=lambda: client.exec_wait(exec_id, wait_sec=3)
            )
            waiter.start()
            time.sleep(0.3)
            with pytest.raises(ProtocolError) as caught:
                client.exec_wait(exec_id, wait_sec=3)
            assert (caught.value.code, caught.value.field) == ("busy", "requests")
            waiter.join(10)
            assert not waiter.is_alive()
            finish(kit, docker, 0)
        finally:
            server.stop()


def test_long_polls_due_now_answer_while_every_waiter_place_is_taken(kit):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots, SandboxServer

    work = open_work(kit)
    with tempfile.TemporaryDirectory(prefix="rsi-v2-") as root:
        path = Path(root) / "s"
        slots = SandboxRequestSlots(1, 1, waiters=1)
        enter_waiter = slots.enter_waiter
        waiting = threading.Event()

        def entered():
            taken = enter_waiter()
            if taken:
                waiting.set()
            return taken

        slots.enter_waiter = entered
        server = SandboxServer(kit.broker, path, work.owner, slots=slots)
        server.start()
        try:
            client = SandboxClient(path, work.credential)
            env_id = client.env_create(single(pulled(client, kit)))["env_id"]
            client.env_start(env_id, 30)
            join_starter(kit, env_id)
            exec_id = client.exec_start(env_id, "main", ["sh", "-c", "work"])
            docker = kit.api.last()
            docker.pid = 600
            kit.host.add(docker.pid, start=5600)
            kit.io()
            waiter = threading.Thread(
                target=lambda: client.exec_wait(exec_id, wait_sec=3)
            )
            waiter.start()
            assert waiting.wait(10)
            assert not enter_waiter()  # the only waiter place is held
            began = time.monotonic()
            # Nothing to wait for: answered at once, with no waiter place.
            assert client.exec_wait(exec_id, wait_sec=0)["state"] == "running"
            assert client.env_status(env_id, wait_sec=10)["state"] == "ready"
            assert time.monotonic() - began < 1.0
            # A real wait still needs a waiter place.
            with pytest.raises(ProtocolError) as caught:
                client.exec_wait(exec_id, wait_sec=3)
            assert (caught.value.code, caught.value.field) == ("busy", "requests")
            waiter.join(10)
            assert not waiter.is_alive()
            finish(kit, docker, 0)
        finally:
            server.stop()


def test_a_long_poll_needs_a_request_place_to_read_its_body(kit):
    from rsi_harness.runtime.sandbox_server import SandboxRequestSlots, SandboxServer

    work = open_work(kit)
    with tempfile.TemporaryDirectory(prefix="rsi-v2-") as root:
        path = Path(root) / "s"
        slots = SandboxRequestSlots(1, 1, waiters=1)
        server = SandboxServer(kit.broker, path, work.owner, slots=slots)
        server.start()
        try:
            client = SandboxClient(path, work.credential)
            env_id = client.env_create(single(pulled(client, kit)))["env_id"]
            # Every request place is taken; the waiter place is free.
            assert slots.enter() and slots.enter()
            try:
                for wait_sec in (0, 3):
                    with pytest.raises(ProtocolError) as caught:
                        client.env_status(env_id, wait_sec=wait_sec)
                    assert (caught.value.code, caught.value.field) == (
                        "busy",
                        "requests",
                    )
                    assert "request queue is full" in str(caught.value)
            finally:
                slots.leave()
                slots.leave()
            # Nothing was leaked: both gates admit again.
            assert client.env_status(env_id, wait_sec=0)["env_id"] == env_id
            assert slots.enter_waiter()
            slots.leave_waiter()
        finally:
            server.stop()


def test_real_socket_env_lifecycle_with_framed_stages(served):
    kit, work, path = served
    client = SandboxClient(path, work.credential)
    assert client.capabilities()["versions"] == [1, 2]
    handle = pulled(client, kit)
    env_id = client.env_create(single(handle))["env_id"]
    client.env_start(env_id, 30)
    join_starter(kit, env_id)
    assert client.wait_env(env_id, 5)["state"] == "ready"

    data = tar_of({"seed.txt": b"s" * 300_000})
    staged = client.upload_stage(data)
    assert staged["bytes"] == len(data)
    assert client.copy_in(env_id, "main", "/app", staged["stage_id"]) == {
        "entries": 1,
        "bytes": 300_000,
    }
    out = client.copy_out(env_id, "main", "/app/out.txt", 1 << 20)
    target = io.BytesIO()
    assert client.download_stage(out["stage_id"], target) == len(target.getvalue())
    with tarfile.open(fileobj=io.BytesIO(target.getvalue())) as archive:
        assert archive.extractfile("out.txt").read() == b"result"
    assert client.path_stat(env_id, "main", "/app")["exists"] is True
    assert [item["handle"] for item in client.image_list()["images"]] == [handle]
    assert client.env_list()["envs"][0]["env_id"] == env_id
    assert client.env_destroy(env_id) == {"state": "removed"}
    assert client.image_release(handle) == {"ok": True}


def test_long_polls_wait_outside_the_only_active_slot(served):
    kit, work, path = served
    client = SandboxClient(path, work.credential)
    env_id = client.env_create(single(pulled(client, kit)))["env_id"]
    client.env_start(env_id, 30)
    join_starter(kit, env_id)
    exec_id = client.exec_start(env_id, "main", ["sh", "-c", "work"])
    docker = kit.api.last()
    docker.pid = 300
    kit.host.add(docker.pid, start=5300)
    kit.io()
    results = {}

    def long_poll():
        began = time.monotonic()
        results["view"] = client.exec_wait(exec_id, wait_sec=10)
        results["elapsed"] = time.monotonic() - began

    waiter = threading.Thread(target=long_poll)
    waiter.start()
    time.sleep(0.3)
    # One active slot, yet a status answers while the long-poll waits.
    began = time.monotonic()
    assert client.env_status(env_id)["state"] == "ready"
    assert time.monotonic() - began < 1.0
    assert waiter.is_alive()
    docker.peer.sendall(b"\x01\x00\x00\x00\x00\x00\x00\x03out")
    kit.io()
    waiter.join(5)
    assert results["elapsed"] < 5
    assert base64.b64decode(results["view"]["stdout_b64"]) == b"out"
    finish(kit, docker, 0)
    final = client.follow_exec(exec_id)
    assert (final["state"], final["exit_code"]) == ("exited", 0)


def test_environment_operations_on_a_profile_grant_never_read_the_body(tmp_path):
    from fastapi.testclient import TestClient

    from rsi_harness.runtime.recovery import LeaseStore
    from rsi_harness.runtime.sandbox import SandboxBroker
    from rsi_harness.runtime.sandbox_budget import SandboxJournal
    from rsi_harness.runtime.sandbox_contracts import SandboxOwner
    from rsi_harness.runtime.sandbox_server import create_sandbox_app
    from tests.runtime.test_sandbox_budget import authority
    from tests.sandbox_helpers import (
        FakeClock,
        FakeSandboxBackend,
        make_sandbox_grant,
    )

    journal = SandboxJournal(authority(LeaseStore(tmp_path)))
    broker = SandboxBroker(
        make_sandbox_grant(), FakeSandboxBackend(), journal, FakeClock()
    )
    session = broker.open_session(
        SandboxOwner(run_id="run-1", task_id="task", phase="work"), 1000
    )
    with TestClient(create_sandbox_app(broker, session.owner)) as client:
        client.headers["Authorization"] = "Bearer " + session.credential
        response = client.post("/v1/stage_put", content=b"\0" * (32 * 1024**2))
        assert response.status_code == 401
        assert response.json()["error"]["field"] == "operation"
        assert client.post("/v1/env_list", json={}).status_code == 401
        capabilities = client.post("/v1/capabilities", json={}).json()
        assert capabilities["environments"] is None


def test_v1_operations_on_an_environment_grant_are_refused_unread(kit):
    """Deviation from spec B3 (documented): on an env grant the v1 surface is
    not granted at all, so a v1 exec of an env handle is refused at the gate
    with permission before its body is read; on a profile grant the broker
    answers invalid (tested with the broker)."""
    from fastapi.testclient import TestClient

    from rsi_harness.runtime.sandbox_server import create_sandbox_app

    work = open_work(kit)
    with TestClient(create_sandbox_app(kit.broker, work.owner)) as client:
        client.headers["Authorization"] = "Bearer " + work.credential
        response = client.post(
            "/v1/exec",
            json={
                "child_id": "e" + "0" * 32,
                "argv": ["true"],
                "cwd": "/workspace",
                "env": {},
                "timeout_sec": 5,
            },
        )
        assert response.status_code == 401
        assert response.json()["error"]["field"] == "operation"


def run_cli(path, work, *args, stdin=None, timeout=30):
    from rsi_harness.integrations import sandbox_client as wire

    script = Path(path).parent / "rsi-sandbox"
    if not script.exists():
        script.write_bytes(Path(wire.__file__).read_bytes())
    environment = {
        "PATH": os.environ["PATH"],
        "RSI_SANDBOX_SOCKET": str(path),
        "RSI_SANDBOX_TOKEN": work.credential,
    }
    return subprocess.Popen(
        [sys.executable, "-I", str(script), *args],
        env=environment,
        stdin=subprocess.PIPE if stdin is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def completed(process, stdin=None, timeout=30):
    out, err = process.communicate(stdin, timeout=timeout)
    return process.returncode, out, err


def test_standalone_cli_drives_envs_images_copies_and_execs(served, tmp_path):
    kit, work, path = served
    code, out, err = completed(run_cli(path, work, "pull", BUSYBOX))
    assert code == 0, err
    handle = json.loads(out)["result"]["image"]["handle"]
    assert b"Pull complete" in err

    spec = json.dumps(single(handle)).encode()
    process = run_cli(path, work, "up", "-", "--wait-timeout", "30", stdin=spec)
    code, out, err = completed(process, spec)
    assert code == 0, err
    status = json.loads(out)
    env_id = status["env_id"]
    assert status["state"] == "ready"

    code, out, _ = completed(run_cli(path, work, "ps"))
    assert [item["env_id"] for item in json.loads(out)["envs"]] == [env_id]
    code, out, _ = completed(run_cli(path, work, "ps", env_id))
    assert json.loads(out)["state"] == "ready"

    local = tmp_path / "local"
    (local / "sub").mkdir(parents=True)
    (local / "sub" / "a.txt").write_text("alpha")
    code, out, err = completed(
        run_cli(path, work, "cp", str(local), f"{env_id}:main:/app")
    )
    assert code == 0, err
    assert json.loads(out)["entries"] == 3
    sent = kit.transfers[0].calls[-1]
    with tarfile.open(fileobj=io.BytesIO(sent[3])) as archive:
        assert sorted(archive.getnames()) == ["local", "local/sub", "local/sub/a.txt"]
    fetched = tmp_path / "fetched"
    code, out, err = completed(
        run_cli(path, work, "cp", f"{env_id}:main:/app/out.txt", str(fetched))
    )
    assert code == 0, err
    assert (fetched / "out.txt").read_bytes() == b"result"

    process = run_cli(path, work, "exec", env_id, "--", "sh", "-c", "work")
    deadline = time.monotonic() + 10
    while not kit.api.created:
        assert time.monotonic() < deadline
        time.sleep(0.05)
    docker = kit.api.last()
    docker.pid = 400
    kit.host.add(docker.pid, start=5400)
    kit.io()
    docker.peer.sendall(b"\x01\x00\x00\x00\x00\x00\x00\x04out\n")
    docker.peer.sendall(b"\x02\x00\x00\x00\x00\x00\x00\x04err\n")
    finish(kit, docker, 3)
    code, out, err = completed(process)
    assert (code, out, err) == (3, b"out\n", b"err\n")

    code, out, _ = completed(run_cli(path, work, "stop-service", env_id, "main"))
    assert json.loads(out) == {"state": "exited", "exit_code": 0}
    code, out, _ = completed(run_cli(path, work, "images"))
    assert json.loads(out)["images"][0]["in_use"] is True
    code, out, err = completed(run_cli(path, work, "image-rm", handle))
    assert code == 2 and b"busy" in err
    code, out, _ = completed(run_cli(path, work, "rm", env_id))
    assert json.loads(out) == {"state": "removed"}
    code, out, _ = completed(run_cli(path, work, "image-rm", handle))
    assert json.loads(out) == {"ok": True}


class StubClient:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def execute(self, child_id, argv, cwd, env, timeout):
        self.calls.append(("execute", child_id, argv, cwd, env, timeout))
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    def exec_start(self, env_id, service, argv, **fields):
        self.calls.append(("exec_start", env_id, service, argv))
        return "x" + "0" * 32

    def follow_exec(self, exec_id, on_output=None, wait_sec=25):
        return {"state": "exited", "exit_code": 0}


@pytest.mark.parametrize(
    ("child", "operation"),
    [
        ("e" + "0" * 31, "execute"),  # a v1 child ID that starts with "e"
        ("a" + "0" * 31, "execute"),
        ("e" + "0" * 32, "exec_start"),  # an env handle
    ],
)
def test_cli_exec_dispatches_on_the_exact_handle_grammar(monkeypatch, child, operation):
    from rsi_harness.integrations import sandbox_client as wire

    StubClient.calls = []
    monkeypatch.setattr(wire, "SandboxClient", StubClient)
    assert wire.main(["exec", child, "--", "true"]) == 0
    assert [call[0] for call in StubClient.calls] == [operation]
    if operation == "execute":
        assert StubClient.calls[0][1:] == (child, ["true"], "/workspace", {}, 30)


def test_cli_v1_exec_refuses_env_only_options(monkeypatch, capsys):
    from rsi_harness.integrations import sandbox_client as wire

    StubClient.calls = []
    monkeypatch.setattr(wire, "SandboxClient", StubClient)
    child = "e" + "0" * 31
    assert wire.main(["exec", child, "--service", "db", "--", "true"]) == 2
    assert "--service" in capsys.readouterr().err
    assert StubClient.calls == []


def tar_with(*members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for info, data in members:
            archive.addfile(info, None if data is None else io.BytesIO(data))
    buffer.seek(0)
    return buffer


def member(name, data=None, *, link=None):
    info = tarfile.TarInfo(name)
    if link is not None:
        info.type, info.linkname = tarfile.SYMTYPE, link
    elif data is None:
        info.type = tarfile.DIRTYPE
    else:
        info.size = len(data)
    return info, data


@pytest.mark.filterwarnings("ignore:Python 3.14 will:DeprecationWarning")
@pytest.mark.parametrize("filtered", [True, False])
def test_cli_copy_out_unpacks_safely_without_the_data_filter(
    monkeypatch, tmp_path, filtered
):
    from rsi_harness.integrations import sandbox_client as wire

    if not filtered:
        monkeypatch.delattr(tarfile, "data_filter")
    good = tar_with(
        member("out"), member("out/a.txt", b"alpha"), member("out/l", link="a.txt")
    )
    with tarfile.open(fileobj=good) as archive:
        wire._extract(archive, tmp_path / "good")
    assert (tmp_path / "good" / "out" / "l").read_bytes() == b"alpha"
    for bad in (
        tar_with(member("../escape.txt", b"x")),
        tar_with(member("out/l", link="../../etc")),
        tar_with(member("out/l", link="/etc")),
    ):
        with (
            tarfile.open(fileobj=bad) as archive,
            pytest.raises((ProtocolError, tarfile.TarError)),
        ):
            wire._extract(archive, tmp_path / "bad")
    assert not (tmp_path / "escape.txt").exists()


def test_cli_build_args_follow_docker_build():
    """``rsi-sandbox build --build-arg``: ``NAME=VALUE``, or a bare ``NAME``
    read from the environment and omitted when unset, as docker build does."""
    from rsi_harness.integrations.sandbox_compose import build_args

    environ = {"FROM_ENV": "yes", "EMPTY": ""}
    assert build_args(
        ["A=1", "B=x=y", "C=", "FROM_ENV", "EMPTY", "UNSET"], environ
    ) == (("A", "1"), ("B", "x=y"), ("C", ""), ("FROM_ENV", "yes"), ("EMPTY", ""))
