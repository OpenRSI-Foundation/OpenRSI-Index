"""The remote env backends' shared seam (sandbox_remote) and contracts,
apart from any one backend, and the emulation of what a backend lacks;
tests/runtime/test_sandbox_e2b.py runs the whole layer over the fake, with
and without that emulation."""

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime import sandbox_daytona, sandbox_e2b, sandbox_modal
from rsi_harness.runtime.sandbox_contracts import (
    EnvDaytonaHost,
    EnvE2BHost,
    EnvHostPolicy,
    EnvModalHost,
    SandboxOwner,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    SandboxEnvLease,
    SandboxEnvServiceLease,
    env_container_name,
)
from rsi_harness.runtime.sandbox_remote import (
    FREEZE_SCRIPT,
    THAW_SCRIPT,
    RemoteTemplates,
    SandboxEnvRemoteBackend,
    _Box,
    _Service,
    config_env,
    group_argv,
    kill_run_sandboxes,
    remote_backend,
    remote_env_runtime,
    start_group,
)
from tests.remote_fake import RemoteFake

RUN = {"rsi_run_id": "run-1"}


class Lagging:
    """A client whose listing keeps showing killed sandboxes; ``gone_after``
    state calls after its kill, a sandbox's state is gone (None)."""

    name = "lagging"

    def __init__(self, gone_after: dict[str, int]) -> None:
        self.gone_after = gone_after
        self.killed: list[str] = []
        self.asked: dict[str, int] = {}

    def list(self, metadata):
        return [(sandbox_id, dict(RUN)) for sandbox_id in self.gone_after]

    def kill(self, sandbox_id):
        self.killed.append(sandbox_id)

    def state(self, sandbox_id):
        if sandbox_id not in self.killed:
            return "running"
        self.asked[sandbox_id] = self.asked.get(sandbox_id, 0) + 1
        gone = self.asked[sandbox_id] > self.gone_after[sandbox_id]
        return None if gone else "running"


def test_run_cleanup_waits_out_a_listing_that_lags_the_kill():
    client = Lagging({"sb-a": 0, "sb-b": 3})
    assert kill_run_sandboxes(client, "run-1", poll=0) == ("sb-a", "sb-b")
    assert client.killed == ["sb-a", "sb-b"]


def test_run_cleanup_names_only_the_sandboxes_not_proven_gone():
    client = Lagging({"sb-a": 0, "sb-b": 10**9})
    with pytest.raises(InfrastructureError) as left:
        kill_run_sandboxes(client, "run-1", wait_sec=0.05, poll=0.01)
    assert str(left.value) == "lagging sandboxes of run run-1 remain after kill: sb-b"


def test_a_create_in_flight_is_also_killed_by_its_name(tmp_path):
    """A sandbox a listing does not show yet: the client's lookup finds it."""
    fake = RemoteFake(tmp_path / "e2b")
    fake.templates["rsi-x"] = {}
    unlisted = fake.create("rsi-x", timeout=60, metadata={}, allow_internet=False)
    fake.lookup = lambda env_id: unlisted if env_id == "e1" else None
    backend = SandboxEnvRemoteBackend(fake, RemoteTemplates(fake, "rsi"), poll=0)
    try:
        backend._kill_env(SimpleNamespace(env_id="e1"))
        assert fake.sandboxes[unlisted].state is None
    finally:
        fake.shutdown()


def test_each_remote_backend_names_its_module_and_reads_its_credentials(
    monkeypatch,
):
    e2b = remote_backend("e2b")
    assert e2b is sandbox_e2b.BACKEND
    monkeypatch.setenv("RSI_TEST_E2B_KEY", "k3y")
    assert e2b.credentials(EnvE2BHost(api_key_env="RSI_TEST_E2B_KEY")) == ("k3y",)
    host = dict(pool_disk_mb=1024, disk_floor_mb=512, disk_hard_floor_mb=256)
    assert EnvHostPolicy(**host).remote is None
    settings = EnvE2BHost(api_key_env="K")
    assert EnvHostPolicy(**host, backend="e2b", e2b=settings).remote == settings
    modal = remote_backend("modal")
    assert modal is sandbox_modal.BACKEND
    monkeypatch.setenv("RSI_TEST_MODAL_ID", "ak-1")
    monkeypatch.setenv("RSI_TEST_MODAL_SECRET", "as-2")
    tokens = EnvModalHost(
        api_key_id_env="RSI_TEST_MODAL_ID", api_key_env="RSI_TEST_MODAL_SECRET"
    )
    assert modal.credentials(tokens) == ("ak-1", "as-2")
    assert EnvHostPolicy(**host, backend="modal", modal=tokens).remote == tokens
    with pytest.raises(ValueError, match="go together"):
        EnvHostPolicy(**host, backend="modal", e2b=settings)
    daytona = remote_backend("daytona")
    assert daytona is sandbox_daytona.BACKEND
    monkeypatch.setenv("RSI_TEST_DAYTONA_KEY", "dtn-1")
    table = EnvDaytonaHost(api_key_env="RSI_TEST_DAYTONA_KEY", target="us")
    assert daytona.credentials(table) == ("dtn-1",)
    assert EnvHostPolicy(**host, backend="daytona", daytona=table).remote == table
    with pytest.raises(ValueError, match="target"):
        EnvDaytonaHost(api_key_env="RSI_TEST_DAYTONA_KEY")


def test_a_client_without_a_build_form_offers_no_builds(tmp_path):
    host = EnvHostPolicy(
        pool_disk_mb=1024,
        disk_floor_mb=512,
        disk_hard_floor_mb=256,
        backend="e2b",
        e2b=EnvE2BHost(api_key_env="K"),
    )
    fake = RemoteFake(tmp_path / "e2b")
    assert remote_env_runtime(fake, spool_root=tmp_path, host=host).builds
    fake.build_form = None
    assert remote_env_runtime(fake, spool_root=tmp_path, host=host).builds is None


def test_the_policy_keeps_the_docker_backend_by_default():
    host = dict(pool_disk_mb=1024, disk_floor_mb=512, disk_hard_floor_mb=256)
    assert EnvHostPolicy(**host).backend == "docker"
    with pytest.raises(ValueError, match="go together"):
        EnvHostPolicy(**host, backend="e2b")
    with pytest.raises(ValueError, match="go together"):
        EnvHostPolicy(**host, e2b=EnvE2BHost(api_key_env="K"))


def test_docker_env_leases_keep_their_bridge_rules():
    owner = SandboxOwner(run_id="run-1", task_id="task", phase="work")
    env_id = "e" + "1" * 32
    service = SandboxEnvServiceLease(
        idx=0,
        name="main",
        planned_name=env_container_name(env_id, 0),
        image="i" + "a" * 32,
        image_id="sha256:" + "b" * 64,
    )
    values = dict(
        owner=owner,
        env_id=env_id,
        spec_sha256="c" * 64,
        created_at=1.0,
        expires_at=2.0,
        network_mode="public",
        cpus_milli=500,
        memory_mb=64,
        disk_mb=64,
        services=(service,),
    )
    with pytest.raises(ValueError, match="private firewalled bridge"):
        SandboxEnvLease(**values)
    lease = SandboxEnvLease(**values, backend="e2b")
    assert lease.backend == "e2b" and lease.network_name is None
    with pytest.raises(ValueError, match="E2B sandbox"):
        SandboxEnvLease(
            **{
                **values,
                "services": (service.model_copy(update={"container_id": "d" * 64}),),
            },
            backend="e2b",
        )


# -- emulation ------------------------------------------------------------------


class Scripted:
    """A client process yielding ``events`` (a ``None`` blocks forever, an
    Event until it is set, an exception breaks the stream)."""

    pid = 4242

    def __init__(self, events, exit_code=0):
        self._events = events
        self.exit_code = None
        self._code = exit_code
        self.closed = False

    def __iter__(self):
        for event in self._events:
            if event is None:
                threading.Event().wait()
            if isinstance(event, threading.Event):
                event.wait()
                continue
            if isinstance(event, BaseException):
                raise event
            yield event
        self.exit_code = self._code

    def close(self):
        self.closed = True


class Starting:
    """A client without native pids that hands out one Scripted process,
    then ones that end at once."""

    name = "emu"
    native_pid = False

    def __init__(self, process):
        self.process = process
        self.started = []

    def start(self, sandbox_id, argv, *, envs, cwd, user):
        self.started.append(tuple(argv))
        return self.process if len(self.started) == 1 else Scripted([])


def test_start_reads_the_pid_line_and_replays_what_came_with_it():
    process = Scripted([(2, b"warn\n"), (1, b"31"), (1, b"7\nout"), (1, b"put\n")], 3)
    client = Starting(process)
    started = start_group(client, "sb", ["prog", "-x"], envs={}, cwd="/", user="root")
    assert client.started == [tuple(group_argv(["prog", "-x"]))]
    assert started.pid == 317
    assert list(started) == [(2, b"warn\n"), (1, b"out"), (1, b"put\n")]
    assert started.exit_code == 3


def test_start_fails_without_a_pid_line_in_time():
    hung = Scripted([(2, b"slow"), None])
    begin = time.monotonic()
    with pytest.raises(InfrastructureError, match="no pid within 0.2 s"):
        start_group(
            Starting(hung), "sb", ["x"], envs={}, cwd="/", user="root", timeout=0.2
        )
    assert time.monotonic() - begin < 2
    broken = Scripted([(2, b"sh: setsid: not found\n")], 127)
    with pytest.raises(InfrastructureError, match="needs sh and setsid.*not found"):
        start_group(Starting(broken), "sb", ["x"], envs={}, cwd="/", user="root")


def test_a_command_whose_pid_line_comes_late_is_killed():
    """It ran all the same, and no caller holds it: its group is killed."""
    late = threading.Event()
    process = Scripted([(2, b"slow"), late, (1, b"77\n"), None])
    client = Starting(process)
    with pytest.raises(InfrastructureError, match="no pid within 0.2 s"):
        start_group(client, "sb", ["x"], envs={}, cwd="/", user="root", timeout=0.2)
    assert len(client.started) == 1
    late.set()
    deadline = time.monotonic() + 10
    while len(client.started) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.started[1] == ("kill", "-s", "KILL", "--", "-77")
    assert process.closed


class Following:
    """A client without native pids: ``connect`` follows a broken stream
    again with a process that reports no pid (as Daytona's), which ends
    once the broker's ``kill`` runs."""

    name = "emu"
    native_pid = False
    freeze = "signal"

    def __init__(self):
        self.connected = []
        self.commands = []
        self.killed = threading.Event()

    def state(self, sandbox_id):
        return "running"

    def connect(self, sandbox_id, pid):
        self.connected.append(pid)
        follow = Scripted([self.killed], 143)
        follow.pid = 0
        return follow

    def start(self, sandbox_id, argv, *, envs, cwd, user):
        self.commands.append(tuple(argv))
        self.killed.set()
        return Scripted([])


def test_a_service_followed_again_is_still_signalled_as_its_group():
    client = Following()
    backend = SandboxEnvRemoteBackend(client, RemoteTemplates(client, "rsi"), poll=0)
    service = _Service(Scripted([(1, b"up\n"), ConnectionResetError("cut")]), 4242)
    backend._boxes["sb-1"] = _Box("sb-1", {}, "/", None, "running", service)
    threading.Thread(target=backend._drain, args=("sb-1", service), daemon=True).start()
    deadline = time.monotonic() + 10
    while not client.connected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert client.connected == [4242]
    record = SandboxEnvServiceLease(
        idx=0,
        name="main",
        planned_name=env_container_name("e" + "1" * 32, 0),
        image="i" + "a" * 32,
        image_id="sha256:" + "b" * 64,
        sandbox_id="sb-1",
        state="running",
    )
    lease = SandboxEnvLease(
        owner=SandboxOwner(run_id="run-1", task_id="task", phase="work"),
        env_id="e" + "1" * 32,
        spec_sha256="c" * 64,
        created_at=1.0,
        expires_at=2.0,
        network_mode="public",
        cpus_milli=500,
        memory_mb=64,
        disk_mb=64,
        services=(record,),
        backend="daytona",
    )
    _, code = backend.stop_service(lease, lambda lease: lease, "main", timeout_sec=5)
    assert client.commands == [("kill", "-s", "TERM", "--", "-4242")]
    assert code == 143


@pytest.mark.parametrize(
    ("argv", "signum", "code"),
    [
        (["sh", "-c", "exit 7"], None, 7),
        (["sleep", "30"], signal.SIGINT, 130),
        (["sh", "-c", "sleep 30 & sleep 30; wait"], signal.SIGTERM, 143),
    ],
)
def test_the_wrapped_command_leads_its_group_and_its_exit_is_the_shells(
    argv, signum, code
):
    """Run here for real: the reported pid is the group's, a signal to the
    group reaches the command (SIGINT included) and the exit is 128 + N."""
    process = subprocess.Popen(
        group_argv(argv), stdout=subprocess.PIPE, start_new_session=True
    )
    try:
        pid = int(process.stdout.readline())
        assert os.getpgid(pid) == pid != process.pid
        if signum is not None:
            os.killpg(pid, signum)
        assert process.wait(10) == code
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.stdout.close()


def test_the_wrapped_command_reads_no_stdin():
    """Its stdin is /dev/null, though the backend's pipe stays open."""
    process = subprocess.Popen(
        group_argv(["sh", "-c", "cat; echo eof"]),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert process.wait(10) == 0
        assert process.stdout.read().split(b"\n")[1] == b"eof"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.stdin.close()
        process.stdout.close()


def test_the_registry_env_applies_as_a_mapping():
    assert config_env(["A=1", "B=x=y", "C=", "junk", "=v"]) == {
        "A": "1",
        "B": "x=y",
        "C": "",
    }


# The freeze scripts run here only on a made-up /proc, with a signal command
# that edits it: nothing on this host is ever signalled.
SIGNAL_STUB = """#!{python} -I
import pathlib, re, sys
root = pathlib.Path({root!r})
_, name, pid = sys.argv[1:]
with (root.parent / "signals").open("a") as log:
    log.write(f"{{name}} {{pid}}\\n")
if (root.parent / f"stuck-{{pid}}").exists():
    sys.exit(0)
stat = root / pid / "stat"
state = "T" if name == "STOP" else "S"
stat.write_text(re.sub(r"\\) . ", f") {{state}} ", stat.read_text()))
# A fork just before the stop: its child shows up afterwards.
child = root.parent / "forks" / pid
if name == "STOP" and child.exists():
    child.rename(root / (root.parent / "forks" / (pid + ".child")).read_text())
"""


def proc_entry(root, pid, ppid, state="S", *, comm="app", exe="/usr/bin/app"):
    entry = root / str(pid)
    entry.mkdir(parents=True)
    fields = " ".join(["0"] * 17 + [str(1000 + pid), "0", "0"])
    (entry / "stat").write_text(f"{pid} ({comm}) {state} {ppid} {fields}\n")
    (entry / "exe").symlink_to(exe)


@pytest.fixture
def proc(tmp_path):
    root = tmp_path / "proc"
    proc_entry(root, 1, 0, exe="/sbin/tini")
    proc_entry(root, 10, 1, exe="/sbin/tini")  # the backend's daemon
    proc_entry(root, 11, 1, exe="/sbin/tini")  # another of its processes
    proc_entry(root, 20, 10, comm="bash")  # its session shell
    proc_entry(root, 21, 20, comm="sh")  # the freeze script
    proc_entry(root, 22, 21, comm="readlink")  # the script's own command
    proc_entry(root, 30, 1)
    proc_entry(root, 31, 30, "R", comm="a) b (c")
    proc_entry(root, 32, 1, "T")  # stopped by its owner: stays so
    proc_entry(root, 33, 1, "Z")
    proc_entry(root, 34, 1, exe="/opt/agent")
    # 30 forks 35 as it is stopped: only a second round sees 35.
    proc_entry(tmp_path / "forks", 30, 1)
    (tmp_path / "forks" / "30").rename(tmp_path / "forks" / "staged")
    proc_entry(tmp_path / "forks", 35, 30)
    (tmp_path / "forks" / "35").rename(tmp_path / "forks" / "30")
    (tmp_path / "forks" / "30.child").write_text("35")
    stub = tmp_path / "signal"
    stub.write_text(SIGNAL_STUB.format(python=sys.executable, root=str(root)))
    stub.chmod(0o755)
    return root, stub


def freeze(root, stub, record, *spare, tries=30, env=None):
    return subprocess.run(
        ["sh", "-c", FREEZE_SCRIPT, "rsi-freeze", str(root), str(record), str(stub)]
        + ["21", str(tries), *spare],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def thaw(root, stub, record):
    return subprocess.run(
        ["sh", "-c", THAW_SCRIPT, "rsi-thaw", str(root), str(record), str(stub)],
        capture_output=True,
        text=True,
        timeout=60,
    )


def signals(root):
    return (root.parent / "signals").read_text().split("\n")[:-1]


def test_a_signal_freeze_stops_all_but_init_itself_its_line_and_the_spared(proc):
    root, stub = proc
    record = root.parent / "run" / ".rsi-frozen"
    done = freeze(root, stub, record, "init", "/opt/agent")
    assert done.returncode == 0, done.stderr
    # Round after round until none is left: 35 appeared after 30 stopped.
    assert signals(root) == ["STOP 30", "STOP 31", "STOP 35"]
    assert record.read_text() == "30 1030\n31 1031\n35 1035\n"
    assert oct(record.stat().st_mode & 0o777) == "0o600"

    # A recorded pid that now names another process is left alone.
    stat = root / "31" / "stat"
    stat.write_text(stat.read_text().replace(" 1031 ", " 9999 "))
    done = thaw(root, stub, record)
    assert done.returncode == 0, done.stderr
    # Last first: a child goes on before its parent.
    assert signals(root)[3:] == ["CONT 35", "CONT 30"]
    assert not record.exists()
    assert thaw(root, stub, record).returncode == 1


def test_a_signal_freeze_spares_only_what_the_backend_names(proc):
    root, stub = proc
    done = freeze(root, stub, root.parent / "record")
    assert done.returncode == 0, done.stderr
    # The script's ancestors (10, 20) and its own command (22) stay.
    assert signals(root) == ["STOP 11", "STOP 30", "STOP 31", "STOP 34", "STOP 35"]


def test_a_signal_freeze_without_readlink_spares_nothing_and_needs_init(proc, tmp_path):
    root, stub = proc
    tools = tmp_path / "bin"
    tools.mkdir()
    for tool in ("sh", "mkdir", "sleep"):
        (tools / tool).symlink_to(shutil.which(tool))
    env = {"PATH": str(tools)}
    done = freeze(root, stub, root.parent / "record", "init", env=env)
    assert done.returncode == 1
    assert done.stderr.strip() == "cannot read init's executable"
    assert not (root.parent / "signals").exists()
    done = freeze(root, stub, root.parent / "record", "/opt/agent", env=env)
    assert done.returncode == 0, done.stderr
    assert "STOP 34" in signals(root)


def test_a_signal_freeze_spares_every_process_of_its_caller(proc):
    root, stub = proc
    # The session shell's other children (output relays, a stdin holder and
    # its sleep) wait for the freeze to end: they stay.
    proc_entry(root, 23, 20, comm="bash")
    proc_entry(root, 24, 20, comm="bash")
    proc_entry(root, 25, 24, comm="sleep", exe="/bin/sleep")
    # Another session's shell and command stop.
    proc_entry(root, 40, 10, comm="bash")
    proc_entry(root, 41, 40)
    done = freeze(root, stub, root.parent / "record", "init", "caller")
    assert done.returncode == 0, done.stderr
    assert signals(root) == [
        "STOP 30",
        "STOP 31",
        "STOP 34",
        "STOP 40",
        "STOP 41",
        "STOP 35",
    ]


def test_a_signal_freeze_fails_when_a_process_does_not_show_stopped(proc):
    root, stub = proc
    (root.parent / "stuck-31").touch()
    done = freeze(root, stub, root.parent / "record", "init", tries=2)
    assert done.returncode == 1
    assert done.stderr.strip() == "not stopped: 31"
