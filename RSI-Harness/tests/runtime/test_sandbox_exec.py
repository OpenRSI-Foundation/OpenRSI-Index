"""Exec engine: one pump, frame demux, bounded spools, group kills, OOM deltas."""

import base64
import ctypes
import hashlib
import logging
import os
import shutil
import signal
import socket
import stat
import struct
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
import requests
from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime import sandbox_exec
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_exec import (
    DRAIN_MAX_SEC,
    DRAIN_SEC,
    KILL_GRACE_SEC,
    MAX_READ_BYTES,
    RELEASE_GRACE_SEC,
    RETAIN_SEC,
    TICK_SEC,
    ExecProcess,
    ExecPump,
    ExecTarget,
    HostPidfdKiller,
    InContainerKiller,
    ProcessTable,
    default_exec_killer,
    parse_exec_request,
    pidfd_calls,
)
from tests.sandbox_helpers import FakeClock

CONTAINER = "c" * 64
SCOPE = f"/system.slice/docker-{CONTAINER}.scope"
INIT_PID = 100
SESSION = "work-session"
ENV_ID = "e" + "1" * 32
MIB = 1024**2


def frame(stream, data):
    return struct.pack("!BxxxI", stream, len(data)) + data


def outputs(view):
    return base64.b64decode(view["stdout_b64"]), base64.b64decode(view["stderr_b64"])


def daemon_error(message, status):
    response = requests.Response()
    response.status_code = status
    return APIError(message, response=response, explanation=message)


class FakeHost:
    """A fake /proc and cgroupfs: one container scope and its processes."""

    def __init__(self, root):
        self.proc = root / "proc"
        self.scope = root / "cgroup" / SCOPE.lstrip("/")
        self.scope.mkdir(parents=True)
        self.pids = []
        self.table = ProcessTable(proc_root=self.proc, cgroup_root=root / "cgroup")
        self.set_oom(0)
        self.add(INIT_PID, start=10, ns=(1, 1), comm="docker-init")

    def add(self, pid, *, pgrp=None, start=1000, ns=None, state="S", comm="sh"):
        directory = self.proc / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        pgrp = pid if pgrp is None else pgrp
        # state ppid pgrp session, 15 more fields, then starttime (field 22).
        fields = [state, "1", str(pgrp), str(pgrp), *["0"] * 15, str(start), "0"]
        (directory / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields) + "\n")
        ns_pid, ns_pgid = ns or (pid - 190, pgrp - 190)
        (directory / "status").write_text(
            f"Name:\t{comm}\nNSpid:\t{pid}\t{ns_pid}\nNSpgid:\t{pgrp}\t{ns_pgid}\n"
        )
        (directory / "cgroup").write_text(f"0::{SCOPE}\n")
        if pid not in self.pids:
            self.pids.append(pid)
        self._write_procs()

    def remove(self, pid):
        shutil.rmtree(self.proc / str(pid), ignore_errors=True)
        if pid in self.pids:
            self.pids.remove(pid)
        self._write_procs()

    def set_oom(self, count):
        (self.scope / "memory.events").write_text(
            f"low 0\nhigh 0\nmax 0\noom {count}\noom_kill {count}\noom_group_kill 0\n"
        )

    def _write_procs(self):
        (self.scope / "cgroup.procs").write_text("".join(f"{p}\n" for p in self.pids))


class DockerExec:
    def __init__(self, container, command, options):
        self.container = container
        self.command = command
        self.options = options
        self.running = True
        self.exit_code = None
        self.pid = 0
        self.peer = None
        self.detached = False


class FakeApi:
    """The low-level exec calls; attach streams are real socket pairs."""

    def __init__(self):
        self.execs = {}
        self.created = []
        self.create_error = None
        self.start_error = None
        self.inspect_errors = []
        self.handle = None

    def exec_create(self, container, cmd, **options):
        if self.create_error is not None:
            raise self.create_error
        identity = hashlib.sha256(str(len(self.execs)).encode()).hexdigest()
        self.execs[identity] = DockerExec(container, list(cmd), options)
        self.created.append(identity)
        return {"Id": identity}

    def exec_start(self, exec_id, detach=False, tty=False, socket=False):
        record = self.execs[exec_id]
        if self.start_error is not None:
            raise self.start_error
        if detach:
            record.detached = True
            return b""
        assert socket and not tty
        if self.handle is not None:
            handle, record.peer = self.handle()
            return handle
        ours, record.peer = _socketpair()
        return ours

    def exec_inspect(self, exec_id):
        if self.inspect_errors:
            raise self.inspect_errors.pop(0)
        record = self.execs[exec_id]
        return {
            "Running": record.running,
            "ExitCode": record.exit_code,
            "Pid": record.pid,
        }

    def last(self):
        return self.execs[self.created[-1]]


def _socketpair():
    ours, theirs = socket.socketpair()
    theirs.settimeout(5)
    return ours, theirs


class RecordingKiller:
    def __init__(self):
        self.calls = []
        self.delivered = True
        self.effect = None

    def signal(self, process, signum, *, group):
        self.calls.append((process.host_pid, signum, group))
        if self.effect is not None:
            self.effect(process, signum, group)
        return self.delivered


class Rig:
    """One pump driven by hand: fake clock, fake Engine, fake host tree."""

    def __init__(self, tmp_path, *, killer=None, **options):
        self.clock = FakeClock()
        self.host = FakeHost(tmp_path / "host")
        self.api = FakeApi()
        self.killer = killer if killer is not None else RecordingKiller()
        self.summaries = []
        self.state = {"Running": True, "Paused": False, "Pid": INIT_PID}
        self.spool = tmp_path / "spool"
        self.pump = ExecPump(
            self.api,
            self.spool,
            killer=self.killer,
            table=self.host.table,
            on_finish=self.summaries.append,
            clock=self.clock,
            start_threads=False,
            **options,
        )
        self.next_pid = 200

    def target(self, on_inspect=None):
        def inspect():
            view = None if self.state is None else dict(self.state)
            if on_inspect is not None:
                on_inspect()  # e.g. the service ends right after it was seen
            if view is None:
                return None
            return {"Id": CONTAINER, "State": view}

        return ExecTarget(CONTAINER, inspect)

    def start(
        self,
        argv=("sh", "-c", "work"),
        *,
        session=SESSION,
        request_id="r1",
        output_limit=1024,
        deadline=None,
        started=True,
        target=None,
        **fields,
    ):
        request = parse_exec_request({"service": "main", "argv": list(argv), **fields})
        exec_id = self.pump.start(
            target or self.target(),
            request,
            session=session,
            env_id=ENV_ID,
            request_id=request_id,
            output_limit=output_limit,
            deadline=deadline,
        )
        docker = self.api.last()
        if started:
            self.launch(docker)
        self.io()
        return exec_id, docker

    def launch(self, docker):
        docker.pid = self.next_pid
        self.next_pid += 10
        self.host.add(docker.pid, start=5000 + docker.pid)

    def io(self):
        for _ in range(64):
            self.pump.step_io(0)

    def emit(self, docker, stream, data):
        docker.peer.sendall(frame(stream, data))
        self.io()

    def exit(self, docker, code, *, close=True):
        docker.running, docker.exit_code = False, code
        self.host.remove(docker.pid)
        if close:
            docker.peer.close()
        self.io()

    def turn(self, seconds=0.0):
        """Advance the fake clock tick by tick, running both loops."""
        # A rounded grid: float drift must not move a deadline by a tick.
        end = round(self.clock.now + seconds, 6)
        while True:
            self.io()
            self.pump.step_control()
            self.io()
            if self.clock.now >= end:
                return
            self.clock.now = round(min(end, self.clock.now + TICK_SEC), 6)

    def read(self, exec_id, out=0, err=0, *, session=SESSION, max_bytes=MAX_READ_BYTES):
        return self.pump.read(
            exec_id,
            session=session,
            stdout_offset=out,
            stderr_offset=err,
            max_bytes=max_bytes,
        )

    def exec_dir(self, exec_id):
        return self.spool / "x" / exec_id


# -- request -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "code", "field"),
    [
        ({"service": "main", "argv": []}, "invalid", "argv"),
        ({"service": "main", "argv": "sh -c true"}, "invalid", "argv"),
        ({"service": "main", "argv": ["true"], "cwd": "work"}, "invalid", "cwd"),
        ({"service": "main", "argv": ["true"], "cwd": "/a/../b"}, "invalid", "cwd"),
        ({"service": "main", "argv": ["true"], "timeout_sec": 0}, "invalid", "timeout"),
        ({"service": "main", "argv": ["true"], "tty": True}, "invalid", "tty"),
        ({"service": "-x", "argv": ["true"]}, "invalid", "service"),
        ({"service": "main", "argv": ["true"], "user": "a b"}, "invalid", "user"),
        ({"service": "main", "argv": ["x" * 65537]}, "quota", "argv"),
    ],
)
def test_requests_are_strict_and_oversize_commands_are_quota(raw, code, field):
    with pytest.raises(SandboxError) as caught:
        parse_exec_request(raw)
    assert caught.value.code == code
    assert caught.value.field.startswith(field)


def test_request_errors_never_echo_argv_or_env_values():
    secret = "hunter2-" * 9000
    for raw in (
        {"service": "main", "argv": [secret]},
        {"service": "main", "argv": ["true"], "env": {"TOKEN": secret}},
        {"service": "main", "argv": ["true"], "env": {"TOKEN": "a\x00" + secret}},
    ):
        with pytest.raises(SandboxError) as caught:
            parse_exec_request(raw)
        assert "hunter2" not in str(caught.value)
    request = parse_exec_request(
        {"service": "main", "argv": ["true"], "env": {"A": "1"}, "timeout_sec": 2}
    )
    assert (request.timeout_sec, dict(request.env), request.merge_stderr) == (
        2.0,
        {"A": "1"},
        False,
    )


# -- start, demux, spool -----------------------------------------------------------


def test_start_attaches_without_a_tty_and_demuxes_frames_split_anywhere(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(
        ["python3", "-m", "task"],
        cwd="/app",
        env={"MODE": "fast"},
        user="1000:1000",
    )
    assert sandbox_exec.EXEC_ID.fullmatch(exec_id)
    assert docker.container == CONTAINER
    assert docker.command == ["python3", "-m", "task"]
    assert docker.options == {
        "stdout": True,
        "stderr": True,
        "stdin": False,
        "tty": False,
        "environment": {"MODE": "fast"},
        "workdir": "/app",
        "user": "1000:1000",
    }
    stream = frame(1, b"hello ") + frame(2, b"oops\n") + frame(1, b"world\n")
    for byte in stream:  # every header and payload split at every byte
        docker.peer.sendall(bytes([byte]))
        rig.io()

    view = rig.read(exec_id)

    assert outputs(view) == (b"hello world\n", b"oops\n")
    assert (view["stdout_total"], view["stderr_total"]) == (12, 5)
    assert (view["stdout_offset"], view["stderr_offset"]) == (12, 5)
    assert (view["state"], view["exit_code"], view["truncated"]) == (
        "running",
        None,
        False,
    )
    directory = rig.exec_dir(exec_id)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((rig.spool / "x").stat().st_mode) == 0o700
    for name in ("stdout", "stderr"):
        assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600
    assert (directory / "stdout").read_bytes() == b"hello world\n"


def test_merged_output_keeps_the_frame_order(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(merge_stderr=True)
    for stream, data in ((1, b"a\n"), (2, b"b\n"), (1, b"c\n")):
        rig.emit(docker, stream, data)
    view = rig.read(exec_id)
    assert outputs(view) == (b"a\nb\nc\n", b"")
    assert view["stderr_total"] == 0


def test_output_past_the_cap_is_read_and_discarded_without_stalling(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(output_limit=1024)
    chunk = b"z" * (256 * 1024)
    written = threading.Event()

    def produce():
        for _ in range(32):  # 8 MiB, far beyond any socket buffer
            docker.peer.sendall(frame(1, chunk))
        docker.peer.sendall(frame(2, b"tail\n"))
        written.set()

    writer = threading.Thread(target=produce)
    writer.start()
    deadline = time.monotonic() + 10
    while not written.is_set() and time.monotonic() < deadline:
        rig.pump.step_io(0.01)
    writer.join(timeout=1)
    rig.io()

    assert written.is_set(), "the producer stalled on a full pipe"
    view = rig.read(exec_id)
    assert (view["stdout_total"], view["stderr_total"], view["truncated"]) == (
        1024,
        5,
        True,
    )
    assert outputs(view) == (b"z" * 1024, b"tail\n")
    assert (rig.exec_dir(exec_id) / "stdout").stat().st_size == 1024


def test_one_chatty_exec_cannot_starve_the_others_in_a_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_exec, "READ_CHUNK", 1024)
    rig = Rig(tmp_path)
    chatty, loud = rig.start(output_limit=MIB)
    quiet, calm = rig.start(request_id="r2")
    loud.peer.sendall(frame(1, b"a" * 100_000))
    calm.peer.sendall(frame(1, b"b" * 10))

    rig.pump.step_io(0)

    assert rig.read(chatty)["stdout_total"] == 4 * 1024 - 8  # four reads a turn
    assert rig.read(quiet)["stdout_total"] == 10
    rig.io()
    assert rig.read(chatty, max_bytes=8)["stdout_total"] == 100_000


def test_stream_bytes_buffered_during_the_header_read_are_not_lost(tmp_path):
    rig = Rig(tmp_path)

    def over_read():
        ours, theirs = _socketpair()
        theirs.sendall(b"HTTP/1.1 101 UPGRADED\r\n\r\n" + frame(1, b"early\n"))
        reader = ours.makefile("rb")
        # http.client reads the head line by line through a buffered reader.
        assert reader.readline() == b"HTTP/1.1 101 UPGRADED\r\n"
        assert reader.readline() == b"\r\n"
        response = SimpleNamespace(
            raw=SimpleNamespace(_fp=SimpleNamespace(fp=reader)), close=reader.close
        )
        handle = SimpleNamespace(_sock=ours, _response=response, close=lambda: None)
        return handle, theirs

    rig.api.handle = over_read
    exec_id, docker = rig.start()
    rig.emit(docker, 1, b"late\n")
    assert outputs(rig.read(exec_id))[0] == b"early\nlate\n"


def test_a_corrupt_frame_truncates_and_the_exec_still_ends_by_inspect(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    docker.peer.sendall(frame(1, b"ok\n") + frame(3, b"junk"))
    rig.io()
    rig.exit(docker, 0, close=False)
    rig.turn(0.1)
    view = rig.read(exec_id)
    assert (view["state"], view["exit_code"], view["truncated"]) == ("exited", 0, True)
    assert outputs(view)[0] == b"ok\n"


# -- completion --------------------------------------------------------------------


def test_an_exec_ends_when_the_engine_reports_exit_and_the_stream_closes(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.emit(docker, 1, b"done\n")
    rig.turn(1.0)
    assert rig.read(exec_id)["state"] == "running"

    rig.exit(docker, 3)
    rig.turn(TICK_SEC)

    view = rig.read(exec_id)
    assert (view["state"], view["exit_code"], view["reason"], view["signal"]) == (
        "exited",
        3,
        None,
        None,
    )
    assert outputs(view) == (b"done\n", b"")
    (summary,) = rig.summaries
    assert (summary.exec_id, summary.session, summary.env_id, summary.state) == (
        exec_id,
        SESSION,
        ENV_ID,
        "exited",
    )
    assert (summary.stdout_total, summary.stderr_total, summary.output_limit) == (
        5,
        0,
        1024,
    )
    rig.turn(5.0)
    assert len(rig.summaries) == 1


def test_the_spec_timings_are_pinned():
    assert (DRAIN_SEC, DRAIN_MAX_SEC, KILL_GRACE_SEC, RETAIN_SEC) == (
        1.0,
        3.0,
        2.0,
        600.0,
    )


def test_a_stream_held_by_a_background_process_is_closed_after_one_second(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.emit(docker, 1, b"started\n")
    rig.exit(docker, 0, close=False)  # a background child keeps stdout open
    rig.turn(0.6)
    exited_at = rig.clock.now
    assert rig.read(exec_id)["state"] == "running"

    rig.turn(1.1)

    view = rig.read(exec_id)
    # Whatever the child writes later is not captured: the output is cut.
    assert (view["state"], view["exit_code"], view["truncated"]) == (
        "exited",
        0,
        True,
    )
    assert outputs(view)[0] == b"started\n"
    assert rig.clock.now - exited_at <= 1.6
    assert docker.peer.recv(1) == b""  # our end of the stream is closed


def test_the_drain_waits_while_output_still_arrives_then_ends_complete(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.exit(docker, 3, close=False)
    rig.turn(TICK_SEC)
    for index in range(4):  # the pump is behind: bytes keep coming after exit
        rig.turn(0.5)
        rig.emit(docker, 1, b"%d" % index)
    docker.peer.close()
    rig.turn(TICK_SEC)

    view = rig.read(exec_id)
    assert (view["state"], view["exit_code"], view["truncated"]) == (
        "exited",
        3,
        False,
    )
    assert outputs(view)[0] == b"0123"


def test_the_drain_ends_three_seconds_after_the_exit_and_marks_truncation(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.exit(docker, 0, close=False)
    exited_at = rig.clock.now
    rig.turn(TICK_SEC)
    while rig.read(exec_id)["state"] == "running":
        assert rig.clock.now - exited_at < 3.0 + 2 * TICK_SEC
        docker.peer.sendall(frame(1, b"."))  # a child that never stops writing
        rig.turn(TICK_SEC)
    view = rig.read(exec_id)
    assert rig.clock.now - exited_at >= 3.0
    assert (view["state"], view["exit_code"], view["truncated"]) == (
        "exited",
        0,
        True,
    )


def test_bytes_already_sent_are_kept_when_the_stream_is_cut(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    docker.peer.sendall(frame(1, b"sent before the stop\n"))  # not read yet
    rig.pump.env_stopped(ENV_ID)
    rig.turn()
    view = rig.read(exec_id)
    assert (view["state"], view["reason"], view["truncated"]) == (
        "killed",
        "env_stopped",
        True,
    )
    assert outputs(view)[0] == b"sent before the stop\n"


def test_a_timeout_never_signals_what_an_exited_command_left_behind(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(timeout_sec=0.5)
    rig.exit(docker, 0, close=False)  # `sleep 30 & echo started`
    rig.turn(1.2)
    view = rig.read(exec_id)
    assert (view["state"], view["reason"], view["exit_code"]) == ("exited", None, 0)
    assert rig.killer.calls == []


def test_a_timeout_due_after_an_unseen_exit_never_signals(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(timeout_sec=1.9)  # `sh -c 'sleep 100 & sleep 1.8'`
    rig.turn(1.8)
    rig.exit(docker, 0, close=False)  # no inspect runs until 2.0 s
    rig.turn(0.1)

    # At 1.9 s /proc shows the pinned leader gone: its exit is asked for first.
    assert rig.killer.calls == []
    rig.turn(1.2)
    view = rig.read(exec_id)
    assert (view["state"], view["reason"], view["exit_code"], view["signal"]) == (
        "exited",
        None,
        0,
        None,
    )
    assert rig.killer.calls == []


def test_a_closed_stream_does_not_end_a_running_exec(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    docker.peer.close()  # e.g. `exec >&- 2>&-; sleep 100`
    rig.io()
    rig.turn(5.0)
    assert rig.read(exec_id)["state"] == "running"
    rig.exit(docker, 0, close=False)
    rig.turn(1.0)
    assert rig.read(exec_id)["state"] == "exited"


def test_an_exec_the_engine_forgets_is_lost(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.api.inspect_errors = [NotFound("no such exec")]
    rig.turn(1.0)
    view = rig.read(exec_id)
    assert (view["state"], view["exit_code"], view["reason"]) == ("lost", None, None)

    rig.start(request_id="r2")
    rig.api.inspect_errors = [OSError("daemon unreachable")] * 20
    rig.turn(9.0)
    assert rig.pump.running(SESSION) == 1
    rig.turn(1.5)
    assert rig.pump.running(SESSION) == 0
    assert [summary.state for summary in rig.summaries] == ["lost", "lost"]


def test_an_exec_killed_with_its_service_is_env_stopped(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.state = {"Running": False, "Paused": False, "Pid": 0}
    rig.exit(docker, 137)
    rig.turn(TICK_SEC)
    view = rig.read(exec_id)
    assert (view["state"], view["exit_code"], view["reason"]) == (
        "killed",
        137,
        "env_stopped",
    )


# -- timeouts, deadlines and kills -------------------------------------------------


def dies_on(rig, docker, *signals, code=None):
    """The leader exits when it receives one of ``signals``."""

    def effect(process, signum, group):
        if signum in signals and docker.running:
            rig.exit(docker, code or 128 + signum)

    return effect


def test_timeout_signals_the_group_term_then_kill_and_keeps_the_service(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(["sleep", "300"], timeout_sec=2)
    started = rig.clock.now
    rig.killer.effect = dies_on(rig, docker, signal.SIGKILL)

    rig.turn(2.0 - TICK_SEC)
    assert rig.killer.calls == []
    rig.turn(TICK_SEC)
    assert rig.killer.calls == [(docker.pid, signal.SIGTERM, True)]
    assert rig.read(exec_id)["state"] == "running"  # TERM was ignored
    rig.turn(2.0 - TICK_SEC)
    assert len(rig.killer.calls) == 1  # KILL follows TERM by 2 s, not sooner

    rig.turn(TICK_SEC)
    rig.turn(TICK_SEC)

    assert rig.killer.calls[1:] == [(docker.pid, signal.SIGKILL, True)]
    view = rig.read(exec_id)
    assert (view["state"], view["reason"], view["signal"], view["exit_code"]) == (
        "timed_out",
        "timeout",
        "KILL",
        137,
    )
    assert rig.clock.now - started <= 4.0 + 2 * TICK_SEC
    assert rig.state["Running"] is True  # the service is never touched


def test_a_leader_that_dies_on_term_ends_in_one_tick_without_kill(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(["sleep", "300"], timeout_sec=2)
    rig.killer.effect = dies_on(rig, docker, signal.SIGTERM)
    rig.turn(2.0 + 2 * TICK_SEC)
    view = rig.read(exec_id)
    assert (view["state"], view["signal"], view["exit_code"]) == (
        "timed_out",
        "TERM",
        143,
    )
    assert [call[1] for call in rig.killer.calls] == [signal.SIGTERM]


def test_a_timeout_waits_for_members_that_ignore_term_then_kills_them(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(timeout_sec=1)
    member = docker.pid + 1
    rig.host.add(member, pgrp=docker.pid, start=5000 + member)
    rig.killer.effect = dies_on(rig, docker, signal.SIGTERM)

    rig.turn(1.0 + 3 * TICK_SEC)
    assert rig.read(exec_id)["state"] == "running"  # the group still has a member

    rig.killer.effect = lambda process, signum, group: rig.host.remove(member)
    rig.turn(2.0)
    rig.turn(TICK_SEC)

    assert [call[1] for call in rig.killer.calls] == [signal.SIGTERM, signal.SIGKILL]
    view = rig.read(exec_id)
    assert (view["state"], view["signal"], view["exit_code"]) == (
        "timed_out",
        "KILL",
        143,
    )


def test_the_env_deadline_ends_an_exec_without_a_timeout(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(deadline=rig.clock.now + 5)
    rig.killer.effect = dies_on(rig, docker, signal.SIGTERM)
    rig.turn(4.9)
    assert rig.killer.calls == []
    rig.turn(0.2)
    view = rig.read(exec_id)
    assert (view["state"], view["reason"]) == ("killed", "deadline")


def test_the_earlier_of_timeout_and_deadline_names_the_reason(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(timeout_sec=10, deadline=rig.clock.now + 3)
    rig.killer.effect = dies_on(rig, docker, signal.SIGTERM)
    rig.turn(3.2)
    assert rig.read(exec_id)["reason"] == "deadline"


def test_a_leader_that_survives_kill_is_abandoned(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(timeout_sec=1)
    rig.turn(1.0 + 2.0 + sandbox_exec.ABANDON_SEC + 2 * TICK_SEC)
    view = rig.read(exec_id)
    assert (view["state"], view["exit_code"], view["signal"]) == (
        "timed_out",
        None,
        "KILL",
    )
    assert docker.peer.recv(1) == b""


@pytest.mark.parametrize("reported", [True, False])
def test_a_killed_leader_waits_for_its_late_exit_report(tmp_path, reported):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(timeout_sec=1)

    def killed(process, signum, group):
        if signum == signal.SIGKILL:
            rig.host.remove(docker.pid)  # gone; the Engine has not said so

    rig.killer.effect = killed
    rig.turn(1.0 + 2.0 + 2.0 + 1.0)  # past the plain 2 s abandon step
    assert rig.read(exec_id)["state"] == "running"
    if reported:
        rig.exit(docker, 137)
    rig.turn(1.0 + TICK_SEC)

    view = rig.read(exec_id)
    assert (view["state"], view["signal"], view["exit_code"]) == (
        "timed_out",
        "KILL",
        137 if reported else None,
    )


def test_the_leader_is_resolved_lazily_and_pinned_before_any_signal(tmp_path):
    rig = Rig(tmp_path)
    # The Engine reports Pid 0 until the process started (VERIFIED).
    exec_id, docker = rig.start(timeout_sec=0.1, started=False)
    rig.turn(0.2)
    assert rig.killer.calls == []  # nothing to signal yet
    rig.launch(docker)
    rig.killer.effect = dies_on(rig, docker, signal.SIGKILL)
    rig.turn(2.0)
    assert rig.killer.calls == [(docker.pid, signal.SIGKILL, True)]
    assert rig.read(exec_id)["state"] == "timed_out"


def test_kill_interrupts_once_without_escalation(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.killer.effect = dies_on(rig, docker, signal.SIGINT, code=130)

    result = rig.pump.kill(exec_id, session=SESSION, signal="INT", scope="group")

    assert result == {"delivered": True, "state": "running"}
    rig.turn(5.0)
    view = rig.read(exec_id)
    assert (view["state"], view["reason"], view["signal"], view["exit_code"]) == (
        "killed",
        "interrupt",
        "INT",
        130,
    )
    assert rig.killer.calls == [(docker.pid, signal.SIGINT, True)]
    again = rig.pump.kill(exec_id, session=SESSION, signal="KILL", scope="group")
    assert again == {"delivered": False, "state": "killed"}
    assert len(rig.killer.calls) == 1


def test_an_exec_that_ends_while_its_kill_is_delivered_is_an_interrupt(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()

    def dies_before_the_call_returns(process, signum, group):
        rig.exit(docker, 143)
        rig.turn(TICK_SEC)  # the control thread sees the end meanwhile

    rig.killer.effect = dies_before_the_call_returns
    assert rig.pump.kill(exec_id, session=SESSION)["delivered"] is True
    rig.turn(TICK_SEC)
    view = rig.read(exec_id)
    assert (view["state"], view["reason"], view["signal"]) == (
        "killed",
        "interrupt",
        "TERM",
    )


def test_process_scope_signals_only_the_leader(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.pump.kill(exec_id, session=SESSION, signal="HUP", scope="process")
    assert rig.killer.calls == [(docker.pid, signal.SIGHUP, False)]


def test_an_undelivered_kill_does_not_mark_an_interrupt(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.killer.delivered = False
    assert rig.pump.kill(exec_id, session=SESSION)["delivered"] is False
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    assert (rig.read(exec_id)["state"], rig.read(exec_id)["signal"]) == (
        "exited",
        None,
    )


@pytest.mark.parametrize("code", [0, 1, 143])
def test_a_command_that_survives_its_signal_exits_normally(tmp_path, code):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()  # `trap 'echo reload' HUP; ...`
    result = rig.pump.kill(exec_id, session=SESSION, signal="HUP", scope="process")
    assert result == {"delivered": True, "state": "running"}
    rig.turn(1.0)
    rig.exit(docker, code)
    rig.turn(TICK_SEC)
    view = rig.read(exec_id)
    # Only 128 + HUP would be a death by the signal; 143 is some other end.
    assert (view["state"], view["reason"], view["exit_code"], view["signal"]) == (
        "exited",
        None,
        code,
        "HUP",
    )


@pytest.mark.parametrize("stop", ["exit", "env_stopped"])
def test_on_finish_runs_before_anyone_sees_the_end(tmp_path, stop):
    seen = []

    def settle(summary):
        view = rig.read(summary.exec_id)
        seen.append((summary.state, view["state"], rig.pump.running(SESSION)))

    rig = Rig(tmp_path)
    rig.pump._on_finish = settle
    exec_id, docker = rig.start()
    if stop == "exit":
        rig.exit(docker, 0)
    else:
        rig.pump.env_stopped(ENV_ID, release=True)
    rig.turn(TICK_SEC)

    # The broker settles execs_running while the exec still shows as running.
    assert seen == [("exited" if stop == "exit" else "killed", "running", 1)]
    assert rig.read(exec_id)["state"] == seen[0][0]
    assert rig.pump.running(SESSION) == 0


@pytest.mark.parametrize(
    ("options", "code", "field"),
    [
        ({"signal": "STOP"}, "invalid", "signal"),
        ({"signal": 9}, "invalid", "signal"),
        ({"scope": "container"}, "invalid", "scope"),
        ({"session": "judge-session"}, "permission", "exec_id"),
        ({"exec_id": "x" + "0" * 32}, "permission", "exec_id"),
    ],
)
def test_kill_validates_signal_scope_and_ownership(tmp_path, options, code, field):
    rig = Rig(tmp_path)
    exec_id, _ = rig.start()
    arguments = {"exec_id": exec_id, "session": SESSION, **options}
    with pytest.raises(SandboxError) as caught:
        rig.pump.kill(
            arguments.pop("exec_id"), session=arguments.pop("session"), **arguments
        )
    assert (caught.value.code, caught.value.field) == (code, field)


# -- OOM ---------------------------------------------------------------------------


def test_oom_kills_are_the_memory_events_delta_over_the_exec(tmp_path):
    rig = Rig(tmp_path)
    rig.host.set_oom(3)  # earlier kills, e.g. a previous exec: never sticky
    exec_id, docker = rig.start(["dd", "if=/dev/zero", "of=/dev/null", "bs=100M"])
    assert rig.read(exec_id)["oom_kills"] == 0
    rig.host.set_oom(4)
    assert rig.read(exec_id)["oom_kills"] == 1  # live while running
    rig.exit(docker, 137)
    rig.turn(TICK_SEC)

    view = rig.read(exec_id)
    assert (view["state"], view["exit_code"], view["oom_kills"]) == ("exited", 137, 1)
    after, docker = rig.start(["true"], request_id="r2")
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    assert rig.read(after)["oom_kills"] == 0


# -- idempotency and admission -----------------------------------------------------


def test_request_id_replays_the_same_exec_and_refuses_a_different_one(tmp_path):
    rig = Rig(tmp_path)
    exec_id, _ = rig.start(["make", "test"], request_id="req-1")
    assert rig.start(["make", "test"], request_id="req-1", started=False)[0] == exec_id
    assert len(rig.api.created) == 1

    with pytest.raises(SandboxError) as caught:
        rig.start(["make", "lint"], request_id="req-1", started=False)
    assert (caught.value.code, caught.value.field) == ("invalid", "request_id")
    other, _ = rig.start(["make", "test"], request_id="req-1", session="judge")
    assert other != exec_id and len(rig.api.created) == 2

    rig.state["Paused"] = True
    with pytest.raises(SandboxError) as caught:
        rig.start(request_id="req-2", started=False)
    assert caught.value.code == "busy"
    rig.state["Paused"] = False
    with pytest.raises(SandboxError) as caught:  # a failure replays, never reruns
        rig.start(request_id="req-2", started=False)
    assert caught.value.code == "busy"
    assert len(rig.api.created) == 2


@pytest.mark.parametrize("request_id", ["", "a" * 65, "a b", "ü", None, 7])
def test_request_ids_must_be_safe(tmp_path, request_id):
    rig = Rig(tmp_path)
    with pytest.raises(SandboxError) as caught:
        rig.start(request_id=request_id, started=False)
    assert (caught.value.code, caught.value.field) == ("invalid", "request_id")


@pytest.mark.parametrize(
    ("state", "code"),
    [
        (None, "invalid"),
        ({"Running": False, "Paused": False, "Pid": 0}, "invalid"),
        ({"Running": True, "Paused": True, "Pid": INIT_PID}, "busy"),
    ],
)
def test_start_refuses_absent_stopped_and_paused_services(tmp_path, state, code):
    rig = Rig(tmp_path)
    rig.state = state
    with pytest.raises(SandboxError) as caught:
        rig.start(started=False)
    assert (caught.value.code, caught.value.field) == (code, "service")
    assert rig.api.created == []
    assert list((rig.spool / "x").iterdir()) == []


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (daemon_error("container is paused", 409), "busy"),
        (daemon_error("no such container", 404), "invalid"),
    ],
)
def test_engine_refusals_are_caller_errors_and_leave_no_spool(tmp_path, error, code):
    rig = Rig(tmp_path)
    rig.api.create_error = error
    with pytest.raises(SandboxError) as caught:
        rig.start(started=False)
    assert caught.value.code == code
    assert list((rig.spool / "x").iterdir()) == []


def test_an_unattached_start_is_contained_and_reported(tmp_path):
    rig = Rig(tmp_path)
    rig.api.start_error = OSError("connection reset")
    original = rig.api.exec_start

    def start_then_fail(exec_id, **options):
        rig.launch(rig.api.execs[exec_id])  # the process may run unattached
        return original(exec_id, **options)

    rig.api.exec_start = start_then_fail
    with pytest.raises(SandboxError) as caught:
        rig.start(started=False)
    assert (caught.value.code, caught.value.field) == ("infrastructure", "exec")
    assert rig.killer.calls == [(rig.api.last().pid, signal.SIGKILL, True)]
    assert list((rig.spool / "x").iterdir()) == []


def test_an_unknown_user_is_a_caller_error_that_never_echoes_the_engine(tmp_path):
    rig = Rig(tmp_path)
    rig.api.create_error = daemon_error(
        "unable to find user nosuchuser: no matching entries in passwd file", 400
    )
    with pytest.raises(SandboxError) as caught:
        rig.start(started=False, user="nosuchuser")
    assert (caught.value.code, caught.value.field) == ("invalid", "user")
    assert "nosuchuser" not in str(caught.value)
    with pytest.raises(SandboxError) as caught:  # replayed, never rerun
        rig.start(started=False, user="nosuchuser")
    assert (caught.value.code, caught.value.field) == ("invalid", "user")
    with pytest.raises(SandboxError) as caught:
        rig.start(request_id="r2", started=False)
    assert (caught.value.code, caught.value.field) == ("invalid", "exec")
    assert list((rig.spool / "x").iterdir()) == []


@pytest.mark.parametrize(
    "ending",
    [
        "init_exits",  # the Engine still reports the old Pid
        "service_stops",
        "service_is_replaced",  # restarted: another init
        "cgroup_is_removed",
        "init_is_a_zombie",  # dead, not yet reaped, still reported running
    ],
)
def test_a_service_ending_during_start_is_not_running_not_a_broker_failure(
    tmp_path, ending
):
    rig = Rig(tmp_path)
    ended = []

    def end_service():  # right after the first inspect saw it running
        if ended:
            return
        ended.append(ending)
        if ending == "init_exits":
            rig.host.remove(INIT_PID)
        elif ending == "service_stops":
            rig.state = {"Running": False, "Paused": False, "Pid": 0}
            rig.host.remove(INIT_PID)
        elif ending == "service_is_replaced":
            rig.state = {"Running": True, "Paused": False, "Pid": INIT_PID + 1}
            rig.host.remove(INIT_PID)
        elif ending == "cgroup_is_removed":
            rig.state = {"Running": False, "Paused": False, "Pid": 0}
            (rig.host.scope / "memory.events").unlink()
        else:
            rig.host.add(INIT_PID, start=10, ns=(1, 1), state="Z", comm="init")
            (rig.host.scope / "memory.events").unlink()

    target = rig.target(on_inspect=end_service)
    with pytest.raises(SandboxError) as caught:
        rig.start(started=False, target=target)
    assert (caught.value.code, caught.value.field) == ("invalid", "service")
    with pytest.raises(SandboxError) as caught:  # a retry replays the error
        rig.start(started=False, target=target)
    assert (caught.value.code, caught.value.field) == ("invalid", "service")
    assert rig.api.created == []


def test_a_running_service_with_an_unreadable_cgroup_fails_only_this_call(tmp_path):
    rig = Rig(tmp_path)
    (rig.host.scope / "memory.events").unlink()
    with pytest.raises(SandboxError) as caught:
        rig.start(started=False)
    assert (caught.value.code, caught.value.field) == ("infrastructure", "service")
    with pytest.raises(SandboxError) as caught:
        rig.start(started=False)
    assert caught.value.code == "infrastructure"
    rig.state["Pid"] = 999  # the Engine names a PID outside the container
    with pytest.raises(SandboxError) as caught:
        rig.start(request_id="r2", started=False)
    assert (caught.value.code, caught.value.field) == ("invalid", "service")
    assert rig.api.created == []


def test_an_env_identity_failure_reaches_the_broker_and_replays(tmp_path):
    rig = Rig(tmp_path)

    def drifted():
        raise InfrastructureError("recovery_required: service is not owned")

    target = ExecTarget(CONTAINER, drifted)
    for _ in range(2):
        with pytest.raises(InfrastructureError, match="recovery_required"):
            rig.start(started=False, target=target)
    assert rig.api.created == []


def test_running_execs_are_bounded_across_sessions(tmp_path):
    rig = Rig(tmp_path, max_running=2)
    rig.start(request_id="a")
    rig.start(request_id="b", session="judge")
    with pytest.raises(SandboxError) as caught:
        rig.start(request_id="c", started=False)
    assert (caught.value.code, caught.value.field) == ("quota", "execs_running")


def test_start_returns_once_the_engine_reports_the_process(tmp_path):
    # The stream is up before runc starts the process; a freeze in between
    # would make runc refuse the start.
    rig = Rig(tmp_path, start_confirm_sec=5.0)
    polls = []
    exec_inspect = rig.api.exec_inspect

    def inspect(exec_id):
        polls.append(exec_id)
        if len(polls) == 3:
            rig.launch(rig.api.execs[exec_id])
        return exec_inspect(exec_id)

    rig.api.exec_inspect = inspect
    exec_id, docker = rig.start(started=False)
    assert len(polls) == 3
    assert rig.pump._records[exec_id].host_pid == docker.pid > 0


def test_start_confirmation_is_bounded_and_ends_with_the_exec(tmp_path):
    rig = Rig(tmp_path, start_confirm_sec=0.05)
    began = time.monotonic()
    rig.start(started=False, request_id="never")
    assert 0.05 <= time.monotonic() - began < 2

    exec_start = rig.api.exec_start

    def ends_at_once(exec_id, **options):
        handle = exec_start(exec_id, **options)
        rig.api.execs[exec_id].running = False
        rig.api.execs[exec_id].exit_code = 126
        return handle

    rig.pump._start_confirm_sec = 60.0
    rig.api.exec_start = ends_at_once
    began = time.monotonic()
    rig.start(started=False, request_id="ended")
    assert time.monotonic() - began < 2


def test_starts_in_flight_count_against_the_running_bound(tmp_path):
    rig = Rig(tmp_path, max_running=2)
    rig.start(request_id="a")
    entered, release = threading.Event(), threading.Event()
    original = rig.api.exec_create

    def slow_create(container, cmd, **options):
        entered.set()
        assert release.wait(5)
        return original(container, cmd, **options)

    rig.api.exec_create = slow_create
    request = parse_exec_request({"service": "main", "argv": ["true"]})
    slow = threading.Thread(
        target=rig.pump.start,
        args=(rig.target(), request),
        kwargs={
            "session": SESSION,
            "env_id": ENV_ID,
            "request_id": "b",
            "output_limit": 1024,
        },
    )
    slow.start()
    try:
        assert entered.wait(5)
        with pytest.raises(SandboxError) as caught:
            rig.pump.start(
                rig.target(),
                request,
                session="judge",
                env_id=ENV_ID,
                request_id="c",
                output_limit=1024,
            )
        assert (caught.value.code, caught.value.field) == ("quota", "execs_running")
    finally:
        release.set()
        slow.join(timeout=5)
    assert rig.pump.running(SESSION) == 2


def test_replay_keys_of_retained_execs_outlive_the_key_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_exec, "MAX_REPLAYS", 3)
    rig = Rig(tmp_path)
    kept, docker = rig.start(request_id="kept")
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    rig.state["Paused"] = True
    for index in range(6):  # failed starts leave keys and no exec
        with pytest.raises(SandboxError):
            rig.start(request_id=f"busy-{index}", started=False)
    rig.state["Paused"] = False

    assert len(rig.pump._replays) == 3
    assert rig.start(request_id="kept", started=False)[0] == kept
    assert len(rig.api.created) == 1
    with pytest.raises(SandboxError) as caught:  # the newest failure still replays
        rig.start(request_id="busy-5", started=False)
    assert caught.value.code == "busy"


# -- reads and retention -----------------------------------------------------------


def test_reads_are_bounded_and_serve_stdout_before_stderr(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.emit(docker, 1, b"0123456789")
    rig.emit(docker, 2, b"abcdef")

    first = rig.read(exec_id, max_bytes=8)
    assert outputs(first) == (b"01234567", b"")
    second = rig.read(exec_id, 8, 0, max_bytes=8)
    assert outputs(second) == (b"89", b"abcdef")
    assert (second["stdout_offset"], second["stderr_offset"]) == (10, 6)
    for offsets, field in (((11, 0), "stdout_offset"), ((0, 7), "stderr_offset")):
        with pytest.raises(SandboxError) as caught:
            rig.read(exec_id, *offsets)
        assert (caught.value.code, caught.value.field) == ("invalid", field)
    for size in (0, MAX_READ_BYTES + 1, 1.5):
        with pytest.raises(SandboxError) as caught:
            rig.read(exec_id, max_bytes=size)
        assert caught.value.field == "max_bytes"
    with pytest.raises(SandboxError) as caught:
        rig.read(exec_id, -1, 0)
    assert caught.value.field == "stdout_offset"
    with pytest.raises(SandboxError) as caught:
        rig.read(exec_id, session="judge-session")
    assert caught.value.code == "permission"


def test_output_read_to_the_end_is_released_after_a_grace(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.emit(docker, 1, b"result\n")
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    view = rig.read(exec_id)
    assert outputs(view)[0] == b"result\n"

    rig.turn(RELEASE_GRACE_SEC - 1)
    assert rig.read(exec_id)["stdout_total"] == 7  # a retried read still works
    rig.turn(1.0 + sandbox_exec.PURGE_SEC)

    assert not rig.exec_dir(exec_id).exists()
    final = rig.read(exec_id, 7, 0)  # the state stays readable at the end
    assert (final["state"], final["exit_code"], final["stdout_b64"]) == (
        "exited",
        0,
        "",
    )
    with pytest.raises(SandboxError) as caught:
        rig.read(exec_id)
    assert (caught.value.code, caught.value.field) == ("invalid", "stdout_offset")


def test_unread_output_is_kept_ten_minutes_after_exit(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.emit(docker, 1, b"never read\n")
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    rig.clock.now += 600.0 - 2
    rig.turn(1.0)
    assert (rig.exec_dir(exec_id) / "stdout").read_bytes() == b"never read\n"
    rig.clock.now += 2
    rig.turn()
    assert not rig.exec_dir(exec_id).exists()
    with pytest.raises(SandboxError) as caught:
        rig.read(exec_id)
    assert caught.value.code == "permission"
    # The key still names the exec: a late retry never runs it again.
    assert rig.start(request_id="r1", started=False)[0] == exec_id


def test_finished_records_are_bounded_per_session(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_exec, "MAX_RETAINED_EXECS", 2)
    rig = Rig(tmp_path)
    ids = []
    for index in range(3):
        exec_id, docker = rig.start(request_id=f"r{index}")
        rig.exit(docker, 0)
        rig.turn(TICK_SEC)
        ids.append(exec_id)
    with pytest.raises(SandboxError):
        rig.read(ids[0])
    assert [rig.read(exec_id)["state"] for exec_id in ids[1:]] == ["exited"] * 2
    assert not rig.exec_dir(ids[0]).exists()


def test_env_stop_ends_its_execs_and_destroy_drops_their_output(tmp_path):
    rig = Rig(tmp_path)
    done, docker = rig.start(request_id="done")
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    running, docker = rig.start(request_id="running")
    rig.emit(docker, 1, b"partial")

    rig.pump.env_stopped(ENV_ID, service="main")
    rig.turn()

    view = rig.read(running)
    assert (view["state"], view["reason"], view["exit_code"]) == (
        "killed",
        "env_stopped",
        None,
    )
    assert outputs(view)[0] == b"partial"
    assert docker.peer.recv(1) == b""
    assert [summary.state for summary in rig.summaries] == ["exited", "killed"]

    rig.pump.env_stopped(ENV_ID, release=True)
    assert list((rig.spool / "x").iterdir()) == []
    # The owner still learns how each exec ended; unread output is gone.
    for exec_id, state, reason, total in (
        (done, "exited", None, 0),
        (running, "killed", "env_stopped", 7),
    ):
        for offset in (0, total):
            view = rig.read(exec_id, offset)
            assert (view["state"], view["reason"], view["stdout_b64"]) == (
                state,
                reason,
                "",
            )
            assert (view["stdout_offset"], view["stdout_total"]) == (total, total)
            assert view["truncated"] is (total > 0)


def test_env_destroy_ends_running_execs_and_keeps_their_final_state(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(deadline=rig.clock.now + 60)
    rig.emit(docker, 2, b"unread\n")
    rig.pump.env_stopped(ENV_ID, release=True)
    rig.turn()
    view = rig.read(exec_id)
    assert (view["state"], view["reason"], view["exit_code"]) == (
        "killed",
        "env_stopped",
        None,
    )
    assert (view["stderr_offset"], view["stderr_b64"], view["truncated"]) == (
        7,
        "",
        True,
    )
    assert [(summary.state, summary.reason) for summary in rig.summaries] == [
        ("killed", "env_stopped")
    ]
    assert list((rig.spool / "x").iterdir()) == []
    assert rig.pump.kill(exec_id, session=SESSION)["delivered"] is False
    with pytest.raises(SandboxError) as caught:
        rig.read(exec_id, session="judge-session")
    assert caught.value.code == "permission"

    rig.clock.now += 599.0
    rig.turn()
    assert rig.read(exec_id)["state"] == "killed"
    rig.clock.now += 1.0
    rig.turn()  # kept 10 min, then dropped
    with pytest.raises(SandboxError) as caught:
        rig.read(exec_id)
    assert caught.value.code == "permission"


def test_other_envs_are_untouched_by_an_env_stop(tmp_path):
    rig = Rig(tmp_path)
    exec_id, _ = rig.start()
    rig.pump.env_stopped("e" + "2" * 32, release=True)
    rig.pump.env_stopped(ENV_ID, service="db")
    rig.turn()
    assert rig.read(exec_id)["state"] == "running"


def test_close_session_releases_its_execs_and_replay_keys(tmp_path):
    rig = Rig(tmp_path)
    judge, _ = rig.start(session="judge", request_id="same")
    work, _ = rig.start(session=SESSION, request_id="same")
    ended, docker = rig.start(session="judge", request_id="ended")
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    rig.pump.close_session("judge")
    rig.turn()
    for exec_id in (judge, ended):
        with pytest.raises(SandboxError) as caught:
            rig.read(exec_id, session="judge")
        assert caught.value.code == "permission"
    assert rig.read(work)["state"] == "running"
    assert rig.pump.running("judge") == 0
    assert list((rig.spool / "x").iterdir()) == [rig.exec_dir(work)]
    # The keys went too: the same request is a new exec, never a replay.
    again, _ = rig.start(session="judge", request_id="same")
    assert again not in (judge, work) and len(rig.api.created) == 4


# -- freeze ------------------------------------------------------------------------


def test_frozen_sessions_hold_their_deadlines_and_refuse_start_and_kill(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start(timeout_sec=2)
    judge, _ = rig.start(timeout_sec=2, session="judge", request_id="j")
    rig.turn(1.0)
    rig.pump.freeze(SESSION)
    rig.turn(10.0)

    # Only the Judge exec timed out; the frozen Work exec saw no signal.
    assert {call[0] for call in rig.killer.calls} == {rig.api.last().pid}
    with pytest.raises(SandboxError) as caught:
        rig.start(request_id="r2", started=False)
    assert (caught.value.code, caught.value.field) == ("busy", "session")
    with pytest.raises(SandboxError) as caught:
        rig.pump.kill(exec_id, session=SESSION)
    assert caught.value.code == "busy"

    rig.pump.thaw(SESSION)
    rig.killer.calls.clear()
    rig.turn(1.0 - TICK_SEC)
    assert rig.killer.calls == []
    rig.turn(2 * TICK_SEC)
    assert rig.killer.calls == [(docker.pid, signal.SIGTERM, True)]


def test_a_freeze_stops_the_retention_clocks_of_ended_execs(tmp_path):
    rig = Rig(tmp_path)
    read, docker = rig.start(request_id="read")
    rig.emit(docker, 1, b"result\n")
    rig.exit(docker, 0)
    unread, docker = rig.start(request_id="unread")
    rig.emit(docker, 1, b"later\n")
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    assert outputs(rig.read(read))[0] == b"result\n"  # its release grace starts

    rig.pump.freeze(SESSION)  # a Judge round with a 900 s verifier
    rig.clock.now += 900.0
    rig.turn(1.0)
    rig.pump.thaw(SESSION)
    rig.turn(1.0)

    # Reads lost or retried across the round still work after resume.
    assert outputs(rig.read(read))[0] == b"result\n"
    assert outputs(rig.read(unread))[0] == b"later\n"
    rig.clock.now += 600.0
    rig.turn()
    with pytest.raises(SandboxError) as caught:  # the clocks run again
        rig.read(unread)
    assert caught.value.code == "permission"


def test_an_exit_seen_during_a_freeze_drains_from_the_thaw(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    rig.turn(TICK_SEC)
    rig.pump.freeze(SESSION)
    rig.clock.now += 50.0
    rig.exit(docker, 0, close=False)  # a child keeps the stream open
    rig.pump._inspect(rig.pump._records[exec_id])  # a call in flight at freeze
    rig.clock.now += 50.0
    rig.pump.thaw(SESSION)
    rig.turn(1.0 + TICK_SEC)
    assert rig.read(exec_id)["state"] == "exited"  # drained 1 s after the thaw


def test_a_blocked_engine_call_never_delays_another_execs_timeout(tmp_path):
    host = FakeHost(tmp_path / "host")
    api = FakeApi()
    killer = RecordingKiller()
    signalled = {}
    killer.effect = lambda process, signum, group: signalled.setdefault(
        process.host_pid, time.monotonic()
    )
    pump = ExecPump(api, tmp_path / "spool", killer=killer, table=host.table)
    target = ExecTarget(
        CONTAINER,
        lambda: {"Id": CONTAINER, "State": {"Running": True, "Pid": INIT_PID}},
    )
    blocked, release = threading.Event(), threading.Event()
    original = api.exec_inspect

    def inspect(exec_id):
        record = api.execs[exec_id]
        if record.pid == 200 and not record.running:
            # The Engine holds an exited exec's lock while a background
            # process keeps its stream open (VERIFIED: 1.8 s).
            blocked.set()
            release.wait(3.0)
        return original(exec_id)

    api.exec_inspect = inspect

    def start(argv, pid, **fields):
        request = parse_exec_request({"service": "main", "argv": argv, **fields})
        exec_id = pump.start(
            target,
            request,
            session=SESSION,
            env_id=ENV_ID,
            request_id=f"r{pid}",
            output_limit=1024,
        )
        docker = api.last()
        docker.pid = pid
        host.add(pid, start=5000 + pid)
        return exec_id, docker

    try:
        _, held = start(["sh", "-c", "sleep 30 & echo started"], 200)
        held.running, held.exit_code = False, 0
        host.remove(200)  # the leader exited; a child keeps the stream
        assert blocked.wait(5)
        began = time.monotonic()
        exec_id, _ = start(["sleep", "300"], 210, timeout_sec=0.5)

        deadline = time.monotonic() + 5
        while 210 not in signalled and time.monotonic() < deadline:
            time.sleep(0.01)
        # On time, although the other exec's inspect stays blocked for 3 s.
        assert signalled[210] - began < 0.5 + 0.3
        assert killer.calls[0] == (210, signal.SIGTERM, True)
    finally:
        release.set()
        pump.close()


def test_generation_changes_with_output_and_completion(tmp_path):
    rig = Rig(tmp_path)
    exec_id, docker = rig.start()
    before = (rig.pump.generation(), rig.pump.generation(exec_id))
    rig.emit(docker, 1, b"x")
    middle = (rig.pump.generation(), rig.pump.generation(exec_id))
    rig.exit(docker, 0)
    rig.turn(TICK_SEC)
    after = rig.pump.generation(exec_id)
    assert before[0] < middle[0] and before[1] < middle[1] < after
    assert rig.pump.generation("x" + "0" * 32) == 0


def test_threads_wake_waiters_and_close_removes_the_spool(tmp_path):
    host = FakeHost(tmp_path / "host")
    api = FakeApi()
    finished = []
    pump = ExecPump(
        api,
        tmp_path / "spool",
        killer=RecordingKiller(),
        table=host.table,
        on_finish=finished.append,
    )
    target = ExecTarget(
        CONTAINER, lambda: {"Id": CONTAINER, "State": {"Running": True, "Pid": 100}}
    )
    try:
        request = parse_exec_request({"service": "main", "argv": ["true"]})
        exec_id = pump.start(
            target,
            request,
            session=SESSION,
            env_id=ENV_ID,
            request_id="r1",
            output_limit=1024,
        )
        docker = api.last()
        docker.pid = 200
        host.add(200)
        timer = threading.Timer(0.2, lambda: docker.peer.sendall(frame(1, b"hi\n")))
        timer.start()
        began = time.monotonic()
        view = pump.wait(
            exec_id, session=SESSION, stdout_offset=0, stderr_offset=0, wait_sec=10
        )
        assert outputs(view)[0] == b"hi\n" and time.monotonic() - began < 5

        docker.running, docker.exit_code = False, 0
        docker.peer.close()
        view = pump.wait(
            exec_id, session=SESSION, stdout_offset=3, stderr_offset=0, wait_sec=10
        )
        assert (view["state"], view["exit_code"]) == ("exited", 0)
        deadline = time.monotonic() + 5
        while not finished and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [summary.state for summary in finished] == ["exited"]
        with pytest.raises(SandboxError):
            pump.wait(
                exec_id, session=SESSION, stdout_offset=0, stderr_offset=0, wait_sec=31
            )
    finally:
        pump.close()
    assert not (tmp_path / "spool" / "x").exists()
    assert all(not thread.is_alive() for thread in pump._threads)


def test_close_wakes_blocked_waiters(tmp_path):
    rig = Rig(tmp_path)
    exec_id, _ = rig.start()
    errors = []

    def blocked():
        try:
            rig.pump.wait(
                exec_id, session=SESSION, stdout_offset=0, stderr_offset=0, wait_sec=30
            )
        except SandboxError as error:
            errors.append(error.code)

    waiter = threading.Thread(target=blocked)
    waiter.start()
    time.sleep(0.1)
    began = time.monotonic()
    rig.pump.close()
    waiter.join(timeout=5)
    assert errors == ["permission"] and time.monotonic() - began < 5


def test_the_spool_must_be_broker_owned(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox_exec.os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(InfrastructureError):
        ExecPump(
            FakeApi(), tmp_path / "spool", killer=RecordingKiller(), start_threads=False
        )


def test_a_new_pump_removes_what_an_earlier_broker_left_in_its_spool(tmp_path):
    first = Rig(tmp_path)
    exec_id, docker = first.start()
    first.emit(docker, 1, b"left behind\n")  # the broker dies here
    spool = tmp_path / "spool" / "x"
    (spool / "stray").write_bytes(b"x")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "keep").write_bytes(b"k")
    (spool / "link").symlink_to(tmp_path / "elsewhere")

    ExecPump(
        FakeApi(), tmp_path / "spool", killer=RecordingKiller(), start_threads=False
    )

    assert list(spool.iterdir()) == []
    assert (tmp_path / "elsewhere" / "keep").read_bytes() == b"k"  # never followed


def test_the_spool_never_follows_a_link(tmp_path):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool" / "x").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError):
        ExecPump(
            FakeApi(), tmp_path / "spool", killer=RecordingKiller(), start_threads=False
        )


# -- host process views and killers ------------------------------------------------


def process(host, pid, *, container=CONTAINER):
    found = host.table.process(container, host.scope, pid)
    assert found is not None
    return found


def test_process_table_reads_proc_and_cgroup_views(tmp_path):
    host = FakeHost(tmp_path)
    host.add(300, start=777, ns=(9, 9), comm="odd ) name (x")
    assert host.table.stat(300) == sandbox_exec.ProcStat("S", 300, 777)
    assert host.table.namespaced(300) == (9, 9)
    assert host.table.scope(CONTAINER, INIT_PID) == host.scope
    assert host.table.members(host.scope) == (INIT_PID, 300)
    host.set_oom(5)
    assert host.table.oom_kills(host.scope) == 5
    assert process(host, 300) == ExecProcess(CONTAINER, host.scope, 300, 300, 777, 9, 9)
    for pid in (0, None, "100"):
        with pytest.raises(InfrastructureError):
            host.table.scope(CONTAINER, pid)
    with pytest.raises(InfrastructureError):
        host.table.scope("d" * 64, INIT_PID)  # the scope must name the container
    assert host.table.process(CONTAINER, host.scope, 999) is None
    other = host.proc / "301"
    host.add(301)
    (other / "cgroup").write_text("0::/user.slice/session-1.scope\n")
    assert host.table.process(CONTAINER, host.scope, 301) is None


def test_targets_are_live_members_of_the_group_only(tmp_path):
    host = FakeHost(tmp_path)
    host.add(200, start=5000)
    host.add(201, pgrp=200, start=5001)
    host.add(202, pgrp=200, start=5002, state="Z")  # a zombie cannot be signalled
    host.add(203, pgrp=200, start=4000)  # older than its leader: not the group
    host.add(204, start=5003)  # its own group (setsid): survives by design
    leader = process(host, 200)
    assert host.table.targets(leader, group=True) == ((200, 5000), (201, 5001))
    assert host.table.targets(leader, group=False) == ((200, 5000),)
    host.remove(200)  # the leader exited; its group lives on
    assert host.table.targets(leader, group=True) == ((201, 5001),)
    assert host.table.targets(leader, group=False) == ()
    host.add(200, start=9000)  # the leader PID was reused: the group is gone
    assert host.table.targets(leader, group=True) == ()


class FakePidfds:
    def __init__(self, host):
        self.host = host
        self.opened = []
        self.sent = []
        self.closed = []
        self.on_open = None
        self.kills = True

    def open(self, pid):
        if not (self.host.proc / str(pid)).exists():
            raise ProcessLookupError(pid)
        if self.on_open is not None:
            self.on_open(pid)
        self.opened.append(pid)
        return 1000 + pid

    def send(self, descriptor, signum):
        pid = descriptor - 1000
        self.sent.append((pid, signum))
        if self.kills and signum == signal.SIGKILL:
            self.host.remove(pid)

    def close(self, descriptor):
        self.closed.append(descriptor)

    def killer(self):
        return HostPidfdKiller(
            self.host.table,
            pidfd_open=self.open,
            send_signal=self.send,
            close=self.close,
        )


def test_pidfd_killer_signals_every_live_member_through_a_pinned_pidfd(tmp_path):
    host = FakeHost(tmp_path)
    host.add(200, start=5000)
    host.add(201, pgrp=200, start=5001)
    host.add(204, start=5003)
    pidfds = FakePidfds(host)

    assert pidfds.killer().signal(process(host, 200), signal.SIGTERM, group=True)

    assert pidfds.sent == [(200, signal.SIGTERM), (201, signal.SIGTERM)]
    assert sorted(pidfds.closed) == [1200, 1201]
    assert (host.proc / "204").exists()
    pidfds.sent.clear()
    assert pidfds.killer().signal(process(host, 200), signal.SIGINT, group=False)
    assert pidfds.sent == [(200, signal.SIGINT)]


def test_pidfd_killer_never_signals_a_reused_pid(tmp_path):
    host = FakeHost(tmp_path)
    host.add(200, start=5000)
    host.add(201, pgrp=200, start=5001)
    leader = process(host, 200)
    pidfds = FakePidfds(host)

    # 201 exits and its PID is reused between the listing and the pin.
    pidfds.on_open = lambda pid: pid == 201 and host.add(201, pgrp=201, start=9999)
    assert pidfds.killer().signal(leader, signal.SIGTERM, group=True)
    assert pidfds.sent == [(200, signal.SIGTERM)]
    assert sorted(pidfds.closed) == [1200, 1201]

    pidfds.on_open = None
    host.add(200, start=9000)  # the leader PID itself was reused
    pidfds.sent.clear()
    assert pidfds.killer().signal(leader, signal.SIGKILL, group=True) is False
    assert pidfds.sent == []


def test_pidfd_killer_repeats_kill_for_members_forked_meanwhile(tmp_path):
    host = FakeHost(tmp_path)
    host.add(200, start=5000)
    pidfds = FakePidfds(host)
    forked = []

    def send(descriptor, signum):
        pidfds.send(descriptor, signum)
        if not forked:  # the leader forked right before it died
            forked.append(210)
            host.add(210, pgrp=200, start=5010)

    killer = HostPidfdKiller(
        host.table, pidfd_open=pidfds.open, send_signal=send, close=pidfds.close
    )
    assert killer.signal(process(host, 200), signal.SIGKILL, group=True)
    assert pidfds.sent == [(200, signal.SIGKILL), (210, signal.SIGKILL)]

    pidfds.kills = False  # nothing dies: at most three passes
    host.add(220, start=6000)
    pidfds.sent.clear()
    killer = pidfds.killer()
    assert killer.signal(process(host, 220), signal.SIGKILL, group=True)
    assert pidfds.sent == [(220, signal.SIGKILL)] * 3


def test_pidfd_killer_reports_refusals_as_not_delivered(tmp_path, caplog):
    host = FakeHost(tmp_path)
    host.add(200, start=5000)

    def refuse(descriptor, signum):
        raise PermissionError(1, "Operation not permitted")

    killer = HostPidfdKiller(
        host.table, pidfd_open=lambda pid: 7, send_signal=refuse, close=lambda fd: None
    )
    with caplog.at_level(logging.WARNING):
        assert killer.signal(process(host, 200), signal.SIGTERM, group=True) is False
    assert "cannot signal" in caplog.text


@pytest.mark.parametrize("branch", ["python", "libc", "syscall"])
def test_pidfd_calls_kill_a_real_process_group_from_its_cgroup(monkeypatch, branch):
    """The root path end to end on the real kernel, against our own group
    (no root needed): cgroup.procs, /proc pins and pidfd_send_signal, through
    each of Python's calls, libc's wrappers and the raw system calls."""
    native = hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal")
    libc = ctypes.CDLL(None, use_errno=True)
    if branch == "python" and not native:
        pytest.skip("this Python lacks os.pidfd_open")
    if branch == "libc" and not hasattr(libc, "pidfd_open"):
        pytest.skip("this libc lacks pidfd_open")
    if branch != "python":
        monkeypatch.delattr(os, "pidfd_open", raising=False)
        monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    if branch == "syscall":
        only_syscall = SimpleNamespace(syscall=libc.syscall)
        monkeypatch.setattr(sandbox_exec.ctypes, "CDLL", lambda *a, **k: only_syscall)
    opened, sent = pidfd_calls()
    assert opened.__name__ == ("raw_open" if branch == "syscall" else "pidfd_open")
    table = ProcessTable()
    child = subprocess.Popen(
        ["sh", "-c", "sleep 60 & sleep 61 & wait"], start_new_session=True
    )
    try:
        scope = table.cgroup(child.pid)
        if scope is None or not (scope / "cgroup.procs").exists():
            pytest.skip("cgroup v2 is unavailable")
        leader = table.process("host", scope, child.pid)
        deadline = time.monotonic() + 5
        while len(table.targets(leader, group=True)) < 3:
            assert time.monotonic() < deadline, "the group never formed"
            time.sleep(0.01)

        killer = HostPidfdKiller(table, pidfd_open=opened, send_signal=sent)
        assert killer.signal(leader, signal.SIGTERM, group=True)

        assert child.wait(timeout=5) == -signal.SIGTERM
        while table.targets(leader, group=True):
            assert time.monotonic() < deadline, "a member survived"
            time.sleep(0.01)
        with pytest.raises(ProcessLookupError):
            opened(child.pid)  # reaped: errno ESRCH
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


class KillApi:
    def __init__(self, exit_code=0, *, running_polls=0, error=None):
        self.calls = []
        self.exit_code = exit_code
        self.running_polls = running_polls
        self.error = error

    def exec_create(self, container, cmd, **options):
        if self.error is not None:
            raise self.error
        self.calls.append(("create", container, cmd, options))
        return {"Id": "k" * 64}

    def exec_start(self, exec_id, **options):
        self.calls.append(("start", exec_id, options))
        return b""

    def exec_inspect(self, exec_id):
        if self.running_polls:
            self.running_polls -= 1
            return {"Running": True, "ExitCode": None}
        return {"Running": False, "ExitCode": self.exit_code}


def in_container(host, api):
    clock = FakeClock()

    def sleep(seconds):
        clock.now += seconds

    return InContainerKiller(api, host.table, clock=clock, sleep=sleep, timeout=1.0)


def test_in_container_killer_runs_a_detached_group_kill_as_root(tmp_path):
    host = FakeHost(tmp_path)
    host.add(200, start=5000, ns=(8, 8))
    api = KillApi()

    assert in_container(host, api).signal(
        process(host, 200), signal.SIGTERM, group=True
    )

    assert api.calls == [
        (
            "create",
            CONTAINER,
            ["/bin/sh", "-c", "kill -s TERM -8"],
            {
                "stdout": False,
                "stderr": False,
                "stdin": False,
                "tty": False,
                "user": "0",
                "workdir": "/",
            },
        ),
        ("start", "k" * 64, {"detach": True}),
    ]
    api.calls.clear()
    assert in_container(host, api).signal(
        process(host, 200), signal.SIGINT, group=False
    )
    assert api.calls[0][2] == ["/bin/sh", "-c", "kill -s INT 8"]


def test_in_container_killer_never_signals_a_finished_group(tmp_path):
    host = FakeHost(tmp_path)
    host.add(200, start=5000)
    leader = process(host, 200)
    host.remove(200)
    api = KillApi()
    assert in_container(host, api).signal(leader, signal.SIGKILL, group=True) is False
    assert api.calls == []


@pytest.mark.parametrize(
    "api",
    [
        KillApi(exit_code=1),  # no such group, or no /bin/sh
        KillApi(running_polls=1000),  # an untrusted /bin/sh that never returns
        KillApi(error=daemon_error("container is paused", 409)),
        KillApi(error=OSError("daemon unreachable")),
    ],
)
def test_in_container_killer_failures_are_not_delivered(tmp_path, api):
    host = FakeHost(tmp_path)
    host.add(200, start=5000)
    assert (
        in_container(host, api).signal(process(host, 200), signal.SIGTERM, group=True)
        is False
    )


def test_the_host_pidfd_path_is_the_root_default():
    assert isinstance(default_exec_killer(object(), euid=0), HostPidfdKiller)
    assert isinstance(default_exec_killer(object(), euid=1001), InContainerKiller)
