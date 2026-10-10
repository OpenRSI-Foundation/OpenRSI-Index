"""Brokered execs in env services: one pump, bounded spools, group kills.

The broker owns every exec (spec 3.3). ``start`` returns an ``x<32hex>``
handle at once; one IO thread multiplexes every attach socket with
``selectors``, demultiplexes the Engine's 8-byte frames and spools each
stream to ``<spool>/x/<exec_id>/{stdout,stderr}`` (0700 directories, 0600
files, never opened through a link) up to ``output_limit`` bytes per stream.
Output past the cap is still read and discarded (``truncated``), so a chatty
process never stalls on a full pipe; nothing is buffered beyond one 64 KiB
read. Reads return at most 1 MiB from byte offsets.

A control thread owns time and never waits: Engine and killer calls run on
call threads, at most one inspect and one signal in flight per exec, so a
call blocked for one exec (the Engine holds an exec's lock for up to 2 s
after its exit while a background process keeps its stream open, VERIFIED)
never delays another exec's timeout, and a slow daemon never stalls the
streams. An exec is complete once ``exec_inspect`` reports Running=false and
its stream reached EOF, or after a drain: 1 s after the exit or the last
bytes, at most 3 s after the exit (the Engine closes the streams itself 2 s
after an exit, so bytes still arriving then are output the pump is behind
on). A stream cut by the drain marks the output ``truncated``.
A timeout (or the env deadline, when ``timeout_sec`` is null) signals the
exec's process group, TERM and KILL 2 s later, never the service, and the
exec ends with its whole group; a command whose leader already exited is
never signalled for the stream it left open (a pinned leader's exit is read
from ``/proc``, otherwise an inspect must see the command still running).
The Engine reports one container's exits in order, each behind that exec's
stream wait, so in a container where another exec just left a stream open
an exit is reported up to 2 s late (VERIFIED); signals are never late.
OOM is the delta of the container cgroup's ``memory.events`` oom_kill over
the exec's lifetime, never the sticky ``State.OOMKilled`` (VERIFIED); execs
running concurrently in one service share that window.

Signals reach the group through an ``ExecKiller``. As root the host signals
the group's members found in the container's ``cgroup.procs`` through pidfds
(``HostPidfdKiller``); nothing runs in the child. Otherwise ``/bin/sh -c
'kill -s SIG -<pgid>'`` runs as uid 0 inside the container
(``InContainerKiller``, dev and tests). A docker exec leader is a session
and group leader and env services run docker-init as PID 1, so a killed tree
is reaped (VERIFIED). A process that left the group (setsid, daemons)
survives, like any service the command started.

Output is kept until it was read to the end (after a short grace, so a
retried read still works), 10 min after the exec ended, or the env is
destroyed; the final state stays readable for those 10 min in every case
but a revoked session. The retention clocks of a frozen session stand
still. Execs create no host resources and are never journaled; the pump
owning a spool removes what an earlier broker process left there.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import itertools
import json
import logging
import math
import os
import posixpath
import queue
import re
import secrets
import selectors
import signal as signal_module
import socket
import stat
import struct
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Self

from docker.errors import APIError, DockerException, NotFound
from pydantic import Field, ValidationError, field_validator, model_validator
from pydantic_core import PydanticCustomError

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.sandbox_contracts import (
    MAX_EXECS_RUNNING,
    SandboxError,
    SandboxModel,
    Seconds,
    absolute_path,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    ENV_COMMAND_BYTES,
    ENV_QUOTA_ERROR,
    EnvKey,
    FrozenDict,
    FrozenMap,
    ServiceName,
    Text,
    UserName,
)

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
EXEC_ID = re.compile(r"^x[0-9a-f]{32}$")
REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_READ_BYTES = MIB
MAX_WAIT_SEC = 30.0
READ_CHUNK = 64 * 1024
# Reads per readable socket per turn: one chatty exec cannot starve others.
READS_PER_TURN = 4
# Reads of what is already buffered when a stream is closed (1 MiB).
FINAL_READS = 16
TICK_SEC = 0.05
INSPECT_SEC = 0.5
DRAIN_SEC = 1.0
DRAIN_MAX_SEC = 3.0
KILL_GRACE_SEC = 2.0
# A leader still running this long after KILL is abandoned, never waited on.
ABANDON_SEC = 2.0
RETAIN_SEC = 600.0
RELEASE_GRACE_SEC = 30.0
PURGE_SEC = 1.0
MAX_INSPECT_FAILURES = 20
# Bounded broker memory: finished records per session, replay keys per pump.
MAX_RETAINED_EXECS = 1024
MAX_REPLAYS = 8192
# Work (paused during a round) plus one Judge round, each at the grant cap:
# 1024 execs, each holding its stream socket and two spool descriptors, and
# at most 2048 call threads (_Calls), started only as calls overlap.
MAX_RUNNING_EXECS = 2 * MAX_EXECS_RUNNING
KILL_EXEC_SEC = 5.0
# How long start() waits for the Engine to report the exec's process. The
# hijacked stream is up before runc starts it, and runc refuses to start an
# exec in a paused container, so a freeze right after exec_start would fail
# a command the caller was told is running.
START_CONFIRM_SEC = 5.0
KILL_PASSES = 3
# Unified Linux system call numbers (5.1+ and 5.3+ on every architecture).
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434
SIGNALS = {
    "INT": signal_module.SIGINT,
    "TERM": signal_module.SIGTERM,
    "HUP": signal_module.SIGHUP,
    "KILL": signal_module.SIGKILL,
}
_SIGNAL_NAMES = {number: name for name, number in SIGNALS.items()}
_FRAME = struct.Struct("!BxxxI")
_STREAMS = ("stdout", "stderr")
_OFFSETS = ("stdout_offset", "stderr_offset")

ExecState = Literal["running", "exited", "timed_out", "killed", "lost"]
ExecReason = Literal["timeout", "interrupt", "env_stopped", "deadline"]


# -- request -----------------------------------------------------------------------


class ExecRequest(SandboxModel):
    """exec_start fields besides the env handle and request_id."""

    service: ServiceName
    argv: tuple[Text, ...] = Field(min_length=1)
    cwd: str | None = None
    env: FrozenDict[EnvKey, Text] = Field(default_factory=FrozenMap)
    user: UserName | None = None
    # None: until the env deadline.
    timeout_sec: Seconds | None = None
    merge_stderr: bool = False

    @field_validator("cwd")
    @classmethod
    def absolute_cwd(cls, value: str | None) -> str | None:
        return None if value is None else absolute_path(value)

    @model_validator(mode="after")
    def bounded_command(self) -> Self:
        size = sum(len(item.encode()) for item in self.argv) + sum(
            len(key.encode()) + len(value.encode()) for key, value in self.env.items()
        )
        if size > ENV_COMMAND_BYTES:
            # A quota, as for a v1 exec and a service command.
            raise PydanticCustomError(ENV_QUOTA_ERROR, "argv and env exceed 64 KiB")
        return self


def _request_error(error: ValueError) -> SandboxError:
    # Never echo input values: argv and env may carry task secrets.
    if isinstance(error, ValidationError):
        errors = error.errors(include_input=False, include_url=False)
        first = next(
            (item for item in errors if item["type"] != ENV_QUOTA_ERROR), errors[0]
        )
        code = "quota" if first["type"] == ENV_QUOTA_ERROR else "invalid"
        name = ".".join(str(part) for part in first["loc"]) or "argv"
        return SandboxError(code, name[:256], first["msg"][:1024])
    return SandboxError("invalid", "request", "expected a JSON exec request object")


def parse_exec_request(raw: object) -> ExecRequest:
    """Validate untrusted exec_start fields; JSON mode keeps arrays strict."""
    try:
        return ExecRequest.model_validate_json(json.dumps(raw, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise _request_error(error) from None


def _fingerprint(env_id: str, request: ExecRequest) -> str:
    canonical = json.dumps(
        {"env_id": env_id, **request.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


# -- host process views ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecTarget:
    """One owned env service container as exec sees it.

    ``inspect`` returns identity-attested inspect attributes, or None once
    the container is gone (``SandboxEnvDockerBackend.exec_target``).
    """

    container_id: str
    inspect: Callable[[], Mapping[str, Any] | None]


@dataclass(frozen=True, slots=True)
class ProcStat:
    state: str
    pgrp: int
    start_ticks: int


@dataclass(frozen=True, slots=True)
class ExecProcess:
    """An exec's leader on the host and in the container, pinned by start time."""

    container_id: str
    scope: Path
    host_pid: int
    host_pgid: int
    start_ticks: int
    ns_pid: int
    ns_pgid: int


class ProcessTable:
    """Host ``/proc`` and cgroup v2 reads about env services.

    All of them work as non-root (VERIFIED: stat, status, cgroup,
    ``cgroup.procs`` and ``memory.events`` are world-readable); only
    signalling needs root. The roots are injectable for tests.
    """

    def __init__(
        self,
        *,
        proc_root: Path = Path("/proc"),
        cgroup_root: Path = Path("/sys/fs/cgroup"),
    ) -> None:
        self._proc_root = Path(proc_root)
        self._cgroup_root = Path(cgroup_root)

    def _unified(self, pid: int) -> str | None:
        try:
            lines = (self._proc_root / str(pid) / "cgroup").read_text().splitlines()
        except OSError:
            return None
        unified = [line[3:] for line in lines if line.startswith("0::")]
        return unified[0] if len(unified) == 1 else None

    def scope(self, container_id: str, pid: object) -> Path:
        """The container's cgroup directory, read from its init process."""
        if type(pid) is not int or pid <= 0:
            raise InfrastructureError("sandbox exec service has no init PID")
        path = self._unified(pid)
        if (
            path is None
            or not path.startswith("/")
            or posixpath.normpath(path) != path
            or container_id not in path.rsplit("/", 1)[-1]
        ):
            raise InfrastructureError(
                "sandbox exec service is not in its own cgroup v2 scope"
            )
        return self._cgroup_root / path.lstrip("/")

    def cgroup(self, pid: int) -> Path | None:
        path = self._unified(pid)
        return None if path is None else self._cgroup_root / path.lstrip("/")

    def stat(self, pid: int) -> ProcStat | None:
        try:
            text = (self._proc_root / str(pid) / "stat").read_text()
            # comm may hold spaces and parentheses; fields follow the last ")".
            fields = text[text.rindex(")") + 2 :].split()
            return ProcStat(fields[0], int(fields[2]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            return None

    def namespaced(self, pid: int) -> tuple[int, int] | None:
        """The innermost (NSpid, NSpgid): the IDs inside the container."""
        try:
            lines = (self._proc_root / str(pid) / "status").read_text().splitlines()
        except OSError:
            return None
        values = {}
        for line in lines:
            key, _, rest = line.partition(":")
            if key in ("NSpid", "NSpgid") and rest.split():
                try:
                    values[key] = int(rest.split()[-1])
                except ValueError:
                    return None
        if set(values) != {"NSpid", "NSpgid"}:
            return None
        return values["NSpid"], values["NSpgid"]

    def members(self, scope: Path) -> tuple[int, ...]:
        try:
            text = (scope / "cgroup.procs").read_text()
        except FileNotFoundError:
            return ()  # the service is gone, and every task with it
        except OSError as error:
            LOGGER.warning("cannot list sandbox exec service tasks: %s", error)
            return ()
        return tuple(int(item) for item in text.split() if item.isdigit())

    def oom_kills(self, scope: Path) -> int | None:
        try:
            lines = (scope / "memory.events").read_text().splitlines()
        except OSError:
            return None
        for line in lines:
            key, _, value = line.partition(" ")
            if key == "oom_kill" and value.strip().isdigit():
                return int(value)
        return None

    def process(
        self, container_id: str, scope: Path, host_pid: int
    ) -> ExecProcess | None:
        """The leader, if it still runs in ``scope``; None otherwise."""
        before = self.stat(host_pid)
        if before is None or self.cgroup(host_pid) != scope:
            return None
        ids = self.namespaced(host_pid)
        after = self.stat(host_pid)
        # Read again: the IDs must belong to the process the stat described.
        if ids is None or after is None or after.start_ticks != before.start_ticks:
            return None
        return ExecProcess(
            container_id=container_id,
            scope=scope,
            host_pid=host_pid,
            host_pgid=before.pgrp,
            start_ticks=before.start_ticks,
            ns_pid=ids[0],
            ns_pgid=ids[1],
        )

    def targets(
        self, process: ExecProcess, *, group: bool
    ) -> tuple[tuple[int, int], ...]:
        """Live ``(pid, start)`` pairs to signal; empty once they are gone.

        A group number is never reused while a member holds it, so a leader
        PID alive with another start time means the group is gone. Members
        must have started no earlier than their leader; zombies are skipped.
        """
        leader = self.stat(process.host_pid)
        if leader is not None and leader.start_ticks != process.start_ticks:
            return ()
        if not group:
            if leader is None or leader.state == "Z":
                return ()
            return ((process.host_pid, process.start_ticks),)
        found = []
        for pid in self.members(process.scope):
            info = leader if pid == process.host_pid else self.stat(pid)
            if (
                info is not None
                and info.state != "Z"
                and info.pgrp == process.host_pgid
                and info.start_ticks >= process.start_ticks
            ):
                found.append((pid, info.start_ticks))
        return tuple(found)


# -- killers -----------------------------------------------------------------------


class ExecKiller(Protocol):
    """Deliver one signal to an exec's process group (or just its leader).

    Returns whether it reached a live process; never raises for a process
    that is already gone.
    """

    def signal(self, process: ExecProcess, signum: int, *, group: bool) -> bool: ...


def _checked(result: int) -> int:
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))  # ESRCH: ProcessLookupError
    return result


def pidfd_calls() -> tuple[Callable[[int], int], Callable[[int, int], None]]:
    """``pidfd_open(pid)`` and ``pidfd_send_signal(fd, signum)``.

    The Anaconda 3.13 build the Harness runs on lacks both
    ``os.pidfd_open`` and ``signal.pidfd_send_signal`` (VERIFIED) although
    the kernel and glibc 2.39 provide them, so libc's wrappers are used,
    or else the raw system calls (the same numbers on every architecture).
    """
    if hasattr(os, "pidfd_open") and hasattr(signal_module, "pidfd_send_signal"):
        return os.pidfd_open, signal_module.pidfd_send_signal
    libc = ctypes.CDLL(None, use_errno=True)
    if hasattr(libc, "pidfd_open") and hasattr(libc, "pidfd_send_signal"):
        open_call, send_call = libc.pidfd_open, libc.pidfd_send_signal
        open_call.argtypes = (ctypes.c_int, ctypes.c_uint)
        send_call.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint,
        )
        open_call.restype = send_call.restype = ctypes.c_int

        def pidfd_open(pid: int) -> int:
            return _checked(open_call(pid, 0))

        def pidfd_send_signal(descriptor: int, signum: int) -> None:
            _checked(send_call(descriptor, signum, None, 0))

        return pidfd_open, pidfd_send_signal
    syscall = libc.syscall
    syscall.restype = ctypes.c_long

    def raw_open(pid: int) -> int:
        return _checked(syscall(_SYS_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0)))

    def raw_send(descriptor: int, signum: int) -> None:
        _checked(
            syscall(
                _SYS_PIDFD_SEND_SIGNAL,
                ctypes.c_int(descriptor),
                ctypes.c_int(signum),
                None,
                ctypes.c_uint(0),
            )
        )

    return raw_open, raw_send


class HostPidfdKiller:
    """Root: signal the group's members from the host through pidfds.

    Members come from the container's ``cgroup.procs``; each is pinned with
    ``pidfd_open`` and its start time read again before
    ``pidfd_send_signal``, so a reused PID is never signalled. KILL repeats
    up to three passes for members forked meanwhile. Nothing runs in the
    child, so images without a shell work alike. The pidfd calls are
    injectable; by default ``pidfd_calls()``.
    """

    def __init__(
        self,
        table: ProcessTable | None = None,
        *,
        pidfd_open: Callable[[int], int] | None = None,
        send_signal: Callable[[int, int], None] | None = None,
        close: Callable[[int], None] = os.close,
    ) -> None:
        if pidfd_open is None or send_signal is None:
            default_open, default_send = pidfd_calls()
            pidfd_open = pidfd_open or default_open
            send_signal = send_signal or default_send
        self._table = table or ProcessTable()
        self._pidfd_open = pidfd_open
        self._send_signal = send_signal
        self._close = close

    def signal(self, process: ExecProcess, signum: int, *, group: bool) -> bool:
        delivered = False
        passes = KILL_PASSES if group and signum == signal_module.SIGKILL else 1
        for _ in range(passes):
            targets = self._table.targets(process, group=group)
            if not targets:
                break
            for pid, start in targets:
                delivered = self._send(pid, start, signum) or delivered
        return delivered

    def _send(self, pid: int, start: int, signum: int) -> bool:
        try:
            descriptor = self._pidfd_open(pid)
        except ProcessLookupError:
            return False
        except OSError as error:
            LOGGER.warning("cannot pin sandbox exec process %s: %s", pid, error)
            return False
        try:
            current = self._table.stat(pid)
            if current is None or current.start_ticks != start:
                return False  # the PID was reused before it was pinned
            self._send_signal(descriptor, signum)
            return True
        except ProcessLookupError:
            return False
        except OSError as error:
            LOGGER.warning("cannot signal sandbox exec process %s: %s", pid, error)
            return False
        finally:
            self._close(descriptor)


class InContainerKiller:
    """Non-root (dev, tests): ``/bin/sh -c 'kill -s SIG -<pgid>'`` as uid 0.

    A non-root broker cannot signal container processes (EPERM, VERIFIED)
    but can read them, so the group is looked up first and a finished group
    is never signalled by number. The helper exec is detached with nothing
    attached (the image's /bin/sh is untrusted) and waited for at most
    ``timeout``; an image without /bin/sh reports not delivered.
    """

    def __init__(
        self,
        api: Any,
        table: ProcessTable | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = KILL_EXEC_SEC,
    ) -> None:
        self._api = api
        self._table = table or ProcessTable()
        self._clock = clock
        self._sleep = sleep
        self._timeout = timeout

    def signal(self, process: ExecProcess, signum: int, *, group: bool) -> bool:
        if not self._table.targets(process, group=group):
            return False
        # Busybox kill rejects "--"; "-s NAME -PGID" works in ash, dash, bash.
        target = f"-{process.ns_pgid}" if group else str(process.ns_pid)
        command = ["/bin/sh", "-c", f"kill -s {_SIGNAL_NAMES[signum]} {target}"]
        try:
            created = self._api.exec_create(
                process.container_id,
                command,
                stdout=False,
                stderr=False,
                stdin=False,
                tty=False,
                user="0",
                workdir="/",
            )
            identity = created["Id"]
            self._api.exec_start(identity, detach=True)
            deadline = self._clock() + self._timeout
            while True:
                state = self._api.exec_inspect(identity)
                code = state.get("ExitCode")
                if not state.get("Running") and type(code) is int:
                    return code == 0
                if self._clock() >= deadline:
                    return False
                self._sleep(0.01)
        except (DockerException, OSError, KeyError, TypeError) as error:
            LOGGER.warning("sandbox exec in-container signal failed: %s", error)
            return False


def default_exec_killer(
    api: Any, *, euid: int | None = None, table: ProcessTable | None = None
) -> ExecKiller:
    """Host pidfds as root (production); the in-container kill otherwise."""
    if (os.geteuid() if euid is None else euid) == 0:
        return HostPidfdKiller(table)
    return InContainerKiller(api, table)


# -- spool -------------------------------------------------------------------------


class _Spool:
    """``<root>/x/<exec_id>/{stdout,stderr}``: 0700 directories, 0600 files.

    One pump owns a spool root. Execs are never journaled, so what an
    earlier broker process of the run left under ``x/`` is removed here.
    """

    def __init__(self, root: Path) -> None:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        parent = os.open(root, flags)
        try:
            self._require_owned(parent)
            try:
                os.mkdir("x", 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            self._dir = os.open("x", flags, dir_fd=parent)
        finally:
            os.close(parent)
        try:
            self._require_owned(self._dir)
            os.fchmod(self._dir, 0o700)
            self._clear()
        except BaseException:
            os.close(self._dir)
            raise
        self.path = root / "x"

    @staticmethod
    def _require_owned(descriptor: int) -> None:
        if os.fstat(descriptor).st_uid != os.geteuid():
            raise InfrastructureError("sandbox exec spool is not broker-owned")

    def _clear(self) -> None:
        """Remove every entry an earlier pump left, never through a link."""
        for name in os.listdir(self._dir):
            try:
                mode = os.stat(name, dir_fd=self._dir, follow_symlinks=False).st_mode
                if not stat.S_ISDIR(mode):
                    os.unlink(name, dir_fd=self._dir)
                    continue
                directory = self._open_dir(name)
                try:
                    for entry in os.listdir(directory):
                        os.unlink(entry, dir_fd=directory)
                finally:
                    os.close(directory)
                os.rmdir(name, dir_fd=self._dir)
            except FileNotFoundError:
                pass
            except OSError as error:
                LOGGER.warning("cannot remove stale sandbox exec output: %s", error)

    def _open_dir(self, exec_id: str) -> int:
        return os.open(
            exec_id,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=self._dir,
        )

    def create(self, exec_id: str) -> list[int]:
        """The two append-only write descriptors of a new exec."""
        os.mkdir(exec_id, 0o700, dir_fd=self._dir)
        descriptors: list[int] = []
        try:
            directory = self._open_dir(exec_id)
            try:
                for name in _STREAMS:
                    descriptors.append(
                        os.open(
                            name,
                            os.O_WRONLY
                            | os.O_CREAT
                            | os.O_EXCL
                            | os.O_APPEND
                            | os.O_NOFOLLOW
                            | os.O_CLOEXEC,
                            0o600,
                            dir_fd=directory,
                        )
                    )
            finally:
                os.close(directory)
        except BaseException:
            for descriptor in descriptors:
                os.close(descriptor)
            self.remove(exec_id)
            raise
        return descriptors

    def open_read(self, exec_id: str, stream: int) -> int:
        directory = self._open_dir(exec_id)
        try:
            return os.open(
                _STREAMS[stream],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=directory,
            )
        finally:
            os.close(directory)

    def remove(self, exec_id: str) -> None:
        try:
            directory = self._open_dir(exec_id)
        except FileNotFoundError:
            return
        try:
            for name in _STREAMS:
                try:
                    os.unlink(name, dir_fd=directory)
                except FileNotFoundError:
                    pass
        finally:
            os.close(directory)
        try:
            os.rmdir(exec_id, dir_fd=self._dir)
        except FileNotFoundError:
            pass

    def close(self) -> None:
        os.close(self._dir)
        try:
            self.path.rmdir()
        except OSError:
            pass


def _write_all(descriptor: int, data: memoryview) -> None:
    while data:
        written = os.write(descriptor, data)
        data = data[written:]


def _pread(descriptor: int, length: int, offset: int) -> bytes:
    parts = []
    while length:
        chunk = os.pread(descriptor, length, offset)
        if not chunk:
            break
        parts.append(chunk)
        length -= len(chunk)
        offset += len(chunk)
    return b"".join(parts)


def _buffered_prefix(handle: Any) -> bytes:
    """Stream bytes http.client already buffered while reading the headers.

    The Engine writes its response head before it starts the process, so a
    buffered header read can pull the first frames into the response's
    reader; reading only the raw socket would lose them. The socket is
    non-blocking here, so this never waits.
    """
    response = getattr(handle, "_response", None)
    reader = getattr(getattr(getattr(response, "raw", None), "_fp", None), "fp", None)
    if reader is None or not hasattr(reader, "peek"):
        return b""
    try:
        data = reader.peek()
        return reader.read(len(data)) if data else b""
    except (OSError, ValueError):
        return b""


def _close_handle(handle: Any) -> None:
    raw = getattr(handle, "_sock", None)
    response = getattr(handle, "_response", None)
    for target in (response, handle, raw):
        close = getattr(target, "close", None)
        if close is None:
            continue
        try:
            close()
        except Exception as error:  # closing must never stop the pump
            LOGGER.debug("closing a sandbox exec stream failed: %s", error)


# -- pump --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecSummary:
    """An ended exec, handed to ``on_finish`` once (live and log settlement)."""

    exec_id: str
    session: Hashable
    env_id: str
    state: ExecState
    reason: ExecReason | None
    stdout_total: int
    stderr_total: int
    output_limit: int
    merge_stderr: bool


@dataclass(slots=True)
class _Replay:
    fingerprint: str
    exec_id: str | None = None
    error: SandboxError | InfrastructureError | None = None
    pending: bool = True


@dataclass(eq=False, slots=True)
class _Exec:
    exec_id: str
    session: Hashable
    env_id: str
    service: str
    target: ExecTarget
    docker_id: str
    scope: Path
    oom_base: int
    merge_stderr: bool
    output_limit: int
    started_at: float
    timeout_at: float | None
    deadline_at: float | None
    next_inspect: float
    # IO thread: the attach stream, write descriptors and frame state.
    sock: Any = None
    handle: Any = None
    fds: list[int] = field(default_factory=lambda: [-1, -1])
    header: bytearray = field(default_factory=bytearray)
    frame_stream: int = 0
    frame_left: int = 0
    totals: list[int] = field(default_factory=lambda: [0, 0])
    truncated: bool = False
    spool_failed: bool = False
    eof: bool = False
    data_at: float | None = None
    # Control thread and its calls: completion, deadlines and kills.
    eof_seen: float | None = None
    host_pid: int | None = None
    process: ExecProcess | None = None
    inspecting: bool = False
    signalling: bool = False
    inspect_failures: int = 0
    # When the last inspect that found the command running was sent.
    alive_at: float | None = None
    exited_at: float | None = None
    exit_code: int | None = None
    oom_kills: int = 0
    expiry: Literal["timeout", "deadline"] | None = None
    kill_step: int = 0
    kill_at: float | None = None
    # Signals exec_kill delivered; a leader dying of one is an interrupt.
    delivered: set[int] = field(default_factory=set)
    # exec_kill calls in flight: their result decides the verdict.
    killing: int = 0
    stopped: bool = False
    signal: str | None = None
    # Under the pump lock: the decided outcome, the final view, retention.
    outcome: tuple[Any, ...] | None = None
    state: ExecState = "running"
    reason: ExecReason | None = None
    expires_at: float | None = None
    release_at: float | None = None
    released: bool = False
    # Output removed with its env before it was read to the end.
    discarded: bool = False
    # Once ended: "output" (env destroyed) or "record" (session revoked).
    release: Literal["output", "record"] | None = None
    generation: int = 0


class _Calls:
    """Engine and killer calls, off the control thread.

    Daemon threads are added on demand and kept for reuse. Each exec has at
    most one inspect and one signal in flight, so below ``limit`` no call
    ever queues behind a blocked one. ``threads=False`` runs every call in
    the caller's turn (tests with a fake clock).
    """

    def __init__(self, limit: int, *, threads: bool) -> None:
        self._limit = limit
        self._enabled = threads
        self._jobs: queue.SimpleQueue[Callable[[], None] | None] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._idle = 0
        self._threads: list[threading.Thread] = []
        self._closed = False

    def submit(self, job: Callable[[], None]) -> None:
        if not self._enabled:
            job()
            return
        with self._lock:
            if self._closed:
                return
            if self._idle:
                self._idle -= 1
            elif len(self._threads) < self._limit:
                thread = threading.Thread(
                    target=self._work, name="rsi-sandbox-exec-call", daemon=True
                )
                self._threads.append(thread)
                thread.start()
        self._jobs.put(job)

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                job()
            except Exception:
                LOGGER.exception("sandbox exec call failed")
            with self._lock:
                self._idle += 1

    def close(self) -> None:
        """Let idle threads end; a call still blocked ends on its own."""
        with self._lock:
            self._closed = True
            count = len(self._threads)
        for _ in range(count):
            self._jobs.put(None)


class ExecPump:
    """Every exec of one broker: start, spool, complete, kill, read, release.

    ``session`` is the hashable identity of the broker session owning an
    exec (for example its ``SandboxSessionCredentials``); reads and kills
    must present an equal one, any other handle is ``permission`` (S2).
    ``deadline`` is the env's own deadline on ``clock``. ``on_finish``
    receives one ``ExecSummary`` per exec that ends while the pump runs, on
    the IO thread and before any reader sees the final state, so the broker
    settles ``execs_running`` and the unused ``max_log_bytes`` reservation
    before a client can start the next exec. It must be quick, and ``close``
    must not be called while holding a lock it takes.

    ``start_threads=False`` leaves ``step_io`` and ``step_control`` to the
    caller, which tests drive with a fake clock, and runs every Engine and
    killer call within the control turn that makes it.
    """

    def __init__(
        self,
        api: Any,
        spool_root: Path,
        *,
        killer: ExecKiller | None = None,
        table: ProcessTable | None = None,
        on_finish: Callable[[ExecSummary], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_running: int = MAX_RUNNING_EXECS,
        start_threads: bool = True,
        start_confirm_sec: float = 0.0,
    ) -> None:
        self._api = api
        self._start_confirm_sec = start_confirm_sec
        self._table = table or ProcessTable()
        self._killer = killer or default_exec_killer(api, table=self._table)
        self._on_finish = on_finish
        self._clock = clock
        self._max_running = max_running
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._records: dict[str, _Exec] = {}
        self._replays: OrderedDict[tuple[Hashable, str], _Replay] = OrderedDict()
        self._frozen: dict[Hashable, float] = {}
        self._launching = 0
        self._commands: deque[tuple[str, _Exec, bytes | None]] = deque()
        self._counter = itertools.count(1)
        self._generation = 0
        self._next_purge = 0.0
        self._closed = False
        self._stop = threading.Event()
        self._control_wake = threading.Event()
        # One inspect and one signal per running exec.
        self._calls = _Calls(2 * max_running, threads=start_threads)
        self._spool = _Spool(Path(spool_root))
        self._selector = selectors.DefaultSelector()
        self._wake_read, self._wake_write = socket.socketpair()
        self._wake_read.setblocking(False)
        self._wake_write.setblocking(False)
        self._selector.register(self._wake_read, selectors.EVENT_READ, None)
        self._view = memoryview(bytearray(READ_CHUNK))
        self._threads: list[threading.Thread] = []
        if start_threads:
            for name, loop in (
                ("rsi-sandbox-exec-pump", self._io_loop),
                ("rsi-sandbox-exec-control", self._control_loop),
            ):
                thread = threading.Thread(target=loop, name=name, daemon=True)
                thread.start()
                self._threads.append(thread)

    # -- start ---------------------------------------------------------------------

    def start(
        self,
        target: ExecTarget,
        request: ExecRequest,
        *,
        session: Hashable,
        env_id: str,
        request_id: str,
        output_limit: int,
        deadline: float | None = None,
    ) -> str:
        """Start one exec and return its handle at once.

        Idempotent per ``(session, request_id)``: the same request returns
        the same exec (running or not), a different one is ``invalid`` and a
        failed start replays its error, so a retried request never runs a
        command twice. ``output_limit`` caps each retained stream.
        """
        if not isinstance(request, ExecRequest):
            raise TypeError("exec start needs a validated ExecRequest")
        if type(request_id) is not str or REQUEST_ID.fullmatch(request_id) is None:
            raise SandboxError(
                "invalid", "request_id", "expected 1..64 safe ASCII characters"
            )
        if type(output_limit) is not int or output_limit <= 0:
            raise SandboxError("invalid", "output_limit", "expected a positive size")
        fingerprint = _fingerprint(env_id, request)
        key = (session, request_id)
        with self._lock:
            if self._closed:
                raise SandboxError("infrastructure", "exec", "exec engine is closed")
            replay = self._replays.get(key)
            if replay is not None:
                return self._replay(replay, fingerprint)
            if session in self._frozen:
                raise SandboxError("busy", "session", "session is frozen")
            running = sum(
                record.state == "running" for record in self._records.values()
            )
            if running + self._launching >= self._max_running:
                raise SandboxError(
                    "quota", "execs_running", "too many running execs in the broker"
                )
            replay = self._replays[key] = _Replay(fingerprint)
            self._launching += 1
            self._trim_replays()
        exec_id = None
        failure: SandboxError | InfrastructureError = SandboxError(
            "infrastructure", "exec", "exec could not be started"
        )
        try:
            exec_id = self._launch(
                target,
                request,
                session=session,
                env_id=env_id,
                output_limit=output_limit,
                deadline=deadline,
            )
        except SandboxError as error:
            failure = error
            raise
        except InfrastructureError as error:
            # Only ExecTarget.inspect raises this (M3's identity-attested
            # inspect): the broker quarantines that env (S8), never the whole
            # broker, and a replay raises it again.
            failure = error
            raise
        except Exception as error:
            raise failure from error
        finally:
            with self._lock:
                self._launching -= 1
                replay.pending = False
                replay.exec_id = exec_id
                if exec_id is None:
                    replay.error = failure
        return exec_id

    @staticmethod
    def _replay(replay: _Replay, fingerprint: str) -> str:
        if replay.fingerprint != fingerprint:
            raise SandboxError(
                "invalid", "request_id", "conflicting idempotency key reuse"
            )
        if replay.pending:
            raise SandboxError("busy", "request_id", "original request still pending")
        error = replay.error
        if isinstance(error, SandboxError):
            raise SandboxError(error.code, error.field, error.message)
        if error is not None:
            raise InfrastructureError(str(error))
        assert replay.exec_id is not None
        return replay.exec_id

    def _trim_replays(self) -> None:
        # Keys of execs still retained stay; the oldest others go first.
        if len(self._replays) <= MAX_REPLAYS:
            return
        for key in list(self._replays):
            if len(self._replays) <= MAX_REPLAYS:
                return
            replay = self._replays[key]
            if not replay.pending and replay.exec_id not in self._records:
                del self._replays[key]

    def _launch(
        self,
        target: ExecTarget,
        request: ExecRequest,
        *,
        session: Hashable,
        env_id: str,
        output_limit: int,
        deadline: float | None,
    ) -> str:
        state = self._running_service(target.inspect())
        scope, oom_base = self._service_scope(target, state.get("Pid"))
        exec_id = "x" + secrets.token_hex(16)
        fds = self._spool.create(exec_id)
        docker_id = handle = None
        try:
            created = self._api.exec_create(
                target.container_id,
                list(request.argv),
                stdout=True,
                stderr=True,
                stdin=False,
                tty=False,
                environment=dict(request.env) or None,
                workdir=request.cwd,
                user=request.user or "",
            )
            docker_id = created["Id"]
            handle = self._api.exec_start(
                docker_id, detach=False, tty=False, socket=True
            )
            sock = getattr(handle, "_sock", handle)
            sock.setblocking(False)
            prefix = _buffered_prefix(handle)
        except BaseException as error:
            if handle is not None:
                _close_handle(handle)
            for descriptor in fds:
                os.close(descriptor)
            self._spool.remove(exec_id)
            if isinstance(error, APIError) and error.status_code is not None:
                if docker_id is None and error.status_code == 404:
                    raise SandboxError(
                        "invalid", "service", "service is absent"
                    ) from None
                if error.status_code == 409:
                    raise SandboxError(
                        "busy", "service", "service is paused or not running"
                    ) from None
                if docker_id is None and error.status_code == 400:
                    # An unknown user (VERIFIED); never echo the Engine's text.
                    raise SandboxError(
                        "invalid",
                        "user" if request.user is not None else "exec",
                        "the Engine refused the exec, for example an unknown user",
                    ) from None
            elif docker_id is not None:
                # The start may have run the command without a stream.
                self._contain(target, docker_id, scope)
            raise
        now = self._clock()
        record = _Exec(
            exec_id=exec_id,
            session=session,
            env_id=env_id,
            service=request.service,
            target=target,
            docker_id=docker_id,
            scope=scope,
            oom_base=oom_base,
            merge_stderr=request.merge_stderr,
            output_limit=output_limit,
            started_at=now,
            timeout_at=None
            if request.timeout_sec is None
            else now + request.timeout_sec,
            deadline_at=deadline,
            next_inspect=now,
            sock=sock,
            handle=handle,
            fds=fds,
        )
        with self._lock:
            self._records[exec_id] = record
            self._commands.append(("add", record, prefix))
        self._wake()
        self._await_process(record)
        return exec_id

    def _await_process(self, record: _Exec) -> None:
        """Bounded wait until the Engine reports the leader or its exit."""
        give_up = time.monotonic() + self._start_confirm_sec
        while time.monotonic() < give_up:
            try:
                info = self._api.exec_inspect(record.docker_id)
            except (DockerException, OSError):
                return
            pid = info.get("Pid")
            if type(pid) is int and pid > 0:
                if record.host_pid is None:
                    record.host_pid = pid
                return
            if not info.get("Running") and type(info.get("ExitCode")) is int:
                return
            time.sleep(0.005)

    @staticmethod
    def _running_service(attrs: Mapping[str, Any] | None) -> Mapping[str, Any]:
        if attrs is None:
            raise SandboxError("invalid", "service", "service is absent")
        state = attrs.get("State") or {}
        if state.get("Paused"):
            raise SandboxError("busy", "service", "a paused service cannot exec")
        if not state.get("Running"):
            raise SandboxError("invalid", "service", "service is not running")
        return state

    def _service_scope(self, target: ExecTarget, pid: object) -> tuple[Path, int]:
        """The service's cgroup scope and its oom_kill count before the exec.

        Task code can end the service (its PID 1) at any moment, so a scope
        that cannot be read is a stopped service unless the same init still
        runs; only then is it an infrastructure error, of this call alone
        (S8: never the broker's recovery_required).
        """
        try:
            scope = self._table.scope(target.container_id, pid)
            oom_base = self._table.oom_kills(scope)
        except InfrastructureError:
            oom_base = None
        if oom_base is not None:
            return scope, oom_base
        state = self._running_service(target.inspect())
        init = self._table.stat(pid) if type(pid) is int else None
        if state.get("Pid") != pid or init is None or init.state in ("Z", "X"):
            raise SandboxError("invalid", "service", "service is not running")
        LOGGER.warning(
            "sandbox exec service %s has no readable cgroup v2 scope",
            target.container_id[:12],
        )
        raise SandboxError("infrastructure", "service", "service cgroup is unreadable")

    def _contain(self, target: ExecTarget, docker_id: str, scope: Path) -> None:
        try:
            pid = self._api.exec_inspect(docker_id).get("Pid")
            if type(pid) is int and pid > 0:
                process = self._table.process(target.container_id, scope, pid)
                if process is not None:
                    self._killer.signal(process, signal_module.SIGKILL, group=True)
        except Exception as error:  # best effort; the env deadline bounds it
            LOGGER.warning("cannot contain an unattached sandbox exec: %s", error)

    # -- read, wait, kill ----------------------------------------------------------

    def _owned(self, session: Hashable, exec_id: object) -> _Exec:
        record = self._records.get(exec_id) if type(exec_id) is str else None
        if record is None or record.session != session:
            raise SandboxError(
                "permission", "exec_id", "exec is not owned by this session"
            )
        return record

    @staticmethod
    def _offsets(stdout_offset: object, stderr_offset: object) -> tuple[int, int]:
        for name, value in zip(_OFFSETS, (stdout_offset, stderr_offset), strict=True):
            if type(value) is not int or value < 0:
                raise SandboxError("invalid", name, "expected a nonnegative offset")
        return stdout_offset, stderr_offset  # type: ignore[return-value]

    def read(
        self,
        exec_id: str,
        *,
        session: Hashable,
        stdout_offset: int,
        stderr_offset: int,
        max_bytes: int = MAX_READ_BYTES,
    ) -> dict[str, Any]:
        """The exec_wait view: state and at most ``max_bytes`` of new output.

        stdout is served before stderr. Output read to the end of an ended
        exec is released ``RELEASE_GRACE_SEC`` later; reading below a
        released offset is ``invalid``. Output removed with its env before
        it was read ends at the totals, ``truncated``, with the final state.
        """
        offsets = self._offsets(stdout_offset, stderr_offset)
        if type(max_bytes) is not int or not 0 < max_bytes <= MAX_READ_BYTES:
            raise SandboxError("invalid", "max_bytes", "expected 1..1048576 bytes")
        descriptors = [-1, -1]
        with self._lock:
            record = self._owned(session, exec_id)
            totals = tuple(record.totals)
            for index, name in enumerate(_OFFSETS):
                if offsets[index] > totals[index]:
                    raise SandboxError("invalid", name, "offset beyond the output")
            lost = record.released and offsets != totals
            if lost and not record.discarded:
                raise SandboxError(
                    "invalid", "stdout_offset", "output was read and released"
                )
            if lost:
                offsets = totals
            wanted = [min(totals[0] - offsets[0], max_bytes)]
            wanted.append(min(totals[1] - offsets[1], max_bytes - wanted[0]))
            try:
                for index in (0, 1):
                    if wanted[index]:
                        descriptors[index] = self._spool.open_read(exec_id, index)
            except BaseException:
                for descriptor in descriptors:
                    if descriptor >= 0:
                        os.close(descriptor)
                raise
            state, reason = record.state, record.reason
            exit_code, sent, truncated = (
                record.exit_code,
                record.signal,
                record.truncated or lost,
            )
            oom_kills = record.oom_kills
        try:
            chunks = [
                _pread(descriptors[index], wanted[index], offsets[index])
                if wanted[index]
                else b""
                for index in (0, 1)
            ]
        finally:
            for descriptor in descriptors:
                if descriptor >= 0:
                    os.close(descriptor)
        if state == "running":
            oom_kills = self._oom(record)
        after = (offsets[0] + len(chunks[0]), offsets[1] + len(chunks[1]))
        if state != "running" and after == totals:
            with self._lock:
                if not record.released:
                    grace = self._clock() + RELEASE_GRACE_SEC
                    record.release_at = min(record.release_at or math.inf, grace)
        return {
            "state": state,
            "exit_code": exit_code,
            "signal": sent,
            "stdout_b64": base64.b64encode(chunks[0]).decode("ascii"),
            "stderr_b64": base64.b64encode(chunks[1]).decode("ascii"),
            "stdout_offset": after[0],
            "stderr_offset": after[1],
            "stdout_total": totals[0],
            "stderr_total": totals[1],
            "truncated": truncated,
            "oom_kills": oom_kills,
            "reason": reason,
        }

    def wait(
        self,
        exec_id: str,
        *,
        session: Hashable,
        stdout_offset: int,
        stderr_offset: int,
        wait_sec: float,
        max_bytes: int = MAX_READ_BYTES,
    ) -> dict[str, Any]:
        """Block up to ``wait_sec`` (≤30) for new output or the end, then read.

        The broker's async long-poll uses ``generation`` instead and never
        blocks a worker; this is the synchronous form.
        """
        offsets = self._offsets(stdout_offset, stderr_offset)
        if (
            type(wait_sec) not in (int, float)
            or not math.isfinite(wait_sec)
            or not 0 <= wait_sec <= MAX_WAIT_SEC
        ):
            raise SandboxError("invalid", "wait_sec", "expected 0..30 seconds")
        end = time.monotonic() + wait_sec
        with self._changed:
            while True:
                record = self._owned(session, exec_id)
                # A differing offset is new output, or an error for read().
                if record.state != "running" or tuple(record.totals) != offsets:
                    break
                remaining = end - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(remaining)
        return self.read(
            exec_id,
            session=session,
            stdout_offset=offsets[0],
            stderr_offset=offsets[1],
            max_bytes=max_bytes,
        )

    def kill(
        self,
        exec_id: str,
        *,
        session: Hashable,
        signal: str = "TERM",
        scope: str = "group",
    ) -> dict[str, Any]:
        """exec_kill: one signal to the exec's group (or leader), no escalation.

        A leader that dies of a delivered signal (exit code 128 + its number)
        ends the exec as ``killed``/``interrupt`` unless a deadline or env
        stop explains it first; one that survives it and exits is ``exited``,
        with ``signal`` still naming what was delivered.
        """
        signum = SIGNALS.get(signal) if type(signal) is str else None
        if signum is None:
            raise SandboxError("invalid", "signal", "expected INT, TERM, HUP or KILL")
        if scope not in ("group", "process"):
            raise SandboxError("invalid", "scope", "expected group or process")
        with self._lock:
            record = self._owned(session, exec_id)
            if record.state != "running" or record.outcome is not None:
                return {"delivered": False, "state": record.state}
            if record.session in self._frozen:
                raise SandboxError("busy", "session", "session is frozen")
            # The exec may end before the signal call returns; its verdict
            # waits for this delivery (_conclude).
            record.killing += 1
        delivered = False
        try:
            process = self._resolve(record)
            delivered = process is not None and self._killer.signal(
                process, signum, group=scope == "group"
            )
        finally:
            with self._lock:
                record.killing -= 1
                if delivered:
                    record.delivered.add(signum)
                    record.signal = signal
                record.next_inspect = self._clock()
                state = record.state
            self._control_wake.set()
        return {"delivered": delivered, "state": state}

    # -- broker hooks --------------------------------------------------------------

    def generation(self, exec_id: str | None = None) -> int:
        """Lock-free change counter for long-polls: the pump's or one exec's."""
        if exec_id is None:
            return self._generation
        record = self._records.get(exec_id)
        return 0 if record is None else record.generation

    def idle(
        self,
        exec_id: str,
        *,
        session: Hashable,
        stdout_offset: object,
        stderr_offset: object,
    ) -> bool:
        """Whether ``wait`` would block: the exec runs and has nothing unread.

        Ownership and offsets are checked as ``read`` checks them, so a
        malformed long-poll is answered at once with the read's own error.
        """
        offsets = self._offsets(stdout_offset, stderr_offset)
        with self._lock:
            record = self._owned(session, exec_id)
            return record.state == "running" and tuple(record.totals) == offsets

    def running(self, session: Hashable) -> int:
        with self._lock:
            return sum(
                record.session == session and record.state == "running"
                for record in self._records.values()
            )

    def freeze(self, session: Hashable) -> None:
        """freeze_work: the session's services are about to pause.

        No deadline, kill, completion or retention step runs for its execs
        until ``thaw``; ``start`` and ``kill`` are refused as ``busy``.
        """
        with self._lock:
            self._frozen.setdefault(session, self._clock())

    def thaw(self, session: Hashable) -> None:
        """Resume the session's execs; their clocks move by the frozen time.

        Timeouts, kills and drains of running execs and the retention of
        ended ones move alike, so a read retried after a Judge round still
        works. The env deadline is the broker's and does not move here.
        """
        with self._lock:
            since = self._frozen.pop(session, None)
            if since is None:
                return
            now = self._clock()
            shift = max(0.0, now - since)
            for record in self._records.values():
                if record.session != session:
                    continue
                if record.state != "running":
                    if record.expires_at is not None:
                        record.expires_at += shift
                    if record.release_at is not None:
                        record.release_at += shift
                    continue
                if record.timeout_at is not None:
                    record.timeout_at += shift
                if record.kill_at is not None:
                    record.kill_at += shift
                if record.exited_at is not None:
                    # An inspect in flight at the freeze may have seen it later.
                    record.exited_at = min(record.exited_at + shift, now)
                record.next_inspect = now
        self._control_wake.set()

    def env_stopped(
        self, env_id: str, *, service: str | None = None, release: bool = False
    ) -> None:
        """The broker stops (or with ``release`` destroys) an env or a service.

        Running execs there end now as ``killed``/``env_stopped`` (their
        sockets close, ``on_finish`` follows); ``release`` also removes the
        output of every exec of the env. Their final state stays readable
        by the owner until it expires (``read``).
        """
        with self._lock:
            for record in list(self._records.values()):
                if record.env_id == env_id and service in (None, record.service):
                    self._stop_record(record, release="output" if release else None)
        self._wake()

    def close_session(self, session: Hashable) -> None:
        """A revoked session: end and drop its execs and replay keys."""
        with self._lock:
            for record in list(self._records.values()):
                if record.session == session:
                    self._stop_record(record, release="record")
            for key in [key for key in self._replays if key[0] == session]:
                del self._replays[key]
            self._frozen.pop(session, None)
        self._wake()

    def _stop_record(
        self, record: _Exec, *, release: Literal["output", "record"] | None
    ) -> None:
        if record.state == "running":
            record.stopped = True
            if record.outcome is None:
                state, reason = self._verdict(record)
                record.outcome = (
                    state,
                    reason,
                    record.exit_code,
                    record.signal,
                    record.oom_kills,
                )
                self._commands.append(("finish", record, None))
            if release is not None and record.release != "record":
                record.release = release  # applied by _complete
        elif release == "record":
            self._drop(record)
        elif release == "output":
            self._discard(record)

    def close(self) -> None:
        """Stop both loops, close every stream and remove the spool."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._wake()
        self._control_wake.set()
        for thread in self._threads:
            thread.join(timeout=6.0)
            if thread.is_alive():
                LOGGER.warning("sandbox exec thread %s did not stop", thread.name)
        self._calls.close()
        with self._changed:
            records = list(self._records.values())
            self._records.clear()
            self._commands.clear()
            self._changed.notify_all()  # waiters find their exec gone
        for record in records:
            self._close_io(record)
            if not record.released:
                self._remove_output(record)
        self._selector.close()
        self._wake_read.close()
        self._wake_write.close()
        self._spool.close()

    # -- IO thread -----------------------------------------------------------------

    def _io_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step_io(2 * TICK_SEC)
            except Exception:
                LOGGER.exception("sandbox exec pump step failed")

    def _wake(self) -> None:
        try:
            self._wake_write.send(b"\0")
        except (BlockingIOError, OSError):
            pass  # a wakeup is already pending, or the pump is closed

    def step_io(self, timeout: float) -> None:
        """One IO turn: apply registrations and closes, then read."""
        changed = self._apply_commands()
        for key, _ in self._selector.select(timeout):
            record = key.data
            if record is None:
                try:
                    while self._wake_read.recv(4096):
                        pass
                except (BlockingIOError, OSError):
                    pass
                continue
            changed = self._read(record) or changed
        changed = self._apply_commands() or changed
        if changed:
            with self._changed:
                self._changed.notify_all()

    def _apply_commands(self) -> bool:
        changed = False
        while True:
            try:
                kind, record, prefix = self._commands.popleft()
            except IndexError:
                return changed
            changed = True
            if kind == "add":
                if prefix and not self._demux(record, memoryview(prefix)):
                    self._end_stream(record)
                    continue
                try:
                    self._selector.register(record.sock, selectors.EVENT_READ, record)
                except (OSError, ValueError) as error:
                    LOGGER.warning(
                        "sandbox exec %s stream cannot be watched: %s",
                        record.exec_id,
                        error,
                    )
                    self._end_stream(record)
            else:
                if record.sock is not None:
                    # Keep what the Engine already sent; a stream still open
                    # after that was cut, so the output is incomplete.
                    self._read(record, FINAL_READS)
                    if record.sock is not None:
                        record.truncated = True
                self._close_io(record)
                self._complete(record)

    def _read(self, record: _Exec, reads: int = READS_PER_TURN) -> bool:
        changed = False
        for _ in range(reads):
            try:
                count = record.sock.recv_into(self._view)
            except (BlockingIOError, InterruptedError):
                break
            except OSError as error:
                LOGGER.warning(
                    "sandbox exec %s stream failed: %s", record.exec_id, error
                )
                count = 0
            if count:
                record.data_at = self._clock()
            ended = count == 0 or not self._demux(record, self._view[:count])
            changed = True
            if ended:
                self._end_stream(record)
                break
        if changed:
            record.generation = self._generation = next(self._counter)
        return changed

    def _demux(self, record: _Exec, data: memoryview) -> bool:
        """Split Engine frames into the spools; False on a corrupt stream."""
        position = 0
        while position < len(data):
            if record.frame_left == 0:
                needed = _FRAME.size - len(record.header)
                record.header += data[position : position + needed]
                position += min(needed, len(data) - position)
                if len(record.header) < _FRAME.size:
                    break
                stream, length = _FRAME.unpack(record.header)
                record.header.clear()
                if stream not in (1, 2):
                    LOGGER.warning(
                        "sandbox exec %s sent an invalid stream frame", record.exec_id
                    )
                    record.truncated = True
                    return False
                record.frame_stream, record.frame_left = stream, length
                continue
            take = min(record.frame_left, len(data) - position)
            self._store(record, record.frame_stream, data[position : position + take])
            position += take
            record.frame_left -= take
        return True

    def _store(self, record: _Exec, stream: int, data: memoryview) -> None:
        index = 0 if stream == 1 or record.merge_stderr else 1
        room = max(0, record.output_limit - record.totals[index])
        keep = data[:room]
        if len(keep) < len(data):
            record.truncated = True  # read, counted as lost, discarded
        if not keep or record.spool_failed:
            return
        try:
            _write_all(record.fds[index], keep)
        except OSError as error:
            LOGGER.warning("sandbox exec %s spool failed: %s", record.exec_id, error)
            record.spool_failed = record.truncated = True
            return
        record.totals[index] += len(keep)

    def _close_socket(self, record: _Exec) -> None:
        if record.sock is None:
            return
        try:
            self._selector.unregister(record.sock)
        except (KeyError, ValueError, OSError):
            pass
        _close_handle(record.handle)
        record.sock = record.handle = None

    def _end_stream(self, record: _Exec) -> None:
        self._close_socket(record)
        record.eof = True
        self._control_wake.set()

    def _close_io(self, record: _Exec) -> None:
        self._close_socket(record)
        for index, descriptor in enumerate(record.fds):
            if descriptor >= 0:
                os.close(descriptor)
                record.fds[index] = -1

    def _complete(self, record: _Exec) -> None:
        state, reason, exit_code, sent, oom_kills = record.outcome
        if self._on_finish is not None:
            # Settled before any reader can see the end (see the class).
            try:
                self._on_finish(
                    ExecSummary(
                        exec_id=record.exec_id,
                        session=record.session,
                        env_id=record.env_id,
                        state=state,
                        reason=reason,
                        stdout_total=record.totals[0],
                        stderr_total=record.totals[1],
                        output_limit=record.output_limit,
                        merge_stderr=record.merge_stderr,
                    )
                )
            except Exception:
                LOGGER.exception("sandbox exec %s settlement failed", record.exec_id)
        now = self._clock()
        with self._lock:
            record.state, record.reason = state, reason
            record.exit_code, record.signal, record.oom_kills = (
                exit_code,
                sent,
                oom_kills,
            )
            record.expires_at = now + RETAIN_SEC
            record.generation = self._generation = next(self._counter)
            if self._records.get(record.exec_id) is record:
                if record.release == "record":
                    self._drop(record)
                else:
                    if record.release == "output":
                        self._discard(record)
                    self._retain(record.session)

    # -- control thread ------------------------------------------------------------

    def _control_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.step_control()
            except Exception:
                LOGGER.exception("sandbox exec control step failed")
            self._control_wake.wait(TICK_SEC)
            self._control_wake.clear()

    def step_control(self) -> None:
        """One control turn: deadlines, kills, completion, purge.

        Nothing here waits on the Engine or a killer: inspects, signals and
        service checks are calls (``_Calls``), at most one inspect and one
        signal in flight per exec, whose results later turns act on.
        """
        now = self._clock()
        with self._lock:
            active = [
                record
                for record in self._records.values()
                if record.state == "running"
                and record.outcome is None
                and record.session not in self._frozen
            ]
        for record in active:
            try:
                self._advance(record, now)
            except Exception:
                LOGGER.exception("sandbox exec %s control failed", record.exec_id)
                self._conclude(record, lost=True)
        if now >= self._next_purge:
            self._next_purge = now + PURGE_SEC
            self._purge(now)

    def _advance(self, record: _Exec, now: float) -> None:
        if record.eof and record.eof_seen is None:
            record.eof_seen = record.next_inspect = now  # the exit is near
        # Only a command still running expires; after its exit only the drain
        # runs, and an exit no inspect reported yet is asked for first.
        due = self._due(record)
        if (
            due is not None
            and now >= due[1]
            and record.expiry is None
            and record.exited_at is None
        ):
            if not self._running_since(record, due[1]):
                self._request_inspect(record)
            if record.exited_at is None and self._running_since(record, due[1]):
                record.expiry = due[0]
        if record.expiry is not None and record.outcome is None:
            self._escalate(record, now)
        if record.exited_at is None and now >= record.next_inspect:
            self._request_inspect(record)
        if record.outcome is not None:
            return
        if (
            record.exited_at is not None
            and (record.eof or now >= self._drain_end(record))
            # An expired exec ends with its whole group, not just its leader.
            and (record.expiry is None or record.kill_step > 1 or self._gone(record))
        ):
            self._finish(record)

    @staticmethod
    def _due(record: _Exec) -> tuple[Literal["timeout", "deadline"], float] | None:
        """The earlier of the exec's timeout and the env deadline."""
        timeout_at, deadline_at = record.timeout_at, record.deadline_at
        if timeout_at is not None and (
            deadline_at is None or timeout_at <= deadline_at
        ):
            return "timeout", timeout_at
        if deadline_at is not None:
            return "deadline", deadline_at
        return None

    def _running_since(self, record: _Exec, at: float) -> bool:
        """Whether the command still ran at or after ``at``.

        A pinned leader is read from ``/proc`` now; otherwise an inspect
        sent no earlier than ``at`` must have found the exec running.
        """
        if record.process is not None:
            return bool(self._table.targets(record.process, group=False))
        return record.alive_at is not None and record.alive_at >= at

    @staticmethod
    def _drain_end(record: _Exec) -> float:
        """1 s after the exit or the last bytes, at most 3 s after the exit."""
        assert record.exited_at is not None
        last = max(record.exited_at, record.data_at or record.exited_at)
        return min(record.exited_at + DRAIN_MAX_SEC, last + DRAIN_SEC)

    def _escalate(self, record: _Exec, now: float) -> None:
        """TERM at expiry, KILL to what is left 2 s later, then stop waiting."""
        if record.kill_step == 0:
            if self._request_signal(record, signal_module.SIGTERM):
                record.kill_step, record.kill_at = 1, now + KILL_GRACE_SEC
        elif record.kill_step == 1 and now >= record.kill_at:
            if self._request_signal(record, signal_module.SIGKILL):
                record.kill_step, record.kill_at = 2, now + ABANDON_SEC
        elif (
            record.kill_step == 2 and record.exited_at is None and now >= record.kill_at
        ):
            process = record.process
            if (
                process is not None
                and now < record.kill_at + ABANDON_SEC
                and not self._table.targets(process, group=False)
            ):
                # Gone, but the Engine reports a container's exits in order,
                # behind any exec's 2 s stream wait: its exit code comes later.
                return
            LOGGER.warning(
                "sandbox exec %s has no exit after SIGKILL; abandoning it",
                record.exec_id,
            )
            self._conclude(record, abandoned=True)

    def _gone(self, record: _Exec) -> bool:
        process = record.process
        return process is None or not self._table.targets(process, group=True)

    def _request_inspect(self, record: _Exec) -> None:
        if record.inspecting or record.outcome is not None:
            return
        record.inspecting = True
        self._calls.submit(lambda: self._inspect(record))

    def _request_signal(self, record: _Exec, signum: int) -> bool:
        """Signal the exec's group on a call; False while one is in flight."""
        if record.signalling:
            return False
        record.signalling = True
        self._calls.submit(lambda: self._signal(record, signum))
        return True

    def _signal(self, record: _Exec, signum: int) -> None:
        """A call: one signal to the group of a pinned leader."""
        try:
            process = self._resolve(record)
            if process is not None and self._killer.signal(process, signum, group=True):
                with self._lock:
                    record.signal = _SIGNAL_NAMES[signum]
        except Exception:
            LOGGER.exception("sandbox exec %s signal failed", record.exec_id)
        finally:
            record.signalling = False
            record.next_inspect = self._clock()  # learn of the exit at once
            self._control_wake.set()

    def _resolve(self, record: _Exec) -> ExecProcess | None:
        """The leader, resolved once: the Engine reports Pid 0 until the
        process started (VERIFIED)."""
        if record.process is not None:
            return record.process
        if record.host_pid is None:
            try:
                pid = self._api.exec_inspect(record.docker_id).get("Pid")
            except (DockerException, OSError):
                return None
            if type(pid) is int and pid > 0:
                record.host_pid = pid
        if record.host_pid is not None:
            record.process = self._table.process(
                record.target.container_id, record.scope, record.host_pid
            )
        return record.process

    def _inspect(self, record: _Exec) -> None:
        """A call: one exec_inspect for the leader, the exit and the cadence."""
        try:
            sent = self._clock()
            try:
                info = self._api.exec_inspect(record.docker_id)
            except NotFound:
                LOGGER.warning(
                    "sandbox exec %s vanished from the Engine", record.exec_id
                )
                self._conclude(record, lost=True)
                return
            except (DockerException, OSError) as error:
                record.inspect_failures += 1
                record.next_inspect = self._clock() + INSPECT_SEC
                if record.inspect_failures >= MAX_INSPECT_FAILURES:
                    LOGGER.warning(
                        "sandbox exec %s cannot be inspected: %s",
                        record.exec_id,
                        error,
                    )
                    self._conclude(record, lost=True)
                return
            now = self._clock()
            record.inspect_failures = 0
            pid, code = info.get("Pid"), info.get("ExitCode")
            if record.host_pid is None and type(pid) is int and pid > 0:
                record.host_pid = pid
                record.process = self._table.process(
                    record.target.container_id, record.scope, pid
                )
            self._oom(record)
            if not info.get("Running") and type(code) is int:
                record.exit_code, record.exited_at = code, now
                return
            record.alive_at = sent
            if (record.host_pid is None and now - record.started_at < DRAIN_SEC) or (
                record.eof_seen is not None and now - record.eof_seen < DRAIN_SEC
            ):
                record.next_inspect = now + TICK_SEC  # starting, or ending soon
            else:
                record.next_inspect = now + INSPECT_SEC
        except Exception:
            LOGGER.exception("sandbox exec %s inspect failed", record.exec_id)
            self._conclude(record, lost=True)
        finally:
            record.inspecting = False
            self._control_wake.set()

    def _oom(self, record: _Exec) -> int:
        current = self._table.oom_kills(record.scope)
        if current is not None:
            record.oom_kills = max(0, current - record.oom_base)
        return record.oom_kills

    def _verdict(self, record: _Exec) -> tuple[ExecState, ExecReason | None]:
        if record.expiry == "timeout":
            return "timed_out", "timeout"
        if record.expiry == "deadline":
            return "killed", "deadline"
        if record.stopped:
            return "killed", "env_stopped"
        code = record.exit_code
        if code is not None and code - 128 in record.delivered:
            return "killed", "interrupt"
        return "exited", None

    def _finish(self, record: _Exec) -> None:
        """Conclude a drained exec; a signal-like exit asks after its service."""
        code = record.exit_code
        if code is None or code < 128 or self._verdict(record)[0] != "exited":
            self._conclude(record)
        elif not record.inspecting:
            # Killed with its service (stop, OOM of its init, outside kill)?
            record.inspecting = True
            self._calls.submit(lambda: self._conclude_checked(record))

    def _conclude_checked(self, record: _Exec) -> None:
        """A call: conclude after one inspect of the exec's service."""
        try:
            self._conclude(record, with_service=self._service_stopped(record))
        finally:
            record.inspecting = False
            self._control_wake.set()

    def _service_stopped(self, record: _Exec) -> bool:
        try:
            attrs = record.target.inspect()
        except Exception:
            return False
        return attrs is None or not (attrs.get("State") or {}).get("Running")

    def _conclude(
        self,
        record: _Exec,
        *,
        lost: bool = False,
        abandoned: bool = False,
        with_service: bool = False,
    ) -> None:
        oom_kills = self._oom(record)
        with self._lock:
            if record.outcome is not None:
                return
            if (record.killing or record.signalling) and not (lost or abandoned):
                return  # the next turn knows whether the signal landed
            state, reason = self._verdict(record)
            if lost:
                state, reason = "lost", None
            elif state == "exited" and with_service:
                state, reason = "killed", "env_stopped"
            exit_code = None if lost or abandoned else record.exit_code
            record.outcome = (state, reason, exit_code, record.signal, oom_kills)
            self._commands.append(("finish", record, None))
        self._wake()

    # -- retention -----------------------------------------------------------------

    def _remove_output(self, record: _Exec) -> None:
        try:
            self._spool.remove(record.exec_id)
        except OSError as error:
            LOGGER.warning(
                "cannot remove sandbox exec %s output: %s", record.exec_id, error
            )
        record.released = True

    def _discard(self, record: _Exec) -> None:
        """The env was destroyed: its output goes, the final state stays."""
        if not record.released:
            self._remove_output(record)
            record.discarded = True

    def _drop(self, record: _Exec) -> None:
        if self._records.get(record.exec_id) is record:
            del self._records[record.exec_id]
        if not record.released:
            self._remove_output(record)

    def _retain(self, session: Hashable) -> None:
        ended = sorted(
            (
                record
                for record in self._records.values()
                if record.session == session and record.state != "running"
            ),
            key=lambda record: record.expires_at or 0.0,
        )
        for record in ended[: max(0, len(ended) - MAX_RETAINED_EXECS)]:
            self._drop(record)

    def _purge(self, now: float) -> None:
        with self._lock:
            for record in list(self._records.values()):
                # A frozen session's clocks stand still (thaw moves them).
                if record.state == "running" or record.session in self._frozen:
                    continue
                if record.expires_at is not None and now >= record.expires_at:
                    self._drop(record)
                elif (
                    not record.released
                    and record.release_at is not None
                    and now >= record.release_at
                ):
                    self._remove_output(record)


__all__ = [
    "EXEC_ID",
    "ExecKiller",
    "ExecProcess",
    "ExecPump",
    "ExecRequest",
    "ExecState",
    "ExecSummary",
    "ExecTarget",
    "HostPidfdKiller",
    "InContainerKiller",
    "ProcStat",
    "ProcessTable",
    "SIGNALS",
    "default_exec_killer",
    "parse_exec_request",
    "pidfd_calls",
]
