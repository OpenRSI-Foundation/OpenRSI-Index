"""An in-process fake of a remote backend's client (a
sandbox_remote.RemoteClient), by default E2B's (sandbox_e2b.SdkE2BClient).

No network and no account: a sandbox is a directory, and a command runs as
a real local process with the sandbox's absolute paths mapped under that
directory (every argument after the program, and the working directory),
so ``setsid``, ``kill``, ``tar`` and ``stat`` behave for real. The user is
recorded, not switched. Pause stops the sandbox's process groups (SIGSTOP),
resume continues them, kill ends them.

As on E2B, a pause cuts every process stream (the process lives on) and
``connect`` follows a process again once the sandbox runs; envd's own
``E2B_*`` variables are in every process's environment unless the request
sets them, and the build step's recording carries the build sandbox's.

``EMULATED`` makes it Modal's (sandbox_modal.SdkModalClient): exact
shapes, Dockerfile builds as lines, and none of E2B's pids, users and
pause, so sandbox_remote emulates them; ``DAYTONA`` makes it Daytona's
(sandbox_daytona.SdkDaytonaClient), emulated the same way, with shapes in
cores and GiB and no builds. Emulated, commands come through
``group_argv`` (run for real) and
``setpriv`` (its user recorded, the prefix dropped: tests run
unprivileged); SignalFreeze's scripts, which walk /proc, never run: the
fake stops and continues the sandbox's own process groups instead, and
cuts no stream. Its processes get the registry config's ENV (no envd).
"""

from __future__ import annotations

import os
import queue
import secrets
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from rsi_harness.runtime import sandbox_daytona, sandbox_modal
from rsi_harness.runtime.sandbox_e2b import RECIPE, recorded_image_env, template_shape
from rsi_harness.runtime.sandbox_remote import (
    FREEZE_SCRIPT,
    THAW_SCRIPT,
    ChunkReader,
    config_env,
    group_argv,
)

# What envd hands a template build step: the image ENV with E2B's fallback
# PATH suffix, plus HOME/USER/LOGNAME (sandbox_e2b drops those again).
IMAGE_ENV = (
    b"PATH=/usr/local/bin:/usr/bin:/bin"
    b":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\0"
    b"LANG=C.UTF-8\0IMAGE_ONLY=from-image\0HOME=/root\0USER=root\0LOGNAME=root\0"
    b"E2B_SANDBOX=true\0E2B_SANDBOX_ID=ibuildsandbox\0E2B_TEMPLATE_ID=tbuild\0"
    b"E2B_EVENTS_ADDRESS=http://192.0.2.1\0"
)
# What the SDK reports when a pause ends a process stream.
CUT = "the connection to sandbox {} ended before the stream completed"
# What the image's registry holds beyond the ENV (RemoteFake.image_configs).
IMAGE_CONFIG = {"Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8"]}
PASSWD = b"root:x:0:0:root:/root:/bin/sh\nagent:x:1000:1000::/home/agent:/bin/sh\n"
# RemoteFake as Modal's client: every gap of it sandbox_remote emulates.
EMULATED = {
    "name": "modal",
    "native_pid": False,
    "native_user": False,
    "freeze": "signal",
}
# RemoteFake as Daytona's client: emulated as Modal's.
DAYTONA = {**EMULATED, "name": "daytona"}
# The fake for each backend name.
FAKES = {"e2b": {}, "modal": EMULATED, "daytona": DAYTONA}


class FakeProcess:
    def __init__(
        self, popen: subprocess.Popen, sandbox_id: str = "", *, wrapped=False
    ) -> None:
        self.pid = popen.pid
        self.exit_code: int | None = None
        self.popen = popen
        self._sandbox_id = sandbox_id
        self._open = 2
        self._queue: queue.Queue = queue.Queue()
        # A group_argv command's group: its first stdout line.
        self._group = None if wrapped else popen.pid
        self._line = b""
        self._grouped = threading.Event()
        if not wrapped:
            self._grouped.set()
        for stream, pipe in ((1, popen.stdout), (2, popen.stderr)):
            threading.Thread(
                target=self._read, args=(stream, pipe), daemon=True
            ).start()

    def group(self):
        """The process group to signal (the wrapper's command's)."""
        self._grouped.wait(2)
        return self._group or self.pid

    def _read(self, stream, pipe):
        try:
            for chunk in iter(lambda: os.read(pipe.fileno(), 65536), b""):
                if stream == 1 and not self._grouped.is_set():
                    self._line += chunk
                    if b"\n" in self._line:
                        self._group = int(self._line.split(b"\n", 1)[0])
                        self._grouped.set()
                self._queue.put((stream, chunk))
        finally:
            pipe.close()
            self._grouped.set()
            self._queue.put((stream, None))

    def cut(self):
        """A pause: the stream breaks; connect() follows the rest."""
        self._queue.put(("cut", None))

    def __iter__(self):
        while self._open:
            stream, chunk = self._queue.get()
            if stream == "cut":
                raise RuntimeError(CUT.format(self._sandbox_id))
            if chunk is None:
                self._open -= 1
                continue
            yield stream, chunk
        code = self.popen.wait()
        # As the SDK adapter reports a signal death (shell_exit_code).
        self.exit_code = 128 - code if code < 0 else code

    def close(self):
        pass  # a disconnect: the process keeps running


@dataclass(eq=False)
class FakeSandbox:
    sandbox_id: str
    root: Path
    template: str
    metadata: dict
    timeout: int
    allow_internet: bool
    allow_out: tuple = ()
    state: str | None = "running"
    processes: list = field(default_factory=list)
    # Stopped by SignalFreeze's script: the backend still sees it running.
    frozen: bool = False

    @property
    def paused(self):
        return self.state == "paused" or self.frozen


class RemoteFake:
    """``RemoteClient`` over local directories and processes, sized and
    named as SdkE2BClient (or, named "modal" or "daytona", SdkModalClient or
    SdkDaytonaClient) does.

    ``build_gate`` (an Event) holds builds; ``build_error`` fails them;
    ``dockerfile_builds`` records each build's steps, context files and
    force; ``fail`` maps an operation name to the exception it raises next.
    ``started`` holds each command as the backend would run it (sandbox,
    argv, envs, cwd, user): a grouped one as ``("setsid", *argv)`` and as
    its user, however it came; ``raw`` holds the argv ``start`` got.
    """

    def __init__(
        self,
        base: Path,
        *,
        name: str = "e2b",
        recorded_env: bytes = IMAGE_ENV,
        max_hours: int = 1,
        native_pid: bool = True,
        native_user: bool = True,
        freeze: str = "native",
    ) -> None:
        self.name = name
        modal = name == "modal"
        self.recipe = {
            "modal": sandbox_modal.RECIPE,
            "daytona": sandbox_daytona.RECIPE,
        }.get(name, RECIPE)
        self.build_form = {"modal": "lines", "daytona": None}.get(name, "steps")
        self.argv_bytes = sandbox_modal.ARGV_BYTES if modal else None
        self.freeze_spare = ()
        self.native_pid = native_pid
        self.native_user = native_user
        self.freeze = freeze
        # E2B's envd: its variables and its recording of the image ENV.
        self.envd = native_pid and native_user and freeze == "native"
        self.base = Path(base)
        # The team's maximum sandbox length: a longer timeout is refused.
        self.max_hours = max_hours
        # "domain/remote:tag" (or @digest) -> its config; default IMAGE_CONFIG.
        self.image_configs: dict[str, dict] = {}
        self.base.mkdir(parents=True, exist_ok=True)
        self.recorded_env = recorded_env
        self.templates: dict[str, dict] = {}
        self.builds: list[tuple] = []
        self.dockerfile_builds: list[tuple] = []
        self.build_gate: threading.Event | None = None
        self.build_error: Exception | None = None
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.started: list[tuple] = []
        self.raw: list[tuple] = []
        self.calls: list[tuple] = []
        # Bytes each read_stream handed out, one entry per opened stream.
        self.streamed: list[int] = []
        self.fail: dict[str, Exception] = {}
        self.closed = False
        self._lock = threading.Lock()

    def _hook(self, operation, *args):
        self.calls.append((operation, *args))
        error = self.fail.pop(operation, None)
        if error is not None:
            raise error

    # -- templates ---------------------------------------------------------

    def shape(self, cpus, memory_mb, disk_mb):
        if self.name == "daytona":
            return sandbox_daytona.sandbox_shape(cpus, memory_mb, disk_mb)
        if self.name == "modal":
            return (float(cpus), int(memory_mb))
        return template_shape(cpus, memory_mb)

    def template_exists(self, name):
        self._hook("template_exists", name)
        return name in self.templates

    def build_template(
        self, name, image, shape, *, log, steps=(), context=None, force=False
    ):
        self._hook("build_template", name)
        assert steps == () or self.build_form is not None
        cpu_count, memory_mb = shape[:2]
        if self.name == "e2b" and cpu_count != 1 and cpu_count % 2:
            raise RuntimeError("400: CPU count must be 1 or an even number")
        if self.name != "daytona" and memory_mb < 128:
            raise RuntimeError("400: Memory must be at least 128 MiB")
        self.builds.append((name, image, cpu_count, memory_mb))
        # A Dockerfile build's steps, the context files COPY may read, and
        # whether E2B's cache was skipped (FakeE2B.dockerfile_builds).
        files = (
            None
            if context is None
            else sorted(
                str(path.relative_to(context))
                for path in Path(context).rglob("*")
                if path.is_file()
            )
        )
        self.dockerfile_builds.append((name, image, tuple(steps), files, force))
        log("Step 1/2: FROM " + image + "\n")
        if self.build_gate is not None:
            self.build_gate.wait(10)
        if self.build_error is not None:
            raise self.build_error
        self.templates[name] = {"image": image, "cpu": cpu_count, "memory": memory_mb}

    def image_config(self, repository, tag):
        self._hook("image_config", repository, tag)
        separator = "@" if tag.startswith("sha256:") else ":"
        return dict(
            self.image_configs.get(f"{repository}{separator}{tag}", IMAGE_CONFIG)
        )

    # -- sandboxes ---------------------------------------------------------

    def _check_timeout(self, timeout):
        # E2B's API (timeout_helper.go): a timeout above the team's limit.
        if timeout > self.max_hours * 3600:
            raise RuntimeError(
                f"400: Timeout cannot be greater than {self.max_hours} hours"
            )

    def create(self, template, *, timeout, metadata, allow_internet, allow_out=()):
        self._hook("create", template)
        self._check_timeout(timeout)
        if template not in self.templates:
            raise RuntimeError(f"template {template} not found")
        sandbox_id = "i" + secrets.token_hex(9)
        root = self.base / sandbox_id
        for directory in ("etc/rsi-harness", "tmp", "root", "usr/local/bin"):
            (root / directory).mkdir(parents=True)
        (root / "etc/rsi-harness/image.env").write_bytes(self.recorded_env)
        (root / "etc/passwd").write_bytes(PASSWD)
        with self._lock:
            self.sandboxes[sandbox_id] = FakeSandbox(
                sandbox_id,
                root,
                template,
                dict(metadata),
                timeout,
                allow_internet,
                tuple(allow_out),
            )
        return sandbox_id

    def image_env(self, sandbox_id, image_env):
        if self.envd:
            return recorded_image_env(self, sandbox_id)
        return config_env(image_env)

    def lookup(self, env_id):
        return None

    def live(self):
        return [box for box in self.sandboxes.values() if box.state is not None]

    def state(self, sandbox_id):
        self._hook("state", sandbox_id)
        box = self.sandboxes.get(sandbox_id)
        return None if box is None else box.state

    def _signal_all(self, box, signum):
        # Only processes not reaped yet: a reused PID is never signalled.
        for process in box.processes:
            if process.popen.poll() is not None:
                continue
            for send in (os.killpg, os.kill):
                try:
                    send(process.group(), signum)
                    break
                except (ProcessLookupError, PermissionError):
                    continue

    def kill(self, sandbox_id):
        self._hook("kill", sandbox_id)
        box = self.sandboxes.get(sandbox_id)
        if box is None or box.state is None:
            return
        self._signal_all(box, signal.SIGKILL)
        box.state = None

    def pause(self, sandbox_id):
        assert self.freeze == "native"
        self._hook("pause", sandbox_id)
        box = self.sandboxes[sandbox_id]
        if box.state == "running":
            self._signal_all(box, signal.SIGSTOP)
            box.state = "paused"
            for process in box.processes:
                if process.exit_code is None:
                    process.cut()

    def resume(self, sandbox_id, *, timeout):
        assert self.freeze == "native"
        self._hook("resume", sandbox_id)
        self._check_timeout(timeout)
        box = self.sandboxes[sandbox_id]
        if box.state == "paused":
            box.state = "running"
            box.timeout = timeout
            self._signal_all(box, signal.SIGCONT)

    def list(self, metadata):
        self._hook("list", dict(metadata))
        return [
            (box.sandbox_id, dict(box.metadata))
            for box in self.live()
            if metadata.items() <= box.metadata.items()
        ]

    # -- files and processes ------------------------------------------------

    def _running(self, sandbox_id):
        box = self.sandboxes.get(sandbox_id)
        if box is None or box.state is None:
            raise RuntimeError(f"sandbox {sandbox_id} not found")
        if box.state != "running":
            raise RuntimeError(f"sandbox {sandbox_id} is paused")
        return box

    @staticmethod
    def _map(root, path):
        return str(root) + path if path.startswith("/") else path

    def write_file(self, sandbox_id, path, data):
        self._hook("write_file", path)
        box = self._running(sandbox_id)
        target = Path(self._map(box.root, path))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data if isinstance(data, bytes) else data.read())

    def read_file(self, sandbox_id, path):
        self._hook("read_file", path)
        box = self._running(sandbox_id)
        target = Path(self._map(box.root, path))
        if not target.is_file():
            raise FileNotFoundError(path)
        return target.read_bytes()

    def read_stream(self, sandbox_id, path):
        def chunks(file, index):
            with file:
                for chunk in iter(lambda: file.read(65536), b""):
                    self.streamed[index] += len(chunk)
                    yield chunk

        def opened():
            # As the SDK does: the request opens on the first read.
            self._hook("read_stream", path)
            file = open(self._map(self._running(sandbox_id).root, path), "rb")
            self.streamed.append(0)
            return chunks(file, len(self.streamed) - 1)

        return ChunkReader(opened, lambda error: error)

    def _freeze_script(self, box, script):
        """SignalFreeze's script, answered for the sandbox's own processes."""
        self.calls.append(("freeze" if script == FREEZE_SCRIPT else "thaw",))
        if script == FREEZE_SCRIPT:
            self._signal_all(box, signal.SIGSTOP)
            box.frozen = True
            return ["true"]
        if not box.frozen:
            return ["sh", "-c", "echo no freeze record >&2; exit 1"]
        box.frozen = False
        self._signal_all(box, signal.SIGCONT)
        return ["true"]

    def _user_of(self, box, options):
        uid = next(int(item[8:]) for item in options if item.startswith("--reuid="))
        for line in (box.root / "etc/passwd").read_text().splitlines():
            name, _, found, *_ = line.split(":")
            if int(found) == uid:
                return name
        return str(uid)

    def start(self, sandbox_id, argv, *, envs, cwd, user):
        self._hook("start", tuple(argv))
        box = self._running(sandbox_id)
        self.raw.append(tuple(argv))
        if not self.native_user:
            assert user == "root", "a client without native_user runs only root"
        argv = list(argv)
        wrapped = argv[:4] == group_argv([])
        assert not (wrapped and self.native_pid)
        group = wrapped or argv[0] == "setsid"
        command = argv[4:] if wrapped else argv[1:] if group else argv
        if command[0] == "setpriv":
            assert not self.native_user
            end = command.index("--")
            user = self._user_of(box, command[1:end])
            command = command[end + 1 :]
        self.started.append(
            (
                sandbox_id,
                ("setsid", *command) if group else tuple(command),
                dict(envs),
                cwd,
                user,
            )
        )
        if command[:2] == ["sh", "-c"] and command[2] in (FREEZE_SCRIPT, THAW_SCRIPT):
            assert self.freeze == "signal" and not group
            command = self._freeze_script(box, command[2])
        mapped = [command[0], *(self._map(box.root, item) for item in command[1:])]
        if wrapped:
            mapped = group_argv(mapped)
        elif group:
            mapped = ["setsid", *mapped]
        envd = {
            "E2B_SANDBOX": "true",
            "E2B_SANDBOX_ID": sandbox_id,
            "E2B_TEMPLATE_ID": box.template,
        }
        popen = subprocess.Popen(
            mapped,
            cwd=self._map(box.root, cwd),
            env={**(envd if self.envd else {}), **envs},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process = FakeProcess(popen, sandbox_id, wrapped=wrapped)
        box.processes.append(process)
        return process

    def connect(self, sandbox_id, pid):
        self._hook("connect", sandbox_id, pid)
        box = self._running(sandbox_id)
        for process in box.processes:
            if process.pid == pid:
                return process
        raise RuntimeError(f"process with pid {pid} not found")

    def close(self):
        self.closed = True

    def shutdown(self):
        """Tests: end every process still running."""
        for box in self.sandboxes.values():
            self._signal_all(box, signal.SIGCONT)
            self._signal_all(box, signal.SIGKILL)
