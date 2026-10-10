"""An in-process fake of the E2B SDK surface sandbox_e2b uses (E2BClient).

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
# What the image's registry holds beyond the ENV (FakeE2B.image_configs).
IMAGE_CONFIG = {"Env": ["PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8"]}
PASSWD = b"root:x:0:0:root:/root:/bin/sh\nagent:x:1000:1000::/home/agent:/bin/sh\n"


class FakeProcess:
    def __init__(self, popen: subprocess.Popen, sandbox_id: str = "") -> None:
        self.pid = popen.pid
        self.exit_code: int | None = None
        self._popen = popen
        self._sandbox_id = sandbox_id
        self._open = 2
        self._queue: queue.Queue = queue.Queue()
        for stream, pipe in ((1, popen.stdout), (2, popen.stderr)):
            threading.Thread(
                target=self._read, args=(stream, pipe), daemon=True
            ).start()

    def _read(self, stream, pipe):
        try:
            for chunk in iter(lambda: os.read(pipe.fileno(), 65536), b""):
                self._queue.put((stream, chunk))
        finally:
            pipe.close()
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
        code = self._popen.wait()
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
    state: str | None = "running"
    pids: list = field(default_factory=list)
    processes: list = field(default_factory=list)


class FakeE2B:
    """``E2BClient`` over local directories and processes.

    ``build_gate`` (an Event) holds builds; ``build_error`` fails them;
    ``fail`` maps an operation name to the exception it raises next.
    """

    def __init__(
        self, base: Path, *, image_env: bytes = IMAGE_ENV, max_hours: int = 1
    ) -> None:
        self.base = Path(base)
        # The team's maximum sandbox length: a longer timeout is refused.
        self.max_hours = max_hours
        # "domain/remote:tag" (or @digest) -> its config; default IMAGE_CONFIG.
        self.image_configs: dict[str, dict] = {}
        self.base.mkdir(parents=True, exist_ok=True)
        self.image_env = image_env
        self.templates: dict[str, dict] = {}
        self.builds: list[tuple] = []
        self.build_gate: threading.Event | None = None
        self.build_error: Exception | None = None
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.started: list[tuple] = []
        self.calls: list[tuple] = []
        self.fail: dict[str, Exception] = {}
        self.closed = False
        self._lock = threading.Lock()

    def _hook(self, operation, *args):
        self.calls.append((operation, *args))
        error = self.fail.pop(operation, None)
        if error is not None:
            raise error

    # -- templates ---------------------------------------------------------

    def template_exists(self, name):
        self._hook("template_exists", name)
        return name in self.templates

    def build_template(self, name, image, *, cpu_count, memory_mb, log):
        self._hook("build_template", name)
        if cpu_count != 1 and cpu_count % 2:
            raise RuntimeError("400: CPU count must be 1 or an even number")
        if memory_mb < 128:
            raise RuntimeError("400: Memory must be at least 128 MiB")
        self.builds.append((name, image, cpu_count, memory_mb))
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

    def create(self, template, *, timeout, metadata, allow_internet):
        self._hook("create", template)
        self._check_timeout(timeout)
        if template not in self.templates:
            raise RuntimeError(f"template {template} not found")
        sandbox_id = "i" + secrets.token_hex(9)
        root = self.base / sandbox_id
        for directory in ("etc/rsi-harness", "tmp", "root", "usr/local/bin"):
            (root / directory).mkdir(parents=True)
        (root / "etc/rsi-harness/image.env").write_bytes(self.image_env)
        (root / "etc/passwd").write_bytes(PASSWD)
        with self._lock:
            self.sandboxes[sandbox_id] = FakeSandbox(
                sandbox_id, root, template, dict(metadata), timeout, allow_internet
            )
        return sandbox_id

    def live(self):
        return [box for box in self.sandboxes.values() if box.state is not None]

    def state(self, sandbox_id):
        self._hook("state", sandbox_id)
        box = self.sandboxes.get(sandbox_id)
        return None if box is None else box.state

    def _signal_all(self, box, signum):
        # Only processes not reaped yet: a reused PID is never signalled.
        for popen in box.pids:
            if popen.poll() is not None:
                continue
            for send in (os.killpg, os.kill):
                try:
                    send(popen.pid, signum)
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
        self._hook("pause", sandbox_id)
        box = self.sandboxes[sandbox_id]
        if box.state == "running":
            self._signal_all(box, signal.SIGSTOP)
            box.state = "paused"
            for process in box.processes:
                if process.exit_code is None:
                    process.cut()

    def resume(self, sandbox_id, *, timeout):
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

    def start(self, sandbox_id, argv, *, envs, cwd, user):
        self._hook("start", tuple(argv))
        box = self._running(sandbox_id)
        self.started.append((sandbox_id, tuple(argv), dict(envs), cwd, user))
        program = 2 if argv[0] == "setsid" else 1
        mapped = [
            *argv[:program],
            *(self._map(box.root, item) for item in argv[program:]),
        ]
        envd = {
            "E2B_SANDBOX": "true",
            "E2B_SANDBOX_ID": sandbox_id,
            "E2B_TEMPLATE_ID": box.template,
        }
        popen = subprocess.Popen(
            mapped,
            cwd=self._map(box.root, cwd),
            env={**envd, **envs},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        box.pids.append(popen)
        process = FakeProcess(popen, sandbox_id)
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
