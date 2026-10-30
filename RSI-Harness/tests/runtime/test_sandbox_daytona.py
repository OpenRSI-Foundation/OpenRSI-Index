"""The Daytona SDK adapter (sandbox_daytona) over a fake async ``daytona``
module, and its proxy over the real pinned SDK and a local CONNECT proxy.

The fake runs each session command as the daemon does: the session's shell
sources the command (``{ . file; }``) and writes its exit code afterwards,
with stdin held open, and the log follows line by line. The broker over
Daytona's flags runs in tests/runtime/test_sandbox_e2b.py's generic tests.

``FakeDaytona.can_delete`` and ``listing`` make the key one with
``write:sandboxes`` only, as measured on 2026-10-30: a delete is refused
(403), the listing comes back empty (or refused), and a stop removes an
ephemeral sandbox.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import io
import itertools
import json
import os
import sys
import tomllib
import types
from pathlib import Path

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime import sandbox_daytona
from rsi_harness.runtime.sandbox_contracts import (
    EnvDaytonaHost,
    ProviderCapacityError,
    SandboxOwner,
)
from rsi_harness.runtime.sandbox_daytona import (
    SDK_VERSION,
    SdkDaytonaClient,
    check_daytona_host,
    read_api_key,
    sandbox_shape,
)
from rsi_harness.runtime.sandbox_remote import (
    kill_run_sandboxes,
    sandbox_metadata,
    start_group,
    template_name,
)
from rsi_harness.runtime.sandbox_remote_ledger import SandboxLedger
from tests.runtime.test_sandbox_modal import ConnectProxy

KEY = "dtn_" + "5ec" * 8
ENV_ID = "e" + "a" * 32
METADATA = sandbox_metadata(
    SandboxOwner(run_id="run-1", task_id="task", phase="work"), ENV_ID, "main"
)


# -- a fake daytona module -------------------------------------------------------------


class DaytonaError(Exception):
    status_code = None


class DaytonaNotFoundError(DaytonaError):
    status_code = 404


class DaytonaConflictError(DaytonaError):
    status_code = 409


class DaytonaForbiddenError(DaytonaError):
    status_code = 403


class DaytonaRateLimitError(DaytonaError):
    status_code = 429


class DaytonaBadRequestError(DaytonaError):
    status_code = 400


class DaytonaTimeoutError(DaytonaError):
    sandbox_id = None


class DaytonaQueueTimeoutError(DaytonaTimeoutError):
    pass


DENIED = "Failed to remove sandbox: Access denied"


def model(name):
    """A params class of the SDK: it keeps what it is given."""
    return type(name, (types.SimpleNamespace,), {})


class FakeImage:
    """``daytona.Image``: the Dockerfile lines it is given."""

    def __init__(self, lines):
        self.lines = lines

    @classmethod
    def base(cls, image):
        return cls([f"FROM {image}"])

    def entrypoint(self, argv):
        return FakeImage([*self.lines, f"ENTRYPOINT {argv}"])


class FakeCommand:
    """One session command run for real: the session's shell sources it,
    with stdin held open, and writes its exit code once it ends."""

    def __init__(self, log_dir, cmd_id, line):
        self.cmd_id = cmd_id
        self.log = []
        self.exit_code = None
        self.done = asyncio.Event()
        self.script = log_dir / f"{cmd_id}.sh"
        self.script.write_text(line)
        self.exit_file = log_dir / f"{cmd_id}.exit"

    async def run(self):
        shell = await asyncio.create_subprocess_exec(
            "sh",
            "-c",
            '{ . "$1"; }; echo "$?" > "$2"',
            "session",
            str(self.script),
            str(self.exit_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def relay(stream, reader):
            while line := await reader.readline():
                self.log.append((stream, line.decode()))

        await asyncio.gather(relay(1, shell.stdout), relay(2, shell.stderr))
        await shell.wait()
        if self.exit_file.exists():
            self.exit_code = int(self.exit_file.read_text())
        self.done.set()


class FakeProcessApi:
    def __init__(self, sdk, box):
        self.sdk = sdk
        self.box = box
        self.sessions = set()
        self.commands = {}

    async def create_session(self, session_id):
        self.sdk.calls.append(("session", self.box.id, session_id))
        self.sessions.add(session_id)

    async def execute_session_command(self, session_id, request):
        assert session_id in self.sessions and request.run_async
        command = FakeCommand(
            self.sdk.log_dir, f"c{len(self.sdk.calls)}", request.command
        )
        self.sdk.calls.append(("execute", session_id, request.command))
        self.commands[command.cmd_id] = command
        asyncio.get_running_loop().create_task(command.run())
        return types.SimpleNamespace(cmd_id=command.cmd_id)

    async def get_session_command_logs_async(
        self, session_id, cmd_id, on_stdout, on_stderr
    ):
        """The whole log from its start, followed until the command ends."""
        self.sdk.calls.append(("logs", session_id, cmd_id))
        command = self.commands[cmd_id]
        sent = 0
        while True:
            while sent < len(command.log):
                stream, text = command.log[sent]
                sent += 1
                (on_stdout if stream == 1 else on_stderr)(text)
                if self.sdk.cut_after == sent:
                    self.sdk.cut_after = None
                    raise ConnectionResetError(f"websocket cut ({KEY})")
            if command.done.is_set() and sent == len(command.log):
                return
            await asyncio.sleep(0.01)

    async def get_session_command(self, session_id, cmd_id):
        return types.SimpleNamespace(exit_code=self.commands[cmd_id].exit_code)

    async def delete_session(self, session_id):
        self.sdk.calls.append(("delete_session", session_id))
        self.sessions.discard(session_id)


class FakeFs:
    def __init__(self, sdk):
        self.sdk = sdk
        self.files = {}

    async def upload_files(self, files, timeout):
        for upload in files:
            source = upload.source
            if not isinstance(source, bytes):
                with open(source, "rb") as file:  # as the SDK: a path, or here a fd
                    source = file.read()
            self.files[upload.destination] = source

    async def download_file(self, path):
        if path not in self.files:
            raise DaytonaNotFoundError(f"file {path} not found")
        return self.files[path]

    async def download_file_stream(self, path, timeout):
        data = self.files[path]

        async def chunks():
            try:
                for start in range(0, len(data), 2):
                    yield data[start : start + 2]
            finally:
                self.sdk.calls.append(("stream_closed", path))

        return chunks()


class FakeBox:
    def __init__(self, sdk, sandbox_id, name, labels):
        self.id = sandbox_id
        self.name = name
        self.labels = labels
        self.state = "started"
        self.sdk = sdk
        self.process = FakeProcessApi(sdk, self)
        self.fs = FakeFs(sdk)

    async def delete(self, timeout):
        self.sdk.calls.append(("delete", self.id))
        if not self.sdk.can_delete:
            raise DaytonaForbiddenError(DENIED)
        self.state = "destroyed"

    async def stop(self, timeout, force):
        """An ephemeral sandbox is gone once stopped (unless ``stop_leaves``)."""
        self.sdk.calls.append(("stop", self.id, force))
        if self.sdk.stop_leaves or self.id in self.sdk.stuck:
            self.state = "stopped"
        else:
            self.sdk.boxes.pop(self.id, None)


class FakeDaytona:
    """The slice of ``daytona`` sandbox_daytona calls, as a module."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.calls = []
        self.configs = []
        self.boxes = {}
        self.fail_build = None
        self.cut_after = None
        # A key with write:sandboxes only: no delete, an empty (or refused)
        # listing; ``refuse`` holds the errors the next creates raise, and a
        # create whose answer is lost still makes its sandbox.
        self.can_delete = True
        self.listing = "full"  # "empty" | "refused"
        self.stop_leaves = False
        self.refuse = []
        # Errors raised after the sandbox was made (the SDK's polls after
        # the request); a queue timeout's sandbox the API destroyed first.
        self.refuse_after = []
        # Errors the next gets raise.
        self.fail_get = []
        # Sandbox ids a stop leaves in place (``stop_leaves`` for all).
        self.stuck = set()
        self.lose_answer = False
        # Sandbox id -> reads before a starting one has started.
        self.settle = {}
        ids = itertools.count()
        sdk = self
        module = types.ModuleType("daytona")
        module.DaytonaError = DaytonaError
        module.DaytonaNotFoundError = DaytonaNotFoundError
        module.DaytonaConflictError = DaytonaConflictError
        module.DaytonaForbiddenError = DaytonaForbiddenError
        module.DaytonaRateLimitError = DaytonaRateLimitError
        module.DaytonaTimeoutError = DaytonaTimeoutError
        module.DaytonaQueueTimeoutError = DaytonaQueueTimeoutError
        module.Image = FakeImage
        for name in (
            "DaytonaConfig",
            "Resources",
            "CreateSandboxFromImageParams",
            "ListSandboxesQuery",
            "FileUpload",
            "SessionExecuteRequest",
        ):
            setattr(module, name, model(name))

        class AsyncDaytona:
            def __init__(self, config):
                sdk.configs.append(config)

            async def create(self, params, *, timeout, on_snapshot_create_logs=None):
                sdk.calls.append(("create", params, timeout))
                if sdk.refuse:
                    raise sdk.refuse.pop(0)
                if on_snapshot_create_logs is not None:
                    on_snapshot_create_logs(f"Building {params.image.lines[0]}")
                    if sdk.fail_build is not None:
                        raise sdk.fail_build
                box = FakeBox(
                    sdk,
                    f"sb-{next(ids)}",
                    params.name,
                    {
                        **getattr(params, "labels", {}),
                        "code-toolbox-language": "python",
                    },
                )
                sdk.boxes[box.id] = box
                if sdk.refuse_after:
                    error = sdk.refuse_after.pop(0)
                    if isinstance(error, DaytonaQueueTimeoutError):
                        del sdk.boxes[box.id]
                    raise error
                if sdk.lose_answer:
                    sdk.lose_answer = False
                    raise ConnectionResetError("the answer was lost")
                return box

            async def get(self, id_or_name):
                sdk.calls.append(("get", id_or_name))
                if sdk.fail_get:
                    raise sdk.fail_get.pop(0)
                for box in sdk.boxes.values():
                    if id_or_name in (box.id, box.name):
                        if box.id in sdk.settle:
                            sdk.settle[box.id] -= 1
                            if sdk.settle[box.id] <= 0:
                                del sdk.settle[box.id]
                                box.state = "started"
                        return box
                raise DaytonaNotFoundError(f"sandbox {id_or_name} not found")

            async def list(self, query):
                sdk.calls.append(("list", dict(query.labels)))
                if sdk.listing == "refused":
                    raise DaytonaForbiddenError("Forbidden resource")
                if sdk.listing == "empty":
                    return
                for box in list(sdk.boxes.values()):
                    if query.labels.items() <= box.labels.items():
                        yield box

            async def close(self):
                sdk.calls.append(("close",))

        module.AsyncDaytona = AsyncDaytona
        self.module = module


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    fake = FakeDaytona(tmp_path)
    monkeypatch.setitem(sys.modules, "daytona", fake.module)
    return fake


@pytest.fixture
def client(sdk):
    made = SdkDaytonaClient(KEY, target="us")
    made.build_template("rsi-snap", "busybox", (1, 1, 1), log=lambda line: None)
    sdk.calls.clear()
    yield made
    made.close()


def sandbox(client):
    return client.create(
        "rsi-snap", timeout=661, metadata=METADATA, allow_internet=True
    )


# -- the adapter -----------------------------------------------------------------------


def test_the_client_passes_its_key_url_and_target_and_polls(sdk):
    eu = SdkDaytonaClient(KEY, target="eu")
    us = SdkDaytonaClient(KEY, target="us", api_url="https://daytona.example/api")
    eu.close()
    us.close()
    first, second = sdk.configs
    # Passed, so the SDK reads none of them from the environment or .env;
    # no pool limit, which every followed log would hold a slot of.
    assert vars(first) == {
        "api_key": KEY,
        "api_url": "https://app.daytona.io/api",
        "target": "eu",
        "use_deprecated_polling": True,
        "connection_pool_maxsize": None,
    }
    assert (second.api_url, second.target) == ("https://daytona.example/api", "us")
    assert sdk.calls == [("close",), ("close",)]
    # An image is built in its region: each target names its own.
    shape = sandbox_shape(1, 1024, 1024)
    assert template_name("rsi", "busybox", shape, eu.recipe) != template_name(
        "rsi", "busybox", shape, us.recipe
    )


def test_a_template_is_the_image_built_once_by_a_sandbox_of_its_own(sdk):
    assert sandbox_shape(0.5, 1000, 10240) == (1, 1, 10)
    assert sandbox_shape(2.5, 4097, 1) == (3, 5, 1)
    client = SdkDaytonaClient(KEY, target="us")
    assert client.template_exists("rsi-img") is False
    lines = []
    client.build_template(
        "rsi-img", "docker.io/library/busybox:1.37", (2, 4, 10), log=lines.append
    )
    (create,) = [call for call in sdk.calls if call[0] == "create"]
    _, params, timeout = create
    assert params.image.lines == [
        "FROM docker.io/library/busybox:1.37",
        "ENTRYPOINT ['sleep', 'infinity']",
    ]
    assert vars(params.resources) == {"cpu": 2, "memory": 4, "disk": 10}
    assert params.name.startswith("rsi-img-") and params.ephemeral
    assert (params.auto_stop_interval, params.network_block_all) == (1, True)
    # It waits for the build, then deletes its sandbox.
    assert (
        timeout == 0 and lines[-1] == "Building FROM docker.io/library/busybox:1.37\n"
    )
    assert ("delete", "sb-0") in sdk.calls
    assert client.template_exists("rsi-img") is True
    # Sandboxes of the template are made from the image with its shape.
    client.create("rsi-img", timeout=60, metadata=METADATA, allow_internet=True)
    _, made, _ = [call for call in sdk.calls if call[0] == "create"][-1]
    assert made.image.lines == params.image.lines
    assert vars(made.resources) == {"cpu": 2, "memory": 4, "disk": 10}
    sdk.fail_build = DaytonaError(f"Failed to build, reason: {KEY}")
    with pytest.raises(InfrastructureError) as failed:
        client.build_template("broken", "busybox", (1, 1, 1), log=lines.append)
    assert KEY not in str(failed.value) and "***" in str(failed.value)
    assert client.template_exists("broken") is False
    client.close()


@pytest.mark.parametrize(
    ("allow_internet", "allow_out", "network"),
    [
        (True, (), {}),
        (False, (), {"network_block_all": True}),
        (
            False,
            ("1.1.1.1", "10.0.0.0/8"),
            {"network_allow_list": "1.1.1.1/32,10.0.0.0/8"},
        ),
        (
            False,
            ("example.com", "pypi.org"),
            {"domain_allow_list": "example.com,pypi.org"},
        ),
    ],
)
def test_create_names_labels_and_fences_an_ephemeral_root_sandbox(
    sdk, client, allow_internet, allow_out, network
):
    sandbox_id = client.create(
        "rsi-snap",
        timeout=661,
        metadata=METADATA,
        allow_internet=allow_internet,
        allow_out=allow_out,
    )
    ((_, params, timeout),) = [call for call in sdk.calls if call[0] == "create"]
    # The TTL and the auto-stop are both the broker's timeout, in minutes.
    assert params.image.lines == ["FROM busybox", "ENTRYPOINT ['sleep', 'infinity']"]
    assert vars(params.resources) == {"cpu": 1, "memory": 1, "disk": 1}
    assert {
        k: v for k, v in vars(params).items() if k not in ("image", "resources")
    } == {
        "name": f"rsi-{ENV_ID}",
        "os_user": "root",
        "labels": METADATA,
        "public": False,
        "ephemeral": True,
        "ttl_minutes": 12,
        "auto_stop_interval": 12,
        "queue_timeout": sandbox_daytona.QUEUE_MIN,
        **network,
    }
    assert timeout == sandbox_daytona.CREATE_SEC
    assert sdk.boxes[sandbox_id].name == f"rsi-{ENV_ID}"


def test_state_listing_lookup_and_kill(sdk, client):
    sandbox_id = sandbox(client)
    other = dict(METADATA, rsi_run_id="run-2", rsi_env_id="e" + "b" * 32)
    second = client.create("rsi-snap", timeout=60, metadata=other, allow_internet=True)
    # Only the broker's own labels come back.
    assert client.list({"rsi_run_id": "run-1"}) == [(sandbox_id, METADATA)]
    assert client.lookup(ENV_ID) == sandbox_id
    assert client.lookup("e" + "c" * 32) is None
    assert client.state(sandbox_id) == "running"
    sdk.boxes[sandbox_id].state = "starting"
    assert client.state(sandbox_id) == "starting"
    client.kill(sandbox_id)
    assert ("delete", sandbox_id) in sdk.calls
    assert client.state(sandbox_id) is None
    assert client.lookup(ENV_ID) is None
    assert client.list({"rsi_run_id": "run-1"}) == []
    # A sandbox that cannot run again is deleted and waited out, not gone.
    sdk.boxes[second].state = "error"
    assert client.state(second) == "error"
    assert ("delete", second) in sdk.calls
    assert client.state(second) is None
    assert client.state("sb-unknown") is None
    client.kill("sb-unknown")


def test_a_process_is_one_child_of_its_session_shell(sdk, client):
    sandbox_id = sandbox(client)
    script = 'echo "$A in $PWD"; cat; echo err >&2; exit 3'
    process = start_group(
        client,
        sandbox_id,
        ["sh", "-c", script],
        envs={"A": "it's"},
        cwd="/tmp",
        user="root",
    )
    assert process.pid > 1
    events = list(process)
    # stdin is closed (cat ends) though the session holds it open.
    assert sorted(events) == [(1, b"it's in /tmp\n"), (2, b"err\n")]
    # The session's shell wrote the exit: the command never replaced it.
    assert process.exit_code == 3
    ((_, session_id, line),) = [call for call in sdk.calls if call[0] == "execute"]
    assert line.startswith("sh -c ")
    assert ("delete_session", session_id) in sdk.calls
    assert client._commands == {}


def test_any_variable_name_reaches_the_command(sdk, client):
    """As on Docker, a name that is no shell name (a.b, my-var) is set."""
    sandbox_id = sandbox(client)
    process = client.start(
        sandbox_id,
        ["env"],
        envs={"spring.profiles.active": "prod", "my-var": "x", "A": "it's"},
        cwd="/",
        user="root",
    )
    out = b"".join(data for stream, data in process if stream == 1).decode()
    assert process.exit_code == 0
    assert {"spring.profiles.active=prod", "my-var=x", "A=it's"} <= set(
        out.splitlines()
    )


def test_a_cut_log_is_followed_again_without_repeating_output(sdk, client):
    sandbox_id = sandbox(client)
    sdk.cut_after = 2  # the pid line and "one"
    process = start_group(
        client,
        sandbox_id,
        ["sh", "-c", "echo one; sleep 0.2; echo two; echo three >&2"],
        envs={},
        cwd="/",
        user="root",
    )
    events = []
    with pytest.raises(InfrastructureError) as cut:
        events.extend(process)
    assert KEY not in str(cut.value)
    # The session stays until its exit is seen.
    assert not [call for call in sdk.calls if call[0] == "delete_session"]
    process.close()
    again = client.connect(sandbox_id, process.pid)
    events.extend(again)
    assert events == [(1, b"one\n"), (1, b"two\n"), (2, b"three\n")]
    assert again.exit_code == 0
    assert [call[0] for call in sdk.calls].count("delete_session") == 1
    with pytest.raises(InfrastructureError, match="no command of pid"):
        client.connect(sandbox_id, process.pid)


def test_closing_a_process_stops_following_and_keeps_its_session(sdk, client):
    sandbox_id = sandbox(client)
    process = client.start(sandbox_id, ["sleep", "2"], envs={}, cwd="/", user="root")
    process.close()
    assert list(process) == [] and process.exit_code is None
    client.close()
    assert not [call for call in sdk.calls if call[0] == "delete_session"]
    with pytest.raises(InfrastructureError, match="closed"):
        client.state(sandbox_id)


def test_files_are_uploaded_downloaded_and_streamed(sdk, client, tmp_path):
    sandbox_id = sandbox(client)
    fs = sdk.boxes[sandbox_id].fs
    client.write_file(sandbox_id, "/tmp/a", b"bytes")
    client.write_file(sandbox_id, "/tmp/b", io.BytesIO(b"buffer"))
    stage = tmp_path / "stage.tar"
    stage.write_bytes(b"x" * 100)
    with open(stage, "rb") as file:
        client.write_file(sandbox_id, "/tmp/stage.tar", file)
        assert file.read(1) == b""  # the upload read a duplicate, now closed
    assert fs.files == {
        "/tmp/a": b"bytes",
        "/tmp/b": b"buffer",
        "/tmp/stage.tar": b"x" * 100,
    }
    assert client.read_file(sandbox_id, "/tmp/a") == b"bytes"
    with pytest.raises(FileNotFoundError):
        client.read_file(sandbox_id, "/missing")
    with client.read_stream(sandbox_id, "/tmp/b") as stream:
        assert stream.read() == b"buffer"
    with client.read_stream(sandbox_id, "/tmp/stage.tar") as stream:
        assert stream.read(3) == b"xx"
    assert [call for call in sdk.calls if call[0] == "stream_closed"] == [
        ("stream_closed", "/tmp/b"),
        ("stream_closed", "/tmp/stage.tar"),
    ]


# -- a key with write:sandboxes only ---------------------------------------------------


@pytest.fixture
def quick(monkeypatch):
    """No waiting between the reads that prove a stop."""
    monkeypatch.setattr(sandbox_daytona, "STOP_POLL_SEC", 0)
    monkeypatch.setattr(sandbox_daytona, "GONE_SEC", 0.2)
    monkeypatch.setattr(sandbox_daytona, "CAPACITY_RETRY_SEC", (0, 0))
    monkeypatch.setattr(sandbox_daytona, "LANDED_RETRY_SEC", (0, 0, 0))


def ledger_of(tmp_path, run_id="run-1", clock=None):
    options = {} if clock is None else {"clock": clock}
    return SandboxLedger(
        tmp_path / "leases" / f"{run_id}.sandboxes.jsonl", run_id, **options
    )


def bound(sdk, tmp_path, **options):
    """A client with the run's ledger, its template built, calls cleared."""
    made = SdkDaytonaClient(KEY, target="us")
    made.ledger = ledger_of(tmp_path, **options)
    made.build_template("rsi-snap", "busybox", (1, 1, 1), log=lambda line: None)
    sdk.calls.clear()
    return made


def other_env(sdk, client, letter):
    metadata = dict(METADATA, rsi_env_id="e" + letter * 32)
    return client.create("rsi-snap", timeout=60, metadata=metadata, allow_internet=True)


def calls(sdk, *names):
    return [call for call in sdk.calls if call[0] in names]


def test_a_key_that_may_not_delete_stops_and_proves_the_end(sdk, tmp_path, quick):
    sdk.can_delete = False
    client = SdkDaytonaClient(KEY, target="us")
    client.ledger = ledger_of(tmp_path)
    try:
        client.build_template("rsi-snap", "busybox", (1, 1, 1), log=lambda line: None)
        # The template's build sandbox: the refused delete, then a forced
        # stop, then a read that finds nothing.
        probe = sdk.calls[0][1].name
        assert calls(sdk, "delete", "stop") == [
            ("delete", "sb-0"),
            ("stop", "sb-0", True),
        ]
        assert sdk.calls[-1] == ("get", "sb-0")
        assert client.template_exists("rsi-snap")
        assert client.ledger.open_entries() == []
        sdk.calls.clear()
        first = sandbox(client)
        second = other_env(sdk, client, "b")
        entry = client.ledger.open_entries({"rsi_env_id": ENV_ID})[0]
        assert (entry.name, entry.sandbox_id) == (f"rsi-{ENV_ID}", first)
        assert entry.metadata == METADATA
        # From then on it stops without asking to delete.
        client.kill(first)
        client.kill(second)
        assert calls(sdk, "delete", "stop") == [
            ("stop", first, True),
            ("stop", second, True),
        ]
        assert client.state(first) is None
        assert client.ledger.open_entries() == []
        # Each recorded as it was made and as it ended, beside the lease,
        # the build sandbox under the run but no env.
        records = [
            json.loads(line) for line in client.ledger.path.read_text().splitlines()
        ]
        assert [(r["event"], r["name"]) for r in records if "name" in r][:2] == [
            ("create", probe),
            ("created", probe),
        ]
        assert records[0]["metadata"] == {
            "rsi_run_id": "run-1",
            "rsi_role": "template-build",
        }
        events = [record["event"] for record in records]
        assert events.count("create") == 3 and events.count("end") == 3
        assert KEY not in client.ledger.path.read_text()
    finally:
        client.close()


def test_a_key_that_may_delete_never_stops_and_keeps_the_ledger_short(
    sdk, tmp_path, quick
):
    client = bound(sdk, tmp_path)
    try:
        sandbox_id = sandbox(client)
        client.kill(sandbox_id)
        assert calls(sdk, "delete", "stop") == [("delete", sandbox_id)]
        assert client.ledger.open_entries() == []
        # The listing alone answers when the key may list.
        second = other_env(sdk, client, "b")
        assert client.list({"rsi_run_id": "run-1"}) == [
            (second, dict(METADATA, rsi_env_id="e" + "b" * 32))
        ]
        assert ("get", second) not in sdk.calls
    finally:
        client.close()


def test_a_sandbox_still_starting_is_waited_for_before_its_stop(sdk, tmp_path, quick):
    sdk.can_delete = False
    client = bound(sdk, tmp_path)
    try:
        sandbox_id = sandbox(client)
        sdk.boxes[sandbox_id].state = "pending_build"
        sdk.settle[sandbox_id] = 3
        client.kill(sandbox_id)
        stop = sdk.calls.index(("stop", sandbox_id, True))
        # Read until it started, and only then stopped.
        assert sdk.calls[:stop].count(("get", sandbox_id)) >= 3
        assert sandbox_id not in sdk.boxes
    finally:
        client.close()


def test_a_sandbox_left_after_its_stop_is_reported_and_stays_in_the_ledger(
    sdk, tmp_path, quick
):
    sdk.can_delete = False
    client = bound(sdk, tmp_path)
    sdk.stop_leaves = True
    try:
        sandbox_id = sandbox(client)
        with pytest.raises(InfrastructureError) as left:
            client.kill(sandbox_id)
        assert sandbox_id in str(left.value) and "state stopped" in str(left.value)
        assert KEY not in str(left.value)
        (entry,) = client.ledger.open_entries()
        assert (entry.sandbox_id, entry.note) == (sandbox_id, "leftover (stopped)")
        # Recovery finds it again, from the ledger alone.
        sdk.listing = "empty"
        assert client.list({"rsi_run_id": "run-1"}) == [(sandbox_id, METADATA)]
        # The provider removes it at last: the next read ends its entry.
        del sdk.boxes[sandbox_id]
        assert client.list({"rsi_run_id": "run-1"}) == []
        assert client.ledger.open_entries() == []
    finally:
        client.close()


@pytest.mark.parametrize("listing", ["empty", "refused"])
def test_without_a_listing_recovery_kills_the_runs_sandboxes_from_the_ledger(
    sdk, tmp_path, quick, listing
):
    sdk.can_delete = False
    sdk.listing = listing
    engine = bound(sdk, tmp_path)
    first = sandbox(engine)
    second = other_env(sdk, engine, "b")
    # A create whose answer was lost: the ledger has only its name.
    sdk.lose_answer = True
    with pytest.raises(InfrastructureError, match="answer was lost"):
        other_env(sdk, engine, "c")
    lost = next(box.id for box in sdk.boxes.values() if box.name.endswith("c" * 32))
    # An env's kill finds an unanswered create by the ledger's name too.
    assert engine.list({"rsi_env_id": "e" + "c" * 32}) == [
        (lost, dict(METADATA, rsi_env_id="e" + "c" * 32))
    ]
    engine.close()  # the Engine died; another run's sandbox lives beside
    stranger = SdkDaytonaClient(KEY, target="us")
    stranger.ledger = ledger_of(tmp_path, "run-2")
    stranger._templates["rsi-snap"] = ("busybox", (1, 1, 1))
    theirs = stranger.create(
        "rsi-snap",
        timeout=60,
        metadata=dict(METADATA, rsi_run_id="run-2", rsi_env_id="e" + "d" * 32),
        allow_internet=True,
    )
    stranger.close()

    recovery = SdkDaytonaClient(KEY, target="us")
    recovery.ledger = ledger_of(tmp_path)
    try:
        killed = kill_run_sandboxes(recovery, "run-1", poll=0)
        assert sorted(killed) == sorted([first, second, lost])
        assert list(sdk.boxes) == [theirs]
        assert recovery.ledger.open_entries() == []
        # A second pass finds nothing and still proves it.
        assert kill_run_sandboxes(recovery, "run-1", poll=0) == ()
    finally:
        recovery.close()


def test_a_refused_listing_without_a_ledger_still_fails(sdk, client):
    sdk.listing = "refused"
    with pytest.raises(InfrastructureError, match="Forbidden"):
        client.list({"rsi_run_id": "run-1"})


def test_an_unanswered_create_that_never_landed_ends_after_a_while(
    sdk, tmp_path, quick
):
    now = [1000.0]
    client = bound(sdk, tmp_path, clock=lambda: now[0])
    try:
        sdk.listing = "empty"
        client.ledger.creating("rsi-e" + "f" * 32, METADATA)
        # Still young: its create may yet land, so it stays open.
        assert client.list({"rsi_run_id": "run-1"}) == []
        assert len(client.ledger.open_entries()) == 1
        now[0] += sandbox_daytona.PENDING_SEC
        assert client.list({"rsi_run_id": "run-1"}) == []
        assert client.ledger.open_entries() == []
    finally:
        client.close()


@pytest.mark.parametrize(
    "refusal",
    [
        DaytonaRateLimitError("Too Many Requests"),
        DaytonaForbiddenError("Total CPU limit exceeded. Maximum allowed: 40"),
        DaytonaForbiddenError(f"Organization quota reached ({KEY})"),
        DaytonaBadRequestError("Failed to create sandbox: No available runners"),
    ],
)
def test_a_capacity_refusal_is_typed_and_nothing_stays_in_the_ledger(
    sdk, tmp_path, quick, refusal
):
    client = bound(sdk, tmp_path)
    try:
        sdk.refuse = [refusal]
        with pytest.raises(ProviderCapacityError) as refused:
            sandbox(client)
        assert "capacity" in str(refused.value) and KEY not in str(refused.value)
        assert client.ledger.open_entries() == []
        # Proven by a read of its name that found nothing.
        assert ("get", f"rsi-{ENV_ID}") in sdk.calls
        # The next create, with room again, succeeds, and is recorded.
        assert client.lookup(ENV_ID) is None
        sandbox_id = sandbox(client)
        assert sandbox_id in sdk.boxes
        (entry,) = client.ledger.open_entries()
        assert entry.sandbox_id == sandbox_id
    finally:
        client.close()


@pytest.mark.parametrize(
    "refusal",
    [
        DaytonaForbiddenError("CPU request 16 exceeds maximum allowed per sandbox (4)"),
        DaytonaForbiddenError("Forbidden resource"),
        DaytonaError("Failed to build, reason: quota of layers"),
        DaytonaForbiddenError("Disk capacity must be at most 10GiB"),
        DaytonaForbiddenError("Memory exceeds the maximum limit"),
        DaytonaForbiddenError("Invalid quota value"),
    ],
)
def test_other_refusals_are_not_capacity(sdk, client, refusal):
    sdk.refuse = [refusal]
    with pytest.raises(InfrastructureError) as refused:
        sandbox(client)
    assert not isinstance(refused.value, ProviderCapacityError)


def test_a_refusal_after_the_sandbox_landed_is_no_capacity_and_stays_found(
    sdk, tmp_path, quick
):
    sdk.can_delete = False
    sdk.listing = "empty"
    client = bound(sdk, tmp_path)
    try:
        # The request made the sandbox, then a poll while it started was
        # rate limited: the error carries no id, but a read finds it.
        sdk.refuse_after = [DaytonaRateLimitError("Too Many Requests")]
        with pytest.raises(InfrastructureError) as refused:
            sandbox(client)
        assert not isinstance(refused.value, ProviderCapacityError)
        ((landed, box),) = sdk.boxes.items()
        assert landed in str(refused.value) and KEY not in str(refused.value)
        (entry,) = client.ledger.open_entries()
        assert (entry.name, entry.sandbox_id) == (box.name, landed)
        # An env's kill and the run's recovery both find it.
        assert client.list({"rsi_env_id": ENV_ID}) == [(landed, METADATA)]
        assert kill_run_sandboxes(client, "run-1", poll=0) == (landed,)
        assert sdk.boxes == {} and client.ledger.open_entries() == []
    finally:
        client.close()


def test_a_refusal_whose_landing_cannot_be_read_stays_open(sdk, tmp_path, quick):
    client = bound(sdk, tmp_path)
    try:
        sdk.refuse = [DaytonaRateLimitError("Too Many Requests")]
        sdk.fail_get = [DaytonaRateLimitError("Too Many Requests")] * 4
        with pytest.raises(InfrastructureError, match="landed is unknown") as left:
            sandbox(client)
        assert not isinstance(left.value, ProviderCapacityError)
        assert [call for call in sdk.calls if call[0] == "get"] == [
            ("get", f"rsi-{ENV_ID}")
        ] * 4
        # Unproven: the entry stays open for recovery to read again.
        (entry,) = client.ledger.open_entries()
        assert (entry.name, entry.sandbox_id) == (f"rsi-{ENV_ID}", None)
        # A read that recovers on a later try still proves it.
        sdk.refuse = [DaytonaRateLimitError("Too Many Requests")]
        sdk.fail_get = [DaytonaRateLimitError("Too Many Requests")] * 2
        with pytest.raises(ProviderCapacityError):
            other_env(sdk, client, "b")
    finally:
        client.close()


def test_a_queue_timeout_is_capacity_and_creates_set_one(sdk, tmp_path, quick):
    client = bound(sdk, tmp_path)
    try:
        sdk.refuse_after = [
            DaytonaQueueTimeoutError("Sandbox sb-9 was destroyed after waiting")
        ]
        with pytest.raises(ProviderCapacityError):
            sandbox(client)
        # The API destroyed it: nothing is live, nothing stays open.
        assert [box for box in sdk.boxes.values() if box.state != "destroyed"] == []
        assert client.ledger.open_entries() == []
        params = [call[1] for call in sdk.calls if call[0] == "create"]
        assert params[0].queue_timeout == sandbox_daytona.QUEUE_MIN
        assert sandbox_daytona.QUEUE_MIN * 60 < sandbox_daytona.CREATE_SEC
    finally:
        client.close()


def test_recovery_kills_the_rest_when_one_sandbox_is_left(sdk, tmp_path, quick):
    sdk.can_delete = False
    sdk.listing = "empty"
    client = bound(sdk, tmp_path)
    try:
        first = sandbox(client)
        second = other_env(sdk, client, "b")
        sdk.stuck = {first}
        with pytest.raises(InfrastructureError) as left:
            kill_run_sandboxes(client, "run-1", poll=0, wait_sec=0)
        assert first in str(left.value) and "still there" in str(left.value)
        assert second not in str(left.value)
        # The leftover did not stop the kill of the sandbox after it.
        assert list(sdk.boxes) == [first]
        (entry,) = client.ledger.open_entries()
        assert entry.sandbox_id == first
    finally:
        client.close()


def test_a_build_sandbox_waits_for_capacity_then_gives_up(
    sdk, tmp_path, quick, monkeypatch
):
    client = bound(sdk, tmp_path)
    try:
        lines = []
        sdk.refuse = [DaytonaRateLimitError("Too Many Requests")] * 2
        client.build_template("rsi-wait", "busybox", (1, 1, 1), log=lines.append)
        assert client.template_exists("rsi-wait")
        assert sum("at capacity" in line for line in lines) == 2
        monkeypatch.setattr(sandbox_daytona, "CAPACITY_WAIT_SEC", 0)
        sdk.refuse = [DaytonaRateLimitError("Too Many Requests")]
        with pytest.raises(ProviderCapacityError):
            client.build_template("rsi-full", "busybox", (1, 1, 1), log=lines.append)
        assert not client.template_exists("rsi-full")
        assert client.ledger.open_entries() == []
    finally:
        client.close()


def test_a_build_sandbox_without_an_answer_is_recovered_by_its_name(
    sdk, tmp_path, quick
):
    sdk.can_delete = False
    sdk.listing = "empty"
    client = bound(sdk, tmp_path)
    try:
        sdk.lose_answer = True
        with pytest.raises(InfrastructureError, match="answer was lost"):
            client.build_template("rsi-lost", "busybox", (1, 1, 1), log=print)
        ((lost, box),) = sdk.boxes.items()
        (entry,) = client.ledger.open_entries()
        assert (entry.name, entry.sandbox_id) == (box.name, None)
        # Only the run's recovery finds it (an env's kill does not).
        assert client.list({"rsi_env_id": ENV_ID}) == []
        assert kill_run_sandboxes(client, "run-1", poll=0) == (lost,)
        assert sdk.boxes == {}
        assert client.ledger.open_entries() == []
    finally:
        client.close()


# -- the key and the host check --------------------------------------------------------


def test_the_key_is_read_from_a_file_or_a_variable_never_echoed(tmp_path):
    path = tmp_path / "daytona.key"
    path.write_text(KEY + "\n")
    assert read_api_key(EnvDaytonaHost(api_key_file=str(path), target="us")) == KEY
    settings = EnvDaytonaHost(api_key_env="DAYTONA_KEY", target="us")
    assert read_api_key(settings, {"DAYTONA_KEY": KEY}) == KEY
    with pytest.raises(SetupError, match="holds no key"):
        read_api_key(settings, {"DAYTONA_KEY": f"{KEY} {KEY}"})
    path.unlink()
    with pytest.raises(SetupError, match="unreadable"):
        read_api_key(EnvDaytonaHost(api_key_file=str(path), target="us"))


_VERSION = importlib.metadata.version


def sdk_installed(monkeypatch, version=SDK_VERSION):
    """Package metadata reports daytona at ``version`` (None: missing)."""

    def fake(name):
        if name == "daytona" and version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return version if name == "daytona" else _VERSION(name)

    monkeypatch.setattr(importlib.metadata, "version", fake)


def test_the_submit_host_check_needs_the_pinned_sdk_and_the_key(monkeypatch, tmp_path):
    settings = EnvDaytonaHost(api_key_env="RSI_T_DAYTONA", target="us")
    monkeypatch.setenv("RSI_T_DAYTONA", KEY)
    for version, found in ((None, "none"), ("0.226.0", "0.226.0")):
        sdk_installed(monkeypatch, version)
        with pytest.raises(
            SetupError, match=rf"daytona=={SDK_VERSION} \(found {found}"
        ):
            check_daytona_host(settings)
    sdk_installed(monkeypatch)
    # The SDK's own variables redirect nothing: the policy passes them all.
    monkeypatch.setenv("DAYTONA_API_URL", "https://elsewhere/api")
    monkeypatch.setenv("DAYTONA_TARGET", "eu")
    check_daytona_host(settings)
    auth = tmp_path / "auth.json"
    auth.write_text("{")
    with pytest.raises(SetupError, match="registry_auth_file"):
        check_daytona_host(
            settings.model_copy(update={"registry_auth_file": str(auth)})
        )
    monkeypatch.setenv("RSI_T_DAYTONA", "")
    with pytest.raises(SetupError, match="holds no key"):
        check_daytona_host(settings)


def test_the_daytona_extra_pins_the_checked_sdk_version():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    extra = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    assert f"daytona=={SDK_VERSION}" in extra["daytona"]


# -- the proxy, over the real SDK ------------------------------------------------------


@pytest.fixture
def connect_proxy(monkeypatch, tmp_path):
    pytest.importorskip("daytona")
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env of the checkout
    proxy = ConnectProxy()
    yield proxy
    proxy.close()


def real_client(**options):
    return SdkDaytonaClient(
        KEY, target="us", api_url="https://api.daytona.example/api", **options
    )


def toolbox_box(client):
    """A started sandbox as the API would describe it, cached in ``client``."""
    from daytona._async.sandbox import AsyncSandbox
    from daytona_api_client_async.models import Sandbox

    dto = Sandbox(
        id="sb-1",
        organization_id="org",
        name="rsi-sb",
        user="root",
        env={},
        labels={},
        public=False,
        network_block_all=False,
        kvm=False,
        target="us",
        cpu=1,
        gpu=0,
        memory=1,
        disk=1,
        state="started",
        toolbox_proxy_url="https://toolbox.daytona.example/toolbox",
    )
    sdk = client._sdk

    async def make():
        return AsyncSandbox(
            dto,
            sdk._toolbox_api_client,
            sdk._sandbox_api,
            "python",
            sdk._subscription_manager,
        )

    client._boxes["sb-1"] = client._call(make())
    return client._boxes["sb-1"]


def test_every_daytona_connection_goes_through_the_policy_proxy(connect_proxy):
    environ = dict(os.environ)
    client = real_client(proxy=connect_proxy.url)
    try:
        assert dict(os.environ) == environ  # nothing the Engine's children see
        box = toolbox_box(client)
        followed = sandbox_daytona._Command("sb-1", "s-1", "c-1")
        for attempt in (
            lambda: client.state("sb-2"),  # the API
            lambda: client.build_template("rsi-img", "busybox", (1, 1, 1), log=print),
            lambda: client.start("sb-1", ["true"], envs={}, cwd="/", user="root"),
            lambda: list(client._process(box, followed)),  # the log websocket
            lambda: client.write_file("sb-1", "/tmp/x", io.BytesIO(b"x")),
            lambda: client.read_file("sb-1", "/etc/passwd"),
            lambda: client.read_stream("sb-1", "/tmp/x").read(),
        ):
            with pytest.raises((InfrastructureError, OSError), match="403"):
                attempt()
        # No limit: a followed log holds its connection while it runs.
        assert client._sdk._shared_session._session.connector.limit is None
    finally:
        client.close()
    assert (
        connect_proxy.targets
        == ["CONNECT api.daytona.example:443 HTTP/1.1"] * 2
        + ["CONNECT toolbox.daytona.example:443 HTTP/1.1"] * 5
    )


def test_without_a_policy_proxy_the_sdks_own_lookup_answers(connect_proxy, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", connect_proxy.url)
    client = real_client()
    try:
        with pytest.raises(InfrastructureError, match="403"):
            client.state("sb-2")
    finally:
        client.close()
    assert connect_proxy.targets == ["CONNECT api.daytona.example:443 HTTP/1.1"]
