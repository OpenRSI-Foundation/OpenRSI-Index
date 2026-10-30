"""The Daytona SDK adapter (sandbox_daytona) over a fake async ``daytona``
module, and its proxy over the real pinned SDK and a local CONNECT proxy.

The fake runs each session command as the daemon does: the session's shell
sources the command (``{ . file; }``) and writes its exit code afterwards,
with stdin held open, and the log follows line by line. The broker over
Daytona's flags runs in tests/runtime/test_sandbox_e2b.py's generic tests.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import io
import os
import sys
import tomllib
import types
from pathlib import Path

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime import sandbox_daytona
from rsi_harness.runtime.sandbox_contracts import EnvDaytonaHost, SandboxOwner
from rsi_harness.runtime.sandbox_daytona import (
    SDK_VERSION,
    SdkDaytonaClient,
    check_daytona_host,
    read_api_key,
    sandbox_shape,
)
from rsi_harness.runtime.sandbox_remote import (
    sandbox_metadata,
    start_group,
    template_name,
)
from tests.runtime.test_sandbox_modal import ConnectProxy

KEY = "dtn_" + "5ec" * 8
ENV_ID = "e" + "a" * 32
METADATA = sandbox_metadata(
    SandboxOwner(run_id="run-1", task_id="task", phase="work"), ENV_ID, "main"
)


# -- a fake daytona module -------------------------------------------------------------


class DaytonaError(Exception):
    pass


class DaytonaNotFoundError(DaytonaError):
    pass


class DaytonaConflictError(DaytonaError):
    pass


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
        self.state = "destroyed"


class FakeDaytona:
    """The slice of ``daytona`` sandbox_daytona calls, as a module."""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.calls = []
        self.configs = []
        self.boxes = {}
        self.fail_build = None
        self.cut_after = None
        sdk = self
        module = types.ModuleType("daytona")
        module.DaytonaError = DaytonaError
        module.DaytonaNotFoundError = DaytonaNotFoundError
        module.DaytonaConflictError = DaytonaConflictError
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
                if on_snapshot_create_logs is not None:
                    on_snapshot_create_logs(f"Building {params.image.lines[0]}")
                    if sdk.fail_build is not None:
                        raise sdk.fail_build
                box = FakeBox(
                    sdk,
                    f"sb-{len(sdk.boxes)}",
                    params.name,
                    {
                        **getattr(params, "labels", {}),
                        "code-toolbox-language": "python",
                    },
                )
                sdk.boxes[box.id] = box
                return box

            async def get(self, id_or_name):
                for box in sdk.boxes.values():
                    if id_or_name in (box.id, box.name):
                        return box
                raise DaytonaNotFoundError(f"sandbox {id_or_name} not found")

            async def list(self, query):
                sdk.calls.append(("list", dict(query.labels)))
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
