"""The Modal SDK adapter (sandbox_modal) over a fake ``modal`` module, and
its proxy hook over the real pinned SDK and a local CONNECT proxy.

The broker over Modal's flags (emulated pids, users and freeze, exact
shapes, Dockerfile lines) runs in tests/runtime/test_sandbox_e2b.py's
generic tests, parametrized over e2b and modal.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import io
import os
import socket
import sys
import threading
import tomllib
import types
from pathlib import Path

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime import sandbox_modal
from rsi_harness.runtime.sandbox_contracts import EnvModalHost, SandboxOwner
from rsi_harness.runtime.sandbox_modal import (
    SDK_VERSION,
    SdkModalClient,
    check_modal_host,
    read_tokens,
    refuse_modal_environ,
)
from rsi_harness.runtime.sandbox_remote import (
    reattach,
    sandbox_metadata,
    start_group,
)

TOKENS = ("ak-" + "1d" * 8, "as-" + "5ec" * 8)
METADATA = sandbox_metadata(
    SandboxOwner(run_id="run-1", task_id="task", phase="work"), "e" + "a" * 32, "main"
)


# -- a fake modal module ---------------------------------------------------------------


class NotFoundError(Exception):
    pass


class SandboxFilesystemNotFoundError(Exception):
    pass


class FakeImage:
    def __init__(self, sdk, ops):
        self.sdk = sdk
        self.ops = ops

    def dockerfile_commands(self, lines, *, context_dir=None, force_build=False):
        return FakeImage(
            self.sdk,
            [*self.ops, ("dockerfile", tuple(lines), context_dir, force_build)],
        )

    def entrypoint(self, argv):
        return FakeImage(self.sdk, [*self.ops, ("entrypoint", tuple(argv))])

    def build(self, app):
        self.sdk.calls.append(("build", tuple(self.ops), app.app_id))
        if self.sdk.fail_build is not None:
            raise self.sdk.fail_build
        return self


class FakeWriter:
    def __init__(self):
        self.data = bytearray()
        self.eof = False

    def write(self, chunk):
        self.data += chunk

    def write_eof(self):
        self.eof = True

    def drain(self):
        pass


class FakeReader:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __iter__(self):
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def read(self):
        return b"".join(self.chunks)


class FakeExec:
    def __init__(self, stdout=(), stderr=(), code=0):
        self.stdout = FakeReader(stdout)
        self.stderr = FakeReader(stderr)
        self.stdin = FakeWriter()
        self.code = code

    def wait(self):
        return self.code


class FakeBox:
    def __init__(self, sdk, object_id, tags, name):
        self.sdk = sdk
        self.object_id = object_id
        self.tags = dict(tags)
        self.name = name
        self.code = None
        self.files = {}
        self.execs = []
        self.filesystem = types.SimpleNamespace(read_bytes=self.read_bytes)

    def read_bytes(self, path):
        if path not in self.files:
            raise SandboxFilesystemNotFoundError(f"NotFound: {path}")
        return self.files[path]

    def poll(self):
        return self.code

    def terminate(self):
        self.sdk.calls.append(("terminate", self.object_id))
        self.code = 137

    def get_tags(self):
        return dict(self.tags)

    def exec(self, *argv, env=None, workdir=None, text=True):
        self.sdk.calls.append(("exec", argv, dict(env or {}), workdir, text))
        process = self.sdk.answers.pop(0) if self.sdk.answers else FakeExec()
        self.execs.append((argv, process))
        return process


class FakeModal:
    """The slice of ``modal`` sandbox_modal calls, as modules."""

    def __init__(self):
        self.calls = []
        self.boxes = {}
        self.answers = []
        self.fail_build = None
        self.fail_create = None
        sdk = self
        modal = types.ModuleType("modal")
        config = types.ModuleType("modal.config")
        config._user_config = {"profile": {"server_url": "https://elsewhere"}}
        exception = types.ModuleType("modal.exception")
        exception.NotFoundError = NotFoundError
        exception.SandboxFilesystemNotFoundError = SandboxFilesystemNotFoundError
        utils = types.ModuleType("modal._utils")
        proxy_support = types.ModuleType("modal._utils.proxy_support")
        proxy_support.get_proxy_url = lambda host, *, use_ssl: None
        utils.proxy_support = proxy_support
        modal.config, modal.exception, modal._utils = config, exception, utils

        class Client:
            @staticmethod
            def from_credentials(token_id, token_secret):
                sdk.calls.append(("client", token_id, token_secret))
                return "client"

        class App:
            @staticmethod
            def lookup(name, *, create_if_missing, environment_name, client):
                sdk.calls.append(("app", name, create_if_missing, environment_name))
                return types.SimpleNamespace(app_id="ap-1", client=client)

        class Secret:
            @staticmethod
            def from_dict(values):
                return ("secret", tuple(sorted(values.items())))

        class Image:
            @staticmethod
            def from_registry(reference, secret=None):
                return FakeImage(sdk, [("registry", reference, secret)])

        class Sandbox:
            @staticmethod
            def create(*argv, **options):
                sdk.calls.append(("create", argv, options))
                if sdk.fail_create is not None:
                    raise sdk.fail_create
                box = FakeBox(
                    sdk, f"sb-{len(sdk.boxes)}", options["tags"], options["name"]
                )
                sdk.boxes[box.object_id] = box
                return box

            @staticmethod
            def from_id(sandbox_id, client):
                if sandbox_id not in sdk.boxes:
                    raise NotFoundError(sandbox_id)
                return sdk.boxes[sandbox_id]

            @staticmethod
            def from_name(app_name, name, *, environment_name, client):
                sdk.calls.append(("from_name", app_name, name, environment_name))
                for box in sdk.boxes.values():
                    if box.name == name and box.code is None:
                        return box
                raise NotFoundError(name)

            @staticmethod
            def list(*, app_id, tags, client):
                sdk.calls.append(("list", app_id, dict(tags)))
                for box in sdk.boxes.values():
                    if box.code is None and tags.items() <= box.tags.items():
                        # As Modal's listing: the tags come from get_tags.
                        yield types.SimpleNamespace(
                            object_id=box.object_id, get_tags=box.get_tags
                        )

        modal.Client, modal.App, modal.Secret = Client, App, Secret
        modal.Image, modal.Sandbox = Image, Sandbox
        self.modules = {
            "modal": modal,
            "modal.config": config,
            "modal.exception": exception,
            "modal._utils": utils,
            "modal._utils.proxy_support": proxy_support,
        }

    @property
    def user_config(self):
        return self.modules["modal.config"]._user_config


@pytest.fixture
def proxy_hook(monkeypatch):
    """The hook's process state, restored after the test."""
    proxy_support = pytest.importorskip("modal._utils.proxy_support")

    monkeypatch.setattr(sandbox_modal, "_proxy", None)
    monkeypatch.setattr(sandbox_modal, "_sdk_lookup", None)
    monkeypatch.setattr(proxy_support, "get_proxy_url", proxy_support.get_proxy_url)


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.setattr(sandbox_modal, "_proxy", None)
    monkeypatch.setattr(sandbox_modal, "_sdk_lookup", None)
    fake = FakeModal()
    for name, module in fake.modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return fake


def make_client(**options):
    return SdkModalClient(TOKENS, **options)


def built(client, name="tpl", image="docker.io/library/busybox:1.37", **options):
    client.build_template(name, image, (0.5, 256), log=lambda line: None, **options)
    return name


# -- the adapter -----------------------------------------------------------------------


def test_the_client_takes_its_token_and_drops_the_modal_profiles(sdk, monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-from-the-environment")
    client = make_client(app_name="rsi-test", environment="ci")
    assert sdk.user_config == {}
    assert client.template_exists("tpl") is False
    built(client)
    assert ("client", *TOKENS) in sdk.calls
    assert ("app", "rsi-test", True, "ci") in sdk.calls
    for name in ("MODAL_SERVER_URL", "MODAL_ENVIRONMENT", "MODAL_OVERRIDE_HEADERS"):
        monkeypatch.setenv(name, "x")
        with pytest.raises(SetupError, match=name):
            make_client()
        monkeypatch.delenv(name)


def test_images_pull_with_registry_login_and_build_dockerfile_lines(
    sdk, tmp_path, monkeypatch
):
    auth = tmp_path / "auth.json"
    auth.write_text('{"auths": {"ghcr.io": {"username": "u", "password": "pw"}}}')
    client = make_client(registry_auth_file=str(auth))
    built(client, "pulled", "ghcr.io/org/app:1")
    assert client.template_exists("pulled")
    built(client, "made", steps=("RUN true", "USER 1000"), context="/ctx", force=True)
    builds = [call for call in sdk.calls if call[0] == "build"]
    login = ("secret", (("REGISTRY_PASSWORD", "pw"), ("REGISTRY_USERNAME", "u")))
    assert builds == [
        (
            "build",
            (("registry", "ghcr.io/org/app:1", login), ("entrypoint", ())),
            "ap-1",
        ),
        (
            "build",
            (
                ("registry", "docker.io/library/busybox:1.37", None),
                ("dockerfile", ("RUN true", "USER 1000"), "/ctx", True),
                ("entrypoint", ()),
            ),
            "ap-1",
        ),
    ]
    # Neither the token nor the registry password leaves in an error.
    sdk.fail_build = RuntimeError(f"denied pw {TOKENS[1]}")
    with pytest.raises(InfrastructureError) as failed:
        built(client, "again", "ghcr.io/org/app:2")
    assert "pw" not in str(failed.value) and TOKENS[1] not in str(failed.value)
    assert not client.template_exists("again")


@pytest.mark.parametrize(
    ("allow_internet", "allow_out", "network"),
    [
        (True, (), {}),
        (False, (), {"block_network": True}),
        (
            False,
            ("1.1.1.1", "10.0.0.0/8", "example.com"),
            {
                "outbound_cidr_allowlist": ["1.1.1.1/32", "10.0.0.0/8"],
                "outbound_domain_allowlist": ["example.com"],
            },
        ),
    ],
)
def test_create_names_tags_sizes_and_fences_the_sandbox(
    sdk, allow_internet, allow_out, network
):
    client = make_client()
    sandbox_id = client.create(
        built(client),
        timeout=661,
        metadata=METADATA,
        allow_internet=allow_internet,
        allow_out=allow_out,
    )
    ((_, argv, options),) = [call for call in sdk.calls if call[0] == "create"]
    assert argv == ("sleep", "infinity")
    image = options.pop("image")
    assert image.ops[0] == ("registry", "docker.io/library/busybox:1.37", None)
    assert options.pop("app").app_id == "ap-1"
    assert options == {
        "name": "e" + "a" * 32,
        "tags": METADATA,
        "timeout": 661,
        "cpu": (0.5, 0.5),
        "memory": (256, 256),
        "client": "client",
        **network,
    }
    assert sdk.boxes[sandbox_id].name == METADATA["rsi_env_id"]
    with pytest.raises(InfrastructureError, match="not built here"):
        client.create("other", timeout=1, metadata=METADATA, allow_internet=True)


def test_listing_reads_each_sandboxs_tags_and_lookup_finds_it_by_name(sdk):
    client = make_client()
    template = built(client)
    sandbox_id = client.create(
        template, timeout=60, metadata=METADATA, allow_internet=True
    )
    other = dict(METADATA, rsi_run_id="run-2", rsi_env_id="e" + "b" * 32)
    client.create(template, timeout=60, metadata=other, allow_internet=True)
    assert client.list({"rsi_run_id": "run-1"}) == [(sandbox_id, METADATA)]
    assert ("list", "ap-1", {"rsi_run_id": "run-1"}) in sdk.calls
    assert client.lookup(METADATA["rsi_env_id"]) == sandbox_id
    assert client.lookup("e" + "c" * 32) is None
    assert client.state(sandbox_id) == "running"
    client.kill(sandbox_id)
    assert client.state(sandbox_id) is None
    assert client.lookup(METADATA["rsi_env_id"]) is None
    # A sandbox Modal no longer knows is gone, and killing it again is fine.
    assert client.state("sb-unknown") is None
    client.kill("sb-unknown")


def test_processes_merge_their_streams_and_report_their_group(sdk):
    client = make_client()
    sandbox_id = client.create(
        built(client), timeout=60, metadata=METADATA, allow_internet=True
    )
    sdk.answers.append(FakeExec([b"41", b"7\nout"], [b"err"], code=3))
    process = start_group(
        client, sandbox_id, ["echo", "hi"], envs={"A": "1"}, cwd="/w", user="root"
    )
    assert process.pid == 417
    events = list(process)
    assert sorted(events) == [(1, b"out"), (2, b"err")]
    assert process.exit_code == 3
    ((argv, _),) = sdk.boxes[sandbox_id].execs
    assert argv[:2] == ("sh", "-c") and argv[-2:] == ("echo", "hi")
    assert ("exec", argv, {"A": "1"}, "/w", False) in sdk.calls
    # A broken stream fails the read; it is never followed again.
    sdk.answers.append(FakeExec([RuntimeError(f"cut {TOKENS[0]}")]))
    broken = client.start(sandbox_id, ["true"], envs={}, cwd="/", user="root")
    with pytest.raises(InfrastructureError) as cut:
        list(broken)
    assert TOKENS[0] not in str(cut.value)
    assert reattach(client, sandbox_id, 417, lambda: False) is None


def test_files_go_through_stdin_and_come_back_through_cat(sdk):
    client = make_client()
    sandbox_id = client.create(
        built(client), timeout=60, metadata=METADATA, allow_internet=True
    )
    box = sdk.boxes[sandbox_id]
    writer = FakeExec()
    sdk.answers.append(writer)
    client.write_file(sandbox_id, "/tmp/stage.tar", io.BytesIO(b"x" * 3_000_000))
    assert bytes(writer.stdin.data) == b"x" * 3_000_000 and writer.stdin.eof
    assert box.execs[-1][0][:3] == ("sh", "-c", sandbox_modal._WRITE_SCRIPT)
    sdk.answers.append(FakeExec(stderr=[b"no space"], code=1))
    with pytest.raises(InfrastructureError, match="exited 1: no space"):
        client.write_file(sandbox_id, "/tmp/full", b"y")

    box.files["/etc/passwd"] = b"root:x:0:0::/root:/bin/sh\n"
    assert client.read_file(sandbox_id, "/etc/passwd").startswith(b"root")
    with pytest.raises(FileNotFoundError):
        client.read_file(sandbox_id, "/missing")

    sdk.answers.append(FakeExec([b"ab", b"cd"]))
    with client.read_stream(sandbox_id, "/tmp/out.tar") as stream:
        assert stream.read() == b"abcd"
    assert box.execs[-1][0] == ("cat", "/tmp/out.tar")
    sdk.answers.append(FakeExec([b"a"], [b"gone"], code=1))
    with pytest.raises(OSError, match="gone"):
        client.read_stream(sandbox_id, "/tmp/out.tar").read()


def test_close_stops_following_every_process(sdk):
    client = make_client()
    sandbox_id = client.create(
        built(client), timeout=60, metadata=METADATA, allow_internet=True
    )
    gate = threading.Event()

    class Held(FakeReader):
        def __iter__(self):
            gate.wait(10)
            yield b"late"

    held = FakeExec()
    held.stdout = Held(())
    sdk.answers.append(held)
    process = client.start(sandbox_id, ["sleep", "9"], envs={}, cwd="/", user="root")
    client.close()
    assert list(process) == [] and process.exit_code is None
    gate.set()


# -- tokens and the host check ---------------------------------------------------------


def test_the_token_is_read_from_a_file_or_two_variables_never_echoed(tmp_path):
    path = tmp_path / "modal.toml"
    path.write_text(f'token_id = "{TOKENS[0]}"\ntoken_secret = "{TOKENS[1]}"\n')
    assert read_tokens(EnvModalHost(api_key_file=str(path))) == TOKENS
    settings = EnvModalHost(api_key_id_env="ID", api_key_env="SECRET")
    assert read_tokens(settings, {"ID": TOKENS[0], "SECRET": TOKENS[1]}) == TOKENS
    for broken in ("token_id = 1\n", f'token_id = "{TOKENS[0]}"\n', "token_id = ["):
        path.write_text(broken)
        with pytest.raises(SetupError) as refused:
            read_tokens(EnvModalHost(api_key_file=str(path)))
        assert TOKENS[0] not in str(refused.value)
    with pytest.raises(SetupError, match="hold no token"):
        read_tokens(settings, {"ID": TOKENS[0], "SECRET": " "})


def sdk_installed(monkeypatch, version=SDK_VERSION, missing=()):
    """Package metadata reports modal at ``version`` (None: missing) and the
    rest of its extra installed, but ``missing``."""
    real = importlib.metadata.version
    extra = ("dockerfile-parse", "python-socks", "aiohttp-socks", "socksio")

    def fake(name):
        if name in missing or (name == "modal" and version is None):
            raise importlib.metadata.PackageNotFoundError(name)
        if name == "modal":
            return version
        return "1.0" if name in extra else real(name)

    monkeypatch.setattr(importlib.metadata, "version", fake)


def test_the_submit_host_check_needs_the_pinned_sdk_and_the_token(monkeypatch):
    settings = EnvModalHost(api_key_id_env="RSI_T_ID", api_key_env="RSI_T_SECRET")
    monkeypatch.setenv("RSI_T_ID", TOKENS[0])
    monkeypatch.setenv("RSI_T_SECRET", TOKENS[1])
    for version, found in ((None, "none"), ("1.6.2", "1.6.2")):
        sdk_installed(monkeypatch, version)
        with pytest.raises(SetupError, match=rf"modal=={SDK_VERSION} \(found {found}"):
            check_modal_host(settings)
    sdk_installed(monkeypatch, missing=("aiohttp-socks", "socksio"))
    with pytest.raises(SetupError, match=r"needs aiohttp-socks; install .*\[modal\]"):
        check_modal_host(settings)
    socks = settings.model_copy(update={"proxy": "socks5h://proxy:1080"})
    with pytest.raises(SetupError, match=r"needs aiohttp-socks, socksio; "):
        check_modal_host(socks)
    sdk_installed(monkeypatch)
    check_modal_host(settings)
    monkeypatch.setenv("MODAL_SERVER_URL", "https://api.elsewhere")
    with pytest.raises(SetupError, match="MODAL_SERVER_URL"):
        check_modal_host(settings)
    monkeypatch.delenv("MODAL_SERVER_URL")
    # Only the endpoint variables are refused: the token is passed itself.
    refuse_modal_environ({"MODAL_TOKEN_ID": "x", "MODAL_PROFILE": "y"})
    monkeypatch.setenv("RSI_T_SECRET", "")
    with pytest.raises(SetupError, match="hold no token"):
        check_modal_host(settings)


def test_the_modal_extra_pins_the_checked_sdk_version():
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    extra = tomllib.loads(pyproject.read_text())["project"]["optional-dependencies"]
    assert f"modal[api-proxy-support]=={SDK_VERSION}" in extra["modal"]


# -- the proxy, over the real SDK ------------------------------------------------------


class ConnectProxy:
    """A local HTTP proxy that records each CONNECT and refuses it."""

    def __init__(self):
        self.server = socket.create_server(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{self.server.getsockname()[1]}"
        self.targets = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with conn:
                data = b""
                while b"\r\n\r\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                self.targets.append(data.split(b"\r\n", 1)[0].decode())
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")

    def close(self):
        self.server.close()


@pytest.fixture
def connect_proxy(proxy_hook, monkeypatch):
    pytest.importorskip("modal")
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    proxy = ConnectProxy()
    yield proxy
    proxy.close()


def _api(url):
    from modal._utils.grpc_utils import create_channel

    async def connect():
        channel = create_channel(url)
        try:
            await channel.__connect__()
        finally:
            channel.close()

    return connect()


def _router(monkeypatch, url):
    from modal.sandbox import _task_command_router_client as router

    async def once(channel):
        await channel.__connect__()

    # The router's own retries take minutes: one connect shows the path.
    monkeypatch.setattr(router, "_connect_channel", once)
    return router.TaskCommandRouterClient._connect(
        types.SimpleNamespace(_is_localhost=False),
        "ta-1",
        url,
        "jwt",
        fetch_access=None,
    )


def _blob(url):
    from modal._utils.http_utils import ClientSessionRegistry

    async def put():
        session = ClientSessionRegistry.get_session()
        try:
            async with session.put(url, data=b"x"):
                pass
        finally:
            await ClientSessionRegistry.close_session()

    return put()


def test_every_modal_connection_goes_through_the_policy_proxy(
    connect_proxy, monkeypatch
):
    environ = dict(os.environ)
    sandbox_modal.hook_proxy(connect_proxy.url)
    assert dict(os.environ) == environ  # nothing the Engine's children see
    for connection in (
        _api("https://api.modal.com"),
        _router(monkeypatch, "https://router.example:8443"),
        _blob("https://blob.example/upload"),
    ):
        with pytest.raises(Exception, match="403"):
            asyncio.run(connection)
    assert connect_proxy.targets == [
        "CONNECT api.modal.com:443 HTTP/1.1",
        "CONNECT router.example:8443 HTTP/1.1",
        "CONNECT blob.example:443 HTTP/1.1",
    ]


def test_without_a_policy_proxy_the_sdks_own_lookup_answers(connect_proxy, monkeypatch):
    from modal._utils import proxy_support

    sandbox_modal.hook_proxy(None)
    assert proxy_support.get_proxy_url is sandbox_modal._proxy_url
    monkeypatch.setenv("HTTPS_PROXY", connect_proxy.url)
    with pytest.raises(Exception, match="403"):
        asyncio.run(_api("https://api.modal.com"))
    assert connect_proxy.targets == ["CONNECT api.modal.com:443 HTTP/1.1"]
