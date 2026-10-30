"""The Modal env backend: ``SdkModalClient``, the ``RemoteClient`` of
sandbox_remote over Modal sandboxes.

A template is a Modal image: the reference pulled by Modal's registry pull
with its entrypoint cleared, plus, for ``image_build``, the Dockerfile's
own instructions after FROM (sandbox_e2b_dockerfile.lines), which Modal's
builder runs as Docker does. Modal names and caches images by their
content, so a template name only maps to its image and shape in this
broker; the shape (the phase's exact cpus and memory) is the sandbox's,
given at create. Every sandbox belongs to the deployed app ``app_name``,
carries the env's metadata as tags and is named by its env id, so recovery
lists the app's sandboxes by tag and an unanswered create is found by name.
A sandbox runs ``sleep infinity``; processes start without a shell.

Modal reports no pids, starts every process as root and cannot pause a
sandbox: sandbox_remote emulates those (``group_argv``, ``setpriv``,
``SignalFreeze``). A process stream cannot be followed again once it
breaks.

The token lives only here, as the E2B key does in sandbox_e2b. The SDK
(``modal==1.6.3``, the ``modal`` extra) is imported only by
``SdkModalClient``.
"""

from __future__ import annotations

import importlib.metadata
import io
import logging
import os
import queue
import re
import sys
import threading
import tomllib
import weakref
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.sandbox_contracts import EnvModalHost, SandboxError
from rsi_harness.runtime.sandbox_remote import (
    META_PREFIX,
    MIB,
    ROOT_PATH,
    ChunkReader,
    RemoteBackend,
    config_env,
    read_registry_auth,
    registry_image_config,
)

LOGGER = logging.getLogger(__name__)
# The ``modal`` extra's pin.
SDK_VERSION = "1.6.3"
# Variables that would send the run's Modal calls to another server,
# environment or credentials than the policy's (MODAL_OVERRIDE_HEADERS can
# replace the token headers).
_ENDPOINT_ENV = ("MODAL_SERVER_URL", "MODAL_ENVIRONMENT", "MODAL_OVERRIDE_HEADERS")
# Bump when the image recipe changes: every name changes with it.
RECIPE = "m1"
# The SDK's own limit on a command's arguments (sandbox/_common.py).
ARGV_BYTES = 2**16
# A file write: the path's directory, then stdin into the file.
_WRITE_SCRIPT = 'mkdir -p "${1%/*}" && cat > "$1"'


def read_tokens(
    settings: EnvModalHost, environ: Mapping[str, str] | None = None
) -> tuple[str, str]:
    """The operator's Modal token id and secret; the errors name where they
    are, never them."""
    if settings.api_key_file is not None:
        where, verb = f"api_key_file {settings.api_key_file}", "holds"
        try:
            parsed = tomllib.loads(Path(settings.api_key_file).read_text())
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
            raise SetupError(
                f"sandbox policy environments.host.modal: {where} is unreadable "
                f"({type(error).__name__})"
            ) from None
        found = (parsed.get("token_id"), parsed.get("token_secret"))
    else:
        where = (
            f"api_key_id_env {settings.api_key_id_env} and "
            f"api_key_env {settings.api_key_env}"
        )
        verb = "hold"
        environ = os.environ if environ is None else environ
        found = (
            environ.get(settings.api_key_id_env or ""),
            environ.get(settings.api_key_env or ""),
        )
    tokens = tuple(item.strip() if isinstance(item, str) else "" for item in found)
    if not all(token and not any(c.isspace() for c in token) for token in tokens):
        raise SetupError(
            f"sandbox policy environments.host.modal: {where} {verb} no token"
        )
    return tokens  # type: ignore[return-value]


def refuse_modal_environ(environ: Mapping[str, str] | None = None) -> None:
    """Refuse the SDK's endpoint variables, which would redirect the run's
    Modal calls past the policy (the client drops ~/.modal.toml itself)."""
    environ = os.environ if environ is None else environ
    stray = [name for name in _ENDPOINT_ENV if environ.get(name)]
    if stray:
        raise SetupError(
            f"Modal variables set: {', '.join(stray)}; unset them and set "
            "[environments.host.modal] environment to choose an environment"
        )


def check_modal_host(settings: EnvModalHost) -> None:
    """As check_e2b_host, for the modal extra and its token."""

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    installed = version("modal")
    socks = (settings.proxy or "").startswith("socks")
    # python-socks and aiohttp-socks carry the SDK's proxied connections.
    needed = ("dockerfile-parse", "python-socks", "aiohttp-socks")
    missing = [
        name
        for name in (*needed, *(("socksio",) if socks else ()))
        if version(name) is None
    ]
    if installed != SDK_VERSION:
        missing = [f"modal=={SDK_VERSION} (found {installed or 'none'})"]
    if missing:
        raise SetupError(
            f'backend = "modal" needs {", ".join(missing)}; '
            "install rsi-harness with its modal extra: "
            "uv tool install '/path/to/RSI-Harness[modal]'"
        )
    refuse_modal_environ()
    read_tokens(settings)
    if settings.registry_auth_file is not None:
        read_registry_auth(settings.registry_auth_file, "modal")


# -- the proxy ---------------------------------------------------------------------

# The SDK reads its proxy only from the process environment, which the
# Engine's children inherit. Instead the client swaps the SDK's one lookup,
# modal._utils.proxy_support.get_proxy_url, which every connection imports
# when it opens: the gRPC channels to the API and to each sandbox's command
# router (grpc_utils.ModalChannel) and the HTTP session of blob uploads and
# downloads (http_utils.ClientSessionRegistry). Without a proxy set, the
# SDK's own lookup answers.
_proxy: str | None = None
_sdk_lookup: Callable[..., str | None] | None = None
_lock = threading.Lock()


def _proxy_url(host: str | None, *, use_ssl: bool) -> str | None:
    if _proxy is not None:
        return _proxy
    assert _sdk_lookup is not None
    return _sdk_lookup(host, use_ssl=use_ssl)


def hook_proxy(proxy: str | None) -> None:
    """Send every Modal connection of this process through ``proxy`` (None:
    the SDK's own lookup)."""
    global _proxy, _sdk_lookup
    from modal._utils import proxy_support

    with _lock:
        if proxy_support.get_proxy_url is not _proxy_url:
            _sdk_lookup = proxy_support.get_proxy_url
            # And any module that bound the name itself (none in 1.6.3).
            for name, module in list(sys.modules.items()):
                if name.split(".")[0] == "modal" and (
                    getattr(module, "get_proxy_url", None) is _sdk_lookup
                ):
                    module.get_proxy_url = _proxy_url
        _proxy = proxy


# -- processes ---------------------------------------------------------------------


class _ModalProcess:
    """One exec's stdout and stderr, each read by its own thread (the SDK
    streams them apart). ``close`` stops following: a reader blocked in the
    SDK ends with the stream and drops what it reads."""

    def __init__(
        self, process: Any, clean: Callable[[BaseException], BaseException]
    ) -> None:
        self.pid = 0  # unknown: group_argv reports the group's
        self.exit_code: int | None = None
        self._process = process
        self._clean = clean
        self._queue: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        for stream, reader in ((1, process.stdout), (2, process.stderr)):
            threading.Thread(
                target=self._read,
                args=(stream, reader),
                name="rsi-sandbox-modal-stream",
                daemon=True,
            ).start()

    def _read(self, stream: int, reader: Any) -> None:
        try:
            for chunk in reader:
                if self._closed.is_set():
                    return
                if chunk:
                    self._queue.put((stream, bytes(chunk)))
            self._queue.put((stream, None))
        except BaseException as error:
            self._queue.put(("error", error))

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        open_streams = 2
        while open_streams:
            stream, chunk = self._queue.get()
            if stream == "closed":
                return
            if stream == "error":
                raise self._clean(chunk)
            if chunk is None:
                open_streams -= 1
                continue
            yield stream, chunk
        try:
            self.exit_code = int(self._process.wait())
        except Exception as error:
            raise self._clean(error) from None

    def close(self) -> None:
        self._closed.set()
        self._queue.put(("closed", None))


class SdkModalClient:
    """``RemoteClient`` over the pinned SDK (``modal==1.6.3``).

    The client is made from the token explicitly; the endpoint variables
    are refused (``refuse_modal_environ``) and ~/.modal.toml is ignored, so
    no profile changes the server, environment or credentials. The proxy
    is set by swapping the SDK's lookup (``hook_proxy``), hence the exact
    pin.
    """

    name = "modal"
    recipe = RECIPE
    # Emulated by sandbox_remote: Modal has none of these.
    native_pid = False
    native_user = False
    freeze = "signal"
    freeze_spare: tuple[str, ...] = ()
    build_form = "lines"
    argv_bytes = ARGV_BYTES

    def __init__(
        self,
        tokens: tuple[str, str],
        *,
        app_name: str = "rsi-harness",
        environment: str | None = None,
        proxy: str | None = None,
        registry_auth_file: str | None = None,
    ) -> None:
        refuse_modal_environ()
        import modal
        import modal.config
        import modal.exception

        # No profile of ~/.modal.toml (or MODAL_CONFIG_PATH) applies.
        modal.config._user_config.clear()
        hook_proxy(proxy)
        self._modal = modal
        self._tokens = tokens
        self._app_name = app_name
        self._environment = environment
        self._proxy = proxy
        self._registry_auth_file = registry_auth_file
        self._sdk_client: Any = None
        self._app: Any = None
        # Template name -> (image, (cpus, memory_mb)).
        self._images: dict[str, tuple[Any, tuple]] = {}
        self._boxes: dict[str, Any] = {}
        self._processes: weakref.WeakSet[_ModalProcess] = weakref.WeakSet()
        self._lock = threading.Lock()

    def _hide(self, text: str, *secrets: str) -> str:
        for secret in (*self._tokens, *secrets):
            text = text.replace(secret, "***")
        return text

    def _clean(self, error: BaseException, *secrets: str) -> InfrastructureError:
        text = self._hide(str(error), *secrets)[:512]
        return InfrastructureError(f"modal {type(error).__name__}: {text}")

    def _call(self, action: Callable[[], Any], *secrets: str) -> Any:
        try:
            return action()
        except (SandboxError, FileNotFoundError, InfrastructureError):
            raise
        except Exception as error:
            raise self._clean(error, *secrets) from None

    def _client(self) -> Any:
        with self._lock:
            if self._sdk_client is None:
                self._sdk_client = self._call(
                    lambda: self._modal.Client.from_credentials(*self._tokens)
                )
            return self._sdk_client

    def _deployed(self) -> Any:
        client = self._client()
        with self._lock:
            if self._app is None:
                self._app = self._call(
                    lambda: self._modal.App.lookup(
                        self._app_name,
                        create_if_missing=True,
                        environment_name=self._environment,
                        client=client,
                    )
                )
            return self._app

    def _login(self, reference: str) -> tuple[str, str] | None:
        if self._registry_auth_file is None:
            return None
        registries = read_registry_auth(self._registry_auth_file, "modal")
        return registries.get(reference.partition("/")[0].lower())

    def _box(self, sandbox_id: str) -> Any:
        with self._lock:
            box = self._boxes.get(sandbox_id)
        if box is None:
            client = self._client()
            box = self._call(
                lambda: self._modal.Sandbox.from_id(sandbox_id, client=client)
            )
            with self._lock:
                self._boxes[sandbox_id] = box
        return box

    # -- templates -----------------------------------------------------------

    def shape(self, cpus, memory_mb, disk_mb):
        return (float(cpus), int(memory_mb))  # Modal sizes each sandbox exactly

    def template_exists(self, name):
        with self._lock:
            return name in self._images

    def image_config(self, repository, tag):
        login = self._login(repository)
        hidden = () if login is None else (login[1],)
        return self._call(
            lambda: registry_image_config(
                repository, tag, proxy=self._proxy, login=login
            ),
            *hidden,
        )

    def build_template(
        self, name, image, shape, *, log, steps=(), context=None, force=False
    ):
        login = self._login(image)
        hidden = () if login is None else (login[1],)
        modal = self._modal

        def build() -> Any:
            secret = None
            if login is not None:
                secret = modal.Secret.from_dict(
                    {"REGISTRY_USERNAME": login[0], "REGISTRY_PASSWORD": login[1]}
                )
            built = modal.Image.from_registry(image, secret=secret)
            if steps:
                built = built.dockerfile_commands(
                    list(steps), context_dir=context, force_build=force
                )
            # The sandbox's own command runs, not the image's ENTRYPOINT.
            built = built.entrypoint([])
            return built.build(self._deployed())

        log(f"modal builds {image}{' with its Dockerfile' if steps else ''}\n")
        built = self._call(build, *hidden)
        with self._lock:
            self._images[name] = (built, tuple(shape))

    # -- sandboxes -----------------------------------------------------------

    def create(self, template, *, timeout, metadata, allow_internet, allow_out=()):
        with self._lock:
            found = self._images.get(template)
        if found is None:
            raise InfrastructureError(f"modal image {template} was not built here")
        image, (cpus, memory_mb) = found
        network: dict[str, Any] = {}
        if allow_out:
            # An IP is a /32; a hostname matches the TLS SNI on 443 only.
            cidrs = [
                host if "/" in host else f"{host}/32"
                for host in allow_out
                if re.fullmatch(r"[0-9./]+", host)
            ]
            domains = [
                host for host in allow_out if not re.fullmatch(r"[0-9./]+", host)
            ]
            network = {
                "outbound_cidr_allowlist": cidrs,
                "outbound_domain_allowlist": domains,
            }
        elif not allow_internet:
            network = {"block_network": True}
        app, client = self._deployed(), self._client()
        box = self._call(
            lambda: self._modal.Sandbox.create(
                "sleep",
                "infinity",
                app=app,
                name=metadata.get(f"{META_PREFIX}env_id"),
                tags=dict(metadata),
                image=image,
                timeout=timeout,
                cpu=(cpus, cpus),
                memory=(memory_mb, memory_mb),
                client=client,
                **network,
            )
        )
        with self._lock:
            self._boxes[box.object_id] = box
        return box.object_id

    def image_env(self, sandbox_id, image_env):
        return config_env(image_env)

    def lookup(self, env_id):
        client = self._client()
        try:
            box = self._modal.Sandbox.from_name(
                self._app_name,
                env_id,
                environment_name=self._environment,
                client=client,
            )
        except self._modal.exception.NotFoundError:
            return None
        except Exception as error:
            raise self._clean(error) from None
        return box.object_id

    def state(self, sandbox_id):
        client = self._client()
        try:
            box = self._modal.Sandbox.from_id(sandbox_id, client=client)
            code = box.poll()
        except self._modal.exception.NotFoundError:
            return None
        except Exception as error:
            raise self._clean(error) from None
        return "running" if code is None else None

    def kill(self, sandbox_id):
        client = self._client()
        try:
            self._modal.Sandbox.from_id(sandbox_id, client=client).terminate()
        except self._modal.exception.NotFoundError:
            pass
        except Exception as error:
            raise self._clean(error) from None
        with self._lock:
            self._boxes.pop(sandbox_id, None)

    def list(self, metadata):
        def listing() -> list[tuple[str, dict[str, str]]]:
            app, client = self._deployed(), self._client()
            boxes = self._modal.Sandbox.list(
                app_id=app.app_id, tags=dict(metadata), client=client
            )
            # A listed sandbox carries no tags until asked for them.
            return [(box.object_id, dict(box.get_tags())) for box in boxes]

        return self._call(listing)

    # -- files and processes -------------------------------------------------

    def write_file(self, sandbox_id, path, data):
        source = data if hasattr(data, "read") else io.BytesIO(bytes(data))
        box = self._box(sandbox_id)

        def write() -> None:
            process = box.exec(
                "sh",
                "-c",
                _WRITE_SCRIPT,
                "sh",
                path,
                env={"PATH": ROOT_PATH},
                text=False,
            )
            for chunk in iter(lambda: source.read(MIB), b""):
                process.stdin.write(chunk)
                process.stdin.drain()
            process.stdin.write_eof()
            process.stdin.drain()
            code = process.wait()
            if code != 0:
                err = process.stderr.read()[-512:].decode("utf-8", "replace")
                raise InfrastructureError(f"modal write of {path} exited {code}: {err}")

        self._call(write)

    def read_file(self, sandbox_id, path):
        box = self._box(sandbox_id)
        try:
            return bytes(box.filesystem.read_bytes(path))
        except self._modal.exception.SandboxFilesystemNotFoundError:
            raise FileNotFoundError(path) from None
        except Exception as error:
            raise self._clean(error) from None

    def read_stream(self, sandbox_id, path):
        """``cat`` of the file, its stdout the chunks."""

        def chunks(process: _ModalProcess) -> Iterator[bytes]:
            err = bytearray()
            try:
                for stream, data in process:
                    if stream == 1:
                        yield data
                    else:
                        err += data[: max(0, 512 - len(err))]
            finally:
                process.close()
            if process.exit_code != 0:
                raise InfrastructureError(
                    f"modal read of {path} exited {process.exit_code}: "
                    f"{err.decode('utf-8', 'replace')}"
                )

        return ChunkReader(
            lambda: chunks(
                self.start(
                    sandbox_id,
                    ["cat", path],
                    envs={"PATH": ROOT_PATH},
                    cwd="/",
                    user="root",
                )
            ),
            self._clean,
        )

    def start(self, sandbox_id, argv, *, envs, cwd, user):
        assert user == "root", "Modal starts processes as root only"
        box = self._box(sandbox_id)
        process = self._call(
            lambda: box.exec(*argv, env=dict(envs), workdir=cwd, text=False)
        )
        wrapped = _ModalProcess(process, self._clean)
        self._processes.add(wrapped)
        return wrapped

    def connect(self, sandbox_id, pid):
        raise InfrastructureError("modal cannot follow a process again")

    def close(self):
        for process in list(self._processes):
            process.close()
        with self._lock:
            self._boxes.clear()


def modal_client(
    settings: EnvModalHost, tokens: tuple[str, str] | None = None
) -> SdkModalClient:
    """The production client; the token (read here unless the caller already
    did) is kept only by it."""
    return SdkModalClient(
        read_tokens(settings) if tokens is None else tokens,
        app_name=settings.app_name,
        environment=settings.environment,
        proxy=settings.proxy,
        registry_auth_file=settings.registry_auth_file,
    )


BACKEND = RemoteBackend(
    check_host=check_modal_host,
    credentials=read_tokens,
    client=lambda credentials, settings: modal_client(settings, credentials),
)


__all__ = [
    "ARGV_BYTES",
    "BACKEND",
    "RECIPE",
    "SDK_VERSION",
    "SdkModalClient",
    "check_modal_host",
    "hook_proxy",
    "modal_client",
    "read_tokens",
    "refuse_modal_environ",
]
