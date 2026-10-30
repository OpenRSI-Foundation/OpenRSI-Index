"""The E2B env backend: ``SdkE2BClient``, the ``RemoteClient`` of
sandbox_remote over E2B sandboxes.

A template's name is its image, the phase's rounded per-container cpus and
memory (``template_shape``) and ``RECIPE``. The recipe resets the user to
root and the workdir to /, and records the image's ENV in
``IMAGE_ENV_FILE``, which every exec applies again (E2B starts processes
with its own environment). E2B keeps nothing else of the image config:
sandbox_remote applies the rest from the image's registry. ``image_build``
steps come from sandbox_e2b_dockerfile. Processes start through envd
without a login shell.

The API key lives only here: the broker reads it (``read_api_key``) and
hands it to the SDK; leases, plans, sandbox metadata, capabilities and
logs carry where the key is, never the key. The SDK (``e2b==2.52.0``, the
``e2b`` extra) is imported only by ``SdkE2BClient``.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import os
import shlex
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.sandbox_contracts import EnvE2BHost, SandboxError
from rsi_harness.runtime.sandbox_remote import (
    ChunkReader,
    RemoteBackend,
    read_registry_auth,
    registry_image_config,
)

LOGGER = logging.getLogger(__name__)
# The ``e2b`` extra's pin.
SDK_VERSION = "2.52.0"
# Endpoint variables the SDK reads when no option is passed; the policy's
# ``domain`` is the only endpoint setting.
_ENDPOINT_ENV = ("E2B_DOMAIN", "E2B_API_URL", "E2B_SANDBOX_URL")
# Bump when the template recipe changes: every name changes with it.
RECIPE = "r1"
IMAGE_ENV_FILE = "/etc/rsi-harness/image.env"
# A build step runs as ``bash -l -c`` with the image's ENV (PATH gets E2B's
# fallback directories appended) plus envd's HOME/USER/LOGNAME; the shell's
# own /proc environ is that, before any login profile changed it.
CAPTURE_ENV = f"mkdir -p /etc/rsi-harness && cat /proc/$$/environ > {IMAGE_ENV_FILE}"
_BUILD_PATH_SUFFIX = ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# envd sets these per user and the shell per process: never re-applied.
# envd's own E2B_* defaults are the build sandbox's in the recording (its
# id, its template): re-applied they would mask the live sandbox's values.
_NOT_IMAGE_ENV = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "E2B_SANDBOX",
        "E2B_SANDBOX_ID",
        "E2B_TEMPLATE_ID",
        "E2B_EVENTS_ADDRESS",
    }
)
MEMORY_STEP_MB = 512


def read_api_key(settings: EnvE2BHost, environ: Mapping[str, str] | None = None) -> str:
    """The operator's E2B API key; the errors name where it is, never it."""
    if settings.api_key_file is not None:
        where = f"api_key_file {settings.api_key_file}"
        try:
            text = Path(settings.api_key_file).read_text()
        except (OSError, UnicodeError) as error:
            raise SetupError(
                f"sandbox policy environments.host.e2b: {where} is unreadable "
                f"({type(error).__name__})"
            ) from None
    else:
        where = f"api_key_env {settings.api_key_env}"
        text = (os.environ if environ is None else environ).get(
            settings.api_key_env or "", ""
        )
    key = text.strip()
    if not key or any(character.isspace() for character in key):
        raise SetupError(f"sandbox policy environments.host.e2b: {where} holds no key")
    return key


def refuse_e2b_environ(environ: Mapping[str, str] | None = None) -> None:
    """Refuse the SDK's endpoint and debug variables, which would redirect
    the run's E2B calls past the policy."""
    environ = os.environ if environ is None else environ
    stray = [name for name in _ENDPOINT_ENV if environ.get(name)]
    if environ.get("E2B_DEBUG", "").lower() == "true":
        stray.append("E2B_DEBUG")
    if stray:
        raise SetupError(
            f"E2B variables set: {', '.join(stray)}; unset them and set "
            "[environments.host.e2b] domain for a self-hosted E2B"
        )


def check_e2b_host(settings: EnvE2BHost) -> None:
    """Check, without importing it, that the e2b extra (with the pinned
    SDK) is installed, then the environment, the key and any
    registry_auth_file: a cluster Engine runs this interpreter with the
    submit host's environment."""

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    installed = version("e2b")
    # The registry probe's httpx speaks SOCKS through socksio.
    socks = (settings.proxy or "").startswith("socks")
    missing = [
        name
        for name in ("dockerfile-parse", *(("socksio",) if socks else ()))
        if version(name) is None
    ]
    if installed != SDK_VERSION:
        missing = [f"e2b=={SDK_VERSION} (found {installed or 'none'})"]
    if missing:
        raise SetupError(
            f'backend = "e2b" needs {", ".join(missing)}; '
            "install rsi-harness with its e2b extra: "
            "uv tool install '/path/to/RSI-Harness[e2b]'"
        )
    refuse_e2b_environ()
    read_api_key(settings)
    if settings.registry_auth_file is not None:
        read_registry_auth(settings.registry_auth_file, "e2b")


# Go's names for the signals a ProcessState reports ("signal: killed").
_GO_SIGNALS = {
    "hangup": 1,
    "interrupt": 2,
    "quit": 3,
    "aborted": 6,
    "killed": 9,
    "user defined signal 1": 10,
    "segmentation fault": 11,
    "user defined signal 2": 12,
    "broken pipe": 13,
    "alarm clock": 14,
    "terminated": 15,
}


def shell_exit_code(code: int, status: str) -> int:
    """envd reports a signal death as -1 plus a status; the Engine (and a
    shell) as 128 + the signal number, which ExecPump's verdicts expect."""
    if code >= 0 or not status.startswith("signal: "):
        return code
    name = status.removeprefix("signal: ").removesuffix(" (core dumped)")
    number = _GO_SIGNALS.get(name)
    return code if number is None else 128 + number


class _SdkProcess:
    def __init__(self, pid: int, events: Any) -> None:
        self.pid = pid
        self.exit_code: int | None = None
        self._events = events

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        from protobuf import Oneof

        for event in self._events:
            oneof = event.event.event if event.event is not None else None
            match oneof:
                case Oneof(field="data", value=data):
                    match data.output:
                        case Oneof(field="stdout", value=chunk) if chunk:
                            yield 1, bytes(chunk)
                        case Oneof(field="stderr", value=chunk) if chunk:
                            yield 2, bytes(chunk)
                case Oneof(field="end", value=end):
                    self.exit_code = shell_exit_code(int(end.exit_code), end.status)
                    return

    def close(self) -> None:
        try:
            self._events.close()
        except Exception:
            pass


class SdkE2BClient:
    """``RemoteClient`` over the pinned SDK (``e2b==2.52.0``).

    Every call passes the key and domain explicitly, and the endpoint and
    debug variables are refused (``refuse_e2b_environ``); the SDK still
    reads its transport tuning variables (``E2B_HTTP_VERSION``,
    ``E2B_CONNECTION_RETRIES``, ...). Processes start through envd's Process
    RPC directly, not ``commands.run`` (which always wraps a ``bash -l -c``
    string): argv stays argv and no login profile runs. That RPC is internal
    to the SDK, hence the exact pin.
    """

    name = "e2b"
    recipe = RECIPE
    # envd reports pids and switches users; E2B pauses sandboxes.
    native_pid = True
    native_user = True
    freeze = "native"
    freeze_spare: tuple[str, ...] = ()
    argv_bytes = None
    build_form = "steps"

    def __init__(
        self,
        api_key: str,
        *,
        domain: str | None = None,
        proxy: str | None = None,
        request_timeout: float = 60,
        registry_auth_file: str | None = None,
    ) -> None:
        refuse_e2b_environ()
        import e2b

        self._e2b = e2b
        self._key = api_key
        self._registry_auth_file = registry_auth_file
        self._opts: dict[str, Any] = {
            "api_key": api_key,
            "request_timeout": request_timeout,
        }
        if domain is not None:
            self._opts["domain"] = domain
        if proxy is not None:
            self._opts["proxy"] = proxy
        self._boxes: dict[str, Any] = {}
        self._lock = threading.Lock()

    def _hide(self, text: str, *secrets: str) -> str:
        for secret in (self._key, *secrets):
            text = text.replace(secret, "***")
        return text

    def _clean(self, error: BaseException, *secrets: str) -> InfrastructureError:
        text = self._hide(str(error), *secrets)[:512]
        return InfrastructureError(f"e2b {type(error).__name__}: {text}")

    def _call(self, action: Callable[[], Any], *secrets: str) -> Any:
        try:
            return action()
        except (SandboxError, FileNotFoundError, InfrastructureError):
            raise
        except Exception as error:
            raise self._clean(error, *secrets) from None

    def _login(self, reference: str) -> tuple[str, str] | None:
        """The credentials for the registry of ``reference`` (domain/...),
        read again each time (they may be rotated during a run)."""
        if self._registry_auth_file is None:
            return None
        registries = read_registry_auth(self._registry_auth_file, "e2b")
        return registries.get(reference.partition("/")[0].lower())

    def _box(self, sandbox_id: str) -> Any:
        with self._lock:
            box = self._boxes.get(sandbox_id)
        if box is None:
            box = self._call(
                lambda: self._e2b.Sandbox.connect(sandbox_id, **self._opts)
            )
            with self._lock:
                self._boxes[sandbox_id] = box
        return box

    def template_exists(self, name: str) -> bool:
        """A name whose default build is ready: E2B creates the name before
        it builds, and a failed build leaves the name on a failed build.
        Another team's name (403) and a deployment without tags or build
        status (404) count by the name alone, as ``Template.exists``."""
        from e2b.api.client.api.tags import get_templates_template_id_tags
        from e2b.api.client.api.templates import (
            get_templates_aliases_alias,
            get_templates_template_id_builds_build_id_status,
        )
        from e2b.api.client_sync import get_api_client
        from e2b.connection_config import ConnectionConfig

        def parsed(response: Any) -> Any:
            if response.status_code >= 300 or response.parsed is None:
                raise InfrastructureError(
                    f"e2b template {name}: {response.status_code}"
                )
            return response.parsed

        def ready() -> bool:
            api = get_api_client(ConnectionConfig(**self._opts))
            alias = get_templates_aliases_alias.sync_detailed(alias=name, client=api)
            if alias.status_code in (403, 404):
                return alias.status_code == 403
            template_id = parsed(alias).template_id
            tags = get_templates_template_id_tags.sync_detailed(
                template_id=template_id, client=api
            )
            if tags.status_code == 404:
                return True
            builds = [
                str(item.build_id) for item in parsed(tags) if item.tag == "default"
            ]
            if not builds:
                return False
            status = get_templates_template_id_builds_build_id_status.sync_detailed(
                template_id=template_id, build_id=builds[0], client=api
            )
            if status.status_code == 404:
                return True
            found = parsed(status).status
            return getattr(found, "value", found) == "ready"

        return bool(self._call(ready))

    def image_config(self, repository, tag):
        login = self._login(repository)
        hidden = () if login is None else (login[1],)
        return self._call(
            lambda: registry_image_config(
                repository, tag, proxy=self._opts.get("proxy"), login=login
            ),
            *hidden,
        )

    def shape(self, cpus, memory_mb, disk_mb):
        return template_shape(cpus, memory_mb)

    def build_template(
        self, name, image, shape, *, log, steps=(), context=None, force=False
    ):
        cpu_count, memory_mb = shape
        username, password = self._login(image) or (None, None)
        hidden = () if password is None else (password,)
        template_class = self._e2b.Template
        # COPY sources are read from ``context`` (sandbox_e2b_dockerfile).
        template = template_class(file_context_path=context).from_image(
            image, username=username, password=password
        )
        user = "root"
        for kind, *args in steps:
            if kind == "user":
                user = args[0]
                template = template.set_user(user)
            elif kind == "workdir":
                # As BuildKit: a missing WORKDIR is made the USER's, with
                # its primary group (0 for a uid without a passwd entry).
                path, owner = shlex.quote(args[0]), shlex.quote(user)
                make = f"mkdir -p {path}"
                if user != "root":
                    group = f'"$(id -g {owner} 2>/dev/null || echo 0)"'
                    make = (
                        f"test -d {path} || "
                        f"{{ {make} && chown {owner}:{group} {path}; }}"
                    )
                template = template.run_cmd(make, user="root").set_workdir(args[0])
            elif kind == "env":
                template = template.set_envs(args[0])
            elif kind == "run":
                template = template.run_cmd(args[0])
            else:  # copy, add
                source, dest, owner, mode = args
                template = template.copy(source, dest, user=owner, mode=mode)
        template = (
            template.set_user("root").set_workdir("/").run_cmd(CAPTURE_ENV, user="root")
        )
        # Without request_timeout: context uploads get the SDK's own (1 h).
        options = {
            key: value for key, value in self._opts.items() if key != "request_timeout"
        }
        self._call(
            lambda: template_class.build(
                template,
                name,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                skip_cache=force,
                on_build_logs=lambda entry: log(
                    self._hide(f"{getattr(entry, 'message', '')}\n", *hidden)
                ),
                **options,
            ),
            *hidden,
        )

    def create(self, template, *, timeout, metadata, allow_internet, allow_out=()):
        # Inbound only with the traffic token, which nobody gets.
        network: dict[str, Any] = {"allow_public_traffic": False}
        if allow_out:
            # E2B's allow beats its deny, so this is an allowlist.
            network.update(allow_out=list(allow_out), deny_out=["0.0.0.0/0"])
        box = self._call(
            lambda: self._e2b.Sandbox.create(
                template,
                timeout=timeout,
                metadata=dict(metadata),
                allow_internet_access=allow_internet or bool(allow_out),
                network=network,
                lifecycle={"on_timeout": "kill"},
                **self._opts,
            )
        )
        with self._lock:
            self._boxes[box.sandbox_id] = box
        return box.sandbox_id

    def image_env(self, sandbox_id, image_env):
        return recorded_image_env(self, sandbox_id)

    def lookup(self, env_id):
        return None  # E2B sandboxes have no name: the listing finds them

    def state(self, sandbox_id):
        try:
            info = self._e2b.Sandbox.get_info(sandbox_id, **self._opts)
        except self._e2b.SandboxNotFoundException:
            return None
        except Exception as error:
            raise self._clean(error) from None
        state = getattr(info.state, "value", info.state)
        return str(state)

    def kill(self, sandbox_id):
        self._call(lambda: self._e2b.Sandbox.kill(sandbox_id, **self._opts))
        with self._lock:
            self._boxes.pop(sandbox_id, None)

    def pause(self, sandbox_id):
        self._call(
            lambda: self._e2b.Sandbox.pause(sandbox_id, keep_memory=True, **self._opts)
        )

    def resume(self, sandbox_id, *, timeout):
        # connect resumes a paused sandbox; a pause restarts E2B's timer.
        box = self._call(
            lambda: self._e2b.Sandbox.connect(sandbox_id, timeout=timeout, **self._opts)
        )
        self._call(
            lambda: self._e2b.Sandbox.set_timeout(sandbox_id, timeout, **self._opts)
        )
        with self._lock:
            self._boxes[sandbox_id] = box

    def list(self, metadata):
        def listing() -> list[tuple[str, dict[str, str]]]:
            query = self._e2b.SandboxQuery(metadata=dict(metadata))
            pages = self._e2b.Sandbox.list(query=query, **self._opts)
            found = []
            while pages.has_next:
                for info in pages.next_items():
                    found.append((info.sandbox_id, dict(info.metadata or {})))
            return found

        return self._call(listing)

    def write_file(self, sandbox_id, path, data):
        box = self._box(sandbox_id)
        self._call(lambda: box.files.write(path, data, user="root"))

    def read_file(self, sandbox_id, path):
        box = self._box(sandbox_id)
        try:
            return bytes(box.files.read(path, format="bytes", user="root"))
        except self._e2b.FileNotFoundException:
            raise FileNotFoundError(path) from None
        except Exception as error:
            raise self._clean(error) from None

    def read_stream(self, sandbox_id, path):
        return ChunkReader(
            lambda: self._call(
                lambda: self._box(sandbox_id).files.read(
                    path, format="stream", user="root"
                )
            ),
            self._clean,
        )

    def _process(self, sandbox_id: str, request: Callable[[Any, Any], Any]) -> Any:
        from e2b.connection_config import (
            KEEPALIVE_PING_HEADER,
            KEEPALIVE_PING_INTERVAL_SEC,
        )
        from e2b.envd.client_sync import as_stream
        from e2b.envd.process import process_pb
        from e2b.envd.utils import extract_start_pid

        commands = self._box(sandbox_id).commands
        headers = {KEEPALIVE_PING_HEADER: str(KEEPALIVE_PING_INTERVAL_SEC)}

        def open_stream() -> _SdkProcess:
            events = as_stream(request(commands, process_pb, headers))
            try:
                pid = extract_start_pid(next(events), "start process")
            except BaseException:
                events.close()
                raise
            return _SdkProcess(pid, events)

        return self._call(open_stream)

    def start(self, sandbox_id, argv, *, envs, cwd, user):
        from e2b.envd.utils import authentication_header

        def request(commands: Any, process_pb: Any, headers: dict) -> Any:
            return commands._rpc.start(
                process_pb.StartRequest(
                    process=process_pb.ProcessConfig(
                        cmd=argv[0], args=list(argv[1:]), envs=dict(envs), cwd=cwd
                    ),
                    stdin=False,
                ),
                headers={
                    **authentication_header(commands._envd_version, user),
                    **headers,
                },
                timeout_ms=None,  # no envd deadline: the broker times execs
            )

        return self._process(sandbox_id, request)

    def connect(self, sandbox_id, pid):
        from protobuf import Oneof

        def request(commands: Any, process_pb: Any, headers: dict) -> Any:
            return commands._rpc.connect(
                process_pb.ConnectRequest(
                    process=process_pb.ProcessSelector(selector=Oneof("pid", pid))
                ),
                headers=headers,
                timeout_ms=None,
            )

        return self._process(sandbox_id, request)

    def close(self):
        with self._lock:
            self._boxes.clear()


# -- templates --------------------------------------------------------------------


def template_shape(cpus: float, memory_mb: int) -> tuple[int, int]:
    """E2B sizes are fixed per template: round up. The vCPU count must be 1 or
    even (E2B rejects odd counts above 1), memory whole steps of MiB."""
    cpu_count = max(1, math.ceil(cpus))
    if cpu_count > 1:
        cpu_count += cpu_count % 2
    return (
        cpu_count,
        max(MEMORY_STEP_MB, math.ceil(memory_mb / MEMORY_STEP_MB) * MEMORY_STEP_MB),
    )


def recorded_image_env(client: Any, sandbox_id: str) -> dict[str, str]:
    """The image's ENV as the template build recorded it (CAPTURE_ENV): E2B
    starts processes with its own environment."""
    try:
        return parse_image_env(client.read_file(sandbox_id, IMAGE_ENV_FILE))
    except FileNotFoundError:
        LOGGER.warning("e2b sandbox %s has no recorded image ENV", sandbox_id)
        return {}


def parse_image_env(raw: bytes) -> dict[str, str]:
    """The recorded image ENV, without what envd and the shell add."""
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        key, separator, value = item.decode("utf-8", "replace").partition("=")
        if not separator or not key or key in _NOT_IMAGE_ENV:
            continue
        if key == "PATH" and value.endswith(_BUILD_PATH_SUFFIX):
            value = value[: -len(_BUILD_PATH_SUFFIX)] or value
        env[key] = value
    return env


def e2b_client(settings: EnvE2BHost, api_key: str | None = None) -> SdkE2BClient:
    """The production client; the key (read here unless the caller already
    did) is kept only by it."""
    return SdkE2BClient(
        read_api_key(settings) if api_key is None else api_key,
        domain=settings.domain,
        proxy=settings.proxy,
        registry_auth_file=settings.registry_auth_file,
    )


BACKEND = RemoteBackend(
    check_host=check_e2b_host,
    credentials=lambda settings: (read_api_key(settings),),
    client=lambda credentials, settings: e2b_client(settings, *credentials),
)


__all__ = [
    "BACKEND",
    "CAPTURE_ENV",
    "IMAGE_ENV_FILE",
    "RECIPE",
    "SDK_VERSION",
    "SdkE2BClient",
    "check_e2b_host",
    "e2b_client",
    "parse_image_env",
    "read_api_key",
    "recorded_image_env",
    "refuse_e2b_environ",
    "template_shape",
]
