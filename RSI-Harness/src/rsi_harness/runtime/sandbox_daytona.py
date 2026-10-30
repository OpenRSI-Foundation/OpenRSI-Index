"""The Daytona env backend: ``SdkDaytonaClient``, the ``RemoteClient`` of
sandbox_remote over Daytona sandboxes.

A template is the image reference with the phase's whole cores and GiB
of memory and disk (``shape``), kept in the broker's memory: each sandbox
is created from the image (``FROM`` it, entrypoint ``sleep infinity``)
with that shape, and Daytona builds and caches the image on its first
use, which the template's build makes with a sandbox of its own. The key
thus needs only the sandbox permissions, and nothing is left in the
organization. Every sandbox runs as root, carries the env's metadata as
labels and is named by its env id, so recovery lists them by label and an
unanswered create is found by name. Sandboxes are ephemeral: their TTL
and their auto-stop are both the broker's timeout (inactivity never
outlasts the wall clock), and a stopped one is deleted. There is no
``image_build``: a build with a context goes through the SDK's object
storage upload, which clears the process environment while it runs.

A key with ``write:sandboxes`` only works too. Its delete is refused (403)
and its listing comes back empty, so every create is recorded in the run's
sandbox ledger (sandbox_remote_ledger: the name before the request, the id
after, the end once proven) and ``list`` merges the listing, tolerated
when empty or refused, with the ledger's open entries, each read again.
Where a sandbox is deleted (``kill``, a template's build sandbox,
``state`` of one that cannot run again), a refused delete becomes a stop,
which removes an ephemeral sandbox: one still starting or building is
waited for first, and the end is proven by a read that finds nothing. One
still there afterwards is a leftover, reported with its id and state and
kept open in the ledger. After the first refusal the client stops without
asking to delete. A key that may delete and list never reaches any of
this. A create refused for capacity (429, the organization's quota, or no
runner within the queue timeout) is a ``ProviderCapacityError`` only once a
read by name proves nothing landed: the SDK's polls after the request
(build logs, start) can be refused too, after the sandbox existed. A
template's build sandbox waits for capacity up to ``CAPACITY_WAIT_SEC``
first.

Each process is one command of its own session, run by the session's
shell as a child (``_RUN_SCRIPT``: a top-level ``exec`` would replace the
shell), its output followed over the toolbox's log websocket. Daytona
reports no pids, starts every process as root and cannot pause a sandbox
(on the container class): sandbox_remote emulates those. A broken stream is
followed again from the start of the command's log, skipping what was
already handed out; a session is deleted only once its exit code is seen.

The key lives only here, as the E2B key does in sandbox_e2b. The SDK
(``daytona==0.227.0``, the ``daytona`` extra) is async: each client runs it
on an event loop thread of its own, and only ``SdkDaytonaClient`` imports
it.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import math
import os
import queue
import re
import secrets
import shlex
import threading
import time
import weakref
from collections.abc import Callable, Coroutine, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.sandbox_contracts import (
    EnvDaytonaHost,
    ProviderCapacityError,
    SandboxError,
)
from rsi_harness.runtime.sandbox_remote import (
    META_PREFIX,
    ChunkReader,
    RemoteBackend,
    config_env,
    group_argv,
    read_registry_auth,
    registry_image_config,
    run_metadata,
)
from rsi_harness.runtime.sandbox_remote_ledger import LedgerEntry, SandboxLedger

LOGGER = logging.getLogger(__name__)
# The ``daytona`` extra's pin.
SDK_VERSION = "0.227.0"
# Bump when the image recipe changes: every name changes with it.
RECIPE = "d2"
DEFAULT_API_URL = "https://app.daytona.io/api"
# How long a create waits for the sandbox to start.
CREATE_SEC = 180
DELETE_SEC = 60
# A sandbox whose delete was refused is stopped, then read until it is gone
# for this long, this often; one still starting or building is waited for
# first, up to CREATE_SEC (a stop sent then does not take effect).
GONE_SEC = 30.0
STOP_POLL_SEC = 1.0
# A create without an answer whose name still finds nothing this long
# after its request never landed: its ledger entry ends.
PENDING_SEC = 2 * CREATE_SEC
# A template's build sandbox refused for capacity is asked for again, for
# this long in all, backing off from the first wait to the last.
CAPACITY_WAIT_SEC = 600.0
CAPACITY_RETRY_SEC = (5.0, 60.0)
# After a capacity refusal the sandbox is read by name, again after these
# waits while the read itself fails, to prove that nothing landed.
LANDED_RETRY_SEC = (1.0, 2.0, 4.0)
# Minutes a create waits for a runner before Daytona cancels it (destroys
# the sandbox: a DaytonaQueueTimeoutError, typed as capacity), below
# CREATE_SEC. Daytona honors it only where the organization enables queue
# timeouts; otherwise its own default (or none) applies.
QUEUE_MIN = max(1, CREATE_SEC // 60 - 1)
# After a command's log ends, its exit code is asked for this many times.
EXIT_TRIES = 25
EXIT_POLL_SEC = 0.2
CLOSE_SEC = 10.0
# Sandbox states that never run again by themselves.
_DEAD = frozenset({"error", "build_failed", "stopped", "archived"})
# On the way to running: a stop waits them out.
_SETTLING = frozenset(
    {
        "creating",
        "restoring",
        "starting",
        "pending_build",
        "building_snapshot",
        "pulling_snapshot",
    }
)
# Stopped or on the way out: no stop is sent.
_ENDING = frozenset({"stopping", "stopped", "destroying", "destroyed", "archived"})
# A refused create that is the provider's capacity (the organization's
# quota, its runners), on an API refusal of these statuses: organization
# wide wording only. A per-sandbox size or a validation refusal is not one,
# waiting never helps it. The real quota text is not yet captured.
_CAPACITY_STATUS = frozenset({400, 403, 422, 503})
_CAPACITY = re.compile(
    r"quota|total \w+ limit|organi[sz]ation\W+(?:\w+\W+){0,3}limit"
    r"|insufficient (?:resources|capacity)|(?:at|out of|no) capacity"
    r"|no (?:available )?runners?",
    re.IGNORECASE,
)
_NOT_CAPACITY = re.compile(r"per sandbox|must be|invalid", re.IGNORECASE)
# $1 the cwd, then NAME=value pairs and the command, set by ``env`` (which
# takes any name, as Docker does; ``export`` fails on one that is not a
# shell name), with stdin closed (an async session command's stdin is a
# pipe kept open). The command's own name never holds "=": it is sh or a
# broker command.
_RUN_SCRIPT = """cd "$1" || exit 126
shift
exec env -- "$@" < /dev/null"""


def read_api_key(
    settings: EnvDaytonaHost, environ: Mapping[str, str] | None = None
) -> str:
    """The operator's Daytona API key; the errors name where it is, never it."""
    if settings.api_key_file is not None:
        where = f"api_key_file {settings.api_key_file}"
        try:
            text = Path(settings.api_key_file).read_text()
        except (OSError, UnicodeError) as error:
            raise SetupError(
                f"sandbox policy environments.host.daytona: {where} is unreadable "
                f"({type(error).__name__})"
            ) from None
    else:
        where = f"api_key_env {settings.api_key_env}"
        text = (os.environ if environ is None else environ).get(
            settings.api_key_env or "", ""
        )
    key = text.strip()
    if not key or any(character.isspace() for character in key):
        raise SetupError(
            f"sandbox policy environments.host.daytona: {where} holds no key"
        )
    return key


def check_daytona_host(settings: EnvDaytonaHost) -> None:
    """As check_e2b_host, for the daytona extra and its key. The SDK reads
    no endpoint or credential from the environment or a .env file when, as
    here, the key, api_url and target are all passed: nothing is refused."""

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    installed = version("daytona")
    if installed != SDK_VERSION:
        raise SetupError(
            f'backend = "daytona" needs daytona=={SDK_VERSION} '
            f"(found {installed or 'none'}); "
            "install rsi-harness with its daytona extra: "
            "uv tool install '/path/to/RSI-Harness[daytona]'"
        )
    read_api_key(settings)
    if settings.registry_auth_file is not None:
        read_registry_auth(settings.registry_auth_file, "daytona")


# -- the proxy ---------------------------------------------------------------------


async def hook_proxy(sdk: Any, proxy: str) -> None:
    """Send every connection of ``sdk`` (an AsyncDaytona) through ``proxy``.

    The SDK funnels all of its async traffic through one aiohttp session
    (internal.shared_session.SharedAiohttpSession), which it makes on the
    first request with ``trust_env``: the API's and the toolbox's REST
    clients, the toolbox's log websockets and file uploads and downloads.
    Before any request, this hands that coordinator a session of the same
    settings whose default proxy is ``proxy``, so the SDK never makes its
    own and no process variable is set. The event gateway's socket.io
    connection would not use it: the client turns it off (polling)."""
    import aiohttp

    shared = sdk._shared_session
    rest = sdk._api_client.rest_client
    connector = aiohttp.TCPConnector(
        limit=rest.maxsize,
        keepalive_timeout=30,
        ssl=rest.ssl_context,
        **shared._connector_overrides,
    )
    shared._adopt(aiohttp.ClientSession(connector=connector, proxy=proxy))


# -- processes ---------------------------------------------------------------------


@dataclass(eq=False)
class _Command:
    """One session command: what following it again needs, and how much of
    each stream (stdout, stderr) was handed out, in the characters the log
    sends."""

    sandbox_id: str
    session_id: str
    cmd_id: str
    # A group_argv command: its first stdout line is its group's pid.
    grouped: bool = False
    pid: int | None = None
    line: str = ""
    sent: list[int] = field(default_factory=lambda: [0, 0])


class _DaytonaProcess:
    """One follow of a command's log, on the client's loop; ``close`` stops
    following and leaves the command (and its session) running."""

    def __init__(self, client: SdkDaytonaClient, box: Any, command: _Command) -> None:
        self.pid = 0  # unknown: group_argv reports the group's
        self.exit_code: int | None = None
        self._clean = client._clean
        self._queue: queue.Queue = queue.Queue()
        self._future = client._submit(self._follow(client, box, command))

    async def _follow(
        self, client: SdkDaytonaClient, box: Any, command: _Command
    ) -> None:
        skip = list(command.sent)

        def handler(stream: int) -> Callable[[str], None]:
            def handle(text: str) -> None:
                index = stream - 1
                if skip[index]:
                    cut = min(skip[index], len(text))
                    skip[index] -= cut
                    text = text[cut:]
                if not text:
                    return
                command.sent[index] += len(text)
                if stream == 1 and command.grouped and command.pid is None:
                    client._pid_line(command, text)
                self._queue.put((stream, text.encode()))

            return handle

        try:
            await box.process.get_session_command_logs_async(
                command.session_id, command.cmd_id, handler(1), handler(2)
            )
            code = await client._exit(box, command)
        except Exception as error:
            self._queue.put(("error", error))
            return
        self._queue.put(("end", code))

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        while True:
            stream, item = self._queue.get()
            if stream == "closed":
                return
            if stream == "error":
                raise self._clean(item)
            if stream == "end":
                self.exit_code = item
                return
            yield stream, item

    def close(self) -> None:
        self._future.cancel()
        self._queue.put(("closed", None))


class SdkDaytonaClient:
    """``RemoteClient`` over the pinned SDK (``daytona==0.227.0``).

    The key, ``api_url`` and ``target`` are always passed, so the SDK reads
    neither from the environment nor from a .env file; a proxy is set on
    the SDK's own session (``hook_proxy``), hence the exact pin.
    """

    name = "daytona"
    recipe = RECIPE
    # Emulated by sandbox_remote: Daytona's container sandboxes have none.
    native_pid = False
    native_user = False
    freeze = "signal"
    # The daemon (PID 1) runs every command; the freeze's own session shell
    # waits for its helpers before it reports the freeze's exit.
    freeze_spare: tuple[str, ...] = ("init", "caller")
    argv_bytes = None
    build_form = None
    # The run's sandbox ledger (sandbox_remote.bind_ledger); None: none.
    ledger: SandboxLedger | None = None

    def __init__(
        self,
        api_key: str,
        *,
        target: str,
        api_url: str | None = None,
        template_prefix: str = "rsi",
        proxy: str | None = None,
        registry_auth_file: str | None = None,
    ) -> None:
        import daytona

        self._daytona = daytona
        self.recipe = f"{RECIPE}-{target}"
        self._key = api_key
        self._prefix = template_prefix
        self._proxy = proxy
        self._registry_auth_file = registry_auth_file
        self._lock = threading.Lock()
        self._closed = False
        self._boxes: dict[str, Any] = {}
        # Template name -> (image reference, shape).
        self._templates: dict[str, tuple[str, tuple[int, int, int]]] = {}
        # (sandbox id, group pid) -> its command, until its exit is seen.
        self._commands: dict[tuple[str, int], _Command] = {}
        self._processes: weakref.WeakSet[_DaytonaProcess] = weakref.WeakSet()
        # Set at the first refused delete: from then on a sandbox is stopped.
        self._no_delete = False
        # Sandbox id -> when a stop was last sent without waiting (state).
        self._stopped_at: dict[str, float] = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="rsi-sandbox-daytona", daemon=True
        )
        self._thread.start()
        self._sdk = self._call(self._open(target, api_url or DEFAULT_API_URL))

    async def _open(self, target: str, api_url: str) -> Any:
        daytona = self._daytona
        sdk = daytona.AsyncDaytona(
            daytona.DaytonaConfig(
                api_key=self._key,
                api_url=api_url,
                target=target,
                use_deprecated_polling=True,
                # No connection limit: each followed log holds one while its
                # process runs, and the broker's own calls must not wait.
                connection_pool_maxsize=None,
            )
        )
        if self._proxy is not None:
            await hook_proxy(sdk, self._proxy)
        return sdk

    # -- the loop --------------------------------------------------------------

    def _hide(self, text: str, *secrets: str) -> str:
        for secret in (self._key, *secrets):
            text = text.replace(secret, "***")
        return text

    def _clean(self, error: BaseException, *secrets: str) -> InfrastructureError:
        if isinstance(error, InfrastructureError):
            return error
        text = self._hide(str(error), *secrets)[:512]
        return InfrastructureError(f"daytona {type(error).__name__}: {text}")

    def _submit(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        if self._closed:
            coroutine.close()
            raise InfrastructureError("the daytona client is closed")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _call(self, coroutine: Coroutine[Any, Any, Any], *secrets: str) -> Any:
        try:
            return self._submit(coroutine).result()
        except (SandboxError, FileNotFoundError, InfrastructureError):
            raise
        except Exception as error:
            raise self._clean(error, *secrets) from None

    async def _get(self, id_or_name: str) -> Any:
        """The sandbox (cached for its files and processes), or None."""
        try:
            box = await self._sdk.get(id_or_name)
        except self._daytona.DaytonaNotFoundError:
            return None
        with self._lock:
            self._boxes[box.id] = box
        return box

    def _box(self, sandbox_id: str) -> Any:
        with self._lock:
            box = self._boxes.get(sandbox_id)
        if box is None:
            box = self._call(self._get(sandbox_id))
            if box is None:
                raise InfrastructureError(f"daytona sandbox {sandbox_id} is gone")
        return box

    def _kinds(self, *names: str) -> tuple[type[BaseException], ...]:
        """The SDK's error classes of these names (those it has)."""
        return tuple(
            kind
            for kind in (getattr(self._daytona, name, None) for name in names)
            if isinstance(kind, type)
        )

    def _forbidden(self, error: BaseException) -> bool:
        return (
            isinstance(error, self._kinds("DaytonaForbiddenError"))
            or getattr(error, "status_code", None) == 403
        )

    def _capacity(self, error: BaseException) -> bool:
        """Whether a refused create may be the provider's capacity: a rate
        limit, the API's queue timeout (it destroyed the sandbox: no runner
        came), or an API refusal naming a quota or capacity. A failure
        after the sandbox existed (it carries its id) never is; whether
        anything landed is proven by ``_create``."""
        if isinstance(error, self._kinds("DaytonaQueueTimeoutError")):
            return True
        if getattr(error, "sandbox_id", None):
            return False
        status = getattr(error, "status_code", None)
        if status == 429 or isinstance(error, self._kinds("DaytonaRateLimitError")):
            return True
        refusal = self._kinds(
            "DaytonaForbiddenError",
            "DaytonaBadRequestError",
            "DaytonaUnprocessableEntityError",
            "DaytonaServiceUnavailableError",
        )
        if not (isinstance(error, refusal) or status in _CAPACITY_STATUS):
            return False
        text = str(error)
        return bool(_CAPACITY.search(text)) and not _NOT_CAPACITY.search(text)

    def _login(self, reference: str) -> tuple[str, str] | None:
        if self._registry_auth_file is None:
            return None
        registries = read_registry_auth(self._registry_auth_file, "daytona")
        return registries.get(reference.partition("/")[0].lower())

    # -- templates -----------------------------------------------------------

    def shape(self, cpus, memory_mb, disk_mb):
        return sandbox_shape(cpus, memory_mb, disk_mb)

    def template_exists(self, name):
        with self._lock:
            return name in self._templates

    def image_config(self, repository, tag):
        login = self._login(repository)
        try:
            return registry_image_config(
                repository, tag, proxy=self._proxy, login=login
            )
        except Exception as error:
            raise self._clean(error, *(() if login is None else (login[1],))) from None

    def build_template(
        self, name, image, shape, *, log, steps=(), context=None, force=False
    ):
        assert not steps, "the daytona backend has no image_build"

        # A sandbox of the image makes Daytona build and cache it; being
        # ephemeral with a short auto-stop, it goes even if the broker dies.
        def params() -> Any:
            return self._daytona.CreateSandboxFromImageParams(
                image=self._image(image),
                resources=self._resources(shape),
                name=f"{name}-{secrets.token_hex(4)}",
                os_user="root",
                public=False,
                ephemeral=True,
                auto_stop_interval=1,
                queue_timeout=QUEUE_MIN,
                network_block_all=True,
            )

        log(f"daytona builds {image}\n")
        box = self._build_sandbox(params, image, log)
        try:
            self._call(self._remove(box))
        except InfrastructureError as error:
            # The image is built; the sandbox stops by itself within a
            # minute (its auto-stop), and the ledger keeps it for recovery.
            LOGGER.warning("daytona build sandbox of %s left: %s", image, error)
            log(f"daytona build sandbox left: {error}\n")
        with self._lock:
            self._templates[name] = (image, shape)

    def _build_sandbox(
        self, params: Callable[[], Any], image: str, log: Callable[[str], None]
    ) -> Any:
        """The template's build sandbox, asked for again while Daytona
        refuses it for capacity, up to CAPACITY_WAIT_SEC. One whose create
        failed otherwise stays in the ledger, where recovery finds it."""
        # In the ledger only: recovery of the run finds it, an env's kill not.
        metadata = (
            {}
            if self.ledger is None
            else {
                **run_metadata(self.ledger.run_id),
                f"{META_PREFIX}role": "template-build",
            }
        )
        deadline = time.monotonic() + CAPACITY_WAIT_SEC
        wait, longest = CAPACITY_RETRY_SEC
        while True:
            try:
                return self._create(
                    params(),
                    timeout=0,
                    metadata=metadata,
                    on_snapshot_create_logs=lambda line: log(line + "\n"),
                )
            except ProviderCapacityError as error:
                if time.monotonic() + wait > deadline:
                    raise
                log(f"daytona is at capacity; building {image} again in {wait:.0f} s\n")
                LOGGER.info("daytona build of %s waits for capacity: %s", image, error)
                time.sleep(wait)
                wait = min(longest, wait * 2)

    def _create(
        self,
        params: Any,
        *,
        timeout: float,
        metadata: Mapping[str, str],
        **options: Any,
    ) -> Any:
        """Create the sandbox ``params`` name, in the ledger by name before
        the request and by id after. A capacity refusal is a
        ProviderCapacityError, and its ledger entry ends, only once a read
        by name finds nothing: the SDK's polls after the request can be
        refused too. One that landed is recorded and its refusal is a plain
        InfrastructureError (the caller removes it); one whose read keeps
        failing stays open in the ledger, for recovery."""
        ledger = self.ledger
        if ledger is not None:
            ledger.creating(params.name, metadata)

        async def create() -> Any:
            try:
                return await self._sdk.create(params, timeout=timeout, **options)
            except Exception as error:
                if not self._capacity(error):
                    raise
                text = self._hide(str(error))[:512]
                try:
                    landed = await self._landed(params.name)
                except Exception as failure:
                    raise InfrastructureError(
                        f"daytona refused the sandbox ({text}), and whether "
                        f"{params.name} landed is unknown: {self._clean(failure)}"
                    ) from None
                if landed is not None:
                    raise _Landed(
                        landed,
                        f"daytona sandbox {landed} ({params.name}) was created, "
                        f"then refused while it started: {text}",
                    ) from None
                raise ProviderCapacityError(
                    f"daytona refused the sandbox for capacity: {text}"
                ) from None

        try:
            box = self._call(create())
        except ProviderCapacityError:
            if ledger is not None:
                ledger.ended("refused", name=params.name)
            raise
        except _Landed as landed:
            if ledger is not None:
                ledger.created(params.name, landed.sandbox_id)
            raise
        if ledger is not None:
            ledger.created(params.name, box.id)
        return box

    async def _landed(self, name: str) -> str | None:
        """The id of the sandbox ``name`` if it exists, None if a read proves
        it does not; a read failing every time raises its error."""
        failure: BaseException | None = None
        for wait in (0.0, *LANDED_RETRY_SEC):
            if wait:
                await asyncio.sleep(wait)
            try:
                box = await self._read(name)
            except Exception as error:
                failure = error
                continue
            if box is None or _value(box.state) == "destroyed":
                return None
            return box.id
        assert failure is not None
        raise failure

    def _image(self, image: str) -> Any:
        return self._daytona.Image.base(image).entrypoint(["sleep", "infinity"])

    def _resources(self, shape: tuple[int, int, int]) -> Any:
        cpus, memory_gb, disk_gb = shape
        return self._daytona.Resources(cpu=cpus, memory=memory_gb, disk=disk_gb)

    # -- sandboxes -----------------------------------------------------------

    def _name(self, env_id: str) -> str:
        return f"{self._prefix}-{env_id}"

    def create(self, template, *, timeout, metadata, allow_internet, allow_out=()):
        network: dict[str, Any] = {}
        if allow_out:
            # An IP is a /32; hostnames and IPs never mix (refused before).
            cidrs = [
                host if "/" in host else f"{host}/32"
                for host in allow_out
                if re.fullmatch(r"[0-9./]+", host)
            ]
            domains = [
                host for host in allow_out if not re.fullmatch(r"[0-9./]+", host)
            ]
            if cidrs:
                network["network_allow_list"] = ",".join(cidrs)
            if domains:
                network["domain_allow_list"] = ",".join(domains)
        elif not allow_internet:
            network["network_block_all"] = True
        minutes = math.ceil(timeout / 60)
        with self._lock:
            image, shape = self._templates[template]
        params = self._daytona.CreateSandboxFromImageParams(
            image=self._image(image),
            resources=self._resources(shape),
            name=self._name(metadata[f"{META_PREFIX}env_id"]),
            os_user="root",
            labels=dict(metadata),
            public=False,
            ephemeral=True,
            ttl_minutes=minutes,
            auto_stop_interval=minutes,
            queue_timeout=QUEUE_MIN,
            **network,
        )
        box = self._create(params, timeout=CREATE_SEC, metadata=metadata)
        with self._lock:
            self._boxes[box.id] = box
        return box.id

    def image_env(self, sandbox_id, image_env):
        return config_env(image_env)

    def lookup(self, env_id):
        box = self._call(self._get(self._name(env_id)))
        if box is None or _value(box.state) == "destroyed":
            return None
        return box.id

    async def _read(self, id_or_name: str) -> Any:
        """The sandbox as the API has it now (not cached), or None."""
        try:
            return await self._sdk.get(id_or_name)
        except self._daytona.DaytonaNotFoundError:
            return None

    def _ended(self, outcome: str, sandbox_id: str) -> None:
        self._stopped_at.pop(sandbox_id, None)
        if self.ledger is not None:
            self.ledger.ended(outcome, sandbox_id=sandbox_id)

    async def _remove(self, box: Any, *, confirm: bool = True) -> str:
        """End a sandbox: "deleted" ("gone": it already was). When the key
        may not delete, a stop removes the ephemeral sandbox: one starting
        or building is waited for first, then the stop is sent and, with
        ``confirm``, the sandbox read until it is gone ("stopped"); still
        there after GONE_SEC, it is a leftover (InfrastructureError). Without
        ``confirm`` the stop is only sent ("stopping"), once per GONE_SEC."""
        sandbox_id = box.id
        if not self._no_delete:
            try:
                await box.delete(timeout=DELETE_SEC)
            except self._daytona.DaytonaNotFoundError:
                self._ended("gone", sandbox_id)
                return "gone"
            except Exception as error:
                if not self._forbidden(error):
                    raise
                LOGGER.info(
                    "daytona refused deleting sandbox %s (%s): it is stopped "
                    "instead, which removes an ephemeral sandbox",
                    sandbox_id,
                    type(error).__name__,
                )
                self._no_delete = True
            else:
                self._ended("deleted", sandbox_id)
                return "deleted"
        state = _value(box.state)
        settle = time.monotonic() + CREATE_SEC
        while state in _SETTLING and time.monotonic() < settle:
            await asyncio.sleep(STOP_POLL_SEC)
            fresh = await self._read(sandbox_id)
            if fresh is None:
                self._ended("gone", sandbox_id)
                return "gone"
            state = _value(fresh.state)
        failure = ""
        recent = self._stopped_at.get(sandbox_id)
        if state not in _ENDING and (
            confirm or recent is None or time.monotonic() - recent >= GONE_SEC
        ):
            self._stopped_at[sandbox_id] = time.monotonic()
            try:
                await box.stop(timeout=DELETE_SEC, force=True)
            except self._daytona.DaytonaNotFoundError:
                self._ended("stopped", sandbox_id)
                return "stopped"
            except Exception as error:
                failure = f"; its stop failed: {self._clean(error)}"
        if not confirm:
            return "stopping"
        end = time.monotonic() + GONE_SEC
        while True:
            fresh = await self._read(sandbox_id)
            if fresh is None or _value(fresh.state) == "destroyed":
                self._ended("stopped", sandbox_id)
                return "stopped"
            if time.monotonic() >= end:
                break
            await asyncio.sleep(STOP_POLL_SEC)
        state = _value(fresh.state)
        if self.ledger is not None:
            self.ledger.noted(sandbox_id, f"leftover ({state})")
        raise InfrastructureError(
            f"daytona sandbox {sandbox_id} is still there {GONE_SEC:.0f} s after "
            f"its stop (state {state}{failure}); the key may not delete it, so "
            "it holds quota until its TTL: stop or delete it in Daytona"
        )

    def state(self, sandbox_id):
        async def state() -> str | None:
            box = await self._get(sandbox_id)
            found = None if box is None else _value(box.state)
            if found in (None, "destroyed"):
                return None
            if found in _DEAD:
                # It would linger: remove it, and the caller waits it out.
                await self._remove(box, confirm=False)
            return "running" if found == "started" else found

        found = self._call(state())
        if found is None:
            self._ended("gone", sandbox_id)
        return found

    def kill(self, sandbox_id):
        async def kill() -> None:
            box = await self._get(sandbox_id)
            if box is None or _value(box.state) == "destroyed":
                self._ended("gone", sandbox_id)
            elif _value(box.state) != "destroying":
                await self._remove(box)

        try:
            self._call(kill())
        finally:
            with self._lock:
                self._boxes.pop(sandbox_id, None)

    def list(self, metadata):
        """The listing merged with the ledger's open entries of
        ``metadata``, each read again: one gone (or, unanswered, never
        landed) ends there. With a ledger, a refused listing counts as
        empty, as a key that may not list sees it."""
        query = self._daytona.ListSandboxesQuery(labels=dict(metadata))

        async def listing() -> list[tuple[str, dict[str, str]]]:
            try:
                return [
                    (
                        box.id,
                        {
                            key: value
                            for key, value in (box.labels or {}).items()
                            if key.startswith(META_PREFIX)
                        },
                    )
                    async for box in self._sdk.list(query)
                    if _value(box.state) != "destroyed"
                ]
            except Exception as error:
                if self.ledger is None or not self._forbidden(error):
                    raise
                LOGGER.info("daytona refused the listing: the ledger answers")
                return []

        found = dict(self._call(listing()))
        if self.ledger is not None:
            for entry in self.ledger.open_entries(metadata):
                if entry.sandbox_id in found:
                    continue
                sandbox_id = self._confirm(self.ledger, entry)
                if sandbox_id is not None and sandbox_id not in found:
                    found[sandbox_id] = {
                        key: value
                        for key, value in entry.metadata.items()
                        if key.startswith(META_PREFIX)
                    }
        return list(found.items())

    def _confirm(self, ledger: SandboxLedger, entry: LedgerEntry) -> str | None:
        """The id of a ledger entry's sandbox if it is still there."""
        box = self._call(self._read(entry.sandbox_id or entry.name))
        if box is not None and _value(box.state) != "destroyed":
            if entry.sandbox_id is None:
                ledger.created(entry.name, box.id)
            return box.id
        if entry.sandbox_id is not None:
            ledger.ended("gone", sandbox_id=entry.sandbox_id)
        elif ledger.clock() - entry.created_at >= PENDING_SEC:
            ledger.ended("absent", name=entry.name)
        return None

    # -- files and processes -------------------------------------------------

    def write_file(self, sandbox_id, path, data):
        box = self._box(sandbox_id)
        if isinstance(data, (bytes, bytearray, memoryview)):
            source: bytes | int = bytes(data)
        else:
            try:
                # upload_files opens a str source itself and streams it;
                # open() takes a descriptor too, and closes this duplicate.
                source = os.dup(data.fileno())
            except (AttributeError, OSError):
                source = data.read()
        upload = self._daytona.FileUpload(source=source, destination=path)
        self._call(box.fs.upload_files([upload], timeout=0))

    def read_file(self, sandbox_id, path):
        box = self._box(sandbox_id)

        async def read() -> bytes:
            try:
                return bytes(await box.fs.download_file(path))
            except self._daytona.DaytonaNotFoundError:
                raise FileNotFoundError(path) from None

        return self._call(read())

    def read_stream(self, sandbox_id, path):
        """The file's download, chunk by chunk; closing it ends the request."""

        def chunks() -> Iterator[bytes]:
            box = self._box(sandbox_id)
            stream = self._call(box.fs.download_file_stream(path, timeout=0))
            try:
                while True:
                    chunk = self._call(_next(stream))
                    if chunk is None:
                        return
                    yield chunk
            finally:
                # Ends the request when the reader closes early.
                if not self._closed:
                    try:
                        self._call(stream.aclose())
                    except InfrastructureError as error:
                        LOGGER.info(
                            "daytona download of %s not closed: %s", path, error
                        )

        return ChunkReader(chunks, self._clean)

    def _pid_line(self, command: _Command, text: str) -> None:
        """A grouped command's first stdout line names it for ``connect``."""
        command.line += text
        head, newline, _ = command.line.partition("\n")
        if newline:
            command.pid = int(head) if head.isdigit() else 0
            with self._lock:
                self._commands[(command.sandbox_id, command.pid)] = command

    async def _exit(self, box: Any, command: _Command) -> int:
        """The exit code of a command whose log ended; then its session goes."""
        for _ in range(EXIT_TRIES):
            found = await box.process.get_session_command(
                command.session_id, command.cmd_id
            )
            if found.exit_code is not None:
                break
            await asyncio.sleep(EXIT_POLL_SEC)
        else:
            raise InfrastructureError("daytona command log ended before its exit")
        if command.pid is not None:
            with self._lock:
                self._commands.pop((command.sandbox_id, command.pid), None)
        try:
            await box.process.delete_session(command.session_id)
        except Exception as error:
            LOGGER.info("daytona session %s not deleted: %s", command.session_id, error)
        return int(found.exit_code)

    def _process(self, box: Any, command: _Command) -> _DaytonaProcess:
        process = _DaytonaProcess(self, box, command)
        self._processes.add(process)
        return process

    def start(self, sandbox_id, argv, *, envs, cwd, user):
        assert user == "root", "Daytona starts processes as root only"
        box = self._box(sandbox_id)
        line = shlex.join(
            [
                "sh",
                "-c",
                _RUN_SCRIPT,
                "rsi",
                cwd,
                *(f"{key}={value}" for key, value in envs.items()),
                *argv,
            ]
        )
        session_id = f"rsi-{secrets.token_hex(8)}"
        request = self._daytona.SessionExecuteRequest(command=line, run_async=True)

        async def begin() -> str:
            await box.process.create_session(session_id)
            response = await box.process.execute_session_command(session_id, request)
            return response.cmd_id

        command = _Command(
            sandbox_id,
            session_id,
            self._call(begin()),
            grouped=list(argv[:4]) == group_argv([]),
        )
        return self._process(box, command)

    def connect(self, sandbox_id, pid):
        with self._lock:
            command = self._commands.get((sandbox_id, pid))
        if command is None:
            raise InfrastructureError(f"daytona has no command of pid {pid}")
        return self._process(self._box(sandbox_id), command)

    def close(self):
        """Stop following every process (their sessions stay) and end the
        SDK's connections and the loop."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._boxes.clear()
        for process in list(self._processes):
            process.close()
        try:
            asyncio.run_coroutine_threadsafe(self._sdk.close(), self._loop).result(
                CLOSE_SEC
            )
        except Exception as error:
            LOGGER.info("daytona client close: %s", error)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(CLOSE_SEC)
        if not self._thread.is_alive():
            self._loop.close()


class _Landed(InfrastructureError):
    """A create refused after its sandbox existed: not capacity."""

    def __init__(self, sandbox_id: str, message: str) -> None:
        super().__init__(message)
        self.sandbox_id = sandbox_id


def sandbox_shape(cpus: float, memory_mb: int, disk_mb: int) -> tuple[int, int, int]:
    """Daytona sizes a sandbox in whole cores and GiB: round up."""
    return (
        max(1, math.ceil(cpus)),
        max(1, math.ceil(memory_mb / 1024)),
        max(1, math.ceil(disk_mb / 1024)),
    )


def _value(state: Any) -> str:
    """A state enum's wire value."""
    return str(getattr(state, "value", state))


async def _next(stream: Any) -> bytes | None:
    return await anext(stream, None)


def daytona_client(
    settings: EnvDaytonaHost, api_key: str | None = None
) -> SdkDaytonaClient:
    """The production client; the key (read here unless the caller already
    did) is kept only by it."""
    return SdkDaytonaClient(
        read_api_key(settings) if api_key is None else api_key,
        target=settings.target,
        api_url=settings.api_url,
        template_prefix=settings.template_prefix,
        proxy=settings.proxy,
        registry_auth_file=settings.registry_auth_file,
    )


BACKEND = RemoteBackend(
    check_host=check_daytona_host,
    credentials=lambda settings: (read_api_key(settings),),
    client=lambda credentials, settings: daytona_client(settings, *credentials),
)


__all__ = [
    "BACKEND",
    "RECIPE",
    "SDK_VERSION",
    "SdkDaytonaClient",
    "check_daytona_host",
    "daytona_client",
    "hook_proxy",
    "read_api_key",
    "sandbox_shape",
]
