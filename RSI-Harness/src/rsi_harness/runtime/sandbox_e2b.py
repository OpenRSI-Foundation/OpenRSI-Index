"""E2B-hosted env services: the broker's second env backend (V1, local runs).

``e2b_env_runtime`` is an ``EnvRuntime`` beside ``docker_env_runtime``: the
whole env layer (sessions, two-key grants, quotas, journal, request replay,
freeze and close ordering, the wire protocol and the plugin) stays
``SandboxEnvs``'s; only where a service runs changes. One env is one E2B
sandbox running one service.

Templates are made on demand: an image reference (its digest when given)
plus the phase's rounded per-container cpus and memory and ``RECIPE`` name
one template deterministically. If E2B already has a template of that name
it is used, otherwise ``image_pull`` builds it and waits: E2B itself is the
cache, and one broker builds each name once at a time. The recipe resets
the user to root and the workdir to /, and records the image's ENV in
``IMAGE_ENV_FILE``, which every exec applies again (E2B starts processes
with its own environment). E2B keeps nothing else of the image config, so
the pull reads it from the image's registry (``registry_image_config``)
and env_create applies its WORKDIR, USER, ENTRYPOINT and CMD as the Engine
does. A sandbox's timeout stays within the team's maximum sandbox length.

Execs reuse ``ExecPump`` unchanged through ``E2BExecApi``, the slice of
Docker's exec API the pump calls: each command runs under ``setsid`` (its
own process group) through envd without a login shell, its output is
reframed as the Engine's multiplexed stream, and signals reach the group
through ``kill`` run as root in the sandbox. Copies are tar files moved
with envd's file API and packed or unpacked by ``tar`` in the sandbox.

The API key lives only here: the broker reads it (``read_api_key``) and
hands it to the SDK; leases, plans, sandbox metadata, capabilities and
logs carry where the key is, never the key. The SDK (``e2b==2.52.0``, the
``e2b`` extra) is imported only by ``SdkE2BClient``.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import posixpath
import re
import secrets
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from docker.errors import NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.sandbox_archive import (
    SPECIAL_ROOTS,
    ArchiveTransfer,
    StageStore,
    _exclusions,
    _require_path,
)
from rsi_harness.runtime.sandbox_contracts import (
    EnvE2BHost,
    SandboxError,
    SandboxOwner,
    below,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    EnvReason,
    EnvSpec,
    SandboxEnvLease,
    SandboxEnvServiceLease,
    SandboxImageLease,
    env_container_name,
    env_spec_digest,
)
from rsi_harness.runtime.sandbox_env_docker import (
    EnvStartResult,
    ServiceStatus,
    _clean_dir,
    _revise,
    _service,
)
from rsi_harness.runtime.sandbox_envs import EnvRuntime, spool_floor
from rsi_harness.runtime.sandbox_exec import (
    SIGNALS,
    START_CONFIRM_SEC,
    ExecProcess,
    ExecPump,
    ExecTarget,
)
from rsi_harness.runtime.sandbox_images import (
    Image,
    JobFailed,
    image_view,
    reference_text,
)

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
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
# The broker's own commands (tar, stat, kill) as root.
ROOT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SCRATCH_DIR = "/tmp"
# The sandbox outlives the env deadline by this much, then E2B kills it:
# the dead-man switch when the broker is gone.
TIMEOUT_SLACK_SEC = 60
# Default for [environments.host.e2b] max_sandbox_hours.
MAX_TIMEOUT_SEC = 3_600
# The image config keys env_create applies or image_view reports. E2B runs
# no image healthcheck and stops with TERM: those are left out.
IMAGE_CONFIG_KEYS = ("Env", "WorkingDir", "User", "Entrypoint", "Cmd", "Volumes")
REGISTRY_HOSTS = {"docker.io": "registry-1.docker.io"}
MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
TAIL_BYTES = 4096
POLL_SEC = 0.05
STOP_PROOF_SEC = 10.0
REATTACH_SEC = 1.0
# Failed reconnects to a running sandbox before a stream counts as lost.
REATTACH_TRIES = 3
EXEC_RECORD_SEC = 600.0
MEMORY_STEP_MB = 512
META_PREFIX = "rsi_"
_FRAME = struct.Struct("!BxxxI")
_SIGNAL_NAMES = {number: name for name, number in SIGNALS.items()}
ENV_ROLE = "sandbox-env"


def unsupported(field_name: str, what: str) -> SandboxError:
    return SandboxError(
        "unsupported", field_name, f"{what} unsupported on the e2b backend"
    )


# -- settings and the API key ---------------------------------------------------


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


def sandbox_metadata(owner: SandboxOwner, env_id: str, service: str) -> dict[str, str]:
    """What every sandbox carries: run, task, phase (and round), env, service.

    ``rsi-harness recover`` and ``cleanup`` find a run's sandboxes by it."""
    metadata = {
        f"{META_PREFIX}run_id": owner.run_id,
        f"{META_PREFIX}task_id": owner.task_id,
        f"{META_PREFIX}phase": owner.phase,
        f"{META_PREFIX}role": ENV_ROLE,
        f"{META_PREFIX}env_id": env_id,
        f"{META_PREFIX}service": service,
    }
    if owner.round_id is not None:
        metadata[f"{META_PREFIX}round_id"] = owner.round_id
    return metadata


def run_metadata(run_id: str) -> dict[str, str]:
    return {f"{META_PREFIX}run_id": run_id}


# -- the SDK surface ------------------------------------------------------------


class E2BProcess(Protocol):
    """One process started through envd: ``(1|2, bytes)`` output until it
    ends; then ``exit_code`` is set (None when the stream broke first), a
    signal death as 128 + its number (``shell_exit_code``)."""

    pid: int
    exit_code: int | None

    def __iter__(self) -> Iterator[tuple[int, bytes]]: ...

    def close(self) -> None: ...


class E2BClient(Protocol):
    """The slice of the E2B SDK this backend uses (faked in tests)."""

    def template_exists(self, name: str) -> bool: ...

    def image_config(self, repository: str, tag: str) -> dict[str, Any]: ...

    def build_template(
        self,
        name: str,
        image: str,
        *,
        cpu_count: int,
        memory_mb: int,
        log: Callable[[str], None],
    ) -> None: ...

    def create(
        self,
        template: str,
        *,
        timeout: int,
        metadata: Mapping[str, str],
        allow_internet: bool,
    ) -> str: ...

    def state(self, sandbox_id: str) -> str | None: ...

    def kill(self, sandbox_id: str) -> None: ...

    def pause(self, sandbox_id: str) -> None: ...

    def resume(self, sandbox_id: str, *, timeout: int) -> None: ...

    def list(self, metadata: Mapping[str, str]) -> list[tuple[str, dict[str, str]]]: ...

    def write_file(self, sandbox_id: str, path: str, data: bytes | Any) -> None: ...

    def read_file(self, sandbox_id: str, path: str) -> bytes: ...

    def start(
        self,
        sandbox_id: str,
        argv: Sequence[str],
        *,
        envs: Mapping[str, str],
        cwd: str,
        user: str,
    ) -> E2BProcess: ...

    def connect(self, sandbox_id: str, pid: int) -> E2BProcess: ...

    def close(self) -> None: ...


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


def _registry_token(http: Any, challenge: str) -> str:
    """An anonymous pull token from the registry's Bearer challenge."""
    scheme, _, params = challenge.partition(" ")
    if scheme.lower() != "bearer":
        raise InfrastructureError("the registry wants credentials (not supported)")
    # realm="...",service="...",scope="repository:a/b:pull" (a scope may
    # hold commas inside its quotes).
    fields = dict(re.findall(r'(\w+)="([^"]*)"', params))
    realm = fields.pop("realm", None)
    if not realm:
        raise InfrastructureError("the registry's auth challenge names no realm")
    response = http.get(realm, params=fields)
    response.raise_for_status()
    body = response.json()
    return str(body.get("token") or body.get("access_token") or "")


def registry_image_config(
    repository: str, tag: str, *, proxy: str | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    """The image's config (WorkingDir, User, Entrypoint, ...) from its
    registry, anonymously: E2B's template build keeps only its ENV.

    ``repository`` is ``domain/remote`` (sandbox_images.pull_reference),
    ``tag`` a tag or a sha256 digest; an index resolves to linux/amd64, as
    E2B builds. The config blob is checked against its digest.
    """
    import httpx

    domain, _, remote = repository.partition("/")
    base = f"https://{REGISTRY_HOSTS.get(domain, domain)}/v2/{remote}"
    token = ""
    with httpx.Client(proxy=proxy, timeout=timeout, follow_redirects=True) as http:

        def get(url: str, accept: str | None = None) -> Any:
            nonlocal token
            for _ in range(2):
                headers = {"Accept": accept} if accept else {}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                response = http.get(url, headers=headers)
                challenge = response.headers.get("www-authenticate")
                if response.status_code == 401 and not token and challenge:
                    token = _registry_token(http, challenge)
                    continue
                response.raise_for_status()
                return response
            raise InfrastructureError("the registry refused the anonymous token")

        manifest = get(f"{base}/manifests/{tag}", MANIFEST_TYPES).json()
        if "manifests" in manifest:
            for entry in manifest["manifests"]:
                platform = entry.get("platform") or {}
                if (platform.get("os"), platform.get("architecture")) == (
                    "linux",
                    "amd64",
                ):
                    break
            else:
                raise InfrastructureError("the image has no linux/amd64 manifest")
            manifest = get(f"{base}/manifests/{entry['digest']}", MANIFEST_TYPES)
            manifest = manifest.json()
        digest = manifest["config"]["digest"]
        blob = get(f"{base}/blobs/{digest}").content
    if "sha256:" + hashlib.sha256(blob).hexdigest() != digest:
        raise InfrastructureError("the image config does not match its digest")
    config = json.loads(blob).get("config") or {}
    return {key: config[key] for key in IMAGE_CONFIG_KEYS if config.get(key)}


class SdkE2BClient:
    """``E2BClient`` over the pinned SDK (``e2b==2.52.0``).

    Every call passes the key and domain explicitly; no ``E2B_*`` variable
    is read or set. Processes start through envd's Process RPC directly,
    not ``commands.run`` (which always wraps a ``bash -l -c`` string): argv
    stays argv and no login profile runs. That RPC is internal to the SDK,
    hence the exact pin.
    """

    def __init__(
        self,
        api_key: str,
        *,
        domain: str | None = None,
        proxy: str | None = None,
        request_timeout: float = 60,
    ) -> None:
        import e2b

        self._e2b = e2b
        self._key = api_key
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

    def _clean(self, error: BaseException) -> InfrastructureError:
        text = str(error).replace(self._key, "***")[:512]
        return InfrastructureError(f"e2b {type(error).__name__}: {text}")

    def _call(self, action: Callable[[], Any]) -> Any:
        try:
            return action()
        except (SandboxError, FileNotFoundError, InfrastructureError):
            raise
        except Exception as error:
            raise self._clean(error) from None

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
        return bool(self._call(lambda: self._e2b.Template.exists(name, **self._opts)))

    def image_config(self, repository, tag):
        return self._call(
            lambda: registry_image_config(
                repository, tag, proxy=self._opts.get("proxy")
            )
        )

    def build_template(self, name, image, *, cpu_count, memory_mb, log):
        template_class = self._e2b.Template
        template = (
            template_class()
            .from_image(image)
            .set_user("root")
            .set_workdir("/")
            .run_cmd(CAPTURE_ENV, user="root")
        )
        self._call(
            lambda: template_class.build(
                template,
                name,
                cpu_count=cpu_count,
                memory_mb=memory_mb,
                on_build_logs=lambda entry: log(
                    f"{getattr(entry, 'message', '')}\n".replace(self._key, "***")
                ),
                **self._opts,
            )
        )

    def create(self, template, *, timeout, metadata, allow_internet):
        box = self._call(
            lambda: self._e2b.Sandbox.create(
                template,
                timeout=timeout,
                metadata=dict(metadata),
                allow_internet_access=allow_internet,
                # Inbound only with the traffic token, which nobody gets.
                network={"allow_public_traffic": False},
                lifecycle={"on_timeout": "kill"},
                **self._opts,
            )
        )
        with self._lock:
            self._boxes[box.sandbox_id] = box
        return box.sandbox_id

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


def run_in(
    client: E2BClient,
    sandbox_id: str,
    argv: Sequence[str],
    *,
    user: str = "root",
    limit: int = MIB,
) -> tuple[int | None, bytes, bytes]:
    """One broker command to its end: exit code and bounded output."""
    process = client.start(
        sandbox_id, list(argv), envs={"PATH": ROOT_PATH}, cwd="/", user=user
    )
    out, err = bytearray(), bytearray()
    try:
        for stream, data in process:
            buffer = out if stream == 1 else err
            buffer += data[: max(0, limit - len(buffer))]
    finally:
        process.close()
    return process.exit_code, bytes(out), bytes(err)


def reattach(
    client: E2BClient, sandbox_id: str, pid: int, stopped: Callable[[], bool]
) -> E2BProcess | None:
    """Follow a process again after its stream broke. A pause cuts every
    envd stream while the process lives on (and envd keeps an ended one's
    exit for a reconnect): wait until the sandbox runs, then reconnect.
    None when the sandbox is gone, ``stopped()`` or reconnects keep failing.
    """
    failures = 0
    while not stopped():
        try:
            state = client.state(sandbox_id)
        except Exception:
            return None
        if state is None:
            return None
        if state == "running":
            try:
                return client.connect(sandbox_id, pid)
            except Exception as error:
                # Still pausing, or envd not back yet: ask the state again.
                failures += 1
                if failures >= REATTACH_TRIES:
                    LOGGER.info("e2b reconnect to %s failed: %s", pid, error)
                    return None
        time.sleep(REATTACH_SEC)
    return None


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


def template_name(prefix: str, image: str, cpu_count: int, memory_mb: int) -> str:
    """Deterministic: the image's digest (or its reference when it names
    none), the shape and the recipe."""
    _, at, digest = image.rpartition("@")
    source = digest if at and digest.startswith("sha256:") else image
    key = f"{source}|{cpu_count}|{memory_mb}|{RECIPE}"
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:20]}"


def template_attrs(
    name: str,
    image: str,
    cpu_count: int,
    memory_mb: int,
    config: Mapping[str, Any] | None = None,
) -> dict:
    """The image attributes env_create sees for a template (image_view):
    ``config`` is the image's own (its registry's), which plan applies as
    the Engine does."""
    return {
        "Id": "sha256:" + hashlib.sha256(name.encode()).hexdigest(),
        "Os": "linux",
        "Architecture": "amd64",
        "Size": 0,
        "RepoDigests": [],
        "Config": {"Env": [], "WorkingDir": "", "User": "", **(config or {})},
        "E2BTemplate": name,
        "E2BImage": image,
        "E2BShape": [cpu_count, memory_mb],
    }


@dataclass(eq=False)
class _Build:
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class E2BTemplates:
    """Make sure a template exists, building it at most once per name at a
    time in this broker. Names already seen are not asked again."""

    def __init__(self, client: E2BClient, prefix: str) -> None:
        self._client = client
        self.prefix = prefix
        self._lock = threading.Lock()
        self._building: dict[str, _Build] = {}
        self._ready: set[str] = set()

    def ensure(
        self,
        name: str,
        image: str,
        cpu_count: int,
        memory_mb: int,
        log: Callable[[str], None] = lambda line: None,
    ) -> bool:
        """True when this call built the template."""
        with self._lock:
            if name in self._ready:
                return False
            build = self._building.get(name)
            owner = build is None
            if owner:
                build = self._building[name] = _Build()
        assert build is not None
        if not owner:
            log(f"waiting for the build of template {name}\n")
            build.done.wait()
            if build.error is not None:
                raise JobFailed("image", f"template build failed: {build.error}")
            return False
        built = False
        try:
            if not self._client.template_exists(name):
                log(f"building template {name} from {image}\n")
                self._client.build_template(
                    name, image, cpu_count=cpu_count, memory_mb=memory_mb, log=log
                )
                built = True
            with self._lock:
                self._ready.add(name)
            return built
        except Exception as error:
            build.error = error
            LOGGER.warning("e2b template %s build failed: %s", name, error)
            raise JobFailed("image", f"template build failed: {error}"[:1024]) from None
        finally:
            with self._lock:
                del self._building[name]
            build.done.set()


class E2BImages:
    """``image_pull`` on E2B: make sure the image's template exists (it may
    build), then bind a handle to it. Nothing is downloaded to the host."""

    def __init__(self, templates: E2BTemplates, client: E2BClient) -> None:
        self._templates = templates
        self._client = client
        self._lock = threading.Lock()
        # (repository, tag) -> the image config, once per broker.
        self._configs: dict[tuple[str, str], dict[str, Any]] = {}

    def config(self, repository: str, tag: str) -> dict[str, Any]:
        """The image's WORKDIR, USER, ENTRYPOINT, ... from its registry; a
        pull fails without them rather than run the image differently."""
        with self._lock:
            found = self._configs.get((repository, tag))
        if found is not None:
            return found
        try:
            found = dict(self._client.image_config(repository, tag))
        except Exception as error:
            LOGGER.warning("image config of %s:%s: %s", repository, tag, error)
            raise JobFailed(
                "image", f"the image config is unavailable from its registry: {error}"
            ) from None
        with self._lock:
            self._configs[(repository, tag)] = found
        return found

    def run_pull(self, runner: Any, job: Any) -> None:
        repository, tag, _policy, lease = job.detail
        state, error, image = "failed", None, None
        reference = reference_text(repository, tag)
        try:
            runner.running(job)
            grant = job.session.env_grant
            cpu_count, memory_mb = template_shape(
                grant.cpus_per_container, grant.memory_mb_per_container
            )
            name = template_name(
                self._templates.prefix, reference, cpu_count, memory_mb
            )
            runner.log(job, f"e2b template {name} for {reference}\n")
            config = self.config(repository, tag)
            self._templates.ensure(
                name,
                reference,
                cpu_count,
                memory_mb,
                log=lambda line: runner.log(job, line),
            )
            if job.cancel.is_set():
                raise JobFailed("canceled", "the pull was canceled")
            attrs = template_attrs(name, reference, cpu_count, memory_mb, config)
            with runner._lock:
                if job.session.revoked or job.cancel.is_set():
                    raise JobFailed("canceled", "the pull was canceled")
                lease = SandboxImageLease(
                    owner=lease.owner,
                    handle=job.handle,
                    kind="pulled",
                    image_id=attrs["Id"],
                    state="present",
                )
                runner._envs._commit_image(lease)
                image = Image(
                    job.handle,
                    job.session,
                    "pulled",
                    attrs["Id"],
                    reference,
                    0,
                    attrs,
                    lease,
                )
                runner.bind(image)
                state = "succeeded"
        except Exception as failure:
            state, error = runner.outcome(job, failure)
        finally:
            if state != "succeeded":
                try:
                    runner._envs._commit_image(
                        lease.model_copy(update={"state": "removed"})
                    )
                except InfrastructureError:
                    pass  # already failed closed
            runner.finish(
                job,
                state,
                error,
                None
                if image is None
                else {"image": image_view(image.attrs, image.handle)},
            )


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


# -- plan ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class E2BServicePlan:
    idx: int
    name: str
    container_name: str
    image: str
    image_id: str
    template: str
    source: str
    shape: tuple[int, int]
    argv: tuple[str, ...]
    env: Mapping[str, str]
    user: str | None
    working_dir: str | None
    allow_internet: bool
    implicit_volumes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class E2BEnvPlan:
    """The broker's view of a plan (as sandbox_env_docker.EnvPlan's)."""

    owner: SandboxOwner
    env_id: str
    spec_sha256: str
    network_mode: str
    services: tuple[E2BServicePlan, ...]
    cpus_milli: int
    memory_mb: int
    disk_mb: int
    network: None = None
    volumes: tuple = ()
    swap_mb: int = 0

    def service(self, name: str) -> E2BServicePlan:
        for service in self.services:
            if service.name == name:
                return service
        raise SandboxError("invalid", "service", f"unknown service {name}")

    def lease(self, *, created_at: float, expires_at: float) -> SandboxEnvLease:
        return SandboxEnvLease(
            owner=self.owner,
            env_id=self.env_id,
            spec_sha256=self.spec_sha256,
            backend="e2b",
            state="planned",
            created_at=created_at,
            expires_at=expires_at,
            network_mode=self.network_mode,
            cpus_milli=self.cpus_milli,
            memory_mb=self.memory_mb,
            disk_mb=self.disk_mb,
            services=tuple(
                SandboxEnvServiceLease(
                    idx=service.idx,
                    name=service.name,
                    planned_name=service.container_name,
                    image=service.image,
                    image_id=service.image_id,
                )
                for service in self.services
            ),
            pending_mutation=True,
        )


def refuse_unsupported(spec: EnvSpec) -> None:
    """What one E2B sandbox cannot express is refused, never ignored."""
    if spec.network == "allowlist":
        raise unsupported("spec.network", "network allowlist is")
    if len(spec.services) != 1:
        raise unsupported("spec.services", "more than one service (compose) is")
    if spec.volumes:
        raise unsupported("spec.volumes", "volumes are")
    for name, service in spec.services.items():
        where = f"spec.services.{name}"
        for field_name, present in (
            ("mounts", service.mounts),
            ("tmpfs", service.tmpfs),
            ("read_only", service.read_only),
            ("cap_drop", service.cap_drop),
            ("group_add", service.group_add),
            ("hostname", service.hostname),
            ("extra_hosts", service.extra_hosts),
        ):
            if present:
                raise unsupported(f"{where}.{field_name}", f"{field_name} is")
        if service.healthcheck not in (None, "none"):
            raise unsupported(f"{where}.healthcheck", "a healthcheck is")


# -- backend ------------------------------------------------------------------------


@dataclass(eq=False)
class _Service:
    process: Any
    exit_code: int | None = None
    # Set when the exit is known, or the stream is lost.
    ended: threading.Event = field(default_factory=threading.Event)
    # The stream could not be followed again: the process may still run.
    lost: bool = False
    # The sandbox is being killed: stop following.
    closed: bool = False
    tail: bytearray = field(default_factory=bytearray)


@dataclass(eq=False)
class _Box:
    """One live sandbox as execs see it."""

    sandbox_id: str
    # The container's environment: image ENV, then the spec's env.
    env: dict[str, str]
    cwd: str
    user: str | None
    # The service: created, running, exited.
    state: str = "created"
    service: _Service | None = None
    users: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class E2BTarget:
    """One owned sandbox for copies and path-stat; ``require`` refuses a
    paused or vanished one."""

    sandbox_id: str
    require: Callable[[], None]


Commit = Callable[[SandboxEnvLease], SandboxEnvLease]


class SandboxEnvE2BBackend:
    """``SandboxEnvDockerBackend``'s surface over E2B sandboxes.

    env_create creates the sandbox (so the plugin can seed it before start);
    env_start starts the service command, if the spec has one. Pause and
    resume are E2B's, kill is removal. Every step commits through the
    broker's journal like the Docker backend's.
    """

    def __init__(
        self,
        client: E2BClient,
        templates: E2BTemplates,
        *,
        wall: Callable[[], float] = time.time,
        poll: float = POLL_SEC,
        stop_proof_sec: float = STOP_PROOF_SEC,
        max_timeout_sec: int = MAX_TIMEOUT_SEC,
    ) -> None:
        self._client = client
        self._max_timeout = max_timeout_sec
        self._templates = templates
        self._wall = wall
        self._poll = poll
        self._stop_proof = stop_proof_sec
        self._lock = threading.Lock()
        self._boxes: dict[str, _Box] = {}
        # Set by e2b_env_runtime: stop_service ends the service's execs too.
        self.execs: E2BExecApi | None = None

    # -- planning ------------------------------------------------------------

    def plan(
        self,
        owner: SandboxOwner,
        env_id: str,
        spec: EnvSpec,
        images: Mapping[str, Mapping[str, Any]],
        **_options: Any,
    ) -> E2BEnvPlan:
        refuse_unsupported(spec)
        ((name, service),) = spec.services.items()
        attrs = images.get(service.image)
        if attrs is None:
            raise SandboxError(
                "permission", f"spec.services.{name}.image", "unknown image handle"
            )
        template = attrs.get("E2BTemplate")
        if not isinstance(template, str):
            raise SandboxError(
                "invalid", f"spec.services.{name}.image", "not an e2b template image"
            )
        cpu_count, memory_mb = attrs["E2BShape"]
        # The Engine's merge (sandbox_env_docker): the spec's entrypoint
        # clears the image's Cmd; WORKDIR and USER are the image's defaults.
        config = attrs.get("Config") or {}
        if service.entrypoint is not None:
            entrypoint = list(service.entrypoint)
        else:
            entrypoint = list(config.get("Entrypoint") or ())
        if service.command:
            command = list(service.command)
        elif service.entrypoint:
            command = []
        else:
            command = list(config.get("Cmd") or ())
        return E2BEnvPlan(
            owner=owner,
            env_id=env_id,
            spec_sha256=env_spec_digest(spec),
            network_mode=spec.network,
            services=(
                E2BServicePlan(
                    idx=0,
                    name=name,
                    container_name=env_container_name(env_id, 0),
                    image=service.image,
                    image_id=attrs["Id"],
                    template=template,
                    source=attrs["E2BImage"],
                    shape=(cpu_count, memory_mb),
                    argv=(*entrypoint, *command),
                    env=dict(service.env),
                    user=service.user or config.get("User") or None,
                    working_dir=service.working_dir
                    or _clean_dir(config.get("WorkingDir"))
                    or None,
                    allow_internet=spec.network == "public"
                    and service.network == "env",
                ),
            ),
            cpus_milli=round(service.cpus * 1000),
            memory_mb=service.memory_mb,
            disk_mb=spec.disk_mb,
        )

    def allowlist_notes(self, plan: E2BEnvPlan) -> tuple[str, ...]:
        return ()

    def refresh_network(self, plan: E2BEnvPlan) -> bool:
        return False

    # -- boxes ---------------------------------------------------------------

    @staticmethod
    def _record(lease: SandboxEnvLease, service: str) -> SandboxEnvServiceLease:
        for record in lease.services:
            if record.name == service:
                return record
        raise SandboxError("invalid", "service", f"unknown service {service}")

    def box(self, sandbox_id: str) -> _Box:
        with self._lock:
            box = self._boxes.get(sandbox_id)
        if box is None:
            raise SandboxError("invalid", "service", "service is absent")
        return box

    def _timeout(self, lease: SandboxEnvLease) -> int:
        """The env's remaining lifetime plus slack, within the team's
        maximum sandbox length (E2B refuses a longer timeout)."""
        remaining = max(0.0, lease.expires_at - self._wall())
        wanted = math.ceil(remaining) + TIMEOUT_SLACK_SEC
        if wanted > self._max_timeout:
            LOGGER.info(
                "e2b env %s outlives the %d s sandbox limit: E2B ends it first",
                lease.env_id,
                self._max_timeout,
            )
        return min(self._max_timeout, wanted)

    def _image_env(self, sandbox_id: str) -> dict[str, str]:
        try:
            return parse_image_env(self._client.read_file(sandbox_id, IMAGE_ENV_FILE))
        except FileNotFoundError:
            LOGGER.warning("e2b sandbox %s has no recorded image ENV", sandbox_id)
            return {}

    def user_name(self, box: _Box, user: str | None) -> str:
        """envd takes user names: 0 is root, other uids come from passwd."""
        name = (user or box.user or "root").split(":", 1)[0]
        if not name.isdigit():
            return name
        if int(name) == 0:
            return "root"
        if box.users is None:
            try:
                raw = self._client.read_file(box.sandbox_id, "/etc/passwd")
            except FileNotFoundError:
                raw = b""
            users: dict[str, str] = {}
            for line in raw.decode("utf-8", "replace").splitlines():
                parts = line.split(":")
                if len(parts) > 2 and parts[2].isdigit():
                    users.setdefault(str(int(parts[2])), parts[0])
            box.users = users
        found = box.users.get(str(int(name)))
        if found is None:
            raise SandboxError(
                "invalid", "user", "uid has no user name in the image (e2b backend)"
            )
        return found

    # -- create and start -----------------------------------------------------

    def _gone(self, lease: SandboxEnvLease, commit: Commit) -> SandboxEnvLease:
        services = tuple(
            record.model_copy(update={"state": "removed"}) for record in lease.services
        )
        return commit(
            _revise(
                lease,
                state="removed",
                reason=lease.reason or "start_failed",
                pending_mutation=False,
                services=services,
            )
        )

    def create(
        self, plan: E2BEnvPlan, lease: SandboxEnvLease, commit: Commit
    ) -> SandboxEnvLease:
        """The template, then the sandbox; the service command waits for start."""
        if lease.state != "planned" or not lease.pending_mutation:
            raise InfrastructureError("sandbox env create needs its pending plan")
        service = plan.services[0]
        try:
            self._templates.ensure(service.template, service.source, *service.shape)
        except JobFailed as error:
            self._gone(lease, commit)
            raise SetupError(f"e2b template unavailable: {error.message}") from None
        try:
            sandbox_id = self._client.create(
                service.template,
                timeout=self._timeout(lease),
                metadata=sandbox_metadata(plan.owner, plan.env_id, service.name),
                allow_internet=service.allow_internet,
            )
        except Exception as error:
            # A create without an answer may still land: find it by metadata.
            self._kill_env(lease)
            self._gone(lease, commit)
            raise SetupError(f"e2b refused the sandbox: {error}") from None
        lease = commit(
            _service(lease, service.idx, sandbox_id=sandbox_id, state="created")
        )
        if service.working_dir is not None:
            # As the Engine does for a container's WORKDIR (envd refuses a
            # missing cwd); a failure surfaces at the first exec.
            run_in(self._client, sandbox_id, ["mkdir", "-p", service.working_dir])
        box = _Box(
            sandbox_id,
            {**self._image_env(sandbox_id), **service.env},
            service.working_dir or "/",
            service.user,
        )
        with self._lock:
            self._boxes[sandbox_id] = box
        return commit(_revise(lease, state="created", pending_mutation=False))

    def start(
        self,
        plan: E2BEnvPlan,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        wait_timeout_sec: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> EnvStartResult:
        if lease.state not in ("created", "starting"):
            raise SandboxError("busy", "env_id", f"env is {lease.state}")
        if lease.state == "created":
            lease = commit(_revise(lease, state="starting"))
        service = plan.services[0]
        record = lease.services[service.idx]
        if cancelled is not None and cancelled():
            lease = commit(_revise(lease, state="failed", reason="canceled"))
            return EnvStartResult(lease, "failed", "canceled", service.name)
        box = self.box(record.sandbox_id or "")
        if service.argv:
            try:
                process = self._client.start(
                    box.sandbox_id,
                    ["setsid", *service.argv],
                    envs=box.env,
                    cwd=box.cwd,
                    user=self.user_name(box, None),
                )
            except Exception as error:
                LOGGER.warning(
                    "e2b env %s service start failed: %s", lease.env_id, error
                )
                lease = commit(_revise(lease, state="failed", reason="start_failed"))
                return EnvStartResult(
                    lease,
                    "failed",
                    "start_failed",
                    service.name,
                    "the service command could not start",
                )
            box.service = _Service(process)
            threading.Thread(
                target=self._drain,
                args=(box.sandbox_id, box.service),
                name="rsi-sandbox-e2b-service",
                daemon=True,
            ).start()
        box.state = "running"
        lease = commit(_service(lease, service.idx, state="running"))
        return EnvStartResult(commit(_revise(lease, state="ready")), "ready")

    def _drain(self, sandbox_id: str, service: _Service) -> None:
        """Keep the output tail and the exit code; a stream cut by a pause
        is followed again once the sandbox runs (as an exec's is)."""
        process = service.process
        try:
            while True:
                failure: Exception | None = None
                try:
                    for _stream, data in process:
                        service.tail += data
                        del service.tail[:-TAIL_BYTES]
                except Exception as error:
                    failure = error
                if failure is None and process.exit_code is not None:
                    service.exit_code = process.exit_code
                    return
                process.close()
                follow = reattach(
                    self._client, sandbox_id, process.pid, lambda: service.closed
                )
                if follow is None:
                    if not service.closed:
                        LOGGER.info("e2b service stream lost: %s", failure)
                        service.lost = True
                    return
                process = service.process = follow
        finally:
            service.ended.set()

    # -- runtime ---------------------------------------------------------------

    def _state(self, sandbox_id: str) -> str | None:
        return self._client.state(sandbox_id)

    def status(self, lease: SandboxEnvLease) -> dict[str, ServiceStatus]:
        statuses = {}
        for record in lease.services:
            if record.state in ("planned", "removed") or record.sandbox_id is None:
                statuses[record.name] = ServiceStatus("removed")
                continue
            state = self._state(record.sandbox_id)
            with self._lock:
                box = self._boxes.get(record.sandbox_id)
            service = None if box is None else box.service
            if state is None:
                statuses[record.name] = ServiceStatus("removed")
            elif state == "paused":
                statuses[record.name] = ServiceStatus("paused")
            elif service is not None and service.exit_code is not None:
                statuses[record.name] = ServiceStatus("exited", None, service.exit_code)
            elif box is not None and box.state == "exited":
                statuses[record.name] = ServiceStatus("exited")
            elif record.state == "created":
                statuses[record.name] = ServiceStatus("created")
            else:
                statuses[record.name] = ServiceStatus("running")
        return statuses

    def diagnostics(self, lease: SandboxEnvLease, service: str) -> dict[str, str]:
        record = self._record(lease, service)
        with self._lock:
            box = self._boxes.get(record.sandbox_id or "")
        tail = b"" if box is None or box.service is None else bytes(box.service.tail)
        return {"health_tail": "", "log_tail": tail.decode("utf-8", "replace")}

    def pause(self, lease: SandboxEnvLease, commit: Commit) -> SandboxEnvLease:
        """freeze_work: E2B pauses the whole VM (memory kept); proven by state."""
        if lease.state == "starting":
            raise SandboxError("busy", "env_id", "a starting env cannot pause")
        paused = []
        try:
            for record in lease.services:
                if record.state != "running" or record.sandbox_id is None:
                    if record.state == "paused":
                        paused.append(record.idx)
                    continue
                failure: Exception | None = None
                try:
                    self._client.pause(record.sandbox_id)
                except Exception as error:
                    failure = error
                if self._state(record.sandbox_id) != "paused":
                    raise InfrastructureError(
                        f"sandbox env service {record.name} pause could not be proven"
                    ) from failure
                paused.append(record.idx)
        finally:
            if paused:
                for index in paused:
                    lease = _service(lease, index, state="paused")
                if lease.state == "ready":
                    lease = _revise(lease, state="paused")
                lease = commit(lease)
        return lease

    def resume(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        admission: Callable[[], AbstractContextManager[Any]] | None = None,
    ) -> SandboxEnvLease:
        """Resume, and set E2B's timeout again: a pause restarts its timer."""
        resumed = []
        try:
            for record in lease.services:
                if record.state != "paused" or record.sandbox_id is None:
                    continue
                failure: Exception | None = None
                with admission() if admission is not None else nullcontext():
                    try:
                        self._client.resume(
                            record.sandbox_id, timeout=self._timeout(lease)
                        )
                    except Exception as error:
                        failure = error
                if self._state(record.sandbox_id) != "running":
                    raise InfrastructureError(
                        f"sandbox env service {record.name} resume could not be proven"
                    ) from failure
                resumed.append(record.idx)
        finally:
            if resumed:
                for index in resumed:
                    lease = _service(lease, index, state="running")
                if lease.state == "paused" and not any(
                    record.state == "paused" for record in lease.services
                ):
                    lease = _revise(lease, state="ready")
                lease = commit(lease)
        return lease

    def stop_service(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        service: str,
        *,
        timeout_sec: float,
    ) -> tuple[SandboxEnvLease, int | None]:
        """TERM the service's group (and every exec's), KILL after
        ``timeout_sec``; the sandbox stays, refusing execs like a stopped
        container."""
        record = self._record(lease, service)
        if record.state in ("planned", "removed") or record.sandbox_id is None:
            raise SandboxError("invalid", "service", f"service {service} is absent")
        state = self._state(record.sandbox_id)
        if state == "paused":
            raise SandboxError("busy", "service", "a paused service cannot stop")
        box = self.box(record.sandbox_id)
        code = None
        if state == "running":
            running = box.service
            if running is not None and running.lost:
                # Its end can no longer be seen: no grace period to wait out.
                self._signal(box.sandbox_id, running.process.pid, "KILL")
            elif running is not None and not running.ended.is_set():
                for signum, wait in (("TERM", timeout_sec), ("KILL", self._stop_proof)):
                    self._signal(box.sandbox_id, running.process.pid, signum)
                    if running.ended.wait(wait):
                        break
            if self.execs is not None:
                self.execs.kill_all(box.sandbox_id)
            if running is not None:
                code = running.exit_code
        box.state = "exited"
        return commit(_service(lease, record.idx, state="exited")), code

    def _signal(self, sandbox_id: str, pid: int, name: str) -> None:
        try:
            run_in(self._client, sandbox_id, ["kill", "-s", name, "--", f"-{pid}"])
        except Exception as error:
            LOGGER.warning("e2b signal %s to %s failed: %s", name, pid, error)

    # -- teardown --------------------------------------------------------------

    def _kill(self, sandbox_id: str, *, deadline: float | None = None) -> None:
        """Kill (which removes) one sandbox, proven by its absence."""
        end = time.monotonic() + self._stop_proof if deadline is None else deadline
        failure: Exception | None = None
        try:
            self._client.kill(sandbox_id)
        except Exception as error:
            failure = error
        while True:
            try:
                if self._state(sandbox_id) is None:
                    break
            except Exception as error:
                failure = error
            if time.monotonic() >= end:
                raise InfrastructureError(
                    f"recovery_required: e2b sandbox {sandbox_id} removal is "
                    f"unproven: {failure}"
                ) from failure
            time.sleep(self._poll)
        with self._lock:
            box = self._boxes.pop(sandbox_id, None)
        if box is not None and box.service is not None:
            box.service.closed = True
            box.service.process.close()

    def _kill_env(self, lease: SandboxEnvLease) -> None:
        """Sandboxes of the env the journal may lack (a create in flight)."""
        selector = {f"{META_PREFIX}env_id": lease.env_id}
        for sandbox_id, _metadata in self._client.list(selector):
            self._kill(sandbox_id)

    def terminate(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        deadline: float | None = None,
    ) -> SandboxEnvLease:
        """Nothing of the env executes afterwards: on E2B that is a kill.

        ``deadline`` is on the broker's monotonic clock (close_judge)."""
        end = (
            None
            if deadline is None
            else time.monotonic() + max(0.0, deadline - time.monotonic())
        )
        for record in lease.services:
            if record.state == "removed" or record.sandbox_id is None:
                continue
            self._kill(record.sandbox_id, deadline=end)
            if record.state not in ("planned", "exited"):
                lease = commit(_service(lease, record.idx, state="exited"))
        return lease

    def fail(
        self, lease: SandboxEnvLease, commit: Commit, reason: EnvReason
    ) -> SandboxEnvLease:
        if lease.state not in ("created", "starting", "ready", "failed", "paused"):
            raise SandboxError("busy", "env_id", f"env is {lease.state}")
        lease = self.terminate(lease, commit)
        if lease.state == "failed" and lease.reason is not None:
            return lease
        return commit(_revise(lease, state="failed", reason=reason))

    def destroy(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        reason: EnvReason | None = None,
    ) -> SandboxEnvLease:
        if lease.state == "removed":
            return lease
        try:
            lease = commit(
                _revise(lease, state="stopping", reason=reason or lease.reason)
            )
            for record in lease.services:
                if record.state == "removed":
                    continue
                if record.sandbox_id is not None:
                    self._kill(record.sandbox_id)
                lease = commit(_service(lease, record.idx, state="removed"))
            if lease.pending_mutation:
                self._kill_env(lease)
            return commit(_revise(lease, state="removed", pending_mutation=False))
        except InfrastructureError:
            raise
        except SandboxError:
            raise
        except Exception as error:
            raise InfrastructureError(
                f"recovery_required: e2b env {lease.env_id} removal failed: {error}"
            ) from error

    # -- targets ---------------------------------------------------------------

    def _live(self, lease: SandboxEnvLease, service: str) -> SandboxEnvServiceLease:
        record = self._record(lease, service)
        if record.state in ("planned", "removed") or record.sandbox_id is None:
            raise SandboxError("invalid", "service", f"service {service} is absent")
        return record

    def archive_target(self, lease: SandboxEnvLease, service: str) -> E2BTarget:
        sandbox_id = self._live(lease, service).sandbox_id
        assert sandbox_id is not None

        def require() -> None:
            state = self._state(sandbox_id)
            if state is None:
                raise SandboxError("invalid", "service", "service is absent")
            if state == "paused":
                raise SandboxError(
                    "busy", "service", "a paused service cannot transfer"
                )

        return E2BTarget(sandbox_id, require)

    def exec_target(self, lease: SandboxEnvLease, service: str) -> ExecTarget:
        sandbox_id = self._live(lease, service).sandbox_id
        assert sandbox_id is not None

        def inspect() -> dict[str, Any] | None:
            state = self._state(sandbox_id)
            with self._lock:
                box = self._boxes.get(sandbox_id)
            if state is None or box is None:
                return None
            return {
                "Id": sandbox_id,
                "State": {
                    "Running": state == "paused" or box.state == "running",
                    "Paused": state == "paused",
                    "Pid": 1,
                },
            }

        return ExecTarget(container_id=sandbox_id, inspect=inspect)


# -- execs ------------------------------------------------------------------------


@dataclass(eq=False)
class _E2BExec:
    exec_id: str
    sandbox_id: str
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: str
    user: str
    pid: int | None = None
    exit_code: int | None = None
    ended: bool = False
    lost: bool = False
    closed: bool = False
    writer: socket.socket | None = None
    # Monotonic time the stream ended: the pump may still ask for the exit.
    ended_at: float | None = None


class _Attach:
    """The pump's hijacked-stream handle: one end of a socket pair."""

    def __init__(self, sock: socket.socket, on_close: Callable[[], None]) -> None:
        self._sock = sock
        self._on_close = on_close

    def close(self) -> None:
        self._on_close()


class E2BExecApi:
    """The Docker low-level exec calls ExecPump makes, over envd.

    ``exec_start`` starts the process at once and returns a socket that
    carries its output in the Engine's 8-byte frames; ``exec_inspect``
    answers from what the stream reported. A stream cut by a pause is
    reattached once the sandbox runs again; one that cannot be is ``lost``
    (NotFound, as for an exec the Engine forgot).
    """

    def __init__(self, client: E2BClient, backend: SandboxEnvE2BBackend) -> None:
        self._client = client
        self._backend = backend
        self._lock = threading.Lock()
        self._execs: dict[str, _E2BExec] = {}

    def exec_create(
        self,
        container: str,
        cmd: Sequence[str],
        *,
        environment: Mapping[str, str] | None = None,
        workdir: str | None = None,
        user: str = "",
        **_streams: Any,
    ) -> dict[str, str]:
        box = self._backend.box(container)
        record = _E2BExec(
            exec_id=secrets.token_hex(32),
            sandbox_id=container,
            argv=tuple(cmd),
            env={**box.env, **(environment or {})},
            cwd=workdir or box.cwd,
            user=self._backend.user_name(box, user or None),
        )
        with self._lock:
            # The pump asks about an ended exec for a few seconds at most.
            stale = time.monotonic() - EXEC_RECORD_SEC
            for key in [
                key
                for key, item in self._execs.items()
                if item.closed and item.ended_at is not None and item.ended_at < stale
            ]:
                del self._execs[key]
            self._execs[record.exec_id] = record
        return {"Id": record.exec_id}

    def exec_start(self, exec_id: str, **_options: Any) -> _Attach:
        record = self._execs[exec_id]
        try:
            process = self._client.start(
                record.sandbox_id,
                ["setsid", *record.argv],
                envs=record.env,
                cwd=record.cwd,
                user=record.user,
            )
        except BaseException:
            with self._lock:
                self._execs.pop(exec_id, None)
            raise
        record.pid = process.pid
        ours, theirs = socket.socketpair()
        record.writer = theirs
        threading.Thread(
            target=self._forward,
            args=(record, process),
            name="rsi-sandbox-e2b-exec",
            daemon=True,
        ).start()
        return _Attach(ours, lambda: self._close(record, ours))

    def exec_inspect(self, exec_id: str) -> dict[str, Any]:
        record = self._execs.get(exec_id)
        if record is None or record.lost:
            raise NotFound("e2b exec is gone")
        return {
            "Pid": record.pid or 0,
            "Running": not record.ended,
            "ExitCode": record.exit_code,
        }

    def running(self, sandbox_id: str, pid: int) -> bool:
        with self._lock:
            return any(
                record.sandbox_id == sandbox_id
                and record.pid == pid
                and not record.ended
                for record in self._execs.values()
            )

    def kill_all(self, sandbox_id: str) -> None:
        """SIGKILL every running exec group of a sandbox (stop_service)."""
        with self._lock:
            pids = [
                record.pid
                for record in self._execs.values()
                if record.sandbox_id == sandbox_id and not record.ended and record.pid
            ]
        for pid in pids:
            self._backend._signal(sandbox_id, pid, "KILL")

    def _close(self, record: _E2BExec, sock: socket.socket) -> None:
        record.closed = True
        sock.close()

    def _forward(self, record: _E2BExec, process: E2BProcess) -> None:
        writer = record.writer
        assert writer is not None
        try:
            while True:
                failure: Exception | None = None
                try:
                    for stream, data in process:
                        try:
                            writer.sendall(_FRAME.pack(stream, len(data)) + data)
                        except OSError:
                            return  # the pump closed its end
                except Exception as error:
                    failure = error
                if failure is None and process.exit_code is not None:
                    record.exit_code = process.exit_code
                    return
                process.close()
                reattached = reattach(
                    self._client,
                    record.sandbox_id,
                    record.pid or 0,
                    lambda: record.closed,
                )
                if reattached is None:
                    LOGGER.info("e2b exec %s stream lost: %s", record.exec_id, failure)
                    record.lost = True
                    return
                process = reattached
        finally:
            process.close()
            record.ended_at = time.monotonic()
            record.ended = True
            try:
                writer.close()
            except OSError:
                pass


class E2BProcessTable:
    """ProcessTable's surface for ExecPump: no host /proc or cgroup; a
    leader is alive while its stream is."""

    def __init__(self, execs: E2BExecApi) -> None:
        self._execs = execs

    def scope(self, container_id: str, pid: object) -> Path:
        return Path("/e2b") / container_id

    def oom_kills(self, scope: Path) -> int:
        return 0

    def stat(self, pid: int) -> None:
        return None

    def process(self, container_id: str, scope: Path, pid: int) -> ExecProcess:
        return ExecProcess(container_id, scope, pid, pid, 0, pid, pid)

    def targets(self, process: ExecProcess, *, group: bool) -> tuple[int, ...]:
        alive = self._execs.running(process.container_id, process.host_pid)
        return (process.host_pid,) if alive else ()


class E2BKiller:
    """ExecKiller: ``kill`` as root in the sandbox, to the group (setsid)."""

    def __init__(self, client: E2BClient) -> None:
        self._client = client

    def signal(self, process: ExecProcess, signum: int, *, group: bool) -> bool:
        name = _SIGNAL_NAMES.get(signum)
        if name is None:
            return False
        target = f"-{process.ns_pgid}" if group else str(process.ns_pid)
        code, _, _ = run_in(
            self._client, process.container_id, ["kill", "-s", name, "--", target]
        )
        return code == 0


# -- copies -------------------------------------------------------------------------

# $1 path, $2 follow (1/0). Prints "none", or "<octal> <size> <mtime>",
# the kind and a symlink's target, one per line.
PATH_STAT_SCRIPT = r"""p=$1
[ -e "$p" ] || [ -L "$p" ] || { echo none; exit 0; }
o=
if [ "$2" = 1 ] && [ -e "$p" ]; then o=-L; fi
if [ -z "$o" ] && [ -L "$p" ]; then k=symlink
elif [ -d "$p" ]; then k=dir
elif [ -f "$p" ]; then k=file
else k=other; fi
stat $o -c '%a %s %Y' "$p" || exit 1
echo "$k"
if [ "$k" = symlink ]; then readlink "$p"; fi
"""
# $1 tar, $2 dest: missing ancestors are made (0755 root) first.
COPY_IN_SCRIPT = 'mkdir -p "$2" && tar -xpf "$1" -C "$2"; rc=$?; rm -f "$1"; exit $rc'
# $1 tar, $2 parent, $3 name: exit 3 when the path is absent.
COPY_OUT_SCRIPT = (
    '[ -e "$2/$3" ] || [ -L "$2/$3" ] || exit 3; tar -cf "$1" -C "$2" "./$3"'
)


def _scratch() -> str:
    return posixpath.join(SCRATCH_DIR, f".rsi-{secrets.token_hex(8)}.tar")


def _refuse_special(path: str, field_name: str) -> None:
    if any(below(path, root) for root in SPECIAL_ROOTS):
        raise SandboxError(
            "unsupported", field_name, "kernel filesystems are not copyable"
        )


class E2BTransfer:
    """copy_in/copy_out/path_stat as tar files in the sandbox (root)."""

    def __init__(self, client: E2BClient, stages: StageStore) -> None:
        self._client = client
        self._stages = stages

    def _run(self, target: E2BTarget, script: str, *args: str) -> tuple[int, bytes]:
        code, out, err = run_in(
            self._client, target.sandbox_id, ["sh", "-c", script, "sh", *args]
        )
        if code is None:
            raise SandboxError("unknown-outcome", "path", "sandbox command was cut off")
        if code:
            LOGGER.info("e2b transfer exited %s: %s", code, err[-512:])
        return code, out

    def path_stat(self, target: E2BTarget, path: str, *, follow: bool) -> dict:
        _require_path(path, "path")
        if type(follow) is not bool:
            raise SandboxError("invalid", "follow", "expected a boolean")
        target.require()
        code, out = self._run(target, PATH_STAT_SCRIPT, path, "1" if follow else "0")
        lines = out.decode("utf-8", "surrogateescape").split("\n")
        if code or not lines or lines[0] == "none":
            if code:
                raise SandboxError("invalid", "path", "path stat failed in the sandbox")
            return {
                "exists": False,
                "kind": None,
                "size": None,
                "mode": None,
                "mtime": None,
                "link_target": None,
            }
        try:
            mode, size, mtime = lines[0].split()
            kind = lines[1]
            return {
                "exists": True,
                "kind": kind,
                "size": int(size),
                "mode": int(mode, 8),
                "mtime": datetime.fromtimestamp(int(mtime), UTC).isoformat(),
                "link_target": (lines[2] or None)
                if kind == "symlink" and len(lines) > 2
                else None,
            }
        except (ValueError, IndexError):
            raise SandboxError(
                "infrastructure", "path", "path stat answer is malformed"
            ) from None

    def copy_in(self, target: E2BTarget, dest_dir: str, stage_id: str) -> dict:
        _require_path(dest_dir, "dest_dir")
        _refuse_special(dest_dir, "dest_dir")
        target.require()
        scratch = _scratch()
        with self._stages.consume(stage_id) as (stage, summary):
            self._client.write_file(target.sandbox_id, scratch, stage)
        code, _ = self._run(target, COPY_IN_SCRIPT, scratch, dest_dir)
        if code:
            raise SandboxError("invalid", "dest_dir", "the archive did not extract")
        return {"entries": summary.entries, "bytes": summary.bytes}

    def copy_out(
        self,
        target: E2BTarget,
        path: str,
        *,
        max_bytes: int,
        exclude: Sequence[str] = (),
    ) -> dict:
        _require_path(path, "path")
        if path == "/":
            raise SandboxError("invalid", "path", "copy_out needs a named path")
        if type(max_bytes) is not int or max_bytes < 0:
            raise SandboxError("invalid", "max_bytes", "expected a nonnegative limit")
        patterns = _exclusions(exclude)
        _refuse_special(path, "path")
        target.require()
        parent, name = posixpath.split(path)
        scratch = _scratch()
        try:
            code, _ = self._run(target, COPY_OUT_SCRIPT, scratch, parent, name)
            if code == 3:
                raise SandboxError("invalid", "path", f"{path} does not exist")
            if code:
                raise SandboxError("invalid", "path", "the archive could not be made")
            data = self._client.read_file(target.sandbox_id, scratch)
        finally:
            try:
                run_in(self._client, target.sandbox_id, ["rm", "-f", scratch])
            except Exception as error:
                LOGGER.info("e2b scratch tar not removed: %s", error)
        with self._stages.create_result() as result:
            result.summary = ArchiveTransfer._reemit(
                io.BytesIO(data), result.file, max_bytes, patterns, ()
            )
        summary = result.summary
        return {
            "stage_id": result.stage_id,
            "bytes": summary.bytes,
            "entries": summary.entries,
            "skipped": summary.skipped,
        }


# -- composition and cleanup ----------------------------------------------------------


def e2b_client(settings: EnvE2BHost, api_key: str | None = None) -> SdkE2BClient:
    """The production client; the key (read here unless the caller already
    did) is kept only by it."""
    return SdkE2BClient(
        read_api_key(settings) if api_key is None else api_key,
        domain=settings.domain,
        proxy=settings.proxy,
    )


def e2b_env_runtime(
    client: E2BClient,
    *,
    spool_root: Path,
    host: Any,
    clock: Callable[[], float] = time.monotonic,
) -> EnvRuntime:
    """The E2B composition beside docker_env_runtime: no Docker daemon,
    firewall, disk watchdog or builder; stages and exec output stay in
    the host spool, under its floor."""
    templates = E2BTemplates(client, host.e2b.template_prefix)
    backend = SandboxEnvE2BBackend(
        client, templates, max_timeout_sec=host.e2b.max_sandbox_hours * 3600
    )
    execs = E2BExecApi(client, backend)
    backend.execs = execs
    table = E2BProcessTable(execs)
    return EnvRuntime(
        backend=backend,
        images=E2BImages(templates, client),
        transfer=lambda stages: E2BTransfer(client, stages),
        pump=lambda on_finish: ExecPump(
            execs,
            Path(spool_root),
            killer=E2BKiller(client),
            table=table,
            on_finish=on_finish,
            clock=clock,
            start_confirm_sec=START_CONFIRM_SEC,
        ),
        spool_root=Path(spool_root),
        table=table,
        spool_admit=spool_floor(Path(spool_root), host.disk_floor_mb),
        backend_name="e2b",
    )


def kill_run_sandboxes(client: E2BClient, run_id: str) -> tuple[str, ...]:
    """Recovery: kill every sandbox (running or paused) carrying the run's
    metadata, and prove none is left. Templates stay (a shared cache)."""
    selector = run_metadata(run_id)
    killed = []
    for sandbox_id, metadata in client.list(selector):
        if metadata.get(f"{META_PREFIX}run_id") != run_id:
            continue
        client.kill(sandbox_id)
        killed.append(sandbox_id)
    left = [
        sandbox_id
        for sandbox_id, metadata in client.list(selector)
        if metadata.get(f"{META_PREFIX}run_id") == run_id
    ]
    if left:
        raise InfrastructureError(
            f"e2b sandboxes of run {run_id} remain after kill: {', '.join(left[:8])}"
        )
    return tuple(killed)


__all__ = [
    "CAPTURE_ENV",
    "IMAGE_ENV_FILE",
    "RECIPE",
    "E2BClient",
    "E2BExecApi",
    "E2BImages",
    "E2BTemplates",
    "E2BTransfer",
    "SandboxEnvE2BBackend",
    "SdkE2BClient",
    "e2b_client",
    "e2b_env_runtime",
    "kill_run_sandboxes",
    "parse_image_env",
    "read_api_key",
    "refuse_unsupported",
    "registry_image_config",
    "run_metadata",
    "sandbox_metadata",
    "template_name",
    "template_shape",
]
