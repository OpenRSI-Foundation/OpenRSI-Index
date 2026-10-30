"""Remote env services: the broker's env backends beyond Docker (V1).

``remote_env_runtime`` is an ``EnvRuntime`` beside ``docker_env_runtime``:
the whole env layer (sessions, two-key grants, quotas, journal, request
replay, freeze and close ordering, the wire protocol and the plugin) stays
``SandboxEnvs``'s; only where a service runs changes. One env is one remote
sandbox running one service. What differs per backend is its client
(``RemoteClient``, e.g. ``sandbox_e2b.SdkE2BClient``); ``BACKENDS`` names
each backend's module.

Templates are made on demand: an image reference (its digest when given)
plus the phase's per-container shape (the client's ``shape``) and the
client's recipe name one template deterministically. If the backend already
has a template of that name it is used, otherwise ``image_pull`` builds it
and waits: the backend itself is the cache, and one broker builds each name
once at a time. The pull reads the image config from the image's registry
(``registry_image_config``) and env_create applies its WORKDIR, USER,
ENTRYPOINT and CMD as the Engine does. ``image_build`` (``RemoteBuilds``)
adds a single-stage Dockerfile's steps to that recipe, under a name derived
from what the build sends. A sandbox's timeout stays within the
backend's maximum sandbox length. An allowlist env's entries go to the
backend's firewall (``allow_out``).

Execs reuse ``ExecPump`` unchanged through ``RemoteExecApi``, the slice of
Docker's exec API the pump calls: each command runs under ``setsid`` (its
own process group) without a login shell, its output is reframed as the
Engine's multiplexed stream, and signals reach the group through ``kill``
run as root in the sandbox. Copies are tar files moved with the client's
file calls and packed or unpacked by ``tar`` in the sandbox; copy_out
streams its tar back, so ``max_bytes`` holds as it is read.

What a backend cannot do natively is emulated in the sandbox, as its
client's flags say: without ``native_pid`` a command starts through
``group_argv``, which reports its pid; without ``native_user`` it starts as
root through ``setpriv``; with ``freeze = "signal"`` a freeze stops the
sandbox's processes (``SignalFreeze``).

A client whose key may not list keeps the run's sandbox ledger
(sandbox_remote_ledger, its ``ledger`` attribute, set by ``bind_ledger``
beside the run's lease): its ``list`` merges the listing with the sandboxes
the ledger holds, so recovery and an env's kill find them either way. A
provider's capacity refusal of a create is a ``ProviderCapacityError``:
the env is rolled back and env_create answers ``quota`` (field
``provider_capacity``), which a caller may wait on.

Credentials live only in the client: leases, plans, sandbox metadata,
capabilities and logs carry where they are, never them. Private registry
credentials (``read_registry_auth``) are never recorded either; the client
reads them on each pull, so recovery and cleanup need only the backend's
own credentials.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import io
import json
import logging
import math
import os
import posixpath
import re
import secrets
import shutil
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple, Protocol

from docker.errors import NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations.sandbox_compose import remote_refusal
from rsi_harness.runtime.sandbox_archive import (
    SPECIAL_ROOTS,
    ArchiveTransfer,
    StageStore,
    _exclusions,
    _require_path,
)
from rsi_harness.runtime.sandbox_contracts import (
    ProviderCapacityError,
    SandboxError,
    SandboxOwner,
    below,
    parse_allow_entry,
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
    pull_reference,
    reference_text,
)
from rsi_harness.runtime.sandbox_remote_ledger import (
    SandboxLedger,
    sandbox_ledger_path,
)

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
# [environments.host] backend -> the module of that remote backend, whose
# ``BACKEND`` is its RemoteBackend; no module imports its SDK before a
# client is made.
BACKENDS = {
    "e2b": "rsi_harness.runtime.sandbox_e2b",
    "modal": "rsi_harness.runtime.sandbox_modal",
    "daytona": "rsi_harness.runtime.sandbox_daytona",
}
# The broker's own commands (tar, stat, kill) as root.
ROOT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SCRATCH_DIR = "/tmp"
# The sandbox outlives the env deadline by this much, then the backend
# kills it: the dead-man switch when the broker is gone.
TIMEOUT_SLACK_SEC = 60
# Default for a backend table's max_sandbox_hours.
MAX_TIMEOUT_SEC = 3_600
# The image config keys env_create applies or image_view reports. A remote
# sandbox runs no image healthcheck and stops with TERM: those are left out.
IMAGE_CONFIG_KEYS = ("Env", "WorkingDir", "User", "Entrypoint", "Cmd", "Volumes")
# Read only by image_build's steps: how Docker builds on the image.
BUILD_CONFIG_KEYS = ("Shell", "OnBuild")
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
META_PREFIX = "rsi_"
_FRAME = struct.Struct("!BxxxI")
_SIGNAL_NAMES = {number: name for name, number in SIGNALS.items()}
ENV_ROLE = "sandbox-env"
# group_argv's first stdout line, read before start returns: exec_start's
# caller reads the pid at once (START_CONFIRM_SEC bounds the whole start).
PID_LINE_SEC = 3.0
# A pid line later than that is still waited for this long, to kill the
# command no caller holds.
LATE_PID_SEC = 60.0


def unsupported(field_name: str, what: str, backend: str) -> SandboxError:
    return SandboxError(
        "unsupported", field_name, f"{what} unsupported on the {backend} backend"
    )


# -- backends and credentials ---------------------------------------------------


class RemoteBackend(NamedTuple):
    """One remote backend's host side: ``check_host(settings)`` checks, without
    importing the SDK, its extra, environment and credentials (a cluster
    Engine runs this interpreter with the submit host's environment);
    ``credentials(settings)`` reads its secrets; ``client(credentials,
    settings)`` makes its production client, which alone keeps them."""

    check_host: Callable[[Any], None]
    credentials: Callable[[Any], tuple[str, ...]]
    client: Callable[[tuple[str, ...], Any], RemoteClient]


def remote_backend(name: str) -> RemoteBackend:
    return importlib.import_module(BACKENDS[name]).BACKEND


def remote_client(name: str, settings: Any) -> RemoteClient:
    """A production client of backend ``name``, its credentials read here."""
    backend = remote_backend(name)
    return backend.client(backend.credentials(settings), settings)


def read_registry_auth(path: str, backend: str) -> dict[str, tuple[str, str]]:
    """Registry -> (username, password) from a Docker-style auth file:
    its inline ``auths`` entries (``auth`` or ``username`` and
    ``password``). Credential stores and helpers are ignored, so no host
    program runs; the errors name the path, never a credential."""
    found: dict[str, tuple[str, str]] = {}
    try:
        auths = json.loads(Path(path).read_text()).get("auths") or {}
        for key, entry in auths.items():
            if entry.get("auth"):
                decoded = base64.b64decode(entry["auth"], validate=True).decode()
                username, _, password = decoded.partition(":")
            elif "username" in entry:
                username, password = entry["username"], entry.get("password")
            else:
                continue  # kept by a credential store
            if not all(isinstance(text, str) and text for text in (username, password)):
                raise ValueError("no username and password")
            # https://index.docker.io/v1/ -> docker.io, as pull_reference.
            host = key.split("://", 1)[-1].split("/", 1)[0].lower()
            if host == "index.docker.io":
                host = "docker.io"
            found[host] = (username, password)
    except (OSError, UnicodeError, ValueError, AttributeError, TypeError) as error:
        raise SetupError(
            f"sandbox policy environments.host.{backend}: registry_auth_file "
            f"{path} is unreadable ({type(error).__name__})"
        ) from None
    return found


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


def bind_ledger(client: RemoteClient, lease_root: Path, run_id: str) -> None:
    """Give ``client`` the run's sandbox ledger, beside the run's lease in
    ``lease_root``, when it keeps one (its ``ledger`` attribute): the
    Engine's broker, a resumed attempt and recovery bind the same file."""
    if hasattr(client, "ledger"):
        client.ledger = SandboxLedger(sandbox_ledger_path(lease_root, run_id), run_id)


# -- the SDK surface ------------------------------------------------------------


class RemoteProcess(Protocol):
    """One process started in a sandbox: ``(1|2, bytes)`` output until it
    ends; then ``exit_code`` is set (None when the stream broke first), a
    signal death as 128 + its number, as a shell reports it."""

    pid: int
    exit_code: int | None

    def __iter__(self) -> Iterator[tuple[int, bytes]]: ...

    def close(self) -> None: ...


class RemoteClient(Protocol):
    """The slice of a backend's SDK this module uses (faked in tests).

    ``name`` is the backend's (``[environments.host] backend``), in every
    message; ``recipe`` names how templates are made (bump it and every
    name changes). ``shape`` maps a phase's per-container limits to the
    fixed sizes a template or sandbox gets; ``image_env`` is the image's
    ENV as the sandbox applies it again to every process, given the
    registry config's ``Env``. ``lookup`` finds an env's sandbox by name
    when a listing may not show it yet (None: the backend names none).
    A client with a ``ledger`` attribute records its sandboxes there
    (``bind_ledger``) and lists them from it too; ``create`` raises
    ``ProviderCapacityError`` when the provider refused for capacity and
    the client proved that nothing was made (otherwise another error).

    What the backend lacks is emulated (the module's docstring):
    ``native_pid`` False, ``start`` only runs argv and its process's pid
    may be anything; ``native_user`` False, ``start`` is only asked for
    root; ``freeze`` "signal", ``pause`` and ``resume`` are never called
    and ``freeze_spare`` names the executables a freeze leaves running
    ("init": PID 1's; "caller": every process of the shell that ran the
    freeze). ``connect(sandbox_id, pid)`` follows the process
    ``start`` reported as ``pid``, after what was already read.
    ``argv_bytes``, when set, bounds the arguments ``start`` may send.

    ``build_template``'s steps are sandbox_e2b_dockerfile's step tuples
    when ``build_form`` is "steps", or a Dockerfile's own instructions
    after FROM (``lines``) when it is "lines"; None, the backend has no
    ``image_build``."""

    name: str
    recipe: str
    native_pid: bool
    native_user: bool
    freeze: str  # "native" | "signal"
    freeze_spare: tuple[str, ...]
    argv_bytes: int | None
    build_form: str | None  # "steps" | "lines" | None

    def shape(self, cpus: float, memory_mb: int, disk_mb: int) -> tuple: ...

    def template_exists(self, name: str) -> bool: ...

    def image_config(self, repository: str, tag: str) -> dict[str, Any]: ...

    def build_template(
        self,
        name: str,
        image: str,
        shape: tuple,
        *,
        log: Callable[[str], None],
        steps: Sequence[tuple[Any, ...]] = (),
        context: str | None = None,
        force: bool = False,
    ) -> None: ...

    def create(
        self,
        template: str,
        *,
        timeout: int,
        metadata: Mapping[str, str],
        allow_internet: bool,
        allow_out: Sequence[str] = (),
    ) -> str: ...

    def image_env(
        self, sandbox_id: str, image_env: Sequence[str]
    ) -> dict[str, str]: ...

    def lookup(self, env_id: str) -> str | None: ...

    def state(self, sandbox_id: str) -> str | None: ...

    def kill(self, sandbox_id: str) -> None: ...

    def pause(self, sandbox_id: str) -> None: ...

    def resume(self, sandbox_id: str, *, timeout: int) -> None: ...

    def list(self, metadata: Mapping[str, str]) -> list[tuple[str, dict[str, str]]]: ...

    def write_file(self, sandbox_id: str, path: str, data: bytes | Any) -> None: ...

    def read_file(self, sandbox_id: str, path: str) -> bytes: ...

    def read_stream(self, sandbox_id: str, path: str) -> io.RawIOBase: ...

    def start(
        self,
        sandbox_id: str,
        argv: Sequence[str],
        *,
        envs: Mapping[str, str],
        cwd: str,
        user: str,
    ) -> RemoteProcess: ...

    def connect(self, sandbox_id: str, pid: int) -> RemoteProcess: ...

    def close(self) -> None: ...


def _registry_authorization(
    http: Any, challenge: str, login: tuple[str, str] | None
) -> str:
    """The Authorization the registry's challenge asks for: a pull token
    from its Bearer realm (anonymous without ``login``), or Basic."""
    scheme, _, params = challenge.partition(" ")
    if scheme.lower() == "basic" and login is not None:
        return "Basic " + base64.b64encode(":".join(login).encode()).decode()
    if scheme.lower() != "bearer":
        raise InfrastructureError(
            "the registry wants credentials (set them in registry_auth_file)"
        )
    # realm="...",service="...",scope="repository:a/b:pull" (a scope may
    # hold commas inside its quotes).
    fields = dict(re.findall(r'(\w+)="([^"]*)"', params))
    realm = fields.pop("realm", None)
    if not realm:
        raise InfrastructureError("the registry's auth challenge names no realm")
    response = http.get(realm, params=fields, auth=login)
    response.raise_for_status()
    body = response.json()
    token = str(body.get("token") or body.get("access_token") or "")
    return f"Bearer {token}" if token else ""


def registry_image_config(
    repository: str,
    tag: str,
    *,
    proxy: str | None = None,
    timeout: float = 30.0,
    login: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """The image's config (WorkingDir, User, Entrypoint, ...) from its
    registry, with ``login`` or anonymously: a template keeps at most its
    ENV.

    ``repository`` is ``domain/remote`` (sandbox_images.pull_reference),
    ``tag`` a tag or a sha256 digest; an index resolves to linux/amd64, as
    the backends run. The config blob is checked against its digest.
    """
    import httpx

    domain, _, remote = repository.partition("/")
    base = f"https://{REGISTRY_HOSTS.get(domain, domain)}/v2/{remote}"
    authorization = ""
    with httpx.Client(proxy=proxy, timeout=timeout, follow_redirects=True) as http:

        def get(url: str, accept: str | None = None) -> Any:
            nonlocal authorization
            for _ in range(2):
                headers = {"Accept": accept} if accept else {}
                if authorization:
                    headers["Authorization"] = authorization
                response = http.get(url, headers=headers)
                challenge = response.headers.get("www-authenticate")
                if response.status_code == 401 and not authorization and challenge:
                    authorization = _registry_authorization(http, challenge, login)
                    continue
                response.raise_for_status()
                return response
            raise InfrastructureError("the registry's auth realm gave no token")

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
    keys = IMAGE_CONFIG_KEYS + BUILD_CONFIG_KEYS
    return {key: config[key] for key in keys if config.get(key)}


def run_in(
    client: RemoteClient,
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
    client: RemoteClient, sandbox_id: str, pid: int, stopped: Callable[[], bool]
) -> RemoteProcess | None:
    """Follow a process again after its stream broke. A pause may cut every
    stream while the process lives on (and the sandbox keeps an ended one's
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
                # Still pausing, or its agent not back yet: ask the state again.
                failures += 1
                if failures >= REATTACH_TRIES:
                    LOGGER.info(
                        "%s reconnect to %s failed: %s", client.name, pid, error
                    )
                    return None
        time.sleep(REATTACH_SEC)
    return None


# -- emulation ------------------------------------------------------------------


def config_env(image_env: Sequence[str]) -> dict[str, str]:
    """The registry config's ``Env`` ("NAME=value") as a mapping: the image
    ENV a client without its own record applies again to every process."""
    env: dict[str, str] = {}
    for item in image_env:
        key, separator, value = item.partition("=")
        if separator and key:
            env[key] = value
    return env


# The command becomes the leader of a new session and process group, and
# prints its pid (that group's id) before it execs. The outer sh runs setsid
# in the foreground: setsid forks when its caller leads a group and the
# outer sh then still waits for the command, so its exit (128 + N for a
# signal) is the command's. An ``&`` job instead would start the command
# with SIGINT and SIGQUIT ignored. Its stdin is /dev/null, as a Docker
# exec's (Modal leaves every exec a stdin pipe that never closes).
GROUP_SCRIPT = (
    """setsid sh -c 'printf "%s\\n" "$$"; exec "$@"' rsi "$@" < /dev/null; exit $?"""
)


def group_argv(argv: Sequence[str]) -> list[str]:
    """argv in its own process group, reporting its pid first (stdout)."""
    return ["sh", "-c", GROUP_SCRIPT, "rsi", *argv]


class _GroupProcess:
    """A ``group_argv`` process: its pid is its first stdout line, read
    before the caller gets it; output that came with it is replayed."""

    def __init__(
        self,
        process: RemoteProcess,
        client: RemoteClient,
        sandbox_id: str,
        timeout: float,
    ) -> None:
        backend = client.name
        self._process = process
        self._events = iter(process)
        self._held: list[tuple[int, bytes]] = []
        line = bytearray()
        found: dict[str, Any] = {}
        done = threading.Event()

        def read() -> None:
            try:
                for stream, data in self._events:
                    if stream != 1:
                        self._held.append((stream, data))
                        continue
                    line.extend(data)
                    if b"\n" in line:
                        head, _, rest = bytes(line).partition(b"\n")
                        found["pid"] = head
                        if rest:
                            self._held.append((1, rest))
                        return
            except Exception as error:
                found["error"] = error
            finally:
                done.set()

        def reap() -> None:
            """The command may run all the same: kill its group once its
            pid line comes."""
            done.wait(LATE_PID_SEC)
            process.close()
            head = found.get("pid", b"")
            if head.isdigit():
                kill = ["kill", "-s", "KILL", "--", f"-{int(head)}"]
                try:
                    run_in(client, sandbox_id, kill)
                except Exception as error:
                    LOGGER.warning("%s late command not killed: %s", backend, error)

        threading.Thread(
            target=read, name=f"rsi-sandbox-{backend}-pid", daemon=True
        ).start()
        if not done.wait(timeout):
            threading.Thread(
                target=reap, name=f"rsi-sandbox-{backend}-reap", daemon=True
            ).start()
            raise InfrastructureError(
                f"{backend} command reported no pid within {timeout:g} s"
            )
        if "error" in found:
            process.close()
            raise found["error"]
        try:
            self.pid = int(found["pid"])
        except (KeyError, ValueError):
            process.close()
            tail = b"".join(data for stream, data in self._held if stream == 2)
            raise InfrastructureError(
                f"{backend} command could not start (it needs sh and setsid): "
                f"{tail[-512:].decode('utf-8', 'replace')}"
            ) from None

    @property
    def exit_code(self) -> int | None:
        return self._process.exit_code

    def __iter__(self) -> Iterator[tuple[int, bytes]]:
        held, self._held = self._held, []
        yield from held
        yield from self._events

    def close(self) -> None:
        self._process.close()


def grouped_argv(client: RemoteClient, argv: Sequence[str]) -> list[str]:
    """argv in its own process group, its pid that group's id: ``setsid``
    where the client reports pids, else ``group_argv``."""
    return ["setsid", *argv] if client.native_pid else group_argv(argv)


def start_group(
    client: RemoteClient,
    sandbox_id: str,
    argv: Sequence[str],
    *,
    envs: Mapping[str, str],
    cwd: str,
    user: str,
    timeout: float = PID_LINE_SEC,
) -> RemoteProcess:
    """``grouped_argv`` started; without native pids, its pid read first."""
    grouped = grouped_argv(client, argv)
    process = client.start(sandbox_id, grouped, envs=envs, cwd=cwd, user=user)
    if client.native_pid:
        return process
    return _GroupProcess(process, client, sandbox_id, timeout)


# Where a freeze records the processes it stopped (root's, outside /tmp).
FROZEN_RECORD = "/run/.rsi-frozen"
# Sets st, pp and t (state, ppid, start time) from $root/$1/stat; fails when
# the process is gone. comm (in parentheses) may hold spaces and ")".
_PROC_STAT = r"""stat_of() {
  s=
  { read -r s < "$root/$1/stat"; } 2>/dev/null
  [ -n "$s" ] || return 1
  s=${s##*) }
  st=${s%% *}
  s=${s#* }
  pp=${s%% *}
  i=0
  while [ $i -lt 18 ]; do s=${s#* }; i=$((i + 1)); done
  t=${s%% *}
}
"""
# $1 the proc root, $2 the record, $3 the signal command (kill), $4 this
# shell's pid (empty: $$; tests name one in a made-up root), $5 tries to see
# every stopped process stopped (0.1 s apart), then the executables spared
# ("init": PID 1's; "caller": every descendant of this shell's parent, a
# session shell that waits for its own helpers); an executable that cannot
# be read spares nothing, and "init" unread fails. SIGSTOPs every process
# but PID 1, this shell, its ancestors and children and the spared, round
# after round until a round stops none (a process may fork until it is
# stopped), recording each as "<pid> <start time>"; one already stopped, or
# dead, is left alone. Exits 0 once every recorded process shows stopped
# (or gone).
FREEZE_SCRIPT = (
    _PROC_STAT
    + r"""root=$1 rec=$2 sig=$3 self=${4:-$$} tries=$5
shift 5
init=$(readlink "$root/1/exe" 2>/dev/null)
spare= caller=
for e in "$@"; do
  if [ "$e" = init ]; then
    e=$init
    [ -n "$e" ] || { echo "cannot read init's executable" >&2; exit 1; }
  fi
  if [ "$e" = caller ]; then stat_of "$self" && caller=$pp; continue; fi
  spare="$spare
$e"
done
under() {
  q=$1
  while stat_of "$q" && [ "$pp" != 0 ]; do
    [ "$pp" = "$caller" ] && return 0
    q=$pp
  done
  return 1
}
keep=" 1 $self "
p=$self
while stat_of "$p" && [ "$pp" != 0 ]; do keep="$keep$pp "; p=$pp; done
mkdir -p "${rec%/*}" && (umask 077 && : > "$rec") || exit 1
round=0
while :; do
  n=0
  for d in "$root"/[0-9]*; do
    p=${d##*/}
    case $keep in *" $p "*) continue ;; esac
    if [ -n "$caller" ] && under "$p"; then keep="$keep$p "; continue; fi
    stat_of "$p" || continue
    case $st in [TtZX]) continue ;; esac
    [ "$pp" = "$self" ] && continue
    if [ -n "$spare" ]; then
      e=$(readlink "$d/exe" 2>/dev/null)
      [ -n "$e" ] && case "$spare
" in *"
$e
"*) keep="$keep$p "; continue ;; esac
    fi
    "$sig" -s STOP "$p" 2>/dev/null || continue
    echo "$p $t" >> "$rec"
    keep="$keep$p "
    n=$((n + 1))
  done
  [ $n = 0 ] && break
  round=$((round + 1))
  [ $round -lt 100 ] || { echo "processes keep starting" >&2; exit 1; }
done
i=0
while :; do
  left=
  while read -r p t0; do
    stat_of "$p" && [ "$t" = "$t0" ] || continue
    case $st in [TtZX]) ;; *) left="$left $p" ;; esac
  done < "$rec"
  [ -z "$left" ] && exit 0
  i=$((i + 1))
  [ $i -lt "$tries" ] || break
  sleep 0.1 2>/dev/null || sleep 1
done
echo "not stopped:$left" >&2
exit 1
"""
)
# $1 the proc root, $2 the record, $3 the signal command: SIGCONT what the
# freeze recorded, unless its pid now names another process; last first, so
# children (mostly recorded after their parents) go on before them: a job
# control shell continued first would see its job stopped.
THAW_SCRIPT = (
    _PROC_STAT
    + r"""root=$1 rec=$2 sig=$3
[ -f "$rec" ] || { echo "no freeze record" >&2; exit 1; }
last=
while read -r p t0; do last="$p $t0
$last"; done < "$rec"
printf %s "$last" | while read -r p t0; do
  stat_of "$p" && [ "$t" = "$t0" ] && "$sig" -s CONT "$p" 2>/dev/null
done
rm -f "$rec"
"""
)
FREEZE_TRIES = 30


class SignalFreeze:
    """freeze_work on a backend that cannot pause a sandbox: SIGSTOP its
    processes in it (``FREEZE_SCRIPT``), SIGCONT them on resume. Only this
    broker knows a sandbox is frozen; the backend sees it running (and its
    timeout keeps running, as the env deadline does)."""

    def __init__(self, client: RemoteClient) -> None:
        self._client = client
        self._lock = threading.Lock()
        self._frozen: set[str] = set()

    def frozen(self, sandbox_id: str) -> bool:
        with self._lock:
            return sandbox_id in self._frozen

    def _run(self, sandbox_id: str, what: str, argv: Sequence[str]) -> None:
        code, _, err = run_in(self._client, sandbox_id, argv)
        if code != 0:
            raise InfrastructureError(
                f"{self._client.name} {what} exited {code}: "
                f"{err[-512:].decode('utf-8', 'replace').strip()}"
            )

    def pause(self, sandbox_id: str) -> None:
        spare = tuple(self._client.freeze_spare)
        self._run(
            sandbox_id,
            "freeze",
            ["sh", "-c", FREEZE_SCRIPT, "rsi-freeze", "/proc", FROZEN_RECORD]
            + ["kill", "", str(FREEZE_TRIES), *spare],
        )
        with self._lock:
            self._frozen.add(sandbox_id)

    def resume(self, sandbox_id: str) -> None:
        self._run(
            sandbox_id,
            "thaw",
            ["sh", "-c", THAW_SCRIPT, "rsi-thaw", "/proc", FROZEN_RECORD, "kill"],
        )
        self.forget(sandbox_id)

    def forget(self, sandbox_id: str) -> None:
        with self._lock:
            self._frozen.discard(sandbox_id)


def template_name(prefix: str, image: str, shape: Sequence[Any], recipe: str) -> str:
    """Deterministic: the image's digest (or its reference when it names
    none), the shape and the recipe ("<source>|<cpu>|<memory>|r1" on e2b)."""
    _, at, digest = image.rpartition("@")
    source = digest if at and digest.startswith("sha256:") else image
    key = "|".join(str(item) for item in (source, *shape, recipe))
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:20]}"


def built_template_name(
    prefix: str,
    context_digest: str,
    image: str,
    steps: Sequence[tuple[Any, ...]],
    config: Mapping[str, Any],
    shape: Sequence[Any],
    recipe: str,
) -> str:
    """Deterministic: what a build sends (the context and Dockerfile, the
    base reference, the steps) and the config env_create applies, plus
    the shape and the recipe. A change to the translation renames it."""
    key = json.dumps(
        [context_digest, image, steps, config, *shape, recipe], sort_keys=True
    )
    return f"{prefix}-{hashlib.sha256(key.encode()).hexdigest()[:20]}"


def template_attrs(
    name: str,
    image: str,
    shape: Sequence[Any],
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
        "RemoteTemplate": name,
        "RemoteImage": image,
        "RemoteShape": list(shape),
    }


def grant_shape(client: RemoteClient, grant: Any) -> tuple:
    """The phase's per-container limits as the client sizes a template; a
    pull or build fixes it, and plan reads it from the image."""
    return client.shape(
        grant.cpus_per_container,
        grant.memory_mb_per_container,
        grant.disk_mb_per_container,
    )


@dataclass(eq=False)
class _Build:
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class RemoteTemplates:
    """Make sure a template exists, building it at most once per name at a
    time in this broker. Names already seen are not asked again."""

    def __init__(self, client: RemoteClient, prefix: str) -> None:
        self._client = client
        self.prefix = prefix
        self._lock = threading.Lock()
        self._building: dict[str, _Build] = {}
        self._ready: set[str] = set()

    def ensure(
        self,
        name: str,
        image: str,
        shape: tuple,
        log: Callable[[str], None] = lambda line: None,
        *,
        steps: Sequence[tuple[Any, ...]] = (),
        context: Callable[[], str] | None = None,
        force: bool = False,
    ) -> bool:
        """True when this call built the template. ``context`` makes the
        directory COPY steps read, only when the template is built;
        ``force`` builds it again without the backend's layer cache."""
        while True:
            with self._lock:
                if name in self._ready and not force:
                    return False
                build = self._building.get(name)
                if build is None:
                    build = self._building[name] = _Build()
                    break
            log(f"waiting for the build of template {name}\n")
            build.done.wait()
            if force:
                continue  # a forced build is its own, after the one running
            if build.error is not None:
                raise JobFailed("image", f"template build failed: {build.error}")
            return False
        built = False
        try:
            if force or not self._client.template_exists(name):
                log(f"building template {name} from {image}\n")
                self._client.build_template(
                    name,
                    image,
                    shape,
                    log=log,
                    steps=steps,
                    context=None if context is None else context(),
                    force=force,
                )
                built = True
            with self._lock:
                self._ready.add(name)
            return built
        except Exception as error:
            build.error = error
            LOGGER.warning(
                "%s template %s build failed: %s", self._client.name, name, error
            )
            if isinstance(error, JobFailed):
                raise  # context()'s, of its own kind
            raise JobFailed("image", f"template build failed: {error}"[:1024]) from None
        finally:
            with self._lock:
                del self._building[name]
            build.done.set()


class RemoteImages:
    """``image_pull`` on a remote backend: make sure the image's template
    exists (it may build), then bind a handle to it. Nothing is downloaded
    to the host."""

    def __init__(self, templates: RemoteTemplates, client: RemoteClient) -> None:
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
            shape = grant_shape(self._client, job.session.env_grant)
            name = template_name(
                self._templates.prefix, reference, shape, self._client.recipe
            )
            runner.log(job, f"{self._client.name} template {name} for {reference}\n")
            config = {
                key: item
                for key, item in self.config(repository, tag).items()
                if key in IMAGE_CONFIG_KEYS
            }
            self._templates.ensure(
                name, reference, shape, log=lambda line: runner.log(job, line)
            )
            if job.cancel.is_set():
                raise JobFailed("canceled", "the pull was canceled")
            attrs = template_attrs(name, reference, shape, config)
            image = bind_template(runner, job, lease, attrs, reference)
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


def bind_template(
    runner: Any,
    job: Any,
    lease: SandboxImageLease,
    attrs: Mapping[str, Any],
    ref: str,
    fingerprint: str | None = None,
) -> Image:
    """Journal the job's template lease present and bind its handle, under
    the broker lock, unless the job was cancelled or its session ended (a
    session's removal does not wait for its jobs). A fingerprint marks a
    built image (per-session dedupe, B9)."""
    with runner._lock:
        if job.session.revoked or job.cancel.is_set():
            raise JobFailed("canceled", f"the {job.kind} was canceled")
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
            "pulled" if fingerprint is None else "built",
            attrs["Id"],
            ref,
            0,
            attrs,
            lease,
            fingerprint=fingerprint,
        )
        runner.bind(image)
    return image


class RemoteBuilds:
    """``image_build`` on a remote backend: a single-stage Dockerfile
    becomes steps on the pull recipe (sandbox_e2b_dockerfile), or its own
    instructions where the client's ``build_form`` is "lines". The template
    is named by what its build sends (``built_template_name``), so like a
    pulled one it is a shared cache: its lease is a pulled record, nothing
    runs or stays on the host, and session ends and recovery have nothing
    to remove. A build cannot be stopped once the backend has it; a
    cancelled or timed-out job ends when the build does, and binds
    nothing."""

    symlinks = False  # a context symlink would let the SDK read host files
    lease_kind = "pulled"

    def __init__(
        self,
        envs: Any,
        client: RemoteClient,
        templates: RemoteTemplates,
        images: RemoteImages,
        spool_root: Path,
    ) -> None:
        self._envs = envs
        self._client = client
        self._templates = templates
        self._images = images
        self._spool = Path(spool_root) / "build"

    def input_path(self, job_id: str) -> Path:
        """A private spool file for one build's input tar."""
        self._spool.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._spool, 0o700)
        return self._spool / f"{job_id}.tar"

    def admit(
        self, staged: Any, path: Path, options: Mapping[str, Any], network: str
    ) -> None:
        """At admission: what the backend cannot build as Docker does is
        refused."""
        from rsi_harness.runtime import sandbox_e2b_dockerfile as dockerfile

        backend = self._client.name
        if network != "public":
            raise unsupported("network", "a build without network", backend)
        plan = dockerfile.parse(
            staged.dockerfile, options["build_args"], options["target"], backend
        )
        # Without the base image's config: refused for the Dockerfile alone.
        found, _ = dockerfile.steps(plan, {}, backend)
        dockerfile.refuse_archives(found, path, staged.directory, backend)

    def run(self, job: Any) -> None:
        """The build job's thread body: base config, steps, template, bind."""
        from rsi_harness.runtime import sandbox_e2b_dockerfile as dockerfile

        runner = self._envs.images
        request = job.detail
        state, error, image = "failed", None, None
        context = request.input_path.with_suffix(".ctx")
        try:
            runner.running(job)
            shape = grant_shape(self._client, job.session.env_grant)
            backend = self._client.name
            plan = dockerfile.parse(
                request.input.dockerfile, request.build_args, request.target, backend
            )
            repository, tag = pull_reference(plan.base)
            try:
                steps, config = dockerfile.steps(
                    plan, self._images.config(repository, tag), backend
                )
                dockerfile.refuse_archives(
                    steps, request.input_path, request.input.directory, backend
                )
            except SandboxError as refused:  # by the base image's config
                raise JobFailed("refused", refused.message) from None
            if self._client.build_form == "lines":
                steps = dockerfile.lines(plan, steps)
            name = built_template_name(
                self._templates.prefix,
                request.input.digest,
                plan.base,
                steps,
                config,
                shape,
                self._client.recipe,
            )
            runner.log(
                job, f"{self._client.name} template {name} built on {plan.base}\n"
            )

            def extract() -> str:
                try:  # a second copy of the context in the spool
                    self._envs._admit_spool("stage_id")
                except SandboxError as refused:
                    raise JobFailed(refused.code, refused.message) from None
                dockerfile.extract_context(
                    request.input_path, request.input.directory, context
                )
                return str(context)

            self._templates.ensure(
                name,
                plan.base,
                shape,
                log=lambda line: runner.log(job, line),
                steps=steps,
                context=extract,
                force=request.no_cache,
            )
            attrs = template_attrs(name, plan.base, shape, config)
            image = bind_template(
                runner, job, request.lease, attrs, name, request.fingerprint
            )
            state = "succeeded"
        except Exception as failure:
            state, error = runner.outcome(job, failure)
        finally:
            request.input_path.unlink(missing_ok=True)
            shutil.rmtree(context, ignore_errors=True)
            if state != "succeeded":
                try:
                    runner._envs._commit_image(
                        request.lease.model_copy(update={"state": "removed"})
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

    # Templates are a shared cache: nothing per session to keep or remove
    # (remove_session unbinds the handles, as for pulls).

    def retained(self, session: Any) -> bool:
        return False

    def sweep(self, dead: Callable[[Any], bool]) -> None:
        pass

    def kill_session(self, session: Any, deadline: float) -> list[BaseException]:
        return []

    def remove_session(self, session: Any, deadline: float) -> list[BaseException]:
        return []


# -- plan ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RemoteServicePlan:
    idx: int
    name: str
    container_name: str
    image: str
    image_id: str
    template: str
    source: str
    shape: tuple
    argv: tuple[str, ...]
    env: Mapping[str, str]
    # The registry config's Env ("NAME=value"), for the client's image_env.
    image_env: tuple[str, ...]
    user: str | None
    working_dir: str | None
    allow_internet: bool
    allow_out: tuple[str, ...] = ()
    implicit_volumes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RemoteEnvPlan:
    """The broker's view of a plan (as sandbox_env_docker.EnvPlan's)."""

    owner: SandboxOwner
    env_id: str
    spec_sha256: str
    network_mode: str
    services: tuple[RemoteServicePlan, ...]
    backend: str
    cpus_milli: int
    memory_mb: int
    disk_mb: int
    network: None = None
    volumes: tuple = ()
    swap_mb: int = 0

    def service(self, name: str) -> RemoteServicePlan:
        for service in self.services:
            if service.name == name:
                return service
        raise SandboxError("invalid", "service", f"unknown service {name}")

    def lease(self, *, created_at: float, expires_at: float) -> SandboxEnvLease:
        return SandboxEnvLease(
            owner=self.owner,
            env_id=self.env_id,
            spec_sha256=self.spec_sha256,
            backend=self.backend,
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


def refuse_unsupported(spec: EnvSpec, backend: str) -> None:
    """What one sandbox of ``backend`` cannot express is refused, never
    ignored."""
    found = remote_refusal(spec.model_dump(mode="json"), backend)
    if found is not None:
        raise unsupported(*found, backend)


# -- backend ------------------------------------------------------------------------


@dataclass(eq=False)
class _Service:
    process: Any
    # Its group's: a stream followed again may not report it.
    pid: int = 0
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
    # The image's passwd: uid -> name, and name -> (uid, gid, home).
    users: dict[str, str] | None = None
    accounts: dict[str, tuple[int, int, str]] | None = None


@dataclass(frozen=True, slots=True)
class RemoteTarget:
    """One owned sandbox for copies and path-stat; ``require`` refuses a
    paused or vanished one."""

    sandbox_id: str
    require: Callable[[], None]


Commit = Callable[[SandboxEnvLease], SandboxEnvLease]


class SandboxEnvRemoteBackend:
    """``SandboxEnvDockerBackend``'s surface over remote sandboxes.

    env_create creates the sandbox (so the plugin can seed it before start);
    env_start starts the service command, if the spec has one. Pause and
    resume are the client's (or SignalFreeze's), kill is removal. Every step
    commits through the broker's journal like the Docker backend's.
    """

    def __init__(
        self,
        client: RemoteClient,
        templates: RemoteTemplates,
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
        self._freeze = SignalFreeze(client) if client.freeze == "signal" else None
        # Set by remote_env_runtime: stop_service ends the service's execs too.
        self.execs: RemoteExecApi | None = None

    # -- planning ------------------------------------------------------------

    def plan(
        self,
        owner: SandboxOwner,
        env_id: str,
        spec: EnvSpec,
        images: Mapping[str, Mapping[str, Any]],
        **_options: Any,
    ) -> RemoteEnvPlan:
        backend = self._client.name
        refuse_unsupported(spec, backend)
        ((name, service),) = spec.services.items()
        attrs = images.get(service.image)
        if attrs is None:
            raise SandboxError(
                "permission", f"spec.services.{name}.image", "unknown image handle"
            )
        template = attrs.get("RemoteTemplate")
        if not isinstance(template, str):
            article = "an" if backend[0] in "aeiou" else "a"
            raise SandboxError(
                "invalid",
                f"spec.services.{name}.image",
                f"not {article} {backend} template image",
            )
        # The Engine's merge (sandbox_env_docker): the spec's entrypoint
        # clears the image's Cmd; WORKDIR and USER are the image's defaults.
        config = attrs.get("Config") or {}
        user = service.user or config.get("User") or None
        self._check_user(user)
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
        # Normalized, without the ports refuse_unsupported lets through.
        allow_out = tuple(
            dict.fromkeys(parse_allow_entry(text).host for text in spec.allowlist)
        )
        return RemoteEnvPlan(
            owner=owner,
            env_id=env_id,
            spec_sha256=env_spec_digest(spec),
            network_mode=spec.network,
            services=(
                RemoteServicePlan(
                    idx=0,
                    name=name,
                    container_name=env_container_name(env_id, 0),
                    image=service.image,
                    image_id=attrs["Id"],
                    template=template,
                    source=attrs["RemoteImage"],
                    shape=tuple(attrs["RemoteShape"]),
                    argv=(*entrypoint, *command),
                    env=dict(service.env),
                    image_env=tuple(config.get("Env") or ()),
                    user=user,
                    working_dir=service.working_dir
                    or _clean_dir(config.get("WorkingDir"))
                    or None,
                    allow_internet=spec.network == "public"
                    and service.network == "env",
                    allow_out=allow_out if service.network == "env" else (),
                ),
            ),
            backend=backend,
            cpus_milli=round(service.cpus * 1000),
            memory_mb=service.memory_mb,
            disk_mb=spec.disk_mb,
        )

    def allowlist_notes(self, plan: RemoteEnvPlan) -> tuple[str, ...]:
        return ()

    def refresh_network(self, plan: RemoteEnvPlan) -> bool:
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
        """The env's remaining lifetime plus slack, within the account's
        maximum sandbox length (the backend refuses a longer timeout). The
        policy keeps lifetimes within that length, so at most the slack is
        cut."""
        remaining = max(0.0, lease.expires_at - self._wall())
        return min(self._max_timeout, math.ceil(remaining) + TIMEOUT_SLACK_SEC)

    def _passwd(self, box: _Box) -> None:
        if box.users is not None:
            return
        try:
            raw = self._client.read_file(box.sandbox_id, "/etc/passwd")
        except FileNotFoundError:
            raw = b""
        users: dict[str, str] = {}
        accounts: dict[str, tuple[int, int, str]] = {}
        for line in raw.decode("utf-8", "replace").splitlines():
            parts = line.split(":")
            if len(parts) > 2 and parts[2].isdigit():
                users.setdefault(str(int(parts[2])), parts[0])
            if len(parts) > 5 and parts[2].isdigit() and parts[3].isdigit():
                accounts.setdefault(parts[0], (int(parts[2]), int(parts[3]), parts[5]))
        box.users, box.accounts = users, accounts

    @staticmethod
    def _check_user(user: str | None) -> None:
        if user is not None and ":" in user:
            raise SandboxError(
                "unsupported",
                "user",
                "remote backends do not support an explicit primary group",
            )

    def user_name(self, box: _Box, user: str | None) -> str:
        """Processes start as a user name: 0 is root, other uids come from
        passwd. Where setpriv switches users, a name needs its entry, and a
        uid without one stays a uid (``as_user``)."""
        name = user or box.user or "root"
        self._check_user(name)
        if name.isdigit():
            if int(name) == 0:
                return "root"
            self._passwd(box)
            assert box.users is not None
            found = box.users.get(str(int(name)))
            if found is None and not self._client.native_user:
                return str(int(name))
            if found is None:
                raise SandboxError(
                    "invalid",
                    "user",
                    f"uid has no user name in the image ({self._client.name} backend)",
                )
            name = found
        if not self._client.native_user and name != "root":
            self._passwd(box)
            assert box.accounts is not None
            if name not in box.accounts:
                raise SandboxError(
                    "invalid",
                    "user",
                    f"user has no passwd entry in the image "
                    f"({self._client.name} backend)",
                )
        return name

    def spawn(
        self,
        sandbox_id: str,
        argv: Sequence[str],
        *,
        envs: Mapping[str, str],
        cwd: str,
        user: str,
    ) -> RemoteProcess:
        """A service or exec command, in its own process group, as ``user``
        (a name ``user_name`` gave). A client that starts processes only as
        root gets ``setpriv`` to the user's passwd ids and groups, with the
        user's HOME (``/`` for a bare uid) unless the environment sets one
        (as the Engine does)."""
        if not self._client.native_user and user != "root":
            argv = self.as_user(sandbox_id, argv, user)
            accounts = self.box(sandbox_id).accounts
            assert accounts is not None
            home = accounts[user][2] if user in accounts else "/"
            envs, user = {"HOME": home, **envs}, "root"
        return start_group(
            self._client, sandbox_id, argv, envs=envs, cwd=cwd, user=user
        )

    def as_user(self, sandbox_id: str, argv: Sequence[str], user: str) -> list[str]:
        """argv as ``spawn`` runs it for ``user``, before its group."""
        if self._client.native_user or user == "root":
            return list(argv)
        box = self.box(sandbox_id)
        self._passwd(box)
        assert box.accounts is not None
        if user in box.accounts:
            uid, gid, _ = box.accounts[user]
            ids = [f"--reuid={uid}", f"--regid={gid}", "--init-groups"]
        else:
            # A uid without a passwd entry: gid 0 and no groups, as Docker.
            ids = [f"--reuid={user}", "--regid=0", "--clear-groups"]
        return ["setpriv", *ids, "--", *argv]

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
        self, plan: RemoteEnvPlan, lease: SandboxEnvLease, commit: Commit
    ) -> SandboxEnvLease:
        """The template, then the sandbox; the service command waits for start."""
        if lease.state != "planned" or not lease.pending_mutation:
            raise InfrastructureError("sandbox env create needs its pending plan")
        service = plan.services[0]
        try:
            self._templates.ensure(service.template, service.source, service.shape)
        except JobFailed as error:
            self._gone(lease, commit)
            raise SetupError(
                f"{self._client.name} template unavailable: {error.message}"
            ) from None
        try:
            sandbox_id = self._client.create(
                service.template,
                timeout=self._timeout(lease),
                metadata=sandbox_metadata(plan.owner, plan.env_id, service.name),
                allow_internet=service.allow_internet,
                allow_out=service.allow_out,
            )
        except ProviderCapacityError:
            # The client proved nothing was made (the contract), so nothing
            # is looked up under the rate limit that refused it: absent,
            # then retried by the caller (env_create answers quota
            # provider_capacity).
            self._gone(lease, commit)
            raise
        except Exception as error:
            # A create without an answer may still land: find it by metadata.
            self._kill_env(lease)
            self._gone(lease, commit)
            raise SetupError(
                f"{self._client.name} refused the sandbox: {error}"
            ) from None
        lease = commit(
            _service(lease, service.idx, sandbox_id=sandbox_id, state="created")
        )
        if service.working_dir is not None:
            # As the Engine does for a container's WORKDIR (a missing cwd
            # is refused); a failure surfaces at the first exec.
            run_in(self._client, sandbox_id, ["mkdir", "-p", service.working_dir])
        box = _Box(
            sandbox_id,
            {**self._client.image_env(sandbox_id, service.image_env), **service.env},
            service.working_dir or "/",
            service.user,
        )
        with self._lock:
            self._boxes[sandbox_id] = box
        return commit(_revise(lease, state="created", pending_mutation=False))

    def start(
        self,
        plan: RemoteEnvPlan,
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
                process = self.spawn(
                    box.sandbox_id,
                    service.argv,
                    envs=box.env,
                    cwd=box.cwd,
                    user=self.user_name(box, None),
                )
            except Exception as error:
                LOGGER.warning(
                    "%s env %s service start failed: %s",
                    self._client.name,
                    lease.env_id,
                    error,
                )
                lease = commit(_revise(lease, state="failed", reason="start_failed"))
                return EnvStartResult(
                    lease,
                    "failed",
                    "start_failed",
                    service.name,
                    "the service command could not start",
                )
            box.service = _Service(process, process.pid)
            threading.Thread(
                target=self._drain,
                args=(box.sandbox_id, box.service),
                name=f"rsi-sandbox-{self._client.name}-service",
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
                    self._client, sandbox_id, service.pid, lambda: service.closed
                )
                if follow is None:
                    if not service.closed:
                        LOGGER.info(
                            "%s service stream lost: %s", self._client.name, failure
                        )
                        service.lost = True
                    return
                process = service.process = follow
        finally:
            service.ended.set()

    # -- runtime ---------------------------------------------------------------

    def _state(self, sandbox_id: str) -> str | None:
        state = self._client.state(sandbox_id)
        if state == "running" and self._freeze is not None:
            # The backend sees a signal-frozen sandbox running.
            return "paused" if self._freeze.frozen(sandbox_id) else state
        return state

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
        """freeze_work: the client pauses the sandbox, or SignalFreeze
        stops its processes; proven by state."""
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
                    if self._freeze is None:
                        self._client.pause(record.sandbox_id)
                    else:
                        self._freeze.pause(record.sandbox_id)
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
        """Resume, and set the sandbox's timeout again (a pause may restart
        its timer)."""
        resumed = []
        try:
            for record in lease.services:
                if record.state != "paused" or record.sandbox_id is None:
                    continue
                failure: Exception | None = None
                with admission() if admission is not None else nullcontext():
                    try:
                        if self._freeze is None:
                            self._client.resume(
                                record.sandbox_id, timeout=self._timeout(lease)
                            )
                        else:
                            self._freeze.resume(record.sandbox_id)
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
                self._signal(box.sandbox_id, running.pid, "KILL")
            elif running is not None and not running.ended.is_set():
                for signum, wait in (("TERM", timeout_sec), ("KILL", self._stop_proof)):
                    self._signal(box.sandbox_id, running.pid, signum)
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
            LOGGER.warning(
                "%s signal %s to %s failed: %s", self._client.name, name, pid, error
            )

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
                    f"recovery_required: {self._client.name} sandbox {sandbox_id} "
                    f"removal is unproven: {failure}"
                ) from failure
            time.sleep(self._poll)
        if self._freeze is not None:
            self._freeze.forget(sandbox_id)
        with self._lock:
            box = self._boxes.pop(sandbox_id, None)
        if box is not None and box.service is not None:
            box.service.closed = True
            box.service.process.close()

    def _kill_env(self, lease: SandboxEnvLease) -> None:
        """Sandboxes of the env the journal may lack (a create in flight),
        listed or, before a listing shows them, found by name."""
        selector = {f"{META_PREFIX}env_id": lease.env_id}
        found = [sandbox_id for sandbox_id, _ in self._client.list(selector)]
        named = self._client.lookup(lease.env_id)
        if named is not None and named not in found:
            found.append(named)
        for sandbox_id in found:
            self._kill(sandbox_id)

    def terminate(
        self,
        lease: SandboxEnvLease,
        commit: Commit,
        *,
        deadline: float | None = None,
    ) -> SandboxEnvLease:
        """Nothing of the env executes afterwards: a kill.

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
                f"recovery_required: {self._client.name} env {lease.env_id} "
                f"removal failed: {error}"
            ) from error

    # -- targets ---------------------------------------------------------------

    def _live(self, lease: SandboxEnvLease, service: str) -> SandboxEnvServiceLease:
        record = self._record(lease, service)
        if record.state in ("planned", "removed") or record.sandbox_id is None:
            raise SandboxError("invalid", "service", f"service {service} is absent")
        return record

    def archive_target(self, lease: SandboxEnvLease, service: str) -> RemoteTarget:
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

        return RemoteTarget(sandbox_id, require)

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
class _RemoteExec:
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


class RemoteExecApi:
    """The Docker low-level exec calls ExecPump makes, over the client.

    ``exec_start`` starts the process at once and returns a socket that
    carries its output in the Engine's 8-byte frames; ``exec_inspect``
    answers from what the stream reported. A stream cut by a pause is
    reattached once the sandbox runs again; one that cannot be is ``lost``
    (NotFound, as for an exec the Engine forgot).
    """

    def __init__(self, client: RemoteClient, backend: SandboxEnvRemoteBackend) -> None:
        self._client = client
        self._backend = backend
        self._lock = threading.Lock()
        self._execs: dict[str, _RemoteExec] = {}

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
        record = _RemoteExec(
            exec_id=secrets.token_hex(32),
            sandbox_id=container,
            argv=tuple(cmd),
            env={**box.env, **(environment or {})},
            cwd=workdir or box.cwd,
            user=self._backend.user_name(box, user or None),
        )
        limit = self._client.argv_bytes
        if limit is not None:
            # What start sends: the command, setpriv and the group wrapper.
            sent = grouped_argv(
                self._client,
                self._backend.as_user(container, record.argv, record.user),
            )
            if sum(len(item.encode()) for item in sent) > limit:
                raise SandboxError(
                    "invalid",
                    "cmd",
                    f"the command's arguments exceed {limit} bytes "
                    f"({self._client.name} backend)",
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
            process = self._backend.spawn(
                record.sandbox_id,
                record.argv,
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
            name=f"rsi-sandbox-{self._client.name}-exec",
            daemon=True,
        ).start()
        return _Attach(ours, lambda: self._close(record, ours))

    def exec_inspect(self, exec_id: str) -> dict[str, Any]:
        record = self._execs.get(exec_id)
        if record is None or record.lost:
            raise NotFound(f"{self._client.name} exec is gone")
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

    def _close(self, record: _RemoteExec, sock: socket.socket) -> None:
        record.closed = True
        sock.close()

    def _forward(self, record: _RemoteExec, process: RemoteProcess) -> None:
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
                    LOGGER.info(
                        "%s exec %s stream lost: %s",
                        self._client.name,
                        record.exec_id,
                        failure,
                    )
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


class RemoteProcessTable:
    """ProcessTable's surface for ExecPump: no host /proc or cgroup; a
    leader is alive while its stream is."""

    def __init__(self, execs: RemoteExecApi, backend: str) -> None:
        self._execs = execs
        self._backend = backend

    def scope(self, container_id: str, pid: object) -> Path:
        return Path("/", self._backend, container_id)

    def oom_kills(self, scope: Path) -> int:
        return 0

    def stat(self, pid: int) -> None:
        return None

    def process(self, container_id: str, scope: Path, pid: int) -> ExecProcess:
        return ExecProcess(container_id, scope, pid, pid, 0, pid, pid)

    def targets(self, process: ExecProcess, *, group: bool) -> tuple[int, ...]:
        alive = self._execs.running(process.container_id, process.host_pid)
        return (process.host_pid,) if alive else ()


class RemoteKiller:
    """ExecKiller: ``kill`` as root in the sandbox, to the group (setsid)."""

    def __init__(self, client: RemoteClient) -> None:
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


class ChunkReader(io.RawIOBase):
    """A forward-only file over the chunks ``open_chunks`` returns.

    The chunks open on the first read, so a reader closed unread opens
    nothing; ``close`` closes them. A failure after the chunks open is an
    OSError (``clean`` keeps the key out), as a cut Docker stream is; one
    while opening them is the client's own error.
    """

    def __init__(
        self,
        open_chunks: Callable[[], Iterator[bytes]],
        clean: Callable[[BaseException], BaseException],
    ) -> None:
        self._open = open_chunks
        self._clean = clean
        self._chunks: Iterator[bytes] | None = None
        self._chunk = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        if self._chunks is None:
            self._chunks = self._open()
        while not self._chunk:
            try:
                self._chunk = memoryview(next(self._chunks))
            except StopIteration:
                return 0
            except Exception as error:
                raise OSError(str(self._clean(error))) from None
        count = min(len(buffer), len(self._chunk))
        buffer[:count] = self._chunk[:count]
        self._chunk = self._chunk[count:]
        return count

    def close(self) -> None:
        close = getattr(self._chunks, "close", None)
        if close is not None:
            close()
        super().close()


class RemoteTransfer:
    """copy_in/copy_out/path_stat as tar files in the sandbox (root)."""

    def __init__(self, client: RemoteClient, stages: StageStore) -> None:
        self._client = client
        self._stages = stages

    def _run(self, target: RemoteTarget, script: str, *args: str) -> tuple[int, bytes]:
        code, out, err = run_in(
            self._client, target.sandbox_id, ["sh", "-c", script, "sh", *args]
        )
        if code is None:
            raise SandboxError("unknown-outcome", "path", "sandbox command was cut off")
        if code:
            LOGGER.info(
                "%s transfer exited %s: %s", self._client.name, code, err[-512:]
            )
        return code, out

    def path_stat(self, target: RemoteTarget, path: str, *, follow: bool) -> dict:
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

    def copy_in(self, target: RemoteTarget, dest_dir: str, stage_id: str) -> dict:
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
        target: RemoteTarget,
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
            with (
                self._stages.create_result() as result,
                self._client.read_stream(target.sandbox_id, scratch) as source,
            ):
                result.summary = ArchiveTransfer._reemit(
                    source, result.file, max_bytes, patterns, ()
                )
        finally:
            try:
                run_in(self._client, target.sandbox_id, ["rm", "-f", scratch])
            except Exception as error:
                LOGGER.info("%s scratch tar not removed: %s", self._client.name, error)
        summary = result.summary
        return {
            "stage_id": result.stage_id,
            "bytes": summary.bytes,
            "entries": summary.entries,
            "skipped": summary.skipped,
        }


# -- composition and cleanup ----------------------------------------------------------


def remote_env_runtime(
    client: RemoteClient,
    *,
    spool_root: Path,
    host: Any,
    clock: Callable[[], float] = time.monotonic,
) -> EnvRuntime:
    """The remote composition beside docker_env_runtime: no Docker daemon,
    firewall, disk watchdog or builder (builds are the backend's template
    builds); stages, build inputs and exec output stay in the host spool,
    under its floor. ``host`` is the policy's [environments.host]."""
    settings = host.remote
    templates = RemoteTemplates(client, settings.template_prefix)
    backend = SandboxEnvRemoteBackend(
        client, templates, max_timeout_sec=settings.max_sandbox_hours * 3600
    )
    execs = RemoteExecApi(client, backend)
    backend.execs = execs
    table = RemoteProcessTable(execs, client.name)
    images = RemoteImages(templates, client)
    return EnvRuntime(
        backend=backend,
        images=images,
        transfer=lambda stages: RemoteTransfer(client, stages),
        pump=lambda on_finish: ExecPump(
            execs,
            Path(spool_root),
            killer=RemoteKiller(client),
            table=table,
            on_finish=on_finish,
            clock=clock,
            start_confirm_sec=START_CONFIRM_SEC,
        ),
        spool_root=Path(spool_root),
        table=table,
        spool_admit=spool_floor(Path(spool_root), host.disk_floor_mb),
        builds=None
        if client.build_form is None
        else lambda envs: RemoteBuilds(
            envs, client, templates, images, Path(spool_root)
        ),
        backend_name=client.name,
    )


def kill_run_sandboxes(
    client: RemoteClient,
    run_id: str,
    *,
    wait_sec: float = STOP_PROOF_SEC,
    poll: float = POLL_SEC,
) -> tuple[str, ...]:
    """Recovery: kill every sandbox (running or paused) carrying the run's
    metadata, and prove none is left. A listing may still show a killed
    one: it counts as gone once its state does, within ``wait_sec``. A
    client with a ledger lists the run's sandboxes from it too (the
    module's docstring). A kill that fails does not stop the others; every
    sandbox still there at the end is named. Templates stay (a shared
    cache)."""
    selector = run_metadata(run_id)

    def listed() -> list[str]:
        return [
            sandbox_id
            for sandbox_id, metadata in client.list(selector)
            if metadata.get(f"{META_PREFIX}run_id") == run_id
        ]

    killed = listed()
    failures: dict[str, Exception] = {}
    for sandbox_id in killed:
        try:
            client.kill(sandbox_id)
        except Exception as error:
            LOGGER.warning(
                "%s sandbox %s of run %s not killed: %s",
                client.name,
                sandbox_id,
                run_id,
                error,
            )
            failures[sandbox_id] = error
    end = time.monotonic() + wait_sec
    left = listed()
    while left:
        left = [
            sandbox_id for sandbox_id in left if client.state(sandbox_id) is not None
        ]
        if not left or time.monotonic() >= end:
            break
        time.sleep(poll)
    if left:
        why = "; ".join(
            f"{sandbox_id}: {failures[sandbox_id]}"
            for sandbox_id in left[:8]
            if sandbox_id in failures
        )
        raise InfrastructureError(
            f"{client.name} sandboxes of run {run_id} remain after kill: "
            f"{', '.join(left[:8])}" + (f" ({why})" if why else "")
        )
    # Every one is gone now, a failed kill's too (another way, or late).
    return tuple(killed)


__all__ = [
    "BACKENDS",
    "ChunkReader",
    "RemoteBackend",
    "RemoteBuilds",
    "RemoteClient",
    "RemoteExecApi",
    "RemoteImages",
    "RemoteProcess",
    "RemoteTemplates",
    "RemoteTransfer",
    "SandboxEnvRemoteBackend",
    "SignalFreeze",
    "bind_ledger",
    "built_template_name",
    "config_env",
    "group_argv",
    "grouped_argv",
    "kill_run_sandboxes",
    "read_registry_auth",
    "refuse_unsupported",
    "registry_image_config",
    "remote_backend",
    "remote_client",
    "remote_env_runtime",
    "run_in",
    "run_metadata",
    "sandbox_metadata",
    "start_group",
    "template_name",
]
