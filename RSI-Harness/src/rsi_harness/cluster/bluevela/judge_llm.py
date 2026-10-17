"""The Judge LLM (``judge_model``) on cluster runs.

The same ``rsi-agent llm`` as on local Docker (runtime/judge_llm.py): the
same credentials, launch spec, isolation, exit codes and redaction. What a
cluster changes is where things happen:

- The submit host (the adapter) checks the credential before anything is
  built or submitted, and finds the pinned Codex CLI when a Codex login
  needs it. It records only where things are, never a credential: the job
  resolves the credential again, from the environment the scheduler passes
  on (LSF's default, Slurm's ``--export=ALL``) and from shared storage.
- The Engine runs inside the job, on the compute node, and runs each Judge
  round there. It copies the pinned Codex CLI for the node's own
  architecture to node-local scratch, checking its SHA-256 again, and binds
  that copy read-only at ``/opt/rsi-agent-codex``.
- An operator's Codex login (``--judge-llm-auth codex-login``) is refreshed
  by the Engine, on the compute node, in place: only when the job sees the
  very file the submit host read (same path, inode, owner and filesystem
  type, as on a shared home) and can write it. Otherwise the run is refused
  before the Agent starts, since refreshing a copy, or a refresh that cannot
  be saved, would spend the refresh token and log the operator out.
  Refreshes are serialized across jobs and nodes by a lock file beside
  ``auth.json`` (:func:`cluster_login_refresh`).
- Network (:func:`cluster_judge_egress`): ``site`` gives the Judge the site
  network; ``pinned`` gives it a private network namespace whose only way
  out is the Harness's per-round proxy to the pinned Judge LLM endpoints,
  with no upstream proxy.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import platform
import re
import shutil
import stat
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from rsi_harness.cluster.bluevela.network import (
    JUDGE_PROVIDER_PORT,
    judge_proxy_environment,
)
from rsi_harness.errors import SetupError
from rsi_harness.models import JudgeLLMAuth, JudgeLLMRequest, PersistedModel
from rsi_harness.runtime.judge_llm import JudgeLLMAccess, resolve_judge_llm
from rsi_harness.runtime.judge_login_refresh import (
    MAX_REFRESH_SECONDS,
    LoginRefreshError,
    OperatorCodexLogin,
    host_codex_sha256,
    refresh_codex_login,
)
from rsi_harness.runtime.judge_model_access import JudgeEgress
from rsi_harness.runtime.local_auth import AgentAuthMaterial
from rsi_harness.runtime.network import PinnedEndpoint

# Node-local scratch below RSI_HARNESS_NODE_TMP holding the verified copy.
JUDGE_CODEX_SCRATCH = "judge-codex"
# Beside the operator's auth.json: one refresh at a time, across jobs and nodes.
LOGIN_LOCK = ".rsi-harness-login.lock"
# Another job's refresh, Codex's longest run, then some.
LOGIN_LOCK_TIMEOUT_SECONDS = MAX_REFRESH_SECONDS + 30
_LOCK_POLL_SECONDS = 0.5
_CHUNK = 1 << 20
_MACHINES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


class OperatorLoginFile(PersistedModel):
    """The operator's ``auth.json`` as the submit host saw it, never its
    content: its inode, owner and filesystem type."""

    inode: int
    uid: int
    filesystem: str


class ClusterJudgeLLM(PersistedModel):
    """What the job needs to resolve the Judge LLM again; never a credential.

    ``request`` is the operator's ``--judge-llm-*`` choice (None: the Work
    Agent lends its own), with a Codex login's directory fixed to the one the
    submit host read. ``codex`` holds the pinned Codex CLI found on the
    submit host, on shared storage, for each architecture found, when the
    credential is a Codex login. ``login`` identifies the operator's
    ``auth.json`` as the submit host saw it.
    """

    request: JudgeLLMRequest | None = None
    codex: dict[str, Path] = Field(default_factory=dict)
    login: OperatorLoginFile | None = None

    @field_validator("codex")
    @classmethod
    def _absolute(cls, value: dict[str, Path]) -> dict[str, Path]:
        if any(not path.is_absolute() for path in value.values()):
            raise ValueError("the pinned Codex CLI path must be absolute")
        return value


def cluster_judge_egress(task: Any) -> JudgeEgress:
    """How a cluster Judge with ``rsi-agent`` reaches its models.

    With ``model_access`` it is the Work Agent's network, as before: a
    ``"public"`` Agent's site network (``site``), or a ``"no-network"``
    Agent's pinned endpoints (``pinned``). A Judge with only a
    ``judge_model`` follows its own Verifier instead: ``"public"`` is the
    site network, ``"no-network"`` pins the Judge LLM's endpoints.
    """
    verifier = task.verifier
    if verifier.model_access:
        mode = task.agent.network.mode
        if mode not in ("public", "no-network"):
            raise SetupError(
                "Judge model access on a cluster needs a public or no-network Agent"
            )
    else:
        mode = verifier.network.mode
        if mode not in ("public", "no-network"):
            raise SetupError(
                "a Judge LLM on a cluster needs a public or no-network Verifier"
            )
    return "site" if mode == "public" else "pinned"


def judge_llm_for_egress(
    judge_llm: JudgeLLMAccess, egress: JudgeEgress
) -> JudgeLLMAccess:
    """``judge_llm`` as a cluster Judge with ``egress`` reaches it.

    ``pinned``: the Judge's only way out is the loopback proxy of its
    per-round provider broker, which connects to the pinned endpoints itself
    and cannot chain to an upstream proxy (as for a no-network Work Agent).
    """
    if egress == "site":
        return judge_llm
    spec = dict(judge_llm.spec)
    if judge_llm.kind is JudgeLLMAuth.API:
        if spec.get("proxy"):
            raise SetupError(_pinned_proxy_refusal(judge_llm.model))
        if urlsplit(str(spec["url"])).scheme != "https":
            raise SetupError(
                f"the Judge LLM {judge_llm.model!r} through pinned endpoints on a "
                "cluster needs an https API base URL"
            )
        spec["proxy"] = f"http://127.0.0.1:{JUDGE_PROVIDER_PORT}"
    else:
        if dict(spec.get("environment") or {}):
            raise SetupError(_pinned_proxy_refusal(judge_llm.model))
        spec["environment"] = judge_proxy_environment()
    return replace(judge_llm, spec=spec)


def _pinned_proxy_refusal(model: str) -> str:
    return (
        f"the Judge LLM {model!r} would use an upstream proxy, but a cluster "
        'Judge with [verifier] network_mode = "no-network" reaches it only '
        "through pinned endpoints, without one; drop --judge-llm-proxy and the "
        "run's proxy settings, or use a public Verifier"
    )


def preflight_cluster_judge_llm(
    definition: Any,
    request: JudgeLLMRequest | None,
    *,
    agent_auth: AgentAuthMaterial | None,
    config: Any,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
    digests: Mapping[str, str] | None = None,
) -> tuple[ClusterJudgeLLM, JudgeLLMAccess] | None:
    """On the submit host, before anything is built or submitted.

    Resolves the Judge LLM as the job will, with ``agent_auth`` the Work
    Agent's login as the job reads it (refusing a missing credential, the
    Work Agent's own login as the operator's, or an upstream proxy a pinned
    Judge cannot use), and finds the pinned Codex CLI a Codex login needs.
    Returns None for a task without ``judge_model``.
    """
    if definition.verifier.judge_model is None:
        return None
    if request is not None:
        _refuse_userinfo(request)
    task = cluster_judge_task(definition)
    judge_llm = resolve_judge_llm(
        task, request, config, agent_auth, pin=unresolved_endpoints, environ=environ
    )
    judge_llm_for_egress(judge_llm, cluster_judge_egress(task))
    login = None
    if judge_llm.operator_login is not None:
        assert request is not None
        directory = judge_llm.operator_login.codex_home
        path = directory / "auth.json"
        try:
            found = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise SetupError(
                f"cannot read the Codex login in {directory}: {error.strerror}"
            ) from None
        login = OperatorLoginFile(
            inode=found.st_ino, uid=found.st_uid, filesystem=filesystem_type(path)
        )
        request = request.model_copy(update={"codex_home": directory})
    codex = (
        locate_pinned_codex(which or shutil.which, digests)
        if judge_llm.needs_codex_cli
        else {}
    )
    return ClusterJudgeLLM(request=request, codex=codex, login=login), judge_llm


def _refuse_userinfo(request: JudgeLLMRequest) -> None:
    """The run directory keeps the request: no credentials in its URLs."""
    if request.proxy and _has_userinfo(request.proxy):
        raise SetupError(
            "a cluster run keeps --judge-llm-proxy in its run directory: give "
            "proxy credentials through the run's proxy environment instead"
        )
    if request.base_url and _has_userinfo(request.base_url):
        raise SetupError(
            "a cluster run keeps --judge-llm-base-url in its run directory: "
            "give the API key through --judge-llm-api-key-env or "
            "--judge-llm-api-key-file instead of the URL"
        )


def resolve_cluster_judge_llm(
    task: Any,
    prepared: ClusterJudgeLLM | None,
    *,
    agent_auth: Any,
    config: Any,
    egress: JudgeEgress,
    pin: Callable[[tuple[str, ...]], tuple[PinnedEndpoint, ...]],
    environ: Mapping[str, str] | None = None,
) -> JudgeLLMAccess | None:
    """In the job: the Judge LLM again, as the submit host checked it."""
    if task.verifier.judge_model is None:
        return None
    if prepared is None:
        raise SetupError("the Judge LLM was not prepared on the submit host")
    request = prepared.request
    if request is not None and request.auth is JudgeLLMAuth.CODEX_LOGIN:
        require_login_in_place(request, prepared.login)
    judge_llm = resolve_judge_llm(
        cluster_judge_task(task),
        request,
        config,
        agent_auth,
        pin=pin,
        environ=environ,
    )
    return judge_llm_for_egress(judge_llm, egress)


def require_login_in_place(
    request: JudgeLLMRequest, recorded: OperatorLoginFile | None
) -> None:
    """The operator's ``auth.json`` here must be the file the submit host
    read, and writable.

    The Engine refreshes it in place before a round would outlive its access
    token. A different file at that path (a node-local home, or no shared
    home at all) would be a copy: refreshing it spends the refresh token the
    operator's own login still holds, which logs the operator out. So does a
    refresh Codex cannot save (a home read-only on compute nodes). Inode
    numbers are per filesystem, so the owner and filesystem type must match
    too.
    """
    if request.codex_home is None or recorded is None:
        raise SetupError("the operator's Codex login was not located on submit")
    path = request.codex_home / "auth.json"
    try:
        found = os.stat(path, follow_symlinks=False)
    except OSError as error:
        problem: str | None = f"it is not there ({error.strerror})"
    else:
        if found.st_uid != recorded.uid or filesystem_type(path) != (
            recorded.filesystem
        ):
            problem = "this compute node has another file there"
        elif found.st_ino != recorded.inode:
            problem = (
                "it is not the file the submit host read: a new codex login "
                "since submission (submit again), or this node's own copy"
            )
        else:
            problem = login_unwritable(path)
    if problem is not None:
        raise SetupError(
            f"the operator's Codex login {path} cannot be refreshed in place "
            f"on this compute node: {problem}. The Judge LLM's login is "
            "refreshed where the Engine runs, and refreshing a copy, or a "
            "refresh that cannot be saved, would log the operator out; use "
            "--judge-llm-auth api, or leave out --judge-llm-auth so a Codex "
            "Work Agent with a ChatGPT login lends its own"
        )


def login_unwritable(path: Path) -> str | None:
    """Why the Engine could not save a refreshed ``auth.json`` (None: it
    can). Its directory must be writable too, for the refresh lock."""
    if not os.access(path, os.W_OK):
        return "it is not writable here"
    if not os.access(path.parent, os.W_OK):
        return f"{path.parent} is not writable here"
    try:
        flags = os.statvfs(path.parent).f_flag
    except OSError as error:
        return f"its filesystem cannot be checked ({error.strerror})"
    if flags & os.ST_RDONLY:
        return "its filesystem is read-only here"
    return None


def cluster_login_refresh(
    login: OperatorCodexLogin,
    codex_binary: Path | None,
    *,
    timeout: float,
    saved: Callable[[], bool],
    run: Callable[..., str | None] | None = None,
    lock_timeout: float = LOGIN_LOCK_TIMEOUT_SECONDS,
) -> str | None:
    """The refresh ``renew_operator_login`` runs on a cluster.

    Many jobs on many nodes may use one operator login on a shared home, and
    two refreshes presenting the same refresh token can log the operator out.
    So a refresh first checks the login is still writable here, then takes
    an exclusive POSIX lock on ``LOGIN_LOCK`` beside ``auth.json`` and, holding
    it, refreshes only if no other job has meanwhile (``saved()``). Where the
    filesystem has no locks, the refresh runs unserialized, as on Docker.
    """
    problem = login_unwritable(login.codex_home / "auth.json")
    if problem is not None:
        raise LoginRefreshError(f"{problem}, so this node does not refresh it")
    with _login_lock(login.codex_home, lock_timeout):
        if saved():  # another job refreshed it while this one waited
            return None
        refresh = refresh_codex_login if run is None else run
        return refresh(login, codex_binary, timeout=timeout, saved=saved)


@contextmanager
def _login_lock(
    codex_home: Path,
    timeout: float,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[None]:
    try:
        descriptor = os.open(
            codex_home / LOGIN_LOCK,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except OSError as error:
        raise LoginRefreshError(
            f"its refresh lock cannot be opened: {error.strerror}"
        ) from None
    try:
        deadline = clock() + timeout
        while True:
            try:
                fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    break  # no locks on this filesystem (ENOLCK, ...)
                if clock() >= deadline:
                    raise LoginRefreshError(
                        "another run is still refreshing it"
                    ) from None
                sleep(_LOCK_POLL_SECONDS)
            else:
                break
        yield
    finally:
        os.close(descriptor)  # releases the lock


def filesystem_type(path: Path, mountinfo: Path = Path("/proc/self/mountinfo")) -> str:
    """The type of the filesystem ``path`` is on (``unknown`` without
    ``/proc``): the last mount at the longest mount point above it."""
    target = os.path.realpath(path)
    try:
        lines = mountinfo.read_text().splitlines()
    except OSError:
        return "unknown"
    best, kind = "", "unknown"
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 5 or separator + 1 >= len(fields):
            continue
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4])
        inside = (
            point == "/"
            or target == point
            or target.startswith(point.rstrip("/") + "/")
        )
        if inside and len(point) >= len(best):
            best, kind = point, fields[separator + 1]
    return kind


def cluster_judge_task(definition: Any) -> Any:
    """Every cluster Judge starts from the task image (adapter._run_plan)."""
    verifier = definition.verifier
    if getattr(verifier, "clean_image", True):
        return definition
    return definition.model_copy(
        update={"verifier": verifier.model_copy(update={"clean_image": True})}
    )


def pinned_codex_digests() -> dict[str, str]:
    """The pinned Codex release's SHA-256 for each architecture: digest to
    machine."""
    from rsi_loop.harness.agent import versions

    return {
        versions.CODEX_SHA256_X86_64: "x86_64",
        versions.CODEX_SHA256_AARCH64: "aarch64",
    }


def locate_pinned_codex(
    which: Callable[[str], str | None] = shutil.which,
    digests: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """The pinned Codex CLI's native binaries in the submit host's install,
    by architecture.

    Cluster runs already take the Agent CLI from the submit host's ``PATH``
    (on shared storage, so compute nodes can read it). The Judge LLM's Codex
    CLI comes from there too: ``codex`` itself when it is the native binary,
    or the ones npm installed beside it
    (``@openai/**/vendor/<machine>-unknown-linux-musl/bin/codex``), each used
    only when its SHA-256 is the pinned release's for some architecture. The
    job picks the one for its own node (:func:`node_codex_source`).
    """
    from rsi_loop.harness.agent import versions

    pins = pinned_codex_digests() if digests is None else digests
    found: dict[str, Path] = {}
    executable = which("codex")
    if executable is not None:
        for candidate in _codex_candidates(Path(executable).resolve()):
            try:
                machine = pins.get(_sha256(candidate)) if candidate.is_file() else None
            except OSError:
                continue
            if machine is not None:
                found.setdefault(machine, candidate)
    if found:
        return found
    raise SetupError(
        "the Judge LLM's Codex login needs the pinned Codex CLI "
        f"{versions.CODEX} on shared storage, but `codex` on this host's PATH "
        "is not that release: install it (npm install -g "
        f"@openai/codex@{versions.CODEX}), or use --judge-llm-auth api"
    )


def _codex_candidates(resolved: Path) -> list[Path]:
    candidates = [resolved]
    for parent in resolved.parents:
        if parent.name == "node_modules":
            candidates += sorted(
                (parent / "@openai").glob("**/vendor/*-unknown-linux-musl/bin/codex")
            )
            break
    return candidates


def node_codex_source(sources: Mapping[str, Path]) -> Path:
    """The submit host's pinned Codex CLI for this compute node's
    architecture."""
    from rsi_loop.harness.agent import versions

    machine = _MACHINES.get(platform.machine().lower())
    source = None if machine is None else sources.get(machine)
    if source is None:
        found = ", ".join(sorted(sources)) or "none"
        raise SetupError(
            f"the submit host found the pinned Codex CLI {versions.CODEX} for "
            f"{found}, not for this compute node ({platform.machine()}): make "
            "that architecture's release available beside the submit host's "
            "`codex` install, or use --judge-llm-auth api"
        )
    return source


def install_pinned_codex(
    source: Path, directory: Path, *, expected: str | None = None
) -> Path:
    """Copy ``source`` to ``directory/codex`` if it is this node's release.

    The bytes copied are the bytes checked: the Judge binds the copy,
    read-only, and the login refresh checks it again before running it.
    Returns ``directory``.
    """
    import platform

    digest_expected = host_codex_sha256() if expected is None else expected
    if digest_expected is None:
        raise SetupError(
            "the pinned Codex CLI has no release for this compute node "
            f"({platform.machine()})"
        )
    directory.mkdir(mode=0o755)
    target = directory / "codex"
    try:
        descriptor = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise SetupError(
            f"the pinned Codex CLI {source} is not readable on this compute "
            f"node: {error.strerror}"
        ) from None
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise SetupError(f"the pinned Codex CLI {source} is not a file")
        output = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o700)
        with os.fdopen(output, "wb") as stream:
            while chunk := os.read(descriptor, _CHUNK):
                digest.update(chunk)
                stream.write(chunk)
    finally:
        os.close(descriptor)
    if digest.hexdigest() != digest_expected:
        target.unlink()
        raise SetupError(
            f"the Codex CLI {source} is not the pinned release for this compute "
            f"node ({platform.machine()}); install that release where `codex` "
            "resolves on the submit host"
        )
    target.chmod(0o755)
    return directory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def unresolved_endpoints(targets: tuple[str, ...]) -> tuple[PinnedEndpoint, ...]:
    """Endpoints named, not resolved: for checks before the job, and for a
    site-egress Judge, which resolves names itself."""
    pinned = []
    for url in targets:
        parsed = urlsplit(url)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        pinned.append(
            PinnedEndpoint(hostname=parsed.hostname or "", port=port, addresses=())
        )
    return tuple(pinned)


def _has_userinfo(url: str) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return bool(parsed.username or parsed.password)


__all__ = [
    "JUDGE_CODEX_SCRATCH",
    "LOGIN_LOCK",
    "LOGIN_LOCK_TIMEOUT_SECONDS",
    "ClusterJudgeLLM",
    "OperatorLoginFile",
    "cluster_judge_egress",
    "cluster_judge_task",
    "cluster_login_refresh",
    "filesystem_type",
    "install_pinned_codex",
    "judge_llm_for_egress",
    "locate_pinned_codex",
    "login_unwritable",
    "node_codex_source",
    "pinned_codex_digests",
    "preflight_cluster_judge_llm",
    "require_login_in_place",
    "resolve_cluster_judge_llm",
    "unresolved_endpoints",
]
