"""Harbor environment over brokered sandbox environments (protocol v2).

Code in Work or Judge runs stock Harbor with this plugin::

    PYTHONPATH=$RSI_SANDBOX_PYTHONPATH harbor run \\
        --env rsi_sandbox_harbor:ManagedSandboxEnvironment ...

Every Docker action happens in the host broker behind the phase socket
(``RSI_SANDBOX_SOCKET``/``RSI_SANDBOX_TOKEN``); this process holds no Docker
authority. A task's image, Dockerfile or compose project becomes one broker
env (spec 6): a single image is an env with the one service ``main``, the
Harbor name for the agent's service. Compose is translated here, on the
caller side, by the shared front-end; the broker validates only the EnvSpec.

The harness injects this file into every endpoint as
``py/rsi_sandbox_harbor.py`` next to its stdlib client and compose front-end,
so it imports only the standard library, Harbor and those two siblings.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import os
import posixpath
import re
import signal as signals
import stat
import tarfile
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, override

import harbor
from harbor.constants import MAIN_SERVICE_NAME
from harbor.environments.base import (
    BaseEnvironment,
    ExecResult,
)
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.compose_service_ops import ComposeServiceOpsMixin
from harbor.environments.definition import (
    COMPOSE_FILE_NAME,
    require_agent_environment_definition,
    should_use_prebuilt_docker_image,
)
from harbor.models.task.config import NetworkMode, TaskOS
from harbor.models.trial.config import ResourceMode
from harbor.utils.env import resolve_env_vars
from harbor.utils.path_filter import filter_paths_by_patterns

try:  # The endpoint's injected copies (PYTHONPATH=$RSI_SANDBOX_PYTHONPATH).
    import rsi_sandbox_client as wire
    import rsi_sandbox_compose as compose
except ImportError:  # The installed harness package.
    from rsi_harness.integrations import sandbox_client as wire
    from rsi_harness.integrations import sandbox_compose as compose

HARBOR_SERIES = "0.21."
# Harbor internals the plugin reads; a Harbor without them is refused.
_HARBOR_PRIVATE = (
    "_network_policy",
    "_phase_network_policies",
    "_mounts",
    "_persistent_env",
)
LOG_TARGETS = ("/logs/agent", "/logs/verifier", "/logs/artifacts")
# Live quotas free up when another env ends: env_create retries these.
_LIVE_QUOTAS = frozenset(
    {
        "max_envs_live",
        "max_containers_live",
        "max_cpus_live",
        "max_memory_mb_live",
        "max_swap_mb_live",
        "max_disk_mb_live",
        "max_jobs_running",
    }
)
_LIVE_RETRY_SEC = 2.0
_BUSY_RETRY_SEC = 0.05
_WAIT_SEC = 25
# The start wait is measured here, the broker checks it on arrival: this much
# of the env's lifetime is left for the request's way to the broker.
_START_SLACK_SEC = compose.START_SLACK_SEC
_KILL_GRACE_SEC = 2.0
# Destroy proves every object removed, up to the broker's 60 s drain.
_LIFECYCLE_WAIT_SEC = 90.0
_DEFAULT_DISK_MB = 4096
_SIDECAR_CPUS = 1.0
_SIDECAR_MEMORY_MB = 1024
_MAX_COPY_BYTES = 1024**3
_BASH = ("/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash")


class ManagedSandboxError(RuntimeError):
    """A terminal brokered-environment failure (no retry helps)."""


class SandboxUnknownOutcomeError(ManagedSandboxError):
    """The broker cannot prove whether a mutating operation completed."""


def _check_harbor_version() -> None:
    version = getattr(harbor, "__version__", "")
    if not str(version).startswith(HARBOR_SERIES):
        raise RuntimeError(
            f"rsi_sandbox_harbor supports harbor {HARBOR_SERIES}x, not {version!r}"
        )


def _absolute(path: str | PurePosixPath, label: str) -> str:
    text = str(path)
    if not text.startswith("/"):
        raise ValueError(f"{label} must be an absolute container path: {text!r}")
    return posixpath.normpath(text).replace("//", "/")


def _tar_file(source: Path, name: str, target) -> None:
    info = os.stat(source)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{source} is not a regular file")
    with tarfile.open(fileobj=target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        entry = tarfile.TarInfo(name)
        entry.size = info.st_size
        entry.mode = stat.S_IMODE(info.st_mode)
        entry.mtime = int(info.st_mtime)
        with open(source, "rb") as data:
            archive.addfile(entry, data)


def _tar_tree(source: Path, target) -> None:
    with tarfile.open(fileobj=target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        compose.add_tree(archive, str(source), "")


def _tar_dirs(names: Sequence[str], target) -> None:
    with tarfile.open(fileobj=target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in names:
            entry = tarfile.TarInfo(name)
            entry.type = tarfile.DIRTYPE
            entry.mode = 0o777
            archive.addfile(entry)


def _members(archive: tarfile.TarFile) -> list[tuple[tarfile.TarInfo, str]]:
    """copy_out archives are rooted at the path's basename: strip that root."""
    result = []
    for member in archive.getmembers():
        parts = PurePosixPath(member.name).parts
        if len(parts) > 1:
            result.append((member, "/".join(parts[1:])))
    return result


def _extract(archive: tarfile.TarFile, target: Path, selected=None) -> None:
    target.mkdir(parents=True, exist_ok=True)
    members = []
    for member, relative in _members(archive):
        if selected is not None and relative not in selected:
            continue
        members.append(member.replace(name=relative, deep=False))
    archive.extractall(path=target, members=members, filter="data")


class _ServiceTransport:
    """Harbor's per-service compose surface over the one broker env."""

    def __init__(self, environment: ManagedSandboxEnvironment) -> None:
        self._environment = environment

    async def service_exec(
        self,
        command: str,
        *,
        service: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        # Sidecars are arbitrary images: POSIX sh, and no main defaults.
        return await self._environment._exec(
            service,
            ["sh", "-c", command],
            cwd=cwd,
            env=env,
            user=user,
            timeout_sec=timeout_sec,
        )

    async def service_download_file(
        self, source_path: str, target_path: Path | str, *, service: str
    ) -> None:
        await self._environment._download_file(service, source_path, target_path)

    async def service_download_dir(
        self, source_dir: str, target_dir: Path | str, *, service: str
    ) -> None:
        await self._environment._download_dir(service, source_dir, target_dir)

    async def stop_service(self, service: str) -> None:
        await self._environment._stop_service(service)


class ManagedSandboxEnvironment(ComposeServiceOpsMixin, BaseEnvironment):
    """Harbor's environment API on one brokered env of the caller's phase.

    Keyword arguments (all optional, ``--ek`` on the Harbor CLI):
    ``socket_path`` and ``credential`` (default: the endpoint environment),
    ``lifetime_sec`` (env lifetime; default the grant's), ``on_cancel``
    (``kill-group``: a cancelled exec's process group is killed;
    ``detach``: left running until the env ends, as stock Harbor does),
    ``pull_policy`` (``missing`` or ``always``) and ``inject_tmux``
    (``auto``: when the grant offers the operator's static tmux and main
    has none on PATH, the broker copies it in before Harbor's agent setup,
    so terminus-2 works without network; ``off``: never).
    """

    def __init__(
        self,
        *args: Any,
        socket_path: str | Path | None = None,
        credential: str | None = None,
        lifetime_sec: float | None = None,
        on_cancel: str = "kill-group",
        pull_policy: str = "missing",
        inject_tmux: str = "auto",
        client: Any = None,
        **kwargs: Any,
    ) -> None:
        _check_harbor_version()
        if on_cancel not in ("kill-group", "detach"):
            raise ValueError("on_cancel must be kill-group or detach")
        if pull_policy not in ("missing", "always"):
            raise ValueError("pull_policy must be missing or always")
        if inject_tmux not in ("auto", "off"):
            raise ValueError("inject_tmux must be auto or off")
        # BaseEnvironment validates through capabilities and
        # _validate_definition, so the grant must be known before super().
        self._client = client or wire.SandboxClient(
            None if socket_path is None else str(socket_path), credential
        )
        answer = self._client.capabilities()
        if 2 not in wire._versions(answer):
            raise ValueError(
                "this sandbox phase grants no brokered environments (protocol v2)"
            )
        self._grant: dict[str, Any] = answer["environments"]
        self._limits: dict[str, Any] = self._grant["limits"]
        if lifetime_sec is not None and not (
            0 < float(lifetime_sec) <= self._limits["max_env_lifetime_sec"]
        ):
            raise ValueError("lifetime_sec exceeds the granted max_env_lifetime_sec")
        self._lifetime_sec = None if lifetime_sec is None else float(lifetime_sec)
        self._on_cancel = on_cancel
        self._pull_policy = pull_policy
        self._inject_tmux = inject_tmux
        self._translation: Any = None
        self._env_id: str | None = None
        self._services: tuple[str, ...] = ()
        self._main_shell = "bash"
        # (spec, request_id) of an env_create whose reply may be lost.
        self._create: tuple[dict, str] | None = None
        # request_id -> replay of an image job start whose reply may be lost.
        self._job_requests: dict[str, Any] = {}
        self._jobs: set[str] = set()
        self._built: set[str] = set()
        self._startup_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._kills: set[asyncio.Task[None]] = set()
        self._stop_requested = False
        self._delete = False
        super().__init__(*args, **kwargs)

    # -- Harbor contract ------------------------------------------------------------

    @staticmethod
    @override
    def type() -> str:
        return "rsi-managed-sandbox"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        allowlist = "allowlist" in self._grant["network"]
        # The broker resolves exact hostnames; it cannot enforce wildcards,
        # and env bridges have no IPv6.
        return EnvironmentCapabilities(
            docker_compose=True,
            mounted=False,
            disable_internet="none" in self._grant["network"],
            network_allowlist=allowlist,
            network_allowlist_hostnames=allowlist,
            network_allowlist_ipv4_addresses=allowlist,
            network_allowlist_ipv4_cidrs=allowlist,
        )

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    @classmethod
    @override
    def preflight(cls) -> None:
        missing = [
            name
            for name in ("RSI_SANDBOX_SOCKET", "RSI_SANDBOX_TOKEN")
            if not os.environ.get(name)
        ]
        if missing:
            raise SystemExit(
                "rsi_sandbox_harbor needs a sandbox endpoint; unset: "
                + ", ".join(missing)
            )

    @property
    @override
    def _uses_compose(self) -> bool:
        return (self.environment_dir / COMPOSE_FILE_NAME).exists() or bool(
            self.extra_docker_compose_paths
        )

    def _env_network(self, policy: Any) -> str:
        mode = policy.network_mode
        if mode == NetworkMode.PUBLIC:
            network = "public"
        elif mode == NetworkMode.NO_NETWORK:
            network = "none"
        elif mode == NetworkMode.ALLOWLIST:
            network = "allowlist"
        else:
            raise ValueError(
                f"network_mode={mode.value!r} is unsupported by managed sandboxes"
            )
        if network not in self._grant["network"]:
            raise ValueError(
                f"network_mode={mode.value!r} needs env network {network!r}, "
                "which the sandbox grant does not include"
            )
        if network == "allowlist":
            bounds = self._grant["allowlist"]
            if len(policy.allowed_hosts) > bounds["max_entries"]:
                raise ValueError(
                    f"network allowlist has {len(policy.allowed_hosts)} entries; "
                    f"the sandbox grant allows {bounds['max_entries']}"
                )
        return network

    @property
    def _disk_mb(self) -> int:
        storage = self.task_env_config.storage_mb
        if storage is not None:
            return storage
        return min(
            _DEFAULT_DISK_MB,
            self._limits["disk_mb_per_container"],
            self._limits["max_disk_mb_live"],
        )

    @override
    def _validate_definition(self) -> None:
        missing = [name for name in _HARBOR_PRIVATE if not hasattr(self, name)]
        if missing:
            raise RuntimeError(
                f"harbor {harbor.__version__} lacks {', '.join(missing)}; "
                f"rsi_sandbox_harbor supports harbor {HARBOR_SERIES}x"
            )
        require_agent_environment_definition(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            extra_docker_compose_paths=self.extra_docker_compose_paths,
        )
        if self.os != TaskOS.LINUX:
            raise ValueError("managed sandboxes run Linux containers only")
        # Before Harbor's own check, whose advice (another environment type)
        # does not apply here.
        if self.task_env_config.gpus:
            raise ValueError(
                f"environment gpus={self.task_env_config.gpus}: managed sandbox "
                "environments have no GPUs; use the GPUs of the Work or Judge "
                "container itself"
            )
        for policy in (self._network_policy, *self._phase_network_policies):
            self._env_network(policy)
        config = self.task_env_config
        for label, value, cap in (
            ("cpus", config.cpus, self._limits["cpus_per_container"]),
            ("memory_mb", config.memory_mb, self._limits["memory_mb_per_container"]),
            ("storage_mb", config.storage_mb, self._limits["disk_mb_per_container"]),
            ("storage_mb", config.storage_mb, self._limits["max_disk_mb_live"]),
        ):
            if value is not None and value > cap:
                raise ValueError(
                    f"environment {label}={value} exceeds the sandbox grant "
                    f"({cap}); limits are never clamped"
                )
        for mount in self._mounts:
            if mount.get("target") not in LOG_TARGETS:
                raise ValueError(
                    "managed sandboxes accept only Harbor's log mounts "
                    f"{', '.join(LOG_TARGETS)}; got {mount.get('target')!r}"
                )
        # Translate eagerly: an unsupported compose key fails before start.
        self._translation = self._translate(force_build=False)

    # -- translation -------------------------------------------------------------------

    def _interpolation(self, use_prebuilt: bool) -> dict[str, str]:
        """Harbor's compose variables (docker.py _compose_env_vars): the
        process environment, task/persistent env, then the infra variables.
        The endpoint's own RSI_SANDBOX_* values are never offered."""
        environ = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("RSI_SANDBOX_")
        }
        if self._uses_compose and self.task_env_config.env:
            environ.update(resolve_env_vars(self.task_env_config.env))
        environ.update(self._persistent_env)
        environ["CONTEXT_DIR"] = str(self.environment_dir.resolve())
        environ["MAIN_IMAGE_NAME"] = re.sub(
            r"[^a-z0-9._-]", "-", f"hb__{self.environment_id}".lower()
        )
        if use_prebuilt and self.task_env_config.docker_image:
            environ["PREBUILT_IMAGE_NAME"] = self.task_env_config.docker_image
        if self._effective_cpus is not None:
            environ["CPUS"] = str(self._effective_cpus)
        if self._effective_memory_mb is not None:
            environ["MEMORY"] = f"{self._effective_memory_mb}M"
        suffixes = {"agent": "AGENT_LOGS", "verifier": "VERIFIER_LOGS"}
        suffixes["artifacts"] = "ARTIFACTS"
        for mount in self._mounts:
            target = str(mount.get("target", ""))
            name = suffixes.get(target.rstrip("/").rsplit("/", 1)[-1])
            if name is not None:
                # The host side is never mounted: both name the log target,
                # which the front-end drops (logs are downloaded).
                environ[f"ENV_{name}_PATH"] = environ[f"HOST_{name}_PATH"] = target
        return environ

    def _translate(self, force_build: bool) -> Any:
        """Harbor's compose layer order (docker.py _docker_compose_paths):
        resources, prebuilt or build, the task file, extra files, main env.
        The mounts layer has no equivalent: logs are downloaded."""
        use_prebuilt = should_use_prebuilt_docker_image(
            self.environment_dir,
            docker_image=self.task_env_config.docker_image,
            force_build=force_build,
        )
        resources: dict[str, Any] = {}
        cpus = self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT)
        memory = self._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT)
        if cpus is not None:
            resources["cpus"] = float(cpus)
        if memory is not None:
            resources["mem_limit"] = f"{memory}m"
        keepalive = ["sh", "-c", "sleep infinity"]
        image_layer = (
            {"image": "${PREBUILT_IMAGE_NAME}", "command": keepalive}
            if use_prebuilt
            else {
                "build": {"context": "${CONTEXT_DIR}"},
                "pull_policy": "build",
                "command": keepalive,
            }
        )
        layers: list[Any] = [
            {"services": {MAIN_SERVICE_NAME: resources}},
            {"services": {MAIN_SERVICE_NAME: image_layer}},
        ]
        if (self.environment_dir / COMPOSE_FILE_NAME).exists():
            layers.append(self.environment_dir / COMPOSE_FILE_NAME)
        layers.extend(self.extra_docker_compose_paths)
        layers.append(
            {"services": {MAIN_SERVICE_NAME: {"environment": self._startup_env()}}}
        )
        try:
            project = compose.load_project(
                layers,
                environ=self._interpolation(use_prebuilt),
                project_dir=self.environment_dir,
            )
            translation = compose.translate(
                project,
                project_dir=self.environment_dir,
                network=self._env_network(self._network_policy),
                allowlist=list(self._network_policy.allowed_hosts),
                disk_mb=self._disk_mb,
                lifetime_sec=self._lifetime_sec,
                default_cpus=min(_SIDECAR_CPUS, self._limits["cpus_per_container"]),
                default_memory_mb=min(
                    _SIDECAR_MEMORY_MB, self._limits["memory_mb_per_container"]
                ),
                drop_targets=LOG_TARGETS,
            )
        except compose.ComposeError as error:
            raise ValueError(f"environment compose: {error}") from None
        self._check_grant(translation.spec)
        return translation

    def _check_grant(self, spec: dict) -> None:
        """What no amount of waiting can admit fails now, never clamped."""
        limits = self._limits
        services = spec["services"]
        if len(services) > limits["max_services_per_env"]:
            raise ValueError(
                f"environment has {len(services)} services; the sandbox grant "
                f"allows {limits['max_services_per_env']} per env"
            )
        for name, service in services.items():
            for label, value, cap in (
                ("cpus", service["cpus"], limits["cpus_per_container"]),
                ("memory_mb", service["memory_mb"], limits["memory_mb_per_container"]),
            ):
                if value > cap:
                    raise ValueError(
                        f"service {name} {label}={value} exceeds the sandbox "
                        f"grant ({cap}); limits are never clamped"
                    )
        for label, value, cap in (
            ("cpus", sum(item["cpus"] for item in services.values()), "max_cpus_live"),
            (
                "memory_mb",
                sum(item["memory_mb"] for item in services.values()),
                "max_memory_mb_live",
            ),
            ("containers", len(services), "max_containers_live"),
            ("disk_mb", spec["disk_mb"], "max_disk_mb_live"),
        ):
            if value > limits[cap]:
                raise ValueError(
                    f"environment needs {label}={value}; the sandbox grant "
                    f"allows {limits[cap]} live ({cap})"
                )

    # -- lifecycle --------------------------------------------------------------------

    @staticmethod
    def _owned_task(coroutine) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        # A cancelled caller may never await the task; still retrieve errors.
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        return task

    @staticmethod
    async def _call(function, *args, **kwargs):
        return await asyncio.to_thread(function, *args, **kwargs)

    def _require_env(self) -> str:
        if self._stop_requested:
            raise RuntimeError("managed sandbox is stopping or stopped")
        if self._env_id is None:
            raise RuntimeError("managed sandbox is not started")
        return self._env_id

    @override
    async def start(self, force_build: bool) -> None:
        if (
            self._env_id is not None
            or self._create is not None
            or (self._startup_task is not None and not self._startup_task.done())
            or (self._cleanup_task is not None and not self._cleanup_task.done())
        ):
            raise RuntimeError("managed sandbox is already started or stopping")
        self._stop_requested = False
        self._delete = False
        self._cleanup_task = None
        self._startup_task = self._owned_task(self._start_owned(force_build))
        try:
            await asyncio.shield(self._startup_task)
            if self._stop_requested:
                raise ManagedSandboxError("managed sandbox stopped during startup")
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if (
                self._stop_requested
                and current is not None
                and not current.cancelling()
            ):
                # A concurrent stop() cancelled the startup, not our caller:
                # never hand Harbor a cancellation it did not request.
                raise ManagedSandboxError(
                    "managed sandbox stopped during startup"
                ) from None
            # Harbor's start timeout: cleanup runs on; stop() awaits it.
            self._request_stop()
            raise
        except BaseException:
            await self._wait_cleanup(self._request_stop())
            raise

    async def _start_owned(self, force_build: bool) -> None:
        translation = (
            self._translation if not force_build else self._translate(force_build)
        )
        pulled: dict[tuple[str, str], str] = {}
        handles = {}
        for name in translation.services:
            request = translation.images[name]
            if isinstance(request, compose.ImagePull):
                policy = "always" if self._pull_policy == "always" else request.policy
                key = (request.ref, policy)
                if key not in pulled:
                    pulled[key] = await self._pull(request.ref, policy)
                handles[name] = pulled[key]
            else:
                handles[name] = await self._build(name, request)
        for note in translation.notes:
            self.logger.info("managed sandbox: %s", note)
        expires_at = await self._create_env(translation.env_spec(handles))
        self._services = translation.services
        for seed in translation.seeds:
            await self._copy_seed(seed)
        wait = await self._start_env(expires_at)
        status = await self._wait_ready(wait + _WAIT_SEC)
        if status["state"] != "ready":
            raise ManagedSandboxError(self._describe_failure(status))
        self._main_shell = await self._pick_shell()
        await self._ensure_log_dirs()
        if self._inject_tmux == "auto" and "tmux" in (self._grant.get("tools") or ()):
            await self._ensure_tmux()
        await self._upload_environment_dir_after_start()

    async def _follow_job(self, job_id: str, label: str) -> str:
        offset = 0
        while True:
            view = await self._call(self._client.job_wait, job_id, offset, _WAIT_SEC)
            if view["log"]:
                self.logger.debug("%s: %s", label, view["log"].rstrip())
            offset = view["next_offset"]
            if view["state"] in ("queued", "running") or view["log"]:
                continue
            self._jobs.discard(job_id)
            if view["state"] != "succeeded":
                error = view.get("error") or {}
                raise ManagedSandboxError(
                    f"{label} {view['state']}: {error.get('kind', 'unknown')}: "
                    f"{error.get('message', '')}".rstrip(": ")
                )
            return view["result"]["image"]["handle"]

    async def _start_job(self, start, label: str) -> str:
        request_id = uuid.uuid4().hex
        self._job_requests[request_id] = lambda: start(request_id)
        try:
            job_id = await self._call(start, request_id)
        except wire.ProtocolError as error:
            if error.code != "unknown-outcome":
                # Refused: no job exists. A lost reply stays for cleanup's
                # replay, which learns the job and cancels it.
                self._job_requests.pop(request_id, None)
            raise ManagedSandboxError(f"{label}: {error.code}: {error}") from None
        self._jobs.add(job_id)
        self._job_requests.pop(request_id, None)
        return await self._follow_job(job_id, label)

    async def _pull(self, ref: str, policy: str) -> str:
        return await self._start_job(
            lambda request_id: self._client.image_pull(ref, policy, request_id),
            f"image pull {ref}",
        )

    async def _build(self, name: str, request: Any) -> str:
        build = self._grant.get("build")
        if build is None:
            raise ManagedSandboxError(
                f"service {name} needs an image build, which this sandbox grant "
                "does not offer (unsupported); set docker_image to a pullable "
                "image instead"
            )
        timeout = min(
            float(self.task_env_config.build_timeout_sec), float(build["max_build_sec"])
        )
        network = request.network or (
            "public" if "public" in build["network"] else "none"
        )

        def stage() -> str:
            with tempfile.TemporaryFile() as spool:
                _tar_tree(Path(request.context), spool)
                return self._client.upload_stage(spool)["stage_id"]

        stage_id = await self._call(stage)
        handle = await self._start_job(
            lambda request_id: self._client.image_build(
                stage_id,
                dockerfile=request.dockerfile,
                dockerfile_inline=request.dockerfile_inline,
                target=request.target,
                build_args=dict(request.args),
                no_cache=request.no_cache,
                network=network,
                timeout_sec=timeout,
                request_id=request_id,
            ),
            f"image build for {name}",
        )
        self._built.add(handle)
        return handle

    async def _create_env(self, spec: dict) -> float:
        """Create the env; returns when it expires on this loop's clock,
        counted from before the request so never later than the broker's."""
        loop = asyncio.get_running_loop()
        while True:
            request_id = uuid.uuid4().hex
            self._create = (spec, request_id)
            sent = loop.time()
            try:
                created = await self._call(self._client.env_create, spec, request_id)
            except wire.ProtocolError as error:
                if error.code == "quota" and error.field in _LIVE_QUOTAS:
                    # A failed create made nothing; retry with a new id.
                    self._create = None
                    self.logger.debug("managed sandbox waits for %s", error.field)
                    await asyncio.sleep(_LIVE_RETRY_SEC)
                    continue
                if error.code != "unknown-outcome":
                    self._create = None
                raise ManagedSandboxError(
                    f"env_create {error.code} ({error.field}): {error}"
                ) from None
            self._env_id = created["env_id"]
            self._create = None
            return sent + float(created["expires_in_sec"])

    async def _start_env(self, expires_at: float) -> float:
        """Start the env; returns the readiness wait sent.

        The broker refuses a wait beyond the env's remaining lifetime, which
        the create and the seeding already spent some of (spec A8: a Judge
        whose verifier timeout is below max_wait_timeout_sec). A refusal
        starts nothing: when slot queueing spent the slack, the wait is
        measured again from the expiry and resent once.
        """
        loop = asyncio.get_running_loop()
        retried = False
        while True:
            wait = min(
                float(self._limits["max_wait_timeout_sec"]),
                expires_at - loop.time() - _START_SLACK_SEC,
            )
            if wait <= 0:
                raise ManagedSandboxError(
                    "the environment's lifetime ended before it could start"
                )
            try:
                await self._call(self._client.env_start, self._env_id, wait)
            except wire.ProtocolError as error:
                if retried or not compose.lifetime_refusal(error):
                    raise
                self.logger.debug("managed sandbox resends env_start: %s", error)
                retried = True
                continue
            return wait

    async def _copy_seed(self, seed: Any) -> None:
        def copy() -> None:
            with tempfile.TemporaryFile() as spool:
                compose.write_seed_archive(seed, spool)
                staged = self._client.upload_stage(spool)
            self._client.copy_in(
                self._env_id, seed.service, seed.dest_dir, staged["stage_id"]
            )

        await self._call(copy)

    async def _wait_ready(self, timeout: float) -> dict:
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while True:
            wait = max(0.0, min(30.0, end - loop.time()))
            status = await self._call(self._client.env_status, self._env_id, wait)
            if status["state"] not in ("created", "starting") or not wait:
                return status

    @staticmethod
    def _describe_failure(status: dict) -> str:
        parts = [f"environment {status['state']} ({status.get('reason')})"]
        for name, service in sorted(status["services"].items()):
            line = (
                f"{name}: {service['state']}, health={service['health']}, "
                f"exit={service['exit_code']}"
            )
            diagnostics = service.get("diagnostics") or {}
            tail = diagnostics.get("health_tail") or diagnostics.get("log_tail")
            if tail:
                line += f": {tail.strip()[-512:]}"
            parts.append(line)
        return "; ".join(parts)

    async def _pick_shell(self) -> str:
        # Harbor's main service is a bash image; a prebuilt image without
        # bash (busybox, alpine) still gets its POSIX sh.
        for path in _BASH:
            found = await self._call(
                self._client.path_stat, self._env_id, MAIN_SERVICE_NAME, path, True
            )
            if found["exists"] and found["kind"] == "file":
                return "bash"
        self.logger.info("managed sandbox: main has no bash; commands run in sh")
        return "sh"

    async def _ensure_log_dirs(self) -> None:
        def copy() -> None:
            with tempfile.TemporaryFile() as spool:
                _tar_dirs([posixpath.basename(path) for path in LOG_TARGETS], spool)
                staged = self._client.upload_stage(spool)
            self._client.copy_in(
                self._env_id, MAIN_SERVICE_NAME, "/logs", staged["stage_id"]
            )

        await self._call(copy)

    async def _ensure_tmux(self) -> None:
        """The operator's static tmux for a main that has none on PATH.

        Harbor's terminus-2 checks ``tmux -V`` as root and otherwise installs
        tmux with apt/apk, which an env without network cannot. The broker
        copies the file it verified to /usr/local/bin/tmux of this env only;
        the image is never changed.
        """
        # uid 0, not "root": an image without /etc/passwd still answers.
        found = await self.exec("command -v tmux", user=0, timeout_sec=60)
        if found.return_code == 0:
            return
        try:
            result = await self._call(
                self._client._v2,
                "tool_install",
                {"env_id": self._env_id, "service": MAIN_SERVICE_NAME, "tool": "tmux"},
            )
        except wire.ProtocolError as error:
            raise ManagedSandboxError(
                f"tool_install tmux {error.code} ({error.field}): {error}"
            ) from None
        if not result["installed"]:
            return
        check = await self.exec("tmux -V", user=0, timeout_sec=60)
        if check.return_code != 0:
            self.logger.warning(
                "managed sandbox: the operator's tmux at %s does not run "
                "(is /usr/local/bin on PATH?): %s",
                result["path"],
                (check.stdout or "").strip()[-512:],
            )

    def _request_stop(self, delete: bool = False) -> asyncio.Task[None]:
        self._stop_requested = True
        self._delete = self._delete or delete
        if self._cleanup_task is None or (
            self._cleanup_task.done()
            and (
                self._env_id is not None
                or self._create is not None
                or self._job_requests
                or self._jobs
                # stop(delete=True) after a cleanup that kept built images.
                or (self._delete and self._built)
            )
        ):
            self._cleanup_task = self._owned_task(self._cleanup_owned())
        return self._cleanup_task

    async def _while_busy(self, deadline: float, function, *args):
        """Repeat a lifecycle call while the broker reports ``busy`` (its
        own earlier request is still in flight), until ``deadline``."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                return await self._call(function, *args)
            except wire.ProtocolError as error:
                if error.code != "busy" or loop.time() >= deadline:
                    raise
            await asyncio.sleep(_BUSY_RETRY_SEC)

    async def _cleanup_owned(self) -> None:
        # Bounded like the caller's wait: a later stop() retries the rest.
        deadline = asyncio.get_running_loop().time() + _LIFECYCLE_WAIT_SEC
        startup = self._startup_task
        if startup is not None and not startup.done():
            # Stop issuing startup requests; replies it loses are replayed.
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
        for request_id, replay in list(self._job_requests.items()):
            try:
                self._jobs.add(await self._call(replay))
            except wire.ProtocolError:
                pass
            self._job_requests.pop(request_id, None)
        for job_id in list(self._jobs):
            try:
                await self._call(self._client.job_cancel, job_id)
            except wire.ProtocolError:
                pass
            self._jobs.discard(job_id)
        if self._env_id is None and self._create is not None:
            spec, request_id = self._create
            try:
                # Replayed with the original id; busy while it is in flight.
                created = await self._while_busy(
                    deadline, self._client.env_create, spec, request_id
                )
            except wire.ProtocolError as error:
                if error.code in ("busy", "unknown-outcome"):
                    raise
                # It failed: nothing was created.
            else:
                self._env_id = created["env_id"]
            self._create = None
        if self._env_id is not None:
            await self._while_busy(deadline, self._client.env_destroy, self._env_id)
            self._env_id = None
            self._services = ()
        if self._delete:
            for handle in list(self._built):
                try:
                    await self._call(self._client.image_release, handle)
                except wire.ProtocolError as error:
                    self.logger.warning(
                        "managed sandbox could not release %s: %s", handle, error
                    )
                self._built.discard(handle)

    async def _wait_cleanup(self, task: asyncio.Task[None]) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_LIFECYCLE_WAIT_SEC)
        except TimeoutError as error:
            raise SandboxUnknownOutcomeError(
                "managed sandbox cleanup is pending; retry stop to reconcile"
            ) from error
        except wire.ProtocolError as error:
            raise ManagedSandboxError(
                f"managed sandbox cleanup failed: {error.code}: {error}"
            ) from None

    @override
    async def stop(self, delete: bool) -> None:
        # The env is destroyed whatever ``delete`` says: nothing is kept.
        await self._wait_cleanup(self._request_stop(delete))

    # -- exec ------------------------------------------------------------------------

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return await self._exec(
            MAIN_SERVICE_NAME,
            [self._main_shell, "-c", command],
            cwd=cwd or self.task_env_config.workdir,
            env=self._merge_env(env),
            user=self._resolve_user(user),
            timeout_sec=timeout_sec,
        )

    async def _exec(
        self,
        service: str,
        argv: list[str],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
        timeout_sec: int | float | None,
    ) -> ExecResult:
        env_id = self._require_env()
        request_id = uuid.uuid4().hex
        arguments = dict(
            cwd=None if cwd is None else _absolute(cwd, "cwd"),
            env=dict(env or {}),
            user=None if user is None else str(user),
            timeout_sec=float(timeout_sec) if timeout_sec else None,
            merge_stderr=True,
            request_id=request_id,
        )
        # An owned task: a cancelled caller leaves the worker thread running,
        # and the process it starts must still be found to be killed.
        start = self._owned_task(self._start_exec(env_id, service, argv, arguments))
        try:
            exec_id = await asyncio.shield(start)
        except asyncio.CancelledError:
            if self._on_cancel == "kill-group":
                self._track_kill(self._kill_when_started(start))
            raise
        callback = self._output_callback()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        chunks: list[bytes] = []
        offset = 0
        try:
            while True:
                view = await self._call(
                    self._client.exec_wait, exec_id, offset, 0, _WAIT_SEC
                )
                data = base64.b64decode(view["stdout_b64"])
                offset = view["stdout_offset"]
                if data:
                    chunks.append(data)
                    if callback is not None:
                        text = decoder.decode(data)
                        if text:
                            await callback(text, "stdout")
                if view["state"] != "running" and offset >= view["stdout_total"]:
                    break
        except asyncio.CancelledError:
            if self._on_cancel == "kill-group":
                self._track_kill(self._kill_exec(exec_id))
            raise
        if view["truncated"]:
            self.logger.warning("managed sandbox exec output was truncated")
        if view["state"] == "timed_out":
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds")
        code = view["exit_code"]
        if code is None and view["signal"]:
            code = 128 + int(getattr(signals, "SIG" + view["signal"]))
        if code is None:
            raise ManagedSandboxError(
                f"exec ended {view['state']} ({view['reason']}) without an exit status"
            )
        output = b"".join(chunks).decode("utf-8", errors="replace")
        return ExecResult(stdout=output or None, stderr=None, return_code=code)

    async def _start_exec(
        self, env_id: str, service: str, argv: list[str], arguments: dict
    ) -> str:
        attempts = 0
        while True:
            try:
                return await self._call(
                    self._client.exec_start, env_id, service, argv, **arguments
                )
            except wire.ProtocolError as error:
                # exec_start is idempotent: replay the same request_id.
                attempts += 1
                if error.code != "unknown-outcome" or attempts == 3:
                    raise ManagedSandboxError(
                        f"exec_start {error.code} ({error.field}): {error}"
                    ) from None

    def _track_kill(self, coroutine) -> None:
        task = self._owned_task(coroutine)
        self._kills.add(task)
        task.add_done_callback(self._kills.discard)

    async def _kill_when_started(self, start: asyncio.Task[str]) -> None:
        """Cancelled during exec_start: kill the process once it is known."""
        try:
            exec_id = await start
        except ManagedSandboxError as error:
            # Refused, or never provably started: env destroy ends it.
            self.logger.debug("managed sandbox exec kill: %s", error)
            return
        await self._kill_exec(exec_id)

    async def _kill_exec(self, exec_id: str) -> None:
        """TERM the cancelled exec's process group, KILL it 2 s later."""
        try:
            view = await self._call(self._client.exec_kill, exec_id, "TERM", "group")
            if view["state"] != "running":
                return
            loop = asyncio.get_running_loop()
            end = loop.time() + _KILL_GRACE_SEC
            while loop.time() < end:
                view = await self._call(
                    self._client.exec_wait, exec_id, 0, 0, _KILL_GRACE_SEC, 1
                )
                if view["state"] != "running":
                    return
            await self._call(self._client.exec_kill, exec_id, "KILL", "group")
        except wire.ProtocolError as error:
            self.logger.debug("managed sandbox exec kill: %s", error)

    # -- files ------------------------------------------------------------------------

    async def _stat(self, service: str, path: str) -> dict:
        env_id = self._require_env()
        return await self._call(
            self._client.path_stat, env_id, service, _absolute(path, "path"), True
        )

    @override
    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return (await self._stat(MAIN_SERVICE_NAME, path))["kind"] == "dir"

    @override
    async def is_file(self, path: str, user: str | int | None = None) -> bool:
        return (await self._stat(MAIN_SERVICE_NAME, path))["kind"] == "file"

    @override
    async def service_is_dir(
        self,
        path: str,
        *,
        service: str | None = None,
        user: str | int | None = None,
    ) -> bool:
        target = MAIN_SERVICE_NAME if service is None else service
        return (await self._stat(target, path))["kind"] == "dir"

    async def _copy_in(self, service: str, dest_dir: str, write) -> None:
        env_id = self._require_env()

        def copy() -> None:
            with tempfile.TemporaryFile() as spool:
                write(spool)
                staged = self._client.upload_stage(spool)
            self._client.copy_in(env_id, service, dest_dir, staged["stage_id"])

        await self._call(copy)

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        # docker-cp semantics: into a directory (or a trailing '/'), keep
        # the source name; otherwise the target names the file.
        target = str(target_path)
        clean = _absolute(target, "target_path")
        into = target.endswith("/")
        if not into:
            into = (await self._stat(MAIN_SERVICE_NAME, clean))["kind"] == "dir"
        if into:
            dest_dir, name = clean, Path(source_path).name
        else:
            dest_dir, name = posixpath.dirname(clean) or "/", posixpath.basename(clean)
        await self._copy_in(
            MAIN_SERVICE_NAME,
            dest_dir,
            lambda spool: _tar_file(Path(source_path), name, spool),
        )

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._copy_in(
            MAIN_SERVICE_NAME,
            _absolute(target_dir, "target_dir"),
            lambda spool: _tar_tree(Path(source_dir), spool),
        )

    async def _copy_out(self, service: str, path: str, exclude=(), use=None):
        env_id = self._require_env()
        source = _absolute(path, "source")

        def copy():
            result = self._client.copy_out(
                env_id, service, source, _MAX_COPY_BYTES, exclude
            )
            with tempfile.TemporaryFile() as spool:
                self._client.download_stage(result["stage_id"], spool)
                spool.seek(0)
                with tarfile.open(fileobj=spool, mode="r") as archive:
                    return use(archive)

        return await self._call(copy)

    async def _download_file(
        self, service: str, source_path: str, target_path: Path | str
    ) -> None:
        def use(archive: tarfile.TarFile) -> None:
            members = archive.getmembers()
            if len(members) != 1 or not members[0].isreg():
                raise ManagedSandboxError(f"{source_path} is not a regular file")
            reader = archive.extractfile(members[0])
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as output:
                while chunk := reader.read(1024**2):
                    output.write(chunk)

        await self._copy_out(service, source_path, use=use)

    async def _download_dir(
        self, service: str, source_dir: str, target_dir: Path | str, exclude=()
    ) -> None:
        await self._copy_out(
            service,
            source_dir,
            exclude,
            use=lambda archive: _extract(archive, Path(target_dir)),
        )

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._download_file(MAIN_SERVICE_NAME, source_path, target_path)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        await self._download_dir(MAIN_SERVICE_NAME, source_dir, target_dir)

    @override
    async def download_dir_with_exclusions(
        self, *, source_dir: str, target_dir: Path | str, exclude: list[str]
    ) -> None:
        await self._download_dir(MAIN_SERVICE_NAME, source_dir, target_dir, exclude)

    @override
    async def service_download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
        service: str | None = None,
    ) -> None:
        await self._download_dir(
            MAIN_SERVICE_NAME if service is None else service,
            source_dir,
            target_dir,
            exclude,
        )

    @override
    async def download_dir_filtered(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        include: Sequence[str] | None = None,
        exclude: Sequence[str] | None = None,
        protect: Sequence[str] | None = None,
    ) -> None:
        """Harbor's filter semantics (base.py): exclude wins; ``protect``
        paths are kept whenever present."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        def use(archive: tarfile.TarFile) -> bool:
            paths = [
                relative for member, relative in _members(archive) if member.isreg()
            ]
            selected = filter_paths_by_patterns(paths, include=include, exclude=exclude)
            if protect:
                missing = set(protect) - set(selected)
                selected += [path for path in paths if path in missing]
            if not selected:
                return False
            _extract(archive, target, set(selected))
            return True

        if not await self._copy_out(MAIN_SERVICE_NAME, source_dir, use=use):
            self.logger.warning(
                f"No files in {source_dir!r} matched include={include} "
                f"exclude={exclude}; downloading nothing"
            )

    # -- compose services -------------------------------------------------------------

    @override
    def _compose_service_transport(self, service: str | None) -> _ServiceTransport:
        self._require_env()
        if service not in self._services:
            raise self._compose_unsupported(service)
        return _ServiceTransport(self)

    async def _stop_service(self, service: str) -> None:
        env_id = self._require_env()
        try:
            await self._call(self._client.env_stop_service, env_id, service, 10)
        except wire.ProtocolError as error:
            raise ManagedSandboxError(
                f"stop_service {service}: {error.code}: {error}"
            ) from None


__all__ = [
    "LOG_TARGETS",
    "ManagedSandboxEnvironment",
    "ManagedSandboxError",
    "SandboxUnknownOutcomeError",
]
