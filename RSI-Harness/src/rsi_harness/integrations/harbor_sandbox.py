"""Harbor environment adapter for broker-owned managed CPU sandboxes."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, override

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.models.task.config import NetworkMode, VerifierEnvironmentMode
from harbor.models.task.verifier_mode import (
    resolve_step_verifier_mode,
    resolve_task_verifier_mode,
)
from harbor.trial.network_policy import resolve_trial_network_plan

from rsi_harness.integrations.sandbox_client import (
    ProtocolError,
    SandboxClient,
    iter_local_records,
    iter_selected_records,
    write_local_records,
)
from rsi_harness.runtime.sandbox_contracts import (
    SandboxProfile,
    absolute_path,
    below,
)

_STANDARD_MOUNT_TARGETS = frozenset(
    {"/logs/agent", "/logs/verifier", "/logs/artifacts"}
)
_REQUIRED_SCRATCH_ROOTS = frozenset(
    {"/tests", "/solution", "/logs", "/tmp", "/dev/shm"}
)
_DEFAULT_OPERATION_TIMEOUT_SEC = 30
_LIFECYCLE_WAIT_SEC = 6
_ROOT_USERS = (None, "root", "0", "0:0", 0)


class ManagedSandboxError(RuntimeError):
    """Base error for a terminal managed-sandbox operation."""


class SandboxExecutionTimeoutError(TimeoutError, ManagedSandboxError):
    """The broker contained a child command after its deadline."""


class SandboxQuotaExceededError(ManagedSandboxError):
    """The operation exceeded an approved resource or byte budget."""


class SandboxUnknownOutcomeError(ManagedSandboxError):
    """The broker cannot prove whether a mutating operation completed."""


def _require_root_user(user: str | int | None, label: str) -> None:
    if user not in _ROOT_USERS:
        raise ValueError(f"managed sandbox {label} must use the root user")


def preflight_managed_trial(trial: Any) -> None:
    """Validate the complete parsed Harbor task before sandbox child creation."""

    environment = trial.agent_environment
    if not isinstance(environment, ManagedSandboxEnvironment):
        raise TypeError("managed trial preflight requires ManagedSandboxEnvironment")
    environment._whole_task_preflight_complete = False

    task_config = trial.task.config
    _require_root_user(task_config.agent.user, "agent")
    _require_root_user(task_config.verifier.user, "verifier")
    for hook in task_config.verifier.collect:
        _require_root_user(hook.user, "verifier collect hook")

    steps = task_config.steps or []
    for step in steps:
        _require_root_user(step.agent.user, f"step {step.name!r} agent")
        _require_root_user(step.verifier.user, f"step {step.name!r} verifier")
        for hook in step.verifier.collect:
            _require_root_user(hook.user, f"step {step.name!r} verifier collect hook")

    plan_steps = steps or [None]
    for step in plan_steps:
        verifier_mode = (
            resolve_step_verifier_mode(task_config, step)
            if step is not None
            else resolve_task_verifier_mode(task_config)
        )
        if verifier_mode == VerifierEnvironmentMode.SEPARATE:
            raise ValueError(
                "managed sandboxes do not support separate verifier environments "
                "in version 1"
            )
        plan = resolve_trial_network_plan(
            task_config,
            trial.config.agent,
            trial.config.environment,
            step,
            verifier_mode=verifier_mode,
        )
        policies = (
            ("agent environment", plan.agent_env_baseline),
            ("agent phase", plan.agent_phase),
            ("verifier phase", plan.verifier_phase),
        )
        for label, policy in policies:
            if policy.network_mode != NetworkMode.NO_NETWORK:
                raise ValueError(
                    f"managed sandbox {label} must use network_mode='no-network'"
                )

    environment._whole_task_preflight_complete = True


class ManagedSandboxEnvironment(BaseEnvironment):
    """Adapt Harbor's single-container API to an authorized broker session."""

    def __init__(
        self,
        *args: Any,
        profile: str,
        lifetime_sec: float | None = None,
        socket_path: str | Path | None = None,
        credential: str | None = None,
        client: SandboxClient | None = None,
        **kwargs: Any,
    ) -> None:
        # BaseEnvironment validates through subclass properties and methods.
        # Broker configuration therefore has to be complete before super().
        self._client = client or SandboxClient(socket_path, credential)
        self._profile_name = profile
        capabilities = self._client.capabilities()
        profiles = {
            item.name: item
            for item in (
                SandboxProfile.model_validate_json(json.dumps(raw))
                for raw in capabilities.get("profiles", ())
            )
        }
        if profile not in profiles:
            raise ValueError(f"sandbox profile {profile!r} is not granted")
        self._profile = profiles[profile]
        self._lifetime_sec = (
            self._profile.max_lifetime_sec
            if lifetime_sec is None
            else float(lifetime_sec)
        )
        if not 0 < self._lifetime_sec <= self._profile.max_lifetime_sec:
            raise ValueError("lifetime_sec exceeds the approved sandbox profile")
        self._child_id: str | None = None
        self._create_request_id: str | None = None
        self._create_pending = False
        self._startup_task: asyncio.Task[None] | None = None
        self._cleanup_task: asyncio.Task[None] | None = None
        self._operations: set[asyncio.Task[Any]] = set()
        self._stop_requested = False
        self._whole_task_preflight_complete = False
        super().__init__(*args, **kwargs)

    @staticmethod
    @override
    def type() -> str:
        return "rsi-managed-sandbox"

    @property
    @override
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, mounted=False)

    @classmethod
    @override
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    @override
    def _validate_definition(self) -> None:
        if self._network_policy.network_mode.value != "no-network":
            raise ValueError(
                "managed sandboxes require explicit network_mode='no-network'; "
                f"{self._network_policy.network_mode.value!r} is unsupported"
            )
        for policy in self._phase_network_policies:
            if policy.network_mode.value != "no-network":
                raise ValueError(
                    "managed sandbox phase network policy must be no-network; "
                    f"{policy.network_mode.value!r} is unsupported"
                )

        if self.extra_docker_compose_paths or any(
            (self.environment_dir / name).exists()
            for name in ("docker-compose.yaml", "docker-compose.yml")
        ):
            raise ValueError("managed sandboxes do not support Docker Compose")

        config = self.task_env_config
        expected = {
            "docker_image": self._profile.image,
            "cpus": self._profile.cpus,
            "memory_mb": self._profile.memory_mb,
            "workdir": self._profile.workdir,
        }
        for field, approved in expected.items():
            if getattr(config, field) != approved:
                raise ValueError(
                    f"environment {field} must equal approved profile value "
                    f"{approved!r}"
                )
        if config.storage_mb is not None:
            raise ValueError("environment storage_mb is not an implemented hard limit")
        if config.gpu_types:
            raise ValueError("managed sandboxes do not support GPU types")
        if config.mcp_servers:
            raise ValueError("managed sandboxes do not expose MCP server networking")

        scratch_roots = {path for path, _size in self._profile.tmpfs_mb}
        missing = _REQUIRED_SCRATCH_ROOTS - scratch_roots
        if missing:
            raise ValueError(
                "sandbox profile lacks Harbor scratch roots: "
                + ", ".join(sorted(missing))
            )
        for mount in self._mounts:
            if mount.get("type") != "bind" or mount.get("target") not in (
                _STANDARD_MOUNT_TARGETS
            ):
                raise ValueError(
                    "managed sandbox mounts may only name standard Harbor log "
                    "destinations as scratch hints"
                )

    def _require_child(self) -> str:
        if self._stop_requested:
            raise RuntimeError("managed sandbox is stopping or stopped")
        if self._child_id is None:
            raise RuntimeError("managed sandbox is not started")
        return self._child_id

    def _scratch_path(self, value: str, *, allow_root: bool = True) -> str:
        try:
            absolute_path(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "path must be a normalized absolute scratch path"
            ) from error
        roots = tuple(path for path, _size in self._profile.tmpfs_mb)
        if not any(below(value, root) for root in roots):
            raise ValueError(f"path {value!r} is outside approved scratch")
        if not allow_root and value in roots:
            raise ValueError("file path must name an entry below a scratch root")
        return value

    @staticmethod
    def _split_remote_file(path: str) -> tuple[str, str]:
        remote = PurePosixPath(path)
        return str(remote.parent), remote.name

    @staticmethod
    def _operation_timeout(timeout_sec: int | None) -> int:
        return _DEFAULT_OPERATION_TIMEOUT_SEC if timeout_sec is None else timeout_sec

    @staticmethod
    def _raise_protocol(error: ProtocolError) -> None:
        if error.code == "quota":
            raise SandboxQuotaExceededError(str(error)) from error
        if error.code == "unknown-outcome":
            raise SandboxUnknownOutcomeError(str(error)) from error
        raise error

    @override
    async def start(self, force_build: bool) -> None:
        if not self._whole_task_preflight_complete:
            raise RuntimeError(
                "managed sandbox whole-task preflight is required before start"
            )
        if force_build:
            raise ValueError("managed sandboxes do not support force_build")
        if (
            self._child_id is not None
            or self._create_pending
            or (self._startup_task is not None and not self._startup_task.done())
            or (self._cleanup_task is not None and not self._cleanup_task.done())
        ):
            raise RuntimeError(
                "managed sandbox is already started or cleanup is pending"
            )
        self._stop_requested = False
        self._cleanup_task = None
        self._create_request_id = uuid.uuid4().hex
        self._create_pending = True
        self._startup_task = self._owned_task(self._start_owned())
        try:
            # Cancelling a Harbor waiter must not discard a to_thread result.
            await asyncio.shield(self._startup_task)
            if self._stop_requested:
                raise ManagedSandboxError("managed sandbox stopped during startup")
        except asyncio.CancelledError:
            self._request_stop()
            raise
        except BaseException:
            await self._wait_cleanup(self._request_stop())
            raise

    @staticmethod
    def _owned_task(coroutine) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        # The adapter retains failed tasks for a later stop() retry. Retrieve
        # diagnostics even when a cancelled caller no longer awaits this task.
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        return task

    async def _operation(self, function, *args):
        # Admission and registration have no intervening await: once stop is
        # requested cleanup owns the complete set of admitted child operations.
        child_id = self._require_child()
        task = self._owned_task(asyncio.to_thread(function, child_id, *args))
        self._operations.add(task)
        task.add_done_callback(self._operations.discard)
        return await asyncio.shield(task)

    async def _create_owned(self, *, reconcile: bool = False) -> None:
        try:
            self._child_id = await asyncio.to_thread(
                self._client.create,
                self._profile_name,
                self._lifetime_sec,
                request_id=self._create_request_id,
            )
        except ProtocolError as error:
            if not reconcile and error.code in {
                "invalid",
                "permission",
                "unsupported",
                "quota",
                "expired",
            }:
                self._create_pending = False
            raise
        self._create_pending = False

    async def _start_owned(self) -> None:
        await self._create_owned()
        if self._stop_requested:
            return
        initialized = await self.ensure_dirs(
            ["/logs/agent", "/logs/verifier", "/logs/artifacts", "/tests", "/solution"]
        )
        if initialized is not None and initialized.return_code != 0:
            raise RuntimeError(
                "managed sandbox could not initialize Harbor scratch directories"
            )
        if not self._stop_requested:
            await self._upload_environment_dir_after_start()

    def _request_stop(self) -> asyncio.Task[None]:
        self._stop_requested = True
        if self._cleanup_task is None or (
            self._cleanup_task.done()
            and (self._child_id is not None or self._create_pending)
        ):
            self._cleanup_task = self._owned_task(self._cleanup_owned())
        return self._cleanup_task

    async def _cleanup_owned(self) -> None:
        if self._startup_task is not None:
            try:
                if not self._startup_task.cancelled():
                    await asyncio.shield(self._startup_task)
            except Exception:
                # Failed startup can still own a child or a lost create reply.
                pass
        # A cancelled caller does not cancel its admitted thread operation.
        # Failed operations also finish here; their error must not skip destroy.
        if self._operations:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tuple(self._operations)),
                return_exceptions=True,
            )
        if self._child_id is None and self._create_pending:
            # Only create is replayed, using its original idempotency identity.
            await self._create_owned(reconcile=True)
        if self._child_id is not None:
            while True:
                try:
                    await asyncio.to_thread(self._client.destroy, self._child_id)
                    break
                except ProtocolError as error:
                    if error.code != "busy":
                        raise
                    # A lost operation response can finish its client thread
                    # before the broker finishes. Destroy is idempotent; never
                    # replay execute or a transfer to reconcile that outcome.
                    await asyncio.sleep(0.05)
            # Failed or cancelled destruction is not proof that authority ended.
            self._child_id = None

    async def _wait_cleanup(self, task: asyncio.Task[None]) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_LIFECYCLE_WAIT_SEC)
        except TimeoutError as error:
            raise SandboxUnknownOutcomeError(
                "managed sandbox cleanup is pending; retry stop to reconcile"
            ) from error
        except ProtocolError as error:
            self._raise_protocol(error)

    @override
    async def stop(self, delete: bool) -> None:
        await self._wait_cleanup(self._request_stop())

    @override
    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = self._scratch_path(target_path, allow_root=False)
        root, name = self._split_remote_file(target)
        source = Path(source_path)

        def upload(child_id: str) -> None:
            selected = tuple(iter_selected_records(source.parent, (source.name,)))
            if len(selected) != 1 or selected[0]["kind"] != "file":
                raise ValueError("upload_file source must be one regular file")
            records = ({**selected[0], "path": name},)
            self._client.upload(child_id, root, records)

        try:
            # Own preparation too: stop/restart must not move a pending upload
            # into a replacement child while its local files are being read.
            await self._operation(upload)
        except ProtocolError as error:
            self._raise_protocol(error)

    @override
    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        root = self._scratch_path(target_dir)

        def upload(child_id: str) -> None:
            records = tuple(iter_local_records(Path(source_dir)))
            self._client.upload(child_id, root, records)

        try:
            await self._operation(upload)
        except ProtocolError as error:
            self._raise_protocol(error)

    @override
    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = self._scratch_path(source_path, allow_root=False)
        root, name = self._split_remote_file(source)

        def download(child_id: str) -> None:
            records = self._client.download(child_id, root, (name,))
            if (
                len(records) != 1
                or records[0]["path"] != name
                or records[0]["kind"] != "file"
            ):
                raise ValueError("sandbox returned an invalid single-file download")
            target = Path(target_path)
            renamed = ({**records[0], "path": target.name},)
            write_local_records(target.parent, renamed)

        try:
            await self._operation(download)
        except ProtocolError as error:
            self._raise_protocol(error)

    @override
    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        root = self._scratch_path(source_dir)

        def download(child_id: str) -> None:
            records = self._client.download(child_id, root, (".",))
            write_local_records(Path(target_dir), records)

        try:
            await self._operation(download)
        except ProtocolError as error:
            self._raise_protocol(error)

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
        raise NotImplementedError(
            "filtered managed-sandbox downloads are unsupported in version 1"
        )

    @override
    async def download_dir_with_exclusions(
        self,
        *,
        source_dir: str,
        target_dir: Path | str,
        exclude: list[str],
    ) -> None:
        raise NotImplementedError(
            "exclusion managed-sandbox downloads are unsupported in version 1"
        )

    @override
    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        effective_user = self._resolve_user(user)
        if effective_user not in (None, "root", "0", "0:0", 0):
            raise ValueError("managed sandboxes support only the root user")
        effective_cwd = self._scratch_path(cwd or self._profile.workdir)
        try:
            result = await self._operation(
                self._client.execute,
                ["/bin/sh", "-c", command],
                effective_cwd,
                self._merge_env(env),
                self._operation_timeout(timeout_sec),
            )
        except ProtocolError as error:
            self._raise_protocol(error)

        if result.timed_out:
            raise SandboxExecutionTimeoutError("managed sandbox command timed out")
        if result.output_limited or result.oom_killed or result.truncated:
            reason = "memory" if result.oom_killed else "output"
            raise SandboxQuotaExceededError(
                f"managed sandbox command exceeded its {reason} quota"
            )
        if result.exit_code is None:
            raise SandboxUnknownOutcomeError(
                "managed sandbox command has no terminal exit status"
            )
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.exit_code,
        )
