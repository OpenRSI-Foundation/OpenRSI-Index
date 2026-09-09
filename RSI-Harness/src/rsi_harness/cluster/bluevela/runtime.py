"""Apptainer ports for the native RSI-Harness Engine.

The scheduler is deliberately absent here.  This module runs *inside* one LSF
allocation and implements only the Work/Judge ports consumed by RunCoordinator.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from rsi_harness.cluster.bluevela.allocation import AllocatedPools
from rsi_harness.cluster.bluevela.judge_controller import JudgeControllerControl
from rsi_harness.cluster.bluevela.multinode import (
    MultiNodeBroker,
    RemoteWorkerTemplate,
    WorkerBind,
)
from rsi_harness.cluster.bluevela.network import AgentNetworkBroker
from rsi_harness.errors import (
    InfrastructureError,
    RetryableSubmissionError,
    SetupError,
)
from rsi_harness.integrations.rsi_loop import (
    RSILoopAgentAdapter,
    rsi_loop_runtime_secret_values,
)
from rsi_harness.integrations.submit_client import generate_submit_client
from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    EvaluationRequest,
    ManagedNetwork,
    ManagedWorkdirVolume,
    RootfsSnapshotLease,
    RootfsSnapshotMode,
    RunPlan,
    RunRequest,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
    WorkQuiescence,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.coordinator import (
    EmbeddedSubmissionServerFactory,
    ProductionCoordinatorBackend,
    RunCoordinator,
    RunPreparation,
)
from rsi_harness.runtime.environment import resolve_runtime_environment
from rsi_harness.runtime.local_auth import AgentAuthMaterial, resolve_agent_auth
from rsi_harness.runtime.network import NetworkPolicyEnforcer
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.redaction import redact_exact_values, redact_text
from rsi_harness.runtime.reward import read_reward
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.config import load_config


class _Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def _safe_output(value: str, secrets: set[str]) -> str:
    return redact_text(redact_exact_values(value, secrets))


def _agent_provider_urls(
    payload: Any,
    plan: RunPlan,
    config: Any,
    *,
    auth_resolver: Callable[..., AgentAuthMaterial | None] = resolve_agent_auth,
) -> tuple[str, ...]:
    if config.http_proxy or config.https_proxy:
        raise SetupError(
            "Blue Vela no-network Agent does not support an upstream proxy"
        )
    agent = create_agent(plan.task.agent.name, config)
    auth = auth_resolver(
        source=payload.agent_auth,
        agent_name=plan.task.agent.name,
        agent_api_key=config.agent_api_key,
    )
    if auth is not None and auth.provider_endpoints:
        return auth.provider_endpoints
    provider = config.agent_api_base_url or agent.default_api_base_url
    if not provider:
        raise SetupError("no-network Agent has no provider endpoint")
    return (provider,)


class ApptainerAgentRuntime:
    """Tracked Work process plus fresh Judge processes in one allocation."""

    def __init__(
        self,
        payload: Any,
        plan: RunPlan,
        allocated_pools: AllocatedPools | None = None,
    ) -> None:
        self.payload = payload
        self.plan = plan
        self.profile = payload.profile
        self.workspace = plan.paths.workspace
        self.control = plan.paths.root / "engine-control"
        self.agent_home = plan.paths.root / "agent-home"
        self.task_assets = plan.paths.root / "task-tmp"
        self.rounds = plan.paths.root / "judge-rounds"
        raw_node_tmp = os.environ.get("RSI_HARNESS_NODE_TMP")
        if not raw_node_tmp or not Path(raw_node_tmp).is_absolute():
            raise SetupError("RSI_HARNESS_NODE_TMP must be an absolute path")
        self.node_tmp = Path(raw_node_tmp)
        self.work_tmp = self.node_tmp / "work"
        self.judge_tmp_root = self.node_tmp / "judge"
        self.feedback = (
            plan.paths.logs
            / "runs"
            / payload.run_id
            / plan.task.task_id
            / "feedback"
        )
        self._copies: dict[PurePosixPath, Path] = {}
        self._agent_auth_binds: list[tuple[Path, PurePosixPath, bool]] = []
        self._agent_auth_roots: dict[PurePosixPath, Path] = {}
        self.auth_secret_values: set[str] = set()
        self._process: subprocess.Popen[str] | None = None
        self._network_broker: AgentNetworkBroker | None = None
        self.allocated_pools = allocated_pools
        self.work_broker: MultiNodeBroker | None = None
        self.judge_broker: MultiNodeBroker | None = None
        self._paused = False
        self._lock = threading.RLock()
        if allocated_pools is not None:
            if allocated_pools.run_id != payload.run_id:
                raise SetupError("allocated pools differ from the runtime run id")
            if not allocated_pools.work:
                raise SetupError("multi-node Work pool must contain a controller")
            if allocated_pools.gpus_per_node != self.profile.resources.gpus_per_node:
                raise SetupError(
                    "allocated pool geometry differs from the cluster profile"
                )

    @property
    def work_devices(self) -> tuple[str, ...]:
        if self.allocated_pools is None:
            return self.plan.gpu_plan.work.uuids
        return self.allocated_pools.work[0].cuda_devices

    @property
    def work_ref(self) -> ContainerRef:
        return ContainerRef(
            container_id=f"bluevela-{self.payload.run_id}-work", role="work"
        )

    def _materialize_multinode_torchrun(self) -> Path:
        """Return the executable, node-local shim used by the Work Agent."""

        target = PurePosixPath("/usr/local/bin/torchrun")
        existing = self._copies.get(target)
        if existing is not None:
            if existing.stat().st_mode & 0o111 != 0o111:
                raise SetupError("transparent cluster torchrun shim is not executable")
            return existing
        shim_source = Path(__file__).with_name("torchrun_shim.py")
        if not shim_source.is_file():
            raise SetupError("transparent cluster torchrun shim is unavailable")
        launcher_root = self.node_tmp / "launchers"
        launcher_root.mkdir(mode=0o700, exist_ok=True)
        shim = launcher_root / "torchrun"
        shutil.copyfile(shim_source, shim)
        shim.chmod(0o755)
        self._copies[target] = shim
        return shim

    def _materialize_shared_judge_torchrun(self) -> Path:
        """Return a run-private shim visible from every allocated Judge host."""

        shim_source = Path(__file__).with_name("torchrun_shim.py")
        if not shim_source.is_file():
            raise SetupError("transparent cluster torchrun shim is unavailable")
        launcher_root = self.control / "multinode/launchers"
        launcher_root.mkdir(parents=True, mode=0o700, exist_ok=True)
        shim = launcher_root / "torchrun"
        if not shim.exists():
            shutil.copyfile(shim_source, shim)
            shim.chmod(0o755)
        if not shim.is_file() or shim.stat().st_mode & 0o111 != 0o111:
            raise SetupError("shared Judge torchrun shim is not executable")
        return shim

    def _judge_authority_bind(self) -> tuple[Path, PurePosixPath, bool] | None:
        """Resolve an optional task-specific, Judge-only authority mount."""

        root = self.profile.apptainer.judge_authority_root
        if root is None:
            return None
        source = root / self.plan.task.task_id
        if not source.exists():
            return None
        if not source.is_dir() or source.is_symlink():
            raise SetupError("Judge authority provider must be a real directory")
        return source, PurePosixPath("/run-contract"), True

    def initialize(self) -> None:
        if self.profile.apptainer.mount_policy == "scoped" and "public" in (
            self.plan.task.service.network_mode,
            self.plan.task.agent.network.mode,
            self.plan.task.verifier.network.mode,
        ):
            # Public phases retain their configured caches through narrow mounts.
            self.profile.storage.hf_home.mkdir(parents=True, exist_ok=True)
            self.profile.storage.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.control.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.agent_home.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.task_assets.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.rounds.mkdir(parents=True, mode=0o700, exist_ok=False)
        self.work_tmp.mkdir(mode=0o700)
        self.judge_tmp_root.mkdir(mode=0o700)
        submit = self.control / "rsi-submit"
        submit.write_text(generate_submit_client())
        submit.chmod(0o755)
        self._copies[PurePosixPath("/usr/local/bin/rsi-submit")] = submit
        if self.payload.agent_binary is None or not self.payload.agent_binary.is_file():
            raise SetupError("cluster Agent binary is unavailable on shared storage")
        launcher = getattr(self.payload, "agent_launcher", "codex")
        self._copies[PurePosixPath("/usr/local/bin") / launcher] = (
            self.payload.agent_binary
        )
        for companion in self.payload.agent_companions:
            if not companion.is_file():
                raise SetupError(
                    f"cluster Agent companion is unavailable: {companion.name}"
                )
            self._copies[PurePosixPath("/usr/local/bin") / companion.name] = (
                companion
            )
        relay = Path(__file__).with_name("netns_relay.py")
        if not relay.is_file():
            raise SetupError("cluster Agent network relay is unavailable")
        self._copies[PurePosixPath("/usr/local/bin/rsi-netns-relay")] = relay
        if self.allocated_pools is not None:
            self._materialize_multinode_torchrun()

        seed_target = PurePosixPath("/run/rsi-harness/seed")
        workdir = shlex.quote(str(self.plan.workdir))
        seed = shlex.quote(str(seed_target))
        command = (
            "/bin/bash",
            "-lc",
            "set -o pipefail; "
            "if command -v tar >/dev/null 2>&1; then "
            f"tar -C {workdir} -cf - . | tar -C {seed} -xpf -; "
            "else "
            f"cp -a -- {workdir}/. {seed}/; "
            "fi",
        )
        result = self._run(
            command,
            devices=self.work_devices,
            network_mode=self.plan.task.service.network_mode,
            extra_binds=((self.workspace, seed_target, False),),
            mount_workspace=False,
            timeout_seconds=1800,
        )
        if result.exit_code != 0:
            raise InfrastructureError(
                f"cannot initialize Apptainer workspace: {result.output}"
            )
        assets_target = PurePosixPath("/run/rsi-harness/task-tmp")
        assets = self._run(
            (
                "/bin/bash",
                "-lc",
                f"cp -a -- /tmp/. {assets_target}/",
            ),
            devices=self.work_devices,
            network_mode=self.plan.task.service.network_mode,
            extra_binds=((self.task_assets, assets_target, False),),
            mount_workspace=False,
            mount_agent_home=False,
            containall=False,
            timeout_seconds=600,
        )
        if assets.exit_code != 0:
            raise InfrastructureError(
                f"cannot preserve task image assets: {assets.output}"
            )
        self._install_auth()
        try:
            self._start_work_broker()
            self._smoke_agent()
        except BaseException:
            self.close_multinode()
            raise

    def _start_work_broker(self) -> None:
        pools = self.allocated_pools
        if pools is None:
            return
        if self.work_broker is not None:
            raise InfrastructureError("Work phase broker is already running")
        try:
            sif_sha256 = self.payload.sif_sha256_path.read_text().split()[0]
        except (AttributeError, OSError, IndexError) as error:
            raise SetupError("multi-node runtime cannot read the SIF digest") from error
        if len(sif_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sif_sha256
        ):
            raise SetupError("multi-node runtime SIF digest is invalid")
        network_mode = self.plan.task.agent.network.mode
        if network_mode not in {"public", "no-network"}:
            raise SetupError(
                "Blue Vela multi-node workers do not support allowlist networking"
            )
        binds = [
            WorkerBind(source=self.workspace, target=self.plan.workdir),
            WorkerBind(
                source=self.profile.apptainer.dns_bind,
                target=PurePosixPath(str(self.profile.apptainer.dns_bind)),
                read_only=True,
            ),
        ]
        binds.extend(
            WorkerBind(
                source=binding.source,
                target=binding.target,
                read_only=binding.read_only,
            )
            for binding in self.profile.apptainer.work_binds
        )
        binds.extend(
            WorkerBind(source=source, target=target, read_only=read_only)
            for source, target, read_only in self._cache_binds(network_mode)
        )
        binds.extend(
            WorkerBind(
                source=path,
                target=PurePosixPath(str(path)),
            )
            for path in self.profile.apptainer.rdma_binds
        )
        template = RemoteWorkerTemplate(
            apptainer_binary=self.profile.apptainer.binary,
            sif_path=self.payload.sif_path,
            sif_sha256=sif_sha256,
            container_python=self.profile.apptainer.container_python,
            container_workdir=self.plan.workdir,
            temp_root=self.profile.apptainer.temp_root,
            network_mode=network_mode,
            binds=tuple(binds),
            environment={
                **self.profile.apptainer.work_environment,
                **self._worker_huggingface_environment(network_mode),
            },
        )
        broker = MultiNodeBroker(
            phase="work",
            run_id=self.payload.run_id,
            nodes=pools.work,
            pool_digest=pools.digest,
            root=self.control / "multinode/work-client",
            authority_root=self.control / "multinode/work-authority",
            local_world_size=pools.gpus_per_node,
            worker_template=template,
            remote_binary=self.profile.scheduler.remote_binary,
            remote_host_flag=self.profile.scheduler.remote_host_flag,
        )
        broker.start()
        self.work_broker = broker

    def close_multinode(self) -> None:
        broker = self.work_broker
        self.work_broker = None
        if broker is not None:
            broker.stop()

    def require_work_idle(self) -> None:
        broker = self.work_broker
        if broker is not None:
            broker.require_idle()

    def _install_auth(self) -> None:
        if self.payload.agent_auth is None:
            return
        material = resolve_agent_auth(
            source=self.payload.agent_auth,
            agent_name=self.plan.task.agent.name,
            agent_api_key=None,
        )
        if material is None:
            raise SetupError("cluster local Agent authentication is unavailable")
        self.auth_secret_values.update(material.secret_values)
        auth_root = self.node_tmp / "agent-auth"
        auth_root.mkdir(mode=0o700)
        agent_home = PurePosixPath("/home/agent")
        for index, mount in enumerate(material.mounts):
            target = mount.tmpfs.target
            if target == agent_home or agent_home not in target.parents:
                raise SetupError(
                    "cluster Agent authentication must be rooted under /home/agent"
                )
            source = auth_root / str(index)
            source.mkdir(mode=0o700)
            for entry in mount.files:
                destination = source / entry.path.name
                destination.write_bytes(entry.content)
                destination.chmod(entry.mode)
            self._agent_auth_roots[target] = source
            self._agent_auth_binds.append((source, target, False))

    def configure_agent_network(
        self,
        submit_url: str,
        provider_urls: tuple[str, ...],
    ) -> None:
        if self._network_broker is not None:
            raise InfrastructureError("Agent network broker is already configured")
        endpoints = NetworkPolicyEnforcer(
            run_id=self.payload.run_id
        ).pin_endpoints(provider_urls)
        broker = AgentNetworkBroker(
            self.node_tmp / "network",
            endpoints=endpoints,
            submit_url=submit_url,
        )
        broker.start()
        self._network_broker = broker

    def close_agent_network(self) -> None:
        broker = self._network_broker
        self._network_broker = None
        if broker is not None:
            broker.stop()

    def _smoke_agent(self) -> None:
        version = self.payload.agent_version
        if not version:
            raise SetupError("cluster Agent runtime requires an exact version")
        launcher = getattr(self.payload, "agent_launcher", "codex")
        command = (f"/usr/local/bin/{launcher}", "--version")
        result = self._run(
            command,
            devices=self.work_devices,
            network_mode=self.plan.task.agent.network.mode,
            timeout_seconds=60,
        )
        if result.exit_code != 0 or version not in result.output:
            raise InfrastructureError(
                f"cluster Agent binary smoke failed: {result.output}"
            )

    def copy_to(
        self, container: ContainerRef, source: Path, target: PurePosixPath
    ) -> None:
        self._require_work(container)
        source = Path(source).resolve()
        for target_root, source_root in self._agent_auth_roots.items():
            if target == target_root or target_root in target.parents:
                relative = target.relative_to(target_root)
                destination = source_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
                return
        for container_home in (
            PurePosixPath("/home/agent"),
            PurePosixPath("/root"),
        ):
            if target == container_home or container_home in target.parents:
                relative = target.relative_to(container_home)
                destination = self.agent_home.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
                return
        self._copies[target] = source

    def exec(
        self,
        container: ContainerRef,
        command: str | tuple[str, ...] | list[str],
        *,
        timeout_seconds: float | None = None,
        user: str | None = None,
        environment: dict[str, str] | None = None,
        output_path: Path | None = None,
        output_redact_values: tuple[str, ...] = (),
        output_callback: Any = None,
    ) -> AgentRunResult:
        self._require_work(container)
        del user
        argv = (
            ("/bin/bash", "-lc", command)
            if isinstance(command, str)
            else tuple(command)
        )
        # RSI Loop's Docker hook assumes an injected `agent` account. Apptainer
        # executes as the mapped host UID, so ownership is already correct.
        if argv[:2] == ("/bin/bash", "-lc"):
            argv = (
                *argv[:2],
                argv[2].replace(
                    "chown -R agent:agent /home/agent/.codex",
                    "chmod -R u+rwX /home/agent/.codex",
                ),
                *argv[3:],
            )
        return self._run(
            argv,
            devices=self.work_devices,
            network_mode=self.plan.task.agent.network.mode,
            environment=environment,
            output_path=output_path,
            output_redact_values=output_redact_values,
            output_callback=output_callback,
            timeout_seconds=timeout_seconds,
            track_work=True,
        )

    def run_judge(
        self,
        workspace: Path,
        request: EvaluationRequest,
        environment: dict[str, str],
    ) -> AgentRunResult:
        if self.allocated_pools is not None:
            return self._run_multinode_judge(workspace, request, environment)
        round_digest = hashlib.sha256(request.round_id.encode()).hexdigest()[:16]
        judge_tmp = self.judge_tmp_root / f"round-{round_digest}"
        shutil.copytree(self.task_assets, judge_tmp, symlinks=True)
        try:
            return self._run(
                request.run_plan.task.verifier.command,
                devices=request.run_plan.gpu_plan.judge.uuids,
                network_mode=request.run_plan.task.verifier.network.mode,
                environment=environment,
                output_path=request.verifier_output,
                extra_binds=(
                    (workspace, request.run_plan.workdir, False),
                    (
                        request.run_plan.task.source_dir / "tests",
                        PurePosixPath("/tests"),
                        True,
                    ),
                    (
                        request.verifier_logs,
                        PurePosixPath("/logs/verifier"),
                        False,
                    ),
                    (judge_tmp, PurePosixPath("/tmp"), False),
                ),
                mount_workspace=False,
                mount_agent_home=False,
                phase="judge",
                timeout_seconds=request.run_plan.task.verifier.timeout_seconds,
            )
        finally:
            shutil.rmtree(judge_tmp)

    def _worker_template(
        self,
        *,
        network_mode: Literal["public", "no-network"],
        binds: tuple[WorkerBind, ...],
        phase: Literal["work", "judge"],
    ) -> RemoteWorkerTemplate:
        try:
            sif_sha256 = self.payload.sif_sha256_path.read_text().split()[0]
        except (AttributeError, OSError, IndexError) as error:
            raise SetupError("multi-node runtime cannot read the SIF digest") from error
        return RemoteWorkerTemplate(
            apptainer_binary=self.profile.apptainer.binary,
            sif_path=self.payload.sif_path,
            sif_sha256=sif_sha256,
            container_python=self.profile.apptainer.container_python,
            container_workdir=self.plan.workdir,
            temp_root=self.profile.apptainer.temp_root,
            network_mode=network_mode,
            binds=binds,
            environment={
                **(
                    self.profile.apptainer.work_environment
                    if phase == "work"
                    else self.profile.apptainer.judge_environment
                ),
                **self._worker_huggingface_environment(network_mode),
            },
        )

    def _worker_huggingface_environment(
        self,
        network_mode: Literal["public", "no-network"],
    ) -> dict[str, str]:
        """Use private offline caches or explicitly mounted public caches."""

        if network_mode == "no-network":
            return {
                "HF_HOME": "/tmp/.cache/huggingface",
                "HF_DATASETS_CACHE": "/tmp/.cache/huggingface/datasets",
            }
        if self.profile.apptainer.mount_policy == "scoped":
            return {
                "HF_HOME": "/rsi-cache/huggingface",
                "HF_DATASETS_CACHE": "/rsi-cache/datasets",
            }
        return {
            "HF_HOME": str(self.profile.storage.hf_home),
            "HF_DATASETS_CACHE": str(self.profile.storage.hf_datasets_cache),
        }

    def _cache_binds(
        self, network_mode: str,
    ) -> tuple[tuple[Path, PurePosixPath, bool], ...]:
        if (
            network_mode != "public"
            or self.profile.apptainer.mount_policy != "scoped"
        ):
            return ()
        return (
            (
                self.profile.storage.hf_home,
                PurePosixPath("/rsi-cache/huggingface"),
                False,
            ),
            (
                self.profile.storage.hf_datasets_cache,
                PurePosixPath("/rsi-cache/datasets"),
                False,
            ),
        )

    def _run_multinode_judge(
        self,
        workspace: Path,
        request: EvaluationRequest,
        environment: dict[str, str],
    ) -> AgentRunResult:
        pools = self.allocated_pools
        assert pools is not None
        if not pools.verifier:
            raise SetupError("multi-node Judge pool is empty")
        network_mode = request.run_plan.task.verifier.network.mode
        if network_mode not in {"public", "no-network"}:
            raise SetupError(
                "Blue Vela multi-node Judge does not support allowlist networking"
            )
        round_digest = hashlib.sha256(request.round_id.encode()).hexdigest()[:16]
        round_root = self.control / "multinode/judge" / round_digest
        shim = self._materialize_shared_judge_torchrun()
        binds = [
            WorkerBind(
                source=workspace,
                target=request.run_plan.workdir,
                read_only=True,
            ),
            WorkerBind(
                source=request.run_plan.task.source_dir / "tests",
                target=PurePosixPath("/tests"),
                read_only=True,
            ),
            WorkerBind(
                source=request.verifier_logs,
                target=PurePosixPath("/logs/verifier"),
            ),
            WorkerBind(
                source=shim,
                target=PurePosixPath("/usr/local/bin/torchrun"),
                read_only=True,
            ),
            WorkerBind(
                source=self.profile.apptainer.dns_bind,
                target=PurePosixPath(str(self.profile.apptainer.dns_bind)),
                read_only=True,
            ),
        ]
        authority = self._judge_authority_bind()
        if authority is not None:
            source, target, read_only = authority
            binds.append(
                WorkerBind(source=source, target=target, read_only=read_only)
            )
        binds.extend(
            WorkerBind(
                source=binding.source,
                target=binding.target,
                read_only=binding.read_only,
            )
            for binding in self.profile.apptainer.judge_binds
        )
        binds.extend(
            WorkerBind(source=source, target=target, read_only=read_only)
            for source, target, read_only in self._cache_binds(network_mode)
        )
        binds.extend(
            WorkerBind(source=path, target=PurePosixPath(str(path)))
            for path in self.profile.apptainer.rdma_binds
        )
        worker = self._worker_template(
            network_mode=network_mode,
            binds=tuple(binds),
            phase="judge",
        )
        broker = MultiNodeBroker(
            phase="verifier",
            run_id=self.payload.run_id,
            nodes=pools.verifier,
            pool_digest=pools.digest,
            root=round_root / "client",
            authority_root=round_root / "authority",
            local_world_size=pools.gpus_per_node,
            worker_template=worker,
            remote_binary=self.profile.scheduler.remote_binary,
            remote_host_flag=self.profile.scheduler.remote_host_flag,
        )
        broker.start()
        self.judge_broker = broker
        request_id = hashlib.sha256(
            f"{self.payload.run_id}:{request.round_id}".encode()
        ).hexdigest()[:32]
        control = JudgeControllerControl(
            request_id=request_id,
            run_id=self.payload.run_id,
            host=pools.verifier[0].host,
            worker=worker,
            broker_root=broker.root,
            command=request.run_plan.task.verifier.command,
            environment=environment,
            local_world_size=pools.gpus_per_node,
        )
        control_path = round_root / "controller.json"
        self._atomic_control(control_path, control.model_dump_json(indent=2))
        command = (
            self.profile.scheduler.remote_binary,
            self.profile.scheduler.remote_host_flag,
            pools.verifier[0].host,
            sys.executable,
            "-m",
            "rsi_harness.cluster.bluevela.judge_controller",
            "run",
            "--control",
            str(control_path),
        )
        try:
            result = self._run_host_controller(
                command,
                timeout_seconds=request.run_plan.task.verifier.timeout_seconds,
                output_path=request.verifier_output,
            )
            broker.require_idle()
            return result
        finally:
            try:
                broker.stop()
            finally:
                self.judge_broker = None

    @staticmethod
    def _atomic_control(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{secrets.token_hex(8)}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(payload)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _run_host_controller(
        command: tuple[str, ...],
        *,
        timeout_seconds: float,
        output_path: Path,
    ) -> AgentRunResult:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            output, _stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            stop = (*command[:6], "stop", *command[7:])
            subprocess.run(
                stop,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
            process.terminate()
            output, _stderr = process.communicate(timeout=10)
        bounded = output[:16_000_000]
        text = bounded.decode(errors="replace")
        output_path.write_text(text)
        return AgentRunResult(
            exit_code=process.returncode,
            output=text,
            output_truncated=len(output) > len(bounded),
            full_output_captured=len(output) <= len(bounded),
            timed_out=timed_out,
        )

    def pause(self, container: ContainerRef) -> None:
        self._require_work(container)
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise InfrastructureError("Agent process is not running at submission")
            os.killpg(process.pid, signal.SIGSTOP)
            self._paused = True

    def unpause(self, container: ContainerRef) -> None:
        self._require_work(container)
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise InfrastructureError("Agent process stopped during Judge round")
            os.killpg(process.pid, signal.SIGCONT)
            self._paused = False

    def stop(self, container: ContainerRef) -> None:
        self._require_work(container)
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                self._paused = False
                return
            if self._paused:
                os.killpg(process.pid, signal.SIGCONT)
            os.killpg(process.pid, signal.SIGTERM)
            self._paused = False

    def quiescence(self, container: ContainerRef) -> WorkQuiescence:
        self._require_work(container)
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                return WorkQuiescence.STOPPED
            if not self._paused:
                os.killpg(process.pid, signal.SIGSTOP)
                self._paused = True
            return WorkQuiescence.PAUSED

    def _require_work(self, container: ContainerRef) -> None:
        if container != self.work_ref:
            raise SetupError("Apptainer operation targets an unknown Work process")

    def _base_command(
        self,
        *,
        devices: tuple[str, ...],
        environment: dict[str, str] | None,
        extra_binds: tuple[tuple[Path, PurePosixPath, bool], ...],
        mount_workspace: bool,
        mount_agent_home: bool,
        containall: bool,
        network_mode: Literal["public", "no-network", "allowlist"],
        phase: Literal["work", "judge"] = "work",
    ) -> tuple[str, ...]:
        command = [
            str(self.profile.apptainer.binary),
            "exec",
        ]
        if network_mode == "no-network":
            command.extend(
                ("--net", "--network", "none", "--hostname", "localhost")
            )
        elif network_mode == "allowlist":
            raise SetupError(
                "Blue Vela Apptainer does not support allowlist network policy"
            )
        command.append("--nv")
        if containall:
            command.append("--containall")
        else:
            # Apptainer otherwise bind-mounts the host /tmp over task assets
            # baked into the SIF. This read-only setup exec exposes image /tmp.
            command.extend(("--no-mount", "tmp"))
        command.extend(("--writable-tmpfs", "--cwd", str(self.plan.workdir)))
        binds: list[tuple[Path, PurePosixPath, bool]] = list(extra_binds)
        if (
            network_mode == "no-network"
            and mount_agent_home
            and self._network_broker is not None
        ):
            binds.append(
                (
                    self._network_broker.root,
                    PurePosixPath("/run/rsi-harness/network"),
                    True,
                )
            )
        if containall and not any(
            target == PurePosixPath("/tmp") for _source, target, _ro in binds
        ):
            binds.append((self.work_tmp, PurePosixPath("/tmp"), False))
        if mount_workspace:
            binds.append((self.workspace, self.plan.workdir, False))
            binds.append(
                (self.feedback, PurePosixPath("/run/rsi-harness/feedback"), True)
            )
        if mount_agent_home:
            binds.append(
                (self.agent_home, PurePosixPath("/home/agent"), False)
            )
            binds.extend(self._agent_auth_binds)
            binds.extend(
                (source, target, True) for target, source in self._copies.items()
            )
            if self.work_broker is not None:
                binds.append(
                    (
                        self.work_broker.root,
                        PurePosixPath("/run/rsi-harness/torchrun"),
                        False,
                    )
                )
        multi_node = self.allocated_pools is not None
        scoped = self.profile.apptainer.mount_policy == "scoped"
        if network_mode == "public" and not multi_node and not scoped:
            binds.extend(
                (path, PurePosixPath(str(path)), False)
                for path in self.profile.apptainer.extra_binds
            )
        if multi_node or scoped:
            phase_binds = (
                self.profile.apptainer.work_binds
                if phase == "work"
                else self.profile.apptainer.judge_binds
            )
            binds.extend(
                (binding.source, binding.target, binding.read_only)
                for binding in phase_binds
            )
        binds.extend(self._cache_binds(network_mode))
        if multi_node and phase == "judge":
            authority = self._judge_authority_bind()
            if authority is not None:
                binds.append(authority)
        for source, target, read_only in binds:
            suffix = ":ro" if read_only else ""
            command.extend(("--bind", f"{source}:{target}{suffix}"))
        profile_environment = self.profile.apptainer.environment
        if multi_node or scoped:
            phase_environment = (
                self.profile.apptainer.work_environment
                if phase == "work"
                else self.profile.apptainer.judge_environment
            )
            profile_environment = (
                {**profile_environment, **phase_environment}
                if scoped
                else phase_environment
            )
        values = {
            **profile_environment,
            **self._worker_huggingface_environment(network_mode),
            "TMPDIR": "/tmp",
            "TEMP": "/tmp",
            "TMP": "/tmp",
            "XDG_CACHE_HOME": "/tmp/.cache",
            "TRITON_CACHE_DIR": "/tmp/.cache/triton",
            "TORCH_EXTENSIONS_DIR": "/tmp/.cache/torch-extensions",
            "PIP_CACHE_DIR": "/tmp/.cache/pip",
            **(environment or {}),
            "CUDA_VISIBLE_DEVICES": ",".join(devices),
            "CODEX_HOME": "/home/agent/.codex",
            "CLAUDE_CONFIG_DIR": "/home/agent/.claude",
        }
        if mount_agent_home and self.work_broker is not None:
            values.update(
                {
                    # Apptainer applies PREPEND_PATH after the image's own
                    # environment, so ordinary `torchrun` resolves to the
                    # Harness-owned transparent launcher mounted above.
                    "PREPEND_PATH": "/usr/local/bin",
                    "RSI_MULTINODE_ROOT": "/run/rsi-harness/torchrun",
                    "RSI_LOCAL_WORLD_SIZE": str(
                        self.work_broker.local_world_size
                    ),
                }
            )
        for key, value in values.items():
            command.extend(("--env", f"{key}={value}"))
        command.append(str(self.payload.sif_path))
        return tuple(command)

    def _network_command(
        self,
        command: tuple[str, ...],
        *,
        network_mode: Literal["public", "no-network", "allowlist"],
        mount_agent_home: bool,
    ) -> tuple[str, ...]:
        if (
            network_mode == "no-network"
            and mount_agent_home
            and self._network_broker is not None
        ):
            return self._network_broker.wrap(
                command,
                relay_path=PurePosixPath("/usr/local/bin/rsi-netns-relay"),
            )
        return command

    def _run(
        self,
        command: tuple[str, ...],
        *,
        devices: tuple[str, ...],
        network_mode: Literal["public", "no-network", "allowlist"],
        environment: dict[str, str] | None = None,
        output_path: Path | None = None,
        output_redact_values: tuple[str, ...] = (),
        output_callback: Any = None,
        extra_binds: tuple[tuple[Path, PurePosixPath, bool], ...] = (),
        mount_workspace: bool = True,
        mount_agent_home: bool = True,
        containall: bool = True,
        phase: Literal["work", "judge"] = "work",
        timeout_seconds: float | None,
        track_work: bool = False,
    ) -> AgentRunResult:
        isolated_command = self._network_command(
            tuple(command),
            network_mode=network_mode,
            mount_agent_home=mount_agent_home,
        )
        argv = self._base_command(
            devices=devices,
            environment=environment,
            extra_binds=extra_binds,
            mount_workspace=mount_workspace,
            mount_agent_home=mount_agent_home,
            containall=containall,
            network_mode=network_mode,
            phase=phase,
        ) + isolated_command
        child_env = os.environ.copy()
        child_env["APPTAINER_BIND"] = str(self.profile.apptainer.dns_bind)
        child_env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=child_env,
        )
        if track_work:
            with self._lock:
                if self._process is not None and self._process.poll() is None:
                    process.kill()
                    raise InfrastructureError("an Agent process is already running")
                self._process = process
                self._paused = False
        secrets = set(output_redact_values)
        captured: list[str] = []
        captured_bytes = 0
        truncated = False
        stream = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stream = output_path.open("w")

        def drain() -> None:
            nonlocal captured_bytes, truncated
            assert process.stdout is not None
            for raw in process.stdout:
                safe = _safe_output(raw, secrets)
                encoded = safe.encode()
                if captured_bytes < 16_000_000:
                    remaining = 16_000_000 - captured_bytes
                    captured.append(encoded[:remaining].decode(errors="ignore"))
                    captured_bytes += min(len(encoded), remaining)
                else:
                    truncated = True
                if stream is not None:
                    stream.write(safe)
                    stream.flush()
                if output_callback is not None:
                    output_callback(safe)

        thread = threading.Thread(target=drain, name="apptainer-output", daemon=True)
        thread.start()
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        finally:
            thread.join(timeout=30)
            if stream is not None:
                stream.close()
            if track_work:
                with self._lock:
                    self._paused = False
        return AgentRunResult(
            exit_code=process.returncode,
            output="".join(captured),
            output_truncated=truncated,
            full_output_captured=not truncated,
            timed_out=timed_out,
        )


class NativeJudgeEvaluator:
    def __init__(
        self,
        runtime: ApptainerAgentRuntime,
        artifacts: RunArtifactWriter,
    ) -> None:
        self.runtime = runtime
        self.artifacts = artifacts

    def evaluate(self, request: EvaluationRequest, observer: Any) -> SubmissionReport:
        started = time.monotonic()
        snapshot = self.runtime.rounds / request.round_id / "workspace"
        observer.resource_event(
            "work_pause_planned",
            work_container_id=request.work_container.container_id,
        )
        self.runtime.pause(request.work_container)
        try:
            try:
                self.runtime.require_work_idle()
            except InfrastructureError as error:
                raise RetryableSubmissionError(
                    "active remote Work must finish before submission"
                ) from error
            request.verifier_logs.mkdir(
                parents=True,
                mode=0o700,
                exist_ok=False,
            )
            observer.resource_event(
                "snapshot_planned",
                round_id=request.round_id,
            )
            snapshot.parent.mkdir(parents=True, mode=0o700, exist_ok=False)
            shutil.copytree(self.runtime.workspace, snapshot, symlinks=True)
            observer.resource_event(
                "snapshot_acquired",
                snapshot_lease_id=f"workspace-{request.round_id}",
                snapshot_merged_path=str(snapshot),
            )
            observer.resource_event("judge_planned", round_id=request.round_id)
            judge_id = f"bluevela-{self.runtime.payload.run_id}-{request.round_id}"
            observer.resource_event("judge_created", judge_container_id=judge_id)
            environment = resolve_runtime_environment(
                request.run_plan.task.verifier.environment, os.environ
            )
            result = self.runtime.run_judge(snapshot, request, environment)
            reward = read_reward(
                request.verifier_logs,
                request.run_plan.task.verifier.primary_reward,
            )
            if result.timed_out:
                status = SubmissionStatus.VERIFIER_TIMEOUT
                error = "verifier timed out"
            elif result.exit_code != 0:
                status = SubmissionStatus.VERIFIER_ERROR
                error = f"verifier exited with {result.exit_code}"
            elif reward.error is not None:
                status = SubmissionStatus.VERIFIER_ERROR
                error = reward.error
            else:
                status = SubmissionStatus.COMPLETED
                error = None
            report = SubmissionReport(
                round_id=request.round_id,
                status=status,
                rewards=reward.rewards,
                score=reward.score,
                output=result.output,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                verifier_output_required=True,
                full_output_captured=True,
                duration_seconds=max(0.0, time.monotonic() - started),
                error=error,
            )
            self.artifacts.record_submission(report)
            shutil.rmtree(request.verifier_logs)
            try:
                request.verifier_logs.parent.rmdir()
            except OSError:
                pass
            observer.resource_event("judge_removed", judge_container_id=judge_id)
            observer.resource_event(
                "snapshot_released", snapshot_lease_id=f"workspace-{request.round_id}"
            )
            return report
        finally:
            self.runtime.unpause(request.work_container)
            observer.resource_event(
                "work_unpaused",
                work_container_id=request.work_container.container_id,
            )


class NativeEngineComposition:
    def __init__(
        self,
        payload: Any,
        plan: RunPlan,
        allocated_pools: AllocatedPools | None = None,
    ) -> None:
        self.payload = payload
        self.plan = plan
        self.clock = _Clock()
        self.runtime = ApptainerAgentRuntime(
            payload,
            plan,
            allocated_pools=allocated_pools,
        )
        self.artifacts: RunArtifactWriter | None = None
        config = load_config()
        self.agent = RSILoopAgentAdapter(config, runtime=self.runtime)
        self.provider_urls = (
            _agent_provider_urls(payload, plan, config)
            if plan.task.agent.network.mode == "no-network"
            else ()
        )
        self.evaluator: NativeJudgeEvaluator | None = None
        self.volume: ManagedWorkdirVolume | None = None
        self.secrets = rsi_loop_runtime_secret_values(config)

    def request(self) -> RunRequest:
        return RunRequest(
            task_dir=self.plan.task.source_dir,
            agent_name=self.plan.task.agent.name,
            options=self.payload.options,
            paths=self.plan.paths,
            model=self.plan.task.agent.model,
            reasoning_effort=self.plan.task.agent.reasoning_effort,
            agent_auth=self.payload.agent_auth,
        )

    def backend(self) -> ProductionCoordinatorBackend:
        request = self.request()
        return ProductionCoordinatorBackend(
            compiler=None,
            allocator=lambda _definition, _selectors: self.plan.gpu_plan,
            image_preparer=lambda _definition: self.plan.images,
            plan_preparer=lambda *_args: self.plan,
            artifact_starter=self.start_artifacts,
            server_starter=(
                lambda evaluator, artifacts, clock: EmbeddedSubmissionServerFactory(
                    bind_host="127.0.0.1", port=0, bridge_gateway="127.0.0.1"
                )(evaluator, artifacts, clock)
            ),
            network_planner=lambda _plan, _run: f"bluevela-{self.payload.run_id}",
            network_creator=self.create_network,
            network_remover=lambda _network: None,
            workdir_volume_planner=self.plan_volume,
            workdir_volume_creator=self.create_volume,
            workdir_volume_attester=lambda _work, volume: self._require_volume(volume),
            workdir_volume_remover=lambda _volume: None,
            work_name_planner=lambda _plan, _run: self.runtime.work_ref.container_id,
            work_creator=self.create_work,
            work_feedback_attester=lambda work: self.runtime._require_work(work),
            work_policy_planner=(
                lambda _plan, _work, _network: f"policy-{self.payload.run_id}"
            ),
            work_policy_installer=(
                lambda _plan, _work, _network: f"policy-{self.payload.run_id}"
            ),
            work_policy_remover=lambda _rule: None,
            work_starter=lambda _work: self._event("work_started", None),
            work_remover=lambda _work: None,
            hook_installer=self.install_hooks,
            agent_preparer=self.prepare_agent,
            agent_runner=self.run_agent,
            agent_stopper=self.stop_agent,
            work_quiescer=self.runtime.quiescence,
            retained_work_planner=self.plan_retained,
            work_retainer=self.retain,
            retained_work_releaser=lambda _lease: None,
            evaluator=self.evaluate,
            event_recorder=self._event,
            preparation=RunPreparation(
                request=request,
                definition=self.plan.task,
                gpu_plan=self.plan.gpu_plan,
            ),
        )

    def start_artifacts(self, plan: RunPlan, run_id: str) -> RunArtifactWriter:
        self.artifacts = RunArtifactWriter(plan, run_id=run_id)
        self.artifacts.start()
        self.evaluator = NativeJudgeEvaluator(self.runtime, self.artifacts)
        return self.artifacts

    def create_network(
        self, plan: RunPlan, run_id: str, planned: str
    ) -> ManagedNetwork:
        return ManagedNetwork(
            network_id=planned,
            name=planned,
            run_id=run_id,
            task_id=plan.task.task_id,
            role="work",
            internal=False,
        )

    def plan_volume(self, plan: RunPlan, run_id: str) -> ManagedWorkdirVolume:
        nonce = hashlib.sha256(f"{run_id}:{plan.workdir}".encode()).hexdigest()
        return ManagedWorkdirVolume(
            name=f"bv-{hashlib.sha256(run_id.encode()).hexdigest()[:24]}",
            run_id=run_id,
            task_id=plan.task.task_id,
            target=plan.workdir,
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce=nonce,
        )

    def create_volume(
        self, plan: RunPlan, run_id: str, planned: ManagedWorkdirVolume
    ) -> ManagedWorkdirVolume:
        del plan, run_id
        self.volume = planned
        return planned

    def _require_volume(self, volume: ManagedWorkdirVolume) -> None:
        if volume != self.volume:
            raise SetupError("Work workspace authority differs from its plan")

    def create_work(self, *_args: Any) -> ContainerRef:
        self.runtime.initialize()
        self.secrets.update(self.runtime.auth_secret_values)
        return self.runtime.work_ref

    def install_hooks(
        self, plan: RunPlan, work: ContainerRef, submit_url: str, token: str
    ) -> None:
        self.secrets.update((submit_url, token))
        if plan.task.agent.network.mode == "no-network":
            self.runtime.configure_agent_network(submit_url, self.provider_urls)
        try:
            self.agent.install_hooks(
                AgentHookRequest(
                    run_plan=plan,
                    container=work,
                    submit_url=submit_url,
                    token=token,
                )
            )
        except BaseException:
            self.runtime.close_agent_network()
            raise

    def stop_agent(self, work: ContainerRef) -> None:
        try:
            self.runtime.stop(work)
        finally:
            self.runtime.close_agent_network()

    def prepare_agent(self, plan: RunPlan, maximum: int | None) -> Any:
        if self.artifacts is None:
            raise RuntimeError("native artifacts are unavailable")
        prepared = self.agent.prepare(
            AgentPrepareRequest(
                run_plan=plan,
                prompt_path=self.artifacts.root / "agent_prompt.md",
                max_submissions=maximum,
            )
        )
        environment = dict(prepared.environment)
        environment.update(
            resolve_runtime_environment(plan.task.agent.environment, os.environ)
        )
        environment.update(
            {
                "CODEX_HOME": "/home/agent/.codex",
                "CLAUDE_CONFIG_DIR": "/home/agent/.claude",
            }
        )
        return replace(prepared, environment=tuple(sorted(environment.items())))

    def run_agent(
        self, prepared: Any, work: ContainerRef, timeout: float | None
    ) -> AgentRunResult:
        if self.artifacts is None:
            raise RuntimeError("native artifacts are unavailable")
        self._event("agent_started", {"name": prepared.agent_name})
        try:
            result = self.agent.run(
                AgentRunRequest(
                    prepared=prepared,
                    container=work,
                    timeout_seconds=timeout,
                    output_path=self.artifacts.root / "agent_output.txt",
                    output_redact_values=tuple(sorted(self.secrets)),
                    output_callback=lambda value: self._event(
                        "agent_output", value.rstrip()
                    ),
                )
            )
        finally:
            self.runtime.close_agent_network()
        safe = _safe_output(result.output, self.secrets)
        (self.artifacts.root / "run_agent.log").write_text(safe)
        self._event("agent_finished", {"exit_code": result.exit_code})
        return replace(result, output=safe)

    def evaluate(self, request: EvaluationRequest, observer: Any) -> SubmissionReport:
        if self.evaluator is None:
            raise RuntimeError("native Judge evaluator is unavailable")
        self._event("judge_started", {"round_id": request.round_id})
        report = self.evaluator.evaluate(request, observer)
        self._event(
            "judge_finished",
            {
                "round_id": request.round_id,
                "status": report.status.value,
                "score": report.score,
            },
        )
        return report

    def plan_retained(self, plan: RunPlan, work: ContainerRef) -> str:
        del plan, work
        digest = hashlib.sha256(f"{self.payload.run_id}:final".encode()).hexdigest()
        return f"rsi-harness-rootfs:retained-work-{digest}"

    def retain(
        self, plan: RunPlan, work: ContainerRef, planned: str
    ) -> RootfsSnapshotLease:
        digest = hashlib.sha256(str(self.runtime.workspace).encode()).hexdigest()
        return RootfsSnapshotLease(
            lease_id=f"retained-{digest[:24]}",
            purpose="retained-work",
            run_id=self.payload.run_id,
            task_id=plan.task.task_id,
            round_id="final",
            source_container_id=work.container_id,
            image_id=f"sha256:{digest}",
            image_ref=planned,
        )

    @staticmethod
    def _event(name: str, value: object) -> None:
        print(f"[rsi-engine] {name}: {value}", flush=True)


def validate_native_artifacts(plan: RunPlan, run_id: str) -> Path:
    root = plan.paths.logs / "runs" / run_id / plan.task.task_id
    required = (
        root / "final_result.json",
        root / "agent_output.txt",
        root / "run_agent.log",
        root / "submissions" / "agent-1" / "report.json",
        root / "feedback" / "agent-1.log",
    )
    missing = tuple(path for path in required if not path.is_file())
    if missing:
        raise InfrastructureError(
            "native Engine run is missing artifacts: "
            + ", ".join(str(path) for path in missing)
        )
    return root


def run_native_engine(
    payload: Any,
    plan: RunPlan,
    *,
    allocated_pools: AllocatedPools | None = None,
) -> None:
    """Compose and execute the same coordinator used by local Harness runs."""
    composition = NativeEngineComposition(
        payload,
        plan,
        allocated_pools=allocated_pools,
    )
    try:
        result = RunCoordinator(
            backend=composition.backend(),
            lease_store=LeaseStore(plan.paths.root / "leases"),
            run_id_factory=lambda: payload.run_id,
            clock=composition.clock,
        ).run(composition.request())
    finally:
        composition.runtime.close_multinode()
    (plan.paths.root / "engine-result.json").write_text(
        result.model_dump_json(indent=2)
    )
    if result.status is not RunStatus.COMPLETED:
        raise InfrastructureError(
            f"native Engine ended in {result.status.value}: "
            f"{result.total_rounds} submissions"
        )
    validate_native_artifacts(plan, payload.run_id)


__all__ = ["ApptainerAgentRuntime", "run_native_engine", "validate_native_artifacts"]
