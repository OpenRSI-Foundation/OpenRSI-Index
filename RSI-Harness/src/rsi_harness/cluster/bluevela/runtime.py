"""Apptainer ports for the native RSI-Harness Engine.

The scheduler is deliberately absent here.  This module runs *inside* one batch
allocation and implements only the Work/Judge ports consumed by RunCoordinator.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from rsi_harness.cluster.agent_version import parse_agent_version
from rsi_harness.cluster.bluevela.allocation import AllocatedPools
from rsi_harness.cluster.bluevela.judge_controller import JudgeControllerControl
from rsi_harness.cluster.bluevela.judge_llm import (
    JUDGE_CODEX_SCRATCH,
    cluster_judge_egress,
    cluster_login_refresh,
    install_pinned_codex,
    node_codex_source,
    resolve_cluster_judge_llm,
    unresolved_endpoints,
)
from rsi_harness.cluster.bluevela.multinode import (
    MultiNodeBroker,
    RemoteWorkerTemplate,
    WorkerBind,
    stop_command,
)
from rsi_harness.cluster.bluevela.network import (
    AgentNetworkBroker,
    JudgeProviderBroker,
    judge_proxy_environment,
)
from rsi_harness.cluster.bluevela.sandbox import ClusterSandbox
from rsi_harness.errors import (
    InfrastructureError,
    RetryableSubmissionError,
    SetupError,
    SubmissionError,
)
from rsi_harness.integrations.rsi_loop import (
    RSILoopAgentAdapter,
    rsi_loop_runtime_secret_values,
    task_solution_script,
)
from rsi_harness.integrations.submit_client import generate_submit_client
from rsi_harness.judge_agent import (
    JUDGE_AGENT_CODE_TARGET,
    JUDGE_AGENT_LAUNCHER,
    JUDGE_AGENT_LAUNCHER_TARGET,
    JUDGE_AGENT_STATE_ROOT,
    JUDGE_CODEX_TARGET,
    JUDGE_ISOLATE_CODE_TARGET,
    JUDGE_ISOLATE_LAUNCHER,
    JUDGE_ISOLATE_LAUNCHER_TARGET,
    install_judge_agent,
    install_judge_isolate,
)
from rsi_harness.models import (
    JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL_ENV,
    JUDGE_ISOLATION_FALLBACK_ENV,
    JUDGE_TRAJECTORY_ROOT,
    AgentHookRequest,
    AgentPrepareRequest,
    AgentResume,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    EvaluationRequest,
    GPURequirement,
    JudgeGPUMode,
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
from rsi_harness.runtime.artifacts import (
    RunArtifactWriter,
    _sudo_invoking_owner,
    recorded_submissions,
)
from rsi_harness.runtime.coordinator import (
    EmbeddedSubmissionServerFactory,
    ProductionCoordinatorBackend,
    RunCoordinator,
    RunPreparation,
)
from rsi_harness.runtime.environment import resolve_runtime_environment
from rsi_harness.runtime.gpu import (
    NvidiaSmiInventory,
    assert_gpu_processes_quiescent,
    resolve_allocation,
)
from rsi_harness.runtime.judge import JUDGE_ROUND_END_SLACK_SECONDS
from rsi_harness.runtime.judge_llm import judge_codex_login
from rsi_harness.runtime.judge_login_refresh import renew_operator_login
from rsi_harness.runtime.judge_model_access import (
    JudgeModelAccess,
    judge_needs_rsi_agent,
    resolve_judge_model_access,
)
from rsi_harness.runtime.local_auth import (
    AgentAuthMaterial,
    current_login,
    login_kept_reason,
    read_login_store,
    resolve_agent_auth,
)
from rsi_harness.runtime.network import NetworkPolicyEnforcer
from rsi_harness.runtime.oracle import oracle_failures
from rsi_harness.runtime.output import BoundedBytes
from rsi_harness.runtime.recovery import LeaseStore, retained_round_id
from rsi_harness.runtime.redaction import redact_exact_values, redact_text
from rsi_harness.runtime.resume import ResumeContext, plan_resume
from rsi_harness.runtime.reward import read_reward
from rsi_harness.runtime.sandbox_lifecycle import NullSandboxLifecycle
from rsi_harness.runtime.sandbox_policy import validate_cluster_sandbox
from rsi_harness.runtime.sandbox_server import SANDBOX_TARGET
from rsi_harness.runtime.trajectory import (
    capture_round_trajectory,
    release_round_trajectory,
)
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_harness.task.digest import require_tests_digest
from rsi_loop.harness.agent import create_agent, get_agent_class
from rsi_loop.harness.agent import oracle as oracle_module
from rsi_loop.harness.config import load_config


class _Clock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def _safe_output(value: str, secrets: set[str]) -> str:
    return redact_text(redact_exact_values(value, secrets))


def _gpu_preflight_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SubmissionError(
            f"Work GPU preflight {command[0]} query failed: {error}"
        ) from error


def _agent_provider_urls(
    payload: Any,
    plan: RunPlan,
    config: Any,
    *,
    auth_resolver: Callable[..., AgentAuthMaterial | None] = resolve_agent_auth,
) -> tuple[str, ...]:
    if plan.task.agent.name == "oracle":
        return ()
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
            plan.paths.logs / "runs" / payload.run_id / plan.task.task_id / "feedback"
        )
        self._copies: dict[PurePosixPath, Path] = {}
        self._copied_inputs: dict[PurePosixPath, tempfile.TemporaryDirectory[str]] = {}
        self._oracle_solution: Path | None = None
        # The Agent CLI and its companions, also bound into a model-access Judge.
        self._agent_cli_targets: list[PurePosixPath] = []
        self.auth_material: AgentAuthMaterial | None = None
        # Set by the composition once the Agent's login is known.
        self.judge_model_access: JudgeModelAccess | None = None
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
        # The Work sandbox endpoint directory (cluster/bluevela/sandbox.py).
        self.work_sandbox: Path | None = None
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

    @property
    def public_tests(self) -> bool:
        return self.plan.task.verifier.visibility == "public"

    def _require_public_tests_digest(self) -> None:
        if self.public_tests:
            require_tests_digest(
                self.plan.task.source_dir / "tests", self.plan.task.tests_digest
            )

    def initialize(self, *, resume: bool = False) -> None:
        """Prepare the run's directories and Work. A resumed attempt reuses
        the run's workspace, Agent home, task assets and Judge rounds as
        they are, and seeds nothing; one whose run never finished seeding,
        so no Agent ran on it, is seeded again."""
        if self.plan.task.agent.name == "oracle":
            if (
                self.profile.scheduler.kind != "slurm"
                or self.allocated_pools is not None
            ):
                raise SetupError("Oracle requires a single-node Slurm allocation")
            if resume:
                raise SetupError(
                    "Oracle validation cannot resume; start a fresh validation"
                )
        if resume and self.allocated_pools is not None:
            raise SetupError("a multi-node cluster run cannot be resumed yet")
        # The adapter refused these before submitting; check again in the job.
        require_cluster_judge_model_access(
            self.plan.task, multi_node=self.allocated_pools is not None
        )
        # Public tests are read-only in Work: prove them before Work starts.
        self._require_public_tests_digest()
        if self.profile.apptainer.mount_policy == "scoped" and "public" in (
            self.plan.task.service.network_mode,
            self.plan.task.agent.network.mode,
            self.plan.task.verifier.network.mode,
        ):
            # Public phases retain their configured caches through narrow mounts.
            self.profile.storage.hf_home.mkdir(parents=True, exist_ok=True)
            self.profile.storage.hf_datasets_cache.mkdir(parents=True, exist_ok=True)
        seeded = self.control / "seeded"
        kept = resume and seeded.is_file()
        for directory in (
            self.workspace,
            self.control,
            self.agent_home,
            self.task_assets,
            self.rounds,
        ):
            if not kept:
                # The image's own files seed a partial copy over again.
                directory.mkdir(parents=True, mode=0o700, exist_ok=resume)
            if resume and (not directory.is_dir() or directory.is_symlink()):
                raise SetupError(f"resumed run directory is missing: {directory}")
        self.work_tmp.mkdir(mode=0o700)
        self.judge_tmp_root.mkdir(mode=0o700)
        submit = self.control / "rsi-submit"
        submit.write_text(generate_submit_client())
        submit.chmod(0o755)
        self._copies[PurePosixPath("/usr/local/bin/rsi-submit")] = submit
        launcher = getattr(self.payload, "agent_launcher", "codex")
        if self.plan.task.agent.name == "oracle":
            # The frozen Harness package supplies this script; no installed model
            # CLI, credential, root operation or image rebuild is needed.
            source = Path(oracle_module.__file__).with_suffix("") / "rsi-oracle"
            binary = self.control / "rsi-oracle"
            shutil.copyfile(source, binary)
            binary.chmod(0o755)
        else:
            if (
                self.payload.agent_binary is None
                or not self.payload.agent_binary.is_file()
            ):
                raise SetupError("cluster Agent binary does not exist")
            binary = self.payload.agent_binary
        self._copies[PurePosixPath("/usr/local/bin") / launcher] = binary
        self._agent_cli_targets = (
            []
            if self.plan.task.agent.name == "oracle"
            else [PurePosixPath("/usr/local/bin") / launcher]
        )
        for companion in self.payload.agent_companions:
            if not companion.is_file():
                raise SetupError(
                    f"cluster Agent companion is unavailable: {companion.name}"
                )
            self._copies[PurePosixPath("/usr/local/bin") / companion.name] = companion
            self._agent_cli_targets.append(
                PurePosixPath("/usr/local/bin") / companion.name
            )
        relay = Path(__file__).with_name("netns_relay.py")
        if not relay.is_file():
            raise SetupError("cluster Agent network relay is unavailable")
        self._copies[PurePosixPath("/usr/local/bin/rsi-netns-relay")] = relay
        if self.allocated_pools is not None:
            self._materialize_multinode_torchrun()
        if not kept:
            self._seed()
            self._atomic_control(seeded, "")
        if judge_needs_rsi_agent(self.plan.task.verifier):
            self._require_judge_python()
        self._install_auth()
        try:
            self._start_work_broker()
            self._smoke_agent()
        except BaseException:
            self.close_multinode()
            raise

    def _seed(self) -> None:
        """Copy the image's WORKDIR into the workspace and its /tmp into the
        task assets the Judge starts from."""
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

    def _require_judge_python(self) -> None:
        """rsi-agent runs under the image's own python3, 3.8 or later."""
        check = "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 3)"
        result = self._run(
            ("/bin/sh", "-c", f"exec python3 -c {shlex.quote(check)}"),
            devices=(),
            network_mode="no-network",
            mount_workspace=False,
            mount_agent_home=False,
            timeout_seconds=120,
        )
        if result.exit_code != 0:
            raise SetupError(
                "model_access and judge_model need python3 3.8 or later in the "
                f"task image for rsi-agent: {result.output.strip()[-500:]}"
            )

    def configure_judge_model_access(self, access: JudgeModelAccess | None) -> None:
        if judge_needs_rsi_agent(self.plan.task.verifier) != (access is not None):
            raise SetupError("Judge model access differs from the task declaration")
        if access is not None and access.needs_codex_cli and access.codex_cli is None:
            raise SetupError(
                "the Judge LLM needs the Codex CLI, which was not prepared"
            )
        if (
            access is not None
            and access.egress == "pinned"
            and self.plan.task.verifier.network.mode != "no-network"
        ):
            raise SetupError(
                "Judge model access through pinned endpoints needs "
                '[verifier] network_mode = "no-network" on clusters'
            )
        self.judge_model_access = access

    def _judge_isolate_binds(self) -> list[tuple[Path, PurePosixPath, bool]]:
        """rsi-isolate, read-only, for every Judge."""
        self.control.mkdir(parents=True, mode=0o700, exist_ok=True)
        isolate = install_judge_isolate(self.control)
        return [
            (isolate, JUDGE_ISOLATE_CODE_TARGET, True),
            (isolate / JUDGE_ISOLATE_LAUNCHER, JUDGE_ISOLATE_LAUNCHER_TARGET, True),
        ]

    def _judge_agent_binds(
        self, root: Path, access: JudgeModelAccess, *, ends_at: float
    ) -> list[tuple[Path, PurePosixPath, bool]]:
        """One round's rsi-agent state and login copies, plus rsi-agent, the
        Agent CLI (model access) and the pinned Codex CLI (a Judge LLM with a
        Codex login), read-only; the credentials never reach the procedure.

        ``ends_at`` is when the Harness stops the round: rsi-agent checks a
        login's lifetime against a call's ``--timeout`` only up to then."""
        state = root / "state"
        private = state / "private"
        private.mkdir(parents=True, mode=0o700)
        state.chmod(0o711)
        private.chmod(0o700)
        descriptor = os.open(
            private / "launch.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(access.round_launch(ends_at))
        binds: list[tuple[Path, PurePosixPath, bool]] = [
            (state, JUDGE_AGENT_STATE_ROOT, False)
        ]
        if access.auth is not None:
            # Copies from the login held in memory, never Work's live files.
            for index, mount in enumerate(access.auth.mounts):
                source = root / "auth" / str(index)
                source.mkdir(parents=True, mode=0o700)
                for entry in mount.files:
                    destination = source / entry.path
                    destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
                    destination.write_bytes(entry.content)
                    destination.chmod(entry.mode)
                binds.append((source, mount.tmpfs.target, True))
        self.control.mkdir(parents=True, mode=0o700, exist_ok=True)
        agent = install_judge_agent(self.control)
        binds.append((agent, JUDGE_AGENT_CODE_TARGET, True))
        binds.append((agent / JUDGE_AGENT_LAUNCHER, JUDGE_AGENT_LAUNCHER_TARGET, True))
        if self.plan.task.verifier.model_access:
            binds.extend(
                (self._copies[target], target, True)
                for target in self._agent_cli_targets
            )
        if access.codex_cli is not None:
            # The verified node-local copy, never part of the task image.
            binds.append((access.codex_cli, JUDGE_CODEX_TARGET, True))
        return binds

    def prepare_judge_codex(self, sources: Mapping[str, Path]) -> Path:
        """The pinned Codex CLI for a Judge LLM, as a node-local directory.

        ``sources`` are the binaries the submit host found on shared storage,
        by architecture; this node's is copied here only when its SHA-256 is
        the release pinned for this compute node's architecture.
        """
        if not sources:
            raise SetupError("the Judge LLM's Codex CLI was not found on submit")
        return install_pinned_codex(
            node_codex_source(sources), self.node_tmp / JUDGE_CODEX_SCRATCH
        )

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
            remote_args=self.profile.scheduler.remote_args,
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
        if (
            self.allocated_pools is not None
            or self.plan.gpu_plan.judge_mode is not JudgeGPUMode.RELEASE_ALL
            or not self.work_devices
        ):
            return
        work_pids = self._work_process_ids()
        try:
            allocation = resolve_allocation(
                GPURequirement(count=len(self.work_devices)),
                requested=self.work_devices,
                inventory=NvidiaSmiInventory(
                    runner=_gpu_preflight_command,
                ).list_devices(),
            )
        except SetupError as error:
            raise SubmissionError(
                f"cannot resolve Work GPU allocation: {error}"
            ) from error
        assert_gpu_processes_quiescent(
            allocation,
            work_pids,
            runner=_gpu_preflight_command,
        )

    def _work_process_ids(self) -> frozenset[int]:
        """Attribute host PIDs by Work ancestry and its dedicated session."""
        with self._lock:
            process = self._process
            if process is None or process.poll() is not None:
                raise SubmissionError("Work process disappeared before GPU preflight")
            root_pid = process.pid
        result = _gpu_preflight_command(["ps", "-eo", "pid=,ppid=,sid="])
        if result.returncode != 0:
            raise SubmissionError(f"Work process query failed: {result.stderr.strip()}")
        parents: dict[int, tuple[int, int]] = {}
        try:
            for row in result.stdout.splitlines():
                if not row.strip():
                    continue
                pid, parent, session = map(int, row.split())
                if pid <= 0 or parent < 0 or session < 0 or pid in parents:
                    raise ValueError("invalid or duplicate process identity")
                parents[pid] = (parent, session)
            if root_pid not in parents or parents[root_pid][1] != root_pid:
                raise ValueError("Work session leader is missing or changed")
        except ValueError as error:
            raise SubmissionError(f"ambiguous Work process list: {error}") from error
        # start_new_session=True gives Work its own session. Retain reparented
        # session members and descendants that started new process groups/sessions.
        work_pids = {
            pid for pid, (_, session) in parents.items() if session == root_pid
        }
        while descendants := {
            pid
            for pid, (parent, _) in parents.items()
            if parent in work_pids and pid not in work_pids
        }:
            work_pids.update(descendants)
        return frozenset(work_pids)

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
        self.auth_material = material
        self.auth_secret_values.update(material.secret_values)
        auth_root = self.node_tmp / "agent-auth"
        auth_root.mkdir(mode=0o700)
        agent_home = PurePosixPath("/home/agent")
        # Work's $HOME is Apptainer's in-memory one, not the Agent home: the
        # CLI's sessions go from its node-local login directory to the Agent
        # home on shared storage here, where a resumed attempt finds them.
        store = get_agent_class(self.plan.task.agent.name).session_store
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
            if store is not None and (agent_home / store[0]).parent == target:
                kept = agent_home / store[1]
                self.agent_home.joinpath(store[1]).mkdir(mode=0o700, exist_ok=True)
                source.joinpath(PurePosixPath(store[0]).name).symlink_to(kept)
            self._agent_auth_roots[target] = source
            self._agent_auth_binds.append((source, target, False))

    def configure_agent_network(
        self,
        submit_url: str,
        provider_urls: tuple[str, ...],
    ) -> None:
        if self._network_broker is not None:
            raise InfrastructureError("Agent network broker is already configured")
        endpoints = NetworkPolicyEnforcer(run_id=self.payload.run_id).pin_endpoints(
            provider_urls
        )
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
        if self.plan.task.agent.name == "oracle":
            return
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
        reported = None
        if result.exit_code == 0:
            try:
                reported = parse_agent_version(result.output)
            except SetupError:
                pass
        if reported != version:
            raise InfrastructureError(
                "cluster Agent binary smoke failed: "
                f"{self.payload.agent_binary} as {command[0]} --version "
                f"exited {result.exit_code}; expected {version!r}, "
                f"parsed version {reported!r}; "
                f"output={result.output!r}. Put an executable that works inside "
                "the task container on the submit host's PATH; host-only "
                "wrappers and their dependencies are not installed implicitly."
            )

    def stage_solution(self) -> None:
        """Stage a disposable Work-only copy, owned by the Slurm user."""
        task_solution_script(self.plan.task.source_dir)
        destination = self.node_tmp / "oracle-solution"
        shutil.copytree(
            self.plan.task.source_dir / "solution", destination, symlinks=True
        )
        # The launcher must be able to empty the bind before rsi-submit. Never
        # chmod through a task symlink into the submit host's filesystem.
        for root, directories, files in os.walk(destination, followlinks=False):
            Path(root).chmod(Path(root).stat().st_mode | 0o700)
            for name in files:
                path = Path(root) / name
                if not path.is_symlink():
                    path.chmod(path.stat().st_mode | 0o600)
        script = destination / "solve.sh"
        script.chmod(script.stat().st_mode | 0o100)
        self._oracle_solution = destination

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
        # Match Docker copy_to's source-lifetime contract: callers may remove
        # a temporary source immediately after this returns. Keep these
        # read-only binds under node-owned storage, never a host path derived
        # from the requested container target.
        copies = self.node_tmp / "copied-inputs"
        copies.mkdir(mode=0o700, exist_ok=True)
        staging = tempfile.TemporaryDirectory(prefix="copy-", dir=copies)
        destination = Path(staging.name) / "payload"
        try:
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=True)
            else:
                shutil.copy2(source, destination)
        except BaseException:
            staging.cleanup()
            raise
        previous = self._copied_inputs.get(target)
        self._copied_inputs[target] = staging
        self._copies[target] = destination
        # Bound staging space by current targets. Only explicit copy owners are
        # retired; built-in Agent binaries and shims are never cleanup targets.
        if previous is not None:
            previous.cleanup()

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
        on_exec_start: Callable[[float], None] | None = None,
    ) -> AgentRunResult:
        self._require_work(container)
        del user
        if on_exec_start is not None:
            # The sandbox Work deadline starts with the Agent process.
            if timeout_seconds is None:
                raise SetupError("exec start callback requires a bounded deadline")
            on_exec_start(time.monotonic() + timeout_seconds)
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
        *,
        sandbox_endpoint: Any = None,
        trajectory: Path | None = None,
    ) -> AgentRunResult:
        # A public Judge runs exactly the tests Work saw.
        self._require_public_tests_digest()
        if self.allocated_pools is not None:
            if sandbox_endpoint is not None:
                raise SetupError("multi-node Judge cannot reach the sandbox broker")
            return self._run_multinode_judge(
                workspace, request, environment, trajectory=trajectory
            )
        trajectory_binds: tuple[tuple[Path, PurePosixPath, bool], ...] = ()
        if trajectory is not None:
            trajectory_binds = ((trajectory, JUDGE_TRAJECTORY_ROOT, True),)
        sandbox_binds: tuple[tuple[Path, PurePosixPath, bool], ...] = ()
        redact: tuple[str, ...] = ()
        if sandbox_endpoint is not None:
            environment = {**environment, **sandbox_endpoint.environment}
            sandbox_binds = (
                (sandbox_endpoint.directory, PurePosixPath(SANDBOX_TARGET), True),
            )
            redact = (sandbox_endpoint.environment["RSI_SANDBOX_TOKEN"],)
        round_digest = hashlib.sha256(request.round_id.encode()).hexdigest()[:16]
        judge_tmp = self.judge_tmp_root / f"round-{round_digest}"
        access = self.judge_model_access
        if judge_needs_rsi_agent(request.run_plan.task.verifier) and access is None:
            raise InfrastructureError("Judge model access was not resolved")
        if access is not None and access.work_login is not None:
            access = self._renew_work_login(access, request.round_id)
        command = tuple(request.run_plan.task.verifier.command)
        network_mode = request.run_plan.task.verifier.network.mode
        environment = {**environment, **self._judge_isolation_environment()}
        agent_binds: list[tuple[Path, PurePosixPath, bool]] = []
        agent_root: Path | None = None
        provider: JudgeProviderBroker | None = None
        shutil.copytree(self.task_assets, judge_tmp, symlinks=True)
        try:
            if access is not None:
                agent_root = self.judge_tmp_root / f"round-{round_digest}-agent"
                agent_root.mkdir(mode=0o700)
                timeout = request.run_plan.task.verifier.timeout_seconds
                agent_binds = self._judge_agent_binds(
                    agent_root,
                    access,
                    ends_at=time.time() + timeout + JUDGE_ROUND_END_SLACK_SECONDS,
                )
                # Only the Agent's name, model and effort reach the procedure;
                # the task's own variables cannot override these.
                environment = {**environment, **dict(access.environment)}
                redact = (*redact, *sorted(access.secret_values))
                if access.egress == "site":
                    # The network the Work Agent already uses for its model,
                    # governed by the site rather than by the Harness.
                    network_mode = "public"
                    print(
                        f"[rsi-engine] judge_model_egress: site ({request.round_id})",
                        flush=True,
                    )
                else:
                    network_mode = "no-network"
                    # Short: a Unix socket path is limited to 108 bytes.
                    provider = JudgeProviderBroker(
                        self.node_tmp / f"jnet-{round_digest}",
                        endpoints=access.endpoints,
                    )
                    provider.start()
                    relay = PurePosixPath("/usr/local/bin/rsi-netns-relay")
                    agent_binds += [
                        (
                            provider.root,
                            PurePosixPath("/run/rsi-harness/network"),
                            True,
                        ),
                        (self._copies[relay], relay, True),
                    ]
                    command = provider.wrap(command, relay_path=relay)
            return self._run(
                command,
                devices=request.run_plan.gpu_plan.judge.uuids,
                network_mode=network_mode,
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
                    *trajectory_binds,
                    *sandbox_binds,
                    *agent_binds,
                    *self._judge_isolate_binds(),
                ),
                mount_workspace=False,
                mount_agent_home=False,
                phase="judge",
                declared_network_mode=request.run_plan.task.verifier.network.mode,
                output_redact_values=redact,
                timeout_seconds=request.run_plan.task.verifier.timeout_seconds,
            )
        finally:
            try:
                if provider is not None:
                    provider.stop()
                    shutil.rmtree(provider.root, ignore_errors=True)
            finally:
                if agent_root is not None:
                    shutil.rmtree(agent_root, ignore_errors=True)
                shutil.rmtree(judge_tmp)

    def _renew_work_login(
        self, access: JudgeModelAccess, round_id: str
    ) -> JudgeModelAccess:
        """The login Work's Agent keeps refreshing, read while Work is stopped.

        As on Docker (JudgeRunner._renew_work_login): a missing, malformed or
        foreign login, or any other failure on it, keeps the last good copy,
        with a warning.
        """
        assert access.work_login is not None
        try:
            files = read_login_store(access.work_login, self._agent_auth_roots)
            renewed = access.with_work_login(current_login(access.work_login, files))
        except Exception as error:  # Agent-written data: never fatal
            reason = _safe_output(login_kept_reason(error), set(access.secret_values))
            print(f"[rsi-engine] judge_login_kept: {round_id}: {reason}", flush=True)
            return access
        self.judge_model_access = renewed
        self.auth_secret_values.update(renewed.secret_values)
        return renewed

    def renew_operator_login(self, round_id: str) -> None:
        """This round's Judge LLM gets the operator's current Codex login.

        As on Docker (JudgeRunner._renew_operator_login), here in the Engine
        on the compute node, before Work pauses: the pinned Codex CLI
        refreshes the login in place when its access token would not outlive
        the round, if this node can save it, one job at a time
        (judge_llm.cluster_login_refresh). Whatever fails, the round keeps the
        Judge's current copy and ``judge_login_refresh_failed`` says why.
        """
        access = self.judge_model_access
        if access is None or access.operator_login is None:
            return
        try:
            renewal = renew_operator_login(
                access.operator_login,
                codex_cli=access.codex_cli,
                round_seconds=float(self.plan.task.verifier.timeout_seconds),
                refresh=cluster_login_refresh,
            )
            renewed = access
            if (
                renewal.content is not None
                and judge_codex_login(renewal.content)
                != json.loads(access.launch)["llm"]["auth_json"]
            ):
                renewed = access.with_operator_login(renewal.content)
            warning = renewal.warning
        except Exception as error:  # never fatal: the copy works until it expires
            renewed = access
            warning = f"the login could not be used ({type(error).__name__})"
        if warning is not None:
            reason = _safe_output(warning, set(renewed.secret_values))
            print(
                f"[rsi-engine] judge_login_refresh_failed: {round_id}: {reason}",
                flush=True,
            )
        self.judge_model_access = renewed
        self.auth_secret_values.update(renewed.secret_values)

    def _judge_isolation_environment(self) -> dict[str, str]:
        """The operator's isolation policies, for a science Judge library.

        Always set the fallback and compatibility flag explicitly so an
        ``ENV`` in the task's image, which Apptainer keeps even with
        ``--cleanenv``, never decides.
        """
        return {
            JUDGE_ISOLATION_FALLBACK_ENV: (
                self.profile.apptainer.judge_isolation_fallback
            ),
            JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL_ENV: (
                "1"
                if self.profile.apptainer.judge_allow_missing_landlock_signal
                else "0"
            ),
        }

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
                **(self._judge_isolation_environment() if phase == "judge" else {}),
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
        self,
        network_mode: str,
    ) -> tuple[tuple[Path, PurePosixPath, bool], ...]:
        if network_mode != "public" or self.profile.apptainer.mount_policy != "scoped":
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
        *,
        trajectory: Path | None = None,
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
            *(
                ()
                if trajectory is None
                else (
                    WorkerBind(
                        source=trajectory,
                        target=JUDGE_TRAJECTORY_ROOT,
                        read_only=True,
                    ),
                )
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
            binds.append(WorkerBind(source=source, target=target, read_only=read_only))
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
        binds.extend(
            WorkerBind(source=source, target=target, read_only=read_only)
            for source, target, read_only in self._judge_isolate_binds()
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
            remote_args=self.profile.scheduler.remote_args,
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
            environment={**environment, **self._judge_isolation_environment()},
            local_world_size=pools.gpus_per_node,
        )
        control_path = round_root / "controller.json"
        self._atomic_control(control_path, control.model_dump_json(indent=2))
        command = (
            self.profile.scheduler.remote_binary,
            *self.profile.scheduler.remote_args,
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
        temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
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

    def _run_host_controller(
        self,
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
            stop = stop_command(
                command,
                remote_binary=self.profile.scheduler.remote_binary,
                remote_host_flag=self.profile.scheduler.remote_host_flag,
                remote_args=self.profile.scheduler.remote_args,
                module="rsi_harness.cluster.bluevela.judge_controller",
            )
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
            if process is None:
                self._paused = False
                return
            # A final Judge can finish after the Agent exits; its report is
            # valid. pause() stopped the whole group, so continue it even
            # then: members the Agent left behind must not stay frozen.
            try:
                os.killpg(process.pid, signal.SIGCONT)
            except ProcessLookupError:
                pass  # the whole group has exited
            self._paused = False

    def stop(self, container: ContainerRef) -> None:
        self._require_work(container)
        with self._lock:
            process = self._process
            if process is None:
                self._paused = False
                return
            # Signal the group even after the Agent exits, so members it left
            # behind (possibly stopped by pause()) are terminated too.
            try:
                if self._paused or process.poll() is not None:
                    os.killpg(process.pid, signal.SIGCONT)
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # the whole group has exited
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
        declared_network_mode: Literal["public", "no-network", "allowlist"]
        | None = None,
    ) -> tuple[str, ...]:
        # The phase's declared mode chooses its binds, caches and HF
        # environment; network_mode only its network (a site-egress Judge
        # gets the site network without Work's public binds and caches).
        declared = declared_network_mode or network_mode
        command = [
            str(self.profile.apptainer.binary),
            "exec",
        ]
        if network_mode == "no-network":
            command.extend(("--net", "--network", "none", "--hostname", "localhost"))
        elif network_mode == "allowlist":
            raise SetupError(
                "Blue Vela Apptainer does not support allowlist network policy"
            )
        if devices:
            command.append("--nv")
        command.append("--no-umask")
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
            if self.public_tests:
                binds.append(
                    (
                        self.plan.task.source_dir / "tests",
                        PurePosixPath("/tests"),
                        True,
                    )
                )
        if mount_agent_home:
            if phase == "work" and self._oracle_solution is not None:
                binds.append((self._oracle_solution, PurePosixPath("/solution"), False))
            binds.append((self.agent_home, PurePosixPath("/home/agent"), False))
            binds.extend(self._agent_auth_binds)
            binds.extend(
                (source, target, True) for target, source in self._copies.items()
            )
            if self.work_sandbox is not None and phase == "work":
                binds.append((self.work_sandbox, PurePosixPath(SANDBOX_TARGET), True))
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
        if declared == "public" and not multi_node and not scoped:
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
        binds.extend(self._cache_binds(declared))
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
            **self._worker_huggingface_environment(declared),
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
        prepend: list[str] = []
        if values.get("RSI_SANDBOX_SOCKET") == SANDBOX_TARGET + "/s":
            # As on Docker (DockerRuntime._sandbox_exec_environment): the
            # endpoint directory, with rsi-sandbox, first on PATH.
            prepend.append(SANDBOX_TARGET)
        if mount_agent_home and self.work_broker is not None:
            # Apptainer applies PREPEND_PATH after the image's own
            # environment, so ordinary `torchrun` resolves to the
            # Harness-owned transparent launcher mounted above.
            prepend.append("/usr/local/bin")
            values.update(
                {
                    "RSI_MULTINODE_ROOT": "/run/rsi-harness/torchrun",
                    "RSI_LOCAL_WORLD_SIZE": str(self.work_broker.local_world_size),
                }
            )
        if prepend:
            # Ahead of a PREPEND_PATH the cluster profile may set.
            if values.get("PREPEND_PATH"):
                prepend.append(values["PREPEND_PATH"])
            values["PREPEND_PATH"] = ":".join(prepend)
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
        declared_network_mode: Literal["public", "no-network", "allowlist"]
        | None = None,
        timeout_seconds: float | None,
        track_work: bool = False,
    ) -> AgentRunResult:
        isolated_command = self._network_command(
            tuple(command),
            network_mode=network_mode,
            mount_agent_home=mount_agent_home,
        )
        argv = (
            self._base_command(
                devices=devices,
                environment=environment,
                extra_binds=extra_binds,
                mount_workspace=mount_workspace,
                mount_agent_home=mount_agent_home,
                containall=containall,
                network_mode=network_mode,
                phase=phase,
                declared_network_mode=declared_network_mode,
            )
            + isolated_command
        )
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
        captured = BoundedBytes(16_000_000)
        stream = None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Append: a resumed run's Agent output continues the same file.
            stream = output_path.open("a", encoding="utf-8")

        def drain() -> None:
            assert process.stdout is not None
            for raw in process.stdout:
                safe = _safe_output(raw, secrets)
                encoded = safe.encode()
                captured.append(encoded)
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
        except BaseException:
            # Cancelled (the job's SIGTERM, which Slurm sends the job script
            # alone): stop the group before waiting for its output, or the
            # wait outlasts the job's grace and the attempt is never closed.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            raise
        finally:
            thread.join(timeout=30)
            if stream is not None:
                stream.close()
            if track_work:
                with self._lock:
                    self._paused = False
        return AgentRunResult(
            exit_code=process.returncode,
            output=captured.value.decode("utf-8", errors="ignore"),
            output_truncated=captured.truncated,
            full_output_captured=not captured.truncated,
            timed_out=timed_out,
        )


def require_cluster_judge_model_access(task: Any, *, multi_node: bool) -> None:
    """Refuse a Judge with rsi-agent that a cluster run cannot serve.

    The adapter calls this before it submits anything; the runtime repeats
    the checks inside the job. ``model_access`` needs a single-node Judge
    and a public or no-network Agent, and a no-network Agent's pinned
    endpoints a no-network Verifier. A Judge LLM (``judge_model``) alone
    needs a single-node Judge and a public or no-network Verifier
    (judge_llm.cluster_judge_egress).
    """
    verifier = task.verifier
    if not judge_needs_rsi_agent(verifier):
        return
    if multi_node:
        what = "Judge model access" if verifier.model_access else "A Judge LLM"
        raise SetupError(f"{what} on a multi-node Judge is not supported yet")
    egress = cluster_judge_egress(task)
    if (
        verifier.model_access
        and egress == "pinned"
        and verifier.network.mode != "no-network"
    ):
        raise SetupError(
            "Judge model access through pinned endpoints needs "
            '[verifier] network_mode = "no-network" on clusters'
        )


class NativeJudgeEvaluator:
    """One Judge round. With a sandbox lifecycle it follows the local
    ordering: freeze Work's envs, pause Work, run the round with its own
    endpoint, close the round's envs, then resume Work's envs, unpause Work
    and reopen its endpoint."""

    def __init__(
        self,
        runtime: ApptainerAgentRuntime,
        artifacts: RunArtifactWriter,
        sandbox_lifecycle: Any = None,
    ) -> None:
        self.runtime = runtime
        self.artifacts = artifacts
        self.sandbox = sandbox_lifecycle or NullSandboxLifecycle()

    def evaluate(self, request: EvaluationRequest, observer: Any) -> SubmissionReport:
        started = time.monotonic()
        snapshot = self.runtime.rounds / request.round_id / "workspace"
        # Before Work pauses, as on Docker: a refresh may take a while.
        self.runtime.renew_operator_login(request.round_id)
        # Busy Work envs make the submission retryable before any pause.
        self.sandbox.freeze_work()
        frozen = self.sandbox.enabled
        observer.resource_event(
            "work_pause_planned",
            work_container_id=request.work_container.container_id,
        )
        try:
            self.runtime.pause(request.work_container)
        except BaseException:
            if frozen:
                self._release_sandbox()
            raise
        try:
            try:
                self.runtime.require_work_idle()
            except SubmissionError as error:
                raise RetryableSubmissionError(str(error)) from error
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
            # Freeze the Agent's output while Work is paused: the Judge sees
            # exactly what led to this submission and nothing later.
            trajectory = capture_round_trajectory(
                self.artifacts.root, request.round_id, owner=_sudo_invoking_owner()
            )
            try:
                endpoint = None
                if frozen:
                    endpoint = self.sandbox.prepare_judge(request.round_id)
                    self.sandbox.activate_judge(
                        time.monotonic()
                        + request.run_plan.task.verifier.timeout_seconds
                    )
                result = (
                    self.runtime.run_judge(
                        snapshot, request, environment, trajectory=trajectory
                    )
                    if endpoint is None
                    else self.runtime.run_judge(
                        snapshot,
                        request,
                        environment,
                        sandbox_endpoint=endpoint,
                        trajectory=trajectory,
                    )
                )
            finally:
                release_round_trajectory(trajectory)
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
            observer.resource_event("judge_removed", judge_container_id=judge_id)
            observer.resource_event(
                "snapshot_released", snapshot_lease_id=f"workspace-{request.round_id}"
            )
            return report
        finally:
            # Work that ended (deadline or retired) during the round stays
            # paused for the coordinator to stop.
            if not frozen or self._release_sandbox():
                self.runtime.unpause(request.work_container)
                if frozen:
                    self.sandbox.reopen_work()
                observer.resource_event(
                    "work_unpaused",
                    work_container_id=request.work_container.container_id,
                )

    def _release_sandbox(self) -> bool:
        """End the round's envs, then resume Work's; False when Work ended
        normally meanwhile (nothing to resume)."""
        try:
            self.sandbox.close_judge()
        except Exception as error:
            self.sandbox.contain_work()
            raise InfrastructureError(
                f"recovery_required: sandbox Judge cleanup failed: {error}"
            ) from error
        if not self.sandbox.can_resume:
            if self.sandbox.work_ended_normally:
                return False
            raise InfrastructureError(
                "recovery_required: sandbox Work expired, was cancelled, or "
                "requires recovery; Work family remains contained"
            )
        self.sandbox.resume_work()
        return True


class _ClusterAgentAdapter(RSILoopAgentAdapter):
    @staticmethod
    def _stage_solution(
        runtime: ApptainerAgentRuntime, request: AgentHookRequest
    ) -> None:
        runtime.stage_solution()


class NativeEngineComposition:
    def __init__(
        self,
        payload: Any,
        plan: RunPlan,
        allocated_pools: AllocatedPools | None = None,
        resume: ResumeContext | None = None,
    ) -> None:
        self.payload = payload
        self.plan = plan
        # The next attempt of the run, which keeps its run ID.
        self.resume = resume
        self.attempt = 1 if resume is None else resume.attempt
        self.clock = _Clock()
        self.runtime = ApptainerAgentRuntime(
            payload,
            plan,
            allocated_pools=allocated_pools,
        )
        self.artifacts: RunArtifactWriter | None = None
        config = load_config()
        self.rsi_loop_config = config
        self.agent = _ClusterAgentAdapter(config, runtime=self.runtime)
        self.provider_urls = (
            _agent_provider_urls(payload, plan, config)
            if plan.task.agent.network.mode == "no-network"
            else ()
        )
        self.evaluator: NativeJudgeEvaluator | None = None
        self.volume: ManagedWorkdirVolume | None = None
        self.secrets = rsi_loop_runtime_secret_values(config)
        # E2B envs: the broker runs in this process, its endpoint on node tmp.
        self.sandbox = (
            None
            if plan.sandbox is None
            else ClusterSandbox(plan, payload.run_id, self.runtime.node_tmp / "sb")
        )

    def bind_sandbox(self, mutate: Any) -> None:
        """``on_lease_ready``: start the broker and expose Work's endpoint."""
        assert self.sandbox is not None
        try:
            self.runtime.work_sandbox = self.sandbox.bind(mutate)
        finally:
            self.secrets.update(self.sandbox.secrets)

    def coordinator_options(self) -> dict[str, Any]:
        if self.sandbox is None:
            return {}
        return {
            "on_lease_ready": self.bind_sandbox,
            "sandbox_lifecycle": self.sandbox.lifecycle,
        }

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
            workdir_volume_adopter=self.adopt_volume,
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
        if self.resume is None:
            self.artifacts.start()
        else:
            self.artifacts.reopen(self.resume.attempt)
        self.evaluator = NativeJudgeEvaluator(
            self.runtime,
            self.artifacts,
            None if self.sandbox is None else self.sandbox.lifecycle,
        )
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

    def adopt_volume(
        self, plan: RunPlan, run_id: str, volume: ManagedWorkdirVolume
    ) -> ManagedWorkdirVolume:
        """A resumed run's workspace, which stays where it is."""
        if volume != self.plan_volume(plan, run_id):
            raise SetupError("kept workspace authority differs from the run")
        self.volume = volume
        return volume

    def _require_volume(self, volume: ManagedWorkdirVolume) -> None:
        if volume != self.volume:
            raise SetupError("Work workspace authority differs from its plan")

    def create_work(self, *_args: Any) -> ContainerRef:
        self.runtime.initialize(resume=self.resume is not None)
        self.secrets.update(self.runtime.auth_secret_values)
        access = self.judge_model_access()
        self.runtime.configure_judge_model_access(access)
        if access is not None:
            self.secrets.update(access.secret_values)
        return self.runtime.work_ref

    def judge_model_access(self) -> JudgeModelAccess | None:
        """The run's Agent and the task's Judge LLM, as the Judge reaches them.

        A no-network Agent reaches its model through endpoints the Harness
        pins; so does the Judge (``pinned``), through its own proxy. A public
        Agent reaches it over the site network; so does the Judge (``site``),
        whose egress the site, not the Harness firewall, then governs. A
        Judge with only a Judge LLM follows its Verifier's network instead
        (judge_llm.cluster_judge_egress). The Judge LLM's credential is
        resolved here again, as the submit host checked it.
        """
        task = self.plan.task
        if not judge_needs_rsi_agent(task.verifier):
            return None
        egress = cluster_judge_egress(task)
        enforcer = NetworkPolicyEnforcer(run_id=self.payload.run_id)
        endpoints: tuple[Any, ...] = ()
        extra = None
        if egress == "pinned" and task.verifier.model_access:
            endpoints = enforcer.pin_endpoints(self.provider_urls)
            extra = judge_proxy_environment()
        judge_llm = resolve_cluster_judge_llm(
            task,
            getattr(self.payload, "judge_llm", None),
            agent_auth=self.runtime.auth_material,
            config=self.rsi_loop_config,
            egress=egress,
            # Site egress resolves names in the Judge, as Work does.
            pin=enforcer.pin_endpoints if egress == "pinned" else unresolved_endpoints,
        )
        access = resolve_judge_model_access(
            task,
            self.rsi_loop_config,
            self.runtime.auth_material,
            endpoints,
            judge_llm,
            egress=egress,
            isolation_fallback=self.payload.profile.apptainer.judge_isolation_fallback,
            allow_missing_landlock_signal=(
                self.payload.profile.apptainer.judge_allow_missing_landlock_signal
            ),
            extra_agent_environment=extra,
        )
        assert access is not None
        # One pinned endpoint per name: the Agent and the Judge LLM may share
        # a host, resolved twice.
        unique: dict[tuple[str, int], Any] = {}
        for endpoint in access.endpoints:
            unique.setdefault((endpoint.hostname.lower(), endpoint.port), endpoint)
        access = replace(access, endpoints=tuple(unique.values()))
        if access.needs_codex_cli:
            prepared = getattr(self.payload, "judge_llm", None)
            access = replace(
                access,
                codex_cli=self.runtime.prepare_judge_codex(
                    {} if prepared is None else prepared.codex
                ),
            )
        if judge_llm is not None:
            self._event(
                "judge_llm",
                {
                    "model": judge_llm.model,
                    "effort": judge_llm.effort,
                    "credential": judge_llm.kind.value,
                    "reused": judge_llm.reused,
                    "egress": egress,
                },
            )
        return access

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
                resume=(
                    None
                    if self.resume is None
                    else AgentResume(
                        attempt=self.resume.attempt,
                        message=self.resume.message,
                        agent_output=self.artifacts.root / "agent_output.txt",
                    )
                ),
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
        if self.sandbox is not None:
            environment.update(self.sandbox.work_environment())
        return replace(prepared, environment=tuple(sorted(environment.items())))

    def run_agent(
        self, prepared: Any, work: ContainerRef, timeout: float | None
    ) -> AgentRunResult:
        if self.artifacts is None:
            raise RuntimeError("native artifacts are unavailable")
        if self.sandbox is not None and timeout is None:
            raise SetupError("sandbox requires a finite Work execution timeout")
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
                    on_exec_start=(
                        None
                        if self.sandbox is None
                        else self.sandbox.lifecycle.activate_work
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
        try:
            report = self.evaluator.evaluate(request, observer)
        finally:
            # A renewed Work login: its tokens may show in the Agent's output.
            access = self.runtime.judge_model_access
            if access is not None:
                self.secrets.update(access.secret_values)
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
        round_id = retained_round_id("final", self.attempt)
        digest = hashlib.sha256(
            f"{self.payload.run_id}:{round_id}".encode()
        ).hexdigest()
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
            round_id=retained_round_id("final", self.attempt),
            source_container_id=work.container_id,
            image_id=f"sha256:{digest}",
            image_ref=planned,
        )

    @staticmethod
    def _event(name: str, value: object) -> None:
        print(f"[rsi-engine] {name}: {value}", flush=True)


def validate_native_artifacts(
    plan: RunPlan, run_id: str, *, visibility_timeout: float = 0.0
) -> Path:
    """Require native evidence; the submit host may wait for shared-FS visibility.

    Callers must first confirm the scheduler job succeeded. The compute node
    validates immediately; only the submit host needs the bounded grace period.
    Missing evidence still fails, and malformed evidence is never retried.
    """
    root = plan.paths.logs / "runs" / run_id / plan.task.task_id
    deadline = time.monotonic() + visibility_timeout
    while True:
        # A resumed attempt may have a different first recorded round.
        recorded = recorded_submissions(root)
        first = recorded[0][0].round_id if recorded else "agent-1"
        required = (
            root / "final_result.json",
            root / "agent_output.txt",
            root / "run_agent.log",
            root / "submissions" / first / "report.json",
            root / "feedback" / f"{first}.log",
        )
        missing = tuple(path for path in required if not path.is_file())
        if not missing:
            return root
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise InfrastructureError(
                "native Engine run is missing artifacts: "
                + ", ".join(str(path) for path in missing)
            )
        time.sleep(min(1.0, remaining))


def run_native_engine(
    payload: Any,
    plan: RunPlan,
    *,
    allocated_pools: AllocatedPools | None = None,
) -> None:
    """Compose and execute the same coordinator used by local Harness runs."""
    validate_cluster_sandbox(
        plan.task.sandbox, plan.sandbox, multi_node=allocated_pools is not None
    )
    if (
        payload.resume is None
        and LeaseStore(plan.paths.root / "leases").path_for(payload.run_id).exists()
    ):
        # A rerun of the first attempt's job would start the run over.
        raise SetupError(
            f"run {payload.run_id} already started; rsi-harness resume continues it"
        )
    resume = None if payload.resume is None else _planned_resume(payload, plan)
    composition = NativeEngineComposition(
        payload,
        plan,
        allocated_pools=allocated_pools,
        resume=resume,
    )
    try:
        result = RunCoordinator(
            backend=composition.backend(),
            lease_store=LeaseStore(plan.paths.root / "leases"),
            run_id_factory=lambda: payload.run_id,
            clock=composition.clock,
            **composition.coordinator_options(),
            **({} if resume is None else {"resume": resume}),
        ).run(composition.request())
    finally:
        composition.runtime.close_multinode()
    (plan.paths.root / "engine-result.json").write_text(
        result.model_dump_json(indent=2)
    )
    if plan.task.agent.name == "oracle":
        failures = oracle_failures(result, payload.options.max_submissions or 1)
        if failures:
            raise InfrastructureError("Oracle check failed: " + "; ".join(failures))
    if result.status is not RunStatus.COMPLETED:
        raise InfrastructureError(
            f"native Engine ended in {result.status.value}: "
            f"{result.total_rounds} submissions"
        )
    validate_native_artifacts(plan, payload.run_id)


def _planned_resume(payload: Any, plan: RunPlan) -> ResumeContext:
    """Plan the payload's attempt again from the run's lease and logs, as
    the submit host did after closing the last one."""
    # The job checked the SIF against this sidecar before it started us.
    if payload.sif_sha256_path.read_text().split()[0] != payload.resume.sif_sha256:
        raise SetupError(f"the SIF of run {payload.run_id} was rebuilt since it ran")
    context = plan_resume(
        payload.run_id,
        store=LeaseStore(plan.paths.root / "leases"),
        data_root=plan.paths.root.parent,
        logs_root=plan.paths.logs,
        grants=payload.resume.grants,
        compile_task=lambda request: HarborTaskCompiler().compile(
            request.task_dir, request.options
        ),
        salvage=None,
        retained_images=False,
    )
    if context.salvage_pending or context.attempt != payload.resume.attempt:
        raise SetupError(
            f"run {payload.run_id} changed after attempt "
            f"{payload.resume.attempt} was submitted"
        )
    return context


__all__ = ["ApptainerAgentRuntime", "run_native_engine", "validate_native_artifacts"]
