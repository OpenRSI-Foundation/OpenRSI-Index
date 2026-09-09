"""Blue Vela resource planning and run orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

from rsi_harness.cluster.base import (
    ClusterAdapter,
    ClusterRunRequest,
    ClusterRunResult,
)
from rsi_harness.cluster.bluevela.image import ImagePlan as SIFImagePlan
from rsi_harness.cluster.bluevela.image import (
    plan_image,
    render_build_driver,
    validate_cached_image,
)
from rsi_harness.cluster.bluevela.resources import (
    ClusterResources,
    MultiNodeResources,
    derive_resource_plan,
)
from rsi_harness.cluster.bluevela.resources import derive_resources as derive_resources
from rsi_harness.cluster.config import (
    ClusterProfile,
    load_cluster_profile,
)
from rsi_harness.cluster.schedulers.lsf import (
    LSFJobResult,
    LSFJobSpec,
    LSFScheduler,
)
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    AssetRequirement,
    GPUAllocation,
    GPUDevice,
    ImagePlan,
    JudgeGPUMode,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
    RunPlan,
    RunStatus,
    TaskDefinition,
)
from rsi_harness.runtime.local_auth import resolve_agent_auth
from rsi_harness.runtime.redaction import redact_text
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.agent import get_agent_class


class SchedulerPort(Protocol):
    def render_submit(self, spec: LSFJobSpec) -> tuple[str, ...]: ...

    def require_name_available(self, name: str) -> None: ...

    def submit(self, spec: LSFJobSpec) -> str: ...

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float,
        on_state: Callable[[str], None] | None = None,
    ) -> LSFJobResult: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _safe_component(value: str, *, limit: int) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or "task")[:limit].rstrip("-")


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _agent_launcher(agent_name: str) -> str:
    try:
        command = shlex.split(get_agent_class(agent_name).run_cmd)
    except (AttributeError, TypeError, ValueError) as error:
        raise SetupError(
            f"registered cluster Agent {agent_name!r} has no valid launcher"
        ) from error
    if not command or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", command[0]) is None:
        raise SetupError(
            f"registered cluster Agent {agent_name!r} has no safe launcher"
        )
    return command[0]


def _resolve_agent_version(agent_name: str) -> str | None:
    launcher = _agent_launcher(agent_name)
    executable = shutil.which(launcher)
    if executable is None:
        raise SetupError(f"cluster agent executable is unavailable: {launcher}")
    completed = subprocess.run(
        (executable, "--version"),
        check=False,
        capture_output=True,
        text=True,
    )
    output = f"{completed.stdout}\n{completed.stderr}"
    match = re.search(r"\b(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b", output)
    if completed.returncode != 0 or match is None:
        raise SetupError(f"cannot resolve {agent_name} version from {output.strip()!r}")
    return match.group(1)


class BlueVelaClusterAdapter(ClusterAdapter):
    """Synchronous LSF/Apptainer transport for the native Harness Engine."""

    def __init__(
        self,
        profile: ClusterProfile,
        *,
        scheduler: SchedulerPort | None = None,
        compiler: HarborTaskCompiler | None = None,
        clock: Callable[[], datetime] = _utc_now,
        event_callback: Callable[[str, object], None] | None = None,
        source_root: Path | None = None,
        agent_version_resolver: Callable[[str], str | None] = _resolve_agent_version,
    ) -> None:
        self.profile = profile
        self.scheduler = scheduler or LSFScheduler(
            submit_binary=profile.scheduler.submit_binary,
            status_binary=profile.scheduler.status_binary,
            cancel_binary=profile.scheduler.cancel_binary,
        )
        self.compiler = compiler or HarborTaskCompiler()
        self.clock = clock
        self.event_callback = event_callback or (lambda _name, _value: None)
        self.source_root = (
            Path(source_root).resolve()
            if source_root is not None
            else Path(__file__).resolve().parents[4]
        )
        self.agent_version_resolver = agent_version_resolver

    def run(self, request: ClusterRunRequest) -> ClusterRunResult:
        definition = self._compile(request)
        resource_plan = derive_resource_plan(definition, self.profile)
        resources = resource_plan.single_node or resource_plan.multi_node
        assert resources is not None
        build_context = definition.service.build_context
        if build_context is None:
            raise SetupError("Blue Vela cluster runs require a Docker build context")
        self._validate_runtime_inputs(request)
        agent_version = self.agent_version_resolver(definition.agent.name)
        agent_launcher = _agent_launcher(definition.agent.name)
        raw_agent_binary = shutil.which(agent_launcher)
        agent_binary = (
            None if raw_agent_binary is None else Path(raw_agent_binary).resolve()
        )
        agent_companions = ()
        if definition.agent.name == "codex" and agent_binary is not None:
            code_mode_host = agent_binary.parent / "codex-code-mode-host"
            if code_mode_host.is_file():
                agent_companions = (code_mode_host.resolve(),)

        run_id = self._run_id(definition.task_id)
        run_dir = self.profile.storage.run_root / run_id
        names = {
            stage: self._job_name(
                definition.task_id, stage, definition.agent.name
            )
            for stage in ("build", "run")
        }
        planned_image = plan_image(build_context, self.profile.storage.image_cache)
        build_spec = self._build_spec(
            run_dir, names["build"], resources, run_dir / "control" / "build.sh"
        )
        run_spec = self._run_spec(
            run_dir, names["run"], resources, run_dir / "control" / "run.sh"
        )
        if request.dry_run:
            self.event_callback(
                "dry_run",
                {
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "image": str(planned_image.sif_path),
                    "cache_hit": planned_image.cache_hit,
                    "resources": (
                        None
                        if resource_plan.single_node is None
                        else resource_plan.single_node.model_dump(mode="json")
                    ),
                    "multi_node": (
                        None
                        if resource_plan.multi_node is None
                        else resource_plan.multi_node.model_dump(mode="json")
                    ),
                    "pool_policy": (
                        None
                        if resource_plan.multi_node is None
                        else "ordered Work prefix; ordered Judge suffix"
                    ),
                    "build_argv": self.scheduler.render_submit(build_spec),
                    "run_argv": self.scheduler.render_submit(run_spec),
                    "binds": (
                        *(
                            f"legacy-public:{path}"
                            for path in self.profile.apptainer.extra_binds
                            if resource_plan.single_node is not None
                            and self.profile.apptainer.mount_policy == "legacy"
                        ),
                        *(
                            f"{item.source}:{item.target}"
                            f"{':ro' if item.read_only else ''}"
                            for item in self.profile.apptainer.work_binds
                            if resource_plan.multi_node is not None
                            or self.profile.apptainer.mount_policy == "scoped"
                        ),
                        *(
                            f"{item.source}:{item.target}"
                            f"{':ro' if item.read_only else ''}"
                            for item in self.profile.apptainer.judge_binds
                            if resource_plan.multi_node is not None
                            or self.profile.apptainer.mount_policy == "scoped"
                        ),
                    ),
                    "assets": self._asset_report(
                        definition,
                        resources,
                        require_ready=False,
                    ),
                    "agent_version": agent_version,
                },
            )
            return ClusterRunResult(
                run_id=run_id,
                status=RunStatus.PREPARING,
                log_dir=self.profile.storage.logs_root / "runs" / run_id,
            )

        run_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        (run_dir / "build").mkdir()
        (run_dir / "run").mkdir()
        control = run_dir / "control"
        control.mkdir()
        frozen_task, frozen_source = self._freeze_control(request.task_dir, control)
        definition = self._relocate_definition(definition, frozen_task)
        image = plan_image(
            definition.service.build_context,
            self.profile.storage.image_cache,
        )
        render_build_driver(
            image,
            self.profile,
            definition.service.build_context,
            build_spec.script_path,
            build_spec.local_tmp_mb,
        )
        manifest = self._manifest(
            run_id=run_id,
            run_dir=run_dir,
            definition=definition,
            resources=resources,
            image=image,
            agent_version=agent_version,
        )
        manifest_path = run_dir / "RUN_INFO.json"
        _atomic_json(manifest_path, manifest)
        job_ids: list[str] = []
        try:
            if not image.cache_hit:
                self.scheduler.require_name_available(build_spec.name)
                build_job_id = self.scheduler.submit(build_spec)
                job_ids.append(build_job_id)
                manifest["build_job_id"] = build_job_id
                manifest["state"] = "building"
                _atomic_json(manifest_path, manifest)
                self.event_callback(
                    "job_submitted", {"stage": "build", "job_id": build_job_id}
                )
                build_result = self.scheduler.wait(
                    build_job_id,
                    poll_seconds=self.profile.scheduler.poll_seconds,
                    on_state=lambda state: self.event_callback(
                        "job_state", {"stage": "build", "state": state}
                    ),
                )
                self._require_success("SIF build", build_result)
            if not validate_cached_image(image):
                raise InfrastructureError(
                    "SIF cache is missing or failed SHA256 validation after build"
                )

            manifest["assets"] = self._asset_report(
                definition,
                resources,
                require_ready=True,
            )
            _atomic_json(manifest_path, manifest)

            run_plan = self._run_plan(definition, image, resources, run_dir)
            from rsi_harness.cluster.bluevela.engine import (
                EnginePayload,
                render_engine_driver,
            )
            from rsi_harness.cluster.bluevela.runtime import validate_native_artifacts

            payload = EnginePayload(
                run_id=run_id,
                run_plan=run_plan,
                sif_path=image.sif_path,
                sif_sha256_path=image.sha256_path,
                source_root=frozen_source,
                resources=(
                    resources if isinstance(resources, ClusterResources) else None
                ),
                multi_node=(
                    resources
                    if isinstance(resources, MultiNodeResources)
                    else None
                ),
                profile=self.profile,
                options=request.options,
                agent_auth=request.agent_auth,
                agent_version=agent_version,
                agent_binary=agent_binary,
                agent_launcher=agent_launcher,
                agent_companions=agent_companions,
            )
            render_engine_driver(payload, self.profile, run_spec.script_path)
            manifest["sif_sha256"] = image.sha256_path.read_text().split()[0]
            manifest["control_sha256"] = self._control_hashes(control)
            manifest["state"] = "ready"
            _atomic_json(manifest_path, manifest)

            if isinstance(resources, MultiNodeResources):
                self._require_shared_workspace(run_dir, resources)
            self.scheduler.require_name_available(run_spec.name)
            run_job_id = self.scheduler.submit(run_spec)
            job_ids.append(run_job_id)
            manifest["run_job_id"] = run_job_id
            manifest["state"] = "running"
            _atomic_json(manifest_path, manifest)
            self.event_callback(
                "job_submitted", {"stage": "run", "job_id": run_job_id}
            )
            run_result = self.scheduler.wait(
                run_job_id,
                poll_seconds=self.profile.scheduler.poll_seconds,
                on_state=lambda state: self.event_callback(
                    "job_state", {"stage": "run", "state": state}
                ),
            )
            self._require_success("native Engine run", run_result)
            log_dir = validate_native_artifacts(run_plan, run_id)
            manifest["state"] = "completed"
            _atomic_json(manifest_path, manifest)
            return ClusterRunResult(
                run_id=run_id,
                status=RunStatus.COMPLETED,
                log_dir=log_dir,
                job_ids=tuple(job_ids),
            )
        except BaseException as error:
            manifest["state"] = "failed"
            manifest["error"] = redact_text(str(error))
            manifest["job_ids"] = job_ids
            _atomic_json(manifest_path, manifest)
            raise

    def _compile(self, request: ClusterRunRequest) -> TaskDefinition:
        definition = self.compiler.compile(request.task_dir, request.options)
        updates: dict[str, str] = {}
        if request.model is not None:
            updates["model"] = request.model
        if request.reasoning_effort is not None:
            updates["reasoning_effort"] = request.reasoning_effort
        if updates:
            definition = definition.model_copy(
                update={"agent": definition.agent.model_copy(update=updates)}
            )
        return definition

    def _validate_runtime_inputs(self, request: ClusterRunRequest) -> None:
        if request.agent_name == "codex" and request.model is None:
            raise SetupError("cluster Codex runs require an explicit --model")
        if request.agent_auth is not None and request.agent_auth.value == "local":
            resolve_agent_auth(
                source=request.agent_auth,
                agent_name=request.agent_name,
                agent_api_key=None,
            )
        launcher = _agent_launcher(request.agent_name)
        if shutil.which(launcher) is None:
            raise SetupError(
                f"cluster agent executable is unavailable: {launcher}"
            )
        for label, path in (("Apptainer", self.profile.apptainer.binary),):
            if not path.is_file():
                raise SetupError(f"{label} binary does not exist: {path}")

    def _run_id(self, task_id: str) -> str:
        timestamp = self.clock().astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
        task = _safe_component(task_id, limit=36)
        return f"{timestamp}-{task}-{secrets.token_hex(4)}"

    def _job_name(self, task_id: str, stage: str, agent_name: str) -> str:
        task = _safe_component(task_id, limit=36)
        agent = _safe_component(agent_name, limit=20)
        owner = _safe_component(self.profile.owner, limit=20)
        return f"mt-rsi-{task}-{agent}-{stage}-{owner}"

    def _build_spec(
        self,
        run_dir: Path,
        name: str,
        resources: ClusterResources | MultiNodeResources,
        script: Path,
    ) -> LSFJobSpec:
        return LSFJobSpec(
            name=name,
            queue=self.profile.scheduler.queue,
            group=self.profile.scheduler.group,
            cpu_slots=self.profile.builder.cpu_slots,
            memory_mb=self.profile.builder.memory_mb,
            walltime=resources.build_walltime,
            stdout_path=(run_dir / "build" / "lsf.%J.out").resolve(),
            stderr_path=(run_dir / "build" / "lsf.%J.err").resolve(),
            script_path=script.resolve(),
            local_tmp_mb=max(
                self.profile.builder.min_tmp_mb,
                (
                    resources.local_tmp_mb
                    if isinstance(resources, ClusterResources)
                    else resources.node_tmp_mb
                ),
            ),
            excluded_hosts=self.profile.scheduler.excluded_hosts,
        )

    def _run_spec(
        self,
        run_dir: Path,
        name: str,
        resources: ClusterResources | MultiNodeResources,
        script: Path,
    ) -> LSFJobSpec:
        if isinstance(resources, MultiNodeResources):
            return LSFJobSpec(
                name=name,
                queue=self.profile.scheduler.queue,
                group=self.profile.scheduler.group,
                cpu_slots=(
                    resources.total_nodes * resources.cpu_slots_per_node
                ),
                memory_mb=resources.memory_mb_per_node,
                walltime=resources.run_walltime,
                stdout_path=(run_dir / "run" / "lsf.%J.out").resolve(),
                stderr_path=(run_dir / "run" / "lsf.%J.err").resolve(),
                script_path=script.resolve(),
                gpu_count=resources.gpus_per_node,
                local_tmp_mb=resources.node_tmp_mb,
                one_host=False,
                hosts=resources.total_nodes,
                slots_per_host=resources.cpu_slots_per_node,
                memory_per_host=True,
                exclusive=self.profile.scheduler.exclusive,
                excluded_hosts=self.profile.scheduler.excluded_hosts,
            )
        return LSFJobSpec(
            name=name,
            queue=self.profile.scheduler.queue,
            group=self.profile.scheduler.group,
            cpu_slots=resources.cpu_slots,
            memory_mb=resources.memory_mb,
            walltime=resources.run_walltime,
            stdout_path=(run_dir / "run" / "lsf.%J.out").resolve(),
            stderr_path=(run_dir / "run" / "lsf.%J.err").resolve(),
            script_path=script.resolve(),
            gpu_count=resources.total_gpus,
            local_tmp_mb=resources.local_tmp_mb,
            excluded_hosts=self.profile.scheduler.excluded_hosts,
        )

    @staticmethod
    def _require_shared_workspace(
        path: Path,
        resources: MultiNodeResources,
    ) -> None:
        required_bytes = resources.shared_workspace_mb * 1024 * 1024
        if required_bytes == 0:
            return
        free_bytes = shutil.disk_usage(path).free
        if free_bytes < required_bytes:
            raise InfrastructureError(
                "insufficient shared GPFS workspace: "
                f"need {resources.shared_workspace_mb} MiB, "
                f"have {free_bytes // (1024 * 1024)} MiB"
            )

    def _asset_host_path(
        self,
        requirement: AssetRequirement,
        resources: ClusterResources | MultiNodeResources,
    ) -> Path:
        container_path = requirement.path
        if (
            isinstance(resources, MultiNodeResources)
            or self.profile.apptainer.mount_policy == "scoped"
        ):
            bindings = (
                self.profile.apptainer.work_binds
                if requirement.phase == "work"
                else self.profile.apptainer.judge_binds
            )
            matching = tuple(
                binding
                for binding in bindings
                if container_path == binding.target
                or binding.target in container_path.parents
            )
            if matching:
                binding = max(matching, key=lambda item: len(item.target.parts))
                relative = container_path.relative_to(binding.target)
                return binding.source / Path(*relative.parts)
        else:
            host_path = Path(str(container_path))
            for root in self.profile.apptainer.extra_binds:
                if host_path == root or root in host_path.parents:
                    return host_path
        raise SetupError(
            "declared cluster asset is not covered by its phase bind: "
            f"{requirement.phase} {container_path}"
        )

    def _asset_report(
        self,
        definition: TaskDefinition,
        resources: ClusterResources | MultiNodeResources,
        *,
        require_ready: bool,
    ) -> list[dict[str, object]]:
        report: list[dict[str, object]] = []
        failures: list[str] = []
        for requirement in definition.assets:
            host_path = self._asset_host_path(requirement, resources)
            ready = False
            detail = "missing"
            if requirement.kind == "file" and host_path.is_file():
                size = host_path.stat().st_size
                ready = size >= requirement.min_bytes
                detail = f"size={size}"
            elif requirement.kind == "directory" and host_path.is_dir():
                entries = sum(1 for _item in host_path.iterdir())
                ready = entries >= requirement.min_entries
                detail = f"entries={entries}"
            report.append(
                {
                    **requirement.model_dump(mode="json"),
                    "host_path": str(host_path),
                    "ready": ready,
                    "detail": detail,
                }
            )
            if not ready:
                failures.append(f"{requirement.phase}:{host_path} ({detail})")
        if require_ready and failures:
            raise SetupError(
                "cluster data preparation is incomplete; refusing GPU submission: "
                + "; ".join(failures)
            )
        return report

    def _freeze_control(self, task_dir: Path, control: Path) -> tuple[Path, Path]:
        frozen_task = control / "task"
        shutil.copytree(task_dir, frozen_task, symlinks=True)
        source = self.source_root / "src"
        if not source.is_dir():
            raise SetupError(f"RSI-Harness source tree is unavailable: {source}")
        frozen_source = control / "engine-source"
        shutil.copytree(
            source,
            frozen_source / "src",
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        (control / "profile.json").write_text(
            self.profile.model_dump_json(indent=2)
        )
        _atomic_json(control / "source.json", self._source_metadata())
        return frozen_task.resolve(), frozen_source.resolve()

    def _source_metadata(self) -> dict[str, object]:
        def git(*args: str) -> subprocess.CompletedProcess[bytes]:
            return subprocess.run(
                ("git", "-C", str(self.source_root), *args),
                check=False,
                capture_output=True,
            )

        commit = git("rev-parse", "HEAD")
        diff = git("diff", "--binary", "HEAD")
        return {
            "commit": commit.stdout.decode(errors="replace").strip()
            if commit.returncode == 0
            else None,
            "dirty": bool(diff.stdout),
            "diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
        }

    @staticmethod
    def _relocate_definition(
        definition: TaskDefinition,
        frozen_task: Path,
    ) -> TaskDefinition:
        build_context = frozen_task / "environment"
        service = definition.service.model_copy(
            update={"build_context": build_context}
        )
        return definition.model_copy(
            update={"source_dir": frozen_task, "service": service}
        )

    def _run_plan(
        self,
        definition: TaskDefinition,
        image: SIFImagePlan,
        resources: ClusterResources | MultiNodeResources,
        run_dir: Path,
    ) -> RunPlan:
        if isinstance(resources, MultiNodeResources):
            work_devices = tuple(
                GPUDevice(
                    index=node_rank * resources.gpus_per_node + local_rank,
                    uuid=f"WORK-{node_rank:03d}:GPU-{local_rank}",
                    name="planned Blue Vela Work GPU",
                )
                for node_rank in range(resources.work.node_count)
                for local_rank in range(resources.gpus_per_node)
            )
            judge_offset = len(work_devices)
            judge_devices = tuple(
                GPUDevice(
                    index=(
                        judge_offset
                        + node_rank * resources.gpus_per_node
                        + local_rank
                    ),
                    uuid=f"JUDGE-{node_rank:03d}:GPU-{local_rank}",
                    name="planned Blue Vela Judge GPU",
                )
                for node_rank in range(resources.verifier.node_count)
                for local_rank in range(resources.gpus_per_node)
            )
            devices = work_devices + judge_devices
            work = GPUAllocation(devices=work_devices)
            verifier = GPUAllocation(devices=judge_devices)
            mode = (
                JudgeGPUMode.DISJOINT
                if judge_devices
                else JudgeGPUMode.FREEZE_ONLY
            )
        else:
            devices = tuple(
                GPUDevice(
                    index=index,
                    uuid=f"LSF-{index}",
                    name="LSF allocated GPU",
                )
                for index in range(resources.total_gpus)
            )
            work = GPUAllocation(devices=devices[: resources.work_gpus])
            spares = devices[resources.work_gpus :]
            if resources.verifier_gpus == 0:
                verifier = GPUAllocation()
                mode = JudgeGPUMode.FREEZE_ONLY
            elif resources.verifier_gpus <= len(spares):
                verifier = GPUAllocation(
                    devices=spares[: resources.verifier_gpus]
                )
                mode = JudgeGPUMode.DISJOINT
            else:
                verifier = GPUAllocation(
                    devices=(spares + work.devices)[: resources.verifier_gpus]
                )
                mode = JudgeGPUMode.RELEASE_ALL
        workdir = (
            definition.workdir
            or definition.service.workdir
            or self.profile.apptainer.workspace_target
        )
        if workdir == PurePosixPath("/"):
            workdir = self.profile.apptainer.workspace_target
        runtime_image = ImagePlan(
            base_ref=str(image.sif_path),
            work_ref=str(image.sif_path),
            judge_ref=str(image.sif_path),
            workdir=workdir,
            rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            base_digest=f"sha256:{image.cache_key}",
            work_digest=f"sha256:{image.cache_key}",
            judge_digest=f"sha256:{image.cache_key}",
        )
        return self.compiler.finalize(
            definition,
            runtime_image,
            RunGPUPlan(
                authorized_pool=GPUAllocation(devices=devices),
                work=work,
                judge=verifier,
                judge_mode=mode,
            ),
            RunPaths(
                root=run_dir.resolve(),
                workspace=(run_dir / "workspace").resolve(),
                logs=self.profile.storage.logs_root.resolve(),
            ),
            "bluevela-apptainer-engine",
        )

    def _manifest(
        self,
        *,
        run_id: str,
        run_dir: Path,
        definition: TaskDefinition,
        resources: ClusterResources | MultiNodeResources,
        image: SIFImagePlan,
        agent_version: str | None,
    ) -> dict[str, object]:
        leaf = (
            self.profile.storage.logs_root
            / "runs"
            / run_id
            / definition.task_id
        )
        return {
            "run_id": run_id,
            "owner": self.profile.owner,
            "purpose": f"RSI Harness task {definition.task_id}",
            "submitted_at_utc": self.clock().astimezone(UTC).isoformat(),
            "run_dir": str(run_dir),
            "build_job_id": None,
            "run_job_id": None,
            "source": self._source_metadata(),
            "sif_path": str(image.sif_path),
            "sif_sha256": None,
            "resources": (
                resources.model_dump(mode="json")
                if isinstance(resources, ClusterResources)
                else None
            ),
            "multi_node": (
                resources.model_dump(mode="json")
                if isinstance(resources, MultiNodeResources)
                else None
            ),
            "agent_version": agent_version,
            "expected_outputs": [
                str(leaf / "final_result.json"),
                str(leaf / "agent_output.txt"),
                str(leaf / "run_agent.log"),
                str(leaf / "submissions" / "agent-1" / "report.json"),
            ],
            "state": "prepared",
        }

    @staticmethod
    def _control_hashes(control: Path) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for path in sorted(control.rglob("*")):
            if path.is_symlink():
                content = os.readlink(path).encode()
            elif path.is_file():
                content = path.read_bytes()
            else:
                continue
            hashes[path.relative_to(control).as_posix()] = hashlib.sha256(
                content
            ).hexdigest()
        return hashes

    @staticmethod
    def _require_success(stage: str, result: LSFJobResult) -> None:
        if result.state != "DONE" or result.exit_code != 0:
            raise InfrastructureError(
                f"{stage} job {result.job_id} ended in {result.state} "
                f"with exit {result.exit_code}"
            )


def build_cluster_adapter(
    name_or_path: str | Path,
    *,
    event_callback: Callable[[str, object], None] | None = None,
) -> ClusterAdapter:
    profile = load_cluster_profile(name_or_path)
    if profile.adapter != "bluevela":
        raise SetupError(f"unsupported cluster adapter: {profile.adapter}")
    return BlueVelaClusterAdapter(profile, event_callback=event_callback)
