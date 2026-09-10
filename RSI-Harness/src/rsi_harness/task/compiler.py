"""Compile a supported Harbor GPU task into immutable engine contracts."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

from harbor.models.task.config import TaskOS, VerifierEnvironmentMode
from harbor.models.task.task import Task
from harbor.models.task.verifier_mode import resolve_task_verifier_mode
from harbor.utils.env import get_required_host_vars
from pydantic import ValidationError

from rsi_harness.errors import UnsupportedTaskError
from rsi_harness.models import (
    AgentPlan,
    AssetRequirement,
    CompileOptions,
    GPURequirement,
    ImagePlan,
    MainServiceConfig,
    NetworkPolicy,
    RunGPUPlan,
    RunPaths,
    RunPlan,
    TaskDefinition,
    VerifierPlan,
)
from rsi_harness.task.compose import ComposeMainServiceParser
from rsi_harness.task.digest import hash_tree


class HarborTaskCompiler:
    """Validate and translate the engine's intentionally narrow Harbor subset."""

    def __init__(self, compose_parser: ComposeMainServiceParser | None = None) -> None:
        self._compose_parser = compose_parser or ComposeMainServiceParser()

    def compile(self, task_dir: Path, options: CompileOptions) -> TaskDefinition:
        source_dir = Path(task_dir).resolve()
        source_digest = hash_tree(source_dir)
        try:
            task = Task(source_dir, disable_verification=True)
            # Harbor coerces booleans, floats, and strings into integer counts.
            # Preserve the task author's explicit integer resource declaration.
            raw_environment = tomllib.loads(task.paths.config_path.read_text()).get(
                "environment", {}
            )
            if "gpus" in raw_environment:
                raw_count = raw_environment["gpus"]
                if type(raw_count) is not int or raw_count < 0:
                    raise UnsupportedTaskError(
                        "environment.gpus must be a non-negative integer"
                    )
        except (OSError, ValidationError, ValueError) as error:
            raise UnsupportedTaskError(
                f"invalid or missing required Harbor task file: {error}"
            ) from error

        self._reject_unsupported_config(task)
        test_path = self._require_files(task)
        compose_path = task.paths.environment_dir / "docker-compose.yaml"
        compose_service = (
            self._compose_parser.parse(compose_path)
            if compose_path.exists()
            else MainServiceConfig()
        )
        try:
            service, workdir = self._service(task, compose_service)
            gpu_requirement = self._gpu_requirement(task, compose_service)
            if service.shm_size is None:
                service = service.model_copy(update={"shm_size": "1g"})
            agent_environment = service.environment
            verifier_environment = self._merge_environment(
                agent_environment, task.config.verifier.env
            )
            definition = TaskDefinition(
                task_id=task.short_name,
                instruction=task.instruction,
                source_dir=task.task_dir,
                source_digest=source_digest,
                instruction_digest=hashlib.sha256(task.instruction.encode()).hexdigest(),
                tests_digest=hash_tree(task.paths.tests_dir),
                environment_digest=hash_tree(task.paths.environment_dir),
                workdir=workdir,
                service=service,
                gpu_requirement=gpu_requirement,
                verifier=VerifierPlan(
                    command=(
                        "/bin/bash",
                        f"/tests/{test_path.relative_to(task.paths.tests_dir).as_posix()}",
                    ),
                    timeout_seconds=task.config.verifier.timeout_sec,
                    user=self._user(task.config.verifier.user),
                    environment=verifier_environment,
                    secret_env_names=self._template_names(verifier_environment),
                    primary_reward=options.primary_reward,
                    network=self._network_policy(task, task.config.verifier),
                    gpu_count=self._verifier_gpu_count(task),
                ),
                agent=AgentPlan(
                    name=options.agent_name,
                    timeout_seconds=(
                        options.agent_timeout_seconds
                        if options.agent_timeout_seconds is not None
                        else (
                            60.0
                            if task.config.agent.timeout_sec is None
                            else task.config.agent.timeout_sec
                        )
                    ),
                    user=self._user(task.config.agent.user),
                    environment=agent_environment,
                    secret_env_names=self._template_names(agent_environment),
                    network=self._network_policy(task, task.config.agent),
                    install_stop_hook=not options.disable_stop_hook,
                ),
                assets=self._asset_requirements(task),
                require_disjoint_phase_nodes=self._require_disjoint_phase_nodes(task),
                score_direction=options.score_direction,
            )
        except ValidationError as error:
            raise UnsupportedTaskError(
                f"invalid Harbor WORKDIR or plan: {error}"
            ) from error
        return definition

    def finalize(
        self,
        definition: TaskDefinition,
        images: ImagePlan,
        gpu_plan: RunGPUPlan,
        paths: RunPaths,
        snapshot_kind: str,
    ) -> RunPlan:
        """Attach caller-owned runtime selections without accepting secrets."""
        effective_service = definition.service.model_copy(
            update={"workdir": images.workdir}
        )
        effective_task = definition.model_copy(
            update={"workdir": images.workdir, "service": effective_service}
        )
        return RunPlan(
            schema_version=2,
            task=effective_task,
            workdir=images.workdir,
            rootfs_snapshot_mode=images.rootfs_snapshot_mode,
            images=images,
            gpu_plan=gpu_plan,
            paths=paths,
            snapshot_kind=snapshot_kind,
        )

    def write_rsi_loop_metadata(
        self,
        definition: TaskDefinition,
        destination: Path,
    ) -> Path:
        """Write archive-free RSI Loop display metadata outside the source task."""
        target_dir = Path(destination).resolve()
        if (
            target_dir == definition.source_dir
            or definition.source_dir in target_dir.parents
        ):
            raise UnsupportedTaskError(
                "RSI Loop metadata must not be written into the source task"
            )
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "task.json"
        payload = {
            "task_id": definition.task_id,
            "name": definition.task_id,
            "cwd": definition.workdir.as_posix(),
            "submit_paths": ["."],
            "judge": {
                "parser": "structured_json",
                "score_direction": definition.score_direction,
                "selection": "score_first",
            },
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return target

    def _reject_unsupported_config(self, task: Task) -> None:
        config = task.config
        if config.schema_version != "1.4":
            raise UnsupportedTaskError(
                f"unsupported Harbor schema version {config.schema_version!r}"
            )
        if config.steps:
            raise UnsupportedTaskError("Harbor multi-step tasks are unsupported")
        if config.environment.os == TaskOS.WINDOWS:
            raise UnsupportedTaskError("Windows Harbor tasks are unsupported")
        if resolve_task_verifier_mode(config) == VerifierEnvironmentMode.SEPARATE:
            raise UnsupportedTaskError(
                "independent verifier environments are unsupported"
            )
        if config.environment.tpu is not None:
            raise UnsupportedTaskError("TPU tasks are unsupported")
        if config.environment.mcp_servers:
            raise UnsupportedTaskError("MCP servers are unsupported")
        if config.verifier.collect:
            raise UnsupportedTaskError("verifier collect hooks are unsupported")
        if config.environment.healthcheck is not None:
            raise UnsupportedTaskError("environment healthchecks are unsupported")
        if config.environment.skills_dir is not None:
            raise UnsupportedTaskError("Harbor skills directories are unsupported")
        if config.environment.gpu_types is not None:
            if len(config.environment.gpu_types) != 1:
                raise UnsupportedTaskError(
                    "multiple acceptable GPU types are unsupported"
                )
            if not config.environment.gpu_types[0]:
                raise UnsupportedTaskError("empty GPU types are unsupported")
        if config.multi_step_reward_strategy is not None:
            raise UnsupportedTaskError("multi-step reward strategies are unsupported")
        if config.solution.env:
            raise UnsupportedTaskError("solution environment values are unsupported")
        if config.artifacts:
            for artifact in config.artifacts:
                if not isinstance(artifact, str) and artifact.service not in (
                    None,
                    "main",
                ):
                    raise UnsupportedTaskError(
                        "sidecar artifact collection is unsupported"
                    )
            raise UnsupportedTaskError("Harbor artifact collection is unsupported")

    def _require_files(self, task: Task) -> Path:
        paths = task.paths
        required = (paths.config_path, paths.instruction_path, paths.environment_dir)
        missing = next((path for path in required if not path.exists()), None)
        if missing is not None:
            raise UnsupportedTaskError(f"missing required Harbor task path: {missing}")
        test_path = paths.discovered_test_path_for(task.config.environment.os)
        if test_path is None:
            raise UnsupportedTaskError("missing required Harbor verifier test script")
        compose_path = paths.environment_dir / "docker-compose.yaml"
        dockerfile_path = paths.environment_dir / "Dockerfile"
        if (
            not compose_path.exists()
            and not dockerfile_path.exists()
            and task.config.environment.docker_image is None
        ):
            raise UnsupportedTaskError("missing required Harbor environment definition")
        return test_path

    def _service(
        self, task: Task, compose: MainServiceConfig
    ) -> tuple[MainServiceConfig, Any]:
        config_workdir = task.config.environment.workdir
        if (
            config_workdir is not None
            and compose.workdir is not None
            and str(compose.workdir) != config_workdir
        ):
            raise UnsupportedTaskError(
                "Harbor and Compose WORKDIR declarations must match"
            )
        workdir = compose.workdir or config_workdir
        dockerfile = task.paths.environment_dir / "Dockerfile"
        harbor_image = task.config.environment.docker_image
        if compose.build_context is not None and harbor_image is not None:
            raise UnsupportedTaskError(
                "Compose build conflicts with the Harbor image declaration"
            )
        if (
            compose.image is not None
            and harbor_image is not None
            and compose.image != harbor_image
        ):
            raise UnsupportedTaskError(
                "Compose and Harbor image declarations must match"
            )
        image = compose.image or harbor_image
        build_context = compose.build_context
        if build_context is None and image is None and dockerfile.exists():
            build_context = task.paths.environment_dir.resolve()
        environment = self._merge_environment(
            compose.environment, task.config.environment.env
        )
        baseline = task.config.environment.resolve_baseline()
        return (
            MainServiceConfig(
                build_context=build_context,
                image=image,
                workdir=workdir,
                user=compose.user,
                environment=environment,
                shm_size=compose.shm_size,
                cpus=task.config.environment.cpus,
                memory_mb=task.config.environment.memory_mb,
                storage_mb=task.config.environment.storage_mb,
                build_timeout_seconds=task.config.environment.build_timeout_sec,
                network_mode=baseline.network_mode.value,
                gpu_requirement=compose.gpu_requirement,
            ),
            workdir,
        )

    def _gpu_requirement(
        self, task: Task, compose: MainServiceConfig
    ) -> GPURequirement:
        config_count = task.config.environment.gpus
        compose_count = (
            compose.gpu_requirement.count
            if compose.gpu_requirement is not None
            else None
        )
        if (
            config_count is not None
            and compose_count is not None
            and config_count != compose_count
        ):
            raise UnsupportedTaskError(
                "Harbor and Compose GPU count declarations must match"
            )
        count = compose_count if compose_count is not None else config_count
        if count is None:
            raise UnsupportedTaskError(
                "a Harbor GPU requirement is required; "
                "set environment.gpus = 0 for CPU Work"
            )
        gpu_types = task.config.environment.gpu_types
        if count == 0 and gpu_types is not None:
            raise UnsupportedTaskError(
                "environment.gpus = 0 cannot declare gpu_types"
            )
        return GPURequirement(
            count=count,
            name=gpu_types[0] if gpu_types else None,
        )

    @staticmethod
    def _verifier_gpu_count(task: Task) -> int:
        metadata = task.config.metadata
        if "rsi_harness" not in metadata:
            return 0
        rsi_harness = metadata["rsi_harness"]
        if not isinstance(rsi_harness, dict):
            raise UnsupportedTaskError(
                "metadata.rsi_harness.verifier.gpus must be an integer"
            )
        if "verifier" not in rsi_harness:
            return 0
        verifier = rsi_harness["verifier"]
        if not isinstance(verifier, dict):
            raise UnsupportedTaskError(
                "metadata.rsi_harness.verifier.gpus must be an integer"
            )
        if set(verifier) - {"gpus"}:
            raise UnsupportedTaskError("unknown verifier GPU metadata")
        value = verifier.get("gpus", 0)
        if type(value) is not int or value < 0:
            raise UnsupportedTaskError(
                "metadata.rsi_harness.verifier.gpus must be a non-negative integer"
            )
        return value

    @staticmethod
    def _asset_requirements(task: Task) -> tuple[AssetRequirement, ...]:
        metadata = task.config.metadata
        rsi_harness = metadata.get("rsi_harness")
        if rsi_harness is None:
            return ()
        if not isinstance(rsi_harness, dict):
            raise UnsupportedTaskError(
                "metadata.rsi_harness.assets must be a list of tables"
            )
        raw_assets = rsi_harness.get("assets", ())
        if not isinstance(raw_assets, (list, tuple)):
            raise UnsupportedTaskError(
                "metadata.rsi_harness.assets must be a list of tables"
            )
        try:
            return tuple(AssetRequirement.model_validate(item) for item in raw_assets)
        except (ValidationError, TypeError) as error:
            raise UnsupportedTaskError(
                f"invalid metadata.rsi_harness.assets: {error}"
            ) from error

    @staticmethod
    def _require_disjoint_phase_nodes(task: Task) -> bool:
        metadata = task.config.metadata
        rsi_harness = metadata.get("rsi_harness")
        if rsi_harness is None:
            return False
        if not isinstance(rsi_harness, dict):
            raise UnsupportedTaskError(
                "metadata.rsi_harness.require_disjoint_phase_nodes must be a boolean"
            )
        value = rsi_harness.get("require_disjoint_phase_nodes", False)
        if type(value) is not bool:
            raise UnsupportedTaskError(
                "metadata.rsi_harness.require_disjoint_phase_nodes must be a boolean"
            )
        return value

    @staticmethod
    def _merge_environment(
        base: tuple[tuple[str, str], ...], override: dict[str, str]
    ) -> tuple[tuple[str, str], ...]:
        values = dict(base)
        values.update(override)
        return tuple(sorted(values.items()))

    @staticmethod
    def _template_names(environment: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
        names = (name for name, _ in get_required_host_vars(dict(environment)))
        return tuple(sorted(set(names)))

    @staticmethod
    def _user(user: str | int | None) -> str | None:
        return None if user is None else str(user)

    @staticmethod
    def _network_policy(task: Task, phase: Any) -> NetworkPolicy:
        harbor_policy = phase.explicit_phase_policy()
        if harbor_policy is None:
            harbor_policy = task.config.environment.resolve_baseline()
        return NetworkPolicy(
            mode=harbor_policy.network_mode.value,
            allowlist=tuple(harbor_policy.allowed_hosts),
        )
