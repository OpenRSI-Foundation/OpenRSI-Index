"""Slurm scheduling for the same image, resource, and Work/Judge lifecycle."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import (
    BlueVelaClusterAdapter,
    job_log_detail,
)
from rsi_harness.cluster.bluevela.resources import (
    ClusterResources,
    MultiNodeResources,
    _lsf_walltime_seconds,
    derive_resource_plan,
)
from rsi_harness.cluster.config import SlurmSchedulerProfile
from rsi_harness.cluster.schedulers.slurm import (
    SlurmJobResult,
    SlurmJobSpec,
    SlurmScheduler,
)
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import TaskDefinition

if TYPE_CHECKING:
    from rsi_harness.cluster.bluevela.engine import EnginePayload
    from rsi_harness.runtime.resume import ResumeContext

# Slurm documents that advance signals may arrive up to 60 seconds early.
_SIGNAL_EARLY_SECONDS = 60


class SlurmClusterAdapter(BlueVelaClusterAdapter):
    """Reuse Blue Vela orchestration with Slurm job and launch semantics."""

    scheduler_log_pattern = "slurm.%j"
    supports_cpu_work = True

    def _compile(self, request: ClusterRunRequest) -> TaskDefinition:
        definition = super()._compile(request)
        if definition.gpu_requirement.count == 0 and (
            definition.require_disjoint_phase_nodes
            or definition.verifier.gpu_count > self.profile.resources.gpus_per_node
        ):
            raise SetupError("CPU Work requires a single-node Slurm allocation")
        resources = derive_resource_plan(definition, self.profile)
        self._validate_segment_budget(
            definition,
            agent_seconds=definition.agent.timeout_seconds,
            attempt=1,
            multi_node=resources.multi_node is not None,
        )
        return definition

    def _validate_segment_budget(
        self,
        definition: TaskDefinition,
        *,
        agent_seconds: float,
        attempt: int,
        multi_node: bool = False,
    ) -> None:
        """Check the declared allocation plan, not predicted task performance.

        The final allocation needs no advance signal. Earlier allocations
        reserve Judge drain, configured overhead and Slurm's early delivery
        window. Startup delays and extra interruptions can still consume
        attempts; each resume checks the actual remaining budget again.
        """
        limit = self.profile.scheduler.run_walltime_limit
        if limit is None:
            return
        # Recovery imports the Docker SDK; keep profile/CLI discovery usable
        # on submit hosts that do not load a runtime yet.
        from rsi_harness.runtime.resume import MAX_ATTEMPTS

        limit_seconds = _lsf_walltime_seconds(limit)
        drain_seconds = (
            definition.verifier.timeout_seconds
            + self.profile.resources.walltime_margin_seconds
        )
        # Only the advance signal needs whole seconds. The final job's
        # duration follows resource planning, which rounds the combined sum.
        final_capacity = limit_seconds - drain_seconds
        if agent_seconds <= final_capacity:
            return
        if definition.agent.name == "oracle":
            raise SetupError(
                "run exceeds run_walltime_limit; Oracle cannot use segmented "
                "Slurm runs. Use a partition with enough walltime or "
                "explicitly revise the task's budget."
            )
        if multi_node:
            raise SetupError(
                "multi-node runs cannot resume; run_walltime_limit is too short"
            )
        reserve = math.ceil(drain_seconds)
        segment_capacity = limit_seconds - reserve - _SIGNAL_EARLY_SECONDS
        if not 0 < reserve <= 65535 or segment_capacity <= 0:
            raise SetupError(
                "run_walltime_limit leaves no Work interval after the Judge "
                "drain reserve and Slurm signal margin (or reserve is unavailable); "
                "use a longer allocation or a fresh task plan"
            )
        attempts_left = MAX_ATTEMPTS - attempt + 1
        needed = 1 + math.ceil((agent_seconds - final_capacity) / segment_capacity)
        if needed > attempts_left:
            raise SetupError(
                f"remaining Agent budget needs {needed} planned Slurm allocations "
                f"but only {attempts_left} of {MAX_ATTEMPTS} attempts remain; "
                "use a longer run_walltime_limit or explicitly revise the budget"
            )

    def _make_scheduler(self) -> SlurmScheduler:
        scheduler = self.profile.scheduler
        assert isinstance(scheduler, SlurmSchedulerProfile)
        return SlurmScheduler(
            submit_binary=scheduler.submit_binary,
            status_binary=scheduler.status_binary,
            cancel_binary=scheduler.cancel_binary,
            accounting_binary=scheduler.accounting_binary,
        )

    def _job_spec(self, **values: Any) -> SlurmJobSpec:
        scheduler = self.profile.scheduler
        assert isinstance(scheduler, SlurmSchedulerProfile)
        return SlurmJobSpec(
            **values,
            qos=scheduler.qos,
            constraint=scheduler.constraint,
            request_tmp=scheduler.request_tmp,
            gpu_option=scheduler.gpu_option,
        )

    def _build_spec(
        self, run_dir: Path, name: str, resources: Any, script: Path
    ) -> SlurmJobSpec:
        spec = super()._build_spec(run_dir, name, resources, script)
        overrides = self.profile.scheduler.build
        if overrides is None:
            return spec
        values = spec.model_dump()
        for source, target in (
            ("partition", "queue"),
            ("account", "group"),
            ("qos", "qos"),
            ("constraint", "constraint"),
        ):
            value = getattr(overrides, source)
            if value is not None:
                values[target] = value
        values.update(gpu_count=overrides.gpus, exclusive=overrides.exclusive)
        return SlurmJobSpec.model_validate(values)

    def _run_spec(
        self,
        run_dir: Path,
        name: str,
        resources: ClusterResources | MultiNodeResources,
        script: Path,
        *,
        judge_drain_seconds: int | None = None,
    ) -> SlurmJobSpec:
        spec = super()._run_spec(run_dir, name, resources, script)
        limit = self.profile.scheduler.run_walltime_limit
        if limit is None or _lsf_walltime_seconds(
            spec.walltime
        ) <= _lsf_walltime_seconds(limit):
            return spec
        if isinstance(resources, MultiNodeResources):
            raise SetupError(
                "multi-node runs cannot resume; run_walltime_limit is too short"
            )
        reserve = judge_drain_seconds or 0
        # Slurm may deliver the advance signal up to 60 seconds early. Leave
        # space for that and require a known Judge reserve.
        if (
            not 0 < reserve <= 65535
            or _lsf_walltime_seconds(limit) <= reserve + _SIGNAL_EARLY_SECONDS
        ):
            raise SetupError(
                "run_walltime_limit leaves no Work interval after the Judge "
                "drain reserve and Slurm signal margin (or reserve is unavailable); "
                "use a longer allocation or a fresh task plan"
            )
        return SlurmJobSpec.model_validate(
            {**spec.model_dump(), "walltime": limit, "signal_seconds": reserve}
        )

    def _resume_policy_validator(
        self, payload: EnginePayload
    ) -> Callable[[ResumeContext], None] | None:
        if self.profile.scheduler.run_walltime_limit is None:
            return None

        def validate(context: ResumeContext) -> None:
            self._validate_segment_budget(
                payload.run_plan.task,
                agent_seconds=context.agent_seconds_left,
                attempt=context.attempt,
                multi_node=payload.multi_node is not None,
            )

        return validate

    @staticmethod
    def _require_success(
        stage: str,
        result: SlurmJobResult,
        stderr_path: Path | None = None,
        detail: str = "",
    ) -> None:
        if result.state != "COMPLETED" or result.exit_code != 0:
            raise InfrastructureError(
                f"{stage} job {result.job_id} ended in {result.state} "
                f"with exit {result.exit_code}"
                f"{job_log_detail(stderr_path, result.job_id)}{detail}"
            )
