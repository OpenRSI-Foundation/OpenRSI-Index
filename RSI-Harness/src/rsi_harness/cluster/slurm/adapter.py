"""Slurm scheduling for the same image, resource, and Work/Judge lifecycle."""

from __future__ import annotations

from typing import Any

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.config import SlurmSchedulerProfile
from rsi_harness.cluster.schedulers.slurm import (
    SlurmJobResult,
    SlurmJobSpec,
    SlurmScheduler,
)
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import TaskDefinition


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
        return definition

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
        )

    @staticmethod
    def _require_success(stage: str, result: SlurmJobResult) -> None:
        if result.state != "COMPLETED" or result.exit_code != 0:
            raise InfrastructureError(
                f"{stage} job {result.job_id} ended in {result.state} "
                f"with exit {result.exit_code}"
            )
