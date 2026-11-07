"""One-host batch schedulers (PBS, PJM) on the Slurm adapter's semantics."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import ClusterResumePlan
from rsi_harness.cluster.bluevela.resources import (
    _phase_gpu_counts,
    derive_resource_plan,
)
from rsi_harness.cluster.schedulers.allocation import AllocationScheduler
from rsi_harness.cluster.schedulers.batch import (
    BatchJobResult,
    BatchJobSpec,
    BatchScheduler,
    remove_environment_files,
)
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import SetupError
from rsi_harness.models import TaskDefinition

if TYPE_CHECKING:
    from rsi_harness.cluster.bluevela.engine import EnginePayload
    from rsi_harness.runtime.recovery import AttemptGrants


class OneHostBatchAdapter(SlurmClusterAdapter):
    """Each build and run job is one batch job on one host; the Engine in it
    runs Work and Judge with Apptainer, as on LSF and Slurm. Segmented runs
    and resume follow Slurm's."""

    def _single_node_refusal(self) -> SetupError:
        return SetupError(
            f"{self.scheduler_label} runs use one job on one node; this task "
            "needs several nodes (use Slurm or LSF)"
        )

    def _compile(self, request: ClusterRunRequest) -> TaskDefinition:
        task = self.compiler.compile(request.task_dir, request.options)
        if task.require_disjoint_phase_nodes:
            raise self._single_node_refusal()
        wanted = max(_phase_gpu_counts(task, self.profile))
        node = self.profile.resources.gpus_per_node
        if wanted > node:
            raise SetupError(
                f"this task needs {wanted} GPUs on one node; this profile's "
                f"{self.scheduler_label} jobs have {node} "
                "(resources.gpus_per_node): use a profile whose jobs hold more"
            )
        definition = super()._compile(request)
        if derive_resource_plan(definition, self.profile).multi_node is not None:
            raise self._single_node_refusal()
        return definition

    def _validate_capacity(self, spec: BatchJobSpec, *, build: bool = False) -> None:
        scheduler = self.profile.scheduler
        capacity = scheduler.capacity
        if build and scheduler.build is not None:
            capacity = scheduler.build.capacity or capacity
        if capacity is None:
            return
        shares = 1
        if capacity.per_gpu:
            whole_node = (
                spec.exclusive
                or getattr(spec, "gpu_resource", "gpu") is None
                or "ngpus" not in getattr(spec, "chunk_resources", ("ngpus",))
            )
            shares = (
                self.profile.resources.gpus_per_node if whole_node else spec.gpu_count
            )
            if not shares:
                raise SetupError(
                    f"queue {spec.queue} allocates CPU/RAM per GPU; this job requests "
                    "no GPUs. Use a CPU queue/profile for this task."
                )
        cpu_limit = capacity.cpu_slots * shares
        memory_limit = capacity.memory_mb * shares
        if spec.cpu_slots > cpu_limit or spec.memory_mb > memory_limit:
            stage = "build" if build else "run"
            raise SetupError(
                f"{stage} needs {spec.cpu_slots} CPUs and {spec.memory_mb} MiB RAM, "
                f"but queue {spec.queue}'s allocation provides {cpu_limit} CPUs "
                f"and {memory_limit} MiB RAM. Choose a larger allocation/profile."
            )

    def _check_build_spec(self, spec: Any) -> None:
        # A build in the controller's own job takes that job's resources.
        if self._allocation() is None:
            self._validate_capacity(spec, build=True)

    def _run_spec(self, *args: Any, **kwargs: Any) -> BatchJobSpec:
        spec = super()._run_spec(*args, **kwargs)
        self._validate_capacity(spec)
        return spec

    def _allocation(self) -> AllocationScheduler | None:
        scheduler = self.scheduler
        return scheduler if isinstance(scheduler, AllocationScheduler) else None

    def _allocation_scheduler(self, host: BatchScheduler) -> Any:
        """``host``, or with launch = "allocation" its jobs run inside the
        controller's own job."""
        if getattr(self.profile.scheduler, "launch", "submit") == "submit":
            return host
        return AllocationScheduler(host, record_root=self._local_records)

    @property
    def _local_records(self) -> Path:
        return self.profile.storage.run_root / ".local-jobs"

    def _job_finished(self, job_id: str) -> BatchJobResult | None:
        # A run begun with allocation launch may resume without it.
        scheduler = self._allocation() or AllocationScheduler(
            self.scheduler, record_root=self._local_records
        )
        return scheduler.finished(job_id)

    def _check_frozen_source(self, payload: EnginePayload) -> None:
        # An Engine frozen before launch modes refuses the launch field.
        if (
            getattr(self.profile.scheduler, "launch", "submit") == "allocation"
            and not (
                payload.source_root / "src/rsi_harness/cluster/schedulers/allocation.py"
            ).is_file()
        ):
            raise SetupError(
                f"run {payload.run_id} began on a Harness without allocation "
                'launch: resume it with scheduler.launch = "submit"'
            )

    def plan_resume(
        self, run_id: str, grants: AttemptGrants, *, dry_run: bool = False
    ) -> ClusterResumePlan:
        plan = super().plan_resume(run_id, grants, dry_run=dry_run)
        # The Engine binds its GPUs as the current launch runs it.
        frozen = plan.payload.profile
        scheduler = frozen.scheduler.model_copy(
            update={"launch": getattr(self.profile.scheduler, "launch", "submit")}
        )
        payload = plan.payload.model_copy(
            update={"profile": frozen.model_copy(update={"scheduler": scheduler})}
        )
        return dataclasses.replace(plan, payload=payload)

    def _check_controller_machine(self) -> None:
        super()._check_controller_machine()
        if (allocation := self._allocation()) is not None:
            allocation.allocation()

    def _check_jobs_fit(self, specs: tuple[Any, ...]) -> None:
        if (allocation := self._allocation()) is not None:
            allocation.require_room(specs)

    def _jobs_ended(self, run_dir: Path) -> None:
        # A controller that died before its job started left the job's
        # environment file (credentials included).
        remove_environment_files(run_dir / "build", run_dir / "run")
