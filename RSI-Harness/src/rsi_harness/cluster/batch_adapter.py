"""One-host batch schedulers (PBS, PJM) on the Slurm adapter's semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.resources import (
    _phase_gpu_counts,
    derive_resource_plan,
)
from rsi_harness.cluster.schedulers.batch import BatchJobSpec, remove_environment_files
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import SetupError
from rsi_harness.models import TaskDefinition


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
        self._validate_capacity(spec, build=True)

    def _run_spec(self, *args: Any, **kwargs: Any) -> BatchJobSpec:
        spec = super()._run_spec(*args, **kwargs)
        self._validate_capacity(spec)
        return spec

    def _jobs_ended(self, run_dir: Path) -> None:
        # A controller that died before its job started left the job's
        # environment file (credentials included).
        remove_environment_files(run_dir / "build", run_dir / "run")
