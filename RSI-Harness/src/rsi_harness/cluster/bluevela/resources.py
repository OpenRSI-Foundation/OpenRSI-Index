"""Compatible single-node and multi-node Blue Vela resource planning."""

from __future__ import annotations

import math
import re
from typing import Self

from pydantic import Field, model_validator

from rsi_harness.cluster.config import ClusterProfile
from rsi_harness.errors import SetupError
from rsi_harness.models import PersistedModel, TaskDefinition


class ClusterResources(PersistedModel):
    """The frozen legacy one-host resource shape."""

    work_gpus: int = Field(ge=0)
    verifier_gpus: int = Field(ge=0)
    total_gpus: int = Field(ge=0)
    cpu_slots: int = Field(gt=0)
    memory_mb: int = Field(gt=0)
    local_tmp_mb: int = Field(ge=0)
    build_walltime: str
    run_walltime: str


class PhaseResources(PersistedModel):
    gpu_count: int = Field(ge=0)
    node_count: int = Field(ge=0)


class MultiNodeResources(PersistedModel):
    work: PhaseResources
    verifier: PhaseResources
    total_nodes: int = Field(gt=1)
    gpus_per_node: int = Field(gt=0)
    cpu_slots_per_node: int = Field(gt=0)
    memory_mb_per_node: int = Field(gt=0)
    shared_workspace_mb: int = Field(ge=0)
    node_tmp_mb: int = Field(ge=0)
    build_walltime: str
    run_walltime: str


class BlueVelaResourcePlan(PersistedModel):
    single_node: ClusterResources | None = None
    multi_node: MultiNodeResources | None = None

    @model_validator(mode="after")
    def _exactly_one_branch(self) -> Self:
        if (self.single_node is None) == (self.multi_node is None):
            raise ValueError("exactly one Blue Vela resource branch is required")
        return self


def _lsf_walltime(seconds: float) -> str:
    minutes = max(1, math.ceil(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}"


def _lsf_walltime_seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+):([0-5]\d)", value)
    if match is None:
        raise SetupError(
            f"cluster builder walltime must use HH:MM format, got {value!r}"
        )
    hours, minutes = (int(part) for part in match.groups())
    total = (hours * 60 + minutes) * 60
    if total == 0:
        raise SetupError("cluster builder walltime must be positive")
    return total


def _phase_gpu_counts(
    definition: TaskDefinition,
    profile: ClusterProfile,
) -> tuple[int, int]:
    declared_work = definition.gpu_requirement.count
    if declared_work == "all":
        override = profile.resources.all_gpus_override
        if override is None:
            raise SetupError(
                "task declares gpus='all'; cluster profile needs a numeric override"
            )
        work_gpus = override
    else:
        work_gpus = declared_work
    return work_gpus, definition.verifier.gpu_count


def _walltimes(
    definition: TaskDefinition,
    profile: ClusterProfile,
) -> tuple[str, str]:
    run_seconds = (
        definition.agent.timeout_seconds
        + definition.verifier.timeout_seconds
        + profile.resources.walltime_margin_seconds
    )
    build_seconds = max(
        _lsf_walltime_seconds(profile.builder.walltime),
        definition.service.build_timeout_seconds
        + profile.resources.walltime_margin_seconds,
    )
    return _lsf_walltime(build_seconds), _lsf_walltime(run_seconds)


def derive_resources(
    definition: TaskDefinition,
    profile: ClusterProfile,
) -> ClusterResources:
    """Derive the existing one-node allocation without changing its behavior."""
    work_gpus, verifier_gpus = _phase_gpu_counts(definition, profile)
    phase_gpus = max(work_gpus, verifier_gpus)
    if phase_gpus > profile.resources.gpus_per_node:
        raise SetupError(
            "task phase GPU request exceeds cluster single-node capacity: "
            f"max({work_gpus}, {verifier_gpus})={phase_gpus} > "
            f"{profile.resources.gpus_per_node}"
        )
    requested_gpus = work_gpus + verifier_gpus
    total_gpus = (
        requested_gpus
        if requested_gpus <= profile.resources.gpus_per_node
        else phase_gpus
    )
    build_walltime, run_walltime = _walltimes(definition, profile)
    return ClusterResources(
        work_gpus=work_gpus,
        verifier_gpus=verifier_gpus,
        total_gpus=total_gpus,
        cpu_slots=max(profile.resources.min_cpu_slots, definition.service.cpus or 1),
        memory_mb=max(
            profile.resources.min_memory_mb,
            definition.service.memory_mb or 1,
        ),
        local_tmp_mb=definition.service.storage_mb or 0,
        build_walltime=build_walltime,
        run_walltime=run_walltime,
    )


def derive_resource_plan(
    definition: TaskDefinition,
    profile: ClusterProfile,
) -> BlueVelaResourcePlan:
    """Select legacy one-node behavior or strict full-node multi-node behavior."""
    work_gpus, verifier_gpus = _phase_gpu_counts(definition, profile)
    gpus_per_node = profile.resources.gpus_per_node
    if (
        max(work_gpus, verifier_gpus) <= gpus_per_node
        and not definition.require_disjoint_phase_nodes
    ):
        return BlueVelaResourcePlan(
            single_node=derive_resources(definition, profile)
        )

    invalid = tuple(
        (label, count)
        for label, count in (("Work", work_gpus), ("Judge", verifier_gpus))
        if count and count % gpus_per_node
    )
    if invalid:
        detail = ", ".join(f"{label}={count}" for label, count in invalid)
        raise SetupError(
            "Blue Vela multi-node phases require whole "
            f"{gpus_per_node}-GPU nodes; got {detail}"
        )
    build_walltime, run_walltime = _walltimes(definition, profile)
    work = PhaseResources(
        gpu_count=work_gpus,
        node_count=work_gpus // gpus_per_node,
    )
    verifier = PhaseResources(
        gpu_count=verifier_gpus,
        node_count=verifier_gpus // gpus_per_node,
    )
    return BlueVelaResourcePlan(
        multi_node=MultiNodeResources(
            work=work,
            verifier=verifier,
            total_nodes=work.node_count + verifier.node_count,
            gpus_per_node=gpus_per_node,
            cpu_slots_per_node=max(
                profile.resources.min_cpu_slots,
                definition.service.cpus or 1,
            ),
            memory_mb_per_node=max(
                profile.resources.min_memory_mb,
                definition.service.memory_mb or 1,
            ),
            shared_workspace_mb=definition.service.storage_mb or 0,
            node_tmp_mb=profile.resources.min_runtime_tmp_mb,
            build_walltime=build_walltime,
            run_walltime=run_walltime,
        )
    )


__all__ = [
    "BlueVelaResourcePlan",
    "ClusterResources",
    "MultiNodeResources",
    "PhaseResources",
    "derive_resource_plan",
    "derive_resources",
]
