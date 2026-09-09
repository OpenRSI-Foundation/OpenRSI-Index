from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

import rsi_harness.cluster.bluevela.adapter as bluevela_adapter
from rsi_harness.cluster.bluevela.adapter import derive_resources
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions, GPURequirement
from rsi_harness.task.compiler import HarborTaskCompiler
from tests.factories import write_cluster_task


def _target_definition():
    with TemporaryDirectory() as directory:
        return HarborTaskCompiler().compile(
            write_cluster_task(Path(directory)),
            CompileOptions(agent_name="codex"),
        )


def test_target_resources_are_derived_from_current_task_configuration() -> None:
    resources = derive_resources(
        _target_definition(),
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.work_gpus == 2
    assert resources.verifier_gpus == 2
    assert resources.total_gpus == 4
    assert resources.cpu_slots == 8
    assert resources.memory_mb == 65536
    assert resources.local_tmp_mb == 15360
    assert resources.build_walltime == "02:00"
    assert resources.run_walltime == "02:15"


def test_task_build_timeout_expands_cluster_build_walltime() -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "service": definition.service.model_copy(
                update={"build_timeout_seconds": 21600.0}
            )
        }
    )

    resources = derive_resources(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.build_walltime == "06:15"


def test_task_storage_is_used_as_node_local_tmp_requirement() -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "service": definition.service.model_copy(
                update={"storage_mb": 512000}
            )
        }
    )

    resources = derive_resources(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.local_tmp_mb == 512000


def test_all_gpu_declaration_requires_numeric_profile_override() -> None:
    definition = _target_definition().model_copy(
        update={"gpu_requirement": GPURequirement(count="all")}
    )

    with pytest.raises(SetupError, match="numeric override"):
        derive_resources(
            definition,
            load_cluster_profile("bluevela", {"USER": "alice"}),
        )


@pytest.mark.parametrize("verifier_gpus", (4, 8))
def test_single_node_reuses_work_gpus_when_phase_sum_exceeds_capacity(
    verifier_gpus: int,
) -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=8),
            "verifier": definition.verifier.model_copy(
                update={"gpu_count": verifier_gpus}
            ),
        }
    )

    resources = derive_resources(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert resources.work_gpus == 8
    assert resources.verifier_gpus == verifier_gpus
    assert resources.total_gpus == 8


def test_phase_gpu_limit_is_enforced() -> None:
    definition = _target_definition().model_copy(
        update={"gpu_requirement": GPURequirement(count=9)}
    )

    with pytest.raises(SetupError, match="single-node capacity"):
        derive_resources(
            definition,
            load_cluster_profile("bluevela", {"USER": "alice"}),
        )


def test_legacy_eight_plus_eight_stays_on_single_node_branch() -> None:
    assert hasattr(bluevela_adapter, "derive_resource_plan"), (
        "Blue Vela needs an explicit compatible single/multi dispatch"
    )
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=8),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 8}),
        }
    )

    plan = bluevela_adapter.derive_resource_plan(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert plan.single_node is not None
    assert plan.multi_node is None
    assert plan.single_node.total_gpus == 8


def test_explicit_disjoint_phases_stay_multinode_when_each_phase_fits() -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=16),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 16}),
            "require_disjoint_phase_nodes": True,
        }
    )
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = profile.model_copy(
        update={
            "resources": profile.resources.model_copy(
                update={"gpus_per_node": 16}
            )
        }
    )

    plan = bluevela_adapter.derive_resource_plan(definition, profile)

    assert plan.single_node is None
    assert plan.multi_node is not None
    assert plan.multi_node.work.node_count == 1
    assert plan.multi_node.verifier.node_count == 1
    assert plan.multi_node.total_nodes == 2


def test_thirty_two_plus_sixteen_becomes_four_plus_two_nodes() -> None:
    assert hasattr(bluevela_adapter, "derive_resource_plan"), (
        "Blue Vela needs an explicit compatible single/multi dispatch"
    )
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=32),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 16}),
        }
    )

    plan = bluevela_adapter.derive_resource_plan(
        definition,
        load_cluster_profile("bluevela", {"USER": "alice"}),
    )

    assert plan.single_node is None
    assert plan.multi_node is not None
    assert plan.multi_node.work.gpu_count == 32
    assert plan.multi_node.work.node_count == 4
    assert plan.multi_node.verifier.gpu_count == 16
    assert plan.multi_node.verifier.node_count == 2
    assert plan.multi_node.total_nodes == 6


def test_multinode_geometry_comes_from_profile_not_an_eight_gpu_constant() -> None:
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=12),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 4}),
        }
    )
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = profile.model_copy(
        update={
            "resources": profile.resources.model_copy(
                update={"gpus_per_node": 4}
            )
        }
    )

    plan = bluevela_adapter.derive_resource_plan(definition, profile)

    assert plan.multi_node is not None
    assert plan.multi_node.gpus_per_node == 4
    assert plan.multi_node.work.node_count == 3
    assert plan.multi_node.verifier.node_count == 1
    assert plan.multi_node.total_nodes == 4


@pytest.mark.parametrize(("work_gpus", "verifier_gpus"), ((9, 8), (16, 9)))
def test_multinode_rejects_partial_gpu_nodes(
    work_gpus: int,
    verifier_gpus: int,
) -> None:
    assert hasattr(bluevela_adapter, "derive_resource_plan"), (
        "Blue Vela needs an explicit compatible single/multi dispatch"
    )
    definition = _target_definition()
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=work_gpus),
            "verifier": definition.verifier.model_copy(
                update={"gpu_count": verifier_gpus}
            ),
        }
    )

    with pytest.raises(SetupError, match="whole 8-GPU nodes"):
        bluevela_adapter.derive_resource_plan(
            definition,
            load_cluster_profile("bluevela", {"USER": "alice"}),
        )
