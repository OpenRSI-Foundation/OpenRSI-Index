from __future__ import annotations

from pathlib import Path

from rsi_harness.cluster.bluevela.adapter import ClusterResources
from rsi_harness.cluster.bluevela.allocation import AllocatedNode, AllocatedPools
from rsi_harness.cluster.bluevela.engine import (
    EnginePayload,
    bind_lsf_devices,
    bind_multinode_devices,
    render_engine_driver,
    run_engine_payload,
)
from rsi_harness.cluster.bluevela.resources import (
    MultiNodeResources,
    PhaseResources,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.models import AgentAuthSource, CompileOptions
from tests.factories import make_run_plan


def _payload(tmp_path: Path) -> EnginePayload:
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    plan = make_run_plan(tmp_path)
    devices = plan.gpu_plan.authorized_pool.devices
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "authorized_pool": plan.gpu_plan.authorized_pool.model_copy(
                        update={"devices": devices * 2}
                    ),
                    "work": plan.gpu_plan.work.model_copy(
                        update={"devices": devices}
                    ),
                    "judge": plan.gpu_plan.judge.model_copy(
                        update={"devices": devices[1:] or devices}
                    ),
                }
            )
        }
    )
    # Use four distinct placeholder devices, matching a 2+2 task allocation.
    from rsi_harness.models import GPUAllocation, GPUDevice, RunGPUPlan

    all_devices = tuple(
        GPUDevice(index=index, uuid=f"LSF-{index}", name="allocated")
        for index in range(4)
    )
    plan = plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=all_devices),
                work=GPUAllocation(devices=all_devices[:2]),
                judge=GPUAllocation(devices=all_devices[2:]),
                judge_mode="disjoint",
            )
        }
    )
    return EnginePayload(
        run_id="native-engine-run",
        run_plan=plan,
        sif_path=(tmp_path / "task.sif").resolve(),
        sif_sha256_path=(tmp_path / "task.sif.sha256").resolve(),
        source_root=Path(__file__).resolve().parents[3],
        resources=ClusterResources(
            work_gpus=2,
            verifier_gpus=2,
            total_gpus=4,
            cpu_slots=8,
            memory_mb=65536,
            local_tmp_mb=15360,
            build_walltime="02:00",
            run_walltime="02:15",
        ),
        profile=profile,
        options=CompileOptions(max_submissions=4),
        agent_auth=AgentAuthSource.LOCAL,
        agent_version="0.149.0",
        agent_binary=Path("/bin/true"),
        agent_launcher="codex",
        agent_companions=(Path("/bin/false"),),
    )


def test_lsf_devices_replace_placeholders_and_remain_disjoint(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    plan = bind_lsf_devices(payload.run_plan, ("0", "2", "5", "7"))

    assert plan.gpu_plan.authorized_pool.uuids == ("0", "2", "5", "7")
    assert plan.gpu_plan.work.uuids == ("0", "2")
    assert plan.gpu_plan.judge.uuids == ("5", "7")


def test_lsf_devices_preserve_release_all_overlap(tmp_path: Path) -> None:
    from rsi_harness.models import GPUAllocation, RunGPUPlan

    payload = _payload(tmp_path)
    placeholders = payload.run_plan.gpu_plan.authorized_pool.devices
    run_plan = payload.run_plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=placeholders),
                work=GPUAllocation(devices=placeholders),
                judge=GPUAllocation(devices=placeholders[:2]),
                judge_mode="release-all",
            )
        }
    )

    plan = bind_lsf_devices(run_plan, ("0", "2", "5", "7"))

    assert plan.gpu_plan.authorized_pool.uuids == ("0", "2", "5", "7")
    assert plan.gpu_plan.work.uuids == ("0", "2", "5", "7")
    assert plan.gpu_plan.judge.uuids == ("0", "2")
    assert plan.gpu_plan.judge_mode.value == "release-all"


def test_gpu_driver_invokes_native_engine_without_harbor_run(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    output = (tmp_path / "run.sh").resolve()

    payload_path = render_engine_driver(payload, profile, output)

    script = output.read_text()
    assert "rsi_harness.cluster.bluevela.engine" in script
    assert "harbor run" not in script
    assert "harbor-jobs" not in script
    assert "RSI_HARNESS_NODE_TMP" in script
    assert "available_tmp_kb" in script
    assert "required_tmp_kb=$((15360 * 1024))" in script
    assert payload_path == output.with_suffix(".json")


def test_engine_payload_keeps_native_submission_controls(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    assert payload.options.max_submissions == 4
    assert payload.run_plan.paths.logs.is_absolute()
    assert payload.run_plan.task.verifier.command == ("/bin/bash", "/tests/test.sh")
    assert payload.agent_launcher == "codex"


def _multi_payload(tmp_path: Path) -> EnginePayload:
    payload = _payload(tmp_path)
    from rsi_harness.models import GPUAllocation, GPUDevice, RunGPUPlan

    work = tuple(
        GPUDevice(
            index=index,
            uuid=f"WORK-{index // 8:03d}:GPU-{index % 8}",
            name="planned Work GPU",
        )
        for index in range(16)
    )
    judge = tuple(
        GPUDevice(
            index=16 + index,
            uuid=f"JUDGE-000:GPU-{index}",
            name="planned Judge GPU",
        )
        for index in range(8)
    )
    plan = payload.run_plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=work + judge),
                work=GPUAllocation(devices=work),
                judge=GPUAllocation(devices=judge),
                judge_mode="disjoint",
            )
        }
    )
    return payload.model_copy(
        update={
            "run_plan": plan,
            "resources": None,
            "multi_node": MultiNodeResources(
                work=PhaseResources(gpu_count=16, node_count=2),
                verifier=PhaseResources(gpu_count=8, node_count=1),
                total_nodes=3,
                gpus_per_node=8,
                cpu_slots_per_node=8,
                memory_mb_per_node=65536,
                shared_workspace_mb=15360,
                node_tmp_mb=71680,
                build_walltime="02:00",
                run_walltime="02:15",
            ),
        }
    )


def _allocated_pools() -> AllocatedPools:
    def node(host: str, subnet: int) -> AllocatedNode:
        return AllocatedNode(
            host=host,
            slots=8,
            ipv4=f"10.0.{subnet}.1",
            cuda_devices=tuple(f"GPU-{host}-{index}" for index in range(8)),
        )

    return AllocatedPools(
        run_id="native-engine-run",
        work=(node("work-a", 0), node("work-b", 1)),
        verifier=(node("judge-a", 2),),
        gpus_per_node=8,
        digest="a" * 64,
    )


def test_multinode_devices_bind_host_local_allocations_and_remain_disjoint(
    tmp_path: Path,
) -> None:
    payload = _multi_payload(tmp_path)

    plan = bind_multinode_devices(payload.run_plan, _allocated_pools())

    assert plan.gpu_plan.work.uuids[:9] == (
        "work-a:GPU-work-a-0",
        "work-a:GPU-work-a-1",
        "work-a:GPU-work-a-2",
        "work-a:GPU-work-a-3",
        "work-a:GPU-work-a-4",
        "work-a:GPU-work-a-5",
        "work-a:GPU-work-a-6",
        "work-a:GPU-work-a-7",
        "work-b:GPU-work-b-0",
    )
    assert plan.gpu_plan.judge.uuids[0] == "judge-a:GPU-judge-a-0"
    assert set(plan.gpu_plan.work.uuids).isdisjoint(plan.gpu_plan.judge.uuids)


def test_multinode_driver_uses_allocation_probe_without_controller_cuda_list(
    tmp_path: Path,
) -> None:
    payload = _multi_payload(tmp_path)
    output = (tmp_path / "multi-run.sh").resolve()

    render_engine_driver(payload, payload.profile, output)

    script = output.read_text()
    assert "LSB_MCPU_HOSTS" in script
    assert "CUDA_VISIBLE_DEVICES:-" not in script
    assert "required_tmp_kb=$((71680 * 1024))" in script
    assert "rsi_harness.cluster.bluevela.engine" in script
    assert "harbor run" not in script


def test_multinode_engine_freezes_pools_before_entering_native_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = _multi_payload(tmp_path)
    payload.run_plan.paths.root.mkdir(parents=True)
    (payload.run_plan.paths.root / "control").mkdir()
    payload.sif_sha256_path.write_text("d" * 64 + "  task.sif\n")
    pools = _allocated_pools()
    captured: list[tuple[object, object]] = []
    monkeypatch.setenv(
        "LSB_MCPU_HOSTS",
        "work-a 8 work-b 8 judge-a 8",
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.probe_and_partition",
        lambda **_kwargs: pools,
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.socket.gethostname",
        lambda: "work-a.example",
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.run_native_engine",
        lambda _payload, plan, *, allocated_pools: captured.append(
            (plan, allocated_pools)
        ),
    )

    run_engine_payload(payload)

    plan, authority = captured[0]
    assert authority == pools
    assert plan.gpu_plan.work.uuids[0] == "work-a:GPU-work-a-0"
    frozen = payload.run_plan.paths.root / "control/ALLOCATED_POOLS.json"
    assert AllocatedPools.model_validate_json(frozen.read_text()) == pools
    assert frozen.stat().st_mode & 0o777 == 0o600
    assert not tuple(frozen.parent.glob(".ALLOCATED_POOLS.json.*.tmp"))
