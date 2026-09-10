"""Production CPU preparation without Docker or NVIDIA hardware access."""

import subprocess

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import RunRequest, RunResult, RunStatus
from rsi_harness.runtime.gpu import NvidiaSmiInventory
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import DEFAULT_TASK_TOML, write_harbor_task


def prepare_services(tmp_path, inventory):
    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock):
            self.backend = backend

        def run(self, request):
            definition = self.backend.compile(request)
            captured.append(self.backend.allocate(definition, request.gpu_selectors))
            return RunResult(run_id="prepared", status=RunStatus.COMPLETED)

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=object(),
        inventory=inventory,
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
    )
    return services, captured


def write_task(tmp_path, *, work=0, judge=None):
    task_toml = DEFAULT_TASK_TOML.replace("gpus = 2", f"gpus = {work}")
    if judge is not None:
        task_toml += f"\n[metadata.rsi_harness.verifier]\ngpus = {judge}\n"
    return write_harbor_task(tmp_path, task_toml=task_toml)


def missing_nvidia_smi(_command):
    raise FileNotFoundError("nvidia-smi is not installed")


@pytest.mark.parametrize("judge", (None, 0))
def test_cpu_preparation_does_not_require_nvidia_smi(tmp_path, judge):
    """Unconditional inventory discovery prevents CPU runs on ordinary hosts."""
    services, captured = prepare_services(
        tmp_path, NvidiaSmiInventory(runner=missing_nvidia_smi)
    )

    result = services.run(RunRequest(task_dir=write_task(tmp_path, judge=judge)))

    assert result.status is RunStatus.COMPLETED
    assert len(captured) == 1
    assert captured[0].authorized_pool.uuids == ()
    assert captured[0].work.uuids == ()
    assert captured[0].judge.uuids == ()
    assert captured[0].judge_mode.value == "freeze-only"


def test_cpu_preparation_rejects_unused_selectors_before_inventory(tmp_path):
    services, captured = prepare_services(
        tmp_path, NvidiaSmiInventory(runner=missing_nvidia_smi)
    )

    with pytest.raises(SetupError, match="CPU-only.*--gpus"):
        services.run(RunRequest(task_dir=write_task(tmp_path), gpu_selectors=("0",)))

    assert captured == []
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(("work", "judge"), ((1, 0), (0, 1)))
def test_gpu_phase_never_falls_back_to_cpu_when_inventory_is_unavailable(
    tmp_path, work, judge
):
    services, captured = prepare_services(
        tmp_path, NvidiaSmiInventory(runner=missing_nvidia_smi)
    )

    with pytest.raises(SetupError, match="NVIDIA GPU inventory is unavailable"):
        services.run(
            RunRequest(
                task_dir=write_task(tmp_path, work=work, judge=judge),
                gpu_selectors=("0",),
            )
        )

    assert captured == []
    assert not (tmp_path / "data").exists()


def test_cpu_work_preparation_keeps_explicit_gpu_judge_allocation(tmp_path):
    services, captured = prepare_services(
        tmp_path,
        NvidiaSmiInventory(
            runner=lambda command: subprocess.CompletedProcess(
                command, 0, "0, GPU-test, NVIDIA H100\n", ""
            )
        ),
    )

    services.run(
        RunRequest(task_dir=write_task(tmp_path, judge=1), gpu_selectors=("0",))
    )

    assert captured[0].authorized_pool.uuids == ("GPU-test",)
    assert captured[0].work.uuids == ()
    assert captured[0].judge.uuids == ("GPU-test",)
    assert captured[0].judge_mode.value == "disjoint"
