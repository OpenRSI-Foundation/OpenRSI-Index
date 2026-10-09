"""AMD allocation, container isolation, and Work quiescence on real GPUs.

Run on an AMD host with an explicit pool and an image that ships ROCm's HIP
runtime and Python, for example::

    RSI_TEST_GPUS=2,3 RSI_TEST_ROCM_IMAGE=rocm/pytorch:latest \
        python -m pytest -m gpu tests/hardware/test_amd_gpu.py
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path

import docker
import pytest

from rsi_harness.errors import SubmissionError
from rsi_harness.models import (
    ContainerRef,
    ContainerSpec,
    GPUAllocation,
    GPURequirement,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.gpu import (
    AmdSmiInventory,
    assert_work_gpu_quiescent,
    host_gpu_inventory,
    resolve_allocation,
)

pytestmark = pytest.mark.gpu

TASK_USER = "65534:65534"
PROBE_PATH = "/run/rsi-harness/staging/probe.py"
PROBE = """\
import ctypes, json, os, sys, time

dri = "/dev/dri"
renders = sorted(n for n in os.listdir(dri) if n.startswith("renderD")) \\
    if os.path.isdir(dri) else []
hip = ctypes.CDLL("libamdhip64.so")
count = ctypes.c_int(0)
status = hip.hipGetDeviceCount(ctypes.byref(count))
buses = []
for device in range(count.value if status == 0 else 0):
    bus = ctypes.create_string_buffer(64)
    hip.hipDeviceGetPCIBusId(bus, 64, device)
    buses.append(bus.value.decode().lower())
print(json.dumps({"render_nodes": renders, "hip_status": status, "buses": buses}))
sys.stdout.flush()
if len(sys.argv) > 1:
    hip.hipSetDevice(0)
    pointer = ctypes.c_void_p()
    if hip.hipMalloc(ctypes.byref(pointer), 16 << 20) != 0:
        sys.exit("hipMalloc failed")
    open(sys.argv[1], "w").close()
    while not os.path.exists(sys.argv[1] + ".stop"):
        time.sleep(0.1)
    hip.hipFree(pointer)
"""


def _selectors() -> tuple[str, ...]:
    raw = os.environ.get("RSI_TEST_GPUS")
    if raw is None:
        pytest.skip("set RSI_TEST_GPUS to an explicit comma-separated GPU pool")
    selectors = tuple(part.strip() for part in raw.split(","))
    if not selectors or not all(selectors):
        pytest.fail("RSI_TEST_GPUS must contain only non-empty GPU selectors")
    return selectors


def _image() -> str:
    image = os.environ.get("RSI_TEST_ROCM_IMAGE")
    if not image:
        pytest.skip("set RSI_TEST_ROCM_IMAGE to a local image with HIP and Python")
    return image


def _pci_address(render_node: Path) -> str:
    return Path(f"/sys/class/drm/{render_node.name}/device").resolve().name.lower()


def _wait(condition: Callable[[], bool], what: str, seconds: float = 60.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.2)


@pytest.fixture
def amd_allocation() -> GPUAllocation:
    inventory = host_gpu_inventory()
    if not isinstance(inventory, AmdSmiInventory):
        pytest.skip("host is not an AMD GPU host")
    selectors = _selectors()
    return resolve_allocation(
        GPURequirement(count=len(selectors)),
        requested=selectors,
        inventory=inventory.list_devices(),
    )


def test_amd_container_reaches_exactly_its_gpus_and_blocks_release_until_idle(
    tmp_path: Path, amd_allocation: GPUAllocation
) -> None:
    image = _image()
    client = docker.from_env()
    engine = tmp_path / "engine"
    (engine / "staging").mkdir(parents=True)
    (engine / "staging" / "probe.py").write_text(PROBE)
    (tmp_path / "task").mkdir()
    created: list[tuple[DockerContainerRuntime, ContainerRef]] = []

    def start(role: str, allocation: GPUAllocation) -> ContainerRef:
        runtime = DockerContainerRuntime(
            client,
            run_id=f"amd-{uuid.uuid4().hex[:8]}",
            task_id="amd-gpu",
            role=role,
            task_source_dir=tmp_path / "task",
            allowed_mount_roots=(engine,),
            staging_dir=engine / "staging",
        )
        ref = runtime.create(
            ContainerSpec(
                image=image,
                command=("sleep", "infinity"),
                user=TASK_USER,
                environment=(("HOME", "/tmp"),),
                gpu_allocation=allocation,
            )
        )
        created.append((runtime, ref))
        # Started directly: the network policy lease that runtime.start()
        # demands is not under test, and the container has no network.
        client.containers.get(ref.container_id).start()
        return ref

    def probe_in(ref: ContainerRef) -> dict[str, object]:
        exit_code, output = client.containers.get(ref.container_id).exec_run(
            ["python3", PROBE_PATH], user=TASK_USER, environment={"HOME": "/tmp"}
        )
        text = output.decode(errors="replace")
        assert exit_code == 0, text
        return json.loads(text.strip().splitlines()[-1])

    try:
        work = start("work", amd_allocation)
        renders = sorted(
            device.render_node.name
            for device in amd_allocation.devices
            if device.render_node is not None
        )
        seen = probe_in(work)
        assert seen["render_nodes"] == renders
        assert seen["hip_status"] == 0
        assert sorted(seen["buses"]) == sorted(
            _pci_address(device.render_node)
            for device in amd_allocation.devices
            if device.render_node is not None
        )

        helper = start("helper", GPUAllocation())
        unallocated = probe_in(helper)
        assert unallocated["render_nodes"] == []
        assert unallocated["buses"] == []

        container = client.containers.get(work.container_id)
        container.exec_run(
            ["python3", PROBE_PATH, "/tmp/holder-ready"],
            user=TASK_USER,
            environment={"HOME": "/tmp"},
            detach=True,
        )
        _wait(
            lambda: (
                container.exec_run(["test", "-e", "/tmp/holder-ready"]).exit_code == 0
            ),
            "the HIP holder to allocate GPU memory",
        )
        with pytest.raises(SubmissionError, match="is still active on"):
            assert_work_gpu_quiescent(amd_allocation, container)

        container.exec_run(["touch", "/tmp/holder-ready.stop"], user=TASK_USER)

        def released() -> bool:
            try:
                assert_work_gpu_quiescent(amd_allocation, container)
            except SubmissionError:
                return False
            return True

        _wait(released, "the HIP holder to release its KFD context")
    finally:
        for runtime, ref in reversed(created):
            runtime.stop(ref)
            runtime.remove(ref)
