from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from rsi_harness.errors import SetupError, SubmissionError
from rsi_harness.models import GPUAllocation, GPUDevice, GPURequirement
from rsi_harness.runtime.gpu import (
    AmdSmiInventory,
    NvidiaSmiInventory,
    assert_work_gpu_quiescent,
    host_gpu_inventory,
    nvidia_visible_devices_value,
    resolve_allocation,
    resolve_gpu_plan,
)

EIGHT_H100_CSV = "\n".join(
    f"{index}, GPU-{letter}, NVIDIA H100 80GB HBM3"
    for index, letter in enumerate("abcdefgh")
)


def graphics_xml(*rows: tuple[str, int]) -> str:
    gpus = "".join(
        "<gpu><uuid>"
        f"{uuid}</uuid><processes><process_info><pid>{pid}</pid>"
        "<process_type>G</process_type></process_info></processes></gpu>"
        for uuid, pid in rows
    )
    return f"<nvidia_smi_log>{gpus}</nvidia_smi_log>"


def eight_h100_inventory() -> tuple[GPUDevice, ...]:
    return tuple(
        GPUDevice(index=index, uuid=f"GPU-{letter}", name="NVIDIA H100 80GB HBM3")
        for index, letter in enumerate("abcdefgh")
    )


class RecordingCommandRunner:
    def __init__(self, responses: list[subprocess.CompletedProcess[str]]) -> None:
        self.responses = responses
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(tuple(command))
        return self.responses.pop(0)


class TopContainer:
    def __init__(self, pids: tuple[int, ...]) -> None:
        self._pids = pids

    def top(self, *, ps_args: str) -> dict[str, object]:
        assert ps_args == "-eo pid"
        return {
            "Titles": ["PID"],
            "Processes": [[str(pid)] for pid in self._pids],
        }


class FailingTopContainer:
    def top(self, *, ps_args: str) -> dict[str, object]:
        raise RuntimeError("Docker daemon unavailable")


def test_inventory_queries_index_uuid_and_name_without_a_shell():
    runner = RecordingCommandRunner(
        [subprocess.CompletedProcess([], 0, EIGHT_H100_CSV, "")]
    )

    devices = NvidiaSmiInventory(runner=runner).list_devices()

    assert devices == eight_h100_inventory()
    assert runner.commands == [
        (
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        )
    ]


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (("2", "0"), ("GPU-c", "GPU-a")),
        (("GPU-c", "GPU-a"), ("GPU-c", "GPU-a")),
    ],
)
def test_allocation_accepts_indexes_and_uuids_in_caller_order(requested, expected):
    allocation = resolve_allocation(
        GPURequirement(count=2), requested=requested, inventory=eight_h100_inventory()
    )

    assert allocation.uuids == expected


def test_all_uses_every_caller_allocated_device_not_every_host_device():
    allocation = resolve_allocation(
        GPURequirement(count="all"),
        requested=("GPU-a", "GPU-c"),
        inventory=eight_h100_inventory(),
    )

    assert allocation.uuids == ("GPU-a", "GPU-c")


@pytest.mark.parametrize(
    ("requirement", "requested", "message"),
    [
        (GPURequirement(count=2), ("0", "GPU-a"), "duplicate"),
        (GPURequirement(count=1), ("GPU-unknown",), "unknown"),
        (GPURequirement(count=2), ("GPU-a",), "exactly 2"),
        (GPURequirement(count=9), (), "only 8"),
        (GPURequirement(count="all"), (), "caller-selected"),
        (
            GPURequirement(count=1, name="A100"),
            ("GPU-a",),
            "does not match",
        ),
    ],
)
def test_allocation_rejects_ambiguous_or_unsatisfied_requests(
    requirement, requested, message
):
    with pytest.raises(SetupError, match=message):
        resolve_allocation(
            requirement, requested=requested, inventory=eight_h100_inventory()
        )


def test_numeric_count_without_selectors_allocates_exactly_the_first_devices():
    allocation = resolve_allocation(
        GPURequirement(count=2), requested=(), inventory=eight_h100_inventory()
    )

    assert allocation.uuids == ("GPU-a", "GPU-b")


@pytest.mark.parametrize(
    ("work_count", "judge_count", "requested", "mode", "work", "judge"),
    [
        (2, 0, ("0", "1", "2", "3"), "freeze-only", ("GPU-a", "GPU-b"), ()),
        (
            2,
            2,
            ("0", "1", "2", "3"),
            "disjoint",
            ("GPU-a", "GPU-b"),
            ("GPU-c", "GPU-d"),
        ),
        (
            6,
            4,
            tuple(str(i) for i in range(8)),
            "release-all",
            tuple(f"GPU-{c}" for c in "abcdef"),
            ("GPU-g", "GPU-h", "GPU-a", "GPU-b"),
        ),
        ("all", 2, ("2", "0"), "release-all", ("GPU-c", "GPU-a"), ("GPU-c", "GPU-a")),
    ],
)
def test_phase_aware_gpu_plan(work_count, judge_count, requested, mode, work, judge):
    plan = resolve_gpu_plan(
        GPURequirement(count=work_count),
        judge_count=judge_count,
        requested=requested,
        inventory=eight_h100_inventory(),
    )

    assert plan.judge_mode.value == mode
    assert plan.work.uuids == work
    assert plan.judge.uuids == judge


def test_cpu_plan_needs_no_inventory():
    plan = resolve_gpu_plan(
        GPURequirement(count=0), judge_count=0, requested=(), inventory=()
    )

    assert plan.authorized_pool.uuids == ()
    assert plan.work.uuids == ()
    assert plan.judge.uuids == ()
    assert plan.judge_mode.value == "freeze-only"


@pytest.mark.parametrize("requested", (("0",), ("GPU-a",)))
def test_cpu_plan_rejects_unused_selectors(requested):
    with pytest.raises(SetupError, match="CPU-only.*--gpus"):
        resolve_gpu_plan(
            GPURequirement(count=0),
            judge_count=0,
            requested=requested,
            inventory=eight_h100_inventory(),
        )


def test_gpu_judge_with_cpu_work_requires_explicit_pool():
    with pytest.raises(SetupError, match="explicit.*--gpus"):
        resolve_gpu_plan(
            GPURequirement(count=0),
            judge_count=1,
            requested=(),
            inventory=eight_h100_inventory(),
        )


def test_gpu_judge_with_cpu_work_uses_only_explicit_pool():
    plan = resolve_gpu_plan(
        GPURequirement(count=0),
        judge_count=1,
        requested=("2",),
        inventory=eight_h100_inventory(),
    )

    assert plan.authorized_pool.uuids == ("GPU-c",)
    assert plan.work.uuids == ()
    assert plan.judge.uuids == ("GPU-c",)
    assert plan.judge_mode.value == "disjoint"


def test_authorized_pool_may_exceed_work_count():
    plan = resolve_gpu_plan(
        GPURequirement(count=2),
        judge_count=0,
        requested=("GPU-d", "GPU-c", "GPU-b", "GPU-a"),
        inventory=eight_h100_inventory(),
    )

    assert plan.authorized_pool.uuids == ("GPU-d", "GPU-c", "GPU-b", "GPU-a")
    assert plan.work.uuids == ("GPU-d", "GPU-c")


def test_authorized_pool_without_selectors_contains_exact_work_count():
    plan = resolve_gpu_plan(
        GPURequirement(count=2),
        judge_count=0,
        requested=(),
        inventory=eight_h100_inventory(),
    )

    assert plan.authorized_pool.uuids == ("GPU-a", "GPU-b")


@pytest.mark.parametrize(
    ("requirement", "judge_count", "requested", "message"),
    [
        (GPURequirement(count="all"), 0, (), "caller-selected"),
        (GPURequirement(count=2), 0, ("GPU-a",), "requires 2 GPUs"),
        (GPURequirement(count=1), 3, ("GPU-a", "GPU-b"), "Judge requires 3 GPUs"),
        (GPURequirement(count=1), 0, ("GPU-a", "GPU-a"), "duplicate"),
        (GPURequirement(count=1), 0, ("GPU-unknown",), "unknown"),
    ],
)
def test_authorized_pool_rejects_unsatisfied_or_ambiguous_requests(
    requirement, judge_count, requested, message
):
    with pytest.raises(SetupError, match=message):
        resolve_gpu_plan(
            requirement,
            judge_count=judge_count,
            requested=requested,
            inventory=eight_h100_inventory(),
        )


def test_authorized_pool_applies_gpu_type_to_work_not_judge():
    inventory = (
        GPUDevice(index=0, uuid="GPU-a", name="NVIDIA A100"),
        GPUDevice(index=1, uuid="GPU-b", name="NVIDIA H100"),
    )

    plan = resolve_gpu_plan(
        GPURequirement(count=1, name="A100"),
        judge_count=1,
        requested=("GPU-a", "GPU-b"),
        inventory=inventory,
    )

    assert plan.work.uuids == ("GPU-a",)
    assert plan.judge.uuids == ("GPU-b",)


def test_quiescence_rejects_only_work_container_processes_on_allocated_gpus():
    runner = RecordingCommandRunner(
        [
            subprocess.CompletedProcess(
                [], 0, "GPU-a, 101\nGPU-a, 202\nGPU-z, 303\n", ""
            ),
            subprocess.CompletedProcess([], 0, graphics_xml(("GPU-a", 404)), ""),
        ]
    )
    allocation = GPUAllocation(devices=(eight_h100_inventory()[0],))

    with pytest.raises(SubmissionError, match="PID 202"):
        assert_work_gpu_quiescent(allocation, TopContainer((202, 999)), runner=runner)
    assert runner.commands[1] == ("nvidia-smi", "-q", "-x")


def test_host_and_other_container_gpu_processes_do_not_block_submission():
    runner = RecordingCommandRunner(
        [
            subprocess.CompletedProcess([], 0, "GPU-a, 101\nGPU-z, 202\n", ""),
            subprocess.CompletedProcess([], 0, graphics_xml(("GPU-a", 303)), ""),
        ]
    )

    assert_work_gpu_quiescent(
        GPUAllocation(devices=(eight_h100_inventory()[0],)),
        TopContainer((202, 999)),
        runner=runner,
    )


@pytest.mark.parametrize(
    "responses",
    [
        [subprocess.CompletedProcess([], 1, "", "query unavailable")],
        [subprocess.CompletedProcess([], 0, "not a process row\n", "")],
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, "not xml", ""),
        ],
    ],
)
def test_unavailable_or_ambiguous_process_query_fails_submission(responses):
    runner = RecordingCommandRunner(responses)

    with pytest.raises(SubmissionError, match="GPU process query"):
        assert_work_gpu_quiescent(
            GPUAllocation(devices=(eight_h100_inventory()[0],)),
            TopContainer(()),
            runner=runner,
        )


@pytest.mark.parametrize(
    "xml",
    [
        graphics_xml(("GPU-z", 404)),
        graphics_xml(("GPU-a", 404), ("GPU-a", 405)),
        "<nvidia_smi_log/>",
    ],
)
def test_graphics_query_must_identify_each_allocated_uuid_exactly_once(xml):
    runner = RecordingCommandRunner(
        [
            subprocess.CompletedProcess([], 0, "", ""),
            subprocess.CompletedProcess([], 0, xml, ""),
        ]
    )

    with pytest.raises(SubmissionError, match="GPU process query"):
        assert_work_gpu_quiescent(
            GPUAllocation(devices=(eight_h100_inventory()[0],)),
            TopContainer(()),
            runner=runner,
        )


def test_docker_top_failure_is_a_submission_error():
    with pytest.raises(SubmissionError, match="Work container process list"):
        assert_work_gpu_quiescent(
            GPUAllocation(devices=(eight_h100_inventory()[0],)),
            FailingTopContainer(),
            runner=RecordingCommandRunner([]),
        )


# Two MI355X GPUs as amd-smi 26 and the KFD topology describe them; node 0 is
# the CPU, which KFD lists with gpu_id 0.
AMD_GPUS = (
    {"bdf": "0000:05:00.0", "location_id": 0x0500, "gpu_id": 42583, "minor": 128},
    {"bdf": "0000:75:00.0", "location_id": 0x7500, "gpu_id": 17010, "minor": 152},
)


def amd_smi_list(**overrides: object) -> str:
    rows = [
        {
            "gpu": index,
            "bdf": gpu["bdf"],
            "uuid": f"{index}aff75a3-0000-1000-80e8-c5cd3767811a",
            "kfd_id": gpu["gpu_id"],
            "node_id": index + 1,
            "partition_id": 0,
        }
        for index, gpu in enumerate(AMD_GPUS)
    ]
    rows[-1].update(overrides)
    return json.dumps(rows)


def amd_smi_asic(market_name: str = "AMD Instinct MI355X") -> str:
    return json.dumps(
        {
            "gpu_data": [
                {
                    "gpu": index,
                    "asic": {
                        "market_name": market_name,
                        "target_graphics_version": "gfx950",
                    },
                }
                for index in range(len(AMD_GPUS))
            ]
        }
    )


def amd_host(tmp_path: Path, *, render_nodes: bool = True) -> tuple[Path, Path]:
    kfd_root = tmp_path / "kfd"
    cpu = kfd_root / "topology" / "nodes" / "0"
    cpu.mkdir(parents=True)
    (cpu / "gpu_id").write_text("0\n")
    (cpu / "properties").write_text("cpu_cores_count 128\n")
    for number, gpu in enumerate(AMD_GPUS, start=1):
        node = kfd_root / "topology" / "nodes" / str(number)
        node.mkdir()
        (node / "gpu_id").write_text(f"{gpu['gpu_id']}\n")
        (node / "properties").write_text(
            f"location_id {gpu['location_id']}\n"
            "domain 0\n"
            f"drm_render_minor {gpu['minor']}\n"
        )
    (kfd_root / "proc").mkdir()
    dev_root = tmp_path / "dev"
    (dev_root / "dri").mkdir(parents=True)
    if render_nodes:
        for gpu in AMD_GPUS:
            (dev_root / "dri" / f"renderD{gpu['minor']}").touch()
    return kfd_root, dev_root


def amd_inventory(
    tmp_path: Path, listed: str, asic: str | None = None, **host: bool
) -> AmdSmiInventory:
    kfd_root, dev_root = amd_host(tmp_path, **host)
    runner = RecordingCommandRunner(
        [
            subprocess.CompletedProcess([], 0, listed, ""),
            subprocess.CompletedProcess([], 0, asic or amd_smi_asic(), ""),
        ]
    )
    return AmdSmiInventory(runner=runner, kfd_root=kfd_root, dev_root=dev_root)


def test_amd_inventory_binds_amd_smi_rows_to_kfd_render_nodes(tmp_path):
    devices = amd_inventory(tmp_path, amd_smi_list()).list_devices()

    assert [
        (device.index, device.name, device.render_node, device.kfd_gpu_id)
        for device in devices
    ] == [
        (0, "AMD Instinct MI355X (gfx950)", tmp_path / "dev/dri/renderD128", 42583),
        (1, "AMD Instinct MI355X (gfx950)", tmp_path / "dev/dri/renderD152", 17010),
    ]
    assert {device.vendor for device in devices} == {"amd"}


@pytest.mark.parametrize(
    ("listed", "host", "message"),
    [
        (amd_smi_list(partition_id=1), {}, "compute partition"),
        (amd_smi_list(bdf="0000:76:00.0"), {}, "matches 0 KFD nodes"),
        (amd_smi_list(), {"render_nodes": False}, "render node .* is missing"),
        (amd_smi_list(gpu="1"), {}, "'gpu' is not an integer"),
        (json.dumps({"gpu": 0}), {}, "not a list"),
    ],
)
def test_amd_inventory_refuses_what_it_cannot_bind_exactly(
    tmp_path, listed, host, message
):
    with pytest.raises(SetupError, match=message):
        amd_inventory(tmp_path, listed, **host).list_devices()


def test_amd_inventory_reports_amd_smi_failure(tmp_path):
    runner = RecordingCommandRunner(
        [subprocess.CompletedProcess([], 2, "", "amdgpu driver not loaded")]
    )

    with pytest.raises(SetupError, match="unavailable: amdgpu driver not loaded"):
        AmdSmiInventory(runner=runner, kfd_root=tmp_path).list_devices()


@pytest.mark.parametrize(
    ("market_name", "required", "matches"),
    [
        ("AMD Instinct MI355X", "MI355X", True),
        ("AMD Instinct MI355X", "gfx950", True),
        ("AMD Instinct MI355X", "MI300X", False),
        # One amd-smi release names a MI355X only "AMD Radeon Graphics".
        ("AMD Radeon Graphics", "gfx950", True),
        ("AMD Radeon Graphics", "MI355X", False),
    ],
)
def test_amd_gpu_type_matches_market_name_or_isa(
    tmp_path, market_name, required, matches
):
    inventory = amd_inventory(
        tmp_path, amd_smi_list(), amd_smi_asic(market_name)
    ).list_devices()
    requirement = GPURequirement(count=1, name=required)

    if matches:
        allocation = resolve_allocation(
            requirement, requested=("1",), inventory=inventory
        )
        assert allocation.devices == (inventory[1],)
    else:
        with pytest.raises(SetupError, match="does not match"):
            resolve_allocation(requirement, requested=("1",), inventory=inventory)


def test_one_allocation_cannot_mix_gpu_vendors(tmp_path):
    amd = amd_inventory(tmp_path, amd_smi_list()).list_devices()[0]

    with pytest.raises(ValueError, match="mix vendors"):
        GPUAllocation(devices=(eight_h100_inventory()[1], amd))


def test_amd_allocation_is_void_to_the_nvidia_runtime(tmp_path):
    amd = amd_inventory(tmp_path, amd_smi_list()).list_devices()

    assert nvidia_visible_devices_value(GPUAllocation(devices=amd)) == "void"


def test_amd_quiescence_reads_kfd_contexts_of_work_pids_only(tmp_path):
    inventory = amd_inventory(tmp_path, amd_smi_list()).list_devices()
    kfd_root = tmp_path / "kfd"
    for pid, gpu_ids in {41: (17010,), 42: (42583,), 43: (17010, 42583)}.items():
        process = kfd_root / "proc" / str(pid)
        process.mkdir()
        for gpu_id in gpu_ids:
            (process / f"vram_{gpu_id}").write_text("0\n")
    allocated = GPUAllocation(devices=(inventory[1],))

    # 42 holds only an unallocated GPU and 43 belongs to another container.
    assert_work_gpu_quiescent(allocated, TopContainer((42, 99)), kfd_root=kfd_root)
    with pytest.raises(SubmissionError, match=r"PID 41 is still active"):
        assert_work_gpu_quiescent(allocated, TopContainer((41,)), kfd_root=kfd_root)


def test_amd_quiescence_fails_closed_when_an_allocated_gpu_left_topology(tmp_path):
    inventory = amd_inventory(tmp_path, amd_smi_list()).list_devices()
    kfd_root = tmp_path / "kfd"
    (kfd_root / "topology" / "nodes" / "2" / "gpu_id").write_text("9\n")

    with pytest.raises(SubmissionError, match="17010 appeared 0 times"):
        assert_work_gpu_quiescent(
            GPUAllocation(devices=(inventory[1],)),
            TopContainer((41,)),
            kfd_root=kfd_root,
        )


@pytest.mark.parametrize(
    ("tools", "kfd", "expected"),
    [
        ({"nvidia-smi", "amd-smi"}, True, NvidiaSmiInventory),
        ({"amd-smi"}, True, AmdSmiInventory),
        ({"amd-smi"}, False, NvidiaSmiInventory),
        (set(), False, NvidiaSmiInventory),
    ],
)
def test_host_inventory_keeps_nvidia_default_and_finds_amd(
    tmp_path, tools, kfd, expected
):
    kfd_device = tmp_path / "kfd"
    if kfd:
        kfd_device.touch()

    inventory = host_gpu_inventory(
        which=lambda tool: f"/usr/bin/{tool}" if tool in tools else None,
        kfd_device=kfd_device,
    )

    assert type(inventory) is expected
