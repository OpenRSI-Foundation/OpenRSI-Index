from __future__ import annotations

import subprocess

import pytest

from rsi_harness.errors import SetupError, SubmissionError
from rsi_harness.models import GPUAllocation, GPUDevice, GPURequirement
from rsi_harness.runtime.gpu import (
    NvidiaSmiInventory,
    assert_work_gpu_quiescent,
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
def test_phase_aware_gpu_plan(
    work_count, judge_count, requested, mode, work, judge
):
    plan = resolve_gpu_plan(
        GPURequirement(count=work_count),
        judge_count=judge_count,
        requested=requested,
        inventory=eight_h100_inventory(),
    )

    assert plan.judge_mode.value == mode
    assert plan.work.uuids == work
    assert plan.judge.uuids == judge


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
