"""Explicit NVIDIA inventory, allocation, and Work-process checks."""

from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from typing import Any

from rsi_harness.errors import SetupError, SubmissionError
from rsi_harness.models import (
    GPUAllocation,
    GPUDevice,
    GPURequirement,
    JudgeGPUMode,
    RunGPUPlan,
)

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
NVIDIA_VISIBLE_DEVICES_ENV = "NVIDIA_VISIBLE_DEVICES"
NVIDIA_VISIBLE_DEVICES_VOID = "void"
_NVIDIA_GPU_UUID = re.compile(r"(?:GPU|MIG)-[A-Za-z0-9][A-Za-z0-9_.:/-]*\Z")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=False, text=True)


def nvidia_visible_devices_value(allocation: GPUAllocation) -> str:
    """Encode exact Engine-owned UUID visibility for NVIDIA container runtime."""
    if not allocation.devices:
        return NVIDIA_VISIBLE_DEVICES_VOID
    if any(_NVIDIA_GPU_UUID.fullmatch(uuid) is None for uuid in allocation.uuids):
        raise SetupError(
            "GPU UUID cannot be represented safely in NVIDIA visibility environment"
        )
    return ",".join(allocation.uuids)


class NvidiaSmiInventory:
    """Read stable physical GPU identifiers from ``nvidia-smi``."""

    def __init__(self, *, runner: CommandRunner = _run) -> None:
        self._runner = runner

    def list_devices(self) -> tuple[GPUDevice, ...]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = self._runner(command)
        except OSError as error:
            raise SetupError(f"NVIDIA GPU inventory is unavailable: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or "nvidia-smi exited unsuccessfully"
            raise SetupError(f"NVIDIA GPU inventory is unavailable: {detail}")

        devices: list[GPUDevice] = []
        try:
            for row in result.stdout.splitlines():
                if not row.strip():
                    continue
                index, uuid, name = (part.strip() for part in row.split(",", 2))
                if not uuid or not name:
                    raise ValueError("empty device field")
                devices.append(GPUDevice(index=int(index), uuid=uuid, name=name))
        except (TypeError, ValueError) as error:
            raise SetupError(
                f"ambiguous NVIDIA GPU inventory output: {error}"
            ) from error
        if not devices:
            raise SetupError("NVIDIA GPU inventory returned no devices")
        if len({device.index for device in devices}) != len(devices):
            raise SetupError("NVIDIA GPU inventory contains duplicate indexes")
        try:
            return GPUAllocation(devices=tuple(devices)).devices
        except ValueError as error:
            raise SetupError(f"invalid NVIDIA GPU inventory: {error}") from error


def _resolve_authorized_pool(
    requirement: GPURequirement,
    *,
    requested: Sequence[str | int],
    inventory: Sequence[GPUDevice],
) -> GPUAllocation:
    """Resolve caller selectors to one unique ordered physical GPU pool."""
    devices = tuple(inventory)
    by_uuid = {device.uuid: device for device in devices}
    by_index = {str(device.index): device for device in devices}

    if requirement.count == "all" and not requested:
        raise SetupError(
            "GPU count 'all' requires an explicit caller-selected allocation"
        )
    selectors: Sequence[str | int]
    if requested:
        selectors = requested
    elif isinstance(requirement.count, int):
        selectors = tuple(device.uuid for device in devices[: requirement.count])
    else:  # guarded above, kept exhaustive for type checkers
        selectors = ()

    selected: list[GPUDevice] = []
    for raw_selector in selectors:
        selector = str(raw_selector).strip()
        device = by_uuid.get(selector) or by_index.get(selector)
        if device is None:
            raise SetupError(f"unknown GPU selector {selector!r}")
        if device in selected:
            raise SetupError(
                f"duplicate physical GPU selector {selector!r} resolves to "
                f"{device.uuid}"
            )
        selected.append(device)

    return GPUAllocation(devices=tuple(selected))


def _validate_work_gpu_type(
    requirement: GPURequirement, allocation: GPUAllocation
) -> None:
    if requirement.name:
        expected = requirement.name.casefold()
        mismatch = next(
            (
                device
                for device in allocation.devices
                if expected not in device.name.casefold()
            ),
            None,
        )
        if mismatch is not None:
            raise SetupError(
                f"GPU {mismatch.uuid} type {mismatch.name!r} does not match "
                f"required type {requirement.name!r}"
            )


def resolve_allocation(
    requirement: GPURequirement,
    *,
    requested: Sequence[str | int],
    inventory: Sequence[GPUDevice],
) -> GPUAllocation:
    """Resolve caller selectors to an ordered, exact Work allocation."""
    if isinstance(requirement.count, int) and requirement.count > len(inventory):
        raise SetupError(
            f"task requires {requirement.count} GPUs but only {len(inventory)} "
            "are available"
        )
    if (
        isinstance(requirement.count, int)
        and requested
        and len(requested) < requirement.count
    ):
        raise SetupError(
            f"task requires exactly {requirement.count} GPUs; caller selected "
            f"{len(requested)}"
        )
    allocation = _resolve_authorized_pool(
        requirement, requested=requested, inventory=inventory
    )
    if (
        isinstance(requirement.count, int)
        and len(allocation.devices) != requirement.count
    ):
        raise SetupError(
            f"task requires exactly {requirement.count} GPUs; caller selected "
            f"{len(allocation.devices)}"
        )
    _validate_work_gpu_type(requirement, allocation)
    return allocation


def resolve_gpu_plan(
    requirement: GPURequirement,
    *,
    judge_count: int,
    requested: Sequence[str | int],
    inventory: Sequence[GPUDevice],
) -> RunGPUPlan:
    """Plan deterministic Work and Judge GPU allocations from one pool."""
    if requirement.count == 0:
        if judge_count == 0 and requested:
            raise SetupError("CPU-only Work and Judge do not accept --gpus selectors")
        if judge_count > 0 and not requested:
            raise SetupError("GPU Judge with CPU Work requires an explicit --gpus pool")
    pool = _resolve_authorized_pool(
        requirement, requested=requested, inventory=inventory
    )
    if isinstance(requirement.count, int) and len(pool.devices) < requirement.count:
        raise SetupError(
            f"task requires {requirement.count} GPUs but the authorized pool "
            f"contains {len(pool.devices)}"
        )
    work = (
        pool
        if requirement.count == "all"
        else GPUAllocation(devices=pool.devices[: requirement.count])
    )
    _validate_work_gpu_type(requirement, work)
    work_uuids = set(work.uuids)
    spares = tuple(device for device in pool.devices if device.uuid not in work_uuids)

    if judge_count == 0:
        judge = GPUAllocation()
        mode = JudgeGPUMode.FREEZE_ONLY
    elif judge_count > len(pool.devices):
        raise SetupError(
            f"Judge requires {judge_count} GPUs but the authorized pool "
            f"contains {len(pool.devices)}"
        )
    elif judge_count <= len(spares):
        judge = GPUAllocation(devices=spares[:judge_count])
        mode = JudgeGPUMode.DISJOINT
    else:
        judge_devices = (spares + work.devices)[:judge_count]
        judge = GPUAllocation(devices=judge_devices)
        mode = JudgeGPUMode.RELEASE_ALL

    return RunGPUPlan(
        authorized_pool=pool,
        work=work,
        judge=judge,
        judge_mode=mode,
    )


def _container_pids(container: Any) -> frozenset[int]:
    try:
        table = container.top(ps_args="-eo pid")
        titles = table["Titles"]
        rows = table["Processes"]
        pid_column = titles.index("PID")
        return frozenset(int(row[pid_column]) for row in rows)
    except Exception as error:
        raise SubmissionError(
            f"ambiguous Work container process list: {error}"
        ) from error


def _gpu_processes(
    runner: CommandRunner, allocated: frozenset[str]
) -> tuple[tuple[str, int], ...]:
    compute = _query_process_command(
        runner,
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
    )
    processes: list[tuple[str, int]] = []
    for row in compute.stdout.splitlines():
        if not row.strip():
            continue
        try:
            uuid, raw_pid = (part.strip() for part in row.split(",", 1))
            if not uuid or not raw_pid:
                raise ValueError("empty process field")
            processes.append((uuid, int(raw_pid)))
        except (TypeError, ValueError) as error:
            raise SubmissionError(
                f"ambiguous GPU process query output: {row!r}"
            ) from error

    graphics = _query_process_command(runner, ["nvidia-smi", "-q", "-x"])
    try:
        root = ET.fromstring(graphics.stdout)
        gpu_nodes = root.findall(".//gpu")
        gpu_uuids = tuple((gpu.findtext("uuid") or "").strip() for gpu in gpu_nodes)
        for uuid in allocated:
            if gpu_uuids.count(uuid) != 1:
                raise ValueError(
                    f"allocated UUID {uuid!r} appeared {gpu_uuids.count(uuid)} times"
                )
        for gpu, uuid in zip(gpu_nodes, gpu_uuids, strict=True):
            for process in gpu.findall("./processes/process_info"):
                process_type = (process.findtext("process_type") or "").strip()
                if "G" not in process_type:
                    continue
                raw_pid = (process.findtext("pid") or "").strip()
                if not uuid or not raw_pid:
                    raise ValueError("empty graphics process field")
                processes.append((uuid, int(raw_pid)))
    except (ET.ParseError, TypeError, ValueError) as error:
        raise SubmissionError(f"ambiguous GPU process query output: {error}") from error
    return tuple(dict.fromkeys(processes))


def _query_process_command(
    runner: CommandRunner, command: list[str]
) -> subprocess.CompletedProcess[str]:
    try:
        result = runner(command)
    except OSError as error:
        raise SubmissionError(f"GPU process query is unavailable: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or "nvidia-smi exited unsuccessfully"
        raise SubmissionError(f"GPU process query is unavailable: {detail}")
    return result


def assert_work_gpu_quiescent(
    allocation: GPUAllocation,
    work_container: Any,
    *,
    runner: CommandRunner = _run,
) -> None:
    """Reject only allocated-GPU processes owned by the Work container."""
    work_pids = _container_pids(work_container)
    allocated = frozenset(allocation.uuids)
    processes = _gpu_processes(runner, allocated)
    for uuid, pid in processes:
        if uuid in allocated and pid in work_pids:
            raise SubmissionError(
                f"Work container GPU process PID {pid} is still active on {uuid}"
            )


__all__ = [
    "NVIDIA_VISIBLE_DEVICES_ENV",
    "NVIDIA_VISIBLE_DEVICES_VOID",
    "NvidiaSmiInventory",
    "assert_work_gpu_quiescent",
    "nvidia_visible_devices_value",
    "resolve_allocation",
    "resolve_gpu_plan",
]
