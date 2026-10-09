"""Explicit NVIDIA and AMD inventory, allocation, and Work-process checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
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
AMD_KFD_DEVICE = Path("/dev/kfd")
KFD_SYSFS_ROOT = Path("/sys/class/kfd/kfd")
_PCI_BDF = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\Z")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=False, text=True)


def nvidia_visible_devices_value(allocation: GPUAllocation) -> str:
    """Encode exact Engine-owned UUID visibility for NVIDIA container runtime.

    An AMD allocation is void here: its containers reach only the device nodes
    they are given, and an NVIDIA default runtime must not add any.
    """
    if not allocation.devices or allocation.vendor == "amd":
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


class AmdSmiInventory:
    """Read AMD GPUs from ``amd-smi`` and bind each to its KFD topology node.

    ``amd-smi`` owns the operator-facing index, UUID, and model name; the KFD
    topology owns the render node a container is given and the KFD GPU id that
    process checks read. The two are joined by PCI address and must agree.
    """

    def __init__(
        self,
        *,
        runner: CommandRunner = _run,
        kfd_root: Path = KFD_SYSFS_ROOT,
        dev_root: Path = Path("/dev"),
    ) -> None:
        self._runner = runner
        self._kfd_root = kfd_root
        self._dev_root = dev_root

    def list_devices(self) -> tuple[GPUDevice, ...]:
        listed = self._query(["amd-smi", "list", "--json"])
        asic = self._query(["amd-smi", "static", "--asic", "--json"])
        devices: list[GPUDevice] = []
        try:
            names = _amd_device_names(asic)
            nodes = _kfd_gpu_nodes_by_bdf(self._kfd_root)
            if not isinstance(listed, list):
                raise ValueError("amd-smi list output is not a list")
            for entry in listed:
                index = _json_int(entry, "gpu")
                uuid = _json_str(entry, "uuid")
                bdf = _json_str(entry, "bdf").lower()
                if entry.get("partition_id", 0) != 0:
                    raise SetupError(
                        f"AMD GPU {index} is a compute partition; only "
                        "unpartitioned (SPX) GPUs are supported"
                    )
                matches = nodes.get(bdf, ())
                if len(matches) != 1:
                    raise ValueError(
                        f"GPU {index} at {bdf} matches {len(matches)} KFD nodes"
                    )
                kfd_gpu_id, render_minor = matches[0]
                render_node = self._dev_root / "dri" / f"renderD{render_minor}"
                if not render_node.exists():
                    raise ValueError(f"render node {render_node} is missing")
                if index not in names:
                    raise ValueError(f"amd-smi reported no ASIC data for GPU {index}")
                devices.append(
                    GPUDevice(
                        index=index,
                        uuid=uuid,
                        name=names[index],
                        vendor="amd",
                        render_node=render_node,
                        kfd_gpu_id=kfd_gpu_id,
                    )
                )
        except (OSError, TypeError, ValueError) as error:
            raise SetupError(f"ambiguous AMD GPU inventory: {error}") from error
        if not devices:
            raise SetupError("AMD GPU inventory returned no devices")
        if len({device.index for device in devices}) != len(devices):
            raise SetupError("AMD GPU inventory contains duplicate indexes")
        try:
            return GPUAllocation(devices=tuple(devices)).devices
        except ValueError as error:
            raise SetupError(f"invalid AMD GPU inventory: {error}") from error

    def _query(self, command: list[str]) -> object:
        try:
            result = self._runner(command)
        except OSError as error:
            raise SetupError(f"AMD GPU inventory is unavailable: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or "amd-smi exited unsuccessfully"
            raise SetupError(f"AMD GPU inventory is unavailable: {detail}")
        try:
            return json.loads(result.stdout)
        except ValueError as error:
            raise SetupError(f"ambiguous AMD GPU inventory: {error}") from error


def _json_int(entry: object, key: str) -> int:
    value = entry.get(key) if isinstance(entry, dict) else None
    if type(value) is not int:
        raise ValueError(f"amd-smi field {key!r} is not an integer")
    return value


def _json_str(entry: object, key: str) -> str:
    value = entry.get(key) if isinstance(entry, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"amd-smi field {key!r} is empty")
    return value.strip()


def _amd_device_names(payload: object) -> dict[int, str]:
    """Name each GPU by market name and ISA, e.g. ``AMD Instinct MI355X (gfx950)``.

    The market name comes from a lookup table that differs between amd-smi
    releases (one reports a MI355X as ``AMD Radeon Graphics``), so the ISA is
    kept beside it and a task may require either.
    """
    entries = payload.get("gpu_data") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError("amd-smi static output has no gpu_data list")
    names: dict[int, str] = {}
    for entry in entries:
        index = _json_int(entry, "gpu")
        asic = entry.get("asic")
        market = _json_str(asic, "market_name")
        isa = _json_str(asic, "target_graphics_version")
        if index in names:
            raise ValueError(f"amd-smi static output repeats GPU {index}")
        names[index] = f"{market} ({isa})"
    return names


def _kfd_gpu_nodes_by_bdf(kfd_root: Path) -> dict[str, tuple[tuple[int, int], ...]]:
    """Map each PCI address to its KFD ``(gpu_id, drm_render_minor)`` nodes."""
    found: dict[str, list[tuple[int, int]]] = {}
    for node in sorted((kfd_root / "topology" / "nodes").iterdir()):
        gpu_id = int((node / "gpu_id").read_text().strip())
        if gpu_id == 0:
            continue
        properties: dict[str, int] = {}
        for line in (node / "properties").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in {"domain", "location_id", "drm_render_minor"}:
                properties[key] = int(value)
        missing = {"domain", "location_id", "drm_render_minor"} - properties.keys()
        if missing:
            raise ValueError(f"KFD node {node.name} lacks {sorted(missing)[0]}")
        location = properties["location_id"]
        bdf = (
            f"{properties['domain']:04x}:{location >> 8:02x}:"
            f"{(location >> 3) & 0x1F:02x}.{location & 0x7}"
        )
        if _PCI_BDF.fullmatch(bdf) is None:
            raise ValueError(f"KFD node {node.name} has an invalid PCI address")
        found.setdefault(bdf, []).append((gpu_id, properties["drm_render_minor"]))
    return {bdf: tuple(entries) for bdf, entries in found.items()}


def amd_container_device_nodes(
    allocation: GPUAllocation, *, kfd_device: Path | None = None
) -> tuple[Path, ...]:
    """Host device nodes that give a container exactly an AMD allocation."""
    if allocation.vendor != "amd":
        raise SetupError("AMD device nodes require an AMD GPU allocation")
    render_nodes = tuple(
        device.render_node
        for device in allocation.devices
        if device.render_node is not None
    )
    return (kfd_device or AMD_KFD_DEVICE, *render_nodes)


def host_gpu_inventory(
    *,
    which: Callable[[str], str | None] = shutil.which,
    kfd_device: Path | None = None,
) -> NvidiaSmiInventory | AmdSmiInventory:
    """Choose this host's GPU inventory; NVIDIA remains the default."""
    if (
        which("nvidia-smi") is None
        and which("amd-smi") is not None
        and (kfd_device or AMD_KFD_DEVICE).exists()
    ):
        return AmdSmiInventory()
    return NvidiaSmiInventory()


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


def _amd_gpu_processes(
    allocation: GPUAllocation, kfd_root: Path
) -> tuple[tuple[str, int], ...]:
    """Host PIDs holding a KFD context on each allocated AMD GPU."""
    try:
        topology_ids = [
            int((node / "gpu_id").read_text().strip())
            for node in (kfd_root / "topology" / "nodes").iterdir()
        ]
        for device in allocation.devices:
            count = topology_ids.count(device.kfd_gpu_id or 0)
            if count != 1:
                raise ValueError(
                    f"allocated KFD GPU id {device.kfd_gpu_id} appeared {count} times"
                )
        attached = {
            f"vram_{device.kfd_gpu_id}": device.uuid for device in allocation.devices
        }
        processes: list[tuple[str, int]] = []
        for process in (kfd_root / "proc").iterdir():
            try:
                entries = {entry.name for entry in process.iterdir()}
            except FileNotFoundError:
                continue  # the process released KFD while being listed
            pid = int(process.name)
            processes.extend(
                (uuid, pid) for entry, uuid in attached.items() if entry in entries
            )
    except (OSError, ValueError) as error:
        raise SubmissionError(f"ambiguous GPU process query output: {error}") from error
    return tuple(processes)


def assert_work_gpu_quiescent(
    allocation: GPUAllocation,
    work_container: Any,
    *,
    runner: CommandRunner = _run,
    kfd_root: Path = KFD_SYSFS_ROOT,
) -> None:
    """Reject only allocated-GPU processes owned by the Work container."""
    work_pids = _container_pids(work_container)
    assert_gpu_processes_quiescent(
        allocation, work_pids, runner=runner, kfd_root=kfd_root
    )


def assert_gpu_processes_quiescent(
    allocation: GPUAllocation,
    work_pids: frozenset[int],
    *,
    runner: CommandRunner = _run,
    kfd_root: Path = KFD_SYSFS_ROOT,
) -> None:
    """Check an explicitly attributed host PID set against allocated GPUs."""
    allocated = frozenset(allocation.uuids)
    if allocation.vendor == "amd":
        processes = _amd_gpu_processes(allocation, kfd_root)
    else:
        processes = _gpu_processes(runner, allocated)
    for uuid, pid in processes:
        if uuid in allocated and pid in work_pids:
            raise SubmissionError(
                f"Work container GPU process PID {pid} is still active on {uuid}"
            )


__all__ = [
    "AMD_KFD_DEVICE",
    "KFD_SYSFS_ROOT",
    "NVIDIA_VISIBLE_DEVICES_ENV",
    "NVIDIA_VISIBLE_DEVICES_VOID",
    "AmdSmiInventory",
    "NvidiaSmiInventory",
    "amd_container_device_nodes",
    "assert_gpu_processes_quiescent",
    "assert_work_gpu_quiescent",
    "host_gpu_inventory",
    "nvidia_visible_devices_value",
    "resolve_allocation",
    "resolve_gpu_plan",
]
