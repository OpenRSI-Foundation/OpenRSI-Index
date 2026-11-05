"""The allocation's GPUs as device indices CUDA and nvidia-smi agree on.

Schedulers name a job's GPUs in different ways: indices (LSF, Slurm), GPU
UUIDs (PBS, the Kubernetes device plugin), or not at all, fencing the job
with a device cgroup that opens only its GPUs while ``/dev`` lists every
GPU. Phases get indices, as tools that parse ``CUDA_VISIBLE_DEVICES`` as
integers (vLLM) need. Unset is trusted when a device cgroup visibly fences
the job, the job is exclusive, or its explicit GPU allocation owns every
GPU on the node.

The index space: nvidia-smi and CUDA both enumerate only the GPUs the job
can open, and inside an Apptainer ``--nv`` container too (same device
nodes, same cgroup). nvidia-smi numbers them 0..N-1 in PCI bus order, CUDA
by default fastest first, so derived indices go with
``CUDA_DEVICE_ORDER=PCI_BUS_ID``. Indices the scheduler set are passed on
unchanged, in its own order.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from rsi_harness.errors import InfrastructureError

CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]
_INVENTORY = (
    "nvidia-smi",
    "--query-gpu=index,uuid,pci.bus_id",
    "--format=csv,noheader",
)
GPUS_ROOT = Path("/proc/driver/nvidia/gpus")
DEV_ROOT = Path("/dev")
_PCI = re.compile(r"([0-9A-Fa-f]{4,8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])")


class AllocatedDevices(NamedTuple):
    devices: tuple[str, ...]
    # True when the Engine derived the indices: phases then need
    # CUDA_DEVICE_ORDER=PCI_BUS_ID to count them as nvidia-smi does.
    derived: bool


class _GPU(NamedTuple):
    index: int
    uuid: str
    bus: tuple[int, ...]


def _run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True, timeout=60)


def _bus(value: str) -> tuple[int, ...] | None:
    # nvidia-smi prints an 8-digit PCI domain, procfs 4.
    match = _PCI.fullmatch(value.strip())
    return None if match is None else tuple(int(part, 16) for part in match.groups())


def _inventory(runner: CommandRunner) -> tuple[_GPU, ...]:
    try:
        completed = runner(_INVENTORY)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise InfrastructureError(f"cannot list the job's GPUs: {error}") from error
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise InfrastructureError(f"cannot list the job's GPUs: nvidia-smi: {detail}")
    gpus: list[_GPU] = []
    for row in completed.stdout.splitlines():
        if not row.strip():
            continue
        parts = [part.strip() for part in row.split(",")]
        bus = _bus(parts[2]) if len(parts) == 3 else None
        if bus is None or not parts[0].isdecimal() or not parts[1].startswith("GPU-"):
            raise InfrastructureError(f"unexpected nvidia-smi GPU row: {row!r}")
        gpus.append(_GPU(int(parts[0]), parts[1].lower(), bus))
    if not gpus:
        raise InfrastructureError("nvidia-smi lists no GPU in this job")
    if len({gpu.uuid for gpu in gpus}) != len(gpus):
        raise InfrastructureError("nvidia-smi lists one GPU UUID twice")
    # The indices CUDA gives under PCI_BUS_ID.
    if [gpu.index for gpu in sorted(gpus, key=lambda gpu: gpu.bus)] != list(
        range(len(gpus))
    ):
        raise InfrastructureError(
            "nvidia-smi GPU indices are not 0..N-1 in PCI bus order: "
            + ", ".join(f"{gpu.index}={gpu.uuid}" for gpu in gpus)
        )
    return tuple(gpus)


def _by_uuid(selector: str, gpus: tuple[_GPU, ...]) -> str:
    # CUDA accepts any unique UUID prefix.
    matches = [gpu for gpu in gpus if gpu.uuid.startswith(selector.lower())]
    if len(matches) != 1:
        state = "is ambiguous" if matches else "is not a GPU nvidia-smi lists"
        raise InfrastructureError(f"CUDA_VISIBLE_DEVICES entry {selector} {state}")
    return str(matches[0].index)


def _device_minors(gpus_root: Path) -> dict[tuple[int, ...], int]:
    minors: dict[tuple[int, ...], int] = {}
    try:
        files = sorted(gpus_root.glob("*/information"))
        texts = [path.read_text() for path in files]
    except OSError as error:
        raise InfrastructureError(f"cannot read {gpus_root}: {error}") from error
    for path, text in zip(files, texts, strict=True):
        bus = re.search(r"^Bus Location:\s*(\S+)", text, re.MULTILINE)
        minor = re.search(r"^Device Minor:\s*(\d+)", text, re.MULTILINE)
        key = None if bus is None else _bus(bus.group(1))
        if key is None or minor is None or key in minors:
            raise InfrastructureError(f"unexpected NVIDIA GPU record {path}")
        minors[key] = int(minor.group(1))
    return minors


def _openable(device: Path) -> bool:
    try:
        os.close(os.open(device, os.O_RDWR | os.O_CLOEXEC))
    except OSError:
        return False
    return True


def _check_openable(
    gpus: tuple[_GPU, ...],
    gpus_root: Path,
    dev_root: Path,
    *,
    allocation_size: int | None = None,
) -> bool:
    """Check access, then prove a fence or an allocation of every node GPU."""
    minors = _device_minors(gpus_root)
    for gpu in gpus:
        if gpu.bus not in minors:
            raise InfrastructureError(
                f"nvidia-smi lists GPU {gpu.uuid}, which {gpus_root} does not"
            )
    closed = sorted(
        gpu.uuid for gpu in gpus if not _openable(dev_root / f"nvidia{minors[gpu.bus]}")
    )
    if closed:
        # CUDA would skip them and renumber the rest.
        raise InfrastructureError(
            "nvidia-smi lists GPUs this job cannot open, so its indices are "
            "not CUDA's: " + ", ".join(closed)
        )
    listed = {gpu.bus for gpu in gpus}
    fenced = any(
        not _openable(dev_root / f"nvidia{minor}")
        for bus, minor in minors.items()
        if bus not in listed
    )
    # A gpu=N allocation may own all GPUs without requesting an exclusive
    # node. Then no outside GPU remains to demonstrate the device fence.
    return fenced or allocation_size == len(gpus) == len(minors)


def allocated_cuda_devices(
    raw: str | None,
    *,
    exclusive: bool = False,
    allocation_size: int | None = None,
    runner: CommandRunner | None = None,
    gpus_root: Path | None = None,
    dev_root: Path | None = None,
) -> AllocatedDevices:
    """The job's GPUs from ``CUDA_VISIBLE_DEVICES`` (``None``: unset).

    Unset, the job's GPUs are the ones it can open, which needs a device
    cgroup that shows (some GPU the driver lists cannot be opened), an
    ``exclusive`` job, or a scheduler's explicit ``allocation_size`` that
    matches every GPU the driver and nvidia-smi list."""
    gpus_root = GPUS_ROOT if gpus_root is None else gpus_root
    dev_root = DEV_ROOT if dev_root is None else dev_root
    if raw is not None:
        devices = tuple(item.strip() for item in raw.split(",") if item.strip())
        # MIG instances have no index; CUDA takes their UUIDs as they are.
        if any(item.startswith("MIG-") for item in devices) or not any(
            item.startswith("GPU-") for item in devices
        ):
            return AllocatedDevices(devices, derived=False)
        if not all(item.startswith("GPU-") for item in devices):
            raise InfrastructureError(
                f"CUDA_VISIBLE_DEVICES mixes GPU UUIDs and other entries: {raw!r}"
            )
        gpus = _inventory(runner or _run)
        if gpus_root.is_dir():
            _check_openable(gpus, gpus_root, dev_root)
        return AllocatedDevices(
            tuple(_by_uuid(item, gpus) for item in devices), derived=True
        )
    gpus = _inventory(runner or _run)
    if (
        not _check_openable(gpus, gpus_root, dev_root, allocation_size=allocation_size)
        and not exclusive
    ):
        raise InfrastructureError(
            "CUDA_VISIBLE_DEVICES is unset and nothing fences this job's GPUs "
            "(every GPU on the node can be opened and the job is not "
            "exclusive); have the scheduler set it or request an exclusive node"
        )
    return AllocatedDevices(tuple(str(gpu.index) for gpu in gpus), derived=True)


__all__ = ["AllocatedDevices", "allocated_cuda_devices"]
