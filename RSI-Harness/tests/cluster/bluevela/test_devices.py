from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rsi_harness.cluster.bluevela.devices import allocated_cuda_devices
from rsi_harness.errors import InfrastructureError

# A PBS GB200 node: nvidia-smi index, UUID and PCI bus.
NODE = (
    (0, "GPU-309f6df9-8a3c-89b3-174b-a79af6632cbc", "00000008:01:00.0"),
    (1, "GPU-8dac9b9b-9f77-8bfc-8762-b41f1f00c630", "00000009:01:00.0"),
    (2, "GPU-0c98ab18-3614-dddc-9988-0fdc1d264b64", "00000018:01:00.0"),
    (3, "GPU-a953f662-a4c3-17f1-ef7f-92d903570f74", "00000019:01:00.0"),
)
INVENTORY = ("nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader")
# A node whose driver records cannot be read: the UUID path skips its check.
NO_DRIVER = Path("/nonexistent/proc/driver/nvidia/gpus")


def _smi(rows=NODE, *, code: int = 0, calls: list | None = None):
    stdout = "".join(f"{index}, {uuid}, {bus}\n" for index, uuid, bus in rows)

    def runner(argv):
        if calls is not None:
            calls.append(argv)
        return subprocess.CompletedProcess(argv, code, stdout, "smi failed")

    return runner


def _unused(argv):
    raise AssertionError(argv)


def _node(tmp_path: Path, gpus) -> tuple[Path, Path]:
    """Fake /proc/driver/nvidia/gpus and /dev: (bus, minor, openable) per GPU.
    A GPU the job cannot open is a directory (EISDIR, also for root)."""
    gpus_root, dev = tmp_path / "gpus", tmp_path / "dev"
    gpus_root.mkdir()
    dev.mkdir()
    for bus, minor, openable in gpus:
        record = gpus_root / bus.lower()[-12:]
        record.mkdir()
        (record / "information").write_text(
            f"Model: \t\t NVIDIA H100\nBus Location: \t {bus.lower()[-12:]}\n"
            f"Device Minor: \t {minor}\nGPU Excluded:\t No\n"
        )
        node = dev / f"nvidia{minor}"
        node.write_text("") if openable else node.mkdir()
    return gpus_root, dev


def test_gpu_uuids_become_nvidia_smi_indices_in_allocation_order():
    calls: list = []
    raw = ",".join(uuid for _, uuid, _ in reversed(NODE))

    result = allocated_cuda_devices(raw, runner=_smi(calls=calls), gpus_root=NO_DRIVER)

    assert result == (("3", "2", "1", "0"), True)
    assert calls == [INVENTORY]


def test_a_unique_uuid_prefix_maps_as_cuda_accepts_it():
    result = allocated_cuda_devices(
        "GPU-8DAC9B9B, GPU-a953", runner=_smi(), gpus_root=NO_DRIVER
    )

    assert result.devices == ("1", "3")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0,1",
        " 2 , 3 ",
        "MIG-5f753abc-1811-842d-aa04-3a2f85029eae",
        "MIG-5f753abc-1811-842d-aa04-3a2f85029eae,GPU-309f6df9",
    ],
)
def test_indices_and_mig_uuids_pass_through_without_nvidia_smi(raw):
    result = allocated_cuda_devices(raw, runner=_unused)

    assert result.devices == tuple(
        item.strip() for item in raw.split(",") if item.strip()
    )
    assert not result.derived


@pytest.mark.parametrize(
    ("raw", "runner", "match"),
    [
        ("GPU-00000000-0000", _smi(), "is not a GPU nvidia-smi lists"),
        ("GPU-", _smi(), "is ambiguous"),
        ("0,GPU-309f6df9", _smi(), "mixes GPU UUIDs"),
        ("GPU-309f6df9", _smi(code=9), "smi failed"),
        ("GPU-309f6df9", _smi(()), "lists no GPU"),
        ("GPU-309f6df9", _smi(NODE + (NODE[0],)), "twice"),
        ("GPU-309f6df9", _smi(((0, "N/A", "N/A"),)), "unexpected nvidia-smi"),
        (
            "GPU-309f6df9",
            _smi(((1, NODE[0][1], NODE[0][2]), (0, NODE[1][1], NODE[1][2]))),
            "not 0..N-1 in PCI bus order",
        ),
    ],
)
def test_unmappable_gpu_uuids_are_refused(raw, runner, match):
    with pytest.raises(InfrastructureError, match=match):
        allocated_cuda_devices(raw, runner=runner, gpus_root=NO_DRIVER)


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "No such file or directory", "nvidia-smi"),
        subprocess.TimeoutExpired(INVENTORY, 60),
    ],
)
def test_an_unusable_nvidia_smi_is_an_infrastructure_error(error):
    def runner(argv):
        raise error

    with pytest.raises(InfrastructureError, match="cannot list the job's GPUs"):
        allocated_cuda_devices("GPU-309f6df9", runner=runner, gpus_root=NO_DRIVER)


def test_unset_devices_are_the_gpus_the_job_can_open(tmp_path: Path):
    # A shared PJM node (gpu=1): /dev lists four GPUs; the device cgroup
    # opens one, which nvidia-smi alone lists, as index 0. Minors need not
    # match indices.
    gpus_root, dev = _node(
        tmp_path,
        (
            ("0000:18:00.0", 2, False),
            ("0000:2b:00.0", 0, True),
            ("0000:9a:00.0", 1, False),
            ("0000:ab:00.0", 3, False),
        ),
    )
    smi = _smi(((0, "GPU-8c24fc69-ff0e-b52d-f068-79ecaced2449", "00000000:2B:00.0"),))

    result = allocated_cuda_devices(None, runner=smi, gpus_root=gpus_root, dev_root=dev)

    assert result == (("0",), True)


def _whole_node(tmp_path: Path) -> tuple[Path, Path]:
    minors = (2, 3, 0, 1)
    return _node(
        tmp_path,
        tuple((bus, minor, True) for (_, _, bus), minor in zip(NODE, minors)),
    )


def test_unset_devices_of_an_exclusive_job_are_every_gpu(tmp_path: Path):
    gpus_root, dev = _whole_node(tmp_path)

    result = allocated_cuda_devices(
        None, exclusive=True, runner=_smi(), gpus_root=gpus_root, dev_root=dev
    )

    assert result == (("0", "1", "2", "3"), True)


def test_unset_devices_without_a_fence_or_an_exclusive_job_are_refused(
    tmp_path: Path,
):
    # Every GPU opens, so nothing shows these are the job's own.
    gpus_root, dev = _whole_node(tmp_path)

    with pytest.raises(InfrastructureError, match="nothing fences this job's GPUs"):
        allocated_cuda_devices(None, runner=_smi(), gpus_root=gpus_root, dev_root=dev)


def test_an_explicit_allocation_of_every_gpu_needs_no_outside_fenced_device(
    tmp_path: Path,
):
    gpus_root, dev = _whole_node(tmp_path)

    result = allocated_cuda_devices(
        None, allocation_size=4, runner=_smi(), gpus_root=gpus_root, dev_root=dev
    )

    assert result == (("0", "1", "2", "3"), True)


@pytest.mark.parametrize("allocation_size", [1, 2, 3, 5, 8])
def test_an_allocation_count_never_authorizes_a_different_unfenced_inventory(
    tmp_path: Path,
    allocation_size: int,
):
    gpus_root, dev = _whole_node(tmp_path)

    with pytest.raises(InfrastructureError, match="nothing fences this job's GPUs"):
        allocated_cuda_devices(
            None,
            allocation_size=allocation_size,
            runner=_smi(),
            gpus_root=gpus_root,
            dev_root=dev,
        )


def test_an_allocation_count_must_also_match_the_driver_inventory(tmp_path: Path):
    gpus_root, dev = _whole_node(tmp_path)

    with pytest.raises(InfrastructureError, match="nothing fences this job's GPUs"):
        allocated_cuda_devices(
            None,
            allocation_size=2,
            runner=_smi(NODE[:2]),
            gpus_root=gpus_root,
            dev_root=dev,
        )


def test_gpu_uuids_refuse_a_listed_gpu_the_job_cannot_open(tmp_path: Path):
    # CUDA would number the open ones 0..N-1, unlike the host indices.
    gpus_root, dev = _node(
        tmp_path,
        tuple((bus, index, index != 1) for index, _, bus in NODE),
    )
    raw = f"{NODE[1][1]},{NODE[3][1]}"

    with pytest.raises(InfrastructureError, match="cannot open.*8dac9b9b"):
        allocated_cuda_devices(raw, runner=_smi(), gpus_root=gpus_root, dev_root=dev)


def test_gpu_uuids_map_when_every_listed_gpu_opens(tmp_path: Path):
    gpus_root, dev = _whole_node(tmp_path)

    result = allocated_cuda_devices(
        NODE[2][1], runner=_smi(), gpus_root=gpus_root, dev_root=dev
    )

    assert result == (("2",), True)


def test_unset_devices_refuse_a_listed_gpu_the_job_cannot_open(tmp_path: Path):
    # CUDA would skip it and renumber the rest, unlike nvidia-smi.
    gpus_root, dev = _node(
        tmp_path,
        tuple((bus, index, index != 1) for index, _, bus in NODE),
    )

    with pytest.raises(InfrastructureError, match="cannot open.*8dac9b9b"):
        allocated_cuda_devices(None, runner=_smi(), gpus_root=gpus_root, dev_root=dev)


def test_unset_devices_refuse_a_gpu_without_a_driver_record(tmp_path: Path):
    gpus_root, dev = _node(tmp_path, ((NODE[0][2], 0, True),))

    with pytest.raises(InfrastructureError, match="does not"):
        allocated_cuda_devices(None, runner=_smi(), gpus_root=gpus_root, dev_root=dev)


def test_unset_devices_refuse_a_malformed_driver_record(tmp_path: Path):
    gpus_root, dev = _node(tmp_path, ((NODE[0][2], 0, True),))
    (next(gpus_root.iterdir()) / "information").write_text("Model: H100\n")

    with pytest.raises(InfrastructureError, match="unexpected NVIDIA GPU record"):
        allocated_cuda_devices(
            None, runner=_smi(NODE[:1]), gpus_root=gpus_root, dev_root=dev
        )
