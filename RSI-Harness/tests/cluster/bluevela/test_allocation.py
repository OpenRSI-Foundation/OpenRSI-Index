from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

import pytest

from rsi_harness.cluster.bluevela.resources import (
    MultiNodeResources,
    PhaseResources,
)
from rsi_harness.errors import InfrastructureError


def _module():
    source = (
        Path(__file__).resolve().parents[3]
        / "src/rsi_harness/cluster/bluevela/allocation.py"
    )
    assert source.is_file(), "Blue Vela allocation authority is missing"
    return importlib.import_module("rsi_harness.cluster.bluevela.allocation")


def _resources() -> MultiNodeResources:
    return MultiNodeResources(
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
    )


def test_inventory_preserves_lsf_order() -> None:
    allocation = _module()

    inventory = allocation.parse_lsb_mcpu_hosts(
        "work-a 8 work-b 8 judge-a 8",
        expected_hosts=3,
        expected_slots=8,
    )

    assert tuple(item.host for item in inventory) == (
        "work-a",
        "work-b",
        "judge-a",
    )


@pytest.mark.parametrize(
    ("raw", "match"),
    (
        ("", "missing"),
        ("node-a", "HOST SLOTS pairs"),
        ("node-a x", "integer slots"),
        ("node-a 4 node-b 8 node-c 8", "expected 8 slots"),
        ("node-a 8 node-a 8 node-c 8", "duplicate host"),
        ("node-a 8", "expected 3 hosts"),
    ),
)
def test_invalid_inventory_is_rejected(raw: str, match: str) -> None:
    allocation = _module()

    with pytest.raises(InfrastructureError, match=match):
        allocation.parse_lsb_mcpu_hosts(
            raw,
            expected_hosts=3,
            expected_slots=8,
        )


def test_probed_nodes_are_partitioned_into_immutable_disjoint_pools() -> None:
    allocation = _module()
    inventory = allocation.parse_lsb_mcpu_hosts(
        "work-a 8 work-b 8 judge-a 8",
        expected_hosts=3,
        expected_slots=8,
    )
    probes = {
        host: allocation.NodeProbe(
            ipv4=f"10.0.0.{index}",
            cuda_devices=tuple(f"GPU-{index}-{gpu}" for gpu in range(8)),
            sif_sha256="a" * 64,
            gpfs_visible=True,
            infiniband_visible=True,
            tmp_free_mb=100000,
        )
        for index, host in enumerate(
            ("work-a", "work-b", "judge-a"),
            start=1,
        )
    }

    pools = allocation.freeze_pools(
        run_id="run-1",
        resources=_resources(),
        inventory=inventory,
        probes=probes,
        expected_sif_sha256="a" * 64,
    )

    assert tuple(node.host for node in pools.work) == ("work-a", "work-b")
    assert tuple(node.host for node in pools.verifier) == ("judge-a",)
    assert set(node.host for node in pools.work).isdisjoint(
        node.host for node in pools.verifier
    )
    assert len(pools.digest) == 64


@pytest.mark.parametrize(
    "ipv4",
    ("0.0.0.0", "127.0.0.1", "169.254.10.2", "224.0.0.1", "::1"),
)
def test_unsafe_node_addresses_fail_closed(ipv4: str) -> None:
    allocation = _module()
    inventory = allocation.parse_lsb_mcpu_hosts(
        "work-a 8 work-b 8 judge-a 8",
        expected_hosts=3,
        expected_slots=8,
    )
    probes = {
        host: allocation.NodeProbe(
            ipv4=ipv4 if host == "work-a" else f"10.0.0.{index}",
            cuda_devices=tuple(f"GPU-{index}-{gpu}" for gpu in range(8)),
            sif_sha256="a" * 64,
            gpfs_visible=True,
            infiniband_visible=True,
            tmp_free_mb=100000,
        )
        for index, host in enumerate(
            ("work-a", "work-b", "judge-a"),
            start=1,
        )
    }

    with pytest.raises(InfrastructureError, match="safe unique IPv4"):
        allocation.freeze_pools(
            run_id="run-1",
            resources=_resources(),
            inventory=inventory,
            probes=probes,
            expected_sif_sha256="a" * 64,
        )


def test_probe_rejects_wrong_gpu_count_digest_storage_or_infiniband() -> None:
    allocation = _module()
    inventory = allocation.parse_lsb_mcpu_hosts(
        "work-a 8 work-b 8 judge-a 8",
        expected_hosts=3,
        expected_slots=8,
    )
    valid = allocation.NodeProbe(
        ipv4="10.0.0.1",
        cuda_devices=tuple(f"GPU-{gpu}" for gpu in range(8)),
        sif_sha256="a" * 64,
        gpfs_visible=True,
        infiniband_visible=True,
        tmp_free_mb=100000,
    )
    invalid = (
        valid.model_copy(update={"cuda_devices": ("GPU-0",)}),
        valid.model_copy(update={"sif_sha256": "b" * 64}),
        valid.model_copy(update={"gpfs_visible": False}),
        valid.model_copy(update={"infiniband_visible": False}),
        valid.model_copy(update={"tmp_free_mb": 1}),
    )

    for probe in invalid:
        probes = {item.host: valid for item in inventory}
        probes["work-a"] = probe
        with pytest.raises(InfrastructureError):
            allocation.freeze_pools(
                run_id="run-1",
                resources=_resources(),
                inventory=inventory,
                probes=probes,
                expected_sif_sha256="a" * 64,
            )


def test_probe_and_partition_runs_one_fixed_blaunch_probe_per_host(
    tmp_path: Path,
) -> None:
    allocation = _module()
    assert hasattr(allocation, "probe_and_partition"), (
        "allocation discovery must probe every LSF host before Agent startup"
    )
    inventory = allocation.parse_lsb_mcpu_hosts(
        "work-a 8 work-b 8 judge-a 8",
        expected_hosts=3,
        expected_slots=8,
    )
    calls: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        host = argv[2]
        host_index = ("work-a", "work-b", "judge-a").index(host) + 1
        output = json.dumps(
            {
                "ipv4": f"10.0.0.{host_index}",
                "cuda_devices": [
                    f"GPU-{host_index}-{gpu}" for gpu in range(8)
                ],
                "sif_sha256": "a" * 64,
                "gpfs_visible": True,
                "infiniband_visible": True,
                "tmp_free_mb": 100000,
            }
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    pools = allocation.probe_and_partition(
        run_id="run-1",
        resources=_resources(),
        inventory=inventory,
        sif_path=(tmp_path / "task.sif").resolve(),
        run_dir=tmp_path.resolve(),
        temp_root=Path("/tmp"),
        expected_sif_sha256="a" * 64,
        remote_binary="/site/bin/remote-launch",
        remote_host_flag="--host",
        runner=runner,
    )

    assert tuple(node.host for node in pools.work) == ("work-a", "work-b")
    assert tuple(node.host for node in pools.verifier) == ("judge-a",)
    assert len(calls) == 3
    assert tuple(call[:3] for call in calls) == (
        ("/site/bin/remote-launch", "--host", "work-a"),
        ("/site/bin/remote-launch", "--host", "work-b"),
        ("/site/bin/remote-launch", "--host", "judge-a"),
    )
    assert all(
        "rsi_harness.cluster.bluevela.allocation" in call for call in calls
    )


def test_probe_and_partition_retries_a_transient_remote_launch_failure(
    tmp_path: Path,
) -> None:
    allocation = _module()
    inventory = allocation.parse_lsb_mcpu_hosts(
        "work-a 8 work-b 8 judge-a 8",
        expected_hosts=3,
        expected_slots=8,
    )
    calls: list[str] = []
    sleeps: list[float] = []

    def runner(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        host = argv[2]
        calls.append(host)
        if host == "work-b" and calls.count(host) == 1:
            return subprocess.CompletedProcess(
                argv,
                1,
                "",
                "lsb_pjob_getAckReturn failed",
            )
        host_index = ("work-a", "work-b", "judge-a").index(host) + 1
        output = json.dumps(
            {
                "ipv4": f"10.0.0.{host_index}",
                "cuda_devices": [
                    f"GPU-{host_index}-{gpu}" for gpu in range(8)
                ],
                "sif_sha256": "a" * 64,
                "gpfs_visible": True,
                "infiniband_visible": True,
                "tmp_free_mb": 100000,
            }
        )
        return subprocess.CompletedProcess(argv, 0, output, "")

    pools = allocation.probe_and_partition(
        run_id="run-1",
        resources=_resources(),
        inventory=inventory,
        sif_path=(tmp_path / "task.sif").resolve(),
        run_dir=tmp_path.resolve(),
        temp_root=Path("/tmp"),
        expected_sif_sha256="a" * 64,
        remote_binary="/site/bin/remote-launch",
        remote_host_flag="--host",
        runner=runner,
        probe_attempts=3,
        probe_retry_seconds=2.0,
        sleeper=sleeps.append,
    )

    assert tuple(node.host for node in pools.work) == ("work-a", "work-b")
    assert calls == ["work-a", "work-b", "work-b", "judge-a"]
    assert sleeps == [2.0]
