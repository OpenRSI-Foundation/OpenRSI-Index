"""Fail-closed Blue Vela allocation inventory and pool authority."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import Field

from rsi_harness.cluster.bluevela.resources import MultiNodeResources
from rsi_harness.errors import InfrastructureError
from rsi_harness.models import PersistedModel

_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]{0,252}$")

CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]
Sleeper = Callable[[float], None]


class InventoryHost(PersistedModel):
    host: str
    slots: int = Field(gt=0)


class NodeProbe(PersistedModel):
    ipv4: str
    cuda_devices: tuple[str, ...]
    sif_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gpfs_visible: bool
    infiniband_visible: bool
    tmp_free_mb: int = Field(ge=0)


class AllocatedNode(PersistedModel):
    host: str
    slots: int = Field(gt=0)
    ipv4: str
    cuda_devices: tuple[str, ...]


class AllocatedPools(PersistedModel):
    run_id: str
    work: tuple[AllocatedNode, ...]
    verifier: tuple[AllocatedNode, ...]
    gpus_per_node: int = Field(gt=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


def parse_lsb_mcpu_hosts(
    raw: str,
    *,
    expected_hosts: int,
    expected_slots: int,
) -> tuple[InventoryHost, ...]:
    """Parse the ordered LSF HOST/SLOTS pairs without deduplication or fallback."""
    fields = raw.split()
    if not fields:
        raise InfrastructureError("LSB_MCPU_HOSTS is missing")
    if len(fields) % 2:
        raise InfrastructureError("LSB_MCPU_HOSTS must contain HOST SLOTS pairs")
    result: list[InventoryHost] = []
    seen: set[str] = set()
    for offset in range(0, len(fields), 2):
        host = fields[offset].split(".", 1)[0]
        if not host or _HOST.fullmatch(host) is None:
            raise InfrastructureError(
                f"LSB_MCPU_HOSTS contains an unsafe host: {fields[offset]!r}"
            )
        try:
            slots = int(fields[offset + 1])
        except ValueError as error:
            raise InfrastructureError(
                f"LSB_MCPU_HOSTS host {host} has non-integer slots"
            ) from error
        if slots != expected_slots:
            raise InfrastructureError(
                f"LSB host {host} expected {expected_slots} slots, got {slots}"
            )
        if host in seen:
            raise InfrastructureError(f"LSB inventory contains duplicate host {host}")
        seen.add(host)
        result.append(InventoryHost(host=host, slots=slots))
    if len(result) != expected_hosts:
        raise InfrastructureError(
            f"LSB inventory expected {expected_hosts} hosts, got {len(result)}"
        )
    return tuple(result)


def _safe_ipv4(value: str, *, host: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as error:
        raise InfrastructureError(
            f"host {host} has no safe unique IPv4: {value!r}"
        ) from error
    if (
        address.version != 4
        or address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
    ):
        raise InfrastructureError(
            f"host {host} has no safe unique IPv4: {value!r}"
        )
    return str(address)


def _allocated_node(
    inventory: InventoryHost,
    probe: NodeProbe,
    *,
    resources: MultiNodeResources,
    expected_sif_sha256: str,
) -> AllocatedNode:
    devices = probe.cuda_devices
    if (
        len(devices) != resources.gpus_per_node
        or any(not item for item in devices)
        or len(set(devices)) != len(devices)
    ):
        raise InfrastructureError(
            f"host {inventory.host} must expose exactly "
            f"{resources.gpus_per_node} unique allocated GPUs"
        )
    if probe.sif_sha256 != expected_sif_sha256:
        raise InfrastructureError(
            f"host {inventory.host} sees a different SIF digest"
        )
    if not probe.gpfs_visible:
        raise InfrastructureError(f"host {inventory.host} cannot see GPFS run state")
    if not probe.infiniband_visible:
        raise InfrastructureError(
            f"host {inventory.host} cannot see required InfiniBand devices"
        )
    if probe.tmp_free_mb < resources.node_tmp_mb:
        raise InfrastructureError(
            f"host {inventory.host} has {probe.tmp_free_mb} MiB node tmp; "
            f"requires {resources.node_tmp_mb} MiB"
        )
    return AllocatedNode(
        host=inventory.host,
        slots=inventory.slots,
        ipv4=_safe_ipv4(probe.ipv4, host=inventory.host),
        cuda_devices=devices,
    )


def freeze_pools(
    *,
    run_id: str,
    resources: MultiNodeResources,
    inventory: tuple[InventoryHost, ...],
    probes: Mapping[str, NodeProbe],
    expected_sif_sha256: str,
) -> AllocatedPools:
    """Validate every allocated node and freeze ordered disjoint phase pools."""
    expected_hosts = tuple(item.host for item in inventory)
    if set(probes) != set(expected_hosts):
        raise InfrastructureError(
            "node probes must match the exact frozen LSF host inventory"
        )
    nodes = tuple(
        _allocated_node(
            item,
            probes[item.host],
            resources=resources,
            expected_sif_sha256=expected_sif_sha256,
        )
        for item in inventory
    )
    work_end = resources.work.node_count
    verifier_end = work_end + resources.verifier.node_count
    if len(nodes) != resources.total_nodes or verifier_end != len(nodes):
        raise InfrastructureError(
            "validated nodes do not match the planned Work/Judge partition"
        )
    work = nodes[:work_end]
    verifier = nodes[work_end:verifier_end]
    if set(item.host for item in work) & set(item.host for item in verifier):
        raise InfrastructureError("Work and Judge host pools overlap")
    authority = {
        "run_id": run_id,
        "work": [item.model_dump(mode="json") for item in work],
        "verifier": [item.model_dump(mode="json") for item in verifier],
        "gpus_per_node": resources.gpus_per_node,
    }
    digest = hashlib.sha256(
        json.dumps(
            authority,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return AllocatedPools(digest=digest, **authority)


def _run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def probe_and_partition(
    *,
    run_id: str,
    resources: MultiNodeResources,
    inventory: tuple[InventoryHost, ...],
    sif_path: Path,
    run_dir: Path,
    temp_root: Path,
    expected_sif_sha256: str,
    remote_binary: str = "blaunch",
    remote_host_flag: str = "-z",
    runner: CommandRunner = _run,
    probe_attempts: int = 3,
    probe_retry_seconds: float = 2.0,
    sleeper: Sleeper = time.sleep,
) -> AllocatedPools:
    """Run profile-selected node probes and freeze both phase pools."""
    if probe_attempts < 1:
        raise ValueError("probe_attempts must be positive")
    if probe_retry_seconds < 0:
        raise ValueError("probe_retry_seconds must be non-negative")
    probes: dict[str, NodeProbe] = {}
    for item in inventory:
        argv = (
            remote_binary,
            remote_host_flag,
            item.host,
            sys.executable,
            "-m",
            "rsi_harness.cluster.bluevela.allocation",
            "probe",
            "--host",
            item.host,
            "--sif",
            str(sif_path),
            "--run-dir",
            str(run_dir),
            "--temp-root",
            str(temp_root),
        )
        completed: subprocess.CompletedProcess[str] | None = None
        for attempt in range(1, probe_attempts + 1):
            completed = runner(argv)
            if completed.returncode == 0:
                break
            if attempt < probe_attempts:
                sleeper(probe_retry_seconds)
        assert completed is not None
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise InfrastructureError(
                f"Blue Vela node probe failed on {item.host} after "
                f"{probe_attempts} attempts: {detail}"
            )
        try:
            probes[item.host] = NodeProbe.model_validate_json(completed.stdout)
        except ValueError as error:
            raise InfrastructureError(
                f"Blue Vela node probe returned invalid JSON on {item.host}"
            ) from error
    return freeze_pools(
        run_id=run_id,
        resources=resources,
        inventory=inventory,
        probes=probes,
        expected_sif_sha256=expected_sif_sha256,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_current_node(
    *,
    host: str,
    sif_path: Path,
    run_dir: Path,
    temp_root: Path,
) -> NodeProbe:
    """Collect facts on one blaunch-selected host without fallback values."""
    addresses = {
        item[4][0]
        for item in socket.getaddrinfo(host, None, socket.AF_INET)
    }
    if len(addresses) != 1:
        raise InfrastructureError(
            f"host {host} must resolve to one safe unique IPv4, got {addresses!r}"
        )
    raw_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = tuple(item.strip() for item in raw_devices.split(",") if item.strip())
    if not sif_path.is_file():
        raise InfrastructureError(f"host {host} cannot see SIF {sif_path}")
    return NodeProbe(
        ipv4=_safe_ipv4(next(iter(addresses)), host=host),
        cuda_devices=devices,
        sif_sha256=_sha256(sif_path),
        gpfs_visible=run_dir.is_dir(),
        infiniband_visible=(
            Path("/dev/infiniband").is_dir()
            and Path("/sys/class/infiniband").is_dir()
        ),
        tmp_free_mb=shutil.disk_usage(temp_root).free // (1024 * 1024),
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe")
    probe.add_argument("--host", required=True)
    probe.add_argument("--sif", type=Path, required=True)
    probe.add_argument("--run-dir", type=Path, required=True)
    probe.add_argument("--temp-root", type=Path, required=True)
    args = parser.parse_args()
    result = inspect_current_node(
        host=args.host,
        sif_path=args.sif,
        run_dir=args.run_dir,
        temp_root=args.temp_root,
    )
    print(result.model_dump_json())
    return 0


__all__ = [
    "AllocatedNode",
    "AllocatedPools",
    "InventoryHost",
    "NodeProbe",
    "freeze_pools",
    "inspect_current_node",
    "parse_lsb_mcpu_hosts",
    "probe_and_partition",
]


if __name__ == "__main__":
    raise SystemExit(_main())
