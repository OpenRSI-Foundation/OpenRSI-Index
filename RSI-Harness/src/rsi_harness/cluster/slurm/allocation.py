"""Discover the ordered Slurm allocation without inventing missing resources."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping

from rsi_harness.cluster.bluevela.allocation import InventoryHost
from rsi_harness.errors import InfrastructureError


def _run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def discover_slurm_inventory(
    *,
    expected_hosts: int,
    expected_slots: int,
    control_binary: str = "scontrol",
    environ: Mapping[str, str] | None = None,
    runner: Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]] = _run,
) -> tuple[InventoryHost, ...]:
    environment = os.environ if environ is None else environ
    nodelist = environment.get("SLURM_JOB_NODELIST", "")
    raw_slots = environment.get("SLURM_JOB_CPUS_PER_NODE", "")
    if not nodelist or not raw_slots:
        raise InfrastructureError(
            "SLURM_JOB_NODELIST and SLURM_JOB_CPUS_PER_NODE are required"
        )
    completed = runner((control_binary, "show", "hostnames", nodelist))
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise InfrastructureError(f"cannot expand Slurm allocation hosts: {detail}")
    hosts = completed.stdout.split()
    if len(hosts) != expected_hosts or len(set(hosts)) != len(hosts):
        raise InfrastructureError(
            f"Slurm inventory must contain exactly {expected_hosts} unique hosts"
        )
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", host) is None for host in hosts
    ):
        raise InfrastructureError("Slurm inventory contains an unsafe host")
    slots: list[int] = []
    for term in raw_slots.split(","):
        match = re.fullmatch(r"([0-9]+)(?:\(x([0-9]+)\))?", term.strip())
        if match is None:
            raise InfrastructureError("invalid SLURM_JOB_CPUS_PER_NODE")
        count, repeats = int(match[1]), int(match[2] or 1)
        # Exclusive Slurm jobs may allocate more CPUs than were requested.
        if (
            count < expected_slots
            or repeats < 1
            or len(slots) + repeats > expected_hosts
        ):
            raise InfrastructureError(
                "Slurm CPU allocation does not match the node plan"
            )
        slots.extend([count] * repeats)
    if len(slots) != len(hosts):
        raise InfrastructureError("Slurm host and CPU inventories differ in length")
    return tuple(
        InventoryHost(host=host, slots=count)
        for host, count in zip(hosts, slots, strict=True)
    )
