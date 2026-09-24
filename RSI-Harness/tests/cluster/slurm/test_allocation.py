from __future__ import annotations

import subprocess

import pytest

from rsi_harness.cluster.slurm.allocation import discover_slurm_inventory
from rsi_harness.errors import InfrastructureError


def discover(*, hosts="gpu03\ngpu01\ngpu02\n", cpus="8(x2),16", **updates):
    calls = []

    def runner(argv):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, hosts, "")

    result = discover_slurm_inventory(
        expected_hosts=3,
        expected_slots=8,
        control_binary="/site/scontrol",
        environ={"SLURM_JOB_NODELIST": "gpu[01-03]", "SLURM_JOB_CPUS_PER_NODE": cpus},
        runner=runner,
        **updates,
    )
    assert calls == [("/site/scontrol", "show", "hostnames", "gpu[01-03]")]
    return result


def test_discovery_preserves_scheduler_order_and_actual_cpu_counts():
    nodes = discover()
    assert tuple(node.host for node in nodes) == ("gpu03", "gpu01", "gpu02")
    assert tuple(node.slots for node in nodes) == (8, 8, 16)


@pytest.mark.parametrize(
    "cpus",
    [
        "",
        "8(x0)",
        "8(x4)",
        "8(x2)",
        "4(x3)",
        "8(x-3)",
        "invalid",
        "8(x999999999)",
    ],
)
def test_rejects_missing_malformed_or_insufficient_cpu_inventory(cpus):
    with pytest.raises(InfrastructureError):
        discover(cpus=cpus)


@pytest.mark.parametrize(
    "hosts", ["", "gpu01", "gpu01\ngpu01\ngpu02", "-bad\ngpu02\ngpu03"]
)
def test_rejects_invalid_host_inventory(hosts):
    with pytest.raises(InfrastructureError):
        discover(hosts=hosts)


def test_missing_job_environment_and_control_failure():
    with pytest.raises(InfrastructureError, match="required"):
        discover_slurm_inventory(expected_hosts=3, expected_slots=8, environ={})
    with pytest.raises(InfrastructureError, match="cannot expand"):
        discover_slurm_inventory(
            expected_hosts=3,
            expected_slots=8,
            environ={
                "SLURM_JOB_NODELIST": "gpu[1-3]",
                "SLURM_JOB_CPUS_PER_NODE": "8(x3)",
            },
            runner=lambda argv: subprocess.CompletedProcess(argv, 1, "", "unavailable"),
        )
