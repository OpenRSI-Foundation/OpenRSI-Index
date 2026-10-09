"""Opt-in: tests marked ``acceptance`` run only with RSI_ACCEPTANCE=1.

They need root, the real iptables firewall, public egress and minutes to
hours; scripts/operator/sandbox_root_check.sh and sandbox_acceptance.sh set
the switch and run them as root. Unmarked tests here (the sample task, the
audits, the scripts' dry runs) run in the ordinary suite.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RSI_ACCEPTANCE") == "1":
        return
    skip = pytest.mark.skip(reason="acceptance checks are opt-in (RSI_ACCEPTANCE=1)")
    for item in items:
        if item.get_closest_marker("acceptance") is not None:
            item.add_marker(skip)


@pytest.fixture
def as_root():
    """Root authority for the real firewall, loop devices and cgroup.kill;
    under the operator scripts its absence is a failure, not a skip."""
    if os.geteuid() != 0:
        message = "this check needs root (run the operator script with sudo)"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
