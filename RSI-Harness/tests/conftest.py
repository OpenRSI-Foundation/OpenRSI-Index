from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _host_firewall_by_default(monkeypatch):
    """Keep tests independent of the identity that runs them.

    A non-root Linux controller selects the container-namespace firewall;
    tests of that path enable it explicitly.
    """
    monkeypatch.setattr(
        "rsi_harness.runtime.production.container_netns_firewall_selected",
        lambda: False,
    )
