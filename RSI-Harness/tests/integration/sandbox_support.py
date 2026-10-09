"""Exact-ownership helpers for real managed-sandbox acceptance tests."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import docker
import pytest
from docker.errors import DockerException, NotFound

from rsi_harness.errors import SetupError
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    firewall_rule_chains,
)
from rsi_harness.runtime.recovery import LeaseStore, ResourceLease
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_contracts import (
    SandboxLimits,
    SandboxPhaseGrant,
    SandboxPolicy,
    SandboxProfile,
    SandboxTask,
)
from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend
from rsi_harness.runtime.sandbox_policy import resolve_sandbox_grant
from tests.fakes import FakeFirewallBackend

SANDBOX_IMAGE = "python:3.12-slim-bookworm"
ROOT_MODE = "RSI_SANDBOX_ROOT_MODE"


def root_mode() -> bool:
    """The operator root check (scripts/operator/sandbox_root_check.sh): as
    root with RSI_SANDBOX_ROOT_MODE=1, the D tests that support it use the
    real iptables firewall, and builders a loop-ext4 state filesystem,
    instead of the fakes a non-root user needs."""
    return os.environ.get(ROOT_MODE) == "1" and os.geteuid() == 0


class RootIptablesFirewall(DockerIptablesFirewallBackend):
    """The real firewall, recording like the fake (``events``, ``installed``)
    so a root-mode test knows every rule it made and can prove each one's
    jumps and ``RSI_F_``/``RSI_I_`` chains are gone."""

    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.events: list[tuple[str, object]] = []
        self.installed: dict[str, object] = {}

    def install(self, rule_id: str, rules: Any) -> None:
        self.events.append(("install", rule_id))
        super().install(rule_id, rules)
        self.installed[rule_id] = rules

    def remove(self, rule_id: str) -> None:
        self.events.append(("remove", rule_id))
        super().remove(rule_id)
        self.installed.pop(rule_id, None)


def sandbox_firewall(client: Any) -> Any:
    """The real firewall in root mode, else the fake (iptables needs root)."""
    if not root_mode():
        return FakeFirewallBackend()
    firewall = RootIptablesFirewall(client)
    if not firewall.probe():
        pytest.fail("root mode without DOCKER-USER/INPUT firewall authority")
    return firewall


def assert_no_rules(
    firewall: Any,
    run_id: str,
    *,
    rule_id: str | None = None,
    rule_ids: Iterable[str] = (),
) -> None:
    """No firewall rule of the run (or only of ``rule_id``) is left: the
    fake's whole record; or, on the host, no ``iptables -S`` line naming the
    run (the jumps' comments) and, for every rule this firewall installed
    and each of ``rule_ids`` (rules another process made, from its lease),
    neither jump nor chain (the chains carry no run id)."""
    if isinstance(firewall, FakeFirewallBackend):
        assert firewall.installed == {}
        return
    listing = subprocess.run(
        ["iptables", "--wait", "-S"], capture_output=True, text=True, check=True
    ).stdout
    needle = rule_id or f"rsi-{run_id}"
    assert [line for line in listing.splitlines() if needle in line] == []
    if rule_id is not None:
        rules = {rule_id}
    else:
        made = {
            rule
            for event, rule in getattr(firewall, "events", ())
            if event == "install"
        }
        rules = made | set(rule_ids)
    chains = {chain for rule in rules for chain in firewall_rule_chains(rule)}
    assert [line for line in listing.splitlines() if chains & set(line.split())] == []
    assert [rule for rule in sorted(rules) if firewall.exists(rule)] == []


def require_sandbox_authority() -> tuple[Any, Any]:
    """Return live Docker/image authority, or honor the strict test switch."""

    client = None
    try:
        client = docker.from_env(timeout=5)
        client.ping()
        image = client.images.get(SANDBOX_IMAGE)
        SandboxDockerBackend(client).preflight(sandbox_profile(image.id))
        return client, image
    except (DockerException, OSError, SetupError) as error:
        if client is not None:
            client.close()
        message = f"required Docker/cached sandbox image unavailable: {error}"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)


def sandbox_profile(image_id: str) -> SandboxProfile:
    return SandboxProfile(
        name="offline",
        image=image_id,
        cpus=1,
        memory_mb=128,
        pids=16,
        max_lifetime_sec=30.0,
        workdir="/workspace",
        tmpfs_mb=(("/workspace", 16), ("/tmp", 8), ("/dev/shm", 8)),
    )


def sandbox_grant(image_id: str):
    profile = sandbox_profile(image_id)
    limits = SandboxLimits(
        max_live=2,
        max_created=4,
        max_operations=32,
        max_cpus=2,
        max_memory_mb=256,
        max_lifetime_sec=120.0,
        max_upload_bytes=8 * 1024 * 1024,
        max_download_bytes=8 * 1024 * 1024,
        max_log_bytes=1024 * 1024,
    )
    phase = SandboxPhaseGrant(profiles=(profile.name,), limits=limits)
    task = SandboxTask(version=1, profiles=(profile,), work=phase)
    policy = SandboxPolicy(
        version=1,
        profiles=(profile,),
        work=phase,
        run_limits=limits,
        pool_cpus=8,
        pool_memory_mb=4096,
    )
    return resolve_sandbox_grant(
        task,
        policy,
        image_ids={profile.name: image_id},
        parent_cpus=1,
        parent_memory_mb=128,
    )


def create_broker(
    client: Any,
    store: LeaseStore,
    *,
    run_id: str,
    task_id: str,
    image_id: str,
    coordinator_pid: int | None = None,
) -> SandboxBroker:
    current = ResourceLease(
        run_id=run_id,
        task_id=task_id,
        coordinator_pid=coordinator_pid or os.getpid(),
        coordinator_started_at=1.0,
        phase="agent_running",
    )
    store.write(current)
    lock = threading.RLock()

    def mutate(transform):
        nonlocal current
        with lock:
            updated = transform(current)
            if updated is not current:
                store.write(updated)
                current = updated
            return current

    return SandboxBroker(
        sandbox_grant(image_id),
        SandboxDockerBackend(client),
        SandboxJournal(mutate),
    )


def owned_resources(client: Any, run_id: str) -> dict[str, tuple[str, ...]]:
    filters = {"label": f"rsi-harness.run-id={run_id}"}
    return {
        "containers": tuple(
            sorted(
                item.id for item in client.containers.list(all=True, filters=filters)
            )
        ),
        "images": tuple(
            sorted(item.id for item in client.images.list(filters=filters))
        ),
        "networks": tuple(
            sorted(item.id for item in client.networks.list(filters=filters))
        ),
        "volumes": tuple(
            sorted(item.name for item in client.volumes.list(filters=filters))
        ),
    }


def assert_no_sandbox_resources(
    client: Any, lease_store: LeaseStore, run_id: str
) -> None:
    assert owned_resources(client, run_id) == {
        "containers": (),
        "images": (),
        "networks": (),
        "volumes": (),
    }
    lease = lease_store.read(run_id)
    if lease is not None:
        assert all(child.state == "removed" for child in lease.sandboxes)
        assert all(not child.pending_mutation for child in lease.sandboxes)


def remove_exact_containers(client: Any, identities: set[str]) -> None:
    """Best-effort final cleanup restricted to IDs created by this fixture."""

    for identity in tuple(identities):
        try:
            child = client.containers.get(identity)
            child.reload()
            if child.attrs.get("State", {}).get("Paused"):
                child.unpause()
            child.remove(force=True, v=True)
        except NotFound:
            pass


class EmptySnapshotRecovery:
    def discover_snapshot_leases(self, **_kwargs):
        return ()

    def release_snapshot(self, _authority):
        raise AssertionError("no snapshot authority expected")


class EmptyFirewallRecovery:
    def remove(self, _rule_id):
        raise AssertionError("no firewall authority expected")

    def exists(self, _rule_id):
        return False


def lease_root(root: Path) -> LeaseStore:
    return LeaseStore(root / "leases")
