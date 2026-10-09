"""Env recovery: a crash at any journal or Engine step converges to nothing.

The env is made by the real SandboxEnvDockerBackend over the fake Engine of
test_sandbox_env_docker, and the process "dies" (a BaseException nobody
catches) just before any one event: a journal commit, a rule install, a
bridge, volume or container create, a pause or a removal. Recovery then
runs on the last durable journal through the production env methods of
ProductionRecoveryBackend. Real Docker and kill -9 are in
tests/integration/test_sandbox_env_recovery.py.
"""

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from docker.errors import NotFound

from rsi_harness.models import RunStatus
from rsi_harness.runtime import sandbox_env_contracts as env
from rsi_harness.runtime.network import firewall_rule_chains
from rsi_harness.runtime.production import (
    ProductionRecoveryBackend,
    coordinator_may_run,
    firewall_commands_running,
)
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager, ResourceLease
from rsi_harness.runtime.sandbox_contracts import SandboxReservation
from rsi_harness.runtime.sandbox_env_docker import DockerPausedKiller
from rsi_harness.runtime.sandbox_lifecycle import ENDPOINT_MODULES
from tests.runtime.test_sandbox_budget import make_child
from tests.runtime.test_sandbox_env_docker import (
    ENV_ID,
    MAIN_HANDLE,
    EnvWorld,
    spec,
)
from tests.runtime.test_sandbox_policy_v2 import owner as make_owner
from tests.runtime.test_sandbox_policy_v2 import (
    pulled_image_lease,
)
from tests.sandbox_helpers import make_env_spec

SIBLING_ID = "e" + "2" * 32
RESERVATION = SandboxReservation(
    cpus=4,
    memory_mb=4096,
    pool_cpus=8,
    pool_memory_mb=8192,
    disk_mb=2048,
    pool_disk_mb=40960,
)


class Crash(BaseException):
    """The broker process is killed: no handler of it runs any more."""


class CrashingEvents(list):
    """The shared event log; the ``at``-th append dies before its step."""

    def __init__(self, items, at):
        super().__init__(items)
        self.at = at

    def append(self, item):
        if self.at is not None and len(self) >= self.at:
            raise Crash(item)
        super().append(item)


def arm(world, after):
    """Die just before the ``after``-th event from now; returns the log."""
    events = CrashingEvents(world.events, len(world.events) + after)
    world.events = world.network.events = events
    world.network.firewall.events = world.journal.events = events
    return events


def three_services():
    raw = make_env_spec()
    raw["services"]["web"] = {
        "image": MAIN_HANDLE,
        "command": ["sleep", "infinity"],
        "cpus": 0.5,
        "memory_mb": 256,
        "pids": 256,
    }
    return spec(raw)


def create(world):
    """The plan is journaled before any Docker call (M5 plan_env)."""
    plan = world.plan(three_services())
    lease = world.planned(plan)
    return lambda: world.backend.create(plan, lease, world.journal)


def pause(world):
    _, lease = world.ready(three_services())
    return lambda: world.backend.pause(lease, world.journal)


def destroy(world):
    _, lease = world.ready(three_services())
    return lambda: world.backend.destroy(lease, world.journal)


SCENARIOS = {"create": create, "pause": pause, "destroy": destroy}


def _kinds(name):
    """The event kinds of one uninterrupted scenario step."""
    world = EnvWorld()
    step = SCENARIOS[name](world)
    before = len(world.events)
    step()
    return [event[0] for event in world.events[before:]]


def _events(name):
    return len(_kinds(name))


# Every event of each scenario, plus the finished step (e.g. "whole group
# paused" or "created").
CRASH_POINTS = [
    (name, after) for name in SCENARIOS for after in range(_events(name) + 1)
]


class EnvRecoveryBackend:
    """Recovery's Docker port over the fake Engine.

    The env mutations (kill, remove, bridge and rule) are production's own
    ProductionRecoveryBackend methods on the same fake client; only reads,
    volume removal and parent inventory are modelled here.
    """

    def __init__(self, world, *, alive=False, settles=True):
        self.world = world
        self.engine = world.engine
        self.alive = alive
        self.settles = settles
        self.liveness = []
        self.settlements = []
        # Runs while "settling": a create the dead broker sent may land.
        self.while_settling = lambda: None
        self.production = ProductionRecoveryBackend(
            world.network.client,
            None,
            world.network.firewall,
            paused_killer=DockerPausedKiller(world.network.client.api),
        )

    @staticmethod
    def _matches(labels, required):
        return all((labels or {}).get(key) == value for key, value in required.items())

    @staticmethod
    def _container(attrs):
        return {
            "id": attrs["Id"],
            "name": attrs["Name"][1:],
            "image_id": attrs["Image"],
            "labels": dict(attrs["Config"]["Labels"]),
            "running": attrs["State"]["Running"],
            "paused": attrs["State"]["Paused"],
        }

    def list_containers(self, *, labels):
        return tuple(
            (attrs["Id"], self._container(attrs))
            for attrs in self.engine.containers.values()
            if self._matches(attrs["Config"]["Labels"], labels)
        )

    def inspect_container(self, reference):
        try:
            return self._container(self.engine._get(reference))
        except NotFound:
            return None

    @staticmethod
    def _volume(attrs):
        return {
            "name": attrs["Name"],
            "driver": attrs["Driver"],
            "labels": dict(attrs["Labels"]),
            "options": dict(attrs["Options"] or {}),
            "scope": attrs["Scope"],
        }

    def list_volumes(self, *, labels):
        return tuple(
            (name, self._volume(attrs))
            for name, attrs in self.engine.volumes.items()
            if self._matches(attrs["Labels"], labels)
        )

    def inspect_volume(self, name):
        attrs = self.engine.volumes.get(name)
        return None if attrs is None else self._volume(attrs)

    def volume_in_use(self, name):
        return any(
            mount.get("Name") == name
            for attrs in self.engine.containers.values()
            for mount in attrs["Mounts"]
        )

    def remove_volume(self, name):
        self.engine.remove_volume(name)

    def list_networks(self, *, labels):
        return tuple(
            (network.id, {**network.attrs, "labels": dict(network.attrs["Labels"])})
            for network in self.world.network.store.values()
            if self._matches(network.attrs["Labels"], labels)
        )

    def network_in_use(self, network_id):
        network = self.world.network.store.get(network_id)
        return bool(network is not None and network.attrs["Containers"])

    def remove_network(self, network_id):
        self.world.network.store.pop(network_id, None)

    def policy_exists(self, rule_id):
        return rule_id in self.world.network.firewall.installed

    def remove_policy(self, rule_id):
        self.world.network.firewall.remove(rule_id)

    def kill_sandbox_container(self, container_id):
        self.production.kill_sandbox_container(container_id)

    def remove_sandbox_container(self, container_id):
        self.production.remove_sandbox_container(container_id)

    def recover_sandbox_network(self, lease, *, settled=False):
        return self.production.recover_sandbox_network(lease, settled=settled)

    def coordinator_alive(self, pid, started_at):
        self.liveness.append((pid, started_at))
        return self.alive

    def settle_sandbox_creates(self, rule_ids):
        self.settlements.append(rule_ids)
        self.while_settling()
        return self.settles

    def remove_sandbox_spool(self, path):
        self.production.remove_sandbox_spool(path)

    @staticmethod
    def _image(attrs):
        return {
            "id": attrs["Id"],
            "labels": dict((attrs.get("Config") or {}).get("Labels") or {}),
            "repo_tags": tuple(attrs.get("RepoTags") or ()),
        }

    def list_images(self, *, labels):
        found = {}
        for attrs in self.engine.images.values():
            if self._matches((attrs.get("Config") or {}).get("Labels"), labels):
                found[attrs["Id"]] = self._image(attrs)
        return tuple(found.items())

    def list_tagged_images(self, prefix):
        found = {}
        for attrs in self.engine.images.values():
            if any(tag.startswith(prefix) for tag in attrs.get("RepoTags") or ()):
                found[attrs["Id"]] = self._image(attrs)
        return tuple(found.items())


def work_sibling(world, *, run_id="run-1", ready=False):
    """A second, Work-phase env of the same world."""
    plan = world.backend.plan(
        make_owner("work", run_id=run_id),
        SIBLING_ID,
        three_services(),
        world.images(),
        default_pids=512,
        swap_ratio=1.0,
    )
    lease = world.backend.create(
        plan,
        world.journal(plan.lease(created_at=100.0, expires_at=700.0)),
        world.journal,
    )
    if ready:
        kv = plan.services[0].container_name
        world.at(1.0, lambda: world.engine.health(kv, "healthy"))
        result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
        assert result.state == "ready", result
        lease = result.lease
    return plan, lease


def resource_lease(*envs, **updates):
    values = dict(
        run_id="run-1",
        task_id="task",
        coordinator_pid=4321,
        coordinator_started_at=50.0,
        phase="agent_running",
        sandbox_envs=envs,
        sandbox_reservation=RESERVATION,
    )
    values.update(updates)
    return ResourceLease(**values)


def recovery(tmp_path, world, lease, **backend_options):
    store = LeaseStore(tmp_path / "leases")
    store.write(lease)
    backend = EnvRecoveryBackend(world, **backend_options)
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "managed"
    )
    return manager, store, backend


def stage_file(tmp_path):
    stage = (
        env.sandbox_spool_root(tmp_path / "managed", "run-1")
        / "stage"
        / "judge-agent-1-0a1b2c3d"
    )
    stage.mkdir(parents=True)
    (stage / f"s{'5' * 32}.tar").write_bytes(b"staged")
    return stage


def endpoint(tmp_path, name="0a1b2c3d", *, socket=True, modules=ENDPOINT_MODULES):
    """A phase endpoint as SandboxLifecycle leaves it when the run dies:
    ``<8 hex>/{s, rsi-sandbox, py/<modules>}``."""
    directory = tmp_path / "managed" / "run-1" / "sb" / name
    (directory / "py").mkdir(parents=True)
    if socket:
        os.mknod(directory / "s", stat.S_IFSOCK | 0o600)
    (directory / "rsi-sandbox").write_text("#!/usr/bin/env python3\n")
    for module in modules:
        (directory / "py" / module).write_text("")
    return directory


def assert_nothing_left(world):
    assert world.engine.containers == {}
    assert world.engine.volumes == {}
    assert world.network.store == {}
    assert world.network.firewall.installed == {}


def assert_retained(store):
    retained = store.read("run-1")
    assert retained.recovery_required
    assert retained.sandbox_reservation is not None
    return retained


@pytest.mark.parametrize(("scenario", "after"), CRASH_POINTS)
def test_a_crash_at_any_step_converges_to_all_removed(tmp_path, scenario, after):
    world = EnvWorld()
    step = SCENARIOS[scenario](world)
    events = arm(world, after)
    try:
        step()
    except Crash:
        pass
    events.at = None
    durable = world.journal.leases[-1]
    pulling = pulled_image_lease(state="planned", pre_existing=False)
    stage = stage_file(tmp_path)
    endpoint(tmp_path, "0a1b2c3d")
    endpoint(tmp_path, "4e5f6a7b")
    manager, store, backend = recovery(
        tmp_path, world, resource_lease(durable, sandbox_images=(pulling,))
    )

    assert manager.recover("run-1") == ("run-1",)

    assert_nothing_left(world)
    assert not (tmp_path / "managed" / "run-1" / "sb").exists()
    recovered = store.read("run-1")
    assert recovered.sandbox_envs == ()
    assert recovered.sandbox_images == ()
    assert not recovered.recovery_required
    assert recovered.sandbox_reservation is None
    assert recovered.status is RunStatus.CANCELLED
    assert not stage.exists()
    # Only a pending create whose one in-flight object is missing waits for
    # its coordinator to be gone and the create to settle.
    assert len(backend.settlements) == len(backend.liveness) <= 1
    if backend.liveness:
        assert durable.pending_mutation


def test_a_whole_paused_group_is_killed_before_anything_is_removed(tmp_path):
    world = EnvWorld()
    _, lease = world.ready(three_services())
    lease = world.backend.pause(lease, world.journal)
    assert lease.state == "paused"
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))

    manager.recover("run-1")

    assert_nothing_left(world)
    kills = [i for i, event in enumerate(world.events) if event[0] == "kill"]
    removals = [
        i
        for i, event in enumerate(world.events)
        if event[0] in ("container-remove", "volume-remove", "network-remove")
    ]
    assert len(kills) == 3
    assert all(world.events[i][2] is True for i in kills)  # killed while paused
    assert max(kills) < min(removals)
    assert store.read("run-1").sandbox_envs == ()


def test_teardown_order_is_containers_volumes_bridge_then_rule(tmp_path):
    world = EnvWorld()
    plan, lease = world.create(three_services())
    manager, _, _ = recovery(tmp_path, world, resource_lease(lease))
    del world.events[:]

    manager.recover("run-1")

    removals = ("container-remove", "volume-remove", "network-remove", "remove")
    order = [event[0] for event in world.events if event[0] in removals]
    assert order == ["container-remove"] * 3 + ["volume-remove"] * len(plan.volumes) + [
        "network-remove",
        "remove",
    ]
    assert ("remove", plan.network.rule_id) in world.events
    assert_nothing_left(world)


def crashed_create(world, event, *, offset=0):
    """A create killed just before (offset 0) or after (1) its first ``event``."""
    plan = world.plan(three_services())
    lease = world.planned(plan)
    events = arm(world, _kinds("create").index(event) + offset)
    with pytest.raises(Crash):
        world.backend.create(plan, lease, world.journal)
    events.at = None
    durable = world.journal.leases[-1]
    assert durable.pending_mutation
    assert all(record.container_id is None for record in durable.services)
    return plan, durable


def test_a_pending_create_whose_object_landed_needs_no_settlement(tmp_path):
    world = EnvWorld()
    # Died right after the first container create, before its journal commit.
    _, durable = crashed_create(world, "container-create", offset=1)
    manager, store, backend = recovery(
        tmp_path, world, resource_lease(durable), alive=True
    )

    manager.recover("run-1")

    # Its container proves every earlier call returned and no later one ran.
    assert (backend.liveness, backend.settlements) == ([], [])
    assert_nothing_left(world)
    assert store.read("run-1").sandbox_envs == ()


def test_a_pending_create_with_a_live_coordinator_fails_closed(tmp_path):
    world = EnvWorld()
    _, durable = crashed_create(world, "container-create")
    manager, store, backend = recovery(
        tmp_path, world, resource_lease(durable), alive=True
    )

    with pytest.raises(RuntimeError, match="coordinator may still run"):
        manager.recover("run-1")

    assert backend.liveness == [(4321, 50.0)]
    assert backend.settlements == []
    retained = assert_retained(store)
    (record,) = retained.sandbox_envs
    assert record.pending_mutation
    assert world.engine.volumes and world.network.store
    assert world.network.firewall.installed


def test_an_unsettled_firewall_command_keeps_the_pending_env(tmp_path):
    world = EnvWorld()
    plan, durable = crashed_create(world, "install")
    assert durable.network_id is None
    manager, store, backend = recovery(
        tmp_path, world, resource_lease(durable), settles=False
    )

    with pytest.raises(RuntimeError, match="has not settled"):
        manager.recover("run-1")

    # Only the rule being installed can still land; its env stays pending.
    assert backend.settlements == [(plan.network.rule_id,)]
    (record,) = assert_retained(store).sandbox_envs
    assert (record.state, record.pending_mutation) == ("planned", True)


def test_a_create_landing_while_settling_is_found_and_removed(tmp_path):
    world = EnvWorld()
    plan, durable = crashed_create(world, "container-create")
    manager, store, backend = recovery(tmp_path, world, resource_lease(durable))
    landed = []
    backend.while_settling = lambda: landed.append(
        world.backend._create_container(plan.services[0])
    )

    manager.recover("run-1")

    # The volume create, not the container's, may have been last.
    assert backend.settlements == [()]
    assert landed and landed[0] not in world.engine.containers
    assert ("container-remove", plan.services[0].container_name) in world.events
    assert_nothing_left(world)
    assert store.read("run-1").sandbox_envs == ()


def test_a_settled_bridge_create_that_never_landed_releases_its_rule(tmp_path):
    world = EnvWorld()
    plan, durable = crashed_create(world, "network-create")
    assert plan.network.rule_id in world.network.firewall.installed
    manager, store, backend = recovery(tmp_path, world, resource_lease(durable))

    manager.recover("run-1")

    assert backend.settlements == [(plan.network.rule_id,)]
    assert_nothing_left(world)
    assert not store.read("run-1").recovery_required


def test_a_failed_removal_keeps_the_pending_mark(tmp_path):
    world = EnvWorld()
    _, durable = crashed_create(world, "container-create", offset=1)
    manager, store, backend = recovery(tmp_path, world, resource_lease(durable))
    backend.remove_sandbox_container = lambda container_id: None

    with pytest.raises(RuntimeError, match="removal cannot be proven"):
        manager.recover("run-1")

    # A later recovery still knows a create was pending (M2's rule, S7).
    (record,) = assert_retained(store).sandbox_envs
    assert (record.state, record.pending_mutation) == ("stopping", True)
    assert record.services[0].container_id in world.engine.containers


def _relabel(kv):
    kv["Config"]["Labels"]["rsi-harness.task-id"] = "someone-else"


def _reimage(kv):
    # The planned name and exact labels, but another image.
    kv["Image"] = "sha256:" + "9" * 64


def _rename(kv):
    # The journaled container ID, now under another name.
    kv["Name"] = "/renamed"


@pytest.mark.parametrize("mutate", [_relabel, _reimage, _rename])
def test_a_foreign_object_holding_a_planned_name_is_never_removed(tmp_path, mutate):
    world = EnvWorld()
    plan, lease = world.create(three_services())
    kv = world.container(plan, "kv")
    mutate(kv)
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))
    del world.events[:]

    with pytest.raises(RuntimeError, match="kv is not owned by its planned identity"):
        manager.recover("run-1")

    assert kv["Id"] in world.engine.containers
    assert not any(event[0].endswith("-remove") for event in world.events)
    assert not any(event == ("kill", kv["Name"][1:], False) for event in world.events)
    retained = assert_retained(store)
    assert [record.env_id for record in retained.sandbox_envs] == [ENV_ID]


def test_one_foreign_service_never_leaves_its_owned_siblings_running(tmp_path):
    world = EnvWorld()
    plan, lease = world.ready(three_services())
    kv = world.container(plan, "kv")
    kv["Config"]["Labels"]["rsi-harness.task-id"] = "someone-else"
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))

    with pytest.raises(RuntimeError, match="kv is not owned"):
        manager.recover("run-1")

    running = {
        attrs["Name"][1:]
        for attrs in world.engine.containers.values()
        if attrs["State"]["Running"]
    }
    # The foreign one is kept as found; every provably owned one is killed.
    assert running == {kv["Name"][1:]}
    assert len(world.engine.containers) == 3
    (record,) = assert_retained(store).sandbox_envs
    assert record.state == "ready"


def test_duplicate_identities_for_one_service_fail_closed(tmp_path):
    world = EnvWorld()
    plan, lease = world.create(three_services())
    # The journal names another container than the one holding the name.
    services = list(lease.services)
    services[0] = services[0].model_copy(update={"container_id": "f" * 64})
    world.engine.containers["f" * 64] = {
        **world.container(plan, "kv"),
        "Id": "f" * 64,
        "Name": "/elsewhere",
    }
    lease = lease.model_copy(update={"services": tuple(services)})
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))

    with pytest.raises(RuntimeError, match="ambiguous planned/actual identity"):
        manager.recover("run-1")

    assert_retained(store)
    assert len(world.engine.containers) == 4


def test_unprovable_container_removal_keeps_the_env_and_reservation(tmp_path):
    world = EnvWorld()
    _, lease = world.ready(three_services())
    manager, store, backend = recovery(tmp_path, world, resource_lease(lease))
    backend.remove_sandbox_container = lambda container_id: None

    with pytest.raises(RuntimeError, match="removal cannot be proven"):
        manager.recover("run-1")

    retained = assert_retained(store)
    (record,) = retained.sandbox_envs
    assert record.state == "stopping"
    # Contained first: nothing of it runs any more.
    assert not any(
        attrs["State"]["Running"] for attrs in world.engine.containers.values()
    )
    assert world.network.firewall.installed


def test_a_volume_with_driver_options_is_not_ours_and_fails_closed(tmp_path):
    world = EnvWorld()
    plan, lease = world.create(three_services())
    name = plan.volumes[0].name
    world.engine.volumes[name]["Options"] = {"type": "none", "o": "bind"}
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))

    with pytest.raises(RuntimeError, match=f"volume {name} is not owned"):
        manager.recover("run-1")

    assert name in world.engine.volumes
    assert_retained(store)


def test_one_failed_env_never_skips_the_removal_of_another(tmp_path):
    world = EnvWorld()
    plan, first = world.create(three_services())
    sibling_plan, sibling = work_sibling(world)
    world.container(plan, "web")["Config"]["Labels"]["rsi-harness.role"] = "forged"
    manager, store, _ = recovery(tmp_path, world, resource_lease(first, sibling))

    with pytest.raises(RuntimeError, match="web is not owned"):
        manager.recover("run-1")

    retained = assert_retained(store)
    assert [record.env_id for record in retained.sandbox_envs] == [ENV_ID]
    assert not any(
        attrs["Config"]["Labels"].get("rsi-harness.sandbox-env") == SIBLING_ID
        for attrs in world.engine.containers.values()
    )
    assert sibling_plan.network.rule_id not in world.network.firewall.installed


def test_one_failed_env_removal_never_skips_the_removal_of_another(tmp_path):
    world = EnvWorld()
    _, first = world.create(three_services())
    sibling_plan, sibling = work_sibling(world)
    manager, store, backend = recovery(tmp_path, world, resource_lease(first, sibling))
    remove = backend.remove_sandbox_container
    held = {record.container_id for record in first.services}
    backend.remove_sandbox_container = lambda identity: (
        None if identity in held else remove(identity)
    )

    with pytest.raises(RuntimeError, match="removal cannot be proven"):
        manager.recover("run-1")

    (record,) = assert_retained(store).sandbox_envs
    assert (record.env_id, record.state) == (ENV_ID, "stopping")
    assert not any(world.labelled(SIBLING_ID))
    assert sibling_plan.network.rule_id not in world.network.firewall.installed


def test_judge_envs_are_killed_before_work_envs(tmp_path):
    world = EnvWorld()
    _, judge = world.ready(three_services())
    _, work = work_sibling(world, ready=True)
    manager, _, _ = recovery(tmp_path, world, resource_lease(work, judge))
    del world.events[:]

    manager.recover("run-1")

    kills = [event[1] for event in world.events if event[0] == "kill"]
    assert len(kills) == 6
    judge_services = {env.env_container_name(ENV_ID, idx) for idx in range(3)}
    assert set(kills[:3]) == judge_services
    assert_nothing_left(world)


def test_unprovable_termination_removes_nothing(tmp_path):
    world = EnvWorld()
    _, lease = world.ready(three_services())
    manager, store, backend = recovery(tmp_path, world, resource_lease(lease))
    backend.kill_sandbox_container = lambda container_id: None
    del world.events[:]

    with pytest.raises(RuntimeError, match="termination cannot be proven"):
        manager.recover("run-1")

    assert not any(event[0].endswith("-remove") for event in world.events)
    assert len(world.engine.containers) == 3
    (record,) = assert_retained(store).sandbox_envs
    assert record.state == "ready"


def test_a_volume_mounted_by_another_container_is_retained(tmp_path):
    world = EnvWorld()
    plan, lease = world.create(three_services())
    name = plan.volumes[0].name
    world.engine.containers["f" * 64] = {
        "Id": "f" * 64,
        "Name": "/unrelated",
        "Image": "sha256:" + "9" * 64,
        "Config": {"Labels": {}},
        "State": {"Status": "running", "Running": True, "Paused": False},
        "Mounts": [{"Type": "volume", "Name": name}],
    }
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))

    with pytest.raises(RuntimeError, match=f"volume {name} remains in use"):
        manager.recover("run-1")

    assert name in world.engine.volumes
    assert "f" * 64 in world.engine.containers
    (record,) = assert_retained(store).sandbox_envs
    assert record.state == "stopping" and record.volumes[0].created


def test_unprovable_volume_removal_keeps_the_env_and_reservation(tmp_path):
    world = EnvWorld()
    plan, lease = world.create(three_services())
    name = plan.volumes[0].name
    manager, store, backend = recovery(tmp_path, world, resource_lease(lease))
    backend.remove_volume = lambda volume: None

    with pytest.raises(RuntimeError, match=f"volume {name} removal is unproven"):
        manager.recover("run-1")

    assert name in world.engine.volumes
    (record,) = assert_retained(store).sandbox_envs
    assert record.state == "stopping" and record.volumes[0].created
    assert world.network.firewall.installed


def _spy_parent_containment(manager, world):
    """Records the env containers left when parents are contained."""
    seen = []
    contain = manager._contain_parents_after_sandbox_failure

    def spy(lease):
        seen.append(sorted(attrs["Name"] for attrs in world.engine.containers.values()))
        return contain(lease)

    manager._contain_parents_after_sandbox_failure = spy
    return seen


def test_a_failed_child_still_lets_every_env_converge_before_parents(tmp_path):
    world = EnvWorld()
    _, lease = world.ready(three_services())
    # A pending v1 create that found nothing fails closed (recovery.py rule).
    child = make_child()
    manager, store, _ = recovery(
        tmp_path, world, resource_lease(lease, sandboxes=(child,))
    )
    contained = _spy_parent_containment(manager, world)

    with pytest.raises(RuntimeError, match="sandbox pending mutation outcome"):
        manager.recover("run-1")

    # Every env was killed and removed before any parent was contained (S8).
    assert contained == [[]]
    assert_nothing_left(world)
    retained = assert_retained(store)
    assert retained.sandbox_envs == ()
    assert [item.pending_mutation for item in retained.sandboxes] == [True]


def test_a_failed_env_still_lets_every_child_converge_before_parents(tmp_path):
    world = EnvWorld()
    plan, lease = world.create(three_services())
    kv = world.container(plan, "kv")
    kv["Config"]["Labels"]["rsi-harness.task-id"] = "someone-else"
    child = make_child(pending_mutation=False)
    manager, store, _ = recovery(
        tmp_path, world, resource_lease(lease, sandboxes=(child,))
    )
    contained = _spy_parent_containment(manager, world)

    with pytest.raises(RuntimeError, match="kv is not owned"):
        manager.recover("run-1")

    assert len(contained) == 1
    retained = assert_retained(store)
    assert [item.state for item in retained.sandboxes] == ["removed"]


def test_exact_labeled_orphans_of_the_run_are_removed(tmp_path, caplog):
    world = EnvWorld()
    plan, _ = world.create(three_services())
    # The journal proved this env removed, then a late create finished.
    manager, store, _ = recovery(tmp_path, world, resource_lease())

    manager.recover("run-1")

    assert_nothing_left(world)
    assert "removing unjournaled sandbox env container" in caplog.text
    assert not store.read("run-1").recovery_required


def test_every_later_recovery_still_sweeps_the_run(tmp_path):
    world = EnvWorld()
    _, lease = world.create(three_services())
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))
    manager.recover("run-1")
    recovered = store.read("run-1")
    assert recovered.sandbox_env_authority
    assert (recovered.sandbox_envs, recovered.sandbox_reservation) == ((), None)

    # A create the daemon finished only after its env left the journal.
    world.create(three_services())
    manager.recover("run-1")

    assert_nothing_left(world)
    assert not store.read("run-1").recovery_required


def test_another_runs_env_objects_are_left_alone(tmp_path):
    world = EnvWorld()
    _, lease = world.create(three_services())
    other, _ = work_sibling(world, run_id="run-2")
    manager, store, _ = recovery(tmp_path, world, resource_lease(lease))

    manager.recover("run-1")

    assert not store.read("run-1").recovery_required
    assert not any(world.labelled(ENV_ID))
    containers, volumes, networks = world.labelled(SIBLING_ID)
    assert (len(containers), len(volumes), len(networks)) == (
        3,
        len(other.volumes),
        1,
    )
    assert other.network.rule_id in world.network.firewall.installed


@pytest.mark.parametrize(
    ("kind", "mutate"),
    [
        (
            "container",
            lambda world, plan: world.container(plan, "kv")["Config"]["Labels"].update(
                {"rsi-harness.task-id": "other-task"}
            ),
        ),
        (
            "volume",
            lambda world, plan: world.engine.volumes[plan.volumes[0].name][
                "Labels"
            ].pop("rsi-harness.sandbox-env"),
        ),
        (
            "network",
            lambda world, plan: next(
                iter(world.network.store.values())
            ).attrs.__setitem__("Name", "rsi-sbnet-elsewhere"),
        ),
    ],
)
def test_an_orphan_matching_only_part_of_its_labels_fails_closed(
    tmp_path, kind, mutate
):
    world = EnvWorld()
    plan, _ = world.create(three_services())
    mutate(world, plan)
    manager, store, _ = recovery(tmp_path, world, resource_lease())

    with pytest.raises(RuntimeError):
        manager.recover("run-1")

    assert_retained(store)
    assert world.network.firewall.installed  # the rule outlives no bridge


def test_no_orphan_scan_without_environment_authority(tmp_path):
    world = EnvWorld()
    world.create(three_services())
    manager, store, _ = recovery(
        tmp_path,
        world,
        resource_lease(
            sandbox_reservation=SandboxReservation(
                cpus=4, memory_mb=4096, pool_cpus=8, pool_memory_mb=8192
            )
        ),
    )

    endpoint(tmp_path)

    manager.recover("run-1")

    # A profile-only run never had env authority: its recovery is v1's,
    # which still removes the phase endpoints a crash left.
    assert len(world.engine.containers) == 3
    assert not (tmp_path / "managed" / "run-1" / "sb").exists()
    recovered = store.read("run-1")
    assert not recovered.sandbox_env_authority
    assert not recovered.recovery_required


def test_a_profile_run_fails_closed_on_anything_but_its_endpoints(tmp_path):
    # v1 never makes a stage spool: under a profile run's sb it is foreign.
    world = EnvWorld()
    reservation = SandboxReservation(
        cpus=4, memory_mb=4096, pool_cpus=8, pool_memory_mb=8192
    )
    manager, store, _ = recovery(
        tmp_path, world, resource_lease(sandbox_reservation=reservation)
    )
    stage = stage_file(tmp_path)
    work = endpoint(tmp_path)

    with pytest.raises(RuntimeError, match="foreign entry spool"):
        manager.recover("run-1")

    assert stage.exists() and (work / "s").exists()
    assert store.read("run-1").recovery_required


def test_pulled_images_are_released_and_never_removed(tmp_path):
    world = EnvWorld()
    images = (
        pulled_image_lease(image_id="sha256:" + "a" * 64, state="present"),
        pulled_image_lease(
            "i" + "6" * 32,
            image_id="sha256:" + "b" * 64,
            state="present",
            pre_existing=False,
        ),
    )
    manager, store, backend = recovery(
        tmp_path, world, resource_lease(sandbox_images=images)
    )
    assert not hasattr(backend, "remove_image")

    manager.recover("run-1")

    recovered = store.read("run-1")
    assert recovered.sandbox_images == ()
    assert recovered.sandbox_reservation is None
    assert set(world.engine.images) == {"sha256:" + "a" * 64, "sha256:" + "b" * 64}


def test_a_symlinked_spool_is_never_followed(tmp_path):
    world = EnvWorld()
    outside = tmp_path / "outside"
    (outside / "spool").mkdir(parents=True)
    (outside / "spool" / "keep").write_text("keep")
    (tmp_path / "managed" / "run-1").mkdir(parents=True)
    (tmp_path / "managed" / "run-1" / "sb").symlink_to(outside)
    manager, store, _ = recovery(tmp_path, world, resource_lease())

    with pytest.raises(RuntimeError, match="not the managed run spool"):
        manager.recover("run-1")

    assert (outside / "spool" / "keep").read_text() == "keep"
    assert_retained(store)


def test_an_endpoint_a_crash_left_half_made_is_removed_with_the_root(tmp_path):
    world = EnvWorld()
    endpoint(tmp_path, socket=False, modules=("rsi_sandbox_client.py",))
    empty = tmp_path / "managed" / "run-1" / "sb" / "89abcdef"
    empty.mkdir()
    manager, store, _ = recovery(
        tmp_path, world, resource_lease(sandbox_env_authority=True)
    )

    manager.recover("run-1")

    assert not (tmp_path / "managed" / "run-1" / "sb").exists()
    assert not store.read("run-1").recovery_required


def test_cleanup_removes_what_an_earlier_recovery_could_not(tmp_path):
    world = EnvWorld()
    manager, store, _ = recovery(
        tmp_path,
        world,
        resource_lease(sandbox_env_authority=True, status=RunStatus.CANCELLED),
    )
    endpoint(tmp_path)

    manager.cleanup("run-1")

    assert not (tmp_path / "managed" / "run-1" / "sb").exists()


def _foreign_link(directory, outside):
    (directory / "s").unlink()
    (directory / "s").symlink_to(outside / "keep")


def _foreign_file(directory, outside):
    (directory / "notes").write_text("")


def _foreign_module(directory, outside):
    (directory / "py" / "sitecustomize.py").write_text("")


def _module_link(directory, outside):
    (directory / "py" / "rsi_sandbox_client.py").unlink()
    (directory / "py" / "rsi_sandbox_client.py").symlink_to(outside / "keep")


def _regular_socket(directory, outside):
    (directory / "s").unlink()
    (directory / "s").write_text("")


def _linked_modules(directory, outside):
    for module in ENDPOINT_MODULES:
        (directory / "py" / module).unlink()
    (directory / "py").rmdir()
    (directory / "py").symlink_to(outside)


def _linked_endpoint(directory, outside):
    (directory.parent / "deadbeef").symlink_to(outside)


def _foreign_directory(directory, outside):
    (directory.parent / "keep").mkdir()


@pytest.mark.parametrize(
    "plant",
    [
        _foreign_link,
        _foreign_file,
        _foreign_module,
        _module_link,
        _regular_socket,
        _linked_modules,
        _linked_endpoint,
        _foreign_directory,
    ],
)
def test_anything_but_exactly_an_endpoint_fails_closed_removing_nothing(
    tmp_path, plant
):
    world = EnvWorld()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    directory = endpoint(tmp_path)
    plant(directory, outside)
    before = sorted(
        str(path.relative_to(tmp_path)) for path in (tmp_path / "managed").rglob("*")
    )
    manager, store, _ = recovery(
        tmp_path, world, resource_lease(sandbox_env_authority=True)
    )

    with pytest.raises(RuntimeError, match="foreign entry"):
        manager.recover("run-1")

    after = sorted(
        str(path.relative_to(tmp_path)) for path in (tmp_path / "managed").rglob("*")
    )
    assert after == before
    assert (outside / "keep").read_text() == "keep"
    assert store.read("run-1").recovery_required


def _stat(pid, *, state="S", start_ticks=100):
    fields = [state, "1", *["0"] * 17, str(start_ticks), *["0"] * 30]
    return f"{pid} (rsi (harness) x) " + " ".join(fields)


def test_coordinator_liveness_rejects_reused_pids_and_zombies(tmp_path):
    ticks = os.sysconf("SC_CLK_TCK")
    (tmp_path / "stat").write_text("cpu 1 2 3\nbtime 1000\n")
    for pid, state, start in (
        (10, "S", 100 * ticks),  # started at 1100
        (11, "Z", 100 * ticks),
        (12, "S", 500 * ticks),  # started at 1500: a reused PID
    ):
        (tmp_path / str(pid)).mkdir()
        (tmp_path / str(pid) / "stat").write_text(
            _stat(pid, state=state, start_ticks=start)
        )
    (tmp_path / "13").mkdir()
    (tmp_path / "13" / "stat").write_text("garbage")

    assert coordinator_may_run(10, 1200.0, proc=tmp_path) is True
    assert coordinator_may_run(11, 1200.0, proc=tmp_path) is False
    assert coordinator_may_run(12, 1200.0, proc=tmp_path) is False
    # Even PID 1 is hidden (hidepid): a missing PID proves nothing.
    assert coordinator_may_run(99, 1200.0, proc=tmp_path) is True
    (tmp_path / "1").mkdir()
    (tmp_path / "1" / "stat").write_text(_stat(1))
    assert coordinator_may_run(99, 1200.0, proc=tmp_path) is False
    assert coordinator_may_run(13, 1200.0, proc=tmp_path) is True  # unreadable
    assert coordinator_may_run(0, 1200.0, proc=tmp_path) is True


def test_coordinator_liveness_on_real_processes():
    assert coordinator_may_run(os.getpid(), time.time()) is True
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.monotonic() + 10
    while Path(f"/proc/{child.pid}/stat").read_text().split(") ")[1][0] != "Z":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert coordinator_may_run(child.pid, time.time()) is False  # a zombie
    child.wait()
    assert coordinator_may_run(child.pid, time.time()) is False  # reaped


def test_a_surviving_firewall_command_of_the_rule_is_found(tmp_path):
    rule_id = env.env_rule_id("run-1", ENV_ID)
    forward = firewall_rule_chains(rule_id)[0]
    (tmp_path / "self").mkdir()  # not a PID
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "cmdline").write_bytes(b"sleep\x00600\x00")
    (tmp_path / "8").mkdir()  # exited while listed: no cmdline any more

    assert firewall_commands_running((rule_id,), proc=tmp_path) is False
    # The SIGKILLed broker's iptables --wait child still runs.
    (tmp_path / "9").mkdir()
    (tmp_path / "9" / "cmdline").write_bytes(
        b"\x00".join([b"iptables", b"--wait", b"-N", forward.encode(), b""])
    )
    assert firewall_commands_running((rule_id,), proc=tmp_path) is True
    other = env.env_rule_id("run-2", ENV_ID)
    assert firewall_commands_running((other,), proc=tmp_path) is False
    assert firewall_commands_running((), proc=tmp_path) is False
    assert firewall_commands_running((rule_id,), proc=tmp_path / "gone") is True


def test_production_settlement_waits_then_proves_no_firewall_command(tmp_path):
    world = EnvWorld()
    backend = ProductionRecoveryBackend(
        world.network.client,
        None,
        world.network.firewall,
        settle_sec=0.05,
        proc=tmp_path,
    )
    began = time.monotonic()
    assert backend.settle_sandbox_creates(("rsi-run-1-sbx-x",)) is True
    assert time.monotonic() - began >= 0.05
