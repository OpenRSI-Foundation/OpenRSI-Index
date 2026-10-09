"""Brokered bridges: deterministic identity, rule before bridge, exact rollback."""

import copy

import pytest
import requests
from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerRef, ManagedNetwork, NetworkPolicy
from rsi_harness.runtime import sandbox_env_contracts as env
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    NetworkPolicyEnforcer,
    firewall_rule_chains,
    managed_bridge_interface,
)
from rsi_harness.runtime.sandbox_docker import sandbox_labels
from rsi_harness.runtime.sandbox_network import (
    MAX_ADDRESSES_PER_HOST,
    SandboxNetworkBackend,
    SandboxNetworkPlan,
    env_lease_network_plan,
    plan_builder_network,
    plan_env_network,
)
from tests.fakes import FakeDockerClient, FakeFirewallBackend
from tests.runtime.test_network import InMemoryIptablesRunner, policy_chain, reject
from tests.runtime.test_sandbox_budget import make_child
from tests.runtime.test_sandbox_policy_v2 import make_builder_lease, make_env_lease
from tests.runtime.test_sandbox_policy_v2 import owner as make_owner
from tests.sandbox_helpers import make_env_spec

ENV_ID = "e" + "1" * 32
SIBLING_ID = "e" + "2" * 32
BUILDER_ID = "b" + "3" * 32
GATEWAY = "172.17.0.1"


def spec(network="public", *, single=False):
    raw = make_env_spec()
    raw["network"] = network
    if single:
        del raw["services"]["kv"]
        raw["services"]["main"]["depends_on"] = {}
    return env.parse_env_spec(raw)


def env_plan(network="public", env_id=ENV_ID, phase="judge"):
    plan = plan_env_network(make_owner(phase), env_id, spec(network))
    assert plan is not None
    return plan


def daemon_error(message, status=500):
    """An APIError carrying the Engine's HTTP answer."""
    response = requests.Response()
    response.status_code = status
    return APIError(message, response=response)


class FakeNetwork:
    def __init__(self, world, network_id, attrs):
        self._world = world
        self.id = network_id
        self.attrs = attrs

    def remove(self):
        world = self._world
        world.events.append(("network-remove", self.id))
        if world.fail_network_remove == "before":
            raise APIError("network has active endpoints")
        world.store.pop(self.id, None)
        world.after_remove(self)
        if world.fail_network_remove == "after":
            raise APIError("network remove response lost")


class FakeNetworks:
    """Engine-shaped bridge inspection with failure and drift injection."""

    def __init__(self, world):
        self._world = world

    def create(self, name, **kwargs):
        world = self._world
        world.events.append(("network-create", name))
        world.created.append({"name": name, **kwargs})
        if world.fail_create == "before":
            raise daemon_error("network create denied", 403)
        network_id = f"{len(world.created):064x}"
        attrs = {
            "Id": network_id,
            "Name": name,
            "Driver": kwargs["driver"],
            "Internal": kwargs["internal"],
            "EnableIPv6": bool(kwargs["enable_ipv6"]),
            "IPAM": {
                "Driver": "default",
                "Options": None,
                "Config": [{"Subnet": "172.16.8.0/22", "Gateway": "172.16.8.1"}],
            },
            "Labels": dict(kwargs["labels"]),
            "Options": dict(kwargs["options"]),
            "Containers": {},
        }
        world.drift(attrs)
        network = FakeNetwork(world, network_id, attrs)
        if world.unanswered is not None:
            # The client gives up; dockerd finishes the bridge afterwards.
            world.late.append(network)
            raise world.unanswered
        world.store[network_id] = network
        if world.fail_create == "after":
            raise daemon_error("network create failed after the bridge")
        return network

    def get(self, network_id):
        self._world.gets += 1
        error = self._world.get_errors.get(self._world.gets)
        if error is not None:
            raise error
        if network_id not in self._world.store:
            raise NotFound(network_id)
        return self._world.store[network_id]

    def list(self, *, filters):
        # Like the Engine, the name filter also matches substrings.
        return [
            network
            for network in self._world.store.values()
            if filters["name"] in network.attrs["Name"]
        ]


class NetworkWorld:
    def __init__(self, firewall=None, events=None):
        self.events = [] if events is None else events
        self.created = []
        self.store = {}
        self.late = []
        self.gets = 0
        self.get_errors = {}
        self.fail_create = None
        self.unanswered = None
        self.fail_network_remove = None
        self.after_remove = lambda network: None
        self.drift = lambda attrs: None
        self.client = FakeDockerClient()
        self.client.networks = FakeNetworks(self)
        if firewall is None:
            firewall = FakeFirewallBackend()
            firewall.events = self.events
        self.firewall = firewall
        self.backend = SandboxNetworkBackend(
            self.client, firewall, engine_destinations=(GATEWAY,)
        )

    def names(self):
        return sorted(network.attrs["Name"] for network in self.store.values())

    def settle(self):
        for network in self.late:
            self.store[network.id] = network
        self.late.clear()

    def add_bridge(self, plan, name=None, labels=None):
        """A bridge that no create() of the backend made (crash or foreign)."""
        return self.client.networks.create(
            name or plan.name,
            driver="bridge",
            internal=plan.internal,
            enable_ipv6=False,
            labels=plan.labels if labels is None else labels,
            options={"com.docker.network.bridge.name": plan.bridge_interface},
        ).id


class LoggedIptables(InMemoryIptablesRunner):
    """Real iptables compilation whose commands share the Docker event log."""

    def __init__(self, events):
        super().__init__()
        self._events = events

    def __call__(self, command):
        if command[2] != "-S":
            self._events.append(("iptables", command[2], command[3]))
        return super().__call__(command)


def iptables_world():
    events = []
    runner = LoggedIptables(events)
    firewall = DockerIptablesFirewallBackend(FakeDockerClient(), runner=runner)
    return NetworkWorld(firewall, events), runner


def pending_env_lease(**updates):
    """The env lease as journaled mid-create: planned, pending, no containers."""
    values = dict(
        state="planned",
        network_id=None,
        services=tuple(
            service.model_copy(update={"state": "planned", "container_id": None})
            for service in make_env_lease().services
        ),
        pending_mutation=True,
    )
    values.update(updates)
    return make_env_lease(**values)


@pytest.mark.parametrize(
    ("network", "internal", "mode"),
    (("public", False, "public"), ("none", True, "no-network")),
)
def test_env_plan_derives_every_name_from_run_and_env(network, internal, mode):
    plan = env_plan(network)

    assert plan.name == "rsi-sbnet-1111111111111111" == env.env_network_name(ENV_ID)
    assert plan.rule_id == "rsi-run-1-sbx-1111111111111111"
    assert plan.bridge_interface == managed_bridge_interface(plan.name)
    assert (plan.internal, plan.mode, plan.intra_bridge_accept) == (
        internal,
        mode,
        True,
    )
    assert plan.labels == {
        "rsi-harness.run-id": "run-1",
        "rsi-harness.task-id": "task",
        "rsi-harness.role": "sandbox-env-net",
        "rsi-harness.sandbox-phase": "judge",
        "rsi-harness.round-id": "agent-1",
        "rsi-harness.sandbox-env": ENV_ID,
    }
    assert env_plan(network) == plan
    lease = make_env_lease(
        network_mode=network, network_name=plan.name, rule_id=plan.rule_id
    )
    assert env_lease_network_plan(lease) == plan


def test_work_env_labels_carry_no_round_and_single_service_none_has_no_bridge():
    work = env_plan(phase="work")
    assert "rsi-harness.round-id" not in work.labels
    assert work.labels["rsi-harness.sandbox-phase"] == "work"

    assert plan_env_network(make_owner(), ENV_ID, spec("none", single=True)) is None
    single_public = plan_env_network(make_owner(), ENV_ID, spec(single=True))
    assert single_public is not None and not single_public.internal
    lease = make_env_lease(
        network_mode="none", network_name=None, network_id=None, rule_id=None
    )
    assert env_lease_network_plan(lease) is None


def test_builder_plan_is_the_plain_public_bridge_named_by_builder_identity():
    plan = plan_builder_network(make_owner(), BUILDER_ID)
    lease = make_builder_lease(BUILDER_ID)

    assert (plan.name, plan.rule_id) == (lease.network_name, lease.rule_id)
    assert (plan.internal, plan.mode, plan.intra_bridge_accept) == (
        False,
        "public",
        False,
    )
    assert plan.labels["rsi-harness.role"] == "sandbox-builder-net"
    assert plan.labels["rsi-harness.sandbox-builder"] == BUILDER_ID
    assert "rsi-harness.sandbox-env" not in plan.labels


@pytest.mark.parametrize("phase", ("work", "judge"))
def test_network_labels_share_the_v1_sandbox_label_base(phase):
    # Spec 3.2: one helper (sandbox_object_labels) owns every sandbox label,
    # so the owner-derived keys of children, envs and builders never drift.
    def base(labels, handle_key):
        return {
            key: value
            for key, value in labels.items()
            if key not in ("rsi-harness.role", handle_key)
        }

    owner = make_owner(phase)
    child = base(sandbox_labels(make_child(owner=owner)), "rsi-harness.sandbox-id")
    env_net = env_plan(phase=phase)
    builder_net = plan_builder_network(owner, BUILDER_ID)

    assert base(env_net.labels, "rsi-harness.sandbox-env") == child
    assert base(builder_net.labels, "rsi-harness.sandbox-builder") == child


@pytest.mark.parametrize(
    "make",
    (
        lambda: plan_env_network(make_owner(), BUILDER_ID, spec()),
        lambda: plan_env_network(make_owner(), "e" + "Z" * 32, spec()),
        lambda: plan_builder_network(make_owner(), ENV_ID),
        lambda: SandboxNetworkPlan(make_owner(), BUILDER_ID, "none"),
        lambda: SandboxNetworkPlan(make_owner(), ENV_ID, "egress"),
        lambda: SandboxNetworkPlan(
            make_owner(), ENV_ID, "public", allowlist=("pypi.org",)
        ),
        lambda: SandboxNetworkPlan(
            make_owner(), BUILDER_ID, "allowlist", allowlist=("pypi.org",)
        ),
    ),
)
def test_plans_refuse_identities_they_cannot_recompute(make):
    with pytest.raises(ValueError):
        make()


@pytest.mark.parametrize("network", ("public", "none"))
def test_env_rule_accepts_its_own_bridge_before_private_range_rejects(network):
    world, runner = iptables_world()
    plan = env_plan(network)

    world.backend.create(plan)

    forward = runner.chains[policy_chain(runner, plan.rule_id, "forward")]
    accept = ["-o", plan.bridge_interface, "-j", "ACCEPT"]
    assert forward.index(accept) == 1
    assert forward.index(accept) < forward.index(reject(f"{GATEWAY}/32"))
    assert forward.index(accept) < forward.index(reject("172.16.0.0/12"))
    assert forward[-1][:2] == ["-j", "ACCEPT" if network == "public" else "REJECT"]


def test_builder_rule_set_equals_the_public_parent_rule_set():
    world, runner = iptables_world()
    builder = plan_builder_network(make_owner(), BUILDER_ID)
    world.backend.create(builder)
    public_env = env_plan("public", SIBLING_ID)
    world.backend.create(public_env)

    parent = InMemoryIptablesRunner()
    lease = NetworkPolicyEnforcer(
        run_id="run-1",
        firewall=DockerIptablesFirewallBackend(FakeDockerClient(), runner=parent),
        engine_destinations=(GATEWAY,),
    ).apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="public"),
        network=ManagedNetwork(
            network_id="work-network",
            name=builder.name,
            run_id="run-1",
            task_id="task",
            role="work",
            internal=False,
        ),
    )
    assert world.backend.rules(builder).intra_bridge_accept is False
    for kind in ("forward", "input"):
        assert (
            runner.chains[policy_chain(runner, builder.rule_id, kind)]
            == parent.chains[policy_chain(parent, lease.rule_id, kind)]
        )
    builder_forward = runner.chains[policy_chain(runner, builder.rule_id, "forward")]
    env_forward = runner.chains[policy_chain(runner, public_env.rule_id, "forward")]
    assert env_forward == [
        builder_forward[0],
        ["-o", public_env.bridge_interface, "-j", "ACCEPT"],
        *builder_forward[1:],
    ]


def test_create_installs_rule_before_bridge_then_attests_both():
    world = NetworkWorld()
    plan = env_plan("none")

    network_id = world.backend.create(plan)

    assert world.events == [
        ("probe", None),
        ("install", plan.rule_id),
        ("network-create", plan.name),
        ("is_installed", plan.rule_id),
    ]
    assert world.created == [
        {
            "name": plan.name,
            "driver": "bridge",
            "internal": True,
            "enable_ipv6": False,
            "check_duplicate": True,
            "labels": plan.labels,
            "options": {"com.docker.network.bridge.name": plan.bridge_interface},
        }
    ]
    assert world.firewall.installed == {plan.rule_id: world.backend.rules(plan)}
    world.backend.attest(plan, network_id)


def test_real_iptables_commands_run_before_bridge_and_after_its_removal():
    world, _runner = iptables_world()
    plan = env_plan()

    network_id = world.backend.create(plan)
    created = world.events.index(("network-create", plan.name))
    assert world.events[created - 1] == ("iptables", "-I", "INPUT")
    assert all(event[0] == "iptables" for event in world.events[:created])

    world.events.clear()
    world.backend.remove(plan, network_id)

    assert world.events[0] == ("network-remove", network_id)
    assert {event[0] for event in world.events[1:]} == {"iptables"}
    assert world.firewall.exists(plan.rule_id) is False
    assert world.names() == []


def test_remove_deletes_bridge_before_rule_proves_both_absent_and_is_idempotent():
    world = NetworkWorld()
    plan = env_plan()
    network_id = world.backend.create(plan)
    world.events.clear()

    world.backend.remove(plan, network_id)

    assert world.events == [
        ("network-remove", network_id),
        ("remove", plan.rule_id),
        ("exists", plan.rule_id),
    ]
    assert world.store == {} and world.firewall.installed == {}
    world.events.clear()
    world.backend.remove(plan, network_id)
    assert world.events == [("remove", plan.rule_id), ("exists", plan.rule_id)]


def fail_install(world):
    def install(rule_id, rules):
        world.events.append(("install", rule_id))
        raise SetupError("iptables denied")

    world.firewall.install = install


def flip_internal(world):
    world.drift = lambda attrs: attrs.update(Internal=not attrs["Internal"])


def lose_rule(world):
    world.firewall.is_installed = lambda rule_id, rules: False


@pytest.mark.parametrize(
    ("inject", "bridge_created"),
    (
        (fail_install, False),
        (lambda world: setattr(world, "fail_create", "before"), False),
        (lambda world: setattr(world, "fail_create", "after"), True),
        (flip_internal, True),
        (lose_rule, True),
    ),
    ids=("install", "create", "create-failed-late", "attest-bridge", "attest-rule"),
)
def test_injected_create_failure_rolls_back_bridge_before_rule(inject, bridge_created):
    world = NetworkWorld()
    plan = env_plan()
    inject(world)

    with pytest.raises(SetupError, match="proven absent"):
        world.backend.create(plan)

    assert world.store == {} and world.firewall.installed == {}
    events = [event[0] for event in world.events]
    assert events.index("install") < events.index("remove")
    if bridge_created:
        assert events.index("network-create") < events.index("network-remove")
        assert events.index("network-remove") < events.index("remove")
    else:
        assert "network-remove" not in events
    assert events[-2:] == ["remove", "exists"]


@pytest.mark.parametrize(
    "unanswered",
    (
        requests.exceptions.ReadTimeout("read timed out"),
        requests.exceptions.ConnectionError("connection aborted"),
        APIError("no HTTP response"),
    ),
    ids=("read-timeout", "connection-lost", "api-error-without-status"),
)
def test_unanswered_create_keeps_the_rule_until_recovery_removes_the_late_bridge(
    unanswered,
):
    world = NetworkWorld()
    plan = env_plan()
    world.unanswered = unanswered

    with pytest.raises(InfrastructureError, match="recovery_required.*unknown"):
        world.backend.create(plan)

    # dockerd may still be creating the bridge: nothing is judged absent.
    assert world.events == [
        ("probe", None),
        ("install", plan.rule_id),
        ("network-create", plan.name),
    ]
    assert plan.rule_id in world.firewall.installed
    lease = pending_env_lease()
    with pytest.raises(InfrastructureError, match="recovery_required.*pending create"):
        world.backend.recover(lease)
    assert plan.rule_id in world.firewall.installed

    world.settle()
    [late] = world.store
    assert world.backend.recover(lease) == (late,)

    assert world.store == {} and world.firewall.installed == {}
    events = [event[0] for event in world.events]
    assert events.index("network-remove") < events.index("remove")


def test_failed_probe_mutates_nothing():
    world = NetworkWorld()
    world.firewall.probe_result = False

    with pytest.raises(SetupError, match="probe"):
        world.backend.create(env_plan())

    assert world.events == [("probe", None)]


def test_unproven_bridge_rollback_keeps_the_rule_and_requires_recovery():
    world = NetworkWorld()
    plan = env_plan()
    flip_internal(world)
    world.fail_network_remove = "before"

    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.backend.create(plan)

    assert world.names() == [plan.name]
    assert plan.rule_id in world.firewall.installed
    assert ("remove", plan.rule_id) not in world.events


def test_unproven_rule_rollback_requires_recovery_after_bridge_removal():
    world = NetworkWorld()
    plan = env_plan()
    lose_rule(world)
    world.firewall.remove = lambda rule_id: None

    with pytest.raises(InfrastructureError, match="recovery_required.*remains"):
        world.backend.create(plan)

    assert world.store == {}
    assert world.firewall.exists(plan.rule_id) is True


def test_partial_iptables_install_is_rolled_back_by_exact_removal():
    world, runner = iptables_world()
    runner.fail_input_jump = True

    with pytest.raises(SetupError, match="proven absent"):
        world.backend.create(env_plan())

    assert runner.chains == {"DOCKER-USER": [], "INPUT": []}
    assert not any(event[0] == "network-create" for event in world.events)


def drift(key, value):
    return lambda attrs: attrs.__setitem__(key, value)


@pytest.mark.parametrize(
    "mutate",
    (
        drift("Id", "f" * 64),
        drift("Name", "rsi-sbnet-2222222222222222"),
        drift("Driver", "macvlan"),
        lambda attrs: attrs.update(Internal=not attrs["Internal"]),
        drift("EnableIPv6", True),
        lambda attrs: attrs["Labels"].update({"extra": "1"}),
        lambda attrs: attrs["Labels"].pop("rsi-harness.round-id"),
        drift("Options", {"com.docker.network.bridge.name": "rsi0123456789ab"}),
        lambda attrs: attrs["Options"].update(
            {"com.docker.network.bridge.enable_icc": "true"}
        ),
        lambda attrs: attrs["IPAM"].update(Driver="custom"),
        # Outside the rejected ranges nothing but Docker keeps envs apart.
        lambda attrs: attrs["IPAM"].update(Config=[{"Subnet": "203.0.113.0/24"}]),
        lambda attrs: attrs["IPAM"].update(Config=[{"Subnet": "172.0.0.0/8"}]),
        lambda attrs: attrs["IPAM"]["Config"].append({"Subnet": "fd00:1::/64"}),
        lambda attrs: attrs["IPAM"].update(Config=[]),
        lambda attrs: attrs["IPAM"].update(Config=[{"Gateway": "172.16.8.1"}]),
        lambda attrs: attrs["Containers"].update({"9" * 64: {}}),
    ),
)
def test_attest_rejects_every_bridge_drift(mutate):
    world = NetworkWorld()
    plan = env_plan("none")
    network_id = world.backend.create(plan)
    own = "a" * 64
    world.store[network_id].attrs["Containers"][own] = {"Name": "rsi-sbx-1-0"}
    world.backend.attest(plan, network_id, containers=(own, "b" * 64))

    mutate(world.store[network_id].attrs)

    with pytest.raises(InfrastructureError):
        world.backend.attest(plan, network_id, containers=(own, "b" * 64))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda world, plan: world.firewall.installed.pop(plan.rule_id),
        lambda world, plan: world.firewall.installed.__setitem__(
            plan.rule_id, world.backend.rules(env_plan("public"))
        ),
    ),
    ids=("absent", "public-rules-on-none-bridge"),
)
def test_attest_rejects_a_missing_or_different_rule(mutate):
    world = NetworkWorld()
    plan = env_plan("none")
    network_id = world.backend.create(plan)

    mutate(world, plan)

    with pytest.raises(InfrastructureError, match="not installed exactly"):
        world.backend.attest(plan, network_id)


@pytest.mark.parametrize(
    "subnet", ("10.20.0.0/22", "100.64.4.0/22", "172.31.252.0/22", "192.168.4.0/24")
)
def test_attest_accepts_a_subnet_inside_any_rejected_range(subnet):
    world = NetworkWorld()
    world.drift = lambda attrs: attrs["IPAM"].update(Config=[{"Subnet": subnet}])
    plan = env_plan()

    network_id = world.backend.create(plan)

    world.backend.attest(plan, network_id)


def test_attest_reports_an_absent_bridge():
    world = NetworkWorld()
    plan = env_plan()

    with pytest.raises(InfrastructureError, match="absent"):
        world.backend.attest(plan, "0" * 64)


def test_attest_reports_an_uninspectable_bridge():
    world = NetworkWorld()
    plan = env_plan()
    network_id = world.backend.create(plan)
    world.gets = 0
    world.get_errors = {1: requests.exceptions.ReadTimeout("read timed out")}

    with pytest.raises(InfrastructureError, match="cannot inspect"):
        world.backend.attest(plan, network_id)


def test_removal_never_touches_a_foreign_network_holding_the_planned_name():
    world = NetworkWorld()
    plan = env_plan()
    world.firewall.install(plan.rule_id, world.backend.rules(plan))
    world.add_bridge(plan, labels={"rsi-harness.run-id": "someone-else"})
    world.events.clear()

    with pytest.raises(InfrastructureError, match="recovery_required.*not owned"):
        world.backend.remove(plan)

    assert world.names() == [plan.name]
    assert plan.rule_id in world.firewall.installed
    assert not any(event[0] in ("network-remove", "remove") for event in world.events)


def test_removal_ignores_a_lookalike_that_only_contains_the_planned_name():
    world = NetworkWorld()
    plan = env_plan()
    world.firewall.install(plan.rule_id, world.backend.rules(plan))
    world.add_bridge(plan, name=plan.name + "-longer")
    world.events.clear()

    assert world.backend.remove(plan) == ()

    assert world.names() == [plan.name + "-longer"]
    assert world.firewall.installed == {}
    assert not any(event[0] == "network-remove" for event in world.events)


def test_a_remove_error_after_the_bridge_is_gone_still_proves_absence():
    world = NetworkWorld()
    plan = env_plan()
    network_id = world.backend.create(plan)
    world.fail_network_remove = "after"

    assert world.backend.remove(plan, network_id) == (network_id,)

    assert world.store == {} and world.firewall.installed == {}


def test_a_planned_bridge_reappearing_after_removal_keeps_the_rule():
    world = NetworkWorld()
    plan = env_plan()
    network_id = world.backend.create(plan)
    reappeared = []

    def recreate(network):
        world.after_remove = lambda network: None
        reappeared.append(world.add_bridge(plan))

    world.after_remove = recreate

    with pytest.raises(InfrastructureError, match="recovery_required.*remains"):
        world.backend.remove(plan, network_id)

    assert list(world.store) == reappeared and network_id not in reappeared
    assert plan.rule_id in world.firewall.installed
    assert ("remove", plan.rule_id) not in world.events


@pytest.mark.parametrize(
    ("failing_get", "error"),
    (
        (2, requests.exceptions.ConnectionError("connection refused")),
        (3, requests.exceptions.ReadTimeout("read timed out")),
    ),
    ids=("before-removal", "after-removal"),
)
def test_transport_errors_while_proving_bridge_absence_require_recovery(
    failing_get, error
):
    world = NetworkWorld()
    plan = env_plan()
    network_id = world.backend.create(plan)
    # Gets: discovery, then before and after the remove call.
    world.gets = 0
    world.get_errors = {failing_get: error}

    with pytest.raises(InfrastructureError, match="recovery_required") as raised:
        world.backend.remove(plan, network_id)

    assert raised.value.__cause__ is error
    assert (network_id in world.store) is (failing_get == 2)
    assert plan.rule_id in world.firewall.installed
    assert ("remove", plan.rule_id) not in world.events


def test_removal_requires_the_journaled_bridge_to_be_owned():
    world = NetworkWorld()
    plan = env_plan()
    other = env_plan(env_id=SIBLING_ID)
    other_id = world.backend.create(other)

    with pytest.raises(InfrastructureError, match="not owned"):
        world.backend.remove(plan, other_id)

    assert world.names() == [other.name]


@pytest.mark.parametrize(
    ("crash_point", "found"),
    (
        ("planned", False),
        ("rule-installed", False),
        ("bridge-created", True),
        ("bridge-journaled", True),
    ),
)
def test_recover_from_every_env_journal_point_converges_or_fails_closed(
    crash_point, found
):
    world, runner = iptables_world()
    plan = env_plan()
    network_id = None
    if crash_point != "planned":
        world.firewall.install(plan.rule_id, world.backend.rules(plan))
    if found:
        network_id = world.add_bridge(plan)
    lease = pending_env_lease(
        network_id=network_id if crash_point == "bridge-journaled" else None
    )
    recovered = SandboxNetworkBackend(world.client, world.firewall)

    if found:
        assert recovered.recover(lease) == (network_id,)
    else:
        # As for a pending child: no object may mean a create still in
        # flight, so recovery fails closed and keeps whatever rule exists.
        with pytest.raises(InfrastructureError, match="recovery_required.*pending"):
            recovered.recover(lease)
        assert world.firewall.exists(plan.rule_id) is (crash_point != "planned")
    # Once the lease no longer journals a pending create, recovery converges
    # and stays idempotent.
    settled = lease.model_copy(update={"pending_mutation": False, "network_id": None})
    assert recovered.recover(settled) == ()
    assert recovered.recover(settled) == ()

    assert world.names() == []
    assert runner.chains == {"DOCKER-USER": [], "INPUT": []}


def test_recover_builder_network_and_skip_an_env_without_bridge():
    world = NetworkWorld()
    builder = plan_builder_network(make_owner(), BUILDER_ID)
    world.backend.create(builder)

    [bridge] = world.store

    assert world.backend.recover(make_builder_lease(BUILDER_ID, network_id=None)) == (
        bridge,
    )

    assert world.store == {} and world.firewall.installed == {}
    world.events.clear()
    unbridged = make_env_lease(
        network_mode="none", network_name=None, network_id=None, rule_id=None
    )
    assert world.backend.recover(unbridged) == ()
    assert world.events == []


def test_sibling_env_rules_attest_together_and_remove_independently():
    world, _runner = iptables_world()
    plans = (env_plan(), env_plan("none", SIBLING_ID))
    network_ids = [world.backend.create(plan) for plan in plans]

    for plan, network_id in zip(plans, network_ids, strict=True):
        world.backend.attest(plan, network_id)
    world.backend.remove(plans[0], network_ids[0])

    world.backend.attest(plans[1], network_ids[1])
    assert world.firewall.exists(plans[0].rule_id) is False
    assert world.names() == [plans[1].name]


def test_rules_are_a_pure_function_of_the_plan():
    world = NetworkWorld()
    plan = env_plan()
    rules = world.backend.rules(plan)

    assert rules == world.backend.rules(copy.deepcopy(plan))
    assert rules.bridge_interface == plan.bridge_interface
    assert (rules.mode, rules.intra_bridge_accept) == ("public", True)
    assert rules.exact_endpoints == rules.allow_networks == rules.dns_resolvers == ()
    assert [str(address) for address in rules.engine_destinations] == [GATEWAY]


# -- allowlist -------------------------------------------------------------------

PYPI = "151.101.0.223"
PYPI_NEXT = "151.101.64.223"


class Resolver:
    """Hostname answers by name; an exception value is raised."""

    def __init__(self, answers):
        self.answers = dict(answers)
        self.calls = []

    def __call__(self, hostname):
        self.calls.append(hostname)
        answer = self.answers.get(hostname, ())
        if isinstance(answer, Exception):
            raise answer
        return answer


def allow_plan(entries, private=(), env_id=ENV_ID):
    raw = make_env_spec()
    raw.update(network="allowlist", allowlist=list(entries))
    plan = plan_env_network(
        make_owner(), env_id, env.parse_env_spec(raw), private_cidrs=private
    )
    assert plan is not None
    return plan


def allow_world(answers):
    world, runner = iptables_world()
    resolver = Resolver(answers)
    world.backend = SandboxNetworkBackend(
        world.client, world.firewall, engine_destinations=(GATEWAY,), resolver=resolver
    )
    return world, runner, resolver


def allow_chain(runner, plan):
    return runner.chains[firewall_rule_chains(plan.rule_id)[2]]


def accept(destination, port=None):
    match = ["-d", destination]
    if port is not None:
        match += ["-p", "tcp", "-m", "tcp", "--dport", str(port)]
    return [*match, "-j", "ACCEPT"]


ANSWERS = {
    "pypi.org": (PYPI,),
    "lan.example": ("192.168.1.9",),
    "metadata.example": ("169.254.169.254",),
    "gateway.example": (GATEWAY,),
    "mixed.example": ("8.8.4.4", "172.20.0.5", "fd00::1"),
}
ENTRIES = (
    "PyPI.org.:443",
    "8.8.8.8",
    "1.1.1.0/24:53",
    "10.1.2.3",
    "lan.example",
    "metadata.example",
    "gateway.example",
    "mixed.example",
    "missing.example",
)


def test_allowlist_plan_is_a_public_bridge_whose_rule_ends_in_reject():
    plan = allow_plan(ENTRIES, private=("10.1.0.0/16",))

    assert (plan.internal, plan.mode, plan.intra_bridge_accept) == (
        False,
        "allowlist",
        True,
    )
    assert plan.allowlist[0] == "pypi.org:443"
    assert (plan.name, plan.rule_id) == (env_plan().name, env_plan().rule_id)
    # Removal needs only the names: the lease journals no entries.
    lease = make_env_lease(
        network_mode="allowlist", network_name=plan.name, rule_id=plan.rule_id
    )
    rebuilt = env_lease_network_plan(lease)
    assert (rebuilt.name, rebuilt.rule_id, rebuilt.allowlist) == (
        plan.name,
        plan.rule_id,
        (),
    )


def test_allowlist_rule_jumps_to_resolved_accepts_before_every_private_reject():
    world, runner, _resolver = allow_world(ANSWERS)
    plan = allow_plan(ENTRIES, private=("10.1.0.0/16",))

    network_id = world.backend.create(plan)

    forward = runner.chains[policy_chain(runner, plan.rule_id, "forward")]
    jump = ["-j", firewall_rule_chains(plan.rule_id)[2]]
    assert forward[1] == ["-o", plan.bridge_interface, "-j", "ACCEPT"]
    assert forward.index(reject(f"{GATEWAY}/32")) < forward.index(jump)
    assert forward.index(jump) < forward.index(reject("10.0.0.0/8"))
    assert forward[-1] == ["-j", "REJECT", "--reject-with", "icmp-port-unreachable"]
    # Sorted; private, metadata, the engine and IPv6 answers never appear,
    # except the operator's private CIDR.
    assert allow_chain(runner, plan) == [
        accept("1.1.1.0/24", 53),
        accept("8.8.4.4/32"),
        accept("8.8.8.8/32"),
        accept("10.1.2.3/32"),
        accept(f"{PYPI}/32", 443),
    ]
    assert world.backend.allowlist_notes(plan) == (
        "allowlist entry lan.example reaches only blocked addresses; it allows nothing",
        "allowlist entry metadata.example reaches only blocked addresses; "
        "it allows nothing",
        "allowlist entry gateway.example reaches only blocked addresses; "
        "it allows nothing",
        "allowlist entry missing.example resolves to no address; it allows nothing",
    )
    world.backend.attest(plan, network_id)


def test_without_the_private_cidr_a_listed_private_address_stays_blocked():
    world, runner, _resolver = allow_world(ANSWERS)
    plan = allow_plan(("10.1.2.3", "8.8.8.8"))

    world.backend.create(plan)

    assert allow_chain(runner, plan) == [accept("8.8.8.8/32")]
    assert world.backend.allowlist_notes(plan) == (
        "allowlist entry 10.1.2.3 reaches only blocked addresses; it allows nothing",
    )


def test_refresh_replaces_the_allow_chain_in_place_new_accepts_first():
    world, runner, resolver = allow_world(ANSWERS)
    plan = allow_plan(("pypi.org:443", "8.8.8.8"))
    network_id = world.backend.create(plan)
    rules = list(runner.chains)

    # Unchanged answers and a failed lookup (the old addresses stay) touch
    # nothing.
    assert world.backend.refresh(plan) is False
    resolver.answers["pypi.org"] = OSError("resolver down")
    assert world.backend.refresh(plan) is False
    world.events.clear()

    resolver.answers["pypi.org"] = (PYPI_NEXT,)
    assert world.backend.refresh(plan) is True

    chain = firewall_rule_chains(plan.rule_id)[2]
    mutations = [event for event in world.events if event[0] == "iptables"]
    assert (
        mutations == [("iptables", "-A", chain)] * 3 + [("iptables", "-D", chain)] * 2
    )
    # A new answer joins the addresses seen before (round-robin DNS).
    assert allow_chain(runner, plan) == [
        accept("8.8.8.8/32"),
        accept(f"{PYPI}/32", 443),
        accept(f"{PYPI_NEXT}/32", 443),
    ]
    assert list(runner.chains) == rules  # no chain or jump was added
    world.backend.attest(plan, network_id)


def test_refresh_keeps_a_bounded_newest_first_set_per_hostname():
    world, runner, resolver = allow_world({"cdn.example": ("8.8.0.0",)})
    plan = allow_plan(("cdn.example",))
    world.backend.create(plan)

    for index in range(1, 2 * MAX_ADDRESSES_PER_HOST):
        resolver.answers["cdn.example"] = (f"8.8.0.{index}",)
        world.backend.refresh(plan)

    kept = [rule[1] for rule in allow_chain(runner, plan)]
    newest = range(MAX_ADDRESSES_PER_HOST, 2 * MAX_ADDRESSES_PER_HOST)
    assert kept == [f"8.8.0.{index}/32" for index in newest]


def test_only_literal_entries_are_never_resolved_or_refreshed():
    world, _runner, resolver = allow_world({})
    plan = allow_plan(("8.8.8.8", "1.1.1.0/24:443"))
    world.backend.create(plan)

    assert world.backend.refresh(plan) is False
    assert resolver.calls == []


def test_remove_takes_the_allow_chain_and_a_later_refresh_never_restores_it():
    world, runner, resolver = allow_world(ANSWERS)
    plan = allow_plan(("pypi.org",))
    network_id = world.backend.create(plan)

    world.backend.remove(plan, network_id)
    resolver.answers["pypi.org"] = (PYPI_NEXT,)
    world.events.clear()

    assert world.backend.refresh(plan) is False
    assert world.events == []
    assert runner.chains == {"DOCKER-USER": [], "INPUT": []}


def test_recovery_from_the_lease_removes_all_three_chains():
    world, runner, _resolver = allow_world(ANSWERS)
    plan = allow_plan(("pypi.org", "8.8.8.8"))
    network_id = world.backend.create(plan)
    lease = make_env_lease(
        network_mode="allowlist",
        network_name=plan.name,
        rule_id=plan.rule_id,
        network_id=network_id,
    )

    recovered = SandboxNetworkBackend(world.client, world.firewall)
    assert recovered.recover(lease) == (network_id,)

    assert runner.chains == {"DOCKER-USER": [], "INPUT": []}
    assert world.firewall.exists(plan.rule_id) is False


def test_a_partial_allowlist_install_is_rolled_back_with_its_allow_chain():
    world, runner, _resolver = allow_world(ANSWERS)
    runner.fail_input_jump = True

    with pytest.raises(SetupError, match="proven absent"):
        world.backend.create(allow_plan(("pypi.org",)))

    assert runner.chains == {"DOCKER-USER": [], "INPUT": []}


@pytest.mark.parametrize(
    "mutate",
    (
        lambda chain: chain.append(accept("9.9.9.9/32")),
        lambda chain: chain.pop(),
        lambda chain: chain.reverse(),
    ),
    ids=("extra", "missing", "reordered"),
)
def test_attest_rejects_any_allow_chain_drift(mutate):
    world, runner, _resolver = allow_world(ANSWERS)
    plan = allow_plan(("pypi.org", "8.8.8.8"))
    network_id = world.backend.create(plan)

    mutate(allow_chain(runner, plan))

    with pytest.raises(InfrastructureError, match="not installed exactly"):
        world.backend.attest(plan, network_id)


def test_a_failed_allow_chain_update_raises_and_is_retried_next_turn():
    world = NetworkWorld()
    resolver = Resolver({"pypi.org": (PYPI,)})
    world.backend = SandboxNetworkBackend(
        world.client, world.firewall, resolver=resolver
    )
    plan = allow_plan(("pypi.org",))
    world.backend.create(plan)
    installed = world.firewall.installed[plan.rule_id]
    resolver.answers["pypi.org"] = (PYPI_NEXT,)

    def update(rule_id, rules):
        raise InfrastructureError("iptables denied")

    world.firewall.update = update
    with pytest.raises(InfrastructureError, match="denied"):
        world.backend.refresh(plan)
    del world.firewall.update

    assert world.backend.refresh(plan) is True
    assert world.firewall.installed[plan.rule_id] != installed
    assert [str(rule[0]) for rule in world.backend.rules(plan).allow_rules] == [
        f"{PYPI}/32",
        f"{PYPI_NEXT}/32",
    ]
