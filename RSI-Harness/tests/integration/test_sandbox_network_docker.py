"""Real brokered bridges as a non-root user; only the iptables half is faked.

The firewall's real blocking and the intra-bridge accept need root and are
covered by the operator script, not here: an allowlist env is checked up to
its planned allow rules (R11 checks what they block).
"""

import ipaddress
import time
import uuid
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException, NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime import sandbox_env_contracts as env
from rsi_harness.runtime.sandbox_contracts import SandboxOwner
from rsi_harness.runtime.sandbox_network import (
    SandboxNetworkBackend,
    _resolve_ipv4,
    plan_builder_network,
    plan_env_network,
)
from tests.fakes import FakeFirewallBackend
from tests.runtime.test_sandbox_policy_v2 import make_builder_lease, make_env_lease
from tests.sandbox_helpers import make_env_spec

pytestmark = pytest.mark.integration

IMAGE = "busybox:1.37.0"
SERVER = "mkdir -p /www && echo ok > /www/index.html && exec httpd -f -p 8080 -h /www"


def bridge_exists(interface):
    return Path("/sys/class/net", interface).exists()


class OrderingFirewall(FakeFirewallBackend):
    """Fake iptables that records whether the host bridge existed at each call."""

    def __init__(self):
        super().__init__()
        self.interfaces = {}
        self.bridge_seen = []

    def install(self, rule_id, rules):
        self.interfaces[rule_id] = rules.bridge_interface
        self.bridge_seen.append(("install", bridge_exists(rules.bridge_interface)))
        super().install(rule_id, rules)

    def remove(self, rule_id):
        interface = self.interfaces.get(rule_id)
        if interface is not None:
            self.bridge_seen.append(("remove", bridge_exists(interface)))
        super().remove(rule_id)


@pytest.fixture
def docker_run():
    try:
        client = docker.from_env(timeout=30)
        client.ping()
        client.images.get(IMAGE)
    except (DockerException, OSError) as error:
        pytest.skip(f"Docker/{IMAGE} capability unavailable: {error}")
    run_id = f"m2-net-{uuid.uuid4().hex[:12]}"
    try:
        yield client, run_id
    finally:
        filters = {"label": f"rsi-harness.run-id={run_id}"}
        for container in client.containers.list(all=True, filters=filters):
            try:
                container.remove(force=True, v=True)
            except NotFound:
                pass
        for network in client.networks.list(filters=filters):
            try:
                network.remove()
            except NotFound:
                pass
        leftovers = (
            client.containers.list(all=True, filters=filters),
            client.networks.list(filters=filters),
        )
        client.close()
        assert leftovers == ([], [])


def owner(run_id):
    return SandboxOwner(
        run_id=run_id, task_id="sandbox-network", phase="judge", round_id="round-1"
    )


def env_plan(run_id, network="public", allowlist=()):
    raw = make_env_spec()
    raw["network"] = network
    if allowlist:
        raw["allowlist"] = list(allowlist)
    env_id = "e" + uuid.uuid4().hex
    plan = plan_env_network(owner(run_id), env_id, env.parse_env_spec(raw))
    assert plan is not None
    return plan


def start_service(client, plan, index, aliases, command, *, image=IMAGE):
    """A labelled runc child joined to the bridge with low-level aliases."""
    api = client.api
    created = api.create_container(
        image,
        command=["sh", "-c", command],
        name=env.env_container_name(plan.handle, index),
        labels={
            "rsi-harness.run-id": plan.owner.run_id,
            "rsi-harness.task-id": plan.owner.task_id,
            "rsi-harness.role": "sandbox-env",
            "rsi-harness.sandbox-env": plan.handle,
        },
        environment={"NVIDIA_VISIBLE_DEVICES": "void"},
        host_config=api.create_host_config(
            runtime="runc",
            network_mode=plan.name,
            cap_drop=["NET_RAW"],
            init=True,
            mem_limit=64 * 1024**2,
            pids_limit=64,
        ),
        networking_config=api.create_networking_config(
            {plan.name: api.create_endpoint_config(aliases=list(aliases))}
        ),
    )
    api.start(created["Id"])
    return client.containers.get(created["Id"])


def fetch(container, url):
    deadline = time.monotonic() + 10
    while True:
        result = container.exec_run(["wget", "-qO-", "-T", "2", url])
        if result.exit_code == 0 or time.monotonic() > deadline:
            return result
        time.sleep(0.2)


def assert_gone(client, plan, firewall):
    filters = {"label": f"rsi-harness.sandbox-env={plan.handle}"}
    assert client.networks.list(filters=filters) == []
    assert client.networks.list(filters={"name": plan.name}) == []
    assert not bridge_exists(plan.bridge_interface)
    assert firewall.installed == {}


@pytest.mark.parametrize("network", ("public", "none"))
def test_real_env_bridge_resolves_aliases_and_tears_down_in_reverse(
    docker_run, network
):
    client, run_id = docker_run
    plan = env_plan(run_id, network)
    firewall = OrderingFirewall()
    backend = SandboxNetworkBackend(client, firewall)

    network_id = backend.create(plan)
    services = []
    try:
        assert bridge_exists(plan.bridge_interface)
        attrs = client.networks.get(network_id).attrs
        assert attrs["Internal"] is (network == "none")
        assert attrs["EnableIPv6"] is False
        assert attrs["Labels"] == plan.labels
        assert attrs["Options"] == {
            "com.docker.network.bridge.name": plan.bridge_interface
        }
        services.append(start_service(client, plan, 0, ("kvstore",), SERVER))
        services.append(start_service(client, plan, 1, ("web",), "sleep 120"))
        backend.attest(plan, network_id, containers=[item.id for item in services])
        with pytest.raises(InfrastructureError, match="foreign endpoint"):
            backend.attest(plan, network_id, containers=[services[1].id])

        main = services[1]
        resolved = main.exec_run(["nslookup", "kvstore"])
        assert resolved.exit_code == 0, resolved.output
        services[0].reload()
        kv_address = services[0].attrs["NetworkSettings"]["Networks"][plan.name][
            "IPAddress"
        ]
        assert kv_address.encode() in resolved.output
        served = fetch(main, "http://kvstore:8080/")
        assert (served.exit_code, served.output) == (0, b"ok\n")
        if network == "none":
            # The internal bridge forwards no external names, only peers.
            assert main.exec_run(["nslookup", "example.com"]).exit_code != 0
        with pytest.raises(InfrastructureError, match="recovery_required"):
            backend.remove(plan, network_id)
        assert firewall.bridge_seen == [("install", False)]
        assert plan.rule_id in firewall.installed
    finally:
        for service in services:
            service.remove(force=True, v=True)

    backend.remove(plan, network_id)

    assert firewall.bridge_seen == [("install", False), ("remove", False)]
    assert_gone(client, plan, firewall)


def test_real_attest_failure_rolls_back_the_created_bridge(docker_run):
    client, run_id = docker_run
    plan = env_plan(run_id)
    firewall = OrderingFirewall()
    firewall.is_installed = lambda rule_id, rules: False

    with pytest.raises(SetupError, match="proven absent"):
        SandboxNetworkBackend(client, firewall).create(plan)

    assert firewall.bridge_seen == [("install", False), ("remove", False)]
    assert_gone(client, plan, firewall)


def test_real_recovery_finds_unjournaled_env_and_builder_bridges(docker_run):
    client, run_id = docker_run
    firewall = OrderingFirewall()
    env_net = env_plan(run_id)
    builder_id = "b" + uuid.uuid4().hex
    builder_net = plan_builder_network(owner(run_id), builder_id)
    backend = SandboxNetworkBackend(client, firewall)
    env_network_id = backend.create(env_net)
    builder_network_id = backend.create(builder_net)
    builder_attrs = client.networks.list(filters={"name": builder_net.name})[0].attrs
    assert builder_attrs["Internal"] is False
    assert builder_attrs["Labels"]["rsi-harness.role"] == "sandbox-builder-net"

    # A crash after bridge creation, before its ID reached the journal.
    env_lease = make_env_lease(
        env_net.handle,
        owner=env_net.owner,
        state="planned",
        network_name=env_net.name,
        network_id=None,
        rule_id=env_net.rule_id,
        services=tuple(
            service.model_copy(update={"state": "planned", "container_id": None})
            for service in make_env_lease(env_net.handle).services
        ),
        pending_mutation=True,
    )
    builder_lease = make_builder_lease(
        builder_id,
        owner=builder_net.owner,
        rule_id=builder_net.rule_id,
        state="planned",
        container_id=None,
        network_id=None,
        loop_device=None,
        state_fs="tmpfs",
    )
    recovered = SandboxNetworkBackend(client, firewall)
    # The pending env create is resolved because its bridge is found.
    assert recovered.recover(env_lease) == (env_network_id,)
    assert recovered.recover(builder_lease) == (builder_network_id,)

    assert_gone(client, env_net, firewall)
    assert client.networks.list(filters={"name": builder_net.name}) == []
    assert not bridge_exists(builder_net.bridge_interface)
    assert [seen for _, seen in firewall.bridge_seen] == [False] * 4


def test_real_allowlist_env_resolves_listed_names_through_the_embedded_dns(
    docker_run,
):
    """A public-shaped bridge (embedded DNS forwards from the host namespace)
    whose planned rule accepts exactly the resolved entries; a refresh
    replaces only the allow rules."""
    client, run_id = docker_run
    answers = {}

    def resolver(host):
        return answers.get(host) or _resolve_ipv4(host)

    addresses = _resolve_ipv4("example.com")
    if not addresses:
        pytest.skip("this host resolves no example.com")
    plan = env_plan(run_id, "allowlist", ("example.com:443", "1.1.1.1"))
    firewall = OrderingFirewall()
    backend = SandboxNetworkBackend(client, firewall, resolver=resolver)

    network_id = backend.create(plan)
    services = []
    try:
        assert client.networks.get(network_id).attrs["Internal"] is False
        rules = firewall.installed[plan.rule_id]
        assert rules.mode == "allowlist"
        accepted = {(str(network), port) for network, port in rules.allow_rules}
        assert ("1.1.1.1/32", None) in accepted
        assert {address for (address, port) in accepted if port == 443} & {
            f"{address}/32" for address in addresses
        }
        services.append(start_service(client, plan, 0, ("main",), "sleep 120"))
        backend.attest(plan, network_id, containers=[services[0].id])
        resolved = services[0].exec_run(["nslookup", "example.com"])
        assert resolved.exit_code == 0, resolved.output

        answers["example.com"] = ("93.184.215.14",)
        assert backend.refresh(plan) is True
        assert ("update", plan.rule_id) in firewall.events
        refreshed = firewall.installed[plan.rule_id]
        assert (ipaddress.IPv4Network("93.184.215.14/32"), 443) in (
            refreshed.allow_rules
        )
        assert refreshed.bridge_interface == rules.bridge_interface
        backend.attest(plan, network_id, containers=[services[0].id])
    finally:
        for service in services:
            service.remove(force=True, v=True)

    backend.remove(plan, network_id)

    assert_gone(client, plan, firewall)
