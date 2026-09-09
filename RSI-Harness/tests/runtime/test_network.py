from __future__ import annotations

import subprocess
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import docker as docker_sdk
import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerRef, ManagedNetwork, NetworkPolicy
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    NetworkPolicyEnforcer,
    managed_bridge_interface,
)
from tests.fakes import FakeDockerClient, FakeDockerContainer, FakeFirewallBackend


class StaticResolver:
    def __init__(self, addresses: dict[str, tuple[str, ...]]) -> None:
        self.addresses = addresses
        self.queries: list[str] = []

    def __call__(self, hostname: str) -> Sequence[str]:
        self.queries.append(hostname)
        return self.addresses[hostname]


def make_enforcer(*, probe_result: bool = True):
    resolver = StaticResolver(
        {
            "model.test": ("127.0.0.1",),
            "control.test": ("172.30.0.1",),
            "unrelated.test": ("127.0.0.1",),
            "packages.test": ("93.184.216.34",),
            "judge-model.test": ("198.51.100.8",),
        }
    )
    firewall = FakeFirewallBackend(probe_result=probe_result)
    enforcer = NetworkPolicyEnforcer(
        resolver=resolver,
        firewall=firewall,
        run_id="run-1",
        engine_destinations=("172.30.0.1",),
        dns_resolvers=("127.0.0.11",),
    )
    return enforcer, resolver, firewall


def managed(name: str, role: str, *, network_id: str | None = None):
    return ManagedNetwork(
        network_id=network_id or f"id-{name}",
        name=name,
        run_id="run-1",
        task_id="task-1",
        role=role,
        internal=role == "judge" and "no-network" in name,
    )


def test_work_no_network_reaches_exact_model_and_control_but_not_unrelated_peer():
    enforcer, resolver, firewall = make_enforcer()

    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="no-network"),
        network=managed("work-control-run-1", "work"),
        api_endpoints=("http://model.test:8010/v1",),
        control_endpoints=("http://control.test:8020/api/v1/submit",),
    )

    assert lease.allows_url("http://model.test:8010/v1/chat/completions")
    assert lease.allows_url("http://control.test:8020/api/v1/submit")
    assert not lease.allows_url("http://unrelated.test:8030/")
    assert not lease.allows_url("http://control.test:9999/")
    assert resolver.queries == ["model.test", "control.test"]
    assert firewall.events[:2] == [("probe", None), ("install", lease.rule_id)]
    assert lease.rule_id.startswith("rsi-run-1-work-")


def test_judge_policy_never_inherits_work_control_route_or_network():
    enforcer, _, _ = make_enforcer()
    work = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="no-network"),
        network=managed("work-control-run-1", "work"),
        api_endpoints=("http://model.test:8010/v1",),
        control_endpoints=("http://control.test:8020/",),
    )

    judge = enforcer.apply(
        ContainerRef(container_id="judge", role="judge"),
        NetworkPolicy(mode="allowlist", allowlist=("judge-model.test",)),
        network=managed("judge-round-1", "judge"),
        api_endpoints=("http://judge-model.test:8443/v1",),
        control_endpoints=("http://control.test:8020/",),
    )

    assert work.network_name != judge.network_name
    assert judge.allows_url("http://judge-model.test:8443/v1")
    assert not judge.allows_url("http://control.test:8020/")
    assert not judge.allows_destination("172.30.0.2", 8080)


def test_judge_no_network_requests_internal_bridge_but_egress_modes_do_not():
    enforcer, _, _ = make_enforcer()
    no_network = enforcer.apply(
        ContainerRef(container_id="judge-1", role="judge"),
        NetworkPolicy(mode="no-network"),
        network=managed("judge-no-network-round-1", "judge"),
    )
    public = enforcer.apply(
        ContainerRef(container_id="judge-2", role="judge"),
        NetworkPolicy(mode="public"),
        network=managed("judge-round-2", "judge"),
    )

    assert no_network.internal_network is True
    assert public.internal_network is False


@pytest.mark.parametrize(
    ("role", "mode", "network"),
    [
        (
            "judge",
            "no-network",
            replace(managed("judge-no-network", "judge"), internal=False),
        ),
        (
            "judge",
            "public",
            replace(managed("judge-public", "judge"), internal=True),
        ),
        (
            "work",
            "no-network",
            replace(managed("work-control", "work"), internal=True),
        ),
    ],
)
def test_policy_rejects_network_internal_mode_mismatch(role, mode, network):
    enforcer, _, firewall = make_enforcer()

    with pytest.raises(SetupError, match="internal|egress-capable"):
        enforcer.apply(
            ContainerRef(container_id=role, role=role),
            NetworkPolicy(mode=mode),
            network=network,
        )

    assert firewall.events == []


def test_work_and_judge_cannot_share_a_bridge_even_in_public_mode():
    enforcer, _, _ = make_enforcer()
    enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="public"),
        network=managed("shared", "work", network_id="shared-id"),
    )

    with pytest.raises(SetupError, match="must not share"):
        enforcer.apply(
            ContainerRef(container_id="judge", role="judge"),
            NetworkPolicy(mode="public"),
            network=managed("shared", "judge", network_id="shared-id"),
        )


def test_public_keeps_public_egress_but_blocks_private_metadata_and_engine():
    enforcer, _, _ = make_enforcer()
    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="public"),
        network=managed("work", "work"),
    )

    assert lease.allows_destination("8.8.8.8", 443)
    assert not lease.allows_destination("10.0.0.2", 443)
    assert not lease.allows_destination("169.254.169.254", 80)
    assert not lease.allows_destination("172.30.0.1", 2375)


def test_allowlist_pins_resolved_hostnames_and_cidrs_for_the_run():
    enforcer, resolver, _ = make_enforcer()
    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(
            mode="allowlist",
            allowlist=("packages.test", "8.8.8.0/24", "1.1.1.1"),
        ),
        network=managed("work", "work"),
    )

    resolver.addresses["packages.test"] = ("203.0.113.9",)
    assert lease.allows_destination("93.184.216.34", 443)
    assert not lease.allows_destination("203.0.113.9", 443)
    assert lease.allows_destination("8.8.8.8", 443)
    assert lease.allows_destination("1.1.1.1", 443)
    assert not lease.allows_destination("9.9.9.9", 443)


def test_firewall_permission_probe_fails_before_any_rules_are_installed():
    enforcer, _, firewall = make_enforcer(probe_result=False)

    with pytest.raises(SetupError, match="firewall permission"):
        enforcer.apply(
            ContainerRef(container_id="work", role="work"),
            NetworkPolicy(mode="no-network"),
            network=managed("work", "work"),
        )

    assert firewall.events == [("probe", None)]


def test_concrete_firewall_probe_requires_input_and_forward_authority():
    commands: list[tuple[str, ...]] = []

    def runner(command):
        commands.append(tuple(command))
        return subprocess.CompletedProcess(
            command,
            0 if command[-1] == "DOCKER-USER" else 4,
            "",
            "INPUT denied",
        )

    backend = DockerIptablesFirewallBackend(FakeDockerClient(), runner=runner)

    assert backend.probe() is False
    assert commands == [
        ("iptables", "--wait", "-S", "DOCKER-USER"),
        ("iptables", "--wait", "-S", "INPUT"),
    ]


def test_cleanup_uses_persisted_recovery_identifier():
    enforcer, _, firewall = make_enforcer()
    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="no-network"),
        network=managed("work", "work"),
    )

    enforcer.cleanup(lease)

    assert firewall.events[-1] == ("remove", lease.rule_id)
    assert lease.rule_id not in firewall.installed


def test_policy_identity_can_be_planned_before_firewall_mutation():
    firewall = FakeFirewallBackend()
    enforcer = NetworkPolicyEnforcer(run_id="run-1", firewall=firewall)
    container = ContainerRef(container_id="work-1", role="work")
    network = ManagedNetwork(
        network_id="network-1",
        name="rsi-run-1-task-work-phase",
        run_id="run-1",
        task_id="task",
        role="work",
        internal=False,
    )

    planned = enforcer.plan(
        container,
        NetworkPolicy(mode="public"),
        network=network,
        control_endpoints=("http://172.17.0.1:9020",),
    )

    assert not any(name == "install" for name, _ in firewall.events)
    installed = enforcer.apply(
        container,
        NetworkPolicy(mode="public"),
        network=network,
        control_endpoints=("http://172.17.0.1:9020",),
        planned=planned,
    )
    assert installed == planned
    assert ("install", planned.rule_id) in firewall.events

class RecordingFirewallRunner:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.cleanup_rule: str | None = None
        self.forward_chain: str | None = None
        self.input_chain: str | None = None
        self.bridge_interface: str | None = None
        self.wrong_interface = False
        self.chain_deletes = 0

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(tuple(command))
        if command[:5] == ["iptables", "--wait", "-I", "DOCKER-USER", "1"]:
            self.forward_chain = command[-1]
            self.bridge_interface = command[command.index("-i") + 1]
        if command[:5] == ["iptables", "--wait", "-I", "INPUT", "1"]:
            self.input_chain = command[-1]
            self.bridge_interface = command[command.index("-i") + 1]
        if command == ["iptables", "--wait", "-S"]:
            stdout = ""
            if self.cleanup_rule:
                interface = (
                    "rsi-does-not-match"
                    if self.wrong_interface
                    else self.bridge_interface
                )
                stdout = (
                    f"-N {self.forward_chain}\n"
                    f"-N {self.input_chain}\n"
                    f"-A DOCKER-USER -i {interface} "
                    "-m comment --comment "
                    f"{self.cleanup_rule}:forward -j {self.forward_chain}\n"
                    f"-A INPUT -i {interface} "
                    "-m comment --comment "
                    f"{self.cleanup_rule}:input -j {self.input_chain}\n"
                )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[:3] == ["iptables", "--wait", "-X"]:
            self.chain_deletes += 1
            if self.chain_deletes == 2:
                self.cleanup_rule = None
        if command[:4] == ["iptables", "--wait", "-S", "DOCKER-USER"]:
            stdout = ""
            if self.cleanup_rule:
                interface = (
                    "rsi-does-not-match"
                    if self.wrong_interface
                    else self.bridge_interface
                )
                stdout = (
                    f"-A DOCKER-USER -i {interface} "
                    "-m comment --comment "
                    f"{self.cleanup_rule}:forward -j {self.forward_chain}\n"
                )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[:4] == ["iptables", "--wait", "-S", "INPUT"]:
            stdout = ""
            if self.cleanup_rule:
                interface = (
                    "rsi-does-not-match"
                    if self.wrong_interface
                    else self.bridge_interface
                )
                stdout = (
                    f"-A INPUT -i {interface} "
                    "-m comment --comment "
                    f"{self.cleanup_rule}:input -j {self.input_chain}\n"
                )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        return subprocess.CompletedProcess(command, 0, "", "")


class InMemoryIptablesRunner:
    def __init__(self, *, canonicalize_rules: bool = False) -> None:
        self.chains: dict[str, list[list[str]]] = {
            "DOCKER-USER": [],
            "INPUT": [],
        }
        self.fail_input_jump = False
        self.fail_forward_jump_delete = False
        self.fail_managed_chain_query = False
        self.fail_chain_delete = False
        self.fail_global_list_after: int | None = None
        self.global_list_calls = 0
        self.canonicalize_rules = canonicalize_rules

    def _stored_rule(self, rule: list[str]) -> list[str]:
        if not self.canonicalize_rules:
            return rule
        stored = list(rule)
        if "--ctstate" in stored:
            index = stored.index("--ctstate") + 1
            if stored[index] == "ESTABLISHED,RELATED":
                stored[index] = "RELATED,ESTABLISHED"
        if "-p" in stored and "--dport" in stored:
            index = stored.index("-p") + 1
            protocol = stored[index]
            if ["-m", protocol] not in [stored[i : i + 2] for i in range(len(stored))]:
                stored[index + 1 : index + 1] = ["-m", protocol]
        if stored[-2:] == ["-j", "REJECT"]:
            stored.extend(["--reject-with", "icmp-port-unreachable"])
        return stored

    def __call__(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        operation = command[2]
        if operation == "-S" and len(command) == 3:
            self.global_list_calls += 1
            if (
                self.fail_global_list_after is not None
                and self.global_list_calls >= self.fail_global_list_after
            ):
                return subprocess.CompletedProcess(command, 4, "", "inspection denied")
            lines = []
            for name, rules in self.chains.items():
                if name not in {"DOCKER-USER", "INPUT"}:
                    lines.append(f"-N {name}")
                lines.extend(
                    subprocess.list2cmdline(["-A", name, *rule]) for rule in rules
                )
            stdout = "\n".join(lines)
            return subprocess.CompletedProcess(
                command, 0, stdout + ("\n" if stdout else ""), ""
            )
        chain = command[3]
        if operation == "-S":
            if chain not in self.chains:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            if (
                self.fail_managed_chain_query
                and chain not in {"DOCKER-USER", "INPUT"}
            ):
                return subprocess.CompletedProcess(command, 4, "", "inspection denied")
            lines = [] if chain in {"DOCKER-USER", "INPUT"} else [f"-N {chain}"]
            lines.extend(
                subprocess.list2cmdline(["-A", chain, *rule])
                for rule in self.chains[chain]
            )
            stdout = "\n".join(lines)
            return subprocess.CompletedProcess(
                command, 0, stdout + ("\n" if stdout else ""), ""
            )
        if operation == "-N":
            if chain in self.chains:
                return subprocess.CompletedProcess(command, 1, "", "exists")
            self.chains[chain] = []
        elif operation == "-A":
            self.chains[chain].append(self._stored_rule(command[4:]))
        elif operation == "-I":
            if self.fail_input_jump and chain == "INPUT":
                return subprocess.CompletedProcess(command, 4, "", "denied")
            self.chains[chain].insert(int(command[4]) - 1, command[5:])
        elif operation == "-D":
            if self.fail_forward_jump_delete and chain == "DOCKER-USER":
                return subprocess.CompletedProcess(command, 4, "", "denied")
            try:
                self.chains[chain].remove(command[4:])
            except (KeyError, ValueError):
                return subprocess.CompletedProcess(command, 1, "", "missing")
        elif operation == "-F":
            if chain not in self.chains:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            self.chains[chain].clear()
        elif operation == "-X":
            if self.fail_chain_delete:
                return subprocess.CompletedProcess(command, 4, "", "delete denied")
            if chain not in self.chains or self.chains[chain]:
                return subprocess.CompletedProcess(command, 1, "", "busy")
            del self.chains[chain]
        return subprocess.CompletedProcess(command, 0, "", "")


def exact_firewall_harness():
    client = FakeDockerClient()
    container = FakeDockerContainer("work")
    container.attrs["NetworkSettings"] = {
        "Networks": {"work-network": {"IPAddress": "172.28.0.2"}}
    }
    client.containers.by_id["work"] = container
    runner = InMemoryIptablesRunner()
    backend = DockerIptablesFirewallBackend(client, runner=runner)
    enforcer = NetworkPolicyEnforcer(
        run_id="run-1",
        firewall=backend,
        resolver=StaticResolver({"model.test": ("8.8.8.8",)}),
    )
    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="no-network"),
        network=managed("work-network", "work"),
        api_endpoints=("https://model.test/v1",),
    )
    return backend, enforcer, lease, runner


def test_attestation_matches_canonical_iptables_rule_rendering():
    client = FakeDockerClient()
    runner = InMemoryIptablesRunner(canonicalize_rules=True)
    backend = DockerIptablesFirewallBackend(client, runner=runner)
    enforcer = NetworkPolicyEnforcer(
        run_id="run-1",
        firewall=backend,
        resolver=StaticResolver({"model.test": ("8.8.8.8",)}),
        dns_resolvers=("1.1.1.1",),
    )

    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="no-network"),
        network=managed("work-network", "work"),
        api_endpoints=("https://model.test/v1",),
    )

    assert backend.is_installed(lease.rule_id, lease.rules) is True


def test_firewall_attestation_compares_both_exact_ordered_chains_and_jumps():
    backend, enforcer, lease, runner = exact_firewall_harness()
    enforcer.attest(lease)
    pristine = {
        name: [list(rule) for rule in rules]
        for name, rules in runner.chains.items()
    }
    forward_chain = next(
        rule[-1]
        for rule in runner.chains["DOCKER-USER"]
        if f"{lease.rule_id}:forward" in rule
    )
    input_chain = next(
        rule[-1]
        for rule in runner.chains["INPUT"]
        if f"{lease.rule_id}:input" in rule
    )

    mutations = (
        lambda: runner.chains[forward_chain].clear(),
        lambda: runner.chains[forward_chain].append(["-j", "ACCEPT"]),
        lambda: runner.chains[forward_chain].reverse(),
        lambda: runner.chains[input_chain].__setitem__(-1, ["-j", "ACCEPT"]),
        lambda: runner.chains["INPUT"][0].__setitem__(
            runner.chains["INPUT"][0].index("-i") + 1, "wrong-bridge"
        ),
        lambda: runner.chains["INPUT"].append(list(runner.chains["INPUT"][0])),
        lambda: runner.chains["DOCKER-USER"].insert(
            0, ["-s", "172.28.0.2/32", "-j", "ACCEPT"]
        ),
        lambda: runner.chains["INPUT"].insert(
            0, ["-s", "172.28.0.2/32", "-j", "ACCEPT"]
        ),
    )
    for mutate in mutations:
        runner.chains = {
            name: [list(rule) for rule in rules] for name, rules in pristine.items()
        }
        mutate()
        assert backend.is_installed(lease.rule_id, lease.rules) is False


def test_partial_input_install_rolls_back_both_jumps_and_policy_chains():
    client = FakeDockerClient()
    container = FakeDockerContainer("work")
    container.attrs["NetworkSettings"] = {
        "Networks": {"work-network": {"IPAddress": "172.28.0.2"}}
    }
    client.containers.by_id["work"] = container
    runner = InMemoryIptablesRunner()
    runner.fail_input_jump = True
    backend = DockerIptablesFirewallBackend(client, runner=runner)
    enforcer = NetworkPolicyEnforcer(run_id="run-1", firewall=backend)

    with pytest.raises(SetupError, match="install network policy"):
        enforcer.apply(
            ContainerRef(container_id="work", role="work"),
            NetworkPolicy(mode="no-network"),
            network=managed("work-network", "work"),
        )

    assert runner.chains == {"DOCKER-USER": [], "INPUT": []}


def test_firewall_policy_can_be_installed_before_stopped_container_has_an_ip():
    client = FakeDockerClient()
    container = FakeDockerContainer("work")
    container.attrs["NetworkSettings"] = {
        "Networks": {"work-network": {"IPAddress": ""}}
    }
    client.containers.by_id["work"] = container
    runner = InMemoryIptablesRunner()
    backend = DockerIptablesFirewallBackend(client, runner=runner)
    enforcer = NetworkPolicyEnforcer(run_id="run-1", firewall=backend)

    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="no-network"),
        network=managed("work-network", "work"),
    )

    for host_chain, suffix in (("DOCKER-USER", "forward"), ("INPUT", "input")):
        jump = next(
            rule
            for rule in runner.chains[host_chain]
            if f"{lease.rule_id}:{suffix}" in rule
        )
        assert jump[:2] == ["-i", lease.rules.bridge_interface]
        assert "-s" not in jump


def test_partial_install_rollback_failure_is_recovery_required_and_discoverable():
    client = FakeDockerClient()
    container = FakeDockerContainer("work")
    container.attrs["NetworkSettings"] = {
        "Networks": {"work-network": {"IPAddress": "172.28.0.2"}}
    }
    client.containers.by_id["work"] = container
    runner = InMemoryIptablesRunner()
    runner.fail_input_jump = True
    runner.fail_forward_jump_delete = True
    backend = DockerIptablesFirewallBackend(client, runner=runner)
    enforcer = NetworkPolicyEnforcer(run_id="run-1", firewall=backend)

    with pytest.raises(InfrastructureError, match="recovery_required") as raised:
        enforcer.apply(
            ContainerRef(container_id="work", role="work"),
            NetworkPolicy(mode="no-network"),
            network=managed("work-network", "work"),
        )

    rule_id = next(
        token.removesuffix(":forward")
        for rule in runner.chains["DOCKER-USER"]
        for token in rule
        if token.endswith(":forward")
    )
    assert rule_id in str(raised.value)
    assert backend.exists(rule_id) is True


def test_partial_policy_state_remains_discoverable_and_remove_cleans_everything():
    backend, _enforcer, lease, runner = exact_firewall_harness()
    input_chain = next(
        rule[-1]
        for rule in runner.chains["INPUT"]
        if f"{lease.rule_id}:input" in rule
    )
    runner.chains["INPUT"].clear()
    runner.chains[input_chain].clear()
    del runner.chains[input_chain]

    assert backend.exists(lease.rule_id) is True
    backend.remove(lease.rule_id)

    assert backend.exists(lease.rule_id) is False
    assert runner.chains == {"DOCKER-USER": [], "INPUT": []}


def test_exists_never_treats_managed_chain_query_failure_as_absence():
    backend, _enforcer, lease, runner = exact_firewall_harness()
    runner.chains["DOCKER-USER"].clear()
    runner.chains["INPUT"].clear()
    runner.fail_managed_chain_query = True
    runner.fail_global_list_after = 1

    with pytest.raises(SetupError, match="inspect.*firewall"):
        backend.exists(lease.rule_id)


@pytest.mark.parametrize("failure_kind", ("chain-delete", "final-inspection"))
def test_remove_requires_authoritative_exact_absence_proof(failure_kind):
    backend, _enforcer, lease, runner = exact_firewall_harness()
    if failure_kind == "chain-delete":
        runner.fail_chain_delete = True
    else:
        runner.fail_global_list_after = 2

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.remove(lease.rule_id)

    if failure_kind == "chain-delete":
        assert backend.exists(lease.rule_id) is True


def test_concrete_firewall_scopes_rules_to_phase_bridge_and_recovers_by_rule_id():
    client = FakeDockerClient()
    container = FakeDockerContainer("work")
    container.attrs["NetworkSettings"] = {
        "Networks": {"work-network": {"IPAddress": "172.28.0.2"}}
    }
    client.containers.by_id["work"] = container
    runner = RecordingFirewallRunner()
    backend = DockerIptablesFirewallBackend(client, runner=runner)
    enforcer = NetworkPolicyEnforcer(
        run_id="run-1",
        firewall=backend,
        resolver=StaticResolver({"model.test": ("8.8.8.8",)}),
    )
    network = managed("work-network", "work")

    lease = enforcer.apply(
        ContainerRef(container_id="work", role="work"),
        NetworkPolicy(mode="no-network"),
        network=network,
        api_endpoints=("https://model.test/v1",),
    )

    assert any(
        command[:5] == ("iptables", "--wait", "-I", "DOCKER-USER", "1")
        and "-i" in command
        and "rsi" in command[command.index("-i") + 1]
        and "-s" not in command
        and f"{lease.rule_id}:forward" in command
        for command in runner.commands
    )
    assert any(
        command[:5] == ("iptables", "--wait", "-I", "INPUT", "1")
        and "-i" in command
        and "rsi" in command[command.index("-i") + 1]
        and "-s" not in command
        and f"{lease.rule_id}:input" in command
        for command in runner.commands
    )
    assert any(
        command[:4] == ("iptables", "--wait", "-A", command[3])
        and "8.8.8.8/32" in command
        and "443" in command
        and "ACCEPT" in command
        for command in runner.commands
        if len(command) > 8 and command[2] == "-A"
    )
    assert any(
        "-j" in command and command[command.index("-j") + 1] == "REJECT"
        for command in runner.commands
    )

    runner.cleanup_rule = lease.rule_id
    with pytest.raises(SetupError, match="no longer installed"):
        enforcer.attest(lease)
    runner.wrong_interface = True
    with pytest.raises(SetupError, match="no longer installed"):
        enforcer.attest(lease)
    runner.wrong_interface = False
    runner.cleanup_rule = None
    with pytest.raises(SetupError, match="no longer installed"):
        enforcer.attest(lease)

    runner.cleanup_rule = lease.rule_id
    enforcer.cleanup(lease)
    assert any(command[2:4] == ("-D", "DOCKER-USER") for command in runner.commands)
    runner.cleanup_rule = None
    enforcer.cleanup(lease)


def test_firewall_install_errors_are_setup_errors_and_absent_cleanup_is_success():
    enforcer, _, firewall = make_enforcer()
    firewall.install = lambda *_args: (_ for _ in ()).throw(RuntimeError("denied"))
    with pytest.raises(SetupError, match="install network policy"):
        enforcer.apply(
            ContainerRef(container_id="work", role="work"),
            NetworkPolicy(mode="no-network"),
            network=managed("work", "work"),
        )


@pytest.mark.integration
def test_real_firewall_allows_exact_endpoint_and_blocks_unrelated_container():
    try:
        client = docker_sdk.from_env()
        client.ping()
    except Exception as error:
        pytest.skip(f"Docker daemon capability unavailable: {error}")
    backend = DockerIptablesFirewallBackend(client)
    if not backend.probe():
        pytest.skip("DOCKER-USER/INPUT firewall permission capability unavailable")

    suffix = uuid.uuid4().hex[:10]
    name = f"rsi-firewall-test-{suffix}"
    labels = {
        "rsi-harness.run-id": suffix,
        "rsi-harness.task-id": "firewall-test",
        "rsi-harness.role": "work",
    }
    network = client.networks.create(
        name,
        driver="bridge",
        labels=labels,
        options={"com.docker.network.bridge.name": managed_bridge_interface(name)},
    )
    containers = []
    lease = None
    try:
        for role in ("allowed", "blocked"):
            containers.append(
                client.containers.run(
                    "alpine:latest",
                    [
                        "nc",
                        "-lk",
                        "-p",
                        "8080",
                        "-e",
                        "/bin/true",
                    ],
                    detach=True,
                    network=name,
                    labels={**labels, "test-role": role},
                )
            )
        work = client.containers.create(
            "alpine:latest",
            ["sleep", "60"],
            network=name,
            labels=labels,
        )
        containers.append(work)
        for container in containers:
            container.reload()
        allowed_ip = containers[0].attrs["NetworkSettings"]["Networks"][name][
            "IPAddress"
        ]
        blocked_ip = containers[1].attrs["NetworkSettings"]["Networks"][name][
            "IPAddress"
        ]
        for peer in containers[:2]:
            for _attempt in range(50):
                readiness = peer.exec_run(
                    ["nc", "-z", "-w", "1", "127.0.0.1", "8080"]
                )
                if readiness.exit_code == 0:
                    break
                time.sleep(0.1)
            else:
                pytest.fail(
                    f"peer {peer.name} did not become ready before policy install"
                )
        managed_network = ManagedNetwork(
            network_id=network.id,
            name=name,
            run_id=suffix,
            task_id="firewall-test",
            role="work",
            internal=False,
        )
        lease = NetworkPolicyEnforcer(run_id=suffix, firewall=backend).apply(
            ContainerRef(container_id=work.id, role="work"),
            NetworkPolicy(mode="no-network"),
            network=managed_network,
            api_endpoints=(f"http://{allowed_ip}:8080",),
        )
        work.start()
        assert work.exec_run(["nc", "-z", "-w", "1", allowed_ip, "8080"])[0] == 0
        assert (
            work.exec_run(["nc", "-z", "-w", "1", blocked_ip, "8080"])[0]
            != 0
        )
    finally:
        if lease is not None:
            try:
                backend.remove(lease.rule_id)
            except Exception:
                pass
        for container in reversed(containers):
            try:
                container.remove(force=True)
            except Exception:
                pass
        try:
            network.remove()
        except Exception:
            pass


@pytest.mark.integration
def test_real_internal_bridge_reaches_host_before_policy_then_input_blocks_it():
    try:
        client = docker_sdk.from_env()
        client.ping()
        client.images.get("alpine:latest")
    except Exception as error:
        pytest.skip(f"Docker/Alpine capability unavailable: {error}")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"host-listener")

        def log_message(self, *_args):
            return

    listener = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    listener_thread = threading.Thread(target=listener.serve_forever, daemon=True)
    listener_thread.start()
    suffix = uuid.uuid4().hex[:10]
    name = f"rsi-input-test-{suffix}"
    labels = {
        "rsi-harness.run-id": suffix,
        "rsi-harness.task-id": "input-test",
        "rsi-harness.role": "judge",
    }
    network = client.networks.create(
        name,
        driver="bridge",
        internal=True,
        labels=labels,
        options={
            "com.docker.network.bridge.name": managed_bridge_interface(name)
        },
    )
    container = None
    lease = None
    backend = DockerIptablesFirewallBackend(client)
    try:
        network.reload()
        gateway = network.attrs["IPAM"]["Config"][0]["Gateway"]
        container = client.containers.run(
            "alpine:latest",
            ["sleep", "60"],
            detach=True,
            network=name,
            labels=labels,
            cap_drop=["NET_RAW"],
        )
        url = f"http://{gateway}:{listener.server_port}/"
        before = container.exec_run(["wget", "-T", "2", "-qO-", url])
        assert before.exit_code == 0 and before.output == b"host-listener"

        if not backend.probe():
            pytest.skip(
                "INPUT/DOCKER-USER installation capability unavailable; "
                "uncontrolled internal-bridge host reachability was proved"
            )
        managed_network = ManagedNetwork(
            network_id=network.id,
            name=name,
            run_id=suffix,
            task_id="input-test",
            role="judge",
            internal=True,
        )
        enforcer = NetworkPolicyEnforcer(
            run_id=suffix,
            firewall=backend,
            engine_destinations=(gateway,),
        )
        lease = enforcer.apply(
            ContainerRef(container_id=container.id, role="judge"),
            NetworkPolicy(mode="no-network"),
            network=managed_network,
        )
        after = container.exec_run(["wget", "-T", "1", "-qO-", url])
        assert after.exit_code != 0
    finally:
        if lease is not None:
            try:
                backend.remove(lease.rule_id)
            except Exception:
                pass
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
        try:
            network.remove()
        except Exception:
            pass
        listener.shutdown()
        listener.server_close()
        listener_thread.join(timeout=2)
