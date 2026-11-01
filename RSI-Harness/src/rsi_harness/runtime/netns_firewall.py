"""Per-container network policy for a Linux controller without host root.

A Docker-group operator cannot change the host's netfilter tables, but can
give a short-lived helper CAP_NET_ADMIN inside one container's own network
namespace. Each Work or Judge container therefore joins the namespace of a
holder (``pause``) container on its managed bridge. Before the holder's
dependent starts, a firewall helper installs the policy in that namespace
and its exact filter tables are attested. Work and Judge never hold
CAP_NET_ADMIN, so their root cannot change the rules, and the rules
disappear with the namespace: nothing is left on the host after a crash.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docker.errors import APIError, ImageNotFound, NotFound
from docker.types import LogConfig

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.gpu import (
    NVIDIA_VISIBLE_DEVICES_ENV,
    NVIDIA_VISIBLE_DEVICES_VOID,
)
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    FirewallRuleNotFound,
    NetworkRuleSet,
)

# Pinned multi-architecture indexes; they carry no Harness or task code.
FIREWALL_IMAGE = (
    "registry.k8s.io/build-image/distroless-iptables@sha256:"
    "1c208e5a94c09bfee5fff8ee8db32d6cdb012c527628e17b481ebc58b9f7b28d"
)
HOLDER_IMAGE = (
    "registry.k8s.io/pause@sha256:"
    "ee6521f290b2168b6e0935a181d4cff9be1ac3f505666ef0e3c98fae8199917a"
)
RULE_PREFIX = "rsn"
NETNS_LABEL = "rsi-harness.netns"
_ROLE_LABEL = "rsi-harness.role"
_RUN_LABEL = "rsi-harness.run-id"
_TASK_LABEL = "rsi-harness.task-id"
_HELPER_TIMEOUT_SECONDS = 60
_IPV6_SEPARATOR = "--- rsi-ipv6 ---"
_BUILTIN_CHAINS = ("INPUT", "FORWARD", "OUTPUT")


# Docker Engine 26: volume subpath mounts, which carry single files.
_MINIMUM_API_VERSION = (1, 45)


def container_netns_firewall_selected() -> bool:
    """A non-root Linux controller enforces policy in container namespaces."""
    return sys.platform.startswith("linux") and os.geteuid() != 0


def require_container_netns_support(client: Any) -> None:
    """Refuse a daemon too old for the non-root controller, before any work."""
    raw = str(client.api.api_version)
    try:
        version = tuple(int(part) for part in raw.split(".")[:2])
    except ValueError as error:
        raise SetupError(f"unrecognized Docker API version {raw!r}") from error
    if version < _MINIMUM_API_VERSION:
        raise SetupError(
            "running rsi-harness without root needs Docker Engine 26 or later "
            f"(API 1.45); this daemon speaks API {raw}. Upgrade Docker or run as "
            "root (sudo -E)."
        )


def netns_role(role: str) -> str:
    """The recovery role label of the holder and helpers serving ``role``."""
    return f"{role}-netns"


def parse_host_destinations(fib_trie: str) -> tuple[ipaddress.IPv4Network, ...]:
    """Every local and broadcast IPv4 destination of a host, loopback excepted.

    A container's own namespace cannot tell a host address from a routed
    destination, so these are rejected explicitly where the root backend's
    host INPUT chain would reject them.
    """
    found: list[ipaddress.IPv4Network] = []
    current: str | None = None
    for line in fib_trie.splitlines():
        stripped = line.strip()
        if stripped.startswith("|-- "):
            current = stripped[4:]
            continue
        fields = stripped.split()
        if current is None or len(fields) != 3 or not fields[0].startswith("/"):
            continue
        prefix = int(fields[0][1:])
        if fields[1:] == ["host", "LOCAL"]:
            found.append(ipaddress.IPv4Network(f"{current}/{prefix}", strict=False))
        elif fields[1:] == ["link", "BROADCAST"]:
            found.append(ipaddress.IPv4Network(f"{current}/32"))
    return tuple(dict.fromkeys(network for network in found if not network.is_loopback))


def ensure_image(client: Any, reference: str) -> None:
    try:
        client.images.get(reference)
    except ImageNotFound:
        try:
            client.images.pull(reference)
        except APIError as error:
            raise SetupError(
                f"cannot pull {reference}; pull or load it, then retry: {error}"
            ) from error


def create_holder(
    client: Any,
    *,
    name: str | None,
    labels: Mapping[str, str],
    network: str,
    extra_hosts: Sequence[str],
    dns: Sequence[str],
) -> Any:
    """Create, without starting, the namespace holder of one container."""
    ensure_image(client, HOLDER_IMAGE)
    options: dict[str, Any] = {
        "user": "65535:65535",
        "read_only": True,
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "environment": {NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID},
        "mem_limit": "16m",
        "pids_limit": 8,
        "nano_cpus": 100_000_000,
        "labels": dict(labels),
        "network": network,
        "detach": True,
    }
    if name is not None:
        options["name"] = name
    if extra_hosts:
        options["extra_hosts"] = list(extra_hosts)
    if dns:
        options["dns"] = list(dns)
    try:
        return client.containers.create(HOLDER_IMAGE, None, **options)
    except APIError as error:
        raise SetupError(
            f"failed to create network namespace holder: {error}"
        ) from error


def _parse_rule_id(rule_id: str) -> tuple[str, str]:
    head, role, _ = rule_id.rsplit("-", 2)
    if not head.startswith(f"{RULE_PREFIX}-") or role not in {"work", "judge"}:
        raise SetupError(f"not a container-namespace policy rule: {rule_id}")
    return head.removeprefix(f"{RULE_PREFIX}-"), role


def _chains(rule_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(rule_id.encode()).hexdigest()[:16].upper()
    return f"RSI_O_{digest}", f"RSI_N_{digest}"


def _listing(text: str) -> dict[str, Any]:
    """Policies, chains and per-chain rule order of one ``iptables -S`` table."""
    policies: dict[str, str] = {}
    chains: set[str] = set()
    rules: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        tokens = shlex.split(line)
        if tokens[0] == "-P" and len(tokens) == 3:
            policies[tokens[1]] = tokens[2]
        elif tokens[0] == "-N" and len(tokens) == 2:
            chains.add(tokens[1])
        elif tokens[0] == "-A" and len(tokens) >= 2:
            rules.setdefault(tokens[1], []).append(tokens[2:])
        else:
            raise SetupError(f"unexpected firewall listing line: {line!r}")
    return {"policies": policies, "chains": chains, "rules": rules}


class ContainerNetnsFirewall(DockerIptablesFirewallBackend):
    """Install each policy inside its container's own network namespace."""

    container_netns = True
    rule_prefix = RULE_PREFIX

    def __init__(
        self,
        client: Any,
        *,
        helper: Callable[[Any, Sequence[str]], subprocess.CompletedProcess[str]]
        | None = None,
        daemon_fib_trie: Callable[[], str] | None = None,
        local_fib_trie: Path = Path("/proc/net/fib_trie"),
    ) -> None:
        self._local_fib_trie = local_fib_trie
        super().__init__(client, runner=self._no_host_runner)
        self._helper = helper or self._run_helper
        self._daemon_fib_trie = daemon_fib_trie or self._read_daemon_fib_trie
        self._holders: dict[str, str] = {}
        self._images_ready = False
        self._host_destinations: tuple[ipaddress.IPv4Network, ...] | None = None

    @staticmethod
    def _no_host_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        raise SetupError("container-namespace policy never runs host iptables")

    def probe(self) -> bool:
        if not self._images_ready:
            try:
                ensure_image(self._client, FIREWALL_IMAGE)
                ensure_image(self._client, HOLDER_IMAGE)
            except SetupError:
                return False
            self._images_ready = True
        return True

    def host_destinations(self) -> tuple[ipaddress.IPv4Network, ...]:
        """The daemon host's own destinations, read once from its namespace.

        The controller must share that namespace: its submission endpoint and
        the host checks only hold there (a remote or containerized controller
        is refused, as the root backend's host firewall probe would be).
        """
        if self._host_destinations is None:
            daemon = parse_host_destinations(self._daemon_fib_trie())
            try:
                local = parse_host_destinations(self._local_fib_trie.read_text())
            except OSError as error:
                raise SetupError(f"cannot list host addresses: {error}") from error
            if not daemon or set(daemon) != set(local):
                raise SetupError(
                    "running rsi-harness without root needs the controller in the "
                    "Docker daemon host's network namespace (a local daemon, not "
                    "from inside a container); run it on that host or as root"
                )
            self._host_destinations = daemon
        return self._host_destinations

    def _read_daemon_fib_trie(self) -> str:
        ensure_image(self._client, FIREWALL_IMAGE)
        self._sweep_stale_host_probes()
        command = [
            "-c",
            'while IFS= read -r line; do printf "%s\\n" "$line"; done '
            "< /proc/net/fib_trie",
        ]
        result = self._run_firewall_image(
            command,
            network_mode="host",
            cap_add=[],
            labels={NETNS_LABEL: "host-probe"},
        )
        if result.returncode != 0:
            raise SetupError(
                f"cannot read the Docker host's addresses: {result.stderr.strip()}"
            )
        return result.stdout

    def _sweep_stale_host_probes(self) -> None:
        """A probe precedes any run lease, so recovery cannot find a leftover
        one; the next probe removes those older than any probe can live."""
        try:
            stale = self._client.containers.list(
                all=True,
                filters={
                    "label": f"{NETNS_LABEL}=host-probe",
                    "status": ["created", "exited", "dead"],
                },
            )
        except APIError:
            return
        horizon = time.time() - 2 * _HELPER_TIMEOUT_SECONDS
        for container in stale:
            created = str(container.attrs.get("Created", ""))
            try:
                when = datetime.fromisoformat(created[:26].rstrip("Z")).replace(
                    tzinfo=UTC
                )
            except ValueError:
                continue
            if when.timestamp() < horizon:
                try:
                    container.remove(force=True)
                except APIError:
                    pass

    def _managed_subnets(self, rules: NetworkRuleSet) -> list[str]:
        try:
            network = self._client.networks.get(rules.network_id)
            network.reload()
            config = (network.attrs.get("IPAM") or {}).get("Config") or []
            subnets = [str(ipaddress.IPv4Network(entry["Subnet"])) for entry in config]
        except (APIError, KeyError, TypeError, ValueError) as error:
            raise SetupError(
                f"cannot read the managed bridge subnet: {error}"
            ) from error
        if not subnets:
            raise SetupError("the managed bridge has no IPv4 subnet")
        return subnets

    def _rulesets(
        self, rule_id: str, rules: NetworkRuleSet
    ) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        if rules.allow_rules is not None or rules.intra_bridge_accept:
            raise SetupError(
                "brokered sandbox bridges need the host firewall (run as root)"
            )
        output_chain, input_chain = _chains(rule_id)
        common = list(self._common_accept_rules(rules))
        forward = list(self._compiled_forward_rules(rules, rule_id))
        if forward[: len(common)] != common:
            raise AssertionError("forward rules no longer start with the accepts")
        # After the pinned accepts (the submission endpoint is a host address)
        # and before the mode's verdict: the host itself, this bridge (its
        # gateway and broadcast) and the limited broadcast address.
        rejects = [
            *(str(network) for network in self.host_destinations()),
            *self._managed_subnets(rules),
            "255.255.255.255/32",
        ]
        output_v4 = [
            (("-o", "lo"), "ACCEPT"),
            *common,
            *(
                (("-d", destination), "REJECT")
                for destination in dict.fromkeys(rejects)
            ),
            *forward[len(common) :],
        ]
        input_v4 = [
            (("-i", "lo"), "ACCEPT"),
            (("-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED"), "ACCEPT"),
            ((), "REJECT"),
        ]
        v4 = {
            output_chain: [self._rule_tokens(m, t) for m, t in output_v4],
            input_chain: [self._rule_tokens(m, t) for m, t in input_v4],
        }
        # The managed bridges are IPv4-only: IPv6 leaves only by loopback.
        reject6 = ["-j", "REJECT", "--reject-with", "icmp6-port-unreachable"]
        v6 = {
            output_chain: [["-o", "lo", "-j", "ACCEPT"], reject6],
            input_chain: [["-i", "lo", "-j", "ACCEPT"], reject6],
        }

        def restore(chains: dict[str, list[list[str]]]) -> str:
            lines = ["*filter", *(f":{chain} - [0:0]" for chain in chains)]
            for chain, chain_rules in chains.items():
                lines.extend(f"-A {chain} {shlex.join(rule)}" for rule in chain_rules)
            lines.append(f"-I OUTPUT 1 -j {output_chain}")
            lines.append(f"-I INPUT 1 -j {input_chain}")
            lines.append("COMMIT")
            return "\n".join(lines) + "\n"

        def expected(chains: dict[str, list[list[str]]]) -> dict[str, Any]:
            return {
                "policies": dict.fromkeys(_BUILTIN_CHAINS, "ACCEPT"),
                "chains": set(chains),
                "rules": {
                    "OUTPUT": [["-j", output_chain]],
                    "INPUT": [["-j", input_chain]],
                    **chains,
                },
            }

        return restore(v4), restore(v6), expected(v4), expected(v6)

    def _holder_for(self, rules: NetworkRuleSet) -> Any:
        """The exact holder whose namespace ``rules.container_id`` joined."""
        try:
            container = self._client.containers.get(rules.container_id)
            container.reload()
            mode = container.attrs["HostConfig"]["NetworkMode"]
            if not isinstance(mode, str) or not mode.startswith("container:"):
                raise SetupError("container does not use a namespace holder")
            holder = self._client.containers.get(mode.removeprefix("container:"))
            holder.reload()
        except NotFound as error:
            raise SetupError("network namespace holder is missing") from error
        labels = container.attrs.get("Config", {}).get("Labels") or {}
        holder_labels = holder.attrs.get("Config", {}).get("Labels") or {}
        networks = holder.attrs.get("NetworkSettings", {}).get("Networks") or {}
        attached = networks.get(rules.network_name) or {}
        # Docker fills in the endpoint's network ID only once the holder runs;
        # before that the Harness-created network's unique name binds it.
        running = bool(holder.attrs.get("State", {}).get("Running"))
        if (
            holder_labels.get(NETNS_LABEL) != "holder"
            or holder_labels.get(_ROLE_LABEL) != netns_role(rules.role)
            or labels.get(_ROLE_LABEL) != rules.role
            or any(
                holder_labels.get(key) != labels.get(key) or key not in labels
                for key in (_RUN_LABEL, _TASK_LABEL)
            )
            or holder.attrs.get("Config", {}).get("Image") != HOLDER_IMAGE
            or set(networks) != {rules.network_name}
            or (running and attached.get("NetworkID") != rules.network_id)
        ):
            raise SetupError("network namespace holder authority differs")
        return holder

    def _run_helper(
        self, holder: Any, arguments: Sequence[str]
    ) -> subprocess.CompletedProcess[str]:
        labels = holder.attrs.get("Config", {}).get("Labels") or {}
        return self._run_firewall_image(
            list(arguments),
            network_mode=f"container:{holder.id}",
            cap_add=["NET_ADMIN"],
            labels={key: labels[key] for key in (_RUN_LABEL, _TASK_LABEL, _ROLE_LABEL)}
            | {NETNS_LABEL: "firewall"},
        )

    def _run_firewall_image(
        self,
        command: list[str],
        *,
        network_mode: str,
        cap_add: list[str],
        labels: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        options: dict[str, Any] = {
            "entrypoint": "/bin/sh",
            "user": "0:0",
            "network_mode": network_mode,
            "read_only": True,
            "tmpfs": {"/run": "rw,noexec,nosuid,size=1m"},
            "cap_drop": ["ALL"],
            "cap_add": cap_add,
            "security_opt": ["no-new-privileges:true"],
            "environment": {NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID},
            "mem_limit": "64m",
            "pids_limit": 32,
            "nano_cpus": 500_000_000,
            "log_config": LogConfig(type=LogConfig.types.JSON),
            "labels": labels,
            "detach": True,
        }
        helper = self._client.containers.create(FIREWALL_IMAGE, command, **options)
        try:
            helper.start()
            status = helper.wait(timeout=_HELPER_TIMEOUT_SECONDS)
            return subprocess.CompletedProcess(
                command,
                int(status["StatusCode"]),
                helper.logs(stdout=True, stderr=False).decode("utf-8", "replace"),
                helper.logs(stdout=False, stderr=True).decode("utf-8", "replace"),
            )
        finally:
            try:
                helper.remove(force=True)
            except APIError:
                pass  # labeled: recovery (or the next host probe) removes it

    def _helper_checked(
        self, holder: Any, script: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = self._helper(holder, ["-c", script, "rsi-firewall", *arguments])
        except (APIError, OSError) as error:
            raise SetupError(f"container firewall helper failed: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or "helper exited unsuccessfully"
            raise SetupError(f"container firewall helper failed: {detail}")
        return result

    def _tables(self, holder: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        result = self._helper_checked(
            holder,
            f"iptables-nft -S && echo '{_IPV6_SEPARATOR}' && ip6tables-nft -S",
        )
        v4, separator, v6 = result.stdout.partition(f"{_IPV6_SEPARATOR}\n")
        if not separator:
            raise SetupError("container firewall listing is incomplete")
        return _listing(v4), _listing(v6)

    def install(self, rule_id: str, rules: NetworkRuleSet) -> None:
        restore_v4, restore_v6, _, _ = self._rulesets(rule_id, rules)
        holder = self._holder_for(rules)
        # Known before it starts: a failed start then reads as "no rule".
        self._holders[rule_id] = holder.id
        try:
            if not holder.attrs.get("State", {}).get("Running"):
                holder.start()
        except APIError as error:
            raise SetupError(
                f"failed to start network namespace holder: {error}"
            ) from error
        holder = self._holder_for(rules)
        self._helper_checked(
            holder,
            'printf "%s" "$1" | iptables-nft-restore --noflush -w 5 && '
            'printf "%s" "$2" | ip6tables-nft-restore --noflush -w 5',
            restore_v4,
            restore_v6,
        )
        if not self.is_installed(rule_id, rules):
            raise SetupError(f"container network policy {rule_id} did not attest")

    def is_installed(self, rule_id: str, rules: NetworkRuleSet) -> bool:
        _, _, expected_v4, expected_v6 = self._rulesets(rule_id, rules)
        holder = self._holder_for(rules)
        if self._holders.get(rule_id) != holder.id or not holder.attrs.get(
            "State", {}
        ).get("Running"):
            return False
        # The whole filter table: no rule but this policy's may exist there.
        return self._tables(holder) == (expected_v4, expected_v6)

    def _holder_ids(self, rule_id: str) -> tuple[str, ...]:
        known = self._holders.get(rule_id)
        if known is not None:
            return (known,)
        # A later process (recovery) knows only the rule: every holder and
        # helper of that run and role is torn down with it.
        run_id, role = _parse_rule_id(rule_id)
        found = self._client.containers.list(
            all=True,
            filters={
                "label": [f"{_RUN_LABEL}={run_id}", f"{_ROLE_LABEL}={netns_role(role)}"]
            },
        )
        return tuple(item.id for item in found)

    def _present(self, container_ids: Sequence[str]) -> tuple[str, ...]:
        present: list[str] = []
        for container_id in container_ids:
            try:
                self._client.containers.get(container_id)
            except NotFound:
                continue
            present.append(container_id)
        return tuple(present)

    def exists(self, rule_id: str) -> bool:
        known = self._holders.get(rule_id)
        if known is None:
            # Only a running holder has a namespace that can carry a rule.
            return any(
                self._client.containers.get(container_id)
                .attrs.get("State", {})
                .get("Running")
                for container_id in self._present(self._holder_ids(rule_id))
            )
        try:
            holder = self._client.containers.get(known)
            holder.reload()
        except NotFound:
            return False
        if not holder.attrs.get("State", {}).get("Running"):
            return False  # no namespace, so no rule
        chains = set(_chains(rule_id))
        return any(chains & table["chains"] for table in self._tables(holder))

    def remove(self, rule_id: str) -> None:
        # A holder serves exactly one container; removing it removes the
        # policy with the namespace (its dependent, if any, keeps no route).
        targets = self._present(self._holder_ids(rule_id))
        if not targets:
            self._holders.pop(rule_id, None)
            raise FirewallRuleNotFound(rule_id)
        errors: list[str] = []
        for container_id in targets:
            try:
                self._client.containers.get(container_id).remove(force=True)
            except NotFound:
                continue
            except APIError as error:
                errors.append(str(error))
        if self._present(targets):
            detail = "; ".join(errors) or "holder remains"
            raise InfrastructureError(
                "recovery_required: container network policy removal is "
                f"unproven for {rule_id}: {detail}"
            )
        self._holders.pop(rule_id, None)

    def update(self, rule_id: str, rules: NetworkRuleSet) -> None:
        raise SetupError(
            "brokered sandbox bridges need the host firewall (run as root)"
        )


class RecoveryFirewall:
    """Route each durable rule to the backend that installed it."""

    def __init__(self, host: Any, netns: Any) -> None:
        self._host = host
        self._netns = netns

    def _backend(self, rule_id: str) -> Any:
        return self._netns if rule_id.startswith(f"{RULE_PREFIX}-") else self._host

    def exists(self, rule_id: str) -> bool:
        return bool(self._backend(rule_id).exists(rule_id))

    def remove(self, rule_id: str) -> None:
        self._backend(rule_id).remove(rule_id)


__all__ = [
    "ContainerNetnsFirewall",
    "FIREWALL_IMAGE",
    "HOLDER_IMAGE",
    "NETNS_LABEL",
    "RecoveryFirewall",
    "container_netns_firewall_selected",
    "create_holder",
    "parse_host_destinations",
    "require_container_netns_support",
    "netns_role",
]
