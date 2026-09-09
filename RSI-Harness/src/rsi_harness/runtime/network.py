"""Phase-aware, pinned network policy compilation and lifecycle."""

from __future__ import annotations

import hashlib
import ipaddress
import shlex
import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlsplit

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerRef, ManagedNetwork, NetworkPolicy

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


class FirewallBackend(Protocol):
    """Host firewall operations; implementations must scope rules to a container."""

    def probe(self) -> bool: ...

    def install(self, rule_id: str, rules: NetworkRuleSet) -> None: ...

    def is_installed(self, rule_id: str, rules: NetworkRuleSet) -> bool: ...

    def exists(self, rule_id: str) -> bool: ...

    def remove(self, rule_id: str) -> None: ...


class NetworkPolicyMutationObserver(Protocol):
    """Durable notification boundary surrounding a firewall installation."""

    def policy_install_planned(self, rule_id: str) -> None: ...

    def policy_installed(self, lease: NetworkPolicyLease) -> None: ...


class FirewallRuleNotFound(Exception):
    """The requested recovery rule is already absent."""


class _UnavailableFirewall:
    def probe(self) -> bool:
        return False

    def install(self, rule_id: str, rules: NetworkRuleSet) -> None:
        raise AssertionError("firewall install called after failed probe")

    def is_installed(self, rule_id: str, rules: NetworkRuleSet) -> bool:
        return False

    def exists(self, rule_id: str) -> bool:
        return False

    def remove(self, rule_id: str) -> None:
        raise SetupError("no firewall backend was configured")


def _system_resolver(hostname: str) -> tuple[str, ...]:
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
        }
    except socket.gaierror as error:
        raise SetupError(
            f"cannot resolve network policy hostname {hostname!r}"
        ) from error
    return tuple(sorted(addresses))


@dataclass(frozen=True, slots=True)
class PinnedEndpoint:
    hostname: str
    port: int
    addresses: tuple[IPAddress, ...]


@dataclass(frozen=True, slots=True)
class NetworkRuleSet:
    container_id: str
    network_id: str
    network_name: str
    bridge_interface: str
    role: str
    mode: str
    exact_endpoints: tuple[PinnedEndpoint, ...]
    allow_networks: tuple[IPNetwork, ...]
    engine_destinations: tuple[IPAddress, ...]
    dns_resolvers: tuple[IPAddress, ...]


class DockerIptablesFirewallBackend:
    """Install exact container policy in both host forwarding and input paths."""

    _BLOCKED_V4 = (
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "224.0.0.0/4",
    )

    def __init__(
        self,
        client: object,
        *,
        runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self._client = client
        self._runner = runner or self._run

    @staticmethod
    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, capture_output=True, check=False, text=True)

    def probe(self) -> bool:
        for chain in ("DOCKER-USER", "INPUT"):
            try:
                result = self._runner(["iptables", "--wait", "-S", chain])
            except OSError:
                return False
            if result.returncode != 0:
                return False
        return True

    def install(self, rule_id: str, rules: NetworkRuleSet) -> None:
        forward_chain, input_chain = self._chains(rule_id)
        created: list[str] = []
        jumps: list[tuple[str, list[str]]] = []
        try:
            for chain in (forward_chain, input_chain):
                self._checked(["iptables", "--wait", "-N", chain])
                created.append(chain)
            for match, target in self._compiled_forward_rules(rules):
                self._append(forward_chain, list(match), target)
            for match, target in self._compiled_input_rules(rules):
                self._append(input_chain, list(match), target)
            for host_chain, policy_chain, kind in (
                ("DOCKER-USER", forward_chain, "forward"),
                ("INPUT", input_chain, "input"),
            ):
                jump = self._jump_match(
                    bridge_interface=rules.bridge_interface,
                    comment=f"{rule_id}:{kind}",
                    policy_chain=policy_chain,
                )
                self._checked(
                    ["iptables", "--wait", "-I", host_chain, "1", *jump]
                )
                jumps.append((host_chain, jump))
        except Exception as primary:
            rollback_errors: list[str] = []
            for host_chain, jump in reversed(jumps):
                self._rollback(
                    ["iptables", "--wait", "-D", host_chain, *jump],
                    rollback_errors,
                )
            for chain in reversed(created):
                self._rollback(
                    ["iptables", "--wait", "-F", chain], rollback_errors
                )
                self._rollback(
                    ["iptables", "--wait", "-X", chain], rollback_errors
                )
            try:
                residual = self.exists(rule_id)
            except Exception as error:
                residual = None
                rollback_errors.append(f"residual inspection: {error}")
            if residual is not False:
                detail = "; ".join(rollback_errors) or "partial rules remain"
                raise InfrastructureError(
                    "recovery_required: partial network policy "
                    f"{rule_id} rollback is unproven: {detail}"
                ) from primary
            raise

    def remove(self, rule_id: str) -> None:
        try:
            listing = self._all_rules()
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: network policy removal state is "
                f"uninspectable for {rule_id}: {error}"
            ) from error
        jump_lines = self._owned_jump_lines(listing, rule_id)
        chains = tuple(
            chain
            for chain in self._chains(rule_id)
            if self._chain_exists_in_listing(listing, chain)
        )
        if not jump_lines and not chains:
            raise FirewallRuleNotFound(rule_id)

        mutation_errors: list[str] = []
        for line in jump_lines:
            try:
                tokens = shlex.split(line)
                self._checked(["iptables", "--wait", "-D", *tokens[1:]])
            except Exception as error:
                mutation_errors.append(str(error))
        for chain in chains:
            for operation in ("-F", "-X"):
                try:
                    self._checked(["iptables", "--wait", operation, chain])
                except Exception as error:
                    mutation_errors.append(str(error))
        try:
            residual = self.exists(rule_id)
        except Exception as error:
            detail = "; ".join(mutation_errors)
            suffix = f"; mutation errors: {detail}" if detail else ""
            raise InfrastructureError(
                "recovery_required: network policy removal exact absence is "
                f"unproven for {rule_id}: {error}{suffix}"
            ) from error
        if residual:
            detail = "; ".join(mutation_errors) or "owned rules remain"
            raise InfrastructureError(
                "recovery_required: network policy removal exact absence is "
                f"unproven for {rule_id}: {detail}"
            )

    def exists(self, rule_id: str) -> bool:
        """Discover any partial jump/chain authority owned by a rule ID."""
        listing = self._all_rules()
        return bool(self._owned_jump_lines(listing, rule_id)) or any(
            self._chain_exists_in_listing(listing, chain)
            for chain in self._chains(rule_id)
        )

    def _all_rules(self) -> list[str]:
        try:
            return self._checked(
                ["iptables", "--wait", "-S"]
            ).stdout.splitlines()
        except Exception as error:
            raise SetupError(f"cannot inspect firewall rules: {error}") from error

    @staticmethod
    def _chain_exists_in_listing(listing: Sequence[str], chain: str) -> bool:
        for line in listing:
            tokens = shlex.split(line)
            if (
                len(tokens) >= 2
                and tokens[0] in {"-N", "-A"}
                and tokens[1] == chain
            ):
                return True
        return False

    def is_installed(self, rule_id: str, rules: NetworkRuleSet) -> bool:
        forward_chain, input_chain = self._chains(rule_id)
        for host_chain, policy_chain, kind, compiled in (
            (
                "DOCKER-USER",
                forward_chain,
                "forward",
                self._compiled_forward_rules(rules),
            ),
            ("INPUT", input_chain, "input", self._compiled_input_rules(rules)),
        ):
            listing = self._checked(
                ["iptables", "--wait", "-S", host_chain]
            ).stdout.splitlines()
            expected_jump = [
                "-A",
                host_chain,
                *self._jump_match(
                    bridge_interface=rules.bridge_interface,
                    comment=f"{rule_id}:{kind}",
                    policy_chain=policy_chain,
                ),
            ]
            host_rules = [
                shlex.split(line)
                for line in listing
                if shlex.split(line)[:1] not in (["-N"], ["-P"])
            ]
            if not host_rules or host_rules[0] != expected_jump:
                return False
            owned = self._owned_jump_lines(listing, rule_id)
            if [shlex.split(line) for line in owned] != [expected_jump]:
                return False
            chain_lines = self._checked(
                ["iptables", "--wait", "-S", policy_chain]
            ).stdout.splitlines()
            actual_rules = [shlex.split(line) for line in chain_lines]
            if actual_rules[:1] == [["-N", policy_chain]]:
                actual_rules = actual_rules[1:]
            expected_rules = [
                ["-A", policy_chain, *self._rule_tokens(match, target)]
                for match, target in compiled
            ]
            if actual_rules != expected_rules:
                return False
        return True

    def _compiled_forward_rules(
        self, rules: NetworkRuleSet
    ) -> tuple[tuple[tuple[str, ...], str], ...]:
        compiled = list(self._common_accept_rules(rules))
        for address in rules.engine_destinations:
            self._require_ipv4(address)
            compiled.append((("-d", f"{address}/32"), "REJECT"))
        compiled.extend((("-d", network), "REJECT") for network in self._BLOCKED_V4)
        if rules.mode == "allowlist":
            for network in rules.allow_networks:
                if network.version != 4:
                    raise SetupError("IPv6 allowlist requires an IPv6-safe backend")
                compiled.append((("-d", str(network)), "ACCEPT"))
        compiled.append(((), "ACCEPT" if rules.mode == "public" else "REJECT"))
        return tuple(compiled)

    def _compiled_input_rules(
        self, rules: NetworkRuleSet
    ) -> tuple[tuple[tuple[str, ...], str], ...]:
        # Host services are never public merely because container egress is public.
        return (*self._common_accept_rules(rules), ((), "REJECT"))

    def _common_accept_rules(
        self, rules: NetworkRuleSet
    ) -> tuple[tuple[tuple[str, ...], str], ...]:
        compiled: list[tuple[tuple[str, ...], str]] = [
            (("-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED"), "ACCEPT")
        ]
        for endpoint in rules.exact_endpoints:
            for address in endpoint.addresses:
                self._require_ipv4(address)
                compiled.append(
                    (
                        (
                            "-d",
                            f"{address}/32",
                            "-p",
                            "tcp",
                            "-m",
                            "tcp",
                            "--dport",
                            str(endpoint.port),
                        ),
                        "ACCEPT",
                    )
                )
        for resolver in rules.dns_resolvers:
            self._require_ipv4(resolver)
            for protocol in ("udp", "tcp"):
                compiled.append(
                    (
                        (
                            "-d",
                            f"{resolver}/32",
                            "-p",
                            protocol,
                            "-m",
                            protocol,
                            "--dport",
                            "53",
                        ),
                        "ACCEPT",
                    )
                )
        return tuple(compiled)

    @staticmethod
    def _jump_match(
        *,
        bridge_interface: str,
        comment: str,
        policy_chain: str,
    ) -> list[str]:
        return [
            "-i",
            bridge_interface,
            "-m",
            "comment",
            "--comment",
            comment,
            "-j",
            policy_chain,
        ]

    @staticmethod
    def _owned_jump_lines(listing: Sequence[str], rule_id: str) -> list[str]:
        expected = {f"{rule_id}:forward", f"{rule_id}:input"}
        owned: list[str] = []
        for line in listing:
            tokens = shlex.split(line)
            try:
                comment = tokens[tokens.index("--comment") + 1]
            except (ValueError, IndexError):
                continue
            if comment in expected:
                owned.append(line)
        return owned

    @staticmethod
    def _require_ipv4(address: IPAddress) -> None:
        if not isinstance(address, ipaddress.IPv4Address):
            raise SetupError("IPv6 endpoint requires an IPv6-safe firewall backend")

    def _append(self, chain: str, match: list[str], target: str) -> None:
        self._checked(
            ["iptables", "--wait", "-A", chain, *self._rule_tokens(match, target)]
        )

    @staticmethod
    def _rule_tokens(match: Sequence[str], target: str) -> list[str]:
        tokens = [*match, "-j", target]
        if target == "REJECT":
            tokens.extend(["--reject-with", "icmp-port-unreachable"])
        return tokens

    def _checked(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            result = self._runner(command)
        except OSError as error:
            raise SetupError(
                f"firewall command {command[0]!r} failed: {error}"
            ) from error
        if result.returncode != 0:
            detail = result.stderr.strip() or "command exited unsuccessfully"
            raise SetupError(
                f"firewall command {shlex.join(command)!r} failed: {detail}"
            )
        return result

    def _rollback(self, command: list[str], errors: list[str]) -> None:
        try:
            result = self._runner(command)
        except OSError as error:
            errors.append(f"{shlex.join(command)}: {error}")
            return
        if result.returncode != 0:
            detail = result.stderr.strip() or "command exited unsuccessfully"
            errors.append(f"{shlex.join(command)}: {detail}")

    @staticmethod
    def _chains(rule_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(rule_id.encode()).hexdigest()[:16].upper()
        return f"RSI_F_{digest}", f"RSI_I_{digest}"


@dataclass(frozen=True, slots=True)
class NetworkPolicyLease:
    """Persistable identity plus the pinned destinations active for one phase."""

    rule_id: str
    network_id: str
    network_name: str
    rules: NetworkRuleSet

    @property
    def internal_network(self) -> bool:
        """Whether Docker must create this phase bridge without external egress."""
        return self.rules.role == "judge" and self.rules.mode == "no-network"

    def allows_url(self, url: str) -> bool:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            return False
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        hostname = parsed.hostname.rstrip(".").lower()
        if hostname == "localhost":
            return True
        endpoint = next(
            (
                endpoint
                for endpoint in self.rules.exact_endpoints
                if endpoint.hostname == hostname and endpoint.port == port
            ),
            None,
        )
        if endpoint is not None:
            return True
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return False
        return self.allows_destination(str(address), port)

    def allows_destination(
        self,
        address: str,
        port: int,
        *,
        established: bool = False,
    ) -> bool:
        if established:
            return True
        ip = ipaddress.ip_address(address)
        if port == 53 and ip in self.rules.dns_resolvers:
            return True
        if any(
            ip in endpoint.addresses and port == endpoint.port
            for endpoint in self.rules.exact_endpoints
        ):
            return True
        if ip in self.rules.engine_destinations or _always_blocked(ip):
            return False
        if self.rules.mode == "public":
            return True
        if self.rules.mode == "allowlist":
            return any(ip in network for network in self.rules.allow_networks)
        return False


class NetworkPolicyEnforcer:
    """Resolve once, compile phase policy, and persist cleanup identifiers."""

    def __init__(
        self,
        *,
        run_id: str,
        resolver: Callable[[str], Sequence[str]] = _system_resolver,
        firewall: FirewallBackend | None = None,
        engine_destinations: Sequence[str] = (),
        dns_resolvers: Sequence[str] = (),
    ) -> None:
        self._run_id = run_id
        self._resolver = resolver
        self._firewall = firewall or _UnavailableFirewall()
        self._engine_destinations = tuple(
            ipaddress.ip_address(value) for value in engine_destinations
        )
        self._dns_resolvers = tuple(
            ipaddress.ip_address(value) for value in dns_resolvers
        )
        self._network_roles: dict[str, str] = {}
        self._issued_leases: dict[str, NetworkPolicyLease] = {}

    def plan(
        self,
        container: ContainerRef,
        policy: NetworkPolicy,
        *,
        network: ManagedNetwork,
        api_endpoints: Sequence[str | PinnedEndpoint] = (),
        control_endpoints: Sequence[str | PinnedEndpoint] = (),
    ) -> NetworkPolicyLease:
        """Compile a stable policy identity without mutating host authority."""
        role = container.role
        if network.run_id != self._run_id or network.role != role:
            raise SetupError("network identity is not bound to this run and role")
        requires_internal = role == "judge" and policy.mode == "no-network"
        if network.internal != requires_internal:
            if requires_internal:
                raise SetupError("Judge no-network policy requires an internal bridge")
            raise SetupError(
                f"{role.title()} {policy.mode} policy requires an egress-capable bridge"
            )
        existing_role = self._network_roles.get(network.network_id)
        if existing_role is not None and existing_role != role:
            raise SetupError("Work and Judge must not share a Docker network")
        if not self._firewall.probe():
            raise SetupError("firewall permission probe failed before phase start")

        endpoint_urls = tuple(api_endpoints)
        if role == "work":
            endpoint_urls += tuple(control_endpoints)
        endpoints = tuple(
            endpoint
            if isinstance(endpoint, PinnedEndpoint)
            else self._pin_endpoint(endpoint)
            for endpoint in endpoint_urls
        )
        allow_networks = self._pin_allowlist(policy.allowlist)
        rules = NetworkRuleSet(
            container_id=container.container_id,
            network_id=network.network_id,
            network_name=network.name,
            bridge_interface=managed_bridge_interface(network.name),
            role=role,
            mode=policy.mode,
            exact_endpoints=endpoints,
            allow_networks=allow_networks,
            engine_destinations=self._engine_destinations,
            dns_resolvers=self._dns_resolvers,
        )
        fingerprint = hashlib.sha256(
            repr((network.network_id, rules)).encode("utf-8")
        ).hexdigest()[:12]
        rule_id = f"rsi-{self._run_id}-{role}-{fingerprint}"
        return NetworkPolicyLease(
            rule_id=rule_id,
            network_id=network.network_id,
            network_name=network.name,
            rules=rules,
        )

    def apply(
        self,
        container: ContainerRef,
        policy: NetworkPolicy,
        *,
        network: ManagedNetwork,
        api_endpoints: Sequence[str | PinnedEndpoint] = (),
        control_endpoints: Sequence[str | PinnedEndpoint] = (),
        mutation_observer: NetworkPolicyMutationObserver | None = None,
        planned: NetworkPolicyLease | None = None,
    ) -> NetworkPolicyLease:
        supplied_plan = planned is not None
        lease = planned or self.plan(
            container,
            policy,
            network=network,
            api_endpoints=api_endpoints,
            control_endpoints=control_endpoints,
        )
        rules = lease.rules
        if (
            lease.network_id != network.network_id
            or lease.network_name != network.name
            or rules.container_id != container.container_id
            or rules.role != container.role
            or rules.mode != policy.mode
        ):
            raise SetupError("planned network policy does not match phase inputs")
        if supplied_plan and not self._firewall.probe():
            raise SetupError("firewall permission probe failed before phase start")
        rule_id = lease.rule_id
        if mutation_observer is not None:
            mutation_observer.policy_install_planned(rule_id)
        try:
            self._firewall.install(rule_id, rules)
        except Exception as error:
            recovery_detail: str | None = None
            if isinstance(error, InfrastructureError) and "recovery_required" in str(
                error
            ):
                recovery_detail = str(error)
            else:
                try:
                    residual = self._firewall.exists(rule_id)
                except Exception as inspect_error:
                    recovery_detail = (
                        "recovery_required: partial network policy "
                        f"{rule_id} state is uninspectable: {inspect_error}"
                    )
                else:
                    if residual:
                        recovery_detail = (
                            "recovery_required: partial network policy "
                            f"{rule_id} remains after failed installation"
                        )
            if recovery_detail is not None:
                self._issued_leases[rule_id] = lease
                raise InfrastructureError(recovery_detail) from error
            raise SetupError(
                f"failed to install network policy {rule_id}: {error}"
            ) from error
        self._network_roles[network.network_id] = container.role
        self._issued_leases[rule_id] = lease
        if mutation_observer is not None:
            try:
                mutation_observer.policy_installed(lease)
            except BaseException:
                self.cleanup(lease)
                raise
        return lease

    def pin_endpoints(self, endpoints: Sequence[str]) -> tuple[PinnedEndpoint, ...]:
        """Resolve endpoint authority without mutating Docker or the firewall."""

        return tuple(self._pin_endpoint(endpoint) for endpoint in endpoints)

    def attest(self, lease: NetworkPolicyLease) -> None:
        """Verify issuance and the exact firewall jump immediately before start."""
        if self._issued_leases.get(lease.rule_id) is not lease:
            raise SetupError("network policy lease is not authoritative for this run")
        try:
            installed = self._firewall.is_installed(lease.rule_id, lease.rules)
        except Exception as error:
            raise InfrastructureError(
                f"failed to attest network policy {lease.rule_id}: {error}"
            ) from error
        if not installed:
            raise SetupError(
                f"network policy {lease.rule_id} is no longer installed"
            )

    def cleanup(self, lease: NetworkPolicyLease) -> None:
        try:
            self._firewall.remove(lease.rule_id)
        except FirewallRuleNotFound:
            pass
        except Exception as error:
            raise InfrastructureError(
                f"failed to remove network policy {lease.rule_id}: {error}"
            ) from error
        self._issued_leases.pop(lease.rule_id, None)

    def _pin_endpoint(self, url: str) -> PinnedEndpoint:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise SetupError("invalid HTTP API endpoint")
        hostname = parsed.hostname.rstrip(".").lower()
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
        except ValueError as error:
            raise SetupError(
                f"invalid HTTP API endpoint port for hostname {hostname!r}"
            ) from error
        try:
            direct = ipaddress.ip_address(hostname)
            addresses = (direct,)
        except ValueError:
            raw_addresses = tuple(self._resolver(hostname))
            if not raw_addresses:
                raise SetupError(f"hostname {hostname!r} resolved to no addresses")
            try:
                addresses = tuple(
                    dict.fromkeys(
                        ipaddress.ip_address(value) for value in raw_addresses
                    )
                )
            except ValueError as error:
                raise SetupError(
                    f"hostname {hostname!r} resolved ambiguously: {error}"
                ) from error
        return PinnedEndpoint(hostname=hostname, port=port, addresses=addresses)

    def _pin_allowlist(self, entries: Sequence[str]) -> tuple[IPNetwork, ...]:
        networks: list[IPNetwork] = []
        for entry in entries:
            try:
                network = ipaddress.ip_network(entry, strict=False)
                candidates = (network,)
            except ValueError:
                raw_addresses = tuple(self._resolver(entry))
                if not raw_addresses:
                    raise SetupError(
                        f"allowlist hostname {entry!r} resolved to no addresses"
                    )
                try:
                    candidates = tuple(
                        ipaddress.ip_network(value, strict=False)
                        for value in raw_addresses
                    )
                except ValueError as error:
                    raise SetupError(
                        f"allowlist hostname {entry!r} resolved ambiguously: {error}"
                    ) from error
            networks.extend(
                network
                for network in candidates
                if not _always_blocked(network.network_address)
                and network.network_address not in self._engine_destinations
            )
        return tuple(dict.fromkeys(networks))


def _always_blocked(address: IPAddress) -> bool:
    return (
        address.is_private
        or address.is_link_local
        or address.is_loopback
        or address.is_multicast
        or address.is_unspecified
        or address == ipaddress.ip_address("169.254.169.254")
    )


def managed_bridge_interface(network_name: str) -> str:
    """Return Docker's deterministic, Linux-interface-safe managed bridge name."""
    return "rsi" + hashlib.sha256(network_name.encode()).hexdigest()[:12]


__all__ = [
    "NetworkPolicyEnforcer",
    "NetworkPolicyLease",
    "NetworkPolicyMutationObserver",
    "NetworkRuleSet",
    "PinnedEndpoint",
    "FirewallRuleNotFound",
    "DockerIptablesFirewallBackend",
    "managed_bridge_interface",
]
