"""Brokered env and builder bridges: deterministic plan, rule first, exact attest.

Every name derives from the owner and the env or builder handle, so recovery
rebuilds the network cleanup plan from the lease alone. The order is fixed
(S5): firewall rule, then bridge, then containers; teardown runs in reverse,
so a rule is never removed while its bridge may still exist. Containers are
not managed here.

An allowlist env is a public bridge whose rule ends in REJECT, with one
allow chain of ACCEPTs ahead of the private-range rejects. The broker
resolves listed hostnames when the env is created and again on every
refresh turn, and replaces that chain in place (no new rule or chain, so
nothing new to journal: the rule ID already covers it). DNS is Docker's
embedded resolver, which forwards from the host namespace (spec S5,
VERIFIED); the env itself reaches no resolver.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, replace
from typing import Any, NoReturn

from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    FirewallBackend,
    FirewallRuleNotFound,
    NetworkRuleSet,
    managed_bridge_interface,
)
from rsi_harness.runtime.sandbox_contracts import (
    AllowEntry,
    EnvNetwork,
    SandboxOwner,
    parse_allow_entry,
)
from rsi_harness.runtime.sandbox_env_contracts import (
    BUILDER_NETWORK_ROLE,
    ENV_NETWORK_ROLE,
    BuilderLease,
    EnvSpec,
    SandboxEnvLease,
    builder_network_name,
    builder_rule_id,
    env_network_name,
    env_rule_id,
    sandbox_object_labels,
)

BRIDGE_NAME_OPTION = "com.docker.network.bridge.name"
_HANDLE = re.compile(r"^[eb][0-9a-f]{32}$")
# Public env and builder chains end in ACCEPT, which skips Docker's own
# inter-bridge isolation; the private-range rejects then separate a bridge
# from its siblings and the parents only if its subnet lies inside them.
_REJECTED_V4 = tuple(
    ipaddress.ip_network(value) for value in DockerIptablesFirewallBackend._BLOCKED_V4
)
# Never reachable through an allowlist unless inside an operator-approved
# private CIDR: the rejected ranges plus "this network" and class E.
_RESTRICTED_V4 = (
    *_REJECTED_V4,
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("240.0.0.0/4"),
)
# Addresses kept per hostname across refreshes (newest first): round-robin
# DNS answers differ between the broker's lookup and the env's.
MAX_ADDRESSES_PER_HOST = 16
AllowRule = tuple[ipaddress.IPv4Network, int | None]


def allow_destination(
    network: ipaddress.IPv4Network, private_cidrs: Sequence[str]
) -> bool:
    """Whether an allowlist may accept ``network``: public space only, or
    wholly inside one of the operator's ``private_cidrs``."""
    if any(network.subnet_of(ipaddress.IPv4Network(cidr)) for cidr in private_cidrs):
        return True
    return not any(network.overlaps(blocked) for blocked in _RESTRICTED_V4)


def _resolve_ipv4(hostname: str) -> tuple[str, ...]:
    """The host resolver's IPv4 answers (what dockerd's embedded DNS asks)."""
    try:
        found = socket.getaddrinfo(
            hostname, None, family=socket.AF_INET, type=socket.SOCK_STREAM
        )
    except (socket.gaierror, UnicodeError):
        return ()
    return tuple(dict.fromkeys(item[4][0] for item in found))


@dataclass(frozen=True, slots=True)
class SandboxNetworkPlan:
    """One brokered bridge and its rule; everything else is derived.

    ``handle`` is an env (``e…``) or builder (``b…``) ID. ``network`` is the
    env's mode; a builder bridge is always public. An allowlist env carries
    its normalized entries and the operator's private CIDRs; a plan rebuilt
    from a lease for removal has neither (removal needs only the names).
    """

    owner: SandboxOwner
    handle: str
    network: EnvNetwork
    allowlist: tuple[str, ...] = ()
    private_cidrs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if _HANDLE.fullmatch(self.handle) is None:
            raise ValueError("sandbox network requires an env or builder handle")
        if self.network not in ("public", "none", "allowlist"):
            raise ValueError("sandbox network mode must be public, none or allowlist")
        if not self.is_env and self.network != "public":
            raise ValueError("a builder bridge is always public")
        if self.network != "allowlist" and (self.allowlist or self.private_cidrs):
            raise ValueError("allowlist entries require network allowlist")

    @property
    def entries(self) -> tuple[AllowEntry, ...]:
        return tuple(parse_allow_entry(entry) for entry in self.allowlist)

    @property
    def is_env(self) -> bool:
        return self.handle.startswith("e")

    @property
    def role(self) -> str:
        return ENV_NETWORK_ROLE if self.is_env else BUILDER_NETWORK_ROLE

    @property
    def name(self) -> str:
        if self.is_env:
            return env_network_name(self.handle)
        return builder_network_name(self.handle)

    @property
    def rule_id(self) -> str:
        if self.is_env:
            return env_rule_id(self.owner.run_id, self.handle)
        return builder_rule_id(self.owner.run_id, self.handle)

    @property
    def bridge_interface(self) -> str:
        return managed_bridge_interface(self.name)

    @property
    def internal(self) -> bool:
        # An internal bridge also stops embedded DNS forwarding external names.
        return self.network == "none"

    @property
    def mode(self) -> str:
        return {"public": "public", "none": "no-network"}.get(self.network, "allowlist")

    @property
    def intra_bridge_accept(self) -> bool:
        # A builder is alone on its bridge; env services must reach each other.
        return self.is_env

    @property
    def labels(self) -> dict[str, str]:
        handle_label = "sandbox-env" if self.is_env else "sandbox-builder"
        return sandbox_object_labels(self.owner, self.role, {handle_label: self.handle})


def _require_handle(handle: str, kind: str) -> None:
    if _HANDLE.fullmatch(handle) is None or handle[0] != kind:
        raise ValueError(f"sandbox network requires a {kind}<32hex> handle")


def plan_env_network(
    owner: SandboxOwner,
    env_id: str,
    spec: EnvSpec,
    *,
    private_cidrs: Sequence[str] = (),
) -> SandboxNetworkPlan | None:
    """Public: firewalled public bridge. Allowlist: the same bridge, its rule
    ending in REJECT after the allow chain. None: internal bridge, except
    that a single service gets no bridge and no rule at all (network_mode
    none). ``private_cidrs`` is the operator's allowlist grant."""
    _require_handle(env_id, "e")
    if spec.network == "none" and len(spec.services) == 1:
        return None
    if spec.network != "allowlist":
        return SandboxNetworkPlan(owner=owner, handle=env_id, network=spec.network)
    return SandboxNetworkPlan(
        owner=owner,
        handle=env_id,
        network="allowlist",
        allowlist=spec.allowlist,
        private_cidrs=tuple(private_cidrs),
    )


def env_lease_network_plan(lease: SandboxEnvLease) -> SandboxNetworkPlan | None:
    """The plan an env lease journaled; None when it planned no bridge."""
    if lease.network_name is None:
        return None
    return SandboxNetworkPlan(
        owner=lease.owner, handle=lease.env_id, network=lease.network_mode
    )


def plan_builder_network(owner: SandboxOwner, builder_id: str) -> SandboxNetworkPlan:
    """BuildKit pulls bases through the builder's bridge even for a
    ``network: none`` build, so it always gets the plain public rule set."""
    _require_handle(builder_id, "b")
    return SandboxNetworkPlan(owner=owner, handle=builder_id, network="public")


def _daemon_answered(error: Exception) -> bool:
    # Only an Engine HTTP response ends a create; after a timeout or a lost
    # connection dockerd may still finish making the bridge.
    return isinstance(error, APIError) and error.status_code is not None


def _subnets_rejected(config: Any) -> bool:
    if not isinstance(config, list) or not config:
        return False
    for entry in config:
        try:
            subnet = ipaddress.ip_network(entry["Subnet"])
        except (KeyError, TypeError, ValueError):
            return False
        if subnet.version != 4 or not any(
            subnet.subnet_of(rejected) for rejected in _REJECTED_V4
        ):
            return False
    return True


class SandboxNetworkBackend:
    """Create, attest and remove brokered bridges and their firewall rules.

    As for parent networks, a failed mutation that was rolled back and proven
    absent is a SetupError; unproven absence, an unanswered create and any
    uninspectable Engine state are an InfrastructureError marked
    ``recovery_required``.
    """

    def __init__(
        self,
        client: Any,
        firewall: FirewallBackend,
        *,
        engine_destinations: Sequence[str] = (),
        resolver: Callable[[str], Sequence[str]] = _resolve_ipv4,
    ) -> None:
        self._client = client
        self._firewall = firewall
        self._engine_destinations = tuple(
            ipaddress.ip_address(value) for value in engine_destinations
        )
        self._resolver = resolver
        # Allowlist envs: rule ID -> hostname -> addresses, as installed.
        # The lock orders refresh against create, attest and removal.
        self._resolved: dict[str, dict[str, tuple[str, ...]]] = {}
        self._allow_lock = threading.RLock()

    def rules(self, plan: SandboxNetworkPlan) -> NetworkRuleSet:
        # Rules precede the bridge, so its deterministic name stands in for
        # the Docker ID; the firewall matches only the bridge interface.
        allow_rules = None
        if plan.network == "allowlist":
            with self._allow_lock:
                resolved = self._resolved.get(plan.rule_id, {})
            allow_rules = self._allow_rules(plan, resolved)[0]
        return NetworkRuleSet(
            container_id=plan.handle,
            network_id=plan.name,
            network_name=plan.name,
            bridge_interface=plan.bridge_interface,
            role=plan.role,
            mode=plan.mode,
            exact_endpoints=(),
            allow_networks=(),
            engine_destinations=self._engine_destinations,
            dns_resolvers=(),
            intra_bridge_accept=plan.intra_bridge_accept,
            allow_rules=allow_rules,
        )

    # -- allowlist ---------------------------------------------------------------

    def _lookup(
        self, plan: SandboxNetworkPlan, previous: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        """Resolve every hostname entry; a failed lookup keeps the addresses
        it had, a new answer is merged in front of them (bounded)."""
        resolved: dict[str, tuple[str, ...]] = {}
        for entry in plan.entries:
            if entry.network is not None or entry.host in resolved:
                continue
            try:
                answer = tuple(self._resolver(entry.host))
            except Exception:
                answer = ()
            merged = tuple(dict.fromkeys((*answer, *previous.get(entry.host, ()))))
            resolved[entry.host] = merged[:MAX_ADDRESSES_PER_HOST]
        return resolved

    def _allow_rules(
        self, plan: SandboxNetworkPlan, resolved: dict[str, tuple[str, ...]]
    ) -> tuple[tuple[AllowRule, ...], tuple[str, ...]]:
        """The sorted accepts of an allowlist plan and a note per entry that
        reaches nothing (unresolved, or only blocked addresses)."""
        rules: dict[AllowRule, None] = {}
        notes: list[str] = []
        engine = set(self._engine_destinations)
        for entry in plan.entries:
            if entry.network is not None:
                candidates = (entry.network,)
            else:
                candidates = []
                for value in resolved.get(entry.host, ()):
                    try:
                        address = ipaddress.ip_address(value)
                    except ValueError:
                        continue
                    if isinstance(address, ipaddress.IPv4Address):
                        candidates.append(ipaddress.IPv4Network(address))
            allowed = [
                network
                for network in candidates
                if allow_destination(network, plan.private_cidrs)
                and not any(address in network for address in engine)
            ]
            for network in allowed:
                rules[(network, entry.port)] = None
            if not allowed:
                reason = (
                    "reaches only blocked addresses"
                    if candidates
                    else "resolves to no address"
                )
                notes.append(
                    f"allowlist entry {entry.text} {reason}; it allows nothing"
                )
        ordered = sorted(
            rules,
            key=lambda rule: (
                int(rule[0].network_address),
                rule[0].prefixlen,
                rule[1] or 0,
            ),
        )
        return tuple(ordered), tuple(notes)

    def allowlist_notes(self, plan: SandboxNetworkPlan) -> tuple[str, ...]:
        """What the installed allowlist of ``plan`` cannot reach."""
        if plan.network != "allowlist":
            return ()
        with self._allow_lock:
            resolved = self._resolved.get(plan.rule_id, {})
        return self._allow_rules(plan, resolved)[1]

    def refresh(self, plan: SandboxNetworkPlan) -> bool:
        """Resolve an allowlist env's hostnames again; replace its allow
        chain when the accepts changed. True when the firewall changed.

        Only a rule this backend installed and has not removed is touched:
        an env that is being removed is skipped. A failed update raises
        InfrastructureError (the caller quarantines the env).
        """
        if plan.network != "allowlist" or not any(
            entry.network is None for entry in plan.entries
        ):
            return False
        with self._allow_lock:
            previous = self._resolved.get(plan.rule_id)
        if previous is None:
            return False
        resolved = self._lookup(plan, previous)  # DNS, outside the lock
        with self._allow_lock:
            if self._resolved.get(plan.rule_id) is not previous:
                return False
            before = self._allow_rules(plan, previous)[0]
            after = self._allow_rules(plan, resolved)[0]
            if after != before:
                self._firewall.update(
                    plan.rule_id, replace(self.rules(plan), allow_rules=after)
                )
            self._resolved[plan.rule_id] = resolved
        return after != before

    def create(self, plan: SandboxNetworkPlan) -> str:
        """Install the rule, then create and attest the bridge.

        A failure with a known outcome removes whatever may exist, bridge
        before rule. A create the daemon never answered may still finish, so
        nothing is removed: the rule stays until recovery removes the bridge.
        """
        if not self._firewall.probe():
            raise SetupError("firewall permission probe failed before sandbox network")
        if plan.network == "allowlist":
            resolved = self._lookup(plan, {})
            with self._allow_lock:
                self._resolved[plan.rule_id] = resolved
        try:
            self._firewall.install(plan.rule_id, self.rules(plan))
        except Exception as error:
            self._roll_back(plan, None, error)
        try:
            network = self._client.networks.create(
                plan.name,
                driver="bridge",
                internal=plan.internal,
                enable_ipv6=False,
                check_duplicate=True,
                labels=plan.labels,
                options={BRIDGE_NAME_OPTION: plan.bridge_interface},
            )
        except Exception as error:
            if not _daemon_answered(error):
                raise InfrastructureError(
                    f"recovery_required: sandbox network {plan.name} create "
                    f"outcome is unknown; its rule is retained: {error}"
                ) from error
            self._roll_back(plan, None, error)
        try:
            self.attest(plan, network.id)
        except Exception as error:
            self._roll_back(plan, network.id, error)
        return network.id

    def _roll_back(
        self, plan: SandboxNetworkPlan, network_id: str | None, error: Exception
    ) -> NoReturn:
        try:
            self.remove(plan, network_id)
        except Exception as rollback_error:
            raise InfrastructureError(
                f"recovery_required: partial sandbox network {plan.name} "
                f"rollback is unproven: {rollback_error}"
            ) from error
        raise SetupError(
            f"failed to create sandbox network {plan.name}: {error}; "
            "its bridge and rule are proven absent"
        ) from error

    def attest(
        self,
        plan: SandboxNetworkPlan,
        network_id: str,
        *,
        containers: Collection[str] = (),
    ) -> None:
        """Exact bridge configuration and installed rule, before any start.

        ``containers`` are the owner's container IDs; any other endpoint on
        the bridge is foreign.
        """
        try:
            network = self._client.networks.get(network_id)
        except NotFound:
            raise InfrastructureError(
                f"sandbox network {plan.name} is absent"
            ) from None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot inspect sandbox network {plan.name}: {error}"
            ) from error
        attrs = network.attrs
        expected = {
            "Id": network_id,
            "Name": plan.name,
            "Driver": "bridge",
            "Labels": plan.labels,
            "Options": {BRIDGE_NAME_OPTION: plan.bridge_interface},
        }
        for key, value in expected.items():
            if attrs.get(key) != value:
                raise InfrastructureError(
                    f"sandbox network {plan.name} configuration mismatch: {key}"
                )
        if attrs.get("Internal") is not plan.internal:
            raise InfrastructureError(
                f"sandbox network {plan.name} configuration mismatch: Internal"
            )
        if attrs.get("EnableIPv6") is not False:
            raise InfrastructureError(
                f"sandbox network {plan.name} configuration mismatch: EnableIPv6"
            )
        ipam = attrs.get("IPAM") or {}
        if ipam.get("Driver") != "default" or not _subnets_rejected(ipam.get("Config")):
            raise InfrastructureError(
                f"sandbox network {plan.name} configuration mismatch: IPAM"
            )
        if set(attrs.get("Containers") or {}) - set(containers):
            raise InfrastructureError(
                f"sandbox network {plan.name} has a foreign endpoint"
            )
        try:
            with self._allow_lock:  # never against a half-replaced allow chain
                installed = self._firewall.is_installed(plan.rule_id, self.rules(plan))
        except Exception as error:
            raise InfrastructureError(
                f"failed to attest sandbox network rule {plan.rule_id}: {error}"
            ) from error
        if not installed:
            raise InfrastructureError(
                f"sandbox network rule {plan.rule_id} is not installed exactly"
            )

    def remove(
        self,
        plan: SandboxNetworkPlan,
        network_id: str | None = None,
        *,
        pending: bool = False,
    ) -> tuple[str, ...]:
        """Remove and prove absent the bridge, then the rule.

        Returns the IDs of the bridges found and removed. Idempotent. An
        unjournaled bridge is found by planned name and exact labels; a
        foreign object holding the planned name is never removed. With
        ``pending`` (a create that may still be in flight) finding no bridge
        proves nothing, so the rule is retained and recovery is required.
        """
        known = self._discover(plan)
        if network_id is not None and network_id not in known:
            known = (network_id, *known)
        removed = tuple(
            candidate for candidate in known if self._remove_bridge(plan, candidate)
        )
        if self._discover(plan):
            raise InfrastructureError(
                f"recovery_required: sandbox network {plan.name} remains after removal"
            )
        if pending and not removed:
            raise InfrastructureError(
                f"recovery_required: sandbox network {plan.name} pending create "
                "outcome is unknown; its rule is retained"
            )
        self._remove_rule(plan)
        return removed

    def recover(self, lease: SandboxEnvLease | BuilderLease) -> tuple[str, ...]:
        """Converge a journaled env or builder network to proven absence.

        Returns the removed bridge IDs. While a pending lease has no bridge ID
        its create may still be in flight; finding nothing then fails closed,
        as recovery.py does for a pending child.
        """
        if isinstance(lease, BuilderLease):
            plan = plan_builder_network(lease.owner, lease.builder_id)
        else:
            plan = env_lease_network_plan(lease)
            if plan is None:
                return ()
        pending = lease.pending_mutation and lease.network_id is None
        return self.remove(plan, lease.network_id, pending=pending)

    def _discover(self, plan: SandboxNetworkPlan) -> tuple[str, ...]:
        # The name filter also matches substrings; only the exact name counts.
        try:
            listed = self._client.networks.list(filters={"name": plan.name})
            found = []
            for item in listed:
                if item.attrs.get("Name") != plan.name:
                    continue
                try:
                    network = self._client.networks.get(item.id)
                except NotFound:
                    continue
                self._require_identity(plan, network)
                found.append(network.id)
        except InfrastructureError:
            raise
        except Exception as error:
            raise InfrastructureError(
                f"recovery_required: sandbox network {plan.name} discovery is "
                f"unproven: {error}"
            ) from error
        return tuple(found)

    @staticmethod
    def _require_identity(plan: SandboxNetworkPlan, network: Any) -> None:
        attrs = network.attrs
        if attrs.get("Name") != plan.name or attrs.get("Labels") != plan.labels:
            raise InfrastructureError(
                f"recovery_required: sandbox network {plan.name} is not owned "
                "by its planned identity"
            )

    def _remove_bridge(self, plan: SandboxNetworkPlan, network_id: str) -> bool:
        """False when the bridge was already absent, True once removed."""
        try:
            network = self._client.networks.get(network_id)
        except NotFound:
            return False
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: cannot inspect sandbox network {plan.name} "
                f"before removal: {error}"
            ) from error
        self._require_identity(plan, network)
        failure: Exception | None = None
        try:
            network.remove()
        except (APIError, OSError) as error:
            failure = error
        try:
            self._client.networks.get(network_id)
        except NotFound:
            return True
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"recovery_required: sandbox network {plan.name} absence is "
                f"unproven: {error}"
            ) from error
        detail = f": {failure}" if failure is not None else ""
        raise InfrastructureError(
            f"recovery_required: sandbox network {plan.name} remains after "
            f"removal{detail}"
        ) from failure

    def _remove_rule(self, plan: SandboxNetworkPlan) -> None:
        with self._allow_lock:
            self._resolved.pop(plan.rule_id, None)
            self._remove_rule_locked(plan)

    def _remove_rule_locked(self, plan: SandboxNetworkPlan) -> None:
        try:
            self._firewall.remove(plan.rule_id)
        except FirewallRuleNotFound:
            pass
        except InfrastructureError:
            raise
        except Exception as error:
            raise InfrastructureError(
                f"recovery_required: sandbox network rule {plan.rule_id} "
                f"removal is unproven: {error}"
            ) from error
        try:
            residual = self._firewall.exists(plan.rule_id)
        except Exception as error:
            raise InfrastructureError(
                f"recovery_required: sandbox network rule {plan.rule_id} "
                f"absence is unproven: {error}"
            ) from error
        if residual:
            raise InfrastructureError(
                f"recovery_required: sandbox network rule {plan.rule_id} "
                "remains after removal"
            )


__all__ = [
    "BUILDER_NETWORK_ROLE",
    "ENV_NETWORK_ROLE",
    "MAX_ADDRESSES_PER_HOST",
    "SandboxNetworkBackend",
    "SandboxNetworkPlan",
    "allow_destination",
    "env_lease_network_plan",
    "plan_builder_network",
    "plan_env_network",
]
