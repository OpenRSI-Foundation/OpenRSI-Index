from __future__ import annotations

import ipaddress
import shlex
import subprocess
from pathlib import PurePosixPath

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import ContainerSpec, NetworkPolicy
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.netns_firewall import (
    HOLDER_IMAGE,
    NETNS_LABEL,
    ContainerNetnsFirewall,
    RecoveryFirewall,
    parse_host_destinations,
)
from rsi_harness.runtime.network import (
    FirewallRuleNotFound,
    NetworkPolicyEnforcer,
    NetworkRuleSet,
    PinnedEndpoint,
)
from tests.fakes import FakeDockerClient, FakeDockerContainers


class NamespaceContainers(FakeDockerContainers):
    """Record the network fields the namespace firewall inspects."""

    def __init__(self, networks):
        super().__init__()
        self.networks = networks

    def create(self, image, command, **kwargs):
        container = super().create(image, command, **kwargs)
        container.attrs["State"]["Running"] = False
        container.attrs["HostConfig"]["NetworkMode"] = kwargs.get(
            "network_mode"
        ) or kwargs.get("network")
        name = kwargs.get("network")
        attached = {}
        if name is not None:
            (network,) = self.networks.list(filters={"name": name})
            attached[name] = {"NetworkID": network.id}
        container.attrs["NetworkSettings"] = {"Networks": attached}

        def start(container=container):
            container.events.append("start")
            container.attrs["State"]["Running"] = True

        container.start = start
        return container

    def get(self, container_id):
        from docker.errors import NotFound

        container = self.by_id.get(container_id)
        if container is None or container.removed_kwargs is not None:
            raise NotFound(container_id)
        return container

    def list(self, *, all, filters):
        assert all is True
        required = [label.split("=", 1) for label in filters.get("label", ())]
        return [
            container
            for container in self.by_id.values()
            if container.removed_kwargs is None
            and all_match(container.attrs["Config"]["Labels"], required)
        ]


def all_match(labels, required):
    return all(labels.get(key) == value for key, value in required)


class Netfilter:
    """A namespace's filter tables, as iptables-nft-restore and -S see them."""

    def __init__(self) -> None:
        self.tables = {4: self._empty(), 6: self._empty()}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @staticmethod
    def _empty():
        return {"chains": [], "rules": {"INPUT": [], "FORWARD": [], "OUTPUT": []}}

    def _restore(self, family: int, text: str) -> None:
        table = self.tables[family]
        for line in text.splitlines():
            if line in ("*filter", "COMMIT"):
                continue
            if line.startswith(":"):
                chain = line[1:].split()[0]
                table["chains"].append(chain)
                table["rules"][chain] = []
                continue
            tokens = shlex.split(line)
            if tokens[0] == "-A":
                table["rules"][tokens[1]].append(tokens[2:])
            elif tokens[0] == "-I":
                table["rules"][tokens[1]].insert(int(tokens[2]) - 1, tokens[3:])

    def _listing(self, family: int) -> str:
        table = self.tables[family]
        lines = [f"-P {chain} ACCEPT" for chain in ("INPUT", "FORWARD", "OUTPUT")]
        lines += [f"-N {chain}" for chain in sorted(table["chains"])]
        for chain, rules in table["rules"].items():
            lines += [f"-A {chain} {shlex.join(rule)}" for rule in rules]
        return "\n".join(lines) + "\n"

    def __call__(self, holder, arguments):
        script = arguments[1]
        self.calls.append((holder.id, tuple(arguments)))
        if "restore" in script:
            self._restore(4, arguments[3])
            self._restore(6, arguments[4])
            return subprocess.CompletedProcess(arguments, 0, "", "")
        output = self._listing(4) + "--- rsi-ipv6 ---\n" + self._listing(6)
        return subprocess.CompletedProcess(arguments, 0, output, "")


FIB_TRIE = (
    "Main:\n"
    "  +-- 0.0.0.0/0 3 0 5\n"
    "     |-- 203.0.113.0\n"
    "        /24 link UNICAST\n"
    "     |-- 203.0.113.7\n"
    "        /32 host LOCAL\n"
    "     |-- 203.0.113.255\n"
    "        /32 link BROADCAST\n"
    "     |-- 127.0.0.1\n"
    "        /32 host LOCAL\n"
    "Local:\n"
    "     |-- 172.17.0.1\n"
    "        /32 host LOCAL\n"
    "     |-- 198.18.4.0\n"
    "        /24 host LOCAL\n"
    "     |-- 203.0.113.7\n"
    "        /32 host LOCAL\n"
)


def netns_firewall(client, tmp_path, netfilter=None, *, daemon=FIB_TRIE):
    local = tmp_path / "fib_trie"
    local.write_text(FIB_TRIE)
    return ContainerNetnsFirewall(
        client,
        helper=netfilter or Netfilter(),
        daemon_fib_trie=lambda: daemon,
        local_fib_trie=local,
    )


def namespace_client():
    client = FakeDockerClient()
    client.containers = NamespaceContainers(client.networks)
    client.images.by_ref[HOLDER_IMAGE] = object()
    return client


def runtime_with_netns(tmp_path, client, firewall):
    task_source = tmp_path / "task"
    engine = tmp_path / "engine"
    task_source.mkdir(exist_ok=True)
    engine.mkdir(exist_ok=True)
    provisioner = DockerContainerRuntime(
        client,
        run_id="run1",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )
    network = provisioner.create_network("phase", internal=False)
    client.networks.by_id[network.network_id].attrs["IPAM"] = {
        "Config": [{"Subnet": "172.30.5.0/24", "Gateway": "172.30.5.1"}]
    }
    enforcer = NetworkPolicyEnforcer(
        run_id="run1", firewall=firewall, engine_destinations=("172.17.0.1",)
    )
    runtime = DockerContainerRuntime(
        client,
        run_id="run1",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
        network=network,
        network_policy_enforcer=enforcer,
    )
    return runtime, network, enforcer


def test_container_joins_a_holder_that_owns_bridge_and_resolver(tmp_path):
    client = namespace_client()
    firewall = netns_firewall(client, tmp_path)
    runtime, network, _ = runtime_with_netns(tmp_path, client, firewall)

    ref = runtime.create(
        ContainerSpec(
            image="work",
            extra_hosts=(("api.test", "198.51.100.8"),),
            dns=("192.0.2.1",),
        ),
        planned_name=runtime.planned_container_name("main"),
    )

    holder_record, work_record = client.containers.created
    assert holder_record["image"] == HOLDER_IMAGE
    assert holder_record["network"] == network.name
    assert holder_record["extra_hosts"] == ["api.test:198.51.100.8"]
    assert holder_record["dns"] == ["192.0.2.1"]
    assert holder_record["cap_drop"] == ["ALL"]
    assert holder_record["read_only"] is True
    assert holder_record["labels"][NETNS_LABEL] == "holder"
    assert holder_record["labels"]["rsi-harness.role"] == "work-netns"
    holder_id = next(iter(client.containers.by_id))
    assert work_record["network_mode"] == f"container:{holder_id}"
    for absent in ("network", "extra_hosts", "dns"):
        assert absent not in work_record

    runtime.remove(ref)
    assert all(
        container.removed_kwargs is not None
        for container in client.containers.by_id.values()
    )


def test_policy_is_installed_in_the_namespace_and_attested_exactly(tmp_path):
    client = namespace_client()
    netfilter = Netfilter()
    firewall = netns_firewall(client, tmp_path, netfilter)
    runtime, network, enforcer = runtime_with_netns(tmp_path, client, firewall)
    ref = runtime.create(ContainerSpec(image="work"))
    holder_id = next(iter(client.containers.by_id))

    lease = enforcer.apply(ref, NetworkPolicy(mode="no-network"), network=network)

    assert lease.rule_id.startswith("rsn-run1-work-")
    assert client.containers.by_id[holder_id].attrs["State"]["Running"] is True
    assert {call[0] for call in netfilter.calls} == {holder_id}
    output_chain = next(
        chain for chain in netfilter.tables[4]["chains"] if chain.startswith("RSI_O_")
    )
    output = netfilter.tables[4]["rules"][output_chain]
    assert output[0] == ["-o", "lo", "-j", "ACCEPT"]
    assert [
        "-d",
        "203.0.113.7/32",
        "-j",
        "REJECT",
        "--reject-with",
        "icmp-port-unreachable",
    ] in output
    assert output[-1][:2] == ["-j", "REJECT"]
    assert netfilter.tables[4]["rules"]["OUTPUT"] == [["-j", output_chain]]
    assert netfilter.tables[6]["rules"][output_chain][-1][:2] == ["-j", "REJECT"]

    runtime.install_network_policy(ref, lease)
    runtime.start(ref)

    # Any rule outside the policy fails attestation.
    netfilter.tables[4]["rules"]["OUTPUT"].insert(0, ["-j", "ACCEPT"])
    with pytest.raises(SetupError, match="no longer installed"):
        enforcer.attest(lease)


def test_public_policy_still_rejects_host_and_private_destinations(tmp_path):
    client = namespace_client()
    netfilter = Netfilter()
    firewall = netns_firewall(client, tmp_path, netfilter)
    runtime, network, enforcer = runtime_with_netns(tmp_path, client, firewall)
    ref = runtime.create(ContainerSpec(image="work"))

    enforcer.apply(ref, NetworkPolicy(mode="public"), network=network)

    output_chain = next(
        chain for chain in netfilter.tables[4]["chains"] if chain.startswith("RSI_O_")
    )
    output = netfilter.tables[4]["rules"][output_chain]
    assert output[-1] == ["-j", "ACCEPT"]
    for destination in (
        "203.0.113.7/32",  # a host address
        "198.18.4.0/24",  # a local (AnyIP) route
        "203.0.113.255/32",  # a host broadcast address
        "172.30.5.0/24",  # this bridge, its gateway included
        "255.255.255.255/32",
        "172.16.0.0/12",
    ):
        (rule,) = [rule for rule in output if rule[:2] == ["-d", destination]]
        assert rule[2:4] == ["-j", "REJECT"]
    assert netfilter.tables[6]["rules"][output_chain][-1][:2] == ["-j", "REJECT"]


def test_holder_authority_must_match_the_container_and_network(tmp_path):
    client = namespace_client()
    firewall = netns_firewall(client, tmp_path)
    runtime, network, enforcer = runtime_with_netns(tmp_path, client, firewall)
    ref = runtime.create(ContainerSpec(image="work"))
    holder = next(iter(client.containers.by_id.values()))
    holder.attrs["Config"]["Labels"]["rsi-harness.run-id"] = "other"

    with pytest.raises(SetupError, match="holder authority"):
        enforcer.apply(ref, NetworkPolicy(mode="no-network"), network=network)


def test_brokered_sandbox_rules_are_refused(tmp_path):
    firewall = netns_firewall(namespace_client(), tmp_path)
    rules = NetworkRuleSet(
        container_id="c",
        network_id="n",
        network_name="net",
        bridge_interface="rsi000000000000",
        role="work",
        mode="allowlist",
        exact_endpoints=(),
        allow_networks=(),
        engine_destinations=(),
        dns_resolvers=(),
        allow_rules=((ipaddress.IPv4Network("198.51.100.0/24"), None),),
    )
    with pytest.raises(SetupError, match="host firewall"):
        firewall.install("rsn-run1-work-000000000000", rules)


def test_later_process_finds_and_removes_holders_by_run_and_role(tmp_path):
    client = namespace_client()
    holders = {
        role: client.containers.create(
            HOLDER_IMAGE,
            None,
            labels={"rsi-harness.run-id": "run1", "rsi-harness.role": role},
        )
        for role in ("work-netns", "judge-netns")
    }
    firewall = netns_firewall(client, tmp_path)
    rule = "rsn-run1-work-0123456789ab"

    # A holder that never ran has no namespace, so no rule ...
    assert firewall.exists(rule) is False
    holders["work-netns"].start()
    holders["judge-netns"].start()
    assert firewall.exists(rule) is True
    # ... and removal still takes every holder of that run and role.
    firewall.remove(rule)
    assert firewall.exists(rule) is False
    assert firewall.exists("rsn-run1-judge-0123456789ab") is True
    with pytest.raises(FirewallRuleNotFound):
        firewall.remove(rule)


def test_recovery_routes_each_rule_to_its_installing_backend():
    calls = []

    class Backend:
        def __init__(self, name):
            self.name = name

        def exists(self, rule_id):
            calls.append((self.name, "exists", rule_id))
            return True

        def remove(self, rule_id):
            calls.append((self.name, "remove", rule_id))

    firewall = RecoveryFirewall(Backend("host"), Backend("netns"))
    firewall.exists("rsi-run1-work-0123456789ab")
    firewall.remove("rsn-run1-judge-0123456789ab")
    assert calls == [
        ("host", "exists", "rsi-run1-work-0123456789ab"),
        ("netns", "remove", "rsn-run1-judge-0123456789ab"),
    ]


def test_pinned_endpoints_are_the_only_routed_accepts(tmp_path):
    client = namespace_client()
    netfilter = Netfilter()
    firewall = netns_firewall(client, tmp_path, netfilter)
    runtime, network, enforcer = runtime_with_netns(tmp_path, client, firewall)
    ref = runtime.create(ContainerSpec(image="work"))
    endpoint = PinnedEndpoint(
        hostname="api.test",
        port=443,
        addresses=(ipaddress.IPv4Address("198.51.100.8"),),
    )

    enforcer.apply(
        ref,
        NetworkPolicy(mode="no-network"),
        network=network,
        api_endpoints=(endpoint,),
    )

    output_chain = next(
        chain for chain in netfilter.tables[4]["chains"] if chain.startswith("RSI_O_")
    )
    accepts = [
        rule
        for rule in netfilter.tables[4]["rules"][output_chain]
        if rule[-1] == "ACCEPT" and rule[:1] == ["-d"]
    ]
    assert accepts == [
        [
            "-d",
            "198.51.100.8/32",
            "-p",
            "tcp",
            "-m",
            "tcp",
            "--dport",
            "443",
            "-j",
            "ACCEPT",
        ]
    ]


def test_host_destinations_cover_local_routes_and_broadcasts():
    assert [str(network) for network in parse_host_destinations(FIB_TRIE)] == [
        "203.0.113.7/32",
        "203.0.113.255/32",
        "172.17.0.1/32",
        "198.18.4.0/24",
    ]


def test_a_controller_outside_the_daemon_namespace_is_refused(tmp_path):
    firewall = netns_firewall(
        namespace_client(),
        tmp_path,
        daemon=FIB_TRIE.replace("203.0.113.7", "192.0.2.50"),
    )
    with pytest.raises(SetupError, match="daemon host's network namespace"):
        firewall.host_destinations()


def test_rule_ids_name_the_backend_that_installs_them(tmp_path):
    from tests.fakes import FakeFirewallBackend

    host = NetworkPolicyEnforcer(run_id="run1", firewall=FakeFirewallBackend())
    client = namespace_client()
    netns_firewall_backend = netns_firewall(client, tmp_path)
    runtime, network, enforcer = runtime_with_netns(
        tmp_path, client, netns_firewall_backend
    )
    ref = runtime.create(ContainerSpec(image="work"))
    policy = NetworkPolicy(mode="no-network")

    assert host.container_netns is False
    assert host.plan(ref, policy, network=network).rule_id.startswith("rsi-run1-")
    assert enforcer.container_netns is True
    assert enforcer.plan(ref, policy, network=network).rule_id.startswith("rsn-run1-")


def carrying_client():
    client = namespace_client()
    archives = []
    create = client.containers.create

    def create_with_archive(image, command, **kwargs):
        container = create(image, command, **kwargs)

        def put_archive(path, data, container=container):
            body = data.read() if hasattr(data, "read") else data
            archives.append((container.id, path, kwargs.get("mounts"), body))
            return True

        container.put_archive = put_archive
        return container

    client.containers.create = create_with_archive
    return client, archives


def archive_names(body):
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(body)) as archive:
        return {member.name: (member.uid, member.mode) for member in archive}


def test_engine_owned_read_only_mounts_are_carried_in_a_volume(tmp_path):
    from rsi_harness.models import WORK_FEEDBACK_ROOT, ContainerMount

    client, archives = carrying_client()
    firewall = netns_firewall(client, tmp_path)
    runtime, network, _ = runtime_with_netns(tmp_path, client, firewall)
    feedback = tmp_path / "engine" / "feedback"
    feedback.mkdir(mode=0o755)
    feedback.chmod(0o755)
    (feedback / "agent-1.log").write_text("round one\n")
    (feedback / "agent-1.log").chmod(0o644)
    runtime = DockerContainerRuntime(
        client,
        run_id="run1",
        task_id="t",
        role="work",
        task_source_dir=tmp_path / "task",
        allowed_mount_roots=(tmp_path / "engine",),
        network=network,
        network_policy_enforcer=runtime._network_policy_enforcer,
        work_feedback_dir=feedback,
        staging_dir=tmp_path / "engine" / "staging",
    )

    ref = runtime.create(
        ContainerSpec(
            image="work",
            mounts=(
                ContainerMount(
                    source=feedback, target=WORK_FEEDBACK_ROOT, read_only=True
                ),
            ),
        ),
        planned_name=runtime.planned_container_name("main"),
    )

    (volume_name,) = client.volumes.by_name
    work = client.containers.created[-1]
    assert work["image"] == "work"
    assert [dict(m) for m in work["mounts"]] == [
        {
            "Target": str(WORK_FEEDBACK_ROOT),
            "Source": volume_name,
            "Type": "volume",
            "ReadOnly": True,
            "VolumeOptions": {"Subpath": "m0"},
        }
    ]
    assert archive_names(archives[0][3]) == {
        "m0": (0, 0o755),
        "m0/agent-1.log": (0, 0o644),
    }
    assert client.volumes.by_name[volume_name].attrs["Labels"][NETNS_LABEL] == (
        "carrier"
    )

    (feedback / "agent-2.log").write_text("round two\n")
    runtime.refresh_work_feedback(ref)
    assert "m0/agent-2.log" in archive_names(archives[-1][3])

    runtime.remove(ref)
    assert client.volumes.by_name[volume_name].removed is True
    assert all(
        container.removed_kwargs is not None
        for container in client.containers.by_id.values()
    )


def test_copy_streams_over_exec_stdin_without_a_staging_bind(tmp_path, monkeypatch):
    client = namespace_client()
    firewall = netns_firewall(client, tmp_path)
    runtime, network, enforcer = runtime_with_netns(tmp_path, client, firewall)
    runtime = DockerContainerRuntime(
        client,
        run_id="run1",
        task_id="t",
        role="work",
        task_source_dir=tmp_path / "task",
        allowed_mount_roots=(tmp_path / "engine",),
        network=network,
        network_policy_enforcer=enforcer,
        staging_dir=tmp_path / "engine" / "staging",
    )
    ref = runtime.create(ContainerSpec(image="work"))
    assert client.containers.created[-1]["mounts"] == []
    source = tmp_path / "prompt.md"
    source.write_bytes(b"do the task\n")
    sent = []

    def capture(self, container, command, chunks):
        sent.append((container, command, b"".join(chunks)))
        return True

    monkeypatch.setattr(DockerContainerRuntime, "_exec_with_stdin", capture)

    runtime.copy_to(ref, source, PurePosixPath("/installed-agent/prompt.md"))

    ((container, command, content),) = sent
    assert container == ref
    assert command[-2:] == ("/installed-agent/prompt.md", "/installed-agent")
    assert content == b"do the task\n"


def test_old_daemons_are_refused_before_any_work():
    from types import SimpleNamespace

    from rsi_harness.runtime.netns_firewall import require_container_netns_support

    require_container_netns_support(
        SimpleNamespace(api=SimpleNamespace(api_version="1.45"))
    )
    with pytest.raises(SetupError, match="Docker Engine 26"):
        require_container_netns_support(
            SimpleNamespace(api=SimpleNamespace(api_version="1.44"))
        )


def test_production_selects_the_firewall_by_controller_identity(monkeypatch):
    from types import SimpleNamespace

    from rsi_harness.runtime import production
    from rsi_harness.runtime.network import DockerIptablesFirewallBackend

    client = SimpleNamespace(api=SimpleNamespace(api_version="1.53"))
    task = SimpleNamespace(sandbox=None)

    assert isinstance(
        production.run_firewall(client, task), DockerIptablesFirewallBackend
    )

    monkeypatch.setattr(production, "container_netns_firewall_selected", lambda: True)
    probed = []
    monkeypatch.setattr(
        ContainerNetnsFirewall, "host_destinations", lambda self: probed.append(self)
    )
    selected = production.run_firewall(client, task)
    assert isinstance(selected, ContainerNetnsFirewall)
    assert probed == [selected]

    with pytest.raises(SetupError, match="sandbox tasks need the host firewall"):
        production.run_firewall(client, SimpleNamespace(sandbox=object()))
    old = SimpleNamespace(api=SimpleNamespace(api_version="1.44"))
    with pytest.raises(SetupError, match="Docker Engine 26"):
        production.run_firewall(old, task)


def test_recovery_uses_both_backends_whoever_runs_it():
    from types import SimpleNamespace

    from rsi_harness.runtime import production
    from rsi_harness.runtime.network import DockerIptablesFirewallBackend

    firewall = production.recovery_firewall(SimpleNamespace())
    assert isinstance(
        firewall._backend("rsi-run1-work-0123456789ab"), DockerIptablesFirewallBackend
    )
    assert isinstance(
        firewall._backend("rsn-run1-work-0123456789ab"), ContainerNetnsFirewall
    )


def test_the_console_names_the_namespace_firewall(capsys):
    from rsi_harness.cli import _RunConsole

    _RunConsole()("network_policy", {"backend": "container-namespace"})
    assert "each container's own namespace" in capsys.readouterr().out


def test_judge_helpers_are_carried_including_a_single_file_target(tmp_path):
    from rsi_harness.judge_agent import (
        JUDGE_ISOLATE_CODE_TARGET,
        JUDGE_ISOLATE_LAUNCHER,
        JUDGE_ISOLATE_LAUNCHER_TARGET,
        install_judge_isolate,
    )
    from rsi_harness.models import ContainerMount

    client, archives = carrying_client()
    firewall = netns_firewall(client, tmp_path)
    _, _, enforcer = runtime_with_netns(tmp_path, client, firewall)
    network = DockerContainerRuntime(
        client,
        run_id="run1",
        task_id="t",
        role="judge",
        task_source_dir=tmp_path / "task",
        allowed_mount_roots=(tmp_path / "engine",),
    ).create_network("agent-1", internal=True)
    run_root = tmp_path / "engine" / "run"
    run_root.mkdir()
    isolate = install_judge_isolate(run_root)
    logs = tmp_path / "engine" / "verifier"
    logs.mkdir()
    judge = DockerContainerRuntime(
        client,
        run_id="run1",
        task_id="t",
        role="judge",
        task_source_dir=tmp_path / "task",
        allowed_mount_roots=(tmp_path / "engine",),
        network=network,
        network_policy_enforcer=enforcer,
    )

    judge.create(
        ContainerSpec(
            image="snapshot",
            mounts=(
                ContainerMount(source=logs, target=PurePosixPath("/logs/verifier")),
                ContainerMount(
                    source=isolate, target=JUDGE_ISOLATE_CODE_TARGET, read_only=True
                ),
                ContainerMount(
                    source=isolate / JUDGE_ISOLATE_LAUNCHER,
                    target=JUDGE_ISOLATE_LAUNCHER_TARGET,
                    read_only=True,
                ),
            ),
        )
    )

    mounts = {m["Target"]: dict(m) for m in client.containers.created[-1]["mounts"]}
    assert mounts[str(JUDGE_ISOLATE_CODE_TARGET)]["VolumeOptions"] == {"Subpath": "m0"}
    assert mounts[str(JUDGE_ISOLATE_LAUNCHER_TARGET)]["VolumeOptions"] == {
        "Subpath": "m1"
    }
    assert all(
        mounts[target]["ReadOnly"] and mounts[target]["Type"] == "volume"
        for target in (
            str(JUDGE_ISOLATE_CODE_TARGET),
            str(JUDGE_ISOLATE_LAUNCHER_TARGET),
        )
    )
    # The verifier logs are the private anonymous volume, never carried.
    assert mounts["/logs/verifier"]["Source"] is None
    names = archive_names(archives[0][3])
    assert names["m1"] == (0, 0o755)
    assert f"m0/{JUDGE_ISOLATE_LAUNCHER}" in names


def test_helper_runs_with_only_net_admin_in_the_holder_namespace(tmp_path):
    client = namespace_client()
    created = []

    class Helper:
        id = "helper"

        def start(self):
            pass

        def wait(self, timeout):
            return {"StatusCode": 3}

        def logs(self, stdout, stderr):
            return b"" if stdout else b"iptables-nft-restore: line 2 failed\n"

        def remove(self, force):
            created.append("removed")

    def create(image, command, **options):
        created.append((image, command, options))
        return Helper()

    firewall = ContainerNetnsFirewall(
        client, daemon_fib_trie=lambda: FIB_TRIE, local_fib_trie=tmp_path / "x"
    )
    client.containers.create = create
    holder = SimpleHolder()
    with pytest.raises(SetupError, match="line 2 failed"):
        firewall._helper_checked(holder, "exit 3")
    (image, command, options), removed = created
    assert removed == "removed"
    assert options["network_mode"] == "container:holder-1"
    assert options["cap_drop"] == ["ALL"] and options["cap_add"] == ["NET_ADMIN"]
    assert options["read_only"] is True
    assert options["security_opt"] == ["no-new-privileges:true"]
    assert "privileged" not in options and "pid_mode" not in options
    assert options["labels"]["rsi-harness.netns"] == "firewall"


class SimpleHolder:
    id = "holder-1"
    attrs = {
        "Config": {
            "Labels": {
                "rsi-harness.run-id": "run1",
                "rsi-harness.task-id": "t",
                "rsi-harness.role": "work-netns",
            }
        }
    }


def test_the_next_host_probe_removes_stale_leftovers(tmp_path):
    from types import SimpleNamespace

    removed = []

    def probe(created):
        return SimpleNamespace(
            attrs={"Created": created}, remove=lambda force: removed.append(created)
        )

    listed = [probe("2020-01-01T00:00:00.123456789Z"), probe("2999-01-01T00:00:00Z")]
    client = namespace_client()
    calls = []

    def listing(*, all, filters):
        calls.append(filters)
        return listed

    client.containers.list = listing
    firewall = netns_firewall(client, tmp_path)
    firewall._sweep_stale_host_probes()
    assert calls == [
        {
            "label": "rsi-harness.netns=host-probe",
            "status": ["created", "exited", "dead"],
        }
    ]
    assert removed == ["2020-01-01T00:00:00.123456789Z"]


def test_a_lease_of_another_identity_is_a_setup_error(tmp_path, monkeypatch):
    from rsi_harness.runtime.recovery import LeaseStore

    store = LeaseStore(tmp_path / "leases")
    store.root.mkdir()
    (store.root / "run1.json").write_text("{}")

    def refuse(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(type(store.root / "run1.json"), "read_text", refuse)
    with pytest.raises(SetupError, match="belongs to another user"):
        store.read("run1")
    monkeypatch.setattr("rsi_harness.runtime.recovery.os.open", refuse)
    with pytest.raises(SetupError, match="belongs to another user"):
        with store.lock("run1"):
            pass
