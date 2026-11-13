"""Work's Docker seccomp profile: Docker's default without file watches."""

from __future__ import annotations

import errno
import hashlib
import json
import random
from importlib import resources

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import ContainerSpec, NetworkPolicy
from rsi_harness.runtime.network import NetworkPolicyEnforcer
from rsi_harness.runtime.work_seccomp import (
    F_NOTIFY,
    F_SETDELEG,
    F_SETLEASE,
    WATCH_SYSCALLS,
    work_seccomp_option,
    work_seccomp_profile,
)
from tests.fakes import FakeDockerClient, FakeFirewallBackend
from tests.runtime.test_docker import make_runtime

# moby/profiles seccomp/v0.3.0 seccomp/default.json, byte for byte.
DOCKER_DEFAULT_SHA256 = (
    "6416b47770785a41ac59073cdc77d9fe98517df2799dc83ef207e622de3053f6"
)
FCNTL = {"fcntl", "fcntl64"}


def _default() -> dict:
    source = resources.files("rsi_harness.runtime") / "docker_default_seccomp.json"
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == DOCKER_DEFAULT_SHA256
    return json.loads(data)


def _matches(rule: dict, command: int) -> bool:
    """libseccomp's SCMP_CMP_MASKED_EQ on fcntl's 64-bit command argument."""
    return all(
        argument["op"] == "SCMP_CMP_MASKED_EQ"
        and argument["index"] == 1
        and command & argument["value"] == argument["valueTwo"]
        for argument in rule.get("args") or ()
    )


def test_the_profile_is_docker_s_default_but_for_file_watches():
    default = _default()
    profile = json.loads(work_seccomp_profile())
    removed = set(WATCH_SYSCALLS) | FCNTL

    assert {key: value for key, value in profile.items() if key != "syscalls"} == {
        key: value for key, value in default.items() if key != "syscalls"
    }
    kept = [
        {**rule, "names": [name for name in rule["names"] if name not in removed]}
        for rule in default["syscalls"]
    ]
    kept = [rule for rule in kept if rule["names"]]
    added = profile["syscalls"][len(kept) :]
    assert profile["syscalls"][: len(kept)] == kept
    assert added[0] == {
        "names": list(WATCH_SYSCALLS),
        "action": "SCMP_ACT_ERRNO",
        "errnoRet": errno.ENOSYS,
    }
    assert all(set(rule["names"]) == FCNTL for rule in added[1:])
    assert not any(
        removed & set(rule["names"]) and rule["action"] == "SCMP_ACT_ALLOW"
        for rule in profile["syscalls"]
        if not rule.get("args")
    )
    assert work_seccomp_option() == "seccomp=" + work_seccomp_profile()


def test_fcntl_rules_never_overlap_and_refuse_only_notify_lease_and_deleg():
    """Rules that overlap with different actions resolve by their order:
    each command matches rules of one action only, decided on its low 32
    bits as the kernel reads it."""
    rules = [
        rule
        for rule in json.loads(work_seccomp_profile())["syscalls"]
        if FCNTL & set(rule["names"])
    ]
    generator = random.Random(7)
    # 1042 is unassigned.
    refused = (F_SETLEASE, F_NOTIFY, F_SETDELEG, 1042)
    commands = [*range(2048)]
    commands += [command ^ (1 << bit) for command in refused for bit in range(64)]
    commands += [
        (high << 32) | command
        for command in refused
        for high in (1, 0xFFFF, 0xFFFFFFFF)
    ]
    commands += [generator.getrandbits(64) for _ in range(2000)]
    for command in commands:
        decided = {
            (rule["action"], rule.get("errnoRet"))
            for rule in rules
            if _matches(rule, command)
        }
        expected = (
            ("SCMP_ACT_ERRNO", errno.EINVAL)
            if command & 0xFFFFFFFF in refused
            else ("SCMP_ACT_ALLOW", None)
        )
        assert decided == {expected}, command


@pytest.mark.parametrize("read_only", [False, True])
def test_only_work_containers_get_the_profile(tmp_path, read_only):
    client = FakeDockerClient()
    for role in ("work", "judge"):
        (tmp_path / role).mkdir()
        runtime = make_runtime(client, tmp_path / role, role=role)
        runtime.create(ContainerSpec(image=role, read_only_rootfs=read_only))
    work, judge = client.containers.created
    hardened = ["no-new-privileges:true"] if read_only else []
    assert work["security_opt"] == [*hardened, work_seccomp_option()]
    assert judge.get("security_opt", []) == hardened


def test_a_work_container_without_the_profile_does_not_start(tmp_path):
    client = FakeDockerClient()
    enforcer = NetworkPolicyEnforcer(run_id="r", firewall=FakeFirewallBackend())
    runtime = make_runtime(
        client, tmp_path, networked=True, network_policy_enforcer=enforcer
    )
    ref = runtime.create(ContainerSpec(image="work"))
    network = runtime._network
    lease = enforcer.apply(ref, NetworkPolicy(mode="no-network"), network=network)
    runtime.install_network_policy(ref, lease)
    container = client.containers.by_id[ref.container_id]
    container.attrs["HostConfig"]["SecurityOpt"] = []

    with pytest.raises(SetupError, match="lacks its seccomp profile"):
        runtime.start(ref)
    assert container.events == []

    container.attrs["HostConfig"]["SecurityOpt"] = [work_seccomp_option()]
    runtime.start(ref)
    assert container.events == ["start"]
