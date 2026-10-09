"""Operator root checks (spec 8 step 5; sandbox-root-checks.md, M2-M8).

scripts/operator/sandbox_root_check.sh runs these as root (RSI_ACCEPTANCE=1,
RSI_SANDBOX_ROOT_MODE=1), next to the D tests it re-runs in root mode. Each
check has its own run id: every Docker object it makes carries that run's
labels and is removed by label afterwards, and it installs and removes only
the rules of its own env and builder bridges (``rsi-<run>-...``). Nothing of
another run or user is touched.

RSI_ROOT_CHECK_SELFTEST=1 lets a non-root user run R1-R3 against the fake
firewall: every probe still runs and is reported, and only the assertions
that something is blocked (what needs root) are skipped.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import io
import os
import re
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException

from rsi_harness.runtime import sandbox_env_contracts as env
from rsi_harness.runtime.network import firewall_rule_chains
from rsi_harness.runtime.production import (
    coordinator_may_run,
    firewall_commands_running,
)
from rsi_harness.runtime.sandbox_env_docker import (
    CgroupPausedKiller,
    default_paused_killer,
)
from rsi_harness.runtime.sandbox_network import (
    SandboxNetworkBackend,
    _resolve_ipv4,
    plan_env_network,
)
from tests.fakes import FakeFirewallBackend
from tests.integration.harbor_env_support import (
    live_sandbox as live_sandbox,
)
from tests.integration.sandbox_support import (
    assert_no_rules,
    root_mode,
    sandbox_firewall,
)
from tests.integration.test_sandbox_build_docker import (
    ALPINE,
    BUILDKIT,
    BuildRun,
    build,
)
from tests.integration.test_sandbox_env_docker import (
    BUSYBOX,
    HANDLES,
    RealEnv,
    remove_labelled,
    single,
)
from tests.integration.test_sandbox_env_recovery import SENTINEL, uptime
from tests.integration.test_sandbox_envs_docker import env_spec, pulled, service, up
from tests.integration.test_sandbox_exec_docker import (
    SESSION,
    await_processes,
    destroy,
    ready_env,
    run_to_end,
    sleeping,
    start,
)
from tests.integration.test_sandbox_exec_docker import (
    exec_run as exec_run,
)
from tests.integration.test_sandbox_network_docker import (
    IMAGE,
    SERVER,
    env_plan,
    owner,
    start_service,
)
from tests.sandbox_helpers import make_env_spec

pytestmark = [pytest.mark.acceptance, pytest.mark.integration]

PROBE_IMAGE = "python:3.13-slim-bookworm"
# The host's LAN gateway named by spec R1; another host may pass its own.
LAN = os.environ.get("RSI_ROOT_CHECK_LAN", "192.168.249.1")
CONNECT = (
    "import socket, sys; "
    "socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=3).close()"
)
RESOLVE = "import socket, sys; socket.getaddrinfo(sys.argv[1], 443)"
HTTPS = "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=15)"
# A RUN step's probes: TCP connects and one HTTPS GET, one line each.
RUN_PROBE = b"""import socket, sys, urllib.request


def gateway():
    for line in open("/proc/net/route").read().splitlines()[1:]:
        fields = line.split()
        if fields[1] == "00000000":
            return socket.inet_ntoa(bytes.fromhex(fields[2])[::-1])


targets = dict(item.split("=", 1) for item in sys.argv[1:])
targets["gateway"] = f"{gateway()}:22"
for name, target in targets.items():
    try:
        if target.startswith("https://"):
            urllib.request.urlopen(target, timeout=15)
        else:
            host, port = target.rsplit(":", 1)
            socket.create_connection((host, int(port)), timeout=3).close()
        state = "open"
    except Exception:
        state = "closed"
    print(f"PROBE {name} {state}", flush=True)
"""


def _selftest() -> bool:
    return os.geteuid() != 0 and os.environ.get("RSI_ROOT_CHECK_SELFTEST") == "1"


def _require_root() -> None:
    if os.geteuid() != 0:
        message = "this check needs root (run the operator script with sudo)"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    if not root_mode():
        pytest.fail("root checks run in root mode: set RSI_SANDBOX_ROOT_MODE=1")


@pytest.fixture
def labelled_run():
    """(client, run_id); every object labelled with the run is removed after."""
    if not _selftest():
        _require_root()
    try:
        client = docker.from_env(timeout=120)
        client.ping()
        for image in (IMAGE, PROBE_IMAGE):
            client.images.get(image)
    except (DockerException, OSError) as error:
        pytest.fail(f"Docker or a cached image is unavailable: {error}")
    run_id = f"m9-root-{uuid.uuid4().hex[:12]}"
    try:
        yield client, run_id
    finally:
        errors = remove_labelled(client, {"label": f"rsi-harness.run-id={run_id}"})
        client.close()
        assert errors == []


def _real(firewall) -> bool:
    return not isinstance(firewall, FakeFirewallBackend)


def _check(container, code: str, *args) -> bool:
    argv = ["python3", "-c", code, *(str(item) for item in args)]
    return container.exec_run(argv).exit_code == 0


def _address(container, network: str) -> str:
    container.reload()
    return container.attrs["NetworkSettings"]["Networks"][network]["IPAddress"]


def _gateway(client, network: str) -> str:
    return client.networks.get(network).attrs["IPAM"]["Config"][0]["Gateway"]


def _report(outcomes: dict[str, bool], control: dict[str, bool] | None = None) -> None:
    """One line per probe; with ``control``, whether the same target is
    reached without the firewall, i.e. whether "blocked" proves the firewall
    (a target this host never answers blocks itself)."""
    control = control or {}
    for name, reached in outcomes.items():
        line = f"  {'reached' if reached else 'blocked'}: {name}"
        if name in control:
            unfiltered = "reached" if control[name] else "blocked by the host too"
            line += f" (without the firewall: {unfiltered})"
        print(line)


def _control(client, run_id: str, targets: dict[str, tuple[str, int]]) -> dict:
    """The positive control: the same connects from an unfirewalled
    container on Docker's default bridge, labelled with the run."""
    box = client.containers.run(
        PROBE_IMAGE,
        ["sleep", "300"],
        network="bridge",
        labels={"rsi-harness.run-id": run_id},
        runtime="runc",
        detach=True,
    )
    try:
        return {name: _check(box, CONNECT, *target) for name, target in targets.items()}
    finally:
        box.remove(force=True, v=True)


def _remove_run_containers(client, run_id: str) -> None:
    filters = {"label": f"rsi-harness.run-id={run_id}"}
    for container in client.containers.list(all=True, filters=filters):
        container.remove(force=True, v=True)


class _Bridges:
    """Env bridges made through the production backend; removed (containers
    first) whatever happens."""

    def __init__(self, client, run_id, backend):
        self.client, self.run_id, self.backend = client, run_id, backend
        self.made = []

    def create(self, network="public"):
        plan = env_plan(self.run_id, network)
        self.made.append((plan, self.backend.create(plan)))
        return plan

    def close(self):
        _remove_run_containers(self.client, self.run_id)
        for plan, network_id in reversed(self.made):
            self.backend.remove(plan, network_id)
            assert not Path("/sys/class/net", plan.bridge_interface).exists()


@pytest.fixture
def bridges(labelled_run):
    client, run_id = labelled_run
    firewall = sandbox_firewall(client)
    made = _Bridges(client, run_id, SandboxNetworkBackend(client, firewall))
    try:
        yield made, firewall
    finally:
        made.close()
        assert_no_rules(firewall, run_id)


# -- R1-R3: networks -----------------------------------------------------------


def test_r1_a_public_env_reaches_the_internet_and_nothing_private(
    labelled_run, bridges
):
    """Spec 8 check 1 / M2 R1."""
    client, run_id = labelled_run
    made, firewall = bridges
    plan, sibling_plan = made.create(), made.create()
    # A stand-in for a Harness Work bridge: an ordinary labelled bridge.
    work = client.networks.create(
        f"rsi-rootcheck-{run_id[-12:]}",
        driver="bridge",
        labels={"rsi-harness.run-id": run_id},
    )
    sibling = start_service(client, sibling_plan, 0, ("sibling",), SERVER)
    work_server = client.containers.run(
        IMAGE,
        ["sh", "-c", SERVER],
        network=work.name,
        labels={"rsi-harness.run-id": run_id},
        runtime="runc",
        detach=True,
    )
    probe = start_service(client, plan, 0, ("probe",), "sleep 300", image=PROBE_IMAGE)
    gateway = _gateway(client, plan.name)
    docker0 = _gateway(client, "bridge")
    blocked = {
        "metadata 169.254.169.254:80": ("169.254.169.254", 80),
        "RFC1918 10.0.0.1:80": ("10.0.0.1", 80),
        f"LAN {LAN}:80": (LAN, 80),
        f"env gateway {gateway}:22 (host INPUT)": (gateway, 22),
        "sibling env service": (_address(sibling, sibling_plan.name), 8080),
        "Work-style bridge service": (_address(work_server, work.name), 8080),
        f"docker0 gateway {docker0}:22": (docker0, 22),
    }
    outcomes = {
        "DNS example.com": _check(probe, RESOLVE, "example.com"),
        "https://example.com": _check(probe, HTTPS, "https://example.com/"),
        **{name: _check(probe, CONNECT, *target) for name, target in blocked.items()},
    }
    control = _control(client, run_id, blocked)
    _report(outcomes, control)

    assert outcomes["DNS example.com"] and outcomes["https://example.com"]
    # Without the firewall at least the gateways answer on this host: the
    # control proves the probes can see a reachable target at all.
    assert any(control.values()), control
    if _real(firewall):
        assert [name for name in blocked if outcomes[name]] == []


@pytest.mark.parametrize("network", ["public", "none"])
def test_r2_services_reach_each_other_by_alias_and_never_another_env(
    labelled_run, bridges, network
):
    """Spec 8 check 2 / M2 R2 (and M3: the alias path with the real firewall)."""
    client, run_id = labelled_run
    made, firewall = bridges
    plan, other_plan = made.create(network), made.create(network)
    kv = start_service(client, plan, 0, ("kvstore",), SERVER)
    main = start_service(client, plan, 1, ("main",), "sleep 300", image=PROBE_IMAGE)
    other = start_service(
        client, other_plan, 0, ("other",), "sleep 300", image=PROBE_IMAGE
    )
    outcomes = {
        "main -> kvstore:8080 (alias)": _check(main, CONNECT, "kvstore", 8080),
        "other env -> kv": _check(other, CONNECT, _address(kv, plan.name), 8080),
        "main resolves example.com": _check(main, RESOLVE, "example.com"),
    }
    _report(outcomes)

    assert outcomes["main -> kvstore:8080 (alias)"]
    if network == "none":
        # An internal bridge forwards no external name, only its peers.
        assert not outcomes["main resolves example.com"]
    if _real(firewall):
        assert not outcomes["other env -> kv"]


def test_r2_without_the_intra_bridge_accept_peers_are_rejected(labelled_run):
    """The ACCEPT M2 adds is needed on this host (bridge-nf-call-iptables=1,
    env subnets inside the rejected 172.16/12): without it, peers fail."""
    if _selftest():
        pytest.skip("needs the real firewall")
    client, run_id = labelled_run
    firewall = sandbox_firewall(client)

    class WithoutIntraBridgeAccept(SandboxNetworkBackend):
        def rules(self, plan):
            return dataclasses.replace(super().rules(plan), intra_bridge_accept=False)

    made = _Bridges(client, run_id, WithoutIntraBridgeAccept(client, firewall))
    try:
        plan = made.create()
        start_service(client, plan, 0, ("kvstore",), SERVER)
        main = start_service(client, plan, 1, ("main",), "sleep 300", image=PROBE_IMAGE)
        assert not _check(main, CONNECT, "kvstore", 8080)
    finally:
        made.close()
    assert_no_rules(firewall, run_id)


def _allowlist_plan(run_id, entries, private_cidrs=()):
    raw = {**make_env_spec(), "network": "allowlist", "allowlist": list(entries)}
    env_id = "e" + uuid.uuid4().hex
    return plan_env_network(
        owner(run_id), env_id, env.parse_env_spec(raw), private_cidrs=private_cidrs
    )


def test_r11_an_allowlist_env_reaches_its_entries_and_nothing_else(labelled_run):
    """M2 allowlist: a listed hostname (port-limited) and IP are reached;
    everything else is rejected as in none/public mode, listed private,
    LAN and metadata addresses included, unless the operator approved that
    private CIDR. Listed names resolve through Docker's embedded DNS while
    no resolver is reachable directly. A refresh replaces the allow chain in
    place (attested exactly afterwards)."""
    client, run_id = labelled_run
    firewall = sandbox_firewall(client)
    extra = {}

    def resolver(hostname):
        return (*extra.get(hostname, ()), *_resolve_ipv4(hostname))

    backend = SandboxNetworkBackend(client, firewall, resolver=resolver)
    made = _Bridges(client, run_id, backend)
    try:
        listed = ("example.com:443", "1.1.1.1", LAN, "169.254.169.254", "10.0.0.1")
        plan = _allowlist_plan(run_id, listed)
        approved = _allowlist_plan(run_id, (f"{LAN}:80",), (f"{LAN}/32",))
        for item in (plan, approved):
            made.made.append((item, backend.create(item)))
        probe = start_service(
            client, plan, 0, ("probe",), "sleep 300", image=PROBE_IMAGE
        )
        lan = start_service(
            client, approved, 0, ("lan",), "sleep 300", image=PROBE_IMAGE
        )
        gateway = _gateway(client, plan.name)
        blocked = {
            "example.com:80 (only :443 listed)": ("example.com", 80),
            "8.8.8.8:53 (a resolver, unlisted)": ("8.8.8.8", 53),
            f"LAN {LAN}:53 (listed, private)": (LAN, 53),
            f"LAN {LAN}:80 (listed, private)": (LAN, 80),
            "metadata 169.254.169.254:80 (listed)": ("169.254.169.254", 80),
            "RFC1918 10.0.0.1:80 (listed)": ("10.0.0.1", 80),
            f"env gateway {gateway}:22 (host INPUT)": (gateway, 22),
            "pypi.org:443 (unlisted)": ("pypi.org", 443),
        }
        outcomes = {
            "DNS example.com": _check(probe, RESOLVE, "example.com"),
            "https://example.com": _check(probe, HTTPS, "https://example.com/"),
            "1.1.1.1:443 (listed IP, any port)": _check(probe, CONNECT, "1.1.1.1", 443),
            **{name: _check(probe, CONNECT, *item) for name, item in blocked.items()},
            f"approved LAN {LAN}:80 (operator CIDR)": _check(lan, CONNECT, LAN, 80),
            # Not asserted: the embedded DNS answers any name (spec: DNS
            # exfiltration is a documented limit of allowlist mode).
            "DNS pypi.org (unlisted)": _check(probe, RESOLVE, "pypi.org"),
        }
        control = _control(
            client, run_id, {**blocked, f"approved LAN {LAN}:80": (LAN, 80)}
        )
        _report(outcomes, control)

        assert outcomes["DNS example.com"] and outcomes["https://example.com"]
        assert outcomes["1.1.1.1:443 (listed IP, any port)"]
        assert any(control.values()), control
        if _real(firewall):
            assert [name for name in blocked if outcomes[name]] == []
            if control[f"approved LAN {LAN}:80"]:
                assert outcomes[f"approved LAN {LAN}:80 (operator CIDR)"]

        # A new answer for a listed name is accepted after one refresh; the
        # rest of the rule is untouched and attests exactly.
        extra["example.com"] = ("1.0.0.1",)
        assert backend.refresh(plan) is True
        backend.attest(plan, made.made[0][1], containers=[probe.id])
        refreshed = {
            "1.0.0.1:443 (refreshed)": _check(probe, CONNECT, "1.0.0.1", 443),
            "1.0.0.1:80 (refreshed, port-limited)": _check(
                probe, CONNECT, "1.0.0.1", 80
            ),
        }
        _report(refreshed)
        assert refreshed["1.0.0.1:443 (refreshed)"]
        if _real(firewall):
            assert not refreshed["1.0.0.1:80 (refreshed, port-limited)"]
    finally:
        made.close()
    assert_no_rules(firewall, run_id)


@pytest.fixture
def build_client():
    if not _selftest():
        _require_root()
    try:
        client = docker.from_env(timeout=120)
        client.ping()
        client.images.get(BUILDKIT)
    except (DockerException, OSError) as error:
        pytest.fail(f"Docker/{BUILDKIT} capability unavailable: {error}")
    try:
        yield client
    finally:
        client.close()


def test_r3_builder_run_steps_reach_public_egress_only(build_client, tmp_path):
    """Spec 8 check 3 / M8: a RUN step on the builder's firewalled bridge,
    and BuildKit's own fetches (ADD <url>, a git source, FROM) through it."""
    run = BuildRun(build_client, tmp_path / "run")
    try:
        work = run.endpoint()
        handle = pulled(work)
        env_id = up(work, env_spec(handle, "public", main=service(handle, SERVER)))
        [child] = build_client.containers.list(
            filters={"label": f"rsi-harness.sandbox-env={env_id}"}
        )
        child.reload()
        [child_ip] = [
            item["IPAddress"]
            for item in child.attrs["NetworkSettings"]["Networks"].values()
        ]
        targets = {
            "metadata": "169.254.169.254:80",
            "rfc1918": "10.0.0.1:80",
            "lan": f"{LAN}:80",
            "child": f"{child_ip}:8080",
            "pypi": "https://pypi.org/simple/",
        }
        arguments = " ".join(f"{name}={value}" for name, value in targets.items())
        view, log = build(
            work,
            "FROM public.ecr.aws/docker/library/python:3.13-slim-bookworm\n"
            "COPY probe.py /probe.py\n"
            f"RUN python3 /probe.py {arguments}\n",
            files={"probe.py": RUN_PROBE},
            no_cache=True,
        )
        assert view["state"] == "succeeded", (view, log[-2000:])
        outcomes = {
            name: state == "open"
            for name, state in re.findall(r"PROBE (\S+) (open|closed)", log)
        }

        [network] = run.labelled(role="sandbox-builder-net")[2]
        gateway = network.attrs["IPAM"]["Config"][0]["Gateway"]
        connects = {
            name: tuple(target.rsplit(":", 1))
            for name, target in {**targets, "gateway": f"{gateway}:22"}.items()
            if not target.startswith("https://")
        }
        control = _control(build_client, run.run_id, connects)
        _report(outcomes, control)

        assert set(outcomes) == {*targets, "gateway"}
        assert outcomes["pypi"]
        if _real(run.firewall):
            assert [name for name, open_ in outcomes.items() if open_] == ["pypi"]

        # BuildKit's own fetches (FROM, ADD <url>, git sources) leave from
        # buildkitd, not from a RUN step, through the same bridge and rule.
        fetches = {
            "ADD https://example.com/": "ADD https://example.com/ /fetched",
            "ADD metadata": "ADD http://169.254.169.254/latest/ /fetched",
            "ADD RFC1918": "ADD http://10.0.0.1/ /fetched",
            f"ADD LAN {LAN}:80": f"ADD http://{LAN}/ /fetched",
            f"ADD gateway {gateway}:22": f"ADD http://{gateway}:22/ /fetched",
            "git source RFC1918": "ADD https://10.0.0.1/probe.git /repo",
            "FROM RFC1918 registry": "FROM 10.0.0.1:5000/probe:1",
        }
        fetched = {}
        for name, line in fetches.items():
            head = "" if line.startswith("FROM") else f"FROM {ALPINE}\n"
            view, log = build(work, f"{head}{line}\n", no_cache=True, timeout_sec=60)
            fetched[name] = fetch_state(view, log)
            print(f"  {fetched[name]}: {name}")

        assert fetched["ADD https://example.com/"] == "fetched"
        if _real(run.firewall):
            assert [
                name
                for name, state in fetched.items()
                if state != "unreachable" and name != "ADD https://example.com/"
            ] == []
        work.env_destroy(env_id)
    finally:
        run.close()


# A fetch that never connected, as BuildKit (Go) and git word it.
UNREACHABLE = re.compile(
    r"connection refused|i/o timeout|no route to host|network is unreachable"
    r"|Failed to connect|Couldn't connect|Connection timed out"
    r"|context deadline exceeded",
    re.IGNORECASE,
)


def fetch_state(view: dict, log: str) -> str:
    """``fetched``; ``unreachable`` (no connection, or none before the
    build's deadline); ``connected`` (the target answered, but not with what
    the build wanted)."""
    if view["state"] == "succeeded":
        return "fetched"
    if view["state"] == "timed_out":
        # A silently dropped SYN (git keeps retrying until the deadline).
        return "unreachable"
    text = f"{log}\n{view.get('error')}"
    return "unreachable" if UNREACHABLE.search(text) else "connected"


# -- R5, R6: root-only kill paths -----------------------------------------------


def test_r5_a_paused_env_is_killed_through_cgroup_kill(labelled_run):
    """Spec 8 check 5 / M3: cgroup.kill of a frozen service; no write after
    the freeze, never thawed, exited and removed."""
    if _selftest():
        pytest.skip("needs root")
    client, run_id = labelled_run
    assert isinstance(default_paused_killer(client.api), CgroupPausedKiller)
    raw = single(BUSYBOX, SENTINEL)
    box = RealEnv(client, run_id, raw, {BUSYBOX: HANDLES[BUSYBOX]}).create().start()
    try:
        container = box.container("main")
        time.sleep(1)
        box.lease = box.backend.pause(box.lease, box.commit)
        frozen_at = uptime()
        attrs = client.api.inspect_container(container.id)
        assert attrs["State"]["Paused"] is True
        cgroup = Path(f"/proc/{attrs['State']['Pid']}/cgroup").read_text()
        assert re.fullmatch(
            r"0::/system\.slice/docker-[0-9a-f]{64}\.scope\n", cgroup
        ), cgroup
        time.sleep(1)

        killer = CgroupPausedKiller()
        settled = killer.kill(container.id, attrs)
        deadline = time.monotonic() + 10
        while not settled():
            assert time.monotonic() < deadline, "cgroup.events never populated 0"
            time.sleep(0.05)
        deadline = time.monotonic() + 10
        while client.api.inspect_container(container.id)["State"]["Status"] != (
            "exited"
        ):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        stream, _ = container.get_archive("/sentinel")
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as archive:
            written = archive.extractfile("sentinel").read().split()
        assert written and float(written[-1]) <= frozen_at
    finally:
        box.destroy()


def test_r5_a_group_kill_spares_a_setsid_process_and_other_execs(exec_run):
    """M4: a kill reaches only the exec's process group (pidfd as root): a
    ``setsid`` process it started and a concurrent exec of the same service
    live on. Needs no root; the root check runs it on the pidfd path."""
    client, run_id, pump, _ = exec_run
    box = ready_env(client, run_id)
    try:
        other = start(pump, box, ["sleep", "1003"])
        exec_id = start(
            pump, box, ["sh", "-c", "setsid sleep 1002 & sleep 1000 & wait"]
        )
        await_processes(
            client,
            box,
            lambda found: all(
                sleeping(found, item) for item in ("1000", "1002", "1003")
            ),
        )

        pump.kill(exec_id, session=SESSION, signal="TERM", scope="group")

        view, _, _ = run_to_end(pump, exec_id, timeout=10)
        assert (view["state"], view["signal"]) == ("killed", "TERM")
        found = await_processes(client, box, lambda found: not sleeping(found, "1000"))
        assert sleeping(found, "1002") and sleeping(found, "1003")
        still = pump.wait(
            other, session=SESSION, stdout_offset=0, stderr_offset=0, wait_sec=0
        )
        assert still["state"] == "running"
    finally:
        destroy(pump, box)


def test_r6_recovery_sees_every_process_of_the_host(labelled_run):
    """M6: as root, coordinator_may_run and firewall_commands_running see
    another user's process in /proc (no hidepid in the way)."""
    if _selftest():
        pytest.skip("needs root")
    _, run_id = labelled_run
    rule_id = f"rsi-{run_id}-sbx-{uuid.uuid4().hex[:16]}"
    chain = firewall_rule_chains(rule_id)[0]
    process = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 30; :", chain], user=65534, group=65534
    )
    started = time.time()
    try:
        time.sleep(0.5)
        assert coordinator_may_run(process.pid, started)
        assert firewall_commands_running((rule_id,))
    finally:
        process.kill()
        process.wait()
    assert not firewall_commands_running((rule_id,))


# -- M7: large copies through the plugin -----------------------------------------


@pytest.mark.asyncio
async def test_m7_a_gib_directory_round_trips_through_the_plugin(
    live_sandbox, tmp_path
):
    """M7: copy_in and copy_out of a >= 1 GiB directory through the Harbor
    plugin (stages, archive API) do not time out. Needs no root; runs with
    the other checks so it meets a loaded root daemon."""
    from harbor.models.task.config import (
        EnvironmentConfig,
        NetworkMode,
        NetworkPolicy,
    )
    from harbor.models.trial.paths import TrialPaths

    from rsi_harness.integrations.sandbox_client import SandboxClient
    from rsi_harness.integrations.sandbox_harbor_env import ManagedSandboxEnvironment

    sandbox = live_sandbox("m7-gib", BUSYBOX)
    endpoint = sandbox.work()
    source = tmp_path / "source"
    source.mkdir()
    digest = hashlib.sha256()
    block = os.urandom(1024**2)
    for index in range(8):
        with open(source / f"part-{index}", "wb") as output:
            for _ in range(128):
                output.write(block)
                digest.update(block)
    environment = tmp_path / "environment"
    environment.mkdir()
    policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    plugin = ManagedSandboxEnvironment(
        environment_dir=environment,
        environment_name="gib",
        session_id="gib__m9__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(
            docker_image=BUSYBOX, cpus=1, memory_mb=512, storage_mb=4096
        ),
        network_policy=policy,
        phase_network_policies=[policy],
        client=SandboxClient(
            str(endpoint.directory / "s"), endpoint.environment["RSI_SANDBOX_TOKEN"]
        ),
    )
    try:
        await plugin.start(force_build=False)
        began = time.monotonic()
        await plugin.upload_dir(source, "/data")
        uploaded = time.monotonic() - began
        target = tmp_path / "target"
        await plugin.download_dir("/data", target)
        copied = time.monotonic() - began - uploaded
        print(f"  1 GiB up {uploaded:.1f}s, down {copied:.1f}s")
    finally:
        await asyncio.shield(plugin.stop(delete=True))
    back = hashlib.sha256()
    for index in range(8):
        with open(target / f"part-{index}", "rb") as data:
            while chunk := data.read(1024**2):
                back.update(chunk)
    assert back.hexdigest() == digest.hexdigest()
    shutil.rmtree(source)
    shutil.rmtree(target)
