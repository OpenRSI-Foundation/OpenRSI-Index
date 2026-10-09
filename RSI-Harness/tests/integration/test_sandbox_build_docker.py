"""Real image builds through the Work and Judge sockets, as a non-root user.

The broker, lifecycle, server, client, BuildKit builder and loader are the
production ones; only the iptables half is faked and the builder's state
filesystem is tmpfs (loop-ext4 needs root). Real egress blocking of a
builder bridge and the loop-ext4 ENOSPC path belong to the operator check
(spec 8, items 3 and 4), which runs this file as root in root mode
(RSI_SANDBOX_ROOT_MODE=1: the real firewall and loop-ext4 state). Every
object carries the run's label and is removed by label after each test, and
the host's own build cache is never touched.
"""

from __future__ import annotations

import io
import ipaddress
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from rsi_harness.integrations.sandbox_client import ProtocolError, SandboxClient
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_env_contracts import (
    builder_loop_file,
    sandbox_spool_root,
)
from rsi_harness.runtime.sandbox_envs import docker_env_runtime
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from tests.fakes import FakeFirewallBackend
from tests.integration.sandbox_support import (
    assert_no_rules,
    root_mode,
    sandbox_firewall,
)
from tests.integration.test_sandbox_env_docker import remove_labelled
from tests.integration.test_sandbox_envs_docker import env_spec, run_exec, service, up
from tests.runtime.test_sandbox_budget import authority
from tests.sandbox_helpers import (
    FakeSandboxBackend,
    env_policy_toml,
    load_policy_text,
    make_env_task,
)

pytestmark = pytest.mark.integration

BUILDKIT = "moby/buildkit:v0.27.1"
# Docker's official image through its AWS mirror: each fresh builder pulls
# its base, and anonymous Docker Hub pulls are rate limited per host. The
# module needs that registry's anonymous quota; a build refused with 429
# skips its test (``build``) instead of failing it.
ALPINE = "public.ecr.aws/docker/library/alpine:3.21"
RATE_LIMITED = re.compile(r"429 Too Many Requests|toomanyrequests", re.IGNORECASE)
TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
build = true
[metadata.rsi_harness.sandbox.environments.judge]
network = ["public", "none"]
pull = true
build = true
"""
# A small tmpfs builder: its state counts against its 1 GiB of memory. Root
# mode builds on the production loop-ext4 state filesystem instead.
BUILD = {
    "state_fs": "loop-ext4" if root_mode() else "tmpfs",
    "cpus": 2,
    "memory_mb": 1024,
    "pids": 1024,
    "disk_mb": 384,
    "max_image_mb": 256,
    "max_images_total_mb": 512,
    "max_context_mb": 16,
    "max_build_sec": 600,
}


def buildx_ids():
    """The host's own build-cache record IDs (never touched, B9)."""
    listed = subprocess.run(
        ["docker", "buildx", "du", "--verbose"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert listed.returncode == 0, listed.stderr
    return sorted(re.findall(r"^ID:\s+(\S+)", listed.stdout, re.MULTILINE))


class BuildRun:
    """One run's broker behind the real Work/Judge endpoint lifecycle."""

    def __init__(self, client, tmp_path, **build):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.client = client
        self.run_id = f"m8-build-{uuid.uuid4().hex[:12]}"
        builder = client.api.inspect_image(BUILDKIT)
        grant = resolve_env_grant(
            make_env_task(TASK),
            load_policy_text(
                tmp_path,
                env_policy_toml(
                    build={
                        **BUILD,
                        "builder_image": builder["RepoDigests"][0],
                        **build,
                    }
                ),
            ),
            {"work": builder, "judge": builder},
            parent_cpus=1,
            parent_memory_mb=256,
        )
        self.firewall = sandbox_firewall(client)
        # A short data root: the endpoint socket path must fit 107 bytes.
        self.data = Path(tempfile.mkdtemp(prefix="rsi-m8-"))
        self.sb = self.data / self.run_id / "sb"
        self.sb.mkdir(mode=0o700, parents=True)
        runtime = docker_env_runtime(
            client,
            self.firewall,
            run_id=self.run_id,
            spool_root=sandbox_spool_root(self.data, self.run_id),
            docker_root=client.info()["DockerRootDir"],
            host=grant.environments.host,
            data_root=self.data,
        )
        self.journal = SandboxJournal(
            authority(LeaseStore(tmp_path / "leases"), self.run_id)
        )
        self.broker = SandboxBroker(
            grant, FakeSandboxBackend(), self.journal, time.monotonic, envs=runtime
        )
        self.lifecycle = SandboxLifecycle()
        self.lifecycle.configure(self.broker, self.sb, self.run_id, "task")

    def endpoint(self, phase="work", round_id=None):
        if phase == "work":
            endpoint = self.lifecycle.prepare_work()
            self.lifecycle.activate_work(time.monotonic() + 1800)
        else:
            endpoint = self.lifecycle.prepare_judge(round_id)
            self.lifecycle.activate_judge(time.monotonic() + 1800)
        return SandboxClient(
            endpoint.directory / "s", endpoint.environment["RSI_SANDBOX_TOKEN"]
        )

    def labelled(self, **labels):
        filters = {
            "label": [f"rsi-harness.run-id={self.run_id}"]
            + [f"rsi-harness.{key}={value}" for key, value in labels.items()]
        }
        return (
            self.client.containers.list(all=True, filters=filters),
            self.client.volumes.list(filters=filters),
            self.client.networks.list(filters=filters),
            self.client.images.list(filters=filters),
        )

    def builder(self):
        [container] = self.labelled(role="sandbox-builder")[0]
        return container

    def close(self):
        try:
            self.lifecycle.close()
            sb_left = self.sb.exists()
        finally:
            shutil.rmtree(self.data, ignore_errors=True)
            errors = remove_labelled(
                self.client, {"label": f"rsi-harness.run-id={self.run_id}"}
            )
            leftovers = self.labelled()
            tags = [
                tag
                for image in self.client.images.list(name="rsi-sbx-img")
                for tag in image.tags
            ]
        assert (errors, leftovers) == ([], ([], [], [], []))
        assert not sb_left
        assert not self.broker.recovery_required
        assert_no_rules(self.firewall, self.run_id)
        assert self.journal.builders() == ()
        assert [image.kind for image in self.journal.images()] == []
        assert not any(self.run_id in tag for tag in tags)


@pytest.fixture
def docker_client():
    try:
        client = docker.from_env(timeout=60)
        client.ping()
        client.images.get(BUILDKIT)
    except (DockerException, OSError) as error:
        message = f"Docker/{BUILDKIT} capability unavailable: {error}"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def build_run(docker_client, tmp_path):
    """``build_run(**build_grant_overrides)``: a BuildRun, closed after."""
    made = []

    def make(**build):
        run = BuildRun(docker_client, tmp_path / f"run{len(made)}", **build)
        made.append(run)
        return run

    yield make
    failures = []
    for run in made:
        # Every run is cleaned up before the first failure is raised.
        try:
            run.close()
        except BaseException as error:
            failures.append(error)
    if failures:
        raise failures[0]


def context_tar(files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def build(client, dockerfile, *, files=None, timeout_sec=300, **options):
    """Stage a context, build it and follow the job; returns (view, log)."""
    staged = client.upload_stage(
        context_tar({"Dockerfile": dockerfile.encode(), **(files or {})})
    )
    job_id = client.image_build(staged["stage_id"], timeout_sec=timeout_sec, **options)
    logs = []
    view = client.follow_job(job_id, logs.append)
    log = "".join(logs)
    if view["state"] == "failed" and RATE_LIMITED.search(log):
        pytest.skip(
            f"registry rate limit while building: {RATE_LIMITED.search(log)[0]}"
        )
    return view, log


def test_alpine_curl_build_runs_in_an_env_by_its_handle(build_run):
    run = build_run()
    before = buildx_ids()
    work = run.endpoint()
    build_caps = work.capabilities()["environments"]["build"]
    assert build_caps["network"] == ["public", "none"]
    assert build_caps["remaining_builds"] == 64
    view, log = build(
        work,
        f"FROM {ALPINE}\n"
        "LABEL rsi-harness.run-id=forged\n"
        "RUN apk add --no-cache curl\n",
    )
    assert view["state"] == "succeeded", (view, log[-2000:])
    image = view["result"]["image"]
    handle = image["handle"]
    assert (image["os"], image["arch"]) == ("linux", "amd64")

    # The loaded image: forced labels over the Dockerfile's, one broker tag.
    [loaded] = run.labelled(role="sandbox-build")[3]
    assert loaded.id == image["image_id"]
    labels = loaded.labels
    assert labels["rsi-harness.run-id"] == run.run_id
    assert labels["rsi-harness.sandbox-image"] == handle
    [tag] = loaded.tags
    assert tag.startswith("rsi-sbx-img:") and tag.endswith(handle[1:])

    # The builder: the documented exception and nothing more (B2).
    builder = run.builder()
    host = builder.attrs["HostConfig"]
    assert host["Runtime"] == "runc" and host["Privileged"] is False
    assert host["CapAdd"] == ["CAP_SYS_ADMIN", "CAP_NET_ADMIN"]
    assert host["SecurityOpt"] == [
        "apparmor=unconfined",
        "seccomp=unconfined",
        "writable-cgroups=true",
    ]
    assert (host["MaskedPaths"], host["ReadonlyPaths"]) == ([], [])
    assert not host["Binds"] and not host["Devices"] and not host["DeviceRequests"]
    assert not host["PortBindings"]
    assert [mount["Destination"] for mount in builder.attrs["Mounts"]] == [
        "/var/lib/buildkit"
    ]
    assert not any("docker.sock" in str(mount) for mount in builder.attrs["Mounts"])
    assert "NVIDIA_VISIBLE_DEVICES=void" in builder.attrs["Config"]["Env"]
    nvidia = builder.exec_run(["sh", "-c", "ls /dev | grep -c nvidia || true"])
    assert nvidia.output.strip() == b"0"
    assert (
        builder.exec_run(["test", "-e", "/run/buildkit/buildkitd.sock"]).exit_code == 0
    )

    env_id = up(work, env_spec(handle, "none", main=service(handle)))
    final, stdout = run_exec(work, env_id, ["curl", "--version"])
    assert (final["state"], final["exit_code"]) == ("exited", 0), final
    assert stdout.startswith(b"curl ")
    with pytest.raises(ProtocolError) as caught:
        work.image_release(handle)
    assert caught.value.code == "busy"
    work.env_destroy(env_id)

    # An identical build of the session answers with the same image (B9).
    again, _ = build(
        work,
        f"FROM {ALPINE}\n"
        "LABEL rsi-harness.run-id=forged\n"
        "RUN apk add --no-cache curl\n",
    )
    assert again["result"]["image"]["handle"] == handle
    work.image_release(handle)
    assert run.labelled(role="sandbox-build")[3] == []
    with pytest.raises(ImageNotFound):
        run.client.images.get(image["image_id"])
    # The Work builder (and its cache) stays for the session.
    assert run.builder().status == "running"
    assert buildx_ids() == before


def test_the_cli_builds_a_context_directory(build_run, monkeypatch, tmp_path, capsys):
    from rsi_harness.integrations import sandbox_client as wire

    run = build_run()
    work = run.endpoint()
    monkeypatch.setenv("RSI_SANDBOX_SOCKET", str(work.socket_path))
    monkeypatch.setenv("RSI_SANDBOX_TOKEN", work.credential)
    context = tmp_path / "ctx"
    (context / "docker").mkdir(parents=True)
    (context / "docker" / "Build.file").write_text(
        f"FROM {ALPINE}\nARG GREETING\nCOPY hello.txt /hello.txt\n"
        'RUN echo "$GREETING" >> /hello.txt\n'
    )
    (context / "hello.txt").write_text("hello\n")
    (context / "link").symlink_to("/etc/passwd")
    code = wire.main(
        [
            "build",
            str(context),
            "-f",
            "docker/Build.file",
            "--build-arg",
            "GREETING=from-the-cli",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert code == 0, result
    handle = result["result"]["image"]["handle"]
    [image] = run.labelled(role="sandbox-build")[3]
    assert image.labels["rsi-harness.sandbox-image"] == handle
    env_id = up(work, env_spec(handle, "none", main=service(handle)))
    final, stdout = run_exec(work, env_id, ["cat", "/hello.txt"])
    assert stdout == b"hello\nfrom-the-cli\n"
    work.env_destroy(env_id)
    assert wire.main(["image-rm", handle]) == 0
    assert run.labelled(role="sandbox-build")[3] == []


def test_run_steps_use_the_builder_bridge_and_its_dns(build_run):
    run = build_run()
    work = run.endpoint()
    view, log = build(
        work,
        f"FROM {ALPINE}\n"
        "RUN ip -4 -o addr show eth0 && cat /etc/resolv.conf"
        " && nslookup registry-1.docker.io\n"
        # S1: no Docker, containerd or BuildKit socket, not even buildkitd's
        # trace collector (unlinked once the builder is ready).
        "RUN echo SOCKETS $(find / \\( -path /proc -o -path /sys \\) -prune"
        " -o -type s -print) END\n",
    )
    assert view["state"] == "succeeded", (view, log[-2000:])
    assert re.search(r"SOCKETS\s+END", log), log
    [network] = run.labelled(role="sandbox-builder-net")[2]
    subnet = ipaddress.ip_network(network.attrs["IPAM"]["Config"][0]["Subnet"])
    addresses = re.findall(r"inet (\d+\.\d+\.\d+\.\d+)/", log)
    assert addresses and all(
        ipaddress.ip_address(address) in subnet for address in addresses
    ), log
    assert "nameserver 127.0.0.11" in log
    assert re.search(r"Name:\s+registry-1\.docker\.io", log), log
    rule_id = (
        f"rsi-{run.run_id}-sbb-"
        f"{run.builder().labels['rsi-harness.sandbox-builder'][1:17]}"
    )
    if isinstance(run.firewall, FakeFirewallBackend):
        assert set(run.firewall.installed) == {rule_id}
    else:
        # Root mode: the host's iptables holds the builder's rule.
        assert run.firewall.exists(rule_id)


def test_entitlements_secrets_ssh_and_frontends_are_refused(build_run):
    run = build_run()
    work = run.endpoint()
    for dockerfile, needle in (
        (f"FROM {ALPINE}\nRUN --network=host true\n", "network.host is not allowed"),
        (
            f"FROM {ALPINE}\nRUN --security=insecure true\n",
            "security.insecure is not allowed",
        ),
        (
            f"FROM {ALPINE}\nRUN --mount=type=secret,id=x,required=true true\n",
            "secret x",
        ),
        (
            f'FROM {ALPINE}\nRUN --mount=type=ssh test -S "$SSH_AUTH_SOCK"\n',
            "did not complete successfully",
        ),
    ):
        view, log = build(work, dockerfile)
        assert view["state"] == "failed", view
        assert view["error"]["kind"] == "dockerfile", view
        assert needle in log, log[-2000:]
    # Every form BuildKit's DetectSyntax honours, not only ``# syntax=``.
    for dockerfile in (
        b"# syntax=ghcr.io/evil/frontend:1\nFROM scratch\n",
        b"#!/bin/sh\n# syntax=ghcr.io/evil/frontend:1\nFROM scratch\n",
        b"//syntax=ghcr.io/evil/frontend:1\nFROM scratch\n",
        b'{"syntax": "ghcr.io/evil/frontend:1"}\n',
    ):
        staged = work.upload_stage(context_tar({"Dockerfile": dockerfile}))
        with pytest.raises(ProtocolError, match="not approved"):
            work.image_build(staged["stage_id"], timeout_sec=60)
        with pytest.raises(ProtocolError, match="not approved"):
            work.image_build(
                work.upload_stage(context_tar({}))["stage_id"],
                dockerfile_inline=dockerfile.decode(),
                timeout_sec=60,
            )
    staged = work.upload_stage(context_tar({"Dockerfile": b"FROM scratch\n"}))
    with pytest.raises(ProtocolError, match="reserved"):
        work.image_build(
            staged["stage_id"], timeout_sec=60, build_args={"BUILDKIT_SYNTAX": "x"}
        )


def test_a_tmpfs_oom_is_oom_and_the_builder_survives(build_run):
    run = build_run()
    work = run.endpoint()
    view, log = build(
        work,
        f"FROM {ALPINE}\n"
        "RUN --mount=type=tmpfs,target=/t dd if=/dev/zero of=/t/x bs=1M count=1500\n",
    )
    assert view["state"] == "failed", view
    assert view["error"]["kind"] == "oom", (view, log[-2000:])
    builder = run.builder()
    assert builder.status == "running"
    ok, log = build(work, f"FROM {ALPINE}\nRUN echo still-here\n")
    assert ok["state"] == "succeeded", (ok, log[-2000:])
    assert run.builder().id == builder.id


def test_a_full_state_fs_is_disk_and_prune_frees_it(build_run):
    run = build_run(disk_mb=256)
    work = run.endpoint()
    view, log = build(
        work, f"FROM {ALPINE}\nRUN dd if=/dev/zero of=/big bs=1M count=400\n"
    )
    assert view["state"] == "failed", view
    assert view["error"]["kind"] == "disk", (view, log[-2000:])
    builder = run.builder()
    assert builder.status == "running"
    # The failed RUN's snapshot is pruned after the job ended (it holds
    # the builder until then): the fs is far from full again.
    deadline = time.monotonic() + 60
    while True:
        used = builder.exec_run(["sh", "-c", "df -m /var/lib/buildkit | tail -1"])
        if int(used.output.split()[2]) < 64 or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    assert int(used.output.split()[2]) < 64, used.output
    again, log = build(work, f"FROM {ALPINE}\nRUN echo fits\n")
    assert again["state"] == "succeeded", (again, log[-2000:])


def test_a_timed_out_run_step_ends_within_seven_seconds(build_run):
    run = build_run()
    work = run.endpoint()
    view, log = build(work, f"FROM {ALPINE}\nRUN echo pulled\n")
    assert view["state"] == "succeeded", (view, log[-2000:])
    began = time.monotonic()
    view, log = build(work, f"FROM {ALPINE}\nRUN sleep 600\n", timeout_sec=5)
    assert time.monotonic() - began < 7
    assert view["state"] == "timed_out", view
    assert view["error"]["kind"] == "timeout"
    processes = run.builder().exec_run(["ps", "-o", "args"]).output.decode()
    assert "sleep 600" not in processes and "buildctl build" not in processes


def test_an_image_over_max_image_mb_is_quota_and_never_loaded(build_run):
    run = build_run(max_image_mb=1)
    work = run.endpoint()
    view, log = build(work, f"FROM {ALPINE}\nRUN apk add --no-cache curl\n")
    assert view["state"] == "failed", view
    assert view["error"]["kind"] == "quota", (view, log[-2000:])
    assert run.labelled(role="sandbox-build")[3] == []
    assert run.journal.images() == ()


def test_a_network_none_build_cannot_reach_the_internet(build_run):
    run = build_run()
    work = run.endpoint()
    probe = (
        f"FROM {ALPINE}\n"
        "RUN wget -T 5 -q -O /dev/null http://example.com"
        " && echo NET-OK || echo NET-FAIL\n"
    )
    # BuildKit pulls the base over the builder bridge either way.
    printed = re.compile(r"^#\d+ [\d.]+ (NET-OK|NET-FAIL)$", re.MULTILINE)
    offline, log = build(work, probe, network="none")
    assert offline["state"] == "succeeded", (offline, log[-2000:])
    assert printed.findall(log) == ["NET-FAIL"], log[-2000:]
    online, log = build(work, probe, network="public", no_cache=True)
    assert online["state"] == "succeeded", (online, log[-2000:])
    assert printed.findall(log) == ["NET-OK"], log[-2000:]


def test_gpu_devices_never_reach_a_run_step_or_the_env(build_run):
    run = build_run()
    work = run.endpoint()
    view, log = build(
        work,
        f"FROM {ALPINE}\n"
        "ENV NVIDIA_VISIBLE_DEVICES=all\n"
        "RUN (ls /dev | grep nvidia || true) > /nvidia\n",
    )
    assert view["state"] == "succeeded", (view, log[-2000:])
    handle = view["result"]["image"]["handle"]
    env_id = up(work, env_spec(handle, "none", main=service(handle)))
    final, stdout = run_exec(
        work, env_id, ["sh", "-c", "cat /nvidia; ls /dev | grep nvidia; true"]
    )
    assert (final["exit_code"], stdout) == (0, b"")
    work.env_destroy(env_id)


def test_close_judge_removes_the_round_builder_before_work_resumes(build_run):
    run = build_run()
    work = run.endpoint()
    view, log = build(work, f"FROM {ALPINE}\nRUN echo work\n")
    assert view["state"] == "succeeded", (view, log[-2000:])
    work_builder = run.builder()
    run.lifecycle.freeze_work()
    # An idle Work builder keeps running: no caller code runs in it.
    work_builder.reload()
    assert work_builder.status == "running"

    judge = run.endpoint("judge", "round-1")
    view, log = build(judge, f"FROM {ALPINE}\nRUN echo judge\n")
    assert view["state"] == "succeeded", (view, log[-2000:])
    assert run.labelled(**{"round-id": "round-1"}) != ([], [], [], [])
    began = time.monotonic()
    run.lifecycle.close_judge()
    assert time.monotonic() - began < 60
    assert run.labelled(**{"round-id": "round-1"}) == ([], [], [], [])
    assert not run.broker.recovery_required
    run.lifecycle.resume_work()
    run.lifecycle.reopen_work()
    # Work's builder and cache survive the round.
    assert run.builder().id == work_builder.id
    again, log = build(work, f"FROM {ALPINE}\nRUN echo work\n", no_cache=False)
    assert again["state"] == "succeeded", (again, log[-2000:])


def judge_load(run, docker_client, monkeypatch, size_mib, at):
    """Start a Judge build of a ``size_mib`` random layer; ``at(load, data)``
    runs after each send of its load (data: the bytes sent) and returns
    True once the test should close the round. Returns the Judge's image
    lease and the load at that moment."""
    import threading

    from rsi_harness.runtime import sandbox_build

    run.endpoint()
    run.lifecycle.freeze_work()
    judge = run.endpoint("judge", "round-1")
    reached = threading.Event()
    loads = []
    send = sandbox_build.ImageLoad._send

    def watched(self, sock, data):
        send(self, sock, data)
        if self not in loads:
            loads.append(self)
        if not reached.is_set() and at(self, data):
            reached.set()

    monkeypatch.setattr(sandbox_build.ImageLoad, "_send", watched)
    staged = judge.upload_stage(
        context_tar(
            {
                "Dockerfile": (
                    f"FROM {ALPINE}\n"
                    f"RUN head -c {size_mib * 1048576} /dev/urandom > /blob\n"
                ).encode()
            }
        )
    )
    job_id = judge.image_build(staged["stage_id"], timeout_sec=600)
    if not reached.wait(600):
        logs = []
        view = judge.follow_job(job_id, logs.append)
        if RATE_LIMITED.search("".join(logs)):
            pytest.skip("registry rate limit while building")
        pytest.fail(f"the load never reached its moment: {view}")
    [lease] = [item for item in run.journal.images() if item.owner.phase == "judge"]
    [load] = loads
    return lease, load


def assert_no_judge_image(run, docker_client, digest, *, watch_sec):
    assert not run.broker.recovery_required
    assert [item for item in run.journal.images() if item.owner.phase == "judge"] == []
    # Nothing registers late either: the daemon is done with the load.
    deadline = time.monotonic() + watch_sec
    while True:
        if digest is not None:
            with pytest.raises(ImageNotFound):
                docker_client.images.get(digest)
        assert run.labelled(**{"round-id": "round-1"}) == ([], [], [], [])
        if time.monotonic() >= deadline:
            break
        time.sleep(0.5)
    assert not any(
        run.run_id in tag
        for image in docker_client.images.list("rsi-sbx-img")
        for tag in image.tags
    )


def test_close_judge_cuts_a_real_load_mid_stream_and_leaves_no_image(
    build_run, docker_client, monkeypatch
):
    """A Judge close once 16 MiB of a 64 MiB image's load reached the daemon
    cuts the stream (its end never goes out), ends within KILL_SEC, and the
    daemon refuses the truncated archive: no image, tag or record is left.
    (The config, so the journaled digest, may come later in the stream.)"""
    from rsi_harness.runtime.sandbox_envs import KILL_SEC

    run = build_run()
    sent = [0]

    def at(load, data):
        sent[0] += len(data)
        return load.closed or sent[0] >= 16 * 1048576

    lease, load = judge_load(run, docker_client, monkeypatch, 64, at)
    began = time.monotonic()
    run.lifecycle.close_judge()
    assert time.monotonic() - began < KILL_SEC
    assert load.closed and not load.final and not load.answered
    assert_no_judge_image(run, docker_client, lease.image_id, watch_sec=5)


def test_close_judge_waits_for_a_real_load_s_answer_and_leaves_no_image(
    build_run, docker_client, monkeypatch
):
    """A Judge close right after a 512 MiB image's whole stream went out:
    dockerd keeps loading it after a client disconnects, registering it
    seconds later, so the close waits for the answer (the load thread really
    blocks on the daemon) and then removes the image with proof. Nothing
    registers after close_judge returns."""
    from rsi_harness.runtime.sandbox_envs import DELETE_SEC

    run = build_run(
        memory_mb=3072,
        disk_mb=1536,
        max_image_mb=1024,
        max_images_total_mb=1024,
    )

    def at(load, data):
        return data == b"0\r\n\r\n"

    lease, load = judge_load(run, docker_client, monkeypatch, 512, at)
    assert lease.state == "loading"
    began = time.monotonic()
    run.lifecycle.close_judge()
    assert time.monotonic() - began < DELETE_SEC
    assert load.closed and load.final and load.answered
    assert_no_judge_image(run, docker_client, lease.image_id, watch_sec=20)


# -- kill -9 and recover -------------------------------------------------------


def _hold_build_until_killed(root: str, run_id: str, gate: str, connection) -> None:
    """The coordinator: one broker with a build, stopped forever at ``gate``."""
    import threading

    from rsi_harness.runtime import sandbox_build
    from rsi_harness.runtime.recovery import ResourceLease
    from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool
    from rsi_harness.runtime.sandbox_contracts import SandboxOwner

    base = Path(root)
    client = docker.from_env(timeout=60)
    store = LeaseStore(base / "leases")

    def stop_here(what):
        connection.send((what, os.getpid()))
        connection.close()
        threading.Event().wait()

    with store.lock(run_id):
        current = ResourceLease(
            run_id=run_id,
            task_id="task",
            coordinator_pid=os.getpid(),
            coordinator_started_at=time.time(),
            phase="agent_running",
        )
        store.write(current)
        lock = threading.RLock()

        def mutate(transform):
            nonlocal current
            with lock:
                updated = transform(current)
                if updated is not current:
                    store.write(updated)
                    current = updated
                return current

        builder = client.api.inspect_image(BUILDKIT)
        grant = resolve_env_grant(
            make_env_task(TASK),
            load_policy_text(
                base,
                env_policy_toml(
                    build={**BUILD, "builder_image": builder["RepoDigests"][0]}
                ),
            ),
            {"work": builder, "judge": builder},
            parent_cpus=1,
            parent_memory_mb=256,
        )
        SandboxAdmissionPool(store).reserve_run(run_id, grant, mutate)
        runtime = docker_env_runtime(
            client,
            sandbox_firewall(client),
            run_id=run_id,
            spool_root=sandbox_spool_root(base / "managed", run_id),
            docker_root=client.info()["DockerRootDir"],
            host=grant.environments.host,
            data_root=base / "managed",
        )
        broker = SandboxBroker(
            grant,
            FakeSandboxBackend(),
            SandboxJournal(mutate),
            time.monotonic,
            envs=runtime,
        )
        credential = broker.open_session(
            SandboxOwner(run_id=run_id, task_id="task", phase="work"), None
        ).credential
        broker.activate_work(time.monotonic() + 600)
        api = client.api
        if gate == "builder":
            create = api.create_container_from_config

            def gated(config, name=None):
                create(config, name=name)
                stop_here("builder")

            api.create_container_from_config = gated
        if gate == "load":
            # Journaled digest, stream short of its end: nothing registered.
            original = sandbox_build.BuildService._commit_image

            def gated_commit(self, build, **changes):
                lease = original(self, build, **changes)
                if changes.get("state") == "loading":
                    stop_here("load")
                return lease

            sandbox_build.BuildService._commit_image = gated_commit
        if gate == "loaded":
            # Loaded but never tagged: the journaled digest (and the forced
            # labels) name it; the digest-only path is U-tested.
            api.tag = lambda *args, **kwargs: stop_here("loaded")
        data = context_tar(
            {
                "Dockerfile": (
                    f"FROM {ALPINE}\nRUN sleep 600\n"
                    if gate == "build"
                    else f"FROM {ALPINE}\nRUN echo {run_id} > /id\n"
                ).encode()
            }
        )
        import hashlib

        stage = broker.stage_put(
            credential, None, 0, True, hashlib.sha256(data).hexdigest(), "s-1", data
        )
        job_id = broker.image_build(
            credential,
            stage_id=stage["stage_id"],
            dockerfile=None,
            dockerfile_inline=None,
            target=None,
            build_args={},
            labels={},
            no_cache=False,
            network="public",
            timeout_sec=300,
            request_id="build-1",
        )["job_id"]
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            view = broker.job_wait(credential, job_id, 0, 0)
            if gate == "build" and "sleep 600" in view["log"] and "#5 " in view["log"]:
                stop_here("build")
            if view["state"] not in ("queued", "running"):
                break
            time.sleep(0.05)
        connection.send(("build finished", os.getpid()))


@pytest.mark.parametrize("gate", ["builder", "build", "load", "loaded"])
def test_kill_9_mid_build_then_recover_leaves_nothing(docker_client, tmp_path, gate):
    import multiprocessing
    import signal

    from rsi_harness.runtime.production import ProductionRecoveryBackend
    from rsi_harness.runtime.recovery import RecoveryManager
    from tests.integration.sandbox_support import EmptySnapshotRecovery

    run_id = f"m8-recover-{uuid.uuid4().hex[:12]}"
    # A short root: the builder loop directory and socket paths stay short.
    root = Path(tempfile.mkdtemp(prefix="rsi-m8r-"))
    store = LeaseStore(root / "leases")
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    coordinator = context.Process(
        target=_hold_build_until_killed, args=(str(root), run_id, gate, sender)
    )
    coordinator.start()
    sender.close()
    filters = {"label": f"rsi-harness.run-id={run_id}"}

    def labelled():
        return (
            docker_client.containers.list(all=True, filters=filters),
            docker_client.volumes.list(filters=filters),
            docker_client.networks.list(filters=filters),
            docker_client.images.list(filters=filters),
        )

    try:
        assert receiver.poll(300), f"no gate reached; exit={coordinator.exitcode}"
        assert receiver.recv() == (gate, coordinator.pid)
        os.kill(coordinator.pid, signal.SIGKILL)
        coordinator.join(10)
        assert coordinator.exitcode == -signal.SIGKILL

        crashed = store.read(run_id)
        (builder,) = crashed.sandbox_builders
        containers, volumes, networks, images = labelled()
        assert len(volumes) == len(networks) == 1
        digests = [image.image_id for image in crashed.sandbox_images]
        if gate == "builder":
            # The container exists; its create never answered the broker.
            assert (builder.state, builder.container_id) == ("planned", None)
            assert len(containers) == 1
        else:
            assert builder.state == "running"
            assert [item.status for item in containers] == ["running"]
        if gate == "load":
            (image,) = crashed.sandbox_images
            assert image.state == "loading" and image.image_id is not None
            # Journaled before the end of the stream: nothing registered.
            assert images == []
        if gate == "loaded":
            (image,) = crashed.sandbox_images
            assert image.state == "loading"
            # Loaded, untagged: no tag, only its digest and forced labels.
            assert [item.id for item in images] == [image.image_id]
            assert images[0].tags == []

        firewall = sandbox_firewall(docker_client)
        backend = ProductionRecoveryBackend(
            docker_client, EmptySnapshotRecovery(), firewall
        )
        recovered = RecoveryManager(
            store=store, backend=backend, managed_root=root / "managed"
        ).recover(run_id)

        assert recovered == (run_id,)
        assert labelled() == ([], [], [], [])
        for digest in digests:
            if digest is not None:
                with pytest.raises(ImageNotFound):
                    docker_client.images.get(digest)
        final = store.read(run_id)
        assert (final.sandbox_builders, final.sandbox_images) == ((), ())
        assert not final.recovery_required
        assert final.sandbox_reservation is None
        assert not (root / "managed" / run_id / "sb").exists() or not any(
            (root / "managed" / run_id / "sb").iterdir()
        )
        assert_no_rules(firewall, run_id, rule_ids=[builder.rule_id])
        loop_file = builder_loop_file(root / "managed", run_id, builder.builder_id)
        assert not loop_file.exists()
        if builder.state_fs == "loop-ext4":
            # Root mode: the loop device was detached (losetup -j is empty).
            attached = subprocess.run(
                ["losetup", "-j", str(loop_file)], capture_output=True, text=True
            )
            assert attached.stdout == ""
    finally:
        receiver.close()
        if coordinator.is_alive():
            coordinator.kill()
            coordinator.join(10)
        remove_labelled(docker_client, filters)
        shutil.rmtree(root, ignore_errors=True)
