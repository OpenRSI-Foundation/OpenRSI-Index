"""Real brokered envs as a non-root user; only the iptables half is faked.

Real firewall blocking, cgroup.kill of a paused env and the intra-bridge
accept need root and are covered by the operator script, which also runs
this file as root in root mode (RSI_SANDBOX_ROOT_MODE=1: the real firewall;
the backend's paused killer is then cgroup.kill).
"""

import hashlib
import io
import os
import tarfile
import time
import uuid

import docker
import pytest
from docker.errors import DockerException, NotFound

from rsi_harness.runtime import sandbox_env_contracts as env
from rsi_harness.runtime.sandbox_archive import (
    MAX_STAGE_FRAME,
    ArchiveTransfer,
    StageStore,
)
from rsi_harness.runtime.sandbox_contracts import SandboxError, SandboxOwner
from rsi_harness.runtime.sandbox_disk import (
    DiskWatchdog,
    DockerDiskProbe,
    EnvDiskBudget,
)
from rsi_harness.runtime.sandbox_env_docker import SandboxEnvDockerBackend
from rsi_harness.runtime.sandbox_network import SandboxNetworkBackend
from tests.integration.sandbox_support import assert_no_rules, sandbox_firewall

pytestmark = pytest.mark.integration

MIB = 1024**2
BUSYBOX = "busybox:1.37.0"
UBUNTU = "ubuntu:24.04"
HANDLES = {BUSYBOX: "i" + "b" * 32, UBUNTU: "i" + "c" * 32}
SERVER = "mkdir -p /www && echo ok > /www/index.html && exec httpd -f -p 8080 -h /www"


@pytest.fixture
def docker_run(tmp_path):
    try:
        client = docker.from_env(timeout=60)
        client.ping()
        for image in (BUSYBOX, UBUNTU):
            client.images.get(image)
    except (DockerException, OSError) as error:
        message = f"Docker/{BUSYBOX}/{UBUNTU} capability unavailable: {error}"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    run_id = f"m3-env-{uuid.uuid4().hex[:12]}"
    stages = StageStore(tmp_path / "spool")
    try:
        yield client, run_id, stages
    finally:
        stages.close()
        filters = {"label": f"rsi-harness.run-id={run_id}"}
        errors = remove_labelled(client, filters)
        leftovers = (
            client.containers.list(all=True, filters=filters),
            client.volumes.list(filters=filters),
            client.networks.list(filters=filters),
            client.images.list(filters=filters),
        )
        client.close()
        assert (errors, leftovers) == ([], ([], [], [], []))


def remove_labelled(client, filters):
    """Remove every labelled object; one failure never skips the rest."""
    kinds = (
        (
            lambda: client.containers.list(all=True, filters=filters),
            lambda item: item.remove(force=True, v=True),
        ),
        (
            lambda: client.volumes.list(filters=filters),
            lambda item: item.remove(force=True),
        ),
        (lambda: client.networks.list(filters=filters), lambda item: item.remove()),
        (
            lambda: client.images.list(filters=filters),
            lambda item: client.images.remove(item.id, force=True),
        ),
    )
    errors = []
    for listing, remove in kinds:
        try:
            items = listing()
        except DockerException as error:
            errors.append(error)
            continue
        for item in items:
            try:
                remove(item)
            except NotFound:
                pass
            except DockerException as error:
                errors.append(error)
    return errors


class RealEnv:
    """One env driven through the backend exactly as the broker will."""

    def __init__(self, client, run_id, raw, images, *, stages=None, swap_ratio=1.0):
        self.client = client
        self.firewall = sandbox_firewall(client)
        self.backend = SandboxEnvDockerBackend(
            client,
            SandboxNetworkBackend(client, self.firewall),
            poll_interval=0.2,
        )
        self.owner = SandboxOwner(
            run_id=run_id, task_id="sandbox-env", phase="judge", round_id="round-1"
        )
        self.env_id = "e" + uuid.uuid4().hex
        attrs = {
            handle: self.backend.inspect_image(client.images.get(image).id)
            for image, handle in images.items()
        }
        self.plan = self.backend.plan(
            self.owner,
            self.env_id,
            env.parse_env_spec(raw),
            attrs,
            default_pids=256,
            swap_ratio=swap_ratio,
        )
        now = time.time()
        self.journal = []
        self.lease = self.commit(self.plan.lease(created_at=now, expires_at=now + 900))
        self.transfer = ArchiveTransfer(client, stages) if stages else None

    def commit(self, lease):
        self.journal.append(lease)
        return lease

    def create(self):
        self.lease = self.backend.create(self.plan, self.lease, self.commit)
        return self

    def start(self, timeout=60):
        result = self.backend.start(
            self.plan, self.lease, self.commit, wait_timeout_sec=timeout
        )
        self.lease = result.lease
        assert result.state == "ready", result
        return self

    def container(self, name):
        record = next(item for item in self.lease.services if item.name == name)
        return self.client.containers.get(record.container_id)

    def target(self, name):
        return self.backend.archive_target(self.lease, name)

    def destroy(self):
        self.lease = self.backend.destroy(self.lease, self.commit)
        assert_no_rules(
            self.firewall,
            self.owner.run_id,
            rule_id=env.env_rule_id(self.owner.run_id, self.env_id),
        )
        assert labelled(self.client, "rsi-harness.sandbox-env", self.env_id) == (
            [],
            [],
            [],
        )


def labelled(client, key, value):
    filters = {"label": f"{key}={value}"}
    return (
        client.containers.list(all=True, filters=filters),
        client.volumes.list(filters=filters),
        client.networks.list(filters=filters),
    )


def service(image, command, **values):
    return {
        "image": HANDLES[image],
        "command": command,
        "cpus": 0.5,
        "memory_mb": 128,
        "pids": 128,
        **values,
    }


def single(image, command="sleep 600", *, disk_mb=512, **values):
    return {
        "version": 1,
        "network": "none",
        "disk_mb": disk_mb,
        "services": {"main": service(image, ["sh", "-c", command], **values)},
    }


def stage(stages, files):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    raw = buffer.getvalue()
    stage_id, offset = None, 0
    frame = 4 * 1024**2
    while True:
        chunk = raw[offset : offset + frame]
        final = offset + len(chunk) >= len(raw)
        result = stages.put(
            stage_id,
            offset,
            chunk,
            final=final,
            sha256=hashlib.sha256(raw).hexdigest() if final else None,
            max_bytes=1 << 30,
        )
        stage_id, offset = result["stage_id"], offset + len(chunk)
        if final:
            return stage_id


def download(stages, stage_id):
    data = bytearray()
    while True:
        chunk = stages.read(stage_id, len(data), MAX_STAGE_FRAME)
        if not chunk:
            return bytes(data)
        data += chunk


def run(container, command):
    result = container.exec_run(["sh", "-c", command])
    return result.exit_code, result.output.decode()


def test_real_writable_rootfs_accepts_writes_and_copies_to_app_etc_usr(docker_run):
    client, run_id, stages = docker_run
    ubuntu = RealEnv(
        client, run_id, single(UBUNTU), {UBUNTU: HANDLES[UBUNTU]}, stages=stages
    )
    ubuntu.create().start()
    try:
        main = ubuntu.container("main")
        code, output = run(
            main,
            "mkdir -p /app && echo a > /app/w && echo b > /etc/w && echo c > /usr/w"
            " && cat /app/w /etc/w /usr/w",
        )
        assert (code, output) == (0, "a\nb\nc\n")
        for directory in ("/app", "/etc", "/usr"):
            copied = ubuntu.transfer.copy_in(
                ubuntu.target("main"),
                directory,
                stage(stages, {"copied": directory.encode()}),
            )
            assert copied == {"entries": 1, "bytes": len(directory)}
            assert run(main, f"cat {directory}/copied") == (0, directory)
        attrs = main.attrs
        assert attrs["HostConfig"]["Runtime"] == "runc"
        assert attrs["HostConfig"]["Init"] is True
        assert attrs["HostConfig"]["Binds"] is None
        assert attrs["HostConfig"]["LogConfig"]["Type"] == "json-file"
        assert "NVIDIA_VISIBLE_DEVICES=void" in attrs["Config"]["Env"]
        assert attrs["AppArmorProfile"] == "docker-default"
        assert attrs["NetworkSettings"]["Networks"].keys() == {"none"}
    finally:
        ubuntu.destroy()


@pytest.mark.parametrize("swap_ratio", [1.0, 0.0])
def test_real_swap_is_the_grant_ratio_of_memory(docker_run, swap_ratio):
    """A 128 MiB service fills a 192 MiB buffer only when it may swap."""
    client, run_id, stages = docker_run
    box = RealEnv(
        client,
        run_id,
        single(BUSYBOX),
        {BUSYBOX: HANDLES[BUSYBOX]},
        stages=stages,
        swap_ratio=swap_ratio,
    )
    box.create().start()
    try:
        main = box.container("main")
        host = main.attrs["HostConfig"]
        swap = int(128 * swap_ratio) * MIB
        assert (host["Memory"], host["MemorySwap"]) == (128 * MIB, 128 * MIB + swap)
        assert host["MemorySwappiness"] is None
        box.backend.attest(box.plan, box.lease)
        limits = run(
            main, "cat /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.swap.max"
        )
        assert limits == (0, f"{128 * MIB}\n{swap}\n")
        code, output = run(
            main,
            "dd if=/dev/zero of=/dev/null bs=192M count=1 2>/dev/null; echo $?;"
            " grep oom_kill /sys/fs/cgroup/memory.events",
        )
        assert code == 0
        if swap_ratio:
            assert output == "0\noom_kill 0\n"
        else:
            assert output == "137\noom_kill 1\n"
    finally:
        box.destroy()


def test_real_thousand_files_round_trip_through_an_image_without_python(docker_run):
    client, run_id, stages = docker_run
    box = RealEnv(
        client, run_id, single(BUSYBOX), {BUSYBOX: HANDLES[BUSYBOX]}, stages=stages
    )
    box.create().start()
    try:
        main = box.container("main")
        assert run(main, "command -v python3")[0] != 0
        files = {
            f"d{index % 10}/sub{index % 7}/file-{index:04d}.bin": os.urandom(
                5 * 1024 + index % 17
            )
            for index in range(1000)
        }
        assert 5_000_000 < sum(map(len, files.values())) < 5_500_000
        stage_id = stage(stages, files)

        copied = box.transfer.copy_in(box.target("main"), "/work/in", stage_id)

        assert copied["bytes"] == sum(map(len, files.values()))
        inside = run(main, "cd /work/in && find . -type f | wc -l")
        assert inside == (0, "1000\n")
        result = box.transfer.copy_out(
            box.target("main"), "/work/in", max_bytes=16 << 20
        )
        assert (result["skipped"], result["bytes"]) == (0, copied["bytes"])
        with tarfile.open(
            fileobj=io.BytesIO(download(stages, result["stage_id"]))
        ) as tar:
            returned = {
                member.name.removeprefix("in/"): tar.extractfile(member).read()
                for member in tar
                if member.isreg()
            }
        assert returned == files
    finally:
        box.destroy()


def test_real_tmpfs_targets_are_unsupported_and_paused_services_busy(docker_run):
    client, run_id, stages = docker_run
    raw = single(BUSYBOX, tmpfs={"/scratch": 16})
    box = RealEnv(client, run_id, raw, {BUSYBOX: HANDLES[BUSYBOX]}, stages=stages)
    box.create().start()
    try:
        main = box.container("main")
        assert run(main, "mkdir -p /app && ln -s /scratch /app/scratch")[0] == 0
        for destination in ("/scratch", "/scratch/sub", "/dev/shm", "/app/scratch"):
            with pytest.raises(SandboxError) as caught:
                box.transfer.copy_in(
                    box.target("main"), destination, stage(stages, {"a": b"a"})
                )
            assert caught.value.code == "unsupported", destination
        # An entry below the destination that goes through a container link
        # into the tmpfs would land in the layer under it (VERIFIED).
        for destination, entry in (("/", "app/scratch/x"), ("/app", "scratch/x")):
            with pytest.raises(SandboxError) as caught:
                box.transfer.copy_in(
                    box.target("main"), destination, stage(stages, {entry: b"x"})
                )
            assert caught.value.code == "unsupported", entry
        for path in ("/scratch", "/app/scratch/x"):
            with pytest.raises(SandboxError) as caught:
                box.transfer.copy_out(box.target("main"), path, max_bytes=1024)
            assert caught.value.code == "unsupported", path
        # A link to an ordinary directory is followed, as the daemon does.
        assert run(main, "mkdir -p /opt/real && ln -s /opt/real /app/real")[0] == 0
        box.transfer.copy_in(
            box.target("main"), "/app", stage(stages, {"real/y": b"y"})
        )
        assert run(main, "cat /opt/real/y") == (0, "y")

        box.lease = box.backend.pause(box.lease, box.commit)
        assert main.reload() is None and main.attrs["State"]["Paused"]
        with pytest.raises(SandboxError) as caught:
            box.transfer.copy_in(box.target("main"), "/tmp", stage(stages, {"a": b"a"}))
        assert caught.value.code == "busy"
        with pytest.raises(SandboxError) as caught:
            box.transfer.copy_out(box.target("main"), "/etc", max_bytes=1 << 20)
        assert caught.value.code == "busy"
        with pytest.raises(SandboxError) as caught:
            box.backend.stop_service(box.lease, box.commit, "main", timeout_sec=1)
        assert caught.value.code == "busy"
    finally:
        # Teardown terminates the still-paused service (docker kill as non-root).
        box.destroy()


def test_real_first_entry_names_that_look_compressed_round_trip(docker_run):
    """The daemon sniffs put_archive bodies for compression magic: a ustar
    tar whose first name starts with "BZh" fails as bzip2 (VERIFIED)."""
    client, run_id, stages = docker_run
    box = RealEnv(
        client, run_id, single(BUSYBOX), {BUSYBOX: HANDLES[BUSYBOX]}, stages=stages
    )
    box.create().start()
    try:
        name = "BZh91AY&SYfile"
        box.transfer.copy_in(
            box.target("main"), "/work", stage(stages, {name: b"plain text\n"})
        )
        assert run(box.container("main"), f"cat '/work/{name}'") == (0, "plain text\n")
        result = box.transfer.copy_out(box.target("main"), "/work", max_bytes=1024)
        with tarfile.open(
            fileobj=io.BytesIO(download(stages, result["stage_id"]))
        ) as tar:
            assert tar.extractfile(f"work/{name}").read() == b"plain text\n"
    finally:
        box.destroy()


def test_real_seeds_before_start_are_visible_after_start(docker_run):
    client, run_id, stages = docker_run
    raw = single(
        BUSYBOX,
        "cat /data/seed.txt /etc/seed.conf > /tmp/seen && sleep 600",
        mounts=[{"volume": "shared", "target": "/data"}],
    )
    raw["volumes"] = {"shared": {"seeded": True}}
    box = RealEnv(client, run_id, raw, {BUSYBOX: HANDLES[BUSYBOX]}, stages=stages)
    box.create()
    try:
        assert box.container("main").attrs["State"]["Status"] == "created"
        target = box.target("main")
        box.transfer.copy_in(
            target, "/data", stage(stages, {"seed.txt": b"volume seed\n"})
        )
        box.transfer.copy_in(
            target, "/etc", stage(stages, {"seed.conf": b"rootfs seed\n"})
        )
        assert (
            box.transfer.path_stat(target, "/etc/seed.conf", follow=True)["kind"]
            == "file"
        )

        box.start()

        seen = run(box.container("main"), "cat /tmp/seen")
        assert seen == (0, "volume seed\nrootfs seed\n")
    finally:
        box.destroy()


def test_real_services_reach_each_other_by_alias_after_a_health_wait(docker_run):
    client, run_id, stages = docker_run
    raw = {
        "version": 1,
        "network": "public",
        "disk_mb": 512,
        "services": {
            # kv serves (and turns healthy) only 2 s after it starts.
            "kv": service(
                BUSYBOX,
                ["sh", "-c", f"sleep 2 && {SERVER}"],
                aliases=["kvstore"],
                healthcheck={
                    "test": ["CMD", "wget", "-qO-", "http://127.0.0.1:8080/"],
                    "interval_sec": 0.5,
                    "start_interval_sec": 0.5,
                    "retries": 20,
                },
            ),
            "main": service(
                BUSYBOX,
                ["sh", "-c", SERVER],
                depends_on={"kv": {"condition": "healthy"}},
                extra_hosts=[["mirror.internal", "203.0.113.7"]],
            ),
        },
    }
    pair = RealEnv(client, run_id, raw, {BUSYBOX: HANDLES[BUSYBOX]})
    pair.create()
    assert labelled(client, "rsi-harness.sandbox-env", pair.env_id)[2]
    since = int(time.time()) - 1
    try:
        pair.start()
        kv, main = pair.container("kv"), pair.container("main")
        assert kv.attrs["State"]["Health"]["Status"] == "healthy"
        # The daemon's own event times: main started after kv turned healthy.
        events = list(
            client.api.events(
                since=since,
                until=int(time.time()) + 1,
                decode=True,
                filters={"label": f"rsi-harness.sandbox-env={pair.env_id}"},
            )
        )
        healthy = [
            event["timeNano"]
            for event in events
            if event.get("Action") == "health_status: healthy"
            and event["Actor"]["ID"] == kv.id
        ]
        started = [
            event["timeNano"]
            for event in events
            if event.get("Action") == "start" and event["Actor"]["ID"] == main.id
        ]
        assert healthy and started and min(started) > min(healthy)
        for host in ("kvstore", "kv"):
            code, output = run(main, f"wget -qO- -T 5 http://{host}:8080/")
            assert (code, output) == (0, "ok\n"), host
        assert run(kv, "wget -qO- -T 5 http://main:8080/") == (0, "ok\n")
        assert "203.0.113.7" in run(main, "cat /etc/hosts")[1]
        endpoint = main.attrs["NetworkSettings"]["Networks"][pair.plan.network.name]
        assert endpoint["Aliases"] == ["main"]
        assert run(main, "grep CapEff /proc/self/status")[1].split()[1] == (
            "00000000a80405fb"
        )
        pair.backend.attest(pair.plan, pair.lease)
    finally:
        pair.destroy()
    assert labelled(client, "rsi-harness.run-id", run_id) == ([], [], [])


def test_real_image_volume_gets_an_implicit_labelled_volume(docker_run):
    client, run_id, stages = docker_run
    base = client.containers.create(
        BUSYBOX, ["true"], labels={"rsi-harness.run-id": run_id}, runtime="runc"
    )
    try:
        committed = client.api.commit(
            base.id,
            changes=[
                "VOLUME /var/lib/data",
                f"LABEL rsi-harness.run-id={run_id}",
                'CMD ["sleep", "600"]',
            ],
        )
    finally:
        base.remove()
    image = client.images.get(committed["Id"])
    handle = "i" + "d" * 32
    raw = single(BUSYBOX)
    raw["services"]["main"]["image"] = handle
    raw["services"]["main"]["command"] = None
    box = RealEnv(client, run_id, raw, {image.id: handle}, stages=stages)
    assert box.plan.services[0].implicit_volumes == ("/var/lib/data",)
    box.create().start()
    try:
        (implicit,) = box.lease.volumes
        assert implicit.logical.startswith("implicit:0:")
        mounts = box.container("main").attrs["Mounts"]
        assert [(mount["Name"], mount["Destination"]) for mount in mounts] == [
            (implicit.planned_name, "/var/lib/data")
        ]
        volume = client.volumes.get(implicit.planned_name).attrs
        assert volume["Labels"] == env.env_volume_labels(box.owner, box.env_id)
        assert volume["Options"] in (None, {})
        assert run(box.container("main"), "echo x > /var/lib/data/f")[0] == 0
    finally:
        box.destroy()
    assert client.volumes.list(filters={"name": implicit.planned_name}) == []


class CountingProbe(DockerDiskProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.container_polls = 0

    def container_sizes(self, run_id):
        self.container_polls += 1
        return super().container_sizes(run_id)


def test_real_disk_quota_is_detected_within_two_poll_cycles(docker_run):
    client, run_id, stages = docker_run
    raw = single(BUSYBOX, disk_mb=64, memory_mb=256)
    box = RealEnv(client, run_id, raw, {BUSYBOX: HANDLES[BUSYBOX]})
    box.create().start()
    try:
        probe = CountingProbe(client, docker_root=client.info()["DockerRootDir"])
        watchdog = DiskWatchdog(
            probe,
            run_id=run_id,
            floor_mb=1,
            hard_floor_mb=1,
            container_interval=0.5,
            volume_interval=0.5,
        )
        budgets = [EnvDiskBudget(box.env_id, box.lease.disk_mb)]
        assert watchdog.poll(budgets).over_quota == ()

        code, output = run(
            box.container("main"), "dd if=/dev/zero of=/fill bs=1048576 count=256 2>&1"
        )
        assert code == 0, output
        polls = probe.container_polls
        deadline = time.monotonic() + 10
        while True:
            verdict = watchdog.poll(budgets)
            if verdict.over_quota or time.monotonic() > deadline:
                break
            time.sleep(0.05)

        assert verdict.over_quota == (box.env_id,)
        assert probe.container_polls - polls <= 2
        assert verdict.usage[box.env_id].service_mb("main") >= 256
        box.lease = box.backend.fail(box.lease, box.commit, "disk_quota")
        assert (box.lease.state, box.lease.reason) == ("failed", "disk_quota")
        assert not box.container("main").attrs["State"]["Running"]
    finally:
        box.destroy()


def test_real_volume_writes_count_against_the_soft_limit(docker_run):
    client, run_id, stages = docker_run
    raw = single(
        BUSYBOX,
        disk_mb=64,
        memory_mb=256,
        mounts=[{"volume": "data", "target": "/data"}],
    )
    raw["volumes"] = {"data": {"seeded": False}}
    box = RealEnv(client, run_id, raw, {BUSYBOX: HANDLES[BUSYBOX]})
    box.create().start()
    try:
        watchdog = DiskWatchdog(
            DockerDiskProbe(client, docker_root=client.info()["DockerRootDir"]),
            run_id=run_id,
            floor_mb=1,
            hard_floor_mb=1,
            container_interval=0.5,
            volume_interval=0.5,
        )
        budgets = [EnvDiskBudget(box.env_id, box.lease.disk_mb)]
        assert watchdog.poll(budgets).over_quota == ()

        code, output = run(
            box.container("main"),
            "dd if=/dev/zero of=/data/fill bs=1048576 count=80 2>&1",
        )
        assert code == 0, output
        deadline = time.monotonic() + 10
        while True:
            verdict = watchdog.poll(budgets)
            if verdict.over_quota or time.monotonic() > deadline:
                break
            time.sleep(0.1)

        # 80 MiB, all of it in the env volume: over 64 MiB, under 128 MiB.
        usage = verdict.usage[box.env_id]
        assert verdict.over_quota == (box.env_id,)
        assert verdict.unmeasured == ()
        assert set(usage.volumes) == {box.lease.volumes[0].planned_name}
        assert 80 * MIB <= usage.total < 128 * MIB
    finally:
        box.destroy()


def test_real_start_failure_is_a_user_result_and_still_tears_down(docker_run):
    client, run_id, stages = docker_run
    box = RealEnv(
        client, run_id, single(BUSYBOX, "exit 3"), {BUSYBOX: HANDLES[BUSYBOX]}
    )
    box.create()
    result = box.backend.start(box.plan, box.lease, box.commit, wait_timeout_sec=20)
    box.lease = result.lease
    try:
        assert (result.state, result.reason, result.service) == (
            "failed",
            "start_failed",
            "main",
        )
        assert box.backend.status(box.lease)["main"].exit_code == 3
    finally:
        box.destroy()


def test_real_unknown_user_is_a_user_failure_with_a_fixed_detail(docker_run):
    # Docker 29 refuses this start with HTTP 500, not 4xx.
    client, run_id, stages = docker_run
    raw = single(BUSYBOX, user="nosuchuser")
    box = RealEnv(client, run_id, raw, {BUSYBOX: HANDLES[BUSYBOX]})
    box.create()
    result = box.backend.start(box.plan, box.lease, box.commit, wait_timeout_sec=20)
    box.lease = result.lease
    try:
        assert (result.state, result.reason, result.detail) == (
            "failed",
            "start_failed",
            "user not found in the image",
        )
        assert box.container("main").attrs["State"]["Status"] == "created"
    finally:
        box.destroy()
