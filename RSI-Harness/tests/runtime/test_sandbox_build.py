"""The builder backend and the build service over a fake Engine (spec 4).

The fake extends test_sandbox_env_docker's Engine (dockerd's create-time
merge) with volumes that carry driver options, attached execs over real
socket pairs in Docker's multiplexed frame format, archives and an image
store that loads a docker-archive stream. The broker, journal, job runner,
sanitizer and build context are the production ones. Real BuildKit is in
tests/integration/test_sandbox_build_docker.py.
"""

import copy
import hashlib
import io
import json
import os
import re
import struct
import sys
import tarfile
import threading
import time

import pytest
from docker.errors import NotFound

from rsi_harness.errors import InfrastructureError, RetryableSubmissionError
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_build import (
    BUILDKIT_TRACE_SOCKET,
    BUILDKITD_TOML,
    BuilderBackend,
    BuildRequest,
    build_argv,
    built_image_labels,
)
from rsi_harness.runtime.sandbox_buildfs import TmpfsStateFs
from rsi_harness.runtime.sandbox_contracts import SandboxError, SandboxOwner
from rsi_harness.runtime.sandbox_env_contracts import (
    SandboxImageLease,
    builder_container_name,
    builder_network_name,
    builder_volume_name,
    sandbox_object_labels,
)
from rsi_harness.runtime.sandbox_envs import EnvRuntime
from rsi_harness.runtime.sandbox_exec import ExecPump
from tests.runtime.test_image_archive import CONFIG, LAYER, export
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_env_docker import EngineAPI, FakeEngine
from tests.runtime.test_sandbox_envs import (
    FakeApi,
    FakeDisk,
    FakeEnvBackend,
    FakePuller,
    FakeTransfer,
    RecordingKiller,
    single,
)
from tests.runtime.test_sandbox_network import NetworkWorld
from tests.sandbox_helpers import (
    BUILDER_IMAGE_ID,
    FakeClock,
    FakeSandboxBackend,
    make_env_task,
)

MIB = 1024**2
BUILDER_ID = "b" + "1" * 32
IMAGE_ID = "sha256:" + CONFIG[0]
TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
build = true
[metadata.rsi_harness.sandbox.environments.work.limits]
max_envs_live = 2
[metadata.rsi_harness.sandbox.environments.judge]
network = ["public", "none"]
pull = true
build = true
"""


def owner(phase="work"):
    return SandboxOwner(
        run_id="run-1",
        task_id="task",
        phase=phase,
        round_id="r1" if phase == "judge" else None,
    )


# -- fake Engine ---------------------------------------------------------------


def frames(stream, data):
    return struct.pack("!BxxxI", stream, len(data)) + data


class Exec:
    def __init__(self, engine, container, argv):
        self.engine = engine
        self.container = container
        self.argv = argv
        self.running = False
        self.code = None
        self.writer = None
        self.ended = threading.Event()

    def end(self, code):
        if self.ended.is_set():
            return
        self.code, self.running = code, False
        self.ended.set()
        if self.writer is not None:
            try:
                self.writer.close()
            except OSError:
                pass


class BuildScript:
    """What the next ``buildctl build`` does."""

    def __init__(self):
        self.stdout = export()
        self.stderr = b"#1 [internal] load build definition\n#5 DONE 0.1s\n"
        self.code = 0
        # "hang": run until SIGTERM (term=True) or the builder is killed.
        self.hang = False
        self.term = True
        self.meta_digest = IMAGE_ID
        self.loaded_id = None  # the daemon reports a different ID if set
        self.oom = False
        # The daemon registers the image, then reports a load error.
        self.load_error = False
        # The loaded image's labels (from the argv's label: options) and
        # RepoTags, as the daemon reports them.
        self.labels = lambda labels: labels
        self.repo_tags = []
        # The daemon takes the whole stream but never answers ("late": it
        # registers the image only afterwards, "registered": before, "never":
        # not at all); it holds the connection until the broker drops it.
        self.unanswered = None
        # The daemon stops reading the stream after this many body bytes.
        self.stall_after = None
        # The daemon registers the image and answers this long after the
        # whole stream, whether or not the broker still listens.
        self.answer_delay = 0


class BuilderEngine(FakeEngine):
    """FakeEngine plus what a builder needs: driver options, execs,
    archives, a cgroup and an image store with ``POST /images/load``,
    served as HTTP over a socket pair (``load_connect``)."""

    def __init__(self, world):
        super().__init__(world)
        self.images[BUILDER_IMAGE_ID] = {
            "Id": BUILDER_IMAGE_ID,
            "Config": {
                "Entrypoint": ["buildkitd"],
                "Env": ["PATH=/usr/bin:/bin", "BUILDKIT_SETUP_CGROUPV2_ROOT=1"],
                "Volumes": {"/var/lib/buildkit": {}},
            },
        }
        self.execs = {}
        self.archives = []
        self.script = BuildScript()
        self.build_argv = None
        self.oom_kills = 0
        self.ready_after = 0
        self.reserve = None
        self.reserve_failures = 0
        self.trace_socket_stays = False
        self.fail_start = False
        self.load_requests = []
        # Set once a load waits for its answer, or stalls mid-stream; a held
        # load is set free (its connection was dropped) or released by a test.
        self.load_held = threading.Event()
        self.load_dropped = threading.Event()
        self.load_release = threading.Event()

    # volumes carry options
    def create_volume(self, name, driver=None, driver_opts=None, labels=None):
        self._event("volume-create", name)
        self.volumes[name] = {
            "Name": name,
            "Driver": driver,
            "Labels": dict(labels),
            "Options": dict(driver_opts or {}) or None,
            "Scope": "local",
        }
        return copy.deepcopy(self.volumes[name])

    def create_container_from_config(self, config, name=None):
        config = copy.deepcopy(config)
        for endpoint in (
            (config.get("NetworkingConfig") or {}).get("EndpointsConfig", {}).values()
        ):
            endpoint.setdefault("Aliases", [])
        result = super().create_container_from_config(config, name=name)
        attrs = self.containers[result["Id"]]
        if "apparmor=unconfined" in (config["HostConfig"].get("SecurityOpt") or []):
            attrs["AppArmorProfile"] = "unconfined"
        return result

    def start(self, container):
        if self.fail_start:
            from tests.runtime.test_sandbox_env_docker import daemon_error

            raise daemon_error("OCI runtime create failed", 500)
        super().start(container)

    def kill(self, container, signal=None):
        super().kill(container, signal=signal)
        for item in self.execs.values():
            if item.container == self._get(container)["Id"]:
                item.end(137)

    # archives
    def put_archive(self, container, path, data):
        attrs = self._get(container)
        body = data if isinstance(data, bytes) else data.read()
        self._event("put-archive", attrs["Name"][1:], path)
        self.archives.append((path, body))
        return True

    def get_archive(self, container, path):
        assert path.endswith("/meta.json")
        data = json.dumps(
            {"containerimage.config.digest": self.script.meta_digest}
        ).encode()
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            info = tarfile.TarInfo("meta.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return iter([buffer.getvalue()]), {}

    # execs
    def exec_create(self, container, cmd, **kwargs):
        attrs = self._get(container)
        if not attrs["State"]["Running"]:
            from tests.runtime.test_sandbox_env_docker import daemon_error

            raise daemon_error("container is not running", 409)
        exec_id = hashlib.sha256(f"{len(self.execs)}".encode()).hexdigest()
        self.execs[exec_id] = Exec(self, attrs["Id"], list(cmd))
        self._event("exec", " ".join(cmd)[:60])
        return {"Id": exec_id}

    def exec_start(self, exec_id, socket=False, tty=False, **kwargs):
        import socket as sockets

        item = self.execs[exec_id]
        reader, writer = sockets.socketpair()
        item.writer, item.running = writer, True
        out, err, code, hang = self._behaviour(item)

        def feed():
            try:
                if err:
                    writer.sendall(frames(2, err))
                for start in range(0, len(out), 65536):
                    writer.sendall(frames(1, out[start : start + 65536]))
            except OSError:
                pass
            if not hang:
                item.end(code)

        threading.Thread(target=feed, daemon=True).start()
        return reader

    def exec_inspect(self, exec_id):
        item = self.execs[exec_id]
        return {"Running": item.running, "ExitCode": item.code}

    def _behaviour(self, item):
        argv = item.argv
        if argv[:3] == ["buildctl", "debug", "workers"]:
            if self.ready_after:
                self.ready_after -= 1
                return b"", b"not ready\n", 1, False
            return b"ID PLATFORMS\n", b"", 0, False
        script = argv[2] if len(argv) > 2 else ""
        if "cgroup/buildkit/memory.max" in script and self.reserve_failures:
            # buildkitd has not enabled the subtree controllers yet.
            self.reserve_failures -= 1
            return b"", b"sh: can't create memory.max: Permission denied\n", 1, False
        if "cgroup/buildkit/memory.max" in script:
            memory, pids = re.findall(r"echo (\d+) >", script)
            self.reserve = (int(memory), int(pids))
            return f"{memory}\n{pids}\n".encode(), b"", 0, False
        if script.startswith("kill -TERM"):
            build = next(
                (
                    other
                    for other in self.execs.values()
                    if other.argv[0] == "/bin/sh" and other.running
                ),
                None,
            )
            if build is not None and self.script.term:
                build.end(1)
            return b"", b"", 0, False
        if script.startswith("rm -rf"):
            return b"", b"", 0, False
        if argv == ["rm", "-f", BUILDKIT_TRACE_SOCKET]:
            if self.trace_socket_stays:
                return b"", b"rm: Read-only file system\n", 1, False
            return b"", b"", 0, False
        if argv[0] == "/bin/sh" and "exec buildctl" in script:
            self.build_argv = argv
            if self.script.oom:
                self.oom_kills += 1
            return (
                self.script.stdout,
                self.script.stderr,
                self.script.code,
                self.script.hang,
            )
        raise AssertionError(argv)

    # images
    def _url(self, path):
        return path

    def load_connect(self):
        import socket as sockets

        client, server = sockets.socketpair()
        threading.Thread(target=self._serve_load, args=(server,), daemon=True).start()
        return client

    def _serve_load(self, server):
        """One ``POST /images/load``: a chunked body, then JSON lines."""
        with server, server.makefile("rb") as reader:
            request = reader.readline().decode()
            headers = {}
            while (line := reader.readline()) not in (b"\r\n", b""):
                key, _, value = line.decode().partition(":")
                headers[key.strip().lower()] = value.strip()
            self.load_requests.append((request, headers))
            body = bytearray()
            while True:
                size = reader.readline().strip()
                if not size:
                    # Cut short. Like dockerd, a whole archive (its end
                    # blocks arrived) still registers, a little later.
                    if body.endswith(b"\0" * 2 * tarfile.BLOCKSIZE):
                        time.sleep(0.2)
                        try:
                            loaded, attrs = self._load(bytes(body))
                        except (KeyError, tarfile.TarError):
                            return
                        self.images[loaded] = attrs
                        self._event("load", loaded)
                    return
                if int(size, 16) == 0:
                    reader.readline()
                    break
                body += reader.read(int(size, 16))
                reader.readline()
                if (
                    self.script.stall_after is not None
                    and len(body) >= self.script.stall_after
                ):
                    self.load_held.set()
                    self.load_release.wait(30)
                    return
            loaded, attrs = self._load(bytes(body))
            if self.script.answer_delay:
                self.load_held.set()
                time.sleep(self.script.answer_delay)
            if self.script.unanswered is not None:
                if self.script.unanswered == "registered":
                    self.images[loaded] = attrs
                elif self.script.unanswered == "late":
                    self.late = attrs
                self.load_held.set()
                # Hold the connection until the broker drops it.
                while reader.read(1):
                    pass
                self.load_dropped.set()
                return
            self.images[loaded] = attrs
            self._event("load", loaded)
            if self.script.load_error:
                items = [{"error": "write /var/lib/docker: no space left"}]
            else:
                items = [{"stream": f"Loaded image ID: {loaded}\n"}]
            data = b"".join(json.dumps(item).encode() + b"\r\n" for item in items)
            try:
                server.sendall(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                    + b"%x\r\n" % len(data)
                    + data
                    + b"\r\n0\r\n\r\n"
                )
            except OSError:
                pass  # the broker is gone; the image stays registered

    def _load(self, body):
        with tarfile.open(fileobj=io.BytesIO(body)) as tar:
            manifest = json.loads(tar.extractfile("manifest.json").read())
        image_id = "sha256:" + manifest[0]["Config"].rsplit("/", 1)[1]
        loaded = self.script.loaded_id or image_id
        labels = {
            key: value
            for option in self.build_argv
            if option.startswith("label:")
            for key, _, value in [option[len("label:") :].partition("=")]
        }
        attrs = {
            "Id": loaded,
            "RepoTags": list(self.script.repo_tags),
            "Os": "linux",
            "Architecture": "amd64",
            "Size": 12 * MIB,
            "Config": {
                "Env": ["PATH=/bin"],
                "Cmd": ["sh"],
                "Labels": self.script.labels(labels),
            },
        }
        return loaded, attrs

    def tag(self, image, repository, tag=None):
        self._event("tag", image, f"{repository}:{tag}")
        self.images[image]["RepoTags"].append(f"{repository}:{tag}")
        self.images[f"{repository}:{tag}"] = self.images[image]
        return True

    def remove_image(self, image, force=False, noprune=False):
        assert force is False
        self._event("rmi", image)
        if image not in self.images:
            raise NotFound(image)
        attrs = self.images[image]
        last = image == attrs["Id"] or attrs.get("RepoTags") == [image]
        if last and any(
            item.get("Image") == attrs["Id"] for item in self.containers.values()
        ):
            from tests.runtime.test_sandbox_env_docker import daemon_error

            raise daemon_error("conflict: image is being used by a container", 409)
        self.images.pop(image)
        if ":" in image and not image.startswith("sha256:"):
            attrs["RepoTags"].remove(image)
            if attrs["RepoTags"]:
                return
        for tag in attrs.get("RepoTags") or []:
            self.images.pop(tag, None)
        self.images.pop(attrs["Id"], None)


class BuildWorld:
    def __init__(self):
        self.events = []
        self.clock = FakeClock()
        self.network = NetworkWorld(events=self.events)
        self.engine = BuilderEngine(self)
        api = EngineAPI(self.engine)
        self.network.client.api = api
        self.backend = BuilderBackend(
            self.network.client,
            self.network.backend,
            lambda kind: TmpfsStateFs(api),
            cgroup_root=None,
            clock=time.monotonic,
            sleep=lambda seconds: None,
            kill_proof_sec=1.0,
            load_connect=self.engine.load_connect,
        )


@pytest.fixture
def world():
    return BuildWorld()


def grant(task=TASK, **build):
    task = make_env_task(task)
    import tempfile
    from pathlib import Path

    from tests.sandbox_helpers import env_policy_toml, load_policy_text

    with tempfile.TemporaryDirectory() as root:
        from rsi_harness.runtime.sandbox_policy import resolve_env_grant
        from tests.sandbox_helpers import builder_inspect

        policy = load_policy_text(
            Path(root),
            env_policy_toml(
                build={
                    "state_fs": "tmpfs",
                    "memory_mb": 2048,
                    "disk_mb": 512,
                    "cpus": 2,
                    "pids": 1024,
                    **build,
                }
            ),
        )
        return resolve_env_grant(
            task,
            policy,
            {"work": builder_inspect(), "judge": builder_inspect()},
            parent_cpus=1,
            parent_memory_mb=256,
        )


def journal_list():
    leases = []

    def commit(lease):
        leases.append(lease)
        return lease

    return leases, commit


# -- template and attestation ---------------------------------------------------


def test_the_builder_body_is_the_documented_exception_and_nothing_more(world):
    build = grant().environments.work.build
    plan = world.backend.plan(owner(), BUILDER_ID, build)
    name = builder_network_name(BUILDER_ID)
    assert plan.body() == {
        "Image": BUILDER_IMAGE_ID,
        "Entrypoint": ["buildkitd"],
        "Cmd": [
            "--config",
            "/etc/buildkit/buildkitd.toml",
            "--oci-worker-net=host",
            "--oci-max-parallelism",
            "2",
            "--oci-worker-gc",
            "--oci-worker-gc-keepstorage",
            "153,102,409",
        ],
        "Env": [
            "PATH=/usr/bin:/bin",
            "BUILDKIT_SETUP_CGROUPV2_ROOT=1",
            "NVIDIA_VISIBLE_DEVICES=void",
        ],
        "User": "0:0",
        "Labels": sandbox_object_labels(
            owner(), "sandbox-builder", {"sandbox-builder": BUILDER_ID}
        ),
        "HostConfig": {
            "Runtime": "runc",
            "Privileged": False,
            "CapAdd": ["CAP_SYS_ADMIN", "CAP_NET_ADMIN"],
            "SecurityOpt": [
                "apparmor=unconfined",
                "seccomp=unconfined",
                "writable-cgroups=true",
            ],
            "MaskedPaths": [],
            "ReadonlyPaths": [],
            "CgroupnsMode": "private",
            "IpcMode": "private",
            "NetworkMode": name,
            "Mounts": [
                {
                    "Type": "volume",
                    "Source": builder_volume_name(BUILDER_ID),
                    "Target": "/var/lib/buildkit",
                    "ReadOnly": False,
                }
            ],
            "NanoCpus": 2_000_000_000,
            "Memory": 2048 * MIB,
            "MemorySwap": 2048 * MIB,
            "PidsLimit": 1024,
            "Ulimits": [{"Name": "nofile", "Soft": 65536, "Hard": 65536}],
            "LogConfig": {"Type": "none", "Config": {}},
            "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            "PublishAllPorts": False,
            "ReadonlyRootfs": False,
        },
        "NetworkingConfig": {"EndpointsConfig": {name: {}}},
    }
    # memory - max(256 MiB, 10%) in whole MiB; pids - 64 (B4).
    assert (plan.build_memory_max, plan.build_pids_max) == (1792 * MIB, 960)


def test_create_is_rule_bridge_volume_container_config_start_ready_reserve(world):
    plan = world.backend.plan(owner(), BUILDER_ID, grant().environments.work.build)
    leases, commit = journal_list()
    lease = world.backend.create(plan, plan.lease(), commit)
    kinds = [event[0] for event in world.events if event[0] != "is_installed"]
    assert kinds[:4] == ["probe", "install", "network-create", "volume-create"]
    assert kinds[4:7] == ["container-create", "put-archive", "start"]
    assert [event[1] for event in world.events if event[0] == "exec"] == [
        "buildctl debug workers",
        f"rm -f {BUILDKIT_TRACE_SOCKET}",
        "sh -c echo $$ > /sys/fs/cgroup/init/cgroup.procs && for p in",
    ]
    assert world.engine.archives == [("/etc/buildkit", world.engine.archives[0][1])]
    with tarfile.open(fileobj=io.BytesIO(world.engine.archives[0][1])) as tar:
        assert tar.extractfile("buildkitd.toml").read() == BUILDKITD_TOML
    assert world.engine.reserve == (1792 * MIB, 960)
    assert [(item.state, item.pending_mutation) for item in leases] == [
        ("planned", True),  # network_id
        ("created", True),  # container_id
        ("running", False),
    ]
    assert (
        lease.container_id
        == hashlib.sha256(builder_container_name(BUILDER_ID).encode()).hexdigest()
    )


def test_the_reserve_is_retried_until_buildkitd_enabled_its_controllers(world):
    world.engine.reserve_failures = 2
    plan = world.backend.plan(owner(), BUILDER_ID, grant().environments.work.build)
    _, commit = journal_list()
    lease = world.backend.create(plan, plan.lease(), commit)
    assert lease.state == "running"
    assert world.engine.reserve == (1792 * MIB, 960)


def test_a_start_failure_rolls_the_whole_builder_back(world):
    from rsi_harness.errors import SetupError

    world.engine.fail_start = True
    plan = world.backend.plan(owner(), BUILDER_ID, grant().environments.work.build)
    leases, commit = journal_list()
    with pytest.raises(SetupError, match="proven absent"):
        world.backend.create(plan, plan.lease(), commit)
    assert world.engine.containers == {} and world.engine.volumes == {}
    assert world.network.store == {} and world.network.firewall.installed == {}
    assert leases[-1].state == "removed" and not leases[-1].pending_mutation


def test_a_trace_socket_that_stays_rolls_the_whole_builder_back(world):
    """S1: no BuildKit socket may reach a RUN step, and buildkitd binds its
    trace collector into every step while the socket file exists."""
    from rsi_harness.errors import SetupError

    world.engine.trace_socket_stays = True
    plan = world.backend.plan(owner(), BUILDER_ID, grant().environments.work.build)
    leases, commit = journal_list()
    with pytest.raises(SetupError, match="trace socket"):
        world.backend.create(plan, plan.lease(), commit)
    assert world.engine.containers == {} and world.engine.volumes == {}
    assert world.network.firewall.installed == {}
    assert leases[-1].state == "removed" and not leases[-1].pending_mutation


@pytest.mark.parametrize("checks", range(1, 7))
def test_a_session_end_while_the_builder_is_made_leaves_no_builder(world, checks):
    """The session's kill may find no container yet: create itself stops
    before the container, before and after its start, while it gets ready
    and after the reserve, and rolls everything back."""
    from rsi_harness.errors import SetupError

    plan = world.backend.plan(owner(), BUILDER_ID, grant().environments.work.build)
    leases, commit = journal_list()
    calls = []

    def abort():
        calls.append(len(world.engine.containers))
        return len(calls) >= checks

    with pytest.raises(SetupError, match="session ended"):
        world.backend.create(plan, plan.lease(), commit, abort=abort)
    assert len(calls) == checks
    started = any(event[0] == "start" for event in world.events)
    assert started == (checks >= 3)
    assert world.engine.containers == {} and world.engine.volumes == {}
    assert world.network.store == {} and world.network.firewall.installed == {}
    assert leases[-1].state == "removed" and not leases[-1].pending_mutation


def with_host(key, value):
    return lambda attrs: attrs["HostConfig"].__setitem__(key, value)


def with_config(key, value):
    return lambda attrs: attrs["Config"].__setitem__(key, value)


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (with_host("Privileged", True), "HostConfig.Privileged"),
        (with_host("Binds", ["/:/host"]), "HostConfig.Binds"),
        (
            with_host("Devices", [{"PathOnHost": "/dev/nvidia0"}]),
            "HostConfig.Devices",
        ),
        (
            with_host("DeviceRequests", [{"Driver": "nvidia"}]),
            "HostConfig.DeviceRequests",
        ),
        (
            with_host("CapAdd", ["CAP_SYS_ADMIN", "CAP_NET_ADMIN", "CAP_SYS_MODULE"]),
            "HostConfig.CapAdd",
        ),
        (
            with_host("SecurityOpt", ["apparmor=unconfined", "seccomp=unconfined"]),
            "HostConfig.SecurityOpt",
        ),
        (with_host("Runtime", "nvidia"), "HostConfig.Runtime"),
        (with_host("Runtime", ""), "HostConfig.Runtime"),
        (
            with_host("PortBindings", {"1234/tcp": [{"HostPort": "1234"}]}),
            "HostConfig.PortBindings",
        ),
        (with_host("PublishAllPorts", True), "HostConfig.PublishAllPorts"),
        (with_host("MaskedPaths", ["/proc/kcore"]), "HostConfig.MaskedPaths"),
        (with_host("PidMode", "host"), "HostConfig.PidMode"),
        (with_host("CgroupnsMode", "host"), "HostConfig.CgroupnsMode"),
        (with_host("NanoCpus", 0), "HostConfig.NanoCpus"),
        (with_host("MemorySwap", -1), "HostConfig.MemorySwap"),
        (with_host("LogConfig", {"Type": "journald"}), "HostConfig.LogConfig"),
        (with_host("RestartPolicy", {"Name": "always"}), "HostConfig.RestartPolicy"),
        (
            with_config("Cmd", ["--allow-insecure-entitlement", "network.host"]),
            "Config.Cmd",
        ),
        (with_config("Env", ["DOCKER_HOST=unix:///x"]), "Config.Env"),
        (
            lambda attrs: attrs["Mounts"].append(
                {"Type": "bind", "Source": "/var/run/docker.sock", "RW": True}
            ),
            "Mounts",
        ),
        (
            lambda attrs: attrs["NetworkSettings"]["Networks"].update(bridge={}),
            "Networks",
        ),
        (
            lambda attrs: attrs.__setitem__("AppArmorProfile", "docker-default"),
            "AppArmorProfile",
        ),
        (lambda attrs: attrs["State"].__setitem__("Paused", True), "is not running"),
        (lambda attrs: attrs["State"].__setitem__("Running", False), "is not running"),
    ],
)
def test_attest_mutation_matrix_every_drift_fails(world, mutate, field):
    plan = world.backend.plan(owner(), BUILDER_ID, grant().environments.work.build)
    leases, commit = journal_list()
    lease = world.backend.create(plan, plan.lease(), commit)
    world.backend.attest(plan, lease, started=True)
    mutate(world.engine.containers[lease.container_id])
    with pytest.raises(InfrastructureError, match=re.escape(field)):
        world.backend.attest(plan, lease, started=True)
    assert not world.backend.alive(plan, lease)


def request(**options):
    from rsi_harness.runtime.build_context import BuildInput

    values = dict(
        input=BuildInput(
            directory="rsi-ctx/" + "a" * 32,
            dockerfile=b"FROM x\n",
            syntax=None,
            entries=1,
            bytes=7,
            digest="d" * 64,
        ),
        input_path=None,
        target=None,
        build_args={},
        labels={},
        no_cache=False,
        network="public",
        fingerprint="f" * 64,
        lease=SandboxImageLease(owner=owner(), handle="i" + "2" * 32, kind="built"),
    )
    values.update(options)
    return BuildRequest(**values)


def test_the_build_argv_is_fixed_and_never_grants_anything():
    hostile = "--allow=network.host --secret id=x --ssh default --export-cache x"
    argv = build_argv(
        request(
            target="final",
            build_args={"A": hostile, "B": "--push"},
            labels={"org.example": "--ssh=default"},
            no_cache=True,
            network="none",
        ),
        built_image_labels(owner(), "i" + "2" * 32),
    )
    root = "/var/lib/buildkit/rsi-ctx/" + "a" * 32
    assert argv[:14] == [
        "/bin/sh",
        "-c",
        f'echo $$ > {root}/pid && exec buildctl "$@"',
        "buildctl",
        "build",
        "--progress=plain",
        "--frontend",
        "dockerfile.v0",
        "--local",
        f"context={root}/ctx",
        "--local",
        f"dockerfile={root}/df",
        "--opt",
        "filename=Dockerfile",
    ]
    assert argv[-6:] == [
        "--opt",
        "platform=linux/amd64",
        "--metadata-file",
        f"{root}/meta.json",
        "--output",
        "type=docker,dest=-",
    ]
    options = [argv[i + 1] for i, item in enumerate(argv) if item == "--opt"]
    assert "target=final" in options and "no-cache=" in options
    assert "force-network-mode=none" in options
    assert f"build-arg:A={hostile}" in options and "build-arg:B=--push" in options
    assert "label:rsi-harness.run-id=run-1" in options
    # Every caller value travels inside one --opt key=value argument.
    for flag in ("--allow", "--secret", "--ssh", "--export-cache", "--import-cache"):
        assert not any(item == flag or item.startswith(flag + "=") for item in argv)
    assert argv.count("--output") == 1 and "--push" not in argv
    assert all(
        argv[i - 1] == "--opt"
        for i, item in enumerate(argv)
        if "=" in item and item.split("=", 1)[0].split(":")[0] in ("build-arg", "label")
    )


# -- the broker ------------------------------------------------------------------


class Kit:
    def __init__(self, tmp_path, task=TASK, **build):
        from rsi_harness.runtime.sandbox import SandboxBroker
        from tests.runtime.test_sandbox_exec import FakeHost

        self.world = BuildWorld()
        self.clock = FakeClock()
        self.api, host = FakeApi(), FakeHost(tmp_path / "host")
        self.env_backend = FakeEnvBackend()
        runtime = EnvRuntime(
            backend=self.env_backend,
            images=FakePuller(),
            transfer=lambda stages: FakeTransfer(stages),
            pump=lambda on_finish: ExecPump(
                self.api,
                tmp_path / "spool",
                killer=RecordingKiller(),
                table=host.table,
                on_finish=on_finish,
                clock=self.clock,
                start_threads=False,
            ),
            spool_root=tmp_path / "spool",
            disk=FakeDisk(),
            table=host.table,
            builder=self.world.backend,
        )
        self.store = LeaseStore(tmp_path / "leases")
        self.journal = SandboxJournal(authority(self.store))
        self.broker = SandboxBroker(
            grant(task, **build),
            FakeSandboxBackend(),
            self.journal,
            self.clock,
            envs=runtime,
        )
        self.broker.envs.builds._cancel_grace = 0.5

    @property
    def envs(self):
        return self.broker.envs

    @property
    def engine(self):
        return self.world.engine

    def open(self, phase="work", round_id="r1"):
        if phase == "work":
            session = self.broker.open_session(owner(), None)
            self.broker.activate_work(10_000)
        else:
            session = self.broker.open_judge(owner("judge"), None)
            self.broker.activate_judge(5_000)
        return session

    def stage(self, session, files=None, request_id=None):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, data in (files or {"Dockerfile": b"FROM x\n"}).items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        data = buffer.getvalue()
        return self.broker.stage_put(
            session.credential,
            None,
            0,
            True,
            hashlib.sha256(data).hexdigest(),
            request_id or f"stage-{time.monotonic_ns()}",
            data,
        )["stage_id"]

    def start(self, session, request_id="build", files=None, **options):
        fields = dict(
            dockerfile=None,
            dockerfile_inline=None,
            target=None,
            build_args={},
            labels={},
            no_cache=False,
            network="public",
            timeout_sec=60,
        )
        fields.update(options)
        stage_id = self.stage(session, files)
        return self.broker.image_build(
            session.credential, stage_id=stage_id, request_id=request_id, **fields
        )["job_id"]

    def wait(self, session, job_id, timeout=10):
        job = self.envs.images.jobs[job_id]
        if job.thread is not None:
            job.thread.join(timeout)
            assert not job.thread.is_alive()
        return self.broker.job_wait(session.credential, job_id, 0, 0)

    def build(self, session, request_id="build", **options):
        view = self.wait(session, self.start(session, request_id, **options))
        return view


@pytest.fixture
def kit(tmp_path):
    kit = Kit(tmp_path)
    yield kit
    kit.envs.pump.close()


def test_a_build_binds_a_session_image_usable_by_env_create(kit):
    work = kit.open()
    capabilities = kit.broker.capabilities(work.credential)["environments"]
    assert capabilities["build"] == {
        "network": ["public", "none"],
        "max_build_sec": 3600,
        "max_context_mb": 1024,
        "max_image_mb": 8192,
        "remaining_builds": 64,
    }
    view = kit.build(work)
    assert view["state"] == "succeeded", view
    handle = view["result"]["image"]["handle"]
    assert view["result"]["image"]["image_id"] == IMAGE_ID
    [lease] = kit.journal.images()
    assert (lease.kind, lease.state, lease.image_id) == ("built", "present", IMAGE_ID)
    assert lease.tag == f"rsi-sbx-img:{lease.tag.split(':')[1]}"
    assert kit.engine.images[IMAGE_ID]["RepoTags"] == [lease.tag]
    # Forced labels reached the image through --opt label: (B8).
    assert kit.engine.images[IMAGE_ID]["Config"]["Labels"] == built_image_labels(
        owner(), handle
    )
    [builder] = kit.journal.builders()
    assert builder.state == "running" and builder.state_fs == "tmpfs"
    usage = kit.broker.capabilities(work.credential)["environments"]["usage"]
    assert usage["image_bytes"] == 12 * MIB and usage["jobs_running"] == 0
    env = kit.broker.env_create(work.credential, single(handle), "env")
    assert env["state"] == "created"
    with pytest.raises(SandboxError, match="busy"):
        kit.broker.image_release(work.credential, handle)
    kit.broker.env_destroy(work.credential, env["env_id"])
    assert kit.broker.image_release(work.credential, handle) == {"ok": True}
    assert IMAGE_ID not in kit.engine.images and kit.journal.images() == ()
    usage = kit.broker.capabilities(work.credential)["environments"]["usage"]
    assert usage["image_bytes"] == 0  # live image bytes are refunded
    assert (
        kit.broker.capabilities(work.credential)["environments"]["build"][
            "remaining_builds"
        ]
        == 63
    )  # builds are cumulative


def test_an_identical_build_of_the_session_is_deduplicated(kit):
    work = kit.open()
    first = kit.build(work, "one")
    execs = len(kit.engine.execs)
    second = kit.build(work, "two")
    assert second["state"] == "succeeded"
    assert second["result"]["image"]["handle"] == first["result"]["image"]["handle"]
    assert len(kit.engine.execs) == execs  # nothing ran
    third = kit.build(work, "three", build_args={"V": "2"})
    assert third["result"]["image"]["handle"] != first["result"]["image"]["handle"]


def test_build_handles_resolve_only_in_their_session(kit):
    work = kit.open()
    view = kit.build(work)
    handle, job_id = view["result"]["image"]["handle"], view["job_id"]
    kit.broker.freeze_work()
    judge = kit.open("judge")
    with pytest.raises(SandboxError, match="permission"):
        kit.broker.job_wait(judge.credential, job_id, 0, 0)
    with pytest.raises(SandboxError, match="permission"):
        kit.broker.env_create(judge.credential, single(handle), "env")
    with pytest.raises(SandboxError, match="permission"):
        kit.broker.image_release(judge.credential, handle)
    assert kit.broker.image_list(judge.credential) == {"images": []}


def test_requests_replay_and_conflict(kit):
    work = kit.open()
    stage_id = kit.stage(work)
    fields = dict(
        dockerfile=None,
        dockerfile_inline=None,
        target=None,
        build_args={},
        labels={},
        no_cache=False,
        network="public",
        timeout_sec=60,
    )
    first = kit.broker.image_build(
        work.credential, stage_id=stage_id, request_id="r", **fields
    )
    assert (
        kit.broker.image_build(
            work.credential, stage_id=stage_id, request_id="r", **fields
        )
        == first
    )
    with pytest.raises(SandboxError, match="conflicting"):
        kit.broker.image_build(
            work.credential,
            stage_id=stage_id,
            request_id="r",
            **{**fields, "no_cache": True},
        )
    # The stage was consumed once: a new request cannot reuse it.
    with pytest.raises(SandboxError, match="unknown or expired stage"):
        kit.broker.image_build(
            work.credential, stage_id=stage_id, request_id="other", **fields
        )
    kit.wait(work, first["job_id"])


@pytest.mark.parametrize(
    ("options", "code", "field"),
    [
        ({"network": "host"}, "invalid", "network"),
        ({"build_args": {"BUILDKIT_SYNTAX": "x"}}, "permission", "build_args"),
        ({"build_args": {"buildx_x": "x"}}, "permission", "build_args"),
        ({"build_args": {"1bad": "x"}}, "invalid", "build_args"),
        ({"build_args": {"A": "x" * 5000}}, "invalid", "build_args.A"),
        ({"labels": {"rsi-harness.run-id": "forged"}}, "permission", "labels"),
        ({"labels": {"a=b": "x"}}, "invalid", "labels"),
        ({"target": "../x"}, "invalid", "target"),
        ({"timeout_sec": 0}, "invalid", "timeout_sec"),
        ({"no_cache": "yes"}, "invalid", "no_cache"),
    ],
)
def test_caller_options_follow_the_allowlist(kit, options, code, field):
    work = kit.open()
    with pytest.raises(SandboxError) as caught:
        kit.start(work, **options)
    assert (caught.value.code, caught.value.field) == (code, field)
    assert kit.engine.execs == {}


def test_a_build_network_not_granted_is_refused(tmp_path):
    # Envs may use "none"; builds were approved (and requested) public only.
    task = TASK.replace("build = true", 'build = true\nbuild_network = ["public"]')
    kit = Kit(tmp_path, task, network=["public"])
    try:
        work = kit.open()
        with pytest.raises(SandboxError) as caught:
            kit.start(work, network="none")
        assert (caught.value.code, caught.value.field) == ("permission", "network")
    finally:
        kit.envs.pump.close()


def test_max_builds_is_cumulative_and_a_failed_build_counts(tmp_path):
    kit = Kit(tmp_path, max_builds=2)
    try:
        work = kit.open()
        kit.engine.script.code = 1
        kit.engine.script.stdout = b""
        assert kit.build(work, "one")["state"] == "failed"
        kit.engine.script.code = 0
        kit.engine.script.stdout = export()
        assert kit.build(work, "two", build_args={"X": "1"})["state"] == "succeeded"
        with pytest.raises(SandboxError) as caught:
            kit.start(work, "three", build_args={"X": "2"})
        assert (caught.value.code, caught.value.field) == ("quota", "max_builds")
    finally:
        kit.envs.pump.close()


def test_a_dockerfile_failure_names_its_error_and_prunes(kit):
    work = kit.open()
    kit.engine.script.stdout = b""
    kit.engine.script.code = 1
    kit.engine.script.stderr = (
        b"#5 ERROR: process did not complete successfully\n"
        b'error: failed to solve: process "/bin/sh -c false" exit code: 1\n'
    )
    view = kit.build(work)
    assert view["state"] == "failed"
    assert view["error"]["kind"] == "dockerfile"
    assert "failed to solve" in view["error"]["message"]
    assert "exit code: 1" in view["log"]
    assert kit.journal.images() == ()
    kit.envs.images.jobs[view["job_id"]].thread.join(5)
    assert any(
        "buildctl prune" in item.argv[2]
        for item in kit.engine.execs.values()
        if len(item.argv) > 2
    )


def test_a_full_state_fs_is_a_disk_failure(kit):
    work = kit.open()
    kit.engine.script.stdout = b""
    kit.engine.script.code = 1
    kit.engine.script.stderr = (
        b"#6 0.3 dd: error writing '/big': No space left on device\n"
    )
    assert kit.build(work)["error"]["kind"] == "disk"


def test_a_timeout_sends_sigterm_first_and_needs_no_kill(kit):
    work = kit.open()
    kit.engine.script.stdout = b""
    kit.engine.script.hang = True
    job_id = kit.start(work, timeout_sec=5)
    time.sleep(0.3)
    kit.clock.now += 6
    kit.broker.sweep_expired()
    view = kit.wait(work, job_id)
    assert (view["state"], view["error"]["kind"]) == ("timed_out", "timeout")
    commands = [" ".join(item.argv) for item in kit.engine.execs.values()]
    assert any(command.startswith("sh -c kill -TERM $(cat ") for command in commands)
    [builder] = kit.journal.builders()
    assert kit.engine.containers[builder.container_id]["State"]["Running"]
    assert not kit.envs.builds._builders[work].broken


def test_a_build_ignoring_sigterm_kills_the_builder_which_is_replaced(kit):
    work = kit.open()
    kit.engine.script.stdout = b""
    kit.engine.script.hang = True
    kit.engine.script.term = False
    job_id = kit.start(work, timeout_sec=5)
    time.sleep(0.3)
    kit.clock.now += 6
    kit.broker.sweep_expired()
    view = kit.wait(work, job_id)
    assert view["state"] == "timed_out"
    [old] = kit.journal.builders()
    assert not kit.engine.containers[old.container_id]["State"]["Running"]
    # The next build removes the killed builder and makes a new one.
    kit.engine.script = type(kit.engine.script)()
    assert kit.build(work, "next")["state"] == "succeeded"
    [new] = kit.journal.builders()
    assert new.builder_id != old.builder_id
    assert old.container_id not in kit.engine.containers


def test_a_load_id_mismatch_removes_the_image_and_is_infrastructure(kit):
    work = kit.open()
    other = "sha256:" + "9" * 64
    kit.engine.script.loaded_id = other
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    assert other not in kit.engine.images and IMAGE_ID not in kit.engine.images
    assert ("rmi", other) in kit.world.events
    assert kit.journal.images() == ()


def forged(labels):
    return {**labels, "rsi-harness.sandbox-image": "i" + "9" * 32}


def extra(labels):
    return {**labels, "rsi-harness.extra": "x"}


def missing(labels):
    return {key: value for key, value in labels.items() if "run-id" not in key}


@pytest.mark.parametrize(
    "tamper",
    [
        {"labels": forged},
        {"labels": extra},
        {"labels": missing},
        {"repo_tags": ["someone/else:latest"]},
    ],
)
def test_a_loaded_image_without_exactly_the_forced_identity_is_refused(kit, tamper):
    """The daemon's view of the loaded image must carry exactly the forced
    ``rsi-harness.*`` labels and no tag before the broker tags it."""
    work = kit.open()
    for key, value in tamper.items():
        setattr(kit.engine.script, key, value)
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    assert "is not the build's" in view["error"]["message"]
    assert ("rmi", IMAGE_ID) in kit.world.events
    assert IMAGE_ID not in kit.engine.images
    assert not any(event[0] == "tag" for event in kit.world.events)
    assert kit.journal.images() == ()
    assert (
        kit.broker.capabilities(work.credential)["environments"]["usage"]["image_bytes"]
        == 0
    )


def test_an_unanswered_load_that_registered_is_removed_at_once(kit):
    """No answer, but the image is there: removing it settles the load."""
    work = kit.open()
    kit.engine.script.unanswered = "registered"
    kit.envs.builds._load_timeout = 0.5
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    assert "did not answer" in view["error"]["message"]
    assert ("rmi", IMAGE_ID) in kit.world.events
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_an_unanswered_load_stays_journaled_until_session_end(kit):
    """The daemon took the whole stream but never answered and nothing is
    there yet: absence proves nothing, so the digest stays journaled
    (leaked) and is removed at session end once the image registered."""
    work = kit.open()
    kit.engine.script.unanswered = "late"
    kit.envs.builds._load_timeout = 0.5
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    assert "did not answer" in view["error"]["message"]
    [lease] = kit.journal.images()
    assert (lease.state, lease.image_id) == ("leaked", IMAGE_ID)
    assert not kit.broker.recovery_required
    kit.engine.images[IMAGE_ID] = kit.engine.late
    kit.broker.close()
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_an_unanswered_load_still_absent_at_session_end_fails_closed(kit):
    """Nothing proves the daemon will not register it later: the record
    stays leaked for recovery's label sweep and the run fails closed."""
    work = kit.open()
    kit.engine.script.unanswered = "never"
    kit.envs.builds._load_timeout = 0.5
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    with pytest.raises(InfrastructureError):
        kit.broker.close()
    assert kit.broker.recovery_required
    [lease] = kit.journal.images()
    assert (lease.state, lease.image_id) == ("leaked", IMAGE_ID)


def test_the_load_is_a_chunked_post_on_its_own_connection(kit):
    work = kit.open()
    assert kit.build(work)["state"] == "succeeded"
    [(request, headers)] = kit.engine.load_requests
    assert request == "POST /images/load?quiet=1 HTTP/1.1\r\n"
    assert headers["transfer-encoding"] == "chunked"
    assert headers["connection"] == "close"


def test_image_loads_need_the_daemons_unix_socket(kit):
    """A Docker host that is no Unix socket is refused when a build is
    granted (production's sandbox_builder_images), before any build runs;
    a load reaching it anyway fails as infrastructure and sends nothing."""
    from rsi_harness.errors import SetupError
    from rsi_harness.runtime.sandbox_build import engine_socket

    with pytest.raises(SetupError, match="unix:// Docker host"):
        engine_socket(kit.engine)
    work = kit.open()
    kit.world.backend._load_connect = None
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    assert kit.engine.load_requests == []
    assert kit.journal.images() == ()


def judge_load(kit, *, stall_after=None, unanswered=None, answer_delay=0):
    """A Judge build whose load the daemon holds; returns (judge, job_id)."""
    kit.open()
    kit.broker.freeze_work()
    judge = kit.open("judge")
    kit.engine.script.stall_after = stall_after
    kit.engine.script.unanswered = unanswered
    kit.engine.script.answer_delay = answer_delay
    job_id = kit.start(judge, "judge")
    assert kit.engine.load_held.wait(10)
    return judge, job_id


def test_close_judge_waits_for_the_answer_of_a_whole_load(kit):
    """The daemon has the whole stream and registers the image after a
    while (dockerd does so even when the client is gone): close_judge waits
    for its answer instead of cutting the connection, then removes the
    image, so nothing registers after Work resumes."""
    from rsi_harness.runtime.sandbox_build import LOAD_ANSWER_GRACE_SEC
    from rsi_harness.runtime.sandbox_envs import DELETE_SEC

    assert LOAD_ANSWER_GRACE_SEC <= DELETE_SEC / 2
    _, job_id = judge_load(kit, answer_delay=1.0)
    [lease] = [item for item in kit.journal.images() if item.owner.phase == "judge"]
    assert (lease.state, lease.image_id) == ("loading", IMAGE_ID)
    began = time.monotonic()
    kit.broker.close_judge()
    assert 0.5 < time.monotonic() - began < DELETE_SEC
    job = kit.envs.images.jobs[job_id]
    assert (job.state, job.error["kind"]) == ("canceled", "canceled")
    assert ("rmi", IMAGE_ID) in kit.world.events
    time.sleep(0.5)
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_close_judge_bounds_the_wait_for_an_answer_that_never_comes(kit):
    """The daemon took the whole stream, registered the image and holds its
    answer longer than any socket timeout: close_judge waits only the
    answer grace, and the journaled digest is retried at session end."""
    from rsi_harness.runtime.sandbox_envs import KILL_SEC

    kit.envs.builds._load_grace = 0.5
    _, job_id = judge_load(kit, unanswered="registered")
    assert kit.envs.builds._load_timeout == 600
    began = time.monotonic()
    kit.broker.close_judge()
    assert time.monotonic() - began < KILL_SEC
    assert kit.engine.load_dropped.wait(5)
    job = kit.envs.images.jobs[job_id]
    assert (job.state, job.error["kind"]) == ("canceled", "canceled")
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_close_judge_fails_closed_when_an_unanswered_load_left_nothing(kit):
    """No answer and no image: the daemon may still register it after Work
    resumes, so the round's close fails the run closed (S6, S8) and the
    digest stays leaked for recovery's label sweep."""
    from rsi_harness.runtime.sandbox_envs import KILL_SEC

    kit.envs.builds._load_grace = 0.5
    _, job_id = judge_load(kit, unanswered="never")
    began = time.monotonic()
    with pytest.raises(InfrastructureError):
        kit.broker.close_judge()
    assert time.monotonic() - began < KILL_SEC
    assert kit.broker.recovery_required
    [lease] = [item for item in kit.journal.images() if item.owner.phase == "judge"]
    assert (lease.state, lease.image_id) == ("leaked", IMAGE_ID)


def test_close_judge_interrupts_a_load_the_daemon_stops_reading(kit):
    """A daemon that stops reading blocks the stream's send: close_judge
    cuts it within its bounds and nothing of the load is left."""
    from rsi_harness.runtime.sandbox_envs import KILL_SEC

    big = os.urandom(8 * MIB)
    layer = hashlib.sha256(big).hexdigest(), big
    manifest = [
        {
            "Config": f"blobs/sha256/{CONFIG[0]}",
            "RepoTags": None,
            "Layers": [f"blobs/sha256/{layer[0]}"],
        }
    ]
    kit.engine.script.stdout = export(blobs=(CONFIG, layer), manifest=manifest)
    try:
        _, job_id = judge_load(kit, stall_after=MIB)
        # The stream's send blocks (ImageLoad._send), so only the load's own
        # close can end it, not the builder's kill.
        thread = kit.envs.images.jobs[job_id].thread
        settled, deadline = None, time.monotonic() + 10
        while time.monotonic() < deadline:
            frame = sys._current_frames().get(thread.ident)
            if frame is None or frame.f_code.co_name not in ("_send", "_check"):
                settled = None
            elif settled is None:
                settled = time.monotonic()
            elif time.monotonic() - settled > 0.3:
                break
            time.sleep(0.01)
        assert settled is not None and time.monotonic() - settled > 0.3
        began = time.monotonic()
        kit.broker.close_judge()
        assert time.monotonic() - began < KILL_SEC
    finally:
        kit.engine.load_release.set()
    job = kit.envs.images.jobs[job_id]
    assert (job.state, job.error["kind"]) == ("canceled", "canceled")
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_a_load_past_its_deadline_is_closed_and_retried_at_session_end(kit):
    """The job's deadline closes an unanswered load after the answer grace
    (no socket timeout); the digest stays leaked until the session ends."""
    work = kit.open()
    kit.envs.builds._load_grace = 0.5
    kit.engine.script.unanswered = "late"
    job_id = kit.start(work, timeout_sec=5)
    assert kit.engine.load_held.wait(10)
    kit.clock.now += 6
    kit.broker.sweep_expired()
    view = kit.wait(work, job_id, timeout=5)
    assert (view["state"], view["error"]["kind"]) == ("timed_out", "timeout")
    [lease] = kit.journal.images()
    assert (lease.state, lease.image_id) == ("leaked", IMAGE_ID)
    assert not kit.broker.recovery_required
    kit.engine.images[IMAGE_ID] = kit.engine.late
    kit.broker.close()
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_a_build_canceled_as_its_load_begins_sends_nothing(kit, monkeypatch):
    """The cancel lands before the load connects (hold runs the close at
    once): no load request reaches the daemon and the job is canceled."""
    work = kit.open()
    runner = kit.envs.images
    hold = runner.hold

    def canceled(job, closer):
        job.cancel.set()
        hold(job, closer)

    monkeypatch.setattr(runner, "hold", canceled)
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("canceled", "canceled")
    assert kit.engine.load_requests == []
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_a_close_racing_an_arrived_answer_cancels_and_removes_the_image(
    kit, monkeypatch
):
    """The answer arrived, then the close: the outcome is known, the job is
    canceled all the same and the registered image removed with proof."""
    from rsi_harness.runtime import sandbox_build

    work = kit.open()
    receive = sandbox_build.ImageLoad._receive

    def racing(self, sock):
        data = receive(self, sock)
        self.close()
        return data

    monkeypatch.setattr(sandbox_build.ImageLoad, "_receive", racing)
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("canceled", "canceled")
    assert ("rmi", IMAGE_ID) in kit.world.events
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_a_close_once_the_archive_end_went_out_waits_for_the_answer(kit, monkeypatch):
    """The daemon has the archive's end blocks (the last chunk) but not the
    chunked terminator: cutting now would still register the image, so
    the close lets the stream end and waits for the answer instead."""
    from rsi_harness.runtime import sandbox_build

    work = kit.open()
    send = sandbox_build.ImageLoad._send

    def closing(self, sock, data):
        send(self, sock, data)
        if self.final and data.endswith(b"\0" * 1024 + b"\r\n"):
            self.close()

    monkeypatch.setattr(sandbox_build.ImageLoad, "_send", closing)
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("canceled", "canceled")
    time.sleep(0.5)
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required


def test_a_metadata_digest_mismatch_is_infrastructure(kit):
    work = kit.open()
    kit.engine.script.meta_digest = "sha256:" + "8" * 64
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    assert IMAGE_ID not in kit.engine.images


def test_an_export_over_max_image_mb_is_quota_and_never_loaded(tmp_path):
    kit = Kit(tmp_path, max_image_mb=1)
    try:
        work = kit.open()
        big = hashlib.sha256(b"x" * (2 * MIB)).hexdigest(), b"x" * (2 * MIB)
        manifest = [
            {
                "Config": f"blobs/sha256/{CONFIG[0]}",
                "RepoTags": None,
                "Layers": [f"blobs/sha256/{big[0]}"],
            }
        ]
        kit.engine.script.stdout = export(blobs=(CONFIG, big), manifest=manifest)
        view = kit.build(work)
        assert (view["state"], view["error"]["kind"]) == ("failed", "quota")
        assert not any(event[0] == "load" for event in kit.world.events)
        assert IMAGE_ID not in kit.engine.images
        assert kit.journal.images() == ()
    finally:
        kit.envs.pump.close()


def test_an_oom_in_a_run_step_is_reported_as_oom(kit):
    work = kit.open()
    kit.world.backend.oom_kills = lambda lease: kit.engine.oom_kills
    kit.engine.script.stdout = b""
    kit.engine.script.code = 1
    kit.engine.script.oom = True
    assert kit.build(work)["error"]["kind"] == "oom"


def build_execs(kit):
    return [
        item
        for item in kit.engine.execs.values()
        if item.argv[0] == "/bin/sh" and "exec buildctl" in item.argv[2]
    ]


def test_max_concurrent_builds_queues_the_next_build(tmp_path):
    kit = Kit(tmp_path, max_concurrent_builds=1)
    try:
        work = kit.open()
        kit.engine.script.stdout = b""
        kit.engine.script.hang = True
        first = kit.start(work, "one")
        deadline = time.monotonic() + 5
        while not build_execs(kit) and time.monotonic() < deadline:
            time.sleep(0.01)
        second = kit.start(work, "two", build_args={"X": "1"})
        time.sleep(0.3)
        # The second build waits for the session's only slot.
        assert len(build_execs(kit)) == 1
        assert kit.broker.job_wait(work.credential, second, 0, 0)["state"] == "queued"
        kit.engine.script = type(kit.engine.script)()
        kit.broker.job_cancel(work.credential, first)
        assert kit.wait(work, first)["state"] == "canceled"
        assert kit.wait(work, second)["state"] == "succeeded"
        assert len(build_execs(kit)) == 2
    finally:
        kit.envs.pump.close()


def test_a_build_deadline_is_the_least_of_timeout_grant_and_session():
    from types import SimpleNamespace

    from rsi_harness.runtime.sandbox_build import build_deadline

    build = grant(max_build_sec=100).environments.work.build
    unbounded, ending = SimpleNamespace(deadline=None), SimpleNamespace(deadline=40.0)
    assert build_deadline(unbounded, 50, build, 10.0) == 60.0
    assert build_deadline(unbounded, 500, build, 10.0) == 110.0
    assert build_deadline(ending, 50, build, 10.0) == 40.0


def test_a_timeout_above_max_build_sec_times_out_at_max_build_sec(tmp_path):
    kit = Kit(tmp_path, max_build_sec=10)
    try:
        work = kit.open()
        kit.engine.script.stdout = b""
        kit.engine.script.hang = True
        job_id = kit.start(work, timeout_sec=60)
        time.sleep(0.3)
        kit.clock.now += 11
        kit.broker.sweep_expired()
        view = kit.wait(work, job_id)
        assert (view["state"], view["error"]["kind"]) == ("timed_out", "timeout")
    finally:
        kit.envs.pump.close()


def test_a_build_spool_failure_completes_its_request(kit, monkeypatch):
    """The request is answered (and replayed) with its error, not left busy."""
    work = kit.open()
    stage_id = kit.stage(work)

    def broken(job_id):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(kit.envs.builds, "input_path", broken)
    fields = dict(
        dockerfile=None,
        dockerfile_inline=None,
        target=None,
        build_args={},
        labels={},
        no_cache=False,
        network="public",
        timeout_sec=60,
    )
    with pytest.raises(OSError):
        kit.broker.image_build(
            work.credential, stage_id=stage_id, request_id="r", **fields
        )
    with pytest.raises(SandboxError) as replayed:
        kit.broker.image_build(
            work.credential, stage_id=stage_id, request_id="r", **fields
        )
    assert replayed.value.code != "busy"
    assert all(stage_id not in spool.prepaid for spool in kit.envs._spools.values())
    monkeypatch.undo()
    assert kit.build(work, "other")["state"] == "succeeded"


def test_submit_waits_for_a_work_build_in_flight(kit):
    work = kit.open()
    kit.engine.script.hang = True
    kit.engine.script.stdout = b""
    job_id = kit.start(work)
    time.sleep(0.2)
    with pytest.raises(RetryableSubmissionError):
        kit.broker.freeze_work()
    kit.broker.job_cancel(work.credential, job_id)
    assert kit.wait(work, job_id)["state"] == "canceled"
    kit.broker.freeze_work()


def test_close_judge_kills_then_removes_the_round_builder_and_images(kit):
    work = kit.open()
    assert kit.build(work, "work")["state"] == "succeeded"
    kit.broker.freeze_work()
    judge = kit.open("judge")
    kit.engine.script = type(kit.engine.script)()
    kit.engine.script.stdout = export(blobs=(CONFIG, LAYER))
    view = kit.build(judge, "judge", build_args={"J": "1"})
    assert view["state"] == "succeeded"
    builders = {builder.owner.phase: builder for builder in kit.journal.builders()}
    assert set(builders) == {"work", "judge"}
    kit.broker.close_judge()
    [left] = kit.journal.builders()
    assert left.owner.phase == "work"
    assert builders["judge"].container_id not in kit.engine.containers
    assert [image.owner.phase for image in kit.journal.images()] == ["work"]
    assert not kit.broker.recovery_required
    kit.broker.resume_work()
    # The Work builder, idle through the round, keeps its cache and runs.
    assert kit.engine.containers[left.container_id]["State"]["Running"]


def test_a_vanished_builder_fails_nothing_but_is_replaced(kit):
    work = kit.open()
    assert kit.build(work, "one")["state"] == "succeeded"
    [old] = kit.journal.builders()
    # Another host user removed the builder container meanwhile.
    del kit.engine.containers[old.container_id]
    view = kit.build(work, "two", build_args={"X": "1"})
    assert view["state"] == "succeeded", view
    [new] = kit.journal.builders()
    assert new.builder_id != old.builder_id
    assert not kit.broker.recovery_required


def test_a_builder_journal_failure_fails_the_run_closed(kit, monkeypatch):
    work = kit.open()

    def broken(lease):
        raise OSError("journal disk full")

    monkeypatch.setattr(kit.journal, "plan_builder", broken)
    view = kit.build(work)
    assert view["state"] == "failed"
    assert kit.broker.recovery_required
    assert kit.engine.containers == {}


def test_broker_close_removes_the_builder_and_every_built_image(kit):
    work = kit.open()
    assert kit.build(work)["state"] == "succeeded"
    kit.broker.close()
    assert kit.journal.builders() == () and kit.journal.images() == ()
    assert kit.engine.containers == {} and kit.engine.volumes == {}
    assert IMAGE_ID not in kit.engine.images
    assert kit.world.network.store == {}
    assert kit.world.network.firewall.installed == {}
    assert not kit.broker.recovery_required


def test_a_load_failing_after_the_journaled_digest_is_cleaned_up(kit):
    work = kit.open()
    kit.engine.script.load_error = True
    view = kit.build(work)
    assert (view["state"], view["error"]["kind"]) == ("failed", "infrastructure")
    # The digest was journaled (loading) before the stream ended; the image
    # it names is removed and the record leaves the journal.
    assert ("rmi", IMAGE_ID) in kit.world.events
    assert IMAGE_ID not in kit.engine.images
    assert kit.journal.images() == ()


def test_an_rmi_conflict_leaves_a_leaked_record_retried_at_session_end(kit):
    work = kit.open()
    handle = kit.build(work)["result"]["image"]["handle"]
    # A foreign container (not of this session) uses the image.
    kit.engine.containers["c" * 64] = {
        "Id": "c" * 64,
        "Name": "/foreign",
        "Image": IMAGE_ID,
        "Config": {"Labels": {}},
        "State": {"Running": False, "Paused": False},
        "Mounts": [],
    }
    assert kit.broker.image_release(work.credential, handle) == {"ok": True}
    [lease] = kit.journal.images()
    assert lease.state == "leaked" and IMAGE_ID in kit.engine.images
    # It is not fail-closed: nothing of it runs.
    assert not kit.broker.recovery_required
    del kit.engine.containers["c" * 64]
    kit.broker.close()
    assert kit.journal.images() == () and IMAGE_ID not in kit.engine.images
    assert not kit.broker.recovery_required
