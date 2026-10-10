"""Env services: fixed template, exact attest, ordered start, proven teardown."""

import copy
import hashlib
import logging
import posixpath

import pytest
import requests
from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime import sandbox_env_contracts as env
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_docker import sandbox_labels
from rsi_harness.runtime.sandbox_env_docker import (
    CgroupPausedKiller,
    DockerPausedKiller,
    SandboxEnvDockerBackend,
    default_paused_killer,
    plan_env,
    terminate_container,
)
from tests.runtime.test_sandbox_budget import make_child
from tests.runtime.test_sandbox_network import NetworkWorld
from tests.runtime.test_sandbox_policy_v2 import owner as make_owner
from tests.sandbox_helpers import FakeClock, make_env_spec

ENV_ID = "e" + "1" * 32
MAIN_HANDLE = "i" + "a" * 32
KV_HANDLE = "i" + "b" * 32
MAIN_IMAGE = "sha256:" + "a" * 64
KV_IMAGE = "sha256:" + "b" * 64
MIB = 1024**2


def main_image(**config):
    values = {
        "Env": ["PATH=/usr/bin:/bin"],
        "Cmd": ["/bin/bash"],
        "Labels": {"org.opencontainers.image.version": "24.04"},
    }
    values.update(config)
    return {"Id": MAIN_IMAGE, "Os": "linux", "Config": values}


def kv_image(**config):
    values = {
        "Env": [
            "PATH=/usr/bin:/bin",
            "REDIS_VERSION=7",
            "NVIDIA_DRIVER_CAPABILITIES=all",
        ],
        "Entrypoint": ["docker-entrypoint.sh"],
        "Cmd": ["redis-server"],
        "Volumes": {"/data/": {}},
        "Healthcheck": {"Test": ["CMD", "true"], "Interval": 5 * 10**9},
        "StopSignal": "SIGINT",
    }
    values.update(config)
    return {"Id": KV_IMAGE, "Os": "linux", "Config": values}


def spec(raw=None, **service_updates):
    return env.parse_env_spec(raw or make_env_spec(**service_updates))


def daemon_error(message, status=400):
    """An APIError carrying the Engine's HTTP answer and explanation."""
    response = requests.Response()
    response.status_code = status
    return APIError(message, response=response, explanation=message)


def _labels_match(labels, filters):
    wanted = filters.get("label", ())
    if isinstance(wanted, str):
        wanted = (wanted,)
    return all(
        (labels or {}).get(key) == value
        for key, value in (item.split("=", 1) for item in wanted)
    )


class FakeEngine:
    """The low-level API surface, merging create bodies like dockerd does."""

    def __init__(self, world):
        self.world = world
        self.images = {MAIN_IMAGE: main_image(), KV_IMAGE: kv_image()}
        self.containers = {}
        self.volumes = {}
        self.create_errors = {}
        self.start_errors = {}
        self.on_create = lambda attrs: None
        self.on_start = {}
        self.fail_pause_proof = False
        self.timeout = 5

    def _event(self, *event):
        self.world.events.append(event)

    def inspect_image(self, image_id):
        if image_id not in self.images:
            raise NotFound(image_id)
        return copy.deepcopy(self.images[image_id])

    # volumes
    def create_volume(self, name, driver=None, driver_opts=None, labels=None):
        self._event("volume-create", name)
        assert driver == "local" and driver_opts is None
        error = self.create_errors.get(name)
        if error is not None:
            raise error
        self.volumes[name] = {
            "Name": name,
            "Driver": driver,
            "Labels": dict(labels),
            "Options": None,
            "Scope": "local",
        }
        return copy.deepcopy(self.volumes[name])

    def inspect_volume(self, name):
        if name not in self.volumes:
            raise NotFound(name)
        return copy.deepcopy(self.volumes[name])

    def remove_volume(self, name, force=False):
        assert force is False
        self._event("volume-remove", name)
        if any(
            mount.get("Name") == name
            for attrs in self.containers.values()
            for mount in attrs["Mounts"]
        ):
            raise daemon_error("volume is in use", 409)
        self.volumes.pop(name, None)

    def volumes_list(self, filters):
        return [
            copy.deepcopy(attrs)
            for attrs in self.volumes.values()
            if _labels_match(attrs["Labels"], filters)
        ]

    # networks (the bridge itself lives in the M2 network world)
    def networks(self, filters=None):
        return [
            {
                "Id": network.id,
                "Name": network.attrs["Name"],
                "Labels": network.attrs["Labels"],
            }
            for network in self.world.network.store.values()
            if _labels_match(network.attrs["Labels"], filters or {})
        ]

    def remove_network(self, network_id):
        self._event("network-remove", network_id)
        self.world.network.store.pop(network_id, None)

    def inspect_network(self, network_id):
        if network_id not in self.world.network.store:
            raise NotFound(network_id)
        return self.world.network.store[network_id].attrs

    # containers
    def create_container_from_config(self, config, name=None):
        self._event("container-create", name)
        error = self.create_errors.get(name)
        if error is not None:
            raise error
        if any(attrs["Name"] == "/" + name for attrs in self.containers.values()):
            raise daemon_error("name is already in use", 409)
        identity = hashlib.sha256(name.encode()).hexdigest()
        image = self.images[config["Image"]]["Config"]
        body = copy.deepcopy(config)
        host = body.pop("HostConfig")
        networking = body.pop("NetworkingConfig", None)
        # dockerd's create-time merge with the image config.
        environment = list(body.get("Env") or [])
        keys = {item.split("=", 1)[0] for item in environment}
        environment += [
            item for item in image.get("Env", []) if item.split("=", 1)[0] not in keys
        ]
        body["Env"] = environment
        body["Labels"] = {**(image.get("Labels") or {}), **(body.get("Labels") or {})}
        if not body.get("Entrypoint"):
            if not body.get("Cmd"):
                body["Cmd"] = image.get("Cmd")
            if body.get("Entrypoint") is None:
                body["Entrypoint"] = image.get("Entrypoint")
        for key in ("User", "WorkingDir", "StopSignal"):
            body[key] = (
                body.get(key) or image.get(key) or ("" if key != "StopSignal" else None)
            )
        if body["WorkingDir"]:
            # dockerd keeps the merged WorkingDir as Go's filepath.Clean of it.
            cleaned = posixpath.normpath(body["WorkingDir"])
            body["WorkingDir"] = "/" + cleaned.lstrip("/")
        check = body.get("Healthcheck")
        if check is None:
            check = image.get("Healthcheck")
        elif image.get("Healthcheck"):
            for key, value in image["Healthcheck"].items():
                check.setdefault(key, value)
                if not check[key]:
                    check[key] = value
        body["Healthcheck"] = check
        body["Hostname"] = body.get("Hostname") or identity[:12]
        body["Volumes"] = image.get("Volumes")
        mounts = [
            {
                "Type": "volume",
                "Name": mount["Source"],
                "Source": f"/var/lib/docker/volumes/{mount['Source']}/_data",
                "Destination": mount["Target"],
                "Driver": "local",
                "Mode": "z",
                "RW": not mount.get("ReadOnly", False),
                "Propagation": "",
            }
            for mount in host.get("Mounts") or []
        ]
        covered = {mount["Destination"] for mount in mounts} | set(
            host.get("Tmpfs") or {}
        )
        for path in image.get("Volumes") or {}:
            if path.rstrip("/") not in covered:
                anonymous = hashlib.sha256((identity + path).encode()).hexdigest()
                mounts.append(
                    {
                        "Type": "volume",
                        "Name": anonymous,
                        "Destination": path.rstrip("/"),
                        "Driver": "local",
                        "RW": True,
                    }
                )
        full_host = {
            "Binds": None,
            "CapAdd": None,
            "Devices": None,
            "DeviceRequests": None,
            "DeviceCgroupRules": None,
            "Dns": None,
            "DnsOptions": None,
            "DnsSearch": None,
            "Links": None,
            "VolumesFrom": None,
            "PortBindings": {},
            "PidMode": "",
            "UTSMode": "",
            "UsernsMode": "",
            "Cgroup": "",
            "CgroupParent": "",
            "Isolation": "",
            "VolumeDriver": "",
            "Sysctls": None,
            **host,
        }
        full_host["RestartPolicy"] = {**host["RestartPolicy"], "MaximumRetryCount": 0}
        for key in ("Tmpfs", "ExtraHosts", "GroupAdd"):
            full_host[key] = host.get(key) or None
        if networking:
            networks = {
                network: {"Aliases": list(endpoint["Aliases"]), "NetworkID": ""}
                for network, endpoint in networking["EndpointsConfig"].items()
            }
        else:
            networks = {host["NetworkMode"]: {"Aliases": None, "NetworkID": ""}}
        attrs = {
            "Id": identity,
            "Name": "/" + name,
            "Image": config["Image"],
            "AppArmorProfile": "docker-default",
            "State": {
                "Status": "created",
                "Running": False,
                "Paused": False,
                "ExitCode": 0,
                "Pid": 0,
                "StartedAt": "0001-01-01T00:00:00Z",
            },
            "Config": body,
            "HostConfig": full_host,
            "Mounts": mounts,
            "NetworkSettings": {"Networks": networks},
        }
        self.on_create(attrs)
        self.containers[identity] = attrs
        return {"Id": identity, "Warnings": []}

    def _get(self, reference):
        for identity, attrs in self.containers.items():
            if reference in (identity, attrs["Name"][1:]):
                return attrs
        raise NotFound(reference)

    def inspect_container(self, reference):
        return copy.deepcopy(self._get(reference))

    def start(self, container):
        attrs = self._get(container)
        name = attrs["Name"][1:]
        self._event("start", name)
        error = self.start_errors.get(name)
        if error is not None:
            raise error
        attrs["State"].update(
            Status="running", Running=True, Pid=4242, StartedAt="2026-09-30T00:00:00Z"
        )
        mode = attrs["HostConfig"]["NetworkMode"]
        if mode != "none":
            bridge = next(
                network.id
                for network in self.world.network.store.values()
                if network.attrs["Name"] == mode
            )
            attrs["NetworkSettings"]["Networks"][mode]["NetworkID"] = bridge
        check = attrs["Config"].get("Healthcheck") or {}
        if check.get("Test") and check["Test"][0] != "NONE":
            attrs["State"]["Health"] = {"Status": "starting", "Log": []}
        self.on_start.get(name, lambda attrs: None)(attrs)

    def exit(self, name, code):
        attrs = self._get(name)
        attrs["State"].update(
            Status="exited", Running=False, Paused=False, ExitCode=code, Pid=0
        )

    def health(self, name, status):
        self._get(name)["State"]["Health"] = {
            "Status": status,
            "Log": [{"Output": f"probe says {status}\n" * 200}],
        }

    def pause(self, container):
        attrs = self._get(container)
        self._event("pause", attrs["Name"][1:])
        attrs["State"].update(Status="paused", Paused=not self.fail_pause_proof)

    def unpause(self, container):
        attrs = self._get(container)
        self._event("unpause", attrs["Name"][1:])
        attrs["State"].update(Status="running", Paused=False)

    def kill(self, container, signal=None):
        attrs = self._get(container)
        self._event("kill", attrs["Name"][1:], attrs["State"]["Paused"])
        assert signal == "SIGKILL"
        if not attrs["State"]["Running"]:
            raise daemon_error("container is not running", 409)
        self.exit(attrs["Name"][1:], 137)

    def stop(self, container, timeout=None):
        attrs = self._get(container)
        self._event("stop", attrs["Name"][1:], timeout)
        self.exit(attrs["Name"][1:], 0)

    def remove_container(self, container, v=False, force=False):
        attrs = self._get(container)
        assert v is True and force is False
        assert not attrs["State"]["Running"], "removal follows proven termination"
        self._event("container-remove", attrs["Name"][1:])
        del self.containers[attrs["Id"]]

    def containers_list(self, filters):
        return [
            {
                "Id": attrs["Id"],
                "Names": [attrs["Name"]],
                "Labels": attrs["Config"]["Labels"],
                "State": attrs["State"]["Status"],
            }
            for attrs in self.containers.values()
            if _labels_match(attrs["Config"]["Labels"], filters)
        ]

    def logs(self, container, **kwargs):
        return b"x" * 5000 + b"last line\n"


class EngineAPI:
    """docker-py APIClient names for the engine (containers/volumes clash)."""

    def __init__(self, engine):
        self._engine = engine

    def __getattr__(self, name):
        return getattr(self._engine, name)

    def containers(self, all=False, filters=None, size=False):
        assert all is True
        return self._engine.containers_list(filters or {})

    def volumes(self, filters=None):
        return {"Volumes": self._engine.volumes_list(filters or {}), "Warnings": None}


class Journal:
    def __init__(self, events):
        self.events = events
        self.leases = []
        self.fail_at = None

    def __call__(self, lease):
        if self.fail_at is not None and len(self.leases) + 1 >= self.fail_at:
            raise OSError("journal disk full")
        self.events.append(("commit", lease.state))
        self.leases.append(lease)
        return lease


class EnvWorld:
    def __init__(self, *, no_new_privileges=True, paused_killer=None):
        self.events = []
        self.clock = FakeClock()
        self.network = NetworkWorld(events=self.events)
        self.engine = FakeEngine(self)
        self.network.client.api = EngineAPI(self.engine)
        self.timeline = []
        self.journal = Journal(self.events)
        self.backend = SandboxEnvDockerBackend(
            self.network.client,
            self.network.backend,
            no_new_privileges=no_new_privileges,
            paused_killer=paused_killer or DockerPausedKiller(self.network.client.api),
            clock=self.clock,
            sleep=self.sleep,
            poll_interval=0.25,
            stop_proof_sec=2.0,
        )

    def sleep(self, seconds):
        self.clock.now += seconds
        due = [item for item in self.timeline if item[0] <= self.clock.now]
        self.timeline = [item for item in self.timeline if item[0] > self.clock.now]
        for _, action in due:
            action()

    def at(self, delay, action):
        self.timeline.append((self.clock.now + delay, action))

    def images(self):
        return {
            MAIN_HANDLE: self.backend.inspect_image(MAIN_IMAGE),
            KV_HANDLE: self.backend.inspect_image(KV_IMAGE),
        }

    def plan(self, value=None, *, phase="judge", swap_ratio=1.0):
        return self.backend.plan(
            make_owner(phase),
            ENV_ID,
            value or spec(),
            self.images(),
            default_pids=512,
            swap_ratio=swap_ratio,
        )

    def planned(self, plan):
        return self.journal(plan.lease(created_at=100.0, expires_at=700.0))

    def create(self, value=None, **kwargs):
        plan = self.plan(value, **kwargs)
        lease = self.backend.create(plan, self.planned(plan), self.journal)
        return plan, lease

    def ready(self, value=None):
        plan, lease = self.create(value)
        self.at(
            1.0, lambda: self.engine.health(plan.services[0].container_name, "healthy")
        )
        result = self.backend.start(plan, lease, self.journal, wait_timeout_sec=30)
        assert result.state == "ready", result
        return plan, result.lease

    def container(self, plan, name):
        return self.engine._get(plan.service(name).container_name)

    def labelled(self, env_id=ENV_ID):
        selector = {"label": f"rsi-harness.sandbox-env={env_id}"}
        return (
            self.engine.containers_list(selector),
            self.engine.volumes_list(selector),
            self.engine.networks(selector),
        )


# -- plan and template --------------------------------------------------------


def test_template_is_the_fixed_low_level_body_with_endpoint_aliases():
    world = EnvWorld()
    plan = world.plan()
    kv, main = plan.services
    assert (kv.name, kv.idx, kv.container_name) == (
        "kv",
        0,
        env.env_container_name(ENV_ID, 0),
    )
    assert (main.name, main.idx) == ("main", 1)
    assert plan.order == ("kv", "main")

    body = main.body()
    host = body["HostConfig"]
    assert host == {
        "Runtime": "runc",
        "Privileged": False,
        "CapDrop": ["CAP_NET_RAW"],
        "SecurityOpt": ["no-new-privileges:true", "apparmor=docker-default"],
        "Init": True,
        "IpcMode": "private",
        "CgroupnsMode": "private",
        "NanoCpus": 10**9,
        "Memory": 1024 * MIB,
        "MemorySwap": 2048 * MIB,
        "PidsLimit": 512,
        "Ulimits": [{"Name": "nofile", "Soft": 65536, "Hard": 65536}],
        "ShmSize": 64 * MIB,
        "Tmpfs": {"/scratch": f"rw,nosuid,nodev,exec,size={64 * MIB}"},
        "Mounts": [
            {
                "Type": "volume",
                "Source": env.env_volume_name(ENV_ID, 0),
                "Target": "/data",
                "VolumeOptions": {"NoCopy": True},
            }
        ],
        "NetworkMode": plan.network.name,
        "ExtraHosts": ["mirror.internal:203.0.113.7"],
        "LogConfig": {
            "Type": "json-file",
            "Config": {"max-size": "1m", "max-file": "1"},
        },
        "RestartPolicy": {"Name": "no"},
        "ReadonlyRootfs": False,
        "GroupAdd": None,
        "AutoRemove": False,
        "OomKillDisable": False,
        "PublishAllPorts": False,
    }
    for forbidden in ("Binds", "Devices", "DeviceRequests", "PortBindings", "Dns"):
        assert forbidden not in host
    assert body["NetworkingConfig"] == {
        "EndpointsConfig": {plan.network.name: {"Aliases": ["main"]}}
    }
    assert body["Image"] == MAIN_IMAGE
    assert body["Cmd"] == ["sleep", "infinity"]
    assert body["WorkingDir"] == "/app"
    assert "Hostname" not in body and "Healthcheck" not in body
    assert kv.aliases == ("kv", "kvstore")
    assert kv.body()["HostConfig"]["PidsLimit"] == 256
    assert kv.body()["HostConfig"]["NanoCpus"] == 5 * 10**8
    assert (plan.cpus_milli, plan.memory_mb, plan.disk_mb) == (1500, 1280, 2048)
    assert plan.swap_mb == 1280


@pytest.mark.parametrize(
    ("swap_ratio", "main_swap_mb", "kv_swap_mb"),
    [(0.0, 0, 0), (0.5, 512, 128), (0.3, 307, 76), (1.0, 1024, 256)],
)
def test_swap_is_floor_of_memory_times_the_grant_ratio(
    swap_ratio, main_swap_mb, kv_swap_mb
):
    world = EnvWorld()
    plan = world.plan(swap_ratio=swap_ratio)
    kv, main = plan.services
    for service, swap in ((main, main_swap_mb), (kv, kv_swap_mb)):
        host = service.body()["HostConfig"]
        assert host["MemorySwap"] == host["Memory"] + swap * MIB
        # Swappiness is never sent: the kernel default stays.
        assert "MemorySwappiness" not in host
    assert plan.swap_mb == main_swap_mb + kv_swap_mb
    _, lease = world.create(swap_ratio=swap_ratio)
    world.backend.attest(plan, lease)


def test_every_object_carries_the_exact_shared_label_set():
    world = EnvWorld()
    plan = world.plan()
    base = {
        "rsi-harness.run-id": "run-1",
        "rsi-harness.task-id": "task",
        "rsi-harness.sandbox-phase": "judge",
        "rsi-harness.round-id": "agent-1",
    }
    kv_labels = plan.services[0].body()["Labels"]
    assert kv_labels == {
        **base,
        "rsi-harness.role": "sandbox-env",
        "rsi-harness.sandbox-env": ENV_ID,
        "rsi-harness.sandbox-service": "kv",
        "rsi-harness.sandbox-image": KV_HANDLE,
    }
    main_labels = plan.services[1].body()["Labels"]
    assert main_labels["org.opencontainers.image.version"] == "24.04"
    assert {
        key: value for key, value in main_labels.items() if key.startswith("rsi-")
    } == {
        **base,
        "rsi-harness.role": "sandbox-env",
        "rsi-harness.sandbox-env": ENV_ID,
        "rsi-harness.sandbox-service": "main",
        "rsi-harness.sandbox-image": MAIN_HANDLE,
    }
    for volume in plan.volumes:
        assert volume.labels == {
            **base,
            "rsi-harness.role": "sandbox-env-vol",
            "rsi-harness.sandbox-env": ENV_ID,
        }
    assert plan.network.labels == {
        **base,
        "rsi-harness.role": "sandbox-env-net",
        "rsi-harness.sandbox-env": ENV_ID,
    }
    # v1 children use the same helper: only role and handle keys differ.
    child = sandbox_labels(make_child(owner=make_owner("judge")))
    assert {key: child[key] for key in base} == base


def test_env_is_image_env_then_spec_with_nvidia_blanked_and_devices_void():
    world = EnvWorld()
    raw = make_env_spec()
    raw["services"]["kv"]["env"] = {"REDIS_VERSION": "8", "EXTRA": "1"}
    kv = world.plan(spec(raw)).services[0]

    assert kv.body()["Env"] == [
        "PATH=/usr/bin:/bin",
        "REDIS_VERSION=8",
        "NVIDIA_DRIVER_CAPABILITIES=",
        "EXTRA=1",
        "NVIDIA_VISIBLE_DEVICES=void",
    ]
    with pytest.raises(SandboxError, match="reserved"):
        env.parse_env_spec(make_env_spec(env={"NVIDIA_VISIBLE_DEVICES": "all"}))


@pytest.mark.parametrize(
    ("entrypoint", "command", "expected"),
    (
        (None, None, (["docker-entrypoint.sh"], ["redis-server"])),
        (
            None,
            ["redis-server", "--save", ""],
            (["docker-entrypoint.sh"], ["redis-server", "--save", ""]),
        ),
        (["redis-server"], None, (["redis-server"], None)),
        (["sh", "-c"], ["exec redis-server"], (["sh", "-c"], ["exec redis-server"])),
        ([], None, ([], ["redis-server"])),
    ),
)
def test_entrypoint_override_clears_the_image_cmd_like_docker(
    entrypoint, command, expected
):
    raw = make_env_spec()
    kv = raw["services"]["kv"]
    if entrypoint is not None:
        kv["entrypoint"] = entrypoint
    if command is not None:
        kv["command"] = command
    body = EnvWorld().plan(spec(raw)).services[0].body()
    assert (body["Entrypoint"], body["Cmd"]) == expected
    assert body["StopSignal"] == "SIGINT"


@pytest.mark.parametrize(
    ("image_dir", "planned"),
    (
        ("/testbed/", "/testbed"),
        ("/srv//app/./", "/srv/app"),
        ("//data", "/data"),
        ("/", "/"),
        ("", ""),
    ),
)
def test_an_image_working_dir_is_planned_as_the_daemon_cleans_it(image_dir, planned):
    """The swebench images say WORKDIR /testbed/; dockerd creates and
    inspects the container as /testbed, which the plan must expect."""
    world = EnvWorld()
    world.engine.images[KV_IMAGE] = kv_image(WorkingDir=image_dir)

    plan, lease = world.create()

    assert plan.service("kv").body()["WorkingDir"] == planned
    assert world.container(plan, "kv")["Config"]["WorkingDir"] == planned
    world.backend.attest(plan, lease)


def test_image_volumes_get_implicit_labelled_volumes_unless_exactly_covered():
    world = EnvWorld()
    world.engine.images[KV_IMAGE]["Config"]["Volumes"] = {
        "/data/": {},
        "/var/lib/redis": {},
        "/cache": {},
    }
    raw = make_env_spec()
    raw["services"]["kv"]["tmpfs"] = {"/cache": 16}
    raw["services"]["kv"]["mounts"].append(
        {"volume": "shared", "target": "/var/lib/redis/sub", "read_only": False}
    )
    raw["volumes"]["cache"] = {"seeded": False}
    raw["services"]["main"]["mounts"].append({"volume": "cache", "target": "/cache"})
    plan = world.plan(spec(raw))

    assert [(volume.idx, volume.logical) for volume in plan.volumes] == [
        (0, "cache"),
        (1, "shared"),
        (2, "implicit:0:" + hashlib.sha256(b"/data").hexdigest()[:16]),
        (3, "implicit:0:" + hashlib.sha256(b"/var/lib/redis").hexdigest()[:16]),
    ]
    kv = plan.services[0]
    # Only an identical destination covers a VOLUME; a nested mount does not.
    assert kv.implicit_volumes == ("/data", "/var/lib/redis")
    assert kv.body()["HostConfig"]["Mounts"][-2:] == [
        {"Type": "volume", "Source": env.env_volume_name(ENV_ID, 2), "Target": "/data"},
        {
            "Type": "volume",
            "Source": env.env_volume_name(ENV_ID, 3),
            "Target": "/var/lib/redis",
        },
    ]
    lease = plan.lease(created_at=1.0, expires_at=2.0)
    assert [volume.planned_name for volume in lease.volumes] == [
        env.env_volume_name(ENV_ID, index) for index in range(4)
    ]


@pytest.mark.parametrize(
    ("config", "match"),
    (
        ({"Volumes": {"/proc/x": {}}}, "reserved"),
        ({"Volumes": {"relative": {}}}, "reserved or invalid"),
        ({"Labels": {"rsi-harness.sandbox-id": "0" * 32}}, "reserved label"),
        ({"Env": ["A=b"] * 257}, "bounded metadata"),
    ),
)
def test_unsafe_image_metadata_is_refused(config, match):
    world = EnvWorld()
    world.engine.images[KV_IMAGE]["Config"].update(config)
    with pytest.raises(SandboxError, match=match):
        world.plan()


def test_work_images_may_not_carry_a_judge_round_label():
    world = EnvWorld()
    world.engine.images[KV_IMAGE]["Config"]["Labels"] = {
        "rsi-harness.run-id": "other-run",
        "rsi-harness.round-id": "agent-9",
    }
    # Keys the container sets itself are overridden by the exact labels...
    judge = world.plan().services[0].body()["Labels"]
    assert judge["rsi-harness.run-id"] == "run-1"
    assert judge["rsi-harness.round-id"] == "agent-1"
    # ...but a Work container has no round-id of its own to override it with.
    with pytest.raises(SandboxError, match="rsi-harness.round-id"):
        world.plan(phase="work")


def test_an_image_built_for_the_session_keeps_exact_container_labels():
    world = EnvWorld()
    owner = make_owner("judge")
    # What M8 forces onto a built image with --opt label: (spec 3.2, B8).
    world.engine.images[KV_IMAGE]["Config"]["Labels"] = env.sandbox_object_labels(
        owner, env.BUILD_ROLE, {"sandbox-image": KV_HANDLE}
    )
    plan, lease = world.create()
    # Every inherited key is overridden: the container set stays exact.
    assert world.container(plan, "kv")["Config"]["Labels"] == (
        env.env_container_labels(owner, ENV_ID, "kv", KV_HANDLE)
    )
    assert lease.services[0].image == KV_HANDLE
    world.backend.attest(plan, lease)


def test_unknown_image_handle_is_a_permission_error():
    world = EnvWorld()
    images = world.images()
    del images[KV_HANDLE]
    with pytest.raises(SandboxError) as caught:
        plan_env(make_owner(), ENV_ID, spec(), images, default_pids=64, swap_ratio=1.0)
    assert caught.value.code == "permission"


def test_healthcheck_merge_mirrors_the_daemon():
    world = EnvWorld()
    kv = world.plan().services[0].body()
    # Spec retries/interval win; zero timings come from the image check.
    assert kv["Healthcheck"] == {
        "Test": ["CMD", "redis-cli", "ping"],
        "Interval": 10**9,
        "Timeout": 30 * 10**9,
        "StartInterval": 5 * 10**9,
        "Retries": 30,
    }
    raw = make_env_spec()
    raw["services"]["kv"]["healthcheck"] = "none"
    raw["services"]["main"]["depends_on"] = {}
    kv = world.plan(spec(raw)).services[0]
    assert kv.body()["Healthcheck"] == {"Test": ["NONE"], "Interval": 5 * 10**9}
    assert kv.has_healthcheck is False
    raw["services"]["kv"]["healthcheck"] = "image"
    assert world.plan(spec(raw)).services[0].body()["Healthcheck"] == {
        "Test": ["CMD", "true"],
        "Interval": 5 * 10**9,
    }
    del world.engine.images[KV_IMAGE]["Config"]["Healthcheck"]
    with pytest.raises(SandboxError, match="no healthcheck"):
        world.plan(spec(raw))


def test_no_new_privileges_is_the_operator_switch():
    host = EnvWorld(no_new_privileges=False).plan().services[1].host_config
    assert host["SecurityOpt"] == ["apparmor=docker-default"]


def test_spec_cap_drop_only_narrows_the_base_set():
    raw = make_env_spec(cap_drop=["CAP_SYS_CHROOT", "CAP_NET_RAW", "CAP_MKNOD"])
    host = EnvWorld().plan(spec(raw)).services[1].host_config
    assert host["CapDrop"] == ["CAP_NET_RAW", "CAP_MKNOD", "CAP_SYS_CHROOT"]
    assert "CapAdd" not in host


def test_single_service_none_env_has_no_bridge_and_no_aliases():
    raw = make_env_spec()
    raw["network"] = "none"
    del raw["services"]["kv"]
    raw["services"]["main"]["depends_on"] = {}
    plan = EnvWorld().plan(spec(raw))
    (main,) = plan.services
    assert plan.network is None
    assert (main.network_mode, main.aliases) == ("none", None)
    assert "NetworkingConfig" not in main.body()
    lease = plan.lease(created_at=1.0, expires_at=2.0)
    assert (lease.network_name, lease.rule_id) == (None, None)


# -- create, attest, rollback -------------------------------------------------


def test_create_journals_then_installs_rule_bridge_volumes_containers_without_start():
    world = EnvWorld()
    plan, lease = world.create()

    kinds = [event[0] for event in world.events]
    assert kinds.index("install") < kinds.index("network-create")
    assert kinds.index("network-create") < kinds.index("volume-create")
    assert kinds.index("volume-create") < kinds.index("container-create")
    assert "start" not in kinds
    # The whole plan was journaled before the first Docker call.
    assert world.events[0] == ("commit", "planned")
    assert not any(volume.created for volume in world.journal.leases[0].volumes)
    assert lease.state == "created" and not lease.pending_mutation
    assert lease.network_id is not None
    assert all(record.state == "created" for record in lease.services)
    assert all(volume.created for volume in lease.volumes)
    world.backend.attest(plan, lease)
    assert world.network.firewall.installed == {
        plan.network.rule_id: world.network.backend.rules(plan.network)
    }


def test_volume_is_journaled_before_its_create_call():
    world = EnvWorld()
    plan = world.plan()
    lease = world.planned(plan)
    seen = []

    def create_volume(name, **kwargs):
        seen.append(world.journal.leases[-1].volumes)
        return FakeEngine.create_volume(world.engine, name, **kwargs)

    world.engine.create_volume = create_volume
    world.backend.create(plan, lease, world.journal)
    for index, volumes in enumerate(seen):
        assert volumes[index].created is True


def with_host(key, value):
    return lambda world, plan, lease: world.container(plan, "main")[
        "HostConfig"
    ].__setitem__(key, value)


def with_config(key, value):
    return lambda world, plan, lease: world.container(plan, "main")[
        "Config"
    ].__setitem__(key, value)


def with_label(key, value):
    return lambda world, plan, lease: world.container(plan, "main")["Config"][
        "Labels"
    ].__setitem__(key, value)


def with_volume(key, value):
    return lambda world, plan, lease: world.engine.volumes[
        plan.volumes[0].name
    ].__setitem__(key, value)


def second_network(world, plan, lease):
    world.container(plan, "main")["NetworkSettings"]["Networks"]["bridge"] = {}


def alias_drift(world, plan, lease):
    networks = world.container(plan, "main")["NetworkSettings"]["Networks"]
    networks[plan.network.name]["Aliases"] = ["main", "kvstore"]


def external_volume(world, plan, lease):
    world.container(plan, "main")["Mounts"][0]["Name"] = "someone-elses-volume"


def anonymous_volume(world, plan, lease):
    world.container(plan, "main")["Mounts"].append(
        {
            "Type": "volume",
            "Name": "f" * 64,
            "Destination": "/var/lib/x",
            "Driver": "local",
            "RW": True,
        }
    )


def bind_mount(world, plan, lease):
    world.container(plan, "main")["Mounts"].append(
        {
            "Type": "bind",
            "Source": "/var/run/docker.sock",
            "Destination": "/run/docker.sock",
            "RW": True,
        }
    )


def volume_bind_options(world, plan, lease):
    world.engine.volumes[plan.volumes[0].name]["Options"] = {
        "type": "none",
        "o": "bind",
        "device": "/",
    }


def flip_internal(world, plan, lease):
    attrs = world.network.store[lease.network_id].attrs
    attrs["Internal"] = not attrs["Internal"]


def env_drift(world, plan, lease):
    world.container(plan, "main")["Config"]["Env"].append("NVIDIA_VISIBLE_DEVICES=all")


@pytest.mark.parametrize(
    ("mutate", "field"),
    (
        (with_host("Privileged", True), "Privileged"),
        (with_host("CapAdd", ["SYS_ADMIN"]), "CapAdd"),
        (with_host("Binds", ["/:/host"]), "Binds"),
        (
            with_host("PortBindings", {"80/tcp": [{"HostIp": "", "HostPort": "80"}]}),
            "PortBindings",
        ),
        (with_host("PublishAllPorts", True), "PublishAllPorts"),
        (with_host("Runtime", "nvidia"), "Runtime"),
        (with_host("Init", False), "Init"),
        (with_host("LogConfig", {"Type": "journald", "Config": {}}), "LogConfig"),
        (
            with_host("RestartPolicy", {"Name": "always", "MaximumRetryCount": 0}),
            "RestartPolicy",
        ),
        (second_network, "Networks"),
        (alias_drift, "Aliases"),
        (external_volume, "Mounts"),
        (anonymous_volume, "Mounts"),
        (volume_bind_options, "Options"),
        (flip_internal, "Internal"),
        (with_host("Dns", ["8.8.8.8"]), "Dns"),
        (with_host("Sysctls", {"net.ipv4.ip_forward": "1"}), "Sysctls"),
        (env_drift, "Env"),
        # Beyond the required matrix: every other field spec 5 lists.
        (with_host("NetworkMode", "host"), "NetworkMode"),
        (with_host("Devices", [{"PathOnHost": "/dev/nvidia0"}]), "Devices"),
        (
            with_host("DeviceRequests", [{"Driver": "nvidia", "Count": -1}]),
            "DeviceRequests",
        ),
        (with_host("DeviceCgroupRules", ["c *:* rwm"]), "DeviceCgroupRules"),
        (with_host("Links", ["/other:/alias"]), "Links"),
        (with_host("VolumesFrom", ["other"]), "VolumesFrom"),
        (with_host("PidMode", "host"), "PidMode"),
        (with_host("UTSMode", "host"), "UTSMode"),
        (with_host("UsernsMode", "host"), "UsernsMode"),
        (with_host("Cgroup", "host"), "Cgroup"),
        (with_host("IpcMode", "host"), "IpcMode"),
        (with_host("CgroupnsMode", "host"), "CgroupnsMode"),
        (with_host("CgroupParent", "/"), "CgroupParent"),
        (with_host("SecurityOpt", ["apparmor=unconfined"]), "SecurityOpt"),
        (with_host("CapDrop", []), "CapDrop"),
        (with_host("NanoCpus", 0), "NanoCpus"),
        (with_host("Memory", 0), "Memory"),
        (with_host("MemorySwap", -1), "MemorySwap"),
        (with_host("MemorySwap", 1024 * MIB), "MemorySwap"),
        (with_host("MemorySwappiness", 0), "MemorySwappiness"),
        (with_host("MemorySwappiness", 60), "MemorySwappiness"),
        (with_host("PidsLimit", None), "PidsLimit"),
        (with_host("Ulimits", None), "Ulimits"),
        (with_host("ShmSize", 128 * MIB), "ShmSize"),
        (with_host("OomKillDisable", True), "OomKillDisable"),
        (with_host("ReadonlyRootfs", True), "ReadonlyRootfs"),
        (with_host("AutoRemove", True), "AutoRemove"),
        (with_host("DnsSearch", ["corp.example"]), "DnsSearch"),
        (with_host("DnsOptions", ["ndots:0"]), "DnsOptions"),
        (with_host("Isolation", "hyperv"), "Isolation"),
        (with_host("VolumeDriver", "nfs"), "VolumeDriver"),
        (with_host("Tmpfs", {"/scratch": "rw,size=1g"}), "Tmpfs"),
        (with_host("ExtraHosts", ["metadata:169.254.169.254"]), "ExtraHosts"),
        (with_host("GroupAdd", ["0"]), "GroupAdd"),
        (bind_mount, "Mounts"),
        (with_config("Image", KV_IMAGE), "Image"),
        (with_config("User", "0:0"), "User"),
        (with_config("Cmd", ["sh"]), "Cmd"),
        (with_config("Entrypoint", ["/bin/sh"]), "Entrypoint"),
        (with_config("WorkingDir", "/"), "WorkingDir"),
        (with_config("Hostname", "renamed"), "Hostname"),
        (with_config("StopSignal", "SIGKILL"), "StopSignal"),
        (with_config("StopTimeout", 0), "StopTimeout"),
        (with_config("Tty", True), "Tty"),
        (with_config("Healthcheck", {"Test": ["NONE"]}), "Healthcheck"),
        (with_label("com.example.added", "1"), "Labels"),
        (with_config("OpenStdin", True), "Stdin"),
        (with_volume("Driver", "nfs"), "Driver"),
        (with_volume("Labels", {"owner": "someone"}), "Labels"),
        (with_volume("Scope", "global"), "Scope"),
    ),
)
def test_attest_mutation_matrix_every_drift_fails(mutate, field):
    world = EnvWorld()
    plan, lease = world.create()
    world.backend.attest(plan, lease)

    mutate(world, plan, lease)

    with pytest.raises(InfrastructureError, match=rf"mismatch: {field}$"):
        world.backend.attest(plan, lease)


def test_a_changed_rsi_label_fails_ownership_before_attest():
    world = EnvWorld()
    plan, lease = world.create()
    with_config("Labels", {"rsi-harness.run-id": "run-1"})(world, plan, lease)
    with pytest.raises(InfrastructureError, match="recovery_required.*not owned"):
        world.backend.attest(plan, lease)


def test_attest_service_checks_the_top_level_image_too():
    world = EnvWorld()
    plan, lease = world.create()
    attrs = copy.deepcopy(world.container(plan, "main"))
    main = plan.service("main")
    SandboxEnvDockerBackend.attest_service(main, attrs, network_id=lease.network_id)
    attrs["Image"] = KV_IMAGE
    with pytest.raises(InfrastructureError, match=r"mismatch: Image$"):
        SandboxEnvDockerBackend.attest_service(main, attrs, network_id=lease.network_id)


def test_post_start_attest_requires_docker_default_apparmor():
    world = EnvWorld()
    plan, lease = world.ready()
    world.backend.attest(plan, lease)
    world.container(plan, "main")["AppArmorProfile"] = "unconfined"
    with pytest.raises(InfrastructureError, match="AppArmorProfile"):
        world.backend.attest(plan, lease)
    world.container(plan, "main")["AppArmorProfile"] = "docker-default"
    world.container(plan, "main")["NetworkSettings"]["Networks"][plan.network.name][
        "NetworkID"
    ] = "0" * 64
    with pytest.raises(InfrastructureError, match="NetworkID"):
        world.backend.attest(plan, lease)


def test_drift_at_create_rolls_the_whole_env_back_and_proves_it():
    world = EnvWorld()

    def drift(attrs):
        if attrs["Name"].endswith("-1"):
            attrs["HostConfig"]["Privileged"] = True

    world.engine.on_create = drift
    with pytest.raises(SetupError, match="proven absent"):
        world.create()

    removed = world.journal.leases[-1]
    assert (removed.state, removed.reason) == ("removed", "start_failed")
    assert world.labelled() == ([], [], [])
    assert world.engine.containers == {} and world.engine.volumes == {}
    assert world.network.firewall.installed == {}
    kinds = [event[0] for event in world.events]
    assert kinds.index("container-remove") < kinds.index("volume-remove")
    assert kinds.index("volume-remove") < kinds.index("network-remove")
    assert kinds.index("network-remove") < kinds.index("remove")


def test_answered_container_create_failure_rolls_back():
    world = EnvWorld()
    world.engine.create_errors[env.env_container_name(ENV_ID, 1)] = daemon_error(
        "invalid capability", 400
    )
    with pytest.raises(SetupError, match="invalid capability"):
        world.create()
    assert world.journal.leases[-1].state == "removed"
    assert world.labelled() == ([], [], [])


@pytest.mark.parametrize("kind", ("container", "volume"))
def test_unanswered_create_retains_everything_for_recovery(kind):
    world = EnvWorld()
    name = (
        env.env_container_name(ENV_ID, 1)
        if kind == "container"
        else env.env_volume_name(ENV_ID, 1)
    )
    world.engine.create_errors[name] = requests.exceptions.ReadTimeout("timed out")
    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.create()
    last = world.journal.leases[-1]
    assert last.state == "planned" and last.pending_mutation
    assert "container-remove" not in [event[0] for event in world.events]
    assert world.network.firewall.installed


def test_foreign_volume_holding_a_planned_name_is_never_adopted():
    world = EnvWorld()
    world.engine.volumes[env.env_volume_name(ENV_ID, 1)] = {
        "Name": env.env_volume_name(ENV_ID, 1),
        "Driver": "local",
        "Labels": {"owner": "someone"},
        "Options": None,
        "Scope": "local",
    }
    with pytest.raises(SetupError, match="already exists"):
        world.create()
    # Rolled back without touching the foreign volume.
    assert list(world.engine.volumes) == [env.env_volume_name(ENV_ID, 1)]


def test_journal_failure_stops_every_further_mutation():
    world = EnvWorld()
    plan = world.plan()
    lease = world.planned(plan)
    world.journal.fail_at = 3
    with pytest.raises(OSError, match="journal disk full"):
        world.backend.create(plan, lease, world.journal)
    kinds = [event[0] for event in world.events]
    # No rollback either: the journal can no longer record it.
    assert "network-remove" not in kinds and "volume-remove" not in kinds


# -- start --------------------------------------------------------------------


def test_start_waits_for_a_healthy_dependency_then_is_ready():
    world = EnvWorld()
    plan, lease = world.create()
    kv = plan.services[0].container_name
    world.at(2.0, lambda: world.engine.health(kv, "healthy"))

    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)

    assert (result.state, result.reason) == ("ready", None)
    starts = [
        (event[1], world.clock.now) for event in world.events if event[0] == "start"
    ]
    assert [name for name, _ in starts] == [kv, plan.services[1].container_name]
    assert result.lease.state == "ready"
    assert [record.state for record in result.lease.services] == ["running", "running"]
    assert world.journal.leases[0].state == "planned"
    assert "starting" in [lease.state for lease in world.journal.leases]


def test_ready_needs_every_healthcheck_to_pass():
    world = EnvWorld()
    raw = make_env_spec(healthcheck={"test": ["CMD", "true"]})
    raw["services"]["main"]["depends_on"] = {"kv": {"condition": "started"}}
    plan, lease = world.create(spec(raw))
    world.at(
        1.0, lambda: world.engine.health(plan.services[0].container_name, "healthy")
    )
    world.at(
        3.0, lambda: world.engine.health(plan.services[1].container_name, "healthy")
    )
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert result.state == "ready"
    assert world.clock.now >= 103.0


def test_unhealthy_dependency_fails_the_env_without_starting_dependents():
    world = EnvWorld()
    plan, lease = world.create()
    world.at(
        1.0, lambda: world.engine.health(plan.services[0].container_name, "unhealthy")
    )

    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)

    assert (result.state, result.reason, result.service) == (
        "failed",
        "unhealthy",
        "kv",
    )
    assert plan.services[1].container_name not in [
        e[1] for e in world.events if e[0] == "start"
    ]
    # A failed env stays as it is until destroyed; it is a user result.
    assert world.container(plan, "kv")["State"]["Running"]
    diagnostics = world.backend.diagnostics(result.lease, "kv")
    assert len(diagnostics["health_tail"].encode()) <= 1024
    assert "unhealthy" in diagnostics["health_tail"]
    assert diagnostics["log_tail"].endswith("last line\n")
    assert len(diagnostics["log_tail"].encode()) <= 4096


def test_wait_timeout_fails_the_env():
    world = EnvWorld()
    plan, lease = world.create()
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=5)
    assert (result.state, result.reason) == ("failed", "wait_timeout")
    assert result.lease.state == "failed" and result.lease.reason == "wait_timeout"
    assert 105.0 <= world.clock.now <= 105.5


def completion_spec():
    raw = make_env_spec()
    raw["services"]["kv"]["healthcheck"] = None
    raw["services"]["main"]["depends_on"] = {
        "kv": {"condition": "completed_successfully", "required": True}
    }
    return spec(raw)


@pytest.mark.parametrize(
    ("code", "state", "reason"), ((0, "ready", None), (3, "failed", "start_failed"))
)
def test_completed_successfully_target_may_exit_zero_only(code, state, reason):
    world = EnvWorld()
    plan, lease = world.create(completion_spec())
    kv, main = (service.container_name for service in plan.services)
    exited, started = [], []

    def exit_later(attrs):
        world.at(1.0, lambda: (world.engine.exit(kv, code), exited.append(1)))

    world.engine.on_start[kv] = exit_later
    world.engine.on_start[main] = lambda attrs: started.append(bool(exited))
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert (result.state, result.reason) == (state, reason)
    if state == "ready":
        # main started only once kv had exited 0.
        assert started == [True]
        assert [record.state for record in result.lease.services] == [
            "exited",
            "running",
        ]
    else:
        assert started == []


def test_an_unexpected_exit_fails_the_env():
    world = EnvWorld()
    raw = make_env_spec()
    raw["services"]["main"]["depends_on"] = {"kv": {"condition": "started"}}
    raw["services"]["kv"]["healthcheck"] = "none"
    plan, lease = world.create(spec(raw))
    main = plan.services[1].container_name
    world.engine.on_start[main] = lambda attrs: world.engine.exit(main, 0)
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert (result.state, result.reason, result.service) == (
        "failed",
        "start_failed",
        "main",
    )


def test_optional_dependency_may_fail():
    world = EnvWorld()
    raw = make_env_spec()
    raw["services"]["main"]["depends_on"] = {
        "kv": {"condition": "healthy", "required": False}
    }
    plan, lease = world.create(spec(raw))
    world.at(
        1.0, lambda: world.engine.health(plan.services[0].container_name, "unhealthy")
    )
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert result.state == "ready"
    assert world.container(plan, "main")["State"]["Running"]


@pytest.mark.parametrize(
    ("status", "explanation", "detail"),
    (
        (
            400,
            'runc create failed: exec: "redis-server": executable file not found '
            "in $PATH",
            "entrypoint executable not found in the image",
        ),
        # Docker 29 answers an unknown user with 500 (VERIFIED): still the
        # user's failure, since the service is proven not running.
        (
            500,
            "unable to find user redis: no matching entries in passwd file",
            "user not found in the image",
        ),
        (
            500,
            "error mounting /var/lib/docker/volumes/x/_data to rootfs",
            "the daemon refused to start the service",
        ),
    ),
)
def test_an_answered_start_refusal_is_a_user_failure_with_a_fixed_detail(
    status, explanation, detail
):
    world = EnvWorld()
    plan, lease = world.create()
    world.engine.start_errors[plan.services[0].container_name] = daemon_error(
        explanation, status
    )
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert (result.state, result.reason, result.service, result.detail) == (
        "failed",
        "start_failed",
        "kv",
        detail,
    )
    assert world.container(plan, "kv")["State"]["Status"] == "created"


def lost_start(world):
    def start(container):
        FakeEngine.start(world.engine, container)  # dockerd starts it anyway
        raise requests.exceptions.ReadTimeout("timed out")

    return start


def refused_but_running(world):
    def start(container):
        FakeEngine.start(world.engine, container)
        raise daemon_error("post-start hook failed", 500)

    return start


@pytest.mark.parametrize("start", (lost_start, refused_but_running))
def test_a_start_that_may_run_unattested_quarantines_the_env(start):
    world = EnvWorld()
    plan, lease = world.create()
    world.engine.start = start(world)
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert (result.state, result.reason, result.service, result.detail) == (
        "failed",
        "quarantined",
        "kv",
        "service start outcome is unknown",
    )
    assert not world.container(plan, "kv")["State"]["Running"]
    assert result.lease.services[0].state == "exited"
    assert (result.lease.state, result.lease.reason) == ("failed", "quarantined")


def test_cancelled_start_returns_canceled_before_starting():
    world = EnvWorld()
    plan, lease = world.create()
    result = world.backend.start(
        plan, lease, world.journal, wait_timeout_sec=30, cancelled=lambda: True
    )
    assert (result.state, result.reason) == ("failed", "canceled")
    assert "start" not in [event[0] for event in world.events]


def test_cancel_stops_further_starts_and_leaves_started_services_until_destroy():
    world = EnvWorld()
    raw = make_env_spec()
    raw["services"]["main"]["depends_on"] = {"kv": {"condition": "started"}}
    raw["services"]["kv"]["healthcheck"] = "none"
    plan, lease = world.create(spec(raw))
    kv = plan.services[0].container_name
    cancelled = []
    # Cancelled between kv's start and main's, with main already due.
    world.engine.on_start[kv] = lambda attrs: cancelled.append(True)

    result = world.backend.start(
        plan,
        lease,
        world.journal,
        wait_timeout_sec=30,
        cancelled=lambda: bool(cancelled),
    )

    assert (result.state, result.reason) == ("failed", "canceled")
    assert [event[1] for event in world.events if event[0] == "start"] == [kv]
    # Documented: what already runs keeps running until destroy.
    assert world.container(plan, "kv")["State"]["Running"]
    assert [record.state for record in result.lease.services] == ["running", "created"]
    removed = world.backend.destroy(result.lease, world.journal)
    assert removed.state == "removed" and world.engine.containers == {}


def test_drift_before_start_quarantines_without_starting():
    world = EnvWorld()
    plan, lease = world.create()
    world.container(plan, "kv")["HostConfig"]["Privileged"] = True
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert (result.state, result.reason, result.service) == (
        "failed",
        "quarantined",
        "kv",
    )
    assert "start" not in [event[0] for event in world.events]


def test_drift_after_start_quarantines_and_kills_the_env():
    world = EnvWorld()
    plan, lease = world.create()
    kv = plan.services[0].container_name
    world.engine.on_start[kv] = lambda attrs: attrs["HostConfig"].update(
        Privileged=True
    )
    result = world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)
    assert (result.state, result.reason) == ("failed", "quarantined")
    assert not world.container(plan, "kv")["State"]["Running"]
    assert result.lease.services[0].state == "exited"


def test_start_requires_the_journaled_inventory():
    world = EnvWorld()
    plan, lease = world.create()
    del world.engine.containers[lease.services[1].container_id]
    with pytest.raises(InfrastructureError, match="missing main"):
        world.backend.start(plan, lease, world.journal, wait_timeout_sec=30)


# -- runtime ------------------------------------------------------------------


def test_pause_and_resume_every_running_service_with_proof():
    world = EnvWorld()
    plan, lease = world.ready()
    admitted = []

    lease = world.backend.pause(lease, world.journal)
    assert lease.state == "paused"
    assert [record.state for record in lease.services] == ["paused", "paused"]
    assert all(
        world.container(plan, name)["State"]["Paused"] for name in ("kv", "main")
    )

    class Admission:
        def __enter__(self):
            admitted.append("enter")

        def __exit__(self, *exc):
            admitted.append("exit")

    lease = world.backend.resume(lease, world.journal, admission=Admission)
    assert lease.state == "ready"
    assert [record.state for record in lease.services] == ["running", "running"]
    assert admitted == ["enter", "exit"] * 2


def test_a_starting_env_cannot_pause():
    world = EnvWorld()
    plan, lease = world.create()
    starting = world.journal(lease.model_copy(update={"state": "starting"}))
    with pytest.raises(SandboxError) as caught:
        world.backend.pause(starting, world.journal)
    assert caught.value.code == "busy"
    assert "pause" not in [event[0] for event in world.events]


def test_unproven_pause_is_an_infrastructure_error():
    world = EnvWorld()
    plan, lease = world.ready()
    world.engine.fail_pause_proof = True
    with pytest.raises(InfrastructureError, match="pause could not be proven"):
        world.backend.pause(lease, world.journal)


def test_stop_service_leaves_the_env_and_refuses_paused_services():
    world = EnvWorld()
    plan, lease = world.ready()

    lease, code = world.backend.stop_service(
        lease, world.journal, "kv", timeout_sec=7.5
    )
    assert code == 0
    assert ("stop", plan.services[0].container_name, 8) in world.events
    assert lease.services[0].state == "exited"
    assert world.container(plan, "main")["State"]["Running"]

    lease = world.backend.pause(lease, world.journal)
    with pytest.raises(SandboxError) as caught:
        world.backend.stop_service(lease, world.journal, "main", timeout_sec=1)
    assert caught.value.code == "busy"
    for bad in (-1, 61, float("nan"), "5"):
        with pytest.raises(SandboxError, match="0..60"):
            world.backend.stop_service(lease, world.journal, "kv", timeout_sec=bad)


def test_a_failed_stop_call_is_settled_by_sigkill_and_proof():
    world = EnvWorld()
    plan, lease = world.ready()

    def lost(container, timeout=None):
        raise requests.exceptions.ReadTimeout("timed out")

    world.engine.stop = lost
    lease, code = world.backend.stop_service(lease, world.journal, "kv", timeout_sec=1)
    assert (code, lease.services[0].state) == (137, "exited")

    # When the SIGKILL cannot be sent either, nothing is proven.
    world.engine.stop = lost
    world.engine.kill = connection_lost
    with pytest.raises(InfrastructureError, match="^recovery_required"):
        world.backend.stop_service(lease, world.journal, "main", timeout_sec=1)


def test_status_reports_state_health_and_exit_code():
    world = EnvWorld()
    plan, lease = world.ready()
    world.engine.exit(plan.services[1].container_name, 2)
    status = world.backend.status(lease)
    assert status["kv"].state == "running" and status["kv"].health == "healthy"
    assert (status["main"].state, status["main"].exit_code) == ("exited", 2)


def test_archive_target_exposes_every_tmpfs_root():
    world = EnvWorld()
    plan, lease = world.create()
    target = world.backend.archive_target(lease, "main")
    assert target.container_id == lease.services[1].container_id
    assert target.tmpfs == ("/dev/shm", "/scratch")
    assert target.inspect()["Id"] == target.container_id


def test_exec_target_is_the_attested_container_until_it_is_gone():
    world = EnvWorld()
    plan, lease = world.ready()
    target = world.backend.exec_target(lease, "main")
    assert target.container_id == lease.services[1].container_id
    assert target.inspect()["State"]["Running"] is True

    lease = world.backend.destroy(lease, world.journal)

    assert target.inspect() is None  # the pump tells a service death apart
    for service, message in (("main", "absent"), ("db", "unknown")):
        with pytest.raises(SandboxError, match=message):
            world.backend.exec_target(lease, service)


# -- paused termination -------------------------------------------------------


def test_paused_services_are_killed_by_docker_kill_when_not_root():
    world = EnvWorld()
    plan, lease = world.ready()
    lease = world.backend.pause(lease, world.journal)

    lease = world.backend.terminate(lease, world.journal)

    kills = [event for event in world.events if event[0] == "kill"]
    assert kills == [
        ("kill", plan.services[0].container_name, True),
        ("kill", plan.services[1].container_name, True),
    ]
    assert [record.state for record in lease.services] == ["exited", "exited"]
    assert isinstance(default_paused_killer(object(), euid=1001), DockerPausedKiller)
    assert isinstance(default_paused_killer(object(), euid=0), CgroupPausedKiller)


def fake_cgroup(tmp_path, container_id, *, name=None, populated=0):
    proc = tmp_path / "proc" / "4242"
    proc.mkdir(parents=True)
    scope = f"/system.slice/docker-{name or container_id}.scope"
    (proc / "cgroup").write_text(f"0::{scope}\n")
    root = tmp_path / "cgroup"
    directory = root / scope.lstrip("/")
    directory.mkdir(parents=True)
    (directory / "cgroup.kill").write_text("")
    (directory / "cgroup.events").write_text(f"populated {populated}\nfrozen 1\n")
    return root, directory


def test_root_path_writes_cgroup_kill_on_the_frozen_scope(tmp_path):
    container_id = "c" * 64
    root, directory = fake_cgroup(tmp_path, container_id)
    killer = CgroupPausedKiller(cgroup_root=root, proc_root=tmp_path / "proc")
    attrs = {"State": {"Paused": True, "Running": True, "Pid": 4242}}

    settled = killer.kill(container_id, attrs)

    assert (directory / "cgroup.kill").read_text() == "1"
    assert settled() is True


def test_root_path_refuses_a_scope_that_does_not_name_the_container(tmp_path):
    root, directory = fake_cgroup(tmp_path, "c" * 64, name="d" * 64)
    killer = CgroupPausedKiller(cgroup_root=root, proc_root=tmp_path / "proc")
    with pytest.raises(InfrastructureError, match="does not name"):
        killer.kill("c" * 64, {"State": {"Pid": 4242}})
    assert (directory / "cgroup.kill").read_text() == ""


def test_root_path_needs_proof_the_scope_emptied(tmp_path):
    clock = FakeClock()

    def sleep(seconds):
        clock.now += seconds

    root, directory = fake_cgroup(tmp_path, "c" * 64, populated=1)
    killer = CgroupPausedKiller(cgroup_root=root, proc_root=tmp_path / "proc")
    paused = {"State": {"Paused": True, "Running": True, "Pid": 4242}}
    states = iter([paused])

    def inspect():
        return next(states, {"State": {"Running": False}})

    def terminate():
        return terminate_container(
            None,
            "c" * 64,
            inspect=inspect,
            paused_killer=killer,
            clock=clock,
            sleep=sleep,
            timeout=1,
        )

    # Docker reporting it stopped is not enough: the scope must empty too.
    with pytest.raises(InfrastructureError, match="recovery_required"):
        terminate()
    (directory / "cgroup.events").write_text("populated 0\nfrozen 1\n")
    states = iter([paused])
    assert terminate()["State"]["Running"] is False


def test_env_teardown_uses_the_root_seam_for_paused_services(tmp_path):
    kills = []

    class Seam:
        def kill(self, container_id, attrs):
            kills.append(container_id)
            world.engine.exit(attrs["Name"][1:], 137)

    world = EnvWorld(paused_killer=Seam())
    plan, lease = world.ready()
    lease = world.backend.pause(lease, world.journal)
    lease = world.backend.destroy(lease, world.journal)
    assert kills == [record.container_id for record in lease.services]
    assert "kill" not in [event[0] for event in world.events]
    assert lease.state == "removed"


def test_terminate_container_kills_a_paused_container_by_id():
    world = EnvWorld()
    plan, lease = world.ready()
    container_id = lease.services[1].container_id
    world.engine.pause(container_id)

    final = terminate_container(
        world.network.client.api,
        container_id,
        inspect=lambda: world.engine.inspect_container(container_id),
        paused_killer=DockerPausedKiller(world.network.client.api),
    )
    assert final["State"]["Running"] is False
    assert final["State"]["ExitCode"] == 137


def test_unprovable_termination_is_recovery_required():
    world = EnvWorld()
    plan, lease = world.ready()
    world.engine.kill = lambda container, signal=None: None
    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.backend.terminate(lease, world.journal)


def test_terminate_signals_every_service_before_proving_against_one_deadline():
    world = EnvWorld()
    plan, lease = world.ready()
    kv, main = (service.container_name for service in plan.services)

    def stubborn(container, signal=None):
        name = world.engine._get(container)["Name"][1:]
        world.events.append(("kill", name))
        if name != kv:
            world.engine.exit(name, 137)

    world.engine.kill = stubborn
    start = world.clock.now
    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.backend.terminate(lease, world.journal, deadline=start + 1.0)
    # main was signalled although kv never stopped, and one deadline held.
    assert [event[1] for event in world.events if event[0] == "kill"] == [kv, main]
    assert start + 1.0 <= world.clock.now < start + 1.1


def test_fail_terminates_every_service_and_records_the_reason():
    world = EnvWorld()
    plan, lease = world.ready()

    failed = world.backend.fail(lease, world.journal, "disk_quota")

    assert (failed.state, failed.reason) == ("failed", "disk_quota")
    assert [record.state for record in failed.services] == ["exited", "exited"]
    assert not world.container(plan, "main")["State"]["Running"]
    # An env that already failed keeps its first reason.
    assert world.backend.fail(failed, world.journal, "quarantined").reason == (
        "disk_quota"
    )
    removed = world.backend.destroy(failed, world.journal)
    for ended in (world.journal.leases[0], removed):
        with pytest.raises(SandboxError) as caught:
            world.backend.fail(ended, world.journal, "disk_quota")
        assert caught.value.code == "busy"


def connection_lost(*args, **kwargs):
    raise requests.exceptions.ConnectionError("daemon went away")


@pytest.mark.parametrize("operation", ("terminate", "destroy", "fail"))
@pytest.mark.parametrize("call", ("kill", "inspect_container"))
def test_every_unprovable_teardown_step_is_recovery_required(operation, call):
    world = EnvWorld()
    plan, lease = world.ready()
    setattr(world.engine, call, connection_lost)
    run = getattr(world.backend, operation)
    args = ("disk_quota",) if operation == "fail" else ()
    with pytest.raises(InfrastructureError, match="^recovery_required"):
        run(lease, world.journal, *args)


def test_a_journal_failure_during_teardown_passes_through_unchanged():
    world = EnvWorld()
    plan, lease = world.ready()
    world.journal.fail_at = len(world.journal.leases) + 2
    with pytest.raises(OSError, match="journal disk full"):
        world.backend.destroy(lease, world.journal)


# -- teardown and inventory ---------------------------------------------------


def test_destroy_removes_containers_then_volumes_then_bridge_then_rule():
    world = EnvWorld()
    plan, lease = world.ready()
    world.events.clear()

    removed = world.backend.destroy(lease, world.journal)

    kinds = [event[0] for event in world.events if event[0] != "commit"]
    assert kinds.index("kill") < kinds.index("container-remove")
    last_container = max(
        i for i, kind in enumerate(kinds) if kind == "container-remove"
    )
    first_volume = kinds.index("volume-remove")
    assert last_container < first_volume
    assert max(
        i for i, kind in enumerate(kinds) if kind == "volume-remove"
    ) < kinds.index("network-remove")
    assert kinds.index("network-remove") < kinds.index("remove")
    assert (removed.state, removed.pending_mutation, removed.network_id) == (
        "removed",
        False,
        None,
    )
    assert all(record.state == "removed" for record in removed.services)
    assert [record.container_id for record in removed.services] == [
        record.container_id for record in lease.services
    ]
    assert world.labelled() == ([], [], [])
    assert world.network.firewall.installed == {}

    world.events.clear()
    assert world.backend.destroy(removed, world.journal) is removed
    assert world.events == []


def test_destroy_finds_an_unjournaled_container_by_planned_name():
    world = EnvWorld()
    plan = world.plan()
    lease = world.planned(plan)
    world.engine.create_errors[plan.services[1].container_name] = (
        requests.exceptions.ReadTimeout("t")
    )
    with pytest.raises(InfrastructureError):
        world.backend.create(plan, lease, world.journal)
    # The create finished after the client gave up.
    del world.engine.create_errors[plan.services[1].container_name]
    world.engine.create_container_from_config(
        plan.services[1].body(), name=plan.services[1].container_name
    )
    pending = world.journal.leases[-1]
    assert pending.services[1].container_id is None

    removed = world.backend.destroy(pending, world.journal)

    assert removed.state == "removed"
    assert world.engine.containers == {}
    assert world.labelled() == ([], [], [])


def unanswered(world, kind, index):
    """Make one create time out; returns a function that lands it later."""
    plan = world.plan()
    if kind == "container":
        name = plan.services[index].container_name
        world.engine.create_errors[name] = requests.exceptions.ReadTimeout("t")

        def land():
            del world.engine.create_errors[name]
            world.engine.create_container_from_config(
                plan.services[index].body(), name=name
            )

    else:
        name = plan.volumes[index].name
        world.engine.create_errors[name] = requests.exceptions.ReadTimeout("t")

        def land():
            del world.engine.create_errors[name]
            world.engine.create_volume(
                name, driver="local", labels=dict(plan.volumes[index].labels)
            )

    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.backend.create(plan, world.planned(plan), world.journal)
    return land


@pytest.mark.parametrize(("kind", "index"), (("container", 1), ("volume", 0)))
def test_destroy_fails_closed_until_an_unknown_create_lands(kind, index):
    world = EnvWorld()
    land = unanswered(world, kind, index)
    pending = world.journal.leases[-1]
    assert pending.pending_mutation

    with pytest.raises(InfrastructureError, match="pending create outcome"):
        world.backend.destroy(pending, world.journal)

    # Nothing is reported removed and every later object is retained.
    stopped = world.journal.leases[-1]
    assert stopped.pending_mutation and stopped.state == "stopping"
    assert world.network.firewall.installed and world.network.store

    land()
    removed = world.backend.destroy(stopped, world.journal)
    assert (removed.state, removed.pending_mutation) == ("removed", False)
    assert world.labelled() == ([], [], [])
    assert world.network.firewall.installed == {}


def test_a_retried_pending_destroy_does_not_wait_for_a_removed_bridge():
    world = EnvWorld()
    world.network.unanswered = requests.exceptions.ReadTimeout("t")
    plan = world.plan()
    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.backend.create(plan, world.planned(plan), world.journal)
    world.network.settle()

    def appear(network):
        # Something labelled for the env shows up after the bridge is gone.
        world.network.after_remove = lambda network: None
        world.engine.create_container_from_config(
            plan.services[0].body(), name="late-kv"
        )

    world.network.after_remove = appear
    with pytest.raises(InfrastructureError, match="gained objects"):
        world.backend.destroy(world.journal.leases[-1], world.journal)
    retried = world.journal.leases[-1]
    # The bridge create that was in flight was found: no longer claimed.
    assert (retried.network_id, retried.pending_mutation) == (None, False)
    assert world.backend.destroy(retried, world.journal).state == "removed"


def test_a_landed_last_volume_leaves_the_first_container_ambiguous():
    world = EnvWorld()
    land = unanswered(world, "volume", 1)
    land()
    # The journal cannot tell whether the last volume's create returned and
    # the first container's began: fail closed (spec 5).
    with pytest.raises(InfrastructureError, match="service kv pending create"):
        world.backend.destroy(world.journal.leases[-1], world.journal)
    assert world.network.firewall.installed


def test_destroy_keeps_the_rule_while_a_bridge_create_is_unknown():
    world = EnvWorld()
    world.network.unanswered = requests.exceptions.ReadTimeout("t")
    plan = world.plan()
    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.backend.create(plan, world.planned(plan), world.journal)

    with pytest.raises(InfrastructureError, match="rule is retained"):
        world.backend.destroy(world.journal.leases[-1], world.journal)
    assert world.network.firewall.installed

    world.network.settle()  # dockerd finishes the bridge
    removed = world.backend.destroy(world.journal.leases[-1], world.journal)
    assert removed.state == "removed"
    assert world.network.store == {} and world.network.firewall.installed == {}


def test_destroy_never_removes_a_foreign_container_holding_a_planned_name():
    world = EnvWorld()
    plan, lease = world.create()
    attrs = world.container(plan, "main")
    attrs["Config"]["Labels"] = {"owner": "someone-else"}
    with pytest.raises(InfrastructureError, match="recovery_required"):
        world.backend.destroy(lease, world.journal)
    assert attrs["Id"] in world.engine.containers


def test_inventory_removes_exact_extras_of_every_kind(caplog):
    world = EnvWorld()
    plan, lease = world.create()
    extra = copy.deepcopy(world.container(plan, "kv"))
    extra["Id"], extra["Name"] = "9" * 64, "/stray-kv"
    world.engine.containers[extra["Id"]] = extra
    world.engine.volumes["stray-volume"] = {
        **world.engine.volumes[plan.volumes[0].name],
        "Name": "stray-volume",
    }
    stray_net = world.network.add_bridge(plan.network, name="stray-net")

    with caplog.at_level(logging.WARNING):
        removed = world.backend.inventory(lease, require_present=True)

    assert removed == ("stray-kv", "stray-volume", "stray-net")
    assert "stray-kv" in caplog.text
    assert extra["Id"] not in world.engine.containers
    assert "stray-volume" not in world.engine.volumes
    assert stray_net not in world.network.store
    assert lease.network_id in world.network.store


def partial_container(world, plan):
    partial = copy.deepcopy(world.container(plan, "kv"))
    partial["Id"], partial["Name"] = "8" * 64, "/half-owned"
    partial["Config"]["Labels"] = {"rsi-harness.sandbox-env": ENV_ID}
    world.engine.containers[partial["Id"]] = partial
    return lambda: partial["Id"] in world.engine.containers


def partial_volume(world, plan):
    world.engine.volumes["half-owned"] = {
        **world.engine.volumes[plan.volumes[0].name],
        "Name": "half-owned",
        "Labels": {"rsi-harness.sandbox-env": ENV_ID},
    }
    return lambda: "half-owned" in world.engine.volumes


def partial_network(world, plan):
    network_id = world.network.add_bridge(
        plan.network, name="half-owned", labels={"rsi-harness.sandbox-env": ENV_ID}
    )
    return lambda: network_id in world.network.store


@pytest.mark.parametrize("add", (partial_container, partial_volume, partial_network))
def test_inventory_fails_closed_on_a_partial_label_match(add):
    world = EnvWorld()
    plan, lease = world.create()
    kept = add(world, plan)
    with pytest.raises(
        InfrastructureError, match="recovery_required.*part of its label"
    ):
        world.backend.inventory(lease, require_present=False)
    assert kept()


def test_destroy_never_removes_a_foreign_volume_holding_a_journaled_name():
    world = EnvWorld()
    plan, lease = world.create()
    name = plan.volumes[0].name
    world.engine.volumes[name]["Labels"] = {"owner": "someone-else"}
    with pytest.raises(InfrastructureError, match="recovery_required.*not owned"):
        world.backend.destroy(lease, world.journal)
    assert name in world.engine.volumes
    assert world.network.firewall.installed


def test_a_foreign_endpoint_on_the_env_bridge_rolls_create_back():
    world = EnvWorld()

    def attach(attrs):
        if attrs["Name"].endswith("-1"):
            network = next(iter(world.network.store.values()))
            network.attrs["Containers"]["f" * 64] = {"Name": "intruder"}

    world.engine.on_create = attach
    with pytest.raises(SetupError, match="foreign endpoint"):
        world.create()
    assert world.labelled() == ([], [], [])


def test_objects_gained_during_teardown_fail_closed():
    world = EnvWorld()
    plan, lease = world.ready()
    late = copy.deepcopy(world.container(plan, "kv"))
    late["Id"], late["Name"] = "7" * 64, "/late-kv"
    late["State"].update(Status="exited", Running=False)

    def appear(network):
        world.engine.containers[late["Id"]] = late

    world.network.after_remove = appear
    with pytest.raises(InfrastructureError, match="gained objects during teardown"):
        world.backend.destroy(lease, world.journal)
    # Exact labels: removed, but the env is not reported removed.
    assert late["Id"] not in world.engine.containers
    assert world.journal.leases[-1].state == "stopping"


def test_rollback_cannot_be_proven_is_recovery_required():
    world = EnvWorld()

    def drift(attrs):
        attrs["HostConfig"]["Privileged"] = True

    world.engine.on_create = drift
    world.engine.remove_volume = lambda name, force=False: None
    with pytest.raises(
        InfrastructureError, match="recovery_required: partial sandbox env"
    ):
        world.create()
