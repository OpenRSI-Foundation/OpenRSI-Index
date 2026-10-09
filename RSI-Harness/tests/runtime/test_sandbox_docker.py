"""Child Docker configuration is attested, not inferred from requested kwargs."""

import socket
from types import SimpleNamespace

import pytest
from docker.errors import NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from tests.runtime.test_sandbox_budget import make_child
from tests.sandbox_helpers import make_profile


class DockerChild:
    def __init__(self, client, kwargs):
        self.client = client
        self.id = "b" * 64
        self.removed = False
        self.attrs = {
            "Id": self.id,
            "Name": "/" + kwargs["name"],
            "Image": kwargs["image"],
            "State": {"Running": False, "Paused": False, "OOMKilled": False},
            "AppArmorProfile": "",
            "Config": {
                "Labels": kwargs["labels"],
                "User": kwargs["user"],
                "Entrypoint": kwargs["entrypoint"],
                "Cmd": kwargs["command"],
                "Env": [
                    f"{key}={value}" for key, value in kwargs["environment"].items()
                ],
                "Healthcheck": {"Test": ["NONE"]},
                "WorkingDir": kwargs["working_dir"],
            },
            "HostConfig": {
                "ReadonlyRootfs": kwargs["read_only"],
                "NetworkMode": kwargs["network_mode"],
                "CapDrop": kwargs["cap_drop"],
                "CapAdd": [],
                "Privileged": False,
                "SecurityOpt": kwargs["security_opt"],
                "Devices": [],
                "DeviceRequests": [],
                "Binds": [],
                "PortBindings": {},
                "PidMode": "",
                "IpcMode": "private",
                "UTSMode": "",
                "CgroupnsMode": "private",
                "Runtime": "runc",
                "NanoCpus": kwargs["nano_cpus"],
                "Memory": kwargs["mem_limit"],
                "MemorySwap": kwargs["memswap_limit"],
                "PidsLimit": kwargs["pids_limit"],
                "Ulimits": kwargs["ulimits"],
                "Tmpfs": kwargs["tmpfs"],
                "ShmSize": kwargs["shm_size"],
                "LogConfig": {"Type": "none", "Config": {}},
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
            },
            "Mounts": [],
            "NetworkSettings": {"Networks": {"none": {}}},
        }

    def reload(self):
        if self.removed:
            raise NotFound("gone")

    def start(self):
        self.client.events.append("start")
        self.attrs["State"]["Running"] = True
        self.attrs["AppArmorProfile"] = "docker-default"

    def pause(self):
        self.client.events.append("pause")
        self.attrs["State"]["Paused"] = True

    def unpause(self):
        self.client.events.append("unpause")
        self.attrs["State"]["Paused"] = False

    def kill(self, signal):
        assert signal == "SIGKILL"
        self.client.events.append("kill")
        self.attrs["State"].update(Running=False, Paused=False)

    def remove(self, force=False, v=False):
        assert not force, "removal follows proven termination"
        assert not self.attrs["State"]["Running"]
        self.client.events.append("remove")
        self.removed = True


class DockerClient:
    def __init__(self):
        self.events = []
        self.exec_exit_code = 0
        self.api = SimpleNamespace(
            exec_create=lambda *a, **k: {"Id": "exec-1"},
            exec_start=self.exec_start,
            exec_inspect=lambda _: {"Running": False, "ExitCode": self.exec_exit_code},
        )
        self.last_create_kwargs = None
        self.children = {}
        self.image = SimpleNamespace(
            id=make_profile().image,
            attrs={"Config": {"Volumes": None, "Env": ["UNWANTED=value"]}},
        )
        self.images = SimpleNamespace(get=lambda _: self.image)
        self.containers = SimpleNamespace(
            create=self.create, get=self.get, list=self.list
        )
        self.host = {
            "OSType": "linux",
            "CgroupVersion": "2",
            "MemoryLimit": True,
            "SwapLimit": True,
            "PidsLimit": True,
            "CpuCfsQuota": True,
            "SecurityOptions": [
                "name=apparmor",
                "name=seccomp,profile=builtin",
                "name=cgroupns",
            ],
            "Runtimes": {"runc": {}},
        }

    def info(self):
        return self.host

    def exec_start(self, *args, **kwargs):
        reader, writer = socket.socketpair()
        writer.close()
        return reader

    def create(self, **kwargs):
        self.last_create_kwargs = kwargs
        child = DockerChild(self, kwargs)
        self.children[child.id] = child
        self.events.append("create")
        return child

    def get(self, identity):
        for child in self.children.values():
            if not child.removed and identity in (
                child.id,
                child.attrs["Name"].lstrip("/"),
            ):
                return child
        raise NotFound("gone")

    def list(self, **kwargs):
        return [child for child in self.children.values() if not child.removed]


@pytest.fixture
def backend():
    from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend

    client = DockerClient()
    return SandboxDockerBackend(client), client


def test_child_create_has_no_parent_authority_and_does_not_start(backend):
    backend, client = backend
    lease = make_child()
    identity = backend.create(lease, make_profile())
    call = client.last_create_kwargs
    assert identity == "b" * 64
    assert client.events == ["create"]
    assert call["network_mode"] == "none"
    assert call["read_only"] is True
    assert call["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in call["security_opt"]
    assert call["user"] == "0:0"
    assert not call.get("mounts")
    assert not call.get("device_requests")
    assert not call.get("ports")
    assert call["environment"]["UNWANTED"] == ""
    assert call["nano_cpus"] == 1_000_000_000
    assert call["mem_limit"] == call["memswap_limit"] == 256 * 1024**2
    assert call["pids_limit"] == 32
    assert call["shm_size"] == 8 * 1024**2


def test_start_requires_recorded_identity_and_attests_configuration(backend):
    backend, client = backend
    planned = make_child()
    identity = backend.create(planned, make_profile())
    with pytest.raises(InfrastructureError, match="identity"):
        backend.start(planned)
    lease = planned.model_copy(update={"container_id": identity})
    backend.start(lease)
    assert backend.inspect(lease)["State"]["Running"]
    backend.pause(lease)
    assert backend.inspect(lease)["State"]["Paused"]
    backend.resume(lease)
    assert not backend.inspect(lease)["State"]["Paused"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("ReadonlyRootfs", False),
        ("NetworkMode", "host"),
        ("CapAdd", ["SYS_ADMIN"]),
        ("Privileged", True),
        ("Memory", 0),
        ("MemorySwap", -1),
        ("PidsLimit", -1),
        ("NanoCpus", 0),
        ("Binds", ["/:/host"]),
        ("DeviceRequests", [{"Count": -1}]),
        ("PidMode", "host"),
        ("IpcMode", "host"),
        ("UTSMode", "host"),
        ("CgroupnsMode", "host"),
        ("Runtime", "nvidia"),
        ("SecurityOpt", ["seccomp=unconfined"]),
        ("Tmpfs", {}),
        ("LogConfig", {"Type": "json-file"}),
    ],
)
def test_inspect_drift_blocks_start(backend, field, value):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    client.get(identity).attrs["HostConfig"][field] = value
    with pytest.raises(InfrastructureError, match="sandbox"):
        backend.start(make_child(container_id=identity))
    assert "start" not in client.events


def test_automatic_image_volume_and_writable_host_mount_are_denied(backend):
    backend, client = backend
    client.image.attrs["Config"]["Volumes"] = {"/unbounded": {}}
    with pytest.raises(SetupError, match="volume"):
        backend.preflight(make_profile())
    client.image.attrs["Config"]["Volumes"] = None
    identity = backend.create(make_child(), make_profile())
    client.get(identity).attrs["Mounts"] = [
        {"Type": "bind", "Destination": "/bad", "RW": True}
    ]
    with pytest.raises(InfrastructureError, match="mount"):
        backend.start(make_child(container_id=identity))


@pytest.mark.parametrize(
    "field,value",
    [
        ("MemoryLimit", False),
        ("SwapLimit", False),
        ("PidsLimit", False),
        ("CpuCfsQuota", False),
        ("OSType", "windows"),
        ("CgroupVersion", "1"),
        ("SecurityOptions", []),
        ("Runtimes", {}),
    ],
)
def test_host_must_enforce_every_limit_before_create(backend, field, value):
    backend, client = backend
    client.host[field] = value
    with pytest.raises(SetupError, match="sandbox"):
        backend.create(make_child(), make_profile())
    assert client.events == []


def test_wrong_image_id_is_not_approved_by_tag_resolution(backend):
    backend, client = backend
    client.image.id = "sha256:" + "f" * 64
    with pytest.raises(SetupError, match="image"):
        backend.create(make_child(), make_profile())
    assert client.events == []


def test_create_cannot_exceed_its_durable_resource_reservation(backend):
    backend, client = backend
    profile = make_profile().model_copy(update={"cpus": 2})
    with pytest.raises(InfrastructureError, match="reservation"):
        backend.create(make_child(), profile)
    assert client.events == []


def test_termination_and_removal_of_running_child_are_idempotent(backend):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    backend.terminate(lease)
    backend.remove(lease)
    backend.remove(lease)
    assert client.events == ["create", "start", "kill", "remove"]


def test_paused_termination_fails_closed_without_implicit_docker_thaw(backend):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    backend.pause(lease)
    for operation in (backend.terminate, backend.remove):
        with pytest.raises(InfrastructureError, match="paused"):
            operation(lease)
    assert client.events == ["create", "start", "pause"]
    assert backend.inspect(lease)["State"]["Paused"]


def test_opt_in_paused_killer_terminates_a_paused_child():
    from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend

    client = DockerClient()
    kills = []

    class Killer:
        """Stands in for cgroup.kill: kills the frozen tasks, no thaw."""

        def kill(self, container_id, attrs):
            kills.append((container_id, attrs["State"]["Paused"]))
            client.get(container_id).kill(signal="SIGKILL")

    backend = SandboxDockerBackend(client, paused_killer=Killer())
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    backend.pause(lease)

    backend.terminate(lease)
    backend.remove(lease)

    assert kills == [(identity, True)]
    assert client.events == ["create", "start", "pause", "kill", "remove"]


def test_foreign_labels_prevent_any_lifecycle_mutation(backend):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    child = client.get(identity)
    child.attrs["Config"]["Labels"]["rsi-harness.run-id"] = "other"
    for method in (
        backend.start,
        backend.pause,
        backend.resume,
        backend.terminate,
        backend.remove,
    ):
        with pytest.raises(InfrastructureError):
            method(make_child(container_id=identity))
    assert client.events == ["create"]


def test_failed_kill_never_releases_child(backend, monkeypatch):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    monkeypatch.setattr(client.get(identity), "kill", lambda **kwargs: None)
    with pytest.raises(InfrastructureError, match="termination"):
        backend.terminate(lease)
    assert "remove" not in client.events


def test_missing_shell_is_actionable_before_candidate_execution(backend):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    client.exec_exit_code = 127
    with pytest.raises(InfrastructureError, match="shell"):
        backend.start(make_child(container_id=identity))


@pytest.mark.parametrize("after_create", [False, True])
def test_create_transport_failure_keeps_planned_identity_reconcilable(
    backend, monkeypatch, after_create
):
    backend, client = backend
    original_create = client.containers.create

    def create(**kwargs):
        if after_create:
            original_create(**kwargs)
        raise OSError("daemon connection lost")

    monkeypatch.setattr(client.containers, "create", create)
    with pytest.raises(OSError):
        backend.create(make_child(), make_profile())
    assert "start" not in client.events
    if after_create:
        child = client.get(make_child().planned_name)
        assert child.attrs["Config"]["Labels"]["rsi-harness.sandbox-id"] == "a" * 32


def test_failed_remove_is_not_reported_as_absence(backend, monkeypatch):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    monkeypatch.setattr(client.get(identity), "remove", lambda **kwargs: None)
    with pytest.raises(InfrastructureError, match="removal"):
        backend.remove(make_child(container_id=identity))


def test_proven_absence_retires_live_backend_metadata(backend):
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    assert identity in backend._profiles
    assert identity in backend._environments
    client.get(identity).removed = True
    backend.remove(make_child(container_id=identity))
    assert identity not in backend._profiles
    assert identity not in backend._environments
    assert "remove" not in client.events


def test_exec_startup_delay_is_included_in_deadline(backend, monkeypatch):
    import time

    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    original = client.api.exec_create

    def delayed(*args, **kwargs):
        time.sleep(0.06)
        return original(*args, **kwargs)

    monkeypatch.setattr(client.api, "exec_create", delayed)
    monkeypatch.setattr(
        client.api,
        "exec_start",
        lambda *a, **k: pytest.fail("exec must not start after deadline"),
    )
    result = backend.execute(
        lease, ["/bin/true"], "/workspace", {}, time.monotonic() + 0.02, 1024
    )
    assert result.timed_out
    assert not backend.inspect(lease)["State"]["Running"]


def test_unknown_exec_outcome_is_not_replayed(backend, monkeypatch):
    import time

    from rsi_harness.runtime.sandbox_contracts import SandboxError

    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    calls = []

    def lost_response(*args, **kwargs):
        calls.append(1)
        raise OSError("response lost")

    monkeypatch.setattr(client.api, "exec_start", lost_response)
    with pytest.raises(SandboxError, match="unknown-outcome"):
        backend.execute(
            lease, ["/bin/true"], "/workspace", {}, time.monotonic() + 1, 1024
        )
    assert calls == [1]


def test_exec_close_failure_signals_completion_without_waiting_for_deadline(
    backend, monkeypatch
):
    import time

    from rsi_harness.runtime.sandbox_contracts import SandboxError

    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    original = client.api.exec_start

    def broken_close(*args, **kwargs):
        connection = original(*args, **kwargs)

        def close():
            connection.close()
            raise OSError("fixture close failed")

        return SimpleNamespace(_sock=connection, close=close)

    monkeypatch.setattr(client.api, "exec_start", broken_close)
    started = time.monotonic()
    with pytest.raises(SandboxError, match="unknown-outcome"):
        backend.execute(lease, ["true"], "/workspace", {}, started + 1, 1024)
    assert time.monotonic() - started < 0.8


@pytest.mark.parametrize("flag", ["timed_out", "output_limited", "oom_killed"])
def test_terminal_transfer_error_carries_proven_stopped_state(flag):
    from rsi_harness.runtime.sandbox_contracts import (
        SandboxChildStopped,
        SandboxResult,
    )
    from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend

    result = SandboxResult(exit_code=None, duration_sec=1, **{flag: True})
    with pytest.raises(SandboxChildStopped):
        SandboxDockerBackend._require_transfer_success(result, b"")


def test_transfer_stderr_has_log_limit_not_bundle_limit(backend, monkeypatch):
    import struct
    import threading
    import time

    from rsi_harness.integrations.sandbox_client import MAX_BODY_BYTES, MAX_OUTPUT_BYTES

    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)

    def flood(*args, **kwargs):
        reader, writer = socket.socketpair()

        def send():
            try:
                writer.sendall(struct.pack("!BxxxI", 2, MAX_OUTPUT_BYTES + 1))
                writer.sendall(b"x" * (MAX_OUTPUT_BYTES + 1))
            except OSError:
                pass
            finally:
                writer.close()

        threading.Thread(target=send, daemon=True).start()
        return reader

    monkeypatch.setattr(client.api, "exec_start", flood)
    result, _, stderr = backend._run_exec(
        lease, ["true"], "/workspace", {}, time.monotonic() + 2, MAX_BODY_BYTES
    )
    assert result.output_limited
    assert len(stderr) == MAX_OUTPUT_BYTES
    assert not backend.inspect(lease)["State"]["Running"]


@pytest.mark.parametrize(
    "outcome", ["rejected", "success", "truncated", "invalid-stream"]
)
def test_early_stdin_close_requires_complete_frames_and_failed_exit(
    backend, monkeypatch, outcome
):
    import struct
    import time

    from rsi_harness.runtime.sandbox_contracts import SandboxError

    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    client.exec_exit_code = 0 if outcome == "success" else 2

    def reject(*args, **kwargs):
        reader, writer = socket.socketpair()
        reader.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        stream = 3 if outcome == "invalid-stream" else 2
        length = 100 if outcome == "truncated" else 8
        writer.sendall(struct.pack("!BxxxI", stream, length) + b"rejected")
        writer.close()
        return reader

    monkeypatch.setattr(client.api, "exec_start", reject)
    started = time.monotonic()
    if outcome == "rejected":
        result, stdout, stderr = backend._run_exec(
            lease, ["helper"], "/workspace", {}, started + 2, 8192, b"x" * 1024**2
        )
        assert result.exit_code == 2
        assert stdout == b"" and stderr == b"rejected"
        assert not result.timed_out
    else:
        with pytest.raises(SandboxError) as caught:
            backend._run_exec(
                lease, ["helper"], "/workspace", {}, started + 2, 8192, b"x" * 1024**2
            )
        assert caught.value.code == "unknown-outcome"
    assert time.monotonic() - started < 1
    assert backend.inspect(lease)["State"]["Running"]


def test_exec_transport_accepts_socket_above_select_fd_limit(backend, monkeypatch):
    import fcntl
    import resource
    import time

    if resource.getrlimit(resource.RLIMIT_NOFILE)[0] <= 1024:
        pytest.skip("host descriptor limit does not allow this boundary")
    backend, client = backend
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)

    def high_socket(*args, **kwargs):
        reader, writer = socket.socketpair()
        try:
            descriptor = fcntl.fcntl(reader.fileno(), fcntl.F_DUPFD_CLOEXEC, 1024)
        finally:
            reader.close()
            writer.close()
        return socket.socket(fileno=descriptor)

    monkeypatch.setattr(client.api, "exec_start", high_socket)
    result = backend.execute(
        lease, ["true"], "/workspace", {}, time.monotonic() + 2, 1024
    )
    assert result.exit_code == 0
