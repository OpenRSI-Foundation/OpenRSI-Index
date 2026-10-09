"""Enabled production composition carries durable grants but ephemeral tokens."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.models import AgentRunResult, RunRequest
from rsi_harness.runtime import production
from rsi_harness.runtime.recovery import LeaseStore
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan
from tests.runtime.test_sandbox_budget import authority
from tests.sandbox_helpers import (
    FakeSandboxBackend,
    make_sandbox_policy,
    make_sandbox_task,
)


@pytest.fixture
def composed(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="rsi-prod-") as root:
        root = Path(root)
        plan = make_run_plan(root / "task")
        definition = plan.task.model_copy(
            update={
                "sandbox": make_sandbox_task(),
                "service": plan.task.service.model_copy(
                    update={"cpus": 1, "memory_mb": 256}
                ),
            }
        )
        child_backend = FakeSandboxBackend()
        child_backend.preflight = lambda profile: profile.image
        transports = []

        def from_env(**kwargs):
            transports.append(kwargs)
            return SimpleNamespace(
                close=lambda: None,
                info=lambda: {"ID": "daemon", "DockerRootDir": "/var/lib/docker"},
                api=object(),
                timeout=kwargs.get("timeout"),
            )

        monkeypatch.setattr(production.docker, "from_env", from_env)

        def backend(client, **kwargs):
            child_backend.options = kwargs
            return child_backend

        monkeypatch.setattr(production, "SandboxDockerBackend", backend)
        ports = production._ProductionRunComposition(
            client=SimpleNamespace(info=lambda: {"ID": "daemon"}),
            data_root=root / "data",
            logs_root=root / "logs",
            inventory=object(),
            rsi_loop_config=RSILoopConfig(),
            snapshot=object(),
            firewall=object(),
            bind_host="127.0.0.1",
            bridge_gateway="127.0.0.1",
            omit_gpu_device_requests_for_tests=True,
            agent_adapter_factory=lambda *a: object(),
            quiescence_checker=None,
            api_endpoints=(),
            agent_secret_env={},
            verifier_secret_env={},
            sandbox_policy=make_sandbox_policy(),
        )
        ports.definition = definition
        store = LeaseStore(root / "data" / "leases")
        mutate = authority(store)
        try:
            yield ports, plan, definition, store, mutate, transports
        finally:
            ports.sandbox_lifecycle.close()
            ports.sandbox_lifecycle.release_resources()


def test_resolved_envelope_is_reserved_before_parent_setup(composed):
    ports, plan, definition, store, mutate, transports = composed
    events = []
    ports.event_callback = lambda name, value: events.append((name, value))
    ports.bind_sandbox_lease(mutate)
    assert store.read("run-1").sandbox_reservation.memory_mb == 3592
    assert transports == [{"timeout": 5}]
    # Paused Work children are killable at close() (M0's gap, closed in M5).
    assert ports._sandbox_broker.backend.options["paused_killer"] is not None
    assert events == [("sandbox_reserved", {"cpus": 6, "memory_mb": 3592})]
    final = ports.prepare_plan(
        definition,
        plan.images,
        plan.gpu_plan,
        RunRequest(task_dir=definition.source_dir),
        "run-1",
    )
    assert final.sandbox is not None
    endpoint = ports.sandbox_lifecycle.prepare_work()
    token = endpoint.environment["RSI_SANDBOX_TOKEN"]
    assert token not in final.model_dump_json()
    assert token not in store.path_for("run-1").read_text()


def test_different_child_daemon_is_rejected_before_reservation(composed):
    from rsi_harness.errors import SetupError

    ports, _, _, store, mutate, _ = composed
    ports.client = SimpleNamespace(info=lambda: {"ID": "different-daemon"})
    with pytest.raises(SetupError, match="identities differ"):
        ports.bind_sandbox_lease(mutate)
    assert store.read("run-1").sandbox_reservation is None


def test_console_reports_complete_sandbox_envelope(capsys):
    from rsi_harness.cli import _RunConsole

    _RunConsole()("sandbox_reserved", {"cpus": 6, "memory_mb": 2560})
    output = capsys.readouterr().out
    assert "6 CPUs" in output
    assert "2560 MiB" in output
    assert "parents + children + broker" in output


def test_work_secret_is_exec_only_and_deadline_callback_is_wired(composed):
    ports, plan, definition, _, mutate, _ = composed
    ports.bind_sandbox_lease(mutate)
    final = ports.prepare_plan(
        definition,
        plan.images,
        plan.gpu_plan,
        RunRequest(task_dir=definition.source_dir),
        "run-1",
    )

    @dataclass
    class Prepared:
        environment: tuple = (("PATH", "/custom/bin:/usr/bin"),)
        agent_name: str = "codex"

    captured = []

    def run(request):
        captured.append(request)
        request.on_exec_start(ports.sandbox_lifecycle.broker.clock() + 30)
        return AgentRunResult(exit_code=0)

    ports.agent = SimpleNamespace(prepare=lambda request: Prepared(), run=run)
    ports.artifacts = SimpleNamespace(root=final.paths.root)
    prepared = ports.prepare_agent(final)
    env = dict(prepared.environment)
    assert env["PATH"] == "/custom/bin:/usr/bin"
    assert env["RSI_SANDBOX_TOKEN"]
    ports.run_agent(prepared, SimpleNamespace(), 30)
    assert captured[0].on_exec_start is not None
    assert env["RSI_SANDBOX_TOKEN"] in captured[0].output_redact_values
    assert ports.sandbox_lifecycle.can_resume


ENV_TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
[metadata.rsi_harness.sandbox.environments.judge]
network = ["public", "none"]
pull = true
"""


@pytest.fixture
def composed_env(composed, monkeypatch):
    from rsi_harness.runtime.sandbox_envs import EnvRuntime
    from rsi_harness.runtime.sandbox_exec import ExecPump
    from tests.runtime.test_sandbox_envs import (
        FakeApi,
        FakeEnvBackend,
        FakePuller,
        RecordingKiller,
    )
    from tests.sandbox_helpers import make_env_policy, make_env_task

    ports, plan, definition, store, mutate, transports = composed
    definition = definition.model_copy(update={"sandbox": make_env_task(ENV_TASK)})
    ports.definition = definition
    ports.sandbox_policy = make_env_policy(ports.data_root.parent)
    ports.firewall = SimpleNamespace(name="firewall")
    runtimes = []

    def runtime(client, firewall, **options):
        runtimes.append((client, firewall, options))
        root = options["spool_root"]
        return EnvRuntime(
            backend=FakeEnvBackend(),
            images=FakePuller(),
            transfer=lambda stages: None,
            pump=lambda on_finish: ExecPump(
                FakeApi(),
                root,
                killer=RecordingKiller(),
                on_finish=on_finish,
                start_threads=False,
            ),
            spool_root=root,
        )

    monkeypatch.setattr(production, "docker_env_runtime", runtime)
    return ports, plan, definition, store, mutate, runtimes


def test_environment_grant_wires_the_v2_broker_after_the_run_root(
    composed, composed_env
):
    from rsi_harness.runtime.sandbox_contracts import SandboxEnvGrant
    from rsi_harness.runtime.sandbox_env_contracts import sandbox_spool_root

    ports, plan, definition, store, mutate, runtimes = composed_env
    events = []
    ports.event_callback = lambda name, value: events.append((name, value))
    ports.bind_sandbox_lease(mutate)
    # The 5 s control transport, then the env runtime's own client.
    assert composed[5] == [
        {"timeout": 5},
        {"timeout": production.SANDBOX_ENV_DOCKER_TIMEOUT_SECONDS},
    ]
    grant = ports._sandbox_grant
    assert isinstance(grant, SandboxEnvGrant)
    lease = store.read("run-1")
    # The disk dimension is reserved with the CPU/memory envelope, once.
    assert lease.sandbox_reservation.disk_mb == grant.reserved_disk_mb > 0
    assert lease.sandbox_env_authority
    assert events == [
        (
            "sandbox_reserved",
            {
                "cpus": grant.reserved_cpus,
                "memory_mb": grant.reserved_memory_mb,
                "disk_mb": grant.reserved_disk_mb,
            },
        )
    ]
    # Nothing touches the run root before prepare_plan creates it.
    assert runtimes == [] and ports._sandbox_broker is None
    assert not (ports.data_root / "run-1").exists()
    final = ports.prepare_plan(
        definition,
        plan.images,
        plan.gpu_plan,
        RunRequest(task_dir=definition.source_dir),
        "run-1",
    )
    assert final.sandbox == grant
    [(client, firewall, options)] = runtimes
    assert client.timeout == production.SANDBOX_ENV_DOCKER_TIMEOUT_SECONDS == 60
    assert firewall is ports.firewall
    assert options["run_id"] == "run-1"
    assert options["docker_root"] == "/var/lib/docker"
    assert options["host"] == grant.environments.host
    assert options["engine_destinations"] == ("127.0.0.1",)
    assert options["paused_killer"] is not None
    assert options["data_root"] == ports.data_root
    # Recovery (M6) removes exactly the spool production writes.
    spool = sandbox_spool_root(ports.data_root, "run-1")
    assert options["spool_root"] == spool
    assert (spool / "x").is_dir()
    assert (ports.data_root / "run-1" / "sb").stat().st_mode & 0o777 == 0o700
    endpoint = ports.sandbox_lifecycle.prepare_work()
    token = endpoint.environment["RSI_SANDBOX_TOKEN"]
    assert endpoint.environment["RSI_SANDBOX_PYTHONPATH"] == (
        "/run/rsi-harness/sandbox/py"
    )
    answer = ports._sandbox_broker.capabilities(token)
    assert answer["versions"] == [1, 2] and answer["environments"] is not None
    assert token not in store.path_for("run-1").read_text()
    # A clean close leaves no <run>/sb (A5): endpoints, spool, then sb.
    ports.sandbox_lifecycle.close()
    assert not (ports.data_root / "run-1" / "sb").exists()


def _with_builds(
    ports, definition, monkeypatch, inspect_image, socket_path="/run/docker.sock"
):
    from tests.sandbox_helpers import make_env_task

    ports.definition = definition.model_copy(
        update={
            "sandbox": make_env_task(
                ENV_TASK.replace("pull = true", "pull = true\nbuild = true", 1)
            )
        }
    )

    def from_env(**kwargs):
        return SimpleNamespace(
            close=lambda: None,
            info=lambda: {"ID": "daemon", "DockerRootDir": "/var/lib/docker"},
            api=SimpleNamespace(
                inspect_image=inspect_image,
                _custom_adapter=SimpleNamespace(socket_path=socket_path),
            ),
            timeout=kwargs.get("timeout"),
        )

    monkeypatch.setattr(production.docker, "from_env", from_env)


def test_environment_builds_pin_the_cached_builder_image(
    composed, composed_env, monkeypatch
):
    from tests.sandbox_helpers import BUILDER_IMAGE, BUILDER_IMAGE_ID, builder_inspect

    ports, plan, definition, store, mutate, runtimes = composed_env
    inspected = []

    def inspect_image(reference):
        inspected.append(reference)
        return builder_inspect()

    _with_builds(ports, definition, monkeypatch, inspect_image)
    ports.bind_sandbox_lease(mutate)
    grant = ports._sandbox_grant
    # Only the requested phase's approved image, resolved to its exact ID.
    assert inspected == [BUILDER_IMAGE]
    assert grant.environments.work.build.builder_image == BUILDER_IMAGE_ID
    assert grant.environments.judge.build is None
    assert store.read("run-1").sandbox_reservation.disk_mb == grant.reserved_disk_mb
    ports.prepare_plan(
        definition,
        plan.images,
        plan.gpu_plan,
        RunRequest(task_dir=definition.source_dir),
        "run-1",
    )
    [(_, _, options)] = runtimes
    # Loop-ext4 builder files live under <data>/<run>/sb/build.
    assert options["data_root"] == ports.data_root


def test_an_uncached_builder_image_fails_setup_before_reservation(
    composed_env, monkeypatch
):
    from docker.errors import NotFound

    from rsi_harness.errors import SetupError

    ports, _, definition, store, mutate, runtimes = composed_env

    def inspect_image(reference):
        raise NotFound(reference)

    _with_builds(ports, definition, monkeypatch, inspect_image)
    # The broker never pulls a builder: the operator pre-pulls it.
    with pytest.raises(SetupError, match="work.build.builder_image.*unavailable"):
        ports.bind_sandbox_lease(mutate)
    assert store.read("run-1").sandbox_reservation is None
    assert runtimes == []


def test_builds_on_a_docker_host_without_a_unix_socket_fail_setup(
    composed_env, monkeypatch
):
    """Image loads use the daemon's Unix socket (B8): a tcp, TLS or ssh
    Docker host fails a run that grants builds before anything is reserved,
    not after its first BuildKit build."""
    from rsi_harness.errors import SetupError
    from tests.sandbox_helpers import builder_inspect

    ports, _, definition, store, mutate, runtimes = composed_env
    _with_builds(
        ports, definition, monkeypatch, lambda ref: builder_inspect(), socket_path=None
    )
    with pytest.raises(SetupError, match="unix:// Docker host"):
        ports.bind_sandbox_lease(mutate)
    assert store.read("run-1").sandbox_reservation is None
    assert runtimes == []
