"""Sandbox permission failures occur before any provisioning or auth work."""

import pytest
from typer.testing import CliRunner

from rsi_harness.config import EngineConfig
from rsi_harness.errors import SetupError
from rsi_harness.models import RunRequest
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import DEFAULT_TASK_TOML, write_harbor_task
from tests.sandbox_helpers import make_sandbox_policy, make_sandbox_task, sandbox_toml


def test_local_denial_precedes_inventory_auth_docker_and_roots(tmp_path, monkeypatch):
    import rsi_harness.runtime.production as production

    def forbidden(*args, **kwargs):
        pytest.fail("sandbox authorization must precede resource/auth access")

    class Inventory:
        list_devices = forbidden

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=Inventory(),
        rsi_loop_config=RSILoopConfig(),
    )
    monkeypatch.setattr(services, "_client", forbidden)
    monkeypatch.setattr(production, "resolve_agent_auth", forbidden)
    task = write_harbor_task(tmp_path, task_toml=DEFAULT_TASK_TOML + sandbox_toml())
    with pytest.raises(SetupError, match="sandbox.*policy"):
        services.run(RunRequest(task_dir=task))
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("build", [False, True])
def test_environment_request_is_accepted(tmp_path, monkeypatch, build):
    import rsi_harness.runtime.production as production
    from rsi_harness.models import GPUDevice
    from tests.sandbox_helpers import env_task_toml, make_env_policy

    class Reached(Exception):
        pass

    def reached(*args, **kwargs):
        raise Reached

    class Inventory:
        # The task declares GPUs; no nvidia-smi is needed to reach auth.
        def list_devices(self):
            return tuple(
                GPUDevice(index=i, uuid=f"GPU-{i}", name="test") for i in (0, 1)
            )

    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        rsi_loop_config=RSILoopConfig(),
        engine_config=EngineConfig(sandbox_policy=make_env_policy(tmp_path)),
        inventory=Inventory(),
    )
    monkeypatch.setattr(production, "resolve_agent_auth", reached)
    # Builds are approved once in the policy; the builder image itself is
    # checked when the lease is bound (production tests).
    toml = env_task_toml()
    if not build:
        toml = toml.replace("build = true\n", "")
    task = write_harbor_task(tmp_path, task_toml=DEFAULT_TASK_TOML + toml)
    # Validation passed: the run went on to agent authentication.
    with pytest.raises(Reached):
        services.run(RunRequest(task_dir=task))


def test_policy_plumbing_is_runtime_only(tmp_path, monkeypatch):
    from rsi_harness import cli
    from rsi_harness.runtime import production

    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    policy = make_sandbox_policy()
    config = EngineConfig(sandbox_policy=policy)
    monkeypatch.setattr(production, "ProductionRuntimeServices", factory)
    cli._services(tmp_path / "data", tmp_path / "logs", engine_config=config)
    assert captured["engine_config"].sandbox_policy == policy
    assert "sandbox_policy" not in RunRequest.model_fields


def test_cli_loads_explicit_operator_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from rsi_harness import cli
    from tests.test_cli import FakeServices

    task = tmp_path / "task"
    task.mkdir()
    policy_path = tmp_path / "policy.toml"
    policy = make_sandbox_policy()
    captured = {}

    def load(path):
        assert path == policy_path
        return policy

    def build(**kwargs):
        captured.update(kwargs)
        return FakeServices()

    monkeypatch.setattr(cli, "load_sandbox_policy", load, raising=False)
    monkeypatch.setattr(cli, "build_runtime_services", build)
    result = CliRunner().invoke(
        cli.app, ["run", str(task), "--sandbox-policy", str(policy_path)]
    )
    assert result.exit_code == 0, result.output
    assert captured["engine_config"].sandbox_policy == policy


def test_cli_refuses_docker_policy_with_cluster_before_adapter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from rsi_harness import cli
    from tests.sandbox_helpers import env_policy_toml

    monkeypatch.setattr(
        cli, "build_cluster_adapter", lambda *a, **k: pytest.fail("cluster provisioned")
    )
    # A Docker env policy: brokered Docker envs are local-only.
    (tmp_path / "policy.toml").write_text(env_policy_toml())
    result = CliRunner().invoke(
        cli.app,
        [
            "run",
            str(tmp_path),
            "--cluster",
            "slurm",
            "--sandbox-policy",
            "policy.toml",
        ],
    )
    assert result.exit_code == 2
    assert "local-only" in result.output


@pytest.mark.parametrize("kind", ["bluevela", "slurm"])
@pytest.mark.parametrize("dry_run", [True, False])
def test_cluster_sandbox_rejection_precedes_all_provisioning(
    tmp_path, monkeypatch, kind, dry_run
):
    from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
    from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
    from tests.cluster.bluevela.test_adapter import (
        RecordingScheduler,
        _profile,
        _request,
    )

    task = write_harbor_task(tmp_path, task_toml=DEFAULT_TASK_TOML + sandbox_toml())
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    cls = BlueVelaClusterAdapter if kind == "bluevela" else SlurmClusterAdapter
    adapter = cls(profile, scheduler=scheduler)
    monkeypatch.setattr(
        adapter, "_validate_runtime_inputs", lambda _: pytest.fail("runtime reached")
    )
    request = _request(tmp_path, dry_run=dry_run).model_copy(
        update={"task_dir": task.resolve()}
    )
    with pytest.raises(SetupError, match="sandbox.*local.*Docker"):
        adapter.run(request)
    assert scheduler.events == []
    assert not profile.storage.run_root.parent.exists()
    assert not request.logs_root.exists()


@pytest.mark.parametrize("entry", ["payload", "native"])
def test_direct_cluster_payload_cannot_bypass_sandbox_rejection(
    tmp_path, monkeypatch, entry
):
    from rsi_harness.cluster.bluevela import engine, runtime
    from tests.cluster.bluevela.test_engine import _payload

    payload = _payload(tmp_path)
    task = payload.run_plan.task.model_copy(update={"sandbox": make_sandbox_task()})
    plan = payload.run_plan.model_copy(update={"task": task})
    payload = payload.model_copy(update={"run_plan": plan})
    monkeypatch.setattr(
        engine, "bind_lsf_devices", lambda *a, **k: pytest.fail("GPU binding reached")
    )
    monkeypatch.setattr(
        runtime,
        "NativeEngineComposition",
        lambda *a, **k: pytest.fail("native runtime reached"),
    )
    with pytest.raises(SetupError, match="sandbox.*local.*Docker"):
        if entry == "payload":
            engine.run_engine_payload(payload)
        else:
            runtime.run_native_engine(payload, plan)
