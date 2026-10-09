"""Real two-round parent/child acceptance; missing firewall is not a pass."""

import os
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from docker.errors import NotFound

from rsi_harness.config import EngineConfig
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import CompileOptions, RunRequest, RunStatus
from rsi_harness.runtime.network import DockerIptablesFirewallBackend
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_contracts import SandboxError, SandboxPolicy
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.integration.sandbox_support import (
    assert_no_sandbox_resources,
    require_sandbox_authority,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/tasks/minimal-sandbox"
WORK_SCRIPT = (FIXTURE / "work.sh").read_text()


def _fixture_task(root, image_id):
    task = root / "task"
    shutil.copytree(FIXTURE, task)
    config = task / "task.toml"
    config.write_text(config.read_text().replace("__APPROVED_IMAGE_ID__", image_id))
    return task


def test_minimal_sandbox_fixture_compiles_without_gpu(tmp_path):
    task = _fixture_task(tmp_path, "sha256:" + "a" * 64)
    definition = HarborTaskCompiler().compile(task, CompileOptions())
    assert definition.gpu_requirement.count == definition.verifier.gpu_count == 0
    assert definition.service.cpus == 1
    assert definition.sandbox.work.limits.max_live == 1


@pytest.mark.integration
def test_real_two_round_harness_preserves_work_and_reclaims_judge(monkeypatch):
    client, image = require_sandbox_authority()
    firewall = DockerIptablesFirewallBackend(client)
    if not firewall.probe():
        client.close()
        message = "required host DOCKER-USER/INPUT firewall authority unavailable"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            return replace(prepared, command=("/bin/bash", "-c", WORK_SCRIPT))

    # Keep real image preparation/bootstrap, but no model CLI download or API call.
    monkeypatch.setattr(
        CodexAgent,
        "install_cmds",
        [
            "printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/codex && "
            "chmod 0755 /usr/local/bin/codex"
        ],
    )
    original_prepare = SandboxLifecycle.prepare_judge
    endpoints = []

    def observe_judge(self, round_id):
        endpoint = original_prepare(self, round_id)
        if endpoint is not None and endpoint not in [item[1] for item in endpoints]:
            if endpoints:
                previous = endpoints[-1][1]
                with pytest.raises(SandboxError, match="credential"):
                    self.broker.capabilities(previous.environment["RSI_SANDBOX_TOKEN"])
            endpoints.append((self.broker, endpoint))
        return endpoint

    monkeypatch.setattr(SandboxLifecycle, "prepare_judge", observe_judge)
    # Preserve durable recovery authority if final cleanup cannot prove absence.
    temporary = tempfile.TemporaryDirectory(prefix="rsi-e2e-", delete=False)
    with temporary as directory:
        root = Path(directory)
        task = _fixture_task(root, image.id)
        definition = HarborTaskCompiler().compile(task, CompileOptions())
        declaration = definition.sandbox
        limits = declaration.judge.limits.model_copy(
            update={
                "max_live": 2,
                "max_created": 3,
                "max_cpus": 2,
                "max_memory_mb": 256,
                "max_lifetime_sec": 240,
                "max_log_bytes": 8 * 1024**2,
            }
        )
        policy = SandboxPolicy(
            profiles=declaration.profiles,
            work=declaration.work,
            judge=declaration.judge,
            run_limits=limits,
            pool_cpus=4,
            pool_memory_mb=4096,
        )
        store = LeaseStore(root / "data/leases")
        services = ProductionRuntimeServices(
            data_root=root / "data",
            logs_root=root / "logs",
            docker_client=client,
            firewall_backend=firewall,
            rsi_loop_config=RSILoopConfig(),
            engine_config=EngineConfig(
                data_root=root / "data", logs_root=root / "logs", sandbox_policy=policy
            ),
            agent_adapter_factory=lambda config, runtime: ScriptedAgent(
                config, runtime=runtime
            ),
        )
        try:
            result = services.run(
                RunRequest(task_dir=task, options=CompileOptions(), agent_name="codex")
            )
            assert result.status == RunStatus.COMPLETED
            assert result.total_rounds == 2
            assert result.best_score == 1.0
            lease = store.read(result.run_id)
            judges = [c for c in lease.sandboxes if c.owner.phase == "judge"]
            assert len(judges) == len({c.container_id for c in judges}) == 2
            assert all(c.state == "removed" for c in lease.sandboxes)
            assert lease.sandbox_reservation is None
            # Catch false "removed" journal entries before explicit recovery can
            # hide a normal-run child reclamation regression.
            for child in lease.sandboxes:
                assert child.container_id
                with pytest.raises(NotFound):
                    client.containers.get(child.container_id)
            assert len(endpoints) == 2
            output = next((root / "logs").rglob("agent_output.txt")).read_text()
            assert "preserved-across-two-rounds" in output
            assert '"reward": 0.0' in output and '"reward": 1.0' in output
        finally:
            try:
                for run_id in store.list_run_ids():
                    services.cleanup(run_id, delete_workspace=True)
                    assert_no_sandbox_resources(client, store, run_id)
            except BaseException:
                print(f"E2E cleanup incomplete; recovery data retained at {root}")
                raise
            else:
                temporary.cleanup()
            finally:
                client.close()
