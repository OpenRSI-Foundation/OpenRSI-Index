"""Real Docker: every Judge round reads the Agent trajectory read-only."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import CompileOptions, RunRequest, RunStatus
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

IMAGE = "ubuntu:24.04"

JUDGE_SCRIPT = """#!/bin/bash
set -eu
output=/logs/agent/agent_output.txt
test -r /logs/agent/manifest.json
test -r /logs/agent/agent_prompt.md
# Readable by a non-root Judge user too.
setpriv --reuid=65534 --regid=65534 --clear-groups cat "$output" > /dev/null
if touch /logs/agent/judge-edit 2>/dev/null; then
    printf 'trajectory is writable\\n'
    exit 7
fi
if grep -q after-last-submit "$output"; then
    printf 'trajectory leaked later output\\n'
    exit 8
fi
markers=$(grep -c '^trajectory-marker' "$output" || true)
printf 'judge saw markers=%s\\n' "$markers"
printf '{"reward": %s}\\n' "$markers" > /logs/verifier/reward.json
"""


def _docker():
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
    except Exception as error:
        pytest.skip(f"local cached {IMAGE} Docker authority unavailable: {error}")
    return client


def test_real_judge_reads_each_rounds_frozen_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Full production run as the current user; only iptables is faked."""
    client = _docker()
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    (task / "tests" / "test.sh").write_text(JUDGE_SCRIPT)
    data = tmp_path / "data"

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            return replace(
                prepared,
                command=(
                    "/bin/bash",
                    "-lc",
                    "set -eu\n"
                    "echo trajectory-marker-one\n"
                    "sleep 1\n"
                    "rsi-submit\n"
                    "echo trajectory-marker-two\n"
                    "sleep 1\n"
                    "rsi-submit\n"
                    "echo after-last-submit\n",
                ),
            )

    # Keep real image preparation, but no model CLI download or API call.
    monkeypatch.setattr(
        CodexAgent,
        "install_cmds",
        [
            "printf '#!/bin/sh\\nexit 0\\n' > /usr/local/bin/codex && "
            "chmod 0755 /usr/local/bin/codex"
        ],
    )
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: ScriptedAgent(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
    )

    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)

    result = services.run(
        RunRequest(
            task_dir=task,
            agent_name="codex",
            gpu_selectors=("0",),
            options=CompileOptions(agent_name="codex", primary_reward="reward"),
        )
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    agent_output = (run_root / "agent_output.txt").read_text()
    feedback = run_root / "feedback"
    assert result.status == RunStatus.COMPLETED, agent_output
    assert "after-last-submit" in agent_output
    assert result.total_rounds == 2
    assert "judge saw markers=1" in (feedback / "agent-1.log").read_text()
    assert "judge saw markers=2" in (feedback / "agent-2.log").read_text()
    trajectories = run_root / "judge-trajectory"
    for round_id in ("agent-1", "agent-2"):
        assert sorted(p.name for p in (trajectories / round_id).iterdir()) == [
            "manifest.json"
        ]
    second = json.loads((trajectories / "agent-2" / "manifest.json").read_text())
    output_entry = next(
        entry for entry in second["files"] if entry["path"] == "agent_output.txt"
    )
    assert second["previous_round_id"] == "agent-1"
    assert 0 < output_entry["previous_size"] < output_entry["source_size"]
