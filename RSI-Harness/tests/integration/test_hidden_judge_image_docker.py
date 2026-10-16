"""Real Docker: which image a hidden Judge starts from."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import CompileOptions, RunRequest, RunStatus
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_public_tests_docker import _docker
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

# Installs a command outside WORKDIR, then submits twice.
_AGENT = (
    "set -eu\n"
    "printf '#!/bin/sh\\necho agent-tool\\n' > /usr/local/bin/agent-system-tool\n"
    "chmod 0755 /usr/local/bin/agent-system-tool\n"
    "printf %s 42 > answer.txt\n"
    "rsi-submit\n"
    "rsi-submit\n"
)

_REWARD = (
    'if [ "$(cat "$answer")" = 42 ]; then reward=1; else reward=0; fi\n'
    'printf \'{"reward": %s}\\n\' "$reward" > /logs/verifier/reward.json\n'
)

# The clean Work image with a writable layer that each round discards.
_CLEAN_JUDGE = (
    "#!/bin/bash\n"
    "set -eu\n"
    "answer=/workspace/answer.txt\n"
    "if command -v agent-system-tool; then exit 3; fi\n"
    "test ! -e /etc/judge-only\n"
    "printf judge > /etc/judge-only\n"
    'mkdir -p "$HOME/.cache" /var/tmp /run && touch "$HOME/.cache/x" /var/tmp/x\n'
    "printf '#!/bin/sh\\necho tmp-exec\\n' > /tmp/judge-exec\n"
    "chmod 0755 /tmp/judge-exec && /tmp/judge-exec\n"
    "if touch /workspace/judge-write 2>/dev/null; then exit 4; fi\n" + _REWARD
)

# WORKDIR "/": the Judge runs from the snapshot of Work.
_SNAPSHOT_JUDGE = (
    "#!/bin/bash\n"
    "set -eu\n"
    "answer=/answer.txt\n"
    'test "$(agent-system-tool)" = agent-tool\n' + _REWARD
)


def _run(tmp_path: Path, monkeypatch, request, task: Path):
    client = _docker()
    data = tmp_path / "data"
    events: list[tuple[str, object]] = []

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            return replace(prepared, command=("/bin/bash", "-lc", _AGENT))

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
        event_callback=lambda name, value: events.append((name, value)),
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
    feedback = "".join(
        (run_root / "feedback" / f"agent-{n}.log").read_text()
        for n in (1, 2)
        if (run_root / "feedback" / f"agent-{n}.log").exists()
    )
    assert result.status == RunStatus.COMPLETED, feedback
    assert result.total_rounds == 2
    assert result.best_score == 1, feedback
    plan = json.loads((run_root / "run-plan.json").read_text())
    return plan, events, feedback


def _task(tmp_path: Path, judge: str, *, workdir: str) -> Path:
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    config = task / "task.toml"
    config.write_text(
        config.read_text().replace('workdir = "/workspace"', f'workdir = "{workdir}"')
    )
    (task / "tests" / "test.sh").write_text(judge)
    return task


def test_real_hidden_judge_starts_from_the_clean_image_with_a_writable_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    task = _task(tmp_path, _CLEAN_JUDGE, workdir="/workspace")

    plan, events, feedback = _run(tmp_path, monkeypatch, request, task)

    assert plan["task"]["verifier"]["clean_image"] is True
    assert not [name for name, _ in events if name == "judge_clean_image_unavailable"]
    assert feedback.count("tmp-exec") == 2


def test_real_hidden_judge_with_workdir_root_falls_back_to_the_work_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    task = _task(tmp_path, _SNAPSHOT_JUDGE, workdir="/")

    plan, events, _feedback = _run(tmp_path, monkeypatch, request, task)

    assert plan["task"]["verifier"]["clean_image"] is False
    assert ("judge_clean_image_unavailable", "the task WORKDIR is /") in events
