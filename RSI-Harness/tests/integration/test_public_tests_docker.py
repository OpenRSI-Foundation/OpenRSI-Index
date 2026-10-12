"""Real Docker: a public-visibility Work reads the Judge's tests read-only."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path, PurePosixPath

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    CompileOptions,
    ContainerMount,
    ContainerSpec,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

IMAGE = "ubuntu:24.04"


def _docker():
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
    except Exception as error:
        pytest.skip(f"local cached {IMAGE} Docker authority unavailable: {error}")
    return client


def _public_task(root: Path) -> Path:
    task = root / "task"
    shutil.copytree(FIXTURE, task)
    config = task / "task.toml"
    config.write_text(
        config.read_text()
        + '\n[metadata.rsi_harness.verifier]\nvisibility = "public"\n'
    )
    (task / "tests" / "rubric.txt").write_text("answer must be 42\n")
    # A public Judge runs from the clean image, read-only except /tmp.
    script = task / "tests" / "test.sh"
    script.write_text(
        script.read_text()
        .replace(
            "test ! -e /etc/rsi-harness-judge-only.txt\n",
            "if touch /etc/rsi-harness-judge-only.txt 2>/dev/null; then exit 7; fi\n",
        )
        .replace(
            "> /etc/rsi-harness-judge-only.txt", "> /tmp/rsi-harness-judge-only.txt"
        )
        # Without model access a Judge has no rsi-agent.
        .replace(
            "set -eu\n",
            "set -eu\n"
            "if command -v rsi-agent || test -e /run/rsi-agent; then exit 8; fi\n",
            1,
        )
    )
    return task


def test_real_public_work_mount_is_read_only_and_attested(tmp_path: Path) -> None:
    client = _docker()
    task = _public_task(tmp_path)
    engine = tmp_path / "engine"
    engine.mkdir()
    runtime = DockerContainerRuntime(
        client,
        run_id="public-tests",
        task_id="minimal-gpu",
        role="work",
        task_source_dir=task,
        allowed_mount_roots=(engine,),
        work_public_tests=True,
    )
    work = runtime.create(
        ContainerSpec(
            image=IMAGE,
            command=("sleep", "infinity"),
            user="0",
            mounts=(
                ContainerMount(
                    source=task / "tests",
                    target=PurePosixPath("/tests"),
                    read_only=True,
                ),
            ),
        )
    )
    try:
        runtime.attest_work_tests_mount(work)
        # Start without a Work network policy: this test only checks the bind.
        container = client.containers.get(work.container_id)
        container.start()
        code, shown = container.exec_run(("cat", "/tests/rubric.txt"))
        assert code == 0
        assert b"answer must be 42" in shown
        code, written = container.exec_run(("touch", "/tests/agent-edit"))
        assert code != 0
        assert b"Read-only file system" in written
        assert not (task / "tests" / "agent-edit").exists()
    finally:
        runtime.remove(work)


def test_real_public_run_work_reads_tests_and_judge_runs_the_same_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Full production run as the current user; only iptables is faked."""
    client = _docker()
    task = _public_task(tmp_path)
    before = hash_tree(task / "tests")
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
                    "grep -q 'public: `/tests`' /tmp/rsi-agent-prompt.md\n"
                    "printf 'work saw: %s\\n' \"$(cat /tests/rubric.txt)\"\n"
                    "if touch /tests/agent-edit 2>/dev/null; then exit 9; fi\n"
                    "if command -v rsi-agent; then exit 10; fi\n"
                    "printf %s 42 > answer.txt\n"
                    "rsi-submit\n",
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
    assert result.status == RunStatus.COMPLETED, agent_output
    assert result.total_rounds == 1
    assert result.best_score == 1
    assert "work saw: answer must be 42" in agent_output
    assert "answer accepted" in (run_root / "feedback" / "agent-1.log").read_text()
    assert hash_tree(task / "tests") == before
    assert not (task / "tests" / "agent-edit").exists()
