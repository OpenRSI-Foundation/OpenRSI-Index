"""Real Docker: an Agent started through the real adapter keeps the image's PATH.

The task image is Debian (minimal-sandbox, python:3.12-slim-bookworm), whose
/etc/profile resets PATH in a login shell. A fake Codex CLI, installed by the
Agent's real Work image build and started by the real RSI Loop adapter's
command, must find the image's own ENV PATH entry (``task-python``), the
injected ``rsi-sandbox`` and ``rsi-submit``, and drive a managed sandbox. Like
the real npm CLIs it is a ``#!/usr/bin/env node`` launcher, and the task image
has its own ``node`` ahead of /usr/local/bin: the CLI must still run on the
pinned node, while the commands it runs see the task's.

Run as a non-root user with the fake firewall.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

import pytest

from rsi_harness.config import EngineConfig
from rsi_harness.models import CompileOptions, RunRequest, RunStatus
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_contracts import SandboxPolicy
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.agent.base import NODE_BIN, NODE_INSTALL_CMD
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.sandbox_support import (
    assert_no_sandbox_resources,
    require_sandbox_authority,
)
from tests.integration.test_public_tests_docker import _docker

pytestmark = pytest.mark.integration

SANDBOX_FIXTURE = Path(__file__).parents[1] / "fixtures/tasks/minimal-sandbox"

# The npm-style launcher: on the pinned node it runs the Agent's work.
_FAKE_CODEX_JS = r"""#!/usr/bin/env node
console.log(`fake-agent-node:${process.execPath}`);
const run = require('node:child_process').spawnSync(
  '/usr/local/lib/fake-codex/agent.sh', { stdio: 'inherit' });
process.exit(run.status ?? 1);
"""

# What the fake CLI does with whatever arguments the adapter passes it.
_FAKE_CODEX = r"""#!/bin/bash
set -euo pipefail
echo "fake-agent-path:[$PATH]"
echo "fake-agent-task-node:$(command -v node)"
echo "fake-agent-sandbox:$(command -v rsi-sandbox)"
echo "fake-agent-submit:$(command -v rsi-submit)"
task-python --version
child=$(rsi-sandbox create offline --lifetime 100 --json |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["child_id"])')
rsi-sandbox exec "$child" --json -- python3 -c 'print(6 * 7)' > /tmp/r.json
python3 -c 'import json; print(json.load(open("/tmp/r.json"))["stdout"].strip())' \
    > answer.txt
rsi-sandbox destroy "$child" --json
rsi-submit
echo fake-agent-done
"""


def test_fake_agent_cli_finds_image_path_and_sandbox_client_on_debian(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, image = require_sandbox_authority()
    client.close()
    # The real Work image build, with the fake in place of the pinned CLI,
    # installed like npm does: a symlink to a node script.
    monkeypatch.setattr(
        CodexAgent,
        "install_cmds",
        [
            NODE_INSTALL_CMD,
            "mkdir -p /usr/local/lib/fake-codex\n"
            "cat > /usr/local/lib/fake-codex/agent.sh <<'FAKE_CODEX'\n"
            + _FAKE_CODEX
            + "FAKE_CODEX\n"
            "cat > /usr/local/lib/fake-codex/codex.js <<'FAKE_CODEX_JS'\n"
            + _FAKE_CODEX_JS
            + "FAKE_CODEX_JS\n"
            "chmod 0755 /usr/local/lib/fake-codex/agent.sh"
            " /usr/local/lib/fake-codex/codex.js\n"
            "ln -s ../lib/fake-codex/codex.js /usr/local/bin/codex",
            # Stands in for a task image's own node (nvm, conda, a venv) on its
            # ENV PATH ahead of /usr/local/bin.
            "printf '#!/bin/sh\\necho task-node-ran >&2\\nexit 97\\n'"
            " > /opt/fixture/bin/node && chmod 0755 /opt/fixture/bin/node",
        ],
    )
    # The broker's socket path must fit in sockaddr_un.
    tmp_path = Path(tempfile.mkdtemp(prefix="rsi-p-"))
    request.addfinalizer(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
    task = tmp_path / "task"
    shutil.copytree(SANDBOX_FIXTURE, task)
    config = task / "task.toml"
    config.write_text(config.read_text().replace("__APPROVED_IMAGE_ID__", image.id))
    declaration = HarborTaskCompiler().compile(task, CompileOptions()).sandbox
    policy = SandboxPolicy(
        profiles=declaration.profiles,
        work=declaration.work,
        judge=declaration.judge,
        run_limits=declaration.judge.limits,
        pool_cpus=4,
        pool_memory_mb=4096,
    )
    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=_docker(),
        firewall_backend=FakeFirewallBackend(),
        rsi_loop_config=RSILoopConfig(),
        engine_config=EngineConfig(
            data_root=data, logs_root=tmp_path / "logs", sandbox_policy=policy
        ),
        omit_gpu_device_requests_for_tests=True,
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
            options=CompileOptions(agent_name="codex"),
        )
    )

    store = LeaseStore(data / "leases")
    assert result.artifacts_dir is not None, store.read(result.run_id).error
    output = (result.artifacts_dir / "agent_output.txt").read_text()
    assert result.status is RunStatus.COMPLETED, output
    assert result.agent_exit_code == 0, output
    assert result.best_score == 1.0, output
    assert "fake-agent-done" in output
    # The CLI ran on the pinned node; the Agent's commands see the task's.
    assert f"fake-agent-node:{NODE_BIN}" in output
    assert "fake-agent-task-node:/opt/fixture/bin/node" in output
    assert "task-node-ran" not in output
    # The exec's PATH as Docker gave it: the sandbox client first, then the
    # image's own ENV PATH, which Debian's login profile would have dropped.
    [path] = re.findall(r"fake-agent-path:\[(.*)\]", output)
    entries = path.split(":")
    assert entries[0] == "/run/rsi-harness/sandbox", path
    assert "/opt/fixture/bin" in entries, path
    assert "fake-agent-sandbox:/run/rsi-harness/sandbox/rsi-sandbox" in output
    assert "fake-agent-submit:/usr/local/bin/rsi-submit" in output
    lease = store.read(result.run_id)
    assert sorted(child.owner.phase for child in lease.sandboxes) == [
        "judge",
        "work",
    ]
    assert all(child.state == "removed" for child in lease.sandboxes)
    services.cleanup(result.run_id, delete_workspace=True)
    assert_no_sandbox_resources(_docker(), store, result.run_id)
