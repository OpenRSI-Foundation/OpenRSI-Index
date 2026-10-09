"""Stock Harbor trials through the injected plugin on real E2B (opt-in).

Set ``RSI_E2B_KEY_FILE`` to the operator's key file (read only by the
broker). As test_harbor_env_trial.py, but every env is an E2B sandbox: TB2
fix-git (``RSI_TB2_DIR``; public network) and one SWE-bench Verified task of
sample_tasks/swebench-in-judge (network none, the operator's static tmux
from ``RSI_STATIC_TMUX`` injected). Oracle scores 1, nop 0, and no sandbox
of the run is left. Templates stay on the account (a shared cache).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import tomllib
from pathlib import Path

import pytest

from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_e2b import (
    e2b_client,
    e2b_env_runtime,
    kill_run_sandboxes,
    run_metadata,
)
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from tests.integration.harbor_env_support import TASK, reward, run_trial, use_endpoint
from tests.integration.test_sandbox_offline_tmux import (
    static_tmux as static_tmux,
)
from tests.integration.test_swebench_in_judge_sample import POLICY, SAMPLE, with_tmux
from tests.runtime.test_sandbox_budget import authority
from tests.runtime.test_sandbox_e2b import e2b_policy_text
from tests.sandbox_helpers import FakeSandboxBackend, load_policy_text, make_env_task

KEY_FILE = os.environ.get("RSI_E2B_KEY_FILE")
TB2 = Path(os.environ.get("RSI_TB2_DIR", "/mnt/y1/temp/terminal_bench_2"))
SWE_TASK = "pallets__flask-5014"
E2B = f"api_key_file = {KEY_FILE!r}"
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not KEY_FILE, reason="set RSI_E2B_KEY_FILE for live E2B"),
]


def e2b_backend(policy: str) -> str:
    """``policy`` (an operator policy's text) on the e2b backend."""
    policy = policy.replace(
        "no_new_privileges = true\n", 'no_new_privileges = true\nbackend = "e2b"\n', 1
    )
    return policy.replace(
        "\n[environments.work]",
        f"\n[environments.host.e2b]\n{E2B}\n\n[environments.work]",
        1,
    )


class E2BHarbor:
    """One run's broker on E2B behind the real Work/Judge endpoint lifecycle."""

    def __init__(self, tmp_path: Path, policy: str, task: str, parent) -> None:
        self.run_id = "e2b-harbor-" + os.urandom(4).hex()
        grant = resolve_env_grant(
            make_env_task(task),
            load_policy_text(tmp_path, policy),
            {},
            parent_cpus=parent[0],
            parent_memory_mb=parent[1],
        )
        self.client = e2b_client(grant.environments.host.e2b)
        # A short data root keeps the socket path within 107 bytes.
        self.data = Path(tempfile.mkdtemp(prefix="rsi-e2b-"))
        self.sb = self.data / self.run_id / "sb"
        self.sb.mkdir(mode=0o700, parents=True)
        self.journal = SandboxJournal(
            authority(LeaseStore(tmp_path / "leases"), self.run_id)
        )
        runtime = e2b_env_runtime(
            self.client,
            spool_root=self.data / self.run_id / "spool",
            host=grant.environments.host,
        )
        self.broker = SandboxBroker(
            grant, FakeSandboxBackend(), self.journal, time.monotonic, envs=runtime
        )
        self.lifecycle = SandboxLifecycle()
        self.lifecycle.configure(self.broker, self.sb, self.run_id, "task")

    def work(self):
        endpoint = self.lifecycle.prepare_work()
        self.lifecycle.activate_work(time.monotonic() + 3600)
        return endpoint

    def judge(self):
        self.work()
        self.lifecycle.freeze_work()
        endpoint = self.lifecycle.prepare_judge("round-1")
        self.lifecycle.activate_judge(time.monotonic() + 3600)
        return endpoint

    def close(self) -> None:
        try:
            self.lifecycle.close()
            assert not self.sb.exists()
            assert not self.broker.recovery_required
            assert self.journal.envs() == ()
        finally:
            shutil.rmtree(self.data, ignore_errors=True)
            kill_run_sandboxes(self.client, self.run_id)
            assert self.client.list(run_metadata(self.run_id)) == []


@pytest.fixture
def e2b_harbor(tmp_path):
    made = []

    def make(policy: str, task: str = TASK, parent=(1, 256)) -> E2BHarbor:
        made.append(E2BHarbor(tmp_path, policy, task, parent))
        return made[-1]

    yield make
    for harbor in made:
        harbor.close()


async def oracle_and_nop(task: Path, trials: Path) -> None:
    for agent, expected in (("oracle", 1.0), ("nop", 0.0)):
        result = await run_trial(task, trials, agent)
        assert result.exception_info is None, result.exception_info
        assert reward(result) == expected, agent


@pytest.mark.asyncio
@pytest.mark.skipif(not (TB2 / "fix-git").is_dir(), reason="no TB2 fix-git")
async def test_tb2_fix_git_on_e2b_scores_oracle_one_and_nop_zero(
    e2b_harbor, monkeypatch, tmp_path
):
    """test.sh refuses a cwd of /: the image's WORKDIR must apply."""
    policy = e2b_policy_text(e2b=E2B)
    policy = policy.replace("cpus_per_container = 4\n", "cpus_per_container = 2\n")
    policy = policy.replace(
        "memory_mb_per_container = 8192\n", "memory_mb_per_container = 2048\n"
    )
    # TB2 asks for 10 GiB of (soft) disk.
    policy = policy.replace("max_disk_mb_live = 8192", "max_disk_mb_live = 16384")
    harbor = e2b_harbor(policy)
    use_endpoint(monkeypatch, harbor.work())
    task = tmp_path / "fix-git"
    shutil.copytree(TB2 / "fix-git", task)
    await oracle_and_nop(task, tmp_path / "trials")


@pytest.mark.asyncio
async def test_a_swebench_task_on_e2b_scores_offline_with_tmux(
    e2b_harbor, monkeypatch, tmp_path, static_tmux
):
    policy = with_tmux(e2b_backend(POLICY.read_text()), static_tmux, tmp_path)
    # Keep the run's sandboxes within a small account's concurrency.
    policy = policy.replace("max_envs_live = 3", "max_envs_live = 2")
    sample = (SAMPLE / "task.toml").read_text()
    service = tomllib.loads(sample)["environment"]
    harbor = e2b_harbor(policy, sample, (service["cpus"], service["memory_mb"]))
    envs = harbor.broker.envs
    networks, installs = [], []
    create, install = envs.env_create, envs.tool_install

    def watch_create(credential, spec, request_id):
        networks.append(spec["network"])
        return create(credential, spec, request_id)

    def watch_install(credential, env_id, service, tool):
        installs.append((service, tool))
        return install(credential, env_id, service, tool)

    monkeypatch.setattr(envs, "env_create", watch_create)
    monkeypatch.setattr(envs, "tool_install", watch_install)
    use_endpoint(monkeypatch, harbor.judge())
    task = tmp_path / SWE_TASK
    shutil.copytree(SAMPLE / "tests" / "swebench-verified" / SWE_TASK, task)

    await oracle_and_nop(task, tmp_path / "trials")
    assert networks == ["none", "none"]
    assert installs == [("main", "tmux")] * 2
