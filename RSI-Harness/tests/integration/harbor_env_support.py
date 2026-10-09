"""A real v2 broker for Harbor plugin acceptance, run as the current user.

The broker, lifecycle, server, endpoint injection and Docker backends are the
production ones; only the iptables half is faked (FakeFirewallBackend), so
real egress filtering and intra-bridge ACCEPT stay with the operator's root
check (spec 8), which runs these tests as root in root mode
(RSI_SANDBOX_ROOT_MODE=1: the real firewall and a loop-ext4 builder).
Every Docker object carries the run's label and is removed by label when the
sandbox closes, also after a failure.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

import docker
import pytest
from docker.errors import DockerException, ImageNotFound

from rsi_harness.runtime.production import SANDBOX_ENV_DOCKER_TIMEOUT_SECONDS
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox import SandboxBroker
from rsi_harness.runtime.sandbox_budget import SandboxJournal
from rsi_harness.runtime.sandbox_env_contracts import sandbox_spool_root
from rsi_harness.runtime.sandbox_envs import docker_env_runtime
from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from rsi_harness.runtime.sandbox_policy import resolve_env_grant
from tests.integration.sandbox_support import (
    assert_no_rules,
    root_mode,
    sandbox_firewall,
)
from tests.integration.test_sandbox_env_docker import remove_labelled
from tests.runtime.test_sandbox_budget import authority
from tests.sandbox_helpers import (
    FakeSandboxBackend,
    env_policy_toml,
    load_policy_text,
    make_env_task,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "tasks"
PLUGIN = "rsi_sandbox_harbor:ManagedSandboxEnvironment"
TASK = """
[metadata.rsi_harness.sandbox]
version = 2
[metadata.rsi_harness.sandbox.environments.work]
network = ["public", "none"]
pull = true
[metadata.rsi_harness.sandbox.environments.judge]
network = ["public", "none"]
pull = true
"""
BUILDKIT = "moby/buildkit:v0.27.1"
# A tmpfs builder (loop-ext4 needs root; the root mode uses it): its state
# counts against memory.
BUILD_GRANT = {
    "state_fs": "loop-ext4" if root_mode() else "tmpfs",
    "cpus": 2,
    "memory_mb": 4096,
    "pids": 4096,
    "disk_mb": 2048,
    "max_image_mb": 2048,
    "max_images_total_mb": 2048,
}


def docker_or_skip(*images: str):
    """A Docker client with production's env-runtime timeout.

    Base images must already be cached; a missing one skips the test
    unless RSI_ALLOW_PULL=1, which pulls it once and keeps it (a cache).
    """
    try:
        client = docker.from_env(timeout=SANDBOX_ENV_DOCKER_TIMEOUT_SECONDS)
        client.ping()
        missing = []
        for reference in images:
            try:
                client.images.get(reference)
            except ImageNotFound:
                missing.append(reference)
        if missing and os.environ.get("RSI_ALLOW_PULL") != "1":
            client.close()
            pytest.skip(
                f"base images not cached (set RSI_ALLOW_PULL=1): {', '.join(missing)}"
            )
        for reference in missing:
            if "@" in reference:  # by digest, as a pre-pull manifest names it
                client.images.pull(reference)
                continue
            repository, _, tag = reference.rpartition(":")
            client.images.pull(repository, tag=tag)
        return client
    except (DockerException, OSError) as error:
        message = f"Docker or a base image is unavailable: {error}"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)


class LiveSandbox:
    """One run's broker behind the real Work/Judge endpoint lifecycle.

    ``policy`` (an operator policy's text) with ``task`` (a task.toml's
    text) and ``parent`` (its service CPUs and MiB) grant as that operator
    policy does for that task, e.g. a sample's shipped policy; by default a
    generous test grant."""

    def __init__(
        self,
        client,
        tmp_path: Path,
        name: str,
        *,
        build: bool = False,
        policy: str | None = None,
        task: str | None = None,
        parent: tuple[int, int] = (1, 256),
    ) -> None:
        self.client = client
        self.run_id = f"m7-{name}-{uuid.uuid4().hex[:12]}"
        builders, overrides = {}, None
        if build:
            # Builds through the broker's per-session BuildKit builder (M8).
            inspected = client.api.inspect_image(BUILDKIT)
            builders = {"work": inspected, "judge": inspected}
            overrides = {**BUILD_GRANT, "builder_image": inspected["RepoDigests"][0]}
        if task is None:
            task = TASK
            if build:
                task = TASK.replace("pull = true", "pull = true\nbuild = true")
        if policy is None:
            # TB2 tasks ask for 10 GiB of (soft) disk: widen the live disk.
            policy = env_policy_toml(build=overrides).replace(
                "max_disk_mb_live = 8192", "max_disk_mb_live = 16384"
            )
        grant = resolve_env_grant(
            make_env_task(task),
            load_policy_text(tmp_path, policy),
            builders,
            parent_cpus=parent[0],
            parent_memory_mb=parent[1],
        )
        self.firewall = sandbox_firewall(client)
        # Production's layout: <data>/<run>/sb holds the endpoints and the
        # spool; a short data root keeps the socket path within 107 bytes.
        self.data = Path(tempfile.mkdtemp(prefix="rsi-m7-"))
        self.sb = self.data / self.run_id / "sb"
        self.sb.mkdir(mode=0o700, parents=True)
        self.spool = sandbox_spool_root(self.data, self.run_id)
        runtime = docker_env_runtime(
            client,
            self.firewall,
            run_id=self.run_id,
            spool_root=self.spool,
            docker_root=client.info()["DockerRootDir"],
            host=grant.environments.host,
            data_root=self.data,
        )
        self.journal = SandboxJournal(
            authority(LeaseStore(tmp_path / "leases"), self.run_id)
        )
        self.broker = SandboxBroker(
            grant, FakeSandboxBackend(), self.journal, time.monotonic, envs=runtime
        )
        self.lifecycle = SandboxLifecycle()
        self.lifecycle.configure(self.broker, self.sb, self.run_id, "task")

    def work(self, work_sec: float = 3600):
        endpoint = self.lifecycle.prepare_work()
        self.lifecycle.activate_work(time.monotonic() + work_sec)
        return endpoint

    def judge(
        self,
        round_id: str = "round-1",
        *,
        work_sec: float = 3600,
        verifier_sec: float = 3600,
    ):
        """A Judge round's endpoint, opened as production does: after the
        Work session (``work_sec`` to its deadline) ran and froze; the round
        has ``verifier_sec``."""
        self.work(work_sec)
        self.lifecycle.freeze_work()
        endpoint = self.lifecycle.prepare_judge(round_id)
        self.lifecycle.activate_judge(time.monotonic() + verifier_sec)
        return endpoint

    def round_objects(self, round_id: str = "round-1"):
        filters = {
            "label": [
                f"rsi-harness.run-id={self.run_id}",
                f"rsi-harness.round-id={round_id}",
            ]
        }
        return (
            self.client.containers.list(all=True, filters=filters),
            self.client.volumes.list(filters=filters),
            self.client.networks.list(filters=filters),
            self.client.images.list(filters=filters),
        )

    def labelled(self):
        filters = {"label": f"rsi-harness.run-id={self.run_id}"}
        return (
            self.client.containers.list(all=True, filters=filters),
            self.client.volumes.list(filters=filters),
            self.client.networks.list(filters=filters),
        )

    def built_images(self):
        return self.client.images.list(
            filters={"label": f"rsi-harness.run-id={self.run_id}"}
        )

    def close(self) -> None:
        try:
            self.lifecycle.close()
            # A clean close leaves no <run>/sb: endpoints, spool, sb (A5).
            sb_left = self.sb.exists()
        finally:
            shutil.rmtree(self.data, ignore_errors=True)
            errors = remove_labelled(
                self.client, {"label": f"rsi-harness.run-id={self.run_id}"}
            )
            leftovers = self.labelled()
            images = self.built_images()
        assert (errors, leftovers, images) == ([], ([], [], []), [])
        assert not sb_left
        assert not self.broker.recovery_required
        assert_no_rules(self.firewall, self.run_id)
        assert self.journal.envs() == ()
        assert self.journal.builders() == ()


@pytest.fixture
def live_sandbox(request, tmp_path):
    """``live_sandbox(name, *images, build=False, **grant)``: a LiveSandbox,
    closed after the test; ``build`` grants image builds (the BuildKit image
    must be cached); ``grant`` is LiveSandbox's policy, task and parent."""
    made = []

    def make(name, *images, build=False, **grant):
        client = docker_or_skip(*images, *((BUILDKIT,) if build else ()))
        request.addfinalizer(client.close)
        sandbox = LiveSandbox(client, tmp_path, name, build=build, **grant)
        made.append(sandbox)
        return sandbox

    yield make
    for sandbox in made:
        sandbox.close()


def use_endpoint(monkeypatch, endpoint) -> None:
    """Point this process at the endpoint the way Work or Judge sees it."""
    monkeypatch.setenv("RSI_SANDBOX_SOCKET", str(endpoint.directory / "s"))
    monkeypatch.setenv("RSI_SANDBOX_TOKEN", endpoint.environment["RSI_SANDBOX_TOKEN"])
    # The injected modules, exactly as PYTHONPATH=$RSI_SANDBOX_PYTHONPATH;
    # the real bind is read-only, so never leave __pycache__ in it.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.syspath_prepend(str(endpoint.directory / "py"))


async def run_trial(task_dir: Path, trials_dir: Path, agent: str, **environment):
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial

    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=task_dir),
            trials_dir=trials_dir,
            trial_name=f"{task_dir.name}-{agent}",
            agent=AgentConfig(name=agent),
            environment=EnvironmentConfig(
                import_path=PLUGIN, delete=True, **environment
            ),
        )
    )
    return await trial.run()


def reward(result) -> float | None:
    if result.verifier_result is None:
        return None
    return result.verifier_result.rewards.get("reward")
