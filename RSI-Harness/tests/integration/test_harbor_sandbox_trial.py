"""Real Docker-backed Harbor Trial acceptance for the managed adapter."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial

from rsi_harness.integrations import harbor_sandbox
from rsi_harness.integrations.harbor_sandbox import preflight_managed_trial
from rsi_harness.runtime.recovery import LeaseStore, ResourceLease
from rsi_harness.runtime.sandbox_budget import (
    SandboxAdmissionPool,
    SandboxJournal,
)
from rsi_harness.runtime.sandbox_contracts import (
    SandboxGrant,
    SandboxLimits,
    SandboxOwner,
    SandboxPhaseGrant,
    SandboxProfile,
)
from tests.integration.sandbox_support import require_sandbox_authority


def _authority(store: LeaseStore, run_id: str, task_id: str):
    current = ResourceLease(
        run_id=run_id,
        task_id=task_id,
        coordinator_pid=1,
        coordinator_started_at=1.0,
        phase="preparing",
    )
    store.write(current)
    lock = threading.RLock()

    def mutate(transform):
        nonlocal current
        with lock:
            updated = transform(current)
            if updated is not current:
                store.write(updated)
                current = updated
            return current

    return mutate


def _grant(image_id: str) -> SandboxGrant:
    profile = SandboxProfile(
        name="offline",
        image=image_id,
        cpus=1,
        memory_mb=256,
        pids=64,
        max_lifetime_sec=120.0,
        workdir="/workspace",
        tmpfs_mb=(
            ("/workspace", 16),
            ("/tests", 4),
            ("/solution", 4),
            ("/logs", 8),
            ("/tmp", 8),
            ("/dev/shm", 8),
        ),
    )
    limits = SandboxLimits(
        max_live=1,
        max_created=1,
        max_operations=32,
        max_cpus=1,
        max_memory_mb=256,
        max_lifetime_sec=120.0,
        max_upload_bytes=8 * 1024**2,
        max_download_bytes=8 * 1024**2,
        max_log_bytes=1024**2,
    )
    phase = SandboxPhaseGrant(profiles=("offline",), limits=limits)
    return SandboxGrant(
        version=1,
        profiles=(profile,),
        work=phase,
        judge=None,
        run_limits=limits,
        reserved_cpus=1,
        reserved_memory_mb=2307,
        pool_cpus=2,
        pool_memory_mb=8192,
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "verifier-timeout", "bounded-cleanup"])
async def test_real_offline_oracle_trial_uses_managed_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    from rsi_harness.runtime.sandbox import SandboxBroker
    from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend
    from rsi_harness.runtime.sandbox_server import SandboxServer

    docker_client, image = require_sandbox_authority()
    image_id = image.id
    verifier_timeout = scenario != "success"
    if scenario == "bounded-cleanup":
        monkeypatch.setattr(harbor_sandbox, "_LIFECYCLE_WAIT_SEC", 0.05)

    fixture = (
        Path(__file__).parents[1] / "fixtures" / "tasks" / "minimal-harbor-sandbox"
    )
    task_dir = tmp_path / "task"
    shutil.copytree(fixture, task_dir)
    config_path = task_dir / "task.toml"
    config_path.write_text(
        config_path.read_text().replace("__APPROVED_IMAGE_ID__", image_id)
    )
    if verifier_timeout:
        config_path.write_text(
            config_path.read_text().replace(
                "[verifier]\ntimeout_sec = 30", "[verifier]\ntimeout_sec = 0.4"
            )
        )
        (task_dir / "tests/test.sh").write_text(
            "#!/bin/sh\nset -eu\nsleep 3\nprintf '1.0\\n' > /logs/verifier/reward.txt\n"
        )
    assert "__APPROVED_IMAGE_ID__" not in config_path.read_text()

    suffix = uuid.uuid4().hex[:12]
    run_id, task_id = f"trial-{suffix}", f"task-{suffix}"
    store = LeaseStore(tmp_path / "leases")
    mutate = _authority(store, run_id, task_id)
    grant = _grant(image_id)
    pool = SandboxAdmissionPool(store)
    pool.reserve_run(run_id, grant, mutate)
    journal = SandboxJournal(mutate)
    backend = SandboxDockerBackend(docker_client)
    broker = SandboxBroker(grant, backend, journal)
    owner = SandboxOwner(run_id=run_id, task_id=task_id, phase="work")
    credentials = broker.open_session(owner, time.monotonic() + 120)
    broker.start()

    try:
        with tempfile.TemporaryDirectory(prefix="rsi-harbor-") as socket_root:
            socket_path = Path(socket_root) / "sandbox.sock"
            server = SandboxServer(broker, socket_path, owner)
            server.start()
            monkeypatch.setenv("RSI_SANDBOX_SOCKET", str(socket_path))
            monkeypatch.setenv("RSI_SANDBOX_TOKEN", credentials.credential)
            try:
                trial = await Trial.create(
                    TrialConfig(
                        task=TaskConfig(path=task_dir),
                        trials_dir=tmp_path / "trials",
                        trial_name="managed-sandbox-smoke",
                        agent=AgentConfig(name="oracle"),
                        environment=EnvironmentConfig(
                            import_path=(
                                "rsi_harness.integrations.harbor_sandbox:"
                                "ManagedSandboxEnvironment"
                            ),
                            kwargs={"profile": "offline", "lifetime_sec": 30},
                            force_build=False,
                            delete=True,
                        ),
                    )
                )
                preflight_managed_trial(trial)
                result = await trial.run()
                if verifier_timeout:
                    assert result.exception_info is not None
                    assert (
                        result.exception_info.exception_type == "VerifierTimeoutError"
                    )
                    assert trial._is_agent_environment_stopped
                else:
                    assert result.exception_info is None
                    assert result.verifier_result is not None
                    assert result.verifier_result.rewards == {"reward": 1.0}

                if scenario == "bounded-cleanup":
                    # Harbor has stopped waiting and will not retry stop(). The
                    # adapter's retained cleanup must finish by itself.
                    assert trial.agent_environment._child_id is not None
                    await asyncio.wait_for(
                        asyncio.shield(trial.agent_environment._cleanup_task), 6
                    )
                assert trial.agent_environment._child_id is None
                children = journal.snapshot()
                assert len(children) == 1
                assert children[0].state == "removed"
                assert (
                    docker_client.containers.list(
                        all=True,
                        filters={"label": f"rsi-harness.run-id={run_id}"},
                    )
                    == []
                )
            finally:
                server.stop()
    finally:
        try:
            broker.close()
        finally:
            try:
                pool.release_run(run_id, mutate)
            finally:
                docker_client.close()

    assert store.read(run_id).sandbox_reservation is None
    assert all(child.state == "removed" for child in store.read(run_id).sandboxes)
