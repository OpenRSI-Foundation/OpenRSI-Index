"""Real child authority survives cancellation of Harbor's startup waiter."""

import asyncio
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4

import pytest
from harbor.models.task.config import EnvironmentConfig

from rsi_harness.integrations.sandbox_client import SandboxClient
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_contracts import SandboxGrant, SandboxOwner
from rsi_harness.runtime.sandbox_server import SandboxServer
from tests.integration.sandbox_support import (
    assert_no_sandbox_resources,
    create_broker,
    owned_resources,
    remove_exact_containers,
    require_sandbox_authority,
)
from tests.integrations.test_harbor_sandbox import _environment, _offline_policy

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_cancelled_harbor_start_reconciles_real_child_before_stop(tmp_path):
    docker_client, image = require_sandbox_authority()
    run_id = "harbor-cancel-" + uuid4().hex
    store = LeaseStore(tmp_path / "leases")
    broker = create_broker(
        docker_client, store, run_id=run_id, task_id="cancel", image_id=image.id
    )
    profile = broker.grant.profiles[0].model_copy(
        update={
            "tmpfs_mb": (
                ("/workspace", 16),
                ("/tmp", 8),
                ("/dev/shm", 8),
                ("/tests", 4),
                ("/logs", 4),
                ("/solution", 4),
            )
        }
    )
    broker.grant = SandboxGrant.model_validate(
        broker.grant.model_copy(update={"profiles": (profile,)}).model_dump()
    )
    owner = SandboxOwner(run_id=run_id, task_id="cancel", phase="work")
    credential = broker.open_session(owner, time.monotonic() + 25).credential
    broker.start()
    created, release = threading.Event(), threading.Event()
    exact_ids = set()
    with tempfile.TemporaryDirectory(prefix="rsi-harbor-cancel-") as root:
        server = SandboxServer(broker, Path(root) / "s", owner)
        server.start()
        actual = SandboxClient(Path(root) / "s", credential)

        class DelayedClient:
            def __getattr__(self, name):
                return getattr(actual, name)

            def create(self, *args, **kwargs):
                child_id = actual.create(*args, **kwargs)
                exact_ids.update(owned_resources(docker_client, run_id)["containers"])
                created.set()
                assert release.wait(8), "create response barrier timed out"
                return child_id

        env = _environment(
            tmp_path,
            client=DelayedClient(),
            task_config=EnvironmentConfig(
                docker_image=image.id,
                cpus=1,
                memory_mb=128,
                gpus=0,
                workdir="/workspace",
                network_mode="no-network",
            ),
            network_policy=_offline_policy(),
        )
        starting = asyncio.create_task(env.start(force_build=False))
        try:
            assert await asyncio.to_thread(created.wait, 8)
            starting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await starting
            release.set()
            await env.stop(delete=True)
            assert env._child_id is None
            assert owned_resources(docker_client, run_id)["containers"] == ()
        finally:
            release.set()
            if not starting.done():
                starting.cancel()
            await asyncio.gather(starting, return_exceptions=True)
            try:
                await env.stop(delete=True)
            finally:
                server.stop()
                try:
                    broker.close()
                finally:
                    remove_exact_containers(docker_client, exact_ids)
                    assert_no_sandbox_resources(docker_client, store, run_id)
                    docker_client.close()
