"""Real non-root parent access; no firewall-dependent parent run is implied."""

import json
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from rsi_harness.runtime.sandbox_lifecycle import SandboxLifecycle
from tests.integration.sandbox_support import (
    create_broker,
    lease_root,
    require_sandbox_authority,
)


@pytest.mark.integration
def test_real_nonroot_parent_can_authenticate_but_has_no_host_docker_access():
    client, image = require_sandbox_authority()
    identity = "uid-test-" + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="rsi-uid-") as directory:
        root = Path(directory)
        broker = create_broker(
            client,
            lease_root(root),
            run_id=identity,
            task_id="uid-test",
            image_id=image.id,
        )
        lifecycle = SandboxLifecycle()
        lifecycle.configure(broker, root / "sb", identity, "uid-test")
        try:
            endpoint = lifecycle.prepare_work()
            lifecycle.activate_work(time.monotonic() + 20)
            parent = client.containers.create(
                image.id,
                name=identity,
                user="1000:1000",
                network_mode="none",
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                mem_limit="128m",
                pids_limit=16,
                nano_cpus=1000000000,
                environment=endpoint.environment,
                volumes={
                    str(endpoint.directory): {
                        "bind": str(endpoint.mount.target),
                        "mode": "ro",
                    }
                },
                command=[
                    "/run/rsi-harness/sandbox/rsi-sandbox",
                    "capabilities",
                    "--json",
                ],
                labels={"rsi-harness.run-id": identity},
            )
            parent.start()
            assert parent.wait(timeout=10)["StatusCode"] == 0
            result = json.loads(parent.logs())
            assert result["version"] == 1
            assert result["profiles"][0]["name"] == "offline"
            parent.reload()
            assert all(
                m["Destination"] != "/var/run/docker.sock"
                for m in parent.attrs["Mounts"]
            )
        finally:
            lifecycle.close()
            # By name as well: a create that landed but did not return a handle
            # would otherwise leave the container behind.
            for found in client.containers.list(all=True, filters={"name": identity}):
                found.remove(force=True, v=True)
            client.close()
