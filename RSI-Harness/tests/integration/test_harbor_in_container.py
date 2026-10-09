"""Harbor inside a container whose only mount is the read-only endpoint.

This is the Work/Judge shape: the container has no Docker socket, no
``DOCKER_HOST``, no network and no harness package. Harbor and its
dependencies are copied into the image layer before start (never bound),
and the plugin is imported from ``$RSI_SANDBOX_PYTHONPATH``. Everything the
Trials need happens in the host broker behind the endpoint socket.
"""

from __future__ import annotations

import io
import json
import os
import sysconfig
import tarfile
import tempfile
from pathlib import Path

import pytest
from docker.types import Mount

from tests.integration.harbor_env_support import (
    FIXTURES,
    PLUGIN,
)
from tests.integration.harbor_env_support import (
    live_sandbox as live_sandbox,
)

pytestmark = pytest.mark.integration

HOST_IMAGE = "python:3.13-slim-bookworm"
TARGET = "/run/rsi-harness/sandbox"
SCRIPT = f"""
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.trial.trial import Trial


async def run(agent):
    trial = await Trial.create(
        TrialConfig(
            task=TaskConfig(path=Path("/opt/m7/task")),
            trials_dir=Path("/tmp/trials"),
            trial_name="in-container-" + agent,
            agent=AgentConfig(name=agent),
            environment=EnvironmentConfig(import_path="{PLUGIN}", delete=True),
        )
    )
    result = await trial.run()
    rewards = None if result.verifier_result is None else result.verifier_result.rewards
    return {{"rewards": rewards, "error": str(result.exception_info or "")}}


results = {{agent: asyncio.run(run(agent)) for agent in ("oracle", "nop")}}
results["harness_importable"] = importlib.util.find_spec("rsi_harness") is not None
results["plugin"] = sys.modules["rsi_sandbox_harbor"].__file__
print("RESULT " + json.dumps(results))
"""


def _site_packages_tar(target) -> None:
    """The test venv's packages, minus the harness itself (editable)."""
    root = Path(sysconfig.get_paths()["purelib"])
    with tarfile.open(fileobj=target, mode="w") as archive:
        for child in sorted(root.iterdir()):
            name = child.name
            if name.endswith(".pth") or name.startswith(("__editable__", "rsi_")):
                continue
            archive.add(
                child,
                arcname=f"site/{name}",
                filter=lambda info: None if "__pycache__" in info.name else info,
            )


def _task_tar() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(FIXTURES / "harbor-env-busybox", arcname="m7/task")
        data = SCRIPT.encode()
        entry = tarfile.TarInfo("m7/run_trials.py")
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


def test_harbor_in_a_container_with_only_the_endpoint_mounted(live_sandbox):
    sandbox = live_sandbox("in-container", HOST_IMAGE, "busybox:1.37.0")
    endpoint = sandbox.work()
    client = sandbox.client
    container = client.containers.create(
        HOST_IMAGE,
        ["python3", "/opt/m7/run_trials.py"],
        # Labelled with the run: the sandbox's close removes it by label.
        labels={"rsi-harness.run-id": sandbox.run_id, "rsi-harness.test": "m7"},
        runtime="runc",
        network_mode="none",
        mounts=[
            Mount(
                target=TARGET,
                source=str(endpoint.directory),
                type="bind",
                read_only=True,
            )
        ],
        environment={
            **endpoint.environment,
            "PYTHONPATH": "/opt/site:" + endpoint.environment["RSI_SANDBOX_PYTHONPATH"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": "/root",
        },
    )
    with tempfile.TemporaryFile() as spool:
        _site_packages_tar(spool)
        spool.seek(0)
        assert container.put_archive("/opt", spool)
    assert container.put_archive("/opt", _task_tar())

    container.reload()
    mounts = container.attrs["Mounts"]
    assert [(item["Type"], item["Destination"], item["RW"]) for item in mounts] == [
        ("bind", TARGET, False)
    ]
    assert mounts[0]["Source"] == str(endpoint.directory)
    environment = container.attrs["Config"]["Env"]
    assert not any(item.startswith("DOCKER_HOST=") for item in environment)
    assert not any("docker.sock" in str(item) for item in mounts)

    container.start()
    status = container.wait(timeout=600)
    output = container.logs().decode(errors="replace")
    assert status["StatusCode"] == 0, output[-4000:]
    [line] = [item for item in output.splitlines() if item.startswith("RESULT ")]
    results = json.loads(line.removeprefix("RESULT "))
    assert results["oracle"] == {"rewards": {"reward": 1.0}, "error": ""}, results
    assert results["nop"] == {"rewards": {"reward": 0.0}, "error": ""}, results
    assert results["harness_importable"] is False
    assert results["plugin"] == f"{TARGET}/py/rsi_sandbox_harbor.py"
    # The broker's env containers are gone; only the Harbor host remains.
    containers, volumes, networks = sandbox.labelled()
    assert [item.id for item in containers] == [container.id]
    assert (volumes, networks) == ([], [])
    # The read-only bind kept the endpoint exactly as injected.
    assert sorted(os.listdir(endpoint.directory / "py")) == [
        "rsi_sandbox_client.py",
        "rsi_sandbox_compose.py",
        "rsi_sandbox_harbor.py",
    ]
