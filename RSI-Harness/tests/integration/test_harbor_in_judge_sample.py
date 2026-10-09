"""The harbor-in-judge sample's fixed procedure in the Judge's shape.

``tests/test.sh`` of sample_tasks/harbor-in-judge runs in a container whose
only mount is the read-only Judge endpoint: no Docker socket, no
``DOCKER_HOST``, no network, no harness package. Harbor and its
dependencies are copied into the image layer's site-packages before start
(as the sample's Work image installs them), the sample's tests directory is
copied to /tests, and the procedure writes the reward itself. The broker,
lifecycle and server are the production ones; only the iptables half is
faked (real egress filtering is the operator's root check).
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
    live_sandbox as live_sandbox,
)
from tests.integration.test_harbor_env_trial import TB2

pytestmark = pytest.mark.integration

SAMPLE = Path(__file__).parents[2] / "sample_tasks" / "harbor-in-judge"
HOST_IMAGE = "python:3.13-slim-bookworm"
TARGET = "/run/rsi-harness/sandbox"
SITE = "/usr/local/lib/python3.13/site-packages"
HARBOR_SCRIPT = b"""#!/usr/local/bin/python3
import sys

from harbor.cli.main import app

sys.exit(app())
"""
TB2_OPT_IN = pytest.mark.skipif(
    os.environ.get("RSI_RUN_TB2") != "1" or not (TB2 / "fix-git").is_dir(),
    reason="TB2 suites are opt-in (RSI_RUN_TB2=1; public egress, minutes)",
)


def _site_packages_tar(target) -> None:
    """The test venv's packages, minus the harness itself (editable), laid
    out as the Work image's ``pip install harbor==0.21.0`` leaves them."""
    root = Path(sysconfig.get_paths()["purelib"])
    with tarfile.open(fileobj=target, mode="w") as archive:
        for child in sorted(root.iterdir()):
            name = child.name
            if name.endswith(".pth") or name.startswith(("__editable__", "rsi_")):
                continue
            archive.add(
                child,
                arcname=f"site-packages/{name}",
                filter=lambda info: None if "__pycache__" in info.name else info,
            )


def _file(archive, name: str, data: bytes, mode: int = 0o644) -> None:
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    entry.mode = mode
    archive.addfile(entry, io.BytesIO(data))


def _root_tar(files, tests: Path) -> bytes:
    """/tests (a sample's tests directory), the harbor console script,
    the Judge's /logs/verifier and ``files`` ({path: (data, mode)})."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        archive.add(tests, arcname="tests")
        _file(archive, "usr/local/bin/harbor", HARBOR_SCRIPT, 0o755)
        logs = tarfile.TarInfo("logs/verifier")
        logs.type = tarfile.DIRTYPE
        logs.mode = 0o777
        archive.addfile(logs)
        for path, (data, mode) in files.items():
            _file(archive, path.lstrip("/"), data, mode)
    return buffer.getvalue()


def read_file(container, path: str) -> bytes:
    stream, _ = container.get_archive(path)
    with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as archive:
        return archive.extractfile(Path(path).name).read()


def harbor_host(
    sandbox, endpoint, command, environment, files=None, tests=SAMPLE / "tests"
):
    """A created (not started) Work/Judge-shaped container: the endpoint is
    its only mount (read-only), Harbor is in its site-packages, ``tests``
    at /tests, no network, no Docker socket; labelled with the run so the
    sandbox's close removes it."""
    container = sandbox.client.containers.create(
        HOST_IMAGE,
        command,
        labels={"rsi-harness.run-id": sandbox.run_id, "rsi-harness.test": "m9"},
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
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": "/root",
            **environment,
        },
    )
    with tempfile.TemporaryFile() as spool:
        _site_packages_tar(spool)
        spool.seek(0)
        assert container.put_archive(str(Path(SITE).parent), spool)
    assert container.put_archive("/", _root_tar(files or {}, tests))

    container.reload()
    mounts = container.attrs["Mounts"]
    assert [(item["Type"], item["Destination"], item["RW"]) for item in mounts] == [
        ("bind", TARGET, False)
    ]
    assert not any(
        item.startswith("DOCKER_HOST=") for item in container.attrs["Config"]["Env"]
    )
    return container


def assert_only_host_left(sandbox, container) -> None:
    """The broker's env containers are gone; only the Harbor host remains
    (and a round's builder, which close_judge removes with the round)."""
    containers, volumes, networks = (
        [item for item in listed if not _builder(item)] for listed in sandbox.labelled()
    )
    assert [item.id for item in containers] == [container.id]
    assert (volumes, networks) == ([], [])


def _builder(item) -> bool:
    labels = item.labels if hasattr(item, "labels") else item.attrs["Labels"]
    return (labels or {}).get("rsi-harness.role", "").startswith("sandbox-builder")


def run_fixed_procedure(sandbox, **selection):
    """Run /tests/test.sh against the Judge endpoint; return the reward,
    the summary and the procedure's output."""
    endpoint = sandbox.judge()
    container = harbor_host(
        sandbox,
        endpoint,
        ["bash", "/tests/test.sh"],
        {"RSI_HARBOR_CONCURRENCY": "1", **selection},
    )
    container.start()
    status = container.wait(timeout=3000)
    output = container.logs().decode(errors="replace")
    assert status["StatusCode"] == 0, output[-4000:]
    reward = json.loads(read_file(container, "/logs/verifier/reward.json"))
    summary = json.loads(read_file(container, "/logs/verifier/harbor-summary.json"))
    assert_only_host_left(sandbox, container)
    return reward, summary, output


def assert_suite(summary, suite, trials):
    report = summary["suites"][suite]
    for agent in ("oracle", "nop"):
        assert report[agent]["ok"], report
        assert report[agent]["trials"] == trials, report


def test_the_fixed_procedure_scores_the_compose_sidecar_suite(live_sandbox):
    sandbox = live_sandbox("sample", HOST_IMAGE, "redis:7-alpine")

    reward, summary, output = run_fixed_procedure(sandbox, RSI_HARBOR_SUITES="compose")

    assert reward == {"reward": 1.0}, (summary, output[-4000:])
    assert_suite(summary, "compose", 1)
    assert "harbor run compose oracle: exit 0" in output
    assert "harbor run compose nop: exit 0" in output


@TB2_OPT_IN
def test_the_fixed_procedure_scores_tb2_fix_git(live_sandbox):
    sandbox = live_sandbox("sample-tb2", HOST_IMAGE, "alexgshaw/fix-git:20251031")

    reward, summary, output = run_fixed_procedure(
        sandbox, RSI_HARBOR_SUITES="tb2", RSI_HARBOR_TB2_TASKS="fix-git"
    )

    assert reward == {"reward": 1.0}, (summary, output[-4000:])
    assert_suite(summary, "tb2", 1)
    assert summary["suites"]["tb2"]["tasks"] == ["fix-git"]


@TB2_OPT_IN
def test_the_fixed_procedure_builds_fix_git_from_its_dockerfile(
    live_sandbox, monkeypatch
):
    """Spec A4's suite: the plugin builds fix-git-build through the Judge
    round's builder (public build network), then oracle 1 and nop 0."""
    sandbox = live_sandbox("sample-build", HOST_IMAGE, build=True)
    builds = []
    original = sandbox.broker.envs.image_build

    def watch(*args, **kwargs):
        builds.append(kwargs["network"])
        return original(*args, **kwargs)

    monkeypatch.setattr(sandbox.broker.envs, "image_build", watch)

    reward, summary, output = run_fixed_procedure(sandbox, RSI_HARBOR_SUITES="build")

    assert reward == {"reward": 1.0}, (summary, output[-4000:])
    assert_suite(summary, "build", 1)
    assert builds == ["public", "public"]
    assert sandbox.built_images() == []
