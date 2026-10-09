"""Stock Harbor Trials through the injected v2 plugin on real Docker.

Harbor runs in this process exactly as it would in Work or Judge: it imports
``rsi_sandbox_harbor`` from the endpoint's ``py/`` directory and reaches the
broker only through the phase socket. The oracle must score 1 and the nop
agent 0, and nothing labelled with the run may remain afterwards.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import shutil
import subprocess
import tarfile
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from tests.integration.harbor_env_support import (
    FIXTURES,
    reward,
    run_trial,
    use_endpoint,
)
from tests.integration.harbor_env_support import (
    live_sandbox as live_sandbox,
)

pytestmark = pytest.mark.integration

TB2 = Path(os.environ.get("RSI_TB2_DIR", "/mnt/y1/temp/terminal_bench_2"))


def moment(text: str) -> datetime:
    """A Docker RFC 3339 time (nanoseconds, any offset) as a datetime."""
    return datetime.fromisoformat(re.sub(r"(\.\d{6})\d+", r"\1", text))


def envs_made(sandbox):
    """Env rules the firewall (the fake, or root mode's recording real one)
    saw installed during the run."""
    return [
        rule
        for event, rule in sandbox.firewall.events
        if event == "install" and "-sbx-" in rule
    ]


@pytest.mark.asyncio
async def test_busybox_prebuilt_trial_scores_oracle_one_and_nop_zero(
    live_sandbox, monkeypatch, tmp_path
):
    sandbox = live_sandbox("busybox", "busybox:1.37.0")
    use_endpoint(monkeypatch, sandbox.work())
    task = FIXTURES / "harbor-env-busybox"

    oracle = await run_trial(task, tmp_path / "trials", "oracle")
    assert oracle.exception_info is None, oracle.exception_info
    assert reward(oracle) == 1.0
    nop = await run_trial(task, tmp_path / "trials", "nop")
    assert nop.exception_info is None, nop.exception_info
    assert reward(nop) == 0.0
    # busybox has no bash, so reward 1 means the plugin's sh fallback ran
    # the oracle; its log was downloaded from the env.
    oracle_log = tmp_path / "trials" / "harbor-env-busybox-oracle" / "agent"
    assert (oracle_log / "oracle.txt").exists()
    assert len(envs_made(sandbox)) == 2
    assert sandbox.labelled() == ([], [], [])


@pytest.mark.asyncio
async def test_a_judge_past_the_work_deadline_with_a_900_s_verifier_scores(
    live_sandbox, monkeypatch, tmp_path
):
    # Spec A8: the round's envs live by its own verifier timeout, which here
    # equals max_wait_timeout_sec (900 s), not by the seconds Work had left.
    sandbox = live_sandbox("late-judge", "busybox:1.37.0")
    use_endpoint(monkeypatch, sandbox.judge(work_sec=2, verifier_sec=900))
    await asyncio.sleep(2.5)  # Work's deadline passes
    task = FIXTURES / "harbor-env-busybox"

    oracle = await run_trial(task, tmp_path / "trials", "oracle")

    assert oracle.exception_info is None, oracle.exception_info
    assert reward(oracle) == 1.0
    assert len(envs_made(sandbox)) == 1
    assert sandbox.labelled() == ([], [], [])


@pytest.mark.asyncio
async def test_compose_sidecar_trial_scores_oracle_one_and_nop_zero(
    live_sandbox, monkeypatch, tmp_path
):
    sandbox = live_sandbox("sidecar", "python:3.13-slim-bookworm", "redis:7-alpine")
    use_endpoint(monkeypatch, sandbox.work())
    task = FIXTURES / "harbor-compose-sidecar"
    seen, seeds, starts, watchers = [], [], [], []
    original = sandbox.broker.envs.env_start

    def watch(*args, **kwargs):
        # Inspect the created env: exact aliases, the redis VOLUME as a
        # labelled volume, no host binds, and the seed file already copied
        # into the created (never started) main container.
        containers = sandbox.client.containers.list(
            all=True,
            filters={"label": f"rsi-harness.run-id={sandbox.run_id}"},
        )
        services = {
            item.labels["rsi-harness.sandbox-service"]: item for item in containers
        }
        main = services["main"]
        assert main.attrs["State"]["Status"] == "created"
        stream, _ = main.get_archive("/seed/value.txt")
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as archive:
            seeds.append(archive.extractfile("value.txt").read())
        result = original(*args, **kwargs)
        [network] = sandbox.client.networks.list(
            filters={"label": f"rsi-harness.run-id={sandbox.run_id}"}
        )
        seen.append(
            ({name: item.attrs for name, item in services.items()}, network.attrs)
        )
        watcher = threading.Thread(target=started, args=(services,), daemon=True)
        watcher.start()
        watchers.append(watcher)
        return result

    def started(services):
        # service_healthy: main starts only after a passing kv check.
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            services["main"].reload()
            state = services["main"].attrs["State"]
            if not state["StartedAt"].startswith("0001-"):
                services["kv"].reload()
                starts.append((state["StartedAt"], services["kv"].attrs["State"]))
                return
            time.sleep(0.05)

    monkeypatch.setattr(sandbox.broker.envs, "env_start", watch)

    oracle = await run_trial(task, tmp_path / "trials", "oracle")
    assert oracle.exception_info is None, oracle.exception_info
    assert reward(oracle) == 1.0
    nop = await run_trial(task, tmp_path / "trials", "nop")
    assert nop.exception_info is None, nop.exception_info
    assert reward(nop) == 0.0

    for watcher in watchers:
        watcher.join(timeout=5)
    seed = (task / "environment" / "seed" / "value.txt").read_bytes()
    assert seeds == [seed, seed]
    assert len(starts) == 2
    for main_started, kv_state in starts:
        passed = [
            moment(item["End"])
            for item in kv_state["Health"]["Log"]
            if item["ExitCode"] == 0
        ]
        assert passed and min(passed) <= moment(main_started)
    attrs, network = seen[0]
    assert set(attrs) == {"main", "kv"}
    for service in attrs.values():
        host = service["HostConfig"]
        assert host["Binds"] is None and host["Runtime"] == "runc"
        assert {mount["Type"] for mount in service["Mounts"]} <= {"volume"}
        assert not any("docker.sock" in str(mount) for mount in service["Mounts"])
        assert not any(
            item.startswith("DOCKER_HOST=") for item in service["Config"]["Env"]
        )
    endpoints = attrs["kv"]["NetworkSettings"]["Networks"]
    assert [item["Aliases"] for item in endpoints.values()] == [["kv", "kvstore"]]
    assert [mount["Destination"] for mount in attrs["kv"]["Mounts"]] == ["/data"]
    # no-network: an internal bridge, whose rule rejects all egress.
    assert network["Internal"] is True and list(endpoints) == [network["Name"]]
    assert sandbox.labelled() == ([], [], [])


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RSI_RUN_TB2") != "1" or not (TB2 / "fix-git").is_dir(),
    reason="TB2 fix-git acceptance is opt-in (RSI_RUN_TB2=1; public egress)",
)
async def test_tb2_fix_git_scores_oracle_one_and_nop_zero(
    live_sandbox, monkeypatch, tmp_path
):
    """One Terminal-Bench 2 task with its cached prebuilt image, public
    egress through the (fake-firewalled) env bridge: test.sh installs curl
    and uv from the internet, as in the real benchmark."""
    sandbox = live_sandbox("tb2", "alexgshaw/fix-git:20251031")
    use_endpoint(monkeypatch, sandbox.work())
    task = tmp_path / "fix-git"
    shutil.copytree(TB2 / "fix-git", task)

    oracle = await run_trial(task, tmp_path / "trials", "oracle")
    assert oracle.exception_info is None, oracle.exception_info
    assert reward(oracle) == 1.0
    nop = await run_trial(task, tmp_path / "trials", "nop")
    assert nop.exception_info is None, nop.exception_info
    assert reward(nop) == 0.0
    assert sandbox.labelled() == ([], [], [])


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RSI_RUN_TB2") != "1" or not (TB2 / "fix-git").is_dir(),
    reason="TB2 fix-git acceptance is opt-in (RSI_RUN_TB2=1; public egress)",
)
async def test_tb2_fix_git_built_from_its_dockerfile_scores_oracle_one_and_nop_zero(
    live_sandbox, monkeypatch, tmp_path
):
    """Spec A4 on this host: fix-git without ``docker_image``, so the plugin
    builds environment/Dockerfile through the broker (its ``apt-get install
    git`` needs the public build network), then oracle 1 and nop 0, in a
    Judge round (its own builder, removed by close_judge). The real Judge
    container and its image are M9's A4.

    setup.sh clones https://github.com/TheMikeMerrill/personal-site.git,
    which GitHub answers 404 since (checked 2026-09-30), so the task can no
    longer build anywhere as published. Its history is vendored from the
    cached prebuilt image into the build context, and one Dockerfile line
    before setup.sh points that URL at the vendored copy; setup.sh itself,
    the tests and the oracle are unchanged.
    """
    sandbox = live_sandbox("tb2-build", "alexgshaw/fix-git:20251031", build=True)
    use_endpoint(monkeypatch, sandbox.judge())
    task = tmp_path / "fix-git"
    shutil.copytree(TB2 / "fix-git", task)
    config = task / "task.toml"
    text = config.read_text()
    assert 'docker_image = "alexgshaw/fix-git:20251031"' in text
    config.write_text(
        "\n".join(
            line for line in text.splitlines() if not line.startswith("docker_image")
        )
        + "\n"
    )
    _vendor_personal_site(sandbox, task / "environment")
    builds = []
    original = sandbox.broker.envs.image_build

    def watch(*args, **kwargs):
        builds.append(kwargs["network"])
        return original(*args, **kwargs)

    monkeypatch.setattr(sandbox.broker.envs, "image_build", watch)

    oracle = await run_trial(task, tmp_path / "trials", "oracle")
    assert oracle.exception_info is None, oracle.exception_info
    assert reward(oracle) == 1.0
    nop = await run_trial(task, tmp_path / "trials", "nop")
    assert nop.exception_info is None, nop.exception_info
    assert reward(nop) == 0.0
    # Each trial built its own image through the broker, on the public
    # build network, and released it when the trial stopped.
    assert builds == ["public", "public"]
    assert sandbox.built_images() == []
    # The round's builder is still up (its cache spans the round) until
    # close_judge removes every object of the round.
    containers, volumes, networks, images = sandbox.round_objects()
    assert [item.labels["rsi-harness.role"] for item in containers] == [
        "sandbox-builder"
    ]
    assert images == []
    sandbox.lifecycle.close_judge()
    assert sandbox.round_objects() == ([], [], [], [])
    assert not sandbox.broker.recovery_required


PERSONAL_SITE = "https://github.com/TheMikeMerrill/personal-site.git"


def _vendor_personal_site(sandbox, environment: Path) -> None:
    """A bare clone of the prebuilt image's /app/personal-site history in
    the build context, and a git URL rewrite to it before setup.sh."""
    container = sandbox.client.containers.create(
        "alexgshaw/fix-git:20251031",
        ["true"],
        labels={"rsi-harness.run-id": sandbox.run_id},
        runtime="runc",
        network_mode="none",
    )
    try:
        stream, _ = container.get_archive("/app/personal-site")
        extracted = environment.parent / "extracted"
        with tarfile.open(fileobj=io.BytesIO(b"".join(stream))) as archive:
            archive.extractall(extracted, filter="data")
    finally:
        container.remove(v=True, force=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--bare",
            "--no-local",
            str(extracted / "personal-site"),
            str(environment / "resources" / "personal-site.git"),
        ],
        check=True,
        capture_output=True,
    )
    shutil.rmtree(extracted)
    dockerfile = environment / "Dockerfile"
    text = dockerfile.read_text()
    rewrite = (
        "RUN git config --system "
        f'url."file:///app/resources/personal-site.git".insteadOf {PERSONAL_SITE}\n'
    )
    assert "RUN bash /app/setup.sh" in text
    dockerfile.write_text(
        text.replace("RUN bash /app/setup.sh", rewrite + "\nRUN bash /app/setup.sh")
    )
