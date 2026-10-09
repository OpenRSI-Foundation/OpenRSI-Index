"""The vllm-in-judge sample's Work agent and fixed procedure, in their shapes.

Opt-in (RSI_RUN_VLLM=1; a GPU, public egress for the Hub and for tmux and
uv in the task environments, the sample's Work image, minutes). Run as the
current user against a live broker; only the iptables half is faked, as in
test_harbor_in_judge_sample.py. The root demo through ``rsi-harness run`` is
scripts/operator/vllm_demo.sh.

The broker grants exactly what the sample's shipped operator policy
(sample_tasks/vllm-in-judge/operator-policy.toml) grants its task, so terminus-2's
real requests meet the policy's caps.

1. Work: the sample's Work image, the WORKDIR a fresh volume, public egress
   and a stub ``rsi-submit``, runs work/agent.sh, which downloads the
   checkpoint into the WORKDIR.
2. Judge: the same image, no network, the WORKDIR volume read-only at
   /workspace, one GPU by device request (the first of RSI_TEST_GPUS, or
   RSI_VLLM_GPU), the read-only Judge endpoint as its only bind mount, the
   sample's tests at /tests; it runs /tests/test.sh, which serves the
   checkpoint with vLLM and runs Harbor's terminus-2 against it over the
   sample's tasks through the plugin.

The demo passes if vLLM served the checkpoint on that GPU, the agent made
requests, every one of them metered for its own trial by the sample's
metering proxy, every trial completed without an infrastructure error and
nothing of the run is left; the score does not matter. The Work image is built from
the sample's environment directory with Docker's classic builder (as the
Harness builds it) when it is not cached, and kept as a cache.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import threading
import tomllib
from pathlib import Path

import pytest
from docker.types import DeviceRequest, Mount

from tests.integration.harbor_env_support import (
    live_sandbox as live_sandbox,
)
from tests.integration.test_harbor_in_judge_sample import TARGET, read_file

REPO = Path(__file__).parents[2]
SAMPLE = REPO / "sample_tasks" / "vllm-in-judge"
POLICY = REPO / "sample_tasks" / "vllm-in-judge" / "operator-policy.toml"
PLUGIN = "rsi_sandbox_harbor:ManagedSandboxEnvironment"
IMAGE = "rsi-sample-vllm-in-judge:work"
TASK_IMAGES = ("python:3.13-slim-bookworm", "alexgshaw/regex-log:20251031")
STUB_SUBMIT = b'#!/bin/sh\necho \'{"stub": "submitted"}\'\n'

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RSI_RUN_VLLM") != "1",
        reason="the vLLM demo is opt-in (RSI_RUN_VLLM=1; a GPU, egress, minutes)",
    ),
]


def judge_gpu() -> str:
    """The UUID of the GPU the test may use: RSI_VLLM_GPU, else the first of
    RSI_TEST_GPUS."""
    selector = os.environ.get("RSI_VLLM_GPU") or (
        os.environ.get("RSI_TEST_GPUS", "").split(",")[0].strip()
    )
    if not selector:
        pytest.fail("RSI_RUN_VLLM=1 needs RSI_VLLM_GPU or RSI_TEST_GPUS")
    uuid = subprocess.run(
        ["nvidia-smi", f"--id={selector}", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert uuid.startswith("GPU-"), uuid
    return uuid


def work_image(client):
    """The sample's Work image, built like the Harness's Base image."""
    try:
        return client.images.get(IMAGE)
    except Exception:
        pass
    image, _ = client.images.build(
        path=str(SAMPLE / "environment"),
        tag=IMAGE,
        pull=True,
        rm=True,
        forcerm=True,
        timeout=3600,
    )
    return image


def _tar(entries) -> bytes:
    """``entries``: (name, data or None for a directory, mode), and
    (name, Path) for a directory tree."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data, mode in entries:
            if isinstance(data, Path):
                archive.add(
                    data,
                    arcname=name,
                    filter=lambda info: None if "__pycache__" in info.name else info,
                )
                continue
            entry = tarfile.TarInfo(name)
            entry.mode = mode
            if data is None:
                entry.type = tarfile.DIRTYPE
                archive.addfile(entry)
            else:
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
    return buffer.getvalue()


def run_work(sandbox, volume) -> str:
    """work/agent.sh in Work's shape: the WORKDIR volume, public egress."""
    container = sandbox.client.containers.create(
        IMAGE,
        ["bash", "-c", (SAMPLE / "work" / "agent.sh").read_text()],
        labels={"rsi-harness.run-id": sandbox.run_id, "rsi-harness.test": "vllm"},
        mounts=[Mount(target="/workspace", source=volume.name, type="volume")],
        environment={"HOME": "/root"},
    )
    try:
        assert container.put_archive(
            "/usr/local/bin", _tar([("rsi-submit", STUB_SUBMIT, 0o755)])
        )
        container.start()
        status = container.wait(timeout=1800)
        output = container.logs().decode(errors="replace")
        assert status["StatusCode"] == 0, output[-4000:]
        return output
    finally:
        container.remove(force=True)


def judge_host(sandbox, endpoint, volume, gpu: str, environment):
    """A created Judge-shaped container on the Work image: no network, the
    endpoint its only bind (read-only), the WORKDIR read-only, one GPU."""
    container = sandbox.client.containers.create(
        IMAGE,
        ["bash", "/tests/test.sh"],
        labels={"rsi-harness.run-id": sandbox.run_id, "rsi-harness.test": "vllm"},
        network_mode="none",
        cap_drop=["NET_RAW"],
        working_dir="/workspace",
        shm_size="1g",
        nano_cpus=8 * 10**9,
        mem_limit="24576m",
        device_requests=[
            DeviceRequest(driver="nvidia", device_ids=[gpu], capabilities=[["gpu"]])
        ],
        mounts=[
            Mount(
                target=TARGET,
                source=str(endpoint.directory),
                type="bind",
                read_only=True,
            ),
            Mount(
                target="/workspace", source=volume.name, type="volume", read_only=True
            ),
        ],
        environment={
            **endpoint.environment,
            "HOME": "/root",
            "NVIDIA_VISIBLE_DEVICES": gpu,
            # What the Harness tells a Judge about its GPUs.
            "RSI_HARNESS_EXPECTED_GPU_UUIDS": gpu,
            **environment,
        },
    )
    assert container.put_archive(
        "/",
        _tar(
            [
                ("tests", SAMPLE / "tests", 0),
                ("logs", None, 0o755),
                ("logs/verifier", None, 0o777),
            ]
        ),
    )
    container.reload()
    binds = [
        (item["Type"], item["Destination"], item["RW"])
        for item in container.attrs["Mounts"]
    ]
    assert sorted(binds) == [
        ("bind", TARGET, False),
        ("volume", "/workspace", False),
    ]
    assert not any(
        item.startswith("DOCKER_HOST=") for item in container.attrs["Config"]["Env"]
    )
    return container


def watch_envs(sandbox, stop: threading.Event) -> set[str]:
    """The names of the broker's environment containers seen until
    ``stop`` (filled in by a background thread)."""
    seen: set[str] = set()
    filters = {
        "label": [
            f"rsi-harness.run-id={sandbox.run_id}",
            "rsi-harness.role=sandbox-env",
        ]
    }

    def poll():
        while not stop.wait(1.0):
            try:
                seen.update(
                    item.name
                    for item in sandbox.client.containers.list(
                        all=True, filters=filters
                    )
                )
            except Exception:
                continue

    threading.Thread(target=poll, daemon=True).start()
    return seen


def test_vllm_serves_the_work_checkpoint_to_terminus_2_in_the_judge(live_sandbox):
    gpu = judge_gpu()
    task = (SAMPLE / "task.toml").read_text()
    service = tomllib.loads(task)["environment"]
    sandbox = live_sandbox(
        "vllm",
        *TASK_IMAGES,
        policy=POLICY.read_text(),
        task=task,
        parent=(service["cpus"], service["memory_mb"]),
    )
    client = sandbox.client
    work_image(client)
    labels = {"rsi-harness.run-id": sandbox.run_id, "rsi-harness.test": "vllm"}
    volume = client.volumes.create(labels=labels)

    work = run_work(sandbox, volume)
    assert "RSI-ACCEPTANCE checkpoint-download 0 " in work, work[-4000:]
    assert "RSI-ACCEPTANCE submit-exit 0" in work, work[-4000:]

    endpoint = sandbox.judge(verifier_sec=3600)
    container = judge_host(
        sandbox, endpoint, volume, gpu, {"RSI_VLLM_CONCURRENCY": "2"}
    )
    stop = threading.Event()
    envs = watch_envs(sandbox, stop)
    container.start()
    try:
        status = container.wait(timeout=3300)
    finally:
        stop.set()
    output = container.logs().decode(errors="replace")
    assert status["StatusCode"] == 0, output[-4000:]
    summary = json.loads(read_file(container, "/logs/verifier/vllm-demo-summary.json"))
    reward = json.loads(read_file(container, "/logs/verifier/reward.json"))
    evidence = json.dumps(summary, indent=1)[:6000] + output[-3000:]

    vllm = summary["vllm"]
    assert vllm["healthy"] and vllm["on_judge_gpu"], evidence
    assert vllm["serving_gpus"] == [gpu], evidence
    assert [item["root"] for item in vllm["models"]] == ["/workspace/checkpoint"]
    assert vllm["served_checkpoint"], evidence
    assert summary["completions"]["requests_served"] > 0, evidence
    assert summary["expected_tasks"] == ["regex-log", "rsi/hello-file"]
    assert sorted(item["task"] for item in summary["trials"]) == [
        "regex-log",
        "rsi/hello-file",
    ], evidence
    assert all(item["completed"] for item in summary["trials"]), evidence
    # Each trial in a broker-created environment, its agent asking vLLM.
    assert all(item["environment"] == PLUGIN for item in summary["trials"])
    assert all(item["input_tokens"] > 0 for item in summary["trials"]), evidence
    # Every request metered at the proxy for the trial whose base path it
    # came under, and the score against the tokens.
    usage = [
        json.loads(line)
        for line in read_file(container, "/logs/verifier/usage.jsonl").splitlines()
    ]
    keys = {item["task"].replace("/", "-") for item in summary["trials"]}
    assert usage and {item["trial"] for item in usage} == keys, evidence
    assert all(item["model"] == "rsi-checkpoint" for item in usage), usage[:5]
    for item in summary["trials"]:
        metered = item["metered"]
        assert metered["requests"] > 0 and metered["completion_tokens"] > 0, evidence
    accuracy = summary["accuracy_vs_tokens"]
    assert accuracy["trials"] == 2 and accuracy["solved_fraction"] == summary["reward"]
    assert accuracy["completion_tokens"] == sum(
        item["completion_tokens"] or 0 for item in usage
    )
    # The broker created an environment for each trial.
    assert len(envs) >= len(summary["trials"]), (sorted(envs), evidence)
    assert summary["infra_errors"] == [], evidence
    assert summary["demo_ok"], evidence
    assert reward == {"reward": summary["reward"]}
    assert "harbor run terminus-2: exit 0" in output, output[-4000:]
    # The broker's envs are gone: only the Judge host and the WORKDIR are
    # left, removed with the sandbox.
    containers, volumes, networks = sandbox.labelled()
    assert [item.id for item in containers] == [container.id]
    assert [item.name for item in volumes] == [volume.name]
    assert networks == []
    print(json.dumps(summary, indent=1))
