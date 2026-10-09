"""The swebench-in-judge sample's fixed procedure in the Judge's shape.

Opt-in (RSI_RUN_SWEBENCH=1; the images of
sample_tasks/swebench-in-judge/images.manifest pre-pulled, about 1.5 GB, with
scripts/operator/prepull_images.sh; about 15 s a run). ``tests/test.sh`` of
sample_tasks/swebench-in-judge runs in a container shaped as in
test_harbor_in_judge_sample.py (the read-only Judge endpoint its only
mount, no network, Harbor in its site-packages), against a live broker
granted exactly what the shipped operator policy grants the sample: envs
with network none only, pulls that may bind cached images but download
nothing (max_pull_mb = 1), no builds. Only the iptables half is faked.

With RSI_STATIC_TMUX (see test_sandbox_offline_tmux.py), the procedure
runs again with the operator's static tmux offered, which the plugin copies
into every env (the SWE-bench images have none), and a tmux session runs
offline in such an env.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from pathlib import Path

import pytest

from tests.integration.harbor_env_support import (
    live_sandbox as live_sandbox,
)
from tests.integration.test_harbor_in_judge_sample import (
    HOST_IMAGE,
    assert_only_host_left,
    harbor_host,
    read_file,
)
from tests.integration.test_sandbox_offline_tmux import (
    SESSION,
    TMUX,
    client_of,
    install,
    shell,
)
from tests.integration.test_sandbox_offline_tmux import (
    static_tmux as static_tmux,
)

REPO = Path(__file__).parents[2]
SAMPLE = REPO / "sample_tasks" / "swebench-in-judge"
POLICY = REPO / "sample_tasks" / "swebench-in-judge" / "operator-policy.toml"
MANIFEST = REPO / "sample_tasks" / "swebench-in-judge" / "images.manifest"
SUITE = SAMPLE / "tests" / "swebench-verified"
TASKS = sorted(path.name for path in SUITE.iterdir())
IMAGES = [line for line in MANIFEST.read_text().splitlines() if line and line[0] != "#"]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RSI_RUN_SWEBENCH") != "1",
        reason="the SWE-bench sample is opt-in (RSI_RUN_SWEBENCH=1; "
        "its images pre-pulled)",
    ),
]


def sample_sandbox(live_sandbox, name: str, policy: str):
    """A live broker granting the sample's task what ``policy`` grants."""
    task = (SAMPLE / "task.toml").read_text()
    service = tomllib.loads(task)["environment"]
    return live_sandbox(
        name,
        HOST_IMAGE,
        *IMAGES,
        policy=policy,
        task=task,
        parent=(service["cpus"], service["memory_mb"]),
    )


def with_tmux(policy: str, static_tmux: Path, tmp_path: Path) -> str:
    """``policy`` offering a copy of the operator's static tmux."""
    binary = tmp_path / "tmux"
    shutil.copyfile(static_tmux, binary)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    offered = policy.replace(
        "# [environments.host.tmux]\n"
        '# path = "/opt/rsi/tools/tmux-3.5a"\n'
        '# sha256 = "<the sha256 build_static_tmux.sh printed>"\n',
        f'[environments.host.tmux]\npath = "{binary}"\nsha256 = "{digest}"\n',
    )
    assert "[environments.host.tmux]\npath" in offered
    return offered


def record(monkeypatch, envs, name: str, seen: list, pick) -> None:
    original = getattr(envs, name)

    def watch(*args, **kwargs):
        seen.append(pick(*args, **kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(envs, name, watch)


@pytest.mark.parametrize("tmux", [False, True], ids=["plain", "tmux-offered"])
def test_the_fixed_procedure_scores_swebench_verified_offline(
    live_sandbox, monkeypatch, request, tmp_path, tmux
):
    """With tmux offered, the plugin also copies it into every env, whose
    image has none, before Harbor's agent setup (as for terminus-2)."""
    policy = POLICY.read_text()
    if tmux:
        policy = with_tmux(policy, request.getfixturevalue("static_tmux"), tmp_path)
    sandbox = sample_sandbox(live_sandbox, "swebench", policy)
    envs = sandbox.broker.envs
    networks, pulls, installs = [], [], []
    record(
        monkeypatch, envs, "env_create", networks, lambda c, spec, r: spec["network"]
    )
    record(monkeypatch, envs, "image_pull", pulls, lambda c, ref, policy, r: ref)
    record(monkeypatch, envs, "tool_install", installs, lambda c, e, s, t: (s, t))

    endpoint = sandbox.judge()
    container = harbor_host(
        sandbox,
        endpoint,
        ["bash", "/tests/test.sh"],
        {"RSI_SWEBENCH_CONCURRENCY": "3"},
        tests=SAMPLE / "tests",
    )
    container.start()
    status = container.wait(timeout=3000)
    output = container.logs().decode(errors="replace")
    assert status["StatusCode"] == 0, output[-4000:]
    reward = json.loads(read_file(container, "/logs/verifier/reward.json"))
    summary = json.loads(read_file(container, "/logs/verifier/harbor-summary.json"))

    assert reward == {"reward": 1.0}, (summary, output[-4000:])
    report = summary["suites"]["swebench"]
    assert report["tasks"] == TASKS
    for agent in ("oracle", "nop"):
        assert report[agent]["ok"] and report[agent]["trials"] == 3, report
    assert "harbor run swebench oracle: exit 0" in output
    assert "harbor run swebench nop: exit 0" in output
    # Six envs, none with network, each from its task's pinned image, which
    # the broker bound from the host cache: a download would have exceeded
    # max_pull_mb = 1 and failed the trial.
    assert networks == ["none"] * 6
    assert sorted(pulls) == sorted(
        ref.removeprefix("docker.io/") for ref in IMAGES for _ in range(2)
    )
    assert installs == ([("main", "tmux")] * 6 if tmux else [])
    assert_only_host_left(sandbox, container)


def test_the_operator_tmux_runs_offline_on_a_swebench_image(
    live_sandbox, static_tmux, tmp_path
):
    policy = with_tmux(POLICY.read_text(), static_tmux, tmp_path)
    sandbox = sample_sandbox(live_sandbox, "swebench-tmux", policy)
    client = client_of(sandbox.judge())
    assert client.capabilities()["environments"]["tools"] == ["tmux"]
    view = client.follow_job(client.image_pull(IMAGES[0], "missing"))
    assert view["state"] == "succeeded", view
    handle = view["result"]["image"]["handle"]

    spec = {
        "version": 1,
        "network": "none",
        "lifetime_sec": 600,
        "disk_mb": 256,
        "services": {
            "main": {
                "image": handle,
                "command": ["sleep", "600"],
                "cpus": 1,
                "memory_mb": 512,
            }
        },
    }
    env_id = client.env_create(spec)["env_id"]
    client.env_start(env_id, 120)
    assert client.wait_env(env_id, 120)["state"] == "ready"
    assert shell(client, env_id, "command -v tmux")[0] != 0
    assert install(client, env_id)["installed"] is True
    code, output = shell(client, env_id, "command -v tmux && " + SESSION)
    assert code == 0, output
    assert output.startswith(TMUX + "\n")
    assert "hello-42" in output
    # Offline indeed: the env has no route out.
    code, output = shell(
        client,
        env_id,
        "python3 -c 'import socket; socket.create_connection((\"1.1.1.1\", 53), 3)'",
    )
    assert code != 0 and "unreachable" in output.lower(), output
    assert client.env_destroy(env_id) == {"state": "removed"}
    assert client.image_release(handle) == {"ok": True}
