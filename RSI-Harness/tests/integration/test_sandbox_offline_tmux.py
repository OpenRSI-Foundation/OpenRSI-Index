"""The operator's static tmux in envs without network, on real Docker.

The broker copies the binary built by scripts/operator/build_static_tmux.sh
into images that have no tmux (busybox: no terminfo database at all;
python slim: Debian), and tmux runs there offline: first through
the protocol, then through the Harbor plugin with Harbor's own terminus-2
TmuxSession. RSI_STATIC_TMUX names the binary; a missing file is built
there first (a cache, about 30 s with network), and unset skips.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.integrations.sandbox_client import ProtocolError, SandboxClient
from tests.integration.harbor_env_support import (
    live_sandbox as live_sandbox,
)
from tests.integration.harbor_env_support import use_endpoint
from tests.sandbox_helpers import env_policy_toml

pytestmark = pytest.mark.integration

BUILD = Path(__file__).parents[2] / "scripts" / "operator" / "build_static_tmux.sh"
BUSYBOX = "busybox:1.37.0"
PYTHON = "python:3.13-slim-bookworm"
TMUX = "/usr/local/bin/tmux"
SESSION = (
    "export TERM=xterm-256color; "
    "tmux new-session -d -x 80 -y 24 -s t sh && "
    "tmux send-keys -t t 'echo hello-$((6*7))' Enter && sleep 1 && "
    "tmux capture-pane -p -t t; tmux kill-server"
)


@pytest.fixture(scope="module")
def static_tmux():
    value = os.environ.get("RSI_STATIC_TMUX")
    if not value:
        message = "RSI_STATIC_TMUX names no static tmux (build_static_tmux.sh PATH)"
        if os.environ.get("RSI_REQUIRE_SANDBOX_INTEGRATION") == "1":
            pytest.fail(message)
        pytest.skip(message)
    path = Path(value)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(BUILD), str(path)], check=True, timeout=1800)
    return path


def tmux_policy(binary: Path) -> str:
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    return env_policy_toml(tmux={"path": str(binary), "sha256": digest})


def client_of(endpoint) -> SandboxClient:
    return SandboxClient(
        endpoint.directory / "s", endpoint.environment["RSI_SANDBOX_TOKEN"]
    )


def offline_env(client, handle) -> str:
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
                "memory_mb": 256,
            }
        },
    }
    env_id = client.env_create(spec)["env_id"]
    client.env_start(env_id, 120)
    assert client.wait_env(env_id, 120)["state"] == "ready"
    return env_id


def shell(client, env_id, script) -> tuple[int, str]:
    exec_id = client.exec_start(
        env_id, "main", ["sh", "-c", script], merge_stderr=True, timeout_sec=60
    )
    chunks = []
    view = client.follow_exec(exec_id, lambda stream, data: chunks.append(data))
    return view["exit_code"], b"".join(chunks).decode()


def install(client, env_id) -> dict:
    return client._v2(
        "tool_install", {"env_id": env_id, "service": "main", "tool": "tmux"}
    )


@pytest.mark.parametrize("image", [BUSYBOX, PYTHON])
def test_the_operator_tmux_runs_offline_where_the_image_has_none(
    live_sandbox, static_tmux, tmp_path, image
):
    binary = tmp_path / "tmux"
    shutil.copyfile(static_tmux, binary)
    sandbox = live_sandbox("tmux", image, policy=tmux_policy(binary))
    client = client_of(sandbox.work())
    assert client.capabilities()["environments"]["tools"] == ["tmux"]
    view = client.follow_job(client.image_pull(image, "missing"))
    assert view["state"] == "succeeded", view
    handle = view["result"]["image"]["handle"]

    env_id = offline_env(client, handle)
    assert shell(client, env_id, "command -v tmux")[0] != 0
    assert install(client, env_id) == {
        "tool": "tmux",
        "path": TMUX,
        "installed": True,
    }
    assert install(client, env_id)["installed"] is False  # never replaced
    found = client.path_stat(env_id, "main", TMUX)
    assert (found["kind"], found["mode"], found["size"]) == (
        "file",
        0o755,
        binary.stat().st_size,
    )
    code, output = shell(client, env_id, "command -v tmux && " + SESSION)
    assert code == 0, output
    assert output.startswith(TMUX + "\n")
    assert "hello-42" in output

    # A file that no longer matches its approved hash is never copied.
    binary.write_bytes(binary.read_bytes() + b"\0")
    other = offline_env(client, handle)
    with pytest.raises(ProtocolError) as caught:
        install(client, other)
    assert (caught.value.code, caught.value.field) == ("infrastructure", "tool")
    assert not client.path_stat(other, "main", TMUX)["exists"]
    assert shell(client, other, "command -v tmux")[0] != 0
    for env in (env_id, other):
        assert client.env_destroy(env) == {"state": "removed"}
    assert client.image_release(handle) == {"ok": True}


@pytest.mark.asyncio
async def test_harbors_terminus_tmux_session_runs_offline_through_the_plugin(
    live_sandbox, static_tmux, monkeypatch, tmp_path
):
    from harbor.agents.terminus_2.tmux_session import TmuxSession
    from harbor.models.task.config import (
        EnvironmentConfig,
        NetworkMode,
        NetworkPolicy,
    )
    from harbor.models.trial.paths import TrialPaths

    from rsi_harness.integrations.sandbox_harbor_env import (
        ManagedSandboxEnvironment,
    )

    sandbox = live_sandbox("tmux-harbor", PYTHON, policy=tmux_policy(static_tmux))
    use_endpoint(monkeypatch, sandbox.work())
    (tmp_path / "environment").mkdir()
    policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
    environment = ManagedSandboxEnvironment(
        environment_dir=tmp_path / "environment",
        environment_name="offline-tmux",
        session_id="offline-tmux__env",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image=PYTHON, cpus=1, memory_mb=512),
        network_policy=policy,
        phase_network_policies=[policy],
        mounts=[
            {"type": "bind", "source": str(tmp_path / name), "target": f"/logs/{name}"}
            for name in ("agent", "verifier", "artifacts")
        ],
    )
    commands = []
    exec_command = environment.exec

    async def spy(command, **options):
        commands.append(command)
        return await exec_command(command, **options)

    await environment.start(force_build=False)
    try:
        environment.exec = spy
        # terminus-2's session, as its setup() makes it without recording.
        session = TmuxSession(
            session_name="terminus-2",
            environment=environment,
            logging_path=PurePosixPath("/logs/agent/terminus_2.pane"),
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        await session.start()
        await session.send_keys(["echo hello-$((6*7))", "Enter"], min_timeout_sec=1)
        assert "hello-42" in await session.capture_pane()
        found = await environment.exec("command -v tmux")
        assert (found.stdout or "").strip() == TMUX
    finally:
        await environment.stop(delete=True)
    # Harbor found tmux and installed nothing.
    assert commands[0] == "tmux -V"
    assert not any("apt-get" in command or "apk " in command for command in commands)
