"""Real Docker: Antigravity on a local Google login (``--agent-auth local``).

The first test needs no login: a synthetic token file goes through the real
Harness path (private tmpfs in Work, the launch's link into agy's directory,
redaction, a model-access Judge's copy without the refresh token) with a
stand-in ``agy``. The second runs the real pinned agy on a real login, in
no-network Work and Judge with the real firewall, and is opt-in:
``RSI_ANTIGRAVITY_LOGIN_FILE`` names a copy of agy's token file at
``<home>/.gemini/antigravity-cli/antigravity-oauth-token`` (private, owned by
the current user); ``RSI_ANTIGRAVITY_MODEL`` picks the model (default
``gemini-3.8-flash-low``). It makes two short model calls.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import rsi_harness.runtime.netns_firewall as netns_firewall
import rsi_harness.runtime.production as production
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.local_auth import resolve_agent_auth
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.agent.antigravity import AntigravityAgent
from rsi_loop.harness.agent.base import NODE_INSTALL_CMD
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_judge_model_access_docker import PYTHON_IMAGE, _docker
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

LOGIN_FILE_ENV = "RSI_ANTIGRAVITY_LOGIN_FILE"
LOGIN_PATH = Path(".gemini/antigravity-cli/antigravity-oauth-token")
LIVE_MODEL = os.environ.get("RSI_ANTIGRAVITY_MODEL", "gemini-3.8-flash-low")

# Synthetic, token-shaped values: never sent anywhere.
ACCESS = "ya29.SYNTHETIC-agy-access-token-for-docker-tests"
REFRESH = "1//SYNTHETIC-agy-refresh-token-for-docker-tests"
EMAIL = "synthetic-user@example.test"

# What Work's Agent checks before it submits; it prints the login file, which
# the transcript must show redacted, and logs the account as agy does.
WORK_CHECKS = r"""
login="$HOME/.gemini/antigravity-cli/antigravity-oauth-token"
echo "link=$(readlink "$login")"
echo "log-link=$(readlink "$HOME/.gemini/antigravity-cli/log")"
email=$(sed -n 's/.*"id_token_email": *"\([^"]*\)".*/\1/p' "$login")
echo "OAuth: authenticated successfully as $email" \
    > "$HOME/.gemini/antigravity-cli/log/cli-1.log"
echo "login-dir=$(stat -c %a:%u /home/agent/.gemini/antigravity-login)"
echo "login-file=$(stat -L -c %a "$login")"
awk '$5 == "/home/agent/.gemini/antigravity-login" { print "mount=" $6 }' \
    /proc/self/mountinfo
grep -q modelProvider "$HOME/.gemini/antigravity-cli/settings.json" \
    && echo settings=api-key || echo settings=login
echo "key=${GEMINI_API_KEY-unset}"
cat "$login"; echo
printf 42 > /workspace/answer.txt
rsi-submit
"""

# A stand-in agy for the Judge's one-shot call: it checks the login it was
# lent and answers in agy's stream-json.
FAKE_AGY = r"""#!/bin/sh
cat >/dev/null
dir="$HOME/.gemini/antigravity-cli"
checks=""
[ -L "$dir/antigravity-oauth-token" ] && checks="$checks linked"
grep -q withheld-by-rsi-harness "$dir/antigravity-oauth-token" \
    && checks="$checks withheld"
grep -q modelProvider "$dir/settings.json" || checks="$checks login-settings"
[ -z "${GEMINI_API_KEY-}" ] && checks="$checks no-key"
reply="fake-agy-says-42:$checks"
step='"step_index":1,"step_type":"agent_response"'
printf '{"step_update":{%s,"text_delta":"%s"}}\n' "$step" "$reply"
echo '{"event":"result","result":{"status":"SUCCESS"}}'
"""

# The Judge runs on Work's rootfs snapshot: agy's logs, which name the
# account, must not be in it.
JUDGE_SCRIPT = r"""#!/bin/bash
set -u
echo "work-agy-log=$(readlink /home/agent/.gemini/antigravity-cli/log)"
echo "account-logs=$(grep -rIlE 'authenticated successfully as|email=' \
    /home/agent/.gemini 2>/dev/null | wc -l)"
mkdir -p /tmp/review/work
printf 'Reply with exactly: JUDGE-OK\n' > /tmp/review/prompt.md
rsi-agent exec --prompt-file /tmp/review/prompt.md \
    --output /tmp/review/answer.txt --workdir /tmp/review/work --timeout 240
echo "rsi-agent exit $?"
answer="$(cat /tmp/review/answer.txt 2>/dev/null)"
echo "judge-answer=$answer"
reward=0
if [ "$(cat /workspace/answer.txt 2>/dev/null)" = 42 ] && [ -n "$answer" ]; then
    reward=1
fi
printf '{"reward": %s}\n' "$reward" > /logs/verifier/reward.json
"""


def _rfc3339(seconds: float) -> str:
    moment = datetime.fromtimestamp(seconds, UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f") + "123Z"


def _task(root: Path, *, instruction: str, agent_timeout: int) -> Path:
    task = root / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PYTHON_IMAGE}\nRUN mkdir -p /workspace\nWORKDIR /workspace\n"
    )
    (task / "instruction.md").write_text(instruction)
    (task / "tests" / "test.sh").write_text(JUDGE_SCRIPT)
    config = task / "task.toml"
    text = config.read_text().replace(
        '[agent]\ntimeout_sec = 60\nnetwork_mode = "public"',
        f'[agent]\ntimeout_sec = {agent_timeout}\nnetwork_mode = "no-network"',
    )
    text = text.replace("[verifier]\ntimeout_sec = 30", "[verifier]\ntimeout_sec = 300")
    assert f"timeout_sec = {agent_timeout}" in text
    # A Judge on Work's rootfs snapshot, not the clean image.
    config.write_text(
        text
        + "\n[metadata.rsi_harness.verifier]\nmodel_access = true\n"
        + "clean_image = false\n"
    )
    return task


def _login_from(monkeypatch: pytest.MonkeyPatch, home: Path) -> list[object]:
    """Resolve the login from *home*, never the operator's own."""
    resolved: list[object] = []

    def resolve(**kwargs):
        material = resolve_agent_auth(
            **kwargs,
            environ={},
            user_lookup=lambda uid: SimpleNamespace(pw_dir=str(home), pw_uid=uid),
        )
        resolved.append(material)
        return material

    monkeypatch.setattr(production, "resolve_agent_auth", resolve)
    return resolved


def _services(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    *,
    adapter: type[RSILoopAgentAdapter] = RSILoopAgentAdapter,
    fake_firewall: bool = True,
) -> ProductionRuntimeServices:
    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=_docker(PYTHON_IMAGE),
        firewall_backend=FakeFirewallBackend() if fake_firewall else None,
        inventory=_OneDeviceInventory(),
        # No key at all: the run is on the login.
        rsi_loop_config=RSILoopConfig(),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: adapter(config, runtime=runtime),
        quiescence_checker=lambda _allocation, _container: None,
    )

    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)
    return services


def _run(services, task: Path, model: str, **options):
    return services.run(
        RunRequest(
            task_dir=task,
            agent_name="antigravity",
            model=model,
            gpu_selectors=("0",),
            agent_auth=AgentAuthSource.LOCAL,
            options=CompileOptions(
                agent_name="antigravity", primary_reward="reward", **options
            ),
        )
    )


def _leaked(root: Path, secrets: set[str]) -> list[str]:
    """Files under *root* holding any secret; names only, never the values."""
    return [
        str(path)
        for path in root.rglob("*")
        if path.is_file()
        and any(secret in path.read_text(errors="replace") for secret in secrets)
    ]


def test_a_synthetic_login_reaches_work_and_a_model_access_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    home = tmp_path / "operator-home"
    login = home / LOGIN_PATH
    login.parent.mkdir(parents=True)
    login.write_text(
        json.dumps(
            {
                "token": {
                    "access_token": ACCESS,
                    "token_type": "Bearer",
                    "refresh_token": REFRESH,
                    "expiry": _rfc3339(time.time() + 3600),
                    "expires_in": 3599,
                },
                "auth_method": "consumer",
                "id_token_email": EMAIL,
            }
        )
    )
    login.chmod(0o600)
    resolved = _login_from(monkeypatch, home)
    encoded = base64.b64encode(FAKE_AGY.encode()).decode()
    monkeypatch.setattr(
        AntigravityAgent,
        "install_cmds",
        [
            NODE_INSTALL_CMD,
            f"echo {encoded} | base64 -d > /usr/local/bin/agy"
            " && chmod 0755 /usr/local/bin/agy",
        ],
    )

    class ScriptedWork(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            agent = create_agent("antigravity", self._config)
            prelude = agent.config_prelude(model="m", internet=False)
            script = f"set -u\n{prelude}\n{WORK_CHECKS}"
            return replace(prepared, command=("/bin/bash", "-c", script))

    result = _run(
        _services(tmp_path, request, adapter=ScriptedWork),
        _task(tmp_path, instruction="Unused.\n", agent_timeout=120),
        "gemini-3.8-flash-low",
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    output = (run_root / "agent_output.txt").read_text()
    feedback = (run_root / "feedback" / "agent-1.log").read_text()
    assert result.status == RunStatus.COMPLETED, output[-3000:] + feedback
    assert [material.agent_name for material in resolved] == ["antigravity"]
    assert "link=../antigravity-login/antigravity-oauth-token" in output
    assert "log-link=../antigravity-login/log" in output
    assert "work-agy-log=../antigravity-login/log" in feedback
    assert "account-logs=0" in feedback
    assert "login-dir=700:0" in output and "login-file=600" in output
    assert "mount=rw,nosuid,nodev,noexec" in output.replace(",relatime", "")
    assert "settings=login" in output and "key=unset" in output
    # The Judge's call had the login, linked, without its refresh token.
    assert "rsi-agent exit 0" in feedback
    assert (
        "judge-answer=fake-agy-says-42: linked withheld login-settings no-key"
        in feedback
    )
    assert result.best_score == 1
    assert _leaked(run_root, {ACCESS, REFRESH, EMAIL}) == []


@pytest.mark.skipif(
    not os.environ.get(LOGIN_FILE_ENV), reason=f"{LOGIN_FILE_ENV} is not set"
)
def test_real_agy_runs_and_a_judge_calls_it_on_a_local_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    login = Path(os.environ[LOGIN_FILE_ENV])
    if not login.is_absolute() or login.parts[-3:] != LOGIN_PATH.parts:
        pytest.fail(f"{LOGIN_FILE_ENV} must be an absolute <home>/{LOGIN_PATH}")
    home = login.parents[2]
    secrets = _secrets(login)
    resolved = _login_from(monkeypatch, home)
    instruction = (
        "Run exactly this shell command once, then reply with the single word "
        "DONE and stop: printf 42 > /workspace/answer.txt && rsi-submit\n"
    )

    # The real firewall, whoever runs the test: Work and the Judge reach only
    # the login's pinned endpoints.
    monkeypatch.setattr(
        production,
        "container_netns_firewall_selected",
        netns_firewall.container_netns_firewall_selected,
    )
    result = _run(
        _services(tmp_path, request, fake_firewall=False),
        _task(tmp_path, instruction=instruction, agent_timeout=420),
        LIVE_MODEL,
        disable_stop_hook=True,
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    output = (run_root / "agent_output.txt").read_text()
    feedback_path = run_root / "feedback" / "agent-1.log"
    feedback = feedback_path.read_text() if feedback_path.exists() else ""
    leaked = _leaked(run_root, secrets)
    assert leaked == []
    assert resolved and resolved[0] is not None
    assert result.best_score == 1, (result, output[-3000:], feedback[-3000:])
    assert result.total_rounds >= 1
    assert "rsi-agent exit 0" in feedback
    assert "judge-answer=" in feedback
    assert "JUDGE-OK" in feedback.split("judge-answer=", 1)[1]
    # Work's real agy logged the account to the tmpfs, not the snapshot.
    assert "work-agy-log=../antigravity-login/log" in feedback
    assert "account-logs=0" in feedback


def _secrets(login: Path) -> set[str]:
    content = json.loads(login.read_text())
    token = content.get("token") or {}
    values = {token.get("access_token"), token.get("refresh_token")}
    values.add(content.get("id_token_email"))
    return {value for value in values if isinstance(value, str) and value}
