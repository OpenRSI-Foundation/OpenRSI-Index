"""Real Docker: each Judge round gets the login Work's Agent keeps refreshing.

The Work Agent is Codex with a local ChatGPT login (synthetic). Between its
two submissions it rewrites its auth.json, as Codex does on a token refresh;
the second Judge's ``rsi-agent exec`` and ``rsi-agent llm`` must both see the
new login, with its refresh token withheld. No real credential is involved.
"""

from __future__ import annotations

import base64
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    ContainerTmpfs,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime import production
from rsi_harness.runtime.images import DockerImageBuilder
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
    login_secret_values,
)
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_judge_model_access_docker import PYTHON_IMAGE, _docker
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

FAKE_APP_SERVER = Path(__file__).parents[1] / "fakes" / "fake_codex_app_server.py"

# A stand-in for ``codex exec``: answers with the login it was handed.
FAKE_CODEX = r"""#!/usr/bin/env python3
import base64, json, os, sys
args = sys.argv[1:]
sys.stdin.read()
login = json.load(open(os.path.join(os.environ["HOME"], ".codex", "auth.json")))
tokens = login["tokens"]
payload = tokens["access_token"].split(".")[1]
claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
out = args[args.index("--output-last-message") + 1]
open(out, "w").write(json.dumps({
    "generation": claims["generation"],
    "withheld": tokens["refresh_token"] == "withheld-by-rsi-harness",
    "token": tokens["access_token"],
}))
"""

JUDGE = r"""#!/bin/bash
set -u
mkdir -p /tmp/review
printf 'LOGIN\n' > /tmp/review/prompt.md
rsi-agent exec --prompt-file /tmp/review/prompt.md --output /tmp/review/exec.json \
    --timeout 60
echo "exec exit $?"
rsi-agent llm --prompt-file /tmp/review/prompt.md --output /tmp/review/llm.json \
    --timeout 60
echo "llm exit $?"
cat /tmp/review/exec.json; echo
python3 - <<'PY'
import json
e = json.load(open("/tmp/review/exec.json"))
l = json.load(open("/tmp/review/llm.json"))
print(f"exec generation={e['generation']} withheld={e['withheld']}")
print(f"llm generation={l['generation']} withheld={l['withheld']}")
json.dump({"reward": e["generation"]}, open("/logs/verifier/reward.json", "w"))
PY
"""


def _login(generation: int) -> dict:
    def part(value: dict) -> str:
        raw = json.dumps(value).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    claims = {
        "exp": int(time.time()) + 86400,
        "generation": generation,
        "https://api.openai.com/auth": {"chatgpt_account_id": "acc-1"},
    }
    return {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": f"h.{part({**claims, 'id': generation})}.s",
            "access_token": f"h.{part(claims)}.s",
            "refresh_token": f"SYNTHETIC-WORK-REFRESH-{generation}-0123456789abcdef",
            "account_id": "acc-1",
        },
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _material(login: dict) -> AgentAuthMaterial:
    return AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options="rw,nosuid,nodev,noexec,mode=0700",
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=json.dumps(login).encode(),
                    ),
                ),
            ),
        ),
        secret_values=login_secret_values(login),
    )


def test_the_next_judge_round_sees_the_login_work_refreshed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    client = _docker(PYTHON_IMAGE)
    gateway = next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )
    first, second = _login(1), _login(2)
    # Never the operator's own login: the run-start login is this one.
    monkeypatch.setattr(
        production,
        "resolve_agent_auth",
        lambda **_options: _material(first),
    )
    encoded = base64.b64encode(FAKE_CODEX.encode()).decode()
    monkeypatch.setattr(
        CodexAgent,
        "install_cmds",
        [
            f"echo {encoded} | base64 -d > /usr/local/bin/codex && "
            "chmod 0755 /usr/local/bin/codex"
        ],
    )

    def stand_in(self, task, *, images, work_agent, cache_root):
        directory = cache_root / "codex-test-x86_64"
        directory.mkdir(parents=True, exist_ok=True)
        codex = directory / "codex"
        codex.write_text(FAKE_APP_SERVER.read_text())
        codex.chmod(0o755)
        return directory

    monkeypatch.setattr(DockerImageBuilder, "prepare_judge_codex", stand_in)

    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PYTHON_IMAGE}\nRUN mkdir -p /workspace\nWORKDIR /workspace\n"
    )
    config = task / "task.toml"
    config.write_text(
        config.read_text().replace("timeout_sec = 60", "timeout_sec = 600")
        + "\n[metadata.rsi_harness.verifier]\nmodel_access = true\n"
        + 'judge_model = "gpt-judge"\n'
    )
    (task / "tests" / "test.sh").write_text(JUDGE)

    refreshed = base64.b64encode(json.dumps(second).encode()).decode()

    class ScriptedAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            # Submit, refresh the login in place as Codex does, submit again.
            script = (
                "set -eu\n"
                "rsi-submit\n"
                f"echo {refreshed} | base64 -d > /home/agent/.codex/auth.json\n"
                "rsi-submit\n"
            )
            return replace(prepared, command=("/bin/bash", "-lc", script))

    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        # Address literals, pinned without DNS: a Work provider (exec) and a
        # proxy the lent ChatGPT login goes through (llm). Neither is reached.
        rsi_loop_config=RSILoopConfig(
            agent_api_base_url=f"http://{gateway}:9/v1",
            https_proxy=f"http://{gateway}:9",
        ),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: ScriptedAgent(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
    )

    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)

    result = services.run(
        RunRequest(
            task_dir=task,
            agent_name="codex",
            model="gpt-test",
            gpu_selectors=("0",),
            agent_auth=AgentAuthSource.LOCAL,
            options=CompileOptions(agent_name="codex", primary_reward="reward"),
        )
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    feedback = [
        (run_root / "feedback" / f"agent-{index}.log").read_text() for index in (1, 2)
    ]
    agent_output = (run_root / "agent_output.txt").read_text()
    assert result.status == RunStatus.COMPLETED, "".join(feedback) + agent_output
    for generation, text in enumerate(feedback, start=1):
        assert "exec exit 0" in text and "llm exit 0" in text, text
        assert f"exec generation={generation} withheld=True" in text, text
        assert f"llm generation={generation} withheld=True" in text, text
        assert '"token": "[REDACTED]"' in text, text
    assert result.best_score == 2
    # Neither login's tokens reach any log; no refresh token reached a Judge.
    secrets = login_secret_values(first) | login_secret_values(second)
    for path in (tmp_path / "logs").rglob("*"):
        if path.is_file():
            text = path.read_text(errors="replace")
            assert not [value for value in secrets if value in text], path
