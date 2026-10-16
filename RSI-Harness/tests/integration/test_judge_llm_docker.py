"""Real Docker: a Judge calls the task's Judge LLM through ``rsi-agent llm``.

The model is always a fake: a Responses API server on the host, or a stand-in
for the Codex CLI. The last test runs the real pinned Codex CLI, copied out of
the Codex Work image the Harness builds, against a fake ChatGPT backend.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socketserver
import ssl
import subprocess
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.judge_agent import JUDGE_CODEX_CACHE, rsi_agent
from rsi_harness.models import (
    CompileOptions,
    JudgeLLMAuth,
    JudgeLLMRequest,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.images import DockerImageBuilder
from rsi_harness.runtime.judge_llm import _without_refresh_token
from rsi_harness.runtime.judge_model_access import build_launch_spec
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness.agent import create_agent, versions
from rsi_loop.harness.agent.claude_code import ClaudeCodeAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeFirewallBackend
from tests.integration.test_judge_model_access_docker import (
    PYTHON_IMAGE,
    _docker,
    _install,
)
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

pytestmark = pytest.mark.integration

# Synthetic: accepted only by the fake servers below.
API_KEY = "sk-synthetic-judge-llm-docker-6e3a"
WORK_KEY = "sk-synthetic-work-agent-docker-0b1c"
REFRESH_TOKEN = "SYNTHETIC-JUDGE-LLM-REFRESH-71aa"
FAKE_APP_SERVER = Path(__file__).parents[1] / "fakes" / "fake_codex_app_server.py"
SCHEMA = (
    '{"type":"object","properties":{"verdict":{"type":"string"}},'
    '"required":["verdict"],"additionalProperties":false}'
)


def _login() -> dict:
    """A ChatGPT-plan Codex login with synthetic, unsigned tokens."""

    def part(value: dict) -> str:
        raw = json.dumps(value).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    claims = {
        "exp": int(time.time()) + 86400,
        "https://api.openai.com/auth": {
            "chatgpt_plan_type": "pro",
            "chatgpt_account_id": "acc-1",
            "chatgpt_account_user_id": "user-1__acc-1",
        },
    }
    return {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": f"h.{part(claims)}.s",
            "access_token": f"h.{part(claims)}.s",
            "refresh_token": REFRESH_TOKEN,
            "account_id": "acc-1",
        },
        "last_refresh": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


class _Responses(BaseHTTPRequestHandler):
    """A fake Responses API: echoes the key it got inside its answer."""

    seen: list[tuple[str, dict]] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        authorization = self.headers.get("Authorization", "")
        type(self).seen.append((authorization, body))
        if authorization != f"Bearer {API_KEY}":
            self._json(401, {"error": {"message": "bad key"}})
            return
        text = json.dumps({"verdict": "clear", "echo": API_KEY})
        self._json(
            200,
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
            },
        )

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        del args


def _gateway(client) -> str:
    return next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )


def _task(tmp_path: Path, metadata: str, script: str) -> Path:
    task = tmp_path / "task"
    shutil.copytree(FIXTURE, task)
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {PYTHON_IMAGE}\nRUN mkdir -p /workspace\nWORKDIR /workspace\n"
    )
    config = task / "task.toml"
    config.write_text(
        config.read_text() + "\n[metadata.rsi_harness.verifier]\n" + metadata
    )
    (task / "tests" / "test.sh").write_text(script)
    return task


class _ScriptedAgent(RSILoopAgentAdapter):
    def prepare(self, request):
        prepared = super().prepare(request)
        # Work never has rsi-agent or the Judge's Codex CLI; then submit once.
        script = (
            "set -eu\n"
            "test ! -e /opt/rsi-agent && echo NO-RSI-AGENT-IN-WORK\n"
            "test ! -e /opt/rsi-agent-codex && echo NO-CODEX-CLI-IN-WORK\n"
            "rsi-submit\n"
        )
        return replace(prepared, command=("/bin/bash", "-lc", script))


def _services(tmp_path: Path, client, monkeypatch, base_url: str):
    # The Work Agent is not Codex: the Judge LLM is independent of it.
    monkeypatch.setattr(ClaudeCodeAgent, "install_cmds", _install("claude"))
    return ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=WORK_KEY, agent_api_base_url=base_url
        ),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: _ScriptedAgent(
            config, runtime=runtime
        ),
        quiescence_checker=lambda _allocation, _container: None,
    )


def _run(services, request: pytest.FixtureRequest, data: Path, **options):
    def cleanup() -> None:
        for lease_path in sorted((data / "leases").glob("*.json")):
            services.cleanup(lease_path.stem, delete_workspace=True)

    request.addfinalizer(cleanup)
    return services.run(
        RunRequest(
            task_dir=options.pop("task"),
            agent_name="claude-code",
            model="claude-test",
            gpu_selectors=("0",),
            options=CompileOptions(agent_name="claude-code", primary_reward="reward"),
            **options,
        )
    )


API_JUDGE = r"""#!/bin/bash
set -u
# The procedure holds no credential; only rsi-agent does.
for name in RSI_JUDGE_LLM_API_KEY OPENAI_API_KEY ANTHROPIC_AUTH_TOKEN \
    ANTHROPIC_API_KEY; do
    if [ -n "${!name:-}" ]; then
        echo "procedure sees $name"
        echo '{"reward": 0}' > /logs/verifier/reward.json
        exit 0
    fi
done
printf 'judge-model=%s effort=%s\n' "$RSI_JUDGE_MODEL" "$RSI_JUDGE_MODEL_EFFORT"
mkdir -p /tmp/review
printf 'Rate the submission.\n' > /tmp/review/prompt.md
printf '%s' 'SCHEMA' > /tmp/review/schema.json
rsi-agent llm --prompt-file /tmp/review/prompt.md \
    --output /tmp/review/answer.json --schema /tmp/review/schema.json --timeout 60
echo "llm exit $?"
rsi-agent exec --prompt-file /tmp/review/prompt.md --output /tmp/review/x \
    --timeout 10
echo "exec exit $?"
cat /tmp/review/answer.json; echo
python3 - <<'PY'
import json
a = json.load(open("/tmp/review/answer.json"))
ok = a["verdict"] == "clear" and a["echo"] == "[REDACTED]"
json.dump({"reward": 1 if ok else 0}, open("/logs/verifier/reward.json", "w"))
PY
""".replace("SCHEMA", SCHEMA)


def test_real_judge_calls_the_judge_llm_with_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    client = _docker(PYTHON_IMAGE)
    gateway = _gateway(client)
    _Responses.seen = []
    server = ThreadingHTTPServer(("0.0.0.0", 0), _Responses)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request.addfinalizer(server.shutdown)
    base_url = f"http://{gateway}:{server.server_address[1]}"
    monkeypatch.setenv("RSI_JUDGE_LLM_API_KEY", API_KEY)
    task = _task(
        tmp_path, 'judge_model = "gpt-judge"\njudge_model_effort = "low"\n', API_JUDGE
    )
    services = _services(tmp_path, client, monkeypatch, base_url + "/v1")

    result = _run(
        services,
        request,
        tmp_path / "data",
        task=task,
        judge_llm=JudgeLLMRequest(auth=JudgeLLMAuth.API, base_url=base_url),
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    feedback = (run_root / "feedback" / "agent-1.log").read_text()
    agent_output = (run_root / "agent_output.txt").read_text()
    assert result.status == RunStatus.COMPLETED, feedback + agent_output[-2000:]
    assert result.best_score == 1, feedback
    assert "judge-model=gpt-judge effort=low" in feedback
    assert "llm exit 0" in feedback
    assert "exec exit 3" in feedback
    assert "NO-RSI-AGENT-IN-WORK" in agent_output
    assert "NO-CODEX-CLI-IN-WORK" in agent_output
    [(authorization, body)] = _Responses.seen
    assert authorization == f"Bearer {API_KEY}"
    assert body["model"] == "gpt-judge"
    assert body["reasoning"] == {"effort": "low"}
    assert body["text"]["format"]["strict"] is True
    for path in (tmp_path / "logs").rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text(errors="replace"), path
    plan = json.loads((run_root / "run-plan.json").read_text())
    verifier = plan["task"]["verifier"]
    assert (verifier["judge_model"], verifier["judge_model_effort"]) == (
        "gpt-judge",
        "low",
    )


CODEX_JUDGE = r"""#!/bin/bash
set -u
mkdir -p /tmp/review
printf 'IDENTITY of the caller\n' > /tmp/review/prompt.md
rsi-agent llm --prompt-file /tmp/review/prompt.md \
    --output /tmp/review/answer.json --timeout 60
echo "llm exit $?"
cat /tmp/review/answer.json; echo
if touch /opt/rsi-agent-codex/x 2>/dev/null; then echo CODEX-WRITABLE; fi
if [ -n "${CODEX_HOME:-}" ]; then echo "procedure sees CODEX_HOME"; fi
python3 - <<'PY'
import json
a = json.load(open("/tmp/review/answer.json"))
home = a["home"]
ok = (a["uid"] == 61000 and home.startswith("/run/rsi-agent/jobs/")
      and a["codex_home"] == home + "/.codex" and a["login_mode"] == "0o600"
      and a["login_owner"] == 61000 and a["cwd"].startswith("/run/rsi-agent/jobs/"))
json.dump({"reward": 1 if ok else 0}, open("/logs/verifier/reward.json", "w"))
PY
"""


def test_real_judge_calls_the_codex_cli_as_its_own_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    client = _docker(PYTHON_IMAGE)
    gateway = _gateway(client)
    home = tmp_path / "codex-home"
    home.mkdir()
    (home / "auth.json").write_text(json.dumps(_login()))
    (home / "auth.json").chmod(0o600)
    prepared = []

    def stand_in(self, task, *, images, work_agent, cache_root):
        # The Codex CLI stand-in, where the Harness caches the real one.
        prepared.append(work_agent.name)
        directory = cache_root / "codex-test-x86_64"
        directory.mkdir(parents=True, exist_ok=True)
        codex = directory / "codex"
        codex.write_text(FAKE_APP_SERVER.read_text())
        codex.chmod(0o755)
        return directory

    monkeypatch.setattr(DockerImageBuilder, "prepare_judge_codex", stand_in)
    task = _task(tmp_path, 'judge_model = "gpt-judge"\n', CODEX_JUDGE)
    # The Work provider address doubles as the Judge LLM's proxy: an address
    # literal, pinned without DNS. The stand-in never uses it.
    proxy = f"http://{gateway}:9"
    services = _services(tmp_path, client, monkeypatch, proxy + "/v1")

    result = _run(
        services,
        request,
        tmp_path / "data",
        task=task,
        judge_llm=JudgeLLMRequest(
            auth=JudgeLLMAuth.CODEX_LOGIN, codex_home=home, proxy=proxy
        ),
    )

    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    feedback = (run_root / "feedback" / "agent-1.log").read_text()
    agent_output = (run_root / "agent_output.txt").read_text()
    assert result.status == RunStatus.COMPLETED, feedback + agent_output[-2000:]
    assert result.best_score == 1, feedback
    assert prepared == ["claude-code"]
    assert "llm exit 0" in feedback
    assert "CODEX-WRITABLE" not in feedback
    assert "procedure sees" not in feedback
    assert "NO-CODEX-CLI-IN-WORK" in agent_output
    for path in (tmp_path / "logs").rglob("*"):
        if path.is_file():
            assert REFRESH_TOKEN not in path.read_text(errors="replace"), path


# -- the real Codex CLI ------------------------------------------------------------


class _ChatGPT(BaseHTTPRequestHandler):
    """A fake ChatGPT backend for the Codex CLI: routing, then one answer."""

    protocol_version = "HTTP/1.1"
    seen: list[tuple[str, str, object]] = []

    def log_message(self, *args) -> None:
        del args

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        type(self).seen.append(("GET", self.path, None))
        if self.path.split("?")[0].endswith("/models"):
            # The live catalog's shape for these models: code mode, a clock,
            # user messages, apply_patch and tool search. rsi-agent's own
            # catalog must win over it (and over Codex's bundled one).
            self._json(200, {"models": [_catalog_tools_entry(m) for m in TOOL_MODELS]})
            return
        if "/accounts/check" in self.path:
            account = {
                "id": "acc-1",
                "plan_type": "pro",
                "workspace_backend_origin": "NO_CONSTRAINT",
                "account_routing_override": "NO_CONSTRAINT",
                "structure": "personal",
            }
            self._json(
                200,
                {
                    "accounts": [account],
                    "account_ordering": ["acc-1"],
                    "default_account_id": "acc-1",
                },
            )
            return
        self._json(404, {"error": "not here"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        type(self).seen.append(("POST", self.path, body))
        if not self.path.split("?")[0].endswith("/codex/responses"):
            self._json(404, {"error": "not here"})
            return
        text = json.dumps({"verdict": "clear"})
        item = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        response = {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": body.get("model") if isinstance(body, dict) else None,
            "output": [item],
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 1,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 2,
            },
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for kind, fields in (
            ("response.created", {"response": {**response, "output": []}}),
            ("response.output_item.added", {"output_index": 0, "item": item}),
            ("response.output_item.done", {"output_index": 0, "item": item}),
            ("response.completed", {"response": response}),
        ):
            payload = json.dumps({"type": kind, **fields})
            self.wfile.write(f"event: {kind}\ndata: {payload}\n\n".encode())
        self.wfile.flush()
        self.close_connection = True


# Models whose bundled (and live) catalog entries ask for tools.
TOOL_MODELS = ("gpt-6-sol", "gpt-5.6-sol")


def _catalog_tools_entry(model: str) -> dict:
    [entry] = rsi_agent.codex_catalog(model, "high")["models"]
    return {
        **entry,
        "tool_mode": "code_mode_only",
        "experimental_supported_tools": ["send_user_message_async", "clock"],
        "apply_patch_tool_type": "freeform",
        "shell_type": "shell_command",
        "supports_search_tool": True,
    }


def _tls(directory: Path) -> tuple[Path, ssl.SSLContext]:
    """A throwaway CA and a server certificate for 127.0.0.1."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not installed")
    directory.mkdir()

    def openssl(*arguments: str) -> None:
        subprocess.run(
            ["openssl", *arguments], cwd=directory, check=True, capture_output=True
        )

    openssl(
        "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key",
        "-out", "ca.pem", "-days", "1", "-subj", "/CN=judge-llm-test-ca",
        "-addext", "basicConstraints=critical,CA:TRUE",
        "-addext", "keyUsage=critical,keyCertSign,cRLSign",
    )  # fmt: skip
    openssl(
        "req", "-newkey", "rsa:2048", "-nodes", "-keyout", "server.key",
        "-out", "server.csr", "-subj", "/CN=127.0.0.1",
    )  # fmt: skip
    (directory / "ext.cnf").write_text(
        "subjectAltName=IP:127.0.0.1\nbasicConstraints=CA:FALSE\n"
        "extendedKeyUsage=serverAuth\n"
    )
    openssl(
        "x509", "-req", "-in", "server.csr", "-CA", "ca.pem", "-CAkey", "ca.key",
        "-CAcreateserial", "-out", "server.pem", "-days", "1", "-extfile", "ext.cnf",
    )  # fmt: skip
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(directory / "server.pem", directory / "server.key")
    return directory / "ca.pem", context


def _real_codex_cli(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """The pinned Codex CLI as the Harness prepares it for a Judge LLM."""
    client = _docker(PYTHON_IMAGE)
    task_dir = _task(tmp_path, 'judge_model = "gpt-judge"\n', "#!/bin/bash\n")
    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())
    builder = DockerImageBuilder(client, run_id="judge-codex-test")
    agent = create_agent("claude-code", RSILoopConfig())
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    monkeypatch.setattr(ClaudeCodeAgent, "install_cmds", _install("claude"))
    try:
        images = builder.prepare(definition, agent=agent)
        return builder.prepare_judge_codex(
            definition,
            images=images,
            work_agent=agent,
            cache_root=tmp_path / "data" / JUDGE_CODEX_CACHE,
        )
    except Exception as error:  # the npm registry may be unreachable
        if os.environ.get("RSI_REQUIRE_DOCKER_TESTS") == "1":
            raise
        pytest.skip(f"cannot build the Codex CLI: {error}")


@pytest.mark.parametrize("model", ["gpt-judge", *TOOL_MODELS])
def test_real_codex_cli_answers_through_a_tool_less_turn(
    tmp_path: Path, request: pytest.FixtureRequest, model: str
) -> None:
    """The pinned Codex CLI, built by the Harness, speaks the protocol we use.

    ``gpt-judge`` is unknown to Codex; the others are catalog models that ask
    for code mode and other tools, which rsi-agent's catalog turns off.
    """
    codex = _real_codex_cli(tmp_path, request) / "codex"
    reported = subprocess.run(
        [str(codex), "--version"], capture_output=True, text=True, check=False
    )
    assert reported.stdout.strip() == f"codex-cli {versions.CODEX}"

    ca, context = _tls(tmp_path / "tls")
    _ChatGPT.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatGPT)
    server.daemon_threads = True
    server.socket = context.wrap_socket(server.socket, server_side=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request.addfinalizer(server.shutdown)
    backend = f"https://127.0.0.1:{server.server_address[1]}/backend-api"

    state = tmp_path / "state"
    (state / "private").mkdir(parents=True)
    state.chmod(0o711)
    (state / "private").chmod(0o700)
    # The Judge's copy, as the Harness makes it: the refresh token withheld.
    login = _without_refresh_token(json.dumps(_login()))
    llm = {
        "kind": "codex-login",
        "model": model,
        "effort": "high",
        "codex": str(codex),
        "auth_json": login,
        # Test only: the fake backend in place of chatgpt.com.
        "environment": {"SSL_CERT_FILE": str(ca)},
        "config": [
            f'openai_base_url="{backend}/codex"',
            f'chatgpt_base_url="{backend}/"',
            "features.enable_request_compression=false",
        ],
    }
    (state / "private" / "launch.json").write_bytes(
        build_launch_spec(None, None, {REFRESH_TOKEN}, llm=llm)
    )
    (tmp_path / "prompt.md").write_text("Is the submission honest?")
    (tmp_path / "schema.json").write_text(SCHEMA)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", rsi_agent.__file__, "llm"]
        + ["--prompt-file", str(tmp_path / "prompt.md")]
        + ["--output", str(tmp_path / "answer.json")]
        + ["--schema", str(tmp_path / "schema.json"), "--timeout", "120"]
        + ["--log", str(tmp_path / "call.log")],
        env={"PATH": "/usr/bin:/bin", "RSI_AGENT_STATE_ROOT": str(state)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr + (tmp_path / "call.log").read_text()
    assert json.loads((tmp_path / "answer.json").read_text()) == {"verdict": "clear"}
    [turn] = [
        body
        for method, path, body in _ChatGPT.seen
        if method == "POST" and path.endswith("/codex/responses")
    ]
    # No tools at all, the neutral instructions, the effort and the schema.
    assert turn["tools"] == []
    assert turn["model"] == model
    assert turn["reasoning"]["effort"] == "high"
    assert turn["text"]["format"]["type"] == "json_schema"
    assert turn["text"]["format"]["strict"] is True
    assert json.loads(SCHEMA) == turn["text"]["format"]["schema"]
    developer, user = turn["input"]
    assert developer["role"] == "developer"
    assert developer["content"][0]["text"] == rsi_agent.LLM_INSTRUCTIONS
    assert user["content"][0]["text"] == "Is the submission honest?"
    assert REFRESH_TOKEN not in (tmp_path / "call.log").read_text()
    # Only the ChatGPT backend was asked for anything.
    assert {path.split("/")[1] for _, path, _ in _ChatGPT.seen} == {"backend-api"}


# -- the operator's login, refreshed on the host by the real Codex CLI -----------


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"h.{payload.decode()}.s"


class _AuthServer(BaseHTTPRequestHandler):
    """A fake OAuth token endpoint that rotates refresh tokens, as the real one
    does: only the latest is accepted, each grant issues a new one."""

    seen: list[dict] = []
    valid = REFRESH_TOKEN
    issued: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        server = type(self)
        server.seen.append({"path": self.path, **body})
        if body.get("refresh_token") != server.valid:
            self._json(400, {"error": "refresh_token_reused"})
            return
        number = len(server.issued) + 1
        claims = {
            "exp": int(time.time()) + 86400,
            "grant": number,
            "https://api.openai.com/auth": {
                "chatgpt_plan_type": "pro",
                "chatgpt_account_id": "acc-1",
            },
        }
        server.valid = f"SYNTHETIC-ROTATED-REFRESH-{number}"
        tokens = {
            "id_token": _jwt(claims),
            "access_token": _jwt(claims),
            "refresh_token": server.valid,
        }
        server.issued.append(tokens)
        self._json(200, tokens)

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        del args


class _Sink(socketserver.BaseRequestHandler):
    """A proxy that refuses everything: nothing else leaves the host."""

    seen: list[bytes] = []

    def handle(self) -> None:
        type(self).seen.append(self.request.recv(4096).split(b"\r\n")[0])


def _serve(server, request: pytest.FixtureRequest) -> int:
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    request.addfinalizer(server.shutdown)
    return server.server_address[1]


# 120 s is inside Codex's own refresh window (it also refreshes as it
# starts); 1200 s is outside it, so only account/read's refresh runs.
@pytest.mark.parametrize("left", [120, 1200])
def test_real_codex_cli_refreshes_the_operators_login_in_place(
    tmp_path: Path, request: pytest.FixtureRequest, left: int
) -> None:
    """account/read with refreshToken: Codex refreshes and saves auth.json.

    Codex's refresh endpoint is pointed at a local fake through
    CODEX_REFRESH_TOKEN_URL_OVERRIDE, the variable Codex's own tests use;
    every other request goes to a proxy that refuses it.
    """
    from rsi_harness.runtime.judge_login_refresh import (
        OperatorCodexLogin,
        refresh_codex_login,
        renew_operator_login,
    )
    from rsi_harness.runtime.local_auth import read_codex_login

    directory = _real_codex_cli(tmp_path, request)
    current = _login()
    claims = {
        "exp": int(time.time()) + left,
        "https://api.openai.com/auth": {
            "chatgpt_plan_type": "pro",
            "chatgpt_account_id": "acc-1",
        },
    }
    current["tokens"].update(id_token=_jwt(claims), access_token=_jwt(claims))
    home = tmp_path / "operator-codex"
    home.mkdir(mode=0o700)
    (home / "auth.json").write_text(json.dumps(current))
    (home / "auth.json").chmod(0o600)
    inode = (home / "auth.json").stat().st_ino
    _AuthServer.seen, _AuthServer.issued = [], []
    _AuthServer.valid = REFRESH_TOKEN
    auth = _serve(ThreadingHTTPServer(("127.0.0.1", 0), _AuthServer), request)
    _Sink.seen = []
    sink = _serve(socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Sink), request)
    login = OperatorCodexLogin(
        codex_home=home,
        uid=os.getuid(),
        gid=os.getgid(),
        read=lambda: read_codex_login(home, environ={}),
        proxy=f"http://127.0.0.1:{sink}",
    )
    override = {
        "CODEX_REFRESH_TOKEN_URL_OVERRIDE": f"http://127.0.0.1:{auth}/oauth/token"
    }

    def refresh(login, binary, *, timeout, saved=None):
        return refresh_codex_login(
            login, binary, timeout=timeout, saved=saved, environment=override
        )

    started = time.monotonic()
    renewal = renew_operator_login(
        login, codex_cli=directory, round_seconds=1800, refresh=refresh
    )

    assert renewal.warning is None
    assert time.monotonic() - started < 60
    # Codex's own refresh grants, each with the then-current token, and the
    # last tokens issued saved in place: the same file, still private.
    assert [seen["grant_type"] for seen in _AuthServer.seen] == ["refresh_token"] * (
        len(_AuthServer.issued)
    )
    assert len(_AuthServer.issued) == (2 if left == 120 else 1)
    saved = json.loads((home / "auth.json").read_text())
    latest = _AuthServer.issued[-1]
    assert saved["tokens"]["access_token"] == latest["access_token"]
    assert saved["tokens"]["refresh_token"] == latest["refresh_token"]
    assert json.loads(renewal.content) == saved
    stat = (home / "auth.json").stat()
    assert (stat.st_ino, stat.st_mode & 0o777) == (inode, 0o600)
    assert {seen["path"] for seen in _AuthServer.seen} == {"/oauth/token"}
    # Nothing else got past the refusing proxy, and nothing was left behind.
    assert all(line.startswith(b"CONNECT ") for line in _Sink.seen)
    assert sorted(os.listdir(home)) == ["auth.json"]
