"""rsi-agent llm, run as the current (non-root) user against fakes.

The API backend talks to a fake Responses API server in this process; the
Codex backend drives tests/fakes/fake_codex_app_server.py, a stand-in for
``codex app-server`` that speaks the protocol of the pinned Codex release. The
uid drop and the Judge filesystem run in the Docker tests
(tests/integration/test_judge_llm_docker.py).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.judge_agent import rsi_agent
from rsi_harness.models import JudgeLLMAuth, JudgeLLMRequest
from rsi_harness.runtime.judge_llm import resolve_judge_llm
from rsi_harness.runtime.judge_model_access import build_launch_spec
from rsi_harness.runtime.network import PinnedEndpoint
from rsi_loop.harness.config import RSILoopConfig
from tests.landlock_helpers import native_signal_compatibility

PROGRAM = Path(__file__).parents[2] / "src/rsi_harness/judge_agent/rsi_agent.py"
FAKE_APP_SERVER = Path(__file__).parents[1] / "fakes/fake_codex_app_server.py"
API_KEY = "sk-synthetic-judge-llm-key-7d21"
REFRESH_TOKEN = "SYNTHETIC-REFRESH-TOKEN-4410"
LOGIN = json.dumps(
    {
        "auth_mode": "chatgpt",
        "OPENAI_API_KEY": None,
        "tokens": {
            "id_token": "h.e30.s",
            "access_token": "SYNTHETIC-ACCESS-TOKEN-0c3e",
            "refresh_token": REFRESH_TOKEN,
            "account_id": "acc-1",
        },
    }
)
# What the Harness hands a Judge: the refresh token withheld.
JUDGE_LOGIN = json.dumps(
    {
        **json.loads(LOGIN),
        "tokens": {
            **json.loads(LOGIN)["tokens"],
            "refresh_token": "withheld-by-rsi-harness",
        },
    }
)
# Every -c the Codex call passes before its own catalog: the literal list, so
# a dropped setting fails here.
CODEX_FLAGS = [
    'model_provider="openai"',
    'web_search="disabled"',
    'cli_auth_credentials_store="file"',
    "features.apps=false",
    "features.plugins=false",
    "features.tool_suggest=false",
    "features.goals=false",
    "features.multi_agent=false",
    "features.image_generation=false",
    "features.memories=false",
    "features.hooks=false",
    "features.shell_tool=false",
    "features.view_image=false",
    "features.shell_snapshot=false",
    "tools.experimental_request_user_input.enabled=false",
    "skills.include_instructions=false",
    "skills.bundled.enabled=false",
    "include_permissions_instructions=false",
    "include_apps_instructions=false",
    "include_collaboration_mode_instructions=false",
    "include_environment_context=false",
    "project_doc_max_bytes=0",
    "analytics.enabled=false",
    "check_for_update_on_startup=false",
    "features.sleep_tool=false",
    "features.code_mode=false",
    "features.code_mode_only=false",
]
SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


class _Responses(BaseHTTPRequestHandler):
    """A fake Responses API; ``server.mode`` picks the answer."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        del args

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        server = self.server
        server.seen.append(
            {"path": self.path, "headers": dict(self.headers), "body": body}
        )
        mode = server.mode
        if mode == "hang":
            time.sleep(30)
            return
        if mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if mode in ("e401", "e500"):
            status = 401 if mode == "e401" else 500
            # Providers echo what they were sent; rsi-agent must not.
            message = f"Incorrect API key provided: {API_KEY}"
            self._json(status, {"error": {"message": message, "type": "x"}})
            return
        text = json.dumps({"verdict": "clear"}) if "text" in body else "plain answer"
        response = {
            "id": "resp_1",
            "object": "response",
            "status": "completed",
            "model": body.get("model"),
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                },
            ],
        }
        if mode == "failed":
            response["status"] = "failed"
            response["error"] = {"message": "the model is overloaded"}
        if mode == "incomplete":
            response["status"] = "incomplete"
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        if mode == "refusal":
            response["output"][1]["content"] = [
                {"type": "refusal", "refusal": "I cannot help with that"}
            ]
        if mode == "json" or not body.get("stream"):
            self._json(200, response)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()

        def event(kind: str, **fields) -> None:
            payload = json.dumps({"type": kind, **fields})
            self.wfile.write(f"event: {kind}\ndata: {payload}\n\n".encode())
            self.wfile.flush()

        event("response.created", response={**response, "status": "in_progress"})
        for half in (text[: len(text) // 2], text[len(text) // 2 :]):
            event("response.output_text.delta", item_id="msg_1", delta=half)
        if mode == "truncated":
            self.close_connection = True
            return
        if mode == "stream_error":
            event("error", message="stream broke")
            self.close_connection = True
            return
        final = {
            "failed": "response.failed",
            "incomplete": "response.incomplete",
        }.get(mode, "response.completed")
        event(final, response=response)
        self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802
        self.server.seen.append(
            {"path": self.path, "headers": dict(self.headers), "body": None}
        )
        self._json(200, {})

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _Proxy(_Responses):
    """A plain-HTTP forward proxy that answers itself: it sees absolute URLs."""


@pytest.fixture
def responses():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Responses)
    server.daemon_threads = True
    server.mode = "ok"
    server.seen = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def _pin(endpoints: tuple[str, ...]) -> tuple[PinnedEndpoint, ...]:
    import ipaddress
    from urllib.parse import urlsplit

    pinned = []
    for url in endpoints:
        parsed = urlsplit(url)
        pinned.append(
            PinnedEndpoint(
                hostname=parsed.hostname,
                port=parsed.port or 443,
                addresses=(ipaddress.ip_address("127.0.0.1"),),
            )
        )
    return tuple(pinned)


def _definition(model: str = "gpt-judge", effort: str = "high"):
    return SimpleNamespace(
        verifier=SimpleNamespace(
            judge_model=model,
            judge_model_effort=effort,
            model_access=False,
            visibility="hidden",
            clean_image=True,
        ),
        agent=SimpleNamespace(name="claude-code", model=None),
    )


class Judge:
    """A state root with a launch spec, as the Harness leaves it in a Judge."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.root = tmp_path / "state"
        self.private = self.root / "private"
        self.private.mkdir(parents=True)
        self.root.chmod(0o711)
        self.private.chmod(0o700)
        self.record = tmp_path / "record.jsonl"
        self.prompt = tmp_path / "prompt.md"
        self.prompt.write_text("Is the submission honest?\n")
        self.schema = tmp_path / "schema.json"
        self.schema.write_text(json.dumps(SCHEMA))
        self.output = tmp_path / "answer.txt"
        self.log = tmp_path / "call.log"

    def write(self, llm: dict | None, secrets: set[str] = frozenset()) -> dict:
        spec = json.loads(
            build_launch_spec(
                None,
                None,
                set(secrets),
                llm=llm,
                allow_missing_landlock_signal=native_signal_compatibility(),
            )
        )
        # The fake app server's record: the only file outside its job a
        # Landlock-confined call may write (the Harness grants none).
        self.record.touch()
        spec["writable"] = [str(self.record)]
        (self.private / "launch.json").write_text(json.dumps(spec))
        return spec

    def api(self, server, *, mode: str = "ok", effort: str = "high", **request) -> dict:
        server.mode = mode
        options = {
            "auth": JudgeLLMAuth.API,
            "api_key_env": "KEY_VARIABLE",
            "base_url": f"http://127.0.0.1:{server.server_address[1]}",
            **request,
        }
        access = resolve_judge_llm(
            _definition(effort=effort),
            JudgeLLMRequest(**options),
            RSILoopConfig(),
            None,
            pin=_pin,
            environ={"KEY_VARIABLE": API_KEY},
        )
        return self.write(dict(access.spec), set(access.secret_values))

    def codex(
        self,
        *,
        mode: str = "ok",
        codex: Path | None = None,
        login: str = JUDGE_LOGIN,
        effort: str = "high",
        **extra,
    ) -> dict:
        if codex is None:
            codex = self.tmp / "bin" / "codex"
            codex.parent.mkdir(exist_ok=True)
            codex.write_text(
                f"#!{sys.executable}\n" + FAKE_APP_SERVER.read_text().split("\n", 1)[1]
            )
            codex.chmod(0o755)
        llm = {
            "kind": "codex-login",
            "model": "gpt-judge",
            "effort": effort,
            "codex": str(codex),
            "auth_json": login,
            "environment": {
                "FAKE_MODE": mode,
                "FAKE_RECORD": str(self.record),
                **extra,
            },
            "config": [],
        }
        return self.write(llm, {REFRESH_TOKEN, "SYNTHETIC-ACCESS-TOKEN-0c3e"})

    def run(self, *args: str, timeout: float = 60, schema: bool = True):
        command = [
            "llm",
            "--prompt-file",
            str(self.prompt),
            "--output",
            str(self.output),
            "--log",
            str(self.log),
        ]
        if schema:
            command += ["--schema", str(self.schema)]
        return subprocess.run(
            [sys.executable, "-I", "-S", "-B", str(PROGRAM), *command, *args],
            env={
                "PATH": "/usr/bin:/bin",
                "RSI_AGENT_STATE_ROOT": str(self.root),
                # A caller's proxy settings never reach the call.
                "HTTP_PROXY": "http://127.0.0.1:9",
                "http_proxy": "http://127.0.0.1:9",
            },
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def calls(self) -> list[dict]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def leaked(self, *values: str) -> list[str]:
        texts = [
            path.read_text(errors="replace")
            for path in (self.output, self.log)
            if path.exists()
        ]
        return [value for value in values for text in texts if value in text]


@pytest.fixture
def judge(tmp_path: Path) -> Judge:
    return Judge(tmp_path)


# -- the API backend ----------------------------------------------------------


def test_api_call_posts_one_strict_schema_request(judge: Judge, responses) -> None:
    judge.api(responses)

    result = judge.run()

    assert result.returncode == 0, result.stderr
    assert json.loads(judge.output.read_text()) == {"verdict": "clear"}
    [seen] = responses.seen
    assert seen["path"] == "/v1/responses"
    assert seen["headers"]["Authorization"] == f"Bearer {API_KEY}"
    body = seen["body"]
    assert body["model"] == "gpt-judge"
    assert body["reasoning"] == {"effort": "high"}
    assert body["store"] is False and body["stream"] is True
    assert body["instructions"] == rsi_agent.LLM_INSTRUCTIONS
    assert body["input"] == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Is the submission honest?\n"}],
        }
    ]
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "rsi_agent_output",
        "schema": SCHEMA,
        "strict": True,
    }
    assert "tools" not in body
    assert not judge.leaked(API_KEY)
    assert API_KEY not in result.stderr
    # The call ran in a private job directory that is gone afterwards.
    assert list((judge.root / "jobs").iterdir()) == []
    assert not [
        path.name
        for path in judge.private.iterdir()
        if path.name != "launch.json" and not path.name.startswith("slot-")
    ]


def test_api_call_without_schema_returns_text(judge: Judge, responses) -> None:
    judge.api(responses, base_url=None)
    spec = json.loads((judge.private / "launch.json").read_text())
    assert spec["llm"]["url"] == "https://api.openai.com/v1/responses"
    judge.api(responses, base_url=f"http://127.0.0.1:{responses.server_address[1]}/v1/")

    result = judge.run(schema=False)

    assert result.returncode == 0, result.stderr
    assert judge.output.read_text() == "plain answer"
    assert "text" not in responses.seen[0]["body"]
    assert responses.seen[0]["path"] == "/v1/responses"


def test_api_call_accepts_a_plain_json_answer(judge: Judge, responses) -> None:
    judge.api(responses, mode="json")

    result = judge.run()

    assert result.returncode == 0, result.stderr
    assert json.loads(judge.output.read_text()) == {"verdict": "clear"}


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("e401", "HTTP 401: Incorrect API key provided: [REDACTED]"),
        ("e500", "HTTP 500"),
        ("failed", "the model call failed: the model is overloaded"),
        ("incomplete", "incomplete: max_output_tokens"),
        ("refusal", "the model refused: I cannot help with that"),
        ("truncated", "stream ended before the answer"),
        ("stream_error", "the model call failed: stream broke"),
    ],
)
def test_api_failures_exit_1_with_a_redacted_reason(
    judge: Judge, responses, mode: str, reason: str
) -> None:
    judge.output.write_text("previous answer")
    judge.api(responses, mode=mode)

    result = judge.run()

    assert result.returncode == rsi_agent.EXIT_AGENT, result.stderr
    assert reason in result.stderr
    assert API_KEY not in result.stderr
    assert not judge.leaked(API_KEY)
    # Written only on success.
    assert judge.output.read_text() == "previous answer"


def test_api_redirects_are_refused_with_the_key(judge: Judge, responses) -> None:
    judge.api(responses, mode="redirect")

    result = judge.run()

    assert result.returncode == rsi_agent.EXIT_AGENT
    assert "redirected (HTTP 302); refused" in result.stderr
    # Nothing followed the redirect: the key went to one URL only.
    assert [entry["path"] for entry in responses.seen] == ["/v1/responses"]


def test_api_unreachable_endpoint_exits_1(judge: Judge, responses) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    judge.api(responses, base_url=f"http://127.0.0.1:{port}")

    result = judge.run()

    assert result.returncode == rsi_agent.EXIT_AGENT
    assert "cannot reach the model API" in result.stderr


def test_api_timeout_exits_124(judge: Judge, responses) -> None:
    judge.api(responses, mode="hang")
    started = time.monotonic()

    result = judge.run("--timeout", "2")

    assert result.returncode == rsi_agent.EXIT_TIMEOUT, result.stderr
    assert "did not finish within 2 seconds" in result.stderr
    assert time.monotonic() - started < 20


def test_api_call_goes_through_the_given_proxy(judge: Judge, responses) -> None:
    proxy = f"http://127.0.0.1:{responses.server_address[1]}"
    judge.api(responses, base_url="http://api.example.invalid", proxy=proxy)
    spec = json.loads((judge.private / "launch.json").read_text())
    assert spec["llm"]["proxy"] == proxy

    result = judge.run()

    assert result.returncode == 0, result.stderr
    # A forward proxy receives the absolute URL.
    assert responses.seen[0]["path"] == "http://api.example.invalid/v1/responses"


# -- usage and availability -------------------------------------------------------


def test_llm_without_judge_model_is_unavailable(judge: Judge) -> None:
    result = judge.run()
    assert result.returncode == rsi_agent.EXIT_UNAVAILABLE
    assert "judge_model" in result.stderr

    judge.write(None)
    result = judge.run()
    assert result.returncode == rsi_agent.EXIT_UNAVAILABLE
    assert "no LLM access" in result.stderr and "judge_model" in result.stderr


def test_exec_in_a_judge_model_only_judge_is_unavailable(
    judge: Judge, responses
) -> None:
    judge.api(responses)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(PROGRAM), "exec"]
        + ["--prompt-file", str(judge.prompt), "--output", str(judge.output)],
        env={"PATH": "/usr/bin:/bin", "RSI_AGENT_STATE_ROOT": str(judge.root)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == rsi_agent.EXIT_UNAVAILABLE
    assert "model_access = true" in result.stderr
    assert responses.seen == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda judge: judge.prompt.write_text("  \n"), "the prompt is empty"),
        (lambda judge: judge.schema.write_text("[1]"), "must be a JSON object"),
        (lambda judge: judge.schema.write_text("{"), "schema is not valid JSON"),
        (lambda judge: judge.prompt.write_bytes(b"\xff\xfe"), "not UTF-8"),
    ],
)
def test_llm_usage_errors_exit_2(judge: Judge, responses, change, message) -> None:
    judge.api(responses)
    change(judge)

    result = judge.run()

    assert result.returncode == rsi_agent.EXIT_USAGE
    assert message in result.stderr
    assert responses.seen == []


def test_llm_has_no_workdir_option_and_needs_a_positive_timeout(
    judge: Judge, responses
) -> None:
    judge.api(responses)
    assert judge.run("--workdir", str(judge.tmp)).returncode == rsi_agent.EXIT_USAGE
    result = judge.run("--timeout", "0")
    assert result.returncode == rsi_agent.EXIT_USAGE
    assert "--timeout" in result.stderr


# -- the Codex backend ----------------------------------------------------------


def test_codex_call_is_one_tool_less_ephemeral_turn(judge: Judge) -> None:
    judge.codex()

    result = judge.run()

    assert result.returncode == 0, result.stderr
    assert json.loads(judge.output.read_text()) == {"verdict": "clear"}
    call = next(entry for entry in judge.calls() if "refusal" in entry)
    # Every tool, skill, app, plugin and project doc off; app-server last.
    flags = call["argv"][:-1]
    assert call["argv"][-1] == "app-server"
    catalog = call["env"]["CODEX_HOME"] + "/rsi-agent-models.json"
    assert flags[0::2] == ["-c"] * (len(CODEX_FLAGS) + 1)
    assert flags[1::2] == [*CODEX_FLAGS, f'model_catalog_json="{catalog}"']
    # The model's only catalog entry: no code mode, no catalog tools.
    [entry] = call["catalog"]["models"]
    assert entry["slug"] == "gpt-judge"
    assert entry["tool_mode"] == "direct"
    assert entry["experimental_supported_tools"] == []
    assert entry["apply_patch_tool_type"] is None
    assert entry["shell_type"] == "disabled"
    assert entry["supports_search_tool"] is False
    assert entry["default_reasoning_level"] == "high"
    initialize, initialized, start, turn = call["received"][:4]
    assert initialize["method"] == "initialize"
    assert initialize["params"]["capabilities"] == {"experimentalApi": True}
    assert initialize["params"]["clientInfo"]["name"] == "rsi_agent"
    assert initialized == {"method": "initialized"}
    assert start["method"] == "thread/start"
    assert start["params"] == {
        "model": "gpt-judge",
        "cwd": call["cwd"],
        "approvalPolicy": "never",
        "sandbox": "read-only",
        "baseInstructions": rsi_agent.LLM_INSTRUCTIONS,
        "ephemeral": True,
        "environments": [],
    }
    assert turn["method"] == "turn/start"
    assert turn["params"] == {
        "threadId": "thr_1",
        "input": [
            {
                "type": "text",
                "text": "Is the submission honest?\n",
                "text_elements": [],
            }
        ],
        "effort": "high",
        "outputSchema": SCHEMA,
    }
    # The server's approval request was refused, not answered.
    assert call["refusal"]["id"] == 900
    assert call["refusal"]["error"]["code"] == -32601
    # A private copy of the login in a private HOME.
    environment = call["env"]
    assert environment["CODEX_HOME"] == environment["HOME"] + "/.codex"
    assert environment["HOME"].startswith(str(judge.root / "jobs"))
    assert call["cwd"].startswith(str(judge.root / "jobs"))
    assert call["login"] == JUDGE_LOGIN and call["login_mode"] == "0o600"
    assert "KEY_VARIABLE" not in environment and "HTTP_PROXY" not in environment
    assert list((judge.root / "jobs").iterdir()) == []


def test_codex_call_without_schema_takes_the_last_message(judge: Judge) -> None:
    judge.codex(mode="plain")

    result = judge.run(schema=False)

    assert result.returncode == 0, result.stderr
    assert judge.output.read_text() == "plain text answer"
    call = next(entry for entry in judge.calls() if "refusal" in entry)
    assert "outputSchema" not in call["received"][3]["params"]


def test_codex_call_answers_with_the_final_answer_not_a_later_message(
    judge: Judge,
) -> None:
    judge.codex(mode="after_final")

    result = judge.run()

    assert result.returncode == 0, result.stderr
    assert json.loads(judge.output.read_text()) == {"verdict": "clear"}


def test_codex_refreshed_login_is_not_written_back(judge: Judge) -> None:
    spec = judge.codex(mode="refresh")

    result = judge.run()

    assert result.returncode == 0, result.stderr
    stored = json.loads((judge.private / "launch.json").read_text())
    assert stored["llm"]["auth_json"] == spec["llm"]["auth_json"] == JUDGE_LOGIN
    assert "REFRESHED" not in (judge.private / "launch.json").read_text()
    assert list((judge.root / "jobs").iterdir()) == []


def _jwt_login(expires: float) -> str:
    import base64

    claims = base64.urlsafe_b64encode(json.dumps({"exp": expires}).encode())
    login = json.loads(JUDGE_LOGIN)
    login["tokens"]["access_token"] = "h." + claims.decode().rstrip("=") + ".s"
    return json.dumps(login)


def test_codex_call_refuses_a_login_that_would_need_a_refresh(judge: Judge) -> None:
    # Codex would refresh within 5 minutes of expiry; the Judge cannot.
    judge.codex(login=_jwt_login(time.time() + 400))

    result = judge.run("--timeout", "200")

    assert result.returncode == rsi_agent.EXIT_UNAVAILABLE, result.stderr
    assert "never refreshes it" in result.stderr
    assert judge.calls() == []

    judge.codex(login=_jwt_login(time.time() + 3600))
    assert judge.run("--timeout", "200").returncode == 0


@pytest.mark.parametrize(
    ("round_left", "code"),
    [(None, rsi_agent.EXIT_UNAVAILABLE), (600, 0), (900, rsi_agent.EXIT_UNAVAILABLE)],
)
def test_codex_call_needs_the_login_only_until_the_round_ends(
    judge: Judge, round_left, code
) -> None:
    # 1000 s left: too short for the default --timeout (1800 s), enough for
    # a round the Harness ends within 600 s (Codex's 300 s window included).
    spec = judge.codex(login=_jwt_login(time.time() + 1000))
    if round_left is not None:
        spec["round_ends_at"] = time.time() + round_left
        (judge.private / "launch.json").write_text(json.dumps(spec))

    result = judge.run()

    assert result.returncode == code, result.stderr
    if code:
        assert "never refreshes it" in result.stderr


def test_codex_prompt_over_the_cli_limit_is_a_usage_error(judge: Judge) -> None:
    judge.codex()
    judge.prompt.write_text("x" * (rsi_agent.CODEX_MAX_INPUT_CHARS + 1))

    result = judge.run()

    assert result.returncode == rsi_agent.EXIT_USAGE, result.stderr
    assert "Codex CLI's limit" in result.stderr
    assert judge.calls() == []


def test_ultra_is_sent_as_max_on_both_backends(judge: Judge, responses) -> None:
    judge.api(responses, effort="ultra")
    assert judge.run().returncode == 0
    assert responses.seen[-1]["body"]["reasoning"] == {"effort": "max"}

    judge.codex(effort="ultra")
    assert judge.run().returncode == 0
    call = next(entry for entry in judge.calls() if "refusal" in entry)
    assert call["received"][3]["params"]["effort"] == "max"


def test_codex_call_refuses_a_judge_with_codex_system_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "etc-codex-config.toml"
    config.write_text('[mcp_servers.x]\ncommand = "/bin/true"\n')
    monkeypatch.setattr(
        rsi_agent, "CODEX_SYSTEM_CONFIG", (str(tmp_path / "absent"), str(config))
    )
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 99\n")
    codex.chmod(0o755)

    with pytest.raises(rsi_agent._WorkerFailure) as failure:
        rsi_agent._codex_call({"codex": str(codex)}, str(tmp_path))
    assert failure.value.code == rsi_agent.EXIT_UNAVAILABLE
    assert str(config) in failure.value.message


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        (rsi_agent.EXIT_AGENT, rsi_agent.EXIT_AGENT),
        (rsi_agent.EXIT_USAGE, rsi_agent.EXIT_USAGE),
        (rsi_agent.EXIT_UNAVAILABLE, rsi_agent.EXIT_UNAVAILABLE),
        (rsi_agent.EXIT_INTERNAL, rsi_agent.EXIT_INTERNAL),
        (-9, rsi_agent.EXIT_AGENT),
        (7, rsi_agent.EXIT_AGENT),
    ],
)
def test_worker_exit_codes_keep_the_exec_contract(code: int, expected: int) -> None:
    failure = rsi_agent.worker_failure(code, "rsi-agent-llm: why\n")

    assert failure.code == expected
    assert failure.message == "why"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("fail_turn", "the model call failed: usage limit reached"),
        ("rpc_error", "codex app-server refused thread/start: bad model"),
        ("exit", "codex app-server exited with status 3 before the answer"),
    ],
)
def test_codex_failures_exit_1(judge: Judge, mode: str, reason: str) -> None:
    judge.codex(mode=mode)

    result = judge.run()

    assert result.returncode == rsi_agent.EXIT_AGENT, result.stderr
    assert reason in result.stderr
    assert not judge.output.exists()


def test_codex_output_on_stderr_is_redacted(judge: Judge) -> None:
    judge.codex(mode="leak")

    result = judge.run()

    assert result.returncode == 0, result.stderr
    assert "login: " in judge.log.read_text()
    assert not judge.leaked(REFRESH_TOKEN, "SYNTHETIC-ACCESS-TOKEN-0c3e")
    assert "[REDACTED]" in judge.log.read_text()


def test_codex_timeout_ends_the_app_server(judge: Judge) -> None:
    judge.codex(mode="hang")

    result = judge.run("--timeout", "3")

    assert result.returncode == rsi_agent.EXIT_TIMEOUT
    time.sleep(0.5)
    survivors = subprocess.run(
        ["pgrep", "-f", str(judge.tmp / "bin" / "codex")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert survivors.stdout.strip() == ""


def test_codex_missing_cli_is_unavailable(judge: Judge) -> None:
    judge.codex(codex=judge.tmp / "absent" / "codex")

    result = judge.run()

    assert result.returncode == rsi_agent.EXIT_UNAVAILABLE
    assert "Codex CLI is not available" in result.stderr


def test_codex_proxy_reaches_the_app_server(judge: Judge) -> None:
    judge.codex(https_proxy="http://proxy.example:3128")

    result = judge.run()

    assert result.returncode == 0, result.stderr
    call = next(entry for entry in judge.calls() if "refusal" in entry)
    assert call["env"]["https_proxy"] == "http://proxy.example:3128"


def test_worker_is_not_a_public_command(judge: Judge) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(PROGRAM), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "{exec,llm,isolation}" in result.stdout
    assert rsi_agent.LLM_WORKER not in result.stdout


def test_program_still_parses_with_python38_grammar() -> None:
    # rsi_agent.py runs under the Judge image's own python3 (3.8 or later).
    import ast

    ast.parse(PROGRAM.read_text(), feature_version=(3, 8))
    assert os.access(FAKE_APP_SERVER, os.R_OK)
