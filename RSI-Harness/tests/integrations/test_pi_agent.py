"""pi: config prelude, never-stop extension, install and refusals."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopAgentAdapter,
    _apply_reasoning_effort,
    rsi_loop_agent_environment,
    rsi_loop_runtime_secret_values,
    validate_agent_config,
    validate_agent_reasoning_effort,
)
from rsi_harness.models import AgentAuthSource, AgentPrepareRequest
from rsi_harness.runtime.images import DockerImageBuilder, _registered_agent_launcher
from rsi_harness.runtime.local_auth import resolve_agent_auth
from rsi_loop.harness.agent import create_agent, versions
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.harness.agent.pi import EXTENSION_PATH, PiAgent
from rsi_loop.harness.backend import ExecResult
from rsi_loop.harness.config import RSILoopConfig
from rsi_loop.visualizer.parsers.agent_output import parse
from tests.factories import make_run_plan

API_KEY = "sk-pi-unit-test-key-2f9c"
BASE_URL = "http://203.0.113.23:8000/v1"
PROMPT = "/tmp/rsi-agent-prompt.md"
NODE = shutil.which("node")
ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "src" / "rsi_loop" / "harness" / "agent" / "pi" / "rsi-never-stop.ts"
HOST = ROOT / "tests" / "fakes" / "pi_extension_host.mjs"


def _node_strips_types() -> bool:
    if NODE is None:
        return False
    version = subprocess.run(
        [NODE, "--version"], capture_output=True, text=True, check=False
    ).stdout
    match = re.match(r"v(\d+)\.(\d+)", version)
    return bool(match) and (int(match[1]), int(match[2])) >= (22, 6)


requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
# The extension is TypeScript; node loads it once it can strip the types.
requires_type_stripping = pytest.mark.skipif(
    not _node_strips_types(), reason="node cannot load TypeScript"
)


def _config(**overrides: object) -> RSILoopConfig:
    settings: dict[str, object] = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": BASE_URL,
        "agent_context_window": "131072",
        "agent_max_output_tokens": "16384",
        **overrides,
    }
    return RSILoopConfig(**settings)


def _agent(**overrides: object) -> PiAgent:
    agent = create_agent("pi", _config(**overrides))
    assert isinstance(agent, PiAgent)
    return agent


def _launch(
    command: str, tmp_path: Path, *, environment: dict[str, str] | None = None
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run *command* with a stub ``pi`` that records its argv, env and stdin.

    With the Agent's real environment, so a prelude that wrote the key's
    value anywhere under the home directory would be caught.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    record = tmp_path / "pi.json"
    stub = stub_bin / "pi"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'env': dict(os.environ), "
        f"'stdin': sys.stdin.read()}}, open({str(record)!r}, 'w'))\n"
    )
    stub.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    agent = _agent()
    if environment is None:
        environment = rsi_loop_agent_environment(agent._config, agent, "pi-test")
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        env={
            **environment,
            "PATH": f"{stub_bin}:{Path(str(NODE)).parent}:/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )
    return completed, record


def _prompt_command(agent: PiAgent, prompt: Path, **kwargs: Any) -> str:
    return agent.format_run_cmd(str(prompt), **kwargs)


def test_environment_names_the_endpoint_for_the_config_and_stays_offline() -> None:
    agent = _agent()

    environment = rsi_loop_agent_environment(agent._config, agent, "pi-test")

    assert environment["RSI_AGENT_API_KEY"] == API_KEY
    assert environment["RSI_AGENT_API_BASE_URL"] == BASE_URL
    assert [k for k, v in environment.items() if v == API_KEY] == ["RSI_AGENT_API_KEY"]
    # OPENAI_* would also enable pi's built-in OpenAI provider.
    assert not any(name.startswith("OPENAI_") for name in environment)
    assert environment["PI_OFFLINE"] == "1"
    assert environment["PI_SKIP_VERSION_CHECK"] == "1"
    assert environment["PI_TELEMETRY"] == "0"
    assert environment["RSI_AGENT_VERSION"] == versions.PI
    # The stop hook's switch comes from the command, never the environment.
    assert "RSI_CONTINUATION_REASON" not in environment
    assert API_KEY in rsi_loop_runtime_secret_values(agent._config)


def test_command_runs_pi_in_print_mode_with_the_extension() -> None:
    command = _agent().format_run_cmd(PROMPT, model="pi-test", cwd="/workspace")

    assert command.splitlines()[-1] == (
        "exec pi --offline -p -e /opt/rsi-harness/pi/rsi-never-stop.ts "
        "--provider rsi --model pi-test </tmp/rsi-agent-prompt.md"
    )
    assert _registered_agent_launcher(_agent()) == "pi"
    assert _agent().resume_cmd is None


def test_continuation_is_exported_only_when_the_stop_hook_is_enabled() -> None:
    agent = _agent()

    plain = agent.format_run_cmd(PROMPT, model="pi-test")
    continued = agent.format_run_cmd(
        PROMPT, model="pi-test", continuation=STOP_HOOK_REASON
    )

    assert "RSI_CONTINUATION_REASON" not in plain
    assert f"export RSI_CONTINUATION_REASON='{STOP_HOOK_REASON}' &&" in continued
    assert continued.splitlines()[-1] == plain.splitlines()[-1]


@requires_node
def test_prelude_writes_the_config_and_feeds_the_prompt_on_stdin(
    tmp_path: Path,
) -> None:
    model = "m 'x' \"y\" $HOME `true` \\ \U0001f600"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Solve the task.\n")
    reason = 'Don\'t stop.\nKeep "working" $HOME `true`.'
    agent = _agent()

    completed, record = _launch(
        _prompt_command(agent, prompt, model=model, continuation=reason), tmp_path
    )

    assert completed.returncode == 0, completed.stderr
    launched = json.loads(record.read_text())
    assert launched["argv"] == [
        "--offline",
        "-p",
        "-e",
        EXTENSION_PATH,
        "--provider",
        "rsi",
        "--model",
        model,
    ]
    # The prompt is never in argv.
    assert launched["stdin"] == "Solve the task.\n"
    assert launched["env"]["RSI_CONTINUATION_REASON"] == reason
    agent_dir = tmp_path / "home" / ".pi" / "agent"
    assert launched["env"]["PI_CODING_AGENT_DIR"] == str(agent_dir)
    models = json.loads((agent_dir / "models.json").read_text())
    assert models == {
        "providers": {
            "rsi": {
                "baseUrl": BASE_URL,
                "api": "openai-completions",
                # A reference pi resolves from the environment at request time.
                "apiKey": "$RSI_AGENT_API_KEY",
                "compat": {"supportsDeveloperRole": False, "supportsStore": False},
                "models": [
                    {
                        "id": model,
                        "name": model,
                        "contextWindow": 131072,
                        "maxTokens": 16384,
                        "reasoning": False,
                    }
                ],
            }
        }
    }
    settings = json.loads((agent_dir / "settings.json").read_text())
    assert settings == agent.settings()
    assert settings["compaction"] == {"enabled": True}
    assert settings["httpIdleTimeoutMs"] == 600_000
    assert settings["retry"]["maxRetries"] == 5
    assert settings["enableInstallTelemetry"] is False
    assert settings["extensions"] == ["-builtin:llama.cpp", "-builtin:mcp"]
    for path in (tmp_path / "home").rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text()


@requires_node
def test_pi_never_starts_without_the_endpoint(tmp_path: Path) -> None:
    agent = _agent()
    environment = rsi_loop_agent_environment(agent._config, agent, "pi-test")
    del environment["RSI_AGENT_API_BASE_URL"]

    prompt = tmp_path / "prompt.md"
    prompt.write_text("Solve the task.")

    completed, record = _launch(
        _prompt_command(agent, prompt, model="pi-test"),
        tmp_path,
        environment=environment,
    )

    assert completed.returncode != 0
    assert "RSI_AGENT_API_BASE_URL must be set" in completed.stderr
    assert not record.exists()


@requires_node
@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high"])
def test_reasoning_effort_enables_reasoning_at_that_level(
    effort: str, tmp_path: Path
) -> None:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Solve the task.")
    agent = _agent()
    command = _prompt_command(agent, prompt, model="pi-test")

    completed, record = _launch(
        _apply_reasoning_effort(agent, command, effort), tmp_path
    )

    assert completed.returncode == 0, completed.stderr
    argv = json.loads(record.read_text())["argv"]
    assert argv[:2] == ["--thinking", effort]
    models = json.loads(
        (tmp_path / "home" / ".pi" / "agent" / "models.json").read_text()
    )
    assert models["providers"]["rsi"]["models"][0]["reasoning"] is True


def test_reasoning_effort_outside_the_declared_set_is_refused() -> None:
    validate_agent_reasoning_effort("pi", "minimal")
    with pytest.raises(SetupError, match="choose one of: high, low, medium, minimal"):
        validate_agent_reasoning_effort("pi", "xhigh")
    with pytest.raises(ValueError, match="cannot accept reasoning effort"):
        _agent().apply_reasoning_effort("exec pi --offline -p", "high")


def test_work_image_installs_pinned_pi_with_the_extension() -> None:
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=_agent(),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert (
        f"npm install -g --ignore-scripts @earendil-works/pi-coding-agent@{versions.PI}"
        in dockerfile
    )
    assert "sudo -E" not in dockerfile
    assert f'"{EXTENSION_PATH}"' in dockerfile
    assert f"chmod 0644 {EXTENSION_PATH}" in dockerfile
    assert "command -v pi" in dockerfile
    assert "rsi-agent-driver" not in dockerfile


class _Backend:
    def __init__(self, result: ExecResult) -> None:
        self.result = result
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def exec_run(self, handle: Any, command: Any, **kwargs: Any) -> ExecResult:
        self.calls.append((command, kwargs))
        return self.result


def test_install_stop_hook_checks_pi_and_the_extension_as_the_agent(
    tmp_path: Path,
) -> None:
    backend = _Backend(ExecResult(output="1.1.0\n", exit_code=0))

    _agent().install_stop_hook(
        backend,
        object(),
        tmp_path,
        logging.getLogger(__name__),  # type: ignore[arg-type]
    )

    [(command, kwargs)] = backend.calls
    assert command == ["/bin/sh", "-c", f"test -r {EXTENSION_PATH} && pi --version"]
    # As the Agent's own user: no root override.
    assert "user" not in kwargs
    # Nothing is written into Work: the extension is baked into the image.
    assert list(tmp_path.iterdir()) == []


def test_install_stop_hook_refuses_an_unusable_extension(tmp_path: Path) -> None:
    backend = _Backend(ExecResult(output="permission denied", exit_code=1))

    with pytest.raises(RuntimeError, match="not usable by the Agent"):
        _agent().install_stop_hook(
            backend,
            object(),
            tmp_path,
            logging.getLogger(__name__),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"agent_api_base_url": None}, "set RSI_AGENT_API_BASE_URL"),
        ({"agent_api_key": None}, "set RSI_AGENT_API_KEY"),
    ],
)
def test_check_config_requires_an_endpoint_and_key(
    overrides: dict[str, object], missing: str
) -> None:
    with pytest.raises(SetupError, match=missing):
        validate_agent_config(_config(**overrides), "pi")


def test_check_config_requires_the_model_limits() -> None:
    with pytest.raises(ValueError, match="RSI_AGENT_CONTEXT_WINDOW is not set"):
        _agent(agent_context_window=None).check_config()


def test_a_run_without_a_model_is_refused_before_launch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a model"):
        _agent().format_run_cmd(PROMPT)

    plan = make_run_plan(tmp_path)
    agent_plan = plan.task.agent.model_copy(update={"name": "pi"})
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"agent": agent_plan})}
    )
    with pytest.raises(SetupError, match="set RSI_AGENT_MODEL or --model"):
        RSILoopAgentAdapter(_config()).prepare(
            AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path / "prompt.md")
        )


def test_local_login_is_refused() -> None:
    with pytest.raises(SetupError, match="not supported for Agent 'pi'"):
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL, agent_name="pi", agent_api_key=None
        )


# ---------------------------------------------------------------------------
# The extension, loaded by a stand-in for pi
# ---------------------------------------------------------------------------


def _assistant(*content: dict[str, Any], stop: str = "stop", **extra: Any):
    message = {
        "role": "assistant",
        "model": "fake-model",
        "content": list(content),
        "stopReason": stop,
        "usage": {"input": 100, "output": 10, "cacheRead": 0, "cacheWrite": 0},
        **extra,
    }
    return {"type": "message_end", "message": message}


def _user(text: str) -> dict[str, Any]:
    return {
        "type": "message_end",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _settle(outcome: str = "completed") -> dict[str, Any]:
    return {"type": "agent_before_settle", "outcome": outcome}


_START = {"type": "session_start", "reason": "startup"}
_TOOL_TURN = [
    _user("Solve the task."),
    _assistant(
        {"type": "thinking", "thinking": "plan"},
        {"type": "text", "text": "Let me look."},
        {
            "type": "toolCall",
            "id": "c1",
            "name": "bash",
            "arguments": {"command": "ls"},
        },
        {"type": "toolCall", "id": "c2", "name": "read", "arguments": {"path": "/w/a"}},
        stop="toolUse",
    ),
    {
        "type": "message_end",
        "message": {
            "role": "toolResult",
            "toolCallId": "c1",
            "toolName": "bash",
            "content": [{"type": "text", "text": "answer.txt\n"}],
            "isError": False,
        },
    },
    {
        "type": "message_end",
        "message": {
            "role": "toolResult",
            "toolCallId": "c2",
            "toolName": "read",
            "content": [{"type": "text", "text": "x" * 70_000}],
            "isError": True,
        },
    },
    _assistant({"type": "text", "text": "Done."}),
]


@dataclass
class _HostRun:
    returncode: int
    events: list[dict[str, Any]]
    host: list[dict[str, Any]]
    stderr: str


def _host(
    tmp_path: Path,
    scenario: dict[str, Any],
    *,
    reason: str | None = STOP_HOOK_REASON,
    timings: dict[str, int] | None = None,
) -> _HostRun:
    """Run the extension under the stand-in host with *scenario*."""
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    environment = {
        "PATH": "/usr/bin:/bin",
        "RSI_AGENT_VERSION": "1.1.0",
        "RSI_AGENT_DRIVER_TIMINGS": json.dumps(timings or {"backoffBaseMs": 1}),
    }
    if reason is not None:
        environment["RSI_CONTINUATION_REASON"] = reason
    completed = subprocess.run(
        [str(NODE), "--experimental-strip-types", str(HOST), str(EXTENSION), str(path)],
        env=environment,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    (tmp_path / "agent_output.txt").write_text(completed.stdout)
    host, stderr = [], []
    for line in completed.stderr.splitlines():
        try:
            host.append(json.loads(line)["host"])
        except (json.JSONDecodeError, KeyError, TypeError):
            stderr.append(line)
    return _HostRun(
        returncode=completed.returncode,
        events=[json.loads(line) for line in completed.stdout.splitlines()],
        host=host,
        stderr="\n".join(stderr),
    )


def _results(run: _HostRun, event: str) -> list[Any]:
    return [
        h["value"] for h in run.host if h["kind"] == "result" and h["event"] == event
    ]


@requires_type_stripping
def test_extension_continues_with_a_follow_up_user_message(tmp_path: Path) -> None:
    run = _host(
        tmp_path,
        {
            "events": [
                _START,
                *_TOOL_TURN,
                _settle(),
                _assistant({"type": "text", "text": "Still done."}),
                _settle(),
            ]
        },
    )

    assert run.returncode == 0, run.stderr
    # Queued as a real user message, not a custom_message entry: that is what
    # resets pi's one-shot overflow recovery.
    send = {
        "kind": "send",
        "text": STOP_HOOK_REASON,
        "options": {"deliverAs": "followUp"},
    }
    assert [h for h in run.host if h["kind"] == "send"] == [send, send]
    assert _results(run, "agent_before_settle") == [{"continue": True}] * 2
    init, *events = run.events
    assert init == {
        "type": "system",
        "subtype": "init",
        "agent": "pi",
        "agent_version": "1.1.0",
        "model": "fake-model",
        "session_id": "sess-pi-1",
        "cwd": "/workspace",
        "tools": ["Read", "Bash", "Edit", "Write"],
        "thinking_level": "off",
        "permissionMode": "bypassPermissions",
        "rsi_extension": "rsi-never-stop",
        "rsi_stop_hook": True,
    }
    assert all(e["session_id"] == "sess-pi-1" for e in run.events)
    kinds = [(e["type"], e.get("subtype")) for e in events]
    assert kinds == [
        ("user", None),
        ("assistant", None),
        ("user", None),
        ("user", None),
        ("assistant", None),
        ("result", "turn_end"),
        ("user", None),
        ("assistant", None),
        ("result", "turn_end"),
        ("user", None),
        ("result", "agent_exit"),
    ]
    assert events[0]["message"]["content"] == [
        {"type": "text", "text": "Solve the task."}
    ]
    assert events[1]["message"]["content"] == [
        {"type": "thinking", "thinking": "plan"},
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "c1", "name": "Bash", "input": {"command": "ls"}},
        {
            "type": "tool_use",
            "id": "c2",
            "name": "Read",
            "input": {"path": "/w/a", "file_path": "/w/a"},
        },
    ]
    [bash_result] = events[2]["message"]["content"]
    assert bash_result == {
        "type": "tool_result",
        "tool_use_id": "c1",
        "content": "answer.txt\n",
        "is_error": False,
    }
    [read_result] = events[3]["message"]["content"]
    assert read_result["is_error"] is True
    assert len(read_result["content"]) < 70_000
    assert "characters truncated" in read_result["content"]
    assert events[5]["turn"] == 1 and events[5]["stop_reason"] == "stop"
    # The model gets the bare reason; the transcript marks it as a Stop hook.
    continuations = [e for e in events if e.get("rsi_continuation")]
    assert [e["rsi_continuation"] for e in continuations] == [1, 2]
    for event in continuations:
        assert event["message"]["content"] == [
            {"type": "text", "text": f"Stop hook feedback:\n{STOP_HOOK_REASON}"}
        ]
    assert events[-1] == {
        "type": "result",
        "subtype": "agent_exit",
        "session_id": "sess-pi-1",
        "is_error": False,
        "reason": "completed",
        "exit_code": 0,
        "turns": 2,
        "continuations": 2,
    }
    trajectory = parse(tmp_path / "agent_output.txt")
    assert trajectory.meta["model"] == "fake-model"
    assert trajectory.tool_use_count == 2
    assert len(trajectory.markers) == 2
    [read] = [c for e in trajectory.exchanges for c in e.tool_calls if c.name == "Read"]
    assert read.summary == "/w/a" and read.has_result


@requires_type_stripping
def test_extension_without_the_stop_hook_lets_pi_settle(tmp_path: Path) -> None:
    run = _host(tmp_path, {"events": [_START, *_TOOL_TURN, _settle()]}, reason=None)

    assert run.returncode == 0, run.stderr
    assert not [h for h in run.host if h["kind"] == "send"]
    assert _results(run, "agent_before_settle") == [None]
    assert run.events[0]["rsi_stop_hook"] is False
    assert not [e for e in run.events if e.get("rsi_continuation")]
    assert run.events[-2]["subtype"] == "turn_end"
    assert run.events[-1]["reason"] == "completed"
    assert run.events[-1]["continuations"] == 0


@requires_type_stripping
def test_extension_backs_off_and_continues_after_failed_turns(
    tmp_path: Path,
) -> None:
    error = _assistant(stop="error", errorMessage="500 fake internal server error")
    run = _host(
        tmp_path,
        {
            "events": [
                _START,
                _user("Solve the task."),
                error,
                _settle("error"),
                # An empty reply at once counts as failed too.
                _assistant(),
                _settle(),
            ]
        },
    )

    assert run.returncode == 0, run.stderr
    assert _results(run, "agent_before_settle") == [{"continue": True}] * 2
    [model_error] = [e for e in run.events if e.get("subtype") == "model_error"]
    assert model_error["text"] == "500 fake internal server error"
    failed = [e for e in run.events if e.get("subtype") == "turn_error"]
    assert [e["error"]["code"] for e in failed] == ["error", "empty_turn"]
    assert failed[0]["error"]["message"] == "500 fake internal server error"
    assert [e["rsi_continuation"] for e in run.events if "rsi_continuation" in e] == [
        1,
        2,
    ]


@requires_type_stripping
def test_extension_backoff_grows_to_its_cap_and_resets_after_output(
    tmp_path: Path,
) -> None:
    error = [_assistant(stop="error", errorMessage="500"), _settle("error")]
    output = [_assistant({"type": "text", "text": "Progress."}), _settle()]
    run = _host(
        tmp_path,
        {
            "events": [
                _START,
                _user("Solve the task."),
                *error,
                *error,
                *error,
                *output,
                *error,
            ]
        },
        timings={"backoffBaseMs": 200, "backoffMaxMs": 500},
    )

    assert run.returncode == 0, run.stderr
    settles = [
        h
        for h in run.host
        if h["kind"] == "result" and h["event"] == "agent_before_settle"
    ]
    assert [h["value"] for h in settles] == [{"continue": True}] * 5
    # Doubling from the base up to the cap, none after a turn with output,
    # and from the base again after the next failure.
    expected = [200, 400, 500, 0, 200]
    waits = [h["ms"] for h in settles]
    assert all(
        want <= wait < want + 250 for want, wait in zip(expected, waits, strict=True)
    ), waits


@requires_type_stripping
def test_extension_falls_back_to_an_entry_when_nothing_was_queued(
    tmp_path: Path,
) -> None:
    run = _host(
        tmp_path,
        {
            "queueFollowUps": False,
            "events": [_START, _user("Solve the task."), _assistant(), _settle()],
        },
    )

    assert run.returncode == 0, run.stderr
    assert _results(run, "agent_before_settle") == [
        {
            "entries": [
                {
                    "type": "custom_message",
                    "customType": "rsi-never-stop",
                    "content": STOP_HOOK_REASON,
                    "display": False,
                }
            ],
            "continue": True,
        }
    ]
    [continuation] = [e for e in run.events if e.get("rsi_continuation")]
    assert continuation["rsi_fallback"] == "custom_message"


@requires_type_stripping
def test_extension_never_throws_into_pi(tmp_path: Path) -> None:
    run = _host(
        tmp_path,
        {
            "ctxThrows": True,
            "events": [
                _START,
                {"type": "message_end"},
                {"type": "message_end", "message": {"role": "assistant"}},
                {"type": "message_end", "message": {"role": "toolResult"}},
                {"type": "session_compact_failed", "errorMessage": "400"},
                _settle(),
            ],
        },
    )

    # Every handler returned; the run still continues.
    assert run.returncode == 0, run.stderr
    assert _results(run, "agent_before_settle")[0]["continue"] is True
    [compaction] = [e for e in run.events if e.get("subtype") == "compaction"]
    assert compaction["state"] == "failed" and compaction["error"] == "400"
    assert run.events[-1]["subtype"] == "agent_exit"


@requires_type_stripping
def test_extension_writes_to_stdout_while_pi_redirects_it(tmp_path: Path) -> None:
    run = _host(
        tmp_path,
        {"takeOverStdout": True, "events": [_START, _user("Solve the task.")]},
    )

    assert run.returncode == 0, run.stderr
    assert [e["type"] for e in run.events] == ["system", "user", "result"]


@requires_type_stripping
@pytest.mark.parametrize(
    ("code", "reason"), [(143, "signal"), (129, "signal"), (1, "agent_error")]
)
def test_extension_reports_how_pi_exited(
    tmp_path: Path, code: int, reason: str
) -> None:
    run = _host(tmp_path, {"exitCode": code, "events": [_START]})

    assert run.returncode == code
    assert run.events[-1]["subtype"] == "agent_exit"
    assert run.events[-1]["reason"] == reason
    assert run.events[-1]["exit_code"] == code
