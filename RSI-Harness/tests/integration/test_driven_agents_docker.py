"""Real Docker: every Agent runs its real pinned CLI against a fake model.

Each case builds the Agent's real Work image (pinned node and CLI, plus
rsi-agent-driver for a driven Agent or pi's never-stop extension) on the
fixture task, runs a full production run with only iptables faked, and points
the CLI at tests/fakes/fake_model_server.py on the host, reached through the
default bridge gateway: chat completions for the driven Agents and pi, the
Responses API for Codex, the Messages API for Claude Code. The fake answers
with text and, on some turns, a call to the shell tool the CLI offers, running
the commands below in Work, one of them ``rsi-submit``. Codex and Claude Code
keep working through their native stop hook, so a CLI upgrade that breaks the
hook fails here.

Work images are content addressed, so a case reuses the image an earlier case
or run built; only the first build of an Agent pays for the install.

To add a driven Agent, append a ``DrivenCase``. If its shell tool has a name
the fake does not know yet, add it to ``SHELL_TOOLS`` in the fake (or set
``shell_tools`` in the case's scenario). ``checks`` adds calls to the CLI's
other tools, with the text their result must contain.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import ssl
import subprocess
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import CompileOptions, RunRequest, RunResult, RunStatus
from rsi_harness.runtime.production import ProductionRuntimeServices
from rsi_loop.harness.agent import create_agent, versions
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.harness.config import RSILoopConfig
from rsi_loop.visualizer import scanner as visualizer_scanner
from rsi_loop.visualizer.parsers.agent_output import get_trajectory
from tests.fakes import FakeFirewallBackend
from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory
from tests.integrations.test_agent_driver import _load_fake_model_server

pytestmark = pytest.mark.integration

IMAGE = "ubuntu:24.04"
# Set by scripts/bump_agent.py: a case that cannot run fails instead of
# skipping, so a bump never passes on tests that did not run.
REQUIRE_ENV = "RSI_REQUIRE_DOCKER_TESTS"
MODEL = "fake-model"
# Synthetic: accepted only by the fake model server.
API_KEY = "sk-synthetic-driven-agent-key-7d41"
# Long enough for the setup, a Judge round and ten continuations.
AGENT_TIMEOUT_SECONDS = 60
# Model requests answered with text only, so the stop hook continues the CLI
# many times in a row with no tool use in between: more often than a CLI's
# own cap on consecutive stop hook blocks allows (Claude Code's is 8 unless
# the Agent lifts it), so a cap that comes back in an upgrade fails here.
TEXT_STRETCH = range(20, 40)
BLOCKS_IN_A_ROW = 10
# The driver's production backoff starts at 5 s; keep failed turns cheap.
DRIVER_TIMINGS = {"backoffBaseMs": 500, "backoffMaxMs": 2_000}
MARKER = re.compile(r"work-marker agent-user=(\S+) cli=(\d+)")
MAX_OUTPUT_TOKENS = 8192
IDLE_PROBE = "kimi-wire:"
# Anchored: some CLIs echo the command in the tool result.
KEY_FILES = re.compile(r"^key-files:\[(.*)\]$", re.MULTILINE)
WIRE_FILE = re.compile(r"wire-file (\S+) turn-ends=(\d+)")
# The hosts ZCode logged for its provider endpoint routing table, which it
# fetches only for an https endpoint, and whether a fetch succeeded.
ZCODE_ROUTING = (
    'echo "routing-hosts:[$(cat "$HOME"/.zcode/cli/log/*.jsonl '
    '| grep -oE \'"configHost":"[^"]*"|routing\\.snapshot_updated\' '
    "| sort -u | tr '\\n' ' ')]\""
)


@dataclass(frozen=True)
class DrivenCase:
    agent: str
    # Extra fake-model scenario keys, e.g. ``shell_tools``.
    scenario: Mapping[str, Any] = field(default_factory=dict)
    # A reasoning effort the one-turn run asks for; it must reach the model
    # as the request's ``reasoning_effort``.
    reasoning_effort: str | None = None
    # Further tool calls for the main run, after rsi-submit, as fake-model
    # ``commands`` entries, with a text each one's result must contain.
    checks: tuple[tuple[Any, str], ...] = ()
    # The transcript's last line: the driver's, or pi's extension's.
    exit_event: str = "driver_exit"
    # Serve the fake model over https (self-signed), for a CLI that treats
    # an https endpoint differently.
    tls: bool = False
    # How the continuation reaches the model as the last user text: the
    # stop hook reason alone, or framed by the CLI's native stop hook. A
    # regular expression matched at the start of the text.
    continuation: str = re.escape(STOP_HOOK_REASON) + r"\Z"
    # What agent_output.txt holds: the driver's (or pi extension's)
    # JSON-lines transcript, Claude Code's stream-json, or Codex's text.
    transcript: str = "driver"
    # The CLI sends RSI_AGENT_MAX_OUTPUT_TOKENS as the request's limit.
    output_limit: bool = True
    # The path RSI_AGENT_API_BASE_URL ends in, as a real deployment sets it
    # for this CLI, and the one path its model requests must reach.
    base_path: str = "/v1"
    api_path: str = "/v1/chat/completions"

    def __str__(self) -> str:
        return self.agent


CASES = (
    # Native stop hooks: one CLI process, continued by the hook the Agent
    # installs, which frames the reason in its own text.
    DrivenCase(
        "codex",
        # Its shell returns a session to poll instead of the output when a
        # command runs longer than this, as a Judge round may.
        scenario={"shell_arguments": {"yield_time_ms": 30_000}},
        reasoning_effort="high",
        continuation=(
            r"<hook_prompt [^>]*>" + re.escape(STOP_HOOK_REASON) + r"</hook_prompt>\Z"
        ),
        transcript="codex",
        output_limit=False,
        api_path="/v1/responses",
    ),
    DrivenCase(
        "claude-code",
        # Not its default, so the request shows the flag reached it.
        reasoning_effort="low",
        checks=(
            (
                {"tool": "Read", "arguments": {"file_path": "/workspace/answer.txt"}},
                "42",
            ),
        ),
        # After an empty reply the CLI repeats the request with a nudge
        # appended to the same text.
        continuation=r"Stop hook feedback:\n" + re.escape(STOP_HOOK_REASON),
        transcript="claude",
        output_limit=False,
        # Its base URL has no version path; the CLI adds /v1 itself.
        base_path="",
        api_path="/v1/messages",
    ),
    DrivenCase("qwen-code"),
    DrivenCase("kimi-code", reasoning_effort="high"),
    DrivenCase(
        "opencode",
        reasoning_effort="high",
        # Its grep tool needs ripgrep, which it would otherwise download on
        # first use; the no-network Work must find the baked-in copy.
        checks=(({"tool": "grep", "arguments": {"pattern": "^42$"}}, "answer.txt"),),
    ),
    DrivenCase(
        "zcode",
        reasoning_effort="high",
        checks=(
            # Its file tools run in Work too, not only its shell.
            (
                {"tool": "Read", "arguments": {"file_path": "/workspace/answer.txt"}},
                "42",
            ),
            # Its routing table fetch failed locally instead of reaching the
            # vendor's origin, so the model requests went out unrouted.
            (
                {"tool": "Bash", "arguments": {"command": ZCODE_ROUTING}},
                'routing-hosts:["configHost":"127.0.0.1:9" ]',
            ),
        ),
        tls=True,
    ),
    # Not driven: its own extension continues it and writes the transcript.
    DrivenCase(
        "pi",
        reasoning_effort="high",
        checks=(
            ({"tool": "read", "arguments": {"path": "/workspace/answer.txt"}}, "42"),
        ),
        exit_event="agent_exit",
    ),
)


@dataclass
class DrivenRun:
    case: DrivenCase
    result: RunResult
    run_root: Path
    requests: list[dict[str, Any]]

    @property
    def agent_output(self) -> str:
        path = self.run_root / "agent_output.txt"
        # A run that failed during setup has none; its result says why.
        return path.read_text() if path.exists() else ""

    def continuations(self) -> list[dict[str, Any]]:
        continuation = re.compile(self.case.continuation)
        return [
            r
            for r in self.requests
            if r["last_role"] == "user" and continuation.match(r["last_user"] or "")
        ]

    def tool_results(self) -> list[str]:
        return [r["last_tool"] for r in self.requests if r["last_tool"]]


def _unavailable(reason: str) -> None:
    if os.environ.get(REQUIRE_ENV):
        pytest.fail(f"{reason} ({REQUIRE_ENV} is set)")
    pytest.skip(reason)


def _docker():
    try:
        client = docker.from_env()
        client.ping()
        client.images.get(IMAGE)
    except Exception as error:
        _unavailable(f"local cached {IMAGE} Docker authority unavailable: {error}")
    return client


def _marker(key_env: str = "") -> str:
    # The agent alias exists only in the Work image. $PPID of the shell the
    # CLI starts is the CLI process itself (or its session worker): the same
    # value in every call means one CLI process.
    check = f'; echo "key-check ${key_env}"' if key_env else ""
    same = '$(test "$(id -u)" = "$(id -u agent)" && echo yes || echo no)'
    return f'echo "work-marker agent-user={same} cli=$PPID"{check}'


def _idle_probe_glob(agent_name: str) -> str | None:
    flags = list(getattr(create_agent(agent_name, RSILoopConfig()), "driver_flags", ()))
    if "--idle-probe" not in flags:
        return None
    probe = flags[flags.index("--idle-probe") + 1]
    assert probe.startswith(IDLE_PROBE)
    return probe.removeprefix(IDLE_PROBE)


def _wire_listing(glob: str) -> str:
    # Lists the files the driver's idle probe would read, with the number of
    # turn-end events it parses, so a CLI upgrade that moves or renames them
    # fails here instead of silently degrading to the fixed wait.
    pattern = glob.replace("{session}", "*")
    return (
        f'for f in {pattern}; do echo "wire-file $f '
        'turn-ends=$(grep -c \'"agent.turn.ended"\' "$f")"; done'
    )


def _task(root: Path, *, timeout_seconds: int, dockerfile: str | None = None) -> Path:
    task = root / "task"
    shutil.copytree(FIXTURE, task)
    if dockerfile is not None:
        (task / "environment" / "Dockerfile").write_text(dockerfile)
    config = task / "task.toml"
    text = config.read_text()
    # No-network Work: the CLI reaches only the pinned model endpoint, and
    # an Agent with web tools must run without them.
    text = text.replace(
        '[agent]\ntimeout_sec = 60\nnetwork_mode = "public"',
        f'[agent]\ntimeout_sec = {timeout_seconds}\nnetwork_mode = "no-network"',
    )
    assert f"timeout_sec = {timeout_seconds}" in text
    config.write_text(text)
    return task


def _run(
    tmp_path: Path,
    request: pytest.FixtureRequest,
    case: DrivenCase,
    scenario: dict[str, Any],
    *,
    stop_hook: bool,
    timeout_seconds: int,
    reasoning_effort: str | None = None,
    dockerfile: str | None = None,
) -> DrivenRun:
    """One full production run of ``case`` with the fake model."""
    client = _docker()
    gateway = next(
        item["Gateway"]
        for item in client.networks.get("bridge").attrs["IPAM"]["Config"]
        if item.get("Gateway")
    )
    module = _load_fake_model_server()
    server = module.FakeModelServer(
        tmp_path / "model-requests.jsonl",
        {**scenario, **case.scenario},
        host="0.0.0.0",
    )
    threading.Thread(target=server.httpd.serve_forever, daemon=True).start()
    request.addfinalizer(server.stop)
    port = server.httpd.server_address[1]
    scheme = "http"
    extra_env = {"RSI_AGENT_DRIVER_TIMINGS": json.dumps(DRIVER_TIMINGS)}
    if case.tls:
        scheme = "https"
        server.httpd.socket = _tls_context(tmp_path).wrap_socket(
            server.httpd.socket, server_side=True, do_handshake_on_connect=False
        )
        # The certificate is self-signed and not in Work's trust store.
        extra_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    data = tmp_path / "data"
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        firewall_backend=FakeFirewallBackend(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(
            agent_api_key=API_KEY,
            agent_api_base_url=f"{scheme}://{gateway}:{port}{case.base_path}",
            agent_context_window="200000",
            agent_max_output_tokens=str(MAX_OUTPUT_TOKENS),
            agent_extra_env=extra_env,
        ),
        omit_gpu_device_requests_for_tests=True,
        agent_adapter_factory=lambda config, runtime: RSILoopAgentAdapter(
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
            task_dir=_task(
                tmp_path, timeout_seconds=timeout_seconds, dockerfile=dockerfile
            ),
            agent_name=case.agent,
            model=MODEL,
            reasoning_effort=reasoning_effort,
            gpu_selectors=("0",),
            options=CompileOptions(
                agent_name=case.agent,
                primary_reward="reward",
                disable_stop_hook=not stop_hook,
            ),
        )
    )
    run_root = tmp_path / "logs" / "runs" / result.run_id / "minimal-gpu"
    return DrivenRun(
        case=case, result=result, run_root=run_root, requests=server.requests()
    )


def _tls_context(tmp_path: Path) -> ssl.SSLContext:
    openssl = shutil.which("openssl")
    if openssl is None:
        _unavailable("openssl is not installed")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1"]
        + ["-subj", "/CN=fake-model", "-keyout", str(key), "-out", str(cert)],
        check=True,
        capture_output=True,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return context


def _assert_no_key(run: DrivenRun) -> None:
    assert API_KEY not in run.agent_output
    for path in run.run_root.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text(errors="replace"), path


def _transcript(run: DrivenRun) -> list[dict[str, Any]]:
    events = []
    for line in run.agent_output.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


@pytest.mark.parametrize("case", CASES, ids=str)
def test_driven_agent_keeps_one_session_working_through_a_judge_round(
    tmp_path: Path, request: pytest.FixtureRequest, case: DrivenCase
) -> None:
    key_env = create_agent(case.agent, RSILoopConfig()).api_key_env
    submit = f"printf 42 > /workspace/answer.txt && rsi-submit; {_marker()}"
    commands = [_marker(key_env), submit, _marker(), _marker()]
    wire_glob = _idle_probe_glob(case.agent)
    if wire_glob:
        commands.append(_wire_listing(wire_glob))
    commands += [command for command, _ in case.checks]
    scenario = {
        # A tool call every third request, a text turn end in between.
        "cycle": ["mixed", "text", "text"],
        "at": {
            "8": "e500",
            "12": "empty",
            **{str(n): "text" for n in TEXT_STRETCH},
        },
        "commands": commands,
        # Every later tool call proves the CLI process again, past the
        # injected failures, where a relaunch would show.
        "default_command": _marker(),
    }

    run = _run(
        tmp_path,
        request,
        case,
        scenario,
        stop_hook=True,
        timeout_seconds=AGENT_TIMEOUT_SECONDS,
    )

    output = run.agent_output
    assert run.result.status == RunStatus.COMPLETED, (run.result, output[-4000:])
    # A Judge round from rsi-submit inside a tool call, with its report.
    assert run.result.total_rounds == 1
    assert run.result.best_score == 1
    feedback = run.run_root / "feedback" / "agent-1.log"
    assert "answer accepted" in feedback.read_text()

    requests = run.requests
    actions = {r["n"]: r["action"] for r in requests}
    assert actions[8] == "e500" and actions[12] == "empty"
    # Every continuation reached the model as the last user text, after the
    # injected failures too, in one conversation that only ever grew: a new
    # session or a relaunched CLI would start over from the task prompt.
    continuations = run.continuations()
    assert len(continuations) >= 10, [r["last_user"] for r in requests[:20]]
    assert sum(r["n"] > 12 for r in continuations) >= 5
    # More stop hook blocks in a row than a CLI's own cap would allow, all
    # in the same process (checked below by the markers after them).
    assert _longest_text_stretch(run) >= BLOCKS_IN_A_ROW, [
        (r["n"], r["last_role"]) for r in requests[15:40]
    ]
    assert "consecutive times" not in output
    counts = [r["n_messages"] for r in requests]
    assert counts == sorted(counts)
    assert all(r["authorization"] and r["model"] == MODEL for r in requests)
    # The CLI reached the model where a real endpoint serves it.
    assert {r["path"] for r in requests} == {case.api_path}
    if case.output_limit:
        assert {r["max_tokens"] for r in requests} == {MAX_OUTPUT_TOKENS}

    # The tool calls ran in Work as the Agent user, all under one CLI process,
    # before and after the injected failures.
    results = "\n".join(run.tool_results())
    marked = [
        (r["n"], marker)
        for r in requests
        if r["last_tool"]
        for marker in MARKER.findall(r["last_tool"])
    ]
    markers = [marker for _, marker in marked]
    assert len(markers) >= 3, results[-4000:]
    assert {same for same, _ in markers} == {"yes"}
    assert len({pid for _, pid in markers}) == 1
    assert max(n for n, _ in marked) > 12, marked
    # rsi-submit's report came back to the model as the tool result.
    assert "round: agent-1\nstatus: completed" in results
    for command, expected in case.checks:
        # The request after the call carries its result.
        [call] = [
            r["n"]
            for r in requests
            if r["call"]
            and r["call"]["name"] == command["tool"]
            and command["arguments"].items()
            <= json.loads(r["call"]["arguments"]).items()
        ]
        [answer] = [r for r in requests if r["n"] == call + 1]
        assert expected in (answer["last_tool"] or ""), answer

    received = {r["n_messages"] for r in continuations}
    {
        "driver": _assert_driver_session,
        "claude": _assert_claude_session,
        "codex": _assert_codex_session,
    }[case.transcript](run, received, results)

    _assert_no_key(run)
    assert "key-check [REDACTED]" in output
    # The visualizer reads the transcript and names the Agent and model of a
    # live run.
    trajectory = get_trajectory(run.run_root / "agent_output.txt")
    assert trajectory is not None
    assert trajectory.tool_use_count >= 3
    if case.transcript != "codex":
        # Codex's text output has no stop hook markers to show.
        assert len(trajectory.markers) >= 10
    assert visualizer_scanner._peek_agent(run.run_root) == case.agent
    assert visualizer_scanner._peek_model(run.run_root) == MODEL


def _longest_text_stretch(run: DrivenRun) -> int:
    """The most continuations that reached the model with no tool use between."""
    continuations = {r["n"] for r in run.continuations()}
    longest = stretch = 0
    for r in run.requests:
        if r["last_role"] == "tool":
            stretch = 0
        elif r["n"] in continuations:
            stretch += 1
            longest = max(longest, stretch)
    return longest


def _assert_driver_session(run: DrivenRun, received: set[int], results: str) -> None:
    """One driver (or pi extension) session, as its transcript records it."""
    case = run.case
    events = _transcript(run)
    inits = [
        e for e in events if e.get("type") == "system" and e.get("subtype") == "init"
    ]
    assert len(inits) == 1 and inits[0]["agent"] == case.agent
    session_ids = {e["session_id"] for e in events if e.get("session_id")}
    assert session_ids == {inits[0]["session_id"]}
    # Each continuation the transcript records reached the model as a new
    # user message (a retried request repeats one), give or take the last:
    # pi's extension queued every one as a real follow-up, never its
    # fallback entry, which would lose pi's overflow recovery.
    assert not [e for e in events if "rsi_fallback" in e]
    recorded = [e for e in events if e.get("rsi_continuation")]
    assert abs(len(recorded) - len(received)) <= 1, (len(recorded), len(received))
    exits = [e for e in events if e.get("subtype") == case.exit_event]
    assert [e["reason"] for e in exits] in ([], ["signal"])
    wire_glob = _idle_probe_glob(case.agent)
    if wire_glob:
        # The idle probe finds this session's event file in the real CLI.
        expected = re.sub(r"\$\{?\w+\}?", "*", wire_glob).replace(
            "{session}", inits[0]["session_id"]
        )
        wires = WIRE_FILE.findall(results)
        ends = [int(n) for path, n in wires if fnmatch.fnmatch(path, expected)]
        assert ends and min(ends) > 0, (expected, wires)
    trajectory = get_trajectory(run.run_root / "agent_output.txt")
    assert trajectory is not None
    # The visualizer names the Agent, in the trajectory and for a live run.
    version = inits[0].get("agent_version")
    assert version
    assert trajectory.meta["agent"] == case.agent
    assert f"agent: {case.agent} {version}" in trajectory.exchanges[0].text


def _assert_claude_session(run: DrivenRun, received: set[int], results: str) -> None:
    """One Claude Code session of the pinned version, continued by its hook."""
    events = _transcript(run)
    inits = [
        e for e in events if e.get("type") == "system" and e.get("subtype") == "init"
    ]
    assert len(inits) == 1, len(inits)
    assert inits[0]["claude_code_version"] == versions.CLAUDE_CODE
    session_ids = {e["session_id"] for e in events if e.get("session_id")}
    assert session_ids == {inits[0]["session_id"]}
    # The hook's feedback, which the stream records as a synthetic user
    # message, each time it reached the model (give or take the last).
    recorded = [
        e
        for e in events
        if e.get("type") == "user"
        and e.get("isSynthetic")
        and re.match(run.case.continuation, _event_text(e))
    ]
    assert abs(len(recorded) - len(received)) <= 1, (len(recorded), len(received))
    trajectory = get_trajectory(run.run_root / "agent_output.txt")
    assert trajectory is not None
    assert trajectory.meta["claude_code_version"] == versions.CLAUDE_CODE


def _assert_codex_session(run: DrivenRun, received: set[int], results: str) -> None:
    """One Codex session of the pinned version, continued by its hook."""
    output = run.agent_output
    assert output.count("\nsession id: ") == 1, output[:4000]
    assert output.count("hook: Stop Blocked") >= len(received) - 1
    trajectory = get_trajectory(run.run_root / "agent_output.txt")
    assert trajectory is not None
    assert trajectory.meta["codex_version"] == f"OpenAI Codex v{versions.CODEX}"
    assert trajectory.meta["model"] == MODEL
    # The provider Codex names at launch for a configured endpoint.
    assert trajectory.meta["provider"] == "rsi_proxy"


def _event_text(event: dict[str, Any]) -> str:
    content = (event.get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get("text", "")) for part in content or () if isinstance(part, dict)
    )


@pytest.mark.parametrize("case", CASES, ids=str)
def test_driven_agent_without_stop_hook_runs_one_turn(
    tmp_path: Path, request: pytest.FixtureRequest, case: DrivenCase
) -> None:
    key_env = create_agent(case.agent, RSILoopConfig()).api_key_env
    # Lists the files that hold the key's value: on Work's root filesystem,
    # which a Judge's snapshot captures, and under $HOME, /workspace and /tmp,
    # which may be mounts of their own. A tmpfs such as /dev/shm is neither.
    # Only here: the main test's key echo lands in the CLI's own session
    # history under $HOME.
    key_files = (
        'echo "key-files:[$({ find / -xdev -type f; '
        'find "$HOME" /workspace /tmp -type f; } 2>/dev/null | sort -u '
        f"| tr '\\n' '\\0' | xargs -0 grep -lF -- \"${key_env}\" 2>/dev/null "
        "| tr '\\n' ' ')]\""
    )
    command = (
        f"printf 42 > /workspace/answer.txt && rsi-submit; {_marker()}; {key_files}"
    )
    scenario = {"cycle": ["mixed", "text"], "commands": [command]}

    run = _run(
        tmp_path,
        request,
        case,
        scenario,
        stop_hook=False,
        timeout_seconds=AGENT_TIMEOUT_SECONDS,
        reasoning_effort=case.reasoning_effort,
    )

    output = run.agent_output
    assert run.result.status == RunStatus.COMPLETED, (run.result, output[-4000:])
    assert run.result.total_rounds == 1
    results = "\n".join(run.tool_results())
    assert "round: agent-1\nstatus: completed" in results, results[-4000:]
    assert [same for same, _ in MARKER.findall(results)] == ["yes"]
    assert KEY_FILES.findall(results) == [""], results[-4000:]
    assert not run.continuations()
    user_turns = [r for r in run.requests if r["last_role"] == "user"]
    assert len(user_turns) == 1
    assert {r["reasoning_effort"] for r in run.requests} == {case.reasoning_effort}
    events = _transcript(run)
    if case.transcript == "driver":
        [exit_event] = [e for e in events if e.get("subtype") == case.exit_event]
        assert exit_event["reason"] == "completed" and exit_event["exit_code"] == 0
        assert exit_event["continuations"] == 0
    elif case.transcript == "claude":
        [result] = [e for e in events if e.get("type") == "result"]
        assert result["subtype"] == "success" and not result["is_error"], result
    else:
        assert "hook: Stop" not in output
    # The Agent ended by itself: the run did not wait for its timeout.
    final = json.loads((run.run_root / "final_result.json").read_text())
    assert final.get("timed_out") is False, final
    _assert_no_key(run)
    trajectory = get_trajectory(run.run_root / "agent_output.txt")
    assert trajectory is not None and trajectory.tool_use_count == 1


# A Debian task image (whose /etc/profile resets PATH in a login shell) with a
# tool reachable only through its own ENV PATH.
DEBIAN_IMAGE = "python:3.12-slim-bookworm"
DEBIAN_DOCKERFILE = f"""\
FROM {DEBIAN_IMAGE}
RUN mkdir -p /workspace /opt/task-tools/bin \\
    && printf '#!/bin/sh\\necho image-path-tool\\n' > /opt/task-tools/bin/task-tool \\
    && chmod 0755 /opt/task-tools/bin/task-tool
ENV PATH="/opt/task-tools/bin:${{PATH}}"
WORKDIR /workspace
"""


@pytest.mark.parametrize("case", CASES, ids=str)
def test_agent_shell_tool_keeps_the_image_path_on_debian(
    tmp_path: Path, request: pytest.FixtureRequest, case: DrivenCase
) -> None:
    """The commands an Agent's shell tool runs find the image's ENV PATH.

    The Agent starts outside a login shell; a CLI whose own shell tool used
    one would lose the image's PATH to Debian's /etc/profile here.
    """
    client = _docker()
    try:
        client.images.get(DEBIAN_IMAGE)
    except Exception as error:
        _unavailable(f"local cached {DEBIAN_IMAGE} unavailable: {error}")
    finally:
        client.close()
    command = (
        "printf 42 > /workspace/answer.txt && rsi-submit; "
        'task-tool; echo "agent-path:[$PATH]"'
    )
    scenario = {"cycle": ["mixed", "text"], "commands": [command]}

    run = _run(
        tmp_path,
        request,
        case,
        scenario,
        stop_hook=False,
        timeout_seconds=AGENT_TIMEOUT_SECONDS,
        dockerfile=DEBIAN_DOCKERFILE,
    )

    output = run.agent_output
    assert run.result.status == RunStatus.COMPLETED, (run.result, output[-4000:])
    results = "\n".join(run.tool_results())
    assert "round: agent-1\nstatus: completed" in results, results[-4000:]
    assert "image-path-tool" in results, results[-4000:]
    assert re.search(r"agent-path:\[[^\]]*/opt/task-tools/bin", results), results
