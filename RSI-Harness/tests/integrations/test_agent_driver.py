"""rsi-agent-driver against scripted ACP and ZCode Protocol agents.

The driver is a Node program baked into driven Agents' Work images; these
tests run it on the host with fake agents (tests/fakes/*.mjs) and short
timings, and assert on what the fake agents saw and on the transcript.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from rsi_harness.runtime.docker import _RedactedOutputWriter
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.visualizer.parsers.agent_output import parse

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "src" / "rsi_loop" / "harness" / "agent" / "driver"
DRIVER_SCRIPT = DRIVER / "rsi-agent-driver.mjs"
FAKES = ROOT / "tests" / "fakes"
NODE = shutil.which("node")
PROMPT = "Do the task.\n"
FEEDBACK = f"Stop hook feedback:\n{STOP_HOOK_REASON}"
TIMINGS = {
    "backoffBaseMs": 40,
    "backoffMaxMs": 400,
    "fastTurnMs": 200,
    "queuedTurnMs": 100,
    "queuedWaitMs": 300,
    "idleQuietMs": 150,
    "idlePollMs": 20,
    "setupTimeoutMs": 5_000,
    "zcodeWatchdogMs": 300,
    "zcodeSendTimeoutMs": 2_000,
    "zcodeRetryMs": 20,
    "killGraceMs": 500,
}
FAKE_AGENTS = {"acp": "fake_acp_agent.mjs", "zcode": "fake_zcode_agent.mjs"}

requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")


def _load_fake_model_server() -> Any:
    # tests/fakes.py owns the tests.fakes name, so load the server by path.
    spec = importlib.util.spec_from_file_location(
        "fake_model_server", FAKES / "fake_model_server.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DriverRun:
    def __init__(self, process: subprocess.Popen[bytes], root: Path) -> None:
        self.process = process
        self.root = root
        self.out = root / "driver-stdout.jsonl"
        self.err = root / "driver-stderr.txt"
        self.log = root / "agent-log.jsonl"

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.log.exists():
            return []
        # Polled while the fake agent appends: only lines it finished
        # writing, so a line cut off mid-write is read on the next poll.
        complete = self.log.read_text().split("\n")[:-1]
        events = [json.loads(line) for line in complete if line]
        return [e for e in events if kind is None or e["event"] == kind]

    def prompts(self) -> list[dict[str, Any]]:
        return [e for e in self.events("prompt") if not e.get("queued")]

    def lines(self) -> list[dict[str, Any]]:
        # Only JSON lines may reach stdout, and nothing may reach stderr: the
        # harness merges both into agent_output.txt.
        text = self.out.read_text()
        assert self.err.read_text() == ""
        return [json.loads(line) for line in text.splitlines()]

    def results(self, subtype: str) -> list[dict[str, Any]]:
        return [
            line
            for line in self.lines()
            if line["type"] == "result" and line["subtype"] == subtype
        ]

    def wait_until(self, condition: Callable[[], bool], timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return
            if self.process.poll() is not None and not condition():
                raise AssertionError(
                    f"driver exited {self.process.returncode}:\n{self.out.read_text()}"
                )
            time.sleep(0.02)
        raise AssertionError(f"timed out; transcript:\n{self.out.read_text()}")

    def wait_prompts(self, count: int, timeout: float = 10.0) -> None:
        self.wait_until(lambda: len(self.prompts()) >= count, timeout)

    def wait(self, timeout: float = 10.0) -> int:
        return self.process.wait(timeout=timeout)

    def stop(self, signum: int = signal.SIGTERM) -> int:
        self.process.send_signal(signum)
        return self.wait()

    def agent_pid(self) -> int:
        return self.events("start")[0]["pid"]


@pytest.fixture
def driver(tmp_path: Path) -> Iterator[Callable[..., DriverRun]]:
    runs: list[DriverRun] = []

    def start(
        protocol: str,
        scenario: dict[str, Any],
        *,
        continuation: str | None = STOP_HOOK_REASON,
        flags: tuple[str, ...] = (),
        prompt: str = PROMPT,
        env: dict[str, str] | None = None,
        timings: dict[str, int] | None = None,
        cli: list[str] | None = None,
    ) -> DriverRun:
        root = tmp_path / f"run-{len(runs)}"
        root.mkdir()
        scenario_path = root / "scenario.json"
        scenario_path.write_text(
            json.dumps({**scenario, "log": str(root / "agent-log.jsonl")})
        )
        prompt_path = root / "prompt.md"
        prompt_path.write_text(prompt)
        argv = [
            str(NODE),
            str(DRIVER_SCRIPT),
            "--protocol",
            protocol,
            "--prompt-file",
            str(prompt_path),
            "--agent-name",
            f"fake-{protocol}",
            "--model",
            "fake-model",
            *flags,
        ]
        if continuation is not None:
            argv += ["--continuation-reason", continuation]
        argv += [
            "--",
            *(
                cli
                or [str(NODE), str(FAKES / FAKE_AGENTS[protocol]), str(scenario_path)]
            ),
        ]
        environment = {
            **os.environ,
            "RSI_AGENT_DRIVER_TIMINGS": json.dumps({**TIMINGS, **(timings or {})}),
            **(env or {}),
        }
        with (
            (root / "driver-stdout.jsonl").open("wb") as stdout,
            (root / "driver-stderr.txt").open("wb") as stderr,
        ):
            process = subprocess.Popen(
                argv, stdout=stdout, stderr=stderr, cwd=root, env=environment
            )
        run = DriverRun(process, root)
        runs.append(run)
        return run

    yield start
    for run in runs:
        if run.process.poll() is None:
            run.process.kill()
            run.process.wait()
        for event in run.events("start"):
            try:
                os.kill(event["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def _gaps(events: list[dict[str, Any]]) -> list[int]:
    return [b["t"] - a["t"] for a, b in zip(events, events[1:], strict=False)]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; it has exited for our purposes.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Continuation
# ---------------------------------------------------------------------------

STOP_REASONS = ["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"]


@requires_node
def test_acp_sends_the_continuation_after_every_stop_reason(driver) -> None:
    turns = [{"text": f"turn {r}", "stopReason": r} for r in STOP_REASONS]
    run = driver("acp", {"turns": [*turns, {"hold": True}]})

    run.wait_prompts(len(STOP_REASONS) + 1)
    assert run.stop() == 128 + signal.SIGTERM

    prompts = run.prompts()
    assert [p["text"] for p in prompts] == [PROMPT] + [STOP_HOOK_REASON] * 5
    assert {p["sessionId"] for p in prompts} == {"fake-acp-session-1"}
    assert len(run.events("start")) == 1
    assert [r["stop_reason"] for r in run.results("turn_end")] == STOP_REASONS
    continuations = [line for line in run.lines() if "rsi_continuation" in line]
    assert [c["rsi_continuation"] for c in continuations] == [1, 2, 3, 4, 5]
    assert {c["message"]["content"][0]["text"] for c in continuations} == {FEEDBACK}


@requires_node
def test_zcode_continues_only_after_the_session_is_idle(driver) -> None:
    run = driver(
        "zcode",
        {
            "turns": [
                {"text": "one", "lockMs": 150},
                {"text": "two", "lockMs": 150},
                {"text": "three", "delayMs": 10_000},
            ]
        },
    )

    run.wait_prompts(3)
    run.stop()

    prompts = run.prompts()
    assert [p["text"] for p in prompts] == [PROMPT, STOP_HOOK_REASON, STOP_HOOK_REASON]
    # Sending on turn.completed, or on the other session's idle signal the
    # fake sends mid-turn, would hit the still-held prompt lock.
    assert run.events("send_rejected") == []
    idle = [e for e in run.events("state") if e["reason"] == "prompt_completed"]
    assert prompts[1]["t"] >= idle[0]["t"]
    assert prompts[2]["t"] >= idle[1]["t"]
    assert len(run.events("start")) == 1


@requires_node
def test_zcode_retries_a_send_rejected_while_a_prompt_runs(driver) -> None:
    run = driver(
        "zcode", {"rejectSends": 2, "turns": [{"text": "ok", "delayMs": 10_000}]}
    )

    run.wait_prompts(1)
    run.stop()

    assert len(run.events("send_rejected")) == 2
    assert run.prompts()[0]["text"] == PROMPT
    assert run.results("turn_error") == []


@requires_node
def test_zcode_watchdog_sends_when_the_idle_signal_never_comes(driver) -> None:
    run = driver(
        "zcode",
        {
            "turns": [
                {"text": "one", "noIdle": True},
                {"text": "two", "delayMs": 10_000},
            ]
        },
    )

    run.wait_prompts(2)
    run.stop()

    first, second = run.prompts()
    assert second["t"] - first["t"] >= TIMINGS["zcodeWatchdogMs"]
    assert run.results("turn_end")[0]["stop_reason"] == "watchdog"


@requires_node
def test_zcode_idle_arriving_after_the_watchdog_does_not_end_the_next_turn(
    driver,
) -> None:
    run = driver(
        "zcode",
        {
            "turns": [
                {"text": "t1"},
                {"text": "t2", "lockMs": 4_000},
                {"text": "t3"},
                {"text": "t4"},
                {"text": "t5", "delayMs": 10_000},
            ]
        },
        # Wide margins: a loaded test host must not make a normal turn
        # outlast the watchdog.
        timings={"zcodeWatchdogMs": 1_500},
    )

    run.wait_prompts(5)
    run.stop()

    assert run.results("turn_error") == []
    ends = run.results("turn_end")
    assert [e["stop_reason"] for e in ends] == [
        "prompt_completed",
        "watchdog",
        "prompt_completed",
        "prompt_completed",
    ]
    # Every turn_end follows its own turn's text, not the previous one's.
    last_text = None
    for line in run.lines():
        if line["type"] == "assistant":
            last_text = line["message"]["content"][-1]["text"]
        elif line.get("subtype") == "turn_end":
            assert last_text == f"t{line['turn']}"
    # Sends are refused only while turn 2 still holds the prompt lock.
    third = run.prompts()[2]
    assert run.events("send_rejected")
    assert all(e["t"] <= third["t"] for e in run.events("send_rejected"))


@requires_node
def test_zcode_late_idle_of_the_previous_turn_does_not_end_the_next_turn(
    driver,
) -> None:
    # ZCode frees the session lock before it broadcasts the idle signal. After
    # a watchdog send, the next prompt is accepted in that gap and the old
    # turn's idle signal arrives while the new turn is still running.
    run = driver(
        "zcode",
        {
            "turns": [
                {"text": "t1", "lockMs": 1_000, "idleDelayMs": 300},
                {"text": "t2", "delayMs": 1_000},
                {"text": "t3", "delayMs": 10_000},
            ]
        },
        timings={"zcodeWatchdogMs": 500},
    )

    run.wait_prompts(3)
    run.stop()

    ends = run.results("turn_end")
    assert [e["turn"] for e in ends] == [1, 2]
    assert ends[1]["stop_reason"] == "prompt_completed"
    assert run.results("turn_error") == []
    # Had the stale signal ended turn 2, the next send would have collided
    # with turn 2 still running.
    second = run.prompts()[1]
    assert all(e["t"] <= second["t"] for e in run.events("send_rejected"))


@requires_node
def test_zcode_compacts_on_overflow_and_after_three_failed_turns(driver) -> None:
    failure = {"fail": {"type": "provider_error", "message": "500"}}
    run = driver(
        "zcode",
        {
            "turns": [
                {"fail": {"type": "model_context_exceeded", "message": "too long"}},
                {"text": "recovered"},
                failure,
                failure,
                failure,
                {"text": "recovered again", "delayMs": 10_000},
            ]
        },
    )

    run.wait_prompts(6)
    run.stop()

    received = [e for e in run.events("recv") if e["method"] != "session/subscribe"]
    sequence = [
        "compact" if e["method"] == "session/compact" else e["method"]
        for e in received
        if e["method"] in ("session/send", "session/compact")
    ]
    assert sequence == [
        "session/send",
        "compact",
        "session/send",
        "session/send",
        "session/send",
        "session/send",
        "compact",
        "session/send",
    ]
    compactions = [line for line in run.lines() if line.get("subtype") == "compaction"]
    assert [c["state"] for c in compactions] == ["accepted", "accepted"]


# ---------------------------------------------------------------------------
# Errors and backoff
# ---------------------------------------------------------------------------


@requires_node
def test_acp_backs_off_on_errors_and_resets_after_a_turn_with_output(driver) -> None:
    error = {"error": {"code": -32603, "message": "Internal error: APIError"}}
    run = driver(
        "acp",
        {"turns": [error, error, error, {"text": "progress"}, error, {"hold": True}]},
        timings={"backoffBaseMs": 100, "backoffMaxMs": 10_000},
    )

    run.wait_prompts(6)
    run.stop()

    gaps = _gaps(run.prompts())
    assert gaps[0] >= 100
    assert gaps[1] >= 200
    assert gaps[2] >= 400
    assert gaps[3] < 100
    assert 100 <= gaps[4] < 400
    errors = run.results("turn_error")
    assert len(errors) == 4
    assert errors[0]["error"] == error["error"]
    assert len(run.events("start")) == 1


@requires_node
def test_acp_backoff_is_capped(driver) -> None:
    error = {"error": {"code": -32000, "message": "Authentication required"}}
    run = driver(
        "acp",
        {"turns": [error]},
        timings={"backoffBaseMs": 50, "backoffMaxMs": 120},
    )

    run.wait_prompts(5)
    run.stop()

    gaps = _gaps(run.prompts())
    assert all(gap < 400 for gap in gaps)
    assert gaps[-1] >= 120


@requires_node
def test_a_fast_turn_without_output_is_backed_off_like_an_error(driver) -> None:
    run = driver(
        "acp", {"turns": [{}, {}, {"hold": True}]}, timings={"backoffBaseMs": 100}
    )

    run.wait_prompts(3)
    run.stop()

    gaps = _gaps(run.prompts())
    assert gaps[0] >= 100
    assert gaps[1] >= 200
    errors = run.results("turn_error")
    assert [e["error"]["code"] for e in errors] == ["empty_turn", "empty_turn"]
    assert errors[0]["stop_reason"] == "end_turn"


@requires_node
def test_zcode_backs_off_on_failed_turns_and_resets_after_a_turn_with_output(
    driver,
) -> None:
    failure = {"fail": {"type": "provider_error", "message": "500"}}
    run = driver(
        "zcode",
        {
            "turns": [
                failure,
                failure,
                {"text": "progress"},
                failure,
                {"text": "x", "delayMs": 10_000},
            ]
        },
        timings={"backoffBaseMs": 100, "backoffMaxMs": 10_000},
    )

    run.wait_prompts(5)
    run.stop()

    gaps = _gaps(run.prompts())
    assert gaps[0] >= 100
    assert gaps[1] >= 200
    assert gaps[2] < 100
    assert 100 <= gaps[3] < 400
    errors = run.results("turn_error")
    assert [e["error"]["type"] for e in errors] == ["provider_error"] * 3


@requires_node
def test_zcode_fast_turn_without_output_is_backed_off_like_an_error(driver) -> None:
    run = driver(
        "zcode",
        {"turns": [{"empty": True}, {"empty": True}, {"text": "x", "delayMs": 10_000}]},
        timings={"backoffBaseMs": 100},
    )

    run.wait_prompts(3)
    run.stop()

    gaps = _gaps(run.prompts())
    assert gaps[0] >= 100
    assert gaps[1] >= 200
    errors = run.results("turn_error")
    assert [e["error"]["code"] for e in errors] == ["empty_turn", "empty_turn"]


@requires_node
def test_zcode_send_errors_are_recorded_and_backed_off(driver) -> None:
    error = {"code": -32603, "message": "provider unavailable"}
    run = driver(
        "zcode",
        {"sendErrors": [error, error], "turns": [{"text": "x", "delayMs": 10_000}]},
        timings={"backoffBaseMs": 100},
    )

    run.wait_prompts(1)
    run.stop()

    sends = [e for e in run.events("recv") if e["method"] == "session/send"]
    gaps = _gaps(sends)
    assert len(sends) == 3
    assert gaps[0] >= 100
    assert gaps[1] >= 200
    errors = run.results("turn_error")
    assert [e["error"] for e in errors] == [error, error]
    assert run.prompts()[0]["text"] == PROMPT


@requires_node
def test_queued_prompt_guard_waits_for_the_agent_to_go_idle(
    driver, tmp_path: Path
) -> None:
    home = tmp_path / "kimi-home"
    wire = home / "sessions" / "workdir-hash" / "fake-acp-session-1" / "agents" / "main"
    wire.mkdir(parents=True)
    run = driver(
        "acp",
        {
            "wire": str(wire / "wire.jsonl"),
            "turns": [
                {"text": "start a background job", "selfTurnMs": 800},
                {"text": "after"},
                {"hold": True},
            ],
        },
        flags=(
            "--idle-probe",
            "kimi-wire:$FAKE_KIMI_HOME/sessions/*/{session}/agents/main/wire.jsonl",
        ),
        env={"FAKE_KIMI_HOME": str(home)},
    )

    run.wait_prompts(3)
    run.stop()

    queued = [e for e in run.events("prompt") if e.get("queued")]
    assert [e["queueLength"] for e in queued] == [1]
    self_turn_end = run.events("self_turn_end")[0]
    second = run.prompts()[1]
    assert second["t"] >= self_turn_end["t"] + TIMINGS["idleQuietMs"]
    assert [r.get("rsi_queued", False) for r in run.results("turn_end")][:2] == [
        False,
        True,
    ]


@requires_node
def test_queued_prompt_guard_without_a_wire_file_waits_a_fixed_time(
    driver, tmp_path: Path
) -> None:
    run = driver(
        "acp",
        {
            "turns": [
                {"text": "start", "selfTurnMs": 200},
                {"text": "after"},
                {"hold": True},
            ]
        },
        flags=("--idle-probe", f"kimi-wire:{tmp_path}/no-such-dir/*/wire.jsonl"),
        timings={"queuedWaitMs": 400},
    )

    run.wait_prompts(3)
    run.stop()

    queued = [e for e in run.events("prompt") if e.get("queued")]
    assert len(queued) == 1
    assert run.prompts()[1]["t"] - queued[0]["t"] >= 400


# ---------------------------------------------------------------------------
# Permissions and host requests
# ---------------------------------------------------------------------------

QWEN_PLAN_OPTIONS = [
    {"optionId": "restore_previous", "kind": "allow_once", "name": "restore"},
    {"optionId": "proceed_always", "kind": "allow_always", "name": "always"},
    {"optionId": "proceed_once", "kind": "allow_once", "name": "once"},
    {"optionId": "cancel", "kind": "reject_once", "name": "no"},
]


@requires_node
@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        (
            ("--permission-prefer", "restore_previous,allow_always,allow_once"),
            "restore_previous",
        ),
        ((), "proceed_always"),
    ],
)
def test_permission_requests_pick_the_preferred_option(
    driver, flags: tuple[str, ...], expected: str
) -> None:
    run = driver(
        "acp",
        {
            "turns": [
                {
                    "permission": {"options": QWEN_PLAN_OPTIONS},
                    "clientRequest": "fs/read_text_file",
                    "text": "x",
                },
                {"permission": {"options": [QWEN_PLAN_OPTIONS[-1]]}, "text": "y"},
                {"hold": True},
            ]
        },
        flags=flags,
    )

    run.wait_prompts(3)
    run.stop()

    replies = [e["reply"]["result"]["outcome"] for e in run.events("permission_reply")]
    assert replies == [
        {"outcome": "selected", "optionId": expected},
        {"outcome": "cancelled"},
    ]
    unknown = run.events("client_request_reply")[0]["reply"]
    assert unknown["error"]["code"] == -32601


@requires_node
def test_zcode_host_requests_get_headless_answers(driver) -> None:
    methods = [
        "interaction/requestPermission",
        "interaction/requestUserInput",
        "session/requestRuntimePreferences",
        "subagents/readRuntimeConfig",
        "interaction/requestProviderRuntimeHeaders",
        "browser/open",
    ]
    run = driver(
        "zcode",
        {
            "turns": [
                {"hostRequests": methods, "text": "x"},
                {"text": "y", "delayMs": 10_000},
            ]
        },
    )

    run.wait_prompts(2)
    run.stop()

    replies = {e["method"]: e["reply"] for e in run.events("host_reply")}
    assert replies["interaction/requestPermission"]["result"] == {"decision": "allow"}
    assert replies["interaction/requestUserInput"]["result"]["action"] == "decline"
    assert replies["session/requestRuntimePreferences"]["result"] == {
        "nativeSearchEnhancementsEnabled": False,
        "memoryEnabled": False,
        "askUserQuestionAutoResolutionEnabled": True,
    }
    assert replies["subagents/readRuntimeConfig"]["result"] == {
        "kind": "built-in-fallback"
    }
    assert replies["interaction/requestProviderRuntimeHeaders"]["result"] == {
        "headersApplied": False
    }
    assert replies["browser/open"]["error"]["code"] == -32601


@requires_node
def test_zcode_session_skips_titles_and_denies_the_agents_tools(driver) -> None:
    run = driver(
        "zcode",
        {"turns": [{"text": "x"}, {"text": "y", "delayMs": 10_000}]},
        flags=("--zcode-tool-deny", "AskUserQuestion, EnterPlanMode,CronCreate"),
    )

    run.wait_prompts(2)
    run.stop()

    [create] = [e for e in run.events("recv") if e["method"] == "session/create"]
    # A generated title is a model request of its own.
    assert create["params"]["titleGenerationEnabled"] is False
    assert create["params"]["mode"] == "yolo"
    sends = [e for e in run.events("recv") if e["method"] == "session/send"]
    assert len(sends) == 2
    for send in sends:
        assert send["params"]["toolDenylist"] == [
            "CronCreate",
            "CronDelete",
            "OffPeakCreate",
            "AskUserQuestion",
            "EnterPlanMode",
        ]


@requires_node
def test_zcode_frames_from_the_real_cli_become_the_transcript(driver) -> None:
    """Notifications captured from ZCode 3.15.1 under this driver.

    One turn that streams text, calls Bash and completes, then one that fails
    with a context overflow. If a ZCode upgrade renames an event or field the
    driver relies on, re-capture the frames and this test shows what moved.
    """
    run = driver(
        "zcode",
        {
            "replayFile": str(FAKES / "zcode-3.15.1-frames.jsonl"),
            "turns": [
                {"replay": "ok"},
                {"replay": "fail"},
                {"text": "after", "delayMs": 10_000},
            ],
        },
    )

    run.wait_prompts(3)
    run.stop()

    lines = run.lines()
    assistant = [line for line in lines if line["type"] == "assistant"]
    assert [line["message"]["content"] for line in assistant[:2]] == [
        [
            {"type": "text", "text": "Let me run a command (1)."},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "Bash",
                "input": {
                    "command": "echo key-check $RSI_AGENT_API_KEY ppid=$PPID; id -u"
                },
            },
        ],
        [{"type": "text", "text": "Reply 2: I think I am finished."}],
    ]
    [tool_result] = [
        block
        for line in lines
        if line["type"] == "user"
        for block in line["message"]["content"]
        if block["type"] == "tool_result"
    ]
    assert tool_result == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "key-check [REDACTED] ppid=2137606\n1001",
        "is_error": False,
    }
    [turn_end] = run.results("turn_end")[:1]
    assert turn_end["turn"] == 1
    assert turn_end["stop_reason"] == "prompt_completed"
    assert turn_end["usage"]["inputTokens"] == 200
    [turn_error] = run.results("turn_error")
    assert turn_error["turn"] == 2
    assert turn_error["error"]["type"] == "model_context_exceeded"
    # The overflow made the driver compact before the next prompt.
    methods = [
        e["method"]
        for e in run.events("recv")
        if e["method"] in ("session/send", "session/compact")
    ]
    assert methods == [
        "session/send",
        "session/send",
        "session/compact",
        "session/send",
    ]
    prompts = run.prompts()
    assert [p["text"] for p in prompts] == [PROMPT, STOP_HOOK_REASON, STOP_HOOK_REASON]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@requires_node
@pytest.mark.parametrize("protocol", ["acp", "zcode"])
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_signals_end_driver_and_agent_within_the_grace_period(
    driver, protocol: str, signum: int
) -> None:
    turn = {"hold": True} if protocol == "acp" else {"text": "x", "delayMs": 30_000}
    # The agent outlives a closed stdin, so only the driver's SIGTERM ends it.
    run = driver(protocol, {"ignoreStdinEnd": True, "turns": [turn]})
    run.wait_prompts(1)
    agent_pid = run.agent_pid()

    started = time.monotonic()
    assert run.stop(signum) == 128 + signum
    assert time.monotonic() - started < 10

    assert not _alive(agent_pid)
    assert run.events("sigterm")
    exit_record = run.lines()[-1]
    assert exit_record["subtype"] == "driver_exit"
    assert exit_record["reason"] == "signal"
    assert exit_record["exit_code"] == 128 + signum
    cancel = "session/cancel" if protocol == "acp" else "session/stop"
    assert cancel in [e["method"] for e in run.events("recv")]


@requires_node
@pytest.mark.parametrize("protocol", ["acp", "zcode"])
def test_a_stop_that_signals_driver_and_agent_together_is_a_signal_exit(
    driver, protocol: str
) -> None:
    """The harness's process stop sends SIGTERM to every pid at once."""
    turn = {"hold": True} if protocol == "acp" else {"text": "x", "delayMs": 30_000}
    run = driver(protocol, {"turns": [turn]})
    run.wait_prompts(1)

    run.process.send_signal(signal.SIGTERM)
    os.kill(run.agent_pid(), signal.SIGTERM)
    assert run.wait() == 128 + signal.SIGTERM

    exit_record = run.lines()[-1]
    assert exit_record["reason"] == "signal"
    assert len(run.events("start")) == 1


@requires_node
def test_an_agent_ignoring_sigterm_is_killed(driver) -> None:
    run = driver(
        "acp",
        {"ignoreSigterm": True, "ignoreStdinEnd": True, "turns": [{"hold": True}]},
    )
    run.wait_prompts(1)
    agent_pid = run.agent_pid()

    started = time.monotonic()
    assert run.stop() == 128 + signal.SIGTERM
    elapsed = time.monotonic() - started

    assert TIMINGS["killGraceMs"] / 1000 <= elapsed < 10
    assert run.events("sigterm")
    assert not _alive(agent_pid)


@requires_node
@pytest.mark.parametrize("protocol", ["acp", "zcode"])
@pytest.mark.parametrize(("agent_code", "driver_code"), [(3, 3), (0, 70)])
def test_agent_exit_is_final_and_sets_the_exit_code(
    driver, protocol: str, agent_code: int, driver_code: int
) -> None:
    run = driver(protocol, {"turns": [{"text": "work"}, {"exit": agent_code}]})

    assert run.wait() == driver_code

    assert len(run.events("start")) == 1
    exit_record = run.lines()[-1]
    assert exit_record["reason"] == "agent_exited"
    assert exit_record["agent_exit_code"] == agent_code
    assert exit_record["exit_code"] == driver_code


@requires_node
@pytest.mark.parametrize("protocol", ["acp", "zcode"])
def test_without_a_continuation_reason_the_driver_runs_one_turn(
    driver, protocol: str
) -> None:
    run = driver(
        protocol, {"turns": [{"text": "done"}, {"text": "more"}]}, continuation=None
    )

    assert run.wait() == 0

    assert [p["text"] for p in run.prompts()] == [PROMPT]
    assert run.events("stdin_end")
    assert not [line for line in run.lines() if "rsi_continuation" in line]
    exit_record = run.lines()[-1]
    assert exit_record["reason"] == "completed"
    assert exit_record["exit_code"] == 0


@requires_node
@pytest.mark.parametrize(
    ("protocol", "turn"),
    [
        ("acp", {"error": {"code": -32603, "message": "Internal error"}}),
        ("zcode", {"fail": {"type": "provider_error", "message": "500"}}),
    ],
)
def test_without_a_continuation_reason_a_failed_turn_exits_1(
    driver, protocol: str, turn: dict[str, Any]
) -> None:
    run = driver(protocol, {"turns": [turn, {"text": "more"}]}, continuation=None)

    assert run.wait() == 1

    assert [p["text"] for p in run.prompts()] == [PROMPT]
    assert len(run.results("turn_error")) == 1
    exit_record = run.lines()[-1]
    assert exit_record["reason"] == "completed"
    assert exit_record["exit_code"] == 1


@requires_node
@pytest.mark.parametrize(
    ("protocol", "scenario", "detail"),
    [
        ("acp", {"initError": {"code": -32603, "message": "boom"}}, "initialize"),
        ("acp", {"sessionError": {"code": -32000, "message": "auth"}}, "session/new"),
        ("zcode", {"createError": {"code": -32000, "message": "no"}}, "session/create"),
    ],
)
def test_setup_failures_exit_64(driver, protocol, scenario, detail) -> None:
    run = driver(protocol, scenario)

    assert run.wait() == 64

    exit_record = run.lines()[-1]
    assert exit_record["reason"] == "setup"
    assert detail in exit_record["error"]
    assert run.prompts() == []


# ---------------------------------------------------------------------------
# Resuming a session of an earlier run
# ---------------------------------------------------------------------------


@requires_node
@pytest.mark.parametrize(
    ("protocol", "session", "flags", "expected"),
    [
        (
            "acp",
            "earlier-acp-session",
            ("--acp-mode", "yolo"),
            [
                ("initialize", None),
                # Not session/load, which replays the history into the transcript.
                (
                    "session/resume",
                    {"sessionId": "earlier-acp-session", "mcpServers": []},
                ),
                ("session/set_mode", {"sessionId": "earlier-acp-session"}),
                ("session/prompt", {"sessionId": "earlier-acp-session"}),
                ("session/prompt", {"sessionId": "earlier-acp-session"}),
            ],
        ),
        (
            "zcode",
            "sess_earlier",
            (),
            [
                ("session/resume", {"sessionId": "sess_earlier"}),
                ("session/subscribe", {"sessionId": "sess_earlier"}),
                ("session/send", {"sessionId": "sess_earlier"}),
                ("session/send", {"sessionId": "sess_earlier"}),
            ],
        ),
    ],
)
def test_resume_session_continues_the_named_session_with_the_prompt(
    driver, protocol, session, flags, expected
) -> None:
    run = driver(
        protocol,
        {"turns": [{"text": "where was I"}, {"hold": True, "delayMs": 10_000}]},
        flags=("--resume-session", session, *flags),
    )

    run.wait_prompts(2)
    run.stop()

    received = [e for e in run.events("recv") if e["method"] in dict(expected)]
    assert [e["method"] for e in received] == [method for method, _ in expected]
    for event, (_, params) in zip(received, expected, strict=True):
        assert params is None or params.items() <= event["params"].items()
    assert [p["text"] for p in run.prompts()] == [PROMPT, STOP_HOOK_REASON]
    assert {p["sessionId"] for p in run.prompts()} == {session}
    [init] = [line for line in run.lines() if line.get("subtype") == "init"]
    assert init["session_id"] == session
    users = [line for line in run.lines() if line["type"] == "user"]
    assert users[0]["message"]["content"][0]["text"] == PROMPT


@requires_node
@pytest.mark.parametrize("protocol", ["acp", "zcode"])
def test_a_session_that_cannot_be_resumed_is_a_setup_failure(driver, protocol) -> None:
    run = driver(
        protocol,
        {"resumeError": {"code": -32602, "message": "Unknown sessionId: gone"}},
        flags=("--resume-session", "gone"),
    )

    assert run.wait() == 64

    exit_record = run.lines()[-1]
    assert exit_record["reason"] == "setup"
    assert "session/resume failed" in exit_record["error"]
    assert "Unknown sessionId: gone" in exit_record["error"]
    assert run.prompts() == []
    assert not [
        e
        for e in run.events("recv")
        if e["method"] in {"session/new", "session/create", "session/load"}
    ]


@requires_node
@pytest.mark.parametrize(
    "argv",
    [
        ["--protocol", "acp", "--prompt-file", "/nonexistent/prompt.md", "--", "true"],
        ["--protocol", "smtp", "--prompt-file", "/dev/null", "--", "true"],
        ["--protocol", "acp", "--prompt-file", "/dev/null"],
        ["--protocol", "acp", "--bogus", "--prompt-file", "/dev/null", "--", "true"],
        [
            "--protocol",
            "acp",
            "--prompt-file",
            "/dev/null",
            "--",
            "/nonexistent/agent-cli",
        ],
        [
            "--protocol",
            "acp",
            "--zcode-tool-deny",
            "Bash",
            "--prompt-file",
            "/dev/null",
            "--",
            "true",
        ],
    ],
)
def test_invalid_invocations_exit_64_with_a_driver_exit_record(argv) -> None:
    completed = subprocess.run(
        [str(NODE), str(DRIVER_SCRIPT), *argv],
        capture_output=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 64
    assert completed.stderr == b""
    record = json.loads(completed.stdout.decode().splitlines()[-1])
    assert record["subtype"] == "driver_exit"
    assert record["reason"] == "setup"


@requires_node
@pytest.mark.parametrize("with_cli", [True, False])
def test_version_prints_driver_and_agent_versions(with_cli: bool) -> None:
    # Without a CLI this is the check DrivenAgent.install_stop_hook runs.
    cli = ["--", str(NODE)] if with_cli else []
    completed = subprocess.run(
        [str(NODE), str(DRIVER_SCRIPT), "--version", *cli],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    lines = completed.stdout.splitlines()
    assert len(lines) == (2 if with_cli else 1)
    assert lines[0].startswith("rsi-agent-driver ")
    if with_cli:
        assert lines[1].startswith("v")


def test_launcher_wrapper_execs_the_installed_driver_path() -> None:
    wrapper = (DRIVER / "rsi-agent-driver.sh").read_text()

    assert wrapper.startswith("#!/bin/sh\n")
    assert 'exec /usr/local/bin/node /opt/rsi-harness/agent-driver.mjs "$@"' in wrapper


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


@requires_node
def test_a_leading_slash_prompt_is_not_sent_as_a_slash_command(driver) -> None:
    run = driver("acp", {"turns": [{"hold": True}]}, prompt="/compact the data\n")

    run.wait_prompts(1)
    run.stop()

    assert run.prompts()[0]["text"] == "Task:\n/compact the data\n"


@requires_node
def test_prompt_and_secrets_never_appear_in_process_arguments(driver) -> None:
    if not Path("/proc/self/cmdline").exists():
        pytest.skip("needs /proc")
    secret = "sk-test-cmdline-secret-7f3a"
    prompt = "Secret task text that must stay out of argv.\n"
    run = driver(
        "acp",
        {"turns": [{"hold": True}]},
        prompt=prompt,
        env={"RSI_AGENT_API_KEY": secret},
    )
    run.wait_prompts(1)

    for pid in (run.process.pid, run.agent_pid()):
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode()
        assert prompt.strip() not in cmdline
        assert secret not in cmdline
    assert run.prompts()[0]["text"] == prompt
    run.stop()


@requires_node
def test_secrets_in_tool_output_and_stderr_are_redacted_in_agent_output(
    driver, tmp_path: Path
) -> None:
    secret = "sk-test-redaction-secret-91b2"
    run = driver(
        "acp",
        {
            "turns": [
                {
                    "stderr": [f"debug: key={secret}"],
                    "tool": {
                        "kind": "execute",
                        "title": "env",
                        "rawInput": {"command": "env"},
                        "output": "RSI_AGENT_API_KEY=",
                        "outputEnv": "SECRET_KEY",
                    },
                },
                {"hold": True},
            ]
        },
        env={"SECRET_KEY": secret},
    )
    run.wait_prompts(2)
    run.stop()
    raw = run.out.read_bytes()
    assert secret.encode() in raw

    artifact = tmp_path / "agent_output.txt"
    writer = _RedactedOutputWriter(artifact, (secret,))
    writer.append(raw)
    writer.finish()

    redacted = artifact.read_text()
    assert secret not in redacted
    for line in redacted.splitlines():
        json.loads(line)


@requires_node
@pytest.mark.parametrize("protocol", ["acp", "zcode"])
def test_transcript_renders_through_the_trajectory_parser(
    driver, protocol: str
) -> None:
    if protocol == "acp":
        turns = [
            {
                "thought": "plan",
                "text": "Running ls.",
                "tool": {
                    "kind": "execute",
                    "title": "Shell",
                    "rawInput": {"command": "ls -la"},
                    "output": "file-a\nfile-b",
                    "lateInput": True,
                },
            },
            {"text": "Still working."},
            {"hold": True},
        ]
    else:
        turns = [
            {
                "thought": "plan",
                "text": "Running ls.",
                "tool": {
                    "name": "Bash",
                    "input": {"command": "ls -la"},
                    "output": "file-a\nfile-b",
                },
            },
            {"text": "Still working."},
            {"text": "x", "delayMs": 10_000},
        ]
    run = driver(protocol, {"turns": turns})
    run.wait_prompts(3)
    run.stop()

    trajectory = parse(run.out)

    system, user, first, second = trajectory.exchanges[:4]
    assert system.role == "system"
    assert "model: fake-model" in system.text
    assert trajectory.meta["model"] == "fake-model"
    assert (user.role, user.text) == ("user", PROMPT)
    assert first.thinking == "plan"
    assert first.text == "Running ls."
    [call] = first.tool_calls
    assert (call.name, call.input["command"]) == ("Bash", "ls -la")
    assert call.has_result and call.result_content == "file-a\nfile-b"
    assert second.text == "Still working."
    assert trajectory.tool_use_count == 1
    assert len(trajectory.markers) == 2
    assert {marker.feedback for marker in trajectory.markers} == {FEEDBACK}


@requires_node
def test_tool_results_and_agent_stderr_are_capped(driver) -> None:
    long_line = "e" * 10_000
    run = driver(
        "acp",
        {
            "stderrAtStart": ["starting"],
            "turns": [
                {
                    "stderr": [long_line, *[f"line {i}" for i in range(20)]],
                    "tool": {
                        "kind": "read",
                        "title": "Read",
                        "rawInput": {"path": "/a"},
                        "output": "x",
                        "outputRepeat": 200_000,
                    },
                },
                {"hold": True},
            ],
        },
        timings={"stderrBurst": 5, "stderrWindowMs": 60_000},
    )
    run.wait_prompts(2)
    run.stop()
    lines = run.lines()

    stderr = [line for line in lines if line.get("subtype") == "agent_stderr"]
    texts = [line["text"] for line in stderr if "text" in line]
    assert texts[0] == "starting"
    assert len(texts) == 1 + 5
    assert all(len(text) < 4 * 1024 + 100 for text in texts)
    assert "truncated" in texts[1]
    assert sum(line.get("suppressed", 0) for line in stderr) == 16
    [result] = [
        block
        for line in lines
        if line["type"] == "user"
        for block in line["message"]["content"]
        if block["type"] == "tool_result"
    ]
    assert len(result["content"]) < 64 * 1024 + 100
    assert "truncated" in result["content"]
    [use] = [
        block
        for line in lines
        if line["type"] == "assistant"
        for block in line["message"]["content"]
        if block["type"] == "tool_use"
    ]
    assert use["name"] == "Read"
    assert use["input"]["file_path"] == "/a"


# ---------------------------------------------------------------------------
# Fake model server
# ---------------------------------------------------------------------------


def test_fake_model_server_serves_models_completions_and_errors(tmp_path: Path) -> None:
    module = _load_fake_model_server()
    log = tmp_path / "requests.jsonl"
    with module.FakeModelServer(log, {"cycle": ["tool", "e429"]}) as server:
        with urlopen(f"{server.url}/v1/models", timeout=5) as response:
            assert json.load(response)["data"][0]["id"] == "fake-model"
        body = json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        request = Request(
            f"{server.url}/v1/chat/completions",
            data=body,
            headers={"content-type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            message = json.load(response)["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "bash"
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 429
        requests = server.requests()

    assert [r["action"] for r in requests] == ["tool", "e429"]
    assert requests[0]["last_user"] == "hi"


def test_fake_model_server_calls_the_offered_shell_tool(tmp_path: Path) -> None:
    module = _load_fake_model_server()
    shell = {
        "name": "run_shell_command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "is_background": {"type": "boolean"},
                "description": {"type": "string"},
            },
            "required": ["command", "is_background"],
        },
    }
    tools = [
        {"type": "function", "function": {"name": "read_file"}},
        {"type": "function", "function": shell},
    ]
    scenario = {"cycle": ["tool"], "commands": ["rsi-submit"]}

    def complete(messages: list[dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps({"model": "m", "messages": messages, "tools": tools})
        request = Request(
            f"{server.url}/v1/chat/completions",
            data=body.encode(),
            headers={"content-type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)["choices"][0]["message"]["tool_calls"][0]

    with module.FakeModelServer(tmp_path / "log.jsonl", scenario) as server:
        first = complete([{"role": "user", "content": "go"}])
        second = complete(
            [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "call_1", "content": "submitted"},
                {"role": "user", "content": "again"},
            ]
        )
        requests = server.requests()

    assert first["function"]["name"] == "run_shell_command"
    assert json.loads(first["function"]["arguments"]) == {
        "command": "rsi-submit",
        "is_background": False,
    }
    assert json.loads(second["function"]["arguments"])["command"] == "echo tool-run-2"
    assert requests[0]["tool_names"] == ["read_file", "run_shell_command"]
    assert requests[1]["last_tool"] is None
    assert requests[1]["last_user"] == "again"


def test_fake_model_server_scripts_replies_and_watches_requests(
    tmp_path: Path,
) -> None:
    module = _load_fake_model_server()
    scenario = {"replies": {"2": '{"verdict": "pass"}'}, "watch": ["MARKER-1"]}

    def complete(system: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "go"},
        ]
        request = Request(
            f"{server.url}/v1/chat/completions",
            data=json.dumps({"model": "m", "messages": messages}).encode(),
            headers={"content-type": "application/json"},
        )
        with urlopen(request, timeout=5) as response:
            return json.load(response)["choices"][0]["message"]["content"]

    with module.FakeModelServer(tmp_path / "log.jsonl", scenario) as server:
        first = complete("you are helpful")
        second = complete("project rules: MARKER-1")
        requests = server.requests()

    assert first == "Reply 1: I think I am finished."
    assert second == '{"verdict": "pass"}'
    assert [r["watched"] for r in requests] == [[], ["MARKER-1"]]


def _sse_events(response: Any) -> list[dict[str, Any]]:
    """The JSON payloads of an SSE body, each checked against its event name."""
    events, name = [], None
    for raw in response.read().decode().splitlines():
        if raw.startswith("event: "):
            name = raw.removeprefix("event: ")
        elif raw.startswith("data: "):
            payload = json.loads(raw.removeprefix("data: "))
            assert payload["type"] == name
            events.append(payload)
    return events


def test_fake_model_server_speaks_the_responses_api(tmp_path: Path) -> None:
    module = _load_fake_model_server()
    shell = {
        "type": "function",
        "name": "exec_command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}, "workdir": {"type": "string"}},
            "required": ["cmd"],
        },
    }
    scenario = {
        "cycle": ["mixed"],
        "at": {"2": "e500"},
        "commands": ["rsi-submit"],
        "shell_arguments": {"yield_time_ms": 30_000},
    }

    def post(server: Any, items: list[dict[str, Any]]) -> Any:
        body = {
            "model": "m",
            "input": items,
            "tools": [shell],
            "stream": True,
            "reasoning": {"effort": "high"},
        }
        request = Request(
            f"{server.url}/v1/responses",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "authorization": "Bearer k"},
        )
        return urlopen(request, timeout=5)

    user = {"type": "message", "role": "user", "content": [{"type": "input_text"}]}
    user["content"][0]["text"] = "go"
    with module.FakeModelServer(tmp_path / "log.jsonl", scenario) as server:
        with post(server, [user]) as response:
            events = _sse_events(response)
        result = {"type": "function_call_output", "call_id": "call_1", "output": "ok"}
        with pytest.raises(HTTPError) as error:
            post(server, [user, events[-1]["response"]["output"][-1], result])
        requests = server.requests()

    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    text = "".join(e["delta"] for e in events if e["type"].endswith("text.delta"))
    message, call = events[-1]["response"]["output"]
    assert message["content"][0]["text"] == text == "Let me run a command (1)."
    assert call["type"] == "function_call" and call["name"] == "exec_command"
    assert json.loads(call["arguments"]) == {
        "cmd": "rsi-submit",
        "yield_time_ms": 30_000,
    }
    assert error.value.code == 500
    assert [r["api"] for r in requests] == ["responses", "responses"]
    assert {r["path"] for r in requests} == {"/v1/responses"}
    assert requests[0]["last_user"] == "go" and requests[0]["authorization"]
    assert requests[0]["reasoning_effort"] == "high"
    assert requests[1]["last_role"] == "tool" and requests[1]["last_tool"] == "ok"
    assert requests[1]["n_messages"] == 3


def test_fake_model_server_speaks_the_messages_api(tmp_path: Path) -> None:
    module = _load_fake_model_server()
    bash = {
        "name": "Bash",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    scenario = {"cycle": ["mixed"], "at": {"2": "overflow"}, "commands": ["rsi-submit"]}

    def post(server: Any, messages: list[dict[str, Any]]) -> Any:
        body = {
            "model": "m",
            "max_tokens": 32000,
            "messages": messages,
            "tools": [bash],
            "stream": True,
        }
        request = Request(
            f"{server.url}/v1/messages?beta=true",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "x-api-key": "k"},
        )
        return urlopen(request, timeout=5)

    prompt = [{"type": "text", "text": "<system-reminder>r</system-reminder>"}]
    prompt.append({"type": "text", "text": "go"})
    with module.FakeModelServer(tmp_path / "log.jsonl", scenario) as server:
        with post(server, [{"role": "user", "content": prompt}]) as response:
            events = _sse_events(response)
        [use] = [
            e["content_block"]
            for e in events
            if e["type"] == "content_block_start"
            and e["content_block"]["type"] == "tool_use"
        ]
        result = {"type": "tool_result", "tool_use_id": use["id"], "content": "ok"}
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": [{**use, "input": {}}]},
            {"role": "user", "content": [result]},
            # Claude Code's mid-conversation system context.
            {"role": "system", "content": "context"},
        ]
        with pytest.raises(HTTPError) as error:
            post(server, messages)
        body = json.load(error.value)
        requests = server.requests()

    kinds = [e["type"] for e in events]
    assert kinds[0] == "message_start" and kinds[-2:] == [
        "message_delta",
        "message_stop",
    ]
    assert events[-2]["delta"]["stop_reason"] == "tool_use"
    arguments = "".join(
        e["delta"]["partial_json"]
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"]["type"] == "input_json_delta"
    )
    assert use["name"] == "Bash" and json.loads(arguments) == {"command": "rsi-submit"}
    assert error.value.code == 400
    assert body["type"] == "error"
    assert body["error"]["message"].startswith("prompt is too long")
    assert [r["api"] for r in requests] == ["messages", "messages"]
    # The path as requested, without its query.
    assert {r["path"] for r in requests} == {"/v1/messages"}
    assert requests[0]["last_user"].endswith("go") and requests[0]["authorization"]
    assert requests[0]["max_tokens"] == 32000
    assert requests[1]["last_role"] == "tool" and requests[1]["last_tool"] == "ok"
    assert requests[1]["last_user"].endswith("go") and requests[1]["n_messages"] == 4


@requires_node
def test_driver_fake_agent_and_fake_model_keep_one_growing_conversation(
    driver, tmp_path: Path
) -> None:
    module = _load_fake_model_server()
    log = tmp_path / "model-requests.jsonl"
    scenario = {"cycle": ["mixed", "text"], "at": {"5": "e500"}}
    with module.FakeModelServer(log, scenario) as server:
        run = driver(
            "acp",
            {"model": {"url": server.url, "model": "fake-model"}},
            env={"FAKE_API_KEY": "sk-fake"},
        )
        run.wait_until(lambda: len(server.requests()) >= 9)
        run.stop()
        requests = server.requests()

    counts = [r["n_messages"] for r in requests]
    assert counts == sorted(counts)
    user_turns = [r for r in requests if r["last_role"] == "user"]
    assert user_turns[0]["last_user"] == PROMPT
    assert {r["last_user"] for r in user_turns[1:]} == {STOP_HOOK_REASON}
    assert any(r["last_role"] == "tool" for r in requests)
    assert all(r["stream"] and r["authorization"] for r in requests)
    assert len(run.events("start")) == 1
    assert [e["error"]["code"] for e in run.results("turn_error")] == [-32603]
    trajectory = parse(run.out)
    assert trajectory.tool_use_count >= 2
    assert all(
        call.has_result
        for exchange in trajectory.exchanges
        for call in exchange.tool_calls
    )
