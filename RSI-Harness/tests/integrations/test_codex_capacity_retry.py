from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import InfrastructureError
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    ContainerRef,
    NetworkPolicy,
)
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan
from tests.integrations.test_rsi_loop import RecordingAgentRuntime

SESSION = "01a16125-90ff-70a0-8d09-1a36fffb91d0"
CAPACITY = "ERROR: Selected model is at capacity. Please try a different model."
HEADER = f"OpenAI Codex v0.159.2\n--------\nsession id: {SESSION}\n--------\n"


class FakeClock:
    now = 1000.0

    def __call__(self) -> float:
        return self.now


class SequenceRuntime(RecordingAgentRuntime):
    """Simulate exec's deadline, streaming log and in-container retry delay."""

    def __init__(
        self, clock: FakeClock, results: list[AgentRunResult | BaseException]
    ) -> None:
        super().__init__()
        self.clock = clock
        self.results = list(results)
        self.start_callbacks: list[object] = []
        self.copied_text: dict[PurePosixPath, str] = {}

    def copy_to(self, container, source, target):
        self.copied_text[target] = source.read_text()
        super().copy_to(container, source, target)

    def exec(self, container, command, *, on_exec_start=None, **kwargs):
        self.start_callbacks.append(on_exec_start)
        if on_exec_start is not None:
            on_exec_start(self.clock.now)
        super().exec(container, command, **kwargs)
        assert self.results, "unexpected additional exec"
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        script = command if isinstance(command, str) else command[-1]
        sleep = re.search(r"\bsleep\s+([0-9.]+)", script)
        elapsed = 2.0 + (float(sleep.group(1)) if sleep else 0.0)
        timeout = kwargs.get("timeout_seconds")
        if timeout is not None and elapsed > timeout:
            elapsed = timeout
            result = AgentRunResult(exit_code=124, timed_out=True)
        self.clock.now += elapsed
        safe = result.output
        for secret in kwargs.get("output_redact_values", ()):
            safe = safe.replace(secret, "[REDACTED]")
        path = kwargs.get("output_path")
        if path is not None:
            with path.open("a") as output:
                output.write(safe)
        callback = kwargs.get("output_callback")
        if callback is not None:
            callback(safe)
        return result


def capacity_result(*, ansi: bool = False) -> AgentRunResult:
    output = HEADER + (f"\x1b[31m{CAPACITY}\x1b[0m" if ansi else CAPACITY) + "\n"
    return AgentRunResult(exit_code=1, output=output, full_output_captured=True)


def make_adapter(tmp_path, monkeypatch, results, *, agent="codex", public=False):
    clock = FakeClock()
    monkeypatch.setattr("rsi_harness.integrations.rsi_loop.time.monotonic", clock)
    runtime = SequenceRuntime(clock, results)
    config = RSILoopConfig(agent_api_key="provider-secret")
    adapter = RSILoopAgentAdapter(config, runtime=runtime)
    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={
                            "name": agent,
                            "model": "gpt-6-astra" if agent == "codex" else None,
                            "reasoning_effort": "max" if agent == "codex" else None,
                            "install_stop_hook": False,
                            "network": NetworkPolicy(
                                mode="public" if public else "no-network"
                            ),
                        }
                    )
                }
            )
        }
    )
    prepared = adapter.prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path / "prompt.md")
    )
    container = ContainerRef(container_id="same-live-work", role="work")
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=container,
            submit_url="http://control.internal:8123",
            token="control-secret",
        )
    )
    request = AgentRunRequest(
        prepared=prepared, container=container, timeout_seconds=600.0
    )
    return adapter, runtime, clock, request


@pytest.mark.parametrize("public", [False, True])
def test_capacity_retries_same_session_without_clearing_work_or_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, public: bool
) -> None:
    results = [
        capacity_result(ansi=True),
        capacity_result(),
        AgentRunResult(
            exit_code=0,
            output="continued training provider-secret control-secret\n",
            full_output_captured=True,
        ),
    ]
    adapter, runtime, clock, request = make_adapter(
        tmp_path, monkeypatch, results, public=public
    )
    deadlines, live = [], []
    output_path = tmp_path / "agent_output.txt"
    output_path.write_text("earlier attempt preserved\n")
    request = replace(
        request,
        on_exec_start=deadlines.append,
        output_path=output_path,
        output_callback=live.append,
        output_redact_values=("extra-secret",),
    )

    result = adapter.run(request)

    assert result.exit_code == 0
    assert result.full_output_captured
    assert len(runtime.executions) == 3
    assert deadlines == [1000.0]
    assert runtime.start_callbacks[1:] == [None, None]
    assert [call["timeout_seconds"] for call in runtime.executions] == [600, 598, 566]
    assert clock.now == 1096.0
    for index, call in enumerate(runtime.executions):
        assert call["container"] == "same-live-work"
        assert call["output_path"] == output_path
        assert call["output_callback"] is request.output_callback
        assert call["environment"]["RSI_TOKEN"] == "control-secret"
        assert call["environment"]["RSI_JUDGE_URL"] == "http://control.internal:8123"
        assert set(call["output_redact_values"]) >= {
            "control-secret",
            "provider-secret",
            "extra-secret",
        }
        if index:
            command = call["command"][-1]
            assert "codex exec resume" in command
            assert SESSION in command and "--last" not in command
            assert "--model gpt-6-astra" in command
            assert 'model_reasoning_effort="max"' in command
            assert ('web_search="disabled"' in command) is (not public)
            assert f"sleep {30 * index}" in command
    prompt = runtime.copied_text[PurePosixPath("/tmp/rsi-agent-capacity-retry.md")]
    assert prompt.strip()
    assert "provider-secret" not in prompt and "control-secret" not in prompt
    full_log = output_path.read_text()
    assert full_log.startswith("earlier attempt preserved\n")
    assert full_log.count(CAPACITY) == 2
    for output in (result.output, full_log, "".join(live)):
        assert CAPACITY in output and "continued training" in output
        assert "provider-secret" not in output and "control-secret" not in output
    with pytest.raises(InfrastructureError, match="control environment"):
        adapter.run(request)


@pytest.mark.parametrize(
    "first",
    [
        replace(capacity_result(), exit_code=0),
        replace(capacity_result(), exit_code=2),
        replace(capacity_result(), timed_out=True),
        replace(capacity_result(), cancelled=True),
        replace(capacity_result(), exec_started=False),
        replace(capacity_result(), output=HEADER + "ERROR: HTTP 401 Unauthorized\n"),
        replace(
            capacity_result(), output=HEADER + "ERROR: HTTP 400 unsupported model\n"
        ),
        replace(capacity_result(), output=HEADER + "ERROR: network disconnected\n"),
        replace(capacity_result(), output=CAPACITY + "\n"),
        replace(capacity_result(), output=HEADER + CAPACITY + "\nERROR: HTTP 401\n"),
    ],
    ids=[
        "success",
        "wrong-exit",
        "timeout",
        "cancel",
        "not-started",
        "auth",
        "unsupported",
        "network",
        "no-session",
        "later-auth",
    ],
)
def test_only_explicit_terminal_capacity_failure_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first: AgentRunResult
) -> None:
    adapter, runtime, _, request = make_adapter(tmp_path, monkeypatch, [first])

    result = adapter.run(request)

    assert len(runtime.executions) == 1
    assert result.exit_code == first.exit_code
    assert result.timed_out == first.timed_out
    assert result.cancelled == first.cancelled


def test_earlier_capacity_does_not_mask_later_authentication_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, runtime, _, request = make_adapter(
        tmp_path,
        monkeypatch,
        [
            capacity_result(),
            AgentRunResult(exit_code=1, output=HEADER + "ERROR: HTTP 401\n"),
        ],
    )

    result = adapter.run(request)

    assert len(runtime.executions) == 2
    assert result.exit_code == 1
    assert CAPACITY in result.output and "HTTP 401" in result.output


def test_capacity_is_recognized_before_codex_shutdown_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = (
        HEADER + CAPACITY + "\n"
        "2026-10-22T02:45:00.123Z ERROR codex_core::session: "
        "failed to record rollout items: channel closed\n"
        "2026-10-22T02:45:00.456Z ERROR codex_core::session: "
        "failed to record rollout items: channel closed\n"
        "tokens used\n328,685\n"
    )
    adapter, runtime, _, request = make_adapter(
        tmp_path,
        monkeypatch,
        [AgentRunResult(exit_code=1, output=output), AgentRunResult(exit_code=0)],
    )

    result = adapter.run(request)

    assert len(runtime.executions) == 2
    assert result.exit_code == 0
    assert output in result.output


@pytest.mark.parametrize("suffix", ["ordinary tool output\n", "fatal: bad config\n"])
def test_non_capacity_terminal_output_prevents_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    adapter, runtime, _, request = make_adapter(
        tmp_path,
        monkeypatch,
        [AgentRunResult(exit_code=1, output=HEADER + CAPACITY + "\n" + suffix)],
    )

    result = adapter.run(request)

    assert len(runtime.executions) == 1
    assert result.exit_code == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_cancellation_during_retry_propagates_and_clears_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[BaseException]
) -> None:
    adapter, runtime, _, request = make_adapter(
        tmp_path, monkeypatch, [capacity_result(), error()]
    )

    with pytest.raises(error):
        adapter.run(request)

    assert len(runtime.executions) == 2
    with pytest.raises(InfrastructureError, match="control environment"):
        adapter.run(request)


def test_session_quoted_later_by_tool_does_not_replace_current_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous_session = "01a15f85-9d09-7d71-ab60-5463a6a25b99"
    output = (
        HEADER + "exec\ncat old-trajectory.txt\n"
        f"session id: {previous_session}\n" + CAPACITY + "\n"
    )
    adapter, runtime, _, request = make_adapter(
        tmp_path,
        monkeypatch,
        [AgentRunResult(exit_code=1, output=output), AgentRunResult(exit_code=0)],
    )

    result = adapter.run(request)

    assert result.exit_code == 0
    command = runtime.executions[1]["command"][-1]
    assert SESSION in command
    assert previous_session not in command and "--last" not in command


@pytest.mark.parametrize("key", ["code", "codex_error_info"])
def test_structured_server_overloaded_error_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    output = HEADER + f'ERROR: {{"error": {{"{key}": "server_overloaded"}}}}\n'
    adapter, runtime, _, request = make_adapter(
        tmp_path,
        monkeypatch,
        [AgentRunResult(exit_code=1, output=output), AgentRunResult(exit_code=0)],
    )

    result = adapter.run(request)

    assert result.exit_code == 0
    assert len(runtime.executions) == 2


@pytest.mark.parametrize("budget", [2.0, 12.0])
def test_backoff_never_extends_original_time_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: float
) -> None:
    adapter, runtime, clock, request = make_adapter(
        tmp_path, monkeypatch, [capacity_result(), capacity_result()]
    )

    result = adapter.run(replace(request, timeout_seconds=budget))

    assert clock.now <= 1000.0 + budget
    assert result.timed_out
    assert len(runtime.executions) <= 2
    for execution in runtime.executions:
        assert 0 < execution["timeout_seconds"] <= budget


def test_backoff_is_capped_and_can_recover_after_repeated_overloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, runtime, clock, request = make_adapter(
        tmp_path,
        monkeypatch,
        [capacity_result() for _ in range(6)] + [AgentRunResult(exit_code=0)],
    )

    result = adapter.run(replace(request, timeout_seconds=2000.0))

    assert result.exit_code == 0
    delays = [
        float(re.search(r"\bsleep\s+([0-9.]+)", call["command"][-1]).group(1))
        for call in runtime.executions[1:]
    ]
    assert delays == [30, 60, 120, 240, 300, 300]
    assert clock.now == 1000 + sum(delays) + 2 * 7


@pytest.mark.parametrize("kind", ["other-agent", "custom-command", "unprepared"])
def test_retry_requires_adapter_prepared_codex_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    adapter, runtime, _, request = make_adapter(
        tmp_path,
        monkeypatch,
        [capacity_result()],
        agent="claude-code" if kind == "other-agent" else "codex",
    )
    if kind == "custom-command":
        request = replace(
            request,
            prepared=replace(
                request.prepared, command=("/bin/bash", "-c", "custom-agent")
            ),
        )
    elif kind == "unprepared":
        # This adapter receives a PreparedAgent supplied by a custom caller,
        # without the trusted recipe it would have stored during prepare().
        adapter.clear_transient_bindings()
        plan = make_run_plan(tmp_path)
        plan = plan.model_copy(
            update={
                "task": plan.task.model_copy(
                    update={
                        "agent": plan.task.agent.model_copy(
                            update={"install_stop_hook": False}
                        )
                    }
                )
            }
        )
        adapter.install_hooks(
            AgentHookRequest(
                run_plan=plan,
                container=request.container,
                submit_url="http://control.internal:8123",
                token="control-secret",
            )
        )

    result = adapter.run(request)

    assert result.exit_code == 1
    assert len(runtime.executions) == 1


def test_generated_capacity_resume_shell_runs_with_exact_session_and_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise shell quoting/stdin without Docker, model calls or real HOME."""
    monkeypatch.setattr("rsi_loop.harness.agent.codex._SESSION_STORE", "true")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls_path = tmp_path / "codex-calls.jsonl"
    codex = binaries / "codex"
    codex.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        f"with pathlib.Path({str(calls_path)!r}).open('a') as trace:\n"
        "    trace.write(json.dumps(args) + '\\n')\n"
        "assert args[0] == 'exec', args\n"
        "assert args[args.index('--model') + 1] == 'gpt-6-astra', args\n"
        "assert 'model_reasoning_effort=\"max\"' in args, args\n"
        "assert 'web_search=disabled' in args, args\n"
        "if args[1] == 'resume':\n"
        f"    assert {SESSION!r} in args and '--last' not in args, args\n"
        "    assert args[-1] == '-', args\n"
        "    prompt = sys.stdin.read()\n"
        "    assert 'temporarily at capacity' in prompt, prompt\n"
        "    assert 'background training' in prompt, prompt\n"
        "    print('SAME_SESSION_RECOVERED')\n"
        "else:\n"
        f"    print({HEADER!r}, end='', flush=True)\n"
        f"    print({CAPACITY!r}, file=sys.stderr, flush=True)\n"
        "    sys.exit(1)\n"
    )
    codex.chmod(0o755)
    sleep = binaries / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)

    class LocalRuntime(RecordingAgentRuntime):
        def __init__(self):
            super().__init__()
            self.targets = {}

        def copy_to(self, container, source, target):
            super().copy_to(container, source, target)
            destination = tmp_path / "container" / str(target).lstrip("/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            self.targets[target] = destination

        def exec(self, container, command, **kwargs):
            super().exec(container, command, **kwargs)
            argv = list(command)
            for target, destination in self.targets.items():
                argv = [part.replace(str(target), str(destination)) for part in argv]
            completed = subprocess.run(
                argv,
                cwd=tmp_path,
                env={
                    "PATH": f"{binaries}:{os.defpath}",
                    **kwargs.get("environment", {}),
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=kwargs["timeout_seconds"],
            )
            with kwargs["output_path"].open("a") as output:
                output.write(completed.stdout)
            return AgentRunResult(
                exit_code=completed.returncode,
                output=completed.stdout,
                full_output_captured=True,
            )

    plan = make_run_plan(tmp_path)
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "agent": plan.task.agent.model_copy(
                        update={
                            "model": "gpt-6-astra",
                            "reasoning_effort": "max",
                            "install_stop_hook": False,
                        }
                    )
                }
            )
        }
    )
    runtime = LocalRuntime()
    adapter = RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    prepared = adapter.prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path / "prompt.md")
    )
    work = ContainerRef(container_id="local-smoke-work", role="work")
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=work,
            submit_url="http://control.internal:8123",
            token="test-token",
        )
    )
    output_path = tmp_path / "agent_output.txt"
    output_path.write_text("old attempt\n")

    result = adapter.run(
        AgentRunRequest(
            prepared=prepared,
            container=work,
            timeout_seconds=10.0,
            output_path=output_path,
        )
    )

    assert result.exit_code == 0, result.output
    assert len(runtime.executions) == 2
    calls = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert len(calls) == 2 and calls[1][1] == "resume"
    for output in (result.output, output_path.read_text()):
        assert HEADER in output and CAPACITY in output
        assert "retry 1 in 30s" in output
        assert "SAME_SESSION_RECOVERED" in output
    assert output_path.read_text().startswith("old attempt\n")
