"""Run a staged intake task the way OpenRSI-X runs its own tasks.

Every phase is a plain ``rsi-harness run`` in a child process, configured by
the server's run profile; the task's own ``[rsi]`` evaluation options are
added per task. The result handed back upstream is the run's own
``final_result.json``, the same summary every OpenRSI run produces.

    lint      compile the staged task (no containers)
    prepare   lint, then ``--dry-run`` on the configured cluster, if any
    validate  lint, then the oracle Agent runs the task's solution/
    run       validate, then the configured Agent runs the task
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from rsi_harness.intake.protocol import MAX_DETAIL_CHARS, ExecutionMode
from rsi_harness.intake.staging import StagedTask
from rsi_harness.runtime.redaction import redact_text

ORACLE_AGENT = "oracle"
_TAIL_LINES = 40


@dataclass(frozen=True)
class ExecutionOutcome:
    ok: bool
    detail: str
    result: dict[str, Any] = field(default_factory=dict)


class Executor(Protocol):
    def execute(
        self, mode: ExecutionMode, staged: StagedTask, work_dir: Path
    ) -> ExecutionOutcome: ...


@dataclass(frozen=True)
class RunProfile:
    """The operator's ``rsi-harness run`` settings for intake tasks."""

    agent: str = "codex"
    model: str | None = None
    reasoning_effort: str | None = None
    agent_auth: str | None = None
    timeout: float | None = None
    gpus: str = ""
    cluster: str | None = None
    sandbox_policy: Path | None = None
    judge_llm_auth: str | None = None
    judge_llm_api_key_env: str | None = None
    judge_llm_api_key_file: Path | None = None
    judge_llm_base_url: str | None = None
    judge_llm_proxy: str | None = None
    data_root: Path = Path(".rsi-harness")
    verbose: bool = False

    def shared_args(self) -> list[str]:
        args: list[str] = ["--data-root", str(self.data_root)]
        pairs: list[tuple[str, object | None]] = [
            ("--gpus", self.gpus or None),
            ("--cluster", self.cluster),
            ("--sandbox-policy", self.sandbox_policy),
            ("--judge-llm-auth", self.judge_llm_auth),
            ("--judge-llm-api-key-env", self.judge_llm_api_key_env),
            ("--judge-llm-api-key-file", self.judge_llm_api_key_file),
            ("--judge-llm-base-url", self.judge_llm_base_url),
            ("--judge-llm-proxy", self.judge_llm_proxy),
        ]
        for flag, value in pairs:
            if value is not None:
                args += [flag, str(value)]
        if self.verbose:
            args.append("--verbose")
        return args

    def agent_args(self) -> list[str]:
        args = ["--agent", self.agent]
        for flag, value in (
            ("--model", self.model),
            ("--reasoning-effort", self.reasoning_effort),
            ("--agent-auth", self.agent_auth),
            ("--timeout", self.timeout),
        ):
            if value is not None:
                args += [flag, str(value)]
        return args


def _task_args(staged: StagedTask) -> list[str]:
    options = staged.run_options
    args: list[str] = []
    if options.score_direction:
        args += ["--score-direction", options.score_direction]
    if options.cooldown_seconds:
        args += ["--cooldown", str(options.cooldown_seconds)]
    if options.primary_reward:
        args += ["--primary-reward", options.primary_reward]
    return args


def _tail(log: Path) -> str:
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return redact_text("\n".join(lines[-_TAIL_LINES:]))


def _clip(text: str) -> str:
    return (
        text if len(text) <= MAX_DETAIL_CHARS else "…" + text[-(MAX_DETAIL_CHARS - 1) :]
    )


def _final_result(logs_root: Path) -> tuple[str | None, dict[str, Any] | None]:
    """The newest run's ``final_result.json`` under ``logs_root`` and its run id."""
    found = sorted(
        logs_root.rglob("final_result.json"), key=lambda p: p.stat().st_mtime
    )
    if not found:
        return None, None
    path = found[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    run_id = None
    parts = path.relative_to(logs_root).parts
    if len(parts) >= 3 and parts[0] == "runs":
        run_id = parts[1]
    return run_id, payload if isinstance(payload, dict) else None


Runner = Callable[[list[str], Path, Path, dict[str, str]], int]


def _subprocess_runner(
    command: list[str], cwd: Path, log: Path, env: dict[str, str]
) -> int:
    with log.open("wb") as out:
        return subprocess.run(
            command, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=env, check=False
        ).returncode


class HarnessExecutor:
    def __init__(self, profile: RunProfile, runner: Runner | None = None) -> None:
        self.profile = profile
        self._runner = runner or _subprocess_runner

    # --- phases ---

    def _lint(self, staged: StagedTask) -> ExecutionOutcome:
        from rsi_harness.models import CompileOptions
        from rsi_harness.task.compiler import HarborTaskCompiler

        options = staged.run_options
        try:
            HarborTaskCompiler().compile(
                staged.task_dir,
                CompileOptions(
                    agent_name=self.profile.agent,
                    primary_reward=options.primary_reward,
                    score_direction=options.score_direction or "maximize",
                    cooldown_seconds=options.cooldown_seconds or 0.0,
                ),
            )
        except Exception as error:  # the compiler's own message is the reason
            return ExecutionOutcome(
                False, _clip(f"lint failed: {error}"), {"lint_ok": False}
            )
        return ExecutionOutcome(True, "lint passed", {"lint_ok": True})

    def _harness_run(
        self, staged: StagedTask, work_dir: Path, phase: str, extra: list[str]
    ) -> dict[str, Any]:
        phase_dir = work_dir / phase
        logs_root = phase_dir / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)
        log = phase_dir / "run.log"
        command = [
            sys.executable,
            "-m",
            "rsi_harness.cli",
            "run",
            str(staged.task_dir),
            "--logs-root",
            str(logs_root),
            *self.profile.shared_args(),
            *_task_args(staged),
            *extra,
        ]
        # The intake credentials are the server's, never the run's.
        env = {k: v for k, v in os.environ.items() if not k.startswith("RSI_INTAKE_")}
        exit_code = self._runner(command, phase_dir, log, env)
        run_id, final = _final_result(logs_root)
        return {
            "exit_code": exit_code,
            "run_id": run_id,
            "final_result": final,
            "log": str(log),
            "tail": _tail(log) if exit_code != 0 else "",
        }

    def execute(
        self, mode: ExecutionMode, staged: StagedTask, work_dir: Path
    ) -> ExecutionOutcome:
        lint = self._lint(staged)
        if not lint.ok or mode is ExecutionMode.LINT:
            return lint
        result: dict[str, Any] = dict(lint.result)

        if mode is ExecutionMode.PREPARE:
            if self.profile.cluster is None:
                return ExecutionOutcome(
                    True, "prepared (lint passed; no cluster)", result
                )
            dry = self._harness_run(
                staged, work_dir, "prepare", [*self.profile.agent_args(), "--dry-run"]
            )
            result["prepare"] = dry
            if dry["exit_code"] != 0:
                return ExecutionOutcome(
                    False, _clip(f"prepare failed:\n{dry['tail']}"), result
                )
            return ExecutionOutcome(True, "prepared (cluster dry run passed)", result)

        oracle = self._harness_run(
            staged, work_dir, "validate", ["--agent", ORACLE_AGENT]
        )
        oracle["oracle_passed"] = oracle["exit_code"] == 0
        result["validation"] = oracle
        if not oracle["oracle_passed"]:
            return ExecutionOutcome(
                False, _clip(f"validation (oracle) failed:\n{oracle['tail']}"), result
            )
        if mode is ExecutionMode.VALIDATE:
            result.update(
                run_id=oracle["run_id"],
                exit_code=oracle["exit_code"],
                final_result=oracle["final_result"],
            )
            return ExecutionOutcome(True, "validated (oracle check passed)", result)

        run = self._harness_run(staged, work_dir, "run", self.profile.agent_args())
        result.update(
            run_id=run["run_id"],
            exit_code=run["exit_code"],
            final_result=run["final_result"],
            run_log=run["log"],
        )
        if run["exit_code"] != 0:
            return ExecutionOutcome(False, _clip(f"run failed:\n{run['tail']}"), result)
        final = run["final_result"] or {}
        return ExecutionOutcome(
            True,
            f"completed: status={final.get('status', '?')} "
            f"best_score={final.get('best_score', '-')}",
            result,
        )
