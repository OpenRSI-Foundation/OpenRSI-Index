"""Real Docker: a run with a submission budget of 1 has one final judge.

Run as a non-root user with the fake firewall, on the production run path.
The oracle Agent stands in for the Agent: each ``solve.sh`` behaves as one
kind of Agent (it exits, calls ``rsi-submit`` once and keeps working, never
calls it and runs into its time, or dies early), and the Judge can be slow.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import (
    CompileOptions,
    RunRequest,
    RunResult,
    RunStatus,
    SubmissionStatus,
)
from rsi_harness.runtime.oracle import oracle_failures
from rsi_harness.runtime.recovery import AttemptGrants, LeaseStore
from tests.integration.test_resume_docker import (
    _assert_cleanup_removes_everything,
    _labeled,
    _retained_ref,
    _services,
)

pytestmark = pytest.mark.integration

AGENT_TIMEOUT_SECONDS = 300
SLOW_JUDGE_SECONDS = 40

_TASK_TOML = """\
schema_version = "1.4"

[task]
name = "rsi/final-judge-check"

[environment]
os = "linux"
workdir = "{workdir}"
gpus = 0
network_mode = "no-network"

[agent]
timeout_sec = {agent_timeout}
network_mode = "no-network"
{agent_user}
[verifier]
timeout_sec = 300
network_mode = "no-network"
{verifier_metadata}"""

_TEST = """\
#!/bin/bash
set -eu
if [ -e {workdir_slash}slow-judge ]; then
    sleep {slow}
fi
answer=$(cat {workdir_slash}answer.txt 2>/dev/null || echo none)
if [ "$answer" = 42 ]; then reward=1; else reward=0; fi
echo "final-judge: answer=$answer"
printf '{{"reward": %s}}\\n' "$reward" > /logs/verifier/reward.json
"""

# Exits cleanly after solving: the Harness judges once it exits.
_SOLVE_EXITS = """\
#!/bin/bash
set -u
echo 42 > answer.txt
echo "final-judge: feedback=[$(ls -A /run/rsi-harness/feedback | tr '\\n' ' ')]"
"""

# Calls rsi-submit once, then keeps editing and calls it again: the first
# call is the score, the second is refused, and the Harness ends it.
_SOLVE_SUBMITS_THEN_EDITS = """\
#!/bin/bash
set -u
echo 42 > answer.txt
rsi-submit
echo "final-judge: first rsi-submit exited $?"
echo 0 > answer.txt
rsi-submit
echo "final-judge: second rsi-submit exited $?"
echo "final-judge: still working"
sleep 600
echo "final-judge: never ended"
"""

# Never calls rsi-submit and runs into its time; the Judge is slow. On a
# resumed attempt it writes a worse answer and exits.
_SOLVE_TIMES_OUT = """\
#!/bin/bash
set -u
marker="$HOME/final-judge-attempt-1"
if [ ! -e "$marker" ]; then
    touch "$marker"
    echo 42 > answer.txt
    touch slow-judge
    sync
    echo "final-judge: attempt-1 sleeping"
    sleep 600
    exit 9
fi
rm -f slow-judge
echo 7 > answer.txt
echo "final-judge: feedback=[$(ls -A /run/rsi-harness/feedback | tr '\\n' ' ')]"
rsi-submit --list
echo "final-judge: list exited $?"
"""

# Dies early: its launcher is killed, so the exec ends on a signal.
_SOLVE_CRASHES = """\
#!/bin/bash
echo 42 > answer.txt
echo "final-judge: crashing"
kill -KILL "$PPID"
"""


def _task(
    root: Path,
    solve: str,
    *,
    budget: int | None = 1,
    workdir: str = "/workspace",
) -> Path:
    task = root / "final-judge-check"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    non_root = workdir != "/"
    (task / "task.toml").write_text(
        _TASK_TOML.format(
            workdir=workdir,
            agent_timeout=AGENT_TIMEOUT_SECONDS,
            agent_user='user = "1000"\n' if non_root else "",
            verifier_metadata=(
                ""
                if budget is None
                else f"\n[metadata.rsi_harness.verifier]\nmax_submissions = {budget}\n"
            ),
        )
    )
    (task / "instruction.md").write_text("Write 42 to answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\n"
        + (
            "RUN mkdir -p /workspace && chown 1000:1000 /workspace\n"
            if non_root
            else ""
        )
        + f"WORKDIR {workdir}\n"
    )
    (task / "tests" / "test.sh").write_text(
        _TEST.format(workdir_slash=workdir.rstrip("/") + "/", slow=SLOW_JUDGE_SECONDS)
    )
    (task / "solution" / "solve.sh").write_text(solve)
    return task


def _request(task: Path, **options: Any) -> RunRequest:
    return RunRequest(
        task_dir=task,
        agent_name="oracle",
        options=CompileOptions(agent_name="oracle", **options),
    )


def _root(result: RunResult) -> Path:
    assert result.artifacts_dir is not None
    return result.artifacts_dir


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _cleanup(services, result: RunResult, *rounds: str) -> None:
    root = _root(result)
    _assert_cleanup_removes_everything(
        services,
        result.run_id,
        tuple(_retained_ref(result.run_id, root.name, name) for name in rounds),
    )


@pytest.mark.parametrize("workdir", ["/workspace", "/"], ids=["split", "full-rootfs"])
def test_the_oracle_is_judged_once_by_the_harness_after_it_exits(
    tmp_path: Path, request: pytest.FixtureRequest, workdir: str
) -> None:
    services = _services(tmp_path, request)
    task = _task(tmp_path, _SOLVE_EXITS, workdir=workdir)

    started = time.monotonic()
    result = services.run(_request(task))
    elapsed = time.monotonic() - started

    root = _root(result)
    output = (root / "agent_output.txt").read_text()
    assert result.status is RunStatus.COMPLETED, output[-4000:]
    assert oracle_failures(result, 1) == []
    assert [report.round_id for report in result.reports] == ["agent-1"]
    assert result.best_score == 1
    # The launcher left the round to the Harness, and Work saw no feedback.
    assert "rsi-oracle: submission" not in output
    assert "final-judge: feedback=[]" in output
    prompt = (root / "agent_prompt.md").read_text()
    assert "This run has exactly one final judge." in prompt
    assert "rsi-submit --list" not in prompt and "at most" not in prompt
    plan = _read(root / "run-plan.json")
    assert plan["task"]["verifier"]["max_submissions"] == 1
    assert plan["rootfs_snapshot_mode"] == (
        "split-workdir" if workdir != "/" else "full-rootfs"
    )
    final = _read(root / "final_result.json")
    assert final["timed_out"] is False
    assert final["agent_runtime_seconds"] < final["runtime_seconds"]
    assert elapsed < AGENT_TIMEOUT_SECONDS
    _cleanup(services, result, "final")


def test_the_agents_one_submit_is_the_score_and_ends_the_run(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    services = _services(tmp_path, request)
    task = _task(tmp_path, _SOLVE_SUBMITS_THEN_EDITS)

    result = services.run(_request(task))

    root = _root(result)
    output = (root / "agent_output.txt").read_text()
    assert result.status is RunStatus.COMPLETED, output[-4000:]
    # Judged as the WORKDIR was at the call; the edit after it is not.
    assert [report.round_id for report in result.reports] == ["agent-1"]
    assert result.best_score == 1
    assert "This was the run's one final judge; the run ends now." in output
    assert "final-judge: first rsi-submit exited 0" in output
    assert "rsi-submit can be called once in this run" in output
    assert "final-judge: second rsi-submit exited 22" in output
    assert "final-judge: still working" in output
    assert "final-judge: never ended" not in output
    final = _read(root / "final_result.json")
    assert final["timed_out"] is False
    # Ended about ten seconds after the feedback, not at the Agent timeout.
    assert final["agent_runtime_seconds"] < 120
    assert final["first_submission_seconds"] is not None
    assert _read(root / "submissions" / "agent-1" / "final_judge.json") == {
        "round": "agent-1",
        "requested_by": "agent",
    }
    # The Agent saw its result: the run is finished, not resumable.
    with pytest.raises(SetupError, match="own rsi-submit"):
        services.plan_resume(
            result.run_id, AttemptGrants(add_time_seconds=60.0), dry_run=True
        )
    _cleanup(services, result, "final")


def test_a_timed_out_agent_is_judged_outside_its_budget_and_resumes(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    services = _services(tmp_path, request)
    task = _task(tmp_path, _SOLVE_TIMES_OUT)

    first = services.run(_request(task, agent_timeout_seconds=20))

    root = _root(first)
    output = (root / "agent_output.txt").read_text()
    assert first.status is RunStatus.COMPLETED, output[-4000:]
    assert [report.round_id for report in first.reports] == ["agent-1"]
    assert first.best_score == 1
    assert first.reports[0].duration_seconds >= SLOW_JUDGE_SECONDS
    final = _read(root / "final_result.json")
    assert final["timed_out"] is True
    assert final["agent_runtime_seconds"] < SLOW_JUDGE_SECONDS
    # The Harness's round, not an rsi-submit of the Agent.
    assert final["first_submission_seconds"] is None
    assert _read(root / "submissions" / "agent-1" / "final_judge.json") == {
        "round": "agent-1",
        "requested_by": "harness",
    }
    assert final["runtime_seconds"] > 20 + SLOW_JUDGE_SECONDS
    lease = LeaseStore(tmp_path / "data" / "leases").read(first.run_id)
    assert lease is not None
    [attempt] = lease.attempts
    assert attempt.agent_seconds == pytest.approx(20, abs=3)
    assert _labeled("containers", first.run_id) == []

    context = services.plan_resume(first.run_id, AttemptGrants(add_time_seconds=60.0))
    assert context.final_judge
    assert context.next_round_id == "agent-2"
    assert "This attempt ends with one final judge" in context.message
    assert "agent-1" not in context.message

    second = services.resume(context)

    output = (root / "agent_output.txt").read_text()
    assert second.status is RunStatus.COMPLETED, output[-4000:]
    assert [report.round_id for report in second.reports] == ["agent-1", "agent-2"]
    # The newest round is the score, though agent-1 scored higher.
    assert (second.best_round, second.best_score) == ("agent-2", 0)
    # The resumed attempt saw neither the earlier result nor its log.
    assert "final-judge: feedback=[]" in output
    assert "rsi-submit --list is not available in this run" in output
    prompt = (root / "agent_prompt.md").read_text()
    attempt_2 = prompt[prompt.index("## Attempt 2") :]
    assert "New deadline: " in attempt_2
    assert "it replaces any earlier deadline" in attempt_2
    final = _read(root / "final_result.json")
    assert (final["best_round"], final["best_score"], final["attempts"]) == (
        "agent-2",
        0.0,
        2,
    )
    state = _read(root / "evolve_state.json")
    assert state["best_round"] == "agent-2"
    _cleanup(services, second, "final", "final-a2")


def test_an_agent_that_dies_early_is_not_judged(
    tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    services = _services(tmp_path, request)
    task = _task(tmp_path, _SOLVE_CRASHES)

    result = services.run(_request(task))

    root = _root(result)
    output = (root / "agent_output.txt").read_text()
    assert "final-judge: crashing" in output
    assert result.status is RunStatus.FAILED
    assert result.reports == ()
    assert not (root / "submissions").exists() or not any(
        (root / "submissions").iterdir()
    )
    error = _read(root / "engine_error.json")["error"]
    assert "before its time ended" in error
    # Its time is left for a resume.
    context = services.plan_resume(result.run_id, AttemptGrants(), dry_run=True)
    assert context.final_judge and context.next_round_id == "agent-1"
    _cleanup(services, result, "final")


@pytest.mark.parametrize(
    ("budget", "flag", "rounds"),
    [(1, 2, 2), (None, 1, 1), (3, None, 3)],
    ids=["task-1-flag-2", "flag-1", "task-3"],
)
def test_the_task_budget_and_the_flag_over_it(
    tmp_path: Path, request: pytest.FixtureRequest, budget, flag, rounds
) -> None:
    services = _services(tmp_path, request)
    task = _task(tmp_path, _SOLVE_EXITS, budget=budget)

    result = services.run(_request(task, max_submissions=flag))

    root = _root(result)
    output = (root / "agent_output.txt").read_text()
    assert result.status is RunStatus.COMPLETED, output[-4000:]
    assert oracle_failures(result, rounds) == []
    assert all(r.status is SubmissionStatus.COMPLETED for r in result.reports)
    prompt = (root / "agent_prompt.md").read_text()
    final_judge = rounds == 1
    assert ("exactly one final judge" in prompt) is final_judge
    if final_judge:
        assert "rsi-oracle: submission" not in output
    else:
        # Any other budget: the rounds prompt, and the oracle submits N times.
        assert f"You may submit to the Judge at most {rounds} times" in prompt
        assert f"rsi-oracle: submission {rounds}/{rounds}" in output
        assert "remaining: 0" in output
        assert "Full verifier output: /run/rsi-harness/feedback/agent-1.log" in output
    assert _read(root / "run-plan.json")["task"]["verifier"]["max_submissions"] == (
        flag if flag is not None else budget
    )
    _cleanup(services, result, "final")
