"""What an Agent is told in a run with one final judge."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter, _duration
from rsi_harness.models import (
    AgentPrepareRequest,
    AgentResume,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    RunGPUPlan,
)
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan

_DEADLINE = re.compile(r"deadline (\d{4}-\d\d-\d\d \d\d:\d\d) UTC")


def _plan(
    tmp_path: Path,
    *,
    budget: int | None = 1,
    agent: str = "codex",
    timeout: float = 21600.0,
    stop_hook: bool = True,
    visibility: str = "hidden",
):
    plan = make_run_plan(tmp_path)
    task = plan.task.model_copy(
        update={
            "instruction": "Solve the task.\n",
            "verifier": plan.task.verifier.model_copy(
                update={"max_submissions": budget, "visibility": visibility}
            ),
            "agent": plan.task.agent.model_copy(
                update={
                    "name": agent,
                    "timeout_seconds": timeout,
                    "install_stop_hook": stop_hook,
                }
            ),
        }
    )
    return plan.model_copy(update={"task": task})


def _prepare(tmp_path: Path, plan, **request):
    prompt = (tmp_path / "agent_prompt.md").resolve()
    prepared = RSILoopAgentAdapter(RSILoopConfig()).prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=prompt, **request)
    )
    return prompt.read_text(), prepared


def test_the_prompt_states_one_final_judge_its_budget_and_deadline(tmp_path):
    before = datetime.now(UTC).replace(second=0, microsecond=0)

    text, _ = _prepare(tmp_path, _plan(tmp_path), max_submissions=1)

    evaluation = text[text.index("## Evaluation") :]
    (deadline,) = _DEADLINE.findall(evaluation)
    stated = datetime.strptime(deadline, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    assert (
        before + timedelta(hours=6) <= stated <= before + timedelta(hours=6, minutes=2)
    )
    assert evaluation == (
        "## Evaluation\n\n"
        "This run has exactly one final judge. It evaluates the entire WORKDIR "
        "once,\nand that one result is the score of the run.\n"
        "You may call `rsi-submit` once to be judged early: the Judge then "
        "evaluates\nthe WORKDIR as it is at that moment, you receive the "
        "result, and the run\nends. Otherwise the Judge evaluates the WORKDIR "
        "automatically when your\ntime ends.\n"
        f"Time budget: 6 h of wall-clock time; deadline {deadline} UTC\n"
        "(check with `date -u`). Work stops at the deadline, so keep the "
        "WORKDIR\nin a state you want judged.\n"
        "Checks you run yourself score nothing; only the final judge does.\n"
    )
    for leftover in ("best valid", "repeatedly", "--list", "at most", "unlimited"):
        assert leftover not in evaluation


def test_the_prompt_variants_for_public_tests_gpus_and_no_stop_hook(tmp_path):
    plan = _plan(tmp_path, visibility="public", stop_hook=False, timeout=5400)
    devices = tuple(
        GPUDevice(index=index, uuid=f"GPU-{index}", name="H100") for index in (0, 1)
    )
    plan = plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=devices),
                work=GPUAllocation(devices=devices),
                judge=GPUAllocation(devices=devices),
                judge_mode=JudgeGPUMode.RELEASE_ALL,
            )
        }
    )

    text, _ = _prepare(tmp_path, plan, max_submissions=1)

    assert "If you stop earlier, the Judge evaluates the WORKDIR when you stop." in text
    assert "Time budget: 90 min of wall-clock time" in text
    assert "The Judge's tests are public: `/tests` holds, read-only" in text
    assert "Every Work GPU process must exit before rsi-submit" in text


def test_other_budgets_keep_the_rounds_prompt(tmp_path):
    text, _ = _prepare(tmp_path, _plan(tmp_path, budget=2), max_submissions=2)

    assert "exactly one final judge" not in text
    assert "You may submit to the Judge at most 2 times during this run." in text
    assert "The best valid primary score wins." in text


@pytest.mark.parametrize(
    ("budget", "expected"),
    [(1, "0"), (3, "3"), (None, "1")],
    ids=["final", "3", "none"],
)
def test_the_oracle_leaves_a_final_judge_round_to_the_harness(
    tmp_path, budget, expected
):
    _, prepared = _prepare(
        tmp_path, _plan(tmp_path, budget=budget, agent="oracle"), max_submissions=budget
    )

    assert dict(prepared.environment)["RSI_ORACLE_SUBMISSIONS"] == expected


def test_a_resumed_final_judge_attempt_is_given_its_new_deadline(tmp_path):
    plan = _plan(tmp_path)
    _prepare(tmp_path, plan, max_submissions=1)
    output = tmp_path / "agent_output.txt"
    output.write_text("")
    before = datetime.now(UTC).replace(second=0, microsecond=0)

    text, prepared = _prepare(
        tmp_path,
        plan,
        max_submissions=1,
        resume=AgentResume(
            attempt=2,
            message="rsi-harness: the run stopped and resumes now as attempt 2.\n",
            agent_output=output,
            seconds_left=1200.0,
        ),
    )

    attempt = text[text.index("## Attempt 2") :]
    (deadline,) = re.findall(r"New deadline: (\S+ \S+) UTC", attempt)
    stated = datetime.strptime(deadline, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
    assert before + timedelta(minutes=20) <= stated
    assert stated <= before + timedelta(minutes=22)
    assert "(check with `date -u`); it replaces any earlier deadline." in attempt


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(21600, "6 h"), (5400, "90 min"), (45, "45 s"), (3601, "3601 s"), (0, "0 s")],
)
def test_the_budget_is_written_in_its_largest_exact_unit(seconds, text):
    assert _duration(seconds) == text
