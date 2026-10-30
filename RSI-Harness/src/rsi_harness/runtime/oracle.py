"""Shared success checks for baseline Oracle runs on every backend."""

from rsi_harness.models import RunResult, RunStatus, SubmissionStatus


def oracle_failures(result: RunResult, submissions: int) -> list[str]:
    """Why an oracle run does not show a working Judge; empty when it does."""
    failures = []
    if result.agent_exit_code is None:
        failures.append("solve.sh returned no exit code")
    elif result.agent_exit_code != 0:
        failures.append(f"solve.sh exited {result.agent_exit_code}")
    if result.status is not RunStatus.COMPLETED:
        failures.append(f"run status is {result.status.value}")
    if result.total_rounds != submissions:
        failures.append(
            f"{result.total_rounds} Judge round(s) ran, expected {submissions}"
        )
    failures.extend(
        f"{report.round_id} is {report.status.value}"
        for report in result.reports
        if report.status is not SubmissionStatus.COMPLETED
    )
    return failures
