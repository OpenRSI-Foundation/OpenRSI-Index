"""Rerun rubric reviews that failed only because the review model provider was unavailable.

A provider outage (repeated 5xx or dropped connections) outlasts the review's own
30-minute retry window, and the Discussion then waits for a maintainer. This runs
hourly: for each Discussion whose latest rubric review run failed with the fixed
provider diagnostic, it reruns that run, at most MAX_AUTOMATIC_RERUNS times and
only after the failure is RETRY_AFTER_SECONDS old. Authentication failures and any
other errors are left for a maintainer. A rerun rechecks the current proposal, so a
proposal edited since the failure is never reviewed from stale content.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
import re
import subprocess
import sys

WORKFLOW = "discussion-review.yml"
REVIEW_JOB = "review"
MAX_AUTOMATIC_RERUNS = 3
RETRY_AFTER_SECONDS = 45 * 60
LOOKBACK_DAYS = 3
# Fixed lines printed by checks/rubric_review.py; provider response bodies are never logged.
PROVIDER_FAILURE = "Review provider request failed ("
AUTHENTICATION_FAILURE = "A maintainer must repair the review provider authentication"
TITLE = re.compile(r"Discussion #([1-9][0-9]*) rubric review\Z")
ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def gh(args: list[str]) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return result.stdout


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def recent_runs(api, repository: str, now: datetime) -> list[dict]:
    since = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    output = api(["api", "--paginate", "--jq", ".workflow_runs[]|@json",
                  f"repos/{repository}/actions/workflows/{WORKFLOW}/runs?created=>={since}&per_page=100"])
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def latest_per_discussion(runs: list[dict]) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for run in runs:
        match = TITLE.fullmatch(str(run.get("display_title", "")))
        if not match:
            continue
        number = int(match.group(1))
        if number not in latest or run["created_at"] > latest[number]["created_at"]:
            latest[number] = run
    return latest


def provider_failure(api, repository: str, run: dict) -> bool:
    jobs = json.loads(api(["api", f"repos/{repository}/actions/runs/{run['id']}"
                                  f"/attempts/{run['run_attempt']}/jobs?per_page=100"]))
    review = [job for job in jobs.get("jobs", []) if job.get("name") == REVIEW_JOB]
    if len(review) != 1 or review[0].get("conclusion") != "failure":
        return False
    # Logs carry ANSI colour codes, which gh prints only when explicitly allowed.
    log = ESCAPE.sub("", api(["api", "--allow-escape-sequences",
                              f"repos/{repository}/actions/jobs/{review[0]['id']}/logs"]))
    return PROVIDER_FAILURE in log and AUTHENTICATION_FAILURE not in log


def candidates(api, repository: str, now: datetime) -> list[tuple[int, dict]]:
    chosen = []
    for number, run in sorted(latest_per_discussion(recent_runs(api, repository, now)).items()):
        if run.get("status") != "completed" or run.get("conclusion") != "failure":
            continue
        if int(run.get("run_attempt", 1)) > MAX_AUTOMATIC_RERUNS:
            continue
        if (now - parse_time(run["updated_at"])).total_seconds() < RETRY_AFTER_SECONDS:
            continue
        if provider_failure(api, repository, run):
            chosen.append((number, run))
    return chosen


def main(argv: list[str] | None = None, *, api=gh, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    repository = os.environ["GITHUB_REPOSITORY"]
    now = now or datetime.now(timezone.utc)
    chosen = candidates(api, repository, now)
    for number, run in chosen:
        action = "would rerun" if args.dry_run else "rerunning"
        print(f"Discussion #{number}: {action} run {run['id']} "
              f"(attempt {run['run_attempt']} failed on the review provider)")
        if not args.dry_run:
            api(["api", "-X", "POST", f"repos/{repository}/actions/runs/{run['id']}/rerun"])
    if not chosen:
        print("No provider-failed rubric reviews to rerun.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
