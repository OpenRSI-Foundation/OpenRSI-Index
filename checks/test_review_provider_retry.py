"""Hourly rerun of rubric reviews that failed only on the review model provider."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_provider_retry as retry  # noqa: E402

REPO = "OpenRSI-Foundation/OpenRSI-Index"
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
PROVIDER_LOG = ("\x1b[36mOpenAI temporary error (status=503, code=None); retry 10/12\x1b[0m\n"
                "Review provider request failed (HTTP 503). No review was completed.\n")
AUTH_LOG = ("Review provider request failed (HTTP 401). No review was completed.\n"
            "A maintainer must repair the review provider authentication before rerunning.\n")
OTHER_LOG = "Judge returned a missing or non-canonical decision: None\n"


def run(run_id, number, *, created="2026-11-22T08:00:00Z", updated="2026-11-22T08:30:00Z",
        conclusion="failure", status="completed", attempt=1):
    return {"id": run_id, "display_title": f"Discussion #{number} rubric review", "created_at": created,
            "updated_at": updated, "status": status, "conclusion": conclusion, "run_attempt": attempt}


class FakeApi:
    def __init__(self, runs, logs, review_conclusion="failure"):
        self.runs, self.logs, self.review_conclusion = runs, logs, review_conclusion
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        path = args[-1]
        if "/actions/workflows/discussion-review.yml/runs?created=>=2026-11-19T12:00:00Z" in path:
            assert args[:4] == ["api", "--paginate", "--jq", ".workflow_runs[]|@json"]
            return "\n".join(json.dumps(item) for item in self.runs) + "\n"
        if path.endswith("/rerun"):
            assert args[:3] == ["api", "-X", "POST"]
            return ""
        if "/attempts/" in path:
            run_id = int(path.split("/actions/runs/")[1].split("/")[0])
            return json.dumps({"jobs": [{"name": "reaction-notice", "id": 1, "conclusion": "success"},
                                        {"name": "review", "id": run_id * 10, "conclusion": self.review_conclusion}]})
        if "/actions/jobs/" in path:
            return self.logs[int(path.split("/actions/jobs/")[1].split("/")[0]) // 10]
        raise AssertionError(path)

    def reruns(self):
        return [args[-1] for args in self.calls if args[-1].endswith("/rerun")]


@pytest.fixture(autouse=True)
def repository(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)


def test_reruns_only_provider_failures_that_are_latest_old_enough_and_under_the_cap(capsys):
    runs = [
        run(1, 384),                                                    # provider failure: rerun
        run(2, 385), run(3, 385, created="2026-11-22T09:00:00Z", conclusion="success"),  # superseded
        run(4, 386, attempt=4),                                         # already rerun three times
        run(5, 387, updated="2026-11-22T11:30:00Z"),                    # failed only 30 minutes ago
        run(6, 388),                                                    # authentication: maintainer
        run(7, 389),                                                    # other error: maintainer
        run(8, 390, status="in_progress", conclusion=None),             # still running
        {**run(9, 391), "display_title": "Something else"},
    ]
    api = FakeApi(runs, {1: PROVIDER_LOG, 2: PROVIDER_LOG, 4: PROVIDER_LOG, 5: PROVIDER_LOG,
                         6: AUTH_LOG, 7: OTHER_LOG})
    assert retry.main([], api=api, now=NOW) == 0
    assert api.reruns() == [f"repos/{REPO}/actions/runs/1/rerun"]
    assert "Discussion #384: rerunning run 1" in capsys.readouterr().out


def test_a_rerun_attempt_that_fails_again_is_retried_until_the_cap():
    api = FakeApi([run(1, 384, attempt=3)], {1: PROVIDER_LOG})
    retry.main([], api=api, now=NOW)
    assert api.reruns() == [f"repos/{REPO}/actions/runs/1/rerun"]
    assert any("/attempts/3/jobs" in args[-1] for args in api.calls)


def test_dry_run_starts_nothing(capsys):
    api = FakeApi([run(1, 384)], {1: PROVIDER_LOG})
    retry.main(["--dry-run"], api=api, now=NOW)
    assert api.reruns() == [] and "would rerun run 1" in capsys.readouterr().out


def test_a_run_whose_review_job_did_not_fail_is_left_alone():
    api = FakeApi([run(1, 384)], {1: PROVIDER_LOG}, review_conclusion="success")
    retry.main([], api=api, now=NOW)
    assert api.reruns() == []


def test_nothing_to_do(capsys):
    retry.main([], api=FakeApi([], {}), now=NOW)
    assert "No provider-failed rubric reviews to rerun." in capsys.readouterr().out


def test_workflow_is_hourly_least_privilege_and_calls_the_helper():
    path = Path(__file__).resolve().parents[1] / ".github/workflows/discussion-review-provider-retry.yml"
    workflow = yaml.safe_load(path.read_text())
    triggers = workflow[True]
    assert triggers["schedule"] == [{"cron": "23 * * * *"}] and "workflow_dispatch" in triggers
    assert workflow["permissions"] == {"actions": "write", "contents": "read"}
    steps = workflow["jobs"]["rerun"]["steps"]
    assert steps[0]["with"]["persist-credentials"] is False
    assert "python3 checks/review_provider_retry.py" in steps[1]["run"]
    assert steps[1]["env"]["GH_TOKEN"] == "${{ github.token }}"


def test_the_diagnostics_match_what_the_review_prints():
    source = (Path(__file__).resolve().parent / "rubric_review.py").read_text()
    assert 'f"Review provider request failed ({detail}).' in source
    assert retry.PROVIDER_FAILURE == "Review provider request failed ("
    assert retry.AUTHENTICATION_FAILURE in source
