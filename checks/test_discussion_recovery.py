from __future__ import annotations

import importlib
import json
import subprocess
import os
from pathlib import Path

import pytest
import yaml

from checks.github_retry import GitHubTransientError
from checks.proposal_review_marker import canonical_proposal_hash, render_review_marker


@pytest.fixture
def recovery():
    assert Path(__file__).with_name("discussion_recovery.py").exists(), (
        "Discussion requests need transport recovery and mutation reconciliation"
    )
    return importlib.import_module("checks.discussion_recovery")


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.waits = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


class CommentServer:
    def __init__(self, *, lose_response=False):
        self.comments = []
        self.creates = 0
        self.lose_response = lose_response

    def __call__(self, args):
        fields = dict(
            args[index + 1].split("=", 1)
            for index, value in enumerate(args)
            if value == "-f"
        )
        query = fields["query"]
        if "addDiscussionComment" in query:
            self.creates += 1
            comment = {
                "id": f"DC_{self.creates}",
                "url": "https://github.com/discussion",
                "body": fields["body"],
                "author": {"login": "dispatcher[bot]"},
                "createdAt": "2026-09-13T10:00:00Z",
                "updatedAt": "2026-09-13T10:00:00Z",
            }
            self.comments.append(comment)
            if self.lose_response:
                self.lose_response = False
                raise GitHubTransientError(operation="discussion.comment")
            return json.dumps({"data": {"addDiscussionComment": {"comment": comment}}})
        assert "comments(first:" in query
        return json.dumps(
            {
                "data": {
                    "node": {
                        "id": "D_one",
                        "comments": {
                            "nodes": self.comments,
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        },
                    }
                }
            }
        )


def test_gh_read_retries_503_with_finite_timeout_and_no_secret_output(recovery):
    clock = Clock()
    responses = [
        subprocess.CompletedProcess(
            [],
            1,
            'HTTP/2.0 503 Service Unavailable\n\n{"message":"secret"}',
            "gh: secret (HTTP 503)",
        ),
        subprocess.CompletedProcess([], 0, 'HTTP/2.0 200 OK\n\n{"role":"admin"}', ""),
    ]
    timeouts = []

    def runner(args, **kwargs):
        timeouts.append(kwargs["timeout"])
        return responses.pop(0)

    result = recovery.gh_request(
        ["api", "/orgs/RSI-Index/memberships/alice"],
        runner=runner,
        sleep=clock.sleep,
        clock=clock.time,
    )
    assert json.loads(result) == {"role": "admin"}
    assert clock.waits == [1]
    assert timeouts == [30, 30]


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_permanent_gh_error_is_not_retried_or_printed(recovery, status):
    clock = Clock()

    def runner(args, **kwargs):
        return subprocess.CompletedProcess(
            [],
            1,
            f'HTTP/2.0 {status} Error\n\n{{"message":"private-secret"}}',
            "private-secret",
        )

    with pytest.raises(recovery.GitHubRequestError) as caught:
        recovery.gh_request(
            ["api", "/test"], runner=runner, sleep=clock.sleep, clock=clock.time
        )
    assert caught.value.status == status
    assert "private-secret" not in str(caught.value)
    assert clock.waits == []


def test_owner_lookup_does_not_turn_outage_into_denial(recovery):
    clock = Clock()

    def unavailable(args):
        raise GitHubTransientError(503, operation="discussion.owner")

    with pytest.raises(GitHubTransientError):
        recovery.owner_lookup(
            recovery.GitHubClient(
                call=unavailable, sleep=clock.sleep, clock=clock.time
            ),
            "alice",
        )

    def denied(args):
        raise recovery.GitHubRequestError(404)

    assert recovery.owner_lookup(recovery.GitHubClient(call=denied), "alice") is False


def test_queue_creation_reconciles_lost_response_and_preserves_history(recovery):
    clock = Clock()
    server = CommentServer(lose_response=True)
    original = {
        "id": "DC_old",
        "body": "Task started",
        "author": {"login": "dispatcher[bot]"},
    }
    server.comments.append(original.copy())
    client = recovery.GitHubClient(call=server, sleep=clock.sleep, clock=clock.time)
    for attempt in range(2):
        result = recovery.ensure_comment(
            client,
            "D_one",
            recovery.QUEUE_BODY,
            operation="task.queue",
            run_id="123",
            episode_id="DC_trigger",
            bot_login="dispatcher[bot]",
        )
        assert result["id"] == "DC_1"
    assert server.creates == 1
    assert server.comments[0] == original
    assert (
        server.comments[1]["body"].split("\n\n")[0]
        == "⏳ Your task is queued and will start automatically when a runner is available. No action is needed."
    )
    assert "123" in server.comments[1]["body"]
    assert "DC_trigger" in server.comments[1]["body"]


def test_recovery_notices_deduplicate_episode_and_keep_exhaustion_distinct(recovery):
    server = CommentServer(lose_response=True)
    clock = Clock()
    client = recovery.GitHubClient(call=server, sleep=clock.sleep, clock=clock.time)
    for status in ["retrying", "retrying", "exhausted", "exhausted"]:
        recovery.post_recovery_notice(
            client,
            discussion_id="D_one",
            source_repository="RSI-Index/RSI-Skills",
            source_run_id="123",
            episode_id="episode-1",
            status=status,
            bot_login="dispatcher[bot]",
        )
    assert server.creates == 2
    assert "automatically" in server.comments[0]["body"]
    assert "24-hour limit" in server.comments[1]["body"]
    assert "retrying" in server.comments[0]["body"]
    assert "exhausted" in server.comments[1]["body"]


def test_foreign_bot_cannot_suppress_our_notice(recovery):
    server = CommentServer()
    client = recovery.GitHubClient(call=server)
    arguments = dict(
        operation="task.queue",
        run_id="123",
        episode_id="DC_trigger",
        bot_login="dispatcher[bot]",
    )
    recovery.ensure_comment(client, "D_one", recovery.QUEUE_BODY, **arguments)
    server.comments[0]["author"]["login"] = "alice"
    recovery.ensure_comment(client, "D_one", recovery.QUEUE_BODY, **arguments)
    assert server.creates == 2


def test_completed_review_reuse_requires_current_hash_id_bot_and_completed_status(
    recovery,
):
    marker = render_review_marker(
        "Pass", canonical_proposal_hash("Title", "Proposal"), "D_one"
    )
    comment = {
        "id": "DC_review",
        "author": {"login": "dispatcher[bot]"},
        "body": "<!-- rubric-review-status:completed -->\n" + marker,
        "updatedAt": "2026-09-13T12:00:00Z",
        "createdAt": "2026-09-13T11:00:00Z",
    }
    assert recovery.find_completed_review(
        [comment], "Title", "Proposal", "D_one", "dispatcher[bot]"
    ) == {"decision": "Pass", "review_comment_id": "DC_review"}
    assert (
        recovery.find_completed_review(
            [comment], "Title", "Changed", "D_one", "dispatcher[bot]"
        )
        is None
    )
    assert (
        recovery.find_completed_review(
            [comment], "Title", "Proposal", "D_other", "dispatcher[bot]"
        )
        is None
    )
    assert (
        recovery.find_completed_review(
            [comment], "Title", "Proposal", "D_one", "other[bot]"
        )
        is None
    )
    comment["body"] = marker
    assert (
        recovery.find_completed_review(
            [comment], "Title", "Proposal", "D_one", "dispatcher[bot]"
        )
        is None
    )


def test_dispatch_observes_accepted_run_after_lost_response_without_reposting(recovery):
    clock = Clock()
    posts = []
    observations = []

    def call(args):
        if any(arg.endswith("/dispatches") for arg in args):
            posts.append(args)
            raise GitHubTransientError(503, operation="discussion.dispatch")
        observations.append(args)
        runs = (
            []
            if len(observations) < 3
            else [
                {
                    "id": 90,
                    "display_title": "Discussion #66 task:DC_trigger",
                    "event": "repository_dispatch",
                }
            ]
        )
        return json.dumps({"workflow_runs": runs})

    client = recovery.GitHubClient(call=call, sleep=clock.sleep, clock=clock.time)
    request = {
        "event_type": "discussion_task_command",
        "client_payload": {
            "discussion_number": 66,
            "triggering_comment_node_id": "DC_trigger",
        },
    }
    recovery.dispatch_request(client, request)
    assert len(posts) == 1
    assert len(observations) == 3
    assert clock.waits == [1]


def test_ambiguous_dispatch_without_visible_run_yields_without_duplicate(recovery):
    clock = Clock()
    posts = []

    def call(args):
        if any(arg.endswith("/dispatches") for arg in args):
            posts.append(args)
            raise GitHubTransientError(operation="discussion.dispatch")
        return json.dumps({"workflow_runs": []})

    client = recovery.GitHubClient(call=call, sleep=clock.sleep, clock=clock.time)
    with pytest.raises(GitHubTransientError):
        recovery.dispatch_request(
            client,
            {
                "event_type": "discussion_task_reset",
                "client_payload": {
                    "discussion_number": 66,
                    "triggering_comment_node_id": "DC_reset",
                },
            },
        )
    assert len(posts) == 1
    assert clock.waits == [1, 5, 10, 30, 60]


@pytest.mark.parametrize("refresh,retry_after", [(True, 0), (False, 600)])
def test_dispatch_yields_before_polling_with_expired_credentials_or_long_rate_limit(
    recovery, refresh, retry_after
):
    clock, reads = Clock(), []

    def call(args):
        if any(arg.endswith("/dispatches") for arg in args):
            raise GitHubTransientError(
                403,
                operation="discussion.dispatch",
                refresh_credentials=refresh,
                retry_after=retry_after,
            )
        reads.append(args)
        return json.dumps({"workflow_runs": []})

    with pytest.raises(GitHubTransientError) as caught:
        recovery.dispatch_request(
            recovery.GitHubClient(call=call, sleep=clock.sleep, clock=clock.time),
            {
                "event_type": "discussion_task_command",
                "client_payload": {
                    "discussion_number": 66,
                    "triggering_comment_node_id": "DC_trigger",
                },
            },
        )
    assert len(reads) == 1
    assert clock.waits == []
    assert caught.value.first_failure_at == 1000


def test_recovery_workflow_is_dispatch_only_and_names_the_episode():
    path = (
        Path(__file__).parent.parent
        / ".github/workflows/discussion-recovery-notice.yml"
    )
    assert path.exists(), (
        "The external controller needs a public App-authored recovery notice workflow"
    )
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert set(workflow["on"]["workflow_dispatch"]["inputs"]) == {
        "source_repository",
        "source_run_id",
        "discussion_number",
        "episode_id",
        "status",
    }
    assert (
        workflow["run-name"]
        == "rsi-recovery:${{ inputs.episode_id }}:${{ inputs.status }}"
    )
    job = workflow["jobs"]["notice"]
    assert job["runs-on"] == "ubuntu-latest"
    assert workflow["permissions"] == {"contents": "read"}
    token = next(step for step in job["steps"] if step.get("id") == "comment-token")
    assert token["with"]["repositories"] == "RSIs-First-Exam"
    assert token["with"]["permission-discussions"] == "write"


@pytest.mark.parametrize(
    "filename,source_job",
    [("discussion-review.yml", "review"), ("discussion-task-dispatch.yml", "dispatch")],
)
def test_eyes_outage_fails_only_a_notification_job_and_can_be_recovered(
    filename, source_job
):
    path = Path(__file__).parent.parent / ".github/workflows" / filename
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    assert all(
        "addReaction" not in step.get("run", "")
        for step in workflow["jobs"][source_job]["steps"]
    )
    reactions = [
        (job, step)
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "addReaction" in step.get("run", "")
    ]
    assert len(reactions) == 1
    notification, reaction = reactions[0]
    assert notification["runs-on"] == "ubuntu-latest"
    assert "continue-on-error" not in reaction
    assert all(
        "continue-on-error" not in step
        for step in notification["steps"]
        if step.get("uses", "").startswith("actions/create-github-app-token@")
    )
    if source_job == "dispatch":
        assert (
            notification["if"] == "needs.dispatch.outputs.command_dispatched == 'true'"
        )
        queue = next(
            step
            for step in notification["steps"]
            if step.get("name") == "Post task queue notice"
        )
        assert queue["if"] == "needs.dispatch.outputs.task_dispatched == 'true'"


def test_reused_review_step_does_not_invoke_model_or_require_ephemeral_result(tmp_path):
    path = Path(__file__).parent.parent / ".github/workflows/discussion-review.yml"
    workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    step = next(
        step
        for step in workflow["jobs"]["review"]["steps"]
        if step.get("id") == "review"
    )
    fake_uv = tmp_path / "uv"
    fake_uv.write_text("#!/bin/sh\necho model-was-called >&2\nexit 90\n")
    fake_uv.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": str(tmp_path) + ":" + os.environ["PATH"],
            "GITHUB_OUTPUT": str(output),
            "REUSED_REVIEW_ID": "DC_review",
            "REUSED_DECISION": "Pass",
        },
    )
    assert result.returncode == 0, result.stderr
    assert output.read_text() == "decision=Pass\n"
