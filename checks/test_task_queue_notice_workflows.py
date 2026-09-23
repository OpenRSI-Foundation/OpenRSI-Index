from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).parent.parent / ".github/workflows"
ENTRIES = (
    ("discussion-task-dispatch.yml", "dispatch", "Dispatch privately", "ack-token"),
    (
        "discussion-review.yml",
        "review",
        "Dispatch passed proposal privately",
        "queue-token",
    ),
)


def notice_steps(filename: str, job: str) -> tuple[list[dict], dict]:
    workflow = yaml.load((WORKFLOWS / filename).read_text(), Loader=yaml.BaseLoader)
    assert "queue-notice" not in workflow["jobs"]
    assert workflow["jobs"][job]["runs-on"] == "ubuntu-latest"
    steps = workflow["jobs"][job]["steps"]
    notices = [step for step in steps if step.get("name") == "Post task queue notice"]
    assert len(notices) == 1, (
        "Queued tasks need a notice before a W2 runner is available"
    )
    return steps, notices[0]


@pytest.mark.parametrize("filename,job,dispatch_name,token", ENTRIES)
def test_queue_notice_precedes_dispatch_and_requires_an_authorized_task(
    filename, job, dispatch_name, token
):
    steps, notice = notice_steps(filename, job)
    workflow = yaml.load((WORKFLOWS / filename).read_text(), Loader=yaml.BaseLoader)
    source = workflow["jobs"][job]
    dispatch = next(
        step for step in source["steps"] if step.get("name") == dispatch_name
    )
    assert dispatch["id"] == "task-dispatch"
    queue_token = next(step for step in steps if step.get("id") == token)
    assert steps.index(queue_token) < steps.index(notice) < steps.index(dispatch)
    if job == "dispatch":
        authorization = (
            "steps.gate.outputs.candidate == 'true' && "
            "(steps.gate.outputs.is_author == 'true' || "
            "steps.owner-gate.outputs.is_owner == 'true')"
        )
        assert queue_token["if"] == authorization
        assert notice["if"] == authorization + " && steps.gate.outputs.command == 'task'"
        assert dispatch["if"] == authorization
        trigger_comment_id = "${{ github.event.comment.node_id }}"
    else:
        authorization = (
            "steps.review.outputs.decision == 'Pass' && "
            "steps.publish-review.outcome == 'success' && "
            "steps.publish-review.outputs.review_comment_id != ''"
        )
        assert queue_token["if"] == notice["if"] == authorization
        dispatch_token = next(
            step for step in steps if step.get("id") == "dispatch-token"
        )
        assert steps.index(queue_token) + 1 == steps.index(notice)
        assert steps.index(notice) + 1 == steps.index(dispatch_token)
        assert dispatch["if"] == (
            authorization + " && steps.dispatch-token.outcome == 'success'"
        )
        trigger_comment_id = "${{ steps.publish-review.outputs.review_comment_id }}"
    expected_gate = "steps.task-dispatch.outcome == 'success'"
    if job == "dispatch":
        expected_gate += " && steps.gate.outputs.command == 'task'"
    assert source["outputs"]["task_dispatched"] == "${{ " + expected_gate + " }}"
    # A failed queue notice must prevent dispatch so the worker cannot post
    # "building" before the queue acknowledgement exists.
    assert "continue-on-error" not in queue_token
    assert "continue-on-error" not in notice
    assert queue_token["with"]["repositories"] == "${{ github.event.repository.name }}"
    assert queue_token["with"]["permission-discussions"] == "write"
    assert notice["env"] == {
        "GH_TOKEN": "${{ steps." + token + ".outputs.token }}",
        "BOT_LOGIN": "${{ steps." + token + ".outputs.app-slug }}",
        "TRIGGER_COMMENT_ID": trigger_comment_id,
        "DISCUSSION_ID": "${{ github.event.discussion.node_id }}",
    }


@pytest.mark.parametrize("filename,job,dispatch_name,token", ENTRIES)
def test_queue_notice_posts_one_new_comment_without_modifying_history(
    filename, job, dispatch_name, token, tmp_path
):
    _, notice = notice_steps(filename, job)
    captured = tmp_path / "requests.jsonl"
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'with open(os.environ["CAPTURED_REQUESTS"], "a") as stream:\n'
        '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        "args = sys.argv[1:]\n"
        'fields = dict(args[i + 1].split("=", 1) for i, arg in enumerate(args) if arg == "-f")\n'
        'if "addDiscussionComment" in fields["query"]:\n'
        '    print(json.dumps({"data": {"addDiscussionComment": {"comment": {"id": "DC_notice", "url": "https://github.com/comment"}}}}))\n'
        "else:\n"
        '    print(json.dumps({"data": {"node": {"id": "D_queue_notice", "comments": {"nodes": [], "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}))\n'
    )
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", notice["run"]],
        cwd=WORKFLOWS.parent.parent,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GH_TOKEN": "test-token",
            "BOT_LOGIN": "dispatcher[bot]",
            "GITHUB_RUN_ID": "123",
            "TRIGGER_COMMENT_ID": "DC_trigger",
            "DISCUSSION_ID": "D_queue_notice",
            "CAPTURED_REQUESTS": str(captured),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    requests = [json.loads(line) for line in captured.read_text().splitlines()]
    assert len(requests) == 2
    args = requests[-1]
    assert args[:2] == ["api", "graphql"]
    fields = dict(
        args[index + 1].split("=", 1) for index, arg in enumerate(args) if arg == "-f"
    )
    assert fields["discussionId"] == "D_queue_notice"
    assert "addDiscussionComment" in fields["query"]
    assert "updateDiscussionComment" not in fields["query"]
    assert "deleteDiscussionComment" not in fields["query"]
    assert "Your task is queued" in fields["body"]
    assert "will start automatically when a runner is available" in fields["body"]
    assert "No action is needed." in fields["body"]
