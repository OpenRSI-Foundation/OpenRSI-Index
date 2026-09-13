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
        "comment-token",
    ),
)


def notice_steps(filename: str, job: str) -> tuple[list[dict], dict]:
    workflow = yaml.load((WORKFLOWS / filename).read_text(), Loader=yaml.BaseLoader)
    assert workflow["jobs"][job]["runs-on"] == "ubuntu-latest"
    steps = workflow["jobs"][job]["steps"]
    notices = [step for step in steps if step.get("name") == "Post task queue notice"]
    assert len(notices) == 1, (
        "Queued tasks need a notice before a W2 runner is available"
    )
    return steps, notices[0]


@pytest.mark.parametrize("filename,job,dispatch_name,token", ENTRIES)
def test_queue_notice_is_gated_on_successful_task_dispatch(
    filename, job, dispatch_name, token
):
    steps, notice = notice_steps(filename, job)
    dispatch = next(step for step in steps if step.get("name") == dispatch_name)
    assert dispatch["id"] == "task-dispatch"
    assert steps.index(notice) == steps.index(dispatch) + 1
    expected_gate = f"steps.task-dispatch.outcome == 'success' && steps.{token}.outcome == 'success'"
    if job == "dispatch":
        expected_gate += " && steps.gate.outputs.command == 'task'"
    assert notice["if"] == expected_gate
    assert notice["continue-on-error"] == "true"
    assert notice["env"] == {
        "GH_TOKEN": "${{ steps." + token + ".outputs.token }}",
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
    )
    gh.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-c", notice["run"]],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GH_TOKEN": "test-token",
            "DISCUSSION_ID": "D_queue_notice",
            "CAPTURED_REQUESTS": str(captured),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    requests = [json.loads(line) for line in captured.read_text().splitlines()]
    assert len(requests) == 1
    args = requests[0]
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
