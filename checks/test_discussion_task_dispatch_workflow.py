from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parent.parent / ".github/workflows/discussion-task-dispatch.yml"
CONTRIBUTING = Path(__file__).parent.parent / "CONTRIBUTING.md"
CHECKOUT_SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
APP_TOKEN_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"


def load_workflow() -> tuple[dict, str]:
    raw = WORKFLOW.read_text(encoding="utf-8")
    return yaml.load(raw, Loader=yaml.BaseLoader), raw


def steps_by_name(workflow: dict) -> dict[str, dict]:
    steps = workflow["jobs"]["dispatch"]["steps"]
    return {step["name"]: step for step in steps}


def test_dispatch_workflow_accepts_only_created_discussion_comments():
    workflow, _ = load_workflow()

    assert workflow["on"] == {"discussion_comment": {"types": ["created"]}}


def test_dispatch_workflow_is_public_only_and_minimally_privileged():
    workflow, _ = load_workflow()
    steps = workflow["jobs"]["dispatch"]["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))

    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["jobs"]["dispatch"]["runs-on"] == "ubuntu-latest"
    assert checkout == {
        "name": "Checkout public default branch",
        "uses": f"actions/checkout@{CHECKOUT_SHA}",
        "with": {
            "ref": "${{ github.event.repository.default_branch }}",
            "persist-credentials": "false",
        },
    }
    assert len([step for step in steps if step.get("uses", "").startswith("actions/checkout@")]) == 1


def test_dispatch_workflow_gates_token_and_dispatch_on_author_or_current_owner():
    workflow, _ = load_workflow()
    steps = workflow["jobs"]["dispatch"]["steps"]
    named_steps = steps_by_name(workflow)
    gate_index = next(index for index, step in enumerate(steps) if step["name"] == "Validate event")
    token_index = next(index for index, step in enumerate(steps) if step["name"] == "Create dispatch App token")

    assert "checks/discussion_task_dispatch.py" in named_steps["Validate event"]["run"]
    assert gate_index < token_index
    assert named_steps["Create dispatch App token"] == {
        "name": "Create dispatch App token",
        "id": "app-token",
        "if": "steps.gate.outputs.candidate == 'true'",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "RSI-Skills",
            "permission-contents": "write",
            "permission-members": "read",
        },
    }
    owner_gate = named_steps["Verify current organization Owner"]
    assert owner_gate["id"] == "owner-gate"
    assert owner_gate["if"] == (
        "steps.gate.outputs.candidate == 'true' && "
        "steps.gate.outputs.is_author != 'true'"
    )
    assert owner_gate["env"] == {
        "GH_TOKEN": "${{ steps.app-token.outputs.token }}",
        "COMMENTER_LOGIN": "${{ steps.gate.outputs.commenter_login }}",
    }
    assert "/orgs/RSI-Index/memberships/$COMMENTER_LOGIN" in owner_gate["run"]
    assert "checks/org_owner_gate.py" in owner_gate["run"]

    authorization = (
        "steps.gate.outputs.candidate == 'true' && "
        "(steps.gate.outputs.is_author == 'true' || "
        "steps.owner-gate.outputs.is_owner == 'true')"
    )
    for name in ("Build dispatch request", "Dispatch privately"):
        assert named_steps[name]["if"] == authorization

    assert gate_index < token_index < steps.index(owner_gate)


def test_dispatch_workflow_posts_a_parser_built_identifier_only_request():
    workflow, _ = load_workflow()
    named_steps = steps_by_name(workflow)
    build_request = named_steps["Build dispatch request"]["run"]
    dispatch = named_steps["Dispatch privately"]

    assert "json.load" in build_request
    assert "dispatch-payload.json" in build_request
    assert "json.dump" in build_request
    assert "repository-dispatch.json" in build_request
    assert named_steps["Build dispatch request"]["env"] == {
        "COMMAND": "${{ steps.gate.outputs.command }}"
    }
    assert "discussion_task_command" in build_request
    assert "discussion_task_reset" in build_request
    assert dispatch["env"] == {"GH_TOKEN": "${{ steps.app-token.outputs.token }}"}
    assert "gh api --method POST /repos/RSI-Index/RSI-Skills/dispatches" in dispatch["run"]
    assert "--input repository-dispatch.json" in dispatch["run"]


def test_manual_dispatch_payload_is_explicitly_a_comment_trigger():
    workflow, _ = load_workflow()
    named_steps = steps_by_name(workflow)
    authorization = (
        "steps.gate.outputs.candidate == 'true' && "
        "(steps.gate.outputs.is_author == 'true' || "
        "steps.owner-gate.outputs.is_owner == 'true')"
    )

    assert named_steps["Build dispatch request"]["if"] == authorization
    assert named_steps["Dispatch privately"]["if"] == authorization
    assert "payload = json.load" in named_steps["Build dispatch request"]["run"]
    assert 'payload.get("trigger_kind") != "comment"' in named_steps[
        "Build dispatch request"
    ]["run"]
    assert '"client_payload": payload' in named_steps["Build dispatch request"]["run"]


def test_authorized_task_command_gets_non_blocking_eyes_acknowledgement():
    workflow, _ = load_workflow()
    steps = workflow["jobs"]["dispatch"]["steps"]
    named_steps = steps_by_name(workflow)
    authorization = (
        "steps.gate.outputs.candidate == 'true' && "
        "(steps.gate.outputs.is_author == 'true' || "
        "steps.owner-gate.outputs.is_owner == 'true')"
    )
    token = named_steps["Create acknowledgement App token"]
    reaction = named_steps["Acknowledge accepted task command"]

    assert token == {
        "name": "Create acknowledgement App token",
        "id": "ack-token",
        "if": authorization,
        "continue-on-error": "true",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "${{ github.event.repository.name }}",
            "permission-discussions": "write",
        },
    }
    assert reaction["if"] == authorization + " && steps.ack-token.outcome == 'success'"
    assert reaction["continue-on-error"] == "true"
    assert reaction["env"] == {
        "GH_TOKEN": "${{ steps.ack-token.outputs.token }}",
        "COMMENT_NODE_ID": "${{ github.event.comment.node_id }}",
    }
    assert "addReaction" in reaction["run"]
    assert 'id="$COMMENT_NODE_ID"' in reaction["run"]
    assert 'content="EYES"' in reaction["run"]
    assert steps.index(reaction) < steps.index(named_steps["Dispatch privately"])


def test_dispatch_workflow_never_handles_private_or_untrusted_content():
    _, raw = load_workflow()
    lower = raw.lower()

    for forbidden in (
        "openai/codex-action",
        "openai_api_key",
        "rsi-task-state",
        "upload-artifact",
        "github.event.comment.body",
        "github.event.discussion.body",
        "generated task",
    ):
        assert forbidden not in lower
    assert "repository:" not in lower


def test_plain_task_is_documented_only_for_fresh_assumptions_after_reset():
    _, raw = load_workflow()
    contributing = " ".join(
        CONTRIBUTING.read_text(encoding="utf-8").split()
    )

    assert "send a plain `/task` to start a fresh assumptions pass" in contributing
    assert "`/task <answer or correction>`" in contributing
    assert "`/task confirm`" not in contributing
    assert "COMMAND: ${{ steps.gate.outputs.command }}" in raw
    assert "discussion_task_command" in raw
    assert "task_confirm" not in raw.lower()


def test_contributing_documents_state_only_reset_and_history_preservation():
    contributing = " ".join(
        CONTRIBUTING.read_text(encoding="utf-8").lower().split()
    )

    assert "to discard an unpublished attempt, send `/reset`" in contributing
    assert "reset removes only that attempt's private state" in contributing
    assert "every discussion comment, including `/reset`, remains" in contributing
    assert "wait for the bot's reset-complete reply" in contributing
    assert "send a plain `/task` to start a fresh assumptions pass" in contributing
