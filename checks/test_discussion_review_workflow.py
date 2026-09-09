from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parent.parent / ".github/workflows/discussion-review.yml"
ROOT = WORKFLOW.parent.parent.parent
CHECKOUT_SHA = "fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09"
APP_TOKEN_SHA = "bcd2ba49218906704ab6c1aa796996da409d3eb1"
SETUP_UV_SHA = "38f3f104447c67c051c4a08e39b64a148898af3a"
AUTOMATIC_DISPATCH_GUARD = (
    "steps.review.outputs.decision == 'Pass' && "
    "steps.publish-review.outcome == 'success' && "
    "steps.publish-review.outputs.review_comment_id != ''"
)


def parsed_steps() -> list[dict]:
    workflow = yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return workflow["jobs"]["review"]["steps"]


def parsed_workflow() -> dict:
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def step_named(name: str) -> dict:
    return next(step for step in parsed_steps() if step.get("name") == name)


def test_review_workflow_serializes_each_discussion_and_restarts_on_edit():
    concurrency = parsed_workflow()["concurrency"]

    assert concurrency == {
        "group": "discussion-review-${{ github.event.discussion.node_id }}",
        "cancel-in-progress": "true",
    }


def test_review_workflow_supersedes_only_unfinished_reviews_then_posts_fresh_progress():
    steps = parsed_steps()
    progress = step_named("Post or update running review comment")

    assert progress["id"] == "progress"
    assert progress["if"] == "steps.reaction-token.outcome == 'success'"
    assert progress["continue-on-error"] == "true"
    assert progress["env"] == {
        "GH_TOKEN": "${{ steps.reaction-token.outputs.token }}",
        "BOT_LOGIN": "${{ steps.reaction-token.outputs.app-slug }}",
        "DISCUSSION_ID": "${{ github.event.discussion.node_id }}",
        "DISCUSSION_NUMBER": "${{ github.event.discussion.number }}",
        "REPOSITORY_OWNER": "${{ github.repository_owner }}",
        "REPOSITORY_NAME": "${{ github.event.repository.name }}",
    }
    assert steps.index(progress) == steps.index(step_named("React with eyes")) + 1
    assert steps.index(progress) < steps.index(step_named("Create private Skills read token"))
    assert "<!-- rubric-review-bot -->" in progress["run"]
    assert "<!-- rubric-review-status:running -->" in progress["run"]
    assert "<!-- rubric-review-status:superseded -->" in progress["run"]
    assert "Proposal review is running" in progress["run"]
    assert "usually takes about 1 minute" in progress["run"]
    assert "This comment will update when the review finishes" not in progress["run"]
    assert "updateDiscussionComment" in progress["run"]
    assert "addDiscussionComment" in progress["run"]
    assert "--paginate --slurp" in progress["run"]
    assert 'contains("<!-- rubric-review-status:running -->")' in progress["run"]
    assert "comment_id=" in progress["run"]
    assert '>> "$GITHUB_OUTPUT"' in progress["run"]


def test_review_workflow_replaces_only_current_progress_with_generic_failure():
    steps = parsed_steps()
    failure_token = step_named("Create failure comment Discussion App token")
    failure = step_named("Post or update failed review comment")

    assert failure_token == {
        "name": "Create failure comment Discussion App token",
        "id": "failure-token",
        "if": (
            "always() && !cancelled() && "
            "steps.publish-review.outcome != 'success'"
        ),
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
    assert failure["if"] == (
        "always() && !cancelled() && "
        "steps.publish-review.outcome != 'success' && "
        "steps.failure-token.outcome == 'success'"
    )
    assert failure["continue-on-error"] == "true"
    assert failure["env"]["GH_TOKEN"] == "${{ steps.failure-token.outputs.token }}"
    assert failure["env"]["PROGRESS_COMMENT_ID"] == (
        "${{ steps.progress.outputs.comment_id }}"
    )
    assert steps.index(failure_token) == (
        steps.index(step_named("Dispatch passed proposal privately")) + 1
    )
    assert steps.index(failure) == steps.index(failure_token) + 1
    assert "Proposal review failed before completion" in failure["run"]
    assert "updateDiscussionComment" in failure["run"]
    assert "addDiscussionComment" in failure["run"]
    assert "EXISTING_COMMENT_ID" not in failure["run"]
    assert "comments(first:" not in failure["run"]
    for private_text in ("RSI-Skills", "SKILL.md", "RUBRIC_FILE", "review.log"):
        assert private_text not in failure["run"]


def test_successful_review_updates_only_the_current_runs_progress_comment():
    publish = step_named("Format and post or update comment")

    assert publish["id"] == "publish-review"
    assert publish["env"]["PROGRESS_COMMENT_ID"] == (
        "${{ steps.progress.outputs.comment_id }}"
    )
    assert "EVENT_ACTION" not in publish["env"]
    assert 'if [ "$EVENT_ACTION" = "edited" ]' not in publish["run"]
    assert "updateDiscussionComment" in publish["run"]
    assert "addDiscussionComment" in publish["run"]
    assert "EXISTING_COMMENT_ID" not in publish["run"]
    assert "comments(first:" not in publish["run"]


def test_successful_review_outputs_authoritative_comment_id_from_update_or_create():
    publish = step_named("Format and post or update comment")
    script = publish["run"]

    assert "COMMENT_RESULT=$(gh api graphql" in script
    assert "jq -er '.data.updateDiscussionComment.comment.id'" in script
    assert "jq -er '.data.addDiscussionComment.comment.id'" in script
    assert "printf 'review_comment_id=%s\\n' \"$REVIEW_COMMENT_ID\"" in script
    superseded = script.split("Current review comment is no longer active", 1)[1].split(
        "COMMENT_RESULT=$(gh api graphql", 1
    )[0]
    assert "exit 0" in superseded
    assert "review_comment_id" not in superseded


def test_stale_progress_lookup_paginates_and_excludes_completed_reviews():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    lookup = workflow.split("RUNNING_COMMENT_IDS=$(gh api graphql", 1)[1].split(
        "while IFS= read -r comment_id", 1
    )[0]

    assert "$endCursor: String" in lookup
    assert "comments(first: 100, after: $endCursor)" in lookup
    assert "pageInfo { hasNextPage endCursor }" in lookup
    assert "--paginate --slurp" in lookup
    assert "[.[].data.repository.discussion.comments.nodes[]" in lookup
    assert "author { login }" in lookup
    assert 'select(.author.login == $bot)' in lookup
    assert '--arg bot "$BOT_LOGIN"' in lookup
    assert 'select(.body | contains("<!-- rubric-review-bot -->"))' in lookup
    assert 'select(.body | contains("<!-- rubric-review-status:running -->"))' in lookup


def test_review_workflow_pins_actions_and_drops_checkout_credentials():
    steps = parsed_steps()
    checkouts = [
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    ]
    setup_uv = next(step for step in steps if step.get("name") == "Install uv")

    assert len(checkouts) == 2
    assert [checkout["uses"] for checkout in checkouts] == [
        f"actions/checkout@{CHECKOUT_SHA}",
        f"actions/checkout@{CHECKOUT_SHA}",
    ]
    assert [
        checkout.get("with", {}).get("persist-credentials") for checkout in checkouts
    ] == ["false", "false"]
    assert setup_uv["uses"] == f"astral-sh/setup-uv@{SETUP_UV_SHA}"


def test_review_workflow_reads_the_private_skills_rubric_with_an_app_token():
    steps = {step.get("name"): step for step in parsed_steps()}

    assert steps["Create private Skills read token"] == {
        "name": "Create private Skills read token",
        "id": "skills-token",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "RSI-Skills",
            "permission-contents": "read",
        },
    }
    assert steps["Checkout private discussion rubric"] == {
        "name": "Checkout private discussion rubric",
        "uses": f"actions/checkout@{CHECKOUT_SHA}",
        "with": {
            "repository": "RSI-Index/RSI-Skills",
            "ref": "main",
            "token": "${{ steps.skills-token.outputs.token }}",
            "path": "private-skills",
            "sparse-checkout": "rubrics/task-proposal.md",
            "sparse-checkout-cone-mode": "false",
            "persist-credentials": "false",
        },
    }
    assert steps["Run rubric review"]["env"]["RUBRIC_FILE"] == (
        "${{ github.workspace }}/private-skills/rubrics/task-proposal.md"
    )


def test_review_reactions_and_comments_use_just_in_time_scoped_app_tokens():
    ordered = parsed_steps()
    steps = {step.get("name"): step for step in ordered}
    expected_with = {
        "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
        "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
        "owner": "RSI-Index",
        "repositories": "${{ github.event.repository.name }}",
        "permission-discussions": "write",
    }

    assert steps["Create reaction Discussion App token"] == {
        "name": "Create reaction Discussion App token",
        "id": "reaction-token",
        "continue-on-error": "true",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": expected_with,
    }
    assert steps["Create comment Discussion App token"] == {
        "name": "Create comment Discussion App token",
        "id": "comment-token",
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": expected_with,
    }
    assert steps["React with eyes"]["env"]["GH_TOKEN"] == (
        "${{ steps.reaction-token.outputs.token }}"
    )
    assert steps["React with eyes"]["if"] == "steps.reaction-token.outcome == 'success'"
    assert steps["React with eyes"]["continue-on-error"] == "true"
    assert steps["Format and post or update comment"]["env"]["GH_TOKEN"] == (
        "${{ steps.comment-token.outputs.token }}"
    )
    assert steps["Post or update running review comment"]["env"]["BOT_LOGIN"] == (
        "${{ steps.reaction-token.outputs.app-slug }}"
    )
    assert ordered.index(steps["Create reaction Discussion App token"]) + 1 == (
        ordered.index(steps["React with eyes"])
    )
    assert ordered.index(steps["Create comment Discussion App token"]) + 1 == (
        ordered.index(steps["Format and post or update comment"])
    )
    assert "DISCUSSION_TOKEN" not in WORKFLOW.read_text(encoding="utf-8")
    bot_logins = [
        step["env"]["BOT_LOGIN"]
        for step in ordered
        if "BOT_LOGIN" in step.get("env", {})
    ]
    assert len(bot_logins) == 1
    assert all("[bot]" not in login for login in bot_logins)


def test_public_repository_does_not_ship_or_migrate_the_private_rubric():
    workflows = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / ".github/workflows").glob("*.yml"))
    )

    assert not (ROOT / "rubrics/task-proposal.md").exists()
    assert not (ROOT / ".github/workflows/migrate-private-rubric.yml").exists()
    assert "Zhuofeng-Li/RSI-Index-Rubrics" not in workflows
    assert "RUBRIC_REPO_TOKEN" not in workflows


def test_review_workflow_passes_github_expressions_through_step_environment():
    for step in parsed_steps():
        assert "${{" not in step.get("run", "")


def test_review_workflow_binds_zizmor_flagged_values_as_step_environment():
    steps = {step.get("name"): step for step in parsed_steps()}
    react = steps["React with eyes"]
    progress = steps["Post or update running review comment"]
    format_comment = steps["Format and post or update comment"]

    assert react["env"]["DISCUSSION_NODE_ID"] == "${{ github.event.discussion.node_id }}"
    assert 'id="$DISCUSSION_NODE_ID"' in react["run"]
    assert format_comment["env"]["DECISION"] == "${{ steps.review.outputs.decision }}"
    assert progress["env"]["REPOSITORY_OWNER"] == "${{ github.repository_owner }}"
    assert progress["env"]["REPOSITORY_NAME"] == "${{ github.event.repository.name }}"
    assert 'owner="$REPOSITORY_OWNER"' in progress["run"]
    assert 'repo="$REPOSITORY_NAME"' in progress["run"]


def test_review_workflow_renders_and_appends_the_canonical_marker():
    steps = {step.get("name"): step for step in parsed_steps()}
    render_marker = steps["Render canonical proposal review marker"]
    format_comment = steps["Format and post or update comment"]

    assert render_marker["env"] == {
        "DECISION": "${{ steps.review.outputs.decision }}",
        "DISCUSSION_TITLE": "${{ github.event.discussion.title }}",
        "DISCUSSION_BODY": "${{ github.event.discussion.body }}",
        "DISCUSSION_NODE_ID": "${{ github.event.discussion.node_id }}",
    }
    assert render_marker["run"].strip() == (
        "marker=$(python3 checks/proposal_review_marker.py)\n"
        "printf 'marker=%s\\n' \"$marker\" >> \"$GITHUB_OUTPUT\""
    )
    assert format_comment["env"]["REVIEW_MARKER"] == "${{ steps.marker.outputs.marker }}"

    comment_template = format_comment["run"].split("BODY=$(cat << COMMENT_EOF", 1)[1].split(
        "COMMENT_EOF", 1
    )[0]
    assert "${REVIEW_MARKER}" in comment_template
    assert format_comment["run"].count('-f body="$BODY"') == 2


def test_review_workflow_has_no_synthetic_decision_fallback():
    steps = {step.get("name"): step for step in parsed_steps()}
    run_review = steps["Run rubric review"]["run"]

    assert 'decision = result.get("decision")' in run_review
    assert 'review = result.get("review")' in run_review
    assert "publication_guard" not in run_review
    assert "withheld" not in run_review
    assert "require human review" not in run_review
    assert "Strong Reject" not in run_review


def test_review_workflow_accepts_and_publishes_only_pass_or_reject():
    run_review = step_named("Run rubric review")["run"]
    publish = step_named("Format and post or update comment")["run"]

    assert 'allowed_decisions = {"Reject", "Pass"}' in run_review
    assert '"Pass")   BADGE="🟢 **PASS**"' in publish
    assert '"Reject") BADGE="🔴 **REJECT**"' in publish
    assert "Strong Accept" not in publish
    assert "**Accept**" not in publish


def test_pass_publication_starts_building_without_an_initial_task_instruction():
    publish = step_named("Format and post or update comment")["run"]
    pass_copy = publish.split('"Pass")', 2)[2].split(';;', 1)[0]

    assert "PASS — Initial check passed. Task building starts automatically." in pass_copy
    assert "Minor revisions may still be needed in some cases." in pass_copy
    assert "/task" not in pass_copy


def test_pass_dispatch_uses_separate_conditional_private_write_token():
    steps = parsed_steps()
    token = step_named("Create private Skills dispatch token")

    assert token == {
        "name": "Create private Skills dispatch token",
        "id": "dispatch-token",
        "if": AUTOMATIC_DISPATCH_GUARD,
        "uses": f"actions/create-github-app-token@{APP_TOKEN_SHA}",
        "with": {
            "client-id": "${{ vars.RSI_DISPATCH_APP_CLIENT_ID }}",
            "private-key": "${{ secrets.RSI_DISPATCH_APP_PRIVATE_KEY }}",
            "owner": "RSI-Index",
            "repositories": "RSI-Skills",
            "permission-contents": "write",
        },
    }
    assert steps.index(step_named("Format and post or update comment")) < steps.index(token)
    assert steps.index(token) < steps.index(step_named("Dispatch passed proposal privately"))
    assert step_named("Create private Skills read token")["with"]["permission-contents"] == "read"
    assert step_named("Checkout private discussion rubric")["with"]["token"] == (
        "${{ steps.skills-token.outputs.token }}"
    )


def test_pass_dispatch_builds_exact_request_and_posts_once_after_publication():
    raw = WORKFLOW.read_text(encoding="utf-8")
    dispatch = step_named("Dispatch passed proposal privately")

    assert "Build automatic Pass dispatch request" not in {
        step.get("name") for step in parsed_steps()
    }
    assert dispatch["if"] == (
        AUTOMATIC_DISPATCH_GUARD + " && steps.dispatch-token.outcome == 'success'"
    )
    assert dispatch["env"] == {
        "GH_TOKEN": "${{ steps.dispatch-token.outputs.token }}",
        "REVIEW_COMMENT_NODE_ID": "${{ steps.publish-review.outputs.review_comment_id }}",
    }
    script = dispatch["run"]
    assert (
        'python3 checks/proposal_pass_dispatch.py "$GITHUB_EVENT_PATH" '
        '"$REVIEW_COMMENT_NODE_ID" proposal-pass-payload.json'
    ) in script
    assert script.index("checks/proposal_pass_dispatch.py") < script.index(
        '"event_type": "discussion_task_command"'
    )
    assert script.index('"event_type": "discussion_task_command"') < script.index(
        "gh api --method POST /repos/RSI-Index/RSI-Skills/dispatches"
    )
    assert '"client_payload": payload' in script
    assert "gh api --method POST /repos/RSI-Index/RSI-Skills/dispatches" in script
    assert "--input proposal-pass-request.json" in script
    assert "proposal-pass-live.json" not in script
    assert "viewerDidAuthor" not in script
    assert raw.count("/repos/RSI-Index/RSI-Skills/dispatches") == 1


def test_pass_dispatch_is_guarded_from_reject_failure_supersession_and_missing_id():
    assert step_named("Create private Skills dispatch token")["if"] == (
        AUTOMATIC_DISPATCH_GUARD
    )

    dispatch_condition = step_named("Dispatch passed proposal privately")["if"]
    assert "decision == 'Pass'" in dispatch_condition
    assert "publish-review.outcome == 'success'" in dispatch_condition
    assert "review_comment_id != ''" in dispatch_condition
    assert "dispatch-token.outcome == 'success'" in dispatch_condition


def test_dispatch_eligibility_relies_on_the_successful_current_run_publication():
    steps = parsed_steps()
    publish = step_named("Format and post or update comment")
    dispatch = step_named("Dispatch passed proposal privately")
    script = dispatch["run"]

    assert steps.index(publish) < steps.index(dispatch)
    assert "set -euo pipefail" in script
    assert 'if [ "$valid" != "true" ]' in script
    assert "exit 1" in script.split('if [ "$valid" != "true" ]', 1)[1].split(
        "fi", 1
    )[0]
    assert "proposal_pass_dispatch.py" in script
    assert "proposal-pass-live.json" not in script
    assert "viewerDidAuthor" not in script
