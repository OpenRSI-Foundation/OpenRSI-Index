"""Short GitHub retries and reconciled public Discussion mutations.

No long sleeps or durable recovery loop lives on a hosted runner. Typed failures
are emitted for the external controller, which owns the fixed recovery deadline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

if __package__:
    from .github_retry import (
        REQUEST_TIMEOUT_SECONDS,
        GitHubTransientError,
        emit_retry_diagnostic,
        retry_call,
        transient_http_error,
    )
    from .org_owner_gate import is_active_org_owner
    from .proposal_review_marker import canonical_proposal_hash, parse_review_marker
else:
    from github_retry import (
        REQUEST_TIMEOUT_SECONDS,
        GitHubTransientError,
        emit_retry_diagnostic,
        retry_call,
        transient_http_error,
    )
    from org_owner_gate import is_active_org_owner
    from proposal_review_marker import canonical_proposal_hash, parse_review_marker


QUEUE_BODY = "⏳ Your task is queued and will start automatically when a runner is available. No action is needed."
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,200}\Z")
_OPERATION = re.compile(r"[a-z][a-z0-9_.-]{1,80}\Z")
_COMMENTS_QUERY = """query($discussionId: ID!, $cursor: String) {
  node(id: $discussionId) { ... on Discussion { id comments(first: 100, after: $cursor) {
    nodes { id url body author { login } createdAt updatedAt }
    pageInfo { hasNextPage endCursor }
  } } }
}"""
_ADD_COMMENT = """mutation($discussionId: ID!, $body: String!) {
  addDiscussionComment(input: {discussionId: $discussionId, body: $body}) {
    comment { id url }
  }
}"""


class GitHubRequestError(RuntimeError):
    def __init__(self, status=None):
        self.status = status
        super().__init__(f"GitHub request failed (status={status or 'unknown'})")


def _gh_once(args, *, runner=None, clock=None):
    """Capture headers for classification; never include arguments/body in errors."""
    runner = runner or subprocess.run
    try:
        result = runner(
            ["gh", *args, "--include"],
            capture_output=True,
            text=True,
            timeout=REQUEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise GitHubTransientError(operation="discussion.github") from None
    # A missing gh executable is a setup/configuration error, not a network outage.
    raw = result.stdout.replace("\r\n", "\n")
    status, headers = None, {}
    if raw.startswith("HTTP/"):
        header, separator, raw = raw.partition("\n\n")
        match = re.match(r"HTTP/\S+\s+(\d{3})", header)
        if match:
            status = int(match.group(1))
        for line in header.splitlines()[1:]:
            key, separator, value = line.partition(":")
            if separator:
                headers[key.strip()] = value.strip()
    if status is None:
        match = re.search(r"\(HTTP (\d{3})\)", result.stderr)
        if match:
            status = int(match.group(1))
    error = transient_http_error(
        status or 200, headers, raw, operation="discussion.github", clock=clock
    )
    if error is not None:
        raise error
    if result.returncode:
        transport_terms = (
            "dial tcp",
            "no such host",
            "connection reset",
            "connection refused",
            "i/o timeout",
            "context deadline exceeded",
            "tls handshake timeout",
            "unexpected eof",
            "temporary failure in name resolution",
            "network is unreachable",
            "http2: stream closed",
            "stream error",
        )
        if any(term in result.stderr.casefold() for term in transport_terms):
            raise GitHubTransientError(operation="discussion.github")
        raise GitHubRequestError(status)
    return raw


def gh_request(args, *, runner=None, sleep=None, clock=None):
    return retry_call(
        lambda: _gh_once(args, runner=runner, clock=clock), sleep=sleep, clock=clock
    )


class GitHubClient:
    def __init__(self, *, call=None, sleep=None, clock=None):
        self.call = call or _gh_once
        self.sleep, self.clock = sleep, clock

    def read(self, args):
        return retry_call(lambda: self.call(args), sleep=self.sleep, clock=self.clock)

    def graphql(self, query, **fields):
        args = ["api", "graphql", "-f", "query=" + query]
        for name, value in fields.items():
            if value is not None:
                args.extend(["-F" if type(value) is int else "-f", f"{name}={value}"])
        result = (self.read if query.lstrip().startswith("query") else self.call)(args)
        value = json.loads(result)
        if value.get("errors"):
            error = transient_http_error(
                200, {}, value, operation="discussion.graphql", clock=self.clock
            )
            if error:
                raise error
            raise GitHubRequestError(200)
        return value["data"]

    def comments(self, discussion_id):
        result, cursor, seen = [], None, set()
        while True:
            node = self.graphql(
                _COMMENTS_QUERY, discussionId=discussion_id, cursor=cursor
            )["node"]
            if not node or node["id"] != discussion_id:
                raise GitHubRequestError(404)
            connection = node["comments"]
            result.extend(connection["nodes"])
            page = connection["pageInfo"]
            if not page["hasNextPage"]:
                return result
            cursor = page["endCursor"]
            if not cursor or cursor in seen:
                raise ValueError("invalid Discussion pagination")
            seen.add(cursor)


def _bot_login(login):
    return login if login.endswith("[bot]") else login + "[bot]"


def operation_marker(operation, run_id, episode_id):
    if not _OPERATION.fullmatch(operation) or not all(
        _IDENTIFIER.fullmatch(str(value)) for value in (run_id, episode_id)
    ):
        raise ValueError("invalid operation identity")
    return (
        "<!-- rsi-github-operation:"
        + json.dumps(
            {"episode": str(episode_id), "operation": operation, "run": str(run_id)},
            sort_keys=True,
            separators=(",", ":"),
        )
        + " -->"
    )


def ensure_comment(
    client, discussion_id, body, *, operation, run_id, episode_id, bot_login
):
    marker = operation_marker(operation, run_id, episode_id)
    rendered = body.rstrip() + "\n\n" + marker

    def attempt():
        for comment in client.comments(discussion_id):
            if (comment.get("author") or {}).get("login") == _bot_login(
                bot_login
            ) and comment.get("body", "").endswith(marker):
                return {"id": comment["id"], "url": comment.get("url", "")}
        return client.graphql(_ADD_COMMENT, discussionId=discussion_id, body=rendered)[
            "addDiscussionComment"
        ]["comment"]

    return retry_call(attempt, sleep=client.sleep, clock=client.clock)


def owner_lookup(client, login):
    if (
        re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}", login)
        is None
    ):
        raise ValueError("invalid GitHub login")
    try:
        membership = json.loads(
            client.read(["api", f"/orgs/OpenRSI-Foundation/memberships/{login}"])
        )
    except GitHubRequestError as error:
        if error.status == 404:
            return False
        raise
    return is_active_org_owner(membership, login)


def find_completed_review(comments, title, body, discussion_id, bot_login):
    reviews = []
    for comment in comments:
        if (comment.get("author") or {}).get("login") != _bot_login(bot_login):
            continue
        text = comment.get("body", "")
        marker = parse_review_marker(text, expected_discussion_node_id=discussion_id)
        if (
            marker is not None
            and marker.schema == 2
            and "<!-- rubric-review-status:completed -->" in text
        ):
            reviews.append(
                (
                    comment.get("updatedAt", ""),
                    comment.get("createdAt", ""),
                    comment["id"],
                    marker,
                )
            )
    if not reviews:
        return None
    _, _, comment_id, marker = max(reviews)
    if marker.proposal_sha256 != canonical_proposal_hash(title, body):
        return None
    return {"decision": marker.decision, "review_comment_id": comment_id}


def post_recovery_notice(
    client,
    *,
    discussion_id,
    source_repository,
    source_run_id,
    episode_id,
    status,
    bot_login,
):
    if source_repository not in {"OpenRSI-Foundation/OpenRSI-Index", "OpenRSI-Foundation/RSI-Skills"}:
        raise ValueError("invalid recovery source repository")
    bodies = {
        "retrying": "⏳ GitHub is temporarily unavailable. We are retrying your task automatically for up to 24 hours. No action is needed.",
        "exhausted": "⚠️ Automatic GitHub recovery reached its 24-hour limit. Your saved progress is kept. Retry the failed workflow run once GitHub is available.",
    }
    if status not in bodies:
        raise ValueError("invalid recovery status")
    return ensure_comment(
        client,
        discussion_id,
        bodies[status],
        operation="recovery." + status,
        run_id=source_run_id,
        episode_id=episode_id,
        bot_login=bot_login,
    )


def dispatch_request(client, request):
    kind = {"discussion_task_command": "task", "discussion_task_reset": "reset"}[
        request["event_type"]
    ]
    payload = request["client_payload"]
    expected = f"Discussion #{payload['discussion_number']} {kind}:{payload['triggering_comment_node_id']}"
    workflow = (
        "discussion-task-worker.yml" if kind == "task" else "discussion-task-reset.yml"
    )

    def existing():
        page = 1
        while True:
            response = json.loads(
                client.read(
                    [
                        "api",
                        f"/repos/OpenRSI-Foundation/RSI-Skills/actions/workflows/{workflow}/runs?event=repository_dispatch&per_page=100&page={page}",
                    ]
                )
            )
            runs = response["workflow_runs"]
            if any(
                run.get("display_title") == expected
                and run.get("event") == "repository_dispatch"
                for run in runs
            ):
                return True
            if len(runs) < 100:
                return False
            page += 1

    if existing():
        return
    args = [
        "api",
        "--method",
        "POST",
        "/repos/OpenRSI-Foundation/RSI-Skills/dispatches",
        "-f",
        "event_type=" + request["event_type"],
    ]
    for name, value in payload.items():
        args.extend(
            ["-F" if type(value) is int else "-f", f"client_payload[{name}]={value}"]
        )
    try:
        client.call(args)
    except GitHubTransientError as error:
        if error.first_failure_at is None:
            error.first_failure_at = (client.clock or time.time)()
        if error.refresh_credentials or (error.retry_after or 0) > 60:
            raise

        # An accepted dispatch can take time to appear. Poll its stable title,
        # never repeat this uncertain POST inside the same hosted run attempt.
        def observe():
            if not existing():
                raise error

        retry_call(observe, sleep=client.sleep, clock=client.clock)


def _outputs(**values):
    destination = os.environ.get("GITHUB_OUTPUT")
    if destination:
        with open(destination, "a", encoding="utf-8") as stream:
            for name, value in values.items():
                if "\n" in str(value) or "\r" in str(value):
                    raise ValueError("invalid workflow output")
                stream.write(f"{name}={value}\n")


def _review_reuse(client):
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    discussion = event["discussion"]
    discussion_id = discussion["node_id"]
    current = client.graphql(
        """query($id: ID!) { node(id: $id) {
      ... on Discussion { id title body category { name } }
    } }""",
        id=discussion_id,
    )["node"]
    if (
        not current
        or current["id"] != discussion_id
        or current["category"]["name"] != "Task Ideas"
        or canonical_proposal_hash(current["title"], current["body"])
        != canonical_proposal_hash(discussion["title"], discussion["body"])
    ):
        _outputs(current="false", reused="false")
        return
    result = find_completed_review(
        client.comments(discussion_id),
        current["title"],
        current["body"],
        discussion_id,
        os.environ["BOT_LOGIN"],
    )
    _outputs(current="true", reused="true" if result else "false", **(result or {}))


def proxy_gh(client, args):
    """Preserve existing gh snippets; intercept mutations needing reconciliation."""
    if "/repos/OpenRSI-Foundation/RSI-Skills/dispatches" in args:
        request = json.loads(Path(args[args.index("--input") + 1]).read_text())
        dispatch_request(client, request)
        return ""
    fields = dict(
        args[index + 1].split("=", 1)
        for index, arg in enumerate(args)
        if arg in ("-f", "-F")
    )
    query = fields.get("query", "")
    if "--paginate" in args and "comments(first:" in query:
        return json.dumps(
            [
                {
                    "data": {
                        "repository": {
                            "discussion": {
                                "comments": {
                                    "nodes": client.comments(
                                        os.environ["DISCUSSION_ID"]
                                    ),
                                }
                            }
                        }
                    }
                }
            ]
        )
    if "addDiscussionComment" in query or "updateDiscussionComment" in query:
        body = fields["body"]
        operation = "task.queue" if body == QUEUE_BODY else "review.comment"
        for status in ("running", "completed", "failed", "superseded"):
            if f"<!-- rubric-review-status:{status} -->" in body:
                operation = "review." + status
        run_id = os.environ["GITHUB_RUN_ID"]
        episode = os.environ.get("TRIGGER_COMMENT_ID") or run_id
        if "addDiscussionComment" in query:
            comment = ensure_comment(
                client,
                fields["discussionId"],
                body,
                operation=operation,
                run_id=run_id,
                episode_id=episode,
                bot_login=os.environ["BOT_LOGIN"],
            )
            return json.dumps({"data": {"addDiscussionComment": {"comment": comment}}})
        rendered = body.rstrip() + "\n\n" + operation_marker(operation, run_id, episode)

        def update():
            node = client.graphql(
                """query($id: ID!) { node(id: $id) {
              ... on DiscussionComment { id url body }
            } }""",
                id=fields["commentId"],
            )["node"]
            if not node:
                raise GitHubRequestError(404)
            if node["body"] == rendered:
                return {
                    "updateDiscussionComment": {
                        "comment": {"id": node["id"], "url": node.get("url", "")}
                    }
                }
            if "<!-- rubric-review-status:running -->" not in node["body"]:
                raise ValueError("review progress is no longer active")
            return client.graphql(query, commentId=fields["commentId"], body=rendered)

        return json.dumps(
            {"data": retry_call(update, sleep=client.sleep, clock=client.clock)}
        )
    if "--jq" in args and args[args.index("--jq") + 1] == '.data.node.body // ""':
        index = args.index("--jq")
        value = json.loads(client.read(args[:index] + args[index + 2 :]))
        return ((value.get("data") or {}).get("node") or {}).get("body") or ""
    return client.read(args)


def mark_retryable(error):
    emit_retry_diagnostic(error)
    if os.environ.get("GITHUB_ENV"):
        with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as stream:
            stream.write("RSI_GITHUB_RETRYABLE=true\n")


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("gh", "owner", "review-reuse", "notice"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    client = GitHubClient()
    if args.command == "gh":
        print(proxy_gh(client, args.arguments), end="")
    elif args.command == "owner":
        _outputs(
            is_owner="true"
            if owner_lookup(client, os.environ["COMMENTER_LOGIN"])
            else "false"
        )
    elif args.command == "review-reuse":
        _review_reuse(client)
    else:
        source_run = os.environ["SOURCE_RUN_ID"]
        number = os.environ["DISCUSSION_NUMBER"]
        if (
            not source_run.isdecimal()
            or int(source_run) <= 0
            or not number.isdecimal()
            or int(number) <= 0
        ):
            raise ValueError("invalid recovery source identity")
        discussion = client.graphql(
            """query($number: Int!) {
          repository(owner: "OpenRSI-Foundation", name: "OpenRSI-Index") {
            discussion(number: $number) { id }
          }
        }""",
            number=int(number),
        )["repository"]["discussion"]
        if not discussion:
            raise GitHubRequestError(404)
        result = post_recovery_notice(
            client,
            discussion_id=discussion["id"],
            source_repository=os.environ["SOURCE_REPOSITORY"],
            source_run_id=source_run,
            episode_id=os.environ["EPISODE_ID"],
            status=os.environ["RECOVERY_STATUS"],
            bot_login=os.environ["BOT_LOGIN"],
        )
        print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except GitHubTransientError as failure:
        mark_retryable(failure)
        raise SystemExit(75) from None
    except (GitHubRequestError, ValueError, KeyError):
        print(
            "Discussion GitHub operation failed permanently; check configuration or current authority.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
