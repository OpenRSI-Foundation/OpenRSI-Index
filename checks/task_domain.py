"""Task Ideas Discussion domain labels and proposal pipelines.

A proposal records its task-intake answers as the first rows of its review
table:

    | Task Type | Domain | AI4AI |
    | Task Type | Domain | AI4Sci |
    | Task Type | Involves model training | Yes |
    | Task Type | Involves model training | No |

Matching is case-insensitive and tolerant of spacing. The domain alone decides
the Discussion label; a missing or unknown domain means a general task. Only
labels, never the title or body, are changed: the approval hash covers both.

The training answer decides the pipeline. A proposal recorded as AI4Sci without
model training belongs to the science pipeline, which does not exist yet: the
AI4AI rubric review and task building skip it. Everything else, including a
missing or unknown answer, uses the general pipeline.

`apply` labels one Discussion (the review workflow passes its node id).
`pipeline` prints the pipeline of the Discussion in a workflow event.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable
from pathlib import Path

if __package__:
    from .discussion_recovery import GitHubClient, GitHubRequestError
    from .github_retry import GitHubTransientError, retry_call
else:
    from discussion_recovery import GitHubClient, GitHubRequestError
    from github_retry import GitHubTransientError, retry_call

GENERAL = "general task"
SCIENCE = "science task"
# Canonical domain answer (casefolded) -> Discussion label.
DOMAIN_LABELS = {"ai4ai": GENERAL, "ai4sci": SCIENCE}
DEFAULT_LABEL = GENERAL
MANAGED_LABELS = (GENERAL, SCIENCE)
TASK_IDEAS = "Task Ideas"

GENERAL_PIPELINE = "general"
SCIENCE_PIPELINE = "science"
# Science-pipeline hook: (domain, training) answers routed away from the general
# AI4AI review and task building. Mirrors the task-intake routes.
SCIENCE_PIPELINE_ROUTES = frozenset({("ai4sci", "no")})


def _task_type_row(field: str) -> re.Pattern[str]:
    words = r"[ \t]+".join(map(re.escape, field.split()))
    return re.compile(
        rf"^[ \t]*\|[ \t]*task[ \t]+type[ \t]*\|[ \t]*{words}[ \t]*\|([^|\n]*)\|[ \t]*\r?$",
        re.IGNORECASE | re.MULTILINE,
    )


_DOMAIN_ROW = _task_type_row("domain")
_TRAINING_ROW = _task_type_row("involves model training")

_DISCUSSION = """query($id: ID!) { node(id: $id) { ... on Discussion {
  id number body category { name } repository { owner { login } name }
  labels(first: 100) { nodes { id name } }
} } }"""
_ADD = """mutation($labelable: ID!, $label: ID!) {
  addLabelsToLabelable(input: {labelableId: $labelable, labelIds: [$label]}) { clientMutationId }
}"""
_REMOVE = """mutation($labelable: ID!, $label: ID!) {
  removeLabelsFromLabelable(input: {labelableId: $labelable, labelIds: [$label]}) { clientMutationId }
}"""


def _row_answer(row: re.Pattern[str], body: str | None) -> str | None:
    match = row.search(body or "")
    if not match:
        return None
    return match.group(1).strip().strip("*`").strip() or None


def parse_domain(body: str | None) -> str | None:
    """The first domain row's answer, stripped, or None without one."""
    return _row_answer(_DOMAIN_ROW, body)


def parse_training(body: str | None) -> str | None:
    """The first model-training row's answer, stripped, or None without one."""
    return _row_answer(_TRAINING_ROW, body)


def pipeline(body: str | None) -> str:
    """`science` for AI4Sci without model training, else `general`."""
    answers = ((parse_domain(body) or "").casefold(), (parse_training(body) or "").casefold())
    return SCIENCE_PIPELINE if answers in SCIENCE_PIPELINE_ROUTES else GENERAL_PIPELINE


def domain_label(body: str | None) -> str:
    domain = parse_domain(body)
    return DOMAIN_LABELS.get((domain or "").casefold(), DEFAULT_LABEL)


def plan(body: str | None, current: Iterable[str]) -> tuple[str, list[str], list[str]]:
    """(label, managed labels to add, managed labels to remove)."""
    label = domain_label(body)
    current = set(current)
    add = [] if label in current else [label]
    remove = [name for name in MANAGED_LABELS if name != label and name in current]
    return label, add, remove


class MissingLabelError(RuntimeError):
    pass


def label_ids(client: GitHubClient, owner: str, name: str) -> dict[str, str]:
    """Node ids of the managed labels that exist in the repository."""
    aliases = "\n".join(
        f'l{index}: label(name: "{label}") {{ id name }}'
        for index, label in enumerate(MANAGED_LABELS)
    )
    query = (
        "query($owner: String!, $name: String!) {\n"
        f"  repository(owner: $owner, name: $name) {{\n{aliases}\n  }}\n}}"
    )
    repository = client.graphql(query, owner=owner, name=name)["repository"]
    if not repository:
        raise GitHubRequestError(404)
    return {
        node["name"]: node["id"]
        for node in repository.values()
        if isinstance(node, dict) and node.get("name") in MANAGED_LABELS
    }


def sync(
    client: GitHubClient,
    discussion: dict,
    ids: dict[str, str],
    *,
    apply: bool = True,
) -> tuple[str, list[str], list[str]]:
    current = {node["name"]: node["id"] for node in discussion["labels"]["nodes"]}
    label, add, remove = plan(discussion.get("body"), current)
    if not apply:
        return label, add, remove
    for name in add:
        if name not in ids:
            raise MissingLabelError(
                f"Discussion label {name!r} does not exist in this repository; "
                "a maintainer must create it (the workflow token cannot)."
            )
    for names, mutation, lookup in ((add, _ADD, ids), (remove, _REMOVE, current)):
        for name in names:
            # Both mutations are idempotent, so retrying them is safe.
            retry_call(
                lambda: client.graphql(
                    mutation, labelable=discussion["id"], label=lookup[name]
                ),
                sleep=client.sleep,
                clock=client.clock,
            )
    return label, add, remove


def apply_one(client: GitHubClient, node_id: str) -> None:
    discussion = client.graphql(_DISCUSSION, id=node_id)["node"]
    if not discussion or discussion.get("id") != node_id:
        raise GitHubRequestError(404)
    category = (discussion.get("category") or {}).get("name")
    if category != TASK_IDEAS:
        print(f"Discussion #{discussion['number']} is not in {TASK_IDEAS!r}; no domain label.")
        return
    repository = discussion["repository"]
    ids = label_ids(client, repository["owner"]["login"], repository["name"])
    label, add, remove = sync(client, discussion, ids)
    print(
        f"Discussion #{discussion['number']}: domain "
        f"{parse_domain(discussion.get('body')) or 'unspecified'} -> {label!r}; "
        f"added {add or 'none'}, removed {remove or 'none'}"
    )


def event_pipeline(event_path: str) -> str:
    """The pipeline of the Discussion in a workflow event (its body at event time)."""
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    discussion = event.get("discussion") if isinstance(event, dict) else None
    body = discussion.get("body") if isinstance(discussion, dict) else None
    return pipeline(body if isinstance(body, str) else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("apply", "pipeline"))
    parser.add_argument(
        "--discussion-node-id", default=os.environ.get("DISCUSSION_NODE_ID")
    )
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_PATH"))
    args = parser.parse_args(argv)
    if args.command == "pipeline":
        if not args.event:
            parser.error("--event or GITHUB_EVENT_PATH is required")
        name = event_pipeline(args.event)
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding="utf-8") as stream:
                stream.write(f"pipeline={name}\n")
        print(name)
        return 0
    if not args.discussion_node_id:
        parser.error("--discussion-node-id or DISCUSSION_NODE_ID is required")
    try:
        apply_one(GitHubClient(), args.discussion_node_id)
    except MissingLabelError as error:
        print(f"::error::{error}")
        return 1
    except GitHubTransientError:
        print("::error::GitHub is temporarily unavailable; re-run this job.")
        return 1
    except (GitHubRequestError, KeyError, TypeError):
        print("::error::Labelling the Discussion failed; check the App token's Discussion access.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
