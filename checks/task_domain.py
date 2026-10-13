"""Task Ideas Discussion domain labels.

A proposal declares its domain in one row of its review table:

    | Task Type | Domain | AI4AI |
    | Task Type | Domain | AI4Sci |

Matching is case-insensitive and tolerant of spacing. The domain alone decides
the Discussion label; a missing or unknown domain means a general task. Only
labels, never the title or body, are changed: the approval hash covers both.

`apply` labels one Discussion (the review workflow passes its node id).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Iterable

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

_DOMAIN_ROW = re.compile(
    r"^[ \t]*\|[ \t]*task[ \t]+type[ \t]*\|[ \t]*domain[ \t]*\|([^|\n]*)\|[ \t]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)

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


def parse_domain(body: str | None) -> str | None:
    """The first domain row's answer, stripped, or None without one."""
    match = _DOMAIN_ROW.search(body or "")
    if not match:
        return None
    return match.group(1).strip().strip("*`").strip() or None


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("apply",))
    parser.add_argument(
        "--discussion-node-id", default=os.environ.get("DISCUSSION_NODE_ID")
    )
    args = parser.parse_args(argv)
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
