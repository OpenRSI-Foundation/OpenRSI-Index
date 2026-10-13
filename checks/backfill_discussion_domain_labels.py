"""One-time backfill: give every Task Ideas Discussion its domain label.

Run by a maintainer, with their own `gh` login, after the "general task" and
"science task" labels exist. Prints the plan by default; `--apply` changes
labels. Only the two domain labels are touched, never a title or body.

    python3 checks/backfill_discussion_domain_labels.py            # dry run
    python3 checks/backfill_discussion_domain_labels.py --apply
"""

from __future__ import annotations

import argparse
import sys

if __package__:
    from .discussion_recovery import GitHubClient
    from .task_domain import MANAGED_LABELS, TASK_IDEAS, label_ids, parse_domain, sync
else:
    from discussion_recovery import GitHubClient
    from task_domain import MANAGED_LABELS, TASK_IDEAS, label_ids, parse_domain, sync

_CATEGORIES = """query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    discussionCategories(first: 100) { nodes { id name } }
  }
}"""
_DISCUSSIONS = """query($owner: String!, $name: String!, $category: ID!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $cursor, categoryId: $category) {
      nodes { id number title body labels(first: 100) { nodes { id name } } }
      pageInfo { hasNextPage endCursor }
    }
  }
}"""


def task_ideas(client: GitHubClient, owner: str, name: str) -> list[dict]:
    categories = client.graphql(_CATEGORIES, owner=owner, name=name)["repository"]
    category = next(
        (
            node["id"]
            for node in categories["discussionCategories"]["nodes"]
            if node["name"] == TASK_IDEAS
        ),
        None,
    )
    if category is None:
        raise SystemExit(f"{owner}/{name} has no {TASK_IDEAS!r} Discussion category")
    discussions, cursor, seen = [], None, set()
    while True:
        page = client.graphql(
            _DISCUSSIONS, owner=owner, name=name, category=category, cursor=cursor
        )["repository"]["discussions"]
        discussions.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            return discussions
        cursor = page["pageInfo"]["endCursor"]
        if not cursor or cursor in seen:
            raise SystemExit("invalid Discussion pagination")
        seen.add(cursor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default="OpenRSI-Foundation/OpenRSI-Index")
    parser.add_argument("--apply", action="store_true", help="change labels (default: dry run)")
    args = parser.parse_args(argv)
    owner, _, name = args.repo.partition("/")
    client = GitHubClient()
    ids = label_ids(client, owner, name)
    missing = [label for label in MANAGED_LABELS if label not in ids]
    if missing and args.apply:
        print(f"error: create the labels {missing} first", file=sys.stderr)
        return 1
    if missing:
        print(f"note: labels not created yet: {missing}", file=sys.stderr)
    discussions = task_ideas(client, owner, name)
    changes = 0
    for discussion in sorted(discussions, key=lambda item: item["number"]):
        label, add, remove = sync(client, discussion, ids, apply=args.apply)
        changes += bool(add or remove)
        print(
            f"#{discussion['number']:<5} {parse_domain(discussion.get('body')) or 'unspecified':<12}"
            f" -> {label:<13} add={add or '-'} remove={remove or '-'}  {discussion['title']}"
        )
    verb = "changed" if args.apply else "would change"
    print(f"{len(discussions)} Task Ideas Discussions; {verb} {changes}.")
    if not args.apply and changes:
        print("Dry run only; re-run with --apply to change labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
