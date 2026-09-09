from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

_SOURCE_REPOSITORY = "RSI-Index/RSIs-First-Exam"
_DISCUSSION_ACTIONS = frozenset({"created", "edited"})
_NODE_ID = re.compile(r"[A-Za-z0-9_-]+\Z")


def _node_id(value: object) -> bool:
    return isinstance(value, str) and _NODE_ID.fullmatch(value) is not None


def build_proposal_pass_payload(
    event: object, review_comment_node_id: object
) -> dict[str, object] | None:
    """Build the identifier-only private dispatch payload for a published Pass."""
    if not isinstance(event, dict) or event.get("action") not in _DISCUSSION_ACTIONS:
        return None
    repository = event.get("repository")
    discussion = event.get("discussion")
    if not isinstance(repository, dict) or not isinstance(discussion, dict):
        return None
    if repository.get("full_name") != _SOURCE_REPOSITORY:
        return None
    category = discussion.get("category")
    if not isinstance(category, dict) or category.get("name") != "Task Ideas":
        return None

    discussion_number = discussion.get("number")
    discussion_node_id = discussion.get("node_id")
    if type(discussion_number) is not int or discussion_number <= 0:
        return None
    if not _node_id(discussion_node_id) or not _node_id(review_comment_node_id):
        return None

    return {
        "source_repository": _SOURCE_REPOSITORY,
        "discussion_number": discussion_number,
        "discussion_node_id": discussion_node_id,
        "triggering_comment_node_id": review_comment_node_id,
        "trigger_kind": "proposal_pass",
    }
def _write_json_atomically(path: Path, value: object) -> None:
    path = path.absolute()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                value,
                output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        raise SystemExit(
            "usage: proposal_pass_dispatch.py "
            "EVENT_JSON REVIEW_COMMENT_NODE_ID OUTPUT_JSON"
        )

    event_path = Path(arguments[0])
    review_comment_node_id = arguments[1]
    output_path = Path(arguments[2])
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        event = None
    payload = build_proposal_pass_payload(event, review_comment_node_id)
    _write_json_atomically(output_path, payload)
    print("true" if payload is not None else "false")


if __name__ == "__main__":
    main()
