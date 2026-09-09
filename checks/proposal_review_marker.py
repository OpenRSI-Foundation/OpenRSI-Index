from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass

_MARKER_PREFIX = "<!-- rsi-proposal-review:"
_MARKER_PATTERN = re.compile(r"<!-- rsi-proposal-review:(.*?) -->")
_REQUIRED_KEYS = {"decision", "discussion_node_id", "proposal_sha256", "schema"}
_SCHEMA_DECISIONS = {
    1: frozenset({"Reject", "Accept", "Strong Accept"}),
    2: frozenset({"Reject", "Pass"}),
}
_ELIGIBLE_DECISIONS = {
    1: frozenset({"Accept", "Strong Accept"}),
    2: frozenset({"Pass"}),
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ReviewMarker:
    schema: int
    decision: str
    proposal_sha256: str
    discussion_node_id: str

    @property
    def eligible(self) -> bool:
        return self.decision in _ELIGIBLE_DECISIONS.get(self.schema, frozenset())


def canonical_proposal_bytes(title: str, body: str) -> bytes:
    return json.dumps(
        {"body": body, "title": title},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_proposal_hash(title: str, body: str) -> str:
    return hashlib.sha256(canonical_proposal_bytes(title, body)).hexdigest()


def _is_valid_marker_values(
    schema: object,
    decision: object,
    proposal_sha256: object,
    discussion_node_id: object,
) -> bool:
    return (
        type(schema) is int
        and schema in _SCHEMA_DECISIONS
        and isinstance(decision, str)
        and decision in _SCHEMA_DECISIONS[schema]
        and isinstance(proposal_sha256, str)
        and _SHA256_PATTERN.fullmatch(proposal_sha256) is not None
        and isinstance(discussion_node_id, str)
        and bool(discussion_node_id)
    )


def _json_object_with_unique_keys(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def render_review_marker(
    decision: str,
    proposal_sha256: str,
    discussion_node_id: str,
) -> str:
    if not _is_valid_marker_values(2, decision, proposal_sha256, discussion_node_id):
        raise ValueError("review marker values must be canonical")

    payload = json.dumps(
        {
            "decision": decision,
            "discussion_node_id": discussion_node_id,
            "proposal_sha256": proposal_sha256,
            "schema": 2,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{_MARKER_PREFIX}{payload} -->"


def parse_review_marker(
    body: str, *, expected_discussion_node_id: str | None = None
) -> ReviewMarker | None:
    if not isinstance(body, str) or body.count(_MARKER_PREFIX) != 1:
        return None

    matches = _MARKER_PATTERN.findall(body)
    if len(matches) != 1:
        return None

    try:
        payload = json.loads(matches[0], object_pairs_hook=_json_object_with_unique_keys)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict) or set(payload) != _REQUIRED_KEYS:
        return None

    if not _is_valid_marker_values(
        payload["schema"],
        payload["decision"],
        payload["proposal_sha256"],
        payload["discussion_node_id"],
    ):
        return None

    marker = ReviewMarker(
        schema=payload["schema"],
        decision=payload["decision"],
        proposal_sha256=payload["proposal_sha256"],
        discussion_node_id=payload["discussion_node_id"],
    )
    if (
        expected_discussion_node_id is not None
        and marker.discussion_node_id != expected_discussion_node_id
    ):
        return None
    return marker


def main() -> None:
    print(
        render_review_marker(
            decision=os.environ["DECISION"],
            proposal_sha256=canonical_proposal_hash(
                os.environ["DISCUSSION_TITLE"], os.environ["DISCUSSION_BODY"]
            ),
            discussion_node_id=os.environ["DISCUSSION_NODE_ID"],
        )
    )


if __name__ == "__main__":
    main()
