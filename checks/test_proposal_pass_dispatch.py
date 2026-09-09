from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).with_name("proposal_pass_dispatch.py")


@pytest.fixture
def dispatch():
    spec = importlib.util.spec_from_file_location("proposal_pass_dispatch", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proposal_event(*, action: str = "created") -> dict:
    return {
        "action": action,
        "repository": {"full_name": "RSI-Index/RSIs-First-Exam"},
        "discussion": {
            "number": 128,
            "node_id": "D_kw128",
            "updated_at": "2026-08-30T09:59:00Z",
            "title": "Train a bounded reasoning model",
            "body": "Proposal body with private-looking review content.",
            "category": {"name": "Task Ideas"},
            "user": {"id": 42, "login": "proposal-author"},
        },
        "sender": {"id": 42, "login": "proposal-author"},
        "installation": {"id": 777},
    }


@pytest.mark.parametrize("action", ["created", "edited"])
def test_builds_exact_identifier_only_payload_for_reviewed_discussion(dispatch, action):
    event = proposal_event(action=action)

    payload = dispatch.build_proposal_pass_payload(event, "DC_pass")

    assert payload == {
        "source_repository": "RSI-Index/RSIs-First-Exam",
        "discussion_number": 128,
        "discussion_node_id": "D_kw128",
        "triggering_comment_node_id": "DC_pass",
        "trigger_kind": "proposal_pass",
    }
    serialized = json.dumps(payload, sort_keys=True)
    for forbidden in (
        event["discussion"]["title"],
        event["discussion"]["body"],
        event["discussion"]["user"]["login"],
        "review content",
        "rubric text",
        json.dumps(event, sort_keys=True),
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event.update(action="deleted"),
        lambda event: event["repository"].update(full_name="other/repository"),
        lambda event: event["discussion"]["category"].update(name="General"),
        lambda event: event["discussion"].pop("number"),
        lambda event: event["discussion"].update(number="128"),
        lambda event: event["discussion"].update(number=True),
        lambda event: event["discussion"].update(number=0),
        lambda event: event["discussion"].update(node_id="D bad"),
        lambda event: event["discussion"].update(node_id=""),
    ],
)
def test_rejects_wrong_incomplete_or_malformed_discussion_event(dispatch, mutate):
    event = proposal_event()
    mutate(event)

    assert dispatch.build_proposal_pass_payload(event, "DC_pass") is None


@pytest.mark.parametrize("comment_id", [None, 1, True, [], {}, "", "DC bad", "DC/pass"])
def test_rejects_non_string_or_malformed_review_comment_node_id(dispatch, comment_id):
    assert dispatch.build_proposal_pass_payload(proposal_event(), comment_id) is None


@pytest.mark.parametrize(
    "event",
    [None, [], {}, {"action": "created"}, {"action": "created", "repository": []}],
)
def test_malformed_untrusted_event_shapes_return_none_without_raising(dispatch, event):
    assert dispatch.build_proposal_pass_payload(event, "DC_pass") is None


def test_cli_builds_payload_without_live_graphql_authority(tmp_path):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "dispatch.json"
    event_path.write_text(json.dumps(proposal_event()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(event_path),
            "DC_pass",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "true\n"
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8") == (
        '{"discussion_node_id":"D_kw128","discussion_number":128,'
        '"source_repository":"RSI-Index/RSIs-First-Exam",'
        '"trigger_kind":"proposal_pass",'
        '"triggering_comment_node_id":"DC_pass"}'
    )
    assert not list(tmp_path.glob(".dispatch.json.*"))


def test_cli_writes_null_for_rejected_event(tmp_path):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "dispatch.json"
    event = proposal_event()
    event["discussion"]["category"]["name"] = "General"
    event_path.write_text(json.dumps(event), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(event_path),
            "DC_pass",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "false\n"
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8") == "null"
