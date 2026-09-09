from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from discussion_task_dispatch import build_dispatch_candidate, starts_task_command

SCRIPT = Path(__file__).with_name("discussion_task_dispatch.py")


def accepted_event() -> dict:
    return {
        "action": "created",
        "repository": {"full_name": "RSI-Index/RSIs-First-Exam"},
        "discussion": {
            "number": 128,
            "node_id": "D_kw128",
            "category": {"name": "Task Ideas"},
            "user": {"id": 42, "login": "author"},
        },
        "comment": {
            "node_id": "DC_kw900",
            "body": "  /task use accuracy",
            "user": {"id": 42, "login": "author"},
        },
    }


def test_builds_identifier_only_candidate_for_author_task_command():
    event = accepted_event()

    assert build_dispatch_candidate(event) == {
        "command": "task",
        "payload": {
            "source_repository": "RSI-Index/RSIs-First-Exam",
            "discussion_number": 128,
            "discussion_node_id": "D_kw128",
            "triggering_comment_node_id": "DC_kw900",
            "trigger_kind": "comment",
        },
        "commenter_login": "author",
        "is_author": True,
    }

    output = json.dumps(build_dispatch_candidate(event)["payload"], sort_keys=True)
    for untrusted_value in (
        event["comment"]["body"],
        event["comment"]["user"]["login"],
        "Proposal body",
        "/task",
        json.dumps(event),
    ):
        assert untrusted_value not in output


def test_builds_candidate_for_non_author_so_owner_can_be_checked_authoritatively():
    event = accepted_event()
    event["comment"]["user"] = {"id": 99, "login": "current-owner"}

    candidate = build_dispatch_candidate(event)

    assert candidate == {
        "command": "task",
        "payload": {
            "source_repository": "RSI-Index/RSIs-First-Exam",
            "discussion_number": 128,
            "discussion_node_id": "D_kw128",
            "triggering_comment_node_id": "DC_kw900",
            "trigger_kind": "comment",
        },
        "commenter_login": "current-owner",
        "is_author": False,
    }
    assert "current-owner" not in json.dumps(candidate["payload"], sort_keys=True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda event: event.update(action="edited"),
        lambda event: event["repository"].update(full_name="other/repository"),
        lambda event: event["discussion"]["category"].update(name="General"),
        lambda event: event["discussion"].pop("number"),
        lambda event: event["discussion"].pop("node_id"),
        lambda event: event["comment"].pop("node_id"),
        lambda event: event["discussion"]["user"].pop("id"),
        lambda event: event["comment"]["user"].pop("id"),
        lambda event: event["comment"].update(body="ordinary comment"),
    ],
)
def test_rejects_wrong_or_incomplete_event(mutate):
    event = accepted_event()
    mutate(event)

    assert build_dispatch_candidate(event) is None


@pytest.mark.parametrize(
    "body",
    ["/tasks", "/tasking", "/Task use accuracy", "ordinary comment"],
)
def test_task_command_requires_exact_lowercase_token_boundary(body):
    event = accepted_event()
    event["comment"]["body"] = body

    assert starts_task_command(body) is False
    assert build_dispatch_candidate(event) is None


@pytest.mark.parametrize("body", ["/task", "/task ", "/task use accuracy", "\t/task\n"])
def test_task_command_accepts_identifier_only_command_forms(body):
    assert starts_task_command(body) is True


def test_reset_requires_the_exact_lowercase_command_and_builds_a_reset_candidate():
    event = accepted_event()
    event["comment"]["body"] = "\n/reset\t"

    candidate = build_dispatch_candidate(event)

    assert candidate == {
        "command": "reset",
        "payload": {
            "source_repository": "RSI-Index/RSIs-First-Exam",
            "discussion_number": 128,
            "discussion_node_id": "D_kw128",
            "triggering_comment_node_id": "DC_kw900",
            "trigger_kind": "comment",
        },
        "commenter_login": "author",
        "is_author": True,
    }


@pytest.mark.parametrize("body", ["/Reset", "/resets", "/reset now", "prefix /reset"])
def test_reset_rejects_ambiguous_or_extended_forms(body):
    event = accepted_event()
    event["comment"]["body"] = body

    assert build_dispatch_candidate(event) is None


@pytest.mark.parametrize(
    "event",
    [
        None,
        [],
        {"action": "created"},
        {"action": "created", "repository": []},
        {"action": "created", "repository": {"full_name": []}},
        {"action": "created", "repository": {"full_name": "repo"}, "discussion": None},
        {"action": "created", "repository": {"full_name": "repo"}, "discussion": {}},
        {"action": "created", "repository": {"full_name": "repo"}, "discussion": {"category": []}},
        {"action": "created", "repository": {"full_name": "repo"}, "discussion": {"user": {"id": "42"}}},
        {"action": "created", "repository": {"full_name": "repo"}, "discussion": {"number": True}},
    ],
)
def test_malformed_untrusted_event_shapes_return_none_without_raising(event):
    assert build_dispatch_candidate(event) is None


def test_author_identity_uses_numeric_ids_even_when_logins_match():
    event = accepted_event()
    event["discussion"]["user"] = {"id": 7, "login": "same-login"}
    event["comment"]["user"] = {"id": 8, "login": "same-login"}

    assert build_dispatch_candidate(event)["is_author"] is False


@pytest.mark.parametrize("bad_id", ["42", 42.0, True, None, []])
def test_author_identity_requires_numeric_integer_ids(bad_id):
    event = accepted_event()
    event["comment"]["user"]["id"] = bad_id

    assert build_dispatch_candidate(event) is None


@pytest.mark.parametrize(
    "login",
    ["", "-owner", "owner-", "owner--name", "owner/name", "a" * 40, 42, None],
)
def test_candidate_requires_a_valid_github_commenter_login(login):
    event = accepted_event()
    event["comment"]["user"]["login"] = login

    assert build_dispatch_candidate(event) is None


def test_cli_atomically_writes_compact_sorted_payload_and_only_prints_decision(tmp_path):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "dispatch.json"
    event = accepted_event()
    event["discussion"]["body"] = "Proposal body"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    env = os.environ.copy()
    github_output = tmp_path / "github-output"
    env["GITHUB_OUTPUT"] = str(github_output)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(event_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout == "true\n"
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8") == (
        '{"discussion_node_id":"D_kw128","discussion_number":128,'
        '"source_repository":"RSI-Index/RSIs-First-Exam",'
        '"trigger_kind":"comment",'
        '"triggering_comment_node_id":"DC_kw900"}'
    )
    assert github_output.read_text(encoding="utf-8") == (
        "candidate=true\ncommand=task\nis_author=true\ncommenter_login=author\n"
    )
    assert "Proposal body" not in result.stdout
    assert "Proposal body" not in github_output.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".dispatch.json.*"))


def test_cli_writes_null_and_false_for_rejected_event_without_body_leakage(tmp_path):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "dispatch.json"
    body = "Proposal body /task secret"
    event = accepted_event()
    event["comment"]["body"] = body
    event["discussion"]["category"]["name"] = "General"
    event_path.write_text(json.dumps(event), encoding="utf-8")
    env = os.environ.copy()
    github_output = tmp_path / "github-output"
    env["GITHUB_OUTPUT"] = str(github_output)

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(event_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.stdout == "false\n"
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8") == "null"
    assert github_output.read_text(encoding="utf-8") == "candidate=false\n"
    assert body not in result.stdout
    assert body not in github_output.read_text(encoding="utf-8")


def test_cli_treats_malformed_event_json_as_rejected(tmp_path):
    event_path = tmp_path / "event.json"
    output_path = tmp_path / "dispatch.json"
    event_path.write_text("not-json", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(event_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_OUTPUT": ""},
    )

    assert result.stdout == "false\n"
    assert result.stderr == ""
    assert output_path.read_text(encoding="utf-8") == "null"
