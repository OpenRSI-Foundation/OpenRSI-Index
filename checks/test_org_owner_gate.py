from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from org_owner_gate import is_active_org_owner

SCRIPT = Path(__file__).with_name("org_owner_gate.py")


def active_owner() -> dict:
    return {
        "state": "active",
        "role": "admin",
        "user": {"login": "Current-Owner", "id": 99},
    }


def test_accepts_only_matching_active_org_owner():
    assert is_active_org_owner(active_owner(), "current-owner") is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(state="pending"),
        lambda value: value.update(role="member"),
        lambda value: value["user"].update(login="different-owner"),
        lambda value: value.update(user=None),
        lambda value: value.pop("role"),
    ],
)
def test_rejects_non_owner_stale_or_mismatched_membership(mutate):
    value = active_owner()
    mutate(value)

    assert is_active_org_owner(value, "current-owner") is False


@pytest.mark.parametrize("value", [None, [], "owner", 42, {}])
def test_rejects_malformed_membership_without_raising(value):
    assert is_active_org_owner(value, "current-owner") is False


def test_cli_writes_boolean_output_without_echoing_membership(tmp_path):
    membership_path = tmp_path / "membership.json"
    membership_path.write_text(json.dumps(active_owner()), encoding="utf-8")
    github_output = tmp_path / "github-output"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(membership_path), "current-owner"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_OUTPUT": str(github_output)},
    )

    assert result.stdout == "true\n"
    assert result.stderr == ""
    assert github_output.read_text(encoding="utf-8") == "is_owner=true\n"
