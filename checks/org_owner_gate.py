from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def is_active_org_owner(membership: object, expected_login: str) -> bool:
    if not isinstance(membership, dict):
        return False
    user = membership.get("user")
    if not isinstance(user, dict):
        return False
    actual_login = user.get("login")
    return (
        membership.get("state") == "active"
        and membership.get("role") == "admin"
        and isinstance(actual_login, str)
        and actual_login.casefold() == expected_login.casefold()
    )


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        raise SystemExit("usage: org_owner_gate.py MEMBERSHIP_JSON EXPECTED_LOGIN")

    membership_path = Path(arguments[0])
    try:
        membership = json.loads(membership_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        membership = None
    allowed = is_active_org_owner(membership, arguments[1])

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"is_owner={'true' if allowed else 'false'}\n")
    print("true" if allowed else "false")


if __name__ == "__main__":
    main()
