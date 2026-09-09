from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

_SOURCE_REPOSITORY = "RSI-Index/RSIs-First-Exam"
_TASK_PREFIX = "/task"
_RESET_COMMAND = "/reset"
_GITHUB_LOGIN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$"
)


def starts_task_command(body: object) -> bool:
    if not isinstance(body, str):
        return False
    command = body.lstrip()
    return command.startswith(_TASK_PREFIX) and (
        len(command) == len(_TASK_PREFIX)
        or command[len(_TASK_PREFIX)].isspace()
    )


def parse_dispatch_command(body: object) -> str | None:
    if starts_task_command(body):
        return "task"
    if isinstance(body, str) and body.strip() == _RESET_COMMAND:
        return "reset"
    return None


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _integer(value: object) -> bool:
    return type(value) is int


def build_dispatch_candidate(event: dict) -> dict | None:
    if not isinstance(event, dict) or event.get("action") != "created":
        return None

    repository = event.get("repository")
    discussion = event.get("discussion")
    comment = event.get("comment")
    if not all(isinstance(value, dict) for value in (repository, discussion, comment)):
        return None
    if repository.get("full_name") != _SOURCE_REPOSITORY:
        return None

    category = discussion.get("category")
    discussion_user = discussion.get("user")
    comment_user = comment.get("user")
    if not isinstance(category, dict):
        return None
    if category.get("name") != "Task Ideas":
        return None
    if not isinstance(discussion_user, dict) or not isinstance(comment_user, dict):
        return None

    discussion_author_id = discussion_user.get("id")
    comment_author_id = comment_user.get("id")
    if not _integer(discussion_author_id) or not _integer(comment_author_id):
        return None
    commenter_login = comment_user.get("login")
    if not isinstance(commenter_login, str) or not _GITHUB_LOGIN.fullmatch(
        commenter_login
    ):
        return None

    discussion_number = discussion.get("number")
    discussion_node_id = discussion.get("node_id")
    comment_node_id = comment.get("node_id")
    if not _integer(discussion_number):
        return None
    if not _nonempty_string(discussion_node_id) or not _nonempty_string(comment_node_id):
        return None
    command = parse_dispatch_command(comment.get("body"))
    if command is None:
        return None

    return {
        "command": command,
        "payload": {
            "source_repository": _SOURCE_REPOSITORY,
            "discussion_number": discussion_number,
            "discussion_node_id": discussion_node_id,
            "triggering_comment_node_id": comment_node_id,
            "trigger_kind": "comment",
        },
        "commenter_login": commenter_login,
        "is_author": discussion_author_id == comment_author_id,
    }


def _write_json_atomically(path: Path, value: object) -> None:
    path = path.absolute()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
    if len(arguments) != 2:
        raise SystemExit("usage: discussion_task_dispatch.py EVENT_JSON OUTPUT_JSON")

    event_path, output_path = (Path(argument) for argument in arguments)
    try:
        event = json.loads(event_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        event = None

    candidate = build_dispatch_candidate(event)
    payload = candidate["payload"] if candidate is not None else None
    _write_json_atomically(output_path, payload)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as output:
            output.write(f"candidate={'true' if candidate is not None else 'false'}\n")
            if candidate is not None:
                output.write(f"command={candidate['command']}\n")
                output.write(
                    f"is_author={'true' if candidate['is_author'] else 'false'}\n"
                )
                output.write(f"commenter_login={candidate['commenter_login']}\n")

    print("true" if candidate is not None else "false")


if __name__ == "__main__":
    main()
