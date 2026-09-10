from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn


_SUPPORTED_PLATFORMS = frozenset({"codex", "claude-code"})
PUBLIC_REPOSITORY = "RSI-Index/RSIs-First-Exam"
DISCUSSION_CATEGORY = "Task Ideas"
_DISPATCHER_LOGINS = frozenset(
    {"rsi-index-task-dispatcher", "rsi-index-task-dispatcher[bot]"}
)
_WRITE_PERMISSIONS = frozenset({"WRITE", "MAINTAIN", "ADMIN"})
_REPOSITORY_URL = re.compile(
    r"(?<![A-Za-z0-9_.-])https://github\.com/"
    r"(RSI-Index/[A-Za-z0-9_-](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?)"
    r"(?=$|[\s<>()\[\]{},;!?]|\.(?=\s|$))"
)
_NODE_ID = r"[A-Za-z0-9_-]+"

PUBLIC_REPOSITORY_QUERY = """
query PublicRepository($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    id
    discussionCategories(first: 100, after: $after) {
      nodes { id name }
      pageInfo { hasNextPage endCursor }
    }
  }
}
"""

CREATE_DISCUSSION = """
mutation CreateProposal($repositoryId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {
    repositoryId: $repositoryId,
    categoryId: $categoryId,
    title: $title,
    body: $body
  }) {
    discussion { id number url }
  }
}
"""

UPDATE_DISCUSSION = """
mutation UpdateProposal($id: ID!, $body: String!) {
  updateDiscussion(input: {discussionId: $id, body: $body}) {
    discussion { id number url }
  }
}
"""

DISCUSSION_QUERY = """
query ProposalDiscussion($id: ID!, $after: String) {
  node(id: $id) {
    __typename
    ... on Discussion {
      id number url title body createdAt author { login }
      comments(first: 100, after: $after) {
        nodes {
          id body createdAt author { login }
          replies(first: 100) {
            nodes { id body createdAt author { login } }
            pageInfo { hasNextPage endCursor }
          }
        }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""

DISCUSSION_REPLIES_QUERY = """
query DiscussionReplies($id: ID!, $after: String!) {
  node(id: $id) {
    __typename
    ... on DiscussionComment {
      id
      replies(first: 100, after: $after) {
        nodes { id body createdAt author { login } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


class ProposalSessionError(RuntimeError):
    """A local problem the contributor can resolve before continuing."""


@dataclass(frozen=True)
class DiscussionRef:
    node_id: str
    number: int
    url: str


@dataclass(frozen=True)
class DiscussionEntry:
    node_id: str
    author: str
    body: str
    created_at: str
    parent_node_id: str | None = None


@dataclass(frozen=True)
class DiscussionSnapshot:
    ref: DiscussionRef
    title: str
    body: str
    author: str
    created_at: str
    entries: tuple[DiscussionEntry, ...]


@dataclass(frozen=True)
class TaskRepository:
    name_with_owner: str
    url: str
    visibility: str
    viewer_permission: str
    default_branch: str


@dataclass(frozen=True)
class UploadResult:
    repository: str
    url: str
    commit_sha: str


CommandRunner = Callable[
    [Sequence[str], Path | None], subprocess.CompletedProcess[str]
]


def _fail(message: str) -> NoReturn:
    raise ProposalSessionError(message)


def _state_dir(checkout_root: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout_root), "rev-parse", "--absolute-git-dir"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(f"checkout is not a Git repository: {checkout_root}")
    return Path(completed.stdout.strip()) / "rsi-proposal-session"


def _checkout_root(checkout_root: Path) -> Path:
    try:
        root = checkout_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail(f"checkout does not exist: {checkout_root}")
    if not root.is_dir():
        _fail(f"checkout is not a directory: {root}")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(f"checkout is not a Git repository: {root}")
    actual_root = Path(completed.stdout.strip()).resolve()
    if actual_root != root:
        _fail(f"checkout must be the Git worktree root: {root}")
    return root


def _json_object_with_unique_keys(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_json_object_with_unique_keys
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _fail(f"{label} is malformed: {error}")
    if not isinstance(parsed, dict) or not all(isinstance(key, str) for key in parsed):
        _fail(f"{label} must be a JSON object")
    return parsed


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_transcript(value: Path) -> Path:
    try:
        resolved = value.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail(f"native transcript does not exist: {value}")
    if not resolved.is_file():
        _fail(f"native transcript is not a file: {resolved}")
    return resolved


def _read_discussion(value: object) -> DiscussionRef | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"node_id", "number", "url"}:
        _fail("saved Discussion is malformed")
    node_id = value["node_id"]
    number = value["number"]
    url = value["url"]
    if (
        not isinstance(node_id, str)
        or not node_id
        or type(number) is not int
        or number < 0
        or not isinstance(url, str)
        or not url
    ):
        _fail("saved Discussion has invalid values")
    # Keep sessions bound before the repository rename usable with GitHub's
    # canonical URL, without changing the bound Discussion node or number.
    if url == f"https://github.com/RSI-Index/RSI-Index-Public/discussions/{number}":
        url = f"https://github.com/{PUBLIC_REPOSITORY}/discussions/{number}"
    return DiscussionRef(node_id=node_id, number=number, url=url)


def load_discussion(
    checkout_root: Path, *, required: bool = True
) -> DiscussionRef | None:
    root = _checkout_root(checkout_root)
    state = _state_dir(root)
    discussion_path = state / "discussion.json"
    if discussion_path.is_file():
        ref = _read_discussion(_read_json(discussion_path, "Discussion state"))
    elif (state / "binding.json").is_file():
        # Reuse Discussions created by the former hook-based helper. Neither
        # activation nor a transcript is required to continue that Discussion.
        ref = _read_discussion(_read_json(state / "binding.json", "legacy session").get("discussion"))
    else:
        ref = None
    if required and ref is None:
        _fail("Discussion has not been created")
    return ref


@contextmanager
def _discussion_lock(state_dir: Path):
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(state_dir / "discussion.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        with os.fdopen(descriptor, "a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        _fail(f"cannot lock proposal session state: {error}")


def _run(
    arguments: Sequence[str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(arguments),
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        _fail(f"cannot run {arguments[0]}: {error}")


def _checked_command(
    run: CommandRunner,
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    label: str,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = run(list(arguments), cwd)
    except (OSError, subprocess.SubprocessError) as error:
        _fail(f"{label} failed: {error}")
    if not isinstance(completed, subprocess.CompletedProcess):
        _fail(f"{label} returned an invalid result")
    if completed.returncode != 0:
        detail = completed.stderr.strip() if isinstance(completed.stderr, str) else ""
        suffix = f": {detail}" if detail else ""
        _fail(f"{label} failed{suffix}")
    if not isinstance(completed.stdout, str):
        _fail(f"{label} returned invalid output")
    return completed


def _command_json(
    run: CommandRunner,
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    label: str,
) -> dict[str, object]:
    output = _checked_command(run, arguments, cwd=cwd, label=label).stdout
    try:
        value = json.loads(output, object_pairs_hook=_json_object_with_unique_keys)
    except (ValueError, json.JSONDecodeError) as error:
        _fail(f"{label} returned malformed JSON: {error}")
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail(f"{label} returned a non-object JSON value")
    return value


def _graphql(
    run: CommandRunner,
    query: str,
    variables: Mapping[str, str],
    *,
    label: str,
) -> dict[str, object]:
    arguments = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        arguments.extend(("-f", f"{key}={value}"))
    return _command_json(run, arguments, label=label)


def _ensure_gh_auth(run: CommandRunner) -> None:
    _checked_command(
        run,
        ["gh", "auth", "status"],
        label="GitHub CLI authentication",
    )


def _required_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail(f"{label} is malformed")
    return value


def _required_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        _fail(f"{label} is malformed")
    return value


def _required_string(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        _fail(f"{label} is malformed")
    return value


def _required_node_id(value: object, label: str) -> str:
    node_id = _required_string(value, label)
    if re.fullmatch(_NODE_ID, node_id) is None:
        _fail(f"{label} is malformed")
    return node_id


def _page_info(value: object, label: str) -> tuple[bool, str | None]:
    page = _required_object(value, label)
    if set(page) != {"hasNextPage", "endCursor"}:
        _fail(f"{label} is malformed")
    has_next = page["hasNextPage"]
    cursor = page["endCursor"]
    if type(has_next) is not bool or (cursor is not None and not isinstance(cursor, str)):
        _fail(f"{label} is malformed")
    if has_next and not cursor:
        _fail(f"{label} is missing a pagination cursor")
    return has_next, cursor


def _read_proposal(checkout_root: Path, proposal_path: Path) -> str:
    candidate = proposal_path if proposal_path.is_absolute() else checkout_root / proposal_path
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        _fail(f"proposal file cannot be resolved: {proposal_path}: {error}")
    try:
        resolved.relative_to(checkout_root)
    except ValueError:
        _fail("proposal file must resolve inside the checkout")
    if not resolved.is_file():
        _fail(f"proposal path must be a regular file: {proposal_path}")
    try:
        return resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        _fail(f"proposal file cannot be read: {error}")


def _discussion_ref(value: object, label: str) -> DiscussionRef:
    node = _required_object(value, label)
    node_id = _required_node_id(node.get("id"), f"{label} identity")
    number = node.get("number")
    url = _required_string(node.get("url"), f"{label} URL")
    if type(number) is not int or number <= 0:
        _fail(f"{label} identity is malformed")
    expected_url = f"https://github.com/{PUBLIC_REPOSITORY}/discussions/{number}"
    if url != expected_url:
        _fail(f"{label} identity does not match {PUBLIC_REPOSITORY}")
    return DiscussionRef(node_id=node_id, number=number, url=url)


def _repository_and_category(run: CommandRunner) -> tuple[str, str]:
    after: str | None = None
    repository_id: str | None = None
    categories: list[str] = []
    seen_cursors: set[str] = set()
    while True:
        variables = {"owner": "RSI-Index", "name": "RSIs-First-Exam"}
        if after is not None:
            variables["after"] = after
        payload = _graphql(
            run,
            PUBLIC_REPOSITORY_QUERY,
            variables,
            label="Discussion category lookup",
        )
        data = _required_object(payload.get("data"), "Discussion category response")
        repository = _required_object(data.get("repository"), "source repository")
        page_repository_id = _required_node_id(repository.get("id"), "source repository ID")
        if repository_id is None:
            repository_id = page_repository_id
        elif page_repository_id != repository_id:
            _fail("source repository identity changed during pagination")
        connection = _required_object(
            repository.get("discussionCategories"), "Discussion categories"
        )
        for raw_node in _required_list(connection.get("nodes"), "Discussion categories"):
            node = _required_object(raw_node, "Discussion category")
            category_id = _required_node_id(node.get("id"), "Discussion category ID")
            name = _required_string(node.get("name"), "Discussion category name")
            if name == DISCUSSION_CATEGORY:
                categories.append(category_id)
        has_next, cursor = _page_info(
            connection.get("pageInfo"), "Discussion category pagination"
        )
        if not has_next:
            break
        assert cursor is not None
        if cursor in seen_cursors:
            _fail("Discussion category pagination repeated a cursor")
        seen_cursors.add(cursor)
        after = cursor
    if repository_id is None or len(categories) != 1:
        _fail(f"expected exactly one {DISCUSSION_CATEGORY} Discussion category")
    return repository_id, categories[0]


def create_discussion(
    checkout_root: Path,
    proposal_path: Path,
    title: str,
    *,
    run: CommandRunner = _run,
) -> DiscussionRef:
    if not isinstance(title, str) or not title.strip():
        _fail("Discussion title must be nonempty")
    root = _checkout_root(checkout_root)
    state_dir = _state_dir(root)
    with _discussion_lock(state_dir):
        if load_discussion(root, required=False) is not None:
            _fail("a Discussion already exists for this checkout")
        body = _read_proposal(root, proposal_path)
        _ensure_gh_auth(run)
        repository_id, category_id = _repository_and_category(run)
        payload = _graphql(
            run,
            CREATE_DISCUSSION,
            {
                "repositoryId": repository_id,
                "categoryId": category_id,
                "title": title,
                "body": body,
            },
            label="Discussion creation",
        )
        data = _required_object(payload.get("data"), "Discussion creation response")
        mutation = _required_object(data.get("createDiscussion"), "Discussion creation")
        ref = _discussion_ref(mutation.get("discussion"), "created Discussion")
        _atomic_json(
            state_dir / "discussion.json",
            {"node_id": ref.node_id, "number": ref.number, "url": ref.url},
        )
        return ref


def update_discussion(
    checkout_root: Path,
    proposal_path: Path,
    *,
    run: CommandRunner = _run,
) -> DiscussionRef:
    root = _checkout_root(checkout_root)
    discussion = load_discussion(root)
    body = _read_proposal(root, proposal_path)
    _ensure_gh_auth(run)
    payload = _graphql(
        run,
        UPDATE_DISCUSSION,
        {"id": discussion.node_id, "body": body},
        label="Discussion update",
    )
    data = _required_object(payload.get("data"), "Discussion update response")
    mutation = _required_object(data.get("updateDiscussion"), "Discussion update")
    ref = _discussion_ref(mutation.get("discussion"), "updated Discussion")
    if ref != discussion:
        _fail("updated Discussion identity does not match the bound Discussion")
    return ref


def _author(value: object, label: str) -> str:
    if value is None:
        return "[deleted]"
    author = _required_object(value, label)
    return _required_string(author.get("login"), label)


def _entry(value: object, *, parent_node_id: str | None = None) -> DiscussionEntry:
    node = _required_object(value, "Discussion entry")
    return DiscussionEntry(
        node_id=_required_node_id(node.get("id"), "Discussion entry ID"),
        author=_author(node.get("author"), "Discussion entry author"),
        body=_required_string(node.get("body"), "Discussion entry body", allow_empty=True),
        created_at=_required_string(node.get("createdAt"), "Discussion entry timestamp"),
        parent_node_id=parent_node_id,
    )


def _validate_snapshot_ref(node: Mapping[str, object], ref: DiscussionRef) -> None:
    if node.get("__typename") != "Discussion":
        _fail("Discussion query returned the wrong node type")
    returned = _discussion_ref(node, "fetched Discussion")
    if returned != ref:
        _fail("fetched Discussion identity does not match the bound Discussion")


def _fetch_discussion(
    ref: DiscussionRef,
    *,
    run: CommandRunner,
    authenticate: bool,
) -> DiscussionSnapshot:
    if authenticate:
        _ensure_gh_auth(run)
    after: str | None = None
    seen_comment_cursors: set[str] = set()
    seen_node_ids: set[str] = set()
    entries: list[DiscussionEntry] = []
    title: str | None = None
    body: str | None = None
    author: str | None = None
    created_at: str | None = None
    while True:
        variables = {"id": ref.node_id}
        if after is not None:
            variables["after"] = after
        payload = _graphql(
            run,
            DISCUSSION_QUERY,
            variables,
            label="Discussion fetch",
        )
        data = _required_object(payload.get("data"), "Discussion response")
        node = _required_object(data.get("node"), "Discussion")
        _validate_snapshot_ref(node, ref)
        page_title = _required_string(node.get("title"), "Discussion title")
        page_body = _required_string(node.get("body"), "Discussion body", allow_empty=True)
        page_author = _author(node.get("author"), "Discussion author")
        page_created_at = _required_string(node.get("createdAt"), "Discussion timestamp")
        header = (page_title, page_body, page_author, page_created_at)
        if title is None:
            title, body, author, created_at = header
        elif header != (title, body, author, created_at):
            _fail("Discussion changed while its comments were being fetched")
        comments = _required_object(node.get("comments"), "Discussion comments")
        for raw_comment in _required_list(comments.get("nodes"), "Discussion comments"):
            comment = _entry(raw_comment)
            if comment.node_id in seen_node_ids:
                _fail("Discussion query returned a duplicate entry")
            seen_node_ids.add(comment.node_id)
            entries.append(comment)
            comment_node = _required_object(raw_comment, "Discussion comment")
            replies = _required_object(comment_node.get("replies"), "Discussion replies")
            for raw_reply in _required_list(replies.get("nodes"), "Discussion replies"):
                reply = _entry(raw_reply, parent_node_id=comment.node_id)
                if reply.node_id in seen_node_ids:
                    _fail("Discussion query returned a duplicate entry")
                seen_node_ids.add(reply.node_id)
                entries.append(reply)
            replies_next, replies_cursor = _page_info(
                replies.get("pageInfo"), "Discussion reply pagination"
            )
            seen_reply_cursors: set[str] = set()
            while replies_next:
                assert replies_cursor is not None
                if replies_cursor in seen_reply_cursors:
                    _fail("Discussion reply pagination repeated a cursor")
                seen_reply_cursors.add(replies_cursor)
                reply_payload = _graphql(
                    run,
                    DISCUSSION_REPLIES_QUERY,
                    {"id": comment.node_id, "after": replies_cursor},
                    label="Discussion reply fetch",
                )
                reply_data = _required_object(
                    reply_payload.get("data"), "Discussion reply response"
                )
                reply_node = _required_object(reply_data.get("node"), "Discussion comment")
                if (
                    reply_node.get("__typename") != "DiscussionComment"
                    or reply_node.get("id") != comment.node_id
                ):
                    _fail("Discussion reply identity does not match its parent comment")
                reply_connection = _required_object(
                    reply_node.get("replies"), "Discussion replies"
                )
                for raw_reply in _required_list(
                    reply_connection.get("nodes"), "Discussion replies"
                ):
                    reply = _entry(raw_reply, parent_node_id=comment.node_id)
                    if reply.node_id in seen_node_ids:
                        _fail("Discussion query returned a duplicate entry")
                    seen_node_ids.add(reply.node_id)
                    entries.append(reply)
                replies_next, replies_cursor = _page_info(
                    reply_connection.get("pageInfo"), "Discussion reply pagination"
                )
        has_next, cursor = _page_info(
            comments.get("pageInfo"), "Discussion comment pagination"
        )
        if not has_next:
            break
        assert cursor is not None
        if cursor in seen_comment_cursors:
            _fail("Discussion comment pagination repeated a cursor")
        seen_comment_cursors.add(cursor)
        after = cursor
    assert title is not None and body is not None and author is not None and created_at is not None
    return DiscussionSnapshot(
        ref=ref,
        title=title,
        body=body,
        author=author,
        created_at=created_at,
        entries=tuple(entries),
    )


def fetch_discussion(
    ref: DiscussionRef, *, run: CommandRunner = _run
) -> DiscussionSnapshot:
    return _fetch_discussion(ref, run=run, authenticate=True)


def discussion_status(
    checkout_root: Path, *, run: CommandRunner = _run
) -> DiscussionSnapshot:
    return fetch_discussion(load_discussion(checkout_root), run=run)


def render_discussion_markdown(snapshot: DiscussionSnapshot) -> str:
    lines = [
        f"# {snapshot.title}",
        "",
        f"- **Author:** {snapshot.author}",
        f"- **Created:** {snapshot.created_at}",
        f"- **Discussion:** {snapshot.ref.url}",
        "",
        snapshot.body,
        "",
        "## Comments and replies",
        "",
    ]
    for entry in sorted(snapshot.entries, key=lambda item: (item.created_at, item.node_id)):
        prefix = "  " if entry.parent_node_id is not None else ""
        parent = (
            f" (reply to `{entry.parent_node_id}`)"
            if entry.parent_node_id is not None
            else ""
        )
        lines.append(f"{prefix}- **{entry.author}** — {entry.created_at}{parent}")
        body_prefix = prefix + "  "
        if entry.body:
            lines.extend(f"{body_prefix}{line}" for line in entry.body.splitlines())
        else:
            lines.append(body_prefix)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _publication_repository(snapshot: DiscussionSnapshot) -> tuple[str, str]:
    marker = re.compile(
        rf"<!-- rsi-task-bot:discussion={re.escape(snapshot.ref.node_id)};"
        rf"kind=publication;version=(?:0|[1-9][0-9]*)"
        rf"(?:;trigger={_NODE_ID})? -->\Z"
    )
    candidates: list[tuple[str, str]] = []
    scoped_without_url = False
    for entry in snapshot.entries:
        if entry.author not in _DISPATCHER_LOGINS:
            continue
        if entry.body.count("<!-- rsi-task-bot:") != 1 or marker.search(entry.body) is None:
            continue
        urls = _REPOSITORY_URL.findall(entry.body)
        if len(urls) != 1:
            scoped_without_url = True
            continue
        candidates.append((urls[0], f"https://github.com/{urls[0]}"))
    if len(candidates) != 1:
        if scoped_without_url and not candidates:
            _fail("publication comment must contain exactly one RSI-Index repository URL")
        _fail("exactly one verified publication comment is required")
    return candidates[0]


def _resolve_task_repository(
    snapshot: DiscussionSnapshot,
    *,
    run: CommandRunner,
    authenticate: bool,
) -> TaskRepository:
    if authenticate:
        _ensure_gh_auth(run)
    repository_name, repository_url = _publication_repository(snapshot)
    payload = _command_json(
        run,
        [
            "gh",
            "repo",
            "view",
            repository_name,
            "--json",
            "nameWithOwner,url,visibility,viewerPermission,defaultBranchRef",
        ],
        label="task repository lookup",
    )
    name_with_owner = _required_string(payload.get("nameWithOwner"), "task repository name")
    url = _required_string(payload.get("url"), "task repository URL")
    visibility = _required_string(payload.get("visibility"), "task repository visibility")
    viewer_permission = _required_string(
        payload.get("viewerPermission"), "task repository permission"
    )
    default_branch_ref = _required_object(
        payload.get("defaultBranchRef"), "task repository default branch"
    )
    default_branch = _required_string(
        default_branch_ref.get("name"), "task repository default branch"
    )
    if name_with_owner != repository_name or url != repository_url:
        _fail("task repository identity does not match the publication comment")
    if visibility != "PRIVATE":
        _fail("task repository must be private")
    if viewer_permission not in _WRITE_PERMISSIONS:
        _fail("GitHub identity does not have write access to the task repository")
    return TaskRepository(
        name_with_owner=name_with_owner,
        url=url,
        visibility=visibility,
        viewer_permission=viewer_permission,
        default_branch=default_branch,
    )


def resolve_task_repository(
    snapshot: DiscussionSnapshot, *, run: CommandRunner = _run
) -> TaskRepository:
    return _resolve_task_repository(snapshot, run=run, authenticate=True)


def _authenticated_login(run: CommandRunner) -> str:
    login = _checked_command(
        run,
        ["gh", "api", "user", "--jq", ".login"],
        label="GitHub identity lookup",
    ).stdout.strip()
    if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})", login) is None:
        _fail("GitHub identity lookup returned an invalid login")
    return login


def upload_trajectory(
    checkout_root: Path,
    *,
    platform: str,
    session_id: str,
    transcript_path: Path,
    run: CommandRunner = _run,
) -> UploadResult:
    discussion = load_discussion(checkout_root)
    if platform not in _SUPPORTED_PLATFORMS:
        _fail(f"unsupported platform: {platform}")
    if not isinstance(session_id, str) or not session_id.strip():
        _fail("native session ID must be nonempty")
    native_transcript = _resolve_transcript(transcript_path)
    _ensure_gh_auth(run)
    login = _authenticated_login(run)
    snapshot = _fetch_discussion(discussion, run=run, authenticate=False)
    repository = _resolve_task_repository(snapshot, run=run, authenticate=False)
    rendered = render_discussion_markdown(snapshot)
    transcript_name = f"{platform}-session.jsonl"
    with tempfile.TemporaryDirectory(prefix="rsi-proposal-upload-") as temporary:
        clone = Path(temporary) / "repository"
        _checked_command(
            run,
            ["gh", "repo", "clone", repository.name_with_owner, str(clone)],
            label="task repository clone",
        )
        if not clone.is_dir():
            _fail("task repository clone did not create a checkout")
        _checked_command(
            run,
            ["git", "checkout", repository.default_branch],
            cwd=clone,
            label="default branch checkout",
        )
        siblings = [
            item
            for item in clone.iterdir()
            if item.is_dir()
            and item.name.startswith("proposal-trajectory")
            and item.name != "proposal-trajectory"
        ]
        if siblings:
            _fail("task repository already contains a second trajectory directory")
        trajectory = clone / "proposal-trajectory"
        if trajectory.exists():
            if not trajectory.is_dir():
                _fail("proposal trajectory path is not a directory")
            shutil.rmtree(trajectory)
        trajectory.mkdir()
        transcript = trajectory / transcript_name
        try:
            shutil.copyfile(native_transcript, transcript)
            (trajectory / "discussion.md").write_text(rendered, encoding="utf-8")
            metadata = {
                "discussion_number": discussion.number,
                "discussion_url": discussion.url,
                "platform": platform,
                "session_id": session_id,
                "source_repository": PUBLIC_REPOSITORY,
                "target_repository": repository.name_with_owner,
                "uploaded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            _atomic_json(trajectory / "metadata.json", metadata)
        except OSError as error:
            _fail(f"cannot prepare proposal trajectory: {error}")
        _checked_command(
            run,
            ["git", "add", "--", "proposal-trajectory"],
            cwd=clone,
            label="proposal trajectory staging",
        )
        native_blob = _checked_command(
            run,
            ["git", "hash-object", "--no-filters", f"proposal-trajectory/{transcript_name}"],
            cwd=clone,
            label="native transcript verification",
        ).stdout.strip()
        staged_blob = _checked_command(
            run,
            ["git", "rev-parse", f":proposal-trajectory/{transcript_name}"],
            cwd=clone,
            label="staged transcript verification",
        ).stdout.strip()
        if not native_blob or native_blob != staged_blob:
            _fail("staged transcript bytes do not match the native transcript snapshot")
        _checked_command(
            run,
            [
                "git",
                "-c",
                f"user.name={login}",
                "-c",
                f"user.email={login}@users.noreply.github.com",
                "commit",
                "-m",
                "Upload proposal session trajectory",
                "--",
                "proposal-trajectory",
            ],
            cwd=clone,
            label="proposal trajectory commit",
        )
        commit_sha = _checked_command(
            run,
            ["git", "rev-parse", "HEAD"],
            cwd=clone,
            label="proposal trajectory commit lookup",
        ).stdout.strip()
        if not commit_sha:
            _fail("proposal trajectory commit lookup returned an empty commit")
        _checked_command(
            run,
            ["git", "push", "origin", repository.default_branch],
            cwd=clone,
            label="proposal trajectory push",
        )
    return UploadResult(
        repository=repository.name_with_owner,
        url=repository.url,
        commit_sha=commit_sha,
    )


def _discussion_cli(arguments: argparse.Namespace) -> int:
    checkout = Path(arguments.checkout)
    try:
        if arguments.discussion_command == "create":
            ref = create_discussion(
                checkout,
                Path(arguments.proposal),
                arguments.title,
            )
            print(f"created {ref.url}")
            return 0
        if arguments.discussion_command == "update":
            ref = update_discussion(checkout, Path(arguments.proposal))
            print(f"updated {ref.url}")
            return 0
        print(render_discussion_markdown(discussion_status(checkout)), end="")
        return 0
    except ProposalSessionError as error:
        print(f"proposal-session: {error}", file=sys.stderr)
        return 1


def _upload_cli(arguments: argparse.Namespace) -> int:
    try:
        result = upload_trajectory(
            Path(arguments.checkout),
            platform=arguments.platform,
            session_id=arguments.session_id,
            transcript_path=Path(arguments.transcript),
        )
        print(f"uploaded {result.repository} {result.commit_sha}")
        return 0
    except ProposalSessionError as error:
        print(f"proposal-session: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit proposals and upload original native session trajectories")
    commands = parser.add_subparsers(dest="command", required=True)

    discussion_parser = commands.add_parser("discussion")
    discussion_commands = discussion_parser.add_subparsers(
        dest="discussion_command", required=True
    )
    discussion_create = discussion_commands.add_parser("create")
    discussion_create.add_argument("--checkout", default=".")
    discussion_create.add_argument("--proposal", required=True)
    discussion_create.add_argument("--title", required=True)
    discussion_update = discussion_commands.add_parser("update")
    discussion_update.add_argument("--checkout", default=".")
    discussion_update.add_argument("--proposal", required=True)
    discussion_status = discussion_commands.add_parser("status")
    discussion_status.add_argument("--checkout", default=".")

    upload_parser = commands.add_parser("upload")
    upload_parser.add_argument("--checkout", default=".")
    upload_parser.add_argument("--platform", required=True, choices=sorted(_SUPPORTED_PLATFORMS))
    upload_parser.add_argument("--session-id", required=True)
    upload_parser.add_argument("--transcript", required=True, help="Original native session JSONL path")

    arguments = parser.parse_args(argv)
    if arguments.command == "discussion":
        return _discussion_cli(arguments)
    return _upload_cli(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
