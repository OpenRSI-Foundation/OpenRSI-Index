from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".agents/skills/proposal-agent/scripts/proposal_session.py"
SPEC = importlib.util.spec_from_file_location("proposal_session", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def init_git_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    return checkout


def write_transcript(path: Path, content: bytes = b'{"type":"message"}\n') -> Path:
    path.write_bytes(content)
    return path


def invoke(*args: str, input: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=input,
        capture_output=True,
        text=True,
        check=False,
    )


def test_claude_skill_alias_resolves_to_the_canonical_skill_directory() -> None:
    alias = ROOT / ".claude/skills"
    assert alias.is_symlink()
    assert alias.readlink() == Path("../.agents/skills")
    assert alias.resolve() == (ROOT / ".agents/skills").resolve()


def proposal_checkout(
    tmp_path: Path,
    *,
    platform: str = "codex",
    transcript: bytes = b'{"type":"message"}\n',
) -> tuple[Path, Path]:
    checkout = init_git_checkout(tmp_path)
    transcript_path = write_transcript(tmp_path / f"{platform}.jsonl", transcript)
    return checkout, transcript_path


def bind_discussion(checkout: Path) -> module.DiscussionRef:
    ref = module.DiscussionRef(
        node_id="D_1",
        number=41,
        url="https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
    )
    state_dir = module._state_dir(checkout)
    state_dir.mkdir(exist_ok=True)
    module._atomic_json(
        state_dir / "discussion.json",
        {"node_id": ref.node_id, "number": ref.number, "url": ref.url},
    )
    return ref


def test_legacy_discussion_survives_without_hooks_or_original_log(tmp_path):
    checkout = init_git_checkout(tmp_path)
    state = module._state_dir(checkout)
    state.mkdir()
    module._atomic_json(state / "binding.json", {
        "schema": 1, "platform": "codex", "session_id": "old-session",
        "transcript_path": "/missing/native.jsonl", "checkout_root": str(checkout),
        "discussion": {
            "node_id": "D_1", "number": 41,
            "url": "https://github.com/RSI-Index/RSI-Index-Public/discussions/41",
        },
    })

    ref = module.load_discussion(checkout)
    assert ref.url == "https://github.com/RSI-Index/RSIs-First-Exam/discussions/41"
    assert module.discussion_status(checkout, run=PublishingRunner()).ref == ref


def graphql_variables(arguments: Sequence[str]) -> dict[str, str]:
    assert list(arguments[:3]) == ["gh", "api", "graphql"]
    fields = list(arguments[3:])
    assert len(fields) % 2 == 0
    result: dict[str, str] = {}
    for index in range(0, len(fields), 2):
        assert fields[index] == "-f"
        key, value = fields[index + 1].split("=", 1)
        result[key] = value
    return result


def completed(
    arguments: Sequence[str],
    *,
    stdout: str = "",
    returncode: int = 0,
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(arguments), returncode, stdout, stderr)


class LifecycleRunner:
    def __init__(
        self,
        *,
        category: str = "Task Ideas",
        auth_returncode: int = 0,
        created_id: str = "D_1",
        updated_id: str = "D_1",
    ) -> None:
        self.category = category
        self.auth_returncode = auth_returncode
        self.created_id = created_id
        self.updated_id = updated_id
        self.calls: list[tuple[list[str], Path | None]] = []
        self.graphql_variables: dict[str, str] = {}

    def __call__(
        self, arguments: Sequence[str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        argv = list(arguments)
        self.calls.append((argv, cwd))
        if argv == ["gh", "auth", "status"]:
            return completed(argv, returncode=self.auth_returncode, stderr="not logged in")
        variables = graphql_variables(argv)
        query = variables.pop("query")
        self.graphql_variables = variables
        if "PublicRepository" in query:
            payload = {
                "data": {
                    "repository": {
                        "id": "R_1",
                        "discussionCategories": {
                            "nodes": [{"id": "DIC_1", "name": self.category}],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        },
                    }
                }
            }
        elif "CreateProposal" in query:
            payload = {
                "data": {
                    "createDiscussion": {
                        "discussion": {
                            "id": self.created_id,
                            "number": 41,
                            "url": "https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
                        }
                    }
                }
            }
        elif "UpdateProposal" in query:
            payload = {
                "data": {
                    "updateDiscussion": {
                        "discussion": {
                            "id": self.updated_id,
                            "number": 41,
                            "url": "https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
                        }
                    }
                }
            }
        else:
            raise AssertionError(f"unexpected GraphQL query: {query}")
        return completed(argv, stdout=json.dumps(payload))


def discussion_payload(
    *,
    publication_body: str | None = None,
    publication_author: str = "rsi-index-task-dispatcher[bot]",
) -> dict[str, object]:
    publication = publication_body or (
        "Task repository: https://github.com/RSI-Index/example-d41\n\n"
        "<!-- rsi-task-bot:discussion=D_1;kind=publication;version=2;trigger=DC_9 -->"
    )
    return {
        "data": {
            "node": {
                "__typename": "Discussion",
                "id": "D_1",
                "number": 41,
                "url": "https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
                "title": "Fixed proposal",
                "body": "initial body",
                "createdAt": "2026-09-01T10:00:00Z",
                "author": {"login": "contributor"},
                "comments": {
                    "nodes": [
                        {
                            "id": "DC_LATE",
                            "body": publication,
                            "createdAt": "2026-09-01T13:00:00Z",
                            "author": {"login": publication_author},
                            "replies": {
                                "nodes": [],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            },
                        },
                        {
                            "id": "DC_REVIEW",
                            "body": "review comment",
                            "createdAt": "2026-09-01T11:00:00Z",
                            "author": {"login": "reviewer"},
                            "replies": {
                                "nodes": [
                                    {
                                        "id": "DCR_FIRST",
                                        "body": "first reply",
                                        "createdAt": "2026-09-01T11:30:00Z",
                                        "author": {"login": "contributor"},
                                    }
                                ],
                                "pageInfo": {"hasNextPage": True, "endCursor": "reply-1"},
                            },
                        },
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                },
            }
        }
    }


class PublishingRunner:
    def __init__(
        self,
        *,
        visibility: str = "PRIVATE",
        viewer_permission: str = "WRITE",
        repository: str = "RSI-Index/example-d41",
        publication_body: str | None = None,
        publication_author: str = "rsi-index-task-dispatcher[bot]",
        second_directory: bool = False,
        existing_trajectory: bool = False,
        staged_matches: bool = True,
        paginate_comments: bool = False,
    ) -> None:
        self.visibility = visibility
        self.viewer_permission = viewer_permission
        self.repository = repository
        self.publication_body = publication_body
        self.publication_author = publication_author
        self.second_directory = second_directory
        self.existing_trajectory = existing_trajectory
        self.staged_matches = staged_matches
        self.paginate_comments = paginate_comments
        self.calls: list[tuple[list[str], Path | None]] = []
        self.pushed_tree: dict[str, bytes] = {}

    @property
    def pushed(self) -> bool:
        return any(arguments[:2] == ["git", "push"] for arguments, _ in self.calls)

    def __call__(
        self, arguments: Sequence[str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        argv = list(arguments)
        self.calls.append((argv, cwd))
        if argv == ["gh", "auth", "status"]:
            return completed(argv)
        if argv == ["gh", "api", "user", "--jq", ".login"]:
            return completed(argv, stdout="contributor\n")
        if argv[:3] == ["gh", "api", "graphql"]:
            variables = graphql_variables(argv)
            query = variables["query"]
            if "ProposalDiscussion" in query:
                payload = discussion_payload(
                    publication_body=self.publication_body,
                    publication_author=self.publication_author,
                )
                if self.paginate_comments:
                    node = payload["data"]["node"]
                    comments = node["comments"]
                    if "after" not in variables:
                        comments["nodes"] = [comments["nodes"][1]]
                        comments["pageInfo"] = {
                            "hasNextPage": True,
                            "endCursor": "comment-1",
                        }
                    else:
                        assert variables["after"] == "comment-1"
                        comments["nodes"] = [comments["nodes"][0]]
                return completed(
                    argv,
                    stdout=json.dumps(payload),
                )
            if "DiscussionReplies" in query:
                return completed(
                    argv,
                    stdout=json.dumps(
                        {
                            "data": {
                                "node": {
                                    "__typename": "DiscussionComment",
                                    "id": "DC_REVIEW",
                                    "replies": {
                                        "nodes": [
                                            {
                                                "id": "DCR_NESTED",
                                                "body": "nested reply",
                                                "createdAt": "2026-09-01T12:00:00Z",
                                                "author": {"login": "reviewer"},
                                            }
                                        ],
                                        "pageInfo": {
                                            "hasNextPage": False,
                                            "endCursor": None,
                                        },
                                    },
                                }
                            }
                        }
                    ),
                )
            raise AssertionError(f"unexpected GraphQL query: {query}")
        if argv[:3] == ["gh", "repo", "view"]:
            assert argv == [
                "gh",
                "repo",
                "view",
                self.repository,
                "--json",
                "nameWithOwner,url,visibility,viewerPermission,defaultBranchRef",
            ]
            return completed(
                argv,
                stdout=json.dumps(
                    {
                        "nameWithOwner": self.repository,
                        "url": f"https://github.com/{self.repository}",
                        "visibility": self.visibility,
                        "viewerPermission": self.viewer_permission,
                        "defaultBranchRef": {"name": "main"},
                    }
                ),
            )
        if argv[:3] == ["gh", "repo", "clone"]:
            assert argv[:4] == ["gh", "repo", "clone", self.repository]
            clone = Path(argv[4])
            clone.mkdir()
            if self.second_directory:
                (clone / "proposal-trajectory-copy").mkdir()
            if self.existing_trajectory:
                trajectory = clone / "proposal-trajectory"
                trajectory.mkdir()
                (trajectory / "codex-session.jsonl").write_bytes(b"old")
                (trajectory / "discussion.md").write_text("old", encoding="utf-8")
                (trajectory / "metadata.json").write_text("{}\n", encoding="utf-8")
            return completed(argv)
        if argv == ["git", "checkout", "main"]:
            return completed(argv)
        if argv == ["git", "add", "--", "proposal-trajectory"]:
            return completed(argv)
        if argv[:3] == ["git", "hash-object", "--no-filters"] and argv[3] in {
            "proposal-trajectory/codex-session.jsonl",
            "proposal-trajectory/claude-code-session.jsonl",
        }:
            return completed(argv, stdout="native-blob\n")
        if argv[:2] == ["git", "rev-parse"] and argv[2] in {
            ":proposal-trajectory/codex-session.jsonl",
            ":proposal-trajectory/claude-code-session.jsonl",
        }:
            value = "native-blob" if self.staged_matches else "transformed-blob"
            return completed(argv, stdout=f"{value}\n")
        if argv[:2] == ["git", "-c"]:
            assert argv == [
                "git",
                "-c",
                "user.name=contributor",
                "-c",
                "user.email=contributor@users.noreply.github.com",
                "commit",
                "-m",
                "Upload proposal session trajectory",
                "--",
                "proposal-trajectory",
            ]
            return completed(argv)
        if argv == ["git", "rev-parse", "HEAD"]:
            return completed(argv, stdout="abc123\n")
        if argv == ["git", "push", "origin", "main"]:
            assert cwd is not None
            self.pushed_tree = {
                str(path.relative_to(cwd)): path.read_bytes()
                for path in sorted((cwd / "proposal-trajectory").iterdir())
                if path.is_file()
            }
            return completed(argv)
        raise AssertionError(f"unexpected command: {argv}")


def test_create_discussion_uses_confirmed_file_and_persists_identity(
    tmp_path: Path,
) -> None:
    checkout, _ = proposal_checkout(tmp_path, platform="codex")
    proposal = checkout / "proposal.md"
    proposal.write_text("# Fixed proposal\n", encoding="utf-8")
    runner = LifecycleRunner()

    ref = module.create_discussion(checkout, proposal, "Fixed proposal", run=runner)

    assert ref == module.DiscussionRef(
        node_id="D_1",
        number=41,
        url="https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
    )
    assert module.load_discussion(checkout, required=False) == ref
    assert runner.calls[0] == (["gh", "auth", "status"], None)
    assert runner.calls[1][0] == [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={module.PUBLIC_REPOSITORY_QUERY}",
        "-f",
        "owner=RSI-Index",
        "-f",
        "name=RSIs-First-Exam",
    ]
    assert runner.calls[2][0] == [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={module.CREATE_DISCUSSION}",
        "-f",
        "repositoryId=R_1",
        "-f",
        "categoryId=DIC_1",
        "-f",
        "title=Fixed proposal",
        "-f",
        "body=# Fixed proposal\n",
    ]


def test_update_discussion_reuses_exact_bound_node(tmp_path: Path) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    bind_discussion(checkout)
    proposal = checkout / "proposal.md"
    proposal.write_text("# Revised proposal\n", encoding="utf-8")
    runner = LifecycleRunner()

    ref = module.update_discussion(checkout, proposal, run=runner)

    assert ref.node_id == "D_1"
    assert runner.graphql_variables == {"id": "D_1", "body": "# Revised proposal\n"}
    assert runner.calls[-1][0] == [
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={module.UPDATE_DISCUSSION}",
        "-f",
        "id=D_1",
        "-f",
        "body=# Revised proposal\n",
    ]


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (LifecycleRunner(auth_returncode=1), "authentication"),
        (LifecycleRunner(category="Announcements"), "Task Ideas"),
        (LifecycleRunner(created_id="bad id!"), "identity"),
    ],
)
def test_create_discussion_rejects_auth_category_and_identity_errors(
    tmp_path: Path, runner: LifecycleRunner, message: str
) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    proposal = checkout / "proposal.md"
    proposal.write_text("proposal\n", encoding="utf-8")

    with pytest.raises(module.ProposalSessionError, match=message):
        module.create_discussion(checkout, proposal, "Proposal", run=runner)

    assert module.load_discussion(checkout, required=False) is None


def test_update_requires_bound_discussion_and_exact_mutation_identity(tmp_path: Path) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    proposal = checkout / "proposal.md"
    proposal.write_text("proposal\n", encoding="utf-8")

    with pytest.raises(module.ProposalSessionError, match="created"):
        module.update_discussion(checkout, proposal, run=LifecycleRunner())

    bind_discussion(checkout)
    with pytest.raises(module.ProposalSessionError, match="identity"):
        module.update_discussion(
            checkout,
            proposal,
            run=LifecycleRunner(updated_id="D_OTHER"),
        )


def test_discussion_lifecycle_rejects_disappeared_proposal_file(tmp_path: Path) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    missing = checkout / "missing.md"
    runner = LifecycleRunner()

    with pytest.raises(module.ProposalSessionError, match="proposal"):
        module.create_discussion(checkout, missing, "Proposal", run=runner)
    assert runner.calls == []

    bind_discussion(checkout)
    with pytest.raises(module.ProposalSessionError, match="proposal"):
        module.update_discussion(checkout, missing, run=runner)
    assert runner.calls == []


def test_relative_proposal_resolves_from_checkout_not_process_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    (checkout / "proposal.md").write_text("# Checkout proposal\n", encoding="utf-8")
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    (other_cwd / "proposal.md").write_text("# Wrong proposal\n", encoding="utf-8")
    monkeypatch.chdir(other_cwd)
    runner = LifecycleRunner()

    module.create_discussion(checkout, Path("proposal.md"), "Proposal", run=runner)

    assert runner.graphql_variables["body"] == "# Checkout proposal\n"


def test_create_rejects_proposal_paths_resolving_outside_checkout(
    tmp_path: Path,
) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    symlink = checkout / "outside-link.md"
    symlink.symlink_to(outside)

    for proposal in (Path("../outside.md"), outside.resolve(), Path("outside-link.md")):
        runner = LifecycleRunner()
        with pytest.raises(module.ProposalSessionError, match="inside the checkout"):
            module.create_discussion(checkout, proposal, "Proposal", run=runner)
        assert runner.calls == []


def test_create_requires_proposal_to_be_a_regular_file(tmp_path: Path) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    (checkout / "proposal-dir").mkdir()
    runner = LifecycleRunner()

    with pytest.raises(module.ProposalSessionError, match="regular file"):
        module.create_discussion(checkout, Path("proposal-dir"), "Proposal", run=runner)
    assert runner.calls == []


def test_fetch_discussion_paginates_replies_and_renders_timestamp_order() -> None:
    ref = module.DiscussionRef(
        node_id="D_1",
        number=41,
        url="https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
    )
    runner = PublishingRunner(paginate_comments=True)

    snapshot = module.fetch_discussion(ref, run=runner)
    rendered = module.render_discussion_markdown(snapshot)

    assert snapshot.title == "Fixed proposal"
    assert len(snapshot.entries) == 4
    assert rendered.startswith("# Fixed proposal\n\n")
    assert "initial body" in rendered
    assert rendered.index("review comment") < rendered.index("first reply")
    assert rendered.index("first reply") < rendered.index("nested reply")
    assert rendered.index("nested reply") < rendered.index("Task repository:")
    assert "  - **contributor** — 2026-09-01T11:30:00Z" in rendered
    reply_calls = [
        arguments
        for arguments, _ in runner.calls
        if arguments[:3] == ["gh", "api", "graphql"]
        and "DiscussionReplies" in graphql_variables(arguments)["query"]
    ]
    assert len(reply_calls) == 1
    assert graphql_variables(reply_calls[0]) == {
        "query": module.DISCUSSION_REPLIES_QUERY,
        "id": "DC_REVIEW",
        "after": "reply-1",
    }
    comment_calls = [
        arguments
        for arguments, _ in runner.calls
        if arguments[:3] == ["gh", "api", "graphql"]
        and "ProposalDiscussion" in graphql_variables(arguments)["query"]
    ]
    assert len(comment_calls) == 2
    assert graphql_variables(comment_calls[1]) == {
        "query": module.DISCUSSION_QUERY,
        "id": "D_1",
        "after": "comment-1",
    }


def test_interleaved_replies_keep_global_order_and_name_their_parent() -> None:
    ref = module.DiscussionRef(
        node_id="D_1",
        number=41,
        url="https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
    )
    snapshot = module.DiscussionSnapshot(
        ref=ref,
        title="Interleaved",
        body="body",
        author="contributor",
        created_at="2026-09-01T10:00:00Z",
        entries=(
            module.DiscussionEntry(
                node_id="DC_PARENT",
                author="reviewer-a",
                body="first comment",
                created_at="2026-09-01T11:00:00Z",
            ),
            module.DiscussionEntry(
                node_id="DCR_PARENT",
                author="contributor",
                body="reply to first",
                created_at="2026-09-01T11:10:00Z",
                parent_node_id="DC_PARENT",
            ),
            module.DiscussionEntry(
                node_id="DC_INTERLEAVED",
                author="reviewer-b",
                body="interleaved comment",
                created_at="2026-09-01T11:05:00Z",
            ),
            module.DiscussionEntry(
                node_id="DCR_INTERLEAVED",
                author="contributor",
                body="reply to second",
                created_at="2026-09-01T11:15:00Z",
                parent_node_id="DC_INTERLEAVED",
            ),
        ),
    )

    rendered = module.render_discussion_markdown(snapshot)

    assert rendered.index("first comment") < rendered.index("interleaved comment")
    assert rendered.index("interleaved comment") < rendered.index("reply to first")
    assert rendered.index("reply to first") < rendered.index("reply to second")
    assert (
        "  - **contributor** — 2026-09-01T11:10:00Z "
        "(reply to `DC_PARENT`)"
    ) in rendered
    assert (
        "  - **contributor** — 2026-09-01T11:15:00Z "
        "(reply to `DC_INTERLEAVED`)"
    ) in rendered


def test_upload_preserves_native_bytes_and_complete_discussion(tmp_path: Path) -> None:
    native = b'not-a-stable-schema\x00\n{"tool":"call"}\n'
    checkout, _ = proposal_checkout(tmp_path, platform="codex", transcript=native)
    bind_discussion(checkout)
    runner = PublishingRunner()

    result = module.upload_trajectory(
        checkout, platform="codex", session_id="session_exact",
        transcript_path=checkout.parent / "codex.jsonl", run=runner,
    )

    assert runner.pushed_tree["proposal-trajectory/codex-session.jsonl"] == native
    assert b"initial body" in runner.pushed_tree["proposal-trajectory/discussion.md"]
    assert b"nested reply" in runner.pushed_tree["proposal-trajectory/discussion.md"]
    metadata = json.loads(
        runner.pushed_tree["proposal-trajectory/metadata.json"].decode("utf-8")
    )
    assert metadata == {
        "discussion_number": 41,
        "discussion_url": "https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
        "platform": "codex",
        "session_id": "session_exact",
        "source_repository": "RSI-Index/RSIs-First-Exam",
        "target_repository": "RSI-Index/example-d41",
        "uploaded_at": metadata["uploaded_at"],
    }
    assert "transcript" not in json.dumps(metadata)
    assert result.repository == "RSI-Index/example-d41"
    assert result.commit_sha == "abc123"


def test_upload_accepts_graphql_app_login_without_bot_suffix(tmp_path: Path) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    bind_discussion(checkout)
    runner = PublishingRunner(publication_author="rsi-index-task-dispatcher")

    result = module.upload_trajectory(
        checkout, platform="codex", session_id="session_exact",
        transcript_path=checkout.parent / "codex.jsonl", run=runner,
    )

    assert result.repository == "RSI-Index/example-d41"
    assert result.commit_sha == "abc123"
    assert runner.pushed


@pytest.mark.parametrize(
    ("runner", "message"),
    [
        (PublishingRunner(visibility="PUBLIC"), "private"),
        (PublishingRunner(viewer_permission="READ"), "write"),
        (
            PublishingRunner(
                publication_body=(
                    "No repository yet\n\n"
                    "<!-- rsi-task-bot:discussion=D_1;kind=publication;version=2 -->"
                )
            ),
            "repository URL",
        ),
        (
            PublishingRunner(
                publication_body=(
                    "Issue: https://github.com/RSI-Index/example-d41/issues/1\n\n"
                    "<!-- rsi-task-bot:discussion=D_1;kind=publication;version=2 -->"
                )
            ),
            "repository URL",
        ),
        (
            PublishingRunner(
                publication_body=(
                    "https://github.com/RSI-Index/example-d41\n\n"
                    "<!-- rsi-task-bot:discussion=D_OTHER;kind=publication;version=2 -->"
                )
            ),
            "publication",
        ),
        (PublishingRunner(second_directory=True), "trajectory directory"),
        (PublishingRunner(staged_matches=False), "staged transcript"),
    ],
)
def test_upload_rejects_unverified_targets_before_push(
    tmp_path: Path, runner: PublishingRunner, message: str
) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    bind_discussion(checkout)

    with pytest.raises(module.ProposalSessionError, match=message):
        module.upload_trajectory(
            checkout, platform="codex", session_id="session_exact",
            transcript_path=checkout.parent / "codex.jsonl", run=runner,
        )

    assert not runner.pushed


def test_upload_retry_replaces_the_three_fixed_paths_without_duplicates(
    tmp_path: Path,
) -> None:
    checkout, _ = proposal_checkout(tmp_path)
    bind_discussion(checkout)
    runner = PublishingRunner(existing_trajectory=True)

    module.upload_trajectory(
        checkout, platform="codex", session_id="session_exact",
        transcript_path=checkout.parent / "codex.jsonl", run=runner,
    )

    assert sorted(runner.pushed_tree) == [
        "proposal-trajectory/codex-session.jsonl",
        "proposal-trajectory/discussion.md",
        "proposal-trajectory/metadata.json",
    ]


def test_discussion_and_upload_cli_commands_are_registered() -> None:
    assert invoke("discussion", "--help").returncode == 0
    assert invoke("discussion", "create", "--help").returncode == 0
    assert invoke("discussion", "update", "--help").returncode == 0
    assert invoke("discussion", "status", "--help").returncode == 0
    assert invoke("upload", "--help").returncode == 0


def test_fresh_checkout_can_submit_without_session_setup(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    (checkout / "proposal.md").write_text("# Confirmed proposal\n", encoding="utf-8")

    ref = module.create_discussion(
        checkout, Path("proposal.md"), "Confirmed proposal", run=LifecycleRunner()
    )

    assert ref.number == 41
    assert module.discussion_status(checkout, run=PublishingRunner()).ref == ref


@pytest.mark.parametrize("platform", ["codex", "claude-code"])
def test_explicit_native_upload_without_hooks_from_other_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    checkout = init_git_checkout(tmp_path)
    state = module._state_dir(checkout)
    state.mkdir()
    (state / "discussion.json").write_text(json.dumps({
        "node_id": "D_1", "number": 41,
        "url": "https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
    }), encoding="utf-8")
    native = b'{"sessionId":"original-session","message":"full original text"}\n'
    transcript = write_transcript(tmp_path / "original.jsonl", native)
    monkeypatch.chdir(tmp_path)
    runner = PublishingRunner()

    result = module.upload_trajectory(
        checkout, platform=platform, session_id="original-session",
        transcript_path=transcript, run=runner,
    )

    assert result.commit_sha == "abc123"
    assert sorted(runner.pushed_tree) == sorted([
        f"proposal-trajectory/{platform}-session.jsonl",
        "proposal-trajectory/discussion.md", "proposal-trajectory/metadata.json",
    ])
    assert runner.pushed_tree[f"proposal-trajectory/{platform}-session.jsonl"] == native
    assert b"nested reply" in runner.pushed_tree["proposal-trajectory/discussion.md"]
    assert json.loads(runner.pushed_tree["proposal-trajectory/metadata.json"])["session_id"] == "original-session"


@pytest.mark.parametrize("platform", ["codex", "claude-code"])
def test_hookless_upload_pushes_original_bytes_through_real_git(
    tmp_path: Path, platform: str
) -> None:
    checkout, transcript = proposal_checkout(
        tmp_path, platform=platform, transcript=b'{"message":"original \\u4f60\\u597d"}\n'
    )
    bind_discussion(checkout)
    seed = tmp_path / "seed"
    remote = tmp_path / "task.git"

    def git(*args: str, cwd: Path | None = None):
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )

    git("init", "-q", "-b", "main", str(seed))
    (seed / "README.md").write_text("Existing task workspace\n", encoding="utf-8")
    git("add", "README.md", cwd=seed)
    git("-c", "user.name=Test", "-c", "user.email=test@example.com",
        "commit", "-qm", "Initial task", cwd=seed)
    git("clone", "--bare", str(seed), str(remote))
    github = PublishingRunner()

    def run(arguments, cwd=None):
        if list(arguments[:3]) == ["gh", "repo", "clone"]:
            return git("clone", str(remote), arguments[4])
        if arguments[0] == "git":
            return git(*arguments[1:], cwd=cwd)
        return github(arguments, cwd)

    result = module.upload_trajectory(
        checkout, platform=platform, session_id="original",
        transcript_path=transcript, run=run,
    )

    assert git("--git-dir", str(remote), "rev-parse", "main").stdout.strip() == result.commit_sha
    stored = subprocess.run(
        ["git", "--git-dir", str(remote), "show",
         f"main:proposal-trajectory/{platform}-session.jsonl"],
        capture_output=True, check=True,
    ).stdout
    assert stored == transcript.read_bytes()
    assert git("--git-dir", str(remote), "show", "main:README.md").stdout == "Existing task workspace\n"
