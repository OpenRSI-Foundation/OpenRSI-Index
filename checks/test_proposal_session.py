from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".agents/skills/proposal-agent/scripts/proposal_session.py"
CODEX_HOOKS = ROOT / ".codex/hooks.json"
CLAUDE_SETTINGS = ROOT / ".claude/settings.json"
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


def hook(
    session_id: str,
    transcript: Path,
    checkout: Path,
    *,
    event: str = "Stop",
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "transcript_path": str(transcript),
        "cwd": str(checkout),
        "hook_event_name": event,
    }


def invoke(*args: str, input: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        input=input,
        capture_output=True,
        text=True,
        check=False,
    )


def configured_hook(path: Path, event: str) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    handlers = config["hooks"][event]
    assert len(handlers) == 1
    commands = handlers[0]["hooks"]
    assert len(commands) == 1
    return commands[0]


def test_project_hooks_invoke_the_canonical_helper_for_each_client() -> None:
    root_command = (
        'python3 "$(git rev-parse --show-toplevel)/.agents/skills/proposal-agent/'
        'scripts/proposal_session.py" hook --platform codex --checkout '
        '"$(git rev-parse --show-toplevel)"'
    )
    for event in ("UserPromptSubmit", "Stop"):
        assert configured_hook(CODEX_HOOKS, event) == {
            "type": "command",
            "command": root_command,
        }
        assert configured_hook(CLAUDE_SETTINGS, event) == {
            "type": "command",
            "command": "python3",
            "args": [
                "${CLAUDE_PROJECT_DIR}/.agents/skills/proposal-agent/scripts/proposal_session.py",
                "hook",
                "--platform",
                "claude-code",
                "--checkout",
                "${CLAUDE_PROJECT_DIR}",
            ],
        }


def test_claude_skill_alias_resolves_to_the_canonical_skill_directory() -> None:
    alias = ROOT / ".claude/skills"
    assert alias.is_symlink()
    assert alias.readlink() == Path("../.agents/skills")
    assert alias.resolve() == (ROOT / ".agents/skills").resolve()


def activated_binding(
    tmp_path: Path,
    *,
    platform: str = "codex",
    transcript: bytes = b'{"type":"message"}\n',
) -> tuple[Path, Path]:
    checkout = init_git_checkout(tmp_path)
    transcript_path = write_transcript(tmp_path / f"{platform}.jsonl", transcript)
    module.activate(checkout)
    assert module.capture_hook(
        checkout,
        platform,
        hook("session_exact", transcript_path, checkout),
    )
    assert module.capture_hook(
        checkout,
        platform,
        hook(
            "session_exact",
            transcript_path,
            checkout,
            event="UserPromptSubmit",
        ),
    )
    return checkout, transcript_path


def bind_discussion(checkout: Path) -> module.DiscussionRef:
    binding = module.load_binding(checkout)
    ref = module.DiscussionRef(
        node_id="D_1",
        number=41,
        url="https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
    )
    state_dir = module._state_dir(binding.checkout_root)
    module._atomic_json(
        state_dir / "binding.json",
        module._binding_payload(replace(binding, discussion=ref)),
    )
    return ref


def test_session_bound_before_repository_rename_can_fetch_current_discussion(tmp_path):
    checkout, _ = activated_binding(tmp_path)
    current = bind_discussion(checkout)
    binding = module.load_binding(checkout)
    legacy = replace(current, url="https://github.com/RSI-Index/RSI-Index-Public/discussions/41")
    module._atomic_json(
        module._state_dir(checkout) / "binding.json",
        module._binding_payload(replace(binding, discussion=legacy)),
    )

    assert module.load_binding(checkout).discussion == current
    assert module.discussion_status(checkout, run=PublishingRunner()).ref == current


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
        if argv == [
            "git",
            "hash-object",
            "--no-filters",
            "proposal-trajectory/codex-session.jsonl",
        ]:
            return completed(argv, stdout="native-blob\n")
        if argv == [
            "git",
            "rev-parse",
            ":proposal-trajectory/codex-session.jsonl",
        ]:
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


def test_activate_then_codex_stop_binds_exact_session(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "codex.jsonl")

    state_dir = module.activate(checkout)
    assert state_dir.parent == Path(
        subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "--absolute-git-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert module.capture_hook(checkout, "codex", hook("thr_exact", transcript, checkout))

    binding = module.load_binding(checkout)
    assert binding.session_id == "thr_exact"
    assert binding.transcript_path == transcript.resolve()
    assert binding.checkout_root == checkout.resolve()
    assert not (checkout / ".rsi-proposal-session").exists()


def test_other_session_cannot_replace_binding(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    first = write_transcript(tmp_path / "first.jsonl")
    second = write_transcript(tmp_path / "second.jsonl")
    module.activate(checkout)

    assert module.capture_hook(checkout, "claude-code", hook("one", first, checkout))
    assert not module.capture_hook(checkout, "claude-code", hook("two", second, checkout))

    binding = module.load_binding(checkout)
    assert binding.session_id == "one"
    assert binding.transcript_path == first.resolve()


def test_exact_session_refreshes_its_transcript_path(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    first = write_transcript(tmp_path / "first.jsonl")
    refreshed = write_transcript(tmp_path / "refreshed.jsonl")
    module.activate(checkout)
    assert module.capture_hook(checkout, "codex", hook("thr_same", first, checkout))

    assert module.capture_hook(checkout, "codex", hook("thr_same", refreshed, checkout))
    assert module.load_binding(checkout).transcript_path == refreshed.resolve()


def test_capture_rejects_bad_platform_cwd_and_transcript(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "transcript.jsonl")
    outside = tmp_path / "outside"
    outside.mkdir()
    module.activate(checkout)

    with pytest.raises(module.ProposalSessionError, match="unsupported platform"):
        module.capture_hook(checkout, "other", hook("one", transcript, checkout))
    with pytest.raises(module.ProposalSessionError, match="outside"):
        module.capture_hook(checkout, "codex", hook("one", transcript, outside))
    with pytest.raises(module.ProposalSessionError, match="transcript"):
        module.capture_hook(
            checkout,
            "codex",
            hook("one", tmp_path / "missing.jsonl", checkout),
        )


def test_activate_rejects_a_non_git_directory(tmp_path: Path) -> None:
    with pytest.raises(module.ProposalSessionError, match="Git"):
        module.activate(tmp_path)


def test_hook_cli_ignores_missing_activation_and_other_session(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "transcript.jsonl")
    payload = json.dumps(hook("one", transcript, checkout))

    no_activation = invoke(
        "hook", "--platform", "codex", "--checkout", str(checkout), input=payload
    )
    assert no_activation.returncode == 0
    assert no_activation.stdout == no_activation.stderr == ""

    module.activate(checkout)
    assert module.capture_hook(checkout, "codex", hook("one", transcript, checkout))
    other_session = invoke(
        "hook",
        "--platform",
        "codex",
        "--checkout",
        str(checkout),
        input=json.dumps(hook("two", transcript, checkout)),
    )
    assert other_session.returncode == 0
    assert other_session.stdout == other_session.stderr == ""
    assert module.load_binding(checkout).session_id == "one"


def test_malformed_active_hook_is_diagnostic_and_status_stays_unbound(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    module.activate(checkout)

    malformed = invoke(
        "hook", "--platform", "claude-code", "--checkout", str(checkout), input="{not-json"
    )
    assert malformed.returncode == 0
    assert malformed.stdout == ""
    assert "proposal-session hook:" in malformed.stderr

    status = invoke("status", "--checkout", str(checkout))
    assert status.returncode == 0
    assert status.stdout.strip() == "unbound"
    with pytest.raises(module.ProposalSessionError, match="binding"):
        module.load_binding(checkout)


def test_non_stop_hook_event_is_a_silent_no_op(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    module.activate(checkout)

    result = invoke(
        "hook",
        "--platform",
        "codex",
        "--checkout",
        str(checkout),
        input=json.dumps({"hook_event_name": "Notification"}),
    )

    assert result.returncode == 0
    assert result.stdout == result.stderr == ""
    assert invoke("status", "--checkout", str(checkout)).stdout.strip() == "unbound"


def test_foreign_bound_session_with_invalid_paths_is_a_silent_no_op(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "bound.jsonl")
    module.activate(checkout)
    assert module.capture_hook(checkout, "codex", hook("bound", transcript, checkout))

    foreign = hook("foreign", tmp_path / "missing.jsonl", tmp_path / "outside")
    result = invoke(
        "hook",
        "--platform",
        "codex",
        "--checkout",
        str(checkout),
        input=json.dumps(foreign),
    )

    assert result.returncode == 0
    assert result.stdout == result.stderr == ""
    assert module.load_binding(checkout).session_id == "bound"


def test_concurrent_first_hooks_bind_exactly_one_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = init_git_checkout(tmp_path)
    first = write_transcript(tmp_path / "first.jsonl")
    second = write_transcript(tmp_path / "second.jsonl")
    module.activate(checkout)

    original_load = module._load_existing_binding
    load_count = 0
    count_lock = threading.Lock()
    both_loads_started = threading.Event()

    def synchronized_load(state_dir: Path, checkout_root: Path):
        nonlocal load_count
        binding = original_load(state_dir, checkout_root)
        with count_lock:
            load_count += 1
            if load_count == 2:
                both_loads_started.set()
        both_loads_started.wait(timeout=0.25)
        return binding

    monkeypatch.setattr(module, "_load_existing_binding", synchronized_load)
    start = threading.Barrier(3)
    results: list[bool | None] = [None, None]

    def capture(index: int, session_id: str, transcript: Path) -> None:
        start.wait()
        results[index] = module.capture_hook(
            checkout, "codex", hook(session_id, transcript, checkout)
        )

    threads = [
        threading.Thread(target=capture, args=(0, "first", first)),
        threading.Thread(target=capture, args=(1, "second", second)),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert sorted(results) == [False, True]
    assert module.load_binding(checkout).session_id in {"first", "second"}


def test_current_turn_attestation_serves_multiple_commands_until_stop(
    tmp_path: Path,
) -> None:
    checkout, transcript = activated_binding(tmp_path)
    bind_discussion(checkout)
    runner = PublishingRunner()

    assert module.discussion_status(checkout, run=runner).ref.node_id == "D_1"
    assert module.discussion_status(checkout, run=runner).ref.node_id == "D_1"

    assert module.capture_hook(
        checkout,
        "codex",
        hook("session_exact", transcript, checkout),
    )
    calls_after_stop = list(runner.calls)
    with pytest.raises(module.ProposalSessionError, match="original codex session"):
        module.discussion_status(checkout, run=runner)
    assert runner.calls == calls_after_stop


@pytest.mark.parametrize("command", ["status", "update", "upload"])
def test_foreign_session_cannot_run_lifecycle_commands(
    tmp_path: Path,
    command: str,
) -> None:
    checkout, _ = activated_binding(tmp_path)
    bind_discussion(checkout)
    proposal = checkout / "proposal.md"
    proposal.write_text("# Revised proposal\n", encoding="utf-8")
    foreign_transcript = write_transcript(tmp_path / "foreign.jsonl")

    assert module.capture_hook(
        checkout,
        "claude-code",
        hook(
            "foreign_session",
            foreign_transcript,
            checkout,
            event="UserPromptSubmit",
        ),
    )

    runner = LifecycleRunner() if command == "update" else PublishingRunner()
    with pytest.raises(
        module.ProposalSessionError,
        match="Resume the original codex session session_exact",
    ):
        if command == "status":
            module.discussion_status(checkout, run=runner)
        elif command == "update":
            module.update_discussion(checkout, proposal, run=runner)
        else:
            module.upload_trajectory(checkout, run=runner)
    assert runner.calls == []


def test_disabled_hooks_after_binding_cannot_reuse_prior_turn_proof(
    tmp_path: Path,
) -> None:
    checkout, transcript = activated_binding(tmp_path)
    bind_discussion(checkout)
    assert module.capture_hook(
        checkout,
        "codex",
        hook("session_exact", transcript, checkout),
    )

    runner = PublishingRunner()
    with pytest.raises(module.ProposalSessionError, match="hooks.*enabled"):
        module.upload_trajectory(checkout, run=runner)
    assert runner.calls == []


def test_discussion_create_requires_current_turn_proof(tmp_path: Path) -> None:
    checkout = init_git_checkout(tmp_path)
    transcript = write_transcript(tmp_path / "codex.jsonl")
    module.activate(checkout)
    assert module.capture_hook(
        checkout,
        "codex",
        hook("session_exact", transcript, checkout),
    )
    proposal = checkout / "proposal.md"
    proposal.write_text("# Proposal\n", encoding="utf-8")
    runner = LifecycleRunner()

    with pytest.raises(module.ProposalSessionError, match="original codex session"):
        module.create_discussion(checkout, proposal, "Proposal", run=runner)
    assert runner.calls == []


def test_malformed_prompt_hook_clears_prior_turn_proof_without_blocking(
    tmp_path: Path,
) -> None:
    checkout, _ = activated_binding(tmp_path)
    bind_discussion(checkout)
    malformed_prompt = {
        "hook_event_name": "UserPromptSubmit",
        "session_id": "session_exact",
        "cwd": str(checkout),
    }

    result = invoke(
        "hook",
        "--platform",
        "codex",
        "--checkout",
        str(checkout),
        input=json.dumps(malformed_prompt),
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert "proposal-session hook:" in result.stderr
    runner = PublishingRunner()
    with pytest.raises(module.ProposalSessionError, match="original codex session"):
        module.discussion_status(checkout, run=runner)
    assert runner.calls == []


def test_hook_payload_missing_event_clears_prior_turn_proof(
    tmp_path: Path,
) -> None:
    checkout, transcript = activated_binding(tmp_path)
    bind_discussion(checkout)
    missing_event = hook("session_exact", transcript, checkout)
    missing_event.pop("hook_event_name")

    result = invoke(
        "hook",
        "--platform",
        "codex",
        "--checkout",
        str(checkout),
        input=json.dumps(missing_event),
    )

    assert result.returncode == 0
    assert result.stdout == result.stderr == ""
    runner = PublishingRunner()
    with pytest.raises(module.ProposalSessionError, match="original codex session"):
        module.discussion_status(checkout, run=runner)
    assert runner.calls == []


def test_create_discussion_uses_confirmed_file_and_persists_identity(
    tmp_path: Path,
) -> None:
    checkout, _ = activated_binding(tmp_path, platform="codex")
    proposal = checkout / "proposal.md"
    proposal.write_text("# Fixed proposal\n", encoding="utf-8")
    runner = LifecycleRunner()

    ref = module.create_discussion(checkout, proposal, "Fixed proposal", run=runner)

    assert ref == module.DiscussionRef(
        node_id="D_1",
        number=41,
        url="https://github.com/RSI-Index/RSIs-First-Exam/discussions/41",
    )
    assert module.load_binding(checkout).discussion == ref
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
    checkout, _ = activated_binding(tmp_path)
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
    checkout, _ = activated_binding(tmp_path)
    proposal = checkout / "proposal.md"
    proposal.write_text("proposal\n", encoding="utf-8")

    with pytest.raises(module.ProposalSessionError, match=message):
        module.create_discussion(checkout, proposal, "Proposal", run=runner)

    assert module.load_binding(checkout).discussion is None


def test_update_requires_bound_discussion_and_exact_mutation_identity(tmp_path: Path) -> None:
    checkout, _ = activated_binding(tmp_path)
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
    checkout, _ = activated_binding(tmp_path)
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
    checkout, _ = activated_binding(tmp_path)
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
    checkout, _ = activated_binding(tmp_path)
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
    checkout, _ = activated_binding(tmp_path)
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
    checkout, _ = activated_binding(tmp_path, platform="codex", transcript=native)
    bind_discussion(checkout)
    runner = PublishingRunner()

    result = module.upload_trajectory(checkout, run=runner)

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
    checkout, _ = activated_binding(tmp_path)
    bind_discussion(checkout)
    runner = PublishingRunner(publication_author="rsi-index-task-dispatcher")

    result = module.upload_trajectory(checkout, run=runner)

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
    checkout, _ = activated_binding(tmp_path)
    bind_discussion(checkout)

    with pytest.raises(module.ProposalSessionError, match=message):
        module.upload_trajectory(checkout, run=runner)

    assert not runner.pushed


def test_upload_retry_replaces_the_three_fixed_paths_without_duplicates(
    tmp_path: Path,
) -> None:
    checkout, _ = activated_binding(tmp_path)
    bind_discussion(checkout)
    runner = PublishingRunner(existing_trajectory=True)

    module.upload_trajectory(checkout, run=runner)

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
