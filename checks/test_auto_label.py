"""PR area labels: the path table, the add/remove plan, and the API sync."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from checks import auto_label
from checks.auto_label import (
    LABELS,
    MANAGED,
    GitHub,
    GitHubError,
    labels_for,
    plan,
    sync_pull,
    targets,
)

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
WORKFLOW = WORKFLOWS / "auto-label-pr.yml"


def files(*paths, renamed_from=None):
    items = [{"filename": path, "previous_filename": None} for path in paths]
    if renamed_from:
        items[-1]["previous_filename"] = renamed_from
    return items


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("rsi-tasks/demo/task.toml", {"tasks"}),
        ("rsi-logs/demo/run/result.json", {"logs"}),
        ("RSI-Harness/src/rsi/cli.py", {"harness"}),
        (".github/workflows/discussion-review.yml", {"CI"}),
        ("checks/auto_label.py", {"CI"}),
        (".github/WORKFLOW_API.md", {"CI"}),
        ("README.md", {"docs"}),
        ("CONTRIBUTING.md", {"docs"}),
        ("LICENSE", {"docs"}),
        (".github/pull_request_template.md", {"docs"}),
        ("assets/logo.png", {"docs"}),
        (".agents/skills/proposal-agent/SKILL.md", {"skills"}),
        (".claude/skills", {"skills"}),
        (".codex/config.toml", {"skills"}),
        # Exact files and directory prefixes do not match look-alikes.
        ("rsi-tasks/demo/README.md", {"tasks"}),
        ("RSI-Harness/README.md", {"harness"}),
        ("rsi-tasks-old/x", set()),
        ("README.md.bak", set()),
        (".github/CODEOWNERS", set()),
        (".gitignore", set()),
    ],
)
def test_each_path_maps_to_its_area(path, expected):
    assert labels_for(files(path)) == expected


def test_mixed_pr_gets_every_touched_area():
    assert labels_for(
        files("rsi-tasks/a/task.toml", "rsi-logs/a/x.json", "checks/x.py", "README.md")
    ) == {"tasks", "logs", "CI", "docs"}


def test_rename_counts_both_the_new_and_previous_path():
    assert labels_for(
        files("rsi-tasks/new/task.toml", renamed_from="rsi-logs/old/task.toml")
    ) == {"tasks", "logs"}
    assert labels_for(files("archive/x.md", renamed_from="README.md")) == {"docs"}


def test_label_table_is_well_formed():
    names = [label.name for label in LABELS]
    assert names == ["tasks", "logs", "harness", "CI", "docs", "skills"]
    assert MANAGED == set(names)
    for label in LABELS:
        assert len(label.color) == 6 and int(label.color, 16) >= 0
        assert label.description and len(label.description) <= 100
        assert label.paths and all(not p.startswith("/") for p in label.paths)


def test_plan_only_touches_managed_labels():
    add, remove = plan({"tasks"}, ["CI", "bug", "documentation", "gpu"])
    assert (add, remove) == (["tasks"], ["CI"])


def test_later_push_that_drops_paths_removes_the_label():
    first = labels_for(files("rsi-tasks/a/task.toml", "checks/x.py"))
    assert plan(first, []) == (["CI", "tasks"], [])
    second = labels_for(files("rsi-tasks/a/task.toml"))
    assert plan(second, first) == ([], ["CI"])


def test_partial_file_list_never_removes():
    assert plan({"logs"}, ["tasks"], complete=False) == (["logs"], [])


class FakeGh:
    """Records gh calls and answers from in-memory repository state."""

    def __init__(self, *, pr_files, current, existing=(), changed_files=None, fail=None):
        self.pr_files, self.current = pr_files, list(current)
        self.existing = set(existing) | set(self.current)
        self.changed_files = len(pr_files) if changed_files is None else changed_files
        self.fail, self.calls = fail, []

    def __call__(self, args):
        self.calls.append(list(args))
        endpoint = next(a for a in args if a.startswith("repos/"))

        def reply(stdout="", code=0, stderr=""):
            return subprocess.CompletedProcess(args, code, stdout, stderr)

        if self.fail and self.fail in endpoint:
            return reply('{"message":"Server Error"}', 1, "gh: Server Error (HTTP 500)")
        if endpoint.endswith("/pulls/7"):
            return reply(json.dumps({"changed_files": self.changed_files}))
        if "/pulls/7/files" in endpoint:
            return reply("".join(json.dumps(item) + "\n" for item in self.pr_files))
        if "/issues/7/labels" in endpoint and "DELETE" in args:
            name = endpoint.rsplit("/", 1)[1].replace("%20", " ")
            if name not in self.current:
                return reply('{"message":"Label does not exist"}', 1, "gh: Not Found (HTTP 404)")
            self.current.remove(name)
            return reply("[]")
        if endpoint.endswith("/issues/7/labels") and "-f" in args:
            names = [a.split("=", 1)[1] for a in args if a.startswith("labels[]=")]
            missing = [n for n in names if n not in self.existing]
            if missing:
                return reply('{"message":"Validation Failed"}', 1, "gh: Validation Failed (HTTP 422)")
            self.current.extend(names)
            return reply("[]")
        if "/issues/7/labels" in endpoint:
            return reply("".join(json.dumps(n) + "\n" for n in self.current))
        if endpoint.endswith("/labels"):
            name = next(a for a in args if a.startswith("name=")).split("=", 1)[1]
            if name in self.existing:
                body = '{"message":"Validation Failed","errors":[{"resource":"Label","code":"already_exists","field":"name"}]}'
                return reply(body, 1, "gh: Validation Failed (HTTP 422)")
            self.existing.add(name)
            return reply("{}")
        raise AssertionError(f"unexpected gh call: {args}")


def test_sync_creates_missing_labels_and_adds_and_removes():
    gh = FakeGh(pr_files=files("rsi-tasks/a/task.toml"), current=["CI", "bug"], existing={"tasks"})
    assert sync_pull(GitHub("o/r", gh), 7) == (["tasks"], ["CI"])
    assert gh.current == ["bug", "tasks"]


def test_sync_creates_a_label_that_does_not_exist_yet():
    gh = FakeGh(pr_files=files(".codex/x"), current=[])
    sync_pull(GitHub("o/r", gh), 7)
    create = next(c for c in gh.calls if c[1] == "repos/o/r/labels")
    assert "name=skills" in create and "color=d4c5f9" in create
    assert gh.current == ["skills"]


def test_sync_is_a_no_op_when_labels_already_match():
    gh = FakeGh(pr_files=files("README.md"), current=["docs"])
    assert sync_pull(GitHub("o/r", gh), 7) == ([], [])
    assert all("DELETE" not in c and "-f" not in c for c in gh.calls)


def test_label_removed_meanwhile_is_not_an_error():
    gh = FakeGh(pr_files=files("README.md"), current=["CI"])

    def racing(args):
        if "DELETE" in args:
            gh.current.clear()
        return gh(args)

    assert sync_pull(GitHub("o/r", racing), 7) == (["docs"], ["CI"])


def test_truncated_file_list_only_adds(capsys):
    gh = FakeGh(pr_files=files("rsi-logs/a"), current=["tasks"], changed_files=5000)
    assert sync_pull(GitHub("o/r", gh), 7) == (["logs"], [])
    assert "::warning::" in capsys.readouterr().out


@pytest.mark.parametrize("fail", ["/pulls/7/files", "/issues/7/labels", "repos/o/r/labels"])
def test_api_failures_fail_the_run(fail):
    gh = FakeGh(pr_files=files(".claude/x"), current=["CI"], fail=fail)
    with pytest.raises(GitHubError):
        sync_pull(GitHub("o/r", gh), 7)


def workflow():
    # BaseLoader keeps `on` as a string key.
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_workflow_runs_on_every_pr_push_from_the_base_revision():
    parsed = workflow()
    trigger = parsed["on"]["pull_request_target"]
    assert trigger == {"types": ["opened", "synchronize", "reopened"]}
    assert "pr" in parsed["on"]["workflow_dispatch"]["inputs"]
    assert parsed["permissions"] == {
        "contents": "read",
        "pull-requests": "write",
        "issues": "write",
    }
    assert "github.event.pull_request.number" in parsed["concurrency"]["group"]
    steps = parsed["jobs"]["label"]["steps"]
    checkout = steps[0]
    assert checkout["uses"].startswith("actions/checkout@")
    assert len(checkout["uses"].split("@")[1]) == 40
    assert "ref" not in checkout["with"]
    assert checkout["with"]["persist-credentials"] == "false"
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "head.sha" not in text and "head.ref" not in text
    assert "|| true" not in text
    for step in steps:
        assert "${{" not in step.get("run", "")


def test_only_one_pr_labeler_remains():
    assert sorted(p.name for p in WORKFLOWS.glob("auto-label-*.yml")) == ["auto-label-pr.yml"]


def test_automation_checks_run_when_label_workflows_change():
    automation = yaml.load(
        (WORKFLOWS / "discussion-automation-checks.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    for event in ("push", "pull_request"):
        assert ".github/workflows/auto-label-*.yml" in automation["on"][event]["paths"]
        assert "checks/**" in automation["on"][event]["paths"]


def test_dispatch_target_is_a_pr_number_or_open():
    gh = FakeGh(pr_files=[], current=[])
    github = GitHub("o/r", gh)
    assert targets(github, " 12 ") == [12]
    for bad in ("", "0", "12;rm", "all", "-1"):
        with pytest.raises(ValueError):
            targets(github, bad)

    def open_pulls(args):
        assert args[:2] == ["api", "--paginate"] and "pulls?state=open" in args[2]
        return subprocess.CompletedProcess(args, 0, "3\n5\n", "")

    assert targets(GitHub("o/r", open_pulls), "open") == [3, 5]


def test_one_failing_pr_fails_the_backfill_after_trying_the_rest(monkeypatch, capsys):
    seen = []

    def fake_sync(github, number):
        seen.append(number)
        if number == 3:
            raise GitHubError("boom")

    monkeypatch.setattr(auto_label, "sync_pull", fake_sync)
    monkeypatch.setattr(auto_label, "targets", lambda github, value: [3, 5])
    assert auto_label.main(["--repo", "o/r", "open"]) == 1
    assert seen == [3, 5]
    assert "::error::PR #3" in capsys.readouterr().out
