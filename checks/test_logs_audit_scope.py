import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
WORKFLOWS = HERE.parent / ".github/workflows"
WORKFLOW = WORKFLOWS / "hf-logs-publication-checks.yml"
SPEC = importlib.util.spec_from_file_location("logs_audit_scope", HERE / "logs_audit_scope.py")
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)
ENV = dict(
    os.environ,
    GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
    GIT_AUTHOR_NAME="CI", GIT_AUTHOR_EMAIL="ci@example.invalid",
    GIT_COMMITTER_NAME="CI", GIT_COMMITTER_EMAIL="ci@example.invalid",
)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], env=ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


def write(repo, path, text="x\n"):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def commit(repo, message="change"):
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path):
    repo = tmp_path / "index"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    for task in ("rsi-tasks/alpha", "rsi-tasks/signature-tasks/beta"):
        write(repo, f"{task}/task.toml")
        write(repo, f"{task}/README.md")
    write(repo, "rsi-tasks/README.md")
    write(repo, "checks/check_publication.py")
    write(repo, "README.md")
    commit(repo, "base")
    return repo


def decide_after(repo, change):
    base = git(repo, "rev-parse", "HEAD")
    change(repo)
    return scope.decide(repo, base, commit(repo))


def test_task_content_that_keeps_the_task_list_skips(repo):
    def edit(repo):
        write(repo, "rsi-tasks/alpha/README.md", "new\n")
        write(repo, "rsi-tasks/signature-tasks/beta/task.toml", "new\n")
        write(repo, "rsi-tasks/README.md", "new\n")
        write(repo, "rsi-tasks/alpha/tests/test.sh")
        # Not watched by the workflow, so not a reason to audit.
        write(repo, "README.md", "new\n")

    assert decide_after(repo, edit) == (False, "the task list is unchanged")


@pytest.mark.parametrize(
    "change, added, removed",
    (
        (lambda r: write(r, "rsi-tasks/gamma/task.toml"), "rsi-tasks/gamma", "none"),
        (
            lambda r: write(r, "rsi-tasks/signature-tasks/delta/task.toml"),
            "rsi-tasks/signature-tasks/delta",
            "none",
        ),
        (lambda r: (r / "rsi-tasks/alpha/task.toml").unlink(), "none", "rsi-tasks/alpha"),
        (
            lambda r: shutil.rmtree(r / "rsi-tasks/signature-tasks/beta"),
            "none",
            "rsi-tasks/signature-tasks/beta",
        ),
        (
            lambda r: (r / "rsi-tasks/alpha").rename(r / "rsi-tasks/alpha-v2"),
            "rsi-tasks/alpha-v2",
            "rsi-tasks/alpha",
        ),
    ),
)
def test_added_removed_or_renamed_tasks_audit(repo, change, added, removed):
    audit, reason = decide_after(repo, change)
    assert audit
    assert reason == f"the task list changed (added: {added}; removed: {removed})"


@pytest.mark.parametrize("path", (*scope.ALWAYS, "checks/publication-policy.toml"))
def test_checker_policy_tool_scope_or_workflow_changes_always_audit(repo, path):
    def edit(repo):
        write(repo, path, "changed\n")
        write(repo, "rsi-tasks/alpha/README.md", "new\n")

    assert decide_after(repo, edit) == (True, f"{path} changed")


def test_a_stale_branch_is_compared_with_the_base_tip(repo):
    fork = git(repo, "rev-parse", "HEAD")
    write(repo, "rsi-tasks/gamma/task.toml")
    base = commit(repo, "main adds a task")
    git(repo, "checkout", "-q", "-b", "pr", fork)
    write(repo, "rsi-tasks/alpha/README.md", "new\n")
    head = commit(repo, "pr edits a README")
    audit, reason = scope.decide(repo, base, head)
    assert audit
    assert "removed: rsi-tasks/gamma" in reason


@pytest.mark.parametrize("base", ("", "0" * 40, "f" * 40, "no-such-ref"))
def test_an_unusable_base_audits(repo, base):
    assert scope.decide(repo, base, "HEAD") == (True, "no usable base revision")


def test_unrelated_history_audits(repo, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    git(other, "init", "-q", "-b", "main")
    write(other, "rsi-tasks/alpha/task.toml")
    write(other, "rsi-tasks/signature-tasks/beta/task.toml")
    commit(other)
    git(repo, "fetch", "-q", str(other), "main:unrelated")
    assert scope.decide(repo, "unrelated", "HEAD") == (True, "no usable base revision")


def test_task_roots_are_the_checkers():
    paths = [
        "rsi-tasks/a/task.toml",
        "rsi-tasks/signature-tasks/b/task.toml",
        "rsi-tasks/c/README.md",
        "other/d/task.toml",
    ]
    assert scope.publication.task_roots(paths) == {
        "rsi-tasks/a",
        "rsi-tasks/signature-tasks/b",
    }


def test_cli_writes_the_decision_and_escapes_task_names(repo, tmp_path):
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "rsi-tasks/x\n::error::y/task.toml")
    head = commit(repo)
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-I", str(HERE / "logs_audit_scope.py"), "--repo", str(repo),
         "--base", base, "--head", head, "--github-output", str(output)],
        capture_output=True, text=True, check=True,
    )
    assert output.read_text() == "audit=true\n"
    (line,) = result.stdout.splitlines()
    assert line.startswith("::notice::Running the Logs repository audit")
    assert "rsi-tasks/x%0A::error::y" in line


def _workflow():
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def _step(name):
    (step,) = [s for s in _workflow()["jobs"]["audit"]["steps"] if s.get("name") == name]
    return step


def test_always_matches_the_workflow_triggers():
    on = _workflow()["on"]
    for event in ("push", "pull_request"):
        assert sorted(on[event]["paths"]) == sorted((*scope.ALWAYS, "rsi-tasks/**"))


def test_the_gate_runs_the_trusted_script_and_gates_the_audit():
    workflow = _workflow()
    steps = workflow["jobs"]["audit"]["steps"]
    gate = _step("Skip the audit when the task list cannot change")
    assert gate["id"] == "scope"
    assert gate["env"] == {
        "BASE_REF": "${{ github.event.pull_request.base.sha || github.event.before }}",
        "HEAD_REF": "${{ github.event.pull_request.head.sha || github.sha }}",
    }
    assert 'scope="$GITHUB_WORKSPACE/trusted/checks/logs_audit_scope.py"' in gate["run"]
    assert '--repo "$GITHUB_WORKSPACE/index"' in gate["run"]
    index = next(s for s in steps if s.get("with", {}).get("path") == "index")
    assert index["with"]["fetch-depth"] == "0"
    assert steps.index(index) < steps.index(gate)
    for name in ("Install checksum-pinned secret scanner", "Locate the Logs repository"):
        assert _step(name)["if"] == "steps.scope.outputs.audit == 'true'"
    # Later steps already wait for the located repository.
    for name in ("Audit every published log (--full)", "Job summary"):
        assert "steps.locate.outputs.available == 'true'" in _step(name)["if"]
    automation = yaml.load(
        (WORKFLOWS / "discussion-automation-checks.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert ".github/workflows/hf-logs-publication-checks.yml" in automation["on"]["pull_request"]["paths"]


def run_gate(workspace, event, base, head):
    output = workspace / "output"
    output.write_text("")
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c",
         _step("Skip the audit when the task list cannot change")["run"]],
        cwd=workspace,
        env={**ENV, "GITHUB_WORKSPACE": str(workspace), "GITHUB_OUTPUT": str(output),
             "GITHUB_EVENT_NAME": event, "BASE_REF": base, "HEAD_REF": head},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout, output.read_text().splitlines()[-1]


def test_the_gate_step(repo):
    workspace = repo.parent
    trusted = workspace / "trusted/checks"
    trusted.mkdir(parents=True)
    base = git(repo, "rev-parse", "HEAD")
    write(repo, "rsi-tasks/alpha/README.md", "new\n")
    readme = commit(repo)
    write(repo, "rsi-tasks/gamma/task.toml")
    added = commit(repo)

    # A trusted revision without the script predates it: audit.
    stdout, decision = run_gate(workspace, "pull_request", base, readme)
    assert decision == "audit=true"
    assert "has no checks/logs_audit_scope.py" in stdout
    for name in ("logs_audit_scope.py", "check_publication.py"):
        shutil.copy(HERE / name, trusted / name)
    assert run_gate(workspace, "pull_request", base, readme)[1] == "audit=false"
    assert run_gate(workspace, "push", readme, added)[1] == "audit=true"
    assert run_gate(workspace, "push", "", readme)[1] == "audit=true"
    assert run_gate(workspace, "workflow_dispatch", base, readme)[1] == "audit=true"
    # A script that fails audits rather than failing the job.
    (trusted / "check_publication.py").write_text("raise SystemExit(3)\n")
    stdout, decision = run_gate(workspace, "pull_request", base, readme)
    assert decision == "audit=true"
    assert "Could not compare task lists" in stdout
