"""Run the real gpu-label step against a fake `gh`: a label problem never fails the job."""

import base64
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/auto-label-gpu.yml"
GPU_TASK = b"[environment]\ngpus = 1\n"
CPU_TASK = b"[environment]\ngpus = 0\n"

FAKE_GH = r"""
import os, sys
from pathlib import Path
state = Path(os.environ["FAKE_STATE"])
args = sys.argv[1:]
with (state / "calls").open("a") as log:
    log.write(" ".join(args) + "\n")
label = state / "label-exists"
if args[:2] == ["api", "--paginate"]:
    print("rsi-tasks/demo/task.toml")
elif args[0] == "api" and "/contents/" in args[1]:
    print(os.environ["FAKE_TASK_B64"])
elif args[0] == "api" and args[1].endswith("/labels") and "/issues/" in args[1]:
    print(os.environ["FAKE_HAS_LABEL"])
elif args[0] == "api" and "/labels/" in args[1]:
    sys.exit(0 if label.exists() else 1)
elif args[:2] == ["label", "create"]:
    if os.environ["FAKE_CAN_CREATE"] != "1":
        sys.exit(1)
    label.touch()
elif args[:2] == ["pr", "edit"]:
    if "--add-label" in args and not label.exists():
        print("'gpu' not found", file=sys.stderr)
        sys.exit(1)
    if os.environ.get("FAKE_EDIT_FAILS") == "1":
        sys.exit(1)
else:
    sys.exit(f"unexpected gh call: {args}")
"""


def run_step(tmp_path, *, task, has_label, label_exists, can_create, edit_fails=False):
    step = yaml.safe_load(WORKFLOW.read_text())["jobs"]["label"]["steps"][0]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    gh = bindir / "gh"
    gh.write_text(f"#!{sys.executable}\n{FAKE_GH}")
    gh.chmod(0o755)
    if label_exists:
        (tmp_path / "label-exists").touch()
    env = dict(
        os.environ,
        PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}",
        FAKE_STATE=str(tmp_path),
        FAKE_TASK_B64=base64.b64encode(task).decode(),
        FAKE_HAS_LABEL="true" if has_label else "false",
        FAKE_CAN_CREATE="1" if can_create else "0",
        FAKE_EDIT_FAILS="1" if edit_fails else "0",
        GH_TOKEN="unused",
        REPO="owner/repo",
        PR="7",
        HEAD_SHA="0" * 40,
        LABEL=step["env"]["LABEL"],
    )
    result = subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", step["run"]],
        env=env,
        capture_output=True,
        text=True,
    )
    calls = (tmp_path / "calls").read_text().splitlines()
    return result, calls


def test_missing_label_is_created_then_applied(tmp_path):
    result, calls = run_step(
        tmp_path, task=GPU_TASK, has_label=False, label_exists=False, can_create=True
    )
    assert result.returncode == 0, result.stderr
    assert any(c.startswith("label create gpu") for c in calls)
    assert any("--add-label gpu" in c for c in calls)
    assert "::notice::" not in result.stdout


def test_existing_label_is_not_recreated(tmp_path):
    result, calls = run_step(
        tmp_path, task=GPU_TASK, has_label=False, label_exists=True, can_create=False
    )
    assert result.returncode == 0, result.stderr
    assert not any(c.startswith("label create") for c in calls)
    assert any("--add-label gpu" in c for c in calls)


@pytest.mark.parametrize("edit_fails", (False, True))
def test_label_that_cannot_be_created_or_applied_is_a_notice(tmp_path, edit_fails):
    result, _ = run_step(
        tmp_path,
        task=GPU_TASK,
        has_label=False,
        label_exists=edit_fails,
        can_create=False,
        edit_fails=edit_fails,
    )
    assert result.returncode == 0, result.stderr
    assert "::notice::Could not add the 'gpu' label" in result.stdout


def test_failed_label_removal_is_a_notice(tmp_path):
    result, calls = run_step(
        tmp_path,
        task=CPU_TASK,
        has_label=True,
        label_exists=True,
        can_create=False,
        edit_fails=True,
    )
    assert result.returncode == 0, result.stderr
    assert any("--remove-label gpu" in c for c in calls)
    assert "::notice::Could not remove the 'gpu' label" in result.stdout


def test_automation_checks_run_when_label_workflows_change():
    automation = yaml.safe_load(
        (WORKFLOW.parent / "discussion-automation-checks.yml").read_text()
    )
    for event in ("push", "pull_request"):
        # PyYAML reads the bare `on:` key as boolean True.
        assert ".github/workflows/auto-label-*.yml" in automation[True][event]["paths"]
