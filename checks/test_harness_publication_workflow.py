import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/harness-publication-checks.yml"
REQUIRED = ("pyproject.toml", "uv.lock", "logs/runs/.gitkeep", "src/rsi_harness/task/compiler.py")


def test_publication_check_runs_on_harness_changes_without_private_secrets():
    text = WORKFLOW.read_text()
    workflow = yaml.load(text, Loader=yaml.BaseLoader)

    # Every PR and every push to main that changes the Harness, this workflow
    # or its test. Task and checker changes have their own workflow.
    watched = [
        "RSI-Harness/**",
        ".github/workflows/harness-publication-checks.yml",
        "checks/test_harness_publication_workflow.py",
    ]
    assert workflow["on"]["pull_request"]["paths"] == watched
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["on"]["push"]["paths"] == watched
    assert "pull_request_target" not in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "secrets." not in text
    assert "RSI-Skills" not in text
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-24.04"
    assert "persist-credentials: true" not in text
    assert text.count("persist-credentials: false") == 2
    automation = yaml.load(
        (WORKFLOW.parent / "discussion-automation-checks.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert ".github/workflows/harness-publication-checks.yml" in automation["on"]["pull_request"]["paths"]


def test_publisher_environment_and_archive_are_checked_before_test_dependencies():
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["publication-compatibility"]["steps"]
    named = {step["name"]: step for step in steps if "name" in step}
    archive = named["Archive Harness exactly as the task publisher does"]
    install = named["Install the publisher compiler environment"]
    compile_step = named["Compile CPU and GPU tasks without running them"]
    lint = named["Lint Harness"]
    tests = named["Test Harness without Docker or GPUs"]

    assert "git archive HEAD RSI-Harness" in archive["run"]
    for required in REQUIRED:
        assert required in archive["run"]
    assert install["run"] == "uv sync --frozen --no-dev --python 3.12"
    assert "for gpu_count in (0, 1)" in compile_step["run"]
    assert "HarborTaskCompiler().compile(" in compile_step["run"]
    assert "not integration and not gpu" in tests["run"]
    assert "ruff check ." in lint["run"]
    assert "ruff format --check ." in lint["run"]
    assert "--frozen" in lint["run"]
    assert (
        steps.index(archive)
        < steps.index(install)
        < steps.index(compile_step)
        < steps.index(lint)
        < steps.index(tests)
    )
    for step in (install, compile_step, lint, tests):
        assert step["working-directory"] == "${{ steps.archive.outputs.root }}/RSI-Harness"


@pytest.mark.parametrize("missing", (None, *REQUIRED))
def test_real_archive_step_rejects_missing_publication_files(tmp_path, missing):
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    archive = next(
        step for step in workflow["jobs"]["publication-compatibility"]["steps"]
        if step.get("id") == "archive"
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for relative in REQUIRED:
        if relative != missing:
            path = checkout / "RSI-Harness" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("publication fixture\n")
    environment = dict(
        os.environ,
        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="CI", GIT_AUTHOR_EMAIL="ci@example.invalid",
        GIT_COMMITTER_NAME="CI", GIT_COMMITTER_EMAIL="ci@example.invalid",
        RUNNER_TEMP=str(tmp_path), GITHUB_OUTPUT=str(tmp_path / "output"),
    )
    for args in (("init", "-q"), ("add", "."), ("commit", "-qm", "fixture")):
        subprocess.run(["git", *args], cwd=checkout, env=environment, check=True)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", archive["run"]],
        cwd=checkout, env=environment, capture_output=True, text=True,
    )
    if missing is None:
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "output").read_text().startswith("root=")
    else:
        assert result.returncode != 0
        assert f"Harness publication requires {missing}" in result.stdout
        assert not (tmp_path / "output").exists()


def test_every_published_task_compiles_with_the_proposed_harness():
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    job = workflow["jobs"]["published-tasks"]
    named = {step["name"]: step for step in job["steps"] if "name" in step}
    install = named["Install Harness compiler dependencies (no task dependencies)"]
    compile_step = named["Compile every published task without running it"]
    tests = named["Test publication checker without Docker, GPUs or models"]
    task_log = yaml.load(
        (WORKFLOW.parent / "task-log-publication-checks.yml").read_text(), Loader=yaml.BaseLoader
    )
    task_log_steps = {
        step["name"]: step for step in task_log["jobs"]["checker-tests"]["steps"] if "name" in step
    }

    assert job["timeout-minutes"] == "20"
    assert job["env"]["CUDA_VISIBLE_DEVICES"] == ""
    # The checked-out (proposed) Harness, not the base revision's.
    assert not any("ref" in step.get("with", {}) for step in job["steps"])
    assert install["working-directory"] == compile_step["working-directory"] == "RSI-Harness"
    assert install["run"] == "uv sync --frozen --no-dev --python 3.12"
    run = compile_step["run"]
    assert run.startswith("set -euo pipefail\n")
    assert "uv run --frozen --no-dev python -I ../checks/check_publication.py" in run
    assert '--repo "$GITHUB_WORKSPACE" --full --compile' in run
    # Compatibility only: no secret scan, no Docker.
    for forbidden in ("--gitleaks", "docker", "harbor run"):
        assert forbidden not in run
    # The publication checker's own tests run against the proposed compiler,
    # exactly as the task workflow runs them against an unchanged one.
    for name in (
        "Install checksum-pinned secret scanner",
        "Test publication checker without Docker, GPUs or models",
    ):
        assert named[name] == task_log_steps[name]
    assert "test_publication.py" in tests["run"]
    assert "test_publish_logs.py" in tests["run"]
    assert tests["env"]["PUBLICATION_REQUIRE_INTEGRATION"] == "1"
    steps = job["steps"]
    assert steps.index(install) < steps.index(compile_step) < steps.index(tests)
