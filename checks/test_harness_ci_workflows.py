import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github/workflows"
AGENT_TESTS = WORKFLOWS / "harness-agent-tests.yml"
ROOT_CHECKS = WORKFLOWS / "harness-root-checks.yml"
PUBLICATION = WORKFLOWS / "harness-publication-checks.yml"
HARNESS = ROOT / "RSI-Harness"


def _load(path):
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def _registered_agents():
    factory = (HARNESS / "src/rsi_loop/harness/agent/factory.py").read_text()
    registry = factory[factory.index("_REGISTRY") :]
    registry = registry[: registry.index("}")]
    return sorted(re.findall(r'^\s*"([a-z0-9-]+)":', registry, re.MULTILINE))


@pytest.mark.parametrize("path", (AGENT_TESTS, ROOT_CHECKS))
def test_harness_docker_workflows_run_on_relevant_prs_without_secrets(path):
    text = path.read_text()
    workflow = _load(path)

    assert "pull_request_target" not in workflow["on"]
    assert workflow["on"]["pull_request"]["paths"]
    assert f".github/workflows/{path.name}" in workflow["on"]["pull_request"]["paths"]
    assert "workflow_dispatch" in workflow["on"]
    assert workflow["permissions"] == {"contents": "read"}
    assert "secrets." not in text
    for job in workflow["jobs"].values():
        assert job["runs-on"] == "ubuntu-latest"
    automation = _load(WORKFLOWS / "discussion-automation-checks.yml")
    assert f".github/workflows/{path.name}" in automation["on"]["pull_request"]["paths"]


@pytest.mark.parametrize("path", (AGENT_TESTS, ROOT_CHECKS))
def test_a_skipped_docker_check_fails_its_job(path):
    for job in _load(path)["jobs"].values():
        for step in job["steps"]:
            run = step.get("run", "")
            if "pytest" not in run:
                continue
            assert run.startswith("set -o pipefail\n")
            assert "| tee " in run
            assert "! grep -Eq '[0-9]+ skipped'" in run


def test_every_registered_agent_has_its_own_docker_job():
    workflow = _load(AGENT_TESTS)
    matrix = workflow["jobs"]["agent"]["strategy"]["matrix"]["agent"]
    paths = workflow["on"]["pull_request"]["paths"]

    assert sorted(matrix) == _registered_agents()
    assert "RSI-Harness/src/rsi_loop/harness/agent/**" in paths
    assert (HARNESS / "src/rsi_loop/harness/agent/versions.py").exists()
    run = next(
        step["run"]
        for step in workflow["jobs"]["agent"]["steps"]
        if "pytest" in step.get("run", "")
    )
    assert '-k "[${{ matrix.agent }}]"' in run


def test_root_checks_use_the_real_firewall_as_root():
    steps = _load(ROOT_CHECKS)["jobs"]["root"]["steps"]
    run = next(step["run"] for step in steps if "pytest" in step.get("run", ""))

    assert "sudo -E" in run
    assert "RSI_SANDBOX_ROOT_MODE=1" in run
    assert "RSI_REQUIRE_SANDBOX_INTEGRATION=1" in run


def test_unit_tests_run_on_the_agent_images_node():
    versions = (HARNESS / "src/rsi_loop/harness/agent/versions.py").read_text()
    node = re.search(r'^NODE_VERSION = "([0-9.]+)"$', versions, re.MULTILINE).group(1)
    steps = _load(PUBLICATION)["jobs"]["publication-compatibility"]["steps"]
    named = {step["name"]: step for step in steps if "name" in step}
    setup = named["Set up the node the Agent driver runs on"]

    assert setup["uses"].startswith("actions/setup-node@")
    assert setup["with"]["node-version"] == node
    assert steps.index(setup) < steps.index(named["Test Harness without Docker or GPUs"])
