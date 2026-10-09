"""The acceptance's scripted Work agents in Work's shape, as a non-root user.

Each agent script (tests/acceptance/work) runs in a container whose only
mount is the read-only Work endpoint of a live broker (only the iptables
half is faked), with Harbor in its site-packages, ``rsi-sandbox`` on PATH as
production puts it and a stub ``rsi-submit``: the Work side the root
acceptance (scripts/operator/sandbox_acceptance.sh) relies on.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tests.acceptance.scripted_cli import FILES, bundle
from tests.integration.harbor_env_support import (
    live_sandbox as live_sandbox,
)
from tests.integration.test_harbor_in_judge_sample import (
    HOST_IMAGE,
    SAMPLE,
    TARGET,
    assert_only_host_left,
    harbor_host,
)

pytestmark = pytest.mark.integration

WORK = Path(__file__).parent / "work"
PATH = f"{TARGET}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
STUB_SUBMIT = b'#!/bin/sh\necho \'{"stub": "submitted"}\'\n'


def notes(output: str) -> dict[str, list[str]]:
    """``RSI-ACCEPTANCE <key> <value>`` lines of an agent's output."""
    found: dict[str, list[str]] = {}
    for line in output.splitlines():
        if line.startswith("RSI-ACCEPTANCE "):
            key, _, value = line.removeprefix("RSI-ACCEPTANCE ").partition(" ")
            found.setdefault(key, []).append(value)
    return found


def run_agent(sandbox, script: str, **environment) -> dict[str, list[str]]:
    endpoint = sandbox.work()
    with tempfile.TemporaryDirectory() as raw:
        archive = Path(raw) / "files.tar"
        bundle([SAMPLE / "tests", WORK], archive)
        data = archive.read_bytes()
    container = harbor_host(
        sandbox,
        endpoint,
        ["bash", "-c", (WORK / script).read_text()],
        {"RSI_ACCEPTANCE_FILES": str(FILES), "PATH": PATH, **environment},
        files={
            str(FILES): (data, 0o644),
            "/usr/local/bin/rsi-submit": (STUB_SUBMIT, 0o755),
        },
    )
    container.start()
    status = container.wait(timeout=1800)
    output = container.logs().decode(errors="replace")
    assert status["StatusCode"] == 0, output[-4000:]
    assert_only_host_left(sandbox, container)
    return notes(output)


def test_the_submit_agent_proves_the_work_half_of_a6(live_sandbox):
    sandbox = live_sandbox("work-a6", HOST_IMAGE, "alpine:3.21", build=True)

    found = run_agent(sandbox, "agent-submit.sh")

    [work] = found["a6-work"]
    report = json.loads(work)
    assert report["ok"], report
    assert report["sockets"] == [f"{TARGET}/s"]
    assert report["v1_containers"] == [400, "unsupported"]
    assert report["engine_paths"]["POST /v1/containers/json"] == 400
    assert report["engine_paths"]["GET /v1/containers/json"] == 400
    assert found["a6-run-step"] == ["pass"]
    assert found["a6-child-sockets"] == ["none"]
    assert found["submit-exit"] == ["0"]


def test_the_work_suites_agent_runs_the_judge_commands_in_work(live_sandbox):
    sandbox = live_sandbox("work-a3", HOST_IMAGE, "redis:7-alpine")

    found = run_agent(
        sandbox,
        "agent-work-suites.sh",
        RSI_HARBOR_SUITES="compose",
        RSI_HARBOR_CONCURRENCY="1",
    )

    assert found["work-reward"] == ['{"reward": 1.0}']
    assert found["work-suite"] == ["compose oracle=True nop=True"]
    assert found["submit-exit"] == ["0"]
