"""The scripted Work agent every acceptance scenario runs through
(tests/acceptance/scripted_cli.py), against the real codex adapter with a
recording Agent runtime: only the model CLI and its install change."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path, PurePosixPath

import pytest

import rsi_harness.runtime.production as production
from rsi_harness.models import (
    AgentHookRequest,
    AgentPrepareRequest,
    AgentRunRequest,
    ContainerRef,
)
from rsi_loop.harness.agent.codex import CodexAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.acceptance import scripted_cli
from tests.factories import make_run_plan
from tests.integrations.test_rsi_loop import RecordingAgentRuntime

WORK = ContainerRef(container_id="work-1", role="work")


class ArchiveRecordingRuntime(RecordingAgentRuntime):
    """Also keeps what each copied file held (the scripted agent's tar lives
    in a temporary directory only while it is copied)."""

    def __init__(self) -> None:
        super().__init__()
        self.contents: dict[PurePosixPath, bytes] = {}

    def copy_to(self, container, source, target) -> None:
        super().copy_to(container, source, target)
        self.contents[target] = Path(source).read_bytes()


@pytest.fixture
def swapped():
    """Undo the process-wide swap install_scripted_agent makes."""
    adapter = production.RSILoopAgentAdapter
    installs = CodexAgent.__dict__.get("install_cmds")
    try:
        yield
    finally:
        production.RSILoopAgentAdapter = adapter
        CodexAgent.install_cmds = installs


def test_the_scripted_agent_replaces_only_the_model_cli(tmp_path, swapped):
    files = tmp_path / "work"
    files.mkdir()
    (files / "lib.sh").write_text("note() { :; }\n")
    real = production.RSILoopAgentAdapter

    scripted_cli.install_scripted_agent(
        "echo scripted", [files], {"WORK_TIMEOUT_SEC": "300"}
    )

    assert CodexAgent.install_cmds == [scripted_cli.STUB_LAUNCHER]
    assert issubclass(production.RSILoopAgentAdapter, real)
    runtime = ArchiveRecordingRuntime()
    adapter = production.RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    plan = make_run_plan(tmp_path)
    prepared = adapter.prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=(tmp_path / "p.md").resolve())
    )
    assert prepared.command == ("/bin/bash", "-c", "echo scripted")
    environment = dict(prepared.environment)
    assert environment["WORK_TIMEOUT_SEC"] == "300"
    assert environment["RSI_ACCEPTANCE_FILES"] == str(scripted_cli.FILES)

    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan,
            container=WORK,
            submit_url="http://control.internal:8123",
            token="runtime-only-token",
        )
    )
    adapter.run(AgentRunRequest(prepared=prepared, container=WORK))

    with tarfile.open(fileobj=io.BytesIO(runtime.contents[scripted_cli.FILES])) as tar:
        assert sorted(tar.getnames()) == ["work", "work/lib.sh"]
        assert tar.getmember("work").isdir()
    # The production run path follows: prompt copied, the script executed
    # with the adapter's control environment added.
    execution = runtime.executions[-1]
    assert execution["command"] == prepared.command
    assert execution["environment"]["RSI_ACCEPTANCE_FILES"] == str(scripted_cli.FILES)
    assert execution["environment"]["RSI_TOKEN"] == "runtime-only-token"


def test_without_files_nothing_is_copied_for_the_script(tmp_path, swapped):
    scripted_cli.install_scripted_agent("true", [], {})
    runtime = ArchiveRecordingRuntime()
    adapter = production.RSILoopAgentAdapter(RSILoopConfig(), runtime=runtime)
    plan = make_run_plan(tmp_path)
    prepared = adapter.prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=(tmp_path / "p.md").resolve())
    )
    adapter.install_hooks(
        AgentHookRequest(
            run_plan=plan, container=WORK, submit_url="http://c:1", token="t"
        )
    )
    adapter.run(AgentRunRequest(prepared=prepared, container=WORK))

    assert "RSI_ACCEPTANCE_FILES" not in dict(prepared.environment)
    assert scripted_cli.FILES not in runtime.contents
