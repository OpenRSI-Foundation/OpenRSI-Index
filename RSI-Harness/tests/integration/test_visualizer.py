from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rsi_harness.models import (
    CompileOptions,
    GPUAllocation,
    GPUDevice,
    ImagePlan,
    JudgeGPUMode,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop.harness import judge_server as judge_server_module
from rsi_loop.harness import run_agent as run_agent_module
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.backend.k8s_backend import _sanitize_k8s_name
from rsi_loop.harness.config import RSILoopConfig
from rsi_loop.visualizer import scanner as scanner_module
from rsi_loop.visualizer.parsers.agent_output import parse as agent_output_parse
from rsi_loop.visualizer.scanner import RunsIndex
from rsi_loop.visualizer.server import create_app

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tasks" / "minimal-gpu"


def _write_visualizer_run(tmp_path: Path) -> tuple[Path, Path]:
    compiler = HarborTaskCompiler()
    definition = compiler.compile(
        FIXTURE,
        CompileOptions(agent_name="codex", primary_reward="reward"),
    )
    logs = (tmp_path / "logs").resolve()
    data = (tmp_path / "data").resolve()
    run_root = data / "run-two-rounds"
    plan = compiler.finalize(
        definition,
        ImagePlan(
            base_ref="base@sha256:abc",
            work_ref="work@sha256:def",
            judge_ref="base@sha256:abc",
            workdir=PurePosixPath("/workspace"),
            rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            base_digest="sha256:abc",
            work_digest="sha256:def",
            judge_digest="sha256:abc",
        ),
        RunGPUPlan(
            authorized_pool=GPUAllocation(
                devices=(GPUDevice(index=0, uuid="GPU-test", name="Test GPU"),)
            ),
            work=GPUAllocation(
                devices=(GPUDevice(index=0, uuid="GPU-test", name="Test GPU"),)
            ),
            judge=GPUAllocation(),
            judge_mode=JudgeGPUMode.FREEZE_ONLY,
        ),
        RunPaths(root=run_root, workspace=run_root / "workspace", logs=logs),
        "docker-rootfs",
    )
    writer = RunArtifactWriter(plan, run_id="run-two-rounds")
    writer.start()
    writer.record_submission(
        SubmissionReport(
            round_id="agent-1",
            status=SubmissionStatus.COMPLETED,
            rewards={"reward": 0, "answer_length": 5},
            score=0,
            output="expected the two-character answer 42\n",
            exit_code=0,
        )
    )
    writer.record_submission(
        SubmissionReport(
            round_id="agent-2",
            status=SubmissionStatus.COMPLETED,
            rewards={"reward": 1, "answer_length": 2},
            score=1,
            output="answer accepted\n",
            exit_code=0,
        )
    )
    writer.finalize(status=RunStatus.COMPLETED, runtime_seconds=3.5)
    writer.root.joinpath("agent_output.txt").write_text(
        "OpenAI Codex vtest\n--------\n"
        "workdir: /workspace\nmodel: gpt-test\n--------\n"
        "user\nSolve the task.\ncodex\nI corrected answer.txt after feedback.\n"
    )
    generated = data / "generated-tasks"
    staging = data / "metadata-staging"
    metadata = compiler.write_rsi_loop_metadata(definition, staging)
    generated.mkdir(parents=True)
    (generated / f"{definition.task_id}.json").write_text(metadata.read_text())
    return logs / "runs", generated


def test_stock_visualizer_loads_run_trajectory_and_two_archive_free_rounds(
    tmp_path: Path,
) -> None:
    runs_dir, tasks_dir = _write_visualizer_run(tmp_path)
    index = RunsIndex(
        runs_dir,
        task_meta={
            "minimal-gpu": {
                "score_direction": "maximize",
                "selection": "score_first",
                "is_score_task": True,
            }
        },
    )

    runs = index.list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.run_id == "run-two-rounds"
    assert run.task == "minimal-gpu"
    assert run.total_rounds == 2
    assert run.best_round == "agent-2"
    assert run.best_score == 1
    assert [submission.round_label for submission in run.submissions] == [
        "agent-1",
        "agent-2",
    ]

    client = TestClient(create_app(runs_dir, tasks_dir=tasks_dir))
    index = client.get("/")
    assert index.status_code == 200
    assert "RSI Loop Visualizer" in index.text
    navbar_brand = re.search(r'<a href="/"[^>]*>(.*?)</a>', index.text, re.DOTALL)
    assert navbar_brand is not None
    assert " ".join(re.sub(r"<[^>]+>", " ", navbar_brand.group(1)).split()) == (
        "RSI Loop Visualizer"
    )
    assert "SForge Visualizer" not in index.text
    assert "window.RSILoopScoreView" in index.text
    assert "window.SForgeScoreView" not in index.text
    detail = client.get("/run/run-two-rounds/minimal-gpu")
    trajectory = client.get("/run/run-two-rounds/minimal-gpu/trajectory")
    first = client.get("/run/run-two-rounds/minimal-gpu/submission/agent-1/raw")
    second = client.get("/run/run-two-rounds/minimal-gpu/submission/agent-2/raw")
    first_detail = client.get("/run/run-two-rounds/minimal-gpu/submission/agent-1")
    second_detail = client.get("/run/run-two-rounds/minimal-gpu/submission/agent-2")
    archive = client.get("/run/run-two-rounds/minimal-gpu/submission/agent-2/archive")

    assert detail.status_code == 200
    assert "agent-1" in detail.text and "agent-2" in detail.text
    assert trajectory.status_code == 200
    assert "I corrected answer.txt" in trajectory.text
    assert first_detail.status_code == 200
    assert re.search(r">Score</div><div[^>]*>\s*0\s*</div>", first_detail.text)
    assert second_detail.status_code == 200
    assert re.search(r">Score</div><div[^>]*>\s*1\s*</div>", second_detail.text)
    assert "best" not in first_detail.text
    assert "best" in second_detail.text
    assert first.text == "expected the two-character answer 42\n"
    assert second.text == "answer accepted\n"
    assert archive.status_code == 404
    assert client.get("/").status_code == 200
    assert not list(runs_dir.rglob("*.tar*"))
    assert json.loads((run.path / "final_result.json").read_text())["best_score"] == 1


def test_runtime_container_names_match_producers_scanner_and_k8s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An underscore producer prefix would evade scanner and K8s handling."""
    assert run_agent_module._run_container_name("task-name", "run-id") == (
        "rsi-loop.run.task-name.run-id"
    )
    assert judge_server_module._game_container_name("task-name", "session-id") == (
        "rsi-loop.game.task-name.session-id"
    )
    assert scanner_module._runtime_container_names("task-name", "run-id") == (
        "rsi-loop.run.task-name.run-id",
        "rsi-loop.evolve.task-name.run-id",
    )
    assert (
        _sanitize_k8s_name("rsi-loop.run.task-with-long-name.run-id")
        == "run-task-with-long-run-id"
    )

    commands: list[list[str]] = []

    def docker_ps(command: list[str], **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="rsi-loop.run.task-name.run-id\trunning\n",
        )

    monkeypatch.setattr(scanner_module, "_docker_cache", {})
    monkeypatch.setattr(scanner_module, "_docker_cache_ts", 0.0)
    monkeypatch.setattr(scanner_module.time, "time", lambda: 10.0)
    monkeypatch.setattr(scanner_module.subprocess, "run", docker_ps)

    assert scanner_module._docker_status_map() == {
        "rsi-loop.run.task-name.run-id": "running"
    }
    assert commands == [
        [
            "docker",
            "ps",
            "-a",
            "--format",
            "{{.Names}}\t{{.State}}",
            "--filter",
            "name=rsi-loop.",
        ]
    ]


def test_rendered_visualizer_migrates_valid_legacy_browser_state(
    tmp_path: Path,
) -> None:
    """Renaming browser keys must not reset an existing user's active state."""
    runs_dir, tasks_dir = _write_visualizer_run(tmp_path)
    html = TestClient(create_app(runs_dir, tasks_dir=tasks_dir)).get("/").text

    score_read = html.index("localStorage.getItem(LEGACY_KEY)")
    score_write = html.index("localStorage.setItem(KEY, legacy)")
    score_remove = html.index("localStorage.removeItem(LEGACY_KEY)")
    assert "const LEGACY_KEY = 'sforge_score_view';" in html
    assert "legacy === 'raw' || legacy === 'rescaled'" in html
    assert score_read < score_write < score_remove

    state_read = html.index("parseState(sessionStorage.getItem(legacyStateKey))")
    state_write = html.index("sessionStorage.setItem(stateKey, JSON.stringify(legacy))")
    state_remove = html.index("sessionStorage.removeItem(legacyStateKey)")
    assert 'const legacyStateKey = "sforge.visualizer.index.state.v1";' in html
    assert 'typeof runId === "string"' in html
    assert "Number.isFinite(state.scrollY)" in html
    assert state_read < state_write < state_remove


def _coding_agent_config() -> RSILoopConfig:
    return RSILoopConfig(
        agent_api_key="sk-visualizer-test",
        agent_api_base_url="http://model.invalid/v1",
        agent_context_window="131072",
        agent_max_output_tokens="16384",
    )


def _driver_init(agent: str, version: str, model: str = "test-model") -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "init",
            "agent": agent,
            "agent_version": version,
            "model": model,
            "session_id": "session-1",
            "cwd": "/workspace",
            "permissionMode": "bypassPermissions",
            "rsi_driver": "zcode",
        }
    )


@pytest.mark.parametrize("name", ["kimi-code", "opencode", "pi", "qwen-code", "zcode"])
def test_peek_agent_recognises_coding_agent_launch_lines(
    tmp_path: Path, name: str
) -> None:
    """The log line keeps 200 characters of the real command, as run_agent logs it."""
    agent = create_agent(name, _coding_agent_config())
    command = agent.format_run_cmd("/tmp/prompt.md", model="m", continuation="go on")
    # A driven command without its config prelude, as a supervisor may log it.
    driver_command = command[command.find("exec ") :] if name != "pi" else command
    lines = (
        f"2026-10-12 10:00:00,000 INFO Running agent: {command[:200]}... (timeout=60s)",
        f"2026-10-12 10:00:00,000 INFO Running agent: {driver_command[:200]}...",
        f"2026-10-12 10:00:00,000 INFO Agent supervisor attempt 1: {driver_command}",
        "2026-10-12 10:00:00,000 INFO Resuming agent (attempt 1/100, 30s left): "
        f"{driver_command[:200]}...",
    )
    for i, line in enumerate(lines):
        task_dir = tmp_path / str(i)
        task_dir.mkdir()
        (task_dir / "run_agent.log").write_text(line + "\n")
        if name != "pi" and "--agent-name" not in line:
            # The truncated command hides --agent-name; the init event names it.
            assert scanner_module._peek_agent(task_dir) == ""
            (task_dir / "agent_output.txt").write_text(_driver_init(name, "1.0") + "\n")
        assert scanner_module._peek_agent(task_dir) == name


def test_peek_agent_keeps_legacy_formats_and_ignores_unrelated_lines(
    tmp_path: Path,
) -> None:
    cases = {
        "Running agent: claude -p --output-format stream-json\n": "claude-code",
        "Running agent: codex exec --model m\n": "codex",
        "x - INFO - Agent supervisor attempt 2: claude -p --verbose\n": "claude-code",
        "x - INFO - Agent supervisor attempt 2: codex exec\n": "codex",
        "Running agent: aider --yes\n": "aider",
        # Mentions in other lines, or other commands, identify nothing.
        "note: rsi-agent-driver --agent-name zcode\n": "",
        "Running agent: pip install pi --offline-mode\n": "",
        "Running agent: python -m pipeline --offline\n": "",
    }
    for i, (log, expected) in enumerate(cases.items()):
        task_dir = tmp_path / str(i)
        task_dir.mkdir()
        (task_dir / "run_agent.log").write_text(log)
        assert scanner_module._peek_agent(task_dir) == expected, log


def test_peek_helpers_read_the_coding_agent_init_event(tmp_path: Path) -> None:
    (tmp_path / "run_agent.log").write_text("")
    (tmp_path / "agent_output.txt").write_text(
        "not json\n"
        + json.dumps({"type": "system", "subtype": "stderr", "text": "warming up"})
        + "\n"
        + _driver_init("zcode", "3.15.1", model="glm-test")
        + "\n"
    )
    assert scanner_module._peek_agent(tmp_path) == "zcode"
    assert scanner_module._peek_model(tmp_path) == "glm-test"

    # A Claude Code init has no agent field: the model is still found.
    (tmp_path / "agent_output.txt").write_text(
        json.dumps({"type": "system", "subtype": "init", "model": "claude-test"}) + "\n"
    )
    assert scanner_module._peek_agent(tmp_path) == ""
    assert scanner_module._peek_model(tmp_path) == "claude-test"

    # RSI Harness logs no launch command: Claude Code's own init event and
    # Codex's header name them.
    (tmp_path / "agent_output.txt").write_text(
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "claude-test",
                "claude_code_version": "2.1.0",
            }
        )
        + "\n"
    )
    assert scanner_module._peek_agent(tmp_path) == "claude-code"
    (tmp_path / "agent_output.txt").write_text(
        "Reading additional input from stdin...\nOpenAI Codex v0.1.0\n--------\n"
        "model: gpt-test\n--------\n"
    )
    assert scanner_module._peek_agent(tmp_path) == "codex"
    assert scanner_module._peek_model(tmp_path) == "gpt-test"


def test_trajectory_init_summary_names_the_coding_agent(tmp_path: Path) -> None:
    output = tmp_path / "agent_output.txt"
    output.write_text(_driver_init("pi", "1.1.0") + "\n")
    trajectory = agent_output_parse(output)
    assert trajectory.meta["agent"] == "pi"
    assert trajectory.meta["agent_version"] == "1.1.0"
    summary = trajectory.exchanges[0].text.splitlines()
    assert summary[:2] == ["model: test-model", "agent: pi 1.1.0"]
    assert not any(line.startswith("claude_code:") for line in summary)

    output.write_text(
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "claude-test",
                "claude_code_version": "2.1.0",
                "tools": ["Bash", "Read"],
            }
        )
        + "\n"
    )
    trajectory = agent_output_parse(output)
    assert trajectory.meta["agent"] is None
    summary = trajectory.exchanges[0].text.splitlines()
    assert summary[:2] == ["model: claude-test", "claude_code: 2.1.0"]
    assert "tools (2): Bash, Read" in summary


def test_in_progress_coding_agent_run_shows_agent_model_and_trajectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "runs"
    task_dir = runs_dir / "run-live" / "task-a"
    task_dir.mkdir(parents=True)
    (task_dir / "started_at").write_text("2026-10-12T10:00:00\n1760263200.0\n")
    (task_dir / "run_agent.log").write_text("")
    (task_dir / "agent_output.txt").write_text(
        _driver_init("zcode", "3.15.1", model="glm-test")
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Working on it."}]},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        scanner_module,
        "_docker_status_map",
        lambda: {"rsi-loop.run.task-a.run-live": "running"},
    )

    runs = RunsIndex(runs_dir).list_runs()
    assert [(r.run_id, r.agent, r.model, r.has_final) for r in runs] == [
        ("run-live", "zcode", "glm-test", False)
    ]
    page = TestClient(create_app(runs_dir)).get("/run/run-live/task-a/trajectory")
    assert page.status_code == 200
    assert "zcode 3.15.1" in page.text
    assert "Working on it." in page.text
