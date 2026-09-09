from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = (
    ROOT
    / ".agents"
    / "skills"
    / "extracting-rsi-run-logs-bluevela"
    / "scripts"
    / "extract_logs.py"
)


def _write_complete_run(
    run_root: Path,
    logs_root: Path,
    *,
    run_id: str,
    task_id: str,
    state: str = "completed",
) -> Path:
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "agent-home").mkdir()
    (run_dir / "agent-home" / "auth.json").write_text("do-not-publish")

    leaf = logs_root / "runs" / run_id / task_id
    (leaf / "feedback").mkdir(parents=True)
    (leaf / "submissions" / "agent-1").mkdir(parents=True)
    (leaf / "agent_output.txt").write_text(f"agent output for {run_id}\n")
    (leaf / "agent_prompt.md").write_text("prompt\n")
    (leaf / "run_agent.log").write_text(f"agent log for {run_id}\n")
    (leaf / "run-plan.json").write_text("{}\n")
    (leaf / "started_at").write_text("2026-08-25T19:16:51+00:00\n")
    (leaf / "feedback" / "agent-1.log").write_text("judge output\n")
    (leaf / "submissions" / "agent-1" / "report.json").write_text(
        json.dumps({"round": "agent-1", "status": "completed", "score": 1.0}) + "\n"
    )
    (leaf / "final_result.json").write_text(
        json.dumps({"status": "completed", "total_rounds": 1}) + "\n"
    )
    (run_dir / "RUN_INFO.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "state": state,
                "purpose": f"RSI Harness task {task_id}",
                "expected_outputs": [
                    str(leaf / "final_result.json"),
                    str(leaf / "agent_output.txt"),
                    str(leaf / "run_agent.log"),
                    str(leaf / "submissions" / "agent-1" / "report.json"),
                ],
            }
        )
        + "\n"
    )
    return leaf


def _run_extractor(
    task_id: str,
    *,
    run_root: Path,
    logs_root: Path,
    destination_root: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(SCRIPT),
            task_id,
            "--run-root",
            str(run_root),
            "--logs-root",
            str(logs_root),
            "--destination-root",
            str(destination_root),
        ),
        check=False,
        capture_output=True,
        text=True,
    )


def _file_contents(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_extracts_latest_completed_run_and_only_publishes_formal_logs(tmp_path: Path):
    run_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    destination_root = tmp_path / "public" / "rsi-logs"
    task_id = "molmo2-pointing-refined"
    selected_id = "20260825T191651Z-molmo2-pointing-refined-fa4d1263"
    selected = _write_complete_run(
        run_root,
        logs_root,
        run_id=selected_id,
        task_id=task_id,
    )
    _write_complete_run(
        run_root,
        logs_root,
        run_id="20260826T091500Z-molmo2-pointing-refined-running",
        task_id=task_id,
        state="running",
    )

    result = _run_extractor(
        task_id,
        run_root=run_root,
        logs_root=logs_root,
        destination_root=destination_root,
    )

    target = destination_root / selected_id / task_id
    assert result.returncode == 0, result.stderr
    assert str(target) in result.stdout
    assert _file_contents(target) == _file_contents(selected)
    assert not (destination_root / selected_id / "agent-home").exists()
    assert not (
        destination_root / "20260826T091500Z-molmo2-pointing-refined-running"
    ).exists()


def test_identical_destination_is_idempotent(tmp_path: Path):
    run_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    destination_root = tmp_path / "rsi-logs"
    run_id = "20260825T191651Z-task-a-12345678"
    _write_complete_run(run_root, logs_root, run_id=run_id, task_id="task-a")

    first = _run_extractor(
        "task-a",
        run_root=run_root,
        logs_root=logs_root,
        destination_root=destination_root,
    )
    second = _run_extractor(
        "task-a",
        run_root=run_root,
        logs_root=logs_root,
        destination_root=destination_root,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert "already matches" in second.stdout


def test_refuses_to_merge_logs_into_an_existing_execution_directory(tmp_path: Path):
    run_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    destination_root = tmp_path / "rsi-logs"
    run_id = "20260825T191651Z-task-a-12345678"
    _write_complete_run(run_root, logs_root, run_id=run_id, task_id="task-a")
    conflicting_run = destination_root / run_id
    conflicting_run.mkdir(parents=True)
    marker = conflicting_run / "RUN_INFO.json"
    marker.write_text("execution state must be preserved\n")

    result = _run_extractor(
        "task-a",
        run_root=run_root,
        logs_root=logs_root,
        destination_root=destination_root,
    )

    assert result.returncode != 0
    assert "destination run directory already exists" in result.stderr
    assert marker.read_text() == "execution state must be preserved\n"
    assert not (conflicting_run / "task-a").exists()


def test_reports_when_no_completed_run_exists(tmp_path: Path):
    run_root = tmp_path / "runs"
    logs_root = tmp_path / "logs"
    destination_root = tmp_path / "rsi-logs"
    _write_complete_run(
        run_root,
        logs_root,
        run_id="20260826T091500Z-task-a-running",
        task_id="task-a",
        state="running",
    )

    result = _run_extractor(
        "task-a",
        run_root=run_root,
        logs_root=logs_root,
        destination_root=destination_root,
    )

    assert result.returncode != 0
    assert "no completed run found for task 'task-a'" in result.stderr
    assert not destination_root.exists()
