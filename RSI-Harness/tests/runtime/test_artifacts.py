from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from rsi_harness.models import (
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    RunGPUPlan,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from tests.factories import make_run_plan
from tests.fakes import FakeClock


def report(
    round_id: str,
    *,
    score: float,
    rewards: dict[str, float],
    output: str,
) -> SubmissionReport:
    return SubmissionReport(
        round_id=round_id,
        status=SubmissionStatus.COMPLETED,
        score=score,
        rewards=rewards,
        output=output,
        exit_code=3,
        duration_seconds=1.25,
    )


def test_artifact_start_rejects_preexisting_symlinked_feedback_root(
    tmp_path: Path,
) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.root.mkdir(parents=True)
    redirected = plan.paths.logs / "redirected-feedback"
    redirected.mkdir(parents=True)
    writer.feedback_root.symlink_to(redirected, target_is_directory=True)

    with pytest.raises(NotADirectoryError):
        writer.start()


def test_run_plan_artifact_serializes_exact_gpu_plan_phase_allocations(
    tmp_path: Path,
) -> None:
    devices = (
        GPUDevice(index=0, uuid="GPU-a", name="Test GPU"),
        GPUDevice(index=1, uuid="GPU-b", name="Test GPU"),
    )
    plan = make_run_plan(tmp_path).model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=devices),
                work=GPUAllocation(devices=devices[:1]),
                judge=GPUAllocation(devices=devices[1:]),
                judge_mode=JudgeGPUMode.DISJOINT,
            )
        }
    )

    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()

    payload = json.loads((writer.root / "run-plan.json").read_text())
    assert payload["gpu_plan"] == {
        "authorized_pool": {
            "devices": [
                {"index": 0, "uuid": "GPU-a", "name": "Test GPU"},
                {"index": 1, "uuid": "GPU-b", "name": "Test GPU"},
            ]
        },
        "work": {"devices": [{"index": 0, "uuid": "GPU-a", "name": "Test GPU"}]},
        "judge": {"devices": [{"index": 1, "uuid": "GPU-b", "name": "Test GPU"}]},
        "judge_mode": "disjoint",
    }


def test_writes_exact_rsi_loop_tree_and_complete_multi_reward_payloads(
    tmp_path: Path,
) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    first = report(
        "agent-1",
        score=0.25,
        rewards={"accuracy": 0.25, "latency": 9.0},
        output="first verifier output\n",
    )
    second = report(
        "agent-2",
        score=0.75,
        rewards={"accuracy": 0.75, "latency": 7.0},
        output="second verifier output\n",
    )

    writer.start()
    writer.record_submission(first)
    writer.record_submission(second)
    writer.finalize(
        status=RunStatus.COMPLETED,
        runtime_seconds=42.5,
        timed_out=False,
    )

    root = tmp_path / "logs" / "runs" / "run-1" / "minimal-gpu"
    assert {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    } == {
        "started_at",
        "run-plan.json",
        "run_agent.log",
        "agent_output.txt",
        "evolve_state.json",
        "feedback/agent-1.log",
        "feedback/agent-2.log",
        "final_result.json",
        "submissions/agent-1/report.json",
        "submissions/agent-1/test_output.txt",
        "submissions/agent-2/report.json",
        "submissions/agent-2/test_output.txt",
    }
    first_payload = json.loads((root / "submissions/agent-1/report.json").read_text())
    assert first_payload["score"] == 0.25
    assert first_payload["metrics"] == {"accuracy": 0.25, "latency": 9.0}
    assert "output" not in first_payload
    assert (root / "submissions/agent-1/test_output.txt").read_text() == first.output
    state = json.loads((root / "evolve_state.json").read_text())
    assert state["submissions"] == [
        {
            "at": 1767225600.0,
            "kind": "agent",
            "rewards": {"accuracy": 0.25, "latency": 9.0},
            "round": "agent-1",
            "score": 0.25,
            "status": "completed",
        },
        {
            "at": 1767225600.0,
            "kind": "agent",
            "rewards": {"accuracy": 0.75, "latency": 7.0},
            "round": "agent-2",
            "score": 0.75,
            "status": "completed",
        },
    ]
    final = json.loads((root / "final_result.json").read_text())
    assert final == {
        "agent": "codex",
        "best_rewards": {"accuracy": 0.75, "latency": 9.0},
        "best_round": "agent-2",
        "best_score": 0.75,
        "model": None,
        "runtime_seconds": 42.5,
        "status": "completed",
        "timed_out": False,
        "timeout_seconds": 60.0,
        "total_rounds": 2,
    }
    assert not (root / "submission.tar.gz").exists()
    assert not (root / "final_archive.tar.gz").exists()
    assert float((root / "started_at").read_text().splitlines()[-1]) == 1767225600.0


def test_sudo_run_artifacts_belong_to_invoking_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invoking_gid = next(
        (group for group in os.getgroups() if group != os.getgid()), None
    )
    if invoking_gid is None:
        pytest.skip("ownership regression requires a supplementary group")
    monkeypatch.setattr("rsi_harness.runtime.artifacts.os.geteuid", lambda: 0)
    monkeypatch.setenv("SUDO_UID", str(os.getuid()))
    monkeypatch.setenv("SUDO_GID", str(invoking_gid))
    writer = RunArtifactWriter(
        make_run_plan(tmp_path), run_id="sudo-run", clock=FakeClock()
    )

    writer.start()
    writer.record_submission(
        report(
            "agent-1",
            score=1.0,
            rewards={"reward": 1.0},
            output="visible verifier output",
        )
    )
    writer.finalize()

    entries = (writer.root, *writer.root.rglob("*"))
    assert all(path.stat().st_uid == os.getuid() for path in entries)
    assert all(path.stat().st_gid == invoking_gid for path in entries)


def test_raw_output_is_bounded_and_exists_only_in_text_artifact(tmp_path: Path) -> None:
    plan = make_run_plan(tmp_path)
    verifier = plan.task.verifier.model_copy(update={"output_limit_bytes": 8})
    task = plan.task.model_copy(update={"verifier": verifier})
    plan = plan.model_copy(update={"task": task})
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()

    writer.record_submission(
        report(
            "agent-1",
            score=1.0,
            rewards={"reward": 1.0},
            output="abcdefgh-more-output",
        )
    )

    root = tmp_path / "logs" / "runs" / "run-1" / "minimal-gpu"
    assert (root / "submissions/agent-1/test_output.txt").read_bytes() == b"abcdefgh"
    payload = json.loads((root / "submissions/agent-1/report.json").read_text())
    assert "output" not in payload


def test_submission_artifact_hardlinks_the_complete_feedback_log(tmp_path) -> None:
    plan = make_run_plan(tmp_path)
    verifier = plan.task.verifier.model_copy(update={"output_limit_bytes": 80})
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"verifier": verifier})}
    )
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    complete = b"head\n" + b"middle" * 100 + b"\ntail\n"
    writer.start()
    feedback = writer.feedback_root / "agent-1.log"
    feedback.write_bytes(complete)

    complete_report = report(
        "agent-1", score=1.0, rewards={"reward": 1.0}, output="headtail"
    ).model_copy(
        update={
            "verifier_output_required": True,
            "full_output_captured": True,
        }
    )
    writer.record_submission(complete_report)

    visualizer_output = writer.root / "submissions/agent-1/test_output.txt"
    assert visualizer_output.read_bytes() == complete
    assert visualizer_output.samefile(feedback)
    payload = json.loads(
        (writer.root / "submissions/agent-1/report.json").read_text()
    )
    assert payload["verifier_output_required"] is True
    assert payload["full_output_captured"] is True


def test_task_authored_verifier_output_is_not_redacted(tmp_path: Path) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()
    authored = "access_token=TASK_VALUE\nAuthorization: Digest TASK_RESPONSE\n"

    writer.record_submission(
        report("agent-1", score=1.0, rewards={"reward": 1.0}, output=authored)
    )

    assert (
        writer.root / "submissions/agent-1/test_output.txt"
    ).read_text() == authored


def test_failed_submission_cannot_replace_best_valid_score(tmp_path: Path) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()
    writer.record_submission(
        report(
            "agent-1",
            score=0.5,
            rewards={"reward": 0.5},
            output="valid",
        )
    )
    failed = SubmissionReport(
        round_id="agent-2",
        status=SubmissionStatus.INFRASTRUCTURE_ERROR,
        score=99.0,
        rewards={"reward": 99.0},
        error="judge container failed",
    )

    writer.record_submission(failed)
    writer.finalize()

    final = json.loads((writer.root / "final_result.json").read_text())
    assert final["best_score"] == 0.5
    assert final["best_round"] == "agent-1"
    assert final["best_rewards"] == {"reward": 0.5}


def test_final_result_aggregates_each_reward_key_independently(tmp_path: Path) -> None:
    writer = RunArtifactWriter(make_run_plan(tmp_path), run_id="run-1")
    writer.start()
    writer.record_submission(
        report(
            "agent-1",
            score=0.9,
            rewards={"reward": 0.9, "robustness": 0.2},
            output="first",
        )
    )
    writer.record_submission(
        report(
            "agent-2",
            score=0.8,
            rewards={"reward": 0.8, "robustness": 0.7},
            output="second",
        )
    )

    writer.finalize()

    final = json.loads((writer.root / "final_result.json").read_text())
    assert final["best_score"] == 0.9
    assert final["best_round"] == "agent-1"
    assert final["best_rewards"] == {"reward": 0.9, "robustness": 0.7}


def test_ambiguous_valid_rewards_complete_without_scalar_best(tmp_path: Path) -> None:
    writer = RunArtifactWriter(make_run_plan(tmp_path), run_id="run-1")
    writer.start()
    writer.record_submission(
        SubmissionReport(
            round_id="agent-1",
            status=SubmissionStatus.COMPLETED,
            rewards={"accuracy": 0.9, "latency": 4},
            score=None,
        )
    )

    writer.finalize(status=RunStatus.COMPLETED)

    final = json.loads((writer.root / "final_result.json").read_text())
    assert final["status"] == "completed"
    assert final["best_score"] is None
    assert final["best_round"] is None
    assert final["best_rewards"] == {"accuracy": 0.9, "latency": 4.0}


def test_engine_error_artifact_redacts_secret_values(tmp_path: Path) -> None:
    writer = RunArtifactWriter(make_run_plan(tmp_path), run_id="run-1")
    writer.start()

    writer.record_engine_error(
        "Engine bug: access_token=LEAKME&safe=yes\n"
        "Authorization: Digest username=admin, response=AUTHLEAK\n"
        'headers={"X-API-Key": "HEADERLEAK"} token=abc credential hunter2'
    )

    raw = (writer.root / "engine_error.json").read_text()
    assert "LEAKME" not in raw
    assert "AUTHLEAK" not in raw
    assert "HEADERLEAK" not in raw
    assert "abc" not in raw
    assert "hunter2" not in raw
    assert raw.count("[REDACTED]") == 5


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("invalid_first", [True, False])
def test_nonfinite_score_is_rejected_without_disturbing_valid_ordering(
    tmp_path: Path, nonfinite: float, invalid_first: bool
) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()
    invalid_round = "agent-1" if invalid_first else "agent-2"
    valid_round = "agent-2" if invalid_first else "agent-1"
    invalid = report(
        invalid_round,
        score=nonfinite,
        rewards={"reward": 0.25},
        output="invalid",
    )
    valid = report(
        valid_round,
        score=0.5,
        rewards={"reward": 0.5},
        output="valid",
    )

    ordered = (invalid, valid) if invalid_first else (valid, invalid)
    for submission in ordered:
        if submission is invalid:
            with pytest.raises(ValueError, match="finite"):
                writer.record_submission(submission)
        else:
            writer.record_submission(submission)
    writer.finalize()

    assert not (writer.root / "submissions" / invalid_round).exists()
    state = json.loads((writer.root / "evolve_state.json").read_text())
    assert [entry["round"] for entry in state["submissions"]] == [valid_round]
    final = json.loads((writer.root / "final_result.json").read_text())
    assert final["best_score"] == 0.5
    assert final["best_round"] == valid_round
    assert final["total_rounds"] == 1


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("destination", ["reward", "duration"])
def test_nonfinite_report_json_value_is_rejected_before_submission_artifacts(
    tmp_path: Path, nonfinite: float, destination: str
) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()
    invalid = SubmissionReport(
        round_id="agent-1",
        status=SubmissionStatus.COMPLETED,
        score=0.5,
        rewards={"reward": nonfinite if destination == "reward" else 0.5},
        duration_seconds=nonfinite if destination == "duration" else 1.0,
    )
    previous_state = (writer.root / "evolve_state.json").read_bytes()

    with pytest.raises(ValueError, match="finite"):
        writer.record_submission(invalid)

    assert not (writer.root / "submissions" / "agent-1").exists()
    assert (writer.root / "evolve_state.json").read_bytes() == previous_state


def test_json_serialization_guard_rejects_nonfinite_final_runtime(
    tmp_path: Path,
) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()

    with pytest.raises(ValueError, match="JSON compliant"):
        writer.finalize(runtime_seconds=float("nan"))

    assert not (writer.root / "final_result.json").exists()


def test_new_artifact_directory_entries_are_fsynced_in_their_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced_directories: list[Path] = []
    real_fsync = os.fsync

    def record_fsync(file_descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            target = os.readlink(f"/proc/self/fd/{file_descriptor}")
            synced_directories.append(Path(target))
        real_fsync(file_descriptor)

    monkeypatch.setattr("rsi_harness.runtime.artifacts.os.fsync", record_fsync)
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())

    writer.start()

    expected_start_parents = {
        tmp_path,
        tmp_path / "logs",
        tmp_path / "logs" / "runs",
        tmp_path / "logs" / "runs" / "run-1",
    }
    assert expected_start_parents <= set(synced_directories)

    synced_directories.clear()
    writer.record_submission(
        report(
            "agent-1",
            score=0.5,
            rewards={"reward": 0.5},
            output="valid",
        )
    )

    assert {
        writer.root,
        writer.root / "submissions",
    } <= set(synced_directories)


@pytest.mark.parametrize("has_previous", [False, True])
def test_interruption_before_replace_never_exposes_partial_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_previous: bool
) -> None:
    plan = make_run_plan(tmp_path)
    writer = RunArtifactWriter(plan, run_id="run-1", clock=FakeClock())
    writer.start()
    old_report = report(
        "agent-1",
        score=0.25,
        rewards={"reward": 0.25},
        output="old",
    )
    if has_previous:
        writer.record_submission(old_report)
    target = writer.root / "submissions/agent-1/report.json"
    previous = json.loads(target.read_text()) if has_previous else None

    def interrupt_replace(source: Path | str, destination: Path | str) -> None:
        assert Path(source).parent == Path(destination).parent
        raise RuntimeError("injected interruption before atomic rename")

    monkeypatch.setattr("rsi_harness.runtime.artifacts.os.replace", interrupt_replace)

    with pytest.raises(RuntimeError, match="injected interruption"):
        writer.record_submission(
            report(
                "agent-1",
                score=0.75,
                rewards={"reward": 0.75},
                output="new",
            )
        )

    if has_previous:
        assert json.loads(target.read_text()) == previous
    else:
        assert not target.exists()
