from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import (
    CompileOptions,
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    JudgeLLMAuth,
    JudgeLLMRequest,
    ManagedWorkdirVolume,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunRequest,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.recovery import (
    AttemptGrants,
    LeaseStore,
    PriorRetainedImage,
    ResourceLease,
    RunAttemptLease,
    WorkdirVolumeResourceLease,
    WorkResourceLease,
)
from rsi_harness.runtime.resume import MAX_ATTEMPTS, plan_resume, same_gpu_shape
from rsi_harness.runtime.rootfs_snapshot import rootfs_image_ref
from rsi_harness.runtime.run_request import write_run_request
from tests.factories import make_run_plan

TASK_ID = "minimal-gpu"
FINAL_REF = rootfs_image_ref(
    run_id="run-1", task_id=TASK_ID, round_id="final", purpose="retained-work"
)
FINAL_ID = f"sha256:{'b' * 64}"


def closed_attempt(attempt: int = 1, **updates: object) -> RunAttemptLease:
    return RunAttemptLease(
        attempt=attempt,
        started_at=1.0,
        agent_started_at=2.0,
        agent_deadline=62.0,
        ended_at=40.0,
        end_basis="exited",
        agent_seconds=38.0,
    ).model_copy(update=updates)


class Run:
    """One finished run on disk: logs, request.json and its lease."""

    def __init__(self, tmp_path: Path, *, max_submissions: int | None = 8) -> None:
        self.plan = make_run_plan(tmp_path)
        self.data_root = tmp_path / "data"
        self.logs_root = self.plan.paths.logs
        self.store = LeaseStore(self.data_root / "leases")
        self.writer = RunArtifactWriter(self.plan, run_id="run-1")
        self.writer.start()
        (self.data_root / "run-1").mkdir(parents=True)
        write_run_request(
            self.data_root / "run-1",
            run_id="run-1",
            request=RunRequest(
                task_dir=tmp_path,
                gpu_selectors=("0",),
                options=CompileOptions(
                    max_submissions=max_submissions, cooldown_seconds=5.0
                ),
            ),
        )
        self.volume = ManagedWorkdirVolume(
            name=f"rsi-harness-workdir-{'1' * 64}",
            run_id="run-1",
            task_id=TASK_ID,
            target=self.plan.workdir,
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        )
        self.write_lease()

    def write_lease(self, **updates: object) -> ResourceLease:
        lease = ResourceLease(
            run_id="run-1",
            task_id=TASK_ID,
            coordinator_pid=1,
            coordinator_started_at=1.0,
            phase=RunStatus.COMPLETED.value,
            status=RunStatus.COMPLETED,
            rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            gpu_plan=self.plan.gpu_plan,
            artifact_root=self.writer.root,
            attempts=(closed_attempt(),),
            work=WorkResourceLease(
                planned_retained_image_ref=FINAL_REF,
                retained_image_id=FINAL_ID,
                retained_image_ref=FINAL_REF,
                workdir_volume=WorkdirVolumeResourceLease(
                    planned_name=self.volume.name,
                    planned_target=self.volume.target,
                    planned_snapshot_mode=self.volume.snapshot_mode,
                    planned_freshness_nonce=self.volume.freshness_nonce,
                    actual=self.volume,
                    mounted=True,
                ),
            ),
        ).model_copy(update=updates)
        self.store.write(lease)
        return lease

    def record(self, round_id: str, score: float) -> None:
        self.writer.record_submission(
            SubmissionReport(
                round_id=round_id,
                status=SubmissionStatus.COMPLETED,
                score=score,
                rewards={"reward": score},
                output="ok\n",
            )
        )

    def plan_resume(self, grants: AttemptGrants | None = None, **options: object):
        return plan_resume(
            "run-1",
            store=self.store,
            data_root=self.data_root,
            logs_root=self.logs_root,
            grants=grants or AttemptGrants(),
            compile_task=options.pop("compile_task", lambda _request: self.plan.task),
            salvage=options.pop("salvage", None),
            **options,
        )


def test_plan_rebuilds_rounds_from_the_logs_with_gaps_and_an_interrupted_round(
    tmp_path,
) -> None:
    run = Run(tmp_path)
    run.record("agent-1", 0.5)
    run.record("agent-3", 0.9)
    root = run.writer.root
    # Written before a rejection: report.json alone never counts as a round.
    (root / "submissions" / "agent-4").mkdir()
    (root / "submissions" / "agent-4" / "report.json").write_text("{}")
    # In flight at the crash: only its trajectory copy exists. Its manifest
    # keeps the pause time; releasing the copy later touches the directory.
    trajectory = root / "judge-trajectory" / "agent-5"
    trajectory.mkdir(parents=True)
    (trajectory / "manifest.json").write_text("{}")
    os.utime(trajectory / "manifest.json", (1_700_000_000, 1_700_000_000))
    os.utime(trajectory, (1_700_000_900, 1_700_000_900))

    context = run.plan_resume()

    assert context.attempt == 2
    assert [report.round_id for report in context.reports] == ["agent-1", "agent-3"]
    assert context.charged_rounds == 5
    assert context.next_round_id == "agent-6"
    assert context.last_allocated_at == datetime.fromtimestamp(1_700_000_000, UTC)
    assert context.agent_seconds_used == 38.0
    assert context.agent_seconds_left == 22.0
    assert context.max_submissions == 8
    assert context.submissions_left == 3
    assert context.cooldown_seconds == 5.0
    assert context.request.gpu_selectors == ("0",)
    assert context.base_image_ref == FINAL_REF
    assert context.base_image_id == FINAL_ID
    assert context.base_image_round_id == "final"
    assert context.workdir_volume == run.volume
    assert not context.salvage_pending
    assert "attempt 2" in context.message
    assert "Round agent-4 was in flight" in context.message
    assert "Round agent-5 was in flight" in context.message
    assert "agent-2" not in context.message
    assert "Agent time left: 22 seconds." in context.message
    assert "Submissions left: 3." in context.message
    assert "files are kept" in context.message


def test_plan_counts_every_attempt_and_adds_the_grants(tmp_path) -> None:
    run = Run(tmp_path)
    run.record("agent-1", 0.5)
    earlier = AttemptGrants(
        add_time_seconds=100.0, add_submissions=2, cooldown_seconds=1.0
    )
    run.write_lease(
        attempt=2,
        attempts=(
            closed_attempt(),
            closed_attempt(2, agent_seconds=90.0, grants=earlier),
        ),
    )

    context = run.plan_resume(
        AttemptGrants(add_time_seconds=10.0, add_submissions=1, gpu_selectors=("1",))
    )

    assert context.attempt == 3
    assert context.agent_seconds_used == 128.0
    assert context.agent_seconds_left == 60.0 + 100.0 + 10.0 - 128.0
    assert context.max_submissions == 8 + 2 + 1
    assert context.submissions_left == 10
    assert context.cooldown_seconds == 1.0
    assert context.request.gpu_selectors == ("1",)
    assert context.grants.add_time_seconds == 10.0


def test_the_latest_cooldown_and_gpu_grants_win(tmp_path) -> None:
    run = Run(tmp_path)
    run.write_lease(
        attempt=3,
        attempts=(
            closed_attempt(),
            closed_attempt(
                2, grants=AttemptGrants(cooldown_seconds=1.0, gpu_selectors=("2",))
            ),
            closed_attempt(
                3, grants=AttemptGrants(cooldown_seconds=3.0, gpu_selectors=("3",))
            ),
        ),
    )

    inherited = run.plan_resume(AttemptGrants(add_time_seconds=100.0))
    replaced = run.plan_resume(
        AttemptGrants(
            add_time_seconds=100.0, cooldown_seconds=0.0, gpu_selectors=("1",)
        )
    )

    assert (inherited.cooldown_seconds, inherited.request.gpu_selectors) == (
        3.0,
        ("3",),
    )
    assert (replaced.cooldown_seconds, replaced.request.gpu_selectors) == (0.0, ("1",))


def test_the_last_round_without_a_manifest_takes_its_earliest_trace(
    tmp_path,
) -> None:
    run = Run(tmp_path)
    run.record("agent-1", 0.5)
    root = run.writer.root
    (submitted_at,) = (
        entry["at"]
        for entry in json.loads((root / "evolve_state.json").read_text())["submissions"]
    )

    assert run.plan_resume().last_allocated_at == datetime.fromtimestamp(
        submitted_at, UTC
    )

    # A round with only Judge output: it counts, from its earliest file.
    (root / "verifier" / "agent-2").mkdir(parents=True)
    os.utime(root / "verifier" / "agent-2", (1_700_000_200, 1_700_000_200))
    (root / "feedback" / "agent-2.log").write_text("")
    os.utime(root / "feedback" / "agent-2.log", (1_700_000_100, 1_700_000_100))

    context = run.plan_resume()

    assert context.charged_rounds == 2
    assert context.last_allocated_at == datetime.fromtimestamp(1_700_000_100, UTC)
    assert "Round agent-2 was in flight" in context.message


def test_plan_resumes_from_the_chain_when_the_last_attempt_had_no_work(
    tmp_path,
) -> None:
    run = Run(tmp_path)
    lease = run.write_lease()
    run.write_lease(
        attempt=2,
        attempts=(closed_attempt(), closed_attempt(2, agent_seconds=0.0)),
        work=lease.work.model_copy(
            update={
                "planned_retained_image_ref": None,
                "retained_image_id": None,
                "retained_image_ref": None,
                "prior_retained_images": (
                    PriorRetainedImage(ref=FINAL_REF, id=FINAL_ID, round_id="final"),
                ),
            }
        ),
    )

    context = run.plan_resume()

    assert context.attempt == 3
    assert (context.base_image_ref, context.base_image_id) == (FINAL_REF, FINAL_ID)
    assert context.base_image_round_id == "final"


def test_an_unfinished_run_is_salvaged_first_and_a_dry_run_changes_nothing(
    tmp_path,
) -> None:
    run = Run(tmp_path)
    lease = run.write_lease(
        status=None,
        phase=RunStatus.AGENT_RUNNING.value,
        attempts=(
            closed_attempt().model_copy(
                update={"ended_at": None, "end_basis": None, "agent_seconds": None}
            ),
        ),
        work=run.write_lease().work.model_copy(
            update={
                "container_id": "work-1",
                "planned_retained_image_ref": None,
                "retained_image_id": None,
                "retained_image_ref": None,
                "paused": True,
                "paused_at": 32.0,
            }
        ),
    )
    before = run.store.path_for("run-1").read_bytes()

    dry = run.plan_resume()

    assert dry.salvage_pending
    assert dry.base_image_ref == rootfs_image_ref(
        run_id="run-1", task_id=TASK_ID, round_id="salvage", purpose="retained-work"
    )
    assert dry.base_image_id is None
    assert dry.base_image_round_id == "salvage"
    assert dry.agent_seconds_used == 30.0
    assert run.store.path_for("run-1").read_bytes() == before

    salvaged: list[str] = []

    def salvage(run_id: str) -> None:
        salvaged.append(run_id)
        run.store.write(
            lease.model_copy(
                update={
                    "status": RunStatus.CANCELLED,
                    "attempts": (closed_attempt(agent_seconds=30.0),),
                    "work": lease.work.model_copy(
                        update={
                            "container_id": None,
                            "paused": False,
                            "paused_at": None,
                            "planned_retained_image_ref": dry.base_image_ref,
                            "retained_image_id": FINAL_ID,
                            "retained_image_ref": dry.base_image_ref,
                        }
                    ),
                }
            )
        )

    context = run.plan_resume(salvage=salvage)

    assert salvaged == ["run-1"]
    assert not context.salvage_pending
    assert context.base_image_id == FINAL_ID
    assert context.base_image_round_id == "salvage"
    assert context.agent_seconds_used == 30.0


def test_a_failed_salvage_is_a_refusal(tmp_path) -> None:
    run = Run(tmp_path)
    run.write_lease(status=None)

    def salvage(_run_id: str) -> None:
        raise RuntimeError("Work stop failed; runtime authority remains retained")

    with pytest.raises(SetupError, match="salvage of run run-1 failed: Work stop"):
        run.plan_resume(salvage=salvage)


def _no_time(run: Run) -> None:
    run.write_lease(attempts=(closed_attempt(agent_seconds=60.0),))


def _no_submissions(run: Run) -> None:
    for number in range(1, 9):
        run.record(f"agent-{number}", 0.1)


def _no_request(run: Run) -> None:
    (run.data_root / "run-1" / "request.json").unlink()


def _too_many_attempts(run: Run) -> None:
    run.write_lease(
        attempt=MAX_ATTEMPTS,
        attempts=tuple(closed_attempt(n, agent_seconds=0.0) for n in range(1, 21)),
    )


def _still_needs_recovery(run: Run) -> None:
    run.write_lease(recovery_required=True, error="recovery_required: Work stop")


def _no_volume(run: Run) -> None:
    lease = run.write_lease()
    run.write_lease(
        work=lease.work.model_copy(
            update={"workdir_volume": WorkdirVolumeResourceLease()}
        )
    )


def _redacted_credential(run: Run) -> None:
    write_run_request(
        run.data_root / "run-1",
        run_id="run-1",
        request=RunRequest(
            task_dir=run.plan.task.source_dir,
            options=CompileOptions(max_submissions=8),
            judge_llm=JudgeLLMRequest(
                auth=JudgeLLMAuth.API,
                api_key_env="JUDGE_KEY",
                proxy="http://user:pw@proxy.example:3128",
            ),
        ),
    )


def _no_image(run: Run) -> None:
    lease = run.write_lease()
    run.write_lease(
        work=lease.work.model_copy(
            update={
                "planned_retained_image_ref": None,
                "retained_image_id": None,
                "retained_image_ref": None,
            }
        )
    )


@pytest.mark.parametrize(
    ("prepare", "message"),
    [
        (_no_time, "no Agent time left"),
        (_no_submissions, "no submissions left"),
        (_no_request, "started before resume existed"),
        (_too_many_attempts, f"already had {MAX_ATTEMPTS} attempts"),
        (_still_needs_recovery, "still requires recovery"),
        (_no_image, "kept no Work image"),
        (_no_volume, "kept no WORKDIR volume"),
        (_redacted_credential, "credential in its request"),
    ],
)
def test_plan_refuses_a_run_it_cannot_resume(tmp_path, prepare, message) -> None:
    run = Run(tmp_path)
    prepare(run)

    with pytest.raises(SetupError, match=message):
        run.plan_resume(salvage=lambda _run_id: None)


def test_a_grant_lifts_the_refusal_of_an_exhausted_run(tmp_path) -> None:
    run = Run(tmp_path)
    _no_time(run)
    _no_submissions(run)

    context = run.plan_resume(AttemptGrants(add_time_seconds=30.0, add_submissions=2))

    assert context.agent_seconds_left == 30.0
    assert context.submissions_left == 2
    assert context.next_round_id == "agent-9"


def test_plan_refuses_added_submissions_without_a_limit(tmp_path) -> None:
    run = Run(tmp_path, max_submissions=None)

    assert run.plan_resume().submissions_left is None
    with pytest.raises(SetupError, match="no submission limit"):
        run.plan_resume(AttemptGrants(add_submissions=1))


def test_plan_refuses_a_task_that_changed(tmp_path) -> None:
    run = Run(tmp_path)
    changed = run.plan.task.model_copy(update={"tests_digest": "other"})

    with pytest.raises(SetupError, match="task's tests changed"):
        run.plan_resume(compile_task=lambda _request: changed)


def test_plan_refuses_a_live_run(tmp_path) -> None:
    run = Run(tmp_path)

    with run.store.lock("run-1"):
        with pytest.raises(SetupError, match="still live"):
            run.plan_resume()


def test_plan_refuses_other_logs_roots(tmp_path) -> None:
    run = Run(tmp_path)
    run.logs_root = tmp_path / "elsewhere"

    with pytest.raises(SetupError, match="pass its --logs-root"):
        run.plan_resume()


@pytest.mark.parametrize(
    ("run_id", "message"), [("run-2", "no lease"), ("../x", "safe path component")]
)
def test_plan_refuses_an_unknown_run_without_creating_state(
    tmp_path, run_id, message
) -> None:
    store = LeaseStore(tmp_path / "leases")

    with pytest.raises(SetupError, match=message):
        plan_resume(
            run_id,
            store=store,
            data_root=tmp_path,
            logs_root=tmp_path,
            grants=AttemptGrants(),
            compile_task=lambda _request: None,
            salvage=None,
        )

    assert not (tmp_path / "leases").exists()


def test_same_gpu_shape_compares_counts_and_the_judge_mode() -> None:
    devices = tuple(
        GPUDevice(index=index, uuid=f"GPU-{index}", name="Test GPU")
        for index in range(4)
    )

    def plan(work: slice, judge: slice) -> RunGPUPlan:
        return RunGPUPlan(
            authorized_pool=GPUAllocation(devices=devices),
            work=GPUAllocation(devices=devices[work]),
            judge=GPUAllocation(devices=devices[judge]),
            judge_mode=JudgeGPUMode.DISJOINT,
        )

    assert same_gpu_shape(
        plan(slice(0, 1), slice(1, 2)), plan(slice(2, 3), slice(3, 4))
    )
    assert not same_gpu_shape(
        plan(slice(0, 1), slice(1, 2)), plan(slice(0, 2), slice(2, 3))
    )
    assert not same_gpu_shape(
        plan(slice(0, 1), slice(1, 2)), plan(slice(0, 1), slice(1, 3))
    )


def test_a_run_on_shared_storage_resumes_without_a_retained_image(tmp_path) -> None:
    run = Run(tmp_path)
    run.record("agent-1", 0.5)
    run.write_lease(
        work=run.write_lease().work.model_copy(
            update={
                "planned_retained_image_ref": None,
                "retained_image_id": None,
                "retained_image_ref": None,
            }
        )
    )
    with pytest.raises(SetupError, match="kept no Work image"):
        run.plan_resume()

    context = run.plan_resume(retained_images=False)

    assert context.attempt == 2
    assert context.next_round_id == "agent-2"
    assert (context.base_image_ref, context.base_image_id) == (None, None)
    assert context.base_image_round_id is None
    assert context.workdir_volume == run.volume
