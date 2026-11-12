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
from tests.runtime.test_sandbox_e2b import TASK as ENV_TASK
from tests.runtime.test_sandbox_e2b import e2b_grant

TASK_ID = "minimal-gpu"
FINAL_REF = rootfs_image_ref(
    run_id="run-1", task_id=TASK_ID, round_id="final", purpose="retained-work"
)
FINAL_ID = f"sha256:{'b' * 64}"
JUDGE_ONLY = (
    ENV_TASK[: ENV_TASK.index("[metadata.rsi_harness.sandbox.environments.work]")]
    + ENV_TASK[ENV_TASK.index("[metadata.rsi_harness.sandbox.environments.judge]") :]
)


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

    def __init__(
        self,
        tmp_path: Path,
        *,
        max_submissions: int | None = 8,
        sandbox: str | None = None,
        plan_budget: int | None = None,
    ) -> None:
        self.plan = make_run_plan(tmp_path)
        if plan_budget is not None:
            verifier = self.plan.task.verifier.model_copy(
                update={"max_submissions": plan_budget}
            )
            self.plan = self.plan.model_copy(
                update={
                    "task": self.plan.task.model_copy(update={"verifier": verifier})
                }
            )
        if sandbox is not None:
            self.plan = self.plan.model_copy(
                update={"sandbox": e2b_grant(tmp_path, task=sandbox)}
            )
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


def test_backend_validation_precedes_salvage_under_the_run_lock(tmp_path):
    run = Run(tmp_path)
    lease = run.write_lease(status=None)
    before = run.store.path_for("run-1").read_bytes()
    events = []

    def validate(context):
        events.append("validate")
        assert context.salvage_pending
        assert context.agent_seconds_left == 22.0
        with pytest.raises(SetupError, match="still live"):
            run.plan_resume()
        raise SetupError("backend cannot run this remaining plan")

    def salvage(_run_id):
        events.append("salvage")
        run.store.write(lease.model_copy(update={"status": RunStatus.CANCELLED}))

    with pytest.raises(SetupError, match="backend cannot run"):
        run.plan_resume(salvage=salvage, validate_before_salvage=validate)
    assert events == ["validate"]
    assert run.store.path_for("run-1").read_bytes() == before

    context = run.plan_resume(
        salvage=salvage,
        validate_before_salvage=lambda context: events.append("accepted"),
    )
    assert events == ["validate", "accepted", "salvage"]
    assert not context.salvage_pending


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


@pytest.mark.parametrize(
    ("sandbox", "told"),
    [(ENV_TASK, True), (JUDGE_ONLY, False), (None, False)],
    ids=["work-envs", "judge-envs-only", "no-sandbox"],
)
def test_the_message_says_envs_are_gone_only_to_a_work_with_envs(
    tmp_path, sandbox, told
) -> None:
    run = Run(tmp_path, sandbox=sandbox)

    message = run.plan_resume().message

    assert "files are kept" in message
    assert ("Sandbox envs and image handles are gone" in message) is told


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


@pytest.mark.parametrize("flag", [1, None], ids=["flag", "task"])
def test_a_final_judge_attempt_gets_its_own_round_whatever_was_used(
    tmp_path, flag
) -> None:
    run = Run(tmp_path, max_submissions=flag, plan_budget=1)
    run.record("agent-1", 0.9)

    context = run.plan_resume()

    assert context.final_judge
    assert context.max_submissions == 2
    assert context.submissions_left == 1
    assert context.next_round_id == "agent-2"
    message = context.message
    assert "Agent time left: 22 seconds." in message
    assert "This attempt ends with one final judge" in message
    assert "`rsi-submit` once" in message
    # No earlier result and no submission count reach the Agent.
    assert "agent-1" not in message and "0.9" not in message
    assert "Submissions left" not in message


def test_a_final_judge_attempt_after_an_interrupted_round_says_nothing_of_it(
    tmp_path,
) -> None:
    run = Run(tmp_path, max_submissions=None, plan_budget=1)
    (run.writer.root / "verifier" / "agent-1").mkdir(parents=True)

    context = run.plan_resume()

    assert context.next_round_id == "agent-2"
    assert "in flight" not in context.message and "agent-1" not in context.message


def test_a_final_judge_run_refuses_added_submissions(tmp_path) -> None:
    run = Run(tmp_path, max_submissions=1, plan_budget=1)

    with pytest.raises(SetupError, match="one final judge per attempt"):
        run.plan_resume(AttemptGrants(add_submissions=1))


def test_a_plan_from_before_the_budget_field_keeps_its_old_behaviour(
    tmp_path,
) -> None:
    """--max-submissions 1 before the upgrade: the saved plan has no budget,
    so the run keeps its rsi-submit rounds and refuses a used-up budget."""
    run = Run(tmp_path, max_submissions=1)
    plan = json.loads((run.writer.root / "run-plan.json").read_text())
    assert plan["task"]["verifier"].pop("max_submissions") is None
    (run.writer.root / "run-plan.json").write_text(json.dumps(plan))

    context = run.plan_resume()

    assert not context.final_judge
    assert context.submissions_left == 1
    assert "Submissions left: 1." in context.message
    run.record("agent-1", 0.5)
    with pytest.raises(SetupError, match="no submissions left"):
        run.plan_resume()


def test_a_task_budget_in_the_plan_limits_a_run_without_the_flag(tmp_path) -> None:
    run = Run(tmp_path, max_submissions=None, plan_budget=3)
    run.record("agent-1", 0.5)

    context = run.plan_resume()

    assert not context.final_judge
    assert context.submissions_left == 2
    assert "Submissions left: 2." in context.message


def test_a_run_judged_on_its_agents_own_rsi_submit_cannot_be_resumed(
    tmp_path,
) -> None:
    """The Agent saw that result in its session; another attempt would be a
    retry of the one final judge."""
    run = Run(tmp_path, max_submissions=None, plan_budget=1)
    run.record("agent-1", 0.9)
    run.writer.record_final_round("agent-1", "agent")

    with pytest.raises(SetupError, match="own rsi-submit \\(agent-1\\)"):
        run.plan_resume(AttemptGrants(add_time_seconds=60.0))


@pytest.mark.parametrize(
    ("requested_by", "status"),
    [
        ("harness", SubmissionStatus.COMPLETED),
        ("agent", SubmissionStatus.INFRASTRUCTURE_ERROR),
    ],
    ids=["harness-round", "agent-round-infrastructure-error"],
)
def test_a_final_judge_run_resumes_when_the_agent_saw_no_judged_result(
    tmp_path, requested_by, status
) -> None:
    run = Run(tmp_path, max_submissions=None, plan_budget=1)
    run.writer.record_submission(
        SubmissionReport(
            round_id="agent-1",
            status=status,
            score=0.9 if status is SubmissionStatus.COMPLETED else None,
            rewards={"reward": 0.9} if status is SubmissionStatus.COMPLETED else {},
            output="ok\n",
            error=None if status is SubmissionStatus.COMPLETED else "Judge died",
        )
    )
    run.writer.record_final_round("agent-1", requested_by)

    context = run.plan_resume()

    assert context.final_judge
    assert context.next_round_id == "agent-2"


@pytest.mark.parametrize("plan_budget", [None, 1, 3])
def test_a_resume_compiles_with_its_plans_own_budget(tmp_path, plan_budget) -> None:
    """Whatever --max-submissions the run was started with; a plan written
    before the field (None) compiles as unlimited and keeps its old budget."""
    run = Run(tmp_path, max_submissions=2, plan_budget=plan_budget)
    if plan_budget is None:
        plan = json.loads((run.writer.root / "run-plan.json").read_text())
        plan["task"]["verifier"].pop("max_submissions")
        (run.writer.root / "run-plan.json").write_text(json.dumps(plan))
    compiled: list[int | None] = []

    def compile_task(request):
        compiled.append(request.options.max_submissions)
        return run.plan.task

    context = run.plan_resume(compile_task=compile_task)

    assert compiled == [plan_budget]
    assert context.request.options.max_submissions == plan_budget
    assert context.final_judge is (plan_budget == 1)
    assert context.submissions_left == (1 if plan_budget == 1 else plan_budget or 2)


def test_a_pre_field_budget_one_run_with_a_sandbox_still_resumes(tmp_path) -> None:
    """Budget 1 with a sandbox is refused for runs the field governs only."""
    from rsi_harness.errors import UnsupportedTaskError
    from rsi_harness.runtime.run_request import read_run_request
    from rsi_harness.task.compiler import HarborTaskCompiler
    from tests.sandbox_helpers import sandbox_toml
    from tests.task.test_compiler import DEFAULT_TASK_TOML, write_harbor_task

    task = write_harbor_task(
        tmp_path / "task", task_toml=DEFAULT_TASK_TOML + sandbox_toml()
    )
    run = Run(tmp_path / "run", max_submissions=1)
    plan = json.loads((run.writer.root / "run-plan.json").read_text())
    plan["task"]["verifier"].pop("max_submissions")
    (run.writer.root / "run-plan.json").write_text(json.dumps(plan))
    compiler = HarborTaskCompiler()
    compiled = []

    def compile_task(request):
        compiled.append(compiler.compile(task, request.options))
        return run.plan.task

    context = run.plan_resume(compile_task=compile_task)

    assert compiled[0].verifier.max_submissions is None
    assert not context.final_judge
    assert context.submissions_left == 1
    # The saved options alone, as before this fix, are refused.
    saved = read_run_request(run.data_root / "run-1").request.options
    with pytest.raises(UnsupportedTaskError, match="does not support"):
        compiler.compile(task, saved)
