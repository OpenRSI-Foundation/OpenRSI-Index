"""Plan the next attempt of a run that ended, whatever ended it.

The attempt keeps the run ID, its logs and everything on disk: Work is rebuilt
from the run's retained image and kept WORKDIR volume, and the Agent continues
its own session. Live processes are not restored.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rsi_harness.errors import SetupError
from rsi_harness.models import (
    ManagedWorkdirVolume,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPlan,
    RunRequest,
    SandboxPolicySource,
    SubmissionReport,
    TaskDefinition,
)
from rsi_harness.runtime.artifacts import recorded_submissions
from rsi_harness.runtime.recovery import (
    AttemptGrants,
    LeaseStore,
    ResourceLease,
    retained_round_id,
)
from rsi_harness.runtime.rootfs_snapshot import rootfs_image_ref
from rsi_harness.runtime.run_request import RUN_REQUEST_FILE, read_run_request
from rsi_harness.runtime.trajectory import TRAJECTORY_DIRECTORY, TRAJECTORY_MANIFEST

# Each attempt stacks one more retained image layer, and overlay2 allows
# about 128.
MAX_ATTEMPTS = 20
_ROUND = re.compile(r"agent-([1-9][0-9]*)(?:\.log)?\Z")
_ROUND_DIRECTORIES = ("submissions", "feedback", "verifier", TRAJECTORY_DIRECTORY)
_TASK_DIGESTS = (
    "source_digest",
    "instruction_digest",
    "tests_digest",
    "environment_digest",
)


@dataclass(frozen=True, slots=True)
class ResumeContext:
    """What the next attempt of a run continues from."""

    run_id: str
    attempt: int
    # The saved request, with this attempt's GPU selectors.
    request: RunRequest
    sandbox_policy: SandboxPolicySource | None
    # The run's run-plan.json, and the GPU shape it ran with.
    plan: RunPlan
    gpu_plan: RunGPUPlan | None
    # Rounds evolve_state.json records, oldest first.
    reports: tuple[SubmissionReport, ...]
    charged_rounds: int
    last_allocated_at: datetime | None
    agent_seconds_used: float
    agent_seconds_left: float
    # All attempts' submissions, grants included; None is unlimited.
    max_submissions: int | None
    cooldown_seconds: float
    grants: AttemptGrants
    # None for a run whose Work state is all on shared storage (a cluster).
    base_image_ref: str | None
    # None also while a dry run's salvage has not committed it.
    base_image_id: str | None
    base_image_round_id: str | None
    workdir_volume: ManagedWorkdirVolume | None
    salvage_pending: bool
    # The Agent's next user turn.
    message: str

    @property
    def next_round_id(self) -> str:
        return f"agent-{self.charged_rounds + 1}"

    @property
    def submissions_left(self) -> int | None:
        if self.max_submissions is None:
            return None
        return self.max_submissions - self.charged_rounds


def plan_resume(
    run_id: str,
    *,
    store: LeaseStore,
    data_root: Path,
    logs_root: Path,
    grants: AttemptGrants,
    compile_task: Callable[[RunRequest], TaskDefinition],
    salvage: Callable[[str], None] | None,
    retained_images: bool = True,
    open_attempt_ended_at: float | None = None,
) -> ResumeContext:
    """Plan attempt k+1 of ``run_id`` under its lock; a live run is refused.

    A run whose Work may still exist, or whose lease is not terminal, is
    first salvaged by ``salvage`` (recovery, which the caller binds). With
    ``salvage`` None, as in a dry run, nothing changes and the plan says
    salvage would run. Without ``retained_images`` (a cluster run, whose
    Work state is its shared workspace and Agent home) no Work image is
    needed. An attempt salvage has yet to close is counted up to its pause,
    else ``open_attempt_ended_at`` (when its end is known, as a cluster
    job's), else now.
    """
    try:
        recorded = store.path_for(run_id).exists()
    except ValueError as error:
        raise SetupError(str(error)) from None
    if not recorded:
        raise SetupError(f"run {run_id} has no lease under {store.root}")
    lock = store.lock(run_id, blocking=False)
    try:
        lock.__enter__()
    except BlockingIOError:
        raise SetupError(
            f"run {run_id} is still live (its Engine holds the lease)"
        ) from None
    try:
        return _plan_locked(
            run_id,
            store=store,
            data_root=data_root,
            logs_root=logs_root,
            grants=grants,
            compile_task=compile_task,
            salvage=salvage,
            retained_images=retained_images,
            open_attempt_ended_at=open_attempt_ended_at,
        )
    finally:
        lock.__exit__(None, None, None)


def same_gpu_shape(left: RunGPUPlan, right: RunGPUPlan) -> bool:
    """Whether two GPU plans give Work and Judge the same GPUs in number."""
    return (len(left.work.devices), len(left.judge.devices), left.judge_mode) == (
        len(right.work.devices),
        len(right.judge.devices),
        right.judge_mode,
    )


def _plan_locked(
    run_id: str,
    *,
    store: LeaseStore,
    data_root: Path,
    logs_root: Path,
    grants: AttemptGrants,
    compile_task: Callable[[RunRequest], TaskDefinition],
    salvage: Callable[[str], None] | None,
    retained_images: bool,
    open_attempt_ended_at: float | None,
) -> ResumeContext:
    lease = store.read(run_id)
    if lease is None:
        raise SetupError(f"run {run_id} has no lease under {store.root}")
    if not (data_root / run_id / RUN_REQUEST_FILE).exists():
        raise SetupError(
            f"run {run_id} has no {RUN_REQUEST_FILE}: it started before "
            "resume existed and cannot be resumed"
        )
    record = read_run_request(data_root / run_id)
    if "[REDACTED]" in record.request.model_dump_json():
        raise SetupError(
            f"run {run_id} was started with a credential in its request, such "
            "as a Judge LLM URL with a password, which was not saved; it "
            "cannot be resumed"
        )
    salvage_pending = (
        lease.status is None
        or lease.recovery_required
        or lease.work.container_id is not None
        or any(entry.ended_at is None for entry in lease.attempts)
    )
    if salvage_pending and salvage is not None:
        try:
            salvage(run_id)
        except RuntimeError as error:
            raise SetupError(f"salvage of run {run_id} failed: {error}") from error
        lease = store.read(run_id)
        assert lease is not None
        salvage_pending = False
    if lease.recovery_required and not salvage_pending:
        raise SetupError(f"run {run_id} still requires recovery: {lease.error}")
    attempt = lease.attempt + 1
    if attempt > MAX_ATTEMPTS:
        raise SetupError(f"run {run_id} already had {MAX_ATTEMPTS} attempts")

    root = logs_root / "runs" / run_id / lease.task_id
    if lease.artifact_root is not None and lease.artifact_root != root:
        raise SetupError(
            f"run {run_id} logs are in {lease.artifact_root}; pass its --logs-root"
        )
    if not (root / "run-plan.json").exists():
        raise SetupError(f"run {run_id} has no run-plan.json in {root}")
    plan = RunPlan.model_validate(json.loads((root / "run-plan.json").read_text()))
    definition = compile_task(record.request)
    for field in _TASK_DIGESTS:
        if getattr(definition, field) != getattr(plan.task, field):
            name = field.removesuffix("_digest")
            raise SetupError(f"the task's {name} changed since run {run_id} started")

    recorded = recorded_submissions(root)
    accepted = {_round_number(report.round_id) for report, _ in recorded}
    # Every allocated round counts, the one in flight at a crash included.
    # Allocation itself is in memory only: a round the Engine died on before
    # its first file (while Work was being paused) is not charged.
    charged = max(
        (
            *accepted,
            *(
                number
                for directory in _ROUND_DIRECTORIES
                for number in _round_numbers(root / directory)
            ),
        ),
        default=0,
    )
    used = _agent_seconds_used(lease, open_attempt_ended_at)
    granted = [entry.grants for entry in lease.attempts if entry.grants is not None]
    granted.append(grants)
    budget = plan.task.agent.timeout_seconds + sum(
        entry.add_time_seconds for entry in granted
    )
    left = budget - used
    if left <= 0:
        raise SetupError(
            f"run {run_id} has no Agent time left ({used:.0f}s of {budget:.0f}s "
            "used); grant more with --add-time SECONDS"
        )
    maximum = record.request.options.max_submissions
    if maximum is None and grants.add_submissions:
        raise SetupError(f"run {run_id} has no submission limit to add to")
    if maximum is not None:
        maximum += sum(entry.add_submissions for entry in granted)
        if maximum <= charged:
            raise SetupError(
                f"run {run_id} has no submissions left ({charged} of {maximum} "
                "used); grant more with --add-submissions N"
            )
    cooldown = next(
        (
            entry.cooldown_seconds
            for entry in reversed(granted)
            if entry.cooldown_seconds is not None
        ),
        record.request.options.cooldown_seconds,
    )
    selectors = next(
        (
            entry.gpu_selectors
            for entry in reversed(granted)
            if entry.gpu_selectors is not None
        ),
        record.request.gpu_selectors,
    )

    work = lease.work
    salvage_ref = rootfs_image_ref(
        run_id=run_id,
        task_id=lease.task_id,
        round_id=retained_round_id("salvage", lease.attempt),
        purpose="retained-work",
    )
    base_ref, base_id = work.retained_image_ref, work.retained_image_id
    base_round_id: str | None = None
    if not retained_images:
        base_ref = base_id = None
    elif base_ref is None and salvage_pending and work.container_id is not None:
        base_ref = work.planned_retained_image_ref or salvage_ref
    elif base_ref is None and work.prior_retained_images:
        # The last attempt ended before it had a Work to retain.
        prior = work.prior_retained_images[-1]
        base_ref, base_id, base_round_id = prior.ref, prior.id, prior.round_id
    if retained_images and (
        base_ref is None or (base_id is None and not salvage_pending)
    ):
        raise SetupError(f"run {run_id} kept no Work image to resume from")
    if base_ref is not None and base_round_id is None:
        base_round_id = retained_round_id(
            "salvage" if base_ref == salvage_ref else "final", lease.attempt
        )
    volume = work.workdir_volume.actual
    if (
        lease.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR
        and volume is None
    ):
        raise SetupError(f"run {run_id} kept no WORKDIR volume to resume from")
    last_accepted = max(accepted, default=0)
    return ResumeContext(
        run_id=run_id,
        attempt=attempt,
        request=record.request.model_copy(update={"gpu_selectors": selectors}),
        sandbox_policy=record.sandbox_policy,
        plan=plan,
        gpu_plan=lease.gpu_plan,
        reports=tuple(report for report, _ in recorded),
        charged_rounds=charged,
        last_allocated_at=_allocated_at(root, charged, recorded),
        agent_seconds_used=used,
        agent_seconds_left=left,
        max_submissions=maximum,
        cooldown_seconds=cooldown,
        grants=grants,
        base_image_ref=base_ref,
        base_image_id=base_id,
        base_image_round_id=base_round_id,
        workdir_volume=volume,
        salvage_pending=salvage_pending,
        message=_message(
            attempt,
            interrupted=tuple(
                f"agent-{number}" for number in range(last_accepted + 1, charged + 1)
            ),
            seconds_left=left,
            submissions_left=None if maximum is None else maximum - charged,
        ),
    )


def _agent_seconds_used(lease: ResourceLease, open_ended_at: float | None) -> float:
    """Agent seconds of every attempt; an attempt salvage has yet to close
    is counted up to its pause, else ``open_ended_at``, else now."""
    used = 0.0
    for entry in lease.attempts:
        if entry.agent_seconds is not None:
            used += entry.agent_seconds
        elif entry.agent_started_at is not None and entry.agent_deadline is not None:
            end = min(
                lease.work.paused_at or open_ended_at or time.time(),
                entry.agent_deadline,
            )
            used += max(0.0, end - entry.agent_started_at)
    return used


def _round_number(round_id: str) -> int:
    match = _ROUND.fullmatch(round_id)
    if match is None:
        raise SetupError(f"recorded round {round_id!r} is not an Agent round")
    return int(match.group(1))


def _round_numbers(directory: Path) -> tuple[int, ...]:
    if not directory.is_dir():
        return ()
    return tuple(
        int(match.group(1))
        for child in directory.iterdir()
        if (match := _ROUND.fullmatch(child.name)) is not None
    )


def _allocated_at(
    root: Path,
    charged: int,
    recorded: tuple[tuple[SubmissionReport, float], ...],
) -> datetime | None:
    """When the last round was allocated: its trajectory manifest is written
    once, at the pause; failing that, when its report or its first file was
    recorded."""
    if charged == 0:
        return None
    round_id = f"agent-{charged}"
    manifest = root / TRAJECTORY_DIRECTORY / round_id / TRAJECTORY_MANIFEST
    if manifest.exists():
        return datetime.fromtimestamp(manifest.stat().st_mtime, UTC)
    for report, submitted_at in recorded:
        if report.round_id == round_id:
            return datetime.fromtimestamp(submitted_at, UTC)
    found = [
        path.stat().st_mtime
        for path in (
            root / TRAJECTORY_DIRECTORY / round_id,
            root / "verifier" / round_id,
            root / "feedback" / f"{round_id}.log",
            root / "submissions" / round_id,
        )
        if path.exists()
    ]
    return datetime.fromtimestamp(min(found), UTC) if found else None


def _message(
    attempt: int,
    *,
    interrupted: tuple[str, ...],
    seconds_left: float,
    submissions_left: int | None,
) -> str:
    lines = [f"rsi-harness: the run stopped and resumes now as attempt {attempt}."]
    lines.extend(
        f"Round {round_id} was in flight when the run stopped: it counts as a "
        "submission and has no score; its partial output was withheld."
        for round_id in interrupted
    )
    lines.append("Background processes and GPU state are gone; files are kept.")
    lines.append(f"Agent time left: {seconds_left:.0f} seconds.")
    lines.append(
        "Submissions left: "
        + ("unlimited." if submissions_left is None else f"{submissions_left}.")
    )
    return "\n".join(lines) + "\n"


__all__ = ["MAX_ATTEMPTS", "ResumeContext", "plan_resume", "same_gpu_shape"]
