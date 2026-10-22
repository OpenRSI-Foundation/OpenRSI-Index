"""`rsi-harness resume --cluster`: the submit host closes the run's last
attempt and submits the next one under the same run ID; the scheduler, the
Engine job and E2B are faked."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela import runtime as runtime_module
from rsi_harness.cluster.bluevela.adapter import BlueVelaClusterAdapter
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.bluevela.resources import resumed_run_walltime
from rsi_harness.cluster.bluevela.runtime import NativeEngineComposition
from rsi_harness.cluster.schedulers.lsf import LSFJobResult
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    ContainerRef,
    FrozenRewardMap,
    RootfsSnapshotMode,
    RunStatus,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.recovery import (
    AttemptGrants,
    LeaseStore,
    ResourceLease,
    RunAttemptLease,
    WorkdirVolumeResourceLease,
    WorkResourceLease,
)
from tests.cluster.bluevela.test_adapter import (
    RecordingScheduler,
    _profile,
    _request,
)
from tests.cluster.bluevela.test_adapter import (
    _fake_codex_runtime as _fake_codex_runtime,
)


class ResumeScheduler(RecordingScheduler):
    """Runs attempt 1 as RecordingScheduler does; a resumed attempt reopens
    the run's logs and records its next round."""

    def __init__(self) -> None:
        super().__init__()
        self.active = False
        self.name_taken = False
        self.ended_at: float | None = None

    def finished(self, job_id: str) -> LSFJobResult | None:
        self.events.append(f"finished-{job_id}")
        if self.active:
            return None
        return LSFJobResult(
            job_id=job_id, state="EXIT", exit_code=140, ended_at=self.ended_at
        )

    def require_name_available(self, name: str) -> None:
        super().require_name_available(name)
        if self.name_taken:
            raise InfrastructureError(f"LSF job name is already active: {name}")

    def wait(self, job_id: str, **kwargs) -> LSFJobResult:
        spec = self.specs[int(job_id) - 101]
        if spec.script_path.parent.parent.name != "attempts":
            return super().wait(job_id, **kwargs)
        payload = load_engine_payload(spec.script_path.with_suffix(".json"))
        assert payload.resume is not None
        self.events.append("wait-resume")
        writer = RunArtifactWriter(payload.run_plan, run_id=payload.run_id)
        writer.reopen(payload.resume.attempt)
        (writer.root / "run_agent.log").write_text("attempt log\n")
        writer.record_submission(
            SubmissionReport(
                round_id=payload.resume.next_round_id,
                status=SubmissionStatus.COMPLETED,
                rewards=FrozenRewardMap({"reward": 2.0}),
                score=2.0,
                output="passed\n",
            )
        )
        writer.finalize(attempts=payload.resume.attempt)
        return LSFJobResult(job_id=job_id, state="DONE", exit_code=0)


def _interrupted_run(tmp_path: Path):
    """Attempt 1 recorded agent-1, then its job was killed while agent-2 was
    being judged: Work paused, the round's feedback partly written."""
    profile = _profile(tmp_path)
    scheduler = ResumeScheduler()
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        agent_version_resolver=lambda _name: "0.149.0",
        binary_version=lambda _path: "0.149.0",
    )
    result = adapter.run(_request(tmp_path, dry_run=False))
    run_dir = profile.storage.run_root / result.run_id
    payload = load_engine_payload(run_dir / "control" / "run.json")
    plan = payload.run_plan
    root = result.log_dir
    (root / "final_result.json").unlink()
    (root / "feedback" / "agent-2.log").write_text("partial Judge output\n")
    (root / "verifier" / "agent-2").mkdir(parents=True)
    now = time.time()
    volume = NativeEngineComposition.plan_volume(None, plan, result.run_id)  # type: ignore[arg-type]
    store = LeaseStore(run_dir / "leases")
    store.write(
        ResourceLease(
            run_id=result.run_id,
            task_id=plan.task.task_id,
            coordinator_pid=1,
            coordinator_started_at=now - 1000,
            phase=RunStatus.AGENT_RUNNING.value,
            rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            gpu_plan=plan.gpu_plan,
            artifact_root=root,
            attempts=(
                RunAttemptLease(
                    attempt=1,
                    started_at=now - 1000,
                    agent_started_at=now - 900,
                    agent_deadline=now - 900 + plan.task.agent.timeout_seconds,
                ),
            ),
            work=WorkResourceLease(
                container_id=f"bluevela-{result.run_id}-work",
                paused=True,
                paused_at=now - 300,
                workdir_volume=WorkdirVolumeResourceLease(
                    planned_name=volume.name,
                    planned_target=volume.target,
                    planned_snapshot_mode=volume.snapshot_mode,
                    planned_freshness_nonce=volume.freshness_nonce,
                    actual=volume,
                    mounted=True,
                ),
            ),
        )
    )
    return SimpleNamespace(
        adapter=adapter,
        scheduler=scheduler,
        profile=profile,
        run_id=result.run_id,
        run_dir=run_dir,
        root=root,
        store=store,
        plan=plan,
        paused_at=now - 300,
    )


def _tree(*roots: Path) -> dict[str, bytes | None]:
    return {
        str(path): path.read_bytes() if path.is_file() else None
        for root in roots
        for path in sorted(root.rglob("*"))
        # Taking the lease lock creates its lock file.
        if path.suffix != ".lock"
    }


def test_resume_closes_the_attempt_and_submits_the_next_under_the_run_id(
    tmp_path: Path,
) -> None:
    run = _interrupted_run(tmp_path)
    grants = AttemptGrants(add_time_seconds=60.0)

    plan = run.adapter.plan_resume(run.run_id, grants)

    context = plan.context
    assert (context.attempt, context.charged_rounds) == (2, 2)
    assert context.next_round_id == "agent-3"
    assert context.base_image_ref is None
    assert not context.salvage_pending
    # The attempt ended when Work paused for the round the job died in.
    lease = run.store.read(run.run_id)
    assert lease.status is RunStatus.CANCELLED
    assert lease.work.container_id is None
    (attempt,) = lease.attempts
    assert (attempt.ended_at, attempt.end_basis) == (run.paused_at, "paused")
    assert attempt.agent_seconds == pytest.approx(600.0)
    assert context.agent_seconds_left == pytest.approx(
        run.plan.task.agent.timeout_seconds + 60.0 - 600.0
    )
    assert not (run.root / "feedback" / "agent-2.log").exists()
    assert (
        run.root / "verifier" / "agent-2" / "interrupted-feedback.log"
    ).read_text() == "partial Judge output\n"
    assert plan.spec.walltime == resumed_run_walltime(
        run.plan.task, run.profile, agent_seconds=context.agent_seconds_left
    )
    assert plan.spec.walltime != run.scheduler.specs[-1].walltime

    manifest_path = run.run_dir / "RUN_INFO.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(state="failed", error="job 102 ended in EXIT", job_ids=["102"])
    manifest_path.write_text(json.dumps(manifest))

    result = run.adapter.resume(plan)

    attempt_dir = run.run_dir / "control" / "attempts" / "2"
    payload = load_engine_payload(attempt_dir / "run.json")
    assert payload.run_id == run.run_id
    assert payload.resume is not None
    assert payload.resume.attempt == 2
    assert payload.resume.grants == grants
    assert payload.resume.agent_seconds_left == context.agent_seconds_left
    assert payload.resume.submissions_left == context.submissions_left
    assert payload.resume.next_round_id == "agent-3"
    assert payload.resume.message == context.message
    assert payload.resources.run_walltime == plan.spec.walltime
    assert (attempt_dir / "run.sh").read_text().count(str(attempt_dir / "run.json"))
    first, resumed = run.scheduler.specs[-2:]
    assert resumed.name == first.name
    assert resumed.script_path == attempt_dir / "run.sh"
    assert resumed.stdout_path.parent == first.stdout_path.parent
    assert result.run_id == run.run_id
    assert result.status is RunStatus.COMPLETED
    assert result.job_ids == ("103",)
    manifest = json.loads((run.run_dir / "RUN_INFO.json").read_text())
    assert (manifest["attempt"], manifest["run_job_id"]) == (2, "103")
    assert manifest["attempt_jobs"] == {"1": "102", "2": "103"}
    assert manifest["state"] == "completed"
    # The failure of the attempt before is not this attempt's.
    assert "error" not in manifest and "job_ids" not in manifest
    final = json.loads((run.root / "final_result.json").read_text())
    assert (final["attempts"], final["best_round"]) == (2, "agent-3")


def test_resume_dry_run_changes_nothing(tmp_path: Path) -> None:
    run = _interrupted_run(tmp_path)
    before = _tree(run.run_dir, run.profile.storage.logs_root)
    submitted = len(run.scheduler.specs)

    plan = run.adapter.plan_resume(run.run_id, AttemptGrants(), dry_run=True)

    assert plan.context.salvage_pending
    assert plan.context.attempt == 2
    assert _tree(run.run_dir, run.profile.storage.logs_root) == before
    assert len(run.scheduler.specs) == submitted


@pytest.mark.parametrize("refusal", ["active", "name", "lock", "multi-node"])
def test_resume_is_refused_while_the_run_may_still_be_live(
    tmp_path: Path, refusal: str
) -> None:
    run = _interrupted_run(tmp_path)
    if refusal == "active":
        run.scheduler.active = True
    elif refusal == "name":
        run.scheduler.name_taken = True
    elif refusal == "multi-node":
        manifest = json.loads((run.run_dir / "RUN_INFO.json").read_text())
        manifest["multi_node"] = {"total_nodes": 2}
        (run.run_dir / "RUN_INFO.json").write_text(json.dumps(manifest))
    before = _tree(run.run_dir, run.profile.storage.logs_root)
    expected = {
        "active": "still live in job 102",
        "name": "already active",
        "lock": "its Engine holds the lease",
        "multi-node": "multi-node",
    }[refusal]

    with pytest.raises((SetupError, InfrastructureError), match=expected):
        if refusal == "lock":
            with run.store.lock(run.run_id):
                run.adapter.plan_resume(run.run_id, AttemptGrants())
        else:
            run.adapter.plan_resume(run.run_id, AttemptGrants())

    assert _tree(run.run_dir, run.profile.storage.logs_root) == before


def test_resume_refuses_a_changed_frozen_control(tmp_path: Path) -> None:
    run = _interrupted_run(tmp_path)
    (run.run_dir / "control" / "task" / "instruction.md").write_text("changed\n")

    with pytest.raises(SetupError, match="frozen control .* changed: task/"):
        run.adapter.plan_resume(run.run_id, AttemptGrants())


def test_resume_refuses_a_file_added_to_the_frozen_source(tmp_path: Path) -> None:
    run = _interrupted_run(tmp_path)
    source = run.run_dir / "control" / "engine-source" / "src"
    (source / "__pycache__").mkdir(exist_ok=True)
    (source / "__pycache__" / "x.cpython-312.pyc").write_bytes(b"bytecode")
    (source / "sitecustomize.py").write_text("print('stray')\n")

    with pytest.raises(SetupError, match="changed: engine-source/src/sitecust"):
        run.adapter.plan_resume(run.run_id, AttemptGrants())


def test_resume_refuses_another_agent_cli_version(tmp_path: Path) -> None:
    """The job would refuse it after the allocation, costing an attempt."""
    run = _interrupted_run(tmp_path)
    run.adapter.binary_version = lambda _path: "0.150.0"
    before = _tree(run.run_dir, run.profile.storage.logs_root)
    submitted = len(run.scheduler.specs)

    for dry_run in (True, False):
        with pytest.raises(SetupError, match="0.149.0, which is now 0.150.0"):
            run.adapter.plan_resume(run.run_id, AttemptGrants(), dry_run=dry_run)

    assert _tree(run.run_dir, run.profile.storage.logs_root) == before
    assert len(run.scheduler.specs) == submitted


def test_resume_refuses_a_sif_rebuilt_since_the_run(tmp_path: Path) -> None:
    run = _interrupted_run(tmp_path)
    payload = load_engine_payload(run.run_dir / "control" / "run.json")
    rebuilt = b"rebuilt image bytes"
    payload.sif_path.write_bytes(rebuilt)
    payload.sif_sha256_path.write_text(
        f"{hashlib.sha256(rebuilt).hexdigest()}  {payload.sif_path.name}\n"
    )
    before = _tree(run.run_dir, run.profile.storage.logs_root)

    with pytest.raises(SetupError, match="evicted or rebuilt"):
        run.adapter.plan_resume(run.run_id, AttemptGrants())

    assert _tree(run.run_dir, run.profile.storage.logs_root) == before


def test_resume_closes_an_unpaused_attempt_at_the_job_end(tmp_path: Path) -> None:
    run = _interrupted_run(tmp_path)
    lease = run.store.read(run.run_id)
    run.store.write(
        lease.model_copy(
            update={
                "work": lease.work.model_copy(
                    update={"paused": False, "paused_at": None}
                )
            }
        )
    )
    run.scheduler.ended_at = run.paused_at + 100

    dry = run.adapter.plan_resume(run.run_id, AttemptGrants(), dry_run=True)

    plan = run.adapter.plan_resume(run.run_id, AttemptGrants())

    (attempt,) = run.store.read(run.run_id).attempts
    assert (attempt.ended_at, attempt.end_basis) == (run.paused_at + 100, "finished")
    assert attempt.agent_seconds == pytest.approx(700.0)
    # The dry run charges the open attempt to the job's end too, not to now.
    assert dry.context.agent_seconds_left == pytest.approx(
        plan.context.agent_seconds_left
    )
    assert dry.spec.walltime == plan.spec.walltime


def test_the_engine_runs_the_payload_attempt_in_resume_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On the compute node the Engine plans the attempt again and runs the
    coordinator in resume mode: the workspace is adopted, logs are reopened
    and the retained Work label continues with the attempt."""
    run = _interrupted_run(tmp_path)
    plan = run.adapter.plan_resume(run.run_id, AttemptGrants())
    run.adapter.resume(plan)
    payload = load_engine_payload(
        run.run_dir / "control" / "attempts" / "2" / "run.json"
    )
    # The fake job above ran attempt 2; put the run back as it was before it.
    store = run.store
    lease = store.read(run.run_id)
    assert lease.attempt == 1

    class Stop(Exception):
        pass

    captured: dict[str, object] = {}

    def coordinator(**kwargs):
        captured.update(kwargs)
        raise Stop

    node_tmp = tmp_path / "node"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    monkeypatch.setattr(runtime_module, "RunCoordinator", coordinator)
    monkeypatch.setattr(runtime_module, "_agent_provider_urls", lambda *a, **k: ())

    with pytest.raises(Stop):
        runtime_module.run_native_engine(payload, payload.run_plan)

    resume = captured["resume"]
    assert resume.attempt == 2
    assert resume.base_image_ref is None
    assert resume.next_round_id == "agent-4"  # attempt 2's fake round was agent-3
    backend = captured["backend"]
    composition = backend.artifact_starter.__self__
    assert composition.resume is resume
    run_plan = payload.run_plan
    volume = resume.workdir_volume
    assert backend.adopt_workdir_volume(run_plan, run.run_id, volume) == volume
    with pytest.raises(SetupError, match="kept workspace"):
        backend.adopt_workdir_volume(
            run_plan, run.run_id, volume.model_copy(update={"name": "bv-other"})
        )
    work = ContainerRef(container_id=f"bluevela-{run.run_id}-work", role="work")
    retained = backend.retain_work(
        run_plan, work, backend.plan_retained_work(run_plan, work)
    )
    assert retained.round_id == "final-a2"
    output = run.root / "agent_output.txt"
    before = output.read_text()
    composition.start_artifacts(run_plan, run.run_id)
    assert output.read_text().startswith(before)
    assert "attempt 2" in output.read_text()[len(before) :]

    # A rerun of attempt 1's job would start the run over.
    first = load_engine_payload(run.run_dir / "control" / "run.json")
    with pytest.raises(SetupError, match="already started"):
        runtime_module.run_native_engine(first, first.run_plan)

    # So would a SIF rebuilt since the attempt was submitted.
    sidecar = payload.sif_sha256_path.read_text()
    payload.sif_sha256_path.write_text("0" * 64 + sidecar[64:])
    with pytest.raises(SetupError, match="SIF .* was rebuilt"):
        runtime_module.run_native_engine(payload, payload.run_plan)
    payload.sif_sha256_path.write_text(sidecar)

    # A run that moved on since its attempt was submitted is refused.
    store.write(lease.model_copy(update={"status": None}))
    with pytest.raises(SetupError, match="changed after attempt 2"):
        runtime_module.run_native_engine(payload, payload.run_plan)
