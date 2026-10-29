"""Site policy changes scheduling, not task requirements or total budgets."""

import pytest
from pydantic import ValidationError

from rsi_harness.cluster.bluevela.engine import render_engine_driver
from rsi_harness.cluster.bluevela.image import plan_image, render_build_driver
from rsi_harness.cluster.bluevela.resources import derive_resource_plan
from rsi_harness.cluster.config import SlurmBuildProfile, SlurmSchedulerProfile
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import SetupError
from tests.cluster.bluevela.test_adapter import (
    _fake_codex_runtime as _fake_codex_runtime,
)
from tests.cluster.bluevela.test_adapter import _request
from tests.cluster.bluevela.test_engine import _cpu_payload
from tests.cluster.slurm.test_adapter import RecordingSlurmScheduler
from tests.cluster.slurm.test_oracle import setup_oracle


def _adapter(tmp_path, **policy):
    profile, _ = setup_oracle(tmp_path)
    profile = profile.model_copy(
        update={"scheduler": profile.scheduler.model_copy(update=policy)}
    )
    return SlurmClusterAdapter(
        profile,
        scheduler=RecordingSlurmScheduler(),
        agent_version_resolver=lambda _: "0.149.0",
    )


def _specs(adapter, tmp_path):
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    resources = derive_resource_plan(definition, adapter.profile).single_node
    assert resources is not None
    return (
        definition,
        resources,
        (
            adapter._build_spec(tmp_path, "build", resources, tmp_path / "build.sh"),
            adapter._run_spec(
                tmp_path,
                "run",
                resources,
                tmp_path / "run.sh",
                judge_drain_seconds=int(
                    definition.verifier.timeout_seconds
                    + adapter.profile.resources.walltime_margin_seconds
                ),
            ),
        ),
    )


def test_omit_scheduler_tmp_keeps_build_and_runtime_space_checks(tmp_path):
    adapter = _adapter(tmp_path, request_tmp=False)
    _, resources, specs = _specs(adapter, tmp_path)
    for spec in specs:
        assert spec.local_tmp_mb > 0
        assert not any(
            a.startswith("--tmp=") for a in adapter.scheduler.render_submit(spec)
        )
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    render_build_driver(
        plan_image(context, tmp_path / "cache"),
        adapter.profile,
        context,
        specs[0].script_path,
        specs[0].local_tmp_mb,
    )
    payload = _cpu_payload(tmp_path)
    payload = payload.model_copy(update={"resources": resources})
    render_engine_driver(payload, adapter.profile, specs[1].script_path)
    for spec in specs:
        script = spec.script_path.read_text()
        assert "df -Pk -- " in script
        assert f"required_tmp_kb=$(({spec.local_tmp_mb} * 1024))" in script


def test_build_overrides_do_not_change_work_or_judge_allocation(tmp_path):
    adapter = _adapter(
        tmp_path,
        build=SlurmBuildProfile(
            partition="build-gpu",
            account="build-account",
            qos="short",
            constraint="a100",
            gpus=1,
            exclusive=False,
        ),
    )
    _, resources, (build, run) = _specs(adapter, tmp_path)
    argv = adapter.scheduler.render_submit(build)
    for flag in (
        "--partition=build-gpu",
        "--account=build-account",
        "--qos=short",
        "--constraint=a100",
        "--gres=gpu:1",
    ):
        assert flag in argv
    assert run.queue == adapter.profile.scheduler.queue
    assert run.gpu_count == resources.total_gpus
    assert run.qos == adapter.profile.scheduler.qos
    assert build.cpu_slots == adapter.profile.builder.cpu_slots
    assert build.memory_mb == adapter.profile.builder.memory_mb


def test_run_limit_preserves_total_budget_and_reserves_judge_drain(tmp_path):
    adapter = _adapter(tmp_path, run_walltime_limit="01:45")
    definition, resources, (_, run) = _specs(adapter, tmp_path)
    assert resources.run_walltime != run.walltime
    assert run.walltime == "01:45"
    assert definition.agent.timeout_seconds == 3600
    assert run.signal_seconds == 4500
    assert run.signal_seconds >= definition.verifier.timeout_seconds
    assert f"--signal=B:TERM@{run.signal_seconds}" in adapter.scheduler.render_submit(
        run
    )
    # The final shorter attempt uses the remaining time, without an early stop.
    last = adapter._run_spec(
        tmp_path,
        "last",
        resources.model_copy(update={"run_walltime": "01:30"}),
        tmp_path / "last.sh",
    )
    assert last.walltime == "01:30"
    assert last.signal_seconds == 0


def test_too_short_limit_is_rejected_before_submission(tmp_path):
    adapter = _adapter(tmp_path, run_walltime_limit="00:01")
    with pytest.raises(SetupError, match="Judge drain reserve"):
        adapter.run(_request(tmp_path, dry_run=True))
    assert adapter.scheduler.events == []


def test_oracle_is_not_silently_truncated_to_fit_partition(tmp_path):
    adapter = _adapter(tmp_path, run_walltime_limit="00:01")
    # Reuse the already created oracle task rather than creating it twice.
    from rsi_harness.cluster.base import ClusterRunRequest
    from rsi_harness.models import CompileOptions

    request = ClusterRunRequest(
        task_dir=tmp_path / "task" / "minimal-gpu",
        agent_name="oracle",
        options=CompileOptions(agent_name="oracle"),
        logs_root=tmp_path / "logs",
        dry_run=True,
    )
    with pytest.raises(SetupError, match="Oracle and sandbox"):
        adapter.run(request)
    assert adapter.scheduler.events == []


@pytest.mark.parametrize("value", ["0:00", "00:61", "30", "-1:00"])
def test_walltime_limit_requires_unambiguous_positive_hours_minutes(value):
    with pytest.raises(ValidationError, match="positive HH:MM"):
        SlurmSchedulerProfile(
            kind="slurm", partition="general", run_walltime_limit=value
        )


def test_resume_keeps_remaining_budget_and_caps_only_next_allocation(tmp_path):
    from rsi_harness.runtime.recovery import AttemptGrants
    from tests.cluster.bluevela.test_resume import _interrupted_run

    run = _interrupted_run(tmp_path)
    scheduler = SlurmSchedulerProfile(
        kind="slurm", partition="general", run_walltime_limit="01:45"
    )
    adapter = SlurmClusterAdapter(
        run.profile.model_copy(update={"adapter": "slurm", "scheduler": scheduler}),
        scheduler=run.scheduler,
        binary_version=lambda _: "0.149.0",
    )
    plan = adapter.plan_resume(run.run_id, AttemptGrants(), dry_run=True)
    assert plan.context.grants.add_time_seconds == 0
    assert 0 < plan.context.agent_seconds_left < 3600
    assert plan.payload.resume.agent_seconds_left == plan.context.agent_seconds_left
    assert plan.spec.walltime == "01:45"
    assert plan.spec.signal_seconds == 4500


def test_segment_plan_that_exhausts_attempt_limit_is_rejected_before_launch(tmp_path):
    from rsi_harness.models import CompileOptions

    adapter = _adapter(tmp_path, run_walltime_limit="02:00")
    request = _request(tmp_path, dry_run=True).model_copy(
        update={"options": CompileOptions(agent_timeout_seconds=24 * 3600)}
    )
    # 45-minute Judge + 30-minute site margin + early signal leave at most
    # 44 planned Work minutes per interrupted allocation.
    with pytest.raises(SetupError, match="only 20 of 20 attempts remain"):
        adapter.run(request)
    assert adapter.scheduler.events == []


@pytest.mark.parametrize("attempt", [1, 19, 20])
def test_segment_capacity_boundary_uses_remaining_shared_attempt_limit(
    tmp_path, attempt
):
    from rsi_harness.runtime.resume import MAX_ATTEMPTS

    adapter = _adapter(tmp_path, run_walltime_limit="01:45")
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    # 105min - (45min Judge + 30min overhead) = 1800s on the final job;
    # the earlier jobs also reserve Slurm's possible 60s early delivery.
    remaining = MAX_ATTEMPTS - attempt + 1
    capacity = 1800 + (remaining - 1) * 1740
    adapter._validate_segment_budget(
        definition, agent_seconds=capacity, attempt=attempt
    )
    with pytest.raises(SetupError, match=f"only {remaining} of {MAX_ATTEMPTS}"):
        adapter._validate_segment_budget(
            definition, agent_seconds=capacity + 0.01, attempt=attempt
        )


def test_resume_budget_refusal_precedes_salvage_and_preserves_evidence(tmp_path):
    from rsi_harness.runtime.recovery import AttemptGrants
    from tests.cluster.bluevela.test_resume import _interrupted_run, _tree

    run = _interrupted_run(tmp_path)
    adapter = SlurmClusterAdapter(
        run.profile.model_copy(
            update={
                "adapter": "slurm",
                "scheduler": SlurmSchedulerProfile(
                    kind="slurm", partition="general", run_walltime_limit="01:45"
                ),
            }
        ),
        scheduler=run.scheduler,
        binary_version=lambda _: "0.149.0",
    )
    before = _tree(run.run_dir, run.root)
    # Extra grants are part of the plan too: they cannot silently exceed
    # the remaining attempts even though this is only the first resume.
    with pytest.raises(SetupError, match="only 19 of 20 attempts remain"):
        adapter.plan_resume(run.run_id, AttemptGrants(add_time_seconds=86400))
    assert _tree(run.run_dir, run.root) == before


def test_final_allocation_uses_exact_budget_before_rounding_to_minutes(tmp_path):
    adapter = _adapter(tmp_path)
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    definition = definition.model_copy(
        update={
            "agent": definition.agent.model_copy(update={"timeout_seconds": 59.5}),
            "verifier": definition.verifier.model_copy(update={"timeout_seconds": 0.1}),
        }
    )
    profile = adapter.profile.model_copy(
        update={
            "scheduler": adapter.profile.scheduler.model_copy(
                update={"run_walltime_limit": "00:01"}
            ),
            "resources": adapter.profile.resources.model_copy(
                update={"walltime_margin_seconds": 0}
            ),
        }
    )
    adapter = SlurmClusterAdapter(profile, scheduler=adapter.scheduler)
    adapter._validate_segment_budget(definition, agent_seconds=59.5, attempt=20)
    resources = derive_resource_plan(definition, profile).single_node
    spec = adapter._run_spec(tmp_path, "last", resources, tmp_path / "run.sh")
    assert spec.walltime == "00:01"
    assert spec.signal_seconds == 0


def test_sandbox_resume_refusal_precedes_salvage(tmp_path, monkeypatch):
    from rsi_harness.cluster.bluevela import engine
    from rsi_harness.runtime.recovery import AttemptGrants
    from tests.cluster.bluevela.test_resume import _interrupted_run, _tree
    from tests.sandbox_helpers import make_env_task

    run = _interrupted_run(tmp_path)
    load_payload = engine.load_engine_payload

    def sandbox_payload(path):
        payload = load_payload(path)
        task = payload.run_plan.task.model_copy(update={"sandbox": make_env_task()})
        return payload.model_copy(
            update={"run_plan": payload.run_plan.model_copy(update={"task": task})}
        )

    # The frozen payload declares a sandbox; no live sandbox is needed to
    # test the adapter's ordering relative to its real recovery planner.
    monkeypatch.setattr(engine, "load_engine_payload", sandbox_payload)
    adapter = SlurmClusterAdapter(
        run.profile.model_copy(
            update={
                "adapter": "slurm",
                "scheduler": SlurmSchedulerProfile(
                    kind="slurm", partition="general", run_walltime_limit="01:45"
                ),
            }
        ),
        scheduler=run.scheduler,
        binary_version=lambda _: "0.149.0",
    )
    before = _tree(run.run_dir, run.root)
    with pytest.raises(SetupError, match="sandbox tasks cannot use segmented"):
        adapter.plan_resume(run.run_id, AttemptGrants())
    assert _tree(run.run_dir, run.root) == before


def test_multi_node_budget_cannot_be_silently_segmented(tmp_path):
    from tests.cluster.bluevela.test_engine import _multi_payload

    adapter = _adapter(tmp_path, run_walltime_limit="01:45")
    resources = _multi_payload(tmp_path).multi_node
    with pytest.raises(SetupError, match="multi-node runs cannot resume"):
        adapter._run_spec(tmp_path, "multi", resources, tmp_path / "run.sh")


def test_submit_only_policy_is_not_sent_to_older_frozen_engines(tmp_path):
    import json

    adapter = _adapter(
        tmp_path,
        request_tmp=False,
        run_walltime_limit="01:45",
        build=SlurmBuildProfile(partition="cpu"),
    )
    payload = _cpu_payload(tmp_path).model_copy(update={"profile": adapter.profile})
    path = render_engine_driver(payload, adapter.profile, tmp_path / "job.sh")
    serialized = json.loads(path.read_text())
    assert not {"request_tmp", "build", "run_walltime_limit"}.intersection(
        serialized["profile"]["scheduler"]
    )
    assert "judge_drain_seconds" not in serialized["resources"]
    assert adapter.profile.scheduler.request_tmp is False
