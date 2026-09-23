from __future__ import annotations

import json
from importlib import resources

import pytest

from rsi_harness.cluster.bluevela.adapter import build_cluster_adapter
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.config import ClusterProfile, load_cluster_profile
from rsi_harness.cluster.schedulers.slurm import SlurmJobResult, SlurmScheduler
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import GPURequirement, RunStatus
from tests.cluster.bluevela.test_adapter import RecordingScheduler, _profile, _request


class RecordingSlurmScheduler(RecordingScheduler):
    def __init__(self):
        super().__init__()
        self.renderer = SlurmScheduler()

    def wait(self, job_id, **kwargs):
        super().wait(job_id, **kwargs)
        return SlurmJobResult(job_id=job_id, state="COMPLETED", exit_code=0)


def test_packaged_profile_dispatches_and_round_trips():
    profile = load_cluster_profile("slurm", {"USER": "alice"})
    assert profile.scheduler.queue == "gpu"
    assert profile.scheduler.group == "rsi"
    assert profile.scheduler.remote_binary == "srun"
    assert "--overlap" in profile.scheduler.remote_args
    assert ClusterProfile.model_validate_json(profile.model_dump_json()) == profile
    adapter = build_cluster_adapter("slurm")
    assert isinstance(adapter, SlurmClusterAdapter)
    assert isinstance(adapter.scheduler, SlurmScheduler)


def test_profile_requires_matching_adapter_and_scheduler(tmp_path):
    text = (
        resources.files("rsi_harness.cluster.slurm")
        .joinpath("profile.toml")
        .read_text()
    )
    path = tmp_path / "profile.toml"
    path.write_text(text.replace('adapter = "slurm"', 'adapter = "bluevela"'))
    with pytest.raises(SetupError, match="requires lsf scheduler"):
        load_cluster_profile(path)


@pytest.mark.parametrize("multinode", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_slurm_reuses_build_run_and_artifact_lifecycle(
    tmp_path, monkeypatch, multinode, dry_run
):
    local = _profile(tmp_path)
    profile = load_cluster_profile("slurm", {"USER": "alice"}).model_copy(
        update={
            "storage": local.storage,
            "apptainer": local.apptainer,
            "builder": local.builder,
        }
    )
    scheduler = RecordingSlurmScheduler()
    events = {}
    adapter = SlurmClusterAdapter(
        profile,
        scheduler=scheduler,
        event_callback=lambda key, value: events.update({key: value}),
        agent_version_resolver=lambda _: "0.149.0",
    )
    request = _request(tmp_path, dry_run=dry_run)
    definition = adapter._compile(request)
    if multinode:
        definition = definition.model_copy(
            update={
                "gpu_requirement": GPURequirement(count=32),
                "verifier": definition.verifier.model_copy(update={"gpu_count": 16}),
            }
        )
    monkeypatch.setattr(adapter, "_compile", lambda _: definition)
    # Capacity is covered separately by the shared Blue Vela tests.
    monkeypatch.setattr(adapter, "_require_shared_workspace", lambda *_: None)
    result = adapter.run(request)
    if dry_run:
        assert scheduler.events == []
        assert result.status == RunStatus.PREPARING
        assert not profile.storage.run_root.exists()
        argv = events["dry_run"]["run_argv"]
        assert argv[0] == "sbatch"
        assert f"--nodes={6 if multinode else 1}" in argv
        assert f"--gres=gpu:{8 if multinode else 4}" in argv
        assert "--time=135" in argv
        return
    assert result.status == RunStatus.COMPLETED
    assert scheduler.events == [
        "check-build",
        "submit-build",
        "wait-build",
        "check-run",
        "submit-run",
        "wait-run",
    ]
    assert result.job_ids == ("101", "102")
    assert scheduler.specs[0].gpu_count == 0
    assert scheduler.specs[0].stdout_path.name == "slurm.%j.out"
    payload = load_engine_payload(scheduler.specs[1].script_path.with_suffix(".json"))
    assert payload.profile.scheduler.kind == "slurm"
    script = scheduler.specs[1].script_path.read_text()
    assert "LSB_MCPU_HOSTS" not in script
    assert ("SLURM_JOB_NODELIST" in script) is multinode
    manifest = json.loads((payload.run_plan.paths.root / "RUN_INFO.json").read_text())
    assert manifest["state"] == "completed"
    if multinode:
        assert manifest["multi_node"]["total_nodes"] == 6
        assert payload.multi_node.work.node_count == 4
        assert payload.multi_node.verifier.node_count == 2


@pytest.mark.parametrize(
    ("state", "code"), [("FAILED", 1), ("CANCELLED", 0), ("COMPLETED", 9)]
)
def test_failed_slurm_jobs_cannot_pass_artifact_validation(state, code):
    with pytest.raises(InfrastructureError):
        SlurmClusterAdapter._require_success(
            "run",
            SlurmJobResult(job_id="123", state=state, exit_code=code),
        )
