from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import build_cluster_adapter
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.bluevela.resources import derive_resource_plan
from rsi_harness.cluster.config import (
    ClusterProfile,
    PBSBuildProfile,
    load_cluster_profile,
)
from rsi_harness.cluster.pbs.adapter import PBSClusterAdapter
from rsi_harness.cluster.schedulers.batch import BatchJobResult
from rsi_harness.cluster.schedulers.pbs import PBSJobSpec, PBSScheduler
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import CompileOptions, GPURequirement, RunStatus
from tests.cluster.bluevela.test_adapter import (
    RecordingScheduler,
    _profile,
    _request,
)
from tests.cluster.bluevela.test_adapter import (
    _fake_codex_runtime as _fake_codex_runtime,
)
from tests.factories import DEFAULT_TASK_TOML, write_harbor_task

ENV = {"USER": "alice", "HOME": "/home/alice"}


class RecordingBatchScheduler(RecordingScheduler):
    """Records the adapter's calls; renders with the real scheduler."""

    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def wait(self, job_id, **kwargs):
        super().wait(job_id, **kwargs)
        return BatchJobResult(job_id=job_id, state="COMPLETED", exit_code=0)


def local_profile(tmp_path, name: str, **scheduler) -> ClusterProfile:
    """A packaged profile with everything the jobs use under tmp_path."""
    local = _profile(tmp_path)
    base = load_cluster_profile(name, ENV)
    return base.model_copy(
        update={
            "scheduler": type(base.scheduler).model_validate(
                {**base.scheduler.model_dump(), **scheduler}
            ),
            "storage": local.storage,
            "apptainer": local.apptainer,
            "builder": local.builder.model_copy(
                update={
                    "cpu_slots": base.builder.cpu_slots,
                    "memory_mb": base.builder.memory_mb,
                }
            ),
        }
    )


def _adapter(profile, events=None):
    events = {} if events is None else events
    return PBSClusterAdapter(
        profile,
        scheduler=RecordingBatchScheduler(PBSScheduler(token=lambda: "a1b2c3")),
        event_callback=lambda key, value: events.update({key: value}),
        agent_version_resolver=lambda _: "0.149.0",
    )


def _dry_run(profile, request):
    events: dict = {}
    _adapter(profile, events).run(request)
    return events["dry_run"]


def test_packaged_profiles_dispatch_and_round_trip(monkeypatch):
    profile = load_cluster_profile("pbs", ENV)
    assert profile.adapter == "pbs" and profile.scheduler.kind == "pbs"
    assert profile.scheduler.queue == "gpu"
    assert profile.scheduler.group == "" and profile.scheduler.project is None
    assert profile.scheduler.chunk_resources == ("ncpus", "mem", "ngpus")
    assert profile.scheduler.select_binary == "qselect"
    assert profile.resources.node_tmp_variable is None
    assert ClusterProfile.model_validate_json(profile.model_dump_json()) == profile
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    adapter = build_cluster_adapter("pbs")
    assert isinstance(adapter, PBSClusterAdapter)
    assert isinstance(adapter.scheduler, PBSScheduler)


def test_flow2_shares_one_gpu_per_job_on_lustre():
    profile = load_cluster_profile("flow2", {"USER": "a792144"})
    assert profile.name == "flow2" and profile.adapter == "pbs"
    assert profile.scheduler.queue == "t2-share"
    assert profile.scheduler.chunk_resources == ("ngpus",)
    assert profile.scheduler.exclusive is False
    assert profile.resources.gpus_per_node == 1
    assert str(profile.storage.run_root) == "/data/a792144/rsi/runs"
    assert str(profile.builder.podman_bundle) == "/data/a792144/rsi/podman"
    assert str(profile.apptainer.binary) == "/data/a792144/rsi/apptainer/bin/apptainer"
    assert ClusterProfile.model_validate_json(profile.model_dump_json()) == profile


def test_packaged_profiles_need_their_variables():
    with pytest.raises(SetupError, match="missing environment variable USER"):
        load_cluster_profile("flow2", {})


@pytest.mark.parametrize("name", ["flow2", "flow2-node"])
def test_flow2_profiles_supply_native_rootless_build_helpers(name):
    profile = load_cluster_profile(name, {"USER": "alice"})
    utils = Path("/data/alice/rsi/apptainer/aarch64/utils")
    assert profile.builder.rootless_apt_sandbox is True
    assert profile.builder.faked_binary == utils / "bin/faked-sysv"
    assert (
        profile.builder.fakeroot_library
        == utils / "lib/libfakeroot/libfakeroot-sysv.so"
    )


def _packaged_text(name="profile.toml"):
    return resources.files("rsi_harness.cluster.pbs").joinpath(name).read_text()


def test_profile_requires_the_pbs_scheduler(tmp_path):
    path = tmp_path / "profile.toml"
    path.write_text(_packaged_text().replace('adapter = "pbs"', 'adapter = "slurm"'))
    with pytest.raises(SetupError, match="requires slurm scheduler"):
        load_cluster_profile(path, ENV)


@pytest.mark.parametrize(
    "line",
    [
        'excluded_hosts = ["gpu7"]',
        'chunk_extra = ["host_type"]',
        'chunk_extra = ["a=1", "a=2"]',
        'chunk_extra = ["a=1:b=2"]',
        'chunk_resources = ["ncpus", "scratch"]',
        'run_walltime_limit = "12"',
        "unknown = 1",
    ],
)
def test_profile_refuses_what_pbs_jobs_cannot_honour(tmp_path, line):
    path = tmp_path / "profile.toml"
    path.write_text(
        _packaged_text().replace('kind = "pbs"\n', f'kind = "pbs"\n{line}\n', 1)
    )
    with pytest.raises(SetupError, match="invalid cluster profile"):
        load_cluster_profile(path, ENV)


def test_profile_reads_account_and_build_overrides(tmp_path):
    path = tmp_path / "profile.toml"
    text = _packaged_text().replace(
        'kind = "pbs"\n',
        'kind = "pbs"\naccount = "rsi"\nproject = "p-rsi"\n'
        'chunk_extra = ["host_type=gpu"]\n',
        1,
    )
    path.write_text(text + '\n[scheduler.build]\nqueue = "cpu"\ngpus = 1\n')
    profile = load_cluster_profile(path, ENV)
    assert profile.scheduler.group == "rsi"
    assert profile.scheduler.project == "p-rsi"
    assert profile.scheduler.chunk_extra == ("host_type=gpu",)
    assert profile.scheduler.build == PBSBuildProfile(queue="cpu", gpus=1)


@pytest.mark.parametrize("dry_run", [False, True])
def test_pbs_reuses_build_run_and_artifact_lifecycle(tmp_path, dry_run):
    profile = local_profile(tmp_path, "pbs")
    # Submission and dry-run must work without compute-node software locally.
    profile.apptainer.binary.unlink()
    events: dict = {}
    adapter = _adapter(profile, events)
    result = adapter.run(_request(tmp_path, dry_run=dry_run))
    scheduler = adapter.scheduler
    if dry_run:
        assert scheduler.events == []
        assert result.status == RunStatus.PREPARING
        assert not profile.storage.run_root.exists()
        details = events["dry_run"]
        assert details["adapter"] == "pbs"
        run = details["run_argv"]
        assert run[:3] == ("qsub", "-q", "gpu")
        assert "select=1:ncpus=8:mem=65536mb:ngpus=4" in run
        assert "walltime=02:15:00" in run
        assert "-V" not in run
        # A build asks for no GPU.
        build = details["build_argv"]
        assert not any("ngpus" in arg for arg in build)
        assert any(arg.startswith("select=1:ncpus=") for arg in build)
        assert details["manifests"] is None
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
    build, run = scheduler.specs
    assert isinstance(build, PBSJobSpec) and isinstance(run, PBSJobSpec)
    assert build.gpu_count == 0 and run.gpu_count == 4
    assert build.stdout_path.name == "pbs.%j.out"
    assert run.stderr_path.name == "pbs.%j.err"
    payload = load_engine_payload(run.script_path.with_suffix(".json"))
    assert payload.profile.scheduler.kind == "pbs"
    manifest = json.loads((payload.run_plan.paths.root / "RUN_INFO.json").read_text())
    assert manifest["state"] == "completed"


def test_the_site_chunk_settings_reach_every_job(tmp_path):
    profile = local_profile(
        tmp_path,
        "pbs",
        chunk_resources=["ngpus"],
        chunk_extra=["host_type=gpu"],
        group="rsi",
        project="p-rsi",
        exclusive=True,
    )
    details = _dry_run(profile, _request(tmp_path, dry_run=True))
    run, build = details["run_argv"], details["build_argv"]
    assert "select=1:ngpus=4:host_type=gpu" in run
    assert "select=1:host_type=gpu" in build
    for argv in (run, build):
        pairs = set(zip(argv, argv[1:], strict=False))
        assert {("-A", "rsi"), ("-P", "p-rsi")} <= pairs
    assert "place=excl" in run and "place=excl" not in build


def test_build_overrides_reach_only_the_build_job(tmp_path):
    profile = local_profile(
        tmp_path,
        "pbs",
        build={
            "queue": "cpu",
            "account": "build-acct",
            "project": "build-p",
            "gpus": 1,
            "exclusive": True,
        },
    )
    details = _dry_run(profile, _request(tmp_path, dry_run=True))
    build, run = details["build_argv"], details["run_argv"]
    assert build[:3] == ("qsub", "-q", "cpu")
    assert any(arg.startswith("select=1:") and "ngpus=1" in arg for arg in build)
    build_pairs = set(zip(build, build[1:], strict=False))
    assert {("-A", "build-acct"), ("-P", "build-p"), ("-l", "place=excl")} <= (
        build_pairs
    )
    assert run[:3] == ("qsub", "-q", "gpu")
    assert "-A" not in run and "-P" not in run and "place=excl" not in run
    assert "select=1:ncpus=8:mem=65536mb:ngpus=4" in run


@pytest.mark.parametrize("shape", ["gpus", "judge-gpus", "disjoint"])
def test_tasks_that_need_several_nodes_are_refused(tmp_path, monkeypatch, shape):
    adapter = _adapter(local_profile(tmp_path, "pbs"))
    request = _request(tmp_path, dry_run=True)
    definition = adapter._compile(request)
    updates = {
        "gpus": {"gpu_requirement": GPURequirement(count=32)},
        "judge-gpus": {
            "verifier": definition.verifier.model_copy(update={"gpu_count": 8})
        },
        "disjoint": {"require_disjoint_phase_nodes": True},
    }[shape]
    big = definition.model_copy(update=updates)
    monkeypatch.setattr(adapter.compiler, "compile", lambda *_args, **_kw: big)
    refusal = (
        "PBS runs use one job on one node"
        if shape == "disjoint"
        else "GPUs on one node; this profile's PBS jobs have"
    )
    with pytest.raises(SetupError, match=refusal):
        adapter.run(request)
    assert adapter.scheduler.events == []


def test_work_and_judge_that_do_not_fit_side_by_side_share_one_node(tmp_path):
    task_dir = write_harbor_task(
        tmp_path / "shared-gpus",
        task_toml=(
            DEFAULT_TASK_TOML.replace("gpus = 2", "gpus = 3")
            + "\n[metadata.rsi_harness.verifier]\ngpus = 2\n"
        ),
    )
    request = _request(tmp_path, dry_run=True).model_copy(
        update={"task_dir": task_dir.resolve()}
    )
    details = _dry_run(local_profile(tmp_path, "pbs"), request)
    assert details["multi_node"] is None
    assert details["resources"]["total_gpus"] == 3
    assert "select=1:ncpus=8:mem=65536mb:ngpus=3" in details["run_argv"]


def test_flow2_refuses_a_task_wider_than_its_one_gpu_share(tmp_path):
    adapter = _adapter(local_profile(tmp_path, "flow2"))
    with pytest.raises(SetupError, match="needs 2 GPUs on one node.* have 1"):
        adapter.run(_request(tmp_path, dry_run=True))


def test_flow2_node_runs_wider_tasks_on_whole_nodes_and_builds_in_t2_share(tmp_path):
    details = _dry_run(
        local_profile(tmp_path, "flow2-node"), _request(tmp_path, dry_run=True)
    )
    assert details["resources"]["total_gpus"] == 4
    assert "-q t2-single -l select=1 " in " ".join(details["run_argv"]) + " "
    assert "-q t2-share -l select=1:ngpus=1 " in " ".join(details["build_argv"]) + " "


@pytest.mark.parametrize("work_gpus,judge_gpus", [(0, 0), (1, 0), (0, 1)])
def test_cpu_phases_keep_their_resources(tmp_path, work_gpus, judge_gpus):
    task_dir = write_harbor_task(
        tmp_path / "cpu-phases",
        task_toml=(
            DEFAULT_TASK_TOML.replace("gpus = 2", f"gpus = {work_gpus}")
            + f"\n[metadata.rsi_harness.verifier]\ngpus = {judge_gpus}\n"
        ),
    )
    request = _request(tmp_path, dry_run=True).model_copy(
        update={"task_dir": task_dir.resolve()}
    )
    details = _dry_run(local_profile(tmp_path, "pbs"), request)
    assert details["resources"]["total_gpus"] == work_gpus + judge_gpus
    select = next(arg for arg in details["run_argv"] if arg.startswith("select="))
    assert ("ngpus" in select) is bool(work_gpus + judge_gpus)


def test_oracle_runs_on_pbs(tmp_path, monkeypatch):
    task = write_harbor_task(
        tmp_path / "task",
        task_toml=DEFAULT_TASK_TOML.replace("gpus = 2", "gpus = 1"),
    )
    (task / "solution").mkdir()
    (task / "solution/solve.sh").write_text("#!/bin/sh\nprintf 42 > answer\n")
    profile = local_profile(tmp_path, "pbs")
    profile.apptainer.binary.unlink()
    monkeypatch.setattr("shutil.which", lambda _: None)
    events: dict = {}
    adapter = PBSClusterAdapter(
        profile,
        scheduler=RecordingBatchScheduler(PBSScheduler()),
        agent_version_resolver=lambda _: pytest.fail("Oracle queried a model CLI"),
        event_callback=lambda key, value: events.update({key: value}),
    )
    adapter.run(
        ClusterRunRequest(
            task_dir=task.resolve(),
            agent_name="oracle",
            options=CompileOptions(agent_name="oracle"),
            logs_root=tmp_path / "logs",
            dry_run=True,
        )
    )
    assert adapter.scheduler.events == []
    assert events["dry_run"]["agent_version"] is None
    assert events["dry_run"]["run_argv"][0] == "qsub"


def test_a_walltime_limit_segments_with_the_wrappers_advance_term(tmp_path):
    profile = local_profile(tmp_path, "pbs", run_walltime_limit="01:45")
    adapter = _adapter(profile)
    request = _request(tmp_path, dry_run=True)
    definition = adapter._compile(request)
    resources = derive_resource_plan(definition, profile).single_node
    drain = int(
        definition.verifier.timeout_seconds + profile.resources.walltime_margin_seconds
    )
    run = adapter._run_spec(
        tmp_path, "run", resources, tmp_path / "run.sh", judge_drain_seconds=drain
    )
    assert isinstance(run, PBSJobSpec)
    assert (run.walltime, run.signal_seconds) == ("01:45", drain)
    argv = adapter.scheduler.renderer.render_submit(run)
    assert "walltime=01:45:00" in argv
    script = adapter.scheduler.renderer._job_script(run, tmp_path / "environment")
    assert f"sleep {105 * 60 - drain} " in script


def test_failed_jobs_fail_the_run_with_their_log(tmp_path):
    (tmp_path / "pbs.96979.jobm.err").write_text("build hint: set builder.conmon\n")
    with pytest.raises(InfrastructureError) as caught:
        PBSClusterAdapter._require_success(
            "SIF build",
            BatchJobResult(job_id="96979.jobm", state="TIMEOUT", exit_code=1),
            tmp_path / "pbs.%j.err",
        )
    assert "TIMEOUT" in str(caught.value)
    assert str(caught.value).endswith(
        f"; see {tmp_path}/pbs.96979.jobm.err (build hint: set builder.conmon)"
    )
    PBSClusterAdapter._require_success(
        "run", BatchJobResult(job_id="1.s", state="COMPLETED", exit_code=0)
    )
