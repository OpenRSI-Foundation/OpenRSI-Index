from __future__ import annotations

import json
import platform
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


def test_flow2_controllers_must_run_on_the_aarch64_gpu_nodes(tmp_path, monkeypatch):
    # The Engine runs on the GPU nodes with the controller's Python, so a
    # controller on an x86-64 login node is refused before any job; a dry run
    # still plans there.
    for name in ("flow2", "flow2-node"):
        assert load_cluster_profile(name, ENV).resources.node_machine == "aarch64"
    profile = local_profile(tmp_path, "flow2-node")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    adapter = _adapter(profile)
    with pytest.raises(SetupError, match="compute nodes are aarch64.* runs on x86_64"):
        adapter.run(_request(tmp_path, dry_run=False))
    assert adapter.scheduler.events == [] and adapter.scheduler.specs == []
    assert _dry_run(profile, _request(tmp_path, dry_run=True))


def test_profiles_without_a_node_machine_accept_any_controller(tmp_path, monkeypatch):
    profile = local_profile(tmp_path, "pbs")
    assert profile.resources.node_machine is None
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    adapter = _adapter(profile)
    adapter._check_controller_machine()


def test_an_interrupt_between_submission_and_wait_cancels_the_job(tmp_path):
    # A SIGTERM (as KeyboardInterrupt) that lands after qsub returned but
    # before the wait began would otherwise leave the run job running.
    cancelled = []

    def interrupted(key, value):
        if key == "job_submitted" and value["stage"] == "run":
            raise KeyboardInterrupt

    adapter = PBSClusterAdapter(
        local_profile(tmp_path, "pbs"),
        scheduler=RecordingBatchScheduler(PBSScheduler(token=lambda: "a1b2c3")),
        event_callback=interrupted,
        agent_version_resolver=lambda _: "0.149.0",
    )
    adapter.scheduler.cancel = cancelled.append
    with pytest.raises(KeyboardInterrupt):
        adapter.run(_request(tmp_path, dry_run=False))
    assert adapter.scheduler.events[-1] == "submit-run"
    assert cancelled == [str(100 + len(adapter.scheduler.specs))]


def test_an_interrupted_wait_is_not_cancelled_again(tmp_path):
    # The wait cancels on its own interrupt; LSF and Slurm refuse a second
    # cancel of an ended job, which would hide the interrupt.
    cancelled = []

    class Interrupted(RecordingBatchScheduler):
        def wait(self, job_id, **kwargs):
            if self.events[-1] == "submit-run":
                raise KeyboardInterrupt(f"interrupted; job {job_id} cancelled")
            return super().wait(job_id, **kwargs)

    adapter = PBSClusterAdapter(
        local_profile(tmp_path, "pbs"),
        scheduler=Interrupted(PBSScheduler(token=lambda: "a1b2c3")),
        event_callback=lambda *_: None,
        agent_version_resolver=lambda _: "0.149.0",
    )
    adapter.scheduler.cancel = cancelled.append
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        adapter.run(_request(tmp_path, dry_run=False))
    assert cancelled == []


def test_a_failed_cancel_after_an_interrupt_is_reported(tmp_path):
    def interrupted(key, value):
        if key == "job_submitted" and value["stage"] == "run":
            raise KeyboardInterrupt

    def cancel(job_id):
        raise InfrastructureError(f"cannot cancel PBS job {job_id}: no server")

    profile = local_profile(tmp_path, "pbs")
    adapter = PBSClusterAdapter(
        profile,
        scheduler=RecordingBatchScheduler(PBSScheduler(token=lambda: "a1b2c3")),
        event_callback=interrupted,
        agent_version_resolver=lambda _: "0.149.0",
    )
    adapter.scheduler.cancel = cancel
    with pytest.raises(KeyboardInterrupt) as raised:
        adapter.run(_request(tmp_path, dry_run=False))
    job_id = str(100 + len(adapter.scheduler.specs))
    message = (
        f"interrupted; cancelling job {job_id} failed and it may still run: "
        f"cannot cancel PBS job {job_id}: no server"
    )
    assert str(raised.value) == message
    (manifest,) = profile.storage.run_root.glob("*/RUN_INFO.json")
    assert json.loads(manifest.read_text())["error"] == message


def test_resume_refuses_another_architecture_before_closing_the_run(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    adapter = _adapter(local_profile(tmp_path, "flow2-node"))
    with pytest.raises(SetupError, match="compute nodes are aarch64"):
        adapter.plan_resume("20261106T000000Z-task-0123abcd", None)


# --- Allocation launch ------------------------------------------------------


def _allocation_adapter(profile, events=None):
    events = {} if events is None else events
    return PBSClusterAdapter(
        profile,
        event_callback=lambda key, value: events.update({key: value}),
        agent_version_resolver=lambda _: "0.149.0",
    )


def test_allocation_launch_plans_local_jobs_anywhere(tmp_path, monkeypatch):
    from rsi_harness.cluster.schedulers.allocation import AllocationScheduler

    monkeypatch.delenv("PBS_JOBID", raising=False)
    profile = local_profile(tmp_path, "pbs", launch="allocation")
    assert profile.scheduler.whole_node
    events: dict = {}
    adapter = _allocation_adapter(profile, events)
    assert isinstance(adapter.scheduler, AllocationScheduler)
    assert isinstance(adapter.scheduler.host, PBSScheduler)
    adapter.run(_request(tmp_path, dry_run=True))
    run = events["dry_run"]["run_argv"]
    assert run[1] == "-I" and Path(run[2]).name == "local_job.py"
    assert "qsub" not in run
    assert not profile.storage.run_root.exists()


def test_allocation_launch_is_refused_outside_a_pbs_job(tmp_path, monkeypatch):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    profile = local_profile(tmp_path, "pbs", launch="allocation")
    with pytest.raises(SetupError, match="inside a PBS job"):
        _allocation_adapter(profile).run(_request(tmp_path, dry_run=False))
    assert not profile.storage.run_root.exists()


def test_allocation_launch_checks_the_jobs_walltime_before_the_run(
    tmp_path, monkeypatch
):
    import subprocess
    import time
    from datetime import datetime

    monkeypatch.setenv("PBS_JOBID", "97400.jobm")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    profile = local_profile(tmp_path, "pbs", launch="allocation")
    adapter = _allocation_adapter(profile)
    calls = []
    started = datetime.fromtimestamp(time.time() - 60)

    def qstat(argv):
        calls.append(argv)
        job = {
            "job_state": "R",
            "stime": started.strftime("%a %b %d %H:%M:%S %Y"),
            "Resource_List": {"walltime": "01:00:00"},
        }
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"Jobs": {argv[-1]: job}}), ""
        )

    adapter.scheduler.host._runner = qstat
    with pytest.raises(SetupError, match="PBS job 97400.jobm has 0.98 h"):
        adapter.run(_request(tmp_path, dry_run=False))
    # Nothing was submitted, nothing created.
    assert {argv[0] for argv in calls} == {"qstat"}
    assert not profile.storage.run_root.exists()


@pytest.mark.parametrize("launch", ["submit", "allocation"])
def test_the_payload_carries_the_launch_only_when_set(tmp_path, launch):
    profile = local_profile(tmp_path, "pbs", launch=launch)
    adapter = _adapter(profile)
    adapter.run(_request(tmp_path, dry_run=False))
    payload_path = adapter.scheduler.specs[-1].script_path.with_suffix(".json")
    scheduler = json.loads(payload_path.read_text())["profile"]["scheduler"]
    assert scheduler.get("launch") == (None if launch == "submit" else launch)
    # The controller's own job may hold more GPUs than the run plans.
    loaded = load_engine_payload(payload_path).profile.scheduler
    assert loaded.whole_node is (launch == "allocation")


def test_a_local_job_is_read_whatever_the_launch(tmp_path):
    """A run begun with allocation launch resumes without it: its local
    job's record tells, and a submitted job is the scheduler's."""
    import subprocess
    import time

    from rsi_harness.cluster.schedulers.allocation import LocalJob

    adapter = _allocation_adapter(local_profile(tmp_path, "pbs"))
    assert isinstance(adapter.scheduler, PBSScheduler)
    calls = []

    def qstat(argv):
        calls.append(argv)
        job = {"job_state": "R"}
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"Jobs": {argv[-1]: job}}), ""
        )

    adapter.scheduler._runner = qstat
    job_id = "97400.jobm.local-ab12cd34"
    records = adapter.profile.storage.run_root / ".local-jobs"
    records.mkdir(parents=True)
    end = tmp_path / "end"
    job = LocalJob(
        job_id=job_id,
        name="rsi-task-run-alice",
        allocation="97400.jobm",
        host="another-node",
        pid=1,
        start_ticks=None,
        deadline=time.time() + 3600,
        end_path=end,
        heartbeat_path=tmp_path / "heartbeat",
    )
    (records / f"{job_id}.json").write_text(job.model_dump_json())
    # On another node, it runs while its batch job does.
    assert adapter._job_finished(job_id) is None
    assert calls[-1][-1] == "97400.jobm"
    end.write_text(f"0 {time.time():.3f} done\n")
    assert adapter._job_finished(job_id).state == "COMPLETED"
    assert adapter._job_finished("97401.jobm") is None
    assert calls[-1][-1] == "97401.jobm"


def test_allocation_resume_needs_a_frozen_engine_that_knows_it(tmp_path):
    from tests.cluster.bluevela.test_engine import _payload

    source = tmp_path / "engine-source"
    payload = _payload(tmp_path).model_copy(update={"source_root": source})
    with pytest.raises(SetupError, match='resume it with scheduler.launch = "submit"'):
        _adapter(
            local_profile(tmp_path, "flow2", launch="allocation")
        )._check_frozen_source(payload)
    _adapter(local_profile(tmp_path, "flow2"))._check_frozen_source(payload)
    allocation = source / "src/rsi_harness/cluster/schedulers/allocation.py"
    allocation.parent.mkdir(parents=True)
    allocation.touch()
    _adapter(
        local_profile(tmp_path, "flow2", launch="allocation")
    )._check_frozen_source(payload)


@pytest.mark.parametrize(
    ("started", "resumed"), [("submit", "allocation"), ("allocation", "submit")]
)
def test_resume_runs_the_engine_as_the_current_launch(
    tmp_path, monkeypatch, started, resumed
):
    from rsi_harness.cluster.bluevela.adapter import (
        BlueVelaClusterAdapter,
        ClusterResumePlan,
    )
    from rsi_harness.cluster.bluevela.engine import render_engine_driver
    from tests.cluster.bluevela.test_engine import _payload

    # flow2 shares a node: a submitted job holds the planned GPUs alone.
    frozen = local_profile(tmp_path, "flow2", launch=started)
    payload = _payload(tmp_path).model_copy(update={"profile": frozen})
    monkeypatch.setattr(
        BlueVelaClusterAdapter,
        "plan_resume",
        lambda *_, **__: ClusterResumePlan(context=None, payload=payload, spec=None),
    )
    current = local_profile(tmp_path, "flow2", launch=resumed)
    plan = _adapter(current).plan_resume("20261106T000000Z-task-0123abcd", None)
    assert plan.payload.profile.scheduler.whole_node is (resumed == "allocation")
    output = tmp_path / "attempt" / "run.sh"
    output.parent.mkdir()
    render_engine_driver(plan.payload, current, output)
    loaded = load_engine_payload(output.with_suffix(".json")).profile.scheduler
    assert loaded.launch == resumed
