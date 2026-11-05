from __future__ import annotations

import json
import subprocess
from importlib import resources
from pathlib import Path

import pytest

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import build_cluster_adapter
from rsi_harness.cluster.bluevela.engine import (
    load_engine_payload,
    render_engine_driver,
    with_node_tmp,
)
from rsi_harness.cluster.bluevela.image import plan_image, render_build_driver
from rsi_harness.cluster.bluevela.preflight import node_tmp_assignment
from rsi_harness.cluster.bluevela.resources import derive_resource_plan
from rsi_harness.cluster.config import (
    ClusterProfile,
    PJMBuildProfile,
    load_cluster_profile,
)
from rsi_harness.cluster.pbs.adapter import PBSClusterAdapter
from rsi_harness.cluster.pjm.adapter import PJMClusterAdapter
from rsi_harness.cluster.schedulers.pjm import PJMJobSpec, PJMScheduler
from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions, GPURequirement, RunStatus
from tests.cluster.bluevela.test_adapter import (
    _fake_codex_runtime as _fake_codex_runtime,
)
from tests.cluster.bluevela.test_adapter import _request
from tests.cluster.bluevela.test_engine import _payload
from tests.cluster.pbs.test_adapter import RecordingBatchScheduler, local_profile
from tests.factories import DEFAULT_TASK_TOML, write_harbor_task

ENV = {"USER": "alice", "HOME": "/home/alice"}


def _adapter(profile, events=None):
    events = {} if events is None else events
    return PJMClusterAdapter(
        profile,
        scheduler=RecordingBatchScheduler(PJMScheduler(token=lambda: "a1b2c3")),
        event_callback=lambda key, value: events.update({key: value}),
        agent_version_resolver=lambda _: "0.149.0",
    )


def _dry_run(profile, request=None, tmp_path=None):
    events: dict = {}
    _adapter(profile, events).run(request or _request(tmp_path, dry_run=True))
    return events["dry_run"]


def _resources(argv) -> str:
    return argv[argv.index("-L") + 1]


def test_packaged_profiles_dispatch_and_round_trip(monkeypatch):
    profile = load_cluster_profile("pjm", ENV)
    assert profile.adapter == "pjm" and profile.scheduler.kind == "pjm"
    assert profile.scheduler.queue == "gpu"
    assert profile.scheduler.gpu_resource == "gpu"
    assert profile.scheduler.cpu_resource is None
    assert profile.scheduler.resources == ()
    assert profile.scheduler.build is None
    assert ClusterProfile.model_validate_json(profile.model_dump_json()) == profile
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    adapter = build_cluster_adapter("pjm")
    assert isinstance(adapter, PJMClusterAdapter)
    assert isinstance(adapter.scheduler, PJMScheduler)


def test_genkai_asks_for_gpus_in_its_container_environment():
    profile = load_cluster_profile("genkai", ENV)
    assert profile.name == "genkai" and profile.adapter == "pjm"
    assert profile.scheduler.queue == "b-batch"
    assert profile.scheduler.resources == ("jobenv=singularity",)
    assert profile.scheduler.build == PJMBuildProfile(resource_group="b-batch", gpus=1)
    assert profile.resources.gpus_per_node == 4
    assert profile.resources.node_tmp_variable == "PJM_SSD_DIR"
    assert str(profile.storage.run_root) == "/home/alice/rsi/runs"
    assert str(profile.apptainer.binary) == "/home/alice/rsi/apptainer/bin/apptainer"
    assert ClusterProfile.model_validate_json(profile.model_dump_json()) == profile


def test_genkai_new_dockerfile_builds_use_the_installed_rootless_helpers(tmp_path):
    profile = load_cluster_profile("genkai", ENV)
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nRUN apt-get update && apt-get install -y python3\n"
    )
    script = tmp_path / "build.sh"
    render_build_driver(
        plan_image(context, tmp_path / "images"), profile, context, script, 71680
    )

    rendered = script.read_text()
    helpers = "/home/alice/rsi/apptainer/x86_64/utils"
    assert f"faked={helpers}/bin/faked-sysv\n" in rendered
    assert (
        f"fakeroot_library={helpers}/lib/libfakeroot/libfakeroot-sysv.so\n" in rendered
    )
    assert 'APT::Sandbox::User "root";' in rendered
    assert "if true; then\n  apt_config=" in rendered
    assert '--volume "$fakeroot_library:/usr/lib/libfakeroot.so:ro"' in rendered
    assert 'builder_tmp_root="${PJM_SSD_DIR:-}"' in rendered


def _packaged_text(name="profile.toml"):
    return resources.files("rsi_harness.cluster.pjm").joinpath(name).read_text()


def test_profile_requires_the_pjm_scheduler(tmp_path):
    path = tmp_path / "profile.toml"
    path.write_text(_packaged_text().replace('adapter = "pjm"', 'adapter = "pbs"'))
    with pytest.raises(SetupError, match="requires pbs scheduler"):
        load_cluster_profile(path, ENV)


@pytest.mark.parametrize(
    ("section", "line"),
    [
        ("scheduler", 'excluded_hosts = ["b0001"]'),
        ("scheduler", 'resources = ["jobenv"]'),
        ("scheduler", 'resources = ["a=1", "a=2"]'),
        ("scheduler", 'resources = ["a=1,b=2"]'),
        ("scheduler", 'gpu_resource = "gpu=1"'),
        ("scheduler", 'cpu_resource = "vnode core"'),
        ("scheduler", 'memory_resource = "vnode-mem=1"'),
        ("scheduler", 'run_walltime_limit = "00:00"'),
        ("resources", 'node_tmp_variable = "pjm_ssd_dir"'),
        ("resources", 'node_tmp_variable = "1SSD"'),
    ],
)
def test_profile_refuses_what_pjm_jobs_cannot_honour(tmp_path, section, line):
    path = tmp_path / "profile.toml"
    text = _packaged_text()
    if section == "scheduler":
        text = text.replace('kind = "pjm"\n', f'kind = "pjm"\n{line}\n', 1)
    else:
        text += f"{line}\n"
    path.write_text(text)
    with pytest.raises(SetupError, match="invalid cluster profile"):
        load_cluster_profile(path, ENV)


@pytest.mark.parametrize("dry_run", [False, True])
def test_pjm_reuses_build_run_and_artifact_lifecycle(tmp_path, dry_run):
    profile = local_profile(tmp_path, "pjm")
    profile.apptainer.binary.unlink()  # compute-node software is not local
    events: dict = {}
    adapter = _adapter(profile, events)
    result = adapter.run(_request(tmp_path, dry_run=dry_run))
    scheduler = adapter.scheduler
    if dry_run:
        assert scheduler.events == []
        assert result.status == RunStatus.PREPARING
        assert not profile.storage.run_root.exists()
        details = events["dry_run"]
        assert details["adapter"] == "pjm"
        run, build = details["run_argv"], details["build_argv"]
        assert run[:3] == ("pjsub", "-z", "jid")
        assert _resources(run) == "rscgrp=gpu,elapse=02:15:00,gpu=4"
        assert _resources(build) == "rscgrp=gpu,elapse=02:00:00"
        assert "-X" not in run
        assert run[run.index("-o") + 1].endswith("/run/pjm.%j.out")
        assert build[build.index("-e") + 1].endswith("/build/pjm.%j.err")
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
    build, run = scheduler.specs
    assert isinstance(build, PJMJobSpec) and isinstance(run, PJMJobSpec)
    assert build.gpu_count == 0 and run.gpu_count == 4
    payload = load_engine_payload(run.script_path.with_suffix(".json"))
    assert payload.profile.scheduler.kind == "pjm"
    manifest = json.loads((payload.run_plan.paths.root / "RUN_INFO.json").read_text())
    assert manifest["state"] == "completed"
    # The profile names no node-local scratch: the temp_roots stand.
    assert "${PJM" not in build.script_path.read_text()
    assert "${PJM" not in run.script_path.read_text()


def test_genkai_jobs_use_the_node_ssd_where_pjm_sets_it(tmp_path):
    profile = local_profile(tmp_path, "genkai")
    adapter = _adapter(profile)
    assert adapter.run(_request(tmp_path, dry_run=False)).status == (
        RunStatus.COMPLETED
    )
    build, run = adapter.scheduler.specs
    renderer = adapter.scheduler.renderer
    assert _resources(renderer.render_submit(build)) == (
        "rscgrp=b-batch,elapse=02:00:00,gpu=1,jobenv=singularity"
    )
    assert _resources(renderer.render_submit(run)) == (
        "rscgrp=b-batch,elapse=02:15:00,gpu=4,jobenv=singularity"
    )
    build_script = build.script_path.read_text()
    assert 'builder_tmp_root="${PJM_SSD_DIR:-}"' in build_script
    assert 'mktemp -d "$builder_tmp_root/rsi-harness-sif.XXXXXX"' in build_script
    assert 'runtime_tmp_root="${PJM_SSD_DIR:-}"' in run.script_path.read_text()


def test_build_overrides_reach_only_the_build_job(tmp_path):
    profile = local_profile(
        tmp_path,
        "pjm",
        group="hp250000",
        build={
            "resource_group": "cpu",
            "group": "hp-build",
            "gpus": 0,
            "exclusive": True,
        },
    )
    details = _dry_run(profile, tmp_path=tmp_path)
    build, run = details["build_argv"], details["run_argv"]
    assert _resources(build) == "rscgrp=cpu,elapse=02:00:00,node=1"
    assert build[build.index("-g") + 1] == "hp-build"
    assert _resources(run) == "rscgrp=gpu,elapse=02:15:00,gpu=4"
    assert run[run.index("-g") + 1] == "hp250000"


def test_cpu_jobs_ask_for_cores_where_the_profile_says(tmp_path):
    task_dir = write_harbor_task(
        tmp_path / "cpu",
        task_toml=(
            DEFAULT_TASK_TOML.replace("gpus = 2", "gpus = 0")
            + "\n[metadata.rsi_harness.verifier]\ngpus = 0\n"
        ),
    )
    request = _request(tmp_path, dry_run=True).model_copy(
        update={"task_dir": task_dir.resolve()}
    )
    profile = local_profile(tmp_path, "pjm", cpu_resource="vnode-core")
    details = _dry_run(profile, request)
    rscgrp, elapse, cores = _resources(details["run_argv"]).split(",")
    assert (rscgrp, cores) == ("rscgrp=gpu", "vnode-core=8")
    assert elapse.startswith("elapse=")
    # And their memory, where the profile names it.
    profile = local_profile(
        tmp_path, "pjm", cpu_resource="vnode-core", memory_resource="vnode-mem"
    )
    details = _dry_run(profile, request)
    *_, cores, memory = _resources(details["run_argv"]).split(",")
    assert cores == "vnode-core=8"
    assert memory.startswith("vnode-mem=") and memory.endswith("Mi")


@pytest.mark.parametrize("shape", ["gpus", "disjoint"])
def test_tasks_that_need_several_nodes_are_refused(tmp_path, monkeypatch, shape):
    adapter = _adapter(local_profile(tmp_path, "pjm"))
    request = _request(tmp_path, dry_run=True)
    definition = adapter._compile(request)
    updates = {
        "gpus": {
            "gpu_requirement": GPURequirement(count=32),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 16}),
        },
        "disjoint": {"require_disjoint_phase_nodes": True},
    }[shape]
    big = definition.model_copy(update=updates)
    monkeypatch.setattr(adapter.compiler, "compile", lambda *_args, **_kw: big)
    refusal = (
        "PJM runs use one job on one node"
        if shape == "disjoint"
        else "GPUs on one node; this profile's PJM jobs have"
    )
    with pytest.raises(SetupError, match=refusal):
        adapter.run(request)
    assert adapter.scheduler.events == []


@pytest.mark.parametrize(
    "name,adapter_type", [("pbs", PBSClusterAdapter), ("pjm", PJMClusterAdapter)]
)
@pytest.mark.parametrize("override", [None, 4, 32])
def test_all_gpus_resolves_the_profile_override_before_the_single_node_check(
    tmp_path, name, adapter_type, override
):
    request = _request(tmp_path, dry_run=True)
    task = write_harbor_task(
        tmp_path / "all-gpus",
        task_toml=(
            DEFAULT_TASK_TOML.replace("gpus = 2\n", "")
            + "\n[metadata.rsi_harness.verifier]\ngpus = 1\n"
        ),
    )
    (task / "environment/docker-compose.yaml").write_text(
        "services:\n"
        "  main:\n"
        "    build: .\n"
        "    working_dir: /workspace\n"
        "    deploy:\n"
        "      resources:\n"
        "        reservations:\n"
        "          devices:\n"
        "            - driver: nvidia\n"
        "              count: all\n"
        "              capabilities: [gpu]\n"
    )
    request = request.model_copy(update={"task_dir": task.resolve()})
    profile = local_profile(tmp_path, name)
    profile = profile.model_copy(
        update={
            "resources": profile.resources.model_copy(
                update={"all_gpus_override": override, "gpus_per_node": 4}
            )
        }
    )
    adapter = adapter_type(profile)
    if override != 4:
        message = "numeric override" if override is None else "GPUs on one node"
        with pytest.raises(SetupError, match=message):
            adapter._compile(request)
        return

    definition = adapter._compile(request)
    assert definition.gpu_requirement.count == "all"
    planned = derive_resource_plan(definition, profile).single_node
    assert planned is not None
    assert planned.work_gpus == planned.total_gpus == 4


def test_oracle_runs_on_pjm(tmp_path, monkeypatch):
    task = write_harbor_task(tmp_path / "task")
    (task / "solution").mkdir()
    (task / "solution/solve.sh").write_text("#!/bin/sh\nprintf 42 > answer\n")
    profile = local_profile(tmp_path, "genkai")
    profile.apptainer.binary.unlink()
    monkeypatch.setattr("shutil.which", lambda _: None)
    events: dict = {}
    adapter = PJMClusterAdapter(
        profile,
        scheduler=RecordingBatchScheduler(PJMScheduler()),
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
    assert events["dry_run"]["run_argv"][0] == "pjsub"


# --- Node-local scratch ------------------------------------------------------


def _genkai_payload(tmp_path):
    payload = _payload(tmp_path)
    return payload.model_copy(update={"profile": load_cluster_profile("genkai", ENV)})


def test_with_node_tmp_moves_temp_roots_to_the_jobs_scratch(tmp_path):
    payload = _genkai_payload(tmp_path)
    moved = with_node_tmp(payload, {"PJM_SSD_DIR": "/local/1006903"})
    assert moved.profile.builder.temp_root == Path("/local/1006903")
    assert moved.profile.apptainer.temp_root == Path("/local/1006903")
    # Nothing else changes.
    assert moved.profile.storage == payload.profile.storage
    assert moved.run_plan == payload.run_plan
    assert payload.profile.apptainer.temp_root == Path("/tmp")


@pytest.mark.parametrize("environ", [{}, {"PJM_SSD_DIR": ""}, {"TMPDIR": "/x"}])
def test_with_node_tmp_keeps_temp_roots_where_the_scheduler_set_none(tmp_path, environ):
    payload = _genkai_payload(tmp_path)
    assert with_node_tmp(payload, environ) is payload


def test_with_node_tmp_needs_the_profile_to_name_a_variable(tmp_path, monkeypatch):
    payload = _payload(tmp_path)
    assert payload.profile.resources.node_tmp_variable is None
    assert with_node_tmp(payload, {"PJM_SSD_DIR": "/local/1"}) is payload
    # The job's own environment by default.
    genkai = _genkai_payload(tmp_path)
    monkeypatch.setenv("PJM_SSD_DIR", "/local/2")
    assert with_node_tmp(genkai).profile.apptainer.temp_root == Path("/local/2")


def _assigned(line: str, env: dict[str, str]) -> str:
    job = subprocess.run(
        ["/bin/bash", "-c", f'set -euo pipefail\n{line}\nprintf %s "$root"'],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return job.stdout


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"PJM_SSD_DIR": "/local/1006903"}, "/local/1006903"),
        ({"PJM_SSD_DIR": "/local/with space"}, "/local/with space"),
        ({"PJM_SSD_DIR": ""}, "/tmp/it's here"),
        ({}, "/tmp/it's here"),
    ],
)
def test_the_node_tmp_line_uses_the_variable_where_it_is_set(env, expected):
    line = node_tmp_assignment("root", Path("/tmp/it's here"), "PJM_SSD_DIR")
    assert _assigned(line, env) == expected


def test_the_node_tmp_line_without_a_variable_is_the_temp_root():
    line = node_tmp_assignment("root", Path("/tmp/x y"), None)
    assert line == "root='/tmp/x y'"
    assert _assigned(line, {"PJM_SSD_DIR": "/local/1"}) == "/tmp/x y"


def test_the_engine_and_build_drivers_check_space_where_they_will_write(tmp_path):
    payload = _genkai_payload(tmp_path)
    output = tmp_path / "run.sh"
    render_engine_driver(payload, payload.profile, output)
    script = output.read_text()
    assignment = 'runtime_tmp_root="${PJM_SSD_DIR:-}"\n'
    assert assignment in script
    assert script.index(assignment) < script.index('df -Pk -- "$runtime_tmp_root"')
    context = tmp_path / "context"
    context.mkdir()
    (context / "Dockerfile").write_text("FROM scratch\n")
    build = tmp_path / "build.sh"
    render_build_driver(
        plan_image(context, tmp_path / "cache"), payload.profile, context, build, 1024
    )
    assert 'builder_tmp_root="${PJM_SSD_DIR:-}"\n' in build.read_text()
