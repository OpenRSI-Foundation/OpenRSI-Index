from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela.adapter import build_cluster_adapter
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.config import ClusterProfile, load_cluster_profile
from rsi_harness.cluster.kubernetes.adapter import KubernetesClusterAdapter
from rsi_harness.cluster.schedulers.kubernetes import (
    KubernetesJobResult,
    KubernetesScheduler,
)
from rsi_harness.errors import SetupError
from rsi_harness.models import GPURequirement, RunStatus
from tests.cluster.bluevela.test_adapter import (
    RecordingScheduler,
    _profile,
    _request,
)
from tests.cluster.bluevela.test_adapter import (
    _fake_codex_runtime as _fake_codex_runtime,
)


class RecordingKubernetesScheduler(RecordingScheduler):
    def __init__(self, profile: ClusterProfile):
        super().__init__()
        self.renderer = KubernetesScheduler(
            profile.scheduler, owner="alice", token=lambda: "a1b2c3"
        )

    def render_manifest(self, spec, name=None):
        return self.renderer.render_manifest(spec, name)

    def wait(self, job_id, **kwargs):
        super().wait(job_id, **kwargs)
        return KubernetesJobResult(job_id=job_id, state="COMPLETED", exit_code=0)


def _kubernetes(tmp_path: Path, **scheduler) -> ClusterProfile:
    local = _profile(tmp_path)
    base = load_cluster_profile("kubernetes", {"USER": "alice"})
    values = {
        **base.scheduler.model_dump(),
        # Everything the jobs use lives under tmp_path or the interpreter.
        "volumes": [{"path": str(tmp_path), "host_path": str(tmp_path)}],
        "image_paths": sorted({sys.prefix, sys.base_prefix}),
        "pod_security": "userns",
        **scheduler,
    }
    return base.model_copy(
        update={
            "scheduler": type(base.scheduler).model_validate(values),
            "storage": local.storage,
            "apptainer": local.apptainer.model_copy(
                update={
                    "work_binds": (),
                    "judge_binds": (),
                    "judge_authority_root": None,
                }
            ),
            "builder": local.builder,
        }
    )


def _req(tmp_path, *, dry_run):
    # Kubernetes refuses --agent-auth local; the Agent's key comes from the
    # submit host's environment.
    return _request(tmp_path, dry_run=dry_run).model_copy(update={"agent_auth": None})


def _adapter(profile, events=None):
    events = {} if events is None else events
    return KubernetesClusterAdapter(
        profile,
        scheduler=RecordingKubernetesScheduler(profile),
        event_callback=lambda key, value: events.update({key: value}),
        agent_version_resolver=lambda _: "0.149.0",
    )


def test_packaged_profile_dispatches_and_round_trips():
    profile = load_cluster_profile("kubernetes", {"USER": "alice"})
    assert profile.scheduler.kind == "kubernetes"
    assert profile.scheduler.queue == "rsi"
    assert ClusterProfile.model_validate_json(profile.model_dump_json()) == profile
    adapter = build_cluster_adapter("kubernetes")
    assert isinstance(adapter, KubernetesClusterAdapter)
    assert isinstance(adapter.scheduler, KubernetesScheduler)


def test_profile_requires_the_kubernetes_scheduler(tmp_path):
    source = Path(
        __import__("rsi_harness.cluster.kubernetes", fromlist=["x"]).__file__
    ).with_name("profile.toml")
    path = tmp_path / "profile.toml"
    path.write_text(
        source.read_text().replace('adapter = "kubernetes"', 'adapter = "slurm"')
    )
    with pytest.raises(SetupError, match="requires slurm scheduler"):
        load_cluster_profile(path, {"USER": "alice"})


@pytest.mark.parametrize("dry_run", [False, True])
def test_kubernetes_reuses_build_run_and_artifact_lifecycle(tmp_path, dry_run):
    profile = _kubernetes(tmp_path)
    profile.apptainer.binary.unlink()  # compute-node software is not local
    events: dict = {}
    adapter = _adapter(profile, events)
    result = adapter.run(_req(tmp_path, dry_run=dry_run))
    scheduler = adapter.scheduler
    if dry_run:
        assert scheduler.events == []
        assert result.status == RunStatus.PREPARING
        details = events["dry_run"]
        assert details["run_argv"][:4] == (
            "kubectl",
            "--request-timeout=60s",
            "--namespace",
            "rsi",
        )
        run = details["manifests"]["run"]["spec"]["template"]["spec"]
        (container,) = run["containers"]
        assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
        assert run["activeDeadlineSeconds"] == 135 * 60
        build = details["manifests"]["build"]["spec"]["template"]["spec"]
        assert "nvidia.com/gpu" not in build["containers"][0]["resources"]["limits"]
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
    assert scheduler.specs[0].stdout_path.name == "kubernetes.%j.out"
    payload = load_engine_payload(scheduler.specs[1].script_path.with_suffix(".json"))
    assert payload.profile.scheduler.kind == "kubernetes"
    manifest = json.loads((payload.run_plan.paths.root / "RUN_INFO.json").read_text())
    assert manifest["state"] == "completed"


def test_paths_the_pod_cannot_see_are_refused_before_scheduling(tmp_path):
    profile = _kubernetes(tmp_path, image_paths=[])
    adapter = _adapter(profile)
    with pytest.raises(SetupError, match="would not see these paths") as error:
        adapter.run(_req(tmp_path, dry_run=True))
    assert sys.base_prefix in str(error.value)
    assert adapter.scheduler.events == []


def test_multinode_tasks_are_refused(tmp_path, monkeypatch):
    adapter = _adapter(_kubernetes(tmp_path))
    request = _req(tmp_path, dry_run=True)
    definition = adapter._compile(request)
    big = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=32),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 16}),
        }
    )
    monkeypatch.setattr(adapter.compiler, "compile", lambda *_args, **_kw: big)
    with pytest.raises(SetupError, match="one Pod on one node"):
        adapter.run(request)


def test_privileged_gpu_tasks_are_refused_before_the_build(tmp_path):
    profile = _kubernetes(tmp_path, pod_security="privileged")
    adapter = _adapter(profile)
    with pytest.raises(SetupError, match="every GPU on the node"):
        adapter.run(_req(tmp_path, dry_run=False))
    assert adapter.scheduler.events == []
    allowed = _kubernetes(
        tmp_path, pod_security="privileged", privileged_devices="allocated"
    )
    _adapter(allowed).run(_req(tmp_path, dry_run=True))


def test_build_overrides_reach_only_the_build_job(tmp_path):
    profile = _kubernetes(
        tmp_path,
        node_selector={"pool": "gpu"},
        build={"image": "registry.example/builder:1", "node_selector": {"pool": "cpu"}},
    )
    events: dict = {}
    _adapter(profile, events).run(_req(tmp_path, dry_run=True))
    manifests = events["dry_run"]["manifests"]
    build = manifests["build"]["spec"]["template"]["spec"]
    run = manifests["run"]["spec"]["template"]["spec"]
    assert build["nodeSelector"] == {"pool": "cpu"}
    assert build["containers"][0]["image"] == "registry.example/builder:1"
    assert run["nodeSelector"] == {"pool": "gpu"}
    assert run["containers"][0]["image"] == profile.scheduler.image


def test_a_walltime_limit_segments_with_the_pod_grace_as_drain(tmp_path):
    profile = _kubernetes(tmp_path, run_walltime_limit="02:00")
    events: dict = {}
    _adapter(profile, events).run(_req(tmp_path, dry_run=True))
    run = events["dry_run"]["manifests"]["run"]["spec"]["template"]["spec"]
    grace = run["terminationGracePeriodSeconds"]
    assert grace > 30
    assert run["activeDeadlineSeconds"] + grace == 7200


def test_pods_run_as_the_submitting_user_and_never_as_root(tmp_path, monkeypatch):
    import os

    def pod(adapter):
        spec = adapter._build_spec(
            tmp_path,
            "mt-rsi-x-build-alice",
            _resources(adapter, tmp_path),
            _script(tmp_path),
        )
        return adapter.scheduler.render_manifest(spec)["spec"]["template"]["spec"]

    adapter = KubernetesClusterAdapter(_kubernetes(tmp_path))
    assert pod(adapter)["securityContext"]["runAsUser"] == os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: 0)
    for security in ("privileged", "userns"):
        with pytest.raises(SetupError, match="never root"):
            KubernetesClusterAdapter(_kubernetes(tmp_path, pod_security=security))
    monkeypatch.setattr(os, "getuid", lambda: 70000)
    with pytest.raises(SetupError, match="beyond 65535"):
        KubernetesClusterAdapter(_kubernetes(tmp_path))
    KubernetesClusterAdapter(_kubernetes(tmp_path, pod_security="privileged"))
    monkeypatch.setattr(os, "getuid", lambda: 1000)
    monkeypatch.setattr(os, "getgroups", lambda: [1000, 70000])
    adapter = KubernetesClusterAdapter(_kubernetes(tmp_path))
    assert 70000 not in pod(adapter)["securityContext"].get("supplementalGroups", [])


def test_local_agent_auth_is_refused_and_only_run_jobs_get_the_environment(
    tmp_path,
):
    adapter = _adapter(_kubernetes(tmp_path))
    with pytest.raises(SetupError, match="--agent-auth local"):
        adapter.run(_request(tmp_path, dry_run=True))
    resources = _resources(adapter, tmp_path)
    build = adapter._build_spec(tmp_path, "b", resources, _script(tmp_path))
    run = adapter._run_spec(tmp_path, "r", resources, _script(tmp_path))
    assert (build.forward_environment, run.forward_environment) == (False, True)


def test_resume_accepts_the_files_a_kubernetes_submission_writes(tmp_path, monkeypatch):
    """Job manifests and account files never land in the frozen control."""
    from rsi_harness.cluster.schedulers.kubernetes import KubernetesJobSpec
    from rsi_harness.runtime.recovery import AttemptGrants
    from tests.cluster.bluevela.test_resume import ResumeScheduler, _interrupted_run

    def kubectl(argv, _stdin):
        path = Path(next(arg for arg in argv if arg.startswith("--filename="))[11:])
        name = json.loads(path.read_text())["metadata"]["name"]
        return subprocess.CompletedProcess(argv, 0, f"job.batch/{name}\n", "")

    real = KubernetesScheduler(
        _kubernetes(tmp_path).scheduler,
        owner="alice",
        run_as=(1000, 1000, ()),
        runner=kubectl,
    )
    submit = ResumeScheduler.submit

    def submit_both(self, spec):
        real.submit(KubernetesJobSpec.model_validate(spec.model_dump()))
        return submit(self, spec)

    monkeypatch.setattr(ResumeScheduler, "submit", submit_both)
    run = _interrupted_run(tmp_path)
    assert any((run.run_dir / "run").glob("*/job.json"))
    assert any((run.run_dir / "run").glob("*/passwd"))

    plan = run.adapter.plan_resume(run.run_id, AttemptGrants(), dry_run=True)
    assert plan.context.attempt == 2


def _resources(adapter, tmp_path):
    from rsi_harness.cluster.bluevela.resources import derive_resource_plan

    definition = adapter._compile(_req(tmp_path, dry_run=True))
    return derive_resource_plan(definition, adapter.profile).single_node


def _script(tmp_path):
    script = tmp_path / "build.sh"
    script.write_text("#!/bin/bash\n")
    return script


def test_a_failed_job_names_the_pod_preflight(tmp_path):
    from rsi_harness.cluster.bluevela.adapter import job_log_detail

    log = tmp_path / "kubernetes.%j.err"
    (tmp_path / "kubernetes.job-a1.err").write_text(
        "kubernetes preflight: this Pod cannot create user namespaces\n"
    )
    detail = job_log_detail(log, "job-a1")
    assert "(kubernetes preflight: this Pod cannot create user namespaces)" in detail


def test_user_namespace_pods_name_their_user_in_the_image_accounts(tmp_path):
    import os
    import pwd

    from rsi_harness.cluster.bluevela.runtime import kubernetes_accounts

    image = tmp_path / "image"
    (image / "etc").mkdir(parents=True)
    (image / "etc/passwd").write_text("root:x:0:0:root:/root:/bin/sh\n")
    (image / "etc/group").write_text("root:x:0:\n")
    passwd, group = kubernetes_accounts(image, tmp_path / "accounts")
    name = pwd.getpwuid(os.getuid()).pw_name
    lines = passwd.read_text().splitlines()
    assert lines[0] == "root:x:0:0:root:/root:/bin/sh"
    assert lines[-1] == f"{name}:x:{os.getuid()}:{os.getgid()}:{name}:/home/rsi:/bin/sh"
    assert f":{os.getgid()}:" in group.read_text()

    # The user's shell, as in a privileged Pod, where the image has one.
    (image / "bin").mkdir()
    (image / "bin/bash").write_text("")
    passwd, _group = kubernetes_accounts(image, tmp_path / "accounts")
    assert passwd.read_text().splitlines()[-1].endswith(":/home/rsi:/bin/bash")

    # A SIF is read through Apptainer.
    apptainer = tmp_path / "apptainer"
    apptainer.write_text(
        "#!/bin/bash\nprintf 'root:x:0:0::/root:/bin/sh\\n@@\\nroot:x:0:\\n@@\\n"
        "/bin/bash\\n'\n"
    )
    apptainer.chmod(0o755)
    sif = tmp_path / "task.sif"
    sif.write_text("")
    passwd, group = kubernetes_accounts(sif, tmp_path / "sif", apptainer=apptainer)
    lines = passwd.read_text().splitlines()
    assert lines[0] == "root:x:0:0::/root:/bin/sh"
    assert lines[-1].endswith(":/home/rsi:/bin/bash")
    assert group.read_text().startswith("root:x:0:\n")


def test_an_extraction_failure_names_apptainers_reason(tmp_path, monkeypatch):
    import subprocess as subprocess_module

    from rsi_harness.cluster.bluevela import engine

    payload = SimpleNamespace(
        profile=SimpleNamespace(
            scheduler=SimpleNamespace(kind="kubernetes"),
            apptainer=SimpleNamespace(binary=Path("/usr/bin/apptainer")),
        ),
        sif_path=tmp_path / "image.sif",
    )
    monkeypatch.setattr(engine.os, "access", lambda *_args: False)
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(tmp_path))
    stderr = (
        "\x1b[33mWARNING:\x1b[0m something\n"
        "\x1b[31mFATAL:  \x1b[0m container creation failed: mount proc->/proc\n"
        "\x1b[0m: exit status 255\n"
    )
    monkeypatch.setattr(
        subprocess_module,
        "run",
        lambda *_args, **_kw: subprocess_module.CompletedProcess((), 255, "", stderr),
    )
    with pytest.raises(
        Exception, match="FATAL: +container creation failed: mount proc"
    ):
        engine._usable_image(payload)
