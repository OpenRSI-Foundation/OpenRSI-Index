from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rsi_harness.cluster.config import KubernetesSchedulerProfile
from rsi_harness.cluster.schedulers.kubernetes import (
    KubernetesJobSpec,
    KubernetesScheduler,
    uncovered_paths,
)
from rsi_harness.errors import InfrastructureError, SetupError

NAME = "mt-rsi-task-1-x-codex-run-alice-a1b2c3"
NOT_FOUND = (1, 'Error from server (NotFound): jobs.batch "x" not found')


def _profile(**values) -> KubernetesSchedulerProfile:
    return KubernetesSchedulerProfile.model_validate(
        {
            "kind": "kubernetes",
            "namespace": "rsi",
            "image": "registry.example/runner:1",
            "pod_security": "userns",
            "volumes": [
                {"path": "/srv/rsi", "claim": "shared"},
                {"path": "/scratch", "empty_dir": True},
                # Where the tests' job files live.
                {"path": "/tmp", "host_path": "/tmp"},
            ],
            **values,
        }
    )


def _spec(tmp_path: Path, **values) -> KubernetesJobSpec:
    script = tmp_path / "control" / "run.sh"
    script.parent.mkdir(exist_ok=True)
    script.write_text("#!/bin/bash\n")
    (tmp_path / "run").mkdir(exist_ok=True)
    return KubernetesJobSpec.model_validate(
        {
            "name": "mt-rsi-Task_1.x-codex-run-alice",
            "queue": "rsi",
            "group": "",
            "cpu_slots": 8,
            "memory_mb": 32768,
            "walltime": "02:30",
            "stdout_path": str(tmp_path / "run" / "kubernetes.%j.out"),
            "stderr_path": str(tmp_path / "run" / "kubernetes.%j.err"),
            "script_path": str(script),
            **values,
        }
    )


class Kubectl:
    """Answers kubectl calls from a table keyed on the verb and object."""

    def __init__(self, **answers: object) -> None:
        self.answers = {
            key: value if isinstance(value, list) else [value]
            for key, value in answers.items()
        }
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, argv: tuple[str, ...], _stdin: str | None):
        self.calls.append(argv)
        args = argv[argv.index("--namespace") + 2 :]
        key = re.sub(r"[^a-z]", "_", "_".join(args[:2]))
        queue = self.answers.get(key, [{"items": []}])
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        code, out = answer if isinstance(answer, tuple) else (0, answer)
        if not isinstance(out, str):
            out = json.dumps(out)
        return subprocess.CompletedProcess(
            argv, code, out if not code else "", out if code else ""
        )


class Clock:
    """Advances a minute each time it is read."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 60.0
        return self.now


def _scheduler(runner, profile=None, *, run_as=(1000, 1000, (1000, 2000)), **kw):
    kw.setdefault("clock", Clock())
    return KubernetesScheduler(
        profile or _profile(),
        owner="alice",
        run_as=run_as,
        runner=runner,
        sleeper=lambda _seconds: None,
        token=lambda: "a1b2c3",
        **kw,
    )


def _job(name=NAME, condition=None, *, annotations=None, labels=None, **status):
    conditions = [] if condition is None else [condition]
    metadata = {"name": name, "annotations": annotations or {}, "labels": labels or {}}
    return {"metadata": metadata, "status": {"conditions": conditions, **status}}


def _pod(phase="Running", waiting=None, terminated=None, reason=None, name="p"):
    state = {}
    if waiting:
        state["waiting"] = {"reason": waiting, "message": "detail"}
    if terminated is not None:
        state["terminated"] = {"exitCode": terminated}
    status = {
        "phase": phase,
        "containerStatuses": [{"name": "rsi-harness", "state": state}],
    }
    if reason:
        status["reason"] = reason
    return {"items": [{"metadata": {"name": name}, "status": status}]}


def _event(reason, message="detail", at="2026-11-03T10:00:00Z"):
    return {"items": [{"reason": reason, "message": message, "lastTimestamp": at}]}


def test_manifest_is_one_pod_with_the_allocation_and_no_api_access(tmp_path):
    spec = _spec(tmp_path, gpu_count=2, local_tmp_mb=4096, excluded_hosts=("n1",))
    manifest = _scheduler(Kubectl()).render_manifest(spec)

    assert manifest["metadata"]["name"] == NAME
    assert manifest["metadata"]["labels"]["rsi-harness.io/name"] == (
        "mt-rsi-Task_1.x-codex-run-alice"
    )
    job = manifest["spec"]
    assert job["backoffLimit"] == 0 and job["ttlSecondsAfterFinished"] == 604800
    pod = job["template"]["spec"]
    assert pod["restartPolicy"] == "Never"
    assert pod["activeDeadlineSeconds"] == 9000
    assert pod["terminationGracePeriodSeconds"] == 30
    assert pod["automountServiceAccountToken"] is False
    assert pod["enableServiceLinks"] is False
    assert pod["hostUsers"] is False
    assert pod["securityContext"] == {
        "runAsUser": 1000,
        "runAsGroup": 1000,
        "supplementalGroups": [1000, 2000],
    }
    terms = pod["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    assert terms[0]["matchExpressions"][0]["values"] == ["n1"]
    (container,) = pod["containers"]
    assert container["securityContext"] == {
        "procMount": "Unmasked",
        "seccompProfile": {"type": "Unconfined"},
        "appArmorProfile": {"type": "Unconfined"},
    }
    assert container["resources"] == {
        "requests": {
            "cpu": "8",
            "memory": "32768Mi",
            "nvidia.com/gpu": "2",
            "ephemeral-storage": "4096Mi",
        },
        "limits": {"cpu": "8", "memory": "32768Mi", "nvidia.com/gpu": "2"},
    }
    env = {item["name"]: item["value"] for item in container["env"]}
    # GPUs come from the device plugin's list, never node indices.
    assert "CUDA_VISIBLE_DEVICES" not in env
    assert env["USER"] == "alice"
    run = tmp_path / "run"
    assert container["command"][-7:] == [
        str(run / f"kubernetes.{NAME}.out"),
        str(run / f"kubernetes.{NAME}.err"),
        str(spec.script_path),
        "2",
        "/dev",
        "",
        str(run / NAME / "cancelled"),
    ]
    assert container["workingDir"] == str(run)
    assert pod["volumes"] == [
        {"name": "v0", "persistentVolumeClaim": {"claimName": "shared"}},
        {"name": "v1", "emptyDir": {}},
        {"name": "v2", "hostPath": {"path": "/tmp", "type": "Directory"}},
    ]
    relative = (run / NAME).relative_to("/tmp")
    assert container["volumeMounts"] == [
        {"name": "v0", "mountPath": "/srv/rsi"},
        {"name": "v1", "mountPath": "/scratch"},
        {"name": "v2", "mountPath": "/tmp"},
        {
            "name": "v2",
            "mountPath": "/etc/passwd",
            "subPath": f"{relative}/passwd",
            "readOnly": True,
        },
        {
            "name": "v2",
            "mountPath": "/etc/group",
            "subPath": f"{relative}/group",
            "readOnly": True,
        },
    ]


def _run_wrapper(tmp_path, gpus, devices, nodes=2):
    """Run the Pod's command with a fake unshare and the given GPU list."""
    spec = _spec(tmp_path, gpu_count=gpus)
    spec.script_path.write_text('#!/bin/bash\necho "cuda=$CUDA_VISIBLE_DEVICES"\n')
    command = _scheduler(Kubectl()).render_manifest(spec)["spec"]["template"]["spec"][
        "containers"
    ][0]["command"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "unshare").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "unshare").chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin"}
    if devices is not None:
        env["NVIDIA_VISIBLE_DEVICES"] = devices
    dev = tmp_path / "dev"
    dev.mkdir(exist_ok=True)
    for index in range(nodes):
        (dev / f"nvidia{index}").touch()
    (dev / "nvidiactl").touch()
    completed = subprocess.run(
        [*command[:-3], str(dev), "", command[-1]], env=env, check=False
    )
    out = Path(command[4]).read_text() if Path(command[4]).exists() else ""
    err = Path(command[5]).read_text() if Path(command[5]).exists() else ""
    return completed.returncode, out, err


@pytest.mark.parametrize(
    ("devices", "nodes", "cuda"),
    [
        # The device plugin's UUIDs.
        ("GPU-a,GPU-b", 2, "GPU-a,GPU-b"),
        # CDI, volume mounts or host indices: the Pod's own two GPUs.
        (None, 2, "0,1"),
        ("all", 2, "0,1"),
        ("3,5", 2, "0,1"),
        ("MIG-a,MIG-b", 1, "MIG-a,MIG-b"),
        # Anything else than the allocation is refused.
        ("GPU-a", 2, None),
        (None, 8, None),
    ],
)
def test_the_pod_binds_exactly_the_allocated_gpus(tmp_path, devices, nodes, cuda):
    code, out, err = _run_wrapper(tmp_path, 2, devices, nodes)
    if cuda is not None:
        assert code == 0 and out.strip() == f"cuda={cuda}"
    else:
        assert code == 1 and "GPU devices" in err


def test_the_pods_gpu_uuids_reach_phases_as_indices(tmp_path):
    from rsi_harness.cluster.bluevela.devices import allocated_cuda_devices

    code, out, _ = _run_wrapper(tmp_path, 2, "GPU-bb,GPU-aa")
    smi = "0, GPU-aa, 00000000:01:00.0\n1, GPU-bb, 00000000:02:00.0\n"

    result = allocated_cuda_devices(
        out.strip().removeprefix("cuda="),
        runner=lambda argv: subprocess.CompletedProcess(argv, 0, smi, ""),
        gpus_root=tmp_path / "no-driver",
    )

    assert code == 0 and result == (("1", "0"), True)


def test_a_pod_that_cannot_mount_proc_names_the_runtime(tmp_path):
    spec = _spec(tmp_path)
    command = _scheduler(Kubectl()).render_manifest(spec)["spec"]["template"]["spec"][
        "containers"
    ][0]["command"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "unshare").write_text(
        '#!/bin/bash\n[[ " $* " == *" --mount-proc "* ]] && exit 1\nexit 0\n'
    )
    (bin_dir / "unshare").chmod(0o755)
    completed = subprocess.run(
        command, env={"PATH": f"{bin_dir}:/usr/bin:/bin"}, check=False
    )
    assert completed.returncode == 1
    stderr = Path(command[5]).read_text()
    assert "cannot mount a fresh /proc" in stderr
    if "/proc/driver/nvidia" not in Path("/proc/self/mountinfo").read_text():
        assert "masked /proc" in stderr and "NVIDIA" not in stderr


def test_a_pod_without_user_namespaces_stops_naming_the_setting(tmp_path):
    spec = _spec(tmp_path)
    command = _scheduler(Kubectl()).render_manifest(spec)["spec"]["template"]["spec"][
        "containers"
    ][0]["command"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "unshare").write_text("#!/bin/bash\nexit 1\n")
    (bin_dir / "unshare").chmod(0o755)
    completed = subprocess.run(
        command, env={"PATH": f"{bin_dir}:/usr/bin:/bin"}, check=False
    )
    assert completed.returncode == 1
    assert "apparmor_profile" in Path(command[5]).read_text()


def test_privileged_pods_confine_neither_gpus_nor_root(tmp_path):
    privileged = _profile(pod_security="privileged")
    pod = _scheduler(Kubectl(), privileged).render_manifest(_spec(tmp_path))
    container = pod["spec"]["template"]["spec"]["containers"][0]
    assert container["securityContext"] == {
        "privileged": True,
        "allowPrivilegeEscalation": True,
    }
    with pytest.raises(SetupError, match="every GPU on the node"):
        _scheduler(Kubectl(), privileged).render_manifest(_spec(tmp_path, gpu_count=1))
    allocated = _profile(pod_security="privileged", privileged_devices="allocated")
    _scheduler(Kubectl(), allocated).render_manifest(_spec(tmp_path, gpu_count=1))


def test_a_runtime_class_is_requested(tmp_path):
    profile = _profile(runtime_class="nvidia")
    pod = _scheduler(Kubectl(), profile).render_manifest(_spec(tmp_path))
    assert pod["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"


def test_an_apparmor_profile_replaces_unconfined(tmp_path):
    profile = _profile(apparmor_profile="rsi-userns")
    pod = _scheduler(Kubectl(), profile).render_manifest(_spec(tmp_path))
    security = pod["spec"]["template"]["spec"]["containers"][0]["securityContext"]
    assert security["appArmorProfile"] == {
        "type": "Localhost",
        "localhostProfile": "rsi-userns",
    }


def test_cpu_jobs_and_an_advance_signal(tmp_path):
    pod = _scheduler(Kubectl()).render_manifest(
        _spec(tmp_path, local_tmp_mb=4096, request_tmp=False, signal_seconds=1800)
    )["spec"]["template"]["spec"]
    assert "ephemeral-storage" not in pod["containers"][0]["resources"]["requests"]
    assert pod["activeDeadlineSeconds"] == 9000 - 1800
    assert pod["terminationGracePeriodSeconds"] == 1800


def test_build_overrides_replace_image_and_placement(tmp_path):
    spec = _spec(
        tmp_path,
        image="registry.example/builder:1",
        node_selector={"pool": "cpu"},
        tolerations=({"key": "build", "operator": "Exists"},),
    )
    profile = _profile(node_selector={"pool": "gpu"})
    pod = _scheduler(Kubectl(), profile).render_manifest(spec)["spec"]["template"][
        "spec"
    ]
    assert pod["containers"][0]["image"] == "registry.example/builder:1"
    assert pod["nodeSelector"] == {"pool": "cpu"}
    assert pod["tolerations"] == [{"key": "build", "operator": "Exists"}]


def test_submit_writes_job_files_beside_the_logs_not_in_control(tmp_path):
    runner = Kubectl(create___output_name=f"job.batch/{NAME}\n")
    spec = _spec(tmp_path)
    profile = _profile(kubeconfig="/etc/rsi/kubeconfig", context="lab")
    scheduler = _scheduler(runner, profile)

    dry_run = scheduler.render_submit(spec)
    assert scheduler.submit(spec) == NAME
    (call,) = runner.calls
    assert call == dry_run
    assert call[:8] == (
        "kubectl",
        "--kubeconfig",
        "/etc/rsi/kubeconfig",
        "--context",
        "lab",
        "--request-timeout=60s",
        "--namespace",
        "rsi",
    )
    directory = tmp_path / "run" / NAME
    manifest = directory / "job.json"
    assert json.loads(manifest.read_text())["metadata"]["name"] == NAME
    assert manifest.stat().st_mode & 0o777 == 0o600
    # The frozen control directory that resume checks gains nothing.
    assert [path.name for path in spec.script_path.parent.iterdir()] == ["run.sh"]
    # Apptainer needs the Pod's uid named in passwd and group.
    assert "alice:x:1000:1000:alice:/tmp:/bin/bash\n" in (
        (directory / "passwd").read_text()
    )
    assert (directory / "group").read_text() == (
        "root:x:0:\nalice:x:1000:alice\ng2000:x:2000:alice\n"
    )


def test_run_jobs_get_the_submit_environment_in_a_private_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RSI_AGENT_API_KEY", "sk-synthetic 'quoted'")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://model.example/v1")
    monkeypatch.setenv("PATH", "/submit/only")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    runner = Kubectl(
        create___output_name=[
            f"job.batch/{NAME}\n",
            "job.batch/mt-rsi-task-build-alice-a1b2c3\n",
        ]
    )
    spec = _spec(tmp_path, forward_environment=True)
    scheduler = _scheduler(runner)
    assert scheduler.submit(spec) == NAME
    environment = tmp_path / "run" / NAME / "environment"
    assert environment.stat().st_mode & 0o777 == 0o600
    text = environment.read_text()
    assert "RSI_AGENT_API_KEY='sk-synthetic '\"'\"'quoted'\"'\"''\n" in text
    assert "OPENAI_BASE_URL=http://model.example/v1\n" in text
    names = {line.split("=", 1)[0] for line in text.splitlines()}
    assert "PATH" not in names and "SLURM_JOB_ID" not in names
    manifest = (tmp_path / "run" / NAME / "job.json").read_text()
    assert "sk-synthetic" not in manifest
    command = json.loads(manifest)["spec"]["template"]["spec"]["containers"][0][
        "command"
    ]
    assert command[-2:] == [str(environment), str(environment.parent / "cancelled")]
    # The wrapper exports it before anything else.
    script = tmp_path / "control" / "run.sh"
    script.write_text('#!/bin/bash\nprintf "%s" "$RSI_AGENT_API_KEY"\n')
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "unshare").write_text("#!/bin/bash\nexit 0\n")
    (bin_dir / "unshare").chmod(0o755)
    subprocess.run(command, env={"PATH": f"{bin_dir}:/usr/bin:/bin"}, check=True)
    assert Path(command[4]).read_text() == "sk-synthetic 'quoted'"
    # Read once: the Pod removes it.
    assert not environment.exists()
    environment.write_text("RSI_AGENT_API_KEY=x\n")

    # The profile's own settings win over the submit host's.
    profiled = _scheduler(Kubectl(), _profile(environment={"OPENAI_BASE_URL": "x"}))
    assert "OPENAI_BASE_URL" not in profiled.environment_text(dict(os.environ))
    # Removed once the Job has ended.
    finish = Kubectl(
        get_job=_job(condition={"type": "Complete", "status": "True"}),
        get_pods=_pod("Succeeded", terminated=0),
    )
    scheduler._runner = finish
    scheduler.wait(NAME)
    assert not environment.exists()
    scheduler._runner = runner

    build = _spec(
        tmp_path,
        name="mt-rsi-task-build-alice",
        script_path=str(tmp_path / "control" / "build.sh"),
    )
    build.script_path.write_text("#!/bin/bash\n")
    scheduler.submit(build)
    assert not (tmp_path / "run" / scheduler.job_name(build) / "environment").exists()


def test_a_create_the_server_accepted_before_failing_is_found_by_name(tmp_path):
    ours = _job(labels={"rsi-harness.io/name": "mt-rsi-Task_1.x-codex-run-alice"})
    runner = Kubectl(
        create___output_name=(1, "Error from server (AlreadyExists)"), get_job=ours
    )
    assert _scheduler(runner).submit(_spec(tmp_path)) == NAME

    for answers in (
        {"create___output_name": (1, "forbidden"), "get_job": NOT_FOUND},
        {"create___output_name": (1, "AlreadyExists"), "get_job": _job()},
        {"create___output_name": (0, "job.batch/other")},
    ):
        with pytest.raises(InfrastructureError):
            _scheduler(Kubectl(**answers)).submit(_spec(tmp_path))


def test_root_pods_keep_the_image_accounts(tmp_path):
    scheduler = _scheduler(Kubectl(), run_as=(0, 0, ()))
    pod = scheduler.render_manifest(_spec(tmp_path))["spec"]["template"]["spec"]
    mounts = [mount["mountPath"] for mount in pod["containers"][0]["volumeMounts"]]
    assert "/etc/passwd" not in mounts


def test_account_files_use_the_innermost_shared_volume(tmp_path):
    profile = _profile(
        volumes=[
            {"path": "/tmp", "claim": "a"},
            {"path": str(tmp_path), "claim": "b", "sub_path": "runs"},
        ]
    )
    pod = _scheduler(Kubectl(), profile).render_manifest(_spec(tmp_path))
    mounts = pod["spec"]["template"]["spec"]["containers"][0]["volumeMounts"]
    assert mounts[-1] == {
        "name": "v1",
        "mountPath": "/etc/group",
        "subPath": f"runs/run/{NAME}/group",
        "readOnly": True,
    }
    scratch = _profile(volumes=[{"path": "/scratch", "empty_dir": True}])
    with pytest.raises(SetupError, match="no shared Kubernetes volume"):
        _scheduler(Kubectl(), scratch).render_manifest(_spec(tmp_path))


def test_uncovered_paths_follow_the_innermost_volume():
    profile = _profile(
        image_paths=["/opt/venv"],
        volumes=[
            {"path": "/srv/rsi", "claim": "shared"},
            {"path": "/srv/rsi/scratch", "empty_dir": True},
        ],
    )
    assert uncovered_paths(
        profile,
        (
            Path("/srv/rsi/runs"),
            Path("/opt/venv/bin/python"),
            Path("/srv/rsi/scratch/hf"),
            Path("/srv/rsix"),
        ),
    ) == (Path("/srv/rsi/scratch/hf"), Path("/srv/rsix"))


def test_an_active_job_of_the_same_name_blocks_submission():
    finished = _job("old", {"type": "Complete", "status": "True"})
    runner = Kubectl(get_jobs={"items": [finished]})
    _scheduler(runner).require_name_available("mt-rsi-task-run-alice")
    assert "--selector=rsi-harness.io/name=mt-rsi-task-run-alice" in runner.calls[0]

    runner = Kubectl(get_jobs={"items": [finished, _job("live")]})
    with pytest.raises(InfrastructureError, match="already active: live"):
        _scheduler(runner).require_name_available("mt-rsi-task-run-alice")


def test_wait_reports_states_and_the_pod_exit_code():
    complete = {"type": "Complete", "status": "True"}
    unschedulable = {
        "items": [
            {
                "metadata": {"name": "p"},
                "status": {
                    "phase": "Pending",
                    "conditions": [
                        {
                            "type": "PodScheduled",
                            "status": "False",
                            "reason": "Unschedulable",
                            "message": "0/1 nodes are available",
                        }
                    ],
                },
            }
        ]
    }
    runner = Kubectl(
        get_job=[_job(), _job(), _job(), _job(), _job(condition=complete)],
        get_pods=[
            {"items": []},
            unschedulable,
            _pod(),
            # The Pod's phase ends before its Job's condition: still running.
            _pod("Succeeded", terminated=0),
            _pod("Succeeded", terminated=0),
        ],
    )
    states: list[str] = []
    result = _scheduler(runner).wait(NAME, on_state=states.append)
    assert (result.state, result.exit_code) == ("COMPLETED", 0)
    assert states == [
        "PENDING",
        "PENDING (Unschedulable: 0/1 nodes are available)",
        "RUNNING",
        "COMPLETED",
    ]


def test_a_missing_claim_is_a_start_failure_not_a_queue_wait():
    missing = {
        "items": [
            {
                "metadata": {"name": "p"},
                "status": {
                    "phase": "Pending",
                    "conditions": [
                        {
                            "type": "PodScheduled",
                            "status": "False",
                            "reason": "Unschedulable",
                            "message": 'persistentvolumeclaim "rsi" not found',
                        }
                    ],
                },
            }
        ]
    }
    runner = Kubectl(get_job=_job(), get_pods=missing, patch_job="")
    with pytest.raises(InfrastructureError, match='persistentvolumeclaim "rsi"'):
        _scheduler(runner).wait(NAME)


@pytest.mark.parametrize(
    ("condition", "pod", "expected"),
    [
        (
            {"reason": "BackoffLimitExceeded"},
            _pod("Failed", terminated=3),
            ("FAILED", 3),
        ),
        (
            {"reason": "BackoffLimitExceeded"},
            _pod("Failed", terminated=137, reason="DeadlineExceeded"),
            ("TIMEOUT", 137),
        ),
        ({"reason": "DeadlineExceeded"}, {"items": []}, ("TIMEOUT", 1)),
        (
            {"reason": "BackoffLimitExceeded"},
            _pod("Failed", terminated=0),
            ("FAILED", 1),
        ),
    ],
)
def test_wait_preserves_failures(condition, pod, expected):
    runner = Kubectl(
        get_job=_job(condition={"type": "Failed", "status": "True", **condition}),
        get_pods=pod,
    )
    result = _scheduler(runner).wait(NAME)
    assert (result.state, result.exit_code) == expected


@pytest.mark.parametrize(
    ("answers", "reason", "immediate"),
    [
        (
            {"get_pods": _pod("Pending", waiting="CreateContainerConfigError")},
            "CreateContainerConfigError",
            True,
        ),
        (
            {"get_pods": _pod("Pending", waiting="ImagePullBackOff")},
            "ImagePullBackOff",
            False,
        ),
        (
            {
                "get_pods": {"items": []},
                "get_events": _event("FailedCreate", 'pods "x" is forbidden'),
            },
            "FailedCreate",
            False,
        ),
        (
            {
                "get_pods": _pod("Pending", waiting="ContainerCreating"),
                "get_events": _event("FailedMount"),
            },
            "FailedMount",
            False,
        ),
    ],
)
def test_a_pod_that_cannot_start_is_cancelled_with_its_reason(
    answers, reason, immediate
):
    runner = Kubectl(get_job=_job(), patch_job="", **answers)
    with pytest.raises(InfrastructureError, match=reason):
        _scheduler(runner).wait(NAME)
    polls = sum("pods" in call for call in runner.calls)
    assert polls == 1 if immediate else polls > 1
    assert runner.calls[-1][-5:-3] == ("patch", "job")


def test_a_start_failure_must_persist_before_it_cancels():
    clock = Clock()
    runner = Kubectl(
        get_job=[
            _job(),
            _job(),
            _job(condition={"type": "Complete", "status": "True"}),
        ],
        get_pods=[
            _pod("Pending", waiting="ContainerCreating"),
            _pod(),
            _pod("Succeeded", terminated=0),
        ],
        get_events=_event("FailedMount"),
    )
    assert _scheduler(runner, clock=clock).wait(NAME).state == "COMPLETED"


def test_a_pod_the_kubelet_rejects_names_why():
    runner = Kubectl(
        get_job=_job(condition={"type": "Failed", "status": "True"}),
        get_pods={
            "items": [
                {
                    "metadata": {"name": "p"},
                    "status": {
                        "phase": "Failed",
                        "reason": "AppArmor",
                        "message": "profile rsi-userns is not loaded",
                    },
                }
            ]
        },
    )
    with pytest.raises(InfrastructureError, match="AppArmor: profile rsi-userns"):
        _scheduler(runner).wait(NAME)


@pytest.mark.parametrize(
    "answers",
    [
        {"get_events": _event("FailedCreate", "exceeded quota: compute")},
        {"get_events": _event("Scheduled")},
    ],
)
def test_quota_waits_and_ordinary_events_stay_pending(answers):
    polls = iter(range(20))

    def sleeper(_seconds):
        if next(polls) == 15:
            raise KeyboardInterrupt

    runner = Kubectl(get_job=_job(), get_pods={"items": []}, patch_job="", **answers)
    scheduler = KubernetesScheduler(
        _profile(), owner="alice", runner=runner, sleeper=sleeper
    )
    with pytest.raises(KeyboardInterrupt):
        scheduler.wait(NAME)
    # Interrupted after 15 polls, not cancelled for a start failure first.
    assert sum("pods" in call for call in runner.calls) == 16


@pytest.mark.parametrize(
    "unreachable",
    [
        (1, "Unable to connect to the server: i/o timeout"),
        (
            1,
            "The connection to the server 127.0.0.1:6443 was refused - did you "
            "specify the right host or port?",
        ),
    ],
)
def test_transient_api_errors_are_retried_for_a_bounded_time(unreachable):
    complete = {"type": "Complete", "status": "True"}
    runner = Kubectl(
        get_job=[unreachable, _job(condition=complete)],
        get_pods=_pod("Succeeded", terminated=0),
    )
    assert _scheduler(runner).wait(NAME).state == "COMPLETED"

    clock = iter(range(0, 10000, 300))
    runner = Kubectl(get_job=unreachable)
    with pytest.raises(InfrastructureError, match="unreachable for 15 minutes"):
        _scheduler(runner, clock=lambda: float(next(clock))).wait(NAME)

    with pytest.raises(InfrastructureError, match="no longer exists"):
        _scheduler(Kubectl(get_job=NOT_FOUND)).wait(NAME)


def _interrupted(runner, verb):
    def run(argv, stdin):
        if verb in argv:
            raise KeyboardInterrupt
        return runner(argv, stdin)

    return run


@pytest.mark.parametrize(
    "stop",
    ["start failure", "interrupted wait", "interrupted create", "failed create"],
)
def test_every_end_the_harness_causes_removes_the_environment_file(tmp_path, stop):
    created = f"job.batch/{NAME}\n"
    if stop == "failed create":
        runner = Kubectl(create___output_name=(1, "forbidden"), get_job=NOT_FOUND)
    elif stop == "start failure":
        waiting = _pod("Pending", waiting="CreateContainerConfigError")
        runner = Kubectl(
            create___output_name=created, get_job=_job(), get_pods=waiting, patch_job=""
        )
    else:
        runner = Kubectl(
            create___output_name=created, get_job=_job(), get_pods=_pod(), patch_job=""
        )
    scheduler = _scheduler(runner)
    if stop == "interrupted create":
        scheduler._runner = _interrupted(runner, "create")
    spec = _spec(tmp_path, forward_environment=True)
    environment = tmp_path / "run" / NAME / "environment"
    with pytest.raises((InfrastructureError, KeyboardInterrupt)):
        scheduler.submit(spec)
        assert environment.exists()
        if stop == "interrupted wait":
            scheduler._sleeper = _raise_interrupt
        scheduler.wait(NAME)
    assert not environment.exists()


def _raise_interrupt(_seconds: float) -> None:
    raise KeyboardInterrupt


def test_a_second_interrupt_cannot_stop_the_cancel(tmp_path):
    import signal

    handlers = []

    def patch(argv, stdin):
        handlers.append(
            [signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)]
        )
        return runner(argv, stdin)

    # The create landed after the first look: it is cancelled on the second.
    runner = Kubectl(patch_job=[NOT_FOUND, ""])
    scheduler = _scheduler(_interrupted(patch, "create"))
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        scheduler.submit(_spec(tmp_path))
    assert handlers == [[signal.SIG_IGN, signal.SIG_IGN]] * 2
    assert signal.getsignal(signal.SIGINT) is before


def test_a_job_cancelled_before_its_pod_starts_never_runs_the_payload(tmp_path):
    spec = _spec(tmp_path, forward_environment=True)
    runner = Kubectl(create___output_name=f"job.batch/{NAME}\n", patch_job="")
    scheduler = _scheduler(runner)
    scheduler.submit(spec)
    scheduler.cancel(NAME)
    command = json.loads((tmp_path / "run" / NAME / "job.json").read_text())["spec"][
        "template"
    ]["spec"]["containers"][0]["command"]
    spec.script_path.write_text("#!/bin/bash\necho payload\n")
    completed = subprocess.run(command, check=False)
    assert completed.returncode == 1
    assert "cancelled this job before it started" in Path(command[5]).read_text()
    assert "payload" not in Path(command[4]).read_text()


def test_a_forbidden_create_reports_itself_and_removes_the_environment(tmp_path):
    runner = Kubectl(
        create___output_name=(1, 'cannot create resource "jobs"'),
        get_job=(1, 'cannot get resource "jobs"'),
    )
    with pytest.raises(InfrastructureError, match='cannot create resource "jobs"'):
        _scheduler(runner).submit(_spec(tmp_path, forward_environment=True))
    assert not (tmp_path / "run" / NAME / "environment").exists()


def test_a_cancel_that_fails_after_an_interrupt_says_the_job_may_run(tmp_path):
    clock = iter(range(0, 10000, 60))
    runner = Kubectl(
        create___output_name=f"job.batch/{NAME}\n",
        get_job=_job(),
        get_pods=_pod(),
        patch_job=(1, "Unable to connect to the server: i/o timeout"),
    )
    scheduler = _scheduler(runner, clock=lambda: float(next(clock)))
    scheduler.submit(_spec(tmp_path, forward_environment=True))
    interrupts = iter([True])
    scheduler._sleeper = lambda _s: next(interrupts, False) and _raise_interrupt(_s)
    with pytest.raises(InfrastructureError, match=f"could not cancel .*{NAME}.*may"):
        scheduler.wait(NAME)
    assert not (tmp_path / "run" / NAME / "environment").exists()
    assert (tmp_path / "run" / NAME / "cancelled").exists()


def test_an_interrupt_before_the_create_landed_says_so(tmp_path):
    scheduler = _scheduler(_interrupted(Kubectl(patch_job=NOT_FOUND), "create"))
    with pytest.raises(KeyboardInterrupt, match="before Kubernetes Job .* created"):
        scheduler.submit(_spec(tmp_path))
    assert (tmp_path / "run" / NAME / "cancelled").exists()


def test_a_kubectl_that_hangs_reports_its_own_error(monkeypatch):
    from rsi_harness.cluster.schedulers import kubernetes

    def hang(argv, **_kwargs):
        raise subprocess.TimeoutExpired(
            argv, 120, stderr="dial tcp 10.0.0.1:6443: i/o timeout\n"
        )

    monkeypatch.setattr(kubernetes.subprocess, "run", hang)
    detail = kubernetes._run(("kubectl", "get"), None).stderr
    assert detail.endswith("i/o timeout (dial tcp 10.0.0.1:6443: i/o timeout)")


def test_an_unschedulable_reason_is_shown_without_the_preemption_report():
    from rsi_harness.cluster.schedulers.kubernetes import _short

    message = (
        "0/3 nodes are available: 3 Insufficient nvidia.com/gpu. preemption: "
        "0/3 nodes are available: 3 No preemption victims found."
    )
    assert _short(message) == "0/3 nodes are available: 3 Insufficient nvidia.com/gpu."
    assert _short("word " * 40, 120) == ("word " * 24).strip() + " ..."


def test_interrupt_cancels_the_exact_job_gracefully_and_marks_it():
    runner = Kubectl(get_job=_job(), get_pods=_pod(), patch_job="")

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    scheduler = KubernetesScheduler(
        _profile(), owner="alice", runner=runner, sleeper=interrupt
    )
    with pytest.raises(KeyboardInterrupt, match=f"Job {NAME} cancelled"):
        scheduler.wait(NAME)
    assert runner.calls[-1][-5:] == (
        "patch",
        "job",
        NAME,
        "--type=merge",
        '--patch={"metadata":{"annotations":{"rsi-harness.io/cancelled":"true"}},'
        '"spec":{"suspend":true}}',
    )


def test_a_cancelled_job_ends_once_its_pod_is_gone_and_a_queue_suspension_waits():
    suspended = {
        "type": "Suspended",
        "status": "True",
        "lastTransitionTime": "2026-11-03T10:00:00Z",
    }
    cancelled = {"rsi-harness.io/cancelled": "true"}
    # Still draining: the terminating Pod is listed.
    runner = Kubectl(
        get_job=_job(condition=suspended, annotations=cancelled), get_pods=_pod()
    )
    assert _scheduler(runner).finished(NAME) is None
    runner = Kubectl(get_job=_job(condition=suspended, annotations=cancelled))
    result = _scheduler(runner).finished(NAME)
    assert result is not None and (result.state, result.exit_code) == ("CANCELLED", 1)
    assert result.ended_at == datetime(2026, 11, 3, 10, tzinfo=UTC).timestamp()
    # Suspended by a queue (Kueue), not cancelled: it has not run yet.
    assert _scheduler(Kubectl(get_job=_job(condition=suspended))).finished(NAME) is None


def test_finished_reads_the_end_and_a_removed_job_is_unknown():
    runner = Kubectl(get_job=_job(), get_pods=_pod())
    assert _scheduler(runner).finished(NAME) is None

    runner = Kubectl(
        get_job=_job(
            condition={"type": "Complete", "status": "True"},
            completionTime="2026-11-03T10:00:00Z",
        ),
        get_pods=_pod("Succeeded", terminated=0),
    )
    result = _scheduler(runner).finished(NAME)
    assert result is not None and result.state == "COMPLETED"
    assert result.ended_at == datetime(2026, 11, 3, 10, tzinfo=UTC).timestamp()

    result = _scheduler(Kubectl(get_job=NOT_FOUND)).finished(NAME)
    assert result is not None and (result.state, result.ended_at) == ("UNKNOWN", None)


def test_cancelling_a_removed_job_is_not_an_error():
    _scheduler(Kubectl(patch_job=NOT_FOUND)).cancel(NAME)


@pytest.mark.parametrize("job_id", ["", "Upper", "a" * 60, "x;rm", "-x"])
def test_invalid_job_names_never_reach_kubectl(job_id):
    runner = Kubectl()
    with pytest.raises(SetupError):
        _scheduler(runner).wait(job_id)
    assert not runner.calls


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"namespace": "Bad_NS"}, "DNS-1123"),
        ({"volumes": [{"path": "/a"}]}, "exactly one"),
        ({"volumes": [{"path": "/a", "claim": "c", "empty_dir": True}]}, "exactly one"),
        ({"volumes": [{"path": "/a", "nfs_server": "n"}]}, "together"),
        ({"volumes": [{"path": "/a", "host_path": "/a", "sub_path": "x"}]}, "sub_path"),
        (
            {"volumes": [{"path": "/a", "claim": "c"}, {"path": "/a", "claim": "d"}]},
            "two volumes",
        ),
        ({"volumes": [{"path": "relative", "claim": "c"}]}, "absolute"),
        ({"run_as_user": 1000}, "Extra inputs"),
        ({"exclusive": True}, "exclusive"),
        ({"run_walltime_limit": "0:00"}, "HH:MM"),
        ({"ttl_seconds_after_finished": 60}, "3600"),
    ],
)
def test_profile_rejects_unsafe_values(values, match):
    with pytest.raises(ValueError, match=match):
        _profile(**values)
