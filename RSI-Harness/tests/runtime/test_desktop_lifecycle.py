"""Exercise Desktop integration points with a Docker fake on Linux CI."""

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.errors import InfrastructureError, RecoveryRequiredError, SetupError
from rsi_harness.models import (
    AgentRunResult,
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    SubmissionStatus,
)
from rsi_harness.runtime import desktop
from rsi_harness.runtime.judge import DockerJudgeRoundRuntime
from rsi_harness.runtime.production import ProductionRecoveryBackend
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager
from tests.fakes import FakeDockerClient
from tests.runtime.test_desktop import client_for
from tests.runtime.test_docker import make_runtime
from tests.runtime.test_judge import evaluate, make_harness


def make_judge(tmp_path, monkeypatch, *, platform="darwin"):
    monkeypatch.setattr(desktop, "sys", SimpleNamespace(platform=platform))
    client = FakeDockerClient()
    client.info = lambda: {"OperatingSystem": "Docker Desktop"}
    runtime = make_runtime(client, tmp_path, role="judge")
    target = tmp_path / "engine-root" / "verifier"
    target.mkdir()
    ref = runtime.create(
        ContainerSpec(
            image="task-image",
            command=("sleep", "infinity"),
            mounts=(
                ContainerMount(source=target, target=PurePosixPath(desktop.LOG_TARGET)),
            ),
            labels=((desktop.PRIVATE_LOG_LABEL, "volume"),),
        )
    )
    return runtime, client, ref, target


@pytest.mark.parametrize("platform,kind", [("linux", "bind"), ("darwin", "volume")])
def test_create_start_and_remove_use_only_attested_private_volume(
    tmp_path, monkeypatch, platform, kind
):
    runtime, client, ref, _ = make_judge(tmp_path, monkeypatch, platform=platform)
    created = client.containers.created[0]
    assert created["labels"][desktop.PRIVATE_LOG_LABEL] == kind
    assert created["mounts"][0]["Type"] == kind
    if kind == "volume":
        assert created["mounts"][0]["VolumeOptions"]["NoCopy"] is True
    calls = []
    runtime._container_networks[ref.container_id] = "network"
    runtime._policy_leases[ref.container_id] = "lease"
    runtime._network_policy_enforcer = SimpleNamespace(
        attest=lambda lease: calls.append("attest")
    )
    monkeypatch.setattr(
        runtime,
        "_initialize_private_judge_logs",
        lambda ref: calls.append("initialize"),
    )
    runtime.start(ref)
    assert client.containers.get(ref.container_id).events[-1] == "start"
    assert calls == (["attest", "initialize"] if kind == "volume" else ["attest"])
    monkeypatch.setattr(
        runtime, "collect_private_judge_logs", lambda ref: calls.append("export")
    )
    runtime.remove(ref)
    # Preserve normal runtime cleanup of anonymous image volumes on Linux too.
    assert client.containers.get(ref.container_id).removed_kwargs == {
        "force": True,
        "v": True,
    }
    assert ("export" in calls) is (kind == "volume")


def test_exec_collects_before_reward_consumption_and_removal(tmp_path, monkeypatch):
    runtime, client, ref, target = make_judge(tmp_path, monkeypatch)
    archive = client_for([]).containers.get("judge")
    client.containers.get(ref.container_id).get_archive = archive.get_archive
    monkeypatch.setattr(runtime, "exec", lambda *a, **kw: AgentRunResult(exit_code=0))
    round_runtime = object.__new__(DockerJudgeRoundRuntime)
    round_runtime._runtime = runtime
    round_runtime.exec(ref, ("test",))
    assert ref.container_id not in runtime._private_log_targets
    runtime.remove(ref)
    assert not runtime._private_log_targets


def test_first_content_rejection_during_remove_still_cleans(tmp_path, monkeypatch):
    import rsi_harness.runtime.docker as docker_runtime

    runtime, client, ref, _ = make_judge(tmp_path, monkeypatch)

    def fail(*args):
        raise desktop.PrivateLogContentError("invalid archive")

    monkeypatch.setattr(docker_runtime, "export_private_logs", fail)
    runtime.remove(ref)
    assert client.containers.get(ref.container_id).removed_kwargs == {
        "force": True,
        "v": True,
    }
    assert not runtime._private_log_targets


@pytest.mark.parametrize("content_error", [False, True])
def test_export_fault_controls_cleanup_and_recovery(
    tmp_path, monkeypatch, content_error
):
    runtime, client, ref, _ = make_judge(tmp_path, monkeypatch)
    import rsi_harness.runtime.docker as docker_runtime

    error = (
        desktop.PrivateLogContentError("invalid archive")
        if content_error
        else RecoveryRequiredError("transport failed")
    )

    def fail(*args):
        raise error

    monkeypatch.setattr(docker_runtime, "export_private_logs", fail)
    with pytest.raises(type(error)):
        runtime.collect_private_judge_logs(ref)
    if content_error:
        runtime.remove(ref)
        assert client.containers.get(ref.container_id).removed_kwargs == {
            "force": True,
            "v": True,
        }
    else:
        with pytest.raises(RecoveryRequiredError, match="transport failed"):
            runtime.remove(ref)
        assert client.containers.get(ref.container_id).removed_kwargs is None
        assert ref.container_id in runtime._private_log_targets


@pytest.mark.parametrize(
    "message", ["unsafe archive", "recovery_required in task filename"]
)
def test_invalid_archive_cannot_accept_preexisting_reward(tmp_path, message):
    from rsi_harness.runtime.artifacts import RunArtifactWriter

    runner, runtime, _, _, request = make_harness(tmp_path)
    writer = RunArtifactWriter(request[0], run_id="run-1")
    writer.start()
    runner._artifact_writer = writer
    original = runtime.exec

    def fail(*args, **kwargs):
        # Exercise the real post-exec export wrapper, including propagation of
        # complete output to the strict production artifact writer.
        wrapper = object.__new__(DockerJudgeRoundRuntime)

        def reject(container):
            raise desktop.PrivateLogContentError(message)

        wrapper._runtime = SimpleNamespace(
            exec=original, collect_private_judge_logs=reject
        )
        return wrapper.exec(*args, **kwargs)

    runtime.exec = fail
    report = evaluate(runner, request)
    assert report.status == SubmissionStatus.VERIFIER_ERROR
    assert report.score is None
    assert report.full_output_captured
    assert not runner.recovery_required
    assert any(event[0] == "judge_remove" for event in runtime.events)


@pytest.mark.parametrize("redacted", [False, True])
def test_post_exec_recovery_type_survives_absent_or_redacted_marker(tmp_path, redacted):
    runner, runtime, _, _, request = make_harness(tmp_path)
    message = "recovery_required: private-export-token" if redacted else "export failed"
    if redacted:
        plan, work, logs = request
        plan = plan.model_copy(
            update={
                "task": plan.task.model_copy(
                    update={
                        "verifier": plan.task.verifier.model_copy(
                            update={"environment": (("TOKEN", "${VERIFIER_TOKEN}"),)}
                        )
                    }
                )
            }
        )
        request = plan, work, logs
        runner._verifier_secret_env = {"VERIFIER_TOKEN": message}
    original = runtime.exec

    def fail(*args, **kwargs):
        wrapper = object.__new__(DockerJudgeRoundRuntime)

        def reject(_container):
            raise RecoveryRequiredError(message)

        wrapper._runtime = SimpleNamespace(
            exec=original, collect_private_judge_logs=reject
        )
        return wrapper.exec(*args, **kwargs)

    runtime.exec = fail
    report = evaluate(runner, request)
    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert report.score is None
    assert runner.recovery_required
    assert runtime.work_paused
    assert ("snapshot_release", "agent-1") not in runtime.events
    assert ("unpause", "work-1") not in runtime.events
    if redacted:
        assert report.error.startswith("primary failure: [REDACTED];")
        assert "private-export-token" not in report.error
    else:
        assert report.error.startswith(f"primary failure: {message};")


@pytest.mark.parametrize("failure", ["create-policy", "cleanup-export"])
def test_create_wrapper_preserves_typed_recovery_error(failure):
    from tests.runtime.test_judge import RecordingJudgeResourceObserver

    ref = ContainerRef(container_id="judge", role="judge")

    def fail_policy(*_args, **_kwargs):
        error_type = (
            RecoveryRequiredError if failure == "create-policy" else RuntimeError
        )
        raise error_type("policy failed")

    def remove(_container):
        if failure == "cleanup-export":
            raise RecoveryRequiredError("export failed")

    wrapper = object.__new__(DockerJudgeRoundRuntime)
    wrapper._runtime = SimpleNamespace(
        create=lambda *_a, **_kw: ref,
        remove=remove,
        stop=lambda _ref: None,
        is_stopped_or_gone=lambda _ref: True,
    )
    wrapper._observer = RecordingJudgeResourceObserver()
    wrapper._round_id = "agent-1"
    wrapper._planned_container_name = "judge"
    wrapper._enforcer = SimpleNamespace(plan=fail_policy)
    wrapper._policy = None
    wrapper._network = None
    wrapper._api_endpoints = ()
    wrapper._lease = None
    with pytest.raises(RecoveryRequiredError) as raised:
        wrapper.create(ContainerSpec(image="task", command=("sleep", "60")))
    assert raised.value.code == "infrastructure_error"
    assert not wrapper.safe_to_release_isolation


@pytest.mark.parametrize(
    "error_type,message,retain_lease",
    [
        (RecoveryRequiredError, "installation failed", True),
        (InfrastructureError, "recovery_required: legacy failure", False),
    ],
)
def test_create_retains_acquired_lease_for_typed_primary_after_successful_remove(
    error_type, message, retain_lease
):
    from tests.runtime.test_judge import RecordingJudgeResourceObserver

    ref = ContainerRef(container_id="judge", role="judge")
    lease = object()
    removed = []
    cleaned = []

    def fail_install(_container, acquired):
        assert acquired is lease
        raise error_type(message)

    wrapper = object.__new__(DockerJudgeRoundRuntime)
    wrapper._runtime = SimpleNamespace(
        create=lambda *_a, **_kw: ref,
        remove=removed.append,
        install_network_policy=fail_install,
    )
    wrapper._observer = RecordingJudgeResourceObserver()
    wrapper._round_id = "agent-1"
    wrapper._planned_container_name = "judge"
    wrapper._enforcer = SimpleNamespace(
        plan=lambda *_a, **_kw: lease,
        apply=lambda *_a, **_kw: lease,
        cleanup=cleaned.append,
    )
    wrapper._policy = None
    wrapper._network = None
    wrapper._api_endpoints = ()
    wrapper._lease = None
    with pytest.raises(error_type):
        wrapper.create(ContainerSpec(image="task", command=("sleep", "60")))
    assert removed == [ref]
    assert cleaned == ([] if retain_lease else [lease])
    assert wrapper._lease is (lease if retain_lease else None)
    assert not wrapper.safe_to_release_isolation


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_recovery_attests_platform_mount_and_exports_before_removal(
    tmp_path, monkeypatch, platform
):
    runtime, client, ref, _ = make_judge(tmp_path, monkeypatch, platform=platform)
    container = client.containers.get(ref.container_id)
    # Simulate an inherited malicious image label, even for an older Linux Judge.
    container.attrs["Config"]["Labels"].update(
        {desktop.PRIVATE_LOG_LABEL: "volume", "rsi-harness.round-id": "round-1"}
    )
    backend = ProductionRecoveryBackend(client, None, None)
    inspected = backend.inspect_container(ref.container_id)
    assert inspected["private_log_volume"] is (platform == "darwin")
    calls = []
    backend.collect_private_judge_logs = lambda *a: calls.append("export")
    original_remove = backend.remove_container

    def remove(container_id):
        calls.append("remove")
        original_remove(container_id)

    backend.remove_container = remove
    original_inspect = backend.inspect_container
    backend.inspect_container = lambda i: (
        None if container.removed_kwargs else original_inspect(i)
    )
    manager = RecoveryManager(
        store=LeaseStore(tmp_path / "leases"), backend=backend, managed_root=tmp_path
    )
    lease = SimpleNamespace(
        artifact_root=tmp_path / "runs" / "r" / "t", run_id="r", task_id="t"
    )
    manager._stop_and_remove_judge(lease, ref.container_id, inspected)
    assert calls == (["export", "remove"] if platform == "darwin" else ["remove"])
    assert container.removed_kwargs == (
        {"force": False, "v": True} if platform == "darwin" else {"force": False}
    )


def test_recovery_rejects_label_without_volume(tmp_path, monkeypatch):
    _, client, ref, _ = make_judge(tmp_path, monkeypatch)
    client.containers.get(ref.container_id).attrs["Mounts"][0]["Type"] = "bind"
    backend = ProductionRecoveryBackend(client, None, None)
    with pytest.raises(InfrastructureError, match="Linux volume"):
        backend.inspect_container(ref.container_id)


def test_recovery_of_rejected_content_quarantines_old_reward(tmp_path, monkeypatch):
    import json
    import tarfile

    from rsi_harness.models import RunStatus
    from rsi_harness.runtime.reward import read_reward
    from tests.runtime.test_recovery import RecoveryBackend, lease

    monkeypatch.setattr(desktop, "sys", SimpleNamespace(platform="darwin"))
    root = tmp_path / "logs/runs/run-1/task-1"
    target = root / "verifier/agent-1"
    target.mkdir(parents=True)
    (target / "reward.json").write_text('{"reward": 1}')
    store = LeaseStore(tmp_path / "leases")
    store.write(lease(tmp_path).model_copy(update={"artifact_root": root}))
    backend = RecoveryBackend()
    backend.containers["judge-real"]["private_log_volume"] = True
    backend.containers["judge-real"]["labels"]["rsi-harness.round-id"] = "agent-1"
    exporter = ProductionRecoveryBackend(
        client_for([("verifier/unsafe", tarfile.SYMTYPE, b"")]), None, None
    )
    backend.collect_private_judge_logs = exporter.collect_private_judge_logs
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )
    manager.recover("run-1")
    assert store.read("run-1").status == RunStatus.CANCELLED
    assert store.read("run-1").recovery_required is False
    assert "judge-real" not in backend.containers
    assert read_reward(target, None).score is None
    diagnostic = json.loads(
        target.with_name("agent-1.private-log-export-error.json").read_text()
    )
    assert diagnostic["score"] is None
    assert json.loads(
        (Path(diagnostic["previous_export"]) / "reward.json").read_text()
    ) == {"reward": 1}


@pytest.mark.parametrize("content_error", [False, True])
def test_recovery_content_rejection_is_terminal_but_transport_remains_retryable(
    tmp_path, monkeypatch, content_error
):
    import rsi_harness.runtime.production as production

    backend = ProductionRecoveryBackend(None, None, None)
    error = (
        desktop.PrivateLogContentError("rejected")
        if content_error
        else InfrastructureError("transport")
    )

    def fail(*args):
        raise error

    monkeypatch.setattr(production, "export_private_logs", fail)
    if content_error:
        backend.collect_private_judge_logs("judge", tmp_path)
    else:
        with pytest.raises(InfrastructureError):
            backend.collect_private_judge_logs("judge", tmp_path)


@pytest.mark.parametrize("reason", ["gpu", "daemon"])
def test_production_preflight_rejects_unsupported_mac_before_inventory_or_run(
    tmp_path, monkeypatch, reason
):
    from rsi_harness.models import CompileOptions, RunRequest
    from rsi_harness.runtime import production
    from rsi_harness.runtime.production import ProductionRuntimeServices
    from tests.integration.test_oracle_docker import _task

    task = _task(tmp_path)
    if reason == "gpu":
        config = task / "task.toml"
        config.write_text(config.read_text().replace("gpus = 0", "gpus = 1"))
    monkeypatch.setattr(desktop, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(production, "sys", SimpleNamespace(platform="darwin"))
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=SimpleNamespace(list_devices=lambda: pytest.fail("inventory called")),
    )
    monkeypatch.setattr(
        services,
        "_client",
        lambda: SimpleNamespace(
            api=SimpleNamespace(base_url="http+docker://localhost"),
            info=lambda: {
                "OperatingSystem": "Docker Desktop" if reason == "gpu" else "Linux"
            },
        ),
    )
    with pytest.raises(SetupError, match="CPU-only|requires the Docker Desktop"):
        services.run(
            RunRequest(
                task_dir=task,
                agent_name="oracle",
                options=CompileOptions(agent_name="oracle"),
            )
        )
    assert not (tmp_path / "data").exists()
    if reason == "daemon":
        with pytest.raises(SetupError, match="requires the Docker Desktop"):
            services.recover(None)


def test_unsupported_mac_workspace_rejected_before_run_plan_creation(
    tmp_path, monkeypatch
):
    from rsi_harness.runtime.production import _ProductionRunComposition

    monkeypatch.setattr(desktop, "sys", SimpleNamespace(platform="darwin"))
    composition = object.__new__(_ProductionRunComposition)
    composition.client = SimpleNamespace(
        info=lambda: {"OperatingSystem": "Docker Desktop"}
    )
    with pytest.raises(SetupError, match="split WORKDIR"):
        composition.prepare_plan(
            None,
            SimpleNamespace(rootfs_snapshot_mode="full-rootfs", workdir="/"),
            None,
            None,
            "run-1",
        )
    assert not hasattr(composition, "run_id")
