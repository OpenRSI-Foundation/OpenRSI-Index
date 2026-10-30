"""Public Judge visibility: Work reads the same task tests the Judge runs."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from docker.types import Mount
from pydantic import ValidationError

from rsi_harness.cluster.bluevela.runtime import ApptainerAgentRuntime
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    EvaluationRequest,
    ManagedNetwork,
    RootfsSnapshotMode,
    RunPlan,
    RunRequest,
    RunResult,
    RunStatus,
    SubmissionStatus,
    VerifierPlan,
)
from rsi_harness.runtime.judge import judge_uses_clean_image
from rsi_harness.runtime.network import UNREACHABLE_DNS_SERVER
from rsi_harness.task.digest import hash_tree, require_tests_digest
from tests.factories import make_run_plan
from tests.fakes import FakeDockerClient
from tests.runtime.test_docker import make_runtime
from tests.runtime.test_judge import evaluate, make_harness

TESTS = PurePosixPath("/tests")


def _write_tests(task_dir: Path) -> Path:
    tests = task_dir / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (tests / "test.sh").write_text("#!/bin/bash\necho judged\n")
    (tests / "rubric.json").write_text('{"rubrics": []}\n')
    return tests


def _with_visibility(
    plan, visibility: str, *, tests_digest: str | None = None, clean_image=False
):
    update: dict[str, object] = {
        "verifier": plan.task.verifier.model_copy(
            update={"visibility": visibility, "clean_image": clean_image}
        )
    }
    if tests_digest is not None:
        update["tests_digest"] = tests_digest
    return plan.model_copy(update={"task": plan.task.model_copy(update=update)})


def test_verifier_plan_defaults_to_hidden_and_loads_plans_without_the_field():
    plan = VerifierPlan(command=("/bin/bash", "/tests/test.sh"))
    persisted = plan.model_dump(mode="json")
    persisted.pop("visibility")

    assert plan.visibility == "hidden"
    assert VerifierPlan.model_validate(persisted).visibility == "hidden"
    with pytest.raises(ValidationError):
        VerifierPlan(command=("/bin/bash", "/tests/test.sh"), visibility="shared")


def test_tests_digest_check_detects_any_change(tmp_path: Path) -> None:
    tests = _write_tests(tmp_path)
    digest = hash_tree(tests)
    require_tests_digest(tests, digest)

    (tests / "rubric.json").write_text('{"rubrics": ["changed"]}\n')
    with pytest.raises(SetupError, match="recorded tests digest"):
        require_tests_digest(tests, digest)


# Docker runtime authority


def test_public_work_binds_exact_task_tests_read_only(tmp_path: Path) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, work_public_tests=True)
    tests = _write_tests(tmp_path / "task-source")

    ref = runtime.create(
        ContainerSpec(
            image="work",
            mounts=(ContainerMount(source=tests, target=TESTS, read_only=True),),
        )
    )
    runtime.attest_work_tests_mount(ref)

    assert client.containers.created[0]["mounts"] == [
        Mount(
            source=str(tests.resolve()),
            target="/tests",
            type="bind",
            read_only=True,
        )
    ]


def test_hidden_work_cannot_mount_task_tests(tmp_path: Path) -> None:
    runtime = make_runtime(FakeDockerClient(), tmp_path)
    tests = _write_tests(tmp_path / "task-source")

    with pytest.raises(SetupError, match="task source"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(ContainerMount(source=tests, target=TESTS, read_only=True),),
            )
        )
    with pytest.raises(InfrastructureError, match="tests mount attestation"):
        runtime.attest_work_tests_mount(ContainerRef(container_id="c", role="work"))


def test_public_work_tests_mount_must_be_exact_and_read_only(tmp_path: Path) -> None:
    runtime = make_runtime(FakeDockerClient(), tmp_path, work_public_tests=True)
    tests = _write_tests(tmp_path / "task-source")
    other = tmp_path / "engine-root" / "tests"
    other.mkdir()

    with pytest.raises(SetupError, match="task source"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(ContainerMount(source=tests, target=TESTS, read_only=False),),
            )
        )
    with pytest.raises(SetupError, match="mount target"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(ContainerMount(source=other, target=TESTS, read_only=True),),
            )
        )
    with pytest.raises(SetupError, match="task source"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(
                    ContainerMount(
                        source=tests,
                        target=PurePosixPath("/run/rsi-harness/staging"),
                        read_only=True,
                    ),
                ),
            )
        )


def test_public_tests_authority_is_limited_to_work(tmp_path: Path) -> None:
    with pytest.raises(SetupError, match="limited to Work"):
        make_runtime(FakeDockerClient(), tmp_path, role="judge", work_public_tests=True)


def _tests_mount(kind: str, source: str, rw: bool) -> dict[str, object]:
    return {"Type": kind, "Source": source, "Destination": "/tests", "RW": rw}


@pytest.mark.parametrize(
    "actual_mounts",
    (
        (),
        (_tests_mount("bind", "{tests}", True),),
        (_tests_mount("bind", "{other}", False),),
        (_tests_mount("volume", "{tests}", False),),
        (
            _tests_mount("bind", "{tests}", False),
            _tests_mount("bind", "{tests}", False),
        ),
    ),
)
def test_public_work_tests_mount_is_attested_from_created_container(
    tmp_path: Path, actual_mounts
) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, work_public_tests=True)
    tests = _write_tests(tmp_path / "task-source")
    other = tmp_path / "engine-root" / "other"
    other.mkdir()
    ref = runtime.create(
        ContainerSpec(
            image="work",
            mounts=(ContainerMount(source=tests, target=TESTS, read_only=True),),
        )
    )
    client.containers.by_id[ref.container_id].attrs["Mounts"] = [
        {
            key: value.format(tests=tests.resolve(), other=other.resolve())
            if isinstance(value, str)
            else value
            for key, value in mount.items()
        }
        for mount in actual_mounts
    ]

    with pytest.raises(InfrastructureError, match="tests mount attestation"):
        runtime.attest_work_tests_mount(ref)


# Production Work composition


def _production_composition(tmp_path: Path):
    from rsi_harness.config import EngineConfig
    from rsi_harness.runtime.production import ProductionRuntimeServices
    from rsi_loop.harness.config import RSILoopConfig
    from tests.integration.test_run_engine import FIXTURE, _OneDeviceInventory

    captured = []

    class Coordinator:
        def __init__(self, *, backend, lease_store, clock) -> None:
            del lease_store, clock
            captured.append(backend)

        def run(self, request: RunRequest) -> RunResult:
            del request
            return RunResult(run_id="public", status=RunStatus.COMPLETED)

    engine_config = EngineConfig(
        data_root=tmp_path / "data", logs_root=tmp_path / "logs"
    )
    ProductionRuntimeServices(
        data_root=engine_config.data_root,
        logs_root=engine_config.logs_root,
        docker_client=object(),
        inventory=_OneDeviceInventory(),
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=Coordinator,
        bridge_gateway="127.0.0.1",
        engine_config=engine_config,
    ).run(RunRequest(task_dir=FIXTURE.resolve()))
    return captured[0].work_creator.__self__


class _RecordingWorkRuntime:
    def __init__(self) -> None:
        self.spec: ContainerSpec | None = None
        self.attested: list[str] = []

    def create(self, spec, *, planned_name):
        del planned_name
        self.spec = spec
        return ContainerRef(container_id="work-1", role="work")

    def attest_work_feedback_mount(self, work) -> None:
        self.attested.append("feedback")

    def attest_work_tests_mount(self, work) -> None:
        self.attested.append("tests")


def _create_work(composition, plan):
    composition.artifacts = composition.start_artifacts(plan, "run-1")
    composition.workdir_volume = None
    plan = plan.model_copy(
        update={
            "rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS,
            "workdir": PurePosixPath("/"),
        }
    )
    return composition.create_work(
        plan,
        "run-1",
        ManagedNetwork(
            network_id="network-1",
            name="network-1",
            run_id="run-1",
            task_id=plan.task.task_id,
            role="work",
            internal=False,
        ),
        "planned-work",
        None,
    )


def test_production_public_work_binds_digest_verified_tests(tmp_path: Path) -> None:
    composition = _production_composition(tmp_path)
    plan = make_run_plan(tmp_path / "plan")
    tests = _write_tests(plan.task.source_dir)
    plan = _with_visibility(plan, "public", tests_digest=hash_tree(tests))
    runtime = _RecordingWorkRuntime()
    composition.work_runtime = runtime

    work = _create_work(composition, plan)
    composition.attest_work_feedback_mount(work)

    assert runtime.spec is not None
    assert (
        ContainerMount(source=tests, target=TESTS, read_only=True)
        in runtime.spec.mounts
    )
    assert runtime.attested == ["feedback", "tests"]


def test_production_public_work_refuses_changed_tests(tmp_path: Path) -> None:
    composition = _production_composition(tmp_path)
    plan = make_run_plan(tmp_path / "plan")
    tests = _write_tests(plan.task.source_dir)
    plan = _with_visibility(plan, "public", tests_digest=hash_tree(tests))
    (tests / "test.sh").write_text("#!/bin/bash\necho edited\n")
    runtime = _RecordingWorkRuntime()
    composition.work_runtime = runtime

    with pytest.raises(InfrastructureError, match="tests digest"):
        _create_work(composition, plan)
    assert runtime.spec is None


@pytest.mark.parametrize("clean_image", [False, True])
def test_production_hidden_work_never_binds_tests(
    tmp_path: Path, clean_image: bool
) -> None:
    composition = _production_composition(tmp_path)
    plan = make_run_plan(tmp_path / "plan")
    _write_tests(plan.task.source_dir)
    plan = _with_visibility(plan, "hidden", clean_image=clean_image)
    runtime = _RecordingWorkRuntime()
    composition.work_runtime = runtime

    work = _create_work(composition, plan)
    composition.attest_work_feedback_mount(work)

    assert runtime.spec is not None
    assert all(mount.target != TESTS for mount in runtime.spec.mounts)
    assert runtime.attested == ["feedback"]


def test_production_no_network_work_gets_no_external_dns(tmp_path: Path) -> None:
    composition = _production_composition(tmp_path)
    plan = make_run_plan(tmp_path / "plan")
    assert plan.task.agent.network.mode == "no-network"
    assert composition.api_endpoints
    runtime = _RecordingWorkRuntime()
    composition.work_runtime = runtime

    _create_work(composition, plan)

    assert runtime.spec is not None
    assert runtime.spec.dns == (UNREACHABLE_DNS_SERVER,)
    assert {host for host, _ in runtime.spec.extra_hosts} == {
        endpoint.hostname for endpoint in composition.api_endpoints
    }


# Judge rounds


# The clean Work image a public Judge starts from: no Agent changes in it.
CLEAN_WORK_IMAGE = {"/etc/task-image-state": "clean task image"}


def _judge_with_visibility(
    tmp_path: Path,
    visibility: str,
    *,
    digest: str | None,
    images: dict[str, object] | None = None,
    clean_image: bool = False,
):
    runner, runtime, _snapshot, _artifacts, request = make_harness(tmp_path)
    plan, work, log_dir = request
    tests = plan.task.source_dir / "tests"
    plan = _with_visibility(
        plan,
        visibility,
        tests_digest=hash_tree(tests) if digest is None else digest,
        clean_image=clean_image,
    )
    if images:
        plan = plan.model_copy(update={"images": plan.images.model_copy(update=images)})
    if plan.images.work_digest:
        runtime.snapshot_images[plan.images.work_digest] = CLEAN_WORK_IMAGE.copy()
    return runner, runtime, (plan, work, log_dir)


def test_public_judge_runs_tests_matching_the_recorded_digest(tmp_path: Path) -> None:
    runner, runtime, request = _judge_with_visibility(tmp_path, "public", digest=None)

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert ("tests_inject", "agent-1") in runtime.events


def test_public_judge_refuses_tests_that_differ_from_what_work_saw(
    tmp_path: Path,
) -> None:
    runner, runtime, request = _judge_with_visibility(
        tmp_path, "public", digest="0" * 64
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert "tests digest" in (report.error or "")
    assert ("tests_inject", "agent-1") not in runtime.events
    assert runtime.events[-1] == ("unpause", "work-1")


def test_public_judge_runs_from_the_clean_work_image_read_only(
    tmp_path: Path,
) -> None:
    runner, runtime, request = _judge_with_visibility(tmp_path, "public", digest=None)
    plan, _work, _logs = request

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    spec = runtime.created_specs[0]
    assert spec.image == plan.images.work_digest
    assert spec.image != runner._snapshot_backend.acquired[0].image_id
    # Nothing the Agent changed in Work's rootfs reaches the Judge.
    assert runtime.judge_rootfs == CLEAN_WORK_IMAGE
    assert spec.read_only_rootfs is True
    assert {str(item.target): item.options for item in spec.tmpfs} == {
        "/tests": "rw,exec,nosuid,nodev,mode=0755",
        "/tmp": "rw,nosuid,nodev,noexec,size=256m,mode=1777",
    }
    assert [(m.target, m.read_only) for m in spec.volume_mounts] == [
        (plan.workdir, True)
    ]


def test_hidden_judge_runs_from_the_clean_work_image_with_a_writable_layer(
    tmp_path: Path,
) -> None:
    # A stale digest: the public tests check stays public-only.
    runner, runtime, request = _judge_with_visibility(
        tmp_path, "hidden", digest="0" * 64, clean_image=True
    )
    plan, _work, _logs = request

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    spec = runtime.created_specs[0]
    assert spec.image == plan.images.work_digest
    assert spec.image != runner._snapshot_backend.acquired[0].image_id
    assert runtime.judge_rootfs == CLEAN_WORK_IMAGE
    # Not the public hardening: the container's own layer stays writable and
    # the image's /tmp stays as built; Docker discards both with the Judge.
    assert spec.read_only_rootfs is False
    assert [str(item.target) for item in spec.tmpfs] == ["/tests"]
    assert [(m.target, m.read_only) for m in spec.volume_mounts] == [
        (plan.workdir, True)
    ]
    assert ("tests_inject", "agent-1") in runtime.events


def test_hidden_judge_without_clean_image_runs_from_the_work_snapshot(
    tmp_path: Path,
) -> None:
    runner, runtime, request = _judge_with_visibility(tmp_path, "hidden", digest=None)

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    spec = runtime.created_specs[0]
    assert spec.image == runner._snapshot_backend.acquired[0].image_id
    assert runtime.judge_rootfs != CLEAN_WORK_IMAGE
    assert spec.read_only_rootfs is False
    assert [str(item.target) for item in spec.tmpfs] == ["/tests"]


def test_a_run_plan_persisted_before_clean_image_keeps_the_snapshot_judge(
    tmp_path: Path,
) -> None:
    persisted = make_run_plan(tmp_path).model_dump(mode="json")
    del persisted["task"]["verifier"]["clean_image"]

    plan = RunPlan.model_validate(persisted)

    assert plan.task.verifier.clean_image is False
    assert judge_uses_clean_image(plan) is False
    public = _with_visibility(plan, "public")
    assert judge_uses_clean_image(public) is True


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ({"rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS}, "WORKDIR is /"),
        ({"work_digest": None}, "no immutable digest"),
        ({"work_digest": "work:latest"}, "no immutable digest"),
        ({"work_digest": "sha256:abc"}, "no immutable digest"),
    ],
)
@pytest.mark.parametrize(
    ("visibility", "clean_image"), [("public", False), ("hidden", True)]
)
def test_clean_image_judge_refuses_without_a_clean_image_before_create(
    tmp_path: Path, images, message, visibility, clean_image
) -> None:
    # A hidden plan only carries clean_image with an unusable image when it
    # skipped preflight; the Judge never silently uses the snapshot instead.
    runner, runtime, request = _judge_with_visibility(
        tmp_path, visibility, digest=None, images=images, clean_image=clean_image
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.INFRASTRUCTURE_ERROR
    assert message in (report.error or "")
    assert runtime.created_specs == []
    assert runtime.work_paused is False


def _prepare_with_images(tmp_path: Path, monkeypatch, images):
    from rsi_harness.runtime import production

    composition = _production_composition(tmp_path)
    composition.preparation = None
    events: list[tuple[str, object]] = []
    composition.event_callback = lambda name, value: events.append((name, value))
    plan = make_run_plan(tmp_path / "plan")
    prepared = plan.images.model_copy(update=images)

    class Builder:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def prepare(self, definition, *, agent):
            del definition, agent
            return prepared

    monkeypatch.setattr(production, "DockerImageBuilder", Builder)

    run_ids = iter(range(10))

    def prepare(task):
        events.clear()
        images = composition.prepare_images(task)
        assert images is prepared
        return composition.prepare_plan(
            task,
            images,
            plan.gpu_plan,
            RunRequest(task_dir=task.source_dir),
            f"run-{next(run_ids)}",
        )

    return plan, prepare, events


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ({"rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS}, "WORKDIR is /"),
        ({"work_digest": None}, "no immutable digest"),
    ],
)
def test_production_refuses_a_public_task_without_a_clean_judge_image(
    tmp_path: Path, monkeypatch, images, message
) -> None:
    plan, prepare, events = _prepare_with_images(tmp_path, monkeypatch, images)

    with pytest.raises(SetupError, match=f"public Judge.*{message}"):
        prepare(_with_visibility(plan, "public").task)
    assert ("images_ready", None) not in events


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ({"rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS}, "WORKDIR is /"),
        ({"work_digest": None}, "no immutable digest"),
    ],
)
def test_production_falls_back_to_the_snapshot_judge_loudly(
    tmp_path: Path, monkeypatch, capsys, images, message
) -> None:
    from rsi_harness.cli import _RunConsole

    plan, prepare, events = _prepare_with_images(tmp_path, monkeypatch, images)

    final = prepare(_with_visibility(plan, "hidden", clean_image=True).task)

    # Decided before any run state, and recorded in the run plan.
    [(name, reason)] = [event for event in events if event[0].startswith("judge")]
    assert name == "judge_clean_image_unavailable"
    assert message in str(reason)
    assert events.index((name, reason)) < events.index(("images_ready", None))
    assert final.task.verifier.clean_image is False
    assert judge_uses_clean_image(final) is False
    _RunConsole()(name, reason)
    assert "Warning:" in capsys.readouterr().out

    # An explicit opt-out is the same Judge without the warning.
    final = prepare(_with_visibility(plan, "hidden", clean_image=False).task)
    assert not [event for event in events if event[0].startswith("judge")]
    assert final.task.verifier.clean_image is False


def test_production_keeps_the_clean_judge_when_its_image_exists(
    tmp_path: Path, monkeypatch
) -> None:
    plan, prepare, events = _prepare_with_images(tmp_path, monkeypatch, {})

    # A public plan records the clean image whatever the field says.
    for visibility, clean_image in (
        ("hidden", True),
        ("public", True),
        ("public", False),
    ):
        task = _with_visibility(plan, visibility, clean_image=clean_image).task
        final = prepare(task)
        assert not [event for event in events if event[0].startswith("judge")]
        assert final.task.verifier.clean_image is True
        assert judge_uses_clean_image(final) is True


def test_hidden_judge_keeps_injecting_without_a_digest_check(tmp_path: Path) -> None:
    runner, runtime, request = _judge_with_visibility(
        tmp_path, "hidden", digest="0" * 64
    )

    report = evaluate(runner, request)

    assert report.status == SubmissionStatus.COMPLETED
    assert ("tests_inject", "agent-1") in runtime.events


# Cluster (Apptainer) runtime


def _cluster_runtime(tmp_path: Path, monkeypatch, visibility: str, digest=None):
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    plan = make_run_plan(tmp_path)
    tests = _write_tests(plan.task.source_dir)
    plan = _with_visibility(
        plan, visibility, tests_digest=hash_tree(tests) if digest is None else digest
    )
    runtime = ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        plan,
    )
    return runtime, tests


def _binds(runtime: ApptainerAgentRuntime, *, mount_workspace: bool, phase: str):
    command = runtime._base_command(
        devices=(),
        extra_binds=(),
        mount_workspace=mount_workspace,
        mount_agent_home=mount_workspace,
        network_mode="no-network",
        phase=phase,
    )
    pairs = zip(command, command[1:], strict=False)
    return {value for option, value in pairs if option == "--bind"}


@pytest.mark.parametrize("visibility", ["hidden", "public"])
def test_cluster_work_binds_tests_read_only_only_when_public(
    tmp_path: Path, monkeypatch, visibility
) -> None:
    runtime, tests = _cluster_runtime(tmp_path, monkeypatch, visibility)
    expected = f"{tests}:/tests:ro"

    work = _binds(runtime, mount_workspace=True, phase="work")
    judge_setup = _binds(runtime, mount_workspace=False, phase="judge")

    assert (expected in work) is (visibility == "public")
    assert not any(bind.endswith(":/tests:ro") for bind in judge_setup)


def test_cluster_public_runtime_refuses_changed_tests_before_work_or_judge(
    tmp_path: Path, monkeypatch
) -> None:
    runtime, _tests = _cluster_runtime(tmp_path, monkeypatch, "public", digest="0" * 64)

    def unexpected_run(*_args, **_kwargs):
        raise AssertionError("no Apptainer process may start")

    monkeypatch.setattr(runtime, "_run", unexpected_run)
    with pytest.raises(SetupError, match="tests digest"):
        runtime.initialize()
    assert not runtime.workspace.exists()
    request = EvaluationRequest(
        run_plan=runtime.plan,
        work_container=ContainerRef(container_id="work", role="work"),
        round_id="agent-1",
        verifier_logs=tmp_path / "verifier",
        verifier_output=tmp_path / "feedback.log",
    )
    with pytest.raises(SetupError, match="tests digest"):
        runtime.run_judge(tmp_path / "snapshot", request, {})


# Work prompt


def test_prompt_states_public_tests_only_for_public_visibility() -> None:
    gpu_plan = make_run_plan(Path("/plan")).gpu_plan

    hidden = RSILoopAgentAdapter._prompt("Solve it.", gpu_plan, None)
    public = RSILoopAgentAdapter._prompt("Solve it.", gpu_plan, None, public_tests=True)

    assert "/tests" not in hidden
    assert "`/tests` holds, read-only, exactly the" in public
    assert (
        public.replace(
            "\nThe Judge's tests are public: `/tests` holds, read-only, exactly the\n"
            "files the Judge runs as `/tests/test.sh`.\n",
            "",
        )
        == hidden
    )
