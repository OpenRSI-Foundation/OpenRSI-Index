from __future__ import annotations

import copy
import re
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import docker as docker_sdk
import pytest
from docker.errors import DockerException

from rsi_harness.models import (
    CompileOptions,
    ContainerSpec,
    ContainerVolumeMount,
    EvaluationRequest,
    GPUAllocation,
    GPUDevice,
    ImagePlan,
    JudgeGPUMode,
    RootfsSnapshotMode,
    RunGPUPlan,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.judge import JudgeRunner
from rsi_harness.runtime.rootfs_snapshot import DockerRootfsSnapshotBackend
from rsi_harness.runtime.workdir_volume import DockerWorkdirVolumeBackend
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_harness.task.digest import hash_tree
from tests.factories import make_run_plan
from tests.integration.test_shared_root_judge import (
    _LifecycleObserver,
    _NetworkNoneJudgeFactory,
    _remove_container,
    _SpecRecordingJudgeFactory,
    _tree_identity,
)

_FIXTURE = (
    Path(__file__).parents[1] / "fixtures" / "tasks" / "minimal-split-workdir"
)
_SPARSE_SIZE = 8 * 1024**3
_FAKE_UUIDS = (
    "GPU-generic-fake-one",
    "GPU-generic-fake-two",
    "GPU-generic-fake-three",
    "GPU-generic-fake-four",
)
_WORKDIR = PurePosixPath("/workspace")


class _SplitLifecycleObserver(_LifecycleObserver):
    def __init__(self, client: Any) -> None:
        super().__init__(client)
        self.image_histories: list[list[dict[str, object]]] = []

    def resource_event(self, name: str, **values: object) -> None:
        if name == "snapshot_acquired":
            image_id = str(values["snapshot_image_id"])
            self.image_histories.append(copy.deepcopy(self.client.api.history(image_id)))
        super().resource_event(name, **values)


def _exact_filters(run_id: str, task_id: str) -> dict[str, list[str]]:
    return {
        "label": [
            f"rsi-harness.run-id={run_id}",
            f"rsi-harness.task-id={task_id}",
        ]
    }


def _cleanup_exact_resources(client: Any, *, run_id: str, task_id: str) -> None:
    filters = _exact_filters(run_id, task_id)
    for container in client.containers.list(all=True, filters=filters):
        _remove_container(container)
    for network in client.networks.list(filters=filters):
        try:
            network.remove()
        except DockerException:
            pass
    for volume in client.volumes.list(filters=filters):
        try:
            volume.remove(force=True)
        except DockerException:
            pass
    for image in client.images.list(filters=filters):
        try:
            client.images.remove(image.id, force=True)
        except DockerException:
            pass


def _assert_exact_resources_absent(client: Any, *, run_id: str, task_id: str) -> None:
    filters = _exact_filters(run_id, task_id)
    assert client.containers.list(all=True, filters=filters) == []
    assert client.networks.list(filters=filters) == []
    assert client.volumes.list(filters=filters) == []
    assert client.images.list(filters=filters) == []


def _mount_at(mounts: object, target: PurePosixPath) -> dict[str, Any]:
    assert isinstance(mounts, list)
    matches = [mount for mount in mounts if mount.get("Destination") == str(target)]
    assert len(matches) == 1
    return matches[0]


def _read_checkpoint_metrics(log_dir: Path) -> tuple[int, int]:
    size, blocks = (log_dir / "checkpoint-metrics.txt").read_text().split()
    return int(size), int(blocks)


@pytest.mark.integration
def test_split_workdir_gpu_preserves_rw_work_and_ro_judges_without_rootfs_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Work and Judges must share one volume without committing its contents."""
    try:
        client = docker_sdk.from_env()
        client.ping()
        ubuntu = client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local cached Ubuntu Docker authority unavailable: {error}")

    definition = HarborTaskCompiler().compile(
        _FIXTURE, CompileOptions(primary_reward="reward")
    )
    assert definition.workdir == _WORKDIR
    source_digest = hash_tree(_FIXTURE)
    source_identity = _tree_identity(_FIXTURE)
    ubuntu_id = ubuntu.id
    run_id = f"split-real-{uuid4().hex}"
    task_id = definition.task_id
    fixture_ref = f"rsi-harness-split-fixture:{uuid4().hex}"
    data_root = tmp_path / "engine-data"
    data_root.mkdir()
    snapshot = DockerRootfsSnapshotBackend(client)
    volume_backend = DockerWorkdirVolumeBackend(client)
    work = None
    forbidden_transfer_calls: list[str] = []
    bounded_puts: list[tuple[str, str]] = []

    def forbid_rootfs_transfer(*args: object, **kwargs: object) -> None:
        del args, kwargs
        forbidden_transfer_calls.append("forbidden")
        raise AssertionError("split snapshots must not transfer the Work rootfs")

    for method in ("get_archive", "export", "import_image"):
        if hasattr(client.api, method):
            monkeypatch.setattr(client.api, method, forbid_rootfs_transfer)

    original_put_archive = client.api.put_archive

    def record_bounded_put(*args: object, **kwargs: object):
        container = str(kwargs.get("container", args[0] if args else ""))
        path = str(kwargs.get("path", args[1] if len(args) > 1 else ""))
        bounded_puts.append((container, path))
        return original_put_archive(*args, **kwargs)

    monkeypatch.setattr(client.api, "put_archive", record_bounded_put)

    try:
        fixture_image, _build_logs = client.images.build(
            path=str(_FIXTURE / "environment"),
            dockerfile="Dockerfile",
            tag=fixture_ref,
            labels={
                "rsi-harness.run-id": run_id,
                "rsi-harness.task-id": task_id,
                "rsi-harness.role": "task-fixture",
            },
            rm=True,
        )
        fixture_image.reload()
        fixture_id = fixture_image.id
        assert fixture_image.attrs["Config"]["WorkingDir"] == str(_WORKDIR)
        assert not fixture_image.attrs["Config"].get("Volumes")

        allocation = GPUAllocation(
            devices=tuple(
                GPUDevice(index=index, uuid=uuid, name="test-only fake GPU")
                for index, uuid in enumerate(_FAKE_UUIDS)
            )
        )
        work_allocation = GPUAllocation(devices=allocation.devices[:2])
        judge_allocation = GPUAllocation(devices=allocation.devices[2:])
        planned_volume = volume_backend.plan(
            run_id=run_id,
            task_id=task_id,
            target=_WORKDIR,
        )
        volume = volume_backend.create(planned_volume)
        work_runtime = DockerContainerRuntime(
            client,
            run_id=run_id,
            task_id=task_id,
            role="work",
            task_source_dir=_FIXTURE,
            allowed_mount_roots=(data_root,),
            omit_gpu_device_requests_for_tests=True,
        )
        work_spec = ContainerSpec(
            image=fixture_id,
            command=("/bin/sleep", "infinity"),
            workdir=_WORKDIR,
            user="0:0",
            gpu_allocation=work_allocation,
            volume_mounts=(
                ContainerVolumeMount(
                    volume=volume,
                    target=_WORKDIR,
                    read_only=False,
                ),
            ),
        )
        work_ref = work_runtime.create(work_spec)
        work_runtime.attest_workdir_volume_mount(
            work_ref, work_spec.volume_mounts[0]
        )
        work = client.containers.get(work_ref.container_id)
        work.start()
        mutation = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "set -euo pipefail; "
                "printf 'round-1\\n' > /workspace/round-state.txt; "
                f"truncate -s {_SPARSE_SIZE} /workspace/checkpoint.bin; "
                "printf '#!/bin/sh\\nprintf agent-system-ready\\n' "
                "> /usr/local/bin/agent-system-tool; "
                "chmod 0755 /usr/local/bin/agent-system-tool",
            ]
        )
        assert mutation.exit_code == 0, mutation.output.decode(errors="replace")
        initial_stat = work.exec_run(
            ["/usr/bin/stat", "-c", "%s %b", "/workspace/checkpoint.bin"]
        )
        assert initial_stat.exit_code == 0
        work_checkpoint_size, work_checkpoint_blocks_before = map(
            int, initial_stat.output.decode().split()
        )
        assert work_checkpoint_size == _SPARSE_SIZE
        assert work_checkpoint_blocks_before <= 16

        plan = make_run_plan(data_root).model_copy(
            update={
                "task": definition,
                "workdir": _WORKDIR,
                "rootfs_snapshot_mode": RootfsSnapshotMode.SPLIT_WORKDIR,
                "gpu_plan": RunGPUPlan(
                    authorized_pool=allocation,
                    work=work_allocation,
                    judge=judge_allocation,
                    judge_mode=JudgeGPUMode.DISJOINT,
                ),
                "images": ImagePlan(
                    base_ref=fixture_id,
                    work_ref=fixture_id,
                    judge_ref=fixture_id,
                    workdir=_WORKDIR,
                    rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
                    work_user="0:0",
                    judge_user="0:0",
                    base_digest=fixture_id,
                    work_digest=fixture_id,
                    judge_digest=fixture_id,
                ),
            }
        )
        assert plan.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR
        observer = _SplitLifecycleObserver(client)
        judge_factory = _NetworkNoneJudgeFactory(
            client=client,
            run_id=run_id,
            task_id=task_id,
            task_source_dir=_FIXTURE,
            data_root=data_root,
            observer=observer,
            work_container=work_ref,
        )
        artifacts = RunArtifactWriter(plan, run_id=run_id)
        artifacts.start()
        runner = JudgeRunner(
            run_id=run_id,
            workdir_volume=volume,
            work_runtime=work_runtime,
            judge_runtime_factory=_SpecRecordingJudgeFactory(
                judge_factory, observer=observer
            ),
            snapshot_backend=snapshot,
            artifact_writer=artifacts,
            quiescence_checker=lambda _allocation, _container: None,
            lifecycle_observer=observer,
        )

        reports = []
        for round_number in (1, 2):
            round_id = f"agent-{round_number}"
            round_plan = plan.model_copy(
                update={
                    "task": plan.task.model_copy(
                        update={
                            "verifier": plan.task.verifier.model_copy(
                                update={
                                    "environment": (
                                        ("ROUND_EXPECTED", f"round-{round_number}"),
                                    )
                                }
                            )
                        }
                    )
                }
            )
            report = runner.evaluate(
                EvaluationRequest(
                    run_plan=round_plan,
                    work_container=work_ref,
                    round_id=round_id,
                    verifier_logs=plan.paths.logs / "verifier" / round_id,
                    verifier_output=artifacts.feedback_root / f"{round_id}.log",
                )
            )
            assert report.status == SubmissionStatus.COMPLETED, (
                report.error,
                report.output,
            )
            assert report.rewards == {"reward": 1, "split_workdir": 1}
            reports.append(report)
            work_state = work.exec_run(
                [
                    "/bin/bash",
                    "-lc",
                    "test \"$(cat /workspace/round-state.txt)\" = "
                    f"round-{round_number} "
                    "&& test ! -e /workspace/judge-write "
                    "&& test ! -e /etc/minimal-split-judge-only",
                ]
            )
            assert work_state.exit_code == 0
            if round_number == 1:
                update = work.exec_run(
                    [
                        "/bin/bash",
                        "-lc",
                        "printf 'round-2\\n' > /workspace/round-state.txt",
                    ]
                )
                assert update.exit_code == 0

        assert len(reports) == 2
        work.reload()
        work_mount = _mount_at(work.attrs["Mounts"], _WORKDIR)
        assert work_mount["Type"] == "volume"
        assert work_mount["Name"] == volume.name
        assert work_mount["RW"] is True
        assert work.attrs["HostConfig"].get("DeviceRequests") in (None, [])
        assert all(
            not str(value).startswith("RSI_HARNESS_EXPECTED_GPU_UUIDS=")
            for value in work.attrs["Config"]["Env"]
        )
        assert work_spec.gpu_allocation.uuids == (
            "GPU-generic-fake-one",
            "GPU-generic-fake-two",
        )

        final_stat = work.exec_run(
            ["/usr/bin/stat", "-c", "%s %b", "/workspace/checkpoint.bin"]
        )
        assert final_stat.exit_code == 0
        final_size, work_checkpoint_blocks_after = map(
            int, final_stat.output.decode().split()
        )
        assert final_size == _SPARSE_SIZE
        assert work_checkpoint_blocks_after == work_checkpoint_blocks_before

        assert len(observer.image_histories) == 2
        for round_id, history in zip(
            ("agent-1", "agent-2"), observer.image_histories, strict=True
        ):
            assert history
            committed_top_layer_bytes = int(history[0]["Size"])
            assert committed_top_layer_bytes < 64 * 1024**2
            judge_checkpoint_size, judge_checkpoint_blocks = _read_checkpoint_metrics(
                plan.paths.logs / "verifier" / round_id
            )
            assert judge_checkpoint_size == _SPARSE_SIZE
            assert judge_checkpoint_blocks == work_checkpoint_blocks_before
            assert (
                plan.paths.logs / "verifier" / round_id / "system-tool.txt"
            ).read_text() == "agent-system-ready"
            judge_workdir_write_exit_code = int(
                (
                    plan.paths.logs
                    / "verifier"
                    / round_id
                    / "workdir-write-exit-code.txt"
                ).read_text()
            )
            assert judge_workdir_write_exit_code != 0
            assert tuple(
                (
                    plan.paths.logs / "verifier" / round_id / "judge-gpus.txt"
                ).read_text().strip().split(",")
            ) == ("GPU-generic-fake-three", "GPU-generic-fake-four")
            judge_spec = observer.specs[round_id]
            assert judge_spec.gpu_allocation.uuids == (
                "GPU-generic-fake-three",
                "GPU-generic-fake-four",
            )
            judge_attrs = observer.judges[round_id]
            judge_mount = _mount_at(judge_attrs["mounts"], _WORKDIR)
            assert judge_mount["Type"] == "volume"
            assert judge_mount["Name"] == volume.name
            assert judge_mount["RW"] is False
            assert judge_attrs["host_config"].get("DeviceRequests") in (None, [])
            assert all(
                not str(value).startswith("RSI_HARNESS_EXPECTED_GPU_UUIDS=")
                for value in judge_attrs["config"]["Env"]
            )
            assert all(
                not (
                    mount["Type"] == "bind"
                    and mount["Destination"] == str(_WORKDIR)
                )
                for mount in judge_attrs["mounts"]
            )
            print(
                f"split round={round_id} checkpoint_size={judge_checkpoint_size} "
                f"checkpoint_blocks={judge_checkpoint_blocks} "
                f"committed_top_layer_bytes={committed_top_layer_bytes} "
                f"work_mount_rw={work_mount['RW']} "
                f"judge_mount_rw={judge_mount['RW']} "
                f"gpu_uuids={judge_allocation.uuids}"
            )

        assert len(bounded_puts) == 2
        assert all(
            re.fullmatch(r"/tmp/\.rsi-harness-tests-[0-9a-f]{32}", path)
            for _container, path in bounded_puts
        )
        assert forbidden_transfer_calls == []
        assert hash_tree(_FIXTURE) == source_digest
        assert _tree_identity(_FIXTURE) == source_identity
        assert client.images.get(fixture_ref).id == fixture_id
        assert client.images.get("ubuntu:24.04").id == ubuntu_id
        assert not list(tmp_path.rglob("*.tar*"))
    finally:
        _remove_container(work)
        _cleanup_exact_resources(client, run_id=run_id, task_id=task_id)
        _assert_exact_resources_absent(client, run_id=run_id, task_id=task_id)
