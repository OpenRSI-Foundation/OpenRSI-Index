from __future__ import annotations

from pathlib import PurePosixPath
from uuid import uuid4

import docker as docker_sdk
import pytest

from rsi_harness.models import (
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerTmpfs,
    ContainerVolumeMount,
    EvaluationRequest,
    ImagePlan,
    RootfsSnapshotMode,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.judge import DockerJudgeRuntimeFactory, JudgeRunner
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    NetworkPolicyEnforcer,
)
from rsi_harness.runtime.rootfs_snapshot import DockerRootfsSnapshotBackend
from rsi_harness.runtime.workdir_volume import DockerWorkdirVolumeBackend
from rsi_harness.task.digest import hash_tree
from tests.factories import make_run_plan, write_harbor_task


class _JudgeResourceObserver:
    def resource_event(self, name, **values):
        pass

    def judge_network_planned(self, round_id, planned_name):
        pass

    def judge_network_created(self, round_id, network):
        pass

    def judge_container_planned(self, round_id, planned_name):
        pass

    def judge_container_created(self, round_id, container):
        pass

    def judge_policy_planned(self, round_id, rule_id):
        pass

    def judge_policy_installed(self, round_id, lease):
        pass

    def judge_container_removed(self, round_id, container_id):
        pass

    def judge_policy_removed(self, round_id, rule_id):
        pass

    def judge_network_removed(self, round_id, network_id):
        pass


@pytest.mark.integration
def test_real_rootfs_snapshot_upload_is_private_and_leaves_no_resources(tmp_path):
    try:
        client = docker_sdk.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local cached Ubuntu Docker authority unavailable: {error}")

    task_dir = write_harbor_task(tmp_path / "source")
    data_root = tmp_path / "engine-data"
    verifier_logs = data_root / "logs" / "verifier"
    verifier_logs.mkdir(parents=True)
    source_identity = {
        path.relative_to(task_dir): (
            path.lstat().st_ino,
            path.lstat().st_mtime_ns,
        )
        for path in (task_dir, *task_dir.rglob("*"))
    }
    source_digest = hash_tree(task_dir)
    run_id = f"task4-real-{uuid4().hex}"
    task_id = "task4-rootfs-proof"
    snapshot = DockerRootfsSnapshotBackend(client)
    runtime = DockerContainerRuntime(
        client,
        run_id=run_id,
        task_id=task_id,
        role="judge",
        task_source_dir=task_dir,
        allowed_mount_roots=(data_root,),
    )
    work = client.containers.create(
        "ubuntu:24.04",
        ["/bin/sleep", "infinity"],
        network_mode="none",
        labels={
            "rsi-harness.run-id": run_id,
            "rsi-harness.task-id": task_id,
            "rsi-harness.role": "work",
        },
        detach=True,
    )
    work_ref = ContainerRef(container_id=work.id, role="work")
    active_judge = None
    active_lease = None
    try:
        work.start()
        prepared = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "mkdir -p /workspace && printf 'complete-work-state\\n' "
                "> /workspace/agent-state && touch /root-marker",
            ]
        )
        assert prepared.exit_code == 0
        work.pause()

        for round_id in ("proof-1", "proof-2"):
            planned_ref = snapshot.planned_ref(
                run_id=run_id,
                task_id=task_id,
                round_id=round_id,
                purpose="judge-round",
            )
            active_lease = snapshot.acquire(
                work_ref,
                run_id=run_id,
                task_id=task_id,
                round_id=round_id,
                purpose="judge-round",
                planned_ref=planned_ref,
            )
            active_judge = runtime.create(
                ContainerSpec(
                    image=active_lease.image_id,
                    command=("/bin/sleep", "infinity"),
                    mounts=(
                        ContainerMount(
                            source=verifier_logs,
                            target=PurePosixPath("/logs/verifier"),
                        ),
                    ),
                    tmpfs=(
                        ContainerTmpfs(
                            target=PurePosixPath("/tests"),
                            options="rw,exec,nosuid,nodev,mode=0755",
                        ),
                    ),
                )
            )
            # This proof deliberately exercises Docker-owned rootfs/test transfer
            # without requiring host firewall authority. Production JudgeRunner
            # uses runtime.start only after installing its fail-closed policy.
            client.containers.get(active_judge.container_id).start()
            runtime.inject_directory(
                active_judge,
                task_dir / "tests",
                PurePosixPath("/tests"),
            )
            private_path = (
                "/workspace/judge-only" if round_id == "proof-1" else "/dev/null"
            )
            checked = runtime.exec(
                active_judge,
                (
                    "/bin/bash",
                    "-lc",
                    "set -eux; test \"$(cat /workspace/agent-state)\" = "
                    "complete-work-state "
                    "&& test -e /root-marker && test -f /tests/test.sh "
                    "&& grep -q '/logs/verifier/reward.json' /tests/test.sh "
                    "&& test ! -e /workspace/judge-only "
                    f"&& printf private-write > {private_path}",
                ),
            )
            assert checked.exit_code == 0, checked.output
            runtime.remove(active_judge)
            active_judge = None
            snapshot.release(active_lease)
            active_lease = None

        work.reload()
        assert work.attrs["State"]["Paused"] is True
        assert work.attrs.get("Mounts", []) == []
        assert not (task_dir / "tests" / "private-write").exists()
        assert hash_tree(task_dir) == source_digest
        assert {
            path.relative_to(task_dir): (
                path.lstat().st_ino,
                path.lstat().st_mtime_ns,
            )
            for path in (task_dir, *task_dir.rglob("*"))
        } == source_identity
        judge_labels = [
            f"rsi-harness.run-id={run_id}",
            f"rsi-harness.task-id={task_id}",
            "rsi-harness.role=judge",
        ]
        snapshot_labels = [
            f"rsi-harness.run-id={run_id}",
            f"rsi-harness.task-id={task_id}",
            "rsi-harness.role=rootfs-snapshot",
        ]
        assert client.containers.list(all=True, filters={"label": judge_labels}) == []
        assert client.images.list(filters={"label": snapshot_labels}) == []
    finally:
        if active_judge is not None:
            runtime.remove(active_judge)
        if active_lease is not None:
            snapshot.release(active_lease)
        try:
            work.reload()
            if work.attrs["State"].get("Paused"):
                work.unpause()
        finally:
            work.remove(force=True, v=True)


@pytest.mark.integration
def test_real_disposable_judge_uses_full_work_snapshot_and_discards_writes(tmp_path):
    try:
        client = docker_sdk.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local cached Ubuntu Docker authority unavailable: {error}")

    firewall = DockerIptablesFirewallBackend(client)
    if not firewall.probe():
        pytest.skip(
            "host DOCKER-USER/INPUT firewall authority unavailable for "
            "fail-closed Judge"
        )

    task_dir = write_harbor_task(
        tmp_path / "source",
        test_script=(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "test \"$(cat /workspace/answer.txt)\" = 42\n"
            "test -e /root-marker\n"
            "test ! -e /workspace/judge-only.txt\n"
            "printf 'private-tests-write\\n' > /tests/private-write\n"
            "if printf 'judge-only\\n' > /workspace/judge-only.txt "
            "2>/dev/null; then exit 1; fi\n"
            "printf 'stdout-one\\n'\n"
            "printf 'stderr-two\\n' >&2\n"
            "printf '{\"reward\": 1, \"checks\": 4}\\n' > "
            "/logs/verifier/reward.json\n"
        ),
    )
    data_root = tmp_path / "engine-data"
    data_root.mkdir(parents=True)
    plan = make_run_plan(data_root)
    source_digest = hash_tree(task_dir)
    source_mtimes = {
        path.relative_to(task_dir): path.lstat().st_mtime_ns
        for path in task_dir.rglob("*")
    }
    plan = plan.model_copy(
        update={
            "task": plan.task.model_copy(
                update={
                    "source_dir": task_dir,
                    "source_digest": source_digest,
                    "tests_digest": hash_tree(task_dir / "tests"),
                }
            ),
            "images": ImagePlan(
                base_ref="ubuntu:24.04",
                work_ref="ubuntu:24.04",
                judge_ref="ubuntu:24.04",
                workdir=PurePosixPath("/workspace"),
                rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            ),
        }
    )
    snapshot = DockerRootfsSnapshotBackend(client)
    volume_backend = DockerWorkdirVolumeBackend(client)
    planned_volume = volume_backend.plan(
        run_id="run-real",
        task_id=plan.task.task_id,
        target=plan.workdir,
    )
    workdir_volume = volume_backend.create(planned_volume)

    work = client.containers.create(
        "ubuntu:24.04",
        ["/bin/sleep", "infinity"],
        network_mode="none",
        mounts=(
            docker_sdk.types.Mount(
                target=str(plan.workdir),
                source=workdir_volume.name,
                type="volume",
                read_only=False,
            ),
        ),
        labels={
            "rsi-harness.run-id": "run-real",
            "rsi-harness.task-id": plan.task.task_id,
            "rsi-harness.role": "work",
        },
        detach=True,
    )
    work.start()
    setup_result = work.exec_run(
        [
            "/bin/bash",
            "-lc",
            "mkdir -p /workspace && printf '42\\n' > /workspace/answer.txt "
            "&& touch /root-marker",
        ]
    )
    assert setup_result.exit_code == 0
    work_ref = ContainerRef(container_id=work.id, role="work")
    work_runtime = DockerContainerRuntime(
        client,
        run_id="run-real",
        task_id=plan.task.task_id,
        role="work",
        task_source_dir=task_dir,
        allowed_mount_roots=(data_root,),
    )
    work_runtime.attest_workdir_volume_mount(
        work_ref,
        ContainerVolumeMount(
            volume=workdir_volume,
            target=plan.workdir,
            read_only=False,
        ),
    )
    enforcer = NetworkPolicyEnforcer(
        run_id="run-real",
        firewall=firewall,
    )
    judge_factory = DockerJudgeRuntimeFactory(
        client,
        run_id="run-real",
        task_id=plan.task.task_id,
        task_source_dir=task_dir,
        allowed_mount_roots=(data_root,),
        network_policy_enforcer=enforcer,
        lifecycle_observer=_JudgeResourceObserver(),
        work_container=work_ref,
    )
    artifacts = RunArtifactWriter(plan, run_id="run-real")
    artifacts.start()
    runner = JudgeRunner(
        run_id="run-real",
        workdir_volume=workdir_volume,
        work_runtime=work_runtime,
        judge_runtime_factory=judge_factory,
        snapshot_backend=snapshot,
        artifact_writer=artifacts,
        quiescence_checker=lambda _allocation, _container: None,
    )
    try:
        first = runner.evaluate(
            EvaluationRequest(
                run_plan=plan,
                work_container=work_ref,
                round_id="agent-1",
                verifier_logs=plan.paths.logs / "verifier" / "agent-1",
                verifier_output=artifacts.feedback_root / "agent-1.log",
            )
        )
        second = runner.evaluate(
            EvaluationRequest(
                run_plan=plan,
                work_container=work_ref,
                round_id="agent-2",
                verifier_logs=plan.paths.logs / "verifier" / "agent-2",
                verifier_output=artifacts.feedback_root / "agent-2.log",
            )
        )

        work.reload()
        assert work.attrs["State"]["Paused"] is False
        assert first.status == second.status == SubmissionStatus.COMPLETED
        assert first.rewards == second.rewards == {"reward": 1, "checks": 4}
        assert first.score == second.score == 1
        assert "stdout-one\n" in first.output
        assert "stderr-two\n" in first.output
        assert "Read-only file system" in first.output
        work_state = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "test -e /root-marker && test ! -e /workspace/judge-only.txt "
                "&& test \"$(cat /workspace/answer.txt)\" = 42",
            ]
        )
        assert work_state.exit_code == 0
        assert not (task_dir / "tests" / "private-write").exists()
        assert hash_tree(task_dir) == source_digest
        assert {
            path.relative_to(task_dir): path.lstat().st_mtime_ns
            for path in task_dir.rglob("*")
        } == source_mtimes
        persisted = artifacts.root / "submissions" / "agent-1"
        assert (persisted / "test_output.txt").read_text() == first.output
        assert (persisted / "report.json").is_file()
        work.reload()
        assert [
            (mount["Name"], mount["Destination"], mount["RW"])
            for mount in work.attrs.get("Mounts", [])
            if mount.get("Type") == "volume"
        ] == [(workdir_volume.name, "/workspace", True)]
        judge_labels = [
            "rsi-harness.run-id=run-real",
            "rsi-harness.task-id=minimal-gpu",
            "rsi-harness.role=judge",
        ]
        assert client.containers.list(all=True, filters={"label": judge_labels}) == []
        assert client.networks.list(filters={"label": judge_labels}) == []
        snapshot_labels = [
            "rsi-harness.run-id=run-real",
            "rsi-harness.task-id=minimal-gpu",
            "rsi-harness.role=rootfs-snapshot",
        ]
        assert client.images.list(filters={"label": snapshot_labels}) == []
    finally:
        try:
            work.reload()
            if work.attrs["State"].get("Paused"):
                work.unpause()
        finally:
            work.remove(force=True, v=False)
            volume_backend.remove(workdir_volume)
