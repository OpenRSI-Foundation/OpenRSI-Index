from __future__ import annotations

import copy
import re
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import docker as docker_sdk
import pytest
from docker.errors import DockerException, ImageNotFound

from rsi_harness.errors import InfrastructureError, SetupError, UnsupportedTaskError
from rsi_harness.models import (
    ContainerRef,
    ContainerSpec,
    EvaluationRequest,
    GPUAllocation,
    GPUDevice,
    GPURequirement,
    ImagePlan,
    JudgeGPUMode,
    MainServiceConfig,
    RootfsSnapshotMode,
    RunGPUPlan,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.images import DockerImageBuilder
from rsi_harness.runtime.judge import DockerJudgeRuntimeFactory, JudgeRunner
from rsi_harness.runtime.network import (
    DockerIptablesFirewallBackend,
    NetworkPolicyEnforcer,
)
from rsi_harness.runtime.rootfs_snapshot import DockerRootfsSnapshotBackend
from rsi_harness.task.compose import ComposeMainServiceParser
from rsi_harness.task.digest import hash_tree
from tests.factories import make_run_plan, write_harbor_task
from tests.fakes import FakeDockerClient, FakeDockerImage

_SPARSE_SIZE = 8 * 1024**3
_FAKE_UUIDS = (
    "GPU-task6-fake-a",
    "GPU-task6-fake-b",
    "GPU-task6-fake-c",
    "GPU-task6-fake-d",
)


def _tree_identity(root: Path) -> dict[Path, tuple[int, int]]:
    return {
        path.relative_to(root): (path.lstat().st_ino, path.lstat().st_mtime_ns)
        for path in (root, *root.rglob("*"))
    }


class _LifecycleObserver:
    def __init__(self, client: Any) -> None:
        self.client = client
        self.events: list[tuple[str, dict[str, object]]] = []
        self.judges: dict[str, dict[str, object]] = {}
        self.specs: dict[str, ContainerSpec] = {}

    def _record(self, name: str, **values: object) -> None:
        self.events.append((name, values))

    def resource_event(self, name: str, **values: object) -> None:
        if name == "snapshot_acquired":
            image_id = str(values["snapshot_image_id"])
            values["image_present"] = self.client.images.get(image_id).id == image_id
        elif name == "snapshot_released":
            image_id = next(
                str(event_values["snapshot_image_id"])
                for event_name, event_values in reversed(self.events)
                if event_name == "snapshot_acquired"
                and event_values["snapshot_lease_id"]
                == values["snapshot_lease_id"]
            )
            try:
                self.client.images.get(image_id)
            except ImageNotFound:
                values["image_absent"] = True
            else:
                values["image_absent"] = False
        self._record(name, **values)

    def judge_network_planned(self, round_id: str, planned_name: str) -> None:
        self._record(
            "judge_network_planned", round_id=round_id, planned_name=planned_name
        )

    def judge_network_created(self, round_id: str, network: object) -> None:
        self._record(
            "judge_network_created",
            round_id=round_id,
            network_id=getattr(network, "network_id"),
        )

    def judge_container_planned(self, round_id: str, planned_name: str) -> None:
        self._record(
            "judge_container_planned", round_id=round_id, planned_name=planned_name
        )

    def judge_container_created(self, round_id: str, container: ContainerRef) -> None:
        inspected = self.client.containers.get(container.container_id)
        inspected.reload()
        attrs = inspected.attrs
        self.judges[round_id] = {
            "container_id": container.container_id,
            "image_id": attrs["Image"],
            "config": copy.deepcopy(attrs["Config"]),
            "host_config": copy.deepcopy(attrs["HostConfig"]),
            "mounts": copy.deepcopy(attrs.get("Mounts", [])),
        }
        self._record(
            "judge_container_created",
            round_id=round_id,
            container_id=container.container_id,
        )

    def judge_policy_planned(self, round_id: str, rule_id: str) -> None:
        self._record("judge_policy_planned", round_id=round_id, rule_id=rule_id)

    def judge_policy_installed(self, round_id: str, lease: object) -> None:
        self._record(
            "judge_policy_installed",
            round_id=round_id,
            rule_id=getattr(lease, "rule_id"),
        )

    def judge_container_removed(self, round_id: str, container_id: str) -> None:
        self._record(
            "judge_container_removed",
            round_id=round_id,
            container_id=container_id,
        )

    def judge_policy_removed(self, round_id: str, rule_id: str) -> None:
        self._record("judge_policy_removed", round_id=round_id, rule_id=rule_id)

    def judge_network_removed(self, round_id: str, network_id: str) -> None:
        self._record(
            "judge_network_removed", round_id=round_id, network_id=network_id
        )


class _NetworkNoneJudgeRound:
    """Exercise real Judge I/O when this host cannot own firewall policy."""

    def __init__(
        self,
        *,
        client: Any,
        runtime: DockerContainerRuntime,
        observer: _LifecycleObserver,
        round_id: str,
    ) -> None:
        self._client = client
        self._runtime = runtime
        self._observer = observer
        self._round_id = round_id
        self._container: ContainerRef | None = None
        self._safe_to_release_isolation = True

    def create(self, spec: ContainerSpec) -> ContainerRef:
        planned = self._runtime.planned_container_name(self._round_id)
        self._observer.judge_container_planned(self._round_id, planned)
        container = self._runtime.create(spec, planned_name=planned)
        self._container = container
        self._safe_to_release_isolation = False
        self._observer.judge_container_created(self._round_id, container)
        for mount in spec.volume_mounts:
            self._runtime.attest_workdir_volume_mount(container, mount)
        return container

    def start(self, container: ContainerRef) -> None:
        self._client.containers.get(container.container_id).start()

    def inject_tests(self, container: ContainerRef, source: Path) -> None:
        self._runtime.inject_directory(container, source, PurePosixPath("/tests"))

    def exec(
        self,
        container: ContainerRef,
        command: tuple[str, ...],
        *,
        timeout_seconds: float | None = None,
        environment: dict[str, str] | None = None,
        output_path: Path | None = None,
    ):
        return self._runtime.exec(
            container,
            command,
            timeout_seconds=timeout_seconds,
            environment=environment,
            output_path=output_path,
            redact_output=False,
        )

    def remove(self, container: ContainerRef) -> None:
        self._runtime.remove(container)
        self._safe_to_release_isolation = True
        self._observer.judge_container_removed(
            self._round_id, container.container_id
        )

    def contain_after_remove_failure(self, container: ContainerRef) -> None:
        self._runtime.stop(container)
        if not self._runtime.is_stopped_or_gone(container):
            raise InfrastructureError("network=none Judge containment is unproven")
        self._safe_to_release_isolation = False

    @property
    def safe_to_release_isolation(self) -> bool:
        return self._safe_to_release_isolation

    def recovery_context(self, container: ContainerRef | None = None) -> str:
        resolved = container or self._container
        container_id = "unknown" if resolved is None else resolved.container_id
        return (
            f"judge_container_id={container_id},"
            "network_id=none,policy_rule_id=unavailable"
        )

    def close(self) -> None:
        if not self._safe_to_release_isolation:
            raise InfrastructureError(
                "recovery_required: network=none Judge containment is unproven"
            )


class _NetworkNoneJudgeFactory:
    def __init__(
        self,
        *,
        client: Any,
        run_id: str,
        task_id: str,
        task_source_dir: Path,
        data_root: Path,
        observer: _LifecycleObserver,
        work_container: ContainerRef | None = None,
    ) -> None:
        self._client = client
        self._run_id = run_id
        self._task_id = task_id
        self._task_source_dir = task_source_dir
        self._data_root = data_root
        self._observer = observer
        self._work_container = work_container

    def __call__(self, _plan: object, round_id: str) -> _NetworkNoneJudgeRound:
        runtime = DockerContainerRuntime(
            self._client,
            run_id=self._run_id,
            task_id=self._task_id,
            role="judge",
            task_source_dir=self._task_source_dir,
            allowed_mount_roots=(self._data_root,),
            omit_gpu_device_requests_for_tests=True,
            workdir_volume_references=(
                ()
                if self._work_container is None
                else (self._work_container,)
            ),
        )
        return _NetworkNoneJudgeRound(
            client=self._client,
            runtime=runtime,
            observer=self._observer,
            round_id=round_id,
        )


class _SpecRecordingJudgeRound:
    def __init__(
        self,
        delegate: Any,
        *,
        observer: _LifecycleObserver,
        round_id: str,
    ) -> None:
        self._delegate = delegate
        self._observer = observer
        self._round_id = round_id

    def create(self, spec: ContainerSpec) -> ContainerRef:
        self._observer.specs[self._round_id] = spec
        return self._delegate.create(spec)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _SpecRecordingJudgeFactory:
    def __init__(self, delegate: Any, *, observer: _LifecycleObserver) -> None:
        self._delegate = delegate
        self._observer = observer

    def __call__(self, plan: object, round_id: str) -> _SpecRecordingJudgeRound:
        return _SpecRecordingJudgeRound(
            self._delegate(plan, round_id),
            observer=self._observer,
            round_id=round_id,
        )


def _assert_judge_spec(
    captured: dict[str, object],
    *,
    expected_log_dir: Path,
    expected_image_id: str,
    firewall_available: bool,
) -> None:
    config = captured["config"]
    host_config = captured["host_config"]
    mounts = captured["mounts"]
    assert isinstance(config, dict)
    assert isinstance(host_config, dict)
    assert isinstance(mounts, list)
    assert captured["image_id"] == expected_image_id
    assert config["WorkingDir"] == "/"
    assert config["User"] == "0:0"
    assert "SERVICE_LITERAL=service-exec-only" not in config["Env"]
    assert all(
        not str(value).startswith("RSI_HARNESS_EXPECTED_GPU_UUIDS=")
        for value in config["Env"]
    )
    assert host_config["Privileged"] is False
    assert host_config["NanoCpus"] == 1_000_000_000
    assert host_config["Memory"] == 128 * 1024**2
    assert host_config["ShmSize"] == 64 * 1024**2
    assert host_config.get("DeviceRequests") in (None, [])
    assert host_config["Tmpfs"] == {
        "/tests": "rw,exec,nosuid,nodev,mode=0755"
    }
    if not firewall_available:
        assert host_config["NetworkMode"] == "none"
    assert [
        (mount["Type"], mount["Source"], mount["Destination"], mount["RW"])
        for mount in mounts
    ] == [("bind", str(expected_log_dir), "/logs/verifier", True)]


def _remove_container(container: Any | None) -> None:
    if container is None:
        return
    try:
        container.reload()
        if container.attrs.get("State", {}).get("Paused"):
            container.unpause()
    except DockerException:
        pass
    try:
        container.remove(force=True, v=True)
    except DockerException:
        pass


@pytest.mark.integration
def test_real_two_round_judge_gpu_inherits_complete_work_rootfs_and_discards_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Using a base image instead of paused Work would lose Agent system paths."""
    try:
        client = docker_sdk.from_env()
        client.ping()
        ubuntu = client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local cached Ubuntu Docker authority unavailable: {error}")

    run_id = f"task6-real-{uuid4().hex}"
    task_id = "task6-shared-root-proof"
    fixture_ref = f"rsi-harness-task6-fixture:{uuid4().hex}"
    task_dir = write_harbor_task(
        tmp_path / "source",
        dockerfile="FROM ubuntu:24.04\nWORKDIR /\n",
        test_script=(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "test \"$(cat /testbed/answer.txt)\" = \"$ROUND_EXPECTED\"\n"
            "test \"$(stat -c %a /testbed/answer.txt)\" = 640\n"
            "test \"$(cat /etc/rsi-agent.conf)\" = agent-config\n"
            "test \"$(stat -c %a /etc/rsi-agent.conf)\" = 600\n"
            "test \"$(/usr/local/bin/rsi-agent-tool)\" = agent-tool\n"
            "test \"$(stat -c %a /usr/local/bin/rsi-agent-tool)\" = 755\n"
            "test \"$(cat /tmp/rsi-agent-state)\" = agent-tmp\n"
            "test \"$(stat -c %s /opt/checkpoints/sparse.bin)\" = 8589934592\n"
            "test \"$SERVICE_LITERAL\" = service-exec-only\n"
            "test \"$RSI_HARNESS_EXPECTED_GPU_UUIDS\" = "
            "GPU-task6-fake-c,GPU-task6-fake-d\n"
            "test ! -e /etc/rsi-judge-only\n"
            "test ! -e \"/proc/$WORK_SLEEP_PID\"\n"
            "for process in /proc/[0-9]*/cmdline; do\n"
            "  command=$(tr '\\0' ' ' < \"$process\" 2>/dev/null || true)\n"
            "  test \"$command\" != \"sleep 424242 \"\n"
            "done\n"
            "printf 'judge-only\\n' > /etc/rsi-judge-only\n"
            "printf 'judge-mutated\\n' > /testbed/answer.txt\n"
            "printf '{\"reward\": 1, \"shared_root\": 1}\\n' > "
            "/logs/verifier/reward.json\n"
        ),
    )
    test_script = task_dir / "tests" / "test.sh"
    test_script.chmod(0o755)
    source_digest = hash_tree(task_dir)
    source_identity = _tree_identity(task_dir)
    data_root = tmp_path / "engine-data"
    data_root.mkdir()

    base = None
    work = None
    retained = None
    snapshot = DockerRootfsSnapshotBackend(client)
    forbidden_transfer_calls: list[str] = []
    bounded_puts: list[tuple[str, str]] = []

    def forbid_rootfs_transfer(*_args: object, **_kwargs: object) -> None:
        forbidden_transfer_calls.append("forbidden")
        raise AssertionError("rootfs snapshots must remain local Docker commits")

    for method in ("export", "get_archive", "import_image"):
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
        base, _logs = client.images.build(
            path=str(task_dir / "environment"),
            dockerfile="Dockerfile",
            tag=fixture_ref,
            labels={
                "rsi-harness.run-id": run_id,
                "rsi-harness.task-id": task_id,
                "rsi-harness.role": "task-fixture",
            },
            rm=True,
        )
        base.reload()
        base_id = base.attrs["Id"]
        assert base.attrs["Config"]["WorkingDir"] == "/"
        assert base.attrs["Config"].get("User", "") == ""
        assert not base.attrs["Config"].get("Volumes")
        ubuntu_id = ubuntu.attrs["Id"]

        allocation = GPUAllocation(
            devices=tuple(
                GPUDevice(index=index, uuid=uuid, name="test-only fake GPU")
                for index, uuid in enumerate(_FAKE_UUIDS)
            )
        )
        work_allocation = GPUAllocation(devices=allocation.devices[:2])
        judge_allocation = GPUAllocation(devices=allocation.devices[2:])
        work_runtime = DockerContainerRuntime(
            client,
            run_id=run_id,
            task_id=task_id,
            role="work",
            task_source_dir=task_dir,
            allowed_mount_roots=(data_root,),
            omit_gpu_device_requests_for_tests=True,
        )
        work_ref = work_runtime.create(
            ContainerSpec(
                image=base_id,
                command=(
                    "/bin/bash",
                    "-lc",
                    "sleep 424242 & echo $! > /run/rsi-work-sleep.pid; "
                    "exec sleep infinity",
                ),
                workdir=PurePosixPath("/"),
                user="0:0",
                gpu_allocation=work_allocation,
            )
        )
        work = client.containers.get(work_ref.container_id)
        work.start()
        mutated = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "set -euo pipefail; "
                "install -d -m 0755 /testbed /usr/local/bin /opt/checkpoints; "
                "printf 'round-1\\n' > /testbed/answer.txt; "
                "chmod 0640 /testbed/answer.txt; "
                "printf 'agent-config\\n' > /etc/rsi-agent.conf; "
                "chmod 0600 /etc/rsi-agent.conf; "
                "printf '#!/bin/sh\\nprintf agent-tool\\n' "
                "> /usr/local/bin/rsi-agent-tool; "
                "chmod 0755 /usr/local/bin/rsi-agent-tool; "
                "printf 'agent-tmp\\n' > /tmp/rsi-agent-state; "
                f"truncate -s {_SPARSE_SIZE} /opt/checkpoints/sparse.bin",
            ]
        )
        assert mutated.exit_code == 0, mutated.output.decode(errors="replace")
        sparse_source = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "stat -c '%s %b' /opt/checkpoints/sparse.bin",
            ]
        )
        assert sparse_source.exit_code == 0
        source_size, source_blocks = sparse_source.output.decode().split()
        assert int(source_size) == _SPARSE_SIZE
        assert int(source_blocks) <= 16
        work_sleep_pid = work.exec_run(
            ["/bin/bash", "-lc", "cat /run/rsi-work-sleep.pid"]
        )
        assert work_sleep_pid.exit_code == 0
        sleep_pid = work_sleep_pid.output.decode().strip()
        assert re.fullmatch(r"[1-9][0-9]*", sleep_pid)
        work.reload()
        work_container_id = work.id
        work_image_id = work.attrs["Image"]
        assert work_image_id == base_id
        assert work.attrs.get("Mounts", []) == []

        plan = make_run_plan(data_root).model_copy(
            update={
                "workdir": PurePosixPath("/"),
                "rootfs_snapshot_mode": RootfsSnapshotMode.FULL_ROOTFS,
                "gpu_plan": RunGPUPlan(
                    authorized_pool=allocation,
                    work=work_allocation,
                    judge=judge_allocation,
                    judge_mode=JudgeGPUMode.DISJOINT,
                ),
                "images": ImagePlan(
                    base_ref=base_id,
                    work_ref=base_id,
                    judge_ref=base_id,
                    workdir=PurePosixPath("/"),
                    rootfs_snapshot_mode=RootfsSnapshotMode.FULL_ROOTFS,
                    base_digest=base_id,
                    work_digest=base_id,
                    judge_digest=base_id,
                ),
            }
        )
        plan = plan.model_copy(
            update={
                "task": plan.task.model_copy(
                    update={
                        "task_id": task_id,
                        "source_dir": task_dir,
                        "source_digest": source_digest,
                        "tests_digest": hash_tree(task_dir / "tests"),
                        "workdir": PurePosixPath("/"),
                        "gpu_requirement": GPURequirement(count=2),
                        "service": MainServiceConfig(
                            workdir=PurePosixPath("/"),
                            user="0:0",
                            environment=(
                                ("SERVICE_LITERAL", "service-exec-only"),
                            ),
                            shm_size="64m",
                            cpus=1,
                            memory_mb=128,
                            storage_mb=256,
                        ),
                        "verifier": plan.task.verifier.model_copy(
                            update={
                                "user": None,
                                "environment": (
                                    ("ROUND_EXPECTED", "round-1"),
                                    ("SERVICE_LITERAL", "service-exec-only"),
                                    ("WORK_SLEEP_PID", sleep_pid),
                                ),
                            }
                        ),
                    }
                )
            }
        )
        assert plan.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS
        observer = _LifecycleObserver(client)
        firewall = DockerIptablesFirewallBackend(client)
        firewall_available = firewall.probe()
        if firewall_available:
            enforcer = NetworkPolicyEnforcer(run_id=run_id, firewall=firewall)
            judge_factory = DockerJudgeRuntimeFactory(
                client,
                run_id=run_id,
                task_id=task_id,
                task_source_dir=task_dir,
                allowed_mount_roots=(data_root,),
                network_policy_enforcer=enforcer,
                lifecycle_observer=observer,
                omit_gpu_device_requests_for_tests=True,
            )
            print("task6 firewall policy: production INPUT/DOCKER-USER path enabled")
        else:
            judge_factory = _NetworkNoneJudgeFactory(
                client=client,
                run_id=run_id,
                task_id=task_id,
                task_source_dir=task_dir,
                data_root=data_root,
                observer=observer,
            )
            print(
                "task6 firewall policy: unavailable; rootfs/Judge proof uses "
                "network_mode=none"
            )
        artifacts = RunArtifactWriter(plan, run_id=run_id)
        artifacts.start()
        runner = JudgeRunner(
            run_id=run_id,
            work_runtime=work_runtime,
            judge_runtime_factory=_SpecRecordingJudgeFactory(
                judge_factory, observer=observer
            ),
            snapshot_backend=snapshot,
            artifact_writer=artifacts,
            quiescence_checker=lambda _allocation, _container: None,
            lifecycle_observer=observer,
        )

        first = runner.evaluate(
            EvaluationRequest(
                run_plan=plan,
                work_container=work_ref,
                round_id="agent-1",
                verifier_logs=plan.paths.logs / "verifier" / "agent-1",
                verifier_output=artifacts.feedback_root / "agent-1.log",
            )
        )
        assert first.status == SubmissionStatus.COMPLETED, (first.error, first.output)
        assert first.rewards == {"reward": 1, "shared_root": 1}
        assert first.score == 1
        work.reload()
        assert work.id == work_container_id
        assert work.attrs["Image"] == work_image_id
        assert work.attrs["State"]["Paused"] is False
        first_work_state = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "test \"$(cat /testbed/answer.txt)\" = round-1 && "
                "test ! -e /etc/rsi-judge-only",
            ]
        )
        assert first_work_state.exit_code == 0

        round_two_mutation = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "printf 'round-2\\n' > /testbed/answer.txt && "
                "chmod 0640 /testbed/answer.txt",
            ]
        )
        assert round_two_mutation.exit_code == 0
        second_plan = plan.model_copy(
            update={
                "task": plan.task.model_copy(
                    update={
                        "verifier": plan.task.verifier.model_copy(
                            update={
                                "environment": (
                                    ("ROUND_EXPECTED", "round-2"),
                                    ("SERVICE_LITERAL", "service-exec-only"),
                                    ("WORK_SLEEP_PID", sleep_pid),
                                )
                            }
                        )
                    }
                )
            }
        )
        second = runner.evaluate(
            EvaluationRequest(
                run_plan=second_plan,
                work_container=work_ref,
                round_id="agent-2",
                verifier_logs=plan.paths.logs / "verifier" / "agent-2",
                verifier_output=artifacts.feedback_root / "agent-2.log",
            )
        )
        assert second.status == SubmissionStatus.COMPLETED, second.error
        assert second.rewards == {"reward": 1, "shared_root": 1}
        work.reload()
        assert work.id == work_container_id
        assert work.attrs["Image"] == work_image_id
        assert work.attrs["State"]["Paused"] is False
        second_work_state = work.exec_run(
            [
                "/bin/bash",
                "-lc",
                "test \"$(cat /testbed/answer.txt)\" = round-2 && "
                "test ! -e /etc/rsi-judge-only && "
                "test \"$(stat -c %s /opt/checkpoints/sparse.bin)\" = "
                "8589934592 && "
                "test \"$(stat -c %b /opt/checkpoints/sparse.bin)\" -le 16",
            ]
        )
        assert second_work_state.exit_code == 0

        acquired = [
            values
            for name, values in observer.events
            if name == "snapshot_acquired"
        ]
        released = [
            values
            for name, values in observer.events
            if name == "snapshot_released"
        ]
        assert len(acquired) == len(released) == 2
        assert all(values["image_present"] is True for values in acquired)
        assert all(values["image_absent"] is True for values in released)
        assert acquired[0]["snapshot_image_id"] != acquired[1]["snapshot_image_id"]
        for round_id, acquired_values in zip(
            ("agent-1", "agent-2"), acquired, strict=True
        ):
            spec = observer.specs[round_id]
            assert spec.image == acquired_values["snapshot_image_id"]
            assert spec.workdir == PurePosixPath("/")
            assert spec.user == "0:0"
            assert spec.environment == ()
            assert spec.shm_size == "64m"
            assert spec.cpus == 1
            assert spec.memory_mb == 128
            assert spec.storage_mb == 256
            assert spec.gpu_allocation.uuids == (
                "GPU-task6-fake-c",
                "GPU-task6-fake-d",
            )
            _assert_judge_spec(
                observer.judges[round_id],
                expected_log_dir=(
                    plan.paths.logs / "verifier" / round_id
                ).resolve(),
                expected_image_id=str(acquired_values["snapshot_image_id"]),
                firewall_available=firewall_available,
            )
            removed_index = next(
                index
                for index, (name, values) in enumerate(observer.events)
                if name == "judge_container_removed"
                and values["round_id"] == round_id
            )
            released_index = next(
                index
                for index, (name, values) in enumerate(observer.events)
                if name == "snapshot_released"
                and values["snapshot_lease_id"]
                == acquired_values["snapshot_lease_id"]
            )
            assert removed_index < released_index

        assert all(spec.volume_mounts == () for spec in observer.specs.values())
        assert len(bounded_puts) == 2
        assert all(
            re.fullmatch(r"/tmp/\.rsi-harness-tests-[0-9a-f]{32}", path)
            for _container, path in bounded_puts
        )
        assert forbidden_transfer_calls == []
        assert hash_tree(task_dir) == source_digest
        assert _tree_identity(task_dir) == source_identity
        assert client.images.get(fixture_ref).attrs["Id"] == base_id
        assert client.images.get("ubuntu:24.04").attrs["Id"] == ubuntu_id

        work_runtime.pause(work_ref)
        retained_ref = snapshot.planned_ref(
            run_id=run_id,
            task_id=task_id,
            round_id="final",
            purpose="retained-work",
        )
        retained = snapshot.acquire(
            work_ref,
            run_id=run_id,
            task_id=task_id,
            round_id="final",
            purpose="retained-work",
            planned_ref=retained_ref,
        )
        assert client.images.get(retained.image_id).id == retained.image_id
        work.remove(force=True, v=True)
        work = None
        assert client.images.get(retained.image_id).id == retained.image_id
        snapshot.release(retained)
        retained = None
        client.images.remove(fixture_ref, force=True)
        base = None

        exact_labels = [
            f"rsi-harness.run-id={run_id}",
            f"rsi-harness.task-id={task_id}",
        ]
        assert client.containers.list(
            all=True, filters={"label": exact_labels}
        ) == []
        assert client.networks.list(filters={"label": exact_labels}) == []
        assert client.images.list(
            filters={
                "label": [
                    *exact_labels,
                    "rsi-harness.role=rootfs-snapshot",
                ]
            }
        ) == []
        assert client.images.list(
            filters={
                "label": [
                    *exact_labels,
                    "rsi-harness.role=retained-work-rootfs",
                ]
            }
        ) == []
    finally:
        _remove_container(work)
        if retained is not None:
            try:
                snapshot.release(retained)
            except (DockerException, InfrastructureError):
                try:
                    client.images.remove(retained.image_id, force=True)
                except DockerException:
                    pass
        try:
            client.images.remove(fixture_ref, force=True)
        except DockerException:
            pass
        for image in client.images.list(
            filters={"label": f"rsi-harness.run-id={run_id}"}
        ):
            if image.attrs.get("Config", {}).get("Labels", {}).get(
                "rsi-harness.role"
            ) in {"rootfs-snapshot", "retained-work-rootfs", "task-fixture"}:
                try:
                    client.images.remove(image.id, force=True)
                except DockerException:
                    pass


def test_image_volumes_are_rejected_before_work_or_snapshot_mutation(
    tmp_path: Path,
) -> None:
    """Accepting Config.Volumes would silently omit task state from commit."""
    client = FakeDockerClient()
    plan = make_run_plan(tmp_path)
    plan.task.source_dir.mkdir(parents=True)
    service = plan.task.service.model_copy(update={"image": "task-volume:latest"})
    task = plan.task.model_copy(update={"service": service})
    client.images.by_ref["task-volume:latest"] = FakeDockerImage(
        "sha256:base-id",
        {
            "Id": "sha256:base-id",
            "RepoDigests": ["task-volume@sha256:deadbeef"],
            "Config": {
                "WorkingDir": "/",
                "User": "",
                "Volumes": {"/data": {}},
            },
        },
    )

    with pytest.raises(SetupError, match="declares volumes.*rootfs snapshot"):
        DockerImageBuilder(client).prepare(task, agent=object())  # type: ignore[arg-type]

    assert client.images.built == []
    assert client.containers.created == []
    assert client.containers.runs == []


def test_compose_volumes_are_rejected_before_task_state_is_mutated(
    tmp_path: Path,
) -> None:
    """Accepting Compose volumes would promise state Docker commit cannot copy."""
    environment = tmp_path / "environment"
    environment.mkdir()
    compose = environment / "docker-compose.yaml"
    compose.write_text(
        "services:\n"
        "  main:\n"
        "    image: ubuntu:24.04\n"
        "    volumes: ['./host:/testbed']\n"
    )
    before = (hash_tree(environment), _tree_identity(environment))

    with pytest.raises(UnsupportedTaskError, match="host volume"):
        ComposeMainServiceParser().parse(compose)

    assert (hash_tree(environment), _tree_identity(environment)) == before


def test_process_state_inheritance_request_is_rejected_before_snapshot_mutation(
    tmp_path: Path,
) -> None:
    """A rootfs commit must never claim to clone a live Work process table."""
    client = FakeDockerClient()
    client.containers.by_id["work-process-proof"] = client.containers.create(
        "ubuntu:24.04", ["sleep", "424242"]
    )
    backend = DockerRootfsSnapshotBackend(client)

    with pytest.raises(TypeError, match="include_processes"):
        backend.acquire(
            ContainerRef(container_id="work-process-proof", role="work"),
            run_id="negative",
            task_id="process-state",
            round_id="one",
            include_processes=True,  # type: ignore[call-arg]
        )

    assert client.containers.created == [
        {
            "image": "ubuntu:24.04",
            "command": ["sleep", "424242"],
        }
    ]
    assert client.images.built == []


@pytest.mark.parametrize("operation", ("export", "upload"))
def test_rootfs_transfer_api_requests_fail_before_snapshot_mutation(
    operation: str,
) -> None:
    """Adding a host/registry transfer API would violate local-commit authority."""
    client = FakeDockerClient()
    backend = DockerRootfsSnapshotBackend(client)

    with pytest.raises(AttributeError):
        getattr(backend, operation)(
            ContainerRef(container_id="work-transfer-proof", role="work")
        )

    assert client.containers.created == []
    assert client.containers.runs == []
    assert client.images.built == []
