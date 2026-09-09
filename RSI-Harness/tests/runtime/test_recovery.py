from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

from rsi_harness.errors import InfrastructureError
from rsi_harness.models import (
    GPUAllocation,
    GPUDevice,
    JudgeGPUMode,
    ManagedWorkdirVolume,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunStatus,
)
from rsi_harness.runtime.production import ProductionRecoveryBackend
from rsi_harness.runtime.recovery import (
    JudgeResourceLease,
    LeaseStore,
    RecoveryManager,
    ResourceLease,
    RetainedImageRollbackAuthority,
    SnapshotRecoveryAuthority,
    WorkdirVolumeResourceLease,
    WorkResourceLease,
)
from rsi_harness.runtime.workdir_volume import managed_workdir_volume_labels
from tests.fakes import (
    FakeDockerClient,
    FakeDockerContainer,
    FakeDockerImage,
    FakeFirewallBackend,
)

CLEANUP_IMAGE = f"base@sha256:{'c' * 64}"
RETAINED_IMAGE_ID = f"sha256:{'b' * 64}"
RETAINED_IMAGE_REF = f"rsi-harness-rootfs:retained-work-{'a' * 64}"
JUDGE_IMAGE_ID = f"sha256:{'d' * 64}"
JUDGE_IMAGE_REF = f"rsi-harness-rootfs:judge-round-{'e' * 64}"
SECOND_JUDGE_IMAGE_ID = f"sha256:{'f' * 64}"
SECOND_RETAINED_IMAGE_ID = f"sha256:{'c' * 64}"
SECOND_RETAINED_IMAGE_REF = f"rsi-harness-rootfs:retained-work-{'f' * 64}"
WORKDIR_VOLUME_NAME = f"rsi-harness-workdir-{'1' * 64}"


def workdir_volume(
    *,
    name: str = WORKDIR_VOLUME_NAME,
    run_id: str = "run-1",
    task_id: str = "task-1",
    target: str = "/workspace",
    freshness_nonce: str = "f" * 64,
) -> ManagedWorkdirVolume:
    return ManagedWorkdirVolume(
        name=name,
        run_id=run_id,
        task_id=task_id,
        target=target,
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce=freshness_nonce,
    )


def image_state(
    image_id: str,
    image_ref: str,
    *,
    role: str,
    round_id: str,
    source_container_id: str = "work-real",
    in_use: bool = False,
) -> dict[str, object]:
    return {
        "id": image_id,
        "repo_tags": (image_ref,),
        "labels": {
            "rsi-harness.run-id": "run-1",
            "rsi-harness.task-id": "task-1",
            "rsi-harness.round-id": round_id,
            "rsi-harness.source-container-id": source_container_id,
            "rsi-harness.role": role,
        },
        "in_use": in_use,
    }


def volume_state(
    volume: ManagedWorkdirVolume,
    *,
    driver: str = "local",
    scope: str = "local",
    labels: dict[str, str] | None = None,
    references: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "name": volume.name,
        "driver": driver,
        "labels": labels
        if labels is not None
        else managed_workdir_volume_labels(volume),
        "options": {},
        "scope": scope,
        "container_references": references,
    }


class RecoveryBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.containers = {
            "judge-real": {
                "labels": {
                    "rsi-harness.run-id": "run-1",
                    "rsi-harness.task-id": "task-1",
                    "rsi-harness.role": "judge",
                },
                "running": True,
                "paused": False,
            },
            "work-real": {
                "labels": {
                    "rsi-harness.run-id": "run-1",
                    "rsi-harness.task-id": "task-1",
                    "rsi-harness.role": "work",
                },
                "running": True,
                "paused": True,
            },
        }
        self.mounts = {"/managed/run-1/snap/merged"}
        self.snapshots = {
            "snapshot-real": SnapshotRecoveryAuthority(
                run_id="run-1",
                lease_id="snapshot-real",
                round_id="agent-1",
                merged_path=Path("/managed/run-1/snap/merged"),
                process_id=321,
                manifest_requires_recovery=True,
                layers_present=True,
                process_alive=True,
            )
        }
        self.networks = {
            "network-work-real": {
                "labels": {
                    "rsi-harness.run-id": "run-1",
                    "rsi-harness.task-id": "task-1",
                    "rsi-harness.role": "work",
                }
            },
            "network-judge-real": {
                "labels": {
                    "rsi-harness.run-id": "run-1",
                    "rsi-harness.task-id": "task-1",
                    "rsi-harness.role": "judge",
                }
            },
        }
        self.policies = {"policy-judge-real", "policy-work-real"}
        self.images: dict[str, dict[str, object]] = {}
        self.image_queries: list[dict[str, str]] = []
        self.volumes: dict[str, dict[str, object]] = {}
        self.volume_queries: list[dict[str, str]] = []
        self.volume_query_result: object | None = None
        self.volume_query_error: Exception | None = None
        self.volume_inspect_error: Exception | None = None
        self.volume_reference_error: Exception | None = None
        self.volume_remove_error: Exception | None = None

    def list_containers(self, *, labels):
        self.events.append(("list_containers", labels["rsi-harness.role"]))
        return tuple(self.containers.items())

    def list_networks(self, *, labels):
        self.events.append(("list_networks", labels["rsi-harness.role"]))
        return tuple(self.networks.items())

    def inspect_container(self, container_id: str):
        return self.containers.get(container_id)

    def list_images(self, *, labels):
        self.events.append(("list_images", labels["rsi-harness.role"]))
        self.image_queries.append(dict(labels))
        return tuple(
            (image_id, state)
            for image_id, state in self.images.items()
            if all(
                state["labels"].get(key) == value  # type: ignore[union-attr]
                for key, value in labels.items()
            )
        )

    def inspect_image(self, image_id: str):
        self.events.append(("inspect_image", image_id))
        direct = self.images.get(image_id)
        if direct is not None:
            return direct
        return next(
            (state for state in self.images.values() if image_id in state["repo_tags"]),
            None,
        )

    def image_in_use(self, image_id: str) -> bool:
        self.events.append(("inspect_image_use", image_id))
        state = self.images.get(image_id)
        return bool(state and state.get("in_use", False))

    def remove_image(self, image_id: str) -> None:
        self.events.append(("remove_image", image_id))
        self.images.pop(image_id, None)

    def list_volumes(self, *, labels):
        self.events.append(("list_volumes", labels["rsi-harness.role"]))
        self.volume_queries.append(dict(labels))
        if self.volume_query_error is not None:
            raise self.volume_query_error
        if self.volume_query_result is not None:
            return self.volume_query_result
        return tuple(
            (name, state)
            for name, state in self.volumes.items()
            if all(
                state["labels"].get(key) == value  # type: ignore[union-attr]
                for key, value in labels.items()
            )
        )

    def inspect_volume(self, name: str):
        self.events.append(("inspect_volume", name))
        if self.volume_inspect_error is not None:
            raise self.volume_inspect_error
        return self.volumes.get(name)

    def volume_in_use(self, name: str) -> bool:
        self.events.append(("inspect_volume_use", name))
        if self.volume_reference_error is not None:
            raise self.volume_reference_error
        state = self.volumes.get(name)
        return bool(state and state.get("container_references"))

    def remove_volume(self, name: str) -> None:
        self.events.append(("remove_volume", name))
        if self.volume_remove_error is not None:
            raise self.volume_remove_error
        self.volumes.pop(name, None)

    def stop_container(self, container_id: str) -> None:
        self.events.append(("stop", container_id))
        self.containers[container_id]["running"] = False

    def remove_container(self, container_id: str) -> None:
        self.events.append(("remove", container_id))
        self.containers.pop(container_id, None)

    def is_mounted(self, path: Path) -> bool:
        self.events.append(("inspect_mount", str(path)))
        return str(path) in self.mounts

    def discover_snapshot_leases(self, *, run_id, round_id, lease_id):
        self.events.append(("discover_snapshot", run_id))
        return tuple(
            authority
            for authority in self.snapshots.values()
            if authority.run_id == run_id
            and (round_id is None or authority.round_id == round_id)
            and (lease_id is None or authority.lease_id == lease_id)
        )

    def release_snapshot(self, authority: SnapshotRecoveryAuthority) -> None:
        self.events.append(("release_snapshot", str(authority.merged_path)))
        self.mounts.discard(str(authority.merged_path))
        self.snapshots.pop(authority.lease_id, None)

    def unpause_container(self, container_id: str) -> None:
        self.events.append(("unpause", container_id))
        self.containers[container_id]["paused"] = False

    def network_in_use(self, network_id: str) -> bool:
        self.events.append(("inspect_network", network_id))
        return False

    def remove_network(self, network_id: str) -> None:
        self.events.append(("remove_network", network_id))
        self.networks.pop(network_id, None)

    def remove_policy(self, rule_id: str) -> None:
        self.events.append(("remove_policy", rule_id))
        self.policies.discard(rule_id)

    def policy_exists(self, rule_id: str) -> bool:
        self.events.append(("inspect_policy", rule_id))
        return rule_id in self.policies

    def workspace_is_mounted(self, workspace: Path) -> bool:
        self.events.append(("inspect_workspace", str(workspace)))
        return False

    def delete_workspace(
        self,
        workspace: Path,
        *,
        image_ref: str,
        run_id: str,
        task_id: str,
    ) -> None:
        self.events.append(
            ("delete_workspace", f"{workspace}|{image_ref}|{run_id}|{task_id}")
        )
        shutil.rmtree(workspace)


def lease(tmp_path: Path) -> ResourceLease:
    return ResourceLease(
        run_id="run-1",
        task_id="task-1",
        coordinator_pid=123,
        coordinator_started_at=1.5,
        phase="judging",
        status=None,
        workspace_path=tmp_path / "runs" / "run-1" / "workspace",
        cleanup_image_ref=CLEANUP_IMAGE,
        work=WorkResourceLease(
            planned_container="rsi-run-1-work",
            container_id="work-real",
            planned_network="rsi-run-1-work-network",
            network_id="network-work-stale",
            network_name="rsi-run-1-work-network",
            planned_policy_rule_id="policy-work-real",
            policy_rule_id="policy-work-real",
            paused=True,
        ),
        judge=JudgeResourceLease(
            round_id="agent-1",
            planned_container="rsi-run-1-agent-1-judge",
            container_id="judge-real",
            planned_network="rsi-run-1-agent-1-judge-network",
            network_id="network-judge-stale",
            network_name="rsi-run-1-agent-1-judge-network",
            planned_policy_rule_id="policy-judge-real",
            policy_rule_id="policy-judge-real",
            planned_snapshot="snapshot:run-1:agent-1",
            snapshot_lease_id="snapshot-real",
            snapshot_merged_path=Path("/managed/run-1/snap/merged"),
            snapshot_process_id=321,
        ),
    )


def gpu_plan() -> RunGPUPlan:
    devices = tuple(
        GPUDevice(index=index, uuid=f"GPU-{letter}", name="Test GPU")
        for index, letter in enumerate("abcd")
    )
    return RunGPUPlan(
        authorized_pool=GPUAllocation(devices=devices),
        work=GPUAllocation(devices=devices[:2]),
        judge=GPUAllocation(devices=devices[2:]),
        judge_mode=JudgeGPUMode.DISJOINT,
    )


def split_volume_lease(
    tmp_path: Path,
    *,
    actual: ManagedWorkdirVolume | None = None,
    rollback: ManagedWorkdirVolume | None = None,
    mounted: bool = False,
) -> ResourceLease:
    base = lease(tmp_path)
    expected = workdir_volume()
    return base.model_copy(
        update={
            "rootfs_snapshot_mode": RootfsSnapshotMode.SPLIT_WORKDIR,
            "work": base.work.model_copy(
                update={
                    "workdir_volume": WorkdirVolumeResourceLease(
                        planned_name=expected.name,
                        planned_target=expected.target,
                        planned_snapshot_mode=expected.snapshot_mode,
                        planned_freshness_nonce=expected.freshness_nonce,
                        actual=actual,
                        rollback=rollback,
                        mounted=mounted,
                    )
                }
            ),
        }
    )


def test_store_writes_atomic_secret_free_exact_lease_and_serializes_lock(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    expected = lease(tmp_path)

    with store.lock("run-1"):
        store.write(expected)
        with pytest.raises(BlockingIOError):
            with LeaseStore(tmp_path / "leases").lock("run-1", blocking=False):
                pass

    loaded = store.read("run-1")
    assert loaded == expected
    raw = json.loads(store.path_for("run-1").read_text())
    assert raw["work"]["planned_container"] == "rsi-run-1-work"
    assert raw["work"]["container_id"] == "work-real"
    assert raw["work"]["network_id"] == "network-work-stale"
    assert raw["work"]["policy_rule_id"] == "policy-work-real"
    assert raw["judge"]["network_id"] == "network-judge-stale"
    assert raw["judge"]["policy_rule_id"] == "policy-judge-real"
    assert raw["judge"]["planned_policy_rule_id"] == "policy-judge-real"
    assert raw["judge"]["snapshot_process_id"] == 321
    assert raw["cleanup_image_ref"] == CLEANUP_IMAGE
    assert raw["schema_version"] == 4
    serialized = json.dumps(raw).lower()
    assert "authorization" not in serialized
    assert "bearer" not in serialized
    assert "token" not in serialized
    assert not list(store.path_for("run-1").parent.glob("*.tmp"))


def test_gpu_plan_round_trips_old_and_recovered_leases_without_inventory(
    tmp_path: Path,
) -> None:
    old_payload = lease(tmp_path).model_dump(mode="json")
    old_payload.pop("gpu_plan", None)
    assert ResourceLease.model_validate(old_payload).gpu_plan is None

    expected_plan = gpu_plan()
    current = lease(tmp_path).model_copy(update={"gpu_plan": expected_plan})
    assert current.workspace_path is not None
    current.workspace_path.mkdir(parents=True)
    store = LeaseStore(tmp_path / "leases")
    store.write(current)

    class NoInventoryRecoveryBackend(RecoveryBackend):
        def list_devices(self):
            raise AssertionError("recovery must not reselect GPU devices")

    manager = RecoveryManager(
        store=store,
        backend=NoInventoryRecoveryBackend(),
        managed_root=tmp_path / "runs",
    )

    assert manager.recover("run-1") == ("run-1",)
    recovered = store.read("run-1")
    assert recovered is not None
    assert recovered.gpu_plan == expected_plan
    assert recovered.gpu_plan.authorized_pool.uuids == (
        "GPU-a",
        "GPU-b",
        "GPU-c",
        "GPU-d",
    )


def test_store_rejects_pre_image_recovery_schema_instead_of_guessing(
    tmp_path,
) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload["schema_version"] = 3

    with pytest.raises(ValidationError, match="schema version"):
        ResourceLease.model_validate(payload)


def test_schema_four_full_rootfs_lease_without_volume_fields_defaults_safely(
    tmp_path,
) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload.pop("rootfs_snapshot_mode", None)
    payload["work"].pop("workdir_volume", None)

    loaded = ResourceLease.model_validate(payload)

    assert loaded.schema_version == 4
    assert loaded.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS
    assert loaded.work.workdir_volume == WorkdirVolumeResourceLease()


def test_schema_four_split_plan_requires_and_persists_mode_target_freshness(
    tmp_path,
) -> None:
    """A split plan without freshness provenance could adopt an older volume."""
    base = lease(tmp_path).model_dump(mode="json")
    base["rootfs_snapshot_mode"] = "split-workdir"
    base["work"]["workdir_volume"] = {
        "planned_name": WORKDIR_VOLUME_NAME,
        "planned_target": "/workspace",
    }

    with pytest.raises(ValidationError, match="mode|freshness"):
        ResourceLease.model_validate(base)

    base["work"]["workdir_volume"].update(
        {
            "planned_snapshot_mode": "split-workdir",
            "planned_freshness_nonce": "f" * 64,
        }
    )
    parsed = ResourceLease.model_validate(base)
    assert parsed.work.workdir_volume.planned_snapshot_mode is (
        RootfsSnapshotMode.SPLIT_WORKDIR
    )
    assert parsed.work.workdir_volume.planned_freshness_nonce == "f" * 64


def test_store_persists_exact_split_workdir_volume_authority(tmp_path) -> None:
    expected_volume = workdir_volume()
    original = lease(tmp_path)
    expected = original.model_copy(
        update={
            "rootfs_snapshot_mode": RootfsSnapshotMode.SPLIT_WORKDIR,
            "work": original.work.model_copy(
                update={
                    "workdir_volume": WorkdirVolumeResourceLease(
                        planned_name=WORKDIR_VOLUME_NAME,
                        planned_target=PurePosixPath("/workspace"),
                        planned_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
                        planned_freshness_nonce="f" * 64,
                        actual=expected_volume,
                        mounted=True,
                    )
                }
            ),
        }
    )
    store = LeaseStore(tmp_path / "leases")

    store.write(expected)

    loaded = store.read("run-1")
    assert loaded is not None
    assert loaded.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR
    assert loaded.work.workdir_volume.actual == expected_volume
    assert loaded.work.workdir_volume.mounted is True
    raw = json.loads(store.path_for("run-1").read_text())
    assert raw["rootfs_snapshot_mode"] == "split-workdir"
    assert raw["work"]["workdir_volume"] == {
        "planned_name": WORKDIR_VOLUME_NAME,
        "planned_target": "/workspace",
        "planned_snapshot_mode": "split-workdir",
        "planned_freshness_nonce": "f" * 64,
        "actual": {
            "name": WORKDIR_VOLUME_NAME,
            "driver": "local",
            "run_id": "run-1",
            "task_id": "task-1",
            "target": "/workspace",
            "snapshot_mode": "split-workdir",
            "freshness_nonce": "f" * 64,
        },
        "rollback": None,
        "mounted": True,
    }


@pytest.mark.parametrize(
    ("mode", "volume_payload", "message"),
    (
        (
            "full-rootfs",
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "/workspace",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
            },
            "full-rootfs",
        ),
        (
            "split-workdir",
            {
                "planned_target": "/workspace",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
                "actual": workdir_volume().model_dump(mode="json"),
            },
            "planned",
        ),
        (
            "split-workdir",
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "/workspace",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
                "actual": workdir_volume(
                    name=f"rsi-harness-workdir-{'2' * 64}"
                ).model_dump(mode="json"),
            },
            "planned",
        ),
        (
            "split-workdir",
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "/workspace",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
                "mounted": True,
            },
            "actual",
        ),
        (
            "split-workdir",
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "/workspace",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
                "actual": workdir_volume(run_id="other-run").model_dump(mode="json"),
            },
            "run",
        ),
        (
            "split-workdir",
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "/workspace",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
                "actual": workdir_volume(task_id="other-task").model_dump(mode="json"),
            },
            "task",
        ),
    ),
)
def test_workdir_volume_lease_rejects_inconsistent_mode_or_identity(
    tmp_path,
    mode,
    volume_payload,
    message,
) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload["rootfs_snapshot_mode"] = mode
    payload["work"]["workdir_volume"] = volume_payload

    with pytest.raises(ValidationError, match=message):
        ResourceLease.model_validate(payload)


@pytest.mark.parametrize(
    "volume_payload",
    (
        {"planned_name": "unsafe/volume"},
        {
            "planned_name": WORKDIR_VOLUME_NAME,
            "planned_target": "/workspace",
            "planned_snapshot_mode": "split-workdir",
            "planned_freshness_nonce": "f" * 64,
            "access_token": "TOP-SECRET",
        },
        {
            "planned_name": WORKDIR_VOLUME_NAME,
            "planned_target": "/workspace",
            "planned_snapshot_mode": "split-workdir",
            "planned_freshness_nonce": "f" * 64,
            "actual": {
                **workdir_volume().model_dump(mode="json"),
                "authorization": "Bearer TOP-SECRET",
            },
        },
    ),
)
def test_workdir_volume_lease_rejects_unsafe_or_secret_bearing_fields(
    tmp_path,
    volume_payload,
) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload["rootfs_snapshot_mode"] = "split-workdir"
    payload["work"]["workdir_volume"] = volume_payload

    with pytest.raises(ValidationError):
        ResourceLease.model_validate(payload)


def test_split_workdir_volume_rejects_root_target(tmp_path) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload["rootfs_snapshot_mode"] = "split-workdir"
    payload["work"]["workdir_volume"] = {
        "planned_name": WORKDIR_VOLUME_NAME,
        "planned_target": "/workspace",
        "planned_snapshot_mode": "split-workdir",
        "planned_freshness_nonce": "f" * 64,
        "actual": {
            **workdir_volume().model_dump(mode="json"),
            "target": "/",
        },
    }

    with pytest.raises(ValidationError, match="non-root"):
        ResourceLease.model_validate(payload)


@pytest.mark.parametrize(
    ("volume_payload", "message"),
    (
        ({"planned_name": WORKDIR_VOLUME_NAME}, "target"),
        ({"planned_target": "/workspace"}, "planned name"),
        (
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "workspace",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
            },
            "absolute",
        ),
        (
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "/",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
            },
            "non-root",
        ),
        (
            {
                "planned_name": WORKDIR_VOLUME_NAME,
                "planned_target": "/planned",
                "planned_snapshot_mode": "split-workdir",
                "planned_freshness_nonce": "f" * 64,
                "actual": workdir_volume(target="/actual").model_dump(mode="json"),
            },
            "target",
        ),
    ),
)
def test_workdir_volume_lease_requires_exact_non_root_planned_target(
    tmp_path,
    volume_payload,
    message,
) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload["rootfs_snapshot_mode"] = "split-workdir"
    payload["work"]["workdir_volume"] = volume_payload

    with pytest.raises(ValidationError, match=message):
        ResourceLease.model_validate(payload)


def test_store_persists_exact_untrusted_workdir_volume_rollback_authority(
    tmp_path,
) -> None:
    returned = workdir_volume(
        run_id="wrong-run",
        task_id="wrong-task",
        target="/wrong-target",
    )
    original = lease(tmp_path)
    expected = original.model_copy(
        update={
            "rootfs_snapshot_mode": RootfsSnapshotMode.SPLIT_WORKDIR,
            "work": original.work.model_copy(
                update={
                    "workdir_volume": WorkdirVolumeResourceLease(
                        planned_name=WORKDIR_VOLUME_NAME,
                        planned_target=PurePosixPath("/workspace"),
                        planned_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
                        planned_freshness_nonce="f" * 64,
                        rollback=returned,
                    )
                }
            ),
            "recovery_required": True,
        }
    )
    store = LeaseStore(tmp_path / "leases")

    store.write(expected)

    loaded = store.read("run-1")
    assert loaded is not None
    assert loaded.work.workdir_volume.rollback == returned
    assert loaded.work.workdir_volume.actual is None
    raw = json.loads(store.path_for("run-1").read_text())
    assert raw["work"]["workdir_volume"]["rollback"] == {
        "name": WORKDIR_VOLUME_NAME,
        "driver": "local",
        "run_id": "wrong-run",
        "task_id": "wrong-task",
        "target": "/wrong-target",
        "snapshot_mode": "split-workdir",
        "freshness_nonce": "f" * 64,
    }


def test_workdir_volume_rollback_rejects_forged_secret_fields(tmp_path) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload["rootfs_snapshot_mode"] = "split-workdir"
    payload["work"]["workdir_volume"] = {
        "planned_name": WORKDIR_VOLUME_NAME,
        "planned_target": "/workspace",
        "planned_snapshot_mode": "split-workdir",
        "planned_freshness_nonce": "f" * 64,
        "rollback": {
            **workdir_volume().model_dump(mode="json"),
            "credential": "TOP-SECRET",
        },
    }

    with pytest.raises(ValidationError):
        ResourceLease.model_validate(payload)


def test_store_rejects_mixed_legacy_path_and_rootfs_image_authority(
    tmp_path,
) -> None:
    payload = lease(tmp_path).model_dump(mode="json")
    payload["judge"].update(
        {
            "planned_snapshot_ref": JUDGE_IMAGE_REF,
            "snapshot_image_id": JUDGE_IMAGE_ID,
            "snapshot_image_ref": JUDGE_IMAGE_REF,
            "snapshot_source_container_id": "work-real",
        }
    )

    with pytest.raises(ValidationError, match="path and image"):
        ResourceLease.model_validate(payload)


def test_store_persists_exact_judge_rootfs_snapshot_authority(tmp_path) -> None:
    store = LeaseStore(tmp_path / "leases")
    expected = lease(tmp_path).model_copy(
        update={
            "judge": JudgeResourceLease(
                round_id="agent-1",
                planned_snapshot_ref=JUDGE_IMAGE_REF,
                snapshot_lease_id="snapshot-real",
                snapshot_image_id=JUDGE_IMAGE_ID,
                snapshot_image_ref=JUDGE_IMAGE_REF,
                snapshot_source_container_id="work-real",
            )
        }
    )

    store.write(expected)

    loaded = store.read("run-1")
    assert loaded is not None
    assert loaded.judge == expected.judge
    raw = json.loads(store.path_for("run-1").read_text())["judge"]
    assert raw["planned_snapshot_ref"] == JUDGE_IMAGE_REF
    assert raw["snapshot_image_id"] == JUDGE_IMAGE_ID
    assert raw["snapshot_image_ref"] == JUDGE_IMAGE_REF
    assert raw["snapshot_source_container_id"] == "work-real"


@pytest.mark.parametrize(
    "updates",
    [
        {"planned_snapshot_ref": "judge:latest"},
        {
            "planned_snapshot_ref": JUDGE_IMAGE_REF,
            "snapshot_image_id": "sha256:not-a-digest",
            "snapshot_image_ref": JUDGE_IMAGE_REF,
            "snapshot_source_container_id": "work-real",
        },
        {
            "planned_snapshot_ref": JUDGE_IMAGE_REF,
            "snapshot_image_id": JUDGE_IMAGE_ID,
            "snapshot_image_ref": JUDGE_IMAGE_REF,
        },
        {
            "planned_snapshot_ref": JUDGE_IMAGE_REF,
            "snapshot_image_id": JUDGE_IMAGE_ID,
            "snapshot_image_ref": (f"rsi-harness-rootfs:judge-round-{'f' * 64}"),
            "snapshot_source_container_id": "work-real",
        },
        {
            "planned_snapshot_ref": JUDGE_IMAGE_REF,
            "snapshot_image_id": JUDGE_IMAGE_ID,
            "snapshot_image_ref": JUDGE_IMAGE_REF,
            "snapshot_source_container_id": "unsafe/container",
        },
    ],
)
def test_judge_snapshot_authority_rejects_malformed_or_partial_identity(
    updates,
) -> None:
    with pytest.raises(ValidationError):
        JudgeResourceLease.model_validate(updates)


def test_store_redacts_authorization_value_inside_lease_error(tmp_path) -> None:
    store = LeaseStore(tmp_path / "leases")
    unsafe = lease(tmp_path).model_copy(
        update={
            "error": "access-token=LEAKME&safe=yes\n"
            "Authorization: Custom opaque=AUTHLEAK"
        }
    )

    store.write(unsafe)

    raw = store.path_for("run-1").read_text()
    assert "LEAKME" not in raw
    assert "AUTHLEAK" not in raw
    assert "[REDACTED]" in raw


@pytest.mark.parametrize(
    "image_ref",
    (
        "sha256:",
        "sha256:abc",
        "base@sha256:",
        "base@sha256:not-hex",
        f"base@sha256:{'a' * 63}",
        f"base@sha256:{'a' * 64} trailing",
    ),
)
def test_cleanup_image_authority_requires_exact_full_digest(image_ref, tmp_path):
    payload = lease(tmp_path).model_dump()
    payload["cleanup_image_ref"] = image_ref
    with pytest.raises(ValidationError, match="immutable"):
        ResourceLease.model_validate(payload)

    forged = lease(tmp_path).model_copy(update={"cleanup_image_ref": image_ref})
    with pytest.raises(ValidationError, match="immutable"):
        LeaseStore(tmp_path / "leases").write(forged)


def test_store_persists_exact_retained_work_image_authority(tmp_path) -> None:
    store = LeaseStore(tmp_path / "leases")
    expected = lease(tmp_path).model_copy(
        update={
            "work": lease(tmp_path).work.model_copy(
                update={
                    "planned_retained_image_ref": RETAINED_IMAGE_REF,
                    "retained_image_id": RETAINED_IMAGE_ID,
                    "retained_image_ref": RETAINED_IMAGE_REF,
                }
            )
        }
    )

    store.write(expected)

    loaded = store.read("run-1")
    assert loaded is not None
    assert loaded.work.planned_retained_image_ref == RETAINED_IMAGE_REF
    assert loaded.work.retained_image_id == RETAINED_IMAGE_ID
    assert loaded.work.retained_image_ref == RETAINED_IMAGE_REF
    raw = json.loads(store.path_for("run-1").read_text())["work"]
    assert raw["planned_retained_image_ref"] == RETAINED_IMAGE_REF
    assert raw["retained_image_id"] == RETAINED_IMAGE_ID
    assert raw["retained_image_ref"] == RETAINED_IMAGE_REF


def test_store_persists_divergent_retained_image_rollback_authority(
    tmp_path,
) -> None:
    rollback_ref = f"rsi-harness-rootfs:retained-work-{'d' * 64}"
    payload = lease(tmp_path).model_dump()
    payload["work"].update(
        {
            "planned_retained_image_ref": RETAINED_IMAGE_REF,
            "retained_image_rollback": {
                "image_id": RETAINED_IMAGE_ID,
                "image_ref": rollback_ref,
            },
        }
    )
    expected = ResourceLease.model_validate(payload)
    store = LeaseStore(tmp_path / "leases")

    store.write(expected)

    loaded = store.read("run-1")
    assert loaded is not None
    assert loaded.work.retained_image_id is None
    assert loaded.work.retained_image_ref is None
    assert loaded.work.retained_image_rollback is not None
    assert loaded.work.retained_image_rollback.image_id == RETAINED_IMAGE_ID
    assert loaded.work.retained_image_rollback.image_ref == rollback_ref


def test_store_persists_work_quiescence_plan_before_mutation(tmp_path) -> None:
    payload = lease(tmp_path).model_dump()
    payload["work"].update(
        {
            "paused": False,
            "planned_quiescence": "pause-if-running",
        }
    )
    expected = ResourceLease.model_validate(payload)
    store = LeaseStore(tmp_path / "leases")

    store.write(expected)

    loaded = store.read("run-1")
    assert loaded is not None
    assert loaded.work.planned_quiescence == "pause-if-running"
    assert loaded.work.paused is False
    assert loaded.work.stopped is False


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"planned_retained_image_ref": "retained-work:latest"}, "reference"),
        ({"retained_image_id": "sha256:abc"}, "image ID"),
        ({"retained_image_ref": "retained-work:latest"}, "reference"),
        (
            {
                "planned_retained_image_ref": RETAINED_IMAGE_REF,
                "retained_image_id": RETAINED_IMAGE_ID,
            },
            "together",
        ),
        (
            {
                "planned_retained_image_ref": RETAINED_IMAGE_REF,
                "retained_image_id": RETAINED_IMAGE_ID,
                "retained_image_ref": (f"rsi-harness-rootfs:retained-work-{'d' * 64}"),
            },
            "planned",
        ),
    ),
)
def test_retained_work_image_authority_is_validated_at_durable_boundary(
    tmp_path,
    updates,
    message,
) -> None:
    payload = lease(tmp_path).model_dump()
    payload["work"].update(updates)
    with pytest.raises(ValidationError, match=message):
        ResourceLease.model_validate(payload)

    forged = lease(tmp_path).model_copy(
        update={"work": lease(tmp_path).work.model_copy(update=updates)}
    )
    with pytest.raises(ValidationError, match=message):
        LeaseStore(tmp_path / "leases").write(forged)


def test_work_lease_rejects_claiming_paused_and_stopped_together(tmp_path) -> None:
    payload = lease(tmp_path).model_dump()
    payload["work"]["stopped"] = True
    with pytest.raises(ValidationError, match="paused and stopped"):
        ResourceLease.model_validate(payload)

    forged = lease(tmp_path).model_copy(
        update={"work": lease(tmp_path).work.model_copy(update={"stopped": True})}
    )
    with pytest.raises(ValidationError, match="paused and stopped"):
        LeaseStore(tmp_path / "leases").write(forged)


def rootfs_round_lease(tmp_path: Path, *, actual: bool = True) -> ResourceLease:
    original = lease(tmp_path)
    return original.model_copy(
        update={
            "judge": JudgeResourceLease(
                round_id="agent-1",
                planned_network="rsi-run-1-agent-1-judge-network",
                network_id="network-judge-stale",
                network_name="rsi-run-1-agent-1-judge-network",
                planned_policy_rule_id="policy-judge-real",
                policy_rule_id="policy-judge-real",
                planned_snapshot="snapshot:run-1:agent-1",
                planned_snapshot_ref=JUDGE_IMAGE_REF,
                snapshot_lease_id="snapshot-real" if actual else None,
                snapshot_image_id=JUDGE_IMAGE_ID if actual else None,
                snapshot_image_ref=JUDGE_IMAGE_REF if actual else None,
                snapshot_source_container_id="work-real" if actual else None,
            )
        }
    )


def retained_work_lease(
    tmp_path: Path,
    *,
    actual: bool = True,
    live_work: bool = True,
) -> ResourceLease:
    original = lease(tmp_path)
    work = original.work.model_copy(
        update={
            "container_id": "work-real" if live_work else None,
            "planned_container": "rsi-run-1-work" if live_work else None,
            "planned_network": ("rsi-run-1-work-network" if live_work else None),
            "network_id": "network-work-stale" if live_work else None,
            "network_name": "rsi-run-1-work-network" if live_work else None,
            "planned_policy_rule_id": "policy-work-real" if live_work else None,
            "policy_rule_id": "policy-work-real" if live_work else None,
            "planned_retained_image_ref": RETAINED_IMAGE_REF,
            "retained_image_id": RETAINED_IMAGE_ID if actual else None,
            "retained_image_ref": RETAINED_IMAGE_REF if actual else None,
            "paused": live_work,
        }
    )
    return original.model_copy(
        update={
            "phase": RunStatus.COMPLETED.value,
            "status": RunStatus.COMPLETED,
            "work": work,
            "judge": JudgeResourceLease(),
        }
    )


def empty_non_image_resources(backend: RecoveryBackend) -> None:
    backend.containers = {}
    backend.networks = {}
    backend.policies = set()
    backend.snapshots = {}
    backend.mounts = set()


def test_recovery_clears_planned_retained_ref_after_commit_rollback_absence(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = retained_work_lease(tmp_path, actual=False)
    store.write(original)
    backend = RecoveryBackend()
    backend.containers.pop("judge-real")
    backend.snapshots = {}
    backend.mounts = set()
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    final = store.read("run-1")
    assert final is not None
    assert final.work.planned_retained_image_ref is None
    assert final.work.retained_image_id is None
    assert not any(name == "remove_image" for name, _ in backend.events)
    assert ("list_images", "retained-work-rootfs") in backend.events


def test_recovery_attests_and_preserves_image_after_identity_fsync_rollback_failure(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = retained_work_lease(tmp_path)
    store.write(original)
    backend = RecoveryBackend()
    backend.containers.pop("judge-real")
    backend.snapshots = {}
    backend.mounts = set()
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    final = store.read("run-1")
    assert final is not None
    assert final.work.retained_image_id == RETAINED_IMAGE_ID
    assert final.work.retained_image_ref == RETAINED_IMAGE_REF
    assert RETAINED_IMAGE_ID in backend.images
    assert ("inspect_image", RETAINED_IMAGE_ID) in backend.events


def test_recovery_removes_round_image_after_judge_is_already_absent(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = rootfs_round_lease(tmp_path)
    store.write(original)
    backend = RecoveryBackend()
    backend.containers.pop("judge-real")
    backend.snapshots = {}
    backend.mounts = set()
    backend.images[JUDGE_IMAGE_ID] = image_state(
        JUDGE_IMAGE_ID,
        JUDGE_IMAGE_REF,
        role="rootfs-snapshot",
        round_id="agent-1",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    assert backend.images == {}
    names = [name for name, _ in backend.events]
    assert names.index("remove_network") < names.index("remove_image")
    assert names.index("remove_image") < names.index("unpause")
    final = store.read("run-1")
    assert final is not None
    assert final.judge.planned_snapshot_ref is None
    assert final.judge.snapshot_image_id is None


def test_recovery_discovers_planned_only_round_image_after_commit_crash(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    store.write(rootfs_round_lease(tmp_path, actual=False))
    backend = RecoveryBackend()
    backend.containers.pop("judge-real")
    backend.snapshots = {}
    backend.mounts = set()
    backend.images[JUDGE_IMAGE_ID] = image_state(
        JUDGE_IMAGE_ID,
        JUDGE_IMAGE_REF,
        role="rootfs-snapshot",
        round_id="agent-1",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    assert backend.images == {}
    assert ("remove_image", JUDGE_IMAGE_ID) in backend.events
    final = store.read("run-1")
    assert final is not None
    assert final.judge.planned_snapshot_ref is None


def test_recovery_contains_every_exact_labeled_round_image_without_selecting_one(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    store.write(rootfs_round_lease(tmp_path))
    backend = RecoveryBackend()
    backend.containers.pop("judge-real")
    backend.snapshots = {}
    backend.mounts = set()
    backend.images[JUDGE_IMAGE_ID] = image_state(
        JUDGE_IMAGE_ID,
        JUDGE_IMAGE_REF,
        role="rootfs-snapshot",
        round_id="agent-1",
    )
    backend.images[SECOND_JUDGE_IMAGE_ID] = image_state(
        SECOND_JUDGE_IMAGE_ID,
        "rsi-harness-rootfs:duplicate-round-image",
        role="rootfs-snapshot",
        round_id="agent-1",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    assert backend.images == {}
    assert ("remove_image", JUDGE_IMAGE_ID) in backend.events
    assert ("remove_image", SECOND_JUDGE_IMAGE_ID) in backend.events


def test_recovery_rejects_mismatched_round_source_without_deleting_image(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = rootfs_round_lease(tmp_path)
    store.write(original)
    backend = RecoveryBackend()
    backend.containers.pop("judge-real")
    backend.snapshots = {}
    backend.mounts = set()
    backend.images[JUDGE_IMAGE_ID] = image_state(
        JUDGE_IMAGE_ID,
        JUDGE_IMAGE_REF,
        role="rootfs-snapshot",
        round_id="agent-1",
        source_container_id="other-work",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="attestation"):
        manager.recover("run-1")

    assert JUDGE_IMAGE_ID in backend.images
    assert not any(name == "remove_image" for name, _ in backend.events)
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required
    assert retained.judge.snapshot_image_id == JUDGE_IMAGE_ID
    assert retained.work.paused is True


def test_explicit_cleanup_refuses_referenced_retained_image(tmp_path) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = retained_work_lease(tmp_path, live_work=False)
    store.write(original)
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
        in_use=True,
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="referenced"):
        manager.cleanup("run-1", delete_workspace=True)

    assert RETAINED_IMAGE_ID in backend.images
    assert not any(name == "remove_image" for name, _ in backend.events)
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required
    assert retained.work.retained_image_id == RETAINED_IMAGE_ID


@pytest.mark.parametrize(
    ("actual", "authority"),
    (
        pytest.param(True, RETAINED_IMAGE_ID, id="actual"),
        pytest.param(False, RETAINED_IMAGE_REF, id="planned"),
    ),
)
def test_explicit_cleanup_clears_retained_image_authority_after_proven_absence(
    tmp_path, actual, authority
) -> None:
    store = LeaseStore(tmp_path / "leases")
    store.write(retained_work_lease(tmp_path, actual=actual, live_work=False))
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.cleanup("run-1", delete_workspace=True)
    assert manager.recover("run-1") == ("run-1",)

    assert backend.image_queries == [
        {
            "rsi-harness.run-id": "run-1",
            "rsi-harness.task-id": "task-1",
            "rsi-harness.role": "retained-work-rootfs",
            "rsi-harness.round-id": "final",
        }
    ]
    assert ("inspect_image", authority) in backend.events
    assert not any(
        name in {"inspect_image_use", "remove_image"} for name, _ in backend.events
    )
    final = store.read("run-1")
    assert final is not None and not final.recovery_required
    assert final.work.planned_retained_image_ref is None
    assert final.work.retained_image_id is None
    assert final.work.retained_image_ref is None
    assert final.work.retained_image_rollback is None


def test_recovery_clears_rollback_image_authority_after_proven_absence(
    tmp_path,
) -> None:
    rollback_ref = f"rsi-harness-rootfs:retained-work-{'d' * 64}"
    planned = retained_work_lease(tmp_path, actual=False, live_work=False)
    original = planned.model_copy(
        update={
            "work": planned.work.model_copy(
                update={
                    "retained_image_rollback": RetainedImageRollbackAuthority(
                        image_id=RETAINED_IMAGE_ID,
                        image_ref=rollback_ref,
                    )
                }
            )
        }
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(original)
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    assert manager.recover("run-1") == ("run-1",)
    assert manager.recover("run-1") == ("run-1",)

    assert backend.image_queries == [
        {
            "rsi-harness.run-id": "run-1",
            "rsi-harness.task-id": "task-1",
            "rsi-harness.role": "retained-work-rootfs",
            "rsi-harness.round-id": "final",
        }
    ]
    assert ("inspect_image", RETAINED_IMAGE_ID) in backend.events
    assert not any(
        name in {"inspect_image_use", "remove_image"} for name, _ in backend.events
    )
    final = store.read("run-1")
    assert final is not None and not final.recovery_required
    assert final.work.planned_retained_image_ref is None
    assert final.work.retained_image_id is None
    assert final.work.retained_image_ref is None
    assert final.work.retained_image_rollback is None


@pytest.mark.parametrize("actual", (True, False), ids=("actual", "planned"))
def test_explicit_cleanup_fails_closed_before_mutating_ambiguous_images(
    tmp_path, actual
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = retained_work_lease(tmp_path, actual=actual, live_work=False)
    store.write(original)
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    backend.images[SECOND_RETAINED_IMAGE_ID] = image_state(
        SECOND_RETAINED_IMAGE_ID,
        SECOND_RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="ambiguous retained Work"):
        manager.cleanup("run-1", delete_workspace=True)

    assert set(backend.images) == {
        RETAINED_IMAGE_ID,
        SECOND_RETAINED_IMAGE_ID,
    }
    assert not any(
        name in {"inspect_image_use", "remove_image"} for name, _ in backend.events
    )
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required
    assert retained.work == original.work


def test_recovery_fails_closed_before_mutating_ambiguous_rollback_images(
    tmp_path,
) -> None:
    rollback_ref = f"rsi-harness-rootfs:retained-work-{'d' * 64}"
    planned = retained_work_lease(tmp_path, actual=False, live_work=False)
    original = planned.model_copy(
        update={
            "work": planned.work.model_copy(
                update={
                    "retained_image_rollback": RetainedImageRollbackAuthority(
                        image_id=RETAINED_IMAGE_ID,
                        image_ref=rollback_ref,
                    )
                }
            )
        }
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(original)
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        rollback_ref,
        role="retained-work-rootfs",
        round_id="final",
    )
    backend.images[SECOND_RETAINED_IMAGE_ID] = image_state(
        SECOND_RETAINED_IMAGE_ID,
        SECOND_RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="ambiguous retained Work"):
        manager.recover("run-1")

    assert set(backend.images) == {
        RETAINED_IMAGE_ID,
        SECOND_RETAINED_IMAGE_ID,
    }
    assert not any(
        name in {"inspect_image_use", "remove_image"} for name, _ in backend.events
    )
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required
    assert retained.work == original.work


@pytest.mark.parametrize("failure", ("query", "remove"))
def test_explicit_cleanup_image_backend_error_preserves_exact_authority(
    tmp_path, failure
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = retained_work_lease(tmp_path, live_work=False)
    store.write(original)
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    if failure == "query":
        backend.list_images = lambda **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("image query backend exploded")
        )
    else:
        backend.remove_image = lambda _image_id: (_ for _ in ()).throw(  # type: ignore[method-assign]
            RuntimeError("image remove backend exploded")
        )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match=f"image {failure} backend exploded"):
        manager.cleanup("run-1", delete_workspace=True)

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required
    assert retained.work.retained_image_id == RETAINED_IMAGE_ID


def test_explicit_cleanup_preserves_authority_when_post_remove_absence_query_fails(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = retained_work_lease(tmp_path, live_work=False)
    store.write(original)
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    original_inspect = backend.inspect_image

    def fail_after_removal(image_id: str):
        if image_id not in backend.images:
            raise RuntimeError("image absence query backend exploded")
        return original_inspect(image_id)

    backend.inspect_image = fail_after_removal  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="absence query backend exploded"):
        manager.cleanup("run-1", delete_workspace=True)

    assert RETAINED_IMAGE_ID not in backend.images
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required
    assert retained.work.retained_image_id == RETAINED_IMAGE_ID


def test_ordinary_recover_preserves_final_retained_image_and_authority(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = retained_work_lease(tmp_path, live_work=False)
    store.write(original)
    backend = RecoveryBackend()
    empty_non_image_resources(backend)
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    final = store.read("run-1")
    assert final is not None
    assert final.work.retained_image_id == RETAINED_IMAGE_ID
    assert final.work.retained_image_ref == RETAINED_IMAGE_REF
    assert RETAINED_IMAGE_ID in backend.images
    assert not any(name == "remove_image" for name, _ in backend.events)


def test_ordinary_recover_removes_untrusted_retained_image_rollback_authority(
    tmp_path,
) -> None:
    rollback_ref = f"rsi-harness-rootfs:retained-work-{'d' * 64}"
    original = retained_work_lease(tmp_path, actual=False, live_work=True).model_copy(
        update={
            "work": retained_work_lease(
                tmp_path, actual=False, live_work=True
            ).work.model_copy(
                update={
                    "retained_image_rollback": RetainedImageRollbackAuthority(
                        image_id=RETAINED_IMAGE_ID,
                        image_ref=rollback_ref,
                    )
                }
            )
        }
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(original)
    backend = RecoveryBackend()
    backend.containers.pop("judge-real")
    backend.snapshots = {}
    backend.mounts = set()
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        rollback_ref,
        role="retained-work-rootfs",
        round_id="final",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    final = store.read("run-1")
    assert final is not None
    assert final.work.planned_retained_image_ref is None
    assert final.work.retained_image_rollback is None
    assert RETAINED_IMAGE_ID not in backend.images


def test_explicit_cleanup_removes_retained_image_after_every_runtime_dependency(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = rootfs_round_lease(tmp_path).model_copy(
        update={
            "work": rootfs_round_lease(tmp_path).work.model_copy(
                update={
                    "planned_retained_image_ref": RETAINED_IMAGE_REF,
                    "retained_image_id": RETAINED_IMAGE_ID,
                    "retained_image_ref": RETAINED_IMAGE_REF,
                }
            )
        }
    )
    store.write(original)
    backend = RecoveryBackend()
    backend.snapshots = {}
    backend.mounts = set()
    backend.images[JUDGE_IMAGE_ID] = image_state(
        JUDGE_IMAGE_ID,
        JUDGE_IMAGE_REF,
        role="rootfs-snapshot",
        round_id="agent-1",
    )
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.cleanup("run-1", delete_workspace=True)

    assert backend.images == {}
    names = [name for name, _ in backend.events]
    first_image_remove = names.index("remove_image")
    retained_remove = max(
        index
        for index, event in enumerate(backend.events)
        if event == ("remove_image", RETAINED_IMAGE_ID)
    )
    assert names.index("remove", names.index("stop")) < first_image_remove
    assert first_image_remove < names.index("unpause")
    assert names.index("remove", names.index("unpause")) < retained_remove
    assert names.index("remove_policy", names.index("unpause")) < retained_remove
    assert names.index("remove_network", names.index("unpause")) < retained_remove
    final = store.read("run-1")
    assert final is not None
    assert final.work.planned_retained_image_ref is None
    assert final.work.retained_image_id is None


def test_recovery_distrusts_stale_ids_and_orders_containment_before_unpause(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    current = lease(tmp_path).model_copy(
        update={
            "judge": lease(tmp_path).judge.model_copy(
                update={"container_id": "stale-judge"}
            ),
            "work": lease(tmp_path).work.model_copy(
                update={"container_id": "stale-work"}
            ),
        }
    )
    assert current.workspace_path is not None
    current.workspace_path.mkdir(parents=True)
    store.write(current)
    backend = RecoveryBackend()
    manager = RecoveryManager(
        store=store,
        backend=backend,
        managed_root=tmp_path / "runs",
    )

    recovered = manager.recover("run-1")

    assert recovered == ("run-1",)
    assert backend.events == [
        ("list_containers", "helper"),
        ("list_containers", "judge"),
        ("stop", "judge-real"),
        ("remove", "judge-real"),
        ("inspect_policy", "policy-judge-real"),
        ("remove_policy", "policy-judge-real"),
        ("inspect_policy", "policy-judge-real"),
        ("list_networks", "judge"),
        ("inspect_network", "network-judge-real"),
        ("remove_network", "network-judge-real"),
        ("list_networks", "judge"),
        ("discover_snapshot", "run-1"),
        ("release_snapshot", "/managed/run-1/snap/merged"),
        ("inspect_mount", "/managed/run-1/snap/merged"),
        ("discover_snapshot", "run-1"),
        ("inspect_mount", "/managed/run-1/snap/merged"),
        ("list_containers", "work"),
        ("unpause", "work-real"),
        ("stop", "work-real"),
        ("remove", "work-real"),
        ("inspect_policy", "policy-work-real"),
        ("remove_policy", "policy-work-real"),
        ("inspect_policy", "policy-work-real"),
        ("list_networks", "work"),
        ("inspect_network", "network-work-real"),
        ("remove_network", "network-work-real"),
        ("list_networks", "work"),
    ]
    final = store.read("run-1")
    assert final is not None
    assert final.phase == "interrupted"
    assert final.status == RunStatus.CANCELLED
    assert final.work.container_id is None
    assert final.work.network_id is None
    assert final.work.policy_rule_id is None
    assert final.judge.container_id is None
    assert final.judge.network_id is None
    assert final.judge.policy_rule_id is None
    assert current.workspace_path is not None and current.workspace_path.parent.exists()


def test_recovery_keeps_work_paused_when_live_judge_containment_is_unproven(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    store.write(lease(tmp_path))
    backend = RecoveryBackend()

    def fail_stop(container_id: str) -> None:
        backend.events.append(("stop", container_id))
        raise RuntimeError("cannot prove stop")

    backend.stop_container = fail_stop  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store,
        backend=backend,
        managed_root=tmp_path / "runs",
    )

    with pytest.raises(RuntimeError, match="containment"):
        manager.recover("run-1")

    assert not any(name == "unpause" for name, _ in backend.events)
    assert not any(name == "release_snapshot" for name, _ in backend.events)
    assert store.read("run-1").work.paused is True  # type: ignore[union-attr]


def test_recovery_contains_every_exact_labeled_judge_before_work_discovery(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    store.write(
        original.model_copy(
            update={
                "judge": original.judge.model_copy(
                    update={
                        "planned_snapshot": None,
                        "snapshot_lease_id": None,
                        "snapshot_merged_path": None,
                        "snapshot_process_id": None,
                    }
                )
            }
        )
    )
    backend = RecoveryBackend()
    backend.containers["judge-old"] = {
        **backend.containers["judge-real"],
        "running": True,
    }
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    work_lookup = backend.events.index(("list_containers", "work"))
    assert ("stop", "judge-real") in backend.events[:work_lookup]
    assert ("remove", "judge-real") in backend.events[:work_lookup]
    assert ("stop", "judge-old") in backend.events[:work_lookup]
    assert ("remove", "judge-old") in backend.events[:work_lookup]


def test_recovery_contains_leaked_cleanup_helper_before_other_resources(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    store.write(lease(tmp_path))
    backend = RecoveryBackend()
    backend.containers["helper-leaked"] = {
        "labels": {
            "rsi-harness.run-id": "run-1",
            "rsi-harness.task-id": "task-1",
            "rsi-harness.role": "helper",
        },
        "running": True,
        "paused": False,
    }
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    helper_lookup = backend.events.index(("list_containers", "helper"))
    helper_stop = backend.events.index(("stop", "helper-leaked"))
    helper_remove = backend.events.index(("remove", "helper-leaked"))
    judge_lookup = backend.events.index(("list_containers", "judge"))
    assert helper_lookup < helper_stop < helper_remove < judge_lookup
    assert "helper-leaked" not in backend.containers


def test_cleanup_refuses_when_leaked_helper_removal_is_unproven(tmp_path) -> None:
    managed = tmp_path / "runs"
    workspace = managed / "run-1" / "workspace"
    workspace.mkdir(parents=True)
    store = LeaseStore(tmp_path / "leases")
    store.write(
        lease(tmp_path).model_copy(
            update={
                "workspace_path": workspace,
                "phase": RunStatus.COMPLETED.value,
                "status": RunStatus.COMPLETED,
            }
        )
    )
    backend = RecoveryBackend()
    backend.containers = {
        "helper-leaked": {
            "labels": {
                "rsi-harness.run-id": "run-1",
                "rsi-harness.task-id": "task-1",
                "rsi-harness.role": "helper",
            },
            "running": True,
            "paused": False,
        }
    }

    def fail_remove(container_id: str) -> None:
        backend.events.append(("remove", container_id))
        raise RuntimeError("helper removal failed")

    backend.remove_container = fail_remove  # type: ignore[method-assign]
    manager = RecoveryManager(store=store, backend=backend, managed_root=managed)

    with pytest.raises(RuntimeError, match="helper.*removal"):
        manager.cleanup("run-1", delete_workspace=True)

    assert workspace.exists()
    assert not any(name == "delete_workspace" for name, _ in backend.events)
    assert store.read("run-1").recovery_required is True  # type: ignore[union-attr]


def test_recovery_continues_after_infrastructure_error_for_each_judge(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    store.write(original)
    backend = RecoveryBackend()
    backend.containers["judge-old"] = {
        **backend.containers["judge-real"],
        "running": True,
    }
    original_remove = backend.remove_container

    def fail_first_judge(container_id: str) -> None:
        if container_id == "judge-real":
            backend.events.append(("remove", container_id))
            raise InfrastructureError("docker removal authority failed")
        original_remove(container_id)

    backend.remove_container = fail_first_judge  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="docker removal authority failed"):
        manager.recover("run-1")

    assert ("remove", "judge-real") in backend.events
    assert ("remove", "judge-old") in backend.events
    assert ("list_containers", "work") not in backend.events
    retained = store.read("run-1")
    assert retained is not None
    assert retained.recovery_required is True
    assert retained.judge.container_id == "judge-real"
    assert retained.judge.network_id == "network-judge-stale"
    assert retained.error is not None
    assert "docker removal authority failed" in retained.error


@pytest.mark.parametrize("planned_only", [True, False])
def test_recovery_discovers_and_releases_planned_or_unmounted_snapshot_authority(
    tmp_path, planned_only
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    judge = original.judge.model_copy(
        update={
            "snapshot_lease_id": None if planned_only else "snapshot-real",
            "snapshot_merged_path": None
            if planned_only
            else Path("/managed/run-1/snap/merged"),
            "snapshot_process_id": None,
        }
    )
    store.write(original.model_copy(update={"judge": judge}))
    backend = RecoveryBackend()
    backend.containers = {}
    backend.mounts.clear()
    backend.snapshots["snapshot-real"] = SnapshotRecoveryAuthority(
        run_id="run-1",
        lease_id="snapshot-real",
        round_id="agent-1",
        merged_path=Path("/managed/run-1/snap/merged"),
        process_id=None,
        manifest_requires_recovery=True,
        layers_present=True,
        process_alive=False,
    )
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    final = store.read("run-1")
    assert final is not None
    assert final.judge.planned_snapshot is None
    assert final.judge.snapshot_lease_id is None
    assert backend.snapshots == {}
    assert any(name == "release_snapshot" for name, _ in backend.events)


def test_ambiguous_snapshot_discovery_fails_closed_without_forgetting_fields(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    store.write(original)
    backend = RecoveryBackend()
    backend.containers = {}
    backend.snapshots["snapshot-other"] = SnapshotRecoveryAuthority(
        run_id="run-1",
        lease_id="snapshot-other",
        round_id="agent-1",
        merged_path=Path("/managed/run-1/other/merged"),
        process_id=None,
        manifest_requires_recovery=True,
        layers_present=True,
        process_alive=False,
    )
    # Planned-only recovery searches by round because no actual lease ID was
    # durably observed before the crash.
    planned = original.model_copy(
        update={"judge": original.judge.model_copy(update={"snapshot_lease_id": None})}
    )
    store.write(planned)
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="ambiguous active snapshot"):
        manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required
    assert retained.judge.planned_snapshot == planned.judge.planned_snapshot


def test_cleanup_deletes_only_exact_canonical_managed_workspace(tmp_path) -> None:
    managed = tmp_path / "runs"
    workspace = managed / "run-1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "answer").write_text("keep siblings safe")
    sibling = managed / "run-2" / "workspace"
    sibling.mkdir(parents=True)
    store = LeaseStore(tmp_path / "leases")
    store.write(
        lease(tmp_path).model_copy(
            update={
                "workspace_path": workspace,
                "phase": RunStatus.COMPLETED.value,
                "status": RunStatus.COMPLETED,
            }
        )
    )
    manager = RecoveryManager(
        store=store,
        backend=RecoveryBackend(),
        managed_root=managed,
    )

    manager.cleanup("run-1", delete_workspace=False)
    assert workspace.exists()
    assert store.read("run-1").status == RunStatus.COMPLETED  # type: ignore[union-attr]
    manager.cleanup("run-1", delete_workspace=True)
    assert not workspace.exists()
    assert sibling.exists()
    assert (
        "delete_workspace",
        f"{workspace}|{CLEANUP_IMAGE}|run-1|task-1",
    ) in manager._backend.events

    outside = tmp_path / "outside"
    outside.mkdir()
    store.write(lease(tmp_path).model_copy(update={"workspace_path": outside}))
    with pytest.raises(ValueError, match="exact managed run workspace"):
        manager.cleanup("run-1", delete_workspace=True)
    assert outside.exists()


def test_cleanup_without_fixed_image_authority_fails_closed(tmp_path) -> None:
    managed = tmp_path / "runs"
    workspace = managed / "run-1" / "workspace"
    workspace.mkdir(parents=True)
    store = LeaseStore(tmp_path / "leases")
    store.write(
        lease(tmp_path).model_copy(
            update={
                "workspace_path": workspace,
                "cleanup_image_ref": None,
                "phase": RunStatus.COMPLETED.value,
                "status": RunStatus.COMPLETED,
            }
        )
    )
    manager = RecoveryManager(
        store=store, backend=RecoveryBackend(), managed_root=managed
    )

    with pytest.raises(RuntimeError, match="cleanup image authority"):
        manager.cleanup("run-1", delete_workspace=True)

    assert workspace.exists()


def test_snapshot_release_exception_is_durably_fail_closed_with_authority(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    store.write(original)
    backend = RecoveryBackend()
    backend.containers = {}

    def fail_release(_authority: SnapshotRecoveryAuthority) -> None:
        raise RuntimeError("snapshot release backend exploded")

    backend.release_snapshot = fail_release  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="snapshot release backend exploded"):
        manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.work.paused is True
    assert retained.judge.planned_snapshot == original.judge.planned_snapshot
    assert retained.judge.snapshot_lease_id == original.judge.snapshot_lease_id
    assert retained.judge.snapshot_merged_path == original.judge.snapshot_merged_path
    assert retained.judge.snapshot_process_id == original.judge.snapshot_process_id


def test_policy_remove_exception_is_durably_fail_closed_with_exact_rules(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    original = original.model_copy(
        update={
            "judge": original.judge.model_copy(
                update={
                    "planned_snapshot": None,
                    "snapshot_lease_id": None,
                    "snapshot_merged_path": None,
                    "snapshot_process_id": None,
                }
            )
        }
    )
    store.write(original)
    backend = RecoveryBackend()
    backend.containers = {}

    def fail_remove_policy(_rule_id: str) -> None:
        raise RuntimeError("policy remove backend exploded")

    backend.remove_policy = fail_remove_policy  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="policy remove backend exploded"):
        manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.judge.policy_rule_id == original.judge.policy_rule_id
    assert retained.judge.planned_policy_rule_id == (
        original.judge.planned_policy_rule_id
    )
    assert retained.work.policy_rule_id == original.work.policy_rule_id
    assert retained.work.planned_policy_rule_id == original.work.planned_policy_rule_id


@pytest.mark.parametrize("failure", ("query", "remove"))
def test_network_backend_exception_is_durably_fail_closed_with_exact_authority(
    tmp_path, failure
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    original = original.model_copy(
        update={
            "judge": original.judge.model_copy(
                update={
                    "planned_snapshot": None,
                    "snapshot_lease_id": None,
                    "snapshot_merged_path": None,
                    "snapshot_process_id": None,
                    "planned_policy_rule_id": None,
                    "policy_rule_id": None,
                }
            ),
            "work": original.work.model_copy(
                update={
                    "planned_policy_rule_id": None,
                    "policy_rule_id": None,
                }
            ),
        }
    )
    store.write(original)
    backend = RecoveryBackend()
    backend.containers = {}

    if failure == "query":

        def fail_list_networks(*, labels):
            del labels
            raise RuntimeError("network query backend exploded")

        backend.list_networks = fail_list_networks  # type: ignore[method-assign]
    else:

        def fail_remove_network(_network_id: str) -> None:
            raise RuntimeError("network remove backend exploded")

        backend.remove_network = fail_remove_network  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match=f"network {failure} backend exploded"):
        manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.judge.network_id == original.judge.network_id
    assert retained.judge.network_name == original.judge.network_name
    assert retained.judge.planned_network == original.judge.planned_network
    assert retained.work.network_id == original.work.network_id
    assert retained.work.network_name == original.work.network_name
    assert retained.work.planned_network == original.work.planned_network


def test_container_inspection_exception_preserves_original_paused_lease(
    tmp_path,
) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path)
    store.write(original)
    backend = RecoveryBackend()

    def fail_inspection(_container_id: str):
        raise RuntimeError("container inspection backend exploded")

    backend.inspect_container = fail_inspection  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="container inspection backend exploded"):
        manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.work == original.work
    assert retained.judge == original.judge


def test_workspace_delete_failure_contains_helper_created_before_run_raises(
    tmp_path,
) -> None:
    managed = tmp_path / "runs"
    workspace = managed / "run-1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "checkpoint").write_text("retain")
    store = LeaseStore(tmp_path / "leases")
    completed = lease(tmp_path).model_copy(
        update={
            "workspace_path": workspace,
            "phase": RunStatus.COMPLETED.value,
            "status": RunStatus.COMPLETED,
        }
    )
    store.write(completed)
    backend = RecoveryBackend()

    def create_helper_then_fail(
        _workspace: Path,
        *,
        image_ref: str,
        run_id: str,
        task_id: str,
    ) -> None:
        del _workspace, image_ref
        backend.containers["helper-created-during-delete"] = {
            "labels": {
                "rsi-harness.run-id": run_id,
                "rsi-harness.task-id": task_id,
                "rsi-harness.role": "helper",
            },
            "running": True,
            "paused": False,
        }
        raise RuntimeError("cleanup helper run failed after create")

    backend.delete_workspace = create_helper_then_fail  # type: ignore[method-assign]
    manager = RecoveryManager(store=store, backend=backend, managed_root=managed)

    with pytest.raises(RuntimeError, match="run failed after create"):
        manager.cleanup("run-1", delete_workspace=True)

    assert workspace.exists()
    assert (workspace / "checkpoint").read_text() == "retain"
    assert "helper-created-during-delete" not in backend.containers
    assert ("stop", "helper-created-during-delete") in backend.events
    assert ("remove", "helper-created-during-delete") in backend.events
    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.workspace_path == workspace
    assert retained.cleanup_image_ref == CLEANUP_IMAGE


def test_ambiguous_exact_work_containers_are_durably_fail_closed(tmp_path) -> None:
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path).model_copy(
        update={
            "phase": RunStatus.COMPLETED.value,
            "status": RunStatus.COMPLETED,
            "judge": JudgeResourceLease(),
            "work": WorkResourceLease(
                planned_container="rsi-run-1-work",
                container_id="stale-work",
                paused=True,
            ),
        }
    )
    store.write(original)
    backend = RecoveryBackend()
    backend.containers = {
        identity: {
            "labels": {
                "rsi-harness.run-id": "run-1",
                "rsi-harness.task-id": "task-1",
                "rsi-harness.role": "work",
            },
            "running": True,
            "paused": True,
        }
        for identity in ("work-one", "work-two")
    }
    backend.snapshots = {}
    backend.mounts = set()
    backend.networks = {}
    backend.policies = set()
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="ambiguous.*work"):
        manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.work == original.work
    assert set(backend.containers) == {"work-one", "work-two"}


@pytest.mark.parametrize("authority", ("helper", "judge", "work", "mount"))
def test_late_workspace_deletion_authority_is_durably_fail_closed(
    tmp_path, authority: str
) -> None:
    managed = tmp_path / "runs"
    workspace = managed / "run-1" / "workspace"
    workspace.mkdir(parents=True)
    store = LeaseStore(tmp_path / "leases")
    original = lease(tmp_path).model_copy(
        update={
            "phase": RunStatus.COMPLETED.value,
            "status": RunStatus.COMPLETED,
            "work": WorkResourceLease(),
            "judge": JudgeResourceLease(),
        }
    )
    store.write(original)
    backend = RecoveryBackend()
    backend.containers = {}
    backend.snapshots = {}
    backend.mounts = set()
    backend.networks = {}
    backend.policies = set()
    calls: dict[str, int] = {}
    late_id = f"late-{authority}"
    late_state = {
        "labels": {
            "rsi-harness.run-id": "run-1",
            "rsi-harness.task-id": "task-1",
            "rsi-harness.role": authority,
        },
        "running": True,
        "paused": authority == "work",
    }

    def reveal_after_recovery(*, labels):
        role = labels["rsi-harness.role"]
        calls[role] = calls.get(role, 0) + 1
        if role == authority and calls[role] > 1:
            backend.containers[late_id] = late_state
            return ((late_id, late_state),)
        return ()

    backend.list_containers = reveal_after_recovery  # type: ignore[method-assign]
    if authority == "mount":
        backend.workspace_is_mounted = lambda _workspace: True  # type: ignore[method-assign]
    manager = RecoveryManager(store=store, backend=backend, managed_root=managed)

    with pytest.raises(RuntimeError, match="workspace deletion blocked"):
        manager.cleanup("run-1", delete_workspace=True)

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.workspace_path == workspace
    assert retained.cleanup_image_ref == CLEANUP_IMAGE
    assert workspace.exists()
    if authority != "mount":
        assert backend.containers[late_id]["labels"]["rsi-harness.role"] == authority


def test_ordinary_recovery_preserves_and_attests_final_workdir_volume(
    tmp_path,
) -> None:
    expected_volume = workdir_volume()
    store = LeaseStore(tmp_path / "leases")
    store.write(split_volume_lease(tmp_path, actual=expected_volume, mounted=True))
    backend = RecoveryBackend()
    backend.volumes[expected_volume.name] = volume_state(expected_volume)
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.workdir_volume == WorkdirVolumeResourceLease(
        planned_name=expected_volume.name,
        planned_target=expected_volume.target,
        planned_snapshot_mode=expected_volume.snapshot_mode,
        planned_freshness_nonce=expected_volume.freshness_nonce,
        actual=expected_volume,
        mounted=True,
    )
    assert backend.volumes == {expected_volume.name: volume_state(expected_volume)}
    assert ("remove_volume", expected_volume.name) not in backend.events
    assert backend.volume_queries == [managed_workdir_volume_labels(expected_volume)]


def test_recovery_removes_dedicated_mismatched_rollback_volume(tmp_path) -> None:
    rollback = workdir_volume(
        run_id="wrong-run", task_id="wrong-task", target="/wrong-target"
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(split_volume_lease(tmp_path, rollback=rollback))
    backend = RecoveryBackend()
    backend.volumes[rollback.name] = volume_state(rollback)
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.recover("run-1")

    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.workdir_volume == WorkdirVolumeResourceLease()
    assert backend.volumes == {}
    assert ("remove_volume", rollback.name) in backend.events


def test_explicit_cleanup_deletes_exact_workdir_volume_then_clears_lease(
    tmp_path,
) -> None:
    expected_volume = workdir_volume()
    store = LeaseStore(tmp_path / "leases")
    store.write(split_volume_lease(tmp_path, actual=expected_volume, mounted=True))
    backend = RecoveryBackend()
    backend.volumes[expected_volume.name] = volume_state(expected_volume)
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    manager.cleanup("run-1", delete_workspace=True)

    retained = store.read("run-1")
    assert retained is not None
    assert retained.work.workdir_volume == WorkdirVolumeResourceLease()
    assert backend.volumes == {}
    removal = backend.events.index(("remove_volume", expected_volume.name))
    assert backend.events[:removal].count(("inspect_volume", expected_volume.name)) == 2
    assert (
        backend.events[:removal].count(("inspect_volume_use", expected_volume.name))
        == 2
    )
    assert (
        backend.events[removal + 1 :].count(("inspect_volume", expected_volume.name))
        == 1
    )
    assert len(backend.volume_queries) == 3


@pytest.mark.parametrize(
    "ambiguity",
    (
        "query-failure",
        "direct-failure",
        "direct-missing",
        "forged-labels",
        "malformed-state",
        "multiple",
        "driver",
        "scope",
        "referenced",
        "reference-failure",
        "removal-failure",
        "post-remove-presence",
    ),
)
def test_explicit_workdir_volume_delete_ambiguity_retains_exact_authority(
    tmp_path, ambiguity: str
) -> None:
    expected_volume = workdir_volume()
    expected_state = volume_state(expected_volume)
    store = LeaseStore(tmp_path / "leases")
    original = split_volume_lease(tmp_path, actual=expected_volume, mounted=True)
    store.write(original)
    backend = RecoveryBackend()
    backend.volumes[expected_volume.name] = expected_state
    if ambiguity == "query-failure":
        backend.volume_query_error = OSError("volume list unavailable")
    elif ambiguity == "direct-failure":
        backend.volume_inspect_error = OSError("volume inspect unavailable")
    elif ambiguity == "direct-missing":
        backend.inspect_volume = lambda _name: None  # type: ignore[method-assign]
    elif ambiguity == "forged-labels":
        forged = volume_state(
            expected_volume,
            labels={
                "rsi-harness.run-id": "forged-run",
                "rsi-harness.task-id": "task-1",
                "rsi-harness.role": "workdir-volume",
            },
        )
        backend.volume_query_result = ((expected_volume.name, forged),)
    elif ambiguity == "malformed-state":
        backend.volume_query_result = ((expected_volume.name, object()),)
    elif ambiguity == "multiple":
        backend.volume_query_result = (
            (expected_volume.name, expected_state),
            ("duplicate", {**expected_state, "name": "duplicate"}),
        )
    elif ambiguity == "driver":
        backend.volumes[expected_volume.name] = volume_state(
            expected_volume, driver="nfs"
        )
    elif ambiguity == "scope":
        backend.volumes[expected_volume.name] = volume_state(
            expected_volume, scope="global"
        )
    elif ambiguity == "referenced":
        backend.volumes[expected_volume.name] = volume_state(
            expected_volume, references=("late-helper",)
        )
    elif ambiguity == "reference-failure":
        backend.volume_reference_error = OSError("reference query unavailable")
    elif ambiguity == "removal-failure":
        backend.volume_remove_error = OSError("remove unavailable")
    elif ambiguity == "post-remove-presence":
        backend.remove_volume = lambda _name: None  # type: ignore[method-assign]
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="authority remains retained"):
        manager.cleanup("run-1", delete_workspace=True)

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.work.workdir_volume == original.work.workdir_volume


def test_full_rootfs_recovery_never_queries_volume_ports(tmp_path) -> None:
    store = LeaseStore(tmp_path / "leases")
    store.write(lease(tmp_path))
    backend = RecoveryBackend()
    backend.list_volumes = lambda **_kwargs: pytest.fail("volume list called")  # type: ignore[method-assign]
    backend.inspect_volume = lambda _name: pytest.fail("volume inspect called")  # type: ignore[method-assign]
    backend.volume_in_use = lambda _name: pytest.fail("volume use called")  # type: ignore[method-assign]
    backend.remove_volume = lambda _name: pytest.fail("volume remove called")  # type: ignore[method-assign]

    RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    ).recover("run-1")


def test_production_recovery_normalizes_volume_attrs_and_exact_references() -> None:
    expected_volume = workdir_volume()
    client = FakeDockerClient()
    docker_volume = client.volumes.create(
        expected_volume.name,
        driver="local",
        labels=managed_workdir_volume_labels(expected_volume),
    )
    docker_volume.attrs["Options"] = {"type": "none"}
    reference = FakeDockerContainer("helper-reference")
    reference.attrs["Config"]["Labels"] = {
        "rsi-harness.run-id": "run-1",
        "rsi-harness.task-id": "task-1",
        "rsi-harness.role": "helper",
    }
    reference.attrs["Mounts"] = [
        {
            "Type": "volume",
            "Name": expected_volume.name,
            "Source": expected_volume.name,
            "Destination": "/workspace",
            "RW": False,
        }
    ]
    client.containers.by_id[reference.id] = reference
    backend = ProductionRecoveryBackend(
        client, snapshot=object(), firewall=FakeFirewallBackend()
    )
    labels = managed_workdir_volume_labels(expected_volume)

    assert backend.list_volumes(labels=labels) == (
        (
            expected_volume.name,
            {
                "name": expected_volume.name,
                "driver": "local",
                "labels": labels,
                "options": {"type": "none"},
                "scope": "local",
                "container_references": (
                    {
                        "container_id": "helper-reference",
                        "labels": {
                            "rsi-harness.run-id": "run-1",
                            "rsi-harness.task-id": "task-1",
                            "rsi-harness.role": "helper",
                        },
                    },
                ),
            },
        ),
    )
    assert backend.inspect_volume(expected_volume.name) == dict(
        backend.list_volumes(labels=labels)[0][1]
    )
    assert backend.volume_in_use(expected_volume.name) is True
    client.containers.by_id.clear()

    backend.remove_volume(expected_volume.name)

    assert backend.inspect_volume(expected_volume.name) is None
    assert backend.list_volumes(labels=labels) == ()


@pytest.mark.parametrize(
    ("attribute", "malformed"),
    (
        ("Options", []),
        ("Options", ""),
        ("Labels", []),
        ("Labels", ""),
        ("Scope", []),
    ),
)
def test_production_recovery_rejects_falsy_malformed_volume_attrs(
    attribute: str, malformed: object
) -> None:
    expected_volume = workdir_volume()
    client = FakeDockerClient()
    docker_volume = client.volumes.create(
        expected_volume.name,
        driver="local",
        labels=managed_workdir_volume_labels(expected_volume),
    )
    docker_volume.attrs[attribute] = malformed
    backend = ProductionRecoveryBackend(
        client, snapshot=object(), firewall=FakeFirewallBackend()
    )

    with pytest.raises(TypeError, match="malformed volume"):
        backend.inspect_volume(expected_volume.name)


@pytest.mark.parametrize("malformed_options", ([], ""))
@pytest.mark.parametrize("authority_kind", ("actual", "rollback"))
def test_explicit_cleanup_preserves_all_authority_for_malformed_volume_options(
    tmp_path, malformed_options: object, authority_kind: str
) -> None:
    actual = workdir_volume()
    authority = (
        actual
        if authority_kind == "actual"
        else workdir_volume(
            run_id="rollback-run",
            task_id="rollback-task",
            target="/rollback-target",
        )
    )
    volume_lease = WorkdirVolumeResourceLease(
        planned_name=WORKDIR_VOLUME_NAME,
        planned_target=PurePosixPath("/workspace"),
        planned_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        planned_freshness_nonce="f" * 64,
        actual=actual if authority_kind == "actual" else None,
        rollback=authority if authority_kind == "rollback" else None,
        mounted=authority_kind == "actual",
    )
    original = ResourceLease(
        run_id="run-1",
        task_id="task-1",
        coordinator_pid=123,
        coordinator_started_at=1.5,
        phase=RunStatus.COMPLETED.value,
        status=RunStatus.COMPLETED,
        rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        work=WorkResourceLease(
            planned_retained_image_ref=RETAINED_IMAGE_REF,
            retained_image_id=RETAINED_IMAGE_ID,
            retained_image_ref=RETAINED_IMAGE_REF,
            workdir_volume=volume_lease,
        ),
        judge=JudgeResourceLease(),
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(original)
    client = FakeDockerClient()
    docker_volume = client.volumes.create(
        authority.name,
        driver="local",
        labels=managed_workdir_volume_labels(authority),
    )
    docker_volume.attrs["Options"] = malformed_options
    docker_image = FakeDockerImage(
        RETAINED_IMAGE_ID,
        attrs={
            "Id": RETAINED_IMAGE_ID,
            "RepoTags": [RETAINED_IMAGE_REF],
            "Config": {
                "Labels": image_state(
                    RETAINED_IMAGE_ID,
                    RETAINED_IMAGE_REF,
                    role="retained-work-rootfs",
                    round_id="final",
                )["labels"]
            },
        },
    )
    client.images.by_ref[RETAINED_IMAGE_ID] = docker_image
    manager = RecoveryManager(
        store=store,
        backend=ProductionRecoveryBackend(
            client, snapshot=object(), firewall=FakeFirewallBackend()
        ),
        managed_root=tmp_path / "runs",
    )

    with pytest.raises(RuntimeError, match="authority remains retained"):
        manager.cleanup("run-1", delete_workspace=True)

    retained = store.read("run-1")
    assert retained is not None and retained.recovery_required is True
    assert retained.work == original.work
    assert docker_volume.removed is False
    assert client.images.removed == []
    assert client.images.by_ref[RETAINED_IMAGE_ID] is docker_image


def test_explicit_cleanup_retries_volume_after_image_was_already_deleted(
    tmp_path,
) -> None:
    expected_volume = workdir_volume()
    base = split_volume_lease(tmp_path, actual=expected_volume, mounted=True)
    original = base.model_copy(
        update={
            "work": base.work.model_copy(
                update={
                    "planned_retained_image_ref": RETAINED_IMAGE_REF,
                    "retained_image_id": RETAINED_IMAGE_ID,
                    "retained_image_ref": RETAINED_IMAGE_REF,
                }
            )
        }
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(original)
    backend = RecoveryBackend()
    backend.images[RETAINED_IMAGE_ID] = image_state(
        RETAINED_IMAGE_ID,
        RETAINED_IMAGE_REF,
        role="retained-work-rootfs",
        round_id="final",
    )
    backend.volumes[expected_volume.name] = volume_state(expected_volume)
    backend.volume_remove_error = OSError("first removal failed")
    manager = RecoveryManager(
        store=store, backend=backend, managed_root=tmp_path / "runs"
    )

    with pytest.raises(RuntimeError, match="authority remains retained"):
        manager.cleanup("run-1", delete_workspace=True)

    partial = store.read("run-1")
    assert partial is not None and partial.recovery_required is True
    assert partial.work.retained_image_id is None
    assert partial.work.workdir_volume.actual == expected_volume
    assert RETAINED_IMAGE_ID not in backend.images
    backend.volume_remove_error = None

    manager.cleanup("run-1", delete_workspace=True)

    completed = store.read("run-1")
    assert completed is not None
    assert completed.recovery_required is False
    assert completed.work.workdir_volume == WorkdirVolumeResourceLease()
