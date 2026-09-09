"""Immutable and runtime guards for Engine-owned container mount topology."""

from __future__ import annotations

from pathlib import PurePosixPath

from rsi_harness.errors import SetupError
from rsi_harness.models import WORK_FEEDBACK_ROOT, RootfsSnapshotMode, RunPlan

ENGINE_MOUNT_TARGETS = (
    PurePosixPath("/tests"),
    PurePosixPath("/logs/verifier"),
    PurePosixPath("/run/rsi-harness/staging"),
    WORK_FEEDBACK_ROOT,
)


def _overlaps(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def validate_split_workdir_target(workdir: PurePosixPath) -> None:
    """Reject a split target that shadows or contains an Engine mount."""
    for engine_target in ENGINE_MOUNT_TARGETS:
        if _overlaps(workdir, engine_target):
            raise SetupError(
                "split WORKDIR mount topology overlaps Engine-owned target "
                f"{engine_target}"
            )


def validate_run_plan_mount_topology(plan: RunPlan) -> None:
    """Validate the complete immutable RunPlan before runtime mutations."""
    if not isinstance(plan, RunPlan):
        raise SetupError("runtime preflight returned an untyped RunPlan")
    if (
        plan.workdir != plan.task.workdir
        or plan.workdir != plan.images.workdir
        or (
            plan.task.service.workdir is not None
            and plan.workdir != plan.task.service.workdir
        )
        or plan.rootfs_snapshot_mode is not plan.images.rootfs_snapshot_mode
    ):
        raise SetupError("RunPlan WORKDIR mount topology authority is inconsistent")
    if plan.rootfs_snapshot_mode is RootfsSnapshotMode.FULL_ROOTFS:
        if plan.workdir != PurePosixPath("/"):
            raise SetupError("full-rootfs mount topology requires root WORKDIR")
        return
    if plan.rootfs_snapshot_mode is not RootfsSnapshotMode.SPLIT_WORKDIR:
        raise SetupError("unsupported rootfs snapshot mode")
    validate_split_workdir_target(plan.workdir)


__all__ = [
    "ENGINE_MOUNT_TARGETS",
    "validate_run_plan_mount_topology",
    "validate_split_workdir_target",
]
