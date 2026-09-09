from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.durable import durable_mkdir
from rsi_harness.runtime.snapshot import (
    FuseOverlayRunner,
    NativeOverlayRunner,
    OverlaySnapshotBackend,
)


class RecordingCommandRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any):
        self.calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_native_runner_uses_exact_overlay_arguments_and_clean_unmount(tmp_path):
    commands = RecordingCommandRunner()
    states = iter((True, True, False))
    runner = NativeOverlayRunner(
        command_runner=commands,
        mount_checker=lambda _: next(states),
    )
    lower = tmp_path / "lower"
    upper = tmp_path / "upper"
    work = tmp_path / "work"
    merged = tmp_path / "merged"

    assert runner.mount(lower, upper, work, merged) is None
    runner.unmount(merged, None)

    assert commands.calls == [
        [
            "mount",
            "-t",
            "overlay",
            "overlay",
            "-o",
            f"lowerdir={lower},upperdir={upper},workdir={work}",
            str(merged),
        ],
        ["umount", str(merged)],
    ]


def test_native_and_fuse_reject_overlay_option_delimiters_before_commands(
    tmp_path,
):
    native_commands = RecordingCommandRunner()
    native = NativeOverlayRunner(
        command_runner=native_commands,
        mount_checker=lambda _: False,
    )
    with pytest.raises(SetupError, match="overlay option delimiter"):
        native.mount(
            tmp_path / "lower,extra=bad",
            tmp_path / "upper",
            tmp_path / "work",
            tmp_path / "merged",
        )
    assert native_commands.calls == []

    popen_calls: list[list[str]] = []
    fuse = FuseOverlayRunner(
        process_launcher=lambda command, **kwargs: popen_calls.append(command),
        mount_checker=lambda _: False,
    )
    with pytest.raises(SetupError, match="overlay option delimiter"):
        fuse.mount(
            tmp_path / "lower",
            tmp_path / "upper:bad",
            tmp_path / "work",
            tmp_path / "merged",
        )
    assert popen_calls == []


def test_durable_mkdir_fsyncs_each_new_parent_and_directory(tmp_path, monkeypatch):
    import rsi_harness.runtime.durable as durable_module

    synced: list[Path] = []
    monkeypatch.setattr(durable_module, "fsync_directory", synced.append)
    root = tmp_path / "existing"
    root.mkdir()
    target = root / "one" / "two"

    durable_mkdir(target)

    assert target.is_dir()
    assert synced == [root, root / "one", root / "one", target]


class FakeFuseProcess:
    def __init__(self) -> None:
        self.pid = 4321
        self.wait_calls: list[float] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        self.returncode = 0
        return 0


class StubbornFuseProcess(FakeFuseProcess):
    def __init__(self) -> None:
        super().__init__()
        self.terminate_calls = 0
        self.kill_calls = 0

    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        if self.kill_calls:
            self.returncode = -9
            return -9
        raise subprocess.TimeoutExpired("fuse-overlayfs", timeout)

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1


class UnreapableFuseProcess(StubbornFuseProcess):
    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        raise subprocess.TimeoutExpired("fuse-overlayfs", timeout)


class NonzeroFuseProcess(FakeFuseProcess):
    def wait(self, timeout: float) -> int:
        self.wait_calls.append(timeout)
        self.returncode = 7
        return 7


def test_fuse_runner_waits_for_mount_records_pid_and_uses_fusermount3(tmp_path):
    commands = RecordingCommandRunner()
    process = FakeFuseProcess()
    popen_calls: list[tuple[list[str], dict[str, Any]]] = []
    states = iter((False, False, True, True, False))

    def popen(command: list[str], **kwargs: Any):
        popen_calls.append((command, kwargs))
        return process

    runner = FuseOverlayRunner(
        command_runner=commands,
        process_launcher=popen,
        mount_checker=lambda _: next(states),
        sleep=lambda _: None,
        mount_timeout_seconds=1,
    )
    lower = tmp_path / "lower"
    upper = tmp_path / "upper"
    work = tmp_path / "work"
    merged = tmp_path / "merged"

    pid = runner.mount(lower, upper, work, merged)
    runner.unmount(merged, pid)

    assert pid == 4321
    assert popen_calls == [
        (
            [
                "fuse-overlayfs",
                "-o",
                f"lowerdir={lower},upperdir={upper},workdir={work}",
                str(merged),
            ],
            {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "text": True,
            },
        )
    ]
    assert commands.calls == [["fusermount3", "-u", str(merged)]]
    assert process.wait_calls == [1]


def test_fuse_startup_timeout_terminates_kills_and_reaps_process(tmp_path):
    process = StubbornFuseProcess()
    runner = FuseOverlayRunner(
        command_runner=RecordingCommandRunner(),
        process_launcher=lambda *args, **kwargs: process,
        mount_checker=lambda _: False,
        sleep=lambda _: None,
        mount_timeout_seconds=0,
    )

    with pytest.raises(InfrastructureError, match="readiness timed out"):
        runner.mount(
            tmp_path / "lower",
            tmp_path / "upper",
            tmp_path / "work",
            tmp_path / "merged",
        )

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert len(process.wait_calls) == 2


def test_fuse_startup_timeout_surfaces_owned_pid_when_reap_is_unproven(tmp_path):
    process = UnreapableFuseProcess()
    runner = FuseOverlayRunner(
        command_runner=RecordingCommandRunner(),
        process_launcher=lambda *args, **kwargs: process,
        mount_checker=lambda _: False,
        sleep=lambda _: None,
        mount_timeout_seconds=0,
    )

    with pytest.raises(InfrastructureError) as caught:
        runner.mount(
            tmp_path / "lower",
            tmp_path / "upper",
            tmp_path / "work",
            tmp_path / "merged",
        )

    assert getattr(caught.value, "cleanup_unproven", False) is True
    assert getattr(caught.value, "process_id", None) == process.pid
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert len(process.wait_calls) == 2


@pytest.mark.parametrize("process", [StubbornFuseProcess(), NonzeroFuseProcess()])
def test_fuse_release_always_reaps_after_successful_unmount(tmp_path, process):
    commands = RecordingCommandRunner()
    mounted = True
    runner = FuseOverlayRunner(
        command_runner=commands,
        process_launcher=lambda *args, **kwargs: process,
        mount_checker=lambda _: mounted,
        sleep=lambda _: None,
        mount_timeout_seconds=1,
    )
    merged = tmp_path / "merged"
    pid = runner.mount(
        tmp_path / "lower",
        tmp_path / "upper",
        tmp_path / "work",
        merged,
    )
    mounted = False

    runner.unmount(merged, pid)

    assert commands.calls == []
    assert process.returncode is not None


class BehavioralOverlayRunner:
    def __init__(self, name: str, *, mount_error: str | None = None) -> None:
        self.name = name
        self.mount_error = mount_error
        self.events: list[tuple[str, object]] = []
        self.mounted: set[Path] = set()
        self.fail_unmount = False

    def mount(
        self, lower: Path, upper: Path, work: Path, merged: Path
    ) -> int | None:
        self.events.append(("mount", merged))
        if self.mount_error:
            raise InfrastructureError(self.mount_error)
        shutil.copytree(lower, upper, dirs_exist_ok=True, symlinks=True)
        for source in upper.rglob("*"):
            relative = source.relative_to(upper)
            target = merged / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif source.is_symlink():
                target.symlink_to(source.readlink())
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.hardlink_to(source)
        upper_only = upper / "upper-only.txt"
        upper_only.touch()
        (merged / "upper-only.txt").hardlink_to(upper_only)
        self.mounted.add(merged)
        return 991 if self.name == "fuse-overlayfs" else None

    def unmount(self, merged: Path, process_id: int | None) -> None:
        self.events.append(("unmount", process_id))
        if self.fail_unmount:
            raise InfrastructureError("synthetic unmount failure")
        for child in merged.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        self.mounted.discard(merged)

    def is_mounted(self, merged: Path) -> bool:
        return merged in self.mounted


class PartialMountFailureRunner(BehavioralOverlayRunner):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.fail_after_mount = True

    def mount(
        self, lower: Path, upper: Path, work: Path, merged: Path
    ) -> int | None:
        process_id = super().mount(lower, upper, work, merged)
        if self.fail_after_mount:
            raise InfrastructureError("synthetic failure after mount became live")
        return process_id


class DisappearingProcessRunner(BehavioralOverlayRunner):
    def __init__(self) -> None:
        super().__init__("fuse-overlayfs")
        self.disappear_after_return = False
        self.fail_reap = False
        self.reap_attempts: list[int | None] = []

    def mount(
        self, lower: Path, upper: Path, work: Path, merged: Path
    ) -> int | None:
        process_id = super().mount(lower, upper, work, merged)
        if self.disappear_after_return:
            self.mounted.discard(merged)
            return 8123
        return process_id

    def unmount(self, merged: Path, process_id: int | None) -> None:
        if process_id == 8123:
            self.reap_attempts.append(process_id)
            if self.fail_reap:
                raise InfrastructureError("synthetic process reap unproven")
        super().unmount(merged, process_id)


class FailAfterUnmountOnceRunner(BehavioralOverlayRunner):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.failed = False
        self.fail_next = False

    def unmount(self, merged: Path, process_id: int | None) -> None:
        if self.fail_next and not self.failed:
            super().unmount(merged, process_id)
            self.failed = True
            raise InfrastructureError("process reap interrupted")
        super().unmount(merged, process_id)


def test_probe_behaviorally_prefers_native_and_leaves_lower_unchanged(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace_root.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    fuse = BehavioralOverlayRunner("fuse-overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=fuse,
        sync_runner=RecordingCommandRunner(),
    )

    capabilities = backend.probe(workspace_root)

    assert capabilities.backend == "overlayfs"
    assert capabilities.atomic is True
    assert capabilities.immutable is True
    assert capabilities.copy_on_write is True
    assert capabilities.cleanup is True
    assert capabilities.reason is None
    assert [name for name, _ in native.events] == ["mount", "unmount"]
    assert fuse.events == []
    assert list(workspace_root.iterdir()) == []


def test_probe_falls_back_to_fuse_and_reports_native_reason(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace_root.mkdir(parents=True)
    native = BehavioralOverlayRunner(
        "overlayfs", mount_error="native permission denied"
    )
    fuse = BehavioralOverlayRunner("fuse-overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=fuse,
        sync_runner=RecordingCommandRunner(),
    )

    capabilities = backend.probe(workspace_root)

    assert capabilities.backend == "fuse-overlayfs"
    assert capabilities.copy_on_write is True
    assert capabilities.reason == "overlayfs: native permission denied"
    assert backend.probe_reasons == {
        "overlayfs": "native permission denied",
        "fuse-overlayfs": None,
    }


def test_probe_fails_only_after_native_and_fuse_behavioral_failures(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace_root.mkdir(parents=True)
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native denied"
        ),
        fuse_runner=BehavioralOverlayRunner(
            "fuse-overlayfs", mount_error="fuse missing"
        ),
        sync_runner=RecordingCommandRunner(),
    )

    capabilities = backend.probe(workspace_root)

    assert capabilities.backend == "unavailable"
    assert capabilities.atomic is False
    assert capabilities.immutable is False
    assert capabilities.copy_on_write is False
    assert capabilities.cleanup is False
    assert capabilities.reason == (
        "overlayfs: native denied; fuse-overlayfs: fuse missing"
    )


def test_probe_records_unreapable_fuse_startup_and_does_not_mark_clean(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace_root.mkdir(parents=True)
    process = UnreapableFuseProcess()
    fuse = FuseOverlayRunner(
        command_runner=RecordingCommandRunner(),
        process_launcher=lambda *args, **kwargs: process,
        mount_checker=lambda _: False,
        sleep=lambda _: None,
        mount_timeout_seconds=0,
    )
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native denied"
        ),
        fuse_runner=fuse,
        sync_runner=RecordingCommandRunner(),
    )

    capabilities = backend.probe(workspace_root)

    assert capabilities.backend == "unavailable"
    assert capabilities.cleanup is False
    records = tuple((data_root / "snapshot-probe-failures").glob("*.json"))
    assert len(records) == 1
    persisted = json.loads(records[0].read_text())
    assert persisted["state"] == "recovery_required"
    assert persisted["backend"] == "fuse-overlayfs"
    assert persisted["process_id"] == process.pid
    assert Path(persisted["merged_dir"]).parent.exists()


def test_probe_aborts_fallback_and_durably_records_unknown_live_mount(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace_root.mkdir(parents=True)
    native = PartialMountFailureRunner("overlayfs")
    native.fail_unmount = True
    fuse = BehavioralOverlayRunner("fuse-overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=fuse,
        sync_runner=RecordingCommandRunner(),
    )

    capabilities = backend.probe(workspace_root)

    assert capabilities.backend == "unavailable"
    assert capabilities.cleanup is False
    assert "cleanup" in str(capabilities.reason)
    assert fuse.events == []
    records = tuple((data_root / "snapshot-probe-failures").glob("*.json"))
    assert len(records) == 1
    persisted = json.loads(records[0].read_text())
    assert persisted["state"] == "recovery_required"
    assert persisted["backend"] == "overlayfs"
    assert persisted["merged_dir"] in str(capabilities.reason)


def test_probe_aborts_fallback_and_records_probe_tree_cleanup_failure(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace_root.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    fuse = BehavioralOverlayRunner("fuse-overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=fuse,
        sync_runner=RecordingCommandRunner(),
    )

    def fail_delete(path: Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError(f"cannot delete probe tree {path}")

    monkeypatch.setattr(shutil, "rmtree", fail_delete)

    capabilities = backend.probe(workspace_root)

    assert capabilities.backend == "unavailable"
    assert capabilities.cleanup is False
    assert "cannot delete probe tree" in str(capabilities.reason)
    assert fuse.events == []
    records = tuple((data_root / "snapshot-probe-failures").glob("*.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["state"] == "recovery_required"


def test_acquire_durably_records_partial_mount_when_cleanup_fails(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    native = PartialMountFailureRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    native.fail_after_mount = False
    assert backend.probe(workspace_root).backend == "overlayfs"
    native.fail_after_mount = True
    native.fail_unmount = True

    with pytest.raises(InfrastructureError, match="cleanup.*unmount failure"):
        backend.acquire(workspace, run_id="run-7", round_id="agent-1")

    manifest = json.loads(
        backend.lease_manifest_path("run-7", "agent-1").read_text()
    )
    assert manifest["state"] == "recovery_required"
    assert manifest["released"] is False
    assert "unmount failure" in manifest["last_error"]
    assert native.mounted


@pytest.mark.parametrize(
    ("fail_reap", "expected_state", "expected_released", "root_exists"),
    [
        (True, "recovery_required", False, True),
        (False, "acquire_failed_clean", True, False),
    ],
)
def test_acquire_reaps_returned_fuse_pid_when_mount_disappears(
    tmp_path, fail_reap, expected_state, expected_released, root_exists
):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    fuse = DisappearingProcessRunner()
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native denied"
        ),
        fuse_runner=fuse,
        sync_runner=RecordingCommandRunner(),
    )
    assert backend.probe(workspace_root).backend == "fuse-overlayfs"
    fuse.disappear_after_return = True
    fuse.fail_reap = fail_reap

    with pytest.raises(InfrastructureError):
        backend.acquire(workspace, run_id="run-7", round_id="agent-1")

    assert fuse.reap_attempts == [8123]
    persisted = json.loads(
        backend.lease_manifest_path("run-7", "agent-1").read_text()
    )
    assert persisted["state"] == expected_state
    assert persisted["released"] is expected_released
    assert persisted["lease"]["process_id"] == 8123
    assert (
        data_root / "snapshots" / "run-7" / "agent-1"
    ).exists() is root_exists


def test_acquire_durably_records_layer_deletion_failure(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    assert backend.probe(workspace_root).backend == "overlayfs"
    native.mount_error = "synthetic mount failure"

    def fail_delete(path: Path, *args: Any, **kwargs: Any) -> None:
        raise PermissionError(f"cannot delete {path}")

    monkeypatch.setattr(shutil, "rmtree", fail_delete)

    with pytest.raises(InfrastructureError, match="cleanup.*cannot delete"):
        backend.acquire(workspace, run_id="run-7", round_id="agent-1")

    persisted = json.loads(
        backend.lease_manifest_path("run-7", "agent-1").read_text()
    )
    assert persisted["state"] == "recovery_required"
    assert "cannot delete" in persisted["last_error"]


def test_acquire_syncs_first_persists_ready_lease_and_release_marks_last(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    (workspace / "answer.txt").write_text("work")
    native = BehavioralOverlayRunner("overlayfs")
    commands = RecordingCommandRunner()
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=commands,
    )
    assert backend.probe(workspace_root).backend == "overlayfs"
    native.events.clear()

    lease = backend.acquire(workspace, run_id="run-7", round_id="agent-1")

    assert commands.calls == [["sync", "-f", str(workspace)]]
    assert native.events[0][0] == "mount"
    assert lease.lower_dir == workspace
    assert lease.upper_dir == data_root / "snapshots" / "run-7" / "agent-1" / "upper"
    assert lease.work_dir == data_root / "snapshots" / "run-7" / "agent-1" / "work"
    assert lease.merged_dir == data_root / "snapshots" / "run-7" / "agent-1" / "merged"
    assert lease.backend == "overlayfs"
    assert lease.process_id is None
    manifest_path = backend.lease_manifest_path("run-7", "agent-1")
    persisted = json.loads(manifest_path.read_text())
    assert persisted["released"] is False
    assert persisted["lease"]["lease_id"] == lease.lease_id
    assert persisted["lease"]["merged_dir"] == str(lease.merged_dir)

    backend.release(lease)

    assert native.events[-1] == ("unmount", None)
    persisted = json.loads(manifest_path.read_text())
    assert persisted["released"] is True
    assert not lease.upper_dir.exists()
    assert not lease.work_dir.exists()
    assert not lease.merged_dir.exists()
    assert (workspace / "answer.txt").read_text() == "work"


def test_restart_discovers_and_releases_active_manifest_and_layers(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    assert backend.probe(workspace_root).backend == "overlayfs"
    lease = backend.acquire(workspace, run_id="run-7", round_id="agent-1")
    restarted = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )

    discovered = restarted.discover_snapshot_leases(
        run_id="run-7", round_id="agent-1", lease_id=lease.lease_id
    )
    assert len(discovered) == 1
    assert discovered[0].layers_present is True

    restarted.release_snapshot(discovered[0])

    assert restarted.discover_snapshot_leases(
        run_id="run-7", round_id="agent-1", lease_id=lease.lease_id
    ) == ()
    assert not lease.upper_dir.exists()


def test_restart_releases_planned_only_fuse_manifest_without_pid(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    initial = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native unavailable"
        ),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    assert initial.probe(workspace_root).backend == "fuse-overlayfs"

    def crash_after_planned_manifest(*_args, **_kwargs):
        raise KeyboardInterrupt("simulated process loss")

    initial._sync_runner = crash_after_planned_manifest
    with pytest.raises(KeyboardInterrupt, match="process loss"):
        initial.acquire(workspace, run_id="run-7", round_id="agent-1")

    restarted = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner("overlayfs"),
        fuse_runner=FuseOverlayRunner(
            command_runner=RecordingCommandRunner(),
            mount_checker=lambda _path: False,
            pid_alive=lambda _pid: False,
        ),
        sync_runner=RecordingCommandRunner(),
    )
    discovered = restarted.discover_snapshot_leases(
        run_id="run-7", round_id="agent-1", lease_id=None
    )

    assert len(discovered) == 1
    assert discovered[0].process_id is None
    assert discovered[0].manifest_requires_recovery is True
    restarted.release_snapshot(discovered[0])

    assert restarted.discover_snapshot_leases(
        run_id="run-7", round_id="agent-1", lease_id=None
    ) == ()


def test_restart_releases_active_fuse_manifest_with_recorded_process(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    fuse = BehavioralOverlayRunner("fuse-overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native unavailable"
        ),
        fuse_runner=fuse,
        sync_runner=RecordingCommandRunner(),
    )
    assert backend.probe(workspace_root).backend == "fuse-overlayfs"
    lease = backend.acquire(workspace, run_id="run-7", round_id="agent-1")
    active_processes = {991}

    class RestartFuseRunner(BehavioralOverlayRunner):
        def unmount(self, merged, process_id):
            self.events.append(("unmount", process_id))
            self.mounted.discard(merged)
            active_processes.discard(process_id)

    restarted_fuse = RestartFuseRunner("fuse-overlayfs")
    restarted_fuse.mounted.add(lease.merged_dir)
    restarted = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner("overlayfs"),
        fuse_runner=restarted_fuse,
        sync_runner=RecordingCommandRunner(),
        fuse_process_finder=lambda _merged: tuple(active_processes),
    )

    authority = restarted.discover_snapshot_leases(
        run_id="run-7", round_id="agent-1", lease_id=lease.lease_id
    )[0]
    restarted.release_snapshot(authority)

    assert restarted_fuse.events[-1] == ("unmount", 991)
    assert restarted.discover_snapshot_leases(
        run_id="run-7", round_id="agent-1", lease_id=lease.lease_id
    ) == ()


def test_restart_discovers_unrecorded_active_fuse_process(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    initial = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native unavailable"
        ),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    assert initial.probe(workspace_root).backend == "fuse-overlayfs"

    def crash_after_planned_manifest(*_args, **_kwargs):
        raise KeyboardInterrupt("simulated process loss")

    initial._sync_runner = crash_after_planned_manifest
    with pytest.raises(KeyboardInterrupt):
        initial.acquire(workspace, run_id="run-7", round_id="agent-1")

    active_processes = {777}

    class RestartFuseRunner(BehavioralOverlayRunner):
        def unmount(self, merged, process_id):
            self.events.append(("unmount", process_id))
            self.mounted.discard(merged)
            active_processes.discard(process_id)

    restarted_fuse = RestartFuseRunner("fuse-overlayfs")
    restarted = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner("overlayfs"),
        fuse_runner=restarted_fuse,
        sync_runner=RecordingCommandRunner(),
        fuse_process_finder=lambda _merged: tuple(active_processes),
    )

    authority = restarted.discover_snapshot_leases(
        run_id="run-7", round_id="agent-1", lease_id=None
    )[0]

    assert authority.process_id == 777
    assert authority.process_alive is True
    restarted.release_snapshot(authority)
    assert restarted_fuse.events[-1] == ("unmount", 777)
    assert active_processes == set()


def test_restart_fails_closed_for_ambiguous_unrecorded_fuse_processes(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    initial = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native unavailable"
        ),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    assert initial.probe(workspace_root).backend == "fuse-overlayfs"

    def crash_after_planned_manifest(*_args, **_kwargs):
        raise KeyboardInterrupt("simulated process loss")

    initial._sync_runner = crash_after_planned_manifest
    with pytest.raises(KeyboardInterrupt):
        initial.acquire(workspace, run_id="run-7", round_id="agent-1")
    restarted = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner("overlayfs"),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
        fuse_process_finder=lambda _merged: (777, 778),
    )

    with pytest.raises(InfrastructureError, match="ambiguous.*FUSE processes"):
        restarted.discover_snapshot_leases(
            run_id="run-7", round_id="agent-1", lease_id=None
        )


def test_concrete_snapshot_discovery_exposes_ambiguous_fuse_manifests(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner(
            "overlayfs", mount_error="native unavailable"
        ),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    assert backend.probe(workspace_root).backend == "fuse-overlayfs"

    def crash_after_planned_manifest(*_args, **_kwargs):
        raise KeyboardInterrupt("simulated process loss")

    backend._sync_runner = crash_after_planned_manifest
    for round_id in ("agent-1", "agent-2"):
        with pytest.raises(KeyboardInterrupt):
            backend.acquire(workspace, run_id="run-7", round_id=round_id)

    discovered = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner("overlayfs"),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    ).discover_snapshot_leases(run_id="run-7", round_id=None, lease_id=None)

    assert [authority.round_id for authority in discovered] == [
        "agent-1",
        "agent-2",
    ]


def test_release_fsyncs_layer_root_and_surviving_snapshot_parent(
    tmp_path, monkeypatch
):
    import rsi_harness.runtime.snapshot as snapshot_module

    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    backend.probe(workspace_root)
    lease = backend.acquire(workspace, run_id="run-7", round_id="agent-1")
    synced: list[Path] = []
    monkeypatch.setattr(snapshot_module, "fsync_directory", synced.append)

    backend.release(lease)

    assert lease.upper_dir.parent in synced
    assert lease.upper_dir.parent.parent in synced


def test_failed_unmount_keeps_lease_unreleased_and_write_layer_for_recovery(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    backend.probe(workspace_root)
    lease = backend.acquire(workspace, run_id="run-7", round_id="agent-1")
    native.fail_unmount = True

    with pytest.raises(InfrastructureError, match="unmount failure"):
        backend.release(lease)

    persisted = json.loads(
        backend.lease_manifest_path("run-7", "agent-1").read_text()
    )
    assert persisted["released"] is False
    assert lease.upper_dir.exists()


def test_release_retry_continues_cleanup_when_mount_is_already_absent(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    native = FailAfterUnmountOnceRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    backend.probe(workspace_root)
    lease = backend.acquire(workspace, run_id="run-7", round_id="agent-1")
    native.fail_next = True

    with pytest.raises(InfrastructureError, match="reap interrupted"):
        backend.release(lease)

    manifest_path = backend.lease_manifest_path("run-7", "agent-1")
    failed = json.loads(manifest_path.read_text())
    assert failed["state"] == "recovery_required"
    assert not native.is_mounted(lease.merged_dir)

    backend.release(lease)

    released = json.loads(manifest_path.read_text())
    assert released["state"] == "released"
    assert released["released"] is True
    assert not lease.upper_dir.exists()


@pytest.mark.parametrize(
    ("run_id", "round_id"),
    [
        ("../run", "agent-1"),
        ("nested/run", "agent-1"),
        ("nested\\run", "agent-1"),
        ("run-7", "../agent-1"),
        ("run-7", "nested/round"),
        ("run-7", "nested\\round"),
        (".", "agent-1"),
        ("run-7", ""),
    ],
)
def test_acquire_rejects_run_or_round_identifiers_with_path_syntax(
    tmp_path, run_id, round_id
):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "run-7"
    workspace.mkdir(parents=True)
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner("overlayfs"),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    backend.probe(workspace_root)

    with pytest.raises(SetupError, match="safe path component"):
        backend.acquire(workspace, run_id=run_id, round_id=round_id)


@pytest.mark.parametrize("delimiter", (",", ":", "\\", "\n"))
def test_backend_rejects_workspace_overlay_option_delimiters(
    tmp_path, delimiter
):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / f"bad{delimiter}lower"
    workspace.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    commands = RecordingCommandRunner()
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=commands,
    )

    with pytest.raises(SetupError, match="overlay option delimiter"):
        backend.acquire(workspace, run_id="run-7", round_id="agent-1")

    assert native.events == []
    assert commands.calls == []


def test_lease_identity_is_unambiguous_for_hyphenated_run_round_pairs(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "shared"
    workspace.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    backend.probe(workspace_root)

    first = backend.acquire(workspace, run_id="a-b", round_id="c")
    second = backend.acquire(workspace, run_id="a", round_id="b-c")

    assert first.lease_id != second.lease_id
    backend.release(first)
    backend.release(second)


def test_release_rejects_manifest_run_round_identity_tampering(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace = workspace_root / "shared"
    workspace.mkdir(parents=True)
    native = BehavioralOverlayRunner("overlayfs")
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=native,
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    backend.probe(workspace_root)
    lease = backend.acquire(workspace, run_id="run-7", round_id="agent-1")
    manifest_path = backend.lease_manifest_path("run-7", "agent-1")
    persisted = json.loads(manifest_path.read_text())
    persisted["run_id"] = "forged-run"
    manifest_path.write_text(json.dumps(persisted))

    with pytest.raises(InfrastructureError, match="manifest identity"):
        backend.release(lease)

    assert native.is_mounted(lease.merged_dir)


def test_workspace_outside_data_root_or_through_symlink_is_rejected(tmp_path):
    data_root = tmp_path / "data"
    workspace_root = data_root / "workspaces"
    workspace_root.mkdir(parents=True)
    backend = OverlaySnapshotBackend(
        data_root,
        native_runner=BehavioralOverlayRunner("overlayfs"),
        fuse_runner=BehavioralOverlayRunner("fuse-overlayfs"),
        sync_runner=RecordingCommandRunner(),
    )
    backend.probe(workspace_root)
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(SetupError, match="outside.*data root"):
        backend.acquire(outside, run_id="run-7", round_id="agent-1")

    link = workspace_root / "linked"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(SetupError, match="symlink"):
        backend.acquire(link, run_id="run-7", round_id="agent-1")
