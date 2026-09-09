"""Behaviorally probed native/FUSE OverlayFS snapshot leases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import SnapshotCapabilities, SnapshotLease
from rsi_harness.runtime.durable import durable_mkdir, fsync_directory
from rsi_harness.runtime.recovery import SnapshotRecoveryAuthority

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_OVERLAY_OPTION_DELIMITERS = frozenset({",", ":", "\\", "\n", "\r", "\0"})


def _validate_overlay_paths(*paths: Path) -> None:
    for path in paths:
        if any(
            delimiter in str(path) for delimiter in _OVERLAY_OPTION_DELIMITERS
        ):
            raise SetupError(
                f"overlay path contains unsupported overlay option delimiter: {path}"
            )


class OverlayRunner(Protocol):
    name: str

    def mount(
        self, lower: Path, upper: Path, work: Path, merged: Path
    ) -> int | None: ...

    def unmount(self, merged: Path, process_id: int | None) -> None: ...

    def is_mounted(self, merged: Path) -> bool: ...


class _ProcessCleanupUnproven(InfrastructureError):
    """A started FUSE child remains owned but its reap could not be proven."""

    cleanup_unproven = True

    def __init__(self, message: str, *, process_id: int) -> None:
        super().__init__(message)
        self.process_id = process_id


def _run_checked(
    runner: Callable[..., Any], command: list[str], *, operation: str
) -> Any:
    try:
        return runner(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as error:
        stderr = getattr(error, "stderr", None)
        detail = str(stderr).strip() if stderr else str(error)
        raise InfrastructureError(f"{operation} failed: {detail}") from error


class NativeOverlayRunner:
    """Invoke the kernel OverlayFS mount and unmount commands."""

    name = "overlayfs"

    def __init__(
        self,
        *,
        command_runner: Callable[..., Any] = subprocess.run,
        mount_checker: Callable[[Path], bool] = os.path.ismount,
    ) -> None:
        self._command_runner = command_runner
        self._mount_checker = mount_checker

    def mount(
        self, lower: Path, upper: Path, work: Path, merged: Path
    ) -> None:
        _validate_overlay_paths(lower, upper, work, merged)
        _run_checked(
            self._command_runner,
            [
                "mount",
                "-t",
                "overlay",
                "overlay",
                "-o",
                f"lowerdir={lower},upperdir={upper},workdir={work}",
                str(merged),
            ],
            operation="native OverlayFS mount",
        )
        if not self.is_mounted(merged):
            raise InfrastructureError("native OverlayFS command returned without mount")

    def unmount(self, merged: Path, process_id: int | None) -> None:
        if process_id is not None:
            raise InfrastructureError(
                "native OverlayFS lease has unexpected process ID"
            )
        if not self.is_mounted(merged):
            return
        _run_checked(
            self._command_runner,
            ["umount", str(merged)],
            operation="native OverlayFS unmount",
        )
        if self.is_mounted(merged):
            raise InfrastructureError("native OverlayFS remained mounted after unmount")

    def is_mounted(self, merged: Path) -> bool:
        return bool(self._mount_checker(merged))

class FuseOverlayRunner:
    """Manage a foreground fuse-overlayfs process and fusermount3 release."""

    name = "fuse-overlayfs"

    def __init__(
        self,
        *,
        command_runner: Callable[..., Any] = subprocess.run,
        process_launcher: Callable[..., Any] = subprocess.Popen,
        mount_checker: Callable[[Path], bool] = os.path.ismount,
        sleep: Callable[[float], None] = time.sleep,
        mount_timeout_seconds: float = 5.0,
        pid_alive: Callable[[int], bool] | None = None,
    ) -> None:
        self._command_runner = command_runner
        self._process_launcher = process_launcher
        self._mount_checker = mount_checker
        self._sleep = sleep
        self._mount_timeout_seconds = mount_timeout_seconds
        self._pid_alive = pid_alive or self._default_pid_alive
        self._processes: dict[int, Any] = {}

    def mount(
        self, lower: Path, upper: Path, work: Path, merged: Path
    ) -> int:
        _validate_overlay_paths(lower, upper, work, merged)
        command = [
            "fuse-overlayfs",
            "-o",
            f"lowerdir={lower},upperdir={upper},workdir={work}",
            str(merged),
        ]
        try:
            process = self._process_launcher(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as error:
            raise InfrastructureError(
                f"fuse-overlayfs startup failed: {error}"
            ) from error
        self._processes[process.pid] = process
        deadline = time.monotonic() + self._mount_timeout_seconds
        while not self.is_mounted(merged):
            returncode = process.poll()
            if returncode is not None:
                stderr_stream = getattr(process, "stderr", None)
                detail = stderr_stream.read().strip() if stderr_stream else ""
                process.wait(timeout=self._mount_timeout_seconds)
                self._processes.pop(process.pid, None)
                raise InfrastructureError(
                    "fuse-overlayfs exited before mount readiness"
                    + (f": {detail}" if detail else f" (exit {returncode})")
                )
            if time.monotonic() >= deadline:
                try:
                    self._cleanup_failed_start(merged, process)
                except Exception as cleanup_error:
                    raise _ProcessCleanupUnproven(
                        "fuse-overlayfs startup cleanup could not prove "
                        f"process reap: {cleanup_error}",
                        process_id=int(process.pid),
                    ) from cleanup_error
                raise InfrastructureError("fuse-overlayfs mount readiness timed out")
            self._sleep(0.01)
        return int(process.pid)

    def unmount(self, merged: Path, process_id: int | None) -> None:
        if self.is_mounted(merged):
            _run_checked(
                self._command_runner,
                ["fusermount3", "-u", str(merged)],
                operation="fuse-overlayfs unmount",
            )
        if self.is_mounted(merged):
            raise InfrastructureError("FUSE overlay remained mounted after fusermount3")
        if process_id is None:
            return
        process = self._processes.get(process_id)
        if process is not None:
            try:
                process.wait(timeout=self._mount_timeout_seconds)
            except subprocess.TimeoutExpired:
                self._terminate_kill_reap(process)
            self._processes.pop(process_id, None)
            return
        self._terminate_external_process(process_id)

    def is_mounted(self, merged: Path) -> bool:
        return bool(self._mount_checker(merged))

    def owns_process(self, process_id: int) -> bool:
        """Return whether this live runner can safely reap the launcher PID."""
        return process_id in self._processes

    def _cleanup_failed_start(self, merged: Path, process: Any) -> None:
        if self.is_mounted(merged):
            try:
                _run_checked(
                    self._command_runner,
                    ["fusermount3", "-u", str(merged)],
                    operation="timed-out fuse-overlayfs cleanup",
                )
            except InfrastructureError:
                pass
        try:
            self._terminate_kill_reap(process)
        finally:
            if process.poll() is not None:
                self._processes.pop(process.pid, None)

    def _terminate_kill_reap(self, process: Any) -> None:
        process.terminate()
        try:
            process.wait(timeout=self._mount_timeout_seconds)
            return
        except subprocess.TimeoutExpired:
            process.kill()
        try:
            process.wait(timeout=self._mount_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise InfrastructureError(
                "fuse-overlayfs process could not be killed and reaped"
            ) from error

    def _terminate_external_process(self, process_id: int) -> None:
        if not self._pid_alive(process_id):
            return
        try:
            os.kill(process_id, signal.SIGTERM)
        except ProcessLookupError:
            return
        if self._wait_pid_gone(process_id):
            return
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            return
        if not self._wait_pid_gone(process_id):
            raise InfrastructureError(
                "persisted fuse-overlayfs process remained alive after SIGKILL"
            )

    def _wait_pid_gone(self, process_id: int) -> bool:
        deadline = time.monotonic() + self._mount_timeout_seconds
        while self._pid_alive(process_id):
            if time.monotonic() >= deadline:
                return False
            self._sleep(0.01)
        return True

    @staticmethod
    def _default_pid_alive(process_id: int) -> bool:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


class OverlaySnapshotBackend:
    """Create durable WORKDIR-only COW leases after behavioral capability proof."""

    def __init__(
        self,
        data_root: Path,
        *,
        native_runner: OverlayRunner | None = None,
        fuse_runner: OverlayRunner | None = None,
        sync_runner: Callable[..., Any] = subprocess.run,
        fuse_process_finder: Callable[[Path], tuple[int, ...]] | None = None,
    ) -> None:
        raw_root = Path(data_root)
        _validate_overlay_paths(raw_root)
        if raw_root.is_symlink():
            raise SetupError("configured data root must not be a symlink")
        durable_mkdir(raw_root)
        self._data_root = raw_root.resolve(strict=True)
        self._native = native_runner or NativeOverlayRunner()
        self._fuse = fuse_runner or FuseOverlayRunner()
        self._runners = {
            self._native.name: self._native,
            self._fuse.name: self._fuse,
        }
        self._sync_runner = sync_runner
        self._fuse_process_finder = (
            fuse_process_finder or self._system_fuse_processes
        )
        self._selected_backend: str | None = None
        self._probe_reasons: dict[str, str | None] = {
            self._native.name: None,
            self._fuse.name: None,
        }
        self._lease_manifests: dict[str, Path] = {}

    @property
    def probe_reasons(self) -> dict[str, str | None]:
        return dict(self._probe_reasons)

    def probe(self, workspace_root: Path) -> SnapshotCapabilities:
        """Prove COW and clean release on the workspace filesystem."""
        root = self._validated_path(
            workspace_root, label="workspace root", allow_root=True
        )
        if not root.is_dir():
            raise SetupError("snapshot probe workspace root must be a directory")
        self._selected_backend = None
        self._probe_reasons = {self._native.name: None, self._fuse.name: None}
        for runner in (self._native, self._fuse):
            reason, cleanup_failed = self._behavioral_probe(root, runner)
            self._probe_reasons[runner.name] = reason
            if cleanup_failed:
                return SnapshotCapabilities(
                    backend="unavailable",
                    atomic=False,
                    immutable=False,
                    copy_on_write=False,
                    cleanup=False,
                    reason=reason,
                )
            if reason is None:
                self._selected_backend = runner.name
                fallback_reason = None
                if runner is self._fuse:
                    native_reason = self._probe_reasons[self._native.name]
                    fallback_reason = f"{self._native.name}: {native_reason}"
                return SnapshotCapabilities(
                    backend=runner.name,
                    atomic=True,
                    immutable=True,
                    copy_on_write=True,
                    cleanup=True,
                    reason=fallback_reason,
                )
        combined = "; ".join(
            f"{name}: {reason}" for name, reason in self._probe_reasons.items()
        )
        return SnapshotCapabilities(
            backend="unavailable",
            atomic=False,
            immutable=False,
            copy_on_write=False,
            cleanup=False,
            reason=combined,
        )

    def acquire(
        self, workspace: Path, *, run_id: str, round_id: str
    ) -> SnapshotLease:
        """Sync the paused Work tree, mount a COW view, and persist readiness."""
        self._validate_component(run_id, "run")
        self._validate_component(round_id, "round")
        lower = self._validated_path(workspace, label="workspace")
        if not lower.is_dir():
            raise SetupError("snapshot workspace must be a directory")
        if self._selected_backend is None:
            capabilities = self.probe(lower.parent)
            if not capabilities.copy_on_write:
                raise SetupError(
                    f"no usable overlay snapshot backend: {capabilities.reason}"
                )
        backend_name = self._selected_backend
        if backend_name is None:
            raise SetupError("no behaviorally verified overlay backend selected")
        runner = self._runners[backend_name]
        lease_root = self._snapshot_root(run_id, round_id)
        manifest_path = self.lease_manifest_path(run_id, round_id)
        if lease_root.exists() or manifest_path.exists():
            raise SetupError("snapshot round already has durable state")
        upper = lease_root / "upper"
        work = lease_root / "work"
        merged = lease_root / "merged"
        planned_lease = SnapshotLease(
            lease_id=self._lease_id(run_id, round_id),
            lower_dir=lower,
            upper_dir=upper,
            work_dir=work,
            merged_dir=merged,
            backend=backend_name,
            process_id=None,
        )
        persisted = self._lease_record(
            planned_lease,
            run_id=run_id,
            round_id=round_id,
            state="allocating",
            released=False,
        )
        process_id: int | None = None
        try:
            self._write_json(manifest_path, persisted)
            self._sync_workspace(lower)
            for path in (
                lease_root,
                upper,
                work,
                merged,
            ):
                self._private_mkdir(path)
            process_id = runner.mount(lower, upper, work, merged)
            if not runner.is_mounted(merged):
                raise InfrastructureError(
                    "overlay runner returned before mount readiness"
                )
            lease = SnapshotLease(
                lease_id=planned_lease.lease_id,
                lower_dir=lower,
                upper_dir=upper,
                work_dir=work,
                merged_dir=merged,
                backend=backend_name,
                process_id=process_id,
            )
            persisted = self._lease_record(
                lease,
                run_id=run_id,
                round_id=round_id,
                state="active",
                released=False,
            )
            self._write_json(manifest_path, persisted)
            self._lease_manifests[lease.lease_id] = manifest_path
            return lease
        except Exception as error:
            if isinstance(error, _ProcessCleanupUnproven):
                process_id = error.process_id
            if process_id is not None:
                persisted["lease"]["process_id"] = process_id
            cleanup_error: Exception | None = (
                error if isinstance(error, _ProcessCleanupUnproven) else None
            )
            mounted = self._safe_is_mounted(runner, merged)
            if cleanup_error is None and (mounted or process_id is not None):
                try:
                    runner.unmount(merged, process_id)
                except Exception as cleanup:
                    cleanup_error = cleanup
            if self._safe_is_mounted(runner, merged):
                cleanup_error = cleanup_error or InfrastructureError(
                    "overlay mount remained active after failure cleanup"
                )
            if cleanup_error is None and lease_root.exists():
                try:
                    shutil.rmtree(lease_root)
                    fsync_directory(lease_root.parent)
                except Exception as cleanup:
                    cleanup_error = cleanup
            if cleanup_error is None:
                persisted["state"] = "acquire_failed_clean"
                persisted["released"] = True
                persisted["last_error"] = str(error)
                self._write_json(manifest_path, persisted)
                raise
            persisted["state"] = "recovery_required"
            persisted["released"] = False
            persisted["last_error"] = f"{error}; cleanup failed: {cleanup_error}"
            self._write_json(manifest_path, persisted)
            raise InfrastructureError(
                f"snapshot acquire failed: {error}; cleanup failed: {cleanup_error}"
            ) from error

    def release(
        self,
        lease: SnapshotLease,
        *,
        recovered_process_id: int | None = None,
    ) -> None:
        """Unmount and remove disposable layers before durably marking release."""
        manifest_path, persisted = self._load_lease_manifest(lease)
        if persisted["released"] is True:
            return
        runner = self._runners.get(lease.backend)
        if runner is None:
            raise InfrastructureError(
                f"unknown persisted snapshot backend {lease.backend}"
            )
        mounted = self._safe_is_mounted(runner, lease.merged_dir)
        fuse_processes: tuple[int, ...] = ()
        effective_process_id = lease.process_id
        if lease.backend == self._fuse.name:
            fuse_processes = self._matching_fuse_processes(lease.merged_dir)
            if recovered_process_id is not None and recovered_process_id not in (
                *fuse_processes,
                lease.process_id,
            ):
                raise InfrastructureError(
                    "recovered FUSE process does not match durable authority"
                )
            if len(fuse_processes) == 1:
                discovered_process_id = fuse_processes[0]
                if (
                    lease.process_id is not None
                    and isinstance(runner, FuseOverlayRunner)
                    and runner.owns_process(lease.process_id)
                ):
                    # fuse-overlayfs may daemonize: reap the launcher we own;
                    # the exact-path process is verified gone below.
                    effective_process_id = lease.process_id
                else:
                    effective_process_id = discovered_process_id
            elif lease.process_id is not None:
                # A persisted PID can be reused after restart. Never signal it
                # unless exact command-line discovery still binds it to this
                # merged path.
                effective_process_id = None
            if (
                recovered_process_id is not None
                and fuse_processes
                and recovered_process_id not in fuse_processes
            ):
                raise InfrastructureError(
                    "recovered and discovered FUSE process authority is ambiguous"
                )
        try:
            if mounted or effective_process_id is not None or fuse_processes:
                runner.unmount(lease.merged_dir, effective_process_id)
            if runner.is_mounted(lease.merged_dir):
                raise InfrastructureError(
                    "snapshot mount is still active after release"
                )
            if (
                lease.backend == self._fuse.name
                and self._matching_fuse_processes(lease.merged_dir)
            ):
                raise InfrastructureError(
                    "matching fuse-overlayfs process remains after release"
                )
        except Exception as error:
            persisted["state"] = "recovery_required"
            persisted["released"] = False
            persisted["last_error"] = str(error)
            self._write_json(manifest_path, persisted)
            raise
        lease_root = lease.upper_dir.parent
        expected = {
            lease_root / "upper",
            lease_root / "work",
            lease_root / "merged",
        }
        if {lease.upper_dir, lease.work_dir, lease.merged_dir} != expected:
            raise InfrastructureError(
                "snapshot lease paths do not match managed layout"
            )
        self._validated_path(lease_root, label="snapshot root")
        try:
            for path in (lease.upper_dir, lease.work_dir, lease.merged_dir):
                if path.exists():
                    shutil.rmtree(path)
                    fsync_directory(lease_root)
            if lease_root.exists():
                lease_root.rmdir()
                fsync_directory(lease_root.parent)
        except Exception as error:
            persisted["state"] = "recovery_required"
            persisted["released"] = False
            persisted["last_error"] = str(error)
            self._write_json(manifest_path, persisted)
            raise InfrastructureError(
                f"snapshot layer cleanup failed: {error}"
            ) from error
        persisted["state"] = "released"
        persisted["released"] = True
        persisted["last_error"] = None
        self._write_json(manifest_path, persisted)

    def discover_snapshot_leases(
        self,
        *,
        run_id: str,
        round_id: str | None,
        lease_id: str | None,
    ) -> tuple[SnapshotRecoveryAuthority, ...]:
        """Discover unreleased manifest, mount, process, or layer authority."""
        self._validate_component(run_id, "run")
        if round_id is not None:
            self._validate_component(round_id, "round")
        manifest_root = self._data_root / "snapshot-leases" / run_id
        self._validated_path(manifest_root, label="snapshot manifest root")
        if not manifest_root.exists():
            return ()
        candidates = (
            (manifest_root / f"{round_id}.json",)
            if round_id is not None
            else tuple(sorted(manifest_root.glob("*.json")))
        )
        discovered: list[SnapshotRecoveryAuthority] = []
        for manifest_path in candidates:
            if not manifest_path.exists():
                continue
            try:
                persisted = json.loads(manifest_path.read_text())
                lease = SnapshotLease.model_validate(persisted["lease"])
                persisted_run = str(persisted["run_id"])
                persisted_round = str(persisted["round_id"])
                released = persisted["released"] is True
                state = str(persisted["state"])
            except Exception as error:
                raise InfrastructureError(
                    f"snapshot discovery found invalid manifest: {error}"
                ) from error
            expected_path = self.lease_manifest_path(
                persisted_run, persisted_round
            )
            expected_id = self._lease_id(persisted_run, persisted_round)
            if (
                persisted_run != run_id
                or manifest_path != expected_path
                or lease.lease_id != expected_id
            ):
                raise InfrastructureError(
                    "snapshot discovery found mismatched manifest authority"
                )
            if lease_id is not None and lease.lease_id != lease_id:
                continue
            runner = self._runners.get(lease.backend)
            if runner is None:
                raise InfrastructureError(
                    f"unknown persisted snapshot backend {lease.backend}"
                )
            mounted = self._safe_is_mounted(runner, lease.merged_dir)
            process_id = lease.process_id
            process_alive = False
            if lease.backend == self._fuse.name:
                fuse_processes = self._matching_fuse_processes(lease.merged_dir)
                if process_id is None and len(fuse_processes) == 1:
                    process_id = fuse_processes[0]
                elif fuse_processes:
                    # The exact merged-path process is authoritative after a
                    # daemonizing launcher exits or across a restart.
                    process_id = fuse_processes[0]
                process_alive = process_id in fuse_processes
            layers_present = any(
                path.exists()
                for path in (lease.upper_dir, lease.work_dir, lease.merged_dir)
            )
            if lease.backend != self._fuse.name:
                process_alive = lease.process_id is not None and not released
            requires_recovery = not released or state == "recovery_required"
            if not (
                requires_recovery or mounted or layers_present or process_alive
            ):
                continue
            discovered.append(
                SnapshotRecoveryAuthority(
                    run_id=persisted_run,
                    lease_id=lease.lease_id,
                    round_id=persisted_round,
                    merged_path=lease.merged_dir,
                    process_id=process_id,
                    manifest_requires_recovery=requires_recovery,
                    layers_present=layers_present,
                    process_alive=process_alive,
                )
            )
        return tuple(discovered)

    def release_snapshot(self, authority: SnapshotRecoveryAuthority) -> None:
        """Release one identity after reloading its exact durable manifest."""
        discovered = self.discover_snapshot_leases(
            run_id=authority.run_id,
            round_id=authority.round_id,
            lease_id=authority.lease_id,
        )
        if discovered != (authority,):
            raise InfrastructureError(
                "snapshot recovery authority changed before release"
            )
        manifest_path = self.lease_manifest_path(
            authority.run_id, authority.round_id
        )
        try:
            persisted = json.loads(manifest_path.read_text())
            lease = SnapshotLease.model_validate(persisted["lease"])
        except Exception as error:
            raise InfrastructureError(
                f"snapshot recovery manifest is unreadable: {error}"
            ) from error
        self.release(lease, recovered_process_id=authority.process_id)

    def _matching_fuse_processes(self, merged: Path) -> tuple[int, ...]:
        processes = tuple(dict.fromkeys(self._fuse_process_finder(merged)))
        if len(processes) > 1:
            raise InfrastructureError(
                "ambiguous matching FUSE processes for snapshot mount"
            )
        return processes

    @staticmethod
    def _system_fuse_processes(merged: Path) -> tuple[int, ...]:
        """Find only fuse-overlayfs processes naming this exact merged path."""
        expected = os.fsencode(str(merged))
        matched: list[int] = []
        try:
            entries = tuple(Path("/proc").iterdir())
        except OSError as error:
            raise InfrastructureError(
                f"cannot inspect fuse-overlayfs process authority: {error}"
            ) from error
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                arguments = tuple(
                    argument
                    for argument in (entry / "cmdline").read_bytes().split(b"\0")
                    if argument
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            if not arguments:
                continue
            executable = os.path.basename(os.fsdecode(arguments[0]))
            if executable == "fuse-overlayfs" and expected in arguments[1:]:
                matched.append(int(entry.name))
        return tuple(sorted(matched))


    def lease_manifest_path(self, run_id: str, round_id: str) -> Path:
        self._validate_component(run_id, "run")
        self._validate_component(round_id, "round")
        path = self._data_root / "snapshot-leases" / run_id / f"{round_id}.json"
        self._validated_path(path, label="snapshot lease manifest")
        return path

    def _behavioral_probe(
        self, workspace_root: Path, runner: OverlayRunner
    ) -> tuple[str | None, bool]:
        probe_root: Path | None = None
        merged: Path | None = None
        process_id: int | None = None
        mount_started = False
        release_proven = False
        reason: str | None = None
        cleanup_error: Exception | None = None
        try:
            probe_root = Path(
                tempfile.mkdtemp(prefix=".overlay-probe-", dir=workspace_root)
            )
            probe_root.chmod(0o700)
            fsync_directory(workspace_root)
            fsync_directory(probe_root)
            lower = probe_root / "lower"
            upper = probe_root / "upper"
            work = probe_root / "work"
            merged = probe_root / "merged"
            for path in (lower, upper, work, merged):
                durable_mkdir(path)
            (lower / "baseline.txt").write_text("lower-original")
            process_id = runner.mount(lower, upper, work, merged)
            mount_started = True
            if not runner.is_mounted(merged):
                raise InfrastructureError("mount readiness was not observable")
            if (merged / "baseline.txt").read_text() != "lower-original":
                raise InfrastructureError("merged view did not expose lower data")
            (merged / "upper-only.txt").write_text("upper-write")
            (merged / "baseline.txt").write_text("merged-change")
            if (upper / "upper-only.txt").read_text() != "upper-write":
                raise InfrastructureError("merged write did not land in upper layer")
            if (lower / "upper-only.txt").exists():
                raise InfrastructureError("merged write mutated the lower layer")
            if (lower / "baseline.txt").read_text() != "lower-original":
                raise InfrastructureError("copy-up mutation changed lower data")
            runner.unmount(merged, process_id)
            if runner.is_mounted(merged):
                raise InfrastructureError("release left probe mount active")
            release_proven = True
        except Exception as error:
            reason = str(error)
            if isinstance(error, _ProcessCleanupUnproven):
                process_id = error.process_id
                cleanup_error = error

        if cleanup_error is None and merged is not None and not release_proven:
            if mount_started or self._safe_is_mounted(runner, merged):
                try:
                    runner.unmount(merged, process_id)
                except Exception as cleanup:
                    cleanup_error = cleanup
            if self._safe_is_mounted(runner, merged):
                cleanup_error = cleanup_error or InfrastructureError(
                    "probe mount remained active"
                )
        if cleanup_error is None and probe_root is not None and probe_root.exists():
            try:
                shutil.rmtree(probe_root, ignore_errors=True)
                if probe_root.exists():
                    raise InfrastructureError(
                        f"probe tree remained after cleanup: {probe_root}"
                    )
                fsync_directory(workspace_root)
            except Exception as cleanup:
                cleanup_error = cleanup
        if cleanup_error is not None:
            location = merged or probe_root or workspace_root
            detail = (
                f"{reason}; cleanup failed for {location}: {cleanup_error}"
                if reason is not None
                else f"cleanup failed for {location}: {cleanup_error}"
            )
            self._write_probe_failure(
                runner=runner,
                merged=location,
                process_id=process_id,
                error=detail,
            )
            return detail, True
        return reason, False

    def _load_lease_manifest(
        self, lease: SnapshotLease
    ) -> tuple[Path, dict[str, Any]]:
        manifest_path = self._lease_manifests.get(lease.lease_id)
        if manifest_path is None:
            leases_root = self._data_root / "snapshot-leases"
            matches: list[Path] = []
            if leases_root.exists():
                for candidate in leases_root.glob("*/*.json"):
                    try:
                        raw = json.loads(candidate.read_text())
                        if raw.get("lease", {}).get("lease_id") == lease.lease_id:
                            matches.append(candidate)
                    except (OSError, json.JSONDecodeError, AttributeError):
                        continue
            if len(matches) != 1:
                raise InfrastructureError(
                    "snapshot lease has no unique durable manifest"
                )
            manifest_path = matches[0]
        try:
            persisted = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise InfrastructureError(
                f"snapshot lease manifest is unreadable: {error}"
            ) from error
        if (
            not isinstance(persisted, dict)
            or set(persisted)
            != {
                "schema_version",
                "run_id",
                "round_id",
                "state",
                "released",
                "last_error",
                "lease",
            }
            or persisted["schema_version"] != 1
            or not isinstance(persisted["released"], bool)
            or not isinstance(persisted["run_id"], str)
            or not isinstance(persisted["round_id"], str)
            or not isinstance(persisted["state"], str)
            or (
                persisted["last_error"] is not None
                and not isinstance(persisted["last_error"], str)
            )
        ):
            raise InfrastructureError("snapshot lease manifest schema is invalid")
        try:
            expected_manifest = self.lease_manifest_path(
                persisted["run_id"], persisted["round_id"]
            )
            expected_lease_id = self._lease_id(
                persisted["run_id"], persisted["round_id"]
            )
        except SetupError as error:
            raise InfrastructureError(
                f"snapshot manifest identity is invalid: {error}"
            ) from error
        if manifest_path != expected_manifest or lease.lease_id != expected_lease_id:
            raise InfrastructureError(
                "snapshot manifest identity does not match run/round authority"
            )
        try:
            recorded = SnapshotLease.model_validate(persisted["lease"])
        except Exception as error:
            raise InfrastructureError(
                f"snapshot lease manifest is invalid: {error}"
            ) from error
        if recorded != lease:
            raise InfrastructureError("snapshot lease does not match durable authority")
        for path in (
            lease.lower_dir,
            lease.upper_dir,
            lease.work_dir,
            lease.merged_dir,
            manifest_path,
        ):
            self._validated_path(path, label="persisted snapshot path")
        return manifest_path, persisted

    @staticmethod
    def _lease_id(run_id: str, round_id: str) -> str:
        run = run_id.encode()
        round_value = round_id.encode()
        payload = (
            len(run).to_bytes(8, "big")
            + run
            + len(round_value).to_bytes(8, "big")
            + round_value
        )
        return f"snapshot-{hashlib.sha256(payload).hexdigest()}"

    @staticmethod
    def _lease_record(
        lease: SnapshotLease,
        *,
        run_id: str,
        round_id: str,
        state: str,
        released: bool,
        last_error: str | None = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "round_id": round_id,
            "state": state,
            "released": released,
            "last_error": last_error,
            "lease": lease.model_dump(mode="json"),
        }

    @staticmethod
    def _safe_is_mounted(runner: OverlayRunner, merged: Path) -> bool:
        try:
            return runner.is_mounted(merged)
        except Exception:
            return True

    def _write_probe_failure(
        self,
        *,
        runner: OverlayRunner,
        merged: Path,
        process_id: int | None,
        error: str,
    ) -> None:
        path = (
            self._data_root
            / "snapshot-probe-failures"
            / f"{uuid.uuid4().hex}.json"
        )
        self._write_json(
            path,
            {
                "schema_version": 1,
                "state": "recovery_required",
                "backend": runner.name,
                "merged_dir": str(merged),
                "process_id": process_id,
                "last_error": error,
            },
        )

    def _sync_workspace(self, workspace: Path) -> None:
        _run_checked(
            self._sync_runner,
            ["sync", "-f", str(workspace)],
            operation="paused workspace sync",
        )

    def _snapshot_root(self, run_id: str, round_id: str) -> Path:
        path = self._data_root / "snapshots" / run_id / round_id
        self._validated_path(path, label="snapshot root")
        return path

    @staticmethod
    def _validate_component(value: str, label: str) -> None:
        if _SAFE_COMPONENT.fullmatch(value) is None:
            raise SetupError(f"{label} ID must be one safe path component")

    def _validated_path(
        self, path: Path, *, label: str, allow_root: bool = False
    ) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise SetupError(f"{label} must be absolute")
        _validate_overlay_paths(candidate)
        try:
            relative = candidate.relative_to(self._data_root)
        except ValueError as error:
            raise SetupError(
                f"{label} is outside configured data root: {candidate}"
            ) from error
        current = self._data_root
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise SetupError(f"{label} contains symlink: {current}")
        resolved = candidate.resolve(strict=False)
        if resolved == self._data_root:
            if allow_root:
                return resolved
            raise SetupError(f"{label} must be below configured data root")
        if self._data_root not in resolved.parents:
            raise SetupError(f"{label} is outside configured data root: {candidate}")
        return resolved

    @staticmethod
    def _private_mkdir(path: Path) -> None:
        if path.exists():
            raise FileExistsError(path)
        durable_mkdir(path)

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        durable_mkdir(path.parent)
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp = Path(raw_temp)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            fsync_directory(path.parent)
        finally:
            temp.unlink(missing_ok=True)


__all__ = [
    "FuseOverlayRunner",
    "NativeOverlayRunner",
    "OverlaySnapshotBackend",
]
