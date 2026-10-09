"""The builder's fixed-size state filesystem (spec 4 B5).

BuildKit keeps its whole state (snapshots, pulled bases, cache mounts, the
staged context) under ``/var/lib/buildkit``, a Docker local volume the
broker creates with exact labels. A RUN step writes about 1.4 GB/s, so the
bound must be a real filesystem size, never a polled soft limit:

* ``loop-ext4`` (production, root): a sparse file
  ``<data_root>/<run>/sb/build/<b16>.img`` of ``disk_mb``, ``mkfs.ext4``,
  ``losetup --find --show``, then a volume of ``type=ext4`` on that device.
  ENOSPC fails the build, never the host.
* ``tmpfs`` (tests only, non-root): a volume of ``type=tmpfs`` sized
  ``disk_mb`` and charged to the builder's memory; the same code path.

Commands run through an injectable runner, so the root-only loop path is
unit-tested with a fake one. Every step with a known outcome is rolled back
in reverse on failure; recovery converges from the lease and the derived
loop file alone (``losetup -j <file>`` finds a device the journal missed).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.runtime.sandbox_env_contracts import (
    BUILDER_VOLUME_ROLE,
    BuilderLease,
    builder_loop_file,
    sandbox_object_labels,
)

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
COMMAND_TIMEOUT_SEC = 120.0
LOOP_DEVICE = re.compile(r"^/dev/loop[0-9]{1,6}$")
RECOVERY_REQUIRED = "recovery_required"

Commit = Callable[[BuilderLease], BuilderLease]


class CommandFailed(Exception):
    """A state-fs command exited non-zero (its outcome is known)."""


class CommandRunner(Protocol):
    def __call__(self, argv: Sequence[str]) -> str: ...


def run_command(argv: Sequence[str], *, timeout: float = COMMAND_TIMEOUT_SEC) -> str:
    """Production runner: no shell, bounded time, stdout on success."""
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        # The command may still act: its outcome is unknown.
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: {argv[0]} did not finish in {timeout:g}s"
        ) from error
    except OSError as error:
        raise CommandFailed(f"{argv[0]} could not run: {error}") from error
    if completed.returncode != 0:
        raise CommandFailed(
            f"{argv[0]} exited {completed.returncode}: {completed.stderr.strip()[:512]}"
        )
    return completed.stdout


def builder_volume_labels(lease: BuilderLease) -> dict[str, str]:
    return sandbox_object_labels(
        lease.owner, BUILDER_VOLUME_ROLE, {"sandbox-builder": lease.builder_id}
    )


def _revise(lease: BuilderLease, **changes: Any) -> BuilderLease:
    return BuilderLease(**{**dict(lease), **changes})


class _StateFs:
    """The volume half both kinds share: create, attest and remove it."""

    kind = ""

    def __init__(self, api: Any) -> None:
        self._api = api

    def options(self, lease: BuilderLease) -> dict[str, str]:
        raise NotImplementedError

    def create(self, lease: BuilderLease, commit: Commit) -> BuilderLease:
        raise NotImplementedError

    def remove(self, lease: BuilderLease, commit: Commit) -> BuilderLease:
        raise NotImplementedError

    def _require(self, lease: BuilderLease) -> None:
        if lease.state_fs != self.kind:
            raise InfrastructureError(
                f"builder {lease.builder_id} state fs is {lease.state_fs}, "
                f"not {self.kind}"
            )

    def _create_volume(self, lease: BuilderLease, options: Mapping[str, str]) -> None:
        name = lease.volume_name
        try:
            self._api.inspect_volume(name)
        except NotFound:
            pass
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot prove builder volume {name} absent: {error}"
            ) from error
        else:
            raise SetupError(f"builder volume {name} already exists")
        try:
            self._api.create_volume(
                name,
                driver="local",
                driver_opts=dict(options),
                labels=builder_volume_labels(lease),
            )
        except APIError as error:
            if error.status_code is not None:
                raise SetupError(
                    f"builder volume {name} create failed: {error}"
                ) from error
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder volume {name} create outcome is "
                f"unknown: {error}"
            ) from error
        except Exception as error:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder volume {name} create outcome is "
                f"unknown: {error}"
            ) from error

    def attest(self, lease: BuilderLease) -> None:
        """Exactly the broker's volume: local, exact options and labels."""
        name = lease.volume_name
        try:
            attrs = self._api.inspect_volume(name)
        except NotFound:
            raise InfrastructureError(f"builder volume {name} is absent") from None
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"cannot inspect builder volume {name}: {error}"
            ) from error
        expected = {
            "Name": name,
            "Driver": "local",
            "Labels": builder_volume_labels(lease),
            "Options": self.options(lease),
            "Scope": "local",
        }
        for key, value in expected.items():
            if attrs.get(key) != value:
                raise InfrastructureError(
                    f"builder volume {name} configuration mismatch: {key}"
                )

    def remove_volume(self, lease: BuilderLease) -> None:
        """Remove and prove absent the planned volume, if it is ours."""
        name = lease.volume_name
        try:
            attrs = self._api.inspect_volume(name)
        except NotFound:
            return
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: cannot inspect builder volume {name}: {error}"
            ) from error
        if attrs.get("Labels") != builder_volume_labels(lease):
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder volume {name} is not owned by its "
                "planned identity"
            )
        failure: Exception | None = None
        try:
            self._api.remove_volume(name, force=False)
        except NotFound:
            return
        except (APIError, OSError) as error:
            failure = error
        try:
            self._api.inspect_volume(name)
        except NotFound:
            return
        except (APIError, OSError) as error:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder volume {name} absence is unproven: "
                f"{error}"
            ) from error
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: builder volume {name} remains after removal"
            + (f": {failure}" if failure is not None else "")
        ) from failure


class TmpfsStateFs(_StateFs):
    """Test-only state: a tmpfs volume, charged to the builder's memory."""

    kind = "tmpfs"

    def options(self, lease: BuilderLease) -> dict[str, str]:
        return expected_volume_options(lease) or {}

    def create(self, lease: BuilderLease, commit: Commit) -> BuilderLease:
        self._require(lease)
        try:
            self._create_volume(lease, self.options(lease))
            self.attest(lease)
        except InfrastructureError as error:
            if str(error).startswith(RECOVERY_REQUIRED):
                raise
            self._roll_back(lease, error)
        except SetupError as error:
            self._roll_back(lease, error)
        return lease

    def _roll_back(self, lease: BuilderLease, error: Exception) -> None:
        try:
            self.remove_volume(lease)
        except Exception as rollback_error:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder volume rollback is unproven: "
                f"{rollback_error}"
            ) from error
        raise SetupError(
            f"builder state fs failed: {error}; its volume is proven absent"
        ) from error

    def remove(self, lease: BuilderLease, commit: Commit) -> BuilderLease:
        self._require(lease)
        self.remove_volume(lease)
        return lease


class LoopExt4StateFs(_StateFs):
    """Production state: an ext4 filesystem on a loop device (root only)."""

    kind = "loop-ext4"

    def __init__(
        self,
        api: Any,
        data_root: Path,
        *,
        runner: CommandRunner = run_command,
    ) -> None:
        super().__init__(api)
        self._data_root = Path(data_root)
        self._run = runner

    def path(self, lease: BuilderLease) -> Path:
        return builder_loop_file(self._data_root, lease.owner.run_id, lease.builder_id)

    def options(self, lease: BuilderLease) -> dict[str, str]:
        if lease.loop_device is None:
            raise InfrastructureError(
                f"builder {lease.builder_id} has no journaled loop device"
            )
        return {"type": "ext4", "device": lease.loop_device}

    def create(self, lease: BuilderLease, commit: Commit) -> BuilderLease:
        """File, mkfs, loop device (journaled), volume; each step rolled back
        in reverse when a later one fails with a known outcome."""
        self._require(lease)
        path = self.path(lease)
        try:
            self._make_file(path, lease.disk_mb)
            self._run(
                [
                    "mkfs.ext4",
                    "-q",
                    "-F",
                    "-m",
                    "0",
                    "-E",
                    "lazy_itable_init=1,lazy_journal_init=1",
                    str(path),
                ]
            )
            device = self._run(["losetup", "--find", "--show", str(path)]).strip()
            if LOOP_DEVICE.fullmatch(device) is None:
                raise CommandFailed(f"losetup returned no loop device: {device[:64]!r}")
            lease = commit(_revise(lease, loop_device=device))
            self._create_volume(lease, self.options(lease))
            self.attest(lease)
        except InfrastructureError as error:
            if str(error).startswith(RECOVERY_REQUIRED):
                raise
            self._roll_back(lease, commit, error)
        except (SetupError, CommandFailed, OSError) as error:
            self._roll_back(lease, commit, error)
        return lease

    @staticmethod
    def _make_file(path: Path, disk_mb: int) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            # Sparse: the size is a bound, not an allocation.
            os.ftruncate(descriptor, disk_mb * MIB)
        finally:
            os.close(descriptor)

    def _roll_back(self, lease: BuilderLease, commit: Commit, error: Exception) -> None:
        try:
            self.remove(lease, commit)
        except Exception as rollback_error:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: builder state fs rollback is unproven: "
                f"{rollback_error}"
            ) from error
        raise SetupError(
            f"builder state fs failed: {error}; its volume, loop device and file "
            "are proven absent"
        ) from error

    def attached(self, path: Path) -> tuple[str, ...]:
        return loop_devices(path, self._run)

    def remove(self, lease: BuilderLease, commit: Commit) -> BuilderLease:
        """Volume, every loop device of the file, then the file; proven."""
        self._require(lease)
        self.remove_volume(lease)
        remove_loop_file(self.path(lease), self._run)
        if lease.loop_device is not None:
            lease = commit(_revise(lease, loop_device=None))
        return lease


def loop_devices(path: Path, runner: CommandRunner) -> tuple[str, ...]:
    """Loop devices backed by ``path`` (``losetup -j``)."""
    try:
        output = runner(["losetup", "-j", str(path)])
    except CommandFailed as error:
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: cannot list loop devices of {path}: {error}"
        ) from error
    devices = []
    for line in output.splitlines():
        device = line.split(":", 1)[0].strip()
        if not device:
            continue
        if LOOP_DEVICE.fullmatch(device) is None:
            raise InfrastructureError(
                f"{RECOVERY_REQUIRED}: unexpected losetup output for {path}"
            )
        devices.append(device)
    return tuple(devices)


def remove_loop_file(path: Path, runner: CommandRunner) -> None:
    """Detach every loop device of a builder file, then remove it; proven.

    Recovery finds a device the journal missed (a crash between losetup and
    its commit) through ``losetup -j``, so the file alone is the authority.
    """
    if path.is_symlink():
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: builder loop file {path} is a symlink"
        )
    if not path.exists():
        return
    for device in loop_devices(path, runner):
        try:
            runner(["losetup", "-d", device])
        except CommandFailed as error:
            LOGGER.warning("detaching %s failed: %s", device, error)
    if loop_devices(path, runner):
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: builder loop file {path} stays attached"
        )
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise InfrastructureError(
            f"{RECOVERY_REQUIRED}: builder loop file {path} removal failed: {error}"
        ) from error


def expected_volume_options(lease: BuilderLease) -> dict[str, str] | None:
    """The exact options of a builder's volume; None when a loop-ext4
    builder has no journaled device (any single /dev/loopN is then its)."""
    if lease.state_fs == "tmpfs":
        return {"type": "tmpfs", "device": "tmpfs", "o": f"size={lease.disk_mb}m"}
    if lease.loop_device is None:
        return None
    return {"type": "ext4", "device": lease.loop_device}


def state_fs(
    kind: str,
    api: Any,
    data_root: Path,
    *,
    runner: CommandRunner = run_command,
) -> TmpfsStateFs | LoopExt4StateFs:
    if kind == "tmpfs":
        return TmpfsStateFs(api)
    if kind == "loop-ext4":
        return LoopExt4StateFs(api, data_root, runner=runner)
    raise ValueError(f"unknown builder state fs {kind}")


__all__ = [
    "CommandFailed",
    "CommandRunner",
    "LoopExt4StateFs",
    "expected_volume_options",
    "loop_devices",
    "remove_loop_file",
    "TmpfsStateFs",
    "builder_volume_labels",
    "run_command",
    "state_fs",
]
