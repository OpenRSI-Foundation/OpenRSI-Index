"""One-time initialization and explicit cleanup of persistent run workspaces."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from rsi_harness.errors import SetupError
from rsi_harness.models import RunPlan
from rsi_harness.runtime.durable import durable_mkdir, fsync_directory
from rsi_harness.runtime.gpu import (
    NVIDIA_VISIBLE_DEVICES_ENV,
    NVIDIA_VISIBLE_DEVICES_VOID,
)
from rsi_harness.runtime.image_authority import is_immutable_image_ref

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
# CHOWN preserves numeric owners, DAC_OVERRIDE accesses the private bind target,
# and FOWNER preserves source metadata; the helper receives no other capabilities.
_HELPER_CAPABILITIES = ("CHOWN", "DAC_OVERRIDE", "FOWNER")
_POSIX_ACL_XATTRS = (
    "system.posix_acl_access",
    "system.posix_acl_default",
)
_MISSING_XATTR_ERRNOS = {errno.ENODATA}
if hasattr(errno, "ENOATTR"):
    _MISSING_XATTR_ERRNOS.add(errno.ENOATTR)
_INITIALIZE_SCRIPT = """\
set -eu
if [ -L "$RSI_SOURCE_WORKDIR" ]; then
    echo "image WORKDIR is a symlink" >&2
    exit 86
fi
if [ ! -d "$RSI_SOURCE_WORKDIR" ]; then
    echo "image WORKDIR is not a directory" >&2
    exit 86
fi
uid=$(stat -c %u -- "$RSI_SOURCE_WORKDIR")
gid=$(stat -c %g -- "$RSI_SOURCE_WORKDIR")
cp -a --reflink=auto "$RSI_SOURCE_WORKDIR/." "$RSI_INIT_TARGET/"
chown "$RSI_ENGINE_UID:$RSI_ENGINE_GID" "$RSI_INIT_TARGET"
# The run parent remains Engine-owned 0700.  This bind root must be traversable
# by independently configured Work and Judge identities. Work can revoke the
# mode when it is root or shares the Engine UID, so every paused Judge handoff
# restores and re-attests these exact root attributes.
chmod 0777 "$RSI_INIT_TARGET"
printf '%s:%s\n' "$uid" "$gid"
"""
_JUDGE_HANDOFF_SCRIPT = """\
set -eu
if [ -L "$RSI_INIT_TARGET" ]; then
    echo "managed workspace is a symlink" >&2
    exit 87
fi
if [ ! -d "$RSI_INIT_TARGET" ]; then
    echo "managed workspace is not a directory" >&2
    exit 87
fi
chown "$RSI_ENGINE_UID:$RSI_ENGINE_GID" "$RSI_INIT_TARGET"
chmod 0777 "$RSI_INIT_TARGET"
test "$(stat -c %u -- "$RSI_INIT_TARGET")" = "$RSI_ENGINE_UID"
test "$(stat -c %g -- "$RSI_INIT_TARGET")" = "$RSI_ENGINE_GID"
test "$(stat -c %a -- "$RSI_INIT_TARGET")" = "777"
"""


def delete_managed_workspace(
    client: Any,
    *,
    workspace: Path,
    image_ref: str,
    run_id: str,
    task_id: str,
) -> None:
    """Delete one already-validated workspace with bounded container authority."""

    workspace = Path(workspace)
    if not workspace.is_absolute() or workspace.is_symlink():
        raise SetupError("managed workspace cleanup target is invalid")
    if _SAFE_ID.fullmatch(run_id) is None or _SAFE_ID.fullmatch(task_id) is None:
        raise SetupError("managed workspace cleanup labels are invalid")
    if not is_immutable_image_ref(image_ref):
        raise SetupError("managed workspace cleanup image authority is invalid")
    digest = hashlib.sha256(run_id.encode()).hexdigest()[:12]
    helper_target = f"/run/rsi-harness-cleanup-{digest}"
    if workspace.exists():
        try:
            client.containers.run(
                image_ref,
                ['find "$RSI_INIT_TARGET" -mindepth 1 -delete'],
                entrypoint=["/bin/sh", "-c"],
                user="root",
                working_dir="/",
                environment={
                    NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID,
                    "RSI_INIT_TARGET": helper_target,
                },
                volumes={
                    str(workspace): {
                        "bind": helper_target,
                        "mode": "rw",
                    }
                },
                network_mode="none",
                privileged=False,
                cap_drop=["ALL"],
                cap_add=list(_HELPER_CAPABILITIES),
                labels={
                    "rsi-harness.run-id": run_id,
                    "rsi-harness.task-id": task_id,
                    "rsi-harness.role": "helper",
                },
                remove=True,
            )
        except Exception as error:
            raise SetupError(
                f"managed workspace cleanup helper failed: {error}"
            ) from error
        workspace.rmdir()
    workspace.parent.joinpath("workspace-initialization.json").unlink(
        missing_ok=True
    )
    fsync_directory(workspace.parent)


class WorkspaceManager:
    """Own a persistent WORKDIR copied exactly once from the clean Base image."""

    def __init__(
        self,
        client: Any,
        *,
        plan: RunPlan,
        run_id: str,
        data_root: Path,
    ) -> None:
        if _SAFE_ID.fullmatch(run_id) is None:
            raise SetupError("run ID must be one safe path component")
        self._client = client
        self._plan = plan
        self._run_id = run_id
        self._data_root = Path(data_root).resolve()
        self._run_root = Path(plan.paths.root)
        self._workspace = Path(plan.paths.workspace)
        self._helper_target = self._select_helper_target(plan.workdir, run_id)
        self._require_contained(self._run_root, "run root")
        self._require_contained(self._workspace, "workspace")
        self._manifest_path = self._run_root / "workspace-initialization.json"
        self._lock_path = self._run_root / ".workspace-initialization.lock"
        self._require_contained(self._manifest_path, "workspace manifest")
        self._require_contained(self._lock_path, "workspace lock")
        digest = plan.images.base_digest
        if not digest:
            raise SetupError(
                "workspace initialization requires a fixed Base image digest"
            )

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def initialize(self) -> Path:
        """Copy Base WORKDIR once, or attest the durable completed manifest."""
        self._prepare_private_directory(self._data_root)
        self._prepare_private_directory(self._run_root)
        self._reject_symlink_components(self._workspace)
        self._lock_path.touch(mode=0o600, exist_ok=True)
        with self._lock_path.open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self._manifest_path.exists():
                self._verify_manifest()
                return self._workspace
            if self._workspace.exists() and any(self._workspace.iterdir()):
                raise SetupError(
                    "workspace has content without an initialization manifest; "
                    "explicit cleanup is required"
                )
            self._prepare_private_directory(self._workspace)
            self._write_manifest(
                {
                    "schema_version": 1,
                    "image_digest": self._plan.images.base_digest,
                    "workdir": str(self._plan.workdir),
                    "uid": None,
                    "gid": None,
                    "completed": False,
                }
            )
            try:
                output = self._run_initialization_helper()
                uid, gid = self._parse_identity(output)
            except Exception as error:
                if isinstance(error, SetupError):
                    raise
                raise SetupError(
                    f"Base image WORKDIR initialization failed: {error}"
                ) from error
            self._write_manifest(
                {
                    "schema_version": 1,
                    "image_digest": self._plan.images.base_digest,
                    "workdir": str(self._plan.workdir),
                    "uid": uid,
                    "gid": gid,
                    "completed": True,
                }
            )
            return self._workspace

    def retain(self) -> Path:
        """Retain the initialized workspace, which is the default run policy."""
        self._verify_manifest()
        return self._workspace

    def attest_for_judge(self) -> None:
        """Restore and prove the paused Work bind root for a distinct Judge."""
        self._verify_manifest()
        try:
            self._client.containers.run(
                self._plan.images.base_ref,
                [_JUDGE_HANDOFF_SCRIPT],
                entrypoint=["/bin/sh", "-c"],
                user="root",
                working_dir="/",
                environment={
                    NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID,
                    "RSI_INIT_TARGET": self._helper_target,
                    "RSI_ENGINE_UID": str(os.getuid()),
                    "RSI_ENGINE_GID": str(os.getgid()),
                },
                volumes={
                    str(self._workspace): {
                        "bind": self._helper_target,
                        "mode": "rw",
                    }
                },
                network_mode="none",
                privileged=False,
                cap_drop=["ALL"],
                cap_add=list(_HELPER_CAPABILITIES),
                labels={
                    "rsi-harness.run-id": self._run_id,
                    "rsi-harness.task-id": self._plan.task.task_id,
                    "rsi-harness.role": "helper",
                },
                remove=True,
            )
        except BaseException as error:
            raise SetupError(
                f"managed workspace Judge handoff helper failed: {error}"
            ) from error
        self._clear_posix_acl_authority()
        self._attest_judge_handoff_root()

    def delete(self) -> None:
        """Delete only this validated managed workspace on an explicit request."""
        self._require_contained(self._workspace, "workspace")
        self._reject_symlink_components(self._workspace)
        delete_managed_workspace(
            self._client,
            workspace=self._workspace,
            image_ref=self._plan.images.base_ref,
            run_id=self._run_id,
            task_id=self._plan.task.task_id,
        )

    def _run_initialization_helper(self) -> bytes:
        try:
            return self._client.containers.run(
                self._plan.images.base_ref,
                [_INITIALIZE_SCRIPT],
                entrypoint=["/bin/sh", "-c"],
                user="root",
                working_dir="/",
                environment={
                    NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID,
                    "RSI_SOURCE_WORKDIR": str(self._plan.workdir),
                    "RSI_INIT_TARGET": self._helper_target,
                    "RSI_ENGINE_UID": str(os.getuid()),
                    "RSI_ENGINE_GID": str(os.getgid()),
                },
                volumes={
                    str(self._workspace): {
                        "bind": self._helper_target,
                        "mode": "rw",
                    }
                },
                network_mode="none",
                privileged=False,
                cap_drop=["ALL"],
                cap_add=list(_HELPER_CAPABILITIES),
                labels={
                    "rsi-harness.run-id": self._run_id,
                    "rsi-harness.task-id": self._plan.task.task_id,
                    "rsi-harness.role": "helper",
                },
                remove=True,
            )
        except Exception as error:
            raise SetupError(
                f"Base image WORKDIR helper rejected or failed: {error}"
            ) from error

    def _verify_manifest(self) -> None:
        try:
            raw = json.loads(self._manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SetupError(f"workspace manifest is unreadable: {error}") from error
        expected_keys = {
            "schema_version",
            "image_digest",
            "workdir",
            "uid",
            "gid",
            "completed",
        }
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise SetupError("workspace manifest has an invalid exact schema")
        if raw["schema_version"] != 1:
            raise SetupError("workspace manifest schema is unsupported")
        if raw["image_digest"] != self._plan.images.base_digest:
            raise SetupError(
                "workspace manifest image digest does not match fixed Base"
            )
        if raw["workdir"] != str(self._plan.workdir):
            raise SetupError("workspace manifest WORKDIR does not match RunPlan")
        if raw["completed"] is not True:
            raise SetupError(
                "workspace initialization is incomplete; explicit cleanup is required"
            )
        if (
            not isinstance(raw["uid"], int)
            or isinstance(raw["uid"], bool)
            or raw["uid"] < 0
            or not isinstance(raw["gid"], int)
            or isinstance(raw["gid"], bool)
            or raw["gid"] < 0
        ):
            raise SetupError("workspace manifest UID/GID proof is invalid")
        self._reject_symlink_components(self._workspace)
        if not self._workspace.is_dir():
            raise SetupError("completed workspace manifest has no workspace directory")

    def _attest_judge_handoff_root(self) -> None:
        self._reject_symlink_components(self._workspace)
        try:
            metadata = self._workspace.lstat()
        except OSError as error:
            raise SetupError(
                f"managed workspace Judge handoff is unreadable: {error}"
            ) from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise SetupError("managed workspace Judge handoff is not a directory")
        if metadata.st_uid != os.getuid() or metadata.st_gid != os.getgid():
            raise SetupError("managed workspace Judge handoff owner is unproven")
        if stat.S_IMODE(metadata.st_mode) != 0o777:
            raise SetupError("managed workspace Judge handoff mode is unproven")
        for name in _POSIX_ACL_XATTRS:
            try:
                os.getxattr(self._workspace, name, follow_symlinks=False)
            except OSError as error:
                if error.errno in _MISSING_XATTR_ERRNOS:
                    continue
                raise SetupError(
                    "managed workspace Judge handoff POSIX ACL absence is "
                    f"unproven: {error}"
                ) from error
            raise SetupError(
                "managed workspace Judge handoff retains POSIX ACL authority"
            )

    def _clear_posix_acl_authority(self) -> None:
        for name in _POSIX_ACL_XATTRS:
            try:
                os.removexattr(self._workspace, name, follow_symlinks=False)
            except OSError as error:
                if error.errno in _MISSING_XATTR_ERRNOS:
                    continue
                raise SetupError(
                    "managed workspace Judge handoff POSIX ACL removal is "
                    f"unproven: {error}"
                ) from error

    @staticmethod
    def _parse_identity(output: bytes | str) -> tuple[int, int]:
        text = output.decode() if isinstance(output, bytes) else str(output)
        fields = text.strip().split(":")
        if len(fields) != 2 or not all(field.isdecimal() for field in fields):
            raise SetupError("WORKDIR helper returned no exact numeric UID/GID proof")
        return int(fields[0]), int(fields[1])

    @staticmethod
    def _select_helper_target(workdir: PurePosixPath, run_id: str) -> str:
        for nonce in range(100):
            material = f"{len(run_id)}:{run_id}:{len(str(workdir))}:{workdir}:{nonce}"
            suffix = hashlib.sha256(material.encode()).hexdigest()[:16]
            candidate = PurePosixPath(f"/.rsi-init-{suffix}")
            if (
                candidate != workdir
                and candidate not in workdir.parents
                and workdir not in candidate.parents
            ):
                return str(candidate)
        raise SetupError("could not select a helper target disjoint from WORKDIR")

    def _write_manifest(self, value: dict[str, object]) -> None:
        self._prepare_private_directory(self._manifest_path.parent)
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{self._manifest_path.name}.",
            suffix=".tmp",
            dir=self._manifest_path.parent,
        )
        temp = Path(raw_temp)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(value, stream, sort_keys=True, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, self._manifest_path)
            fsync_directory(self._manifest_path.parent)
        finally:
            temp.unlink(missing_ok=True)

    def _require_contained(self, path: Path, label: str) -> None:
        if not path.is_absolute():
            raise SetupError(f"{label} must be absolute")
        resolved = path.resolve(strict=False)
        if resolved == self._data_root or self._data_root not in resolved.parents:
            raise SetupError(f"{label} is outside configured data root: {path}")

    def _reject_symlink_components(self, path: Path) -> None:
        current = self._data_root
        try:
            relative = path.relative_to(self._data_root)
        except ValueError as error:
            raise SetupError(f"path is outside configured data root: {path}") from error
        for component in relative.parts:
            current /= component
            if current.is_symlink():
                raise SetupError(f"managed workspace path contains symlink: {current}")

    @staticmethod
    def _prepare_private_directory(path: Path) -> None:
        durable_mkdir(path)


__all__ = ["WorkspaceManager"]
