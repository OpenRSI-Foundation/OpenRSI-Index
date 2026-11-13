"""Attested Docker local volumes used solely for a task WORKDIR."""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from docker.errors import NotFound
from docker.types import Mount
from pydantic import ValidationError

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    ContainerRef,
    ManagedWorkdirVolume,
    RootfsSnapshotMode,
)
from rsi_harness.runtime.gpu import (
    NVIDIA_VISIBLE_DEVICES_ENV,
    NVIDIA_VISIBLE_DEVICES_VOID,
)
from rsi_harness.runtime.image_authority import is_immutable_image_ref
from rsi_harness.runtime.special_files import SPECIAL_FILE_LIMIT

_ROLE = "workdir-volume"
_PREFIX = "rsi-harness-workdir-"
_RUN_LABEL = "rsi-harness.run-id"
_TASK_LABEL = "rsi-harness.task-id"
_ROLE_LABEL = "rsi-harness.role"
_SNAPSHOT_MODE_LABEL = "rsi-harness.snapshot-mode"
_TARGET_LABEL = "rsi-harness.workdir-target"
_FRESHNESS_LABEL = "rsi-harness.freshness-nonce"
_SCAN_ROOT = "/run/rsi-harness-scan"
# Print at most RSI_SCAN_LIMIT FIFOs, sockets and devices; a walk that fails
# (an unreadable directory, no find) prints RSI_SCAN_FAILED.
_SCAN_SCRIPT = """\
{ find "$RSI_SCAN_ROOT" -xdev \\( -type p -o -type s -o -type c -o -type b \\) \\
    -print || echo "$RSI_SCAN_FAILED"; } | head -n "$RSI_SCAN_LIMIT"
"""


def _digest(*values: str) -> str:
    payload = b"".join(
        len(value.encode("utf-8")).to_bytes(8, "big") + value.encode("utf-8")
        for value in values
    )
    return hashlib.sha256(payload).hexdigest()


def managed_workdir_volume_labels(
    volume: ManagedWorkdirVolume,
) -> dict[str, str]:
    """Return the one exact Docker label identity for a volume authority."""
    return {
        _RUN_LABEL: volume.run_id,
        _TASK_LABEL: volume.task_id,
        _ROLE_LABEL: _ROLE,
        _SNAPSHOT_MODE_LABEL: volume.snapshot_mode.value,
        _TARGET_LABEL: volume.target.as_posix(),
        _FRESHNESS_LABEL: volume.freshness_nonce,
    }


def attest_normalized_workdir_volume_state(
    state: Mapping[str, Any],
    expected: ManagedWorkdirVolume,
    *,
    expected_references: Sequence[ContainerRef] = (),
) -> None:
    """Attest a normalized volume state shared by runtime and recovery."""
    if not isinstance(state, Mapping):
        raise InfrastructureError(
            "recovery_required: managed WORKDIR volume state is malformed"
        )
    labels = state.get("labels")
    options = state.get("options")
    references = state.get("container_references")
    if (
        state.get("name") != expected.name
        or state.get("driver") != "local"
        or labels != managed_workdir_volume_labels(expected)
        or not isinstance(options, Mapping)
        or bool(options)
        or state.get("scope") != "local"
        or not isinstance(references, (tuple, list))
    ):
        raise InfrastructureError(
            "recovery_required: managed WORKDIR volume does not attest exact authority"
        )
    _attest_normalized_references(
        references,
        expected,
        expected_references=expected_references,
    )


def _attest_normalized_references(
    references: Sequence[Any],
    volume: ManagedWorkdirVolume,
    *,
    expected_references: Sequence[ContainerRef],
) -> None:
    expected_by_id = {
        reference.container_id: reference for reference in expected_references
    }
    if len(expected_by_id) != len(expected_references):
        raise InfrastructureError(
            "recovery_required: expected managed WORKDIR volume references "
            "are duplicated"
        )
    found_by_id: dict[str, Mapping[str, Any]] = {}
    for reference in references:
        if not isinstance(reference, Mapping):
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume reference is malformed"
            )
        container_id = reference.get("container_id")
        labels = reference.get("labels")
        if (
            not isinstance(container_id, str)
            or not container_id
            or not isinstance(labels, Mapping)
            or container_id in found_by_id
        ):
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume reference is malformed"
            )
        found_by_id[container_id] = reference
    if set(found_by_id) != set(expected_by_id):
        raise InfrastructureError(
            "recovery_required: managed WORKDIR volume reference set differs "
            "from authority"
        )
    for container_id, expected in expected_by_id.items():
        labels = found_by_id[container_id]["labels"]
        if (
            labels.get(_RUN_LABEL) != volume.run_id
            or labels.get(_TASK_LABEL) != volume.task_id
            or labels.get(_ROLE_LABEL) != expected.role
        ):
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume reference labels "
                "differ from authority"
            )


def normalized_workdir_volume_container_references(
    client: Any, name: str
) -> tuple[Mapping[str, Any], ...]:
    """Normalize every Docker container reference with its recovery labels."""
    try:
        found = client.containers.list(all=True, filters={"volume": name})
    except Exception as error:
        raise InfrastructureError(
            "recovery_required: managed WORKDIR volume container references "
            f"are unproven: {error}"
        ) from error
    if not isinstance(found, list):
        raise InfrastructureError(
            "recovery_required: Docker returned malformed managed WORKDIR "
            "volume container references"
        )
    references: list[Mapping[str, Any]] = []
    for container in found:
        try:
            container.reload()
            container_id = container.id
            attrs = container.attrs
            config = attrs["Config"]
            labels = config["Labels"]
            if (
                not isinstance(container_id, str)
                or not isinstance(config, Mapping)
                or not isinstance(labels, Mapping)
            ):
                raise TypeError("malformed container reference")
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume container "
                f"reference is unproven: {error}"
            ) from error
        references.append({"container_id": container_id, "labels": dict(labels)})
    return tuple(references)


class DockerWorkdirVolumeBackend:
    """Create and remove only fresh, exact-labeled Engine-owned local volumes."""

    def __init__(
        self,
        client: Any,
        *,
        nonce_factory: Callable[[], str] | None = None,
    ) -> None:
        self._client = client
        self._nonce_factory = nonce_factory or (lambda: secrets.token_hex(32))

    def planned_name(self, *, run_id: str, task_id: str) -> str:
        self._validate_ids(run_id=run_id, task_id=task_id)
        return f"{_PREFIX}{_digest(run_id, task_id)}"

    def plan(
        self,
        *,
        run_id: str,
        task_id: str,
        target: PurePosixPath,
    ) -> ManagedWorkdirVolume:
        """Mint the complete identity that must be durable before create."""
        return self._expected(
            run_id=run_id,
            task_id=task_id,
            target=target,
            planned_name=self.planned_name(run_id=run_id, task_id=task_id),
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce=self._nonce_factory(),
        )

    def create(self, planned: ManagedWorkdirVolume) -> ManagedWorkdirVolume:
        expected = self._canonical(planned)
        self._require_fresh_absence(expected)
        try:
            self._client.volumes.create(
                expected.name,
                driver="local",
                labels=managed_workdir_volume_labels(expected),
            )
        except Exception as error:
            # Docker create is mutating and its response is not a commit point.
            # Even when follow-up queries currently report absence, the exact
            # persisted plan remains recovery authority across that ambiguity.
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume create outcome is "
                "ambiguous; exact planned authority is retained"
            ) from error
        return self.inspect(expected, expected_references=())

    def inspect(
        self,
        volume: ManagedWorkdirVolume,
        *,
        expected_references: Sequence[ContainerRef] = (),
    ) -> ManagedWorkdirVolume:
        expected = self._canonical(volume)
        direct = self._get_exact(
            expected,
            missing_is_none=False,
            expected_references=expected_references,
        )
        assert direct is not None
        discovered = self._discover_exact(
            expected, expected_references=expected_references
        )
        if len(discovered) != 1 or self._name(discovered[0]) != expected.name:
            raise InfrastructureError(
                "recovery_required: exact-labeled managed WORKDIR volume "
                "identity is not unique"
            )
        return expected

    def remove(self, volume: ManagedWorkdirVolume) -> None:
        expected = self._canonical(volume)
        direct = self._get_exact(
            expected,
            missing_is_none=True,
            expected_references=(),
        )
        if direct is None:
            self._require_absent(expected)
            return
        discovered = self._discover_exact(expected, expected_references=())
        if len(discovered) != 1 or self._name(discovered[0]) != expected.name:
            raise InfrastructureError(
                "recovery_required: exact-labeled managed WORKDIR volume "
                "identity is not unique"
            )
        try:
            direct.remove(force=False)
        except Exception as error:
            self._require_absent_after_removal(expected, error)
            return
        self._require_absent_after_removal(expected, None)

    def special_files(
        self, volume: ManagedWorkdirVolume, image: str
    ) -> list[tuple[PurePosixPath, str]]:
        """Up to SPECIAL_FILE_LIMIT FIFOs, sockets and devices in the volume,
        as paths below its root (special_files.py says why). ``image`` is the
        clean Work image, which the Agent cannot change: a helper from it
        walks the volume read-only, with no network, as root that may read
        every directory but write none."""
        expected = self._canonical(volume)
        if not is_immutable_image_ref(image):
            raise InfrastructureError("special-file check needs an immutable image")
        failed = f"rsi-scan-failed-{secrets.token_hex(8)}"
        try:
            output = self._client.containers.run(
                image,
                [_SCAN_SCRIPT],
                name=f"{expected.name}-scan-{secrets.token_hex(4)}",
                entrypoint=["/bin/sh", "-c"],
                user="root",
                working_dir="/",
                environment={
                    NVIDIA_VISIBLE_DEVICES_ENV: NVIDIA_VISIBLE_DEVICES_VOID,
                    "RSI_SCAN_ROOT": _SCAN_ROOT,
                    "RSI_SCAN_LIMIT": str(SPECIAL_FILE_LIMIT),
                    "RSI_SCAN_FAILED": failed,
                },
                mounts=[
                    Mount(
                        target=_SCAN_ROOT,
                        source=expected.name,
                        type="volume",
                        read_only=True,
                        no_copy=True,
                    )
                ],
                network_mode="none",
                privileged=False,
                cap_drop=["ALL"],
                cap_add=["DAC_READ_SEARCH"],
                labels={
                    _RUN_LABEL: expected.run_id,
                    _TASK_LABEL: expected.task_id,
                    _ROLE_LABEL: "helper",
                },
                remove=True,
            )
        except Exception as error:
            raise InfrastructureError(
                f"WORKDIR special-file check failed: {error}"
            ) from error
        lines = [
            line
            for line in bytes(output or b"").decode(errors="replace").splitlines()
            if line
        ]
        if failed in lines:
            raise InfrastructureError("WORKDIR special-file check could not finish")
        prefix = f"{_SCAN_ROOT}/"
        return [
            (
                PurePosixPath(line.removeprefix(prefix))
                if line.startswith(prefix)
                else PurePosixPath(repr(line)),
                "FIFO, socket or device",
            )
            for line in lines
        ]

    def _require_fresh_absence(self, expected: ManagedWorkdirVolume) -> None:
        try:
            self._client.volumes.get(expected.name)
        except NotFound:
            pass
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: pre-create direct-name WORKDIR volume "
                f"absence is unproven: {error}"
            ) from error
        else:
            raise InfrastructureError(
                "recovery_required: pre-existing direct-name WORKDIR volume "
                "prevents fresh creation"
            )
        if self._list_exact(expected):
            raise InfrastructureError(
                "recovery_required: pre-existing exact-labeled WORKDIR volume "
                "prevents fresh creation"
            )

    def _require_absent_after_removal(
        self, expected: ManagedWorkdirVolume, error: Exception | None
    ) -> None:
        try:
            found = self._get_exact(
                expected,
                missing_is_none=True,
                expected_references=(),
            )
            self._require_absent(expected)
        except InfrastructureError:
            raise
        if found is None:
            return
        detail = f": {error}" if error is not None else ""
        raise InfrastructureError(
            "recovery_required: managed WORKDIR volume remains after removal" + detail
        )

    def _require_absent(self, expected: ManagedWorkdirVolume) -> None:
        if self._list_exact(expected):
            raise InfrastructureError(
                "recovery_required: exact-labeled managed WORKDIR volume remains"
            )

    def _get_exact(
        self,
        expected: ManagedWorkdirVolume,
        *,
        missing_is_none: bool,
        expected_references: Sequence[ContainerRef],
    ) -> Any | None:
        try:
            volume = self._client.volumes.get(expected.name)
        except NotFound:
            if missing_is_none:
                return None
            raise InfrastructureError(
                "recovery_required: exact managed WORKDIR volume is absent"
            ) from None
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: exact managed WORKDIR volume inspection is "
                f"unproven: {error}"
            ) from error
        try:
            volume.reload()
            self._attest(
                volume,
                expected,
                expected_references=expected_references,
            )
        except InfrastructureError:
            raise
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: exact managed WORKDIR volume attestation "
                f"is malformed: {error}"
            ) from error
        return volume

    def _list_exact(self, expected: ManagedWorkdirVolume) -> list[Any]:
        try:
            volumes = self._client.volumes.list(
                filters={
                    "label": [
                        f"{key}={value}"
                        for key, value in sorted(
                            managed_workdir_volume_labels(expected).items()
                        )
                    ]
                }
            )
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: exact-labeled managed WORKDIR volume query "
                f"is unproven: {error}"
            ) from error
        if not isinstance(volumes, list):
            raise InfrastructureError(
                "recovery_required: Docker returned malformed exact-labeled "
                "managed WORKDIR volume results"
            )
        return volumes

    def _discover_exact(
        self,
        expected: ManagedWorkdirVolume,
        *,
        expected_references: Sequence[ContainerRef],
    ) -> list[Any]:
        volumes = self._list_exact(expected)
        for volume in volumes:
            try:
                volume.reload()
                self._attest(
                    volume,
                    expected,
                    expected_references=expected_references,
                )
            except InfrastructureError:
                raise
            except Exception as error:
                raise InfrastructureError(
                    "recovery_required: discovered managed WORKDIR volume does "
                    f"not attest exact authority: {error}"
                ) from error
        return volumes

    @staticmethod
    def _name(volume: Any) -> str:
        try:
            attrs = volume.attrs
            name = attrs["Name"]
            if not isinstance(name, str):
                raise TypeError("volume name is not a string")
        except Exception as error:
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume name is malformed"
            ) from error
        return name

    def _attest(
        self,
        volume: Any,
        expected: ManagedWorkdirVolume,
        *,
        expected_references: Sequence[ContainerRef],
    ) -> None:
        attrs = volume.attrs
        if not isinstance(attrs, Mapping):
            raise InfrastructureError(
                "recovery_required: managed WORKDIR volume attributes are malformed"
            )
        labels = attrs.get("Labels")
        options = attrs.get("Options")
        scope = attrs.get("Scope")
        if labels is None:
            labels = {}
        if options is None:
            options = {}
        state = {
            "name": attrs.get("Name"),
            "driver": attrs.get("Driver"),
            "labels": labels,
            "options": options,
            "scope": scope,
            "container_references": normalized_workdir_volume_container_references(
                self._client, expected.name
            ),
        }
        attest_normalized_workdir_volume_state(
            state,
            expected,
            expected_references=expected_references,
        )

    def _canonical(self, volume: ManagedWorkdirVolume) -> ManagedWorkdirVolume:
        if not isinstance(volume, ManagedWorkdirVolume):
            raise SetupError("managed WORKDIR volume authority is untyped")
        return self._expected(
            run_id=volume.run_id,
            task_id=volume.task_id,
            target=volume.target,
            planned_name=volume.name,
            snapshot_mode=volume.snapshot_mode,
            freshness_nonce=volume.freshness_nonce,
        )

    def _expected(
        self,
        *,
        run_id: str,
        task_id: str,
        target: PurePosixPath,
        planned_name: str,
        snapshot_mode: RootfsSnapshotMode,
        freshness_nonce: str,
    ) -> ManagedWorkdirVolume:
        canonical = self.planned_name(run_id=run_id, task_id=task_id)
        if planned_name != canonical:
            raise SetupError("planned managed WORKDIR volume name is not canonical")
        try:
            return ManagedWorkdirVolume(
                name=canonical,
                run_id=run_id,
                task_id=task_id,
                target=target,
                snapshot_mode=snapshot_mode,
                freshness_nonce=freshness_nonce,
            )
        except ValidationError as error:
            raise SetupError(
                f"invalid managed WORKDIR volume authority: {error}"
            ) from error

    @staticmethod
    def _validate_ids(*, run_id: str, task_id: str) -> None:
        for field, value in (("run ID", run_id), ("task ID", task_id)):
            if (
                not isinstance(value, str)
                or not value
                or any(character in value for character in ("\0", "\n", "\r"))
            ):
                raise SetupError(f"unsafe managed WORKDIR volume {field}")


__all__ = [
    "DockerWorkdirVolumeBackend",
    "attest_normalized_workdir_volume_state",
    "managed_workdir_volume_labels",
    "normalized_workdir_volume_container_references",
]
