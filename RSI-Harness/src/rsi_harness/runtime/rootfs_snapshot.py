"""Docker commit-backed, attested snapshots of a paused Work root filesystem."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from docker.errors import APIError, ImageNotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerRef, RootfsSnapshotLease

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_PURPOSES = frozenset({"judge-round", "retained-work"})
_REPOSITORY = "rsi-harness-rootfs"


class RootfsSnapshotNotCreatedError(SetupError):
    """Acquire failed with authoritative proof that no image was created."""


def _digest(*values: str) -> str:
    payload = b"".join(
        len(value.encode("utf-8")).to_bytes(8, "big") + value.encode("utf-8")
        for value in values
    )
    return hashlib.sha256(payload).hexdigest()


def _require_label_value(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or any(
        character in value for character in ("\0", "\n", "\r")
    ):
        raise SetupError(f"unsafe rootfs snapshot {field}")
    return value


class DockerRootfsSnapshotBackend:
    """Commit and attest the Docker-owned writable layer of a Work container."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def planned_ref(
        self, *, run_id: str, task_id: str, round_id: str, purpose: str
    ) -> str:
        """Return the deterministic task-bound local rootfs image tag."""
        self._validate_identity(
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
        )
        return self._reference(
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
        )

    def acquire(
        self,
        work: ContainerRef,
        *,
        run_id: str,
        task_id: str,
        round_id: str,
        purpose: str = "judge-round",
        planned_ref: str | None = None,
    ) -> RootfsSnapshotLease:
        if work.role != "work":
            raise RootfsSnapshotNotCreatedError(
                "rootfs snapshot source must be a Work container"
            )
        try:
            self._validate_identity(
                run_id=run_id,
                task_id=task_id,
                round_id=round_id,
                purpose=purpose,
            )
        except SetupError as error:
            raise RootfsSnapshotNotCreatedError(str(error)) from None
        if _CONTAINER_ID.fullmatch(work.container_id) is None:
            raise RootfsSnapshotNotCreatedError(
                "unsafe rootfs snapshot source container ID"
            )
        reference = self._reference(
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
        )
        if planned_ref is not None and planned_ref != reference:
            raise RootfsSnapshotNotCreatedError(
                "planned rootfs image reference does not match authority"
            )
        repository, tag = reference.split(":", 1)
        labels = self._labels(
            work=work,
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
        )
        container: Any | None = None
        container_error = False
        try:
            container = self._client.containers.get(work.container_id)
        except (APIError, AttributeError, KeyError, TypeError):
            container_error = True
        if container_error:
            self._raise_not_created_after_proof(
                reference,
                labels,
                "failed to inspect Work container for rootfs snapshot",
            )
        if getattr(container, "id", None) != work.container_id:
            self._raise_not_created_after_proof(
                reference,
                labels,
                "Docker returned a different requested Work container",
            )
        self._require_new_snapshot_identity(reference, labels)
        commit_api_error = False
        commit_malformed = False
        try:
            committed = container.commit(
                repository=repository,
                tag=tag,
                pause=False,
                changes=[],
                conf={"Labels": labels},
            )
        except APIError:
            commit_api_error = True
        except (AttributeError, KeyError, TypeError):
            commit_malformed = True
        if commit_api_error:
            self._inspect_ambiguous_commit(reference, labels)
        if commit_malformed:
            raise InfrastructureError(
                "recovery_required: rootfs commit response is malformed"
            )

        committed_id = getattr(committed, "id", None)
        if (
            not isinstance(committed_id, str)
            or _IMAGE_ID.fullmatch(committed_id) is None
        ):
            self._inspect_ambiguous_commit(reference, labels)
        post_commit_error = False
        try:
            image = self._client.images.get(committed_id)
            image_id = self._attest(image, reference=reference, labels=labels)
            if image_id != committed_id:
                raise TypeError("committed image ID changed during inspection")
            self._require_one_exact_labeled_image(
                labels, image_id=image_id, reference=reference
            )
        except (APIError, ImageNotFound, AttributeError, KeyError, TypeError):
            post_commit_error = True
        if post_commit_error:
            self._rollback_after_failed_attestation(committed_id, labels)
        return RootfsSnapshotLease(
            lease_id=_digest(run_id, task_id, round_id, purpose),
            purpose=purpose,  # type: ignore[arg-type]
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            source_container_id=work.container_id,
            image_id=image_id,
            image_ref=reference,
        )

    def release(self, lease: RootfsSnapshotLease) -> None:
        """Remove only the exact, attested image after all users are gone."""
        labels = self._lease_labels(lease)
        image: Any | None = None
        image_missing = False
        image_inspection_error = False
        try:
            image = self._client.images.get(lease.image_id)
        except ImageNotFound:
            image_missing = True
        except (APIError, AttributeError, KeyError, TypeError):
            image_inspection_error = True
        if image_missing:
            self._require_no_exact_labeled_image(labels)
            return
        if image_inspection_error:
            raise InfrastructureError(
                "failed to inspect rootfs snapshot image for release"
            )
        assert image is not None
        attestation_error = False
        try:
            image_id = self._attest(image, reference=lease.image_ref, labels=labels)
        except (AttributeError, KeyError, TypeError):
            attestation_error = True
        if attestation_error:
            raise InfrastructureError(
                "rootfs snapshot image attestation failed"
            )
        if image_id != lease.image_id:
            raise InfrastructureError("rootfs snapshot image attestation failed")
        containers: Any = None
        container_listing_error = False
        try:
            containers = self._client.containers.list(
                all=True, filters={"ancestor": lease.image_id}
            )
        except (APIError, AttributeError, KeyError, TypeError):
            container_listing_error = True
        if container_listing_error:
            raise InfrastructureError(
                "recovery_required: rootfs snapshot container references are "
                "unproven"
            )
        if not isinstance(containers, list):
            raise InfrastructureError(
                "recovery_required: Docker returned malformed rootfs snapshot "
                "container references"
            )
        if containers:
            raise InfrastructureError(
                "rootfs snapshot image still has container references"
            )
        removal_error = False
        try:
            self._client.images.remove(lease.image_id, force=False)
        except ImageNotFound:
            pass
        except (APIError, AttributeError, KeyError, TypeError):
            removal_error = True
        if removal_error:
            raise InfrastructureError(
                "recovery_required: rootfs snapshot image removal is unproven"
            )
        absence_error = False
        image_remains = False
        try:
            self._client.images.get(lease.image_id)
        except ImageNotFound:
            pass
        except (APIError, AttributeError, KeyError, TypeError):
            absence_error = True
        else:
            image_remains = True
        if absence_error:
            raise InfrastructureError(
                "recovery_required: rootfs snapshot image absence is unproven"
            )
        if not image_remains:
            self._require_no_exact_labeled_image(labels)
            return
        raise InfrastructureError(
            "recovery_required: rootfs snapshot image remained after removal"
        )

    def _rollback_after_failed_attestation(
        self, image_id: str, labels: dict[str, str]
    ) -> None:
        cleanup_error = False
        try:
            self._client.images.remove(image_id, force=False)
        except ImageNotFound:
            pass
        except (APIError, AttributeError, KeyError, TypeError):
            cleanup_error = True
        if cleanup_error:
            raise InfrastructureError(
                "recovery_required: rootfs snapshot attestation cleanup is unproven"
            )
        absence_error = False
        image_remains = False
        try:
            self._client.images.get(image_id)
        except ImageNotFound:
            pass
        except (APIError, AttributeError, KeyError, TypeError):
            absence_error = True
        else:
            image_remains = True
        if absence_error:
            raise InfrastructureError(
                "recovery_required: rootfs snapshot rollback absence is unproven"
            )
        if image_remains:
            raise InfrastructureError(
                "recovery_required: rootfs snapshot remained after rollback"
            )
        self._require_no_exact_labeled_image(labels)
        raise InfrastructureError(
            "recovery_required: rootfs snapshot attestation failed after commit"
        )

    def _require_new_snapshot_identity(
        self, reference: str, labels: dict[str, str]
    ) -> None:
        tag_exists = False
        tag_inspection_error = False
        try:
            self._client.images.get(reference)
        except ImageNotFound:
            pass
        except (APIError, AttributeError, KeyError, TypeError):
            tag_inspection_error = True
        else:
            tag_exists = True
        if tag_inspection_error:
            raise InfrastructureError(
                "recovery_required: failed to inspect planned rootfs image"
            ) from None
        if tag_exists:
            raise InfrastructureError(
                "recovery_required: planned rootfs image already exists"
            )
        image_listing_error = False
        images: Any = None
        try:
            images = self._client.images.list(
                filters={"label": [f"{key}={value}" for key, value in labels.items()]}
            )
        except (APIError, AttributeError, KeyError, TypeError):
            image_listing_error = True
        if image_listing_error:
            raise InfrastructureError(
                "recovery_required: failed to inspect exact-labeled rootfs images"
            ) from None
        if not isinstance(images, list):
            raise InfrastructureError(
                "recovery_required: Docker returned malformed exact-labeled "
                "rootfs images"
            )
        if images:
            raise InfrastructureError(
                "recovery_required: exact-labeled rootfs image already exists"
            )

    def _raise_not_created_after_proof(
        self,
        reference: str,
        labels: dict[str, str],
        message: str,
    ) -> None:
        self._require_new_snapshot_identity(reference, labels)
        raise RootfsSnapshotNotCreatedError(message)

    def _inspect_ambiguous_commit(
        self, reference: str, labels: dict[str, str]
    ) -> None:
        try:
            image = self._client.images.get(reference)
            self._attest(image, reference=reference, labels=labels)
            self._require_one_exact_labeled_image(
                labels,
                image_id=image.id,
                reference=reference,
            )
        except (APIError, ImageNotFound, AttributeError, KeyError, TypeError):
            pass
        raise InfrastructureError(
            "recovery_required: rootfs commit response is ambiguous"
        ) from None

    def _require_one_exact_labeled_image(
        self, labels: dict[str, str], *, image_id: str, reference: str
    ) -> None:
        images = self._exact_labeled_images(labels)
        if len(images) != 1:
            raise TypeError("rootfs snapshot labels are not unique")
        listed_id = self._attest(images[0], reference=reference, labels=labels)
        if listed_id != image_id:
            raise TypeError("exact-labeled rootfs image ID changed")

    def _attest(
        self, image: Any, *, reference: str, labels: dict[str, str]
    ) -> str:
        attrs = image.attrs
        image_id = attrs["Id"]
        if not isinstance(image_id, str) or _IMAGE_ID.fullmatch(image_id) is None:
            raise TypeError("invalid image ID")
        if getattr(image, "id", image_id) != image_id:
            raise TypeError("inconsistent image ID")
        repo_tags = attrs["RepoTags"]
        if not isinstance(repo_tags, list) or reference not in repo_tags:
            raise TypeError("missing planned image tag")
        actual_labels = attrs["Config"]["Labels"]
        if not isinstance(actual_labels, dict) or any(
            actual_labels.get(key) != value for key, value in labels.items()
        ):
            raise TypeError("snapshot labels do not attest identity")
        return image_id

    def _require_no_exact_labeled_image(self, labels: dict[str, str]) -> None:
        images = self._exact_labeled_images(labels)
        if images:
            raise InfrastructureError(
                "recovery_required: rootfs exact-labeled image remained after "
                "removal"
            )

    def _exact_labeled_images(self, labels: dict[str, str]) -> list[Any]:
        listing_error = False
        images: Any = None
        try:
            images = self._client.images.list(
                filters={"label": [f"{key}={value}" for key, value in labels.items()]}
            )
        except (APIError, AttributeError, KeyError, TypeError):
            listing_error = True
        if listing_error:
            raise InfrastructureError(
                "recovery_required: exact-labeled rootfs image results are "
                "unproven"
            )
        if not isinstance(images, list):
            raise InfrastructureError(
                "recovery_required: Docker returned malformed exact-labeled "
                "rootfs image results"
            )
        return images

    @staticmethod
    def _reference(
        *, run_id: str, task_id: str, round_id: str, purpose: str
    ) -> str:
        return f"{_REPOSITORY}:{purpose}-{_digest(run_id, task_id, round_id, purpose)}"

    @staticmethod
    def _validate_identity(
        *, run_id: str, task_id: str, round_id: str, purpose: str
    ) -> None:
        _require_label_value(run_id, field="run ID")
        _require_label_value(task_id, field="task ID")
        _require_label_value(round_id, field="round ID")
        if purpose not in _PURPOSES:
            raise SetupError("unsupported rootfs snapshot purpose")

    @staticmethod
    def _labels(
        *,
        work: ContainerRef,
        run_id: str,
        task_id: str,
        round_id: str,
        purpose: str,
    ) -> dict[str, str]:
        return {
            "rsi-harness.run-id": run_id,
            "rsi-harness.task-id": task_id,
            "rsi-harness.round-id": round_id,
            "rsi-harness.source-container-id": work.container_id,
            "rsi-harness.role": (
                "rootfs-snapshot"
                if purpose == "judge-round"
                else "retained-work-rootfs"
            ),
        }

    def _lease_labels(self, lease: RootfsSnapshotLease) -> dict[str, str]:
        self._validate_identity(
            run_id=lease.run_id,
            task_id=lease.task_id,
            round_id=lease.round_id,
            purpose=lease.purpose,
        )
        if _CONTAINER_ID.fullmatch(lease.source_container_id) is None:
            raise InfrastructureError("unsafe rootfs snapshot source container ID")
        if _IMAGE_ID.fullmatch(lease.image_id) is None:
            raise InfrastructureError("invalid rootfs snapshot image ID")
        return self._labels(
            work=ContainerRef(container_id=lease.source_container_id, role="work"),
            run_id=lease.run_id,
            task_id=lease.task_id,
            round_id=lease.round_id,
            purpose=lease.purpose,
        )


__all__ = ["DockerRootfsSnapshotBackend", "RootfsSnapshotNotCreatedError"]
