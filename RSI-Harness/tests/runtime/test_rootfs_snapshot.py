from __future__ import annotations

import hashlib
import traceback
from typing import Any
from uuid import uuid4

import pytest
from docker.errors import APIError, ImageNotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerRef, RootfsSnapshotLease
from rsi_harness.runtime.rootfs_snapshot import (
    DockerRootfsSnapshotBackend,
    RootfsSnapshotNotCreatedError,
)


def _suffix(*values: str) -> str:
    payload = b"".join(
        len(value.encode("utf-8")).to_bytes(8, "big") + value.encode("utf-8")
        for value in values
    )
    return hashlib.sha256(payload).hexdigest()


def _assert_secret_free_exception(error: Exception, secret: str) -> None:
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))


class FakeImage:
    def __init__(
        self, image_id: str, *, tags: list[str], labels: dict[str, str]
    ) -> None:
        self.id = image_id
        self.attrs: dict[str, Any] = {
            "Id": image_id,
            "RepoTags": tags,
            "Config": {"Labels": labels},
        }


class FakeWorkContainer:
    def __init__(self, container_id: str, images: FakeImages) -> None:
        self.id = container_id
        self._images = images
        self.commits: list[dict[str, Any]] = []
        self.next_image_id = "sha256:" + "a" * 64

    def commit(self, **kwargs: Any) -> FakeImage:
        self.commits.append(kwargs)
        ref = f"{kwargs['repository']}:{kwargs['tag']}"
        image = FakeImage(
            self.next_image_id,
            tags=[ref],
            labels=dict(kwargs["conf"]["Labels"]),
        )
        self._images.by_ref[ref] = image
        self._images.by_ref[image.id] = image
        return image


class FakeContainers:
    def __init__(self) -> None:
        self.by_id: dict[str, FakeWorkContainer] = {}
        self.ancestors: dict[str, list[object]] = {}

    def get(self, container_id: str) -> FakeWorkContainer:
        return self.by_id[container_id]

    def list(self, *, filters: dict[str, str], **_: Any) -> list[object]:
        return self.ancestors.get(filters["ancestor"], [])


class FakeImages:
    def __init__(self) -> None:
        self.by_ref: dict[str, FakeImage] = {}
        self.removed: list[tuple[str, bool]] = []
        self.get_calls: list[str] = []

    def get(self, reference: str) -> FakeImage:
        self.get_calls.append(reference)
        if reference not in self.by_ref:
            raise ImageNotFound(reference)
        return self.by_ref[reference]

    def remove(self, image: str, *, force: bool) -> None:
        self.removed.append((image, force))
        found = self.get(image)
        for key, candidate in list(self.by_ref.items()):
            if candidate is found:
                del self.by_ref[key]

    def list(self, *, filters: dict[str, list[str]]) -> list[FakeImage]:
        labels = filters["label"]
        unique = {id(image): image for image in self.by_ref.values()}.values()
        return [
            image
            for image in unique
            if all(
                image.attrs["Config"]["Labels"].get(label.split("=", 1)[0])
                == label.split("=", 1)[1]
                for label in labels
            )
        ]


class FakeClient:
    def __init__(self) -> None:
        self.images = FakeImages()
        self.containers = FakeContainers()


def _backend_and_work() -> tuple[DockerRootfsSnapshotBackend, FakeClient, ContainerRef]:
    client = FakeClient()
    work = ContainerRef(container_id="work-123", role="work")
    client.containers.by_id[work.container_id] = FakeWorkContainer(
        work.container_id, client.images
    )
    return DockerRootfsSnapshotBackend(client), client, work


def test_acquire_commits_attested_rootfs_image_with_deterministic_safe_tag() -> None:
    backend, client, work = _backend_and_work()

    planned = backend.planned_ref(
        run_id="run alpha",
        task_id="task:one",
        round_id="round/1",
        purpose="judge-round",
    )
    lease = backend.acquire(
        work,
        run_id="run alpha",
        task_id="task:one",
        round_id="round/1",
    )

    expected_suffix = _suffix("run alpha", "task:one", "round/1", "judge-round")
    assert planned == f"rsi-harness-rootfs:judge-round-{expected_suffix}"
    assert lease.image_ref == planned
    assert lease.image_id == "sha256:" + "a" * 64
    assert lease.source_container_id == "work-123"
    assert lease.lease_id == _suffix("run alpha", "task:one", "round/1", "judge-round")
    assert client.images.get_calls == [planned, lease.image_id]
    assert client.containers.by_id["work-123"].commits == [
        {
            "repository": "rsi-harness-rootfs",
            "tag": f"judge-round-{expected_suffix}",
            "pause": False,
            "changes": [],
            "conf": {
                "Labels": {
                    "rsi-harness.run-id": "run alpha",
                    "rsi-harness.task-id": "task:one",
                    "rsi-harness.round-id": "round/1",
                    "rsi-harness.source-container-id": "work-123",
                    "rsi-harness.role": "rootfs-snapshot",
                },
            },
        }
    ]


def test_planned_rootfs_refs_bind_distinct_task_ids() -> None:
    backend, _client, work = _backend_and_work()

    first_ref = backend.planned_ref(
        run_id="run", task_id="task-one", round_id="1", purpose="judge-round"
    )
    second_ref = backend.planned_ref(
        run_id="run", task_id="task-two", round_id="1", purpose="judge-round"
    )
    first = backend.acquire(
        work, run_id="run", task_id="task-one", round_id="1", planned_ref=first_ref
    )
    second = backend.acquire(
        work, run_id="run", task_id="task-two", round_id="1", planned_ref=second_ref
    )

    assert first_ref != second_ref
    assert first.image_ref != second.image_ref


@pytest.mark.parametrize(
    ("work", "run_id", "task_id", "round_id", "purpose"),
    [
        (
            ContainerRef(container_id="judge-1", role="judge"),
            "run",
            "task",
            "1",
            "judge-round",
        ),
        (
            ContainerRef(container_id="bad id", role="work"),
            "run",
            "task",
            "1",
            "judge-round",
        ),
        (
            ContainerRef(container_id="work-1", role="work"),
            "bad\x00",
            "task",
            "1",
            "judge-round",
        ),
        (
            ContainerRef(container_id="work-1", role="work"),
            "run",
            "task",
            "1",
            "unknown",
        ),
    ],
)
def test_acquire_rejects_non_work_or_unsafe_snapshot_identity(
    work: ContainerRef, run_id: str, task_id: str, round_id: str, purpose: str
) -> None:
    backend, client, _ = _backend_and_work()

    with pytest.raises(SetupError):
        backend.acquire(
            work,
            run_id=run_id,
            task_id=task_id,
            round_id=round_id,
            purpose=purpose,
        )

    assert client.containers.by_id["work-123"].commits == []


def test_acquire_rejects_mismatched_planned_ref_before_commit() -> None:
    backend, client, work = _backend_and_work()

    with pytest.raises(SetupError, match="planned rootfs image reference"):
        backend.acquire(
            work,
            run_id="run", task_id="task", round_id="1", planned_ref="wrong:tag"
        )

    assert client.containers.by_id["work-123"].commits == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda image: image.attrs["Config"]["Labels"].__setitem__(
            "rsi-harness.role", "work"
        ),
        lambda image: image.attrs.__setitem__("RepoTags", ["elsewhere:mutable"]),
        lambda image: image.attrs.__setitem__("Id", "not-an-immutable-id"),
    ],
)
def test_acquire_rejects_post_commit_image_that_cannot_attest_identity(mutate) -> None:
    backend, client, work = _backend_and_work()
    container = client.containers.by_id[work.container_id]
    original = container.commit

    def commit(**kwargs: Any) -> FakeImage:
        image = original(**kwargs)
        mutate(image)
        return image

    container.commit = commit  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.acquire(work, run_id="run", task_id="task", round_id="1")


def test_acquire_ambiguous_commit_response_is_sanitized_and_requires_recovery() -> None:
    backend, client, work = _backend_and_work()
    container = client.containers.by_id[work.container_id]
    original = container.commit

    def commit(**kwargs: Any) -> FakeImage:
        original(**kwargs)
        raise APIError("secret=value")

    container.commit = commit  # type: ignore[method-assign]
    planned = backend.planned_ref(
        run_id="run", task_id="task", round_id="1", purpose="judge-round"
    )

    with pytest.raises(InfrastructureError, match="recovery_required") as caught:
        backend.acquire(work, run_id="run", task_id="task", round_id="1")

    assert client.images.get_calls[-1] == planned
    _assert_secret_free_exception(caught.value, "secret=value")


def test_acquire_rejects_returned_container_with_a_different_id() -> None:
    backend, client, work = _backend_and_work()
    client.containers.by_id[work.container_id].id = "other-work"

    with pytest.raises(RootfsSnapshotNotCreatedError, match="requested Work"):
        backend.acquire(work, run_id="run", task_id="task", round_id="1")

    assert client.containers.by_id[work.container_id].commits == []


def test_acquire_container_inspection_failure_cancels_only_after_absence_proof(
) -> None:
    backend, client, work = _backend_and_work()
    client.containers.get = lambda _container_id: (_ for _ in ()).throw(  # type: ignore[method-assign]
        APIError("secret=value")
    )

    with pytest.raises(RootfsSnapshotNotCreatedError) as caught:
        backend.acquire(work, run_id="run", task_id="task", round_id="1")

    _assert_secret_free_exception(caught.value, "secret=value")
    planned = backend.planned_ref(
        run_id="run", task_id="task", round_id="1", purpose="judge-round"
    )
    assert client.images.get_calls == [planned]


def test_acquire_retains_authority_for_existing_planned_tag_before_commit() -> None:
    backend, client, work = _backend_and_work()
    planned = backend.planned_ref(
        run_id="run", task_id="task", round_id="1", purpose="judge-round"
    )
    client.images.by_ref[planned] = FakeImage(
        "sha256:" + "e" * 64, tags=[planned], labels={}
    )

    with pytest.raises(
        InfrastructureError,
        match="recovery_required.*planned rootfs image already exists",
    ):
        backend.acquire(work, run_id="run", task_id="task", round_id="1")

    assert client.containers.by_id[work.container_id].commits == []


def test_acquire_retains_authority_for_existing_exact_labeled_image() -> None:
    backend, client, work = _backend_and_work()
    prior = _leased_image(_lease())
    client.images.by_ref["prior"] = prior

    with pytest.raises(InfrastructureError, match="recovery_required.*exact-labeled"):
        backend.acquire(work, run_id="run", task_id="task", round_id="1")

    assert client.containers.by_id[work.container_id].commits == []


def test_acquire_proves_exact_label_uniqueness_and_rolls_back_by_immutable_id() -> None:
    backend, client, work = _backend_and_work()
    container = client.containers.by_id[work.container_id]
    original = container.commit

    def commit(**kwargs: Any) -> FakeImage:
        image = original(**kwargs)
        duplicate = FakeImage(
            "sha256:" + "c" * 64,
            tags=["elsewhere:duplicate"],
            labels=dict(kwargs["conf"]["Labels"]),
        )
        client.images.by_ref["duplicate"] = duplicate
        return image

    container.commit = commit  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.acquire(work, run_id="run", task_id="task", round_id="1")

    assert client.images.removed[0] == ("sha256:" + "a" * 64, False)


def _lease() -> RootfsSnapshotLease:
    return RootfsSnapshotLease(
        lease_id="lease-1",
        purpose="judge-round",
        run_id="run",
        task_id="task",
        round_id="1",
        source_container_id="work-123",
        image_id="sha256:" + "b" * 64,
        image_ref="rsi-harness-rootfs:judge-round-release",
    )


def _leased_image(lease: RootfsSnapshotLease) -> FakeImage:
    return FakeImage(
        lease.image_id,
        tags=[lease.image_ref],
        labels={
            "rsi-harness.run-id": lease.run_id,
            "rsi-harness.task-id": lease.task_id,
            "rsi-harness.round-id": lease.round_id,
            "rsi-harness.source-container-id": lease.source_container_id,
            "rsi-harness.role": "rootfs-snapshot",
        },
    )


def _install_leased_image(client: FakeClient, lease: RootfsSnapshotLease) -> FakeImage:
    image = _leased_image(lease)
    client.images.by_ref[lease.image_id] = image
    client.images.by_ref[lease.image_ref] = image
    return image


def test_release_removes_only_attested_unreferenced_image_and_proves_absence() -> None:
    backend, client, _ = _backend_and_work()
    lease = _lease()
    _install_leased_image(client, lease)

    backend.release(lease)

    assert client.images.removed == [(lease.image_id, False)]
    with pytest.raises(ImageNotFound):
        client.images.get(lease.image_id)
    assert client.images.list(filters={"label": ["rsi-harness.run-id=run"]}) == []


def test_release_is_idempotent_when_image_is_already_absent() -> None:
    backend, client, _ = _backend_and_work()

    backend.release(_lease())

    assert client.images.removed == []


def test_release_fails_closed_while_any_container_references_snapshot_image() -> None:
    backend, client, _ = _backend_and_work()
    lease = _lease()
    _install_leased_image(client, lease)
    client.containers.ancestors[lease.image_id] = [object()]

    with pytest.raises(InfrastructureError, match="container"):
        backend.release(lease)

    assert client.images.removed == []


@pytest.mark.parametrize("response", [None, {"judge": object()}])
def test_release_rejects_malformed_container_reference_response(
    response: object,
) -> None:
    backend, client, _ = _backend_and_work()
    lease = _lease()
    _install_leased_image(client, lease)
    client.containers.list = lambda **_: response  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.release(lease)

    assert client.images.removed == []


def test_release_list_api_error_is_sanitized_and_requires_recovery() -> None:
    backend, client, _ = _backend_and_work()
    lease = _lease()
    _install_leased_image(client, lease)
    client.containers.list = lambda **_: (_ for _ in ()).throw(  # type: ignore[method-assign]
        APIError("secret=value")
    )

    with pytest.raises(InfrastructureError, match="recovery_required") as caught:
        backend.release(lease)

    _assert_secret_free_exception(caught.value, "secret=value")
    assert client.images.removed == []


def test_release_rejects_image_with_wrong_live_labels_before_removal() -> None:
    backend, client, _ = _backend_and_work()
    lease = _lease()
    image = _install_leased_image(client, lease)
    image.attrs["Config"]["Labels"]["rsi-harness.role"] = "work"

    with pytest.raises(InfrastructureError, match="attestation"):
        backend.release(lease)

    assert client.images.removed == []


def test_release_requires_absence_of_every_exact_labeled_image() -> None:
    backend, client, _ = _backend_and_work()
    lease = _lease()
    _install_leased_image(client, lease)
    leftover = _leased_image(lease)
    leftover.id = "sha256:" + "c" * 64
    leftover.attrs["Id"] = leftover.id
    client.images.by_ref["leftover"] = leftover

    with pytest.raises(InfrastructureError, match="exact-labeled"):
        backend.release(lease)


@pytest.mark.parametrize("response", [None, {"image": object()}])
def test_release_rejects_malformed_exact_label_absence_response(
    response: object,
) -> None:
    backend, client, _ = _backend_and_work()
    client.images.list = lambda **_: response  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.release(_lease())


def test_acquire_requires_recovery_if_post_commit_inspection_and_cleanup_fail() -> None:
    backend, client, work = _backend_and_work()
    container = client.containers.by_id[work.container_id]
    container.commit = lambda **_: FakeImage(  # type: ignore[method-assign]
        "sha256:" + "d" * 64, tags=[], labels={}
    )
    client.images.get = lambda _: (_ for _ in ()).throw(ImageNotFound("missing"))  # type: ignore[method-assign]
    client.images.remove = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        APIError("cleanup failed")
    )

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.acquire(work, run_id="run", task_id="task", round_id="1")


def test_release_requires_recovery_when_image_removal_fails() -> None:
    backend, client, _ = _backend_and_work()
    lease = _lease()
    _install_leased_image(client, lease)
    client.images.remove = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        APIError("daemon unavailable")
    )

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.release(lease)


@pytest.mark.integration
def test_real_docker_commit_snapshot_preserves_complete_work_rootfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker = pytest.importorskip("docker")
    client = docker.from_env()
    try:
        client.ping()
    except docker.errors.DockerException as error:
        pytest.skip(f"Docker daemon unavailable: {error}")

    run_id = f"rootfs-cow-{uuid4().hex}"
    backend = DockerRootfsSnapshotBackend(client)
    planned_ref = backend.planned_ref(
        run_id=run_id,
        task_id="rootfs-cow",
        round_id="1",
        purpose="judge-round",
    )
    work = None
    judge = None
    lease = None
    archive_calls: list[str] = []

    def forbid_archive(*_args: Any, **_kwargs: Any) -> None:
        archive_calls.append("archive")
        raise AssertionError("rootfs snapshots must not use archive APIs")

    for method in ("export", "import_image", "get_archive", "put_archive"):
        if hasattr(client.api, method):
            monkeypatch.setattr(client.api, method, forbid_archive)

    try:
        base = client.images.pull("ubuntu:24.04")
        base_id = base.attrs["Id"]
        work = client.containers.create(
            "ubuntu:24.04",
            ["/bin/sh", "-c", "while true; do sleep 3600; done"],
            labels={"rsi-harness.test": run_id},
        )
        work.start()
        mutation = work.exec_run(
            [
                "/bin/sh",
                "-c",
                "mkdir -p /usr/local/bin /testbed; "
                "printf etc > /etc/rsi-rootfs-cow; "
                "printf bin > /usr/local/bin/rsi-rootfs-cow; "
                "printf tmp > /tmp/rsi-rootfs-cow; "
                "printf testbed > /testbed/rsi-rootfs-cow; "
                "truncate -s 8G /testbed/rsi-rootfs-cow.sparse",
            ]
        )
        assert mutation.exit_code == 0
        work.pause()
        lease = backend.acquire(
            ContainerRef(container_id=work.id, role="work"),
            run_id=run_id,
            task_id="rootfs-cow",
            round_id="1",
            planned_ref=planned_ref,
        )
        judge = client.containers.create(
            lease.image_id,
            ["/bin/sh", "-c", "while true; do sleep 3600; done"],
            labels={"rsi-harness.test": run_id},
        )
        judge.start()
        verified = judge.exec_run(
            [
                "/bin/sh",
                "-c",
                "test \"$(cat /etc/rsi-rootfs-cow)\" = etc && "
                "test \"$(cat /usr/local/bin/rsi-rootfs-cow)\" = bin && "
                "test \"$(cat /tmp/rsi-rootfs-cow)\" = tmp && "
                "test \"$(cat /testbed/rsi-rootfs-cow)\" = testbed && "
                "test \"$(stat -c %s /testbed/rsi-rootfs-cow.sparse)\" = 8589934592",
            ]
        )
        assert verified.exit_code == 0
        assert client.images.get("ubuntu:24.04").attrs["Id"] == base_id
        assert archive_calls == []
    finally:
        if judge is not None:
            try:
                judge.remove(force=True)
            except docker.errors.DockerException:
                pass
        if work is not None:
            try:
                work.remove(force=True)
            except docker.errors.DockerException:
                pass
        if lease is not None:
            try:
                backend.release(lease)
            except InfrastructureError:
                try:
                    client.images.remove(lease.image_id, force=True)
                except docker.errors.DockerException:
                    pass
        try:
            client.images.remove(planned_ref, force=True)
        except docker.errors.DockerException:
            pass
