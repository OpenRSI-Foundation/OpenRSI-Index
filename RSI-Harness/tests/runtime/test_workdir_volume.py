from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

import docker as docker_sdk
import pytest
from docker.errors import APIError, NotFound

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerRef, ManagedWorkdirVolume, RootfsSnapshotMode
from rsi_harness.runtime.workdir_volume import (
    DockerWorkdirVolumeBackend,
    managed_workdir_volume_labels,
)

EXPECTED_LABELS = {
    "rsi-harness.run-id": "run-1",
    "rsi-harness.task-id": "task-1",
    "rsi-harness.role": "workdir-volume",
    "rsi-harness.snapshot-mode": "split-workdir",
    "rsi-harness.workdir-target": "/workspace",
    "rsi-harness.freshness-nonce": "f" * 64,
}
FRESHNESS_NONCE = "f" * 64


class FakeVolume:
    def __init__(
        self,
        name: str,
        *,
        driver: str,
        labels: dict[str, str],
        options: dict[str, str] | None = None,
        scope: str = "local",
    ) -> None:
        self.name = name
        self.attrs: dict[str, Any] = {
            "Name": name,
            "Driver": driver,
            "Labels": dict(labels),
            "Options": dict(options or {}),
            "Scope": scope,
        }
        self.removed = False

    def reload(self) -> None:
        return None

    def remove(self, *, force: bool = False) -> None:
        assert force is False
        self.removed = True


class MalformedDirectVolume:
    """A Docker object with no inspect attrs, as a malformed SDK boundary."""

    removed = False


class FakeVolumes:
    def __init__(self) -> None:
        self.by_name: dict[str, FakeVolume] = {}
        self.created: list[dict[str, Any]] = []
        self.create_error: Exception | None = None
        self.create_on_error = False
        self.get_error: Exception | None = None
        self.list_error: Exception | None = None
        self.list_result: object | None = None

    def create(self, name: str, *, driver: str, labels: dict[str, str]) -> FakeVolume:
        self.created.append({"name": name, "driver": driver, "labels": dict(labels)})
        if self.create_error is not None:
            if self.create_on_error:
                self.by_name[name] = FakeVolume(name, driver=driver, labels=labels)
            raise self.create_error
        # Docker volume creation is idempotent by name: an existing volume is
        # returned without applying the requested labels or driver settings.
        if name in self.by_name and not self.by_name[name].removed:
            return self.by_name[name]
        volume = FakeVolume(name, driver=driver, labels=labels)
        self.by_name[name] = volume
        return volume

    def get(self, name: str) -> FakeVolume:
        if self.get_error is not None:
            raise self.get_error
        try:
            volume = self.by_name[name]
        except KeyError as error:
            raise NotFound(name) from error
        if volume.removed:
            raise NotFound(name)
        return volume

    def list(self, *, filters: dict[str, list[str]]) -> object:
        if self.list_error is not None:
            raise self.list_error
        if self.list_result is not None:
            return self.list_result
        labels = filters["label"]
        return [
            volume
            for volume in self.by_name.values()
            if not volume.removed
            and all(
                volume.attrs["Labels"].get(key) == value
                for key, value in (label.split("=", 1) for label in labels)
            )
        ]


class FakeContainers:
    def __init__(self) -> None:
        self.references: object = []
        self.list_error: Exception | None = None

    def list(self, *, all: bool, filters: dict[str, str]) -> object:
        assert all is True
        assert set(filters) == {"volume"}
        if self.list_error is not None:
            raise self.list_error
        return self.references


class FakeContainerReference:
    def __init__(
        self,
        container_id: str,
        *,
        run_id: str = "run-1",
        task_id: str = "task-1",
        role: str = "work",
    ) -> None:
        self.id = container_id
        self.attrs = {
            "Config": {
                "Labels": {
                    "rsi-harness.run-id": run_id,
                    "rsi-harness.task-id": task_id,
                    "rsi-harness.role": role,
                }
            }
        }

    def reload(self) -> None:
        return None


class FakeClient:
    def __init__(self) -> None:
        self.volumes = FakeVolumes()
        self.containers = FakeContainers()


def _create(backend: DockerWorkdirVolumeBackend) -> ManagedWorkdirVolume:
    planned = ManagedWorkdirVolume(
        name=backend.planned_name(run_id="run-1", task_id="task-1"),
        run_id="run-1",
        task_id="task-1",
        target=PurePosixPath("/workspace"),
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce=FRESHNESS_NONCE,
    )
    return backend.create(planned)


def _backend(client: FakeClient) -> DockerWorkdirVolumeBackend:
    return DockerWorkdirVolumeBackend(
        client, nonce_factory=lambda: FRESHNESS_NONCE
    )


def test_plan_mints_snapshot_target_and_freshness_bound_authority() -> None:
    """Dropping any field would let a stale identity alias a new split plan."""
    backend = _backend(FakeClient())

    planned = backend.plan(
        run_id="run-1",
        task_id="task-1",
        target=PurePosixPath("/workspace"),
    )

    assert planned == ManagedWorkdirVolume(
        name=backend.planned_name(run_id="run-1", task_id="task-1"),
        run_id="run-1",
        task_id="task-1",
        target=PurePosixPath("/workspace"),
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce=FRESHNESS_NONCE,
    )


def test_create_error_after_possible_mutation_retains_exact_recovery_authority(
) -> None:
    """A lost create response must never be downgraded to proven absence."""
    client = FakeClient()
    client.volumes.create_error = APIError("response lost")
    client.volumes.create_on_error = True
    backend = _backend(client)
    planned = backend.plan(
        run_id="run-1",
        task_id="task-1",
        target=PurePosixPath("/workspace"),
    )

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.create(planned)

    assert client.volumes.by_name[planned.name].attrs["Labels"] == EXPECTED_LABELS


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        (
            FakeContainerReference("work-1"),
            ContainerRef(container_id="work-1", role="work"),
        ),
        (
            FakeContainerReference("work-1", run_id="other-run"),
            ContainerRef(container_id="work-1", role="work"),
        ),
        (
            FakeContainerReference("work-1", role="helper"),
            ContainerRef(container_id="work-1", role="work"),
        ),
        (
            FakeContainerReference("extra"),
            ContainerRef(container_id="work-1", role="work"),
        ),
    ],
)
def test_inspect_enforces_exact_role_aware_reference_set(
    reference: FakeContainerReference, expected: ContainerRef
) -> None:
    """Wrong-label or extra references must not satisfy Work/Judge authority."""
    client = FakeClient()
    backend = _backend(client)
    volume = _create(backend)
    client.containers.references = [reference]

    if (
        reference.id == expected.container_id
        and reference.attrs["Config"]["Labels"]["rsi-harness.run-id"] == "run-1"
        and reference.attrs["Config"]["Labels"]["rsi-harness.role"] == "work"
    ):
        assert backend.inspect(volume, expected_references=(expected,)) == volume
    else:
        with pytest.raises(InfrastructureError, match="recovery_required"):
            backend.inspect(volume, expected_references=(expected,))


def test_create_attests_direct_and_exact_labeled_local_volume() -> None:
    """Dropping either attestation could accept a forged Docker volume."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)

    volume = _create(backend)

    assert volume == ManagedWorkdirVolume(
        name=backend.planned_name(run_id="run-1", task_id="task-1"),
        run_id="run-1",
        task_id="task-1",
        target=PurePosixPath("/workspace"),
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce=FRESHNESS_NONCE,
    )
    assert client.volumes.created == [
        {"name": volume.name, "driver": "local", "labels": EXPECTED_LABELS}
    ]


def test_create_rejects_preexisting_deterministic_exact_labeled_volume() -> None:
    """Removing the freshness check would adopt stale data through Docker create."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)
    planned = backend.planned_name(run_id="run-1", task_id="task-1")
    stale = FakeVolume(planned, driver="local", labels=EXPECTED_LABELS)
    client.volumes.by_name[planned] = stale

    with pytest.raises((SetupError, InfrastructureError), match="pre-existing|fresh"):
        _create(backend)

    assert client.volumes.by_name[planned] is stale


@pytest.mark.parametrize(
    ("scope", "options"),
    [
        ("global", {}),
        ("local", {"type": "none", "o": "bind", "device": "/host/data"}),
        (
            "local",
            {
                "type": "nfs",
                "o": "addr=192.0.2.10,rw",
                "device": ":/export/workspace",
            },
        ),
    ],
)
def test_inspect_rejects_nonlocal_or_option_backed_volume(
    scope: str, options: dict[str, str]
) -> None:
    """Relaxing local/empty-options checks could expose a host or NFS path RW."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)
    volume = _create(backend)
    docker_volume = client.volumes.by_name[volume.name]
    docker_volume.attrs["Scope"] = scope
    docker_volume.attrs["Options"] = options

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.inspect(volume)


def test_inspect_rejects_unexpected_container_reference() -> None:
    """Skipping reference checks could mount a volume already used elsewhere."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)
    volume = _create(backend)
    client.containers.references = [object()]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.inspect(volume)


def test_planned_name_is_deterministic_safe_and_task_bound() -> None:
    """Losing task identity in the name could alias two task workdirs."""
    backend = DockerWorkdirVolumeBackend(FakeClient())

    first = backend.planned_name(run_id="run one", task_id="task/one")
    second = backend.planned_name(run_id="run one", task_id="task/two")

    assert first.startswith("rsi-harness-workdir-")
    assert first == backend.planned_name(run_id="run one", task_id="task/one")
    assert first != second
    assert all(character.isalnum() or character in "_.-" for character in first)


def test_create_api_failure_with_post_error_absence_still_requires_recovery() -> None:
    """A create error is mutation-ambiguous even when immediate queries are empty."""
    client = FakeClient()
    client.volumes.create_error = APIError("create failed")
    backend = DockerWorkdirVolumeBackend(client)

    with pytest.raises(InfrastructureError, match="recovery_required"):
        _create(backend)


@pytest.mark.parametrize("boundary", ("create", "exact-query", "references"))
def test_unexpected_boundary_exception_retains_typed_recovery_authority(
    boundary: str,
) -> None:
    """Non-API SDK failures are still ambiguity, never permission to continue."""
    client = FakeClient()
    backend = _backend(client)
    volume = None
    if boundary == "create":
        client.volumes.create_error = RuntimeError("transport adapter failed")
    elif boundary == "exact-query":
        client.volumes.list_error = RuntimeError("query decoder failed")
    else:
        volume = _create(backend)
        client.containers.list_error = RuntimeError("reference decoder failed")

    with pytest.raises(InfrastructureError, match="recovery_required"):
        if volume is None:
            _create(backend)
        else:
            backend.inspect(volume)


def test_create_api_failure_with_discovered_exact_volume_requires_recovery() -> None:
    """A confirmed post-failure volume remains owned by its durable plan."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)
    planned = backend.planned_name(run_id="run-1", task_id="task-1")
    assert planned.startswith("rsi-harness-workdir-")
    client.volumes.create_error = APIError("response lost")
    client.volumes.create_on_error = True

    with pytest.raises(InfrastructureError, match="recovery_required"):
        _create(backend)
    assert planned in client.volumes.by_name


@pytest.mark.parametrize(
    "candidate",
    [
        [object()],
        [
            FakeVolume(
                "rsi-harness-workdir-one", driver="local", labels=EXPECTED_LABELS
            ),
            FakeVolume(
                "rsi-harness-workdir-two", driver="local", labels=EXPECTED_LABELS
            ),
        ],
    ],
)
def test_create_api_failure_with_ambiguous_or_forged_results_requires_recovery(
    candidate: object,
) -> None:
    """Accepting malformed discovery could lose authority over a created volume."""
    client = FakeClient()
    client.volumes.create_error = APIError("response lost")
    client.volumes.list_result = candidate
    backend = DockerWorkdirVolumeBackend(client)

    with pytest.raises(InfrastructureError, match="recovery_required"):
        _create(backend)


def test_remove_requires_exact_attestation_and_no_container_references() -> None:
    """Removing a referenced or forged volume could destroy live task state."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)
    volume = _create(backend)
    client.containers.references = [object()]

    with pytest.raises(InfrastructureError, match="container reference"):
        backend.remove(volume)
    assert client.volumes.by_name[volume.name].removed is False

    client.containers.references = []
    backend.remove(volume)
    assert client.volumes.by_name[volume.name].removed is True


@pytest.mark.parametrize("failure", ("get", "list", "remove"))
def test_remove_ambiguity_requires_recovery(failure: str) -> None:
    """An unproved removal state must retain the authority for recovery."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)
    volume = _create(backend)
    docker_volume = client.volumes.by_name[volume.name]
    if failure == "get":
        client.volumes.get_error = APIError("inspect failed")
    elif failure == "list":
        client.containers.list_error = APIError("list failed")
    else:
        def fail_remove(*, force: bool = False) -> None:
            del force
            raise APIError("remove failed")

        docker_volume.remove = fail_remove  # type: ignore[method-assign]

    with pytest.raises(InfrastructureError, match="recovery_required"):
        backend.remove(volume)


@pytest.mark.parametrize("operation", ("create-failure", "inspect", "remove"))
def test_malformed_direct_volume_view_requires_recovery(operation: str) -> None:
    """Raw malformed Docker views must not escape the recovery error boundary."""
    client = FakeClient()
    backend = DockerWorkdirVolumeBackend(client)
    planned = backend.planned_name(run_id="run-1", task_id="task-1")
    client.volumes.by_name[planned] = MalformedDirectVolume()  # type: ignore[assignment]
    client.volumes.list_result = []
    volume = ManagedWorkdirVolume(
        name=planned,
        run_id="run-1",
        task_id="task-1",
        target=PurePosixPath("/workspace"),
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce=FRESHNESS_NONCE,
    )
    if operation == "create-failure":
        client.volumes.create_error = APIError("response lost")

    with pytest.raises(InfrastructureError, match="recovery_required"):
        if operation == "create-failure":
            _create(backend)
        elif operation == "inspect":
            backend.inspect(volume)
        else:
            backend.remove(volume)


def _real_docker_client():
    try:
        client = docker_sdk.from_env()
        client.ping()
    except Exception as error:
        pytest.skip(f"local Docker volume capability unavailable: {error}")
    return client


def _assert_real_volume_absent(client, name: str) -> None:
    with pytest.raises(NotFound):
        client.volumes.get(name)


@pytest.mark.integration
def test_real_docker_rejects_preexisting_exact_freshness_identity() -> None:
    """The real Docker idempotent-create behavior must never adopt stale data."""
    client = _real_docker_client()
    suffix = uuid4().hex
    backend = DockerWorkdirVolumeBackend(client)
    planned = backend.plan(
        run_id=f"stale-real-{suffix}",
        task_id="task",
        target=PurePosixPath("/workspace"),
    )
    stale = None
    try:
        stale = client.volumes.create(
            planned.name,
            driver="local",
            labels=managed_workdir_volume_labels(planned),
        )

        with pytest.raises(InfrastructureError, match="pre-existing"):
            backend.create(planned)

        stale.reload()
        assert stale.attrs["Name"] == planned.name
    finally:
        if stale is not None:
            stale.remove(force=False)
    _assert_real_volume_absent(client, planned.name)


@pytest.mark.integration
def test_real_docker_rejects_local_bind_backed_volume_options(tmp_path) -> None:
    """A local driver can still expose a host path through non-empty options."""
    client = _real_docker_client()
    suffix = uuid4().hex
    host_source = tmp_path / "host-source"
    host_source.mkdir()
    backend = DockerWorkdirVolumeBackend(client)
    planned = backend.plan(
        run_id=f"options-real-{suffix}",
        task_id="task",
        target=PurePosixPath("/workspace"),
    )
    option_backed = None
    try:
        option_backed = client.volumes.create(
            planned.name,
            driver="local",
            driver_opts={
                "type": "none",
                "o": "bind",
                "device": str(host_source),
            },
            labels=managed_workdir_volume_labels(planned),
        )
        option_backed.reload()
        assert option_backed.attrs["Scope"] == "local"
        assert option_backed.attrs["Options"]

        with pytest.raises(InfrastructureError, match="recovery_required"):
            backend.inspect(planned)
    finally:
        if option_backed is not None:
            option_backed.remove(force=False)
    _assert_real_volume_absent(client, planned.name)


@pytest.mark.integration
def test_real_docker_rejects_unexpected_helper_volume_reference() -> None:
    """A stopped helper reference is still an exact conflict at the Docker boundary."""
    client = _real_docker_client()
    try:
        client.images.get("alpine:latest")
    except Exception as error:
        pytest.skip(f"cached Alpine image unavailable: {error}")
    suffix = uuid4().hex
    run_id = f"reference-real-{suffix}"
    backend = DockerWorkdirVolumeBackend(client)
    planned = backend.plan(
        run_id=run_id,
        task_id="task",
        target=PurePosixPath("/workspace"),
    )
    volume = None
    helper = None
    try:
        volume = backend.create(planned)
        helper = client.containers.create(
            "alpine:latest",
            ["sleep", "60"],
            labels={
                "rsi-harness.run-id": run_id,
                "rsi-harness.task-id": "task",
                "rsi-harness.role": "helper",
            },
            volumes={planned.name: {"bind": "/workspace", "mode": "rw"}},
        )

        with pytest.raises(InfrastructureError, match="reference"):
            backend.inspect(volume)
    finally:
        if helper is not None:
            helper.remove(force=True)
        if volume is not None:
            backend.remove(volume)
    _assert_real_volume_absent(client, planned.name)
