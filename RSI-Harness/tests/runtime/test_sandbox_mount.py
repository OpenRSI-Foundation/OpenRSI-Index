"""A broker socket grants one exact parent-only mount, not a generic root."""

from pathlib import PurePosixPath

import pytest

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import ContainerMount, ContainerSpec, ContainerTmpfs
from tests.fakes import FakeDockerClient
from tests.runtime.test_docker import make_runtime

TARGET = PurePosixPath("/run/rsi-harness/sandbox")


@pytest.mark.parametrize("role", ["work", "judge"])
def test_exact_phase_socket_directory_is_read_only_and_attested(tmp_path, role):
    directory = tmp_path / "engine-root" / "endpoint"
    directory.mkdir(parents=True)
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role=role, sandbox_socket_dir=directory)
    mount = ContainerMount(source=directory, target=TARGET, read_only=True)
    ref = runtime.create(ContainerSpec(image="parent", mounts=(mount,)))
    runtime.attest_sandbox_mount(ref)
    with pytest.raises(SetupError, match="sandbox"):
        runtime.create(
            ContainerSpec(
                image="parent", mounts=(mount.model_copy(update={"read_only": False}),)
            )
        )
    other = directory.parent / "other"
    other.mkdir()
    with pytest.raises(SetupError, match="sandbox"):
        runtime.create(
            ContainerSpec(
                image="parent", mounts=(mount.model_copy(update={"source": other}),)
            )
        )


def test_same_path_replaced_directory_does_not_retain_mount_authority(tmp_path):
    directory = tmp_path / "engine-root" / "endpoint"
    directory.mkdir(parents=True)
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, sandbox_socket_dir=directory)
    directory.rename(directory.with_name("old"))
    directory.mkdir()
    with pytest.raises(SetupError, match="sandbox.*identity"):
        runtime.create(
            ContainerSpec(
                image="parent",
                mounts=(
                    ContainerMount(source=directory, target=TARGET, read_only=True),
                ),
            )
        )
    assert not client.containers.created


def test_disabled_runtime_and_helpers_cannot_mount_broker_socket(tmp_path):
    directory = tmp_path / "engine-root" / "endpoint"
    directory.mkdir(parents=True)
    runtime = make_runtime(FakeDockerClient(), tmp_path)
    with pytest.raises(SetupError):
        runtime.create(
            ContainerSpec(
                image="parent",
                mounts=(
                    ContainerMount(source=directory, target=TARGET, read_only=True),
                ),
            )
        )
    with pytest.raises(SetupError, match="parent"):
        make_runtime(
            FakeDockerClient(), tmp_path, role="helper", sandbox_socket_dir=directory
        )


def test_actual_wrong_socket_mount_is_not_attested(tmp_path):
    directory = tmp_path / "engine-root" / "endpoint"
    directory.mkdir(parents=True)
    runtime = make_runtime(FakeDockerClient(), tmp_path, sandbox_socket_dir=directory)
    ref = runtime.create(ContainerSpec(image="parent"))
    with pytest.raises(InfrastructureError, match="sandbox.*mount"):
        runtime.attest_sandbox_mount(ref)


def test_parent_tmpfs_cannot_shadow_socket_mount(tmp_path):
    directory = tmp_path / "engine-root" / "endpoint"
    directory.mkdir(parents=True)
    runtime = make_runtime(FakeDockerClient(), tmp_path, sandbox_socket_dir=directory)
    with pytest.raises(SetupError, match="sandbox"):
        runtime.create(
            ContainerSpec(
                image="parent",
                tmpfs=(
                    ContainerTmpfs(target=PurePosixPath("/run"), options="size=1m"),
                ),
            )
        )
