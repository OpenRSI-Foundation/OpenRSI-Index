"""Tests staging for /tests injection, including read-only-rootfs Judges."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path, PurePosixPath

import docker
import pytest
from docker.errors import APIError

from rsi_harness.errors import InfrastructureError
from rsi_harness.models import ContainerMount, ContainerSpec, ContainerTmpfs
from tests.fakes import FakeDockerClient
from tests.runtime.test_docker import make_runtime

IMAGE = "sha256:" + "4" * 64
REAL_IMAGE = "python:3.12-slim-bookworm"


def create_judge(
    client,
    tmp_path,
    *,
    readonly=True,
    log_mount=True,
    log_readonly=False,
    tests_mode="0755",
    image=IMAGE,
):
    runtime = make_runtime(
        client, tmp_path, role="judge", omit_gpu_device_requests_for_tests=True
    )
    tests = tmp_path / "task-source/tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\nprintf 'BENIGN_INJECTION_OK\\n'\n")
    (tests / "test.sh").chmod(0o751)
    diagnostic = tests / "domain_checks.sh"
    diagnostic.write_text("printf 'NONROOT_DIAGNOSTIC_OK\\n'\n")
    diagnostic.chmod(0o644)
    nested = tests / "nested"
    nested.mkdir()
    (nested / "data.txt").write_text("benign fixture\n")
    logs = tmp_path / "engine-root/verifier"
    logs.mkdir(mode=0o700)
    private_log = logs / "private.log"
    private_log.write_text("protected verifier log\n")
    private_log.chmod(0o600)
    spec = ContainerSpec(
        image=image,
        command=("sleep", "infinity"),
        user="0",
        read_only_rootfs=readonly,
        mounts=(
            ContainerMount(
                source=logs,
                target=PurePosixPath("/logs/verifier"),
                read_only=log_readonly,
            ),
        )
        if log_mount
        else (),
        tmpfs=(
            ContainerTmpfs(
                target=PurePosixPath("/tests"),
                options=f"rw,exec,nosuid,nodev,mode={tests_mode}",
            ),
            ContainerTmpfs(
                target=PurePosixPath("/tmp"),
                options="rw,nosuid,nodev,size=32m,mode=1777",
            ),
        ),
    )
    ref = runtime.create(spec)
    return runtime, ref, tests, logs


@pytest.mark.parametrize("readonly", [False, True])
def test_docker_rootfs_options_are_explicit_opt_in(tmp_path, readonly):
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    runtime.create(
        ContainerSpec(
            image=IMAGE,
            read_only_rootfs=readonly,
            tmpfs=(
                ContainerTmpfs(
                    target=PurePosixPath("/tmp"),
                    options="rw,nosuid,nodev,size=256m,mode=1777",
                ),
            ),
        )
    )
    created = client.containers.created[0]
    assert created["tmpfs"] == {"/tmp": "rw,nosuid,nodev,size=256m,mode=1777"}
    assert created["privileged"] is False
    if readonly:
        assert created["read_only"] is True
        assert created["security_opt"] == ["no-new-privileges:true"]
    else:
        assert "read_only" not in created
        assert "security_opt" not in created


def test_container_spec_default_rootfs_is_writable():
    assert ContainerSpec(image=IMAGE).read_only_rootfs is False


@pytest.mark.parametrize("readonly", [False, True])
def test_test_archive_selects_daemon_visible_private_staging(tmp_path, readonly):
    client = FakeDockerClient()
    runtime, ref, tests, _ = create_judge(client, tmp_path, readonly=readonly)
    uploads = []

    def upload(path, stream):
        uploads.append((path, b"".join(stream)))
        return True

    client.containers.get(ref.container_id).put_archive = upload
    runtime.inject_directory(ref, tests, PurePosixPath("/tests"))
    [(staging, data)] = uploads
    parent = "/logs/verifier" if readonly else "/tmp"
    assert staging.startswith(parent + "/.rsi-harness-tests-")
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert archive.extractfile("nested/data.txt").read() == b"benign fixture\n"
        assert archive.getmember("test.sh").mode == 0o751
    prepared, populated, removed = [
        call["command"] for call in client.api.exec_create_calls
    ]
    assert prepared == ("/usr/bin/install", "-d", "-m", "0700", staging)
    assert populated[:2] == ("/bin/sh", "-c")
    assert populated[3:] == ("rsi-harness-copy-tests", staging, "/tests")
    assert removed == ("/bin/rm", "-rf", "--", staging)


@pytest.mark.parametrize("mount_present", [False, True])
def test_readonly_test_archive_requires_writable_verifier_mount(
    tmp_path, mount_present
):
    client = FakeDockerClient()
    runtime, ref, tests, _ = create_judge(
        client, tmp_path, log_mount=mount_present, log_readonly=True
    )
    with pytest.raises(InfrastructureError, match="writable.*verifier"):
        runtime.inject_directory(ref, tests, PurePosixPath("/tests"))
    assert client.api.exec_create_calls == []


def test_private_staging_removed_when_docker_upload_fails(tmp_path):
    client = FakeDockerClient()
    runtime, ref, tests, _ = create_judge(client, tmp_path)

    def upload(path, stream):
        b"".join(stream)
        raise APIError("benign upload failure")

    client.containers.get(ref.container_id).put_archive = upload
    with pytest.raises(InfrastructureError, match="failed to inject tests"):
        runtime.inject_directory(ref, tests, PurePosixPath("/tests"))
    commands = [call["command"] for call in client.api.exec_create_calls]
    assert commands[-1] == ("/bin/rm", "-rf", "--", commands[0][-1])


def test_private_staging_removed_when_copy_into_tests_fails(tmp_path):
    client = FakeDockerClient()
    runtime, ref, tests, _ = create_judge(client, tmp_path)

    def upload(path, stream):
        b"".join(stream)
        return True

    def inspect(_exec_id):
        command = client.api.exec_create_calls[-1]["command"]
        return {"ExitCode": 1 if command[0] == "/bin/sh" else 0}

    client.containers.get(ref.container_id).put_archive = upload
    client.api.exec_inspect = inspect
    with pytest.raises(InfrastructureError, match="failed to populate"):
        runtime.inject_directory(ref, tests, PurePosixPath("/tests"))
    commands = [call["command"] for call in client.api.exec_create_calls]
    assert commands[-1] == ("/bin/rm", "-rf", "--", commands[0][-1])


def real_image(client) -> str:
    try:
        client.ping()
        return client.images.get(REAL_IMAGE).id
    except Exception as error:
        pytest.skip(f"local cached {REAL_IMAGE} Docker image unavailable: {error}")


@pytest.mark.integration
@pytest.mark.parametrize("tests_mode", ["0755", "0700"])
def test_real_readonly_judge_injects_tests_through_verifier_bind(
    tmp_path: Path, tests_mode: str
):
    client = docker.from_env()
    image = real_image(client)
    runtime, ref, tests, logs = create_judge(
        client, tmp_path, image=image, tests_mode=tests_mode
    )
    container = client.containers.get(ref.container_id)
    try:
        container.start()
        upload_paths = []
        original_upload = client.api.put_archive

        def attest_staging(ident, path, stream):
            assert ident == ref.container_id
            assert path.startswith("/logs/verifier/.rsi-harness-tests-")
            private_mode = container.exec_run(["stat", "-c", "%u:%a", path])
            assert private_mode.exit_code == 0, private_mode.output
            assert private_mode.output == b"0:700\n"
            inaccessible = container.exec_run(
                ["/bin/test", "-x", path], user="65042:65042"
            )
            assert inaccessible.exit_code == 1, inaccessible.output
            upload_paths.append(path)
            return original_upload(ident, path, stream)

        client.api.put_archive = attest_staging
        runtime.inject_directory(ref, tests, PurePosixPath("/tests"))
        assert len(upload_paths) == 1
        diagnostic = container.exec_run(
            ["/bin/sh", "/tests/domain_checks.sh"], user="65042:65042"
        )
        if tests_mode == "0755":
            assert diagnostic.exit_code == 0, diagnostic.output
            assert diagnostic.output == b"NONROOT_DIAGNOSTIC_OK\n"
        else:
            assert diagnostic.exit_code != 0
            assert b"Permission denied" in diagnostic.output
        target_mode = container.exec_run(["stat", "-c", "%a", "/tests"])
        assert target_mode.exit_code == 0, target_mode.output
        assert target_mode.output == f"{tests_mode[1:]}\n".encode()
        private_log = container.exec_run(
            ["/bin/cat", "/logs/verifier/private.log"], user="65042:65042"
        )
        assert private_log.exit_code != 0
        assert b"Permission denied" in private_log.output
        result = container.exec_run(["/bin/sh", "/tests/test.sh"])
        assert result.exit_code == 0, result.output
        assert result.output == b"BENIGN_INJECTION_OK\n"
        result = container.exec_run(
            [
                "/bin/sh",
                "-c",
                'test "$(stat -c %a /tests/test.sh)" = 751 && '
                "test \"$(cat /tests/nested/data.txt)\" = 'benign fixture' && "
                'test -z "$(find /logs/verifier -maxdepth 1 '
                "-name '.rsi-harness-tests-*' -print)\"",
            ]
        )
        assert result.exit_code == 0, result.output
        assert list(logs.iterdir()) == [logs / "private.log"]
        assert (logs / "private.log").read_text() == "protected verifier log\n"
        assert (logs.stat().st_mode & 0o777) == 0o700
        assert ((logs / "private.log").stat().st_mode & 0o777) == 0o600
        container.reload()
        assert container.attrs["HostConfig"]["ReadonlyRootfs"] is True
        assert set(container.attrs["HostConfig"]["Tmpfs"]) == {"/tmp", "/tests"}
        assert not any(
            mount["Destination"] == "/tests" for mount in container.attrs["Mounts"]
        )
    finally:
        runtime.remove(ref)


@pytest.mark.integration
def test_real_failed_copy_preserves_target_mode_and_removes_staging(tmp_path: Path):
    client = docker.from_env()
    image = real_image(client)
    runtime, ref, tests, logs = create_judge(client, tmp_path, image=image)
    container = client.containers.get(ref.container_id)
    try:
        container.start()
        # A directory cannot be overwritten by the diagnostic regular file.
        collision = container.exec_run(["/bin/mkdir", "/tests/domain_checks.sh"])
        assert collision.exit_code == 0, collision.output
        with pytest.raises(InfrastructureError, match="failed to populate"):
            runtime.inject_directory(ref, tests, PurePosixPath("/tests"))
        target_mode = container.exec_run(["stat", "-c", "%a", "/tests"])
        assert target_mode.exit_code == 0, target_mode.output
        assert target_mode.output == b"755\n"
        assert list(logs.iterdir()) == [logs / "private.log"]
    finally:
        runtime.remove(ref)
