from __future__ import annotations

import io
import json
import logging
import os
import shutil
import socket
import stat
import subprocess
import tarfile
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath

import docker as docker_sdk
import pytest
from docker.errors import APIError, NotFound
from docker.types import DeviceRequest, Mount

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopBackendBridge,
    RSILoopContainerHandle,
)
from rsi_harness.models import (
    ContainerMount,
    ContainerRef,
    ContainerSpec,
    ContainerTmpfs,
    ContainerVolumeMount,
    GPUAllocation,
    GPUDevice,
    ManagedNetwork,
    ManagedWorkdirVolume,
    NetworkPolicy,
    RootfsSnapshotMode,
)
from rsi_harness.runtime import docker as docker_runtime
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.local_auth import (
    AgentAuthFile,
    AgentAuthMaterial,
    AgentAuthMount,
)
from rsi_harness.runtime.network import NetworkPolicyEnforcer, managed_bridge_interface
from rsi_harness.runtime.rootfs_snapshot import DockerRootfsSnapshotBackend
from rsi_harness.runtime.workdir_volume import (
    DockerWorkdirVolumeBackend,
    managed_workdir_volume_labels,
)
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeDockerClient, FakeDockerContainer, FakeFirewallBackend


def allocation() -> GPUAllocation:
    return GPUAllocation(
        devices=(
            GPUDevice(index=0, uuid="GPU-a", name="NVIDIA H100"),
            GPUDevice(index=2, uuid="GPU-c", name="NVIDIA H100"),
        )
    )


def make_runtime(
    client,
    tmp_path,
    *,
    role="work",
    run_id="r",
    task_id="t",
    networked=False,
    staging=False,
    **kwargs,
):
    task_source = tmp_path / "task-source"
    engine = tmp_path / "engine-root"
    task_source.mkdir(exist_ok=True)
    engine.mkdir(exist_ok=True)
    common = {
        "run_id": run_id,
        "task_id": task_id,
        "role": role,
        "task_source_dir": task_source,
        "allowed_mount_roots": (engine,),
        **kwargs,
    }
    if staging:
        common["staging_dir"] = engine / "staging"
    if networked:
        provisioner = DockerContainerRuntime(client, **common)
        common["network"] = provisioner.create_network(
            "phase", internal=role == "judge"
        )
    return DockerContainerRuntime(client, **common)


def test_create_passes_exact_gpu_uuids_typed_mounts_and_recovery_labels(tmp_path):
    client = FakeDockerClient()
    runtime = make_runtime(
        client, tmp_path, run_id="run-7", task_id="task-3", networked=True
    )
    workspace = tmp_path / "engine-root" / "workspace"
    workspace.mkdir()

    ref = runtime.create(
        ContainerSpec(
            image="work@sha256:abc",
            command=("sleep", "infinity"),
            workdir=PurePosixPath("/workspace"),
            environment=(("HOME", "/home/agent"),),
            mounts=(
                ContainerMount(
                    source=workspace.resolve(),
                    target=PurePosixPath("/run/rsi-harness/staging"),
                ),
            ),
            gpu_allocation=allocation(),
        )
    )

    assert ref == ContainerRef(container_id="container-1", role="work")
    create = client.containers.created[0]
    assert create["device_requests"] == [
        DeviceRequest(
            driver="nvidia",
            device_ids=["GPU-a", "GPU-c"],
            capabilities=[["gpu"]],
        )
    ]
    assert create["mounts"] == [
        Mount(
            source=str(workspace.resolve()),
            target="/run/rsi-harness/staging",
            type="bind",
            read_only=False,
        )
    ]
    assert create["network"] == "rsi-run-7-task-3-work-phase"
    assert create["privileged"] is False
    assert create["cap_drop"] == ["NET_RAW"]
    assert create["labels"] == {
        "rsi-harness.run-id": "run-7",
        "rsi-harness.task-id": "task-3",
        "rsi-harness.role": "work",
    }
    assert "devices" not in create
    assert "network_mode" not in create


def test_work_feedback_mount_requires_exact_read_only_authority(tmp_path):
    client = FakeDockerClient()
    feedback = tmp_path / "engine-root" / "feedback"
    feedback.mkdir(parents=True)
    runtime = make_runtime(client, tmp_path, work_feedback_dir=feedback)
    exact = ContainerMount(
        source=feedback,
        target=PurePosixPath("/run/rsi-harness/feedback"),
        read_only=True,
    )

    ref = runtime.create(ContainerSpec(image="work", mounts=(exact,)))
    runtime.attest_work_feedback_mount(ref)

    assert client.containers.created[0]["mounts"] == [
        Mount(
            source=str(feedback.resolve()),
            target="/run/rsi-harness/feedback",
            type="bind",
            read_only=True,
        )
    ]
    other = tmp_path / "engine-root" / "other"
    other.mkdir()
    with pytest.raises(SetupError, match="feedback mount differs"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(exact.model_copy(update={"read_only": False}),),
            )
        )
    with pytest.raises(SetupError, match="feedback mount differs"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(exact.model_copy(update={"source": other}),),
            )
        )


def test_work_feedback_authority_rejects_symlinked_directory(tmp_path):
    client = FakeDockerClient()
    engine = tmp_path / "engine-root"
    engine.mkdir()
    redirected = engine / "redirected"
    redirected.mkdir()
    feedback = engine / "feedback"
    feedback.symlink_to(redirected, target_is_directory=True)

    with pytest.raises(SetupError, match="feedback.*symlink"):
        make_runtime(client, tmp_path, work_feedback_dir=feedback)


@pytest.mark.parametrize(
    "actual_mounts",
    (
        (),
        (
            {
                "Type": "bind",
                "Source": "{feedback}",
                "Destination": "/run/rsi-harness/feedback",
                "RW": True,
            },
        ),
        (
            {
                "Type": "bind",
                "Source": "{other}",
                "Destination": "/run/rsi-harness/feedback",
                "RW": False,
            },
        ),
        (
            {
                "Type": "bind",
                "Source": "{feedback}",
                "Destination": "/run/rsi-harness/feedback",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": "{feedback}",
                "Destination": "/run/rsi-harness/feedback",
                "RW": False,
            },
        ),
    ),
)
def test_work_feedback_mount_is_attested_from_created_container(
    tmp_path, actual_mounts
):
    client = FakeDockerClient()
    feedback = tmp_path / "engine-root" / "feedback"
    feedback.mkdir(parents=True)
    other = tmp_path / "engine-root" / "other"
    other.mkdir()
    runtime = make_runtime(client, tmp_path, work_feedback_dir=feedback)
    ref = runtime.create(
        ContainerSpec(
            image="work",
            mounts=(
                ContainerMount(
                    source=feedback,
                    target=PurePosixPath("/run/rsi-harness/feedback"),
                    read_only=True,
                ),
            ),
        )
    )
    container = client.containers.by_id[ref.container_id]
    container.attrs["Mounts"] = [
        {
            key: value.format(feedback=feedback.resolve(), other=other.resolve())
            if isinstance(value, str)
            else value
            for key, value in mount.items()
        }
        for mount in actual_mounts
    ]

    with pytest.raises(InfrastructureError, match="feedback mount attestation"):
        runtime.attest_work_feedback_mount(ref)


@pytest.mark.parametrize(
    ("role", "gpu_allocation", "expected_visibility"),
    (
        ("work", allocation(), "GPU-a,GPU-c"),
        ("work", GPUAllocation(), "void"),
        ("judge", GPUAllocation(), "void"),
    ),
)
def test_create_owns_gpu_visibility_over_task_and_inherited_image_environment(
    tmp_path, role, gpu_allocation, expected_visibility
):
    """Removing the reserved override could expose image-default host GPUs."""
    client = FakeDockerClient()
    image = f"{role}-snapshot-with-visible-all"
    client.containers.image_environments[image] = {
        "IMAGE_ONLY": "preserved",
        "NVIDIA_VISIBLE_DEVICES": "all",
    }
    runtime = make_runtime(client, tmp_path, role=role)

    ref = runtime.create(
        ContainerSpec(
            image=image,
            environment=(
                ("TASK_ONLY", "preserved"),
                ("NVIDIA_VISIBLE_DEVICES", "all"),
            ),
            gpu_allocation=gpu_allocation,
        )
    )

    create = client.containers.created[0]
    assert create["environment"] == {
        "TASK_ONLY": "preserved",
        "NVIDIA_VISIBLE_DEVICES": expected_visibility,
    }
    config_environment = dict(
        value.split("=", 1)
        for value in client.containers.get(ref.container_id).attrs["Config"]["Env"]
    )
    assert config_environment == {
        "IMAGE_ONLY": "preserved",
        "NVIDIA_VISIBLE_DEVICES": expected_visibility,
        "TASK_ONLY": "preserved",
    }
    expected_requests = (
        []
        if not gpu_allocation.devices
        else [
            DeviceRequest(
                driver="nvidia",
                device_ids=["GPU-a", "GPU-c"],
                capabilities=[["gpu"]],
            )
        ]
    )
    assert create["device_requests"] == expected_requests


@pytest.mark.parametrize("uuid", ("all", "GPU-a,GPU-b", "GPU-a\nGPU-b"))
def test_create_rejects_gpu_uuid_that_is_ambiguous_in_visibility_environment(
    tmp_path, uuid
):
    """An ambiguous identifier must never become an NVIDIA control value."""
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path)

    with pytest.raises(SetupError, match="GPU UUID.*visibility"):
        runtime.create(
            ContainerSpec(
                image="work-image",
                gpu_allocation=GPUAllocation(
                    devices=(GPUDevice(index=0, uuid=uuid, name="Test GPU"),)
                ),
            )
        )

    assert client.containers.created == []


def _workdir_volume(*, run_id: str = "r", task_id: str = "t") -> ManagedWorkdirVolume:
    return ManagedWorkdirVolume(
        name="rsi-harness-workdir-7af9cc65bec820f44338dbcde5021a5b92d31c8774c740b0150b49f7a05ea69e",
        run_id=run_id,
        task_id=task_id,
        target=PurePosixPath("/workspace"),
        snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        freshness_nonce="f" * 64,
    )


def _install_workdir_volume(client, volume: ManagedWorkdirVolume, *, labels=None):
    client.volumes.create(
        volume.name,
        driver="local",
        labels=labels or managed_workdir_volume_labels(volume),
    )


def _install_work_volume_reference(client, volume: ManagedWorkdirVolume):
    reference = FakeDockerContainer("work-reference")
    reference.attrs["Config"]["Labels"] = {
        "rsi-harness.run-id": volume.run_id,
        "rsi-harness.task-id": volume.task_id,
        "rsi-harness.role": "work",
    }
    reference.attrs["Mounts"] = [
        {
            "Type": "volume",
            "Name": volume.name,
            "Source": volume.name,
            "Destination": str(volume.target),
            "RW": True,
        }
    ]
    client.containers.by_id[reference.id] = reference
    return ContainerRef(container_id=reference.id, role="work")


@pytest.mark.parametrize(
    ("role", "read_only"), (("work", False), ("judge", True))
)
def test_create_passes_exact_attested_workdir_volume_with_role_access(
    tmp_path, role, read_only
):
    """Changing volume type, source, target, or access would expose task state."""
    client = FakeDockerClient()
    volume = _workdir_volume()
    _install_workdir_volume(client, volume)
    work_reference = (
        _install_work_volume_reference(client, volume)
        if role == "judge"
        else None
    )
    runtime = make_runtime(
        client,
        tmp_path,
        role=role,
        workdir_volume_references=(
            () if work_reference is None else (work_reference,)
        ),
    )

    runtime.create(
        ContainerSpec(
            image="image",
            workdir=PurePosixPath("/workspace"),
            volume_mounts=(
                ContainerVolumeMount(
                    volume=volume,
                    target=PurePosixPath("/workspace"),
                    read_only=read_only,
                ),
            ),
        )
    )

    create = client.containers.created[0]
    assert create["mounts"] == [
        Mount(
            source=volume.name,
            target="/workspace",
            type="volume",
            read_only=read_only,
        )
    ]


@pytest.mark.parametrize(
    ("role", "read_only"),
    (("work", True), ("judge", False), ("helper", False)),
)
def test_create_rejects_workdir_volume_access_not_authorized_for_role(
    tmp_path, role, read_only
):
    """A role-access inversion would let Judge modify Work's only writable state."""
    client = FakeDockerClient()
    volume = _workdir_volume()
    _install_workdir_volume(client, volume)
    runtime = make_runtime(client, tmp_path, role=role)

    with pytest.raises(SetupError, match="role|access|only valid"):
        runtime.create(
            ContainerSpec(
                image="image",
                workdir=PurePosixPath("/workspace"),
                volume_mounts=(
                    ContainerVolumeMount(
                        volume=volume,
                        target=PurePosixPath("/workspace"),
                        read_only=read_only,
                    ),
                ),
            )
        )
    assert client.containers.created == []


@pytest.mark.parametrize(
    "case",
    ("wrong-workdir", "forged-labels", "missing", "docker-socket-name"),
)
def test_create_rejects_unattested_or_misplaced_workdir_volume_before_docker_create(
    tmp_path, case
):
    """Skipping authority checks would allow a forged volume to replace WORKDIR."""
    client = FakeDockerClient()
    volume = _workdir_volume()
    workdir = PurePosixPath("/workspace")
    if case == "forged-labels":
        _install_workdir_volume(
            client,
            volume,
            labels={
                "rsi-harness.run-id": "other",
                "rsi-harness.task-id": "t",
                "rsi-harness.role": "workdir-volume",
            },
        )
    elif case != "missing":
        _install_workdir_volume(client, volume)
    if case == "wrong-workdir":
        workdir = PurePosixPath("/other")
    if case == "docker-socket-name":
        volume = ManagedWorkdirVolume(
            name="docker.sock",
            run_id="r",
            task_id="t",
            target=PurePosixPath("/workspace"),
            snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
            freshness_nonce="f" * 64,
        )
    runtime = make_runtime(client, tmp_path)

    with pytest.raises(
        (SetupError, InfrastructureError),
        match="WORKDIR|authority|socket|recovery_required",
    ):
        runtime.create(
            ContainerSpec(
                image="image",
                workdir=workdir,
                volume_mounts=(
                    ContainerVolumeMount(
                        volume=volume,
                        target=PurePosixPath("/workspace"),
                        read_only=False,
                    ),
                ),
            )
        )
    assert client.containers.created == []


@pytest.mark.parametrize("role", ("work", "judge", "helper"))
def test_create_rejects_bind_mount_at_declared_workdir_for_every_role(tmp_path, role):
    """A bind at WORKDIR would bypass the managed-volume access authority."""
    client = FakeDockerClient()
    source = tmp_path / "engine-root" / "workspace"
    source.mkdir(parents=True)
    runtime = make_runtime(client, tmp_path, role=role)

    with pytest.raises(SetupError, match="WORKDIR.*managed volume"):
        runtime.create(
            ContainerSpec(
                image="image",
                workdir=PurePosixPath("/workspace"),
                mounts=(
                    ContainerMount(
                        source=source, target=PurePosixPath("/workspace")
                    ),
                ),
            )
        )
    assert client.containers.created == []


@pytest.mark.parametrize(
    "target",
    (
        "/tests",
        "/tests/cases",
        "/logs",
        "/logs/verifier",
        "/logs/verifier/output",
        "/run",
        "/run/rsi-harness",
        "/run/rsi-harness/staging",
        "/run/rsi-harness/staging/cache",
    ),
)
def test_runtime_rejects_split_workdir_engine_mount_overlap_before_side_effects(
    tmp_path, target
):
    """Runtime defense must run before staging creation or Docker create."""
    client = FakeDockerClient()
    volume = _workdir_volume().model_copy(
        update={"target": PurePosixPath(target)}
    )
    _install_workdir_volume(client, volume)
    runtime = make_runtime(client, tmp_path, role="work", staging=True)

    with pytest.raises(SetupError, match="mount topology"):
        runtime.create(
            ContainerSpec(
                image="image",
                workdir=volume.target,
                volume_mounts=(
                    ContainerVolumeMount(
                        volume=volume,
                        target=volume.target,
                        read_only=False,
                    ),
                ),
            )
        )

    assert not (tmp_path / "engine-root" / "staging").exists()
    assert client.containers.created == []


def test_judge_keeps_non_workdir_verifier_log_bind(tmp_path):
    """Rejecting all Judge binds would prevent verifier logs from being collected."""
    client = FakeDockerClient()
    logs = tmp_path / "engine-root" / "logs"
    logs.mkdir(parents=True)
    runtime = make_runtime(client, tmp_path, role="judge")

    runtime.create(
        ContainerSpec(
            image="image",
            workdir=PurePosixPath("/workspace"),
            mounts=(
                ContainerMount(
                    source=logs,
                    target=PurePosixPath("/logs/verifier"),
                ),
            ),
        )
    )
    assert client.containers.created[0]["mounts"] == [
        Mount(
            source=str(logs),
            target="/logs/verifier",
            type="bind",
            read_only=False,
        )
    ]


@pytest.mark.parametrize("query_view", ("duplicate", "forged"))
def test_create_rejects_ambiguous_workdir_volume_label_query_before_docker_create(
    tmp_path, query_view
):
    """Direct-only inspection would accept a forged or duplicate label-query view."""
    client = FakeDockerClient()
    volume = _workdir_volume()
    _install_workdir_volume(client, volume)
    if query_view == "duplicate":
        client.volumes.create(
            "rsi-harness-workdir-other",
            driver="local",
            labels=managed_workdir_volume_labels(volume),
        )
    else:
        client.volumes.list_result = [object()]
    runtime = make_runtime(client, tmp_path)

    with pytest.raises(InfrastructureError, match="recovery_required"):
        runtime.create(
            ContainerSpec(
                image="image",
                workdir=PurePosixPath("/workspace"),
                volume_mounts=(
                    ContainerVolumeMount(
                        volume=volume,
                        target=PurePosixPath("/workspace"),
                        read_only=False,
                    ),
                ),
            )
        )
    assert client.containers.created == []


@pytest.mark.integration
def test_real_docker_volume_is_rw_for_work_ro_for_judge_and_removed(tmp_path):
    """A writable Judge mount or leaked volume would break task isolation."""
    try:
        client = docker_sdk.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local Docker volume capability unavailable: {error}")
    task_source = tmp_path / "task-source"
    engine = tmp_path / "engine"
    task_source.mkdir()
    engine.mkdir()
    backend = DockerWorkdirVolumeBackend(client)
    planned = backend.plan(
        run_id="real-volume",
        task_id="task",
        target=PurePosixPath("/workspace"),
    )
    volume = backend.create(planned)
    work_runtime = DockerContainerRuntime(
        client,
        run_id="real-volume",
        task_id="task",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )
    judge_runtime = None
    work = judge = None
    try:
        work = work_runtime.create(
            ContainerSpec(
                image="ubuntu:24.04",
                command=("sleep", "infinity"),
                workdir=PurePosixPath("/workspace"),
                volume_mounts=(
                    ContainerVolumeMount(
                        volume=volume,
                        target=PurePosixPath("/workspace"),
                        read_only=False,
                    ),
                ),
            )
        )
        client.containers.get(work.container_id).start()
        assert client.containers.get(work.container_id).exec_run(
            ["sh", "-c", "printf work > /workspace/probe"]
        ).exit_code == 0
        judge_runtime = DockerContainerRuntime(
            client,
            run_id="real-volume",
            task_id="task",
            role="judge",
            task_source_dir=task_source,
            allowed_mount_roots=(engine,),
            workdir_volume_references=(work,),
        )
        judge = judge_runtime.create(
            ContainerSpec(
                image="ubuntu:24.04",
                command=("sleep", "infinity"),
                workdir=PurePosixPath("/workspace"),
                volume_mounts=(
                    ContainerVolumeMount(
                        volume=volume,
                        target=PurePosixPath("/workspace"),
                        read_only=True,
                    ),
                ),
            )
        )
        client.containers.get(judge.container_id).start()
        assert client.containers.get(judge.container_id).exec_run(
            ["sh", "-c", "printf judge > /workspace/probe"]
        ).exit_code != 0
    finally:
        if judge is not None and judge_runtime is not None:
            judge_runtime.remove(judge)
        if work is not None:
            work_runtime.remove(work)
        backend.remove(volume)
    assert client.volumes.list(
        filters={
            "label": [
                f"{key}={value}"
                for key, value in sorted(
                    managed_workdir_volume_labels(volume).items()
                )
            ]
        }
    ) == []


def test_gpu_device_request_can_only_be_omitted_by_explicit_test_injection(
    tmp_path,
):
    client = FakeDockerClient()
    runtime = make_runtime(
        client,
        tmp_path,
        omit_gpu_device_requests_for_tests=True,
    )

    runtime.create(
        ContainerSpec(
            image="work@sha256:abc",
            gpu_allocation=allocation(),
        )
    )

    assert client.containers.created[0]["device_requests"] == []


def test_create_passes_validated_shm_size_to_docker(tmp_path) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path)

    runtime.create(ContainerSpec(image="work@sha256:abc", shm_size="2g"))

    assert client.containers.created[0]["shm_size"] == "2g"


def test_create_applies_official_harbor_docker_resource_limits(tmp_path) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path)

    runtime.create(
        ContainerSpec(
            image="work@sha256:abc",
            cpus=2,
            memory_mb=8192,
            storage_mb=15360,
        )
    )

    create = client.containers.created[0]
    assert create["nano_cpus"] == 2_000_000_000
    assert create["mem_limit"] == "8192m"
    assert "storage_opt" not in create


def test_create_maps_private_executable_tests_tmpfs_to_docker(tmp_path) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")

    runtime.create(
        ContainerSpec(
            image="snapshot-image",
            tmpfs=(
                ContainerTmpfs(
                    target=PurePosixPath("/tests"),
                    options="rw,exec,nosuid,nodev,mode=0755",
                ),
            ),
        )
    )

    assert client.containers.created[0]["tmpfs"] == {
        "/tests": "rw,exec,nosuid,nodev,mode=0755"
    }


def test_inject_agent_auth_uses_declared_work_tmpfs_and_memory_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="work")
    options = "rw,nosuid,nodev,noexec,mode=0700"
    ref = runtime.create(
        ContainerSpec(
            image="work-image",
            tmpfs=(
                ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options=options,
                ),
            ),
        )
    )
    container = client.containers.by_id[ref.container_id]
    container.attrs["HostConfig"] = {
        "Tmpfs": {"/home/agent/.codex": options}
    }
    container.attrs["Mounts"].append(
        {
            "Type": "tmpfs",
            "Source": "",
            "Destination": "/home/agent/.codex",
            "RW": True,
        }
    )
    secret_document = b'{"tokens":{"access_token":"LOCAL-LOGIN-SECRET"}}'
    auth = AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options=options,
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=secret_document,
                        mode=0o600,
                    ),
                ),
            ),
        ),
        secret_values=frozenset({"LOCAL-LOGIN-SECRET"}),
    )

    streamed = []

    def capture_stream(self, work, *, target, content, mode):
        streamed.append((self, work, target, content, mode))

    monkeypatch.setattr(
        DockerContainerRuntime, "_stream_agent_auth_file", capture_stream
    )

    runtime.inject_agent_auth(ref, auth)

    assert streamed == [
        (
            runtime,
            ref,
            PurePosixPath("/home/agent/.codex/auth.json"),
            secret_document,
            0o600,
        )
    ]
    assert [call["command"] for call in client.api.exec_create_calls] == [
        ("/bin/chmod", "0700", "--", "/home/agent/.codex"),
        (
            "/bin/sh",
            "-c",
            'mode="$(/usr/bin/stat -c %a:%u:%g -- "$1")" && '
            '[ "$mode" = "700:0:0" ]',
            "rsi-agent-auth-directory",
            "/home/agent/.codex",
        ),
    ]
    assert secret_document not in repr(client.api.exec_create_calls).encode()


@pytest.mark.parametrize("role", ("judge", "helper"))
def test_inject_agent_auth_rejects_non_work_roles(tmp_path: Path, role: str) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role=role)
    ref = runtime.create(ContainerSpec(image="image"))

    with pytest.raises(SetupError, match="Work"):
        runtime.inject_agent_auth(
            ref,
            AgentAuthMaterial(agent_name="codex", mounts=(), secret_values=frozenset()),
        )


def test_inject_agent_auth_rejects_work_without_exact_tmpfs(tmp_path: Path) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="work")
    ref = runtime.create(ContainerSpec(image="work-image"))

    with pytest.raises(InfrastructureError, match="tmpfs"):
        runtime.inject_agent_auth(
            ref,
            AgentAuthMaterial(
                agent_name="codex",
                mounts=(
                    AgentAuthMount(
                        tmpfs=ContainerTmpfs(
                            target=PurePosixPath("/home/agent/.codex"),
                            options="rw,nosuid,nodev,noexec,mode=0700",
                        ),
                        files=(
                            AgentAuthFile(
                                path=PurePosixPath("auth.json"),
                                content=b"{}",
                                mode=0o600,
                            ),
                        ),
                    ),
                ),
                secret_values=frozenset(),
            ),
        )


@pytest.mark.integration
def test_real_agent_auth_tmpfs_is_excluded_from_rootfs_snapshot(
    tmp_path: Path,
) -> None:
    try:
        client = docker_sdk.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local Ubuntu Docker capability unavailable: {error}")

    run_id = f"agent-auth-{time.time_ns()}"
    options = "rw,nosuid,nodev,noexec,mode=0700"
    secret = "SYNTHETIC-LOCAL-AUTH-MUST-NOT-BE-SNAPSHOTTED"
    material = AgentAuthMaterial(
        agent_name="codex",
        mounts=(
            AgentAuthMount(
                tmpfs=ContainerTmpfs(
                    target=PurePosixPath("/home/agent/.codex"),
                    options=options,
                ),
                files=(
                    AgentAuthFile(
                        path=PurePosixPath("auth.json"),
                        content=json.dumps({"access_token": secret}).encode(),
                        mode=0o600,
                    ),
                ),
            ),
        ),
        secret_values=frozenset({secret}),
    )
    task_source = tmp_path / "task-source"
    engine_root = tmp_path / "engine-root"
    task_source.mkdir()
    engine_root.mkdir()
    runtime = DockerContainerRuntime(
        client,
        run_id=run_id,
        task_id="local-auth",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine_root,),
        omit_gpu_device_requests_for_tests=True,
    )
    snapshots = DockerRootfsSnapshotBackend(client)
    planned_ref = snapshots.planned_ref(
        run_id=run_id,
        task_id="local-auth",
        round_id="1",
        purpose="judge-round",
    )
    work = judge = lease = None
    try:
        work = runtime.create(
            ContainerSpec(
                image="ubuntu:24.04",
                command=("sleep", "infinity"),
                user="0",
                tmpfs=tuple(mount.tmpfs for mount in material.mounts),
            )
        )
        work_container = client.containers.get(work.container_id)
        work_container.start()
        runtime.inject_agent_auth(work, material)
        visible = work_container.exec_run(
            [
                "/bin/sh",
                "-c",
                "test -s /home/agent/.codex/auth.json && "
                "test \"$(stat -c %a /home/agent/.codex/auth.json)\" = 600",
            ]
        )
        assert visible.exit_code == 0, visible.output

        work_container.pause()
        lease = snapshots.acquire(
            work,
            run_id=run_id,
            task_id="local-auth",
            round_id="1",
            planned_ref=planned_ref,
        )
        assert secret not in json.dumps(
            client.images.get(lease.image_id).attrs, sort_keys=True
        )
        judge = client.containers.create(
            lease.image_id,
            ["sleep", "infinity"],
            network_mode="none",
        )
        judge.start()
        absent = judge.exec_run(
            ["test", "!", "-e", "/home/agent/.codex/auth.json"]
        )
        assert absent.exit_code == 0, absent.output
    finally:
        if judge is not None:
            judge.remove(force=True)
        if lease is not None:
            snapshots.release(lease)
        if work is not None:
            runtime.remove(work)


def test_inject_directory_streams_modes_and_symlinks_without_mutating_source(
    tmp_path,
) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    tests_dir = tmp_path / "task-source" / "tests"
    nested = tests_dir / "nested"
    nested.mkdir(parents=True)
    script = tests_dir / "test.sh"
    helper = nested / "helper.txt"
    script.write_text("#!/bin/bash\n")
    helper.write_text("helper\n")
    script.chmod(0o751)
    nested.chmod(0o750)
    helper.chmod(0o640)
    (nested / "helper-link").symlink_to("helper.txt")
    source_before = (
        hash_tree(tests_dir),
        tests_dir.stat().st_ino,
        tests_dir.stat().st_mtime_ns,
    )
    ref = runtime.create(ContainerSpec(image="snapshot-image"))
    uploaded: list[tuple[str, bytes, bool]] = []

    def capture(path, data):
        declared_length = (
            os.fstat(data.fileno()).st_size if hasattr(data, "fileno") else None
        )
        archive = (
            data.read(declared_length)
            if declared_length is not None
            else b"".join(data)
        )
        uploaded.append((path, archive, not isinstance(data, bytes)))
        return True

    client.containers.by_id[ref.container_id].put_archive = capture

    runtime.inject_directory(ref, tests_dir, PurePosixPath("/tests"))

    assert source_before == (
        hash_tree(tests_dir),
        tests_dir.stat().st_ino,
        tests_dir.stat().st_mtime_ns,
    )
    [(target, archive, streamed)] = uploaded
    assert target.startswith("/tmp/.rsi-harness-tests-")
    assert ".." not in target
    assert streamed is True
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as payload:
        members = {member.name: member for member in payload.getmembers()}
        assert stat.S_IMODE(members["test.sh"].mode) == 0o751
        assert members["nested"].isdir()
        assert stat.S_IMODE(members["nested"].mode) == 0o750
        assert stat.S_IMODE(members["nested/helper.txt"].mode) == 0o640
        assert members["nested/helper-link"].issym()
        assert members["nested/helper-link"].linkname == "helper.txt"
        assert payload.extractfile("nested/helper.txt").read() == b"helper\n"
    commands = [call["command"] for call in client.api.exec_create_calls]
    assert commands == [
        ("/usr/bin/install", "-d", "-m", "0700", target),
        ("/bin/cp", "-a", f"{target}/.", "/tests"),
        ("/bin/rm", "-rf", "--", target),
    ]


def test_inject_directory_never_opens_outside_content_during_directory_swap(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rsi_harness.runtime.docker as docker_module

    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    tests_dir = tmp_path / "task-source" / "tests"
    nested = tests_dir / "nested"
    nested.mkdir(parents=True)
    (tests_dir / "aaa-padding").write_bytes(b"p" * (2 * 1024 * 1024))
    (nested / "payload.txt").write_text("inside\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_payload = outside / "payload.txt"
    outside_payload.write_text("OUTSIDE-SECRET\n")
    outside_identity = (outside_payload.stat().st_dev, outside_payload.stat().st_ino)
    outside_opened = False
    real_open = os.open

    def traced_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal outside_opened
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        identity = os.fstat(descriptor)
        if (identity.st_dev, identity.st_ino) == outside_identity:
            outside_opened = True
        return descriptor

    monkeypatch.setattr(docker_module.os, "open", traced_open)
    ref = runtime.create(ContainerSpec(image="snapshot-image"))
    container = client.containers.by_id[ref.container_id]
    archived = b""

    def swap_then_capture(_path, data):
        nonlocal archived
        original = tests_dir / "nested-original"
        nested.rename(original)
        nested.symlink_to(outside, target_is_directory=True)
        try:
            archived = b"".join(data)
        finally:
            nested.unlink()
            original.rename(nested)
        return True

    container.put_archive = swap_then_capture

    with pytest.raises(SetupError, match="changed|cannot open"):
        runtime.inject_directory(ref, tests_dir, PurePosixPath("/tests"))

    assert outside_opened is False
    assert b"OUTSIDE-SECRET" not in archived


@pytest.mark.parametrize("entry_kind", ["fifo", "socket"])
def test_inject_directory_rejects_special_files_before_docker_mutation(
    tmp_path, entry_kind
) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    tests_dir = tmp_path / "task-source" / "tests"
    tests_dir.mkdir()
    special = tests_dir / "special"
    open_socket = None
    if entry_kind == "fifo":
        os.mkfifo(special)
    else:
        open_socket = socket.socket(socket.AF_UNIX)
        open_socket.bind(str(special))
    ref = runtime.create(ContainerSpec(image="snapshot-image"))
    uploads = 0

    def capture(_path, _data):
        nonlocal uploads
        uploads += 1
        return True

    client.containers.by_id[ref.container_id].put_archive = capture
    try:
        with pytest.raises(SetupError, match="unsupported.*entry"):
            runtime.inject_directory(ref, tests_dir, PurePosixPath("/tests"))
    finally:
        if open_socket is not None:
            open_socket.close()

    assert uploads == 0


def test_inject_directory_rejects_device_before_docker_mutation(
    tmp_path, monkeypatch
) -> None:
    import rsi_harness.runtime.docker as docker_module

    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    tests_dir = tmp_path / "task-source" / "tests"
    tests_dir.mkdir()
    ref = runtime.create(ContainerSpec(image="snapshot-image"))
    uploads = 0

    class DeviceEntry:
        name = "device"
        path = str(tests_dir / name)

        @staticmethod
        def stat(*, follow_symlinks):
            assert follow_symlinks is False
            return os.stat_result((stat.S_IFCHR | 0o600,) + (0,) * 9)

    class DeviceScan:
        def __enter__(self):
            return iter((DeviceEntry(),))

        def __exit__(self, *_errors):
            return False

    def capture(_path, _data):
        nonlocal uploads
        uploads += 1
        return True

    client.containers.by_id[ref.container_id].put_archive = capture
    monkeypatch.setattr(docker_module.os, "scandir", lambda _path: DeviceScan())

    with pytest.raises(SetupError, match="unsupported.*entry"):
        runtime.inject_directory(ref, tests_dir, PurePosixPath("/tests"))

    assert uploads == 0


def test_inject_directory_rejects_outside_source_and_lexical_traversal(
    tmp_path,
) -> None:
    runtime = make_runtime(FakeDockerClient(), tmp_path, role="judge")
    tests_dir = tmp_path / "task-source" / "tests"
    tests_dir.mkdir()
    outside = tmp_path / "engine-root" / "outside-tests"
    outside.mkdir()
    ref = ContainerRef(container_id="judge", role="judge")

    with pytest.raises(SetupError, match="exact task tests"):
        runtime.inject_directory(ref, outside, PurePosixPath("/tests"))
    with pytest.raises(SetupError, match="traversal"):
        runtime.inject_directory(
            ref,
            tests_dir / "nested" / "..",
            PurePosixPath("/tests"),
        )
    with pytest.raises(SetupError, match="traversal"):
        runtime.inject_directory(
            ref,
            tests_dir,
            PurePosixPath("/tests/../workspace"),
        )


def test_inject_directory_enforces_entry_and_regular_byte_budgets(
    tmp_path, monkeypatch
) -> None:
    import rsi_harness.runtime.docker as docker_module

    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    tests_dir = tmp_path / "task-source" / "tests"
    tests_dir.mkdir()
    ref = runtime.create(ContainerSpec(image="snapshot-image"))
    container = client.containers.by_id[ref.container_id]
    uploads = 0

    def capture(_path, _data):
        nonlocal uploads
        uploads += 1
        return True

    container.put_archive = capture
    monkeypatch.setattr(docker_module, "_MAX_INJECTED_TEST_ENTRIES", 2)
    for name in ("one", "two", "three"):
        (tests_dir / name).write_text(name)
    with pytest.raises(SetupError, match="entry limit"):
        runtime.inject_directory(ref, tests_dir, PurePosixPath("/tests"))

    for child in tests_dir.iterdir():
        child.unlink()
    oversized = tests_dir / "oversized"
    with oversized.open("wb") as stream:
        stream.truncate(1_073_741_825)
    monkeypatch.setattr(docker_module, "_MAX_INJECTED_TEST_ENTRIES", 100_000)
    with pytest.raises(SetupError, match="byte limit"):
        runtime.inject_directory(ref, tests_dir, PurePosixPath("/tests"))

    assert uploads == 0


@pytest.mark.integration
def test_engine_container_cannot_open_raw_socket_for_spoofed_source_packet(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler capability unavailable")
    try:
        client = docker_sdk.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local ubuntu Docker capability unavailable: {error}")

    task_source = tmp_path / "task-source"
    engine = tmp_path / "engine-root"
    probe_dir = engine / "probe"
    task_source.mkdir()
    probe_dir.mkdir(parents=True)
    source = probe_dir / "spoof.c"
    source.write_text(
        textwrap.dedent(
            r"""
            #include <arpa/inet.h>
            #include <errno.h>
            #include <netinet/ip.h>
            #include <netinet/udp.h>
            #include <stdio.h>
            #include <string.h>
            #include <sys/socket.h>
            #include <unistd.h>

            int main(int argc, char **argv) {
                if (argc != 2) return 3;
                int fd = socket(AF_INET, SOCK_RAW, IPPROTO_RAW);
                if (fd < 0) return errno == EPERM ? 0 : 4;
                unsigned char packet[sizeof(struct iphdr) + sizeof(struct udphdr)];
                memset(packet, 0, sizeof(packet));
                struct iphdr *ip = (struct iphdr *)packet;
                ip->version = 4;
                ip->ihl = 5;
                ip->tot_len = htons(sizeof(packet));
                ip->ttl = 64;
                ip->protocol = IPPROTO_UDP;
                ip->saddr = inet_addr("198.51.100.77");
                ip->daddr = inet_addr(argv[1]);
                struct udphdr *udp = (struct udphdr *)(packet + sizeof(*ip));
                udp->source = htons(40000);
                udp->dest = htons(9);
                udp->len = htons(sizeof(*udp));
                struct sockaddr_in destination = {0};
                destination.sin_family = AF_INET;
                destination.sin_addr.s_addr = ip->daddr;
                int sent = sendto(fd, packet, sizeof(packet), 0,
                                  (struct sockaddr *)&destination,
                                  sizeof(destination));
                close(fd);
                return sent == (int)sizeof(packet) ? 42 : 5;
            }
            """
        )
    )
    compile_result = subprocess.run(
        [compiler, "-O2", "-o", str(probe_dir / "spoof"), str(source)],
        capture_output=True,
        check=False,
        text=True,
    )
    if compile_result.returncode != 0:
        pytest.skip(
            f"raw-socket probe build unavailable: {compile_result.stderr}"
        )

    provisioner = DockerContainerRuntime(
        client,
        run_id="spoof",
        task_id="raw-socket",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )
    network = provisioner.create_network("control", internal=False)
    runtime = DockerContainerRuntime(
        client,
        run_id="spoof",
        task_id="raw-socket",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
        network=network,
    )
    ref = runtime.create(
        ContainerSpec(
            image="ubuntu:24.04",
            command=("sleep", "infinity"),
            workdir=PurePosixPath("/workspace"),
            mounts=(
                ContainerMount(
                    source=probe_dir,
                    target=PurePosixPath("/run/rsi-harness/staging"),
                ),
            ),
        )
    )
    container = client.containers.get(ref.container_id)
    container.start()
    control = None
    try:
        container.reload()
        gateway = client.networks.get(network.network_id).attrs["IPAM"]["Config"][
            0
        ]["Gateway"]
        control = client.containers.create(
            "ubuntu:24.04",
            ["/probe/spoof", gateway],
            network=network.name,
            mounts=[
                Mount(
                    source=str(probe_dir),
                    target="/probe",
                    type="bind",
                    read_only=True,
                )
            ],
        )
        control.start()
        control_status = control.wait()["StatusCode"]
        if control_status != 42:
            pytest.skip(
                f"raw spoof control capability unavailable: exit {control_status}"
            )
        result = container.exec_run(["/run/rsi-harness/staging/spoof", gateway])
        assert result.exit_code == 0, result.output
    finally:
        if control is not None:
            control.remove(force=True)
        runtime.stop(ref)
        runtime.remove(ref)
        runtime.remove_network(network)


def test_container_spec_has_no_unsafe_docker_escape_hatches():
    for unsafe in ("network_mode", "privileged", "devices"):
        with pytest.raises(TypeError):
            ContainerSpec(image="work", **{unsafe: object()})


@pytest.mark.parametrize("target", ["/var/run/docker.sock", "/run/docker.sock"])
def test_docker_socket_mount_is_rejected(tmp_path, target):
    socket = tmp_path / "docker.sock"
    socket.touch()
    runtime = make_runtime(FakeDockerClient(), tmp_path)

    with pytest.raises(SetupError, match="Docker socket"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(
                    ContainerMount(
                        source=socket.resolve(), target=PurePosixPath(target)
                    ),
                ),
            )
        )


def test_task_source_cannot_be_mounted(tmp_path):
    source = tmp_path / "task"
    source.mkdir()
    runtime = DockerContainerRuntime(
        FakeDockerClient(),
        run_id="r",
        task_id="t",
        role="work",
        task_source_dir=source.resolve(),
        allowed_mount_roots=(tmp_path.resolve(),),
    )

    with pytest.raises(SetupError, match="task source"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(
                    ContainerMount(
                        source=source.resolve(),
                        target=PurePosixPath("/task"),
                    ),
                ),
            )
        )


def test_pause_waits_until_daemon_observes_paused_state(tmp_path):
    client = FakeDockerClient()
    container = FakeDockerContainer("work", pause_states=(False, False, True))
    client.containers.by_id["work"] = container
    runtime = make_runtime(client, tmp_path, poll_interval_seconds=0)

    runtime.pause(ContainerRef(container_id="work", role="work"))

    assert container.events == ["pause", "reload", "reload", "reload"]


def test_unpause_is_idempotent_when_container_is_already_running(tmp_path):
    client = FakeDockerClient()
    container = FakeDockerContainer("work", pause_states=(False,))
    client.containers.by_id["work"] = container
    runtime = make_runtime(client, tmp_path)

    runtime.unpause(ContainerRef(container_id="work", role="work"))

    assert container.events == ["reload"]


def test_exec_continuously_drains_and_keeps_bounded_head_and_tail(tmp_path):
    client = FakeDockerClient(chunks=(b"abcd", b"efgh", b"ijkl"), exit_code=7)
    client.containers.by_id["work"] = FakeDockerContainer("work")
    runtime = make_runtime(client, tmp_path, exec_output_limit_bytes=8)

    result = runtime.exec(
        ContainerRef(container_id="work", role="work"),
        ("sh", "-c", "produce-output"),
        user="agent",
        environment={"HOME": "/home/agent"},
    )

    assert result.exit_code == 7
    assert result.output == "abcdijkl"
    assert result.output_truncated is True
    assert client.api.drained_chunks == 3
    assert client.api.stream_closed is True
    assert client.api.response_closed is True
    assert client.api.exec_create_calls == [
        {
            "container": "work",
            "command": ("sh", "-c", "produce-output"),
            "stdout": True,
            "stderr": True,
            "stdin": False,
            "tty": False,
            "privileged": False,
            "user": "agent",
            "environment": {"HOME": "/home/agent"},
        }
    ]


def test_exec_streams_complete_redacted_output_while_return_stays_bounded(tmp_path):
    client = FakeDockerClient(
        chunks=(b"abcd", b"runtime-", b"secret token=opaque\n", b"ijkl"),
        exit_code=7,
    )
    client.containers.by_id["work"] = FakeDockerContainer("work")
    runtime = make_runtime(client, tmp_path, exec_output_limit_bytes=8)
    output_path = (tmp_path / "engine-root" / "agent-output.txt").resolve()
    live_output: list[str] = []

    result = runtime.exec(
        ContainerRef(container_id="work", role="work"),
        ("sh", "-c", "produce-output"),
        output_path=output_path,
        output_redact_values=("runtime-secret",),
        output_callback=live_output.append,
    )

    assert result.output == "abcdijkl"
    assert result.output_truncated is True
    assert result.full_output_captured is True
    assert output_path.read_text() == (
        "abcd[REDACTED] token=[REDACTED]\nijkl"
    )
    assert "".join(live_output) == output_path.read_text()


def test_exec_live_output_callback_failure_does_not_disturb_capture(tmp_path):
    client = FakeDockerClient(chunks=(b"first line\n", b"second line\n"))
    client.containers.by_id["work"] = FakeDockerContainer("work")
    runtime = make_runtime(client, tmp_path)
    output_path = (tmp_path / "engine-root" / "agent-output.txt").resolve()

    def fail_console(_value: str) -> None:
        raise RuntimeError("console closed")

    result = runtime.exec(
        ContainerRef(container_id="work", role="work"),
        ("sh", "-c", "produce-output"),
        output_path=output_path,
        output_callback=fail_console,
    )

    assert result.exit_code == 0
    assert result.full_output_captured is True
    assert output_path.read_text() == "first line\nsecond line\n"


def test_exec_streams_complete_task_output_without_redaction(tmp_path):
    authored = b"Authorization: Bearer task-literal\naccess_token=visible\n"
    client = FakeDockerClient(chunks=(authored,), exit_code=0)
    client.containers.by_id["judge"] = FakeDockerContainer("judge")
    runtime = make_runtime(
        client, tmp_path, role="judge", exec_output_limit_bytes=8
    )
    output_path = (tmp_path / "engine-root" / "agent-1.log").resolve()

    result = runtime.exec(
        ContainerRef(container_id="judge", role="judge"),
        ("/bin/bash", "/tests/test.sh"),
        output_path=output_path,
        redact_output=False,
    )

    assert result.output_truncated is True
    assert result.full_output_captured is True
    assert output_path.read_bytes() == authored
    assert output_path.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("timeout_seconds", (None, 0.01))
def test_exec_raw_output_fsync_failure_is_never_published(
    tmp_path, monkeypatch, timeout_seconds
):
    client = FakeDockerClient(
        chunks=(b"partial verifier output",),
        chunk_delay_seconds=0.5 if timeout_seconds is not None else 0,
    )
    client.containers.by_id["judge"] = FakeDockerContainer("judge")
    runtime = make_runtime(client, tmp_path, role="judge")
    output_path = tmp_path / "engine-root" / "agent-1.log"

    def fail_fsync(_descriptor):
        raise OSError("simulated verifier output fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(InfrastructureError, match="output capture failed"):
        runtime.exec(
            ContainerRef(container_id="judge", role="judge"),
            ("/bin/bash", "/tests/test.sh"),
            timeout_seconds=timeout_seconds,
            output_path=output_path,
            redact_output=False,
        )

    assert not output_path.exists()
    assert not tuple(output_path.parent.glob(".agent-1.log.*"))


def test_exec_raw_output_rejects_symlinked_parent_directory(tmp_path):
    client = FakeDockerClient(chunks=(b"verifier output",), exit_code=0)
    client.containers.by_id["judge"] = FakeDockerContainer("judge")
    runtime = make_runtime(client, tmp_path, role="judge")
    redirected = tmp_path / "engine-root" / "redirected"
    redirected.mkdir()
    feedback = tmp_path / "engine-root" / "feedback"
    feedback.symlink_to(redirected, target_is_directory=True)

    with pytest.raises(InfrastructureError, match="failed to open complete"):
        runtime.exec(
            ContainerRef(container_id="judge", role="judge"),
            ("/bin/bash", "/tests/test.sh"),
            output_path=feedback / "agent-1.log",
            redact_output=False,
        )

    assert not (redirected / "agent-1.log").exists()


def test_exec_raw_output_open_failure_cleans_temporary_file(
    tmp_path, monkeypatch
):
    client = FakeDockerClient(chunks=(b"verifier output",), exit_code=0)
    client.containers.by_id["judge"] = FakeDockerContainer("judge")
    runtime = make_runtime(client, tmp_path, role="judge")
    output_path = tmp_path / "engine-root" / "agent-1.log"

    def fail_fchmod(_descriptor, _mode):
        raise OSError("simulated fchmod failure")

    monkeypatch.setattr(os, "fchmod", fail_fchmod)
    with pytest.raises(InfrastructureError, match="failed to open complete"):
        runtime.exec(
            ContainerRef(container_id="judge", role="judge"),
            ("/bin/bash", "/tests/test.sh"),
            output_path=output_path,
            redact_output=False,
        )

    assert not output_path.exists()
    assert not tuple(output_path.parent.glob(".agent-1.log.*"))


def test_exec_timeout_stops_before_draining_buffered_output(tmp_path, monkeypatch):
    client = FakeDockerClient(
        chunks=(b"before-timeout\n", b"docker-buffered\n"),
        chunk_delay_seconds=0.02,
    )
    container = FakeDockerContainer("judge")
    client.containers.by_id["judge"] = container
    runtime = make_runtime(client, tmp_path, role="judge")
    output_path = tmp_path / "engine-root" / "agent-1.log"
    events: list[str] = []
    original_stop = container.stop
    original_close = docker_runtime._close_docker_stream

    def record_stop(**kwargs):
        events.append("stop")
        original_stop(**kwargs)

    def record_close(stream):
        events.append("close")
        original_close(stream)

    monkeypatch.setattr(container, "stop", record_stop)
    monkeypatch.setattr(docker_runtime, "_close_docker_stream", record_close)

    result = runtime.exec(
        ContainerRef(container_id="judge", role="judge"),
        ("/bin/bash", "/tests/test.sh"),
        timeout_seconds=0.03,
        output_path=output_path,
        redact_output=False,
    )

    assert result.timed_out is True
    assert output_path.read_bytes() == b"before-timeout\ndocker-buffered\n"
    assert events[:2] == ["stop", "close"]


def test_exec_timeout_forced_stream_close_never_publishes_raw_output(tmp_path):
    client = FakeDockerClient(
        chunks=(b"daemon-buffered output",), chunk_delay_seconds=0.5
    )
    client.containers.by_id["judge"] = FakeDockerContainer("judge")
    runtime = make_runtime(
        client, tmp_path, role="judge", pause_timeout_seconds=0.02
    )
    output_path = tmp_path / "engine-root" / "agent-1.log"

    with pytest.raises(
        InfrastructureError, match="recovery_required.*natural EOF"
    ):
        runtime.exec(
            ContainerRef(container_id="judge", role="judge"),
            ("/bin/bash", "/tests/test.sh"),
            timeout_seconds=0.01,
            output_path=output_path,
            redact_output=False,
        )

    assert not output_path.exists()
    assert not tuple(output_path.parent.glob(".agent-1.log.*"))


def test_exec_timeout_stop_failure_aborts_unpublished_output(
    tmp_path, monkeypatch
):
    client = FakeDockerClient(
        chunks=(b"late verifier output",), chunk_delay_seconds=0.5
    )
    container = FakeDockerContainer("judge")
    client.containers.by_id["judge"] = container
    runtime = make_runtime(client, tmp_path, role="judge")
    output_path = tmp_path / "engine-root" / "agent-1.log"

    def fail_stop(**_kwargs):
        raise APIError("simulated stop failure")

    monkeypatch.setattr(container, "stop", fail_stop)
    with pytest.raises(InfrastructureError, match="failed to stop"):
        runtime.exec(
            ContainerRef(container_id="judge", role="judge"),
            ("/bin/bash", "/tests/test.sh"),
            timeout_seconds=0.01,
            output_path=output_path,
            redact_output=False,
        )

    assert not output_path.exists()
    assert not tuple(output_path.parent.glob(".agent-1.log.*"))
    assert not any(
        thread.name == "exec-exec-1" for thread in threading.enumerate()
    )


def test_exec_timeout_does_not_block_on_stuck_output_fsync(
    tmp_path, monkeypatch
):
    client = FakeDockerClient(chunks=(b"verifier output",), exit_code=0)
    client.containers.by_id["judge"] = FakeDockerContainer("judge")
    runtime = make_runtime(
        client, tmp_path, role="judge", pause_timeout_seconds=0.02
    )
    output_path = tmp_path / "engine-root" / "agent-1.log"
    fsync_started = threading.Event()
    release_fsync = threading.Event()
    original_fsync = os.fsync

    def blocking_fsync(descriptor):
        fsync_started.set()
        release_fsync.wait(timeout=2)
        original_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", blocking_fsync)
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            runtime.exec(
                ContainerRef(container_id="judge", role="judge"),
                ("/bin/bash", "/tests/test.sh"),
                timeout_seconds=0.01,
                output_path=output_path,
                redact_output=False,
            )
        except BaseException as error:
            errors.append(error)

    caller = threading.Thread(target=invoke)
    caller.start()
    assert fsync_started.wait(timeout=1)
    caller.join(timeout=0.75)
    bounded = not caller.is_alive()
    release_fsync.set()
    caller.join(timeout=1)
    deadline = time.monotonic() + 1
    while any(
        thread.name == "exec-exec-1" for thread in threading.enumerate()
    ) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert bounded is True
    assert len(errors) == 1
    assert isinstance(errors[0], InfrastructureError)
    assert "recovery_required" in str(errors[0])
    assert "output capture" in str(errors[0])
    assert "stop" not in client.containers.by_id["judge"].events
    assert not output_path.exists()
    assert not tuple(output_path.parent.glob(".agent-1.log.*"))


def test_exec_returns_at_wall_clock_timeout_when_stream_blocks(tmp_path):
    client = FakeDockerClient(chunks=(b"eventual output",), chunk_delay_seconds=0.5)
    client.containers.by_id["work"] = FakeDockerContainer("work")
    runtime = make_runtime(
        client,
        tmp_path,
        exec_output_limit_bytes=8,
        pause_timeout_seconds=0.1,
    )
    started = time.monotonic()

    result = runtime.exec(
        ContainerRef(container_id="work", role="work"),
        ("blocked-command",),
        timeout_seconds=0.01,
    )

    assert time.monotonic() - started < 0.3
    assert result.exit_code is None
    assert result.timed_out is True
    container = client.containers.by_id["work"]
    assert "stop" in container.events
    assert container.attrs["State"]["Running"] is False
    assert client.api.stream_closed is True
    assert client.api.response_closed is True
    assert not any(thread.name == "exec-exec-1" for thread in threading.enumerate())


def test_exec_timeout_contains_container_when_stream_close_raises(tmp_path):
    client = FakeDockerClient(
        chunks=(b"eventual output",),
        chunk_delay_seconds=0.5,
        stream_close_error=AttributeError("reader raced with close"),
    )
    client.containers.by_id["work"] = FakeDockerContainer("work")
    runtime = make_runtime(client, tmp_path, exec_output_limit_bytes=8)

    result = runtime.exec(
        ContainerRef(container_id="work", role="work"),
        ("blocked-command",),
        timeout_seconds=0.01,
    )

    assert result.timed_out is True
    container = client.containers.by_id["work"]
    assert "stop" in container.events
    assert container.attrs["State"]["Running"] is False
    assert client.api.response_closed is True
    assert not any(thread.name == "exec-exec-1" for thread in threading.enumerate())


def test_logs_are_byte_bounded_and_process_listing_is_parsed(tmp_path):
    client = FakeDockerClient()
    client.containers.by_id["work"] = FakeDockerContainer(
        "work", logs_chunks=(b"abcd", b"efgh", b"ijkl")
    )
    runtime = make_runtime(client, tmp_path, log_output_limit_bytes=8)
    ref = ContainerRef(container_id="work", role="work")

    logs = runtime.logs(ref, tail=25)
    assert logs.output == "abcdijkl"
    assert logs.output_truncated is True
    container = client.containers.by_id["work"]
    assert container.logs_stream_closed is True
    assert container.logs_response_closed is True
    assert runtime.processes(ref) == (101,)


def test_copy_to_uses_engine_staging_mount_not_docker_archive(tmp_path):
    client = FakeDockerClient(chunks=(b"",), exit_code=0)
    staging = tmp_path / "engine-root" / "staging"
    source = tmp_path / "prompt.md"
    source.write_text("hello")
    runtime = make_runtime(client, tmp_path, staging=True)
    ref = runtime.create(ContainerSpec(image="work"))

    runtime.copy_to(ref, source, PurePosixPath("/tmp/prompt.md"))

    parent_command = client.api.exec_create_calls[-2]["command"]
    command = client.api.exec_create_calls[-1]["command"]
    assert parent_command == ("/usr/bin/install", "-d", "-m", "0755", "/tmp")
    assert command[:2] == ("/usr/bin/install", "-m")
    assert command[-1] == "/tmp/prompt.md"
    assert not tuple(staging.iterdir())


def test_real_claude_hook_copy_creates_settings_parent(tmp_path):
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, staging=True)
    ref = runtime.create(ContainerSpec(image="work"))
    agent = create_agent("claude-code", RSILoopConfig())
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    agent.install_stop_hook(
        RSILoopBackendBridge(runtime),
        RSILoopContainerHandle(ref),
        log_dir,
        logging.getLogger("hook-copy-test"),
    )

    commands = [call["command"] for call in client.api.exec_create_calls]
    settings_install = next(
        index
        for index, command in enumerate(commands)
        if isinstance(command, tuple)
        and command[-1] == "/home/agent/.claude/settings.json"
    )
    assert commands[settings_install - 1] == (
        "/usr/bin/install",
        "-d",
        "-m",
        "0755",
        "/home/agent/.claude",
    )


@pytest.mark.integration
def test_real_claude_hook_copy_installs_settings_in_real_container(tmp_path):
    try:
        client = docker_sdk.from_env()
        client.ping()
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local ubuntu Docker capability unavailable: {error}")
    task_source = tmp_path / "task-source"
    engine = tmp_path / "engine-root"
    task_source.mkdir()
    engine.mkdir()
    runtime = DockerContainerRuntime(
        client,
        run_id="hook-copy",
        task_id="claude",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
        staging_dir=engine / "staging",
    )
    ref = runtime.create(
        ContainerSpec(image="ubuntu:24.04", command=("sleep", "infinity"))
    )
    container = client.containers.get(ref.container_id)
    container.start()
    try:
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        create_agent("claude-code", RSILoopConfig()).install_stop_hook(
            RSILoopBackendBridge(runtime),
            RSILoopContainerHandle(ref),
            log_dir,
            logging.getLogger("real-hook-copy-test"),
        )

        result = container.exec_run(
            ["test", "-s", "/home/agent/.claude/settings.json"]
        )
        assert result.exit_code == 0, result.output
    finally:
        runtime.stop(ref)
        runtime.remove(ref)


@pytest.mark.parametrize(
    "target",
    [PurePosixPath("/tmp/../etc/passwd"), PurePosixPath("/../../etc/passwd")],
)
def test_copy_to_rejects_lexical_traversal(tmp_path, target):
    source = tmp_path / "source"
    source.write_text("x")
    runtime = make_runtime(FakeDockerClient(), tmp_path, staging=True)

    with pytest.raises(SetupError, match="traversal"):
        runtime.copy_to(ContainerRef(container_id="work", role="work"), source, target)


def test_network_lifecycle_uses_private_bridge_and_recovery_labels(tmp_path):
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")

    network = runtime.create_network("round-2", internal=True)
    runtime.remove_network(network)

    assert client.networks.created == [
        {
            "name": "rsi-r-t-judge-round-2",
            "driver": "bridge",
            "internal": True,
            "check_duplicate": True,
            "labels": {
                "rsi-harness.run-id": "r",
                "rsi-harness.task-id": "t",
                "rsi-harness.role": "judge",
            },
            "options": {
                "com.docker.network.bridge.name": managed_bridge_interface(
                    "rsi-r-t-judge-round-2"
                )
            },
        }
    ]
    assert client.networks.by_id[network.network_id].removed is True


def test_network_create_api_error_removes_discovered_exact_mutation(tmp_path):
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    original_create = client.networks.create

    def mutate_then_raise(name, **kwargs):
        original_create(name, **kwargs)
        raise APIError("connection dropped after create")

    client.networks.create = mutate_then_raise

    with pytest.raises(SetupError, match="proven absent"):
        runtime.create_network("round-2", internal=True)

    [network] = client.networks.by_id.values()
    assert network.removed is True
    assert client.networks.list() == []


def test_network_create_api_error_with_unproved_cleanup_is_recovery_required(
    tmp_path,
) -> None:
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path, role="judge")
    original_create = client.networks.create

    def mutate_then_raise(name, **kwargs):
        network = original_create(name, **kwargs)

        def fail_remove():
            raise APIError("network remove response is ambiguous")

        network.remove = fail_remove
        raise APIError("connection dropped after create")

    client.networks.create = mutate_then_raise

    with pytest.raises(InfrastructureError, match="recovery_required.*network"):
        runtime.create_network("round-2", internal=True)

    [network] = client.networks.by_id.values()
    assert network.removed is False


def test_stop_and_remove_are_safe_repeatable_cleanup_operations(tmp_path):
    client = FakeDockerClient()
    client.containers.by_id["work"] = FakeDockerContainer("work")
    runtime = make_runtime(client, tmp_path)
    ref = ContainerRef(container_id="work", role="work")

    runtime.stop(ref)
    runtime.remove(ref)

    container = client.containers.by_id["work"]
    assert container.events == ["stop", "remove"]
    assert container.removed_kwargs == {"force": True, "v": True}


@pytest.mark.parametrize("operation", ["stop", "remove"])
def test_absent_container_cleanup_is_idempotent(operation, tmp_path):
    client = FakeDockerClient()

    def missing(_container_id):
        raise NotFound("already gone")

    client.containers.get = missing
    runtime = make_runtime(client, tmp_path)

    getattr(runtime, operation)(ContainerRef(container_id="gone", role="work"))


def test_container_and_network_sdk_failures_are_typed(tmp_path):
    client = FakeDockerClient()
    runtime = make_runtime(client, tmp_path)

    def fail_create(*args, **kwargs):
        raise APIError("daemon rejected container")

    client.containers.create = fail_create
    with pytest.raises(SetupError, match="create Work container"):
        runtime.create(ContainerSpec(image="work"))

    def fail_network(*args, **kwargs):
        raise APIError("network pool exhausted")

    client.networks.create = fail_network
    with pytest.raises(SetupError, match="create work network"):
        runtime.create_network("round", internal=True)

    client.containers.get = lambda _id: (_ for _ in ()).throw(APIError("offline"))
    with pytest.raises(InfrastructureError, match="stop Work container"):
        runtime.stop(ContainerRef(container_id="work", role="work"))


@pytest.mark.parametrize(
    ("operation", "failure_method", "message"),
    [
        ("pause", "pause", "pause Work container work"),
        ("unpause", "reload", "inspect Work container work before unpause"),
        ("logs", "logs", "read logs from Work container work"),
        ("processes", "top", "list processes in Work container work"),
    ],
)
def test_container_operation_sdk_failures_are_typed(
    tmp_path, operation, failure_method, message
):
    client = FakeDockerClient()
    container = FakeDockerContainer("work")
    client.containers.by_id["work"] = container

    def fail(*args, **kwargs):
        raise APIError("daemon unavailable")

    setattr(container, failure_method, fail)
    runtime = make_runtime(client, tmp_path)

    with pytest.raises(InfrastructureError, match=message):
        getattr(runtime, operation)(ContainerRef(container_id="work", role="work"))


def test_exec_sdk_create_failure_is_typed_with_container_and_command_context(tmp_path):
    client = FakeDockerClient()

    def fail(*args, **kwargs):
        raise APIError("exec service unavailable")

    client.api.exec_create = fail
    runtime = make_runtime(client, tmp_path)

    with pytest.raises(
        InfrastructureError,
        match=r"create Docker exec in Work container work.*probe",
    ):
        runtime.exec(
            ContainerRef(container_id="work", role="work"),
            ("/usr/bin/probe", "--quiet"),
        )


def test_absent_network_cleanup_is_idempotent(tmp_path):
    client = FakeDockerClient()
    client.networks.get = lambda _id: (_ for _ in ()).throw(NotFound("gone"))
    runtime = make_runtime(client, tmp_path)

    runtime.remove_network("gone")


def test_unmanaged_or_builtin_networks_are_rejected(tmp_path):
    task_source = tmp_path / "task"
    engine = tmp_path / "engine"
    task_source.mkdir()
    engine.mkdir()

    with pytest.raises(SetupError, match="Engine-created"):
        DockerContainerRuntime(
            FakeDockerClient(),
            run_id="r",
            task_id="t",
            role="work",
            task_source_dir=task_source,
            allowed_mount_roots=(engine,),
            network="host",
        )


def test_forged_arbitrary_network_name_is_rejected_even_with_matching_labels(
    tmp_path,
):
    client = FakeDockerClient()
    network = client.networks.create(
        "attacker-selected",
        driver="bridge",
        internal=False,
        labels={
            "rsi-harness.run-id": "r",
            "rsi-harness.task-id": "t",
            "rsi-harness.role": "work",
        },
    )
    task_source = tmp_path / "task"
    engine = tmp_path / "engine"
    task_source.mkdir()
    engine.mkdir()

    with pytest.raises(SetupError, match="Engine-created"):
        DockerContainerRuntime(
            client,
            run_id="r",
            task_id="t",
            role="work",
            task_source_dir=task_source,
            allowed_mount_roots=(engine,),
            network=ManagedNetwork(
                network_id=network.id,
                name="attacker-selected",
                run_id="r",
                task_id="t",
                role="work",
                internal=False,
            ),
        )


def test_start_requires_exact_installed_container_network_role_lease(tmp_path):
    client = FakeDockerClient()
    task_source = tmp_path / "task"
    engine = tmp_path / "engine"
    task_source.mkdir()
    engine.mkdir()
    provisioner = DockerContainerRuntime(
        client,
        run_id="r",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )
    network = provisioner.create_network("control", internal=False)
    enforcer = NetworkPolicyEnforcer(run_id="r", firewall=FakeFirewallBackend())
    runtime = DockerContainerRuntime(
        client,
        run_id="r",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
        network=network,
        network_policy_enforcer=enforcer,
    )
    ref = runtime.create(ContainerSpec(image="work"))

    with pytest.raises(SetupError, match="installed network policy lease"):
        runtime.start(ref)

    lease = enforcer.apply(
        ref,
        NetworkPolicy(mode="no-network"),
        network=network,
    )
    runtime.install_network_policy(ref, lease)
    runtime.start(ref)
    assert client.containers.by_id[ref.container_id].events == ["start"]


def configured_runtime_and_lease(tmp_path):
    client = FakeDockerClient()
    task_source = tmp_path / "task"
    engine = tmp_path / "engine"
    task_source.mkdir()
    engine.mkdir()
    provisioner = DockerContainerRuntime(
        client,
        run_id="r",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )
    network = provisioner.create_network("control", internal=False)
    firewall = FakeFirewallBackend()
    enforcer = NetworkPolicyEnforcer(run_id="r", firewall=firewall)
    runtime = DockerContainerRuntime(
        client,
        run_id="r",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
        network=network,
        network_policy_enforcer=enforcer,
    )
    ref = runtime.create(ContainerSpec(image="work"))
    lease = enforcer.apply(ref, NetworkPolicy(mode="no-network"), network=network)
    return client, runtime, ref, network, enforcer, firewall, lease


@pytest.mark.parametrize(
    "forge",
    [
        lambda lease: replace(lease),
        lambda lease: replace(lease, rule_id="forged-rule-id"),
        lambda lease: replace(lease, network_name="forged-network"),
        lambda lease: replace(
            lease, rules=replace(lease.rules, network_id="forged-network-id")
        ),
        lambda lease: replace(
            lease, rules=replace(lease.rules, network_name="forged-network")
        ),
    ],
)
def test_forged_or_internally_inconsistent_lease_cannot_unlock_start(
    tmp_path, forge
):
    client, runtime, ref, _, _, _, lease = configured_runtime_and_lease(tmp_path)

    with pytest.raises(SetupError, match="authoritative|match"):
        runtime.install_network_policy(ref, forge(lease))

    with pytest.raises(SetupError, match="installed network policy lease"):
        runtime.start(ref)
    assert client.containers.by_id[ref.container_id].events == []


def test_container_role_mismatch_cannot_register_an_otherwise_valid_lease(tmp_path):
    _, runtime, ref, _, _, _, lease = configured_runtime_and_lease(tmp_path)
    forged_ref = ContainerRef(container_id=ref.container_id, role="judge")

    with pytest.raises(SetupError, match="match container, network, and role"):
        runtime.install_network_policy(forged_ref, lease)


@pytest.mark.parametrize("remove_through_enforcer", [True, False])
def test_start_rechecks_policy_is_still_installed_after_registration(
    tmp_path, remove_through_enforcer
):
    client, runtime, ref, _, enforcer, firewall, lease = (
        configured_runtime_and_lease(tmp_path)
    )
    runtime.install_network_policy(ref, lease)
    if remove_through_enforcer:
        enforcer.cleanup(lease)
    else:
        firewall.installed.pop(lease.rule_id)

    with pytest.raises(SetupError, match="no longer installed|authoritative"):
        runtime.start(ref)
    assert client.containers.by_id[ref.container_id].events == []


def test_mount_sources_are_canonical_engine_owned_and_task_source_is_excluded(
    tmp_path,
):
    client = FakeDockerClient()
    task_source = tmp_path / "task"
    engine = tmp_path / "engine"
    task_source.mkdir()
    engine.mkdir()
    allowed = engine / "workspace"
    allowed.mkdir()
    outside_link = engine / "outside-link"
    outside_link.symlink_to("/etc")
    runtime = DockerContainerRuntime(
        client,
        run_id="r",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )

    for source in (
        Path("/etc/hosts"),
        outside_link / "hosts",
        task_source,
        tmp_path,
    ):
        with pytest.raises(SetupError, match="Engine-owned|task source"):
            runtime.create(
                ContainerSpec(
                    image="work",
                    workdir=PurePosixPath("/workspace"),
                    mounts=(
                        ContainerMount(
                            source=source.resolve(),
                            target=PurePosixPath("/run/rsi-harness/staging"),
                        ),
                    ),
                )
            )

    ref = runtime.create(
        ContainerSpec(
            image="work",
            workdir=PurePosixPath("/workspace"),
            mounts=(
                ContainerMount(
                    source=allowed.resolve(),
                    target=PurePosixPath("/run/rsi-harness/staging"),
                ),
            ),
        )
    )
    assert ref.role == "work"


def test_mount_targets_are_role_specific(tmp_path):
    task_source = tmp_path / "task"
    engine = tmp_path / "engine"
    source = engine / "data"
    task_source.mkdir()
    source.mkdir(parents=True)
    runtime = DockerContainerRuntime(
        FakeDockerClient(),
        run_id="r",
        task_id="t",
        role="work",
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )

    with pytest.raises(SetupError, match="mount target"):
        runtime.create(
            ContainerSpec(
                image="work",
                workdir=PurePosixPath("/workspace"),
                mounts=(
                    ContainerMount(
                        source=source.resolve(), target=PurePosixPath("/tests")
                    ),
                ),
            )
        )
