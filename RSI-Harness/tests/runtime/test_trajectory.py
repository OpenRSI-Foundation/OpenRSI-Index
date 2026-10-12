from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path, PurePosixPath

import pytest
from docker.types import Mount

from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import JUDGE_TRAJECTORY_ROOT, ContainerMount, ContainerSpec
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.mount_topology import validate_split_workdir_target
from rsi_harness.runtime.trajectory import (
    TRAJECTORY_DIRECTORY,
    capture_round_trajectory,
    release_round_trajectory,
)
from tests.fakes import FakeDockerClient


def _run_root(tmp_path: Path, run_id: str = "run-1") -> Path:
    root = tmp_path / "logs" / "runs" / run_id / "task"
    root.mkdir(parents=True)
    return root


def _manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text())


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_capture_copies_the_run_trajectory_readably_with_a_manifest(tmp_path):
    root = _run_root(tmp_path)
    (root / "agent_prompt.md").write_text("Solve the task.\n")
    (root / "agent_output.txt").write_text("step 1\nstep 2\n")
    (root / "agent_output.txt").chmod(0o600)

    directory = capture_round_trajectory(root, "agent-1")

    assert directory == root / TRAJECTORY_DIRECTORY / "agent-1"
    assert (directory / "agent_prompt.md").read_text() == "Solve the task.\n"
    assert (directory / "agent_output.txt").read_text() == "step 1\nstep 2\n"
    # Readable by any Judge user, but only through a private host parent.
    assert _mode(directory.parent) == 0o700
    assert _mode(directory) == 0o755
    assert _mode(directory / "agent_output.txt") == 0o644
    assert _mode(directory / "manifest.json") == 0o644
    assert _manifest(directory) == {
        "schema_version": 1,
        "round_id": "agent-1",
        "previous_round_id": None,
        "limit_bytes": 256 * 1024 * 1024,
        "files": [
            {
                "path": "agent_prompt.md",
                "source_size": 16,
                "offset": 0,
                "bytes": 16,
                "sha256": hashlib.sha256(b"Solve the task.\n").hexdigest(),
                "truncated": False,
                "previous_size": None,
            },
            {
                "path": "agent_output.txt",
                "source_size": 14,
                "offset": 0,
                "bytes": 14,
                "sha256": hashlib.sha256(b"step 1\nstep 2\n").hexdigest(),
                "truncated": False,
                "previous_size": None,
            },
        ],
    }


def test_capture_omits_missing_sources(tmp_path):
    root = _run_root(tmp_path)
    (root / "agent_output.txt").write_text("")

    directory = capture_round_trajectory(root, "agent-1")

    assert [entry["path"] for entry in _manifest(directory)["files"]] == [
        "agent_output.txt"
    ]
    assert not (directory / "agent_prompt.md").exists()


def test_capture_keeps_only_the_most_recent_bytes_over_the_limit(tmp_path):
    root = _run_root(tmp_path)
    payload = b"0123456789" * 10
    (root / "agent_output.txt").write_bytes(payload)

    directory = capture_round_trajectory(root, "agent-1", limit_bytes=25)

    assert (directory / "agent_output.txt").read_bytes() == payload[-25:]
    (entry,) = _manifest(directory)["files"]
    assert entry["source_size"] == 100
    assert entry["offset"] == 75
    assert entry["bytes"] == 25
    assert entry["truncated"] is True
    assert entry["sha256"] == hashlib.sha256(payload[-25:]).hexdigest()


def test_later_rounds_record_the_previous_round_sizes(tmp_path):
    root = _run_root(tmp_path)
    output = root / "agent_output.txt"
    output.write_text("first\n")
    first = capture_round_trajectory(root, "agent-1")
    with output.open("a") as stream:
        stream.write("second\n")

    second = capture_round_trajectory(root, "agent-2")

    manifest = _manifest(second)
    assert manifest["previous_round_id"] == "agent-1"
    (entry,) = manifest["files"]
    assert entry["previous_size"] == len("first\n")
    assert entry["source_size"] == len("first\nsecond\n")
    # The earlier copy is frozen at its own submission.
    assert (first / "agent_output.txt").read_text() == "first\n"
    assert (second / "agent_output.txt").read_text() == "first\nsecond\n"


def test_previous_round_skips_rounds_without_a_manifest(tmp_path):
    root = _run_root(tmp_path)
    (root / "agent_output.txt").write_text("a\n")
    capture_round_trajectory(root, "agent-1")
    # A round whose capture failed before its manifest was written.
    (root / TRAJECTORY_DIRECTORY / "agent-2").mkdir()

    manifest = _manifest(capture_round_trajectory(root, "agent-3"))

    assert manifest["previous_round_id"] == "agent-1"


def test_bytes_appended_after_capture_never_reach_the_copy(tmp_path):
    root = _run_root(tmp_path)
    output = root / "agent_output.txt"
    output.write_text("submitted\n")

    directory = capture_round_trajectory(root, "agent-1")
    with output.open("a") as stream:
        stream.write("after the submission\n")

    assert (directory / "agent_output.txt").read_text() == "submitted\n"
    assert _manifest(directory)["files"][0]["source_size"] == len("submitted\n")


def test_symlinked_source_is_rejected_instead_of_reading_another_run(tmp_path):
    root = _run_root(tmp_path)
    other = _run_root(tmp_path, "run-2")
    (other / "agent_output.txt").write_text("another run\n")
    (root / "agent_output.txt").symlink_to(other / "agent_output.txt")

    with pytest.raises(InfrastructureError, match="not a regular file"):
        capture_round_trajectory(root, "agent-1")


def test_symlinked_trajectory_directory_is_rejected(tmp_path):
    root = _run_root(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / TRAJECTORY_DIRECTORY).symlink_to(elsewhere)

    with pytest.raises(InfrastructureError, match="not a real directory"):
        capture_round_trajectory(root, "agent-1")
    assert not any(elsewhere.iterdir())


def test_a_round_is_captured_once_and_round_ids_are_exact(tmp_path):
    root = _run_root(tmp_path)
    capture_round_trajectory(root, "agent-1")

    with pytest.raises(InfrastructureError, match="already exists"):
        capture_round_trajectory(root, "agent-1")
    for invalid in ("agent-0", "../agent-1", "agent-1/x", "final"):
        with pytest.raises(InfrastructureError, match="round identity"):
            capture_round_trajectory(root, invalid)


@pytest.mark.parametrize("name", ["../escape.txt", "/abs.txt", "manifest.json"])
def test_unsafe_source_names_are_rejected(tmp_path, name):
    root = _run_root(tmp_path)

    with pytest.raises(ValueError, match="safe relative path"):
        capture_round_trajectory(root, "agent-1", sources=(name,))


def test_nested_sources_keep_their_relative_path(tmp_path):
    root = _run_root(tmp_path)
    (root / "sessions" / "2").mkdir(parents=True)
    (root / "sessions" / "2" / "agent_output.txt").write_text("session 2\n")

    directory = capture_round_trajectory(
        root, "agent-1", sources=("sessions/2/agent_output.txt",)
    )

    assert (directory / "sessions/2/agent_output.txt").read_text() == "session 2\n"
    assert _mode(directory / "sessions") == 0o755
    release_round_trajectory(directory)
    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json"]


def test_release_keeps_only_the_manifest_and_never_raises(tmp_path):
    root = _run_root(tmp_path)
    (root / "agent_prompt.md").write_text("prompt\n")
    (root / "agent_output.txt").write_text("output\n")
    directory = capture_round_trajectory(root, "agent-1")
    manifest = _manifest(directory)

    release_round_trajectory(directory)
    release_round_trajectory(directory)
    release_round_trajectory(tmp_path / "missing")

    assert sorted(path.name for path in directory.iterdir()) == ["manifest.json"]
    assert _manifest(directory) == manifest
    # The source still holds the exact range the Judge read.
    entry = manifest["files"][1]
    source = (root / entry["path"]).read_bytes()
    window = source[entry["offset"] : entry["offset"] + entry["bytes"]]
    assert hashlib.sha256(window).hexdigest() == entry["sha256"]


def test_trajectory_target_is_an_engine_mount_target():
    with pytest.raises(SetupError, match="/logs/agent"):
        validate_split_workdir_target(PurePosixPath("/logs/agent/work"))


def _docker_runtime(client, tmp_path: Path, role: str) -> DockerContainerRuntime:
    task_source = tmp_path / "task-source"
    engine = tmp_path / "engine-root"
    task_source.mkdir(exist_ok=True)
    engine.mkdir(exist_ok=True)
    return DockerContainerRuntime(
        client,
        run_id="r",
        task_id="t",
        role=role,
        task_source_dir=task_source,
        allowed_mount_roots=(engine,),
    )


def test_docker_judge_mounts_the_trajectory_read_only(tmp_path):
    client = FakeDockerClient()
    runtime = _docker_runtime(client, tmp_path, "judge")
    trajectory = tmp_path / "engine-root" / TRAJECTORY_DIRECTORY / "agent-1"
    trajectory.mkdir(parents=True)
    exact = ContainerMount(
        source=trajectory, target=JUDGE_TRAJECTORY_ROOT, read_only=True
    )

    runtime.create(ContainerSpec(image="judge", mounts=(exact,)))

    assert client.containers.created[0]["mounts"] == [
        Mount(
            source=str(trajectory.resolve()),
            target="/logs/agent",
            type="bind",
            read_only=True,
        )
    ]
    with pytest.raises(SetupError, match="trajectory mount differs"):
        runtime.create(
            ContainerSpec(
                image="judge",
                mounts=(exact.model_copy(update={"read_only": False}),),
            )
        )
    other = tmp_path / "engine-root" / "other"
    other.mkdir()
    with pytest.raises(SetupError, match="trajectory mount differs"):
        runtime.create(
            ContainerSpec(
                image="judge",
                mounts=(exact.model_copy(update={"source": other}),),
            )
        )


def test_docker_work_cannot_mount_a_trajectory(tmp_path):
    runtime = _docker_runtime(FakeDockerClient(), tmp_path, "work")
    trajectory = tmp_path / "engine-root" / TRAJECTORY_DIRECTORY / "agent-1"
    trajectory.mkdir(parents=True)

    with pytest.raises(SetupError, match="trajectory mount differs"):
        runtime.create(
            ContainerSpec(
                image="work",
                mounts=(
                    ContainerMount(
                        source=trajectory,
                        target=JUDGE_TRAJECTORY_ROOT,
                        read_only=True,
                    ),
                ),
            )
        )


def test_docker_judge_trajectory_must_be_engine_owned(tmp_path):
    runtime = _docker_runtime(FakeDockerClient(), tmp_path, "judge")
    outside = tmp_path / "outside" / TRAJECTORY_DIRECTORY / "agent-1"
    outside.mkdir(parents=True)

    with pytest.raises(SetupError, match="Engine-owned"):
        runtime.create(
            ContainerSpec(
                image="judge",
                mounts=(
                    ContainerMount(
                        source=outside,
                        target=JUDGE_TRAJECTORY_ROOT,
                        read_only=True,
                    ),
                ),
            )
        )
