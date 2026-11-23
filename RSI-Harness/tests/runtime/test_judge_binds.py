"""The Harness side of a Judge's bind record (runtime.judge_binds): what it
records, where it keeps it, and which Judges get it."""

from __future__ import annotations

import json
import os
import stat as stat_module
from pathlib import Path, PurePosixPath

import pytest
from docker.types import Mount

from rsi_harness.judge_agent import judge_isolation as isolation
from rsi_harness.models import ContainerMount, ContainerSpec
from rsi_harness.runtime import docker as docker_runtime
from rsi_harness.runtime import judge_binds
from rsi_harness.runtime.mount_topology import validate_split_workdir_target
from tests.runtime.test_docker import FakeDockerClient, make_runtime


def _device(path: Path) -> str:
    return isolation.device_of(os.stat(path))


def _mountinfo(point: Path, kind: str, device: str | None = None) -> str:
    return (
        "22 1 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n"
        f"30 22 {device or _device(point)} / {point} rw,relatime - "
        f"{kind} server:/export rw\n"
    )


def _trees(tmp_path: Path) -> tuple[Path, Path, Path]:
    logs, isolate = tmp_path / "logs" / "verifier", tmp_path / "logs" / "isolate"
    (logs / "deep").mkdir(parents=True)
    (logs / "deep" / "reward.json").write_text("{}")
    (isolate / "lib").mkdir(parents=True)
    (isolate / "lib" / "rsi_isolate.py").write_text("")
    launcher = isolate / "rsi-isolate"
    launcher.write_text("")
    return logs, isolate, launcher


def test_binds_from_nfs_are_recorded_with_what_they_share(tmp_path):
    logs, isolate, launcher = _trees(tmp_path)
    (tmp_path / "link").symlink_to(isolate)
    record = judge_binds.nfs_bind_record(
        [
            (logs, PurePosixPath("/logs/verifier")),
            (tmp_path / "link", "/opt/rsi-isolate/"),
            (launcher, "/usr/local/bin/rsi-isolate"),
            (tmp_path / "missing", "/gone"),
        ],
        _mountinfo(tmp_path, "nfs4"),
    )
    device = _device(tmp_path)
    assert record == {
        "version": 2,
        "binds": {
            "/logs/verifier": {
                "device": device,
                "inode": os.stat(logs).st_ino,
                "disjoint": ["/opt/rsi-isolate", "/usr/local/bin/rsi-isolate"],
            },
            # The launcher is a file in the isolate tree: they share it.
            "/opt/rsi-isolate": {
                "device": device,
                "inode": os.stat(isolate).st_ino,
                "disjoint": ["/logs/verifier"],
            },
            "/usr/local/bin/rsi-isolate": {
                "device": device,
                "inode": os.stat(launcher).st_ino,
                "disjoint": ["/logs/verifier"],
            },
        },
    }
    # What the Judge reads back is exactly this.
    data = json.dumps(record).encode()
    assert isolation._parse_bind_record(data) == record["binds"]


def test_trees_that_share_a_file_are_not_apart(tmp_path):
    """A file reachable from both trees (a hard link here; a directory a
    server shows at two paths is the same) keeps them together, whatever
    their paths say."""
    logs, isolate, _launcher = _trees(tmp_path)
    os.link(logs / "deep" / "reward.json", isolate / "lib" / "copy.json")
    record = judge_binds.nfs_bind_record(
        [(logs, "/logs/verifier"), (isolate, "/opt/rsi-isolate")],
        _mountinfo(tmp_path, "nfs"),
    )
    assert record is not None
    assert [entry["disjoint"] for entry in record["binds"].values()] == [[], []]


def test_a_tree_it_cannot_walk_fully_proves_nothing(tmp_path, monkeypatch):
    logs, isolate, _launcher = _trees(tmp_path)
    binds = [(logs, "/logs/verifier"), (isolate, "/opt/rsi-isolate")]
    text = _mountinfo(tmp_path, "nfs")
    # Too many entries, or too long.
    for budget in ({"entries": 2}, {"seconds": -1.0}):
        record = judge_binds.nfs_bind_record(binds, text, **budget)
        assert [entry["disjoint"] for entry in record["binds"].values()] == [[], []]
    # Each bind has its own bound: a large tree leaves a small one proven
    # against everything but itself.
    large = tmp_path / "logs" / "workspace"
    for index in range(6):
        (large / str(index)).mkdir(parents=True)
    record = judge_binds.nfs_bind_record(
        [*binds, (large, "/workspace")], text, entries=5
    )
    assert record["binds"]["/workspace"]["disjoint"] == []
    assert "/workspace" not in record["binds"]["/logs/verifier"]["disjoint"]
    # A directory it cannot read.
    real_open = os.open

    def refuse(path, flags, *args, **kwargs):
        if Path(path) == isolate / "lib":
            raise PermissionError(path)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(judge_binds.os, "open", refuse)
    record = judge_binds.nfs_bind_record(binds, text)
    assert [entry["disjoint"] for entry in record["binds"].values()] == [[], []]


def _another_device(status: os.stat_result) -> os.stat_result:
    values = list(status)
    values[stat_module.ST_DEV] = status.st_dev + 1
    return os.stat_result(values)


def test_a_tree_with_another_file_system_in_it_proves_nothing(tmp_path, monkeypatch):
    """A mount inside a bind's tree is a mount of its own in the Judge,
    which the record says nothing about; and an automount the walk's open
    triggers is one too."""
    logs, isolate, _launcher = _trees(tmp_path)
    binds = [(logs, "/logs/verifier"), (isolate, "/opt/rsi-isolate")]
    text = _mountinfo(tmp_path, "nfs")
    real_lstat, real_fstat = os.lstat, os.fstat

    def nested(path, *args, **kwargs):
        status = real_lstat(path, *args, **kwargs)
        return _another_device(status) if str(path) == "lib" else status

    monkeypatch.setattr(judge_binds.os, "lstat", nested)
    record = judge_binds.nfs_bind_record(binds, text)
    assert record["binds"]["/logs/verifier"]["disjoint"] == []
    monkeypatch.setattr(judge_binds.os, "lstat", real_lstat)

    opened = []

    def automounted(descriptor):
        status = real_fstat(descriptor)
        opened.append(status.st_ino)
        if status.st_ino == real_lstat(isolate / "lib").st_ino:
            return _another_device(status)
        return status

    monkeypatch.setattr(judge_binds.os, "fstat", automounted)
    record = judge_binds.nfs_bind_record(binds, text)
    assert opened and record["binds"]["/logs/verifier"]["disjoint"] == []


def test_only_the_file_systems_of_hidden_targets_are_walked(tmp_path, monkeypatch):
    logs, isolate, launcher = _trees(tmp_path)
    walked = []
    real = judge_binds._reach

    def spy(source, status, budget):
        walked.append(source)
        return real(source, status, budget)

    monkeypatch.setattr(judge_binds, "_reach", spy)
    text = _mountinfo(tmp_path, "nfs")
    record = judge_binds.nfs_bind_record(
        [(isolate, "/opt/rsi-isolate"), (launcher, "/usr/local/bin/rsi-isolate")],
        text,
    )
    # No hidden target: nothing to set apart, nothing walked.
    assert walked == []
    assert [entry["disjoint"] for entry in record["binds"].values()] == [[], []]
    record = judge_binds.nfs_bind_record(
        [
            (isolate, "/opt/rsi-isolate"),
            (launcher, "/usr/local/bin/rsi-isolate"),
            (logs, "/logs/verifier"),
        ],
        text,
    )
    # The hidden tree first; and only pairs with it are set apart.
    assert walked[0] == str(logs)
    assert record["binds"]["/opt/rsi-isolate"]["disjoint"] == ["/logs/verifier"]


def test_no_record_without_a_bind_from_nfs(tmp_path):
    binds = [(tmp_path, "/logs/verifier")]
    assert judge_binds.nfs_bind_record(binds, _mountinfo(tmp_path, "ext4")) is None
    # A source whose device is not its mount's (a remount, another namespace)
    # is left out.
    assert (
        judge_binds.nfs_bind_record(binds, _mountinfo(tmp_path, "nfs", "0:999")) is None
    )


def test_the_record_is_written_read_only_away_from_nfs(tmp_path, monkeypatch):
    record = {"version": 2, "binds": {}}
    directory = judge_binds.write_bind_record(record, tmp_path / "records")
    path = directory / judge_binds.BIND_RECORD_NAME
    assert json.loads(path.read_text()) == record
    assert path.stat().st_mode & 0o777 == 0o444
    assert directory.stat().st_mode & 0o777 == 0o755
    assert directory.parent == tmp_path / "records"
    judge_binds.remove_bind_record(directory)
    assert not directory.exists()
    judge_binds.remove_bind_record(None)

    monkeypatch.setattr(
        judge_binds, "mount_table", lambda text=None: {"/": ("0:60", "/", "rw", "nfs")}
    )
    with pytest.raises(OSError, match="NFS"):
        judge_binds.write_bind_record(record, tmp_path / "records")
    # Then the Judge gets no record, and treats NFS as it always has.
    monkeypatch.setattr(
        judge_binds, "nfs_bind_record", lambda binds, mountinfo=None: record
    )
    assert judge_binds.bind_record_for([], tmp_path / "records") is None
    assert list((tmp_path / "records").iterdir()) == []


def test_split_workdir_cannot_cover_the_record():
    with pytest.raises(Exception, match="judge-binds"):
        validate_split_workdir_target(PurePosixPath("/run/rsi-harness/judge-binds/w"))


# -- local Docker ---------------------------------------------------------------


def _judge_spec(logs: Path) -> ContainerSpec:
    return ContainerSpec(
        image="image",
        workdir=PurePosixPath("/workspace"),
        mounts=(ContainerMount(source=logs, target=PurePosixPath("/logs/verifier")),),
    )


def test_a_root_docker_judge_gets_the_record_of_its_binds(tmp_path, monkeypatch):
    client = FakeDockerClient()
    logs = tmp_path / "engine-root" / "logs"
    logs.mkdir(parents=True)
    record = tmp_path / "record"
    record.mkdir()
    seen, removed = [], []

    def fake_record(binds, parent):
        seen.append((list(binds), parent))
        return record

    monkeypatch.setattr(docker_runtime, "bind_record_for", fake_record)
    monkeypatch.setattr(docker_runtime, "remove_bind_record", removed.append)
    monkeypatch.setattr(docker_runtime.os, "geteuid", lambda: 0)
    runtime = make_runtime(client, tmp_path, role="judge")

    ref = runtime.create(_judge_spec(logs))

    assert seen == [
        ([(str(logs), "/logs/verifier")], docker_runtime.JUDGE_BIND_RECORD_ROOT)
    ]
    assert client.containers.created[0]["mounts"][-1] == Mount(
        source=str(record),
        target="/run/rsi-harness/judge-binds",
        type="bind",
        read_only=True,
    )
    runtime.remove(ref)
    assert removed == [record]


def test_only_a_root_docker_judge_gets_a_record(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        docker_runtime,
        "bind_record_for",
        lambda binds, parent: calls.append(binds) or tmp_path,
    )
    logs = tmp_path / "engine-root" / "logs"
    logs.mkdir(parents=True)
    # Not root: no host bind it could record.
    monkeypatch.setattr(docker_runtime.os, "geteuid", lambda: 1000)
    client = FakeDockerClient()
    make_runtime(client, tmp_path, role="judge").create(_judge_spec(logs))
    assert calls == []
    assert all(
        mount["Target"] != "/run/rsi-harness/judge-binds"
        for mount in client.containers.created[0]["mounts"]
    )


def test_a_failed_docker_create_removes_the_record(tmp_path, monkeypatch):
    from docker.errors import APIError

    client = FakeDockerClient()
    logs = tmp_path / "engine-root" / "logs"
    logs.mkdir(parents=True)
    removed = []
    monkeypatch.setattr(
        docker_runtime, "bind_record_for", lambda binds, parent: tmp_path / "r"
    )
    monkeypatch.setattr(docker_runtime, "remove_bind_record", removed.append)
    monkeypatch.setattr(docker_runtime.os, "geteuid", lambda: 0)

    def refuse(*_args, **_kwargs):
        raise APIError("refused")

    monkeypatch.setattr(client.containers, "create", refuse)
    with pytest.raises(Exception, match="failed to create"):
        make_runtime(client, tmp_path, role="judge").create(_judge_spec(logs))
    assert removed == [tmp_path / "r"]
