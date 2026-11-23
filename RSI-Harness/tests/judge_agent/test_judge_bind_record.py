"""The Harness's record of a Judge's NFS binds (judge_isolation.BIND_RECORD):
NFS shows "/" as the root of every mount, so only the binds it records are
told apart from a hidden path; anything it cannot decide stays an alias."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from rsi_harness.judge_agent import judge_isolation as isolation


def _device(path: Path) -> str:
    return isolation.device_of(os.lstat(path))


class Judge:
    """Real directories standing for a Judge's mount points (so lstat sees
    real devices and inodes), all on one NFS mount as far as mountinfo
    says."""

    def __init__(self, root: Path, *names: str) -> None:
        self.root = root
        self.points = {}
        for name in names:
            path = root / name
            path.mkdir(parents=True)
            self.points[name] = str(path)
        self.device = _device(root)

    def mountinfo(self, kind: str = "nfs4", extra: str = "") -> str:
        lines = ["22 1 0:50 / / rw,relatime - overlay overlay rw"]
        for index, point in enumerate(self.points.values()):
            lines.append(
                f"{30 + index} 22 {self.device} / {point} rw,relatime - "
                f"{kind} server:/export rw"
            )
        return "\n".join(lines) + "\n" + extra

    def entry(self, name: str, *disjoint: str) -> dict:
        return {
            "device": self.device,
            "inode": os.lstat(self.points[name]).st_ino,
            "disjoint": sorted(self.points[other] for other in disjoint),
        }

    def apart(self) -> dict:
        """Each recorded bind shown to share no file with the others."""
        names = list(self.points)
        return {
            self.points[name]: self.entry(name, *(n for n in names if n != name))
            for name in names
        }


@pytest.fixture
def judge(tmp_path: Path) -> Judge:
    return Judge(tmp_path, "verifier", "isolate", "run")


@pytest.fixture
def local(tmp_path: Path):
    """A directory on another device than ``tmp_path``, as the record's own
    storage is never the NFS file system of the binds."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("no /dev/shm")
    directory = Path(tempfile.mkdtemp(dir="/dev/shm"))
    try:
        if _device(directory) == _device(tmp_path):
            pytest.skip("/dev/shm shares the test directory's device")
        yield directory
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def test_recorded_binds_that_share_no_file_are_not_aliases(judge: Judge) -> None:
    verifier = judge.points["verifier"]
    assert (
        isolation.mount_aliases([verifier], judge.mountinfo(), record=judge.apart())
        == []
    )


@pytest.mark.parametrize(
    "spoil",
    [
        # The Harness found a shared file, or could not walk one of them.
        lambda record, judge: record[judge.points["isolate"]]["disjoint"].remove(
            judge.points["verifier"]
        ),
        lambda record, judge: record[judge.points["verifier"]]["disjoint"].remove(
            judge.points["isolate"]
        ),
        # The bind at the point is not the recorded directory.
        lambda record, judge: record[judge.points["isolate"]].update(inode=1),
        lambda record, judge: record[judge.points["verifier"]].update(inode=1),
        lambda record, judge: record[judge.points["isolate"]].update(device="9:9"),
        # Not recorded at all (a site bind, a mount the Harness did not make).
        lambda record, judge: record.pop(judge.points["isolate"]),
        lambda record, judge: record.pop(judge.points["verifier"]),
    ],
)
def test_what_the_record_cannot_show_apart_stays_an_alias(judge: Judge, spoil) -> None:
    """Where it always was: the whole other mount."""
    verifier = judge.points["verifier"]
    record = judge.apart()
    spoil(record, judge)
    aliases = isolation.mount_aliases([verifier], judge.mountinfo(), record=record)
    assert judge.points["isolate"] + "/" in aliases


def test_without_a_record_every_nfs_bind_stays_an_alias(judge: Judge) -> None:
    verifier = judge.points["verifier"]
    for record in (None, {}):
        assert isolation.mount_aliases(
            [verifier], judge.mountinfo(), record=record
        ) == sorted(judge.points[name] + "/" for name in ("isolate", "run"))


def test_a_record_changes_nothing_off_nfs(judge: Judge) -> None:
    """Other file systems keep their exact answer, whatever a record says."""
    verifier = judge.points["verifier"]
    text = judge.mountinfo(kind="ext4")
    assert isolation.mount_aliases(
        [verifier], text, record=judge.apart()
    ) == isolation.mount_aliases([verifier], text)
    assert isolation.mount_aliases([verifier], text) == sorted(
        judge.points[name] + "/" for name in ("isolate", "run")
    )


# -- reading the record ---------------------------------------------------------


class RecordMount:
    """A record file at a test path, with a mountinfo line that binds its
    directory read-only from a local file system."""

    def __init__(self, root: Path, monkeypatch, binds: dict) -> None:
        self.directory = root / "judge-binds"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "binds.json"
        self.path.write_text(json.dumps({"version": 2, "binds": binds}))
        self.path.chmod(0o444)
        monkeypatch.setattr(isolation, "BIND_RECORD_DIR", str(self.directory))
        monkeypatch.setattr(isolation, "BIND_RECORD", str(self.path))
        real = os.fstatvfs

        def read_only(descriptor):
            result = real(descriptor)
            return os.statvfs_result(
                tuple(result[:8]) + (result.f_flag | os.ST_RDONLY,) + tuple(result[9:])
            )

        monkeypatch.setattr(os, "fstatvfs", read_only)

    def line(self, options: str = "ro,relatime", kind: str = "ext4") -> str:
        return (
            f"90 22 {_device(self.path)} /binds {self.directory} {options} - "
            f"{kind} /dev/sda1 rw\n"
        )


def test_the_harness_record_decides_through_hidden_aliases(
    judge: Judge, local: Path, monkeypatch
) -> None:
    """End to end: read from its read-only mount, it lets rsi-isolate grant
    the Harness's own NFS binds."""
    verifier = judge.points["verifier"]
    binds = judge.apart()
    mount = RecordMount(local / "record", monkeypatch, binds)
    text = judge.mountinfo(extra=mount.line())
    monkeypatch.setattr(isolation, "_mountinfo", lambda: text)
    assert isolation.hidden_aliases([verifier], [judge.points["isolate"]], 125) == []
    assert isolation.bind_record(isolation.mount_table(text), text) == binds


@pytest.mark.parametrize(
    "spoil",
    [
        lambda mount: mount.line(options="rw,relatime"),
        lambda mount: mount.line(kind="nfs4"),
        lambda mount: "",
        # Another mount shows the record's directory, so the Judge could
        # rewrite the record there.
        lambda mount: (
            mount.line()
            + f"91 22 {_device(mount.path)} /binds /elsewhere rw - ext4 /dev/sda1 rw\n"
        ),
        lambda mount: (
            mount.line()
            + f"91 22 {_device(mount.path)} / /whole rw - ext4 /dev/sda1 rw\n"
        ),
        lambda mount: mount.line().replace(_device(mount.path), "9:99"),
    ],
)
def test_a_record_the_judge_could_have_written_is_ignored(
    judge: Judge, local: Path, monkeypatch, spoil
) -> None:
    verifier = judge.points["verifier"]
    binds = judge.apart()
    mount = RecordMount(local / "record", monkeypatch, binds)
    text = judge.mountinfo(extra=spoil(mount))
    assert isolation.bind_record(isolation.mount_table(text), text) is None
    monkeypatch.setattr(isolation, "_mountinfo", lambda: text)
    with pytest.raises(isolation.Failure, match="is another mount of a hidden path"):
        isolation.hidden_aliases([verifier], [judge.points["isolate"]], 125)


def test_a_record_file_anyone_could_change_is_ignored(
    judge: Judge, local: Path, tmp_path: Path, monkeypatch
) -> None:
    mount = RecordMount(local / "record", monkeypatch, {})
    text = judge.mountinfo(extra=mount.line())
    table = isolation.mount_table(text)
    assert isolation.bind_record(table, text) == {}
    mount.path.chmod(0o666)
    assert isolation.bind_record(table, text) is None
    mount.path.chmod(0o444)
    monkeypatch.setattr(os, "geteuid", lambda: os.getuid() + 1)
    if os.getuid() != 0:
        assert isolation.bind_record(table, text) is None
    monkeypatch.undo()
    # A link in its place is not followed.
    mount = RecordMount(local / "link", monkeypatch, {})
    mount.path.unlink()
    (tmp_path / "elsewhere.json").write_text('{"version": 2, "binds": {}}')
    mount.path.symlink_to(tmp_path / "elsewhere.json")
    text = judge.mountinfo(extra=mount.line())
    assert isolation.bind_record(isolation.mount_table(text), text) is None


GOOD = {"device": "0:60", "inode": 7, "disjoint": ["/b"]}


@pytest.mark.parametrize(
    "record",
    [
        b"not json",
        b"\xff",
        json.dumps([]).encode(),
        json.dumps({"version": 1, "binds": {}}).encode(),
        json.dumps({"version": 2, "binds": {}, "more": 1}).encode(),
        json.dumps({"version": 2, "binds": {"relative": GOOD}}).encode(),
        json.dumps({"version": 2, "binds": {"/a/../b": GOOD}}).encode(),
        json.dumps({"version": 2, "binds": {"/a": {**GOOD, "extra": 1}}}).encode(),
        json.dumps({"version": 2, "binds": {"/a": {**GOOD, "inode": True}}}).encode(),
        json.dumps({"version": 2, "binds": {"/a": {**GOOD, "inode": 0}}}).encode(),
        json.dumps({"version": 2, "binds": {"/a": {**GOOD, "device": "60"}}}).encode(),
        json.dumps(
            {"version": 2, "binds": {"/a": {**GOOD, "disjoint": "/b"}}}
        ).encode(),
        json.dumps(
            {"version": 2, "binds": {"/a": {**GOOD, "disjoint": ["b"]}}}
        ).encode(),
    ],
)
def test_a_malformed_record_is_ignored_whole(record: bytes) -> None:
    assert isolation._parse_bind_record(record) is None


def test_a_well_formed_record_is_read() -> None:
    data = json.dumps({"version": 2, "binds": {"/a": GOOD, "/": GOOD}}).encode()
    assert isolation._parse_bind_record(data) == {"/a": GOOD, "/": GOOD}


def test_a_record_that_is_not_a_small_file_on_a_read_only_mount_is_ignored(
    judge: Judge, local: Path, monkeypatch
) -> None:
    real_fstatvfs = os.fstatvfs
    mount = RecordMount(local / "record", monkeypatch, judge.apart())
    text = judge.mountinfo(extra=mount.line())
    table = isolation.mount_table(text)
    assert isolation.bind_record(table, text) is not None
    # Larger than the bound.
    monkeypatch.setattr(isolation, "_BIND_RECORD_MAX_BYTES", 8)
    assert isolation.bind_record(table, text) is None
    monkeypatch.setattr(isolation, "_BIND_RECORD_MAX_BYTES", 1 << 20)
    # A file system the kernel does not hold read-only.
    monkeypatch.setattr(os, "fstatvfs", real_fstatvfs)
    assert isolation.bind_record(table, text) is None
    # Not a regular file.
    mount = RecordMount(local / "directory", monkeypatch, {})
    mount.path.unlink()
    mount.path.mkdir()
    text = judge.mountinfo(extra=mount.line())
    assert isolation.bind_record(isolation.mount_table(text), text) is None
