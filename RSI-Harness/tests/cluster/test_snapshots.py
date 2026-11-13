from __future__ import annotations

import contextlib
import errno
import mmap
import os
import shutil
import socket
import stat
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from rsi_harness.cluster import snapshots
from rsi_harness.cluster.snapshots import (
    RETAINED,
    JudgeSnapshots,
    remove_retained,
)
from rsi_harness.errors import InfrastructureError

REAL_CHECK = snapshots._mapped_writes_change_ctime


@pytest.fixture(autouse=True)
def _reuse_on_any_tmp_file_system(monkeypatch):
    """The tests' tmp directory may be tmpfs, where nothing is reused."""
    monkeypatch.setattr(snapshots, "_mapped_writes_change_ctime", lambda _fd: True)


def _fs_of(path: Path) -> bool:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return REAL_CHECK(fd)
    finally:
        os.close(fd)


def source(tmp_path: Path) -> Path:
    path = tmp_path / "source"
    path.mkdir()
    (path / "answer.py").write_text("original")
    return path


def settle(directory: Path) -> None:
    """Wait until the file system clock has moved past every change so far,
    as Agent time does between rounds."""

    def now() -> int:
        marker = directory / ".settle"
        marker.touch()
        try:
            return marker.stat().st_ctime_ns
        finally:
            marker.unlink()

    first = now()
    while now() <= first:
        pass


def work_tree(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    (work / "src" / "deep").mkdir(parents=True)
    (work / "answer.py").write_text("answer = 42\n")
    (work / "src" / "model.py").write_text("weights = [1, 2, 3]\n")
    (work / "src" / "deep" / "data.bin").write_bytes(os.urandom(70_000))
    (work / "src" / "run.sh").write_text("#!/bin/sh\n")
    (work / "src" / "run.sh").chmod(0o750)
    os.utime(work / "answer.py", ns=(1_000_000_000, 2_000_000_000))
    settle(tmp_path)
    return work


def files(root: Path) -> dict[str, os.stat_result]:
    found = {}
    for directory, _dirs, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink():
                found[path.relative_to(root).as_posix()] = path.lstat()
    return found


@contextlib.contextmanager
def held(root: Path) -> Iterator[dict[str, os.stat_result]]:
    """The files below ``root``, kept open so that their inode numbers are
    not reused while the test compares them."""
    found = files(root)
    descriptors = [os.open(root / path, os.O_RDONLY) for path in found]
    try:
        yield found
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


_KEPT: list[int] = []


@pytest.fixture(autouse=True)
def _close_kept() -> Iterator[None]:
    yield
    while _KEPT:
        os.close(_KEPT.pop())


def kept(root: Path) -> dict[str, os.stat_result]:
    """As ``held``, until the test ends: the next round removes retained files
    it does not link, whose inode numbers could otherwise be reused."""
    found = files(root)
    _KEPT.extend(os.open(root / path, os.O_RDONLY) for path in found)
    return found


def linked(before: dict[str, os.stat_result], after: Path) -> set[str]:
    """Paths whose inode in ``after`` is the one they had in ``before``."""
    return {
        path
        for path, found in files(after).items()
        if path in before and before[path].st_ino == found.st_ino
    }


def shared_inodes(left: Path, right: Path) -> set[tuple[int, int]]:
    def inodes(root: Path) -> set[tuple[int, int]]:
        return {(found.st_dev, found.st_ino) for found in files(root).values()}

    return inodes(left) & inodes(right)


def contents(root: Path) -> dict[str, bytes]:
    return {path: (root / path).read_bytes() for path in files(root)}


def take(snaps: JudgeSnapshots, work: Path, name: str, **kwargs):
    """Acquire as a reusing round does: the clock is read before the pause."""
    return snaps.acquire(work, name, since=snaps.clock(), **kwargs)


def one_round(snaps: JudgeSnapshots, work: Path, name: str) -> Path:
    """Acquire, then retain as a finished round does; the retained copy."""
    lease = take(snaps, work, name)
    snaps.retain(lease)
    return snaps.rounds / RETAINED


def test_first_round_copies_content_modes_and_times(tmp_path):
    work = work_tree(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_text("not copied")
    (work / "linked-dir").symlink_to(outside, target_is_directory=True)
    (work / "dangling").symlink_to("missing")
    os.link(work / "answer.py", work / "hard-link.py")
    readonly = work / "readonly"
    readonly.mkdir()
    (readonly / "data").write_text("keep")
    readonly.chmod(0o500)
    snaps = JudgeSnapshots(tmp_path / "rounds")

    copy = take(snaps, work, "agent-1").workspace

    assert contents(copy) == contents(work)
    for path, found in files(work).items():
        copied = (copy / path).lstat()
        assert copied.st_mode == found.st_mode
        assert copied.st_mtime_ns == found.st_mtime_ns
    assert os.readlink(copy / "linked-dir") == str(outside)
    assert os.readlink(copy / "dangling") == "missing"
    assert (copy / "readonly").stat().st_mode & 0o777 == 0o500
    assert (copy / "readonly" / "data").read_text() == "keep"
    # An internal hard link becomes two independent copies.
    assert (copy / "answer.py").stat().st_ino != (copy / "hard-link.py").stat().st_ino
    assert (copy / "answer.py").stat().st_nlink == 1
    assert not shared_inodes(work, copy)
    readonly.chmod(0o700)


def test_an_unchanged_round_links_every_file_and_shares_nothing_with_work(
    tmp_path,
):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    retained = one_round(snaps, work, "agent-1")
    before = kept(retained)

    copy = take(snaps, work, "agent-2").workspace

    assert linked(before, copy) == set(files(work))
    assert contents(copy) == contents(work)
    assert not shared_inodes(work, copy)
    # The retained copy is gone once the new one is complete.
    assert sorted(path.name for path in snaps.rounds.iterdir()) == ["agent-2"]


def _rewrite_same_size(work: Path) -> set[str]:
    path = work / "src" / "model.py"
    found = path.stat()
    path.write_text("weights = [9, 9, 9]\n")
    os.utime(path, ns=(found.st_atime_ns, found.st_mtime_ns))
    return {"src/model.py"}


def _append(work: Path) -> set[str]:
    with (work / "answer.py").open("a") as stream:
        stream.write("more = 1\n")
    return {"answer.py"}


def _recreate(work: Path) -> set[str]:
    path = work / "src" / "model.py"
    found = path.stat()
    path.unlink()
    path.write_text("weights = [7, 7, 7]\n")
    os.utime(path, ns=(found.st_atime_ns, found.st_mtime_ns))
    return {"src/model.py"}


def _chmod(work: Path) -> set[str]:
    (work / "src" / "run.sh").chmod(0o700)
    return {"src/run.sh"}


def _swap(work: Path) -> set[str]:
    left, right = work / "answer.py", work / "src" / "model.py"
    right.write_text("answer = 42\n")  # the same size as the left file
    settle(work.parent)
    os.utime(right, ns=(left.stat().st_atime_ns, left.stat().st_mtime_ns))
    parked = work / "parked"
    left.rename(parked)
    right.rename(left)
    parked.rename(right)
    return {"answer.py", "src/model.py"}


def _add_and_delete(work: Path) -> set[str]:
    (work / "src" / "deep" / "data.bin").unlink()
    (work / "new.txt").write_text("new")
    return {"new.txt"}


@pytest.mark.parametrize(
    "mutate", [_rewrite_same_size, _append, _recreate, _chmod, _swap, _add_and_delete]
)
def test_each_change_is_copied_again(tmp_path, mutate):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    before = kept(one_round(snaps, work, "agent-1"))
    changed = mutate(work)

    copy = take(snaps, work, "agent-2").workspace

    assert contents(copy) == contents(work)
    assert stat.S_IMODE((copy / "src/run.sh").stat().st_mode) == stat.S_IMODE(
        (work / "src/run.sh").stat().st_mode
    )
    assert linked(before, copy) == set(files(work)) - changed
    assert not shared_inodes(work, copy)


def test_retained_files_not_linked_are_removed_before_copying(tmp_path, monkeypatch):
    """A round holds about one copy: the old version of a changed file, and
    files Work deleted, leave the retained copy before anything is copied."""
    work = work_tree(tmp_path)
    (work / "old-checkpoint").mkdir()
    (work / "old-checkpoint" / "weights").write_bytes(os.urandom(50_000))
    # Read-only directories, in the retained copy too.
    (work / "old-checkpoint").chmod(0o500)
    (work / "src").chmod(0o500)
    settle(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    retained = one_round(snaps, work, "agent-1")
    (work / "old-checkpoint").chmod(0o700)
    shutil.rmtree(work / "old-checkpoint")
    (work / "src" / "model.py").write_text("weights = [4, 5, 6, 7]\n")
    (work / "new-checkpoint").write_bytes(os.urandom(50_000))
    real = snapshots._copy_file
    seen = []

    def record(source, target, name, path, walk):
        seen.append((path, sorted(files(retained))))
        real(source, target, name, path, walk)

    monkeypatch.setattr(snapshots, "_copy_file", record)
    copy = take(snaps, work, "agent-2").workspace

    assert seen == [
        (
            "new-checkpoint",
            ["answer.py", "src/deep/data.bin", "src/model.py", "src/run.sh"],
        ),
        ("src/model.py", ["answer.py", "src/deep/data.bin", "src/run.sh"]),
    ]
    assert contents(copy) == contents(work)
    assert stat.S_IMODE((copy / "src").stat().st_mode) == 0o500


def test_a_file_changed_during_its_copy_is_copied_again(tmp_path, monkeypatch):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    real = snapshots._copy_content
    target = work / "src" / "model.py"

    def racing(read_fd, write_fd, size):
        copied = real(read_fd, write_fd, size)
        if os.fstat(read_fd).st_ino == target.stat().st_ino:
            # A Work woken during the round rewrites the file meanwhile.
            found = target.stat()
            target.write_text("weights = [5, 5, 5]\n")
            os.utime(target, ns=(found.st_atime_ns, found.st_mtime_ns))
        return copied

    monkeypatch.setattr(snapshots, "_copy_content", racing)
    first = take(snaps, work, "agent-1")
    assert (first.workspace / "src/model.py").read_text() == "weights = [1, 2, 3]\n"
    monkeypatch.setattr(snapshots, "_copy_content", real)
    snaps.retain(first)
    before = kept(snaps.rounds / RETAINED)

    copy = take(snaps, work, "agent-2").workspace

    assert (copy / "src/model.py").read_text() == "weights = [5, 5, 5]\n"
    assert linked(before, copy) == set(files(work)) - {"src/model.py"}


def test_a_write_through_an_earlier_mapping_is_copied_again(tmp_path, monkeypatch):
    """syncfs write-protects the mapping, so its next write updates ctime."""
    monkeypatch.setattr(snapshots, "_mapped_writes_change_ctime", REAL_CHECK)
    if not _fs_of(tmp_path):
        pytest.skip("the tmp directory has no writeback")
    work = work_tree(tmp_path)
    descriptor = os.open(work / "src" / "deep" / "data.bin", os.O_RDWR)
    try:
        with mmap.mmap(descriptor, 4096) as mapped:
            mapped[:4] = b"AAAA"  # dirty before the first round
            settle(tmp_path)
            snaps = JudgeSnapshots(tmp_path / "rounds")
            retained = one_round(snaps, work, "agent-1")
            assert (retained / "src/deep/data.bin").read_bytes()[:4] == b"AAAA"
            mapped[:4] = b"BBBB"  # no system call
            copy = take(snaps, work, "agent-2").workspace
    finally:
        os.close(descriptor)

    assert (copy / "src/deep/data.bin").read_bytes()[:4] == b"BBBB"


def test_a_workspace_on_tmpfs_is_copied_whole_and_not_kept(tmp_path, monkeypatch):
    """tmpfs never writes pages back: a write through a mapping leaves ctime
    unchanged, so no file there is reused."""
    monkeypatch.setattr(snapshots, "_mapped_writes_change_ctime", REAL_CHECK)
    if not Path("/dev/shm").is_dir():
        pytest.skip("no /dev/shm")
    with tempfile.TemporaryDirectory(dir="/dev/shm") as other:
        assert not _fs_of(Path(other))
        work = work_tree(Path(other))
        snaps = JudgeSnapshots(Path(other) / "rounds")
        lease = take(snaps, work, "agent-1")
        assert contents(lease.workspace) == contents(work)
        snaps.retain(lease)
        assert os.listdir(snaps.rounds) == []


def test_a_file_changed_after_the_clock_is_copied_again(tmp_path, monkeypatch):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    real = snapshots._syncfs

    def change_then_sync(fd):
        # After the clock, before the walk: the copy has the new content,
        # but the file may change again unseen within the same clock tick.
        (work / "answer.py").write_text("answer = 43\n")
        real(fd)

    monkeypatch.setattr(snapshots, "_syncfs", change_then_sync)
    first = take(snaps, work, "agent-1")
    monkeypatch.setattr(snapshots, "_syncfs", real)
    assert (first.workspace / "answer.py").read_text() == "answer = 43\n"
    snaps.retain(first)
    before = kept(snaps.rounds / RETAINED)

    copy = take(snaps, work, "agent-2").workspace

    assert linked(before, copy) == set(files(work)) - {"answer.py"}


def test_syncfs_runs_before_the_walk_with_an_os_sync_fallback(tmp_path, monkeypatch):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    seen = []
    real = snapshots._syncfs

    def record(fd):
        seen.append(sorted(os.listdir(snaps.rounds / "agent-1")))
        real(fd)

    monkeypatch.setattr(snapshots, "_syncfs", record)
    take(snaps, work, "agent-1")
    assert seen == [[]]

    monkeypatch.setattr(snapshots, "_syncfs", real)
    synced = []

    class NoSyncfs:
        def __init__(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr(snapshots.ctypes, "CDLL", NoSyncfs)
    monkeypatch.setattr(os, "sync", lambda: synced.append(True))
    copy = take(snaps, work, "agent-2").workspace
    assert synced == [True]
    assert contents(copy) == contents(work)


def _rewrite_retained_file(retained: Path) -> None:
    path = retained / "answer.py"
    found = path.stat()
    path.chmod(0o600)
    path.write_text("answer = 00\n")
    os.utime(path, ns=(found.st_atime_ns, found.st_mtime_ns))
    path.chmod(stat.S_IMODE(found.st_mode))


def _symlink_retained_directory(retained: Path) -> None:
    work = retained.parent.parent / "work"
    (retained / "src").rename(retained / "src.moved")
    (retained / "src").symlink_to(work / "src", target_is_directory=True)


@pytest.mark.parametrize(
    "tamper", [_rewrite_retained_file, _symlink_retained_directory]
)
def test_a_changed_retained_copy_gives_a_full_copy(
    tmp_path, tamper, capsys, monkeypatch
):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    retained = one_round(snaps, work, "agent-1")
    before = kept(retained)
    tamper(retained)
    real = snapshots._copy_workspace
    held_during = []

    def record(source, root_fd, lease, base, walk):
        held_during.append(os.path.lexists(snaps.rounds / RETAINED))
        return real(source, root_fd, lease, base, walk)

    monkeypatch.setattr(snapshots, "_copy_workspace", record)
    copy = take(snaps, work, "agent-2").workspace

    assert contents(copy) == contents(work)
    assert linked(before, copy) == set()
    assert not shared_inodes(work, copy)
    assert "judge_snapshot_base_changed: agent-2" in capsys.readouterr().out
    # The full copy starts once the retained copy is gone.
    assert held_during == [True, False]


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens every file")
def test_special_and_unreadable_entries_are_omitted_and_logged(
    tmp_path, monkeypatch, capsys
):
    work = work_tree(tmp_path)
    os.mkfifo(work / "src" / "pipe")
    monkeypatch.chdir(work / "src")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind("control.sock")  # relative: AF_UNIX paths are short
    (work / "secret.txt").write_text("hidden")
    (work / "secret.txt").chmod(0)
    (work / "closed").mkdir()
    (work / "closed" / "inside").write_text("hidden")
    (work / "closed").chmod(0)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    try:
        copy = take(snaps, work, "agent-1").workspace
    finally:
        listener.close()
        (work / "closed").chmod(0o700)

    for omitted in ("src/pipe", "src/control.sock", "secret.txt", "closed"):
        assert not os.path.lexists(copy / omitted)
    assert (copy / "answer.py").read_text() == "answer = 42\n"
    log = capsys.readouterr().out
    assert "judge_snapshot_omitted: agent-1 " in log
    for omitted in ("src/pipe", "src/control.sock", "secret.txt", "closed"):
        assert omitted in log


def test_the_omitted_log_shows_five_paths(tmp_path, capsys):
    work = tmp_path / "work"
    work.mkdir()
    for index in range(7):
        os.mkfifo(work / f"pipe-{index}")
    JudgeSnapshots(tmp_path / "rounds").acquire(work, "agent-1")
    line = capsys.readouterr().out.strip()
    assert line == (
        "[rsi-engine] judge_snapshot_omitted: agent-1 "
        "pipe-0 pipe-1 pipe-2 pipe-3 pipe-4 (+2)"
    )


def test_the_omitted_log_is_one_line_whatever_the_names(tmp_path, capsys):
    work = tmp_path / "work"
    work.mkdir()
    os.mkfifo(work / "a\n[rsi-engine] forged")
    JudgeSnapshots(tmp_path / "rounds").acquire(work, "agent-1")
    assert capsys.readouterr().out == (
        "[rsi-engine] judge_snapshot_omitted: agent-1 a\\n[rsi-engine] forged\n"
    )


def test_directories_deeper_than_the_limit_are_omitted(tmp_path, monkeypatch, capsys):
    """Each level holds descriptors and stack frames: a deep tree is cut at
    the limit, never an error, under a 1024 descriptor limit."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1024:
        pytest.skip("descriptor limit below 1024")
    work = tmp_path / "work"
    deep = work.joinpath(*["d"] * (snapshots.MAX_DEPTH + 150))
    deep.mkdir(parents=True)
    (deep / "file").write_text("deep")
    limit = work.joinpath(*["d"] * snapshots.MAX_DEPTH)
    (limit / "kept").write_text("kept")
    settle(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    resource.setrlimit(resource.RLIMIT_NOFILE, (1024, hard))
    try:
        one_round(snaps, work, "agent-1")
        (limit / "kept").write_text("again")
        copy = take(snaps, work, "agent-2").workspace
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    below = copy.joinpath(*["d"] * snapshots.MAX_DEPTH)
    assert (below / "kept").read_text() == "again"
    assert os.listdir(below) == ["kept"]
    shown = "/".join(["d"] * (snapshots.MAX_DEPTH + 1))[:200] + "..."
    assert f"judge_snapshot_omitted: agent-2 {shown}\n" in capsys.readouterr().out


def test_a_file_system_without_links_copies_for_the_rest_of_the_run(
    tmp_path, monkeypatch
):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    with held(one_round(snaps, work, "agent-1")) as before:
        _without_links(snaps, work, before, monkeypatch)


def _without_links(snaps, work, before, monkeypatch):
    real = os.link
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise OSError(errno.EPERM, "no hard links here")

    monkeypatch.setattr(os, "link", refuse)
    lease = take(snaps, work, "agent-2")
    assert len(calls) == 1
    assert linked(before, lease.workspace) == set()
    assert contents(lease.workspace) == contents(work)
    assert snaps.links_ok is False
    snaps.retain(lease)
    # Nothing can be linked from it: not kept.
    assert sorted(os.listdir(snaps.rounds)) == []
    monkeypatch.setattr(os, "link", real)

    copy = take(snaps, work, "agent-3").workspace

    assert linked(before, copy) == set()
    assert contents(copy) == contents(work)
    assert snaps.links_ok is False


def test_another_device_disables_reuse(tmp_path):
    if not Path("/dev/shm").is_dir():
        pytest.skip("no /dev/shm")
    with tempfile.TemporaryDirectory(dir="/dev/shm") as other:
        if Path(other).stat().st_dev == tmp_path.stat().st_dev:
            pytest.skip("/dev/shm is on the same device")
        work = work_tree(tmp_path)
        snaps = JudgeSnapshots(Path(other) / "rounds")
        assert not snapshots.can_link(work, snaps.rounds.parent)
        lease = take(snaps, work, "agent-1")
        snaps.retain(lease)
        # Nothing could be linked from it: not kept.
        assert os.listdir(snaps.rounds) == []
        copy = take(snaps, work, "agent-2").workspace
        assert contents(copy) == contents(work)
        assert all(found.st_nlink == 1 for found in files(copy).values())


def test_no_reuse_copies_every_file_and_keeps_no_copy(tmp_path, monkeypatch):
    """Without ``since`` nothing is kept, so there is no syncfs or clock
    wait: the round is a plain copy."""
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    seen = []
    real = snapshots._copy_workspace

    def record(source, root_fd, lease, base, walk):
        # Removed before the copy starts: one copy at a time.
        seen.append(sorted(os.listdir(snaps.rounds)))
        return real(source, root_fd, lease, base, walk)

    def unused(*_args):
        raise AssertionError("not needed without reuse")

    with held(one_round(snaps, work, "agent-1")) as before:
        monkeypatch.setattr(snapshots, "_copy_workspace", record)
        for name in ("_syncfs", "_tick", "_fs_now_ns"):
            monkeypatch.setattr(snapshots, name, unused)
        lease = snaps.acquire(work, "agent-2")
        assert linked(before, lease.workspace) == set()

    assert seen == [["agent-2"]]
    assert contents(lease.workspace) == contents(work)
    snaps.retain(lease)
    assert os.listdir(snaps.rounds) == []


def test_a_write_that_starts_after_the_clock_is_never_reused(tmp_path, monkeypatch):
    """Work wakes itself after the pause signal and rewrites a file in place:
    its ctime is set when the write starts, before the copy reads it, so the
    copy may be torn while Work's metadata stays the same. A clock read
    before the pause signal keeps that copy from being linked later."""
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    since = snaps.clock()  # before the pause signal
    _rewrite_same_size(work)  # the write starts after it
    real = snapshots._copy_content
    model = work / "src" / "model.py"

    def torn(read_fd, write_fd, size):
        if os.fstat(read_fd).st_ino == model.stat().st_ino:
            # What a copy that overtook the write would read.
            os.write(write_fd, b"torn")
        return real(read_fd, write_fd, size)

    monkeypatch.setattr(snapshots, "_copy_content", torn)
    first = snaps.acquire(work, "agent-1", since=since)
    monkeypatch.setattr(snapshots, "_copy_content", real)
    assert (first.workspace / "src/model.py").read_bytes().startswith(b"torn")
    snaps.retain(first)
    before = kept(snaps.rounds / RETAINED)
    settle(tmp_path)

    copy = take(snaps, work, "agent-2").workspace

    assert contents(copy) == contents(work)
    assert linked(before, copy) == set(files(work)) - {"src/model.py"}


def test_the_clock_reads_the_rounds_file_system(tmp_path):
    snaps = JudgeSnapshots(tmp_path / "rounds")
    since = snaps.clock()
    assert os.listdir(snaps.rounds) == []
    (tmp_path / "later").write_text("x")
    assert (tmp_path / "later").stat().st_ctime_ns >= since


def test_retain_keeps_one_copy_and_every_inode_predates_j0(tmp_path):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    lease = take(snaps, work, "agent-1")
    snaps.retain(lease)
    assert sorted(os.listdir(snaps.rounds)) == [RETAINED]
    assert contents(snaps.rounds / RETAINED) == contents(work)

    lease = take(snaps, work, "agent-2")

    assert sorted(os.listdir(snaps.rounds)) == ["agent-2"]
    j0 = snaps._pending[1].j0_ns
    for found in files(lease.workspace).values():
        assert found.st_ctime_ns < j0
    for directory, _dirs, _names in os.walk(lease.workspace):
        assert Path(directory).lstat().st_ctime_ns < j0
    with pytest.raises(InfrastructureError, match="not acquired"):
        snaps.retain(lease.__class__(lease.root, "agent-1", lease.identity))


def test_after_drop_the_next_round_copies_everything(tmp_path):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    before = kept(one_round(snaps, work, "agent-1"))
    failed = take(snaps, work, "agent-2")
    snaps.drop()  # the round failed: its copy stays as evidence

    copy = take(snaps, work, "agent-3").workspace

    assert files(failed.workspace)
    assert linked(before, copy) == set()
    assert linked(files(failed.workspace), copy) == set()
    assert sorted(os.listdir(snaps.rounds)) == ["agent-2", "agent-3"]


def test_a_copy_error_keeps_partial_evidence_and_drops_the_base(tmp_path, monkeypatch):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    with held(one_round(snaps, work, "agent-1")) as before:
        _copy_error_then_full_copy(snaps, work, before, monkeypatch)


def _copy_error_then_full_copy(snaps, work, before, monkeypatch):
    def full(*_args):
        raise OSError(errno.ENOSPC, "quota exhausted")

    monkeypatch.setattr(snapshots, "_copy_content", full)
    (work / "answer.py").write_text("changed\n")
    with pytest.raises(OSError, match="quota exhausted"):
        take(snaps, work, "agent-2")
    assert (snaps.rounds / "agent-2/workspace").is_dir()
    assert (work / "answer.py").read_text() == "changed\n"
    monkeypatch.undo()

    copy = take(snaps, work, "agent-3").workspace

    assert linked(before, copy) == set()


def test_remove_retained_handles_read_only_directories(tmp_path):
    work = work_tree(tmp_path)
    (work / "src").chmod(0o500)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    try:
        one_round(snaps, work, "agent-1")
    finally:
        (work / "src").chmod(0o700)
    assert (snaps.rounds / RETAINED / "src").stat().st_mode & 0o777 == 0o500

    snaps.close()

    assert not (snaps.rounds / RETAINED).exists()
    remove_retained(snaps.rounds)  # missing is fine
    remove_retained(tmp_path / "never-created")
    assert contents(work)["answer.py"] == b"answer = 42\n"


@pytest.mark.parametrize("mode", [0o555, 0o500], ids=["0555", "0500"])
def test_a_read_only_workspace_root_is_retained_and_linked(tmp_path, mode):
    work = work_tree(tmp_path)
    work.chmod(mode)
    try:
        snaps = JudgeSnapshots(tmp_path / "rounds")
        before = kept(one_round(snaps, work, "agent-1"))
        lease = take(snaps, work, "agent-2")
        assert stat.S_IMODE(lease.workspace.stat().st_mode) == mode
        assert linked(before, lease.workspace) == set(files(work))
        snaps.retain(lease)
        snaps.close()
    finally:
        work.chmod(0o700)
    assert os.listdir(snaps.rounds) == []


@pytest.mark.parametrize(
    "mode", [0o000, 0o100, 0o200, 0o300], ids=["0000", "0100", "0200", "0300"]
)
@pytest.mark.parametrize("reuse", [True, False], ids=["reuse", "full"])
def test_an_unreadable_workspace_root_gives_an_empty_copy(
    tmp_path, capsys, mode, reuse
):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    one_round(snaps, work, "agent-1")
    capsys.readouterr()
    work.chmod(mode)
    try:
        kwargs = {"since": snaps.clock()} if reuse else {}
        lease = snaps.acquire(work, "agent-2", **kwargs)
        assert not (snaps.rounds / RETAINED).exists()
        assert stat.S_IMODE(lease.workspace.lstat().st_mode) == mode
        assert capsys.readouterr().out == (
            "[rsi-engine] judge_snapshot_omitted: agent-2 .\n"
        )
        snaps.retain(lease)  # nothing to keep: released
        assert os.listdir(snaps.rounds) == []
    finally:
        work.chmod(0o700)
    lease = take(snaps, work, "agent-3")
    assert contents(lease.workspace) == contents(work)
    snaps.retain(lease)
    snaps.close()
    assert os.listdir(snaps.rounds) == []


def _closed_during_walk(monkeypatch, work: Path, closed: str) -> None:
    """Work makes ``closed`` mode 000 once the walk is inside it, so its copy
    gets that mode."""
    real = snapshots._copy_tree

    def walk(source, target, base, prefix, state, depth=0):
        real(source, target, base, prefix, state, depth)
        if prefix == closed:
            (work / closed).chmod(0)

    monkeypatch.setattr(snapshots, "_copy_tree", walk)


def _copy_closed(snaps, work, closed, monkeypatch, **kwargs):
    real = snapshots._copy_tree
    _closed_during_walk(monkeypatch, work, closed)
    try:
        lease = snaps.acquire(work, "agent-1", **kwargs)
    finally:
        (work / closed).chmod(0o755)
        monkeypatch.setattr(snapshots, "_copy_tree", real)
    assert stat.S_IMODE((lease.workspace / closed).lstat().st_mode) == 0
    return lease


@pytest.mark.parametrize("closed", ["", "src/"])
def test_a_directory_closed_during_the_walk_is_released(tmp_path, monkeypatch, closed):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    lease = _copy_closed(snaps, work, closed, monkeypatch)

    snaps.retain(lease)  # nothing to keep: released

    assert os.listdir(snaps.rounds) == []


@pytest.mark.parametrize("then", ["kept", "deleted", "close", "resume"])
@pytest.mark.parametrize("closed", ["", "src/"])
def test_a_directory_closed_during_the_walk_never_blocks_cleanup(
    tmp_path, monkeypatch, closed, then
):
    work = work_tree(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    lease = _copy_closed(snaps, work, closed, monkeypatch, since=snaps.clock())
    snaps.retain(lease)
    assert (snaps.rounds / RETAINED).is_dir()

    if then == "close":
        snaps.close()
    elif then == "resume":
        remove_retained(snaps.rounds)
    else:
        if then == "deleted":
            shutil.rmtree(work / "src")
        before = kept(snaps.rounds / RETAINED)
        lease = take(snaps, work, "agent-2")
        assert contents(lease.workspace) == contents(work)
        assert "answer.py" in linked(before, lease.workspace)
        snaps.retain(lease)
        snaps.close()

    assert os.listdir(snaps.rounds) == []


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_remove_retained_refuses_what_is_not_its_directory(tmp_path, kind):
    work = work_tree(tmp_path)
    rounds = tmp_path / "rounds"
    rounds.mkdir()
    if kind == "symlink":
        (rounds / RETAINED).symlink_to(work, target_is_directory=True)
    else:
        (rounds / RETAINED).write_text("not a copy")

    with pytest.raises(InfrastructureError, match="not a directory"):
        remove_retained(rounds)
    with pytest.raises(InfrastructureError, match="not a directory"):
        JudgeSnapshots(rounds).acquire(work, "agent-1")

    assert (work / "answer.py").read_text() == "answer = 42\n"


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../other", "x/y", "x\\y", "x\0y", ".retained", ".x"]
)
def test_invalid_identity_is_rejected(tmp_path, name):
    with pytest.raises(InfrastructureError, match="identity"):
        JudgeSnapshots(tmp_path / "rounds").acquire(source(tmp_path), name)


def test_overlapping_root_is_rejected(tmp_path):
    original = source(tmp_path)
    with pytest.raises(InfrastructureError, match="overlap"):
        JudgeSnapshots(original / "rounds").acquire(original, "round")


def test_user_extended_attributes_are_copied(tmp_path):
    work = work_tree(tmp_path)
    try:
        os.setxattr(work / "answer.py", "user.origin", b"agent")
    except OSError as error:
        pytest.skip(f"no user xattrs here: {error}")
    copy = JudgeSnapshots(tmp_path / "rounds").acquire(work, "agent-1").workspace
    assert os.getxattr(copy / "answer.py", "user.origin") == b"agent"


def test_release_removes_only_acquired_copy_and_is_idempotent(tmp_path):
    original = source(tmp_path)
    root = tmp_path / "rounds"
    lease = JudgeSnapshots(root).acquire(original, "round")
    unknown = root / "previous-run"
    unknown.mkdir()
    (unknown / "data").write_text("keep")
    lease.release()
    lease.release()
    assert not (root / "round").exists()
    assert (original / "answer.py").read_text() == "original"
    assert (unknown / "data").read_text() == "keep"


@pytest.mark.parametrize("replacement", ["directory", "symlink", "workspace-symlink"])
def test_replaced_paths_do_not_authorize_deletion(tmp_path, replacement):
    original = source(tmp_path)
    lease = JudgeSnapshots(tmp_path / "rounds").acquire(original, "round")
    directory = lease.workspace.parent
    if replacement == "workspace-symlink":
        shutil.rmtree(lease.workspace)
        lease.workspace.symlink_to(original, target_is_directory=True)
    else:
        directory.rename(lease.root / "saved")
        if replacement == "directory":
            directory.mkdir()
        else:
            directory.symlink_to(original, target_is_directory=True)
    with pytest.raises((OSError, InfrastructureError)):
        lease.release()
    assert (original / "answer.py").read_text() == "original"


@pytest.mark.parametrize("replacement", ["directory", "workspace-symlink"])
def test_replaced_paths_are_not_retained(tmp_path, replacement):
    original = source(tmp_path)
    snaps = JudgeSnapshots(tmp_path / "rounds")
    lease = take(snaps, original, "round")
    if replacement == "workspace-symlink":
        shutil.rmtree(lease.workspace)
        lease.workspace.symlink_to(original, target_is_directory=True)
    else:
        lease.workspace.parent.rename(lease.root / "saved")
        lease.workspace.parent.mkdir()
        (lease.workspace).mkdir()
    with pytest.raises((OSError, InfrastructureError)):
        snaps.retain(lease)
    assert not (snaps.rounds / RETAINED).exists()
    assert (original / "answer.py").read_text() == "original"


def test_cleanup_can_retry_after_partial_removal(tmp_path, monkeypatch):
    lease = JudgeSnapshots(tmp_path / "rounds").acquire(source(tmp_path), "round")
    real = shutil.rmtree

    def fail(*args, **kwargs):
        real(*args, **kwargs)
        raise OSError(errno.EIO, "cleanup interrupted")

    monkeypatch.setattr(shutil, "rmtree", fail)
    with pytest.raises(OSError, match="interrupted"):
        lease.release()
    monkeypatch.setattr(shutil, "rmtree", real)
    lease.release()
    assert not lease.workspace.parent.exists()
