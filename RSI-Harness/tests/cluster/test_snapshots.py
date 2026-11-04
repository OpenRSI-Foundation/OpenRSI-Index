from __future__ import annotations

import errno
import shutil

import pytest

from rsi_harness.cluster.snapshots import DirectorySnapshot
from rsi_harness.errors import InfrastructureError


def source(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    (path / "answer.py").write_text("original")
    return path


def test_release_removes_only_acquired_copy_and_is_idempotent(tmp_path):
    original = source(tmp_path)
    root = tmp_path / "rounds"
    lease = DirectorySnapshot.acquire(original, root, "round")
    unknown = root / "previous-run"
    unknown.mkdir()
    (unknown / "data").write_text("keep")
    (lease.workspace / "answer.py").write_text("Judge changed this")
    lease.release()
    lease.release()
    assert not (root / "round").exists()
    assert (original / "answer.py").read_text() == "original"
    assert (unknown / "data").read_text() == "keep"


def test_symlink_targets_and_readonly_source_are_untouched(tmp_path):
    original = source(tmp_path)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "weights").write_text("keep")
    (original / "link").symlink_to(shared, target_is_directory=True)
    (original / "dangling").symlink_to("missing")
    nested = original / "readonly"
    nested.mkdir()
    (nested / "data").write_text("keep")
    nested.chmod(0o555)
    lease = DirectorySnapshot.acquire(original, tmp_path / "rounds", "round")
    lease.release()
    assert (shared / "weights").read_text() == "keep"
    assert nested.stat().st_mode & 0o777 == 0o555
    assert (nested / "data").read_text() == "keep"


@pytest.mark.parametrize("name", ["", ".", "..", "../other", "x/y", "x\\y", "x\0y"])
def test_invalid_identity_is_rejected(tmp_path, name):
    with pytest.raises(InfrastructureError, match="identity"):
        DirectorySnapshot.acquire(source(tmp_path), tmp_path / "rounds", name)


def test_overlapping_root_is_rejected(tmp_path):
    original = source(tmp_path)
    with pytest.raises(InfrastructureError, match="overlap"):
        DirectorySnapshot.acquire(original, original / "rounds", "round")


@pytest.mark.parametrize("replacement", ["directory", "symlink", "workspace-symlink"])
def test_replaced_paths_do_not_authorize_deletion(tmp_path, replacement):
    original = source(tmp_path)
    lease = DirectorySnapshot.acquire(original, tmp_path / "rounds", "round")
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


def test_copy_quota_error_keeps_partial_evidence(tmp_path, monkeypatch):
    original = source(tmp_path)
    root = tmp_path / "rounds"

    def partial_copy(_source, target, **_options):
        target.mkdir()
        (target / "partial").write_text("evidence")
        raise OSError(errno.EDQUOT, "quota exhausted")

    monkeypatch.setattr(shutil, "copytree", partial_copy)
    with pytest.raises(OSError, match="quota exhausted"):
        DirectorySnapshot.acquire(original, root, "round")
    assert (root / "round/workspace/partial").exists()
    assert (original / "answer.py").exists()


def test_cleanup_can_retry_after_partial_removal(tmp_path, monkeypatch):
    lease = DirectorySnapshot.acquire(source(tmp_path), tmp_path / "rounds", "round")
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
