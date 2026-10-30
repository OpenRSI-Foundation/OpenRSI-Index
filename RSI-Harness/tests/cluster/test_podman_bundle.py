from __future__ import annotations

import errno
import hashlib
import io
import multiprocessing
import tarfile
from pathlib import Path

import pytest

from rsi_harness.cluster import podman_bundle
from rsi_harness.errors import SetupError

FILES = {"bin/podman": b"podman", "lib/podman/conmon": b"conmon", "bin/runc": b"runc"}


def _archive() -> bytes:
    sums = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(FILES.items())
    )
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in {**FILES, "SHA256SUMS": sums.encode()}.items():
            member = tarfile.TarInfo(f"podman-static-{podman_bundle.RELEASE}/{name}")
            member.size, member.mode = len(data), 0o755
            archive.addfile(member, io.BytesIO(data))
    return stream.getvalue()


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    data = _archive()
    monkeypatch.setattr(podman_bundle, "SHA256", hashlib.sha256(data).hexdigest())
    calls: list[str] = []

    def download(url: str, target: Path) -> None:
        calls.append(url)
        target.write_bytes(data)

    monkeypatch.setattr(podman_bundle, "_download", download)
    return calls


def test_installs_once_and_reinstalls_a_damaged_tree(tmp_path, downloads) -> None:
    root = tmp_path / "podman"
    tree = podman_bundle.bundle_dir(root)

    assert podman_bundle.ensure_installed(root) is True
    assert (tree / "bin/podman").read_bytes() == b"podman"
    assert (tree / "bin/podman").stat().st_mode & 0o111
    assert podman_bundle.ensure_installed(root) is False
    assert downloads == [podman_bundle.URL]

    (tree / "bin/runc").write_bytes(b"other")
    assert podman_bundle.ensure_installed(root) is True
    assert (tree / "bin/runc").read_bytes() == b"runc"
    assert {path.name for path in root.iterdir()} == {
        podman_bundle.RELEASE,
        f".{podman_bundle.RELEASE}.lock",
    }


def test_a_wrong_or_failed_download_installs_nothing(
    tmp_path, downloads, monkeypatch
) -> None:
    root = tmp_path / "podman"
    monkeypatch.setattr(podman_bundle, "SHA256", "0" * 64)
    with pytest.raises(SetupError, match="fails its SHA256"):
        podman_bundle.ensure_installed(root)

    def offline(_url: str, _target: Path) -> None:
        raise OSError("network is unreachable")

    monkeypatch.setattr(podman_bundle, "_download", offline)
    with pytest.raises(SetupError, match="cannot download"):
        podman_bundle.ensure_installed(root)
    assert [path.name for path in root.iterdir()] == [f".{podman_bundle.RELEASE}.lock"]


def _install_in_process(root, data, results, downloads, entered, release, locking):
    # Each child is a fresh interpreter, using a fixture archive rather than the
    # network. Events force the second install to contend with the first one.
    podman_bundle.SHA256 = hashlib.sha256(data).hexdigest()
    original_flock = podman_bundle.fcntl.flock

    def flock(file, operation):
        if locking is not None:
            locking.set()
        return original_flock(file, operation)

    def download(_url, target):
        downloads.put(True)
        if entered is not None:
            entered.set()
            if not release.wait(15):
                raise RuntimeError("test did not release the first download")
        target.write_bytes(data)

    podman_bundle.fcntl.flock = flock
    podman_bundle._download = download
    try:
        results.put(("ok", podman_bundle.ensure_installed(Path(root))))
    except Exception as error:
        results.put(("error", repr(error)))


@pytest.mark.parametrize("damaged", [False, True])
def test_concurrent_install_or_repair_installs_only_once(
    tmp_path, monkeypatch, damaged
):
    root = tmp_path / "podman"
    tree = podman_bundle.bundle_dir(root)
    data = _archive()
    monkeypatch.setattr(podman_bundle, "SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(
        podman_bundle, "_download", lambda url, path: path.write_bytes(data)
    )
    if damaged:
        podman_bundle.ensure_installed(root)
        (tree / "bin/runc").write_bytes(b"damaged")

    context = multiprocessing.get_context("spawn")
    results, downloads = context.Queue(), context.Queue()
    entered, release, locking = context.Event(), context.Event(), context.Event()
    first = context.Process(
        target=_install_in_process,
        args=(str(root), data, results, downloads, entered, release, None),
    )
    second = context.Process(
        target=_install_in_process,
        args=(str(root), data, results, downloads, None, None, locking),
    )
    started = []
    try:
        first.start()
        started.append(first)
        assert entered.wait(15), "first install did not reach the download"
        second.start()
        started.append(second)
        assert locking.wait(15), "second install did not attempt the shared lock"
        release.set()
        outcome = [results.get(timeout=15), results.get(timeout=15)]
        for process in started:
            process.join(15)
            assert process.exitcode == 0
        assert sorted(outcome) == [("ok", False), ("ok", True)]
        assert downloads.get(timeout=1) is True
        assert downloads.empty(), "a waiting installer downloaded a second copy"
        assert podman_bundle._verified(tree)
    finally:
        release.set()
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(5)
        results.close()
        downloads.close()


def test_lock_failure_leaves_the_existing_tree_untouched(
    tmp_path, downloads, monkeypatch
):
    root = tmp_path / "podman"
    tree = podman_bundle.bundle_dir(root)
    podman_bundle.ensure_installed(root)
    (tree / "bin/runc").write_bytes(b"damaged")
    downloads.clear()

    def no_lock(_file, _operation):
        raise OSError(errno.ENOLCK, "filesystem locking unavailable")

    monkeypatch.setattr(podman_bundle.fcntl, "flock", no_lock)
    with pytest.raises(SetupError, match="filesystem locking unavailable"):
        podman_bundle.ensure_installed(root)
    assert downloads == []
    assert (tree / "bin/runc").read_bytes() == b"damaged"
    assert not list(root.glob(".install-*"))


def test_failed_publication_preserves_tree_and_releases_lock(
    tmp_path, downloads, monkeypatch
):
    root = tmp_path / "podman"
    tree = podman_bundle.bundle_dir(root)
    podman_bundle.ensure_installed(root)
    (tree / "bin/runc").write_bytes(b"damaged")
    rename = Path.rename

    def fail_publish(path, target):
        if path.parent.name == "tree":
            raise OSError(errno.EIO, "publication failed")
        return rename(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail_publish)
        with pytest.raises(SetupError, match="publication failed"):
            podman_bundle.ensure_installed(root)
    assert (tree / "bin/runc").read_bytes() == b"damaged"
    assert not list(root.glob(".install-*"))
    assert podman_bundle.ensure_installed(root) is True
    assert podman_bundle._verified(tree)


def test_failed_repair_download_preserves_existing_tree(
    tmp_path, downloads, monkeypatch
):
    root = tmp_path / "podman"
    tree = podman_bundle.bundle_dir(root)
    podman_bundle.ensure_installed(root)
    (tree / "bin/runc").write_bytes(b"damaged")

    with monkeypatch.context() as patch:
        patch.setattr(podman_bundle, "SHA256", "0" * 64)
        with pytest.raises(SetupError, match="fails its SHA256"):
            podman_bundle.ensure_installed(root)
    assert (tree / "bin/runc").read_bytes() == b"damaged"
    assert not list(root.glob(".install-*"))
    assert podman_bundle.ensure_installed(root) is True
