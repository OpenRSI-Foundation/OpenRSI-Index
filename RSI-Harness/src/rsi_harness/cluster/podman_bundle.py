"""The static Podman bundle the Harness installs for cluster image builds.

A profile's ``builder.podman_bundle`` names a directory on storage the
compute nodes share; the bundle lives in its ``RELEASE`` subdirectory.
``scripts/podman-bundle/build.sh`` builds it; each build is published as
the OpenRSI-Index release ``harness-podman-<RELEASE>`` and pinned here.
"""

from __future__ import annotations

import fcntl
import hashlib
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from rsi_harness.errors import SetupError

RELEASE = "5.8.9-1"
ARCHIVE = f"podman-static-{RELEASE}-linux-amd64.tar.gz"
URL = (
    "https://github.com/OpenRSI-Foundation/OpenRSI-Index/releases/download/"
    f"harness-podman-{RELEASE}/{ARCHIVE}"
)
SHA256 = "166cfd407dda254a02792977f1d44b58b37f633b83c4694f2149065adaee13bc"
# Written last: the archive digest this tree was extracted from.
_MARKER = ".rsi-archive-sha256"

PODMAN = Path("bin/podman")
CONMON = Path("lib/podman/conmon")
RUNTIME = Path("bin/runc")
CONTAINERS_CONF = Path("etc/containers/containers.conf")
STORAGE_CONF = Path("etc/containers/storage.conf")
POLICY = Path("etc/containers/policy.json")


def bundle_dir(root: Path) -> Path:
    return Path(root) / RELEASE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(tree: Path) -> bool:
    """Every file SHA256SUMS lists is present with its digest."""
    try:
        if (tree / _MARKER).read_text().strip() != SHA256:
            return False
        lines = (tree / "SHA256SUMS").read_text().splitlines()
        for line in lines:
            digest, name = line.split(maxsplit=1)
            if _sha256(tree / name) != digest:
                return False
        return bool(lines)
    except (OSError, ValueError):
        return False


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, 8 * 1024 * 1024)


def ensure_installed(root: Path) -> bool:
    """Install the bundle under ``root`` unless it is there intact.

    Returns whether it installed. Repairs serialize on a persistent per-release
    lock; the shared filesystem must support cross-node ``flock`` locks. Never
    unlink the lock file: waiters must continue to lock the same inode.
    """
    target = bundle_dir(root)
    if _verified(target):
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with (target.parent / f".{RELEASE}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Another process may have installed or repaired it while we waited.
            if _verified(target):
                return False
            _install(target)
    except OSError as error:
        raise SetupError(
            f"cannot install the Podman bundle at {target}: {error}"
        ) from error
    return True


def _install(target: Path) -> None:
    """Download, verify and publish while holding the release lock."""
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".install-") as tmp:
        staging = Path(tmp)
        archive = staging / ARCHIVE
        try:
            _download(URL, archive)
        except OSError as error:
            raise SetupError(
                f"cannot download the Podman bundle from {URL}: {error}"
            ) from error
        if _sha256(archive) != SHA256:
            raise SetupError(f"the Podman bundle from {URL} fails its SHA256")
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging / "tree", filter="data")
        tree = staging / "tree" / f"podman-static-{RELEASE}"
        (tree / _MARKER).write_text(SHA256 + "\n")
        if not _verified(tree):
            raise SetupError("the Podman bundle does not match its SHA256SUMS")
        stale = staging / "stale"
        if target.exists():
            target.rename(stale)
        try:
            tree.rename(target)
        except OSError:
            # Preserve the previous tree if publishing the verified copy fails.
            if stale.exists():
                stale.rename(target)
            raise
