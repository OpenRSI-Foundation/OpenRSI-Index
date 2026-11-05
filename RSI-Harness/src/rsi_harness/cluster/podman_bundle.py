"""The static Podman bundle the Harness installs for cluster image builds.

A profile's ``builder.podman_bundle`` names a directory on storage the
compute nodes share; each architecture's bundle lives in its own
subdirectory (``bundle_dir``). ``scripts/podman-bundle/build.sh`` builds it;
each build is published as the OpenRSI-Index release
``harness-podman-<RELEASE>`` and pinned here.
"""

from __future__ import annotations

import fcntl
import hashlib
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from rsi_harness.errors import SetupError

RELEASE = "5.8.9-1"
# Linux `uname -m` of each architecture with a bundle.
MACHINES = {"amd64": "x86_64", "arm64": "aarch64"}
# Archive digests.
SHA256: dict[str, str] = {
    "amd64": "166cfd407dda254a02792977f1d44b58b37f633b83c4694f2149065adaee13bc",
    "arm64": "d7f7ad6cded3470f34802a45e224b85200fba4cae3c6535cbe82feead836e24c",
}
# Written last: the archive digest this tree was extracted from.
_MARKER = ".rsi-archive-sha256"

PODMAN = Path("bin/podman")
CONMON = Path("lib/podman/conmon")
RUNTIME = Path("bin/runc")
CONTAINERS_CONF = Path("etc/containers/containers.conf")
STORAGE_CONF = Path("etc/containers/storage.conf")
POLICY = Path("etc/containers/policy.json")


def _known_architecture() -> str | None:
    machine = platform.machine().lower()
    for arch, name in MACHINES.items():
        if machine in (arch, name):
            return arch
    return None


def architecture() -> str:
    """This machine's bundle architecture. The build nodes share it: the
    Engine runs on the compute nodes with this machine's Python."""
    arch = _known_architecture()
    if arch is None:
        raise SetupError(
            f"no static Podman bundle for this machine ({platform.machine()}); "
            "remove builder.podman_bundle and set builder binary to a Podman on "
            "the build nodes"
        )
    return arch


def archive(arch: str) -> str:
    return f"podman-static-{RELEASE}-linux-{arch}.tar.gz"


def url(arch: str) -> str:
    return (
        "https://github.com/OpenRSI-Foundation/OpenRSI-Index/releases/download/"
        f"harness-podman-{RELEASE}/{archive(arch)}"
    )


def bundle_dir(root: Path, arch: str | None = None) -> Path:
    """amd64 keeps the path it had before other architectures, so existing
    installs stay valid; the others install under ``<arch>/``. A machine
    without a bundle gets a path too, so its profile loads; installing and
    building refuse."""
    arch = arch or _known_architecture() or platform.machine().lower()
    if arch == "amd64":
        return Path(root) / RELEASE
    return Path(root) / arch / RELEASE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(tree: Path, digest: str) -> bool:
    """Extracted from the archive with ``digest``, and every file
    SHA256SUMS lists is present with its digest."""
    try:
        if (tree / _MARKER).read_text().strip() != digest:
            return False
        lines = (tree / "SHA256SUMS").read_text().splitlines()
        for line in lines:
            expected, name = line.split(maxsplit=1)
            if _sha256(tree / name) != expected:
                return False
        return bool(lines)
    except (OSError, ValueError):
        return False


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out, 8 * 1024 * 1024)


def ensure_installed(root: Path) -> bool:
    """Install this machine's bundle under ``root`` unless it is there intact.

    Returns whether it installed. Repairs serialize on a persistent per-release
    lock; the shared filesystem must support cross-node ``flock`` locks. Never
    unlink the lock file: waiters must continue to lock the same inode.
    """
    arch = architecture()
    digest = SHA256[arch]
    target = bundle_dir(root, arch)
    if _verified(target, digest):
        return False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with (target.parent / f".{RELEASE}.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Another process may have installed or repaired it while we waited.
            if _verified(target, digest):
                return False
            _install(target, arch, digest)
    except OSError as error:
        raise SetupError(
            f"cannot install the Podman bundle at {target}: {error}"
        ) from error
    return True


def _install(target: Path, arch: str, digest: str) -> None:
    """Download, verify and publish while holding the release lock."""
    source = url(arch)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".install-") as tmp:
        staging = Path(tmp)
        downloaded = staging / archive(arch)
        try:
            _download(source, downloaded)
        except OSError as error:
            raise SetupError(
                f"cannot download the Podman bundle from {source}: {error}"
            ) from error
        if _sha256(downloaded) != digest:
            raise SetupError(f"the Podman bundle from {source} fails its SHA256")
        with tarfile.open(downloaded) as bundle:
            bundle.extractall(staging / "tree", filter="data")
        tree = staging / "tree" / f"podman-static-{RELEASE}"
        (tree / _MARKER).write_text(digest + "\n")
        if not _verified(tree, digest):
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
