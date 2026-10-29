"""Docker Desktop boundaries for a controller running natively on macOS."""

from __future__ import annotations

import errno
import ipaddress
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from rsi_harness.errors import (
    InfrastructureError,
    RecoveryRequiredError,
    SetupError,
    VerifierError,
)
from rsi_harness.models import AgentRunResult, RootfsSnapshotMode
from rsi_harness.runtime.durable import durable_mkdir, fsync_directory

PRIVATE_LOG_LABEL = "org.openrsi.private-verifier-logs"
LOG_TARGET = "/logs/verifier"
HELPER_IMAGE = (
    "debian@sha256:c2880112cc5c61e1200c26f106e4123627b49726375eb5846313da9cca117337"
)
MAX_LOG_BYTES = 1 << 30
MAX_LOG_MEMBERS = 100_000
# Separate transport bound includes tar headers, padding and extended metadata.
MAX_ARCHIVE_BYTES = MAX_LOG_BYTES + MAX_LOG_MEMBERS * 10_240


class PrivateLogContentError(VerifierError):
    """The completed export contains deterministically unacceptable artifacts."""

    exec_result: AgentRunResult | None = None


def is_desktop(client: Any) -> bool:
    if sys.platform != "darwin":
        return False
    return client.info().get("OperatingSystem", "").lower() == "docker desktop"


def validate_local_daemon(client: Any) -> None:
    if sys.platform == "darwin" and not is_desktop(client):
        raise SetupError(
            "The native macOS controller requires the Docker Desktop Linux daemon; "
            "other local or remote daemons are not supported by this backend."
        )
    if sys.platform == "darwin" and client.api.base_url != "http+docker://localhost":
        # This is the SDK's Unix-socket transport identity, not a hostname guess.
        # TCP/SSH can reach another machine even when its daemon is Desktop.
        raise SetupError(
            "The native macOS backend requires the local Docker Desktop Unix "
            "socket; TCP and SSH Docker endpoints are not supported."
        )


def validate_desktop_task(definition: Any, judge_llm: Any) -> None:
    """Refuse paths that still require Linux host facilities before a run."""
    if definition.sandbox is not None:
        raise SetupError(
            "The native macOS Docker Desktop backend does not yet support "
            "sandbox tasks; use a validated Linux execution host."
        )
    if definition.gpu_requirement.count or definition.verifier.gpu_count:
        raise SetupError("The native macOS Docker Desktop backend is CPU-only")
    if judge_llm is not None and judge_llm.operator_login is not None:
        raise SetupError(
            "Standalone --judge-llm-auth codex-login renewal needs a Linux host. "
            "On native macOS use a supported Judge API credential or a Codex "
            "Work Agent that lends its login, after task-specific validation."
        )


def validate_desktop_images(images: Any) -> None:
    if (
        images.rootfs_snapshot_mode != RootfsSnapshotMode.SPLIT_WORKDIR
        or str(images.workdir) == "/"
    ):
        raise SetupError(
            "The native macOS Docker Desktop backend requires a split WORKDIR "
            "below /; full-rootfs and other workspace modes are not supported."
        )


class DesktopHost:
    """Run only host firewall commands in the daemon's Linux namespaces.

    The helper has no Docker socket, host filesystem bind or credentials.
    Its nsenter capabilities apply only to this trusted short-lived helper,
    never to Work or Judge. LinuxKit's own iptables and lock are used.
    """

    def __init__(self, client: Any) -> None:
        self.client = client
        self.image: str | None = None

    def _image(self) -> str:
        from docker.errors import ImageNotFound

        if self.image is None:
            try:
                image = self.client.images.get(HELPER_IMAGE)
            except ImageNotFound:
                image = self.client.images.pull(HELPER_IMAGE)
            self.image = image.id
        return self.image

    def _execute(
        self, command: list[str], *, host: bool
    ) -> subprocess.CompletedProcess[str]:
        options: dict[str, Any] = {
            "user": "0:0",
            "read_only": True,
            "network_mode": "none" if host else "bridge",
            "cap_drop": ["ALL"],
            "mem_limit": "64m",
            "nano_cpus": 500_000_000,
            "labels": {"org.openrsi.desktop-helper": "true"},
        }
        if host:
            options.update(
                pid_mode="host",
                cap_add=["SYS_ADMIN", "NET_ADMIN", "SYS_PTRACE", "SYS_CHROOT"],
                security_opt=["apparmor=unconfined"],
            )
        from rsi_harness.runtime.desktop_helpers import (
            HELPER_LABEL,
            DesktopHelperIntents,
        )

        image = self._image()
        with DesktopHelperIntents(self.client).operation(image) as intent:
            options["name"] = intent["name"]
            options["labels"][HELPER_LABEL] = intent["nonce"]
            helper = self.client.containers.create(image, command, **options)
            helper.start()
            result = helper.wait(timeout=30)
            return subprocess.CompletedProcess(
                command,
                int(result["StatusCode"]),
                helper.logs(stdout=True, stderr=False).decode("utf-8", "replace"),
                helper.logs(stdout=False, stderr=True).decode("utf-8", "replace"),
            )

    def firewall(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:2] != ["iptables", "--wait"]:
            raise SetupError(
                "Desktop host helper accepts only bounded iptables commands"
            )
        return self._execute(
            [
                "/usr/bin/nsenter",
                "-t",
                "1",
                "-m",
                "-n",
                "--",
                "/sbin/iptables",
                "--wait",
                "10",
                *command[2:],
            ],
            host=True,
        )

    def gateway(self) -> str:
        result = self._execute(
            ["getent", "ahostsv4", "host.docker.internal"], host=False
        )
        if result.returncode != 0 or not result.stdout.split():
            raise SetupError(
                "Docker Desktop cannot resolve its macOS controller address"
            )
        return str(ipaddress.IPv4Address(result.stdout.split()[0]))


def attest_private_log_volume(client: Any, container_id: str) -> Any:
    """Require the exact writable Linux volume before trusted initialization/export."""
    if not is_desktop(client):
        raise InfrastructureError("private Judge volumes require the Desktop backend")
    container = client.containers.get(container_id)
    container.reload()
    labels = container.attrs.get("Config", {}).get("Labels") or {}
    if (
        labels.get(PRIVATE_LOG_LABEL) != "volume"
        or labels.get("rsi-harness.role") != "judge"
    ):
        raise InfrastructureError("private Judge log storage authority is missing")
    mounts = [
        m
        for m in container.attrs.get("Mounts", [])
        if m.get("Destination") == LOG_TARGET
    ]
    if (
        len(mounts) != 1
        or mounts[0].get("Type") != "volume"
        or mounts[0].get("RW") is not True
    ):
        raise InfrastructureError(
            "private Judge logs are not on a writable Linux volume"
        )
    return container


def has_private_log_volume(client: Any, container: Any) -> bool:
    """A task image's label alone never selects the Desktop recovery path."""
    if not is_desktop(client):
        return False
    labels = container.attrs.get("Config", {}).get("Labels") or {}
    if (
        labels.get(PRIVATE_LOG_LABEL) != "volume"
        or labels.get("rsi-harness.role") != "judge"
    ):
        return False
    attest_private_log_volume(client, container.id)
    return True


def _record_content_rejection(target: Path, error: Exception) -> None:
    # A sibling of the archive destination cannot be replaced by an archive entry.
    diagnostic = target.with_name(target.name + ".private-log-export-error.json")
    # Keep earlier evidence, but never leave an old reward at the normal reader
    # path after rejecting the current export (including during recovery).
    previous = None
    if any(target.iterdir()):
        quarantine = Path(
            tempfile.mkdtemp(prefix=".rejected-log-export-", dir=target.parent)
        )
        previous = quarantine / "previous-export"
        os.replace(target, previous)
        target.mkdir(mode=0o700)
        fsync_directory(quarantine)
        fsync_directory(target.parent)
    descriptor, name = tempfile.mkstemp(prefix=".log-rejection-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "status": "verifier_error",
                    "score": None,
                    "error": str(error),
                    "previous_export": str(previous) if previous is not None else None,
                },
                output,
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, diagnostic)
        fsync_directory(target.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def _reject_content(target: Path, error: Exception) -> NoReturn:
    rejection = PrivateLogContentError(f"private Judge log content rejected: {error}")
    try:
        _record_content_rejection(target, rejection)
    except OSError as diagnostic_error:
        raise RecoveryRequiredError(
            "recovery_required: cannot record private log rejection: "
            f"{diagnostic_error}"
        ) from error
    raise rejection from error


def export_private_logs(client: Any, container_id: str, target: Path) -> None:
    """Copy bounded regular files from a Judge-owned Linux volume to the Mac.

    Archive links/devices/absolute paths are refused. Extraction happens in a
    fresh private directory before replacing any operator-visible artifacts.
    Content rejection fails the round; transport failures retain recovery evidence.
    """
    # Keep CLI/profile-only imports independent of the Docker execution backend.
    from rsi_harness.runtime.sandbox_archive import read_tar

    container = attest_private_log_volume(client, container_id)
    target = Path(target)
    if target.is_symlink():
        raise InfrastructureError("Judge log destination is a symlink")
    durable_mkdir(target, mode=0o700)
    extracting = False
    try:
        with tempfile.TemporaryDirectory(
            prefix=".log-export-", dir=target.parent
        ) as temporary:
            staging = Path(temporary)
            archive_path = staging / "archive.tar"
            stream, _ = container.get_archive(LOG_TARGET)
            total = 0
            try:
                with archive_path.open("xb") as archive_file:
                    for chunk in stream:
                        total += len(chunk)
                        if total > MAX_ARCHIVE_BYTES:
                            raise PrivateLogContentError(
                                "Judge log archive transport bound exceeded"
                            )
                        archive_file.write(chunk)
            finally:
                close = getattr(stream, "close", None)
                if close:
                    close()
            contents = staging / "contents"
            contents.mkdir(mode=0o700)
            seen: set[str] = set()
            canonical_names: dict[str, str] = {}
            kinds: dict[str, bool] = {}
            file_bytes = 0
            extracting = True
            with archive_path.open("rb") as source, read_tar(source) as archive:
                for index, member in enumerate(archive):
                    parts = PurePosixPath(member.name).parts
                    if (
                        index >= MAX_LOG_MEMBERS
                        or not parts
                        or parts[0] != "verifier"
                        or ".." in parts
                        or "\x00" in member.name
                        or str(PurePosixPath(member.name)) in seen
                        or not (member.isdir() or member.isfile())
                        or (len(parts) == 1 and not member.isdir())
                    ):
                        raise PrivateLogContentError(
                            "unsafe or oversized Judge log archive"
                        )
                    seen.add(str(PurePosixPath(member.name)))
                    # Check implicit parents too: A/x and a/y alias on APFS.
                    for end in range(1, len(parts) + 1):
                        name = "/".join(parts[:end])
                        canonical = unicodedata.normalize("NFC", name).casefold()
                        previous = canonical_names.setdefault(canonical, name)
                        if previous != name:
                            raise PrivateLogContentError(
                                "case or Unicode-colliding Judge log paths"
                            )
                        directory = end < len(parts) or member.isdir()
                        if name in kinds and kinds[name] != directory:
                            raise PrivateLogContentError(
                                "Judge log path is both file and directory"
                            )
                        kinds[name] = directory
                    file_bytes += member.size if member.isfile() else 0
                    if file_bytes > MAX_LOG_BYTES or member.size < 0:
                        raise PrivateLogContentError(
                            "Judge log file contents exceed 1 GiB"
                        )
                    destination = contents.joinpath(*parts[1:])
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True, mode=0o700)
                    else:
                        destination.parent.mkdir(
                            parents=True, exist_ok=True, mode=0o700
                        )
                        reader = archive.extractfile(member)
                        if reader is None:
                            raise PrivateLogContentError(
                                "Judge log file has no archive content"
                            )
                        with reader, destination.open("xb") as output:
                            shutil.copyfileobj(reader, output)
                            os.fchmod(output.fileno(), 0o600)
                            output.flush()
                            os.fsync(output.fileno())
            for directory, _, _ in os.walk(contents, topdown=False):
                fsync_directory(Path(directory))
            extracting = False
            for index, entry in enumerate(contents.iterdir()):
                final = target / entry.name
                if final.is_symlink():
                    raise InfrastructureError(
                        "Judge log destination contains a symlink"
                    )
                # Archive basenames can already fill the filesystem's name limit.
                # Use our bounded entry index, never extend an artifact basename.
                backup = staging / f"previous-{index}"
                existed = final.exists()
                if existed:
                    os.replace(final, backup)
                try:
                    os.replace(entry, final)
                except BaseException:
                    if existed:
                        os.replace(backup, final)
                    raise
            fsync_directory(target)
            fsync_directory(target.parent)

    except (PrivateLogContentError, tarfile.TarError, UnicodeError) as error:
        _reject_content(target, error)
    except Exception as error:
        if (
            extracting
            and isinstance(error, OSError)
            and error.errno == errno.ENAMETOOLONG
        ):
            _reject_content(target, error)
        raise RecoveryRequiredError(
            f"recovery_required: private Judge log export failed: {error}"
        ) from error
