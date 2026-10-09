"""The host's pull ledger and ``rsi-harness sandbox prune-images`` (S9, Q4).

Pulled images are a host cache the Harness never removes on its own:
removing them when a run ends would need cross-run reference counting (spec
Q4). The ledger names the images a brokered pull first brought to the host,
so an operator can remove exactly those. An image whose ID was on the host
before the sandbox first pulled it, under any name or none, never gets an
entry and is never touched, whoever put it there (an operator, a user, an
acceptance cache); a later pull of an image that has an entry refreshes
``last_used_at`` and adds the reference it pulled when that reference is the
sandbox's (the pull placed it, or moved it off an image the ledger records
under it). A name someone else put on the host is never recorded, so it
keeps the image.

One JSON file under the managed data root, next to the lease store, with
LeaseStore's conventions: owned by whoever runs the Harness (root in
production), a 0700 directory and a 0600 file replaced atomically with
fsync, and an exclusive ``flock`` around every read-modify-write, so runs
pulling concurrently never lose an update. A ledger error never fails a
pull (sandbox_images logs it): the image simply stays unprunable. A pull
waits at most ``LOCK_WAIT`` seconds for the lock, so a stuck holder never
stalls one.

Pruning takes the ledger lock only to read and to forget entries, never
across a Docker call, so pulls are never held behind it.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from docker.errors import APIError, DockerException, NotFound
from pydantic import AwareDatetime, Field, field_validator, model_validator

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.durable import durable_mkdir, fsync_directory
from rsi_harness.runtime.sandbox_contracts import Name, SandboxError, SandboxModel
from rsi_harness.runtime.sandbox_env_contracts import ImageId
from rsi_harness.runtime.sandbox_images import (
    pull_reference,
    reference_text,
    registry_of,
)

LOGGER = logging.getLogger(__name__)
LEDGER_DIR = "sandbox-images"
LEDGER_FILE = "pulled.json"
LEDGER_LOCK = "pulled.lock"
# Bounds: one entry per image the sandbox first brought to the host, a few
# references each. A pull past them is not recorded (the image stays).
MAX_ENTRIES = 4096
MAX_REFERENCES = 32
MAX_REASON = 96
# Seconds a pull waits for the ledger lock before it gives up on recording.
LOCK_WAIT = 2.0

Reference = Annotated[str, Field(min_length=1, max_length=512)]
Registry = Annotated[str, Field(min_length=1, max_length=255)]


def pull_ledger_root(data_root: Path) -> Path:
    """``<data_root>/sandbox-images``, next to ``<data_root>/leases``."""
    return Path(data_root) / LEDGER_DIR


def normalized(reference: object) -> str | None:
    """A reference in the ledger's form (``docker.io/library/busybox:1.37.0``);
    None for one that is no pullable reference (``<none>:<none>``)."""
    try:
        return reference_text(*pull_reference(reference))
    except SandboxError:
        return None


class PulledImage(SandboxModel):
    """An image a brokered pull first brought to the host."""

    image_id: ImageId
    # The sandbox's names of the image; none when it arrived under a name
    # someone else had put on the host (that name then keeps it).
    references: tuple[Reference, ...] = Field(max_length=MAX_REFERENCES)
    registry: Registry
    # Only an image the sandbox brought to the host gets an entry.
    pre_existing: Literal[False] = False
    run_id: Name
    last_run_id: Name
    first_pulled_at: AwareDatetime
    last_used_at: AwareDatetime

    @field_validator("references")
    @classmethod
    def normalized_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value) or any(normalized(v) != v for v in value):
            raise ValueError("ledger references must be distinct and normalized")
        return value

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.last_used_at < self.first_pulled_at:
            raise ValueError("an image cannot be used before it was pulled")
        return self


class PullLedgerFile(SandboxModel):
    schema_version: Literal[1] = 1
    images: tuple[PulledImage, ...] = Field(default=(), max_length=MAX_ENTRIES)

    @model_validator(mode="after")
    def distinct(self) -> Self:
        ids = [image.image_id for image in self.images]
        if len(set(ids)) != len(ids):
            raise ValueError("one ledger entry per image")
        return self


class PullLedger:
    """The durable host-level record of the images the sandbox pulled first."""

    def __init__(
        self,
        root: Path,
        *,
        clock: Callable[[], float] = time.time,
        lock_wait: float = LOCK_WAIT,
    ) -> None:
        self.root = Path(root).resolve()
        self.path = self.root / LEDGER_FILE
        self._clock = clock
        self._lock_wait = lock_wait

    def entries(self) -> tuple[PulledImage, ...]:
        """The current entries; a missing ledger is empty and is not created."""
        return self._read().images

    def record(
        self,
        image_id: str,
        repository: str,
        tag: str,
        *,
        pre_existing: bool,
        before: str | None,
        run_id: str,
    ) -> None:
        """Record one successful brokered pull (sandbox_images.run_pull).

        ``pre_existing``: the image ID was on the host before this pull; with
        no entry it never gets one. ``before``: the image ID the pulled
        reference named before this pull, None when the pull placed it. The
        reference is recorded only when it is the sandbox's: placed by this
        pull, or moved off an image whose entry records it. Raises when the
        lock is not free within ``lock_wait`` seconds.
        """
        reference = reference_text(repository, tag)
        with self._locked(self._lock_wait):
            ledger = self._read()
            now = datetime.fromtimestamp(self._clock(), UTC)
            images = {image.image_id: image for image in ledger.images}
            entry = images.get(image_id)
            ours = before is None or (
                before in images and reference in images[before].references
            )
            if entry is None:
                if pre_existing:
                    return
                if len(images) >= MAX_ENTRIES:
                    raise ValueError("the pull ledger is full")
                entry = PulledImage(
                    image_id=image_id,
                    references=(reference,) if ours else (),
                    registry=registry_of(repository),
                    run_id=run_id,
                    last_run_id=run_id,
                    first_pulled_at=now,
                    last_used_at=now,
                )
            else:
                references = entry.references
                if (
                    ours
                    and reference not in references
                    and len(references) < MAX_REFERENCES
                ):
                    references = (*references, reference)
                entry = PulledImage.model_validate(
                    {
                        **entry.model_dump(),
                        "references": references,
                        "last_run_id": run_id,
                        "last_used_at": max(entry.last_used_at, now),
                    }
                )
            images[image_id] = entry
            self._write(PullLedgerFile(images=tuple(images.values())))

    def forget(self, removed: Mapping[str, datetime]) -> None:
        """Drop the entries of removed images, unless a pull used the image
        again meanwhile (its ``last_used_at`` moved)."""
        if not removed:
            return
        with self._locked():
            ledger = self._read()
            kept = tuple(
                image
                for image in ledger.images
                if removed.get(image.image_id) != image.last_used_at
            )
            if len(kept) != len(ledger.images):
                self._write(PullLedgerFile(images=kept))

    @contextmanager
    def _locked(self, wait: float | None = None) -> Iterator[None]:
        """The exclusive ledger lock; ``wait`` bounds the wait (seconds)."""
        durable_mkdir(self.root)
        descriptor = os.open(self.root / LEDGER_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if wait is None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            else:
                deadline = time.monotonic() + wait
                while True:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise InfrastructureError(
                                "the sandbox pull ledger is busy"
                            ) from None
                        time.sleep(0.02)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read(self) -> PullLedgerFile:
        try:
            return PullLedgerFile.model_validate_json(self.path.read_text())
        except FileNotFoundError:
            return PullLedgerFile()
        except (OSError, ValueError) as error:
            raise InfrastructureError(
                "the sandbox pull ledger is unreadable"
            ) from error

    def _write(self, ledger: PullLedgerFile) -> None:
        descriptor, raw_temporary = tempfile.mkstemp(
            prefix=f".{LEDGER_FILE}.", suffix=".tmp", dir=self.root
        )
        temporary = Path(raw_temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                json.dump(
                    ledger.model_dump(mode="json"),
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)


# -- prune-images ----------------------------------------------------------------

WOULD_REMOVE = "would remove"
WOULD_DROP = "would drop"
REMOVED = "removed"
DROPPED = "dropped"
KEPT = "kept"


@dataclass(frozen=True)
class PruneRow:
    """One ledger entry's verdict; ``bytes`` is the image's size (what
    removing it frees, less layers other images share)."""

    image: PulledImage
    action: str
    reason: str = ""
    bytes: int = 0


def bounded(text: str) -> str:
    return text if len(text) <= MAX_REASON else text[: MAX_REASON - 3] + "..."


def age(seconds: float) -> str:
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit}"
    return f"{max(0, int(seconds))}s"


class ImagePruner:
    """Remove the images the ledger names, and only those, never with force.

    An image is kept while any container on the host uses it (any state) or
    a run's lease in this data root holds it as a pulled handle; both are
    checked again right before it is removed. So is one that would keep a
    reference the ledger did not record (another tag, or a digest of another
    repository): nothing of it is removed. Otherwise its recorded tags and
    the digests of their repositories are removed by name, each only while
    it still names the image, and Docker deletes the image with its last
    reference; only an image left with no reference at all is removed by
    ID. A Docker conflict keeps it. An entry whose image is gone is dropped.
    """

    def __init__(
        self,
        ledger: PullLedger,
        leases: Any,
        api: Any,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._ledger = ledger
        self._leases = leases
        self._api = api
        self._clock = clock

    def run(
        self,
        older_than: float | None,
        *,
        dry_run: bool,
        confirm: Callable[[tuple[PruneRow, ...]], bool],
    ) -> tuple[PruneRow, ...] | None:
        """The plan (``dry_run``), or what was done; None when not confirmed."""
        rows = self.plan(older_than)
        if dry_run:
            return rows
        if any(row.action == WOULD_REMOVE for row in rows) and not confirm(rows):
            return None
        done = tuple(
            self._remove(row.image)
            if row.action == WOULD_REMOVE
            else self._drop(row.image)
            if row.action == WOULD_DROP
            else row
            for row in rows
        )
        try:
            self._ledger.forget(
                {
                    row.image.image_id: row.image.last_used_at
                    for row in done
                    if row.action in (REMOVED, DROPPED)
                }
            )
        except Exception as error:
            # A stale entry is harmless: the next prune drops it.
            LOGGER.warning("sandbox pull ledger not updated: %s", type(error).__name__)
        return done

    def plan(self, older_than: float | None) -> tuple[PruneRow, ...]:
        images = self._ledger.entries()
        if not images:
            return ()
        try:
            containers = self._containers()
        except (DockerException, OSError) as error:
            raise InfrastructureError("cannot list the host's containers") from error
        held = self._held()
        now = self._clock()
        rows = []
        for image in images:
            try:
                attrs = self._inspect(image.image_id)
            except (DockerException, OSError) as error:
                raise InfrastructureError("cannot inspect a pulled image") from error
            idle = now - image.last_used_at.timestamp()
            if attrs is None:
                rows.append(PruneRow(image, WOULD_DROP, "no longer on the host"))
            elif older_than is not None and idle < older_than:
                reason = f"last used {age(idle)} ago"
                rows.append(PruneRow(image, KEPT, reason, _size(attrs)))
            elif refusal := self._refusal(image, containers, held) or _leftover(
                image, attrs
            ):
                rows.append(PruneRow(image, KEPT, refusal, _size(attrs)))
            else:
                rows.append(PruneRow(image, WOULD_REMOVE, "", _size(attrs)))
        return tuple(rows)

    def _remove(self, image: PulledImage) -> PruneRow:
        attrs: Mapping[str, Any] | None = None
        try:
            attrs = self._inspect(image.image_id)
            if attrs is None:
                return PruneRow(image, DROPPED, "no longer on the host")
            action, reason = self._removal(image, attrs)
        except APIError as error:
            if error.status_code == 409:
                action, reason = KEPT, "Docker refused: conflict"
            else:
                LOGGER.warning("sandbox image prune failed: %s", type(error).__name__)
                action, reason = KEPT, "Docker error"
        except (DockerException, OSError) as error:
            LOGGER.warning("sandbox image prune failed: %s", type(error).__name__)
            action, reason = KEPT, "Docker error"
        return PruneRow(image, action, reason, 0 if attrs is None else _size(attrs))

    def _removal(self, image: PulledImage, attrs: Mapping[str, Any]) -> tuple[str, str]:
        """Re-check, then remove the recorded references by name.

        Docker has no conditional untag: each name is inspected right before
        it is removed and skipped once it names another image, which narrows
        (never closes) the window of a concurrent pull moving it.
        """
        refusal = self._refusal(image, self._containers(), self._held()) or (
            _leftover(image, attrs)
        )
        if refusal:
            return KEPT, refusal
        repositories = _repositories(image)
        names = [tag for tag in _tags(attrs) if normalized(tag) in image.references]
        names += [
            digest for digest in _digests(attrs) if _repository(digest) in repositories
        ]
        untagged = []
        for name in names:
            named = self._inspect(name)
            if named is None or named.get("Id") != image.image_id:
                continue
            try:
                self._api.remove_image(name, force=False, noprune=False)
            except NotFound:
                continue
            untagged.append(name)
        attrs = self._inspect(image.image_id)
        if attrs is None:
            return REMOVED, ""
        remaining = [*_tags(attrs), *_digests(attrs)]
        if remaining:
            done = f"untagged {len(untagged)}; " if untagged else ""
            return KEPT, bounded(done + "references remain: " + ", ".join(remaining))
        # No reference at all: only removal by ID deletes it, and Docker
        # refuses that while a container uses it.
        try:
            self._api.remove_image(image.image_id, force=False, noprune=False)
        except NotFound:
            pass
        if self._inspect(image.image_id) is None:
            return REMOVED, ""
        return KEPT, "still on the host"

    def _drop(self, image: PulledImage) -> PruneRow:
        try:
            if self._inspect(image.image_id) is None:
                return PruneRow(image, DROPPED, "no longer on the host")
        except (DockerException, OSError):
            return PruneRow(image, KEPT, "Docker error")
        return PruneRow(image, KEPT, "back on the host")

    def _inspect(self, image_id: str) -> Mapping[str, Any] | None:
        try:
            return self._api.inspect_image(image_id)
        except NotFound:
            return None

    def _containers(self) -> dict[str, str]:
        """Image ID -> one container using it, in any state."""
        users: dict[str, str] = {}
        for container in self._api.containers(all=True):
            image_id = container.get("ImageID")
            if isinstance(image_id, str):
                users.setdefault(image_id, str(container.get("Id") or ""))
        return users

    def _held(self) -> dict[str, str] | str:
        """Image ID -> the run whose lease holds it as a pulled handle; or
        the label of a lease that cannot be read, which holds everything."""
        held: dict[str, str] = {}
        try:
            # A lease store this user cannot list must not look empty.
            if self._leases.root.exists():
                os.listdir(self._leases.root)
        except OSError:
            return "the run leases are unreadable"
        for run_id in self._leases.list_run_ids():
            try:
                lease = self._leases.read(run_id)
            except Exception:
                return bounded(f"run lease {run_id} is unreadable")
            for image in () if lease is None else lease.sandbox_images:
                if image.kind == "pulled" and image.state == "present":
                    held.setdefault(image.image_id, run_id)
        return held

    @staticmethod
    def _refusal(
        image: PulledImage,
        containers: Mapping[str, str],
        held: Mapping[str, str] | str,
    ) -> str:
        if image.image_id in containers:
            return f"used by container {containers[image.image_id][:12]}"
        if isinstance(held, str):
            return held
        if image.image_id in held:
            return bounded(f"held by run {held[image.image_id]}")
        return ""


def _tags(attrs: Mapping[str, Any]) -> list[str]:
    return [tag for tag in attrs.get("RepoTags") or () if tag != "<none>:<none>"]


def _digests(attrs: Mapping[str, Any]) -> list[str]:
    return [item for item in attrs.get("RepoDigests") or () if item != "<none>@<none>"]


def _repository(reference: str) -> str | None:
    """The normalized repository of a tag or digest reference."""
    text = normalized(reference)
    return None if text is None else pull_reference(text)[0]


def _repositories(image: PulledImage) -> set[str]:
    return {pull_reference(reference)[0] for reference in image.references}


def _leftover(image: PulledImage, attrs: Mapping[str, Any]) -> str:
    """Why removing the recorded references would leave the image on the
    host: a tag the ledger did not record, or a digest of a repository it
    recorded no tag of (Docker drops a repository's digests with its last
    tag). Empty when nothing would remain."""
    others = [tag for tag in _tags(attrs) if normalized(tag) not in image.references]
    if others:
        return bounded("other tags remain: " + ", ".join(others))
    repositories = _repositories(image)
    if any(_repository(item) not in repositories for item in _digests(attrs)):
        return "references from other repositories remain"
    return ""


def _size(attrs: Mapping[str, Any]) -> int:
    size = attrs.get("Size")
    return size if type(size) is int and size >= 0 else 0


__all__ = [
    "DROPPED",
    "KEPT",
    "REMOVED",
    "WOULD_DROP",
    "WOULD_REMOVE",
    "ImagePruner",
    "PruneRow",
    "PullLedger",
    "PulledImage",
    "pull_ledger_root",
]
