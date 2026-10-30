"""The run's remote sandbox ledger: how a client whose key cannot list
finds the run's sandboxes again.

A Daytona key with ``write:sandboxes`` only can create a sandbox, stop it
and read it by its id, but its listing comes back empty and a delete is
refused (403). Recovery finds a run's sandboxes by listing their
``rsi_run_id``: with such a key it would find none and report the run
clean while they live on. A client that keeps a ledger (its ``ledger``
attribute, set by ``bind_ledger``) therefore writes each sandbox it creates
here: a ``create`` record with the sandbox's unique name and metadata just
before the request, a ``created`` record with its id once the request
answers, and an ``end`` record once it is proven gone. Its ``list`` merges
the provider's listing with the ledger's open entries, each read again, so
``kill_run_sandboxes`` (``recover``, ``cleanup``, ``--cluster`` too) and an
env's kill find them either way.

One JSON lines file per run, ``<run_id>.sandboxes.jsonl`` beside the run's
lease in the lease store (``<data_root>/leases``, or a cluster run's
``leases`` directory under the profile's ``run_root``): a resumed attempt
and the submit host's recovery read what the Engine wrote. Each record is
appended under an exclusive ``flock`` and fsynced, so a ``create`` record
is durable before its request leaves; a record that cannot be written
first refuses the create. The rest never fail an operation: a lost
``created`` leaves the name to find the sandbox by, a lost ``end`` costs a
read. A record names where a sandbox is, never a credential. An end is
final: a later answer or note of a name or id that ended never reopens it,
only a new ``create`` record (a new request) does. A torn last line (a
crash mid-write) is skipped, and the next append first ends it with a
newline, so the record after it is whole. Appends of several processes
are serialized by the lock (on one host; across hosts as far as the shared
file system honors ``flock``).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.durable import durable_mkdir, fsync_directory

LOGGER = logging.getLogger(__name__)
LEDGER_SUFFIX = ".sandboxes.jsonl"
# Longer lines are not ours: skipped as a torn or foreign line.
MAX_LINE = 16384


def sandbox_ledger_path(lease_root: Path, run_id: str) -> Path:
    """``<lease_root>/<run_id>.sandboxes.jsonl``: beside ``<run_id>.json``
    (LeaseStore lists ``*.json`` only)."""
    return Path(lease_root) / f"{run_id}{LEDGER_SUFFIX}"


@dataclass
class LedgerEntry:
    """One sandbox the ledger has no end for: its unique name, the metadata
    it was created with, when the request left (wall clock), its id once
    known, and the last outcome that left it in place (``leftover``)."""

    name: str
    metadata: dict[str, str]
    created_at: float
    sandbox_id: str | None = None
    note: str | None = None


@dataclass
class _Fold:
    open: dict[str, LedgerEntry] = field(default_factory=dict)
    ended_names: set[str] = field(default_factory=set)
    ended_ids: set[str] = field(default_factory=set)

    def apply(self, record: Mapping[str, object]) -> None:
        event = record.get("event")
        name = record.get("name")
        sandbox_id = record.get("id")
        name = name if isinstance(name, str) else None
        sandbox_id = sandbox_id if isinstance(sandbox_id, str) else None
        if event == "create" and name is not None:
            # A new request, even of a name that ended (a create refused
            # for capacity, asked again).
            if name in self.open:
                return
            self.ended_names.discard(name)
            metadata = record.get("metadata")
            at = record.get("at")
            self.open[name] = LedgerEntry(
                name,
                {
                    str(key): str(value)
                    for key, value in (
                        metadata.items() if isinstance(metadata, Mapping) else ()
                    )
                },
                float(at) if isinstance(at, (int, float)) else 0.0,
            )
            return
        entry = self._find(name, sandbox_id)
        if event == "created":
            if entry is not None and sandbox_id is not None:
                if sandbox_id in self.ended_ids:
                    self._end(entry, sandbox_id)
                else:
                    entry.sandbox_id = sandbox_id
        elif event == "note":
            if entry is not None:
                note = record.get("note")
                entry.note = note if isinstance(note, str) else None
        elif event == "end":
            if sandbox_id is not None:
                self.ended_ids.add(sandbox_id)
            if name is not None:
                self.ended_names.add(name)
            if entry is not None:
                self._end(entry, sandbox_id)

    def _find(self, name: str | None, sandbox_id: str | None) -> LedgerEntry | None:
        if name is not None and name in self.open:
            return self.open[name]
        if sandbox_id is not None:
            for entry in self.open.values():
                if entry.sandbox_id == sandbox_id:
                    return entry
        return None

    def _end(self, entry: LedgerEntry, sandbox_id: str | None) -> None:
        del self.open[entry.name]
        self.ended_names.add(entry.name)
        for known in (entry.sandbox_id, sandbox_id):
            if known is not None:
                self.ended_ids.add(known)


class SandboxLedger:
    """The append-only record of one run's sandboxes (see the module).

    The file is read once, on first use, and kept folded in memory: one
    process writes a run's ledger at a time (the Engine, or recovery once
    the Engine let go of the lease)."""

    def __init__(
        self, path: Path, run_id: str, *, clock: Callable[[], float] = time.time
    ) -> None:
        # Absolute, as durable_mkdir needs (a relative root is the cwd's).
        self.path = Path(path).absolute()
        self.run_id = run_id
        # The wall clock of every record (and of an entry's ``created_at``).
        self.clock = clock
        self._lock = threading.Lock()
        self._fold: _Fold | None = None

    # -- records -------------------------------------------------------------

    def creating(self, name: str, metadata: Mapping[str, str]) -> None:
        """Before a create's request: durable, or the create is refused."""
        record = {
            "event": "create",
            "name": name,
            "metadata": dict(metadata),
            "at": self.clock(),
        }
        with self._lock:
            fold = self._load()
            try:
                self._append(record)
            except OSError as error:
                raise InfrastructureError(
                    f"sandbox ledger {self.path} is unwritable "
                    f"({type(error).__name__}): no sandbox was created"
                ) from None
            fold.apply(record)

    def created(self, name: str, sandbox_id: str) -> None:
        """Once the create answered with the sandbox's id."""
        self._record({"event": "created", "name": name, "id": sandbox_id})

    def noted(self, sandbox_id: str, note: str) -> None:
        """An outcome that left the sandbox in place (it stays open)."""
        self._record({"event": "note", "id": sandbox_id, "note": note})

    def ended(
        self, outcome: str, *, sandbox_id: str | None = None, name: str | None = None
    ) -> None:
        """The sandbox is proven gone (or its create never landed); nothing
        is written for one the ledger holds no open entry of."""
        with self._lock:
            try:
                if self._load()._find(name, sandbox_id) is None:
                    return
            except InfrastructureError as error:
                LOGGER.warning("%s: end record not written", error)
                return
        record = {"event": "end", "outcome": outcome}
        if name is not None:
            record["name"] = name
        if sandbox_id is not None:
            record["id"] = sandbox_id
        self._record(record)

    def open_entries(
        self, selector: Mapping[str, str] | None = None
    ) -> list[LedgerEntry]:
        """Copies of the open entries whose metadata holds ``selector``."""
        with self._lock:
            entries = list(self._load().open.values())
        wanted = dict(selector or {})
        return [
            replace(entry, metadata=dict(entry.metadata))
            for entry in entries
            if wanted.items() <= entry.metadata.items()
        ]

    # -- the file ------------------------------------------------------------

    def _record(self, record: dict[str, object]) -> None:
        record["at"] = self.clock()
        with self._lock:
            try:
                fold = self._load()
            except (InfrastructureError, OSError) as error:
                LOGGER.warning(
                    "sandbox ledger %s: %s record not written (%s)",
                    self.path,
                    record["event"],
                    error
                    if isinstance(error, InfrastructureError)
                    else type(error).__name__,
                )
                return
            try:
                self._append(record)
            except OSError as error:
                LOGGER.warning(
                    "sandbox ledger %s: %s record not written (%s)",
                    self.path,
                    record["event"],
                    type(error).__name__,
                )
            fold.apply(record)

    def _load(self) -> _Fold:
        if self._fold is not None:
            return self._fold
        fold = _Fold()
        try:
            descriptor = os.open(self.path, os.O_RDONLY)
        except FileNotFoundError:
            self._fold = fold
            return fold
        except OSError as error:
            raise InfrastructureError(
                f"sandbox ledger {self.path} is unreadable ({type(error).__name__})"
            ) from None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH)
            with os.fdopen(os.dup(descriptor), "rb") as stream:
                lines = stream.read().split(b"\n")
        finally:
            os.close(descriptor)
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                if len(line) > MAX_LINE:
                    raise ValueError("line too long")
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("not an object")
            except ValueError:
                LOGGER.warning("sandbox ledger %s: line %d skipped", self.path, number)
                continue
            fold.apply(record)
        self._fold = fold
        return fold

    def _append(self, record: Mapping[str, object]) -> None:
        line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        durable_mkdir(self.path.parent)
        new = not self.path.exists()
        descriptor = os.open(
            self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            size = os.fstat(descriptor).st_size
            data = line.encode()
            if size and os.pread(descriptor, 1, size - 1) != b"\n":
                # A torn last line (a crash mid-write): end it, or this
                # record would join it and be skipped with it.
                data = b"\n" + data
            os.write(descriptor, data)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if new:
            fsync_directory(self.path.parent)


__all__ = ["LedgerEntry", "SandboxLedger", "sandbox_ledger_path"]
