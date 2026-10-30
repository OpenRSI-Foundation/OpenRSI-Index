"""The run's remote sandbox ledger (sandbox_remote_ledger): one JSON lines
file beside the run's lease, folded into the sandboxes not proven gone."""

from __future__ import annotations

import json
import multiprocessing
import os
import stat
from pathlib import Path

import pytest

from rsi_harness.errors import InfrastructureError
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.sandbox_remote import bind_ledger
from rsi_harness.runtime.sandbox_remote_ledger import (
    SandboxLedger,
    sandbox_ledger_path,
)

RUN = {"rsi_run_id": "run-1"}


def make(tmp_path, clock=None):
    path = sandbox_ledger_path(tmp_path / "leases", "run-1")
    return SandboxLedger(path, "run-1", clock=clock or (lambda: 1000.0))


def test_the_ledger_sits_beside_the_lease_and_is_no_lease(tmp_path):
    store = LeaseStore(tmp_path / "leases")
    store.root.mkdir()
    store.path_for("run-1").write_text("{}")  # a lease, as far as listing goes
    ledger = make(tmp_path)
    ledger.creating("rsi-a", {**RUN, "rsi_env_id": "ea"})
    assert ledger.path == store.root / "run-1.sandboxes.jsonl"
    assert stat.S_IMODE(os.stat(ledger.path).st_mode) == 0o600
    # The lease store still lists only leases.
    assert store.list_run_ids() == ("run-1",)


def test_entries_open_at_create_learn_their_id_and_close_at_their_end(tmp_path):
    ledger = make(tmp_path)
    ledger.creating("rsi-a", {**RUN, "rsi_env_id": "ea"})
    ledger.creating("rsi-b", {**RUN, "rsi_env_id": "eb"})
    ledger.created("rsi-a", "sb-a")
    (a,) = ledger.open_entries({"rsi_env_id": "ea"})
    assert (a.name, a.sandbox_id, a.created_at) == ("rsi-a", "sb-a", 1000.0)
    ledger.noted("sb-a", "leftover (stopped)")
    ledger.ended("stopped", sandbox_id="sb-a")
    ledger.ended("absent", name="rsi-b")
    assert ledger.open_entries() == []
    # An end of what the ledger never held writes nothing.
    size = ledger.path.stat().st_size
    ledger.ended("gone", sandbox_id="sb-unknown")
    assert ledger.path.stat().st_size == size


def test_a_new_reader_folds_the_file_as_the_writer_left_it(tmp_path):
    writer = make(tmp_path)
    writer.creating("rsi-a", RUN)
    writer.created("rsi-a", "sb-a")
    writer.creating("rsi-b", RUN)
    writer.ended("refused", name="rsi-b")
    writer.noted("sb-a", "leftover (error)")
    reader = make(tmp_path)
    ((entry),) = reader.open_entries(RUN)
    assert (entry.name, entry.sandbox_id, entry.note) == (
        "rsi-a",
        "sb-a",
        "leftover (error)",
    )
    assert reader.open_entries({"rsi_run_id": "run-2"}) == []


def test_an_end_is_final_and_a_torn_line_is_skipped(tmp_path):
    ledger = make(tmp_path)
    ledger.creating("rsi-a", RUN)
    ledger.created("rsi-a", "sb-a")
    ledger.ended("stopped", sandbox_id="sb-a")
    with open(ledger.path, "a") as file:
        # A late answer naming the id that ended never reopens it; then a
        # line cut by a crash.
        file.write(json.dumps({"event": "created", "name": "rsi-a", "id": "sb-a"}))
        file.write('\n{"event": "create", "name": "rsi-c"')
    assert make(tmp_path).open_entries() == []


def test_a_new_request_of_a_name_that_ended_opens_it_again(tmp_path):
    ledger = make(tmp_path)
    ledger.creating("rsi-a", RUN)
    ledger.ended("refused", name="rsi-a")
    ledger.creating("rsi-a", RUN)
    ledger.created("rsi-a", "sb-a2")
    for reader in (ledger, make(tmp_path)):
        ((entry),) = reader.open_entries()
        assert (entry.name, entry.sandbox_id) == ("rsi-a", "sb-a2")


def test_a_create_is_refused_when_its_record_cannot_be_written(tmp_path):
    blocked = tmp_path / "leases"
    blocked.write_text("not a directory")
    ledger = make(tmp_path)
    with pytest.raises(InfrastructureError, match="sandbox ledger .* is unre"):
        ledger.creating("rsi-a", RUN)
    # The other records only log.
    ledger.created("rsi-a", "sb-a")
    ledger.ended("gone", sandbox_id="sb-a")
    if os.geteuid() == 0:
        return  # root writes into a read-only directory
    blocked.unlink()
    blocked.mkdir()
    blocked.chmod(0o500)
    try:
        with pytest.raises(InfrastructureError, match="no sandbox was created"):
            make(tmp_path).creating("rsi-a", RUN)
    finally:
        blocked.chmod(0o700)


def test_only_a_client_that_keeps_a_ledger_is_bound(tmp_path):
    class Keeps:
        ledger = None

    class Lacks:
        pass

    keeps, lacks = Keeps(), Lacks()
    bind_ledger(keeps, tmp_path / "leases", "run-1")
    bind_ledger(lacks, tmp_path / "leases", "run-1")
    assert keeps.ledger.path == tmp_path / "leases" / "run-1.sandboxes.jsonl"
    assert keeps.ledger.run_id == "run-1"
    assert not hasattr(lacks, "ledger")


def test_an_append_after_a_torn_line_is_whole(tmp_path):
    ledger = make(tmp_path)
    ledger.creating("rsi-a", RUN)
    with open(ledger.path, "a") as file:
        file.write('{"event":"created","id":"sb-a","na')  # a crash mid-write
    # The resumed attempt's first record: a durable create.
    resumed = make(tmp_path)
    resumed.creating("rsi-b", RUN)
    names = sorted(entry.name for entry in make(tmp_path).open_entries())
    assert names == ["rsi-a", "rsi-b"]
    assert ledger.path.read_bytes().endswith(b"\n")


def _append_creates(path, prefix, count):
    ledger = SandboxLedger(path, "run-1")
    for number in range(count):
        ledger.creating(f"{prefix}-{number}", RUN)


def test_two_processes_appending_to_one_ledger_lose_no_record(tmp_path):
    path = sandbox_ledger_path(tmp_path / "leases", "run-1")
    context = multiprocessing.get_context("spawn")
    workers = [
        context.Process(target=_append_creates, args=(path, prefix, 50))
        for prefix in ("rsi-a", "rsi-b")
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)
        assert worker.exitcode == 0
    assert len(SandboxLedger(path, "run-1").open_entries()) == 100


def test_a_relative_ledger_path_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ledger = SandboxLedger(sandbox_ledger_path(Path("leases"), "run-1"), "run-1")
    assert ledger.path == tmp_path / "leases" / "run-1.sandboxes.jsonl"
    ledger.creating("rsi-a", RUN)
    assert [entry.name for entry in make(tmp_path).open_entries()] == ["rsi-a"]
