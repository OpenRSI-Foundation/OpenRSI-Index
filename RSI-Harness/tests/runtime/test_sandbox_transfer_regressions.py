"""Transfer permission, control-size, and trusted failure-accounting regressions."""

import io
import subprocess
import time
from contextlib import contextmanager

import pytest

from rsi_harness.errors import InfrastructureError
from rsi_harness.integrations import sandbox_client as wire
from rsi_harness.runtime.sandbox_contracts import (
    SandboxChildStopped,
    SandboxError,
    SandboxResult,
)
from rsi_harness.runtime.sandbox_docker import SandboxDockerBackend
from tests.runtime.test_sandbox_budget import make_child
from tests.runtime.test_sandbox_docker import DockerClient
from tests.sandbox_helpers import make_profile


@pytest.fixture
def local_transfer(tmp_path, monkeypatch):
    """Run the actual fixed transfer helper, replacing only Docker transport."""
    backend = SandboxDockerBackend(DockerClient())
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)

    def run_helper(lease, argv, cwd, env, deadline, output_limit, input_bytes=None):
        completed = subprocess.run(
            argv, input=input_bytes, capture_output=True, timeout=3, check=False
        )
        return (
            SandboxResult(exit_code=completed.returncode, duration_sec=0.01),
            completed.stdout,
            completed.stderr,
        )

    monkeypatch.setattr(backend, "_run_exec", run_helper)
    return backend, lease, tmp_path


@pytest.mark.parametrize("parent_first", [True, False])
def test_directory_modes_are_applied_after_descendants(tmp_path, parent_first):
    """Applying 0555/0500 eagerly must not prevent writing child records."""
    parent = {"path": "fixtures", "kind": "directory", "mode": 0o555, "data": b""}
    nested = {
        "path": "fixtures/nested",
        "kind": "directory",
        "mode": 0o500,
        "data": b"",
    }
    content = {
        "path": "fixtures/nested/result",
        "kind": "file",
        "mode": 0o444,
        "data": b"expected",
    }
    records = (parent, nested, content) if parent_first else (content, parent, nested)
    try:
        wire.write_local_records(tmp_path, records)
        assert (tmp_path / "fixtures/nested/result").read_bytes() == b"expected"
        assert (tmp_path / "fixtures").stat().st_mode & 0o777 == 0o555
        assert (tmp_path / "fixtures/nested").stat().st_mode & 0o777 == 0o500
        assert (tmp_path / "fixtures/nested/result").stat().st_mode & 0o777 == 0o444
    finally:
        for path in (tmp_path / "fixtures", tmp_path / "fixtures/nested"):
            if path.exists():
                path.chmod(0o700)


@pytest.mark.parametrize(
    "mode,parent_first", [(0o555, False), (0o555, True), (0o300, True)]
)
def test_repeated_transfer_prepares_declared_directories(tmp_path, mode, parent_first):
    parent = {"path": "d", "kind": "directory", "mode": mode, "data": b""}
    content = {"path": "d/f", "kind": "file", "mode": 0o644, "data": b"first"}
    try:
        wire.write_local_records(tmp_path, (parent, content))
        content = {**content, "data": b"second"}
        records = (parent, content) if parent_first else (content, parent)
        wire.write_local_records(tmp_path, iter(records))
        assert (tmp_path / "d").stat().st_mode & 0o777 == mode
        assert (tmp_path / "d/f").read_bytes() == b"second"
    finally:
        (tmp_path / "d").chmod(0o700)


def test_directory_modes_are_restored_when_later_record_fails(tmp_path, monkeypatch):
    records = (
        {"path": "fixtures", "kind": "directory", "mode": 0o555, "data": b""},
        {"path": "fixtures/f", "kind": "file", "mode": 0o644, "data": b"bad"},
    )

    def fail_replace(*args, **kwargs):
        raise OSError("write failed")

    monkeypatch.setattr(wire.os, "replace", fail_replace)
    try:
        with pytest.raises(OSError, match="write failed"):
            wire.write_local_records(tmp_path, records)
        assert (tmp_path / "fixtures").stat().st_mode & 0o777 == 0o555
    finally:
        (tmp_path / "fixtures").chmod(0o700)


def test_deferred_mode_does_not_change_replacement_directory(tmp_path):
    """Postorder chmod must retain inode authority across a directory replacement."""

    (tmp_path / "fixtures").mkdir(mode=0o555)

    def records():
        yield {"path": "fixtures", "kind": "directory", "mode": 0o555, "data": b""}
        (tmp_path / "fixtures").rename(tmp_path / "original")
        (tmp_path / "fixtures").mkdir(mode=0o755)
        yield {"path": "result", "kind": "file", "mode": 0o644, "data": b"ok"}

    with pytest.raises(wire.ProtocolError, match="directory changed"):
        wire.write_local_records(tmp_path, records())
    assert (tmp_path / "fixtures").stat().st_mode & 0o777 == 0o755


def test_mode_restoration_rejects_directory_replaced_during_write(
    tmp_path, monkeypatch
):
    original = wire.os.replace

    def replace_then_move(*args, **kwargs):
        original(*args, **kwargs)
        (tmp_path / "d").rename(tmp_path / "moved")
        (tmp_path / "d").mkdir(mode=0o755)

    monkeypatch.setattr(wire.os, "replace", replace_then_move)
    records = (
        {"path": "d", "kind": "directory", "mode": 0o555, "data": b""},
        {"path": "d/f", "kind": "file", "mode": 0o644, "data": b"ok"},
    )
    with pytest.raises(wire.ProtocolError, match="directory changed"):
        wire.write_local_records(tmp_path, records)
    assert (tmp_path / "d").stat().st_mode & 0o777 == 0o755


def test_undeclared_readonly_parent_is_not_chmodded(tmp_path):
    (tmp_path / "unrelated").mkdir(mode=0o555)
    records = ({"path": "unrelated/f", "kind": "file", "mode": 0o644, "data": b"x"},)
    try:
        with pytest.raises(PermissionError):
            wire.write_local_records(tmp_path, records)
        assert (tmp_path / "unrelated").stat().st_mode & 0o777 == 0o555
        assert not (tmp_path / "unrelated/f").exists()
    finally:
        (tmp_path / "unrelated").chmod(0o700)


def test_preparation_cannot_chmod_directory_replaced_by_symlink(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir(mode=0o300)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o500)
    original = wire.os.chmod
    replaced = False

    def replace_before_chmod(path, mode, *args, **kwargs):
        nonlocal replaced
        if str(path).startswith("/proc/self/fd/") and not replaced:
            replaced = True
            (tmp_path / "d").rename(tmp_path / "moved")
            (tmp_path / "d").symlink_to(outside, target_is_directory=True)
        return original(path, mode, *args, **kwargs)

    monkeypatch.setattr(wire.os, "chmod", replace_before_chmod)
    with pytest.raises(OSError):
        wire.write_local_records(
            tmp_path, ({"path": "d", "kind": "directory", "mode": 0o555, "data": b""},)
        )
    assert outside.stat().st_mode & 0o777 == 0o500
    original(outside, 0o700)


def test_preparation_validates_entry_bound_before_mutating(tmp_path):
    records = (
        {"path": f"d{i}", "kind": "directory", "mode": 0o555, "data": b""}
        for i in range(1025)
    )
    with pytest.raises(wire.ProtocolError, match="quota"):
        wire.write_local_records(tmp_path, records)
    assert list(tmp_path.iterdir()) == []


def test_deep_bundle_paths_do_not_retain_every_ancestor_prefix():
    import tracemalloc

    records = tuple(
        {
            "path": f"{index:04d}-" + "x" * 3000 + "/d" * 31,
            "kind": "directory",
            "mode": 0o755,
            "data": b"",
        }
        for index in range(64)
    )
    tracemalloc.start()
    try:
        encoded = wire.encode_records(records)
        decoded = tuple(wire.iter_bundle(io.BytesIO(encoded)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert decoded == records
    assert peak < 2 * 1024**2, peak


def test_valid_selection_metadata_does_not_share_exec_argv_budget(local_transfer):
    """The injected helper's source must not consume valid download metadata."""
    backend, lease, root = local_transfer
    paths = tuple(f"{index:04d}-" + "f" * 64 for index in range(1024))
    for path in paths:
        (root / path).write_bytes(b"x")
    records = backend.download(lease, str(root), paths, time.monotonic() + 5)
    assert len(records) == 1024
    assert sum(len(record.data) for record in records) == 1024


def test_public_exec_argument_limit_is_not_relaxed(local_transfer):
    backend, lease, _root = local_transfer
    with pytest.raises(SandboxError, match="64 KiB"):
        backend.execute(
            lease,
            ["echo", "x" * wire.MAX_COMMAND_BYTES],
            "/workspace",
            {},
            time.monotonic() + 5,
            1024,
        )


@pytest.mark.parametrize(
    "paths, expected_bytes", [(("missing",), 0), (("present", "missing"), 3)]
)
def test_completed_download_failure_reports_observed_bytes(
    local_transfer, paths, expected_bytes
):
    backend, lease, root = local_transfer
    (root / "present").write_bytes(b"abc")
    with pytest.raises(SandboxError) as caught:
        backend.download(lease, str(root), paths, time.monotonic() + 5)
    assert getattr(caught.value, "download_bytes", None) == expected_bytes
    assert getattr(caught.value, "operation_started", None) is True


def test_download_preexec_rejection_reports_no_operation_or_bytes(local_transfer):
    backend, lease, root = local_transfer
    with pytest.raises(SandboxError) as caught:
        backend.download(lease, str(root), ("../invalid",), time.monotonic() + 5)
    assert getattr(caught.value, "download_bytes", None) == 0
    assert getattr(caught.value, "operation_started", None) is False


def test_download_control_limit_still_rejects_before_operation(local_transfer):
    backend, lease, root = local_transfer
    paths = tuple(f"{index}-" + "f" * 3000 for index in range(64))
    with pytest.raises(SandboxError, match="selection exceeds limit") as caught:
        backend.download(lease, str(root), paths, time.monotonic() + 5)
    assert caught.value.download_bytes == 0
    assert caught.value.operation_started is False


def test_partial_download_charges_observed_bytes_not_helper_claims(
    local_transfer, monkeypatch
):
    backend, lease, root = local_transfer
    payload = (
        wire.encode_records(
            (
                {
                    "path": "present",
                    "kind": "file",
                    "mode": 0o644,
                    "data": b"abc",
                },
            )
        )[:-4]
        + b"garbage"
    )
    monkeypatch.setattr(
        backend,
        "_run_exec",
        lambda *a, **k: (
            SandboxResult(exit_code=2, duration_sec=0.1),
            payload,
            b'{"download_bytes":0}',
        ),
    )
    with pytest.raises(SandboxError) as caught:
        backend.download(lease, str(root), ("present",), time.monotonic() + 5)
    # Three validated file bytes plus seven unparsed bytes; the child cannot
    # make host accounting trust its claim that nothing was transferred.
    assert caught.value.download_bytes == 10
    assert caught.value.operation_started is True


def test_download_unknown_transport_outcome_has_no_refundable_evidence(
    local_transfer, monkeypatch
):
    backend, lease, root = local_transfer

    def unknown(*args, **kwargs):
        raise SandboxError("unknown-outcome", "exec", "worker has not reconciled")

    monkeypatch.setattr(backend, "_run_exec", unknown)
    with pytest.raises(SandboxError) as caught:
        backend.download(lease, str(root), ("present",), time.monotonic() + 5)
    assert caught.value.code == "unknown-outcome"
    assert not hasattr(caught.value, "download_bytes")


def test_stopped_download_keeps_containment_type_and_full_reservation(
    local_transfer, monkeypatch
):
    backend, lease, root = local_transfer
    payload = io.BytesIO()
    wire.write_bundle(
        payload,
        (
            {
                "path": "present",
                "kind": "file",
                "mode": 0o644,
                "data": b"abc",
            },
        ),
    )
    monkeypatch.setattr(
        backend,
        "_run_exec",
        lambda *a, **k: (
            SandboxResult(exit_code=None, duration_sec=0.1, timed_out=True),
            payload.getvalue(),
            b"",
        ),
    )
    with pytest.raises(SandboxChildStopped) as caught:
        backend.download(
            lease, str(root), ("present",), time.monotonic() + 5, byte_limit=1024
        )
    assert getattr(caught.value, "download_bytes", None) == 1024
    assert getattr(caught.value, "operation_started", None) is True


def test_resume_admission_rejection_prevents_actual_unpause():
    """A cancellation observed after inspection must gate the final mutation."""
    client = DockerClient()
    backend = SandboxDockerBackend(client)
    identity = backend.create(make_child(), make_profile())
    lease = make_child(container_id=identity)
    backend.start(lease)
    backend.pause(lease)

    @contextmanager
    def denied():
        raise InfrastructureError("resume admission closed")
        yield

    with pytest.raises(InfrastructureError, match="admission closed"):
        backend.resume(lease, admission=denied)
    assert backend.inspect(lease)["State"]["Paused"] is True
