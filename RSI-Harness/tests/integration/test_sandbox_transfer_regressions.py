"""Bounded real-child coverage of transfer permission and selection boundaries."""

import time
from pathlib import Path

import pytest

from rsi_harness.integrations import sandbox_client as wire
from rsi_harness.runtime.sandbox_contracts import (
    SandboxBundleEntry,
    SandboxDownloadError,
    SandboxError,
)
from tests.integration.test_sandbox_resources import live_sandbox as live_sandbox

pytestmark = pytest.mark.integration


def test_real_upload_early_helper_rejection_preserves_child(live_sandbox):
    client, broker, credentials, child, _store, _run_id, _ids = live_sandbox
    result = broker.execute(
        credentials.credential,
        child.child_id,
        ["mkdir", "conflict"],
        "/workspace",
        {},
        3.0,
    )
    assert result.exit_code == 0
    for size in (0, 4 * 1024**2):
        records = [
            SandboxBundleEntry(path="conflict", kind="file", mode=0o644, data=b"x")
        ]
        if size:
            records.append(
                SandboxBundleEntry(
                    path="large", kind="file", mode=0o644, data=b"x" * size
                )
            )
        started = time.monotonic()
        with pytest.raises(SandboxError) as caught:
            broker.upload(
                credentials.credential,
                child.child_id,
                "/workspace",
                tuple(records),
                f"conflict-{size}",
                5.0,
            )
        elapsed = time.monotonic() - started
        assert caught.value.code == "invalid"
        assert "destination is not a unique regular file" in str(caught.value)
        assert elapsed < 3.0
        assert (
            broker.status(credentials.credential, child.child_id)["state"] == "running"
        )
        assert client.containers.get(child.container_id).attrs["State"]["Running"]


@pytest.mark.parametrize(
    "mode,parent_first", [(0o555, False), (0o555, True), (0o300, True)]
)
def test_real_repeated_upload_prepares_declared_directories(
    live_sandbox, mode, parent_first
):
    _client, broker, credentials, child, _store, _run_id, _ids = live_sandbox
    parent = SandboxBundleEntry(path="d", kind="directory", mode=mode)
    content = SandboxBundleEntry(path="d/f", kind="file", mode=0o644, data=b"first")
    broker.upload(
        credentials.credential,
        child.child_id,
        "/workspace",
        (parent, content),
        "first",
        3.0,
    )
    content = content.model_copy(update={"data": b"second"})
    records = (parent, content) if parent_first else (content, parent)
    broker.upload(
        credentials.credential, child.child_id, "/workspace", records, "second", 3.0
    )
    result = broker.execute(
        credentials.credential,
        child.child_id,
        [
            "python3",
            "-c",
            "import os, sys; "
            "assert os.stat('d').st_mode & 0o777 == int(sys.argv[1]); "
            "assert open('d/f','rb').read() == b'second'",
            str(mode),
        ],
        "/workspace",
        {},
        3.0,
    )
    assert result.exit_code == 0, result.stderr


def test_real_readonly_directory_upload_and_download_preserve_modes(
    live_sandbox, tmp_path
):
    _client, broker, credentials, child, _store, _run_id, _ids = live_sandbox
    entries = (
        SandboxBundleEntry(path="fixtures", kind="directory", mode=0o555),
        SandboxBundleEntry(path="fixtures/nested", kind="directory", mode=0o500),
        SandboxBundleEntry(
            path="fixtures/nested/test.txt", kind="file", mode=0o444, data=b"fixture"
        ),
    )
    broker.upload(
        credentials.credential,
        child.child_id,
        "/workspace",
        entries,
        "readonly-tree",
        5.0,
    )
    records = broker.download(
        credentials.credential, child.child_id, "/workspace", ("fixtures",), 5.0
    )
    assert {record.path: record for record in records} == {
        record.path: record for record in entries
    }
    try:
        wire.write_local_records(tmp_path, (record.model_dump() for record in records))
        assert (tmp_path / "fixtures/nested/test.txt").read_bytes() == b"fixture"
        assert (tmp_path / "fixtures").stat().st_mode & 0o777 == 0o555
        assert (tmp_path / "fixtures/nested").stat().st_mode & 0o777 == 0o500
    finally:
        for path in (tmp_path / "fixtures", tmp_path / "fixtures/nested"):
            if path.exists():
                path.chmod(0o700)


def test_real_nonroot_writer_populates_readonly_directory(live_sandbox):
    client, _broker, _credentials, child, _store, _run_id, _ids = live_sandbox
    records = (
        {"path": "fixtures", "kind": "directory", "mode": 0o555, "data": b""},
        {
            "path": "fixtures/test.txt",
            "kind": "file",
            "mode": 0o444,
            "data": b"fixture",
        },
    )
    source = Path(wire.__file__).read_text()
    script = (
        "import os,signal;signal.alarm(5);"
        "scope={'__name__':'_rsi_transfer'};exec(" + repr(source) + ",scope);"
        "assert os.geteuid()==65534;"
        "scope['write_local_records']('/workspace/nonroot'," + repr(records) + ");"
        "assert open('/workspace/nonroot/fixtures/test.txt','rb').read()==b'fixture';"
        "assert os.stat('/workspace/nonroot/fixtures').st_mode & 0o777 == 0o555;"
        "assert os.stat('/workspace/nonroot/fixtures/test.txt').st_mode "
        "& 0o777 == 0o444"
    )
    result = client.containers.get(child.container_id).exec_run(
        ["python3", "-I", "-S", "-c", script],
        user="65534:65534",
        workdir="/workspace",
    )
    assert result.exit_code == 0, result.output


def test_real_download_accepts_full_bounded_selection(live_sandbox):
    _client, broker, credentials, child, _store, _run_id, _ids = live_sandbox
    result = broker.execute(
        credentials.credential,
        child.child_id,
        [
            "python3",
            "-c",
            "from pathlib import Path; "
            "[Path(f'{i:04d}-'+'f'*64).write_bytes(b'x') for i in range(1024)]",
        ],
        "/workspace",
        {},
        5.0,
    )
    assert result.exit_code == 0, result.stderr
    paths = tuple(f"{index:04d}-" + "f" * 64 for index in range(1024))
    records = broker.download(
        credentials.credential, child.child_id, "/workspace", paths, 5.0
    )
    assert len(records) == 1024
    assert all(record.data == b"x" for record in records)


def test_real_missing_download_preserves_allowance_for_corrected_selection(
    live_sandbox,
):
    _client, broker, credentials, child, _store, _run_id, _ids = live_sandbox
    result = broker.execute(
        credentials.credential,
        child.child_id,
        [
            "python3",
            "-c",
            "from pathlib import Path; Path('result.txt').write_bytes(b'ok')",
        ],
        "/workspace",
        {},
        5.0,
    )
    assert result.exit_code == 0, result.stderr
    with pytest.raises(SandboxDownloadError, match="No such file") as caught:
        broker.download(
            credentials.credential, child.child_id, "/workspace", ("missing.txt",), 5.0
        )
    assert caught.value.download_bytes == 0
    records = broker.download(
        credentials.credential, child.child_id, "/workspace", ("result.txt",), 5.0
    )
    assert len(records) == 1
    assert records[0].data == b"ok"
    assert broker._usage["max_download_bytes"] == 2
