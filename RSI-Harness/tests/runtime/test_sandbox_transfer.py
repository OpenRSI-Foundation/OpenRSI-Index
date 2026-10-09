"""Untrusted bundle names, metadata, sizes and local filesystem races."""

import io
import json
import os
import struct

import pytest


@pytest.fixture
def api():
    from rsi_harness.runtime import sandbox_contracts, sandbox_transfer

    return sandbox_contracts, sandbox_transfer


def frame(path="result", kind="file", mode=0o644, data=b"x", **changes):
    header = {"path": path, "kind": kind, "mode": mode, "size": len(data)}
    header.update(changes)
    encoded = json.dumps(header).encode()
    return struct.pack("!I", len(encoded)) + encoded + data


def bundle(*frames):
    return b"RSIBNDL1\n" + b"".join(frames) + struct.pack("!I", 0)


@pytest.mark.parametrize(
    "path",
    ["../escape", "/absolute", "a/../../b", "a\x00b", "a//b", "./a", "", "a/", "a\\b"],
)
def test_invalid_names_cannot_escape_target(api, path):
    contracts, transfer = api
    with pytest.raises(contracts.SandboxError):
        transfer.decode_bundle(bundle(frame(path=path)))


def test_directory_and_file_roundtrip_preserves_empty_directory_and_modes(api):
    contracts, transfer = api
    entries = (
        contracts.SandboxBundleEntry(path="empty", kind="directory", mode=0o755),
        contracts.SandboxBundleEntry(
            path="result", kind="file", mode=0o640, data=b"ok"
        ),
    )
    assert transfer.decode_bundle(transfer.encode_bundle(entries)) == entries


@pytest.mark.parametrize(
    "changes",
    [
        {"size": -1},
        {"size": True},
        {"size": "1"},
        {"size": 16 * 1024**2 + 1},
        {"kind": "symlink"},
        {"kind": "hardlink"},
        {"kind": "fifo"},
        {"mode": 0o4755},
        {"mode": -1},
        {"mode": True},
        {"extra": "unknown"},
        {"kind": "directory", "size": 1},
    ],
)
def test_invalid_headers_rejected_before_body_allocation(api, changes):
    contracts, transfer = api
    with pytest.raises(contracts.SandboxError):
        transfer.decode_bundle(bundle(frame(**changes)))


def test_duplicate_paths_and_file_as_parent_are_rejected(api):
    contracts, transfer = api
    for payload in (
        bundle(frame(), frame()),
        bundle(frame(path="a"), frame(path="a/b")),
        bundle(frame(path="a/b"), frame(path="a")),
    ):
        with pytest.raises(contracts.SandboxError):
            transfer.decode_bundle(payload)


@pytest.mark.parametrize(
    "payload",
    [
        b"bad",
        b"RSIBNDL1\n\x00",
        b"RSIBNDL1\n" + struct.pack("!I", 4097),
        bundle(frame())[:-1],
        bundle(frame()) + b"extra",
        bundle(frame(size=10)),
    ],
)
def test_truncation_oversized_metadata_and_trailing_data_are_errors(api, payload):
    contracts, transfer = api
    with pytest.raises(contracts.SandboxError):
        transfer.decode_bundle(payload)


def test_entry_count_and_file_byte_boundaries_are_enforced(api):
    contracts, transfer = api
    entries = tuple(
        contracts.SandboxBundleEntry(path=f"empty-{i}", kind="directory", mode=0o755)
        for i in range(1024)
    )
    assert len(transfer.decode_bundle(transfer.encode_bundle(entries))) == 1024
    with pytest.raises(contracts.SandboxError):
        transfer.decode_bundle(
            bundle(
                *(frame(path=f"d-{i}", kind="directory", data=b"") for i in range(1025))
            )
        )
    exact = b"x" * (16 * 1024**2)
    assert len(transfer.decode_bundle(bundle(frame(data=exact)))[0].data) == len(exact)
    with pytest.raises(contracts.SandboxError):
        transfer.decode_bundle(
            bundle(
                frame(path="a", data=exact),
                frame(path="b", data=exact),
                frame(path="c"),
            )
        )


def test_stream_decoder_does_not_read_unbounded_or_entire_body_at_once():
    from rsi_harness.integrations.sandbox_client import iter_bundle

    class BoundedRead(io.BytesIO):
        def read(self, count=-1):
            assert 0 <= count <= 65536
            return super().read(count)

    records = tuple(iter_bundle(BoundedRead(bundle(frame(data=b"a" * 100000)))))
    assert records[0]["data"] == b"a" * 100000


def test_real_local_io_roundtrip_and_empty_directories(api, tmp_path):
    _, transfer = api
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty").mkdir()
    (source / "file").write_bytes(b"real")
    os.chmod(source / "file", 0o640)
    entries = transfer.read_local_bundle(source)
    target = tmp_path / "target"
    transfer.write_local_bundle(target, entries)
    assert (target / "empty").is_dir()
    assert (target / "file").read_bytes() == b"real"
    assert (target / "file").stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_local_export_never_reads_links_or_special_files(api, tmp_path, kind):
    contracts, transfer = api
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "private"
    outside.write_bytes(b"private")
    if kind == "symlink":
        (root / "bad").symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, root / "bad")
    else:
        os.mkfifo(root / "bad")
    with pytest.raises(contracts.SandboxError):
        transfer.read_local_bundle(root)


def test_download_destination_symlink_cannot_write_outside_root(api, tmp_path):
    contracts, transfer = api
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "dir").symlink_to(outside, target_is_directory=True)
    entries = (
        contracts.SandboxBundleEntry(
            path="dir/file", kind="file", mode=0o644, data=b"bad"
        ),
    )
    with pytest.raises(contracts.SandboxError):
        transfer.write_local_bundle(root, entries)
    assert list(outside.iterdir()) == []


def test_local_root_ancestor_symlink_is_rejected(api, tmp_path):
    contracts, transfer = api
    actual = tmp_path / "actual"
    actual.mkdir()
    (tmp_path / "alias").symlink_to(actual, target_is_directory=True)
    with pytest.raises(contracts.SandboxError):
        transfer.write_local_bundle(tmp_path / "alias" / "child", ())
    assert list(actual.iterdir()) == []


def test_deep_paths_and_deep_json_are_bounded_errors(api):
    contracts, transfer = api
    payloads = [
        bundle(frame(path="/".join(["d"] * 33))),
        b"RSIBNDL1\n" + struct.pack("!I", 2201) + b"[" * 1100 + b"0" + b"]" * 1100,
    ]
    for payload in payloads:
        with pytest.raises(contracts.SandboxError):
            transfer.decode_bundle(payload)


def test_export_detects_directory_to_symlink_replacement(api, tmp_path, monkeypatch):
    contracts, transfer = api
    from rsi_harness.integrations import sandbox_client

    root = tmp_path / "root"
    root.mkdir()
    (root / "child").mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret").write_text("must not export")
    original = os.open

    def replace(name, flags, *args, **kwargs):
        if name == "child":
            (root / "child").rmdir()
            (root / "child").symlink_to(private, target_is_directory=True)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(sandbox_client.os, "open", replace)
    with pytest.raises(contracts.SandboxError):
        transfer.read_local_bundle(root)


def test_selected_download_does_not_touch_unrequested_special_file(tmp_path):
    from rsi_harness.integrations.sandbox_client import iter_selected_records

    (tmp_path / "wanted").write_bytes(b"selected")
    os.mkfifo(tmp_path / "unrequested")
    records = tuple(iter_selected_records(tmp_path, ("wanted",)))
    assert len(records) == 1
    assert records[0]["data"] == b"selected"


def test_remaining_transfer_budget_rejects_before_file_read(tmp_path, monkeypatch):
    from rsi_harness.integrations import sandbox_client as wire

    (tmp_path / "too-large").write_bytes(b"1234")
    real_open = wire.os.open

    def deny_read(name, flags, *args, **kwargs):
        if name == "too-large":
            pytest.fail("remaining byte budget must be checked before opening the file")
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(wire.os, "open", deny_read)
    with pytest.raises(wire.ProtocolError, match="quota"):
        tuple(wire.iter_selected_records(tmp_path, ("too-large",), max_bytes=3))


def test_decoder_uses_smaller_remaining_byte_budget(api):
    contracts, transfer = api
    with pytest.raises(contracts.SandboxError, match="quota"):
        transfer.decode_bundle(bundle(frame(data=b"1234")), byte_limit=3)
