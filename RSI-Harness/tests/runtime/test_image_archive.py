"""The built-image export sanitizer on crafted docker-archive streams (B8)."""

import hashlib
import io
import json
import tarfile

import pytest

from rsi_harness.runtime.image_archive import (
    ImageArchiveError,
    ImageArchiveSanitizer,
)

MIB = 1024**2


def blob(data):
    return hashlib.sha256(data).hexdigest(), data


CONFIG = blob(b'{"architecture":"amd64","os":"linux","config":{}}')
LAYER = blob(b"layer-bytes" * 100)
OTHER = blob(b"other-layer")
MANIFEST = blob(b'{"schemaVersion":2}')


def export(
    *,
    blobs=(CONFIG, LAYER, MANIFEST),
    manifest=None,
    index=None,
    extra=(),
    order=("blobs", "index.json", "manifest.json", "oci-layout"),
):
    """A BuildKit-shaped ``type=docker`` export: blobs, then the JSON."""
    if manifest is None:
        manifest = [
            {
                "Config": f"blobs/sha256/{CONFIG[0]}",
                "RepoTags": None,
                "Layers": [f"blobs/sha256/{LAYER[0]}"],
            }
        ]
    if index is None:
        index = {
            "schemaVersion": 2,
            "manifests": [
                {
                    "digest": "sha256:" + MANIFEST[0],
                    "annotations": {
                        "org.opencontainers.image.created": "2026-09-30T00:00:00Z"
                    },
                }
            ],
        }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:

        def add(name, data, kind=tarfile.REGTYPE, **fields):
            info = tarfile.TarInfo(name)
            info.type = kind
            for key, value in fields.items():
                setattr(info, key, value)
            if kind == tarfile.REGTYPE:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)

        for part in order:
            if part == "blobs":
                add("blobs", b"", tarfile.DIRTYPE)
                add("blobs/sha256", b"", tarfile.DIRTYPE)
                for digest, data in blobs:
                    add(f"blobs/sha256/{digest}", data)
            elif part == "index.json":
                add("index.json", json.dumps(index).encode())
            elif part == "manifest.json":
                add("manifest.json", json.dumps(manifest).encode())
            else:
                add("oci-layout", b'{"imageLayoutVersion":"1.0.0"}')
        for name, data, kind, fields in extra:
            add(name, data, kind, **fields)
    return buffer.getvalue()


def run(data, *, max_bytes=64 * MIB):
    events = []
    output = bytearray()

    def on_config(image_id):
        events.append(("config", image_id, len(output)))

    sanitizer = ImageArchiveSanitizer(
        io.BytesIO(data), max_bytes=max_bytes, on_config=on_config
    )
    try:
        for chunk in sanitizer:
            output.extend(chunk)
    except ImageArchiveError as error:
        return sanitizer, events, bytes(output), error
    return sanitizer, events, bytes(output), None


def members(data):
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {
            member.name: (tar.extractfile(member).read() if member.isreg() else None)
            for member in tar.getmembers()
        }


def test_a_clean_export_passes_with_forced_null_tags_and_its_config_id():
    sanitizer, events, output, error = run(export())
    assert error is None
    image_id = "sha256:" + CONFIG[0]
    assert sanitizer.result.image_id == image_id
    assert sanitizer.result.layers == ("sha256:" + LAYER[0],)
    found = members(output)
    assert set(found) == {
        "blobs",
        "blobs/sha256",
        f"blobs/sha256/{CONFIG[0]}",
        f"blobs/sha256/{LAYER[0]}",
        f"blobs/sha256/{MANIFEST[0]}",
        "index.json",
        "manifest.json",
        "oci-layout",
    }
    assert json.loads(found["manifest.json"])[0]["RepoTags"] is None
    assert found[f"blobs/sha256/{LAYER[0]}"] == LAYER[1]


def test_injected_repo_tags_are_rewritten_to_null():
    manifest = [
        {
            "Config": f"blobs/sha256/{CONFIG[0]}",
            "RepoTags": ["ubuntu:24.04", "busybox:1.37.0"],
            "Layers": [f"blobs/sha256/{LAYER[0]}"],
        }
    ]
    _, _, output, error = run(export(manifest=manifest))
    assert error is None
    assert json.loads(members(output)["manifest.json"]) == [
        {
            "Config": f"blobs/sha256/{CONFIG[0]}",
            "RepoTags": None,
            "Layers": [f"blobs/sha256/{LAYER[0]}"],
        }
    ]


def test_index_ref_name_annotations_are_stripped():
    index = {
        "schemaVersion": 2,
        "manifests": [
            {
                "digest": "sha256:" + MANIFEST[0],
                "annotations": {
                    "io.containerd.image.name": "docker.io/library/ubuntu:24.04",
                    "org.opencontainers.image.ref.name": "24.04",
                    "org.opencontainers.image.created": "now",
                },
            }
        ],
    }
    _, _, output, error = run(export(index=index))
    assert error is None
    annotations = json.loads(members(output)["index.json"])["manifests"][0][
        "annotations"
    ]
    assert annotations == {"org.opencontainers.image.created": "now"}


def test_two_images_in_the_manifest_are_refused():
    one = {
        "Config": f"blobs/sha256/{CONFIG[0]}",
        "RepoTags": None,
        "Layers": [f"blobs/sha256/{LAYER[0]}"],
    }
    _, events, output, error = run(export(manifest=[one, one]))
    assert error.kind == "invalid" and "exactly one image" in error.message
    assert events == [] and not output.endswith(b"\0" * 1024)


def test_a_missing_layer_is_refused_before_the_end_of_the_stream():
    manifest = [
        {
            "Config": f"blobs/sha256/{CONFIG[0]}",
            "RepoTags": None,
            "Layers": [f"blobs/sha256/{LAYER[0]}", f"blobs/sha256/{OTHER[0]}"],
        }
    ]
    _, events, output, error = run(export(manifest=manifest))
    assert error.kind == "invalid" and OTHER[0] in error.message
    # The daemon never saw the end: on_config never ran, no EOF blocks.
    assert events == []
    assert output and not output.endswith(b"\0" * 1024)


@pytest.mark.parametrize(
    "extra",
    [
        ("etc/passwd", b"x", tarfile.REGTYPE, {}),
        ("blobs/sha256/../../x", b"x", tarfile.REGTYPE, {}),
        ("blobs/sha256/" + "a" * 64, b"", tarfile.SYMTYPE, {"linkname": "/etc"}),
        ("blobs/sha256/" + "b" * 64, b"", tarfile.LNKTYPE, {"linkname": "index.json"}),
        ("repositories", b"{}", tarfile.REGTYPE, {}),
    ],
)
def test_only_the_member_allowlist_passes(extra):
    _, events, _, error = run(export(extra=[extra]))
    assert error is not None and error.kind == "invalid"
    assert events == []


def test_a_blob_that_does_not_match_its_digest_is_refused():
    forged = (LAYER[0], b"something else")
    _, events, _, error = run(export(blobs=(CONFIG, forged, MANIFEST)))
    assert error.kind == "invalid" and "digest" in error.message
    assert events == []


def test_duplicate_members_are_refused():
    extra = ("manifest.json", b"[]", tarfile.REGTYPE, {})
    _, _, _, error = run(export(extra=[extra]))
    assert error.kind == "invalid" and "duplicate" in error.message


def test_an_export_over_its_cap_stops_before_reading_the_member():
    data = export()
    _, events, output, error = run(data, max_bytes=len(CONFIG[1]) + 10)
    assert error.kind == "quota"
    assert events == []
    # Refused at the header: the oversize layer's bytes were never emitted.
    assert LAYER[1][:64] not in output


def test_on_config_runs_before_the_last_byte_is_emitted():
    sanitizer, events, output, error = run(export())
    assert error is None
    [(kind, image_id, emitted)] = events
    assert (kind, image_id) == ("config", "sha256:" + CONFIG[0])
    # Everything but the end-of-archive blocks was out; those came after.
    assert emitted == len(output) - 1024
    assert output[emitted:] == b"\0" * 1024


def test_a_failing_journal_write_keeps_the_end_of_the_stream_back():
    def on_config(image_id):
        raise OSError("journal disk full")

    sanitizer = ImageArchiveSanitizer(
        io.BytesIO(export()), max_bytes=64 * MIB, on_config=on_config
    )
    output = bytearray()
    with pytest.raises(OSError, match="journal"):
        for chunk in sanitizer:
            output.extend(chunk)
    assert not output.endswith(b"\0" * 1024)
    assert sanitizer.result is None


def test_manifest_before_blobs_is_checked_at_the_end():
    order = ("manifest.json", "index.json", "blobs", "oci-layout")
    sanitizer, events, _, error = run(export(order=order))
    assert error is None and len(events) == 1
    assert sanitizer.result.image_id == "sha256:" + CONFIG[0]


def test_not_a_tar_is_invalid():
    _, events, _, error = run(b"\x89PNG not a tar" * 100)
    assert error is not None and error.kind == "invalid"
    assert events == []
