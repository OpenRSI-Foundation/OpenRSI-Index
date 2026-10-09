"""Typed host boundary over the same stdlib primitives used by the client."""

import io

from rsi_harness.integrations import sandbox_client as wire
from rsi_harness.runtime.sandbox_contracts import SandboxBundleEntry, SandboxError


def _translate(operation):
    try:
        return operation()
    except (wire.ProtocolError, OSError, ValueError) as error:
        raise SandboxError(
            getattr(error, "code", "invalid"),
            getattr(error, "field", "bundle"),
            str(error),
        ) from error


def encode_bundle(entries, *, byte_limit=wire.MAX_BUNDLE_BYTES):
    return _translate(
        lambda: wire.encode_records(
            (entry.model_dump() for entry in entries), max_bytes=byte_limit
        )
    )


def decode_bundle(data, *, byte_limit=wire.MAX_BUNDLE_BYTES):
    return _translate(
        lambda: tuple(
            SandboxBundleEntry(**entry)
            for entry in wire.iter_bundle(io.BytesIO(data), max_bytes=byte_limit)
        )
    )


def read_local_bundle(root):
    return _translate(
        lambda: tuple(
            SandboxBundleEntry(**entry) for entry in wire.iter_local_records(root)
        )
    )


def write_local_bundle(root, entries):
    return _translate(
        lambda: wire.write_local_records(
            root, (entry.model_dump() for entry in entries)
        )
    )
