"""Operator-pinned static tools the broker copies into env services.

Only tmux today: Harbor's terminus-2 installs it with apt/apk when an image
has none, which an env without network cannot. The operator builds it once
(scripts/operator/build_static_tmux.sh) and names it with its SHA-256 in
``[environments.host.tmux]``; a caller then asks ``tool_install`` for an env
of its own. The broker reads the file, never Work or Judge code, and copies
the very bytes it hashed through the archive copy path: no host path, mount
or socket reaches a child, and a mismatch copies nothing.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import stat
from typing import Any

from rsi_harness.errors import SetupError
from rsi_harness.runtime.sandbox_archive import CanonicalTarWriter
from rsi_harness.runtime.sandbox_contracts import EnvToolFile, SandboxError

LOGGER = logging.getLogger(__name__)
MIB = 1024**2
TOOLS = ("tmux",)
TOOL_DIR = "/usr/local/bin"
TOOL_MODE = 0o755
# One stage frame; a static tmux is about 1.4 MiB.
MAX_TOOL_BYTES = 15 * MIB


def tool_file(host: Any, tool: object) -> EnvToolFile:
    """The operator's file for ``tool``; refused unless the policy names it."""
    if type(tool) is not str or tool not in TOOLS:
        raise SandboxError("invalid", "tool", "expected one of: " + ", ".join(TOOLS))
    source = None if host is None else getattr(host, tool, None)
    if source is None:
        raise SandboxError(
            "permission", "tool", f"{tool} is not approved by the operator policy"
        )
    return source


def read_tool(source: EnvToolFile) -> bytes:
    """The file's bytes, only when their SHA-256 is the approved one.

    Fixed refusal texts: the host path and the found hash stay in the
    broker's log, never in an answer to a child.
    """
    try:
        # Non-blocking: a FIFO in its place must not hang the broker.
        descriptor = os.open(source.path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        with open(descriptor, "rb") as file:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("not a regular file")
            data = file.read(MAX_TOOL_BYTES + 1)
    except OSError as error:
        LOGGER.warning("sandbox tool file %s: %s", source.path, error)
        raise SandboxError(
            "infrastructure", "tool", "the operator's tool file is unreadable"
        ) from error
    if len(data) > MAX_TOOL_BYTES:
        raise SandboxError(
            "infrastructure", "tool", "the operator's tool file exceeds 15 MiB"
        )
    found = hashlib.sha256(data).hexdigest()
    if found != source.sha256:
        LOGGER.warning(
            "sandbox tool file %s has sha256 %s, not %s",
            source.path,
            found,
            source.sha256,
        )
        raise SandboxError(
            "infrastructure",
            "tool",
            "the operator's tool file differs from its approved sha256",
        )
    return data


def tool_tar(tool: str, data: bytes) -> bytes:
    """A canonical tar of the one root-owned 0755 file ``tool``."""
    buffer = io.BytesIO()
    writer = CanonicalTarWriter(buffer, max_bytes=len(data))
    writer.add_file((tool,), io.BytesIO(data), len(data), mode=TOOL_MODE, mtime=0)
    writer.close()
    return buffer.getvalue()


def check_tools(host: Any) -> None:
    """Setup: every tool the policy names is readable and matches its hash."""
    for tool in TOOLS:
        source = getattr(host, tool, None)
        if source is None:
            continue
        try:
            read_tool(source)
        except SandboxError as error:
            raise SetupError(
                f"sandbox policy environments.host.{tool} ({source.path}): "
                f"{error.message}"
            ) from None


__all__ = [
    "MAX_TOOL_BYTES",
    "TOOLS",
    "TOOL_DIR",
    "TOOL_MODE",
    "check_tools",
    "read_tool",
    "tool_file",
    "tool_tar",
]
