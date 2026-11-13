"""FIFOs, sockets and devices in a paused Work's WORKDIR volume on Docker.

A split-WORKDIR Judge reads Work's paused WORKDIR volume in place, read-only,
yet can still write into a FIFO, connect to a socket or open a device there:
read-only stops changes to the files, not data passed through them. If Work
holds the other end, it reads what the Judge's code sent once it resumes,
such as the hidden tests. So before such a round, while Work is paused, the
Harness looks for these files (DockerWorkdirVolumeBackend.special_files) and
refuses the round, not counted, while any exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath

# The most paths a refusal lists; the walk stops once it has found these.
SPECIAL_FILE_LIMIT = 5


def special_files_refusal(
    found: Sequence[tuple[PurePosixPath, str]], workdir: PurePosixPath
) -> str:
    """The rsi-submit message for a round refused over ``found``."""
    listed = ", ".join(f"{workdir / path} ({kind})" for path, kind in found)
    more = " and maybe more" if len(found) >= SPECIAL_FILE_LIMIT else ""
    return (
        "submission refused and not counted: the Judge reads WORKDIR in place, "
        f"so it must hold no FIFO, socket or device: {listed}{more}. "
        "Remove them, then submit again."
    )
