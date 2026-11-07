"""Native artifact bytes on Modal Volumes, which do not support hard links."""

import os
import shutil
import stat
import tempfile
from pathlib import Path

from rsi_harness.runtime.artifacts import RunArtifactWriter


class ModalArtifactWriter(RunArtifactWriter):
    def _hardlink_output(self, source: Path, target: Path) -> None:
        # Preserve the native names and bytes. Publish only a complete fsynced
        # copy, after the Judge and its output capture have both terminated.
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        temporary = None
        try:
            with os.fdopen(descriptor, "rb") as reader:
                if not stat.S_ISREG(os.fstat(reader.fileno()).st_mode):
                    raise ValueError("complete verifier output must be regular")
                fd, name = tempfile.mkstemp(
                    prefix=f".{target.name}.", dir=target.parent
                )
                temporary = Path(name)
                with os.fdopen(fd, "wb") as writer:
                    shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    writer.flush()
                    os.fchmod(writer.fileno(), 0o644)
                    os.fsync(writer.fileno())
            os.replace(temporary, target)
            self._fsync_directory(target.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
