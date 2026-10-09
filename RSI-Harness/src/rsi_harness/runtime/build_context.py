"""Build contexts: tar validation, Dockerfile directives, builder input (spec 4 B6).

A build context exists only as a caller-uploaded stage. It is re-validated
here as a stream (regular files, directories and symlinks with relative
names free of ``..``; never a hardlink, device or FIFO) and rewritten into
the one tar the broker puts into the builder's state volume::

    rsi-ctx/<job>/ctx/...        the context, exactly as uploaded
    rsi-ctx/<job>/df/Dockerfile  the Dockerfile (from the context or inline)

BuildKit then reads only ``--local context=<ctx>`` and ``--local
dockerfile=<df>``; symlinks stay symlinks and are resolved only inside the
builder. A syntax directive (``# syntax=``, also after a ``#!`` line, ``//
syntax=`` or a JSON ``{"syntax": ...}`` file: every form BuildKit's
DetectSyntax honours) selects a frontend image that runs inside the
builder, so it must name an operator-approved repository.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import IO

from rsi_harness.runtime.sandbox_archive import (
    MAX_ARCHIVE_ENTRIES,
    CanonicalTarWriter,
    entry_parts,
    read_tar,
)
from rsi_harness.runtime.sandbox_contracts import SandboxError
from rsi_harness.runtime.sandbox_images import pull_reference

MIB = 1024**2
MAX_DOCKERFILE_BYTES = MIB
DEFAULT_DOCKERFILE = "Dockerfile"
# Relative to the builder's state volume (/var/lib/buildkit).
BUILD_INPUT_ROOT = "rsi-ctx"
# BuildKit: ``^([a-zA-Z][a-zA-Z0-9]*)\s*=\s*(.+?)\s*$`` after the comment
# prefix and any leading space; ``\s`` is Unicode here, a superset.
_DIRECTIVE = re.compile(r"([A-Za-z][A-Za-z0-9]*)\s*=\s*(.+?)\s*")
_BOM = b"\xef\xbb\xbf"
_CHUNK = MIB


@dataclass(frozen=True, slots=True)
class BuildInput:
    """What one staged context became: its builder directory and identity.

    ``digest`` covers the context content and the Dockerfile; the broker
    adds the build options to it for per-session deduplication (B9).
    """

    directory: str
    dockerfile: bytes
    syntax: str | None
    entries: int
    bytes: int
    digest: str


def _directive_body(dockerfile: bytes) -> bytes:
    """What BuildKit's DetectSyntax scans: without a UTF-8 BOM, and without
    a first line starting with ``#!`` (a shebang)."""
    data = dockerfile.removeprefix(_BOM)
    first, _, rest = data.partition(b"\n")
    return rest if first.startswith(b"#!") else data


def parser_directives(dockerfile: bytes, *, comment: bytes = b"#") -> dict[str, str]:
    """Dockerfile parser directives, as BuildKit's parser reads them.

    Directives are ``<comment> key=value`` lines at the very top of the file
    (after an optional UTF-8 BOM and a ``#!`` shebang line); the first line
    that is not one (a blank line, a comment, an instruction) ends them. The
    comment prefix is ``#``, or ``//`` for the form BuildKit also accepts
    for ``syntax``. Keys are case-insensitive; a key given twice is refused.

    This reads at least every directive BuildKit v0.27 does: whitespace is
    any Unicode space (BuildKit: any after the prefix, ASCII elsewhere) and
    an unknown key does not end the directives (it does in BuildKit).
    """
    found: dict[str, str] = {}
    for raw in _directive_body(dockerfile).split(b"\n"):
        line = raw.removesuffix(b"\r")
        if not line.startswith(comment):
            break
        text = line[len(comment) :].decode("utf-8", "replace").lstrip()
        match = _DIRECTIVE.fullmatch(text)
        if match is None:
            break
        key = match.group(1).lower()
        if key in found:
            raise SandboxError(
                "invalid", "dockerfile", f"parser directive {key} is given twice"
            )
        found[key] = match.group(2)
    return found


class _Pairs(list):
    """A JSON object as its (key, value) pairs, duplicates kept."""


def _json_syntax(body: bytes) -> list[str]:
    """``syntax`` values of a whole-file JSON object (BuildKit's third form).

    A Dockerfile never starts with ``{``, so anything that does and is not
    plainly parsed here is refused rather than guessed at.
    """
    if not body.lstrip(b" \t\r\n").startswith(b"{"):
        return []
    try:
        value = json.loads(
            body.decode("utf-8", "surrogateescape"),
            object_pairs_hook=_Pairs,
            strict=False,
        )
    except ValueError:
        return []
    except RecursionError:
        raise SandboxError(
            "invalid",
            "dockerfile",
            "the Dockerfile is a JSON document too deep to read",
        ) from None
    if type(value) is not _Pairs:
        return []
    return [item for key, item in value if key == "syntax" and type(item) is str]


def syntax_directives(dockerfile: bytes) -> list[str]:
    """Every frontend reference BuildKit's DetectSyntax could select, in its
    order: ``# syntax=``, then ``// syntax=``, then a JSON ``{"syntax": ...}``
    file. BuildKit forwards the build to the first one found."""
    found = [
        directives["syntax"]
        for comment in (b"#", b"//")
        if "syntax" in (directives := parser_directives(dockerfile, comment=comment))
    ]
    return found + _json_syntax(_directive_body(dockerfile))


def _repository(value: str) -> str:
    return pull_reference(value)[0]


def syntax_frontend(dockerfile: bytes, allowed: Sequence[str]) -> str | None:
    """The syntax directive's frontend reference, refused unless its
    repository is one of the operator's ``syntax_frontends``.

    Every form BuildKit honours is checked (``# syntax=`` after an optional
    shebang, ``// syntax=``, a JSON ``syntax`` key), not only the one it
    would pick, so no form can smuggle an unapproved frontend (B6).
    """
    values = syntax_directives(dockerfile)
    approved = {_repository(item) for item in allowed}
    for value in values:
        try:
            repository = _repository(value)
        except SandboxError:
            raise SandboxError(
                "invalid",
                "dockerfile",
                "the syntax directive is not an image reference",
            ) from None
        if repository not in approved:
            raise SandboxError(
                "permission",
                "dockerfile",
                f"syntax frontend {repository} is not approved by the operator",
            )
    return values[0] if values else None


def _dockerfile_parts(dockerfile: str | None) -> tuple[str, ...]:
    name = DEFAULT_DOCKERFILE if dockerfile is None else dockerfile
    if type(name) is not str:
        raise SandboxError("invalid", "dockerfile", "expected a relative path")
    try:
        parts = entry_parts(name)
    except ValueError as error:
        raise SandboxError("invalid", "dockerfile", str(error)) from None
    if not parts:
        raise SandboxError("invalid", "dockerfile", "expected a file in the context")
    return parts


def _inline(value: object) -> bytes:
    if type(value) is not str or "\x00" in value:
        raise SandboxError("invalid", "dockerfile_inline", "expected text")
    data = value.encode("utf-8", "surrogatepass")
    if len(data) > MAX_DOCKERFILE_BYTES:
        raise SandboxError("quota", "dockerfile_inline", "the Dockerfile exceeds 1 MiB")
    return data


def prepare_build_input(
    stage: IO[bytes],
    dest: IO[bytes],
    *,
    job_id: str,
    dockerfile: str | None,
    dockerfile_inline: str | None,
    max_bytes: int,
    syntax_frontends: Sequence[str],
    max_entries: int = MAX_ARCHIVE_ENTRIES,
) -> BuildInput:
    """Validate a staged context and write the builder input tar to ``dest``.

    ``dest`` must be a real, seekable file (CanonicalTarWriter). The
    Dockerfile is ``dockerfile_inline`` when given, else the regular file
    ``dockerfile`` (default ``Dockerfile``) of the context; it is at most
    1 MiB. Context content beyond ``max_bytes`` is a quota error.
    """
    if dockerfile is not None and dockerfile_inline is not None:
        raise SandboxError(
            "invalid", "dockerfile", "give dockerfile or dockerfile_inline, not both"
        )
    inline = None if dockerfile_inline is None else _inline(dockerfile_inline)
    wanted = None if inline is not None else _dockerfile_parts(dockerfile)
    directory = f"{BUILD_INPUT_ROOT}/{job_id}"
    root = (BUILD_INPUT_ROOT, job_id)
    # Four prefix directories and the Dockerfile come on top of the
    # context's own entries and bytes.
    writer = CanonicalTarWriter(
        dest,
        max_bytes=max_bytes + MAX_DOCKERFILE_BYTES,
        max_entries=max_entries + 5,
    )
    for parts in (root[:1], root, (*root, "ctx"), (*root, "df")):
        writer.add_dir(parts, mode=0o755, mtime=0)
    content = hashlib.sha256()
    found: bytes | None = None
    try:
        with read_tar(stage) as archive:
            for member in archive:
                parts = entry_parts(member.name)
                if not parts:
                    if not member.isdir():
                        raise ValueError("only a directory can be the context root")
                    continue
                target = (*root, "ctx", *parts)
                content.update(repr((parts, member.type, member.mode)).encode())
                if member.isdir():
                    writer.add_dir(target, mode=member.mode, mtime=member.mtime)
                elif member.issym():
                    content.update(member.linkname.encode("utf-8", "surrogateescape"))
                    writer.add_symlink(
                        target, member.linkname, mode=member.mode, mtime=member.mtime
                    )
                elif member.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    reader = archive.extractfile(member)
                    assert reader is not None
                    keep = parts == wanted
                    if keep and member.size > MAX_DOCKERFILE_BYTES:
                        raise SandboxError(
                            "quota", "dockerfile", "the Dockerfile exceeds 1 MiB"
                        )
                    tee = _Tee(reader, content, keep=keep)
                    writer.add_file(
                        target, tee, member.size, mode=member.mode, mtime=member.mtime
                    )
                    if keep:
                        found = tee.kept()
                else:
                    raise ValueError(
                        "a build context holds only regular files, directories "
                        "and symlinks (no hardlinks, devices or FIFOs)"
                    )
    except SandboxError as error:
        if error.field in ("bytes", "entries"):
            raise SandboxError(
                "quota", "stage_id", f"the build context is too large: {error.message}"
            ) from None
        raise
    except (tarfile.TarError, UnicodeError, ValueError, OSError, EOFError) as error:
        raise SandboxError(
            "invalid", "stage_id", f"invalid build context: {error}"
        ) from None
    if inline is not None:
        found = inline
    elif found is None:
        raise SandboxError(
            "invalid",
            "dockerfile",
            f"{'/'.join(wanted or ())} is not a regular file of the context",
        )
    syntax = syntax_frontend(found, syntax_frontends)
    writer.add_file(
        (*root, "df", DEFAULT_DOCKERFILE),
        io.BytesIO(found),
        len(found),
        mode=0o644,
        mtime=0,
    )
    summary = writer.close()
    if summary.bytes - len(found) > max_bytes:
        raise SandboxError(
            "quota", "stage_id", f"the build context exceeds {max_bytes} bytes"
        )
    content.update(b"\0dockerfile\0" + found)
    return BuildInput(
        directory=directory,
        dockerfile=found,
        syntax=syntax,
        entries=summary.entries - 5,
        bytes=summary.bytes - len(found),
        digest=content.hexdigest(),
    )


class _Tee(io.RawIOBase):
    """Read-through that hashes a member and optionally keeps it (the
    Dockerfile, at most 1 MiB)."""

    def __init__(self, reader: IO[bytes], digest, *, keep: bool) -> None:
        self._reader = reader
        self._digest = digest
        self._keep = bytearray() if keep else None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        data = self._reader.read(min(len(buffer), _CHUNK))
        buffer[: len(data)] = data
        self._digest.update(data)
        if self._keep is not None:
            self._keep.extend(data)
        return len(data)

    def kept(self) -> bytes:
        return bytes(self._keep or b"")


def build_fingerprint(input_digest: str, options: Mapping[str, object]) -> str:
    """Per-session dedupe key (B9): context, Dockerfile and build options."""
    canonical = repr(sorted(options.items())).encode()
    return hashlib.sha256(input_digest.encode() + b"\0" + canonical).hexdigest()


__all__ = [
    "BUILD_INPUT_ROOT",
    "DEFAULT_DOCKERFILE",
    "MAX_DOCKERFILE_BYTES",
    "BuildInput",
    "build_fingerprint",
    "parser_directives",
    "prepare_build_input",
    "syntax_directives",
    "syntax_frontend",
]
