"""Build contexts: stream validation, the builder input tar, directives (B6)."""

import io
import tarfile

import pytest

from rsi_harness.runtime.build_context import (
    parser_directives,
    prepare_build_input,
    syntax_directives,
    syntax_frontend,
)
from rsi_harness.runtime.sandbox_contracts import SandboxError

MIB = 1024**2
JOB = "0" * 32
FRONTENDS = ("docker.io/docker/dockerfile",)


def context(*entries):
    """(name, kind, data_or_link) entries as a tar stream."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, kind, value in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.mode = 0o755 if kind == tarfile.DIRTYPE else 0o644
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                info.linkname = value
                tar.addfile(info)
            elif kind == tarfile.REGTYPE:
                info.size = len(value)
                tar.addfile(info, io.BytesIO(value))
            else:
                tar.addfile(info)
    buffer.seek(0)
    return buffer


def prepare(stage, tmp_path, **options):
    values = dict(
        job_id=JOB,
        dockerfile=None,
        dockerfile_inline=None,
        max_bytes=MIB,
        syntax_frontends=FRONTENDS,
    )
    values.update(options)
    target = tmp_path / "input.tar"
    with open(target, "w+b") as dest:
        result = prepare_build_input(stage, dest, **values)
    with tarfile.open(target) as tar:
        found = {
            member.name: (
                tar.extractfile(member).read()
                if member.isreg()
                else member.linkname or member.type
            )
            for member in tar.getmembers()
        }
    return result, found


DOCKERFILE = b"FROM alpine:3.21\nCOPY . /src\n"


def test_the_input_holds_the_context_and_the_dockerfile_apart(tmp_path):
    stage = context(
        ("Dockerfile", tarfile.REGTYPE, DOCKERFILE),
        ("app", tarfile.DIRTYPE, None),
        ("app/main.py", tarfile.REGTYPE, b"print(1)\n"),
        ("app/link", tarfile.SYMTYPE, "/etc/passwd"),
    )
    result, found = prepare(stage, tmp_path)
    root = f"rsi-ctx/{JOB}"
    assert result.directory == root
    assert result.dockerfile == DOCKERFILE
    assert (result.entries, result.bytes) == (4, len(DOCKERFILE) + 9)
    # Symlinks stay symlinks; only the builder ever resolves them.
    assert found[f"{root}/ctx/app/link"] == "/etc/passwd"
    assert found[f"{root}/ctx/app/main.py"] == b"print(1)\n"
    assert found[f"{root}/df/Dockerfile"] == DOCKERFILE
    assert set(found) == {
        "rsi-ctx",
        root,
        f"{root}/ctx",
        f"{root}/df",
        f"{root}/ctx/Dockerfile",
        f"{root}/ctx/app",
        f"{root}/ctx/app/main.py",
        f"{root}/ctx/app/link",
        f"{root}/df/Dockerfile",
    }


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        (("hard", tarfile.LNKTYPE, "Dockerfile"), "hardlinks"),
        (("dev", tarfile.CHRTYPE, None), "devices"),
        (("fifo", tarfile.FIFOTYPE, None), "FIFOs"),
        (("/abs", tarfile.REGTYPE, b"x"), "relative"),
        (("a/../../x", tarfile.REGTYPE, b"x"), r"'\.\.'"),
    ],
)
def test_hardlinks_devices_and_escaping_names_are_refused(tmp_path, entry, match):
    stage = context(("Dockerfile", tarfile.REGTYPE, DOCKERFILE), entry)
    with pytest.raises(SandboxError, match=match) as caught:
        prepare(stage, tmp_path)
    assert (caught.value.code, caught.value.field) == ("invalid", "stage_id")


def test_an_entry_below_a_symlink_is_refused(tmp_path):
    stage = context(
        ("Dockerfile", tarfile.REGTYPE, DOCKERFILE),
        ("out", tarfile.SYMTYPE, "/"),
        ("out/etc/cron.d/x", tarfile.REGTYPE, b"x"),
    )
    with pytest.raises(SandboxError, match="below a file or symlink"):
        prepare(stage, tmp_path)


def test_a_named_dockerfile_or_an_inline_one(tmp_path):
    stage = context(
        ("docker", tarfile.DIRTYPE, None),
        ("docker/Build.file", tarfile.REGTYPE, b"FROM busybox\n"),
    )
    result, found = prepare(stage, tmp_path, dockerfile="docker/Build.file")
    assert found[f"rsi-ctx/{JOB}/df/Dockerfile"] == b"FROM busybox\n"
    inline, found = prepare(
        context(), tmp_path, dockerfile_inline="FROM scratch\nCOPY x /\n"
    )
    assert found[f"rsi-ctx/{JOB}/df/Dockerfile"] == b"FROM scratch\nCOPY x /\n"
    assert inline.entries == 0
    with pytest.raises(SandboxError, match="not both"):
        prepare(context(), tmp_path, dockerfile="Dockerfile", dockerfile_inline="x")


@pytest.mark.parametrize(
    ("entries", "dockerfile", "code"),
    [
        ((), None, "invalid"),
        ((("Dockerfile", tarfile.SYMTYPE, "/etc/passwd"),), None, "invalid"),
        ((("Dockerfile", tarfile.DIRTYPE, None),), None, "invalid"),
        ((("Dockerfile", tarfile.REGTYPE, b"x" * (MIB + 1)),), None, "quota"),
        ((("Dockerfile", tarfile.REGTYPE, DOCKERFILE),), "../Dockerfile", "invalid"),
    ],
)
def test_the_dockerfile_must_be_a_bounded_regular_file_of_the_context(
    tmp_path, entries, dockerfile, code
):
    with pytest.raises(SandboxError) as caught:
        prepare(context(*entries), tmp_path, dockerfile=dockerfile, max_bytes=4 * MIB)
    assert caught.value.code == code


def test_a_context_over_max_bytes_is_a_quota_error(tmp_path):
    stage = context(
        ("Dockerfile", tarfile.REGTYPE, DOCKERFILE),
        ("big", tarfile.REGTYPE, b"x" * (2 * MIB)),
    )
    with pytest.raises(SandboxError) as caught:
        prepare(stage, tmp_path, max_bytes=MIB)
    assert (caught.value.code, caught.value.field) == ("quota", "stage_id")


def test_the_digest_follows_content_not_mtimes(tmp_path):
    def staged(data, mtime):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            for name, value in (("Dockerfile", DOCKERFILE), ("a.txt", data)):
                info = tarfile.TarInfo(name)
                info.size, info.mtime = len(value), mtime
                tar.addfile(info, io.BytesIO(value))
        buffer.seek(0)
        return buffer

    first, _ = prepare(staged(b"one", 1000), tmp_path)
    touched, _ = prepare(staged(b"one", 2000), tmp_path)
    changed, _ = prepare(staged(b"two", 1000), tmp_path)
    assert first.digest == touched.digest != changed.digest


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (b"# syntax=docker/dockerfile:1\nFROM x\n", {"syntax": "docker/dockerfile:1"}),
        (b"#syntax=docker/dockerfile:1.7\n", {"syntax": "docker/dockerfile:1.7"}),
        (b"# SYNTAX = docker/dockerfile:1 \n", {"syntax": "docker/dockerfile:1"}),
        (b"\xef\xbb\xbf# syntax=a/b\n", {"syntax": "a/b"}),
        (b"# escape=`\n# syntax=a/b\r\nFROM x\n", {"escape": "`", "syntax": "a/b"}),
        # A directive after any other line is only a comment.
        (b"FROM x\n# syntax=a/b\n", {}),
        (b"# a comment\n# syntax=a/b\n", {}),
        (b"\n# syntax=a/b\n", {}),
        (b"# syntax=\n", {}),
        # BuildKit skips a first #! line, then reads directives.
        (b"#!/bin/sh\n# syntax=a/b\n", {"syntax": "a/b"}),
        (b"\xef\xbb\xbf#!x\r\n#syntax=a/b\r\n", {"syntax": "a/b"}),
        (b"#!/bin/sh", {}),
        # Any Unicode space after the prefix (BuildKit: unicode.IsSpace).
        (b"#\x0bsyntax=a/b\n", {"syntax": "a/b"}),
        (b"#\xc2\xa0syntax=a/b\n", {"syntax": "a/b"}),
        # Wider than BuildKit on purpose: an unknown key does not end them.
        (b"# foo=bar\n# syntax=a/b\n", {"foo": "bar", "syntax": "a/b"}),
    ],
)
def test_parser_directives_follow_buildkit(text, expected):
    assert parser_directives(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (b"# syntax=a/b\nFROM x\n", ["a/b"]),
        (b"// syntax=a/b\nFROM x\n", ["a/b"]),
        (b"//syntax = a/b\r\n", ["a/b"]),
        (b"#!/x\n//syntax=a/b\n", ["a/b"]),
        (b'{"syntax": "a/b"}', ["a/b"]),
        (b' \n{"x": [1, {"syntax": "c/d"}], "syntax": "a/b"}\n', ["a/b"]),
        (b'{"syntax": "a/b", "syntax": "c/d"}', ["a/b", "c/d"]),
        (b"#!/x\n" + b'{"syntax": "a/b"}', ["a/b"]),
        # Not a directive in any form.
        (b"FROM x\n// syntax=a/b\n", []),
        (b"# escape=`\n// syntax=a/b\n", []),
        (b'["syntax", "a/b"]', []),
        (b'{"syntax": 1}', []),
        (b'{"Syntax": "a/b"}', []),
        (b'{"syntax": "a/b"} FROM x', []),
    ],
)
def test_syntax_directives_cover_every_form_buildkit_detects(text, expected):
    assert syntax_directives(text) == expected


def test_a_repeated_directive_is_refused():
    with pytest.raises(SandboxError, match="twice"):
        parser_directives(b"# syntax=a/b\n# syntax=c/d\n")


@pytest.mark.parametrize(
    "reference",
    [
        "docker/dockerfile:1",
        "docker/dockerfile:1-labs",
        "docker.io/docker/dockerfile:1.7",
        "docker/dockerfile@sha256:" + "a" * 64,
    ],
)
def test_approved_frontends_are_accepted(reference):
    text = f"# syntax={reference}\nFROM x\n".encode()
    assert syntax_frontend(text, FRONTENDS) == reference


@pytest.mark.parametrize(
    "reference", ["ghcr.io/evil/frontend:1", "evil/dockerfile:1", "docker/dockerfile2"]
)
def test_unapproved_frontends_are_refused(tmp_path, reference):
    text = f"# syntax={reference}\nFROM x\n".encode()
    with pytest.raises(SandboxError) as caught:
        syntax_frontend(text, FRONTENDS)
    assert (caught.value.code, caught.value.field) == ("permission", "dockerfile")
    stage = context(("Dockerfile", tarfile.REGTYPE, text))
    with pytest.raises(SandboxError, match="not approved"):
        prepare(stage, tmp_path)
    # No approved frontend at all: any # syntax= is refused.
    with pytest.raises(SandboxError, match="not approved"):
        syntax_frontend(b"# syntax=docker/dockerfile:1\n", ())


@pytest.mark.parametrize(
    "text",
    [
        b"#!/bin/sh\n# syntax=evil/fe:1\nFROM x\n",
        b"// syntax=evil/fe:1\nFROM x\n",
        b"#!/bin/sh\n//syntax=evil/fe:1\n",
        b'{"syntax": "evil/fe:1"}',
        b'\xef\xbb\xbf\n {"syntax": "evil/fe:1", "x": null}\n',
        b'{"syntax": "docker/dockerfile:1", "syntax": "evil/fe:1"}',
        b"#\x0bsyntax=evil/fe:1\n",
        b"#\xc2\xa0syntax=evil/fe:1\n",
        b"# foo=bar\n# syntax=evil/fe:1\n",
    ],
)
def test_every_syntax_form_needs_an_approved_frontend(tmp_path, text):
    """BuildKit's DetectSyntax also honours a shebang-prefixed, a ``//``
    and a JSON directive: none of them reaches an unapproved frontend."""
    with pytest.raises(SandboxError, match="evil/fe is not approved") as caught:
        syntax_frontend(text, FRONTENDS)
    assert (caught.value.code, caught.value.field) == ("permission", "dockerfile")
    with pytest.raises(SandboxError, match="not approved"):
        prepare(context(("Dockerfile", tarfile.REGTYPE, text)), tmp_path)
    with pytest.raises(SandboxError, match="not approved"):
        prepare(context(), tmp_path, dockerfile_inline=text.decode("utf-8"))


@pytest.mark.parametrize(
    "text",
    [
        b"#!/bin/sh\n# syntax=docker/dockerfile:1\nFROM x\n",
        b"// syntax=docker/dockerfile:1\nFROM x\n",
        b'{"syntax": "docker/dockerfile:1"}',
    ],
)
def test_an_approved_frontend_is_accepted_in_every_form(text):
    assert syntax_frontend(text, FRONTENDS) == "docker/dockerfile:1"


def test_a_value_that_is_no_reference_or_unreadable_json_is_invalid():
    for text in (b"// syntax=not a reference\n", b'{"syntax": "\\udc80"}'):
        with pytest.raises(SandboxError) as caught:
            syntax_frontend(text, FRONTENDS)
        assert (caught.value.code, caught.value.field) == ("invalid", "dockerfile")
    deep = b'{"syntax": "evil/fe:1", "x": ' + b"[" * 100_000 + b"]" * 100_000 + b"}"
    with pytest.raises(SandboxError, match="too deep") as caught:
        syntax_frontend(deep, FRONTENDS)
    assert caught.value.code == "invalid"
