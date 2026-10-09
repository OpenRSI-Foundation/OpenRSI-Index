"""Archive copies: canonical tar, one-shot stages, no tmpfs or paused writes."""

import base64
import hashlib
import io
import json
import os
import posixpath
import stat
import tarfile
import tracemalloc

import pytest
import requests
from docker.errors import APIError

from rsi_harness.runtime import sandbox_archive
from rsi_harness.runtime.sandbox_archive import (
    MAX_EXTENDED_HEADER,
    MAX_STAGE_FRAME,
    PATH_STAT_HEADER,
    ArchiveTarget,
    ArchiveTransfer,
    CanonicalTarWriter,
    StageStore,
    canonicalize_tar,
    decode_path_stat,
    directory_tar,
    path_stat_view,
    read_tar,
    stat_kind,
)
from rsi_harness.runtime.sandbox_contracts import SandboxError
from tests.sandbox_helpers import FakeClock

DIR_MODE = 1 << 31
LINK_MODE = 1 << 27


def tar_bytes(*members):
    """Members as (name, type, payload, extra TarInfo attributes)."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for name, kind, payload, extra in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            for key, value in extra.items():
                setattr(info, key, value)
            if kind == tarfile.REGTYPE:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = payload
                archive.addfile(info)
    return buffer.getvalue()


def reg(name, data=b"x", **extra):
    return (name, tarfile.REGTYPE, data, extra)


def directory(name, **extra):
    return (name, tarfile.DIRTYPE, None, extra)


def symlink(name, target, **extra):
    return (name, tarfile.SYMTYPE, target, extra)


def canonical(raw, *, max_bytes=1 << 20, tmp_path=None):
    source = io.BytesIO(raw)
    dest = io.BytesIO()
    summary = canonicalize_tar(source, dest, max_bytes=max_bytes)
    return summary, dest.getvalue()


def members(data):
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        return [
            (
                member.name,
                member.type,
                oct(member.mode),
                member.uid,
                member.gid,
                member.uname,
                member.linkname,
                archive.extractfile(member).read() if member.isreg() else None,
            )
            for member in archive
        ]


# -- canonical tar ------------------------------------------------------------


def test_canonical_form_normalizes_names_modes_and_owners():
    raw = tar_bytes(
        directory("./", mode=0o700),
        directory("./app", mode=0o4775, uid=1000, gid=1000, uname="me"),
        reg("./app//x.txt", b"hello", mode=0o6755, uid=5),
        symlink("app/link", "/etc/passwd", mode=0o777),
        reg("app/sub/deep", b"deep"),
    )
    summary, data = canonical(raw)

    assert (summary.entries, summary.bytes) == (4, 9)
    assert members(data) == [
        ("app", tarfile.DIRTYPE, "0o775", 0, 0, "", "", None),
        ("app/x.txt", tarfile.REGTYPE, "0o755", 0, 0, "", "", b"hello"),
        ("app/link", tarfile.SYMTYPE, "0o777", 0, 0, "", "/etc/passwd", None),
        ("app/sub/deep", tarfile.REGTYPE, "0o644", 0, 0, "", "", b"deep"),
    ]
    # No extended attributes survive; the only PAX record is the leading path.
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        first, *rest = archive.getmembers()
        assert first.pax_headers == {"path": "app"}
        assert all(member.pax_headers == {} for member in rest)
    # Docker sniffs compression magic in the first block: it is a PAX header.
    assert data[:14] == b"././@PaxHeader"


def test_repeated_directories_are_written_once():
    summary, data = canonical(tar_bytes(directory("a"), directory("a/"), reg("a/f")))
    assert summary.entries == 2
    assert [item[0] for item in members(data)] == ["a", "a/f"]


def fifo(name):
    return (name, tarfile.FIFOTYPE, None, {})


def device(name):
    return (name, tarfile.CHRTYPE, None, {"devmajor": 1, "devminor": 3})


def hardlink(name, target):
    return (name, tarfile.LNKTYPE, target, {})


@pytest.mark.parametrize(
    "raw",
    (
        tar_bytes(reg("f"), hardlink("g", "f")),
        tar_bytes(device("null")),
        tar_bytes(fifo("pipe")),
        tar_bytes(reg("/etc/passwd")),
        tar_bytes(reg("a/../../etc/passwd")),
        tar_bytes(reg("/".join(["d"] * 65))),
        tar_bytes(symlink("link", "")),
        tar_bytes(symlink("link", "x" * 4097)),
        tar_bytes(symlink("link", "/etc"), reg("link/passwd")),
        tar_bytes(reg("f"), reg("f")),
        tar_bytes(reg("f"), directory("f")),
        tar_bytes(reg("f"), reg("f/g")),
        tar_bytes(reg(".")),
        b"not a tar archive" * 64,
    ),
    ids=(
        "hardlink",
        "device",
        "fifo",
        "absolute",
        "dotdot",
        "too-deep",
        "empty-link",
        "long-link",
        "below-symlink",
        "duplicate-file",
        "file-then-dir",
        "below-file",
        "file-root",
        "garbage",
    ),
)
def test_uploads_outside_the_canonical_form_are_invalid(raw):
    with pytest.raises(SandboxError) as caught:
        canonical(raw)
    assert (caught.value.code, caught.value.field) == ("invalid", "stage")


def test_non_utf8_names_are_invalid():
    header = tarfile.TarInfo("placeholder")
    header.size = 0
    block = bytearray(header.tobuf(format=tarfile.USTAR_FORMAT))
    block[:11] = b"bad\xff\xfename"
    block[148:156] = b"        "
    block[148:155] = f"{sum(block):06o}\0".encode()
    raw = bytes(block) + b"\0" * 1024
    with pytest.raises(SandboxError, match="invalid"):
        canonical(raw)


class ClaimedBody(io.RawIOBase):
    """One header block, then as many zero bytes as it claims, made lazily."""

    def __init__(self, header, size):
        self._header = header
        self._remaining = size + 2 * tarfile.BLOCKSIZE

    def readable(self):
        return True

    def readinto(self, buffer):
        if self._header:
            count = min(len(buffer), len(self._header))
            buffer[:count] = self._header[:count]
            self._header = self._header[count:]
            return count
        count = min(len(buffer), self._remaining)
        buffer[:count] = bytes(count)
        self._remaining -= count
        return count


def raw_header(kind, size, name="././@PaxHeader"):
    """A ustar header block of ``kind`` claiming ``size`` body bytes."""
    info = tarfile.TarInfo(name)
    info.type = kind
    block = bytearray(info.tobuf(format=tarfile.USTAR_FORMAT))
    block[124:136] = f"{size:011o}\0".encode()
    block[148:156] = b"        "
    block[148:155] = f"{sum(block):06o}\0".encode()
    return bytes(block)


@pytest.mark.parametrize(
    "kind",
    (tarfile.XHDTYPE, tarfile.XGLTYPE, tarfile.GNUTYPE_LONGNAME),
    ids=("pax", "pax-global", "gnu-longname"),
)
def test_oversized_extended_headers_are_refused_before_buffering(kind):
    claimed = 64 * 1024**2
    source = io.BufferedReader(ClaimedBody(raw_header(kind, claimed), claimed))
    tracemalloc.start()
    try:
        with pytest.raises(SandboxError) as caught:
            canonicalize_tar(source, io.BytesIO(), max_bytes=1 << 40)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert (caught.value.code, caught.value.field) == ("invalid", "stage")
    assert peak < 4 * 1024**2


def pax_record(key, value):
    body = f" {key}={value}\n"
    length = len(body) + 1
    length += len(str(length + len(str(length)))) - 1
    return f"{length}{body}".encode()


def test_extended_header_chains_globals_and_sparse_members_are_refused():
    def extended(kind, records):
        payload = b"".join(pax_record(key, value) for key, value in records)
        padding = -len(payload) % tarfile.BLOCKSIZE
        return raw_header(kind, len(payload)) + payload + bytes(padding)

    member = tar_bytes(reg("f"))
    small = extended(tarfile.XHDTYPE, [("comment", "x")])
    assert canonical(small * 8 + member)[0].entries == 1
    global_half = extended(
        tarfile.XGLTYPE, [("comment", "g" * (MAX_EXTENDED_HEADER // 2))]
    )
    for raw in (
        small * 9 + member,
        # Globals accumulate for the whole archive: one 64 KiB budget.
        # (the first 1024 bytes of a GNU tar of one short file: header, data)
        global_half + tar_bytes(reg("a"))[:1024] + global_half + member,
        extended(tarfile.XHDTYPE, [("GNU.sparse.map", "0,1")]) + member,
        extended(
            tarfile.XHDTYPE,
            [("GNU.sparse.size", "1"), ("GNU.sparse.offset", "0")],
        )
        + member,
        extended(
            tarfile.XHDTYPE, [("GNU.sparse.major", "1"), ("GNU.sparse.minor", "0")]
        )
        + member,
        raw_header(tarfile.GNUTYPE_SPARSE, 0, name="sparse") + bytes(1024),
    ):
        with pytest.raises(SandboxError) as caught:
            canonical(raw)
        assert (caught.value.code, caught.value.field) == ("invalid", "stage")


def test_tar_reading_keeps_no_member_objects():
    raw = tar_bytes(*(reg(f"d/{index}") for index in range(50)))
    with read_tar(io.BytesIO(raw)) as archive:
        for member in archive:
            assert archive.members == []


def test_canonical_writer_keeps_no_member_objects(tmp_path):
    with open(tmp_path / "out.tar", "w+b") as out:
        tracemalloc.start()
        try:
            writer = CanonicalTarWriter(out, max_bytes=0)
            name = "n" * 1000
            for index in range(5_000):
                writer.add_dir((f"{name}-{index}",), mode=0o755, mtime=0)
            assert writer._archive.members == []
            current = tracemalloc.get_traced_memory()[0]
        finally:
            tracemalloc.stop()
    # Fixed-size keys only, not 5000 TarInfos with 1 KB names (> 10 MB).
    assert current < 4 * 1024**2


def test_content_and_entry_bounds_are_quotas():
    with pytest.raises(SandboxError) as caught:
        canonical(tar_bytes(reg("a", b"x" * 10), reg("b", b"y" * 10)), max_bytes=15)
    assert caught.value.code == "quota"
    writer = CanonicalTarWriter(io.BytesIO(), max_bytes=0, max_entries=2)
    writer.add_dir(("a",), mode=0o755, mtime=0)
    writer.add_dir(("b",), mode=0o755, mtime=0)
    with pytest.raises(SandboxError, match="exceeds 2 entries"):
        writer.add_dir(("c",), mode=0o755, mtime=0)


def test_directory_tar_creates_root_owned_0755_ancestors():
    assert [item[:5] for item in members(directory_tar(("a", "b", "c")))] == [
        ("a", tarfile.DIRTYPE, "0o755", 0, 0),
        ("a/b", tarfile.DIRTYPE, "0o755", 0, 0),
        ("a/b/c", tarfile.DIRTYPE, "0o755", 0, 0),
    ]


# -- path-stat ----------------------------------------------------------------


def encoded(value):
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    return base64.b64encode(raw).decode()


def pathstat(name="etc", mode=DIR_MODE | 0o755, link="", size=4096):
    return {
        "name": name,
        "size": size,
        "mode": mode,
        "mtime": "2026-09-30T00:00:00Z",
        "linkTarget": link,
    }


def test_path_stat_decodes_the_docker_header_strictly():
    assert decode_path_stat(encoded(pathstat())) == pathstat()
    assert stat_kind(pathstat()) == "dir"
    assert stat_kind(pathstat(mode=0o644)) == "file"
    assert stat_kind(pathstat(mode=LINK_MODE | 0o777, link="/usr/bin")) == "symlink"
    assert stat_kind(pathstat(mode=(1 << 25) | 0o644)) == "other"
    view = path_stat_view(pathstat(mode=(1 << 23) | (1 << 20) | 0o755))
    assert view == {
        "exists": True,
        "kind": "file",
        "size": 4096,
        "mode": stat.S_ISUID | stat.S_ISVTX | 0o755,
        "mtime": "2026-09-30T00:00:00Z",
        "link_target": None,
    }
    assert path_stat_view(None)["exists"] is False


@pytest.mark.parametrize(
    "value",
    (
        None,
        "",
        "!!!not-base64",
        encoded(b"[1, 2]"),
        encoded({**pathstat(), "extra": 1}),
        encoded({key: value for key, value in pathstat().items() if key != "mode"}),
        encoded({**pathstat(), "size": -1}),
        encoded({**pathstat(), "mode": "493"}),
        encoded({**pathstat(), "name": "a\x00b"}),
        encoded(
            b'{"name":"a","name":"b","size":1,"mode":1,"mtime":"","linkTarget":""}'
        ),
        encoded(b'{"name":"a","size":NaN,"mode":1,"mtime":"","linkTarget":""}'),
        "A" * 16385,
    ),
)
def test_malformed_path_stat_is_an_infrastructure_error(value):
    with pytest.raises(SandboxError) as caught:
        decode_path_stat(value)
    assert caught.value.code == "infrastructure"


# -- stages -------------------------------------------------------------------


@pytest.fixture
def stages(tmp_path):
    clock = FakeClock()
    store = StageStore(tmp_path / "spool", clock=clock, ttl_sec=600)
    store.clock = clock
    yield store
    store.close()


def upload(stages, raw, *, frame=None, max_bytes=1 << 30):
    frame = frame or len(raw) or 1
    stage_id, offset = None, 0
    while True:
        chunk = raw[offset : offset + frame]
        final = offset + len(chunk) >= len(raw)
        result = stages.put(
            stage_id,
            offset,
            chunk,
            final=final,
            sha256=hashlib.sha256(raw).hexdigest() if final else None,
            max_bytes=max_bytes,
        )
        stage_id, offset = result["stage_id"], offset + len(chunk)
        if final:
            return result


def test_stage_spool_is_private_and_frames_are_sequential(stages, tmp_path):
    raw = tar_bytes(reg("a", b"A" * 3000), reg("b", b"B" * 3000))
    first = stages.put(None, 0, raw[:4096], final=False, sha256=None, max_bytes=1 << 20)
    assert first["entries"] is None and first["stage_id"].startswith("s")
    spool = tmp_path / "spool"
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
    (part,) = spool.iterdir()
    assert stat.S_IMODE(part.stat().st_mode) == 0o600
    with pytest.raises(SandboxError, match="expected offset 4096"):
        stages.put(
            first["stage_id"],
            0,
            raw[4096:],
            final=False,
            sha256=None,
            max_bytes=1 << 20,
        )
    final = stages.put(
        first["stage_id"],
        4096,
        raw[4096:],
        final=True,
        sha256=hashlib.sha256(raw).hexdigest(),
        max_bytes=1 << 20,
    )
    assert (final["bytes"], final["entries"]) == (len(raw), 2)
    assert [path.suffix for path in spool.iterdir()] == [".tar"]
    with pytest.raises(SandboxError, match="already final"):
        stages.put(
            first["stage_id"],
            len(raw),
            b"",
            final=False,
            sha256=None,
            max_bytes=1 << 20,
        )


def test_hash_mismatch_or_invalid_tar_discards_the_stage(stages, tmp_path):
    raw = tar_bytes(reg("a"))
    with pytest.raises(SandboxError, match="hash differs"):
        stages.put(None, 0, raw, final=True, sha256="0" * 64, max_bytes=1 << 20)
    bad = tar_bytes(fifo("p"))
    with pytest.raises(SandboxError, match="invalid tar"):
        upload(stages, bad)
    assert list((tmp_path / "spool").iterdir()) == []


def test_stage_frames_and_budget_are_bounded(stages, tmp_path):
    with pytest.raises(SandboxError) as caught:
        stages.put(
            None,
            0,
            b"x" * (MAX_STAGE_FRAME + 1),
            final=False,
            sha256=None,
            max_bytes=1 << 30,
        )
    assert caught.value.code == "quota"
    with pytest.raises(SandboxError) as caught:
        stages.put(None, 0, b"x" * 10, final=False, sha256=None, max_bytes=9)
    assert (caught.value.code, caught.value.field) == ("quota", "stage")
    first = stages.put(None, 0, b"x" * 8, final=False, sha256=None, max_bytes=10)
    with pytest.raises(SandboxError, match="upload budget"):
        stages.put(
            first["stage_id"], 8, b"x" * 8, final=False, sha256=None, max_bytes=10
        )
    for bad in (
        dict(stage_id="s" + "0" * 32, offset=0, data=b"", final=False, sha256=None),
        dict(stage_id="bad", offset=0, data=b"", final=False, sha256=None),
        dict(stage_id=None, offset=4, data=b"", final=False, sha256=None),
        dict(stage_id=None, offset=0, data=b"", final=True, sha256=None),
        dict(stage_id=None, offset=0, data=b"", final=False, sha256="0" * 64),
        dict(stage_id=None, offset=0, data="text", final=False, sha256=None),
    ):
        stage_id = bad.pop("stage_id")
        with pytest.raises(SandboxError) as caught:
            stages.put(stage_id, max_bytes=100, **bad)
        assert caught.value.code == "invalid"


def test_stage_is_consumed_once_and_read_in_bounded_slices(stages):
    raw = tar_bytes(reg("a", b"payload"))
    stage_id = upload(stages, raw, frame=512)["stage_id"]
    size = stages.size(stage_id)
    data = stages.read(stage_id, 0, 100) + stages.read(stage_id, 100, size)
    assert members(data)[0][-1] == b"payload"
    with pytest.raises(SandboxError, match="16 MiB"):
        stages.read(stage_id, 0, MAX_STAGE_FRAME + 1)
    with stages.consume(stage_id) as (file, summary):
        assert (summary.entries, summary.bytes) == (1, 7)
        assert file.read() == data
    with pytest.raises(SandboxError, match="unknown or expired"):
        stages.consume(stage_id).__enter__()


def test_unused_stages_expire(stages, tmp_path):
    kept = upload(stages, tar_bytes(reg("a")))["stage_id"]
    stages.clock.now += 300
    stale = upload(stages, tar_bytes(reg("b")))["stage_id"]
    stages.clock.now += 300
    stages.read(stale, 0, 10)
    assert stages.sweep() == (kept,)
    stages.clock.now += 600
    assert stages.sweep() == (stale,)
    assert list((tmp_path / "spool").iterdir()) == []


def test_failed_result_stage_leaves_nothing(stages, tmp_path):
    with pytest.raises(RuntimeError):
        with stages.create_result() as result:
            result.file.write(b"partial")
            raise RuntimeError("copy failed")
    assert list((tmp_path / "spool").iterdir()) == []


# -- archive transfer ---------------------------------------------------------


class Response:
    def __init__(self, status, *, header=None, body=b""):
        self.status_code = status
        self.headers = {} if header is None else {PATH_STAT_HEADER: header}
        self.raw = io.BytesIO(body)
        self.closed = False

    def close(self):
        self.closed = True


class FakeArchiveAPI:
    """A container filesystem behind HEAD/GET/PUT /containers/{id}/archive."""

    timeout = 5

    def __init__(self):
        self.files = {"/": ("dir", None, 0o755)}
        self.puts = []
        self.gets = []
        self.heads = []

    def add(self, path, kind="file", data=b"", mode=0o644):
        for depth in range(1, path.count("/")):
            parent = "/" + "/".join(path.strip("/").split("/")[:depth])
            self.files.setdefault(parent, ("dir", None, 0o755))
        self.files[path] = (kind, data, mode)

    def resolve(self, path, *, final=True):
        """Evaluate symlinks in container scope, like FollowSymlinkInScope."""
        current = "/"
        parts = [part for part in path.split("/") if part]
        for index, part in enumerate(parts):
            candidate = posixpath.join(current, part)
            entry = self.files.get(candidate)
            last = index == len(parts) - 1
            if entry and entry[0] == "symlink" and (final or not last):
                candidate = self.resolve(entry[1])
            current = candidate
        return current

    def _url(self, template, container):
        return template.format(container)

    def _set_request_timeout(self, kwargs):
        kwargs.setdefault("timeout", self.timeout)
        return kwargs

    def _stat(self, path):
        resolved = self.resolve(path, final=False)
        entry = self.files.get(resolved)
        if entry is None:
            return None
        kind, data, mode = entry
        bits = {"dir": DIR_MODE, "file": 0, "symlink": LINK_MODE}[kind]
        return pathstat(
            name=posixpath.basename(path) or "/",
            mode=bits | mode,
            link=self.resolve(path) if kind == "symlink" else "",
            size=len(data) if kind == "file" else 0,
        )

    def head(self, url, params, timeout):
        self.heads.append(params["path"])
        found = self._stat(params["path"])
        if found is None:
            return Response(404)
        return Response(200, header=encoded(found))

    def _raise_for_status(self, response):
        if response.status_code >= 400:
            http = requests.Response()
            http.status_code = response.status_code
            raise APIError("archive error", response=http, explanation="archive error")

    def _get(self, url, params, headers, stream):
        # The header is defensive: decode_content also undoes a gzip body.
        assert headers == {"Accept-Encoding": "identity"} and stream is True
        self.gets.append(params["path"])
        path = params["path"]
        found = self._stat(path)
        if found is None:
            return Response(404)
        root = self.resolve(path, final=False)
        base = posixpath.basename(root)
        buffer = io.BytesIO()
        with tarfile.open(
            fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT
        ) as archive:
            for name in sorted(self.files):
                if name != root and not name.startswith(root.rstrip("/") + "/"):
                    continue
                kind, data, mode = self.files[name]
                info = tarfile.TarInfo(base + name[len(root) :])
                info.mode, info.uid = mode, 1000
                info.pax_headers = {"SCHILY.xattr.security.capability": "cap"}
                if kind == "dir":
                    info.type = tarfile.DIRTYPE
                    archive.addfile(info)
                elif kind == "symlink":
                    info.type, info.linkname = tarfile.SYMTYPE, data
                    archive.addfile(info)
                elif kind == "hardlink":
                    info.type, info.linkname = tarfile.LNKTYPE, base + data[len(root) :]
                    archive.addfile(info)
                elif kind == "fifo":
                    info.type = tarfile.FIFOTYPE
                    archive.addfile(info)
                else:
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        return Response(200, header=encoded(found), body=buffer.getvalue())

    def put_archive(self, container, path, data):
        body = data if isinstance(data, bytes) else data.read()
        self.puts.append((path, body))
        assert self._stat(path)["mode"] & DIR_MODE, "put targets a directory"
        with tarfile.open(fileobj=io.BytesIO(body)) as archive:
            for member in archive:
                target = posixpath.join(path, member.name)
                if member.isdir():
                    self.add(target, "dir", None, member.mode)
                elif member.issym():
                    self.add(target, "symlink", member.linkname, 0o777)
                else:
                    self.add(
                        target, "file", archive.extractfile(member).read(), member.mode
                    )
        return True


class Client:
    def __init__(self, api):
        self.api = api


def target(api, *, state=None, tmpfs=("/dev/shm", "/scratch")):
    status = state or {"Status": "running", "Running": True, "Paused": False}
    return ArchiveTarget(
        container_id="c" * 64, tmpfs=tmpfs, inspect=lambda: {"State": status}
    )


@pytest.fixture
def transfer(stages):
    api = FakeArchiveAPI()
    api.add("/etc/os-release", data=b"ID=busybox\n")
    api.add("/usr/bin", "dir")
    api.add("/bin", "symlink", "/usr/bin", 0o777)
    api.add("/app/link-to-scratch", "symlink", "/scratch", 0o777)
    api.add("/scratch", "dir")
    return api, ArchiveTransfer(Client(api), stages), stages


def test_path_stat_follows_links_in_container_scope(transfer):
    api, archive, _ = transfer
    link = archive.path_stat(target(api), "/bin", follow=False)
    assert (link["kind"], link["link_target"]) == ("symlink", "/usr/bin")
    followed = archive.path_stat(target(api), "/bin", follow=True)
    assert (followed["kind"], followed["link_target"]) == ("dir", None)
    assert archive.path_stat(target(api), "/missing", follow=True)["exists"] is False
    with pytest.raises(SandboxError, match="normalized absolute"):
        archive.path_stat(target(api), "relative", follow=True)


def test_copy_in_creates_missing_ancestors_then_extracts(transfer):
    api, archive, stages = transfer
    stage_id = upload(stages, tar_bytes(reg("seed.txt", b"seed")))["stage_id"]

    result = archive.copy_in(target(api), "/app/deep/dir", stage_id)

    assert result == {"entries": 1, "bytes": 4}
    (ancestors_path, ancestors), (dest, payload) = api.puts
    assert ancestors_path == "/app"
    assert [item[:3] for item in members(ancestors)] == [
        ("deep", tarfile.DIRTYPE, "0o755"),
        ("deep/dir", tarfile.DIRTYPE, "0o755"),
    ]
    assert dest == "/app/deep/dir"
    assert api.files["/app/deep/dir/seed.txt"][1] == b"seed"
    with pytest.raises(SandboxError, match="unknown or expired"):
        archive.copy_in(target(api), "/app", stage_id)


@pytest.mark.parametrize(
    "dest",
    (
        "/scratch",
        "/scratch/sub",
        "/dev/shm/x",
        "/dev",
        "/proc/1",
        "/app/link-to-scratch",
    ),
)
def test_copy_in_refuses_tmpfs_and_kernel_filesystem_targets(transfer, dest):
    api, archive, stages = transfer
    stage_id = upload(stages, tar_bytes(reg("a")))["stage_id"]
    with pytest.raises(SandboxError) as caught:
        archive.copy_in(target(api), dest, stage_id)
    assert caught.value.code == "unsupported"
    assert api.puts == []
    # A refused destination does not consume the stage.
    assert stages.size(stage_id) > 0


@pytest.mark.parametrize(
    ("dest", "entry"),
    (
        ("/", "scratch/hidden"),
        ("/", "scratch"),
        ("/", "dev/shm/x"),
        # An existing container symlink below the destination leads there.
        ("/", "app/link-to-scratch/x"),
        ("/app", "link-to-scratch/x"),
        ("/app", "link-to-scratch/deep/er/x"),
    ),
)
def test_copy_in_refuses_entries_that_reach_a_tmpfs_below_the_destination(
    transfer, dest, entry
):
    api, archive, stages = transfer
    stage_id = upload(stages, tar_bytes(reg("ok"), reg(entry)))["stage_id"]
    with pytest.raises(SandboxError, match="reach a tmpfs") as caught:
        archive.copy_in(target(api), dest, stage_id)
    assert caught.value.code == "unsupported"
    assert api.puts == []


def test_copy_in_follows_ordinary_container_symlinks_and_resolves_each_dir_once(
    transfer,
):
    api, archive, stages = transfer
    api.add("/usr/lib/x", "dir")
    api.add("/lib", "symlink", "/usr/lib", 0o777)
    raw = tar_bytes(
        reg("bin/a"), reg("bin/b"), reg("lib/x/c"), reg("new/d/e"), reg("new/d/f")
    )
    stage_id = upload(stages, raw)["stage_id"]
    api.heads.clear()

    assert archive.copy_in(target(api), "/", stage_id)["entries"] == 5

    # bin and lib are links (each also stats its target), x is resolved
    # below the link target, new is absent: nothing below it is looked up.
    assert sorted(api.heads) == [
        "/bin",
        "/lib",
        "/new",
        "/usr/bin",
        "/usr/lib",
        "/usr/lib/x",
    ]


def test_copy_in_bounds_the_directories_it_resolves(transfer, monkeypatch):
    api, archive, stages = transfer
    for name in ("a", "b", "c"):
        api.add(f"/app/{name}", "dir")
    monkeypatch.setattr(sandbox_archive, "MAX_RESOLVED_DIRS", 2)
    stage_id = upload(stages, tar_bytes(reg("a/1"), reg("b/1"), reg("c/1")))
    with pytest.raises(SandboxError) as caught:
        archive.copy_in(target(api), "/app", stage_id["stage_id"])
    assert caught.value.code == "quota"
    assert api.puts == []


def test_copy_in_refuses_a_tmpfs_destination_even_if_its_lower_layer_links_out(
    transfer,
):
    api, archive, stages = transfer
    # The daemon sees the layer under the tmpfs; the service sees the tmpfs.
    api.add("/scratch/sub", "symlink", "/etc", 0o777)
    stage_id = upload(stages, tar_bytes(reg("a")))["stage_id"]
    with pytest.raises(SandboxError) as caught:
        archive.copy_in(target(api), "/scratch/sub", stage_id)
    assert caught.value.code == "unsupported"
    assert api.puts == []


@pytest.mark.parametrize(
    ("state", "code"),
    (
        ({"Status": "paused", "Running": True, "Paused": True}, "busy"),
        ({"Status": "restarting", "Running": True, "Paused": False}, "busy"),
        ({"Status": "created", "Running": False, "Paused": False}, None),
        ({"Status": "exited", "Running": False, "Paused": False}, None),
    ),
)
def test_copies_run_on_created_running_exited_but_never_paused(transfer, state, code):
    api, archive, stages = transfer
    stage_id = upload(stages, tar_bytes(reg("a")))["stage_id"]
    service = target(api, state=state)
    if code is None:
        archive.copy_in(service, "/app", stage_id)
        archive.copy_out(service, "/app", max_bytes=100)
        return
    with pytest.raises(SandboxError) as caught:
        archive.copy_in(service, "/app", stage_id)
    assert caught.value.code == code
    with pytest.raises(SandboxError) as caught:
        archive.copy_out(service, "/app", max_bytes=100)
    assert caught.value.code == code
    assert api.puts == [] and api.gets == []


def test_copy_in_into_a_file_or_through_a_dangling_link_is_invalid(transfer):
    api, archive, stages = transfer
    api.add("/app/dangling", "symlink", "/nowhere", 0o777)
    for dest in ("/etc/os-release", "/etc/os-release/x", "/app/dangling/x"):
        stage_id = upload(stages, tar_bytes(reg("a")))["stage_id"]
        with pytest.raises(SandboxError) as caught:
            archive.copy_in(target(api), dest, stage_id)
        assert caught.value.code == "invalid"


def test_copy_out_reemits_canonical_tar_with_exclusions_and_skips(transfer):
    api, archive, stages = transfer
    api.add("/app/src/main.py", data=b"print(1)\n", mode=0o755)
    api.add("/app/src/main.pyc", data=b"\0" * 10)
    api.add("/app/node_modules/pkg/index.js", data=b"x" * 50)
    api.add("/app/src/other.py", "hardlink", "/app/src/main.py")
    api.add("/app/pipe", "fifo")
    api.add("/app/scratch/hidden", data=b"lower layer")
    shadowed = target(api, tmpfs=("/dev/shm", "/app/scratch"))

    result = archive.copy_out(
        shadowed, "/app", max_bytes=1024, exclude=["*.pyc", "./node_modules"]
    )

    # main.pyc, node_modules + 2 below it, the fifo, scratch + 1 below it.
    assert result["skipped"] == 7
    data = stages.read(result["stage_id"], 0, MAX_STAGE_FRAME)
    listed = {item[0]: item for item in members(data)}
    assert sorted(listed) == [
        "app",
        "app/link-to-scratch",
        "app/src",
        "app/src/main.py",
        "app/src/other.py",
    ]
    # The hardlink becomes a regular copy of already written data.
    assert listed["app/src/other.py"][1] == tarfile.REGTYPE
    assert listed["app/src/other.py"][-1] == b"print(1)\n"
    assert all(item[3:6] == (0, 0, "") for item in listed.values())
    assert (result["entries"], result["bytes"]) == (5, 18)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        assert all("SCHILY.xattr.security.capability" not in m.pax_headers for m in tar)


def test_copy_out_budget_is_a_quota_and_leaves_no_stage(transfer, tmp_path):
    api, archive, _ = transfer
    api.add("/app/big", data=b"x" * 2048)
    with pytest.raises(SandboxError) as caught:
        archive.copy_out(target(api), "/app", max_bytes=1024)
    assert caught.value.code == "quota"
    assert list((tmp_path / "spool").iterdir()) == []


def test_copy_out_refuses_tmpfs_missing_and_root_paths(transfer):
    api, archive, _ = transfer
    for path, code in (
        ("/scratch", "unsupported"),
        ("/dev/shm", "unsupported"),
        # The parent resolves through a container symlink into the tmpfs.
        ("/app/link-to-scratch/x", "unsupported"),
        ("/missing/file", "invalid"),
        ("/missing", "invalid"),
        ("/", "invalid"),
    ):
        with pytest.raises(SandboxError) as caught:
            archive.copy_out(target(api), path, max_bytes=100)
        assert caught.value.code == code, path
    with pytest.raises(SandboxError, match="64 glob"):
        archive.copy_out(target(api), "/app", max_bytes=100, exclude=["*"] * 65)


def test_copy_out_of_a_symlink_archives_the_link_itself(transfer):
    api, archive, stages = transfer
    result = archive.copy_out(target(api), "/bin", max_bytes=100)
    data = stages.read(result["stage_id"], 0, 4096)
    assert [item[:2] + item[6:7] for item in members(data)] == [
        ("bin", tarfile.SYMTYPE, "/usr/bin")
    ]
    assert api.gets == ["/bin"]


def test_daemon_rejections_are_reported_not_hidden(transfer):
    api, archive, stages = transfer

    def refuse(container, path, data):
        http = requests.Response()
        http.status_code = 403
        raise APIError(
            "ro",
            response=http,
            explanation="mkdir /var/lib/docker/overlay2/4f1c/merged/app: read-only",
        )

    api.put_archive = refuse
    stage_id = upload(stages, tar_bytes(reg("a")))["stage_id"]
    with pytest.raises(SandboxError, match="read-only") as caught:
        archive.copy_in(target(api), "/app", stage_id)
    assert caught.value.code == "invalid"
    # A fixed text: Engine explanations can name host paths.
    assert "/var/lib/docker" not in str(caught.value)

    def lost(*args, **kwargs):
        raise requests.exceptions.ConnectionError("daemon went away")

    api.head = lost
    with pytest.raises(SandboxError) as caught:
        archive.path_stat(target(api), "/app", follow=False)
    assert caught.value.code == "unknown-outcome"


def test_a_symlinked_spool_is_refused_and_a_loose_one_is_tightened(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir(mode=0o755)
    os.symlink(spool, tmp_path / "link")
    with pytest.raises(OSError):
        StageStore(tmp_path / "link")
    StageStore(spool).close()
    assert stat.S_IMODE(spool.stat().st_mode) == 0o700
