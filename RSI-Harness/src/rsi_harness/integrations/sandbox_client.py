#!/usr/bin/env python3
"""Standalone stdlib sandbox client and safe bounded file primitives."""

from __future__ import annotations

import argparse
import base64
import bisect
import hashlib
import http.client
import io
import json
import os
import re
import socket
import stat
import struct
import sys
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import PurePosixPath
from types import SimpleNamespace

MAGIC = b"RSIBNDL1\n"
MAX_BUNDLE_BYTES = 32 * 1024**2
MAX_BODY_BYTES = 48 * 1024**2
MAX_FILE_BYTES = 16 * 1024**2
MAX_ENTRIES = 1024
MAX_HEADER_BYTES = 4096
MAX_CONTROL_BYTES = 128 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 1024**2
MAX_EXEC_RESPONSE_BYTES = 8 * 1024**2
MAX_PATH_DEPTH = 32
# Protocol versions this client speaks; v2 adds brokered environments.
VERSIONS = (1, 2)
MAX_STAGE_BYTES = 16 * 1024**2
MAX_WAIT_SEC = 30
# A v1 child ID is 32 hex digits; an env handle is "e" and 32 hex digits.
ENV_HANDLE = re.compile(r"e[0-9a-f]{32}")
# Requests whose transport failure leaves their effect unknown.
_MUTATIONS = frozenset(
    {
        "create",
        "exec",
        "upload",
        "destroy",
        "stage_put",
        "image_pull",
        "image_build",
        "job_cancel",
        "image_release",
        "env_create",
        "env_start",
        "env_stop_service",
        "env_destroy",
        "exec_start",
        "exec_kill",
        "copy_in",
        "copy_out",
    }
)
_RESPONSE_LIMITS = {
    "download": MAX_BODY_BYTES,
    "stage_get": MAX_STAGE_BYTES + 65536,
}


class ProtocolError(ValueError):
    def __init__(self, message, *, field="bundle", code="invalid"):
        super().__init__(message)
        self.field = field
        self.code = code


def validate_name(value):
    if not isinstance(value, str):
        raise ProtocolError("entry path must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or str(path) != value
        or value == "."
        or ".." in path.parts
        or "\x00" in value
        or "\\" in value
        or len(path.parts) > MAX_PATH_DEPTH
        or len(value.encode("utf-8")) > MAX_HEADER_BYTES
    ):
        raise ProtocolError("entry path must be normalized and relative", field="path")
    return value


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError("duplicate JSON field")
        result[key] = value
    return result


def _check_header(header):
    if not isinstance(header, dict) or set(header) != {"path", "kind", "mode", "size"}:
        raise ProtocolError("entry header requires path, kind, mode and size only")
    validate_name(header["path"])
    if header["kind"] not in ("file", "directory"):
        raise ProtocolError(
            "only regular files and directories are supported", field="kind"
        )
    mode, size = header["mode"], header["size"]
    if type(mode) is not int or not 0 <= mode <= 0o777:
        raise ProtocolError(
            "mode must contain ordinary permission bits only", field="mode"
        )
    if type(size) is not int or not 0 <= size <= MAX_FILE_BYTES:
        raise ProtocolError(
            "file size exceeds limit or is not a nonnegative integer", field="size"
        )
    if header["kind"] == "directory" and size != 0:
        raise ProtocolError("directory entry must have zero size", field="size")


class _BundleBounds:
    def __init__(self, max_bytes=MAX_BUNDLE_BYTES):
        if type(max_bytes) is not int or not 0 <= max_bytes <= MAX_BUNDLE_BYTES:
            raise ProtocolError("invalid remaining byte limit", code="quota")
        self.max_bytes = max_bytes
        self.names = {}
        self.ordered_names = []
        self.bytes = 0

    def add(self, header):
        _check_header(header)
        name = header["path"]
        if name in self.names:
            raise ProtocolError("duplicate entry path", field="path")
        if header["kind"] == "file":
            prefix = name + "/"
            index = bisect.bisect_left(self.ordered_names, prefix)
            if index < len(self.ordered_names) and self.ordered_names[index].startswith(
                prefix
            ):
                raise ProtocolError(
                    "file entry conflicts with child path", field="path"
                )
        parts = name.split("/")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            if self.names.get(parent) == "file":
                raise ProtocolError("entry parent is a file", field="path")
        self.names[name] = header["kind"]
        # Keep one reference per complete name, not a copy of every ancestor:
        # long first components and deep paths must not amplify metadata memory.
        bisect.insort(self.ordered_names, name)
        self.bytes += header["size"]
        if len(self.names) > MAX_ENTRIES or self.bytes > self.max_bytes:
            raise ProtocolError("bundle entry/byte quota exceeded", code="quota")


def _read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        part = stream.read(min(65536, size - len(result)))
        if not part:
            raise ProtocolError("truncated bundle")
        result.extend(part)
    return bytes(result)


def iter_bundle(stream, *, max_bytes=MAX_BUNDLE_BYTES):
    if _read_exact(stream, len(MAGIC)) != MAGIC:
        raise ProtocolError("unsupported bundle version")
    bounds = _BundleBounds(max_bytes)
    body_size = len(MAGIC)
    while True:
        size = struct.unpack("!I", _read_exact(stream, 4))[0]
        body_size += 4
        if size == 0:
            if stream.read(1):
                raise ProtocolError("trailing data after bundle terminator")
            return
        if size > MAX_HEADER_BYTES:
            raise ProtocolError("entry metadata exceeds limit", code="quota")
        try:
            header = json.loads(
                _read_exact(stream, size), object_pairs_hook=_unique_json
            )
        except (ValueError, UnicodeError, RecursionError) as error:
            raise ProtocolError("invalid entry JSON") from error
        bounds.add(header)
        body_size += size + header["size"]
        if body_size > MAX_BODY_BYTES:
            raise ProtocolError("bundle wire size exceeds limit", code="quota")
        yield {
            "path": header["path"],
            "kind": header["kind"],
            "mode": header["mode"],
            "data": _read_exact(stream, header["size"]),
        }


def write_bundle(stream, records, *, max_bytes=MAX_BUNDLE_BYTES):
    stream.write(MAGIC)
    bounds = _BundleBounds(max_bytes)
    wire_size = len(MAGIC) + 4
    for record in records:
        if set(record) != {"path", "kind", "mode", "data"} or not isinstance(
            record["data"], bytes
        ):
            raise ProtocolError("invalid file record")
        header = {key: record[key] for key in ("path", "kind", "mode")}
        header["size"] = len(record["data"])
        bounds.add(header)
        encoded = json.dumps(header, separators=(",", ":")).encode()
        wire_size += 4 + len(encoded) + header["size"]
        if len(encoded) > MAX_HEADER_BYTES or wire_size > MAX_BODY_BYTES:
            raise ProtocolError("bundle metadata/wire quota exceeded", code="quota")
        stream.write(struct.pack("!I", len(encoded)))
        stream.write(encoded)
        view = memoryview(record["data"])
        for offset in range(0, len(view), 65536):
            stream.write(view[offset : offset + 65536])
    stream.write(struct.pack("!I", 0))


def encode_records(records, *, max_bytes=MAX_BUNDLE_BYTES):
    stream = io.BytesIO()
    write_bundle(stream, records, max_bytes=max_bytes)
    return stream.getvalue()


@contextmanager
def _directory(root, *, create=False):
    """Open every ancestor without following links, including the user root."""
    absolute = os.path.abspath(os.fspath(root))
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for name in absolute.split("/")[1:]:
            if not name:
                continue
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def iter_local_records(root, *, max_bytes=MAX_BUNDLE_BYTES, _bounds=None, _prefix=""):
    bounds = _bounds if _bounds is not None else _BundleBounds(max_bytes)

    def walk(descriptor, prefix):
        with os.scandir(descriptor) as children:
            for item in children:
                path = prefix + item.name
                before = item.stat(follow_symlinks=False)
                mode = stat.S_IMODE(before.st_mode)
                if stat.S_ISDIR(before.st_mode):
                    kind = "directory"
                elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                    kind = "file"
                else:
                    raise ProtocolError(
                        "links and special files are forbidden", field="path"
                    )
                header = {
                    "path": path,
                    "kind": kind,
                    "mode": mode,
                    "size": before.st_size if kind == "file" else 0,
                }
                bounds.add(header)
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                if kind == "directory":
                    flags |= os.O_DIRECTORY
                child = os.open(item.name, flags, dir_fd=descriptor)
                try:
                    after = os.fstat(child)
                    if (
                        before.st_dev,
                        before.st_ino,
                        before.st_mode,
                        before.st_nlink,
                    ) != (after.st_dev, after.st_ino, after.st_mode, after.st_nlink):
                        raise ProtocolError(
                            "filesystem entry changed during transfer", field="path"
                        )
                    if kind == "directory":
                        yield {"path": path, "kind": kind, "mode": mode, "data": b""}
                        yield from walk(child, path + "/")
                    else:
                        with os.fdopen(os.dup(child), "rb") as source:
                            data = _read_exact(source, before.st_size)
                            if source.read(1):
                                raise ProtocolError(
                                    "file grew during transfer", field="size"
                                )
                        yield {"path": path, "kind": kind, "mode": mode, "data": data}
                finally:
                    os.close(child)

    with _directory(root) as descriptor:
        yield from walk(descriptor, _prefix)


@contextmanager
def _relative_directory(descriptor, parts, *, create=False, path_only=False):
    current = os.dup(descriptor)
    try:
        for name in parts:
            if create:
                try:
                    os.mkdir(name, 0o700, dir_fd=current)
                except FileExistsError:
                    pass
            child = os.open(
                name,
                (os.O_PATH if path_only else os.O_RDONLY)
                | os.O_DIRECTORY
                | os.O_NOFOLLOW,
                dir_fd=current,
            )
            os.close(current)
            current = child
        yield current
    finally:
        os.close(current)


def write_local_records(root, records):
    bounds = _BundleBounds()
    directory_modes = {}
    with _directory(root, create=True) as descriptor:
        # Bound and validate the complete bundle before preparing directories.
        # Files may precede their explicitly declared, currently readonly parents.
        # Reject observable file conflicts immediately, including while stdin is
        # still streaming, so an invalid upload need not consume its whole body.
        prepared = []
        directory_identities = {}
        for record in records:
            if set(record) != {"path", "kind", "mode", "data"} or not isinstance(
                record["data"], bytes
            ):
                raise ProtocolError("invalid file record")
            record = dict(record)
            header = {key: record[key] for key in ("path", "kind", "mode")}
            header["size"] = len(record["data"])
            bounds.add(header)
            parts = record["path"].split("/")
            try:
                with _relative_directory(
                    descriptor, parts[:-1], path_only=True
                ) as parent:
                    target = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                    if record["kind"] == "directory":
                        if not stat.S_ISDIR(target.st_mode):
                            raise ProtocolError(
                                "destination is not a directory", field="path"
                            )
                        directory_identities[record["path"]] = (
                            target.st_dev,
                            target.st_ino,
                        )
                    elif not stat.S_ISREG(target.st_mode) or target.st_nlink != 1:
                        raise ProtocolError(
                            "destination is not a unique regular file", field="path"
                        )
            except (FileNotFoundError, PermissionError):
                # Missing or unsearchable ancestors are checked again after
                # declared directories have been safely prepared.
                pass
            prepared.append(record)
        try:
            directories = sorted(
                (record for record in prepared if record["kind"] == "directory"),
                key=lambda record: record["path"].count("/"),
            )
            for record in directories:
                with _relative_directory(
                    descriptor, record["path"].split("/"), create=True, path_only=True
                ) as child:
                    identity = os.fstat(child)
                    expected = directory_identities.get(record["path"])
                    if expected is not None and expected != (
                        identity.st_dev,
                        identity.st_ino,
                    ):
                        raise ProtocolError(
                            "directory changed during transfer", field="path"
                        )
                    directory_modes[record["path"]] = (
                        record["mode"],
                        identity.st_dev,
                        identity.st_ino,
                    )
                    # O_PATH pins even a 0300/0000 directory without requiring
                    # read permission. procfs addresses that exact open inode;
                    # chmod still requires normal ownership, without following
                    # any untrusted filesystem link or changing implicit parents.
                    os.chmod(f"/proc/self/fd/{child}", 0o700)
            for record in prepared:
                if record["kind"] == "directory":
                    continue
                parts = record["path"].split("/")
                with _relative_directory(descriptor, parts[:-1], create=True) as parent:
                    name = parts[-1]
                    if record["kind"] == "file":
                        try:
                            target = os.stat(name, dir_fd=parent, follow_symlinks=False)
                        except FileNotFoundError:
                            target = None
                        if target is not None and (
                            not stat.S_ISREG(target.st_mode) or target.st_nlink != 1
                        ):
                            raise ProtocolError(
                                "destination is not a unique regular file", field="path"
                            )
                        temporary = ".rsi-transfer-" + uuid.uuid4().hex
                        output = os.open(
                            temporary,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            0o600,
                            dir_fd=parent,
                        )
                        try:
                            with os.fdopen(output, "wb") as destination:
                                destination.write(record["data"])
                                destination.flush()
                                os.fchmod(destination.fileno(), record["mode"])
                            os.replace(
                                temporary, name, src_dir_fd=parent, dst_dir_fd=parent
                            )
                        finally:
                            try:
                                os.unlink(temporary, dir_fd=parent)
                            except FileNotFoundError:
                                pass
        finally:
            # Reopen at most one path at a time: a 1,024-entry bundle must not
            # exhaust the child's 1,024-fd limit. Never follow replacement links
            # or apply a saved mode to a different directory inode.
            restoration_error = None
            for path in sorted(
                directory_modes, key=lambda p: p.count("/"), reverse=True
            ):
                mode, device, inode = directory_modes[path]
                try:
                    with _relative_directory(descriptor, path.split("/")) as child:
                        identity = os.fstat(child)
                        if (identity.st_dev, identity.st_ino) != (device, inode):
                            raise ProtocolError(
                                "directory changed during transfer", field="path"
                            )
                        os.fchmod(child, mode)
                except (OSError, ProtocolError) as error:
                    if restoration_error is None:
                        restoration_error = error
            if restoration_error is not None:
                raise restoration_error


def iter_selected_records(root, selections, *, max_bytes=MAX_BUNDLE_BYTES):
    """Export only requested entries; dot selects the contents of the root."""
    bounds = _BundleBounds(max_bytes)
    for selected in selections:
        if selected == ".":
            records = iter_local_records(root, _bounds=bounds)
        else:
            validate_name(selected)
            target = os.path.join(os.fspath(root), selected)
            with _directory(os.path.dirname(target)) as parent:
                name = os.path.basename(target)
                before = os.stat(name, dir_fd=parent, follow_symlinks=False)
                mode = stat.S_IMODE(before.st_mode)
                if stat.S_ISDIR(before.st_mode):
                    bounds.add(
                        {"path": selected, "kind": "directory", "mode": mode, "size": 0}
                    )

                    def directory_records():
                        yield {
                            "path": selected,
                            "kind": "directory",
                            "mode": mode,
                            "data": b"",
                        }
                        yield from iter_local_records(
                            target, _bounds=bounds, _prefix=selected + "/"
                        )

                    records = directory_records()
                elif stat.S_ISREG(before.st_mode) and before.st_nlink == 1:
                    bounds.add(
                        {
                            "path": selected,
                            "kind": "file",
                            "mode": mode,
                            "size": before.st_size,
                        }
                    )
                    descriptor = os.open(
                        name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
                    )
                    with os.fdopen(descriptor, "rb") as source:
                        after = os.fstat(source.fileno())
                        if (
                            before.st_dev,
                            before.st_ino,
                            before.st_mode,
                            before.st_nlink,
                        ) != (
                            after.st_dev,
                            after.st_ino,
                            after.st_mode,
                            after.st_nlink,
                        ):
                            raise ProtocolError("selected file changed during transfer")
                        data = _read_exact(source, before.st_size)
                        if source.read(1):
                            raise ProtocolError("selected file grew during transfer")
                    records = (
                        {"path": selected, "kind": "file", "mode": mode, "data": data},
                    )
                else:
                    raise ProtocolError(
                        "selected path is a link or special file", field="path"
                    )
        yield from records


def _transfer_main(operation, root, selections, byte_limit=MAX_BUNDLE_BYTES):
    """Host-injected fixed helper; no Docker credentials or host paths."""
    try:
        if operation == "upload":
            write_local_records(
                root, iter_bundle(sys.stdin.buffer, max_bytes=byte_limit)
            )
            sys.stdout.buffer.write(b"ok\n")
        elif operation == "download":
            if selections is None:
                control = sys.stdin.buffer.read(MAX_CONTROL_BYTES + 1)
                if len(control) > MAX_CONTROL_BYTES:
                    raise ProtocolError(
                        "download selection exceeds limit", code="quota"
                    )
                selections = json.loads(control, object_pairs_hook=_unique_json)
                if (
                    not isinstance(selections, list)
                    or not 1 <= len(selections) <= MAX_ENTRIES
                ):
                    raise ProtocolError("expected bounded nonempty download selection")
            write_bundle(
                sys.stdout.buffer,
                iter_selected_records(root, selections, max_bytes=byte_limit),
                max_bytes=byte_limit,
            )
        else:
            raise ProtocolError("unsupported transfer operation")
        sys.stdout.buffer.flush()
    except (OSError, ValueError) as error:
        sys.stderr.write(
            json.dumps(
                {
                    "code": getattr(error, "code", "invalid"),
                    "field": getattr(error, "field", "path"),
                    "message": str(error)[:4096],
                }
            )
        )
        raise SystemExit(2) from None


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = str(socket_path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        try:
            self.sock.connect(self.socket_path)
        except BaseException:
            self.sock.close()
            self.sock = None
            raise


def _versions(result):
    """Protocol versions a capabilities answer offers and this client shares.

    ``version`` stays the v1 field; a v2 server also lists every version it
    speaks in ``versions``, which must then include ``version``. A phase may
    use v1 only with a non-null ``grant`` and v2 only with non-null
    ``environments``; those select what is actually offered.
    """
    if not isinstance(result, dict) or type(result.get("version")) is not int:
        return frozenset()
    versions = result.get("versions")
    if versions is None:
        versions = [1] if result["version"] == 1 else []
    if (
        not isinstance(versions, list)
        or any(type(item) is not int for item in versions)
        or (versions and result["version"] not in versions)
    ):
        return frozenset()
    offered = set(versions) & set(VERSIONS)
    if "grant" in result and result["grant"] is None:
        offered.discard(1)
    if result.get("environments") is None:
        offered.discard(2)
    return frozenset(offered)


class SandboxClient:
    """No retries, Docker fallback, package dependencies or agent-specific hooks.

    Mutations carry a ``request_id``: after a transport failure the caller
    may repeat one with the same id, and the broker replays its result.
    """

    def __init__(self, socket_path=None, credential=None):
        self.socket_path = socket_path or os.environ.get("RSI_SANDBOX_SOCKET")
        self.credential = credential or os.environ.get("RSI_SANDBOX_TOKEN")
        self._capabilities = None

    def _request(self, operation, metadata, *, bundle=None, timeout_sec=30):
        if not self.socket_path or not self.credential:
            raise ProtocolError(
                "sandbox endpoint is not configured", code="unsupported"
            )
        body = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
        if len(body) > MAX_CONTROL_BYTES:
            raise ProtocolError("control request exceeds limit", code="quota")
        if bundle is not None:
            body = struct.pack("!I", len(body)) + body + bundle
        if len(body) > MAX_BODY_BYTES:
            raise ProtocolError("request body exceeds limit", code="quota")
        connection = UnixHTTPConnection(
            self.socket_path, timeout=float(timeout_sec) + 10
        )
        try:
            connection.request(
                "POST",
                "/v1/" + operation,
                body,
                headers={
                    "Authorization": "Bearer " + self.credential,
                    "Content-Type": "application/octet-stream"
                    if bundle is not None
                    else "application/json",
                },
            )
            response = connection.getresponse()
            limit = _RESPONSE_LIMITS.get(operation, MAX_EXEC_RESPONSE_BYTES)
            declared = response.getheader("Content-Length")
            if declared is not None and (
                not declared.isdecimal() or int(declared) > limit
            ):
                raise ProtocolError("response length exceeds limit", code="quota")
            chunks, total = [], 0
            while True:
                chunk = response.read(min(65536, limit + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > limit:
                    raise ProtocolError("response body exceeds limit", code="quota")
            payload = b"".join(chunks)
            if response.status != 200:
                try:
                    error = json.loads(payload, object_pairs_hook=_unique_json)["error"]
                    raise ProtocolError(
                        error["message"], code=error["code"], field=error["field"]
                    )
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise ProtocolError("invalid sandbox error response") from error
            if operation == "download":
                return tuple(iter_bundle(io.BytesIO(payload)))
            if operation == "stage_get":
                return payload
            return json.loads(payload, object_pairs_hook=_unique_json)
        except (FileNotFoundError, ConnectionRefusedError) as error:
            raise ProtocolError(
                "sandbox endpoint is unavailable", code="unsupported"
            ) from error
        except (OSError, http.client.HTTPException) as error:
            code = "unknown-outcome" if operation in _MUTATIONS else "infrastructure"
            raise ProtocolError(
                "sandbox transport failed; do not replay exec automatically", code=code
            ) from error
        finally:
            connection.close()

    def capabilities(self):
        result = self._request("capabilities", {})
        if not _versions(result):
            raise ProtocolError(
                "unsupported sandbox protocol version", code="unsupported"
            )
        self._capabilities = result
        return result

    def _handshake(self, version=1):
        if self._capabilities is None:
            self.capabilities()
        if version not in _versions(self._capabilities):
            raise ProtocolError(
                f"sandbox protocol version {version} is not offered",
                code="unsupported",
            )

    def create(self, profile, lifetime_sec, request_id=None):
        self._handshake()
        result = self._request(
            "create",
            {
                "profile": profile,
                "lifetime_sec": lifetime_sec,
                "request_id": request_id or uuid.uuid4().hex,
            },
        )
        return result["child_id"]

    def execute(self, child_id, argv, cwd, env=None, timeout_sec=30):
        self._handshake()
        result = self._request(
            "exec",
            {
                "child_id": child_id,
                "argv": list(argv),
                "cwd": cwd,
                "env": {} if env is None else env,
                "timeout_sec": timeout_sec,
            },
            timeout_sec=timeout_sec,
        )
        return SimpleNamespace(**result)

    def upload(self, child_id, root, records, request_id=None, timeout_sec=30):
        self._handshake()
        return self._request(
            "upload",
            {
                "child_id": child_id,
                "root": root,
                "request_id": request_id or uuid.uuid4().hex,
                "timeout_sec": timeout_sec,
            },
            bundle=encode_records(records),
            timeout_sec=timeout_sec,
        )

    def download(self, child_id, root, paths, timeout_sec=30):
        self._handshake()
        return self._request(
            "download",
            {
                "child_id": child_id,
                "root": root,
                "paths": list(paths),
                "timeout_sec": timeout_sec,
            },
            timeout_sec=timeout_sec,
        )

    def status(self, child_id):
        self._handshake()
        return self._request("status", {"child_id": child_id})

    def destroy(self, child_id):
        self._handshake()
        return self._request("destroy", {"child_id": child_id})

    # -- environments (protocol version 2) --------------------------------------

    def _v2(self, operation, metadata, **options):
        self._handshake(2)
        return self._request(operation, metadata, **options)

    def env_create(self, spec, request_id=None):
        return self._v2(
            "env_create",
            {"spec": spec, "request_id": request_id or uuid.uuid4().hex},
        )

    def env_start(self, env_id, wait_timeout_sec, request_id=None):
        return self._v2(
            "env_start",
            {
                "env_id": env_id,
                "wait_timeout_sec": wait_timeout_sec,
                "request_id": request_id or uuid.uuid4().hex,
            },
        )

    def env_status(self, env_id, wait_sec=0):
        return self._v2(
            "env_status",
            {"env_id": env_id, "wait_sec": wait_sec},
            timeout_sec=wait_sec + 30,
        )

    def env_stop_service(self, env_id, service, timeout_sec=10, request_id=None):
        return self._v2(
            "env_stop_service",
            {
                "env_id": env_id,
                "service": service,
                "timeout_sec": timeout_sec,
                "request_id": request_id or uuid.uuid4().hex,
            },
            timeout_sec=timeout_sec + 30,
        )

    def env_destroy(self, env_id):
        return self._v2("env_destroy", {"env_id": env_id}, timeout_sec=90)

    def env_list(self):
        return self._v2("env_list", {})

    def wait_env(self, env_id, timeout_sec):
        """Long-poll env_status until the env is no longer starting."""
        end = time.monotonic() + timeout_sec
        while True:
            wait = max(0, min(MAX_WAIT_SEC, end - time.monotonic()))
            status = self.env_status(env_id, wait)
            if status["state"] not in ("created", "starting") or not wait:
                return status

    def exec_start(
        self,
        env_id,
        service,
        argv,
        *,
        cwd=None,
        env=None,
        user=None,
        timeout_sec=None,
        merge_stderr=False,
        request_id=None,
    ):
        return self._v2(
            "exec_start",
            {
                "env_id": env_id,
                "service": service,
                "argv": list(argv),
                "cwd": cwd,
                "env": {} if env is None else env,
                "user": user,
                "timeout_sec": timeout_sec,
                "merge_stderr": merge_stderr,
                "request_id": request_id or uuid.uuid4().hex,
            },
        )["exec_id"]

    def exec_wait(
        self,
        exec_id,
        stdout_offset=0,
        stderr_offset=0,
        wait_sec=25,
        max_bytes=MAX_OUTPUT_BYTES,
    ):
        return self._v2(
            "exec_wait",
            {
                "exec_id": exec_id,
                "stdout_offset": stdout_offset,
                "stderr_offset": stderr_offset,
                "wait_sec": wait_sec,
                "max_bytes": max_bytes,
            },
            timeout_sec=wait_sec + 30,
        )

    def exec_kill(self, exec_id, signal="TERM", scope="group"):
        return self._v2(
            "exec_kill", {"exec_id": exec_id, "signal": signal, "scope": scope}
        )

    def follow_exec(self, exec_id, on_output=None, wait_sec=25):
        """Read an exec to its end; ``on_output(stream, bytes)`` sees each chunk.

        Returns the final view without output fields.
        """
        offsets = [0, 0]
        while True:
            view = self.exec_wait(exec_id, offsets[0], offsets[1], wait_sec)
            for index, name in enumerate(("stdout", "stderr")):
                data = base64.b64decode(view[name + "_b64"])
                if data and on_output is not None:
                    on_output(name, data)
                offsets[index] = view[name + "_offset"]
            if view["state"] != "running" and offsets == [
                view["stdout_total"],
                view["stderr_total"],
            ]:
                return {
                    key: value
                    for key, value in view.items()
                    if not key.endswith("_b64")
                }

    def stage_put(
        self, data, *, stage_id=None, offset=0, final=True, sha256=None, request_id=None
    ):
        self._handshake(2)
        return self._request(
            "stage_put",
            {
                "stage_id": stage_id,
                "offset": offset,
                "final": final,
                "sha256": sha256,
                "request_id": request_id or uuid.uuid4().hex,
            },
            bundle=bytes(data),
            timeout_sec=60,
        )

    def upload_stage(self, source):
        """Stage a tar from bytes or a binary file in frames of at most 16 MiB."""
        stream = io.BytesIO(source) if isinstance(source, bytes) else source
        stream.seek(0)
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
        stream.seek(0)
        stage_id, offset = None, 0
        chunk = stream.read(MAX_STAGE_BYTES)
        while True:
            following = stream.read(MAX_STAGE_BYTES)
            final = not following
            result = self.stage_put(
                chunk,
                stage_id=stage_id,
                offset=offset,
                final=final,
                sha256=digest.hexdigest() if final else None,
            )
            stage_id, offset = result["stage_id"], offset + len(chunk)
            if final:
                return result
            chunk = following

    def stage_get(self, stage_id, offset=0, length=MAX_STAGE_BYTES):
        return self._v2(
            "stage_get",
            {"stage_id": stage_id, "offset": offset, "length": length},
            timeout_sec=60,
        )

    def download_stage(self, stage_id, target):
        """Write a whole stage into a binary file object."""
        offset = 0
        while True:
            chunk = self.stage_get(stage_id, offset)
            if not chunk:
                return offset
            target.write(chunk)
            offset += len(chunk)

    def copy_in(self, env_id, service, dest_dir, stage_id, request_id=None):
        return self._v2(
            "copy_in",
            {
                "env_id": env_id,
                "service": service,
                "dest_dir": dest_dir,
                "stage_id": stage_id,
                "request_id": request_id or uuid.uuid4().hex,
            },
            timeout_sec=300,
        )

    def copy_out(self, env_id, service, path, max_bytes, exclude=()):
        return self._v2(
            "copy_out",
            {
                "env_id": env_id,
                "service": service,
                "path": path,
                "max_bytes": max_bytes,
                "exclude": list(exclude),
            },
            timeout_sec=300,
        )

    def path_stat(self, env_id, service, path, follow=True):
        return self._v2(
            "path_stat",
            {"env_id": env_id, "service": service, "path": path, "follow": follow},
        )

    def image_pull(self, ref, policy="missing", request_id=None):
        return self._v2(
            "image_pull",
            {
                "ref": ref,
                "policy": policy,
                "request_id": request_id or uuid.uuid4().hex,
            },
        )["job_id"]

    def image_build(
        self,
        stage_id,
        *,
        dockerfile=None,
        dockerfile_inline=None,
        target=None,
        build_args=None,
        labels=None,
        no_cache=False,
        network="public",
        timeout_sec,
        request_id=None,
    ):
        return self._v2(
            "image_build",
            {
                "stage_id": stage_id,
                "dockerfile": dockerfile,
                "dockerfile_inline": dockerfile_inline,
                "target": target,
                "build_args": {} if build_args is None else build_args,
                "labels": {} if labels is None else labels,
                "no_cache": no_cache,
                "network": network,
                "timeout_sec": timeout_sec,
                "request_id": request_id or uuid.uuid4().hex,
            },
        )["job_id"]

    def job_wait(self, job_id, log_offset=0, wait_sec=25):
        return self._v2(
            "job_wait",
            {"job_id": job_id, "log_offset": log_offset, "wait_sec": wait_sec},
            timeout_sec=wait_sec + 30,
        )

    def follow_job(self, job_id, on_log=None, wait_sec=25):
        offset = 0
        while True:
            view = self.job_wait(job_id, offset, wait_sec)
            if view["log"] and on_log is not None:
                on_log(view["log"])
            offset = view["next_offset"]
            if view["state"] not in ("queued", "running") and not view["log"]:
                return view

    def job_cancel(self, job_id):
        return self._v2("job_cancel", {"job_id": job_id})

    def image_list(self):
        return self._v2("image_list", {})

    def image_release(self, image):
        return self._v2("image_release", {"image": image})


def _local_tar(path, target):
    """A tar of one local file or tree rooted at its basename.

    Only regular files, directories and symlinks, never a hardlink entry:
    the broker accepts nothing else.
    """
    root = os.path.abspath(path)
    base = os.path.basename(root.rstrip("/")) or "."
    with tarfile.open(fileobj=target, mode="w", format=tarfile.PAX_FORMAT) as archive:

        def add(source, name):
            info = os.lstat(source)
            entry = tarfile.TarInfo(name)
            entry.mode = stat.S_IMODE(info.st_mode)
            entry.mtime = int(info.st_mtime)
            if stat.S_ISDIR(info.st_mode):
                entry.type = tarfile.DIRTYPE
                archive.addfile(entry)
                for child in sorted(os.listdir(source)):
                    add(os.path.join(source, child), name + "/" + child)
            elif stat.S_ISLNK(info.st_mode):
                entry.type = tarfile.SYMTYPE
                entry.linkname = os.readlink(source)
                archive.addfile(entry)
            elif stat.S_ISREG(info.st_mode):
                entry.size = info.st_size
                with open(source, "rb") as data:
                    archive.addfile(entry, data)
            else:
                raise ProtocolError(f"{source}: special files are not copied")

        add(root, base)


def _remote(value):
    """``ENV_ID:SERVICE:/path`` or None for a local path."""
    parts = value.split(":", 2)
    if len(parts) == 3 and ENV_HANDLE.fullmatch(parts[0]):
        return parts
    return None


def _extract(archive, destination):
    """Unpack a copy_out tar with the ``data`` filter, or, on a Python without
    it (before 3.12 and the backported security releases), with the same
    refusals checked by hand: only files, directories and relative symlinks
    that never name ``..``, and no member outside ``destination``."""
    if hasattr(tarfile, "data_filter"):
        archive.extractall(destination, filter="data")
        return
    members = []
    for member in archive.getmembers():
        parts = PurePosixPath(member.name).parts
        if member.name.startswith("/") or ".." in parts:
            raise ProtocolError(f"{member.name}: unsafe archive member", field="path")
        if member.issym():
            target = member.linkname
            if target.startswith("/") or ".." in PurePosixPath(target).parts:
                raise ProtocolError(
                    f"{member.name}: symlink leaves the copy", field="path"
                )
        elif not (member.isreg() or member.isdir()):
            raise ProtocolError(
                f"{member.name}: unsupported archive member", field="path"
            )
        # As the data filter: no special bits or group/other write, and
        # directories and files stay usable by the extracting user.
        member.mode = 0o755 if member.isdir() else (member.mode & 0o755) | 0o600
        members.append(member)
    archive.extractall(destination, members=members)


def _env_exec(client, args, argv):
    environment = {}
    for item in args.env:
        key, separator, value = item.partition("=")
        if not separator:
            raise ProtocolError("--env requires NAME=VALUE")
        environment[key] = value
    exec_id = client.exec_start(
        args.child_id,
        args.service or "main",
        argv,
        cwd=args.cwd,
        env=environment,
        user=args.user,
        timeout_sec=args.timeout,
        merge_stderr=args.merge_stderr,
    )

    def show(stream, data):
        target = sys.stdout if stream == "stdout" else sys.stderr
        target.buffer.write(data)
        target.flush()

    try:
        final = client.follow_exec(exec_id, show)
    except KeyboardInterrupt:
        client.exec_kill(exec_id, "INT")
        final = client.follow_exec(exec_id, show)
    if final["exit_code"] is not None:
        return final["exit_code"]
    return 124 if final["state"] == "timed_out" else 125


def _copy(client, source, destination, max_bytes):
    remote_source, remote_target = _remote(source), _remote(destination)
    if (remote_source is None) == (remote_target is None):
        raise ProtocolError("cp needs exactly one ENV_ID:SERVICE:/path side")
    with tempfile.TemporaryFile() as spool:
        if remote_target is not None:
            env_id, service, dest_dir = remote_target
            _local_tar(source, spool)
            staged = client.upload_stage(spool)
            return client.copy_in(env_id, service, dest_dir, staged["stage_id"])
        env_id, service, path = remote_source
        result = client.copy_out(env_id, service, path, max_bytes)
        client.download_stage(result["stage_id"], spool)
        spool.seek(0)
        os.makedirs(destination, exist_ok=True)
        with tarfile.open(fileobj=spool, mode="r") as archive:
            _extract(archive, destination)
        return result


def _compose_module():
    """The compose front-end: the endpoint's ``py/`` copy beside this file
    (``/run/rsi-harness/sandbox/rsi-sandbox``), else the installed package."""
    try:
        import rsi_sandbox_compose as module
    except ImportError:
        injected = os.path.join(os.path.dirname(os.path.abspath(__file__)), "py")
        if os.path.isfile(os.path.join(injected, "rsi_sandbox_compose.py")):
            sys.path.insert(0, injected)
            import rsi_sandbox_compose as module
        else:
            from rsi_harness.integrations import sandbox_compose as module
    return module


class _Compose:
    """``rsi-sandbox compose``: one compose project as one env, translated
    here by the same front-end as the Harbor plugin (spec 6). The env handle
    of a project lives in a 0600 state file between invocations."""

    def __init__(self, client, args):
        self.client = client
        self.args = args
        self.module = _compose_module()
        files = [os.path.abspath(path) for path in args.files]
        directory = args.project_directory or (
            os.path.dirname(files[0]) if files else os.getcwd()
        )
        self.project_dir = os.path.abspath(directory)
        if not files:
            for name in (
                "compose.yaml",
                "compose.yml",
                "docker-compose.yaml",
                "docker-compose.yml",
            ):
                candidate = os.path.join(self.project_dir, name)
                if os.path.isfile(candidate):
                    files = [candidate]
                    break
            else:
                raise ProtocolError("no compose file found", field="files")
        self.files = files
        name = args.project_name or os.path.basename(self.project_dir)
        self.project = re.sub(r"[^a-z0-9_-]", "-", name.lower()) or "default"
        # Keyed by the phase session too: a file Work left in a shared
        # $TMPDIR never names a project of the Judge derived from it.
        session = hashlib.sha256((client.credential or "").encode()).hexdigest()
        self.state_path = os.path.join(
            tempfile.gettempdir(),
            f"rsi-sandbox-compose-{os.getuid()}-{session[:12]}-{self.project}.json",
        )

    def translate(self):
        from pathlib import Path

        environ = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("RSI_SANDBOX_")
        }
        project = self.module.load_project(
            [Path(path) for path in self.files],
            environ=environ,
            project_dir=Path(self.project_dir),
        )
        network = self.args.network
        if network is None:
            granted = (self.client.capabilities().get("environments") or {}).get(
                "network", ()
            )
            network = "public" if "public" in granted else "none"
        return self.module.translate(
            project,
            project_dir=Path(self.project_dir),
            network=network,
            disk_mb=self.args.disk_mb,
            lifetime_sec=self.args.lifetime,
            profiles=self.args.profile,
        )

    def state(self):
        try:
            descriptor = os.open(self.state_path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise ProtocolError(
                f"compose project {self.project} is not up", field="project"
            ) from None
        except OSError as error:
            raise ProtocolError(
                f"compose state {self.state_path} is unusable: {error}",
                field="project",
            ) from None
        with os.fdopen(descriptor, encoding="utf-8") as source:
            return json.load(source)

    def save(self, value):
        try:
            # Never reuse or follow a file someone else placed there.
            descriptor = os.open(
                self.state_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
        except FileExistsError:
            raise ProtocolError(
                f"compose project {self.project} is already up", field="project"
            ) from None
        with os.fdopen(descriptor, "w") as output:
            json.dump(value, output)

    def image(self, name, request):
        if isinstance(request, self.module.ImagePull):
            job_id = self.client.image_pull(request.ref, request.policy)
        else:
            job_id = self.module.start_build(
                self.client, request, self.args.wait_timeout or 300, f"service {name}"
            )
        view = self.client.follow_job(job_id, lambda log: sys.stderr.write(log))
        if view["state"] != "succeeded":
            error = view.get("error") or {}
            raise ProtocolError(
                f"service {name}: image {view['state']}: {error.get('message', '')}",
                code="invalid",
                field="image",
            )
        return view["result"]["image"]["handle"], view["kind"] == "build"

    def up(self):
        if os.path.lexists(self.state_path):
            raise ProtocolError(
                f"compose project {self.project} is already up", field="project"
            )
        translation = self.translate()
        for note in translation.notes:
            print(f"note: {note}", file=sys.stderr)
        handles, built = {}, []
        for name in translation.services:
            handles[name], is_built = self.image(name, translation.images[name])
            if is_built:
                built.append(handles[name])
        sent = time.monotonic()
        created = self.client.env_create(translation.env_spec(handles))
        env_id = created["env_id"]
        try:
            self.save({"env_id": env_id, "built": built})
        except ProtocolError:
            # Another `up` of this project won the race: undo this env.
            self.client.env_destroy(env_id)
            raise
        for seed in translation.seeds:
            with tempfile.TemporaryFile() as spool:
                self.module.write_seed_archive(seed, spool)
                staged = self.client.upload_stage(spool)
            self.client.copy_in(env_id, seed.service, seed.dest_dir, staged["stage_id"])
        return self.module.start_env(self.client, created, sent, self.args.wait_timeout)

    def down(self):
        state = self.state()
        result = self.client.env_destroy(state["env_id"])
        for handle in state.get("built", ()):
            self.client.image_release(handle)
        os.unlink(self.state_path)
        return result

    def remote(self, value):
        service, separator, path = value.partition(":")
        if separator and path.startswith("/") and "/" not in service:
            return f"{self.state()['env_id']}:{service}:{path}"
        return value


def _compose(client, args, child_argv):
    project = _Compose(client, args)
    action, rest = args.action, args.args
    if action == "config":
        result = project.module.translation_json(project.translate())
    elif action == "up":
        result = project.up()
        print(json.dumps(result))
        return 0 if result["state"] == "ready" else 1
    elif action == "down":
        result = project.down()
    elif action == "ps":
        result = client.env_status(project.state()["env_id"])
    elif action == "stop":
        if len(rest) != 1:
            raise ProtocolError("stop needs one SERVICE", field="args")
        result = client.env_stop_service(project.state()["env_id"], rest[0])
    elif action == "exec":
        if len(rest) != 1 or not child_argv:
            raise ProtocolError("exec needs SERVICE -- ARGV", field="args")
        options = SimpleNamespace(
            child_id=project.state()["env_id"],
            service=rest[0],
            env=args.env,
            cwd=args.cwd,
            user=args.user,
            timeout=args.timeout,
            merge_stderr=False,
        )
        return _env_exec(client, options, child_argv)
    elif action == "cp":
        if len(rest) != 2:
            raise ProtocolError("cp needs SOURCE and DESTINATION", field="args")
        result = _copy(
            client, project.remote(rest[0]), project.remote(rest[1]), args.max_bytes
        )
    else:
        raise ProtocolError(f"unknown compose action {action}", field="action")
    print(json.dumps(result))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Managed CPU sandboxes (no Docker authority)"
    )
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in (
        "capabilities",
        "create",
        "exec",
        "upload",
        "download",
        "status",
        "destroy",
    ):
        command = subparsers.add_parser(operation)
        command.add_argument("--json", action="store_true")
        if operation == "create":
            command.add_argument("profile")
            command.add_argument("--lifetime", type=float, required=True)
            command.add_argument("--request-id")
        elif operation not in ("capabilities",):
            command.add_argument("child_id")
        if operation in ("exec", "upload", "download"):
            command.add_argument("--timeout", type=float, default=None)
        if operation == "exec":
            # An env handle (e...) selects the brokered env exec (v2).
            command.add_argument("--cwd", default=None)
            command.add_argument("--env", action="append", default=[])
            command.add_argument("--service")
            command.add_argument("--user")
            command.add_argument("--merge-stderr", action="store_true")
        if operation in ("upload", "download"):
            command.add_argument("directory")
            command.add_argument("--root", default="/workspace")
        if operation == "upload":
            command.add_argument("--request-id")
        if operation == "download":
            command.add_argument("--paths", nargs="+", default=["."])
    command = subparsers.add_parser("pull", help="pull an image for envs")
    command.add_argument("ref")
    command.add_argument("--policy", choices=("missing", "always"), default="missing")
    command = subparsers.add_parser("build", help="build an image for envs")
    command.add_argument("context")
    command.add_argument("-f", "--file")
    command.add_argument("--target")
    command.add_argument("--build-arg", action="append", default=[])
    command.add_argument("--no-cache", action="store_true")
    command.add_argument("--network", choices=("public", "none"))
    command.add_argument("--timeout", type=float, default=600)
    command = subparsers.add_parser("up", help="create and start an env")
    command.add_argument("spec", help="EnvSpec JSON file, or - for stdin")
    command.add_argument("--wait-timeout", type=float)
    command.add_argument("--request-id")
    command = subparsers.add_parser("ps", help="list envs, or one env's status")
    command.add_argument("env_id", nargs="?")
    command = subparsers.add_parser("cp", help="copy between here and an env")
    command.add_argument("source", help="local path or ENV_ID:SERVICE:/path")
    command.add_argument("destination", help="local dir or ENV_ID:SERVICE:/dir")
    command.add_argument("--max-bytes", type=int, default=1024**3)
    command = subparsers.add_parser("stop-service", help="stop one env service")
    command.add_argument("env_id")
    command.add_argument("service")
    command.add_argument("--timeout", type=float, default=10)
    command = subparsers.add_parser("rm", help="destroy an env")
    command.add_argument("env_id")
    subparsers.add_parser("images", help="list this session's image handles")
    command = subparsers.add_parser("image-rm", help="release an image handle")
    command.add_argument("image")
    command = subparsers.add_parser(
        "compose",
        help="run a compose project as one env (config|up|ps|exec|cp|stop|down)",
    )
    command.add_argument("-f", "--file", action="append", default=[], dest="files")
    command.add_argument("--project-directory")
    command.add_argument("-p", "--project-name")
    command.add_argument("--profile", action="append", default=[])
    command.add_argument("--network", choices=("public", "none", "allowlist"))
    command.add_argument("--disk-mb", type=int, default=1024)
    command.add_argument("--lifetime", type=float)
    command.add_argument("--wait-timeout", type=float)
    command.add_argument("--env", action="append", default=[])
    command.add_argument("--cwd")
    command.add_argument("--user")
    command.add_argument("--timeout", type=float)
    command.add_argument("--max-bytes", type=int, default=1024**3)
    command.add_argument(
        "action", choices=("config", "up", "ps", "exec", "cp", "stop", "down")
    )
    command.add_argument("args", nargs="*")
    args_list = list(sys.argv[1:] if argv is None else argv)
    child_argv = []
    if "--" in args_list:
        index = args_list.index("--")
        child_argv = args_list[index + 1 :]
        args_list = args_list[:index]
    args = parser.parse_args(args_list)
    client = SandboxClient()
    try:
        if args.operation == "capabilities":
            result = client.capabilities()
        elif args.operation == "create":
            result = {
                "child_id": client.create(args.profile, args.lifetime, args.request_id)
            }
        elif args.operation == "exec" and ENV_HANDLE.fullmatch(args.child_id):
            if not child_argv:
                raise ProtocolError(
                    "exec requires explicit argv after --", field="argv"
                )
            return _env_exec(client, args, child_argv)
        elif args.operation == "exec":
            if not child_argv:
                raise ProtocolError(
                    "exec requires explicit argv after --", field="argv"
                )
            if args.service or args.user or args.merge_stderr:
                raise ProtocolError(
                    "--service, --user and --merge-stderr need an env handle",
                    field="argv",
                )
            environment = {}
            for item in args.env:
                key, separator, value = item.partition("=")
                if not separator:
                    raise ProtocolError("--env requires NAME=VALUE")
                environment[key] = value
            result = vars(
                client.execute(
                    args.child_id,
                    child_argv,
                    args.cwd or "/workspace",
                    environment,
                    30 if args.timeout is None else args.timeout,
                )
            )
        elif args.operation == "upload":
            result = client.upload(
                args.child_id,
                args.root,
                iter_local_records(args.directory),
                args.request_id,
                30 if args.timeout is None else args.timeout,
            )
        elif args.operation == "download":
            records = client.download(
                args.child_id,
                args.root,
                args.paths,
                30 if args.timeout is None else args.timeout,
            )
            write_local_records(args.directory, records)
            result = {"entries": len(records)}
        elif args.operation == "pull":
            job = client.image_pull(args.ref, args.policy)
            result = client.follow_job(job, lambda log: sys.stderr.write(log))
            print(json.dumps(result))
            return 0 if result["state"] == "succeeded" else 1
        elif args.operation == "build":
            module = _compose_module()
            request = module.ImageBuild(
                os.path.abspath(args.context),
                args.file,
                target=args.target,
                args=module.build_args(args.build_arg),
                network=args.network,
                no_cache=args.no_cache,
            )
            job_id = module.start_build(client, request, args.timeout, args.context)
            result = client.follow_job(job_id, lambda log: sys.stderr.write(log))
            print(json.dumps(result))
            return 0 if result["state"] == "succeeded" else 1
        elif args.operation == "up":
            if args.spec == "-":
                spec = json.load(sys.stdin)
            else:
                with open(args.spec, encoding="utf-8") as source:
                    spec = json.load(source)
            sent = time.monotonic()
            created = client.env_create(spec, args.request_id)
            result = _compose_module().start_env(
                client, created, sent, args.wait_timeout
            )
            print(json.dumps(result))
            return 0 if result["state"] == "ready" else 1
        elif args.operation == "ps":
            result = (
                client.env_status(args.env_id) if args.env_id else client.env_list()
            )
        elif args.operation == "cp":
            result = _copy(client, args.source, args.destination, args.max_bytes)
        elif args.operation == "stop-service":
            result = client.env_stop_service(args.env_id, args.service, args.timeout)
        elif args.operation == "rm":
            result = client.env_destroy(args.env_id)
        elif args.operation == "images":
            result = client.image_list()
        elif args.operation == "image-rm":
            result = client.image_release(args.image)
        elif args.operation == "compose":
            return _compose(client, args, child_argv)
        else:
            result = getattr(client, args.operation)(args.child_id)
        print(json.dumps(result))
        return 0
    except (ProtocolError, OSError, ValueError, tarfile.TarError) as error:
        print(
            json.dumps(
                {
                    "error": {
                        "code": getattr(error, "code", "invalid"),
                        "field": getattr(error, "field", "request"),
                        "message": str(error),
                    }
                }
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
