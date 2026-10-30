"""Rootless network-policy helpers for Blue Vela Apptainer runs."""

from __future__ import annotations

import os
import select
import socket
import struct
import threading
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from rsi_harness.errors import SetupError
from rsi_harness.runtime.network import PinnedEndpoint

_MAX_PROXY_HEADER_BYTES = 16 * 1024


def _peer_pid(connection: socket.socket) -> int:
    """The connecting process, as the kernel recorded it at connect()."""
    try:
        credentials = connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
    except OSError:
        return 0
    return struct.unpack("3i", credentials)[0]


class UnixConnectProxy:
    """Expose only pinned HTTPS CONNECT targets on a private Unix socket.

    With ``pin_first_peer`` the first connection only registers its process
    (the relay, which connects before it starts anything else) and is closed;
    afterwards a connection from any other process is closed unanswered. A
    Judge's routes share the relay's uid and, below Landlock ABI 7, can still
    connect to a socket path they cannot read, so the path alone is no
    boundary: the kernel-recorded peer pid is.
    """

    def __init__(
        self,
        socket_path: Path,
        *,
        endpoints: tuple[PinnedEndpoint, ...],
        connect_timeout_seconds: float = 10.0,
        pin_first_peer: bool = False,
    ) -> None:
        if not endpoints:
            raise SetupError("Agent provider proxy requires pinned endpoints")
        self.socket_path = Path(socket_path)
        self._endpoints = {
            (endpoint.hostname.rstrip(".").lower(), endpoint.port): endpoint
            for endpoint in endpoints
        }
        if len(self._endpoints) != len(endpoints):
            raise SetupError("Agent provider proxy endpoints must be unique")
        self._connect_timeout = connect_timeout_seconds
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._client_threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._pin_first_peer = pin_first_peer
        self.owner_pid: int | None = None
        self.refused_peers = 0

    def start(self) -> None:
        if self._listener is not None:
            raise RuntimeError("Agent provider proxy is already started")
        self.socket_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.socket_path.exists():
            raise SetupError("Agent provider proxy socket path already exists")
        listener = socket.socket(socket.AF_UNIX)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen()
            listener.settimeout(0.2)
        except BaseException:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener
        self._accept_thread = threading.Thread(
            target=self._accept,
            name="rsi-bluevela-provider-proxy",
            daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        self._stopping.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
            self._accept_thread = None
        with self._lock:
            client_threads = tuple(self._client_threads)
        for thread in client_threads:
            thread.join(timeout=2)
        self.socket_path.unlink(missing_ok=True)

    def _accept(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stopping.is_set():
            try:
                client, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stopping.is_set():
                    return
                continue
            if self._pin_first_peer and not self._admit(client):
                client.close()
                continue
            thread = threading.Thread(
                target=self._serve,
                args=(client,),
                name="rsi-bluevela-provider-connection",
                daemon=True,
            )
            with self._lock:
                self._client_threads.append(thread)
            thread.start()

    def _admit(self, client: socket.socket) -> bool:
        """Pin the first peer (and close it); admit only that process after."""
        pid = _peer_pid(client)
        if self.owner_pid is None:
            # Only the first peer registers. One the kernel cannot name
            # (pid 0) pins nobody: every later connection is refused.
            self.owner_pid = pid if pid > 0 else -1
            return False
        if pid <= 0 or pid != self.owner_pid:
            self.refused_peers += 1
            return False
        return True

    def _serve(self, client: socket.socket) -> None:
        try:
            client.settimeout(self._connect_timeout)
            header, remainder = self._read_header(client)
            target = self._target(header)
            if target is None:
                client.sendall(b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n")
                return
            upstream = self._connect(target)
            if upstream is None:
                client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n")
                return
            with upstream:
                client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if remainder:
                    upstream.sendall(remainder)
                client.settimeout(None)
                upstream.settimeout(None)
                self._relay(client, upstream)
        except (OSError, UnicodeError, ValueError):
            try:
                client.sendall(b"HTTP/1.1 400 Bad Request\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
        finally:
            client.close()

    @staticmethod
    def _read_header(client: socket.socket) -> tuple[bytes, bytes]:
        content = bytearray()
        while b"\r\n\r\n" not in content:
            if len(content) >= _MAX_PROXY_HEADER_BYTES:
                raise ValueError("proxy request header is too large")
            chunk = client.recv(4096)
            if not chunk:
                raise ValueError("proxy request ended before its header")
            content.extend(chunk)
        header, remainder = bytes(content).split(b"\r\n\r\n", 1)
        return header, remainder

    def _target(self, header: bytes) -> PinnedEndpoint | None:
        request_line = header.split(b"\r\n", 1)[0].decode("ascii")
        method, authority, version = request_line.split(" ", 2)
        if method != "CONNECT" or not version.startswith("HTTP/1."):
            return None
        hostname, separator, raw_port = authority.rpartition(":")
        if not separator or not hostname or not raw_port.isascii():
            return None
        port = int(raw_port)
        if not 1 <= port <= 65535:
            return None
        return self._endpoints.get((hostname.rstrip(".").lower(), port))

    def _connect(self, endpoint: PinnedEndpoint) -> socket.socket | None:
        for address in endpoint.addresses:
            try:
                return socket.create_connection(
                    (str(address), endpoint.port),
                    timeout=self._connect_timeout,
                )
            except OSError:
                continue
        return None

    def _relay(self, left: socket.socket, right: socket.socket) -> None:
        sockets = (left, right)
        while not self._stopping.is_set():
            readable, _writable, _errors = select.select(sockets, (), (), 0.2)
            for source in readable:
                target = right if source is left else left
                data = source.recv(64 * 1024)
                if not data:
                    return
                target.sendall(data)


class UnixTcpRelay:
    """Relay a private Unix socket to one fixed TCP control endpoint."""

    def __init__(
        self,
        socket_path: Path,
        *,
        target_host: str,
        target_port: int,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        if not target_host or not 1 <= target_port <= 65535:
            raise SetupError("TCP relay requires one valid fixed target")
        self.socket_path = Path(socket_path)
        self._target = (target_host, target_port)
        self._connect_timeout = connect_timeout_seconds
        self._listener: socket.socket | None = None
        self._accept_thread: threading.Thread | None = None
        self._client_threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._listener is not None:
            raise RuntimeError("fixed TCP relay is already started")
        self.socket_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        if self.socket_path.exists():
            raise SetupError("fixed TCP relay socket path already exists")
        listener = socket.socket(socket.AF_UNIX)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen()
            listener.settimeout(0.2)
        except BaseException:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener
        self._accept_thread = threading.Thread(
            target=self._accept,
            name="rsi-bluevela-submit-relay",
            daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        self._stopping.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2)
            self._accept_thread = None
        with self._lock:
            client_threads = tuple(self._client_threads)
        for thread in client_threads:
            thread.join(timeout=2)
        self.socket_path.unlink(missing_ok=True)

    def _accept(self) -> None:
        listener = self._listener
        assert listener is not None
        while not self._stopping.is_set():
            try:
                client, _address = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stopping.is_set():
                    return
                continue
            thread = threading.Thread(
                target=self._serve,
                args=(client,),
                name="rsi-bluevela-submit-connection",
                daemon=True,
            )
            with self._lock:
                self._client_threads.append(thread)
            thread.start()

    def _serve(self, client: socket.socket) -> None:
        try:
            upstream = socket.create_connection(
                self._target,
                timeout=self._connect_timeout,
            )
            with upstream:
                client.settimeout(None)
                upstream.settimeout(None)
                sockets = (client, upstream)
                while not self._stopping.is_set():
                    readable, _writable, _errors = select.select(sockets, (), (), 0.2)
                    for source in readable:
                        target = upstream if source is client else client
                        data = source.recv(64 * 1024)
                        if not data:
                            return
                        target.sendall(data)
        except OSError:
            return
        finally:
            client.close()


class AgentNetworkBroker:
    """Own the host-side sockets exposed to one no-network Agent: the submit
    relay, and the provider proxy unless the Agent has no provider (Oracle,
    which only submits)."""

    _CONTAINER_ROOT = PurePosixPath("/run/rsi-harness/network")

    def __init__(
        self,
        root: Path,
        *,
        endpoints: tuple[PinnedEndpoint, ...],
        submit_url: str,
    ) -> None:
        parsed = urlsplit(submit_url)
        try:
            port = parsed.port
        except ValueError as error:
            raise SetupError("Agent submit endpoint has an invalid port") from error
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise SetupError("Agent submit endpoint must be local HTTP")
        self.root = Path(root)
        self.provider_socket: Path | None = (
            self.root / "provider.sock" if endpoints else None
        )
        self.submit_socket = self.root / "submit.sock"
        self.submit_port = port
        self._provider = (
            None
            if self.provider_socket is None
            else UnixConnectProxy(self.provider_socket, endpoints=endpoints)
        )
        self._submit = UnixTcpRelay(
            self.submit_socket,
            target_host="127.0.0.1",
            target_port=port,
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Agent network broker is already started")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=False)
        if self._provider is not None:
            self._provider.start()
        try:
            self._submit.start()
        except BaseException:
            if self._provider is not None:
                self._provider.stop()
            raise
        self._started = True

    def stop(self) -> None:
        self._submit.stop()
        if self._provider is not None:
            self._provider.stop()
        self._started = False

    def wrap(
        self,
        command: tuple[str, ...],
        *,
        relay_path: PurePosixPath,
    ) -> tuple[str, ...]:
        if not self._started:
            raise RuntimeError("Agent network broker is not started")
        if self.provider_socket is None:
            raise RuntimeError("Agent network broker has no provider proxy")
        return (
            "/usr/bin/env",
            "python3",
            str(relay_path),
            "--provider-socket",
            str(self._CONTAINER_ROOT / self.provider_socket.name),
            "--submit-socket",
            str(self._CONTAINER_ROOT / self.submit_socket.name),
            "--submit-port",
            str(self.submit_port),
            "--",
            *command,
        )


class JudgeProviderBroker:
    """One Judge round's provider proxy: the model endpoints pinned for Work.

    A model-access Judge whose Agent has no network runs in its own network
    namespace; the relay exposes this socket on loopback port
    :data:`JUDGE_PROVIDER_PORT`, which rsi-agent's launch spec names as the
    Agent's proxy. No submit relay: a Judge never submits. The relay
    registers itself first and the proxy then serves only the relay's
    process, so a route (same uid, no internet sockets) that finds the
    socket still cannot use it.
    """

    _CONTAINER_ROOT = PurePosixPath("/run/rsi-harness/network")

    def __init__(self, root: Path, *, endpoints: tuple[PinnedEndpoint, ...]) -> None:
        self.root = Path(root)
        self.provider_socket = self.root / "provider.sock"
        self._provider = UnixConnectProxy(
            self.provider_socket, endpoints=endpoints, pin_first_peer=True
        )
        self._started = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Judge provider broker is already started")
        self.root.mkdir(parents=True, mode=0o700, exist_ok=False)
        self._provider.start()
        self._started = True

    def stop(self) -> None:
        self._provider.stop()
        self._started = False

    def wrap(
        self, command: tuple[str, ...], *, relay_path: PurePosixPath
    ) -> tuple[str, ...]:
        if not self._started:
            raise RuntimeError("Judge provider broker is not started")
        return (
            "/usr/bin/env",
            "python3",
            str(relay_path),
            "--provider-socket",
            str(self._CONTAINER_ROOT / self.provider_socket.name),
            "--provider-port",
            str(JUDGE_PROVIDER_PORT),
            "--register",
            "--",
            *command,
        )


# Loopback port of a model-access Judge's provider proxy, fixed because
# rsi-agent's launch spec names it before the Judge starts.
JUDGE_PROVIDER_PORT = 3128


def judge_proxy_environment() -> dict[str, str]:
    """What the Agent's own environment needs to reach the Judge's proxy."""
    url = f"http://127.0.0.1:{JUDGE_PROVIDER_PORT}"
    environment = {
        name: url
        for name in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        )
    }
    environment.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    return environment


__all__ = [
    "JUDGE_PROVIDER_PORT",
    "AgentNetworkBroker",
    "JudgeProviderBroker",
    "UnixConnectProxy",
    "UnixTcpRelay",
    "judge_proxy_environment",
]
