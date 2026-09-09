from __future__ import annotations

import ipaddress
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath

from rsi_harness.runtime.network import PinnedEndpoint


def _read_headers(connection: socket.socket) -> bytes:
    content = bytearray()
    while b"\r\n\r\n" not in content:
        chunk = connection.recv(4096)
        if not chunk:
            break
        content.extend(chunk)
    return bytes(content)


def test_connect_proxy_relays_only_to_the_pinned_provider_address(
    tmp_path: Path,
) -> None:
    from rsi_harness.cluster.bluevela.network import UnixConnectProxy

    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()
    upstream_port = int(upstream.getsockname()[1])

    def echo_once() -> None:
        connection, _address = upstream.accept()
        with connection:
            connection.sendall(connection.recv(4))

    echo_thread = threading.Thread(target=echo_once, daemon=True)
    echo_thread.start()
    proxy = UnixConnectProxy(
        tmp_path / "provider.sock",
        endpoints=(
            PinnedEndpoint(
                hostname="provider.test",
                port=upstream_port,
                addresses=(ipaddress.ip_address("127.0.0.1"),),
            ),
        ),
    )
    proxy.start()
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(proxy.socket_path))
            client.sendall(
                f"CONNECT provider.test:{upstream_port} HTTP/1.1\r\n\r\n".encode()
            )
            assert _read_headers(client).startswith(b"HTTP/1.1 200 ")
            client.sendall(b"ping")
            assert client.recv(4) == b"ping"
    finally:
        proxy.stop()
        upstream.close()
        echo_thread.join(timeout=1)


def test_connect_proxy_rejects_an_unpinned_destination(tmp_path: Path) -> None:
    from rsi_harness.cluster.bluevela.network import UnixConnectProxy

    proxy = UnixConnectProxy(
        tmp_path / "provider.sock",
        endpoints=(
            PinnedEndpoint(
                hostname="provider.test",
                port=443,
                addresses=(ipaddress.ip_address("192.0.2.10"),),
            ),
        ),
    )
    proxy.start()
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(proxy.socket_path))
            client.sendall(b"CONNECT huggingface.co:443 HTTP/1.1\r\n\r\n")
            assert _read_headers(client).startswith(b"HTTP/1.1 403 ")
    finally:
        proxy.stop()


def test_unix_tcp_relay_reaches_only_its_fixed_tcp_target(tmp_path: Path) -> None:
    from rsi_harness.cluster.bluevela.network import UnixTcpRelay

    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()
    upstream_port = int(upstream.getsockname()[1])

    def echo_once() -> None:
        connection, _address = upstream.accept()
        with connection:
            connection.sendall(connection.recv(4))

    echo_thread = threading.Thread(target=echo_once, daemon=True)
    echo_thread.start()
    relay = UnixTcpRelay(
        tmp_path / "submit.sock",
        target_host="127.0.0.1",
        target_port=upstream_port,
    )
    relay.start()
    try:
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(relay.socket_path))
            client.sendall(b"ping")
            assert client.recv(4) == b"ping"
    finally:
        relay.stop()
        upstream.close()
        echo_thread.join(timeout=1)


def test_netns_relay_exposes_only_provider_proxy_and_submit_loopback(
    tmp_path: Path,
) -> None:
    relay_script = (
        Path(__file__).resolve().parents[3]
        / "src/rsi_harness/cluster/bluevela/netns_relay.py"
    )
    provider_socket = tmp_path / "provider.sock"
    submit_socket = tmp_path / "submit.sock"

    def unix_echo_once(path: Path, ready: threading.Event) -> None:
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            listener.listen()
            ready.set()
            connection, _address = listener.accept()
            with connection:
                content = connection.recv(8)
                connection.sendall(content)

    provider_ready = threading.Event()
    submit_ready = threading.Event()
    threads = (
        threading.Thread(
            target=unix_echo_once,
            args=(provider_socket, provider_ready),
            daemon=True,
        ),
        threading.Thread(
            target=unix_echo_once,
            args=(submit_socket, submit_ready),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()
    assert provider_ready.wait(timeout=1)
    assert submit_ready.wait(timeout=1)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        submit_port = int(reserved.getsockname()[1])

    child = """
import os, socket
from urllib.parse import urlsplit
proxy = urlsplit(os.environ['HTTPS_PROXY'])
with socket.create_connection((proxy.hostname, proxy.port), timeout=2) as client:
    client.sendall(b'provider')
    assert client.recv(8) == b'provider'
submit = urlsplit(os.environ['RSI_JUDGE_URL'])
with socket.create_connection((submit.hostname, submit.port), timeout=2) as client:
    client.sendall(b'submit')
    assert client.recv(6) == b'submit'
assert os.environ['HTTP_PROXY'] == os.environ['HTTPS_PROXY']
print('netns-relay: OK')
"""
    environment = os.environ.copy()
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
    ):
        environment.pop(name, None)
    environment["RSI_JUDGE_URL"] = f"http://127.0.0.1:{submit_port}"
    result = subprocess.run(
        (
            sys.executable,
            str(relay_script),
            "--provider-socket",
            str(provider_socket),
            "--submit-socket",
            str(submit_socket),
            "--submit-port",
            str(submit_port),
            "--",
            sys.executable,
            "-c",
            child,
        ),
        env=environment,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "netns-relay: OK\n"
    for thread in threads:
        thread.join(timeout=1)


def test_agent_network_broker_wraps_one_isolated_agent_lifecycle(
    tmp_path: Path,
) -> None:
    from rsi_harness.cluster.bluevela.network import AgentNetworkBroker

    broker = AgentNetworkBroker(
        tmp_path / "broker",
        endpoints=(
            PinnedEndpoint(
                hostname="provider.test",
                port=443,
                addresses=(ipaddress.ip_address("192.0.2.10"),),
            ),
        ),
        submit_url="http://127.0.0.1:43123",
    )
    broker.start()
    try:
        assert broker.provider_socket.is_socket()
        assert broker.submit_socket.is_socket()
        assert broker.wrap(
            ("codex", "exec"),
            relay_path=PurePosixPath("/usr/local/bin/rsi-netns-relay"),
        ) == (
            "/usr/bin/env",
            "python3",
            "/usr/local/bin/rsi-netns-relay",
            "--provider-socket",
            "/run/rsi-harness/network/provider.sock",
            "--submit-socket",
            "/run/rsi-harness/network/submit.sock",
            "--submit-port",
            "43123",
            "--",
            "codex",
            "exec",
        )
    finally:
        broker.stop()
    assert not broker.provider_socket.exists()
    assert not broker.submit_socket.exists()
