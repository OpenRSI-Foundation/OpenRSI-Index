from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.runtime.network import PinnedEndpoint
from tests.landlock_helpers import native_signal_compatibility


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


_PROVIDER_CHILD = r"""
import json, os, socket, subprocess, sys
from urllib.parse import urlsplit

provider_socket, port, agent = sys.argv[1], int(sys.argv[2]), sys.argv[3]
allow_missing_signal = sys.argv[4]
request = f"CONNECT provider.test:{port} HTTP/1.1\r\n\r\n".encode()


def headers(client):
    content = b""
    while b"\r\n\r\n" not in content:
        try:
            chunk = client.recv(4096)
        except ConnectionResetError:
            break
        if not chunk:
            break
        content += chunk
    return content


report = {}
# What rsi-agent's Agent does: the relay's loopback proxy.
proxy = urlsplit(os.environ["HTTPS_PROXY"])
with socket.create_connection((proxy.hostname, proxy.port), timeout=5) as client:
    client.sendall(request)
    report["relay"] = headers(client).split(b"\r\n", 1)[0].decode()
    client.sendall(b"ping")
    report["echo"] = client.recv(4).decode()
# Any other process of the Judge (this one included) is turned away.
with socket.socket(socket.AF_UNIX) as client:
    client.settimeout(5)
    client.connect(provider_socket)
    try:
        client.sendall(request)
    except BrokenPipeError:
        pass  # Refused and closed before the request was written.
    report["direct"] = headers(client).decode() or "closed"
sys.path.insert(0, agent)
import rsi_agent

libc = rsi_agent._load_libc()
if rsi_agent.landlock_abi(libc)[0] >= 1:
    # A same-uid process confined by Landlock that cannot read the socket's
    # directory: below ABI 7 it can still connect to the path, but the proxy
    # serves only the relay.
    probe = (
        "import ctypes, json, os, socket, sys\n"
        "sys.path.insert(0, sys.argv[3])\n"
        "import rsi_agent as ra\n"
        "libc = ra._load_libc()\n"
        "directory = os.path.dirname(sys.argv[1])\n"
        "ruleset = ra.Ruleset(libc, ra.landlock_abi(libc)[0], "
        "allow_missing_landlock_signal=(sys.argv[4] == '1'))\n"
        "for path in ra.readable_paths([directory]):\n"
        "    ruleset.allow(path, ra.FS_READ)\n"
        "assert libc.prctl(ra.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0\n"
        "ruleset.restrict()\n"
        "out = {}\n"
        "try:\n"
        "    os.listdir(directory); out['list'] = 'ok'\n"
        "except OSError as error:\n"
        "    out['list'] = error.errno\n"
        "with socket.socket(socket.AF_UNIX) as client:\n"
        "    client.settimeout(5)\n"
        "    client.connect(sys.argv[1])\n"
        "    try:\n"
        "        client.sendall(sys.argv[2].encode())\n"
        "        out['unix'] = client.recv(64).decode() or 'closed'\n"
        "    except (BrokenPipeError, ConnectionResetError):\n"
        "        out['unix'] = 'closed'\n"
        "print(json.dumps(out))\n"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe, provider_socket, request.decode(),
         agent, allow_missing_signal],
        capture_output=True,
        text=True,
        timeout=60,
    )
    report["confined"] = json.loads(run.stdout or "null") or run.stderr
print(json.dumps(report))
"""


def test_the_judge_provider_proxy_serves_only_the_registered_relay(
    tmp_path: Path,
) -> None:
    import errno
    import json
    import tempfile

    from rsi_harness.cluster.bluevela.network import JudgeProviderBroker

    source = Path(__file__).resolve().parents[3] / "src/rsi_harness"
    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()
    upstream.settimeout(0.2)
    upstream_port = int(upstream.getsockname()[1])
    stopping = threading.Event()

    def echo() -> None:
        while not stopping.is_set():
            try:
                connection, _address = upstream.accept()
            except OSError:
                continue
            with connection:
                connection.sendall(connection.recv(4))

    echo_thread = threading.Thread(target=echo, daemon=True)
    echo_thread.start()
    # Short: a Unix socket path has a 108-byte limit.
    root = Path(tempfile.mkdtemp(prefix="rsi-pp-", dir="/tmp"))
    broker = JudgeProviderBroker(
        root / "network",
        endpoints=(
            PinnedEndpoint(
                hostname="provider.test",
                port=upstream_port,
                addresses=(ipaddress.ip_address("127.0.0.1"),),
            ),
        ),
    )
    broker.start()
    try:
        relay = subprocess.Popen(
            (
                sys.executable,
                str(source / "cluster/bluevela/netns_relay.py"),
                "--provider-socket",
                str(broker.provider_socket),
                "--provider-port",
                "0",
                "--register",
                "--",
                sys.executable,
                "-c",
                _PROVIDER_CHILD,
                str(broker.provider_socket),
                str(upstream_port),
                str(source / "judge_agent"),
                str(int(native_signal_compatibility())),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = relay.communicate(timeout=120)
        assert relay.returncode == 0, stderr
        report = json.loads(stdout)
        assert broker._provider.owner_pid == relay.pid
    finally:
        broker.stop()
        stopping.set()
        echo_thread.join(timeout=2)
        upstream.close()
        shutil.rmtree(root)
    assert report["relay"].startswith("HTTP/1.1 200 ")
    assert report["echo"] == "ping"
    assert report["direct"] == "closed"
    if "confined" in report:
        assert report["confined"] == {"list": errno.EACCES, "unix": "closed"}
    assert broker._provider.refused_peers == 1 + ("confined" in report)


def test_a_judge_provider_proxy_whose_first_peer_has_no_pid_serves_nobody(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsi_harness.cluster.bluevela import network

    proxy = network.UnixConnectProxy(
        tmp_path / "provider.sock",
        endpoints=(
            PinnedEndpoint(
                hostname="provider.test",
                port=443,
                addresses=(ipaddress.ip_address("127.0.0.1"),),
            ),
        ),
        pin_first_peer=True,
    )
    pids = iter((0, 4242, 4242, 0))
    monkeypatch.setattr(network, "_peer_pid", lambda _client: next(pids))

    # The registration the kernel could not name pins nobody, so a later
    # peer (a route, say) never takes the relay's place.
    assert [proxy._admit(None) for _ in range(4)] == [False] * 4
    assert proxy.owner_pid == -1
    assert proxy.refused_peers == 3


def test_an_agent_without_a_provider_gets_only_the_submit_relay(
    tmp_path: Path,
) -> None:
    from rsi_harness.cluster.bluevela.network import AgentNetworkBroker

    broker = AgentNetworkBroker(
        tmp_path / "broker", endpoints=(), submit_url="http://127.0.0.1:43123"
    )
    broker.start()
    try:
        assert broker.provider_socket is None
        assert [path.name for path in broker.root.iterdir()] == ["submit.sock"]
        with pytest.raises(RuntimeError, match="no provider proxy"):
            broker.wrap(
                ("rsi-oracle",),
                relay_path=PurePosixPath("/usr/local/bin/rsi-netns-relay"),
            )
    finally:
        broker.stop()
    assert not broker.submit_socket.exists()
