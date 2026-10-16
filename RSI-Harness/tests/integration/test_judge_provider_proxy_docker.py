"""Real Docker: the Judge's provider proxy pins its relay across pid namespaces.

On a cluster the proxy runs in the Harness on the node while the relay runs
in the Judge's container, which has its own pid namespace. SO_PEERCRED gives
the peer's pid as the proxy's namespace sees it; a container stands in for
the Judge here, so a namespace layout that hid the relay's pid (pid 0, every
forward refused) fails this test instead of a cluster round.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import socket
import tempfile
import threading
from pathlib import Path

import pytest

from rsi_harness.cluster.bluevela.network import JudgeProviderBroker
from rsi_harness.runtime.network import PinnedEndpoint
from tests.cluster.bluevela.test_network import _PROVIDER_CHILD
from tests.integration.test_judge_model_access_docker import PYTHON_IMAGE, _docker

pytestmark = pytest.mark.integration

SOURCE = Path(__file__).resolve().parents[2] / "src/rsi_harness"
CHILD = _PROVIDER_CHILD.replace(
    "print(json.dumps(report))",
    # The relay's pid inside the Judge's own namespace.
    'report["relay_namespace_pid"] = os.getppid()\nprint(json.dumps(report))',
)


def test_the_proxy_admits_its_relay_from_another_pid_namespace() -> None:
    client = _docker(PYTHON_IMAGE)
    upstream = socket.socket()
    upstream.bind(("127.0.0.1", 0))
    upstream.listen()
    upstream.settimeout(0.2)
    port = int(upstream.getsockname()[1])
    stopping = threading.Event()

    def echo() -> None:
        while not stopping.is_set():
            try:
                connection, _address = upstream.accept()
            except OSError:
                continue
            with connection:
                connection.sendall(connection.recv(4))

    thread = threading.Thread(target=echo, daemon=True)
    thread.start()
    # Short: a Unix socket path has a 108-byte limit.
    root = Path(tempfile.mkdtemp(prefix="rsi-pp-", dir="/tmp"))
    broker = JudgeProviderBroker(
        root / "network",
        endpoints=(
            PinnedEndpoint(
                hostname="provider.test",
                port=port,
                addresses=(ipaddress.ip_address("127.0.0.1"),),
            ),
        ),
    )
    broker.start()
    container = None
    try:
        container = client.containers.run(
            PYTHON_IMAGE,
            [
                "python3",
                "/opt/relay/netns_relay.py",
                "--provider-socket",
                "/run/rsi-harness/network/provider.sock",
                "--provider-port",
                "0",
                "--register",
                "--",
                "python3",
                "-c",
                CHILD,
                "/run/rsi-harness/network/provider.sock",
                str(port),
                "/opt/judge_agent",
            ],
            detach=True,
            network_mode="none",
            user=f"{os.getuid()}:{os.getgid()}",
            read_only=True,
            security_opt=["no-new-privileges"],
            cap_drop=["ALL"],
            tmpfs={"/tmp": "rw,nosuid,nodev,size=16m,mode=1777"},
            volumes={
                str(broker.root): {"bind": "/run/rsi-harness/network", "mode": "ro"},
                str(SOURCE / "cluster/bluevela"): {"bind": "/opt/relay", "mode": "ro"},
                str(SOURCE / "judge_agent"): {"bind": "/opt/judge_agent", "mode": "ro"},
            },
            labels={"rsi-harness.test": "judge-provider-proxy"},
        )
        result = container.wait(timeout=120)
        output = container.logs(stdout=True, stderr=False).decode()
        assert result["StatusCode"] == 0, container.logs().decode()
        report = json.loads(output)
    finally:
        if container is not None:
            container.remove(force=True)
        broker.stop()
        stopping.set()
        thread.join(timeout=2)
        upstream.close()
        shutil.rmtree(root)

    owner = broker._provider.owner_pid
    # The proxy named the relay by its pid in the proxy's own namespace.
    assert owner is not None and owner > 0
    assert report["relay_namespace_pid"] != owner
    # The relay's forwards were admitted; every other process was refused.
    assert report["relay"].startswith("HTTP/1.1 200 "), report
    assert report["echo"] == "ping"
    assert report["direct"] == "closed"
    assert broker._provider.refused_peers == 1 + ("confined" in report)
    if "confined" in report:
        assert report["confined"]["unix"] == "closed"
