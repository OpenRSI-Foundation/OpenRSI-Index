"""Real Unix transport and a copied stdlib-only CLI, independent of agents."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.runtime.test_sandbox import kit as kit
from tests.runtime.test_sandbox import work as work


@pytest.fixture
def endpoint(kit, work, tmp_path):
    # pytest's full test path can exceed Linux's 107-byte Unix pathname ceiling.
    import tempfile

    from rsi_harness.runtime.sandbox_server import SandboxServer

    with tempfile.TemporaryDirectory(prefix="rsi-sock-") as root:
        path = Path(root) / "s"
        server = SandboxServer(kit[0], path, work.owner)
        server.start()
        try:
            yield path
        finally:
            server.stop()
            kit[0].close()


def test_real_uds_client_lifecycle(endpoint, work):
    from rsi_harness.integrations.sandbox_client import SandboxClient

    client = SandboxClient(endpoint, work.credential)
    assert client.capabilities()["version"] == 1
    child = client.create("offline", 30, "one")
    result = client.execute(child, ["false"], "/workspace", {}, 5)
    assert result.exit_code == 3
    assert result.stdout == "out"
    assert client.status(child)["state"] == "running"
    client.destroy(child)
    client.destroy(child)
    assert client.status(child)["state"] == "removed"


def test_standalone_cli_requires_no_harness_install(endpoint, work, tmp_path):
    from rsi_harness.integrations import sandbox_client as wire

    script = tmp_path / "rsi-sandbox"
    script.write_bytes(Path(wire.__file__).read_bytes())
    environment = {
        "PATH": os.environ["PATH"],
        "RSI_SANDBOX_SOCKET": str(endpoint),
        "RSI_SANDBOX_TOKEN": work.credential,
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            str(script),
            "create",
            "offline",
            "--lifetime",
            "10",
            "--request-id",
            "cli",
            "--json",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert len(json.loads(completed.stdout)["child_id"]) == 32
    assert completed.stderr == ""


def test_client_missing_endpoint_is_unsupported_never_docker(tmp_path):
    from rsi_harness.integrations.sandbox_client import ProtocolError, SandboxClient

    with pytest.raises(ProtocolError, match="endpoint") as info:
        SandboxClient(tmp_path / "absent", "nope").create("offline", 10, "one")
    assert info.value.code == "unsupported"


def test_client_version_mismatch_refuses_mutation(endpoint, work, kit, monkeypatch):
    from rsi_harness.integrations.sandbox_client import ProtocolError, SandboxClient

    original = kit[0].capabilities
    monkeypatch.setattr(
        kit[0], "capabilities", lambda token: {**original(token), "version": 99}
    )
    with pytest.raises(ProtocolError, match="version"):
        SandboxClient(endpoint, work.credential).create("offline", 10, "one")
    assert kit[0].journal.snapshot() == ()
