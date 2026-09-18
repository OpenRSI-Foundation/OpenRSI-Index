from __future__ import annotations

from pathlib import Path

from rsi_harness.runtime import production


class _PingingClient:
    def __init__(self) -> None:
        self.pinged = False

    def ping(self) -> bool:
        self.pinged = True
        return True


def test_docker_client_waits_longer_than_the_sdk_default_for_a_rootfs_commit(
    tmp_path: Path, monkeypatch
) -> None:
    # A Judge round commits the Work rootfs; the SDK's 60 s read timeout turned a
    # multi-minute commit of a large Work tree into an infrastructure_error.
    captured: dict[str, object] = {}
    client = _PingingClient()

    def fake_from_env(**kwargs: object) -> _PingingClient:
        captured.update(kwargs)
        return client

    monkeypatch.setattr(production.docker, "from_env", fake_from_env)
    services = production.ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        inventory=object(),
        rsi_loop_config=object(),
    )

    assert services._client() is client
    assert client.pinged
    assert captured["timeout"] == production.DOCKER_API_TIMEOUT_SECONDS
    assert production.DOCKER_API_TIMEOUT_SECONDS >= 600
    assert services._client() is client  # cached; no second construction
