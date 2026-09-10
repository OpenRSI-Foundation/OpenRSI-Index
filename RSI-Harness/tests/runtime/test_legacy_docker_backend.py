from __future__ import annotations

import pytest

from rsi_loop.harness.backend import docker_backend
from tests.fakes import FakeDockerClient


@pytest.fixture
def client(monkeypatch):
    def reject_live_client():
        pytest.fail("Legacy backend tests must use the fake Docker client")

    monkeypatch.setattr(docker_backend.docker, "from_env", reject_live_client)
    return FakeDockerClient()


@pytest.mark.parametrize("gpus", [None, "", " \t ", "0", " 0 ", "+0", "-0"])
def test_create_omits_gpu_requests_for_unset_empty_or_zero(monkeypatch, client, gpus):
    if gpus is None:
        monkeypatch.delenv("RSI_DOCKER_GPUS", raising=False)
    else:
        monkeypatch.setenv("RSI_DOCKER_GPUS", gpus)

    handle = docker_backend.DockerBackend(client=client).create_container(
        "work", "legacy-work"
    )

    assert handle.id == "container-1"
    assert client.containers.created == [
        {
            "image": "work",
            "name": "legacy-work",
            "detach": True,
            "command": "tail -f /dev/null",
            "environment": {},
        }
    ]


@pytest.mark.parametrize(
    ("gpus", "count"),
    [("1", 1), ("2", 2), (" 2 ", 2), ("all", -1), (" ALL ", -1), ("-1", -1)],
)
def test_create_preserves_positive_and_all_gpu_requests(
    monkeypatch, client, gpus, count
):
    monkeypatch.setenv("RSI_DOCKER_GPUS", gpus)

    docker_backend.DockerBackend(client=client).create_container("work", "legacy-work")

    requests = client.containers.created[0]["device_requests"]
    assert len(requests) == 1
    assert requests[0]["Count"] == count
    assert requests[0]["Capabilities"] == [["gpu"]]


@pytest.mark.parametrize("gpus", ["invalid", "0.0"])
def test_create_rejects_invalid_gpu_counts_before_creating_container(
    monkeypatch, client, gpus
):
    monkeypatch.setenv("RSI_DOCKER_GPUS", gpus)

    with pytest.raises(ValueError):
        docker_backend.DockerBackend(client=client).create_container(
            "work", "legacy-work"
        )

    assert client.containers.created == []
