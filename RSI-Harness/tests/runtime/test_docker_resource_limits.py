import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions, MainServiceConfig, RunRequest
from rsi_harness.runtime import production
from rsi_loop.harness.config import RSILoopConfig


@pytest.mark.parametrize(
    ("cpu_quota", "memory_limit", "missing"),
    [(False, True, "CpuCfsQuota"), (True, False, "MemoryLimit")],
)
def test_unavailable_limit_refused_before_run_state(
    tmp_path, monkeypatch, cpu_quota, memory_limit, missing
):
    # Model a local Linux daemon; the BusyBox task requests 1 CPU and 256 MiB.
    monkeypatch.setattr(production.sys, "platform", "linux")
    fixture = Path(__file__).resolve().parents[1] / "fixtures/tasks/harbor-env-busybox"
    task = tmp_path / "task"
    shutil.copytree(fixture, task)
    (task / "environment").mkdir()
    config = task / "task.toml"
    config.write_text(
        config.read_text().replace("[environment]", "[environment]\ngpus = 0")
    )
    client = Mock()
    client.info.return_value = {"CpuCfsQuota": cpu_quota, "MemoryLimit": memory_limit}
    coordinator = Mock(side_effect=AssertionError("unexpected coordinator startup"))
    services = production.ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=client,
        bridge_gateway="127.0.0.1",
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=coordinator,
    )
    with pytest.raises(SetupError, match=missing):
        services.run(
            RunRequest(
                task_dir=task,
                agent_name="oracle",
                options=CompileOptions(agent_name="oracle"),
            )
        )
    coordinator.assert_not_called()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    ("cpus", "memory_mb", "info"),
    [
        (4, 8192, {"CpuCfsQuota": True, "MemoryLimit": True}),
        (1, None, {"CpuCfsQuota": True, "MemoryLimit": False}),
        (None, 256, {"CpuCfsQuota": False, "MemoryLimit": True}),
        (None, None, {}),
        (0, 0, {}),
    ],
)
def test_only_positive_requested_limits_need_daemon_support(cpus, memory_mb, info):
    client = Mock()
    client.info.return_value = info
    service = MainServiceConfig(cpus=cpus, memory_mb=memory_mb)
    production._check_docker_resource_limits(client, service)
    assert client.info.call_count == int(bool(cpus or memory_mb))
