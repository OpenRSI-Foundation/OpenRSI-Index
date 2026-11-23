import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import CompileOptions, MainServiceConfig, RunRequest
from rsi_harness.runtime import production
from rsi_loop.harness.config import RSILoopConfig


def _services(tmp_path, monkeypatch, info, *, root=True, events=None):
    # Model a local Linux daemon; the BusyBox task requests 1 CPU and 256 MiB.
    monkeypatch.setattr(production.sys, "platform", "linux")
    monkeypatch.setattr(
        production, "container_netns_firewall_selected", lambda: not root
    )
    fixture = Path(__file__).resolve().parents[1] / "fixtures/tasks/harbor-env-busybox"
    task = tmp_path / "task"
    shutil.copytree(fixture, task)
    (task / "environment").mkdir()
    config = task / "task.toml"
    config.write_text(
        config.read_text().replace("[environment]", "[environment]\ngpus = 0")
    )
    client = Mock()
    client.info.return_value = info
    coordinator = Mock(side_effect=AssertionError("coordinator started"))
    services = production.ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=client,
        bridge_gateway="127.0.0.1",
        rsi_loop_config=RSILoopConfig(),
        coordinator_factory=coordinator,
        event_callback=(lambda name, value: events.append((name, value)))
        if events is not None
        else None,
    )
    request = RunRequest(
        task_dir=task, agent_name="oracle", options=CompileOptions(agent_name="oracle")
    )
    return services, request, coordinator


def test_a_daemon_without_cpu_quota_is_refused_before_run_state(tmp_path, monkeypatch):
    """Docker would refuse the task's CPU limit at container creation, after
    image preparation; refuse before any run state instead."""
    services, request, coordinator = _services(
        tmp_path, monkeypatch, {"CpuCfsQuota": False, "MemoryLimit": True}
    )
    with pytest.raises(SetupError, match="CpuCfsQuota"):
        services.run(request)
    coordinator.assert_not_called()
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "logs").exists()


def test_a_daemon_without_memory_limits_runs_and_warns(tmp_path, monkeypatch):
    """Docker drops a memory limit it cannot enforce and runs the container:
    the run goes on (here, to the coordinator) with a warning."""
    events = []
    services, request, coordinator = _services(
        tmp_path,
        monkeypatch,
        {"CpuCfsQuota": True, "MemoryLimit": False},
        events=events,
    )
    with pytest.raises(AssertionError, match="coordinator started"):
        services.run(request)
    assert ("docker_memory_limit_unsupported", 256) in events


@pytest.mark.parametrize(
    ("cpus", "memory_mb", "holders", "info", "dropped"),
    [
        (4, 8192, False, {"CpuCfsQuota": True, "MemoryLimit": True}, False),
        (4, 8192, False, {"CpuCfsQuota": True, "MemoryLimit": False}, True),
        (1, None, False, {"CpuCfsQuota": True, "MemoryLimit": False}, False),
        (None, 256, False, {"CpuCfsQuota": False, "MemoryLimit": True}, False),
        (None, 256, False, {"CpuCfsQuota": False}, True),
        (None, None, False, {}, False),
        (0, 0, False, {}, False),
        # A non-root controller's helpers carry a CPU limit of their own.
        (None, None, True, {"CpuCfsQuota": True}, False),
    ],
)
def test_only_the_limits_a_run_uses_are_asked_about(
    cpus, memory_mb, holders, info, dropped
):
    client = Mock()
    client.info.return_value = info
    service = MainServiceConfig(cpus=cpus, memory_mb=memory_mb)
    assert (
        production._check_docker_resource_limits(client, service, holders=holders)
        is dropped
    )
    assert client.info.call_count == int(bool(cpus or memory_mb or holders))


@pytest.mark.parametrize(
    ("cpus", "holders", "match"),
    [(2, False, "task's CPU limit"), (None, True, "non-root controller's helper")],
)
def test_a_cpu_limit_the_daemon_would_refuse_is_refused(cpus, holders, match):
    client = Mock()
    client.info.return_value = {"CpuCfsQuota": False, "MemoryLimit": True}
    service = MainServiceConfig(cpus=cpus, memory_mb=None)
    with pytest.raises(SetupError, match=match):
        production._check_docker_resource_limits(client, service, holders=holders)


def test_the_cli_warns_when_the_memory_limit_is_dropped(capsys):
    from rsi_harness.cli import _RunConsole

    _RunConsole()("docker_memory_limit_unsupported", 8192)
    out = capsys.readouterr().out
    assert "Warning:" in out and "8192 MiB limit is not enforced" in out
