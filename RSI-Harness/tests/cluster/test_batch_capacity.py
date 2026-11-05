from __future__ import annotations

import pytest

from rsi_harness.errors import SetupError
from tests.cluster.bluevela.test_adapter import (
    _fake_codex_runtime as _fake_codex_runtime,
)
from tests.cluster.bluevela.test_adapter import _request
from tests.cluster.pbs.test_adapter import _adapter as pbs_adapter
from tests.cluster.pbs.test_adapter import local_profile
from tests.cluster.pjm.test_adapter import _adapter as pjm_adapter
from tests.factories import DEFAULT_TASK_TOML, write_harbor_task


def _plan(tmp_path, name, *, cpus=8, memory_mb=65536, gpus=1, builder=None):
    task = write_harbor_task(
        tmp_path / "task",
        task_toml=(
            DEFAULT_TASK_TOML.replace(
                "gpus = 2",
                f"gpus = {gpus}\ncpus = {cpus}\nmemory_mb = {memory_mb}",
            )
            + "\n[metadata.rsi_harness.verifier]\ngpus = 0\n"
        ),
    )
    profile = local_profile(tmp_path, name)
    if builder:
        profile = profile.model_copy(
            update={"builder": profile.builder.model_copy(update=builder)}
        )
    events = {}
    adapter = (pjm_adapter if name == "genkai" else pbs_adapter)(profile, events)
    request = _request(tmp_path, dry_run=True).model_copy(update={"task_dir": task})
    adapter.run(request)
    return events["dry_run"]


@pytest.mark.parametrize(
    ("name", "cpu_limit", "ram_limit"),
    [("flow2", 36, 221184), ("flow2-node", 144, 884736), ("genkai", 30, 232140)],
)
@pytest.mark.parametrize("oversize", ["cpu", "memory"])
def test_fixed_share_rejects_requests_the_scheduler_would_underallocate(
    tmp_path, name, cpu_limit, ram_limit, oversize
):
    with pytest.raises(SetupError, match="run needs.*allocation provides"):
        _plan(
            tmp_path,
            name,
            cpus=cpu_limit + (oversize == "cpu"),
            memory_mb=ram_limit + (oversize == "memory"),
        )


@pytest.mark.parametrize(
    ("name", "cpus", "memory_mb", "gpus"),
    [
        ("flow2", 36, 221184, 1),
        ("flow2-node", 144, 884736, 1),
        ("genkai", 30, 232140, 1),
        ("genkai", 60, 464280, 2),
        ("genkai", 120, 928560, 4),
    ],
)
def test_supported_task_sizes_keep_their_declared_resources(
    tmp_path, name, cpus, memory_mb, gpus
):
    result = _plan(tmp_path, name, cpus=cpus, memory_mb=memory_mb, gpus=gpus)
    assert result["resources"]["cpu_slots"] == cpus
    assert result["resources"]["memory_mb"] == memory_mb


def test_whole_node_flow2_build_uses_the_share_queues_limit(tmp_path):
    # The run queue allows 144 cores, but its build queue allows only 36.
    with pytest.raises(SetupError, match="build needs 40 CPUs.*queue t2-share"):
        _plan(tmp_path, "flow2-node", builder={"cpu_slots": 40})


@pytest.mark.parametrize("name", ["flow2", "genkai"])
def test_gpu_only_share_queues_explain_cpu_only_requests(tmp_path, name):
    with pytest.raises(SetupError, match="Use a CPU queue/profile"):
        _plan(tmp_path, name, gpus=0)


def test_portable_pbs_keeps_explicit_cpu_ram_requests_without_fixed_capacity(tmp_path):
    result = _plan(tmp_path, "pbs", cpus=200, memory_mb=1048576)
    assert "select=1:ncpus=200:mem=1048576mb:ngpus=1" in result["run_argv"]


def test_a_cached_image_needs_no_build_capacity(tmp_path, monkeypatch):
    from rsi_harness.cluster.bluevela import adapter

    real = adapter.plan_image
    monkeypatch.setattr(
        adapter,
        "plan_image",
        lambda *args: real(*args).model_copy(update={"cache_hit": True}),
    )
    result = _plan(tmp_path, "flow2-node", builder={"cpu_slots": 40})
    assert result["cache_hit"] is True
