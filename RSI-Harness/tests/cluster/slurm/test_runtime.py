from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rsi_harness.cluster.bluevela.allocation import NodeProbe, probe_and_partition
from rsi_harness.cluster.bluevela.engine import run_engine_payload
from rsi_harness.cluster.bluevela.multinode import MultiNodeBroker, stop_command
from rsi_harness.cluster.bluevela.runtime import ApptainerAgentRuntime
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.cluster.slurm.allocation import discover_slurm_inventory
from rsi_harness.errors import InfrastructureError
from tests.cluster.bluevela import test_multinode as shared
from tests.cluster.bluevela.test_engine import _multi_payload


def test_engine_discovers_probes_and_freezes_slurm_work_judge_pools(
    tmp_path, monkeypatch
):
    profile = load_cluster_profile("slurm")
    payload = _multi_payload(tmp_path).model_copy(update={"profile": profile})
    payload.run_plan.paths.root.mkdir(parents=True)
    (payload.run_plan.paths.root / "control").mkdir()
    payload.sif_sha256_path.write_text("d" * 64 + "  task.sif\n")
    calls, captured = [], []
    monkeypatch.delenv("LSB_MCPU_HOSTS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def runner(argv):
        calls.append(argv)
        if argv[0] == "scontrol":
            return subprocess.CompletedProcess(argv, 0, "gpu01\ngpu02\ngpu03\n", "")
        host = argv[argv.index("--nodelist") + 1]
        index = int(host[-2:])
        probe = NodeProbe(
            ipv4=f"10.0.0.{index}",
            cuda_devices=tuple(str(i) for i in range(8)),
            sif_sha256="d" * 64,
            gpfs_visible=True,
            infiniband_visible=True,
            tmp_free_mb=100000,
        )
        return subprocess.CompletedProcess(argv, 0, probe.model_dump_json(), "")

    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.discover_slurm_inventory",
        lambda **kwargs: discover_slurm_inventory(
            **kwargs,
            runner=runner,
            environ={
                "SLURM_JOB_NODELIST": "gpu[01-03]",
                "SLURM_JOB_CPUS_PER_NODE": "64(x3)",
            },
        ),
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.probe_and_partition",
        lambda **kwargs: probe_and_partition(**kwargs, runner=runner),
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.socket.gethostname", lambda: "gpu01"
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.run_native_engine",
        lambda _, plan, *, allocated_pools: captured.append((plan, allocated_pools)),
    )
    run_engine_payload(payload)
    plan, pools = captured[0]
    assert tuple(node.host for node in pools.work) == ("gpu01", "gpu02")
    assert tuple(node.host for node in pools.verifier) == ("gpu03",)
    assert plan.gpu_plan.work.uuids[0] == "gpu01:0"
    assert plan.gpu_plan.judge.uuids[0] == "gpu03:0"
    assert set(plan.gpu_plan.work.uuids).isdisjoint(plan.gpu_plan.judge.uuids)
    assert len(calls) == 4
    assert all(command[:2] == ("srun", "--overlap") for command in calls[1:])
    assert (payload.run_plan.paths.root / "control/ALLOCATED_POOLS.json").is_file()


def test_rank_workers_and_cleanup_use_the_same_slurm_launch_prefix(tmp_path):
    scheduler = load_cluster_profile("slurm").scheduler
    module = shared._module()
    broker = MultiNodeBroker(
        phase="work",
        run_id="slurm-run",
        nodes=shared._nodes("work", 2),
        pool_digest="a" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=shared._worker(module, tmp_path),
        remote_binary=scheduler.remote_binary,
        remote_host_flag=scheduler.remote_host_flag,
        remote_args=scheduler.remote_args,
    )
    broker.initialize()
    reserved = broker.reserve(
        shared._request(module, run_id="slurm-run", pool_digest="a" * 64)
    )
    prefix = ("srun", *scheduler.remote_args, "--nodelist")
    for command in reserved.commands:
        assert command[: len(prefix)] == prefix
        assert "--nodes=1" in command and "--ntasks=1" in command
        stopped = stop_command(
            command,
            remote_binary="srun",
            remote_host_flag="--nodelist",
            remote_args=scheduler.remote_args,
        )
        action = command.index("rsi_harness.cluster.bluevela.remote_worker") + 1
        assert stopped == (*command[:action], "stop", *command[action + 1 :])
        with pytest.raises(InfrastructureError, match="unsafe"):
            stop_command(command)


def test_judge_timeout_stops_remote_controller_with_slurm_arguments(
    tmp_path, monkeypatch
):
    profile = load_cluster_profile("slurm")
    command = (
        "srun",
        *profile.scheduler.remote_args,
        "--nodelist",
        "gpu03",
        sys.executable,
        "-m",
        "rsi_harness.cluster.bluevela.judge_controller",
        "run",
        "--control",
        str(tmp_path / "controller.json"),
    )
    process = Mock(returncode=-15)
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(command, 1),
        (b"timeout", None),
    ]
    cleanup = Mock()
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.subprocess.Popen", lambda *a, **k: process
    )
    monkeypatch.setattr("rsi_harness.cluster.bluevela.runtime.subprocess.run", cleanup)
    result = ApptainerAgentRuntime._run_host_controller(
        SimpleNamespace(profile=profile),
        command,
        timeout_seconds=1,
        output_path=tmp_path / "output.txt",
    )
    assert result.timed_out
    assert cleanup.call_args.args[0] == (*command[:-3], "stop", *command[-2:])
    process.terminate.assert_called_once()
