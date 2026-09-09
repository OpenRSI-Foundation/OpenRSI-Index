from __future__ import annotations

import importlib
import io
import json
import os
import sys
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from rsi_harness.cluster.bluevela.allocation import AllocatedNode
from rsi_harness.errors import InfrastructureError


def _module():
    source = (
        Path(__file__).resolve().parents[3]
        / "src/rsi_harness/cluster/bluevela/multinode.py"
    )
    assert source.is_file(), "Blue Vela phase broker is missing"
    return importlib.import_module("rsi_harness.cluster.bluevela.multinode")


def _remote_module():
    source = (
        Path(__file__).resolve().parents[3]
        / "src/rsi_harness/cluster/bluevela/remote_worker.py"
    )
    assert source.is_file(), "fixed-shape Blue Vela remote worker is missing"
    return importlib.import_module("rsi_harness.cluster.bluevela.remote_worker")


def _judge_module():
    source = (
        Path(__file__).resolve().parents[3]
        / "src/rsi_harness/cluster/bluevela/judge_controller.py"
    )
    assert source.is_file(), "GPU-free Blue Vela Judge controller is missing"
    return importlib.import_module("rsi_harness.cluster.bluevela.judge_controller")


def _nodes(prefix: str, count: int) -> tuple[AllocatedNode, ...]:
    return tuple(
        AllocatedNode(
            host=f"{prefix}-{index}",
            slots=8,
            ipv4=f"10.0.{0 if prefix == 'work' else 1}.{index + 1}",
            cuda_devices=tuple(f"GPU-{prefix}-{index}-{gpu}" for gpu in range(8)),
        )
        for index in range(count)
    )


def _worker(multinode, tmp_path: Path, *, network_mode: str = "public"):
    return multinode.RemoteWorkerTemplate(
        apptainer_binary=Path("/usr/bin/apptainer"),
        sif_path=tmp_path / "image.sif",
        sif_sha256="d" * 64,
        container_python="/opt/venv/bin/python",
        container_workdir="/workspace/project",
        temp_root=tmp_path / "node-tmp",
        network_mode=network_mode,
        binds=(
            multinode.WorkerBind(
                source=tmp_path / "workspace",
                target="/workspace",
                read_only=False,
            ),
        ),
        environment={"NCCL_DEBUG": "INFO"},
    )


def test_subpool_leases_are_atomic_disjoint_and_reusable() -> None:
    multinode = _module()
    allocator = multinode.SubpoolAllocator(_nodes("work", 6))

    first = allocator.acquire(4)
    second = allocator.acquire(2)

    assert first.pool_ranks == (0, 1, 2, 3)
    assert second.pool_ranks == (4, 5)
    assert set(first.pool_ranks).isdisjoint(second.pool_ranks)
    with pytest.raises(InfrastructureError, match="free nodes"):
        allocator.acquire(1)

    allocator.release(first.lease_id)
    replacement = allocator.acquire(3)
    assert replacement.pool_ranks == (0, 1, 2)


def test_allocator_rejects_unknown_or_double_release() -> None:
    multinode = _module()
    allocator = multinode.SubpoolAllocator(_nodes("work", 2))
    lease = allocator.acquire(1)

    allocator.release(lease.lease_id)

    with pytest.raises(InfrastructureError, match="unknown subpool lease"):
        allocator.release(lease.lease_id)


def test_rank_environments_are_relative_to_only_the_selected_phase_subpool() -> None:
    multinode = _module()
    selected = _nodes("judge", 3)

    environments = multinode.rank_environments(
        selected,
        pool_size=5,
        pool_ranks=(1, 3, 4),
        local_world_size=8,
        request_id="a" * 32,
    )

    assert tuple(item["RSI_NODE_RANK"] for item in environments) == (
        "0",
        "1",
        "2",
    )
    assert tuple(item["RSI_POOL_RANK"] for item in environments) == (
        "1",
        "3",
        "4",
    )
    assert {item["RSI_MASTER_ADDR"] for item in environments} == {
        selected[0].ipv4
    }
    assert {item["RSI_MASTER_PORT"] for item in environments}.__len__() == 1
    assert all(item["RSI_NUM_NODES"] == "3" for item in environments)
    assert all(item["RSI_POOL_SIZE"] == "5" for item in environments)
    assert all(item["RSI_LOCAL_WORLD_SIZE"] == "8" for item in environments)


def test_work_and_judge_allocators_cannot_cross_pool_boundaries() -> None:
    multinode = _module()
    work = multinode.SubpoolAllocator(_nodes("work", 2))
    judge = multinode.SubpoolAllocator(_nodes("judge", 1))

    work_lease = work.acquire(2)
    judge_lease = judge.acquire(1)

    assert all(node.host.startswith("work-") for node in work_lease.nodes)
    assert all(node.host.startswith("judge-") for node in judge_lease.nodes)


def _request(multinode, **updates):
    values = {
        "request_id": "b" * 32,
        "run_id": "run-123",
        "pool_digest": "c" * 64,
        "node_count": 2,
        "local_world_size": 8,
        "argv": (
            "--nnodes",
            "2",
            "--nproc-per-node",
            "8",
            "train.py",
        ),
        "working_directory": "/workspace/project",
        "environment": {"TRAINING_PROFILE": "E3"},
    }
    values.update(updates)
    return multinode.MultiNodeRequest(**values)


def test_closed_request_model_cannot_name_hosts_or_phase() -> None:
    multinode = _module()

    with pytest.raises(ValidationError, match="host"):
        multinode.MultiNodeRequest(
            **_request(multinode).model_dump(),
            host="judge-0",
        )
    with pytest.raises(ValidationError, match="phase"):
        multinode.MultiNodeRequest(
            **_request(multinode).model_dump(),
            phase="verifier",
        )


def test_broker_freezes_authority_and_builds_only_fixed_remote_commands(
    tmp_path: Path,
) -> None:
    multinode = _module()
    client = tmp_path / "client"
    authority = tmp_path / "authority"
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 3),
        pool_digest="c" * 64,
        root=client,
        authority_root=authority,
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
        remote_binary="/site/bin/remote-launch",
        remote_host_flag="--host",
    )
    broker.initialize()

    ready = json.loads((client / "READY.json").read_text())
    assert ready == {
        "local_world_size": 8,
        "pool_digest": "c" * 64,
        "pool_size": 3,
        "run_id": "run-123",
    }
    assert client.stat().st_mode & 0o777 == 0o700
    assert all(
        (client / name).stat().st_mode & 0o777 == 0o700
        for name in ("cancel", "heartbeats", "requests", "results")
    )

    reserved = broker.reserve(_request(multinode))

    assert reserved.lease.pool_ranks == (0, 1)
    assert reserved.control_path.parent == authority / "requests"
    assert reserved.control_path.stat().st_mode & 0o777 == 0o600
    assert tuple(item[0:3] for item in reserved.commands) == (
        ("/site/bin/remote-launch", "--host", "work-0"),
        ("/site/bin/remote-launch", "--host", "work-1"),
    )
    for node_rank, command in enumerate(reserved.commands):
        assert command[3:6] == (
            sys.executable,
            "-m",
            "rsi_harness.cluster.bluevela.remote_worker",
        )
        assert command[6] == "run"
        assert command[-6:] == (
            "--request",
            "b" * 32,
            "--pool-rank",
            str(node_rank),
            "--node-rank",
            str(node_rank),
        )
        assert "judge-" not in " ".join(command)

    with pytest.raises(InfrastructureError, match="active remote requests"):
        broker.require_idle()
    broker.release(reserved)
    broker.require_idle()


@pytest.mark.parametrize(
    "updates",
    (
        {"run_id": "wrong"},
        {"pool_digest": "d" * 64},
        {"node_count": 4, "argv": ("--nnodes", "4", "train.py")},
        {"local_world_size": 4},
        {"environment": {"LSB_JOBID": "123"}},
    ),
)
def test_broker_rejects_forged_or_oversized_requests_without_a_lease(
    tmp_path: Path,
    updates: dict[str, object],
) -> None:
    multinode = _module()
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 3),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )
    broker.initialize()

    with pytest.raises(InfrastructureError):
        broker.reserve(_request(multinode, **updates))

    broker.require_idle()


def test_remote_worker_builds_only_the_broker_owned_apptainer_shape(
    tmp_path: Path,
    monkeypatch,
) -> None:
    multinode = _module()
    remote = _remote_module()
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 2),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )
    broker.initialize()
    reserved = broker.reserve(
        _request(
            multinode,
            environment={
                "PROMPT_COMMAND": 'PS1="Apptainer> "; unset PROMPT_COMMAND',
                "TRAINING_PROFILE": 'E3,"quoted"',
            },
        )
    )
    control = remote.load_control(reserved.control_path)
    node_tmp = remote.request_tmp_dir(control, pool_rank=0)

    command = remote.build_apptainer_command(
        control,
        request_id="b" * 32,
        pool_rank=0,
        node_rank=0,
        current_host="work-0",
        local_tmp=node_tmp,
    )

    assert command[:2] == ("/usr/bin/apptainer", "exec")
    assert "--cleanenv" in command
    assert "--no-eval" in command
    assert "--env" not in command
    assert "--nv" in command
    assert "--containall" in command
    assert f"{tmp_path / 'workspace'}:/workspace" in command
    assert f"{node_tmp}:/tmp" in command
    monkeypatch.setenv("APPTAINER_BIND", "/host-controlled")
    monkeypatch.setenv("APPTAINERENV_FORGED", "forged")
    environment = remote.build_apptainer_environment(
        control,
        request_id="b" * 32,
        pool_rank=0,
        node_rank=0,
        current_host="work-0",
    )
    assert "APPTAINER_BIND" not in environment
    assert "APPTAINERENV_FORGED" not in environment
    assert environment["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == ",".join(
        reserved.lease.nodes[0].cuda_devices
    )
    assert environment["APPTAINERENV_PROMPT_COMMAND"] == (
        'PS1="Apptainer> "; unset PROMPT_COMMAND'
    )
    assert environment["APPTAINERENV_TRAINING_PROFILE"] == 'E3,"quoted"'
    python_offset = command.index("/opt/venv/bin/python")
    distributed = command[python_offset:]
    assert distributed[:3] == (
        "/opt/venv/bin/python",
        "-m",
        "torch.distributed.run",
    )
    assert "--nnodes=2" in distributed
    assert "--nproc-per-node=8" in distributed
    assert "--node-rank=0" in distributed
    assert any(item.startswith("--master-addr=10.0.0.1") for item in distributed)
    assert sum(item.startswith("--nnodes") for item in distributed) == 1
    assert sum(item.startswith("--nproc") for item in distributed) == 1
    assert distributed[-1] == "train.py"
    assert node_tmp.is_relative_to(tmp_path / "node-tmp" / "rsi-harness")


def test_remote_worker_allows_a_frozen_image_working_directory(
    tmp_path: Path,
) -> None:
    """A task may launch torchrun from an immutable directory in its SIF."""
    multinode = _module()
    remote = _remote_module()
    broker = multinode.MultiNodeBroker(
        phase="verifier",
        run_id="run-123",
        nodes=_nodes("judge", 2),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )
    broker.initialize()
    control = remote.load_control(
        broker.reserve(
            _request(multinode, working_directory="/opt/datacomp")
        ).control_path
    )

    command = remote.build_apptainer_command(
        control,
        request_id="b" * 32,
        pool_rank=0,
        node_rank=0,
        current_host="judge-0",
        local_tmp=remote.request_tmp_dir(control, pool_rank=0),
    )

    cwd_index = command.index("--cwd")
    assert command[cwd_index + 1] == "/opt/datacomp"


def test_no_network_policy_remote_rank_still_inherits_host_network_for_rendezvous(
    tmp_path: Path,
) -> None:
    """Blue Vela cannot isolate a rank from the network it needs for torchrun."""
    multinode = _module()
    remote = _remote_module()
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 2),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(
            multinode,
            tmp_path,
            network_mode="no-network",
        ),
    )
    broker.initialize()
    control = remote.load_control(
        broker.reserve(_request(multinode)).control_path
    )

    command = remote.build_apptainer_command(
        control,
        request_id="b" * 32,
        pool_rank=0,
        node_rank=0,
        current_host="work-0",
        local_tmp=remote.request_tmp_dir(control, pool_rank=0),
    )

    assert control.worker.network_mode == "no-network"
    assert "--net" not in command
    assert "--network" not in command
    assert "--hostname" not in command
    assert "--master-addr=10.0.0.1" in command


def test_judge_controller_uses_argv_safe_clean_apptainer_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    multinode = _module()
    judge = _judge_module()
    control = judge.JudgeControllerControl(
        request_id="e" * 32,
        run_id="run-123",
        host="judge-0",
        worker=_worker(multinode, tmp_path),
        broker_root=tmp_path / "broker",
        command=("/bin/bash", "/tests/test.sh"),
        environment={"VERIFIER_PROFILE": 'judge,"quoted"'},
        local_world_size=8,
    )
    local_tmp = judge.controller_tmp_dir(control)

    command = judge.build_apptainer_command(
        control,
        current_host="judge-0",
        local_tmp=local_tmp,
    )

    assert "--cleanenv" in command
    assert "--no-eval" in command
    assert "--env" not in command
    monkeypatch.setenv("APPTAINER_BIND", "/host-controlled")
    environment = judge.build_apptainer_environment(control)
    assert "APPTAINER_BIND" not in environment
    assert environment["APPTAINERENV_VERIFIER_PROFILE"] == 'judge,"quoted"'
    assert environment["APPTAINERENV_RSI_MULTINODE_ROOT"] == (
        "/run/rsi-harness/torchrun"
    )
    assert environment["APPTAINERENV_PREPEND_PATH"] == "/usr/local/bin"


@pytest.mark.parametrize(
    ("current_host", "pool_rank", "node_rank", "match"),
    (
        ("judge-0", 0, 0, "host"),
        ("work-0", 1, 0, "rank"),
        ("work-0", 0, 1, "rank"),
    ),
)
def test_remote_worker_rejects_identity_outside_frozen_control(
    tmp_path: Path,
    current_host: str,
    pool_rank: int,
    node_rank: int,
    match: str,
) -> None:
    multinode = _module()
    remote = _remote_module()
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 2),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )
    broker.initialize()
    control = remote.load_control(
        broker.reserve(_request(multinode)).control_path
    )

    with pytest.raises(InfrastructureError, match=match):
        remote.build_apptainer_command(
            control,
            request_id="b" * 32,
            pool_rank=pool_rank,
            node_rank=node_rank,
            current_host=current_host,
            local_tmp=remote.request_tmp_dir(control, pool_rank=0),
        )


def test_broker_executes_all_ranks_publishes_fsynced_result_and_releases(
    tmp_path: Path,
) -> None:
    multinode = _module()
    commands: list[tuple[str, ...]] = []
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 2),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )
    broker.initialize()

    def rank_runner(command, cancelled):
        assert not cancelled.is_set()
        commands.append(command)
        node_rank = int(command[-1])
        return 0, f"rank-{node_rank}\n".encode()

    result = broker.execute(_request(multinode), rank_runner=rank_runner)

    assert len(commands) == 2
    assert result == multinode.MultiNodeResult(
        request_id="b" * 32,
        ranks=(
            multinode.RankResult(pool_rank=0, node_rank=0, returncode=0),
            multinode.RankResult(pool_rank=1, node_rank=1, returncode=0),
        ),
    )
    status = tmp_path / "client/results" / ("b" * 32 + ".json")
    assert multinode.MultiNodeResult.model_validate_json(
        status.read_text()
    ) == result
    first_output = tmp_path / "client/results" / ("b" * 32 + ".rank-0000.out")
    second_output = tmp_path / "client/results" / ("b" * 32 + ".rank-0001.out")
    assert first_output.read_bytes() == b"rank-0\n"
    assert second_output.read_bytes() == b"rank-1\n"
    broker.require_idle()


def test_rank_runner_does_not_require_descendant_pipe_eof_after_launcher_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A remote descendant may retain stdout after its launcher has exited."""
    multinode = _module()
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 1),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )
    read_fd, write_fd = os.pipe()

    class ExitedLauncher:
        def __init__(self) -> None:
            self.stdout = os.fdopen(read_fd, "rb", buffering=0)
            self.returncode = 0

        def poll(self) -> int:
            return self.returncode

        def terminate(self) -> None:
            raise AssertionError("an exited launcher must not be terminated")

        def kill(self) -> None:
            raise AssertionError("pipe EOF is not process failure")

    class NoDrainThread:
        """Keep the legacy blocking-reader implementation from waiting 30s."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def join(self, timeout=None) -> None:
            pass

        def is_alive(self) -> bool:
            return True

    launcher = ExitedLauncher()
    monkeypatch.setattr(multinode.subprocess, "Popen", lambda *a, **k: launcher)
    monkeypatch.setattr(multinode.threading, "Thread", NoDrainThread)
    monkeypatch.setattr(
        multinode,
        "_OUTPUT_DRAIN_GRACE_SECONDS",
        0.01,
        raising=False,
    )
    try:
        returncode, output = broker._default_rank_runner(
            ("remote-launch",), threading.Event()
        )
    finally:
        os.close(write_fd)
        launcher.stdout.close()

    assert returncode == 0
    assert output == b""


def test_broker_cancellation_stops_exact_request_and_releases_subpool(
    tmp_path: Path,
) -> None:
    multinode = _module()
    broker = multinode.MultiNodeBroker(
        phase="work",
        run_id="run-123",
        nodes=_nodes("work", 2),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )
    broker.initialize()
    started = threading.Event()

    def rank_runner(command, cancelled):
        started.set()
        assert cancelled.wait(timeout=2)
        stop = multinode.stop_command(command)
        assert stop[6] == "stop"
        assert stop[2] in {"work-0", "work-1"}
        return 143, b"cancelled\n"

    result: list[object] = []
    thread = threading.Thread(
        target=lambda: result.append(
            broker.execute(_request(multinode), rank_runner=rank_runner)
        )
    )
    thread.start()
    assert started.wait(timeout=2)
    (tmp_path / "client/cancel" / ("b" * 32)).touch()
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert result[0].cancelled is True
    assert all(item.returncode == 143 for item in result[0].ranks)
    broker.require_idle()


def test_private_file_protocol_runs_without_any_additional_user_cli(
    tmp_path: Path,
) -> None:
    multinode = _module()
    shim = importlib.import_module(
        "rsi_harness.cluster.bluevela.torchrun_shim"
    )
    broker = multinode.MultiNodeBroker(
        phase="verifier",
        run_id="run-123",
        nodes=_nodes("judge", 2),
        pool_digest="c" * 64,
        root=tmp_path / "client",
        authority_root=tmp_path / "authority",
        local_world_size=8,
        worker_template=_worker(multinode, tmp_path),
    )

    def rank_runner(command, cancelled):
        assert not cancelled.is_set()
        return 0, f"judge-rank-{command[-1]}\n".encode()

    broker.start(rank_runner=rank_runner)
    try:
        invocation = shim.parse_torchrun_invocation(
            ("--nnodes", "2", "--nproc-per-node", "8", "judge.py"),
            expected_local_world_size=8,
        )
        ready = json.loads((broker.root / "READY.json").read_text())
        request = shim.publish_request(
            broker.root,
            invocation,
            ready=ready,
            environment={"EVAL_PROFILE": "terminal"},
            working_directory="/workspace/project",
            request_id="9" * 32,
        )
        output = io.BytesIO()

        returncode = shim.wait_for_result(
            broker.root,
            request_id=request.request_id,
            node_count=2,
            output=output,
            timeout_seconds=2,
            poll_interval=0.001,
        )

        assert returncode == 0
        assert output.getvalue() == b"judge-rank-0\njudge-rank-1\n"
        broker.require_idle()
    finally:
        broker.stop()
