from __future__ import annotations

import importlib
import io
import json
import os
import threading
import time
from pathlib import Path

import pytest

from rsi_harness.errors import SetupError


def _module():
    source = (
        Path(__file__).resolve().parents[3]
        / "src/rsi_harness/cluster/bluevela/torchrun_shim.py"
    )
    assert source.is_file(), "transparent Blue Vela torchrun shim is missing"
    return importlib.import_module("rsi_harness.cluster.bluevela.torchrun_shim")


@pytest.mark.parametrize(
    "argv",
    (
        (
            "--nnodes",
            "4",
            "--nproc-per-node",
            "8",
            "train.py",
            "--arg",
            "value",
        ),
        (
            "--nnodes=4",
            "--nproc_per_node=8",
            "--no-python",
            "/task-tools/train.sh",
        ),
    ),
)
def test_fixed_torchrun_arguments_select_a_four_node_subpool(
    argv: tuple[str, ...],
) -> None:
    shim = _module()

    invocation = shim.parse_torchrun_invocation(
        argv,
        expected_local_world_size=8,
    )

    assert invocation.node_count == 4
    assert invocation.local_world_size == 8
    assert invocation.distributed is True
    assert not any(item.startswith("--node-rank") for item in invocation.argv)
    assert not any(item.startswith("--master-addr") for item in invocation.argv)


def test_one_node_torchrun_selects_a_one_node_phase_subpool() -> None:
    shim = _module()

    invocation = shim.parse_torchrun_invocation(
        ("--nnodes", "1", "--nproc-per-node", "8", "train.py"),
        expected_local_world_size=8,
    )

    assert invocation.node_count == 1
    assert invocation.distributed is True
    assert invocation.argv[-1] == "train.py"


@pytest.mark.parametrize(
    ("argv", "match"),
    (
        (("--nnodes", "2:4", "train.py"), "fixed integer"),
        (("--nnodes", "0", "train.py"), "positive"),
        (("--nnodes", "4", "--node-rank", "0", "train.py"), "node rank"),
        (
            ("--nnodes", "4", "--master-addr", "host", "train.py"),
            "rendezvous",
        ),
        (
            ("--nnodes", "4", "--rdzv-endpoint=host:2000", "train.py"),
            "rendezvous",
        ),
        (("--nnodes", "4", "--nproc-per-node", "4", "train.py"), "8"),
        (("--nnodes", "4"), "training entrypoint"),
    ),
)
def test_task_cannot_supply_cluster_topology(
    argv: tuple[str, ...],
    match: str,
) -> None:
    shim = _module()

    with pytest.raises(SetupError, match=match):
        shim.parse_torchrun_invocation(argv, expected_local_world_size=8)


def test_forwarded_environment_excludes_scheduler_runtime_and_secrets() -> None:
    shim = _module()
    environment = {
        "TRAINING_PROFILE": "E5",
        "WANDB_MODE": "offline",
        "LSB_JOBID": "123",
        "LSF_ENVDIR": "/secret",
        "APPTAINER_CONTAINER": "/image.sif",
        "SINGULARITY_NAME": "image.sif",
        "OPENAI_API_KEY": "secret",
        "ANTHROPIC_API_KEY": "secret",
        "RSI_SUBMIT_TOKEN": "secret",
        "HOME": "/home/agent",
    }

    forwarded = shim.forwarded_environment(environment)

    assert forwarded == {
        "TRAINING_PROFILE": "E5",
        "WANDB_MODE": "offline",
    }


def test_forwarded_environment_preserves_tokenizer_configuration() -> None:
    shim = _module()
    environment = {
        "LM_EVAL_TOKENIZER_MODEL": "/rsi-data/tokenizer-snapshot",
        "LM_EVAL_TOKENIZER_REVISION": "pinned-revision",
        "DATA_TOKENIZER_MODEL": "meta-llama/Llama-3.1-8B",
        "TOKENIZER_ACCESS_TOKEN": "secret",
        "RSI_SUBMIT_TOKEN": "secret",
    }

    forwarded = shim.forwarded_environment(environment)

    assert forwarded == {
        "LM_EVAL_TOKENIZER_MODEL": "/rsi-data/tokenizer-snapshot",
        "LM_EVAL_TOKENIZER_REVISION": "pinned-revision",
        "DATA_TOKENIZER_MODEL": "meta-llama/Llama-3.1-8B",
    }


def test_client_publishes_an_owner_only_atomic_closed_request(
    tmp_path: Path,
) -> None:
    shim = _module()
    root = tmp_path / "protocol"
    (root / "requests").mkdir(parents=True)
    invocation = shim.parse_torchrun_invocation(
        ("--nnodes", "2", "--nproc-per-node", "8", "train.py"),
        expected_local_world_size=8,
    )

    request = shim.publish_request(
        root,
        invocation,
        ready={
            "run_id": "run-1",
            "pool_digest": "a" * 64,
            "pool_size": 2,
            "local_world_size": 8,
        },
        environment={"PROFILE": "E3", "LSB_JOBID": "forged"},
        working_directory="/workspace/project",
        request_id="f" * 32,
    )

    path = root / "requests" / f"{request.request_id}.json"
    assert path.is_file()
    assert path.stat().st_mode & 0o777 == 0o600
    assert not tuple((root / "requests").glob("*.tmp.*"))
    payload = json.loads(path.read_text())
    assert payload["environment"] == {"PROFILE": "E3"}
    assert "host" not in payload
    assert "phase" not in payload


def test_client_rejects_a_request_larger_than_the_current_phase_pool(
    tmp_path: Path,
) -> None:
    shim = _module()
    (tmp_path / "requests").mkdir()
    invocation = shim.parse_torchrun_invocation(
        ("--nnodes", "3", "train.py"), expected_local_world_size=8
    )

    with pytest.raises(SetupError, match="phase pool"):
        shim.publish_request(
            tmp_path,
            invocation,
            ready={
                "run_id": "run-1",
                "pool_digest": "a" * 64,
                "pool_size": 2,
                "local_world_size": 8,
            },
            environment=os.environ,
            working_directory="/workspace",
            request_id="e" * 32,
        )


def test_client_emits_rank_output_in_deterministic_order_and_returns_status(
    tmp_path: Path,
) -> None:
    shim = _module()
    multinode = importlib.import_module(
        "rsi_harness.cluster.bluevela.multinode"
    )
    results = tmp_path / "results"
    heartbeats = tmp_path / "heartbeats"
    results.mkdir()
    heartbeats.mkdir()
    request_id = "a" * 32
    (results / f"{request_id}.rank-0001.out").write_bytes(b"second\n")
    (results / f"{request_id}.rank-0000.out").write_bytes(b"first\n")
    result = multinode.MultiNodeResult(
        request_id=request_id,
        ranks=(
            multinode.RankResult(pool_rank=0, node_rank=0, returncode=0),
            multinode.RankResult(pool_rank=1, node_rank=1, returncode=7),
        ),
    )
    (results / f"{request_id}.json").write_text(result.model_dump_json())
    output = io.BytesIO()

    returncode = shim.wait_for_result(
        tmp_path,
        request_id=request_id,
        node_count=2,
        output=output,
        timeout_seconds=1,
        poll_interval=0.001,
    )

    assert output.getvalue() == b"first\nsecond\n"
    assert returncode == 7
    assert not (heartbeats / request_id).exists()


def test_one_node_cli_publishes_to_the_phase_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = _module()
    monkeypatch.setattr(shim.signal, "signal", lambda *_args: None)
    root = tmp_path / "phase-broker"
    for name in ("requests", "results", "heartbeats", "cancel"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "READY.json").write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "pool_digest": "a" * 64,
                "pool_size": 2,
                "local_world_size": 8,
            }
        )
    )
    returned: list[int] = []

    def reject_local_execute(_path: str, _argv: tuple[str, ...]) -> None:
        raise AssertionError("one-node torchrun bypassed the phase broker")

    thread = threading.Thread(
        target=lambda: returned.append(
            shim.run(
                ("--nnodes", "1", "--nproc-per-node", "8", "train.py"),
                environment={
                    "RSI_LOCAL_WORLD_SIZE": "8",
                    "RSI_MULTINODE_ROOT": str(root),
                },
                working_directory="/workspace",
                output=io.BytesIO(),
                execute=reject_local_execute,
            )
        )
    )
    thread.start()
    deadline = time.monotonic() + 2
    requests = list((root / "requests").glob("*.json"))
    while not requests and time.monotonic() < deadline:
        time.sleep(0.01)
        requests = list((root / "requests").glob("*.json"))
    assert len(requests) == 1
    payload = json.loads(requests[0].read_text())
    assert payload["node_count"] == 1
    request_id = payload["request_id"]
    (root / "results" / f"{request_id}.rank-0000.out").write_bytes(b"ok\n")
    (root / "results" / f"{request_id}.json").write_text(
        json.dumps(
            {
                "request_id": request_id,
                "ranks": [
                    {"pool_rank": 1, "node_rank": 0, "returncode": 0}
                ],
                "cancelled": False,
                "timed_out": False,
            }
        )
    )
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert returned == [0]


def test_installed_torchrun_shim_has_an_executable_script_entrypoint() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "src/rsi_harness/cluster/bluevela/torchrun_shim.py"
    )
    text = source.read_text()

    assert text.startswith("#!/usr/bin/env python3\n")
    assert 'if __name__ == "__main__":' in text
