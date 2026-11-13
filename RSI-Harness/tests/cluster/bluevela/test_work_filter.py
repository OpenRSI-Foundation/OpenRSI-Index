"""Every cluster Work exec runs without file watches: rsi-box's Work layer
in harness mode, its filter alone otherwise, and on remote Work ranks."""

from __future__ import annotations

from pathlib import Path

import pytest

from rsi_harness.errors import InfrastructureError
from tests.cluster.bluevela.test_isolation_modes import BOX, _runtime
from tests.cluster.bluevela.test_multinode import (
    _module,
    _nodes,
    _remote_module,
    _request,
    _worker,
)


@pytest.mark.parametrize("network_isolation", ["namespace", "seccomp"])
@pytest.mark.parametrize(
    ("process_namespaces", "layer"),
    [("apptainer", "--work-filter"), ("harness", "--work")],
)
def test_every_work_exec_gets_a_watch_filter(
    tmp_path, monkeypatch, network_isolation, process_namespaces, layer
):
    runtime = _runtime(
        tmp_path,
        monkeypatch,
        network_isolation=network_isolation,
        process_namespaces=process_namespaces,
    )
    for network_mode in ("public", "no-network"):
        assert runtime._box_layer("work", network_mode) == [*BOX, layer, "--"]
    # The Judge never gets Work's filter.
    for network_mode in ("public", "no-network"):
        judge = runtime._box_layer("judge", network_mode) or []
        assert not {"--work", "--work-filter"} & set(judge)
    assert runtime._box_layer(None, "public") is None


def _control(tmp_path: Path, phase: str, *, bound: bool = True):
    multinode = _module()
    worker = _worker(multinode, tmp_path)
    if not bound:
        worker = worker.model_copy(update={"binds": worker.binds[:1]})
    broker = multinode.MultiNodeBroker(
        phase=phase,
        run_id="run-123",
        nodes=_nodes(phase, 2),
        pool_digest="c" * 64,
        root=tmp_path / phase / "client",
        authority_root=tmp_path / phase / "authority",
        local_world_size=8,
        worker_template=worker,
    )
    broker.initialize()
    remote = _remote_module()
    control = remote.load_control(broker.reserve(_request(multinode)).control_path)
    return remote, control


def _command(remote, control):
    return remote.build_apptainer_command(
        control,
        request_id="b" * 32,
        pool_rank=0,
        node_rank=0,
        current_host=control.lease.nodes[0].host,
        local_tmp=remote.request_tmp_dir(control, pool_rank=0),
    )


def test_only_remote_work_ranks_run_under_the_filter(tmp_path):
    remote, work = _control(tmp_path, "work")
    assert "--work-filter" in _command(remote, work)
    remote, verifier = _control(tmp_path, "verifier")
    command = _command(remote, verifier)
    assert "--work-filter" not in command
    sif = command.index(str(tmp_path / "image.sif"))
    assert command[sif + 1 : sif + 3] == ("/opt/venv/bin/python", "-m")


def test_a_remote_work_rank_without_rsi_box_does_not_start(tmp_path):
    remote, control = _control(tmp_path, "work", bound=False)
    with pytest.raises(InfrastructureError, match="no rsi-box bind"):
        _command(remote, control)
