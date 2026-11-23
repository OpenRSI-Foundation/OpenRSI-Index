"""Cluster Judges get the record of their NFS binds (runtime.judge_binds),
made on the node that starts each container, from its final bind list;
Work never does."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela import runtime as runtime_module
from rsi_harness.cluster.bluevela.runtime import ApptainerAgentRuntime
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import InfrastructureError
from tests.cluster.bluevela.test_multinode import _judge_module, _module, _worker
from tests.cluster.bluevela.test_work_filter import _command, _control
from tests.factories import make_run_plan

RECORD_BIND = "/run/rsi-harness/judge-binds:ro"


def _pairs(command):
    return list(zip(command, command[1:], strict=False))


class Recorder:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.calls: list[tuple[list, Path]] = []
        self.removed: list[Path | None] = []

    def record(self, binds, parent):
        self.calls.append(([(str(s), str(t)) for s, t in binds], Path(parent)))
        return self.directory

    def remove(self, directory):
        self.removed.append(directory)


# -- single node ------------------------------------------------------------------


def _runtime(tmp_path: Path, monkeypatch) -> ApptainerAgentRuntime:
    node_tmp = tmp_path / "node-tmp"
    node_tmp.mkdir()
    monkeypatch.setenv("RSI_HARNESS_NODE_TMP", str(node_tmp))
    return ApptainerAgentRuntime(
        SimpleNamespace(
            profile=load_cluster_profile("bluevela"),
            run_id="native-run",
            sif_path=tmp_path / "task.sif",
        ),
        make_run_plan(tmp_path),
    )


def test_a_single_node_judge_records_its_final_binds(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path / "record")
    monkeypatch.setattr(runtime_module, "bind_record_for", recorder.record)
    runtime = _runtime(tmp_path, monkeypatch)
    logs = tmp_path / "logs" / "verifier"
    parent = tmp_path / "node-tmp" / "judge" / "round-x-binds"

    command = runtime._base_command(
        devices=(),
        extra_binds=((logs, PurePosixPath("/logs/verifier"), False),),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="no-network",
        phase="judge",
        bind_record_parent=parent,
    )

    [(binds, where)] = recorder.calls
    assert where == parent
    assert (str(logs), "/logs/verifier") in binds
    # Every other bind of the command is in it too, and the record comes last.
    bound = [value for flag, value in _pairs(command) if flag == "--bind"]
    assert [value.removesuffix(":ro").rsplit(":", 1)[0] for value in bound[:-1]] == [
        source for source, _target in binds
    ]
    assert bound[-1] == f"{recorder.directory}:{RECORD_BIND}"


def test_work_and_unrecorded_execs_get_no_record(tmp_path, monkeypatch):
    recorder = Recorder(tmp_path / "record")
    monkeypatch.setattr(runtime_module, "bind_record_for", recorder.record)
    runtime = _runtime(tmp_path, monkeypatch)
    for phase, parent in (("work", tmp_path / "p"), ("judge", None)):
        command = runtime._base_command(
            devices=(),
            extra_binds=(),
            mount_workspace=False,
            mount_agent_home=False,
            network_mode="public",
            phase=phase,
            bind_record_parent=parent,
        )
        assert not any(RECORD_BIND in value for value in command)
    assert recorder.calls == []


def test_a_judge_without_an_nfs_bind_keeps_its_command(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    arguments = dict(
        devices=(),
        extra_binds=(),
        mount_workspace=False,
        mount_agent_home=False,
        network_mode="no-network",
        phase="judge",
    )
    monkeypatch.setattr(runtime_module, "bind_record_for", lambda binds, parent: None)
    assert runtime._base_command(
        **arguments, bind_record_parent=tmp_path / "p"
    ) == runtime._base_command(**arguments)


# -- multi-node -------------------------------------------------------------------


def test_a_remote_judge_rank_binds_its_record_and_work_ranks_cannot(tmp_path):
    remote, verifier = _control(tmp_path, "verifier")
    plain = _command(remote, verifier)
    local_tmp = remote.request_tmp_dir(verifier, pool_rank=0)
    with_record = remote.build_apptainer_command(
        verifier,
        request_id="b" * 32,
        pool_rank=0,
        node_rank=0,
        current_host=verifier.lease.nodes[0].host,
        local_tmp=local_tmp,
        bind_record=tmp_path / "record",
    )
    sif = plain.index(str(tmp_path / "image.sif"))
    assert with_record == (
        plain[:sif] + ("--bind", f"{tmp_path / 'record'}:{RECORD_BIND}") + plain[sif:]
    )
    remote, work = _control(tmp_path / "w", "work")
    with pytest.raises(InfrastructureError, match="only a Judge rank"):
        remote.build_apptainer_command(
            work,
            request_id="b" * 32,
            pool_rank=0,
            node_rank=0,
            current_host=work.lease.nodes[0].host,
            local_tmp=remote.request_tmp_dir(work, pool_rank=0),
            bind_record=tmp_path / "record",
        )


class FakeProcess:
    pid = os.getpid()

    def __init__(self, command, **_kwargs) -> None:
        FakeProcess.command = command

    def wait(self) -> int:
        return 0

    def poll(self) -> int:
        return 0


@pytest.mark.parametrize("phase", ["verifier", "work"])
def test_a_remote_rank_records_on_its_own_node(tmp_path, monkeypatch, phase):
    remote, control = _control(tmp_path, phase)
    recorder = Recorder(tmp_path / "record")
    monkeypatch.setattr(remote, "bind_record_for", recorder.record)
    monkeypatch.setattr(remote, "remove_bind_record", recorder.remove)
    monkeypatch.setattr(remote.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(remote, "is_singularity_ce", lambda _binary: False)
    monkeypatch.setattr(
        remote.socket, "gethostname", lambda: control.lease.nodes[0].host
    )

    assert (
        remote.run_worker(control, request_id="b" * 32, pool_rank=0, node_rank=0) == 0
    )

    local_tmp = remote.request_tmp_dir(control, pool_rank=0)
    if phase == "work":
        assert recorder.calls == [] and recorder.removed == [None]
        assert not any(RECORD_BIND in value for value in FakeProcess.command)
        return
    [(binds, parent)] = recorder.calls
    assert parent == local_tmp.parent
    assert binds == [
        (str(s), str(t)) for s, t, _ro in remote.worker_binds(control, local_tmp)
    ]
    assert (str(local_tmp), "/tmp") in binds
    assert f"{recorder.directory}:{RECORD_BIND}" in FakeProcess.command
    assert recorder.removed == [recorder.directory]


def test_the_judge_controller_records_on_its_node(tmp_path, monkeypatch):
    judge = _judge_module()
    control = judge.JudgeControllerControl(
        request_id="e" * 32,
        run_id="run-123",
        host="judge-0",
        worker=_worker(_module(), tmp_path),
        broker_root=tmp_path / "broker",
        command=("/bin/bash", "/tests/test.sh"),
        environment={},
        local_world_size=8,
    )
    recorder = Recorder(tmp_path / "record")
    monkeypatch.setattr(judge, "bind_record_for", recorder.record)
    monkeypatch.setattr(judge, "remove_bind_record", recorder.remove)
    monkeypatch.setattr(judge.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(judge, "is_singularity_ce", lambda _binary: False)
    monkeypatch.setattr(judge.socket, "gethostname", lambda: "judge-0")

    assert judge._run(control) == 0

    local_tmp = judge.controller_tmp_dir(control)
    [(binds, parent)] = recorder.calls
    assert parent == local_tmp.parent
    assert (str(tmp_path / "broker"), "/run/rsi-harness/torchrun") in binds
    assert (str(local_tmp), "/tmp") in binds
    assert f"{recorder.directory}:{RECORD_BIND}" in FakeProcess.command
    assert recorder.removed == [recorder.directory]
    # Without a record the command is the one it always was.
    plain = judge.build_apptainer_command(
        control, current_host="judge-0", local_tmp=local_tmp
    )
    assert not any(RECORD_BIND in value for value in plain)


def test_a_round_keeps_its_record_outside_every_writable_bind(tmp_path, monkeypatch):
    """run_judge: the record's parent is node-local, outside every writable
    bind of the Judge (which could rewrite it), and gone after the round."""
    from tests.cluster.bluevela.test_isolation_modes import (
        _judge_request,
        _RecordingPopen,
    )
    from tests.cluster.bluevela.test_isolation_modes import _runtime as boxed_runtime

    runtime = boxed_runtime(tmp_path, monkeypatch)
    made = []

    def record(binds, parent):
        directory = Path(parent) / "judge-binds-x"
        directory.mkdir(parents=True)
        made.append((list(binds), directory))
        return directory

    monkeypatch.setattr(runtime_module, "bind_record_for", record)
    monkeypatch.setattr(_RecordingPopen, "calls", [])
    monkeypatch.setattr(runtime_module.subprocess, "Popen", _RecordingPopen)
    runtime.run_judge(_judge_request(runtime, tmp_path), {"RSI_JUDGE_NETNS": "private"})

    [(binds, directory)] = made
    assert directory.parent.parent == runtime.judge_tmp_root
    assert (str(tmp_path / "verifier"), "/logs/verifier") in [
        (str(source), str(target)) for source, target in binds
    ]
    [(judge, _environment)] = _RecordingPopen.calls[-1:]
    writable = [
        value.rsplit(":", 1)[0]
        for flag, value in _pairs(judge)
        if flag == "--bind" and not value.endswith(":ro")
    ]
    assert writable and not any(
        str(directory).startswith(source.rstrip("/") + "/") for source in writable
    )
    assert f"{directory}:{RECORD_BIND}" in judge
    assert not directory.parent.exists()
