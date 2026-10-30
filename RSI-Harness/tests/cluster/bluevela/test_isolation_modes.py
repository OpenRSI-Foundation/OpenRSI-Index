"""The profile's isolation modes: network_isolation and process_namespaces,
the pre-search isolation check, and what the default profile keeps."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.bluevela import engine
from rsi_harness.cluster.bluevela import runtime as runtime_module
from rsi_harness.cluster.bluevela.adapter import _record_isolation
from rsi_harness.cluster.bluevela.runtime import (
    _process_table,
    isolation_report,
    require_cluster_isolation,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.cluster.slurm.adapter import SlurmClusterAdapter
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import AgentRunResult, ContainerRef, EvaluationRequest
from tests.cluster.bluevela.test_runtime import (
    _codex_runtime,
    _RecordingPopen,
)
from tests.cluster.slurm.test_adapter import RecordingSlurmScheduler
from tests.cluster.slurm.test_oracle import setup_oracle
from tests.factories import make_run_plan

BOX = ("/opt/venv/bin/python", "-I", "-S", "-B", "/usr/local/bin/rsi-box")


def _profile(**apptainer):
    profile = load_cluster_profile("bluevela")
    return profile.model_copy(
        update={"apptainer": profile.apptainer.model_copy(update=apptainer)}
    )


def _runtime(tmp_path, monkeypatch, *, plan=None, **apptainer):
    runtime = _codex_runtime(tmp_path, monkeypatch, profile=_profile(**apptainer))
    if plan is not None:
        runtime.plan = plan
    return runtime


def _with_networks(plan, *, agent="no-network", verifier="no-network", name=None):
    task = plan.task
    agent_plan = task.agent.model_copy(
        update={
            "network": task.agent.network.model_copy(update={"mode": agent}),
            **({} if name is None else {"name": name}),
        }
    )
    verifier_plan = task.verifier.model_copy(
        update={"network": task.verifier.network.model_copy(update={"mode": verifier})}
    )
    return plan.model_copy(
        update={
            "task": task.model_copy(
                update={"agent": agent_plan, "verifier": verifier_plan}
            )
        }
    )


def _judge_request(runtime, tmp_path):
    runtime.task_assets.mkdir(parents=True)
    runtime.judge_tmp_root.mkdir(parents=True)
    workspace = runtime.rounds / "agent-1" / "workspace"
    workspace.mkdir(parents=True)
    return workspace, EvaluationRequest(
        run_plan=runtime.plan,
        work_container=ContainerRef(container_id="work", role="work"),
        round_id="agent-1",
        verifier_logs=tmp_path / "verifier",
        verifier_output=tmp_path / "feedback.log",
    )


def _record_execs(runtime, tmp_path, monkeypatch):
    """Seed, a Work exec and a Judge, through the real argv builder."""
    monkeypatch.setattr(_RecordingPopen, "calls", [])
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
    runtime._seed()
    runtime.exec(runtime.work_ref, ("codex", "exec"), timeout_seconds=5)
    workspace, request = _judge_request(runtime, tmp_path)
    runtime.run_judge(workspace, request, {"RSI_JUDGE_NETNS": "private"})
    return _RecordingPopen.calls


def test_the_default_profile_keeps_every_argv(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    (seed, _), (assets, _), (work, work_env), (judge, judge_env) = _record_execs(
        runtime, tmp_path, monkeypatch
    )

    no_network = ("--net", "--network", "none", "--hostname", "localhost")
    contained = ("--no-umask", "--containall", "--no-mount")
    head = ("/usr/bin/apptainer", "exec", *no_network, *contained)
    cwd = ("--writable-tmpfs", "--cwd", "/workspace")
    dns = ("--bind", "/etc/resolv.conf:/etc/resolv.conf:ro")
    work_binds = tuple(
        argument
        for binding in runtime.profile.apptainer.work_binds
        for argument in ("--bind", f"{binding.source}:{binding.target}:ro")
    )
    judge_binds = tuple(
        argument
        for binding in runtime.profile.apptainer.judge_binds
        for argument in ("--bind", f"{binding.source}:{binding.target}:ro")
    )
    sif = str(tmp_path / "task.sif")
    tmp = ("--bind", f"{runtime.work_tmp}:/tmp")
    assert seed[:-3] == (
        *head,
        "hostfs",
        *cwd,
        *dns,
        "--bind",
        f"{runtime.workspace}:/run/rsi-harness/seed",
        *tmp,
        "--bind",
        f"{runtime.agent_home}:/home/agent",
        *work_binds,
        sif,
    )
    assert assets[: len(head) + 1] == (*head, "hostfs,tmp")
    assert work == (
        *head,
        "hostfs",
        *cwd,
        *dns,
        *tmp,
        "--bind",
        f"{runtime.workspace}:/workspace",
        "--bind",
        f"{runtime.feedback}:/run/rsi-harness/feedback:ro",
        "--bind",
        f"{runtime.agent_home}:/home/agent",
        *work_binds,
        sif,
        "codex",
        "exec",
    )
    isolate = runtime.control / "judge-isolate"
    assert judge[: len(head) + 6] == (*head, "hostfs", *cwd, *dns)
    assert judge[-len(judge_binds) - 5 :] == (
        "--bind",
        f"{isolate / 'rsi-isolate'}:/usr/local/bin/rsi-isolate:ro",
        *judge_binds,
        sif,
        "/bin/bash",
        "/tests/test.sh",
    )
    assert not any("rsi-box" in argument for argument in (*seed, *work, *judge))
    assert "APPTAINERENV_RSI_SUBMIT_SOCKET" not in work_env
    # A no-network Judge has its own network namespace, whatever the task says.
    assert judge_env["APPTAINERENV_RSI_JUDGE_NETNS"] == "private"


def test_seccomp_mode_filters_no_network_execs_on_the_host(
    tmp_path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path, monkeypatch, network_isolation="seccomp")
    monkeypatch.setattr(_RecordingPopen, "calls", [])
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
    options = {"devices": (), "timeout_seconds": 5, "mount_agent_home": False}

    runtime._run(("true",), network_mode="no-network", **options)
    runtime._run(("true",), network_mode="public", **options)

    (filtered, _), (public, _) = _RecordingPopen.calls
    wrapper = Path(runtime_module.__file__).with_name("rsi_nonet.py")
    assert filtered[:5] == (sys.executable, "-I", "-S", "-B", str(wrapper))
    assert filtered[6:9] == ("--", "/usr/bin/apptainer", "exec")
    apptainer = filtered[7:]
    assert not {"--net", "--network", "--hostname"} & set(apptainer)
    assert public[:2] == ("/usr/bin/apptainer", "exec")


def test_harness_mode_boxes_work_seed_smoke_and_judge(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    calls = _record_execs(runtime, tmp_path, monkeypatch)
    runtime._require_judge_python()

    box_bind = f"{runtime.control / 'rsi-box/judge_isolation.py'}"
    box_bind += ":/usr/local/bin/rsi-box:ro"
    sif = str(tmp_path / "task.sif")
    layers = []
    for argv, _environment in _RecordingPopen.calls:
        after = argv[argv.index(sif) + 1 :]
        layers.append(after[5:7] if after[:5] == BOX else None)
        assert (box_bind in argv) is (after[:5] == BOX)
    assert len(calls) == 5
    # Seed (two execs), Work and the Judge; never the python check.
    assert layers == [
        ("--work", "--"),
        ("--work", "--"),
        ("--work", "--"),
        ("--judge", "--"),
        None,
    ]
    smoke = _scripted(runtime, monkeypatch, _ok("0.149.0\n"))
    runtime._smoke_agent()
    assert smoke[0][1]["box"] == "work"


def test_seccomp_mode_boxes_only_a_no_network_judge_without_unix_sockets(
    tmp_path, monkeypatch
) -> None:
    runtime = _runtime(tmp_path, monkeypatch, network_isolation="seccomp")

    assert runtime._box_layer("work", "no-network") is None
    assert runtime._box_layer("judge", "public") is None
    assert runtime._box_layer("judge", "no-network") == [
        *BOX,
        "--judge",
        "--no-unix",
        "--",
    ]
    (tmp_path / "harness").mkdir()
    harness = _runtime(
        tmp_path / "harness",
        monkeypatch,
        network_isolation="seccomp",
        process_namespaces="harness",
    )
    assert harness._box_layer("judge", "public") == [*BOX, "--judge", "--"]


@pytest.mark.parametrize(
    ("network_isolation", "network_mode", "expected"),
    [
        ("namespace", "no-network", "private"),
        ("namespace", "public", "shared"),
        ("seccomp", "no-network", "shared"),
    ],
)
def test_the_judge_learns_whether_its_network_namespace_is_shared(
    tmp_path, monkeypatch, network_isolation, network_mode, expected
) -> None:
    plan = _with_networks(make_run_plan(tmp_path), verifier=network_mode)
    runtime = _runtime(
        tmp_path, monkeypatch, plan=plan, network_isolation=network_isolation
    )
    seen = {}

    def fake_run(command, **options):
        seen.update(options["environment"])
        return AgentRunResult(exit_code=0, output="")

    monkeypatch.setattr(runtime, "_run", fake_run)
    workspace, request = _judge_request(runtime, tmp_path)

    runtime.run_judge(workspace, request, {"RSI_JUDGE_NETNS": "private"})

    assert seen["RSI_JUDGE_NETNS"] == expected


def _task(*, agent="no-network", name="codex", verifier="no-network", **extra):
    return SimpleNamespace(
        agent=SimpleNamespace(name=name, network=SimpleNamespace(mode=agent)),
        verifier=SimpleNamespace(
            network=SimpleNamespace(mode=verifier),
            model_access=extra.get("model_access", False),
            judge_model=extra.get("judge_model"),
        ),
        sandbox=extra.get("sandbox"),
    )


@pytest.mark.parametrize(
    ("task", "multi_node", "message"),
    [
        (_task(), False, "no-network Agent"),
        (_task(agent="public", verifier="public", judge_model="gpt"), False, None),
        (
            _task(agent="public", model_access=True),
            False,
            None,
        ),
        (
            _task(agent="public", verifier="no-network", judge_model="gpt"),
            False,
            "pinned Judge model access",
        ),
        (
            _task(
                agent="public",
                sandbox=SimpleNamespace(
                    environments=SimpleNamespace(judge=object(), work=None)
                ),
            ),
            False,
            "sandbox.environments.judge",
        ),
        (_task(agent="public"), True, "multi-node"),
        (_task(name="oracle"), False, None),
    ],
)
def test_seccomp_mode_refuses_what_it_cannot_isolate(task, multi_node, message):
    apptainer = SimpleNamespace(
        network_isolation="seccomp", process_namespaces="apptainer"
    )
    if message is None:
        require_cluster_isolation(task, apptainer, multi_node=multi_node)
        return
    with pytest.raises(SetupError, match=message):
        require_cluster_isolation(task, apptainer, multi_node=multi_node)


def test_harness_mode_refuses_multi_node_and_defaults_refuse_nothing():
    harness = SimpleNamespace(
        network_isolation="namespace", process_namespaces="harness"
    )
    with pytest.raises(SetupError, match="multi-node"):
        require_cluster_isolation(_task(), harness, multi_node=True)
    defaults = SimpleNamespace(
        network_isolation="namespace", process_namespaces="apptainer"
    )
    require_cluster_isolation(
        _task(judge_model="gpt", model_access=True), defaults, multi_node=True
    )


def test_the_adapter_refuses_before_any_job(tmp_path):
    profile, request = setup_oracle(tmp_path)
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={"network_isolation": "seccomp"}
            )
        }
    )
    request = request.model_copy(
        update={
            "agent_name": "claude-code",
            "options": request.options.model_copy(update={"agent_name": "claude-code"}),
            "dry_run": False,
        }
    )
    scheduler = RecordingSlurmScheduler()

    with pytest.raises(SetupError, match="no-network Agent"):
        SlurmClusterAdapter(profile, scheduler=scheduler).run(request)

    assert scheduler.events == []
    assert not profile.storage.run_root.exists()


def _scripted(runtime, monkeypatch, *results):
    """_run answers in order and records each call."""
    calls = []
    answers = list(results)

    def fake_run(command, **options):
        calls.append((tuple(command), options))
        return answers.pop(0)

    monkeypatch.setattr(runtime, "_run", fake_run)
    return calls


def _ok(output=""):
    return AgentRunResult(exit_code=0, output=output)


def test_seccomp_mode_needs_the_filter_in_the_container(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, network_isolation="seccomp")
    _scripted(runtime, monkeypatch, _ok("WARNING: no pid ns\nSeccomp:\t0\n"))

    with pytest.raises(InfrastructureError, match="network filter"):
        runtime._require_site_isolation()


def test_seccomp_mode_names_a_setuid_apptainer(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, network_isolation="seccomp")
    error = "ERROR  : Installation issue: starter-suid doesn't have setuid bit set"
    _scripted(runtime, monkeypatch, AgentRunResult(exit_code=255, output=error))

    with pytest.raises(InfrastructureError, match=r"setuid bit set; Apptainer runs"):
        runtime._require_site_isolation()


def test_box_modes_need_python_and_a_working_box(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    _scripted(runtime, monkeypatch, AgentRunResult(exit_code=127, output="nope"))
    with pytest.raises(SetupError, match="python3 3.8 or later at /opt/venv"):
        runtime._require_site_isolation()

    failed = AgentRunResult(
        exit_code=125, output="rsi-box: mount proc: Operation not permitted\n"
    )
    calls = _scripted(runtime, monkeypatch, _ok(), failed)
    with pytest.raises(InfrastructureError, match="mount proc: Operation not"):
        runtime._require_site_isolation()
    assert calls[1][1]["box"] == "work"

    (tmp_path / "s").mkdir()
    seccomp = _runtime(tmp_path / "s", monkeypatch, network_isolation="seccomp")
    calls = _scripted(seccomp, monkeypatch, _ok("Seccomp:\t2\n"), _ok(), _ok())
    seccomp._require_site_isolation()
    assert [options.get("box") for _command, options in calls] == [
        None,
        None,
        "judge",
    ]


def test_the_default_profile_runs_no_site_checks(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    _scripted(runtime, monkeypatch)

    runtime._require_site_isolation()


def test_a_missing_network_namespace_names_seccomp_mode(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    error = "FATAL: Failed to create network namespace: maximum number exceeded"
    _scripted(runtime, monkeypatch, AgentRunResult(exit_code=255, output=error))

    with pytest.raises(InfrastructureError, match='network_isolation = "seccomp"'):
        runtime._seed()


_FAKE_BOX = r"""
import os, signal, subprocess, sys, time
init = os.fork()
if init == 0:
    targets = []
    pauses = []
    def act(signum, frame):
        stop = signum == signal.SIGUSR1
        pauses.append(stop)
        # "late": the first pause misses a process; "half": every pause does.
        reached = {
            "obey": targets,
            "late": targets[: 1 if pauses.count(True) == 1 else 2],
            "half": targets[:1],
            "ignore": [],
        }[sys.argv[2]]
        for pid in targets if not stop and sys.argv[2] != "ignore" else reached:
            os.kill(pid, signal.SIGSTOP if stop else signal.SIGCONT)
    signal.signal(signal.SIGUSR1, act)
    signal.signal(signal.SIGUSR2, act)
    targets.append(subprocess.Popen(["sleep", "60"]).pid)
    targets.append(subprocess.Popen(["sleep", "60"], start_new_session=True).pid)
    while True:
        signal.pause()
for signum in (signal.SIGUSR1, signal.SIGUSR2):
    signal.signal(signum, lambda signum, frame: os.kill(init, signum))
os.waitpid(init, 0)
"""


def _fake_box(tmp_path, behaviour):
    """A process tree shaped as Apptainer's starter, the box's outer process,
    its pid 1 and two Work processes (one in its own session)."""
    script = tmp_path / "fake_box.py"
    script.write_text(_FAKE_BOX)
    process = subprocess.Popen(
        [
            "/bin/sh",
            "-c",
            f'"{sys.executable}" {script} /usr/local/bin/rsi-box {behaviour}; :',
        ],
        start_new_session=True,
        text=True,
    )
    deadline = time.monotonic() + 20
    while True:
        table = _process_table()
        sleeps = [pid for pid, row in table.items() if row[2] == "sleep 60"]
        mine = [pid for pid in sleeps if _below(table, pid, process.pid)]
        if len(mine) == 2:
            return process, mine
        assert time.monotonic() < deadline
        time.sleep(0.05)


def _below(table, pid, root):
    while pid in table and pid != root:
        pid = table[pid][0]
    return pid == root


def _states(pids):
    table = _process_table()
    return [table[pid][1][:1] for pid in pids]


def _end(process, pids):
    os.killpg(process.pid, signal.SIGKILL)
    for pid in pids:
        os.kill(pid, signal.SIGKILL)
    process.wait()


def test_harness_mode_pauses_every_process_through_the_box(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    process, workers = _fake_box(tmp_path, "obey")
    try:
        runtime._process = process
        runtime.pause(runtime.work_ref)
        # The session leader too, which escapes a process-group stop.
        assert _states(workers) == ["T", "T"]
        runtime.unpause(runtime.work_ref)
        deadline = time.monotonic() + 10
        while "T" in _states(workers):
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        _end(process, workers)


def test_a_pause_is_sent_again_until_every_process_stays_stopped(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    process, workers = _fake_box(tmp_path, "late")
    try:
        runtime._process = process
        runtime.pause(runtime.work_ref)
        assert _states(workers) == ["T", "T"]
    finally:
        _end(process, workers)


def test_a_failed_pause_continues_the_box_and_stop_still_ends_it(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    monkeypatch.setattr(runtime_module, "_BOX_PAUSE_SECONDS", 0.5)
    process, workers = _fake_box(tmp_path, "half")
    try:
        runtime._process = process
        with pytest.raises(InfrastructureError, match="did not stop"):
            runtime.pause(runtime.work_ref)
        deadline = time.monotonic() + 10
        while "T" in _states(workers):
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        _end(process, workers)


def test_a_thread_still_running_is_not_a_stopped_process(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    monkeypatch.setattr(runtime_module, "_BOX_PAUSE_SECONDS", 0.3)
    box = "python -I -S -B /usr/local/bin/rsi-box --work -- codex"
    # The main thread has exited (Z) while another thread runs (R).
    table = {
        9: (1, "S", "apptainer"),
        10: (9, "S", box),
        11: (10, "S", box),
        12: (11, "ZR", "work"),
    }
    sent = []
    monkeypatch.setattr(runtime_module, "_process_table", lambda: table)
    monkeypatch.setattr(os, "kill", lambda pid, signum: sent.append((pid, signum)))
    runtime._process = SimpleNamespace(pid=9, poll=lambda: None)

    with pytest.raises(InfrastructureError, match=r"running \[12\]"):
        runtime.pause(runtime.work_ref)

    assert sent.count((10, signal.SIGUSR1)) > 1
    assert sent[-1] == (10, signal.SIGUSR2)


def test_the_process_table_has_every_thread_s_state():
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import threading, time\n"
            "threading.Thread(target=time.sleep, args=(60,), daemon=True).start()\n"
            "time.sleep(60)",
        ]
    )
    try:
        deadline = time.monotonic() + 10
        while len(_process_table().get(process.pid, (0, ""))[1]) != 2:
            assert time.monotonic() < deadline
            time.sleep(0.05)
    finally:
        process.kill()
        process.wait()


def test_stop_terminates_work_after_its_box_has_gone(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    process = subprocess.Popen(["sleep", "60"], start_new_session=True)
    monkeypatch.setattr(runtime_module, "_process_table", dict)
    try:
        runtime._process = process
        runtime._paused = True
        runtime.unpause(runtime.work_ref)
        runtime._paused = True
        runtime.stop(runtime.work_ref)
        assert process.wait(timeout=10) == -signal.SIGTERM
    finally:
        process.kill()
        process.wait()


def test_a_box_that_does_not_stop_its_processes_fails_the_pause(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    monkeypatch.setattr(runtime_module, "_BOX_PAUSE_SECONDS", 0.5)
    process, workers = _fake_box(tmp_path, "ignore")
    try:
        runtime._process = process
        with pytest.raises(InfrastructureError, match="did not stop"):
            runtime.pause(runtime.work_ref)
        assert runtime._paused is False
    finally:
        _end(process, workers)


def _probe(*lines, end=True):
    return "\n".join((*lines, *(["isolation probe end"] if end else []))) + "\n"


_HOST_MOUNTINFO = (
    "20 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
    "30 20 0:50 / /scratch rw - lustre scratch rw\n"
)


def _report(output, **options):
    defaults = {
        "planned": (PurePosixPath("/workspace"),),
        "forbidden": (Path("/scratch/runs"),),
        "network_isolation": "namespace",
        "network_mode": "public",
        "boxed": False,
        "host_mountinfo": _HOST_MOUNTINFO,
    }
    return isolation_report(output, **{**defaults, **options})


@pytest.mark.parametrize(
    ("output", "options", "failure"),
    [
        (_probe(end=False), {}, "did not finish"),
        (_probe("visible /scratch/runs"), {}, "Work sees /scratch/runs"),
        (_probe("engine signal"), {}, 'process_namespaces = "harness"'),
        (_probe("engine proc"), {}, "Engine's process"),
        (
            _probe("mount 41 40 0:50 /runs/r1 /data rw - lustre scratch rw"),
            {},
            "Work mounts /scratch/runs at /data",
        ),
        (
            _probe("mount 41 40 0:50 / /data rw - lustre scratch rw"),
            {},
            "Work mounts /scratch/runs at /data",
        ),
        (_probe("seccomp 0"), {"boxed": True}, "box has no filter"),
        (
            _probe("seccomp 2", "inet open"),
            {"network_isolation": "seccomp", "network_mode": "no-network"},
            "can create IP sockets",
        ),
        (
            _probe("seccomp 0", "inet denied"),
            {"network_isolation": "seccomp", "network_mode": "no-network"},
            "can create IP sockets",
        ),
        (
            _probe("interface lo", "interface eth0"),
            {"network_mode": "no-network"},
            "interfaces ['eth0']",
        ),
    ],
)
def test_the_isolation_check_fails_closed(output, options, failure):
    report = _report(output, **options)

    assert report["passed"] is False
    assert any(failure in item for item in report["failures"]), report["failures"]


def test_the_isolation_check_passes_planned_and_reports_unknown_mounts():
    report = _report(
        _probe(
            "seccomp 2",
            "inet denied",
            "interface lo",
            "interface eth0",
            "mount 41 40 0:50 /runs/r1/workspace /workspace rw - lustre s rw",
            "mount 42 40 0:60 / /opt/tools rw - nfs tools rw",
            "mount 43 40 0:5 / /proc rw - proc proc rw",
            "keyring shared",
        ),
        network_isolation="seccomp",
        network_mode="no-network",
        boxed=True,
    )

    assert report == {
        "passed": True,
        "seccomp": "2",
        "keyring_shared": True,
        "other_mounts": ["/opt/tools"],
        "failures": [],
    }
    assert _report(_probe("keyring private"))["keyring_shared"] is False
    assert _report(_probe())["keyring_shared"] == "unknown"


def test_initialize_fails_closed_before_the_agent_and_records_why(
    tmp_path, monkeypatch
):
    runtime = _runtime(tmp_path, monkeypatch)
    calls = []

    def fake_run(command, **options):
        calls.append(tuple(command))
        if command[:2] == ("/bin/sh", "-c"):
            hidden = str(runtime.plan.task.source_dir)
            return _ok(_probe(f"visible {hidden}", "engine signal"))
        return _ok("0.149.0\n")

    monkeypatch.setattr(runtime, "_run", fake_run)

    with pytest.raises(InfrastructureError, match="isolation check failed"):
        runtime.initialize()

    probe = calls[-1]
    assert probe[4] == str(os.getpid())
    assert "/srv/rsi/harness/runs" in probe[7:]
    # At, above or below a bind (here /tmp), or the contained $HOME: never
    # probed.
    assert str(runtime.node_tmp) not in probe[7:]
    report = json.loads((runtime.feedback.parent / "isolation_check.json").read_text())
    assert report["passed"] is False
    assert report["network_isolation"] == "namespace"
    assert report["process_namespaces"] == "apptainer"
    assert len(report["failures"]) == 2
    manifest = {}
    _record_isolation(manifest, runtime.plan, "native-run")
    assert manifest["isolation"] == {
        "passed": False,
        "network_isolation": "namespace",
        "process_namespaces": "apptainer",
        "judge_layer": None,
        "keyring_shared": "unknown",
        "failures": report["failures"],
    }


def test_an_attempt_that_stops_before_the_check_records_none(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch)
    earlier = runtime.feedback.parent / "isolation_check.json"
    earlier.parent.mkdir(parents=True)
    earlier.write_text(json.dumps({"passed": True}))

    def no_box():
        raise InfrastructureError("rsi-box cannot run on this node")

    monkeypatch.setattr(runtime, "_require_site_isolation", no_box)
    with pytest.raises(InfrastructureError, match="rsi-box"):
        runtime.initialize()

    assert not earlier.exists()
    manifest = {"isolation": {"passed": True}}
    _record_isolation(manifest, runtime.plan, "native-run")
    assert "isolation" not in manifest


def test_the_isolation_check_records_the_judge_layer(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    runtime.control.mkdir(parents=True)
    monkeypatch.setattr(
        runtime,
        "_run",
        lambda *_args, **_kwargs: _ok(_probe("seccomp 2", "interface lo")),
    )

    runtime._check_isolation()

    report = json.loads((runtime.feedback.parent / "isolation_check.json").read_text())
    assert report["passed"] is True
    assert report["judge_layer"] == "/usr/local/bin/rsi-box --judge"


def test_the_isolation_check_probes_neither_home_nor_site_bind_paths(
    tmp_path, monkeypatch
):
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.control.mkdir(parents=True)
    site = "/usr/local/cuda"
    monkeypatch.setattr(runtime_module, "_site_bind_paths", lambda _: (Path(site),))
    # $HOME lies below the run root, as /home/users/<user> below a site's
    # /home/users.
    monkeypatch.setattr(Path, "home", lambda: Path("/srv/rsi/harness/runs/me"))
    forbidden = []

    def report(output, **options):
        forbidden.extend(options["forbidden"])
        return isolation_report(output, **options)

    def fake_run(command, **_options):
        # The task image has its own /usr/local/cuda.
        seen = [f"visible {path}" for path in command[7:] if path == site]
        return _ok(_probe(*seen, "interface lo"))

    monkeypatch.setattr(runtime_module, "isolation_report", report)
    monkeypatch.setattr(runtime, "_run", fake_run)

    runtime._check_isolation()

    report = json.loads((runtime.feedback.parent / "isolation_check.json").read_text())
    assert report["passed"] is True
    assert "/srv/rsi/harness/runs" not in report["checked_paths"]
    assert site not in report["checked_paths"]
    # A mount of a site bind path's files still fails.
    assert Path(site) in forbidden


def test_the_isolation_check_asks_for_its_own_key_then_unlinks_it(
    tmp_path, monkeypatch
):
    runtime = _runtime(tmp_path, monkeypatch)
    runtime.control.mkdir(parents=True)
    calls = []
    probes = []

    def keyring(index, *args):
        calls.append((index, *args))
        return 77 if index == 0 else 0

    def fake_run(command, **_options):
        probes.append((command, list(calls)))
        return _ok(_probe("interface lo", "keyring shared"))

    monkeypatch.setattr(runtime_module, "_keyring_call", keyring)
    monkeypatch.setattr(runtime, "_run", fake_run)

    runtime._check_isolation()

    ((command, before),) = probes
    added = calls[0]
    assert added[:2] == (0, b"user") and added[-1] == -4 and before == [added]
    assert command[6] == added[2].decode()
    assert calls[1:] == [(2, 9, 77, -4)]
    report = json.loads((runtime.feedback.parent / "isolation_check.json").read_text())
    assert report["keyring_shared"] is True


def test_site_bind_paths_read_the_installation_s_apptainer_conf(tmp_path, monkeypatch):
    binary = tmp_path / "prefix/bin/apptainer"
    conf = tmp_path / "prefix/etc/apptainer/apptainer.conf"
    conf.parent.mkdir(parents=True)
    conf.write_text(
        "# bind path = /commented\n"
        "bind path = /etc/localtime\n"
        "bind path = /etc/hosts\n"
        "bind path = /scratch\n"
        "bind path = /oak:/data\n"
        "bind path=relative\n"
        "mount home = yes\n"
    )
    assert runtime_module._site_bind_paths(binary) == (
        Path("/scratch"),
        Path("/oak"),
    )
    conf.unlink()
    tried = []

    def read_text(path, *_args, **_kwargs):
        tried.append(path)
        raise FileNotFoundError(path)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert runtime_module._site_bind_paths(binary) == ()
    assert tried == [conf, Path("/etc/apptainer/apptainer.conf")]


def test_the_engine_is_not_dumpable_in_harness_mode(monkeypatch):
    calls = []

    class Libc:
        def prctl(self, *arguments):
            calls.append(arguments)
            return 0

    def stop(*_args, **_kwargs):
        raise StopIteration

    monkeypatch.setattr(engine.ctypes, "CDLL", lambda _name: Libc())
    monkeypatch.setattr(engine, "validate_cluster_sandbox", stop)
    for mode, expected in (("apptainer", []), ("harness", [(4, 0, 0, 0, 0)])):
        calls.clear()
        payload = SimpleNamespace(
            profile=_profile(process_namespaces=mode),
            run_plan=SimpleNamespace(task=SimpleNamespace(sandbox=None), sandbox=None),
            multi_node=None,
        )
        with pytest.raises(StopIteration):
            engine.run_engine_payload(payload)
        assert calls == expected


def test_an_oracle_without_network_submits_over_the_socket(tmp_path, monkeypatch):
    plan = _with_networks(make_run_plan(tmp_path), name="oracle")
    runtime = _runtime(tmp_path, monkeypatch, plan=plan)
    runtime.configure_agent_network("http://127.0.0.1:43123", ())
    try:
        assert runtime._network_broker.provider_socket is None
        command = runtime._network_command(
            ("rsi-oracle",), network_mode="no-network", mount_agent_home=True
        )
        monkeypatch.setattr(_RecordingPopen, "calls", [])
        monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
        runtime._run(
            ("rsi-oracle",), devices=(), network_mode="no-network", timeout_seconds=5
        )
    finally:
        runtime.close_agent_network()

    assert command == ("rsi-oracle",)
    ((argv, environment),) = _RecordingPopen.calls
    network = runtime.node_tmp / "network"
    assert f"{network}:/run/rsi-harness/network:ro" in argv
    assert (
        environment["APPTAINERENV_RSI_SUBMIT_SOCKET"]
        == "/run/rsi-harness/network/submit.sock"
    )


def test_the_probe_reports_what_this_host_shows(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    full = tmp_path / "full"
    full.mkdir()
    (full / "secret").write_text("x")
    probe = ["/bin/sh", "-c", runtime_module._ISOLATION_PROBE, "probe"]
    probe += [str(os.getpid()), sys.executable]
    description = f"rsi-isolation-test:{os.getpid()}"
    key = runtime_module._keyring_call(0, b"user", description.encode(), b"1", 1, -4)
    assert key >= 0
    try:
        result = subprocess.run(
            [*probe, description, str(empty), str(full), str(tmp_path / "missing")],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        runtime_module._keyring_call(2, 9, key, -4)
    lines = result.stdout.splitlines()
    gone = subprocess.run(
        [*probe, description], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    assert [line for line in lines if line.startswith("visible ")] == [
        f"visible {full}"
    ]
    assert {"engine signal", "engine proc", "inet open"} <= set(lines)
    # This process's own keyring: the key, then after the unlink none.
    assert "keyring shared" in lines and "keyring private" in gone
    assert "interface lo" in lines
    assert any(line.startswith("seccomp ") for line in lines)
    assert any(line.startswith("mount ") for line in lines)
    assert lines[-1] == "isolation probe end"
    report = _report(result.stdout, host_mountinfo=None)
    assert report["passed"] is False
    assert any(f"Work sees {full}" == item for item in report["failures"])


def test_initialize_proves_the_box_before_any_task_code(tmp_path, monkeypatch):
    runtime = _runtime(tmp_path, monkeypatch, process_namespaces="harness")
    calls = _scripted(runtime, monkeypatch, AgentRunResult(exit_code=127, output=""))

    with pytest.raises(SetupError, match="rsi-box"):
        runtime.initialize()

    # The python check failed: no seed, no Agent.
    assert [command[0] for command, _options in calls] == ["/opt/venv/bin/python"]
