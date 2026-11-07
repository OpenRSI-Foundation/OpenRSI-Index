from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import pytest

from rsi_harness.cluster.schedulers import allocation, local_job
from rsi_harness.cluster.schedulers.allocation import (
    JOB_GPUS_VARIABLE,
    AllocationScheduler,
    LocalJob,
)
from rsi_harness.cluster.schedulers.pbs import PBSJobSpec, PBSScheduler
from rsi_harness.cluster.schedulers.pjm import PJMScheduler
from rsi_harness.errors import InfrastructureError, SetupError

GPUS = "GPU-aaaa,GPU-bbbb,GPU-cccc,GPU-dddd"
PBS_JOB = "97400.jobm"


def qstat_answer(*, walltime="05:00:00", started_seconds_ago=60, state="R"):
    started = datetime.fromtimestamp(time.time() - started_seconds_ago)
    job = {
        "job_state": state,
        "stime": started.strftime("%a %b %d %H:%M:%S %Y"),
        "Resource_List": {"walltime": walltime, "ngpus": 4, "nodect": 1},
    }

    def answer(argv):
        # qselect: no submitted job of that name.
        listing = "" if argv[0] == "qselect" else json.dumps({"Jobs": {argv[-1]: job}})
        return subprocess.CompletedProcess(argv, 0, listing, "")

    return answer


_STARTED: list[AllocationScheduler] = []


@pytest.fixture(autouse=True)
def _end_leftover_jobs():
    """A failed assertion must not leave a test's local jobs running."""
    yield
    while _STARTED:
        local = _STARTED.pop()
        for job_id in list(local._processes):
            local.cancel(job_id)


def scheduler(tmp_path, *, environ=None, runner=None, pbs_job=PBS_JOB, **options):
    environment = {
        "PBS_JOBID": pbs_job,
        "CUDA_VISIBLE_DEVICES": GPUS,
        "PATH": os.environ["PATH"],
        "SECRET": "only-in-the-local-environment",
    }
    if environ is not None:
        environment = environ
    ledger = tmp_path / "ledger-root"
    ledger.mkdir(exist_ok=True)
    local = AllocationScheduler(
        PBSScheduler(runner=runner or qstat_answer(), sleeper=lambda _: None),
        record_root=tmp_path / "runs" / ".local-jobs",
        environ=environment,
        ledger_root=ledger,
        **options,
    )
    _STARTED.append(local)
    return local


def spec(tmp_path, body, *, name="rsi-task-run-alice", gpus=2, walltime="03:45"):
    directory = tmp_path / name
    directory.mkdir(exist_ok=True)
    script = directory / "run.sh"
    script.write_text("#!/bin/bash\nset -eu\n" + body)
    return PBSJobSpec(
        name=name,
        queue="t2-single",
        group="",
        cpu_slots=24,
        memory_mb=65536,
        walltime=walltime,
        stdout_path=directory / "pbs.%j.out",
        stderr_path=directory / "pbs.%j.err",
        script_path=script,
        gpu_count=gpus,
        local_tmp_mb=1,
    )


def log(request, job_id, stream="out"):
    return request.stdout_path.with_name(f"pbs.{job_id}.{stream}")


def ledger_entries(tmp_path):
    return list((tmp_path / "ledger-root").glob("*/*.json"))


def gone(pid):
    process = allocation._process(pid)
    return process is None or process[0] == "Z"


def test_a_job_runs_locally_in_the_controllers_job(tmp_path):
    local = scheduler(tmp_path)
    request = spec(tmp_path, 'echo "$CUDA_VISIBLE_DEVICES|$SECRET|$PBS_JOBID"\n')
    job_id = local.submit(request)
    assert job_id.startswith(f"{PBS_JOB}.local-")
    states = []
    result = local.wait(job_id, poll_seconds=0.1, on_state=states.append)
    assert (result.state, result.exit_code) == ("COMPLETED", 0)
    assert result.ended_at is not None and abs(result.ended_at - time.time()) < 30
    assert states == ["RUNNING", "COMPLETED"]
    # The batch job's own GPUs and environment, as they are.
    assert log(request, job_id).read_text().strip() == (
        f"{GPUS}|only-in-the-local-environment|{PBS_JOB}"
    )
    assert oct(log(request, job_id).stat().st_mode & 0o777) == "0o600"
    assert not ledger_entries(tmp_path)
    record = tmp_path / "runs" / ".local-jobs" / f"{job_id}.json"
    assert oct(record.stat().st_mode & 0o777) == "0o600"
    assert oct(record.parent.stat().st_mode & 0o777) == "0o700"
    # The end record and heartbeat live in the job's directory.
    saved = LocalJob.model_validate_json(record.read_text())
    assert saved.end_path.parent == local.job_directory(request)
    assert saved.end_path.parent.parent == request.stdout_path.parent


def test_one_local_job_at_a_time_holds_a_batch_job(tmp_path):
    first = scheduler(tmp_path)
    held = first.submit(spec(tmp_path, "exec sleep 30\n", name="run-a"))
    with pytest.raises(InfrastructureError, match="one local job at a time"):
        scheduler(tmp_path).submit(spec(tmp_path, "true\n", name="run-b"))
    first.cancel(held)
    second = scheduler(tmp_path)
    second.wait(second.submit(spec(tmp_path, "true\n", name="run-b")), poll_seconds=0.1)


def test_a_job_wider_than_the_batch_job_is_refused(tmp_path):
    with pytest.raises(SetupError, match="holds 4 GPUs"):
        scheduler(tmp_path).submit(spec(tmp_path, "true\n", gpus=5))


def test_unset_devices_are_the_gpus_the_batch_job_requested(tmp_path):
    # PBS says the job requested 4 (ngpus); a stale value is not passed on.
    environ = {
        "PBS_JOBID": PBS_JOB,
        "PATH": os.environ["PATH"],
        JOB_GPUS_VARIABLE: "8",
    }
    local = scheduler(tmp_path, environ=environ)
    with pytest.raises(SetupError, match=r"holds 4 GPUs \(requested\)"):
        local.submit(spec(tmp_path, "true\n", gpus=5))
    request = spec(tmp_path, f'echo "${JOB_GPUS_VARIABLE}"\n')
    job_id = local.submit(request)
    local.wait(job_id, poll_seconds=0.1)
    assert log(request, job_id).read_text().strip() == "4"


def test_set_devices_are_the_gpus_the_engine_takes(tmp_path):
    environ = {
        "PBS_JOBID": PBS_JOB,
        "CUDA_VISIBLE_DEVICES": GPUS,
        "PATH": os.environ["PATH"],
        JOB_GPUS_VARIABLE: "8",
    }
    local = scheduler(tmp_path, environ=environ)
    request = spec(tmp_path, f'echo "${{{JOB_GPUS_VARIABLE}-unset}}"\n')
    job_id = local.submit(request)
    local.wait(job_id, poll_seconds=0.1)
    assert log(request, job_id).read_text().strip() == "unset"


def test_a_job_that_cannot_finish_in_the_batch_jobs_walltime_is_refused(tmp_path):
    # 03:45 of work and 5 min of teardown, with 3:45 left.
    tight = qstat_answer(walltime="04:00:00", started_seconds_ago=900)
    with pytest.raises(SetupError, match="request a longer walltime"):
        scheduler(tmp_path, runner=tight).submit(spec(tmp_path, "true\n"))
    # Build and run one after the other must both fit.
    roomy = scheduler(tmp_path, runner=qstat_answer(walltime="04:20:00"))
    with pytest.raises(SetupError, match="need 7.50 h"):
        roomy.require_room(
            (spec(tmp_path, "true\n", name="build"), spec(tmp_path, "true\n"))
        )
    result = roomy.wait(roomy.submit(spec(tmp_path, "true\n")), poll_seconds=0.1)
    assert result.state == "COMPLETED"


def test_a_walltime_the_scheduler_does_not_tell_is_not_guessed(tmp_path):
    def silent(argv):
        job = {"job_state": "R"}
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"Jobs": {argv[-1]: job}}), ""
        )

    with pytest.raises(InfrastructureError, match="cannot tell how much walltime"):
        scheduler(tmp_path, runner=silent).submit(spec(tmp_path, "true\n"))


def test_pjm_jobs_are_read_from_pjstat(tmp_path):
    table = "JOB_ID     ST  ELAPSE_TIM ELAPSE_LIM\n1006903    RUN 0:10:00    6:00:00\n"
    host = PJMScheduler(
        runner=lambda argv: subprocess.CompletedProcess(argv, 0, table, ""),
        sleeper=lambda _: None,
    )
    local = AllocationScheduler(
        host,
        record_root=tmp_path / "records",
        environ={"PJM_JOBID": "1006903", "PATH": os.environ["PATH"]},
        ledger_root=tmp_path,
    )
    _STARTED.append(local)
    # CUDA_VISIBLE_DEVICES unset: the Engine finds the GPUs the job can open.
    result = local.wait(
        local.submit(spec(tmp_path, "true\n", walltime="05:30")), poll_seconds=0.1
    )
    assert result.state == "COMPLETED"
    with pytest.raises(SetupError, match="PJM job 1006903 has 5.83 h"):
        local.require_room((spec(tmp_path, "true\n", walltime="05:50"),))


def test_allocation_launch_needs_a_batch_job(tmp_path):
    local = scheduler(tmp_path, environ={"PATH": "/usr/bin"})
    with pytest.raises(SetupError, match="inside a PBS job"):
        local.submit(spec(tmp_path, "true\n"))
    # A dry run renders the runner anywhere.
    assert "--walltime" in local.render_submit(spec(tmp_path, "true\n"))


def test_a_failing_job_reports_its_exit_code(tmp_path):
    local = scheduler(tmp_path)
    result = local.wait(local.submit(spec(tmp_path, "exit 3\n")), poll_seconds=0.1)
    assert (result.state, result.exit_code) == ("FAILED", 3)


def test_cancel_ends_the_whole_process_group(tmp_path, until):
    local = scheduler(tmp_path)
    marker = tmp_path / "child.pid"
    job_id = local.submit(spec(tmp_path, f"sleep 300 & echo $! > {marker}\nwait\n"))
    assert until(lambda: marker.exists() and marker.read_text().strip())
    child = int(marker.read_text())
    local.cancel(job_id)
    assert until(lambda: gone(child))
    ended = local.finished(job_id)
    assert ended is not None and ended.state == "CANCELLED"
    assert not ledger_entries(tmp_path)


def test_an_interrupted_wait_cancels_the_job(tmp_path, until, monkeypatch):
    local = scheduler(tmp_path)
    marker = tmp_path / "child.pid"
    job_id = local.submit(spec(tmp_path, f"sleep 300 & echo $! > {marker}\nwait\n"))
    assert until(lambda: marker.exists() and marker.read_text().strip())
    process = local._processes[job_id]

    def interrupted(timeout=None):
        if timeout is not None:
            raise KeyboardInterrupt
        return subprocess.Popen.wait(process)

    monkeypatch.setattr(process, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt, match=f"PBS local job {job_id} cancelled"):
        local.wait(job_id, poll_seconds=0.1)
    assert until(lambda: gone(int(marker.read_text())))


@pytest.mark.parametrize("writes", [0, 1])
def test_an_interrupted_submit_ends_the_job_it_started(
    tmp_path, until, monkeypatch, writes
):
    # A SIGTERM to the controller (as KeyboardInterrupt) while the ledger
    # or the record is written: the caller never learns the job's ID.
    local = scheduler(tmp_path)
    marker = tmp_path / "child.pid"
    written = []
    write = AllocationScheduler._write

    def interrupted(directory, job):
        if len(written) == writes:
            assert until(lambda: marker.exists() and marker.read_text().strip())
            raise KeyboardInterrupt
        written.append(directory)
        write(directory, job)

    monkeypatch.setattr(AllocationScheduler, "_write", staticmethod(interrupted))
    request = spec(tmp_path, f"sleep 300 & echo $! > {marker}\nwait\n")
    with pytest.raises(KeyboardInterrupt):
        local.submit(request)
    assert gone(int(marker.read_text()))
    assert not local._processes and not ledger_entries(tmp_path)
    assert not list((tmp_path / "runs" / ".local-jobs").glob("*.json"))
    monkeypatch.undo()
    again = scheduler(tmp_path)
    again.wait(again.submit(spec(tmp_path, "true\n")), poll_seconds=0.1)


def test_render_shows_the_runner_that_holds_the_job(tmp_path):
    request = spec(tmp_path, "true\n").model_copy(update={"signal_seconds": 900})
    argv = scheduler(tmp_path).render_submit(request)
    assert argv[1:3] == (
        "-I",
        str(Path(allocation.__file__).with_name("local_job.py")),
    )
    assert argv[argv.index("--script") + 1] == str(request.script_path)
    assert argv[argv.index("--walltime") + 1] == str(3 * 3600 + 45 * 60)
    # The advance SIGTERM of a segmented run, as the batch wrapper sends it.
    assert argv[argv.index("--signal-after") + 1] == str(3 * 3600 + 45 * 60 - 900)


@pytest.fixture
def short_walltime(monkeypatch):
    """A job walltime of "00:01" lasts 2 s; SIGKILL follows SIGTERM by 1 s."""
    original = allocation._walltime_seconds
    monkeypatch.setattr(allocation, "WATCHDOG_POLL_SECONDS", 1)
    monkeypatch.setattr(
        allocation,
        "_walltime_seconds",
        lambda value: 2 if value == "00:01" else original(value),
    )
    monkeypatch.setattr(allocation, "KILL_DELAY_SECONDS", 1.0)


@pytest.mark.parametrize("ignores_term", [False, True])
def test_a_job_past_its_walltime_is_terminated_then_killed(
    tmp_path, short_walltime, ignores_term
):
    local = scheduler(tmp_path)
    body = "trap '' TERM\n" if ignores_term else ""
    job_id = local.submit(spec(tmp_path, body + "sleep 300\n", walltime="00:01"))
    started = time.monotonic()
    result = local.wait(job_id, poll_seconds=0.1)
    assert (result.state, result.exit_code) == ("TIMEOUT", 1)
    assert time.monotonic() - started < 10


def test_the_walltime_holds_without_a_controller_and_a_later_one_sees_it(
    tmp_path, short_walltime, until
):
    job_id = scheduler(tmp_path).submit(spec(tmp_path, "sleep 300\n", walltime="00:01"))
    later = scheduler(tmp_path)
    assert until(lambda: later.finished(job_id) is not None, seconds=20)
    ended = later.finished(job_id)
    assert ended.state == "TIMEOUT" and ended.ended_at is not None


def test_a_later_controller_learns_how_and_when_a_job_ended(tmp_path, until):
    job_id = scheduler(tmp_path).submit(spec(tmp_path, "exit 0\n"))
    later = scheduler(tmp_path)
    assert until(lambda: later.finished(job_id) is not None)
    ended = later.finished(job_id)
    assert ended.state == "COMPLETED" and abs(ended.ended_at - time.time()) < 30


def test_a_later_controller_sees_a_live_job_and_can_cancel_it(tmp_path, until):
    local = scheduler(tmp_path)
    job_id = local.submit(spec(tmp_path, "exec sleep 30\n"))
    later = scheduler(tmp_path)
    assert later.finished(job_id) is None
    later.cancel(job_id)
    assert until(lambda: later.finished(job_id) is not None)
    local.wait(job_id, poll_seconds=0.1)


def test_a_job_on_another_node_runs_as_long_as_its_batch_job(tmp_path):
    local = scheduler(tmp_path)
    job_id = local.submit(spec(tmp_path, "exec sleep 30\n"))
    record = tmp_path / "runs" / ".local-jobs" / f"{job_id}.json"
    job = LocalJob.model_validate_json(record.read_text())
    record.write_text(job.model_copy(update={"host": "elsewhere"}).model_dump_json())
    running = scheduler(tmp_path, pbs_job="97401.jobm")
    assert running.finished(job_id) is None
    finished = scheduler(
        tmp_path,
        pbs_job="97401.jobm",
        runner=lambda argv: subprocess.CompletedProcess(
            argv,
            0,
            json.dumps({"Jobs": {argv[-1]: {"job_state": "F", "Exit_status": "0"}}}),
            "",
        ),
    )
    assert finished.finished(job_id).state == "UNKNOWN"
    local.cancel(job_id)


def test_a_name_is_unique_among_live_local_jobs_of_any_batch_job(tmp_path):
    local = scheduler(tmp_path)
    job_id = local.submit(spec(tmp_path, "exec sleep 30\n"))
    elsewhere = scheduler(tmp_path, pbs_job="97401.jobm")
    with pytest.raises(InfrastructureError, match="still running"):
        elsewhere.require_name_available("rsi-task-run-alice")
    with pytest.raises(InfrastructureError, match="still running"):
        elsewhere.submit(spec(tmp_path, "true\n"))
    local.cancel(job_id)
    elsewhere.require_name_available("rsi-task-run-alice")


def test_a_reused_pid_is_not_mistaken_for_the_recorded_job(tmp_path):
    local = scheduler(tmp_path)
    job_id = local.submit(spec(tmp_path, "exec sleep 30\n"))
    for path in [
        *ledger_entries(tmp_path),
        tmp_path / "runs" / ".local-jobs" / f"{job_id}.json",
    ]:
        job = LocalJob.model_validate_json(path.read_text())
        changed = job.model_copy(update={"start_ticks": job.start_ticks - 1})
        path.write_text(changed.model_dump_json())
    later = scheduler(tmp_path)
    later.require_name_available("rsi-task-run-alice")
    assert later.finished(job_id).state == "UNKNOWN"
    local.cancel(job_id)


def test_the_ledger_must_be_a_private_directory(tmp_path):
    local = scheduler(tmp_path)
    shared = (
        tmp_path / "ledger-root" / f"rsi-harness-allocation-{os.getuid()}-{PBS_JOB}"
    )
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(InfrastructureError, match="not a private directory"):
        local.submit(spec(tmp_path, "true\n"))


def test_without_an_end_record_the_job_ended_at_its_last_heartbeat(tmp_path, until):
    """A runner killed outright: resume charges the attempt up to when it
    was last known alive, never less."""
    local = scheduler(tmp_path)
    request = spec(tmp_path, "exec sleep 30\n")
    job_id = local.submit(request)
    heartbeat = local.job_directory(request) / "heartbeat"
    assert until(heartbeat.exists)
    os.killpg(local._processes[job_id].pid, 9)
    local._processes[job_id].wait()
    later = scheduler(tmp_path)
    ended = later.finished(job_id)
    assert ended.state == "UNKNOWN"
    assert ended.ended_at == heartbeat.stat().st_mtime
    heartbeat.unlink()
    assert later.finished(job_id).ended_at is None


def _kill_runner_only(local, job_id, marker, until):
    """Kill a job's runner outright, leaving its payload running in the
    runner's process group."""
    assert until(lambda: marker.exists() and marker.read_text().strip())
    payload = int(marker.read_text())
    runner = local._processes[job_id]
    os.kill(runner.pid, 9)
    assert until(lambda: runner.poll() is not None)
    return payload


def test_a_runner_gone_without_its_record_leaves_no_job_behind(tmp_path, until):
    local = scheduler(tmp_path)
    marker = tmp_path / "payload.pid"
    job_id = local.submit(spec(tmp_path, f"echo $$ > {marker}\nsleep 300\n"))
    assert until(lambda: marker.exists() and marker.read_text().strip())
    payload = int(marker.read_text())
    os.kill(local._processes[job_id].pid, 9)
    result = local.wait(job_id, poll_seconds=0.1)
    assert result.state == "FAILED"
    assert until(lambda: gone(payload))
    assert not ledger_entries(tmp_path)


def test_a_later_controller_sees_a_job_whose_runner_alone_was_killed(tmp_path, until):
    local = scheduler(tmp_path)
    marker = tmp_path / "payload.pid"
    job_id = local.submit(spec(tmp_path, f"echo $$ > {marker}\nsleep 300\n"))
    payload = _kill_runner_only(local, job_id, marker, until)
    later = scheduler(tmp_path)
    assert later.finished(job_id) is None
    with pytest.raises(InfrastructureError, match="still running"):
        later.require_name_available("rsi-task-run-alice")
    later.cancel(job_id)
    assert until(lambda: gone(payload))
    local._processes.pop(job_id)


def test_cancel_ends_the_group_a_dead_runner_left_behind(tmp_path, until):
    local = scheduler(tmp_path)
    marker = tmp_path / "payload.pid"
    job_id = local.submit(spec(tmp_path, f"echo $$ > {marker}\nsleep 300\n"))
    payload = _kill_runner_only(local, job_id, marker, until)
    local.cancel(job_id)
    assert until(lambda: gone(payload))
    assert not ledger_entries(tmp_path)


def test_a_group_counts_only_processes_started_by_the_cut_off(until):
    group = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        assert until(lambda: local_job.group_alive(group.pid))
        started = allocation._process(group.pid)[1]
        assert local_job.group_alive(group.pid, started_by=started)
        assert not local_job.group_alive(group.pid, started_by=started - 1)
    finally:
        group.kill()
        group.wait()


def test_a_submitted_job_stays_the_schedulers(tmp_path):
    """A run begun without allocation launch resumes with it: its job is a
    batch job, which the scheduler follows and deletes."""
    calls = []

    def runner(argv):
        calls.append(argv)
        job = {"job_state": "R"}
        return subprocess.CompletedProcess(
            argv, 0, json.dumps({"Jobs": {argv[-1]: job}}), ""
        )

    local = scheduler(tmp_path, runner=runner)
    assert local.finished("96979.jobm") is None
    local.cancel("96979.jobm")
    assert calls[-1] == ("qdel", "96979.jobm")
