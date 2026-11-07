from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rsi_harness.cluster.schedulers.local_job import (
    STOP_KILL_SECONDS,
    descendants,
    read_end,
    runner_argv,
    stat_fields,
)


@pytest.fixture
def start(tmp_path):
    """Start a runner for a payload; every runner's process group, and every
    process a test records in ``started.pids``, is ended afterwards."""
    runners: list[subprocess.Popen] = []

    def started(
        body,
        *,
        walltime=60.0,
        signal_after=0.0,
        kill_delay=1.0,
        heartbeat=None,
        **popen,
    ):
        script = tmp_path / "run.sh"
        script.write_text("#!/bin/bash\n" + body)
        end = tmp_path / "job.end"
        heartbeat = heartbeat or tmp_path / "job.heartbeat"
        runner = subprocess.Popen(
            runner_argv(
                python=sys.executable,
                script=script,
                end=end,
                heartbeat=heartbeat,
                walltime=walltime,
                signal_after=signal_after,
                kill_delay=kill_delay,
                poll=0.2,
            ),
            start_new_session=True,
            **popen,
        )
        runners.append(runner)
        return runner, end, heartbeat

    started.pids = []
    yield started
    for runner in runners:
        try:
            os.killpg(runner.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        runner.wait()
    for pid in started.pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def alive(pid: int) -> bool:
    fields = stat_fields(pid)
    return fields is not None and fields[0] not in ("Z", "X")


def recorded_pid(path: Path, until) -> int:
    assert until(lambda: path.exists() and path.read_text().strip())
    return int(path.read_text())


def test_the_payloads_status_and_end_are_recorded(start):
    runner, end, heartbeat = start("exit 3\n")
    assert runner.wait(timeout=20) == 3
    recorded = read_end(end)
    assert (recorded.exit_code, recorded.timed_out) == (3, False)
    assert abs(recorded.ended_at - time.time()) < 30
    assert heartbeat.exists()


def test_orphans_in_other_sessions_are_ended_once_the_payload_has(
    start, tmp_path, until
):
    """A Work or Judge container runs in a session of its own; the job must
    not leave it holding a GPU. Its status is still the payload's."""
    marker = tmp_path / "orphan.pid"
    runner, end, _ = start(f"setsid sleep 300 & echo $! > {marker}\nsleep 0.5\n")
    orphan = recorded_pid(marker, until)
    start.pids.append(orphan)
    assert runner.wait(timeout=20) == 0
    assert not alive(orphan)
    assert read_end(end).exit_code == 0


def test_cancel_reaches_processes_in_other_sessions(start, tmp_path, until):
    marker = tmp_path / "work.pid"
    runner, end, _ = start(f"setsid sleep 300 & echo $! > {marker}\nsleep 300\n")
    work = recorded_pid(marker, until)
    start.pids.append(work)
    os.killpg(runner.pid, signal.SIGTERM)
    assert runner.wait(timeout=20) == 128 + signal.SIGTERM
    assert not alive(work)
    assert read_end(end).timed_out is False


@pytest.mark.parametrize("stop", ["signal", "walltime"])
def test_a_stop_signal_kills_what_left_the_group_within_seconds(
    start, tmp_path, until, stop
):
    """The scheduler ending the batch job kills the runner after its own
    kill delay; only the runner reaches a SIGTERM-ignoring Judge session. The
    Engine's group, and a stop at the walltime, keep the full kill delay."""
    marker = tmp_path / "judge.pid"
    runner, _, _ = start(
        "trap '' TERM\n"
        f"setsid bash -c \"trap '' TERM; sleep 300\" & echo $! > {marker}\n"
        "sleep 300\n",
        walltime=60.0 if stop == "signal" else 0.5,
        kill_delay=60.0,
    )
    judge = recorded_pid(marker, until)
    start.pids.append(judge)
    if stop == "signal":
        os.kill(runner.pid, signal.SIGTERM)
    else:
        time.sleep(1.0)  # past the walltime
    killed = until(lambda: not alive(judge), seconds=STOP_KILL_SECONDS + 3)
    assert killed is (stop == "signal")
    assert runner.poll() is None


def test_the_walltime_ends_the_whole_tree_even_a_payload_ignoring_sigterm(
    start, tmp_path, until
):
    marker = tmp_path / "judge.pid"
    runner, end, _ = start(
        "trap '' TERM\n"
        f"setsid bash -c \"trap '' TERM; sleep 300\" & echo $! > {marker}\n"
        "sleep 300\n",
        walltime=1.0,
    )
    judge = recorded_pid(marker, until)
    start.pids.append(judge)
    runner.wait(timeout=20)
    assert not alive(judge)
    assert read_end(end).timed_out is True


def test_the_advance_signal_lets_the_payload_drain(start, tmp_path, until):
    """As Slurm's --signal=B:TERM: the payload (the Engine's driver) drains
    and exits, and the job has not timed out."""
    marker = tmp_path / "work.pid"
    seen = tmp_path / "term"
    runner, end, _ = start(
        f"trap 'echo term > {seen}; exit 0' TERM\n"
        f"setsid sleep 300 & echo $! > {marker}\nsleep 300 & wait\n",
        signal_after=0.5,
    )
    work = recorded_pid(marker, until)
    start.pids.append(work)
    assert runner.wait(timeout=30) == 0
    assert seen.read_text() == "term\n"
    assert read_end(end).timed_out is False
    # What it left in other sessions is ended with the job.
    assert not alive(work)


def test_a_tree_that_keeps_forking_is_ended_all_the_same(start, tmp_path, until):
    """A SIGTERM-ignoring orphan that starts a child every 20 ms: SIGKILL is
    repeated until nothing is left."""
    marker = tmp_path / "forker.pid"
    # A sleep length no other process on this host carries in its arguments.
    tag = f"7.{secrets.randbelow(10**9):09d}"
    runner, _, _ = start(
        "setsid bash -c \"trap '' TERM; "
        f'while :; do (sleep {tag} &); sleep 0.02; done"'
        f" & echo $! > {marker}\nsleep 0.5\n"
    )
    start.pids.append(recorded_pid(marker, until))
    assert runner.wait(timeout=60) == 0
    survivors = []
    for entry in Path("/proc").iterdir():
        try:
            command = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if entry.name.isdigit() and tag.encode() in command:
            fields = stat_fields(int(entry.name)) or []
            survivors.append((entry.name, command.replace(b"\0", b" "), fields[:3]))
    assert survivors == [], f"runner {runner.pid}: {survivors}"


def test_a_heartbeat_that_cannot_be_written_never_stops_the_runner(
    start, tmp_path, until
):
    """Shared storage may fail (ESTALE, a moved log directory): the walltime
    must still hold, the tree still end and the end still be recorded."""
    gone = tmp_path / "gone"
    gone.mkdir()
    heartbeat = gone / "job.heartbeat"
    marker = tmp_path / "work.pid"
    runner, end, _ = start(
        f"setsid sleep 300 & echo $! > {marker}\nsleep 300\n",
        walltime=2.0,
        heartbeat=heartbeat,
    )
    assert until(heartbeat.exists)
    shutil.rmtree(gone)
    work = recorded_pid(marker, until)
    start.pids.append(work)
    runner.wait(timeout=30)
    assert read_end(end).timed_out is True
    assert not alive(work)


def test_nothing_it_cannot_write_stops_the_runner_not_even_its_stderr(
    start, tmp_path, until
):
    """stderr on the same failing storage: even the report of a failed write
    must not end the runner while it holds a Work or Judge session."""
    gone = tmp_path / "gone"
    gone.mkdir()
    marker = tmp_path / "work.pid"
    with open("/dev/full", "w") as full:
        runner, end, _ = start(
            f"setsid sleep 300 & echo $! > {marker}\nexit 5\n",
            heartbeat=gone / "job.heartbeat",
            stderr=full,
        )
    assert until(lambda: (gone / "job.heartbeat").exists())
    shutil.rmtree(gone)
    work = recorded_pid(marker, until)
    start.pids.append(work)
    assert runner.wait(timeout=30) == 5
    assert not alive(work)


def test_an_end_record_that_cannot_be_written_keeps_the_payloads_status(
    start, tmp_path
):
    runner, end, _ = start("exit 7\n")
    end.mkdir()  # the record's own path is taken: replace() must fail
    assert runner.wait(timeout=30) == 7
    assert read_end(end) is None


def test_a_stop_as_the_payload_ends_keeps_its_own_deadline(start, tmp_path, until):
    """Cancel arriving just as the payload exits: what the payload leaves
    behind is ended within the kill delay counted from the stop."""
    marker = tmp_path / "judge.pid"
    runner, _, _ = start(
        "trap 'exit 0' TERM\n"
        f"setsid bash -c \"trap '' TERM; sleep 300\" & echo $! > {marker}\n"
        "sleep 300 & wait\n",
        kill_delay=2.0,
    )
    judge = recorded_pid(marker, until)
    start.pids.append(judge)
    stopped = time.monotonic()
    os.killpg(runner.pid, signal.SIGTERM)
    runner.wait(timeout=30)
    assert not alive(judge)
    # The kill delay from the stop, a poll and a few scans: well under 2 x.
    assert time.monotonic() - stopped < 2.0 * 2


def test_ignored_signals_of_the_controller_never_reach_the_payload(start, tmp_path):
    """A background payload of a non-interactive shell ignores SIGINT and
    SIGQUIT, a nohup'ed controller SIGHUP; the payload must ignore none."""
    shown = tmp_path / "status"

    def ignore_like_nohup():
        for number in (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT):
            signal.signal(number, signal.SIG_IGN)

    runner, _, _ = start(
        f"grep SigIgn /proc/self/status > {shown}\n", preexec_fn=ignore_like_nohup
    )
    runner.wait(timeout=20)
    ignored = int(shown.read_text().split()[1], 16)
    for number in (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT):
        assert not ignored & (1 << (number - 1)), number


def test_the_heartbeat_follows_a_running_job(start, until):
    runner, _, heartbeat = start("sleep 2\n")
    assert until(heartbeat.exists)
    first = heartbeat.stat().st_mtime
    assert until(lambda: heartbeat.stat().st_mtime > first, seconds=5)
    runner.wait(timeout=20)


def test_descendants_finds_the_tree_below_a_process(until):
    child = subprocess.Popen(["bash", "-c", "sleep 30 & wait"])
    try:
        assert until(lambda: len(descendants(child.pid)) == 1)
    finally:
        # The grandchild first: killed after its parent it would outlive it.
        for pid in descendants(child.pid):
            os.kill(pid, signal.SIGKILL)
        os.kill(child.pid, signal.SIGKILL)
        child.wait()
