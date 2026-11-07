"""The process that holds one local job of ``launch = "allocation"``.

    python -I .../local_job.py --script RUN.sh --end END --heartbeat BEAT \\
        --walltime SECONDS --signal-after SECONDS --kill-delay SECONDS \\
        --poll SECONDS

``runner_argv`` builds that command line. Started by path with ``-I``, it
imports only the standard library, whatever the job's environment holds.

It runs the payload (``/bin/bash RUN.sh``) with default signal dispositions,
becomes the subreaper of everything the payload starts, and:

- touches the heartbeat every poll, so a later controller knows when the
  job was last alive even if this process is killed outright;
- sends the payload alone SIGTERM ``--signal-after`` seconds in (0: never),
  as Slurm's ``--signal=B:TERM`` does, so a segmented run drains its Judge;
- at the walltime, or on SIGTERM/SIGINT/SIGHUP/SIGQUIT (cancel), ends the
  whole tree it holds, Work and Judge containers in sessions of their own
  included: SIGTERM, then SIGKILL once the kill delay has passed, repeated
  until the tree is gone, as a scheduler ends a job. On a signal (the
  scheduler may be ending the batch job, and will kill this runner after
  its own, shorter kill delay) what left its process group gets SIGKILL
  after ``STOP_KILL_SECONDS``: only the runner reaches those processes;
- once the payload has ended, ends whatever it left behind (no orphan keeps
  a GPU), keeping the deadline an earlier stop set, then writes the end
  record ``<exit> <epoch seconds> <done|timeout>`` atomically and exits with
  the payload's status.

Nothing it cannot write (heartbeat, end record, even stderr) stops it.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

_PR_SET_CHILD_SUBREAPER = 36
END_FORMAT = "{exit} {ended:.3f} {how}"
# While ending a tree: how often it is scanned again, and how long processes
# may linger after SIGKILL (one in uninterruptible I/O dies when it returns).
_RESCAN_SECONDS = 0.1
VANISH_SECONDS = 30.0
# After a stop signal: how soon it is seen, and how long the processes
# outside the runner's group get before SIGKILL, well within a scheduler's
# kill delay (PBS: 10 s by default).
_STOP_CHECK_SECONDS = 1.0
STOP_KILL_SECONDS = 3.0


class LocalEnd(NamedTuple):
    """How a local job ended, as its runner recorded it."""

    exit_code: int
    ended_at: float
    timed_out: bool


def read_end(path: Path) -> LocalEnd | None:
    try:
        status, ended, how = path.read_text().split()
        return LocalEnd(int(status), float(ended), how == "timeout")
    except (OSError, ValueError):
        return None


def stat_fields(pid: int) -> list[str] | None:
    """``/proc/<pid>/stat`` from its third field (state, ppid, pgrp, ...):
    the command name before it may hold spaces and parentheses."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def shell_status(code: int) -> int:
    """A Popen return code as a shell exit status (signal N -> 128 + N)."""
    return 128 - code if code < 0 else code


def runner_argv(
    *,
    python: str,
    script: Path,
    end: Path,
    heartbeat: Path,
    walltime: float,
    signal_after: float,
    kill_delay: float,
    poll: float,
) -> tuple[str, ...]:
    """The command that runs ``script`` as a local job."""
    return (
        python,
        "-I",
        str(Path(__file__).resolve()),
        "--script",
        str(script),
        "--end",
        str(end),
        "--heartbeat",
        str(heartbeat),
        "--walltime",
        str(walltime),
        "--signal-after",
        str(signal_after),
        "--kill-delay",
        str(kill_delay),
        "--poll",
        str(poll),
    )


def worst_case_seconds(kill_delay: float, poll: float) -> float:
    """How long a runner may take to end its job after a stop request: the
    poll that notices it, the kill delay, and the lingering after SIGKILL."""
    return poll + kill_delay + VANISH_SECONDS


def processes() -> dict[int, list[str]]:
    """Every process's ``stat_fields``, by PID: one walk of /proc."""
    found: dict[int, list[str]] = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            fields = stat_fields(int(entry.name))
            if fields is not None and len(fields) > 2:
                found[int(entry.name)] = fields
    return found


def boot_ticks() -> int:
    """Now, in the clock ticks since boot that /proc start times use."""
    uptime = float(Path("/proc/uptime").read_text().split()[0])
    return int(uptime * os.sysconf("SC_CLK_TCK"))


def group_alive(pgid: int, *, started_by: int | None = None) -> bool:
    """Some process of process group ``pgid`` still runs (a group outlives
    its leader). With ``started_by`` (boot ticks), only processes started by
    then count: once its leader is reaped and the group has emptied, the
    leader's PID may lead a new, unrelated group."""
    return any(
        fields[2] == str(pgid)
        and fields[0] not in ("Z", "X")
        and (started_by is None or int(fields[19]) <= started_by)
        for fields in processes().values()
        if len(fields) > 19
    )


def descendants(root: int) -> tuple[int, ...]:
    """Every live process below ``root`` (orphans of a subreaper stay below
    it), in no particular order."""
    children: dict[int, list[int]] = {}
    for pid, fields in processes().items():
        if fields[1].isdigit():
            children.setdefault(int(fields[1]), []).append(pid)
    found, frontier = [], [root]
    while frontier:
        for child in children.get(frontier.pop(), ()):
            found.append(child)
            frontier.append(child)
    return tuple(found)


def _signal(pids: Iterable[int], number: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, number)
        except (ProcessLookupError, PermissionError):
            pass


def _outside_group(pids: Iterable[int]) -> tuple[int, ...]:
    """Those of ``pids`` not in this process's group (other sessions)."""
    group = str(os.getpgrp())
    return tuple(
        pid
        for pid in pids
        if (fields := stat_fields(pid)) is not None and fields[2] != group
    )


def _reap(keep: int | None = None) -> None:
    """Collect adopted orphans that have exited; never ``keep`` (the payload,
    whose status Popen must collect itself)."""
    while True:
        try:
            exited = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return
        if exited is None or exited.si_pid == keep:
            return
        try:
            os.waitpid(exited.si_pid, 0)
        except ChildProcessError:
            return


_reported: set[str] = set()


def _warn_once(what: str, error: OSError) -> None:
    """Report a write that failed, once; stderr may sit on the same failing
    storage, so not even the report may stop the runner."""
    if what not in _reported:
        _reported.add(what)
        try:
            print(f"local_job: cannot write {what}: {error}", file=sys.stderr)
        except OSError:
            pass


def _touch(path: Path) -> None:
    try:
        path.touch()
    except OSError as error:
        _warn_once(str(path), error)


def _end_tree(kill_at: float, stop: _Stop) -> None:
    """End every process below this one: SIGTERM, SIGKILL at ``kill_at``
    (monotonic; for those outside its group, STOP_KILL_SECONDS after a stop
    signal), again and again (a process may fork between scan and signal),
    until two scans in a row find none or VANISH_SECONDS have passed."""
    me = os.getpid()
    _signal(descendants(me), signal.SIGTERM)
    while (left := descendants(me)) and time.monotonic() < kill_at:
        if stop.due(STOP_KILL_SECONDS):
            _signal(_outside_group(left), signal.SIGKILL)
        _reap()
        time.sleep(_RESCAN_SECONDS)
    vanish = time.monotonic() + VANISH_SECONDS
    empty_scans = 0
    # A child forked while its parent was being killed may appear in /proc
    # only after the scan that found none.
    while empty_scans < 2 and time.monotonic() < vanish:
        left = descendants(me)
        empty_scans = 0 if left else empty_scans + 1
        _signal(left, signal.SIGKILL)
        _reap()
        time.sleep(_RESCAN_SECONDS)
    _reap()
    if left := descendants(me):
        try:
            print(
                f"local_job: {len(left)} processes outlived SIGKILL: {sorted(left)}",
                file=sys.stderr,
            )
        except OSError:
            pass


def _write_end(path: Path, exit_status: int, timed_out: bool) -> None:
    record = END_FORMAT.format(
        exit=exit_status, ended=time.time(), how="timeout" if timed_out else "done"
    )
    partial = path.with_name(f"{path.name}.partial")
    try:
        partial.write_text(record + "\n")
        partial.replace(path)
    except OSError as error:
        # The controller then has the heartbeat (or nothing) to go by.
        _warn_once(str(path), error)


class _Stop:
    """When the first stop signal (TERM, INT, HUP, QUIT) arrived, if one did."""

    def __init__(self) -> None:
        self.at: float | None = None

    def __call__(self, *_: object) -> None:
        if self.at is None:
            self.at = time.monotonic()

    def due(self, seconds: float) -> bool:
        """A stop arrived at least ``seconds`` ago."""
        return self.at is not None and time.monotonic() >= self.at + seconds


def run(
    script: Path,
    end: Path,
    heartbeat: Path,
    *,
    walltime: float,
    signal_after: float = 0.0,
    kill_delay: float,
    poll: float,
) -> int:
    """Run the payload until it ends, its walltime passes or a stop arrives;
    then end everything below this process, record the end, and return the
    payload's status."""
    for stale in (end, end.with_name(f"{end.name}.partial")):
        try:
            stale.unlink(missing_ok=True)
        except OSError as error:
            _warn_once(str(stale), error)
    _touch(heartbeat)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER)")
    stop = _Stop()
    # Handled here, these never kill the runner outright, and exec gives the
    # payload their default dispositions: inherited ignores (a nohup'ed or
    # background controller) do not reach it.
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(number, stop)
    me = os.getpid()
    payload = subprocess.Popen(["/bin/bash", str(script)])
    started = touched = time.monotonic()
    timed_out = advance_sent = False
    kill_at: float | None = None
    while (code := payload.poll()) is None:
        now = time.monotonic()
        if now - touched >= poll:
            _touch(heartbeat)
            touched = now
        if kill_at is None and (now - started >= walltime or stop.at is not None):
            timed_out = stop.at is None
            # This reaches the Work and Judge sessions too.
            _signal(descendants(me), signal.SIGTERM)
            kill_at = (stop.at if stop.at is not None else now) + kill_delay
        elif kill_at is not None and now >= kill_at:
            _signal(descendants(me), signal.SIGKILL)
        elif signal_after and not advance_sent and now - started >= signal_after:
            advance_sent = True
            _signal((payload.pid,), signal.SIGTERM)
        if stop.due(STOP_KILL_SECONDS):
            _signal(_outside_group(descendants(me)), signal.SIGKILL)
        _reap(keep=payload.pid)
        try:
            # Returns as soon as the payload ends.
            payload.wait(timeout=min(poll, _STOP_CHECK_SECONDS))
        except subprocess.TimeoutExpired:
            pass
    _touch(heartbeat)
    # Whatever the payload left behind must not keep the job's GPUs; a stop
    # (also one that arrived as the payload ended) keeps its own deadlines.
    if kill_at is None:
        kill_at = (stop.at if stop.at is not None else time.monotonic()) + kill_delay
    _end_tree(kill_at, stop)
    status = shell_status(code)
    _write_end(end, status, timed_out)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one local job.")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--end", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--walltime", type=float, required=True)
    parser.add_argument("--signal-after", type=float, default=0.0)
    parser.add_argument("--kill-delay", type=float, required=True)
    parser.add_argument("--poll", type=float, required=True)
    args = parser.parse_args(argv)
    return run(
        args.script,
        args.end,
        args.heartbeat,
        walltime=args.walltime,
        signal_after=args.signal_after,
        kill_delay=args.kill_delay,
        poll=args.poll,
    )


if __name__ == "__main__":
    raise SystemExit(main())
