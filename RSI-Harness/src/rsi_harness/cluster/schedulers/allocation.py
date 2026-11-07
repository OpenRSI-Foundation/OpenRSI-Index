"""Run Harness jobs inside the controller's own PBS or PJM job.

With ``scheduler.launch = "allocation"`` the controller runs in a batch job
on a compute node and starts each Harness job (an image build, a run
attempt) there as a local process group instead of submitting a new job:
a run queues once and no controller job waits on an idle GPU. The local job
gets the controller's environment as it is (the batch job's GPUs, node
scratch and credentials; nothing reaches the server's job records), with
the GPUs the batch job requested where CUDA_VISIBLE_DEVICES is unset, and
the Engine uses the controller's interpreter and Agent CLI.

One local job at a time holds a batch job, on its one node. Each runs under
``local_job``, a small runner that is the subreaper of everything the job
starts: it enforces the walltime itself (SIGTERM to the whole tree, Work
and Judge containers in sessions of their own included, then SIGKILL),
ends whatever the payload leaves behind, touches a heartbeat and records
how and when the job ended, in the job's directory. A record of each local
job is kept on shared storage (``record_root``) so a later controller,
possibly in another batch job on another node, can tell whether the job
still runs and when it ended (``finished``, which ``resume`` calls) and
refuse a second live job of the same name. A local job's walltime must end
before its batch job's does.
"""

from __future__ import annotations

import fcntl
import functools
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from pydantic import ValidationError

from rsi_harness.cluster.schedulers.batch import (
    BatchJobResult,
    BatchJobSpec,
    BatchScheduler,
    bounded_name,
)
from rsi_harness.cluster.schedulers.local_job import (
    boot_ticks,
    group_alive,
    read_end,
    runner_argv,
    shell_status,
    stat_fields,
    worst_case_seconds,
)
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import PersistedModel

KILL_DELAY_SECONDS = 60.0
# How often a runner touches its heartbeat.
WATCHDOG_POLL_SECONDS = 5.0
# A local job must leave this much of the batch job's walltime for the
# controller's own teardown.
TEARDOWN_SECONDS = 300.0
_LEDGER_PREFIX = "rsi-harness-allocation"
# A local job's ID: its batch job's, this, and a token.
_LOCAL = ".local-"
# Set in a local job whose batch job leaves CUDA_VISIBLE_DEVICES unset: the
# GPUs that job requested, as its scheduler tells (the Engine's device fence).
JOB_GPUS_VARIABLE = "RSI_HARNESS_JOB_GPUS"


def _cancel_grace() -> float:
    """How long cancel waits for the runner to end the job's tree before it
    kills the job's process group."""
    return worst_case_seconds(KILL_DELAY_SECONDS, WATCHDOG_POLL_SECONDS) + 30


def _backstop(deadline: float) -> float:
    """The controller's own deadline for a job, should its runner be gone."""
    return deadline + _cancel_grace() + 60


def _walltime_seconds(value: str) -> int:
    hours, minutes = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60


class LocalJob(PersistedModel):
    """A local job: where it runs and where it records its end."""

    job_id: str
    name: str
    allocation: str
    host: str
    pid: int
    start_ticks: int | None
    deadline: float
    end_path: Path
    heartbeat_path: Path


def _process(pid: int) -> tuple[str, int] | None:
    """The process's state letter and start time in boot ticks; the start
    time tells a reused PID from the recorded one."""
    fields = stat_fields(pid)
    try:
        return None if fields is None else (fields[0], int(fields[19]))
    except (IndexError, ValueError):
        return None


def _runs_here(job: LocalJob) -> bool:
    """The job runs on this host: its runner, or (the runner killed outright)
    some process of the runner's group, which outlives its leader. A runner
    PID held by a process started at another time means the group is gone:
    a PID is not reused while it names a live group."""
    if job.host != socket.gethostname():
        return False
    process = _process(job.pid)
    if process is not None and process[1] != job.start_ticks:
        return False
    if process is not None and process[0] not in ("Z", "X"):
        return True
    return group_alive(job.pid)


class AllocationScheduler:
    """The batch scheduler interface, run within the current batch job of
    ``host`` (a PBSScheduler or PJMScheduler, which still answers for the
    batch jobs themselves)."""

    def __init__(
        self,
        host: BatchScheduler,
        *,
        record_root: Path,
        environ: Mapping[str, str] | None = None,
        ledger_root: Path = Path("/tmp"),
        sleeper: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
        token: Callable[[], str] = lambda: secrets.token_hex(4),
    ) -> None:
        self.host = host
        self.label = host.label
        self._record_root = record_root
        self._environ = os.environ if environ is None else environ
        self._ledger_root = ledger_root
        self._sleeper = sleeper
        self._wall_clock = wall_clock
        self._token = token
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._cancelled: set[str] = set()
        self._directories: dict[str, tuple[Path, str]] = {}

    # --- the batch job that holds the run -------------------------------

    def allocation(self) -> str:
        """The batch job this controller runs in; refused outside one."""
        job_id = self._environ.get(self.host.job_id_variable, "").strip()
        if not job_id:
            raise SetupError(
                f'scheduler.launch = "allocation" runs inside a {self.label} job: '
                "start rsi-harness from a job script on a compute node"
            )
        return job_id

    @functools.cached_property
    def _requested_gpus(self) -> int | None:
        return self.host.job_gpus(self.allocation(), self._environ)

    def require_room(self, specs: tuple[BatchJobSpec, ...]) -> None:
        """Refuse jobs the current batch job cannot hold: its GPUs (where
        the scheduler names or tells them) and the walltime it has left, for
        all of ``specs`` one after another."""
        allocation = self.allocation()
        devices = self._environ.get("CUDA_VISIBLE_DEVICES")
        if devices is not None:
            held = len([device for device in devices.split(",") if device])
            source = "CUDA_VISIBLE_DEVICES"
        else:
            held, source = self._requested_gpus, "requested"
        for spec in specs:
            if held is not None and spec.gpu_count > held:
                raise SetupError(
                    f"{self.label} job {allocation} holds {held} GPUs "
                    f"({source}); the {spec.name} job needs "
                    f"{spec.gpu_count}: request a larger {self.label} job"
                )
        needed = sum(_walltime_seconds(spec.walltime) for spec in specs)
        left = self.host.walltime_left(allocation, self._environ)
        if left < needed + TEARDOWN_SECONDS:
            raise SetupError(
                f"{self.label} job {allocation} has {max(0.0, left) / 3600:.2f} h "
                f"of walltime left; {', '.join(spec.name for spec in specs)} "
                f"need {needed / 3600:.2f} h plus {TEARDOWN_SECONDS / 60:.0f} min: "
                "request a longer walltime"
            )

    # --- names, files and records ---------------------------------------

    def _job_files(self, spec: BatchJobSpec) -> tuple[Path, str]:
        """The job's directory (end record and heartbeat, beside its logs)
        and the token its ID ends with."""
        key = str(spec.script_path)
        if key not in self._directories:
            token = self._token()
            name = bounded_name(spec.name, self.host.name_limit)
            self._directories[key] = (
                spec.stdout_path.parent / f"{name}-{token}",
                token,
            )
        return self._directories[key]

    def job_directory(self, spec: BatchJobSpec) -> Path:
        return self._job_files(spec)[0]

    def render_submit(self, spec: BatchJobSpec) -> tuple[str, ...]:
        directory = self.job_directory(spec)
        walltime = _walltime_seconds(spec.walltime)
        return runner_argv(
            python=sys.executable,
            script=spec.script_path,
            end=directory / "end",
            heartbeat=directory / "heartbeat",
            walltime=walltime,
            signal_after=(
                max(1, walltime - spec.signal_seconds) if spec.signal_seconds else 0
            ),
            kill_delay=KILL_DELAY_SECONDS,
            poll=WATCHDOG_POLL_SECONDS,
        )

    def _ledger(self, allocation: str) -> Path:
        """This batch job's node-local ledger of its live local job."""
        name = f"{_LEDGER_PREFIX}-{os.getuid()}-{allocation}"
        return _private_directory(self._ledger_root / name)

    @contextmanager
    def _locked(self, allocation: str) -> Iterator[Path]:
        ledger = self._ledger(allocation)
        descriptor = os.open(
            ledger / ".lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield ledger
        finally:
            os.close(descriptor)

    @staticmethod
    def _jobs(directory: Path) -> dict[str, LocalJob]:
        jobs: dict[str, LocalJob] = {}
        for path in directory.glob("*.json"):
            try:
                job = LocalJob.model_validate_json(path.read_text())
            except (OSError, ValidationError):
                continue
            jobs[job.job_id] = job
        return jobs

    @staticmethod
    def _write(directory: Path, job: LocalJob) -> None:
        path = directory / f"{job.job_id}.json"
        partial = path.with_suffix(".partial")
        partial.write_text(job.model_dump_json())
        partial.chmod(0o600)
        partial.replace(path)

    def _record(self, job_id: str) -> LocalJob | None:
        try:
            text = (self._record_root / f"{job_id}.json").read_text()
            return LocalJob.model_validate_json(text)
        except (OSError, ValidationError):
            return None

    def _live(self, ledger: Path) -> list[LocalJob]:
        """The ledger's jobs that still run; the others leave it."""
        live = []
        for job_id, job in self._jobs(ledger).items():
            if _runs_here(job):
                live.append(job)
            else:
                (ledger / f"{job_id}.json").unlink(missing_ok=True)
        return live

    def _still_runs(self, job: LocalJob) -> bool:
        if job.end_path.exists():
            return False
        if job.host == socket.gethostname():
            return _runs_here(job)
        # On another node: it runs as long as its batch job does.
        return self.host.finished(job.allocation) is None

    def _name_is_live(self, name: str, ledger: Path | None) -> bool:
        if ledger is not None and any(job.name == name for job in self._live(ledger)):
            return True
        if not self._record_root.is_dir():
            return False
        # Past its backstop a job cannot run (its runner ended it): only
        # recent records are asked about, however many accumulate.
        now = self._wall_clock()
        return any(
            job.name == name and _backstop(job.deadline) > now and self._still_runs(job)
            for job in self._jobs(self._record_root).values()
        )

    # --- the scheduler interface ----------------------------------------

    def require_name_available(self, name: str) -> None:
        # A submitted job of that name (a run started without allocation
        # launch) holds it too.
        self.host.require_name_available(name)
        allocation = self._environ.get(self.host.job_id_variable, "").strip()
        if allocation:
            with self._locked(allocation) as ledger:
                live = self._name_is_live(name, ledger)
        else:
            live = self._name_is_live(name, None)
        if live:
            raise InfrastructureError(f"a local job named {name} is still running")

    def submit(self, spec: BatchJobSpec) -> str:
        if not spec.script_path.is_file():
            raise SetupError(f"payload does not exist: {spec.script_path}")
        allocation = self.allocation()
        self.require_room((spec,))
        records = _private_directory(self._record_root)
        directory, token = self._job_files(spec)
        job_id = f"{allocation}{_LOCAL}{token}"
        with self._locked(allocation) as ledger:
            if self._name_is_live(spec.name, ledger):
                raise InfrastructureError(
                    f"a local job named {spec.name} is still running"
                )
            if live := self._live(ledger):
                raise InfrastructureError(
                    f"{self.label} job {allocation} already runs "
                    f"{', '.join(sorted(job.name for job in live))}; one local "
                    f"job at a time holds a {self.label} job"
                )
            _private_directory(directory)
            try:
                process = self._start(spec, job_id)
                stat = _process(process.pid)
                job = LocalJob(
                    job_id=job_id,
                    name=spec.name,
                    allocation=allocation,
                    host=socket.gethostname(),
                    pid=process.pid,
                    start_ticks=None if stat is None else stat[1],
                    deadline=self._wall_clock() + _walltime_seconds(spec.walltime),
                    end_path=directory / "end",
                    heartbeat_path=directory / "heartbeat",
                )
                self._write(ledger, job)
                self._write(records, job)
            except BaseException:
                # An interrupt (or a failed write) before the caller has the
                # ID: nobody else could end the job.
                if (started := self._processes.pop(job_id, None)) is not None:
                    self._end_own(started)
                    started.wait()
                for root in (ledger, records):
                    (root / f"{job_id}.json").unlink(missing_ok=True)
                raise
        return job_id

    def _start(self, spec: BatchJobSpec, job_id: str) -> subprocess.Popen[bytes]:
        def log(path: Path) -> int:
            return os.open(
                str(path).replace("%j", job_id),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
            )

        environment = dict(self._environ)
        environment.pop(JOB_GPUS_VARIABLE, None)
        if "CUDA_VISIBLE_DEVICES" not in environment:
            if (gpus := self._requested_gpus) is not None:
                environment[JOB_GPUS_VARIABLE] = str(gpus)
        stdout = log(spec.stdout_path)
        try:
            stderr = log(spec.stderr_path)
            try:
                # Kept at once: an interrupt from here on still finds it.
                process = self._processes[job_id] = subprocess.Popen(
                    self.render_submit(spec),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env=environment,
                    # Its own process group (cancel ends the job, never the
                    # controller) in the batch job's session, which the
                    # scheduler ends with the batch job.
                    process_group=0,
                    close_fds=True,
                )
            finally:
                os.close(stderr)
        finally:
            os.close(stdout)
        return process

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float = 5.0,
        on_state: Callable[[str], None] | None = None,
    ) -> BatchJobResult:
        process = self._processes.get(job_id)
        job = self._record(job_id)
        if process is None or job is None:
            raise InfrastructureError(
                f"local job {job_id} was not started by this controller"
            )
        if on_state is not None:
            on_state("RUNNING")
        # The runner ends the job at its walltime; this is the backstop
        # should the runner itself be gone.
        backstop = _backstop(job.deadline)
        try:
            while True:
                try:
                    code: int | None = process.wait(timeout=max(0.1, poll_seconds))
                    break
                except subprocess.TimeoutExpired:
                    pass
                if self._wall_clock() >= backstop:
                    self._end_own(process)
                    code = process.wait()
                    break
        except KeyboardInterrupt:
            self.cancel(job_id)
            raise KeyboardInterrupt(
                f"interrupted; {self.label} local job {job_id} cancelled"
            ) from None
        if read_end(job.end_path) is None:
            # The runner died without ending its job: its process group (the
            # payload, the Engine) outlives it.
            reaped = boot_ticks()
            self._end_group(
                process.pid, lambda: group_alive(process.pid, started_by=reaped)
            )
        result = self._result(job, code)
        self._release(job)
        if on_state is not None:
            on_state(result.state)
        return result

    def _result(self, job: LocalJob, code: int | None) -> BatchJobResult:
        """Slurm states, as the batch schedulers report them. Without the
        runner's end record (the runner or its batch job killed outright)
        the job ended at its last heartbeat, or at an unknown time (None),
        which resume charges in full."""
        recorded = read_end(job.end_path)
        if recorded is None:
            try:
                ended_at: float | None = job.heartbeat_path.stat().st_mtime
            except OSError:
                ended_at = None
            if code is None:
                return BatchJobResult(
                    job_id=job.job_id, state="UNKNOWN", exit_code=1, ended_at=ended_at
                )
            exit_code, timed_out = shell_status(code), False
        else:
            exit_code, ended_at, timed_out = recorded
        if timed_out:
            state, exit_code = "TIMEOUT", 1
        elif job.job_id in self._cancelled:
            state, exit_code = "CANCELLED", exit_code or 1
        elif exit_code == 0:
            state = "COMPLETED"
        else:
            state = "FAILED"
        return BatchJobResult(
            job_id=job.job_id, state=state, exit_code=exit_code, ended_at=ended_at
        )

    def _release(self, job: LocalJob) -> None:
        self._processes.pop(job.job_id, None)
        with self._locked(job.allocation) as ledger:
            (ledger / f"{job.job_id}.json").unlink(missing_ok=True)

    def _end_own(self, process: subprocess.Popen[bytes]) -> None:
        """End a runner this controller started, or the group it left."""
        seen = boot_ticks()
        self._end_group(
            process.pid,
            lambda: process.poll() is None or group_alive(process.pid, started_by=seen),
        )

    def _end_group(self, pid: int, running: Callable[[], bool]) -> None:
        """SIGTERM to the job's process group: its runner then ends every
        process it holds (other sessions too) within its kill delay; SIGKILL
        to the group only if that does not happen in time. Nothing is sent
        to a group that has gone (its leader's PID may be reused)."""
        if not running():
            return
        _signal_group(pid, signal.SIGTERM)
        ends = time.monotonic() + _cancel_grace()
        while running() and time.monotonic() < ends:
            self._sleeper(0.2)
        if running():
            _signal_group(pid, signal.SIGKILL)

    def finished(self, job_id: str) -> BatchJobResult | None:
        """The job's end, or None while it runs. A job with no record, or
        whose batch job ended without its runner recording the end, is
        ``UNKNOWN`` (ended at its last heartbeat, if any). A submitted job
        (the run began without allocation launch) is the scheduler's."""
        if _LOCAL not in job_id:
            return self.host.finished(job_id)
        job = self._record(job_id)
        if job is None:
            return BatchJobResult(job_id=job_id, state="UNKNOWN", exit_code=1)
        process = self._processes.get(job_id)
        if process is not None:
            code = process.poll()
            return None if code is None else self._result(job, code)
        if self._still_runs(job):
            return None
        return self._result(job, None)

    def cancel(self, job_id: str) -> None:
        """End the job's process group; a job already gone, or on another
        node, is left to its runner and batch job."""
        if _LOCAL not in job_id:
            self.host.cancel(job_id)
            return
        self._cancelled.add(job_id)
        job = self._record(job_id)
        process = self._processes.get(job_id)
        if process is not None:
            self._end_own(process)
            process.wait()
        elif job is not None and _runs_here(job):
            self._end_group(job.pid, lambda: _runs_here(job))
        if job is not None and job.host == socket.gethostname():
            self._release(job)


def _private_directory(path: Path) -> Path:
    """``path`` as a directory only this user can enter, created if needed."""
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = path.lstat()
    except OSError as error:
        raise InfrastructureError(
            f"cannot create the directory {path}: {error.strerror}"
        ) from None
    if (
        path.is_symlink()
        or not path.is_dir()
        or info.st_uid != os.getuid()
        or info.st_mode & 0o077
    ):
        raise InfrastructureError(f"{path} is not a private directory of this user")
    return path


def _signal_group(pid: int, number: int) -> None:
    try:
        os.killpg(pid, number)
    except ProcessLookupError:
        pass


__all__ = ["AllocationScheduler", "LocalJob"]
