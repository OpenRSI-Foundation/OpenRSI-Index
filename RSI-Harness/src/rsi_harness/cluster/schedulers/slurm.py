"""Synchronous Slurm submission with exact-job accounting and cancellation."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable

from pydantic import field_validator

from rsi_harness.cluster.schedulers.base import JobResult, JobSpec
from rsi_harness.errors import InfrastructureError, SetupError

_JOB_ID = re.compile(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?")
_TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
_ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
    "RESV_DEL_HOLD",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
    "SUSPENDED",
}
_TRANSIENT_ERRORS = (
    "socket timed out",
    "communication time out",
    "connection timed out",
)
CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


def _run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)


class SlurmJobSpec(JobSpec):
    qos: str | None = None
    constraint: str | None = None

    @field_validator("walltime")
    @classmethod
    def _walltime(cls, value: str) -> str:
        if re.fullmatch(r"\d+:[0-5]\d", value) is None or not int(
            value.replace(":", "")
        ):
            raise ValueError("cluster walltime must be positive HH:MM")
        return value


class SlurmJobResult(JobResult):
    """Terminal allocation status, excluding individual step records."""


class SlurmScheduler:
    def __init__(
        self,
        *,
        submit_binary: str = "sbatch",
        status_binary: str = "squeue",
        cancel_binary: str = "scancel",
        accounting_binary: str = "sacct",
        runner: CommandRunner = _run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._submit_binary = submit_binary
        self._status_binary = status_binary
        self._cancel_binary = cancel_binary
        self._accounting_binary = accounting_binary
        self._runner = runner
        self._sleeper = sleeper

    def render_submit(self, spec: SlurmJobSpec) -> tuple[str, ...]:
        # The shared planner uses HH:MM; Slurm reads two fields as MM:SS.
        hours, minutes = (int(part) for part in spec.walltime.split(":"))
        argv = [
            self._submit_binary,
            "--parsable",
            f"--job-name={spec.name}",
            f"--partition={spec.queue}",
            f"--nodes={spec.hosts}",
            f"--ntasks={spec.hosts}",
            "--ntasks-per-node=1",
            f"--cpus-per-task={spec.slots_per_host or spec.cpu_slots}",
            f"--mem={spec.memory_mb}M",
            f"--time={hours * 60 + minutes}",
            "--export=ALL",
        ]
        if spec.group:
            argv.append(f"--account={spec.group}")
        if spec.qos:
            argv.append(f"--qos={spec.qos}")
        if spec.constraint:
            argv.append(f"--constraint={spec.constraint}")
        if spec.gpu_count:
            argv.append(f"--gres=gpu:{spec.gpu_count}")
        if spec.local_tmp_mb:
            argv.append(f"--tmp={spec.local_tmp_mb}M")
        if spec.exclusive:
            argv.append("--exclusive")
        if spec.excluded_hosts:
            argv.append(f"--exclude={','.join(spec.excluded_hosts)}")
        argv.extend(
            (
                f"--output={spec.stdout_path}",
                f"--error={spec.stderr_path}",
                str(spec.script_path),
            )
        )
        return tuple(argv)

    def submit(self, spec: SlurmJobSpec) -> str:
        if not spec.script_path.is_file():
            raise SetupError(f"Slurm payload does not exist: {spec.script_path}")
        completed = self._runner(self.render_submit(spec))
        if completed.returncode:
            raise InfrastructureError(
                f"Slurm submission failed: {self._detail(completed)}"
            )
        match = _JOB_ID.fullmatch(completed.stdout.strip())
        if match is None:
            raise InfrastructureError("Slurm submission returned no exact job ID")
        return match.group(1)

    def require_name_available(self, name: str) -> None:
        completed = self._runner((self._status_binary, "--noheader", "--format=%j"))
        if completed.returncode:
            raise InfrastructureError(
                f"cannot inspect active Slurm jobs: {self._detail(completed)}"
            )
        if name in {line.strip() for line in completed.stdout.splitlines()}:
            raise InfrastructureError(f"Slurm job name is already active: {name}")

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float = 5.0,
        on_state: Callable[[str], None] | None = None,
    ) -> SlurmJobResult:
        self._require_job_id(job_id)
        previous_state: str | None = None
        missing_polls = 0
        try:
            while True:
                completed = self._runner(
                    (
                        self._status_binary,
                        "--noheader",
                        f"--jobs={job_id}",
                        "--format=%i|%T",
                    )
                )
                if completed.returncode:
                    detail = self._detail(completed).lower()
                    if any(marker in detail for marker in _TRANSIENT_ERRORS):
                        self._sleeper(poll_seconds)
                        continue
                    if "invalid job id" not in detail:
                        raise InfrastructureError(
                            f"cannot query Slurm job {job_id}: {detail}"
                        )
                state = self._exact_status(job_id, completed.stdout, accounting=False)
                result = None
                if state is None or state[0] in _TERMINAL_STATES:
                    accounting = self._runner(
                        (
                            self._accounting_binary,
                            "--noheader",
                            "--parsable2",
                            "--allocations",
                            f"--jobs={job_id}",
                            "--format=JobIDRaw,State%32,ExitCode",
                        )
                    )
                    if accounting.returncode:
                        detail = self._detail(accounting)
                        if any(
                            marker in detail.lower() for marker in _TRANSIENT_ERRORS
                        ):
                            self._sleeper(poll_seconds)
                            continue
                        raise InfrastructureError(
                            f"cannot query Slurm accounting for {job_id}: {detail}"
                        )
                    state = self._exact_status(
                        job_id, accounting.stdout, accounting=True
                    )
                    if state is not None and state[0] in _TERMINAL_STATES:
                        result = SlurmJobResult(
                            job_id=job_id, state=state[0], exit_code=state[1]
                        )
                if state is None:
                    # Accounting may lag behind squeue, but never wait forever
                    # for a missing/purged job or disabled accounting service.
                    missing_polls += 1
                    if missing_polls >= 12:
                        raise InfrastructureError(
                            f"Slurm job {job_id} is absent from queue and accounting"
                        )
                else:
                    missing_polls = 0
                    if state[0] != previous_state and on_state is not None:
                        on_state(state[0])
                    previous_state = state[0]
                if result is not None:
                    return result
                self._sleeper(poll_seconds)
        except KeyboardInterrupt:
            self.cancel(job_id)
            raise

    def cancel(self, job_id: str) -> None:
        self._require_job_id(job_id)
        completed = self._runner((self._cancel_binary, job_id))
        if completed.returncode:
            raise InfrastructureError(
                f"cannot cancel Slurm job {job_id}: {self._detail(completed)}"
            )

    @staticmethod
    def _require_job_id(job_id: str) -> None:
        if re.fullmatch(r"[0-9]+", job_id) is None:
            raise SetupError(f"invalid exact Slurm job ID: {job_id!r}")

    @staticmethod
    def _detail(completed: subprocess.CompletedProcess[str]) -> str:
        return completed.stderr.strip() or completed.stdout.strip()

    @staticmethod
    def _exact_status(
        job_id: str,
        output: str,
        *,
        accounting: bool,
    ) -> tuple[str, int] | None:
        rows = [
            fields
            for line in output.splitlines()
            if (fields := line.strip().split("|"))[0] == job_id
        ]
        if not rows:
            return None
        if len(rows) != 1 or len(rows[0]) != (3 if accounting else 2):
            raise InfrastructureError(f"invalid Slurm status for exact job {job_id}")
        # sacct may append a cancelling UID ("CANCELLED by 1234").
        state = rows[0][1].strip().split(" ", 1)[0].upper()
        if state not in _ACTIVE_STATES | _TERMINAL_STATES:
            raise InfrastructureError(f"unknown Slurm state for {job_id}: {state!r}")
        exit_code = 0
        if accounting:
            match = re.fullmatch(r"([0-9]+):([0-9]+)", rows[0][2].strip())
            if match is None:
                raise InfrastructureError(f"invalid Slurm exit code for {job_id}")
            code, signal = (int(part) for part in match.groups())
            exit_code = code or (128 + signal if signal else 0)
            if state in _TERMINAL_STATES and state != "COMPLETED" and not exit_code:
                exit_code = 1
        return state, exit_code
