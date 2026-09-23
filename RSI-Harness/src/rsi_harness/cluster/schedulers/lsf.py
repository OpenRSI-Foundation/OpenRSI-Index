"""Synchronous Spectrum LSF job submission."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable

from rsi_harness.cluster.schedulers.base import JobResult, JobSpec
from rsi_harness.errors import InfrastructureError, SetupError

_JOB_ID = re.compile(r"^Job <([0-9]+)> is submitted", re.MULTILINE)
_TERMINAL_STATES = {"DONE", "EXIT"}
_TRANSIENT_STATUS_ERRORS = ("communication time out",)

CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]


def _run(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, check=False, capture_output=True, text=True)


class LSFJobSpec(JobSpec):
    """Validated LSF allocation request."""


class LSFJobResult(JobResult):
    """Terminal LSF job status."""


class LSFScheduler:
    def __init__(
        self,
        *,
        submit_binary: str = "bsub",
        status_binary: str = "bjobs",
        cancel_binary: str = "bkill",
        runner: CommandRunner = _run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._submit_binary = submit_binary
        self._status_binary = status_binary
        self._cancel_binary = cancel_binary
        self._runner = runner
        self._sleeper = sleeper

    def render_submit(self, spec: LSFJobSpec) -> tuple[str, ...]:
        argv = [
            self._submit_binary,
            "-J",
            spec.name,
            "-G",
            spec.group,
            "-q",
            spec.queue,
            "-n",
            str(spec.cpu_slots),
            "-M",
            f"{spec.memory_mb}M",
            "-W",
            spec.walltime,
        ]
        resource_terms = []
        select_conditions = []
        if spec.local_tmp_mb:
            select_conditions.append(f"tmp>={spec.local_tmp_mb}")
        select_conditions.extend(
            f"hname!='{host}'" for host in spec.excluded_hosts
        )
        if select_conditions:
            resource_terms.append(f"select[{' && '.join(select_conditions)}]")
        if spec.one_host:
            resource_terms.append("span[hosts=1]")
        else:
            assert spec.slots_per_host is not None
            resource_terms.append(f"span[ptile={spec.slots_per_host}]")
            if spec.memory_per_host:
                resource_terms.append(f"rusage[mem={spec.memory_mb}]")
        if resource_terms:
            argv.extend(("-R", " ".join(resource_terms)))
        if spec.gpu_count:
            argv.extend(
                (
                    "-gpu",
                    f"num={spec.gpu_count}:mode=exclusive_process",
                )
            )
        if spec.exclusive:
            argv.append("-x")
        argv.extend(
            (
                "-o",
                str(spec.stdout_path),
                "-e",
                str(spec.stderr_path),
                str(spec.script_path),
            )
        )
        return tuple(argv)

    def submit(self, spec: LSFJobSpec) -> str:
        if not spec.script_path.is_file():
            raise SetupError(f"LSF payload does not exist: {spec.script_path}")
        completed = self._runner(self.render_submit(spec))
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise InfrastructureError(
                f"LSF submission failed with exit {completed.returncode}: {detail}"
            )
        match = _JOB_ID.search(completed.stdout)
        if match is None:
            raise InfrastructureError(
                "LSF submission succeeded without a parseable exact job ID"
            )
        return match.group(1)

    def require_name_available(self, name: str) -> None:
        """Reject an exact active-name collision on the shared LSF account."""
        completed = self._runner(
            (self._status_binary, "-noheader", "-o", "job_name stat")
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise InfrastructureError(f"cannot inspect active LSF jobs: {detail}")
        active_names = {
            fields[0]
            for line in completed.stdout.splitlines()
            if (fields := line.split())
        }
        if name in active_names:
            raise InfrastructureError(f"LSF job name is already active: {name}")

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float = 5.0,
        on_state: Callable[[str], None] | None = None,
    ) -> LSFJobResult:
        self._require_job_id(job_id)
        previous_state: str | None = None
        try:
            while True:
                completed = self._runner(
                    (
                        self._status_binary,
                        "-a",
                        "-noheader",
                        "-o",
                        "stat exit_code",
                        job_id,
                    )
                )
                if completed.returncode != 0:
                    detail = completed.stderr.strip() or completed.stdout.strip()
                    if any(
                        marker in detail.lower()
                        for marker in _TRANSIENT_STATUS_ERRORS
                    ):
                        self._sleeper(poll_seconds)
                        continue
                    raise InfrastructureError(
                        f"cannot query LSF job {job_id}: {detail}"
                    )
                state, exit_code = self._parse_status(job_id, completed.stdout)
                if state != previous_state and on_state is not None:
                    on_state(state)
                previous_state = state
                if state in _TERMINAL_STATES:
                    return LSFJobResult(
                        job_id=job_id,
                        state=state,
                        exit_code=exit_code,
                    )
                self._sleeper(poll_seconds)
        except KeyboardInterrupt:
            self.cancel(job_id)
            raise

    def cancel(self, job_id: str) -> None:
        self._require_job_id(job_id)
        self._runner((self._cancel_binary, job_id))

    @staticmethod
    def _require_job_id(job_id: str) -> None:
        if not job_id.isdecimal():
            raise SetupError(f"invalid exact LSF job ID: {job_id!r}")

    @staticmethod
    def _parse_status(job_id: str, output: str) -> tuple[str, int]:
        fields = output.strip().split()
        if not fields:
            raise InfrastructureError(f"LSF job {job_id} returned an empty status")
        state = fields[0].upper()
        raw_exit_code = fields[1] if len(fields) > 1 else "-"
        if raw_exit_code == "-":
            exit_code = 0 if state == "DONE" else 1
        else:
            try:
                exit_code = int(raw_exit_code)
            except ValueError as error:
                raise InfrastructureError(
                    f"LSF job {job_id} returned invalid exit code {raw_exit_code!r}"
                ) from error
        return state, exit_code
