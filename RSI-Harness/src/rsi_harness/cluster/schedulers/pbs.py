"""PBS Pro submission with exact-job status and cancellation.

``qstat -x -f -F json <id>`` gives a job's state, also after it finished
(the server's job history must be enabled). The adapters read the Slurm
states the Slurm adapter already uses:

- ``job_state`` Q, H, W and T are ``PENDING``; R and B ``RUNNING``; E
  (exiting) ``COMPLETING``; S and U ``SUSPENDED``.
- F (finished; X for an array subjob) with ``Exit_status`` 0 is
  ``COMPLETED``. Deleted with qdel (``substate`` 91), before or while it
  ran, it is ``CANCELLED``. Otherwise ``Exit_status`` decides: -29 (killed
  at its walltime) ``TIMEOUT``; -27 or -26 (over its mem or vmem)
  ``OUT_OF_MEMORY``; any other negative value (PBS could not run it) and
  any nonzero exit ``FAILED``. A value of 256 or more is a job shell killed
  by signal ``Exit_status - 256``, reported as exit ``128 + signal``.
- A job the server no longer knows (history purged) is ``UNKNOWN``.

qsub takes output paths literally (no job ID placeholder): PBS's own job
output goes to the job's directory, and the wrapper writes the payload's
output to the adapter's ``pbs.<job ID>`` logs itself.
"""

from __future__ import annotations

import getpass
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from rsi_harness.cluster.schedulers.batch import (
    ACTIVE_STATES,
    BatchJobResult,
    BatchJobSpec,
    BatchScheduler,
    TransientSchedulerError,
    command_detail,
    wrapper,
)
from rsi_harness.errors import InfrastructureError, SetupError

# qsub prints the full job ID: "<sequence number>.<server>" ("96979.jobm").
_JOB_ID = re.compile(r"[0-9]+(?:\[[0-9]*\])?(?:\.[A-Za-z0-9][A-Za-z0-9_.-]*)?")
_ACTIVE = {
    "B": "RUNNING",
    "E": "COMPLETING",
    "H": "PENDING",
    "Q": "PENDING",
    "R": "RUNNING",
    "S": "SUSPENDED",
    "T": "PENDING",
    "U": "SUSPENDED",
    "W": "PENDING",
}
_FINISHED = frozenset({"F", "X"})
_SUBSTATE_DELETED = 91
_EXIT_WALLTIME = -29
_EXIT_MEMORY = (-27, -26)
_STATUS_ATTRIBUTES = (
    "job_state",
    "substate",
    "Exit_status",
    "obittime",
    "stime",
    "resources_used.walltime",
    "Resource_List.walltime",
    "Resource_List.ngpus",
    "Resource_List.nodect",
)
# An attribute line of plain `qstat -f`; continuations start with a tab.
_TEXT_ATTRIBUTE = re.compile(r" {4}([A-Za-z_][A-Za-z0-9_.]*) = (.*)")
_UNKNOWN_JOB = "unknown job id"
_FINISHED_JOB = "job has finished"


class PBSJobSpec(BatchJobSpec):
    """One PBS job on one host."""

    project: str | None = None
    chunk_resources: tuple[str, ...] = ("ncpus", "mem", "ngpus")
    chunk_extra: tuple[str, ...] = ()


class PBSScheduler(BatchScheduler):
    label = "PBS"
    job_id_variable = "PBS_JOBID"
    # PBS Pro allows longer names; 63 keeps them readable in qstat.
    name_limit = 63
    transient_errors = (
        "cannot connect to server",
        "communication failure",
        "connection refused",
        "connection timed out",
        "pbs_iff",
        "server is busy",
        "try again",
    )
    # The server refused the request: nothing was queued.
    unsent_errors = (
        "cannot connect to server",
        "connection refused",
        "server is busy",
        "try again",
    )

    def __init__(
        self,
        *,
        submit_binary: str = "qsub",
        status_binary: str = "qstat",
        cancel_binary: str = "qdel",
        select_binary: str = "qselect",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._walltime_ends: dict[str, float | None] = {}
        self._submit_binary = submit_binary
        self._status_binary = status_binary
        self._cancel_binary = cancel_binary
        self._select_binary = select_binary

    def _job_script(self, spec: BatchJobSpec, environment_file: Path) -> str:
        def log(path: Path) -> str:
            # The payload's logs, named by the job's own ID as Slurm and LSF
            # name theirs.
            return str(path).replace('"', '\\"').replace("%j", "${PBS_JOBID}")

        return wrapper(
            spec.script_path,
            environment_file,
            signal_after_seconds=self._signal_after(spec),
            stdout=log(spec.stdout_path),
            stderr=log(spec.stderr_path),
        )

    def _submit_argv(self, spec: BatchJobSpec, script: Path) -> tuple[str, ...]:
        assert isinstance(spec, PBSJobSpec)
        # The shared planner uses HH:MM; PBS reads two fields as MM:SS.
        hours, minutes = (int(part) for part in spec.walltime.split(":"))
        planned = {
            "ncpus": spec.cpu_slots,
            "mem": f"{spec.memory_mb}mb",
            "ngpus": spec.gpu_count,
        }
        chunk = [
            f"{name}={planned[name]}"
            for name in ("ncpus", "mem", "ngpus")
            if name in spec.chunk_resources and (name != "ngpus" or spec.gpu_count)
        ]
        argv = [
            self._submit_binary,
            "-q",
            spec.queue,
            "-l",
            ":".join(("select=1", *chunk, *spec.chunk_extra)),
            "-l",
            f"walltime={hours:02d}:{minutes:02d}:00",
            "-N",
            self.job_name(spec),
        ]
        if spec.group:
            argv.extend(("-A", spec.group))
        if spec.project:
            argv.extend(("-P", spec.project))
        if spec.exclusive:
            argv.extend(("-l", "place=excl"))
        argv.extend(
            (
                # PBS's own output of the job (its kill notices among them).
                "-j",
                "oe",
                "-o",
                str(script.with_name("pbs.log")),
                # A rerun would run the payload over the same logs (and find
                # its environment file consumed).
                "-r",
                "n",
                str(script),
            )
        )
        return tuple(argv)

    def _job_id(self, stdout: str) -> str | None:
        job_id = stdout.strip()
        return job_id if _JOB_ID.fullmatch(job_id) else None

    def _active_ids(self, name: str) -> tuple[str, ...]:
        """The user's unfinished jobs named ``name`` (qselect)."""
        return self._selected_ids(name)

    def _submission_ids(self, name: str) -> tuple[str, ...]:
        return self._selected_ids(name, history=True)

    def _selected_ids(self, name: str, *, history: bool = False) -> tuple[str, ...]:
        completed = self._checked(
            (
                self._select_binary,
                *(("-x",) if history else ()),
                "-u",
                getpass.getuser(),
                "-N",
                name,
            ),
            "list PBS jobs" if history else "list active PBS jobs",
        )
        ids = tuple(completed.stdout.split())
        if any(_JOB_ID.fullmatch(item) is None for item in ids):
            raise InfrastructureError("qselect returned an invalid job ID")
        return ids

    def _cancel(self, job_id: str) -> None:
        completed = self._runner((self._cancel_binary, job_id))
        detail = command_detail(completed)
        lowered = detail.lower()
        if (
            not completed.returncode
            or _UNKNOWN_JOB in lowered
            or _FINISHED_JOB in lowered
        ):
            return
        if self._transient(detail):
            raise TransientSchedulerError(f"cannot cancel PBS job {job_id}: {detail}")
        raise InfrastructureError(f"cannot cancel PBS job {job_id}: {detail}")

    def _status(self, job_id: str) -> BatchJobResult | None:
        attributes = self._attributes(job_id)
        if attributes is None:
            return None
        result = job_result(job_id, attributes)
        if result.state not in ACTIVE_STATES:
            self._walltime_ends[job_id] = walltime_end(attributes)
        return result

    def _after_outage(
        self, result: BatchJobResult, last_running: float | None
    ) -> BatchJobResult:
        estimate = self._walltime_ends.get(result.job_id)
        if estimate is None or result.ended_at is None:
            return result
        if last_running is not None:
            estimate = max(estimate, last_running)
        return result.model_copy(update={"ended_at": min(estimate, result.ended_at)})

    def _attributes(self, job_id: str) -> Mapping[str, Any] | None:
        """The job's attributes, or None for a job the server does not know.
        Some servers escape unusual variable values badly in JSON; the plain
        listing has the same attributes."""
        completed = self._query(job_id, "-F", "json")
        if completed is None:
            return None
        try:
            return json_attributes(job_id, completed)
        except ValueError:
            text = self._query(job_id)
            return None if text is None else text_attributes(job_id, text)

    def _walltime_left(self, job_id: str, environ: Mapping[str, str]) -> float | None:
        attributes = self._attributes(job_id)
        if attributes is None:
            return None
        started = pbs_time(attributes.get("stime"))
        seconds = _clock_seconds(_requested(attributes, "walltime"))
        if started is None or seconds is None:
            return None
        return started + seconds - self._wall_clock()

    def _job_gpus(self, job_id: str, environ: Mapping[str, str]) -> int | None:
        # ngpus sums all the job's chunks: this node's share only on one node.
        attributes = self._attributes(job_id)
        if attributes is None or str(_requested(attributes, "nodect")) != "1":
            return None
        gpus = _requested(attributes, "ngpus")
        return int(gpus) if str(gpus).isdigit() else None

    def _query(self, job_id: str, *options: str) -> str | None:
        completed = self._runner((self._status_binary, "-x", "-f", *options, job_id))
        if not completed.returncode:
            return completed.stdout
        detail = command_detail(completed)
        if _UNKNOWN_JOB in detail.lower():
            return None
        if self._transient(detail):
            raise TransientSchedulerError(f"cannot query PBS job {job_id}: {detail}")
        raise InfrastructureError(f"cannot query PBS job {job_id}: {detail}")

    def _require_job_id(self, job_id: str) -> None:
        if _JOB_ID.fullmatch(job_id) is None:
            raise SetupError(f"invalid exact PBS job ID: {job_id!r}")


def _same_job(job_id: str, listed: str) -> bool:
    """``96979`` names ``96979.jobm``; a full ID must match exactly."""
    return listed == job_id or ("." not in job_id and listed.startswith(f"{job_id}."))


def pbs_time(value: Any) -> float | None:
    """A qstat time (``Tue Nov  3 19:23:44 2026``, local time) as a
    timestamp, or None."""
    if value is None:
        return None
    try:
        parsed = datetime.strptime(" ".join(str(value).split()), "%a %b %d %H:%M:%S %Y")
    except ValueError:
        return None
    return parsed.timestamp()


def json_attributes(job_id: str, output: str) -> Mapping[str, Any]:
    """The job's attributes from ``qstat -f -F json``; ValueError when the
    output is not JSON (values may hold control characters)."""
    listing = json.loads(output, strict=False)
    if not isinstance(listing, dict):
        raise ValueError("qstat output is not a JSON object")
    jobs = listing.get("Jobs")
    if not isinstance(jobs, dict):
        raise InfrastructureError(f"PBS returned no status for job {job_id}")
    found = [value for key, value in jobs.items() if _same_job(job_id, key)]
    if len(found) != 1 or not isinstance(found[0], dict):
        raise InfrastructureError(f"invalid PBS status for exact job {job_id}")
    return found[0]


def text_attributes(job_id: str, output: str) -> Mapping[str, Any]:
    """The status attributes from plain ``qstat -f`` for the exact job."""
    lines = output.splitlines()
    header = re.fullmatch(r"Job Id: (\S+)", lines[0].strip()) if lines else None
    if header is None or not _same_job(job_id, header.group(1)):
        raise InfrastructureError(f"invalid PBS status for exact job {job_id}")
    attributes: dict[str, str] = {}
    current: str | None = None
    for line in lines[1:]:
        if line.startswith("Job Id:"):
            raise InfrastructureError(f"invalid PBS status for exact job {job_id}")
        if line.startswith("\t") and current is not None:
            attributes[current] += line.strip()
            continue
        current = None
        match = _TEXT_ATTRIBUTE.fullmatch(line)
        if match is None or match.group(1) not in _STATUS_ATTRIBUTES:
            continue
        if match.group(1) in attributes:
            raise InfrastructureError(f"ambiguous PBS status for job {job_id}")
        current = match.group(1)
        attributes[current] = match.group(2).strip()
    return attributes


def _integer(job_id: str, name: str, value: Any) -> int | None:
    if value is None:
        return None
    try:
        # qstat's JSON writes negative numbers as strings ("-29").
        return int(str(value).strip())
    except ValueError:
        raise InfrastructureError(f"invalid PBS {name} for job {job_id}") from None


def job_result(job_id: str, attributes: Mapping[str, Any]) -> BatchJobResult:
    letter = str(attributes.get("job_state", "")).strip().upper()
    if letter in _ACTIVE:
        return BatchJobResult(job_id=job_id, state=_ACTIVE[letter], exit_code=0)
    if letter not in _FINISHED:
        # M: moved to another server, which this one cannot follow.
        raise InfrastructureError(f"unknown PBS state for {job_id}: {letter!r}")
    status = _integer(job_id, "Exit_status", attributes.get("Exit_status"))
    substate = _integer(job_id, "substate", attributes.get("substate"))
    if substate == _SUBSTATE_DELETED:
        state = "CANCELLED"
    elif status is None:
        state = "FAILED"
    elif status == 0:
        state = "COMPLETED"
    elif status == _EXIT_WALLTIME:
        state = "TIMEOUT"
    elif status in _EXIT_MEMORY:
        state = "OUT_OF_MEMORY"
    else:
        state = "FAILED"
    if status is None or status < 0:
        exit_code = 0 if state == "COMPLETED" else 1
    elif status >= 256:
        exit_code = 128 + status - 256
    else:
        exit_code = status
    if state != "COMPLETED" and not exit_code:
        exit_code = 1
    return BatchJobResult(
        job_id=job_id,
        state=state,
        exit_code=exit_code,
        ended_at=pbs_time(attributes.get("obittime")),
    )


def _requested(attributes: Mapping[str, Any], name: str) -> Any:
    """A ``Resource_List`` value, from either listing."""
    limits = attributes.get("Resource_List")
    if isinstance(limits, Mapping):
        return limits.get(name)
    return attributes.get(f"Resource_List.{name}")


def _clock_seconds(value: Any) -> int | None:
    """A PBS duration (``HH:MM:SS``) in seconds, or None."""
    match = re.fullmatch(r"(\d+):([0-5]\d):([0-5]\d)", str(value or "").strip())
    if match is None:
        return None
    hours, minutes, seconds = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def walltime_end(attributes: Mapping[str, Any]) -> float | None:
    """Start plus walltime used: when a job ended whose end notice (obittime)
    a server outage delayed. A suspended job's walltime leaves out the time
    it was suspended, so this alone may be too early."""
    ended = pbs_time(attributes.get("obittime"))
    started = pbs_time(attributes.get("stime"))
    used = attributes.get("resources_used")
    walltime = used.get("walltime") if isinstance(used, Mapping) else None
    if walltime is None:
        walltime = attributes.get("resources_used.walltime")
    seconds = _clock_seconds(walltime)
    if ended is None or started is None or seconds is None:
        return ended
    return min(ended, started + seconds)
