"""Fujitsu Technical Computing Suite (PJM) submission with exact-job status
and cancellation.

``pjsub -z jid`` prints only the job ID. ``pjstat -v --choose
jid,st,ec,pc,sn,sdt,elp <id>`` shows an unfinished job, ``pjstat -H`` an
ended one; the adapters read the Slurm states the Slurm adapter already uses:

- ``ST`` ACC, QUE, RNA, RNP and HLD are ``PENDING``; RUN and RNE
  ``RUNNING``; RNO ``COMPLETING``; SPP, SPD and RSM ``SUSPENDED``.
- EXT (ended) is decided by the PJM code ``PC``: 0 with exit code ``EC`` 0 is
  ``COMPLETED``; 11 (over its elapse limit) ``TIMEOUT``; 12 and 23 (over its
  memory) ``OUT_OF_MEMORY``; 1 (pjdel) ``CANCELLED``; anything else, and a
  nonzero exit, ``FAILED``. CCL (cancelled) is ``CANCELLED``; RJT (refused)
  ``FAILED``. A job killed by signal ``SN`` reports exit ``128 + SN``.
- ERR (held for an error) stays queued until someone deletes it: the
  scheduler deletes it (pjdel) and reports ``FAILED``; the history then
  shows it cancelled.
- An ended job ended at its start (``START_DATE``, which has no year: the
  latest one not in the future) plus its elapsed time (``ELAPSE_TIM``,
  without the time it was suspended), both in the controller's local time.
- A job neither listing knows is ``UNKNOWN``.
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from rsi_harness.cluster.schedulers.batch import (
    BatchJobResult,
    BatchJobSpec,
    BatchScheduler,
    TransientSchedulerError,
    command_detail,
)
from rsi_harness.errors import InfrastructureError, SetupError

_JOB_ID = re.compile(r"[0-9]{1,10}")
_ACTIVE = {
    "ACC": "PENDING",
    "QUE": "PENDING",
    "RNA": "PENDING",
    "RNP": "PENDING",
    "HLD": "PENDING",
    "RUN": "RUNNING",
    "RNE": "RUNNING",
    "RNO": "COMPLETING",
    "SPP": "SUSPENDED",
    "SPD": "SUSPENDED",
    "RSM": "SUSPENDED",
}
_PC_CANCELLED = 1
_PC_TIMEOUT = 11
_PC_MEMORY = (12, 23)
# How far back the history is searched for an ended job.
HISTORY_DAYS = 31
_COLUMNS = ("JOB_ID", "ST", "EC", "PC", "SN")
# When the job started and how long it ran, where the listing has them.
_TIMES = ("START_DATE", "ELAPSE_TIM")
_CHOOSE = "jid,st,ec,pc,sn,sdt,elp"
# "MM/DD hh:mm:ss" (START_DATE, ACCEPT) is two words of a row: one value.
_DAY = re.compile(r"\(?(?:\d{4}/)?\d{2}/\d{2}")
_CLOCK = re.compile(r"\d{2}:\d{2}(?::\d{2})?\)?[<@#]?")
_STARTED = re.compile(r"(?:(\d{4})/)?(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2})[<@#]?")
_ELAPSED = re.compile(r"(\d+):([0-5]\d)(?::([0-5]\d))?")
# A running job's elapsed time and its limit.
_LIMITS = ("ELAPSE_TIM", "ELAPSE_LIM")


class PJMJobSpec(BatchJobSpec):
    """One PJM job on one node."""

    gpu_resource: str | None = "gpu"
    cpu_resource: str | None = None
    memory_resource: str | None = None
    resources: tuple[str, ...] = ()


class PJMScheduler(BatchScheduler):
    label = "PJM"
    job_id_variable = "PJM_JOBID"
    name_limit = 63
    transient_errors = (
        # PJM 0090/0190/0290 and 0091/0191/0291: the job manager did not
        # answer or failed internally; the reference says to run it again.
        "daemon is not present",
        "internal error",
        "connection refused",
        "connection timed out",
        "communication",
        "temporarily",
        "try again",
        "busy",
    )
    # The job manager refused the request: nothing was queued. An internal
    # error may come after it accepted the job.
    unsent_errors = (
        "daemon is not present",
        "connection refused",
        "temporarily",
        "try again",
        "busy",
    )

    def __init__(
        self,
        *,
        submit_binary: str = "pjsub",
        status_binary: str = "pjstat",
        cancel_binary: str = "pjdel",
        gpu_resource: str | None = "gpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._submit_binary = submit_binary
        self._status_binary = status_binary
        self._cancel_binary = cancel_binary
        self._gpu_resource = gpu_resource

    def _submit_argv(self, spec: BatchJobSpec, script: Path) -> tuple[str, ...]:
        assert isinstance(spec, PJMJobSpec)
        hours, minutes = (int(part) for part in spec.walltime.split(":"))
        resources = [f"rscgrp={spec.queue}", f"elapse={hours:02d}:{minutes:02d}:00"]
        if spec.exclusive:
            resources.append("node=1")
        elif spec.gpu_count and spec.gpu_resource:
            resources.append(f"{spec.gpu_resource}={spec.gpu_count}")
        elif not spec.gpu_count:
            if spec.cpu_resource:
                resources.append(f"{spec.cpu_resource}={spec.cpu_slots}")
            if spec.memory_resource:
                resources.append(f"{spec.memory_resource}={spec.memory_mb}Mi")
        resources.extend(spec.resources)
        argv = [
            self._submit_binary,
            "-z",
            "jid",
            "-N",
            self.job_name(spec),
            "-L",
            ",".join(resources),
            # PJM names the logs itself: %j is the job ID.
            "-o",
            str(spec.stdout_path),
            "-e",
            str(spec.stderr_path),
            # A rerun would run the payload over the same logs.
            "--norestart",
        ]
        if spec.group:
            argv.extend(("-g", spec.group))
        argv.append(str(script))
        return tuple(argv)

    def _job_id(self, stdout: str) -> str | None:
        job_id = stdout.strip()
        return job_id if _JOB_ID.fullmatch(job_id) else None

    def _active_ids(self, name: str) -> tuple[str, ...]:
        """The user's unfinished jobs named ``name``."""
        return self._named_ids(name)

    def _submission_ids(self, name: str) -> tuple[str, ...]:
        # Active first: a job that finishes between these queries is then in
        # history. Deduplicate jobs visible in both listings during transition.
        return tuple(
            dict.fromkeys(
                (*self._named_ids(name), *self._named_ids(name, history=True))
            )
        )

    def _named_ids(self, name: str, *, history: bool = False) -> tuple[str, ...]:
        completed = self._checked(
            (
                self._status_binary,
                *(("-H", f"day={HISTORY_DAYS}") if history else ()),
                "-v",
                "--choose",
                "jid,jnam,st",
                "--filter",
                f"jnam={_escape(name)}",
            ),
            "list PJM job history" if history else "list active PJM jobs",
        )
        # The filter matches the whole name; the listing shows only its start.
        ids = tuple(
            row["JOB_ID"]
            for row in _rows(completed.stdout, ("JOB_ID", "JOB_NAME", "ST"))
        )
        if any(_JOB_ID.fullmatch(item) is None for item in ids):
            raise InfrastructureError("pjstat returned an invalid job ID")
        return ids

    def _cancel(self, job_id: str) -> None:
        completed = self._runner((self._cancel_binary, job_id))
        if not completed.returncode:
            return
        detail = command_detail(completed)
        if self._transient(detail):
            raise TransientSchedulerError(f"cannot cancel PJM job {job_id}: {detail}")
        # A job that has already ended cannot be deleted; that is no error.
        result = self._status(job_id)
        if result is None or result.state not in _ACTIVE.values():
            return
        raise InfrastructureError(f"cannot cancel PJM job {job_id}: {detail}")

    def _status(self, job_id: str) -> BatchJobResult | None:
        for history in (False, True):
            argv = [self._status_binary]
            if history:
                argv += ["-H", f"day={HISTORY_DAYS}"]
            argv += ["-v", "--choose", _CHOOSE, job_id]
            completed = self._checked(tuple(argv), f"query PJM job {job_id}")
            found = [
                row
                for row in _rows(completed.stdout, _COLUMNS, _TIMES)
                if row["JOB_ID"] == job_id
            ]
            if len(found) > 1:
                raise InfrastructureError(f"ambiguous PJM status for job {job_id}")
            if found:
                if not history and found[0]["ST"].upper() == "ERR":
                    self._delete_held(job_id)
                return job_result(job_id, found[0], now=self._wall_clock())
        return None

    def _walltime_left(self, job_id: str, environ: Mapping[str, str]) -> float | None:
        completed = self._checked(
            (self._status_binary, "-v", "--choose", "jid,st,elp,elpl", job_id),
            f"query PJM job {job_id}",
        )
        rows = [
            row
            for row in _rows(completed.stdout, ("JOB_ID", "ST"), _LIMITS)
            if row["JOB_ID"] == job_id
        ]
        if len(rows) != 1:
            return None
        # An adaptive limit ("min-max", "min-") is sure of its minimum only:
        # past it, PJM may end the job for another.
        minimum, ranged, _ = rows[0].get("ELAPSE_LIM", "-").partition("-")
        mode = environ.get("PJM_ELAPSED_TIME_MODE", "fixed").strip().lower()
        if mode != "fixed" and not ranged:
            raise SetupError(
                f"PJM job {job_id} has an adaptive elapse limit "
                f"(PJM_ELAPSED_TIME_MODE={mode}) whose minimum pjstat does not "
                "show: request a fixed elapse limit"
            )
        limit = _seconds(minimum)
        used = _seconds(rows[0].get("ELAPSE_TIM", "-"))
        if limit is None or used is None:
            return None
        return float(limit - used)

    def _job_gpus(self, job_id: str, environ: Mapping[str, str]) -> int | None:
        # The job's custom resources ("gpu/n=4,jobenv=singularity"; "/n":
        # per node, without it the job's total); pjstat does not list them.
        for item in environ.get("PJM_CUSTOM_RESOURCES", "").split(","):
            name, _, value = item.partition("=")
            if self._gpu_resource and name == f"{self._gpu_resource}/n":
                return int(value) if value.isdigit() else None
        return None

    def _delete_held(self, job_id: str) -> None:
        """Delete a job held for an error, which would otherwise stay queued
        (and keep its name taken) until someone deletes or releases it."""
        try:
            completed = self._checked(
                (self._status_binary, "-v", "--choose", "jid,ermsg", job_id),
                f"query PJM job {job_id}",
            )
            rows = _rows(completed.stdout, ("JOB_ID", "REASON"))
            reason = next(
                (row["REASON"] for row in rows if row["JOB_ID"] == job_id), "-"
            )
        except InfrastructureError:
            reason = "-"
        completed = self._runner((self._cancel_binary, job_id))
        if completed.returncode:
            detail = command_detail(completed)
            error = (
                TransientSchedulerError
                if self._transient(detail)
                else InfrastructureError
            )
            raise error(
                f"PJM job {job_id} is held for an error ({reason}) and cannot "
                f"be deleted: {detail}; pjdel it"
            )
        print(
            f"PJM job {job_id} was held for an error ({reason}); deleted it",
            file=sys.stderr,
            flush=True,
        )

    def _require_job_id(self, job_id: str) -> None:
        if _JOB_ID.fullmatch(job_id) is None:
            raise SetupError(f"invalid exact PJM job ID: {job_id!r}")


def _escape(value: str) -> str:
    """A ``--filter`` value matched literally."""
    return re.sub(r"([.*+,\\])", r"\\\1", value)


def _words(line: str) -> list[str]:
    """A row's words, with a date and its time as one value."""
    words: list[str] = []
    for word in line.split():
        if words and _CLOCK.fullmatch(word) and _DAY.fullmatch(words[-1]):
            words[-1] += f" {word}"
        else:
            words.append(word)
    return words


def _rows(
    output: str, columns: tuple[str, ...], optional: tuple[str, ...] = ()
) -> list[dict[str, str]]:
    """The rows of a pjstat table with ``columns`` (its header names them),
    and those of ``optional`` it has.

    Values are single words ("-" when unset) or dates, and a last REASON
    column takes the rest of the row; a table without that header (no such
    job) has no rows."""
    lines = [_words(line) for line in output.splitlines() if line.strip()]
    for index, words in enumerate(lines):
        if words[: len(columns)] == list(columns) or (
            set(columns) <= set(words) and words[0] == columns[0]
        ):
            header = words
            break
    else:
        return []
    kept = columns + tuple(column for column in optional if column in header)
    rows = []
    for words in lines[index + 1 :]:
        if header[-1] == "REASON" and len(words) > len(header):
            words = [*words[: len(header) - 1], " ".join(words[len(header) - 1 :])]
        if len(words) != len(header):
            continue
        row = dict(zip(header, words, strict=True))
        rows.append({column: row[column] for column in kept})
    return rows


def _number(job_id: str, name: str, value: str) -> int | None:
    if value in ("", "-"):
        return None
    try:
        return int(value)
    except ValueError:
        raise InfrastructureError(f"invalid PJM {name} for job {job_id}") from None


def _seconds(value: str) -> int | None:
    """``hhhh:mm:ss`` (``hhhh:mm`` when the column overflows) in seconds."""
    match = _ELAPSED.fullmatch(value.strip())
    if match is None:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds or 0)


def ended_at(row: dict[str, str], now: float) -> float | None:
    """When an ended job ended: its start plus its elapsed time, or None for
    a job that never started (or a listing without them)."""
    started = _STARTED.fullmatch(row.get("START_DATE", "-"))
    elapsed = _ELAPSED.fullmatch(row.get("ELAPSE_TIM", "-"))
    if started is None or elapsed is None:
        return None
    year, month, day, hour, minute, second = started.groups()
    clock = datetime.fromtimestamp(now)
    start: float | None = None
    # No year: the latest one that puts the start in the past (Feb 29 too).
    for candidate in (int(year),) if year else range(clock.year, clock.year - 9, -1):
        try:
            moment = datetime(
                candidate, int(month), int(day), int(hour), int(minute), int(second)
            )
        except ValueError:
            continue
        if year or moment <= clock:
            start = moment.timestamp()
            break
    if start is None:
        return None
    hours, minutes, seconds = elapsed.groups()
    # Without its seconds (a long run), the end of that minute: an attempt
    # is never charged short.
    used = int(hours) * 3600 + int(minutes) * 60 + int(seconds or 59)
    return min(start + used, now)


def job_result(
    job_id: str, row: dict[str, str], *, now: float | None = None
) -> BatchJobResult:
    letters = row["ST"].upper()
    if letters in _ACTIVE:
        return BatchJobResult(job_id=job_id, state=_ACTIVE[letters], exit_code=0)
    exit_status = _number(job_id, "EC", row["EC"])
    code = _number(job_id, "PC", row["PC"])
    signal = _number(job_id, "SN", row["SN"])
    if letters == "CCL" or code == _PC_CANCELLED:
        state = "CANCELLED"
    elif letters in ("RJT", "ERR"):
        state = "FAILED"
    elif letters != "EXT":
        raise InfrastructureError(f"unknown PJM state for {job_id}: {letters!r}")
    elif code == _PC_TIMEOUT:
        state = "TIMEOUT"
    elif code in _PC_MEMORY:
        state = "OUT_OF_MEMORY"
    elif code == 0 and exit_status == 0 and not signal:
        state = "COMPLETED"
    else:
        state = "FAILED"
    exit_code = 128 + signal if signal else (exit_status or 0)
    if state != "COMPLETED" and not exit_code:
        exit_code = 1
    return BatchJobResult(
        job_id=job_id,
        state=state,
        exit_code=exit_code,
        ended_at=ended_at(row, time.time() if now is None else now),
    )
