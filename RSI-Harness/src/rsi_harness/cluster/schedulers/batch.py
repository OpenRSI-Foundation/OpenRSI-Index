"""What the PBS and PJM schedulers share: a job wrapper, the submit
environment through a private file, bounded client calls and status polls.

A job gets the submit environment as Slurm's ``--export=ALL`` gives it, but
never through the scheduler's own environment option (``qsub -V``,
``pjsub -X``): that copies every variable, credentials included, into the
server's job record. The variables go to a private file instead (mode 0600,
in the job's own directory beside its logs), which the job's wrapper loads
and deletes before it runs the payload. The controller deletes a file the
job never loaded once it sees the job end or cancels it; ``resume``, once
the run's jobs have all ended, deletes those a controller that died first
left behind. Variables the scheduler sets for the job itself are left to it.

The wrapper also sends the payload the advance SIGTERM that Slurm's
``--signal`` gives (``signal_seconds`` before the walltime), so segmented
runs drain the Judge before the scheduler ends the job, and passes on a
SIGTERM, SIGINT or SIGHUP the scheduler sends the job (a cancel).
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
import shlex
import stat
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from rsi_harness.cluster.schedulers.slurm import SlurmJobResult, SlurmJobSpec
from rsi_harness.errors import InfrastructureError, SetupError

CommandRunner = Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]]

# No client call may hang the controller: past this the answer counts as a
# scheduler that did not answer.
COMMAND_TIMEOUT_SECONDS = 120.0
# Attempts of a client call that met a scheduler that did not answer.
COMMAND_ATTEMPTS = 3
# How long status polls tolerate a scheduler that does not answer before the
# wait fails (the job keeps running; resume can follow it later).
STATUS_OUTAGE_SECONDS = 2 * 3600.0
MAX_BACKOFF_SECONDS = 300.0
# How long a submission waits for a scheduler that refuses connections.
SUBMIT_OUTAGE_SECONDS = 1800.0
# While a job's state stays the same this long, poll at most once a minute.
STABLE_AFTER_SECONDS = 300.0
STABLE_POLL_SECONDS = 60.0
# Polls of a job the scheduler does not know before the wait gives up.
MISSING_POLLS = 12

ACTIVE_STATES = frozenset(
    {"PENDING", "RUNNING", "COMPLETING", "SUSPENDED"},
)

# Variables a scheduler sets for every job, or that describe only the
# submitting process or shell (some are read-only in bash).
_JOB_LOCAL_PREFIXES = (
    "PBS_",
    "PJM_",
    "SLURM_",
    "LSB_",
    "LSF_",
    "OMPI_",
    "PMI_",
    "PMIX_",
    "BASH",
)
_JOB_LOCAL_NAMES = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "NCPUS",
        "OMP_NUM_THREADS",
        "TMPDIR",
        "ENVIRONMENT",
        "HOSTNAME",
        "HOST",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "SHELLOPTS",
        "EUID",
        "PPID",
        "UID",
        "_",
        "PROMPT_COMMAND",
        "PS1",
        "TERM",
        "TMOUT",
        "DISPLAY",
        "XDG_RUNTIME_DIR",
        "XDG_SESSION_ID",
        "SSH_AUTH_SOCK",
        "SSH_CLIENT",
        "SSH_CONNECTION",
        "SSH_TTY",
    }
)
_VARIABLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def run_command(argv: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    """Run a scheduler client command, bounded by COMMAND_TIMEOUT_SECONDS."""
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            argv, 124, "", f"{argv[0]}: timed out after {COMMAND_TIMEOUT_SECONDS:.0f}s"
        )
    except OSError as error:
        # A missing client is a refusal, not a crash.
        return subprocess.CompletedProcess(
            argv, 127, "", f"{argv[0]}: {error.strerror or error}"
        )


def command_detail(completed: subprocess.CompletedProcess[str]) -> str:
    return completed.stderr.strip() or completed.stdout.strip()


def job_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The submit environment a job receives: all but job-local variables."""
    return {
        name: value
        for name, value in environ.items()
        if _VARIABLE_NAME.fullmatch(name)
        and name not in _JOB_LOCAL_NAMES
        and not name.startswith(_JOB_LOCAL_PREFIXES)
        and "\0" not in value
    }


def bounded_name(name: str, limit: int) -> str:
    """``name`` if it fits ``limit`` bytes, else a stable shortened form."""
    if len(name.encode()) <= limit:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:10]
    return f"{name[: limit - len(digest) - 1].rstrip('-_.')}-{digest}"


def write_private(path: Path, data: bytes, mode: int) -> None:
    """Create ``path`` afresh with ``mode``, never through a link."""
    try:
        if path.is_symlink() or path.exists():
            path.unlink()
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            mode,
        )
    except OSError as error:
        raise InfrastructureError(
            f"cannot write job file {path}: {error.strerror}"
        ) from None
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
    os.chmod(path, mode)


def remove_private(path: Path) -> None:
    """Delete ``path`` if it is a regular file of this user, never a link."""
    try:
        status = path.lstat()
        if stat.S_ISREG(status.st_mode) and status.st_uid == os.getuid():
            path.unlink()
    except OSError:
        pass


def remove_environment_files(*roots: Path) -> None:
    """Delete the environment files left in the job directories under
    ``roots``, whose jobs have all ended."""
    for root in roots:
        try:
            entries = list(os.scandir(root))
        except OSError:
            continue
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                remove_private(Path(entry.path) / "environment")


def wrapper(
    script: Path,
    environment_file: Path,
    *,
    signal_after_seconds: int = 0,
    stdout: str | None = None,
    stderr: str | None = None,
) -> str:
    """The job script the scheduler runs. ``stdout``/``stderr``: where the
    payload writes, as shell words (may name the job's ID variable), when
    the scheduler cannot name the job's own logs."""
    redirect = ""
    if stdout is not None and stderr is not None:
        redirect = f'exec >>"{stdout}" 2>>"{stderr}"\n'
    timer = ""
    if signal_after_seconds:
        # The advance signal, as Slurm's --signal=B:TERM sends it.
        # Ended early, the timer takes its sleep with it.
        timer = (
            f"( sleep {signal_after_seconds} & sleeper=$!\n"
            "  trap 'kill \"$sleeper\" 2>/dev/null; exit 0' TERM\n"
            '  wait "$sleeper" && kill -TERM "$payload" ) '
            "</dev/null >/dev/null 2>&1 &\ntimer=$!\n"
        )
    return f"""#!/bin/bash
# rsi-harness job. The submit environment arrives through a private file,
# never the server's job record; it is deleted once loaded.
set -uo pipefail
{redirect}environment_file={shlex.quote(str(environment_file))}
if [[ ! -r "$environment_file" ]]; then
  echo "rsi-harness: job environment $environment_file is missing" >&2
  exit 1
fi
while IFS= read -r -d '' assignment; do
  # A variable the job's own shell made read-only keeps the job's value.
  ( export "$assignment" ) 2>/dev/null && export "$assignment"
done < "$environment_file"
rm -f -- "$environment_file"
/bin/bash {shlex.quote(str(script))} &
payload=$!
trap 'kill -TERM "$payload" 2>/dev/null' TERM INT HUP
timer=
{timer}while :; do
  wait "$payload"
  status=$?
  # A trapped signal interrupts wait; the payload still runs (or is a
  # zombie not yet reaped).
  if (( status > 128 )) && kill -0 "$payload" 2>/dev/null; then
    continue
  fi
  break
done
[[ -n "$timer" ]] && kill "$timer" 2>/dev/null
exit "$status"
"""


class BatchJobSpec(SlurmJobSpec):
    """Share the cluster allocation fields; render one batch job."""


class BatchJobResult(SlurmJobResult):
    """Terminal job status, in the Slurm states the Slurm adapter reads."""


class TransientSchedulerError(InfrastructureError):
    """The scheduler did not answer (this time)."""


class BatchScheduler:
    """Submit, follow and cancel one-host batch jobs.

    Subclasses render the commands and read the scheduler's answers:
    ``_submit_argv``, ``_job_id``, ``_status`` (a result, or None for a job the
    scheduler does not know), ``_active_ids``, ``_submission_ids`` and ``_cancel``.
    """

    label = "batch"
    name_limit = 63
    # Words of the client's error text that mean "did not answer", and of
    # those, the ones that mean the request never reached the scheduler.
    transient_errors: tuple[str, ...] = ()
    unsent_errors: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        runner: CommandRunner = run_command,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
        environ: Mapping[str, str] | None = None,
        token: Callable[[], str] = lambda: secrets.token_hex(3),
    ) -> None:
        self._runner = runner
        self._sleeper = sleeper
        self._clock = clock
        self._wall_clock = wall_clock
        self._environ = environ
        self._token = token
        # Environment files of submitted jobs, removed once the job is known
        # to have ended (a job that never started leaves its file otherwise).
        self._environment_files: dict[str, Path] = {}
        self._directories: dict[str, Path] = {}

    # -- names and files -------------------------------------------------

    def job_name(self, spec: BatchJobSpec) -> str:
        return bounded_name(spec.name, self.name_limit)

    def job_directory(self, spec: BatchJobSpec) -> Path:
        """The job's wrapper and environment file, beside its logs and
        outside the run's frozen control directory."""
        key = str(spec.script_path)
        if key not in self._directories:
            self._directories[key] = (
                spec.stdout_path.parent / f"{self.job_name(spec)}-{self._token()}"
            )
        return self._directories[key]

    def _signal_after(self, spec: BatchJobSpec) -> int:
        if not spec.signal_seconds:
            return 0
        hours, minutes = (int(part) for part in spec.walltime.split(":"))
        return max(1, (hours * 60 + minutes) * 60 - spec.signal_seconds)

    def _job_script(self, spec: BatchJobSpec, environment_file: Path) -> str:
        return wrapper(
            spec.script_path,
            environment_file,
            signal_after_seconds=self._signal_after(spec),
        )

    # -- scheduler port --------------------------------------------------

    def render_submit(self, spec: BatchJobSpec) -> tuple[str, ...]:
        return self._submit_argv(spec, self.job_directory(spec) / "job.sh")

    def submit(self, spec: BatchJobSpec) -> str:
        if not spec.script_path.is_file():
            raise SetupError(f"{self.label} payload does not exist: {spec.script_path}")
        directory = self.job_directory(spec)
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            raise InfrastructureError(
                f"cannot create job directory {directory}: {error.strerror}"
            ) from None
        environment_file = directory / "environment"
        environment = job_environment(
            os.environ if self._environ is None else self._environ
        )
        write_private(
            environment_file,
            b"".join(
                f"{name}={value}".encode() + b"\0"
                for name, value in sorted(environment.items())
            ),
            0o600,
        )
        script = directory / "job.sh"
        write_private(script, self._job_script(spec, environment_file).encode(), 0o700)
        argv = self.render_submit(spec)
        name = self.job_name(spec)
        detail = ""
        started = self._clock()
        try:
            # Exclude older jobs of this name without changing the active-only
            # name check. A lost submit answer may belong to a job that has
            # already finished by the time we query the scheduler again.
            while True:
                try:
                    previous_ids = set(self._submission_ids(name))
                    break
                except TransientSchedulerError:
                    if self._clock() - started >= SUBMIT_OUTAGE_SECONDS:
                        raise
                    self._sleeper(min(MAX_BACKOFF_SECONDS, 30.0))
            while True:
                completed = self._runner(argv)
                if not completed.returncode:
                    job_id = self._job_id(completed.stdout)
                    if job_id is None:
                        raise InfrastructureError(
                            f"{self.label} submission returned no exact job ID"
                        )
                    self._environment_files[job_id] = environment_file
                    return job_id
                detail = command_detail(completed)
                if not self._transient(detail):
                    break
                if self._unsent(detail):
                    # Nothing reached the scheduler: ask again while it is
                    # down, for as long as a status poll would wait.
                    if self._clock() - started >= SUBMIT_OUTAGE_SECONDS:
                        break
                    self._sleeper(min(MAX_BACKOFF_SECONDS, 30.0))
                    continue
                # Never repeat an ambiguous submission. Look in both current
                # jobs and history, allowing a short delay for it to appear.
                try:
                    for attempt in range(COMMAND_ATTEMPTS):
                        found = tuple(
                            job_id
                            for job_id in self._retry(
                                lambda: self._submission_ids(name)
                            )
                            if job_id not in previous_ids
                        )
                        if len(found) == 1:
                            self._environment_files[found[0]] = environment_file
                            return found[0]
                        if found:
                            raise InfrastructureError(
                                f"several new {self.label} jobs are named {name}: "
                                f"{', '.join(found)}"
                            )
                        if attempt + 1 < COMMAND_ATTEMPTS:
                            self._sleeper(5.0 * 2**attempt)
                    raise InfrastructureError(
                        "no new job was found in active jobs or history"
                    )
                except InfrastructureError as error:
                    # Repeating now could queue the job twice; its environment
                    # file stays for a job that may exist.
                    raise _Kept(
                        f"{self.label} submission of {name} has an unknown "
                        f"outcome ({detail}; {error}); check the scheduler for "
                        "a job of that name before running again"
                    ) from None
            raise InfrastructureError(f"{self.label} submission failed: {detail}")
        except _Kept as error:
            raise InfrastructureError(str(error)) from None
        except BaseException:
            environment_file.unlink(missing_ok=True)
            raise

    def require_name_available(self, name: str) -> None:
        active = self._call_ids(bounded_name(name, self.name_limit))
        if active:
            raise InfrastructureError(
                f"{self.label} job name is already active: {name} ({', '.join(active)})"
            )

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float = 5.0,
        on_state: Callable[[str], None] | None = None,
    ) -> BatchJobResult:
        self._require_job_id(job_id)
        previous: str | None = None
        changed_at = self._clock()
        outage_since: float | None = None
        backoff = poll_seconds
        missing = 0
        # When a poll last saw the job run, and whether the scheduler went
        # quiet after it (its record of the end may then be late).
        last_running: float | None = None
        outage_after_running = False
        try:
            while True:
                try:
                    result = self._status(job_id)
                except TransientSchedulerError as error:
                    outage_after_running = True
                    now = self._clock()
                    outage_since = now if outage_since is None else outage_since
                    if now - outage_since >= STATUS_OUTAGE_SECONDS:
                        raise InfrastructureError(
                            f"{self.label} did not answer status polls for job "
                            f"{job_id} for {STATUS_OUTAGE_SECONDS / 3600:.0f} h "
                            f"({error}); the job may still run: check it, then "
                            "resume"
                        ) from None
                    backoff = min(MAX_BACKOFF_SECONDS, max(poll_seconds, backoff * 2))
                    self._sleeper(backoff)
                    continue
                outage_since = None
                backoff = poll_seconds
                if result is None:
                    missing += 1
                    if missing >= MISSING_POLLS:
                        self._forget_environment(job_id)
                        raise InfrastructureError(
                            f"{self.label} job {job_id} is unknown to the "
                            "scheduler, also in its history"
                        )
                else:
                    missing = 0
                    if result.state != previous:
                        changed_at = self._clock()
                        if on_state is not None:
                            on_state(result.state)
                    previous = result.state
                    if result.state not in ACTIVE_STATES:
                        self._forget_environment(job_id)
                        if outage_after_running:
                            result = self._after_outage(result, last_running)
                        return result
                    last_running = self._wall_clock()
                    outage_after_running = False
                interval = poll_seconds
                if self._clock() - changed_at >= STABLE_AFTER_SECONDS:
                    interval = max(poll_seconds, STABLE_POLL_SECONDS)
                self._sleeper(interval)
        except KeyboardInterrupt:
            self.cancel(job_id)
            raise KeyboardInterrupt(
                f"interrupted; {self.label} job {job_id} cancelled"
            ) from None

    def finished(self, job_id: str) -> BatchJobResult | None:
        """The job's end, or None while it runs or waits; state ``UNKNOWN``
        once the scheduler has forgotten it."""
        self._require_job_id(job_id)
        result = self._retry(lambda: self._status(job_id))
        if result is None:
            return BatchJobResult(job_id=job_id, state="UNKNOWN", exit_code=1)
        if result.state in ACTIVE_STATES:
            return None
        return result

    def cancel(self, job_id: str) -> None:
        """Delete the job; one that already ended counts as cancelled."""
        self._require_job_id(job_id)
        try:
            self._retry(lambda: self._cancel(job_id))
        finally:
            self._forget_environment(job_id)

    # -- helpers ---------------------------------------------------------

    def _transient(self, detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in self.transient_errors) or (
            "timed out after" in lowered
        )

    def _after_outage(
        self, result: BatchJobResult, last_running: float | None
    ) -> BatchJobResult:
        """The end of a job whose end the scheduler recorded after an outage."""
        return result

    def _unsent(self, detail: str) -> bool:
        lowered = detail.lower()
        return any(marker in lowered for marker in self.unsent_errors)

    def _retry[T](self, call: Callable[[], T]) -> T:
        """``call()`` again while the scheduler does not answer, up to
        COMMAND_ATTEMPTS times."""
        for attempt in range(1, COMMAND_ATTEMPTS + 1):
            try:
                return call()
            except TransientSchedulerError:
                if attempt == COMMAND_ATTEMPTS:
                    raise
                self._sleeper(5.0 * attempt)
        raise AssertionError("COMMAND_ATTEMPTS must be positive")

    def _call_ids(self, name: str) -> tuple[str, ...]:
        return self._retry(lambda: self._active_ids(name))

    def _checked(
        self, argv: tuple[str, ...], what: str
    ) -> subprocess.CompletedProcess[str]:
        """Run ``argv``; a scheduler that did not answer raises
        TransientSchedulerError, any other failure InfrastructureError."""
        completed = self._runner(argv)
        if completed.returncode:
            detail = command_detail(completed)
            error = (
                TransientSchedulerError
                if self._transient(detail)
                else InfrastructureError
            )
            raise error(f"cannot {what}: {detail}")
        return completed

    def _forget_environment(self, job_id: str) -> None:
        path = self._environment_files.pop(job_id, None)
        if path is not None:
            remove_private(path)

    # -- per scheduler ---------------------------------------------------

    def _submit_argv(self, spec: BatchJobSpec, script: Path) -> tuple[str, ...]:
        raise NotImplementedError

    def _job_id(self, stdout: str) -> str | None:
        raise NotImplementedError

    def _status(self, job_id: str) -> BatchJobResult | None:
        raise NotImplementedError

    def _active_ids(self, name: str) -> tuple[str, ...]:
        raise NotImplementedError

    def _submission_ids(self, name: str) -> tuple[str, ...]:
        """The user's jobs named ``name``, including completed history."""
        raise NotImplementedError

    def _cancel(self, job_id: str) -> None:
        raise NotImplementedError

    def _require_job_id(self, job_id: str) -> None:
        raise NotImplementedError


class _Kept(InfrastructureError):
    """A failed submission whose environment file must stay."""
