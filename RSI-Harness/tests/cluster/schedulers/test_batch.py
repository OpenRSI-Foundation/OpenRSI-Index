"""The batch schedulers' shared job wrapper, submit environment and client
calls, on a minimal scheduler and in real bash."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

from rsi_harness.cluster.schedulers import batch
from rsi_harness.cluster.schedulers.batch import (
    MAX_BACKOFF_SECONDS,
    MISSING_POLLS,
    STABLE_POLL_SECONDS,
    STATUS_OUTAGE_SECONDS,
    BatchJobResult,
    BatchJobSpec,
    BatchScheduler,
    bounded_name,
    job_environment,
    run_command,
    wrapper,
)
from rsi_harness.errors import InfrastructureError, SetupError

TOKEN = "a1b2c3"
BASE_PATH = "/usr/bin:/bin"
NAME = "mt-rsi-task-codex-run-alice"
# The name the Toy scheduler (24 bytes) gives the job.
JOB_NAME = bounded_name(NAME, 24)


def completed(stdout="", *, stderr="", code=0):
    return subprocess.CompletedProcess((), code, stdout, stderr)


DOWN = completed(stderr="tstat: server down", code=2)


class Runner:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        return next(self.results)


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class ToyScheduler(BatchScheduler):
    """The smallest scheduler: ``tsub`` prints the ID, ``tstat`` prints
    ``STATE EXIT`` (nothing for an unknown job), ``tsel`` the active IDs of
    a name, ``tdel`` deletes."""

    label = "Toy"
    name_limit = 24
    transient_errors = ("server down",)

    def __init__(self, **kwargs):
        kwargs.setdefault("token", lambda: TOKEN)
        kwargs.setdefault("environ", {})
        super().__init__(**kwargs)

    def _submit_argv(self, spec, script):
        return ("tsub", "-N", self.job_name(spec), str(script))

    def _job_id(self, stdout):
        job_id = stdout.strip()
        return job_id if job_id.isdigit() else None

    def _status(self, job_id):
        words = self._checked(("tstat", job_id), "query").stdout.split()
        if not words:
            return None
        return BatchJobResult(job_id=job_id, state=words[0], exit_code=int(words[1]))

    def _active_ids(self, name):
        return tuple(self._checked(("tsel", name), "list").stdout.split())

    def _submission_ids(self, name):
        return tuple(self._checked(("thistory", name), "list history").stdout.split())

    def _cancel(self, job_id):
        self._checked(("tdel", job_id), "cancel")

    def _require_job_id(self, job_id):
        if not job_id.isdigit():
            raise SetupError(f"invalid exact Toy job ID: {job_id!r}")


def spec(tmp_path: Path, **updates) -> BatchJobSpec:
    script = tmp_path / "run.sh"
    if not script.exists():
        script.write_text("exit 0\n")
    return BatchJobSpec(
        **{
            "name": NAME,
            "queue": "gpu",
            "group": "",
            "cpu_slots": 8,
            "memory_mb": 65536,
            "walltime": "02:15",
            "stdout_path": tmp_path / "run" / "toy.%j.out",
            "stderr_path": tmp_path / "run" / "toy.%j.err",
            "script_path": script,
            **updates,
        }
    )


def job_files(scheduler: BatchScheduler, request: BatchJobSpec) -> tuple[Path, Path]:
    directory = scheduler.job_directory(request)
    return directory / "job.sh", directory / "environment"


def records(path: Path) -> dict[str, str]:
    return dict(
        item.split("=", 1) for item in path.read_bytes().decode().split("\0") if item
    )


def write_environment(path: Path, environment: dict[str, str]) -> Path:
    path.write_bytes(
        b"".join(
            f"{name}={value}".encode() + b"\0" for name, value in environment.items()
        )
    )
    path.chmod(0o600)
    return path


def run_bash(script: Path, env: dict[str, str] | None = None, **kwargs):
    return subprocess.run(
        ["/bin/bash", str(script)],
        env={"PATH": BASE_PATH, **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
        **kwargs,
    )


# --- The submit environment, without the server's job record --------------

SUBMIT_ENVIRONMENT = {
    "PATH": "/opt/agents/bin:/usr/bin:/bin",
    "HOME": "/home/alice",
    "RSI_AGENT_API_KEY": "sk-not-in-the-job-record",
    "RSI_AGENT_API_BASE_URL": "https://api.example.test/v1",
    "https_proxy": "http://login-ib0:3128",
    # Values a sourced shell file would mangle.
    "QUOTED": 'it\'s "quoted" $HOME `id` \\ end',
    "MULTILINE": "first\nsecond\n",
    "EMPTY": "",
    # Set by the scheduler for the job itself, or only true of the
    # submitting shell.
    "PBS_JOBID": "97176.jobm",
    "PBS_NODEFILE": "/var/spool/pbs/aux/97176.jobm",
    "PJM_JOBID": "1006903",
    "PJM_SSD_DIR": "/local/1006903",
    "SLURM_JOB_ID": "5",
    "NCPUS": "36",
    "OMP_NUM_THREADS": "36",
    "CUDA_VISIBLE_DEVICES": "GPU-5f753abc-1811-842d-aa04-3a2f8503f00d",
    "TMPDIR": "/tmp/pbs.97176.jobm",
    "HOSTNAME": "flow2-login09",
    "SHELLOPTS": "braceexpand:hashall",
    "BASHOPTS": "checkwinsize",
    "BASH_FUNC_module%%": "() {  eval $(modulecmd bash $*)\n}",
    "SHLVL": "2",
    "TMOUT": "900",
    "SSH_AUTH_SOCK": "/tmp/ssh-x/agent.1",
    "_": "/usr/bin/rsi-harness",
}
JOB_LOCAL = {
    "PBS_JOBID",
    "PBS_NODEFILE",
    "PJM_JOBID",
    "PJM_SSD_DIR",
    "SLURM_JOB_ID",
    "NCPUS",
    "OMP_NUM_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "TMPDIR",
    "HOSTNAME",
    "SHELLOPTS",
    "BASHOPTS",
    "BASH_FUNC_module%%",
    "SHLVL",
    "TMOUT",
    "SSH_AUTH_SOCK",
    "_",
}


def test_the_job_environment_leaves_job_local_variables_to_the_scheduler():
    environment = job_environment(SUBMIT_ENVIRONMENT)
    assert environment == {
        name: value
        for name, value in SUBMIT_ENVIRONMENT.items()
        if name not in JOB_LOCAL
    }
    assert environment["RSI_AGENT_API_KEY"] == "sk-not-in-the-job-record"
    assert environment["MULTILINE"] == "first\nsecond\n"
    assert environment["EMPTY"] == ""
    # A value no NUL-separated record can hold is dropped, not truncated.
    assert job_environment({"A": "x\0y", "B": "ok"}) == {"B": "ok"}


def test_submit_writes_a_private_environment_and_wrapper(tmp_path):
    runner = Runner([completed(), completed("41\n")])
    scheduler = ToyScheduler(runner=runner, environ=SUBMIT_ENVIRONMENT)
    request = spec(tmp_path)
    assert scheduler.submit(request) == "41"
    script, environment_file = job_files(scheduler, request)
    # Beside the job's logs, named by the job, never in the frozen control
    # directory.
    assert script.parent == tmp_path / "run" / f"{JOB_NAME}-{TOKEN}"
    assert runner.calls == [
        ("thistory", JOB_NAME),
        ("tsub", "-N", JOB_NAME, str(script)),
    ]
    assert stat.S_IMODE(script.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(environment_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(script.stat().st_mode) == 0o700
    assert records(environment_file) == job_environment(SUBMIT_ENVIRONMENT)
    text = script.read_text()
    assert "sk-not-in-the-job-record" not in text
    assert str(environment_file) in text and str(request.script_path) in text
    # render_submit names the same wrapper (a dry run shows what runs).
    assert scheduler.render_submit(request)[-1] == str(script)


def test_submit_replaces_a_stale_file_and_never_follows_a_link(tmp_path):
    scheduler = ToyScheduler(
        runner=Runner([completed(), completed("41")]), environ={"A": "1"}
    )
    request = spec(tmp_path)
    _, environment_file = job_files(scheduler, request)
    environment_file.parent.mkdir(parents=True)
    target = tmp_path / "elsewhere"
    target.write_text("keep")
    environment_file.symlink_to(target)
    scheduler.submit(request)
    assert target.read_text() == "keep"
    assert not environment_file.is_symlink()
    assert records(environment_file) == {"A": "1"}


def test_submit_refuses_a_missing_payload_before_writing_anything(tmp_path):
    runner = Runner([])
    scheduler = ToyScheduler(runner=runner)
    request = spec(tmp_path)
    request.script_path.unlink()
    with pytest.raises(SetupError, match="payload does not exist"):
        scheduler.submit(request)
    assert runner.calls == []
    assert not scheduler.job_directory(request).exists()


@pytest.mark.parametrize("output", ["", "Job 41 submitted.", "41\n42"])
def test_submit_rejects_an_answer_without_one_exact_job_id(tmp_path, output):
    scheduler = ToyScheduler(
        runner=Runner([completed(), completed(output)]), environ={"K": "v"}
    )
    request = spec(tmp_path)
    with pytest.raises(InfrastructureError, match="exact job ID"):
        scheduler.submit(request)
    # The job (if any) never finds an environment to run with.
    assert not job_files(scheduler, request)[1].exists()


def test_a_rejected_submission_is_not_repeated_and_removes_its_environment(
    tmp_path,
):
    runner = Runner([completed(), completed(stderr="tsub: unknown queue gpu", code=1)])
    scheduler = ToyScheduler(runner=runner, environ={"K": "v"})
    request = spec(tmp_path)
    with pytest.raises(InfrastructureError, match="Toy submission failed: .*unknown"):
        scheduler.submit(request)
    assert [call[0] for call in runner.calls] == ["thistory", "tsub"]
    assert not job_files(scheduler, request)[1].exists()


def test_a_lost_submit_answer_finds_the_job_by_its_name(tmp_path):
    runner = Runner([completed(), DOWN, completed("41\n")])
    scheduler = ToyScheduler(runner=runner, sleeper=lambda _: None)
    request = spec(tmp_path)
    assert scheduler.submit(request) == "41"
    # Found, not repeated.
    assert runner.calls[2] == ("thistory", JOB_NAME)
    assert [call[0] for call in runner.calls] == ["thistory", "tsub", "thistory"]
    # The adopted job's environment is its own: cleaned once it ends.
    environment_file = job_files(scheduler, request)[1]
    assert environment_file.exists()
    scheduler._runner = Runner([completed("CANCELLED 1")])
    scheduler.wait("41")
    assert not environment_file.exists()


def test_reconciliation_waits_for_a_delayed_job_record_without_resubmitting(tmp_path):
    runner = Runner(
        [
            completed(),
            DOWN,
            completed(""),
            completed(stderr="tsub: timed out after 120s", code=124),
            completed(""),
            completed("43\n"),
        ]
    )
    sleeps = []
    job_id = ToyScheduler(runner=runner, sleeper=sleeps.append).submit(spec(tmp_path))
    assert job_id == "43"
    assert [call[0] for call in runner.calls] == ["thistory", "tsub"] + ["thistory"] * 4
    assert sleeps == [5.0, 5.0, 10.0]


def test_a_short_job_finishing_before_submit_times_out_is_not_run_twice(tmp_path):
    request = spec(tmp_path)
    marker = tmp_path / "executions"
    request.script_path.write_text(f"echo ran >> {marker}\n")
    calls = []
    accepted = False

    def runner(argv):
        nonlocal accepted
        calls.append(argv)
        if argv[0] == "thistory":
            # A previous completed job has the same name; only 42 is new.
            return completed("41 42" if accepted else "41")
        assert argv[0] == "tsub" and not accepted
        accepted = True
        result = run_bash(Path(argv[-1]))
        assert result.returncode == 0, result.stderr
        return completed(stderr="tsub: timed out after 120s", code=124)

    scheduler = ToyScheduler(runner=runner, sleeper=lambda _: None)
    assert scheduler.submit(request) == "42"
    assert marker.read_text() == "ran\n"
    assert not job_files(scheduler, request)[1].exists()
    assert [call[0] for call in calls] == ["thistory", "tsub", "thistory"]


def test_a_snapshot_outage_is_waited_for_before_submitting(tmp_path):
    runner = Runner([DOWN, completed("40"), completed("41")])
    clock = Clock()
    scheduler = ToyScheduler(runner=runner, clock=clock, sleeper=clock.sleep)
    assert scheduler.submit(spec(tmp_path)) == "41"
    assert [call[0] for call in runner.calls] == ["thistory", "thistory", "tsub"]
    assert clock.sleeps == [30.0]


def test_a_submission_missing_from_all_listings_is_not_resubmitted(
    tmp_path,
):
    runner = Runner([completed(), DOWN, *[completed("")] * 3])
    sleeps = []
    scheduler = ToyScheduler(runner=runner, sleeper=sleeps.append, environ={"K": "v"})
    request = spec(tmp_path)
    with pytest.raises(InfrastructureError, match="unknown outcome.*no new job"):
        scheduler.submit(request)
    assert [call[0] for call in runner.calls].count("tsub") == 1
    assert sleeps == [5.0, 10.0]
    assert job_files(scheduler, request)[1].exists()


@pytest.mark.parametrize(
    "listing", [DOWN, completed(stderr="tsel: permission denied", code=1)]
)
def test_an_unknown_submit_outcome_is_never_submitted_twice(tmp_path, listing):
    # A listing that does not answer is asked three times.
    tries = 3 if listing is DOWN else 1
    runner = Runner([completed(), DOWN, *tries * [listing]])
    scheduler = ToyScheduler(runner=runner, sleeper=lambda _: None, environ={"K": "v"})
    request = spec(tmp_path)
    with pytest.raises(InfrastructureError, match="unknown outcome") as error:
        scheduler.submit(request)
    assert JOB_NAME in str(error.value)
    assert [call[0] for call in runner.calls] == ["thistory", "tsub"] + tries * [
        "thistory"
    ]
    # A job that may exist still finds its environment.
    assert job_files(scheduler, request)[1].exists()


def test_several_jobs_of_the_name_are_never_adopted(tmp_path):
    runner = Runner([completed(), DOWN, completed("41 42\n")])
    scheduler = ToyScheduler(runner=runner, sleeper=lambda _: None, environ={"K": "v"})
    request = spec(tmp_path)
    with pytest.raises(InfrastructureError, match="several new Toy jobs.*41, 42"):
        scheduler.submit(request)
    assert [call[0] for call in runner.calls].count("tsub") == 1
    # One may be ours; don't break its queued environment or submit again.
    assert job_files(scheduler, request)[1].exists()


def test_a_missing_client_is_a_refusal_that_leaves_no_environment(tmp_path):
    class Missing(ToyScheduler):
        def _submission_ids(self, name):
            return ()

        def _submit_argv(self, spec, script):
            return (str(tmp_path / "no-such-tsub"), str(script))

    scheduler = Missing(environ={"K": "v"})
    request = spec(tmp_path)
    with pytest.raises(InfrastructureError, match="Toy submission failed"):
        scheduler.submit(request)
    assert not job_files(scheduler, request)[1].exists()


def test_name_collisions_use_the_bounded_job_name():
    runner = Runner([completed(""), completed("41\n")])
    scheduler = ToyScheduler(runner=runner)
    scheduler.require_name_available(NAME)
    with pytest.raises(InfrastructureError, match="already active.*41"):
        scheduler.require_name_available(NAME)
    # The name submit gives the job.
    assert runner.calls[0] == ("tsel", JOB_NAME)
    # A scheduler that does not answer is asked three times.
    runner = Runner([DOWN] * 3)
    sleeps = []
    with pytest.raises(InfrastructureError, match="server down"):
        ToyScheduler(runner=runner, sleeper=sleeps.append).require_name_available("x")
    assert len(runner.calls) == 3 and sleeps == [5.0, 10.0]


def test_bounded_names_are_stable_and_fit():
    assert bounded_name("short", 24) == "short"
    long = "mt-rsi-a-very-long-task-name-codex-run-alice"
    shortened = bounded_name(long, 24)
    assert len(shortened.encode()) <= 24
    assert shortened == bounded_name(long, 24) != bounded_name(long + "x", 24)
    assert shortened.startswith("mt-rsi-a-very")


def test_the_job_directory_is_kept_for_one_payload(tmp_path):
    tokens = iter(["aaaaaa", "bbbbbb"])
    scheduler = ToyScheduler(token=lambda: next(tokens))
    first = spec(tmp_path)
    other = spec(tmp_path, script_path=tmp_path / "other.sh")
    assert scheduler.job_directory(first) == scheduler.job_directory(first)
    assert scheduler.job_directory(first).name.endswith("aaaaaa")
    assert scheduler.job_directory(other).name.endswith("bbbbbb")


# --- Bounded client calls ----------------------------------------------------


def test_every_client_call_is_bounded_in_time(monkeypatch):
    monkeypatch.setattr(batch, "COMMAND_TIMEOUT_SECONDS", 0.2)
    started = time.monotonic()
    result = run_command(("sleep", "5"))
    assert time.monotonic() - started < 3
    assert result.returncode == 124
    # Counts as a scheduler that did not answer, for any scheduler.
    assert ToyScheduler()._transient(result.stderr)


def test_a_missing_client_is_a_refusal_not_a_crash(tmp_path):
    result = run_command((str(tmp_path / "qsub"),))
    assert result.returncode == 127 and "qsub" in result.stderr
    assert not ToyScheduler()._transient(result.stderr)


# --- Waiting, polling and cancelling ----------------------------------------


def test_wait_reports_each_state_once_and_returns_the_end():
    runner = Runner(
        [
            completed("PENDING 0"),
            completed("RUNNING 0"),
            completed("RUNNING 0"),
            completed("COMPLETING 0"),
            completed("FAILED 3"),
        ]
    )
    states, sleeps = [], []
    result = ToyScheduler(runner=runner, sleeper=sleeps.append).wait(
        "41", poll_seconds=2, on_state=states.append
    )
    assert (result.state, result.exit_code) == ("FAILED", 3)
    assert states == ["PENDING", "RUNNING", "COMPLETING", "FAILED"]
    assert sleeps == [2, 2, 2, 2]


def test_wait_retries_transient_errors_with_backoff():
    runner = Runner(
        [DOWN, DOWN, completed("RUNNING 0"), DOWN, completed("COMPLETED 0")]
    )
    sleeps = []
    result = ToyScheduler(runner=runner, sleeper=sleeps.append).wait(
        "41", poll_seconds=5
    )
    assert result.state == "COMPLETED"
    # Less and less often while it does not answer; normal again once it does.
    assert sleeps == [10.0, 20.0, 5.0, 10.0]


def test_wait_gives_up_after_a_long_scheduler_outage():
    clock = Clock()
    runner = Runner([DOWN] * 1000)
    scheduler = ToyScheduler(runner=runner, sleeper=clock.sleep, clock=clock)
    with pytest.raises(InfrastructureError, match="did not answer status polls"):
        scheduler.wait("41", poll_seconds=10)
    assert clock.now >= STATUS_OUTAGE_SECONDS
    # Backed off to the cap: some 30 polls in two hours, not 720.
    assert max(clock.sleeps) == MAX_BACKOFF_SECONDS and len(runner.calls) < 40


def test_wait_polls_a_job_whose_state_stays_the_same_less_often():
    clock = Clock()
    runner = Runner([completed("RUNNING 0")] * 40 + [completed("COMPLETED 0")])
    ToyScheduler(runner=runner, sleeper=clock.sleep, clock=clock).wait(
        "41", poll_seconds=10
    )
    assert clock.sleeps[:30] == [10.0] * 30
    assert set(clock.sleeps[30:]) == {STABLE_POLL_SECONDS}


def test_wait_bounds_a_job_the_scheduler_does_not_know(tmp_path):
    scheduler = ToyScheduler(
        runner=Runner([completed(), completed("41"), *[completed("")] * MISSING_POLLS]),
        sleeper=lambda _: None,
        environ={"K": "v"},
    )
    request = spec(tmp_path)
    scheduler.submit(request)
    with pytest.raises(InfrastructureError, match="unknown to the scheduler"):
        scheduler.wait("41")
    assert not job_files(scheduler, request)[1].exists()


def test_a_job_that_was_briefly_unknown_is_still_followed():
    runner = Runner([completed(""), completed(""), completed("COMPLETED 0")])
    result = ToyScheduler(runner=runner, sleeper=lambda _: None).wait("41")
    assert result.state == "COMPLETED"


def test_wait_deletes_the_environment_of_a_job_that_never_started(tmp_path):
    scheduler = ToyScheduler(
        runner=Runner(
            [
                completed(),
                completed("41"),
                completed("PENDING 0"),
                completed("CANCELLED 1"),
            ]
        ),
        sleeper=lambda _: None,
        environ={"K": "v"},
    )
    request = spec(tmp_path)
    scheduler.wait(scheduler.submit(request))
    assert not job_files(scheduler, request)[1].exists()


def test_a_running_jobs_environment_is_left_alone(tmp_path):
    scheduler = ToyScheduler(
        runner=Runner([completed(), completed("41"), completed("RUNNING 0")]),
        environ={"K": "v"},
    )
    request = spec(tmp_path)
    scheduler.submit(request)
    assert scheduler.finished("41") is None
    assert job_files(scheduler, request)[1].exists()


def test_a_fresh_process_removes_the_environment_of_an_ended_job(tmp_path, monkeypatch):
    # The controller that submitted the job died while it was queued.
    request = spec(tmp_path)
    ToyScheduler(
        runner=Runner([completed(), completed("41")]), environ={"K": "v"}
    ).submit(request)
    left = job_files(ToyScheduler(), request)[1]
    # Not this user's, a link, a link to a directory: kept.
    foreign = tmp_path / "run" / "foreign"
    foreign.mkdir()
    (foreign / "environment").write_text("K=v")
    target = tmp_path / "target"
    target.write_text("kept")
    linked = tmp_path / "run" / "linked"
    linked.mkdir()
    (linked / "environment").symlink_to(target)
    (tmp_path / "run" / "outside").symlink_to(foreign)
    uid = os.getuid()
    owner = {str(foreign / "environment"): uid + 1}
    real_lstat = Path.lstat

    def lstat(path):
        status = real_lstat(path)
        if str(path) not in owner:
            return status
        values = list(status)
        values[stat.ST_UID] = owner[str(path)]
        return os.stat_result(values)

    monkeypatch.setattr(Path, "lstat", lstat)

    fresh = ToyScheduler(runner=Runner([completed("CANCELLED 1")]))
    assert fresh.finished("41").state == "CANCELLED"
    assert left.exists()
    batch.remove_environment_files(tmp_path / "run", tmp_path / "missing")

    assert not left.exists()
    assert (foreign / "environment").exists()
    assert (linked / "environment").is_symlink() and target.read_text() == "kept"


def test_interrupting_wait_cancels_the_exact_job(tmp_path):
    scheduler = ToyScheduler(
        runner=Runner(
            [completed(), completed("41"), completed("RUNNING 0"), completed()]
        ),
        environ={"K": "v"},
    )
    request = spec(tmp_path)
    scheduler.submit(request)

    def interrupt(_):
        raise KeyboardInterrupt

    scheduler._sleeper = interrupt
    with pytest.raises(KeyboardInterrupt, match="Toy job 41 cancelled"):
        scheduler.wait("41")
    assert scheduler._runner.calls[-1] == ("tdel", "41")
    assert not job_files(scheduler, request)[1].exists()


def test_finished_is_none_while_active_and_unknown_once_forgotten():
    for state in ("PENDING", "RUNNING", "COMPLETING", "SUSPENDED"):
        runner = Runner([completed(f"{state} 0")])
        assert ToyScheduler(runner=runner).finished("41") is None
    result = ToyScheduler(runner=Runner([completed("")])).finished("41")
    assert (result.job_id, result.state, result.exit_code) == ("41", "UNKNOWN", 1)
    result = ToyScheduler(runner=Runner([completed("TIMEOUT 1")])).finished("41")
    assert result.state == "TIMEOUT"


def test_finished_asks_a_scheduler_that_does_not_answer_again():
    runner = Runner([DOWN, completed("COMPLETED 0")])
    sleeps = []
    result = ToyScheduler(runner=runner, sleeper=sleeps.append).finished("41")
    assert result.state == "COMPLETED" and sleeps == [5.0]
    runner = Runner([DOWN] * 3)
    with pytest.raises(InfrastructureError, match="down"):
        ToyScheduler(runner=runner, sleeper=lambda _: None).finished("41")


def test_cancel_retries_a_scheduler_that_does_not_answer_then_reports(tmp_path):
    runner = Runner([DOWN, completed()])
    sleeps = []
    ToyScheduler(runner=runner, sleeper=sleeps.append).cancel("41")
    assert len(runner.calls) == 2 and sleeps == [5.0]
    scheduler = ToyScheduler(
        runner=Runner(
            [completed(), completed("41"), completed(stderr="not yours", code=1)]
        ),
        environ={"K": "v"},
    )
    request = spec(tmp_path)
    scheduler.submit(request)
    with pytest.raises(InfrastructureError, match="not yours"):
        scheduler.cancel("41")
    # Even a failed cancel leaves no environment behind.
    assert not job_files(scheduler, request)[1].exists()


@pytest.mark.parametrize("job_id", ["all", "-x", "41 42", ""])
def test_an_invalid_job_id_never_reaches_the_scheduler(job_id):
    runner = Runner([])
    scheduler = ToyScheduler(runner=runner)
    for call in (scheduler.cancel, scheduler.finished, scheduler.wait):
        with pytest.raises(SetupError, match="exact Toy job ID"):
            call(job_id)
    assert runner.calls == []


# --- The job wrapper, in real bash -------------------------------------------


def test_the_wrapper_restores_the_environment_and_deletes_its_file(tmp_path):
    scheduler = ToyScheduler(
        runner=Runner([completed(), completed("41")]), environ=SUBMIT_ENVIRONMENT
    )
    request = spec(tmp_path)
    request.script_path.write_text(
        'printf "%s\\0" "$RSI_AGENT_API_KEY" "$QUOTED" "$MULTILINE" '
        '"${EMPTY-unset}" "$PBS_JOBID" "$NCPUS" "${TMOUT-unset}" "$PATH"\n'
    )
    scheduler.submit(request)
    script, environment_file = job_files(scheduler, request)
    # As the scheduler runs it: in the job's own environment.
    job = run_bash(script, {"PBS_JOBID": "97177.jobm", "NCPUS": "144"})
    assert job.returncode == 0, job.stderr
    assert job.stdout.split("\0")[:-1] == [
        "sk-not-in-the-job-record",
        SUBMIT_ENVIRONMENT["QUOTED"],
        "first\nsecond\n",
        "",
        # The scheduler's values for the job, not the submit host's.
        "97177.jobm",
        "144",
        "unset",
        SUBMIT_ENVIRONMENT["PATH"],
    ]
    assert not environment_file.exists()


def test_a_variable_the_jobs_shell_made_read_only_keeps_the_jobs_value(tmp_path):
    """A hardened site may export a read-only variable from /etc/profile.d;
    the job must still start."""
    request = spec(tmp_path)
    request.script_path.write_text('echo "LIMIT=$LIMIT OTHER=$OTHER"\n')
    scheduler = ToyScheduler(
        runner=Runner([completed(), completed("41")]),
        environ={"LIMIT": "0", "OTHER": "kept"},
    )
    scheduler.submit(request)
    script, environment_file = job_files(scheduler, request)
    job = subprocess.run(
        ["/bin/bash", "-c", f"readonly LIMIT=900; export LIMIT; source {script}"],
        env={"PATH": BASE_PATH},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert job.returncode == 0, job.stderr
    assert job.stdout.strip() == "LIMIT=900 OTHER=kept"
    assert not environment_file.exists()


def test_the_wrapper_refuses_to_run_without_its_environment(tmp_path):
    marker = tmp_path / "ran"
    payload = tmp_path / "payload.sh"
    payload.write_text(f": > {marker}\n")
    script = tmp_path / "job.sh"
    script.write_text(wrapper(payload, tmp_path / "environment"))
    job = run_bash(script)
    assert job.returncode == 1
    assert "job environment" in job.stderr and "missing" in job.stderr
    assert not marker.exists()


@pytest.mark.parametrize("code", [0, 3, 137])
def test_the_wrapper_exits_with_the_payloads_status(tmp_path, code):
    payload = tmp_path / "payload.sh"
    payload.write_text(f"exit {code}\n")
    script = tmp_path / "job.sh"
    script.write_text(wrapper(payload, write_environment(tmp_path / "env", {})))
    assert run_bash(script).returncode == code


def test_the_wrapper_redirects_the_payloads_output(tmp_path):
    payload = tmp_path / "payload.sh"
    payload.write_text("echo out; echo err >&2\n")
    script = tmp_path / "job.sh"
    logs = tmp_path / "with space"
    logs.mkdir()
    script.write_text(
        wrapper(
            payload,
            write_environment(tmp_path / "env", {}),
            stdout=f"{logs}/job.${{JOB}}.out",
            stderr=f"{logs}/job.${{JOB}}.err",
        )
    )
    (logs / "job.7.out").write_text("earlier\n")
    job = run_bash(script, {"JOB": "7"})
    assert job.returncode == 0 and job.stdout == "" and job.stderr == ""
    # Appended, as a resumed job's logs are.
    assert (logs / "job.7.out").read_text() == "earlier\nout\n"
    assert (logs / "job.7.err").read_text() == "err\n"
    # Without both paths the scheduler names the logs itself.
    script.write_text(
        wrapper(payload, write_environment(tmp_path / "env", {}), stdout="x")
    )
    job = run_bash(script)
    assert (job.stdout, job.stderr) == ("out\n", "err\n")


# A payload that reports the TERM it gets, then exits 7.
TRAPPING = """\
trap 'echo "TERM after $SECONDS s" > "$MARK"; kill "$child" 2>/dev/null; exit 7' TERM
: > "$READY"
sleep 30 &
child=$!
wait "$child"
echo "no TERM" > "$MARK"
"""


def test_the_wrapper_sends_the_advance_term_at_its_own_time(tmp_path):
    payload = tmp_path / "payload.sh"
    payload.write_text(TRAPPING)
    mark = tmp_path / "mark"
    environment = write_environment(
        tmp_path / "env", {"MARK": str(mark), "READY": str(tmp_path / "ready")}
    )
    script = tmp_path / "job.sh"
    script.write_text(wrapper(payload, environment, signal_after_seconds=1))
    started = time.monotonic()
    job = run_bash(script)
    elapsed = time.monotonic() - started
    assert job.returncode == 7, job.stderr
    assert mark.read_text().startswith("TERM after")
    assert 0.9 <= elapsed < 15


def test_the_advance_term_comes_signal_seconds_before_the_walltime(tmp_path):
    scheduler = ToyScheduler()
    assert scheduler._signal_after(spec(tmp_path)) == 0
    assert scheduler._signal_after(spec(tmp_path, signal_seconds=600)) == 135 * 60 - 600
    # Never at or before the start.
    assert (
        scheduler._signal_after(spec(tmp_path, walltime="00:05", signal_seconds=900))
        == 1
    )
    without = scheduler._job_script(spec(tmp_path), tmp_path / "env")
    assert "sleep" not in without
    script = scheduler._job_script(spec(tmp_path, signal_seconds=600), tmp_path / "env")
    assert f"sleep {135 * 60 - 600}" in script


def test_a_payload_that_ends_first_gets_no_advance_term(tmp_path):
    payload = tmp_path / "payload.sh"
    payload.write_text("exit 4\n")
    script = tmp_path / "job.sh"
    script.write_text(
        wrapper(
            payload, write_environment(tmp_path / "env", {}), signal_after_seconds=30
        )
    )
    started = time.monotonic()
    assert run_bash(script).returncode == 4
    # The wrapper does not wait for its timer.
    assert time.monotonic() - started < 10


def _start(tmp_path, payload_text):
    payload = tmp_path / "payload.sh"
    payload.write_text(payload_text)
    ready = tmp_path / "ready"
    environment = write_environment(
        tmp_path / "env", {"MARK": str(tmp_path / "mark"), "READY": str(ready)}
    )
    script = tmp_path / "job.sh"
    script.write_text(wrapper(payload, environment))
    job = subprocess.Popen(
        ["/bin/bash", str(script)],
        env={"PATH": BASE_PATH},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while not ready.exists():
        assert time.monotonic() < deadline and job.poll() is None
        time.sleep(0.01)
    # Past the wrapper's trap.
    time.sleep(0.2)
    return job


@pytest.mark.parametrize("sent", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_signal_to_the_wrapper_reaches_the_payload_as_term(tmp_path, sent):
    job = _start(tmp_path, TRAPPING)
    os.kill(job.pid, sent)
    try:
        # The wrapper waits for the payload's own end and exits with it.
        assert job.wait(timeout=20) == 7
    finally:
        job.kill()
    assert (tmp_path / "mark").read_text().startswith("TERM after")


def test_a_payload_killed_by_the_forwarded_term_reports_it(tmp_path):
    job = _start(tmp_path, ': > "$READY"\nexec sleep 30\n')
    os.kill(job.pid, signal.SIGTERM)
    try:
        assert job.wait(timeout=20) == 128 + signal.SIGTERM
    finally:
        job.kill()
