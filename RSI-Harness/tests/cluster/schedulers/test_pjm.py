"""Fujitsu TCS (PJM) commands and the states they report, with answers
shaped as the PJM command reference documents them: pjsub -z jid prints the
job ID alone; pjstat prints a table whose header names the --choose items,
"-" for an unset value, and only the header for a job it does not know."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from rsi_harness.cluster.schedulers import pjm
from rsi_harness.cluster.schedulers.pjm import HISTORY_DAYS, PJMJobSpec, PJMScheduler
from rsi_harness.errors import InfrastructureError, SetupError

JOB = "1006903"
TOKEN = "a1b2c3"
DAEMON = "[ERR.] PJM 0290 pjstat PJM daemon is not present."
LOADING = "[INFO] PJM 0281 pjstat Data loading..."


def completed(stdout="", *, stderr="", code=0):
    return subprocess.CompletedProcess((), code, stdout, stderr)


def table(*rows: tuple[str, ...], header=("JOB_ID", "ST", "EC", "PC", "SN")):
    """pjstat's fixed-width table."""
    widths = (10, 3, 3, 3, 3)[: len(header)] if header[1] == "ST" else (10, 10, 3)
    lines = [header, *rows]
    return completed(
        "".join(
            " ".join(
                value.ljust(width) for value, width in zip(line, widths, strict=False)
            ).rstrip()
            + "\n"
            for line in lines
        )
    )


def row(st, ec="-", pc="-", sn="-", job_id=JOB):
    return (job_id, st, ec, pc, sn)


NONE = table()


def current(st, **values):
    """The job in pjstat's current listing."""
    return table(row(st, **values))


def ended(st, ec="-", pc="-", sn="-"):
    """The job only in the history (pjstat -H)."""
    return [NONE, table(row(st, ec, pc, sn))]


class Runner:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        return next(self.results)


def scheduler(*results, **kwargs) -> PJMScheduler:
    kwargs.setdefault("sleeper", lambda _: None)
    kwargs.setdefault("token", lambda: TOKEN)
    kwargs.setdefault("environ", {})
    return PJMScheduler(runner=Runner(results), **kwargs)


def spec(tmp_path: Path, **updates) -> PJMJobSpec:
    script = tmp_path / "run.sh"
    if not script.exists():
        script.write_text("exit 0\n")
    return PJMJobSpec(
        **{
            "name": "mt-rsi-task-codex-run-alice",
            "queue": "b-batch",
            "group": "",
            "cpu_slots": 32,
            "memory_mb": 131072,
            "walltime": "02:15",
            "stdout_path": tmp_path / "run" / "pjm.%j.out",
            "stderr_path": tmp_path / "run" / "pjm.%j.err",
            "script_path": script,
            "gpu_count": 4,
            **updates,
        }
    )


def job_dir(tmp_path: Path) -> Path:
    return tmp_path / "run" / f"mt-rsi-task-codex-run-alice-{TOKEN}"


# --- Rendering --------------------------------------------------------------


def test_render_one_job_with_its_gpus_and_elapse_limit(tmp_path):
    argv = scheduler().render_submit(spec(tmp_path, resources=("jobenv=singularity",)))
    assert argv == (
        "pjsub",
        "-z",
        "jid",
        "-N",
        "mt-rsi-task-codex-run-alice",
        "-L",
        "rscgrp=b-batch,elapse=02:15:00,gpu=4,jobenv=singularity",
        # PJM names the logs itself (%j is the job ID).
        "-o",
        str(tmp_path / "run" / "pjm.%j.out"),
        "-e",
        str(tmp_path / "run" / "pjm.%j.err"),
        "--norestart",
        # The wrapper that loads the private environment, never -X.
        str(job_dir(tmp_path) / "job.sh"),
    )


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        # A whole node, whatever its GPUs.
        ({"exclusive": True}, "rscgrp=b-batch,elapse=02:15:00,node=1"),
        # A build job without GPUs asks for cores where the group needs them.
        (
            {"gpu_count": 0, "cpu_resource": "vnode-core"},
            "rscgrp=b-batch,elapse=02:15:00,vnode-core=32",
        ),
        ({"gpu_count": 0}, "rscgrp=b-batch,elapse=02:15:00"),
        # And their memory where the group needs it.
        (
            {
                "gpu_count": 0,
                "cpu_resource": "vnode-core",
                "memory_resource": "vnode-mem",
            },
            "rscgrp=b-batch,elapse=02:15:00,vnode-core=32,vnode-mem=131072Mi",
        ),
        (
            {"gpu_count": 0, "memory_resource": "node-mem"},
            "rscgrp=b-batch,elapse=02:15:00,node-mem=131072Mi",
        ),
        # A gpu=N job gets the node's share; an exclusive one the whole node.
        (
            {"memory_resource": "vnode-mem"},
            "rscgrp=b-batch,elapse=02:15:00,gpu=4",
        ),
        (
            {"exclusive": True, "memory_resource": "vnode-mem"},
            "rscgrp=b-batch,elapse=02:15:00,node=1",
        ),
        # A GPU job never asks for cores separately.
        ({"cpu_resource": "vnode-core"}, "rscgrp=b-batch,elapse=02:15:00,gpu=4"),
        ({"gpu_resource": None}, "rscgrp=b-batch,elapse=02:15:00"),
        (
            {"walltime": "168:00", "resources": ("jobenv=singularity", "x=1")},
            "rscgrp=b-batch,elapse=168:00:00,gpu=4,jobenv=singularity,x=1",
        ),
    ],
)
def test_the_profile_chooses_the_resources(tmp_path, updates, expected):
    argv = scheduler().render_submit(spec(tmp_path, **updates))
    assert argv[argv.index("-L") + 1] == expected


def test_group_and_binary(tmp_path):
    argv = scheduler(submit_binary="/usr/bin/pjsub").render_submit(
        spec(tmp_path, group="hp250000")
    )
    assert argv[0] == "/usr/bin/pjsub"
    assert argv[-3:-1] == ("-g", "hp250000")
    assert "-g" not in scheduler().render_submit(spec(tmp_path))


# --- Submission -------------------------------------------------------------


def test_submit_returns_the_job_id_alone(tmp_path):
    pjm_scheduler = scheduler(names(), names(), completed(f"{JOB}\n"))
    assert pjm_scheduler.submit(spec(tmp_path)) == JOB
    _, _, argv = pjm_scheduler._runner.calls
    assert "-X" not in argv and "-x" not in argv
    assert (job_dir(tmp_path) / "environment").exists()


@pytest.mark.parametrize(
    "output",
    [
        "",
        # Without -z jid.
        f"[INFO] PJM 0000 pjsub Job {JOB} submitted.",
        # A bulk job's subjob IDs.
        f"{JOB}[1]\n{JOB}[2]\n",
        "12345678901",
    ],
)
def test_submit_rejects_anything_but_one_job_id(tmp_path, output):
    with pytest.raises(InfrastructureError, match="exact job ID"):
        scheduler(names(), names(), completed(output)).submit(spec(tmp_path))


def test_a_rejected_submission_is_reported_once(tmp_path):
    error = "[ERR.] PJM 0058 pjsub rscgrp=b-batchx does not exist."
    pjm_scheduler = scheduler(names(), names(), completed(stderr=error, code=1))
    with pytest.raises(InfrastructureError, match="does not exist"):
        pjm_scheduler.submit(spec(tmp_path))
    assert [call[0] for call in pjm_scheduler._runner.calls] == [
        "pjstat",
        "pjstat",
        "pjsub",
    ]
    assert not (job_dir(tmp_path) / "environment").exists()


def names(*rows):
    return table(*rows, header=("JOB_ID", "JOB_NAME", "ST"))


@pytest.mark.parametrize(
    "error",
    [
        # Documented: retry the command, the system may not have accepted
        # requests for a moment.
        "[ERR.] PJM 0091 pjsub Internal error: code=1 detail=2.",
        "pjsub: timed out after 120s",
    ],
)
def test_a_lost_submit_answer_is_found_again_by_its_name(tmp_path, error):
    pjm_scheduler = scheduler(
        names(),
        names(),
        completed(stderr=error, code=1),
        names((JOB, "mt-rsi-tas", "QUE")),
        names(),
    )
    assert pjm_scheduler.submit(spec(tmp_path)) == JOB
    assert pjm_scheduler._runner.calls[3] == (
        "pjstat",
        "-v",
        "--choose",
        "jid,jnam,st",
        "--filter",
        "jnam=mt-rsi-task-codex-run-alice",
    )
    assert [call[0] for call in pjm_scheduler._runner.calls] == [
        "pjstat",
        "pjstat",
        "pjsub",
        "pjstat",
        "pjstat",
    ]


def test_reconciliation_waits_for_a_delayed_job_record_without_resubmitting(tmp_path):
    sleeps = []
    pjm_scheduler = scheduler(
        names(),
        names(),
        completed(stderr="[ERR.] PJM 0091 pjsub Internal error: code=1.", code=1),
        names(),
        names(),
        names((JOB, "mt-rsi-tas", "QUE")),
        names(),
        sleeper=sleeps.append,
    )
    assert pjm_scheduler.submit(spec(tmp_path)) == JOB
    assert [call[0] for call in pjm_scheduler._runner.calls] == [
        "pjstat",
        "pjstat",
        "pjsub",
        "pjstat",
        "pjstat",
        "pjstat",
        "pjstat",
    ]
    assert sleeps == [5.0]


def test_lost_submit_answer_recovers_the_new_finished_job_not_old_history(tmp_path):
    old = ("1006902", "mt-rsi-tas", "EXT")
    pjm_scheduler = scheduler(
        names(),
        names(old),
        completed(stderr="pjsub: timed out after 120s", code=124),
        names(),
        names(old, (JOB, "mt-rsi-tas", "EXT")),
        *ended("EXT", "0", "0", "0"),
    )
    assert pjm_scheduler.submit(spec(tmp_path)) == JOB
    assert pjm_scheduler.wait(JOB).state == "COMPLETED"
    calls = pjm_scheduler._runner.calls
    assert sum(call[0] == "pjsub" for call in calls) == 1
    assert calls[1][1:3] == ("-H", f"day={HISTORY_DAYS}")
    assert calls[4][1:3] == ("-H", f"day={HISTORY_DAYS}")


def test_an_unknown_submit_outcome_is_never_submitted_twice(tmp_path):
    pjm_scheduler = scheduler(
        names(),
        names(),
        completed(stderr="pjsub: timed out after 120s", code=124),
        *3 * [completed(stderr=DAEMON, code=1)],
    )
    with pytest.raises(InfrastructureError, match="unknown outcome"):
        pjm_scheduler.submit(spec(tmp_path))
    assert [call[0] for call in pjm_scheduler._runner.calls] == [
        "pjstat",
        "pjstat",
        "pjsub",
    ] + 3 * ["pjstat"]
    assert (job_dir(tmp_path) / "environment").exists()


def test_a_job_manager_that_is_not_running_is_waited_for(tmp_path):
    sleeps = []
    pjm_scheduler = scheduler(
        names(),
        names(),
        completed(stderr=DAEMON, code=1),
        completed(f"{JOB}\n"),
        sleeper=sleeps.append,
    )
    assert pjm_scheduler.submit(spec(tmp_path)) == JOB
    assert [call[0] for call in pjm_scheduler._runner.calls] == [
        "pjstat",
        "pjstat",
        "pjsub",
        "pjsub",
    ]
    assert sleeps == [30.0]


def test_name_collisions_match_the_whole_name_literally():
    pjm_scheduler = scheduler(names(), names(("1006901", "mt-rsi-tas", "RUN")))
    pjm_scheduler.require_name_available("mt-rsi-task.v2+x-run-alice")
    with pytest.raises(InfrastructureError, match="already active.*1006901"):
        pjm_scheduler.require_name_available("mt-rsi-task.v2+x-run-alice")
    # "." and "+" are --filter wildcards and separators.
    assert pjm_scheduler._runner.calls[0][-1] == r"jnam=mt-rsi-task\.v2\+x-run-alice"


def test_pjstat_must_list_exact_job_ids():
    pjm_scheduler = scheduler(names((f"{JOB}[1]", "mt-rsi-tas", "RUN")))
    with pytest.raises(InfrastructureError, match="invalid job ID"):
        pjm_scheduler.require_name_available("task")


def test_the_wrapper_needs_no_redirect_as_pjm_writes_the_logs(tmp_path):
    scheduler(names(), names(), completed(JOB)).submit(spec(tmp_path))
    assert "exec >>" not in (job_dir(tmp_path) / "job.sh").read_text()


# --- Status -----------------------------------------------------------------


def test_status_asks_the_current_listing_then_the_history():
    pjm_scheduler = scheduler(*ended("EXT", "0", "0", "0"))
    result = pjm_scheduler.finished(JOB)
    assert result is not None and (result.state, result.exit_code) == ("COMPLETED", 0)
    assert pjm_scheduler._runner.calls == [
        ("pjstat", "-v", "--choose", "jid,st,ec,pc,sn,sdt,elp", JOB),
        (
            "pjstat",
            "-H",
            f"day={HISTORY_DAYS}",
            "-v",
            "--choose",
            "jid,st,ec,pc,sn,sdt,elp",
            JOB,
        ),
    ]


@pytest.mark.parametrize(
    ("st", "state"),
    [
        ("ACC", "PENDING"),
        ("QUE", "PENDING"),
        ("RNA", "PENDING"),
        ("RNP", "PENDING"),
        ("HLD", "PENDING"),
        ("RUN", "RUNNING"),
        ("RNE", "RUNNING"),
        ("RNO", "COMPLETING"),
        ("SPP", "SUSPENDED"),
        ("SPD", "SUSPENDED"),
        ("RSM", "SUSPENDED"),
    ],
)
def test_an_unfinished_job_is_active(st, state):
    pjm_scheduler = scheduler(current(st))
    assert pjm_scheduler.finished(JOB) is None
    # The current listing answers; the history is not asked.
    assert len(pjm_scheduler._runner.calls) == 1
    states = []
    scheduler(current(st), *ended("EXT", "0", "0")).wait(JOB, on_state=states.append)
    assert states == [state, "COMPLETED"]


@pytest.mark.parametrize(
    ("st", "ec", "pc", "sn", "state", "code"),
    [
        ("EXT", "0", "0", "-", "COMPLETED", 0),
        ("EXT", "0", "0", "0", "COMPLETED", 0),
        # The job script's own failure.
        ("EXT", "3", "0", "-", "FAILED", 3),
        # Killed by a signal: as a shell reports it.
        ("EXT", "0", "0", "9", "FAILED", 137),
        ("EXT", "-", "0", "15", "FAILED", 143),
        # PC 11: over its elapse limit.
        ("EXT", "-", "11", "9", "TIMEOUT", 137),
        ("EXT", "-", "11", "-", "TIMEOUT", 1),
        # PC 12: over its memory; 23: the OOM killer.
        ("EXT", "-", "12", "9", "OUT_OF_MEMORY", 137),
        ("EXT", "137", "23", "-", "OUT_OF_MEMORY", 137),
        # PC 1: pjdel, while it ran or before.
        ("EXT", "-", "1", "15", "CANCELLED", 143),
        ("CCL", "-", "1", "-", "CANCELLED", 1),
        ("CCL", "-", "7", "-", "CANCELLED", 1),
        # Refused, or held for an error (never runs without an operator).
        ("RJT", "-", "2", "-", "FAILED", 1),
        ("ERR", "-", "28", "-", "FAILED", 1),
        # Node down, shell failure, prologue error.
        ("EXT", "-", "20", "-", "FAILED", 1),
        ("EXT", "-", "21", "-", "FAILED", 1),
        ("EXT", "0", "26", "-", "FAILED", 1),
    ],
)
def test_ends_map_to_slurm_states(st, ec, pc, sn, state, code):
    result = scheduler(*ended(st, ec, pc, sn)).finished(JOB)
    assert result is not None
    assert (result.job_id, result.state, result.exit_code) == (JOB, state, code)


def test_an_ended_job_still_in_the_current_listing_is_read_there():
    pjm_scheduler = scheduler(current("EXT", ec="0", pc="0", sn="0"))
    assert pjm_scheduler.finished(JOB).state == "COMPLETED"
    assert len(pjm_scheduler._runner.calls) == 1


def test_a_job_neither_listing_knows_has_ended():
    result = scheduler(NONE, NONE).finished(JOB)
    assert result is not None and (result.state, result.exit_code) == ("UNKNOWN", 1)
    # An empty answer, too.
    result = scheduler(completed(""), completed("\n")).finished(JOB)
    assert result is not None and result.state == "UNKNOWN"


def test_wait_bounds_a_job_missing_from_the_history():
    with pytest.raises(InfrastructureError, match="unknown to the scheduler"):
        scheduler(*[NONE] * 24).wait(JOB)


def test_information_lines_around_the_table_are_ignored():
    answer = completed(f"{LOADING}\n" + current("RUN").stdout + "\n")
    assert scheduler(answer).finished(JOB) is None


def test_the_listing_of_another_job_is_not_this_one():
    other = table(row("RUN", job_id="1006904"))
    result = scheduler(other, other).finished(JOB)
    assert result is not None and result.state == "UNKNOWN"


def test_a_full_v_header_with_the_chosen_columns_is_read():
    # A site whose pjstat ignores --choose: single-word values still read.
    header = ("JOB_ID", "JOB_NAME", "MD", "ST", "USER", "EC", "PC", "SN", "RSC_GRP")
    values = (JOB, "mt-rsi-tas", "NM", "EXT", "a12345", "0", "0", "0", "b-batch")
    answer = completed(" ".join(header) + "\n" + " ".join(values) + "\n")
    result = scheduler(answer).finished(JOB)
    assert result is not None and result.state == "COMPLETED"


@pytest.mark.parametrize(
    "answers",
    [
        [table(row("RUN"), row("RUN"))],
        [current("XYZ")],
        [current("EXT", ec="zero", pc="0")],
        [current("EXT", ec="0", pc="x")],
        [completed(stderr="[ERR.] PJM 0270 pjstat No execute permission.", code=1)],
        [NONE, completed(stderr="[ERR.] PJM 0204 pjstat Invalid option.", code=1)],
    ],
    ids=["ambiguous", "unknown-state", "bad-ec", "bad-pc", "refused", "no-history"],
)
def test_status_that_names_no_exact_end_is_an_error(answers):
    with pytest.raises(InfrastructureError):
        scheduler(*answers).finished(JOB)


@pytest.mark.parametrize(
    "error",
    [
        DAEMON,
        "[ERR.] PJM 0291 pjstat Internal error: code=3 detail=4.",
        "pjstat: Connection timed out",
        "pjstat: timed out after 120s",
    ],
)
def test_wait_retries_a_scheduler_that_does_not_answer(error):
    sleeps = []
    down = completed(stderr=error, code=1)
    pjm_scheduler = scheduler(
        down, down, current("RUN"), *ended("EXT", "0", "0"), sleeper=sleeps.append
    )
    assert pjm_scheduler.wait(JOB, poll_seconds=5).state == "COMPLETED"
    assert sleeps == [10.0, 20.0, 5.0]


# --- Cancellation -----------------------------------------------------------


def test_cancel_deletes_the_exact_job():
    pjm_scheduler = scheduler(completed(f"[INFO] PJM 0100 pjdel Accepted job {JOB}."))
    pjm_scheduler.cancel(JOB)
    assert pjm_scheduler._runner.calls == [("pjdel", JOB)]


@pytest.mark.parametrize(
    ("error", "answers"),
    [
        (f"[ERR.] PJM 0112 pjdel Job {JOB} does not exist.", [NONE, NONE]),
        (
            f"[ERR.] PJM 0113 pjdel Job {JOB} is not in the state that pjdel "
            "can be accepted.",
            ended("EXT", "0", "0"),
        ),
        (
            f"[ERR.] PJM 0113 pjdel Job {JOB} is not in the state that pjdel "
            "can be accepted.",
            [current("CCL", pc="1")],
        ),
    ],
    ids=["gone", "ended", "cancelled"],
)
def test_cancel_counts_an_ended_job_as_cancelled(error, answers):
    pjm_scheduler = scheduler(completed(stderr=error, code=1), *answers)
    pjm_scheduler.cancel(JOB)
    assert pjm_scheduler._runner.calls[0] == ("pjdel", JOB)


def test_cancel_reports_a_refusal_of_a_job_that_still_runs():
    refused = completed(stderr=f"[ERR.] PJM 0171 pjdel No permission: {JOB}.", code=1)
    with pytest.raises(InfrastructureError, match="No permission"):
        scheduler(refused, current("RUN")).cancel(JOB)


def test_cancel_retries_a_scheduler_that_does_not_answer(tmp_path):
    sleeps = []
    down = completed(stderr="[ERR.] PJM 0190 pjdel PJM daemon is not present.", code=1)
    pjm_scheduler = scheduler(
        names(), names(), completed(JOB), down, completed(), sleeper=sleeps.append
    )
    pjm_scheduler.submit(spec(tmp_path))
    pjm_scheduler.cancel(JOB)
    assert [call[0] for call in pjm_scheduler._runner.calls] == [
        "pjstat",
        "pjstat",
        "pjsub",
        "pjdel",
        "pjdel",
    ]
    assert sleeps == [5.0]
    assert not (job_dir(tmp_path) / "environment").exists()


@pytest.mark.parametrize(
    "job_id",
    ["all", "-x", "", "1-100", "101[1-10]", "100_1", f"{JOB} 1", "12345678901"],
)
def test_invalid_job_id_never_reaches_the_scheduler(job_id):
    pjm_scheduler = scheduler()
    for call in (pjm_scheduler.cancel, pjm_scheduler.finished, pjm_scheduler.wait):
        with pytest.raises(SetupError, match="exact PJM job ID"):
            call(job_id)
    assert pjm_scheduler._runner.calls == []


def test_the_configured_binaries_are_used(tmp_path):
    pjm_scheduler = scheduler(
        names(),
        names(),
        names(),
        completed(JOB),
        current("RUN"),
        completed(),
        submit_binary="/opt/pjm/pjsub",
        status_binary="/opt/pjm/pjstat",
        cancel_binary="/opt/pjm/pjdel",
    )
    pjm_scheduler.require_name_available("task")
    pjm_scheduler.submit(spec(tmp_path))
    pjm_scheduler.finished(JOB)
    pjm_scheduler.cancel(JOB)
    assert [call[0] for call in pjm_scheduler._runner.calls] == [
        "/opt/pjm/pjstat",
        "/opt/pjm/pjstat",
        "/opt/pjm/pjstat",
        "/opt/pjm/pjsub",
        "/opt/pjm/pjstat",
        "/opt/pjm/pjdel",
    ]


def test_rows_are_read_by_their_header():
    output = f"{LOADING}\nJOB_ID     ST  EC  PC  SN\n{JOB}    RUN -   -   -\n"
    assert pjm._rows(output, ("JOB_ID", "ST", "EC", "PC", "SN")) == [
        {"JOB_ID": JOB, "ST": "RUN", "EC": "-", "PC": "-", "SN": "-"}
    ]
    assert pjm._rows("", ("JOB_ID", "ST")) == []
    assert pjm._rows("JOB_ID     ST  EC  PC  SN\n", ("JOB_ID", "ST")) == []


# --- End time ---------------------------------------------------------------

TIMES = "JOB_ID     ST  EC  PC  SN  START_DATE      ELAPSE_TIM\n"


def timed(st, start, elapsed, ec="0", pc="0", sn="0"):
    return completed(f"{TIMES}{JOB} {st} {ec} {pc} {sn} {start} {elapsed}\n")


def at(*fields) -> float:
    return datetime(*fields).timestamp()


@pytest.mark.parametrize(
    ("start", "elapsed", "now", "expected"),
    [
        (
            "11/03 19:23:44",
            "0002:10:05",
            at(2026, 11, 4, 9),
            at(2026, 11, 3, 21, 33, 49),
        ),
        # Backfilled or started at a set time: the same start.
        ("11/03 19:23:44<", "0000:00:16", at(2026, 11, 4), at(2026, 11, 3, 19, 24)),
        ("11/03 19:23:44@", "0000:00:16", at(2026, 11, 4), at(2026, 11, 3, 19, 24)),
        # No year: the latest one not in the future.
        ("12/31 23:00:00", "0002:00:00", at(2027, 1, 1, 3), at(2027, 1, 1, 1)),
        ("02/29 10:00:00", "0001:00:00", at(2029, 1, 10), at(2028, 2, 29, 11)),
        # Without its seconds: the end of that minute.
        (
            "11/03 19:23:44",
            "0002:10",
            at(2026, 11, 4),
            at(2026, 11, 3, 21, 33, 44) + 59,
        ),
        # Never later than now.
        ("11/03 19:23:44", "0099:00:00", at(2026, 11, 4), at(2026, 11, 4)),
    ],
)
def test_an_ended_job_ended_at_its_start_plus_its_elapsed_time(
    start, elapsed, now, expected
):
    pjm_scheduler = scheduler(
        NONE, timed("EXT", start, elapsed), wall_clock=lambda: now
    )
    result = pjm_scheduler.finished(JOB)
    assert result is not None and result.state == "COMPLETED"
    assert result.ended_at == expected


@pytest.mark.parametrize(
    ("start", "elapsed"),
    [
        # Cancelled before it started: a planned start, or none.
        ("(11/04 10:00)", "0000:00:00"),
        ("(2027/11/04)", "-"),
        ("-", "-"),
    ],
)
def test_a_job_that_never_started_has_no_end_time(start, elapsed):
    answer = timed("CCL", start, elapsed, ec="-", pc="1", sn="-")
    result = scheduler(NONE, answer).finished(JOB)
    assert result is not None and result.state == "CANCELLED"
    assert result.ended_at is None


def test_an_active_job_has_no_end_time():
    row = {"ST": "RUN", "START_DATE": "11/03 19:23:44", "ELAPSE_TIM": "0000:10:00"}
    assert pjm.job_result(JOB, row).ended_at is None


def test_a_full_v_listing_with_dates_and_a_reason_is_read():
    header = (
        "JOB_ID JOB_NAME MD ST USER GROUP START_DATE ELAPSE_TIM ELAPSE_LIM "
        "RANK LST EC PC SN PRI ACCEPT RSC_GRP REASON"
    )
    values = (
        f"{JOB} mt-rsi-tas NM EXT a12345 gk77 11/03 19:23:44 0000:00:16 "
        "0002:15:00 - RUN 137 23 9 127 11/03 19:20:01 b-batch MEMORY LIMIT EXCEEDED"
    )
    answer = completed(f"{header}\n{values}\n")
    result = scheduler(answer, wall_clock=lambda: at(2026, 11, 4)).finished(JOB)
    assert result is not None and result.state == "OUT_OF_MEMORY"
    assert result.ended_at == at(2026, 11, 3, 19, 24)


# --- Held for an error ------------------------------------------------------


def test_a_job_held_for_an_error_is_deleted_and_fails(capsys):
    reason = completed(f"JOB_ID     REASON\n{JOB} JOB ENV ERROR\n")
    pjm_scheduler = scheduler(
        current("ERR", pc="28"),
        reason,
        completed(f"[INFO] PJM 0100 pjdel Accepted job {JOB}."),
    )
    result = pjm_scheduler.finished(JOB)
    assert result is not None and (result.state, result.exit_code) == ("FAILED", 1)
    assert pjm_scheduler._runner.calls[1:] == [
        ("pjstat", "-v", "--choose", "jid,ermsg", JOB),
        ("pjdel", JOB),
    ]
    assert "held for an error (JOB ENV ERROR); deleted it" in capsys.readouterr().err
    # Then the history has it cancelled.
    later = scheduler(NONE, ended("CCL", pc="1")[1]).finished(JOB)
    assert later is not None and later.state == "CANCELLED"


def test_a_held_job_that_cannot_be_deleted_is_an_error():
    refused = completed(stderr=f"[ERR.] PJM 0171 pjdel No permission: {JOB}.", code=1)
    pjm_scheduler = scheduler(current("ERR", pc="28"), NONE, refused)
    with pytest.raises(InfrastructureError, match=r"held for an error \(-\).*pjdel"):
        pjm_scheduler.finished(JOB)


def test_wait_ends_on_a_held_job_once_deleted():
    pjm_scheduler = scheduler(
        current("RUN"), current("ERR", pc="28"), NONE, completed()
    )
    assert pjm_scheduler.wait(JOB).state == "FAILED"
    assert pjm_scheduler._runner.calls[-1] == ("pjdel", JOB)


# --- Allocation launch ------------------------------------------------------


@pytest.mark.parametrize(
    ("limit", "left"),
    [
        ("6:00:00", 5 * 3600 - 75),
        # An adaptive limit is sure of its minimum only.
        ("2:00:00-6:00:00", 3600 - 75),
        ("2:00:00-", 3600 - 75),
    ],
)
def test_walltime_left_is_the_elapse_limit_less_the_time_used(limit, left):
    header = ("JOB_ID", "ST", "ELAPSE_TIM", "ELAPSE_LIM")
    pjm = scheduler(*[table((JOB, "RUN", "1:01:15", limit), header=header)] * 2)
    assert pjm.walltime_left(JOB, {}) == left
    assert pjm.walltime_left(JOB, {"PJM_ELAPSED_TIME_MODE": "fixed"}) == left
    assert pjm._runner.calls[0] == (
        "pjstat",
        "-v",
        "--choose",
        "jid,st,elp,elpl",
        JOB,
    )


def test_walltime_left_of_a_job_pjstat_does_not_list_must_be_told():
    with pytest.raises(InfrastructureError, match="cannot tell how much walltime"):
        scheduler(*([NONE] * 3)).walltime_left(JOB, {})


@pytest.mark.parametrize(("limit", "left"), [("6:00:00", None), ("2:00:00-", 3525)])
def test_an_adaptive_elapse_limit_needs_its_minimum(limit, left):
    # The job's environment says the limit is adaptive; pjstat may show only
    # the maximum, which PJM may not give.
    header = ("JOB_ID", "ST", "ELAPSE_TIM", "ELAPSE_LIM")
    pjm = scheduler(table((JOB, "RUN", "1:01:15", limit), header=header))
    environ = {"PJM_ELAPSED_TIME_MODE": "adaptive"}
    if left is None:
        with pytest.raises(SetupError, match="adaptive elapse limit"):
            pjm.walltime_left(JOB, environ)
    else:
        assert pjm.walltime_left(JOB, environ) == left


@pytest.mark.parametrize(
    ("resources", "gpus"),
    [
        # Genkai's listing: the per-node resources end in "/n".
        ("rsc015=0,gpu/n=4,jobenv=singularity,shared/n=true", 4),
        ("gpu/n=2", 2),
        # A job-wide total is not this node's share.
        ("gpu=2", None),
        ("jobenv=singularity", None),
        ("", None),
    ],
)
def test_job_gpus_are_the_jobs_gpu_resource(resources, gpus):
    pjm = scheduler()
    assert pjm.job_gpus(JOB, {"PJM_CUSTOM_RESOURCES": resources}) == gpus
    assert pjm._runner.calls == []
    assert (
        scheduler(gpu_resource=None).job_gpus(JOB, {"PJM_CUSTOM_RESOURCES": "gpu/n=2"})
        is None
    )
