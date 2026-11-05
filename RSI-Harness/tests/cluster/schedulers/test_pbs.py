"""PBS Pro commands and the states they report, with answers shaped like
Flow2's (PBS Pro 2025.2)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from rsi_harness.cluster.schedulers import pbs
from rsi_harness.cluster.schedulers.pbs import PBSJobSpec, PBSScheduler
from rsi_harness.errors import InfrastructureError, SetupError

UNKNOWN = "qstat: Unknown Job Id 96979.jobm"
DOWN = "qstat: cannot connect to server jobm (errno=15010)"
# What `qstat -x -f -F json` prints besides the job.
HEADER = {"timestamp": 1793701943, "pbs_version": "2025.2.2", "pbs_server": "jobm"}
TOKEN = "a1b2c3"


def completed(stdout="", *, stderr="", code=0):
    return subprocess.CompletedProcess((), code, stdout, stderr)


def status(job_state, *, job_id="96979.jobm", **attributes):
    """A `qstat -x -f -F json` answer, shaped like Flow2's for its jobs."""
    job = {
        "Job_Name": "mt-rsi-task-codex-run-alice",
        "Job_Owner": "a792144@flow2-login09",
        "job_state": job_state,
        "queue": "t2-share",
        "Variable_List": {"PBS_O_HOME": "/home/a792144", "PBS_O_SHELL": "/bin/bash"},
    }
    job.update(attributes)
    return completed(json.dumps({**HEADER, "Jobs": {job_id: job}}, indent=4))


def finished(substate, exit_status=None, **attributes):
    values = {"substate": substate, **attributes}
    if exit_status is not None:
        values["Exit_status"] = exit_status
    return status("F", **values)


class Runner:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        return next(self.results)


def scheduler(*results, **kwargs) -> PBSScheduler:
    kwargs.setdefault("sleeper", lambda _: None)
    kwargs.setdefault("token", lambda: TOKEN)
    kwargs.setdefault("environ", {})
    return PBSScheduler(runner=Runner(results), **kwargs)


def spec(tmp_path: Path, **updates) -> PBSJobSpec:
    script = tmp_path / "run.sh"
    if not script.exists():
        script.write_text("exit 0\n")
    return PBSJobSpec(
        **{
            "name": "mt-rsi-task-codex-run-alice",
            "queue": "gpu",
            "group": "",
            "cpu_slots": 24,
            "memory_mb": 65536,
            "walltime": "03:45",
            "stdout_path": tmp_path / "run" / "pbs.%j.out",
            "stderr_path": tmp_path / "run" / "pbs.%j.err",
            "script_path": script,
            "gpu_count": 2,
            "local_tmp_mb": 153600,
            **updates,
        }
    )


def job_dir(tmp_path: Path) -> Path:
    return tmp_path / "run" / f"mt-rsi-task-codex-run-alice-{TOKEN}"


# --- Rendering --------------------------------------------------------------


def test_render_single_chunk_resources_and_walltime(tmp_path):
    argv = scheduler().render_submit(spec(tmp_path))
    assert argv == (
        "qsub",
        "-q",
        "gpu",
        "-l",
        "select=1:ncpus=24:mem=65536mb:ngpus=2",
        # The shared planner's HH:MM; two fields would be MM:SS to PBS.
        "-l",
        "walltime=03:45:00",
        "-N",
        "mt-rsi-task-codex-run-alice",
        # PBS's own output of the job, in the job's directory.
        "-j",
        "oe",
        "-o",
        str(job_dir(tmp_path) / "pbs.log"),
        # A rerun would run over the same logs, its environment consumed.
        "-r",
        "n",
        # The wrapper that loads the private environment, never -V.
        str(job_dir(tmp_path) / "job.sh"),
    )


@pytest.mark.parametrize(
    ("chunk", "extra", "gpus", "select"),
    [
        # A whole-node queue that rejects explicit resources.
        ((), (), 2, "select=1"),
        # Flow2's t2-share: a queue that shares a node's GPUs.
        (("ngpus",), (), 1, "select=1:ngpus=1"),
        # A CPU queue that rejects ncpus/mem but needs NUMA domains.
        ((), ("nnumas=2",), 0, "select=1:nnumas=2"),
        (("ncpus",), ("host_type=gpu",), 1, "select=1:ncpus=24:host_type=gpu"),
        # A build job requests no GPUs, whatever the chunk names.
        (("ncpus", "mem", "ngpus"), (), 0, "select=1:ncpus=24:mem=65536mb"),
    ],
)
def test_the_profile_chooses_what_the_select_chunk_names(
    tmp_path, chunk, extra, gpus, select
):
    request = spec(tmp_path, chunk_resources=chunk, chunk_extra=extra, gpu_count=gpus)
    argv = scheduler().render_submit(request)
    assert argv[3:5] == ("-l", select)


def test_account_project_and_exclusive_placement(tmp_path):
    request = spec(
        tmp_path, group="rsi", project="p-rsi", exclusive=True, walltime="168:00"
    )
    argv = scheduler(submit_binary="/opt/pbs/bin/qsub").render_submit(request)
    assert argv[0] == "/opt/pbs/bin/qsub"
    assert "walltime=168:00:00" in argv
    flags = set(zip(argv, argv[1:], strict=False))
    assert {("-A", "rsi"), ("-P", "p-rsi"), ("-l", "place=excl")} <= flags
    plain = scheduler().render_submit(spec(tmp_path))
    assert "-A" not in plain and "-P" not in plain and "place=excl" not in plain


def test_long_names_are_shortened_stably(tmp_path):
    name = "mt-rsi-" + "a-very-long-task-name-" * 4 + "codex-run-alice"
    first = scheduler().render_submit(spec(tmp_path, name=name))
    again = scheduler().render_submit(spec(tmp_path, name=name))
    shortened = first[first.index("-N") + 1]
    assert len(shortened) <= PBSScheduler.name_limit
    assert shortened == again[again.index("-N") + 1]


def test_the_wrapper_writes_the_payloads_logs_named_by_the_job_id(tmp_path):
    request = spec(tmp_path)
    request.script_path.write_text("echo to-out; echo to-err >&2; exit 3\n")
    pbs_scheduler = scheduler(completed(), completed("97177.jobm\n"))
    pbs_scheduler.submit(request)
    wrapper = job_dir(tmp_path) / "job.sh"
    job = subprocess.run(
        ["/bin/bash", str(wrapper)],
        env={"PATH": "/usr/bin:/bin", "PBS_JOBID": "97177.jobm"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert job.returncode == 3 and job.stdout == "" and job.stderr == ""
    # Where the adapter (and job_log_detail) looks for them.
    assert (tmp_path / "run" / "pbs.97177.jobm.out").read_text() == "to-out\n"
    assert (tmp_path / "run" / "pbs.97177.jobm.err").read_text() == "to-err\n"


# --- Submission -------------------------------------------------------------


@pytest.mark.parametrize("output", ["96979.jobm\n", "96979\n", "96979.jobm.example.jp"])
def test_submit_returns_the_full_job_id(tmp_path, output):
    pbs_scheduler = scheduler(completed(), completed(output))
    assert pbs_scheduler.submit(spec(tmp_path)) == output.strip()
    _, argv = pbs_scheduler._runner.calls
    assert "-V" not in argv and "-v" not in argv


@pytest.mark.parametrize(
    "output", ["", "96979.jobm\n96980.jobm", "qsub: job 96979.jobm", "-x"]
)
def test_submit_rejects_an_ambiguous_job_id(tmp_path, output):
    with pytest.raises(InfrastructureError, match="exact job ID"):
        scheduler(completed(), completed(output)).submit(spec(tmp_path))


def test_submit_failure_is_reported_once(tmp_path):
    pbs_scheduler = scheduler(
        completed(), completed(stderr="qsub: Unknown queue", code=170)
    )
    with pytest.raises(InfrastructureError, match="Unknown queue"):
        pbs_scheduler.submit(spec(tmp_path))
    assert [call[0] for call in pbs_scheduler._runner.calls] == ["qselect", "qsub"]
    assert not (job_dir(tmp_path) / "environment").exists()


def test_a_lost_submit_answer_is_found_again_by_its_name(tmp_path, monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "a792144")
    pbs_scheduler = scheduler(
        completed(),
        completed(stderr="pbs_iff: cannot read reply from pbs_server", code=2),
        completed("97177.jobm\n"),
    )
    assert pbs_scheduler.submit(spec(tmp_path)) == "97177.jobm"
    assert pbs_scheduler._runner.calls[2] == (
        "qselect",
        "-x",
        "-u",
        "a792144",
        "-N",
        "mt-rsi-task-codex-run-alice",
    )


def test_lost_submit_answer_recovers_the_new_finished_job_not_old_history(tmp_path):
    pbs_scheduler = scheduler(
        completed("97176.jobm\n"),
        completed(stderr="qsub: timed out after 120s", code=124),
        completed("97176.jobm\n97177.jobm\n"),
        status("F", job_id="97177.jobm", substate=92, Exit_status=0),
    )
    job_id = pbs_scheduler.submit(spec(tmp_path))
    assert job_id == "97177.jobm"
    assert pbs_scheduler.wait(job_id).state == "COMPLETED"
    calls = pbs_scheduler._runner.calls
    assert [call[0] for call in calls] == ["qselect", "qsub", "qselect", "qstat"]
    assert "-x" in calls[0] and "-x" in calls[2]


@pytest.mark.parametrize(
    "error",
    [
        "pbs_iff: cannot read reply from pbs_server",
        "qsub: timed out after 120s",
    ],
)
def test_reconciliation_waits_for_a_delayed_job_record_without_resubmitting(
    tmp_path, error
):
    sleeps = []
    pbs_scheduler = scheduler(
        completed(),
        completed(stderr=error, code=2),
        completed(""),
        completed("97178.jobm\n"),
        sleeper=sleeps.append,
    )
    assert pbs_scheduler.submit(spec(tmp_path)) == "97178.jobm"
    assert [call[0] for call in pbs_scheduler._runner.calls] == [
        "qselect",
        "qsub",
        "qselect",
        "qselect",
    ]
    assert sleeps == [5.0]


def test_a_busy_server_refused_the_job_so_it_is_submitted_again(tmp_path):
    sleeps = []
    pbs_scheduler = scheduler(
        completed(),
        completed(stderr="qsub: Server is busy, try again later", code=2),
        completed("97180.jobm\n"),
        sleeper=sleeps.append,
    )
    assert pbs_scheduler.submit(spec(tmp_path)) == "97180.jobm"
    assert [call[0] for call in pbs_scheduler._runner.calls] == [
        "qselect",
        "qsub",
        "qsub",
    ]
    assert sleeps == [30.0]


def test_an_unknown_submit_outcome_is_never_submitted_twice(tmp_path):
    lost = completed(stderr="qsub: timed out after 120s", code=124)
    unreachable = completed(stderr="qselect: cannot connect to server", code=1)
    pbs_scheduler = scheduler(completed(), lost, unreachable, unreachable, unreachable)
    with pytest.raises(InfrastructureError, match="unknown outcome"):
        pbs_scheduler.submit(spec(tmp_path))
    assert [call[0] for call in pbs_scheduler._runner.calls] == [
        "qselect",
        "qsub",
    ] + 3 * ["qselect"]
    # A job that may exist still finds its environment.
    assert (job_dir(tmp_path) / "environment").exists()


def test_a_server_that_refuses_connections_is_waited_for(tmp_path):
    # Nothing reached the server: ask again, without looking for the job.
    refused = completed(
        stderr="qsub: cannot connect to server jobm (errno=111)", code=1
    )
    sleeps = []
    pbs_scheduler = scheduler(
        completed(), refused, refused, completed("97179.jobm\n"), sleeper=sleeps.append
    )
    assert pbs_scheduler.submit(spec(tmp_path)) == "97179.jobm"
    assert [call[0] for call in pbs_scheduler._runner.calls] == ["qselect"] + 3 * [
        "qsub"
    ]
    assert sleeps == [30.0, 30.0]


def test_a_server_down_too_long_fails_the_submission_without_its_file(tmp_path):
    from rsi_harness.cluster.schedulers.batch import SUBMIT_OUTAGE_SECONDS

    refused = completed(stderr="Connection refused", code=1)
    clock = iter([0.0, 0.0, SUBMIT_OUTAGE_SECONDS])
    pbs_scheduler = scheduler(
        completed(), refused, refused, sleeper=lambda _: None, clock=lambda: next(clock)
    )
    with pytest.raises(InfrastructureError, match="submission failed: Connection"):
        pbs_scheduler.submit(spec(tmp_path))
    assert not (job_dir(tmp_path) / "environment").exists()


def test_qselect_must_list_exact_job_ids(tmp_path):
    pbs_scheduler = scheduler(completed("96984.jobm junk;id\n"))
    with pytest.raises(InfrastructureError, match="invalid job ID"):
        pbs_scheduler.require_name_available("task")


def test_name_collision_lists_only_the_users_unfinished_jobs(monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    pbs_scheduler = scheduler(completed(""), completed("96984.jobm\n"))
    pbs_scheduler.require_name_available("task")
    with pytest.raises(InfrastructureError, match="already active.*96984.jobm"):
        pbs_scheduler.require_name_available("task")
    assert pbs_scheduler._runner.calls[0] == ("qselect", "-u", "alice", "-N", "task")
    sleeps = []
    down = completed(stderr="qselect: cannot connect to server", code=1)
    with pytest.raises(InfrastructureError, match="cannot list active PBS jobs"):
        scheduler(down, down, down, sleeper=sleeps.append).require_name_available(
            "task"
        )
    # A server that does not answer is asked three times.
    assert sleeps == [5.0, 10.0]


# --- Status -----------------------------------------------------------------


def test_wait_follows_the_exact_job_into_its_history():
    sleeps, states = [], []
    pbs_scheduler = scheduler(
        status("Q"),
        status("R"),
        status("R"),
        status("E"),
        finished(92, 0, obittime="Tue Nov  3 19:23:44 2026"),
        sleeper=sleeps.append,
    )
    result = pbs_scheduler.wait("96979.jobm", poll_seconds=2, on_state=states.append)
    assert (result.state, result.exit_code) == ("COMPLETED", 0)
    assert result.ended_at == datetime(2026, 11, 3, 19, 23, 44).timestamp()
    assert states == ["PENDING", "RUNNING", "COMPLETING", "COMPLETED"]
    assert sleeps == [2, 2, 2, 2]
    assert pbs_scheduler._runner.calls[0] == (
        "qstat",
        "-x",
        "-f",
        "-F",
        "json",
        "96979.jobm",
    )


@pytest.mark.parametrize(
    ("substate", "exit_status", "state", "code"),
    [
        # Flow2 96979: "... and finished".
        (92, 0, "COMPLETED", 0),
        # 96562: "... and failed".
        (93, 1, "FAILED", 1),
        # 95655: "... and exceeded resource walltime" (JSON gives a string).
        (93, "-29", "TIMEOUT", 1),
        (93, -29, "TIMEOUT", 1),
        (93, "-27", "OUT_OF_MEMORY", 1),
        (93, "-26", "OUT_OF_MEMORY", 1),
        # PBS could not start or run the job.
        (93, "-1", "FAILED", 1),
        # 95983: "... and terminated by a792144@flow2-login10" (qdel).
        (91, 143, "CANCELLED", 143),
        (91, 0, "CANCELLED", 1),
        # 94229: deleted while queued, so it never ran.
        (91, None, "CANCELLED", 1),
        (93, None, "FAILED", 1),
        # The job shell itself killed by SIGKILL, then by SIGTERM.
        (93, 265, "FAILED", 137),
        (93, 271, "FAILED", 143),
    ],
)
def test_finished_maps_pbs_ends_to_slurm_states(substate, exit_status, state, code):
    result = scheduler(finished(substate, exit_status)).finished("96979.jobm")
    assert result is not None
    assert (result.state, result.exit_code, result.ended_at) == (state, code, None)


def test_an_array_subjob_end_is_an_end():
    result = scheduler(status("X", substate=92, Exit_status=0)).finished("96979.jobm")
    assert result is not None and result.state == "COMPLETED"


@pytest.mark.parametrize("letter", ["Q", "H", "W", "T", "R", "B", "E", "S", "U"])
def test_finished_is_none_while_pbs_queues_or_runs_the_job(letter):
    assert scheduler(status(letter)).finished("96979.jobm") is None


def test_a_short_job_id_names_the_servers_full_one():
    result = scheduler(finished(92, 0)).finished("96979")
    assert result is not None and result.state == "COMPLETED"


def test_a_job_the_server_no_longer_knows_has_ended():
    answer = completed(json.dumps(HEADER), stderr=UNKNOWN, code=153)
    result = scheduler(answer).finished("96979.jobm")
    assert result is not None
    assert (result.state, result.exit_code, result.ended_at) == ("UNKNOWN", 1, None)


def test_wait_bounds_a_job_missing_from_the_history():
    answers = [completed(json.dumps(HEADER), stderr=UNKNOWN, code=153)] * 12
    with pytest.raises(InfrastructureError, match="unknown to the scheduler"):
        scheduler(*answers).wait("96979.jobm")


def test_wait_retries_transient_errors():
    sleeps = []
    pbs_scheduler = scheduler(
        completed(stderr="pbs_iff: cannot connect to host", code=2),
        completed(stderr=DOWN, code=15010),
        finished(92, 0),
        sleeper=sleeps.append,
    )
    assert pbs_scheduler.wait("96979.jobm").state == "COMPLETED"
    # A server that does not answer is polled less and less often.
    assert sleeps == [10.0, 20.0]


@pytest.mark.parametrize(
    "answer",
    [
        status("M"),
        status("F", job_id="96980.jobm", substate=92, Exit_status=0),
        status("F", job_id="969790.jobm", substate=92, Exit_status=0),
        finished(92, "zero"),
        finished("ninety-two", 0),
        completed(json.dumps({**HEADER, "Jobs": {}})),
        completed(json.dumps(HEADER)),
        completed(stderr="qstat: illegal option", code=2),
    ],
    ids=[
        "moved",
        "other-job",
        "longer-id",
        "bad-exit",
        "bad-substate",
        "no-job",
        "no-jobs",
        "error",
    ],
)
def test_status_that_names_no_exact_end_is_an_error(answer):
    with pytest.raises(InfrastructureError):
        scheduler(answer).finished("96979.jobm")


PLAIN = (
    "Job Id: 96979.jobm\n"
    "    Job_Name = mt-rsi-task-codex-run-alice\n"
    "    job_state = F\n"
    "    comment = Job run at Tue Nov 03 at 19:20 on (t2049:ncpus=144:\n"
    "\tmem=905969664kb:ngpus=4) and finished\n"
    "    obittime = Tue Nov  3 19:23:44 2026\n"
    "    substate = 92\n"
    "    Variable_List = PBS_O_HOME=/home/alice,\n"
    "\tPBS_O_SHELL=/bin/bash, job_state = R\n"
    "    Exit_status = 0\n"
)


def test_invalid_json_falls_back_to_the_plain_listing():
    # An exported environment value PBS escaped badly breaks the JSON.
    broken = completed('{"Jobs": {"96979.jobm": {"Variable_List": {"PS1": "\\[x"}}}}')
    pbs_scheduler = scheduler(broken, completed(PLAIN))
    result = pbs_scheduler.finished("96979.jobm")
    assert result is not None
    assert (result.state, result.exit_code) == ("COMPLETED", 0)
    assert result.ended_at == datetime(2026, 11, 3, 19, 23, 44).timestamp()
    assert pbs_scheduler._runner.calls[1] == ("qstat", "-x", "-f", "96979.jobm")
    other = completed("Job Id: 96980.jobm\n    job_state = F\n")
    with pytest.raises(InfrastructureError, match="exact job"):
        scheduler(broken, other).finished("96979.jobm")
    gone = completed(stderr=UNKNOWN, code=153)
    assert scheduler(broken, gone).finished("96979.jobm").state == "UNKNOWN"


def test_json_with_control_characters_in_values_still_reads():
    # PBS writes a variable's newline raw into the JSON string.
    raw = status("R").stdout.replace('"/bin/bash"', '"/bin/\nbash"')
    assert scheduler(completed(raw)).finished("96979.jobm") is None


def test_plain_qstat_joins_continuations_and_ignores_other_attributes():
    attributes = pbs.text_attributes("96979.jobm", PLAIN)
    # The "job_state = R" inside Variable_List's continuation is not the state.
    assert attributes == {
        "job_state": "F",
        "obittime": "Tue Nov  3 19:23:44 2026",
        "substate": "92",
        "Exit_status": "0",
    }
    # qstat wraps a long value with a newline and a tab, nothing else.
    wrapped = "Job Id: 96979.jobm\n    Exit_status = 2\n\t71\n"
    assert pbs.text_attributes("96979.jobm", wrapped)["Exit_status"] == "271"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Job Id: 96979.jobm\n    job_state = F\nJob Id: 96980.jobm\n",
        "Job Id: 96979.jobm\n    job_state = F\n    job_state = R\n",
    ],
    ids=["empty", "two-jobs", "two-states"],
)
def test_plain_qstat_must_name_one_job_once(text):
    with pytest.raises(InfrastructureError):
        pbs.text_attributes("96979.jobm", text)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("Tue Nov  3 19:23:44 2026", datetime(2026, 11, 3, 19, 23, 44).timestamp()),
        ("Wed Nov 11 01:02:03 2026", datetime(2026, 11, 11, 1, 2, 3).timestamp()),
        (None, None),
        ("yesterday", None),
    ],
)
def test_pbs_times_are_read_as_local_time(value, expected):
    assert pbs.pbs_time(value) == expected


# --- Cancellation -----------------------------------------------------------


def test_interrupt_cancels_only_the_exact_job():
    def interrupt(_):
        raise KeyboardInterrupt

    pbs_scheduler = scheduler(status("R"), completed(), sleeper=interrupt)
    with pytest.raises(KeyboardInterrupt):
        pbs_scheduler.wait("96979.jobm")
    assert pbs_scheduler._runner.calls[-1] == ("qdel", "96979.jobm")


@pytest.mark.parametrize(
    "answer",
    [
        completed(),
        completed(stderr="qdel: Unknown Job Id 96979.jobm", code=153),
        completed(stderr="qdel: Job has finished 96979.jobm", code=35),
    ],
)
def test_cancel_counts_an_ended_job_as_cancelled(answer):
    pbs_scheduler = scheduler(answer)
    pbs_scheduler.cancel("96979.jobm")
    assert pbs_scheduler._runner.calls == [("qdel", "96979.jobm")]


def test_cancel_retries_a_server_that_does_not_answer():
    sleeps = []
    pbs_scheduler = scheduler(
        completed(stderr="qdel: cannot connect to server jobm", code=15010),
        completed(),
        sleeper=sleeps.append,
    )
    pbs_scheduler.cancel("96979.jobm")
    assert len(pbs_scheduler._runner.calls) == 2 and sleeps == [5.0]


def test_cancel_reports_a_refusal():
    pbs_scheduler = scheduler(completed(stderr="qdel: Unauthorized Request", code=159))
    with pytest.raises(InfrastructureError, match="Unauthorized"):
        pbs_scheduler.cancel("96979.jobm")


@pytest.mark.parametrize(
    "job_id", ["all", "-x", "96979.jobm extra", "", "96979;id", "jobm"]
)
def test_invalid_job_id_never_reaches_the_scheduler(job_id):
    pbs_scheduler = scheduler()
    for call in (pbs_scheduler.cancel, pbs_scheduler.finished, pbs_scheduler.wait):
        with pytest.raises(SetupError, match="exact PBS job ID"):
            call(job_id)
    assert pbs_scheduler._runner.calls == []


def test_the_configured_binaries_are_used(tmp_path, monkeypatch):
    monkeypatch.setattr("getpass.getuser", lambda: "alice")
    pbs_scheduler = scheduler(
        completed(""),
        completed(""),
        completed("1.s\n"),
        status("R", job_id="1.s"),
        completed(),
        submit_binary="/pbs/qsub",
        status_binary="/pbs/qstat",
        cancel_binary="/pbs/qdel",
        select_binary="/pbs/qselect",
    )
    pbs_scheduler.require_name_available("task")
    pbs_scheduler.submit(spec(tmp_path))
    pbs_scheduler.finished("1.s")
    pbs_scheduler.cancel("1.s")
    assert [call[0] for call in pbs_scheduler._runner.calls] == [
        "/pbs/qselect",
        "/pbs/qselect",
        "/pbs/qsub",
        "/pbs/qstat",
        "/pbs/qdel",
    ]


def test_an_end_the_server_heard_of_late_is_dated_by_the_walltime_used():
    from rsi_harness.cluster.schedulers.pbs import job_result, pbs_time, walltime_end

    attributes = {
        "job_state": "F",
        "substate": 92,
        "Exit_status": 0,
        "stime": "Wed Nov  4 21:34:39 2026",
        "obittime": "Wed Nov  4 21:37:11 2026",
        "resources_used": {"walltime": "00:01:30"},
    }
    # The server's record stands; the estimate is kept for an outage.
    assert job_result("7.server", attributes).ended_at == pbs_time(
        "Wed Nov  4 21:37:11 2026"
    )
    assert walltime_end(attributes) == pbs_time("Wed Nov  4 21:36:09 2026")
    plain = {**attributes, "resources_used.walltime": "00:01:30"}
    del plain["resources_used"]
    assert walltime_end(plain) == walltime_end(attributes)
    del plain["stime"]
    assert walltime_end(plain) == pbs_time("Wed Nov  4 21:37:11 2026")


def _finished_record(stime, obittime, walltime):
    record = {
        "Jobs": {
            "7.server": {
                "job_state": "F",
                "substate": 92,
                "Exit_status": 0,
                "stime": stime,
                "obittime": obittime,
                "resources_used": {"walltime": walltime},
            }
        }
    }
    return completed(json.dumps(record))


def _running_record():
    return completed(json.dumps({"Jobs": {"7.server": {"job_state": "R"}}}))


def test_wait_dates_an_end_heard_of_after_an_outage_by_the_walltime_used():
    from rsi_harness.cluster.schedulers.pbs import pbs_time

    down = completed(stderr="qstat: cannot connect to server", code=1)
    clock = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    pbs_scheduler = scheduler(
        _running_record(),
        down,
        _finished_record(
            "Wed Nov  4 21:34:39 2026", "Wed Nov  4 21:37:11 2026", "00:01:30"
        ),
        clock=lambda: next(clock),
        wall_clock=lambda: pbs_time("Wed Nov  4 21:35:00 2026"),
    )
    result = pbs_scheduler.wait("7.server")
    assert result.ended_at == pbs_time("Wed Nov  4 21:36:09 2026")


def test_a_suspended_jobs_end_is_never_dated_before_it_was_last_seen_running():
    from rsi_harness.cluster.schedulers.pbs import pbs_time

    down = completed(stderr="qstat: cannot connect to server", code=1)
    # Suspended for most of its run: walltime used is only 3 s.
    clock = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    pbs_scheduler = scheduler(
        _running_record(),
        down,
        _finished_record(
            "Wed Nov  4 22:25:01 2026", "Wed Nov  4 22:25:26 2026", "00:00:03"
        ),
        clock=lambda: next(clock),
        wall_clock=lambda: pbs_time("Wed Nov  4 22:25:20 2026"),
    )
    result = pbs_scheduler.wait("7.server")
    assert result.ended_at == pbs_time("Wed Nov  4 22:25:20 2026")


def test_without_an_outage_the_servers_end_record_stands():
    from rsi_harness.cluster.schedulers.pbs import pbs_time

    pbs_scheduler = scheduler(
        _running_record(),
        _finished_record(
            "Wed Nov  4 22:25:01 2026", "Wed Nov  4 22:25:26 2026", "00:00:03"
        ),
    )
    assert pbs_scheduler.wait("7.server").ended_at == pbs_time(
        "Wed Nov  4 22:25:26 2026"
    )
