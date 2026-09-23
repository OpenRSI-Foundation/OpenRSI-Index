from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rsi_harness.cluster.schedulers.slurm import SlurmJobSpec, SlurmScheduler
from rsi_harness.errors import InfrastructureError, SetupError


def completed(stdout="", *, stderr="", code=0):
    return subprocess.CompletedProcess((), code, stdout, stderr)


class Runner:
    def __init__(self, results):
        self.results = iter(results)
        self.calls = []

    def __call__(self, argv):
        self.calls.append(argv)
        return next(self.results)


def spec(tmp_path: Path, **updates) -> SlurmJobSpec:
    script = tmp_path / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    return SlurmJobSpec(
        **{
            "name": "rsi-task-run-alice",
            "queue": "gpu",
            "group": "research",
            "cpu_slots": 8,
            "memory_mb": 65536,
            "walltime": "02:15",
            "stdout_path": tmp_path / "slurm.%j.out",
            "stderr_path": tmp_path / "slurm.%j.err",
            "script_path": script,
            "gpu_count": 4,
            "local_tmp_mb": 71680,
            **updates,
        }
    )


def test_render_single_node_resources_and_walltime(tmp_path):
    argv = SlurmScheduler().render_submit(spec(tmp_path))
    assert argv == (
        "sbatch",
        "--parsable",
        "--job-name=rsi-task-run-alice",
        "--partition=gpu",
        "--nodes=1",
        "--ntasks=1",
        "--ntasks-per-node=1",
        "--cpus-per-task=8",
        "--mem=65536M",
        "--time=135",
        "--export=ALL",
        "--account=research",
        "--gres=gpu:4",
        "--tmp=71680M",
        f"--output={tmp_path}/slurm.%j.out",
        f"--error={tmp_path}/slurm.%j.err",
        str(tmp_path / "run.sh"),
    )


def test_render_multinode_requests_per_node_resources(tmp_path):
    request = spec(
        tmp_path,
        one_host=False,
        hosts=6,
        slots_per_host=16,
        cpu_slots=96,
        memory_per_host=True,
        gpu_count=8,
        exclusive=True,
        excluded_hosts=("gpu7", "gpu9"),
        qos="normal",
        constraint="a100",
    )
    argv = SlurmScheduler().render_submit(request)
    assert set(
        (
            "--nodes=6",
            "--ntasks=6",
            "--cpus-per-task=16",
            "--gres=gpu:8",
            "--mem=65536M",
            "--exclusive",
            "--exclude=gpu7,gpu9",
            "--qos=normal",
            "--constraint=a100",
        )
    ).issubset(argv)


def test_build_job_does_not_request_gpus_or_an_implicit_account(tmp_path):
    argv = SlurmScheduler().render_submit(spec(tmp_path, gpu_count=0, group=""))
    assert not any(arg.startswith(("--gres", "--account")) for arg in argv)


@pytest.mark.parametrize("output", ["1234\n", "1234;cluster-a\n"])
def test_submit_parses_exact_job_id(tmp_path, output):
    runner = Runner([completed(output)])
    assert SlurmScheduler(runner=runner).submit(spec(tmp_path)) == "1234"


@pytest.mark.parametrize(
    "output", ["Submitted batch job 1234", "1234\n1235", "1234_1", ""]
)
def test_submit_rejects_ambiguous_job_id(tmp_path, output):
    with pytest.raises(InfrastructureError, match="exact job ID"):
        SlurmScheduler(runner=Runner([completed(output)])).submit(spec(tmp_path))


def test_submit_failure_and_missing_payload(tmp_path):
    request = spec(tmp_path)
    scheduler = SlurmScheduler(runner=Runner([completed(stderr="bad account", code=1)]))
    with pytest.raises(InfrastructureError, match="bad account"):
        scheduler.submit(request)
    request.script_path.unlink()
    with pytest.raises(SetupError, match="does not exist"):
        scheduler.submit(request)


def test_wait_uses_exact_allocation_after_accounting_delay():
    runner = Runner(
        [
            completed("123|PENDING\n"),
            completed("123|RUNNING\n"),
            completed("123|RUNNING\n"),
            completed(),
            completed(),
            completed(),
            completed("123.batch|FAILED|9:0\n123|COMPLETED|0:0\n"),
        ]
    )
    states, sleeps = [], []
    result = SlurmScheduler(runner=runner, sleeper=sleeps.append).wait(
        "123",
        poll_seconds=2,
        on_state=states.append,
    )
    assert (result.state, result.exit_code) == ("COMPLETED", 0)
    assert states == ["PENDING", "RUNNING", "COMPLETED"]
    assert sleeps == [2, 2, 2, 2]
    assert runner.calls[-1] == (
        "sacct",
        "--noheader",
        "--parsable2",
        "--allocations",
        "--jobs=123",
        "--format=JobIDRaw,State%32,ExitCode",
    )


@pytest.mark.parametrize(
    ("state", "code", "expected"),
    [
        ("FAILED", "7:0", 7),
        ("CANCELLED by 1000", "0:15", 143),
        ("TIMEOUT", "0:0", 1),
        ("OUT_OF_MEMORY", "0:9", 137),
        ("NODE_FAIL", "0:0", 1),
        ("BOOT_FAIL", "0:0", 1),
        ("PREEMPTED", "0:0", 1),
        ("DEADLINE", "0:0", 1),
        ("COMPLETED", "3:0", 3),
    ],
)
def test_wait_preserves_terminal_failures_and_signals(state, code, expected):
    runner = Runner(
        [
            completed(stderr="slurm_load_jobs error: Invalid job id specified", code=1),
            completed(f"123|{state}|{code}\n"),
        ]
    )
    result = SlurmScheduler(runner=runner).wait("123")
    assert result.state == state.split()[0]
    assert result.exit_code == expected


@pytest.mark.parametrize(
    "output",
    [
        "123|COMPLETED|bad",
        "123|UNKNOWN|0:0",
        "123|FAILED|0:0\n123|COMPLETED|0:0",
    ],
)
def test_wait_rejects_malformed_accounting(output):
    runner = Runner([completed(), completed(output)])
    with pytest.raises(InfrastructureError):
        SlurmScheduler(runner=runner).wait("123")


def test_wait_bounds_missing_accounting():
    runner = Runner([completed()] * 24)
    with pytest.raises(InfrastructureError, match="absent"):
        SlurmScheduler(runner=runner, sleeper=lambda _: None).wait("123")


def test_wait_retries_transient_errors():
    runner = Runner(
        [
            completed(stderr="Socket timed out on send/recv operation", code=1),
            completed(),
            completed("123|COMPLETED|0:0"),
        ]
    )
    sleeps = []
    assert (
        SlurmScheduler(runner=runner, sleeper=sleeps.append).wait("123").exit_code == 0
    )
    assert sleeps == [5.0]


def test_interrupt_cancels_only_the_exact_job():
    runner = Runner([completed("123|RUNNING"), completed()])

    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        SlurmScheduler(runner=runner, sleeper=interrupt).wait("123")
    assert runner.calls[-1] == ("scancel", "123")


@pytest.mark.parametrize("job_id", ["all", "123_1", "123.batch", "--user=alice"])
def test_invalid_job_id_never_reaches_scheduler(job_id):
    runner = Runner([])
    with pytest.raises(SetupError, match="exact Slurm job ID"):
        SlurmScheduler(runner=runner).cancel(job_id)
    assert runner.calls == []


def test_name_collision_is_exact():
    scheduler = SlurmScheduler(
        runner=Runner([completed("task-other\n"), completed("task\n")])
    )
    scheduler.require_name_available("task")
    with pytest.raises(InfrastructureError, match="already active"):
        scheduler.require_name_available("task")
