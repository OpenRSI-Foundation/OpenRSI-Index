"""Execute generated jobs to check failures happen on the execution host."""

import shlex
import subprocess
from pathlib import Path

import pytest

from rsi_harness.cluster.bluevela.engine import render_engine_driver
from rsi_harness.cluster.bluevela.image import plan_image, render_build_driver
from rsi_harness.cluster.bluevela.preflight import (
    apptainer_preflight,
    is_singularity_ce,
    refuse_singularity_ce,
)
from rsi_harness.errors import SetupError
from tests.cluster.bluevela.test_engine import _cpu_payload, _multi_payload, _payload


@pytest.mark.parametrize("driver_kind", ["build", "cpu", "gpu", "multi"])
@pytest.mark.parametrize(
    "problem", ["missing", "nonexecutable", "directory", "singularity-ce"]
)
def test_jobs_reject_unavailable_apptainer_before_any_work(
    tmp_path, driver_kind, problem
):
    binary = tmp_path / "compute only ' apptainer"
    if problem == "directory":
        binary.mkdir()
    elif problem == "nonexecutable":
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o600)
    elif problem == "singularity-ce":
        _runtime(binary, answer="singularity-ce version 4.1.3")
    factory = {"cpu": _cpu_payload, "multi": _multi_payload}.get(driver_kind, _payload)
    payload = factory(tmp_path)
    profile = payload.profile.model_copy(
        update={
            "apptainer": payload.profile.apptainer.model_copy(update={"binary": binary})
        }
    )
    driver = tmp_path / "job.sh"
    if driver_kind == "build":
        context = tmp_path / "context"
        context.mkdir()
        (context / "Dockerfile").write_text("FROM scratch\n")
        plan = plan_image(context, tmp_path / "cache")
        render_build_driver(plan, profile, context, driver, 1)
    else:
        render_engine_driver(payload, profile, driver)
    # No valid scratch, SIF, scheduler allocation, builder, or Engine is needed:
    # the explicit executable check must fail before touching any of them.
    completed = subprocess.run(
        ["/bin/bash", str(driver)], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 1
    assert (
        "SingularityCE is not supported"
        if problem == "singularity-ce"
        else "compute-node preflight: Apptainer is missing or not executable"
    ) in completed.stderr
    assert str(binary) in completed.stderr
    assert not (tmp_path / "cache").exists()


def test_preflight_is_evaluated_at_execution_not_render_time(tmp_path):
    binary = tmp_path / "compute only ' apptainer"
    driver = tmp_path / "job.sh"
    driver.write_text("set -eu\n" + apptainer_preflight(binary))
    assert not binary.exists()
    # Model the file becoming visible only in the execution environment.
    executable = tmp_path / "installed-apptainer"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    binary.symlink_to(executable)
    completed = subprocess.run(
        ["/bin/bash", str(driver)], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr


def _runtime(path: Path, *, answer: str | None, link_to: str | None = None) -> Path:
    """A fake runtime at ``path`` that prints ``answer`` for --version, or
    cannot answer (exit 1) when it is None; ``link_to`` makes ``path`` a link
    to a sibling of that name, as Apptainer installs ``singularity``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if link_to is None else path.parent / link_to
    reply = "exit 1" if answer is None else f"printf '%s\\n' {shlex.quote(answer)}"
    target.write_text(f'#!/bin/sh\n[ "$1" = --version ] && {reply}\nexit 0\n')
    target.chmod(0o755)
    if link_to is not None:
        path.symlink_to(link_to)
    return path


@pytest.mark.parametrize(
    ("name", "answer", "link_to", "singularity_ce"),
    [
        ("apptainer", "apptainer version 1.5.5", None, False),
        # An Apptainer installed or wrapped under the singularity name.
        ("singularity", "apptainer version 1.3.4-1.el9", None, False),
        ("singularity", "singularity-ce version 4.1.3", None, True),
        ("singularity", "SingularityCE version 3.9.0", None, True),
        # Leading whitespace is trimmed on both sides.
        ("singularity", "\n  singularity-ce version 4.1.3", None, True),
        # Cannot answer: judged by a resolved name of exactly singularity.
        ("singularity", None, None, True),
        ("singularity", None, "apptainer", False),
        ("singularity", None, "singularity.real", False),
        ("singularity-wrapper", None, None, False),
        # Answers, but nothing recognisable: Apptainer.
        ("singularity", "", None, False),
    ],
)
def test_submit_host_and_job_tell_singularity_ce_alike(
    tmp_path, name, answer, link_to, singularity_ce
):
    binary = _runtime(tmp_path / "bin" / name, answer=answer, link_to=link_to)
    assert is_singularity_ce(binary) is singularity_ce
    driver = tmp_path / "job.sh"
    driver.write_text("set -euo pipefail\n" + apptainer_preflight(binary))
    completed = subprocess.run(
        ["/bin/bash", str(driver)], capture_output=True, text=True, timeout=10
    )
    if singularity_ce:
        with pytest.raises(SetupError, match="SingularityCE is not supported"):
            refuse_singularity_ce(binary)
        assert completed.returncode == 1
        assert "SingularityCE is not supported" in completed.stderr
        assert "install-unprivileged.sh" in completed.stderr
        assert "apptainer.binary" in completed.stderr
    else:
        refuse_singularity_ce(binary)
        assert completed.returncode == 0, completed.stderr


def test_a_runtime_that_answers_in_bytes_that_are_not_utf8_is_apptainer(tmp_path):
    binary = tmp_path / "bin" / "apptainer"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nprintf '\\377\\376singularity\\n'\n")
    binary.chmod(0o755)
    driver = tmp_path / "job.sh"
    driver.write_text("set -euo pipefail\n" + apptainer_preflight(binary))

    assert is_singularity_ce(binary) is False
    refuse_singularity_ce(binary)
    completed = subprocess.run(
        ["/bin/bash", str(driver)], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, completed.stderr


def test_a_runtime_the_submit_host_does_not_see_is_left_to_the_job(tmp_path):
    refuse_singularity_ce(tmp_path / "compute-only" / "singularity")
