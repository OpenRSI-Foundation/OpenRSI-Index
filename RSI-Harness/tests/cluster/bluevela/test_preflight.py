"""Execute generated jobs to check failures happen on the execution host."""

import subprocess

import pytest

from rsi_harness.cluster.bluevela.engine import render_engine_driver
from rsi_harness.cluster.bluevela.image import plan_image, render_build_driver
from rsi_harness.cluster.bluevela.preflight import apptainer_preflight
from tests.cluster.bluevela.test_engine import _cpu_payload, _multi_payload, _payload


@pytest.mark.parametrize("driver_kind", ["build", "cpu", "gpu", "multi"])
@pytest.mark.parametrize("problem", ["missing", "nonexecutable", "directory"])
def test_jobs_reject_unavailable_apptainer_before_any_work(
    tmp_path, driver_kind, problem
):
    binary = tmp_path / "compute only ' apptainer"
    if problem == "directory":
        binary.mkdir()
    elif problem == "nonexecutable":
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o600)
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
    assert "compute-node preflight: Apptainer is missing or not executable" in (
        completed.stderr
    )
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
