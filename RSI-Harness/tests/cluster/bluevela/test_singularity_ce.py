"""SingularityCE: the same container argv as Apptainer, its own variable
names, nested mount points, and the setuid combinations that are refused."""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.cluster.bluevela import runtime as runtime_module
from rsi_harness.cluster.bluevela.apptainer_environment import (
    isolated_apptainer_environment,
)
from rsi_harness.cluster.bluevela.engine import render_engine_driver
from rsi_harness.cluster.bluevela.runtime import create_nested_mount_points
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError
from tests.cluster.bluevela.test_engine import _cpu_payload
from tests.cluster.bluevela.test_runtime import _codex_runtime, _RecordingPopen


def _execs(tmp_path: Path, monkeypatch, *, singularity_ce: bool):
    """A seed and a Work exec through the real argv builder."""
    if singularity_ce:
        monkeypatch.setenv("RSI_HARNESS_SINGULARITY_CE", "1")
    else:
        monkeypatch.delenv("RSI_HARNESS_SINGULARITY_CE", raising=False)
    monkeypatch.setenv("APPTAINER_TMPDIR", "/lscratch/tmp")
    monkeypatch.setenv("SINGULARITY_CACHEDIR", "/lscratch/cache")
    monkeypatch.setenv("SINGULARITY_BINDPATH", "/oak")
    tmp_path.mkdir()
    runtime = _codex_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(_RecordingPopen, "calls", [])
    monkeypatch.setattr(subprocess, "Popen", _RecordingPopen)
    runtime._seed()
    runtime._run(
        ("true",),
        devices=(),
        network_mode="no-network",
        environment={"OPENAI_API_KEY": "sk-test", "LIST": "a,b"},
        timeout_seconds=5,
    )
    return [
        (tuple(argument.replace(str(tmp_path), "ROOT") for argument in argv), env)
        for argv, env in _RecordingPopen.calls
    ]


def test_singularity_ce_runs_the_apptainer_argv_with_its_own_variable_names(
    tmp_path, monkeypatch
):
    apptainer = _execs(tmp_path / "a", monkeypatch, singularity_ce=False)
    singularity = _execs(tmp_path / "s", monkeypatch, singularity_ce=True)

    # Byte for byte the same command lines (after the temporary root).
    assert [argv for argv, _ in singularity] == [argv for argv, _ in apptainer]
    _, apptainer_env = apptainer[-1]
    _, singularity_env = singularity[-1]
    container = {
        name.removeprefix("APPTAINERENV_"): value
        for name, value in apptainer_env.items()
        if name.startswith("APPTAINERENV_")
    }
    assert container["OPENAI_API_KEY"] == "sk-test"
    assert {
        name.removeprefix("SINGULARITYENV_"): value
        for name, value in singularity_env.items()
        if name.startswith("SINGULARITYENV_")
    } == container
    assert not any(name.startswith("APPTAINERENV_") for name in singularity_env)
    # Apptainer: only its own operator names; SingularityCE: its own, with
    # the operator's APPTAINER_TMPDIR under its name, and no site binds.
    assert apptainer_env["APPTAINER_TMPDIR"] == "/lscratch/tmp"
    assert "SINGULARITY_CACHEDIR" not in apptainer_env
    assert singularity_env["SINGULARITY_TMPDIR"] == "/lscratch/tmp"
    assert singularity_env["SINGULARITY_CACHEDIR"] == "/lscratch/cache"
    assert "APPTAINER_TMPDIR" not in singularity_env
    assert "SINGULARITY_BINDPATH" not in {**apptainer_env, **singularity_env}


def test_isolated_environment_keeps_an_operator_singularity_setting_first():
    host = {"APPTAINER_TMPDIR": "/a", "SINGULARITY_TMPDIR": "/s", "PATH": "/bin"}
    assert isolated_apptainer_environment({"X": "1"}, host, singularity_ce=True) == {
        "PATH": "/bin",
        "SINGULARITY_TMPDIR": "/s",
        "SINGULARITYENV_X": "1",
    }
    assert isolated_apptainer_environment({"X": "1"}, host) == {
        "PATH": "/bin",
        "APPTAINER_TMPDIR": "/a",
        "APPTAINERENV_X": "1",
    }


def test_nested_mount_points_are_created_in_writable_binds_only(tmp_path):
    work_tmp = tmp_path / "work-tmp"
    work_tmp.mkdir()
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt")
    tree = tmp_path / "tree"
    tree.mkdir()
    existing = work_tmp / "kept"
    existing.write_text("Work's file")
    outside = tmp_path / "outside"
    outside.mkdir()
    (work_tmp / "link").symlink_to(outside)

    create_nested_mount_points(
        [
            (work_tmp, PurePosixPath("/tmp"), False),
            (prompt, PurePosixPath("/tmp/rsi-agent-prompt.md"), True),
            (tree, PurePosixPath("/tmp/a/b"), True),
            (prompt, PurePosixPath("/tmp/kept"), True),
            (prompt, PurePosixPath("/tmp/link/escaped"), True),
            (read_only, PurePosixPath("/opt/site"), True),
            (prompt, PurePosixPath("/opt/site/prompt.md"), True),
        ]
    )

    assert (work_tmp / "rsi-agent-prompt.md").read_text() == ""
    assert (work_tmp / "a" / "b").is_dir()
    assert existing.read_text() == "Work's file"
    # No link Work made is followed, and a read-only source is never written.
    assert list(outside.iterdir()) == []
    assert list(read_only.iterdir()) == []


def _singularity(tmp_path: Path, *, suid: str) -> Path:
    binary = tmp_path / "bin" / "singularity"
    binary.parent.mkdir(parents=True)
    binary.write_text(
        "#!/bin/sh\n"
        f'[ "$1" = buildcfg ] && echo SINGULARITY_SUID_INSTALL={suid} && exit 0\n'
        "echo 'singularity-ce version 4.5.0'\n"
    )
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize(
    ("suid", "network_isolation", "user_namespace", "refused"),
    [
        ("1", "seccomp", False, "seccomp"),
        ("1", "namespace", True, "user namespace"),
        ("1", "namespace", False, None),
        ("0", "seccomp", True, None),
    ],
)
def test_the_job_refuses_a_setuid_singularity_ce_that_cannot_isolate(
    tmp_path, monkeypatch, suid, network_isolation, user_namespace, refused
):
    profile = load_cluster_profile("bluevela")
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={
                    "binary": _singularity(tmp_path, suid=suid),
                    "network_isolation": network_isolation,
                }
            )
        }
    )
    runtime = _codex_runtime(tmp_path, monkeypatch, profile=profile)
    monkeypatch.setattr(runtime_module, "_in_user_namespace", lambda: user_namespace)
    monkeypatch.setenv("RSI_HARNESS_SINGULARITY_CE", "1")
    if refused is None:
        runtime._require_runtime_mode()
    else:
        with pytest.raises(SetupError, match=f"setuid SingularityCE.*{refused}"):
            runtime._require_runtime_mode()
    # Apptainer is never asked.
    monkeypatch.setenv("RSI_HARNESS_SINGULARITY_CE", "0")
    runtime._require_runtime_mode()


def test_a_singularity_ce_job_passes_its_variables_by_its_own_names(
    tmp_path, monkeypatch
):
    payload = _cpu_payload(tmp_path)
    payload.sif_path.write_bytes(b"cpu-image")
    payload.sif_sha256_path.write_text(
        f"{hashlib.sha256(b'cpu-image').hexdigest()}  {payload.sif_path.name}\n"
    )
    result = (
        payload.run_plan.paths.logs
        / "runs"
        / payload.run_id
        / payload.run_plan.task.task_id
        / "final_result.json"
    )
    result.parent.mkdir(parents=True)
    engine = tmp_path / "fake-engine"
    engine.write_text(
        "#!/bin/sh\nset -eu\n"
        'test "$RSI_HARNESS_SINGULARITY_CE" = 1\n'
        'test -z "$SINGULARITYENV_CUDA_VISIBLE_DEVICES"\n'
        'test "${APPTAINERENV_CUDA_VISIBLE_DEVICES-unset}" = unset\n'
        # The operator's scratch settings, under SingularityCE's names.
        'test "$SINGULARITY_TMPDIR" = /scratch/tmp\n'
        'test "$SINGULARITY_CACHEDIR" = /scratch/own-cache\n'
        'test "${SINGULARITY_BIND-unset}" = unset\n'
        f"printf '{{}}' > {shlex.quote(str(result))}\n"
    )
    engine.chmod(0o700)
    singularity = _singularity(tmp_path, suid="1")
    profile = payload.profile.model_copy(
        update={
            "apptainer": payload.profile.apptainer.model_copy(
                update={"binary": singularity, "temp_root": tmp_path}
            )
        }
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.sys.executable", str(engine)
    )
    output = tmp_path / "cpu-run.sh"
    render_engine_driver(payload, profile, output)
    env = {
        **os.environ,
        "SINGULARITY_BIND": "/oak",
        "APPTAINER_TMPDIR": "/scratch/tmp",
        "APPTAINER_CACHEDIR": "/scratch/cache",
        "SINGULARITY_CACHEDIR": "/scratch/own-cache",
    }
    env.pop("SINGULARITY_TMPDIR", None)

    completed = subprocess.run(
        ["/bin/bash", str(output)], env=env, capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert result.is_file()
