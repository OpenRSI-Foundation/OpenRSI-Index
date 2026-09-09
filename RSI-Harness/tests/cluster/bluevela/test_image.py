from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from rsi_harness.cluster.bluevela.image import (
    plan_image,
    render_build_driver,
    validate_cached_image,
)
from rsi_harness.cluster.config import load_cluster_profile


def _context(root: Path, content: str = "FROM scratch\n") -> Path:
    context = root / "context"
    context.mkdir()
    (context / "Dockerfile").write_text(content)
    return context


def _executable(path: Path, body: str) -> Path:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
    path.chmod(0o700)
    return path


def test_image_key_changes_when_docker_context_changes(tmp_path: Path) -> None:
    context = _context(tmp_path)
    first = plan_image(context, tmp_path / "cache")

    (context / "Dockerfile").write_text("FROM scratch\nLABEL revision=2\n")
    second = plan_image(context, tmp_path / "cache")

    assert first.cache_key != second.cache_key
    assert first.sif_path != second.sif_path


def test_cache_hit_requires_matching_nonempty_sif_and_sidecar(tmp_path: Path) -> None:
    context = _context(tmp_path)
    plan = plan_image(context, tmp_path / "cache")
    plan.sif_path.parent.mkdir(parents=True)
    plan.sif_path.write_bytes(b"sif")
    digest = hashlib.sha256(b"sif").hexdigest()
    plan.sha256_path.write_text(f"{digest}  {plan.sif_path.name}\n")

    assert validate_cached_image(plan) is True
    assert plan_image(context, tmp_path / "cache").cache_hit is True

    plan.sif_path.write_bytes(b"corrupt")
    assert validate_cached_image(plan) is False
    assert plan_image(context, tmp_path / "cache").cache_hit is False


def test_build_driver_executes_archive_conversion_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, "FROM ubuntu:22.04\nRUN apt-get update\n")
    calls = tmp_path / "calls.log"
    podman = _executable(
        tmp_path / "podman",
        '[[ -n "${XDG_RUNTIME_DIR:-}" && -d "$XDG_RUNTIME_DIR" ]] || exit 42\n'
        '[[ -f "${CONTAINERS_REGISTRIES_CONF:-}" ]] || exit 43\n'
        'grep -q \'short-name-mode = "disabled"\' "$CONTAINERS_REGISTRIES_CONF"\n'
        f'printf \'podman %s\\n\' "$*" >> {calls}\n'
        'if [[ " $* " == *" save "* ]]; then\n'
        '  while (($#)); do\n'
        '    if [[ "$1" == "-o" ]]; then printf archive > "$2"; exit 0; fi\n'
        '    shift\n'
        '  done\n'
        "fi\n",
    )
    apptainer = _executable(
        tmp_path / "apptainer",
        f'printf \'apptainer %s\\n\' "$*" >> {calls}\n'
        'if [[ "$1" == "build" ]]; then\n'
        '  shift\n'
        '  while [[ "$1" == --* ]]; do\n'
        '    if [[ "$1" == "--mksquashfs-args" ]]; then shift 2; else shift; fi\n'
        '  done\n'
        '  printf sif > "$1"\n'
        "fi\n",
    )
    faked = _executable(
        tmp_path / "faked",
        "sleep 300 >/dev/null 2>&1 &\n"
        "printf '12345:%s\\n' \"$!\"\n",
    )
    fakeroot_library = tmp_path / "libfakeroot-sysv.so"
    fakeroot_library.write_bytes(b"fake library")
    base = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = base.model_copy(
        update={
            "builder": base.builder.model_copy(
                update={
                    "binary": podman,
                    "temp_root": tmp_path,
                    "min_tmp_mb": 1,
                    "faked_binary": faked,
                    "fakeroot_library": fakeroot_library,
                }
            ),
            "apptainer": base.apptainer.model_copy(
                update={"binary": apptainer}
            ),
        }
    )
    plan = plan_image(context, tmp_path / "cache")
    driver = tmp_path / "build.sh"
    render_build_driver(plan, profile, context, driver, 1)
    env = {
        key: value
        for key, value in os.environ.items()
        if key != "XDG_RUNTIME_DIR"
    }

    completed = subprocess.run(
        (str(driver),),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    assert validate_cached_image(plan) is True
    assert "--storage-driver=overlay" in calls.read_text()
    assert "--storage-opt ignore_chown_errors=true" in calls.read_text()
    assert "--cgroup-manager=cgroupfs" in calls.read_text()
    assert "--events-backend=file" in calls.read_text()
    assert " build --isolation rootless " in calls.read_text()
    assert " save " in calls.read_text()
    assert "docker-archive://" in calls.read_text()
    assert "--mksquashfs-args -processors 64" in calls.read_text()
    assert "system reset --force" in calls.read_text()
    assert "/etc/apt/apt.conf.d/99-rsi-rootless.conf:ro" in calls.read_text()
    assert "--ipc host" in calls.read_text()
    assert "libfakeroot-sysv.so:/usr/lib/libfakeroot.so:ro" in calls.read_text()
    assert "--unsetenv LD_PRELOAD" in calls.read_text()
    assert "available_tmp_kb" in driver.read_text()
    assert "required_tmp_kb=$((1 * 1024))" in driver.read_text()
    assert not list(plan.sif_path.parent.glob("*.partial"))
