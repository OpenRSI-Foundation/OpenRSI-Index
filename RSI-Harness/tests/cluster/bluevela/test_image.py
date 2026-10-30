from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from rsi_harness.cluster import podman_bundle
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
        f"printf 'podman %s\\n' \"$*\" >> {calls}\n"
        'if [[ " $* " == *" save "* ]]; then\n'
        "  while (($#)); do\n"
        '    if [[ "$1" == "-o" ]]; then printf archive > "$2"; exit 0; fi\n'
        "    shift\n"
        "  done\n"
        "fi\n",
    )
    apptainer = _executable(
        tmp_path / "apptainer",
        f"printf 'apptainer %s\\n' \"$*\" >> {calls}\n"
        'if [[ "$1" == "build" ]]; then\n'
        "  shift\n"
        '  while [[ "$1" == --* ]]; do\n'
        '    if [[ "$1" == "--mksquashfs-args" ]]; then shift 2; else shift; fi\n'
        "  done\n"
        '  printf sif > "$1"\n'
        "fi\n",
    )
    faked = _executable(
        tmp_path / "faked",
        "sleep 300 >/dev/null 2>&1 &\nprintf '12345:%s\\n' \"$!\"\n",
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
            "apptainer": base.apptainer.model_copy(update={"binary": apptainer}),
        }
    )
    plan = plan_image(context, tmp_path / "cache")
    driver = tmp_path / "build.sh"
    render_build_driver(plan, profile, context, driver, 1)
    env = {key: value for key, value in os.environ.items() if key != "XDG_RUNTIME_DIR"}

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


def _run_build(
    tmp_path: Path, build_body: str, **builder: object
) -> tuple[subprocess.CompletedProcess[str], str]:
    """Run the rendered driver with a fake Podman whose `build` runs
    ``build_body``; return the result and every recorded call."""
    context = _context(tmp_path, "FROM alpine:3.20\nRUN true\n")
    calls = tmp_path / "calls.log"
    podman = _executable(
        tmp_path / "podman",
        f"printf 'podman %s\\n' \"$*\" >> {calls}\n"
        'if [[ " $* " == *" build "* ]]; then\n'
        f"{build_body}\n"
        "fi\n"
        'if [[ " $* " == *" save "* ]]; then\n'
        '  while (($#)); do [[ "$1" == -o ]] && printf archive > "$2"; shift; done\n'
        "fi\n",
    )
    apptainer = _executable(
        tmp_path / "apptainer",
        'if [[ "$1" == build ]]; then printf sif > "${@: -2:1}"; fi\n',
    )
    base = load_cluster_profile("bluevela", {"USER": "alice"})
    profile = base.model_copy(
        update={
            "builder": base.builder.model_copy(
                update={
                    "binary": podman,
                    "temp_root": tmp_path,
                    "min_tmp_mb": 1,
                    **builder,
                }
            ),
            "apptainer": base.apptainer.model_copy(update={"binary": apptainer}),
        }
    )
    plan = plan_image(context, tmp_path / "cache")
    driver = tmp_path / "build.sh"
    render_build_driver(plan, profile, context, driver, 1)
    completed = subprocess.run(
        (str(driver),), check=False, capture_output=True, text=True
    )
    return completed, calls.read_text() if calls.exists() else ""


def test_build_driver_uses_the_site_builder_keys(tmp_path: Path) -> None:
    conmon = _executable(tmp_path / "conmon", "")
    runc = _executable(tmp_path / "runc", "")

    completed, calls = _run_build(
        tmp_path,
        "  true",
        conmon=conmon,
        runtime=runc,
        storage_driver="vfs",
        host_namespaces=("network", "uts"),
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        f"--storage-driver=vfs --storage-opt vfs.ignore_chown_errors=true "
        f"--conmon {conmon} --runtime {runc}"
    ) in calls
    assert " build --isolation rootless --format docker --network=host --uts=host " in (
        calls
    )
    runroot = calls.split("--runroot ", 1)[1].split()[0]
    assert len(runroot) <= 50 and not Path(runroot).exists()


def test_build_driver_uses_the_bundle_configuration(tmp_path: Path) -> None:
    root = tmp_path / "bundles"
    tree = podman_bundle.bundle_dir(root)
    (tree / "lib/podman").mkdir(parents=True)
    (tree / "bin").mkdir()
    conmon = _executable(tree / "lib/podman/conmon", "")
    runc = _executable(tree / "bin/runc", "")

    completed, calls = _run_build(
        tmp_path,
        '  echo "conf $CONTAINERS_CONF $CONTAINERS_STORAGE_CONF"'
        f" >> {tmp_path}/calls.log",
        podman_bundle=root,
        conmon=conmon,
        runtime=runc,
    )

    assert completed.returncode == 0, completed.stderr
    etc = tree / "etc/containers"
    policy = f"--signature-policy {etc}/policy.json "
    assert f" build --isolation rootless --format docker {policy}" in calls
    assert f" save {policy}--format docker-archive " in calls
    assert f"conf {etc}/containers.conf {etc}/storage.conf" in calls


def test_build_driver_refuses_a_missing_helper(tmp_path: Path) -> None:
    completed, calls = _run_build(tmp_path, "  true", conmon=tmp_path / "conmon")

    assert completed.returncode == 1
    assert f"builder tool is missing or not executable: {tmp_path}/conmon" in (
        completed.stderr
    )
    assert not calls


def test_build_driver_finds_a_builder_named_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    completed, calls = _run_build(tmp_path, "  true", binary=Path("podman"))

    assert completed.returncode == 0, completed.stderr
    assert " build " in calls


def test_build_driver_rebuilds_without_a_fakeroot_the_image_cannot_load(
    tmp_path: Path,
) -> None:
    faked = _executable(
        tmp_path / "faked", 'sleep 300 >/dev/null 2>&1 &\necho "1:$!"\n'
    )
    library = tmp_path / "libfakeroot.so"
    library.write_bytes(b"glibc library")

    completed, calls = _run_build(
        tmp_path,
        '  if [[ "$*" == *LD_PRELOAD* ]]; then\n'
        '    echo "Error loading shared library /usr/lib/libfakeroot.so" >&2\n'
        "    exit 1\n"
        "  fi",
        faked_binary=faked,
        fakeroot_library=library,
    )

    assert completed.returncode == 0, completed.stderr
    builds = [line for line in calls.splitlines() if " build " in line]
    assert len(builds) == 2
    assert "LD_PRELOAD" in builds[0] and "LD_PRELOAD" not in builds[1]
    assert "rebuilding without it" in completed.stderr


def test_build_driver_keeps_fakeroot_when_another_step_fails(
    tmp_path: Path,
) -> None:
    faked = _executable(
        tmp_path / "faked", 'sleep 300 >/dev/null 2>&1 &\necho "1:$!"\n'
    )
    library = tmp_path / "libfakeroot.so"
    library.write_bytes(b"glibc library")

    # Podman echoes the ENV step that sets LD_PRELOAD in every build.
    completed, calls = _run_build(
        tmp_path,
        '  echo \'STEP 2/3: ENV "LD_PRELOAD"="/usr/lib/libfakeroot.so"\'\n'
        '  echo "creating container: clone: No space left on device" >&2\n'
        "  exit 1",
        faked_binary=faked,
        fakeroot_library=library,
    )

    assert completed.returncode == 1
    assert len([line for line in calls.splitlines() if " build " in line]) == 1
    assert "rebuilding without it" not in completed.stderr
    assert "build hint: a namespace limit is 0" in completed.stderr


def test_build_driver_names_the_cause_of_an_ownership_failure(
    tmp_path: Path,
) -> None:
    completed, _calls = _run_build(
        tmp_path,
        '  echo "Error: lchown /etc/shadow: invalid argument" >&2\n  exit 125',
    )

    assert completed.returncode == 1
    assert "build hint: the Dockerfile changes file ownership" in completed.stderr


def test_default_build_driver_renders_the_existing_storage_and_namespaces(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    driver = tmp_path / "build.sh"

    render_build_driver(
        plan_image(context, tmp_path / "cache"), profile, context, driver, 1
    )

    script = driver.read_text()
    assert "--storage-driver=overlay --storage-opt ignore_chown_errors=true\n" in script
    assert "--conmon" not in script and "--runtime" not in script
    assert "--signature-policy" not in script and "CONTAINERS_CONF" not in script
    assert "--format docker \\\n" in script
    assert "for ns in net uts; do" in script
