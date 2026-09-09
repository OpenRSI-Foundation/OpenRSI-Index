from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.errors import SetupError


def test_packaged_bluevela_profile_contains_cluster_policy() -> None:
    profile = load_cluster_profile("bluevela", {"USER": "alice"})

    assert profile.name == "bluevela"
    assert profile.owner == "alice"
    assert profile.scheduler.kind == "lsf"
    assert profile.scheduler.queue == "normal"
    assert profile.scheduler.exclusive is True
    assert profile.scheduler.group == "rsi"
    assert profile.scheduler.remote_binary == "blaunch"
    assert profile.scheduler.remote_host_flag == "-z"
    assert profile.storage.run_root.is_absolute()
    assert profile.builder.cpu_slots == 64
    assert profile.builder.memory_mb == 524288
    assert profile.builder.min_tmp_mb == 71680
    assert profile.builder.rootless_apt_sandbox is True
    assert profile.builder.faked_binary is not None
    assert profile.builder.fakeroot_library is not None
    assert profile.apptainer.temp_root.as_posix() == "/tmp"
    assert profile.apptainer.environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.work_environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.judge_environment["RSI_SHARED_DATA_ROOT"] == "/rsi-data"
    assert profile.apptainer.extra_binds == ()
    assert profile.apptainer.mount_policy == "scoped"
    assert profile.apptainer.work_binds
    assert profile.apptainer.judge_binds
    assert not any(
        "paloma" in str(binding.source)
        for binding in profile.apptainer.work_binds
    )
    assert any(
        "paloma" in str(binding.source)
        for binding in profile.apptainer.judge_binds
    )
    assert all(binding.read_only for binding in profile.apptainer.work_binds)
    assert all(binding.read_only for binding in profile.apptainer.judge_binds)
    assert all(
        binding.source.is_relative_to("/srv/rsi")
        for binding in (
            *profile.apptainer.work_binds, *profile.apptainer.judge_binds
        )
    )
    assert profile.resources.gpus_per_node == 8


def test_profile_runtime_environment_is_expanded_and_portable(tmp_path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            'RSI_SHARED_DATA_ROOT = "/rsi-data"',
            "RSI_SHARED_DATA_ROOT = \"${DATA_ROOT}\"",
            1,
        )
    )

    profile = load_cluster_profile(
        profile_path,
        {"USER": "alice", "DATA_ROOT": "/cluster/shared/rsi-data"},
    )

    assert profile.apptainer.environment == {
        "RSI_SHARED_DATA_ROOT": "/cluster/shared/rsi-data"
    }


def test_profile_expands_only_explicit_environment_references(tmp_path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            "/srv/rsi/harness",
            "${RUN_BASE}/rsi-harness",
        )
    )

    profile = load_cluster_profile(profile_path, {"USER": "alice", "RUN_BASE": "/x"})

    assert profile.storage.run_root.as_posix() == "/x/rsi-harness/runs"


def test_profile_rejects_unknown_environment_reference(tmp_path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            "/srv/rsi/harness",
            "${MISSING}/runs",
        )
    )

    with pytest.raises(SetupError, match="MISSING"):
        load_cluster_profile(profile_path, {"USER": "alice"})


def test_profile_rejects_unknown_cluster_name() -> None:
    with pytest.raises(SetupError, match="unknown cluster"):
        load_cluster_profile("not-a-cluster", {"USER": "alice"})


def test_profile_rejects_legacy_shared_root_mounts(tmp_path: Path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            "extra_binds = []", 'extra_binds = ["/srv/shared-project"]'
        )
    )

    with pytest.raises(SetupError, match="use work_binds and judge_binds"):
        load_cluster_profile(profile_path, {"USER": "alice"})


def test_existing_profile_retains_legacy_mount_policy(tmp_path: Path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath(
        "profile.toml"
    )
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text()
        .replace('mount_policy = "scoped"\n', "")
        .replace("extra_binds = []", 'extra_binds = ["/srv/shared-project"]')
    )

    profile = load_cluster_profile(profile_path, {"USER": "alice"})

    assert profile.apptainer.mount_policy == "legacy"
    assert profile.apptainer.extra_binds == (Path("/srv/shared-project"),)
