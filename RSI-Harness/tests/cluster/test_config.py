from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from rsi_harness.cluster import podman_bundle
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
        "paloma" in str(binding.source) for binding in profile.apptainer.work_binds
    )
    assert any(
        "paloma" in str(binding.source) for binding in profile.apptainer.judge_binds
    )
    assert all(binding.read_only for binding in profile.apptainer.work_binds)
    assert all(binding.read_only for binding in profile.apptainer.judge_binds)
    assert all(
        binding.source.is_relative_to("/srv/rsi")
        for binding in (*profile.apptainer.work_binds, *profile.apptainer.judge_binds)
    )
    assert profile.resources.gpus_per_node == 8


@pytest.mark.parametrize("binary", ["apptainer", "bin/apptainer", "./apptainer"])
def test_apptainer_binary_must_be_absolute_before_scheduler_use(tmp_path, binary):
    packaged = resources.files("rsi_harness.cluster.slurm").joinpath("profile.toml")
    path = tmp_path / "site.toml"
    profile_text = packaged.read_text()
    import re

    profile_text = re.sub(
        r'(?m)^binary = "[^"]*apptainer"$', f'binary = "{binary}"', profile_text
    )
    path.write_text(profile_text)
    with pytest.raises(SetupError, match="absolute compute-node path"):
        load_cluster_profile(path, {"USER": "alice"})


def test_apptainer_absolute_path_need_not_exist_on_submit_host(tmp_path):
    from rsi_harness.cluster.config import ApptainerProfile

    profile = load_cluster_profile("slurm", {"USER": "alice"})
    missing_binary = tmp_path / "compute-only" / "apptainer"
    assert not missing_binary.exists()
    updated = ApptainerProfile.model_validate(
        {**profile.apptainer.model_dump(), "binary": missing_binary}
    )
    assert updated.binary == missing_binary


def test_profile_runtime_environment_is_expanded_and_portable(tmp_path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath("profile.toml")
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            'RSI_SHARED_DATA_ROOT = "/rsi-data"',
            'RSI_SHARED_DATA_ROOT = "${DATA_ROOT}"',
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
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath("profile.toml")
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
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath("profile.toml")
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
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath("profile.toml")
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text().replace(
            "extra_binds = []", 'extra_binds = ["/srv/shared-project"]'
        )
    )

    with pytest.raises(SetupError, match="use work_binds and judge_binds"):
        load_cluster_profile(profile_path, {"USER": "alice"})


def test_existing_profile_retains_legacy_mount_policy(tmp_path: Path) -> None:
    packaged = resources.files("rsi_harness.cluster.bluevela").joinpath("profile.toml")
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text()
        .replace('mount_policy = "scoped"\n', "")
        .replace("extra_binds = []", 'extra_binds = ["/srv/shared-project"]')
    )

    profile = load_cluster_profile(profile_path, {"USER": "alice"})

    assert profile.apptainer.mount_policy == "legacy"
    assert profile.apptainer.extra_binds == (Path("/srv/shared-project"),)


def _slurm_profile(tmp_path: Path, *, scheduler: str = "", builder: str = "") -> Path:
    packaged = resources.files("rsi_harness.cluster.slurm").joinpath("profile.toml")
    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        packaged.read_text()
        .replace("[storage]\n", f"{scheduler}\n[storage]\n")
        .replace("[apptainer]\n", f"{builder}\n[apptainer]\n")
    )
    return profile_path


def test_site_keys_default_to_the_existing_behaviour(tmp_path: Path) -> None:
    profile = load_cluster_profile(_slurm_profile(tmp_path), {"USER": "alice"})

    assert profile.scheduler.gpu_option == "gres"
    assert (profile.builder.conmon, profile.builder.runtime) == (None, None)
    assert profile.builder.storage_driver == "overlay"
    assert profile.builder.host_namespaces == ()
    assert profile.apptainer.network_isolation == "namespace"
    assert profile.apptainer.process_namespaces == "apptainer"


def test_site_keys_load_and_reject_bad_values(tmp_path: Path) -> None:
    builder = (
        'conmon = "/opt/podman/bin/conmon"\nruntime = "/opt/podman/bin/runc"\n'
        'storage_driver = "vfs"\nhost_namespaces = ["network", "uts"]\n'
    )
    profile = load_cluster_profile(
        _slurm_profile(tmp_path, scheduler='gpu_option = "gpus"', builder=builder),
        {"USER": "alice"},
    )
    assert profile.scheduler.gpu_option == "gpus"
    assert profile.builder.storage_driver == "vfs"
    assert profile.builder.host_namespaces == ("network", "uts")

    for scheduler, bad_builder, match in (
        ('gpu_option = "gpu"', "", "gpu_option"),
        ("", 'conmon = "bin/conmon"', "absolute"),
        ("", 'storage_driver = "btrfs"', "storage_driver"),
        ("", 'host_namespaces = ["pid"]', "host_namespaces"),
        ("", 'host_namespaces = ["uts", "uts"]', "duplicate"),
        ("", 'podman_bundle = "/shared/podman"', "binary comes from podman_bundle"),
    ):
        with pytest.raises(SetupError, match=match):
            load_cluster_profile(
                _slurm_profile(tmp_path, scheduler=scheduler, builder=bad_builder),
                {"USER": "alice"},
            )


SHERLOCK = Path(__file__).parents[2] / "src/rsi_harness/cluster/slurm/sherlock.toml"
SHERLOCK_ENV = {
    "USER": "alice",
    "SCRATCH": "/scratch/users/alice",
    "OAK": "/oak/stanford/groups/lab",
    "GROUP_HOME": "/home/groups/lab",
}


def test_the_sherlock_site_profile_loads(tmp_path: Path) -> None:
    profile = load_cluster_profile("sherlock", SHERLOCK_ENV)
    assert profile == load_cluster_profile(SHERLOCK, SHERLOCK_ENV)

    assert (profile.adapter, profile.owner) == ("slurm", "alice")
    scheduler = profile.scheduler
    assert (scheduler.queue, scheduler.group, scheduler.exclusive) == (
        "normal",
        "",
        False,
    )
    assert (scheduler.gpu_option, scheduler.request_tmp) == ("gpus", False)
    assert scheduler.run_walltime_limit == "48:00"
    assert scheduler.build is not None and scheduler.build.partition == "normal"
    assert profile.storage.run_root == Path("/scratch/users/alice/rsi/runs")
    assert profile.storage.logs_root == Path("/oak/stanford/groups/lab/alice/rsi/logs")
    builder = profile.builder
    bundle = Path("/home/groups/lab/rsi-harness/podman")
    assert builder.podman_bundle == bundle
    tree = podman_bundle.bundle_dir(bundle)
    assert (builder.binary, builder.conmon, builder.runtime) == (
        tree / "bin/podman",
        tree / "lib/podman/conmon",
        tree / "bin/runc",
    )
    assert (builder.storage_driver, builder.host_namespaces) == (
        "vfs",
        ("network", "uts"),
    )
    assert builder.temp_root == Path("/lscratch/alice")
    apptainer = profile.apptainer
    assert (apptainer.network_isolation, apptainer.process_namespaces) == (
        "seccomp",
        "harness",
    )
    assert (apptainer.mount_policy, apptainer.build_args) == ("scoped", ())
    assert apptainer.judge_isolation_fallback == "none"

    gpu = tmp_path / "sherlock-gpu.toml"
    # The first partition is the scheduler's; the build's stays "normal".
    gpu.write_text(
        SHERLOCK.read_text().replace('partition = "normal"', 'partition = "gpu"', 1)
    )
    gpu_scheduler = load_cluster_profile(gpu, SHERLOCK_ENV).scheduler
    assert gpu_scheduler.queue == "gpu"
    assert gpu_scheduler.build is not None
    assert gpu_scheduler.build.partition == "normal"
    missing = {k: v for k, v in SHERLOCK_ENV.items() if k != "GROUP_HOME"}
    with pytest.raises(SetupError, match="GROUP_HOME"):
        load_cluster_profile(SHERLOCK, missing)
