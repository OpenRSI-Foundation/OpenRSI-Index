"""Author declarations cannot grant host authority or bypass finite limits."""

import copy
import tomllib

import pytest
from pydantic import ValidationError

from rsi_harness.errors import SetupError


@pytest.fixture
def api():
    # Deferred imports let the initial RED run also exercise old compiler tests.
    from rsi_harness.runtime import sandbox_contracts, sandbox_policy
    from tests import sandbox_helpers

    return sandbox_contracts, sandbox_policy, sandbox_helpers


@pytest.mark.parametrize("bad", [True, 0, -1, "1", 1.5])
def test_profile_rejects_nonpositive_or_coerced_cpu(api, bad):
    contracts, _, helpers = api
    raw = helpers.make_profile().model_dump()
    raw["cpus"] = bad
    with pytest.raises(ValidationError):
        contracts.SandboxProfile.model_validate(raw)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("version", True),
        ("version", 2),
        ("version", "1"),
        ("network", "public"),
        ("gpus", 1),
        ("privileged", True),
    ],
)
def test_task_contract_rejects_unknown_or_unsafe_fields(api, field, bad):
    _, policy, helpers = api
    raw = tomllib.loads(helpers.sandbox_toml())["metadata"]["rsi_harness"]["sandbox"]
    raw[field] = bad
    with pytest.raises(SetupError, match="sandbox"):
        policy.parse_sandbox_task(raw)


@pytest.mark.parametrize("image", ["ubuntu:latest", "ubuntu", "sha256:abcd", ""])
def test_image_must_be_immutable(api, image):
    contracts, _, helpers = api
    raw = helpers.make_profile().model_dump()
    raw["image"] = image
    with pytest.raises(ValidationError, match="image"):
        contracts.SandboxProfile.model_validate(raw)


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/proc",
        "/proc/data",
        "/sys",
        "/dev",
        "/dev/data",
        "/run",
        "/run/rsi-harness/sandbox",
        "/workspace/../tmp",
        "relative",
        "/tmp/",
    ],
)
def test_tmpfs_cannot_mask_authority_or_overlap(api, path):
    contracts, _, helpers = api
    raw = helpers.make_profile().model_dump()
    raw["tmpfs_mb"] += ((path, 1),)
    with pytest.raises(ValidationError):
        contracts.SandboxProfile.model_validate(raw)


@pytest.mark.parametrize("change", ["duplicate", "nested", "missing_tmp", "cwd"])
def test_profile_scratch_is_complete_and_unambiguous(api, change):
    contracts, _, helpers = api
    raw = helpers.make_profile().model_dump()
    if change == "duplicate":
        raw["tmpfs_mb"] += (("/workspace", 1),)
    elif change == "nested":
        raw["tmpfs_mb"] += (("/workspace/sub", 1),)
    elif change == "missing_tmp":
        raw["tmpfs_mb"] = (("/workspace", 1), ("/dev/shm", 1))
    else:
        raw["workdir"] = "/opt"
    with pytest.raises(ValidationError):
        contracts.SandboxProfile.model_validate(raw)


@pytest.mark.parametrize("bad", [True, "120", float("nan"), float("inf"), 0, -1])
def test_lifetime_is_finite_and_typed(api, bad):
    contracts, _, helpers = api
    raw = helpers.make_profile().model_dump()
    raw["max_lifetime_sec"] = bad
    with pytest.raises(ValidationError):
        contracts.SandboxProfile.model_validate(raw)


def test_task_request_does_not_grant_operator_permission(api):
    _, policy, helpers = api
    with pytest.raises(SetupError, match="sandbox.*policy"):
        policy.validate_sandbox_policy(helpers.make_sandbox_task(), None, "docker")


def test_absent_capability_stays_disabled_even_with_operator_policy(api):
    _, policy, helpers = api
    assert policy.validate_sandbox_policy(None, None, "docker") is None
    assert (
        policy.validate_sandbox_policy(None, helpers.make_sandbox_policy(), "docker")
        is None
    )


@pytest.mark.parametrize("backend", ["bluevela", "slurm", "apptainer", "remote"])
def test_unsupported_backend_never_falls_back(api, backend):
    _, policy, helpers = api
    with pytest.raises(SetupError, match="sandbox.*local.*Docker"):
        policy.validate_sandbox_policy(
            helpers.make_sandbox_task(), helpers.make_sandbox_policy(), backend
        )


def test_toml_normalization_preserves_strict_scalars_and_roundtrips(api):
    contracts, policy, helpers = api
    raw = tomllib.loads(helpers.sandbox_toml())["metadata"]["rsi_harness"]["sandbox"]
    original = copy.deepcopy(raw)
    task = policy.parse_sandbox_task(raw)
    assert task.work.profiles == ("offline",)
    assert task.judge is None
    assert task.profiles == (helpers.make_profile(),)
    assert contracts.SandboxTask.model_validate_json(task.model_dump_json()) == task
    assert raw == original


@pytest.mark.parametrize("which", ["missing", "duplicate", "unknown", "too_large"])
def test_phase_membership_and_consistency(api, which):
    _, policy, helpers = api
    raw = tomllib.loads(helpers.sandbox_toml())["metadata"]["rsi_harness"]["sandbox"]
    if which == "missing":
        raw.pop("work")
    elif which == "duplicate":
        raw["profiles"] *= 2
    elif which == "unknown":
        raw["work"]["profiles"] = ["other"]
    else:
        raw["profiles"][0]["cpus"] = 3
    with pytest.raises(SetupError):
        policy.parse_sandbox_task(raw)


def test_every_phase_ceiling_is_rejected_not_clamped(api):
    contracts, policy, helpers = api
    for field in contracts.SandboxLimits.model_fields:
        requested = helpers.make_sandbox_task()
        limits = requested.work.limits.model_copy(
            update={field: getattr(requested.work.limits, field) + 1}
        )
        requested = requested.model_copy(
            update={"work": requested.work.model_copy(update={"limits": limits})}
        )
        with pytest.raises(SetupError, match=field):
            policy.validate_sandbox_policy(
                requested, helpers.make_sandbox_policy(), "docker"
            )


def test_operator_must_approve_phase_and_exact_profile(api):
    _, policy, helpers = api
    allowed = helpers.make_sandbox_policy()
    with pytest.raises(SetupError, match="judge"):
        policy.validate_sandbox_policy(
            helpers.make_sandbox_task(),
            allowed.model_copy(update={"judge": None}),
            "docker",
        )
    changed = helpers.make_profile().model_copy(update={"image": "sha256:" + "b" * 64})
    with pytest.raises(SetupError, match="profiles.offline"):
        policy.validate_sandbox_policy(
            helpers.make_sandbox_task(),
            allowed.model_copy(update={"profiles": (changed,)}),
            "docker",
        )


def test_resolved_envelope_counts_both_parents_paused_work_and_io_headroom(api):
    contracts, _, helpers = api
    grant = helpers.make_sandbox_grant()
    assert (
        grant.reserved_cpus == 6
    )  # conservative 2 parents + both child phase ceilings
    assert grant.reserved_memory_mb == (
        2 * 256 + 2 * 512 + contracts.BROKER_HEADROOM_MB + 8
    )  # 108 records * 64 KiB + 2 live environments * 512 KiB, rounded up
    assert grant.profiles[0].image == "sha256:" + "a" * 64


@pytest.mark.parametrize("field", ["max_operations", "max_created"])
def test_run_cumulative_metadata_is_reserved_independently_of_live_children(api, field):
    _, policy, helpers = api
    task, allowed = helpers.make_sandbox_task(), helpers.make_sandbox_policy()
    base = helpers.make_sandbox_grant()
    limits = allowed.run_limits.model_copy(
        update={field: getattr(allowed.run_limits, field) + 16}
    )
    raised = allowed.model_copy(update={"run_limits": limits})
    grant = policy.resolve_sandbox_grant(
        task, raised, {"offline": helpers.make_profile().image}, 1, 256
    )
    assert grant.reserved_memory_mb == base.reserved_memory_mb + 1
    assert grant.reserved_cpus == base.reserved_cpus
    with pytest.raises(SetupError, match="pool_memory_mb"):
        policy.resolve_sandbox_grant(
            task,
            raised.model_copy(update={"pool_memory_mb": base.reserved_memory_mb}),
            {"offline": helpers.make_profile().image}, 1, 256,
        )


def test_large_cumulative_metadata_cannot_borrow_fixed_transfer_headroom(api):
    _, policy, helpers = api
    allowed = helpers.make_sandbox_policy()
    limits = allowed.run_limits.model_copy(update={"max_operations": 1_000_000})
    with pytest.raises(SetupError, match="pool_memory_mb"):
        policy.resolve_sandbox_grant(
            helpers.make_sandbox_task(),
            allowed.model_copy(update={"run_limits": limits}),
            {"offline": helpers.make_profile().image}, 1, 256,
        )


def test_live_backend_environment_metadata_has_separate_allowance(api):
    _, policy, helpers = api
    allowed = helpers.make_sandbox_policy()
    limits = allowed.run_limits.model_copy(update={"max_live": 4})
    grant = policy.resolve_sandbox_grant(
        helpers.make_sandbox_task(),
        allowed.model_copy(update={"run_limits": limits}),
        {"offline": helpers.make_profile().image}, 1, 256,
    )
    assert grant.reserved_memory_mb == (
        helpers.make_sandbox_grant().reserved_memory_mb + 1
    )


@pytest.mark.parametrize("cpus,memory", [(None, 256), (1, None), (0, 256), (1, True)])
def test_parent_envelope_requires_finite_limits(api, cpus, memory):
    _, policy, helpers = api
    with pytest.raises(SetupError, match="parent"):
        policy.resolve_sandbox_grant(
            helpers.make_sandbox_task(),
            helpers.make_sandbox_policy(),
            image_ids={"offline": helpers.make_profile().image},
            parent_cpus=cpus,
            parent_memory_mb=memory,
        )


def test_unresolved_image_and_partition_overflow_are_actionable(api):
    _, policy, helpers = api
    task, allowed = helpers.make_sandbox_task(), helpers.make_sandbox_policy()
    with pytest.raises(SetupError, match="offline.*image"):
        policy.resolve_sandbox_grant(
            task, allowed, image_ids={}, parent_cpus=1, parent_memory_mb=256
        )
    with pytest.raises(SetupError, match="pool_memory_mb"):
        policy.resolve_sandbox_grant(
            task,
            allowed.model_copy(update={"pool_memory_mb": 1000}),
            image_ids={"offline": helpers.make_profile().image},
            parent_cpus=1,
            parent_memory_mb=256,
        )


def test_policy_file_errors_name_the_file(api, tmp_path):
    _, policy, _ = api
    path = tmp_path / "missing-policy.toml"
    with pytest.raises(SetupError, match="missing-policy.toml"):
        policy.load_sandbox_policy(path)
