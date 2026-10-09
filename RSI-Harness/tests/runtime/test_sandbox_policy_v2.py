"""Environment grants need both keys, never clamp, and journal recoverably."""

import copy
import hashlib
import json
import re
import tomllib
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from rsi_harness.errors import InfrastructureError, SetupError, UnsupportedTaskError
from rsi_harness.models import RunPlan, TaskDefinition
from rsi_harness.runtime import sandbox_env_contracts as env
from rsi_harness.runtime.recovery import LeaseStore, RecoveryManager
from rsi_harness.runtime.sandbox_budget import SandboxAdmissionPool
from rsi_harness.runtime.sandbox_contracts import (
    BROKER_HEADROOM_MB,
    ENV_METADATA_MAX_MB,
    ENV_METADATA_MIN_MB,
    MAX_CONTAINERS_LIVE,
    MAX_ENVS_LIVE,
    MAX_EXECS_RUNNING,
    MAX_IMAGE_HANDLES,
    MAX_JOBS_RUNNING,
    MAX_WAITERS,
    EnvLimits,
    EnvLimitsRequest,
    SandboxEnvGrant,
    SandboxEnvTask,
    SandboxError,
    SandboxGrant,
    SandboxOwner,
    SandboxPolicy,
    SandboxReservation,
    SandboxTask,
    env_metadata_mb,
)
from rsi_harness.runtime.sandbox_policy import (
    builder_image_id,
    parse_sandbox_task,
    resolve_env_grant,
    resolve_sandbox_grant,
    validate_sandbox_policy,
)
from tests import sandbox_helpers as helpers
from tests.factories import DEFAULT_TASK_TOML, make_run_plan, write_harbor_task
from tests.runtime.test_recovery import lease as parent_lease
from tests.runtime.test_sandbox_budget import authority, make_lease
from tests.runtime.test_sandbox_recovery import ChildRecoveryBackend

ENV_ID = "e" + "1" * 32
IMAGE_ID = "sha256:" + "c" * 64


def task_with(tmp_path, *, phase="judge", **values):
    """A v2 task requesting one phase; TOML-free for concise matrices."""
    request = {"network": ("public",), **values}
    return SandboxEnvTask.model_validate(
        {"version": 2, "environments": {phase: request}}
    )


def resolve(task, policy, builders=None):
    return resolve_env_grant(
        task,
        policy,
        {"work": helpers.builder_inspect(), "judge": helpers.builder_inspect()}
        if builders is None
        else builders,
        parent_cpus=1,
        parent_memory_mb=256,
    )


def with_phase(policy, phase="judge", **updates):
    grant = getattr(policy.environments, phase).model_copy(update=updates)
    environments = policy.environments.model_copy(update={phase: grant})
    return policy.model_copy(update={"environments": environments})


def test_documented_request_resolves_to_narrowed_operator_grant(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    task = helpers.make_env_task()
    assert isinstance(task, SandboxEnvTask)
    assert validate_sandbox_policy(task, policy, "docker") is task
    grant = resolve(task, policy)
    judge = grant.environments.judge
    assert grant.environments.work is None  # granted, but not requested
    assert judge.network == ("public",)
    assert judge.pull is True and judge.registries == ("docker.io",)
    assert judge.build.builder_image == helpers.BUILDER_IMAGE_ID
    # Builds get the requested network, not the operator's whole build grant.
    assert judge.build.network == ("public",)
    assert policy.environments.judge.build.network == ("public", "none")
    assert judge.max_envs_live == 2  # the task tightened this one
    operator = policy.environments.judge
    for field in EnvLimits.model_fields:
        if field != "max_envs_live":
            assert getattr(judge, field) == getattr(operator, field), field


def test_unrequested_capabilities_stay_disabled(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    grant = resolve(task_with(tmp_path, network=("none",)), policy, builders={})
    judge = grant.environments.judge
    assert judge.network == ("none",)
    assert judge.pull is False and judge.registries == ()
    assert judge.build is None
    assert policy.environments.judge.pull and policy.environments.judge.build


@pytest.mark.parametrize("phase", ["work", "judge"])
@pytest.mark.parametrize(
    "denied", ["phase", "network", "pull", "build", "build_network", "section"]
)
def test_request_outside_operator_grant_is_setup_error(tmp_path, phase, denied):
    policy = helpers.make_env_policy(tmp_path)
    task = task_with(tmp_path, phase=phase, pull=True, build=True)
    prefix = f"environments.{phase}"
    if denied == "phase":
        environments = policy.environments.model_copy(update={phase: None})
        policy = policy.model_copy(update={"environments": environments})
        match = f"{prefix}: phase"
    elif denied == "network":
        policy = with_phase(policy, phase, network=("none",))
        match = f"{prefix}.network: public"
    elif denied == "pull":
        policy = with_phase(policy, phase, pull=False)
        match = f"{prefix}.pull:"
    elif denied == "build":
        policy = with_phase(policy, phase, build=None)
        match = f"{prefix}.build:"
    elif denied == "build_network":
        build = getattr(policy.environments, phase).build
        offline = build.model_copy(update={"network": ("none",)})
        policy = with_phase(policy, phase, build=offline)
        match = f"{prefix}.build_network: public"
    else:
        policy = helpers.make_sandbox_policy()
        match = "sandbox.environments"
    with pytest.raises(SetupError, match=match):
        validate_sandbox_policy(task, policy, "docker")


def test_operator_pull_defaults_off(tmp_path):
    text = helpers.env_policy_toml().replace("pull = true\n", "")
    policy = helpers.load_policy_text(tmp_path, text)
    for phase in ("work", "judge"):
        grant = getattr(policy.environments, phase)
        assert grant.pull is False and grant.registries == ("docker.io",)
        task = task_with(tmp_path, phase=phase, pull=True)
        with pytest.raises(SetupError, match=f"environments.{phase}.pull:"):
            validate_sandbox_policy(task, policy, "docker")


def test_build_network_needs_both_keys(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    offline = resolve(task_with(tmp_path, network=("none",), build=True), policy)
    assert offline.environments.judge.build.network == ("none",)
    # An offline env may still build online, but only when the task says so.
    online = resolve(
        task_with(tmp_path, network=("none",), build=True, build_network=("public",)),
        policy,
    )
    assert online.environments.judge.network == ("none",)
    assert online.environments.judge.build.network == ("public",)
    build = policy.environments.judge.build.model_copy(update={"network": ("public",)})
    narrow = with_phase(policy, build=build)
    with pytest.raises(SetupError, match="judge.build_network: none"):
        validate_sandbox_policy(
            task_with(tmp_path, network=("none",), build=True), narrow, "docker"
        )


@pytest.mark.parametrize("field", list(EnvLimits.model_fields))
def test_every_limit_above_the_grant_is_rejected_not_clamped(tmp_path, field):
    policy = helpers.make_env_policy(tmp_path)
    if field == "swap_ratio":  # a fraction: at most 1 on either side
        policy = with_phase(policy, swap_ratio=0.5)
    permitted = getattr(policy.environments.judge, field)
    above = 0.75 if field == "swap_ratio" else permitted + 1
    task = task_with(tmp_path, limits={field: above})
    with pytest.raises(SetupError, match=f"limits.{field}"):
        validate_sandbox_policy(task, policy, "docker")


def test_swap_defaults_to_stock_docker_and_a_task_may_only_lower_it(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    operator = policy.environments.judge
    # Stock Harbor/Docker: as much swap as memory.
    assert operator.swap_ratio == 1.0
    assert operator.max_swap_mb_live == operator.max_memory_mb_live == 32768
    assert resolve(task_with(tmp_path), policy).environments.judge.swap_ratio == 1.0
    off = resolve(task_with(tmp_path, limits={"swap_ratio": 0}), policy)
    judge = off.environments.judge
    assert (judge.swap_ratio, judge.max_swap_mb_live) == (0.0, 0)
    half = resolve(task_with(tmp_path, limits={"swap_ratio": 0.3}), policy)
    assert half.environments.judge.max_swap_mb_live == 9830  # floor(0.3 * 32768)
    # Swap is not RAM: the memory reservation does not change with it.
    assert half.reserved_memory_mb == off.reserved_memory_mb
    disabled = with_phase(policy, swap_ratio=0.0)
    with pytest.raises(SetupError, match="limits.swap_ratio: requested 0.5"):
        validate_sandbox_policy(
            task_with(tmp_path, limits={"swap_ratio": 0.5}), disabled, "docker"
        )
    text = helpers.env_policy_toml().replace(
        "max_operations = 1000000\n", "max_operations = 1000000\nswap_ratio = 0\n"
    )
    loaded = helpers.load_policy_text(tmp_path, text)
    assert loaded.environments.work.swap_ratio == 0.0
    for value in (float("nan"), float("inf"), 1.01, -0.01, True):
        with pytest.raises(ValidationError, match="swap_ratio"):
            EnvLimitsRequest(swap_ratio=value)


def test_limit_request_fields_mirror_operator_limits():
    assert list(EnvLimitsRequest.model_fields) == list(EnvLimits.model_fields)


def test_tightened_limits_must_stay_consistent(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    task = task_with(tmp_path, limits={"max_envs_created": 1})  # live stays 4
    with pytest.raises(SetupError, match="max_envs_live"):
        validate_sandbox_policy(task, policy, "docker")


def test_environment_request_needs_local_docker_and_an_operator(tmp_path):
    task = helpers.make_env_task()
    with pytest.raises(SetupError, match="sandbox.*policy"):
        validate_sandbox_policy(task, None, "docker")
    with pytest.raises(SetupError, match="sandbox.*local.*Docker"):
        validate_sandbox_policy(task, helpers.make_env_policy(tmp_path), "bluevela")


def test_profile_task_is_not_approved_by_environment_only_policy(tmp_path):
    with pytest.raises(SetupError, match="profiles.offline"):
        validate_sandbox_policy(
            helpers.make_sandbox_task(), helpers.make_env_policy(tmp_path), "docker"
        )


@pytest.mark.parametrize(
    "table,key,value,match",
    [
        ("judge", "max_operations", "3000000", "run_limits.max_operations"),
        ("judge", "privileged", "true", "privileged"),
        ("judge", "pull", '"yes"', "pull"),
        ("judge", "network", '["public", "public"]', "duplicate"),
        ("judge", "network", '["host"]', "network"),
        ("judge", "network", "[]", "network"),
        ("judge", "registries", "[]", "registry"),
        ("judge", "registries", '["docker.io", "docker.io"]', "duplicate"),
        ("judge", "max_services_per_env", "9", "max_services_per_env"),
        ("judge", "swap_ratio", "1.5", "swap_ratio"),
        ("judge", "swap_ratio", "-0.5", "swap_ratio"),
        ("judge", "swap_ratio", "true", "swap_ratio"),
        ("judge", "swap_ratio", '"1"', "swap_ratio"),
        ("judge", "max_envs_live", "401", "max_envs_live"),
        ("judge", "max_envs_live", str(MAX_ENVS_LIVE + 1), "max_envs_live"),
        (
            "judge",
            "max_containers_live",
            str(MAX_CONTAINERS_LIVE + 1),
            "max_containers_live",
        ),
        ("judge", "max_execs_running", str(MAX_EXECS_RUNNING + 1), "max_execs_running"),
        ("judge", "max_jobs_running", str(MAX_JOBS_RUNNING + 1), "max_jobs_running"),
        ("judge.build", "builder_image", '"moby/buildkit:v0.27.1"', "immutable"),
        ("judge.build", "max_builds", "257", "max_builds"),
        ("judge.build", "max_concurrent_builds", "5", "max_concurrent_builds"),
        ("judge.build", "memory_mb", "512", "memory_mb"),
        ("judge.build", "pids", "256", "pids"),
        ("judge.build", "state_fs", '"tmpfs"', "tmpfs"),
        ("judge.build", "allow", '["network.host"]', "allow"),
        (
            "judge.build",
            "syntax_frontends",
            '["docker.io/docker/dockerfile", "docker.io/docker/dockerfile"]',
            "duplicate",
        ),
        ("host", "disk_hard_floor_mb", "20000", "disk_hard_floor_mb"),
        ("host", "waiters", "0", "waiters"),
        ("host", "waiters", str(MAX_WAITERS + 1), "waiters"),
    ],
)
def test_operator_policy_rejects_unsafe_or_inconsistent_grants(
    tmp_path, table, key, value, match
):
    raw = tomllib.loads(helpers.env_policy_toml())
    section = raw["environments"]
    for part in table.split("."):
        section = section[part]
    if key == "registries":
        section["pull"] = True
    section[key] = tomllib.loads(f"v = {value}")["v"]
    with pytest.raises(SetupError, match=match):
        helpers.load_policy_text(tmp_path, _dump_toml(raw))


def big_caps_toml():
    raw = tomllib.loads(helpers.env_policy_toml())
    raw["environments"]["host"]["waiters"] = MAX_WAITERS
    for phase in ("work", "judge"):
        raw["environments"][phase].update(
            max_envs_live=128, max_containers_live=512, max_execs_running=512
        )
    return raw


def test_the_live_caps_admit_128_envs_and_refuse_129(tmp_path):
    raw = big_caps_toml()
    policy = helpers.load_policy_text(tmp_path, _dump_toml(raw))
    grant = resolve(task_with(tmp_path), policy)
    judge = grant.environments.judge
    assert (judge.max_envs_live, judge.max_containers_live) == (128, 512)
    assert (judge.max_execs_running, grant.environments.host.waiters) == (512, 1024)
    assert (MAX_ENVS_LIVE, MAX_CONTAINERS_LIVE, MAX_EXECS_RUNNING) == (128, 512, 512)
    assert (judge.max_image_handles, MAX_WAITERS) == (256, 1024)
    # The metadata allowance grows with the live caps: this Judge-only task
    # at the bounds is half of ENV_METADATA_MAX_MB's phases plus the waiters.
    base = resolve(task_with(tmp_path), helpers.make_env_policy(tmp_path))
    judge_only = (ENV_METADATA_MAX_MB - 128) // 2 + 128
    assert env_metadata_mb([judge], MAX_WAITERS) == judge_only == 812
    assert grant.reserved_memory_mb - base.reserved_memory_mb == (
        judge_only - ENV_METADATA_MIN_MB
    )
    raw["environments"]["judge"]["max_envs_live"] = 129
    with pytest.raises(SetupError, match="max_envs_live"):
        helpers.load_policy_text(tmp_path, _dump_toml(raw))
    # A task may tighten below the operator's 128, never raise past it.
    narrowed = resolve(task_with(tmp_path, limits={"max_envs_live": 100}), policy)
    assert narrowed.environments.judge.max_envs_live == 100
    assert narrowed.environments.judge.max_image_handles == 200
    with pytest.raises(SetupError, match="max_envs_live"):
        resolve(task_with(tmp_path, limits={"max_envs_live": 129}), policy)


@pytest.mark.parametrize(
    ("table", "field", "value"),
    [
        ("judge", "max_containers_live", MAX_CONTAINERS_LIVE + 1),
        ("judge", "max_execs_running", MAX_EXECS_RUNNING + 1),
        ("judge", "max_jobs_running", MAX_JOBS_RUNNING + 1),
        ("host", "waiters", MAX_WAITERS + 1),
    ],
)
def test_the_live_caps_refuse_one_past_each_bound(tmp_path, table, field, value):
    raw = big_caps_toml()
    raw["environments"][table][field] = value
    with pytest.raises(SetupError, match=field):
        helpers.load_policy_text(tmp_path, _dump_toml(raw))


def test_operator_must_approve_something(tmp_path):
    with pytest.raises(SetupError, match="profiles or environments"):
        helpers.load_policy_text(tmp_path, "pool_cpus = 8\npool_memory_mb = 4096\n")
    raw = tomllib.loads(helpers.env_policy_toml())
    raw["environments"].pop("work")
    raw["environments"].pop("judge")
    with pytest.raises(SetupError, match="grant work or judge"):
        helpers.load_policy_text(tmp_path, _dump_toml(raw))


def _without_run_limits(text):
    return text.rsplit("\n[run_limits]", 1)[0] + "\n"


def _without_profiles(text):
    head, rest = text.split("[[profiles]]", 1)
    return head + "[work]" + rest.split("[work]", 1)[1]


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("missing", ["run_limits", "profiles", "profile_tables"])
def test_profile_approval_still_requires_profiles_and_run_limits(
    tmp_path, missing, mixed
):
    """v1 required both before environments made them optional fields."""
    text = helpers.sandbox_policy_toml(pool=not mixed)
    if missing == "run_limits":
        text = _without_run_limits(text)
    elif missing == "profiles":
        text = _without_profiles(text)
    else:  # a bare [run_limits] table approves no profile at all
        text = "\n[run_limits]" + text.rsplit("\n[run_limits]", 1)[1]
        if not mixed:
            text = "version = 1\npool_cpus = 8\npool_memory_mb = 4096\n" + text
    if mixed:
        text = helpers.env_policy_toml() + text
    with pytest.raises(SetupError, match="profiles and run_limits"):
        helpers.load_policy_text(tmp_path, text)


@pytest.mark.parametrize(
    "change",
    [
        "unknown_phase_key",
        "profiles_with_v2",
        "unknown_limit",
        "host_network",
        "duplicate_network",
        "coerced_bool",
        "empty_environments",
        "coerced_version",
        "limit_zero",
        "build_network_without_build",
        "duplicate_build_network",
        "host_build_network",
    ],
)
def test_task_metadata_v2_is_intent_only_and_strict(change):
    raw = tomllib.loads(helpers.env_task_toml())["metadata"]["rsi_harness"]["sandbox"]
    judge = raw["environments"]["judge"]
    if change == "unknown_phase_key":
        judge["privileged"] = True
    elif change == "profiles_with_v2":
        raw["profiles"] = []
    elif change == "unknown_limit":
        judge["limits"]["max_gpus"] = 1
    elif change == "host_network":
        judge["network"] = ["host"]
    elif change == "duplicate_network":
        judge["network"] = ["public", "public"]
    elif change == "coerced_bool":
        judge["pull"] = "true"
    elif change == "empty_environments":
        raw["environments"] = {}
    elif change == "coerced_version":
        raw["version"] = True
    elif change == "limit_zero":
        judge["limits"]["max_envs_live"] = 0
    elif change == "build_network_without_build":
        judge.update(build=False, build_network=["public"])
    elif change == "duplicate_build_network":
        judge["build_network"] = ["none", "none"]
    else:
        judge["build_network"] = ["host"]
    with pytest.raises(UnsupportedTaskError, match="metadata.rsi_harness.sandbox"):
        parse_sandbox_task(raw)


def test_compiler_carries_v2_request_through_task_definition(tmp_path):
    from rsi_harness.models import CompileOptions
    from rsi_harness.task.compiler import HarborTaskCompiler

    task = write_harbor_task(
        tmp_path, task_toml=DEFAULT_TASK_TOML + helpers.env_task_toml()
    )
    definition = HarborTaskCompiler().compile(task, CompileOptions())
    assert definition.sandbox == helpers.make_env_task()
    assert TaskDefinition.model_validate(dict(definition)) == definition
    union = TypeAdapter(TaskDefinition.model_fields["sandbox"].annotation)
    restored = union.validate_json(definition.sandbox.model_dump_json())
    assert type(restored) is SandboxEnvTask and restored == definition.sandbox
    v1 = union.validate_json(helpers.make_sandbox_task().model_dump_json())
    assert type(v1) is SandboxTask


@pytest.mark.parametrize("kind", ["v1", "v2", "mixed"])
def test_policy_toml_round_trips(tmp_path, kind):
    text = {
        "v1": helpers.sandbox_policy_toml(),
        "v2": helpers.env_policy_toml(),
        "mixed": helpers.env_policy_toml() + helpers.sandbox_policy_toml(pool=False),
    }[kind]
    policy = helpers.load_policy_text(tmp_path, text)
    assert SandboxPolicy.model_validate_json(policy.model_dump_json()) == policy
    assert (policy.environments is not None) == (kind != "v1")
    assert bool(policy.profiles) == (kind != "v2")
    if kind == "v1":
        assert policy == helpers.make_sandbox_policy()


def test_mixed_policy_serves_both_task_versions_without_changing_v1(tmp_path):
    mixed = helpers.load_policy_text(
        tmp_path, helpers.env_policy_toml() + helpers.sandbox_policy_toml(pool=False)
    )
    v1_only = helpers.make_sandbox_policy().model_copy(
        update={"pool_cpus": mixed.pool_cpus, "pool_memory_mb": mixed.pool_memory_mb}
    )
    image = {"offline": helpers.make_profile().image}
    task = helpers.make_sandbox_task()
    assert validate_sandbox_policy(task, mixed, "docker") is task
    assert resolve_sandbox_grant(task, mixed, image, 1, 256) == resolve_sandbox_grant(
        task, v1_only, image, 1, 256
    )
    assert isinstance(resolve(helpers.make_env_task(), mixed), SandboxEnvGrant)


def test_task_request_round_trips_and_leaves_input_untouched():
    raw = tomllib.loads(helpers.env_task_toml())["metadata"]["rsi_harness"]["sandbox"]
    original = copy.deepcopy(raw)
    task = parse_sandbox_task(raw)
    assert SandboxEnvTask.model_validate_json(task.model_dump_json()) == task
    assert raw == original
    # Only an exact integer version 2 selects environments; v1 stays profiles.
    for version in (1, True, "2"):
        with pytest.raises(UnsupportedTaskError):
            parse_sandbox_task(copy.deepcopy(original) | {"version": version})
    assert isinstance(
        parse_sandbox_task(helpers.make_sandbox_task().model_dump()), SandboxTask
    )


def test_run_plan_union_keeps_each_grant_version(tmp_path):
    plan = dict(make_run_plan(tmp_path))
    union = TypeAdapter(RunPlan.model_fields["sandbox"].annotation)
    for grant, kind in (
        (helpers.make_env_grant(tmp_path), SandboxEnvGrant),
        (helpers.make_sandbox_grant(), SandboxGrant),
    ):
        assert RunPlan.model_validate(plan | {"sandbox": grant}).sandbox is grant
        restored = union.validate_json(grant.model_dump_json())
        assert type(restored) is kind
        assert restored == grant


def test_reservation_counts_parents_env_ceilings_builders_and_disk(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    both = SandboxEnvTask.model_validate(
        {
            "version": 2,
            "environments": {
                "work": {"network": ("public",), "build": True},
                "judge": {"network": ("none",), "build": True},
            },
        }
    )
    grant = resolve(both, policy)
    assert grant.reserved_cpus == 2 * 1 + 2 * (16 + 4)
    assert grant.reserved_memory_mb == (
        2 * 256 + 2 * (32768 + 8192) + BROKER_HEADROOM_MB + ENV_METADATA_MIN_MB
    )
    # max_disk_mb_live + builder state fs + loaded images, for both phases.
    assert grant.reserved_disk_mb == 2 * (8192 + 6144 + 6144) == grant.pool_disk_mb
    judge_only = resolve(task_with(tmp_path, network=("none",)), policy, builders={})
    assert judge_only.reserved_cpus == 2 + 16
    assert judge_only.reserved_memory_mb == (
        2 * 256 + 32768 + BROKER_HEADROOM_MB + ENV_METADATA_MIN_MB
    )
    assert judge_only.reserved_disk_mb == 8192


def test_tightened_request_shrinks_the_reservation(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    small = task_with(
        tmp_path,
        limits={"max_cpus_live": 2, "max_memory_mb_live": 1024, "max_disk_mb_live": 64},
    )
    grant = resolve(small, policy, builders={})
    assert (grant.reserved_cpus, grant.reserved_disk_mb) == (4, 64)
    assert (
        grant.reserved_memory_mb
        == 512 + 1024 + BROKER_HEADROOM_MB + ENV_METADATA_MIN_MB
    )


def test_environment_metadata_does_not_grow_with_operations(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    base = resolve(helpers.make_env_task(), policy)
    run_limits = policy.environments.run_limits.model_copy(
        update={"max_operations": 10**9, "max_envs_created": 10**6}
    )
    environments = policy.environments.model_copy(update={"run_limits": run_limits})
    raised = resolve(
        helpers.make_env_task(),
        policy.model_copy(update={"environments": environments}),
    )
    assert raised.reserved_memory_mb == base.reserved_memory_mb


def test_environment_metadata_allowance_covers_bounded_live_records(tmp_path):
    """The per-record allowances documented beside env_metadata_mb."""
    kib = 1024
    per_phase = (
        MAX_ENVS_LIVE * 64 * kib
        + (MAX_CONTAINERS_LIVE + MAX_EXECS_RUNNING + MAX_IMAGE_HANDLES) * 512 * kib
        + MAX_JOBS_RUNNING * 8 * 1024 * kib
        + 4096 * kib  # MAX_TOMBSTONES of about 1 KiB
    )
    # Work plus one Judge round, and the host's long-poll waiters.
    total = 2 * per_phase + MAX_WAITERS * 128 * kib
    assert total == ENV_METADATA_MAX_MB * 1024 * kib
    judge = helpers.make_env_policy(tmp_path).environments.judge
    largest = judge.model_copy(
        update={
            "max_envs_live": MAX_ENVS_LIVE,
            "max_containers_live": MAX_CONTAINERS_LIVE,
            "max_execs_running": MAX_EXECS_RUNNING,
            "max_jobs_running": MAX_JOBS_RUNNING,
        }
    )
    assert largest.max_image_handles == MAX_IMAGE_HANDLES
    assert env_metadata_mb([largest, largest], MAX_WAITERS) == ENV_METADATA_MAX_MB
    # Small grants reserve the allowance they did before the bounds grew.
    small = largest.model_copy(
        update={"max_envs_live": 1, "max_containers_live": 1, "max_execs_running": 1}
    )
    assert small.max_image_handles == 64
    assert env_metadata_mb([small, small], 1) == ENV_METADATA_MIN_MB


@pytest.mark.parametrize(
    "field", ["pool_cpus", "pool_memory_mb", "environments.host.pool_disk_mb"]
)
def test_pool_overflow_names_the_dimension(tmp_path, field):
    policy = helpers.make_env_policy(tmp_path)
    grant = resolve(helpers.make_env_task(), policy)
    if field == "pool_cpus":
        policy = policy.model_copy(update={"pool_cpus": grant.reserved_cpus - 1})
    elif field == "pool_memory_mb":
        policy = policy.model_copy(
            update={"pool_memory_mb": grant.reserved_memory_mb - 1}
        )
    else:
        host = policy.environments.host.model_copy(
            update={"pool_disk_mb": grant.reserved_disk_mb - 1}
        )
        environments = policy.environments.model_copy(update={"host": host})
        policy = policy.model_copy(update={"environments": environments})
    with pytest.raises(SetupError, match=field):
        resolve(helpers.make_env_task(), policy)


def test_builder_image_resolves_to_the_inspected_local_id(tmp_path):
    policy = helpers.make_env_policy(tmp_path)
    with pytest.raises(SetupError, match="judge.build.builder_image.*unresolved"):
        resolve(helpers.make_env_task(), policy, builders={})
    pinned = with_phase(
        policy,
        build=policy.environments.judge.build.model_copy(
            update={"builder_image": IMAGE_ID}
        ),
    )
    with pytest.raises(SetupError, match="differs from approved"):
        resolve(helpers.make_env_task(), pinned)
    # An approved ID needs no digest; a digest reference needs no repository.
    by_id = helpers.builder_inspect(IMAGE_ID) | {"RepoDigests": []}
    assert resolve(helpers.make_env_task(), pinned, {"judge": by_id})
    other_repository = helpers.builder_inspect() | {
        "RepoDigests": ["docker.io/moby/buildkit@sha256:" + "f" * 64]
    }
    assert resolve(helpers.make_env_task(), policy, {"judge": other_repository})
    grant = resolve(helpers.make_env_task(), policy)
    raw = json.loads(grant.model_dump_json())
    raw["environments"]["judge"]["build"]["builder_image"] = helpers.BUILDER_IMAGE
    with pytest.raises(ValidationError, match="exact image ID"):
        SandboxEnvGrant.model_validate_json(json.dumps(raw))


def _inspect(**updates):
    return helpers.builder_inspect() | updates


@pytest.mark.parametrize(
    "inspected,match",
    [
        (None, "unavailable or unresolved"),
        ([], "unavailable or unresolved"),
        (_inspect(Id="moby/buildkit:v0.27.1"), "unavailable or unresolved"),
        (_inspect(RepoDigests=["moby/buildkit@sha256:" + "e" * 64]), "digest"),
        (_inspect(RepoDigests=None), "digest"),
        (_inspect(RepoDigests=["moby/buildkit:v0.27.1"]), "digest"),
        (_inspect(Config=None), "Entrypoint"),
        (helpers.builder_inspect(Entrypoint=None), "Entrypoint"),
        (helpers.builder_inspect(Entrypoint="buildkitd"), "Entrypoint"),
        (helpers.builder_inspect(Entrypoint=["/bin/sh", "-c"]), "Entrypoint"),
        (helpers.builder_inspect(Entrypoint=["buildkitd", "--debug"]), "Entrypoint"),
        (
            helpers.builder_inspect(Volumes={"/var/lib/buildkit": {}, "/cache": {}}),
            "volumes other than /var/lib/buildkit",
        ),
        (helpers.builder_inspect(Volumes=["/var/lib/buildkit"]), "volumes"),
    ],
)
def test_builder_image_must_be_the_cached_buildkitd(tmp_path, inspected, match):
    with pytest.raises(SetupError, match=f"judge.build.builder_image: .*{match}"):
        resolve(
            helpers.make_env_task(),
            helpers.make_env_policy(tmp_path),
            {"judge": inspected},
        )


def test_builder_image_without_declared_volume_is_accepted():
    inspected = helpers.builder_inspect(Volumes=None)
    assert (
        builder_image_id("builder", helpers.BUILDER_IMAGE, inspected)
        == helpers.BUILDER_IMAGE_ID
    )


@pytest.mark.parametrize("cpus,memory", [(None, 256), (1, None), (0, 256), (1, True)])
def test_parent_envelope_requires_finite_limits(tmp_path, cpus, memory):
    with pytest.raises(SetupError, match="parent"):
        resolve_env_grant(
            helpers.make_env_task(),
            helpers.make_env_policy(tmp_path),
            {"judge": helpers.builder_inspect()},
            parent_cpus=cpus,
            parent_memory_mb=memory,
        )


def _reserve(store, run_id, grant):
    SandboxAdmissionPool(store).reserve_run(run_id, grant, authority(store, run_id))


def test_admission_pool_reserves_disk_alongside_profile_runs(tmp_path):
    store = LeaseStore(tmp_path / "leases")
    policy = helpers.make_env_policy(tmp_path)
    host = policy.environments.host.model_copy(update={"pool_disk_mb": 40000})
    environments = policy.environments.model_copy(update={"host": host})
    grant = resolve(
        helpers.make_env_task(),
        policy.model_copy(update={"environments": environments}),
    )
    profiles = resolve_sandbox_grant(
        helpers.make_sandbox_task(),
        helpers.make_sandbox_policy().model_copy(
            update={
                "pool_cpus": policy.pool_cpus,
                "pool_memory_mb": policy.pool_memory_mb,
            }
        ),
        {"offline": helpers.make_profile().image},
        1,
        256,
    )
    _reserve(store, "profiles", profiles)
    _reserve(store, "env-1", grant)
    _reserve(store, "env-1", grant)  # idempotent envelope
    assert store.read("profiles").sandbox_reservation.pool_disk_mb is None
    assert store.read("env-1").sandbox_reservation == SandboxReservation(
        cpus=grant.reserved_cpus,
        memory_mb=grant.reserved_memory_mb,
        pool_cpus=grant.pool_cpus,
        pool_memory_mb=grant.pool_memory_mb,
        disk_mb=20480,
        pool_disk_mb=40000,
    )
    # CPU and memory still fit a second run; only the disk pool is exhausted.
    with pytest.raises(SetupError, match="20480 MiB disk; .*19520 MiB disk"):
        _reserve(store, "env-2", grant)
    other_pool = grant.model_copy(
        update={
            "environments": grant.environments.model_copy(
                update={"host": host.model_copy(update={"pool_disk_mb": 81920})}
            )
        }
    )
    with pytest.raises(SetupError, match="capacity differs"):
        _reserve(store, "env-3", other_pool)


def test_disk_reservation_requires_its_pool():
    with pytest.raises(ValidationError, match="pool capacity"):
        SandboxReservation(
            cpus=1, memory_mb=1, pool_cpus=1, pool_memory_mb=1, disk_mb=1
        )


def test_valid_spec_is_strict_frozen_and_digest_ignores_key_order():
    raw = helpers.make_env_spec()
    spec = env.parse_env_spec(raw)
    assert spec.services["main"].command == ("sleep", "infinity")
    assert spec.services["main"].nofile == 65536 and spec.services["main"].pids is None
    assert spec.services["kv"].healthcheck.test == ("CMD", "redis-cli", "ping")
    assert env.EnvSpec.model_validate_json(spec.model_dump_json()) == spec
    reordered = dict(reversed(list(raw.items())))
    reordered["services"] = dict(reversed(list(raw["services"].items())))
    assert env.env_spec_digest(env.parse_env_spec(reordered)) == env.env_spec_digest(
        spec
    )
    changed = helpers.make_env_spec(env={"MODE": "other"})
    assert env.env_spec_digest(env.parse_env_spec(changed)) != env.env_spec_digest(spec)
    with pytest.raises(ValidationError):
        spec.network = "none"


def _assign(mapping, key, value):
    mapping[key] = value


def _delete(mapping, key):
    del mapping[key]


def test_valid_spec_is_deeply_frozen_and_hashable():
    spec = env.parse_env_spec(helpers.make_env_spec())
    digest = env.env_spec_digest(spec)
    main = spec.services["main"]
    for mutate in (
        lambda: _assign(main.env, "NVIDIA_VISIBLE_DEVICES", "all"),
        lambda: _assign(spec.services, "evil", main),
        lambda: _delete(spec.services, "kv"),
        lambda: _assign(spec.volumes, "other", spec.volumes["shared"]),
        lambda: _assign(main.tmpfs, "/proc", 1),
        lambda: _delete(main.depends_on, "kv"),
    ):
        with pytest.raises(TypeError):
            mutate()
    for mapping in (spec.services, spec.volumes, main.env, main.tmpfs):
        assert not any(
            hasattr(mapping, name) for name in ("update", "pop", "setdefault", "clear")
        )
    with pytest.raises(ValidationError):
        main.env = {}
    assert env.env_spec_digest(spec) == digest
    assert hash(spec) == hash(env.parse_env_spec(helpers.make_env_spec()))
    assert main.env == {"MODE": "test"} and dict(main.tmpfs) == {"/scratch": 64}
    # A dump is a detached plain copy; the validated spec never changes.
    dumped = spec.model_dump()
    assert type(dumped["services"]) is dict
    assert type(dumped["services"]["main"]["env"]) is dict
    dumped["services"]["main"]["env"]["NVIDIA_VISIBLE_DEVICES"] = "all"
    assert "NVIDIA_VISIBLE_DEVICES" not in main.env
    assert env.EnvSpec.model_validate(spec.model_dump()) == spec


def _services(raw, count):
    """``count`` copies of the sidecar, without aliases that would collide."""
    template = {
        key: value for key, value in raw["services"]["kv"].items() if key != "aliases"
    }
    raw["services"] = {f"svc{index}": copy.deepcopy(template) for index in range(count)}
    return raw


def _capabilities(count):
    return [
        f"CAP_{chr(65 + index // 26)}{chr(65 + index % 26)}" for index in range(count)
    ]


def test_env_spec_accepts_every_documented_maximum():
    raw = _services(helpers.make_env_spec(), env.MAX_ENV_SERVICES)
    raw["volumes"] = {"shared": {"seeded": True}} | {
        f"v{index}": {} for index in range(1, env.MAX_ENV_VOLUMES)
    }
    raw["services"]["svc0"].update(
        env={f"K{index}": "v" for index in range(256)},
        group_add=[f"g{index}" for index in range(16)],
        aliases=[f"alias{index}" for index in range(16)],
        extra_hosts=[[f"h{index}", "203.0.113.7"] for index in range(64)],
        cap_drop=_capabilities(64),
        tmpfs={f"/t{index}": 1 for index in range(16)},
        mounts=[
            {"volume": f"v{1 + index % 7}", "target": f"/m{index}"}
            for index in range(16)
        ],
        depends_on={f"svc{index}": {} for index in range(1, env.MAX_ENV_SERVICES)},
        stop_grace_sec=30,
    )
    spec = env.parse_env_spec(raw)
    assert len(spec.services) == env.MAX_ENV_SERVICES
    assert len(spec.volumes) == env.MAX_ENV_VOLUMES
    assert len(spec.services["svc0"].depends_on) == env.MAX_ENV_SERVICES - 1


def _mutations():
    """Each case: (mutation, exact error field, message fragment).

    Fixtures stay minimal so only the rule under test can fire.
    """

    def service(name, **updates):
        def apply(raw):
            raw["services"][name].update(updates)
            return raw

        return apply

    def top(**updates):
        def apply(raw):
            raw.update(updates)
            return raw

        return apply

    def cycle(raw):
        raw["services"]["kv"]["depends_on"] = {"main": {"condition": "started"}}
        return raw

    def tmpfs(path):
        return service("main", tmpfs={path: 1}, mounts=[])

    main, kv = "spec.services.main", "spec.services.kv"
    reserved = "reserved mount target"
    extra = "Extra inputs are not permitted"
    absolute = "normalized absolute path"
    return {
        "depends_on_cycle": (cycle, "spec", "depends_on cycle"),
        "self_dependency": (
            service("kv", depends_on={"kv": {}}),
            "spec",
            "invalid dependency kv",
        ),
        "unknown_dependency": (
            service("main", depends_on={"db": {}}),
            "spec",
            "invalid dependency db",
        ),
        "healthy_without_healthcheck": (
            service("kv", healthcheck=None),
            "spec",
            "healthy dependency kv declares no healthcheck",
        ),
        "healthy_on_disabled_healthcheck": (
            service("kv", healthcheck="none"),
            "spec",
            "healthy dependency kv declares no healthcheck",
        ),
        "nvidia_env": (
            service("main", env={"NVIDIA_VISIBLE_DEVICES": "all"}),
            f"{main}.env",
            "NVIDIA_VISIBLE_DEVICES is reserved",
        ),
        "nvidia_env_lowercase": (
            service("main", env={"nvidia_driver_capabilities": "x"}),
            f"{main}.env",
            "nvidia_driver_capabilities is reserved",
        ),
        "tmpfs_root": (tmpfs("/"), main, f"{reserved} /"),
        "tmpfs_on_proc": (tmpfs("/proc"), main, f"{reserved} /proc"),
        "tmpfs_below_proc": (tmpfs("/proc/sys"), main, f"{reserved} /proc/sys"),
        "tmpfs_sys": (tmpfs("/sys/fs"), main, f"{reserved} /sys/fs"),
        "tmpfs_dev_shm": (tmpfs("/dev/shm"), main, f"{reserved} /dev/shm"),
        "tmpfs_endpoint": (
            tmpfs("/run/rsi-harness/sandbox"),
            main,
            f"{reserved} /run/rsi-harness/sandbox",
        ),
        "tmpfs_relative": (tmpfs("scratch"), main, absolute),
        "tmpfs_unnormalized": (tmpfs("/scratch/"), main, absolute),
        "tmpfs_over_mount": (
            service("main", tmpfs={"/data/cache": 1}),
            main,
            "overlapping mount target /data",
        ),
        "tmpfs_and_shm_exceed_memory": (
            service("main", tmpfs={"/scratch": 1000}),
            main,
            "tmpfs and shm_mb together exceed memory_mb",
        ),
        "mount_root": (
            service("kv", mounts=[{"volume": "shared", "target": "/"}]),
            f"{kv}.mounts.0.target",
            f"{reserved} /",
        ),
        "mount_reserved_target": (
            service("kv", mounts=[{"volume": "shared", "target": "/dev/disk"}]),
            f"{kv}.mounts.0.target",
            f"{reserved} /dev/disk",
        ),
        "undeclared_volume": (
            service("kv", mounts=[{"volume": "other", "target": "/other"}]),
            "spec",
            "kv: undeclared volume other",
        ),
        "too_many_services": (
            lambda raw: _services(raw, env.MAX_ENV_SERVICES + 1),
            "spec.services",
            "at most 8 items",
        ),
        "no_services": (top(services={}), "spec.services", "at least 1 item"),
        "too_many_volumes": (
            top(volumes={f"v{index}": {} for index in range(9)}),
            "spec.volumes",
            "at most 8 items",
        ),
        "env_over_256": (
            service("main", env={f"K{index}": "v" for index in range(257)}),
            f"{main}.env",
            "at most 256 items",
        ),
        "group_add_over_16": (
            service("main", group_add=[f"g{index}" for index in range(17)]),
            f"{main}.group_add",
            "at most 16 items",
        ),
        "aliases_over_16": (
            service("kv", aliases=[f"a{index}" for index in range(17)]),
            f"{kv}.aliases",
            "at most 16 items",
        ),
        "extra_hosts_over_64": (
            service(
                "main",
                extra_hosts=[[f"h{index}", "203.0.113.7"] for index in range(65)],
            ),
            f"{main}.extra_hosts",
            "at most 64 items",
        ),
        "cap_drop_over_64": (
            service("main", cap_drop=_capabilities(65)),
            f"{main}.cap_drop",
            "at most 64 items",
        ),
        "tmpfs_over_16": (
            service("main", tmpfs={f"/t{index}": 1 for index in range(17)}, mounts=[]),
            f"{main}.tmpfs",
            "at most 16 items",
        ),
        "mounts_over_16": (
            service(
                "main",
                mounts=[
                    {"volume": "shared", "target": f"/m{index}"} for index in range(17)
                ],
            ),
            f"{main}.mounts",
            "at most 16 items",
        ),
        "depends_on_over_7": (
            service("main", depends_on={f"s{index}": {} for index in range(8)}),
            f"{main}.depends_on",
            "at most 7 items",
        ),
        "duplicate_aliases": (
            service("kv", aliases=["cache", "cache"]),
            f"{kv}.aliases",
            "duplicate aliases",
        ),
        "duplicate_group_add": (
            service("main", group_add=["audio", "audio"]),
            f"{main}.group_add",
            "duplicate group_add",
        ),
        "duplicate_cap_drop": (
            service("main", cap_drop=["CAP_NET_RAW", "CAP_NET_RAW"]),
            f"{main}.cap_drop",
            "duplicate cap_drop",
        ),
        "unknown_env_field": (top(privileged=True), "spec.privileged", extra),
        "unknown_service_field": (
            service("main", privileged=True),
            f"{main}.privileged",
            extra,
        ),
        "cap_add": (
            service("main", cap_add=["CAP_SYS_ADMIN"]),
            f"{main}.cap_add",
            extra,
        ),
        "devices": (
            service("main", devices=["/dev/nvidia0"]),
            f"{main}.devices",
            extra,
        ),
        "ports": (service("main", ports=["8080:80"]), f"{main}.ports", extra),
        "runtime": (service("main", runtime="nvidia"), f"{main}.runtime", extra),
        "alias_duplicates_service_name": (
            service("kv", aliases=["MAIN"]),
            "spec",
            "kv: alias MAIN already names main",
        ),
        "alias_across_services": (
            service("main", aliases=["kvstore"]),
            "spec",
            "kv: alias kvstore already names main",
        ),
        "alias_on_isolated_service": (
            service("kv", network="none"),
            kv,
            "a service without a network cannot have aliases",
        ),
        "host_gateway": (
            service("main", extra_hosts=[["host.docker.internal", "host-gateway"]]),
            f"{main}.extra_hosts",
            "expected an IP literal (no host-gateway)",
        ),
        "scoped_ipv6": (
            service("main", extra_hosts=[["peer", "fe80::1%eth0"]]),
            f"{main}.extra_hosts",
            "scoped addresses are refused",
        ),
        "relative_working_dir": (
            service("main", working_dir="app"),
            f"{main}.working_dir",
            absolute,
        ),
        "cpus_granularity": (
            service("main", cpus=0.015),
            f"{main}.cpus",
            "cpus must be a multiple of 0.01",
        ),
        "cpus_coerced": (service("main", cpus="1"), f"{main}.cpus", "valid number"),
        "memory_coerced": (
            service("main", memory_mb=True),
            f"{main}.memory_mb",
            "valid integer",
        ),
        "image_not_handle": (
            service("main", image="sha256:" + "a" * 64),
            f"{main}.image",
            "should match pattern",
        ),
        "image_by_name": (
            service("main", image="busybox:1.37.0"),
            f"{main}.image",
            "should match pattern",
        ),
        "network_host": (
            top(network="host"),
            "spec.network",
            "'public', 'none' or 'allowlist'",
        ),
        "version_2": (top(version=2), "spec.version", "less than or equal to 1"),
        "stop_grace_over_30": (
            service("main", stop_grace_sec=31),
            f"{main}.stop_grace_sec",
            "less than or equal to 30",
        ),
        "stop_grace_fraction": (
            service("main", stop_grace_sec=1.5),
            f"{main}.stop_grace_sec",
            "valid integer",
        ),
        "healthcheck_none_test": (
            service("kv", healthcheck={"test": ["NONE", "x"]}),
            f"{kv}.healthcheck.EnvHealthcheck.test",
            "test must be",
        ),
        "healthcheck_shell_args": (
            service("kv", healthcheck={"test": ["CMD-SHELL", "a", "b"]}),
            f"{kv}.healthcheck.EnvHealthcheck.test",
            "test must be",
        ),
        "healthcheck_retries": (
            service("kv", healthcheck={"test": ["CMD", "true"], "retries": 101}),
            f"{kv}.healthcheck.EnvHealthcheck.retries",
            "less than or equal to 100",
        ),
        "volume_name_uppercase": (
            top(volumes={"Shared": {}}),
            "spec.volumes.Shared.[key]",
            "should match pattern",
        ),
        "user_injection": (
            service("main", user="root;id"),
            f"{main}.user",
            "should match pattern",
        ),
        "nofile_over_cap": (
            service("main", nofile=env.ENV_NOFILE_CAP + 1),
            f"{main}.nofile",
            "less than or equal to",
        ),
        "lifetime_infinite": (
            top(lifetime_sec=float("inf")),
            "spec",
            "expected a JSON EnvSpec object",
        ),
    }


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_env_spec_rejects_unsafe_or_ambiguous_shapes(name):
    mutate, field, fragment = _mutations()[name]
    with pytest.raises(SandboxError) as caught:
        env.parse_env_spec(mutate(helpers.make_env_spec()))
    assert (caught.value.code, caught.value.field) == ("invalid", field)
    assert fragment in caught.value.message


def _command_spec(total):
    """main's entrypoint, command and env carry exactly ``total`` bytes."""
    entrypoint, key, value = "e" * 1000, "BLOB", "v" * 20000
    command = "c" * (total - len(entrypoint) - len(key) - len(value))
    return helpers.make_env_spec(
        entrypoint=[entrypoint], command=[command], env={key: value}
    )


def test_argv_and_env_are_a_64_kib_quota_like_a_v1_command():
    spec = env.parse_env_spec(_command_spec(env.ENV_COMMAND_BYTES))
    assert len(spec.services["main"].entrypoint[0]) == 1000
    # UTF-8 bytes, not characters.
    env.parse_env_spec(helpers.make_env_spec(command=["\u00e9" * 32768], env={}))
    for raw in (
        _command_spec(env.ENV_COMMAND_BYTES + 1),
        helpers.make_env_spec(command=["\u00e9" * 32768 + "x"], env={}),
    ):
        with pytest.raises(SandboxError) as caught:
            env.parse_env_spec(raw)
        assert (caught.value.code, caught.value.field) == (
            "quota",
            "spec.services.main",
        )
        assert "argv and env exceed 64 KiB" in caught.value.message
    # A malformed spec outranks an oversized one.
    raw = _command_spec(env.ENV_COMMAND_BYTES + 1)
    raw["services"]["kv"]["privileged"] = True
    with pytest.raises(SandboxError) as caught:
        env.parse_env_spec(raw)
    assert (caught.value.code, caught.value.field) == (
        "invalid",
        "spec.services.kv.privileged",
    )


@pytest.mark.parametrize("raw", [None, [], "spec", {"services": {1, 2}}])
def test_env_spec_requires_a_json_object(raw):
    with pytest.raises(SandboxError, match="invalid"):
        env.parse_env_spec(raw)


def test_spec_errors_never_echo_env_or_argv_values():
    secret = "sk-live-" + "9" * 24
    for raw in (
        helpers.make_env_spec(env={"NVIDIA_TOKEN": secret}),
        helpers.make_env_spec(command=[secret * 3000]),
        helpers.make_env_spec(env={"KEY": secret + "\x00"}),
    ):
        with pytest.raises(SandboxError) as caught:
            env.parse_env_spec(raw)
        assert secret not in str(caught.value)


def owner(phase="judge", run_id="run-1", task_id="task"):
    return SandboxOwner(
        run_id=run_id,
        task_id=task_id,
        phase=phase,
        round_id="agent-1" if phase == "judge" else None,
    )


def make_env_lease(env_id=ENV_ID, **updates):
    values = dict(
        owner=owner(),
        env_id=env_id,
        spec_sha256="d" * 64,
        state="ready",
        created_at=100.0,
        expires_at=700.0,
        network_mode="public",
        network_name=env.env_network_name(env_id),
        network_id="a" * 64,
        rule_id=env.env_rule_id("run-1", env_id),
        cpus_milli=1500,
        memory_mb=1280,
        disk_mb=2048,
        services=(
            env.SandboxEnvServiceLease(
                idx=0,
                name="kv",
                planned_name=env.env_container_name(env_id, 0),
                image="i" + "b" * 32,
                image_id=IMAGE_ID,
                container_id="b" * 64,
                state="running",
            ),
            env.SandboxEnvServiceLease(
                idx=1,
                name="main",
                planned_name=env.env_container_name(env_id, 1),
                image="i" + "a" * 32,
                image_id=IMAGE_ID,
                container_id="c" * 64,
                state="running",
            ),
        ),
        volumes=(
            env.SandboxEnvVolumeLease(
                idx=0,
                logical="shared",
                planned_name=env.env_volume_name(env_id, 0),
                created=True,
            ),
            env.SandboxEnvVolumeLease(
                idx=1,
                logical="implicit:0:" + "e" * 16,
                planned_name=env.env_volume_name(env_id, 1),
                created=True,
            ),
        ),
    )
    values.update(updates)
    return env.SandboxEnvLease(**values)


def removed_env_lease(env_id=ENV_ID, **updates):
    """Proven removal: every service, volume and the bridge are gone."""
    live = make_env_lease(env_id)
    values = dict(
        state="removed",
        network_id=None,
        services=tuple(
            service.model_copy(update={"state": "removed"}) for service in live.services
        ),
        volumes=tuple(
            volume.model_copy(update={"created": False}) for volume in live.volumes
        ),
    )
    values.update(updates)
    return make_env_lease(env_id, **values)


def make_image_lease(handle="i" + "2" * 32, **updates):
    values = dict(
        owner=owner(),
        handle=handle,
        kind="built",
        image_id=IMAGE_ID,
        tag=env.built_image_tag("run-1", handle),
        state="present",
        bytes=4096,
    )
    values.update(updates)
    return env.SandboxImageLease(**values)


def pulled_image_lease(handle="i" + "4" * 32, **updates):
    values = dict(kind="pulled", tag=None, pre_existing=True, bytes=0)
    values.update(updates)
    return make_image_lease(handle, **values)


def make_builder_lease(builder_id="b" + "3" * 32, **updates):
    values = dict(
        owner=owner(),
        builder_id=builder_id,
        container_name=env.builder_container_name(builder_id),
        volume_name=env.builder_volume_name(builder_id),
        network_name=env.builder_network_name(builder_id),
        rule_id=env.builder_rule_id("run-1", builder_id),
        state_fs="loop-ext4",
        loop_device="/dev/loop7",
        network_id="e" * 64,
        container_id="f" * 64,
        state="running",
        cpus=4,
        memory_mb=8192,
        disk_mb=6144,
    )
    values.update(updates)
    return env.BuilderLease(**values)


def removed_builder_lease(builder_id="b" + "3" * 32, **updates):
    values = dict(state="removed", loop_device=None, network_id=None)
    values.update(updates)
    return make_builder_lease(builder_id, **values)


def test_deterministic_names_follow_the_single_naming_table():
    assert env.env_container_name(ENV_ID, 3) == "rsi-sbx-1111111111111111-3"
    assert env.env_volume_name(ENV_ID, 0) == "rsi-sbvol-1111111111111111-0"
    assert env.env_network_name(ENV_ID) == "rsi-sbnet-1111111111111111"
    assert env.env_rule_id("run-1", ENV_ID) == "rsi-run-1-sbx-1111111111111111"
    builder = "b" + "3" * 32
    assert env.builder_container_name(builder) == "rsi-sbb-" + "3" * 16
    assert env.builder_volume_name(builder) == "rsi-sbbvol-" + "3" * 16
    assert env.builder_network_name(builder) == "rsi-sbbnet-" + "3" * 16
    assert env.builder_rule_id("run-1", builder) == "rsi-run-1-sbb-" + "3" * 16
    # Derived under the recovering data root, never read from the journal.
    assert env.builder_loop_file(Path("/data"), "run-1", builder) == Path(
        "/data/run-1/sb/build/" + "3" * 16 + ".img"
    )
    tag = env.built_image_tag("run-1", "i" + "2" * 32)
    assert tag.startswith("rsi-sbx-img:") and tag.endswith("-" + "2" * 32)
    assert len(tag.split(":")[1].split("-")[0]) == 12


def test_schema_five_lease_loads_through_the_single_migration(tmp_path):
    store = LeaseStore(tmp_path)
    raw = make_lease("old").model_dump(mode="json")
    raw["schema_version"] = 5
    for key in ("sandbox_envs", "sandbox_images", "sandbox_builders"):
        raw.pop(key)
    raw["sandbox_reservation"] = {
        "cpus": 6,
        "memory_mb": 4000,
        "pool_cpus": 8,
        "pool_memory_mb": 8192,
    }
    path = store.path_for("old")
    path.write_text(json.dumps(raw))
    before = path.read_bytes()
    lease = store.read("old")
    assert lease.schema_version == 6
    assert (lease.sandbox_envs, lease.sandbox_images, lease.sandbox_builders) == (
        (),
        (),
        (),
    )
    assert lease.sandbox_reservation.disk_mb == 0
    assert lease.sandbox_reservation.pool_disk_mb is None
    assert path.read_bytes() == before
    store.write(lease)
    assert json.loads(path.read_text())["schema_version"] == 6


@pytest.mark.parametrize("key", ["sandbox_envs", "sandbox_images", "sandbox_builders"])
def test_schema_five_cannot_smuggle_environment_authority(tmp_path, key):
    store = LeaseStore(tmp_path)
    raw = make_lease().model_dump(mode="json")
    raw.update({"schema_version": 5, key: [{"env_id": "hidden"}]})
    store.path_for("run-1").write_text(json.dumps(raw))
    with pytest.raises((ValidationError, ValueError), match="schema 5"):
        store.read("run-1")


def test_env_image_and_builder_records_round_trip_through_the_store(tmp_path):
    store = LeaseStore(tmp_path)
    lease = make_lease().model_copy(
        update={
            "sandbox_envs": (make_env_lease(),),
            "sandbox_images": (make_image_lease(), pulled_image_lease()),
            "sandbox_builders": (make_builder_lease(),),
        }
    )
    store.write(lease)
    assert store.read("run-1") == lease


def test_journal_records_survive_redaction_with_keyword_names(tmp_path):
    """Redaction rewrites values like ``token: x``; no record field has one."""
    run_id = "secret-token"
    store = LeaseStore(tmp_path)
    keyed = owner(run_id=run_id)
    live = make_env_lease(owner=keyed, rule_id=env.env_rule_id(run_id, ENV_ID))
    record = live.model_copy(
        update={
            "services": tuple(
                service.model_copy(update={"name": name})
                for service, name in zip(live.services, ("password", "token"))
            ),
            "volumes": (
                live.volumes[0].model_copy(update={"logical": "credential"}),
                live.volumes[1],
            ),
        }
    )
    handle = "i" + "2" * 32
    builder = "b" + "3" * 32
    lease = make_lease(run_id).model_copy(
        update={
            "sandbox_envs": (record,),
            "sandbox_images": (
                make_image_lease(
                    handle, owner=keyed, tag=env.built_image_tag(run_id, handle)
                ),
                pulled_image_lease(owner=keyed),
            ),
            "sandbox_builders": (
                make_builder_lease(
                    builder, owner=keyed, rule_id=env.builder_rule_id(run_id, builder)
                ),
            ),
        }
    )
    store.write(lease)
    assert store.read(run_id) == lease


def test_pulled_image_record_stores_no_caller_reference():
    for ref in (
        "docker.io/acme/secret:latest",
        "ghcr.io/acme/token:v1",
        "docker.io/library/busybox:1.37.0",
    ):
        with pytest.raises(ValidationError, match="store no reference"):
            pulled_image_lease(tag=ref)


def test_store_refuses_a_redaction_it_could_not_read_back(tmp_path):
    store = LeaseStore(tmp_path)
    lease = make_lease()
    store.write(lease)
    path = store.path_for("run-1")
    before = path.read_bytes()
    # Valid authority whose redacted form ("password:[REDACTED]") is not.
    keyed = "docker.io/acme/password:v1@sha256:" + "a" * 64
    with pytest.raises(ValidationError, match="cleanup image"):
        store.write(lease.model_copy(update={"cleanup_image_ref": keyed}))
    assert path.read_bytes() == before
    assert store.read("run-1") == lease
    assert not list(tmp_path.glob("*.tmp"))


def _lease_mutations():
    """Each case reaches one invariant alone; the fragment names that check."""
    services = make_env_lease().services
    kv, main = services
    names = "service planned name must match env identity"
    ordered = "env services must be unique and ordered by name"
    paired = "env bridge and firewall rule must be planned together"
    removed_env = "removed env cannot retain"
    removed_builder = "removed builder cannot retain"
    return {
        "service_name": (
            lambda: make_env_lease(
                services=(kv.model_copy(update={"planned_name": "rsi-sbx-x-0"}),)
            ),
            names,
        ),
        "service_index": (
            lambda: make_env_lease(
                services=(kv.model_copy(update={"idx": 1}), main),
            ),
            names,
        ),
        # Positions match idx and planned names; only the ordering is wrong.
        "service_order": (
            lambda: make_env_lease(
                services=(
                    main.model_copy(update={"idx": 0, "planned_name": kv.planned_name}),
                    kv.model_copy(update={"idx": 1, "planned_name": main.planned_name}),
                )
            ),
            ordered,
        ),
        "duplicate_service": (
            lambda: make_env_lease(
                services=(kv, main.model_copy(update={"name": "kv"})),
            ),
            ordered,
        ),
        "volume_name": (
            lambda: make_env_lease(
                volumes=(
                    env.SandboxEnvVolumeLease(
                        idx=0, logical="shared", planned_name="rsi-sbvol-other-0"
                    ),
                )
            ),
            "volume planned name must match env identity",
        ),
        "duplicate_volume": (
            lambda: make_env_lease(
                volumes=tuple(
                    env.SandboxEnvVolumeLease(
                        idx=index,
                        logical="shared",
                        planned_name=env.env_volume_name(ENV_ID, index),
                    )
                    for index in range(2)
                )
            ),
            "duplicate env volume",
        ),
        "implicit_volume_form": (
            lambda: env.SandboxEnvVolumeLease(
                idx=0, logical="implicit:8:" + "e" * 16, planned_name="x"
            ),
            "should match pattern",
        ),
        "foreign_rule": (
            lambda: make_env_lease(rule_id=env.env_rule_id("run-2", ENV_ID)),
            "env network and rule must match env identity",
        ),
        "public_without_bridge": (
            lambda: make_env_lease(network_name=None, network_id=None, rule_id=None),
            "public env requires its private firewalled bridge",
        ),
        "bridge_without_rule": (lambda: make_env_lease(rule_id=None), paired),
        "rule_without_bridge": (
            lambda: make_env_lease(
                network_mode="none", network_name=None, network_id=None
            ),
            paired,
        ),
        "network_id_without_bridge": (
            lambda: make_env_lease(
                network_mode="none", network_name=None, rule_id=None
            ),
            "env network identity requires a planned bridge",
        ),
        "service_active_without_container": (
            lambda: env.SandboxEnvServiceLease.model_validate(
                kv.model_dump() | {"container_id": None}
            ),
            "active service state requires actual container identity",
        ),
        "active_env_with_planned_service": (
            lambda: make_env_lease(
                services=(
                    kv.model_copy(update={"state": "planned", "container_id": None}),
                    main,
                )
            ),
            "active env state requires every container identity",
        ),
        "removed_with_live_service": (
            lambda: removed_env_lease(services=services),
            removed_env,
        ),
        "removed_with_created_volume": (
            lambda: removed_env_lease(volumes=make_env_lease().volumes),
            removed_env,
        ),
        "removed_with_network": (
            lambda: removed_env_lease(network_id="a" * 64),
            removed_env,
        ),
        "removed_with_pending_mutation": (
            lambda: removed_env_lease(pending_mutation=True),
            removed_env,
        ),
        "reason_on_ready_env": (
            lambda: make_env_lease(reason="expired"),
            "only an ending env can carry a reason",
        ),
        "expiry_before_creation": (
            lambda: make_env_lease(expires_at=50.0),
            "expires_at must follow created_at",
        ),
        "env_handle_form": (
            lambda: make_env_lease(env_id="x" + "1" * 32),
            "should match pattern",
        ),
        "built_tag_of_other_run": (
            lambda: make_image_lease(tag=env.built_image_tag("run-2", "i" + "2" * 32)),
            "built image tag must match run and handle",
        ),
        "built_pre_existing": (
            lambda: make_image_lease(pre_existing=True),
            "a built image cannot be pre-existing",
        ),
        "pulled_with_reference": (
            lambda: pulled_image_lease(
                tag=env.built_image_tag("run-1", "i" + "4" * 32)
            ),
            "pulled image records store no reference",
        ),
        "pulled_loading": (
            lambda: pulled_image_lease(state="loading"),
            "pulled images are never loaded or removed",
        ),
        "pulled_leaked": (
            lambda: pulled_image_lease(state="leaked"),
            "pulled images are never loaded or removed",
        ),
        "present_without_id": (
            lambda: make_image_lease(image_id=None),
            "present or leaked image requires actual image identity",
        ),
        "builder_name": (
            lambda: make_builder_lease(container_name="rsi-sbb-other"),
            "builder planned names must match builder identity",
        ),
        "builder_rule": (
            lambda: make_builder_lease(
                rule_id=env.builder_rule_id("run-2", "b" + "3" * 32)
            ),
            "builder planned names must match builder identity",
        ),
        "builder_stores_loop_file": (
            lambda: make_builder_lease(loop_file="/data/run-1/sb/build/x.img"),
            "Extra inputs are not permitted",
        ),
        "tmpfs_builder_loop": (
            lambda: make_builder_lease(state_fs="tmpfs"),
            "tmpfs builder state cannot own a loop device",
        ),
        "builder_loop_device": (
            lambda: make_builder_lease(loop_device="/dev/sda"),
            "should match pattern",
        ),
        "builder_without_container": (
            lambda: make_builder_lease(container_id=None),
            "active builder state requires actual container identity",
        ),
        "removed_builder_pending": (
            lambda: removed_builder_lease(pending_mutation=True),
            removed_builder,
        ),
        "removed_builder_loop": (
            lambda: removed_builder_lease(loop_device="/dev/loop7"),
            removed_builder,
        ),
        "removed_builder_network": (
            lambda: removed_builder_lease(network_id="e" * 64),
            removed_builder,
        ),
    }


@pytest.mark.parametrize("name", sorted(_lease_mutations()))
def test_lease_identity_is_deterministic_and_state_consistent(name):
    build, fragment = _lease_mutations()[name]
    with pytest.raises(ValidationError, match=re.escape(fragment)):
        build()


def test_proven_removal_records_are_valid():
    assert removed_env_lease().state == "removed"
    assert removed_builder_lease().state == "removed"
    assert removed_builder_lease(state_fs="tmpfs").loop_device is None
    assert make_image_lease(state="removed").state == "removed"
    assert pulled_image_lease(state="removed").state == "removed"


def _colliding(prefix):
    """Two full identities that share the 16 hex digits in every Docker name."""
    return tuple(prefix + "1" * 16 + digit * 16 for digit in "ab")


@pytest.mark.parametrize(
    "records,match",
    [
        (
            {
                "sandbox_envs": (
                    make_env_lease(
                        owner=owner(run_id="run-2"),
                        rule_id=env.env_rule_id("run-2", ENV_ID),
                    ),
                )
            },
            "environment owner identity differs",
        ),
        (
            {"sandbox_envs": (make_env_lease(owner=owner(task_id="other")),)},
            "environment owner identity differs",
        ),
        (
            {"sandbox_envs": (make_env_lease(), make_env_lease())},
            "duplicate sandbox environment identity",
        ),
        (
            {"sandbox_envs": tuple(map(make_env_lease, _colliding("e")))},
            "duplicate sandbox environment identity",
        ),
        (
            {"sandbox_images": (make_image_lease(), make_image_lease())},
            "duplicate sandbox image identity",
        ),
        (
            {"sandbox_builders": (make_builder_lease(), make_builder_lease())},
            "duplicate sandbox builder identity",
        ),
        (
            {"sandbox_builders": tuple(map(make_builder_lease, _colliding("b")))},
            "duplicate sandbox builder identity",
        ),
    ],
)
def test_resource_lease_rejects_foreign_or_duplicate_records(tmp_path, records, match):
    with pytest.raises(ValidationError, match=match):
        LeaseStore(tmp_path).write(make_lease().model_copy(update=records))


def test_release_waits_for_env_built_image_and_builder_cleanup(tmp_path):
    store = LeaseStore(tmp_path)
    mutate = authority(store)
    pool = SandboxAdmissionPool(store)
    pool.reserve_run("run-1", helpers.make_env_grant(tmp_path), mutate)
    assert store.read("run-1").sandbox_env_authority
    for records in (
        {"sandbox_envs": (make_env_lease(),)},
        {"sandbox_envs": (make_env_lease(pending_mutation=True, state="failed"),)},
        {"sandbox_images": (make_image_lease(),)},
        # An rmi conflict still holds disk counted in the reservation.
        {"sandbox_images": (make_image_lease(state="leaked"),)},
        {"sandbox_builders": (make_builder_lease(),)},
    ):
        mutate(lambda lease, records=records: lease.model_copy(update=records))
        with pytest.raises(InfrastructureError, match="environment, image"):
            pool.release_run("run-1", mutate)
        assert store.read("run-1").sandbox_reservation is not None
        mutate(
            lambda lease: lease.model_copy(
                update={
                    "sandbox_envs": (),
                    "sandbox_images": (),
                    "sandbox_builders": (),
                }
            )
        )
    # Proven removal releases; a pulled image is a cache outside the pool.
    mutate(
        lambda lease: lease.model_copy(
            update={
                "sandbox_envs": (removed_env_lease(),),
                "sandbox_images": (
                    make_image_lease(state="removed"),
                    pulled_image_lease(),
                ),
                "sandbox_builders": (removed_builder_lease(),),
            }
        )
    )
    pool.release_run("run-1", mutate)
    released = store.read("run-1")
    assert released.sandbox_reservation is None
    # Env authority outlives the reservation: recovery keeps sweeping.
    assert released.sandbox_env_authority


def _recovery(tmp_path, **records):
    original = parent_lease(tmp_path)

    def owned(record):
        return record.model_copy(
            update={
                "owner": record.owner.model_copy(update={"task_id": original.task_id})
            }
        )

    reservation = SandboxReservation(
        cpus=4, memory_mb=4096, pool_cpus=8, pool_memory_mb=8192
    )
    store = LeaseStore(tmp_path / "leases")
    store.write(
        original.model_copy(
            update={
                **{key: tuple(map(owned, value)) for key, value in records.items()},
                "sandbox_reservation": reservation,
            }
        )
    )
    manager = RecoveryManager(
        store=store, backend=ChildRecoveryBackend(), managed_root=tmp_path / "runs"
    )
    return manager, store


def test_recovery_releases_proven_removed_records(tmp_path):
    manager, store = _recovery(
        tmp_path,
        sandbox_envs=(removed_env_lease(),),
        sandbox_images=(make_image_lease(state="removed"), pulled_image_lease()),
        sandbox_builders=(removed_builder_lease(),),
    )
    assert manager.recover("run-1") == ("run-1",)
    recovered = store.read("run-1")
    assert not recovered.recovery_required
    assert recovered.sandbox_reservation is None


def _dump_toml(value, prefix=""):
    """Minimal TOML writer for the policy shapes used above."""
    scalars, tables = [], []
    for key, item in value.items():
        if isinstance(item, dict):
            tables.append((key, item))
        else:
            scalars.append(f"{key} = {json.dumps(item)}")
    text = "\n".join(scalars) + "\n"
    for key, item in tables:
        name = f"{prefix}.{key}" if prefix else key
        text += f"\n[{name}]\n" + _dump_toml(item, name)
    return text


def test_minimal_toml_writer_round_trips_the_documented_policy():
    raw = tomllib.loads(helpers.env_policy_toml())
    assert tomllib.loads(_dump_toml(raw)) == raw


# -- allowlist ----------------------------------------------------------------------

ALLOWLIST = {"max_entries": 4, "patterns": ["*.pypi.org"], "private_cidrs": []}


def test_allowlist_needs_both_keys_and_the_operator_bounds(tmp_path):
    offered = helpers.make_env_policy(tmp_path, allowlist=ALLOWLIST)
    task = task_with(tmp_path, network=("allowlist",))

    granted = resolve(task, offered).environments.judge
    assert granted.network == ("allowlist",)
    assert (granted.allowlist.max_entries, granted.allowlist.patterns) == (
        4,
        ("*.pypi.org",),
    )
    assert granted.allowlist.refresh_sec == 60.0
    # The operator's bounds hold only where the task asked for allowlist.
    assert resolve(task_with(tmp_path), offered).environments.judge.allowlist is None
    with pytest.raises(SetupError, match="judge.network: allowlist not approved"):
        validate_sandbox_policy(task, helpers.make_env_policy(tmp_path), "docker")


def test_builds_of_an_allowlist_phase_are_public_or_none(tmp_path):
    policy = helpers.make_env_policy(tmp_path, allowlist=ALLOWLIST)
    mixed = task_with(tmp_path, network=("allowlist", "none"), build=True)
    assert resolve(mixed, policy).environments.judge.build.network == ("none",)
    named = task_with(
        tmp_path, network=("allowlist",), build=True, build_network=("public",)
    )
    assert resolve(named, policy).environments.judge.build.network == ("public",)
    with pytest.raises(SetupError, match="judge.build_network: builds are public"):
        validate_sandbox_policy(
            task_with(tmp_path, network=("allowlist",), build=True), policy, "docker"
        )
    with pytest.raises(ValidationError):
        task_with(tmp_path, build=True, build_network=("allowlist",))


@pytest.mark.parametrize(
    ("change", "match"),
    (
        ({"network": ["public", "none", "allowlist"]}, "go together"),
        ({"allowlist": {}}, "go together"),
        ({"allowlist": {"max_entries": 65}}, "less than or equal to 64"),
        ({"allowlist": {"private_cidrs": ["169.254.0.0/16"]}}, "not inside"),
        ({"allowlist": {"private_cidrs": ["8.8.8.0/24"]}}, "not inside"),
        ({"allowlist": {"private_cidrs": ["10.0.0.1/8"]}}, "host bits"),
        ({"allowlist": {"patterns": ["exa mple.com"]}}, "hostname glob"),
        ({"allowlist": {"refresh_sec": 1}}, "greater than or equal to 5"),
    ),
)
def test_operator_allowlist_bounds_are_validated(tmp_path, change, match):
    raw = tomllib.loads(helpers.env_policy_toml())
    raw["environments"]["judge"].update(change)
    with pytest.raises(SetupError, match=match):
        helpers.load_policy_text(tmp_path, _dump_toml(raw))


@pytest.mark.parametrize(
    ("entries", "normalized"),
    (
        (["PyPI.org.", "pypi.org:443"], ("pypi.org", "pypi.org:443")),
        (["8.8.8.8/32", "1.1.1.0/24:53"], ("8.8.8.8", "1.1.1.0/24:53")),
        ([], ()),
    ),
)
def test_spec_allowlist_entries_are_normalized(entries, normalized):
    raw = {**helpers.make_env_spec(), "network": "allowlist", "allowlist": entries}

    assert env.parse_env_spec(raw).allowlist == normalized


@pytest.mark.parametrize(
    ("network", "entries"),
    (
        ("public", ["pypi.org"]),
        ("allowlist", ["*.pypi.org"]),
        ("allowlist", ["2001:db8::1"]),
        ("allowlist", ["1.2.3.4/8"]),
        ("allowlist", ["pypi.org:0"]),
        ("allowlist", ["https://pypi.org"]),
        ("allowlist", ["pypi.org", "PYPI.org"]),
        ("allowlist", [f"h{index}.example" for index in range(65)]),
    ),
)
def test_spec_refuses_malformed_or_misplaced_allowlists(network, entries):
    raw = {**helpers.make_env_spec(), "network": network, "allowlist": entries}

    with pytest.raises(SandboxError) as caught:
        env.parse_env_spec(raw)
    assert caught.value.code in ("invalid", "quota")


def test_an_allowlist_lease_needs_its_bridge():
    make_env_lease(network_mode="allowlist")
    with pytest.raises(ValidationError, match="allowlist env requires"):
        make_env_lease(
            network_mode="allowlist", network_name=None, network_id=None, rule_id=None
        )


def test_the_allowlist_enters_the_spec_digest_only_when_listed():
    plain = env.parse_env_spec(helpers.make_env_spec())
    dumped = plain.model_dump(mode="json")
    assert dumped.pop("allowlist") == []
    canonical = json.dumps(dumped, sort_keys=True, separators=(",", ":"))
    # Specs without an allowlist keep the digest journaled before it existed.
    assert env.env_spec_digest(plain) == hashlib.sha256(canonical.encode()).hexdigest()
    raw = {**helpers.make_env_spec(), "network": "allowlist"}
    digests = {
        env.env_spec_digest(env.parse_env_spec({**raw, "allowlist": entries}))
        for entries in ([], ["pypi.org"], ["pypi.org:443"])
    }
    assert len(digests) == 3


# -- operator tools ------------------------------------------------------------------

TMUX = {"path": "/opt/rsi/tools/tmux", "sha256": "a" * 64}


def test_the_operator_tmux_is_a_host_path_and_hash_the_grant_carries(tmp_path):
    assert helpers.make_env_policy(tmp_path).environments.host.tmux is None
    policy = helpers.make_env_policy(tmp_path, tmux=TMUX)
    assert policy.environments.host.tmux.model_dump() == TMUX
    grant = resolve(helpers.make_env_task(), policy)
    # The broker reads it from the persisted grant, never from the task.
    assert grant.environments.host.tmux == policy.environments.host.tmux
    assert SandboxEnvGrant.model_validate_json(grant.model_dump_json()) == grant


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"path": "tools/tmux"}, "path"),
        ({"path": "/opt/../tmux"}, "path"),
        ({"path": "/opt//tmux"}, "path"),
        ({"sha256": "A" * 64}, "sha256"),
        ({"sha256": "a" * 63}, "sha256"),
        ({"sha256": None}, "sha256"),
        ({"url": "https://example.com/tmux"}, "url"),
    ],
)
def test_the_operator_tmux_table_is_validated(tmp_path, change, match):
    table = {key: value for key, value in (TMUX | change).items() if value}
    with pytest.raises(SetupError, match=match):
        helpers.make_env_policy(tmp_path, tmux=table)
