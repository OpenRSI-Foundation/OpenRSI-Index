"""Pure opt-in authorization; no Docker, inventory, authentication or writes."""

from __future__ import annotations

import copy
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from rsi_harness.errors import SetupError, UnsupportedTaskError
from rsi_harness.runtime.sandbox_contracts import (
    BROKER_HEADROOM_MB,
    IMAGE_ID,
    EnvGrant,
    EnvLimits,
    EnvPolicy,
    SandboxEnvGrant,
    SandboxEnvTask,
    SandboxGrant,
    SandboxLimits,
    SandboxPolicy,
    SandboxTask,
    env_metadata_mb,
)

BUILDER_ENTRYPOINT = "buildkitd"
BUILDER_STATE_DIR = "/var/lib/buildkit"


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _tuples(item) for key, item in value.items()}
    return value


def _normalize(raw: Any) -> Any:
    """Normalize TOML collections only; never coerce scalar values."""
    result = copy.deepcopy(raw)
    if not isinstance(result, dict):
        return result
    if isinstance(result.get("environments"), dict):
        # Environment tables hold only scalar arrays: every array is a tuple.
        result["environments"] = _tuples(result["environments"])
    profiles = result.get("profiles")
    if isinstance(profiles, list):
        for profile in profiles:
            if isinstance(profile, dict) and isinstance(profile.get("tmpfs_mb"), dict):
                profile["tmpfs_mb"] = tuple(profile["tmpfs_mb"].items())
        result["profiles"] = tuple(profiles)
    for phase in ("work", "judge"):
        value = result.get(phase)
        if isinstance(value, dict) and isinstance(value.get("profiles"), list):
            value["profiles"] = tuple(value["profiles"])
    return result


def parse_sandbox_task(raw: Any) -> SandboxTask | SandboxEnvTask:
    version = raw.get("version") if isinstance(raw, dict) else None
    model = SandboxEnvTask if type(version) is int and version == 2 else SandboxTask
    try:
        return model.model_validate(_normalize(raw))
    except ValidationError as error:
        raise UnsupportedTaskError(f"metadata.rsi_harness.sandbox: {error}") from error


def load_sandbox_policy(path: Path) -> SandboxPolicy:
    try:
        with Path(path).open("rb") as source:
            return SandboxPolicy.model_validate(_normalize(tomllib.load(source)))
    except (OSError, ValueError, ValidationError) as error:
        raise SetupError(f"sandbox policy {path}: {error}") from error


def require_within(field: str, requested: int | float, permitted: int | float) -> None:
    if requested > permitted:
        raise SetupError(
            f"sandbox.{field}: requested {requested}, operator permits {permitted}"
        )


def validate_sandbox_policy(
    task: SandboxTask | SandboxEnvTask | None,
    policy: SandboxPolicy | None,
    backend: str,
) -> SandboxTask | SandboxEnvTask | None:
    if task is None:
        return None
    if backend == "cluster":
        _require_cluster_e2b(task, getattr(policy, "environments", None))
    elif backend != "docker":
        raise SetupError(
            "sandbox requires local Linux Docker; this backend is unsupported"
        )
    if policy is None:
        raise SetupError(
            "sandbox requested but no operator policy supplied (--sandbox-policy PATH)"
        )
    if isinstance(task, SandboxEnvTask):
        _effective_environments(task, policy)
        return task
    approved = {profile.name: profile for profile in policy.profiles}
    for profile in task.profiles:
        if profile != approved.get(profile.name):
            raise SetupError(
                f"sandbox.profiles.{profile.name}: exact profile/image "
                "not approved by operator policy"
            )
    for phase_name in ("work", "judge"):
        requested = getattr(task, phase_name)
        if requested is None:
            continue
        permitted = getattr(policy, phase_name)
        if permitted is None:
            raise SetupError(
                f"sandbox.{phase_name}: phase not approved by operator policy"
            )
        if not set(requested.profiles).issubset(permitted.profiles):
            raise SetupError(
                f"sandbox.{phase_name}.profiles: requested profile not approved"
            )
        for field in SandboxLimits.model_fields:
            value = getattr(requested.limits, field)
            require_within(
                f"{phase_name}.limits.{field}", value, getattr(permitted.limits, field)
            )
            require_within(
                f"{phase_name}.run_limits.{field}",
                value,
                getattr(policy.run_limits, field),
            )
    return task


def _require_cluster_e2b(task: Any, environments: Any) -> None:
    """Clusters have no Docker daemon or root: only environment tasks whose
    envs the operator's policy sends to E2B (``backend = "e2b"``) run there."""
    if not (
        isinstance(task, SandboxEnvTask)
        and environments is not None
        and environments.host.backend == "e2b"
    ):
        raise SetupError(
            "sandbox requires local Linux Docker; on a cluster backend only "
            "environment tasks (version 2) with an operator policy of "
            '[environments.host] backend = "e2b" are supported'
        )


def validate_cluster_sandbox(
    task: SandboxTask | SandboxEnvTask | None,
    grant: SandboxGrant | SandboxEnvGrant | None,
    *,
    multi_node: bool,
) -> SandboxEnvGrant | None:
    """The cluster Engine's check of a frozen plan: no sandbox, or an E2B
    environment grant (resolved on the submit host). A multi-node Judge
    runs on remote hosts the broker's socket does not reach: refused."""
    if task is None and grant is None:
        return None
    environments = grant.environments if isinstance(grant, SandboxEnvGrant) else None
    _require_cluster_e2b(task, environments)
    assert isinstance(grant, SandboxEnvGrant)
    if multi_node and grant.environments.judge is not None:
        raise SetupError(
            "sandbox.environments.judge: unsupported on multi-node cluster runs "
            "(the Judge runs on remote hosts)"
        )
    return grant


def _effective_environments(task: SandboxEnvTask, policy: SandboxPolicy) -> EnvPolicy:
    """Both keys: a capability exists only if requested and operator-approved."""
    permitted = policy.environments
    if permitted is None:
        raise SetupError(
            "sandbox.environments: environments not approved by operator policy"
        )
    phases: dict[str, EnvGrant | None] = {}
    for phase_name in ("work", "judge"):
        requested = getattr(task.environments, phase_name)
        if requested is None:
            phases[phase_name] = None
            continue
        grant = getattr(permitted, phase_name)
        prefix = f"sandbox.environments.{phase_name}"
        if grant is None:
            raise SetupError(f"{prefix}: phase not approved by operator policy")
        denied = [mode for mode in requested.network if mode not in grant.network]
        if denied:
            raise SetupError(
                f"{prefix}.network: {', '.join(denied)} not approved by operator policy"
            )
        if requested.pull and not grant.pull:
            raise SetupError(f"{prefix}.pull: image pull not approved by operator")
        if requested.build and permitted.host.backend == "e2b":
            raise SetupError(
                f"{prefix}.build: image_build is unsupported on the e2b backend"
            )
        if requested.build and grant.build is None:
            raise SetupError(f"{prefix}.build: image build not approved by operator")
        effective = grant.model_dump()
        if requested.build:
            # Builds get only the networks the task asked for, never the
            # operator's whole build grant (and never allowlist).
            build_network = requested.build_network or tuple(
                mode for mode in requested.network if mode != "allowlist"
            )
            if not build_network:
                raise SetupError(
                    f"{prefix}.build_network: builds are public or none; "
                    "name one when network is only allowlist"
                )
            denied = [mode for mode in build_network if mode not in grant.build.network]
            if denied:
                raise SetupError(
                    f"{prefix}.build_network: {', '.join(denied)} not approved "
                    "for builds by operator policy"
                )
            effective["build"]["network"] = build_network
        for field in EnvLimits.model_fields:
            value = getattr(requested.limits, field)
            if value is None:
                continue  # an omitted limit takes the operator's value
            require_within(
                f"environments.{phase_name}.limits.{field}",
                value,
                getattr(grant, field),
            )
            effective[field] = value
        effective.update(
            network=requested.network,
            pull=requested.pull,
            registries=grant.registries if requested.pull else (),
            build=effective["build"] if requested.build else None,
            # The operator's bounds hold only for a requested allowlist.
            allowlist=(
                effective["allowlist"] if "allowlist" in requested.network else None
            ),
        )
        try:
            phases[phase_name] = EnvGrant.model_validate(effective)
        except ValidationError as error:
            raise SetupError(f"{prefix}.limits: {error}") from error
    return EnvPolicy(
        host=permitted.host,
        run_limits=permitted.run_limits,
        work=phases["work"],
        judge=phases["judge"],
    )


def _require_parent(parent_cpus: int | None, parent_memory_mb: int | None) -> None:
    for field, value in (("cpus", parent_cpus), ("memory_mb", parent_memory_mb)):
        if type(value) is not int or value <= 0:
            raise SetupError(
                f"sandbox parent.{field}: finite positive integer limit required"
            )


def resolve_sandbox_grant(
    task: SandboxTask,
    policy: SandboxPolicy,
    image_ids: dict[str, str],
    parent_cpus: int | None,
    parent_memory_mb: int | None,
) -> SandboxGrant:
    validate_sandbox_policy(task, policy, "docker")
    _require_parent(parent_cpus, parent_memory_mb)
    profiles = []
    for profile in task.profiles:
        image = image_ids.get(profile.name, "")
        if not IMAGE_ID.fullmatch(image):
            raise SetupError(
                f"sandbox.profiles.{profile.name}.image: "
                "unavailable or unresolved image"
            )
        if IMAGE_ID.fullmatch(profile.image) and image != profile.image:
            raise SetupError(
                f"sandbox.profiles.{profile.name}.image: "
                "resolved ID differs from approved ID"
            )
        profiles.append(profile.model_copy(update={"image": image}))
    # Conservative reservation: neither paused parent nor paused Work children
    # release memory during Judge. CPU is also reserved, rather than borrowed.
    phases = tuple(phase for phase in (task.work, task.judge) if phase is not None)
    cpus = 2 * parent_cpus + sum(phase.limits.max_cpus for phase in phases)
    # Records outlive completed operations/removed children across Judge rounds.
    # Reserve 64 KiB each for bounded replay diagnostics/fingerprints, or child
    # models and journal serialization copies; do not borrow transfer headroom.
    # Live backend image environments may expand from 64 KiB UTF-8 to wider
    # Python strings/dicts, so also reserve 512 KiB per live child.
    metadata_bytes = 64 * 1024 * (
        policy.run_limits.max_operations + policy.run_limits.max_created
    ) + 512 * 1024 * policy.run_limits.max_live
    metadata_mb = (metadata_bytes + 1024**2 - 1) // 1024**2
    memory = (
        2 * parent_memory_mb
        + sum(phase.limits.max_memory_mb for phase in phases)
        + BROKER_HEADROOM_MB
        + metadata_mb
    )
    require_within("pool_cpus", cpus, policy.pool_cpus)
    require_within("pool_memory_mb", memory, policy.pool_memory_mb)
    return SandboxGrant(
        profiles=tuple(profiles),
        work=task.work,
        judge=task.judge,
        run_limits=policy.run_limits,
        reserved_cpus=cpus,
        reserved_memory_mb=memory,
        pool_cpus=policy.pool_cpus,
        pool_memory_mb=policy.pool_memory_mb,
    )


def builder_image_id(field: str, reference: str, inspected: Any) -> str:
    """Check the cached builder image's inspect result; return its exact ID.

    The broker fixes buildkitd's arguments and mounts only its state volume, so
    the image must run buildkitd directly and may declare no other VOLUME (an
    unmounted VOLUME leaves an anonymous volume behind on removal).
    """
    image = inspected.get("Id") if isinstance(inspected, Mapping) else None
    if not isinstance(image, str) or not IMAGE_ID.fullmatch(image):
        raise SetupError(f"{field}: unavailable or unresolved image")
    if IMAGE_ID.fullmatch(reference):
        if image != reference:
            raise SetupError(f"{field}: resolved ID differs from approved ID")
    else:
        digest = reference.rsplit("@", 1)[1]
        digests = inspected.get("RepoDigests")
        if not isinstance(digests, list) or not any(
            isinstance(item, str) and "@" in item and item.rsplit("@", 1)[1] == digest
            for item in digests
        ):
            raise SetupError(f"{field}: cached image lacks the approved digest")
    config = inspected.get("Config")
    if not isinstance(config, Mapping) or config.get("Entrypoint") != [
        BUILDER_ENTRYPOINT
    ]:
        raise SetupError(f'{field}: image Entrypoint must be ["{BUILDER_ENTRYPOINT}"]')
    volumes = config.get("Volumes") or {}
    if not isinstance(volumes, Mapping) or set(volumes) - {BUILDER_STATE_DIR}:
        raise SetupError(
            f"{field}: image declares volumes other than {BUILDER_STATE_DIR}"
        )
    return image


def resolve_env_grant(
    task: SandboxEnvTask,
    policy: SandboxPolicy,
    builder_images: Mapping[str, Any],
    parent_cpus: int | None,
    parent_memory_mb: int | None,
) -> SandboxEnvGrant:
    """Resolve the effective environment grant and its host pool envelope.

    ``builder_images`` maps a phase with an effective build grant to the local
    ``docker image inspect`` result for its approved builder_image; the broker
    never pulls the builder.
    """
    validate_sandbox_policy(task, policy, "docker")
    environments = _effective_environments(task, policy)
    _require_parent(parent_cpus, parent_memory_mb)
    phases: dict[str, EnvGrant | None] = {}
    for phase_name in ("work", "judge"):
        grant = getattr(environments, phase_name)
        if grant is not None and grant.build is not None:
            image = builder_image_id(
                f"sandbox.environments.{phase_name}.build.builder_image",
                grant.build.builder_image,
                builder_images.get(phase_name),
            )
            build = grant.build.model_copy(update={"builder_image": image})
            grant = grant.model_copy(update={"build": build})
        phases[phase_name] = grant
    active = [grant for grant in phases.values() if grant is not None]
    builds = [grant.build for grant in active if grant.build is not None]
    # Same conservative parent envelope as profiles; each builder is reserved
    # in full, and its state fs plus loaded images count against the disk pool.
    cpus = (
        2 * parent_cpus
        + sum(grant.max_cpus_live for grant in active)
        + sum(build.cpus for build in builds)
    )
    memory = (
        2 * parent_memory_mb
        + sum(grant.max_memory_mb_live for grant in active)
        + sum(build.memory_mb for build in builds)
        + BROKER_HEADROOM_MB
        + env_metadata_mb(active, environments.host.waiters)
    )
    disk = sum(grant.max_disk_mb_live for grant in active) + sum(
        build.disk_mb + build.max_images_total_mb for build in builds
    )
    require_within("pool_cpus", cpus, policy.pool_cpus)
    require_within("pool_memory_mb", memory, policy.pool_memory_mb)
    require_within(
        "environments.host.pool_disk_mb", disk, environments.host.pool_disk_mb
    )
    return SandboxEnvGrant(
        environments=EnvPolicy(
            host=environments.host,
            run_limits=environments.run_limits,
            work=phases["work"],
            judge=phases["judge"],
        ),
        reserved_cpus=cpus,
        reserved_memory_mb=memory,
        reserved_disk_mb=disk,
        pool_cpus=policy.pool_cpus,
        pool_memory_mb=policy.pool_memory_mb,
    )
