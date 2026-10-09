import re
from pathlib import Path

import pytest
from harbor.models.task.config import TaskConfig

from rsi_harness.runtime.sandbox_policy import (
    load_sandbox_policy,
    parse_sandbox_task,
    resolve_sandbox_grant,
    validate_sandbox_policy,
)

DOC = Path(__file__).parents[1] / "docs" / "harbor-task-authoring" / "sandboxes.md"


def documented_toml(name: str) -> str:
    match = re.search(
        rf"<!-- {re.escape(name)} -->\s*```toml\n(.*?)\n```",
        DOC.read_text(),
        flags=re.DOTALL,
    )
    assert match is not None, f"missing {name} TOML example"
    return match.group(1)


def test_documented_task_and_policy_are_accepted_by_runtime_models(tmp_path):
    task_config = TaskConfig.model_validate_toml(documented_toml("sandbox-task"))
    sandbox = parse_sandbox_task(task_config.metadata["rsi_harness"]["sandbox"])
    profile = sandbox.profiles[0]
    assert task_config.environment.docker_image == profile.image
    assert task_config.environment.cpus == profile.cpus
    assert task_config.environment.memory_mb == profile.memory_mb
    assert task_config.environment.gpus == 0
    assert task_config.agent.user == task_config.verifier.user == "root"
    assert {
        task_config.environment.network_mode.value,
        task_config.agent.network_mode.value,
        task_config.verifier.network_mode.value,
    } == {"no-network"}

    policy_path = tmp_path / "sandbox-policy.toml"
    policy_path.write_text(documented_toml("sandbox-policy"))
    policy = load_sandbox_policy(policy_path)

    assert validate_sandbox_policy(sandbox, policy, "docker") == sandbox
    grant = resolve_sandbox_grant(
        sandbox,
        policy,
        image_ids={"offline-python": "sha256:" + "a" * 64},
        parent_cpus=task_config.environment.cpus,
        parent_memory_mb=task_config.environment.memory_mb,
    )
    assert grant.reserved_cpus == 5
    assert grant.reserved_memory_mb == 3338
    assert grant.work is not grant.judge


def test_documented_environment_task_and_policy_resolve_one_grant(tmp_path):
    from rsi_harness.runtime.sandbox_contracts import SandboxEnvTask
    from rsi_harness.runtime.sandbox_policy import resolve_env_grant

    task_config = TaskConfig.model_validate_toml(documented_toml("sandbox-env-task"))
    sandbox = parse_sandbox_task(task_config.metadata["rsi_harness"]["sandbox"])
    assert isinstance(sandbox, SandboxEnvTask)
    assert sandbox.environments.work is None
    assert not sandbox.environments.judge.build

    policy_path = tmp_path / "sandbox-policy.toml"
    policy_path.write_text(documented_toml("sandbox-env-policy"))
    policy = load_sandbox_policy(policy_path)
    assert validate_sandbox_policy(sandbox, policy, "docker") == sandbox
    grant = resolve_env_grant(
        sandbox,
        policy,
        {},
        parent_cpus=task_config.environment.cpus,
        parent_memory_mb=task_config.environment.memory_mb,
    )
    judge = grant.environments.judge
    assert judge.network == ("public",)
    assert judge.max_envs_live == 2  # the task tightened the operator's 4
    assert grant.reserved_disk_mb == judge.max_disk_mb_live


# -- docs/sandbox-operator-guide.md: the operator reference --------------------

REFERENCE = Path(__file__).parents[1] / "docs" / "sandbox-operator-guide.md"
ROOT_CHECK = (
    Path(__file__).parents[1] / "scripts" / "operator" / "sandbox_root_check.sh"
)


def reference_block(marker: str) -> str:
    """The paragraph or table right after ``<!-- marker -->``."""
    text = REFERENCE.read_text()
    start = text.index(f"<!-- {marker} -->\n") + len(marker) + 9
    end = text.find("\n\n", start)
    return text[start : None if end < 0 else end]


def table_names(marker: str) -> list[str]:
    """The first backticked name of every row of the table after ``marker``."""
    return [
        re.match(r"\| `([^`]+)`", line)[1]
        for line in reference_block(marker).splitlines()
        if line.startswith("| `")
    ]


def listed(marker: str) -> list[str]:
    return re.findall(r"`([^`]+)`", reference_block(marker))


def test_every_documented_grant_request_and_spec_field_is_the_contracts():
    from rsi_harness.runtime import sandbox_contracts as contracts
    from rsi_harness.runtime import sandbox_env_contracts as env

    models = (
        contracts.EnvHostPolicy,
        contracts.EnvE2BHost,
        contracts.EnvToolFile,
        contracts.EnvGrant,
        contracts.EnvBuildGrant,
        contracts.EnvAllowlistGrant,
        contracts.EnvRunLimits,
        contracts.EnvPhaseRequest,
        env.EnvSpec,
        env.ServiceSpec,
        env.EnvMount,
        env.EnvHealthcheck,
        env.EnvDependency,
    )
    for model in models:
        documented = table_names(f"fields: {model.__name__}")
        assert len(documented) == len(set(documented)), model.__name__
        assert set(documented) == set(model.model_fields), model.__name__
    # A task's limits table can tighten exactly the operator's phase limits.
    assert set(contracts.EnvLimitsRequest.model_fields) == set(
        contracts.EnvLimits.model_fields
    )


def test_the_documented_compose_keys_are_the_front_ends():
    from rsi_harness.integrations import sandbox_compose

    service_keys = listed("compose-service-keys")
    assert len(service_keys) == len(set(service_keys))
    assert set(service_keys) == sandbox_compose._SERVICE_KEYS
    assert set(listed("compose-build-keys")) == sandbox_compose._BUILD_KEYS


@pytest.mark.parametrize("fragment", ["{}", *table_names("compose-refused")])
def test_every_documented_compose_refusal_is_refused(tmp_path, fragment):
    import yaml

    from rsi_harness.integrations.sandbox_compose import (
        ComposeError,
        load_project,
        translate,
    )

    service = {"image": "busybox:1.37.0", **yaml.safe_load(fragment)}
    path = tmp_path / "docker-compose.yaml"
    path.write_text(yaml.safe_dump({"services": {"main": service}}))

    def run():
        loaded = load_project([path], environ={}, project_dir=tmp_path)
        return translate(loaded, project_dir=tmp_path, network="public", disk_mb=1024)

    if fragment == "{}":
        assert run().services == ("main",)  # the service itself is fine
        return
    # Refused for its own key, not for anything else in the fragment.
    [(key, value)] = yaml.safe_load(fragment).items()
    if key == "build":
        [key] = [f"build.{item}" for item in value if item != "context"]
    with pytest.raises(ComposeError, match=re.escape(f"services.main.{key}")):
        run()


def test_the_documented_builder_exception_is_the_builders():
    from rsi_harness.runtime import sandbox_build

    assert listed("builder-cap-add") == list(sandbox_build.BUILDER_CAP_ADD)
    # systempaths=unconfined is the builder's empty MaskedPaths/ReadonlyPaths.
    assert listed("builder-security-opt") == [
        *sandbox_build.BUILDER_SECURITY_OPT,
        "systempaths=unconfined",
    ]


def test_the_documented_build_refusals_hold():
    from rsi_harness.runtime import sandbox_envs
    from rsi_harness.runtime.sandbox_build import build_argv, built_image_labels
    from rsi_harness.runtime.sandbox_contracts import SandboxError
    from tests.runtime.test_sandbox_build import owner, request

    assert listed("build-refused-arg-prefixes") == [
        prefix + "*" for prefix in sandbox_envs._DENIED_BUILD_ARGS
    ]
    for prefix in sandbox_envs._DENIED_BUILD_ARGS:
        with pytest.raises(SandboxError, match="reserved"):
            sandbox_envs._build_options(
                None, None, None, {prefix + "X": "1"}, {}, False
            )
    with pytest.raises(SandboxError, match="reserved"):
        sandbox_envs._build_options(None, None, None, {}, {"rsi-harness.x": "1"}, False)
    argv = build_argv(
        request(
            build_args={"A": "--allow=network.host --secret id=x"},
            no_cache=True,
            network="none",
        ),
        built_image_labels(owner(), "i" + "2" * 32),
    )
    for flag in listed("buildctl-never"):
        assert not any(item == flag or item.startswith(flag + "=") for item in argv)
        assert not any(
            flag in item for item in argv if not item.startswith("build-arg:")
        )


def test_the_documented_roles_are_the_contracts():
    from rsi_harness.runtime import sandbox_env_contracts as env

    assert listed("roles") == [
        env.ENV_ROLE,
        env.ENV_NETWORK_ROLE,
        env.ENV_VOLUME_ROLE,
        env.BUILDER_ROLE,
        env.BUILDER_NETWORK_ROLE,
        env.BUILDER_VOLUME_ROLE,
        env.BUILD_ROLE,
    ]


def test_the_documented_operations_are_the_servers():
    from rsi_harness.runtime import sandbox_server

    rows = {}
    for line in reference_block("wire-ops").splitlines():
        if line.startswith("| `"):
            _, op, fields, _ = line.split("|")
            rows[op.strip().strip("`")] = set(re.findall(r"`([^`]+)`", fields))
    assert rows == sandbox_server._ALL_FIELDS
    assert list(rows)[:7] == list(sandbox_server._FIELDS)


def test_the_documented_error_codes_and_statuses_are_the_servers():
    from rsi_harness.runtime.sandbox_contracts import SandboxError
    from rsi_harness.runtime.sandbox_server import _error

    rows = [
        re.match(r"\| `([^`]+)` \| (\d+) \|", line).groups()
        for line in reference_block("error-codes").splitlines()
        if line.startswith("| `")
    ]
    assert [code for code, _ in rows] == [
        "permission",
        "unsupported",
        "invalid",
        "busy",
        "quota",
        "expired",
        "unknown-outcome",
        "infrastructure",
    ]
    for code, status in rows:
        assert _error(SandboxError(code, "f", "m")).status_code == int(status)


def test_the_documented_root_checks_are_the_scripts():
    text = REFERENCE.read_text()
    documented = re.findall(r"^\| (R\d+) \|", text, re.MULTILINE)
    scripted = re.findall(
        r'^\s*(?:pytest_)?check (R\d+) "', ROOT_CHECK.read_text(), re.MULTILINE
    )
    assert sorted(documented, key=lambda item: int(item[1:])) == sorted(
        scripted, key=lambda item: int(item[1:])
    )
    assert "28 CPUs, 51456 MiB and 65536 MiB" in " ".join(text.split())


def test_the_documented_prune_images_options_are_the_commands():
    import typer.main

    from rsi_harness import cli

    group = typer.main.get_command(cli.app).commands["sandbox"]
    command = group.commands["prune-images"]
    options = {option for param in command.params for option in param.opts}
    documented = {item for item in listed("prune-images-options") if item[:2] == "--"}
    assert documented == options


def test_the_documented_prepull_example_names_a_manifest_digest():
    from rsi_harness.runtime.sandbox_images import pull_reference

    text = REFERENCE.read_text()
    section = text[text.index("### Pre-pulling large image sets") :]
    section = section[: section.index("### Many envs at once")]
    manifest = (
        REFERENCE.parents[1] / "sample_tasks" / "harbor-in-judge" / "images.manifest"
    ).read_text()
    examples = set(re.findall(r"[a-z0-9./-]+@sha256:[0-9a-f]{64}", section))
    # One example image, the manifest's fix-git line, as the broker parses it.
    [example] = examples
    assert example in manifest.splitlines()
    assert pull_reference(example)[1].startswith("sha256:")
    assert 'docker_image = "alexgshaw/fix-git@sha256:' in section
    assert "rsi-sandbox pull alexgshaw/fix-git@sha256:" in section
