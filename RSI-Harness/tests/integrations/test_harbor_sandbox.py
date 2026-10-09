"""Contract tests for Harbor's managed-sandbox environment adapter."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.models.task.config import (
    AgentConfig as TaskAgentConfig,
)
from harbor.models.task.config import (
    EnvironmentConfig,
    NetworkMode,
    NetworkPolicy,
    StepConfig,
    TpuSpec,
    VerifierCollectConfig,
    VerifierConfig,
)
from harbor.models.task.config import (
    TaskConfig as HarborTaskConfig,
)
from harbor.models.trial.config import (
    AgentConfig as TrialAgentConfig,
)
from harbor.models.trial.config import (
    EnvironmentConfig as TrialEnvironmentConfig,
)
from harbor.models.trial.config import (
    ServiceVolumeConfig,
)
from harbor.models.trial.paths import TrialPaths

from rsi_harness.integrations.sandbox_client import ProtocolError
from tests.sandbox_helpers import make_profile, make_sandbox_grant


def _managed_profile():
    return make_profile().model_copy(
        update={
            "tmpfs_mb": (
                ("/workspace", 32),
                ("/tests", 8),
                ("/solution", 8),
                ("/logs", 8),
                ("/tmp", 8),
                ("/dev/shm", 8),
            )
        }
    )


class FakeClient:
    def __init__(self) -> None:
        grant = make_sandbox_grant()
        self.capability_result = {
            "version": 1,
            "owner": {
                "run_id": "run",
                "task_id": "task",
                "phase": "work",
                "round_id": None,
            },
            "grant": grant.work.model_dump(mode="json"),
            "profiles": [_managed_profile().model_dump(mode="json")],
        }
        self.calls: list[tuple] = []
        self.execute_result = SimpleNamespace(
            exit_code=0,
            stdout="out",
            stderr="err",
            timed_out=False,
            oom_killed=False,
            output_limited=False,
            truncated=False,
            duration_sec=0.25,
        )
        self.download_result: tuple[dict, ...] = ()

    def capabilities(self):
        self.calls.append(("capabilities",))
        return self.capability_result

    def create(self, profile, lifetime_sec, request_id=None):
        self.calls.append(("create", profile, lifetime_sec, request_id))
        return "child"

    def execute(self, handle, argv, cwd, env=None, timeout_sec=30):
        self.calls.append(("execute", handle, argv, cwd, env, timeout_sec))
        return self.execute_result

    def upload(self, handle, root, records, request_id=None, timeout_sec=30):
        self.calls.append(
            ("upload", handle, root, tuple(records), request_id, timeout_sec)
        )

    def download(self, handle, root, paths, timeout_sec=30):
        self.calls.append(("download", handle, root, tuple(paths), timeout_sec))
        return self.download_result

    def destroy(self, handle):
        self.calls.append(("destroy", handle))


def _task_config(**updates) -> EnvironmentConfig:
    profile = _managed_profile()
    fields = {
        "docker_image": profile.image,
        "workdir": profile.workdir,
        "cpus": profile.cpus,
        "memory_mb": profile.memory_mb,
        "gpus": 0,
        "network_mode": "no-network",
    }
    fields.update(updates)
    return EnvironmentConfig(**fields)


def _environment(
    tmp_path: Path,
    *,
    client: FakeClient | None = None,
    task_config: EnvironmentConfig | None = None,
    network_policy: NetworkPolicy | None = None,
    phase_network_policies: list[NetworkPolicy] | None = None,
    mounts: list[ServiceVolumeConfig] | None = None,
    extra_docker_compose: list[Path] | None = None,
    preflight: bool = True,
):
    from rsi_harness.integrations.harbor_sandbox import (
        ManagedSandboxEnvironment,
        preflight_managed_trial,
    )

    environment_dir = tmp_path / "environment"
    environment_dir.mkdir(exist_ok=True)
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    kwargs = {}
    if network_policy is not None:
        kwargs["network_policy"] = network_policy
    effective_task_config = task_config or _task_config()
    environment = ManagedSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="minimal",
        session_id="session",
        trial_paths=paths,
        task_env_config=effective_task_config,
        profile="offline",
        client=client or FakeClient(),
        phase_network_policies=phase_network_policies,
        mounts=mounts,
        extra_docker_compose=extra_docker_compose,
        **kwargs,
    )
    if preflight:
        preflight_managed_trial(
            _trial_for_preflight(
                environment,
                _harbor_task_config(environment=effective_task_config),
            )
        )
    return environment


def _offline_policy() -> NetworkPolicy:
    return NetworkPolicy(network_mode=NetworkMode.NO_NETWORK, allowed_hosts=[])


def _harbor_task_config(**updates) -> HarborTaskConfig:
    fields = {
        "environment": _task_config(),
        "agent": TaskAgentConfig(user="root", network_mode="no-network"),
        "verifier": VerifierConfig(user="root", network_mode="no-network"),
    }
    fields.update(updates)
    return HarborTaskConfig(**fields)


def _trial_for_preflight(
    env,
    task_config: HarborTaskConfig,
    *,
    trial_agent: TrialAgentConfig | None = None,
):
    return SimpleNamespace(
        task=SimpleNamespace(config=task_config),
        config=SimpleNamespace(
            agent=trial_agent or TrialAgentConfig(name="oracle"),
            environment=TrialEnvironmentConfig(),
            verifier=SimpleNamespace(disable=False),
        ),
        agent_environment=env,
    )


@pytest.mark.asyncio
async def test_start_requires_whole_task_preflight_before_child_creation(tmp_path):
    client = FakeClient()
    env = _environment(
        tmp_path,
        client=client,
        network_policy=_offline_policy(),
        preflight=False,
    )

    with pytest.raises(RuntimeError, match="whole-task preflight"):
        await env.start(force_build=False)

    assert not any(call[0] == "create" for call in client.calls)


@pytest.mark.parametrize(
    "location",
    [
        "task-agent",
        "task-verifier",
        "step-agent",
        "step-verifier",
        "task-collect",
        "step-collect",
    ],
)
def test_whole_task_preflight_rejects_every_nonroot_user_before_create(
    tmp_path, location
):
    from rsi_harness.integrations.harbor_sandbox import preflight_managed_trial

    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    agent = TaskAgentConfig(user="root", network_mode="no-network")
    verifier = VerifierConfig(user="root", network_mode="no-network")
    steps = None
    if location == "task-agent":
        agent = TaskAgentConfig(user="nobody", network_mode="no-network")
    elif location == "task-verifier":
        verifier = VerifierConfig(user=1000, network_mode="no-network")
    elif location == "step-agent":
        steps = [
            StepConfig(
                name="one",
                agent=TaskAgentConfig(user="nobody", network_mode="no-network"),
            )
        ]
    elif location == "step-verifier":
        steps = [
            StepConfig(
                name="one",
                verifier=VerifierConfig(user=1000, network_mode="no-network"),
            )
        ]
    elif location == "task-collect":
        verifier = VerifierConfig(
            user="root",
            network_mode="no-network",
            collect=[VerifierCollectConfig(command="true", user="nobody")],
        )
    else:
        steps = [
            StepConfig(
                name="one",
                verifier=VerifierConfig(
                    network_mode="no-network",
                    collect=[VerifierCollectConfig(command="true", user=1000)],
                ),
            )
        ]
    task_config = _harbor_task_config(
        agent=agent,
        verifier=verifier,
        steps=steps,
    )

    with pytest.raises(ValueError, match="root user"):
        preflight_managed_trial(_trial_for_preflight(env, task_config))

    assert not any(call[0] == "create" for call in client.calls)


def test_whole_task_preflight_rejects_separate_verifier_before_create(tmp_path):
    from harbor.models.task.config import VerifierEnvironmentMode

    from rsi_harness.integrations.harbor_sandbox import preflight_managed_trial

    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    task_config = _harbor_task_config(
        verifier=VerifierConfig(
            user="root",
            network_mode="no-network",
            environment_mode=VerifierEnvironmentMode.SEPARATE,
        )
    )

    with pytest.raises(ValueError, match="separate verifier"):
        preflight_managed_trial(_trial_for_preflight(env, task_config))

    assert not any(call[0] == "create" for call in client.calls)


def test_whole_task_preflight_rejects_runtime_phase_network_override_before_create(
    tmp_path,
):
    from rsi_harness.integrations.harbor_sandbox import preflight_managed_trial

    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    trial = _trial_for_preflight(
        env,
        _harbor_task_config(),
        trial_agent=TrialAgentConfig(
            name="oracle",
            extra_allowed_hosts=["example.com"],
        ),
    )

    with pytest.raises(ValueError, match="no-network"):
        preflight_managed_trial(trial)

    assert not any(call[0] == "create" for call in client.calls)


def test_capabilities_are_offline_unmounted_with_hard_limits_only(tmp_path):
    from harbor.environments.capabilities import EnvironmentResourceCapabilities

    env = _environment(tmp_path, network_policy=_offline_policy())

    assert env.type() == "rsi-managed-sandbox"
    assert env.capabilities.disable_internet is True
    assert env.capabilities.mounted is False
    assert env.capabilities.gpus is False
    assert env.capabilities.network_allowlist is False
    assert env.resource_capabilities() == EnvironmentResourceCapabilities(
        cpu_limit=True,
        memory_limit=True,
    )


def test_implicit_public_network_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="public"):
        _environment(tmp_path)


def test_allowlist_is_rejected_for_baseline_and_any_phase(tmp_path):
    allowlist = NetworkPolicy(
        network_mode=NetworkMode.ALLOWLIST,
        allowed_hosts=["example.com"],
    )
    with pytest.raises(ValueError, match="allowlist"):
        _environment(tmp_path, network_policy=allowlist)
    with pytest.raises(ValueError, match="allowlist"):
        _environment(
            tmp_path,
            network_policy=_offline_policy(),
            phase_network_policies=[allowlist],
        )


@pytest.mark.parametrize(
    "change",
    [
        {"gpus": 1},
        {"tpu": TpuSpec(type="v5e", topology="1x1")},
    ],
)
def test_accelerators_are_rejected(tmp_path, change):
    with pytest.raises(RuntimeError):
        _environment(
            tmp_path,
            task_config=_task_config(**change),
            network_policy=_offline_policy(),
        )


def test_compose_and_extra_compose_are_rejected(tmp_path):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "docker-compose.yaml").write_text("services: {}\n")
    from rsi_harness.integrations.harbor_sandbox import ManagedSandboxEnvironment

    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    with pytest.raises(ValueError, match="Compose"):
        ManagedSandboxEnvironment(
            environment_dir=environment_dir,
            environment_name="minimal",
            session_id="session",
            trial_paths=paths,
            task_env_config=_task_config(),
            profile="offline",
            client=FakeClient(),
            network_policy=_offline_policy(),
        )

    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("services: {}\n")
    with pytest.raises(ValueError, match="compose"):
        _environment(
            tmp_path,
            network_policy=_offline_policy(),
            extra_docker_compose=[overlay],
        )


@pytest.mark.parametrize(
    "change,match",
    [
        ({"docker_image": "sha256:" + "b" * 64}, "image"),
        ({"cpus": 2}, "cpus"),
        ({"memory_mb": 512}, "memory_mb"),
        ({"storage_mb": 1}, "storage_mb"),
        ({"workdir": "/tmp"}, "workdir"),
    ],
)
def test_unapproved_image_resources_and_workdir_are_rejected(tmp_path, change, match):
    with pytest.raises(ValueError, match=match):
        _environment(
            tmp_path,
            task_config=_task_config(**change),
            network_policy=_offline_policy(),
        )


def test_only_standard_harbor_log_mount_destinations_are_scratch_hints(tmp_path):
    standard = [
        ServiceVolumeConfig(
            type="bind",
            source=str(tmp_path / "host-agent"),
            target="/logs/agent",
        ),
        ServiceVolumeConfig(
            type="bind",
            source=str(tmp_path / "host-verifier"),
            target="/logs/verifier",
        ),
        ServiceVolumeConfig(
            type="bind",
            source=str(tmp_path / "host-artifacts"),
            target="/logs/artifacts",
        ),
    ]
    _environment(
        tmp_path,
        network_policy=_offline_policy(),
        mounts=standard,
    )

    with pytest.raises(ValueError, match="mount"):
        _environment(
            tmp_path,
            network_policy=_offline_policy(),
            mounts=[
                ServiceVolumeConfig(
                    type="bind", source=str(tmp_path), target="/workspace/host"
                )
            ],
        )


@pytest.mark.asyncio
async def test_force_build_is_rejected_before_child_creation(tmp_path):
    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())

    with pytest.raises(ValueError, match="force_build"):
        await env.start(force_build=True)

    assert not any(call[0] == "create" for call in client.calls)


@pytest.mark.asyncio
async def test_start_exec_and_stop_translate_to_client_contract(tmp_path):
    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())

    await env.start(force_build=False)
    client.execute_result.exit_code = 7
    result = await env.exec(
        "printf hello",
        cwd="/workspace",
        env={"LOCAL": "yes"},
        timeout_sec=9,
        user="root",
    )
    await env.stop(delete=True)

    assert result.stdout == "out"
    assert result.stderr == "err"
    assert result.return_code == 7
    assert (
        "execute",
        "child",
        ["/bin/sh", "-c", "printf hello"],
        "/workspace",
        {"LOCAL": "yes"},
        9,
    ) in client.calls
    assert client.calls[-1] == ("destroy", "child")


@pytest.mark.asyncio
async def test_start_uploads_prebuilt_environment_directory_to_workdir(tmp_path):
    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    bootstrap = env.environment_dir / "bootstrap.txt"
    bootstrap.write_text("baked dependency\n")
    bootstrap.chmod(0o640)

    await env.start(force_build=False)

    uploads = [call for call in client.calls if call[0] == "upload"]
    assert len(uploads) == 1
    assert uploads[0][2] == "/workspace"
    assert uploads[0][3] == (
        {
            "path": "bootstrap.txt",
            "kind": "file",
            "mode": 0o640,
            "data": b"baked dependency\n",
        },
    )


@pytest.mark.asyncio
async def test_exec_merges_persistent_environment(tmp_path):
    client = FakeClient()
    from rsi_harness.integrations.harbor_sandbox import (
        ManagedSandboxEnvironment,
        preflight_managed_trial,
    )

    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    paths = TrialPaths(tmp_path / "trial")
    paths.mkdir()
    task_config = _task_config(env={"BASE": "one", "SAME": "task"})
    env = ManagedSandboxEnvironment(
        environment_dir=environment_dir,
        environment_name="minimal",
        session_id="session",
        trial_paths=paths,
        task_env_config=task_config,
        profile="offline",
        client=client,
        persistent_env={"TRIAL": "two", "SAME": "trial"},
        network_policy=_offline_policy(),
    )
    preflight_managed_trial(
        _trial_for_preflight(env, _harbor_task_config(environment=task_config))
    )
    await env.start(False)

    await env.exec("true", env={"LOCAL": "three", "SAME": "exec"})

    execute = [call for call in client.calls if call[0] == "execute"][-1]
    assert execute[4] == {
        "BASE": "one",
        "TRIAL": "two",
        "LOCAL": "three",
        "SAME": "exec",
    }


@pytest.mark.asyncio
async def test_exec_rejects_cwd_outside_scratch_and_nonroot_effective_user(tmp_path):
    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(False)

    with pytest.raises(ValueError, match="scratch"):
        await env.exec("true", cwd="/etc")
    with pytest.raises(ValueError, match="root"):
        await env.exec("true", user="nobody")
    with env.with_default_user(1000):
        with pytest.raises(ValueError, match="root"):
            await env.exec("true")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,exception_name",
    [
        ({"timed_out": True}, "SandboxExecutionTimeoutError"),
        ({"output_limited": True}, "SandboxQuotaExceededError"),
        ({"oom_killed": True}, "SandboxQuotaExceededError"),
    ],
)
async def test_terminal_exec_flags_raise_distinct_errors(
    tmp_path, failure, exception_name
):
    import rsi_harness.integrations.harbor_sandbox as adapter

    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(False)
    vars(client.execute_result).update(failure)

    with pytest.raises(getattr(adapter, exception_name)):
        await env.exec("true")


@pytest.mark.asyncio
async def test_protocol_quota_and_unknown_outcome_are_not_returned_as_success(tmp_path):
    import rsi_harness.integrations.harbor_sandbox as adapter

    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(False)

    def fail(*args, **kwargs):
        raise ProtocolError("budget", code="quota")

    client.execute = fail
    with pytest.raises(adapter.SandboxQuotaExceededError):
        await env.exec("true")

    def unknown(*args, **kwargs):
        raise ProtocolError("lost reply", code="unknown-outcome")

    client.execute = unknown
    with pytest.raises(adapter.SandboxUnknownOutcomeError):
        await env.exec("true")


@pytest.mark.asyncio
async def test_direct_file_and_directory_transfers_preserve_paths_and_modes(tmp_path):
    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(False)
    source_file = tmp_path / "source.txt"
    source_file.write_text("hello")
    source_file.chmod(0o640)
    source_dir = tmp_path / "tree"
    source_dir.mkdir()
    (source_dir / "run.sh").write_text("#!/bin/sh\n")
    (source_dir / "run.sh").chmod(0o755)

    await env.upload_file(source_file, "/workspace/renamed.txt")
    await env.upload_dir(source_dir, "/tests")

    uploads = [call for call in client.calls if call[0] == "upload"]
    assert uploads[0][2] == "/workspace"
    assert uploads[0][3] == (
        {"path": "renamed.txt", "kind": "file", "mode": 0o640, "data": b"hello"},
    )
    assert uploads[1][2] == "/tests"
    assert uploads[1][3] == (
        {"path": "run.sh", "kind": "file", "mode": 0o755, "data": b"#!/bin/sh\n"},
    )

    client.download_result = (
        {"path": "remote.txt", "kind": "file", "mode": 0o600, "data": b"remote"},
    )
    downloaded_file = tmp_path / "download" / "local.txt"
    await env.download_file("/workspace/remote.txt", downloaded_file)
    assert downloaded_file.read_bytes() == b"remote"
    assert downloaded_file.stat().st_mode & 0o777 == 0o600

    client.download_result = (
        {"path": "sub", "kind": "directory", "mode": 0o750, "data": b""},
        {"path": "sub/value", "kind": "file", "mode": 0o640, "data": b"v"},
    )
    downloaded_dir = tmp_path / "downloaded-tree"
    await env.download_dir("/logs/verifier", downloaded_dir)
    assert (downloaded_dir / "sub" / "value").read_bytes() == b"v"


@pytest.mark.asyncio
async def test_local_upload_rejects_symlinks_without_calling_client(tmp_path):
    client = FakeClient()
    env = _environment(tmp_path, client=client, network_policy=_offline_policy())
    await env.start(False)
    source = tmp_path / "tree"
    source.mkdir()
    (source / "real").write_text("secret")
    (source / "link").symlink_to("real")

    with pytest.raises(ValueError, match="links"):
        await env.upload_dir(source, "/workspace")

    assert len([call for call in client.calls if call[0] == "upload"]) == 0


@pytest.mark.asyncio
async def test_inherited_archive_download_helpers_are_unsupported(tmp_path):
    env = _environment(tmp_path, network_policy=_offline_policy())

    with pytest.raises(NotImplementedError, match="filtered"):
        await env.download_dir_filtered(
            source_dir="/logs/agent",
            target_dir=tmp_path / "agent",
            include=["*.txt"],
        )
    with pytest.raises(NotImplementedError, match="exclusion"):
        await env.download_dir_with_exclusions(
            source_dir="/logs/agent",
            target_dir=tmp_path / "agent",
            exclude=["*.tmp"],
        )
