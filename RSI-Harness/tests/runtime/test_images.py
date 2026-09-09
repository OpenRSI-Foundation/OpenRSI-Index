from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath
from uuid import uuid4

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentPlan,
    GPUAllocation,
    GPURequirement,
    JudgeGPUMode,
    MainServiceConfig,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
    TaskDefinition,
    VerifierPlan,
)
from rsi_harness.runtime import images as images_module
from rsi_harness.runtime.images import (
    DockerImageBuilder,
    build_agent_identity_script,
    resolve_image_workdir,
    resolve_rootfs_snapshot_mode,
)
from rsi_harness.runtime.mount_topology import validate_run_plan_mount_topology
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_loop import __version__ as rsi_loop_version
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeDockerClient, FakeDockerImage

RSI_LOOP_VERSION = rsi_loop_version


def task_definition(
    tmp_path: Path,
    *,
    image: str | None = None,
    workdir: PurePosixPath | str | None = PurePosixPath("/workspace"),
    agent_user: str | None = "1001",
    verifier_user: str | None = "1002",
    service_user: str | None = None,
) -> TaskDefinition:
    source = tmp_path / "task"
    environment = source / "environment"
    tests = source / "tests"
    environment.mkdir(parents=True)
    tests.mkdir()
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\nWORKDIR /workspace\n")
    (environment / "environment-only.txt").write_text("base input")
    (tests / "test.sh").write_text("secret verifier")
    (source / "solution.sh").write_text("secret solution")
    return TaskDefinition(
        task_id="minimal-gpu",
        source_dir=source.resolve(),
        source_digest="source-digest",
        instruction_digest="instruction-digest",
        tests_digest="tests-digest",
        environment_digest="environment-digest-abcdef",
        workdir=workdir,
        service=MainServiceConfig(
            build_context=None if image else environment.resolve(),
            image=image,
            workdir=workdir,
            user=service_user,
        ),
        gpu_requirement=GPURequirement(count=1),
        verifier=VerifierPlan(
            command=("/bin/bash", "/tests/test.sh"), user=verifier_user
        ),
        agent=AgentPlan(name="codex", user=agent_user),
    )


def test_dockerfile_base_uses_only_environment_and_work_is_derived_independently(
    tmp_path,
):
    client = FakeDockerClient()
    task = task_definition(tmp_path)
    agent = create_agent("codex", RSILoopConfig())
    builder = DockerImageBuilder(
        client,
        run_id="run-7",
        rsi_loop_version=RSI_LOOP_VERSION,
        bootstrap_version="7",
    )

    plan = builder.prepare(task, agent=agent)

    assert len(client.images.built) == 2
    base_build, work_build = client.images.built
    assert Path(base_build["path"]) == task.source_dir / "environment"
    assert set(base_build["context_files"]) == {
        "Dockerfile",
        "environment-only.txt",
    }
    assert set(work_build["context_files"]) == {"Dockerfile", "rsi-submit"}
    assert base_build["labels"] == {
        "rsi-harness.run-id": "run-7",
        "rsi-harness.task-id": "minimal-gpu",
        "rsi-harness.role": "judge",
    }
    assert work_build["labels"] == {
        "rsi-harness.run-id": "run-7",
        "rsi-harness.task-id": "minimal-gpu",
        "rsi-harness.role": "work",
    }
    assert "secret verifier" not in repr(client.images.built)
    assert "secret solution" not in repr(client.images.built)
    assert "environment-digest-abcdef" in work_build["tag"]
    assert "codex" in work_build["tag"]
    assert "1.0.0-rsi.1" in work_build["tag"]
    assert "v7" in work_build["tag"]
    assert plan.work_ref != plan.base_ref
    assert plan.judge_ref == plan.base_ref
    assert plan.judge_digest == plan.base_digest
    assert plan.work_digest != plan.base_digest
    assert "rsi-submit" in work_build["context_files"]["Dockerfile"]
    dockerfile = work_build["context_files"]["Dockerfile"]
    assert "@openai/codex@0.147.0" in dockerfile
    assert "@openai/codex@0.143.0" not in dockerfile


def test_work_cache_identity_uses_rsi_loop_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The former product-version key would incorrectly reuse cached Work images."""
    captured_identities: list[dict[str, object]] = []
    original_dumps = images_module.json.dumps

    def capture_identity(value: object, *args: object, **kwargs: object) -> str:
        if isinstance(value, dict) and "base_digest" in value:
            captured_identities.append(value)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(images_module.json, "dumps", capture_identity)
    DockerImageBuilder(None, rsi_loop_version=RSI_LOOP_VERSION)._work_tag(
        task_definition(tmp_path),
        create_agent("codex", RSILoopConfig()),
        base_digest="sha256:base",
        effective_user="1001",
        workdir=PurePosixPath("/workspace"),
        dockerfile="FROM base",
        submit_client="submit client",
    )

    assert captured_identities == [
        {
            "base_digest": "sha256:base",
            "effective_user": "1001",
            "workdir": "/workspace",
            "dockerfile": "FROM base",
            "submit_client": "submit client",
            "image_labels": {
                "rsi-harness.run-id": "image-preparation",
                "rsi-harness.task-id": "minimal-gpu",
                "rsi-harness.role": "work",
            },
            "rsi_loop_version": RSI_LOOP_VERSION,
            "bootstrap_version": "1",
        }
    ]
    image_identity = captured_identities[0]
    assert image_identity["rsi_loop_version"] == RSI_LOOP_VERSION
    assert "sforge_version" not in image_identity


def test_image_builds_use_harbor_environment_timeout(tmp_path) -> None:
    client = FakeDockerClient()
    task = task_definition(tmp_path)
    task = task.model_copy(
        update={
            "service": task.service.model_copy(
                update={"build_timeout_seconds": 1800.0}
            )
        }
    )

    DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )

    assert [build["timeout"] for build in client.images.built] == [1800, 1800]


def test_prebuilt_image_is_pulled_when_absent_and_pinned_by_repo_digest(tmp_path):
    client = FakeDockerClient()
    task = task_definition(
        tmp_path,
        image="registry.test/base:latest",
        agent_user=None,
        verifier_user=None,
    )
    pulled = FakeDockerImage(
        "sha256:base-id",
        {
            "Id": "sha256:base-id",
            "RepoDigests": ["registry.test/base@sha256:deadbeef"],
            "Config": {"WorkingDir": "/workspace", "User": "1000"},
        },
    )

    def pull(ref: str):
        client.images.pulled.append(ref)
        client.images.by_ref[ref] = pulled
        return pulled

    client.images.pull = pull
    plan = DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )

    assert client.images.pulled == ["registry.test/base:latest"]
    assert plan.base_ref == "registry.test/base@sha256:deadbeef"
    assert plan.base_digest == "sha256:base-id"
    assert plan.judge_ref == plan.base_ref
    assert plan.work_user == "1000"
    assert plan.judge_user == "1000"


@pytest.mark.parametrize("configured", [None, PurePosixPath("/configured")])
def test_workdir_uses_configuration_or_non_root_image_metadata(configured):
    image = FakeDockerImage(
        "sha256:x",
        {
            "Id": "sha256:x",
            "RepoDigests": [],
            "Config": {"WorkingDir": "/from-image", "User": ""},
        },
    )

    assert resolve_image_workdir(configured, image) == (
        configured or PurePosixPath("/from-image")
    )


@pytest.mark.parametrize("workdir", ["", ".", "/"])
def test_image_without_workdir_uses_docker_root(workdir):
    image = FakeDockerImage(
        "sha256:x",
        {
            "Id": "sha256:x",
            "RepoDigests": [],
            "Config": {"WorkingDir": workdir, "User": ""},
        },
    )

    assert resolve_image_workdir(None, image) == PurePosixPath("/")


@pytest.mark.parametrize(
    ("configured", "image_workdir", "expected_workdir", "expected_mode"),
    [
        (
            PurePosixPath("/workspace"),
            "/image-cwd",
            PurePosixPath("/workspace"),
            RootfsSnapshotMode.SPLIT_WORKDIR,
        ),
        (
            None,
            "/image-cwd",
            PurePosixPath("/image-cwd"),
            RootfsSnapshotMode.SPLIT_WORKDIR,
        ),
        ("", "/image-cwd", PurePosixPath("/"), RootfsSnapshotMode.FULL_ROOTFS),
        (".", "/image-cwd", PurePosixPath("/"), RootfsSnapshotMode.FULL_ROOTFS),
        (None, "", PurePosixPath("/"), RootfsSnapshotMode.FULL_ROOTFS),
        (
            PurePosixPath("/"),
            "/image-cwd",
            PurePosixPath("/"),
            RootfsSnapshotMode.FULL_ROOTFS,
        ),
    ],
)
def test_image_plan_resolves_effective_workdir_and_snapshot_mode(
    tmp_path, configured, image_workdir, expected_workdir, expected_mode
):
    """Changing the image or declared WORKDIR must choose the matching mode."""
    task = task_definition(tmp_path, workdir=configured, image="task:latest")
    client = FakeDockerClient()
    client.images.by_ref["task:latest"] = FakeDockerImage(
        image_id="sha256:" + "a" * 64,
        attrs={
            "Config": {"WorkingDir": image_workdir, "User": "", "Volumes": None},
            "RepoDigests": ["task@sha256:" + "b" * 64],
        },
    )

    plan = DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )

    assert plan.workdir == expected_workdir
    assert plan.rootfs_snapshot_mode is expected_mode


@pytest.mark.parametrize(
    ("configured", "image_workdir", "expected"),
    [
        (PurePosixPath("/tests"), "/image-cwd", PurePosixPath("/tests")),
        (None, "/logs", PurePosixPath("/logs")),
    ],
)
def test_finalized_topology_rejects_configured_and_image_fallback_overlap(
    tmp_path, configured, image_workdir, expected
):
    """Both WORKDIR provenance paths must reach the immutable topology guard."""
    task = task_definition(
        tmp_path,
        image="task:latest",
        workdir=configured,
    )
    client = FakeDockerClient()
    client.images.by_ref["task:latest"] = FakeDockerImage(
        image_id="sha256:" + "a" * 64,
        attrs={
            "Config": {
                "WorkingDir": image_workdir,
                "User": "",
                "Volumes": None,
            },
            "RepoDigests": ["task@sha256:" + "b" * 64],
        },
    )
    images = DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )
    run_plan = HarborTaskCompiler().finalize(
        task,
        images,
        RunGPUPlan(
            authorized_pool=GPUAllocation(),
            work=GPUAllocation(),
            judge=GPUAllocation(),
            judge_mode=JudgeGPUMode.FREEZE_ONLY,
        ),
        RunPaths(
            root=(tmp_path / "run").resolve(),
            workspace=(tmp_path / "run" / "workspace").resolve(),
            logs=(tmp_path / "logs").resolve(),
        ),
        "docker-rootfs",
    )

    assert run_plan.workdir == expected
    with pytest.raises(SetupError, match="mount topology"):
        validate_run_plan_mount_topology(run_plan)


def test_root_workdir_requires_full_rootfs_snapshot_mode():
    """A non-root classification would omit mutations below Docker's root."""
    assert resolve_rootfs_snapshot_mode(PurePosixPath("/")) is (
        RootfsSnapshotMode.FULL_ROOTFS
    )


@pytest.mark.parametrize("workdir", ["relative", "relative/..", "/workspace/.."])
def test_relative_or_traversing_image_workdir_is_rejected(workdir):
    image = FakeDockerImage(
        "sha256:x",
        {
            "Id": "sha256:x",
            "RepoDigests": [],
            "Config": {"WorkingDir": workdir, "User": ""},
        },
    )

    with pytest.raises(SetupError, match="absolute WORKDIR"):
        resolve_image_workdir(None, image)


@pytest.mark.parametrize("volumes", [{}, None])
def test_image_without_declared_volumes_is_prepared(tmp_path, volumes):
    client = FakeDockerClient()
    task = task_definition(tmp_path, image="registry.test/base:latest")
    client.images.by_ref[task.service.image] = FakeDockerImage(
        "sha256:base-id",
        {
            "Id": "sha256:base-id",
            "RepoDigests": ["registry.test/base@sha256:deadbeef"],
            "Config": {"WorkingDir": "/workspace", "User": "", "Volumes": volumes},
        },
    )

    DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )

    assert len(client.images.built) == 1


def test_image_with_declared_volumes_is_rejected_before_work_creation(tmp_path):
    client = FakeDockerClient()
    task = task_definition(tmp_path, image="registry.test/base:latest")
    client.images.by_ref[task.service.image] = FakeDockerImage(
        "sha256:base-id",
        {
            "Id": "sha256:base-id",
            "RepoDigests": ["registry.test/base@sha256:deadbeef"],
            "Config": {
                "WorkingDir": "/workspace",
                "User": "",
                "Volumes": {"/data": {}},
            },
        },
    )

    with pytest.raises(SetupError, match="declares volumes"):
        DockerImageBuilder(client).prepare(
            task, agent=create_agent("codex", RSILoopConfig())
        )

    assert client.images.built == []
    assert client.containers.runs == []


def test_dynamic_preflight_uses_phase_users_and_agent_home(tmp_path):
    client = FakeDockerClient()
    task = task_definition(tmp_path, agent_user="1001", verifier_user="1002")

    DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )

    work, judge = client.containers.runs
    assert work["user"] == "1001"
    assert work["environment"] == {
        "HOME": "/home/agent",
        "NVIDIA_VISIBLE_DEVICES": "void",
    }
    assert "id -u agent" in work["command"][2]
    assert 'id -u)"' in work["command"][2]
    assert 'id -g)"' in work["command"][2]
    assert "id -u 1001" not in work["command"][2]
    assert "test -w /workspace" in work["command"][2]
    assert judge["user"] == "1002"
    assert judge["command"] == ["/bin/sh", "-c", "test -d /workspace"]
    assert judge["environment"] == {"NVIDIA_VISIBLE_DEVICES": "void"}
    assert work["network_mode"] == "none"
    assert judge["network_mode"] == "none"
    assert work["labels"]["rsi-harness.role"] == "work"
    assert judge["labels"]["rsi-harness.role"] == "judge"
    assert work["labels"]["rsi-harness.task-id"] == "minimal-gpu"

    hook_commands = [call["command"] for call in client.api.exec_create_calls]
    assert any(
        isinstance(command, tuple)
        and command[-1] == "/home/agent/.claude/settings.json"
        for command in hook_commands
    )
    assert any(
        isinstance(command, list) and "chown -R agent:agent" in command[-1]
        for command in hook_commands
    )


def test_disabled_stop_hook_skips_image_hook_preflight(tmp_path) -> None:
    client = FakeDockerClient()
    task = task_definition(tmp_path)
    task = task.model_copy(
        update={
            "agent": task.agent.model_copy(
                update={"install_stop_hook": False}
            )
        }
    )

    DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )

    assert client.api.exec_create_calls == []


def test_service_user_is_shared_fallback_for_work_image_and_both_phases(
    tmp_path,
) -> None:
    client = FakeDockerClient()
    task = task_definition(
        tmp_path,
        agent_user=None,
        verifier_user=None,
        service_user="2001:2002",
    )

    plan = DockerImageBuilder(client).prepare(
        task, agent=create_agent("codex", RSILoopConfig())
    )

    work_build = client.images.built[-1]
    assert "USER 2001:2002" in work_build["context_files"]["Dockerfile"]
    work, judge = client.containers.runs
    assert work["user"] == "2001:2002"
    assert judge["user"] == "2001:2002"
    assert plan.work_user == "2001:2002"
    assert plan.judge_user == "2001:2002"


def test_work_cache_changes_with_base_user_workdir_and_generated_client(
    tmp_path, monkeypatch
):
    client = FakeDockerClient()
    image_ref = "registry.test/base:latest"
    first_base = FakeDockerImage(
        "sha256:base-one",
        {
            "Id": "sha256:base-one",
            "RepoDigests": [],
            "Config": {"WorkingDir": "/workspace", "User": "1000"},
        },
    )
    second_base = FakeDockerImage(
        "sha256:base-two",
        {
            "Id": "sha256:base-two",
            "RepoDigests": [],
            "Config": {"WorkingDir": "/workspace", "User": "1000"},
        },
    )
    client.images.by_ref[image_ref] = first_base
    builder = DockerImageBuilder(client)
    agent = create_agent("codex", RSILoopConfig())
    builder.prepare(task_definition(tmp_path / "one", image=image_ref), agent=agent)
    first_tag = client.images.built[-1]["tag"]

    client.images.by_ref[image_ref] = second_base
    builder.prepare(task_definition(tmp_path / "two", image=image_ref), agent=agent)
    second_tag = client.images.built[-1]["tag"]

    user_task = task_definition(tmp_path / "three", image=image_ref, agent_user="2002")
    builder.prepare(user_task, agent=agent)
    user_tag = client.images.built[-1]["tag"]

    import rsi_harness.runtime.images as images_module

    monkeypatch.setattr(
        images_module,
        "generate_submit_client",
        lambda: "#!/bin/sh\necho changed-client\n",
    )
    builder.prepare(
        task_definition(tmp_path / "four", image=image_ref, agent_user="2002"),
        agent=agent,
    )
    client_tag = client.images.built[-1]["tag"]

    DockerImageBuilder(client, run_id="different-run").prepare(
        task_definition(tmp_path / "five", image=image_ref, agent_user="2002"),
        agent=agent,
    )
    configuration_tag = client.images.built[-1]["tag"]

    assert len(
        {first_tag, second_tag, user_tag, client_tag, configuration_tag}
    ) == 5


def test_build_failure_reports_the_agent_setup_boundary(tmp_path):
    from docker.errors import BuildError

    client = FakeDockerClient()
    task = task_definition(tmp_path)
    original_build = client.images.build

    def fail_work_build(**kwargs):
        if kwargs["tag"].startswith("rsi-harness-work:"):
            raise BuildError("RUN npm install failed", [])
        return original_build(**kwargs)

    client.images.build = fail_work_build

    with pytest.raises(SetupError, match="codex.*pinned install commands"):
        DockerImageBuilder(client).prepare(
            task, agent=create_agent("codex", RSILoopConfig())
        )


def test_image_builder_drains_returned_docker_build_logs(tmp_path: Path) -> None:
    class BuildLogStream:
        def __init__(self) -> None:
            self._events = iter(({"stream": "one"}, {"stream": "two"}))
            self.exhausted = False

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._events)
            except StopIteration:
                self.exhausted = True
                raise

    client = FakeDockerClient()
    original_build = client.images.build
    streams: list[BuildLogStream] = []

    def build_with_tracked_logs(**kwargs):
        image, _logs = original_build(**kwargs)
        stream = BuildLogStream()
        streams.append(stream)
        return image, stream

    client.images.build = build_with_tracked_logs

    DockerImageBuilder(client).prepare(
        task_definition(tmp_path), agent=create_agent("codex", RSILoopConfig())
    )

    assert len(streams) == 2
    assert all(stream.exhausted for stream in streams)


@pytest.mark.integration
def test_generated_work_dockerfile_with_multiline_bootstrap_passes_docker_check(
    tmp_path: Path,
) -> None:
    probe = subprocess.run(
        ["docker", "version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"Docker parser capability unavailable: {probe.stderr}")

    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=create_agent("codex", RSILoopConfig()),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="0",
    )
    (tmp_path / "rsi-submit").write_text("#!/bin/sh\nexit 0\n")

    result = subprocess.run(
        ["docker", "build", "--check", "-f", "-", "."],
        cwd=tmp_path,
        input=dockerfile,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.integration
def test_work_image_replaces_a_preexisting_registered_agent_launcher(
    tmp_path: Path,
) -> None:
    probe = subprocess.run(
        ["docker", "version"],
        capture_output=True,
        check=False,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"Docker build capability unavailable: {probe.stderr}")

    identity = uuid4().hex
    base_tag = f"rsi-harness-test:preexisting-agent-launcher-base-{identity}"
    work_tag = f"rsi-harness-test:preexisting-agent-launcher-work-{identity}"
    base = tmp_path / "base"
    work = tmp_path / "work"
    base.mkdir()
    work.mkdir()
    (base / "Dockerfile").write_text(
        "FROM alpine:latest\n"
        "RUN printf '#!/bin/sh\\nexit 91\\n' > /usr/local/bin/codex "
        "&& chmod 0755 /usr/local/bin/codex\n"
    )
    agent = type(
        "RegisteredAgent",
        (),
        {
            "name": "codex",
            "run_cmd": "codex exec --dangerously-bypass-approvals-and-sandbox",
            "install_cmds": (
                "printf '#!/bin/sh\\nexit 0\\n' > /tmp/pinned-codex "
                "&& chmod 0755 /tmp/pinned-codex "
                "&& ln /tmp/pinned-codex /usr/local/bin/codex",
            ),
        },
    )()
    (work / "Dockerfile").write_text(
        DockerImageBuilder(None)._work_dockerfile(
            agent=agent,  # type: ignore[arg-type]
            base_ref=base_tag,
            workdir=PurePosixPath("/workspace"),
            final_user="0",
        )
    )
    (work / "rsi-submit").write_text("#!/bin/sh\nexit 0\n")

    try:
        base_build = subprocess.run(
            ["docker", "build", "-q", "-t", base_tag, "."],
            cwd=base,
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        assert base_build.returncode == 0, base_build.stdout + base_build.stderr

        work_build = subprocess.run(
            ["docker", "build", "-q", "-t", work_tag, "."],
            cwd=work,
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
        assert work_build.returncode == 0, work_build.stdout + work_build.stderr

        execution = subprocess.run(
            ["docker", "run", "--rm", "--network", "none", work_tag, "codex"],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        assert execution.returncode == 0, execution.stdout + execution.stderr
    finally:
        subprocess.run(
            ["docker", "image", "rm", "-f", work_tag, base_tag],
            capture_output=True,
            check=False,
            text=True,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("effective_user", "expected_uid", "expected_gid"),
    [
        ("1001", "1001", "1001"),
        ("nobody", "65534", "65534"),
        ("2001:2002", "2001", "2002"),
        ("nobody:nogroup", "65534", "65534"),
    ],
)
def test_agent_identity_alias_matches_effective_user_in_real_container(
    effective_user,
    expected_uid,
    expected_gid,
):
    probe = subprocess.run(
        ["docker", "image", "inspect", "ubuntu:24.04"],
        capture_output=True,
        check=False,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"local ubuntu Docker image capability unavailable: {probe.stderr}")
    script = build_agent_identity_script(effective_user)
    script += (
        f'\ntest "$(id -u agent)" = "{expected_uid}"'
        f'\ntest "$(id -g agent)" = "{expected_gid}"'
        '\ntest "$(stat -c %u /home/agent)" = "$(id -u agent)"\n'
        'test "$(stat -c %g /home/agent)" = "$(id -g agent)"\n'
    )

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "ubuntu:24.04",
            "/bin/bash",
            "-lc",
            script,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_agent_identity_repairs_preexisting_agent_group_to_effective_gid():
    probe = subprocess.run(
        ["docker", "image", "inspect", "ubuntu:24.04"],
        capture_output=True,
        check=False,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(f"local ubuntu Docker image capability unavailable: {probe.stderr}")
    script = "groupadd -g 1234 agent\n" + build_agent_identity_script("1001")
    script += '\ntest "$(id -g agent)" = "1001"\n'

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "ubuntu:24.04",
            "/bin/bash",
            "-lc",
            script,
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
