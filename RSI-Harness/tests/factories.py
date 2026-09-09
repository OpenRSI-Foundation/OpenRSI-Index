from pathlib import Path, PurePosixPath

from rsi_harness.models import (
    AgentPlan,
    GPUAllocation,
    GPURequirement,
    ImagePlan,
    JudgeGPUMode,
    MainServiceConfig,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
    RunPlan,
    TaskDefinition,
    VerifierPlan,
)

DEFAULT_TASK_TOML = """\
schema_version = "1.4"

[task]
name = "rsi/minimal-gpu"

[environment]
os = "linux"
workdir = "/workspace"
gpus = 2
network_mode = "no-network"

[agent]
timeout_sec = 60

[verifier]
timeout_sec = 30
"""


def write_harbor_task(
    tmp_path: Path,
    *,
    instruction: str = "Improve the contents of the workspace.\n",
    task_toml: str = DEFAULT_TASK_TOML,
    dockerfile: str = "FROM ubuntu:24.04\nWORKDIR /workspace\n",
    test_script: str = (
        "#!/bin/bash\n"
        "set -eu\n"
        "printf '{\"reward\": 1}\\n' > /logs/verifier/reward.json\n"
    ),
) -> Path:
    """Create a minimal standard Harbor GPU task from explicit text inputs."""
    task_dir = tmp_path / "minimal-gpu"
    environment_dir = task_dir / "environment"
    tests_dir = task_dir / "tests"
    environment_dir.mkdir(parents=True)
    tests_dir.mkdir()
    (task_dir / "instruction.md").write_text(instruction)
    (task_dir / "task.toml").write_text(task_toml)
    (environment_dir / "Dockerfile").write_text(dockerfile)
    (tests_dir / "test.sh").write_text(test_script)
    return task_dir


def write_cluster_task(tmp_path: Path) -> Path:
    """Self-contained two-GPU task for scheduler and profile regression tests."""
    fixture = Path(__file__).parent / "fixtures/tasks/minimal-bluevela-multinode"
    task_toml = (
        (fixture / "task.toml").read_text()
        .replace("gpus = 16", "gpus = 2")
        .replace("gpus = 8", "gpus = 2")
        .replace("timeout_sec = 180", "timeout_sec = 3600")
        .replace("timeout_sec = 30", "timeout_sec = 3600")
    )
    return write_harbor_task(tmp_path, task_toml=task_toml)


def make_run_plan(
    tmp_path: Path, *, secret_env_names: tuple[str, ...] = ()
) -> RunPlan:
    root = tmp_path.resolve()
    task = TaskDefinition(
        task_id="minimal-gpu",
        source_dir=root / "task",
        source_digest="task-digest",
        instruction_digest="instruction-digest",
        tests_digest="tests-digest",
        workdir=PurePosixPath("/workspace"),
        service=MainServiceConfig(),
        gpu_requirement=GPURequirement(count=1),
        verifier=VerifierPlan(
            command=("/bin/bash", "/tests/test.sh"),
            secret_env_names=secret_env_names,
        ),
        agent=AgentPlan(name="codex", secret_env_names=secret_env_names),
    )
    return RunPlan(
        schema_version=2,
        task=task,
        workdir=PurePosixPath("/workspace"),
        rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        images=ImagePlan(
            base_ref=f"base@sha256:{'a' * 64}",
            work_ref=f"work@sha256:{'b' * 64}",
            judge_ref=f"judge@sha256:{'c' * 64}",
            workdir=PurePosixPath("/workspace"),
            rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        ),
        gpu_plan=RunGPUPlan(
            authorized_pool=GPUAllocation(),
            work=GPUAllocation(),
            judge=GPUAllocation(),
            judge_mode=JudgeGPUMode.FREEZE_ONLY,
        ),
        paths=RunPaths(
            root=root / "runs",
            workspace=root / "workspace",
            logs=root / "logs",
        ),
        snapshot_kind="docker-rootfs",
    )
