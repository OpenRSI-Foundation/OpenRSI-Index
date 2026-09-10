import hashlib
import json
from pathlib import PurePosixPath

import pytest
from harbor.models.task.task import Task

from rsi_harness.errors import UnsupportedTaskError
from rsi_harness.models import (
    CompileOptions,
    GPUAllocation,
    GPUDevice,
    ImagePlan,
    JudgeGPUMode,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
)
from rsi_harness.task.compiler import HarborTaskCompiler
from tests.factories import DEFAULT_TASK_TOML, write_harbor_task


def append_toml(task_dir, text):
    config = task_dir / "task.toml"
    config.write_text(config.read_text() + "\n" + text)


def write_compose(task_dir, service_text, *, sidecar_text=""):
    path = task_dir / "environment" / "docker-compose.yaml"
    path.write_text("services:\n  main:\n" + service_text + sidecar_text)
    return path


def add_step(task_dir):
    append_toml(task_dir, '[[steps]]\nname = "phase-one"\n')


def add_sidecar(task_dir):
    write_compose(
        task_dir,
        "    build: .\n",
        sidecar_text="  database:\n    image: postgres:17\n",
    )


def add_host_volume(task_dir):
    write_compose(task_dir, "    build: .\n    volumes: ['./host:/data']\n")


def set_privileged(task_dir):
    write_compose(task_dir, "    build: .\n    privileged: true\n")


def set_host_network(task_dir):
    write_compose(task_dir, "    build: .\n    network_mode: host\n")


def set_verifier_environment(task_dir):
    append_toml(
        task_dir,
        '[verifier.environment]\ndocker_image = "judge:latest"\n',
    )


def test_compile_uses_official_harbor_model_and_gpu_requirement(tmp_path):
    """Bypassing Harbor would lose its name, paths, and canary normalization."""
    task_dir = write_harbor_task(
        tmp_path,
        instruction="<!-- benchmark canary marker -->\n\nImprove it.\n",
    )

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.task_id == "minimal-gpu"
    assert definition.instruction == "Improve it.\n"
    assert definition.workdir == PurePosixPath("/workspace")
    assert definition.gpu_requirement.count == 2
    assert definition.verifier.command == ("/bin/bash", "/tests/test.sh")
    assert definition.service.build_context == (task_dir / "environment").resolve()


@pytest.mark.parametrize("judge_count", (None, 0, 1))
def test_compile_accepts_explicit_cpu_work(tmp_path, judge_count):
    """Explicit zero must survive Harbor compilation without inheriting GPUs."""
    task_toml = DEFAULT_TASK_TOML.replace("gpus = 2", "gpus = 0")
    if judge_count is not None:
        task_toml += f"\n[metadata.rsi_harness.verifier]\ngpus = {judge_count}\n"
    task = write_harbor_task(tmp_path, task_toml=task_toml)

    definition = HarborTaskCompiler().compile(task, CompileOptions())

    assert definition.gpu_requirement.count == 0
    assert definition.gpu_requirement.name is None
    assert definition.verifier.gpu_count == (0 if judge_count is None else judge_count)


def test_compile_does_not_treat_missing_work_gpu_count_as_cpu(tmp_path):
    task = write_harbor_task(
        tmp_path, task_toml=DEFAULT_TASK_TOML.replace("gpus = 2\n", "")
    )

    with pytest.raises(UnsupportedTaskError, match="GPU requirement"):
        HarborTaskCompiler().compile(task, CompileOptions())


def test_compile_rejects_gpu_type_for_cpu_work(tmp_path):
    task = write_harbor_task(
        tmp_path,
        task_toml=DEFAULT_TASK_TOML.replace(
            "gpus = 2", 'gpus = 0\ngpu_types = ["H100"]'
        ),
    )

    with pytest.raises(UnsupportedTaskError, match="gpus = 0.*gpu_types"):
        HarborTaskCompiler().compile(task, CompileOptions())


@pytest.mark.parametrize(
    "raw_count", ("false", "true", "0.0", "1.0", '"0"', '"1"', "-1")
)
def test_compile_rejects_noninteger_or_negative_work_gpu_declaration(
    tmp_path, raw_count
):
    """Harbor's integer coercion must not turn malformed values into CPU intent."""
    task = write_harbor_task(
        tmp_path,
        task_toml=DEFAULT_TASK_TOML.replace("gpus = 2", f"gpus = {raw_count}"),
    )

    with pytest.raises(
        UnsupportedTaskError, match="environment.gpus.*non-negative integer"
    ):
        HarborTaskCompiler().compile(task, CompileOptions())


def test_compiles_namespaced_verifier_gpu_count(tmp_path):
    """The approved verifier extension must reach the compiled plan."""
    task = write_harbor_task(
        tmp_path,
        task_toml=DEFAULT_TASK_TOML
        + "\n[metadata.rsi_harness.verifier]\ngpus = 4\n",
    )

    definition = HarborTaskCompiler().compile(task, CompileOptions())

    assert definition.verifier.gpu_count == 4


def test_compiles_portable_cluster_asset_requirements(tmp_path):
    task = write_harbor_task(
        tmp_path,
        task_toml=DEFAULT_TASK_TOML
        + """

[[metadata.rsi_harness.assets]]
path = "/rsi-data/data/train/manifest.json"
phase = "work"
kind = "file"
min_bytes = 1

[[metadata.rsi_harness.assets]]
path = "/rsi-data/data/paloma"
phase = "judge"
kind = "directory"
min_entries = 1
""",
    )

    definition = HarborTaskCompiler().compile(task, CompileOptions())

    assert tuple(item.model_dump(mode="json") for item in definition.assets) == (
        {
            "path": "/rsi-data/data/train/manifest.json",
            "phase": "work",
            "kind": "file",
            "min_bytes": 1,
            "min_entries": 0,
        },
        {
            "path": "/rsi-data/data/paloma",
            "phase": "judge",
            "kind": "directory",
            "min_bytes": 0,
            "min_entries": 1,
        },
    )


@pytest.mark.parametrize(
    "metadata",
    (
        '[[metadata.rsi_harness.assets]]\npath = "relative"\nphase = "work"',
        '[[metadata.rsi_harness.assets]]\npath = "/x"\nphase = "other"',
        '[metadata.rsi_harness]\nassets = "not-a-list"',
    ),
)
def test_rejects_invalid_cluster_asset_requirements(tmp_path, metadata):
    task = write_harbor_task(
        tmp_path,
        task_toml=DEFAULT_TASK_TOML + "\n" + metadata + "\n",
    )

    with pytest.raises(UnsupportedTaskError, match="metadata.rsi_harness.assets"):
        HarborTaskCompiler().compile(task, CompileOptions())


def test_missing_verifier_gpu_metadata_defaults_to_zero(tmp_path):
    """Tasks without the extension must not reserve a verifier GPU."""
    definition = HarborTaskCompiler().compile(
        write_harbor_task(tmp_path), CompileOptions()
    )

    assert definition.verifier.gpu_count == 0


@pytest.mark.parametrize(
    "metadata",
    (
        {"rsi_harness": None},
        {"rsi_harness": {"verifier": None}},
    ),
)
def test_verifier_gpu_count_rejects_explicit_null_metadata_tables(tmp_path, metadata):
    """Explicit nulls cannot masquerade as absent private metadata tables."""
    task = Task(write_harbor_task(tmp_path), disable_verification=True)
    task.config.metadata.clear()
    task.config.metadata.update(metadata)

    with pytest.raises(
        UnsupportedTaskError, match="metadata.rsi_harness.verifier.gpus"
    ):
        HarborTaskCompiler()._verifier_gpu_count(task)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            "[metadata.rsi_harness.verifier]\ngpus = true",
            "metadata.rsi_harness.verifier.gpus",
        ),
        (
            '[metadata.rsi_harness.verifier]\ngpus = "4"',
            "metadata.rsi_harness.verifier.gpus",
        ),
        (
            "[metadata.rsi_harness.verifier]\ngpus = -1",
            "metadata.rsi_harness.verifier.gpus",
        ),
        (
            "[metadata.rsi_harness.verifier]\ngpus = [4]",
            "metadata.rsi_harness.verifier.gpus",
        ),
        (
            '[metadata]\nrsi_harness = "not-a-table"',
            "metadata.rsi_harness.verifier.gpus",
        ),
        (
            "[metadata.rsi_harness.verifier]\nextra = 1",
            "unknown verifier GPU metadata",
        ),
    ],
)
def test_compile_rejects_malformed_verifier_gpu_metadata(tmp_path, metadata, message):
    """Malformed private metadata must fail without exposing other metadata."""
    task = write_harbor_task(
        tmp_path,
        task_toml=(
            DEFAULT_TASK_TOML
            + "\n"
            + metadata
            + '\n[metadata.other]\nvalue = "secret"\n'
        ),
    )

    with pytest.raises(UnsupportedTaskError, match=message) as error:
        HarborTaskCompiler().compile(task, CompileOptions())

    assert "secret" not in str(error.value)


def test_compile_defaults_undeclared_gpu_shared_memory_to_one_gibibyte(tmp_path):
    """Leaving GPU tasks at Docker's 64 MiB default can break NCCL communication."""
    task_dir = write_harbor_task(tmp_path)

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.service.shm_size == "1g"


def test_compile_preserves_explicit_gpu_shared_memory(tmp_path):
    """The engine default must not override a task author's Compose limit."""
    task_dir = write_harbor_task(tmp_path)
    write_compose(task_dir, "    build: .\n    shm_size: 4g\n")

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.service.shm_size == "4g"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (add_step, "multi-step"),
        (add_sidecar, "sidecar"),
        (add_host_volume, "host volume"),
        (set_privileged, "privileged"),
        (set_host_network, "host network"),
        (set_verifier_environment, "independent verifier"),
    ],
)
def test_compile_rejects_unsupported_harbor_features(tmp_path, mutation, message):
    """Unsupported task semantics must fail before an Agent can start."""
    task_dir = write_harbor_task(tmp_path)
    mutation(task_dir)

    with pytest.raises(UnsupportedTaskError, match=message):
        HarborTaskCompiler().compile(task_dir, CompileOptions())


def test_compile_accepts_harbor_legacy_version_field(tmp_path):
    """Duplicating TOML parsing could reject Harbor's supported version alias."""
    task_toml = DEFAULT_TASK_TOML.replace("schema_version", "version", 1)
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.task_id == "minimal-gpu"


@pytest.mark.parametrize(
    "relative_path",
    ("instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"),
)
def test_compile_rejects_missing_required_task_files(tmp_path, relative_path):
    """A partial task cannot produce a runnable immutable definition."""
    task_dir = write_harbor_task(tmp_path)
    (task_dir / relative_path).unlink()

    with pytest.raises(UnsupportedTaskError, match="missing|required"):
        HarborTaskCompiler().compile(task_dir, CompileOptions())


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        ('os = "linux"', 'os = "windows"', "Windows"),
        ('workdir = "/workspace"', 'workdir = "workspace"', "WORKDIR"),
        (
            "gpus = 2",
            'tpu = { type = "v6e", topology = "2x2" }',
            "TPU",
        ),
        (
            "gpus = 2",
            'gpus = 2\nmcp_servers = [{ name = "tools", url = "http://mcp" }]',
            "MCP",
        ),
        (
            'workdir = "/workspace"',
            'workdir = "/workspace"\nhealthcheck = { command = "true" }',
            "healthcheck",
        ),
        (
            'workdir = "/workspace"',
            'workdir = "/workspace"\nskills_dir = "/skills"',
            "skills",
        ),
        (
            "gpus = 2",
            'gpus = 2\ngpu_types = ["H100", "A100"]',
            "GPU types",
        ),
    ],
)
def test_compile_rejects_unsupported_config_fields(tmp_path, before, after, message):
    """Nonempty Harbor fields that the plan cannot express must not vanish."""
    task_dir = write_harbor_task(
        tmp_path, task_toml=DEFAULT_TASK_TOML.replace(before, after)
    )

    with pytest.raises(UnsupportedTaskError, match=message):
        HarborTaskCompiler().compile(task_dir, CompileOptions())


def test_compile_rejects_collect_hooks_and_sidecar_artifacts(tmp_path):
    """Hooks or sidecar collection would depend on unsupported live services."""
    collect_task = write_harbor_task(tmp_path / "collect")
    append_toml(
        collect_task,
        '[[verifier.collect]]\ncommand = "dump"\nservice = "main"\n',
    )
    with pytest.raises(UnsupportedTaskError, match="collect hook"):
        HarborTaskCompiler().compile(collect_task, CompileOptions())

    artifact_task = write_harbor_task(tmp_path / "artifact")
    config = artifact_task / "task.toml"
    config.write_text(
        'artifacts = [{ source = "/var/log/db", service = "database" }]\n'
        + config.read_text()
    )
    with pytest.raises(UnsupportedTaskError, match="sidecar artifact"):
        HarborTaskCompiler().compile(artifact_task, CompileOptions())


def test_compile_preserves_unresolved_environment_and_phase_policy(tmp_path):
    """Resolving templates here could persist host secrets in the definition."""
    task_toml = (
        DEFAULT_TASK_TOML.replace(
            'network_mode = "no-network"',
            'network_mode = "no-network"\n'
            'env = { HOST = "${HOST_TOKEN}", FALLBACK = "${OPTIONAL:-safe}", '
            'LITERAL = "visible" }',
        )
        .replace(
            "[agent]\ntimeout_sec = 60",
            '[agent]\ntimeout_sec = 12\nuser = 1001\nnetwork_mode = "allowlist"\n'
            'allowed_hosts = ["API.EXAMPLE.COM"]',
        )
        .replace(
            "[verifier]\ntimeout_sec = 30",
            '[verifier]\ntimeout_sec = 7\nuser = "judge"\nnetwork_mode = "no-network"\n'
            'env = { JUDGE = "${JUDGE_TOKEN}", STATIC = "value" }',
        )
    )
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)

    definition = HarborTaskCompiler().compile(
        task_dir,
        CompileOptions(
            agent_timeout_seconds=9,
            primary_reward="accuracy",
            disable_stop_hook=True,
        ),
    )
    payload = definition.model_dump_json()

    assert definition.service.environment == (
        ("FALLBACK", "${OPTIONAL:-safe}"),
        ("HOST", "${HOST_TOKEN}"),
        ("LITERAL", "visible"),
    )
    assert definition.agent.environment == definition.service.environment
    assert definition.agent.secret_env_names == ("HOST_TOKEN", "OPTIONAL")
    assert definition.agent.user == "1001"
    assert definition.agent.timeout_seconds == 9
    assert definition.agent.install_stop_hook is False
    assert definition.agent.network.mode == "allowlist"
    assert definition.agent.network.allowlist == ("api.example.com",)
    verifier_environment = dict(definition.verifier.environment)
    assert verifier_environment["JUDGE"] == "${JUDGE_TOKEN}"
    assert verifier_environment["STATIC"] == "value"
    assert definition.verifier.secret_env_names == (
        "HOST_TOKEN",
        "JUDGE_TOKEN",
        "OPTIONAL",
    )
    assert definition.verifier.user == "judge"
    assert definition.verifier.timeout_seconds == 7
    assert definition.verifier.primary_reward == "accuracy"
    assert definition.verifier.network.mode == "no-network"
    assert "secret-from-host" not in payload


def test_compile_preserves_explicit_zero_agent_timeout(tmp_path):
    """Treating zero as absent changes Harbor's phase-timeout semantics."""
    task_toml = DEFAULT_TASK_TOML.replace("timeout_sec = 60", "timeout_sec = 0")
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.agent.timeout_seconds == 0.0


def test_compile_accepts_official_default_build_timeout(tmp_path):
    """Rejecting Harbor's explicit default would reject otherwise standard tasks."""
    task_toml = DEFAULT_TASK_TOML.replace(
        "gpus = 2", "gpus = 2\nbuild_timeout_sec = 600.0"
    )
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.task_id == "minimal-gpu"


def test_compile_preserves_absent_workdir_and_official_resources(tmp_path):
    task_toml = DEFAULT_TASK_TOML.replace(
        'workdir = "/workspace"\n',
        "",
    ).replace(
        "gpus = 2",
        "gpus = 2\n"
        "build_timeout_sec = 1800.0\n"
        "cpus = 2\n"
        'memory = "8G"\n'
        'storage = "15G"',
    )
    task_dir = write_harbor_task(
        tmp_path,
        task_toml=task_toml,
        dockerfile="FROM ubuntu:24.04\nWORKDIR /\n",
    )

    with pytest.warns(DeprecationWarning) as warnings:
        definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert [str(warning.message) for warning in warnings] == [
        "The 'memory' field is deprecated. Use 'memory_mb' instead.",
        "The 'storage' field is deprecated. Use 'storage_mb' instead.",
    ]
    assert definition.workdir is None
    assert definition.service.workdir is None
    assert definition.service.cpus == 2
    assert definition.service.memory_mb == 8192
    assert definition.service.storage_mb == 15360
    assert definition.service.build_timeout_seconds == 1800.0


def test_compile_preserves_absent_workdir_for_image_inspection(tmp_path):
    """A Docker image must supply its own directory when Harbor declares none."""
    task_toml = DEFAULT_TASK_TOML.replace('workdir = "/workspace"\n', "")
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.workdir is None
    assert definition.service.workdir is None


@pytest.mark.parametrize(
    ("source", "declaration"),
    [
        ("Harbor", '""'),
        ("Harbor", '"."'),
        ("Compose", '""'),
        ("Compose", "."),
    ],
)
def test_compile_normalizes_explicit_root_equivalent_workdir_declarations(
    tmp_path, source, declaration
):
    """Empty/dot declarations must use Docker root rather than be rejected."""
    task_toml = DEFAULT_TASK_TOML
    if source == "Harbor":
        task_toml = task_toml.replace(
            'workdir = "/workspace"', f"workdir = {declaration}"
        )
    else:
        task_toml = task_toml.replace('workdir = "/workspace"\n', "")
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)
    if source == "Compose":
        write_compose(task_dir, f"    build: .\n    working_dir: {declaration}\n")

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.workdir == PurePosixPath("/")
    assert definition.service.workdir == PurePosixPath("/")


@pytest.mark.parametrize("work_count", (0, 2))
def test_compile_requires_matching_task_and_compose_gpu_counts(tmp_path, work_count):
    """Choosing either of two conflicting GPU declarations is unsafe."""
    task_dir = write_harbor_task(
        tmp_path,
        task_toml=DEFAULT_TASK_TOML.replace("gpus = 2", f"gpus = {work_count}"),
    )
    write_compose(
        task_dir,
        "    build: .\n"
        "    working_dir: /workspace\n"
        "    deploy:\n"
        "      resources:\n"
        "        reservations:\n"
        "          devices:\n"
        "            - driver: nvidia\n"
        "              count: 1\n"
        "              capabilities: [gpu]\n",
    )

    with pytest.raises(UnsupportedTaskError, match="GPU count"):
        HarborTaskCompiler().compile(task_dir, CompileOptions())


def test_compile_accepts_compose_all_when_it_is_the_only_gpu_declaration(tmp_path):
    """The all sentinel must mean all GPUs allocated by the caller."""
    task_toml = DEFAULT_TASK_TOML.replace("gpus = 2\n", "")
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)
    write_compose(
        task_dir,
        "    build: .\n"
        "    working_dir: /workspace\n"
        "    deploy:\n"
        "      resources:\n"
        "        reservations:\n"
        "          devices:\n"
        "            - driver: nvidia\n"
        "              count: all\n"
        "              capabilities: [gpu]\n",
    )

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.gpu_requirement.count == "all"


@pytest.mark.parametrize(
    ("harbor_image", "compose_source", "message"),
    (
        pytest.param(
            "registry.test/base:latest",
            "    build: .\n    working_dir: /workspace\n",
            "Compose build.*Harbor image",
            id="compose-build-with-harbor-image",
        ),
        pytest.param(
            "registry.test/harbor:latest",
            "    image: registry.test/compose:latest\n    working_dir: /workspace\n",
            "Compose and Harbor image",
            id="conflicting-images",
        ),
    ),
)
def test_compile_rejects_conflicting_compose_and_harbor_image_sources(
    tmp_path, harbor_image, compose_source, message
) -> None:
    task_toml = DEFAULT_TASK_TOML.replace(
        'workdir = "/workspace"',
        f'workdir = "/workspace"\ndocker_image = "{harbor_image}"',
    )
    task_dir = write_harbor_task(tmp_path, task_toml=task_toml)
    write_compose(task_dir, compose_source)

    with pytest.raises(UnsupportedTaskError, match=message):
        HarborTaskCompiler().compile(task_dir, CompileOptions())


def test_compile_explicit_image_does_not_infer_dockerfile_build(tmp_path) -> None:
    task_dir = write_harbor_task(tmp_path)
    write_compose(
        task_dir,
        "    image: registry.test/compose:latest\n    working_dir: /workspace\n",
    )

    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())

    assert definition.service.image == "registry.test/compose:latest"
    assert definition.service.build_context is None


def test_compile_digests_are_separate_and_source_is_never_mutated(tmp_path):
    """Compilation must be read-only and each cache boundary independently keyed."""
    task_dir = write_harbor_task(tmp_path)

    def source_state(root):
        return {
            path.relative_to(root).as_posix(): (
                path.stat().st_ino,
                path.stat().st_mtime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            for path in root.rglob("*")
            if path.is_file()
        }

    before = source_state(task_dir)
    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())
    after = source_state(task_dir)

    assert after == before
    assert definition.source_digest
    assert definition.instruction_digest
    assert definition.tests_digest
    assert definition.environment_digest
    assert (
        len(
            {
                definition.source_digest,
                definition.instruction_digest,
                definition.tests_digest,
                definition.environment_digest,
            }
        )
        == 4
    )


def test_finalize_adds_only_caller_owned_runtime_values(tmp_path):
    """Finalization must not need or accept any resolved secret mapping."""
    definition = HarborTaskCompiler().compile(
        write_harbor_task(tmp_path), CompileOptions()
    )
    paths = RunPaths(
        root=(tmp_path / "run").resolve(),
        workspace=(tmp_path / "workspace").resolve(),
        logs=(tmp_path / "logs").resolve(),
    )
    images = ImagePlan(
        base_ref="base",
        work_ref="work",
        judge_ref="judge",
        workdir=PurePosixPath("/effective-workdir"),
        rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        base_digest="sha256:base",
        work_digest="sha256:work",
        judge_digest="sha256:judge",
    )
    allocation = GPUAllocation(
        devices=(GPUDevice(index=0, uuid="GPU-one", name="H100"),)
    )
    gpu_plan = RunGPUPlan(
        authorized_pool=allocation,
        work=allocation,
        judge=GPUAllocation(),
        judge_mode=JudgeGPUMode.FREEZE_ONLY,
    )

    plan = HarborTaskCompiler().finalize(
        definition, images, gpu_plan, paths, "docker-rootfs"
    )

    assert plan.task is not definition
    assert plan.task.workdir == PurePosixPath("/effective-workdir")
    assert plan.task.service.workdir == PurePosixPath("/effective-workdir")
    assert plan.images is images
    assert plan.gpu_plan is gpu_plan
    assert plan.schema_version == 2
    assert plan.paths is paths
    assert plan.workdir == PurePosixPath("/effective-workdir")
    assert plan.rootfs_snapshot_mode is RootfsSnapshotMode.SPLIT_WORKDIR
    assert plan.snapshot_kind == "docker-rootfs"


def test_write_rsi_loop_metadata_is_generic_and_cache_only(tmp_path):
    """Metadata must describe whole-WORKDIR scoring without adding archive behavior."""
    compiler = HarborTaskCompiler()
    task_dir = write_harbor_task(tmp_path)
    definition = compiler.compile(task_dir, CompileOptions(score_direction="minimize"))
    cache_task_dir = tmp_path / "run-cache" / "task"

    metadata_path = compiler.write_rsi_loop_metadata(
        definition, destination=cache_task_dir
    )
    payload = json.loads(metadata_path.read_text())

    assert metadata_path.parent == cache_task_dir
    assert metadata_path == cache_task_dir.resolve() / "task.json"
    assert payload["task_id"] == "minimal-gpu"
    assert payload["submit_paths"] == ["."]
    assert payload["cwd"] == "/workspace"
    assert payload["judge"] == {
        "parser": "structured_json",
        "score_direction": "minimize",
        "selection": "score_first",
    }
    assert not (task_dir / "task.json").exists()
    with pytest.raises(UnsupportedTaskError, match="source task"):
        compiler.write_rsi_loop_metadata(definition, task_dir)
