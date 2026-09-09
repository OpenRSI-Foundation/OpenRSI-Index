from __future__ import annotations

import hashlib
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from rsi_harness.cluster.base import ClusterRunRequest
from rsi_harness.cluster.bluevela.adapter import (
    BlueVelaClusterAdapter,
    derive_resources,
)
from rsi_harness.cluster.bluevela.engine import load_engine_payload
from rsi_harness.cluster.bluevela.image import ImagePlan as SIFImagePlan
from rsi_harness.cluster.bluevela.resources import derive_resource_plan
from rsi_harness.cluster.config import (
    ApptainerBindProfile,
    ClusterProfile,
    load_cluster_profile,
)
from rsi_harness.cluster.schedulers.lsf import (
    LSFJobResult,
    LSFJobSpec,
    LSFScheduler,
)
from rsi_harness.errors import InfrastructureError, SetupError
from rsi_harness.models import (
    AgentAuthSource,
    AssetRequirement,
    CompileOptions,
    FrozenRewardMap,
    GPURequirement,
    JudgeGPUMode,
    SubmissionReport,
    SubmissionStatus,
)
from rsi_harness.runtime.artifacts import RunArtifactWriter
from tests.factories import write_cluster_task


def _task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "minimal-gpu"
    if not task_dir.exists():
        write_cluster_task(tmp_path)
    return task_dir.resolve()


def _profile(tmp_path: Path) -> ClusterProfile:
    base = load_cluster_profile("bluevela", {"USER": "alice"})
    root = (tmp_path / "cluster").resolve()
    apptainer = tmp_path / "apptainer"
    apptainer.write_text("#!/bin/sh\nexit 0\n")
    apptainer.chmod(0o700)
    return base.model_copy(
        update={
            "storage": base.storage.model_copy(
                update={
                    "run_root": root / "runs",
                    "image_cache": root / "images",
                    "logs_root": root / "logs",
                    "hf_home": root / "hf",
                    "hf_datasets_cache": root / "datasets",
                }
            ),
            "builder": base.builder.model_copy(update={"temp_root": tmp_path}),
            "apptainer": base.apptainer.model_copy(
                update={"binary": apptainer}
            ),
        }
    )


def _request(tmp_path: Path, *, dry_run: bool) -> ClusterRunRequest:
    return ClusterRunRequest(
        task_dir=_task_dir(tmp_path),
        agent_name="codex",
        options=CompileOptions(agent_name="codex"),
        logs_root=(tmp_path / "ignored-local-logs").resolve(),
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        agent_auth=AgentAuthSource.LOCAL,
        dry_run=dry_run,
    )


def _claude_request(tmp_path: Path, *, dry_run: bool) -> ClusterRunRequest:
    return ClusterRunRequest(
        task_dir=_task_dir(tmp_path),
        agent_name="claude-code",
        options=CompileOptions(agent_name="claude-code"),
        logs_root=(tmp_path / "ignored-local-logs").resolve(),
        agent_auth=AgentAuthSource.LOCAL,
        dry_run=dry_run,
    )


class RecordingScheduler:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.specs: list[LSFJobSpec] = []
        self.renderer = LSFScheduler()

    def render_submit(self, spec: LSFJobSpec) -> tuple[str, ...]:
        return self.renderer.render_submit(spec)

    def require_name_available(self, name: str) -> None:
        stage = "build" if "-build-" in name else "run"
        self.events.append(f"check-{stage}")

    def submit(self, spec: LSFJobSpec) -> str:
        self.specs.append(spec)
        stage = "build" if spec.gpu_count == 0 else "run"
        self.events.append(f"submit-{stage}")
        return str(100 + len(self.specs))

    def wait(self, job_id: str, **_kwargs) -> LSFJobResult:
        spec = self.specs[int(job_id) - 101]
        stage = "build" if spec.gpu_count == 0 else "run"
        self.events.append(f"wait-{stage}")
        if stage == "build":
            assignments = {
                line.split("=", 1)[0]: shlex.split(line.split("=", 1)[1])[0]
                for line in spec.script_path.read_text().splitlines()
                if line.startswith(("final_sif=", "final_sha="))
            }
            sif = Path(assignments["final_sif"])
            sha = Path(assignments["final_sha"])
            sif.parent.mkdir(parents=True, exist_ok=True)
            sif.write_bytes(b"test-sif")
            digest = hashlib.sha256(b"test-sif").hexdigest()
            sha.write_text(f"{digest}  {sif.name}\n")
        else:
            payload = load_engine_payload(spec.script_path.with_suffix(".json"))
            writer = RunArtifactWriter(payload.run_plan, run_id=payload.run_id)
            writer.start()
            (writer.root / "agent_prompt.md").write_text(
                payload.run_plan.task.instruction
            )
            (writer.root / "agent_output.txt").write_text("agent output\n")
            (writer.root / "run_agent.log").write_text("trial log\n")
            writer.record_submission(
                SubmissionReport(
                    round_id="agent-1",
                    status=SubmissionStatus.COMPLETED,
                    rewards=FrozenRewardMap({"reward": 1.0}),
                    score=1.0,
                    output="passed\n",
                )
            )
            writer.finalize()
        return LSFJobResult(job_id=job_id, state="DONE", exit_code=0)


def test_dry_run_resolves_four_gpus_without_mutation(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    events: list[tuple[str, object]] = []
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        clock=lambda: datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.149.0",
    )

    result = adapter.run(_request(tmp_path, dry_run=True))

    assert result.job_ids == ()
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()
    dry_run = dict(events)["dry_run"]
    assert dry_run["resources"] == {
        "work_gpus": 2,
        "verifier_gpus": 2,
        "total_gpus": 4,
        "cpu_slots": 8,
        "memory_mb": 65536,
        "local_tmp_mb": 15360,
        "build_walltime": "02:00",
        "run_walltime": "02:15",
    }
    assert "num=4:mode=exclusive_process" in dry_run["run_argv"]
    assert "select[tmp>=15360] span[hosts=1]" in dry_run["run_argv"]
    assert dry_run["binds"]
    assert all(binding.endswith(":ro") for binding in dry_run["binds"])
    assert not any("legacy-public" in binding for binding in dry_run["binds"])
    assert dry_run["agent_version"] == "0.149.0"


def test_dry_run_automatically_dispatches_task_gpu_fields_to_multinode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    events: list[tuple[str, object]] = []
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        clock=lambda: datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.149.0",
    )
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=32),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 16}),
        }
    )
    monkeypatch.setattr(adapter, "_compile", lambda _request: definition)

    result = adapter.run(_request(tmp_path, dry_run=True))

    assert result.job_ids == ()
    assert scheduler.events == []
    assert not profile.storage.run_root.exists()
    dry_run = dict(events)["dry_run"]
    assert dry_run["resources"] is None
    assert dry_run["multi_node"] == {
        "work": {"gpu_count": 32, "node_count": 4},
        "verifier": {"gpu_count": 16, "node_count": 2},
        "total_nodes": 6,
        "gpus_per_node": 8,
        "cpu_slots_per_node": 8,
        "memory_mb_per_node": 65536,
        "shared_workspace_mb": 15360,
        "node_tmp_mb": 71680,
        "build_walltime": "02:00",
        "run_walltime": "02:15",
    }
    assert dry_run["pool_policy"] == (
        "ordered Work prefix; ordered Judge suffix"
    )
    assert "48" in dry_run["run_argv"]
    resource = dry_run["run_argv"][dry_run["run_argv"].index("-R") + 1]
    assert "span[ptile=8]" in resource
    assert "span[hosts=1]" not in resource
    assert "num=8:mode=exclusive_process" in dry_run["run_argv"]
    assert "rsi-multinode" not in repr(dry_run)
    assert "legacy-public:/proj" not in dry_run["binds"]
    assert any("/rsi-data" in item for item in dry_run["binds"])


def test_multinode_exclusive_policy_comes_from_cluster_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = _profile(tmp_path)
    profile = base.model_copy(
        update={"scheduler": base.scheduler.model_copy(update={"exclusive": False})}
    )
    events: list[tuple[str, object]] = []
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=RecordingScheduler(),
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.149.0",
    )
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=16),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 8}),
        }
    )
    monkeypatch.setattr(adapter, "_compile", lambda _request: definition)

    adapter.run(_request(tmp_path, dry_run=True))

    run_argv = dict(events)["dry_run"]["run_argv"]
    assert "-x" not in run_argv


def test_unhealthy_host_exclusions_come_from_cluster_profile(
    tmp_path: Path,
) -> None:
    base = _profile(tmp_path)
    profile = base.model_copy(
        update={
            "scheduler": base.scheduler.model_copy(
                update={"excluded_hosts": ("worker-a",)}
            )
        }
    )
    events: list[tuple[str, object]] = []
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=RecordingScheduler(),
        event_callback=lambda name, value: events.append((name, value)),
        agent_version_resolver=lambda _name: "0.149.0",
    )

    adapter.run(_request(tmp_path, dry_run=True))

    dry_run = dict(events)["dry_run"]
    for argv in (dry_run["build_argv"], dry_run["run_argv"]):
        resource = argv[argv.index("-R") + 1]
        assert "hname!='worker-a'" in resource


def test_run_plan_overlaps_work_and_judge_when_single_node_requires_reuse(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=RecordingScheduler(),
        agent_version_resolver=lambda _name: "0.149.0",
    )
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=8),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 4}),
        }
    )
    resources = derive_resources(definition, profile)
    image = SIFImagePlan(
        cache_key="0" * 64,
        sif_path=(tmp_path / "task.sif").resolve(),
        sha256_path=(tmp_path / "task.sif.sha256").resolve(),
        cache_hit=False,
    )

    plan = adapter._run_plan(definition, image, resources, tmp_path / "run")

    assert plan.gpu_plan.judge_mode is JudgeGPUMode.RELEASE_ALL
    assert plan.gpu_plan.authorized_pool.uuids == tuple(
        f"LSF-{index}" for index in range(8)
    )
    assert plan.gpu_plan.work.uuids == plan.gpu_plan.authorized_pool.uuids
    assert plan.gpu_plan.judge.uuids == plan.gpu_plan.work.uuids[:4]


def test_multinode_run_plan_uses_disjoint_host_local_placeholders(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=RecordingScheduler(),
        agent_version_resolver=lambda _name: "0.149.0",
    )
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=16),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 8}),
        }
    )
    resources = derive_resource_plan(definition, profile).multi_node
    assert resources is not None
    image = SIFImagePlan(
        cache_key="0" * 64,
        sif_path=(tmp_path / "task.sif").resolve(),
        sha256_path=(tmp_path / "task.sif.sha256").resolve(),
        cache_hit=False,
    )

    plan = adapter._run_plan(definition, image, resources, tmp_path / "run")

    assert len(plan.gpu_plan.work.devices) == 16
    assert len(plan.gpu_plan.judge.devices) == 8
    assert plan.gpu_plan.judge_mode is JudgeGPUMode.DISJOINT
    assert plan.gpu_plan.work.uuids[:9] == (
        "WORK-000:GPU-0",
        "WORK-000:GPU-1",
        "WORK-000:GPU-2",
        "WORK-000:GPU-3",
        "WORK-000:GPU-4",
        "WORK-000:GPU-5",
        "WORK-000:GPU-6",
        "WORK-000:GPU-7",
        "WORK-001:GPU-0",
    )
    assert plan.gpu_plan.judge.uuids[0] == "JUDGE-000:GPU-0"
    assert set(plan.gpu_plan.work.uuids).isdisjoint(plan.gpu_plan.judge.uuids)


def test_multinode_shared_workspace_capacity_is_checked_separately_from_tmp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = _profile(tmp_path)
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=RecordingScheduler(),
        agent_version_resolver=lambda _name: "0.149.0",
    )
    definition = adapter._compile(_request(tmp_path, dry_run=True))
    definition = definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=16),
            "verifier": definition.verifier.model_copy(update={"gpu_count": 8}),
        }
    )
    resources = derive_resource_plan(definition, profile).multi_node
    assert resources is not None
    resources = resources.model_copy(update={"shared_workspace_mb": 2048})
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.adapter.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=1024 * 1024 * 1024),
    )

    with pytest.raises(InfrastructureError, match="shared GPFS workspace"):
        adapter._require_shared_workspace(tmp_path, resources)


def test_standard_multinode_fixture_needs_no_task_owned_cluster_directory(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=RecordingScheduler(),
        agent_version_resolver=lambda _name: "0.149.0",
    )
    fixture = (
        Path(__file__).resolve().parents[2]
        / "fixtures/tasks/minimal-bluevela-multinode"
    )
    request = ClusterRunRequest(
        task_dir=fixture,
        agent_name="codex",
        options=CompileOptions(agent_name="codex"),
        logs_root=(tmp_path / "logs").resolve(),
        model="gpt-test",
        dry_run=True,
    )

    definition = adapter._compile(request)
    resources = derive_resource_plan(definition, profile).multi_node

    assert resources is not None
    assert resources.work.node_count == 2
    assert resources.verifier.node_count == 1
    assert resources.total_nodes == 3
    assert not (fixture / "cluster").exists()
    assert definition.agent.timeout_seconds >= 180
    instruction = definition.instruction
    assert "one foreground command" in instruction
    assert "Do not add" in instruction
    assert "--node_rank" not in instruction
    assert "--rdzv" not in instruction
    combined = "\n".join(
        path.read_text()
        for path in (fixture / "solution/solve.sh", fixture / "tests/test.sh")
    )
    assert "torchrun --nnodes" in combined
    assert "bsub" not in combined
    assert "blaunch" not in combined
    assert "LSB_" not in combined
    assert "apptainer" not in combined.lower()
    verifier = (fixture / "tests/test.sh").read_text()
    assert "/task-tools/judge.py" in verifier
    assert "/task-tools/smoke.py" not in verifier
    judge_source = (fixture / "environment/judge.py").read_text()
    assert "/logs/verifier/judge-ranks" in judge_source
    assert ".write_text" not in "\n".join(
        line for line in judge_source.splitlines() if "/workspace" in line
    )


def test_cache_miss_waits_for_build_then_run_and_records_manifest(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        clock=lambda: datetime(2026, 8, 23, 12, 0, tzinfo=UTC),
        agent_version_resolver=lambda _name: "0.149.0",
    )

    result = adapter.run(_request(tmp_path, dry_run=False))

    assert scheduler.events == [
        "check-build",
        "submit-build",
        "wait-build",
        "check-run",
        "submit-run",
        "wait-run",
    ]
    assert result.job_ids == ("101", "102")
    assert scheduler.specs[0].gpu_count == 0
    assert scheduler.specs[0].local_tmp_mb == profile.builder.min_tmp_mb
    assert scheduler.specs[1].gpu_count == 4
    assert scheduler.specs[1].local_tmp_mb == 15360
    payload = load_engine_payload(scheduler.specs[1].script_path.with_suffix(".json"))
    assert payload.source_root.parent.name == "control"
    assert payload.source_root.name == "engine-source"
    assert (payload.source_root / "src/rsi_harness/__init__.py").is_file()
    assert result.log_dir.is_dir()
    manifest = json.loads(
        (profile.storage.run_root / result.run_id / "RUN_INFO.json").read_text()
    )
    assert manifest["build_job_id"] == "101"
    assert manifest["run_job_id"] == "102"
    assert manifest["resources"]["total_gpus"] == 4
    assert manifest["state"] == "completed"
    assert "auth.json" not in json.dumps(manifest)


@pytest.mark.parametrize("multi_node", (False, True))
def test_missing_declared_asset_stops_after_cpu_build_before_gpu_submission(
    tmp_path: Path, monkeypatch, multi_node: bool,
) -> None:
    profile = _profile(tmp_path)
    asset_root = tmp_path / "assets"
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={
                    "work_binds": (
                        ApptainerBindProfile(
                            source=asset_root,
                            target=PurePosixPath("/rsi-data"),
                            read_only=True,
                        ),
                    )
                }
            )
        }
    )
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        agent_version_resolver=lambda _name: "0.149.0",
    )
    base_definition = adapter._compile(_request(tmp_path, dry_run=False))
    definition = base_definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=16 if multi_node else 2),
            "verifier": base_definition.verifier.model_copy(
                update={"gpu_count": 8 if multi_node else 2}
            ),
            "assets": (
                AssetRequirement(
                    path=PurePosixPath("/rsi-data/train/manifest.json"),
                    phase="work",
                    kind="file",
                    min_bytes=1,
                ),
            ),
        }
    )
    monkeypatch.setattr(adapter, "_compile", lambda _request: definition)

    with pytest.raises(SetupError, match="manifest.json"):
        adapter.run(_request(tmp_path, dry_run=False))

    assert scheduler.events == ["check-build", "submit-build", "wait-build"]
    assert all(spec.gpu_count == 0 for spec in scheduler.specs)


@pytest.mark.parametrize("multi_node", (False, True))
def test_valid_declared_assets_allow_gpu_submission(
    tmp_path: Path, monkeypatch, multi_node: bool,
) -> None:
    profile = _profile(tmp_path)
    asset_root = tmp_path / "assets"
    manifest = asset_root / "train/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n")
    profile = profile.model_copy(
        update={
            "apptainer": profile.apptainer.model_copy(
                update={
                    "work_binds": (
                        ApptainerBindProfile(
                            source=asset_root,
                            target=PurePosixPath("/rsi-data"),
                            read_only=True,
                        ),
                    )
                }
            )
        }
    )
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        agent_version_resolver=lambda _name: "0.149.0",
    )
    base_definition = adapter._compile(_request(tmp_path, dry_run=False))
    definition = base_definition.model_copy(
        update={
            "gpu_requirement": GPURequirement(count=16 if multi_node else 2),
            "verifier": base_definition.verifier.model_copy(
                update={"gpu_count": 8 if multi_node else 2}
            ),
            "assets": (
                AssetRequirement(
                    path=PurePosixPath("/rsi-data/train/manifest.json"),
                    phase="work",
                    kind="file",
                    min_bytes=1,
                ),
            ),
        }
    )
    monkeypatch.setattr(adapter, "_compile", lambda _request: definition)

    result = adapter.run(_request(tmp_path, dry_run=False))

    assert result.status.value == "completed"
    assert scheduler.events[-3:] == ["check-run", "submit-run", "wait-run"]


def test_claude_run_resolves_registered_launcher_into_engine_payload(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "claude-config"
    config.mkdir()
    credentials = config / ".credentials.json"
    credentials.write_text('{"claudeAiOauth":{"accessToken":"secret"}}')
    credentials.chmod(0o600)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config))
    resolved: list[str] = []

    def which(name: str) -> str | None:
        resolved.append(name)
        return sys.executable if name == "claude" else None

    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.adapter.shutil.which", which
    )
    profile = _profile(tmp_path)
    scheduler = RecordingScheduler()
    adapter = BlueVelaClusterAdapter(
        profile,
        scheduler=scheduler,
        agent_version_resolver=lambda _name: "2.1.246",
    )

    result = adapter.run(_claude_request(tmp_path, dry_run=False))

    assert result.status.value == "completed"
    payload = load_engine_payload(scheduler.specs[1].script_path.with_suffix(".json"))
    assert payload.agent_launcher == "claude"
    assert payload.agent_binary == Path(sys.executable).resolve()
    assert "claude" in resolved
    assert "claude-code" not in resolved


def test_job_names_distinguish_agents_for_concurrent_task_runs(tmp_path: Path) -> None:
    adapter = BlueVelaClusterAdapter(
        _profile(tmp_path),
        scheduler=RecordingScheduler(),
        agent_version_resolver=lambda _name: "test",
    )

    codex_name = adapter._job_name("vlmr1-rec-curriculum", "run", "codex")
    claude_name = adapter._job_name(
        "vlmr1-rec-curriculum", "run", "claude-code"
    )

    assert codex_name != claude_name
    assert "codex" in codex_name
    assert "claude-code" in claude_name
