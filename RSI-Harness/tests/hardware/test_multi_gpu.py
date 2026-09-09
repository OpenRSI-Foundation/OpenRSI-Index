from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import CompileOptions, GPURequirement, RunRequest, RunStatus
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.gpu import NvidiaSmiInventory, resolve_allocation
from rsi_harness.runtime.network import DockerIptablesFirewallBackend
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.config import RSILoopConfig

pytestmark = pytest.mark.gpu

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tasks" / "minimal-multi-gpu"


def _caller_selectors() -> tuple[str, ...]:
    raw = os.environ.get("RSI_TEST_GPUS")
    if raw is None:
        pytest.skip("set RSI_TEST_GPUS to an explicit comma-separated GPU allocation")
    selectors = tuple(part.strip() for part in raw.split(","))
    if not selectors or any(not selector for selector in selectors):
        pytest.fail("RSI_TEST_GPUS must contain only non-empty GPU selectors")
    if len(selectors) < 2:
        pytest.skip("multi-GPU acceptance requires at least two caller selectors")
    return selectors


def _all_task(tmp_path: Path, *, judge_count: int) -> Path:
    task = tmp_path / "minimal-all-gpu"
    shutil.copytree(FIXTURE, task)
    task_toml = task / "task.toml"
    task_toml.write_text(
        task_toml.read_text()
        .replace("gpus = 2\n", "", 1)
        .replace(
            "[metadata.rsi_harness.verifier]\ngpus = 2\n",
            f"[metadata.rsi_harness.verifier]\ngpus = {judge_count}\n",
        )
    )
    (task / "environment" / "docker-compose.yaml").write_text(
        """services:
  main:
    build: .
    working_dir: /workspace
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""
    )
    return task


def _read_uuids(path: Path) -> tuple[str, ...]:
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def _source_state(task_dir: Path) -> tuple[str, dict[Path, tuple[int, int]]]:
    return (
        hash_tree(task_dir),
        {
            path.relative_to(task_dir): (path.stat().st_ino, path.stat().st_mtime_ns)
            for path in task_dir.rglob("*")
            if path.is_file()
        },
    )


def _run_real_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    task_dir: Path,
    selectors: tuple[str, ...],
    requirement: GPURequirement,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[str, ...], ...],
    tuple[tuple[str, ...], ...],
]:
    from rsi_harness.runtime.production import ProductionRuntimeServices

    inventory = NvidiaSmiInventory()
    devices = inventory.list_devices()
    task_id = HarborTaskCompiler().compile(task_dir, CompileOptions()).task_id
    allocation = resolve_allocation(
        requirement,
        requested=selectors,
        inventory=devices,
    )
    source_before = _source_state(task_dir)

    try:
        client = docker.from_env()
        client.ping()
    except Exception as error:
        pytest.skip(f"multi-GPU Docker authority unavailable: {error}")
    firewall_trace: list[str] = []

    def recording_firewall_runner(
        command: list[str],
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(command, capture_output=True, check=False, text=True)
        if command[2:3] == ["-S"]:
            lines = result.stdout.splitlines()
            if len(command) == 4 and command[3] in {"DOCKER-USER", "INPUT"}:
                lines = [line for line in lines if "rsi-" in line]
            elif len(command) == 3:
                lines = [
                    line
                    for line in lines
                    if "rsi-" in line or "RSI_" in line
                ]
            firewall_trace.append(
                f"$ {' '.join(command)}\n"
                + "\n".join(lines)
                + (f"\nstderr: {result.stderr.strip()}" if result.stderr else "")
            )
        return result

    firewall = DockerIptablesFirewallBackend(
        client, runner=recording_firewall_runner
    )
    if not firewall.probe():
        pytest.skip(
            "multi-GPU fail-closed gate: host DOCKER-USER/INPUT firewall "
            "authority is unavailable"
        )
    data = tmp_path / "data"
    class EnumeratingAgent(RSILoopAgentAdapter):
        def prepare(self, request):
            prepared = super().prepare(request)
            return replace(
                prepared,
                command=(
                    "/bin/bash",
                    "-lc",
                    "set -eu\n"
                    "nvidia-smi --query-gpu=uuid --format=csv,noheader "
                    "| sed 's/[[:space:]]*$//' > work-gpus.txt\n"
                    "rsi-submit\n"
                    "rsi-submit\n",
                ),
            )

    created: list[dict[str, object]] = []
    original_create = DockerContainerRuntime.create

    def recording_create(self, spec, *, planned_name=None):
        ref = original_create(self, spec, planned_name=planned_name)
        inspected = client.containers.get(ref.container_id)
        inspected.reload()
        created.append(
            {
                "run_id": self.recovery_labels["rsi-harness.run-id"],
                "role": ref.role,
                "spec_gpu_uuids": spec.gpu_allocation.uuids,
                "workdir": spec.workdir,
                "mounts": copy.deepcopy(inspected.attrs["Mounts"]),
                "device_requests": copy.deepcopy(
                    inspected.attrs["HostConfig"].get("DeviceRequests")
                ),
            }
        )
        return ref

    monkeypatch.setattr(DockerContainerRuntime, "create", recording_create)
    services = ProductionRuntimeServices(
        data_root=data,
        logs_root=tmp_path / "logs",
        docker_client=client,
        inventory=inventory,
        rsi_loop_config=RSILoopConfig(),
        firewall_backend=firewall,
        agent_adapter_factory=lambda config, runtime: EnumeratingAgent(
            config, runtime=runtime
        ),
    )

    result = None
    try:
        result = services.run(
            RunRequest(
                task_dir=task_dir.resolve(),
                agent_name="codex",
                gpu_selectors=selectors,
                options=CompileOptions(
                    agent_name="codex", primary_reward="reward", max_submissions=2
                ),
            )
        )

        lease_path = data / "leases" / f"{result.run_id}.json"
        diagnostic = None
        if lease_path.is_file():
            diagnostic = json.loads(lease_path.read_text()).get("error")
        assert result.status == RunStatus.COMPLETED, (
            f"{diagnostic}\n\nFirewall attestation trace:\n"
            + "\n\n".join(firewall_trace)
        )
        assert result.total_rounds == 2
        assert result.best_score == 1
        run_created = [
            entry for entry in created if entry["run_id"] == result.run_id
        ]
        run_roles = [entry["role"] for entry in run_created]
        assert run_roles.count("work") == 1
        assert run_roles.count("judge") == 2
        volume_names: set[str] = set()
        for entry in run_created:
            assert entry["spec_gpu_uuids"] == allocation.uuids
            assert entry["workdir"] == PurePosixPath("/workspace")
            mounts = entry["mounts"]
            assert isinstance(mounts, list)
            workdir_mounts = [
                mount
                for mount in mounts
                if mount.get("Destination") == "/workspace"
            ]
            assert len(workdir_mounts) == 1
            workdir_mount = workdir_mounts[0]
            assert workdir_mount["Type"] == "volume"
            assert isinstance(workdir_mount["Name"], str)
            volume_names.add(workdir_mount["Name"])
            assert workdir_mount["RW"] is (entry["role"] == "work")
            device_requests = entry["device_requests"]
            assert isinstance(device_requests, list)
            assert len(device_requests) == 1
            assert tuple(device_requests[0]["DeviceIDs"]) == allocation.uuids
        assert len(volume_names) == 1
        assert _source_state(task_dir) == source_before
        assert not list(tmp_path.rglob("*.tar*"))
        labels = {"label": f"rsi-harness.run-id={result.run_id}"}
        assert client.containers.list(all=True, filters=labels) == []
        assert client.networks.list(filters=labels) == []
        verifier_root = (
            tmp_path
            / "logs"
            / "runs"
            / result.run_id
            / task_id
            / "verifier"
        )
        work = tuple(
            _read_uuids(
                verifier_root
                / f"agent-{round_number}"
                / "work-gpus.txt"
            )
            for round_number in (1, 2)
        )
        judges = tuple(
            _read_uuids(
                verifier_root
                / f"agent-{round_number}"
                / "judge-gpus.txt"
            )
            for round_number in (1, 2)
        )
        return allocation.uuids, work, judges
    finally:
        if result is not None:
            services.cleanup(result.run_id, delete_workspace=True)
            labels = {"label": f"rsi-harness.run-id={result.run_id}"}
            assert client.containers.list(all=True, filters=labels) == []
            assert client.networks.list(filters=labels) == []
            assert client.images.list(filters=labels) == []
            assert client.volumes.list(filters=labels) == []


def _assert_exact_visibility(
    expected: tuple[str, ...],
    work_rounds: tuple[tuple[str, ...], ...],
    judge_rounds: tuple[tuple[str, ...], ...],
) -> None:
    assert work_rounds == (expected, expected)
    assert judge_rounds == (expected, expected)
    assert all(set(visible) == set(expected) for visible in work_rounds)
    assert all(set(visible) == set(expected) for visible in judge_rounds)
    assert work_rounds == judge_rounds


def test_two_selected_physical_gpus_are_identical_in_work_and_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selectors = _caller_selectors()[:2]
    definition = HarborTaskCompiler().compile(FIXTURE, CompileOptions())
    assert definition.gpu_requirement.count == 2

    visible = _run_real_round(
        tmp_path,
        monkeypatch,
        task_dir=FIXTURE,
        selectors=selectors,
        requirement=definition.gpu_requirement,
    )

    _assert_exact_visibility(*visible)


def test_count_all_uses_every_and_only_caller_selected_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selectors = _caller_selectors()
    task_dir = _all_task(tmp_path, judge_count=len(selectors))
    definition = HarborTaskCompiler().compile(task_dir, CompileOptions())
    assert definition.gpu_requirement.count == "all"

    visible = _run_real_round(
        tmp_path,
        monkeypatch,
        task_dir=task_dir,
        selectors=selectors,
        requirement=definition.gpu_requirement,
    )

    _assert_exact_visibility(*visible)
