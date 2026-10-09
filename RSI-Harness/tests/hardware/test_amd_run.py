"""A complete run on AMD GPUs: Work, a release-all or disjoint Judge, a score.

Needs root (a run installs host firewall rules and overlay snapshots), an AMD
host, an explicit pool, and a local image that ships PyTorch for ROCm::

    sudo -E env RSI_TEST_GPUS=2,3 RSI_TEST_ROCM_IMAGE=rocm/pytorch:latest \
        .venv/bin/python -m pytest -m gpu tests/hardware/test_amd_run.py
"""

from __future__ import annotations

import json
import os
import textwrap
import threading
import time
from pathlib import Path

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    AgentRunRequest,
    AgentRunResult,
    CompileOptions,
    GPUDevice,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.gpu import AmdSmiInventory, host_gpu_inventory
from rsi_loop.harness.config import RSILoopConfig

pytestmark = pytest.mark.gpu

REJECTED = "release all Work GPU processes before retrying"

SOLVE = """\
set -eu
cat > /workspace/scale.py <<'PY'
def scale(x):
    return 2 * x
PY
python3 -c 'import torch; print(torch.cuda.get_device_name(0))' \\
    > /workspace/work-gpu.txt
cat > /workspace/holder.py <<'PY'
import os, sys, time
import torch
held = torch.empty(16 << 20, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
open(sys.argv[1], "w").write(str(os.getpid()))
while not os.path.exists(sys.argv[2]):
    time.sleep(0.05)
del held
os.remove(sys.argv[1])
PY
"""

VERIFIER = """\
#!/bin/bash
set -u
python3 - <<'PY'
import json, os, sys
sys.path.insert(0, "/workspace")
result = {"reward": 0}
try:
    import torch
    from scale import scale
    x = torch.randn(4096, device="cuda")
    result.update(
        correct=bool(torch.equal(scale(x), 2 * x)),
        judge_visible_gpus=torch.cuda.device_count(),
        judge_gpu=torch.cuda.get_device_name(0),
        judge_render_nodes=sorted(
            n for n in os.listdir("/dev/dri") if n.startswith("renderD")
        ),
        expected_gpu_uuids=os.environ.get("RSI_HARNESS_EXPECTED_GPU_UUIDS"),
        work_gpu=open("/workspace/work-gpu.txt").read().strip(),
    )
    result["reward"] = int(result["correct"] and result["judge_visible_gpus"] == 1)
except Exception as error:
    result["error"] = f"{type(error).__name__}: {error}"
print("AMD-VERIFIER " + json.dumps(result))
json.dump({"reward": result["reward"]}, open("/logs/verifier/reward.json", "w"))
PY
"""


def _task(tmp_path: Path, image: str) -> Path:
    task = tmp_path / "amd-gpu-run"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        textwrap.dedent(
            """\
            schema_version = "1.4"

            [task]
            name = "rsi/amd-gpu-run"

            [environment]
            os = "linux"
            workdir = "/workspace"
            gpus = 1
            network_mode = "no-network"

            [agent]
            timeout_sec = 600
            network_mode = "public"

            [verifier]
            timeout_sec = 300
            network_mode = "no-network"

            [metadata.rsi_harness.verifier]
            gpus = 1
            """
        )
    )
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {image}\n\nRUN mkdir -p /workspace\nWORKDIR /workspace\n"
    )
    (task / "instruction.md").write_text("Write scale.py and submit.\n")
    (task / "tests" / "test.sh").write_text(VERIFIER)
    (task / "tests" / "test.sh").chmod(0o755)
    return task


class ScriptedAgent(RSILoopAgentAdapter):
    """Solve in Work; in release-all mode, first submit while holding the GPU."""

    def __init__(self, config: RSILoopConfig, *, runtime: object, hold: bool) -> None:
        super().__init__(config, runtime=runtime)  # type: ignore[arg-type]
        self._hold = hold
        self.rejection: str | None = None
        self.attempts = 0

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        control = self._control
        assert control is not None
        runtime = self._require_runtime()
        environment = dict(request.prepared.environment)
        environment.update(
            {"RSI_JUDGE_URL": control.submit_url, "RSI_TOKEN": control.token}
        )

        def run(command: str, timeout: float = 300) -> AgentRunResult:
            return runtime.exec(  # type: ignore[attr-defined]
                request.container,
                ("/bin/bash", "-lc", command),
                timeout_seconds=timeout,
                environment=environment,
            )

        try:
            solved = run(SOLVE)
            assert solved.exit_code == 0, solved.output
            if self._hold:
                holder = threading.Thread(
                    target=run,
                    args=("python3 /workspace/holder.py /tmp/held /tmp/release",),
                    daemon=True,
                )
                holder.start()
                ready = run(
                    "for _ in $(seq 600); do test -s /tmp/held && exit 0; "
                    "sleep 0.1; done; exit 1"
                )
                assert ready.exit_code == 0, ready.output
                rejected = run("rsi-submit")
                assert rejected.exit_code not in (None, 0)
                self.rejection = rejected.output
                run("touch /tmp/release")
                holder.join(60)
                assert not holder.is_alive()
            # KFD can lag a process exit briefly; a rejected preflight is free.
            for _ in range(20):
                self.attempts += 1
                submitted = run("rsi-submit")
                if submitted.exit_code == 0 or REJECTED not in submitted.output:
                    break
                time.sleep(1)
            assert submitted.exit_code == 0, submitted.output
            return AgentRunResult(exit_code=0, output=submitted.output)
        finally:
            self.clear_transient_bindings()


@pytest.mark.parametrize("mode", ["release-all", "disjoint"])
def test_amd_run_scores_on_exactly_the_allocated_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    if os.geteuid() != 0:
        pytest.skip("a complete run needs root for firewall rules and snapshots")
    raw = os.environ.get("RSI_TEST_GPUS")
    image = os.environ.get("RSI_TEST_ROCM_IMAGE")
    if not raw or not image:
        pytest.skip("set RSI_TEST_GPUS and RSI_TEST_ROCM_IMAGE")
    inventory = host_gpu_inventory()
    if not isinstance(inventory, AmdSmiInventory):
        pytest.skip("host is not an AMD GPU host")
    pool = tuple(part.strip() for part in raw.split(","))
    if mode == "release-all":
        pool = pool[:1]
    elif len(pool) < 2:
        pytest.skip("disjoint mode needs two GPUs in RSI_TEST_GPUS")
    else:
        pool = pool[:2]
    devices = {str(device.index): device for device in inventory.list_devices()}
    devices.update({device.uuid: device for device in devices.values()})
    work_gpu: GPUDevice = devices[pool[0]]
    judge_gpu: GPUDevice = devices[pool[-1]]

    created: dict[str, dict[str, object]] = {}
    original_create = DockerContainerRuntime.create

    def recording_create(self, spec, *, planned_name=None):
        ref = original_create(self, spec, planned_name=planned_name)
        # Image preparation also creates a GPU-less "work" container first.
        if ref.role in {"work", "judge"} and spec.gpu_allocation.devices:
            attrs = client.containers.get(ref.container_id).attrs
            host = attrs["HostConfig"]
            environment = dict(
                item.split("=", 1) for item in attrs["Config"]["Env"] or ()
            )
            created[ref.role] = {
                "devices": [item["PathOnHost"] for item in host["Devices"] or ()],
                "device_requests": host["DeviceRequests"] or [],
                "nvidia": environment.get("NVIDIA_VISIBLE_DEVICES"),
            }
        return ref

    monkeypatch.setattr(DockerContainerRuntime, "create", recording_create)
    client = docker.from_env()
    agents: list[ScriptedAgent] = []

    def agent_factory(config: RSILoopConfig, runtime: object) -> ScriptedAgent:
        agents.append(
            ScriptedAgent(config, runtime=runtime, hold=mode == "release-all")
        )
        return agents[-1]

    from rsi_harness.runtime.production import ProductionRuntimeServices

    events: list[tuple[str, object]] = []
    services = ProductionRuntimeServices(
        data_root=tmp_path / "data",
        logs_root=tmp_path / "logs",
        docker_client=client,
        rsi_loop_config=RSILoopConfig(),
        agent_adapter_factory=agent_factory,
        event_callback=lambda name, value: events.append((name, value)),
    )
    result = None
    try:
        result = services.run(
            RunRequest(
                task_dir=_task(tmp_path, image).resolve(),
                agent_name="codex",
                gpu_selectors=pool,
                options=CompileOptions(
                    agent_name="codex", primary_reward="reward", max_submissions=1
                ),
            )
        )
        assert result.status is RunStatus.COMPLETED, events
        assert result.total_rounds == 1
        assert result.best_score == 1

        for role, gpu in (("work", work_gpu), ("judge", judge_gpu)):
            assert created[role] == {
                "devices": ["/dev/kfd", str(gpu.render_node)],
                "device_requests": [],
                "nvidia": "void",
            }
        if mode == "release-all":
            assert agents[0].rejection is not None
            assert REJECTED in agents[0].rejection

        verified = [
            json.loads(line.split("AMD-VERIFIER ", 1)[1])
            for log in (tmp_path / "logs").rglob("agent-1.log")
            for line in log.read_text().splitlines()
            if line.startswith("AMD-VERIFIER ")
        ]
        assert verified, "the Judge printed no verifier result"
        assert verified[-1]["judge_visible_gpus"] == 1
        assert verified[-1]["judge_render_nodes"] == [judge_gpu.render_node.name]
        assert verified[-1]["expected_gpu_uuids"] == judge_gpu.uuid
        assert verified[-1]["work_gpu"] == verified[-1]["judge_gpu"]
    finally:
        if result is not None:
            services.cleanup(result.run_id, delete_workspace=True)
