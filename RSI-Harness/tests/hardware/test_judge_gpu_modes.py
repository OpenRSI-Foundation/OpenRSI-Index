from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Literal

import docker
import pytest

from rsi_harness.integrations.rsi_loop import RSILoopAgentAdapter
from rsi_harness.models import (
    AgentPrepareRequest,
    AgentRunRequest,
    AgentRunResult,
    CompileOptions,
    ContainerRef,
    ContainerSpec,
    GPUDevice,
    PreparedAgent,
    RunPlan,
    RunRequest,
    RunStatus,
)
from rsi_harness.runtime.docker import DockerContainerRuntime
from rsi_harness.runtime.gpu import NvidiaSmiInventory
from rsi_harness.runtime.network import DockerIptablesFirewallBackend
from rsi_harness.runtime.recovery import LeaseStore
from rsi_harness.runtime.snapshot import OverlaySnapshotBackend
from rsi_harness.task.compiler import HarborTaskCompiler
from rsi_harness.task.digest import hash_tree
from rsi_loop.harness.config import RSILoopConfig

FIXTURE = Path(__file__).parents[1] / "fixtures" / "tasks" / "minimal-multi-gpu"
Mode = Literal["freeze-only", "disjoint", "release-all"]


def _controlled_run_plan(tmp_path: Path) -> RunPlan:
    return RunPlan.model_validate(
        {
            "schema_version": 2,
            "task": {
                "task_id": "minimal-gpu",
                "source_dir": tmp_path / "task",
                "source_digest": "task-digest",
                "instruction_digest": "instruction-digest",
                "tests_digest": "tests-digest",
                "workdir": "/workspace",
                "service": {},
                "gpu_requirement": {"count": 1},
                "verifier": {"command": ["/bin/bash", "/tests/test.sh"]},
                "agent": {"name": "codex"},
            },
            "workdir": "/workspace",
            "rootfs_snapshot_mode": "split-workdir",
            "images": {
                "base_ref": f"base@sha256:{'a' * 64}",
                "work_ref": f"work@sha256:{'b' * 64}",
                "judge_ref": f"judge@sha256:{'c' * 64}",
                "workdir": "/workspace",
                "rootfs_snapshot_mode": "split-workdir",
            },
            "gpu_plan": {
                "authorized_pool": {},
                "work": {},
                "judge": {},
                "judge_mode": "freeze-only",
            },
            "paths": {
                "root": tmp_path / "data" / "run-authority",
                "workspace": tmp_path / "data" / "run-authority" / "workspace",
                "logs": tmp_path / "logs",
            },
            "snapshot_kind": "docker-rootfs",
        }
    )


@dataclass(slots=True)
class HolderObservation:
    marker_pid: int
    host_pid: int
    gpu_processes: tuple[tuple[str, int], ...]


@dataclass(slots=True)
class AgentEvidence:
    run_plan: RunPlan | None = None
    work_uuids: tuple[str, ...] | None = None
    before: HolderObservation | None = None
    after: HolderObservation | None = None
    first_rejection: AgentRunResult | None = None
    first_http_status: int | None = None
    rejection_had_no_verifier_dir: bool | None = None
    rejection_had_no_formal_report: bool | None = None
    rejection_consumed_no_history: bool | None = None
    holder_stopped: bool = False
    outputs: list[str] = field(default_factory=list)


def _cuda_holder_script() -> str:
    return textwrap.dedent(
        """\
        import ctypes
        import os
        import sys
        import time
        from pathlib import Path

        ready = Path(sys.argv[1])
        stop = Path(sys.argv[2])
        cuda = ctypes.CDLL("libcuda.so.1")

        cuda.cuInit.argtypes = [ctypes.c_uint]
        cuda.cuInit.restype = ctypes.c_int
        cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cuda.cuDeviceGet.restype = ctypes.c_int
        cuda.cuCtxCreate_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int
        ]
        cuda.cuCtxCreate_v2.restype = ctypes.c_int
        cuda.cuMemAlloc_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t
        ]
        cuda.cuMemAlloc_v2.restype = ctypes.c_int
        cuda.cuMemFree_v2.argtypes = [ctypes.c_uint64]
        cuda.cuMemFree_v2.restype = ctypes.c_int
        cuda.cuCtxDestroy_v2.argtypes = [ctypes.c_void_p]
        cuda.cuCtxDestroy_v2.restype = ctypes.c_int

        def checked(name, code):
            if code != 0:
                raise RuntimeError(f"{name} failed with CUDA result {code}")

        device = ctypes.c_int()
        context = ctypes.c_void_p()
        allocation = ctypes.c_uint64()
        try:
            checked("cuInit", cuda.cuInit(0))
            checked("cuDeviceGet", cuda.cuDeviceGet(ctypes.byref(device), 0))
            checked(
                "cuCtxCreate_v2",
                cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device),
            )
            checked(
                "cuMemAlloc_v2",
                cuda.cuMemAlloc_v2(ctypes.byref(allocation), 16 * 1024 * 1024),
            )
            ready.write_text(f"{os.getpid()}\\n")
            while not stop.exists():
                time.sleep(0.05)
        finally:
            ready.unlink(missing_ok=True)
            free_result = None
            destroy_result = None
            if allocation.value:
                free_result = cuda.cuMemFree_v2(allocation)
            if context.value:
                destroy_result = cuda.cuCtxDestroy_v2(context)
            if free_result is not None:
                checked("cuMemFree_v2", free_result)
            if destroy_result is not None:
                checked("cuCtxDestroy_v2", destroy_result)
        """
    )


def _selectors() -> tuple[str, str, str, str]:
    raw = os.environ.get("RSI_TEST_GPUS")
    if raw is None:
        pytest.fail("RSI_TEST_GPUS must authorize exactly four physical GPUs")
    selected = tuple(part.strip() for part in raw.split(","))
    if len(selected) != 4 or any(not part for part in selected):
        pytest.fail("RSI_TEST_GPUS must contain exactly four non-empty selectors")
    if len(set(selected)) != 4:
        pytest.fail("RSI_TEST_GPUS must contain four distinct selectors")
    return selected  # type: ignore[return-value]


def _resolve_selected(
    selectors: tuple[str, ...], inventory: tuple[GPUDevice, ...]
) -> tuple[GPUDevice, ...]:
    by_selector = {
        selector: device
        for device in inventory
        for selector in (str(device.index), device.uuid)
    }
    unknown = tuple(selector for selector in selectors if selector not in by_selector)
    if unknown:
        pytest.fail(f"RSI_TEST_GPUS contains unknown NVIDIA selectors: {unknown}")
    devices = tuple(by_selector[selector] for selector in selectors)
    if len({device.uuid for device in devices}) != len(devices):
        pytest.fail("RSI_TEST_GPUS selectors do not resolve to distinct physical GPUs")
    return devices


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, check=False, text=True)


def _gpu_processes(*, capability_gate: bool = False) -> tuple[tuple[str, int], ...]:
    result = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ]
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "nvidia-smi exited unsuccessfully"
        message = f"NVIDIA process inventory unavailable: {detail}"
        if capability_gate:
            pytest.skip(message)
        raise AssertionError(message)
    rows: list[tuple[str, int]] = []
    try:
        for line in result.stdout.splitlines():
            if line.strip():
                uuid, pid = (part.strip() for part in line.split(",", 1))
                rows.append((uuid, int(pid)))
    except (TypeError, ValueError) as error:
        message = f"NVIDIA process inventory is ambiguous: {error}"
        if capability_gate:
            pytest.skip(message)
        raise AssertionError(message) from error
    return tuple(rows)


def _hardware_gate(
    tmp_path: Path,
) -> tuple[
    docker.DockerClient,
    NvidiaSmiInventory,
    DockerIptablesFirewallBackend,
    OverlaySnapshotBackend,
    tuple[GPUDevice, ...],
]:
    selectors = _selectors()
    inventory = NvidiaSmiInventory()
    try:
        devices = inventory.list_devices()
    except Exception as error:
        pytest.skip(f"NVIDIA inventory gate unavailable: {error}")
    selected = _resolve_selected(selectors, devices)
    selected_uuids = {device.uuid for device in selected}
    active = tuple(
        row for row in _gpu_processes(capability_gate=True) if row[0] in selected_uuids
    )
    if active:
        pytest.skip(
            "NVIDIA authorized-pool idle gate found existing processes; refusing "
            f"to interfere: {active}"
        )

    try:
        client = docker.from_env()
        client.ping()
    except Exception as error:
        pytest.skip(f"Docker authority gate unavailable: {error}")

    firewall = DockerIptablesFirewallBackend(client)
    if not firewall.probe():
        pytest.skip(
            "INPUT/DOCKER-USER fail-closed firewall authority gate unavailable; "
            "no CUDA holder was started"
        )

    data = tmp_path / "data"
    snapshot = OverlaySnapshotBackend(data)
    capabilities = snapshot.probe(data)
    if not (
        capabilities.atomic
        and capabilities.immutable
        and capabilities.copy_on_write
        and capabilities.cleanup
    ):
        pytest.skip(
            "snapshot capability gate unavailable; no CUDA holder was started: "
            f"{capabilities.reason}"
        )
    return client, inventory, firewall, snapshot, selected


def _task_copy(tmp_path: Path, mode: Mode) -> Path:
    task = tmp_path / f"minimal-multi-gpu-{mode}"
    shutil.copytree(FIXTURE, task)
    if mode == "freeze-only":
        task_toml = task / "task.toml"
        task_toml.write_text(
            task_toml.read_text().replace(
                "\n[metadata.rsi_harness.verifier]\ngpus = 2\n", "\n"
            )
        )
    dockerfile = task / "environment" / "Dockerfile"
    dockerfile.write_text(
        dockerfile.read_text()
        + "\nENV NVIDIA_VISIBLE_DEVICES=all\n"
        + "\nRUN apt-get update && DEBIAN_FRONTEND=noninteractive "
        "apt-get install -y python3 && rm -rf /var/lib/apt/lists/*\n"
        "COPY gpu_holder.py /usr/local/libexec/rsi-gpu-holder.py\n"
    )
    (task / "environment" / "gpu_holder.py").write_text(_cuda_holder_script())
    return task


def _source_state(task: Path) -> tuple[str, dict[Path, tuple[int, int]]]:
    return (
        hash_tree(task),
        {
            path.relative_to(task): (path.stat().st_ino, path.stat().st_mtime_ns)
            for path in task.rglob("*")
            if path.is_file()
        },
    )


def _assert_exec_ok(result: AgentRunResult, action: str) -> None:
    assert result.exit_code == 0 and not result.timed_out, (
        f"{action} failed with exit={result.exit_code}: {result.output}"
    )


def _enumerate_work_gpus(runtime: object, request: AgentRunRequest) -> tuple[str, ...]:
    result = runtime.exec(  # type: ignore[attr-defined]
        request.container,
        (
            "/bin/bash",
            "-lc",
            "set -eu\n"
            "nvidia-smi --query-gpu=uuid --format=csv,noheader "
            "| sed 's/[[:space:]]*$//' "
            "| tee /workspace/work-gpus.txt\n",
        ),
        timeout_seconds=10,
    )
    _assert_exec_ok(result, "Work GPU enumeration")
    visible = tuple(line.strip() for line in result.output.splitlines() if line.strip())
    assert visible and len(set(visible)) == len(visible), (
        f"Work GPU enumeration was empty or duplicated: {visible}"
    )
    return visible


def _host_namespace_pids(host_pid: int) -> tuple[int, ...]:
    try:
        status = Path(f"/proc/{host_pid}/status").read_text()
        raw = next(
            line.removeprefix("NSpid:").strip()
            for line in status.splitlines()
            if line.startswith("NSpid:")
        )
        pids = tuple(int(value) for value in raw.split())
    except (OSError, StopIteration, ValueError) as error:
        raise AssertionError(
            f"cannot map holder host PID {host_pid} into its PID namespace: {error}"
        ) from error
    assert pids, f"holder host PID {host_pid} has an empty namespace PID map"
    return pids


def _host_command_line(host_pid: int) -> tuple[str, ...]:
    try:
        raw = Path(f"/proc/{host_pid}/cmdline").read_bytes()
    except OSError as error:
        raise AssertionError(
            f"cannot inspect holder host PID {host_pid} command line: {error}"
        ) from error
    return tuple(value.decode(errors="strict") for value in raw.split(b"\0") if value)


def _resolve_holder_host_pid(
    table: dict[str, object],
    *,
    marker_pid: int,
    namespace_pids=_host_namespace_pids,
    command_line=_host_command_line,
) -> int:
    titles = table["Titles"]
    processes = table["Processes"]
    assert isinstance(titles, list) and isinstance(processes, list)
    pid_column = titles.index("PID")
    expected_arguments = (
        "/usr/local/libexec/rsi-gpu-holder.py",
        "/workspace/holder.ready",
        "/workspace/holder.stop",
    )
    matches: list[int] = []
    for row in processes:
        host_pid = int(row[pid_column])
        nspids = namespace_pids(host_pid)
        argv = command_line(host_pid)
        if (
            nspids[-1] == marker_pid
            and argv
            and Path(argv[0]).name == "python3"
            and argv[1:] == expected_arguments
        ):
            matches.append(host_pid)
    assert len(matches) == 1, (
        f"ready marker PID {marker_pid} did not identify exactly one holder: {matches}"
    )
    return matches[0]


def _submission_http_status(output: str) -> int:
    statuses = tuple(
        int(match)
        for match in re.findall(
            r"(?m)^rsi-submit: judge returned HTTP ([0-9]{3})$", output
        )
    )
    assert statuses == (409,), (
        f"release-all first response must be HTTP 409, observed {statuses}"
    )
    return statuses[0]

def _observe_holder(
    runtime: object,
    client: docker.DockerClient,
    request: AgentRunRequest,
) -> HolderObservation:
    exec_result = runtime.exec(  # type: ignore[attr-defined]
        request.container,
        (
            "/bin/bash",
            "-lc",
            'set -eu; pid=$(cat /workspace/holder.ready); kill -0 "$pid"; '
            "printf '%s\\n' \"$pid\"",
        ),
        timeout_seconds=5,
    )
    _assert_exec_ok(exec_result, "holder liveness inspection")
    marker_pid = int(exec_result.output.strip())

    container = client.containers.get(request.container.container_id)
    table = container.top(ps_args="-eo pid,args")
    host_pid = _resolve_holder_host_pid(table, marker_pid=marker_pid)
    return HolderObservation(
        marker_pid=marker_pid,
        host_pid=host_pid,
        gpu_processes=_gpu_processes(),
    )


def _run_artifact_root(plan: RunPlan) -> Path:
    return (
        plan.paths.logs
        / "runs"
        / plan.paths.root.name
        / plan.task.task_id
    )


def _run_verifier_dir(plan: RunPlan, round_id: str) -> Path:
    return _run_artifact_root(plan) / "verifier" / round_id


class HardwareAgent(RSILoopAgentAdapter):
    def __init__(
        self,
        config: RSILoopConfig,
        *,
        runtime: object,
        client: docker.DockerClient,
        mode: Mode,
        evidence: AgentEvidence,
    ) -> None:
        super().__init__(config, runtime=runtime)  # type: ignore[arg-type]
        self._client = client
        self._mode = mode
        self._evidence = evidence
        self._holder_thread: threading.Thread | None = None
        self._holder_results: list[AgentRunResult] = []
        self._holder_errors: list[BaseException] = []
        self._holder_started = False
        self._run_plan: RunPlan | None = None

    def prepare(self, request: AgentPrepareRequest) -> PreparedAgent:
        assert self._run_plan is None, "Hardware Agent already captured a RunPlan"
        prepared = super().prepare(request)
        self._run_plan = request.run_plan
        self._evidence.run_plan = request.run_plan
        return prepared

    def run(self, request: AgentRunRequest) -> AgentRunResult:
        control = self._control
        assert control is not None
        plan = self._run_plan
        assert plan is not None, "Hardware Agent has no prepared RunPlan"
        assert self._evidence.run_plan is plan, (
            "Hardware Agent RunPlan attestation failed"
        )
        runtime = self._require_runtime()
        environment = dict(request.prepared.environment)
        environment.update(
            {"RSI_JUDGE_URL": control.submit_url, "RSI_TOKEN": control.token}
        )
        output: list[str] = []
        agent_result: AgentRunResult | None = None
        primary_error: BaseException | None = None
        try:
            self._evidence.work_uuids = _enumerate_work_gpus(runtime, request)
            self._start_holder(runtime, request)
            self._evidence.before = _observe_holder(runtime, self._client, request)

            if self._mode == "release-all":
                rejected = runtime.exec(
                    request.container,
                    ("/bin/bash", "-lc", "rsi-submit"),
                    timeout_seconds=30,
                    environment=environment,
                )
                self._evidence.first_rejection = rejected
                self._evidence.first_http_status = _submission_http_status(
                    rejected.output
                )
                output.append(rejected.output)
                assert rejected.exit_code not in (None, 0)
                assert (
                    "release all Work GPU processes before retrying" in rejected.output
                )
                artifact_root = _run_artifact_root(plan)
                verifier_dir = _run_verifier_dir(plan, "agent-1")
                self._evidence.rejection_had_no_verifier_dir = not verifier_dir.exists()
                self._evidence.rejection_had_no_formal_report = not (
                    artifact_root / "submissions"
                ).exists()
                state = json.loads((artifact_root / "evolve_state.json").read_text())
                self._evidence.rejection_consumed_no_history = (
                    state["submissions"] == []
                )
                self._evidence.after = _observe_holder(runtime, self._client, request)
                self._stop_holder(runtime, request)
                submitted = runtime.exec(
                    request.container,
                    ("/bin/bash", "-lc", "rsi-submit"),
                    timeout_seconds=30,
                    environment=environment,
                )
            else:
                submitted = runtime.exec(
                    request.container,
                    ("/bin/bash", "-lc", "rsi-submit"),
                    timeout_seconds=30,
                    environment=environment,
                )
                self._evidence.after = _observe_holder(runtime, self._client, request)
                self._stop_holder(runtime, request)

            _assert_exec_ok(submitted, "successful submission")
            output.append(submitted.output)
            self._evidence.outputs.extend(output)
            agent_result = AgentRunResult(exit_code=0, output="\n".join(output))
        except BaseException as error:
            primary_error = error
        self._finish_run_cleanup(runtime, request, primary_error)
        assert agent_result is not None
        return agent_result

    def _finish_run_cleanup(
        self,
        runtime: object,
        request: AgentRunRequest,
        primary_error: BaseException | None,
    ) -> None:
        cleanup_errors: list[str] = []
        if self._holder_started and not self._evidence.holder_stopped:
            try:
                self._stop_holder(runtime, request)
            except BaseException as error:
                cleanup_errors.append(f"CUDA holder cleanup: {error}")
        try:
            self.clear_transient_bindings()
        except BaseException as error:
            cleanup_errors.append(f"Agent binding cleanup: {error}")
        if cleanup_errors:
            primary = "none" if primary_error is None else str(primary_error)
            raise AssertionError(
                f"Hardware Agent primary failure: {primary}; cleanup failures: "
                + "; ".join(cleanup_errors)
            ) from primary_error
        if primary_error is not None:
            raise primary_error

    def _stop_holder(self, runtime: object, request: AgentRunRequest) -> None:
        stopped = runtime.exec(  # type: ignore[attr-defined]
            request.container,
            (
                "/bin/bash",
                "-lc",
                "set -eu\n"
                "touch /workspace/holder.stop\n"
                "for ignored in $(seq 1 200); do "
                "test ! -e /workspace/holder.ready && exit 0; sleep 0.05; done\n"
                "exit 1\n",
            ),
            timeout_seconds=15,
        )
        _assert_exec_ok(stopped, "CUDA holder shutdown")
        assert self._holder_thread is not None
        self._holder_thread.join(15)
        assert not self._holder_thread.is_alive(), "CUDA holder was not reaped"
        if self._holder_errors:
            raise self._holder_errors[0]
        assert len(self._holder_results) == 1
        _assert_exec_ok(self._holder_results[0], "CUDA holder lifetime")
        if self._evidence.before is not None:
            table = self._client.containers.get(request.container.container_id).top(
                ps_args="-eo pid,args"
            )
            pid_column = table["Titles"].index("PID")
            remaining = {int(row[pid_column]) for row in table["Processes"]}
            assert self._evidence.before.host_pid not in remaining, (
                "reaped CUDA holder host PID remains in the Work container"
            )
            assert all(
                pid != self._evidence.before.host_pid for _uuid, pid in _gpu_processes()
            ), "reaped CUDA holder PID remains in NVIDIA process inventory"
        self._evidence.holder_stopped = True

    def _start_holder(self, runtime: object, request: AgentRunRequest) -> None:
        prepared = runtime.exec(  # type: ignore[attr-defined]
            request.container,
            (
                "/bin/rm",
                "-f",
                "/workspace/holder.ready",
                "/workspace/holder.stop",
            ),
            timeout_seconds=5,
        )
        _assert_exec_ok(prepared, "CUDA holder marker preparation")

        def hold_and_reap() -> None:
            try:
                result = runtime.exec(  # type: ignore[attr-defined]
                    request.container,
                    (
                        "/usr/bin/python3",
                        "/usr/local/libexec/rsi-gpu-holder.py",
                        "/workspace/holder.ready",
                        "/workspace/holder.stop",
                    ),
                    timeout_seconds=60,
                )
                self._holder_results.append(result)
            except BaseException as error:
                self._holder_errors.append(error)

        self._holder_thread = threading.Thread(
            target=hold_and_reap,
            daemon=True,
            name="rsi-test-cuda-holder",
        )
        self._holder_started = True
        self._holder_thread.start()
        ready = runtime.exec(  # type: ignore[attr-defined]
            request.container,
            (
                "/bin/bash",
                "-lc",
                "set -eu\n"
                "for ignored in $(seq 1 200); do "
                "test -s /workspace/holder.ready && exit 0; sleep 0.05; done\n"
                "exit 1\n",
            ),
            timeout_seconds=15,
        )
        _assert_exec_ok(ready, "CUDA holder startup")


def _device_requests(attrs: dict[str, object]) -> tuple[str, ...]:
    host_config = attrs["HostConfig"]
    assert isinstance(host_config, dict)
    requests = host_config.get("DeviceRequests") or []
    assert isinstance(requests, list)
    if not requests:
        return ()
    assert len(requests) == 1
    request = requests[0]
    assert request["Driver"] == "nvidia"
    assert request["Capabilities"] == [["gpu"]]
    assert request["Count"] == 0
    return tuple(request["DeviceIDs"])


def _assert_no_runtime_leaks(
    client: docker.DockerClient, data: Path, run_id: str
) -> None:
    labels = {"label": f"rsi-harness.run-id={run_id}"}
    assert client.containers.list(all=True, filters=labels) == []
    assert client.networks.list(filters=labels) == []
    assert client.images.list(filters=labels) == []
    assert client.volumes.list(filters=labels) == []
    run_root = (data / run_id).resolve(strict=False)
    mountinfo = Path("/proc/self/mountinfo").read_text()
    assert str(run_root) not in mountinfo.replace("\\040", " ")
    fuse = subprocess.run(
        ["pgrep", "-af", "fuse-overlayfs"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert str(run_root) not in fuse.stdout


def _known_run_ids(data: Path, created: list[dict[str, object]]) -> set[str]:
    run_ids = {
        str(entry["run_id"])
        for entry in created
        if entry.get("role") in {"work", "judge"}
        and entry.get("run_id") != "image-preparation"
    }
    leases = data / "leases"
    if leases.is_dir():
        run_ids.update(path.stem for path in leases.glob("*.json"))
    return run_ids


def _run_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: Mode) -> None:
    client, inventory, firewall, snapshot, selected = _hardware_gate(tmp_path)

    class CountingInventory:
        def __init__(self) -> None:
            self.calls = 0

        def list_devices(self):
            self.calls += 1
            return inventory.list_devices()

    counted_inventory = CountingInventory()
    compile_calls = 0
    original_compile = HarborTaskCompiler.compile

    def recording_compile(compiler, task_dir, options):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(compiler, task_dir, options)

    monkeypatch.setattr(HarborTaskCompiler, "compile", recording_compile)
    selectors = tuple(device.uuid for device in selected)
    authorized = selectors if mode != "release-all" else selectors[:2]
    expected_work = selectors[:2]
    expected_judge = {
        "freeze-only": (),
        "disjoint": selectors[2:],
        "release-all": selectors[:2],
    }[mode]
    task = _task_copy(tmp_path, mode)
    fixture_before = _source_state(FIXTURE)
    task_before = _source_state(task)
    evidence = AgentEvidence()
    created: list[dict[str, object]] = []
    events: list[tuple[str, object]] = []
    original_create = DockerContainerRuntime.create

    def recording_create(self, spec, *, planned_name=None):
        ref = original_create(self, spec, planned_name=planned_name)
        container = client.containers.get(ref.container_id)
        container.reload()
        config_environment = dict(
            value.split("=", 1) for value in container.attrs["Config"]["Env"]
        )
        created.append(
            {
                "run_id": self.recovery_labels["rsi-harness.run-id"],
                "role": ref.role,
                "gpu_allocation": spec.gpu_allocation,
                "spec_gpu_uuids": spec.gpu_allocation.uuids,
                "nvidia_visible_devices": config_environment[
                    "NVIDIA_VISIBLE_DEVICES"
                ],
                "workdir": spec.workdir,
                "mounts": copy.deepcopy(container.attrs["Mounts"]),
                "device_requests": _device_requests(container.attrs),
            }
        )
        return ref

    monkeypatch.setattr(DockerContainerRuntime, "create", recording_create)
    data = tmp_path / "data"
    services = None
    result = None
    run_ids: set[str] = set()
    gpu_events: list[object] = []
    try:
        from rsi_harness.runtime.production import ProductionRuntimeServices

        services = ProductionRuntimeServices(
            data_root=data,
            logs_root=tmp_path / "logs",
            docker_client=client,
            inventory=counted_inventory,
            rsi_loop_config=RSILoopConfig(),
            snapshot_backend=snapshot,
            firewall_backend=firewall,
            agent_adapter_factory=lambda config, runtime: HardwareAgent(
                config,
                runtime=runtime,
                client=client,
                mode=mode,
                evidence=evidence,
            ),
            event_callback=lambda name, value: events.append((name, value)),
        )
        result = services.run(
            RunRequest(
                task_dir=task.resolve(),
                agent_name="codex",
                gpu_selectors=authorized,
                options=CompileOptions(
                    agent_name="codex",
                    primary_reward="reward",
                    max_submissions=1,
                ),
            )
        )
        gpu_events = [value for name, value in events if name == "gpu_plan"]
        run_ids.add(result.run_id)
        assert result.status is RunStatus.COMPLETED
        assert result.total_rounds == 1
        assert [report.round_id for report in result.reports] == ["agent-1"]
        assert result.best_round == "agent-1"
        assert result.best_score == 1
        assert compile_calls == 1
        assert counted_inventory.calls == 1

        assert evidence.work_uuids == expected_work
        assert evidence.before is not None and evidence.after is not None
        assert evidence.before.host_pid == evidence.after.host_pid
        assert evidence.before.marker_pid == evidence.after.marker_pid
        for observation in (evidence.before, evidence.after):
            assert (expected_work[0], observation.host_pid) in observation.gpu_processes
        assert evidence.holder_stopped is True

        run_created = [entry for entry in created if entry["run_id"] == result.run_id]
        assert [entry["role"] for entry in run_created].count("work") == 1
        assert [entry["role"] for entry in run_created].count("judge") == 1
        work_entry = next(entry for entry in run_created if entry["role"] == "work")
        judge_entry = next(entry for entry in run_created if entry["role"] == "judge")
        assert work_entry["spec_gpu_uuids"] == expected_work
        assert work_entry["device_requests"] == expected_work
        assert work_entry["nvidia_visible_devices"] == ",".join(expected_work)
        assert judge_entry["spec_gpu_uuids"] == expected_judge
        assert judge_entry["device_requests"] == expected_judge
        assert judge_entry["nvidia_visible_devices"] == (
            ",".join(expected_judge) if expected_judge else "void"
        )
        assert work_entry["workdir"] == PurePosixPath("/workspace")
        assert judge_entry["workdir"] == PurePosixPath("/workspace")

        assert len(gpu_events) == 1
        gpu_plan = gpu_events[0]
        assert gpu_plan.authorized_pool.uuids == authorized
        assert gpu_plan.work.uuids == expected_work
        assert gpu_plan.judge.uuids == expected_judge
        assert work_entry["gpu_allocation"] is gpu_plan.work
        assert judge_entry["gpu_allocation"] is gpu_plan.judge
        durable = LeaseStore(data / "leases").read(result.run_id)
        assert durable is not None
        assert durable.gpu_plan == gpu_plan
        run_plan = evidence.run_plan
        assert run_plan is not None
        assert run_plan.paths.root.name == result.run_id

        volume_names: set[str] = set()
        for entry in (work_entry, judge_entry):
            mounts = entry["mounts"]
            assert isinstance(mounts, list)
            workdir_mounts = [
                mount for mount in mounts if mount.get("Destination") == "/workspace"
            ]
            assert len(workdir_mounts) == 1
            mount = workdir_mounts[0]
            assert mount["Type"] == "volume"
            assert mount["RW"] is (entry["role"] == "work")
            volume_names.add(mount["Name"])
        assert len(volume_names) == 1

        verifier = _run_verifier_dir(run_plan, "agent-1")
        work_visible = tuple(
            line.strip()
            for line in verifier.joinpath("work-gpus.txt").read_text().splitlines()
            if line.strip()
        )
        judge_visible = tuple(
            line.strip()
            for line in verifier.joinpath("judge-gpus.txt").read_text().splitlines()
            if line.strip()
        )
        judge_expected = tuple(
            line.strip()
            for line in verifier.joinpath("expected-judge-gpus.txt")
            .read_text()
            .splitlines()
            if line.strip()
        )
        assert work_visible == expected_work
        assert judge_expected == expected_judge
        assert judge_visible == expected_judge
        nvidia_status = int(
            verifier.joinpath("judge-nvidia-status.txt").read_text().strip()
        )
        judge_devices = tuple(
            line.strip()
            for line in verifier.joinpath("judge-gpu-devices.txt")
            .read_text()
            .splitlines()
            if line.strip()
        )
        if expected_judge:
            assert nvidia_status == 0
        else:
            assert judge_devices == ()
        assert set(work_visible).isdisjoint(judge_visible) is (
            mode in {"freeze-only", "disjoint"}
        )

        if mode == "release-all":
            assert evidence.first_rejection is not None
            assert evidence.first_http_status == 409
            assert evidence.first_rejection.exit_code not in (None, 0)
            assert (
                "release all Work GPU processes before retrying"
                in evidence.first_rejection.output
            )
            assert evidence.rejection_had_no_verifier_dir is True
            assert evidence.rejection_had_no_formal_report is True
            assert evidence.rejection_consumed_no_history is True
            artifact_root = _run_artifact_root(run_plan)
            assert sorted(
                path.name for path in artifact_root.joinpath("submissions").iterdir()
            ) == ["agent-1"]
            final = json.loads(artifact_root.joinpath("final_result.json").read_text())
            assert final["total_rounds"] == 1

        assert _source_state(FIXTURE) == fixture_before
        assert _source_state(task) == task_before
        assert not list(tmp_path.rglob("*.tar*"))
    finally:
        run_ids.update(_known_run_ids(data, created))
        if services is not None:
            for run_id in sorted(run_ids):
                services.cleanup(run_id, delete_workspace=True)
                _assert_no_runtime_leaks(client, data, run_id)
                recovered = LeaseStore(data / "leases").read(run_id)
                assert recovered is not None
                if (
                    result is not None
                    and run_id == result.run_id
                    and len(gpu_events) == 1
                ):
                    assert recovered.gpu_plan == gpu_events[0]
        assert _source_state(FIXTURE) == fixture_before
        assert _source_state(task) == task_before


@pytest.mark.gpu
def test_freeze_only_retains_work_gpu_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_mode(tmp_path, monkeypatch, "freeze-only")


@pytest.mark.gpu
def test_freeze_only_is_gpu_void_with_default_nvidia_runtime(
    tmp_path: Path,
) -> None:
    try:
        client = docker.from_env()
        client.ping()
        info = client.info()
    except Exception as error:
        pytest.skip(f"Docker default-runtime gate unavailable: {error}")
    default = str(info.get("DefaultRuntime") or "")
    runtimes = info.get("Runtimes") or {}
    runtime = runtimes.get(default) if isinstance(runtimes, dict) else None
    signature = f"{default} {runtime}".casefold()
    if "nvidia" not in signature:
        pytest.skip(
            "Docker default runtime is not NVIDIA-backed: "
            f"{default or 'unknown'}"
        )
    try:
        client.images.get("ubuntu:24.04")
    except Exception as error:
        pytest.skip(f"local Ubuntu image gate unavailable: {error}")

    source = None
    snapshot = None
    runtime = None
    judge = None
    try:
        source = client.containers.create(
            "ubuntu:24.04",
            ["/bin/true"],
            environment={"NVIDIA_VISIBLE_DEVICES": "all"},
            network_mode="none",
            labels={"rsi-harness.test": "default-nvidia-runtime-void"},
        )
        snapshot = source.commit(
            repository="rsi-harness-test",
            tag=f"default-nvidia-runtime-void-{tmp_path.name}",
        )
        assert "NVIDIA_VISIBLE_DEVICES=all" in snapshot.attrs["Config"]["Env"]

        runtime = DockerContainerRuntime(
            client,
            run_id=f"default-runtime-void-{tmp_path.name}",
            task_id="default-runtime-void",
            role="judge",
            task_source_dir=tmp_path,
            allowed_mount_roots=(tmp_path,),
        )
        judge = runtime.create(
            ContainerSpec(
                image=snapshot.id,
                command=(
                    "/bin/sh",
                    "-c",
                    "set -eu; "
                    "test \"${NVIDIA_VISIBLE_DEVICES-}\" = void; "
                    "test -z \"$(find /dev -maxdepth 1 -name 'nvidia*' "
                    "-print -quit)\"; "
                    "test ! -e /usr/bin/nvidia-smi",
                ),
            )
        )
        container = client.containers.get(judge.container_id)
        container.reload()
        assert "NVIDIA_VISIBLE_DEVICES=void" in container.attrs["Config"]["Env"]
        assert _device_requests(container.attrs) == ()
        container.start()
        outcome = container.wait(timeout=15)
        assert outcome["StatusCode"] == 0, container.logs().decode(errors="replace")
    finally:
        if judge is not None and runtime is not None:
            runtime.remove(judge)
        if source is not None:
            source.remove(force=True, v=True)
        if snapshot is not None:
            client.images.remove(snapshot.id, force=True)


@pytest.mark.gpu
def test_disjoint_judge_gpus_do_not_overlap_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_mode(tmp_path, monkeypatch, "disjoint")


@pytest.mark.gpu
def test_release_all_rejection_refunds_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_mode(tmp_path, monkeypatch, "release-all")


def test_work_enumeration_writes_and_returns_exact_order(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    binary = tmp_path / "bin"
    workspace.mkdir()
    binary.mkdir()
    nvidia_smi = binary / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nprintf 'GPU-b\\nGPU-a\\n'\n")
    nvidia_smi.chmod(0o755)

    class Runtime:
        def exec(self, _container, command, **_kwargs):
            local = (*command[:2], command[2].replace("/workspace", str(workspace)))
            result = subprocess.run(
                local,
                capture_output=True,
                check=False,
                text=True,
                env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
            )
            return AgentRunResult(exit_code=result.returncode, output=result.stdout)

    runtime = Runtime()
    request = SimpleNamespace(
        container=ContainerRef(container_id="work-controlled", role="work")
    )

    visible = _enumerate_work_gpus(runtime, request)

    assert visible == ("GPU-b", "GPU-a")
    assert workspace.joinpath("work-gpus.txt").read_text() == "GPU-b\nGPU-a\n"


def test_holder_pid_mapping_uses_ready_marker_namespace_pid() -> None:
    table = {
        "Titles": ["PID", "COMMAND"],
        "Processes": [
            ["100", "/bin/bash -lc python3 /usr/local/libexec/rsi-gpu-holder.py"],
            ["101", "python3 /usr/local/libexec/rsi-gpu-holder.py"],
        ],
    }
    namespace_pids = {100: (100, 16), 101: (101, 17)}
    command_lines = {
        100: ("/bin/bash", "-lc", "python3 /usr/local/libexec/rsi-gpu-holder.py"),
        101: (
            "python3",
            "/usr/local/libexec/rsi-gpu-holder.py",
            "/workspace/holder.ready",
            "/workspace/holder.stop",
        ),
    }

    host_pid = _resolve_holder_host_pid(
        table,
        marker_pid=17,
        namespace_pids=lambda pid: namespace_pids[pid],
        command_line=lambda pid: command_lines[pid],
    )

    assert host_pid == 101


@pytest.mark.parametrize(
    "result",
    (
        subprocess.CompletedProcess([], 1, "", "NVML failed"),
        subprocess.CompletedProcess([], 0, "not-a-row\n", ""),
    ),
)
def test_post_compute_gpu_inventory_errors_fail_instead_of_skip(
    monkeypatch: pytest.MonkeyPatch, result: subprocess.CompletedProcess[str]
) -> None:
    monkeypatch.setattr(f"{__name__}._run_command", lambda _command: result)

    with pytest.raises(AssertionError, match="NVIDIA process inventory"):
        _gpu_processes(capability_gate=False)


def test_capability_gpu_inventory_error_skips_before_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        f"{__name__}._run_command",
        lambda _command: subprocess.CompletedProcess([], 1, "", "NVML failed"),
    )

    with pytest.raises(pytest.skip.Exception, match="NVIDIA process inventory"):
        _gpu_processes(capability_gate=True)


def test_freeze_verifier_observes_unexpected_actual_gpu(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    logs = tmp_path / "logs"
    binary = tmp_path / "bin"
    workspace.mkdir()
    logs.mkdir()
    binary.mkdir()
    workspace.joinpath("work-gpus.txt").write_text("GPU-work-a\nGPU-work-b\n")
    invoked = tmp_path / "nvidia-smi-invoked"
    nvidia_smi = binary / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n: > \"$RSI_NVIDIA_SMI_INVOKED\"\nprintf 'GPU-unexpected\\n'\n"
    )
    nvidia_smi.chmod(0o755)
    verifier = tmp_path / "test.sh"
    verifier.write_text(
        FIXTURE.joinpath("tests/test.sh")
        .read_text()
        .replace("/workspace", str(workspace))
        .replace("/logs/verifier", str(logs))
    )

    result = subprocess.run(
        ("/bin/bash", verifier),
        capture_output=True,
        check=False,
        text=True,
        env={
            **os.environ,
            "PATH": f"{binary}:{os.environ['PATH']}",
            "RSI_HARNESS_EXPECTED_GPU_UUIDS": "",
            "RSI_NVIDIA_SMI_INVOKED": str(invoked),
        },
    )

    assert result.returncode == 0
    assert invoked.is_file()
    assert logs.joinpath("judge-gpus.txt").read_text() == "GPU-unexpected\n"
    assert json.loads(logs.joinpath("reward.json").read_text()) == {"reward": 0}


def test_agent_cleanup_failure_preserves_primary_and_cleanup_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = object.__new__(HardwareAgent)
    agent._evidence = AgentEvidence()
    agent._holder_started = True

    def fail_stop(_runtime, _request) -> None:
        raise RuntimeError("holder reap failed")

    monkeypatch.setattr(agent, "_stop_holder", fail_stop)
    monkeypatch.setattr(agent, "clear_transient_bindings", lambda: None)

    with pytest.raises(AssertionError, match="submission failed.*holder reap failed"):
        agent._finish_run_cleanup(
            object(),
            SimpleNamespace(),
            ValueError("submission failed"),
        )


def test_release_all_status_parser_requires_exact_409() -> None:
    assert _submission_http_status(
        "body\nrsi-submit: judge returned HTTP 409\n"
    ) == 409
    with pytest.raises(AssertionError, match="HTTP 409"):
        _submission_http_status("body\nrsi-submit: judge returned HTTP 500\n")


def test_run_artifact_paths_are_scoped_by_captured_plan_run_id(
    tmp_path: Path,
) -> None:
    plan = _controlled_run_plan(tmp_path)

    artifact_root = _run_artifact_root(plan)

    assert artifact_root == (
        tmp_path / "logs" / "runs" / "run-authority" / "minimal-gpu"
    )
    assert _run_verifier_dir(plan, "agent-1") == artifact_root / "verifier" / "agent-1"


def test_hardware_agent_captures_exact_prepare_run_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _controlled_run_plan(tmp_path)
    prepared = PreparedAgent(
        agent_name="controlled",
        command=("/bin/true",),
        prompt_path=tmp_path / "prompt.md",
    )
    evidence = AgentEvidence()
    agent = HardwareAgent(
        RSILoopConfig(),
        runtime=object(),
        client=SimpleNamespace(),  # type: ignore[arg-type]
        mode="freeze-only",
        evidence=evidence,
    )
    monkeypatch.setattr(
        RSILoopAgentAdapter,
        "prepare",
        lambda _agent, _request: prepared,
    )

    actual = agent.prepare(
        AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path / "prompt.md")
    )

    assert actual is prepared
    assert agent._run_plan is plan
    assert evidence.run_plan is plan


def test_known_run_ids_excludes_image_preparation_and_non_run_helpers(
    tmp_path: Path,
) -> None:
    leases = tmp_path / "leases"
    leases.mkdir()
    leases.joinpath("durable-run.json").write_text("{}")
    created = [
        {"run_id": "image-preparation", "role": "work"},
        {"run_id": "created-run", "role": "judge"},
        {"run_id": "helper-authority", "role": "helper"},
    ]

    assert _known_run_ids(tmp_path, created) == {"created-run", "durable-run"}
