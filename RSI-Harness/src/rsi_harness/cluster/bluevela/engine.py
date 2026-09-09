"""Native RSI-Harness Engine execution inside one Blue Vela allocation."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shlex
import socket
import sys
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator

from rsi_harness.cluster.bluevela.adapter import ClusterResources
from rsi_harness.cluster.bluevela.allocation import (
    AllocatedPools,
    parse_lsb_mcpu_hosts,
    probe_and_partition,
)
from rsi_harness.cluster.bluevela.resources import MultiNodeResources
from rsi_harness.cluster.config import ClusterProfile
from rsi_harness.errors import SetupError
from rsi_harness.models import (
    AgentAuthSource,
    CompileOptions,
    GPUAllocation,
    GPUDevice,
    PersistedModel,
    RunGPUPlan,
    RunPlan,
)


class EnginePayload(PersistedModel):
    """Frozen inputs consumed by the compute-node RSI-Harness Engine."""

    run_id: str
    run_plan: RunPlan
    sif_path: Path
    sif_sha256_path: Path
    source_root: Path
    resources: ClusterResources | None = None
    multi_node: MultiNodeResources | None = None
    profile: ClusterProfile
    options: CompileOptions
    agent_auth: AgentAuthSource | None = None
    agent_version: str | None = None
    agent_binary: Path | None = None
    agent_launcher: str = Field(
        default="codex", pattern=r"[A-Za-z0-9][A-Za-z0-9._+-]*"
    )
    agent_companions: tuple[Path, ...] = ()

    @model_validator(mode="after")
    def _exactly_one_resource_branch(self) -> Self:
        if (self.resources is None) == (self.multi_node is None):
            raise ValueError(
                "Engine payload requires exactly one Blue Vela resource branch"
            )
        return self

    @field_validator(
        "sif_path", "sif_sha256_path", "source_root", "agent_binary"
    )
    @classmethod
    def _absolute(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        if not value.is_absolute():
            raise ValueError("cluster Engine payload paths must be absolute")
        return value

    @field_validator("agent_companions")
    @classmethod
    def _absolute_companions(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        if any(not path.is_absolute() for path in value):
            raise ValueError("cluster Agent companion paths must be absolute")
        return value


def load_engine_payload(path: Path) -> EnginePayload:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SetupError(f"invalid cluster Engine payload {path}: {error}") from error
    return EnginePayload.model_validate(value)


def bind_lsf_devices(
    plan: RunPlan,
    allocated_devices: tuple[str, ...],
) -> RunPlan:
    """Replace planning placeholders with the exact ordered LSF allocation."""
    expected = len(plan.gpu_plan.authorized_pool.devices)
    if (
        len(allocated_devices) != expected
        or any(not device for device in allocated_devices)
        or len(set(allocated_devices)) != len(allocated_devices)
    ):
        raise SetupError(
            "LSF CUDA_VISIBLE_DEVICES does not match the Engine plan: "
            f"expected {expected}, got {allocated_devices!r}"
        )
    devices = tuple(
        GPUDevice(index=index, uuid=value, name="LSF allocated GPU")
        for index, value in enumerate(allocated_devices)
    )
    mapped_devices = {
        placeholder.uuid: device
        for placeholder, device in zip(
            plan.gpu_plan.authorized_pool.devices, devices, strict=True
        )
    }

    def remap(allocation: GPUAllocation) -> GPUAllocation:
        return GPUAllocation(
            devices=tuple(mapped_devices[device.uuid] for device in allocation.devices)
        )

    gpu_plan = RunGPUPlan(
        authorized_pool=GPUAllocation(devices=devices),
        work=remap(plan.gpu_plan.work),
        judge=remap(plan.gpu_plan.judge),
        judge_mode=plan.gpu_plan.judge_mode,
    )
    return plan.model_copy(update={"gpu_plan": gpu_plan})


def bind_multinode_devices(
    plan: RunPlan,
    pools: AllocatedPools,
) -> RunPlan:
    """Bind phase placeholders to exact host-qualified allocated devices."""
    work_values = tuple(
        (node.host, device)
        for node in pools.work
        for device in node.cuda_devices
    )
    judge_values = tuple(
        (node.host, device)
        for node in pools.verifier
        for device in node.cuda_devices
    )
    if len(work_values) != len(plan.gpu_plan.work.devices) or len(
        judge_values
    ) != len(plan.gpu_plan.judge.devices):
        raise SetupError(
            "frozen Blue Vela pools do not match planned Work/Judge devices"
        )
    qualified = tuple(f"{host}:{device}" for host, device in work_values + judge_values)
    if len(set(qualified)) != len(qualified):
        raise SetupError("frozen Blue Vela pools contain duplicate devices")
    devices = tuple(
        GPUDevice(index=index, uuid=value, name="allocated Blue Vela GPU")
        for index, value in enumerate(qualified)
    )
    work_count = len(work_values)
    gpu_plan = RunGPUPlan(
        authorized_pool=GPUAllocation(devices=devices),
        work=GPUAllocation(devices=devices[:work_count]),
        judge=GPUAllocation(devices=devices[work_count:]),
        judge_mode=plan.gpu_plan.judge_mode,
    )
    return plan.model_copy(update={"gpu_plan": gpu_plan})


def _quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def _atomic_model(path: Path, value: PersistedModel) -> None:
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(value.model_dump_json(indent=2))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def render_engine_driver(
    payload: EnginePayload,
    profile: ClusterProfile,
    output_path: Path,
) -> Path:
    """Render a GPU driver that starts the native Engine, never Harbor CLI."""
    output = Path(output_path).resolve()
    payload_path = output.with_suffix(".json")
    payload_path.write_text(payload.model_dump_json(indent=2))
    leaf = (
        payload.run_plan.paths.logs
        / "runs"
        / payload.run_id
        / payload.run_plan.task.task_id
    )
    binds = [
        item
        for path in profile.apptainer.extra_binds
        for item in ("--bind", f"{path}:{path}")
    ]
    bind_args = " ".join(_quote(item) for item in binds)
    if payload.resources is not None:
        script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077

test -n "${{CUDA_VISIBLE_DEVICES:-}}"
export APPTAINER_BIND={_quote(profile.apptainer.dns_bind)}
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export HF_HOME={_quote(profile.storage.hf_home)}
export HF_DATASETS_CACHE={_quote(profile.storage.hf_datasets_cache)}
export PYTHONPATH={_quote(payload.source_root / 'src')}
runtime_tmp_root={_quote(profile.apptainer.temp_root)}
available_tmp_kb="$(df -Pk -- "$runtime_tmp_root" | awk 'NR == 2 {{print $4}}')"
required_tmp_kb=$(({payload.resources.local_tmp_mb} * 1024))
if [[ ! "$available_tmp_kb" =~ ^[0-9]+$ ]] || \
   (( available_tmp_kb < required_tmp_kb )); then
  echo "insufficient node-local runtime scratch:" \
    "need {payload.resources.local_tmp_mb} MiB under $runtime_tmp_root," \
    "have ${{available_tmp_kb:-unknown}} KiB" >&2
  exit 1
fi
export RSI_HARNESS_NODE_TMP
RSI_HARNESS_NODE_TMP="$(mktemp -d "$runtime_tmp_root/rsi-harness-engine.XXXXXX")"
cleanup() {{
  rm -rf -- "$RSI_HARNESS_NODE_TMP"
}}
trap cleanup EXIT

(cd {_quote(payload.sif_path.parent)} && \
  sha256sum -c {_quote(payload.sif_sha256_path.name)})
{_quote(profile.apptainer.binary)} exec --nv --containall --writable-tmpfs \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" {bind_args} \
  {_quote(payload.sif_path)} nvidia-smi

{_quote(sys.executable)} -m rsi_harness.cluster.bluevela.engine \
  --payload {_quote(payload_path)}
test -s {_quote(leaf / 'final_result.json')}
"""
    else:
        assert payload.multi_node is not None
        script = f"""#!/usr/bin/env bash
set -euo pipefail
umask 077

test -n "${{LSB_MCPU_HOSTS:-}}"
export APPTAINER_BIND={_quote(profile.apptainer.dns_bind)}
export HF_HOME={_quote(profile.storage.hf_home)}
export HF_DATASETS_CACHE={_quote(profile.storage.hf_datasets_cache)}
export PYTHONPATH={_quote(payload.source_root / 'src')}
runtime_tmp_root={_quote(profile.apptainer.temp_root)}
available_tmp_kb="$(df -Pk -- "$runtime_tmp_root" | awk 'NR == 2 {{print $4}}')"
required_tmp_kb=$(({payload.multi_node.node_tmp_mb} * 1024))
if [[ ! "$available_tmp_kb" =~ ^[0-9]+$ ]] || \
   (( available_tmp_kb < required_tmp_kb )); then
  echo "insufficient controller node-local runtime scratch:" \
    "need {payload.multi_node.node_tmp_mb} MiB under $runtime_tmp_root," \
    "have ${{available_tmp_kb:-unknown}} KiB" >&2
  exit 1
fi
export RSI_HARNESS_NODE_TMP
RSI_HARNESS_NODE_TMP="$(mktemp -d "$runtime_tmp_root/rsi-harness-engine.XXXXXX")"
cleanup() {{
  rm -rf -- "$RSI_HARNESS_NODE_TMP"
}}
trap cleanup EXIT

(cd {_quote(payload.sif_path.parent)} && \
  sha256sum -c {_quote(payload.sif_sha256_path.name)})
{_quote(sys.executable)} -m rsi_harness.cluster.bluevela.engine \
  --payload {_quote(payload_path)}
test -s {_quote(leaf / 'final_result.json')}
"""
    output.write_text(script)
    output.chmod(0o700)
    return payload_path


def run_engine_payload(payload: EnginePayload) -> None:
    """Run the native RSI-Harness Engine inside the current LSF allocation."""
    allocated_pools: AllocatedPools | None = None
    if payload.resources is not None:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        devices = tuple(item.strip() for item in raw.split(",") if item.strip())
        plan = bind_lsf_devices(payload.run_plan, devices)
    else:
        resources = payload.multi_node
        assert resources is not None
        inventory = parse_lsb_mcpu_hosts(
            os.environ.get("LSB_MCPU_HOSTS", ""),
            expected_hosts=resources.total_nodes,
            expected_slots=resources.cpu_slots_per_node,
        )
        try:
            expected_digest = payload.sif_sha256_path.read_text().split()[0]
        except (OSError, IndexError) as error:
            raise SetupError("cannot read frozen SIF digest") from error
        allocated_pools = probe_and_partition(
            run_id=payload.run_id,
            resources=resources,
            inventory=inventory,
            sif_path=payload.sif_path,
            run_dir=payload.run_plan.paths.root,
            temp_root=payload.profile.apptainer.temp_root,
            expected_sif_sha256=expected_digest,
            remote_binary=payload.profile.scheduler.remote_binary,
            remote_host_flag=payload.profile.scheduler.remote_host_flag,
        )
        controller = (
            allocated_pools.work[0]
            if allocated_pools.work
            else allocated_pools.verifier[0]
        )
        if socket.gethostname().split(".", 1)[0] != controller.host:
            raise SetupError(
                "multi-node Engine controller is not the first frozen phase host"
            )
        control_path = payload.run_plan.paths.root / "control/ALLOCATED_POOLS.json"
        _atomic_model(control_path, allocated_pools)
        plan = bind_multinode_devices(payload.run_plan, allocated_pools)
    from rsi_harness.cluster.bluevela.runtime import run_native_engine

    run_native_engine(payload, plan, allocated_pools=allocated_pools)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    args = parser.parse_args()
    run_engine_payload(load_engine_payload(args.payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
