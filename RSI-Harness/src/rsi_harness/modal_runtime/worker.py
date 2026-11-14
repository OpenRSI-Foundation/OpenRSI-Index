"""Trusted subprocess entrypoint inside one Modal Function allocation."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path, PurePosixPath

from rsi_harness.errors import SetupError
from rsi_harness.models import (
    CompileOptions,
    GPUAllocation,
    ImagePlan,
    JudgeGPUMode,
    RootfsSnapshotMode,
    RunGPUPlan,
    RunPaths,
    RunPlan,
)
from rsi_harness.runtime.coordinator import RunCoordinator
from rsi_harness.runtime.gpu import NvidiaSmiInventory
from rsi_harness.runtime.recovery import AttemptGrants, LeaseStore
from rsi_harness.runtime.resume import plan_resume
from rsi_harness.runtime.run_request import write_run_request
from rsi_harness.task.compiler import HarborTaskCompiler

from .config import ModalRunConfig, validate_task
from .engine import ModalEngine


def execute(payload):
    config = ModalRunConfig.model_validate(payload["config"])
    options = CompileOptions.model_validate(payload["options"])
    run_id = payload["run_id"]
    # Modal exposes a volume mount through a symlink. Native artifact writers
    # deliberately reject symlink ancestors, so use its canonical mount path.
    root = Path("/rsi-state").resolve(strict=True)
    run_root = root / "data" / run_id
    compiler = HarborTaskCompiler()
    task = compiler.compile(Path("/opt/rsi-task"), options)
    task = task.model_copy(
        update={
            "agent": task.agent.model_copy(
                update={
                    "model": payload.get("model"),
                    "reasoning_effort": payload.get("reasoning_effort"),
                }
            )
        }
    )
    validate_task(task, config)
    if task.agent.name == "codex" and not os.environ.get("RSI_AGENT_API_KEY"):
        raise SetupError("named Secret must contain RSI_AGENT_API_KEY")
    if task.source_digest != payload["task_digest"]:
        raise SetupError("staged task differs from the operator-compiled digest")
    for asset in config.assets:
        if not asset.source.exists():
            raise SetupError(f"asset missing: {asset.source}")
        if not any(
            asset.source.resolve().is_relative_to(Path(mount).resolve())
            for mount in config.asset_volumes
        ):
            raise SetupError("asset resolves outside its approved volume")
    for required in task.assets:
        if not any(
            a.phase == required.phase
            and (a.target == required.path or a.target in required.path.parents)
            for a in config.assets
        ):
            raise SetupError(
                f"task asset has no explicit phase binding: {required.path}"
            )
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "modal-state.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "deadline": time.time() + task.agent.timeout_seconds,
            "reserved_compute_seconds": 0,
            "segments": 0,
            "task_digest": task.source_digest,
            "image_id": payload["image_id"],
        }
    )
    if state["segments"] + 1 != payload["expected_segment"]:
        raise SetupError("duplicate or out-of-order segment invocation refused")
    if (state["task_digest"], state["image_id"]) != (
        task.source_digest,
        payload["image_id"],
    ):
        raise SetupError("task or image changed since the first segment")
    if state["segments"] >= config.max_segments:
        raise SetupError("segment cap reached")
    remaining = state["deadline"] - time.time()
    if remaining <= 0:
        raise SetupError("original run deadline has expired")
    reserve = min(config.segment_seconds, remaining) + config.cleanup_seconds
    if state["reserved_compute_seconds"] + reserve > config.max_compute_seconds:
        raise SetupError("compute ceiling would be exceeded by the next segment")
    state["reserved_compute_seconds"] += reserve
    state["segments"] += 1
    state_path.write_text(json.dumps(state, indent=2))
    devices = NvidiaSmiInventory().list_devices() if config.gpu else ()
    if len(devices) != (1 if config.gpu else 0):
        raise SetupError("allocation does not have the configured physical GPU count")
    pool = GPUAllocation(devices=devices)
    empty = GPUAllocation()
    gpu = RunGPUPlan(
        authorized_pool=pool,
        work=empty,
        judge=pool if task.verifier.gpu_count else empty,
        judge_mode=JudgeGPUMode.DISJOINT
        if task.verifier.gpu_count
        else JudgeGPUMode.FREEZE_ONLY,
    )
    paths = RunPaths(
        root=run_root, workspace=run_root / "phases" / "workspace", logs=root / "logs"
    )
    image = payload["image_id"]
    plan = RunPlan(
        schema_version=2,
        task=task,
        workdir=PurePosixPath("/workspace"),
        rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        images=ImagePlan(
            base_ref=image,
            work_ref=image,
            judge_ref=image,
            workdir=task.workdir,
            rootfs_snapshot_mode=RootfsSnapshotMode.SPLIT_WORKDIR,
        ),
        gpu_plan=gpu,
        paths=paths,
        snapshot_kind="modal-namespace-copy",
    )
    store = LeaseStore(root / "leases")
    resume = None
    if store.path_for(run_id).exists():
        resume = plan_resume(
            run_id,
            store=store,
            data_root=root / "data",
            logs_root=root / "logs",
            grants=AttemptGrants(),
            compile_task=lambda _: task,
            salvage=None,
            retained_images=False,
        )
        if resume.salvage_pending:
            raise SetupError(
                "previous segment needs recovery; automatic replay refused"
            )
        resume = replace(
            resume, agent_seconds_left=min(resume.agent_seconds_left, remaining)
        )
    engine = ModalEngine(
        run_root / "phases", plan, config, run_id, options, resume=resume
    )
    if resume is None:
        write_run_request(run_root, run_id=run_id, request=engine.request())
    coordinator = RunCoordinator(
        backend=engine.backend(),
        lease_store=store,
        clock=engine.clock,
        run_id_factory=lambda: run_id,
        resume=resume,
    )
    boundary = min(config.segment_seconds, remaining)
    engine.deadline = time.time() + boundary
    interrupted = threading.Event()

    def cut_segment():
        interrupted.set()
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(boundary, cut_segment)
    timer.daemon = True
    if remaining > config.segment_seconds:
        timer.start()
    try:
        result = coordinator.run(engine.request())
    finally:
        timer.cancel()
        engine.stop_agent(engine.runtime.work_ref)
    result_payload = result.model_dump(mode="json")
    diagnostic = ""
    if result.agent_exit_code not in (None, 0) and engine.artifacts is not None:
        with (engine.artifacts.root / "agent_output.txt").open("rb") as stream:
            stream.seek(max(0, os.fstat(stream.fileno()).st_size - 4096))
            diagnostic = stream.read().decode(errors="replace")
    receipt = {
        "run_id": run_id,
        "segment": state["segments"],
        "result": result_payload,
        "deadline": state["deadline"],
        "reserved_compute_seconds": state["reserved_compute_seconds"],
        "segment_boundary": interrupted.is_set(),
        "continue": interrupted.is_set() and state["deadline"] - time.time() > 5,
        "gpu_uuids": list(pool.uuids),
        "image_id": image,
        "state_root": str(root),
        "error": store.read(run_id).error,
        "agent_diagnostic": diagnostic,
    }
    (run_root / "segment-result.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    execute(json.loads(Path(sys.argv[1]).read_text()))
