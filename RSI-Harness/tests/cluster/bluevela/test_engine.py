from __future__ import annotations

import json
from pathlib import Path

import pytest

from rsi_harness.cluster.bluevela.adapter import ClusterResources
from rsi_harness.cluster.bluevela.allocation import AllocatedNode, AllocatedPools
from rsi_harness.cluster.bluevela.engine import (
    EnginePayload,
    bind_lsf_devices,
    bind_multinode_devices,
    load_engine_payload,
    render_engine_driver,
    run_engine_payload,
)
from rsi_harness.cluster.bluevela.resources import (
    MultiNodeResources,
    PhaseResources,
)
from rsi_harness.cluster.config import load_cluster_profile
from rsi_harness.models import AgentAuthSource, CompileOptions
from tests.factories import make_run_plan


def _payload(tmp_path: Path) -> EnginePayload:
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    plan = make_run_plan(tmp_path)
    devices = plan.gpu_plan.authorized_pool.devices
    plan = plan.model_copy(
        update={
            "gpu_plan": plan.gpu_plan.model_copy(
                update={
                    "authorized_pool": plan.gpu_plan.authorized_pool.model_copy(
                        update={"devices": devices * 2}
                    ),
                    "work": plan.gpu_plan.work.model_copy(update={"devices": devices}),
                    "judge": plan.gpu_plan.judge.model_copy(
                        update={"devices": devices[1:] or devices}
                    ),
                }
            )
        }
    )
    # Use four distinct placeholder devices, matching a 2+2 task allocation.
    from rsi_harness.models import GPUAllocation, GPUDevice, RunGPUPlan

    all_devices = tuple(
        GPUDevice(index=index, uuid=f"LSF-{index}", name="allocated")
        for index in range(4)
    )
    plan = plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=all_devices),
                work=GPUAllocation(devices=all_devices[:2]),
                judge=GPUAllocation(devices=all_devices[2:]),
                judge_mode="disjoint",
            )
        }
    )
    return EnginePayload(
        run_id="native-engine-run",
        run_plan=plan,
        sif_path=(tmp_path / "task.sif").resolve(),
        sif_sha256_path=(tmp_path / "task.sif.sha256").resolve(),
        source_root=Path(__file__).resolve().parents[3],
        resources=ClusterResources(
            work_gpus=2,
            verifier_gpus=2,
            total_gpus=4,
            cpu_slots=8,
            memory_mb=65536,
            local_tmp_mb=15360,
            build_walltime="02:00",
            run_walltime="02:15",
        ),
        profile=profile,
        options=CompileOptions(max_submissions=4),
        agent_auth=AgentAuthSource.LOCAL,
        agent_version="0.149.0",
        agent_binary=Path("/bin/true"),
        agent_launcher="codex",
        agent_companions=(Path("/bin/false"),),
    )


def test_lsf_devices_replace_placeholders_and_remain_disjoint(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    plan = bind_lsf_devices(payload.run_plan, ("0", "2", "5", "7"))

    assert plan.gpu_plan.authorized_pool.uuids == ("0", "2", "5", "7")
    assert plan.gpu_plan.work.uuids == ("0", "2")
    assert plan.gpu_plan.judge.uuids == ("5", "7")


def test_lsf_devices_preserve_release_all_overlap(tmp_path: Path) -> None:
    from rsi_harness.models import GPUAllocation, RunGPUPlan

    payload = _payload(tmp_path)
    placeholders = payload.run_plan.gpu_plan.authorized_pool.devices
    run_plan = payload.run_plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=placeholders),
                work=GPUAllocation(devices=placeholders),
                judge=GPUAllocation(devices=placeholders[:2]),
                judge_mode="release-all",
            )
        }
    )

    plan = bind_lsf_devices(run_plan, ("0", "2", "5", "7"))

    assert plan.gpu_plan.authorized_pool.uuids == ("0", "2", "5", "7")
    assert plan.gpu_plan.work.uuids == ("0", "2", "5", "7")
    assert plan.gpu_plan.judge.uuids == ("0", "2")
    assert plan.gpu_plan.judge_mode.value == "release-all"


def test_gpu_driver_invokes_native_engine_without_harbor_run(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    profile = load_cluster_profile("bluevela", {"USER": "alice"})
    output = (tmp_path / "run.sh").resolve()

    payload_path = render_engine_driver(payload, profile, output)

    script = output.read_text()
    assert "rsi_harness.cluster.bluevela.engine" in script
    assert "harbor run" not in script
    assert "harbor-jobs" not in script
    assert "RSI_HARNESS_NODE_TMP" in script
    assert "available_tmp_kb" in script
    assert "required_tmp_kb=$((15360 * 1024))" in script
    assert payload_path == output.with_suffix(".json")
    assert 'test -n "${CUDA_VISIBLE_DEVICES:-}"' in script
    assert "exec --nv --containall --no-mount hostfs" in script
    assert "--env" not in script
    assert "nvidia-smi" in script


def _cpu_payload(tmp_path: Path) -> EnginePayload:
    payload = _payload(tmp_path)
    plan = make_run_plan(tmp_path)
    return payload.model_copy(
        update={
            "profile": load_cluster_profile("slurm", {"USER": "alice"}),
            "run_plan": plan,
            "resources": payload.resources.model_copy(
                update={"work_gpus": 0, "verifier_gpus": 0, "total_gpus": 0}
            ),
        }
    )


def test_cpu_driver_omits_nvidia_preflight_and_clears_inherited_devices(
    tmp_path: Path,
) -> None:
    payload = _cpu_payload(tmp_path)
    output = tmp_path / "cpu-run.sh"

    render_engine_driver(payload, payload.profile, output)

    script = output.read_text()
    assert "nvidia-smi" not in script
    assert "--nv" not in script
    assert "unset CUDA_VISIBLE_DEVICES" in script
    assert 'APPTAINERENV_CUDA_VISIBLE_DEVICES=""' in script
    assert "rsi_harness.cluster.bluevela.engine" in script
    assert "sha256sum -c" in script
    assert "required_tmp_kb" in script
    assert "final_result.json" in script


@pytest.mark.parametrize("inherited_devices", [None, "GPU-inherited"])
def test_cpu_driver_executes_without_nvidia_tools(
    tmp_path: Path,
    monkeypatch,
    inherited_devices,
) -> None:
    import hashlib
    import os
    import shlex
    import subprocess

    payload = _cpu_payload(tmp_path)
    payload.sif_path.write_bytes(b"cpu-image")
    payload.sif_sha256_path.write_text(
        f"{hashlib.sha256(b'cpu-image').hexdigest()}  {payload.sif_path.name}\n"
    )
    result = (
        payload.run_plan.paths.logs
        / "runs"
        / payload.run_id
        / payload.run_plan.task.task_id
        / "final_result.json"
    )
    result.parent.mkdir(parents=True)
    engine = tmp_path / "fake-engine"
    engine.write_text(
        "#!/bin/sh\nset -eu\n"
        'test "${CUDA_VISIBLE_DEVICES-unset}" = unset\n'
        'test -z "$APPTAINERENV_CUDA_VISIBLE_DEVICES"\n'
        # Only the operator's temp and cache directories pass the scrub.
        'test "${APPTAINER_BINDPATH-unset}${SINGULARITY_BIND-unset}" = unsetunset\n'
        'test "$APPTAINER_CACHEDIR" = /scratch/cache\n'
        'test "$1" = -m\n'
        'test "$2" = rsi_harness.cluster.bluevela.engine\n'
        f"printf '{{}}' > {shlex.quote(str(result))}\n"
    )
    engine.chmod(0o700)
    # A CPU driver must never invoke the configured Apptainer GPU preflight.
    apptainer = tmp_path / "forbidden-nvidia-preflight"
    apptainer.write_text("#!/bin/sh\nexit 99\n")
    apptainer.chmod(0o700)
    profile = payload.profile.model_copy(
        update={
            "apptainer": payload.profile.apptainer.model_copy(
                update={"binary": apptainer, "temp_root": tmp_path}
            )
        }
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.sys.executable", str(engine)
    )
    output = tmp_path / "cpu-run.sh"
    render_engine_driver(payload, profile, output)
    env = {
        **os.environ,
        "APPTAINER_BINDPATH": "/oak",
        "SINGULARITY_BIND": "/oak",
        "APPTAINER_CACHEDIR": "/scratch/cache",
    }
    if inherited_devices is None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = inherited_devices

    completed = subprocess.run(
        ["/bin/bash", str(output)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert result.is_file()


@pytest.mark.parametrize("signalled", ["script", "job"])
def test_a_cancelled_job_waits_for_the_engine_to_close_its_attempt(
    tmp_path: Path, monkeypatch, signalled: str
) -> None:
    """scancel and the walltime signal the job script alone (Slurm) or the
    whole job (LSF), and kill what is left once the script exits: the
    script forwards SIGTERM to the Engine and waits for it."""
    import hashlib
    import os
    import signal
    import subprocess
    import sys
    import time

    payload = _cpu_payload(tmp_path)
    payload.sif_path.write_bytes(b"cpu-image")
    payload.sif_sha256_path.write_text(
        f"{hashlib.sha256(b'cpu-image').hexdigest()}  {payload.sif_path.name}\n"
    )
    ready, closed = tmp_path / "ready", tmp_path / "closed"
    engine = tmp_path / "fake-engine"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, signal, sys, time\n"
        "def close(*_):\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    time.sleep(1)\n"
        f"    pathlib.Path({str(closed)!r}).write_text('closed')\n"
        "    sys.exit(1)\n"
        "signal.signal(signal.SIGTERM, close)\n"
        f"pathlib.Path({str(ready)!r}).write_text('ready')\n"
        "time.sleep(60)\n"
    )
    engine.chmod(0o700)
    profile = payload.profile.model_copy(
        update={
            "apptainer": payload.profile.apptainer.model_copy(
                update={"temp_root": tmp_path, "binary": Path("/bin/sh")}
            )
        }
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.sys.executable", str(engine)
    )
    output = tmp_path / "cpu-run.sh"
    render_engine_driver(payload, profile, output)
    process = subprocess.Popen(
        ["/bin/bash", str(output)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 30
        while not ready.exists():
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        if signalled == "script":
            os.kill(process.pid, signal.SIGTERM)
        else:
            os.killpg(process.pid, signal.SIGTERM)

        assert process.wait(timeout=30) == 1
        assert closed.read_text() == "closed"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def test_cpu_engine_binds_an_empty_gpu_pool(tmp_path: Path, monkeypatch) -> None:
    payload = _cpu_payload(tmp_path)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    captured = []
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.run_native_engine",
        lambda _payload, plan, **kwargs: captured.append(plan),
    )

    run_engine_payload(payload)

    assert len(captured) == 1
    assert captured[0].gpu_plan.authorized_pool.devices == ()
    assert captured[0].gpu_plan.work.devices == ()
    assert captured[0].gpu_plan.judge.devices == ()


def test_cpu_device_binding_rejects_an_unexpected_gpu(tmp_path: Path) -> None:
    from rsi_harness.errors import SetupError

    with pytest.raises(SetupError, match="expected 0"):
        bind_lsf_devices(_cpu_payload(tmp_path).run_plan, ("GPU-unexpected",))


def test_engine_payload_keeps_native_submission_controls(tmp_path: Path) -> None:
    payload = _payload(tmp_path)

    assert payload.options.max_submissions == 4
    assert payload.run_plan.paths.logs.is_absolute()
    assert payload.run_plan.task.verifier.command == ("/bin/bash", "/tests/test.sh")
    assert payload.agent_launcher == "codex"


@pytest.mark.parametrize("clean_image", [True, False, None])
def test_engine_payload_loads_with_or_without_the_clean_image_field(
    tmp_path: Path, clean_image: bool | None
) -> None:
    # The cluster Judge always starts from the task SIF and ignores the field.
    payload = _payload(tmp_path).model_dump(mode="json")
    verifier = payload["run_plan"]["task"]["verifier"]
    if clean_image is None:
        del verifier["clean_image"]
    else:
        verifier["clean_image"] = clean_image
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload))

    loaded = load_engine_payload(path)

    assert loaded.run_plan.task.verifier.clean_image is bool(clean_image)


def _multi_payload(tmp_path: Path) -> EnginePayload:
    payload = _payload(tmp_path)
    from rsi_harness.models import GPUAllocation, GPUDevice, RunGPUPlan

    work = tuple(
        GPUDevice(
            index=index,
            uuid=f"WORK-{index // 8:03d}:GPU-{index % 8}",
            name="planned Work GPU",
        )
        for index in range(16)
    )
    judge = tuple(
        GPUDevice(
            index=16 + index,
            uuid=f"JUDGE-000:GPU-{index}",
            name="planned Judge GPU",
        )
        for index in range(8)
    )
    plan = payload.run_plan.model_copy(
        update={
            "gpu_plan": RunGPUPlan(
                authorized_pool=GPUAllocation(devices=work + judge),
                work=GPUAllocation(devices=work),
                judge=GPUAllocation(devices=judge),
                judge_mode="disjoint",
            )
        }
    )
    return payload.model_copy(
        update={
            "run_plan": plan,
            "resources": None,
            "multi_node": MultiNodeResources(
                work=PhaseResources(gpu_count=16, node_count=2),
                verifier=PhaseResources(gpu_count=8, node_count=1),
                total_nodes=3,
                gpus_per_node=8,
                cpu_slots_per_node=8,
                memory_mb_per_node=65536,
                shared_workspace_mb=15360,
                node_tmp_mb=71680,
                build_walltime="02:00",
                run_walltime="02:15",
            ),
        }
    )


def _allocated_pools() -> AllocatedPools:
    def node(host: str, subnet: int) -> AllocatedNode:
        return AllocatedNode(
            host=host,
            slots=8,
            ipv4=f"10.0.{subnet}.1",
            cuda_devices=tuple(f"GPU-{host}-{index}" for index in range(8)),
        )

    return AllocatedPools(
        run_id="native-engine-run",
        work=(node("work-a", 0), node("work-b", 1)),
        verifier=(node("judge-a", 2),),
        gpus_per_node=8,
        digest="a" * 64,
    )


def test_multinode_devices_bind_host_local_allocations_and_remain_disjoint(
    tmp_path: Path,
) -> None:
    payload = _multi_payload(tmp_path)

    plan = bind_multinode_devices(payload.run_plan, _allocated_pools())

    assert plan.gpu_plan.work.uuids[:9] == (
        "work-a:GPU-work-a-0",
        "work-a:GPU-work-a-1",
        "work-a:GPU-work-a-2",
        "work-a:GPU-work-a-3",
        "work-a:GPU-work-a-4",
        "work-a:GPU-work-a-5",
        "work-a:GPU-work-a-6",
        "work-a:GPU-work-a-7",
        "work-b:GPU-work-b-0",
    )
    assert plan.gpu_plan.judge.uuids[0] == "judge-a:GPU-judge-a-0"
    assert set(plan.gpu_plan.work.uuids).isdisjoint(plan.gpu_plan.judge.uuids)


def test_multinode_driver_uses_allocation_probe_without_controller_cuda_list(
    tmp_path: Path,
) -> None:
    payload = _multi_payload(tmp_path)
    output = (tmp_path / "multi-run.sh").resolve()

    render_engine_driver(payload, payload.profile, output)

    script = output.read_text()
    assert "LSB_MCPU_HOSTS" in script
    assert "CUDA_VISIBLE_DEVICES:-" not in script
    assert "required_tmp_kb=$((71680 * 1024))" in script
    assert "rsi_harness.cluster.bluevela.engine" in script
    assert "harbor run" not in script


def test_multinode_engine_freezes_pools_before_entering_native_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = _multi_payload(tmp_path)
    payload.run_plan.paths.root.mkdir(parents=True)
    (payload.run_plan.paths.root / "control").mkdir()
    payload.sif_sha256_path.write_text("d" * 64 + "  task.sif\n")
    pools = _allocated_pools()
    captured: list[tuple[object, object]] = []
    monkeypatch.setenv(
        "LSB_MCPU_HOSTS",
        "work-a 8 work-b 8 judge-a 8",
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.probe_and_partition",
        lambda **_kwargs: pools,
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.engine.socket.gethostname",
        lambda: "work-a.example",
    )
    monkeypatch.setattr(
        "rsi_harness.cluster.bluevela.runtime.run_native_engine",
        lambda _payload, plan, *, allocated_pools: captured.append(
            (plan, allocated_pools)
        ),
    )

    run_engine_payload(payload)

    plan, authority = captured[0]
    assert authority == pools
    assert plan.gpu_plan.work.uuids[0] == "work-a:GPU-work-a-0"
    frozen = payload.run_plan.paths.root / "control/ALLOCATED_POOLS.json"
    assert AllocatedPools.model_validate_json(frozen.read_text()) == pools
    assert frozen.stat().st_mode & 0o777 == 0o600
    assert not tuple(frozen.parent.glob(".ALLOCATED_POOLS.json.*.tmp"))


@pytest.mark.parametrize("compute_side", [False, True])
def test_payload_keeps_an_older_runtime_loading_it(
    tmp_path: Path, compute_side: bool
) -> None:
    """Submit-host-only keys never reach the payload; compute-side keys only
    when set, so an older runtime refuses only what it could not honour."""
    from tests.cluster.legacy_config import ClusterProfile as LegacyProfile

    payload = _cpu_payload(tmp_path)
    profile = payload.profile
    profile = profile.model_copy(
        update={
            "scheduler": profile.scheduler.model_copy(update={"gpu_option": "gpus"}),
            "builder": profile.builder.model_copy(
                update={
                    "conmon": Path("/opt/podman/bin/conmon"),
                    "runtime": Path("/opt/podman/bin/runc"),
                    "storage_driver": "vfs",
                    "host_namespaces": ("network", "uts"),
                }
            ),
            "apptainer": profile.apptainer.model_copy(
                update={"network_isolation": "seccomp"} if compute_side else {}
            ),
        }
    )
    payload = payload.model_copy(update={"profile": profile})
    output = tmp_path / "run.sh"

    serialized = json.loads(render_engine_driver(payload, profile, output).read_text())

    for section, name in (
        ("scheduler", "gpu_option"),
        ("builder", "conmon"),
        ("builder", "runtime"),
        ("builder", "storage_driver"),
        ("builder", "host_namespaces"),
        ("apptainer", "process_namespaces"),
    ):
        assert name not in serialized["profile"][section]
    loaded = load_engine_payload(output.with_suffix(".json"))
    assert loaded.profile.apptainer.network_isolation == (
        "seccomp" if compute_side else "namespace"
    )
    if compute_side:
        with pytest.raises(ValueError, match="network_isolation"):
            LegacyProfile.model_validate(serialized["profile"])
    else:
        LegacyProfile.model_validate(serialized["profile"])
