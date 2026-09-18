"""Task-owned, bounded data parallelism at the unchanged encode boundary.

Importing this module never imports Torch. Spawn workers restrict GPU visibility
before loading the immutable evaluator and model. No candidate code is imported.
"""
from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import evaluation_protocol as protocol

ZERO_ROOT = Path("/opt/reasonir-task/base/zero-adapter")
CHUNK_SIZE = 256
DIMENSION = 4096


class DeviceAuthorizationError(RuntimeError):
    hint = "numeric CUDA mask requires explicit CUDA_DEVICE_ORDER=PCI_BUS_ID; use full UUID selectors otherwise"


class WorkerCandidateInvalid(Exception):
    """Already-redacted worker diagnostic, re-sanitized again by formal output."""
    def __init__(self, payload):
        super().__init__("worker candidate invalid")
        self.payload = payload


def raise_worker_failure(response):
    if isinstance(response, dict) and response.get("op") == "error":
        if isinstance(response.get("candidate_invalid"), dict):
            raise WorkerCandidateInvalid(response["candidate_invalid"])
        raise RuntimeError("parallel worker failed")


def validate_gpu_uuids(gpus):
    if (not isinstance(gpus, list) or len(gpus) != 4 or len(set(gpus)) != 4
            or any(not isinstance(gpu, str) or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu) is None for gpu in gpus)):
        raise RuntimeError("exactly four unique authorized GPU UUIDs required")


def select_gpu_uuids(inventory, environment):
    """NVML inventory is container-visible; CUDA ordinals use explicit PCI order.

    NVIDIA ordinals refer to NVML indices, CUDA ordinals to the container's
    PCI-ordered authorized subset. UUIDs must be fully specified, never prefixes.
    """
    if (not inventory or any(len(row) != 3 for row in inventory)
            or len({row[0] for row in inventory}) != len(inventory)
            or len({row[1] for row in inventory}) != len(inventory)):
        raise RuntimeError("invalid authorized device inventory")
    allowed = list(inventory)
    nvidia = environment.get("NVIDIA_VISIBLE_DEVICES", "all")
    # CDI may set NVIDIA_VISIBLE_DEVICES=void after device injection to stop
    # further runtime modification. Empty/unset are likewise control markers,
    # not device names. Only the actually exposed inventory can be selected;
    # the CUDA mask below still restricts it, and explicit 'none' still denies.
    if nvidia not in ("all", "void", "", None):
        selectors = nvidia.split(",")
        if len(set(selectors)) != len(selectors):
            raise RuntimeError("duplicate device selector")
        selected = []
        for selector in selectors:
            matches = [row for row in inventory if selector in row[:2]]
            if len(matches) != 1:
                raise RuntimeError("device outside authorized inventory")
            selected.append(matches[0])
        allowed = selected
    allowed.sort(key=lambda row: row[2])
    cuda = environment.get("CUDA_VISIBLE_DEVICES")
    if cuda is not None:
        if (any(selector.isdecimal() for selector in cuda.split(","))
                and environment.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"):
            raise DeviceAuthorizationError(DeviceAuthorizationError.hint)
        selected = []
        for selector in cuda.split(","):
            if selector.isdecimal():
                index = int(selector)
                matches = allowed[index:index + 1]
            else:
                matches = [row for row in allowed if row[1] == selector]
            if len(matches) != 1:
                raise RuntimeError("CUDA device outside authorized inventory")
            selected.append(matches[0])
        allowed = selected
    result = [row[1] for row in allowed]
    validate_gpu_uuids(result)
    return result


def discover_gpu_uuids():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    inventory = [tuple(field.strip() for field in line.split(",")) for line in result.stdout.splitlines() if line.strip()]
    return select_gpu_uuids(inventory, os.environ)


def configure_worker_environment(gpu):
    if not isinstance(gpu, str) or re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu) is None:
        raise RuntimeError("invalid worker GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.update({"TOKENIZERS_PARALLELISM": "false", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})


def load_evaluator():
    spec = importlib.util.spec_from_file_location("fixed_reasonir_evaluator", "/tests/evaluate.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def validate_candidate(evaluator, canonical=False):
    evaluator.verify_hash_manifest(evaluator.TESTS / "assets", evaluator.TESTS / "assets/manifest.json", True)
    evaluator.verify_hash_manifest(evaluator.BASE, evaluator.TESTS / "model-lock.json", False)
    evaluator.verify_workspace()
    adapter, baseline, digest = evaluator.validate_adapter()
    if canonical and baseline is not True:
        raise RuntimeError("parallel calibration requires canonical baseline")
    for name in ("adapter_model.safetensors", "adapter_config.json") if canonical else ():
        if (adapter / name).resolve(strict=True) != (ZERO_ROOT / name).resolve(strict=True):
            raise RuntimeError("parallel calibration requires canonical baseline links")
    binding = {"adapter_sha256": digest, "baseline": baseline, "adapter_path": str(adapter.resolve(strict=True)),
               "config_sha256": evaluator.sha256(adapter / "adapter_config.json"),
               "manifest_sha256": evaluator.sha256(evaluator.SUBMISSION / "manifest.json")}
    return adapter, binding


def recheck_binding(evaluator, adapter, binding):
    if (str(adapter.resolve(strict=True)) != binding["adapter_path"]
            or evaluator.sha256(adapter / "adapter_model.safetensors") != binding["adapter_sha256"]
            or evaluator.sha256(adapter / "adapter_config.json") != binding["config_sha256"]
            or evaluator.sha256(evaluator.SUBMISSION / "manifest.json") != binding["manifest_sha256"]):
        raise RuntimeError("adapter immutable binding changed")


def scratch_environment(scratch):
    os.environ.update({"HF_HOME": str(scratch / "hf"), "HF_MODULES_CACHE": str(scratch / "modules"),
                       "TRANSFORMERS_CACHE": str(scratch / "transformers"), "XDG_CACHE_HOME": str(scratch / "xdg"),
                       "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})


def validate_worker_stats(stats, gpus, encodes):
    if (len(stats) != 4 or [s.get("gpu_uuid") for s in stats] != gpus
            or len({s.get("pid") for s in stats}) != 4):
        raise RuntimeError("invalid worker identity coverage")
    for stat in stats:
        if type(stat.get("pid")) is not int or stat["pid"] <= 0 or type(stat.get("metric_encodes")) is not int:
            raise RuntimeError("invalid worker identity/count statistics")
        for key in ("metric_encodes", "peak_gpu_memory_mb", "load_seconds"):
            value = stat.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError("invalid worker resource statistics")
    if sum(s["metric_encodes"] for s in stats) != encodes:
        raise RuntimeError("worker encode coverage mismatch")


def run_evaluation(evaluator, gpus, canonical=False, pool_factory=None):
    """One common candidate/calibration path; never load a coordinator model."""
    validate_gpu_uuids(gpus)
    adapter, binding = validate_candidate(evaluator, canonical)
    factory = ParallelEncoder if pool_factory is None else pool_factory
    with tempfile.TemporaryDirectory(prefix="reasonir-parallel-evaluation-", dir="/tmp") as temporary:
        scratch = Path(temporary)
        scratch_environment(scratch)
        with factory(gpus, binding, canonical=canonical) as model:
            consistency = model.consistency_check()
            if consistency != {"passed": True, "bitwise_equal": True, "worker_count": 4,
                               "text_count": 3, "protocol_count": 2, "reference_worker": 0}:
                raise RuntimeError("synthetic bitwise consistency did not pass")
            bright = evaluator.evaluate_bright(model, scratch)
            beir = evaluator.evaluate_beir(model, scratch)
            r, g = protocol.validate_metrics(bright, beir, model.total_metric_encodes)
            stats = model.collect_stats()
            validate_worker_stats(stats, gpus, model.total_metric_encodes)
            recheck_binding(evaluator, adapter, binding)
    return {"R": r, "G": g, "bright": bright, "beir": beir, "binding": binding,
            "adapter": adapter, "worker_stats": stats, "synthetic_consistency": consistency,
            "total_metric_encodes": protocol.TOTAL_METRIC_ENCODES,
            "peak_gpu_memory_mb": round(sum(s["peak_gpu_memory_mb"] for s in stats), 1)}


class ChunkAssembler:
    def __init__(self, size, call_id, dimension=DIMENSION):
        self.output = np.empty((size, dimension), dtype=np.float32)
        self.seen = np.zeros(size, dtype=np.bool_)
        self.call_id = call_id

    def accept(self, call_id, offset, count, embedding):
        if (type(call_id) is not int or call_id != self.call_id or type(offset) is not int
                or type(count) is not int or offset < 0 or count <= 0
                or offset + count > len(self.output)):
            raise RuntimeError("stale or out-of-range chunk")
        if (not isinstance(embedding, np.ndarray) or embedding.dtype != np.float32
                or embedding.shape != (count, self.output.shape[1]) or not np.isfinite(embedding).all()):
            raise RuntimeError("malformed or nonfinite embedding chunk")
        if self.seen[offset:offset + count].any():
            raise RuntimeError("duplicate embedding chunk")
        self.output[offset:offset + count] = embedding
        self.seen[offset:offset + count] = True

    def finish(self):
        if not self.seen.all():
            raise RuntimeError("incomplete embedding coverage")
        return self.output


def dispatch_chunks(texts, instruction, max_length, connections, call_id, wait_ready, check_alive,
                    chunk_size=CHUNK_SIZE, dimension=DIMENSION):
    """One outstanding job per worker; return exact original row order."""
    if not connections or chunk_size <= 0:
        raise RuntimeError("workers and bounded chunks required")
    assembler = ChunkAssembler(len(texts), call_id, dimension)
    pending = {}
    next_offset = 0
    counts = [0] * len(connections)
    while next_offset < len(texts) or pending:
        check_alive()
        for index, connection in enumerate(connections):
            if connection not in pending and next_offset < len(texts):
                count = min(chunk_size, len(texts) - next_offset)
                connection.send({"op": "encode", "call_id": call_id, "offset": next_offset,
                    "texts": texts[next_offset:next_offset + count], "instruction": instruction,
                    "max_length": max_length, "metric": True})
                pending[connection] = (index, next_offset, count)
                next_offset += count
        ready_connections = wait_ready(list(pending), 1)
        check_alive()
        for connection in ready_connections:
            if connection not in pending:
                raise RuntimeError("unexpected worker response")
            index, offset, count = pending.pop(connection)
            response = connection.recv()
            check_alive()
            raise_worker_failure(response)
            if (not isinstance(response, dict) or response.get("op") != "encoded"
                    or type(response.get("worker")) is not int or response.get("worker") != index
                    or response.get("call_id") != call_id
                    or response.get("offset") != offset or response.get("count") != count):
                raise RuntimeError("worker failed or response does not match assignment")
            assembler.accept(response["call_id"], response["offset"], response["count"], response.get("embedding"))
            counts[index] += count
    return assembler.finish(), counts


def require_bitwise_equal(reference, actual):
    if (not isinstance(actual, np.ndarray) or reference.dtype != np.float32
            or actual.dtype != np.float32 or actual.shape != reference.shape
            or not np.isfinite(reference).all() or not np.isfinite(actual).all()
            or reference.tobytes() != actual.tobytes()):
        raise RuntimeError("synthetic serial/parallel bitwise consistency failed")


def worker_main(connection, index, gpu, binding, canonical):
    # Spawn imports this module (NumPy only); Torch is first imported below.
    configure_worker_environment(gpu)
    stage = "initialization"
    evaluator = None
    try:
        evaluator = load_evaluator()
        if evaluator.torch.cuda.device_count() != 1:
            raise RuntimeError("worker must see exactly one GPU")
        evaluator.torch.cuda.set_device(0)
        started = time.monotonic()
        adapter, actual_binding = validate_candidate(evaluator, canonical)
        if actual_binding != binding:
            raise RuntimeError("worker adapter binding differs")
        with tempfile.TemporaryDirectory(prefix=f"reasonir-worker-{index}-", dir="/tmp") as temporary:
            scratch_environment(Path(temporary))
            evaluator.torch.cuda.reset_peak_memory_stats()
            base = evaluator.load_base_model()
            model = evaluator.load_candidate(base, adapter)
            recheck_binding(evaluator, adapter, binding)
            load_seconds = time.monotonic() - started
            metric_encodes = 0
            connection.send({"op": "ready", "worker": index, "gpu_uuid": gpu,
                             "pid": os.getpid(), "binding": binding})
            while True:
                request = connection.recv()
                operation = request.get("op")
                if operation == "stop":
                    return
                if operation == "stats":
                    connection.send({"op": "stats", "worker": index, "stats": {
                        "gpu_uuid": gpu, "pid": os.getpid(), "load_seconds": round(load_seconds, 3),
                        "metric_encodes": metric_encodes,
                        "peak_gpu_memory_mb": round(evaluator.torch.cuda.max_memory_allocated() / 1024**2, 1)}})
                    continue
                if operation == "probe":
                    stage = "synthetic_probe"
                    from probe_batching import measure_model
                    result = measure_model(evaluator, model)
                    connection.send({"op": "probe", "worker": index, "result": result})
                    continue
                stage = "encoding"
                if (operation != "encode" or not isinstance(request.get("texts"), list)
                        or not 0 < len(request["texts"]) <= CHUNK_SIZE
                        or not all(isinstance(text, str) for text in request["texts"])
                        or not isinstance(request.get("instruction"), str)
                        or request.get("max_length") not in (2048, 32768)
                        or type(request.get("metric")) is not bool):
                    raise RuntimeError("invalid encode request")
                embedding = evaluator.encode(model, request["texts"], request["instruction"], request["max_length"])
                if request["metric"]:
                    metric_encodes += len(request["texts"])
                connection.send({"op": "encoded", "worker": index, "call_id": request["call_id"],
                    "offset": request["offset"], "count": len(request["texts"]), "embedding": embedding})
    except BaseException as exc:
        # Preserve task-owned structured diagnostics; raw exception text is never IPC.
        try:
            failure = {"op": "error", "worker": index, "stage": stage, "error": type(exc).__name__}
            if evaluator is not None and isinstance(exc, getattr(evaluator, "CandidateInvalid", ())):
                failure["candidate_invalid"] = evaluator.safe_candidate_diagnostic(exc.payload)
            connection.send(failure)
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise SystemExit(1)
    finally:
        connection.close()


class ParallelEncoder:
    def __init__(self, gpus, binding, canonical=False):
        validate_gpu_uuids(gpus)
        self.gpus = list(gpus)
        self.binding, self.canonical = binding, canonical
        self.connections = []
        self.processes = []
        self.call_id = 0
        self.total_metric_encodes = 0
        self.counts = [0] * len(gpus)
        self.deadline = time.monotonic() + protocol.DEADLINE_SECONDS
        # Preserve evaluate.encode unchanged through its model-shaped interface.
        self.base_model = SimpleNamespace(model=self)

    def check_alive(self):
        self.check_deadline()
        if any(not process.is_alive() for process in self.processes):
            raise RuntimeError("parallel worker exited")

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("parallel evaluation deadline")

    def receive(self, connection):
        self.check_deadline()
        while not connection.poll(1):
            self.check_alive()
        self.check_deadline()
        response = connection.recv()
        self.check_deadline()
        # An exiting failed worker may have buffered its actionable diagnostic.
        # Drain that reply before liveness rejection; never accept late success.
        raise_worker_failure(response)
        self.check_alive()
        if not isinstance(response, dict):
            raise RuntimeError("parallel worker failed")
        return response

    def __enter__(self):
        context = mp.get_context("spawn")
        try:
            for index, gpu in enumerate(self.gpus):
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=worker_main, args=(child, index, gpu, self.binding, self.canonical), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
                response = self.receive(parent)  # bounded, staggered model initialization
                if (response.get("op") != "ready" or response.get("worker") != index
                        or response.get("gpu_uuid") != gpu or response.get("pid") != process.pid
                        or response.get("binding") != self.binding):
                    raise RuntimeError("worker initialization binding failed")
                print(f"worker_ready index={index} count=4", flush=True)
            return self
        except BaseException:
            self.close()
            raise

    def close(self):
        failed = False
        for connection in self.connections:
            try:
                connection.close()
            except OSError:
                pass
        for process in self.processes:
            try:
                if process.is_alive():
                    process.terminate()
            except OSError:
                pass
        for process in self.processes:
            try:
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
                failed = failed or process.is_alive()
            except OSError:
                failed = True
        if failed:
            raise RuntimeError("worker cleanup could not confirm exit")

    def __exit__(self, *args):
        self.close()

    def encode(self, sentences, *, instruction, batch_size, max_length, convert_to_tensor):
        if (batch_size != 1 or convert_to_tensor is not False or max_length not in (2048, 32768)
                or not isinstance(sentences, list) or not all(isinstance(s, str) for s in sentences)):
            raise RuntimeError("encode boundary differs from fixed protocol")
        self.call_id += 1
        result, counts = dispatch_chunks(sentences, instruction, max_length, self.connections,
                                         self.call_id, wait, self.check_alive)
        self.counts = [a + b for a, b in zip(self.counts, counts)]
        self.total_metric_encodes += len(sentences)
        print(f"encode_complete call={self.call_id} rows={len(sentences)} total={self.total_metric_encodes}", flush=True)
        return result

    def consistency_check(self):
        texts = ["A short nonempty calibration sentence.", "A calibration sentence about careful measurement.", "Unicode α β: two plus two equals four."]
        for instruction, maximum in (("Represent this text: ", 32768), ("", 2048)):
            reference = None
            for index in [-1, *range(len(self.connections))]:
                worker = max(index, 0)
                self.call_id += 1
                self.connections[worker].send({"op": "encode", "call_id": self.call_id, "offset": 0,
                    "texts": texts, "instruction": instruction, "max_length": maximum, "metric": False})
                response = self.receive(self.connections[worker])
                if (response.get("op") != "encoded" or response.get("worker") != worker
                        or response.get("call_id") != self.call_id or response.get("offset") != 0
                        or response.get("count") != len(texts)):
                    raise RuntimeError("synthetic consistency response mismatch")
                assembler = ChunkAssembler(len(texts), self.call_id)
                assembler.accept(self.call_id, 0, len(texts), response.get("embedding"))
                actual = assembler.finish()
                if reference is None:
                    reference = actual
                else:
                    require_bitwise_equal(reference, actual)
        print("synthetic_consistency=PASS bitwise=true workers=4 protocols=2", flush=True)
        return {"passed": True, "bitwise_equal": True, "worker_count": len(self.connections),
                "text_count": 3, "protocol_count": 2, "reference_worker": 0}

    def collect_stats(self):
        stats = []
        for index, connection in enumerate(self.connections):
            connection.send({"op": "stats"})
            response = self.receive(connection)
            if response.get("op") != "stats" or response.get("worker") != index:
                raise RuntimeError("worker stats response mismatch")
            item = response["stats"]
            if (item.get("gpu_uuid") != self.gpus[index] or item.get("pid") != self.processes[index].pid
                    or item.get("metric_encodes") != self.counts[index]):
                raise RuntimeError("worker metric coverage mismatch")
            stats.append(item)
        return stats

    def probe(self):
        # Only public synthetic inputs are generated inside the worker. The
        # coordinator receives aggregate checks, never embeddings or rankings.
        self.connections[0].send({"op": "probe"})
        response = self.receive(self.connections[0])
        if response.get("op") != "probe" or response.get("worker") != 0 or not isinstance(response.get("result"), dict):
            raise RuntimeError("invalid synthetic probe response")
        return response["result"]
