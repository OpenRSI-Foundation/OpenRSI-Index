#!/usr/bin/env python3
"""Safe public-fixture batch probe. Never benchmarks hidden data or emits reward."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation_protocol as protocol
import parallel_encoder as parallel
from calibrate_baseline_multi_gpu import sha256


def measure_model(evaluator, model):
    # Public fixtures only: enough short rows for real 2/4 batches, varied
    # lengths, and a long singleton in BRIGHT (truncated normally for BEIR).
    queries = ["A short sentence about " + topic + "." for topic in
               ("biology", "programming", "mathematics", "music", "weather", "books", "history", "measurement")]
    documents = [queries[i] + " More explanatory context." * (i + 1) for i in range(8)]
    documents[-1] += " careful measurement" * 5000
    texts = queries + documents
    results = []
    for instruction, maximum in (("Represent this text: ", 32768), ("", 2048)):
        policies = {}
        reference = None
        # Warmup is synthetic, not counted as metric work or timed reference.
        protocol.encode_batched(model, queries[:1], instruction, maximum, 1)
        for size in (1, 2, 4):
            evaluator.torch.cuda.synchronize()
            evaluator.torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            actual = protocol.encode_batched(model, texts, instruction, maximum, size)
            evaluator.torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            if reference is None:
                reference = actual
            comparison = protocol.compare_embeddings(reference, actual, len(queries))
            policies[str(size)] = {**comparison, "seconds": elapsed,
                "texts_per_second": len(texts) / elapsed,
                "peak_gpu_memory_mb": round(evaluator.torch.cuda.max_memory_allocated() / 1024**2, 1)}
        results.append({"max_length": maximum, "instruction_present": bool(instruction),
                        "text_count": len(texts), "query_count": len(queries), "policies": policies})
    return {"synthetic_only": True, "full_metric_equivalence_proven": False,
            "protocol_results": results, "production_batch_size": protocol.BATCH_SIZE,
            "passed": all(row["policies"][str(size)]["passed"] for row in results for size in (2, 4))}


def measure(evaluator, log_directory, gpus, pool_factory=None):
    if not log_directory.is_dir() or any(log_directory.iterdir()):
        raise RuntimeError("probe requires fresh verifier logs")
    adapter, binding = parallel.validate_candidate(evaluator, canonical=True)
    factory = parallel.ParallelEncoder if pool_factory is None else pool_factory
    with factory(gpus, binding, canonical=True) as pool:
        if pool.consistency_check() != {"passed": True, "bitwise_equal": True, "worker_count": 4,
                                        "text_count": 3, "protocol_count": 2, "reference_worker": 0}:
            raise RuntimeError("synthetic worker consistency failed")
        result = pool.probe()
        stats = pool.collect_stats()
        parallel.validate_worker_stats(stats, gpus, 0)
        parallel.recheck_binding(evaluator, adapter, binding)
    result.update({"protocol": protocol.descriptor(), "worker_stats": stats,
                   "protocol_sha256": sha256(protocol.__file__), "worker_sha256": sha256(parallel.__file__),
                   "evaluator_sha256": sha256(evaluator.__file__), "probe_sha256": sha256(__file__),
                   "adapter_sha256": binding["adapter_sha256"]})
    with (log_directory / "batch-probe.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def main():
    try:
        gpus = parallel.discover_gpu_uuids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        result = measure(parallel.load_evaluator(), Path("/logs/verifier"), gpus)
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as exc:
        print(json.dumps({"status": "probe_failed", "error": type(exc).__name__, "protocol": protocol.PROTOCOL_ID}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__": main()
