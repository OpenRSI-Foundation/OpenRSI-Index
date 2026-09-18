#!/usr/bin/env python3
"""Four-GPU canonical calibration using the formal lightweight Judge path."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parallel_encoder as parallel
import evaluation_protocol as protocol


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(evaluator, log_directory, gpu_uuids, image_id, pool_factory=None):
    if not log_directory.is_dir() or any(log_directory.iterdir()):
        raise RuntimeError("parallel calibration requires fresh verifier logs")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise RuntimeError("immutable image ID required")
    started = time.monotonic()
    measured = parallel.run_evaluation(evaluator, gpu_uuids, canonical=True, pool_factory=pool_factory)
    binding = measured["binding"]
    result = {"R": f"{measured['R']:.5f}", "G": f"{measured['G']:.5f}", "provenance": {
        "schema_version": 3, "mode": "validator-only-canonical-zero-baseline-parallel-lightweight",
        "protocol": protocol.descriptor(), "protocol_sha256": sha256(protocol.__file__),
        "base_revision": evaluator.BASE_REVISION, "adapter_sha256": binding["adapter_sha256"],
        "adapter_config_sha256": binding["config_sha256"],
        "evaluator_sha256": sha256(evaluator.__file__), "calibrator_sha256": sha256(__file__),
        "worker_sha256": sha256(parallel.__file__),
        "model_lock_sha256": sha256(evaluator.TESTS / "model-lock.json"),
        "asset_manifest_sha256": sha256(evaluator.TESTS / "assets/manifest.json"),
        "workspace_manifest_sha256": sha256(evaluator.TESTS / "workspace-tree.json"),
        "baseline_manifest_sha256": binding["manifest_sha256"],
        "image_id": image_id, "gpu_uuids": gpu_uuids, "worker_count": 4, "batch_size": protocol.BATCH_SIZE,
        "chunk_size": parallel.CHUNK_SIZE, "max_inflight_per_worker": 1,
        "bright_subject_count": 3, "beir_dataset_count": 4, "total_metric_encodes": 167188,
        "synthetic_consistency": measured["synthetic_consistency"], "worker_stats": measured["worker_stats"],
        "peak_gpu_memory_mb": measured["peak_gpu_memory_mb"],
        "runtime_sec": round(time.monotonic() - started, 3),
    }}
    protocol.baseline_values(result)
    with (log_directory / "baseline-calibration.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return result


def main():
    try:
        gpus = parallel.discover_gpu_uuids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        evaluator = parallel.load_evaluator()
        result = measure(evaluator, Path("/logs/verifier"), gpus,
                         os.environ.get("REASONIR_CALIBRATION_IMAGE_ID", ""))
        print(json.dumps(result, sort_keys=True), flush=True)
    except BaseException as exc:
        print(json.dumps({"status": "calibration_failed", "error": type(exc).__name__, "protocol": protocol.PROTOCOL_ID}), flush=True)
        raise SystemExit(1)


if __name__ == "__main__": main()
