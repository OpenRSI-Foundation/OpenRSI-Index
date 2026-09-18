"""Immutable lightweight protocol and CPU-only coverage/numerical helpers."""
from __future__ import annotations
import math
import numpy as np

PROTOCOL_ID = "reasonir-lightweight-biology-pony-theorems-beir4-v1"
SUBJECTS = ("biology", "pony", "theoremqa_theorems")
BEIR_DATASETS = {"nfcorpus": "NFCorpus", "scifact": "SciFact", "fiqa": "FiQA-2018", "arguana": "ArguAna"}
DATASET_COUNTS = {"biology": (57359, 103), "pony": (7894, 112), "theoremqa_theorems": (23839, 76),
                  "nfcorpus": (3633, 323), "scifact": (5183, 300), "fiqa": (57638, 648), "arguana": (8674, 1406)}
TOTAL_METRIC_ENCODES = 167188
WORKER_COUNT = 4
DEADLINE_SECONDS = 4500
# Change only in trusted source, before BOTH calibration and formal evaluation.
# No environment/candidate override; 2/4 remain experimental until GPU proof.
BATCH_SIZE = 1
PADDED_TOKEN_BUDGET = 16384
LONG_TEXT_TOKENS = 8192


def descriptor():
    return {"id": PROTOCOL_ID, "bright_subjects": list(SUBJECTS), "beir_datasets": list(BEIR_DATASETS),
            "dataset_counts": {k: list(v) for k, v in DATASET_COUNTS.items()},
            "total_metric_encodes": TOTAL_METRIC_ENCODES, "worker_count": WORKER_COUNT,
            "batch_size": BATCH_SIZE, "padded_token_budget": PADDED_TOKEN_BUDGET,
            "long_text_tokens": LONG_TEXT_TOKENS, "precision": "bfloat16",
            "bright_max_length": 32768, "beir_max_length": 2048}


def validate_dataset_counts(key, documents, queries):
    if key not in DATASET_COUNTS or (documents, queries) != DATASET_COUNTS[key]:
        raise RuntimeError("lightweight dataset coverage mismatch")


def validate_metrics(bright, beir, encodes):
    if (set(bright) != set(SUBJECTS) or set(beir) != set(BEIR_DATASETS.values())
            or type(encodes) is not int or encodes != TOTAL_METRIC_ENCODES):
        raise RuntimeError("incomplete lightweight metric coverage")
    if not all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
               and 0 <= v <= 1 for v in [*bright.values(), *beir.values()]):
        raise RuntimeError("invalid lightweight metric")
    return round(sum(bright.values()) / 3, 5), round(sum(beir.values()) / 4, 5)


def baseline_values(artifact):
    if not isinstance(artifact, dict) or artifact.get("provenance", {}).get("protocol") != descriptor():
        raise RuntimeError("incompatible baseline protocol")
    values = [artifact.get("R"), artifact.get("G")]
    try:
        valid = all(isinstance(v, str) and math.isfinite(float(v)) and 0 <= float(v) <= 1 for v in values)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError("invalid baseline aggregates")
    return tuple(values)


def batch_indices(lengths, batch_size):
    if type(batch_size) is not int or batch_size not in (1, 2, 4):
        raise RuntimeError("unsupported batch policy")
    if any(type(n) is not int or not 0 < n <= 32768 for n in lengths):
        raise RuntimeError("invalid bounded token lengths")
    if batch_size == 1:
        return [[i] for i in range(len(lengths))]
    groups, current = [], []
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        length = lengths[index]
        if current and (length > LONG_TEXT_TOKENS or len(current) == batch_size
                        or (len(current) + 1) * length > PADDED_TOKEN_BUDGET):
            groups.append(current)
            current = []
        if length > LONG_TEXT_TOKENS:
            groups.append([index])
        else:
            current.append(index)
    if current:
        groups.append(current)
    return groups


def restore_batches(size, groups, arrays):
    if len(groups) != len(arrays) or not arrays:
        raise RuntimeError("incomplete batch output")
    dimension = arrays[0].shape[1] if isinstance(arrays[0], np.ndarray) and arrays[0].ndim == 2 else 0
    output = np.empty((size, dimension), dtype=np.float32)
    seen = set()
    for group, array in zip(groups, arrays):
        if (not isinstance(array, np.ndarray) or array.dtype != np.float32
                or array.shape != (len(group), dimension) or not np.isfinite(array).all()
                or any(type(i) is not int or not 0 <= i < size or i in seen for i in group)
                or len(set(group)) != len(group)):
            raise RuntimeError("malformed batch output")
        output[group] = array
        seen.update(group)
    if len(seen) != size:
        raise RuntimeError("incomplete batch output")
    return output


def encode_batched(model, texts, instruction, max_length, batch_size):
    encoder = model.base_model.model
    if type(batch_size) is not int or batch_size not in (1, 2, 4) or max_length not in (2048, 32768):
        raise RuntimeError("unsupported encode policy")
    if batch_size == 1:
        return encoder.encode(texts, instruction=instruction, batch_size=1,
                              max_length=max_length, convert_to_tensor=False)
    # Exactly upstream instruction + text + embed_eos, special tokens and fixed
    # truncation. One bounded token-ID row at a time; no corpus token cache.
    lengths = [len(encoder.tokenizer(instruction + text + encoder.embed_eos,
                    padding=False, truncation=True, max_length=max_length,
                    add_special_tokens=True)["input_ids"]) for text in texts]
    groups = batch_indices(lengths, batch_size)
    arrays = [encoder.encode([texts[i] for i in group], instruction=instruction,
              batch_size=len(group), max_length=max_length, convert_to_tensor=False) for group in groups]
    return restore_batches(len(texts), groups, arrays)


def compare_embeddings(reference, actual, query_count):
    for array in (reference, actual):
        if (not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.ndim != 2
                or not np.isfinite(array).all()):
            raise RuntimeError("invalid synthetic embedding")
    if actual.shape != reference.shape or not 0 < query_count < len(reference):
        raise RuntimeError("invalid synthetic embedding shape")
    drift = float(np.max(np.abs(reference - actual)))
    ref64, new64 = reference.astype(np.float64), actual.astype(np.float64)
    norm = np.linalg.norm(ref64, axis=1) * np.linalg.norm(new64, axis=1)
    if np.any(norm == 0):
        raise RuntimeError("zero-norm synthetic embedding")
    min_cosine = float(np.min(np.sum(ref64 * new64, axis=1) / norm))
    ref_order = np.argsort(-(reference[:query_count] @ reference[query_count:].T), axis=1, kind="stable")
    new_order = np.argsort(-(actual[:query_count] @ actual[query_count:].T), axis=1, kind="stable")
    ranking_equal = bool(np.array_equal(ref_order, new_order))
    # Public fixture: each query's corresponding document is relevant. Same
    # NDCG@10 discount as retrieval, measured independently for both arrays.
    def ndcg(order):
        return float(np.mean([1 / math.log2(list(row[:10]).index(i) + 2) if i in row[:10] else 0
                              for i, row in enumerate(order)]))
    ref_metric, metric = ndcg(ref_order), ndcg(new_order)
    return {"passed": bool(drift <= 0.0001 and min_cosine >= 0.99999 and ranking_equal and ref_metric == metric),
            "min_cosine_similarity": min_cosine,
            "max_abs_drift": drift, "ranking_equal": ranking_equal,
            "ndcg_at_10_equal": ref_metric == metric, "reference_ndcg_at_10": ref_metric,
            "candidate_ndcg_at_10": metric, "shape": list(actual.shape), "dtype": str(actual.dtype)}
