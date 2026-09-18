#!/usr/bin/env python3
"""Trusted candidate-only ReasonIR evaluation over task-owned offline assets."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import zipfile

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
# -I excludes script directory: import ONLY readonly task-owned helpers.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluation_protocol as protocol
import parallel_encoder as parallel


def __getattr__(name):
    # Workers establish CUDA visibility before this first Torch import. Keeping
    # spawn's __main__ import GPU-library-free also covers the formal entrypoint.
    if name == "torch":
        import torch
        return torch
    raise AttributeError(name)

TESTS = Path("/tests")
WORKSPACE = Path("/workspace")
SUBMISSION = WORKSPACE / "submission"
BASE = Path("/opt/reasonir-task/base/ReasonIR-8B")
BASE_REVISION = "c3d0690370ff4a8c3d3882d8dfa85c43650034fa"
TARGETS = ["q_proj", "o_proj", "v_proj", "k_proj", "w1", "w2", "w3"]
CANONICAL_LORA = {
    "r": 16,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "inference_mode": False,
    "target_modules": TARGETS,
    "bias": "none",
    "task_type": "FEATURE_EXTRACTION",
}
EXPECTED_CONFIG_DEFAULTS = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": str(BASE),
    "bias": "none",
    "eva_config": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layer_replication": None,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": 64,
    "lora_bias": False,
    "lora_dropout": 0.1,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": 16,
    "rank_pattern": {},
    "revision": None,
    "task_type": "FEATURE_EXTRACTION",
    "use_dora": False,
    "use_rslora": False,
}
EXPECTED_CONFIG_FIELDS = set(EXPECTED_CONFIG_DEFAULTS) | {"target_modules"}
PROJECTION_SHAPES = {
    "q_proj": ((16, 4096), (4096, 16)),
    "o_proj": ((16, 4096), (4096, 16)),
    "v_proj": ((16, 4096), (1024, 16)),
    "k_proj": ((16, 4096), (1024, 16)),
    "w1": ((16, 4096), (14336, 16)),
    "w2": ((16, 14336), (4096, 16)),
    "w3": ((16, 4096), (14336, 16)),
}
BRIGHT_MAX_LENGTH = 32768
SUBJECTS = protocol.SUBJECTS
TASK_NAMES = {"biology": "Biology", "earth_science": "Earth Science", "economics": "Economics", "psychology": "Psychology", "robotics": "Robotics", "stackoverflow": "Stack Overflow", "sustainable_living": "Sustainable Living"}
BEIR_DATASETS = protocol.BEIR_DATASETS


class CandidateInvalid(Exception):
    def __init__(self, code: str, path: Path, condition: str, expected, actual, hint: str) -> None:
        super().__init__(condition)
        self.payload = {"status": "candidate_invalid", "code": code, "path": str(path), "condition": condition, "expected": expected, "actual": actual, "hint": hint}


def safe_candidate_diagnostic(payload):
    """Code-specific task-owned expectations; never serialize raw model errors.

    Called both before worker transport and at the final output boundary.
    Candidate strings, arbitrary paths/keys and unbounded numbers are not logs.
    """
    submission = "/workspace/submission"
    manifest, config, adapter = (submission + suffix for suffix in
        ("/manifest.json", "/adapter/adapter_config.json", "/adapter/adapter_model.safetensors"))
    fixed_budget = {"updates": 1000, "exposures": 64000, "world_size": 4, "per_device_batch": 4, "gradient_accumulation": 4}
    training_fields = set(fixed_budget) | {"schedule_sha256", "policy_sha256", "accounting_sha256"}
    normal_fields = {"schema_version", "base_revision", "adapter_file", "adapter_sha256", "adapter_config_file", "adapter_config_sha256", "lora", "training"}
    baseline_fields = normal_fields - {"training"} | {"baseline"}
    inventory = ["adapter/adapter_config.json", "adapter/adapter_model.safetensors", "manifest.json"]
    # Values here are trusted definitions, not values copied from exception text.
    definitions = {
        "manifest": (manifest, "regular JSON <=65536 bytes", "manifest missing or too large", "run the fixed selector or baseline Solution"),
        "manifest_parse": (manifest, "JSON object", "manifest is not valid bounded JSON", "run the fixed selector or baseline Solution"),
        "manifest_type": (manifest, "object", "manifest root is not an object", "run the fixed selector or baseline Solution"),
        "manifest_fields": (manifest, sorted(normal_fields), "manifest field set differs", "use a fixed task materializer"),
        "submission_inventory": (submission, inventory, "submission must contain only one adapter and manifest", "remove every non-ingestion file from /workspace/submission"),
        "base": (manifest, {"schema_version": 1, "base_revision": BASE_REVISION}, "base/schema mismatch", "use the fixed base"),
        "adapter_hash": (adapter, "SHA256 must match the manifest", "adapter hash mismatch", "flush the adapter and regenerate the manifest"),
        "config_hash": (config, "SHA256 must match the manifest", "adapter config hash mismatch", "use the fixed selector"),
        "lora": (manifest, CANONICAL_LORA, "LoRA schema mismatch", "use the fixed trainer"),
        "config_parse": (config, "JSON object", "adapter config is not valid bounded JSON", "use the fixed trainer"),
        "adapter_config_type": (config, "object", "adapter config root is not an object", "use the fixed trainer or baseline materializer"),
        "adapter_config_fields": (config, sorted(EXPECTED_CONFIG_FIELDS), "adapter config fields differ from the task-owned PEFT schema", "remove schema extensions and use the fixed trainer"),
        "adapter_config_field": (config, "fixed PEFT configuration", "adapter config field differs", "use the fixed trainer; alternate scaling, ranks, layers, biases, and saved modules are prohibited"),
        "adapter_targets": (config, sorted(TARGETS), "adapter target modules differ", "use exactly the fixed seven-target schema"),
        "path": (submission, "relative path", "artifact path must be relative", "use the fixed selector"),
        "path_escape": (submission, submission, "artifact must remain under submission or be the canonical baseline link", "remove traversal and noncanonical external symlinks"),
        "missing_file": (submission, True, "artifact must be a closed regular file or canonical baseline link", "copy the adapter into submission or rerun Solution"),
        "size": (submission, 2 * 1024**3, "artifact exceeds bound", "use only the fixed adapter"),
        "tensor_key": (adapter, "one fixed A/B projection key per target in each of 32 layers", "adapter tensor key differs from the fixed inventory", "use the fixed trainer"),
        "tensor": (adapter, {"shape": "fixed projection shape", "dtype": "torch.float32", "finite": True}, "adapter tensor shape, dtype, or values differ", "rerun the fixed trainer"),
        "baseline_tensor": (adapter, 0, "canonical baseline adapter is not all-zero", "rerun Solution"),
        "safetensors_parse": (adapter, "valid fixed-schema safetensors", "adapter is not a readable safetensors file", "rerun the fixed trainer"),
        "tensor_count": (adapter, 448, "adapter tensor count mismatch", "use all seven targets on 32 layers"),
        "training_fields": (manifest, sorted(training_fields), "training metadata fields differ", "use the fixed selector"),
        "training_budget": (manifest, fixed_budget, "fixed budget metadata mismatch", "select one completed fixed trial"),
        "training_digest": (manifest, "64 lowercase hexadecimal characters", "training digest is malformed", "use the fixed selector"),
        "adapter_load": (submission + "/adapter", "loadable fixed-schema adapter", "fixed PEFT loader rejected the candidate", "rerun the fixed trainer and selector"),
    }
    code = payload.get("code") if isinstance(payload, dict) else None
    if not isinstance(code, str) or code not in definitions:
        return {"status": "candidate_invalid", "code": "candidate_invalid", "path": submission,
                "condition": "candidate violates the fixed contract", "expected": "fixed-schema artifacts",
                "actual": "<redacted>", "hint": "use the fixed trainer and selector"}
    path, expected, condition, hint = definitions[code]
    actual = payload.get("actual")

    def scalar(value, strings=()):
        if value is None or type(value) is bool:
            return value
        if type(value) in (int, float) and abs(value) <= 2**40 and math.isfinite(value):
            return value
        if isinstance(value, str) and value in strings:
            return value
        return "<redacted>"

    def fields(value, allowed):
        return {key: scalar(value[key], TARGETS + ["FEATURE_EXTRACTION", "none", BASE_REVISION])
                for key in allowed if key in value} if isinstance(value, dict) else "<redacted>"

    safe_actual = scalar(actual, ("dict", "list", "str", "int", "float", "bool", "NoneType", "nonzero"))
    if code == "adapter_config_field":
        match = re.fullmatch(r"adapter config field ([a-z_]+) differs", payload.get("condition", ""))
        field = match.group(1) if match else None
        if field in EXPECTED_CONFIG_DEFAULTS:
            expected = EXPECTED_CONFIG_DEFAULTS[field]
            condition = f"adapter config field {field} differs"
            safe_actual = scalar(actual, (expected,) if isinstance(expected, str) else ())
            if isinstance(expected, dict) and isinstance(actual, dict):
                safe_actual = {"field_count": min(len(actual), 1000000)}
    elif code == "tensor":
        supplied = payload.get("expected")
        shape = supplied.get("shape") if isinstance(supplied, dict) else None
        if isinstance(shape, list) and any(shape == list(size) for pair in PROJECTION_SHAPES.values() for size in pair):
            expected = {"shape": shape, "dtype": "torch.float32", "finite": True}
        if isinstance(actual, dict):
            safe_actual = {}
            dims = actual.get("shape")
            if isinstance(dims, list) and len(dims) <= 8 and all(type(n) is int and 0 <= n <= 2**31 for n in dims):
                safe_actual["shape"] = dims
            dtype = actual.get("dtype")
            if dtype in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64", "torch.int32", "torch.int64"):
                safe_actual["dtype"] = dtype
            key = actual.get("key")
            if isinstance(key, str) and len(key) < 128 and re.fullmatch(r"base_model\.model\.layers\.(?:[0-9]|[12][0-9]|3[01])\.(?:self_attn|mlp)\.(?:q_proj|o_proj|v_proj|k_proj|w1|w2|w3)\.lora_[AB]\.weight", key):
                safe_actual["key"] = key
    elif code == "training_budget":
        safe_actual = fields(actual, fixed_budget)
    elif code in ("base", "lora"):
        safe_actual = fields(actual, expected)
    elif code in ("adapter_config_fields", "manifest_fields", "training_fields", "submission_inventory", "adapter_targets"):
        if code == "manifest_fields" and payload.get("expected") == sorted(baseline_fields):
            expected = sorted(baseline_fields)
        # Re-sanitizing a worker's already-safe field summary preserves only counts.
        if isinstance(actual, list):
            safe_actual = {"field_count": min(len(actual), 1000000), "known_fields": [s for s in expected if s in actual]}
        elif isinstance(actual, dict):
            safe_actual = {"field_count": scalar(actual.get("field_count")),
                           "known_fields": [s for s in expected if s in actual.get("known_fields", [])]}
    elif code in ("path", "path_escape", "missing_file", "size"):
        if payload.get("path") in (manifest, config, adapter, submission + "/adapter"):
            path = payload["path"]
        if code == "size" and path in (manifest, config):
            expected = 65536
    elif code == "adapter_load":
        safe_actual = "<redacted>"
    return {"status": "candidate_invalid", "code": code, "path": path, "condition": condition,
            "expected": expected, "actual": safe_actual, "hint": hint}


def require(condition: bool, code: str, path: Path, message: str, expected, actual, hint: str) -> None:
    if not condition:
        raise CandidateInvalid(code, path, message, expected, actual, hint)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_peft_config(document, path: Path) -> None:
    require(isinstance(document, dict), "adapter_config_type", path, "adapter config root is not an object", "object", type(document).__name__, "use the fixed trainer or baseline materializer")
    require(set(document) == EXPECTED_CONFIG_FIELDS, "adapter_config_fields", path, "adapter config fields differ from the task-owned PEFT 0.14 schema", sorted(EXPECTED_CONFIG_FIELDS), sorted(document), "remove every schema extension and use the fixed trainer")
    for field, expected in EXPECTED_CONFIG_DEFAULTS.items():
        actual = document[field]
        if field in {"r", "lora_alpha"}:
            matches = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
        elif field == "lora_dropout":
            matches = isinstance(actual, (int, float)) and not isinstance(actual, bool) and float(actual) == expected
        else:
            matches = actual == expected and type(actual) is type(expected)
        require(matches, "adapter_config_field", path, f"adapter config field {field} differs", expected, actual, "use the fixed trainer; alternate scaling, ranks, layers, biases, and saved modules are prohibited")
    targets = document["target_modules"]
    matches = isinstance(targets, list) and len(targets) == len(TARGETS) and set(targets) == set(TARGETS) and all(isinstance(item, str) for item in targets)
    require(matches, "adapter_targets", path, "adapter target modules differ", sorted(TARGETS), targets, "use exactly the fixed seven-target schema")


def expected_tensor(layer: int, target: str, side: str) -> tuple[str, tuple[int, int]]:
    parent = "self_attn" if target in {"q_proj", "o_proj", "v_proj", "k_proj"} else "mlp"
    key = f"base_model.model.layers.{layer}.{parent}.{target}.lora_{side}.weight"
    shape = PROJECTION_SHAPES[target][0 if side == "A" else 1]
    return key, shape


def parse_baseline(name: str) -> float:
    raw = os.environ.get(name, "")
    if not re.fullmatch(r"(?:0\.\d{5}|1\.00000)", raw):
        raise RuntimeError(f"operator input {name} must be a five-decimal value in [0,1]")
    value = float(raw)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise RuntimeError(f"operator input {name} is outside [0,1]")
    return value


def verify_hash_manifest(root: Path, manifest_path: Path, exact: bool) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest["files"]
    if exact:
        actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != manifest_path}
        if actual_names != set(files):
            raise RuntimeError(f"trusted asset inventory mismatch: missing={sorted(set(files)-actual_names)[:10]} added={sorted(actual_names-set(files))[:10]}")
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise RuntimeError(f"trusted file hash mismatch: {path}")


def verify_workspace() -> None:
    expected = json.loads((TESTS / "workspace-tree.json").read_text(encoding="utf-8"))
    actual: dict[str, str] = {}
    for path in sorted(WORKSPACE.rglob("*")):
        relative = path.relative_to(WORKSPACE)
        if relative.parts and relative.parts[0] in {"policy", "trials", "submission"}:
            continue
        if path.is_symlink():
            raise RuntimeError(f"immutable workspace contains symlink: {relative}")
        if path.is_file():
            actual[relative.as_posix()] = sha256(path)
    if expected != {"version": 1, "root": "/workspace", "files": actual}:
        missing = sorted(set(expected.get("files", {})) - set(actual))[:20]
        added = sorted(set(actual) - set(expected.get("files", {})))[:20]
        changed = sorted(k for k in set(actual) & set(expected.get("files", {})) if actual[k] != expected["files"][k])[:20]
        raise RuntimeError(f"immutable workspace changed: missing={missing} added={added} changed={changed}")
    allowed = {"policy", "trials", "submission"}
    unexpected = sorted(p.name for p in WORKSPACE.iterdir() if p.name not in allowed | {"reasonir"})
    if unexpected:
        raise RuntimeError(f"unexpected top-level workspace paths: {unexpected}")


def owned_file(relative: str, maximum: int, baseline: bool = False) -> Path:
    require(isinstance(relative, str) and relative and not relative.startswith("/"), "path", SUBMISSION, "artifact path must be relative", "relative path", relative, "use the fixed selector")
    path = SUBMISSION / relative
    resolved = path.resolve()
    zero_root = Path("/opt/reasonir-task/base/zero-adapter").resolve()
    within_submission = SUBMISSION.resolve() in resolved.parents
    canonical_zero = baseline and zero_root in resolved.parents
    require(within_submission or canonical_zero, "path_escape", path, "artifact must remain under submission or be the canonical baseline link", str(SUBMISSION), str(resolved), "remove traversal and noncanonical external symlinks")
    require(path.is_file() and (not path.is_symlink() or canonical_zero), "missing_file", path, "artifact must be a closed regular file or canonical baseline link", True, path.exists(), "copy the adapter into submission or rerun Solution")
    require(path.stat().st_size <= maximum, "size", path, "artifact exceeds bound", maximum, path.stat().st_size, "use only the fixed adapter")
    return path


def validate_adapter() -> tuple[Path, bool, str]:
    import torch
    manifest_path = SUBMISSION / "manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink() and manifest_path.stat().st_size <= 65536, "manifest", manifest_path, "manifest missing or too large", "regular JSON <=65536 bytes", manifest_path.exists(), "run the fixed selector or baseline Solution")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateInvalid("manifest_parse", manifest_path, "manifest is not valid bounded JSON", "JSON object", type(exc).__name__, "run the fixed selector or baseline Solution") from exc
    require(isinstance(manifest, dict), "manifest_type", manifest_path, "manifest root is not an object", "object", type(manifest).__name__, "run the fixed selector or baseline Solution")
    baseline = manifest.get("baseline") is True
    normal_fields = {"schema_version", "base_revision", "adapter_file", "adapter_sha256", "adapter_config_file", "adapter_config_sha256", "lora", "training"}
    baseline_fields = {"schema_version", "baseline", "base_revision", "adapter_file", "adapter_sha256", "adapter_config_file", "adapter_config_sha256", "lora"}
    require(set(manifest) == (baseline_fields if baseline else normal_fields), "manifest_fields", manifest_path, "manifest field set differs", sorted(baseline_fields if baseline else normal_fields), sorted(manifest), "use a fixed task materializer")
    expected_inventory = {"manifest.json", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"}
    actual_inventory = {path.relative_to(SUBMISSION).as_posix() for path in SUBMISSION.rglob("*") if path.is_file() or path.is_symlink()}
    require(actual_inventory == expected_inventory, "submission_inventory", SUBMISSION, "submission must contain only one adapter and manifest", sorted(expected_inventory), sorted(actual_inventory), "remove every non-ingestion file from /workspace/submission")
    require(manifest["schema_version"] == 1 and manifest["base_revision"] == BASE_REVISION, "base", manifest_path, "base/schema mismatch", {"schema_version": 1, "base_revision": BASE_REVISION}, {"schema_version": manifest.get("schema_version"), "base_revision": manifest.get("base_revision")}, "use the fixed base")
    adapter = owned_file(manifest["adapter_file"], 2 * 1024**3, baseline)
    config = owned_file(manifest["adapter_config_file"], 65536, baseline)
    require(sha256(adapter) == manifest["adapter_sha256"], "adapter_hash", adapter, "adapter hash mismatch", manifest["adapter_sha256"], sha256(adapter), "flush the adapter and regenerate the manifest")
    require(sha256(config) == manifest["adapter_config_sha256"], "config_hash", config, "adapter config hash mismatch", manifest["adapter_config_sha256"], sha256(config), "use the fixed selector")
    require(manifest["lora"] == CANONICAL_LORA, "lora", manifest_path, "LoRA schema mismatch", CANONICAL_LORA, manifest["lora"], "use the fixed trainer")
    try:
        adapter_config = json.loads(config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateInvalid("config_parse", config, "adapter config is not valid bounded JSON", "JSON object", type(exc).__name__, "use the fixed trainer") from exc
    validate_peft_config(adapter_config, config)
    tensor_count = 0
    tensor_identities = set()
    try:
        with safe_open(adapter, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                match = re.fullmatch(r"base_model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|o_proj|v_proj|k_proj|w1|w2|w3)\.lora_([AB])\.weight", key)
                require(match is not None, "tensor_key", adapter, "unexpected adapter tensor key", "fixed layer/target LoRA A or B key", key, "use the fixed trainer")
                layer, parent, target, side = int(match.group(1)), match.group(2), match.group(3), match.group(4)
                expected_key, expected_shape = expected_tensor(layer, target, side)
                require(key == expected_key and parent == ("self_attn" if target in {"q_proj", "o_proj", "v_proj", "k_proj"} else "mlp"), "tensor_key", adapter, "adapter tensor name differs from the fixed projection inventory", expected_key, key, "use the fixed trainer")
                identity = (layer, target, side)
                require(0 <= identity[0] < 32 and identity not in tensor_identities, "tensor_key", adapter, "duplicate or out-of-range adapter tensor key", "one A/B tensor per target in each of 32 layers", key, "use the fixed trainer")
                tensor_identities.add(identity)
                tensor = handle.get_tensor(key)
                actual = {"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                require(tuple(tensor.shape) == expected_shape and tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all()), "tensor", adapter, "adapter tensor shape, dtype, or values differ", {"shape": list(expected_shape), "dtype": "torch.float32", "finite": True}, actual, "rerun the fixed trainer")
                if baseline:
                    require(bool(torch.count_nonzero(tensor) == 0), "baseline_tensor", adapter, "canonical baseline adapter is not all-zero", 0, "nonzero", "rerun Solution")
                tensor_count += 1
    except CandidateInvalid:
        raise
    except Exception as exc:
        raise CandidateInvalid("safetensors_parse", adapter, "adapter is not a readable safetensors file", "valid fixed-schema safetensors", type(exc).__name__, "rerun the fixed trainer") from exc
    require(tensor_count == 448, "tensor_count", adapter, "adapter tensor count mismatch", 448, tensor_count, "use all seven targets on 32 layers")
    if not baseline:
        training = manifest["training"]
        fixed = {"updates": 1000, "exposures": 64000, "world_size": 4, "per_device_batch": 4, "gradient_accumulation": 4}
        training_fields = set(fixed) | {"schedule_sha256", "policy_sha256", "accounting_sha256"}
        require(isinstance(training, dict) and set(training) == training_fields, "training_fields", manifest_path, "training metadata fields differ", sorted(training_fields), sorted(training) if isinstance(training, dict) else type(training).__name__, "use the fixed selector")
        require(all(training.get(k) == v for k, v in fixed.items()), "training_budget", manifest_path, "fixed budget metadata mismatch", fixed, {k: training.get(k) for k in fixed}, "select one completed fixed trial")
        for field in ("schedule_sha256", "policy_sha256", "accounting_sha256"):
            require(isinstance(training[field], str) and re.fullmatch(r"[0-9a-f]{64}", training[field]) is not None, "training_digest", manifest_path, f"{field} is malformed", "64 lowercase hexadecimal characters", training[field], "use the fixed selector")
    return adapter.parent, baseline, manifest["adapter_sha256"]


def load_base_model():
    import torch
    from torch import nn
    from transformers import AutoModel
    class AliasedMLP(nn.Module):
        def __init__(self, original):
            super().__init__()
            self.w1, self.w2, self.w3 = original.gate_proj, original.down_proj, original.up_proj
            self.act_fn = original.act_fn
        def forward(self, hidden_states):
            return self.w2(self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states))
    model = AutoModel.from_pretrained(BASE, local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16)
    for layer in model.layers:
        layer.mlp = AliasedMLP(layer.mlp)
    return model


def load_candidate(model, adapter_dir: Path):
    from peft import LoraConfig, PeftModel
    try:
        canonical = LoraConfig(**CANONICAL_LORA)
        model = PeftModel.from_pretrained(model, adapter_dir, config=canonical, is_trainable=False)
    except Exception as exc:
        observed = f"{type(exc).__name__}: {exc}"[:500]
        raise CandidateInvalid("adapter_load", adapter_dir, "fixed PEFT loader rejected the candidate", "loadable fixed-schema adapter", observed, "rerun the fixed trainer and selector") from exc
    model.eval().to("cuda")
    return model


def encode(model, texts: list[str], instruction: str, max_length: int) -> np.ndarray:
    if isinstance(model, parallel.ParallelEncoder):
        return model.encode(texts, instruction=instruction, batch_size=1, max_length=max_length, convert_to_tensor=False)
    return protocol.encode_batched(model, texts, instruction, max_length, protocol.BATCH_SIZE)


def top_results(query: np.ndarray, documents: np.ndarray, query_ids: list[str], document_ids: list[str], excluded: dict[str, set[str]], ignore_identical: bool) -> dict[str, dict[str, float]]:
    import torch
    result: dict[str, dict[str, float]] = {}
    docs = torch.from_numpy(documents.astype(np.float32, copy=False))
    for offset, vector in enumerate(query):
        scores = torch.mv(docs, torch.from_numpy(vector.astype(np.float32, copy=False))).numpy()
        candidate_count = min(len(scores), 1000 + len(excluded.get(query_ids[offset], set())) + (1 if ignore_identical else 0))
        indexes = np.argpartition(scores, -candidate_count)[-candidate_count:]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]
        qid = query_ids[offset]
        ranked = {}
        for index in indexes:
            docid = str(document_ids[index])
            if docid in excluded.get(qid, set()) or (ignore_identical and docid == qid):
                continue
            ranked[docid] = float(scores[index])
            if len(ranked) == 1000:
                break
        result[qid] = ranked
    return result


def evaluate_bright(model, scratch: Path) -> dict[str, float]:
    import pytrec_eval

    values = {}
    for subject in SUBJECTS:
        examples = pq.read_table(TESTS / "assets" / "bright" / "examples" / f"{subject}.parquet").to_pylist()
        document_subject = "aops" if subject == "theoremqa_questions" else subject
        shards = sorted((TESTS / "assets" / "bright" / "documents").glob(f"{document_subject}-*.parquet"))
        if not shards:
            shards = [TESTS / "assets" / "bright" / "documents" / f"{document_subject}.parquet"]
        documents = []
        for shard in shards:
            documents.extend(pq.read_table(shard).to_pylist())
        protocol.validate_dataset_counts(subject, len(documents), len(examples))
        query_ids = [str(row["id"]) for row in examples]
        document_ids = [str(row["id"]) for row in documents]
        excluded = {str(row["id"]): {str(value) for value in row["excluded_ids"] if value != "N/A"} for row in examples}
        config = json.loads((TESTS / "bright-configs" / f"{subject}.json").read_text(encoding="utf-8"))
        task = TASK_NAMES.get(subject, subject)
        query_instruction = config["instructions"]["query"].format(task=task)
        document_instruction = config["instructions"]["document"]
        document_embeddings = encode(model, [row["content"] for row in documents], document_instruction, BRIGHT_MAX_LENGTH)
        query_embeddings = encode(model, [row["query"] for row in examples], query_instruction, BRIGHT_MAX_LENGTH)
        run = top_results(query_embeddings, document_embeddings, query_ids, document_ids, excluded, False)
        qrels = {str(row["id"]): {str(docid): 1 for docid in row["gold_ids"]} for row in examples}
        scores = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"}).evaluate(run)
        if set(scores) != set(qrels):
            raise RuntimeError(f"BRIGHT {subject} result cardinality mismatch")
        values[subject] = round(sum(row["ndcg_cut_10"] for row in scores.values()) / len(scores), 5)
        del document_embeddings, query_embeddings, documents
    return values


def evaluate_beir(model, scratch: Path) -> dict[str, float]:
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval

    values = {}
    for key, label in BEIR_DATASETS.items():
        archive = TESTS / "assets" / "beir" / f"{key}.zip"
        root = scratch / key
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                destination = (root / member.filename).resolve()
                if root.resolve() not in destination.parents and destination != root.resolve():
                    raise RuntimeError(f"unsafe archive member in {archive.name}")
            zipped.extractall(root)
        candidates = [path for path in root.rglob("qrels") if path.is_dir()]
        if len(candidates) != 1:
            raise RuntimeError(f"unexpected BEIR archive layout for {label}")
        dataset_root = candidates[0].parent
        corpus, queries, qrels = GenericDataLoader(str(dataset_root)).load(split="test")
        protocol.validate_dataset_counts(key, len(corpus), len(queries))
        query_ids = list(queries)
        document_ids = list(corpus)
        query_embeddings = encode(model, [queries[key] for key in query_ids], "", 2048)
        texts = [(corpus[key].get("title", "") + " " + corpus[key].get("text", "")).strip() for key in document_ids]
        document_embeddings = encode(model, texts, "", 2048)
        run = top_results(query_embeddings, document_embeddings, query_ids, document_ids, {}, True)
        ndcg, _, _, _ = EvaluateRetrieval.evaluate(qrels, run, [10], ignore_identical_ids=True)
        value = ndcg.get("NDCG@10")
        if value is None or not math.isfinite(float(value)):
            raise RuntimeError(f"BEIR {label} did not produce finite NDCG@10")
        values[label] = round(float(value), 5)
        del document_embeddings, query_embeddings, corpus, queries, qrels
        shutil.rmtree(root)
    return values


def judge(evaluator, log_directory, gpus, r0, g0, pool_factory=None):
    if not log_directory.is_dir() or any(log_directory.iterdir()):
        raise RuntimeError("evaluation requires fresh verifier logs")
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (r0, g0)):
        raise RuntimeError("invalid baseline inputs")
    started = time.monotonic()
    result = parallel.run_evaluation(evaluator, gpus, pool_factory=pool_factory)
    r, g = result["R"], result["G"]
    reward = r if g >= g0 else round(g - g0, 5)
    if not all(math.isfinite(value) for value in [r, g, reward]):
        raise RuntimeError("evaluation produced a non-finite aggregate")
    summary = {
        "status": "success",
        "artifact": {"status": "valid", "baseline": result["binding"]["baseline"], "sha256": result["binding"]["adapter_sha256"]},
        "reward": reward,
        "R": r,
        "G": g,
        "non_regression": g >= g0,
        "R_minus_R0": round(r - r0, 5),
        "G_minus_G0": round(g - g0, 5),
        "bright": result["bright"],
        "beir": result["beir"],
        "protocol": protocol.descriptor(),
        "worker_stats": result["worker_stats"],
        "total_metric_encodes": result["total_metric_encodes"],
        "runtime_sec": round(time.monotonic() - started, 3),
        "peak_gpu_memory_mb": result["peak_gpu_memory_mb"],
    }
    with (log_directory / "reward.json").open("x", encoding="utf-8") as stream:
        json.dump({"reward": reward}, stream)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    try:
        gpus = parallel.discover_gpu_uuids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        judge(sys.modules[__name__], Path("/logs/verifier"), gpus,
              parse_baseline("REASONIR_BASELINE_R0"), parse_baseline("REASONIR_BASELINE_G0"))
    except BaseException as exc:
        # Code-specific task-owned expectations and bounded metadata stay
        # actionable; uncontrolled paths/keys/strings never reach output.
        candidate_invalid = isinstance(exc, (CandidateInvalid, parallel.WorkerCandidateInvalid))
        failure = {"status": "candidate_invalid" if candidate_invalid else "evaluation_failed",
                   "error": type(exc).__name__, "protocol": protocol.PROTOCOL_ID}
        if candidate_invalid:
            failure.update(safe_candidate_diagnostic(exc.payload))
        elif isinstance(exc, parallel.DeviceAuthorizationError):
            failure.update({"code": "gpu_visibility_order", "hint": parallel.DeviceAuthorizationError.hint})
        print(json.dumps(failure, sort_keys=True), flush=True)
        raise SystemExit(2 if candidate_invalid else 1)


if __name__ == "__main__":
    main()
