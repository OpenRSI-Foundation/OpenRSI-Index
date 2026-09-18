#!/usr/bin/env python3
"""Read-only public validation of candidate-owned structural invariants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from safetensors import safe_open

ROOT = Path("/workspace/submission")
BASE = Path("/opt/reasonir-task/base/ReasonIR-8B")
BASE_REVISION = "c3d0690370ff4a8c3d3882d8dfa85c43650034fa"
TARGETS = {"q_proj", "o_proj", "v_proj", "k_proj", "w1", "w2", "w3"}
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


class CandidateInvalid(Exception):
    def __init__(self, code: str, path: Path, condition: str, expected, actual, hint: str) -> None:
        super().__init__(condition)
        self.payload = {
            "status": "candidate_invalid",
            "code": code,
            "path": str(path),
            "condition": condition,
            "expected": expected,
            "actual": actual,
            "hint": hint,
        }


def require(condition: bool, code: str, path: Path, message: str, expected, actual, hint: str) -> None:
    if not condition:
        raise CandidateInvalid(code, path, message, expected, actual, hint)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_peft_config(document, path: Path) -> None:
    require(isinstance(document, dict), "adapter_config_type", path, "adapter config root is not an object", "object", type(document).__name__, "use the fixed trainer")
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
    matches = isinstance(targets, list) and len(targets) == len(TARGETS) and set(targets) == TARGETS and all(isinstance(item, str) for item in targets)
    require(matches, "adapter_targets", path, "adapter target modules differ", sorted(TARGETS), targets, "use exactly the fixed seven-target schema")


def expected_tensor(layer: int, target: str, side: str) -> tuple[str, tuple[int, int]]:
    parent = "self_attn" if target in {"q_proj", "o_proj", "v_proj", "k_proj"} else "mlp"
    key = f"base_model.model.layers.{layer}.{parent}.{target}.lora_{side}.weight"
    shape = PROJECTION_SHAPES[target][0 if side == "A" else 1]
    return key, shape


def resolve_owned(relative: str, maximum: int) -> Path:
    require(isinstance(relative, str) and relative and not relative.startswith("/"), "path", ROOT, "artifact path must be relative", "relative path", relative, "use a path below /workspace/submission")
    path = (ROOT / relative).resolve()
    require(ROOT.resolve() in path.parents, "path_escape", ROOT / relative, "artifact path must remain in submission", str(ROOT), str(path), "remove traversal or symlinks")
    require(path.is_file() and not path.is_symlink(), "missing_file", path, "artifact must be a regular file", True, path.exists(), "materialize a closed regular file")
    require(path.stat().st_size <= maximum, "file_size", path, "artifact exceeds size bound", maximum, path.stat().st_size, "write only the fixed adapter artifact")
    return path


def validate() -> dict:
    manifest_path = ROOT / "manifest.json"
    require(manifest_path.is_file() and manifest_path.stat().st_size <= 64 * 1024, "manifest", manifest_path, "manifest missing or too large", "regular file <=65536 bytes", manifest_path.exists(), "select one complete trial with the fixed tool")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateInvalid("manifest_parse", manifest_path, "manifest is not valid bounded JSON", "JSON object", type(exc).__name__, "use the fixed selector output") from exc
    require(isinstance(manifest, dict), "manifest_type", manifest_path, "manifest root is not an object", "object", type(manifest).__name__, "use the fixed selector output")
    expected_top = {"schema_version", "base_revision", "adapter_file", "adapter_sha256", "adapter_config_file", "adapter_config_sha256", "lora", "training"}
    require(set(manifest) == expected_top, "manifest_fields", manifest_path, "manifest fields differ", sorted(expected_top), sorted(manifest), "use the fixed selector output")
    expected_inventory = {"manifest.json", "adapter/adapter_model.safetensors", "adapter/adapter_config.json"}
    actual_inventory = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file() or path.is_symlink()}
    require(actual_inventory == expected_inventory, "submission_inventory", ROOT, "submission must contain only one adapter and manifest", sorted(expected_inventory), sorted(actual_inventory), "run the fixed selector")
    require(manifest["schema_version"] == 1 and manifest["base_revision"] == BASE_REVISION, "base_revision", manifest_path, "schema/base revision mismatch", {"schema_version": 1, "base_revision": BASE_REVISION}, {"schema_version": manifest.get("schema_version"), "base_revision": manifest.get("base_revision")}, "train against the fixed base")
    adapter = resolve_owned(manifest["adapter_file"], 2 * 1024**3)
    config = resolve_owned(manifest["adapter_config_file"], 64 * 1024)
    for path, field in [(adapter, "adapter_sha256"), (config, "adapter_config_sha256")]:
        actual = sha256(path)
        require(actual == manifest[field], "hash", path, "artifact hash mismatch", manifest[field], actual, "close the artifact and regenerate the manifest")
    expected_lora = {"r": 16, "lora_alpha": 64, "lora_dropout": 0.1, "inference_mode": False, "target_modules": ["q_proj", "o_proj", "v_proj", "k_proj", "w1", "w2", "w3"], "bias": "none", "task_type": "FEATURE_EXTRACTION"}
    require(manifest["lora"] == expected_lora, "lora_schema", manifest_path, "LoRA schema mismatch", expected_lora, manifest["lora"], "use the fixed trainer")
    training = manifest["training"]
    fixed_training = {"updates": 1000, "exposures": 64000, "world_size": 4, "per_device_batch": 4, "gradient_accumulation": 4}
    training_fields = set(fixed_training) | {"schedule_sha256", "policy_sha256", "accounting_sha256"}
    require(isinstance(training, dict) and set(training) == training_fields, "training_fields", manifest_path, "training metadata fields differ", sorted(training_fields), sorted(training) if isinstance(training, dict) else type(training).__name__, "use the fixed selector output")
    require(all(training.get(k) == v for k, v in fixed_training.items()), "training_budget", manifest_path, "training budget metadata mismatch", fixed_training, {k: training.get(k) for k in fixed_training}, "use a completed fixed-trainer trial")
    for field in ("schedule_sha256", "policy_sha256", "accounting_sha256"):
        require(isinstance(training[field], str) and re.fullmatch(r"[0-9a-f]{64}", training[field]) is not None, "training_digest", manifest_path, f"{field} is malformed", "64 lowercase hexadecimal characters", training[field], "use the fixed selector output")
    try:
        adapter_config = json.loads(config.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CandidateInvalid("config_parse", config, "adapter config is not valid bounded JSON", "JSON object", type(exc).__name__, "use the fixed trainer") from exc
    validate_peft_config(adapter_config, config)
    tensor_count = 0
    identities = set()
    try:
        with safe_open(adapter, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                match = re.fullmatch(r"base_model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(q_proj|o_proj|v_proj|k_proj|w1|w2|w3)\.lora_([AB])\.weight", key)
                require(match is not None, "tensor_key", adapter, "unexpected adapter tensor key", "fixed layer/target LoRA A or B key", key, "use the fixed trainer")
                layer, parent, target, side = int(match.group(1)), match.group(2), match.group(3), match.group(4)
                expected_key, expected_shape = expected_tensor(layer, target, side)
                require(key == expected_key and parent == ("self_attn" if target in {"q_proj", "o_proj", "v_proj", "k_proj"} else "mlp"), "tensor_key", adapter, "adapter tensor name differs from the fixed projection inventory", expected_key, key, "use the fixed trainer")
                identity = (layer, target, side)
                require(0 <= identity[0] < 32 and identity not in identities, "tensor_key", adapter, "duplicate or out-of-range adapter tensor key", "one A/B tensor per target in each of 32 layers", key, "use the fixed trainer")
                identities.add(identity)
                tensor = handle.get_tensor(key)
                actual = {"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
                require(tuple(tensor.shape) == expected_shape and str(tensor.dtype) == "torch.float32" and torch_isfinite(tensor), "tensor", adapter, "adapter tensor shape, dtype, or values differ", {"shape": list(expected_shape), "dtype": "torch.float32", "finite": True}, actual, "rerun the fixed trainer")
                tensor_count += 1
    except CandidateInvalid:
        raise
    except Exception as exc:
        raise CandidateInvalid("safetensors_parse", adapter, "adapter is not a readable safetensors file", "valid fixed-schema safetensors", type(exc).__name__, "rerun the fixed trainer") from exc
    require(tensor_count == 32 * 7 * 2, "tensor_count", adapter, "adapter tensor count mismatch", 448, tensor_count, "use the fixed seven-target schema for all 32 layers")
    return {"status": "valid", "adapter_sha256": manifest["adapter_sha256"], "updates": 1000, "exposures": 64000, "tensor_count": tensor_count}


def torch_isfinite(tensor) -> bool:
    import torch
    return bool(torch.isfinite(tensor).all().item())


def main() -> None:
    try:
        result = validate()
    except CandidateInvalid as exc:
        print(json.dumps(exc.payload, sort_keys=True), flush=True)
        raise SystemExit(2)
    except Exception as exc:
        print(json.dumps({"status": "validator_error", "type": type(exc).__name__, "message": str(exc)}), flush=True)
        raise SystemExit(3)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
