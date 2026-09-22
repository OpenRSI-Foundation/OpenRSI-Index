"""Build-time immutable asset staging; never trains or loads model weights."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
DATASET = "jaredpalmer/kev-suites"
DATA_REVISION = "a957287d1c502a4e2e3b9d9d1325c2c6f27f181c"
BASE_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
              "tokenizer_config.json", "merges.txt", "vocab.json", "LICENSE")
BASE_WEIGHTS_SHA256 = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
TRAIN_SHA256 = "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad"
MANIFEST_SHA256 = "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_training_context(path, tokenizer, encode, materialize):
    count = maximum = 0
    with Path(path).open() as stream:
        for line in stream:
            record = json.loads(line)
            identity = record.get("_meta", {}).get("id", f"row {count + 1}")
            try:
                encoded = encode(tokenizer, materialize(record), strict=True)
                if len(encoded["ids"]) > 2048:
                    raise ValueError("packed input exceeds 2048 tokens")
            except ValueError as error:
                raise ValueError(f"fixed training record {identity} does not fit Qwen2.5: {error}; no records were filtered") from error
            maximum = max(maximum, len(encoded["ids"]))
            count += 1
    return {"records": count, "maximum_packed_tokens": maximum, "filtered_records": 0}


def main():
    from huggingface_hub import hf_hub_download
    root = Path("/opt/kev-assets")
    base, train = root / "base", root / "train"
    base.mkdir(parents=True, exist_ok=True)
    train.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kev-build-downloads-") as cache:
        for filename in BASE_FILES:
            cached = hf_hub_download(BASE_MODEL, filename, revision=BASE_REVISION, cache_dir=cache)
            shutil.copyfile(cached, base / filename)
        for filename in ("manifest.json", "train.jsonl"):
            cached = hf_hub_download(DATASET, f"v7/decision-v7/{filename}", repo_type="dataset",
                                     revision=DATA_REVISION, cache_dir=cache)
            shutil.copyfile(cached, train / filename)
    for path, expected in ((base / "model.safetensors", BASE_WEIGHTS_SHA256),
                           (train / "manifest.json", MANIFEST_SHA256), (train / "train.jsonl", TRAIN_SHA256)):
        if digest(path) != expected:
            raise ValueError(f"immutable asset checksum mismatch: {path}")
    manifest = json.loads((train / "manifest.json").read_text())
    if manifest["files"]["train.jsonl"]["records"] != 12576:
        raise ValueError("unexpected fixed training count")
    # Tokenization-only feasibility check: no backbone construction, model execution,
    # augmentation, training, dropping, or truncation occurs during image preparation.
    from kev.model import encode, load_tokenizer
    from kev.data import materialize
    coverage = validate_training_context(train / "train.jsonl", load_tokenizer(str(base)), encode, materialize)
    if coverage["records"] != 12576:
        raise ValueError("training file does not contain all 12,576 records")
    report = {"base_model": BASE_MODEL, "base_revision": BASE_REVISION, "dataset": DATASET,
              "dataset_revision": DATA_REVISION, "training_context": coverage,
              "sha256": {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*")) if path.is_file()}}
    (root / "asset-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(coverage))


if __name__ == "__main__":
    main()
