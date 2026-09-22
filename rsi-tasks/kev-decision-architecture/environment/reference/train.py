"""Train the editable Kev implementation from the fixed Qwen base in Work.

This reference launcher never resumes a run or loads a released Kev adapter.
The resulting manifest describes self-reported training; it is not attestation.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
BASE_DIR = Path("/opt/kev-assets/base")
TRAIN_DIR = Path("/opt/kev-assets/train")
TRAIN_SHA256 = "7ed5254b5cb5291baefaceb09edf7e13110258211518c8038f4a12c11bd628ad"
TRAIN_MANIFEST_SHA256 = "a8f50e481b7d90b97da049e0ff6a01cee2f1ed204aed61a8265af0edbb5514d2"
TRAIN_RECORDS = 12576


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", default="/workspace/candidate/checkpoint")
    result.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    result.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    result.add_argument("--epochs", type=int, default=2)
    result.add_argument("--batch", type=int, default=4, help="records per GPU per forward pass")
    result.add_argument("--accum", type=int, default=1, help="microbatches per optimizer step; default global batch is 2 GPUs x 4 = 8")
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--lora", type=int, default=16)
    result.add_argument("--head-dim", type=int, default=256)
    result.add_argument("--checkpointing", type=int, choices=(0, 1), default=1)
    result.add_argument("--dry-run", action="store_true", help="print the command without loading assets or training")
    return result


def training_command(args, output):
    if min(args.epochs, args.batch, args.accum, args.lora, args.head_dim) < 1:
        raise ValueError("epochs, batch, accum, lora, and head-dim must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    if args.dtype == "bf16" and args.device != "cuda":
        raise ValueError("bf16 training requires CUDA; use --dtype fp32 for CPU")
    launcher = ([sys.executable, "-m", "torch.distributed.run", "--standalone",
                 "--nnodes=1", "--nproc-per-node=2", "--module", "kev.train"]
                if args.device == "cuda" else [sys.executable, "-m", "kev.train"])
    return launcher + [
        "--base", str(BASE_DIR), "--base_revision", BASE_REVISION,
        "--suite", str(TRAIN_DIR), "--out", str(output),
        "--epochs", str(args.epochs), "--batch", str(args.batch),
        "--accum", str(args.accum), "--lr", str(args.lr),
        "--seed", str(args.seed), "--lora", str(args.lora),
        "--head_dim", str(args.head_dim), "--checkpointing", str(args.checkpointing),
        "--device", args.device, "--dtype", args.dtype,
        "--weights_dtype", "fp32",
    ]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_training_file(path, expected_hash, expected_records):
    if sha256(path) != expected_hash:
        raise ValueError("fixed training data checksum mismatch")
    with Path(path).open() as stream:
        records = sum(1 for line in stream if line.strip())
    if records != expected_records:
        raise ValueError("fixed training data record count mismatch")


def verify_assets():
    if sha256(TRAIN_DIR / "manifest.json") != TRAIN_MANIFEST_SHA256:
        raise ValueError("fixed training manifest checksum mismatch")
    verify_training_file(TRAIN_DIR / "train.jsonl", TRAIN_SHA256, TRAIN_RECORDS)
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        if not (BASE_DIR / name).is_file():
            raise FileNotFoundError(f"fixed base asset missing: {BASE_DIR / name}")


def finalize_checkpoint(checkpoint, args):
    checkpoint = Path(checkpoint)
    for name in ("head.pt", "adapter_model.safetensors", "adapter_config.json",
                 "training_metrics.json", "training_config.json"):
        path = checkpoint / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"incomplete reference checkpoint: {name}")
    metrics = json.loads((checkpoint / "training_metrics.json").read_text())
    expected_world_size = 2 if args.device == "cuda" else 1
    if metrics.get("world_size") != expected_world_size:
        raise ValueError(f"training world_size must be {expected_world_size}; got {metrics.get('world_size')!r}")
    expected = TRAIN_RECORDS * args.epochs
    if (metrics.get("requested_records") != expected or metrics.get("records_seen") != expected
            or metrics.get("rejected_records") != 0 or metrics.get("truncated_records") != 0):
        raise ValueError("training did not account for every fixed record in every epoch")
    files = []
    for path in sorted(checkpoint.rglob("*")):
        if path.is_symlink():
            raise ValueError("checkpoint files may not be symlinks")
        # safetensors writes 0600 files; Judge loads the checkpoint as an unprivileged user.
        path.chmod(0o755 if path.is_dir() else 0o644)
        if path.is_file() and path.name not in ("manifest.json", ".manifest.tmp"):
            files.append(path.relative_to(checkpoint).as_posix())
    checkpoint.chmod(0o755)
    manifest = {
        "schema_version": 1, "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "training_suite": "decision-v7", "training_records": TRAIN_RECORDS,
        "train_data_sha256": TRAIN_SHA256, "checkpoint_files": files,
        "training": {"provenance": "self-reported", "initialization": "original Qwen base and random pointer head",
                     "hyperparameters": {key: value for key, value in vars(args).items() if key not in ("out", "dry_run")},
                     "metrics": metrics},
    }
    temporary = checkpoint / ".manifest.tmp"
    with temporary.open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(checkpoint / "manifest.json")


def train(args):
    output = Path(args.out).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"checkpoint already exists: {output}; use a fresh --out path")
    training_command(args, output)  # reject invalid options before creating files
    verify_assets()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-training-", dir=output.parent))
    pending = staging / "checkpoint"  # upstream requires an absent output directory
    command = training_command(args, pending)
    candidate = Path(__file__).resolve().parent
    environment = dict(os.environ, PYTHONPATH=str(candidate), HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONUNBUFFERED="1")
    started = time.monotonic()
    try:
        subprocess.run(command, cwd=candidate, env=environment, check=True)
        finalize_checkpoint(pending, args)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during training: {output}")
        pending.rename(output)
        staging.rmdir()
    except BaseException:
        print(f"Training did not publish a checkpoint. Partial files, if any: {staging}", file=sys.stderr)
        raise
    print(f"Completed checkpoint: {output} ({time.monotonic() - started:.1f}s). Judge may now evaluate it.")


if __name__ == "__main__":
    options = parser().parse_args()
    if options.dry_run:
        print(json.dumps(training_command(options, Path(options.out)), indent=2))
    else:
        train(options)
