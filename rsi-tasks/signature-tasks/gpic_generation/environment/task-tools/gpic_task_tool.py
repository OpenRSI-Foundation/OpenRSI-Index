#!/usr/bin/env python3
"""Frozen task tool for the gpic-generation task.

Subcommands:
  summarize --attempt <dir>       real-evidence summary of one attempt
  stage --checkpoint <ckpt> --images <dir_of_50k_pngs>
        [--generate-cmd "<recorded generate.py invocation>"]
                                  stage exactly one submission: copies the
                                  checkpoint, exports tensor-only
                                  model_state.pt (its sha256 seeds the
                                  verifier's hidden spot subset), validates
                                  the image set against the frozen caption
                                  jsonl, and links it under
                                  /app/output/submission/images
  audit --output-root <dir>       pre-verifier self-audit of required outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

OUTPUT_ROOT_DEFAULT = "/app/output"
EVAL_JSONL = Path(os.environ.get("GPIC_EVAL_JSONL", "/datasets/gpic/gpic_eval_50k.jsonl"))
IMAGES_CEILING = int(os.environ.get("GPIC_IMAGES_CEILING", "100000000"))
ALLOWLIST = {entry.strip() for entry in
             os.environ.get("GPIC_PRETRAINED_ALLOWLIST", "Qwen/Qwen3-1.7B").split(",")
             if entry.strip()}


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return "(missing)"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_names() -> list[str]:
    names = []
    with EVAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                names.append(f"{row['caption_type']}_{row['key']}.png")
    return names


def cmd_summarize(args: argparse.Namespace) -> int:
    attempt = Path(args.attempt)
    status_path = attempt / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    print(json.dumps({
        "attempt": attempt.name,
        "state": status.get("state", "unknown"),
        "run_kind": status.get("run_kind"),
        "charged_gpu_hours": status.get("charged_gpu_hours"),
        "gpu_hours_cap": status.get("gpu_hours_cap"),
        "cumulative_read_bytes": status.get("cumulative_read_bytes"),
        "data_overread_flag": status.get("data_overread_flag"),
        "start_utc": status.get("start_utc"),
        "end_utc": status.get("end_utc"),
        "checkpoints": sorted(str(p.relative_to(attempt))
                              for suffix in ("*.ckpt", "*.pt", "*.pth", "*.safetensors")
                              for p in attempt.rglob(suffix)),
    }, indent=2))
    print("--- last training log lines ---")
    print(tail(attempt / "logs" / "train.log"))
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    checkpoint = Path(args.checkpoint)
    images_dir = Path(args.images)
    if not checkpoint.exists():
        print(f"error: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1
    if not images_dir.is_dir():
        print(f"error: images dir not found: {images_dir}", file=sys.stderr)
        return 1

    names = expected_names()
    missing = [n for n in names if not (images_dir / n).exists()]
    if missing:
        print(f"error: images dir missing {len(missing)} of {len(names)} "
              f"expected files (e.g. {missing[:3]})", file=sys.stderr)
        return 1
    extra = {p.name for p in images_dir.iterdir() if p.is_file()} - set(names)
    if extra:
        print(f"error: images dir has {len(extra)} files not in the caption "
              f"set (e.g. {sorted(extra)[:3]})", file=sys.stderr)
        return 1

    import torch  # local import: heavy

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "ema"):
            if key in payload and isinstance(payload[key], dict):
                state = payload[key]
                break
    tensors = {k: v for k, v in state.items() if hasattr(v, "dtype")}
    if not tensors:
        print("error: could not extract a tensor state dict from the checkpoint",
              file=sys.stderr)
        return 1
    parameters = sum(int(v.numel()) for v in tensors.values())

    submission = output_root / "submission"
    submission.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, submission / "model.ckpt")
    torch.save(tensors, output_root / "model_state.pt")
    model_state_sha = sha256_file(output_root / "model_state.pt")

    staged_images = submission / "images"
    if staged_images.exists():
        shutil.rmtree(staged_images)
    staged_images.mkdir()
    # Copy (not hardlink): the verifier requires each submitted image to be a
    # unique regular file (nlink==1) so a staged dataset image cannot be
    # aliased in as a "generation".
    for name in names:
        shutil.copy2((images_dir / name).resolve(), staged_images / name)

    selection = {
        "source_checkpoint": str(checkpoint),
        "source_images": str(images_dir),
        "generate_cmd": args.generate_cmd,
        "parameters": parameters,
        "n_images": len(names),
        "model_state_sha256": model_state_sha,
        "eval_jsonl": str(EVAL_JSONL),
        "staged": {
            "checkpoint": str(submission / "model.ckpt"),
            "model_state": str(output_root / "model_state.pt"),
            "images": str(staged_images),
        },
    }
    (output_root / "submission-selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in selection.items() if k != "staged"}, indent=2))
    print("Staged. Now write provenance.json (sampling.guidance must be 1.0 and "
          "sampling.seed must reproduce these images via /app/project/generate.py), "
          "then run: gpic_task_tool.py audit")
    return 0


REQUIRED_PROVENANCE = [
    "model_description", "parameters", "training_data", "shards_consumed",
    "images_seen", "resume_events", "gpu_hours", "training_command",
    "sampling", "pretrained_weights", "generation_command",
    "upstream_commits", "evaluation_commands", "optimization_attempts",
    "web_search",
]


def cmd_audit(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    problems: list[str] = []

    selection_path = output_root / "submission-selection.json"
    checkpoint = output_root / "submission" / "model.ckpt"
    model_state = output_root / "model_state.pt"
    images = output_root / "submission" / "images"
    generate = Path("/app/project/generate.py")
    for path in (selection_path, checkpoint, model_state, images, generate):
        if not path.exists():
            problems.append(f"missing required artifact: {path}")

    provenance_path = output_root / "provenance.json"
    provenance = {}
    if not provenance_path.exists():
        problems.append("missing provenance.json")
    else:
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"provenance.json is not valid JSON: {exc}")
        for field in REQUIRED_PROVENANCE:
            if provenance.get(field) in (None, "", []):
                problems.append(f"provenance missing field: {field}")
        sampling = provenance.get("sampling")
        if isinstance(sampling, dict):
            if float(sampling.get("guidance", -1)) != 1.0:
                problems.append("provenance.sampling.guidance must be 1.0 "
                                "(pure conditional sampling)")
            if "seed" not in sampling:
                problems.append("provenance.sampling.seed missing")
        else:
            problems.append("provenance.sampling must be a dict with guidance/sampler/steps/seed")
        try:
            images_seen = int(provenance.get("images_seen"))
            if images_seen > IMAGES_CEILING * 1.005:
                problems.append(
                    f"images_seen {images_seen} exceeds the one-epoch ceiling {IMAGES_CEILING}")
        except (TypeError, ValueError):
            problems.append("provenance.images_seen must be an integer")
        weights = provenance.get("pretrained_weights")
        if isinstance(weights, list):
            for entry in weights:
                name = entry.get("hf_id") or entry.get("name") if isinstance(entry, dict) else entry
                if name not in ALLOWLIST:
                    problems.append(f"pretrained weight off the allowlist: {name!r}")
        elif weights not in ("none", "None"):
            problems.append("provenance.pretrained_weights must be 'none' or a "
                            "list of allowlisted entries with source id and hash")

    ledger_path = output_root / "experiments.jsonl"
    selected_rows = 0
    ledger_gpu_hours = 0.0
    if not ledger_path.exists():
        problems.append("missing experiments.jsonl")
    else:
        for lineno, line in enumerate(
                ledger_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"experiments.jsonl line {lineno}: invalid JSON")
                continue
            if row.get("status") == "selected":
                selected_rows += 1
            try:
                ledger_gpu_hours += float(row.get("gpu_hours", 0.0))
            except (TypeError, ValueError):
                problems.append(f"experiments.jsonl line {lineno}: bad gpu_hours")
        if selected_rows != 1:
            problems.append(
                f"experiments.jsonl must contain exactly one 'selected' row, found {selected_rows}")

    measured_gpu_hours = 0.0
    attempts_dir = output_root / "attempts"
    if attempts_dir.exists():
        for status_path in attempts_dir.glob("*/status.json"):
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                measured_gpu_hours += float(status.get("charged_gpu_hours", 0.0))
                if status.get("data_overread_flag"):
                    problems.append(
                        f"attempt {status_path.parent.name} carries the trusted "
                        "data_overread_flag (read more bytes than one training-set "
                        "pass allows); the verifier will fail it")
            except (json.JSONDecodeError, TypeError, ValueError):
                problems.append(f"unreadable status: {status_path}")
    if ledger_path.exists() and attempts_dir.exists():
        if measured_gpu_hours > 0 and abs(ledger_gpu_hours - measured_gpu_hours) > \
                max(0.5, 0.05 * measured_gpu_hours):
            problems.append(
                "ledger GPU-hours ({:.2f}) diverge from measured status.json GPU-hours "
                "({:.2f}); reconcile before submitting".format(
                    ledger_gpu_hours, measured_gpu_hours))

    report = {
        "ok": not problems,
        "problems": problems,
        "ledger_gpu_hours": round(ledger_gpu_hours, 3),
        "measured_gpu_hours": round(measured_gpu_hours, 3),
    }
    (output_root / "audit-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_summarize = sub.add_parser("summarize")
    p_summarize.add_argument("--attempt", required=True)

    p_stage = sub.add_parser("stage")
    p_stage.add_argument("--checkpoint", required=True)
    p_stage.add_argument("--images", required=True,
                         help="directory holding the 50k generated PNGs")
    p_stage.add_argument("--generate-cmd", default="",
                         help="the exact generate.py invocation that produced --images")
    p_stage.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)

    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT)

    args = parser.parse_args()
    if args.cmd == "summarize":
        return cmd_summarize(args)
    if args.cmd == "stage":
        return cmd_stage(args)
    if args.cmd == "audit":
        return cmd_audit(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
