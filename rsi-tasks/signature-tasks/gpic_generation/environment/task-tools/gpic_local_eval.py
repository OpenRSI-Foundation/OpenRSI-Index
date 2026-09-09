#!/usr/bin/env python3
"""Agent-side screening FD (not the frozen verifier).

Computes FD-DINOv2 of a directory of generated PNGs against the VAL-split
reference statistics (``GPIC_VAL_STATS``) with the pinned gpic_eval toolkit.
The val reference is disjoint from the sealed test reference the frozen
verifier uses, so this is an honest screening proxy that cannot be
overfit into the scored number. GPU time counts against the attempt's
budget — record it in the ledger.

The DINO ban (policy RH-004) still applies: this tool exists for run
triage; wiring its output (or any DINO feature) into a gradient, data
filter, or automated search loop is a hard-zero violation.

Usage:
  python /task-tools/gpic_local_eval.py --images <dir> [--limit N]
      [--batch-size 256] [--num-workers 8] [--output out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

VAL_STATS = os.environ.get("GPIC_VAL_STATS", "/datasets/gpic/val_stats.npz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="directory of generated PNGs")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N images (sorted); FD from "
                             "small N is noisy — 50k is the toolkit recommendation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    images = Path(args.images).resolve()
    if not images.is_dir():
        print(f"error: not a directory: {images}", file=sys.stderr)
        return 1
    if not Path(VAL_STATS).exists():
        print(f"error: val reference stats missing: {VAL_STATS}", file=sys.stderr)
        return 1

    sample_path = images
    if args.limit:
        names = sorted(p for p in images.iterdir() if p.is_file())[:args.limit]
        subset = Path(tempfile.mkdtemp(prefix="gpic-screen-"))
        for path in names:
            (subset / path.name).symlink_to(path)
        sample_path = subset

    from gpic_eval.eval import eval_with_ref_stats

    result = eval_with_ref_stats(
        str(sample_path), VAL_STATS, models=["dino"], metrics=["fd"],
        batch_size=args.batch_size, num_workers=args.num_workers,
        device="cuda:0")
    summary = {
        "fd_dinov2_val": float(result.fd["dino"]),
        "reference": VAL_STATS,
        "images": str(images),
        "n_images": args.limit or sum(1 for p in images.iterdir() if p.is_file()),
        "note": "screening proxy vs VAL stats; the scored metric is FD vs the sealed TEST stats",
    }
    output = Path(args.output) if args.output else images.parent / "screening_fd.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
