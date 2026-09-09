#!/usr/bin/env python3
"""FD-DINOv2 reward for gpic-generation (lower FD is better).

r      = B_FD / fd_dinov2                # = 1 at the sealed baseline
reward = 0                               if fd invalid/non-finite or r <= R_FLOOR
reward = clip((r - 1)/(R_REF - 1), 0, 1) otherwise

PROVISIONAL ANCHORS: no FD-DINOv2 number exists for the pinned PixelGen
GPIC-full config sampled at guidance 1.0 — the repo publishes none, and the
authors' own sampling run name encodes cfg 1.75. The task authors run the
pristine config once (1 epoch, 1x8 H100), sample the 50k captions at
guidance 1.0 / 50 Euler steps, and evaluate with the pinned gpic_eval; the
sealed run replaces every constant. GPIC_R_FLOOR starts at 1.0 (any
improvement scores) and may be raised above the measured regeneration-noise
band at sealing; GPIC_R_REF = 1.25 means a 20% FD reduction saturates the
reward. Env overrides GPIC_B_FD / GPIC_R_REF / GPIC_R_FLOOR are honored for
that purpose.
"""
import json
import math
import os
from pathlib import Path

B_FD = float(os.environ.get("GPIC_B_FD", "200.0"))
R_REF = float(os.environ.get("GPIC_R_REF", "1.25"))
R_FLOOR = float(os.environ.get("GPIC_R_FLOOR", "1.0"))
VERIFIER_LOG = Path(os.environ.get("GPIC_VERIFIER_LOG_DIR", "/logs/verifier"))

metrics = json.loads((VERIFIER_LOG / "metrics.json").read_text())
if metrics.get("smoke"):
    raise SystemExit("smoke-mode metrics must not be scored")

fd = float(metrics["fd_dinov2"])
valid = math.isfinite(fd) and fd > 0
r = B_FD / fd if valid else float("nan")
if not math.isfinite(r) or r <= R_FLOOR:
    reward = 0.0
else:
    reward = max(0.0, min(1.0, (r - 1.0) / (R_REF - 1.0)))
result = {
    "reward": round(reward, 6),
    "policy_gate": 1,
    "metric": "fd_dinov2",
    "fd_dinov2": fd if valid else None,
    "r": round(r, 6) if math.isfinite(r) else None,
    "r_ref": R_REF,
    "r_floor": R_FLOOR,
    "baseline_fd": B_FD,
    "anchors_status": "provisional_values_pending_author_reproduction",
    **metrics,
}
(VERIFIER_LOG / "reward.json").write_text(json.dumps(result, indent=2) + "\n")
(VERIFIER_LOG / "reward.txt").write_text(f"{reward:.6f}\n")
print(json.dumps(result, indent=2))
