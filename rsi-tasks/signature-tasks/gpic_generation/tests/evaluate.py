#!/usr/bin/env python3
"""Frozen FD-DINOv2 evaluator for gpic-generation.

Runs after the policy gate. Two-stage design:

1. PRIMARY METRIC on the submitted images (cheap, deterministic): the 50k
   PNGs staged under /app/output/submission/images are validated against
   the frozen caption set (exact filename set {caption_type}_{key}.png, all
   256x256 RGB), then scored with the pinned gpic_eval toolkit against the
   SEALED test reference statistics (never staged in the agent image).

2. SPOT-REGENERATION GATE binding the images to the submitted model: a
   hidden subset of GPIC_SPOT_N caption keys — chosen by a PRNG seeded from
   sha256(model_state.pt) combined with a verifier-only salt, so the subset
   is reproducible by the verifier yet unpredictable to the agent — is
   regenerated through the agent's frozen generation contract:

     python /app/project/generate.py --captions <jsonl> --out <dir> \
         --seed <provenance.sampling.seed> [--devices N]

   and compared per key in the metric's own feature space (DINOv2-vitl14
   embedding cosine, same transform as the FD pipeline). Submitting images
   that did not come from the staged model via this contract (dataset
   images, cherry-picks, guided sampling done off-line) fails the gate.
   Thresholds are calibrated for ODE-sampler floating-point noise across
   reruns and are re-sealed from the author baseline reproduction.

Provenance cross-checks (mechanical): sampling.guidance == 1.0,
images_seen <= ceiling, per-attempt charged GPU-hours <= cap, and no
attempt flagged for data over-read by the trusted watchdog.

Writes /app/output/verifier-metrics.json; score.py maps it to the reward.
Env knobs (task infrastructure): GPIC_VERIFIER_SMOKE=1 restricts to the
first 1024 captions and a 64-key spot check — infrastructure debugging
only; score.py refuses smoke metrics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Container defaults; env overrides exist for containerless infrastructure
# smoke testing only (the harness never sets them).
OUTPUT_ROOT = Path(os.environ.get("GPIC_OUTPUT_ROOT", "/app/output"))
CANDIDATE_ROOT = Path(os.environ.get("GPIC_CANDIDATE_ROOT", "/app/project"))
VERIFIER_LOG = Path(os.environ.get("GPIC_VERIFIER_LOG_DIR", "/logs/verifier"))
EVAL_JSONL = Path(os.environ.get("GPIC_EVAL_JSONL", "/opt/gpic-ref/gpic_eval_50k.jsonl"))
TEST_STATS = Path(os.environ.get("GPIC_TEST_STATS", "/opt/gpic-ref/test_stats.npz"))
IMAGES_CEILING = int(os.environ.get("GPIC_IMAGES_CEILING", "100000000"))
GPU_HOURS_CAP = float(os.environ.get("GPIC_GPU_HOURS_CAP", "1000.0"))
PRETRAINED_ALLOWLIST = {w.strip() for w in os.environ.get(
    "GPIC_PRETRAINED_ALLOWLIST", "Qwen/Qwen3-1.7B").split(",") if w.strip()}
EVAL_BATCH = int(os.environ.get("GPIC_EVAL_BATCH", "256"))
EVAL_WORKERS = int(os.environ.get("GPIC_EVAL_WORKERS", "8"))
SMOKE = os.environ.get("GPIC_VERIFIER_SMOKE", "0") == "1"
SPOT_N = int(os.environ.get("GPIC_SPOT_N", "64" if SMOKE else "2048"))
SPOT_COS = float(os.environ.get("GPIC_SPOT_COS", "0.98"))
SPOT_PASS_FRAC = float(os.environ.get("GPIC_SPOT_PASS_FRAC", "0.99"))
SPOT_MEDIAN_COS = float(os.environ.get("GPIC_SPOT_MEDIAN_COS", "0.995"))
SPOT_TIMEOUT = int(os.environ.get("GPIC_SPOT_TIMEOUT", "7200"))
# Positive binding: regeneration from a materially PERTURBED checkpoint must
# move the output. If the clean and perturbed regenerations are nearly
# identical (median DINOv2 cosine >= this), generate.py is ignoring the
# weights — i.e. copying/looking up images rather than sampling the model.
SPOT_BIND_MAX_COS = float(os.environ.get("GPIC_SPOT_BIND_MAX_COS", "0.999"))
SPOT_PERTURB_STD = float(os.environ.get("GPIC_SPOT_PERTURB_STD", "0.05"))
RESUME_REPLAY_TOLERANCE = 1.005  # 0.5% duplicate samples allowed across resumes
# Secrets the untrusted generate.py must never see (sealed reference, spot
# subset salt, all gate thresholds, reward anchors).
SECRET_ENV_PREFIXES = ("GPIC_SPOT", "GPIC_TEST_STATS", "GPIC_B_FD", "GPIC_R_",
                       "POLICY_JUDGE", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                       "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _child_env() -> dict:
    """Minimal env for the untrusted generate.py: keep only what CUDA/python
    need, drop every verifier secret and gate threshold."""
    keep = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_HOME",
            "NVIDIA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_HOME",
            "TRANSFORMERS_OFFLINE", "GPIC_QWEN_DIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env["PYTHONPATH"] = str(CANDIDATE_ROOT)
    return env


def fail(reason: str) -> int:
    print(f"verifier failure: {reason}", file=sys.stderr)
    VERIFIER_LOG.mkdir(parents=True, exist_ok=True)
    (VERIFIER_LOG / "evaluate-failure.json").write_text(
        json.dumps({"reason": reason}, indent=2) + "\n", encoding="utf-8")
    return 1


def load_captions() -> list[dict]:
    rows = []
    with EVAL_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if SMOKE:
        rows = rows[:1024]
    return rows


def expected_name(row: dict) -> str:
    # gpic convention (src/data/dataset/gpic_jsonl_dataset.py):
    # gpic_jsonl_save_fn writes f"{caption_type}_{key}.png"
    return f"{row['caption_type']}_{row['key']}.png"


def validate_images(images_dir: Path, rows: list[dict]) -> str | None:
    from PIL import Image

    expected = {expected_name(r) for r in rows}
    actual = {p.name for p in images_dir.iterdir() if p.is_file()}
    if SMOKE:
        # smoke submissions may carry only the smoke subset
        actual = {a for a in actual if a in expected}
    missing = expected - actual
    extra = actual - expected
    if missing:
        return f"images missing for {len(missing)} captions (e.g. {sorted(missing)[:3]})"
    if extra:
        return f"{len(extra)} images not in the caption set (e.g. {sorted(extra)[:3]})"
    for name in sorted(expected):
        path = images_dir / name
        # Submitted images must be genuine regular files produced here — not
        # symlinks or hardlinks aliasing a staged dataset image (which would
        # let real photographs masquerade as generations). The spot-gate
        # binding is the deep defense; this is the cheap first line.
        if path.is_symlink():
            return f"image {name} is a symlink; submissions must be regular files"
        try:
            st = path.stat()
        except OSError as exc:
            return f"image {name} is unstatable: {exc}"
        if st.st_nlink != 1:
            return f"image {name} is hardlinked (nlink={st.st_nlink}); submissions must be unique regular files"
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                if img.size != (256, 256):
                    return f"image {name} is {img.size}, expected (256, 256)"
                if img.mode != "RGB":
                    return f"image {name} mode is {img.mode}, expected RGB"
        except Exception as exc:  # noqa: BLE001
            return f"image {name} is not a decodable RGB PNG: {exc}"
    return None


def compute_fd(images_dir: Path) -> float:
    from gpic_eval.eval import eval_with_ref_stats

    result = eval_with_ref_stats(
        str(images_dir), str(TEST_STATS),
        models=["dino"], metrics=["fd"],
        batch_size=EVAL_BATCH, num_workers=EVAL_WORKERS, device="cuda:0")
    return float(result.fd["dino"])


def dino_embeddings(paths: list[Path]) -> "list":
    import numpy as np
    import torch
    from gpic_eval.backends.dino import build_dinov2, _dino_transform
    from PIL import Image

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_dinov2(device)
    feats = []
    with torch.no_grad():
        for start in range(0, len(paths), 64):
            batch = []
            for path in paths[start:start + 64]:
                arr = np.asarray(Image.open(path).convert("RGB"))
                batch.append(_dino_transform(arr))
            out = model(torch.stack(batch).to(device))
            feats.append(out.float().cpu())
    return torch.cat(feats).numpy()


def _cos_rows(a, b):
    import numpy as np
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.maximum(den, 1e-12)


def _run_generate(names: list[str], spot_jsonl: Path, ckpt: Path,
                  seed: int, out_dir: Path) -> str | None:
    """Invoke the untrusted generate.py under isolation. Returns a failure
    reason, or None on success (out_dir then holds one PNG per name)."""
    generate = CANDIDATE_ROOT / "generate.py"
    cmd = [sys.executable, str(generate),
           "--captions", str(spot_jsonl), "--out", str(out_dir),
           "--seed", str(seed), "--checkpoint", str(ckpt)]
    print("+", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, env=_child_env(), cwd=str(out_dir.parent),
                              timeout=SPOT_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"generate.py exceeded the {SPOT_TIMEOUT}s spot timeout"
    if proc.returncode != 0:
        return f"generate.py exited {proc.returncode} on the spot subset"
    missing = [n_ for n_ in names if not (out_dir / n_).exists()]
    if missing:
        return f"generate.py produced no image for {len(missing)} spot keys"
    return None


def spot_regenerate(rows: list[dict], images_dir: Path,
                    sampling_seed: int, model_state: Path,
                    model_ckpt: Path) -> dict | str:
    """Bind the submitted images to the submitted model.

    The agent authors generate.py, so the gate must resist a generate.py that
    (a) reads back the submitted images, (b) leaks the verifier's secrets, or
    (c) ignores the checkpoint and returns fixed/looked-up images. Defenses:
      * the hidden spot subset is seeded by sha256(model_state)+secret salt;
      * generate.py runs with a scrubbed env (no salt/stats/thresholds) and
        the submitted images directory is moved out of reach for the run;
      * generate.py is handed the checkpoint explicitly via --checkpoint from
        a verifier-controlled path;
      * POSITIVE BINDING: a second regeneration from a materially perturbed
        checkpoint must move the output — output that does not depend on the
        weights is not coming from the model.
    Returns a metrics dict, or a failure-reason string.
    """
    import numpy as np
    import torch

    salt = os.environ.get("GPIC_SPOT_SALT", "")
    if not salt:
        if not SMOKE:
            return "GPIC_SPOT_SALT not provided to the verifier"
        salt = "smoke"
    digest = hashlib.sha256()
    with model_state.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    seed_spot = int.from_bytes(
        hashlib.sha256(digest.digest() + salt.encode()).digest()[:8], "big")
    rng = random.Random(seed_spot)
    n = min(SPOT_N, len(rows))
    picked = sorted(rng.sample(range(len(rows)), n))
    spot_rows = [rows[i] for i in picked]
    names = [expected_name(r) for r in spot_rows]

    # Capture the submitted embeddings BEFORE running any agent code.
    sub = dino_embeddings([images_dir / n_ for n_ in names])

    workdir = Path(tempfile.mkdtemp(prefix="gpic-spot-"))
    spot_jsonl = workdir / "spot.jsonl"
    with spot_jsonl.open("w", encoding="utf-8") as handle:
        for row in spot_rows:
            handle.write(json.dumps(row) + "\n")
    clean_dir = workdir / "regen_clean"
    pert_dir = workdir / "regen_perturbed"
    clean_dir.mkdir()
    pert_dir.mkdir()
    # Verifier-controlled checkpoints outside the (hidden) submission dir.
    clean_ckpt = workdir / "clean.ckpt"
    pert_ckpt = workdir / "perturbed.ckpt"
    shutil.copy2(model_ckpt, clean_ckpt)
    payload = torch.load(model_ckpt, map_location="cpu", weights_only=False)
    _perturb_checkpoint(payload)
    torch.save(payload, pert_ckpt)

    # Hide the submitted images so a copy-based generate.py cannot read them.
    # Must stay on the same filesystem as OUTPUT_ROOT (a /tmp workdir would be
    # a cross-device rename), and carry a random name generate.py cannot
    # guess. The perturbation binding below is the deep defense; this is
    # defense-in-depth on top of the container/namespace isolation the
    # production verifier adds.
    submission = images_dir.parent
    hidden = submission.parent / f".gpic_submission_hidden_{os.getpid()}_{seed_spot & 0xffffff}"
    os.rename(submission, hidden)
    try:
        reason = _run_generate(names, spot_jsonl, clean_ckpt, sampling_seed, clean_dir)
        if reason is None:
            reason = _run_generate(names, spot_jsonl, pert_ckpt, sampling_seed, pert_dir)
    finally:
        os.rename(hidden, submission)
    if reason:
        return reason

    reg = dino_embeddings([clean_dir / n_ for n_ in names])
    pert = dino_embeddings([pert_dir / n_ for n_ in names])
    cos = _cos_rows(sub, reg)                 # submitted vs honest regen
    bind = _cos_rows(reg, pert)               # clean regen vs perturbed regen
    pass_frac = float((cos >= SPOT_COS).mean())
    median_cos = float(np.median(cos))
    bind_median = float(np.median(bind))
    metrics = {"n": n, "pass_frac": round(pass_frac, 6),
               "median_cos": round(median_cos, 6),
               "min_cos": round(float(cos.min()), 6),
               "bind_median_cos": round(bind_median, 6)}
    print("spot gate:", json.dumps(metrics))
    if bind_median >= SPOT_BIND_MAX_COS:
        return ("submitted images do not depend on the model weights "
                "(perturbing the checkpoint did not change generate.py's "
                f"output): {json.dumps(metrics)}")
    if pass_frac < SPOT_PASS_FRAC or median_cos < SPOT_MEDIAN_COS:
        return ("submitted images are not reproducible from the staged model "
                f"via generate.py: {json.dumps(metrics)}")
    return metrics


def _perturb_checkpoint(payload) -> None:
    """Add large Gaussian noise to the tensor leaves of a checkpoint in place."""
    import torch

    def walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)
        elif torch.is_tensor(obj) and obj.is_floating_point():
            scale = obj.detach().std().clamp_min(1e-4) * SPOT_PERTURB_STD
            obj.add_(torch.randn_like(obj) * scale)
    walk(payload)


def provenance_cross_checks(provenance: dict) -> str | None:
    sampling = provenance.get("sampling")
    if not isinstance(sampling, dict):
        return "provenance.sampling missing"
    if float(sampling.get("guidance", -1)) != 1.0:
        return f"provenance.sampling.guidance must be 1.0, got {sampling.get('guidance')!r}"
    if "seed" not in sampling:
        return "provenance.sampling.seed missing"
    try:
        images_seen = int(provenance.get("images_seen"))
    except (TypeError, ValueError):
        return "provenance.images_seen missing or non-integer"
    if images_seen > IMAGES_CEILING * RESUME_REPLAY_TOLERANCE:
        return (f"images_seen {images_seen} exceeds the one-epoch ceiling "
                f"{IMAGES_CEILING} (+0.5% resume tolerance)")
    # Declared pretrained weights must all be on the allowlist (trust-side
    # check; the agent-side audit tool is advisory).
    weights = provenance.get("pretrained_weights")
    if isinstance(weights, list):
        for entry in weights:
            wid = (entry.get("hf_id") or entry.get("name")
                   if isinstance(entry, dict) else entry)
            if wid not in PRETRAINED_ALLOWLIST:
                return f"pretrained weight off the allowlist: {wid!r}"
    elif weights not in ("none", "None", None):
        return ("provenance.pretrained_weights must be 'none' or a list of "
                "allowlisted {hf_id, sha256} entries")
    # Fail closed: a scored submission must carry the async-wrapper ledger.
    # An absent attempts dir is not "no violations" — it means the training
    # was not run under the budget watchdog, so budget/one-pass claims are
    # unverifiable. (Smoke fixtures may omit it.)
    attempts_dir = OUTPUT_ROOT / "attempts"
    status_paths = sorted(attempts_dir.glob("*/status.json")) if attempts_dir.exists() else []
    if not status_paths and not SMOKE:
        return ("no attempt ledger (/app/output/attempts/*/status.json): the "
                "candidate was not run under the budget watchdog")
    for status_path in status_paths:
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return f"unreadable attempt status: {status_path}"
        if status.get("state") == "running":
            return f"attempt {status_path.parent.name} status is still 'running'"
        charged = float(status.get("charged_gpu_hours", 0.0))
        if charged > GPU_HOURS_CAP * 1.01:
            return (f"attempt {status_path.parent.name} charged "
                    f"{charged} GPU-hours > cap {GPU_HOURS_CAP}")
        if status.get("data_overread_flag"):
            return (f"attempt {status_path.parent.name} flagged by the "
                    "trusted watchdog for reading more bytes than one "
                    "pass over the training set allows")
    return None


def main() -> int:
    provenance_path = OUTPUT_ROOT / "provenance.json"
    if not provenance_path.exists():
        return fail("missing provenance.json")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return fail(f"provenance.json is not valid JSON: {exc}")

    selection_path = OUTPUT_ROOT / "submission-selection.json"
    if not selection_path.exists():
        return fail("missing submission-selection.json (run gpic_task_tool.py stage)")
    model_ckpt = OUTPUT_ROOT / "submission" / "model.ckpt"
    model_state = OUTPUT_ROOT / "model_state.pt"
    images_dir = OUTPUT_ROOT / "submission" / "images"
    generate = CANDIDATE_ROOT / "generate.py"
    for path in (model_ckpt, model_state, images_dir, generate):
        if not path.exists():
            return fail(f"missing staged artifact: {path}")

    if not EVAL_JSONL.exists():
        return fail(f"frozen caption set missing: {EVAL_JSONL}")
    if not TEST_STATS.exists():
        return fail(f"sealed reference statistics missing: {TEST_STATS}")

    reason = provenance_cross_checks(provenance)
    if reason:
        return fail(reason)

    rows = load_captions()
    reason = validate_images(images_dir, rows)
    if reason:
        return fail(reason)

    import torch

    state = torch.load(model_state, map_location="cpu", weights_only=True)
    parameters = sum(int(v.numel()) for v in state.values())

    try:
        fd = compute_fd(images_dir)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the gate
        return fail(f"FD-DINOv2 computation failed: {exc}")
    if not math.isfinite(fd) or fd <= 0:
        return fail(f"FD-DINOv2 is not a positive finite number: {fd}")

    spot = spot_regenerate(rows, images_dir,
                           int(provenance["sampling"]["seed"]), model_state,
                           model_ckpt)
    if isinstance(spot, str):
        return fail(spot)

    metrics = {
        "fd_dinov2": fd,
        "n_images": len(rows),
        "parameters": parameters,
        "images_seen": int(provenance.get("images_seen")),
        "spot": spot,
        "smoke": SMOKE,
    }
    (OUTPUT_ROOT / "verifier-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
