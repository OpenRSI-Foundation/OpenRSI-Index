#!/usr/bin/env python3
"""Cheat battery: exercise the gpic_generation verifier against concrete
cheat submissions and confirm each is penalized (reward 0), while an honest
submission reaches the scoring path.

A toy honest model stands in for the real 1.1B PixelGen model so the battery
runs in seconds on one GPU: an honest generate.py derives each image
deterministically from (checkpoint bytes, seed, key). This exercises the
REAL verifier code paths (tests/evaluate.py, policy_check.py, score.py) —
the spot gate is model-agnostic, so a toy honest model is a faithful test of
whether cheats are caught.

Run inside the tests-equivalent env (torch + gpic_eval + dinov2 hub cache).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PKG = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("GPIC_WORK_ROOT") or PKG).expanduser().resolve()
SANDBOX = WORK / "cheat/sandbox"
REF = WORK / "gpic_verifier/reference_stats/test_stats.npz"   # sealed test stats
EVAL_JSONL = WORK / "cheat/eval_16.jsonl"                     # tiny frozen caption set
N = 16

# ---- honest toy generate.py: image = f(sha256(checkpoint), seed, key) ----
HONEST_GENERATE = '''\
import argparse, hashlib, json
import numpy as np
from PIL import Image
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--captions", required=True); p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, required=True); p.add_argument("--checkpoint", required=True)
p.add_argument("--batch-size", type=int, default=8); p.add_argument("--devices", type=int, default=1)
a = p.parse_args()
ck = hashlib.sha256(Path(a.checkpoint).read_bytes()).digest()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for line in Path(a.captions).read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    h = hashlib.sha256(ck + str(a.seed).encode() + r["key"].encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    # low-frequency structured image (8x8 color field upsampled) so distinct
    # checkpoints/seeds yield well-separated DINOv2 embeddings, like a real
    # generative model (unlike random noise, which DINOv2 collapses together)
    low = rng.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    img = np.array(Image.fromarray(low, "RGB").resize((256, 256), Image.NEAREST))
    Image.fromarray(img, "RGB").save(out / f'{r["caption_type"]}_{r["key"]}.png')
'''

# ---- cheat generate.py: ignores --checkpoint, copies submitted images ----
COPY_GENERATE = '''\
import argparse, json, shutil
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--captions", required=True); p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, required=True); p.add_argument("--checkpoint", required=True)
p.add_argument("--batch-size", type=int, default=8); p.add_argument("--devices", type=int, default=1)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
# try to read the (hidden) submitted images and echo them back
subs = [Path("/app/output/submission/images"), Path(a.out).parent.parent/"submission"/"images"]
for line in Path(a.captions).read_text().splitlines():
    if not line.strip(): continue
    r = json.loads(line); name = f'{r["caption_type"]}_{r["key"]}.png'
    for s in subs:
        if (s/name).exists(): shutil.copy2(s/name, out/name); break
    else:
        # fall back to a fixed gray image (checkpoint-independent)
        from PIL import Image; import numpy as np
        Image.fromarray(np.full((256,256,3),127,np.uint8),"RGB").save(out/name)
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_images(gen_py, out_dir, ckpt, seed, rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir.parent / "_gensrc.py"
    tmp.write_text(gen_py)
    cj = out_dir.parent / "_caps.jsonl"
    cj.write_text("".join(json.dumps(r) + "\n" for r in rows))
    subprocess.run([sys.executable, str(tmp), "--captions", str(cj),
                    "--out", str(out_dir), "--seed", str(seed),
                    "--checkpoint", str(ckpt)], check=True)


def build_base(root: Path, rows, generate_src=HONEST_GENERATE):
    """A complete, honest submission skeleton under root (mimics /app/output +
    /app/project)."""
    out = root / "output"
    proj = root / "project"
    (out / "submission").mkdir(parents=True, exist_ok=True)
    proj.mkdir(parents=True, exist_ok=True)
    # a small 'checkpoint' + tensor-only state
    ckpt = out / "submission" / "model.ckpt"
    state = {"w": torch.randn(2048, 512)}
    torch.save({"state_dict": state}, ckpt)
    torch.save(state, out / "model_state.pt")
    (proj / "generate.py").write_text(generate_src)
    # honest images from the checkpoint
    make_images(HONEST_GENERATE, out / "submission" / "images", ckpt, 0, rows)
    # provenance / selection / ledger
    prov = {
        "model_description": "toy", "parameters": 1, "training_data": "gpic_train",
        "shards_consumed": "00000..07999", "images_seen": 100000000, "resume_events": 0,
        "gpu_hours": 500.0, "training_command": "launch_gpic_1x8.sh ...",
        "sampling": {"guidance": 1.0, "sampler": "euler", "num_steps": 50, "seed": 0},
        "pretrained_weights": "none", "generation_command": "generate.py ...",
        "upstream_commits": "afa82da", "evaluation_commands": "gpic_local_eval",
        "optimization_attempts": 1, "web_search": "disabled"}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    (out / "submission-selection.json").write_text(json.dumps({
        "parameters": 1, "n_images": len(rows),
        "model_state_sha256": sha256(out / "model_state.pt")}, indent=2))
    (out / "experiments.jsonl").write_text(json.dumps({
        "attempt_id": "gpic-001", "status": "selected", "gpu_hours": 500.0}) + "\n")
    att = out / "attempts" / "gpic-001"
    att.mkdir(parents=True, exist_ok=True)
    (att / "status.json").write_text(json.dumps({
        "attempt_id": "gpic-001", "state": "completed", "charged_gpu_hours": 500.0,
        "data_overread_flag": False}))
    return out, proj, ckpt, prov


def run_verifier(out: Path, proj: Path, label: str, extra_env=None) -> dict:
    log = SANDBOX / f"log_{label}"
    log.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({
        "GPIC_OUTPUT_ROOT": str(out), "GPIC_CANDIDATE_ROOT": str(proj),
        "GPIC_VERIFIER_LOG_DIR": str(log), "GPIC_TESTS_DIR": str(PKG / "tests"),
        "GPIC_POLICY_PYTHON": sys.executable,
        "GPIC_CLEAN_ROOT": str(WORK / "gpic"), "GPIC_CANDIDATE_ROOT_": str(proj),
        "GPIC_EVAL_JSONL": str(EVAL_JSONL), "GPIC_TEST_STATS": str(REF),
        "GPIC_VERIFIER_SMOKE": "0", "POLICY_JUDGE_SKIP_LLM": "1",
        "GPIC_SPOT_N": str(N), "GPIC_SPOT_SALT": "battery-salt",
        "GPIC_EVAL_BATCH": "16", "GPIC_EVAL_WORKERS": "2",
        "GPIC_B_FD": "200.0", "GPIC_R_REF": "1.25", "GPIC_R_FLOOR": "1.0",
    })
    if extra_env:
        env.update(extra_env)
    # policy_check needs the clean/candidate diff; point clean at the same proj
    # so only genuine added markers (not the whole gpic tree) show as diffs.
    proc = subprocess.run(["bash", str(PKG / "tests" / "test.sh")], env=env,
                          cwd=str(SANDBOX), capture_output=True, text=True)
    (log / "testsh.out").write_text((proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
    reward = None
    reason = None
    rp = log / "reward.txt"
    if rp.exists():
        reward = float(rp.read_text().strip())
    for fn in ("evaluate-failure.json", "failure.json"):
        if (log / fn).exists():
            reason = json.loads((log / fn).read_text()).get("reason")
            break
    return {"reward": reward, "reason": reason}
