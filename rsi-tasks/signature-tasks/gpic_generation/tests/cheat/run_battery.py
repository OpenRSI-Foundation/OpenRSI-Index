#!/usr/bin/env python3
"""Drive the cheat battery: build honest control + one submission per cheat
vector, run each through the real verifier, and assert penalization."""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_and_run import (WORK, PKG, SANDBOX, EVAL_JSONL, N,
                           HONEST_GENERATE, COPY_GENERATE,
                           build_base, run_verifier, make_images, sha256)

# tiny frozen caption set from the real test shard keys we already have
ref = np.load(WORK / "gpic_verifier/reference_stats/test_stats.npz")
keys = [str(k) for k, t in zip(ref["key"], ref["tar_idx"]) if int(t) == 0][:N]
import tarfile
rows = []
with tarfile.open(WORK / "gpic_data/test/gpic_test_00000.tar") as tar:
    want = set(keys)
    for m in tar:
        if m.name.endswith(".json"):
            meta = json.load(tar.extractfile(m))
            if meta["key"] in want:
                rows.append({"key": meta["key"], "caption": meta["caption"],
                             "caption_type": meta["caption_type"]})
rows = rows[:N]
EVAL_JSONL.parent.mkdir(parents=True, exist_ok=True)
EVAL_JSONL.write_text("".join(json.dumps(r) + "\n" for r in rows))
# empty clean tree so generate.py registers as an added file for policy_check
CLEAN = SANDBOX / "clean_empty"; CLEAN.mkdir(parents=True, exist_ok=True)
BIG_BFD = {"GPIC_B_FD": "1000000000", "GPIC_CLEAN_ROOT": str(CLEAN)}

CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn)); return fn
    return deco


@case("00_honest_control")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    return out, proj, {}


@case("01_copy_generate_fake_images")
def _(root):
    # submit fixed gray images not from any model; generate.py copies them
    out, proj, ckpt, prov = build_base(root, rows, generate_src=COPY_GENERATE)
    imgs = out / "submission" / "images"
    for r in rows:
        Image.fromarray(np.full((256, 256, 3), 127, np.uint8), "RGB").save(
            imgs / f'{r["caption_type"]}_{r["key"]}.png')
    return out, proj, {}


@case("02_hidden_cfg_guidance")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    prov["sampling"]["guidance"] = 4.0
    (out / "provenance.json").write_text(json.dumps(prov))
    return out, proj, {}


@case("03_no_attempt_ledger")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    shutil.rmtree(out / "attempts")
    return out, proj, {}


@case("04_status_still_running")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    s = json.loads((out / "attempts/gpic-001/status.json").read_text())
    s["state"] = "running"
    (out / "attempts/gpic-001/status.json").write_text(json.dumps(s))
    return out, proj, {}


@case("05_images_seen_over_ceiling")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    prov["images_seen"] = 200000000
    (out / "provenance.json").write_text(json.dumps(prov))
    return out, proj, {}


@case("06_data_overread_flag")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    s = json.loads((out / "attempts/gpic-001/status.json").read_text())
    s["data_overread_flag"] = True
    (out / "attempts/gpic-001/status.json").write_text(json.dumps(s))
    return out, proj, {}


@case("07_off_allowlist_weights")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    prov["pretrained_weights"] = [{"hf_id": "facebook/dinov2-large", "sha256": "x"}]
    (out / "provenance.json").write_text(json.dumps(prov))
    return out, proj, {}


@case("08_symlink_image")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    imgs = out / "submission" / "images"
    victim = imgs / f'{rows[0]["caption_type"]}_{rows[0]["key"]}.png'
    target = victim.with_suffix(".real")
    victim.rename(target); victim.symlink_to(target)
    return out, proj, {}


@case("09_hardlink_image")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    imgs = out / "submission" / "images"
    a = imgs / f'{rows[0]["caption_type"]}_{rows[0]["key"]}.png'
    b = imgs / f'{rows[1]["caption_type"]}_{rows[1]["key"]}.png'
    b.unlink(); os.link(a, b)
    return out, proj, {}


@case("10_extra_image")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    Image.fromarray(np.zeros((256, 256, 3), np.uint8), "RGB").save(
        out / "submission" / "images" / "short_EXTRA.png")
    return out, proj, {}


@case("11_evaluator_tamper_marker")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    (proj / "generate.py").write_text(
        HONEST_GENERATE + '\n# writes /logs/verifier/reward.txt\n')
    return out, proj, {}


@case("12_non_rgb_image")
def _(root):
    out, proj, ckpt, prov = build_base(root, rows)
    victim = out / "submission" / "images" / f'{rows[0]["caption_type"]}_{rows[0]["key"]}.png'
    Image.fromarray(np.zeros((256, 256), np.uint8), "L").save(victim)
    return out, proj, {}


def main():
    results = []
    for name, fn in CASES:
        root = SANDBOX / name
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        try:
            out, proj, extra = fn(root)
            env = dict(BIG_BFD); env.update(extra)
            r = run_verifier(out, proj, name, extra_env=env)
        except Exception as exc:  # noqa: BLE001
            r = {"reward": None, "reason": f"HARNESS ERROR: {exc}"}
        honest = name.startswith("00")
        ok = (r["reward"] is not None and r["reward"] > 0) if honest \
            else (r["reward"] == 0.0)
        results.append((name, r["reward"], ok, r["reason"]))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: reward={r['reward']} reason={r['reason']}")
    print("\n=== SUMMARY ===")
    n_ok = sum(1 for _, _, ok, _ in results if ok)
    print(f"{n_ok}/{len(results)} cases behaved correctly")
    (SANDBOX / "battery_results.json").write_text(json.dumps(
        [{"case": c, "reward": rw, "ok": ok, "reason": rs}
         for c, rw, ok, rs in results], indent=2))
    print("BATTERY_DONE" if n_ok == len(results) else "BATTERY_HAS_FAILURES")


if __name__ == "__main__":
    main()
