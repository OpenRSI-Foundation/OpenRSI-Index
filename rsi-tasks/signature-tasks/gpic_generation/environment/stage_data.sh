#!/usr/bin/env bash
# One-time data staging for the gpic-generation task. Run by task
# infrastructure, NOT inside the task container. Produces:
#
#   $STAGE_ROOT/gpic/train/                agent-visible, read-only (12.8 TB full)
#   $STAGE_ROOT/gpic/val/                  agent-visible, read-only
#   $STAGE_ROOT/gpic/gpic_eval_50k.jsonl   agent-visible frozen caption set
#   $STAGE_ROOT/gpic/val_stats.npz         agent-visible screening reference
#   $VERIFIER_STAGE/test_stats.npz         VERIFIER-ONLY sealed reference
#   $VERIFIER_STAGE/gpic_eval_50k.jsonl    verifier-owned caption copy
#   $STAGE_ROOT/gpic/GPIC_STAGING_MANIFEST.json
#
# The GPIC dataset (HF: stanford-vision-lab/gpic) is gated (auto-approve):
# accept the terms once in a browser, then export HF_TOKEN. Test IMAGE
# bytes are never written anywhere agent-visible — the caption jsonl is
# built by streaming the test tars and keeping only {key, caption,
# caption_type}. test_stats.npz never touches $STAGE_ROOT.
#
# Smoke staging: GPIC_STAGE_SMOKE=1 downloads only train shards 00000-00001
# and val shard 00000 (~4 GB) instead of the full corpus.
set -euo pipefail

HF_REPO="stanford-vision-lab/gpic"
STAGE_ROOT="${STAGE_ROOT:-/datasets}"
VERIFIER_STAGE="${VERIFIER_STAGE:?set to a verifier-only staging dir (never mounted into the agent container)}"
: "${HF_TOKEN:?export HF_TOKEN with gated access to ${HF_REPO}}"

GPIC="${STAGE_ROOT}/gpic"
mkdir -p "${GPIC}/train" "${GPIC}/val" "${VERIFIER_STAGE}"

hf_get() {  # hf_get <repo-relative-path> <local-root>
  python3 - "$1" "$2" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="stanford-vision-lab/gpic", repo_type="dataset",
                       filename=sys.argv[1], local_dir=sys.argv[2],
                       token=os.environ["HF_TOKEN"])
print("staged", path)
PY
}

# 1) Train + val shards.
if [ "${GPIC_STAGE_SMOKE:-0}" = "1" ]; then
  TRAIN_LAST=1; VAL_LAST=0
  echo ">>> SMOKE staging: 2 train shards + 1 val shard"
else
  TRAIN_LAST=7999; VAL_LAST=31
fi
for i in $(seq -f "%05g" 0 "${TRAIN_LAST}"); do
  hf_get "train/gpic_train_${i}.tar" "${GPIC}"
done
for i in $(seq -f "%05g" 0 "${VAL_LAST}"); do
  hf_get "val/gpic_val_${i}.tar" "${GPIC}"
done

# 2) Reference statistics: val (agent screening) vs test (SEALED).
hf_get "reference_stats/val_stats.npz" "${GPIC}"
mv "${GPIC}/reference_stats/val_stats.npz" "${GPIC}/val_stats.npz" 2>/dev/null || true
hf_get "reference_stats/test_stats.npz" "${VERIFIER_STAGE}"
mv "${VERIFIER_STAGE}/reference_stats/test_stats.npz" "${VERIFIER_STAGE}/test_stats.npz" 2>/dev/null || true
rmdir "${GPIC}/reference_stats" "${VERIFIER_STAGE}/reference_stats" 2>/dev/null || true

# 3) Frozen 50k caption set, EXACTLY aligned with the paper's reference:
#    test_stats.npz carries `key` (50,000 sha256 keys) and `tar_idx` (which
#    of the 128 test tars holds each key) — the caption jsonl is built for
#    precisely those keys, in npz order. The repo ships no
#    gpic_eval_50k.jsonl (the sampling config references it as a placeholder
#    path); this reconstruction is the task's frozen definition — sha256
#    recorded in the manifest so the agent-visible and verifier-baked copies
#    provably match. Test tars are streamed from HF and deleted after
#    caption extraction; test image bytes never land in agent-visible
#    storage.
python3 - "${GPIC}" "${VERIFIER_STAGE}" <<'PY'
import hashlib, json, os, sys, tarfile, tempfile
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

gpic_root, verifier_stage = sys.argv[1], sys.argv[2]
ref = np.load(os.path.join(verifier_stage, "test_stats.npz"))
keys = [str(k) for k in ref["key"]]
tar_idx = [int(t) for t in ref["tar_idx"]]
order = {k: i for i, k in enumerate(keys)}
by_tar = defaultdict(set)
for k, t in zip(keys, tar_idx):
    by_tar[t].add(k)
rows = [None] * len(keys)
with tempfile.TemporaryDirectory() as tmp:
    for t in sorted(by_tar):
        name = f"test/gpic_test_{t:05d}.tar"
        path = hf_hub_download(repo_id="stanford-vision-lab/gpic",
                               repo_type="dataset", filename=name,
                               local_dir=tmp, token=os.environ["HF_TOKEN"])
        wanted = by_tar[t]
        with tarfile.open(path) as tar:
            for member in tar:
                if not member.name.endswith(".json"):
                    continue
                stem = member.name.rsplit("/", 1)[-1][:-5]
                if stem not in wanted:
                    continue
                meta = json.load(tar.extractfile(member))
                rows[order[meta["key"]]] = {
                    "key": meta["key"], "caption": meta["caption"],
                    "caption_type": meta["caption_type"]}
        os.unlink(path)
missing = sum(row is None for row in rows)
assert not missing, f"{missing} reference keys not found in the test tars"
payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
for root in (gpic_root, verifier_stage):
    with open(os.path.join(root, "gpic_eval_50k.jsonl"), "w", encoding="utf-8") as f:
        f.write(payload)
print(json.dumps({"records": len(rows), "shards_read": len(by_tar), "sha256": sha}))
with open(os.path.join(gpic_root, "gpic_eval_50k.sha256"), "w") as f:
    f.write(sha + "\n")
PY

# 4) Manifest + seal. train_total_bytes seeds GPIC_TRAIN_BYTES for the
#    watchdog's one-pass over-read tripwire.
python3 - "${GPIC}" <<'PY'
import hashlib, json, os, sys
root = sys.argv[1]
manifest = {}
for sub in ("train", "val"):
    base = os.path.join(root, sub)
    files = sorted(os.listdir(base)) if os.path.isdir(base) else []
    manifest[sub] = {"files": len(files),
                     "bytes": sum(os.path.getsize(os.path.join(base, f)) for f in files)}
manifest["gpic_eval_50k_sha256"] = open(os.path.join(root, "gpic_eval_50k.sha256")).read().strip()
manifest["val_stats_bytes"] = os.path.getsize(os.path.join(root, "val_stats.npz"))
manifest["train_total_bytes"] = manifest["train"]["bytes"]
out = os.path.join(root, "GPIC_STAGING_MANIFEST.json")
json.dump(manifest, open(out, "w"), indent=2)
print(json.dumps(manifest, indent=2))
print("manifest:", out)
print("Set GPIC_TRAIN_BYTES=%d in task.toml [environment.env]" % manifest["train_total_bytes"])
PY
chmod -R a-w "${GPIC}" || true
echo "Staging complete. Mount ${GPIC} read-only at /datasets/gpic in the agent"
echo "container. Copy ${VERIFIER_STAGE}/{test_stats.npz,gpic_eval_50k.jsonl}"
echo "next to tests/Dockerfile before building the tests image."
