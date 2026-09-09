# Train any text-to-image model on GPIC in one epoch, scored by FD-DINOv2 at guidance 1

## Research Question

With a strict one-pass data budget (the 100M-image GPIC training corpus,
each sample consumed at most once), pure conditional sampling (guidance
fixed to 1.0 — no classifier-free guidance, no negative prompts, no
sampling-time crutches), and the evaluation representation off-limits as a
training signal, which text-to-image modeling choices actually carry sample
quality? Unlike language-model pretraining, the objective itself is a free
variable here: flow matching, diffusion, autoregression, masked prediction,
GANs, and hybrids are all admissible, as are any architecture and any
parameter count that fit the compute budget. The fixed-data single-epoch
regime means scale is not free lunch — under fixed data, unregularized
bigger models are known to lose to smaller ones — so the question is one of
data efficiency per unit compute, not size. The pinned PixelGen JiT_T2I
pixel-space flow-matching baseline is the attribution anchor; your goal is
to train, from scratch, a model whose 50k generated images score a lower
FD-DINOv2 against the sealed GPIC test reference than that baseline.

Changing only the sampler step count, batch shape, or precision of the
pristine baseline is not a new model; the scored deliverable must embody at
least one substantive modeling decision (objective, architecture,
conditioning, noise schedule/parameterization, tokenization, or training
recipe) that you can name and ablate.

## Reference Baselines

The sole scientific source is the pinned repository
`keshik6/gpic@afa82daab73cfeab4ef9fbd174ff62b68bffc456`, vendored writable
at `/app/project/gpic`. The reference baseline is the unchanged
`baselines/PixelGen/config_pretrain_gpic_full.yaml` under this exact
contract: JiT_T2I denoiser (~1.12B parameters: patch 16, hidden 1536, 16
blocks + 4 text blocks, pixel-space — the autoencoder is the identity),
frozen Qwen3-1.7B text encoder (staged offline at `/opt/hf/Qwen3-1.7B`),
flow matching with lognorm time sampling and `null_condition_p: 0.1`
conditioning dropout, AdamW lr 1e-4, global batch 256 (32 per GPU x 8),
bf16-mixed, on 1 node x 8 H100. One full epoch is 390,625 steps (100M /
256); the config's `max_steps: 400001` is an upstream artifact — the
one-epoch contract binds, not that number. Sampling: `EulerSamplerJiT`, 50
steps, **guidance 1.0** (the shipped sampling config says 4.0 and its run
name says 1.75; the task fixes 1.0 — at guidance 1.0 the update is exactly
the conditional prediction). The known `main.py` import breakage
(`src.lightning_*_keshik`) is already fixed by task infrastructure in both
the pristine and writable trees.

### Baseline metrics

The sealed baseline is the pristine config trained one full epoch and
sampled at guidance 1.0. **No FD-DINOv2 number exists for this
configuration anywhere — the repository publishes none. Every value below
is provisional; the task authors run the pristine config once on this
infrastructure and every number is replaced by that sealed run (plus two
extra baseline regenerations to measure the noise band for the reward
anchors and the spot-gate thresholds).**

| Quantity | Provisional value | Sealed value |
|---|---:|---:|
| Denoiser parameters | ~1,120,000,000 | pending author run |
| FD-DINOv2 (50k captions, guidance 1.0) | PLACEHOLDER 200.0 | pending author run |
| Training GPU-hours (1 epoch, 1x8 H100) | ~500-650 (estimate) | pending author run |
| Images seen | 100,000,000 (exactly one pass) | 100,000,000 |
| Reward | 0.0 at baseline by construction | 0.0 by construction |

`/task-tools/gpic_local_eval.py` FD against the val reference is your
screening signal; the sealed run will also record the baseline's val-FD at
several checkpoints so weak candidates can be killed early at a fraction of
a full budget.

## Evaluation and success

### Primary metric

FD-DINOv2, lower is better: generate exactly one 256x256 RGB PNG per
caption of the frozen 50k caption set (`$GPIC_EVAL_JSONL`, one JSON per
line with `key`, `caption`, `caption_type`), named
`{caption_type}_{key}.png`, and the frozen verifier scores the set with the
pinned `gpic_eval` toolkit (DINOv2 ViT-L/14 features, Frechet distance)
against sealed test-split reference statistics that are never present in
this environment. Screening FD against the val reference is available via
`python /task-tools/gpic_local_eval.py --images <dir>` (GPU time charged).

Generation must go through a frozen contract you implement at
`/app/project/generate.py`:

```bash
python /app/project/generate.py \
    --captions <jsonl> --out <dir> --seed <int> --checkpoint <path> \
    [--batch-size <int>] [--devices <int>]
```

Requirements: reads caption rows `{key, caption, caption_type}`; writes
exactly one `{caption_type}_{key}.png` (256x256 RGB, a genuine regular
file — not a symlink or hardlink) per row into `--out`; loads the model
weights from `--checkpoint` (plus declared allowlisted weights) and from
nowhere else; is deterministic given `--seed` and `--checkpoint` (all
randomness derived from the seed and the row, never wall clock or unseeded
global RNG — reruns must reproduce each image up to floating-point noise);
samples purely conditionally (exactly one conditioning stream per generated
image); `--batch-size`/`--devices` may change throughput only, never pixels
beyond numerical noise. The verifier regenerates a hidden subset of caption
keys through this script — from a verifier-controlled checkpoint path, with
the submitted images directory moved out of reach and the environment
scrubbed of verifier secrets — and (a) compares each regeneration to your
submitted image in DINOv2 feature space and (b) re-runs it from a perturbed
copy of the checkpoint and requires the output to change. Images that do
not come from the staged model via this contract — dataset images,
cherry-picks, or output that ignores `--checkpoint` — score zero. Generate
your 50k submission images with this same script (charged to your budget,
`run_kind=generation`).

### Secondary metrics and validity gates (optional)

Reported but not scored: parameter count, images_seen, per-attempt
GPU-hours, spot-gate cosine statistics. Hard validity gates (any failure
scores zero): the policy gate over `/tests/policy.yaml` (rules RH-001 to
RH-010 — notably the DINO-representation ban RH-004, the pure-conditional
sampling rule RH-006, and the one-pass data budget RH-007), exact image-set
match against the frozen captions, the spot-regeneration gate,
`provenance.sampling.guidance == 1.0`, `images_seen <= 100,000,000`
(+0.5% resume tolerance), per-attempt GPU-hours within the cap, and no
trusted data-overread flag.

### Reward

```text
r      = B_FD / fd_dinov2            # = 1 at the sealed baseline
reward = 0                           if any validity gate fails,
                                     fd is invalid, or r <= r_floor (1.0)
reward = clip((r - 1) / (r_ref - 1), 0, 1)    otherwise, r_ref = 1.25
```

A 20% FD reduction relative to the sealed baseline saturates the reward;
any regression scores zero. `r_floor` and `r_ref` are provisional and are
re-placed outside the measured regeneration-noise band at sealing.
Scientific reporting must include the raw FD; do not optimize a rounded
reward.

## Workspace

| Item | Path / value |
|---|---|
| Writable workspace | `/app/project/gpic`, pinned `keshik6/gpic@afa82daab73cfeab4ef9fbd174ff62b68bffc456` (gpic_eval + baselines/PixelGen), main.py import fix pre-applied |
| Generation contract | `/app/project/generate.py` — you write it; interface above is frozen |
| Training data | `$GPIC_TRAIN_DIR` = `/datasets/gpic/train` — `gpic_train_{00000..07999}.tar`, 100M image-text webdataset shards, read-only. **At most one pass per attempt; every consumed sample unique** |
| Val data | `$GPIC_VAL_DIR` = `/datasets/gpic/val` — validation loaders and screening only, never training data |
| Frozen captions | `$GPIC_EVAL_JSONL` = `/datasets/gpic/gpic_eval_50k.jsonl` — generation prompts ONLY; any other use (training, tuning, retrieval, selection) is a hard zero |
| Screening reference | `$GPIC_VAL_STATS` = `/datasets/gpic/val_stats.npz` via `/task-tools/gpic_local_eval.py`; the sealed test reference is verifier-only |
| Pretrained allowlist | `Qwen/Qwen3-1.7B` at `/opt/hf/Qwen3-1.7B` (text encoder; frozen or fine-tuned). Nothing else — no DINO-family or DINO-distilled weights in any role |
| Resolution | 256x256 output, fixed (note: the pinned dataset transform hardcodes 256 in `src/data/dataset/gpic_dataset.py`; internal representations are free) |
| Topology | 1 node x 8 H100, exactly; launch training via `/task-tools/launch_gpic_1x8.sh <config> [args...]` |
| Training budget | at most `$GPIC_GPU_HOURS_CAP` (= 1000.0, provisional) H100-hours per attempt (`wall_seconds x 8 / 3600`); watchdog hard-kills over-budget attempts. All GPU work charged (training, 50k generation, screening); CPU-only work free; frozen verification uncharged |
| Data budget | `images_seen <= 100,000,000` per attempt, each sample at most once; resume continues the stream without rewinding (0.5% replay tolerance); the trusted watchdog records read-bytes and flags physically impossible single-pass claims |
| Submission | `python /task-tools/gpic_task_tool.py stage --checkpoint <ckpt> --images <dir_of_50k_pngs> --generate-cmd "<cmd>"`, then `audit` |
| Network | disabled; the pinned repository, the staged data, and your own measurements are the only inputs |

## What you may change

Everything about the generative model is yours: architecture and size,
objective (flow matching, diffusion, autoregressive, masked prediction,
GAN, hybrids), pixel- vs latent-space design (any autoencoder must itself
be trained here, inside the same data/compute budget), optimizer,
schedules, batch shape, precision, EMA, augmentation, image/text
tokenization, data ordering within the single pass, dataloader and
throughput engineering, the sampler and its step count, `main.py`, the
trainer classes, and training-time conditioning dropout. Distillation
between models trained inside this environment is allowed within the shared
one-pass data budget. The Qwen3-1.7B text encoder may be used frozen,
fine-tuned, or replaced by one you train from scratch.

Frozen: the GPIC train corpus and the at-most-one-pass budget, the 256x256
output resolution, the pure-conditional sampling contract (RH-006: no
combination of multiple differently-conditioned forwards into a sampling
update), the frozen caption set, the pretrained-weights allowlist (RH-005),
the DINO-representation ban (RH-004 — including the repo's own
`training_repa_JiT_LPIPS_DINO*` perceptual-loss trainers, which must remain
unused), the GPU-hour cap and 1x8 topology, and the frozen verifier.

## Persistent experiment loop

One scientific single run is one candidate trained within the GPU-hour and
one-pass data budgets on 1x8 ranks. Screening runs (partial-epoch prefixes,
reduced shard ranges) are allowed for triage; label them `screening` in the
ledger — the scored candidate must be a completed full pass.

For each attempt:

1. State one falsifiable hypothesis — modeling mechanism, expected effect
   on FD (or on val-FD screening), and why.
2. Change one interpretable variable group; give the candidate a new named
   config under your workspace; record the exact patch or source hash.
3. Submit asynchronously and end your turn (do not sleep or poll):

   ```bash
   attempt=gpic-001-<short-name>
   python /task-tools/gpic_async_run.py submit \
     --attempt-id "$attempt" --poll-seconds 1800 --run-kind full_budget -- \
     /task-tools/launch_gpic_1x8.sh \
       <your-config>.yaml \
       --trainer.default_root_dir "/app/output/attempts/$attempt/outputs"
   ```

   The async adapter keeps the environment alive, checks `status.json`
   every 30 minutes without an LLM call, and resumes this session when the
   attempt reaches a terminal state (`completed`, `failed`, or
   `killed_budget`). On resume, summarize the real evidence:

   ```bash
   python /task-tools/gpic_task_tool.py summarize \
     --attempt "/app/output/attempts/$attempt"
   ```

4. Generate the candidate's 50k submission images with your frozen
   contract (charged, `run_kind=generation`), then screen:

   ```bash
   python /task-tools/gpic_async_run.py submit \
     --attempt-id "$attempt-gen" --run-kind generation -- \
     python /app/project/generate.py \
       --captions "$GPIC_EVAL_JSONL" \
       --out "/app/output/attempts/$attempt/images" \
       --seed 0 --devices 8
   python /task-tools/gpic_local_eval.py \
     --images "/app/output/attempts/$attempt/images"
   ```

5. Append one truthful JSON object to `/app/output/experiments.jsonl` per
   decision cycle. Required keys: `attempt_id`, `hypothesis`, `change`,
   `command`, `source_diff_sha256`, `run_kind`, `gpu_hours`, `images_seen`,
   `fd_val_screening`, `status`. Statuses include `rejected`, `failed`,
   `killed_budget`, and exactly one final `selected`.
6. Keeping `/app/output/RESEARCH_NOTES.md` up to date is encouraged (which
   mechanisms moved FD and why, citing `attempt_id`s; negative results) —
   it is your notebook, not a scored deliverable.
7. Select exactly one completed full-pass checkpoint, then stage and audit:

   ```bash
   python /task-tools/gpic_task_tool.py stage \
     --checkpoint /app/output/attempts/<selected>/outputs/.../<ckpt> \
     --images /app/output/attempts/<selected>/images \
     --generate-cmd "python /app/project/generate.py --captions $GPIC_EVAL_JSONL --out ... --seed 0"
   python /task-tools/gpic_task_tool.py audit --output-root /app/output
   python /task-tools/gpic_async_run.py complete-control
   ```

## Research deadline and handoff

Sessions are asynchronous: submitted attempts keep running while you are
offline and the harness resumes you on terminal states. Stage the first
passing candidate early, then keep improving and re-stage better ones until
the session deadline; the verifier scores only the staged submission.
Ending the session without a staged submission, a complete provenance
record, and a reconciled ledger scores zero. Interruption is not
completion: resumed attempts continue from the trainer's own checkpointing
without rewinding the data stream, and the ledger must record executed —
not intended — work.

## Required artifacts

- `/app/output/submission/model.ckpt` — the selected checkpoint, staged by
  the frozen task tool;
- `/app/output/submission/images/` — the 50k `{caption_type}_{key}.png`
  submission images, staged by the frozen task tool;
- `/app/output/model_state.pt` — tensor-only weights, exported by the
  staging tool (its hash seeds the verifier's hidden spot subset);
- `/app/output/submission-selection.json` — from the frozen task tool;
- `/app/project/generate.py` — the frozen generation contract;
- `/app/output/provenance.json` with `model_description`, `parameters`,
  `training_data`, `shards_consumed`, `images_seen`, `resume_events`,
  `gpu_hours` (per attempt and total), `training_command`,
  `sampling` (`{guidance: 1.0, sampler, num_steps, seed}`),
  `pretrained_weights` (allowlisted entries with source id and hash, or
  `"none"`), `generation_command`, `upstream_commits`,
  `evaluation_commands`, `optimization_attempts`, `web_search: "disabled"`;
- `/app/output/experiments.jsonl`, one row per decision cycle;
- useful logs under `/app/output/logs` and per-attempt
  `/app/output/attempts/<id>/`.
