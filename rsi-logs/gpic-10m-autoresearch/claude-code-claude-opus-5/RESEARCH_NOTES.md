# Claude GPIC 10M autoresearch — research notes

Fresh independent trajectory started 2026-09-21 18:37 UTC (deadline epoch 1790361447, 96 h).
Code: worktree `/workspace/gpic-worktrees/claude`, branch `claude-10m-fresh`, forked from the
pinned commit `afa82da`. The commits of the earlier Claude trajectory (branch `autoresearch-claude`)
were deliberately NOT used: no code, configs, checkpoints, results or decisions from them.

## Fixed protocol

- Common root: `/shared/gpic-output/baseline_16x8/exp_pretraining_jit_256_gpic_full/epoch=0-step=39060.ckpt`
  (weights-only; contains `denoiser.*` and `ema_denoiser.*`, 1,122,396,928 denoiser params; baseline
  trained 39060 x 256 = 10.0M images with warmup-4000 then flat LR 1e-4, EMA 0.9999).
- Data: shared subset `/shared/gpic-data/gpic/subsets/train10m_seed20260921/shards` (802 shards,
  manifest sha256 e496f954...). Read via glob `gpic_train_*.tar` (sorted), split by rank then worker;
  webdataset without resampling + `max_epochs: 1` => every sample read at most once per run.
- Topology: 1 node x 4 H100 (P3) for everything (training, generation, evaluation).
- Default geometry: microbatch 32/GPU x accum 2 x 4 GPUs = global batch 256 (same effective batch as
  the baseline; micro-batching changed only for throughput).
- Screening metric: FD-DINOv2 vs the VAL reference (`val_stats.npz`, 50k val images) through the frozen
  `/task-tools/gpic_local_eval.py`. Screening captions = the val captions paired with the reference
  images (`common/val_captions_ref50k.jsonl`, 49,990 matched), random 10k subset with seed 20260921
  (`common/val_captions_screen10k.jsonl`). The frozen 50k eval captions are used ONLY for the final
  submission generation. FD at n=10k is biased upward relative to 50k; only compare at equal n.
- Sampling: `generate.py` (worktree root), pure conditional, one conditional forward per solver
  stage, noise per image seeded by sha256(seed|caption_type|key); default 50-step Euler, EMA weights.

## Log of decision cycles

(see experiments.jsonl for the machine-readable ledger)

### Cycle 0 — L0 systems probe + root anchor (submitted 2026-09-21)
- c10m-000-l0-probe: 300 steps from root, microbatch 32 x accum 2; throughput/memory/plumbing; 2k-image
  FD smoke on the step-300 checkpoint.
- c10m-001-root-eval: screening FD (10k) of the root EMA and raw weights; reproducibility check of
  generate.py (batch 64 x 4 devices vs batch 16 x 1 device).

## Hypothesis backlog (to be tested with matched controls)
- H-cooldown: annealing LR to 0 at the end of a continuation lowers FD vs constant LR at equal images.
- H-crop: training with center crops (the reference framing) instead of random 0.8-1.0 zoom crops at
  random positions lowers FD vs the center-cropped reference.
- H-nullp: null_condition_p 0.1 -> 0 (unconditional branch never used at guidance 1) frees capacity.
- H-ema: EMA decay choice for short continuations (0.9999 horizon ~10k steps).
- H-sampler: step count / time shift of the conditional ODE solver (only in combination with a model change).
- H-tsample: training timestep distribution (P_mean/P_std) shifted toward the regimes that dominate FD.

### Cycle 0 results (2026-09-21 ~19:30 UTC)
- First submission failed at start (script-ordering bug, 0 GPU-h); fixed, resubmitted as c10m-002/003.
- Throughput: ~1.1 s per optimizer step (global 256, mb32 x acc2, compiled) => ~4.7 GPU-h per 1M images;
  a full 10M pass would be ~48 GPU-h (~12 h wall). Cheap enough for many screens.
- Root anchor (n=10k, seed 0): EMA 1286.45, raw 1276.86.
- Reproducibility: mean |diff| 0.044/255 across batch/device configurations.

### Cycle 1 — wave-1 L1 screens (submitted ~19:35 UTC)
All from the common root, 6000 steps (1.536M images), fresh AdamW, warmup 250, EMA 0.9995 (short-run EMA),
same data order/seed; each evaluated at step 6000 with EMA and raw weights (n=10k).
- c10m-010-ctrl: control (const LR 1e-4).
- c10m-011-cooldown: linear LR decay to 0 over last 3000 steps. (H-cooldown)
- c10m-012-cropc: center-crop training transform. (H-crop)
- c10m-013-nullp0: null_condition_p 0. (H-nullp)
- c10m-014-mg05: guidance-target distillation from the frozen root EMA teacher, lam 0.5. (H-mg)
- c10m-015-diag-cfg-noise: seed-noise replicate of root FD; diagnostic-only CFG sampling w=1.5/2/3.

### Cycle 1 results (2026-09-21 ~22:30 UTC) — FD n=10k, step 6000 (1.536M images)
| arm | EMA 0.9995 | raw | dEMA vs ctrl |
|---|---|---|---|
| root (step 0) | 1286.45 (seed1: 1286.03) | 1276.86 | |
| ctrl | 1033.32 | 1092.64 | 0 |
| cooldown | 1024.96 | 1060.72 | -8.4 |
| center crop | 1020.26 | 1065.91 | -13.1 |
| null_p 0 | 1019.57 | 1064.24 | -13.8 |
| MG lam 0.5 | 924.30 | 998.34 | -109.0 |
Diagnostic-only CFG on root: w1.5 1170.7, w2 1092.2, w3 989.6 (never submitted; sizes the guidance-target objective).
Observations: (1) continuing training helps a lot (-253 in 1.5M images) — the root is far from converged;
(2) EMA beats raw by 40-70 everywhere; (3) sampling noise ~0.4 FD, training noise unknown;
(4) MG is the dominant lever. Throughput: 0.47 s/step (ctrl), 0.60 s/step (MG, +2 teacher fwd);
cropc ran 1.4 s/step due to node contention (ledger GPU-h truthful, but inflated).
A full 10M pass costs only ~20-25 GPU-h => budget is not binding; wall-clock (96 h) is.

### Cycle 2 — wave 2 (submitted ~22:35 UTC)
- c10m-020-mg10 / c10m-021-mg20: MG lam 1.0 / 2.0 (w_eff 2 / 3).
- c10m-022-ctrl-seed1: control replicate with seed_everything 1 (training-noise estimate).
- c10m-023-stack3: cooldown + center crop + null_p 0 (additivity of the small effects).
- c10m-024-mg10-stack3: MG lam 1.0 + stack3 (candidate).
- c10m-025-ctrl-long: control to 18000 steps, eval at 12k/18k (data-scaling slope).

### Cycle 2 results (2026-09-22 ~06:40 UTC) — FD n=10k EMA, step 6000 unless noted
| arm | FD | note |
|---|---|---|
| ctrl seed0 / seed1 | 1033.32 / 1029.99 | training noise ~±2-3 |
| stack3 (cooldown+crop+null_p0) | 1013.11 | -18.5 vs ctrl mean; sub-additive but real |
| MG lam 1.0 | 858.91 | -65 vs lam 0.5 |
| MG lam 1.0 + stack3 | 849.34 | stack still helps on MG (-9.6) |
| ctrl long 12k / 18k | 970.02 / 918.33 | ~-55 per 1.5M images, far from saturated |
| MG lam 2.0 | failed | EADDRINUSE master port; fixed via bind(0) port |

Decision (L3 promotion): MG+stack3 is -180 vs matched control at 1.5M images with noise ~±3, and data
scaling of the control is steep => a full-pass (38000 steps = 9.73M images, ~7 h, ~30 GPU-h = 0.5% budget)
candidate plus a matched full-pass pristine control are justified. Intermediate evals at 19k/28.5k/38k.
Expected: both improve with data; candidate retains a large gap. Risk: frozen-root teacher guidance
direction may become stale as the student improves (compare lam sweep + EMA-teacher variant).

### Cycle 3 — wave 3 (submitted ~06:50 UTC)
- c10m-030-full-mg10s3 (L3 candidate), c10m-031-full-ctrl (L3 matched control), EMA 0.9998, 38000 steps.
- c10m-032-mg20s3, c10m-033-mg30s3: lam 2 / 3 with stack3 (6000 steps).
- c10m-034-mgema05cc: EMA-teacher (self-referential) lam 0.5 + cooldown + crop (null_p 0.1 kept, needed).
- Infra: no-replay resume skip (resume_skip_batches), free-port rendezvous.
