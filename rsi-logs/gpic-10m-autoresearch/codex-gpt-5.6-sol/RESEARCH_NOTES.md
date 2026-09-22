# Codex GPIC 10M autoresearch

Trajectory start: 2026-09-21 UTC. This is an independent trajectory rooted only in the common immutable weights checkpoint and shared immutable 10M-subset manifest.

## L0: root-load and optimizer sanity probe

- Attempt: `l0-rootload-5step`
- Hypothesis: the common weights-only checkpoint can initialize both live and EMA denoisers while a fresh AdamW/scheduler state completes five optimizer updates on 4 H100s without replay or checkpoint-loading errors.
- Expected quality effect: none claimed; this is a systems/mechanistic gate before matched quality screens.
- Start: `/shared/gpic-output/baseline_16x8/exp_pretraining_jit_256_gpic_full/epoch=0-step=39060.ckpt` (weights-only; live denoiser initialization, fresh optimizer/scheduler).
- Data: fixed `train10m_seed20260921` shards; effective batch 256; maximum 1,280 image exposures.
- Topology: P3, 1 node x 4 H100, 30-minute segment.
- Estimated upper-bound cost: 2 H100-hours; expected cost is much lower.
- Promotion gate: verify successful terminal state, five executed optimizer steps, finite loss, and a checkpoint containing live and EMA tensors. No quality conclusion will be drawn from loss.

Result: failed before checkpoint loading or data consumption. Slurm job 2416986 ran 23 seconds on 4 H100s (0.0256 H100-hours). The offline conditioner attempted to resolve `Qwen/Qwen3-1.7B` instead of the staged allowlisted directory. Retry `l0-rootload-5step-r1` changes only the conditioner path to `/opt/hf/Qwen3-1.7B`.

The `r1` retry also failed before data use because the container path is not mounted verbatim on the Slurm host (0.0167 H100-hours). Infrastructure-translated retry `l0-rootload-5step-v2` succeeded in 0.3556 H100-hours. Its `last.ckpt` is at global step 5 with optimizer/scheduler state and both live and EMA denoiser tensors; it consumed 1,280 images. Final logged flow-matching loss was finite (0.102), which is only a systems check.

## L1: matched 1,000-update control

- Attempt: `l1-control-drop01-1000`
- Hypothesis: 1,000 baseline-recipe continuation updates from the common root provide an inexpensive, stable matched quality control for testing whether conditioning dropout is wasteful at fixed guidance 1.0.
- Start: immutable common weights; fresh AdamW at 1e-4 and flat schedule.
- Data: fixed shared subset, effective batch 256, exactly 256,000 maximum image exposures.
- Topology: P3, 1 node x 4 H100, 3-hour segment; attempt cap 12 H100-hours.
- Matched candidate planned: identical run with `null_condition_p=0.0`.
- Quality gate: generate identical deterministic caption prompts from root, control, and candidate checkpoints, then compare permitted val-reference FD before promotion.

Result: completed exactly 1,000 updates in 45:01, charged 3.0011 H100-hours, and consumed 256,000 images. The checkpoint reports global step 1000 and contains live/EMA tensors and optimizer state. Final loss 0.067 was finite; quality remains unevaluated. Proceeding to the matched conditional-only ablation before identical generation/evaluation.

## L1: conditional-only matched ablation

- Attempt: `l1-candidate-drop00-1000`
- Falsifiable hypothesis: because evaluation is strictly guidance 1.0, replacing 10% null-conditioning dropout with fully conditional training gives every update the inference-time conditioning distribution and will lower matched FD after 1,000 updates.
- Single changed variable: `null_condition_p`, 0.1 to 0.0.
- All other initialization, data order/seed, effective batch, optimizer, schedule, duration, and topology match `l1-control-drop01-1000`.
- Planned exposure: 256,000 images; estimated cost based on control: about 3.0 H100-hours.

Result: completed exactly 1,000 updates in 41:00, charged 2.7333 H100-hours, with 256,000 exposures. Final loss was lower than control, but no quality conclusion is drawn from loss. Next gate is generation correctness and matched FD.

## L0: generation-contract smoke

- Attempt: `l0-generation-smoke`.
- Hypothesis: the candidate EMA can generate valid 256x256 RGB PNGs deterministically from the frozen caption prompts using exactly one conditional denoiser stream per Euler step.
- Four prompts, seed 0, 50 Euler steps, two independent invocations, 4 H100s; exact PNG hashes must match.
- `GPIC_GENERATE_LIMIT` is an internal smoke-only limiter; absent this environment variable, the frozen CLI emits one image for every caption.

Result: passed in 0.1200 H100-hours. All four images from two independent invocations were byte-identical and valid 256x256 RGB PNGs.

## L1: paired 4,096-image FD screen

- Attempt: `l1-fd-screen-4096`.
- Hypothesis: the conditional-only checkpoint has lower val-reference FD than the matched dropout-0.1 control after equal 1,000-update training.
- Both checkpoints use the same first 4,096 frozen generation prompts, per-row seed 0, EMA weights, batch size 2/device, 50 Euler steps, guidance 1.0, and pure conditional single-stream inference.
- FD-DINOv2 is used only through the permitted frozen local evaluation tool for run triage, never as a gradient, data filter, or automated optimization target.
- The 4,096-sample estimate is explicitly noisy; a promising difference must be replicated or evaluated at larger N before substantial promotion.

Result: candidate FD 1287.3553; matched control FD 1289.7331. The candidate is lower by 2.3779 (0.184%). Both sets contain exactly 4,096 images. This is promising but too small for promotion from one seed, so the next experiment is an exact seed-1 replication.

## L2: seed replication of paired FD screen

- Attempt: `l2-fd-replication-4096-seed1`.
- Same checkpoints, first 4,096 frozen prompts, batch geometry, sampler, and evaluation as the L1 screen; only generation seed changes from 0 to 1.
- Decision rule declared in advance: consistent candidate improvement promotes to a larger-N confirmation; reversal rejects or weakens the null-dropout hypothesis.

Result: replicated. Candidate FD 1281.4701 versus control 1283.7857, an improvement of 2.3156 (0.180%). The two independent 4,096-image seeds agree closely (mean advantage 2.3467), satisfying the predeclared promotion rule.

## L2: 16,384-image fresh-seed confirmation

- Attempt: `l2-fd-confirm-16384-seed2`.
- Hypothesis: the candidate's approximately 0.18% matched FD advantage persists with four times more samples and a fresh seed.
- Same checkpoints and protocol, first 16,384 frozen prompts, seed 2.
- Promotion rule: consistent lower candidate FD supports carrying conditional-only training into a longer matched stage; loss of direction rejects the small effect as insufficiently robust.

Result: confirmed. Candidate FD 1242.9012 versus control 1245.9152, an improvement of 3.0140 (0.242%) on 16,384 images. Conditional-only training is retained as the new research anchor.

## L2: faster-EMA ablation

- Attempt: `l2-candidate-drop00-ema999-1000`.
- Falsifiable hypothesis: for a 1,000-update continuation, EMA decay 0.999 will lower FD versus the matched 0.9999 conditional-only anchor because it incorporates approximately 63% rather than 9.5% new-model mass.
- Single changed variable versus `l1-candidate-drop00-1000`: EMA decay 0.9999 to 0.999.
- Same common root, fixed data subset/order, null conditioning 0.0, 1,000 updates, effective batch 256, optimizer, schedule, and 1x4 H100 topology.
- Planned exposure: 256,000 images; estimated cost about 3 H100-hours.

Result: completed exactly 1,000 updates and 256,000 exposures for 2.9389 H100-hours. Checkpoint state is complete. Proceeding to a paired 4,096-image screen against the existing conditional-only EMA-0.9999 anchor.

## L2: faster-EMA paired FD screen

- Attempt: `l2-fd-ema999-vs-ema9999-4096`.
- Candidate: conditional-only, EMA 0.999; control: matched conditional-only, EMA 0.9999.
- First 4,096 frozen prompts, seed 3, identical pure-conditional 50-step sampling and val-reference evaluation.
- Decision rule: a material improvement advances EMA 0.999 to replication; regression rejects it. A very small improvement is replicated once before promotion.

Result: material improvement. EMA-0.999 FD 1176.6009 versus EMA-0.9999 FD 1283.7130, lower by 107.1122 (8.344%). This is large enough to advance, but an independent seed is required before longer training.

## L2: faster-EMA seed replication

- Attempt: `l2-fd-ema999-replication-4096-seed4`.
- Identical checkpoints and protocol to the seed-3 screen; only generation seed changes to 4.
- Consistent material improvement promotes EMA 0.999 to larger-N confirmation and longer training.

Result: replicated. EMA-0.999 FD 1189.4820 versus EMA-0.9999 FD 1298.3544, lower by 108.8724 (8.386%). Seeds 3 and 4 closely agree, promoting decay 0.999.

## L3: 4,000-update promoted recipe

- Attempt: `l3-drop00-ema999-4000`.
- Hypothesis: extending the promoted conditional-only/EMA-0.999 recipe from 1,000 to 4,000 updates from the common root will further lower FD.
- Fixed subset, one run with no replay, effective batch 256, 1,024,000 maximum unique exposures.
- Fresh AdamW lr 1e-4, flat schedule, 1x4 H100 P3 topology.
- Estimated cost: about 12 H100-hours based on measured 1,000-step throughput.
- Quality gate: evaluate immediately against the promoted 1,000-step EMA-0.999 checkpoint; do not extend again without new FD evidence.

Result: completed exactly 4,000 updates and 1,024,000 exposures in 2:37:38, charged 10.5089 H100-hours. Checkpoint state is complete. Proceeding immediately to paired FD evaluation.

## L3: 4,000-vs-1,000 update paired FD screen

- Attempt: `l3-fd-4000-vs-1000-4096`.
- Candidate and anchor share conditional-only training, EMA 0.999, initialization, optimizer, and sampling; training duration is the only model difference.
- First 4,096 frozen prompts, seed 5, identical 50-step pure-conditional generation and val-reference FD.
- Improvement promotes the 4,000-step checkpoint; regression stops duration scaling and retains the 1,000-step checkpoint.

Result: material improvement. The 4,000-step FD is 1102.1357 versus 1192.2768 for the 1,000-step anchor, lower by 90.1411 (7.560%). Proceeding to a 16,384-image fresh-seed confirmation before further training.

## L3: 4,000-step 16,384-image confirmation

- Attempt: `l3-fd-confirm-4000-vs-1000-16384-seed6`.
- Same paired checkpoints and protocol, first 16,384 frozen prompts, fresh seed 6.
- Consistent material improvement promotes the 4,000-step checkpoint and justifies the next training rung.

Result: confirmed. The 4,000-step FD is 1043.3926 versus 1128.5966, lower by 85.2040 (7.550%) on 16,384 images. The 4,000-step checkpoint is promoted.

## L3: 8,000-update promoted recipe

- Attempt: `l3-drop00-ema999-8000`.
- Hypothesis: doubling promoted training from 4,000 to 8,000 updates will further lower FD.
- Common-root fresh run on the fixed subset; effective batch 256; 2,048,000 maximum unique exposures, no replay.
- Same conditional-only objective, EMA 0.999, AdamW lr 1e-4 flat schedule, and 1x4 H100 P3 topology.
- Estimated cost: about 21 H100-hours from measured 4,000-step throughput.
- Quality gate: compare immediately with the confirmed 4,000-step checkpoint before any further extension.
