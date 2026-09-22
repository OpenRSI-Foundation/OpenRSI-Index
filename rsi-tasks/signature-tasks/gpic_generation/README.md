# GPIC 10M continuation autoresearch

**The task.** Improve a text-to-image model by continuing from a shared
PixelGen JiT_T2I checkpoint on a fixed, approximately 10M-image GPIC training
subset. Each candidate may consume the subset at most once. Run matched
experiments on **1 node × 4 H100**, generate 256×256 images with pure
conditional sampling (guidance 1.0), and measure FD-DINOv2 against the
validation reference. The research question is which changes to the
continuation objective and training recipe improve quality at equal data
exposure from the same starting weights.

This description follows the published [GPIC 10M research logs][logs],
covering the trajectories started on September 21, 2026. Those runs use
checkpoint continuation, a shared subset, and a smaller per-run topology
than the original 100M-image, from-scratch Harbor task.

## Settings recorded in the logs

| Setting | Logged protocol |
|---|---|
| Source | `keshik6/gpic@afa82daab73cfeab4ef9fbd174ff62b68bffc456`; PixelGen and the pinned `gpic_eval` toolkit |
| Initialization | Common immutable `epoch=0-step=39060.ckpt`, loaded weights-only; fresh optimizer and scheduler for each new candidate |
| Root model | Pixel-space JiT_T2I, **1,122,396,928 denoiser parameters**, with Qwen3-1.7B text conditioning; checkpoint contains live and EMA denoiser weights |
| Root training | 39,060 updates × global batch 256 = **9,999,360 image exposures**; 4,000-step warmup, then flat AdamW LR 1e-4; EMA 0.9999 |
| Continuation data | Shared `train10m_seed20260921` subset, **802 shards**; one pass at most per candidate, no resampling or replay on resume |
| Batch | Effective global batch **256**; Claude records microbatch 32/GPU × accumulation 2 × 4 GPUs |
| Topology | **1 node × 4 H100 on P3 per job**, including training, generation, and evaluation; independent jobs may run concurrently |
| Compute accounting | **6,144 H100-hours per trajectory**, plus a per-job cap recorded in each `status.json`; charge actual GPU time for training, generation, and screening |
| Research window | Claude's research notes specify **96 hours**; the published Codex notes do not specify a wall-clock deadline |
| Output and sampling | 256×256 RGB; default 50-step Euler with EMA weights; guidance **1.0**, one conditional stream per solver stage, deterministic per-caption noise |
| Screening metric | FD-DINOv2 against `val_stats.npz` via `/task-tools/gpic_local_eval.py`, lower is better; caption source and image count differ by trajectory as detailed below |
| Final-generation target | One image per frozen 50k evaluation caption; the published snapshot contains screening results and ongoing runs, not a sealed final 50k test score |

The shared starting checkpoint is recorded as:

```text
/shared/gpic-output/baseline_16x8/exp_pretraining_jit_256_gpic_full/epoch=0-step=39060.ckpt
```

The continuation shards are staged at:

```text
/shared/gpic-data/gpic/subsets/train10m_seed20260921/shards
```

The manifest SHA-256 recorded in the [Codex ledger][codex-ledger] is:

```text
e496f9541370423a50d5448da202f26901986a4a19a4ad8ee43895754406ab55
```

These are paths on the logged Slurm deployment; the checkpoint, shards,
manifest, and deployment-specific launch scripts are not bundled here.
The root's pretraining exposures and each candidate's continuation
exposures are separate quantities. The one-pass constraint applies within
each continuation attempt; independent ablations reuse the common subset.
Count actual exposures rather than assuming that the nominal “10M” subset
or a run named “full” proves an exact completed pass.

## Evaluation and baseline interpretation

The [Claude notes][claude-notes] use validation captions paired with the
50k-image validation reference: `common/val_captions_ref50k.jsonl` contains
49,990 matched captions, and `common/val_captions_screen10k.jsonl` is a
10,000-caption subset selected with seed 20260921. The measured common-root
anchor is **FD 1286.4504** for EMA weights (raw weights: approximately
1276.86), using 10k captions, generation seed 0, 50 Euler steps, and
guidance 1.0. This is a validation screening baseline.

The [Codex notes][codex-notes] and [ledger][codex-ledger] instead record
screens using the first **4,096** or **16,384** prompts from
`/shared/gpic-verifier/gpic_eval_50k.jsonl`, also evaluated against the
validation reference. They use paired candidate/control generation seeds,
followed by seed replication or a larger sample count before promotion.
This caption usage differs from Claude's validation-caption protocol and
the original Harbor prompt's restriction on evaluation-caption tuning.

Compare candidate and control with the same caption set, image count,
reference statistics, seed, and sampling settings. The [trajectory
summary][summary] reports best observed screening FD of 1043.3926 for Codex
and 729.3343 for Claude, but these use different screening protocols and
are not a matched cross-agent comparison. Claude also logged CFG > 1
diagnostics explicitly marked as never submitted; those are not
guidance-1.0 candidate results.

The old Harbor `GPIC_B_FD = 200.0` is a provisional placeholder for a
different, unsealed 100M/50k-test contract. It must not be presented as the
measured baseline of these trajectories, and the 10k validation root FD
must not be substituted into that final-test reward formula.

## Research loop and boundaries

Both trajectories start independently from the common root and shared
subset. Establish a matched continuation control, change an interpretable
variable group, screen quality, and promote only with measured evidence.
The logs explore conditioning dropout, EMA decay, LR schedules, cropping,
training duration, and guidance-target distillation during training.
Training-time teacher use is distinct from sampling-time guidance:
candidate images still use a single conditional stream at guidance 1.0.
DINO weights and features remain reserved for the evaluation tool, with
no use as training losses, teachers, filters, or initializations.

Record initialization, manifest, source revision, actual images seen,
resume state, GPU-hours, caption selection, generation seed, sample count,
FD, and the next decision in `experiments.jsonl`. Keep training loss as a
systems diagnostic; it is not evidence of generation quality. For example,
Claude's promoted “full” runs target 38,000 × 256 = **9,728,000 exposures**
and are still marked `running` in the published snapshot; their names do
not establish completion or a final submission.

## Packaged Harbor implementation

The checked-in [`instruction.md`](instruction.md), legacy metadata and
runtime sections of [`task.toml`](task.toml), [`policy.yaml`](policy.yaml),
[`environment/`](environment/), and [`tests/`](tests/) preserve the original
**100M, from-scratch, 1×8 H100** Harbor implementation. The manifest's task
description and `metadata.logged_protocol` describe the published 10M
trajectories; its legacy sections are explicitly marked. The original
1000-H100-hour per-attempt cap, Qwen-only pretrained allowlist, and
provisional reward anchors are not the settings of the logged continuation
deployment. Its [13-case anti-cheat battery](tests/cheat/) validates that
original package, not the later 10M runs.

This is a task-description update from the logs, not a migration of that
runtime. Reproducing the logged runs also requires their common checkpoint,
subset manifest, host launchers, and matching policy/evaluation setup.

[logs]: ../../../rsi-logs/signature-tasks/gpic-10m-autoresearch/
[codex-notes]: ../../../rsi-logs/signature-tasks/gpic-10m-autoresearch/codex-gpt-5.6-sol/RESEARCH_NOTES.md
[codex-ledger]: ../../../rsi-logs/signature-tasks/gpic-10m-autoresearch/codex-gpt-5.6-sol/experiments.jsonl
[claude-notes]: ../../../rsi-logs/signature-tasks/gpic-10m-autoresearch/claude-code-claude-opus-5/RESEARCH_NOTES.md
[summary]: ../../../rsi-logs/signature-tasks/gpic-10m-autoresearch/summary/traj_progress_summary.md
