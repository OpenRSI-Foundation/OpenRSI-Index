# gpic_generation

**The task.** Start from nothing but the [GPIC corpus][gpic] — 100M
permissively-licensed captioned images — and train, from scratch, any
text-to-image model that beats the pinned [PixelGen][pixelgen] baseline on
FD-DINOv2 over the frozen 50k GPIC test captions. Architecture, parameter
count, and training objective are all free; the data (one full pass, no
more), the 256×256 output resolution, pure conditional sampling
(guidance = 1.0), the per-attempt compute cap (1000 H100-hours,
provisional), and the frozen evaluator are not. Optimizing the evaluation
representation is banned outright: no DINO-family weights or features as
losses, teachers, filters, or initializations, and no DINO-distilled
encoders.

- [`instruction.md`](instruction.md) — the runtime prompt for the research
  agent: what it starts from, what it may change, how it is scored.
- [`task.toml`](task.toml) / [`policy.yaml`](policy.yaml) /
  [`environment/`](environment/) / [`tests/`](tests/) — the Harbor package.
- [`tests/cheat/`](tests/cheat/) — the anti-cheat battery and its recorded
  13/13 result (`battery_results.json`).

**Status: implemented and smoke-tested end-to-end on 1×8 H100
(train → guidance-1.0 sampling → FD-DINOv2); verifier hardened and
adversarially verified — a 13-case cheat battery penalizes every hack vector
0/0 while an honest submission scores (`tests/cheat/battery_results.json`);
baseline unsealed.** The reward anchors (`GPIC_B_FD`, `r_ref`, `r_floor`,
spot-gate thresholds) are provisional until the pristine PixelGen config is
reproduced once under this exact contract (~500–650 H100-hours, one run).

## Task card — why this setting

**Fixed data, fixed epochs, free model.** This task deliberately inverts
the usual benchmark design: instead of fixing the model family and scaling
data, it freezes the data budget (100M images, exactly one pass) and frees
everything else. Two consequences make this scientifically interesting.
First, *scale stops being free lunch*: in the fixed-small-data regime,
unregularized bigger models lose — [NanoGPT Slowrun][slowrun], the closest
LM analogue (100M FineWeb tokens, unlimited compute), found a 1.4B model
beating a 2.7B one outright, saw its leaderboard migrate from 2.7B down to
1.4B across twenty records, and measured a strict size-ordering *inversion*
at the lowest data budgets. An agent cannot win here by requesting a bigger
model; it must find modeling mechanisms that convert one data pass into
quality. Second, *one epoch is the production regime*: real generative
pretraining is effectively single-epoch, and results earned by looping
ImageNet for hundreds of epochs — the dominant academic diffusion setting —
are known not to transfer. A recipe that wins at one pass over 100M images
is a recipe about data efficiency, not memorization.

**Open objective, pixel space.** Unlike LLM pretraining (where
next-token prediction is fixed), the generative objective here is a free
variable — flow matching, diffusion, autoregression, masked prediction,
GANs, hybrids — which is exactly the axis where text-to-image research has
the most unexplored freedom. The pinned baseline is *pixel-space* flow
matching by design: many scientific domains where generative pretraining
matters (astrophysics, microscopy, embryo imaging, world models for
robotics) have no pretrained latent space to lean on, so pixel-space
recipes generalize where VAE-latent recipes do not. Agents may still build
latent models — but the autoencoder must be trained inside the same
data/compute budget, which prices the latent honestly.

**Why FD-DINOv2 and guidance = 1.** FD-DINOv2 over 50k captions is the
GPIC paper's own protocol and the field's current best distribution-level
metric. Guidance is fixed to 1.0 because CFG scale is a sampling-time knob
that moves FD substantially without changing the model; freezing it makes
the metric measure the *model*, and pure conditional quality is also the
honest measure of how well the model learned the conditional distribution.
Both are enforced, not requested: the verifier recomputes FD on the
submitted images against reference statistics the agent never sees, and a
hidden-subset spot regeneration through the agent's own frozen
`generate.py` binds the submitted images to the submitted checkpoint —
images sampled with guidance off-line, cherry-picked, or copied from the
training set fail the gate mechanically.

**Why the DINO ban.** FD-DINOv2 is trivially hackable by optimizing DINO
features directly (perceptual losses, DINO teachers, DINO-initialized
encoders — the pinned repo itself ships an unused DINO-perceptual-loss
trainer). The ban (policy RH-004) covers weights, features, surrogates,
and DINO-distilled encoders, and is enforced by a mechanical pretrained
allowlist (Qwen3-1.7B text encoder only), an LLM policy judge over the
source diff, and no network access; DINOv2 hub weights are staged solely
for the val-screening tool, with any other use a hard-zero violation (an
open alternative is to keep them out entirely and screen with Inception-FID).

**Budgets.** One attempt ≤ 1000 H100-hours (provisional; the pristine
baseline epoch is ~500–650, so the cap admits models a few times larger —
or many times faster). The total iteration budget across attempts is set
by the operators after the first agent trajectories, per the org's SOP.

[gpic]: https://github.com/keshik6/gpic
[pixelgen]: https://github.com/keshik6/gpic/tree/main/baselines/PixelGen
[slowrun]: https://github.com/qlabs-eng/slowrun
