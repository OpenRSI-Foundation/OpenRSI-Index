# From pretrained language models to Jev-style System One decisions

Status: research proposal. This PR defines an adaptation study; it does not add
a runnable benchmark task or report new training results.

## Research question and scope

Given a fixed pretrained language model, how should we construct data and train
it to make fast, structured decisions with useful probabilities? Study the
complete conversion procedure: data construction, supervised adaptation,
distillation, reinforcement learning where feedback warrants it, and calibration.
Measure which stages improve transfer and which merely add data or compute.

The target is **Jev-style System One behavior**: a state and typed questions go
in; binary, categorical, or ordinal probability distributions come out, without
autoregressive reasoning or answer generation at deployment. TypeSafe describes
this interface in its [System One documentation][system-one] and calls its
training direction [reinforcement learning for calibrated decisions (RLCD)][rlcd].
The reviewed public documentation does not specify a reproducible RLCD training
recipe. The algorithms below are proposed experiments, not claims about Jev's
internal implementation.

Use [Kev at a pinned revision][kev] as the open experimental starting point.
The existing [decision architecture task](rsi-tasks/kev-decision-architecture/README.md)
already allows changes to heads, representations, adapters, losses and schedules,
but fixes the training corpus and excludes teacher outputs and additional data.
This proposal opens **the data and adaptation procedure** under an explicit
budget. Initially keep the decision architecture fixed so that data, curriculum
and RL effects can be identified. Architecture changes belong in separate
ablations.

The study has two connected lanes:

| Lane | Output meaning | Main question |
| --- | --- | --- |
| A: calibrated judgments | Probability of an answer or ordered outcome | Which conversion recipe best retains knowledge and generalizes to new decisions? |
| B: sequential action selection | Policy probability over currently legal actions | Does interaction feedback improve task return beyond imitation and additional supervised data? |

Lane A is the first deliverable. Lane B is a planned extension with its own
environment and evaluation contract; completing A does not validate B or RLCD.

## Fixed starting point and deployment contract

For the first controlled study, reuse the existing task's original
`Qwen/Qwen2.5-0.5B` checkpoint at
`060db6499f32faf8b98477b0a26969ef7d8b9987`, tokenizer, and Kev source at
`c096660c8da20a80ce7c61c63d224960f497623a`. This older base is a reproducible
comparison anchor, not a claim that it is the best current model. Every
independent recipe starts from that original base and a seeded new head; later
stages within a recipe inherit its checkpoints. Scaling to a newer or larger
base is a separate experiment.

The conversion is a learned readout and adaptation of the pretrained backbone:

```text
state + question + candidate descriptions
                 |
      pretrained backbone + adaptation parameters
                 |
        option-conditioned decision head
                 |
       logits -> normalized probabilities
```

The pinned [Kev model][kev-model] supplies the decision-head implementation.
Start with its head and LoRA configuration from the existing task: head width
256, LoRA rank 16. Compare joint head/LoRA training with head-only training and
head warm-up followed by joint adaptation. Do not assume warm-up is superior.

- Noul: return probabilities for `false` and `true`.
- Choice: return probabilities keyed by the supplied candidate identifiers.
- Score: return a distribution over the supplied ordered levels; its expected
  level is a derived score, not a substitute for the distribution.
- Question branches may share the state representation but must not depend on
  sibling questions, their order, previous requests, or training-only metadata.
- Probabilities must be finite, nonnegative and sum to one. Test Choice option
  permutation and arbitrary identifier changes after mapping outputs back to
  their meanings. Preserve ordinal order when testing Score.
- Deployment performs one bounded decision computation per state/question
  batch, with no generated chain of thought, search, teacher call or test-time
  parameter update. Measure actual latency; this contract alone does not prove
  a speedup.

The existing [predictor entrypoint](rsi-tasks/kev-decision-architecture/environment/reference/entrypoint.py)
is the initial inference interface. Retain its separation between model inputs
and labels. Reusing it does not make newly constructed data legal in the old
benchmark: the proposed study needs its own training and evaluation assets.

## 1. Construct decision data before changing training

Start with the existing decision-v7 corpus, pinned in the
[task asset manifest](rsi-tasks/kev-decision-architecture/README.md#assets-and-environment),
as the reference training set. Added sources must be versioned and permitted for
the study. Freeze their inclusion rules before reading locked test outcomes.

Build three explicit data components:

1. **Natural labeled decisions:** task routing, entailment, evidence-based
   questions and ordered judgments. Convert each source to a shared request
   schema without hiding labels in the input. Retain source-specific metrics.
2. **Executable synthetic decisions:** generate states and policy/rule questions
   with independently computed answers. Vary paraphrases, distractors, evidence
   location, option count and compositional difficulty. Keep the solver and
   generator on the training side; never call them from the submitted predictor.
3. **Teacher-assisted decisions:** generate or relabel training-only examples,
   including hard negatives and soft targets. Record teacher version, prompts,
   sampling parameters, verification method and cost. A teacher answer is not
   automatically ground truth. Audit disagreement and reject examples that
   cannot be validated for the intended target semantics.

Use hard-negative mining on training errors. Development data can select a
recipe, but must not be copied, paraphrased or teacher-labeled into training.
Teacher rationales can help offline verification; they are not student input or
required deployment output. An uncertainty case needs an explicit unknown option,
an adjudicated distribution or a diagnostic-only designation. Missing evidence
does not by itself justify inventing a uniform target.

Split by parent document, entity, conversation or episode **before** creating
variants. For synthetic transfer, also hold out rule/template families, not just
random seeds. Deduplicate exact and normalized examples and audit semantic
near-duplicates. Keep related examples in one split. Record that public data may
have appeared in pretraining; split hygiene cannot establish its absence.
For strict unseen-source or unseen-rule evaluation, keep those families out of
train, development and calibration. Source-specific calibration is a separate
adaptation setting and must be reported as such.

Use four disjoint partitions: train, development for recipe selection,
calibration for final probability/threshold fitting, and locked test. The final
calibration partition is opened only after selecting the recipe; repeated
calibration-method tuning needs a development-only inner split.

Minimum artifact contract:

| Artifact | Required information |
| --- | --- |
| Dataset manifest | Source/revision/license, schema version, checksums, split and exclusion rules, generator version/seed, parent-group assignment |
| Labeled decision | State, typed question, candidate meanings/keys, hard label or validated soft distribution, source and parent ID, annotation provenance |
| Derived example | Parent ID, transformation, updated label mapping, verifier result, generation/teacher cost |
| Interactive transition | Episode/step ID, causal observation/history, legal actions, selected action, reward, next observation, termination and truncation flags, behavior-policy version and selected-action log probability |

Dataset, parent and episode IDs, labels, provenance, future observations and
returns stay outside the inference request. Preserve the question/candidate keys
required by the API, without encoding labels or source identity in them.
Log every filtered example and reason; do not silently shorten
evidence or discard difficult examples to fit the context window. Initially
preserve the reference context limits, then study longer contexts separately.

## 2. Convert the backbone with supervised adaptation

The starting comparison is the task-owned two-epoch Kev recipe on the unchanged
corpus. Reproduce it with the new study's evaluation before claiming improvement;
the old task's published result is not a matched baseline for expanded data.

For answer probabilities `p_theta(y | state, question, options)`, use a proper
probability objective: categorical cross-entropy for Choice and Score, binary
cross-entropy or the equivalent two-class loss for Noul. Compare an additional
ranked probability score for ordinal levels. Weight sources and question types
explicitly so records containing many sibling questions do not dominate by
accident.

An experimental combined objective is:

```text
L = L_label + lambda_distill * KL(q_teacher || p_theta)
            + lambda_order * L_ordinal
            + lambda_perm * L_permutation
            + lambda_retain * L_retention
```

All extra terms are optional ablations. The pinned
[reference trainer](rsi-tasks/kev-decision-architecture/environment/reference/kev/train.py)
already contains hard/soft-label cross-entropy, optional ordinal scoring,
teacher-anchor KL and permutation consistency. These code paths are useful
starting points, not evidence that every combination improves transfer. Avoid
double-counting identical teacher soft targets as two independent signals.

Compare head-only training, joint LoRA/head training, and staged unfreezing with
the same data and budget. Limit any full-backbone adaptation to a separately
declared arm. Freeze the attention/packing semantics for the main comparisons;
changing them simultaneously would mix architecture and training effects.

For retention, compare training-data replay against anchoring on distributions
from the original frozen base. A base anchor can preserve mistakes as well as
knowledge, so measure retention on independent held-out questions. Keep the
teacher's option mapping and probability normalization explicit; verbal
confidence or sampled answer frequency is not a calibrated soft target by fiat.
[Knowledge distillation][distillation] motivates the soft-target comparison,
but does not establish the best teacher or mixture for this setting.

## 3. Add RL only with an explicit feedback problem

Fully labeled static questions already support direct cross-entropy or Brier
optimization. Calling their negative loss a reward does not create a distinct
RL experiment. Use the following separation:

| Feedback available | Proposed method | Required control |
| --- | --- | --- |
| Full answer/target distribution | Supervised proper-loss training and distillation | Same data and model with the reference supervised recipe |
| Outcome only for the chosen action | Contextual bandit policy learning | Matched feedback budget; behavior propensities and action coverage |
| Actions affect later observations and reward | Behavior cloning followed by trajectory RL | BC-only and learner-state relabeling under matched interaction budgets |

For Lane B, a concrete candidate is the pinned OpenTinker ALFWorld text
environment. Its [game adapter][alfworld-game] exposes admissible commands that
can become Choice candidates. Supply only observable history and legal commands,
never latent simulator state, future observations or task-success labels. Freeze
the history/context policy and test for action-list truncation before training.
Reset episode history between games. Freeze task-family splits and an environment
interaction cap before collecting trajectories.

Proposed sequential pipeline:

1. Collect train-split demonstrations with a fixed, cost-accounted teacher or
   expert; store failed and partial episodes as well as successes.
2. Warm-start the categorical action head with behavior cloning.
3. Collect student rollouts, recording selected-action log probabilities and
   episode boundaries. Compare expert relabeling of learner-visited states
   ([DAgger][dagger]) against interaction-based policy updates.
4. As an initial RL baseline, implement categorical-action [PPO][ppo] with a
   training-time value head. Compute ratios from action probabilities, not
   language-token log probabilities. Version rollouts, mask illegal actions,
   handle truncation/bootstrap correctly, and keep the old policy fixed for
   each update batch.
5. Evaluate the saved policy on held-out episodes with no teacher, learning,
   search or reward access during action selection.

OpenTinker's [current training service][opentinker-trainer] operates on generated
token sequences, response masks and token log probabilities. Its environment
services are reusable, but a decision-head rollout/update adapter must be built
and validated. A head swap is not an implemented RL integration.

Start with environment-defined rewards. Treat shaped rewards as separate
experiments with a documented relation to task success; do not reward confidence
as a substitute for correctness. If studying offline bandits, record the
behavior propensity and diagnose support before using importance weighting or
[doubly robust evaluation][dr]. Missing action coverage cannot be repaired by an
unqualified off-policy estimate. Fresh held-out simulator episodes are the
preferred final evaluation for the sequential lane.

### Action policy and calibrated belief are different quantities

`pi(action | state)` describes how a policy selects actions. Optimizing task
return does not make it a calibrated estimate that each action is correct.
When a calibrated judgment distribution is the product output, train
`p(outcome | state)` with a proper scoring rule and derive actions through an
explicit utility function:

```text
chosen_action = argmax_action sum_outcome p(outcome | state) * U(action, outcome)
```

This expression assumes the predicted outcome is action-independent, such as a
ticket's category. If actions change the outcome, the corresponding model is
`p(outcome | state, action)` and needs action-conditioned evidence.

This is a proposed experimental design, not a description of Jev internals.
Keep policy probabilities and judgment probabilities separately named and
evaluated. If they share a backbone, measure interference and retention after RL.
If RL changes the same output head directly, report reward and calibration
separately; do not rename policy propensities as calibrated answer confidence.

## 4. Calibrate, select and evaluate

After recipe selection, fit a simple temperature on the reserved calibration
partition using NLL, following the [temperature-scaling baseline][calibration].
Compare raw and calibrated distributions; calibration can change probability
quality without changing the argmax, and cannot repair missing knowledge or
guarantee calibration after distribution shift. Select any abstention threshold
on this partition, freeze it, then measure held-out risk and coverage.

| Dimension | Measurements |
| --- | --- |
| Judgment quality | Accuracy and macro NLL by source and question type; Brier score; ordinal RPS and level error |
| Calibration | Reliability plots, raw/calibrated NLL and Brier, confident errors; ECE as a secondary diagnostic |
| Useful automation | Risk-coverage curve; held-out coverage and error at a calibration-selected error target, including uncertainty |
| Transfer and retention | Held-out source/rule families, changed option counts, paraphrases, distractors and context lengths; untouched knowledge and policy decisions |
| Interactive behavior | Held-out success/return, steps, invalid actions and failure categories, separately from judgment calibration |
| Deployment cost | End-to-end p50/p95 latency, throughput and peak memory including tokenization, packing and output mapping |
| Adaptation cost | Data/teacher generation, verification, training and rollout accelerator-hours, elapsed time, examples/tokens and environment interactions |

For Lane A, preregister source weights and use macro NLL as the primary metric;
report accuracy and calibration alongside it. Retain the existing task's
knowledge/other grouping only if the new source manifest supports that
comparison. For Lane B, use held-out success/return as the primary metric. Do not
combine both lanes into an arbitrary weighted score. Any future benchmark needs
its exact workload, aggregation and resource limits frozen before submission.

Measure latency at fixed hardware, precision, batch size, state length, question
count and option count. Report cold and warmed execution/cache conditions
separately and give the autoregressive baseline the same observable inputs.
Do not infer speed from token counts alone.

Use paired evaluation cases and preregister multiple training seeds (initial
target: three for shortlisted recipes). Resample parent groups, not correlated
variants, for uncertainty estimates. Distinguish training-seed variance from
finite-test uncertainty. Development screens can be cheaper, but locked test
results must not feed the next training-data or hyperparameter iteration.

## Minimum experiment matrix

| Arm | Change from comparison | What it tests |
| --- | --- | --- |
| B0 | Original base with fixed answer likelihood/constrained answer protocol | Knowledge available before conversion; tokenization/length handling must be specified |
| B1 | Frozen backbone, trained decision head | What a new readout alone can recover |
| B2 | Reference Kev LoRA/head supervised recipe | Reproducible conversion baseline |
| D | B2 plus non-teacher constructed/filtered/rebalanced hard-label data | Data quality and coverage |
| G | D plus independently verified teacher-generated examples, using hard labels | Added teacher-generated data; compare an equal-cost non-teacher data arm |
| T | On the identical D or G input/hard-label pool, add teacher soft targets | Distillation with the data pool and available hard labels held fixed |
| R | Selected supervised recipe plus replay/anchor | Retention versus adaptation trade-off |
| I | BC-only versus learner-state relabeling versus BC + RL | Extra interaction data versus the RL update itself, within Lane B |
| C | Selected, frozen judgment models before/after final calibration, with no reselection | Probability correction separate from accuracy gains; shortlist comparisons use development-only calibration |

Use both a fixed-data ablation and a fixed-total-cost comparison. In teacher
experiments, charge teacher inference and label verification; in RL experiments,
charge all rollouts, including failed or discarded ones. Extra supervision is a
reported resource, not a free improvement attributed to the optimizer. Under a
fixed interaction budget, BC/relabeling and RL will visit different states;
report that distinction and, where feasible, add a shared-transition-data control.

## Implementation milestones and compute

1. Freeze the source/evaluation manifests, split rules, inference contract and
   budget; reproduce B0/B1/B2 on development data. Publish the input conversion
   and data-validation rules alongside the results.
2. Implement data construction/provenance and supervised/distillation configs.
   Deliver resumable stage checkpoints, a standalone predictor and an experiment
   ledger linking each checkpoint to its exact data and configuration.
3. Run controlled adaptation and calibration comparisons. Shortlist on development
   data, lock the recipe, then report the final held-out measurements.
4. Build and verify the action-level environment/optimizer adapter for Lane B.
   Validate rollout probability bookkeeping and termination handling, then run
   BC, learner-state relabeling and RL comparisons. Report this lane separately.
5. If contributing an executable OpenRSI task, convert the validated study into
   a separate task package with frozen assets, a baseline, evaluator, resource
   contract and failure behavior through the repository contribution workflow.

Planning target for the initial 0.5B supervised lane: one node with two GPUs;
use 2 x H100 80 GB as a budgeting reference, not a measured requirement. Start
with a 24-hour research budget and reassess after one complete measured cycle
including data preparation, training, evaluation and artifact overhead. The
existing task's runtime is only a feasibility reference for its fixed recipe;
it does not establish capacity for new data, teachers or RL. Lane B and any
larger teacher require their own measured budget before execution.

This PR launches no GPU training, teacher API calls or environment collection.
Runtime, achievable loop count and performance gains for the proposed study
remain unmeasured. The immediate deliverable is this reviewable research design;
new experiment results and a runnable benchmark are subsequent milestones.

## Evidence

Public documentation describes the desired interface; pinned source establishes
what the open baseline actually implements. Recommendations and study choices
above are hypotheses to test.

[system-one]: https://docs.typesafe.ai/concepts/system-one
[rlcd]: https://docs.typesafe.ai/introduction/machine-learning-primer
[kev]: https://github.com/jaredpalmer/kev/tree/c096660c8da20a80ce7c61c63d224960f497623a
[kev-model]: https://github.com/jaredpalmer/kev/blob/c096660c8da20a80ce7c61c63d224960f497623a/kev/model.py
[distillation]: https://arxiv.org/abs/1503.02531
[calibration]: https://proceedings.mlr.press/v70/guo17a.html
[dagger]: https://proceedings.mlr.press/v15/ross11a.html
[ppo]: https://arxiv.org/abs/1707.06347
[dr]: https://arxiv.org/abs/1103.4601
[alfworld-game]: https://github.com/open-tinker/OpenTinker/blob/f87fe25fc483e7f1d7596182fc55115cbc992724/opentinker/environment/alfworld/alfworld_game.py
[opentinker-trainer]: https://github.com/open-tinker/OpenTinker/blob/f87fe25fc483e7f1d7596182fc55115cbc992724/opentinker/server/http_training_server.py
