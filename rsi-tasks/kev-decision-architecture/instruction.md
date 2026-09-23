# Small-model decision architecture

Improve the quality and calibration of a 0.5B-scale neural decision model by
jointly researching its architecture and training method.

## Goal

Use a fixed training corpus to improve transfer to new domains and harder
decisions. Study representations, decision heads, option interactions and
backbone adaptation, not just learning-rate tuning. Return probabilities for
binary, multiple-choice and ordinal questions without autoregressive generation.

## Workspace

| Item | Location |
| --- | --- |
| Work directory | `/workspace` |
| Editable implementation | `/workspace/candidate/kev/`, `entrypoint.py`, `train.py` and other files inside `/workspace/candidate/` |
| Original base and tokenizer | `/opt/kev-assets/base/` (Qwen2.5-0.5B) |
| Fixed training corpus | `/opt/kev-assets/train/train.jsonl` (12,576 records / 15,576 questions) |
| Public development data | `/opt/kev-assets/public/development.jsonl` |
| Notes, backups, alternate trials | `/workspace/experiments/` |
| Submitted checkpoint | `/workspace/candidate/checkpoint/`, including `manifest.json` |

## Reference baseline

| Item | Value |
| --- | --- |
| Baseline | Kev pointer head + LoRA, original Qwen base, random head, two epochs, rank 16, head width 256; task-owned two-GPU DDP adaptation, no released Kev checkpoint |
| Reported result | Not reported for this exact base/data/evaluation combination |
| Status | Not yet trained or measured for this task |
| Comparison | Train and evaluate the original implementation once; later submissions score only the candidate, never a live baseline pair |

Restoring reference code does not create trained weights.

## Research loop

1. Inspect the starting model and form a falsifiable architecture/training hypothesis.
2. Modify candidate code and train in Work, starting each independent experiment
   from the fixed original base. For the reference run:
   `python /workspace/candidate/train.py`.
   The launcher starts two synchronized GPU workers, each with batch 4 and
   accumulation 1: global batch 8, unchanged from the single-GPU reference.
   They partition each epoch's records; only rank 0 saves the final checkpoint.
   Existing outputs are never overwritten; archive the previous checkpoint in
   `/workspace/experiments/` before a new run.
3. Evaluate on public development data with
   `python /opt/kev-runtime/evaluate_public.py --candidate /workspace/candidate`.
   Public development examples may guide selection but are not training data.
4. Run `validate-candidate --json`, then `rsi-submit`. Inspect aggregate feedback
   and use `rsi-submit --list` to compare experiments. Keep research notes.

## What you may change

All implementation and training choices inside `/workspace/candidate/` are open:
head structure, learned pooling, state/option fusion, cross-option attention,
adapters, bounded backbone edits, losses, optimization, sampling and schedules.
New modules may be randomly initialized; retained pretrained parameters must
come from the supplied original base. Same-experiment training may resume.
Keep the fixed tokenizer and vocabulary; no other pretrained model or teacher.

The predictor interface is `entrypoint.load(checkpoint_dir, device)`, returning
an object with a PyTorch `.model` and `.predict(request)`. Requests contain only
`state` and `questions`; questions have `type`, `instructions` and, when applicable,
`criteria`. Return `{"probabilities": {question_id: {option_key: probability}}}`.
Binary keys are `false`/`true`; choice keys are the supplied criteria keys;
ordinal keys are zero-based strings. Each distribution must be finite,
nonnegative and sum to one within 1e-5. Question siblings, their order and prior
requests must not change a question's probabilities beyond 1e-3 absolute error.
All learned inference parameters must be registered in `.model`.

## What stays fixed

- Train only on the fixed training records. Do not add external, development or
  test examples, teacher outputs or memorized evaluation answers. Reweighting
  and resampling the fixed records are allowed; the reference uses every record.
- No released Kev adapter, alternative pretrained base, external API, answer
  lookup, hardcoded solver for evaluation families, or metadata-based answer
  routing. Predictions must come from the trained neural model, and every
  probability must be produced by the Kev decision head.
- Edit only `/workspace/candidate/` and `/workspace/experiments/`. Keep `/opt/`,
  runtime dependencies, fixed assets, evaluator and Harness unchanged. Do not
  inspect reserved tests, parent-process memory or hidden outputs.
- At most 600 million loaded registered parameters; checkpoint tree at most
  3 GiB; peak Torch-allocated inference memory at most 16 GiB. Work uses two
  GPUs; Judge uses one. The run budget is 12 hours by default.
- Judge loads once, receives one request at a time and performs inference only.
  Its worker budget is 20 minutes total, including at most 180 seconds to load
  and 45 seconds per request; the outer verifier timeout is 30 minutes.

## Evaluation and feedback

The fixed workload includes 500 MMLU and 500 MMLU-Pro questions, plus 958 other
scored decisions from transfer-v9 and fresh task-owned rule, date and routing
instances. A further 110 evidence-removed requests are diagnostic-only. It covers
linguistic transfer, knowledge, buried evidence, composition and ordinal reasoning.
The other group includes held-out/transfer tasks, not only in-domain tasks. Published
upstream cases are already public; this is not a claim of fully secret or
pretraining-decontaminated evaluation. Fresh instances and all per-case Judge
results remain reserved.

For each source, compute mean negative log probability of the correct answer
with probabilities floored at 1e-9. Average MMLU and MMLU-Pro source means to
form the knowledge group; separately average the other 12 source means.
`macro_nll = 0.5 * knowledge_nll + 0.5 * other_nll`, then maximize
`reward = exp(-macro_nll)`. Thus each knowledge source carries 25% of the total
NLL weight; the other sources share 50%. Reward lies in [0, 1]. Unknown-evidence
cases have no meaningful correctness label: they contribute confidence
diagnostics only, never NLL. Aggregate accuracy and Brier score use the same
group weights; ordinal error is secondary. The public evaluator shares this
scorer. Partial smoke results renormalize over groups actually present and
explicitly report incomplete group coverage; they are not formal Judge scores.

Feedback contains aggregate source metrics, timing/memory and actionable
candidate-owned file/configuration errors. It excludes answers, per-example
results and fresh case text. Candidate prints during hidden inference are not
returned because they may reproduce reserved inputs. Prediction errors report
fixed safe categories and the candidate entrypoint; arbitrary exception/frame
strings are withheld too. Model-loading logs and diagnostics remain complete,
before any hidden inputs are sent. Use the public evaluator for unrestricted debugging.
Declared invalid artifacts, outputs or candidate execution errors receive reward zero. Infrastructure
failure, timeout or incomplete execution yields no reward, not a quality score.

## Submission checklist

- Training has stopped and all checkpoint files are flushed. Judge needs no live
  Work process and will never train missing weights. Release both Work GPUs
  before submission so Judge can reuse one of them.
- `manifest.json` records the original base identity, fixed training corpus,
  training configuration and relative checkpoint file list; the reference
  launcher demonstrates the format. It is self-reported provenance, not proof.
- `validate-candidate --json` passes. This checks files and manifest only;
  run public inference as well to check the model behavior.
- Candidate source/checkpoint files are readable by the Judge's unprivileged
  inference process. No symlinks in checkpoints and no files outside editable
  workspace directories.
