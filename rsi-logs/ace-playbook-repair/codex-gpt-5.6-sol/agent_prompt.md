# Repair ACE playbooks

Test whether periodic intrinsic inspection and repair of weak ACE playbook bullets improves held-out Formula exact-answer accuracy under a fixed local Qwen protocol. This is the lightweight, single-seed protocol (task version 1.1.0).

## Goal

- Improve over stock sequential ACE by identifying inaccurate, contradictory, redundant, stale, or weakly supported bullets during adaptation.
- Keep the model, data order, single-seed adaptation budget, generator protocol, and hidden evaluation fixed.
- Use aggregate evidence from submissions to decide which inspection and repair hypotheses generalize.

## Workspace

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Starting state | Pinned ACE source, the original public data files (500 training and 300 validation examples), a local Qwen checkpoint, fixed runners, and a scoreable single-repeat unadapted no-op with one canonical empty playbook. The fixed runner uses only the first 200 training and first 100 validation examples in their pinned order. Hidden test data is absent. |
| Deliverable | One closed playbook at `/workspace/submission/runs/repeat-1/best_playbook.txt`, its required audit and adaptation log, and permitted source/configuration changes. Extra repeat directories are invalid. |

## Reference baseline

| Item | Value |
| --- | --- |
| Baseline | Stock sequential ACE from a canonical empty seven-section playbook; seed 0, one epoch over the first 200 training examples, batch size 1, up to three reflection rounds, curator frequency 1, validation on the same first 100 validation examples every 50 training examples, 4,096 output tokens, and an 80,000-token playbook budget. |
| Reported result | Not reported for the fixed local-Qwen Formula protocol. The repository's `+8.6%` is an aggregate FiNER/Formula result under a different protocol. |
| Status | A fully adapted baseline for this lightweight protocol has not yet been measured; the nearby aggregate is source-reported and protocol-mismatched. |
| Comparison | `/usr/local/bin/run-stock-baseline` produces a separate Work-side reference candidate using exactly the same lightweight budget as `/usr/local/bin/run-ace-candidate`. Each Judge submission evaluates only its submitted playbook; no live baseline is run. Do not treat scores from the old three-seed, 500-example protocol as matched comparisons. |

## Research loop

1. Inspect stock source, public data, and prior safe submission feedback.
2. Form one falsifiable inspection/repair hypothesis and change only the allowed candidate surface.
3. Run `/usr/local/bin/run-ace-candidate --output-root /workspace/submission`; the complete 200-example adaptation must finish before submission.
4. Run `/usr/local/bin/validate-ace-candidate --json`, submit the complete snapshot, and compare aggregate evidence.
5. Keep, revise, or reject the hypothesis without using hidden cases or per-example outcomes.

## What you may change

- `/workspace/playbook_utils.py`
- `/workspace/ace/ace.py`, `/workspace/ace/core/reflector.py`, and `/workspace/ace/core/curator.py`
- `/workspace/ace/prompts/reflector.py` and `/workspace/ace/prompts/curator.py`
- New files below `/workspace/ace/inspectors/` and `/workspace/candidate_configs/`
- Research logs and the complete `/workspace/submission/` tree

Inspection is limited to one combined inspect/repair event at each of the four validation checkpoints (50, 100, 150, and 200). The audit manifest is self-reported evidence and must describe every event.

The `adaptation.jsonl` file is bounded self-reported audit evidence, not scoring authority. It must be either the single exact `unadapted_noop` run row with seed 0, zero examples, and no checkpoints, or the fixed 205-row completed schema: one run row, exactly 200 ordered `processed` example rows indexed `0..199`, and exactly four ordered `complete` checkpoint rows for `50,100,150,200`. Every row carries schema version 1 and seed 0.

## What stays fixed

- The baked Qwen 2.5 7B Instruct BF16 model at its fixed revision, static YaRN factor 4, 131,072 context, and 4,096 reserved output tokens.
- Formula training and validation selection and order (fixed first 200/100 rows), seed `0`, one epoch, batch size 1, reflection, curation, and validation budgets, local serving settings, the task-owned maximum-three-attempt request layer, and the generator implementation and prompt.
- Formula processing/scoring, hidden test data, dependency/runtime files, aggregation, and every undeclared workspace path.
- Do not fetch data, use network services, alter fixed assets, read `/tests`, or rely on processes, sockets, caches, or GPU state after `rsi-submit`.

## Evaluation and feedback

| Item | Value |
| --- | --- |
| Fixed workload | The single playbook answers all of the same 200 hidden Formula examples: 200 logical temperature-zero generations, with at most three infrastructure attempts per call and no resampling. The hidden dataset is not reduced. |
| Reward | `correct / 200`, a finite scalar in `[0,1]`, maximized. `mean_accuracy` reports this single accuracy; `sample_standard_deviation` is `null` because there is only one repeat, not evidence of zero variance. |
| Scoreable candidate | One regular, non-symlink UTF-8 playbook with canonical sections and valid bullets, plus complete bounded audit material under `repeat-1`. Judge treats only playbook bytes as prompt data. |
| Visible feedback | Single-repeat accuracy; unavailable sample standard deviation (`null`); separately labeled ACE-budget `cl100k_base` and Qwen model-context playbook token counts; bullet counts; section aggregates; helpful, harmful, and unused aggregates; total unparsable answers; artifact validity; actionable candidate-owned validation errors; and the Harness footer. |
| Hidden | Test inputs/targets, predictions, per-example correctness, raw responses/server logs, detailed Judge logs, evaluator source, and internals. |

Candidate-invalid, timeout, incomplete, infrastructure, and ambiguous failures are unscored and leave no reward. Completed unparsable outputs count as incorrect in the fixed denominator. Use `rsi-submit --list` to inspect prior submission summaries.

This lightweight protocol supports faster iteration but does not measure across-seed robustness. Its shorter adaptation may expose fewer late-stage stale or contradictory bullets than the old protocol; interpret improvements within this budget.

## Submission checklist

- The playbook, `audit.json`, and `adaptation.jsonl` files are complete and flushed below `/workspace/submission/runs/repeat-1/`; there are no other repeat directories.
- All candidate-producing Work commands have finished; Judge performs no adaptation and needs no Work process state.
- `/usr/local/bin/validate-ace-candidate --json` passes; it is advisory and Judge independently checks the snapshot.
- No prohibited state or live process is required by Judge.
- The next submission tests a stated hypothesis.

## Evaluation

The Judge evaluates the entire current WORKDIR as it exists when you submit.
The best valid primary score wins. You may call `rsi-submit` repeatedly to
receive feedback while improving the same workspace.
`rsi-submit --list` shows previous submissions; `rsi-submit --help` shows local usage.
Each submission stores complete Judge stdout and stderr in
`/run/rsi-harness/feedback/agent-N.log`, where N is the submission number.
Judge submissions are unlimited during this run.

Every Work GPU process must exit before rsi-submit. A rejected preflight does not consume a submission.
