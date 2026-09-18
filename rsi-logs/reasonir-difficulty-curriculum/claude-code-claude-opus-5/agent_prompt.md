# ReasonIR difficulty-conditioned curriculum research

Test whether offline query-difficulty policies improve fixed-budget ReasonIR-8B post-training without regressing general retrieval.

## Goal

- Form a mechanism-level hypothesis about difficulty scoring, filtering, weighting, sampling, or curriculum order over the fixed HQ/VL pool.
- Compare fixed-schema LoRA candidates trained for exactly 1,000 completed optimizer updates and 64,000 attempted triplet exposures.
- Complete at least two matched policy trials before making a curriculum-mechanism claim; this is a scientific reporting requirement, not a Judge input or reward gate.

## Workspace

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Starting state | Pristine ReasonIR source, the fixed offline base at `/opt/reasonir-task/base/ReasonIR-8B`, fixed HQ/VL assets, a policy compiler and fixed four-process trainer, plus a canonical baseline materializer. |
| Deliverable | `/workspace/submission/adapter/**` and `/workspace/submission/manifest.json` |

## Reference baseline

| Item | Value |
| --- | --- |
| Baseline | The pinned immutable ungated `reasonir/ReasonIR-8B` artifact with the canonical all-zero adapter. |
| Reported result | The source reports 24.4 nDCG@10 points on full original-query BRIGHT. That is not the baseline of this three-subject task; no matched four-dataset general-retrieval result is reported. |
| Status | The task validator measures a fresh baseline on this exact three-subject plus four-dataset protocol and the same frozen encoding policy as candidate evaluation. |
| Comparison | Baseline and candidate use the same adapter-plus-manifest ingestion and fixed candidate-only evaluation. Judge does not retrain or rerun a live baseline. |

## Research loop

1. Inspect `/workspace/policy`, prior Work diagnostics under `/workspace/trials`, and prior safe submission feedback.
2. State one falsifiable mechanism hypothesis and change only the permitted policy surface.
3. Run `python /opt/reasonir-task/runtime/run_fixed.py --policy /workspace/policy/policy.json --trial /workspace/trials/<trial-name>`; wait for all four processes and artifact finalization.
4. Select a completed trial with `python /opt/reasonir-task/runtime/select_trial.py --trial /workspace/trials/<trial-name>`, run `/usr/local/bin/validate-reasonir-candidate --json`, and submit.
5. Compare aggregate reasoning and general-retrieval evidence, then retain, revise, or reject the hypothesis.

## What you may change

- Files under `/workspace/policy/**`: offline difficulty scores, buckets, filters, nonnegative weights, deterministic schedules, ordering, and policy metadata derived only from the fixed pool.
- Files under `/workspace/trials/**`: completed fixed-trainer outputs and Work-owned diagnostics.
- The selected adapter under `/workspace/submission/adapter/**` and `/workspace/submission/manifest.json` through the fixed selection tool.

## What stays fixed

- Base model/tokenizer, model architecture, seven-target LoRA schema and hyperparameters, objective, seed policy, optimizer schedule, training text/labels, row identities, and exposure/update accounting.
- Exactly 1,000 completed updates and exactly 64,000 admitted triplet exposures per scoreable trial; failed post-forward attempts keep their exposures.
- ReasonIR source, `/opt/reasonir-task/**`, all paths outside the candidate-owned allowlist, evaluation protocol, metrics, evaluator, and Judge assets.
- No additional or generated data, runtime network, external service, evaluator access, candidate scoring code/cache, benchmark hard-coding, or routine mixture-ratio sweep.

## Evaluation and feedback

| Item | Value |
| --- | --- |
| Fixed workload | BRIGHT Biology (`biology`), Pony (`pony`), and TheoremQA-theorems (`theoremqa_theorems`), plus NFCorpus, SciFact, FiQA-2018, and ArguAna test splits. Every selected dataset retains its complete queries, candidate corpus, relevance judgments and exclusions; one complete deterministic pass per submission. |
| Interpretation | `R` is a fixed three-subject reasoning proxy, not full BRIGHT or a proven substitute for its ranking. No per-round resampling or candidate-dependent dataset selection is allowed. |
| Execution | Four Judge GPUs perform real parallel encoding; Work training also uses four GPUs. The encoding policy is fixed by the task and shared by calibration and candidate evaluation. Text limits and BF16 precision are unchanged. |
| Reward | Let `R` and `G` be equal-weight means of the three selected BRIGHT and four BEIR nDCG@10 values. After five-decimal rounding, `S = R` when `G >= G0`; otherwise `S = G - G0`. Higher is better. |
| Scoreable candidate | One closed PEFT safetensors adapter and bounded manifest for the fixed base/schema, produced in Work with exactly 1,000 updates and 64,000 exposures. Candidate-invalid and attributable failed artifacts are unscored. |
| Visible feedback | `S`, `R`, `G`, non-regression, deltas from baseline, all subject/dataset aggregates, artifact status, runtime, peak memory, actionable candidate validation errors, and the Harness footer. |
| Hidden | Evaluation examples, corpora, qrels, per-query values, rankings, relevant IDs, embeddings, evaluator traces, and task-owned implementation details. |

Use `rsi-submit --list` to inspect prior submission summaries. Complete verifier output is available in the durable Agent log; every printed evaluator line is safe under this contract.

## Submission checklist

- Adapter and manifest are complete, closed, and flushed below `/workspace/submission`.
- The fixed Work training command and selection command finished; no process, GPU state, socket, or cache is required by Judge.
- `/usr/local/bin/validate-reasonir-candidate --json` succeeds without changing the candidate.
- Nothing outside `/workspace/policy`, `/workspace/trials`, and `/workspace/submission` was changed.
- The next submission tests a stated hypothesis, and mechanism claims rely on at least two completed matched policies.

## Evaluation

The Judge evaluates the entire current WORKDIR as it exists when you submit.
The best valid primary score wins. You may call `rsi-submit` repeatedly to
receive feedback while improving the same workspace.
`rsi-submit --list` shows previous submissions; `rsi-submit --help` shows local usage.
Each submission stores complete Judge stdout and stderr in
`/run/rsi-harness/feedback/agent-N.log`, where N is the submission number.
Judge submissions are unlimited during this run.

Every Work GPU process must exit before rsi-submit. A rejected preflight does not consume a submission.
