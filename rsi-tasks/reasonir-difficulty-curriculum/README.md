# ReasonIR difficulty curriculum — three-subject evaluation

## Overview

This Harbor task studies offline difficulty-conditioned weighting, filtering, sampling, and ordering of the fixed ReasonIR HQ/VL pool. Work produces a fixed-schema PEFT LoRA adapter in exactly 1,000 completed updates and 64,000 attempted triplet exposures. Judge evaluates one selected adapter plus its declarative manifest on three fixed BRIGHT subjects and all four original BEIR datasets.

The BRIGHT subjects are Biology, Pony, and TheoremQA-theorems. Each retains its complete queries and candidate corpus: the task reduces domain coverage, not the difficulty of retrieval within a selected domain. The reasoning mean is a three-subject proxy, not full BRIGHT; agreement with full-BRIGHT candidate rankings has not been established. The fixed workload contains 167,188 query/document encodes, compared with 1,412,355 in the former full evaluation. Every submission uses the same dataset selection.

## Baseline and evidence

| Item | Pin or status |
| --- | --- |
| ReasonIR source | `facebookresearch/ReasonIR@0aac96269e455965949df16520fab72da68ffc22` |
| Base artifact | `reasonir/ReasonIR-8B@c3d0690370ff4a8c3d3882d8dfa85c43650034fa` |
| Training data | `reasonir/reasonir-data@0275f825929b206d4ead23d34b4f8a50d4eddbc8` |
| BRIGHT | `xlangai/BRIGHT@3066d29c9651a576c8aba4832d249807b181ecae` |
| GritLM / BEIR | `971068105a8508bca421841c59fddba7f6596402` / `6ef8c9097ebfb203ad360bd64e0cfb93e64f4a44` |
| Reported result | 24.4 nDCG@10 points on full original-query BRIGHT, not a baseline or target score for this three-subject task; no matched fixed-protocol `G0` is reported. |

The Dockerfile, asset lock, and `tests/assets/manifest.json` record build and evaluation provenance. Validator-produced finite five-decimal `REASONIR_BASELINE_R0` and `REASONIR_BASELINE_G0` values are required verifier-only operator inputs. For this exact three-subject, batch-one protocol and pinned assets, the 2026-09-16 four-H100-NVL calibration measured `REASONIR_BASELINE_R0=0.21141` and `REASONIR_BASELINE_G0=0.36126`; the separate full formal Judge reproduced both values. Recalibrate when dataset selection, assets, or encoding policy changes; old full-BRIGHT values and the reported 24.4 cannot be substituted.

The [published runs](../../rsi-logs/reasonir-difficulty-curriculum/) contain 10 Claude submissions and 8 Codex submissions, with best rewards of `0.22340` and `0.21811`. Five submissions receive real negative general-retrieval regression penalties; these are valid measured rewards, not failed submissions.

All 30 payloads in `tests/assets/manifest.json` are required byte-for-byte. The repository includes 24 BRIGHT Parquet files. The four BEIR archives are downloaded locally from their official distribution endpoint because their individual terms differ from the BEIR code license. The BRIGHT Earth Science and Psychology document files are also fetched from the fixed official Hugging Face revision: their original crawled webpage content contains third-party credential-like strings, so these two files are not redistributed in Git. Downloading preserves their original bytes and hashes; no benchmark records are redacted or rewritten. Although Judge scores only the three BRIGHT subjects named above, exact inventory validation and training-positive resolution still depend on the complete locked asset set. Do not remove the other subjects or recompress the assets. See [third-party notices](THIRD_PARTY.md) for benchmark attribution and terms.

## Environment and evaluation

The effective WORKDIR is `/workspace`. Work sees pristine ReasonIR source, the immutable base at `/opt/reasonir-task/base/ReasonIR-8B`, fixed training assets, and fixed runtime tools; BRIGHT/BEIR evaluation assets and evaluator code are injected only into Judge. Both Solution and Judge intentionally use that same immutable base path. Work and Judge are offline. Work uses four GPUs and Judge uses four independent encoding workers; shared memory is 64 GiB. The task allows only `/workspace/policy/**`, `/workspace/trials/**`, `/workspace/submission/adapter/**`, and `/workspace/submission/manifest.json` to change.

The fixed trainer produces a durable fixed-budget journal and an atomic adapter/optimizer/scheduler/RNG checkpoint for restart consistency. Judge validates the starting-state closure, bounded declared training metadata and digest syntax, safetensors adapter, base revision, canonical PEFT configuration, and every tensor's exact name, shape, dtype, and finiteness before deterministic candidate-only evaluation. BRIGHT retains the 32,768-token query and document maxima; BEIR retains the fixed 2,048-token limit and empty query/document instructions. Batch size is fixed at one for baseline and candidates: the local batch-two/four probe failed the predeclared numerical checks, so larger batches were not adopted. Four-GPU calibration took 31m01s; the complete formal Judge took 30m31s on four H100 NVL GPUs, with all 167,188 encodes completed. These are measured local baseline runtimes, not a guarantee for every candidate or machine.

Judge writes no reward for malformed or candidate-attributably failed artifacts, ambiguous failures, timeouts, dependency/infrastructure failures, or incomplete evaluation. A valid general-regressing candidate receives `G-G0`; otherwise reward is `R`, the equal-weight mean of the three selected BRIGHT scores. The four-dataset BEIR mean `G` retains the general-retrieval non-regression check.

The shared Base/Judge architecture cannot provide root-independent proof that Work used the fixed trainer or prevent all system-package tampering. Candidate journals and hashes are consistency evidence only; Judge trusts task-owned validation and recomputed metrics. Public benchmarks and repeated aggregate feedback also leave adaptive-optimization risk.

## Run and validate

From the OpenRSI-Index repository root, first read the [dataset terms](THIRD_PARTY.md) and confirm that your intended use is permitted. Then fetch the four BEIR archives and two BRIGHT document files, and verify the complete asset inventory before starting Harness:

```bash
python3 rsi-tasks/reasonir-difficulty-curriculum/prepare_eval_assets.py --accept-dataset-terms
```

This one-time preparation needs network access to the official BEIR endpoint and Hugging Face. It verifies all 30 original pinned SHA-256 values and leaves the six downloads ignored by Git. Correct existing files are reused without network access or rewriting; a mismatched file or download stops preparation. It does not run training or evaluation. Work and Judge remain offline; neither downloads data at runtime. A fresh clone is not ready for evaluation until this command succeeds.

After installing [RSI-Harness](../../RSI-Harness/), use the calibrated values only with the unchanged pinned protocol:

```bash
export REASONIR_BASELINE_R0=0.21141
export REASONIR_BASELINE_G0=0.36126
sudo -E REASONIR_BASELINE_R0="$REASONIR_BASELINE_R0" REASONIR_BASELINE_G0="$REASONIR_BASELINE_G0" \
  rsi-harness run "$PWD/rsi-tasks/reasonir-difficulty-curriculum" \
  --agent codex --model gpt-5.6-sol --reasoning-effort xhigh --agent-auth local \
  --gpus 0,1,2,3,4,5,6,7 --primary-reward reward --score-direction maximize \
  --verbose
```

Python syntax check (not an environment build or GPU validation):

```bash
python3 -m compileall -q rsi-tasks/reasonir-difficulty-curriculum
```

The eight-device example lets Harness allocate four GPUs to Work and four separate GPUs to Judge. With only four devices in the caller's pool, Harness must release Work's GPU state for Judge instead; the task's four-GPU requirement does not change.

The modified task passed fresh CPU-only starting-state inspection, four-GPU calibration, and a separate complete formal Judge on 2026-09-16. The formal Judge exited zero, emitted only the finite reward artifact (`0.21141`), and released all four GPUs without OOM or timeout. This establishes task execution readiness for the validated protocol and environment. The subsequent published agent runs provide separate training and multi-round evidence; neither establishes full-BRIGHT ranking correlation.
