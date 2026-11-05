# MOSS v1.5: default-on TTS runtime research

This is a new single-model dual-mode task, version 4.0.0 / protocol v8, derived from the D154 task family. Both non-streaming and streaming throughput are strict primary objectives; streaming has twice the reward exponent. Model weights, data and other performance/quality guard thresholds retain their pinned definitions. See the repository provenance for exact donor commits.

## Goal and starting state

Improve **BOTH MOSS v1.5 non-streaming and streaming English TTS requests/s at concurrency 16**, through default-on scheduling, CPU dispatch, inter-stage transport, allocation/state reuse, reference-conditioning reuse or pipeline organization. Both mode throughput means must strictly improve against Bk; preserve all quality and other performance guards in both modes. Keep a falsifiable hypothesis, matched controls and controlled ablations for each change.

- Source: unmodified SGLang Omni main `cb75ab409669a6ab672fa42421b41b11af2fe263` (2026-10-17); the source archive is GitHub's archive of that commit and the native benchmark is its own. The interpreter/dependency lock is re-resolved for its `pyproject.toml` (SGLang 0.5.21) with `generation/resolve_metadata.py --prefer`, keeping every other template pin that still resolves.
- Released checkpoint: `OpenMOSS-Team/MOSS-TTS-v1.5` at `cdd3b911b1585e3f2dbc7775ef10f9926f58850a`. Exact asset inventories and hashes are in `environment/assets.json` and `tests/assets.json`.
- Model-specific native benchmark arguments: `--ref-format references --token-count auto`. Other native model sampling, precision, voice, codec and chunking defaults remain fixed. The token cap is **2,048**, including models with a different cookbook example cap.
- Baseline: a fresh complete pristine six-run calibration on the same image, GPU, CPU/NUMA allocation and evaluator as the candidate. GPU validation and calibration have **not** been performed for this derivative. Earlier D154 and vLLM/SGLang comparison numbers are not calibration data.

## Workspace and action space

Work starts in `/workspace`, with the complete runtime in `/workspace/sglang-omni`, restoration source in `/opt/rsi-source`, dependencies in `/opt/rsi-deps` and pinned assets in `/opt/rsi-assets`. Judge reloads the complete materialized Work snapshot and injects task-owned tests; this is the template's shared snapshot model, not an independent clean-Base verifier.

Edit TTS-relevant code under `sglang_omni/` for runtime optimization. Within `sglang_omni/models/`, only `moss_tts/` is editable. Fixed ASR source and `sglang_omni/sampling/` remain protected. A permitted path does not authorize semantic changes to model behavior. No training, checkpoint/architecture/precision/tokenizer/sampling/voice/reference changes, special-casing examples, answer caches, evaluator detection, omission, replacement, resampling, truncation, time stretching or fabricated metrics. Preserve request isolation, cancellation cleanup and default-on operation.

Keep benchmark/client/timing/loading/metric code, ASR, dependencies, assets, interpreter, clocks and all other source unchanged. No external services, web research, extra datasets/checkpoints, runtime installs/downloads or uploads; native localhost serving is allowed. The provider route is not research egress.

Changed candidates must deliver nonempty `/workspace/research/hypothesis.md`, `experiments.json`, `reproduction.json` and `source.diff`, with profiles, complete experiment history, controls, ablations and limitations. Candidate tests and temporary output belong outside the source tree. Source and records must be closed/flushed, and GPU processes stopped, before `rsi-submit`. Work records never determine the reward. Protocol v8 requires fresh B0 and A/A; prior single-objective calibrations and rewards are incompatible. The baseline is the pinned starting runtime, not a historical optimized candidate.

Before batch experiments, trace benchmark defaults through server CLI overrides, config merging and effective scheduler/CUDA graph settings. Client concurrency is 16; native `max_running_requests` and `cuda_graph_max_bs` are both 64. A lower-priority model config edit may be shadowed. Record actual command vectors, value provenance, effective settings and observed batch sizes; restore fixed evaluation settings after diagnostic overrides.

## Fixed evaluation

| Item | Contract |
| --- | --- |
| Workload | All 1,088 English rows of `zhaochenyang20/seed-tts-eval-arrow`, revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`, in native order |
| Workload keys | `moss` (non-streaming), `moss_stream` (streaming); separate outputs and statistics |
| Runs | Three non-streaming runs, then three streaming runs, serially on one GPU |
| Sampling seeds | 42, 43, 44 in that order for each mode, for both pristine baseline and candidates; arrival seed unset |
| Concurrency / warmup | 16 / 16 for both TTS modes; warmups excluded from scored throughput |
| Count | 6,528 scored TTS requests and 96 TTS warmups per complete evaluation |
| Coverage | Every run needs 1,088 valid native WAVs and transcripts, zero failures/skips; no per-sample retries |
| Cache/process policy | Fresh processes and empty compilation caches for every repetition |
| Quality | Stock diagnostic ASR, followed by independent original-source authoritative `Qwen/Qwen3-ASR-1.7B` on the exact generated WAVs; ASR concurrency 32 and warmup 64 |
| Hardware | One GPU, 16 distinct physical CPU cores without SMT siblings, 128 GiB unique resident memory per lane; explicit CPU/NUMA ownership and foreign CPU load at most two cores |
| Time | 14,400-second complete Judge deadline; startup, compilation, both ASR passes and cleanup count; Work budget 24 hours, operator extension to 48 hours allowed |

All settings above, including full audio/ASR, CPU contention, native format, integrity, source scope and cleanup checks, retain the template logic. A retry is a new complete evaluation. Seeds control request sampling, not all waveform RNG or numerical/scheduling nondeterminism. The deadline is not a measured runtime or a promise of ten adaptive loops; measure actual full-cycle cadence before research.

## Metrics and reward

Compute arithmetic means of three **per-run** values separately for the two modes; do not pool requests across runs, discard repetitions, clip ratios or add rounding. Both modes report corpus micro-WER (including high-error examples), latency and RTF mean/median/p95/p99, requests/s and audio-seconds/s. Streaming additionally reports TTFA (`audio_ttfp_*_s`) mean/median/p95/p99 and C50. C50 is the percentage of multi-chunk requests with maximum playback underrun at most 50 ms. Single-chunk requests are N/A, never zero or 100; an entirely N/A cell fails the original completeness rule. Per-request first-audio/chunk evidence must agree with summary values within native rounding. C50 can be zero but must be at most 100.

Let B be the last accepted baseline and C the candidate's corresponding three-run mean:

| Role | Eligibility rule |
| --- | --- |
| Primary: non-streaming MOSS v1.5 requests/s | **C > B**; equality or slowdown is ineligible |
| Primary: streaming MOSS v1.5 requests/s | **C > B**; equality or slowdown is ineligible |
| Other performance, in both modes | Lower is better: **C ≤ 1.03 × B**; higher is better: **C ≥ 0.97 × B** |
| Quality, independently in both modes | Corpus WER **C ≤ 0.02**, an absolute 2% ceiling |

All **27** model/mode/metric gates must pass, together with integrity, validity, resource, coverage and cleanup checks. The original stock precision is retained (three decimals for latency/throughput, four for RTF and computed WER precision). A 3% margin is an engineering tolerance, not a confidence interval. A 10% speedup is not a required minimum and cannot compensate for a failed guard.

Define R_nonstream(C;B) and R_stream(C;B) as the corresponding three-run mean QPS ratios. Reward is **R_stream(C;B0)^(2/3) × R_nonstream(C;B0)^(1/3)**, a finite, dimensionless, unclipped weighted geometric mean. Acceptance separately requires **R_nonstream(C;Bk) > 1 AND R_stream(C;Bk) > 1**. A weighted score above 1 never compensates for equality or regression in either mode. Eligibility always compares with the **last accepted** baseline Bk. The host operator alone promotes a complete eligible candidate using `noise.freeze_accepted`, retaining B0, every generation, evaluated source digest, code/config/results and rejected attempts. No-op, failed, incomplete, timed-out or ineligible rounds never advance the baseline. Recompute all guards and verify the evaluated snapshot before atomic promotion. Work cannot edit operator-owned baseline state.

Only a complete eligible evaluation writes `/logs/verifier/reward.json`. Gate failure is valid-but-ineligible/unscored; attributable compile failure, illegal memory access, invalid audio, crash or rule violation is candidate-invalid/unscored. Infrastructure, dependency errors, OOM, ambiguous attribution, missing/mismatched calibration, timeout and incomplete execution are separately identified and unscored. There is no finite failure scalar or artificial no-op score of 1.

Feedback includes per-repetition metrics, separate means, coverage, timings, resource peaks, contention, full safe child diagnostics, baseline/candidate values, signed/relative deltas, comparator, threshold, guard role and pass/fail. WER also reports percentage-point deltas; relative values are null when baseline is zero. Report current generation/source, initial-baseline comparisons, both per-mode step C/Bk and cumulative C/B0 ratios, plus the step and cumulative weighted rewards. Withhold per-example ASR hypotheses, answer/error-ranking tables, protected Judge-check contents and credentials. Public text/reference audio remain public. WER does not establish speaker identity, fidelity, prosody or generalization. Public-workload adaptation, fixed-tolerance noise and shared root/interpreter/library state remain limitations; source/hash checks are not arbitrary-root isolation.

## Validation and operation

The unchanged Judge entrypoint is `tests/test.sh`. `solution/solve.sh` only restores the original runtime; it does not train, evaluate or fabricate a score. The bundled read-only `RSI-Harness` and validator/runner skills are inherited unchanged from the template.

Before research, build the image, verify imports and offline aliases, verify pristine scope, allocation/device/resource mapping, then run `/opt/sglang/bin/python3 -I /tests/calibrate.py` with the task's tests injected into the prepared container. Provide exactly one assigned UUID in `RSI_HARNESS_EXPECTED_GPU_UUIDS`, one array of 16 physical CPU IDs in `RSI_TTS_CPU_LANES`, and one corresponding NUMA-node array in `RSI_TTS_NUMA_LANES`. Never use placeholder allocations as real ownership evidence.

Retain the complete `RSI_TTS_CALIBRATION=` record. Run `/opt/sglang/bin/python3 -I /tests/pool_calibrations.py <record-1>` and bind its schema-3 frozen JSON to Judge-only `RSI_TTS_BASELINE_JSON`. Keep this raw state out of Work. Recalibrate after source, protocol, evaluator, assets, software, hardware, clocks or allocation changes. The new protocol hash rejects all template and other-model calibrations.

One separate held-out pristine A/A evaluation must exercise the full Judge path before research; it cannot change B0 or any threshold. Keep its real result even if it is valid-but-ineligible. Check failure attribution, redaction, cleanup and measured cadence. Static and synthetic checks do not establish image/GPU validation, WER or throughput.

On the authorized compute node, keep environments, weights, datasets, caches, audio, per-request output and full logs there. The control machine retains only necessary configuration/version IDs, aggregate metrics, failure summaries and recovery/resource evidence. This repository contains source and metadata only; it includes no historical trajectory or old calibration masquerading as this model's result.


## Public task copy

This package is published in OpenRSI-Index. [Publication provenance](publication.json) records the source repository, exact commit and hashes of every original task file. The environment, evaluator, fixed assets, source archives and scoring contract are copied unchanged. Historical setup statements describe the source run; a new run requires its own matching environment validation and calibration. No GPU evaluation was performed for this publication.
