# CosyVoice3: default-on TTS runtime research

This is a new single-model dual-mode task, version 4.0.0 / protocol v8, derived from the D154 task family. Both non-streaming and streaming throughput are strict primary objectives; streaming has twice the reward exponent. Model weights, data and other performance/quality guard thresholds retain their pinned definitions. See the repository provenance for exact donor commits.

## Goal and starting state

Can profiling-guided, execution-preserving optimization of the CosyVoice3 AR + DiT + HiFT pipeline—including scheduling and batching, CPU/GPU handoffs, state and buffer reuse, compilation and CUDA graphs, attention and operator execution, and invariant precomputation—improve **both streaming and non-streaming English TTS requests/s at concurrency 16**, while preserving the pinned model semantics, workload, and every existing quality, latency and continuity guard?

Both mode throughput means must strictly improve against Bk through default-on changes within the existing action space. Keep a falsifiable hypothesis, matched controls and controlled ablations for each change.

- Source: SGLang-Omni [0.1.9 / fa90213e1674e99ef15e6c3ba6fa9bb15255414a](https://github.com/sgl-project/sglang-omni/blob/fa90213e1674e99ef15e6c3ba6fa9bb15255414a/sglang_omni/__init__.py) runtime (upstream main resolved on 2026-10-21), with SGLang `0.5.22` and its required dependency updates. The original native `benchmarks/` tree at `1523543056bf334bc2f4d69db23aad0a002cedf2` is retained byte-for-byte so version changes do not alter token limits, timing, metrics or evaluation behavior. Source archives and inventories record this explicit composition; see `.rsi/version-upgrade.json` within this task package.
- Released checkpoint: `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` at `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`. Exact asset inventories and hashes are in `environment/assets.json` and `tests/assets.json`.
- Model-specific native benchmark arguments: no `--ref-format` override; retain the CosyVoice3 benchmark defaults. Other native model sampling, precision, voice, codec and chunking defaults remain fixed. The token cap is **2,048**, including models with a different cookbook example cap.
- Baseline: a fresh complete pristine six-run calibration on the same image, GPU, CPU/NUMA allocation and evaluator as the candidate. GPU validation and calibration have **not** been performed for this derivative. Earlier D154 and vLLM/SGLang comparison numbers are not calibration data.

## Research guidance

The contributor supplied the following directions as **optional hypotheses**, not a list of promised gains or changes to the evaluation contract. Begin with profiling, form a falsifiable hypothesis, make one controlled change, inspect feedback, and retain or revise it. Combine compatible changes only after useful controls/ablations.

Reference: [CosyVoice AR + DiT + Vocoder optimization report](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/d1f309b46f81a23e8b6ca0718c05bc897fafacf7/sglang/sglang-omni/cosyvoice.md), article code revision `0ee0304a5` (2026-10-16). The contributor-provided full text is preserved byte-for-byte in `research-guidance/cosyvoice-optimization-report.md` within this task package for the operator. The table below is embedded in the task prompt so research does not depend on web access.

The article describes a different source revision and its own workloads. Check which mechanisms already exist in this task's pinned source; its code snippets, constants and reported gains are evidence to investigate, not baseline measurements for this task. Keep the existing action space, precision, sampling, chunking, workload, dependency/network boundaries, hardware, formal six-run order, quality gates and reward unchanged. Optional Work diagnostics and matched comparisons must not replace formal Judge evaluation. No new services, data, packages or formal metrics are authorized.

| Direction | Hypotheses and techniques | Boundaries |
| --- | --- | --- |
| End-to-end profiling and validation | Locate stage costs on the actual workload; compare matched controls and isolate each change. Inspect throughput, latency, first audio and quality together. | A faster operator is not an end-to-end gain. WER and speaker similarity do not fully describe perceived quality; similarity is supplementary, not a new reward gate. |
| Runtime overhead first | Investigate repeated work, CPU stalls, stage handoffs, batching and idle GPU time before deep kernel work. | Validate along the complete serving path; upstream scheduling and synchronization can hide kernel gains. |
| Non-streaming dynamic batching | Study AR continuous batching and independent length-aware Flow and HiFT batches. | Stages need not share a batch; larger batches can increase waiting and tail latency. |
| Padding and grouping cost | Study length sorting, grouping or dynamic programming under length-gap and padding budgets. | Account for wait windows, request counts, total frames and length differences; maximizing batch size alone is insufficient. |
| Streaming packed inference and CFG batching | Pack separate request rows along the sequence dimension; batch conditional and unconditional CFG rows. | Keep attention isolated between requests and preserve CFG math and branch ordering. CFG batching does not remove either branch. |
| Streaming chunks and scheduling | Use the report's 25-token hop + 3-token lookahead and 25 → 50 → 100 growth as context; investigate deadline/playback-aware scheduling and compatible hop/leftover batching. | These are reference values, not permission to change fixed chunking or lookahead. Preserve native chunk boundaries, first-audio semantics and continuity. |
| Streaming Flow history cache | Investigate reusing historical K/V per Euler step, layer and CFG branch so later hops compute new frames. | A causal mask alone does not eliminate recomputation. Preserve cache isolation, lifetime, capacity fallback and first-hop costs. |
| Batched and windowed HiFT | Study one HiFT call per streaming step and decoding new frames with the necessary left context (29 frames in the report). | Verify receptive fields against the pinned code. Preserve F0/sine history, phase, noise positions, final right alignment and STFT/ISTFT boundaries. |
| Parallel CPU preprocessing | Move independent reference-audio work outside shared-state locks; inspect real concurrency and ONNX thread pools/spinning. | Configured worker counts do not prove parallel execution; lock scope and competing thread pools matter. Keep resource ownership unchanged. |
| Avoid host synchronization | Study pinned buffers, asynchronous D2H transfer, completion events and deduplicated readbacks. | Separate CUDA streams alone do not remove blocking; preserve data dependencies, buffer lifetimes and cancellation cleanup. |
| Device-resident constants and buffers | Reuse STFT windows, source constants and device-allocated buffers; remove repeated transfer, conversion and allocation. | Plain tensor attributes may not follow module .to(device); verify device and dtype explicitly. |
| CUDA graphs | Study whole-Euler capture, (batch, frames) buckets, shared graph pools and correct eager fallback. | The report's graph implementation targets non-streaming buffered inference. Account for padding, hit rate, graph count, startup and peak memory; do not assume it applies to streaming. |
| torch.compile and warmup | Compile eligible DiT/PackedDiT paths and make only required dimensions dynamic; cover actual serving shapes. | Warmup must match serving inference_mode and autocast. Track recompilation, graph breaks and startup time; use only installed dependencies. |
| Ragged/FlashAttention | Where available, study varlen or paged attention to avoid padding and scatter/attention/gather overhead. | Preserve streaming chunk visibility and full-row non-streaming semantics; keep hardware/dtype fallbacks. Prefix reuse and faster attention are separate mechanisms. |
| Pipeline memory allocation | Size reusable state against useful request/context capacity while leaving room for Flow, HiFT, graphs and caches. | Preserve the fixed effective benchmark settings and resource budget; enlarging one pool can starve another stage. |
| Stage overlap and replicas | Investigate legal intra-pipeline overlap, including an independent vocoder stream. | The formal task retains one GPU and its fixed launch topology/concurrency. The report's multi-process, MPS, replica and multi-GPU results are context only; aggregate throughput is not single-request speedup. |
| Per-stage precision | Understand why the report uses BF16 DiT with an independently configured, normally FP32 HiFT. | The task forbids changing precision. Optimize execution under the pinned precision; do not generalize one stage's numerical tolerance to the pipeline. |
| Operator fusion and memory traffic | Investigate equivalent Q/K/V projection fusion, shared rotary angles and fewer temporary tensors/conversions. | Preserve weights and numerical semantics; check whether the compiler already performs the same fusion. |
| Invariant precomputation | Distinguish model-level constants, request-level invariants and step-varying state; consider fixed-schedule time embeddings and AdaLN modulation. | Retain Euler steps, schedule and model semantics. Key cached data to the model/dtype/schedule and include startup and memory costs. |
| Workload-specific backends | Assess an already available backend against actual batch/shape support and the packed serving path. | The report's optional TensorRT CFG batch of 2 is an example, not an expected win or permission to install a backend. Respect dependencies, fallback correctness and the unchanged candidate scope. |

Record actual per-mode throughput, latency, first-audio/continuity and WER evidence under the fixed protocol. For relevant changes, also report startup cost, peak GPU memory, graph count and cache behavior as diagnostics. Any supplementary listening or similarity evidence must use already available assets/tools and remain separate from the unchanged reward. Historical article numbers and inherited run logs cannot substitute for a fresh matching calibration.

## Workspace and action space

Work starts in `/workspace`, with the complete runtime in `/workspace/sglang-omni`, restoration source in `/opt/rsi-source`, dependencies in `/opt/rsi-deps` and pinned assets in `/opt/rsi-assets`. Judge reloads the complete materialized Work snapshot and injects task-owned tests; this is the template's shared snapshot model, not an independent clean-Base verifier.

Edit TTS-relevant code under `sglang_omni/` for runtime optimization. Within `sglang_omni/models/`, only `fun_cosyvoice3/` is editable. Fixed ASR source and `sglang_omni/sampling/` remain protected. A permitted path does not authorize semantic changes to model behavior. No training, checkpoint/architecture/precision/tokenizer/sampling/voice/reference changes, special-casing examples, answer caches, evaluator detection, omission, replacement, resampling, truncation, time stretching or fabricated metrics. Preserve request isolation, cancellation cleanup and default-on operation.

Keep benchmark/client/timing/loading/metric code, ASR, dependencies, assets, interpreter, clocks and all other source unchanged. No external services, web research, extra datasets/checkpoints, runtime installs/downloads or uploads; native localhost serving is allowed. The provider route is not research egress.

Changed candidates must deliver nonempty `/workspace/research/hypothesis.md`, `experiments.json`, `reproduction.json` and `source.diff`, with profiles, complete experiment history, controls, ablations and limitations. Candidate tests and temporary output belong outside the source tree. Source and records must be closed/flushed, and GPU processes stopped, before `rsi-submit`. Work records never determine the reward. Protocol v8 requires fresh B0 and A/A; prior single-objective calibrations and rewards are incompatible. The baseline is the pinned starting runtime, not a historical optimized candidate.

Before batch experiments, trace benchmark defaults through server CLI overrides, config merging and effective scheduler/CUDA graph settings. Client concurrency is 16; native `max_running_requests` and `cuda_graph_max_bs` are both 64. A lower-priority model config edit may be shadowed. Record actual command vectors, value provenance, effective settings and observed batch sizes; restore fixed evaluation settings after diagnostic overrides.

## Fixed evaluation

| Item | Contract |
| --- | --- |
| Workload | All 1,088 English rows of `zhaochenyang20/seed-tts-eval-arrow`, revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`, in native order |
| Workload keys | `cosy` (non-streaming), `cosy_stream` (streaming); separate outputs and statistics |
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
| Primary: non-streaming CosyVoice3 requests/s | **C > B**; equality or slowdown is ineligible |
| Primary: streaming CosyVoice3 requests/s | **C > B**; equality or slowdown is ineligible |
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
