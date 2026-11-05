# Ming-3.0-tts-0.5B: non-streaming TTS runtime research

Improve non-streaming English TTS requests/s at concurrency 16 through default-on runtime scheduling, dispatch, inter-stage transport, allocation/state reuse, reference-conditioning reuse or pipeline organization. Streaming is outside this task, as requested by the contributor. Task version 5.0.0 / protocol v9 retains the reference task's non-streaming evaluation.

## Baseline and model

- Unmodified SGLang-Omni 0.1.9 at `ca9e8789fcd84ed6f7272276cb99447eec751787`; no streaming prerequisite patch.
- Official checkpoint `inclusionAI/Ming-3.0-tts-0.5B` at `8d66b70a00f05ad183fa8a863b4ce609fc83cbc0`. Exact assets and dependency hashes are in the paired environment/tests inventories.
- Request token cap remains 2,048. The model's native clamp and sampling semantics remain fixed: 1,000 latent steps, temperature 0, minimum 5 decode steps; reference audio plus transcript supplies prompt latents. Audio is native 44.1 kHz. Do not replace one checkpoint's conditioning or defaults with the other's.
- Establish fresh B0 and an independent pristine A/A on the exact prepared image, evaluator, GPU and CPU allocation. No previous model/task calibration or optimized candidate is reused.

## Fixed evaluation

All 1,088 English rows of `zhaochenyang20/seed-tts-eval-arrow` at `27f4c1adee83b5b29b7c4b375f6b976324bda308`, in native order. Three serial non-streaming runs use seeds 42, 43, 44, concurrency 16 and 16 warmups each: 3,264 scored requests plus 48 warmups. Arrival seed remains unset. No retries, skips, sample replacement or subset scoring. Every repetition starts fresh processes and empty compilation caches.

The fixed native benchmark and an independent original-source authoritative `Qwen/Qwen3-ASR-1.7B` score the exact generated WAVs. ASR concurrency is 32, warmup 64. Coverage, native audio format, source integrity, device mapping, resource monitoring and cleanup retain the reference checks. The benchmark overrides `max_running_requests=64` and `cuda_graph_max_bs=64`; verify the effective settings before experiments.

One GPU, 16 distinct physical CPU cores, the same CPU/NUMA mapping, and 128 GiB memory per evaluation lane. The inherited CPU-contention threshold remains at most two foreign cores. Judge deadline is 14,400 seconds including startup, compilation, both ASR passes and cleanup. Research budget is 24 hours including Judge waits; submissions are unlimited. A timeout is a bound, not a measured evaluation duration.

## Score and acceptance

Compute arithmetic means of the three per-run metrics. Non-streaming throughput must be strictly greater than the last accepted baseline Bk. Equality or regression is ineligible. Reward is the candidate's three-run mean QPS divided by the initial B0 mean QPS. B0 stays fixed for reward; only the operator can advance Bk after independently verifying the exact evaluated candidate and every gate.

All 11 non-streaming gates remain: corpus micro-WER at most 0.02; latency and RTF mean/median/p95/p99 at most 1.03 times Bk; audio-seconds/s at least 0.97 times Bk; requests/s strictly greater than Bk. Keep native precision and full coverage. Streaming TTFA and C50 are not evaluated. A complete, eligible evaluation alone writes `reward.json`; ineligible, invalid, infrastructure-failed, incomplete and timed-out evaluations are unscored and never advance Bk.

## Workspace and boundaries

Work starts in `/workspace`; runtime is `/workspace/sglang-omni`, restoration source `/opt/rsi-source`, pinned assets `/opt/rsi-assets`. Edit TTS-relevant runtime code under `sglang_omni/`; within `sglang_omni/models/`, only `ming3_tts/` is editable. ASR and `sglang_omni/sampling/` are protected. Keep benchmarks, clients, timing, loading, metrics, dependencies, weights, architecture, precision, tokenizer, sampling, voice/reference semantics, clocks and all other source fixed. No training, answer caching, dataset special-casing, evaluator detection, truncation, resampling, time stretching or fabricated metrics. Preserve request isolation, cancellation and default-on operation.

No external services, web research, extra models/data or runtime installs/downloads/uploads; native localhost serving is allowed. Judge reloads the complete materialized Work snapshot and injects task-owned tests. The shared snapshot contract is not a clean-Base security boundary. Work must not inspect Judge tests, baseline state or credentials.

For each changed candidate, deliver nonempty `/workspace/research/hypothesis.md`, `experiments.json`, `reproduction.json` and `source.diff`. Record the falsifiable hypothesis, profiles, matched controls, ablations, effective settings, complete history and limitations. Close files and stop GPU processes before `rsi-submit`. Operator records and native measurements determine acceptance; self-reported work records do not determine reward.

## Validation and feedback

Build the environment and verify pristine scope, offline assets, source/dependency identities and actual resource mapping. Inject tests read-only and run `/opt/sglang/bin/python3 -I /tests/calibrate.py`. Supply the assigned UUID and actual CPU/NUMA arrays via the three runtime variables declared in `task.toml`. Freeze the complete calibration with `pool_calibrations.py`; bind `RSI_TTS_BASELINE_JSON` to Judge only. Recalibrate after any identity change. The separate A/A must exercise the full native Judge path; equality remains valid-but-ineligible and does not change B0.

Feedback includes per-repetition metrics, means, coverage, timings, resource peaks, contention, safe diagnostics, signed/relative differences and every guard decision, Bk generation/source, B0 comparisons and cumulative reward. Withhold per-example ASR hypotheses and answer/error-ranking tables. WER alone does not establish speaker similarity, fidelity, prosody or generalization. Public workload adaptation and measurement noise remain limitations.

The bundled RSI-Harness remains read-only. Keep large assets, environments and audio on Radix; continuously preserve minimum result/recovery state outside the node. Release the assignment after all authorized work and final persistence verification complete.


## Public task copy

This package is published in OpenRSI-Index. [Publication provenance](publication.json) records the source repository, exact commit and hashes of every original task file. The environment, evaluator, fixed assets, source archives and scoring contract are copied unchanged. Historical setup statements describe the source run; a new run requires its own matching environment validation and calibration. No GPU evaluation was performed for this publication.
