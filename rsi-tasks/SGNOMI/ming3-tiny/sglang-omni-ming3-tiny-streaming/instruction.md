# Ming-3.0-tts-tiny: streaming TTS runtime research

Improve streaming English TTS requests/s at concurrency 16 through default-on runtime scheduling, dispatch, inter-stage transport, allocation/state reuse, reference-conditioning reuse or pipeline organization. Requests must use native stream=true and PCM; every chunk must carry valid timing/duration evidence. Task version 6.0.0 / protocol v10 retains the reference task's streaming evaluation.

## Baseline and model

- SGLang-Omni main with the declared common setup compatibility port at `0c663861d7d4a558e4f53258c4d48cfce577b47e`; common setup changes are declared below.
- Official checkpoint `inclusionAI/Ming-3.0-tts-tiny` at `7bd9446a8d47e462362261a0150258da6d458318`. Exact assets and dependency hashes are in the paired environment/tests inventories.
- Request token cap remains 2,048. The model's native clamp and sampling semantics remain fixed: 250 latent steps, temperature 1.5, minimum 13 decode steps; reference audio supplies a speaker embedding. Audio is native 44.1 kHz. Do not replace one checkpoint's conditioning or defaults with the other's.
- The operator must complete fresh B0 on the exact prepared image, evaluator, GPU and CPU allocation. The user explicitly waived independent A/A and requested immediate eight-hour research after B0. No previous model/task calibration or optimized candidate is reused.

## Fixed evaluation

All 1,088 English rows of `zhaochenyang20/seed-tts-eval-arrow` at `27f4c1adee83b5b29b7c4b375f6b976324bda308`, in native order. Three serial streaming runs use seeds 42, 43, 44, concurrency 16 and 16 warmups each: 3,264 scored requests plus 48 warmups. Arrival seed remains unset. No retries, skips, sample replacement or subset scoring. Every repetition starts fresh processes and empty compilation caches.

The fixed native benchmark and an independent original-source authoritative `Qwen/Qwen3-ASR-1.7B` score the exact generated WAVs. ASR concurrency is 32, warmup 64. Coverage, native audio format, source integrity, device mapping, resource monitoring and cleanup retain the reference checks. The benchmark overrides `max_running_requests=64` and `cuda_graph_max_bs=64`; verify the effective settings before experiments.

One GPU and 128 GiB memory per evaluation lane, with the same recorded CPU/NUMA mapping for baseline and candidates. The inherited CPU-contention threshold remains at most two foreign cores. Judge deadline is 14,400 seconds including startup, compilation, both ASR passes and cleanup. Research budget is 8 hours including Judge waits; submissions are unlimited. A timeout is a bound, not a measured evaluation duration.

## Score and acceptance

Compute arithmetic means of the three per-run metrics. Streaming throughput must be strictly greater than the last accepted baseline Bk. Equality or regression is ineligible. Reward is the candidate's three-run mean QPS divided by the initial B0 mean QPS. B0 stays fixed for reward; only the operator can advance Bk after independently verifying the exact evaluated candidate and every gate.

All 16 streaming gates remain: corpus micro-WER at most 0.02; latency and RTF mean/median/p95/p99 at most 1.03 times Bk; audio-seconds/s at least 0.97 times Bk; requests/s strictly greater than Bk. Keep native precision and full coverage. Streaming TTFA mean/median/p95/p99 and C50 continuity are additional 3% relative guards. Single-chunk responses are N/A for continuity; at least one multi-chunk response is required. A complete, eligible evaluation alone writes `reward.json`; ineligible, invalid, infrastructure-failed, incomplete and timed-out evaluations are unscored and never advance Bk.

## Workspace and boundaries

Work starts in `/workspace`; runtime is `/workspace/sglang-omni`, restoration source `/opt/rsi-source`, pinned assets `/opt/rsi-assets`. Edit TTS-relevant runtime code under `sglang_omni/`; within `sglang_omni/models/`, only `ming3_tts/` is editable. ASR and `sglang_omni/sampling/` are protected. Keep benchmarks, clients, timing, loading, metrics, dependencies, weights, architecture, precision, tokenizer, sampling, voice/reference semantics, clocks and all other source fixed. No training, answer caching, dataset special-casing, evaluator detection, truncation, resampling, time stretching or fabricated metrics. Preserve request isolation, cancellation and default-on operation.

No external services, web research, extra models/data or runtime installs/downloads/uploads; native localhost serving is allowed. Judge reloads the complete materialized Work snapshot and injects task-owned tests. The shared snapshot contract is not a clean-Base security boundary. Work must not inspect Judge tests, baseline state or credentials.

For each changed candidate, deliver nonempty `/workspace/research/hypothesis.md`, `experiments.json`, `reproduction.json` and `source.diff`. Record the falsifiable hypothesis, profiles, matched controls, ablations, effective settings, complete history and limitations. Close files and stop GPU processes before `rsi-submit`. Operator records and native measurements determine acceptance; self-reported work records do not determine reward.

## Validation and feedback

Operator-only setup and full B0 are already complete before this Agent starts. The measured starting point below is authoritative for this run. Start runtime profiling and optimization immediately. Do not invoke /tests/calibrate.py, inject tests, rebuild the environment, or freeze calibration: these are operator duties, and /tests is intentionally absent from Work. Do not call rsi-submit with unchanged source to reproduce B0 or test setup. Use rsi-submit only for changed candidates with the required research records. The unchanged native Judge evaluates candidates against the frozen B0/Bk and all existing acceptance gates. Do not inspect private tests or baseline state.

Feedback includes per-repetition metrics, means, coverage, timings, resource peaks, contention, safe diagnostics, signed/relative differences and every guard decision, Bk generation/source, B0 comparisons and cumulative reward. Withhold per-example ASR hypotheses and answer/error-ranking tables. WER alone does not establish speaker similarity, fidelity, prosody or generalization. Public workload adaptation and measurement noise remain limitations.

The bundled RSI-Harness remains read-only. Run Agent `codex`, model `gpt-6-astra`, reasoning effort `xhigh`. Keep large assets, environments and audio on Radix; continuously preserve minimum result/recovery state outside the node. Release the assignment after all authorized work and final persistence verification complete.

## Measured starting point

Fresh streaming B0 completed on the exact pinned image and GPU. Three-run aggregate: {"audio_throughput_s_per_s": 19.083333333333332, "audio_ttfp_mean_s": 1.1801, "audio_ttfp_median_s": 1.1716666666666666, "audio_ttfp_p95_s": 1.4296999999999997, "audio_ttfp_p99_s": 1.6840000000000002, "c50": 88.81666666666666, "latency_mean_s": 2.9933333333333336, "latency_median_s": 2.9463333333333335, "latency_p95_s": 4.389, "latency_p99_s": 5.175, "rtf_mean": 0.8375666666666666, "rtf_median": 0.8290666666666667, "rtf_p95": 0.9567666666666667, "rtf_p99": 1.0517666666666665, "throughput_qps": 5.3133333333333335, "wer_corpus": 0.011024589020067544}. These measurements are calibration, not an optimization result. Candidate acceptance retains every original quality and performance gate.

## Source refresh

SGLang-Omni `0c663861d7d4a558e4f53258c4d48cfce577b47e`; compatible SGLang core `b8100b74bf4f6563f402c20a1e1af685cbd2246e`. Omni is frozen at the latest observed main; core uses the already prepared compatible revision. Both remain fixed throughout B0 and research; no old calibration, accepted candidate, PR3217 patch or research clock is reused. Source task: `c2e016ae654c171e4a724e3956d957eb7f66b462`.

The contributor explicitly set the formal research budget to 8 hours on 2026-11-02. Setup and full B0 precede the research clock. Independent A/A and repeated subset smoke are omitted by explicit user instruction. Stop at formal Agent start + 28,800 seconds, preserve verified candidates and finish already accepted Judge work, then prepare evidence-backed upstream PRs. Do not extend or reset the research clock.

The baseline also contains the explicit `upstream-compatibility.patch`: Omni forwards the latest core draft-layout arguments, obtains cache slots through the current storage interface, follows current scheduler component, request/step-state and descriptor contracts, imports the ASR multimodal format enum from its current core module, and accepts/forwards ASR attention batch context without changing the fused arithmetic. The original and adapted archive/file hashes are recorded in `source-compatibility.json` and the protocol. This is a common setup port for B0 and every candidate, not an agent improvement; “pristine” validation means this declared baseline. Core source files remain unchanged.
