# Ming-3.0-tts-0.5B: streaming TTS runtime research

Improve streaming English TTS requests/s at concurrency 16 through default-on runtime scheduling, dispatch, inter-stage transport, allocation/state reuse, reference-conditioning reuse or pipeline organization. Requests must use native stream=true and PCM; every chunk must carry valid timing/duration evidence. Task version 6.0.0 / protocol v10 retains the reference task's streaming evaluation.

## Baseline and model

- SGLang-Omni main with the declared common setup compatibility port at `0c663861d7d4a558e4f53258c4d48cfce577b47e`; common setup changes are declared below.
- Official checkpoint `inclusionAI/Ming-3.0-tts-0.5B` at `8d66b70a00f05ad183fa8a863b4ce609fc83cbc0`. Exact assets and dependency hashes are in the paired environment/tests inventories.
- Request token cap remains 2,048. The model's native clamp and sampling semantics remain fixed: 1,000 latent steps, temperature 0, minimum 5 decode steps; reference audio plus transcript supplies prompt latents. Audio is native 44.1 kHz. Do not replace one checkpoint's conditioning or defaults with the other's.
- Complete fresh B0 on the exact prepared image, evaluator, GPU and CPU allocation, then immediately begin eight-hour research. Independent A/A is waived by explicit user instruction. No previous model/task calibration or optimized candidate is reused.

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

Build the environment and verify pristine scope, offline assets, source/dependency identities and actual resource mapping. Inject tests read-only and run `/opt/sglang/bin/python3 -I /tests/calibrate.py`. Supply the assigned UUID and actual CPU/NUMA arrays via the three runtime variables declared in `task.toml`. Freeze the complete calibration with `pool_calibrations.py`; bind `RSI_TTS_BASELINE_JSON` to Judge only. Recalibrate after any identity change. Independent A/A is explicitly waived by the user. Launch research immediately after full B0; candidate evaluation retains the unchanged native Judge path and every acceptance gate.

Feedback includes per-repetition metrics, means, coverage, timings, resource peaks, contention, safe diagnostics, signed/relative differences and every guard decision, Bk generation/source, B0 comparisons and cumulative reward. Withhold per-example ASR hypotheses and answer/error-ranking tables. WER alone does not establish speaker similarity, fidelity, prosody or generalization. Public workload adaptation and measurement noise remain limitations.

The bundled RSI-Harness remains read-only. Keep large assets, environments and audio on Radix; continuously preserve minimum result/recovery state outside the node. Release the assignment after all authorized work and final persistence verification complete.

## Source refresh

SGLang-Omni `0c663861d7d4a558e4f53258c4d48cfce577b47e`; compatible SGLang core `b8100b74bf4f6563f402c20a1e1af685cbd2246e`. Omni is frozen at the latest observed main; core uses the already prepared compatible revision. Both remain fixed throughout B0 and research; no old calibration, accepted candidate, PR3217 patch or research clock is reused. Source task: `4ab5b1bc0d7ddb825d3a363d3e9dc36a04c7fd8c`.

The contributor explicitly set the formal research budget to 8 hours on 2026-11-02. Setup and full B0 precede the research clock. Independent A/A and repeated subset smoke are omitted by explicit user instruction. Stop at formal Agent start + 28,800 seconds, preserve verified candidates and finish already accepted Judge work, then prepare evidence-backed upstream PRs. Do not extend or reset the research clock.

The latest core removed the old `rmsnorm_hf` module in #47503. The task declares an installed API bridge in `environment/core_compat.py`: it allocates an output and calls the latest unified kernel with `cast_x_before_out_mul=True`. Core source files remain unchanged; the additional Omni source port is declared below. The bridge is included in the full protected software fingerprint, is shared by B0 and every candidate, and is not an optimization result. Its GPU output, input-ownership, explicit-output and CUDA graph behavior must pass `generation/validate_core_compat_gpu.py` before research.

The baseline also contains the explicit `upstream-compatibility.patch`: Omni forwards the latest core draft-layout arguments, obtains cache slots through the current storage interface, follows current scheduler component, request/step-state and descriptor contracts, imports the ASR multimodal format enum from its current core module, and accepts/forwards ASR attention batch context without changing the fused arithmetic. The original and adapted archive/file hashes are recorded in `source-compatibility.json` and the protocol. This is a common setup port for B0 and every candidate, not an agent improvement; “pristine” validation means this declared baseline. Core source files remain unchanged.


## Public task copy

This package is published in OpenRSI-Index. [Publication provenance](publication.json) records the source repository, exact commit and hashes of every original task file. The environment, evaluator, fixed assets, source archives and scoring contract are copied unchanged. Historical setup statements describe the source run; a new run requires its own matching environment validation and calibration. No GPU evaluation was performed for this publication.
