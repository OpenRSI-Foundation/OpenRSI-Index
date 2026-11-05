# MOSS-Transcribe-Diarize streaming throughput

Improve fixed-weight MOSS-Transcribe-Diarize serving throughput in both non-streaming and post-upload SSE modes on one GPU, preserving accuracy and latency.

## Goal

- Test falsifiable hypotheses about encoder batching, admission, host/GPU overlap, decoding execution and kernels.
- Improve both mode-level throughput ratios against the last accepted candidate Bk. Streaming has twice the reward exponent of non-streaming.
- Explain the mechanisms and trade-offs in your research records.

## Workspace

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Source | `/workspace/sglang-omni` |
| Configuration | `/workspace/candidate.yaml`, initially `{}` |
| Research records and synthetic diagnostics | `/workspace/research/` |
| Fixed model/tokenizer | `/opt/models/moss-transcribe-diarize` |
| Public restoration assets | `/opt/asr-reference` |
| Python/tooling | `/opt/asr-venv/bin/python` |
| Deliverable | Complete source/configuration files and concise research records in the paths above |

## Reference baseline

Baseline: B0 is the starting MOSS-Transcribe-Diarize implementation with BF16, greedy decoding, compilation enabled, upstream preprocessing/language hints/output-token budgets, encoder batching, both encoder caches disabled and zero multimodal embedding cache, disabled radix cache, disabled fingerprint-keyed in-flight merging, disabled batch-failure retries/recovery, and a fixed 0.05-second emission policy. The common compatibility adapter in MODEL_ADAPTATION.md is included before B0 calibration and in every candidate. Candidate batch and queue policies are tunable.

Reported result: Not reported for the fixed protocol. Status: Not reproduced. No historical or other-model measurement initializes B0. Comparison: The operator calibrates B0 and runs a complete identical-baseline A/A check before research. B0 normalization is 1. Normal submissions measure only your candidate; an unchanged candidate receives no exemption from strict improvement.

## Research loop

1. Inspect the baseline and prior aggregate feedback with `rsi-submit --list`.
2. Record one hypothesis and change the permitted execution surface.
3. Run local synthetic smoke tests/profiling if useful. Produce and close all source/configuration files in Work, stop GPU processes, and submit with `rsi-submit`.
4. Compare repeat aggregates and failed conditions; retain, revise or reject the hypothesis.

## What you may change

Source directories below `/workspace/sglang-omni/sglang_omni/`: `models/moss_transcribe_diarize`, `scheduling`, `comm`, `http`, `serve`, `model_runner`, `pipeline`, `relay`, `sampling`, `utils`, and `kernels`. Changes may implement batching, queues, coalescing, asynchronous overlap, graph capture, compiler internals and memory management. Fixed preprocessing, token budgets, language handling and cold-input policy functions remain protected within these surfaces. Numerical reordering is permitted subject to quality gates.

`/workspace/candidate.yaml` is a flat mapping of optional dotted keys. Omitted keys retain upstream defaults. Supported factory keys have the prefix `asr.factory.`: `encoder_max_batch_size`, `request_build_max_workers`, `request_build_max_pending`, `prefill_coalesce_requests`, `prefill_coalesce_wait_ms`, `prefill_coalesce_when_idle`, `prefill_coalesce_requires_pending_builds`, and `prefill_coalesce_after_builds_during_decode`. Supported engine keys have the prefix `asr.engine.`: `max_running_requests`, `torch_compile_max_bs`, `cuda_graph_max_bs`, `disable_cuda_graph`, `mem_fraction_static`, and `kv_cache_bytes`. Use upstream types/ranges. The byte-capacity field `kv_cache_bytes` accepts positive integers below 2^63, including multi-GiB allocations such as 8589934592 (8 GiB); this parser bound does not guarantee device fit. Other numeric parser bounds are below 2^31. Coalescing requests is 0 or at least 2; its wait is positive. Batch sizes, worker counts and capacities are positive. Memory fraction lies strictly between 0 and 1. Do not set both memory fraction and KV byte capacity. Python serving-source files must fit the 8 MiB parser bound; configuration is at most 64 KiB. Put research notes and synthetic inputs only in `/workspace/research/`.

## What stays fixed

- Weights, tokenizer, BF16, greedy decoding, language hints, preprocessing, output-token budgets, dependencies, other models, benchmark/metrics code, manifests, timing, B0/Bk state, task policy, client concurrency and data.
- One GPU; Work and Judge time-share the same exclusive card. Release GPU workloads before submitting. Judge starts fresh processes from a read-only snapshot; processes, sockets, caches and GPU state are not deliverables. Compilation scratch is separate from your workspace.
- No training, checkpoint replacement, extra GPUs, external services/data, answer lookup, evaluation detection, fabricated outputs, cross-request reuse of answers/features/audio-dependent KV, or complete-transcript buffering disguised as streaming. Synthetic data is allowed only for local diagnostics, never as added scored/training data.
- Work, Agent and Judge have no external research network access. The operator separately configures Agent provider routing. Judge only contacts its local candidate service.

## Evaluation and feedback

Every submission evaluates all EN 1088 and ZH 2020 rows, each in non-streaming and SSE mode, with three independent repetitions: 12 scored passes, 18,648 scored requests and 18,648 warmup requests. Each pass starts a fresh service with empty compiler caches, performs one complete successful warmup and then one scored pass. No extra run0, startup warmup requests, filtering, deduplication, replacement, resampling or failed-request retries are allowed. Native order and closed-loop concurrency 32 are fixed. Per-request timeout is 300 seconds. Original reference audio is transcribed against its reference text.

Timing spans request submission through complete receipt. QPS is requests/wall seconds; RTFx is audio seconds/wall seconds; RTF is request latency/audio duration. English word and Chinese character-token errors use corpus edit counts. Percentiles use linear interpolation. Metrics are compared without rounding and averaged arithmetically across three repeats.

SSE must contain ordered nonempty text deltas, final text and DONE; concatenated deltas must match final text. Empty deltas do not start TTFT. Inter-chunk intervals use successive nonempty deltas. Single-delta short outputs are valid; unavailable intervals stay unavailable. A three-repeat interval aggregate is unavailable if any repetition has no interval; available repeat-level values remain visible. There is no minimum chunk count, fixed segmentation or universal measured 50-ms bound. Emission and buffering diagnostics are reported.

For mode m, define `R_m(C;B) = sqrt((Q[m,en](C)/Q[m,en](B)) * (Q[m,zh](C)/Q[m,zh](B)))` using three-repeat mean QPS. Maximize the finite, dimensionless, unclipped reward `R_stream(C;B0)^(2/3) * R_nonstream(C;B0)^(1/3)`.

Acceptance requires both mode ratios against Bk strictly greater than 1; each workload QPS and RTFx at least 97% of Bk; mean/p95 latency and RTF at most 103%; streaming mean/p95 TTFT and available inter-chunk metrics at most 103%; and each workload's three-run corpus-error mean at most 1.03 times the same language/mode's initial measured B0 corpus-error mean. Cumulative changes against B0 are reported without a cumulative 3% gate. Only a complete accepted candidate receives reward and advances Bk. Failed gates, invalid output, attributable compilation/OOM/illegal-memory/crash failures, timeouts and incomplete coverage are unscored. Infrastructure/evaluator failures are diagnosed separately. Bk never advances on failure.

Feedback includes repeat aggregates, means/ranges, coverage, ratios, failed conditions and safe repair diagnostics. References, transcripts, per-request correctness, answer mappings and protected scoring implementation are withheld. Every emitted byte and the Harness footer are Agent-visible through the durable submission log. Public-data memorization and inherited interpreter/library tampering remain limitations of the shared runtime.

## Submission checklist

- Complete and flush the candidate's files under `/workspace`.
- Finish candidate-producing compute and release all GPU processes.
- Keep dependencies, fixed assets and protected source untouched; do not require live Work state in Judge.
- Record the next hypothesis. A submission may consume the full three-hour Judge allowance plus Harness overhead.

The default run budget is 24 hours; the operator may select the approved 48-hour profile. The submission ceiling is 64, not a target of 64 experiments. Ten conservative complete loops use 42.5 hours plus 5.5 hours of margin in that extended profile; these allowances are unmeasured.

MOSS output keeps its native timestamp and speaker markers during generation, transfer and timing. As in the other ASR tasks, Judge scores transcript quality only: it removes native timestamp and speaker markers and computes ASR error on the remaining text, without validating segment structure. SeedTTS has no diarization reference labels, so this task does not score DER or cpCER. Raw-text TTFT may begin with a timestamp or speaker marker.
