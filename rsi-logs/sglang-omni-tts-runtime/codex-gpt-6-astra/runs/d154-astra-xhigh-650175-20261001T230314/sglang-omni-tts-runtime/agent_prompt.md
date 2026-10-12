# Default-on TTS runtime research

Improve CosyVoice3 non-streaming English TTS requests/s at concurrency 16 while preserving every quality and performance guard across four models.

## Goal

Study whether scheduling, CPU dispatch, inter-stage transport, allocation/state reuse, reference-conditioning reuse or pipeline organization removes a measurable bottleneck. Keep a falsifiable hypothesis and controlled ablation for each change. Qwen3-TTS Base, Qwen3-TTS CustomVoice and MOSS-TTS-v1.5 are mandatory guards; their results do not enter an averaged objective.

## Workspace

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Source | `/workspace/sglang-omni` contains the complete starting runtime |
| Deliverable | Closed source plus hypotheses, profiles, ablations, experiment history, reproduction manifest and source diff under `/workspace/research` |
| Public restoration source | `/opt/rsi-source` |
| Dependencies and released assets | `/opt/rsi-deps` and `/opt/rsi-assets` |

## Reference baseline

| Item | Value |
| --- | --- |
| Baseline | Unmodified starting runtime with released weights, original command defaults and the fixed repetition seed schedule below |
| Reported result | Not reported for this fixed protocol; historical 12.63 requests/s, WER 0.0183 and 1.222 s latency used a different protocol and are not gates |
| Status | Not reproduced during authoring; operator calibration is required before research |
| Comparison | The operator freezes twelve matched baseline runs before research; normal submissions evaluate only the candidate against that calibration |

A new unchanged evaluation may fail strict guards through noise; it is never assigned an artificial score of 1.

## Research loop

1. Inspect the baseline, profile one bottleneck and inspect earlier feedback with `rsi-submit --list`.
2. Record a causal hypothesis, edit the permitted runtime surface, and run controlled Work-side profiles, ablations and state/abort tests.
3. Finish all processes, save complete source and research records, release GPU processes, then submit with `rsi-submit`.
4. Compare all aggregates and guards. Keep, revise or reject the hypothesis without replacing earlier records or cherry-picking repetitions.

Before any batch-related experiment, trace configuration from benchmark CLI/defaults through managed server arguments, config merging, engine initialization and effective scheduler/CUDA graph settings. The original `benchmark_tts_seedtts.py` defaults to `max_running_requests=64` and `cuda_graph_max_bs=64` and forwards both as stage-specific server CLI overrides for all four models even when those flags are omitted. A lower-level model config edit can therefore have no effect. Client concurrency 16, running-request capacity 64 and CUDA graph maximum batch size 64 are different quantities. Record the exact benchmark/server commands, value provenance, resolved engine settings and observed batch sizes in `/workspace/research/reproduction.json` and each relevant experiment. An edit shadowed by a higher-priority value is not a tested treatment. Work-only diagnostic overrides must be labelled; restore the fixed evaluation commands/defaults for comparable runs. Do not bypass the protected benchmark or silently change formal evaluation settings.

## What you may change

Edit TTS-relevant code under `/workspace/sglang-omni/sglang_omni` for scheduling, dispatch, communication, pipeline organization, memory/state/reference-conditioning reuse and corresponding CosyVoice3, Qwen and MOSS integrations. This is a semantic boundary: a directory match does not permit changing model behavior. New candidate tests and generated research files belong under `/workspace/research`. Put temporary files and bytecode outside the source tree. No candidate-producing work happens in Judge.

For a changed candidate, provide `/workspace/research/hypothesis.md`, `/workspace/research/experiments.json`, `/workspace/research/reproduction.json` and `/workspace/research/source.diff`, each complete and nonempty. Keep profiles and controlled ablations alongside them. Fixed ASR source and sampling primitives remain protected even inside the source package.

## What stays fixed

- Keep all four released models, weights, architecture, precision, speech tokenizers, sampling settings, the fixed sampler seeds, speed, maximum tokens 2048, English language, voice, reference conditioning and native server defaults. CustomVoice uses Ryan without reference audio; Base and MOSS use references; MOSS token-count is auto. CosyVoice uses default flat references.
- Preserve request isolation, abort cleanup and default-on operation. No new opt-in flag, target-audio/answer cache, sample-specific behavior, evaluator detection, omission, replacement, resampling, truncation, time stretching or fabricated metrics.
- Protect benchmark code and loading, request, timing and metric dependencies, reference ASR, assets, dependencies, interpreter and clocks. Candidate tests do not replace protected tests. Preserve all source outside the permitted surface, including hidden and ignored files; Git metadata is not scoring authority.
- No training, external services, web research, extra datasets/checkpoints, runtime installs/downloads or uploads. Native localhost serving is allowed. The provider route does not authorize research egress.
- Work has one GPU and an intended 16-core/128-GiB budget. Stop its GPU processes before submission. Judge uses four concurrent, disjoint 16-core/128-GiB lanes. Shared Docker CPU/RAM limits do not reserve host cores. Total budget is 24 hours; the operator may extend it to 48 hours.

## Evaluation and feedback

Every evaluation uses all 1,088 EN rows in native order. Each model performs three serial repetitions within its concurrent lane, fresh processes and fresh empty compilation caches each repetition. There are 16 first-sample TTS warmups followed by 1,088 scored requests per repetition: 13,056 scored requests and 192 warmups overall. Stock TTS plus diagnostic ASR is followed by original-source authoritative ASR on the actual WAVs, with concurrency 32 and 64 warmups. Each run needs 1,088 valid WAVs/transcripts, zero failures and zero skips. Startup, compilation, both ASR passes and cleanup count toward wall time; TTS timing retains the stock steady-state window. No per-sample retries are allowed. A retry is a new complete evaluation.

Protocol version 2 fixes sampler seeds to **42, 43, 44** for repetitions 1, 2, 3, respectively, for every model and for both baseline and candidates. The operator passes the original benchmark's `--seed`; every warmup/scored TTS request in a repetition receives that same public seed. `--arrival-seed` stays unset for this closed-loop workload. Reproduce this schedule in matched Work experiments, retain every repetition and verify the emitted benchmark config and request-to-sampler path; do not select seeds after seeing results. The protocol hash invalidates earlier unseeded calibrations. This controls request sampling, not every waveform RNG or GPU scheduling/numerical effect, and does not guarantee bitwise determinism or independent statistical replicates.

For each model, compare arithmetic means of three per-run corpus micro-WER, latency mean/median/p95/p99, RTF mean/median/p95/p99, requests/s and audio-seconds/s. Corpus WER includes examples above 50% WER. Every WER, latency and RTF mean must be no greater than baseline; both throughputs must be no smaller. Average run percentiles, without pooling, tolerances, extra rounding or ratio clipping. Stock precision is retained: three decimals for latency/throughput, four for RTF, computed precision for WER.

After all integrity, validity, coverage, contention and nonregression gates pass, reward is mean CosyVoice3 requests/s divided by its frozen baseline mean; maximize it. Equality yields 1 and improvement requires greater than 1. No finite failure scalar exists. Gate failures are valid-but-ineligible/unscored; attributable compile failures, illegal memory access, invalid audio, crashes or rule violations are candidate-invalid/unscored. Infrastructure, missing/mismatched calibration, timeout, ambiguous attribution and incomplete evaluation are separately identified and unscored. All leave reward absent, which Harness reports as a verifier error.

Feedback includes per-model/per-repetition aggregates and means, coverage, guards, timings, resource peaks, contention and actionable diagnostics. Full safe stdout/stderr and the Harness footer reach Work, including durable logs. Per-example ASR hypotheses, error rankings/answer tables, protected Judge-check contents and credentials are withheld. Public workload text/audio and upstream evaluator code are public. WER does not establish speaker identity, prosody, fidelity or generalization. Strict guards have noise risk, and repeated adaptation can overfit this public workload. Candidate-authored records are self-reported; shared root/interpreter/library state does not provide arbitrary-root isolation.

## Submission checklist

- All deliverables are closed and flushed under `/workspace`; Judge reloads a read-only snapshot and needs no live process or cache state.
- Source outside the allowed surface is pristine; source caches, logs and installation products are absent.
- Research records state the hypothesis, commands, ablations, complete prior outcomes and limitations.
- GPU processes have stopped and the next submission tests a stated hypothesis.

## Evaluation

The Judge evaluates the entire current WORKDIR as it exists when you submit.
The best valid primary score wins. You may call `rsi-submit` repeatedly to
receive feedback while improving the same workspace.
`rsi-submit --list` shows previous submissions; `rsi-submit --help` shows local usage.
Each submission stores complete Judge stdout and stderr in
`/run/rsi-harness/feedback/agent-N.log`, where N is the submission number.
Judge submissions are unlimited during this run.

Every Work GPU process must exit before rsi-submit. A rejected preflight does not consume a submission.
