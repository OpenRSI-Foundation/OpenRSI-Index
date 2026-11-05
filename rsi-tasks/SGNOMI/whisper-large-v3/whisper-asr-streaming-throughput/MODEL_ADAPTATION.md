# Whisper ASR model adaptation

This private task derives from Qwen D257 commit `69d933bd2884b094b098d8e1514648822e9d6492`. Authors: Zhuofeng Li, Chenyang Zhao, db-oi. Contact: zhuofengli12345@gmail.com. The contributor authorized direct creation without another Discussion and confirmed baseline-relative quality guards for this model.

The source stays pinned at `eb7e97ed9438ef4e4352656aec6170aeeb90966f`. The model is `openai/whisper-large-v3` at `06f233fe06e710322aca913c1bc4249a0d71fce1`, in native FP16 precision. The model manifest includes byte hashes from that immutable Hugging Face revision. Weights and tokenizer stay fixed. The complete original source archive, dependency locks, public SeedTTS input order and all 3,108 rows are inherited unchanged.

## Where B0 error values live

There are no measured B0 error values in this repository. `tests/comparison-state.json` deliberately starts with `phase: unprepared`. Run the documented `tools/launch.py --prepare` on the selected single GPU to measure B0 and a full A/A check before research. The operator's `$RSI_STATE_DIR/state.json` then stores four baseline means under `b0.means[workload].corpus_wer`, with workloads `nonstream/en`, `nonstream/zh`, `stream/en`, `stream/zh`. `b0.repeats` retains all three observations for each workload. The launcher injects the protected state into Judge; candidate files cannot supply these numbers.

For each workload, its quality ceiling is **1.03 × its own initial B0 mean error**. This means 3% relative increase, not three percentage points. The same rule applies to A/A and research. If B0 error is zero, the ceiling is zero. The ceiling never changes when Bk advances. Do not substitute source CI constants or another model's B0. Calibration accepts complete finite measurements as the model's initial reference; it does not fabricate a fixed 2% baseline gate.

Both throughput modes, 2:1 streaming reward exponents, the 3% Bk performance/latency guards, concurrency 32, 12 fresh-service scored passes, 18,648 scored requests and 18,648 warmup requests remain as in the parent. Work and Judge use one GPU serially. Runtime networking remains disabled.

## Common baseline adaptations

`environment/adapt_baseline.py` applies the compatibility changes to pristine source during image construction, before copying the starting workspace or calibrating B0. `solution/solve.sh` restores that same adapted baseline. The immutable policy is generated from the resulting model-specific AST, rather than copied from Qwen.

Encoder cache lookup/write and fingerprint-keyed in-flight sharing are disabled while ordinary batching remains available. Failed batches are not silently retried per item. The text adapter decodes cumulative token prefixes at the fixed 50 ms emission policy, holds incomplete Unicode and trailing whitespace, and flushes at termination. Its concatenated deltas must exactly equal the native final text; this preserves the strict inherited SSE check. The adapter is part of both B0 and every candidate and earns no standalone optimization credit.

Pinned Whisper did not wire an incremental text stream builder. This task adds that builder through the existing stage/scheduler interface; it does not claim stock Whisper already emitted deltas. Native FP16, greedy transcription, language prefixes, suppression rules, 256-token default and 448-position decoder-context limits are retained. No unsupported Qwen multimodal-cache flag is passed. Every pinned SeedTTS audio header was audited: maximum EN 8.8054 s and ZH 10.1547 s, below Whisper's 30 s streaming limit; no row is filtered.

## Validation scope

Repository checks validate syntax, immutable assets, package manifests, model-specific baseline patching, quality boundaries and text-adapter behavior. GPU/environment validation and actual B0/A/A values remain pending. The parent generation PASS is not evidence of variant GPU performance or quality.
