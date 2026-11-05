# MOSS-Transcribe-Diarize model adaptation

This private task derives from Qwen D257 commit `69d933bd2884b094b098d8e1514648822e9d6492`. Authors: Zhuofeng Li, Chenyang Zhao, db-oi. Contact: zhuofengli12345@gmail.com. The contributor authorized direct creation without another Discussion and confirmed baseline-relative quality guards for this model.

The source is pinned at `8220d220703cf8d8f5c5d9a2cde00b5ed13717b8`, upstream SGLang-Omni `main` on 2026-10-17, packaged unchanged as its GitHub archive (see the 2026-10-17 section below). The model is `OpenMOSS-Team/MOSS-Transcribe-Diarize` at `704aa4a9c304e8520be88901e0d1960158ef5b15`, in native BF16 precision. The model manifest includes byte hashes from that immutable Hugging Face revision. Weights and tokenizer stay fixed. Dependency locks, public SeedTTS input order and all 3,108 rows are inherited unchanged.

## Where B0 error values live

There are no measured B0 error values in this repository. `tests/comparison-state.json` deliberately starts with `phase: unprepared`. Run the documented `tools/launch.py --prepare` on the selected single GPU to measure B0 and a full A/A check before research. The operator's `$RSI_STATE_DIR/state.json` then stores four baseline means under `b0.means[workload].corpus_wer`, with workloads `nonstream/en`, `nonstream/zh`, `stream/en`, `stream/zh`. `b0.repeats` retains all three observations for each workload. The launcher injects the protected state into Judge; candidate files cannot supply these numbers.

For each workload, its quality ceiling is **1.03 × its own initial B0 mean error**. This means 3% relative increase, not three percentage points. The same rule applies to A/A and research. If B0 error is zero, the ceiling is zero. The ceiling never changes when Bk advances. Do not substitute source CI constants or another model's B0. Calibration accepts complete finite measurements as the model's initial reference; it does not fabricate a fixed 2% baseline gate.

Both throughput modes, 2:1 streaming reward exponents, the 3% Bk performance/latency guards, concurrency 32, 12 fresh-service scored passes, 18,648 scored requests and 18,648 warmup requests remain as in the parent. Work and Judge use one GPU serially. Runtime networking remains disabled.

## Common baseline adaptations

`environment/adapt_baseline.py` applies the compatibility changes to pristine source during image construction, before copying the starting workspace or calibrating B0. `solution/solve.sh` restores that same adapted baseline. The immutable policy is generated from the resulting model-specific AST, rather than copied from Qwen.

Encoder cache lookup/write and fingerprint-keyed in-flight sharing are disabled while ordinary batching remains available. Failed batches are not silently retried per item. The text adapter decodes cumulative token prefixes at the fixed 50 ms emission policy, holds incomplete Unicode and trailing whitespace, and flushes at termination. Its concatenated deltas must exactly equal the native final text; this preserves the strict inherited SSE check. The adapter is part of both B0 and every candidate and earns no standalone optimization credit.

MOSS retains its native transcribe-and-diarize prompt and duration-scaled output-token budget. Both the model encoder cache and the separate pre-LM service cache are disabled. The compatibility stream adapter adds terminal flushing and preserves native full-output postprocessing, including timestamp/speaker markers.

Judge times and transfers the full marked output. Quality is scored as in the other D257 ASR tasks: `tests/moss_text.py` removes special tokens and native timestamp (`[12.34]`) and speaker (`[S01]`) markers, and the remaining text gets the same English WER/Chinese character-token error as Qwen and Whisper. Segment structure, timestamps, speaker labels and audio duration are not validated, and no output is rejected for its markup; ordinary bracketed words stay speech. Rows are never dropped. Raw first-text TTFT may start with a timestamp marker. This public SeedTTS ASR slice does not measure diarization accuracy, DER, cpCER or long-meeting performance.

## Validation scope

Repository checks validate syntax, immutable assets, package manifests, model-specific baseline patching, quality boundaries and text-adapter behavior. GPU/environment validation and actual B0/A/A values remain pending. The parent generation PASS is not evidence of variant GPU performance or quality.


## Speaker-only output protocol repair (superseded on 2026-10-17)

Two non-scored probes of the failed Radix B0 reproduced three native speaker
markers with text spans and no timestamps; the old timestamp-only parser
rejected all spans. The revised parser accepts this complete speaker-only form
without changing the native prompt, generation, wire output or timing. This
repairs the ASR extraction contract; it does not establish timestamp or
speaker-attribution accuracy. No raw benchmark answers are included in tests.

Parser failures now expose a fixed structure reason, without generated text or
reference data. The immutable-manifest digest changes the protocol identity;
old comparison state is rejected by the launcher. Preserve failed runs and
start a fresh B0 and full strict A/A before research under this protocol. CPU
regressions are not a GPU calibration or a scored performance result.

## 2026-10-17: ASR-identical scoring and current SGLang-Omni

The repaired speaker-only parser still rejected the B0 run that followed it
(`Incomplete MOSS segment coverage` on `nonstream/en`). At the contributor's
request, quality scoring now matches the other D257 ASR tasks: the Judge no
longer validates MOSS segment structure, and `speech_text` only strips native
markers before normalization (see above). Error metrics, quality ceilings,
throughput objectives, guards, data and timing are unchanged.

The source archive moves from `eb7e97ed9438ef4e4352656aec6170aeeb90966f`
(2026-10-14) to upstream `main` at `8220d220703cf8d8f5c5d9a2cde00b5ed13717b8`
(29 commits later). `sglang_omni/models/moss_transcribe_diarize/` and
`pyproject.toml` are byte-identical between the two, `environment/adapt_baseline.py`
applies unchanged, and the dependency locks are kept. The immutable-manifest
digest changes the protocol identity, so earlier comparison state is rejected
and a fresh B0 and full A/A are required.
