# SGLang Omni TTS runtime — recorded research

One research archive for [the task](../../rsi-tasks/sglang-omni-tts-runtime/) proposed in [Discussion #154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154). It contains three recorded segments by **Codex · GPT-6 Astra · xhigh**, not three independent tasks or a two-model comparison.

Source: `OpenRSI-Foundation/sglang-omni-tts-runtime-d154`, commit `c819dc10aaf7dab0c2d0ed5eea5499e449a6a9d5`. Task version 2.0.0 / protocol v5 is the single current published task; historical instructions and measurements are retained here as evidence, not republished as runnable task versions.

## Recorded segments

| Segment | Persisted submissions | Complete throughput measurements | Numeric task scores |
| --- | ---: | ---: | ---: |
| `d154-astra-xhigh-650175-20261001T230314` | 3 | 3 | 0 |
| `d154-rolling-astra-xhigh-655820-20261002T120036` | 6 | 6 | 0 |
| `omni-resume3-669485-20261004T141956Z-resume3` | 5 | 2 | 1 |

Every persisted evolution-index entry, submission report, feedback log and complete agent transcript is preserved. The 13 unscored records remain `null`, not zero. Gaps in round IDs are original. Only the middle segment contains a final-result file; no missing final result or closing summary is invented.

The earlier research used four-model checks, including Qwen CustomVoice. The continuation's eligible Judge result has three lanes: CosyVoice3, Qwen Base and MOSS. Its original prompt is retained, including historical four-model wording. The newly published protocol v5 also fixes repetition seeds and requires a fresh matching calibration; **the archived result is not claimed to be a fresh run of that package**. Recorded H200/Slurm/Apptainer research and a published Docker build declaration are likewise distinct; no Docker or GPU validation was rerun for publication.

## Formal Judge result

The continuation's `agent-4` feedback records:

- Initial CosyVoice3 mean throughput: **9.232 requests/s**.
- Candidate mean throughput: **14.207333333333333 requests/s**.
- Official task reward: **1.5389225880993647**, the ratio of those means.
- All 33 reported per-model guards pass; later `agent-5` has no score.

[results.json](results.json) extracts this exact Judge record and identifies its source feedback and candidate digest. It is not a reinterpretation of the generated research summary, nor proof that the host operator subsequently promoted the accepted baseline. The three runs' original `evolve_state.json` files remain authoritative for submission scores and result timestamps.

## Measured performance, including unscored submissions

**No reward does not mean no performance measurement.** Eleven submissions have complete Judge feedback with three repetitions per model, all 1,088 TTS/WAV/ASR samples and no failed/skipped rows. Ten of these evaluations fail at least one original guard and receive no reward. Three other submissions have incomplete evaluations; their aggregate throughput stays null. Partial TTS output is not promoted to a complete three-run measurement.

[measurements.json](measurements.json) extracts all 14 submission records, the 11 complete per-model means and repetitions, original guard outcomes, protocol digests, feedback locations/line numbers/SHA-256 values, and recorded times. Throughput is **CosyVoice3 requests/s**, not an inferred reward. The three segments use different protocol hashes; do not join them into a single accepted-baseline history or relabel earlier evaluations under the current protocol.

| Segment | Original round | CosyVoice3 requests/s | Official reward |
| --- | --- | ---: | ---: |
| Initial | agent-1 | 12.916333 | — |
| Initial | agent-2 | 12.580333 | — |
| Initial | agent-8 | 13.243000 | — |
| Rolling | agent-1 | 9.183000 | — |
| Rolling | agent-2 | 13.384000 | — |
| Rolling | agent-5 | 13.172333 | — |
| Rolling | agent-6 | 11.905333 | — |
| Rolling | agent-7 | 13.214000 | — |
| Rolling | agent-13 | 12.302333 | — |
| Continuation | agent-1 | — | — |
| Continuation | agent-2 | — | — |
| Continuation | agent-3 | 14.069000 | — |
| Continuation | agent-4 | 14.207333 | 1.5389225880993647 |
| Continuation | agent-5 | — | — |

The first segment has no matching embedded baseline, so its measurements are not assigned speedup ratios. Later feedback contains the initial reference **9.232 requests/s**. Rolling `agent-7` measures **43.1%** higher throughput, but its original four-model result fails the CustomVoice WER guard. Continuation `agent-3` measures **14.069 requests/s** but fails Qwen Base's `rtf_p99` guard. Only continuation `agent-4` passes every original gate. No historical score or status has been rewritten.

### Local controlled studies

The complete continuation transcript also contains two local matched CosyVoice-only studies, each with three complete runs per side. These use diagnostic ASR rather than full Judge evaluation; H20 separates generation and transcription. Their controls differ, and their improvements must not be added together or treated as task rewards.

| Study | Control mean requests/s | Candidate mean requests/s | Change |
| --- | ---: | ---: | ---: |
| H19 reference-feature transport | 13.456667 | 14.222000 | +5.69% |
| H20 typed-byte transport | 14.019333 | 14.341000 | +2.29% |

The six individual runs in each study, diagnostic WER and exact restored transcript line ranges/hash are included in `measurements.json`. To regenerate the Judge extraction without GPU work, run:

```bash
python3 rsi-logs/sglang-omni-tts-runtime/summarize_measurements.py --index . --output /tmp/sglang-measurements.json
```

To include the two local studies, additionally pass `--transcripts` pointing at a folder containing the fully restored sanitized `codex-gpt-6-astra/runs/.../sglang-omni-tts-runtime/agent_output.txt` tree. The extractor checks complete coverage, recalculates all reported means from repetitions, preserves the original scores, and does not execute the evaluator.

## Upstream contributions

The research led to three merged SGLang-Omni PRs:

- [#2524](https://github.com/sgl-project/sglang-omni/pull/2524): park idle CosyVoice3 speaker-encoder workers.
- [#2525](https://github.com/sgl-project/sglang-omni/pull/2525): reuse MOSS input-embedding accumulators.
- [#2555](https://github.com/sgl-project/sglang-omni/pull/2555): avoid reference-feature list conversion between CosyVoice3 stages.

Their matched serving studies have their own commands and workloads. Do not sum their improvement percentages, assign them additional task scores, or treat them as three separate tasks.

## Transcript restoration and integrity

Transcripts are losslessly gzip-compressed **after sanitization**, without line or size truncation. The largest complete gzip is split into two ordered parts below GitHub's 100-MiB per-file limit. In its `agent_output.txt.gz.parts/` folder, restore the full sanitized text with:

```bash
cat part-0001.bin part-0002.bin | gzip -dc > ../agent_output.txt
```

For an unsplit transcript:

```bash
gzip -dc agent_output.txt.gz > agent_output.txt
```

The chunk manifest gives every part's size/hash, the concatenated gzip size/hash and restored text size/hash/line count. [publication.json](publication.json) records all source-derived published files, SHA-256 values, source-file inventories, compression metadata and redaction counts. It preserves source numerics, timestamps, scores, rewards, statuses and guard results; private paths, site/host names, private IPs, GPU UUIDs and credentials are sanitized. Credential-pattern redaction counts are zero in this source snapshot; automated scanner matches remain subject to manual review.

Original private publication receipts are retained as `source-publication.json` so they cannot be mistaken for the new public-byte manifest. This archive contains no weights, benchmark dataset or external evaluation credential.
