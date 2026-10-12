# SGLang Omni TTS runtime — recorded research

One research archive for [the task](../../rsi-tasks/sglang-omni-tts-runtime/) proposed in [Discussion #154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154). It contains three recorded segments by **Codex · GPT-6 Astra · xhigh**, not three independent tasks or a two-model comparison.

Source: `OpenRSI-Foundation/sglang-omni-tts-runtime-d154`, commit `c819dc10aaf7dab0c2d0ed5eea5499e449a6a9d5`. Task version 2.0.0 / protocol v5 is the single current published task; historical instructions and measurements are retained here as evidence, not republished as runnable task versions.

## Recorded segments

| Segment | Persisted submissions | Numeric task scores |
| --- | ---: | ---: |
| `d154-astra-xhigh-650175-20261001T230314` | 3 | 0 |
| `d154-rolling-astra-xhigh-655820-20261002T120036` | 6 | 0 |
| `omni-resume3-669485-20261004T141956Z-resume3` | 5 | 1 |

Every persisted evolution-index entry, submission report, feedback log and complete agent transcript is preserved. The 13 unscored records remain `null`, not zero. Gaps in round IDs are original. Only the middle segment contains a final-result file; no missing final result or closing summary is invented.

The earlier research used four-model checks, including Qwen CustomVoice. The continuation's eligible Judge result has three lanes: CosyVoice3, Qwen Base and MOSS. Its original prompt is retained, including historical four-model wording. The newly published protocol v5 also fixes repetition seeds and requires a fresh matching calibration; **the archived result is not claimed to be a fresh run of that package**. Recorded H200/Slurm/Apptainer research and a published Docker build declaration are likewise distinct; no Docker or GPU validation was rerun for publication.

## Formal Judge result

The continuation's `agent-4` feedback records:

- Initial CosyVoice3 mean throughput: **9.232 requests/s**.
- Candidate mean throughput: **14.207333333333333 requests/s**.
- Official task reward: **1.5389225880993647**, the ratio of those means.
- All 33 reported per-model guards pass; later `agent-5` has no score.

[results.json](results.json) extracts this exact Judge record and identifies its source feedback and candidate digest. It is not a reinterpretation of the generated research summary, nor proof that the host operator subsequently promoted the accepted baseline. The three runs' original `evolve_state.json` files remain authoritative for submission scores and result timestamps.

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
