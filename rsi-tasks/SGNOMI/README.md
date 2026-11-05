# SGNOMI: SGLang-Omni tasks

One task entry per model, with the final non-streaming and streaming task versions. The ten models below cover the eight model families in [OpenRSI × SGLang-Omni: Scaling RSI for AI Infra](https://claude.ai/artifact/QkFUUcF1qsLK7vwFWsGhPo); Qwen3-TTS Base 0.6B / 1.7B and Ming3-TTS 0.5B / Tiny each have their own entry. Historical iterations, reward-weight variants and objective ablations are omitted.

Each model has its own directory below. Task links open complete public copies containing `task.toml`, instructions, environment build files, the evaluator and its fixed assets. All twelve packages are available in this repository; the original task directory names are retained inside the ten model directories.

## Model index

| Model | Final task package | Evaluation modes | Related merged upstream PRs |
| --- | --- | --- | --- |
| CosyVoice3 | [Dual-mode task](cosyvoice3/sglang-omni-cosyvoice3-dual-mode-runtime/) | Non-streaming + streaming audio | [#2524](https://github.com/sgl-project/sglang-omni/pull/2524), [#2555](https://github.com/sgl-project/sglang-omni/pull/2555) |
| MOSS-TTS v1.5 | [Dual-mode task](moss-tts-v1.5/sglang-omni-moss-v1.5-dual-mode-runtime/) | Non-streaming + streaming audio | [#2525](https://github.com/sgl-project/sglang-omni/pull/2525), [#2859](https://github.com/sgl-project/sglang-omni/pull/2859) |
| Fish S2-Pro | [Dual-mode task](fish-s2-pro/sglang-omni-fish-s2-pro-dual-mode-runtime/) | Non-streaming + streaming audio | [#2778](https://github.com/sgl-project/sglang-omni/pull/2778) |
| Qwen3-TTS Base 0.6B | [Dual-mode task](qwen3-tts-0.6b/sglang-omni-qwen3-tts-base-0.6b-dual-mode-runtime/) | Non-streaming + streaming audio | [#2779](https://github.com/sgl-project/sglang-omni/pull/2779) |
| Qwen3-TTS Base 1.7B | [Dual-mode task](qwen3-tts-1.7b/sglang-omni-qwen3-tts-base-1.7b-dual-mode-runtime/) | Non-streaming + streaming audio | [#2779](https://github.com/sgl-project/sglang-omni/pull/2779) |
| Qwen3-ASR 1.7B | [Dual-mode task](qwen3-asr-1.7b/qwen3-asr-streaming-throughput/) | Non-streaming + post-upload SSE text streaming | [#2880](https://github.com/sgl-project/sglang-omni/pull/2880), [#2881](https://github.com/sgl-project/sglang-omni/pull/2881), [#2882](https://github.com/sgl-project/sglang-omni/pull/2882) |
| Whisper large-v3 | [Dual-mode task](whisper-large-v3/whisper-asr-streaming-throughput/) | Non-streaming + post-upload SSE text streaming | [#2901](https://github.com/sgl-project/sglang-omni/pull/2901) |
| MOSS-Transcribe-Diarize | [Dual-mode task](moss-transcribe-diarize/moss-transcribe-diarize-streaming-throughput/) | Non-streaming + post-upload SSE text streaming | [#3071](https://github.com/sgl-project/sglang-omni/pull/3071) |
| Ming3-TTS 0.5B | [Non-streaming](ming3-0.5b/sglang-omni-ming3-0.5b-nonstream/) / [Streaming](ming3-0.5b/sglang-omni-ming3-0.5b-streaming/) | Separate packages for the two modes; see below | [#3136](https://github.com/sgl-project/sglang-omni/pull/3136), [#3190](https://github.com/sgl-project/sglang-omni/pull/3190) |
| Ming3-TTS Tiny | [Non-streaming](ming3-tiny/sglang-omni-ming3-tiny-nonstream/) / [Streaming](ming3-tiny/sglang-omni-ming3-tiny-streaming/) | Separate packages for the two modes; see below | [#3189](https://github.com/sgl-project/sglang-omni/pull/3189) |

The dual-mode tasks evaluate both modes in one task and require both throughput ratios to improve. The listed versions use streaming:non-streaming reward weights of 2:1. CosyVoice3 uses the final guided task on SGLang-Omni 0.1.9 at `fa90213e1674`.

Ming3 has one index entry per model, with the final package for each mode linked in that entry. Its existing packages have separate evaluators and baselines: non-streaming uses `ca9e8789fcd8`, while native streaming uses `0c663861d7d4` with the declared compatibility port. They are not a combined dual-mode evaluator; follow each package's own protocol and calibration.

The TTS and ASR task families originate from proposals [#154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154) and [#257](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/257). Upstream PRs are related results for the model family, not measurements against every final task baseline.

## Run a task

Open the selected package and follow its README, instruction.md and task.toml. Use this repository's [RSI-Harness](../../RSI-Harness/) to compile and run it on resources matching the task contract. Preserve the pinned model assets, runtime, evaluation protocol and scoring rules. A new run needs a fresh matching environment validation and calibration; source-run baseline statements are historical context.

Each package includes publication.json with its source commit and original file hashes. Source archives, dependency locks, evaluator code and fixed data are retained. Model weights and dependencies declared for build-time download remain pinned in the package. No GPU evaluation was performed for this publication.

Task packages copied and mode coverage checked on November 5, 2026.
