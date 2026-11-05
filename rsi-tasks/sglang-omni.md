# SGLang-Omni tasks

One task entry per model, with the final non-streaming and streaming task versions. The ten models below cover the eight model families in [OpenRSI × SGLang-Omni: Scaling RSI for AI Infra](https://claude.ai/artifact/QkFUUcF1qsLK7vwFWsGhPo); Qwen3-TTS Base 0.6B / 1.7B and Ming3-TTS 0.5B / Tiny each have their own entry. Historical iterations, reward-weight variants and objective ablations are omitted.

Task links open the package containing `task.toml`, instructions and the evaluator. The linked task repositories are private and require repository access.

## Model index

| Model | Final task package | Evaluation modes | Related merged upstream PRs |
| --- | --- | --- | --- |
| CosyVoice3 | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-0.1.9-fa90213e1674-tts-cosyvoice3-dual-mode-guided-d154/tree/main/sglang-omni-cosyvoice3-dual-mode-runtime) | Non-streaming + streaming audio | [#2524](https://github.com/sgl-project/sglang-omni/pull/2524), [#2555](https://github.com/sgl-project/sglang-omni/pull/2555) |
| MOSS-TTS v1.5 | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-dual-mode-d154/tree/main/sglang-omni-moss-v1.5-dual-mode-runtime) | Non-streaming + streaming audio | [#2525](https://github.com/sgl-project/sglang-omni/pull/2525), [#2859](https://github.com/sgl-project/sglang-omni/pull/2859) |
| Fish S2-Pro | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-tts-fish-s2-pro-dual-mode-d154/tree/main/sglang-omni-fish-s2-pro-dual-mode-runtime) | Non-streaming + streaming audio | [#2778](https://github.com/sgl-project/sglang-omni/pull/2778) |
| Qwen3-TTS Base 0.6B | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-0.6b-dual-mode-d154/tree/main/sglang-omni-qwen3-tts-base-0.6b-dual-mode-runtime) | Non-streaming + streaming audio | [#2779](https://github.com/sgl-project/sglang-omni/pull/2779) |
| Qwen3-TTS Base 1.7B | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-1.7b-dual-mode-d154/tree/main/sglang-omni-qwen3-tts-base-1.7b-dual-mode-runtime) | Non-streaming + streaming audio | [#2779](https://github.com/sgl-project/sglang-omni/pull/2779) |
| Qwen3-ASR 1.7B | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-d257/tree/main/qwen3-asr-streaming-throughput) | Non-streaming + post-upload SSE text streaming | [#2880](https://github.com/sgl-project/sglang-omni/pull/2880), [#2881](https://github.com/sgl-project/sglang-omni/pull/2881), [#2882](https://github.com/sgl-project/sglang-omni/pull/2882) |
| Whisper large-v3 | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-asr-whisper-large-v3-d257/tree/main/whisper-asr-streaming-throughput) | Non-streaming + post-upload SSE text streaming | [#2901](https://github.com/sgl-project/sglang-omni/pull/2901) |
| MOSS-Transcribe-Diarize | [Dual-mode task](https://github.com/OpenRSI-Foundation/sglang-omni-asr-moss-transcribe-diarize-d257/tree/main/moss-transcribe-diarize-streaming-throughput) | Non-streaming + post-upload SSE text streaming | [#3071](https://github.com/sgl-project/sglang-omni/pull/3071) |
| Ming3-TTS 0.5B | [Non-streaming](https://github.com/OpenRSI-Foundation/sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-0.5b-nonstream/tree/main/sglang-omni-ming3-0.5b-nonstream) / [Streaming](https://github.com/OpenRSI-Foundation/sglang-omni-ming3-0.5b-streaming/tree/main/sglang-omni-ming3-0.5b-streaming) | Separate packages for the two modes; see below | [#3136](https://github.com/sgl-project/sglang-omni/pull/3136), [#3190](https://github.com/sgl-project/sglang-omni/pull/3190) |
| Ming3-TTS Tiny | [Non-streaming](https://github.com/OpenRSI-Foundation/sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-tiny-nonstream/tree/main/sglang-omni-ming3-tiny-nonstream) / [Streaming](https://github.com/OpenRSI-Foundation/sglang-omni-ming3-tiny-streaming/tree/main/sglang-omni-ming3-tiny-streaming) | Separate packages for the two modes; see below | [#3189](https://github.com/sgl-project/sglang-omni/pull/3189) |

The dual-mode tasks evaluate both modes in one task and require both throughput ratios to improve. The listed versions use streaming:non-streaming reward weights of 2:1. CosyVoice3 uses the final guided task on SGLang-Omni 0.1.9 at `fa90213e1674`.

Ming3 has one index entry per model, with the final package for each mode linked in that entry. Its existing packages have separate evaluators and baselines: non-streaming uses `ca9e8789fcd8`, while native streaming uses `0c663861d7d4` with the declared compatibility port. They are not a combined dual-mode evaluator; follow each package's own protocol and calibration.

The TTS and ASR task families originate from proposals [#154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154) and [#257](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/257). Upstream PRs are related results for the model family, not measurements against every final task baseline.

## Run a task

Open the selected task package and follow its repository README and bundled validator and runner skills. Preserve its model assets, runtime pins, evaluation protocol, calibration and scoring contract. See [RSI-Harness](../RSI-Harness/) for the execution runtime.

Task packages and mode coverage checked on November 5, 2026.
