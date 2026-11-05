# SGLang-Omni task repositories

Task repositories for the eight model families in [OpenRSI × SGLang-Omni: Scaling RSI for AI Infra](https://claude.ai/artifact/QkFUUcF1qsLK7vwFWsGhPo), including their model-size, objective and reward variants. Each task-package link opens the directory containing its `task.toml`, instructions and evaluator.

The 27 source repositories below are private and require repository access. The [CosyVoice3 public task package](sglang-omni-tts-runtime/) and its [recorded research archive](https://huggingface.co/datasets/OpenRSI-Foundation/OpenRSI-Index-Logs/tree/main/sglang-omni-tts-runtime) are publicly available. Use each task's own protocol and baseline when running or comparing results.

## Model index

| Model family | Runtime research | Related merged upstream PRs |
| --- | --- | --- |
| [CosyVoice3](#cosyvoice3) | TTS serving, cross-stage feature transport and PackedDiT compilation | [#2524](https://github.com/sgl-project/sglang-omni/pull/2524), [#2555](https://github.com/sgl-project/sglang-omni/pull/2555) |
| [MOSS-TTS](#moss-tts) | Embeddings, CUDA graphs and overlapping decode | [#2525](https://github.com/sgl-project/sglang-omni/pull/2525), [#2859](https://github.com/sgl-project/sglang-omni/pull/2859) |
| [Fish S2-Pro](#fish-s2-pro) | Quantizer and waveform processing | [#2778](https://github.com/sgl-project/sglang-omni/pull/2778) |
| [Qwen3-TTS](#qwen3-tts) | Base 0.6B and 1.7B prefill and vocoder execution | [#2779](https://github.com/sgl-project/sglang-omni/pull/2779) |
| [Qwen3-ASR](#qwen3-asr) | Prefill, decode and encoder scheduling | [#2880](https://github.com/sgl-project/sglang-omni/pull/2880), [#2881](https://github.com/sgl-project/sglang-omni/pull/2881), [#2882](https://github.com/sgl-project/sglang-omni/pull/2882) |
| [Whisper](#whisper) | Incremental text streaming and inference execution | [#2901](https://github.com/sgl-project/sglang-omni/pull/2901) |
| [MOSS-TD](#moss-td) | Transcription and diarization serving runtime | [#3071](https://github.com/sgl-project/sglang-omni/pull/3071) |
| [Ming3-TTS](#ming3-tts) | Non-streaming synthesis and native-streaming follow-ups | [#3136](https://github.com/sgl-project/sglang-omni/pull/3136), [#3189](https://github.com/sgl-project/sglang-omni/pull/3189), [#3190](https://github.com/sgl-project/sglang-omni/pull/3190) |

The PR links identify results for each model family; the repository lists also include subsequent research variants.

## CosyVoice3

Proposal [#154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154) defines the original TTS runtime task. Its [public task package](sglang-omni-tts-runtime/) optimizes CosyVoice3 with Qwen3-TTS Base and MOSS-TTS regression checks. Proposal [#188](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/188) is a separate PackedDiT compilation task.

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-tts-cosyvoice3-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-d154) | Original three-model task; public package source. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-d154/tree/main/sglang-omni-tts-runtime) |
| [`sglang-omni-tts-cosyvoice3-final-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-final-d154) | Three-model task with non-streaming and streaming evaluation at concurrency 16. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-final-d154/tree/main/sglang-omni-tts-runtime) |
| [`sglang-omni-tts-cosyvoice3-search-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-search-d154) | CosyVoice3-focused Work experiments with the full three-model formal verifier. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-search-d154/tree/main/sglang-omni-tts-runtime) |
| [`sglang-omni-tts-cosyvoice3-dual-mode-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-dual-mode-d154) | Single-model task requiring both streaming and non-streaming throughput to improve. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-dual-mode-d154/tree/main/sglang-omni-cosyvoice3-dual-mode-runtime) |
| [`sglang-omni-tts-cosyvoice3-dual-mode-guided-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-dual-mode-guided-d154) | Dual-mode task with profiling and AR, DiT and vocoder optimization guidance. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-dual-mode-guided-d154/tree/main/sglang-omni-cosyvoice3-dual-mode-runtime) |
| [`sglang-omni-0.1.9-fa90213e1674-tts-cosyvoice3-dual-mode-guided-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-0.1.9-fa90213e1674-tts-cosyvoice3-dual-mode-guided-d154) | Guided dual-mode task on SGLang-Omni 0.1.9 at fa90213e1674. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-0.1.9-fa90213e1674-tts-cosyvoice3-dual-mode-guided-d154/tree/main/sglang-omni-cosyvoice3-dual-mode-runtime) |
| [`sglang-omni-tts-cosyvoice3-d188`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-d188) | PackedDiT representation and compilation optimization under fixed regression tests. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-cosyvoice3-d188/tree/main/cosyvoice3-packed-dit-compile) |

## MOSS-TTS

MOSS-TTS-v1.5 single-model derivatives of the TTS task from proposal [#154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154). The dual-mode variants require both modes to improve; their reward weights differ.

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-tts-moss-v1.5-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-d154) | Single-model non-streaming throughput objective with streaming regression checks. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-d154/tree/main/sglang-omni-moss-v1.5-runtime) |
| [`sglang-omni-tts-moss-v1.5-dual-mode-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-dual-mode-d154) | Dual-mode objective; streaming:non-streaming reward weights 2:1. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-dual-mode-d154/tree/main/sglang-omni-moss-v1.5-dual-mode-runtime) |
| [`sglang-omni-tts-moss-v1.5-dual-mode-d154-s1-n1`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-dual-mode-d154-s1-n1) | Dual-mode objective; streaming:non-streaming reward weights 1:1. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-dual-mode-d154-s1-n1/tree/main/sglang-omni-moss-v1.5-dual-mode-runtime) |
| [`sglang-omni-tts-moss-v1.5-dual-mode-d154-s1-n2`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-dual-mode-d154-s1-n2) | Dual-mode objective; streaming:non-streaming reward weights 1:2. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-moss-v1.5-dual-mode-d154-s1-n2/tree/main/sglang-omni-moss-v1.5-dual-mode-runtime) |

## Fish S2-Pro

Fish S2-Pro single-model derivatives of the TTS task from proposal [#154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154).

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-tts-fish-s2-pro-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-fish-s2-pro-d154) | Non-streaming throughput objective with streaming regression checks. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-fish-s2-pro-d154/tree/main/sglang-omni-fish-s2-pro-runtime) |
| [`sglang-omni-tts-fish-s2-pro-dual-mode-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-fish-s2-pro-dual-mode-d154) | Both streaming and non-streaming throughput must improve. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-fish-s2-pro-dual-mode-d154/tree/main/sglang-omni-fish-s2-pro-dual-mode-runtime) |

## Qwen3-TTS

Qwen3-TTS Base tasks cover the 0.6B and 1.7B models separately, with lineage to proposal [#154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154).

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-tts-qwen3-tts-base-0.6b-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-0.6b-d154) | Base 0.6B; non-streaming throughput objective with streaming regression checks. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-0.6b-d154/tree/main/sglang-omni-qwen3-tts-base-0.6b-runtime) |
| [`sglang-omni-tts-qwen3-tts-base-1.7b-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-1.7b-d154) | Base 1.7B; non-streaming throughput objective with streaming regression checks. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-1.7b-d154/tree/main/sglang-omni-qwen3-tts-base-1.7b-runtime) |
| [`sglang-omni-tts-qwen3-tts-base-0.6b-dual-mode-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-0.6b-dual-mode-d154) | Base 0.6B; both streaming and non-streaming throughput must improve. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-0.6b-dual-mode-d154/tree/main/sglang-omni-qwen3-tts-base-0.6b-dual-mode-runtime) |
| [`sglang-omni-tts-qwen3-tts-base-1.7b-dual-mode-d154`](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-1.7b-dual-mode-d154) | Base 1.7B; both streaming and non-streaming throughput must improve. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-tts-qwen3-tts-base-1.7b-dual-mode-d154/tree/main/sglang-omni-qwen3-tts-base-1.7b-dual-mode-runtime) |

## Qwen3-ASR

Proposal [#257](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/257) defines fixed-weight Qwen3-ASR-1.7B runtime optimization for non-streaming and post-upload SSE text streaming. The reward variants retain the task layout and have separate evaluator identities.

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-asr-qwen3-asr-1.7b-d257`](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-d257) | Both modes must improve; streaming:non-streaming reward weights 2:1. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-d257/tree/main/qwen3-asr-streaming-throughput) |
| [`sglang-omni-asr-qwen3-asr-1.7b-d257-s1-n1`](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-d257-s1-n1) | Both modes must improve; streaming:non-streaming reward weights 1:1. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-d257-s1-n1/tree/main/qwen3-asr-streaming-throughput) |
| [`sglang-omni-asr-qwen3-asr-1.7b-d257-s1-n2`](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-d257-s1-n2) | Both modes must improve; streaming:non-streaming reward weights 1:2. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-d257-s1-n2/tree/main/qwen3-asr-streaming-throughput) |
| [`sglang-omni-asr-qwen3-asr-1.7b-nonstream-d257`](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-nonstream-d257) | Non-streaming objective ablation retaining the original streaming and quality guards. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-asr-qwen3-asr-1.7b-nonstream-d257/tree/main/qwen3-asr-streaming-throughput) |

## Whisper

Whisper large-v3 model derivative of proposal [#257](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/257), covering non-streaming and post-upload SSE text-streaming throughput.

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-asr-whisper-large-v3-d257`](https://github.com/OpenRSI-Foundation/sglang-omni-asr-whisper-large-v3-d257) | Fixed-weight Whisper ASR runtime optimization with streaming priority. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-asr-whisper-large-v3-d257/tree/main/whisper-asr-streaming-throughput) |

## MOSS-TD

MOSS-Transcribe-Diarize model derivative of proposal [#257](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/257), covering non-streaming and post-upload SSE text-streaming throughput.

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-asr-moss-transcribe-diarize-d257`](https://github.com/OpenRSI-Foundation/sglang-omni-asr-moss-transcribe-diarize-d257) | Fixed-weight MOSS-Transcribe-Diarize runtime optimization with streaming priority. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-asr-moss-transcribe-diarize-d257/tree/main/moss-transcribe-diarize-streaming-throughput) |

## Ming3-TTS

Separate tasks cover Ming-3.0-TTS 0.5B and Tiny. The original tasks optimize non-streaming synthesis on SGLang-Omni 0.1.9 at ca9e8789fcd8. The native-streaming follow-ups use a later baseline and preserve first-audio latency and continuity guards. Upstream [#3136](https://github.com/sgl-project/sglang-omni/pull/3136) optimizes non-streaming 0.5B synthesis; [#3189](https://github.com/sgl-project/sglang-omni/pull/3189) and [#3190](https://github.com/sgl-project/sglang-omni/pull/3190) add streaming audio for Tiny and 0.5B.

| Repository | Scope | Task package |
| --- | --- | --- |
| [`sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-0.5b-nonstream`](https://github.com/OpenRSI-Foundation/sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-0.5b-nonstream) | 0.5B non-streaming throughput optimization. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-0.5b-nonstream/tree/main/sglang-omni-ming3-0.5b-nonstream) |
| [`sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-tiny-nonstream`](https://github.com/OpenRSI-Foundation/sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-tiny-nonstream) | Tiny non-streaming throughput optimization. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-v0.1.9-ca9e8789fcd8-tts-ming3-tiny-nonstream/tree/main/sglang-omni-ming3-tiny-nonstream) |
| [`sglang-omni-ming3-0.5b-streaming`](https://github.com/OpenRSI-Foundation/sglang-omni-ming3-0.5b-streaming) | 0.5B native-streaming throughput optimization on the refreshed baseline. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-ming3-0.5b-streaming/tree/main/sglang-omni-ming3-0.5b-streaming) |
| [`sglang-omni-ming3-tiny-streaming`](https://github.com/OpenRSI-Foundation/sglang-omni-ming3-tiny-streaming) | Tiny native-streaming throughput optimization on the refreshed baseline. | [Task files](https://github.com/OpenRSI-Foundation/sglang-omni-ming3-tiny-streaming/tree/main/sglang-omni-ming3-tiny-streaming) |

## Run a task

Open the task repository and follow its README and bundled validator and runner skills. Task variants carry their own model assets, runtime pins, calibration and scoring contract.

For a task available directly in this repository, start with the [public CosyVoice3 package](sglang-omni-tts-runtime/). See [RSI-Harness](../RSI-Harness/) for the execution runtime.

Repository inventory checked on November 5, 2026.
