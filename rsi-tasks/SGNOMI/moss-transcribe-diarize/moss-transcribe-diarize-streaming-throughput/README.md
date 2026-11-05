# MOSS-Transcribe-Diarize streaming throughput

## Overview

This ML/Inference task studies fixed-weight MOSS-Transcribe-Diarize scheduling, batching, overlap and kernels on one GPU. Work submits source/configuration; Judge reloads the read-only `/workspace` snapshot and measures real local HTTP responses. Primary contributor: Zhuofeng Li, zhuofengli12345@gmail.com. Coauthors: Chenyang Zhao and db-oi.

## Baseline and evidence

The official [SGLang-Omni source](https://github.com/sgl-project/sglang-omni/tree/8220d220703cf8d8f5c5d9a2cde00b5ed13717b8) is pinned at `8220d220703cf8d8f5c5d9a2cde00b5ed13717b8` (upstream `main` on 2026-10-17). The model is [OpenMOSS-Team/MOSS-Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/tree/704aa4a9c304e8520be88901e0d1960158ef5b15), and the dataset is [SeedTTS Arrow](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-arrow/tree/27f4c1adee83b5b29b7c4b375f6b976324bda308). See the task-owned source/model/data manifests for byte identities. Source and model are Apache-2.0; retain upstream notices. The pinned dataset card does not declare a license field; this task retains its source attribution and the contributor-approved redistribution boundary without inventing a license.

B0 uses BF16, greedy decoding, compilation, upstream language/preprocessing/token-budget semantics and encoder batching. The fixed cold-input restrictions disable cache lookup/write, fingerprint-keyed merging and failure retries/recovery. They apply equally to B0 and candidates. No current-protocol B0 error or throughput has been measured. Historical results from other models or protocols do not initialize comparison state. See MODEL_ADAPTATION.md for the exact common baseline and calibration storage.

`solution/solve.sh` restores candidate source surfaces and configuration from public Environment assets; it performs no install, training or evaluation. Preparation measures full B0 and one full A/A before research. B0 normalization is 1; normal unchanged submissions remain subject to strict improvement.

## Environment and evaluation

| Item | Contract |
| --- | --- |
| Build | Linux amd64/Python 3.12; CUDA 13.0 Ubuntu 24.04; full Dockerfile |
| Work/Judge | Standard root; one shared service; one GPU per phase, same exclusive physical card, release-all mode |
| Resources | 16 reserved physical CPU cores, 131072 MiB RAM, 32 GiB shared memory |
| Storage | 204800 MiB planning estimate; Harness does not enforce a storage quota |
| Assets | `/opt/models/moss-transcribe-diarize`, `/opt/asr-reference`, `/opt/asr-venv` |
| Candidate | `/workspace/sglang-omni`, `/workspace/candidate.yaml`, `/workspace/research` |
| Network | All runtime policies no-network; Judge contacts only loopback; no Judge model access |
| Full Judge | 12 fresh-service passes; 18648 warmup + 18648 scored requests; EN/ZH × nonstream/SSE × 3 |
| Timeouts | Build 14400 s; Judge 10800 s including bounded 600 s readiness per pass; Agent default 86400 s |
| Reward | `R_stream(C;B0)^(2/3) * R_nonstream(C;B0)^(1/3)`, maximize, no clipping |
| Acceptance | Both mode ratios >1 against Bk, local 3% performance guards, corpus error ≤1.03×initial B0 error for each workload |

See `instruction.md` for the complete scientific contract and `tests/contract.py` for arithmetic. Judge scores only the candidate. It never optimizes or recreates a candidate, nor runs a live paired baseline. Failed gates, candidate runtime failures and incomplete evaluations receive no scalar. All failures leave Bk unchanged. Complete feedback includes aggregates and repair diagnostics; references/transcripts/per-request correctness and scoring internals stay hidden. Service streams that can contain those values are captured, filtered and labeled; the bounded last 2000 diagnostic lines are retained. Full Judge stdout is Agent-visible. This is a shared snapshot, not an independent clean-Base verifier: inherited binaries/libraries and public-data memorization are residual risks.

Native Parquet shards preserve all original values, audio bytes and order. Generation verified ordered Arrow-table equality against both original files with the bound SHA-256 digests. Every shard is below 100 MiB and the whole injected tests tree is below 1 GiB/100000 entries. There is no Git LFS or runtime download. Dependency locks preserve upstream pins. The wheel resolver does not execute source build hooks: source-only packages are separately pinned by hash and their metadata dependencies are included in the wheel lock. CUDA Tile uses NVIDIA's real CPython 3.12 wheel instead of its PyPI downloader stub. Build-time source packages use the locked installed build tooling without isolation. No donor compiler caches are copied. Compatibility, cold compilation and GPU fit remain execution checks.

`environment/initialize.py` closes the full pristine inventory after installations and cleanup, including directories and symlinks. The host captures that inventory before Agent execution and injects it as Judge authority. The same scope gate accepts the pristine state and every submission. Runtime compiler/library caches live in fresh disposable scratch outside `/workspace`; all services and scratch are removed before reward creation. Research logs are self-reported and never scoring authority.

Correction round 1 replaces the protobuf resolver override with intersecting constraints in `environment/constraints.txt`. The regenerated wheel lock selects gRPC 1.81.1 and protobuf 6.33.6 while preserving the confirmed upstream pins. Official [gRPC health metadata](https://pypi.org/pypi/grpcio-health-checking/1.81.1/json) and [reflection metadata](https://pypi.org/pypi/grpcio-reflection/1.81.1/json) require protobuf 6.33.5 or newer within major 6. The selected wheels' generated protobuf code, including the serving adapter's protocol package, requires 6.33.5. `tools/dependency-evidence.json` records official metadata and inspected wheel hashes; `tools/check_dependencies.py` checks active dependency constraints and generated-code requirements without installing or importing those distributions. Source-only package pins and hashes are unchanged. Metadata compatibility does not establish build/import/ABI compatibility.

KV byte capacities now use a positive 64-bit parser range. Child-log filtering redacts labeled answer regions, complete answers and complete partial SSE answers; individual deltas never become global substring filters. Short answer strings are matched only as whole diagnostic lines or inside answer-bearing regions, preserving unrelated paths and errors. Known state, hardware, asset and dependency failures have distinct safe codes, conditions and repair hints. Unknown exceptions still withhold protected scorer frames and data-derived messages. Arbitrarily encoded or deliberately disguised candidate output is not exhaustively proven safe; inherited-runtime tampering remains the documented shared-runtime limitation.

## Run and validate

Use the exact Harness source interfaces recorded in `tools/harness-manifest.json` and its installed Python environment. The launcher refuses mismatches. It uses production lifecycle callbacks before Judge injection and after completed reports, with a host-only locked task copy. It verifies fixed files and updates only declared comparison-state and clean-build inventory inputs. Bk advances atomically only when the final reward, completed Harness status, complete aggregate feedback, run/round/nonce, protocol, hardware and generation agree. Missing or stale state fails closed. Never run this task through the plain CLI: the host state adapter is required.

The operator reserves one compatible GPU and 16 distinct physical CPU cores, sets `RSI_GPU_POOL` to one selector and `RSI_CPU_SET` to 16 comma-separated CPU IDs, and supplies `RSI_MODEL`, `RSI_REASONING_EFFORT`, `RSI_STATE_DIR` (a new absolute host directory), and the Harness environment's Python as `HARNESS_PYTHON`. H100 80GB is a reference, not a type requirement. Recalibrate on different hardware. The operator must configure the Harness's explicit Agent API base/proxy and runtime credentials through its normal configuration; do not put secrets into the task or grant candidate service egress. Reserve these CPU cores from other jobs; the launcher pins both containers but cannot prevent unrelated host jobs from using them.

First, during separately authorized execution validation, prepare B0 and A/A using real Work, submissions and Judge, without executing a model Agent:

```bash
sudo -E "$HARNESS_PYTHON" /absolute/path/to/moss-transcribe-diarize-streaming-throughput/tools/launch.py --prepare --state-dir "$RSI_STATE_DIR" --gpus "$RSI_GPU_POOL" --cpu-set "$RSI_CPU_SET" --model "$RSI_MODEL" --reasoning-effort "$RSI_REASONING_EFFORT"
```

Preparation has its own eight-hour wall allowance for two full evaluations and overhead. It retains measured B0 and A/A in host state. It refuses research when A/A itself would count as an accepted optimization, violates quality, or shows more than 3% two-sided variation in available performance metrics. This is a conservative deployment/noise check, not a change to research reward gates. Resolve the environment if it fails; do not cherry-pick calibrations or relax gates. Infrastructure retries require a diagnosed cause and an entire unchanged evaluation, separately accounted in operator logs. The launcher does not automatically retry.

After execution validation admits the prepared environment, use the conservative 48-hour research profile:

```bash
sudo -E "$HARNESS_PYTHON" /absolute/path/to/moss-transcribe-diarize-streaming-throughput/tools/launch.py --state-dir "$RSI_STATE_DIR" --gpus "$RSI_GPU_POOL" --cpu-set "$RSI_CPU_SET" --model "$RSI_MODEL" --reasoning-effort "$RSI_REASONING_EFFORT" --primary-reward reward --score-direction maximize --max-submissions 64 --hours 48
```

Omit `--hours 48` for the default 24-hour profile. Ten conservative complete loops use 10 × (1 h Work + 3 h Judge + 0.25 h Harness overhead) = 42.5 h, leaving 5.5 h in the extended profile. The 64-submission ceiling is not a loop target. Calibration/A/A are separate. These allowances are unmeasured; cadence acceptance remains pending. A state directory owns one research trajectory and cannot be reused after it finishes. Keep private host logs/state for diagnostics and recovery; neither is mounted into Work as authority.

Static package and dependency checks, from this repository:

```bash
python3.12 -B moss-transcribe-diarize-streaming-throughput/tools/check_dependencies.py
python3.12 -B moss-transcribe-diarize-streaming-throughput/tools/check_package.py
python3.12 -B moss-transcribe-diarize-streaming-throughput/tools/test_adaptation.py
```

This is an authorized direct model derivation of the published Qwen D257 task. Its parent's generation attestation is retained as provenance and does not certify the modified task. Static checks and CPU-only adaptation regressions are recorded separately. Later Radix preparation reached GPU inference but failed the timestamp-only parser; two non-scored probes identified complete speaker-only output. The parser now accepts both complete forms, with regression evidence in the repository validation record. No successful full B0/A/A or optimization trajectory is certified for this variant. A fresh state directory and full strict B0/A/A are required under the revised immutable-manifest protocol. Build/import compatibility, GPU fit, real SSE output, measured quality and cadence remain execution-validation requirements.


## Public task copy

This package is published in OpenRSI-Index. [Publication provenance](publication.json) records the source repository, exact commit and hashes of every original task file. The environment, evaluator, fixed assets, source archives and scoring contract are copied unchanged. Historical setup statements describe the source run; a new run requires its own matching environment validation and calibration. No GPU evaluation was performed for this publication.
