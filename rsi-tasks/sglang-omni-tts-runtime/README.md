# SGLang Omni TTS runtime

Public task package for [OpenRSI proposal #154](https://github.com/OpenRSI-Foundation/OpenRSI-Index/discussions/154). Optimize the serving runtime for faster CosyVoice3 speech synthesis while retaining the fixed models and checking Qwen3-TTS Base and MOSS-TTS for regressions.

The [recorded research archive](../../rsi-logs/sglang-omni-tts-runtime/) contains the original research and its continuation. Three resulting upstream improvements were merged into SGLang-Omni: [idle speaker workers #2524](https://github.com/sgl-project/sglang-omni/pull/2524), [MOSS embedding accumulator #2525](https://github.com/sgl-project/sglang-omni/pull/2525), and [reference-feature transport #2555](https://github.com/sgl-project/sglang-omni/pull/2555). Those PRs contain their own matched serving measurements; their percentages do not sum to the task reward.

The archive has **11 complete throughput measurements across 14 submissions**, not just the one eligible reward. [Measured evidence](../../rsi-logs/sglang-omni-tts-runtime/measurements.json) retains original feedback, means, repetitions, failed guards and two separate local controlled studies. A complete evaluation can show faster CosyVoice3 serving without earning a reward if another original quality/latency guard fails. Incomplete evaluations have no aggregate measurement. Historical four-model and continuation results are not retroactively scored as this protocol-v5 task; publication leaves the evaluator, guards and original rewards unchanged.

This directory publishes one three-model task package from source commit `c819dc10aaf7dab0c2d0ed5eea5499e449a6a9d5`, retaining its original `sglang-omni-tts-runtime` name. Runtime code, the evaluator, fixed assets, the pristine source archives and all metric gates are retained. Publication adds repository canary comments, required metadata and the canonical instruction suffix. The 24-hour research budget is retained. Recorded runs used H200 GPUs with Slurm/Apptainer; the Docker build declared here has not been rerun for this publication. Logs retain their original instructions and do not imply a fresh run of this published package.

Task 2.0.0 / protocol v5 evaluates only **CosyVoice3, Qwen Base and MOSS**. Qwen CustomVoice is excluded from model execution, asset requirements, calibration, all guards and reward. The retained models keep their original commands, seeds and thresholds. This version is for a new run with a fresh matching calibration; do not hot-swap it into an active run or relabel a four-model baseline. Existing runs retain their pinned task and allocations.

## Overview

Optimize default-on CosyVoice3 requests/s through runtime research. The candidate is closed source under `/workspace/sglang-omni`, with hypotheses, profiles, ablations and complete experiment history under `/workspace/research`. Qwen3-TTS Base and MOSS-TTS-v1.5 are mandatory nonregression guards. Changed candidates provide `hypothesis.md`, `experiments.json`, `reproduction.json` and `source.diff` in the research directory; additional profiles and tests may be kept there. These records are self-reported, never scoring authority.

## Baseline and evidence

The source and baseline are [SGLang Omni at 1523543056bf334bc2f4d69db23aad0a002cedf2](https://github.com/sgl-project/sglang-omni/tree/1523543056bf334bc2f4d69db23aad0a002cedf2). The original `benchmarks/eval/benchmark_tts_seedtts.py`, `benchmarks/tasks/{tts,asr}.py`, `benchmarks/metrics/{performance,wer}.py`, `tests/utils/ci_cpu_contention.py`, model configs, `pyproject.toml` and `docker/Dockerfile` determine implementation. The full original archive preserves its license/notices; its digest and pristine inventory are task-owned. [Asset identities](tests/assets.json), [source dependencies](tests/source-deps.json), [package metadata](environment/python-artifacts.json) and [fixed command vectors](tests/protocol.json) contain the exact pins.

No fresh matching protocol-v5 baseline was measured during authoring or publication. The archive contains historical calibrated feedback, including a 9.232 requests/s reference for the later segments, but that is not a new calibration of this package. Other historical 12.63 requests/s, 0.0183 WER and 1.222 s latency observations do not match this full protocol and are never numeric gates. `solution/solve.sh` restores only candidate runtime source from `/opt/rsi-source`, retains research history, and performs no evaluation or GPU work. Harness does not execute Solution.

Model cards declare Apache-2.0. The dataset card declares no license field; no license is inferred, and dataset bytes are downloaded at build time, not redistributed in this repository. Upstream source and dependency licenses remain in the archive/image. `environment/requirements.lock` is resolved from official PyPI metadata without target-package execution. The explicit Qwen-TTS no-dependencies install and upstream Torch/Transformers/protobuf/setuptools overrides are recorded in the metadata manifest. The pinned base is a build ingredient, not a prebuilt task. UCX build parallelism is capped at eight. A realized distribution manifest is emitted at image build before calibration.

The cuda-tile PyPI sdist is a wheel downloader stub; the lock instead selects the official NVIDIA CPython 3.12 Linux x86-64 wheel by verified SHA-256. Other source-only packages use the pinned build tools with build isolation disabled, without floating build-time resolution. Archive extraction normalizes group/other write bits through Python's data filter; the pristine manifest describes these final modes. All installation runs outside WORKDIR, which is created afresh only at final closure.

## Environment and evaluation

| Setting | Contract |
| --- | --- |
| WORKDIR | `/workspace`; snapshot reload is read-only |
| Public source/dependencies/assets | `/opt/rsi-source`, `/opt/rsi-deps`, `/opt/rsi-assets` |
| Work / Judge GPUs | 1 / 3, one native GPU per model lane |
| Shared resources | 48 CPUs, 384 GiB RAM, 32 GiB shared memory, 1 TiB planning storage |
| Phase budgets | Work intended 16 cores/128 GiB; Judge three disjoint 16-core/128-GiB lanes |
| Timeouts | Build 14400 s; Judge 7200 s; whole Agent trajectory 86400 s |
| Runtime network | Environment, Agent and Judge `no-network`; localhost serving allowed |
| Judge entry | `/bin/bash /tests/test.sh`, isolated baked Python launcher |

Use an operator pool of **exactly three** eligible owned GPUs, with H100 80 GB as an unverified planning reference. Hardware model is not mandated. Harness release-all mode preserves the three-GPU peak: Work must stop its GPU processes before submitting. CPU/RAM Docker limits are shared, not phase reservations. The operator reserves CPU/NUMA capacity and excludes other workloads; launchers apply affinity, verify physical cores and GPU UUID mappings, and monitor unique/RSS memory, GPU memory and foreign CPU load. Any sampler error, zero completed windows or foreign-load peak above two cores is infrastructure/unscored. Storage is not an enforced Harness quota.

Judge makes disposable candidate copies and supplies the original protected benchmark. Three lanes run concurrently, each with three serial full repetitions, fresh processes and empty lane/repetition compilation caches. Each repetition runs the unchanged stock TTS-plus-ASR command then original-source authoritative Qwen3-ASR on the actual WAVs. Original ASR has a separate import root without candidate source. All 1,088 English examples remain in native order. TTS uses concurrency/warmup 16; authoritative ASR uses concurrency 32 and warmup 64. All samples must complete with valid WAVs and transcripts. Temporary results and logs remain in private scratch and are removed before final reward.

The trusted client importer keeps the stock benchmark's Omni loading/configuration dependencies original while allowing its native serving subprocesses to load the candidate. Argument vectors and stock timing remain unchanged. Authoritative ASR uses only the original tree. Offline dataset caches are copied from identical baked contents into disposable process-group scratch, and compilation caches start empty. Each lane monitors a 128-GiB scratch bound and 256-MiB per-log bound. Descendant processes are tracked across session boundaries and terminated on failure; a leak makes the evaluation unscored. A failed lane cancels the other lanes before further repetitions.

Protocol version 5 retains the original benchmark's per-request sampler `--seed` to **42, 43, 44** for repetitions 1, 2, 3 of every model. Calibration and candidates use the same schedule, including warmups, and retain the actual seed with each run. The evaluator checks the emitted benchmark config and rejects missing/mismatched seeds; `--arrival-seed` remains unset for the closed-loop workload. The protocol hash prevents reuse of an unseeded calibration. Do not choose seeds after observing scores. Request seeds reduce sampling variation; they do not guarantee bitwise-identical waveforms, remove timing noise, or make three repetitions independent statistical replicates.

The benchmark also defaults `max_running_requests=64` and `cuda_graph_max_bs=64` and forwards these values as stage-specific server CLI overrides for all three models, even when omitted from the command. These override lower-level model configuration; neither is the client concurrency of 16. Before a batch-related experiment, record the benchmark defaults/explicit flags, generated server command, resolved engine values and observed batch sizes. See [the agent instructions](instruction.md) for the required configuration audit. Formal evaluation retains the native 64/64 defaults and verifies them in benchmark output.

Take arithmetic means of the three **per-run** metrics, including the percentiles. CosyVoice3 requests/s must strictly improve, with no tolerance. Other performance guards allow 3% relative regression of their means; Each model must have mean WER at most 0.02, an absolute 2% ceiling. The precise conditions appear in the protocol-v5 table below. Native rounding is retained without additional rounding, filtering or clipping. Corpus micro-WER includes examples above 50% WER. Reward is candidate mean CosyVoice3 requests/s divided by initial baseline mean, issued only after all gates against the current accepted baseline pass; equality is ineligible.

No candidate-failure scalar exists. Gate failure is valid-but-ineligible; attributable compile/illegal-memory/correctness failures are candidate-invalid; timeout, incomplete evaluation, ambiguous crashes and setup faults are separately diagnosed. All are unscored with **no reward file**, producing a Harness verifier error. Only complete eligible evaluation writes one final `/logs/verifier/reward.json`. Normal submissions never remeasure the baseline.

Full safe child diagnostics and aggregates reach the Agent, including durable logs. Streams containing ASR hypotheses are captured privately and selectively redacted; ordinary paths, errors and tracebacks remain visible. Per-example ASR hypotheses, rankings/answer tables, protected-check source and credentials are withheld. Public text/reference audio and upstream code are public. Source scope, syntax, pinned hashes, exact coverage and independent audio/ASR checks are finite defenses. Semantic restrictions on weights, architecture, precision, sampling, voice, conditioning, abort cleanup, request isolation, special-casing and answer caches still require source review. They are not a proof against arbitrary hostile code. Root Work and inherited interpreter/libraries remain shared; the operator-bound calibration fingerprints detect changes but are not a separate trusted Base or attestation service. WER does not establish speaker identity, prosody, fidelity or generalization. Public-workload adaptation and multiple noisy guards limit scientific inference.

## Run and validate

Execution is intentionally pending for `harbor-task-validator`. It must build the image, verify dependency imports and offline aliases, test pristine scope acceptance, check CPU/GPU/resource mapping, perform complete calibration, then test no-op/failure/redaction/cleanup paths and measure variance/cadence before a real Agent. No Environment, baseline, evaluator, Solution or real Agent was executed during generation.

Before research, set `RSI_TTS_CPU_LANES` to JSON containing three arrays of 16 reserved physical CPU IDs, and `RSI_TTS_NUMA_LANES` to JSON containing the corresponding three node-ID arrays. Reserve these resources on one node. Set `RSI_GPU_POOL` to the three authorized physical GPU IDs and `RSI_TTS_GPU_UUIDS` to their comma-separated UUIDs in the same order. These are real operator allocation inputs, not arbitrary defaults. Native `/proc`, bash, nvidia-smi and all mapped devices must be available.

The validator executes `/opt/sglang/bin/python3 -I /tests/calibrate.py` in a fresh prepared task container with the **same three GPUs, image, dependencies and CPU/NUMA allocation** used for candidates. Inject this task's entire `tests` tree before startup. The entrypoint verifies pristine source, executes all nine runs using the normal evaluation engine, and emits one `RSI_TTS_CALIBRATION=` line through stdout. No reward is written. Retain that complete aggregate record and all earlier calibration attempts. The single full baseline record uses [calibration.schema.json](tests/calibration.schema.json); `noise.py` validates it and freezes the schema-3 frozen baseline described below.

For a direct validator setup using a prepared image tagged `rsi-tts-validation`, these native Docker commands inject the task-owned checks without authored Compose volumes or a separate evaluator image:

```bash
RSI_CALIBRATION_CONTAINER=$(docker create --network none --gpus "\"device=$RSI_GPU_POOL\"" --cpus 48 --memory 384g --shm-size 32g --env RSI_HARNESS_EXPECTED_GPU_UUIDS="$RSI_TTS_GPU_UUIDS" --env RSI_TTS_CPU_LANES --env RSI_TTS_NUMA_LANES rsi-tts-validation /opt/sglang/bin/python3 -I /tests/calibrate.py)
docker cp output/sglang-omni-tts-runtime/tests "$RSI_CALIBRATION_CONTAINER":/tests
docker start -a "$RSI_CALIBRATION_CONTAINER"
docker rm "$RSI_CALIBRATION_CONTAINER"
```

This is a later validator recipe, not a claimed calibration. Use the validator's prepared image if runtime provisioning affects the fixed interpreter/dependency fingerprint. Retain the one complete calibration record, invoke `/opt/sglang/bin/python3 -I /tests/pool_calibrations.py <record-1>`, and export the JSON after `RSI_TTS_NOISE_CALIBRATION=` as `RSI_TTS_BASELINE_JSON`. Keep the raw record out of Work inputs; aggregate comparisons are disclosed in Judge feedback. The Judge-only `[verifier.env]` templates inject it. Missing, malformed, stale or mismatched input is infrastructure/no-reward. Recalibrate after source/protocol, software, hardware, clock/power policy or allocation changes. Clock/power settings must remain fixed by the operator; application/max clock policies and power limit are recorded, and throttling remains an execution-validation concern.

Supply `RSI_MODEL`, `RSI_REASONING_EFFORT` and the Harness provider's explicit API-base/proxy route and credentials using Harness runtime configuration. Provider access does not grant research egress; no evaluation credential or external evaluation service is required.

The device probes compare the complete canonical GPU UUID. PyTorch 2.13 reports
the UUID without NVML's `GPU-` display prefix; the checks normalize that prefix
and retain exact full-UUID equality, rejecting malformed or different IDs.

```bash
sudo -E rsi-harness run /opt/actions-runner/_work/RSI-Skills/RSI-Skills/agent-workspace/output/sglang-omni-tts-runtime --agent codex --model "$RSI_MODEL" --reasoning-effort "$RSI_REASONING_EFFORT" --gpus "$RSI_GPU_POOL" --primary-reward reward --score-direction maximize --max-submissions 64 --verbose
```

The run command uses this invocation's destination; after copying the task, replace only that task path. The 64-submission cap is administrative, separate from the ten-loop plan. Unverified estimates are 30 minutes Work plus 60 minutes full Judge per adaptive loop: 2 hours setup/calibration + 10 × 90-minute loops + 7 hours margin = 24 hours. The 120-minute Judge timeout is not a runtime estimate. An operator may extend the whole Agent budget to 172800 s using Harness's `--timeout 172800`; this deliberately overrides the default. Three parallel model lanes are parts of one evaluation, not three research loops.

Static/compiler validation command at this destination:

```bash
python3 /opt/actions-runner/_work/RSI-Skills/RSI-Skills/agent-workspace/.agents/skills/harbor-task-agent/scripts/validate_task.py /opt/actions-runner/_work/RSI-Skills/RSI-Skills/agent-workspace/output/sglang-omni-tts-runtime --harness-root /opt/actions-runner/_work/RSI-Skills/RSI-Skills/agent-workspace/public-evidence/RSI-Harness
```

The generation result records the actual static/compiler outcomes. Stage 9 independent review is outside this invocation. Publication uses ordinary Git; each file is at most 100 MiB and no weights/data payload or Git LFS is included. Task-owned checks are below the Harness 1-GiB/100,000-entry injection limits.

## Protocol v5: rolling accepted baseline, three-run means

The initial baseline B0 is the pinned unmodified SGLang Omni source at `1523543056bf334bc2f4d69db23aad0a002cedf2`. Measure it once using three models, three serial full runs per model, seeds 42, 43, 44. Freeze those three-run arithmetic means. Each candidate also runs three times per model. Do not select a best run, discard an unfavorable repetition or pool request-level percentiles across runs.

After each complete eligible candidate, its evaluated code, fixed configuration and three-run results become the next baseline Bk+1. A candidate is compared against the **last accepted baseline Bk**, never automatically against the latest main commit. Failed, incomplete, timed-out and ineligible rounds do not advance it. All baseline generations and rejected attempts remain auditable. The host operator owns the state and binds one immutable baseline to each submission through the Harness evaluator port; Work cannot update that state. A candidate's own research records never authorize promotion.

Let B and C be the current accepted baseline and candidate arithmetic means for the same model and metric:

| Role | Metrics | Eligibility condition |
| --- | --- | --- |
| Primary performance objective | CosyVoice3 requests/s | **C > B**, no tolerance; equality or slowdown is ineligible |
| Other performance guards | All latency and RTF mean/median/p95/p99; all audio-seconds/s; requests/s of Base and MOSS | Lower-is-better: **C <= 1.03 * B**; higher-is-better: **C >= 0.97 * B** |
| Quality guards | Corpus micro-WER, separately for all three models | **C <= 0.02**: an absolute WER ceiling of 2% |

The 3% allowance is relative to Bk and applies to other performance guards, not the primary objective. It is an engineering tolerance, not measured variance or a confidence interval. WER is an absolute score: 1.8% passes and 2.1% fails, regardless of baseline WER. A 10% speedup is a research target, not a required minimum, and cannot offset a failed guard. Every comparison uses the three-run mean, including the arithmetic mean of each run's p95/p99. Other guard regressions may accumulate across accepted versions, so also report every metric against B0 without adding an unapproved cumulative gate.

Report the current baseline generation and source digest, all three results, current and initial baseline means, candidate mean, signed/relative deltas, gate role, comparator, threshold, and pass/fail. WER thresholds and deltas are also displayed in percentage points. Report both step throughput ratio C/Bk and cumulative ratio C/B0. Harness reward uses cumulative CosyVoice3 C/B0 so scores stay comparable as Bk changes; eligibility always requires improvement over Bk plus all guards. Only eligible candidates receive a reward file.

`pool_calibrations.py <record-1>` freezes the one pristine baseline as a schema-3 record with status `frozen`, generation 0, exact evaluated source digest and initial means. After a successful native Judge report, the operator validates the full feedback, recomputes all gates, checks the evaluated source snapshot, archives code/configuration/results and atomically advances the current baseline. It preserves the initial record, every accepted generation and crash-recovery state. Old schema-2 noise-band pools and earlier protocol identities are rejected; historical measurements are not relabelled as current results.

One separate held-out pristine A/A evaluation validates the final Judge path before the Agent. It is not accepted into the research baseline chain and cannot change B0 or any threshold. A no-op may be valid-but-ineligible because QPS did not strictly improve or a guard failed. Retain its real result; never assign an artificial score of 1. Incomplete or invalid execution blocks research.

All 1,088 examples must complete in each run with valid native audio and transcripts, zero failures and zero skips. Benchmark command vectors, seeds, models, weights, precision, sampling, voice/reference conditioning, workload, authoritative ASR, resource/contention checks, cleanup and editable source scope remain fixed. The initial calibration and each candidate comprise 9 complete runs and 9,792 scored requests. Earlier trajectories retain their original protocol and scores.
