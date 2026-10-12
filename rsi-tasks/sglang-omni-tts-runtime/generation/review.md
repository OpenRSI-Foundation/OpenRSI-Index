# Stage 8 Generator review

Historical review of the original four-model task. It does not validate task 2.0.0 / protocol v5; see the current README and generation contract tests for the three-model revision.

This is the single Generator review and its closed worklist, not an independent review. Stages 1–6 were not repeated and Stage 9 was not performed. Destination reconfirmed as the newly created `output/sglang-omni-tts-runtime`; all controller inputs remained read-only. Correction count for independent-review rounds is zero.

The first combined static/Harness compiler invocation passed: zero errors, eight warnings, compiled WORKDIR `/workspace`, Work/Judge GPUs 1/4, Agent/Judge timeouts 86400/7200. The commands are rerun after this report and the last implementation edit; the controller-facing generation result records those final outcomes.

## Closed findings

| Finding | Closure and direct dependents inspected |
| --- | --- |
| A PyPI CUDA Tile source artifact was a downloader stub rather than the required binary | Inspected its build metadata and the official NVIDIA index, then hash-verified the CPython 3.12 x86-64 wheel. Updated both artifact manifests, both hash locks, Docker package discovery and the metadata authoring helper. No target package was imported or built. |
| Static parsing treated the separate `--find-links` value as an unpinned package | Used pip's equivalent `--find-links=URL` spelling so the validator recognizes the option. The exact version and wheel SHA-256 remain required by the unchanged hash lock. |
| Archive permission normalization would reject the pristine source | Python's `data` extraction filter removes group/other write bits. Both pristine manifests and the static checker now represent final normalized modes. Build extraction and Judge extraction use that same filter, with byte hashes unchanged. |
| A base image could leave incidental WORKDIR files | Final build closure recreates `/workspace` only after all install/asset initialization, then copies pristine source and creates research. Installation metadata stays outside WORKDIR. Build checks and Judge scope checks agree on every file/symlink. |
| Original benchmark source could still import candidate-controlled client dependencies | Added the task-owned client import finder for the unchanged benchmark command. Benchmark and Omni client imports resolve to original source; native serving children retain candidate imports. Authoritative ASR has an entirely original cwd/import root. Reviewed subprocess argv, environment and import ordering. |
| Native server sessions could outlive a failed benchmark | Track descendant PID/create-time identities across native sessions, terminate leaks, check cleanup and cancel peer lanes on failure. Parent allows lane cleanup headroom; required GPU cleanup and scratch removal both run even if either fails. No failure path writes reward. |
| A fixed-runtime probe traceback could print protected check source | Probe errors use structured diagnostics with paths/type/stack locations but no protected source lines. Ordinary native child traces remain visible. |
| Log filtering could unnecessarily hide public TTS text | Hypothesis substitution is now confined to ASR contexts, plus explicitly labeled hypothesis fields. Stock and authoritative server log files are included. No credential is forwarded to native children; result tables are never emitted. |
| WAV headers could inflate duration-based metrics | Independently decode all actual WAVs, require finite samples and duration consistency with stock four-decimal per-request records, hash-check the ASR copies, and compare native sample rate/channel/sample format with calibration. Native aggregate metric precision is unchanged. |
| Offline dataset clients could create shared cache locks | Each process group receives a disposable copy of identical baked dataset-cache contents. Other compilation/runtime caches start empty and are lane/repetition-local. Public model snapshot inventory and digests are checked before and after scoring. |
| Scratch/log growth and resource errors needed an explicit terminal path | Monitor per-lane scratch and log bounds along with affinity, memory, GPU memory and original foreign-load sampling. Monitoring failure cancels the complete evaluation and remains unscored. |
| Baseline prose lacked static-validator labels | Instruction now explicitly names Baseline, Reported result, Status and Comparison while preserving the scientific contract. |

## Warning dispositions

- `README_TIMEOUT_OVERRIDE`: accepted. The run command preserves 86400 seconds; the separately documented 172800-second operator extension is explicitly intentional and contributor-bound.
- `GIT_REF_MUTABLE`: accepted as a pattern limitation. The UCX clone is followed in the same RUN by `git -C ... checkout d8e50df6651b9ea5b76f23aee0aefbf053a4137a` and a matching `rev-parse HEAD` equality check. CosyVoice/Matcha clones likewise checkout and verify manifest pins in build preparation.
- Four `INSTRUCTION_BASELINE_INCOMPLETE` warnings: fixed by the explicit reference-baseline table.
- `VLLM_HOST_IP_RUNTIME_MISSING`: accepted. Dependency metadata contains distributed-framework references; this evaluator launches single-node native SGLang Omni processes, not Ray/vLLM data-parallel or distributed workers. No relevant routable-address bootstrap path is invoked.
- `VERIFIER_NETWORK_FETCH` for `tests/requirements.lock`: accepted. The lock is inert Judge authority (including the package named wget), hashed for identity; it is never installed or fetched by Judge. Only the Docker build consumes the Environment copy. All runtime serving is localhost and all assets are offline.

## Contract and terminal-path trace

Source, six model revisions, dataset revision/hash, extra source refs, original four argument vectors, row order, concurrency/warmup/repetition counts, corpus micro-WER, per-run precision and arithmetic means match the bound confirmation. Public assets are downloaded during build; source authority and all evaluator code/schema/manifests are packaged. No numeric baseline or measurement is invented. Model checkpoints/data are not task Git payloads; all files and tests injection remain within publication limits.

Only Work produces candidates. Solution restores the same unmodified source without training/evaluation/submission. Normal Judge evaluation consumes only the candidate and an operator-supplied, schema-validated measured calibration with matching source/assets/protocol/software/GPU UUID/model/driver/clock/power/CPU/NUMA identity. Calibration uses pristine source and the same twelve-run engine, emits a complete stdout record, and never writes reward.

Scope errors and proven candidate runtime/audio failures emit actionable candidate-invalid diagnostics and no reward. Every native repetition needs exact TTS/WAV/transcript coverage; authoritative ASR is recomputed from the actual audio with original code. Missing/malformed calibration, infrastructure/dependency/device/contention problems, timeout, ambiguous crashes and incomplete results are unscored. All guards are checked without extra tolerance or rounding. Eligible candidates alone reach the positive finite CosyVoice3 ratio; ineligible valid candidates emit aggregates without reward. A no-op receives its real ratio only if it independently passes all strict guards. The exclusive final reward write occurs after mandatory cleanup, outside `finally`.

Protected source, dependencies and asset checks use task-owned manifests plus the explicitly approved operator calibration identity. Standard root Work, inherited libraries and self-reported research records do not provide arbitrary-root isolation. Finite source/audio/quality guards do not prove all semantic restrictions, speaker fidelity or generalization. These limitations are documented, not replaced with an unsupported trust service.

No public self-check is needed for this source-only artifact. No training, live paired baseline, external inference, task volumes, separate Judge image, GPU type mandate, budget flag, Ray/vLLM distributed evaluator or lm-eval multi-filter evaluator was added. Build/import/offline, actual launcher behavior, calibration, GPU/resource cadence, failure/redaction/cleanup and real-Agent execution remain pending for the separately invoked execution validator.

Final read-only checks parse Python/JSON/TOML and shell syntax, inspect packaged archive identities/modes, check lock hashes/platform artifact families and publication size, then run the prescribed combined static validator and pinned supplied Harness compiler. They do not import or execute task/repository code.
