# OpenTinker rollout scheduling

## Overview

Research causal admission, ready-turn ordering, and two-replica routing for synchronous multi-turn ALFWorld inference. The only executable candidate is `/workspace/candidate/policy.py`; optional notes are inert. Both phases use the same Dockerfile-built Base with split, read-only Judge WORKDIR snapshots.

Version 1.2.0 uses a quality-based execution contract. Read the repository's
`validation/20261002-batch-invariant/report.json` for its acceptance status.
The user explicitly
approved allowing different trajectories while retaining task success. Earlier
exact-equivalence failures remain recorded in `validation/20261002-blackwell-auto/`
and `validation/20261002-batch-invariant/`; they are not scores for this revision.

## Baseline and evidence

[OpenTinker at f87fe25fc483e7f1d7596182fc55115cbc992724](https://github.com/open-tinker/OpenTinker/tree/f87fe25fc483e7f1d7596182fc55115cbc992724) and its [verl gitlink 4bf4bd32d049be648867ade8c72ee3f5c27ebfcf](https://github.com/volcengine/verl/tree/4bf4bd32d049be648867ade8c72ee3f5c27ebfcf) define the semantic source. The source-derived baseline uses eight static chunks, eager coroutines, independent per-worker cumulative assignment counters and sticky routing; fixed replica order replaces nondeterministic hash/handle ties. No matched result is reported or reproduced. Existing serial evaluation and training examples do not measure this protocol.

The common [adapter](environment/support/runtime.py) uses the real `AgentLoopManager` without training/reward-model workers, `GenericAgentLoop` initial tokenization, `GymEnvironmentInteraction`, ALFWorld game methods, and `DataProto`. It repairs pre-call bounds and packing rather than truncating accepted output. Environment shards are serialized CPU Ray processes calling the same pinned game reset/step code directly; this avoids the HTTP launcher's process-global selection and parser races while preserving fixed episode/shard assignment. Each reset constrains the upstream selector to the exact selected game. No preliminary environment rollout occurs.

[ALFWorld 1558ba46d078279ecb4c5d33a6cdffc96714a2d2](https://github.com/alfworld/alfworld/tree/1558ba46d078279ecb4c5d33a6cdffc96714a2d2) supplies the game implementation and domain/grammar. Construction inspected the checksum-pinned 0.4.2 PDDL archive and deduplicated actual game bytes. Formal games and orders are in [formal.json](tests/formal.json); the public 24-game development selection uses train only and excludes formal hashes. Model and tokenizer are [Qwen2.5-3B-Instruct at aa8e72537993ba99e69dfaafa59ed015b17504d1](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/tree/aa8e72537993ba99e69dfaafa59ed015b17504d1). Preserve the model's Qwen Research license and source/data licenses; model LICENSE and source licenses are retained in the image. Immutable identities and build preparation are task-owned; weights are downloaded and verified during build, not committed in this package.

## Revised fixed backend

Version 1.2.0 pins vLLM 0.12.0 with torch 2.9.0, Transformers 4.57.3,
Python 3.10, and the digest-pinned CUDA 12.9.1 base. Every inference actor
enables `VLLM_BATCH_INVARIANT=1` and `FLASH_ATTN`. The pinned verl source
receives only documented API compatibility patches in `environment/patches.json`.
The model, BF16 precision, greedy decoding, two replicas, prefix caching,
chunked prefill and workload are unchanged. The quality gate permits at most
a 5-percentage-point aggregate success-rate drop over all four rows. Scores from this revision describe this revised backend.
The previous FAIL evidence remains valid for version 1.0.0; version 1.2.0
must pass fresh live validation before research.

## Environment and evaluation

| Setting | Contract |
| --- | --- |
| Platform | Linux amd64, CUDA 12.9.1 devel Ubuntu 22.04, Python 3.10 in `/opt/venv` |
| Compute | 32 CPUs, 131072 MiB RAM, 16 GiB shared memory; exactly two compatible GPUs on one node, each at least 48 GB |
| Reference GPU pool | Two H100 80 GB devices; task does not require H100 specifically |
| Storage | 262144 MiB planning allowance; Harness does not enforce a storage quota |
| Work/Judge | Two GPUs per phase, root Work, release-all with a two-device pool; stop Work GPU processes before submission |
| Network | Work and Judge `no-network`; all libraries, public assets and development tools baked before runtime |
| Workspace | `/workspace/candidate/policy.py`, fixed source manifest, optional `/workspace/notes.md`; closed files only |
| Solution | `solution/solve.sh` idempotently copies the reference files; no training, evaluation, submission or reward |
| Timeouts | Build 14400 s; full verifier 10800 s; whole Agent run 172800 s |
| Score | 100 × geometric mean of four live paired baseline/candidate timing ratios, maximize |

The paired protocol, bounds, feedback and policy ABI are specified in [instruction.md](instruction.md). [launch.py](tests/launch.py) validates fixed assets and scope, runs all eight sequential passes, checks complete valid traces and aggregate task success, deletes private scratch, and writes exactly one finite positive reward. No failure scalar exists. Warmups, startup and cleanup count against wall time even when excluded from the rollout metric. Ten planned loops at 30 minutes Work + 180 minutes Judge allowance + 15 minutes mandatory overhead consume 37.5 hours, leaving 10.5 hours for research and margin; these are estimates, not measurements. Submission count is unlimited unless the operator explicitly sets a cap.

Formal logs contain protected trajectory contents and are captured in disposable mode-0700 Judge scratch. Only aggregate feedback and safe repair envelopes reach Harness logs. Candidate repair details preserve phase, exception type, candidate-only source location when available, and specific hints; compiler errors also preserve line/column/reason. Callback exception messages can embed formal identifiers and are replaced by exception-specific hints. Execution and result-serialization errors have distinct codes. The same bounded details survive public development exceptions. Successful feedback includes model and environment reset/step request latency aggregates and completed trajectories per second, with units and definitions in the instruction. These diagnostics do not change the scoring interval or reward.

Candidate callbacks use minimal-env pipes, preloaded helpers, UID/GID reduction, a default-deny syscall filter, memory/message bounds, and five-second deadlines. Judge source/identity authority is injected under `/tests`; required shared assets are checked against it. These controls do **not** isolate Judge from arbitrary root modification of the shared interpreter, system libraries or runtime. Candidate-authored logs/manifests are never timing or score authority. No independent image, attestation, or root-resistant provenance is claimed.

## Run and validate

Copy this directory into an RSI-Harness workspace. Set `RSI_MODEL`, `RSI_REASONING_EFFORT`, and `RSI_GPU_POOL` to the operator's model, effort and two compatible devices. Supply the provider credential and an explicit Agent API base/proxy through Harness's provider configuration; offline Agent operation requires that route. No evaluation-service credential is used. Honor the model license.

```bash
sudo -E rsi-harness run /absolute/path/to/opentinker-rollout-scheduling --agent codex --model "$RSI_MODEL" --reasoning-effort "$RSI_REASONING_EFFORT" --gpus "$RSI_GPU_POOL" --primary-reward reward --score-direction maximize --verbose
```

Generation validation uses the pinned local Harness checkout and does not execute the package:

```bash
python3 /opt/actions-runner/_work/RSI-Skills/RSI-Skills/agent-workspace/.agents/skills/harbor-task-agent/scripts/validate_task.py /opt/actions-runner/_work/RSI-Skills/RSI-Skills/agent-workspace/output/opentinker-rollout-scheduling --harness-root /opt/actions-runner/_work/RSI-Skills/RSI-Skills/agent-workspace/public-evidence/RSI-Harness
```

Static/compiler outcomes, the Generator checks, and correction round 1 are recorded in [generation-checks.md](generation-checks.md). Both supplied initial-review findings were corrected; this invocation starts no review. Build/image metadata, imports, fresh-container scope, offline address bootstrap, callback restrictions, real two-GPU quality-based baseline scoring and iteration-budget evidence belong in the external validation report. Research run evidence is published separately. Tokens, actions and probabilities may differ under the user-approved quality contract; replay and incomplete workloads remain forbidden. No execution result, measured runtime or speedup is claimed.

The vLLM 0.12.0 backend also includes the checksum-pinned SM120 dispatch
repair in `environment/backend_patches.json`: RTX Blackwell with torch 2.9
uses vLLM's existing invariant GEMM kernels, as the SM100 path already does.
Unpatched SM120 produced batch-dependent BF16 values in the hardware probe;
the patched kernels passed the same probe. Full live workload and quality qualification
is still required. The Judge verifies the patched backend file hash.
