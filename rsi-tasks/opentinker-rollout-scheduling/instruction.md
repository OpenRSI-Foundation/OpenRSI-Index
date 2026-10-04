# Causal rollout scheduling

Reduce synchronous multi-turn rollout completion time by improving trajectory admission, ready-turn ordering, and routing between two fixed inference replicas.

## Goal

- Test scheduling hypotheses that trade prefix locality against queue imbalance.
- Complete every episode and preserve task success while reducing total rollout time.
- Use aggregate feedback to explain which workload regimes benefit.

## Workspace

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Deliverable | Complete, closed `/workspace/candidate/policy.py`, at most 1 MiB |
| Optional notes | `/workspace/notes.md`, inert UTF-8 research notes, at most 4 MiB |
| Starting state | A working eager/sticky baseline policy and fixed `/workspace/candidate/source_manifest.json` |
| Public tools | `/opt/opentinker_task/dev.py`, train-only `/opt/opentinker_task/dev_manifest.json`, reference files in `/opt/opentinker_task/reference` |
| Fixed assets | Source under `/opt/src`, model/tokenizer under `/opt/models/qwen2.5-3b-instruct`, public game universe under `/opt/data/alfworld` |

## Reference baseline

The source-derived matched baseline divides every batch into eight static equal chunks and eagerly creates each worker's trajectory coroutines. Each worker assigns new trajectories to the least cumulative assignment counter, breaking ties by fixed replica order, then routes that trajectory stickily. Its counters persist across batches within a pass and reset between passes. It does not balance active requests.

| Item | Value |
| --- | --- |
| Baseline | Source-derived eager admission with per-worker cumulative counters and sticky routing |
| Reported result | Not reported for this fixed scheduling protocol |
| Status | Not reproduced during task authoring |
| Comparison | Live paired baseline and candidate passes in every row |

A baseline candidate receives the measured continuous paired score; it is conceptually near 100 but is never assigned 100 automatically. Public baseline copies are `/opt/opentinker_task/reference/policy.py` and `/opt/opentinker_task/reference/source_manifest.json`.

## What you may change

- `/workspace/candidate/policy.py`: admission, ready-turn ordering, causal workload estimation, active-trajectory limits, and per-turn replica routing.
- `/workspace/notes.md`: optional inert research documentation.

## Policy interface

Define a self-contained `Policy` class in `/workspace/candidate/policy.py`:

```python
class Policy:
    def reset(self, metadata):
        self.state = {}

    def schedule(self, view):
        return {"admit": [], "dispatch": []}
```

`reset` receives `workers=8`, `replicas=2`, `batch_size`, and `environment_shards`. State survives batches within a pass only. Each `schedule` result has exactly two lists: ordered `admit` IDs and ordered `dispatch` records of the form `{"id": opaque_id, "replica": 0}`. Replica must be integer 0 or 1. The empty decision shown above is only an interface example: when nothing is in flight, you must admit or dispatch available work.

`view` contains `batch_index`, `trajectories`, `replica_outstanding` (two integer request counts), and `events` (new inference-completion events since the previous callback). Each trajectory contains:

| Field | Meaning |
| --- | --- |
| `id`, `worker` | Opaque per-pass trajectory ID and fixed worker slot 0–7 |
| `state` | `pending`, `environment`, `ready`, `running`, or `done` |
| `prompt_tokens` | Actual context length at the latest ready turn; null before first readiness |
| `generated_tokens` | Cumulative already-observed generated token count |
| `completed_turns` | Completed environment steps |
| `past_service_seconds` | Observed inference service durations including backend queueing |
| `routing` | Previously chosen replicas |

Events contain `id`, `replica`, `service_seconds`, and already-observed `generated_tokens`. No prompt, token identity, action, reward, game path, future length, or reference trace is provided. No preliminary rollout supplies metadata. Admit only pending trajectories, dispatch only ready turns, and never repeat IDs in a decision. You may defer work while other work is in flight. Decisions cannot change worker or environment-shard assignments.

Callbacks run in a separate process with no filesystem, network, process creation, or direct engine access. Initialization and each callback have five seconds; memory is bounded at 512 MiB and each JSON decision at 1 MiB. Permitted preloaded helpers include `collections`, `functools`, `heapq`, `itertools`, `json`, `math`, `random`, `statistics`, and `time`. Use only causal metadata and in-process state. Do not access protocol pipes, inspect other processes, or persist state across passes. Arbitrary imports requiring filesystem access fail.

## Research loop

1. Inspect `/workspace/candidate/policy.py` and prior aggregate feedback.
2. State a falsifiable scheduling hypothesis; change only the policy and optional notes.
3. Develop in Work with real train episodes: `/opt/venv/bin/python /opt/opentinker_task/dev.py --policy /workspace/candidate/policy.py --batch-size 8 --env-shards 2 --episodes 8`.
4. Stop all development GPU processes, close candidate files, and run `rsi-submit`.
5. Compare row timing and wait diagnostics; keep, revise, or reject the hypothesis. Use `rsi-submit --list` for prior summaries.

Development exposes full train traces and writes no Harbor reward. Eight development episodes cost real GPU/environment time. Research, policy optimization, and candidate selection belong in Work; Judge does fixed evaluation only.

## Fixed backend revision

This is task version 1.2.0: vLLM 0.12.0, torch 2.9.0, Transformers 4.57.3,
Python 3.10 and CUDA 12.9.1. Batch-invariant mode and the FLASH_ATTN backend
are fixed and propagated to all inference actors. Do not change them.
Model identity and workloads retain the original contract. Version 1.2.0
allows different trajectories and uses the task-success quality floor below.

## What stays fixed

- All paths except `/workspace/candidate/policy.py` and optional `/workspace/notes.md`, including the source manifest, tools, sources, model, tokenizer, data, engine, evaluator and timers. No extra workspace files, symlinks, runtime downloads, external data, web research, or external services.
- Two identical Qwen2.5-3B-Instruct BF16 replicas, each TP/DP/PP 1, with real weights; eight workers. Greedy decoding: temperature 0, top-p 1, top-k -1, repetition penalty 1 and model EOS behavior. Prefix caching, chunked prefill and eager execution stay enabled; each engine has 32 sequence slots, 8192 batched tokens, and 0.8 memory utilization. Engine batching is outside your control.
- At most 20 model turns and 20 environment steps per episode, 128 new tokens per request, and 8192 actual context tokens. The permitted final action executes and its complete environment observation is recorded. Context exhaustion ends the episode without another request, invented token, partial observation, or post-generation truncation.
- All episodes exactly once per pass. No dropping, replaying, extra generation, resampling, or automatic retries. Only the closed source artifact survives the read-only workspace snapshot; live processes, GPU state and caches are not deliverables.

## Evaluation and feedback

Each of four equal-weight rows `(batch_size, environment_shards) = (8,2), (8,8), (32,2), (32,8)` processes the same 96 distinct reserved unseen games, 16 per task type, in frozen row order. Each row runs full baseline and candidate passes, alternating B/C and C/B order. Each pass starts fresh model, policy, trajectory and environment-process state; eight fixed train warmups use baseline scheduling, then requests drain and prefix caches clear before measurement. Policy counters reset after warmup.

Timing in seconds includes policy execution, admission/reset, inference, queueing, environment work, finalization, original-order DataProto packing, inter-batch orchestration, request draining, and device synchronization. Model/process initialization and declared warmup are excluded from the metric but included in wall-time budgets. Formal work totals 768 episodes, at most 15360 model calls and 1966080 generated tokens; warmup and context processing add work.

Reward is `100 * exp(mean(log(baseline_seconds / candidate_seconds)))` over all four rows, higher is better, with no clipping or successful-subset scoring. Different schedules may produce different tokens, actions, observations, rewards, termination reasons and masks. Each pass must still complete the full workload with valid finite outputs. A successful episode has a terminal ALFWorld success reward (+10). Across the four rows (384 episodes per role), candidate success rate must be at least baseline success rate minus 5 percentage points; at most 19 fewer successes can qualify. This is a threshold on observed results, not a statistical guarantee. All four rows contribute, including unsuccessful episodes. Feedback reports per-row success rates, aggregate quality, token/call counts, and trajectory-match counts; the latter are diagnostic only. Speed ratios describe end-to-end task throughput, not equal-token inference speed.

Any import failure, invalid/stalled decision, crash, illegal memory access, quality regression, incomplete work, OOM, timeout, or infrastructure failure leaves **no reward**. Candidate exception feedback identifies compilation, module initialization, construction, reset, schedule, or result serialization; it includes the exception type, candidate source line when available, and a specific repair hint. Syntax errors include line, column, and the bounded compiler reason. Callback messages and runtime values are withheld because they can contain formal identifiers; protected evaluator frames are never included. The public development command retains these repair details as well as its train traces.

Successful feedback reports row times/ratios, success rates, aggregate quality, trajectory-match counts, completion/model-call/token counts, batch mean/tail durations, and the following per-pass aggregates for both baseline and candidate:

| Fields | Definition and units |
| --- | --- |
| `completed_trajectories_per_sec` | Completed episodes divided by the full measured rollout `seconds`, in trajectories/second |
| `ready_wait_mean_sec`, `ready_wait_max_sec` | Mean and maximum seconds from a turn becoming ready to its dispatch |
| `model_wait_count`, `model_wait_total_sec`, `model_wait_mean_sec`, `model_wait_max_sec` | Number of model requests and sum/mean/maximum seconds awaiting a completed, validated backend response, including transport, backend queueing and inference |
| `environment_reset_wait_count`, `environment_reset_wait_total_sec`, `environment_reset_wait_mean_sec`, `environment_reset_wait_max_sec` | Number of resets and sum/mean/maximum seconds awaiting a reset response, including transport, serialized shard queueing and reset work |
| `environment_step_wait_count`, `environment_step_wait_total_sec`, `environment_step_wait_mean_sec`, `environment_step_wait_max_sec` | Number of steps and sum/mean/maximum seconds awaiting a step response, including transport, serialized shard queueing and step work |

Wait totals sum overlapping request latencies and can exceed rollout wall time; they are not a decomposition of wall time or pure queue delay. Empty request sets have zero count/total/mean/maximum. Environment close/finalization stays in the scoring interval but is outside reset/step diagnostics. Warmups contribute to none of these formal aggregates. Formal selection/order/seeds, episode contents, per-episode outcomes, raw logs, and evaluator implementation remain Judge-only. Every printed line and the Harness footer are visible to Work, including status, duration, exit code, timeouts, errors, remaining submissions and reward when available.

## Submission checklist

- Candidate and optional notes are regular, complete, flushed files under the allowed paths; the fixed manifest is unchanged.
- All Work experiments and GPU processes have stopped; Judge needs no live Work state.
- The next submission tests a stated hypothesis within the remaining wall-time and submission budgets.

The vLLM 0.12.0 backend also includes the checksum-pinned SM120 dispatch
repair in `environment/backend_patches.json`: RTX Blackwell with torch 2.9
uses vLLM's existing invariant GEMM kernels, as the SM100 path already does.
Unpatched SM120 produced batch-dependent BF16 values in the hardware probe;
the patched kernels passed the same probe. Full live workload and quality qualification
is still required. The Judge verifies the patched backend file hash.
