<!-- Historical record; not the acceptance status for task version 1.2.0. -->

# Historical execution validation: version 1.0.0

This historical exact-equivalence validation has status **FAIL**. The fixed backend has not
qualified for the exact semantic gate, so there is no valid benchmark score
and no research Agent has been launched. Evidence and public development
traces are in `../validation/20261002-blackwell-auto/`.

Three startup defects were repaired without changing model weights,
decoding, workloads, baseline scheduling, or the semantic comparison:

- Nested verl configuration dataclasses now have their importable Hydra
  targets instead of empty `_target_` values.
- The Broker treats cancellation of its idle scheduling pump during its
  deliberate shutdown as normal. Unexpected failures still propagate.
- The framework file-size bound permits Ray's fixed 8 GiB plasma file.
  The supervisor separately enforces the existing 512 MiB raw-log guard;
  policy file writes and result bounds remain restricted.

The current image builds, passes fixed-asset/pristine-workspace checks, and
completes public ALFWorld episodes with the real BF16 Qwen2.5-3B model. The
32-episode scratch Judge reaches a live baseline/candidate pair but rejects
the unchanged baseline policy with `semantic_mismatch`. The final 96-episode,
four-row acceptance has therefore not passed or been scored.

A train-only diagnostic on physical GPUs 2/3 confirms the failure outside
the protected Judge: two fresh public passes using the same baseline and
initial inputs disagree on generated tokens in seven of eight episodes.
Actions, observations, rewards, and termination also differ. Requests
explicitly retain greedy decoding (`temperature=0`, `top_p=1`, `top_k=-1`,
`repetition_penalty=1`). Identical prompt/token prefixes also have log
probability differences beyond 1e-5. This establishes failure of the required
invariance; it does not isolate one particular CUDA kernel as the cause.

The task's vLLM 0.11.0 pin and exact semantic gates are preserved. A change
to the fixed inference backend would constitute a revised benchmark and
requires a separately reviewed decision. The proposed revision and its
acceptance requirements are recorded with the validation evidence.


Version 1.2.0 was authorized by the user to allow different trajectories with
at most a 5-percentage-point aggregate success-rate drop. Its independent
acceptance report is `../validation/20261002-batch-invariant/report.json`.
Earlier failures are retained and never relabeled as successful scores.
