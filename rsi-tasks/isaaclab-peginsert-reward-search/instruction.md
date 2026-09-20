# Reward Structure Search for Isaac Lab PegInsert

Design a bounded reward graph that improves terminal peg-insertion success after a fixed fresh PPO training budget.

## Goal

- Test whether reward structure, weights, phase conditions, or bounded reward-only memory improve insertion success over the original reward.
- Keep `Isaac-Factory-PegInsert-Direct-v0`, PPO, seed, transition count, and evaluation fixed.
- Use each submission to produce evidence about specific reward-design hypotheses: Judge is the experiment, and every round can test two recipes at once.

## Workspace

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Starting state | Original-reward `reward.json` copied from `/opt/peginsert_public/baseline_reward.json`, an empty `/workspace/research/`, fixed Isaac Lab tools, offline simulator assets, and the fixed environment constants in `/opt/peginsert_public/fixed_env.json` |
| Deliverable | One closed regular `/workspace/reward.json` of at most 64 KiB, plus one optional extra recipe `/workspace/candidates/reward_1.json` (same limits) |
| GPUs | Two Work GPUs (`cuda:0`, `cuda:1`); Judge has two GPUs of its own |

`reward.json` contains `schema_version`, `selected_epoch`, `memory_size`, a topologically ordered `nodes` array, an `output` scalar reference, and simultaneous scalar `memory_updates`. Run `/usr/local/bin/validate-reward --json /workspace/reward.json` for the public grammar, type, size, node-count, depth, and memory checks.

Public scalar inputs are `peg_diameter`, `peg_height`, `hole_diameter`, `hole_height`, `episode_progress`, `keypoint_distance`, `engaged`, and `success`. Vector/quaternion inputs are the peg, hole, and fingertip `position`, `quaternion`, `linear_velocity`, and `angular_velocity` values, plus `joint_position`, `joint_velocity`, `action`, and `previous_action`. Memory inputs are named `memory_0` through `memory_31` as declared by `memory_size`.

Each node has `id`, `op`, and `args`. The closed operations are finite `const`; `add`, `sub`, `mul`, guarded `div`, `scale`, `min`, and `max`; `neg`, `abs`, `square`, nonnegative `sqrt`, `tanh`, `sigmoid`, clamped exponential, and `log1p_abs`; vector `norm`, `dot`, component sum/mean, named `x`/`y`/`z`/`w`, and fixed `component_0` through `component_6`; quaternion conjugation/multiplication/rotation; scalar comparisons and Boolean logic; typed `where`; bounded `clamp`; and `keypoint_squash` with nonnegative `a` and `b`. References may use only public inputs, prior nodes, or declared prior-value memory. The public validator is the canonical detailed syntax guide; Judge independently validates the same language.

## Reference baseline

| Item | Value |
| --- | --- |
| Baseline | Original three keypoint-squash terms with `(a,b)` values `(5,4)`, `(50,2)`, `(100,0)`, plus unit engagement and success bonuses and zero action penalties; epoch 50 |
| Reported result | `0.378` (387/1024) under this task's fixed protocol, measured during task validation with the original reward |
| Status | Baseline reproduced under the fixed protocol (deterministic across repeated runs); the released reference checkpoint was trained under the original 10 s / narrow-reset setting and is not a comparable score |
| Comparison | A baseline submission and every candidate use the same fresh 50-epoch Judge route; an ordinary submission trains and evaluates only its submitted recipe |

## Research loop

1. Inspect the baseline recipe and prior safe feedback with `rsi-submit --list`.
2. State up to two falsifiable reward hypotheses and write one recipe each: `/workspace/reward.json` plus `/workspace/candidates/reward_1.json` (use `/workspace/research/` for notes and Work-only tooling).
3. Validate every recipe with `/usr/local/bin/validate-reward --json <recipe>`, then submit. Judge trains both in parallel on its own GPUs, so a round with two recipes costs the same wall time as a round with one; submit two whenever you have two distinct hypotheses.
4. Compare the per-candidate scores, then keep, revise, or reject each hypothesis and write the next two.

Do not spend Work time on full local training: `/usr/local/bin/peginsert-develop --recipe <recipe> --output-dir /workspace/research/trial --device cuda:N` runs the same 50-epoch training as Judge (about 40 minutes per GPU), produces no Judge evidence, and only delays the next submission. It exists for short diagnostic runs of a recipe's numerical behaviour, not as a pre-submission gate.

## What you may change

- `/workspace/reward.json`: reward nodes, finite constants, graph structure, phase conditions, bounded scalar memory updates, and `selected_epoch` from 1 through 50.
- `/workspace/candidates/reward_1.json`: optional extra recipe with the same contract, trained and scored independently in the same submission.
- `/workspace/research/`: Work-only notes, scripts, logs, trajectories, and development policies. Judge ignores this directory.
- The reward language exposes copied peg, hole, and fingertip poses and velocities; seven robot joints; current and previous six-dimensional actions; fixed geometry; episode progress; keypoint distance; engagement; and success.

## What stays fixed

- Simulator assets, geometry, physics, materials, observations/actions, controller and smoothing, termination, and terminal-success predicate.
- The environment constants in `/opt/peginsert_public/fixed_env.json`, applied identically by Judge and Work: 6 s episodes (instead of the stock 10 s) and wider reset randomization — hand start offset up to ±5/±5/±2 cm, start yaw up to ±90°, and peg-in-gripper offset up to ±6/±3/±6 mm. The stock Factory reward reaches `0.378` here, so the policy must learn to locate, align, and insert quickly.
- Actor/critic architecture, PPO configuration, training seed `0`, single-GPU execution per recipe, 128 environments, horizon 128, and exactly 50 epochs (6,400 vector steps; 819,200 transitions).
- Candidate JSON cannot contain code, paths, imports, callbacks, tensors, arbitrary indexing, batch reductions, RNG, I/O, restore state, or a reference-evaluation selector.
- Work checkpoints, logs, processes, sockets, caches, and GPU state are not deliverables and cannot establish a score.

## Evaluation and feedback

| Item | Value |
| --- | --- |
| Fixed workload | Judge validates every submitted recipe, then for each one on its own GPU in parallel: freshly trains all 50 epochs, retains the model immediately after `selected_epoch`, and evaluates exactly 1,024 episodes in eight batches of 128 |
| Reward | Per recipe `successful terminal episodes / 1024`, in `[0,1]`; the submission reward is the best recipe's value (ties go to the lowest index); maximize |
| Scoreable candidate | Valid `/workspace/reward.json`; Judge-side fresh training is the confirmed exception required to enforce initialization and transition accounting |
| Visible feedback | Per-candidate score/totals, elapsed time, selected epoch, charged/completed counts, aggregate reset statistics, and device, the best candidate index and its score (`candidate_0` .. `candidate_3` also appear in `reward.json`), bounded diagnostics, safe tracebacks, and the Harness footer/full log path |
| Hidden | Evaluation seeds and sampled resets, per-episode outcomes, trajectories, raw Judge training trajectories, selected tensors, and private tests |

Invalid, nonfinite, crashed, timed-out, incomplete, dependency/device, evaluator, or infrastructure paths are unscored and write no reward; one invalid or failed recipe voids the whole submission. A complete run with zero successful episodes truthfully scores `0.0`.

## Submission checklist

- `/workspace/reward.json` (and every `/workspace/candidates/reward_N.json`) is a flushed regular file no larger than 64 KiB; nothing else is under `/workspace/candidates`.
- `/usr/local/bin/validate-reward --json <recipe>` succeeds for every submitted recipe.
- Candidate-producing Work commands are finished; no Work process or checkpoint is required by Judge.
- No path other than `/workspace/reward.json` and `/workspace/candidates/reward_N.json` is expected to affect scoring.
- The next submission tests a stated hypothesis.
