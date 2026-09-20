# Isaac Lab PegInsert reward search

## Overview

This Harbor task studies bounded reward-graph design for fixed-budget PPO on `Isaac-Factory-PegInsert-Direct-v0`. The candidate is `/workspace/reward.json` plus one extra recipe under `/workspace/candidates/`; Judge validates each, freshly trains each for 819,200 transitions (50 epochs) on its own GPU in parallel, retains the selected-epoch player tensors, scores terminal insertion success over 1,024 episodes per recipe, and reports the best recipe's value as the submission reward.

## Baseline and evidence

| Item | Value |
| --- | --- |
| Official source | `isaac-sim/IsaacLab` at `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8` |
| RL-Games source | `isaac-sim/rl_games` at `6b3534f29568158e9e29ec8bf83cc88fce5f0cae` |
| Matched baseline | Source-derived keypoint-squash `(5,4)`, `(50,2)`, `(100,0)`, engagement and success recipe; `selected_epoch=50` |
| Reported result | `0.378` (387/1024) for the original reward under the fixed protocol (validation run, 2026-09-19); the stock 10 s / narrow-reset environment scores `0.839` at 50 epochs and `0.985` at 200 |
| Released reference | Build-only authenticated conversion of the official 212,889,115-byte checkpoint, SHA-256 `1a1388408d119959a19d28efd1e56c3d498cd6eed1d259c48e9319c756081496`; not reachable from ordinary scoring |

The build applies the complete exact CPython 3.11/Linux closure in `environment/requirements.lock` in one dependency-resolution-disabled install, including every selected Isaac Sim 5.1.0 component, PyTorch/torchvision/torchaudio 2.7.0 CUDA 12.8 and their CUDA wheels, NumPy 1.26.4, and Safetensors 0.6.2. Isaac Lab is then installed editable (its packages read `config/extension.toml` from the source tree, as upstream `isaaclab.sh -i` does) and RL-Games as a regular wheel, both from immutable local checkouts with dependency resolution disabled; `poetry-core` is present only while RL-Games builds. The lock deliberately preserves the confirmed NumPy 1.26.4, Isaac Lab Starlette 0.49.1, and RL-Games Watchdog 2.3.1 runtime pins where older upstream package metadata conflicts; final API compatibility remains an execution check. Build-time and Judge preflights independently require every locked distribution and the five local source distributions to match exactly. See `environment/prepare_assets.py` and `environment/convert_reference.py` for setup identities. The reference converter uses authenticated, restricted CPU loading and exports only the 28 player tensors.

## Environment and evaluation

| Item | Value |
| --- | --- |
| Workdir | Split `/workspace`, initialized with the baseline recipe and empty `research/` |
| Environment constants | `environment/fixed_env.json` (shipped to `/opt/peginsert_public`, hash-pinned by Judge): 6 s episodes, hand start noise ±[0.05, 0.05, 0.02] m, yaw noise ±1.57 rad, held-asset offset noise ±[0.006, 0.003, 0.006] m; applied by both Judge and `peginsert-develop` |
| Work/Judge | Two GPUs each (one single-GPU trial or Judge recipe per device); 32 CPUs; 131,072 MiB RAM; 32 GiB shared memory |
| Network | Build may fetch pinned public inputs; Work and Judge are `no-network` |
| Public tools | `/usr/local/bin/validate-reward`; `/usr/local/bin/peginsert-develop --recipe /workspace/reward.json --output-dir /workspace/research/trial [--device cuda:N]` (diagnostic only; the intended loop submits two recipes per round and lets Judge train them) |
| Judge | Per recipe on `cuda:0` / `cuda:1`: candidate-only fresh training, then eight 128-environment batches; `successes / 1024`; submission reward = best recipe, with `candidate_N` values beside it in `reward.json` |
| Timeouts | Build 7,200 s; Agent 75,600 s; Verifier 19,800 s |

`/tests/reference_eval.py` is a maintainer-only measurement entrypoint for the authenticated released Safetensors artifact. It never writes the ordinary submission reward and is not selectable through candidate data.

Judge copies each regular candidate into its own private scratch before simulator imports, validates all of them, and only then starts one worker process per candidate. `/tests/bootstrap.py` checks candidate scope, immutable repository revisions/cleanliness, pinned package identities, wheel RECORD bytes, task-owned runtime bytes, and offline asset/reference manifests. The evaluator never imports from `/workspace`. The public validator is convenient but not authoritative.

The shared Base/Judge model is not root-independent attestation. Task checks reduce task-specific tampering, while fresh Judge state and the read-only workspace snapshot keep Work processes and modified base files out of scoring.

## Run and validate

```bash
export RSI_MODEL='operator-selected-model'
export RSI_REASONING_EFFORT='operator-selected-effort'
export RSI_GPU_POOL='operator-selected-compatible-pool'
sudo -E rsi-harness run "$PWD/rsi-tasks/isaaclab-peginsert-reward-search" \
  --agent codex \
  --model "$RSI_MODEL" \
  --reasoning-effort "$RSI_REASONING_EFFORT" \
  --gpus "$RSI_GPU_POOL" \
  --primary-reward reward \
  --score-direction maximize \
  --verbose
```

Run the example from the OpenRSI-Index repository root after installing [RSI-Harness](../../RSI-Harness/). Set the model, reasoning effort, and GPU pool variables for your environment; the pool needs four GPUs (two Work, two Judge). There is no task-imposed submission-count cap.

Python syntax check (not an environment build or GPU validation):

```bash
python3 -m compileall -q rsi-tasks/isaaclab-peginsert-reward-search
```

Generation does not build or run the image, Solution, evaluator, baseline, reference, GPU workload, or Agent. The `harbor-task-validator` must still validate Docker buildability, recursive asset closure/hashes, dependency/runtime compatibility, restricted reference conversion and its derived hash, pristine state, baseline/reference loading and scores, transition/episode counters, GPU compatibility/memory, wall time/storage/nondeterminism, fresh-container execution, and real-Agent behavior.
