# Optimizer update-geometry scaling-ladder harness

This is the candidate workspace for a single optimizer mechanism evaluated
from 550M through 2.545B parameters. Follow `/app/task/instruction.md` and use
`/task-tools/run_optimizer_scaling_ladder.sh`; never submit a nested LSF job.

Candidate logic belongs in `runtime/optimizer_candidate.py`. The trusted
pretraining entrypoint invokes its optimizer factory during both training and
verifier checkpoint reload. The checked-in implementation reproduces AdamH for
infrastructure smoke only and is not an eligible research submission.

## Locked scaling ladder

| Rung | Model | Parameters | Tokens | GPUs | Batch |
|---|---|---:|---:|---:|---|
| E0 | d1152-L12 | 550.3M | 2.904B | 8 | GBS16 / MBS2 |
| E1 | d1408-L15 | 837.0M | 3.613B | 8 | GBS16 / MBS2 |
| E2 | d1536-L16 | 998.0M | 4.983B | 8 | GBS32 / MBS4 |
| E3 | d1792-L18 | 1.385B | 10.560B | 32 | GBS64 / MBS2 |
| E4 | d2048-L21 | 1.935B | 14.805B | 32 | GBS64 / MBS2 |
| E5 | d2304-L23 | 2.545B | 18.617B | 128 | GBS128 / MBS1 |

All rungs use TP1 / PP1, context length 4096, the fixed production batch/data
trajectory, and seed 0. E0-E5 may run concurrently, but a valid submission
must use the same exact scale-general optimizer source inventory.

## Evaluation

Paloma runs every 5,000 optimizer updates and at the final update. The trusted
entrypoint records synchronized active-step GPU time and token-normalized
training loss in
`optimizer_cost_trace.jsonl`. Matched-cost scoring combines those traces with
the Paloma checkpoints and the offline AdamH reference manifest.

## Runtime layout

- `runtime/optimizer_candidate.py`: candidate-owned scale-general optimizer factory.
- `runtime/pretrain_gpt_marin_adamh.py`: trusted entrypoint.
- `runtime/lm_eval_runner.py`: trusted Paloma evaluator.
- `data_generation/validate_adamh_v6_blend.py`: trusted staged-data validator.
- `run_marin_adamh_bluevela_lsf.sh`: trusted multi-node RDMA launcher.
