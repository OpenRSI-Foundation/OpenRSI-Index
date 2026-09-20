# Qwen-122B-RL — historical research report

Task: [instructions](../../../rsi-tasks/signature-tasks/post-training-qwen-122B-rl/instruction.md) · [maintainer guide](../../../rsi-tasks/signature-tasks/post-training-qwen-122B-rl/README.md) · [scientific reference](../../../rsi-tasks/signature-tasks/post-training-qwen-122B-rl/environment/assets/scientific-contract.md).

The [Formal v8 research visualization](formal-v8-research-visualization.html)
records a 24-hour synthetic RL data search for Qwen3.5-122B-A10B. Download the
HTML and open it in a browser to view the report; its styles and charts are
self-contained and require no server or external assets.

## Reported results

Values below retain the supplied report's displayed precision. The macro is
the equal-weight mean of the five benchmark scores, in percentage points.

| Checkpoint / training data | PolyMath | LongBench V2 | IFBench | LiveCodeBench | MMLU-Pro | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| t0 Original Checkpoint | 68.06 | 64.02 | 67.01 | 72.06 | 86.30 | 71.490 |
| t40 Human Pick / Anchor Data | 68.21 | 63.42 | 66.67 | 73.43 | 86.81 | 71.708 |
| t40 Research Agent Data | 68.21 | 64.02 | 67.35 | 73.54 | 86.44 | 71.912 |

The report describes 2,560 frozen synthetic records (512 per capability),
11 admitted revisions, and peak probe use of 31 out of 33 GPU nodes. It reports
a macro difference of +0.204 percentage points versus Human Pick / Anchor.

## Provenance and interpretation

This is an operator-supplied historical report, not a reproduced Harness run.
The [task documentation](../../../rsi-tasks/signature-tasks/post-training-qwen-122B-rl/README.md#provenance-and-scientific-boundary)
records a protocol mismatch: the historical Research Agent used a different
training recipe. The report's original "matched t40" wording is retained as
part of the historical narrative; it does not establish a controlled
data-selection comparison. The portable task checks the data contract only
and does not reproduce these downstream model scores.

The supplied HTML did not include its `formal-v8-research-results.json`
companion, the referenced `formal_v8` / `formal_v6` implementation files,
checkpoint hashes, or raw evaluator receipts. Their original filenames are
retained in the page as provenance notes rather than broken links. No missing
run artifacts or measurements have been reconstructed. No replicate seeds,
confidence intervals, or significance tests are supplied.

The archive aligns the page title and task name with `Qwen-122B-RL`, adds this
context and working repository links, and preserves the reported scores,
charts, telemetry, and frozen-data digest.
