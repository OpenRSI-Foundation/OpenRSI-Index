# [Task Title]

Complete every row below. Keep the proposal concise and use repository evidence
rather than paper claims. This proposal is written before baseline reproduction,
so do not invent measured baseline results, run-to-run variance, verified
runtime, or final verifier behavior.

| Section | Field | Proposal |
| --- | --- | --- |
| Contributor | Full name | [Contributor's full professional name; contributor provides, proposal agent verifies against public expertise evidence] |
| Contributor | Email | [Contributor-provided email for publication metadata; proposal agent records verbatim and does not infer] |
| Research Question | Repository URL | [Official source repository URL; contributor provides, proposal agent verifies] |
| Research Question | Exact commit/tag | [Resolved immutable commit SHA; include the requested tag or branch only as additional provenance] |
| Research Question | Scientific question | [Focused falsifiable model-development hypothesis; proposal agent drafts, contributor confirms] |
| Research Question | Why this is scientific research rather than pure engineering optimization | [Hypothesis, matched comparison, iterative change-run-observe-update loop, and knowledge gained; proposal agent drafts, contributor confirms] |
| Research Question | Starting environment and artifacts | [Repository state, configs, data, official checkpoints, and other starting assets, with verified public availability or a concrete existing delivery/access interface usable by the current task workflow; use the current shared Base/Work/Judge snapshot model rather than an independent clean Base; proposal agent drafts, contributor confirms] |
| Research Question | Agent's final deliverable | [Candidate-owned model, checkpoint, code, configuration or manifest, data artifact, or other output; candidate-producing compute runs in Work and Judge reloads and evaluates the complete materialized candidate snapshot by default; proposal agent drafts, contributor confirms] |
| Research Question | Files/components the agent may modify | [Explicit research action space; proposal agent drafts, contributor confirms] |
| Research Question | Prohibited actions | [Task-changing behavior and task-specific reward-hacking paths; proposal agent drafts, contributor confirms] |
| Reference Baseline | Baseline implementation or artifact | [Prefer an official released evaluation-ready checkpoint or artifact; use a training recipe only when no suitable artifact exists or fresh training is scientifically required; state how Solution materializes the traceable baseline without training or evaluation; contributor chooses, proposal agent verifies] |
| Reference Baseline | Repository evidence paths | [Repository-relative paths and external artifact revisions supporting the baseline, configuration, evaluator, metric, and matched comparison; proposal agent verifies] |
| Reference Baseline | Baseline evaluation path | [Exact evaluator and command or launch path that scores the baseline artifact under the matched protocol; proposal agent verifies] |
| Reference Baseline | Baseline metric reported by the repository | [Metric, value if reported, and source; label it not yet reproduced when applicable; do not invent a missing value] |
| Evaluation | Fixed evaluation protocol | [Fixed dataset/workload, evaluator, comparison semantics, per-run budget, counted unit, replacement/retry/resampling accounting, candidate-failure behavior—including attributable compile failure, illegal memory access, and candidate process crash for executable candidates—and matched baseline/candidate execution; use candidate-only evaluation by default rather than a live paired baseline; for retraining, include the evaluator-owned source, data, seeds, training configuration, and metric capture as the fixed evaluation execution contract; contributor drafts, proposal agent verifies] |
| Evaluation | Evaluation mode and input | [Prefer direct evaluation of a submitted model, checkpoint, or artifact. If using evaluation-time retraining under a fixed protocol, name the submitted declarative configuration or manifest and provide a retraining justification explaining why direct evaluation cannot answer the scientific question] |
| Evaluation | Reward or score definition | [Metric, direction, aggregation, units, and candidate-invalid behavior; candidate-invalid artifacts are unscored, as are timeout, infrastructure, and incomplete outcomes, unless the contributor explicitly defines a finite candidate-failure scalar; contributor drafts, proposal agent verifies] |
| Evaluation | Feedback visible to the agent | [All feedback returned after each submitted candidate; normally aggregate scores and bounded diagnostics, plus any additional errors, logs, or trajectories] |
| Evaluation | Evaluation information hidden from the agent | [Evaluation content not returned by default, such as examples, answers, per-example outcomes, and evaluator internals; state any exceptions] |
| Evaluation | Measures preventing reward hacking | [Realistic controls against leakage, memorization, hard-coding, evaluator tampering, fabricated results, and adaptive overfitting, plus material residual limitations of the shared Base/Work/Judge environment: Judge uses the Work snapshot and task-owned tests are Judge-only, not an independent clean Base; proposal agent drafts, contributor confirms] |
| Evaluation | Noise handling and meaningful improvement | [Proposal-stage plausibility argument; after baseline reproduction, use repeats or formal uncertainty estimates only when observed variability could change the conclusion] |
| Workspace | Is web search required? | [Disabled by default, or define its research purpose and boundary; contributor decides and justifies] |
| Workspace | May the agent use external services? | [Record web search separately from each external service. For each service, state No or list its purpose, data flow, boundary, and concrete existing delivery/access interface usable by the current workflow; contributor decides and justifies] |
| Workspace | May the agent construct or collect additional data? | [State whether additional-data access is disabled, or define the allowed operations and scope separately from web and service access; contributor decides] |
| Workspace | Leakage and reward-hacking safeguards | [Concrete safeguards for every enabled access or data path; proposal agent drafts, contributor confirms] |
| Compute Feasibility | GPU type and number per single experiment run | [Contributor estimate, including physical-node count; Work must use at least 1 GPU, while Judge may use zero GPUs when appropriate; when Work candidate production and Judge evaluation differ, give each phase's peak hardware as well as the end-to-end peak for one fixed candidate] |
| Compute Feasibility | Estimated runtime per single experiment run | [Contributor estimate; when Work candidate production and Judge evaluation differ, give each phase's wall time and the end-to-end wall time; distinguish all of them from full trajectory time] |
| Compute Feasibility | Early-stopping signals / lower-cost proxy experiments | [For significantly over-reference runs, repository-supported plan chosen by contributor; for ordinary runs, N/A or clear failure termination only] |

The admitted lane must use at least 1 GPU in Work, fit on a single physical node,
and use at most 8 GPUs at peak. H100 is the budgeting reference, not a required
model: compatible A100, B100, or other GPUs are allowed unless the task genuinely
requires a specific GPU model. Judge may use zero GPUs when appropriate.
CPU-only Work, multi-node, or larger-peak lanes are rejected unless
the contributor selects a faithful repository-supported eligible lane.
Runtime over 12 hours remains a non-blocking resource-review flag after baseline
reproduction. Early-stop or proxy advice is reserved for long-runtime or
comparably material repository-supported cost.
