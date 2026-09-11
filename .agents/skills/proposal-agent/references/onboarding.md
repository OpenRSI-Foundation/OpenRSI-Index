# FrontierRSI AutoResearch Contributor Orientation

Deliver the orientation below once at the start of a new proposal. Use the contributor's language while preserving its meaning and requirements.

Present all onboarding information below before beginning the Round 0 questions; do not omit any section. Do not disclose the full Round 0 question list at once.

Present all content between `(START)` and `(END)` below. For English, reproduce the wording exactly; for other contributor languages, translate it faithfully while preserving every section and requirement. Do not include the boundary markers in the response.

(START)
## What FrontierRSI is evaluating

FrontierRSI turns representative, fully open model-development projects into auditable research environments. A research agent starts from a human-built implementation or artifact, proposes a hypothesis, changes a candidate, runs a fixed experiment, observes declared feedback, updates its hypothesis, and repeats. The objective is to measure whether the agent can discover an improvement over a matched human baseline while preserving the experiment's scientific contract.

## Design principles

1. **Optimization, not reproduction.** The AutoResearch task must give the agent room to improve a real model-development workflow beyond the original human baseline.
2. **Scientific discovery, not mechanical search.** The task should support distinct hypotheses and teach something about why an approach helps or fails. A one-shot implementation, routine port, or ordinary hyperparameter sweep is not enough.
3. **Production relevance, not a toy result.** The selected experiment should preserve the causal mechanism and evaluation path of the source project. When the full workflow is too expensive, a faithful slice is acceptable only if improvement on that slice remains informative about the real problem.
4. **Prefer a live frontier.** Frontier timing is a prioritization signal. Prefer questions that address newly important capability in the current research cycle. A less active area receives lower priority on an active frontier.

## Bring one project

Start by choosing one project through either route:

1. **Coauthored route:** a representative project you coauthored; or
2. **Domain-expertise route:** a well-known project in your field whose codebase you know especially well.

In either route, explain your relationship to the project and why your experience matches the proposed research area. Contributor-domain fit is a strict eligibility gate.

Use the official open-source repository when available and pin an immutable commit or exact tag. A paper may explain the motivation, but repository and artifact evidence must support a runnable task.

## What makes a project suitable for AutoResearch

| Characteristic | What it means |
|---|---|
| Fully open | The selected lane has public, usable code, configurations, required model/checkpoint and data assets, and an evaluation path. A vague promise that an operator will pre-provision missing assets later does not establish availability. |
| Influential | The project shaped its area through adoption, citations, community use for frontier work. |
| Timely | The research area remains active in the current research cycle. |
| Reproducible | The official experiment can run end to end. Missing critical assets or an unverifiable metric block admission. |
| Iterative | The task supports repeated hypothesis → change → run → observe → update cycles. |
| Verifiable | A separate fixed evaluator can score the submitted artifact and enforce integrity boundaries without trusting self-reported results. |

## Common poor fits

- A paper reproduction with no subsequent improvement loop.
- A one-shot feature, port, refactor, or bug fix.
- A search space dominated by routine parameter sweeping.
- A cheap toy proxy whose gains are not informative about the source project's real workflow.

## Compute orientation

One candidate experiment means taking one fixed idea and configuration from launch through any required training or optimization and evaluation until it produces a valid score. Work and Judge may each use zero GPUs; CPU-only tasks are eligible under the same research and evaluation standards. The selected lane must fit on a single physical node and use at most 8 GPUs at peak. For GPU lanes, H100 is the budgeting reference, not a required model: compatible A100, B100, or other GPUs are allowed unless the task genuinely requires a specific GPU model. A lane that inherently requires multi-node execution or a larger peak is not admitted; a faithful repository-supported eligible lane may be selected instead. Runtime over 12 wall-clock hours remains a non-blocking compute flag for later resource review. This applies to one experiment, not the full multi-experiment AutoResearch trajectory.
(END)

## What to provide in Round 0

Ask for:

Ask only one item at a time. Wait for the contributor's answer before asking the next item. Do not ask for the entire list in one response.

1. the contributor's full professional name and publication email;
2. which project-selection route the contributor wants to choose: **Coauthored route:** a representative project they coauthored; or **Domain-expertise route:** a well-known project in their field whose codebase they know especially well;
3. the one selected project and its official repository URL, and a brief explanation of the contributor's relationship to the project and domain fit;
4. the exact commit or tag, if already known;
5. a few sentences describing the initial research question;
6. why the underlying area is foundational or frontier-relevant currently; and
7. whether the task uses CPUs only or GPUs, and whether one candidate experiment likely needs multiple physical nodes, more than 8 GPUs at peak, or more than 12 wall-clock hours.

Do not request exact compute details yet. The proposal agent investigates repository facts and current frontier evidence; the contributor supplies expert judgment and explicit decisions.
