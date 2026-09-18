# MolmoWeb interaction context — recorded runs

Task: [instructions](../../rsi-tasks/molmoweb-interaction-context/instruction.md) · [maintainer guide](../../rsi-tasks/molmoweb-interaction-context/README.md).

| Agent | Reasoning effort | Persisted submissions | Best reward | First best submission | Elapsed hours |
| --- | --- | ---: | ---: | --- | ---: |
| [Claude Opus 5](claude-code-claude-opus-5/) | max | 39 | 0.6057279631114341 | agent-8 | 24.01 |
| [GPT-5.6 Sol](codex-gpt-5.6-sol/) | xhigh | 40 | 0.6053201061214462 | agent-37 | 24.01 |

Reward is the macro mean of graded per-action means over the fixed 948-step replay evaluation, higher is better. Both agents' first submissions reproduce the unchanged-strategy baseline: `0.5966237350781738`. These replay scores are not the upstream live-browser benchmark.

Each run includes `run-plan.json`, `final_result.json`, `evolve_state.json`, all submission and verifier artifacts, and complete sanitized agent/runner output. All 79 persisted submissions have completed numeric results. The shutdown summaries are stale by one submission: `final_result.json` reports 38 for Claude and 39 for Codex, while the persisted state and per-submission records contain 39 and 40. Both original records are preserved; the table and website use the persisted submission history.

Run identifiers and original ordering are retained. Sanitization removes operator and infrastructure details, not task content, scores, timings, model identities, or artifact digests; see [the sanitization policy](../SANITIZATION.md). Final process/shutdown status is retained as evidence and is separate from individual submission scores.

Task and run source snapshot: `c4d8cc3997ca1a3845422ed037926d71c666b8ef`. Public task documentation and contact metadata are adapted for this repository; the experimental protocol is unchanged.
