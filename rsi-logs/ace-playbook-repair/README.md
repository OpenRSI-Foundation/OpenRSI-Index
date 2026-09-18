# ACE playbook repair — recorded runs

Task: [instructions](../../rsi-tasks/ace-playbook-repair/instruction.md) · [maintainer guide](../../rsi-tasks/ace-playbook-repair/README.md).

| Agent | Reasoning effort | Persisted submissions | Best reward | First best submission | Elapsed hours |
| --- | --- | ---: | ---: | --- | ---: |
| [Claude Opus 5](claude-code-claude-opus-5/) | max | 30 | 0.730 | agent-14 | 24.01 |
| [GPT-5.6 Sol](codex-gpt-5.6-sol/) | xhigh | 18 | 0.705 | agent-10 | 24.00 |

Reward is accuracy on the fixed 200-example hidden Formula evaluation, higher is better. The Judge-measured stock **adapted** ACE baseline is `0.570`, recorded as Codex submission 1. Claude's initial empty-playbook score (`0.565`) is a different, unadapted control.

Each run includes `run-plan.json`, `final_result.json`, `evolve_state.json`, the ordered submissions and verifier artifacts, and the complete sanitized agent/runner output. Submission 2 in each run has a verifier error and no numeric score; it remains in the archive and is not interpreted as a zero. Thus the 48 persisted submissions contain 46 scored results.

The Codex transcript is losslessly compressed as `agent_output.txt.gz` because its uncompressed text exceeds GitHub's per-file limit. Its run-local README gives the restoration command and `agent_output.manifest.json` records the published checksums and sizes. No transcript events were dropped for size.

Run identifiers and original ordering are retained. Sanitization removes operator and infrastructure details, not task content, scores, timings, model identities, or artifact digests; see [the sanitization policy](../SANITIZATION.md). Final process/shutdown status is retained as evidence and is separate from individual submission scores.

Task and run source snapshot: `58160d280060d2e512a020545fc1ee859b0ea6ae`. Public task documentation and contact metadata are adapted for this repository; the experimental protocol is unchanged.
