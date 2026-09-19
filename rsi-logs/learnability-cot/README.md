# Learnability-aware long/short CoT adaptation — recorded runs

Task: [instructions](../../rsi-tasks/learnability-cot/instruction.md) · [maintainer guide](../../rsi-tasks/learnability-cot/README.md).

| Agent | Reasoning effort | Persisted submissions | Best reward | First best submission | Elapsed hours |
| --- | --- | ---: | ---: | --- | ---: |
| [Claude Fable 5](claude-code-claude-fable-5/) | default | 23 | 47.518 | agent-20 | 7.70 |
| [GPT-5.6 Sol](codex-gpt-5.6-sol/) | xhigh | 27 | 46.081 | agent-15 | 7.81 |

Reward is the unweighted mean of the five fixed benchmark accuracies (AIME 2024, AMC 2023, English OlympiadBench, GSM8K test, MATH-500), in percentage points, higher is better. The Judge-measured official Mix-Long baseline is `46.073` (per-task 10.0 / 47.5 / 26.96 / 81.50 / 64.4), reported alongside every scored candidate as `candidate_minus_baseline`.

Each run includes `run-plan.json`, `final_result.json`, `evolve_state.json`, the ordered submissions, per-round Judge feedback and verifier rewards, and the complete sanitized agent/runner output. Submissions that failed the candidate correctness gate (Claude: 3, Codex: 4, all early rounds) score `0.0` by protocol and remain in the archive.

The Codex run was recorded on 2026-09-18 with a 27,720 s (7.7 h) Agent budget so that both agents received the same wall-clock allowance. RSI-Harness marks a run that reaches its Agent budget as `status: failed, timed_out: true` in `final_result.json`; the scores are complete and the last Judge round finished before shutdown. This archive replaces an earlier Codex recording that ended after 6,461 s of a 216,000 s budget and therefore did not reflect a comparable allowance.

Run identifiers and original ordering are retained. Sanitization removes operator and infrastructure details, not task content, scores, timings, model identities, or artifact digests; see [the sanitization policy](../SANITIZATION.md).

Codex run task and Harness snapshot: `738c0ea3` (OpenRSI-Index main, 2026-09-18).
