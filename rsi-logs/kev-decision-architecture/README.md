# Kev decision architecture — recorded runs

Task: [instructions](../../rsi-tasks/kev-decision-architecture/instruction.md) · [maintainer guide](../../rsi-tasks/kev-decision-architecture/README.md).

| Agent | Reasoning effort | Persisted submissions | Best reward | First best submission | Elapsed hours |
| --- | --- | ---: | ---: | --- | ---: |
| [Claude Opus 5](claude-code-claude-opus-5/) | max | 10 | 0.305 | agent-9 | 12.01 |
| [GPT-5.6 Sol](codex-gpt-5.6-sol/) | xhigh | 49 | 0.292 | agent-44 | 12.01 |

Reward is `exp(-macro_nll)` over the 1,958 scored Judge questions, where `macro_nll` weights the knowledge group (MMLU, MMLU-Pro) and the other group (12 transfer and control sources) equally; it lies in `[0, 1]`, higher is better. The validation-measured baseline, the untouched Kev reference trained once in Work, is `0.25186` (`macro_nll` 1.3789); uniform predictions score `0.2488`. Both runs beat the baseline from their first or second submission. Claude's best `0.30518` first appeared at `agent-9` and was reproduced at `agent-10`; Codex's best `0.29233` first appeared at `agent-44` and was reproduced at `agent-49`.

The runs differ in submission cadence. Codex submitted after most changes (49 submissions). Claude selected its candidates on the public development data first and made its first submission after about 4.5 hours; its first Judge score was `0.2913`. About 3.5 hours into the run, one Claude API request returned HTTP 500; the session then waited on two of its own background polling commands whose conditions could not occur, and resumed about 32 minutes later when those commands were stopped by the operator. The transcript records both events; no Agent state was changed.

All 59 persisted submissions completed with a numeric score. RSI-Harness marks a run that reaches its Agent budget as `status: failed, timed_out: true` in `final_result.json`; the scores are complete and publication does not change either outcome.

Each run includes `run-plan.json`, `final_result.json`, `evolve_state.json`, the ordered submissions and verifier rewards, per-round Judge feedback, and the complete sanitized agent/runner output. The Codex transcript is losslessly compressed as `agent_output.txt.gz` because its uncompressed text exceeds GitHub's per-file limit; its run-local README gives the restoration command and `agent_output.manifest.json` records the published checksums and sizes. No transcript events were dropped for size. The two runs executed concurrently on one host, each with a two-GPU pool that Work and Judge shared (release-all).

Run identifiers and original ordering are retained. Sanitization removes operator and infrastructure details, not task content, scores, timings, model identities, or artifact digests; see [the sanitization policy](../SANITIZATION.md).

Task snapshot: `e872c39` (private task repository, published here as `rsi-tasks/kev-decision-architecture/` with public documentation adapted; protocol unchanged). The bundled RSI-Harness was identical to this repository's `RSI-Harness/` at `1b30b3c`.
