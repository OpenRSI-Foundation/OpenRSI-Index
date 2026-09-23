# Kev decision architecture — recorded runs

Task: [instructions](../../rsi-tasks/kev-decision-architecture/instruction.md) · [maintainer guide](../../rsi-tasks/kev-decision-architecture/README.md).

| Agent | Reasoning effort | Persisted submissions | Best reward | First best submission | Elapsed hours |
| --- | --- | ---: | ---: | --- | ---: |
| [GPT-5.6 Sol](codex-gpt-5.6-sol/) | xhigh | 49 | 0.292 | agent-44 | 12.01 |
| [Claude Opus 5](claude-code-claude-opus-5/) | max | 22 | 0.290 | agent-16 | 12.01 |

Reward is `exp(-macro_nll)` over the 1,958 scored Judge questions, where `macro_nll` weights the knowledge group (MMLU, MMLU-Pro) and the other group (12 transfer and control sources) equally; it lies in `[0, 1]`, higher is better. The validation-measured baseline, the untouched Kev reference trained once in Work, is `0.25186` (`macro_nll` 1.3789); uniform predictions score `0.2488`. Both runs beat the baseline from their first or second submission. Codex's best `0.29233` first appeared at `agent-44` and was reproduced at `agent-49`; Claude's best `0.28955` first appeared at `agent-16` and was reproduced at `agent-17` through `agent-20` and `agent-22`.

The Codex run used task snapshot `e872c39`. An earlier Claude run under that snapshot was withdrawn: its best candidate (`0.30518`) fused the Kev members with a channel that scores answer letters directly from the base model, and its Judge evaluations took about 12 minutes each. The task then gained one rule, every probability must be produced by the Kev decision head (`bb4b066`), and the Claude run recorded here was made under that snapshot; its Judge evaluations took 164 to 222 seconds against 45 seconds for the reference.

All 71 persisted submissions completed with a numeric score. Claude `agent-21` scored `0.0` as `candidate_invalid` (`editable_scope_violation`: a `/workspace/runs.md` note outside `candidate/` and `experiments/`); the Agent removed it and resubmitted as `agent-22`. RSI-Harness marks a run that reaches its Agent budget as `status: failed, timed_out: true` in `final_result.json`; the scores are complete and publication does not change either outcome.

Each run includes `run-plan.json`, `final_result.json`, `evolve_state.json`, the ordered submissions and verifier rewards, per-round Judge feedback, and the complete sanitized agent/runner output. The Codex transcript is losslessly compressed as `agent_output.txt.gz` because its uncompressed text exceeds GitHub's per-file limit; its run-local README gives the restoration command and `agent_output.manifest.json` records the published checksums and sizes. No transcript events were dropped for size. Each run used a two-GPU pool that Work and Judge shared (release-all); the Codex run and the withdrawn Claude run executed concurrently, the recorded Claude run afterwards on the same host.

Run identifiers and original ordering are retained. Sanitization removes operator and infrastructure details, not task content, scores, timings, model identities, or artifact digests; see [the sanitization policy](../SANITIZATION.md).

Task snapshots: Codex `e872c39`, Claude `bb4b066` (private task repository, published here as `rsi-tasks/kev-decision-architecture/` with public documentation adapted; protocol unchanged apart from the added decision-head rule, which the Codex run did not have). The bundled RSI-Harness was identical to this repository's `RSI-Harness/` at `1b30b3c`.
