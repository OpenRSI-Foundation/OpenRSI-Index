# Isaac Lab PegInsert reward search — recorded runs

Task: [instructions](../../rsi-tasks/isaaclab-peginsert-reward-search/instruction.md) · [maintainer guide](../../rsi-tasks/isaaclab-peginsert-reward-search/README.md).

| Agent | Reasoning effort | Persisted submissions | Best reward | First best submission | Elapsed hours |
| --- | --- | ---: | ---: | --- | ---: |
| [Claude Opus 5](claude-code-claude-opus-5/) | max | 32 | 0.602 | agent-14 | 24.37 |
| [GPT-5.6 Sol](codex-gpt-5.6-sol/) | xhigh | 32 | 0.604 | agent-23 | 24.23 |

Reward is terminal insertion success over 1,024 private-reset episodes after a fresh 50-epoch PPO training of the submitted reward graph under the fixed 6 s / wide-reset protocol, in `[0, 1]`, higher is better. Each submission carries up to two recipes; Judge trains and scores both in parallel and reports the better one as the submission reward, with `candidate_0` and `candidate_1` beside it. The validation-measured baseline for the original reward is `0.378` (387/1024). Neither run submitted the stock recipe unchanged, so every score is a modified reward. Claude's `0.602` (616/1024) first appeared at `agent-14` and was reproduced at `agent-16`, `agent-18`, and `agent-26` through `agent-31`; Codex's `0.604` (619/1024) appeared once.

Each run includes `run-plan.json`, `final_result.json`, `evolve_state.json`, the ordered submissions and verifier rewards, per-round Judge feedback, and the complete sanitized agent/runner output. The two runs executed concurrently on one host, each with two Work and two Judge GPUs.

Four persisted submissions have no numeric score and are not interpreted as zeros:

- Codex `agent-2` (`infrastructure_error`): the per-round Judge network was missing when the Judge container started, a transient Docker condition; the Agent resubmitted.
- Claude `agent-17` (`verifier_error`): the submission carried an empty `/workspace/candidates/` directory, which the evaluator's candidate-scope check rejects.
- Codex and Claude `agent-32` (`infrastructure_error`): the Judge round in flight when each run reached its 86,400 s Agent budget. Judge measured both recipes (Codex `0.426` / `0.140`, Claude `0.602` / `0.484`, retained in `metrics`), but the budget shutdown had removed the Work container before Judge cleanup could unpause it, so the round closed with `score: null`.

Thus the 64 persisted submissions contain 60 scored results. RSI-Harness marks a run that reaches its Agent budget as `status: failed, timed_out: true` in `final_result.json`; the scores are complete and publication does not change either outcome.

Run identifiers and original ordering are retained. Sanitization removes operator and infrastructure details, not task content, scores, timings, model identities, or artifact digests; see [the sanitization policy](../SANITIZATION.md).

Task snapshot: `a9f8a820e51e894d75a75aa7c9e0f3dac1b3c6a2` (private task repository, published here as `rsi-tasks/isaaclab-peginsert-reward-search/` with public documentation adapted; protocol unchanged). The bundled RSI-Harness was identical to this repository's `RSI-Harness/` at `b0f3514`.
