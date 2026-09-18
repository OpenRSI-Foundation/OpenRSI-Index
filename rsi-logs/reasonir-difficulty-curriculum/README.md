# ReasonIR difficulty curriculum — recorded runs

Task: [instructions](../../rsi-tasks/reasonir-difficulty-curriculum/instruction.md) · [maintainer guide](../../rsi-tasks/reasonir-difficulty-curriculum/README.md).

| Agent | Reasoning effort | Persisted submissions | Best reward | First best submission | Elapsed hours |
| --- | --- | ---: | ---: | --- | ---: |
| [Claude Opus 5](claude-code-claude-opus-5/) | max | 10 | 0.22340 | agent-6 | 24.01 |
| [GPT-5.6 Sol](codex-gpt-5.6-sol/) | xhigh | 8 | 0.21811 | agent-5 | 24.00 |

The fixed zero-LoRA baseline is `R0 = 0.21141`, with general-retrieval guardrail `G0 = 0.36126`, measured by the formal Judge calibration for this exact pinned protocol. `R` is the mean nDCG@10 over Biology, Pony, and TheoremQA-theorems; it is not full BRIGHT. `G` is the mean over the four fixed BEIR datasets. Reward is `R` when `G >= G0`, and `G - G0` otherwise.

All 18 persisted submissions are completed finite measurements. Five negative rewards are valid general-retrieval regression penalties, not errors or missing results. Claude first reaches its best on submission 6 and repeats it on submissions 8 and 10.

Each run includes `run-plan.json`, `final_result.json`, `evolve_state.json`, all submission and verifier artifacts, and complete sanitized agent/runner output. Run identifiers and original ordering are retained. Sanitization removes operator and infrastructure details, not task content, scores, timings, model identities, or artifact digests; see [the sanitization policy](../SANITIZATION.md). Final process/shutdown status is retained as evidence and is separate from individual submission scores. Neither run has a paired final research summary, so none is invented for the website.

The original Claude transcript contains seven malformed JSONL physical records. They are preserved at the same positions rather than silently repaired or dropped; consumers should not assume every transcript line parses as standalone JSON. The structured submission and reward JSON files are valid and unchanged.

Task and run source snapshot: `ecb044c86add3b6cc52fe44be3f8a7e2696e5c5b`. Public task documentation, contact metadata, and dataset acquisition instructions are adapted for this repository; the experimental protocol and required asset checksums are unchanged.
