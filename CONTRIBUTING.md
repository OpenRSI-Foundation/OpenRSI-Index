# Contributing an RSI task with RSI-Anything

> [!TIP]
>
> ### RSI-Anything: from idea to a ready RSI task <= 1h
>
> Bring the idea and our agents handle the rest. Review, validate, and have fun!

**RSI-Anything** is the RSIs-First-Exam contribution pipeline. It turns your research idea into a private, ready-to-use RSI task repository through an automated workflow, asking for your input only when a task-defining choice genuinely needs it.

![Contributor and agent roles across five steps: share your idea while proposal-agent drafts (20 min); confirm the proposal for automatic Discussion submission and review (5 min); answer only if asked while agents build the private repo (25 min); provide GPUs and launch skills while agents validate, run, and recover (varies by task); request uploads and share feedback while agents upload logs (a few minutes).](assets/figures/rsi-anything-workflow.svg)

*Times are estimated elapsed durations, including agent work.*

### 1. Create a proposal

Clone this repository:

```bash
git clone https://github.com/RSI-Index/RSIs-First-Exam.git
cd RSIs-First-Exam
```

Open Codex or Claude Code (app or CLI), start a fresh session, and paste:

```text
Use the proposal-agent skill at .agents/skills/proposal-agent/SKILL.md in my local RSIs-First-Exam repository. Locate the repository if needed, then follow the skill to complete setup and guide me step by step through creating and submitting an RSI proposal.
```

Run it on a frontier model at its maximum reasoning setting: at least **Codex
with GPT-5.6 Sol at max** or **Claude Code with Opus 5 at max**.
Use that one session for one proposal. The agent will help you define the
research question, baseline, evaluation, scope, and compute requirements, then
generate an RSI task proposal. Review the completed proposal before confirming
it.

### 2. Submit the proposal

After you confirm the proposal, `proposal-agent` creates the
[Task Ideas Discussion](https://github.com/RSI-Index/RSIs-First-Exam/discussions/categories/task-ideas)
from the confirmed file. Return to the original session when review feedback
arrives; the agent fetches the current Discussion, helps revise the proposal,
and updates the same Discussion. A new review runs on every edit.

### 3. Automatic task building

The standard contributor path is:

1. Submit or edit the Discussion.
2. Receive the PASS or REJECT initial check.
3. After PASS, task building starts automatically.
4. Send `/task <answer or correction>` only if asked a genuine task-defining
   question; the answer resumes building automatically.
5. The bot posts ACCEPTED when assumptions are resolved, then continues to the
   private task repository.

At ACCEPTED, keep the instruction with your original session:

> Keep this agent session to address any feedback or revisions that may arise.
> Once the private task repository is ready, return here and ask me to upload
> this session's required trajectory and Discussion record.

To discard an unpublished attempt, send `/reset`. Reset removes only that
attempt's private State; every Discussion comment, including `/reset`, remains
in the timeline. Wait for the bot's reset-complete reply, then send a plain
`/task` to start a fresh assumptions pass from the accepted proposal.

### 4. Use the private task repository

The generated private repository is a self-contained workspace containing:

- the generated Harbor task in its own directory;
- the bundled read-only `RSI-Harness`;
- the repository-local `harbor-task-validator` and `rsi-task-runner` skills.

The `$...` lines below are skill invocations for your coding agent, not shell
commands.

> [!IMPORTANT]
> **GPU access is required to complete a contribution.** You must have access
> to the required GPU resources declared by your task. Choose a task scope,
> experiment plan, and compute budget that fit those resources.

When your private task repository is available, return to the original
proposal-agent session and paste:

```text
Find this session's complete original trajectory and submit it, together with the Discussion record, to the generated private task repository using proposal-agent's upload format.
```

Keep the local session history until it is uploaded. After the agent reports
the private repository and upload
commit SHA, clone that repository on a machine with the required GPU resources
and follow its README:

1. Invoke the bundled validator skill to perform real environment validation:

   ```text
   $harbor-task-validator ./<task-name>
   ```

2. After validation passes, invoke the bundled runner skill to run, monitor,
   and recover the task through `RSI-Harness`:

   ```text
   $rsi-task-runner
   ```

3. Upload the separate completed experiment trajectory to the private task
   repository.

> [!CAUTION]
> We review every task for scientific soundness, rigor, and novelty, and independently reproduce submitted tasks. Proposal approval and trajectory submission do not guarantee inclusion. A task with an unreasonable, unsupported, or irreproducible design may still be rejected after trajectory submission and excluded from RSIs-First-Exam.

### 5. Share feedback

Please leave brief feedback on `proposal-agent` and the overall RSI-Anything pipeline in [this Discussion](https://github.com/RSI-Index/RSIs-First-Exam/discussions/7).

After pushing the completed experiment trajectory, your task is ready for review. Thank you for contributing to RSIs-First-Exam! 🙏
