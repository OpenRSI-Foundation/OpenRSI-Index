# Discussion lifecycle and trajectory

Use one fresh Codex or Claude Code session started at the Public repository root for exactly one proposal in that checkout. Keep that same session for proposal feedback, revisions, contributor-owned `/task <answer or correction>` responses, and final upload. A closed client may resume the same native session; a replacement session or a second proposal requires a separate checkout.

After reading the mandatory Round 0 references, the first Round 0 action is to activate native session capture before the first contributor-facing response or repository research:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py activate --checkout <public-root>
```

After the contributor confirms the final rendered proposal, write and reread it under the Round 6 file rules, then directly create the Task Ideas Discussion from that confirmed file and title. The confirmation authorizes this handoff; do not ask a second permission question before running:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py discussion create --checkout <public-root> --proposal <proposal-path> --title "<confirmed-title>"
```

Continue in the same session. This agent is not a background monitor: GitHub cannot wake it. On every later or resumed turn after Discussion creation, before deciding what to do, first fetch the current Discussion record:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py discussion status --checkout <public-root>
```

Treat the Discussion as a serialized workflow: allow only one Discussion mutation at a time. After `discussion create`, `discussion update`, one `/task ...` reply, or `/reset`, wait until `discussion status` reports the corresponding completed result before editing or posting again. A normal proposal pass starts task preparation automatically; do not edit or post `/task` merely to start, hurry, or retry it. A successful `/reset` deletes only the unpublished private task State, preserves every Discussion comment including `/reset`, and posts an explicit reset-complete reply. Only after that reply, one plain `/task` may start a fresh assumptions pass from the accepted proposal.

If any Discussion or upload command reports missing current-turn proof or a session mismatch, do not work around it. Explain that repository hooks must remain enabled, ask the contributor to resume the original bound Codex or Claude Code session, and retry there.

When feedback arrives, use exactly one path. If it requires a scientific-contract change, route it back through the owning round and obtain the contributor's confirmation instead of silently changing a confirmed decision. Rewrite and reread the same proposal file under the Round 6 overwrite rule, then update the same Discussion rather than creating another one:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py discussion update --checkout <public-root> --proposal <proposal-path>
```

If task preparation instead asks a genuine task-defining question that does not require changing the proposal body, help the contributor resolve it in this same session and post their exact contributor-owned answer as one `/task <answer or correction>`. Never update the proposal and post `/task` for the same feedback. If GitHub, the runner, Codex, or another transport or service fails, report the failure and wait for the contributor; do not autonomously retry by editing the Discussion or posting another command.

After `discussion status` shows the verified private task repository is ready, upload the native session trajectory and complete Discussion record:

```text
python3 .agents/skills/proposal-agent/scripts/proposal_session.py upload --checkout <public-root>
```

Do not finish the contributor workflow until `upload` reports both the committed private repository and its commit SHA. Report both values to the contributor, then direct them to clone that repository on a machine with the GPU resources declared by the task, follow its README to validate and run the task, and push the completed experiment trajectory.
