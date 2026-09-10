# Discussion lifecycle and trajectory

After reading the mandatory Round 0 references, first run `gh --version` and `gh auth status --hostname github.com`. If `gh` is missing, help install it for the contributor's system. If signed out, run `gh auth login --hostname github.com --web` and let the contributor complete browser authorization. Reuse an existing login; do not request a pasted token. Check Git and Python availability for the helper. If Git needs GitHub credentials, use `gh auth setup-git --hostname github.com`.

Locate the contributor's local `RSIs-First-Exam` checkout from their supplied path or the accessible workspace. If it is not clear which checkout to use, ask for its path. The client need not start in that directory; use its absolute path as `<public-root>` below, including for the helper itself. If the client cannot access the checkout, ask the contributor to open or attach that folder.

Use one native Codex or Claude Code session for one proposal. Keep that session for feedback, revisions, contributor-owned `/task <answer or correction>` responses, and final upload; a closed client may resume it. Use a separate checkout for a second proposal. The client records its own trajectory: no hook setup, activation, early binding, or frozen snapshot is needed.

After the contributor confirms the final rendered proposal, write and reread it under the Round 6 file rules, then directly create the Task Ideas Discussion from that confirmed file and title. The confirmation authorizes this handoff; do not ask a second permission question before running:

```text
python3 "<public-root>/.agents/skills/proposal-agent/scripts/proposal_session.py" discussion create --checkout "<public-root>" --proposal "<proposal-path>" --title "<confirmed-title>"
```

Continue in the same session. This agent is not a background monitor: GitHub cannot wake it. On every later or resumed turn after Discussion creation, before deciding what to do, first fetch the current Discussion record:

```text
python3 "<public-root>/.agents/skills/proposal-agent/scripts/proposal_session.py" discussion status --checkout "<public-root>"
```

Treat the Discussion as a serialized workflow: allow only one Discussion mutation at a time. After `discussion create`, `discussion update`, one `/task ...` reply, or `/reset`, wait until `discussion status` reports the corresponding completed result before editing or posting again. A normal proposal pass starts task preparation automatically; do not edit or post `/task` merely to start, hurry, or retry it. A successful `/reset` deletes only the unpublished private task State, preserves every Discussion comment including `/reset`, and posts an explicit reset-complete reply. Only after that reply, one plain `/task` may start a fresh assumptions pass from the accepted proposal.

When feedback arrives, use exactly one path. If it requires a scientific-contract change, route it back through the owning round and obtain the contributor's confirmation instead of silently changing a confirmed decision. Rewrite and reread the same proposal file under the Round 6 overwrite rule, then update the same Discussion rather than creating another one:

```text
python3 "<public-root>/.agents/skills/proposal-agent/scripts/proposal_session.py" discussion update --checkout "<public-root>" --proposal "<proposal-path>"
```

If task preparation instead asks a genuine task-defining question that does not require changing the proposal body, help the contributor resolve it in this same session and post their exact contributor-owned answer as one `/task <answer or correction>`. Never update the proposal and post `/task` for the same feedback. If GitHub, the runner, Codex, or another transport or service fails, report the failure and wait for the contributor; do not autonomously retry by editing the Discussion or posting another command.

At ACCEPTED, tell the contributor:

> Keep this agent session to address any feedback or revisions that may arise.
> Once the private task repository is ready, return here and ask me to upload
> this session's trajectory and Discussion record.

After the contributor requests upload and `discussion status` shows the verified private task repository is ready, locate this session's original native JSONL:

- **Codex:** use the current session identity exposed by the client (for example `CODEX_THREAD_ID` in its shell), then locate its log under the client's Codex home, normally `~/.codex/sessions/`; archived logs may be under `archived_sessions/`. Match the session ID with the log's `session_meta` record.
- **Claude Code:** use the current session ID from the client or native skill context (`${CLAUDE_SESSION_ID}` when expanded by Claude Code). Logs normally live at `~/.claude/projects/<project>/<session-id>.jsonl`. Match the log's `sessionId`. The project may reflect the directory where the client started, not the Public checkout.

Respect a custom client storage location. If the client has not exposed the ID, locate the original log using this conversation's distinctive content and native metadata; do not simply choose the newest file. If no unambiguous original log is available, ask for the session ID or native log path. Keep the client history available until upload; hooks are not needed to record it.

Pass that original file directly to the helper. Do not summarize, reconstruct, filter, or rewrite it:

```text
python3 "<public-root>/.agents/skills/proposal-agent/scripts/proposal_session.py" upload --checkout "<public-root>" --platform <codex-or-claude-code> --session-id "<native-session-id>" --transcript "<absolute-native-jsonl-path>"
```

The helper fetches the complete current Discussion (including all comment and reply pages) and pushes exactly `proposal-trajectory/<platform>-session.jsonl`, `proposal-trajectory/discussion.md`, and `proposal-trajectory/metadata.json` to the linked private task repository. It copies the native log bytes as they exist at upload time; later uploads update the same paths. Discussion submission and updates do not require locating a transcript first.

Do not finish the contributor workflow until `upload` reports both the committed private repository and its commit SHA. Report both values to the contributor, then direct them to clone that repository on a machine with the GPU resources declared by the task, follow its README to validate and run the task, and push the completed experiment trajectory.
