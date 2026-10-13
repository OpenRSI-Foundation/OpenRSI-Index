---
name: task-intake
description: Use when a contributor starts a new OpenRSI-Index task proposal; asks the task domain and whether the task involves model training, then hands both answers to the matching proposal skill.
---

# Task intake

This is the entry point for every new OpenRSI-Index task proposal. It asks two
questions, one at a time, and then hands the contributor to the proposal skill
for that kind of task. It does not research repositories, draft a proposal, or
create a Discussion.

Before the first response, read [references/task-types.md](references/task-types.md)
completely. It lists the domains, the routes, and the rows every proposal skill
records.

## How to ask

- Write each question directly in the normal chat response, in the
  contributor's language. Do not use structured question widgets or forms.
- Ask exactly one question per response, then end the turn and wait for the
  contributor's reply. Do not treat silence or a timeout as an answer.
- Do not infer either answer from the research idea or from pasted material,
  and do not recommend an answer. Explain the options neutrally; the contributor
  decides. An explicit answer the contributor already gave in this session
  counts, so do not ask it again.

## Question 1: domain

Briefly say that a proposal starts with two short questions. List every domain
in the Domains table with its display name and description, and ask which one
describes the task. Accept exactly one listed domain. If the answer matches none
of them, say which domains are available and ask again; never add a domain.

## Question 2: model training

Ask whether the task involves model training, with Yes or No, and explain the
meaning given in [references/task-types.md](references/task-types.md). Accept
only Yes or No.

## Hand-off

1. Find the next skill in the Routes table for the two answers.
2. Check that the next skill exists in the same skills collection: the
   directory that contains this `task-intake` directory must also contain
   `<next skill>/SKILL.md`.
3. If it is missing, tell the contributor plainly that this route is not
   available yet, repeat their two answers, and stop. Do not fall back to
   another proposal skill, draft a proposal, or create a Discussion. The
   contributor may correct an answer if it was a mistake; route again only on
   a corrected answer.
4. Otherwise, tell the contributor in one sentence which skill comes next, then
   read that skill's `SKILL.md` and follow it in this same session without
   waiting for another message. Use one session per proposal: the next skill's
   session record includes this intake.

Hand over both answers in this form, and treat them as contributor-confirmed:

```text
Task intake: Domain = <ID>; Involves model training = <Yes or No>
```

The next skill does not ask these questions again. It records both answers as
the first two rows of its proposal table, exactly as shown under "Recorded rows"
in [references/task-types.md](references/task-types.md). This applies to
`proposal-agent` and `science-proposal-agent` alike, and to any skill added for
a new domain.

If the contributor later corrects an answer, route again. When the route is
unchanged, the current skill updates its two rows; when it changes, stop the
current skill and start the new one from its beginning.
