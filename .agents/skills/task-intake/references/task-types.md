# Task types

This file is the single list of task domains and their routes. `task-intake`
reads it before its first question, and `proposal-agent` reads it when started
directly.

## Domains

Ask the contributor to choose exactly one domain from this table. Show the
display name and description of each entry; record the ID.

| ID | Display name | Discussion label | Description |
| --- | --- | --- | --- |
| AI4AI | AI for AI | general task | Improving how AI models are built, trained, served, or evaluated, such as model architectures, training recipes, data pipelines, inference systems, and kernels. |
| AI4Sci | AI for Science | science task | Using AI to advance a scientific field outside AI itself, such as physics, chemistry, biology, materials, climate, or mathematics. |

The Discussion label is applied by the OpenRSI-Index review workflow from the
recorded Domain row. Skills never apply labels themselves.

## Routes

The second question is: does the task involve model training? Answer **Yes**
when one candidate experiment trains or fine-tunes model weights, in Work or as
part of evaluation. Answer **No** when it does not, for example when it changes
an algorithm, simulator, solver, or pipeline around a fixed pre-trained model.

| Domain | Involves model training | Next skill |
| --- | --- | --- |
| AI4AI | Yes | proposal-agent |
| AI4AI | No | proposal-agent |
| AI4Sci | Yes | proposal-agent |
| AI4Sci | No | science-proposal-agent |

`proposal-agent` uses the general pipeline: AI4AI rubric review, then automatic
task building. An AI4Sci task that trains a model uses this pipeline too; it
still receives the `science task` label.

`science-proposal-agent` is the science pipeline's proposal skill. It is not
part of every skills collection yet; `task-intake` checks for it before handing
over and stops when it is absent.

## Recorded rows

Every proposal skill reached through these routes records both answers as the
first two rows of its `Section | Field | Proposal` table, before any other row,
exactly as:

```text
| Task Type | Domain | <ID> |
| Task Type | Involves model training | <Yes or No> |
```

For example:

```text
| Task Type | Domain | AI4Sci |
| Task Type | Involves model training | No |
```

Write the ID exactly as listed above and `Yes` or `No` exactly; add no other
text, formatting, or explanation to these cells. OpenRSI-Index automation reads
these rows: the Domain row alone decides the Discussion label (a missing or
unknown domain is labelled `general task`), and a proposal recorded as AI4Sci
without model training is left to the science pipeline instead of the AI4AI
rubric review and task building.

## Adding a domain

To add a vertical, add one row to the Domains table and one row per training
answer to the Routes table. A new domain may name its own Discussion label and
its own next skill. OpenRSI-Index then needs matching changes:

- **Label:** its domain-to-label mapping must list the same ID and label, and a
  maintainer must create the label, before proposals in the new domain are
  labelled; until then they are labelled `general task`.
- **Pipeline:** its pipeline routing sends every answer pair to the general
  pipeline (AI4AI rubric review and automatic task building) unless listed
  otherwise. Each new route whose next skill is not `proposal-agent` needs its
  own entry there, or its proposals still get the general review. Its contract
  test checks every route here against that routing; a route to a third
  pipeline also needs that test and the review workflow extended.
