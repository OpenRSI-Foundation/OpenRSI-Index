# Zero-field 3D Ising thermodynamics

Develop a unified, error-controlled construction of the zero-field thermodynamics of the simple-cubic Ising ferromagnet, from which its transition coupling and leading critical laws follow.

## Objective and scope

The model is the uniform nearest-neighbor ferromagnetic Ising model on the simple-cubic lattice at zero field. Determine the infinite-volume pressure p(K), energy/J=-p'(K) and heat capacity/k_B=K^2 p''(K) for all finite K>=0, with high/low-temperature asymptotics, a transition coupling Kc derived from the same construction, and the leading singularities on both critical sides. The exact conventions and allowed premises are in `/tests/science-contract.yaml`. Nonzero fields, all correlation functions and a full conformal spectrum are outside scope; passing this scope does not establish every problem concerning 3D Ising models.

There are 26 public scientific claims in `/tests/rubrics.json`, each assessed independently: construction, model and limit; pressure, energy and heat capacity over each ordinary phase; their high/low-temperature endpoints; Kc; critical pressure and energy values; each leading observable law on each critical side; coherent branches; and independent scientific reproduction. A correct endpoint expansion is useful progress but does not establish the intervening phase, and a matching pressure curve does not automatically validate its derivatives.

## Workspace and candidate

| Item | Value |
| --- | --- |
| Working directory | `/workspace` |
| Candidate | `/workspace/candidate`, at most 512 files and 32 MiB, no symlinks |
| Starting material | `/opt/science/starter` |

Work starts with a copy of the starter in `/workspace/candidate`. The starter is explicitly insufficient: replace or extend it, start from scratch, or continue inherited research. If the candidate directory is ever removed, restore it from `/opt/science/starter`.

## Evidence and methods

Keep `/workspace/candidate/theory.md` as the main argument, with supporting derivations, code, data, sources and reproduction instructions in the candidate. Choose any supported scientific method; a fully formalized exact proof is not required. Ordinary mathematical reasoning and controlled approximation evidence are accepted.

Explain the actual model-to-construction chain, essential premises and applicability, quantitative errors and limits, parameter provenance, scientific predictions, critical regular/singular separation and evidence locations. Trace finite-size, truncation, arithmetic and sampling errors where relevant. Whole-phase claims need justified quantitative accuracy and refinement; asymptotic claims need remainders that actually isolate their leading terms. Accuracy or a remainder for pressure alone does not establish energy or heat-capacity accuracy. Missing critical steps stay unresolved even when familiar exponents or constants have been inserted.

Public research, published mathematics, prior valid results and local calculations are permitted. Importing a target answer as an input does not derive that answer, and calibration data cannot independently validate their own fit. Independent corroboration may be mathematical or computational; no unseen dataset, holdout or new measurement is required. Present counterevidence and honest limitations. Listing references, reporting success or widening uncertainty earns no claim.

Two optional candidate files are recomputed by the Judge:

- `model.py` may expose `pressure(K)` for finite floating-point pressure and derivative probes. Its finite probes cannot establish whole-phase or critical laws and need not make divergent critical quantities finite.
- `probe.py` may reproduce a meaningful bounded calculation and print its result. It runs without network or prior state; include every file it needs in the candidate. State the reproduced claim, input, precision and result; printing a pass assertion alone is insufficient.

## Judge procedure

The complete Judge is public and read-only at `/tests`; the rubrics are in `/tests/rubrics.json`. The Judge runs from the clean task image Work started from, not from Work's system, so tools or packages installed in Work are unavailable to it; it reads only `/workspace`. Each submission is recomputed offline by these routes, each on its own copy of the frozen candidate as its own unprivileged user, with no internet sockets and no credentials:

| Route | Required file | Timeout |
| --- | --- | --- |
| `candidate_reproduction` | `/workspace/candidate/probe.py` | 60 s |
| `public_pressure_probes` | `/workspace/candidate/model.py` | 60 s |

Preview the recompute in Work with
`python3 /tests/judge.py --recompute /workspace/candidate`.
It runs no audit or review and writes no reward.

Then the run's own Agent, called inside the Judge through `rsi-agent`, audits the round (20 minutes) and reviews each rubric in its own call, four at a time under one 30-minute deadline. Reviewers read the frozen candidate, the recompute record and your trajectory; they may reason and calculate independently but never execute candidate code and do not search the web. The Judge library (`/tests/science_judge.py`) and this task's configuration (`/tests/judge.py`) are public.

## Integrity

Before scientific judgment, an automatic audit under `/tests/integrity.md` checks every submission and the trajectory captured up to it. Only `clear` lets the round be counted; a violation, an unresolved audit or a failed audit is a Judge error with no reward, not a scientific zero. Do not tamper with task-owned assets, fabricate Judge-like evidence, misrepresent captured results, or place reviewer-directed instructions in evidence. Reading the public rules, feedback and references, continuing valid inherited results, transparent approximations and candid proof gaps are permitted.

## Submission and feedback

Submit with `rsi-submit`; a review can take up to an hour, and Work is paused meanwhile. The reward is the number of rubrics this candidate passes after a clear audit; all 26 on one candidate completes the task, and a count is not a percentage solved. Feedback lists the recompute diagnostics, the audit and every rubric's verdict (`pass`, `fail` or `unresolved`), reason, evidence and next step. Revise the substantive argument or evidence before submitting again; do not resubmit unchanged work for a different count, self-award passes, or union verdicts across submissions. Judge errors are execution failures, not scientific conclusions. No iteration requires human approval.

Make an initial coherent submission early, then address the uncovered domain, derivative or error issue, critical dependency or reproduction mismatch named in its feedback. Keep `/workspace/research_summary.md` (outside the candidate) up to date with the arguments, calculations, submissions and verdicts, revisions, established results and remaining gaps. Claim scoped completion only when one valid submission passed all 26.

## Submission checklist

- `theory.md` and every supporting file are complete and flushed under `/workspace/candidate`.
- The recompute preview runs and its output matches what `theory.md` reports.
- The next submission tests a stated hypothesis or closes a gap named in the last feedback.
