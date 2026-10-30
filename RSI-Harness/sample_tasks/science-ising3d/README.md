# science-ising3d

## Overview

An open-ended science task with a public, model-access Judge: derive the
zero-field thermodynamics of the simple-cubic Ising ferromagnet with
controlled errors. Work reads every rubric and the whole Judge at `/tests`;
each `rsi-submit` is recomputed in a Judge started from the clean task image,
audited by the run's own Agent, reviewed rubric by rubric by the Judge model,
both through `rsi-agent`, and rewarded with the number of rubrics the
candidate passes.

It is an RSI Harness demonstration of `visibility = "public"`,
`model_access = true` and `judge_model`. Its Judge is the generic public
science Judge library, `tests/science_judge.py`, copied unchanged from
OpenRSI-X's `science-task-agent` skill (`assets/science_judge.py`), plus this
task's configuration in `tests/judge.py`; the end-to-end tests in RSI Harness
therefore exercise the same Judge code that tasks authored with that skill
use.

## Scientific target and rubrics

The target (`tests/science-contract.yaml`) asks for the infinite-volume
pressure, energy and heat capacity for all finite K>=0, high/low-temperature
asymptotics, Kc from the same construction and the leading singularities on
both critical sides. 26 rubrics (`tests/rubrics.json`, R01–R26) each state one
separately judgeable result; `tests/coverage.md` maps the claims to rubrics,
records the conventions chosen and the sources. All 26 passing on one
candidate completes the task; the reward is the literal count. Launch
blockers: none. The contract stays `design_draft` with `launch_ready: false`
until execution validation with a real Agent passes.

## Environment and Judge

- `WORKDIR` `/workspace`; the candidate is `/workspace/candidate`, at most 512
  regular files and 32 MiB, no symlinks. The image (Python 3.11, standard
  library only, pinned by digest) places the starter there and at
  `/opt/science/starter`.
- Routes, each offline on its own copy of the frozen candidate with rlimits,
  `no_new_privs` and a seccomp filter that denies internet sockets:
  `candidate_reproduction` runs `probe.py`, `public_pressure_probes` probes
  `model.py:pressure(K)`; 60 s each. No domain checker. A root Judge (local
  Docker) runs each as its own uid (64001, 64002); a Judge that is not root
  (a cluster) confines each with Landlock, or else user namespaces (`userns`).
- Model access through `rsi-agent`: one integrity audit by the run's Agent
  (deadline 1200 s), then one plain call to the Judge model (`gpt-6-astra`,
  medium effort) per rubric, four at a time, under one shared 1800 s review
  deadline. The Judge procedure never calls a model API or an Agent CLI and
  sees no credential.
- Resources: 4 CPUs, 8 GiB shared by Work and the Judge, no GPUs; Agent
  timeout 6 hours; Verifier timeout 3600 s (2 × 60 + 1200 + 1800 = 3120 s plus
  slack).
- Judge library: `science_judge.__version__` `"9.4"`; `tests/sources.json`
  cites the external facts the rubrics use.

Outputs per round in `/logs/verifier`: `bundle.json`, `integrity.json`,
`review.json`, `reward.json` (`{"reward": <passes>}`), or `review-error.json`
when no reward is written, plus `model/<job>/` records and `timing.json`.

## Known limitations

- The Agent's own tools and sandbox decide what else it does with the
  Judge's network view during the audit, which reaches only the pinned model
  endpoints (and, when the Agent uses one, its forward proxy).
- Automatic model review can misjudge subtle mathematics, evidence
  independence and reference applicability; it is not a certificate.
- Still to verify with a real model and root: review quality and duration
  inside the Judge, a local login shared by parallel review calls, the real
  egress firewall, and a long run with reviews of up to an hour each.

## Run and validate

```bash
sudo -E rsi-harness run /absolute/path/to/sample_tasks/science-ising3d \
  --agent "$RSI_AGENT" \
  --model "$RSI_MODEL" \
  --reasoning-effort "$RSI_REASONING_EFFORT" \
  --primary-reward reward \
  --score-direction maximize \
  --max-submissions 10 \
  --verbose
```

`$RSI_AGENT` is any Agent the Harness supports that can run one-shot in a
Judge; the audit uses that same Agent, model and effort. The reviews need a
Judge model credential: `--judge-llm-auth`, or what a Codex Agent lends (see
the Harness README). Ten submissions is the operator's choice for a six-hour
run whose reviews can each take up to an hour. The run needs an Agent API base
URL or proxy that the Harness can pin for the Judge.

Static validation with the skill's validator, and the Harness tests:

```bash
python3 <science-task-agent>/scripts/validate_task.py \
  /absolute/path/to/sample_tasks/science-ising3d \
  --harness-root /absolute/path/to/RSI-Harness
pytest tests/acceptance/test_science_sample.py
RSI_RUN_SCIENCE=1 pytest tests/integration/test_science_in_judge_sample.py
```

The integration test runs the Judge in a container shaped like a public Judge
(root, read-only root filesystem, `/tmp` tmpfs, the real `rsi-agent`) with a
fake Codex and a fake Claude Code CLI reaching a fake model, and a full
scripted run (25 then 26 rubrics) in which no Judge sees what the Agent
changed in Work outside `/workspace`.
