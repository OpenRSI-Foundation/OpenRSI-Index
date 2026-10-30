# Sample tasks

| Task | What it shows |
| --- | --- |
| [`harbor-in-judge`](harbor-in-judge) | A Judge whose fixed procedure runs Harbor over a Terminal-Bench 2 subset, a Compose sidecar and a Dockerfile-built task, each environment through the sandbox broker |
| [`vllm-in-judge`](vllm-in-judge) | Work leaves a model checkpoint; the Judge serves it with vLLM on its GPU and runs an LLM agent over Harbor tasks in brokered environments |
| [`swebench-in-judge`](swebench-in-judge) | Three SWE-bench Verified tasks run offline through the sandbox broker |
| [`science-ising3d`](science-ising3d) | A public Judge for open-ended science: from the clean task image, the Judge re-executes the candidate isolated, the Agent model audits the trajectory and the Judge model reviews 26 rubrics in parallel; the reward is the number this candidate passes |
| [`huggingface__transformers…test_serve…lv1`](huggingface__transformers.e2e8dbed.test_serve.4e7860c7.lv1) | GPU research task: implement `transformers` OpenAI-compatible model serving (2 GPUs) |
| [`linkedin__liger-kernel…test_fused_neighborhood_attention…lv2`](linkedin__liger-kernel.c856fbab.test_fused_neighborhood_attention.78217be4.lv2) | GPU research task: implement `liger-kernel` fused neighborhood attention (2 GPUs) |
| [`linkedin__liger-kernel…test_poly_norm…lv1`](linkedin__liger-kernel.c856fbab.test_poly_norm.7b0e3399.lv1) | GPU research task: implement `liger-kernel` polynomial normalization (2 GPUs) |

The `*-in-judge` samples carry the operator policy that approves them (`operator-policy.toml`) and, where images must be pre-pulled, an
`images.manifest`; see [the operator guide](../docs/sandbox-operator-guide.md).

## `science-ising3d`

The zero-field 3D Ising thermodynamics task with 26 public scientific rubrics,
run by the standard `rsi-harness run`; [its README](science-ising3d/README.md)
has the details. Its `task.toml` uses every public-Judge feature:

- `visibility = "public"`: Work reads the rubrics, the review instructions and
  the Judge itself at `/tests`, the same bytes every Judge runs. Every Judge
  starts from the clean image Work started from (the task image plus the
  Agent CLI), read-only, and reads only Work's `/workspace`, so nothing the
  Agent installs or changes in Work, such as the Python interpreter or the
  Agent CLI, reaches the programs that judge it.
- The Judge is the public science Judge library, `tests/science_judge.py`,
  copied unchanged from OpenRSI-X's `science-task-agent` skill, configured by
  `tests/judge.py`. It freezes the candidate and re-executes its optional
  `probe.py` and `model.py` probes, each on its own copy with rlimits, a
  minimal environment, a seccomp filter that denies internet sockets, and a
  60-second bound: as its own unprivileged user under a root Judge, with
  Landlock or user namespaces under a cluster Judge. Work can preview this
  step with `python3 /tests/judge.py --recompute /workspace/candidate`.
- `model_access = true`: the Judge audits the submission and the trajectory at
  `/logs/agent` with the run's own Agent, in one `rsi-agent exec`, which runs
  the Agent (Codex, Claude Code, or any Agent the Harness supports) with
  Work's credentials and settings, isolated from the Judge. The Agent reads a
  read-only copy of the candidate and the recompute record, and the
  trajectory; it cannot write the Judge's outputs.
- `judge_model`: each rubric is then reviewed in a separate `rsi-agent llm`
  call to the Judge model, four at a time under one shared deadline, with the
  evidence in its prompt and no tools. The Judge itself never calls a model
  and never sees a credential.
- The reward is the number of rubrics this one candidate passes, written only
  after a clear audit and a complete, valid review; anything else is a Verifier
  error, not a score.

Run it as root with the Agent's model and credential, for example
`sudo -E "$(command -v rsi-harness)" run sample_tasks/science-ising3d --agent
codex --model MODEL --reasoning-effort EFFORT`. The audit uses the same Agent,
model and effort; the reviews use the Judge model, with a Codex Agent's
credential or `--judge-llm-auth`. No operator policy is needed: nothing is
brokered.

It replaces OpenRSI-X's RSI-Science layer. The separate science CLI, its
supervisor epochs, workspace checkpoints, the host-side reviewer and the
Harness result-processor hook are gone: one run is one Agent session within
the task's time and submission limits, the reviewer is the Judge in the clean
task image, `capture_trajectory` is the Harness trajectory at `/logs/agent`,
and `completed_rubrics` is the reward. Three behaviours differ: an integrity
violation now invalidates only its round instead of stopping the campaign, a
run does not stop early when one candidate passes all 26, and there is no
cross-process resume beyond `rsi-harness recover`. Results recorded under the
RSI-Science protocol are not results of this task.

`tests/acceptance/test_science_sample.py` runs whole Judge rounds of the
sample without root or Docker, through the real `rsi-agent` with a fake Codex
CLI and the Harness's own trajectory capture, and checks that the library
agrees with what RSI Harness gives a Judge. Without root,
`RSI_RUN_SCIENCE=1 pytest tests/integration/test_science_in_judge_sample.py`
runs the Judge in a container shaped like a public Judge (root, read-only root
filesystem, /tmp tmpfs, the real `rsi-agent`; once with a fake Codex CLI and
once with a fake Claude Code CLI, each reaching a fake model at the Agent's
base URL), and a full scripted run (25 then 26 rubrics, and neither Judge sees
what the Agent changed in Work outside `/workspace`). Still to verify with a
real model and root:

- the review quality and duration of the real Agent inside the Judge;
- a local login shared by the parallel review calls;
- the real egress firewall: the Judge reaches only the pinned model
  endpoints, which the tests only fake;
- a long run with reviews of up to an hour each.
