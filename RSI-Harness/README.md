# RSI Harness

RSI Harness powers OpenRSI Index for ultra-long-horizon RSI runs, natively
supporting [Harbor-format tasks](../rsi-tasks/) with or without GPUs on local
Docker, and GPU tasks on multi-node clusters.

Agents such as Claude Code, Codex, Qwen Code, Kimi Code, OpenCode, ZCode and
pi develop solutions in
a persistent, isolated Work container and submit them to a fresh Judge for
scoring. They can iterate on test feedback until the submission limit or
timeout, with the highest score retained as the final result.

RSI Harness runs the **outer loop** only: preparing the task, pausing and
snapshotting Work, scoring each submission in a fresh Judge, and selecting the
final result. The **inner loop** is whichever agent you put in it — Claude Code,
Codex, Qwen Code, Kimi Code, OpenCode, ZCode, pi, or an agent you design
yourself.


## How it works

A run starts from a supported Harbor task directory containing `task.toml`,
`instruction.md`, `environment/`, and `tests/test.sh`. RSI Harness validates the
task, builds or resolves its container image, and records its resources, network
policy, and timeouts in a run plan. It then follows this lifecycle:

1. **Work:** RSI Harness starts a persistent, isolated Work container. The Agent
   edits the task workspace here, but the task's private `tests/` directory is
   never mounted into this container (unless the task declares a public Judge,
   below).
2. **Submit:** When the Agent runs `rsi-submit`, the client sends a token-scoped
   control request. RSI Harness pauses Work and snapshots its filesystem state;
   the client does not package or upload the workspace.
3. **Judge:** RSI Harness starts a fresh, isolated Judge container from the
   clean image Work started from, with the snapshot of Work's `WORKDIR`
   mounted read-only (or, for a `WORKDIR` of `/` or
   [`clean_image = false`](docs/harbor-task-authoring/task-toml.md#clean_image),
   from the whole Work snapshot), injects the private tests, runs
   `/tests/test.sh`, records the test output and reward, and removes the Judge
   container. The Judge can also read the Agent's trajectory up to this
   submission, read-only, at `/logs/agent`
   ([details](docs/harbor-task-authoring/task-toml.md#agent-trajectory-in-the-judge)).
4. **Iterate or finish:** RSI Harness resumes the same Work container and returns
   the Judge feedback to the Agent. The cycle repeats until the submission limit
   or timeout. Among valid submissions, the snapshot with the best primary
   reward, according to the configured score direction, becomes the final
   result.

Work persists across rounds, while each submission is evaluated in a fresh
Judge, letting the Agent improve from feedback without access to the private
tests.

A task can instead make its Judge public, for example when the tests are a
published rubric or review procedure:

```toml
[metadata.rsi_harness.verifier]
visibility = "public"
```

Work then also sees the same `tests/` directory read-only at `/tests`. The run
plan records its digest, and RSI Harness checks it before Work starts and
before every Judge round, so the Agent reads exactly what the Judge runs. A
public Judge is also stricter than a hidden one: it has a read-only root
filesystem and a small `noexec` `/tmp`. The default, `"hidden"`, keeps the
tests private. See [`task-toml.md`](docs/harbor-task-authoring/task-toml.md#visibility).

A Judge can also call the run's own Agent, for example to review a
submission, when the task declares `model_access = true` in the same table.
`rsi-agent exec --prompt-file FILE --output FILE [--schema FILE]` is then the
only supported way for the Judge to call a model: it runs the run's Agent,
with Work's credentials and settings, on one prompt, isolated (its own
unprivileged user in a root Judge, Landlock otherwise), and the Judge
procedure never sees the credentials. Locally the Judge reaches only the
Agent's pinned model endpoints; on a cluster it uses the network the Work
Agent uses for its model. See
[`task-toml.md`](docs/harbor-task-authoring/task-toml.md#model_access).

A task can instead, or also, name a Judge model with `judge_model = "<id>"`:
`rsi-agent llm` then makes one plain call to that model, independent of the
Agent, with a credential the operator gives through `--judge-llm-auth` (see
[Common `run` options](#common-run-options)). See
[`task-toml.md`](docs/harbor-task-authoring/task-toml.md#judge_model).

Every Judge also has `rsi-isolate`, which runs one command, such as a
candidate's code, as an isolated unprivileged identity: its own uid in a
root Judge (local Docker), Landlock otherwise (clusters), with only the
paths, network and environment it is granted. A Judge therefore never
switches users itself. See
[`task-toml.md`](docs/harbor-task-authoring/task-toml.md#rsi-isolate).

[`sample_tasks/science-ising3d`](sample_tasks/science-ising3d) combines these
for open-ended science: a public rubric, a Judge started from the clean task
image, and a trajectory audit and per-rubric review by the Agent model. Its
copy of the science Judge library is an older version that needs a root
Judge (local Docker); a task meant for a cluster copies the current library
from OpenRSI-X's `science-task-agent` skill.

![RSI Harness workflow: one-time Harbor task preparation starts an agent in a persistent Work container. Each rsi-submit pauses and snapshots Work for a fresh, isolated Judge, which runs private tests and returns feedback. Work resumes for further iterations until the submission or time limit; the best valid submission is selected by its primary reward.](../assets/figures/rsi-harness.png)


## Requirements

- Linux with a working `docker` command
- for GPU tasks, NVIDIA GPUs, the NVIDIA Container Toolkit, and a working
  `nvidia-smi` command
- Python 3.12 or 3.13
- permission to manage Docker and the host `DOCKER-USER` and `INPUT` iptables
  chains (usually by running the command with `sudo -E`)
- enough Docker storage for the task image and temporary Judge snapshots
- a supported local Harbor task directory

## Install

Install from a checkout:

```bash
uv tool install /path/to/RSI-Harness
```

Or build and install the wheel:

```bash
uv build
uv tool install dist/rsi_harness-0.1.0-py3-none-any.whl
```

The package includes the run engine and RSI Loop integration. No custom
`PYTHONPATH` is required.

## Configure the Agent

Provide an API key:

```bash
export RSI_AGENT_API_KEY="..."
```

Or reuse an existing Codex or Claude Code login:

```bash
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task --agent codex --agent-auth local --gpus 4,5
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task --agent claude-code --agent-auth local --gpus 4,5
```

When using a compatible API proxy, also set its base URL and optionally a
model:

```bash
export RSI_AGENT_API_BASE_URL="https://your-api.example/v1"
export RSI_AGENT_MODEL="your-model"
```

These values are passed only to the Agent process; credentials are redacted
from Engine artifacts. Do not put credentials in the Harbor task directory.

`RSI_AGENT_EXTRA_ENV` passes extra Agent settings; common credentials are hidden
from logs, while ordinary settings remain unchanged. For credentials with
nonstandard names, list the variable names in `RSI_AGENT_SECRET_ENV_NAMES`
(comma-separated).

### Agents

| `--agent` | Agent | Authentication | Required settings | Reasoning effort |
| --- | --- | --- | --- | --- |
| `codex` | Codex | `RSI_AGENT_API_KEY` or `--agent-auth local` | None; the base URL defaults to the provider's public API | Yes |
| `claude-code` | Claude Code | `RSI_AGENT_API_KEY` or `--agent-auth local` | None; the base URL defaults to the provider's public API | Yes |
| `qwen-code` | Qwen Code | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | No |
| `kimi-code` | Kimi Code | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | `low`, `medium`, `high`, sent as the endpoint's `reasoning_effort` |
| `opencode` | OpenCode | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | `low`, `medium`, `high`, sent as the endpoint's `reasoning_effort` |
| `zcode` | ZCode | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | `low`, `medium`, `high`, sent as the endpoint's `reasoning_effort` |
| `pi` | pi | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | `minimal`, `low`, `medium`, `high`, sent as the endpoint's `reasoning_effort` |
| `oracle` | Oracle: the task's own `solution/solve.sh`, no model ([Validate a task's Judge with the oracle](#validate-a-tasks-judge-with-the-oracle)) | None; no credential reaches Work | `solution/solve.sh` in the task | No |

For example, pass `--agent codex --model gpt-6-astra --reasoning-effort max`
or `--agent claude-code --model claude-opus-5-5 --reasoning-effort max`
to `rsi-harness run`.

Each Agent CLI is installed at an exact pinned version, over HTTPS, into the
Work image when the image is built, together with the node it runs on. All of
these versions live in one table,
[`src/rsi_loop/harness/agent/versions.py`](src/rsi_loop/harness/agent/versions.py).
Native downloads (node, the OpenCode binary, its ripgrep and the ZCode
package) follow the image's architecture as `uname -m` reports it during the
build: `x86_64` gets the x64 release and `aarch64` the arm64 one; on any other
architecture the Work image build fails with a message naming it.
To fetch the ZCode package from a mirror with the same
`<version>/linux-<x64|arm64>/<file>` layout, set `RSI_ZCODE_MIRROR_URL` where
`rsi-harness run` runs; the Work image build receives it as a build argument.

To upgrade an Agent, or node for all of them:

```bash
uv run python scripts/bump_agent.py codex 0.160.0   # or scripts/bump-agent.sh
```

The script checks that the version is published (for node, OpenCode, ZCode
and ripgrep, both the x64 and the arm64 release), rewrites that one entry in
the table, prints the diff and runs the Agent's Docker tests (every Agent's
for `node`), exiting non-zero if they fail or cannot run. `--no-test` only
edits, and `--no-check` skips the published check (offline, or without npm).
`ripgrep`, which OpenCode's image carries, is bumped the same way and runs
OpenCode's tests. A new version gets a new Work image; cached images of other
versions are not reused. The Docker tests run each Agent's real CLI in a full
run against a local fake model that speaks chat completions, the Responses API
(Codex) and the Messages API (Claude Code). They check that one CLI process
keeps working through many stop-hook continuations, including a long stretch
with no tool use, a Judge round, a failed and an empty model reply, and that
the API key stays out of every artifact.

Codex and Claude Code keep working through their native stop hook. The stop
hooks of Qwen Code, Kimi Code and ZCode allow only a limited number of
continuations, and OpenCode has none, so RSI Harness runs them under
`rsi-agent-driver`, a small program baked into their Work image. pi's own stop
point has no limit, so it needs no driver: an extension baked into its Work
image does the same job inside pi. For these five Agents:

- **Same process, same session.** The CLI is started once. The driver talks to
  it over the Agent Client Protocol (ZCode over its own app-server protocol)
  and, whenever the Agent ends a turn, sends the stop hook's continuation text
  as the next prompt into the same session; pi's extension queues the same
  text as a follow-up user message whenever pi is about to stop. Failed or
  empty model replies are retried with a backoff. For Kimi Code, a prompt the
  CLI queued behind a turn it started by itself (for example when a background
  command finishes) is not followed by more prompts until the CLI is idle
  again. This is the stop-hook equivalent, so the run ends at the Agent timeout
  as it does for Codex and Claude Code. With `--disable-stop-hook` the Agent
  runs a single turn and the run ends when it finishes.
- **Never relaunched.** If the CLI process exits, the Agent run ends; the
  driver exits with it.
- **Same transcript.** The driver, or pi's extension, writes
  `agent_output.txt` in the same JSON-lines format as Claude Code's, so the
  visualizer, redaction and Judge trajectory work unchanged. Its first event
  names the Agent and its version, which the visualizer shows.
- **Endpoint.** `RSI_AGENT_API_BASE_URL` is an OpenAI chat-completions
  compatible base URL, including its version path (for example
  `https://your-api.example/v1`). There is no default endpoint and no local
  login.
- **Model limits.** `RSI_AGENT_CONTEXT_WINDOW` and
  `RSI_AGENT_MAX_OUTPUT_TOKENS` are the served model's real context window and
  output limit in tokens. Setup fails without them; they are never guessed,
  because a window larger than the real one makes the CLI overflow instead of
  compacting its context.
- **No other traffic.** The API key stays in the Agent's environment and is
  never written to its configuration. ZCode is the exception: it reads an
  OpenAI-compatible provider's key only from its provider file, so that one
  file is written to a private directory on the `/dev/shm` tmpfs, which a
  Judge's snapshot of Work does not include. If `/dev/shm` is not a tmpfs,
  ZCode does not start. Telemetry, update checks, model-catalog
  downloads, ZCode's endpoint routing lookups and other background model
  requests are turned off, so a
  no-network Work reaches only the pinned model endpoint; web tools are
  disabled there.
- **Local Docker only.** These five Agents need files baked into their Work
  image, which cluster images do not carry yet, so `--cluster` refuses them.

## Run a Harbor task

This mode runs a Harbor task locally with Docker on a single node. For cluster
or multi-node execution, see
[Run multi-node tasks on a cluster](#run-multi-node-tasks-on-a-cluster).

```bash
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task --gpus 0,1
```

`--gpus` accepts GPU indexes or UUIDs. Work and Judge GPU counts, images,
timeouts, and other runtime settings come from the task. Run
`rsi-harness run --help` for optional controls.

For CPU-only tasks, declare Work's count explicitly in `task.toml`:

```toml
[environment]
gpus = 0

[metadata.rsi_harness.verifier]
gpus = 0
```

The Judge declaration may be omitted; it defaults to zero. Omit `gpu_types`
and the Compose NVIDIA reservation, and run without `--gpus`:

```bash
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/cpu-task --agent codex
```

Both containers receive no GPU devices, and NVIDIA tools are not required.
CPU-only tasks reject `--gpus`; `--gpus 0` selects physical GPU index 0.
An omitted Work count remains unspecified and must inherit a Compose GPU
reservation. Explicit Work zero cannot conflict with a positive reservation.
CPU Work with a GPU Judge is supported when an explicit `--gpus` pool is
provided. GPU requirements still fail if NVIDIA devices are unavailable.

CPU-only tasks run with local Docker or single-node Slurm/Apptainer.
For Slurm, use a site profile with a CPU partition and omit `--gpus`;
no GPU allocation or NVIDIA tools are required. Mixed CPU/GPU tasks need a GPU
partition, and only GPU phases enable NVIDIA passthrough. Blue Vela still
requires Work GPUs. Multi-node CPU Work is not supported.

## Managed child sandboxes

Code in Work or Judge can create bounded child containers without access to
the Docker socket. The task requests them in `task.toml`, and a trusted
operator approves the request with a TOML policy:

```bash
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task \
  --agent codex \
  --sandbox-policy /absolute/path/to/sandbox-policy.toml
```

- **Version 1 profiles:** cached, digest-pinned images run as root inside a
  read-only, no-network, no-GPU child with bounded tmpfs scratch.
- **Version 2 brokered environments:** pulled and built images, Compose
  services, long-running execs and copies, used by stock Harbor through an
  injected plugin. Envs run on local Docker or, with
  `backend = "e2b"`, as E2B sandboxes.

Docker-backed sandboxes need `rsi-harness run` as root next to a rootful
Docker daemon; the BlueVela and Slurm backends accept only E2B-backed
environment grants (see
[Host requirements](docs/sandbox-operator-guide.md#host-requirements)).
Task authors start with
[docs/harbor-task-authoring/sandboxes.md](docs/harbor-task-authoring/sandboxes.md);
operators with [docs/sandbox-operator-guide.md](docs/sandbox-operator-guide.md)
(every grant field, the refusals, the BuildKit builder exception, the root
checks and the acceptance). Examples are in [sample_tasks](sample_tasks).

## Validate a task's Judge with the oracle

`--agent oracle` runs the task's reference solution instead of a model, on the
production path, so a task author or validator can check that the Judge
scores the baseline before any Agent experiment. Only the Agent differs: the
Work image, the Judge (the clean Work image, or a snapshot of Work for
`clean_image = false` or a WORKDIR of `/`), the firewall, the GPU plan, a
managed sandbox grant (`--sandbox-policy`), `rsi-isolate`, the trajectory,
the reward and `final_result.json` are those of every run.

```bash
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task --agent oracle \
  --gpus 0 --sandbox-policy /absolute/path/to/sandbox-policy.toml \
  --data-root /scratch/.rsi-harness --logs-root /scratch/evidence/logs
```

Omit `--gpus` for a CPU task and `--sandbox-policy` for a task without
sandboxes. Run it as root (`sudo -E`) like every run: the firewall needs it,
and without root the command stops at the firewall probe. (A Harness that
runs as non-root with a test firewall also cannot read a `reward.json` that
`test.sh` wrote as root with mode `0600`, and reports a false
`verifier_error`.) With `--max-submissions N`, pass `--timeout` of at least
the solve time plus N times the verifier's `timeout_sec`: the Agent timeout
is wall-clock and includes the Judge rounds, so a shorter one ends the oracle
before its last submission. The sandbox samples
([harbor-in-judge](sample_tasks/harbor-in-judge),
[swebench-in-judge](sample_tasks/swebench-in-judge)) ship a trivial
`solution/solve.sh` for this command. The run:

1. Stages the task's `solution/` at `/solution` in Work before the oracle
   starts (as root, then owned by the Work user).
2. Runs `rsi-oracle`, the oracle's launcher in its Work image (the task image
   with the standard Work layer; there is no Agent CLI or node). It runs
   `/solution/solve.sh` in WORKDIR by its shebang, with
   `DEBIAN_FRONTEND=noninteractive` and the Agent's exec environment (not a
   login shell, as for every Agent, so the image's `PATH` and the injected
   `rsi-sandbox` stay),
   then deletes `/solution` (or empties it, when the exec user cannot unlink
   it from `/`) so no Judge snapshot carries it.
3. Calls `rsi-submit` once, or N times with `--max-submissions N` (for
   timing). It submits even when `solve.sh` failed, so the Judge's output is
   there to read, and exits with `solve.sh`'s status.

No API key, base URL, model, `RSI_AGENT_EXTRA_ENV` value or provider endpoint
reaches Work; the proxy settings do, and a no-network Work reaches only the
submission endpoint. `solve.sh` must not read `/tests`: a hidden Judge's tests
are never in Work, and a public Judge's are mounted there read-only as for
every Agent.

Setup refuses, before any Docker work, a task without `solution/solve.sh` (a
regular file) and a Judge with `model_access` (it calls the run's Agent, and
the oracle has none). A task's `judge_model` needs `--judge-llm-auth`.
`--cluster`, `--agent-auth` and `--reasoning-effort` are refused.

The command exits 0 only when the run completed, `solve.sh` exited 0, exactly
N Judge rounds ran (1 without `--max-submissions`) and every round's status is
`completed`, whatever its score. Otherwise it prints
`Oracle check failed: <reasons>` and exits 1. Its summary names the
artifacts directory and, per round, the score, `test.sh`'s exit code and the
round's runtime.

Treat the exit code as the infrastructure gate only, not as the verdict on
the baseline. A baseline that the Judge scores 0 still exits 0, and a Judge
that by design writes no reward for an ineligible baseline gives a round with
`verifier_error` and exits 1. Judge each round from
`submissions/agent-N/report.json` (`status`, `exit_code`, `error`),
`feedback/agent-N.log` and `best_score`, compared with the reward the task
expects for its baseline. Under `logs/runs/<run_id>/<task_id>/`:

| File | Contents |
| --- | --- |
| `final_result.json` | `status` (`completed`, `no_valid_submission`, `failed` or `cancelled`), `best_score`, `best_rewards`, `best_round`, `total_rounds`, `timed_out`, `runtime_seconds`, and for every run: `agent_exit_code` (here `solve.sh`'s), `agent_runtime_seconds` (the Agent's exec) and `first_submission_seconds` (from the Agent's start to the start of the first Judge round, or `null`) |
| `submissions/agent-N/report.json` | The round's `status` (`completed`, `verifier_error`, `verifier_timeout`, `infrastructure_error` or `submission_error`), `metrics`, `score`, `exit_code` of `test.sh`, `timed_out`, `runtime_seconds` (the whole round: snapshot, Judge start, `test.sh` and reward) and `error` |
| `submissions/agent-N/test_output.txt`, `feedback/agent-N.log` | The round's complete Judge stdout and stderr |
| `verifier/agent-N/` | The round's `/logs/verifier` (`reward.json` and whatever `test.sh` wrote; root-owned under `sudo`) |
| `agent_output.txt` | `rsi-oracle`'s, `solve.sh`'s and `rsi-submit`'s output |
| `engine_error.json` | Only after an Engine or infrastructure failure: the redacted error |
| `run-plan.json` | The images, GPU plan, `clean_image` and sandbox grant used |

The oracle's Work image, and so its clean Judge image, lacks the Agent CLI
and node. A `test.sh` that relied on them by accident fails here although it
would pass with that Agent; and a pass here does not show that the target
Agent's CLI installs on the task image. A short run with that Agent covers
it: building its Work image is the run's first step.
Leave out `--cooldown`: a submission inside it is refused rather than
delayed, so fewer than N rounds run and the check fails.

## Run multi-node tasks on a cluster

The `bluevela` adapter supports LSF clusters; `slurm` uses the same image build,
Apptainer runtime, Work/Judge node pools, and transparent `torchrun` flow on Slurm.
Their packaged profiles are portable examples:
`/srv/rsi` paths and the scheduler account/group are placeholders, not a working
site configuration. For an existing installation, keep its working profile
unchanged outside this repository and pass that file with `--cluster`. Profiles
without `mount_policy` retain the legacy behavior, including single-node mounts,
cache locations, and environment values.

For a new configuration, start from the portable example:

```bash
mkdir -p "$HOME/.config/rsi-harness/profiles"
cp -n src/rsi_harness/cluster/bluevela/profile.toml \
  "$HOME/.config/rsi-harness/profiles/cluster.local.toml"
chmod 600 "$HOME/.config/rsi-harness/profiles/cluster.local.toml"
```

Edit that copy to set your storage roots, installed Apptainer/Podman/fakeroot
tools, and LSF queue/group. Keep only the data binds needed by the task. Profile
values also accept `${ENV_VAR}` references; missing variables cause a setup error.
Files ending in `.local.toml` are ignored by Git as an additional safeguard.
For Slurm, copy `src/rsi_harness/cluster/slurm/profile.toml` instead and set
`scheduler.partition`, `scheduler.account` (optional), and optionally `qos` and
`constraint`. Add the task's Work/Judge data binds to the empty lists. Both
profiles use `builder.walltime` in **HH:MM**; the Slurm adapter converts it to
minutes for `sbatch`.

Preview the resolved job without submitting it:

```bash
uv run rsi-harness run "$TASK" \
  --cluster "$HOME/.config/rsi-harness/profiles/cluster.local.toml" \
  --dry-run \
  --agent codex \
  --model gpt-6-astra \
  --reasoning-effort max \
  --agent-auth local
```

Remove `--dry-run` to submit the job. GPU counts come from the task; for
multi-node tasks, the adapter maps the Work and Judge requirements to full-node
LSF or Slurm allocations using the profile's GPUs-per-node setting. `--gpus` is
only for local runs, and `gpus = "all"` needs a numeric cluster override.

Slurm uses `sbatch` for build/run jobs, `squeue` for active status, `sacct` for the
exact allocation's final state and exit code, and `scancel` on interruption.
Accounting must be enabled. Multi-node discovery expands `SLURM_JOB_NODELIST`
with `scontrol show hostnames` and validates `SLURM_JOB_CPUS_PER_NODE`. Per-node
probes, rank workers, Judge controllers, and cleanup use `srun --overlap` so a
controller can coexist with its workers. The same ordered Work prefix and Judge
suffix remain disjoint. See the [Slurm srun reference](https://slurm.schedmd.com/srun.html)
for step resource sharing.

Like Blue Vela, this transport requires shared run/image/log paths, the same
host Python environment on every node, Apptainer, and InfiniBand for multi-node
runs. Each node is probed for allocated GPUs, image digest, shared storage,
InfiniBand, and scratch space before the Agent starts. Site scheduler binaries
and remote launch arguments can be overridden in the profile.

The portable example explicitly selects `apptainer.mount_policy = "scoped"`.
Under that policy, both single-node and multi-node runs use
`apptainer.work_binds` and `apptainer.judge_binds`. Each entry names a host
`source`, a container `target`, and a `read_only` setting (default `true`). The
sample data mappings retain the logical `/rsi-data` paths used by tasks; changing
a host location does not require editing task code. Judge data should be listed
only in `judge_binds`.

Enabling `mount_policy = "scoped"` on an existing installation is a separate,
explicit migration. Remove nonempty `extra_binds` (rejected in scoped mode) and
replace them with the specific phase data mappings above. Check the resolved
profile and smoke-test the task before switching an existing run configuration.
Legacy mode continues honoring its existing `extra_binds`; moving that profile
out of the repository hides its host paths but does not narrow its permissions.

In scoped mode, public phases retain access to the
two configured Hugging Face cache directories, mounted at `/rsi-cache/huggingface`
and `/rsi-cache/datasets`; these caches remain writable and shared as configured.
Use dedicated cache directories if runs must not share cache contents. Offline
phases continue using their private `/tmp/.cache/huggingface` cache. Common
`apptainer.environment` values are combined with phase environment overrides;
use `RSI_SHARED_DATA_ROOT = "/rsi-data"` for the sample mappings.

Every cluster Judge starts from the task image with a copy of the workspace,
as a clean-image Judge does locally: nothing the Agent changed outside the
workspace reaches it, `clean_image = false` has no effect there, and the run
plan records `clean_image: true`.

Cluster Judges run as the invoking user, so `rsi-agent` and `rsi-isolate`
confine what they run with Landlock instead of a separate uid, and so can a
science Judge library a task carries for the routes it runs. On a kernel
without Landlock such a call fails, unless the profile sets
`apptainer.judge_isolation_fallback = "same-user"`, which runs it as the
Judge's user without filesystem confinement. Every Judge gets that choice as
`RSI_JUDGE_ISOLATION_FALLBACK`, set explicitly (`"none"` by default, and
always `"none"` locally), so neither a task's settings nor its image can set
it. A model-access Judge reaches the model the way the Work Agent does:
through a private proxy to the pinned endpoints for a `"no-network"` Agent
(the proxy serves only the Judge's relay process), or over the site network
for a `"public"` Agent, whose egress the site governs, not the Harness. Model
access on a multi-node Judge is not supported yet.

`rsi-isolate isolation` and `rsi-agent isolation` also try creating the actual
Landlock ruleset; an ABI query alone does not establish that it can be used.
For Landlock they report `landlock_scoped` (the selected scope bitmask) and
`landlock_signal_scope` in addition to the existing fields.

Some RHEL 9 kernels report ABI 6 but omit `LANDLOCK_SCOPE_SIGNAL` (see
[the XZ upstream report](https://github.com/tukaani-project/xz/commit/5630c33a43a28a3d11030aa9d25fa8617e98da91)).
The default remains fail-closed. Prefer a corrected kernel or another isolation
backend when the task requires signal confinement. If the task's isolation
contract explicitly permits missing signal confinement, the operator may set
`apptainer.judge_allow_missing_landlock_signal = true` in the cluster profile
(default `false`). This requires no system administrator privileges: the operator
is the person launching the Harness run, not the task or its research Agent.
Every Judge gets `RSI_JUDGE_ALLOW_MISSING_LANDLOCK_SIGNAL` explicitly as `0` or
`1`; Docker always uses `0`. Image `ENV` values cannot opt in. Task environment
tables and all Apptainer environment tables reject this setting. `rsi-agent`
reads the policy only from its Harness-generated private launch spec, ignoring
its process environment; older launch specs default to `false`.

This is an opt-in policy decision, not an equivalent replacement for ABI 6:
only on reported ABI 6 and `EINVAL` does it retry without the signal scope.
All filesystem rights and abstract UNIX socket scoping remain required;
other failures still fail closed. Successful compatibility use prints a warning
on stderr and reports `landlock_scoped: 1`, `landlock_signal_scope: false`.
This does not enable the unconfined `same-user` fallback and is not inferred
from kernel version strings. `rsi-isolate --env` cannot forward this policy to
its child. The shared ruleset code accepts an explicit policy argument, never
an ambient environment variable.

A `judge_model` task runs on a cluster too, with the same `--judge-llm-*`
options, on a single-node Judge with a `"public"` or `"no-network"` Verifier.
An operator's Codex login works only when the compute nodes see and can
write the submit host's `auth.json` in place (a shared home); see
[`task-toml.md`](docs/harbor-task-authoring/task-toml.md#judge_model) for the
network, credential and Codex CLI details.

Cluster mode preserves the native RSI-Harness run and log format. Harbor only
compiles the task format; the adapter never invokes `harbor run`.


## View results

The default data and log roots are `.rsi-harness` and `logs`; run artifacts are
written under `logs/runs/RUN_ID`. From the repository root, start the stock
RSI Loop visualizer with:

```bash
rsi-harness visualize
```

Then open `http://127.0.0.1:8000`.

## Recovery and cleanup

Normal multi-round runs manage their own resources. `recover` is only needed
after interruption, a host restart, or an infrastructure failure:

```bash
sudo -E "$(command -v rsi-harness)" recover RUN_ID
```

Omit `RUN_ID` to recover every unfinished recorded run.

A completed run keeps its final Work state as a Docker image, plus a managed
WORKDIR volume when split mode is used. `recover` and `cleanup` keep an
interrupted run's Work state the same way, so it can be resumed. Ordinary
cleanup removes leftover runtime resources but keeps that state:

```bash
sudo -E "$(command -v rsi-harness)" cleanup RUN_ID
```

Delete the retained state explicitly when it is no longer needed:

```bash
sudo -E "$(command -v rsi-harness)" cleanup RUN_ID \
  --delete-workspace --yes
```

### Resume

`resume` continues a local run as its next attempt, whatever ended it: a
finished, cancelled, crashed or timed-out run. A run that did not end cleanly
is recovered first. Work is rebuilt from the retained image and WORKDIR
volume, so files are kept; background processes and GPU state are not. The
Agent continues its own session (or, if none is found, a new one with the
original prompt) and is told what changed.

```bash
sudo -E "$(command -v rsi-harness)" resume RUN_ID --dry-run
sudo -E "$(command -v rsi-harness)" resume RUN_ID --add-time 3600
```

The run keeps its Agent time and submissions left. `--add-time SECONDS` and
`--add-submissions N` grant more, `--cooldown SECONDS` replaces the cooldown,
and `--gpus LIST` picks other GPUs of the same count. Everything else comes
from the run's saved request. A round that was being judged at a crash counts
and has no score. An Agent left running by a crash may work past its deadline
until recovery; that work is kept and its attempt is charged up to the
deadline, so the run resumes only with `--add-time`. `final_result.json` of a
resumed run records its `attempts`.

On a cluster, `resume RUN_ID --cluster PROFILE` runs on the submit host once
the run's job has ended, and submits the next attempt under the same run ID
with a walltime sized to the time left; multi-node runs cannot resume yet.
An attempt whose job end the scheduler no longer records is charged up to
its deadline. Only the workspace and the Agent home are kept there: other
files Work wrote outside them are lost with the job.

## Quick Commands

Use `rsi-harness COMMAND --help` for the complete CLI reference. The examples
below assume that `rsi-harness` is on `PATH`; local Docker lifecycle commands
may need to be prefixed with `sudo -E "$(command -v rsi-harness)"` as shown
above.

| Command | Purpose | Example |
| --- | --- | --- |
| `run TASK_DIR` | Run a Harbor task locally, or submit it to a cluster when `--cluster` is set. | `rsi-harness run ../rsi-tasks/minference-sparse-prefill --agent codex --agent-auth local --model gpt-6-astra --reasoning-effort max --gpus 0` |
| `visualize` | Serve the run visualizer. | `rsi-harness visualize` |
| `recover [RUN_ID]` | Recover one interrupted run, or all unfinished runs when `RUN_ID` is omitted. | `rsi-harness recover RUN_ID` |
| `resume RUN_ID` | Continue a run as its next attempt ([Resume](#resume)). | `rsi-harness resume RUN_ID --add-time 3600` |
| `cleanup RUN_ID` | Remove leftover runtime resources while retaining the final workspace. | `rsi-harness cleanup RUN_ID` |
| `cleanup RUN_ID --delete-workspace --yes` | Also delete the retained final workspace without prompting. | `rsi-harness cleanup RUN_ID --delete-workspace --yes` |
| `sandbox prune-images` | Remove images that managed sandboxes first pulled to this host ([Pulled images](docs/sandbox-operator-guide.md#pulled-images)). | `rsi-harness sandbox prune-images --dry-run` |

### Common `run` options

| Option | Purpose | Default |
| --- | --- | --- |
| `TASK_DIR` | Path to a supported Harbor task directory. | Required |
| `--agent NAME` | Agent adapter to run: `codex`, `claude-code`, `qwen-code`, `kimi-code`, `opencode`, `zcode`, `pi`, or `oracle` for the task's own `solution/solve.sh` ([Agents](#agents)). | `codex` |
| `--gpus LIST` | Ordered, comma-separated local GPU indexes or UUIDs. The Work container receives the count declared by the task. This option cannot be combined with `--cluster`. | Automatically select the task-declared GPU count; `gpus = "all"` requires an explicit list |
| `--model NAME` | Override the model used by the selected Agent. | `RSI_AGENT_MODEL`, then the Agent's default |
| `--reasoning-effort LEVEL` | Set an Agent-supported reasoning effort: `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or `ultra` for Codex; `low`, `medium`, `high`, `xhigh` or `max` for Claude Code; `low`, `medium` or `high` for Kimi Code, OpenCode and ZCode; `minimal`, `low`, `medium` or `high` for pi. Other Agents refuse the option. | Agent default |
| `--agent-auth local` | Reuse credentials from the local Codex or Claude Code login. | Not set; use `RSI_AGENT_API_KEY` when provided |
| `--judge-llm-auth api\|codex-login` | Credential for a task's `judge_model`: an OpenAI API key, or a Codex login made with a ChatGPT plan (the Judge gets its access token only; before a Judge round that would outlive it, the Harness runs its pinned Codex CLI as you to refresh the login in place; never a login a Work Agent holds a copy of, which that refresh logs out; on a cluster, only a login the compute nodes see in place). Ignored, with a warning, for a task without `judge_model`. | Not set; a Codex Agent with a ChatGPT login or an OpenAI API key on OpenAI's own API lends its own (whatever its model; a login is renewed from Work before each Judge round), otherwise a `judge_model` task fails before the Agent starts |
| `--judge-llm-api-key-env NAME` / `--judge-llm-api-key-file PATH` | Where the `api` key is read from. | `RSI_JUDGE_LLM_API_KEY` |
| `--judge-llm-base-url URL` | API base URL for `api`; `/v1/responses` is appended (once). | `https://api.openai.com` |
| `--judge-llm-proxy URL` | HTTP(S) proxy the Judge uses to reach the Judge model. | The run's proxy settings (`RSI_HTTPS_PROXY`, `HTTPS_PROXY`, ...) |
| `--judge-llm-codex-home DIR` | Codex home holding `auth.json` for `codex-login`. | `$CODEX_HOME`, then `~/.codex` |
| `--timeout SECONDS` | Override the Agent time limit. | Task `agent.timeout_sec`, or 60 seconds if unset |
| `--max-submissions N` | Limit how many Judge submissions the Agent may make. The `oracle` submits exactly N times. | Unlimited; the `oracle` submits once |
| `--cooldown SECONDS` | Require a minimum delay between submissions. | `0.0` |
| `--primary-reward KEY` | Select the reward key used as the primary score. | `reward`, or the only reward key when exactly one exists |
| `--score-direction DIRECTION` | Choose whether the best score is the maximum or minimum: `maximize` or `minimize`. | `maximize` |
| `--disable-stop-hook` | Let the Agent stop naturally instead of installing the RSI Loop stop hook (or, for the Agents run under `rsi-agent-driver` and for pi, sending continuations). | Disabled; the stop hook is installed |
| `--sandbox-policy PATH` | Operator policy TOML that approves the task's [managed sandbox](#managed-child-sandboxes) request. | Not set; nothing is granted |
| `--cluster NAME_OR_PROFILE` | Run through a named cluster integration or profile TOML instead of local Docker. | Not set; run locally |
| `--dry-run` | Resolve and print cluster submissions without submitting jobs. Requires `--cluster`. | Disabled |
| `--data-root PATH` | Store runtime state under this directory. | `.rsi-harness` |
| `--logs-root PATH` | Store logs and run artifacts under this directory. | `logs` |
| `--verbose` | Include diagnostic details in errors. | Disabled |

The default roots are resolved relative to the current working directory.
`visualize` uses the same roots and listens on `127.0.0.1:8000` by default.
Both `recover` and `cleanup` also accept `--data-root`, `--logs-root`,
`--verbose`, and `--cluster NAME` (to kill a cluster run's E2B sandboxes);
`resume` accepts the same roots, `--verbose` and `--cluster NAME`.
`rsi-sandbox` is not a host command: it is the client injected into Work and
Judge ([Using the broker](docs/harbor-task-authoring/sandboxes.md#using-the-broker)).

## Acknowledgements

RSI Harness builds on [Harbor](https://github.com/harbor-framework/harbor) and
[EdgeBench](https://github.com/ByteDance-Seed/EdgeBench); we thank both projects
and their contributors.

## Citation

If you use RSI Harness in your research, please cite:

```bibtex
@misc{openrsi2026index,
  title        = {{OpenRSI Index}},
  author       = {{OpenRSI Foundation}},
  year         = {2026},
  howpublished = {\url{https://github.com/OpenRSI-Foundation/OpenRSI-Index}},
  note         = {Accessed: 2026-10-21}
}
```
