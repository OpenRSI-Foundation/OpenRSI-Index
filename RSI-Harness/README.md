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
3. **Judge:** RSI Harness starts a fresh, isolated Judge container from that
   snapshot, injects the private tests, runs `/tests/test.sh`, records the test
   output and reward, and removes the Judge container. The Judge can also read
   the Agent's trajectory up to this submission, read-only, at `/logs/agent`
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
public Judge also starts from the clean image Work started from, read-only,
and reads only Work's `WORKDIR`, so the Agent cannot alter the programs that
judge it. The default, `"hidden"`, keeps the tests private. See
[`task-toml.md`](docs/harbor-task-authoring/task-toml.md#visibility).

A Judge can also call the run's own Agent, for example to review a
submission, when the task declares `model_access = true` in the same table.
`rsi-agent exec --prompt-file FILE --output FILE [--schema FILE]` is then the
only supported way for the Judge to call a model: it runs the run's Agent,
with Work's credentials and settings, on one prompt, as its own unprivileged
user, and the Judge procedure never sees the credentials. The Judge reaches
only the Agent's pinned model endpoints, never general egress; local Docker
runs only. See
[`task-toml.md`](docs/harbor-task-authoring/task-toml.md#model_access).

[`sample_tasks/science-ising3d`](sample_tasks/science-ising3d) combines these
for open-ended science: a public rubric, a Judge started from the clean task
image, and a trajectory audit and per-rubric review by the Agent model.

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
| `opencode` | OpenCode (linux-x64) | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | `low`, `medium`, `high`, sent as the endpoint's `reasoning_effort` |
| `zcode` | ZCode (linux-x64) | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | `low`, `medium`, `high`, sent as the endpoint's `reasoning_effort` |
| `pi` | pi | `RSI_AGENT_API_KEY` only | `RSI_AGENT_API_BASE_URL`, `RSI_AGENT_CONTEXT_WINDOW`, `RSI_AGENT_MAX_OUTPUT_TOKENS`; the model from `--model` or `RSI_AGENT_MODEL` (setup fails without one) | `minimal`, `low`, `medium`, `high`, sent as the endpoint's `reasoning_effort` |

Each Agent CLI is installed at an exact pinned version, over HTTPS, into the
Work image when the image is built, together with the node it runs on. All of
these versions live in one table,
[`src/rsi_loop/harness/agent/versions.py`](src/rsi_loop/harness/agent/versions.py).
To fetch the ZCode package from a mirror with the same
`<version>/linux-x64/<file>` layout, set `RSI_ZCODE_MIRROR_URL` where
`rsi-harness run` runs; the Work image build receives it as a build argument.

To upgrade an Agent, or node for all of them:

```bash
uv run python scripts/bump_agent.py codex 0.160.0   # or scripts/bump-agent.sh
```

The script checks that the version is published, rewrites that one entry in
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
  --model gpt-5.6-sol \
  --reasoning-effort xhigh \
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
WORKDIR volume when split mode is used. Ordinary cleanup removes leftover
runtime resources but keeps that final state:

```bash
sudo -E "$(command -v rsi-harness)" cleanup RUN_ID
```

Delete the retained final state explicitly when it is no longer needed:

```bash
sudo -E "$(command -v rsi-harness)" cleanup RUN_ID \
  --delete-workspace --yes
```

## Quick Commands

Use `rsi-harness COMMAND --help` for the complete CLI reference. The examples
below assume that `rsi-harness` is on `PATH`; local Docker lifecycle commands
may need to be prefixed with `sudo -E "$(command -v rsi-harness)"` as shown
above.

| Command | Purpose | Example |
| --- | --- | --- |
| `run TASK_DIR` | Run a Harbor task locally, or submit it to a cluster when `--cluster` is set. | `rsi-harness run ../rsi-tasks/minference-sparse-prefill --agent codex --agent-auth local --model gpt-5.6-sol --reasoning-effort xhigh --gpus 0` |
| `visualize` | Serve the run visualizer. | `rsi-harness visualize` |
| `recover [RUN_ID]` | Recover one interrupted run, or all unfinished runs when `RUN_ID` is omitted. | `rsi-harness recover RUN_ID` |
| `cleanup RUN_ID` | Remove leftover runtime resources while retaining the final workspace. | `rsi-harness cleanup RUN_ID` |
| `cleanup RUN_ID --delete-workspace --yes` | Also delete the retained final workspace without prompting. | `rsi-harness cleanup RUN_ID --delete-workspace --yes` |
| `sandbox prune-images` | Remove images that managed sandboxes first pulled to this host ([Pulled images](docs/sandbox-operator-guide.md#pulled-images)). | `rsi-harness sandbox prune-images --dry-run` |

### Common `run` options

| Option | Purpose | Default |
| --- | --- | --- |
| `TASK_DIR` | Path to a supported Harbor task directory. | Required |
| `--agent NAME` | Agent adapter to run: `codex`, `claude-code`, `qwen-code`, `kimi-code`, `opencode`, `zcode` or `pi` ([Agents](#agents)). | `codex` |
| `--gpus LIST` | Ordered, comma-separated local GPU indexes or UUIDs. The Work container receives the count declared by the task. This option cannot be combined with `--cluster`. | Automatically select the task-declared GPU count; `gpus = "all"` requires an explicit list |
| `--model NAME` | Override the model used by the selected Agent. | `RSI_AGENT_MODEL`, then the Agent's default |
| `--reasoning-effort LEVEL` | Set an Agent-supported reasoning effort: `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or `ultra` for Codex; `low`, `medium`, `high`, `xhigh` or `max` for Claude Code; `low`, `medium` or `high` for Kimi Code, OpenCode and ZCode; `minimal`, `low`, `medium` or `high` for pi. Other Agents refuse the option. | Agent default |
| `--agent-auth local` | Reuse credentials from the local Codex or Claude Code login. | Not set; use `RSI_AGENT_API_KEY` when provided |
| `--timeout SECONDS` | Override the Agent time limit. | Task `agent.timeout_sec`, or 60 seconds if unset |
| `--max-submissions N` | Limit how many Judge submissions the Agent may make. | Unlimited |
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
`--verbose`, and `--cluster NAME` (to kill a cluster run's E2B sandboxes).
`rsi-sandbox` is not a host command: it is the client injected into Work and
Judge ([Using the broker](docs/harbor-task-authoring/sandboxes.md#using-the-broker)).

## Acknowledgements

RSI Harness builds on [Harbor](https://github.com/harbor-framework/harbor) and
[EdgeBench](https://github.com/ByteDance-Seed/EdgeBench); we thank both projects
and their contributors.
