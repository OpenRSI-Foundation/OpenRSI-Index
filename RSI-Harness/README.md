# RSI Harness

RSI Harness powers RSI's First Exam for ultra-long-horizon RSI runs, natively
supporting [Harbor-format tasks](../rsi-tasks/) with or without GPUs on local
Docker, and GPU tasks on multi-node clusters.

Agents such as Claude Code and Codex develop solutions in a persistent, isolated
Work container and submit them to a fresh Judge for scoring. They can iterate on
test feedback until the submission limit or timeout, with the highest score
retained as the final result.

RSI Harness runs the **outer loop** only: preparing the task, pausing and
snapshotting Work, scoring each submission in a fresh Judge, and selecting the
final result. The **inner loop** is whichever agent you put in it — Claude Code,
Codex, or an agent you design yourself.


## How it works

A run starts from a supported Harbor task directory containing `task.toml`,
`instruction.md`, `environment/`, and `tests/test.sh`. RSI Harness validates the
task, builds or resolves its container image, and records its resources, network
policy, and timeouts in a run plan. It then follows this lifecycle:

1. **Work:** RSI Harness starts a persistent, isolated Work container. The Agent
   edits the task workspace here, but the task's private `tests/` directory is
   never mounted into this container.
2. **Submit:** When the Agent runs `rsi-submit`, the client sends a token-scoped
   control request. RSI Harness pauses Work and snapshots its filesystem state;
   the client does not package or upload the workspace.
3. **Judge:** RSI Harness starts a fresh, isolated Judge container from that
   snapshot, injects the private tests, runs `/tests/test.sh`, records the test
   output and reward, and removes the Judge container.
4. **Iterate or finish:** RSI Harness resumes the same Work container and returns
   the Judge feedback to the Agent. The cycle repeats until the submission limit
   or timeout. Among valid submissions, the snapshot with the best primary
   reward, according to the configured score direction, becomes the final
   result.

Work persists across rounds, while each submission is evaluated in a fresh
Judge, letting the Agent improve from feedback without access to the private
tests.

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

These values are passed only to the Agent process and are redacted from Engine
artifacts. Do not put credentials in the Harbor task directory.

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

For CPU-only local tasks, declare Work's count explicitly in `task.toml`:

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
CPU-only execution is currently supported by the local Docker backend;
Blue Vela's launch path still requires GPUs.

## Run multi-node tasks on a cluster

The `bluevela` adapter supports LSF clusters. Its packaged profile is a portable
example: `/srv/rsi` paths and the scheduler group are placeholders, not a working
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
LSF allocations using the profile's GPUs-per-node setting. `--gpus` is only for
local runs, and `gpus = "all"` needs a numeric cluster override.

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

### Common `run` options

| Option | Purpose | Default |
| --- | --- | --- |
| `TASK_DIR` | Path to a supported Harbor task directory. | Required |
| `--agent NAME` | Agent adapter to run: `codex` or `claude-code`. | `codex` |
| `--gpus LIST` | Ordered, comma-separated local GPU indexes or UUIDs. The Work container receives the count declared by the task. This option cannot be combined with `--cluster`. | Automatically select the task-declared GPU count; `gpus = "all"` requires an explicit list |
| `--model NAME` | Override the model used by the selected Agent. | `RSI_AGENT_MODEL`, then the Agent's default |
| `--reasoning-effort LEVEL` | Set an Agent-supported reasoning effort. | Agent default |
| `--agent-auth local` | Reuse credentials from the local Codex or Claude Code login. | Not set; use `RSI_AGENT_API_KEY` when provided |
| `--timeout SECONDS` | Override the Agent time limit. | Task `agent.timeout_sec`, or 60 seconds if unset |
| `--max-submissions N` | Limit how many Judge submissions the Agent may make. | Unlimited |
| `--cooldown SECONDS` | Require a minimum delay between submissions. | `0.0` |
| `--primary-reward KEY` | Select the reward key used as the primary score. | `reward`, or the only reward key when exactly one exists |
| `--score-direction DIRECTION` | Choose whether the best score is the maximum or minimum: `maximize` or `minimize`. | `maximize` |
| `--disable-stop-hook` | Let the Agent stop naturally instead of installing the RSI Loop stop hook. | Disabled; the stop hook is installed |
| `--cluster NAME_OR_PROFILE` | Run through a named cluster integration or profile TOML instead of local Docker. | Not set; run locally |
| `--dry-run` | Resolve and print cluster submissions without submitting jobs. Requires `--cluster`. | Disabled |
| `--data-root PATH` | Store runtime state under this directory. | `.rsi-harness` |
| `--logs-root PATH` | Store logs and run artifacts under this directory. | `logs` |
| `--verbose` | Include diagnostic details in errors. | Disabled |

The default roots are resolved relative to the current working directory.
`visualize` uses the same roots and listens on `127.0.0.1:8000` by default.
Both `recover` and `cleanup` also accept `--data-root`, `--logs-root`, and
`--verbose`.

## Acknowledgements

RSI Harness builds on [Harbor](https://github.com/harbor-framework/harbor) and
[EdgeBench](https://github.com/ByteDance-Seed/EdgeBench); we thank both projects
and their contributors.
