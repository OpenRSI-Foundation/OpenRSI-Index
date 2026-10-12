# `task.toml` reference for RSI Harness

This document describes the `task.toml` fields that RSI Harness reads for a
standard Harbor GPU task. It describes file syntax, not the names of internal
Python model fields.

RSI Harness intentionally supports one continuous Linux task, one environment,
one Agent phase, and a disposable Judge created for each submission. Unsupported
Harbor features are listed at the end of this document.

For the supported Compose and Dockerfile fields, see
[`docker-compose.md`](docker-compose.md).

A task directory has this basic shape:

```text
my-task/
├── task.toml
├── instruction.md
├── environment/
│   ├── Dockerfile
│   └── docker-compose.yaml
└── tests/
    └── test.sh
```

The Dockerfile or Compose file may be omitted when another supported image
source is present, but `task.toml`, `instruction.md`, and the platform test
script are required.

## Complete example

```toml
schema_version = "1.4"

[task]
name = "example/my-gpu-task"
version = "1.0.0"
description = "Implement and verify a GPU feature"

[metadata]
author_name = "Example Author"
author_email = "author@example.com"
difficulty = "medium"
category = "feature"
tags = ["gpu"]

[environment]
os = "linux"
cpus = 4
memory_mb = 16384
storage_mb = 30720
build_timeout_sec = 1800
workdir = "/testbed"
gpus = 2
gpu_types = ["NVIDIA H100 NVL"]
network_mode = "no-network"

[environment.env]
ORDINARY_SETTING = "value"
OPTIONAL_PORT = "${OPTIONAL_PORT:-8000}"
PRIVATE_TOKEN = "${PRIVATE_TOKEN}"

[agent]
timeout_sec = 3600
network_mode = "public"

[verifier]
timeout_sec = 3600
network_mode = "no-network"

[verifier.env]
VERIFIER_SETTING = "value"

[metadata.rsi_harness.verifier]
gpus = 2
```

The example requests two GPUs for Work and two GPUs for Judge. The two counts
are independent. They do not mean that four GPUs are always required: when the
caller provides only two GPUs, RSI Harness can use release-all mode and let
Work and Judge use the same two GPUs at different times.

## Root fields

### `schema_version`

```toml
schema_version = "1.4"
```

- Type: string.
- Effective requirement: `"1.4"`.
- Harbor currently defaults an omitted value to `"1.4"`, but task authors
  should declare it explicitly.
- RSI Harness rejects other schema versions before creating runtime resources.

### `[task]`

```toml
[task]
name = "example/my-gpu-task"
version = "1.0.0"
description = "Implement and verify a GPU feature"
authors = []
keywords = ["gpu"]
```

- `name` is the Harbor package name and must use `organization/name` form.
- `version`, `description`, `authors`, and `keywords` are Harbor package
  metadata. They do not change container resources or Judge behavior.
- The complete `[task]` table is optional. When it is absent, Harbor derives
  the task identifier from the task directory name.

### `[metadata]`

```toml
[metadata]
author_name = "Example Author"
author_email = "author@example.com"
difficulty = "medium"
category = "feature"
tags = ["gpu"]
```

These are descriptive benchmark fields. RSI Harness preserves Harbor parsing
but does not use them to allocate resources. RSI Harness reads runtime
metadata only from its namespaced `[metadata.rsi_harness]` tables, such as the
Judge fields described below.

### `source`

```toml
source = "https://example.com/task-source"
```

- Type: string.
- Optional Harbor provenance metadata.
- RSI Harness does not clone or fetch this value. The local task directory
  supplied to `rsi-harness run` remains the authoritative task source.

## `[environment]`

The `[environment]` table describes the common task environment. Work uses
these values, and Judge inherits the applicable resource and baseline policy
unless a supported phase override says otherwise.

### `os`

```toml
[environment]
os = "linux"
```

- Type: string.
- Default: `"linux"`.
- Supported value in RSI Harness: `"linux"` only.
- Windows and TPU tasks are rejected.

### `cpus`

```toml
[environment]
cpus = 4
```

- Type: integer.
- Optional.
- Applied as the Docker CPU limit for Work and each Judge.
- Choose a value large enough for the Agent's local tests and the complete
  Verifier suite.

### `memory_mb`

```toml
[environment]
memory_mb = 16384
```

- Type: integer MiB.
- Optional.
- Applied as the Docker memory limit for Work and each Judge.
- Harbor still accepts deprecated values such as `memory = "16G"`, but new
  tasks should use `memory_mb` so the unit is unambiguous.
- This field is separate from `/dev/shm`; configure shared memory with Compose
  `services.main.shm_size`.

### `storage_mb`

```toml
[environment]
storage_mb = 30720
```

- Type: integer MiB.
- Optional.
- Harbor still accepts deprecated values such as `storage = "30G"`, but new
  tasks should use `storage_mb`.
- RSI Harness records this value for planning and artifacts. Docker must still
  have enough real host space for images, checkpoints, and temporary Judge
  snapshots; this field does not create a per-container disk quota.

### `build_timeout_sec`

```toml
[environment]
build_timeout_sec = 1800
```

- Type: number of seconds.
- Harbor default: `600`.
- Controls how long image preparation may take.
- It does not control Agent execution or Verifier execution. Those have their
  own `timeout_sec` fields.

### `docker_image`

```toml
[environment]
docker_image = "registry.example/task@sha256:..."
```

- Type: image reference string.
- Optional alternative to a local `environment/Dockerfile` build.
- Prefer an immutable digest.
- It must not conflict with Compose `services.main.build`.
- If Compose also declares `services.main.image`, the two image strings must
  be identical.

### `gpus`

```toml
[environment]
gpus = 2
```

- Type: positive integer.
- Meaning: number of GPUs visible to Work.
- A GPU task must declare its Work GPU count either here or in the Compose
  NVIDIA reservation.
- If both this field and Compose `devices[].count` are present, their values
  must match.
- Work receives the first matching devices from the ordered GPU pool supplied
  by the run command's `--gpus` option.
- `count: all` exists only in Compose. It gives Work every GPU in the caller's
  pool and should be used only when the task genuinely scales to the entire
  assigned pool.

This is not the Judge GPU field. Judge GPU allocation is described under
`[metadata.rsi_harness.verifier]`.

### `gpu_types`

```toml
[environment]
gpu_types = ["NVIDIA H100 NVL"]
```

- Type: list of strings.
- Optional.
- RSI Harness currently accepts at most one required Work GPU type.
- The selected Work devices must match it.
- Judge-specific GPU type selection is not currently supported.

### `workdir`

```toml
[environment]
workdir = "/testbed"
```

- Type: absolute POSIX path.
- Examples include `/testbed`, `/repo`, `/workspace`, or `/app`. `/workspace`
  is only an example; it has no special status.
- Relative paths and paths containing `..` are rejected.
- `""`, `"."`, and `"/"` all resolve to the root WORKDIR `/`.

The effective WORKDIR is chosen in this order:

1. `task.toml [environment].workdir`;
2. Compose `services.main.working_dir`;
3. the prepared image's Docker `Config.WorkingDir`, normally set by Dockerfile
   `WORKDIR`;
4. `/` when none of the above supplies a non-root path.

If both `task.toml [environment].workdir` and Compose
`services.main.working_dir` are present, they must be identical. An explicit
task or Compose value overrides the image's default `WORKDIR` metadata.

A non-root effective WORKDIR must already exist in the clean Base/Judge image.
There are three normal ways to ensure that:

```dockerfile
# Creates the directory and also makes it the image default.
WORKDIR /testbed
```

```dockerfile
# Creates it without making it the image default.
RUN mkdir -p /testbed
```

or use a prebuilt image that already contains `/testbed`. Declaring
`workdir = "/testbed"` without putting `/testbed` in the image fails the image
preflight. The Engine does not accept an arbitrary nonexistent path and treat
it as a valid empty code directory.

When the effective WORKDIR is non-root, RSI Harness uses split-WORKDIR mode:

- a fresh Engine-owned Docker volume is mounted read-write into Work at the
  exact effective WORKDIR;
- Docker copies the image's existing files at that directory into the fresh
  volume on first use, so a pre-populated `/testbed` is preserved;
- every Judge receives a read-only CoW snapshot of that volume at the same
  path;
- Judge writes cannot modify Work or later Judge rounds;
- large checkpoints inside the WORKDIR are not copied into every rootfs image;
- system changes outside the WORKDIR are still captured so a hidden Judge sees
  packages, commands, and libraries installed by the Agent; a public Judge
  starts from the clean image instead (see
  [`visibility`](#visibility)).

The WORKDIR must not equal, contain, or sit below Engine-owned mount targets
such as `/tests`, `/logs/verifier`, `/logs/agent`, or
`/run/rsi-harness/staging`.

When the effective WORKDIR is `/`, RSI Harness uses full-rootfs mode. Judge sees
the complete modified Work filesystem, but each submission requires a Docker
rootfs snapshot. Large files anywhere in `/` can therefore increase snapshot
time and disk use.

For more examples and the corresponding Compose/Dockerfile behavior, see
[`docker-compose.md`](docker-compose.md#servicesmainworking_dir-and-dockerfile-workdir).

### `network_mode`

```toml
[environment]
network_mode = "no-network"
```

- Type: `"public"`, `"no-network"`, or `"allowlist"`.
- Harbor default: `"public"`.
- This is the baseline for Agent and Verifier.
- RSI Harness creates and enforces its own Docker network policy. Do not use
  Compose `network_mode`.
- With `"no-network"`, Work reaches only the pinned Agent provider or proxy
  endpoints, named in its `/etc/hosts`, and cannot resolve any other name.

### `allowed_hosts`

```toml
[environment]
network_mode = "allowlist"
allowed_hosts = ["api.example.com", "192.0.2.10/32"]
```

- Type: list of hostnames, IP literals/CIDRs, or supported leading-wildcard
  host patterns.
- Valid only with `network_mode = "allowlist"`.
- Names are resolved and pinned before runtime mutation; an invalid or
  unresolvable required endpoint fails setup.

### `allow_internet` (deprecated)

```toml
[environment]
allow_internet = false
```

- Type: boolean.
- Deprecated Harbor compatibility field.
- Harbor maps `true` to public networking and `false` to no-network only when
  current `network_mode`/`allowed_hosts` fields do not already define a policy.
- New tasks should use `network_mode` and `allowed_hosts` directly.

### `[environment.env]`

```toml
[environment.env]
ORDINARY_SETTING = "value"
PRIVATE_TOKEN = "${PRIVATE_TOKEN}"
OPTIONAL_PORT = "${OPTIONAL_PORT:-8000}"
```

- Keys and values are strings.
- Literal values become ordinary environment values.
- `${NAME}` requires the Engine host to supply `NAME`.
- `${NAME:-default}` uses the host value when present and otherwise uses the
  stated default.
- Template-derived secret values are resolved at runtime and passed only to
  the relevant execution phase. They are redacted from Engine diagnostics.
- Never place the real credential in the task directory.
- When the same key exists in Compose `services.main.environment`, this table
  takes precedence.

## `[agent]`

### `timeout_sec`

```toml
[agent]
timeout_sec = 3600
```

- Type: seconds.
- RSI Harness uses `60` seconds when this field and the CLI `--timeout` option
  are both absent.
- The CLI `--timeout` value overrides this field.

### `user`

```toml
[agent]
user = "root"
```

- Type: username or numeric UID.
- Optional.
- It participates in image identity and preflight. RSI Harness currently runs
  the Work container and Agent with root authority so an official Harbor task
  may change system packages, shell commands, and PATH contents. Do not treat
  this field as a sandbox boundary.

### `network_mode` and `allowed_hosts`

```toml
[agent]
network_mode = "allowlist"
allowed_hosts = ["api.openai.com"]
```

- Omit `network_mode` to inherit the `[environment]` baseline.
- Setting it creates an Agent-phase override.
- `allowed_hosts` follows the same rule as the environment field and is valid
  only for allowlist mode.

`[agent]` has no supported `env` table. Put shared values under
`[environment.env]`; Agent provider credentials are supplied to RSI Harness at
runtime.

## `[verifier]`

### `timeout_sec`

```toml
[verifier]
timeout_sec = 3600
```

- Type: seconds.
- Harbor default: `600`.
- Applies independently to every Judge submission.

### `user`

```toml
[verifier]
user = "root"
```

- Type: username or numeric UID.
- Optional.
- Overrides the Compose/image user for Judge execution when set.
- The selected user must exist and must be able to access the effective
  WORKDIR and required test dependencies.
- The Judge's `/tests` stays mode `0755` after the task tests are copied in,
  so a non-root user can read them; file modes come from the task's `tests/`.

### `network_mode` and `allowed_hosts`

```toml
[verifier]
network_mode = "no-network"
```

- Omit `network_mode` to inherit the `[environment]` baseline.
- Setting it creates a Verifier-phase override.

### `[verifier.env]`

```toml
[verifier.env]
VERIFIER_SETTING = "value"
VERIFIER_TOKEN = "${VERIFIER_TOKEN}"
```

- Merged on top of the common environment only for Verifier execution.
- Supports the same `${NAME}` and `${NAME:-default}` runtime templates.
- Resolved secret values are not placed in persisted container configuration.

### `environment_mode`

```toml
[verifier]
environment_mode = "shared"
```

- `"shared"` is compatible with this Engine's model: Judge is disposable but
  derived from the task/Work environment.
- Harbor `"separate"` Verifier environments and `[verifier.environment]` are
  rejected. RSI Harness does not accept a separately authored Judge image.

### Agent trajectory in the Judge

Every Judge round, with either visibility, can read the Agent's trajectory
read-only at `/logs/agent`. Nothing needs to be declared. It holds:

- `agent_prompt.md`: the prompt the Agent was started with;
- `agent_output.txt`: the Agent's output as captured by RSI Harness, with
  secrets redacted, up to the moment Work was paused for this submission;
- `manifest.json`: `{schema_version, round_id, previous_round_id,
  limit_bytes, files}`, with `schema_version` the integer `1`, `round_id`
  this round (`agent-<n>`), `previous_round_id` the round the previous sizes
  come from (or `null`) and `limit_bytes` the per-file limit. Each entry of
  `files` is `{path, source_size, offset, bytes, sha256, truncated,
  previous_size}`: the path of the copy below `/logs/agent`, the source size
  at capture, the copied range of the source (`offset` to `offset + bytes`),
  the copy's sha256, whether the range was truncated, and the source size at
  the previous round (`null` for the first), so the Judge can read only what
  is new since the last submission: the copy's bytes from
  `max(previous_size - offset, 0)`.

The copy comes from the current run only and is frozen when Work is paused,
so it never contains another run's output or anything the Agent did after
submitting. Each file is limited to its most recent 256 MiB; `offset` is then
non-zero and `truncated` is true. The files are world-readable inside the
Judge, so a non-root Verifier user can read them. After the round only the
manifest is kept, in `judge-trajectory/<round>/` of the run's log directory;
the copied range can be recovered from the run's `agent_output.txt`.

## `[metadata.rsi_harness.verifier]`

[`sample_tasks/science-ising3d`](../../sample_tasks/science-ising3d)
illustrates `visibility` and `model_access`: its Judge, the public science
Judge library of OpenRSI-X's `science-task-agent` skill with the task's own
configuration, re-executes the candidate as unprivileged users and reviews it
with the run's Agent through `rsi-agent`. It is a demonstration, not a
template to copy.

### `gpus`

```toml
[metadata.rsi_harness.verifier]
gpus = 2
```

- Type: non-negative integer.
- Default when the table or field is absent: `0`.
- Meaning: number of GPUs visible to each disposable Judge.
- `0` means CPU-only Judge.
- Unknown fields in this namespaced `verifier` table are rejected.

The caller's ordered `--gpus` value is a pool. For example, with Work
`gpus = 2`, Judge `gpus = 2`, and `--gpus 4,5,6,7`:

- Work receives GPUs 4 and 5;
- Judge receives GPUs 6 and 7;
- Work is paused during Judge but its GPU processes are not terminated.

With the same task and only `--gpus 4,5`, there are enough total GPUs but no
disjoint spare pair. RSI Harness therefore uses release-all mode:

- Work is paused;
- all Work GPU processes must release their GPUs;
- Judge temporarily uses GPUs 4 and 5;
- Work is resumed after Judge cleanup.

If the pool contains fewer devices than either phase requires, setup fails
before the Agent starts.

### `visibility`

```toml
[metadata.rsi_harness.verifier]
visibility = "public"
```

- Type: `"hidden"` or `"public"`.
- Default when the table or field is absent: `"hidden"`.
- `"hidden"`: the task's `tests/` directory exists only in each Judge, as
  `/tests`. Work never sees it.
- `"public"`: Work also sees the same `tests/` directory, read-only, at
  `/tests` for the whole run. Use it when the Judge's procedure is meant to
  be read, such as a published rubric or a review procedure.

The run plan records one digest of `tests/` at compile time (`tests_digest`
in `run-plan.json`). With `"public"`, RSI Harness checks the directory against
that digest before Work is created and again before every Judge round, so Work
reads exactly the bytes the Judge runs. If the directory changes during the
run, Work creation fails, or the round reports an infrastructure error, instead
of judging with tests the Agent never saw. Files keep their modes. Keep them
readable by the Agent's user.

Because Work can read the Judge procedure, a public Judge also runs from a
clean image, so the Agent cannot change the programs that judge it. Each
public Judge starts from the immutable image Work started from (the task image
plus the Agent CLI, built before the Agent ran) instead of a snapshot of Work,
with a read-only root filesystem and a 256 MiB `noexec` `/tmp` tmpfs: run a
file written there through its interpreter (`python3 file.py`, `sh file.sh`);
executing it or loading a compiled library from it fails, although Work's
`/tmp` allows both. It reads Work's
`WORKDIR` snapshot read-only; nothing the Agent installs or changes elsewhere
reaches it, so a submission must live in `WORKDIR`. A public task therefore
needs a `WORKDIR` other than `/`; setup fails otherwise. On a cluster, every
Judge already runs from the task image with only a copy of the workspace.

A public task should not rely on secrets in `tests/`. Values the Agent must not
see belong in `[verifier.env]` runtime templates, which only the Judge
receives. The Agent prompt states that `/tests` is public.

### `model_access`

```toml
[metadata.rsi_harness.verifier]
model_access = true
```

- Type: boolean.
- Default when the table or field is absent: `false`.
- `true`: each Judge can call the run's own Agent, for example to review a
  submission against a rubric, through `rsi-agent`.

`rsi-agent` is the only supported way for a Judge to call a model. It runs
the run's Agent (the same CLI, credentials, provider or proxy, model and
reasoning effort as Work) on one prompt and writes its final message, so a
task works with every Agent RSI Harness supports and needs no change for a
new one. The Judge procedure never calls a model API or an Agent CLI itself,
and never sees the credentials.

```bash
rsi-agent exec --prompt-file FILE --output FILE [--schema FILE] \
    [--timeout SEC] [--workdir DIR] [--log FILE]
```

- `--prompt-file`: the prompt, read by the caller (at most 4 MiB). The Agent
  reads it as its task and may read files, such as evidence the procedure
  placed in the working directory or anywhere world-readable.
- `--output`: receives the Agent's final message, written atomically only
  when the call succeeds.
- `--schema`: a JSON Schema object. The Agent is asked for a final message
  that is only JSON conforming to it, through its own structured-output
  option when it has one, otherwise in the prompt. Validate the result: the
  schema is a request, not a guarantee.
- `--timeout`: seconds, default 1800, for the whole call. On expiry the
  Agent's whole process group gets `SIGTERM`, then `SIGKILL` after 5
  seconds. A caller with its own deadline should pass a timeout some seconds
  before it, so rsi-agent stops the Agent and cleans up first. rsi-agent
  stops its Agent at once when it receives `SIGTERM` itself.
- `--workdir`: the Agent's working directory. The Agent owns it for the
  call: it may delete, replace or add any entry in it, so never place a
  trusted file there, and read what it leaves as the Agent's output.
  Afterwards the directory's own owner and mode are restored. `--output`
  and `--log` inside it are refused. Without it the Agent works in a private
  scratch directory. `/`, top-level directories, `/tests`, `/logs/verifier`,
  `/logs/agent` and rsi-agent's state are refused.
- `--log`: also writes the Agent's stdout and stderr, redacted.

Exit status: `0` the output was written; `1` the Agent failed (non-zero exit,
no, empty or symlinked final message, or an error it reported), with one
reason line and the tail of its stderr; `2` a usage error (including a prompt over 4 MiB or a schema over
1 MiB); `3` model access
is unavailable (no `model_access = true`, the Agent CLI is missing, or the
run's Agent cannot run one-shot); `124` the timeout expired; `125` an
internal setup failure; `143` rsi-agent was terminated. Credential values
never appear in its output, the final message, the log or stderr.

The Judge receives `rsi-agent` from RSI Harness, read-only, at
`/usr/local/bin/rsi-agent` (and `/opt/rsi-agent`); it is not part of the
Work image, and Judges without model access do not have it. It needs
`python3` 3.8 or later in the Judge's image, which setup checks.

Isolation. When the Judge runs as root, each call runs the Agent as its own
unprivileged user with a fresh private `HOME` holding copies of a local
login's files, and with `no_new_privs`. Calls may run concurrently, up to 16
at a time (more wait for a free user within their timeout), and no two
concurrent calls share a user: uids 61000 to 61015 and gids of the same
numbers, which must not exist in the image (setup checks). That user cannot
write `/logs/verifier` or the reward, cannot read rsi-agent's launch spec or
the credential store, cannot read a hidden Judge's `/tests` (mode `0700`
there), and cannot read, change or signal another call's files or
processes. It can read `WORKDIR`, `/logs/agent`, a public Judge's `/tests`,
its working directory and anything world-readable, so stage evidence the
Agent must read accordingly. Its network is the Judge's. When a call ends,
every process still running as its user is killed, including any that left
the Agent's process group. A Judge whose Verifier user is not root gets no
such isolation: every call's Agent then runs as that user, which rsi-agent
states on stderr.

The Judge also has:

- **Network:** the Judge reaches the Agent provider or proxy endpoints that
  RSI Harness pinned for Work, at the same addresses, named in the Judge's
  `/etc/hosts`. Every other destination follows `[verifier] network_mode`;
  with `"no-network"` it is rejected, and the Judge cannot resolve any other
  name: its DNS server is an address the firewall rejects. Model access never
  opens general egress.
- **Environment:** of the Agent's settings, the test command receives only
  `RSI_AGENT_NAME`, `RSI_AGENT_MODEL` and `RSI_AGENT_REASONING_EFFORT` (when
  set), for records; the task's `[environment.env]` and `[verifier.env]`
  reach it as usual. The Agent's API key, base URL and proxy reach only
  `rsi-agent`.
- **Local login** (`--agent-auth local`): the login files are placed, from
  memory, in a private tmpfs that only root (or a non-root Verifier user)
  reads; `rsi-agent` copies them into each call's private `HOME`. They are
  not part of any snapshot.
- **Redaction:** these credential values are replaced with `[REDACTED]` in
  Judge output, feedback, events and errors.

`[verifier] timeout_sec` bounds the whole test command, every `rsi-agent`
call included: size it to cover the procedure's own work plus its model
calls, and give each call a `--timeout` that ends before the procedure's own
deadline.

Setup fails if no endpoint can be pinned: configure the Agent API base URL or
a proxy, or use an Agent with a registered default API URL. Only local Docker
runs support model access; cluster runs reject the task before submitting a
job.

A hidden Judge's image is a snapshot of Work, so the Agent can change programs
it runs, including the Python interpreter and the model CLI that `rsi-agent`
uses. The Judge holds only the credential the Agent already has, but a hidden
review procedure should not trust Work-installed tools to report its verdict.
A public Judge runs from the clean image instead (see
[`visibility`](#visibility)).

## Unsupported `task.toml` features

RSI Harness rejects the following rather than silently changing their meaning:

- `steps` and multi-step reward strategies;
- Windows environments and TPU requirements;
- MCP servers, skills directories, environment healthchecks, and multiple
  acceptable GPU types;
- Harbor artifact collection and Verifier collect hooks;
- solution environment values;
- `environment_mode = "separate"` and `[verifier.environment]`;
- a separately authored Verifier image.

The standard task must also provide `instruction.md`, an environment definition,
and `tests/test.sh`. The test script should print only feedback the Agent may
see and write `/logs/verifier/reward.json` or
`/logs/verifier/reward.txt`. Missing or malformed reward output is a Verifier
error, not a score of zero.
