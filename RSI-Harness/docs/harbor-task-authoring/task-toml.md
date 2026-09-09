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
but does not use them to allocate resources. The only runtime metadata read by
RSI Harness is the namespaced Judge GPU field described below.

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
- system changes outside the WORKDIR are still captured so Judge sees packages,
  commands, and libraries installed by the Agent.

The WORKDIR must not equal, contain, or sit below Engine-owned mount targets
such as `/tests`, `/logs/verifier`, or `/run/rsi-harness/staging`.

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

## `[metadata.rsi_harness.verifier]`

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
