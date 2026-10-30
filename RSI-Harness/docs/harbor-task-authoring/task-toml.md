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
├── solution/
│   └── solve.sh
└── tests/
    └── test.sh
```

The Dockerfile or Compose file may be omitted when another supported image
source is present, but `task.toml`, `instruction.md`, and the platform test
script are required. `solution/` is optional and never reaches a model
Agent's Work; `--agent oracle` runs its `solve.sh` (as Harbor's oracle does)
in Work and submits, to validate the Judge on the real run path (see the
[README](../../README.md#validate-a-tasks-judge-with-the-oracle)).

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
- the Judge starts from the clean image, so what the Agent changed outside the
  WORKDIR does not reach it, unless the task sets
  [`clean_image = false`](#clean_image).

The WORKDIR must not equal, contain, or sit below Engine-owned mount targets
such as `/tests`, `/logs/verifier`, `/logs/agent`, or
`/run/rsi-harness/staging`.

When the effective WORKDIR is `/`, RSI Harness uses full-rootfs mode. Judge sees
the complete modified Work filesystem, but each submission requires a Docker
rootfs snapshot. Large files anywhere in `/` can therefore increase snapshot
time and disk use. A public task cannot use this mode.

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

### Shell and `PATH`

The Agent starts in `/bin/bash -c`, not a login shell, and the commands its
CLI runs keep its `PATH`: the image's `ENV PATH`, with the sandbox client
directory (`rsi-sandbox`) first when the task has a sandbox.
`/etc/profile`, `/etc/profile.d` and `~/.profile` do not run, so put the
tools the Agent needs on the image's `ENV PATH`, not in a profile script. The
Agent CLI, its node and `rsi-submit` are in `/usr/local/bin`, which that
`PATH` must include. The CLI runs on that pinned node even when the image puts
its own `node` earlier on `PATH`; the commands the Agent runs see the image's
`node`.

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

### `rsi-isolate`

Every Judge, hidden or public, local or on a cluster, can run code it must
not trust, such as a candidate's, through `rsi-isolate`. Nothing needs to be
declared. RSI Harness mounts it read-only at `/usr/local/bin/rsi-isolate`
(and `/opt/rsi-isolate`); it needs `python3` 3.8 or later in the Judge's
image. Use it instead of switching users (`setuid`, `su`, `setpriv`,
`chroot`, ...), which works only in a root Judge.

```bash
rsi-isolate run [--read PATH]... [--write PATH]... [--no-network] \
    [--timeout SEC] [--cwd DIR] [--env NAME[=VALUE]]... [--pass-fd FD]... \
    -- COMMAND [ARG]...
rsi-isolate isolation
```

- `--read PATH`: the command may read and execute PATH and everything below
  it. Otherwise it reads only system paths (`/usr`, `/bin`, `/sbin`,
  `/lib*`, `/etc`, `/opt`, `/proc`, `/sys`), the Python that runs
  `rsi-isolate`, its own executable and character devices. `/tmp`,
  `WORKDIR`, `/tests`, `/logs`, `/home`, `/root`, `/run` and `/var` stay
  closed unless granted, and so does a virtual environment outside `/opt`.
  A root Judge without Landlock confines it by permissions only (see
  Isolation below).
- `--write PATH`: the command may also create, change and delete below PATH.
  It always may write a private `HOME` and `TMPDIR`, deleted afterwards,
  and `/dev/shm`. In a root Judge what it left in `/dev/shm` is removed
  afterwards; in Landlock mode `/dev/shm` is shared with the Judge's user and
  later calls, so do not pass anything through it.
- Refused grants (exit `125`), checked on the resolved path: a missing path;
  a grant that is, contains or lies inside `/logs/verifier` or
  `/run/rsi-agent`; a write grant that is a top-level directory, overlaps a
  system path, `/dev`, `/tests`, `/logs/agent` or `/run/rsi-harness`, or
  holds a file with other hard links.
- `--no-network`: only Unix and `NETLINK_ROUTE` sockets, and no `io_uring`,
  by seccomp. Where the Judge shares its network namespace
  (`RSI_JUDGE_NETNS` is not `private`, as for a `public` Judge on a cluster),
  `socket(AF_UNIX)` is refused too unless Landlock ABI 6 scopes abstract
  sockets; a stream or seqpacket `socketpair` and `--pass-fd` still work.
  Without it the command has the Judge's network.
- `--timeout SEC`: for the whole call. On expiry the command's process
  group gets `SIGTERM`, then `SIGKILL` after 5 seconds. Whenever the command
  ends, everything it started is killed, even processes that left its group.
  In a root Judge they are killed at once, and a call runs at most 4096
  processes and threads. In Landlock mode a command that forks faster than
  its processes can be ended makes the call exit `125`.
- `--cwd DIR`: inside a grant; by default the private `HOME`.
- `--env NAME=VALUE` sets a variable and `--env NAME` passes the caller's
  on; the command gets no other variable but `PATH`, `HOME` and `TMPDIR`.
- `--pass-fd FD`: keeps one more inherited descriptor open, such as a
  socket to the Judge. stdin, stdout and stderr are always the caller's.

Exit status: the command's own (`128+N` when signal `N` ended it), or
`rsi-isolate`'s: `123` the command cannot be isolated; `124` the timeout
expired; `125` a usage error, a refused grant or a setup failure; `126` the
command cannot be executed; `127` it is not found; `128+N` `rsi-isolate`
received signal `N`. Stop a call with `SIGTERM`, not `SIGKILL`, so it can
end the command. A command can exit with these codes too, and write the
same stderr, so a `run` status never proves that `rsi-isolate` failed; treat
it as the command's outcome. To detect a Judge that cannot isolate, run
`rsi-isolate isolation` once before the first command: `123` there is an
infrastructure failure.

Isolation. When the Judge is root (local Docker by default), each call runs
as its own user with `no_new_privs`: uids 62000 to 62063, gids of the same
numbers, no supplementary groups. Up to 64 calls run at once; more wait
within their timeout. The `--write` paths belong to that user during the
call and are handed back afterwards, so concurrent calls must not share
one. The user reaches `--read` paths only through their permissions: each
path and its parent directories must be open to other users, or the call is
refused. A hidden model-access Judge's `/tests` is `0700`, so stage such
inputs in a `0755` directory. An image that uses those ids gets Landlock
instead. When the Judge is not root (every cluster Judge), the command
keeps the Judge's uid and Landlock (Linux 5.13 or later) enforces the
grants; a root Judge's command gets the same Landlock rules where the
kernel has them. Where it does not (`landlock_abi` 0), only file
permissions confine a root Judge's command: it reads whatever other users
can, and every call fails (exit `123`) while `/tests` or `/logs/verifier`
is open to other users (a root Judge can `chmod 0700 /tests` first). A
Landlock-confined command can, below Landlock ABI 6, signal the Judge,
and at any ABI `chmod` the Judge's files, which can only fail the round.
Without root or Landlock the command runs in its own user, mount, PID and
IPC namespaces (`userns`): only the system paths and its grants are shown,
everything but the write grants is read-only, and it sees no other process.
Where those namespaces are unavailable too, every call fails (exit `123`)
unless the operator chose the same-user fallback (see [Judges that are not
root](#judges-that-are-not-root)). In every mode the command gets no kernel
keyrings and no netlink socket but `NETLINK_ROUTE`. `rsi-isolate isolation`
prints `{"mode": "uid" | "landlock" | "userns" | "same_user",
"landlock_abi": <int>, "fallback": "none" | "same-user"}`, and exits `123`
when a command cannot be isolated.

To run a Judge without RSI Harness, as task validation does, bind RSI
Harness's `src/rsi_harness/judge_agent` read-only at `/opt/rsi-isolate` and
its `rsi-isolate` at `/usr/local/bin/rsi-isolate`.

## `[metadata.rsi_harness.verifier]`

[`sample_tasks/science-ising3d`](../../sample_tasks/science-ising3d)
illustrates `visibility`, `model_access` and `judge_model`: its Judge, the
public science Judge library of OpenRSI-X's `science-task-agent` skill with
the task's own configuration, re-executes the candidate isolated, audits it
with the run's Agent and reviews it with the Judge model, both through
`rsi-agent`. It is a demonstration, not a template to copy; it runs locally
and on a cluster.

### Judges that are not root

A Judge's Verifier user need not be root, and every cluster Judge runs as
the invoking user. Code in such a Judge that isolates what it runs cannot
switch to other uids: `rsi-agent` (see [`model_access`](#model_access)) and
[`rsi-isolate`](#rsi-isolate) confine what they run with Landlock (Linux
5.13 or later) instead, or else user namespaces (`userns`), and a science
Judge library a task carries in its `tests/` may do the same for the routes
it runs. Without any of them, isolation fails closed unless the operator set
`judge_isolation_fallback = "same-user"` in the `[apptainer]`
table of the cluster profile, which runs what would be isolated as the
Judge's user without filesystem confinement.

RSI Harness passes that choice to every Judge as
`RSI_JUDGE_ISOLATION_FALLBACK`: `"none"` unless the operator chose
`"same-user"`, always `"none"` on local Docker, and always set, so an `ENV`
in the task's image cannot choose it. `rsi-agent` takes it from its launch
spec, not the environment; `rsi-isolate` reads the variable, so a Judge
procedure must not set it. A task cannot set it either: the compiler refuses
any `RSI_JUDGE_*` name in the task's environment tables.

None of this depends on the image a Judge starts from. RSI Harness mounts
`rsi-agent` and `rsi-isolate` read-only from outside the image and sets the
variable on every Judge alike: a hidden Judge from the clean image (the
default) or from a snapshot of Work ([`clean_image = false`](#clean_image),
or a `WORKDIR` of `/`), and a public Judge with its read-only root
filesystem. Only the `python3` that runs them comes from the image, and
locally also the Agent CLI `rsi-agent` calls: in a snapshot of Work the
Agent can change them (see [`model_access`](#model_access)).

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

Because Work can read the Judge procedure, a public Judge is stricter than a
hidden one. It starts from the clean image (see [`clean_image`](#clean_image),
which has no effect here) with a read-only root filesystem and a 256 MiB
`noexec` `/tmp`: run a file written there through its interpreter
(`python3 file.py`); executing it or loading a compiled library from it fails.
A public task needs a `WORKDIR` other than `/`; setup fails otherwise.

A public task should not rely on secrets in `tests/`. Values the Agent must not
see belong in `[verifier.env]` runtime templates, which only the Judge
receives. The Agent prompt states that `/tests` is public.

### `clean_image`

```toml
[metadata.rsi_harness.verifier]
clean_image = false
```

- Type: boolean.
- Default when the table or field is absent: `true`.
- `true`: each hidden Judge starts from the clean image Work started from (the
  task image plus the Agent CLI), not from a snapshot of Work. Its own writes
  go to a container layer that is thrown away after the round. It reads
  Work's `WORKDIR` read-only; nothing the Agent installed or changed elsewhere
  reaches it, so a submission must live in `WORKDIR`.
- `false`: each hidden Judge starts from a snapshot of Work, with everything
  the Agent changed outside `WORKDIR`. Use it only when the Judge needs those
  changes: the Agent can then also change the programs that judge it.

When the `WORKDIR` is `/`, a hidden Judge cannot use the clean image and runs
from a snapshot of Work instead. The run prints a warning, and `run-plan.json`
records `clean_image: false` next to `rootfs_snapshot_mode: "full-rootfs"`.
On a cluster, every Judge starts from the task image with a copy of the
workspace, whatever this field says, so the Agent's changes outside the
workspace never reach it there and `run-plan.json` records
`clean_image: true`. A cluster Judge also runs as the invoking user, not
root (see [Judges that are not root](#judges-that-are-not-root)).

### `model_access`

```toml
[metadata.rsi_harness.verifier]
model_access = true
```

- Type: boolean.
- Default when the table or field is absent: `false`.
- `true`: each Judge can call the run's own Agent, for example to review a
  submission against a rubric, through `rsi-agent`.

`rsi-agent` is the only supported way for a Judge to call a model.
`rsi-agent exec` runs the run's Agent (the same CLI, credentials, provider or
proxy, model and reasoning effort as Work) on one prompt and writes its final
message, so a task works with every Agent RSI Harness supports and needs no
change for a new one. The Judge procedure never calls a model API or an Agent
CLI itself, and never sees the credentials.

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
is unavailable (no `model_access = true`, the Agent CLI is missing, the
run's Agent cannot run one-shot, the Agent cannot be isolated: no root,
no Landlock and no operator fallback, or the local login's access token
expires before the call could end: by `--timeout`, or by the round's end if
that comes first); `124` the timeout expired; `125` an
internal setup failure; `143` rsi-agent was terminated. Credential values
never appear in its output, the final message, the log or stderr.

The Judge receives `rsi-agent` from RSI Harness, read-only, at
`/usr/local/bin/rsi-agent` (and `/opt/rsi-agent`); it is not part of the
Work image, and Judges without `model_access` or
[`judge_model`](#judge_model) do not have it. It needs `python3` 3.8 or later
in the Judge's image, which setup checks.

Isolation. When the Judge runs as root, each call runs the Agent as its own
unprivileged user with a fresh private `HOME` holding copies of a local
login's files, and with `no_new_privs`. Calls may run concurrently, up to 16
at a time (more wait for a free slot within their timeout), and no two
concurrent calls share a user: uids 61000 to 61015 and gids of the same
numbers, which must not exist in the image (setup checks). That user cannot
write `/logs/verifier` or the reward, cannot read rsi-agent's launch spec or
the credential store, cannot read a hidden Judge's `/tests` (mode `0700`
there), and cannot read, change or signal another call's files or
processes. It can read `WORKDIR`, `/logs/agent`, a public Judge's `/tests`,
its working directory and anything world-readable, so stage evidence the
Agent must read accordingly. Its network is the Judge's. When a call ends,
every process it left running is killed, including any that left the
Agent's process group.

A Judge whose Verifier user is not root (every cluster Judge, for example)
confines each call with Landlock instead: the Agent keeps the Judge's uid,
may write only its own job directory (its `HOME` and `TMPDIR`) and
`--workdir`, and cannot read the launch spec, the login files, another
call's job, `/logs/verifier` or a hidden Judge's `/tests`; `/tmp` and
`/logs/agent` stay readable. Below Landlock ABI 6 such an Agent can still
signal the Judge, and at any ABI it can `chmod` files it cannot read; either
can only fail the round. Without root or Landlock the Agent runs in its own
user, mount, PID and IPC namespaces (`userns`), where those paths are hidden
and only its job directory and `--workdir` are writable. Where those
namespaces are unavailable too, every call fails (exit `3`) unless the
operator chose the same-user fallback. `rsi-agent isolation` prints how
calls are isolated as JSON, `{"mode": "uid" | "landlock" | "userns" |
"same_user", "landlock_abi": <int, 0 without Landlock>, "fallback": "none" |
"same-user"}`, and exits `3` when the Agent cannot be isolated.

The Judge also has:

- **Network:** the Judge reaches the Agent provider or proxy endpoints that
  RSI Harness pinned for Work, at the same addresses, named in the Judge's
  `/etc/hosts`. Every other destination follows `[verifier] network_mode`;
  with `"no-network"` it is rejected, and the Judge cannot resolve any other
  name: its DNS server is an address the firewall rejects. Model access never
  opens general egress. On a cluster the Judge uses the network the Work
  Agent uses for its model: with a `"no-network"` Agent, a private proxy to
  the same pinned endpoints (the Verifier must then be `"no-network"` too);
  with a `"public"` Agent, the site network, whose egress the site governs,
  not the Harness firewall, so the Judge procedure itself then has network
  access. With the private proxy, its Unix socket serves only the relay
  process that exposes it on loopback, so another process of the Judge that
  finds the socket (Landlock below ABI 7 does not control connecting to one)
  is turned away; the loopback port reaches only the pinned endpoints, and
  anything in the Judge that may open internet sockets can use it. Site
  egress changes only the network: the Judge keeps the binds, caches and
  Hugging Face environment of its declared `network_mode`. Clusters do not
  support `"allowlist"`, so an `"allowlist"` Agent or Verifier is refused
  there.
- **Environment:** of the Agent's settings, the test command receives only
  `RSI_AGENT_NAME`, `RSI_AGENT_MODEL` and `RSI_AGENT_REASONING_EFFORT` (when
  set), for records; the task's `[environment.env]` and `[verifier.env]`
  reach it as usual. The Agent's API key, base URL and proxy reach only
  `rsi-agent`.
- **Local login** (`--agent-auth local`): before each round the Judge gets
  the Work Agent's current login files (the copy Work's Agent keeps
  refreshing), placed in a private tmpfs that only root (or a non-root
  Verifier user) reads; `rsi-agent` copies them into each call's private
  `HOME`. Their refresh token is withheld, so a Judge never refreshes, and
  never rotates away, Work's login. A missing or malformed file (a new
  credential that is not token-shaped, at least 32 characters, included),
  or another login or account, is not used: the round keeps the last good
  copy and the run prints a warning. They are not part of any snapshot.
- **Redaction:** these credential values, a round's renewed login
  included (its token fields only, and the run-start login plus the last
  four renewed logins), are replaced with `[REDACTED]` in Judge output,
  feedback, events and errors.

`[verifier] timeout_sec` bounds the whole test command, every `rsi-agent`
call included: size it to cover the procedure's own work plus its model
calls, and give each call a `--timeout` that ends before the procedure's own
deadline.

Setup fails if no endpoint can be pinned: configure the Agent API base URL or
a proxy, or use an Agent with a registered default API URL. Cluster runs
support model access on a single-node Judge. Before submitting a job they
refuse model access on a multi-node Judge, with an Agent that is neither
`"public"` nor `"no-network"`, and with a `"no-network"` Agent whose
Verifier is not `"no-network"`.

A Judge that runs from a snapshot of Work (`clean_image = false`, or a
`WORKDIR` of `/`) runs programs the Agent can change, including the Python
interpreter and the model CLI that `rsi-agent` uses. For `model_access` the
Judge holds only the credential the Agent already has, but such a review
procedure should not trust Work-installed tools to report its verdict. A
[`judge_model`](#judge_model) with the operator's own credential is refused
there.

### `judge_model`

```toml
[metadata.rsi_harness.verifier]
judge_model = "gpt-6-sol"
judge_model_effort = "medium"
```

- `judge_model`: a model id. Each Judge can then call this model, independent
  of the Work Agent, through `rsi-agent llm`. There is no default model.
- `judge_model_effort`: optional; `minimal`, `low`, `medium` (default),
  `high`, `xhigh`, `max` or `ultra` (sent as `max`, as Codex does). The
  model must accept a reasoning effort.
- Independent of `model_access`: a task may declare either or both.

```bash
rsi-agent llm --prompt-file FILE --output FILE [--schema FILE] \
    [--timeout SEC] [--log FILE]
```

One plain call: the prompt in, the model's final text out, with a short
neutral system prompt and no tools. `--schema` asks for strict JSON Schema
output (the schema must follow the OpenAI strict-mode rules). Size limits,
`--timeout`, `--log`, redaction and exit statuses are those of `rsi-agent
exec`; `3` also means the task declares no `judge_model`, the Codex CLI is
missing, the Judge image has Codex system config (`/etc/codex/config.toml`,
`requirements.toml` or `managed_config.toml`), or a Codex login expires
before the call could end. With a Codex login the prompt is at most 1 Mi
characters (`2` otherwise). The procedure sees `RSI_JUDGE_MODEL` and
`RSI_JUDGE_MODEL_EFFORT`, for records.

The credential comes from the run operator (`rsi-harness run
--judge-llm-auth`), never from the task:

- `api`: an OpenAI API key; `rsi-agent` posts to `{base}/v1/responses`,
  following no redirects.
- `codex-login`: a Codex login made with a ChatGPT plan; `rsi-agent` runs the
  Harness's pinned Codex CLI (`codex app-server`, an ephemeral thread with
  every tool off and a one-model catalog that asks for none). Locally the CLI
  is copied from a Codex image built for the task (on a cluster, see below),
  used only when its SHA-256 is the pinned release's, and mounted read-only
  at `/opt/rsi-agent-codex`, not added to the Work image. The Judge gets the
  access token but not the refresh token, so it never refreshes the login
  and cannot spend the
  operator's refresh token. Each call gets a private copy of the login,
  discarded afterwards.

  The Harness keeps that login fresh instead. Before each Judge round it
  re-reads `auth.json` (with the run-start checks); when the access token
  would expire within the round's Verifier timeout plus 15 minutes, the
  pinned CLI (its SHA-256 checked again) runs `codex app-server` and asks
  `account/read` with `refreshToken: true`: as the operator (the `sudo` user
  under `sudo`), with no user config (a private `CODEX_HOME` that links only
  `auth.json`; a system-wide `/etc/codex` config still applies). It normally
  takes a second or two; Codex gets a minute to answer, then two more
  minutes to save a refresh it has already sent, and is stopped after about
  three minutes. Codex saves the rotated tokens to `auth.json` in place, so
  the operator's other Codex sessions keep working; two runs sharing one
  login can still race, as two Codex processes can. A login a Work Agent
  holds a copy of (this run's or another's, through `--agent-auth local`)
  must not be used: the refresh spends the refresh token that copy holds,
  and that Agent is logged out. This run refuses its own Work Agent's login;
  leave out `--judge-llm-auth` so Work lends it instead. If the refresh
  fails, the round keeps its copy, the run prints a warning, and calls the
  access token would not outlive exit `3`.

Without `--judge-llm-auth`, a Codex Work Agent with a ChatGPT login, or with
an API key on OpenAI's own API, lends its credential, whatever model Work
uses; one pointed at another base URL lends nothing. A lent ChatGPT login is
renewed from Work before each round, as for model access (Work's Codex
refreshes it). Otherwise the run fails before the Agent starts. An operator
credential also needs the clean Judge image: a hidden Judge from a snapshot
of Work (`clean_image = false`, or a `WORKDIR` of `/`) is refused before the
Agent starts.

Isolation is that of `rsi-agent exec`, with `rsi-agent`'s own worker as the
child: its own uid with a private `HOME` in a root Judge, Landlock in a Judge
that is not root. The credential is never in the procedure's environment; it
sits in `rsi-agent`'s launch spec on the private tmpfs (which a non-root
Verifier user can read, as for model access) and reaches the worker on its
stdin. The Judge reaches only the API base URL (or the proxy), or
`chatgpt.com` for a Codex login, pinned as for model access; general egress
still follows `[verifier] network_mode`.

On a cluster, a Judge LLM works the same way, except:

- It needs a single-node Judge and a `"public"` or `"no-network"` Verifier.
  Without `model_access`, a `"public"` Verifier's Judge uses the site
  network; a `"no-network"` one reaches only the pinned endpoints through
  its private proxy, so no upstream proxy and an `https` base URL.
- The submit host checks the credential; the job, where the Engine runs the
  Judge rounds, reads it again (an API key from the same variable, which
  the scheduler passes on, or a key file on shared storage). Only where it
  comes from reaches the run directory, so URLs with credentials in them
  are refused.
- The Codex CLI is the pinned release found where `codex` resolves on the
  submit host (`npm install -g @openai/codex@<pinned version>`); the job
  copies the one for its node's architecture to node-local scratch, checks
  its SHA-256 again and binds it read-only at `/opt/rsi-agent-codex`.
- An operator's Codex login is refreshed by the job, only when it sees the
  very `auth.json` the submit host read (same path, inode, owner and
  filesystem type, as on a shared home) and can write it and its directory.
  Otherwise the run fails before the Agent starts, since refreshing a copy,
  or a refresh that cannot be saved, logs the operator out: use an API key,
  or let a Codex Work Agent lend its login. Concurrent jobs take turns
  through a lock file, `.rsi-harness-login.lock`, beside `auth.json` (a
  filesystem without POSIX locks leaves them unserialized). A node that
  cannot reach Codex's auth server keeps the round's copy, as locally.

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
