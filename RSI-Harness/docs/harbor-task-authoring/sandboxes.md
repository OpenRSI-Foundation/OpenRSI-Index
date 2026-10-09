# Managed child sandboxes

Managed sandboxes let code in an RSI Harness Work or Judge container create
short-lived CPU-only child containers without receiving Docker authority. The
feature is opt-in twice: the task requests it in `task.toml`, and a trusted
operator approves that request with `--sandbox-policy`. Task metadata alone
grants nothing.

There are two kinds of grant:

- **Profiles** (metadata `version = 1`, the sections below up to "Validate
  an authoring change"): cached immutable images, no child network, no
  GPUs, a read-only root filesystem and bounded executable tmpfs scratch,
  on local Docker only.
- **Brokered environments** (metadata `version = 2`, see
  [Brokered environments](#brokered-environments-metadata-version-2)):
  pulled and built images, multi-service Compose environments, long-running
  interruptible execs, archive copies and optional egress, run by stock
  Harbor through the injected `rsi_sandbox_harbor` plugin. This is how
  Terminal-Bench-style tasks run inside Work or Judge. Envs run on local
  Docker, or as E2B sandboxes (see the operator guide's
  [E2B backend](../sandbox-operator-guide.md#e2b-backend)).

Neither kind exposes the Docker socket, the Engine API or Docker-in-Docker
to Work or Judge. The operator reference for both is
[`sandbox-operator-guide.md`](../sandbox-operator-guide.md).

## Task request

This complete `task.toml` fragment is intentionally small. Replace the example
digest with a real immutable digest reference. The operator policy repeats the
same profile; a request that differs from it or exceeds it is rejected, never
clamped.

<!-- sandbox-task -->
```toml
schema_version = "1.4"

[task]
name = "example/managed-sandbox"

[environment]
docker_image = "python:3.12-slim-bookworm@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
os = "linux"
workdir = "/workspace"
cpus = 1
memory_mb = 256
gpus = 0
network_mode = "no-network"

[agent]
timeout_sec = 300
user = "root"
network_mode = "no-network"

[verifier]
timeout_sec = 60
user = "root"
network_mode = "no-network"

[metadata.rsi_harness.sandbox]
version = 1

[[metadata.rsi_harness.sandbox.profiles]]
name = "offline-python"
image = "python:3.12-slim-bookworm@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
cpus = 1
memory_mb = 256
pids = 32
max_lifetime_sec = 120.0
workdir = "/workspace"

[metadata.rsi_harness.sandbox.profiles.tmpfs_mb]
"/workspace" = 32
"/tests" = 8
"/solution" = 8
"/logs" = 8
"/tmp" = 8
"/dev/shm" = 8

[metadata.rsi_harness.sandbox.work]
profiles = ["offline-python"]

[metadata.rsi_harness.sandbox.work.limits]
max_live = 2
max_created = 4
max_operations = 40
max_cpus = 2
max_memory_mb = 512
max_lifetime_sec = 240.0
max_upload_bytes = 8388608
max_download_bytes = 8388608
max_log_bytes = 1048576

[metadata.rsi_harness.sandbox.judge]
profiles = ["offline-python"]

[metadata.rsi_harness.sandbox.judge.limits]
max_live = 1
max_created = 2
max_operations = 20
max_cpus = 1
max_memory_mb = 256
max_lifetime_sec = 120.0
max_upload_bytes = 4194304
max_download_bytes = 4194304
max_log_bytes = 1048576
```

`work` and `judge` are independent grants; omit either table to deny that
phase. Every count, lifetime, byte allowance, CPU and memory ceiling is
finite.

- A child's lifetime is wall-clock time reserved at creation; it continues
  while paused and is not refunded by early destruction.
- `max_operations` counts create, exec, transfer and destroy attempts
  (status is free), so create/destroy churn cannot bypass admission; the
  phase and run totals bound it too.
- Upload and download ceilings are cumulative bytes. `max_log_bytes` is the
  cumulative exec stdout/stderr retained, and also bounds each call by what
  is left.
- A failed download charges the bytes actually transferred when that is
  known (a missing file does not consume the allowance), and a rejection
  proven to precede execution refunds its operation. Unknown or stopped
  transfers keep their reserved bytes.

The child runs as root inside its confinement: its image root is read-only
and only the declared tmpfs roots are writable (and executable, so uploaded
scripts and Harbor's Oracle/verifier scripts can run). A tmpfs at
`/workspace`, `/tests` or another root hides image content at that path; put
immutable fixtures elsewhere (for example `/opt/fixture`) and copy them into
scratch. This does not change the outer Work/Judge user; it applies to child
commands and to programs run through the nested Harbor adapter.

## Trusted operator policy

`--sandbox-policy` names an operator-owned **TOML** file (not agent-authored
JSON). It repeats every approved profile exactly, may grant larger phase
ceilings than the task asks for, and bounds the whole run and the host
admission pool. The resolved, non-secret grant is stored in the run plan and
lease.

<!-- sandbox-policy -->
```toml
version = 1
pool_cpus = 6
pool_memory_mb = 4096

[[profiles]]
name = "offline-python"
image = "python:3.12-slim-bookworm@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
cpus = 1
memory_mb = 256
pids = 32
max_lifetime_sec = 120.0
workdir = "/workspace"

[profiles.tmpfs_mb]
"/workspace" = 32
"/tests" = 8
"/solution" = 8
"/logs" = 8
"/tmp" = 8
"/dev/shm" = 8

[work]
profiles = ["offline-python"]

[work.limits]
max_live = 2
max_created = 6
max_operations = 80
max_cpus = 2
max_memory_mb = 512
max_lifetime_sec = 600.0
max_upload_bytes = 16777216
max_download_bytes = 16777216
max_log_bytes = 2097152

[judge]
profiles = ["offline-python"]

[judge.limits]
max_live = 1
max_created = 4
max_operations = 40
max_cpus = 1
max_memory_mb = 256
max_lifetime_sec = 300.0
max_upload_bytes = 8388608
max_download_bytes = 8388608
max_log_bytes = 2097152

[run_limits]
max_live = 3
max_created = 10
max_operations = 120
max_cpus = 3
max_memory_mb = 768
max_lifetime_sec = 900.0
max_upload_bytes = 25165824
max_download_bytes = 25165824
max_log_bytes = 4194304
```

Before the Agent starts, preflight resolves the digest reference to an exact
image ID without pulling it, checks that the image is cached and has no
anonymous `VOLUME`, and verifies Linux cgroup v2, runc, AppArmor, seccomp, and
CPU, memory, swap and PID enforcement. The parent Work/Judge containers must
have finite CPU and memory limits. Admission reserves both parents, both
child phase ceilings, 2 GiB of broker transfer headroom, and an allowance for
retained request and child metadata.

Run the task locally with Docker (the usual host firewall preflight and
privileges apply; version 1 children always use `network_mode=none`):

```bash
sudo -E "$(command -v rsi-harness)" run /absolute/path/to/task \
  --agent codex \
  --sandbox-policy /absolute/path/to/sandbox-policy.toml
```

## Using the broker

RSI Harness injects a phase-scoped `rsi-sandbox` executable (its directory is
prepended to the effective `PATH`) plus `RSI_SANDBOX_SOCKET` and
`RSI_SANDBOX_TOKEN`. Do not copy the credential or try to connect to Docker.
The version 1 operations are `capabilities`, `create`, `exec`, `upload`,
`download`, `status` and `destroy`. Handles and credentials are owner-bound:
Work cannot use Judge children, and every Judge round has a fresh owner and
destroys its children before teardown. A connection that does not read its
response is aborted after a bounded flush window. Disconnecting does not cancel
or release an unfinished operation. Work's connections close when Work is frozen
for a Judge round and reopen only after it resumes.

CLI example:

```bash
rsi-sandbox capabilities --json
rsi-sandbox create offline-python --lifetime 60 --request-id build-1 --json
rsi-sandbox upload CHILD_ID ./input --root /workspace --request-id input-1 --json
rsi-sandbox exec CHILD_ID --cwd /workspace --timeout 30 --json -- \
  /bin/sh -c 'python check.py > result.txt'
rsi-sandbox download CHILD_ID ./export --root /workspace --paths result.txt --json
rsi-sandbox status CHILD_ID --json
rsi-sandbox destroy CHILD_ID --json
```

The equivalent Python API uses the same injected endpoint and credential:

```python
from pathlib import Path

from rsi_harness.integrations.sandbox_client import (
    SandboxClient,
    iter_local_records,
    write_local_records,
)

client = SandboxClient()
child = client.create("offline-python", 60.0, request_id="build-1")
try:
    client.upload(child, "/workspace", iter_local_records(Path("input")))
    result = client.execute(
        child,
        ["/bin/sh", "-c", "python check.py > result.txt"],
        "/workspace",
        timeout_sec=30.0,
    )
    if result.exit_code != 0:
        raise RuntimeError(result.stderr)
    records = client.download(child, "/workspace", ["result.txt"])
    write_local_records(Path("export"), records)
finally:
    client.destroy(child)
```

Uploads and downloads accept only bounded regular files and directories rooted
in approved scratch; symlinks, hard links, special files, traversal and
unbounded bodies are rejected. Child files are not part of the Work snapshot,
run artifacts or the reward: download what must be judged or kept into the
Work workspace before `rsi-submit`.

For code already using Harbor's environment API, the supported adapter is
`rsi_harness.integrations.harbor_sandbox:ManagedSandboxEnvironment`. Configure
it with `kwargs={"profile": "offline-python"}`, `force_build=False`, and
`delete=True`:

```python
from harbor.models.trial.config import EnvironmentConfig

environment = EnvironmentConfig(
    import_path=("rsi_harness.integrations.harbor_sandbox:ManagedSandboxEnvironment"),
    kwargs={"profile": "offline-python"},
    force_build=False,
    delete=True,
)
```

After `Trial.create`, whole-task preflight is mandatory and must precede
`Trial.run`:

```python
from harbor.trial.trial import Trial
from rsi_harness.integrations.harbor_sandbox import preflight_managed_trial

trial = await Trial.create(trial_config)  # trial_config uses environment above
preflight_managed_trial(trial)
result = await trial.run()
```

Preflight checks the top-level and step Agent/verifier/collect-hook users,
Harbor's resolved network policies, and verifier placement: only
default/root inner users, `no-network` throughout, and a verifier in the
managed main environment. Calling `Trial.run` without preflight fails closed
before creating a child.

The adapter supports Harbor exec, file/directory upload and download, and
Oracle/verifier scratch conventions. It rejects Compose, bind mounts other
than standard log destination hints, storage limits without hard enforcement,
non-root users, GPUs, TPUs, MCP networking, and any network mode other than
`no-network`. It needs the broker endpoint injected by RSI Harness; it is not a
standalone Docker adapter.

## Failure and recovery

Errors are `permission`, `unsupported`, `invalid`, `busy`, `quota`,
`expired`, `unknown-outcome` or `infrastructure` (HTTP statuses in the
operator guide's [protocol](../sandbox-operator-guide.md#the-protocol)). Do not automatically retry
`exec` or another mutating request after `unknown-outcome`; inspect `status`,
destroy the exact handle when safe, or let recovery reconcile it.

Work children pause with their parent during submission, so their lifetime can
expire while paused. If the engine cannot prove that an expired paused child
was removed without executing it, it fails closed: the run stays contained,
the reservation is retained, and further sandbox or submission activity is
blocked. The same holds when the lease cannot be written: reservations and
recovery authority are kept, and nothing acts on an identity that is not
durably recorded. After fixing the host or Docker condition, run:

```bash
sudo -E "$(command -v rsi-harness)" recover RUN_ID
```

Recovery uses durable owner and container identity; do not rename or reuse
managed containers. On Docker versions where killing a paused container
implicitly thaws it, `recover` cannot reclaim an expired paused child and keeps
failing closed until an operator establishes its safe termination or absence;
it never unpauses the child or administers host cgroups to get there. Recovery
still contains every other child and running parent; unresolved resources and
their reservations, snapshots and networks stay retained.

## Validate an authoring change

```bash
.venv/bin/pytest -q tests/test_sandbox_docs.py
.venv/bin/pytest -q tests/runtime/test_sandbox_policy.py \
  tests/integrations/test_sandbox_client.py \
  tests/integrations/test_harbor_sandbox.py \
  tests/integrations/test_sandbox_compose.py \
  tests/integrations/test_harbor_env_plugin.py
```

The parser test validates the examples on this page with Harbor's task model
and RSI Harness's strict sandbox models. Full local acceptance also needs a
compatible Docker daemon, the cached image, and host firewall authority.

## Brokered environments (metadata version 2)

A version 2 grant lets code in Work or Judge (an agent, or a task's fixed
procedure that runs Harbor) create whole environments: one or more service
containers on a private bridge, with private volumes. All Docker authority
stays in the host broker; Work and Judge only talk to the phase socket. The
operator approves the grant once per run, not per task or image. The
operator guide lists every field of the grant
([Approve once](../sandbox-operator-guide.md#approve-once)), the request
([The task's request](../sandbox-operator-guide.md#the-tasks-request)) and an
env ([Environments](../sandbox-operator-guide.md#environments)).

### Task request

The task only states intent. Limits are optional and can only tighten the
operator's values; anything not granted is a `SetupError` before the run
starts, never a silent downgrade.

<!-- sandbox-env-task -->
```toml
schema_version = "1.4"

[task]
name = "example/harbor-in-judge"

[environment]
docker_image = "registry.example/judge@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
cpus = 2
memory_mb = 4096

[metadata.rsi_harness.sandbox]
version = 2

[metadata.rsi_harness.sandbox.environments.judge]
network = ["public"]
pull = true

[metadata.rsi_harness.sandbox.environments.judge.limits]
max_envs_live = 2
```

`network` lists what the envs need: `public`, `none` and/or `allowlist`.
`build = true` asks for Dockerfile builds; `build_network` (optional)
narrows their networks and defaults to `network` without `allowlist`, since
builds are `public` or `none`.

### Operator policy

The `[environments.<phase>]` tables of `--sandbox-policy` are the one-time
grant. A trimmed example (the operator guide documents every field, the
build and allowlist subtables, and what a run reserves from the pools):

<!-- sandbox-env-policy -->
```toml
pool_cpus = 64
pool_memory_mb = 131072

[environments.host]
pool_disk_mb = 40960
disk_floor_mb = 10240
disk_hard_floor_mb = 4096
request_slots_active = 8
request_slots_queued = 32
waiters = 64
no_new_privileges = true

[environments.judge]
network = ["public", "none"]
pull = true
registries = ["docker.io"]
max_envs_live = 4
max_envs_created = 400
max_services_per_env = 8
max_containers_live = 16
max_cpus_live = 16
max_memory_mb_live = 32768
max_disk_mb_live = 16384
cpus_per_container = 4
memory_mb_per_container = 8192
swap_ratio = 1.0
pids_per_container = 8192
disk_mb_per_container = 10240
max_env_lifetime_sec = 14400
max_wait_timeout_sec = 900
max_execs_running = 64
max_exec_output_bytes = 16777216
max_jobs_running = 4
max_pull_mb = 20480
max_log_bytes = 2147483648
max_upload_bytes = 17179869184
max_download_bytes = 17179869184
max_operations = 1000000

[environments.run_limits]
max_envs_created = 1000
max_operations = 2000000
max_builds = 256
max_pull_mb = 40960
max_upload_bytes = 34359738368
max_download_bytes = 34359738368
max_log_bytes = 4294967296
```

A `public` env reaches the internet through its own firewalled bridge,
with private, metadata and host addresses blocked; a `none` env has no
egress. Env disk is a soft limit (an env over its `disk_mb` fails with
`disk_quota`), and each service may swap `floor(memory_mb × swap_ratio)`
MiB (a task's `limits.swap_ratio` may only lower the operator's). See
[Environments](../sandbox-operator-guide.md#environments) for every EnvSpec
field.

### Allowlist envs

An `allowlist` env reaches only the destinations its EnvSpec lists, e.g.
`"allowlist": ["pypi.org:443", "files.pythonhosted.org:443",
"151.101.0.0/16"]`: exact hostnames, IPv4 addresses and CIDRs, each
optionally limited to one TCP port; everything else is rejected as in
`public` mode. The task requests `network = ["allowlist"]`; the operator
grants it with an `[environments.<phase>.allowlist]` table that bounds the
entries. Private, metadata and host addresses stay blocked unless the
operator lists them in `private_cidrs`. Hostnames are resolved by the broker
when the env is created and every `refresh_sec`, so fast-rotating CDNs,
shared addresses and DNS exfiltration are limits to know about; see
[Allowlist envs](../sandbox-operator-guide.md#allowlist-envs).

### Builds

A build uploads its context as a stage and names the Dockerfile inside it
(or passes it inline); `rsi-sandbox build DIR` and the plugin do both. The
context holds only files, directories and symlinks. The builder, its
lifetime and what a `none` build still fetches are in
[Builds and the builder exception](../sandbox-operator-guide.md#builds-and-the-builder-exception).

The result is an image handle of the session, usable in `env_create` without
further approval; an identical build in the session returns the same handle.
Releasing the handle, or the end of the session, removes the image. Build
errors are reported by kind (`dockerfile`, `oom`, `disk`, `timeout`,
`quota`, `infrastructure`, ...). A session holds at most max(64, 2 ×
`max_envs_live`) image handles, pulled and built, pending jobs included
(`quota` on `image` past that); a handle a live env uses cannot be released
(`busy`). Refused build options (syntax frontends outside the operator's
list, `BUILDKIT_*`/`BUILDX_*` args, `rsi-harness.*` labels, entitlements,
secrets, SSH) are listed in
[Builds and the builder exception](../sandbox-operator-guide.md#builds-and-the-builder-exception).

### Pulled images

A `missing` pull of an image already on the host under that reference
spends no pull budget, so name operator-pre-pulled images by digest (see
[Pre-pulling large image sets](../sandbox-operator-guide.md#pre-pulling-large-image-sets)). Pulled
images stay in the host's Docker cache after the run; the operator removes
the ones the sandbox first brought to the host with
`sudo -E "$(command -v rsi-harness)" sandbox prune-images` (see
[Pulled images](../sandbox-operator-guide.md#pulled-images)).

### The endpoint

Every Work and Judge exec that has a grant sees:

- `RSI_SANDBOX_SOCKET`, `RSI_SANDBOX_TOKEN`: the phase socket and its
  exec-only bearer credential;
- `RSI_SANDBOX_PYTHONPATH=/run/rsi-harness/sandbox/py`.

`/run/rsi-harness/sandbox` is a read-only bind holding exactly the socket
`s`, the `rsi-sandbox` CLI and `py/` with three modules: `rsi_sandbox_client`
(stdlib only), `rsi_sandbox_compose` (the Compose front-end; PyYAML, or JSON
files without it) and `rsi_sandbox_harbor` (the Harbor plugin; Harbor
`0.21.x`). Work and each Judge round have separate sessions: a handle from
one is invisible to every other.

### Running Harbor

Install `harbor==0.21.0` in the Work image (Judge derives from Work), then:

```bash
PYTHONPATH=$RSI_SANDBOX_PYTHONPATH harbor run \
  --env rsi_sandbox_harbor:ManagedSandboxEnvironment \
  -a oracle -p /tests/tasks/fix-git
```

or, from Python, `EnvironmentConfig(import_path=
"rsi_sandbox_harbor:ManagedSandboxEnvironment", delete=True)`. No preflight
call is needed; every check runs when Harbor constructs and starts the
environment. Optional kwargs (`--ek`): `socket_path`, `credential`,
`lifetime_sec` (default: the grant's), `on_cancel` (`kill-group`, the
default, kills a cancelled exec's process group; `detach` leaves it running
until the env ends, as stock Harbor does), `pull_policy`
(`missing`/`always`) and `inject_tmux` (`auto`, the default, or `off`; see
[Offline agents](#offline-agents)).

What the plugin does:

- The task's `docker_image` (or a Compose `image:`) is pulled by the broker
  (see [Pulled images](#pulled-images)). A task without one (or a Compose
  `build:`) is built from its context by the broker, within the smaller of
  the task's `build_timeout_sec` and the grant's `max_build_sec`; the built
  image is released when the environment stops with `delete`.
- `environment/docker-compose.yaml` and `--extra-docker-compose` files are
  merged in Harbor's layer order (resources, image, task file, extra files,
  main environment) with Harbor's interpolation variables, and translated into
  one env; see
  [Compose inside brokered environments](docker-compose.md#compose-inside-brokered-environments).
- `cpus`, `memory_mb` and `storage_mb` (soft disk) must fit the grant's
  per-container and live limits; they are never clamped. Sidecars without
  limits get at most 1 CPU and 1024 MiB, capped by the grant's per-container
  limits.
- Network `public` maps to a public env, `no-network` to `none` and
  `allowlist` to an allowlist env whose entries are Harbor's
  `allowed_hosts` (wildcard and IPv6 entries are refused, and so is a list
  longer than the grant's `max_entries`). Changing the policy between
  phases is refused.
- Relative Compose file binds and configs/secrets are copied into the
  created services before they start; directory binds become volumes seeded
  once.
- `exec` runs `bash -c` in `main` (or `sh -c` when the image has no bash),
  with Harbor's workdir, user and environment. Sidecar commands use `sh -c`
  without main defaults. A timeout kills only the command's process group and
  raises Harbor's `Command timed out after N seconds`.
- Files move through the Docker archive API (docker-cp semantics); logs are
  downloaded from `/logs/{agent,verifier,artifacts}`, which the plugin
  creates. Filtered and excluded downloads follow Harbor's semantics.
- The readiness wait of `start()` is `max_wait_timeout_sec`, or what is
  left of the env's lifetime when that is shorter (a Judge whose verifier
  timeout is below `max_wait_timeout_sec`); if queueing for a request slot
  used up the 1 s slack, the wait is measured again and sent once more.
- When the grant offers the operator's static tmux and `main` has none, the
  plugin has the broker copy it to `/usr/local/bin/tmux` before Harbor's
  agent setup.
- `stop()` destroys the env whatever `delete` says; `delete=True` also
  releases images the plugin built.

Differences from stock Harbor's Docker environment: cancellation kills the
process group by default; services run with `no-new-privileges` (operator
switch) and without `NET_RAW`; `ports`/`expose` are dropped and `restart` is
always `no`; every service has CPU and memory limits.

### Offline agents

An agent that installs its own tools needs the network unless the task
image ships them. For `terminus-2` in an env without network (Harbor
`no-network`, or an allowlist without the distribution mirrors):

- tmux: ask the operator to offer a static tmux (see
  [tmux for offline envs](../sandbox-operator-guide.md#tmux-for-offline-envs)); the plugin then
  copies it into each env whose image lacks one, and nothing in the task or
  its image changes.
- asciinema: terminus-2 records the session by default and first installs
  asciinema from the network, which fails offline (or waits out its install
  timeouts); terminus-2 then still types `asciinema rec` and a final Ctrl-D
  into the session. Pass `record_terminal_session=false`:
  `harbor run ... -a terminus-2 --ak record_terminal_session=false`, or
  `AgentConfig(name="terminus-2", kwargs={"record_terminal_session": False})`.

`sample_tasks/swebench-in-judge` is a worked example: three SWE-bench
Verified tasks with their image pinned, their `pip install` step dropped and
their grading made offline (see
[The SWE-bench sample](../sandbox-operator-guide.md#the-swe-bench-sample)).

Taking the network away does not remove what an image already holds. A
dataset whose answers can be looked up inside its own images, such as the
future commits of a SWE-bench repository in its `.git` history, must be
cleaned by its owner (rebuild the images without that history and pin them
by digest); the Harness runs task images as they are.

### Without Harbor

`rsi-sandbox` speaks the same protocol: `pull`, `build`, `up` (an EnvSpec
JSON), `ps`, `exec ENV_ID [--service S] -- ARGV`, `cp`, `stop-service`, `rm`,
`images`, `image-rm`, and `compose`, which runs a Compose project through the
same front-end as the plugin:

```bash
rsi-sandbox compose -f docker-compose.yaml up
rsi-sandbox compose -f docker-compose.yaml exec main -- pytest -q
rsi-sandbox compose -f docker-compose.yaml cp main:/app/out ./out
rsi-sandbox compose -f docker-compose.yaml down
```

`up` and `compose up` wait `--wait-timeout` seconds for readiness: by
default 300 s, or what is left of the env's lifetime when that is shorter.
An explicit `--wait-timeout` longer than what is left is refused with
`quota` (`wait_timeout_sec`) and the env is destroyed (`compose down` still
clears the project).

Compose has no egress allowlist; a project lists its entries in the
top-level extension `x-rsi-allowlist: [pypi.org:443, 151.101.0.0/16]` and
runs with `compose --network allowlist` (any other network refuses a
project that has one). Under Harbor, `allowed_hosts` and the extension's
entries are joined.

`compose config` prints the translated EnvSpec, the images it needs and the
notes, without contacting Docker. Between invocations the env handle of a
project lives in a 0600 file in `$TMPDIR`, keyed by the uid, the phase
session and the project name, so a project Work left up is never the one its
Judge addresses.
