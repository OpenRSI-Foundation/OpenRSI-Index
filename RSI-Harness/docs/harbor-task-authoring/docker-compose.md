# Compose, Dockerfile, and WORKDIR reference

RSI Harness accepts a deliberately narrow Docker Compose file at
`environment/docker-compose.yaml`. It uses only one service named `main` and
rejects settings that could bypass Engine-owned isolation or recovery.

This page first describes the Compose file of the RSI task itself (the
Work/Judge parent). A Harbor task run *inside* Work or Judge through a
brokered environment has a much wider Compose subset, with sidecars; see
[Compose inside brokered environments](#compose-inside-brokered-environments).

For `task.toml`, including Work and Judge GPU declarations, see
[`task-toml.md`](task-toml.md).

## Complete Compose example

```yaml
services:
  main:
    build: .
    working_dir: /testbed
    user: root
    shm_size: 8g
    environment:
      ORDINARY_SETTING: value
      PRIVATE_TOKEN: ${PRIVATE_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

The root document must contain only `services`. `services` must contain exactly
one entry named `main`. Sidecars are not supported.

## `services.main.build`

String form:

```yaml
services:
  main:
    build: .
```

Mapping form:

```yaml
services:
  main:
    build:
      context: .
```

- The build context must resolve to the exact local `environment/` directory
  that contains `docker-compose.yaml`.
- Parent directories, sibling directories, remote URLs, and Git contexts are
  rejected.
- Additional Compose build options such as `dockerfile`, `args`, `target`,
  `ssh`, and `secrets` are not supported by this parser.
- `build` and `image` cannot both appear in `services.main`.
- Compose `build` also conflicts with `task.toml
  [environment].docker_image`.

If neither Compose `build` nor any image field is present and
`environment/Dockerfile` exists, RSI Harness automatically builds that
Dockerfile using `environment/` as its context.

## `services.main.image`

```yaml
services:
  main:
    image: registry.example/task@sha256:...
```

- Type: image reference string.
- Prefer an immutable digest.
- This is an alternative to `build`.
- If `task.toml [environment].docker_image` is also present, both strings must
  be identical.
- An explicit image prevents RSI Harness from implicitly building a nearby
  Dockerfile.

## `services.main.working_dir` and Dockerfile `WORKDIR`

```yaml
services:
  main:
    working_dir: /testbed
```

`working_dir` must be an absolute POSIX path without `..`. It chooses the
directory in which Agent commands and Verifier commands run. The name is not
fixed: `/testbed`, `/repo`, `/workspace`, and `/app` are all ordinary examples.

The effective WORKDIR is chosen in this order:

1. `task.toml [environment].workdir`;
2. Compose `services.main.working_dir`;
3. image metadata set by Dockerfile `WORKDIR`;
4. `/` when no non-root path is declared by any layer.

If both the task and Compose declare a path, they must match. An explicit task
or Compose value overrides the image's default `WORKDIR`; it is not required to
match that image metadata.

### Does the directory have to exist?

Yes. A non-root effective WORKDIR must exist in the clean Base/Judge image.
RSI Harness checks this before starting the Agent.

Dockerfile `WORKDIR` creates the directory automatically:

```dockerfile
FROM registry.example/base@sha256:...
WORKDIR /testbed
```

The directory may instead be created without changing the image default:

```dockerfile
FROM registry.example/base@sha256:...
RUN mkdir -p /testbed
WORKDIR /
```

The second image may be paired with either of these declarations:

```toml
[environment]
workdir = "/testbed"
```

```yaml
services:
  main:
    working_dir: /testbed
```

This is valid because `/testbed` already exists in the image. Merely declaring
`/testbed` while the image has no such directory is rejected. The Engine does
not use a nonexistent arbitrary path as a valid empty repository.

### What happens to files already under the directory?

For a non-root effective WORKDIR, RSI Harness creates a fresh Engine-owned
Docker volume and mounts it at that exact path in Work. On the first Work
container, Docker copies the directory's existing image contents into the
fresh volume. For example, source code already stored under `/testbed` remains
available after the volume is mounted; it is not hidden by an empty volume.

The runtime behavior is:

- Work mounts the managed WORKDIR volume read-write.
- Agent commands run with the effective WORKDIR as their current directory.
- Each submission pauses Work and creates a CoW snapshot of the managed volume.
- Judge mounts that snapshot read-only at the same WORKDIR.
- Judge-private writes elsewhere in its disposable container are discarded.
- Judge cannot change Work files or the next Judge round.
- Large files under the WORKDIR stay out of repeated rootfs image snapshots.

Changes outside the WORKDIR are handled separately. RSI Harness captures the
modified Work root filesystem so Judge still sees packages, system libraries,
PATH commands, and other rootfs changes made by the Agent. Task-authored Docker
volumes and external services are not part of that snapshot model and are
therefore rejected.

The non-root WORKDIR must not overlap Engine-owned targets `/tests`,
`/logs/verifier`, or `/run/rsi-harness/staging`. It cannot be one of those
paths, an ancestor containing one, or a child below one.

### Root WORKDIR `/`

These declarations all select root mode:

```toml
[environment]
workdir = "/"
```

```yaml
services:
  main:
    working_dir: /
```

```dockerfile
WORKDIR /
```

If no layer declares a non-root WORKDIR, an empty image `WorkingDir` also
resolves to `/`.

Root mode does not mount a separate WORKDIR volume. Instead, every Judge round
uses a snapshot of the complete Work root filesystem. This provides complete
visibility but can be slow and disk-intensive when the Agent creates large
files. Prefer a real code directory such as `/testbed` when the task has one.

## `services.main.user`

```yaml
services:
  main:
    user: root
```

- Type: username or numeric UID.
- Optional.
- Used as the common task user when a phase-specific user does not override
  it, and included in image identity and preflight.
- `task.toml [agent].user` and `[verifier].user` are the phase-specific fields.
- RSI Harness currently gives the Work/Agent phase root authority so official
  Harbor tasks can change system libraries, shell commands, and PATH contents.
  Do not treat Compose `user` as an isolation boundary for Agent behavior.

The named or numeric identity must be valid in the image when it is used for a
phase preflight or Judge execution.

## `services.main.environment`

Mapping form:

```yaml
services:
  main:
    environment:
      ORDINARY_SETTING: value
      NUMERIC_SETTING: 7
      REQUIRED_FROM_HOST: ${REQUIRED_FROM_HOST}
      ALSO_REQUIRED_FROM_HOST:
```

List form:

```yaml
services:
  main:
    environment:
      - ORDINARY_SETTING=value
      - REQUIRED_FROM_HOST
```

- Both forms are supported.
- A missing value becomes `${VARIABLE_NAME}` and must be supplied by the Engine
  host at runtime.
- Literal values become ordinary task environment values.
- `${NAME}` and `${NAME:-default}` use the same runtime template rules as
  `task.toml`.
- When the same key exists in `task.toml [environment.env]`, the `task.toml`
  value wins.
- Template-derived secrets are injected only at execution time and redacted
  from Engine diagnostics.

Do not put real credentials directly in Compose.

## `services.main.shm_size`

```yaml
services:
  main:
    shm_size: 8g
```

- Type: positive integer with an optional `b`, `k`, `kb`, `m`, `mb`, `g`, or
  `gb` suffix, such as `1073741824`, `1g`, `4gb`, or `8192m`.
- RSI Harness default when absent: `1g`.
- Applied to Work and every Judge.
- Increase it for NCCL, multi-process PyTorch, DataLoader workers, browsers,
  databases, and other shared-memory-heavy software.
- `task.toml [environment].memory_mb` does not change `/dev/shm`.

## Work GPU reservation

```yaml
services:
  main:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
```

The nesting and values are strict:

- `devices` must contain exactly one entry;
- `driver` must be `nvidia`;
- `capabilities` must be exactly `[gpu]`;
- `count` must be a positive integer or the string `all`;
- `device_ids` is forbidden because physical selection belongs to the caller's
  ordered `--gpus` pool.

This declares Work GPUs, not Judge GPUs. Judge GPUs use:

```toml
[metadata.rsi_harness.verifier]
gpus = 2
```

If `task.toml [environment].gpus` and Compose `count` are both present, they
must match. Prefer a fixed count. `count: all` gives Work every GPU passed to
`--gpus`, leaving no disjoint spare GPU for Judge unless release-all mode is
used.

## Dockerfile behavior

The Dockerfile is ordinary Docker build input within the exact
`environment/` context. A typical file is:

```dockerfile
FROM registry.example/base@sha256:...

USER root
RUN install-system-packages
COPY project/ /testbed/
WORKDIR /testbed
```

Important details:

- `FROM` selects the clean task Base/Judge image lineage.
- `RUN`, `COPY`, and related instructions should install every dependency and
  place the initial repository in the image.
- `WORKDIR` both creates the directory and sets image metadata used when the
  task and Compose do not override it.
- Dockerfile `USER` is the image default and is used when task/Compose phase
  fields do not choose another identity.
- Dockerfile `VOLUME` is unsupported. Declared image volumes would hide data
  from the Engine's rootfs snapshot model, so image preflight rejects them.
- Do not bake runtime credentials into image layers.

RSI Harness derives an Agent-capable Work image from this clean Base by adding
the selected Agent tooling and submission client. Judge snapshots are derived
from the task/Work state; there is no separately authored Judge Dockerfile.

## Unsupported Compose fields

Only these `services.main` keys are accepted:

- `build`;
- `image`;
- `working_dir`;
- `user`;
- `environment`;
- `shm_size`;
- the exact NVIDIA `deploy.resources.reservations.devices` structure.

The following are explicitly rejected:

- `volumes` and arbitrary host mounts;
- `network_mode`, including host networking;
- `privileged`;
- `devices` outside the supported NVIDIA reservation;
- `cap_add`;
- `ipc`, `pid`, `links`, and `extra_hosts`;
- sidecar services;
- any unknown `services.main` key.

RSI Harness owns Docker networks, mount targets, GPU UUID selection, temporary
Judge test injection, and recovery labels. Rejecting these Compose fields keeps
that authority in one place rather than silently ignoring task settings.

## Compose inside brokered environments

A Harbor task that code in Work or Judge runs through the
`rsi_sandbox_harbor` plugin (or `rsi-sandbox compose`, see
[`sandboxes.md`](sandboxes.md#brokered-environments-metadata-version-2)) may
use sidecar services. Its Compose files are translated inside Work or Judge
into one environment specification; the host broker never runs `docker
compose` or reads YAML, and the specification cannot express host paths,
devices, extra privileges or host networking. A key that cannot be
translated fails with its key path (for example `services.db.cap_add:
unsupported compose key`) before anything is created; a key translated with
a change is reported as a note.

Loading follows Compose: YAML without tags (`!reset`, `!override` and custom
tags are refused), files merged in order (mappings recursively; `command` and
`entrypoint` replaced; `environment`, `extra_hosts`, `volumes` (by target),
`depends_on` and `networks` merged by key; `cap_drop` a union), variables
from `.env` below the process environment, the task/persistent env and
Harbor's `CONTEXT_DIR`, `PREBUILT_IMAGE_NAME`, `MAIN_IMAGE_NAME`, `CPUS` and
`MEMORY`; `$$`, `${V:-d}`, `${V-d}`, `${V:?e}`, `${V?e}`, `${V:+a}` and
`${V+a}` are supported. The endpoint's own `RSI_SANDBOX_*` variables are
never offered to interpolation. Profiles are honoured (`COMPOSE_PROFILES`).

Accepted:

- `image`, `pull_policy`, `platform: linux/amd64`, and `build` (`context`
  inside the project, `dockerfile`, `dockerfile_inline`, `args`, `target`,
  `network: default|none`, `no_cache`, `pull` and `labels`); builds need a
  builder grant;
- `command`, `entrypoint`, `environment`, `env_file`, `working_dir`, `user`,
  `group_add`, `hostname`;
- one network per project with `aliases`; `container_name` and `links`
  become aliases; `network_mode: none`; `extra_hosts` with IP literals
  (hostnames, `host-gateway` and scoped IPv6 addresses are refused);
- `healthcheck` and `depends_on` (`service_started`, `service_healthy`,
  `service_completed_successfully`, `required`);
- `read_only`, `tty`, `cap_drop` (`ALL` drops Docker's default set),
  `security_opt: [no-new-privileges:true]`;
- `cpus`, `mem_limit`, `pids_limit`, `deploy.resources.limits`, `shm_size`,
  `tmpfs`, `ulimits.nofile`; `deploy.replicas: 1`, `deploy.mode:
  replicated`, `ipc: private`, `cgroup: private` and `init` (an init process
  always runs);
- named and anonymous volumes (local driver, no options); image `VOLUME`
  paths get their own labelled volume;
- relative binds inside the project: a file is copied into the service
  before it starts, a directory becomes a volume seeded once (not read-only,
  not synced back);
- `configs`/`secrets` from `file:`, `content:` or `environment:`, copied in
  before start;
- `stop_signal`, `stop_grace_period` up to 30 s, `profiles`, `x-*`.

Normalized with a note: `ports` and `expose` are dropped (services reach each
other on container ports); `restart` becomes `no`; `memswap_limit` gives way
to the operator's `swap_ratio` (default: as much swap as memory, as in
Docker); reservations (`deploy.resources.reservations`, `mem_reservation`),
`labels`, `logging`, `develop`, `annotations`, `stdin_open`, `cpu_shares`,
`cpu_percent`, `deploy.restart_policy`, `deploy.labels` and other `deploy`
keys are ignored; Harbor's log mounts are dropped because logs are
downloaded; a service without limits gets the default CPU and memory; `ulimits.nproc` is
bounded by the pids limit.

Refused: `privileged: true`, `cap_add`, `devices`, `device_cgroup_rules`,
`gpus`, `runtime`, `isolation`, other `security_opt`, `sysctls`,
`oom_kill_disable`, `storage_opt`, `blkio_config`, `cpuset` and CPU quota
keys; `pid`, `uts`, `userns_mode`, `cgroup_parent`, host `ipc`/`cgroup`;
`network_mode` other than `none`; `volumes_from`, `external_links`;
`domainname` (it would change the service's FQDN); absolute, `~` or
out-of-project binds (including `/var/run/docker.sock`); any volume, bind or
`tmpfs` target at `/` or under `/proc`, `/sys`, `/dev` or `/run/rsi-harness`;
volume drivers, `driver_opts`, `external` volumes and `subpath`; more than one
network, external/internal networks, IPAM, fixed IP or MAC addresses, `dns*`
and `host-gateway`; more than one replica; `use_api_socket`, `provider`,
`models`, lifecycle hooks, `extends`, `include`; build `secrets`, `ssh`,
`network: host`, `additional_contexts`, `cache_from`/`cache_to`,
`entitlements`; any unknown key.
