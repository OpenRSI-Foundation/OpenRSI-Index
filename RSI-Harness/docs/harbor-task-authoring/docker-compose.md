# Compose, Dockerfile, and WORKDIR reference

RSI Harness accepts a deliberately narrow Docker Compose file at
`environment/docker-compose.yaml`. It uses only one service named `main` and
rejects settings that could bypass Engine-owned isolation or recovery.

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
