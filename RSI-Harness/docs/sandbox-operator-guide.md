# Managed Docker for Work and Judge: operator reference

Code running in an RSI Harness Work or Judge container (an agent, or a task's
fixed procedure in `/tests`) can use Docker through the host broker: pull and
build images, bring up multi-service environments ("envs"), run long,
interruptible commands in them, copy files in and out, and destroy them, all
within limits and with cleanup. The Docker socket, the Engine API and every
host path stay on the host: Work and Judge only see a per-phase Unix socket
that speaks the broker's own protocol.

This page is the operator reference: what one grant approves, every field
of it, what is refused, the builder exception, the disk model, names and
labels, the protocol, and how to verify a host. Task authors start with
[`harbor-task-authoring/sandboxes.md`](harbor-task-authoring/sandboxes.md)
(requests, the endpoint, Harbor) and
[`harbor-task-authoring/docker-compose.md`](harbor-task-authoring/docker-compose.md#compose-inside-brokered-environments)
(Compose inside envs).

The Harness only brokers environments: running Harbor, serving models and
scoring belong to the task's fixed procedure in `/tests`. The samples
[`harbor-in-judge`](../sample_tasks/harbor-in-judge),
[`vllm-in-judge`](../sample_tasks/vllm-in-judge) (see
[The vLLM-in-Judge demo](#the-vllm-in-judge-demo)) and
[`swebench-in-judge`](../sample_tasks/swebench-in-judge) (see
[The SWE-bench sample](#the-swe-bench-sample)) show that split.

## Host requirements

With the default Docker backend, `rsi-harness run` runs as root (`sudo -E`,
as for any production run) next to a rootful Docker daemon; the broker adds
no privilege beyond that. Root provides the iptables firewall on every env
and builder bridge (without it production refuses to start),
`cgroup.kill` of paused services and loop-ext4 builder state
(`state_fs = "tmpfs"` avoids loop devices). Builds also need the pinned
`moby/buildkit` image pre-pulled, and `losetup`/`mkfs.ext4` for loop-ext4.

The cluster backends (BlueVela/LSF and Slurm) refuse Docker-backed sandbox
requests before anything is provisioned: their nodes run Work and Judge
through Apptainer as the job user, with no Docker daemon or root component
to enforce per-env limits and firewalls. Run such tasks on a local Docker
host.

With `[environments.host] backend = "e2b"` the envs run as E2B sandboxes
and need neither the firewall nor root (see [E2B backend](#e2b-backend));
Work and Judge still run under local Docker, or under Apptainer on a
cluster (see [E2B on a cluster](#e2b-on-a-cluster)).

## E2B backend

With `backend = "e2b"` every env of a run is an E2B sandbox instead of local
containers. Grants, sessions, quotas, the journal, freeze and close
ordering, the protocol and the Harbor plugin are as on the rest of this
page, and tasks change nothing. To enable it:

1. Install the extra, which pins the SDK: `uv sync --extra e2b`
   (`e2b==2.52.0`).
2. Put the API key in a file only the harness user can read, and name it in
   the policy:

   ```toml
   [environments.host]
   backend = "e2b"
   # pool_disk_mb, disk_floor_mb, ... as for Docker

   [environments.host.e2b]
   api_key_file = "/home/op/.config/rsi-harness/e2b.key"  # or api_key_env = "E2B_API_KEY"
   # domain = "e2b.example.com"                           # a self-hosted E2B
   ```

   The broker reads the key once at run setup; the run plan, lease, sandbox
   metadata, capabilities and logs record only where the key is, never the
   key. `rsi-harness recover` and `cleanup` read it again from there.

How it works:

- Templates are made on demand. `image_pull` names a template from the
  image's digest (or its reference when it names no digest), the phase's
  `cpus_per_container` and `memory_mb_per_container` rounded up (1 or an
  even number of vCPUs, and 512 MiB steps) and a recipe version. If E2B has
  no template of that name, the broker builds it (the image, user root,
  workdir `/`, and a step that records the image's ENV) and waits; the
  pull's log shows the build. Later pulls and runs reuse the template; one
  broker builds a name only once at a time. Nothing is pruned.
- Each pull also reads the image's config from its registry (anonymously,
  through `proxy` if set; once per image and broker) for its WORKDIR, USER,
  ENTRYPOINT and CMD, which the template does not keep. A pull fails if the
  registry cannot be read.
- One env is one sandbox running its one service: `env_create` creates the
  sandbox, `env_start` starts the service command. Every sandbox carries the
  metadata `rsi_run_id`, `rsi_task_id`, `rsi_phase`, `rsi_round_id` (Judge),
  `rsi_env_id`, `rsi_service` and `rsi_role`. Its E2B timeout is the env's
  remaining lifetime plus 60 s, so E2B kills it if the broker dies, but at
  most `max_sandbox_hours`: E2B refuses a longer one, and ends an env that
  outlives it.
- Network `public` lets the sandbox reach the internet; `none` blocks it
  (E2B's firewall). Inbound traffic needs a token that nobody gets.
- Execs run through envd under `setsid`, without a login shell. They get
  the recorded image ENV, then the service spec's `env`, then the exec's.
  The cwd is the exec's, the spec's `working_dir`, the image's WORKDIR or
  `/`. The user is the exec's, the spec's, the image's USER or root;
  numeric uids map through the image's `/etc/passwd` and 0 is root. The
  service command is merged with the image's ENTRYPOINT and CMD as Docker
  does. Output, timeouts, `exec_kill` (to the group, through `kill` as
  root) and retention are the broker's own.
- `copy_in`, `copy_out` and `path_stat` run `tar` and `stat` as root in the
  sandbox, and `tool_install` uploads tmux the same way.
- `freeze_work` pauses Work sandboxes (memory kept). `resume_work` resumes
  them and sets their timeout again. `close_judge` and the run's close kill
  sandboxes; on E2B a kill is the removal. `rsi-harness recover` and
  `cleanup` list the run's sandboxes by `rsi_run_id`, running or paused,
  kill them and check that none is left. When E2B is unreachable they keep
  the lease for another try.

Refused with `unsupported` ("... unsupported on the e2b backend"): more than
one service (Compose), volumes and mounts, tmpfs, healthchecks, `read_only`,
`cap_drop`, `group_add`, `hostname`, `extra_hosts`, network `allowlist`, and
`image_build` (a task that requests builds fails setup). Not enforced:
`pids`, `nofile`, `shm_mb`, swap and the soft disk (`disk_mb` is charged but
not watched).

Fidelity and limits:

- E2B's template build adds packages to the image (systemd, sudo, git, curl
  and more), changes `/etc/profile`, and resets the image's USER, WORKDIR and
  ENV. The broker applies the ENV again (without the image's HOME, USER and
  LOGNAME), and USER and WORKDIR from the registry's image config. The
  image's healthcheck and stop signal are not applied. Images need a
  distribution E2B can provision (not scratch or distroless), `sh`,
  `setsid`, `tar` and `stat`, linux/amd64, and a public registry: no
  registry credentials are passed.
- Each sandbox gets the phase's per-container cpus and memory, rounded up,
  whatever its spec asks; quotas still charge the spec's values.
- The E2B account's limits apply: concurrent sandboxes, the maximum
  sandbox length (set `max_sandbox_hours` to it) and template cpus and
  memory. A paused sandbox never expires on E2B; recovery kills it.
- Images, seeds, the hidden tests and outputs go to E2B's cloud. A tag is
  resolved once, when its template is built.

### E2B on a cluster

The cluster backends (BlueVela/LSF and Slurm) accept an environment task
(version 2) when the operator's policy selects `backend = "e2b"`; any other
sandbox request is still refused before anything is submitted:

```bash
rsi-harness run TASK --cluster bluevela --sandbox-policy policy.toml ...
```

The submit host resolves the grant (the same two keys as locally) and
freezes it in the run plan. The broker runs inside the Engine process on
the compute node, as the job user, with no Docker and no root. Its
per-phase socket directory is on node-local scratch
(`$RSI_HARNESS_NODE_TMP/sb`, whose path must fit a 107-byte socket path) and
is bound read-only at `/run/rsi-harness/sandbox` into the Work Agent and the
Judge verifier, with the same `RSI_SANDBOX_*` variables as locally. The
Engine applies the local ordering around a Judge round: freeze Work's envs,
pause (SIGSTOP) the Work process group, run the round with its own
endpoint, kill the round's envs, then resume Work's envs, continue Work and
reopen its endpoint. The run's close kills every sandbox.

Requirements and limits:

- Compute nodes must reach the E2B API and sandboxes over HTTPS. Without
  direct egress, set `proxy = "http://proxy.example:3128"` in
  `[environments.host.e2b]` (no credentials in the URL; it is stored with
  the plan).
- The key is read on the compute node: use an `api_key_file` on storage the
  node can read, or an `api_key_env` the scheduler passes to the job.
- Stages and exec output spool on node-local scratch, which must keep
  `disk_floor_mb` free.
- A multi-node run may grant Work envs only: its Judge runs on remote hosts
  the socket does not reach, so a Judge grant is refused. In Work only the
  Agent (on the controller host) reaches the socket, not remote ranks.
- `rsi-harness recover --cluster NAME [RUN_ID]` and
  `rsi-harness cleanup --cluster NAME RUN_ID` kill the run's sandboxes by
  `rsi_run_id` (from the run's lease under the profile's `run_root`) once
  its job has ended; a run whose Engine still holds its lease is refused.
  The scheduler already ended Work and Judge. If nothing recovers a killed
  job, each sandbox's E2B timeout (its env's lifetime plus 60 s) ends it,
  except Work sandboxes paused for a Judge round, which only recovery kills.

## Approve once

A capability exists only when the task requests it and the operator's policy
(`--sandbox-policy PATH`) grants it; nothing is silently raised, lowered or
clamped. The grant is per run and per phase (Work, and each Judge round), not
per task or per image: images the phase pulls or builds are usable in its
envs without further approval. A request outside the grant is a `SetupError`
before the run starts. The effective grant is stored in the run plan and the
lease.

Granting builds accepts the builder exception below. Granting `public`
children to a `no-network` Work or Judge lets that phase reach the internet
indirectly (RFC1918, CGNAT, loopback, link-local and metadata addresses and
the host itself stay blocked). Granting `allowlist` children lets it reach
the hosts an env lists, within the operator's bounds (see
[Allowlist envs](#allowlist-envs)).

### `[environments.host]`

<!-- fields: EnvHostPolicy -->
| Field | Meaning |
|---|---|
| `pool_disk_mb` | Host disk pool of all runs; a run reserves every phase's live env disk plus its builders' state and image allowance |
| `disk_floor_mb` | Free space of Docker's root directory below which `env_create`, pulls, builds and loads are refused |
| `disk_hard_floor_mb` | Below it the largest envs are destroyed first |
| `request_slots_active` | Concurrent requests served per broker (default 8) |
| `request_slots_queued` | Requests waiting for a slot (default 32) |
| `waiters` | Concurrent long-polls (`job_wait`, `exec_wait`, `env_status`), which never hold a slot (default 64, at most 1024); raise it to the execs you expect to be awaited at once |
| `no_new_privileges` | Service containers run with `no-new-privileges` (default true) |
| `tmux` | The `[environments.host.tmux]` table: a static tmux the broker may copy into an env service whose image has none (see [tmux for offline envs](#tmux-for-offline-envs)); without it no tool is offered |
| `backend` | Where env services run: `"docker"` (the default, this page) or `"e2b"` (E2B sandboxes, see [E2B backend](#e2b-backend)) |
| `e2b` | The `[environments.host.e2b]` table; present exactly with `backend = "e2b"` |

### `[environments.host.tmux]`

<!-- fields: EnvToolFile -->
| Field | Meaning |
|---|---|
| `path` | Absolute host path of the binary, read by the broker only (keep it root-owned and writable by root only) |
| `sha256` | Its SHA-256 (64 lower-case hex digits); a file that differs is never copied |

### `[environments.host.e2b]`

<!-- fields: EnvE2BHost -->
| Field | Meaning |
|---|---|
| `domain` | The E2B domain; default the SDK's (E2B Cloud, `e2b.app`); set it for a self-hosted E2B |
| `api_key_file` | Absolute path of a file holding the API key, read by the broker only |
| `api_key_env` | Instead of `api_key_file`: the name of an environment variable of the `rsi-harness` process holding the key |
| `template_prefix` | Prefix of the template names the broker makes (default `rsi`) |
| `proxy` | An `http`, `https` or `socks5` proxy URL for every E2B call, without credentials (for cluster compute nodes without direct egress) |
| `max_sandbox_hours` | The account's maximum sandbox length in hours (default 1): no sandbox timeout exceeds it |

### `[environments.work]` and `[environments.judge]`

A phase without a table is not granted. Limits are ceilings per session:
the Work run, or one Judge round.

<!-- fields: EnvGrant -->
| Field | Meaning |
|---|---|
| `network` | Env networks allowed: `public`, `none` and/or `allowlist` |
| `pull` | Image pulls allowed |
| `registries` | Registries a pull may use (required with `pull`), e.g. `docker.io`, `public.ecr.aws` |
| `build` | The `[environments.<phase>.build]` table; without it nothing is built |
| `allowlist` | The `[environments.<phase>.allowlist]` table, required exactly when `network` has `allowlist` |
| `max_envs_live` | Envs alive at once (at most 128; see [Many envs at once](#many-envs-at-once)) |
| `max_envs_created` | Envs created over the session |
| `max_services_per_env` | Services of one env (at most 8) |
| `max_containers_live` | Service containers alive at once (at most 512) |
| `max_cpus_live` | CPUs of all live services |
| `max_memory_mb_live` | Memory of all live services; their swap together is at most `floor(max_memory_mb_live × swap_ratio)` (`max_swap_mb_live`) |
| `max_disk_mb_live` | Soft disk of all live envs |
| `cpus_per_container` | CPUs of one service |
| `memory_mb_per_container` | Memory of one service |
| `swap_ratio` | Swap of each service as a fraction of its memory, 0 to 1 (default 1: Docker's and stock Harbor's memory plus as much swap; 0 disables swap). A service gets `floor(memory_mb × swap_ratio)` MiB of swap on top of its memory |
| `pids_per_container` | PIDs of one service (also its default) |
| `disk_mb_per_container` | Soft disk of one service |
| `max_env_lifetime_sec` | Lifetime of one env; the env is destroyed when it ends |
| `max_wait_timeout_sec` | Longest `env_start` readiness wait; it must also fit in what is left of the env's lifetime |
| `max_execs_running` | Execs running at once (at most 512) |
| `max_exec_output_bytes` | Retained stdout and stderr of one exec, each; the rest is read and dropped |
| `max_jobs_running` | Pulls and builds running at once (at most 4) |
| `max_pull_mb` | Pulled image bytes over the session; a `missing` pull of an image already on the host spends none (see [Pre-pulling large image sets](#pre-pulling-large-image-sets)) |
| `max_log_bytes` | Exec and job output returned over the session |
| `max_upload_bytes` | Bytes staged into the broker over the session |
| `max_download_bytes` | Bytes copied out over the session |
| `max_operations` | Mutating operations over the session |

### `[environments.<phase>.build]`

<!-- fields: EnvBuildGrant -->
| Field | Meaning |
|---|---|
| `builder_image` | The BuildKit image by digest or ID (`moby/buildkit:v0.27.1`), pre-pulled by the operator; its entrypoint must be `buildkitd`, and the broker never pulls it |
| `network` | Networks RUN steps may use: `public` and/or `none` (builds have no allowlist: BuildKit pulls bases over the same bridge) |
| `cpus` | CPUs of the builder (all RUN steps together) |
| `memory_mb` | Memory of the builder (at least 1024) |
| `pids` | PIDs of the builder (at least 512) |
| `disk_mb` | Size of the builder's state filesystem |
| `state_fs` | `loop-ext4` (production: a fixed-size ext4 file on a loop device) or `tmpfs` (tests only, at most half of `memory_mb`) |
| `max_builds` | Builds over the session |
| `max_concurrent_builds` | Builds at once (1 to 4) |
| `max_build_sec` | Longest build; a build's deadline is the smallest of its own timeout, this and the session's remaining time |
| `max_context_mb` | Size of one build context |
| `max_image_mb` | Size of one built image; a larger export stops before it is loaded |
| `max_images_total_mb` | Built images the session holds at once |
| `syntax_frontends` | Repositories a `# syntax=` directive may name, e.g. `docker.io/docker/dockerfile` |

### `[environments.<phase>.allowlist]`

Bounds of the phase's allowlist envs; the env's EnvSpec names the entries.

<!-- fields: EnvAllowlistGrant -->
| Field | Meaning |
|---|---|
| `max_entries` | Entries of one env (default 16, at most 64) |
| `patterns` | If not empty, each hostname entry must match one of these hostname globs (`*.pypi.org`, `pypi.org`) and each IP or CIDR entry must lie within one of these CIDRs |
| `private_cidrs` | Private ranges (inside 10/8, 100.64/10, 172.16/12 or 192.168/16) entries may reach; without one, every private, CGNAT, link-local (metadata), loopback and multicast address stays blocked even when listed. Never name a range that holds Docker's address pools (here `172.16.0.0/12`): it would open other envs and the Work bridges |
| `refresh_sec` | Seconds between resolutions of an env's hostnames (default 60, 5 to 3600) |

```toml
[environments.judge]
network = ["public", "none", "allowlist"]
# ... the phase limits ...

[environments.judge.allowlist]
max_entries = 16
patterns = ["pypi.org", "files.pythonhosted.org", "151.101.0.0/16"]
private_cidrs = []   # e.g. ["10.20.0.0/16"] for an internal mirror
refresh_sec = 60
```

### `[environments.run_limits]`

Cumulative over Work and every Judge round of the run; no phase ceiling may
exceed its run ceiling.

<!-- fields: EnvRunLimits -->
| Field | Meaning |
|---|---|
| `max_envs_created` | Envs created by the run |
| `max_operations` | Mutating operations of the run |
| `max_builds` | Builds of the run |
| `max_pull_mb` | Pulled image bytes of the run |
| `max_upload_bytes` | Bytes staged by the run |
| `max_download_bytes` | Bytes copied out by the run |
| `max_log_bytes` | Output returned to the run |

The run also reserves, from the host pools: CPUs = 2 × parent CPUs + Σ phase
(`max_cpus_live` + build `cpus`); memory = 2 × parent memory + Σ phase
(`max_memory_mb_live` + build `memory_mb`) + 2048 MiB broker headroom +
broker metadata; disk = Σ phase (`max_disk_mb_live` + build `disk_mb` +
`max_images_total_mb`) from `pool_disk_mb`. `pool_cpus` and `pool_memory_mb`
are the policy's top-level pools. Swap is not reserved from
`pool_memory_mb`; the run's env services swap at most Σ phase
`floor(max_memory_mb_live × swap_ratio)`, so on a host whose swap is shared
with other workloads, size `swap_ratio` (or `max_memory_mb_live`) to what may
be taken from it. The sample policy
[`sample_tasks/harbor-in-judge/operator-policy.toml`](../sample_tasks/harbor-in-judge/operator-policy.toml)
reserves 28 CPUs, 51456 MiB and 65536 MiB of disk.

Broker metadata is 256 MiB, or more when the live limits are large: Σ phase
(64 KiB × `max_envs_live` + 512 KiB × (`max_containers_live` +
`max_execs_running` + image handles) + 8 MiB × `max_jobs_running` + 4 MiB)
+ 128 KiB × `host.waiters`, where a phase may hold max(64, 2 ×
`max_envs_live`) image handles. At every hard cap in both phases it is
1496 MiB.

## The task's request

`[metadata.rsi_harness.sandbox]` with `version = 2` and, per phase,
`[metadata.rsi_harness.sandbox.environments.<phase>]`:

<!-- fields: EnvPhaseRequest -->
| Field | Meaning |
|---|---|
| `network` | Env networks the task needs |
| `pull` | The task pulls images |
| `build` | The task builds images |
| `build_network` | Networks its builds need (default: `network` without `allowlist`; required when `network` is only `allowlist`); requires `build` |
| `limits` | Optional table of the phase limits above; each may only tighten the operator's value, an omitted one takes it |

## Environments

An env is one or more service containers on a private bridge with private
volumes, described by an EnvSpec. The broker validates it strictly (unknown
fields are refused); Harbor's plugin and `rsi-sandbox compose` build it from
Compose files inside Work or Judge, never on the host.

<!-- fields: EnvSpec -->
| Field | Meaning |
|---|---|
| `version` | 1 |
| `network` | `public` (egress through the env's firewalled bridge), `none` (an internal bridge; a single service gets no network at all) or `allowlist` (the public bridge, its egress limited to `allowlist`) |
| `allowlist` | With `allowlist` only: up to 64 entries, each a hostname (exact, no wildcard), an IPv4 address or an IPv4 CIDR, optionally `:port` (TCP only); hostnames are lower-cased without a trailing dot |
| `lifetime_sec` | Lifetime (default and ceiling: `max_env_lifetime_sec`) |
| `disk_mb` | Soft disk limit of the env |
| `volumes` | Named env volumes (at most 8), each `{seeded: bool}` |
| `services` | One to `max_services_per_env` services by name |

<!-- fields: ServiceSpec -->
| Field | Meaning |
|---|---|
| `image` | An image handle of this session (`i` + 32 hex), from a pull or a build |
| `entrypoint` | Replaces the image's entrypoint (and clears its command) |
| `command` | Replaces the image's command |
| `env` | Variables (at most 256; no `NVIDIA_*`); argv and env together at most 64 KiB |
| `working_dir` | Absolute working directory (default: the image's, cleaned as Docker cleans it, so `/testbed/` is `/testbed`) |
| `user` | User (and group) |
| `group_add` | Extra groups (at most 16) |
| `hostname` | A DNS label |
| `aliases` | Extra names on the env network (at most 16; the service name is always one) |
| `extra_hosts` | `[name, IP literal]` pairs (no `host-gateway`) |
| `network` | `env` or `none` |
| `read_only` | Read-only root filesystem (default writable) |
| `tty` | Allocate a TTY |
| `cap_drop` | Capabilities to drop (only narrows; `NET_RAW` is always dropped) |
| `cpus` | CPUs (0.01 steps) |
| `memory_mb` | Memory; the service may also swap `floor(memory_mb × swap_ratio)` MiB |
| `pids` | PIDs (default: `pids_per_container`) |
| `nofile` | Open files |
| `shm_mb` | `/dev/shm` size |
| `tmpfs` | Absolute path to size in MiB (at most 16; not `/`, `/proc`, `/sys`, `/dev/...`, `/run/rsi-harness`; tmpfs and shm within `memory_mb`) |
| `mounts` | Env volumes to mount (at most 16) |
| `healthcheck` | `image`, `none`, or a check of its own |
| `depends_on` | Start order and readiness conditions, without cycles |
| `stop_signal` | Signal that stops the service |
| `stop_grace_sec` | Seconds before the stop is a kill (at most 30) |

<!-- fields: EnvMount -->
| Field | Meaning |
|---|---|
| `volume` | A volume of the env |
| `target` | Absolute mount point (the tmpfs rules apply) |
| `read_only` | Mount read-only |

<!-- fields: EnvHealthcheck -->
| Field | Meaning |
|---|---|
| `test` | `["CMD", arg, ...]` or `["CMD-SHELL", command]` |
| `interval_sec` | Seconds between checks |
| `timeout_sec` | Seconds one check may take |
| `start_period_sec` | Grace period after start |
| `start_interval_sec` | Seconds between checks during the grace period |
| `retries` | Failures before unhealthy (at most 100) |

<!-- fields: EnvDependency -->
| Field | Meaning |
|---|---|
| `condition` | `started`, `healthy` (the target needs a healthcheck) or `completed_successfully` |
| `required` | A failed optional dependency does not fail the env |

`env_create` only creates (then files can be copied into created services);
`env_start` starts services in dependency order and the env is `ready` when
every service runs, every checked one is healthy and only
`completed_successfully` targets have exited 0, as `docker compose up
--detach --wait`. A failed start is a result of the env, not a broker
failure.

### The service template

Fixed for every service, and attested field by field before and after
start: runtime `runc`, not privileged, no added capabilities, `NET_RAW`
dropped, `no-new-privileges` (operator switch), AppArmor `docker-default`,
Docker's default seccomp profile, an init process, private IPC and cgroup
namespaces, no host namespaces, devices, device requests, published ports,
DNS overrides or sysctls, finite CPU, memory, swap (`MemorySwap` is memory
plus the grant's `swap_ratio` share of it; swappiness stays the kernel
default), PIDs and open files, logs to a 1 MiB `json-file` (never the host
journal), restart policy `no`, `NVIDIA_VISIBLE_DEVICES=void`, mounts only
of the env's own volumes and tmpfs, and images only by ID. Volumes are local volumes
without driver options (`type=none,o=bind` would reach host directories).

### Allowlist envs

An allowlist env gets the public-shaped bridge, but its firewall rule ends
in REJECT instead of ACCEPT. Ahead of the private-range rejects (after the
intra-bridge accept and the engine rejects), the rule jumps to a chain of
its own, `RSI_A_<hash>`, of ACCEPTs: one per IP or CIDR entry and one per
address a hostname entry resolves to, each limited to the entry's TCP port
if it names one. Everything else is rejected as in a `public` env; traffic
to the host itself (INPUT) is rejected whatever is listed.

- The broker resolves hostnames (IPv4, through the host's resolver) when the
  env is created, then every `refresh_sec` on a thread of its own. A new
  answer joins the addresses seen before (at most 16 per hostname, newest
  first); a failed lookup keeps them. When the accepts change, the broker
  appends the new chain and deletes the old rules from the top, so nothing
  beyond the old and new sets is ever accepted. A refresh makes no new
  chain or jump, so it journals nothing: the rule ID the lease already holds
  names all three chains, and recovery removes them as for any rule. A failed
  replacement quarantines the env.
- An entry, or an address a hostname resolves to, inside a private, CGNAT,
  link-local (metadata), loopback, multicast, `0/8` or `240/4` range or
  equal to an engine address is never accepted, unless it lies inside one
  of the operator's `private_cidrs`. `env_create` refuses such an IP or
  CIDR entry; a hostname that resolves only there, or not at all, is
  reported in the result's `notes`.
- DNS keeps working through Docker's embedded resolver, which forwards
  queries from the host's namespace; the env reaches no resolver directly
  (one is either private or not listed).

Limits:

- CDNs and round-robin DNS: the env resolves on its own and may get an
  address the broker has not seen yet; its connection is rejected until a
  refresh adds it. Hosts behind fast-rotating CDNs work only as well as the
  answers overlap; list a CIDR the operator's `patterns` admit instead.
- Shared addresses: an ACCEPT is per address, not per name. Anything else
  served from a listed host's address (other sites behind the same CDN
  edge, or the same IP's other ports for an entry without a port) is
  reachable too; TLS SNI is not checked.
- DNS: the embedded resolver answers any name, so names can still leak data
  through DNS queries (exfiltration), though no connection follows to an
  unlisted address.
- IPv6, wildcard hostnames and UDP-only ports are not supported (env
  bridges have no IPv6; the broker cannot enumerate a wildcard).

R11 below checks the blocking as root.

## Builds and the builder exception

A build uploads its context as a stage and names a Dockerfile in it or
passes one inline. Each phase session gets its own BuildKit daemon in a
broker-made builder container on its own firewalled bridge, the first time
it builds; Work's builder and its cache last the Work session, a Judge
round's builder is removed with the round. RUN steps share the builder's
network: public egress through the builder's rule, or none for a build with
`network: none`. A `none` build takes the network away from its RUN steps
only: BuildKit itself still fetches every `FROM` base, approved frontend,
`ADD <url>` and git source inside the builder, over its public bridge. The
host daemon never builds and never pulls for a build. The exported image is
checked (one image, config and layers present, no repository tags) and
loaded under the run's `rsi-sbx-img:` tag; releasing the handle, or the end
of the session, removes it.

The load goes to the daemon on a connection of its own, over Docker's Unix
socket (a run that grants builds needs a `unix://` Docker host; any other
fails its setup). Cancelling the build, its deadline or the session's end
cuts the stream within 0.1 s while its last chunk (the archive's end) has
not gone out, and the daemon refuses the truncated archive. Once it has,
the daemon may have the whole image and loads it even after the broker
hangs up, so the broker waits up to 30 s for its answer instead (within
the Judge close's 60 s) and removes the image it reports. A load that
never answers keeps its journaled digest (`leaked`): the image is removed
once it is found, and if it is still absent at the session's end, the run
fails closed and recovery's sweep of the run's labels removes it.

The builder is the one documented exception to the service template, only
for infrastructure: it runs BuildKit, never task code outside RUN steps.

<!-- builder-cap-add -->
Added capabilities: `CAP_SYS_ADMIN`, `CAP_NET_ADMIN`.

<!-- builder-security-opt -->
Security options: `apparmor=unconfined`, `seccomp=unconfined`,
`writable-cgroups=true`, plus `systempaths=unconfined` (empty masked and
read-only paths).

It is still `runc`, not privileged, with private cgroup and IPC namespaces,
one volume (its state filesystem at `/var/lib/buildkit`), no host path,
device, device request or port, finite CPU, memory and PIDs and
`NVIDIA_VISIBLE_DEVICES=void` (no NVIDIA device node). No Docker,
containerd or BuildKit socket reaches a RUN step: buildkitd v0.27.1 always
serves an OpenTelemetry trace collector and would bind its socket into every
step as `/dev/otel-grpc.sock`, so the broker pins it to
`/run/buildkit/otel-grpc.sock` and unlinks it before the first build. RUN
steps keep BuildKit's own seccomp profile, Docker's default capabilities and
a read-only cgroup filesystem. Without `seccomp=unconfined` BuildKit fails
on keyrings, without `CAP_NET_ADMIN` on `bpf_prog_query`. If BuildKit or
runc is exploited the builder is host root, as the Docker daemon's own
BuildKit would be.

Refused build options: build args named

<!-- build-refused-arg-prefixes -->
`BUILDKIT_*`, `BUILDX_*`

and labels named `rsi-harness.*` (the broker forces its own). The broker
builds with one fixed `buildctl build` and never passes

<!-- buildctl-never -->
`--allow`, `--secret`, `--ssh`, `--export-cache`, `--import-cache`, `push=true`

so there are no entitlements, secrets, SSH agents, registry credentials,
cache import or export, pushes, git or HTTP contexts or extra named
contexts. A syntax directive, in any form BuildKit reads (`# syntax=`, also
after a `#!` line, `// syntax=`, or a JSON `{"syntax": ...}` file), must
name one of `syntax_frontends`.

## Compose

Compose files are translated inside Work or Judge (Harbor's plugin,
`rsi-sandbox compose`); the broker never runs `docker compose` or reads
YAML. These are all the service keys the front-end understands, some only in
restricted forms (for example `privileged` only as `false`, `ipc` and
`cgroup` only as `private`, `network_mode` only as `none`, `platform` only
as `linux/amd64`); any other key is refused:

<!-- compose-service-keys -->
`image`, `build`, `pull_policy`, `platform`, `command`, `entrypoint`,
`environment`, `env_file`, `working_dir`, `user`, `group_add`, `hostname`,
`networks`, `network_mode`, `links`, `container_name`, `extra_hosts`,
`healthcheck`, `depends_on`, `init`, `tty`, `stdin_open`, `read_only`,
`privileged`, `cap_drop`, `security_opt`, `cpus`, `mem_limit`,
`memswap_limit`, `mem_reservation`, `cpu_shares`, `cpu_percent`,
`pids_limit`, `deploy`, `shm_size`, `tmpfs`, `ulimits`, `volumes`, `configs`,
`secrets`, `stop_signal`, `stop_grace_period`, `profiles`, `labels`,
`annotations`, `logging`, `develop`, `restart`, `ports`, `expose`, `scale`,
`ipc`, `cgroup`, `oom_kill_disable`

and these the keys of `build`:

<!-- compose-build-keys -->
`context`, `dockerfile`, `dockerfile_inline`, `args`, `target`, `network`,
`no_cache`, `labels`, `pull`

Refused, each with its key path, before anything is created (a service
`main` with these keys; the full list and the notes on what is normalized
are in [docker-compose.md](harbor-task-authoring/docker-compose.md#compose-inside-brokered-environments)):

<!-- compose-refused -->
| Service fragment | Why |
|---|---|
| `privileged: true` | Host authority |
| `cap_add: [SYS_ADMIN]` | Capabilities only narrow |
| `devices: ['/dev/fuse:/dev/fuse']` | Host devices |
| `gpus: all` | No GPUs in envs |
| `runtime: nvidia` | Always `runc` |
| `security_opt: ['seccomp=unconfined']` | Only `no-new-privileges` |
| `sysctls: {net.core.somaxconn: 1024}` | Kernel settings |
| `pid: host` | Host namespace |
| `ipc: host` | Host namespace |
| `userns_mode: host` | Host namespace |
| `cgroup: host` | Host namespace |
| `network_mode: host` | Host network |
| `network_mode: 'service:db'` | Another container's network |
| `volumes_from: [db]` | Another container's mounts |
| `volumes: ['/var/run/docker.sock:/var/run/docker.sock']` | Host path (the Docker socket) |
| `volumes: ['~/data:/data']` | Host path |
| `tmpfs: ['/proc']` | Reserved mount target |
| `dns: [8.8.8.8]` | DNS override |
| `mac_address: 02:42:ac:11:65:43` | Fixed address |
| `extra_hosts: ['host.docker.internal:host-gateway']` | The host |
| `environment: {NVIDIA_VISIBLE_DEVICES: all}` | GPU authority |
| `oom_kill_disable: true` | Unbounded memory |
| `scale: 2` | One container per service |
| `use_api_socket: true` | The Docker API |
| `provider: {type: model}` | Compose providers |
| `models: [llm]` | Compose models |
| `post_start: [{command: id}]` | Lifecycle hooks |
| `extends: {service: base}` | Other files |
| `platform: linux/arm64` | Only `linux/amd64` |
| `build: {context: ., secrets: [token]}` | Build secrets |
| `build: {context: ., ssh: [default]}` | SSH agent |
| `build: {context: ., network: host}` | Host network |
| `build: {context: ., additional_contexts: {extra: ../x}}` | Extra contexts |
| `build: {context: ., cache_from: [x]}` | Cache import |
| `build: {context: ., cache_to: [x]}` | Cache export |
| `build: {context: ., entitlements: [network.host]}` | Entitlements |

## Disk model

Env disk is a soft limit: every 10 s the broker reads each service's
writable layer (`SizeRw`), every 60 s its volumes, every 2 s the free space
of Docker's root directory. An env over its `disk_mb` fails with
`disk_quota` and is removed 60 s later, so it can overshoot by about ten
seconds of writes. Below `disk_floor_mb` new envs, pulls, builds and loads
are refused; below `disk_hard_floor_mb` the largest envs are destroyed. A
builder's state is a hard limit: a fixed-size ext4 file on a loop device
(sparse, so the host can be overcommitted), pruned after a failed build.
Pulled images are the host's cache: never counted against the pool and
never removed by a run; `rsi-harness sandbox prune-images` removes the ones
the sandbox first brought to the host (see [Pulled images](#pulled-images)).

## Names and labels

Every object carries `rsi-harness.run-id`, `rsi-harness.task-id`,
`rsi-harness.sandbox-phase` (and `rsi-harness.round-id` in a Judge round)
and one role:

<!-- roles -->
`sandbox-env`, `sandbox-env-net`, `sandbox-env-vol`, `sandbox-builder`,
`sandbox-builder-net`, `sandbox-builder-vol`, `sandbox-build`

| Object | Name |
|---|---|
| Env (`e` + 32 hex; `e16` its first 16) | containers `rsi-sbx-<e16>-<i>`, volumes `rsi-sbvol-<e16>-<i>`, bridge `rsi-sbnet-<e16>`, rule `rsi-<run>-sbx-<e16>` |
| Builder (`b` + 32 hex) | container `rsi-sbb-<b16>`, volume `rsi-sbbvol-<b16>`, bridge `rsi-sbbnet-<b16>`, rule `rsi-<run>-sbb-<b16>`, loop file `<data_root>/<run>/sb/build/<b16>.img` |
| Built image (`i` + 32 hex) | `rsi-sbx-img:<sha256(run)[:12]>-<32 hex>` |
| Firewall | chains `RSI_F_<hash>`/`RSI_I_<hash>` of the rule (and `RSI_A_<hash>`, an allowlist env's accepts); its jumps carry the rule id as comment |

Names are derived from the run and handle, and every mutation is journaled
before it happens, so `rsi-harness recover` rebuilds the whole cleanup plan
from the lease: builders, volumes, loop devices, loop files, bridges, rules,
built images by ID and tag, then a sweep by the run's labels. It also
removes the run's stage spool and, once no Work or Judge is left, the phase
endpoints a crash left under `<data_root>/<run>/sb` (exactly `s`,
`rsi-sandbox` and `py/` with the three endpoint modules; anything else there
fails closed and is left untouched), then `sb` itself. A v1 profile run's
endpoints are removed the same way.

## The protocol

HTTP/1.1 over the phase socket, `POST /v1/<op>` with `Authorization:
Bearer`, a JSON object with exactly these fields (binary frames for
`upload` and `stage_put`); every mutation takes a `request_id` and is
replayed, not repeated, when it comes again.

<!-- wire-ops -->
| Op | Fields |
|---|---|
| `capabilities` | |
| `create` | `profile`, `lifetime_sec`, `request_id` |
| `exec` | `child_id`, `argv`, `cwd`, `env`, `timeout_sec` |
| `upload` | `child_id`, `root`, `request_id`, `timeout_sec` |
| `download` | `child_id`, `root`, `paths`, `timeout_sec` |
| `status` | `child_id` |
| `destroy` | `child_id` |
| `stage_put` | `stage_id`, `offset`, `final`, `sha256`, `request_id` |
| `stage_get` | `stage_id`, `offset`, `length` |
| `image_pull` | `ref`, `policy`, `request_id` |
| `image_build` | `stage_id`, `dockerfile`, `dockerfile_inline`, `target`, `build_args`, `labels`, `no_cache`, `network`, `timeout_sec`, `request_id` |
| `job_wait` | `job_id`, `log_offset`, `wait_sec` |
| `job_cancel` | `job_id` |
| `image_list` | |
| `image_release` | `image` |
| `env_create` | `spec`, `request_id` |
| `env_start` | `env_id`, `wait_timeout_sec`, `request_id` |
| `env_status` | `env_id`, `wait_sec` |
| `env_stop_service` | `env_id`, `service`, `timeout_sec`, `request_id` |
| `env_destroy` | `env_id` |
| `env_list` | |
| `exec_start` | `env_id`, `service`, `argv`, `cwd`, `env`, `user`, `timeout_sec`, `merge_stderr`, `request_id` |
| `exec_wait` | `exec_id`, `stdout_offset`, `stderr_offset`, `wait_sec`, `max_bytes` |
| `exec_kill` | `exec_id`, `signal`, `scope` |
| `copy_in` | `env_id`, `service`, `dest_dir`, `stage_id`, `request_id` |
| `copy_out` | `env_id`, `service`, `path`, `max_bytes`, `exclude` |
| `path_stat` | `env_id`, `service`, `path`, `follow` |
| `tool_install` | `env_id`, `service`, `tool` |

The first seven are the version 1 profile operations. Errors are
`{"error": {"code", "field", "message"}}`:

<!-- error-codes -->
| Code | HTTP |
|---|---|
| `permission` | 401 |
| `unsupported` | 400 |
| `invalid` | 400 |
| `busy` | 409 |
| `quota` | 413 |
| `expired` | 410 |
| `unknown-outcome` | 502 |
| `infrastructure` | 503 |

An unknown operation is `unsupported`, and so is any other method or path
under `/v1/` or `/v2/`, Engine API paths such as `/v1/containers/json`
included: 400, before authentication and without reading the body.

## Operator steps

1. Pre-pull the builder and pin it:
   `docker pull moby/buildkit:v0.27.1`, then
   `docker image inspect --format '{{index .RepoDigests 0}} {{.Id}}' moby/buildkit:v0.27.1`
   and put the digest in `builder_image`.
2. Check the free space (`df -h /`), then pre-pull what the acceptance uses:
   `alexgshaw/{fix-git,regex-log,adaptive-rejection-sampler,kv-store-grpc,nginx-request-logging,git-multibranch}:20251031`,
   `python:3.13-slim-bookworm`, `redis:7-alpine`, `alpine:3.21`. Large
   image sets: [Pre-pulling large image sets](#pre-pulling-large-image-sets).
3. Make sure loop devices work: `modprobe loop`; `/dev/loop-control` exists.
4. Review and sign the policy (the one-time grant, including the builder
   exception and indirect internet access of `public` children).
5. Run the root checks, which include the acceptance (R10), all must pass:
   `sudo scripts/operator/sandbox_root_check.sh`
   (`--skip-acceptance` leaves R10 out and the table says so; the
   acceptance alone is `sudo scripts/operator/sandbox_acceptance.sh`)
6. Production runs as root (iptables needs it):
   `sudo -E "$(command -v rsi-harness)" run TASK --agent AGENT --sandbox-policy POLICY`
7. Pulled images stay cached after the run; remove the ones the sandbox
   first pulled with `sandbox prune-images` (see [Pulled images](#pulled-images)).

Both scripts print what they would do with `--dry-run`, as any user, and
act only on the runs they start. Their scratch directory is always new and
root's own: `mktemp -d` under `/var/tmp` by default, and a `--scratch DIR`
must not exist yet. A `--only` name that is no check or scenario is refused,
and a scenario that yields no verdict is a FAIL row. There is no non-root
production mode: the firewall probe (`iptables --wait -S DOCKER-USER`) needs
root, and the fake firewall exists only in tests.

### Pulled images

A run never removes a pulled image (that would need reference counting
across runs). Instead every successful brokered pull is recorded in the
host's pull ledger, `<data_root>/sandbox-images/pulled.json` next to
`<data_root>/leases` (owned by whoever runs the Harness, a 0700 directory
and a 0600 file replaced atomically under an exclusive `flock`, so runs
pulling at the same time never lose an entry). An entry
holds the image ID, the normalized references the sandbox put on the host,
the registry, the first and last run that pulled it, `first_pulled_at` and
`last_used_at`. Only an image the sandbox first brought to the host gets an
entry: one whose image ID was on the host before, under any name or none
(a digest-only pull, another tag, a dangling image), such as pre-pulled
images or anything pulled by hand, never does (the broker lists the host's
image IDs before each pull for this; the journal's `pre_existing` looks at
the pulled reference only). A later pull adds only a
reference the sandbox put on the host, or one it moved off an image whose
entry records it. A failed ledger write (or more than two seconds waiting
for its lock) never fails the pull; that image is then never pruned.

```bash
sudo -E "$(command -v rsi-harness)" sandbox prune-images --dry-run
sudo -E "$(command -v rsi-harness)" sandbox prune-images --older-than 7d
```

<!-- prune-images-options -->
`--dry-run` lists and changes nothing, `--older-than` takes a DURATION
(`45m`, `12h`, `7d`, `2w`) and selects by `last_used_at`, `--yes` skips the
confirmation (it shows the plan first, then the result under `Result:`),
and `--data-root` names the data root whose ledger and leases apply
(production: the runs'; a docker-group user can prune a test data root of
their own without root). `--logs-root` and `--verbose` are the other
commands'.

It prints one row per entry: image, size, last use, action (`would remove`
or `removed`, `kept`, `would drop` or `dropped`) and a bounded reason, then
the bytes freed (Docker's image sizes; layers other images share stay). The
rules:

- An image that any container on the host uses, in any state, is kept, and
  so is one that a lease in this data root holds as a present pulled
  handle (a lease that cannot be read keeps every image). Both are checked
  again right before each removal. The container check covers the whole
  host, the lease check only the runs of `--data-root`: do not prune a data
  root while runs under another data root use the same images.
- An image that would keep a reference the ledger did not record (another
  tag, or a digest of a repository it recorded no tag of) is kept and
  nothing of it is removed; the dry run says so too.
- Otherwise the recorded tags and the digests of their repositories are
  removed by name, each only while it still names the image (a tag that
  names another image by then is left alone), and Docker deletes the image with
  its last reference. Only an image left with no reference at all is
  removed by ID. Nothing is forced: a Docker conflict keeps the image.
- An entry whose image is gone is dropped from the ledger. An image without
  an entry is never touched.

A pull with policy `always` that moves a name someone else put on the host
(an operator's `python:3.13-slim-bookworm`, after the registry moved the
tag) gives the new image an entry without that name, so the name keeps it
until the operator removes or moves it; the old image keeps no entry. An
image the sandbox brought first stays prunable when someone later pulls the
same tag by hand; tag it under another name to keep it.

Docker has no conditional untag or delete, so a pull that races a prune
can lose its image (an untag may hit a freshly moved tag, a removal by ID
takes a just-added tag, a bound image may vanish before its env starts).
Prune while nothing pulls the recorded references.

### Pre-pulling large image sets

A benchmark of hundreds of task images (about 500 SWE-bench images, about
130 GB) is best pulled once, ahead of the runs, rather than by the broker
during them: pulls then never wait on the registry inside a phase's
deadline, the broker's anonymous pulls never hit a registry's rate limit,
and the images spend no pull budget.

1. Write a manifest, one `name@sha256:<digest>` per line (`#` comments
   allowed). A digest is what `docker image inspect --format
   '{{index .RepoDigests 0}}' IMAGE` prints for an image pulled once by tag
   (or `docker buildx imagetools inspect REF` without pulling);
   [`sample_tasks/harbor-in-judge/images.manifest`](../sample_tasks/harbor-in-judge/images.manifest)
   is an example.
2. Plan the disk. The images live under Docker's root
   (`docker info --format '{{.DockerRootDir}}'`), outside `pool_disk_mb`,
   and take their unpacked size (`docker system df` counts shared layers
   once), often twice the download. While the broker runs it refuses envs,
   pulls, builds and loads once that filesystem's free space falls below
   `disk_floor_mb`, so keep the set plus the runs' env disk
   (`pool_disk_mb`), builder images and `disk_floor_mb` within it.
3. Pull with plain `docker pull`, as root or a docker-group user, before
   the runs: `scripts/operator/prepull_images.sh [--dry-run] [--retries N]
   MANIFEST` skips what is already there, retries each failure after a
   growing pause, prints the free space before and after, and lists what
   still failed (exit status 1). A `docker login` of that user is used for
   these pulls only; the broker never sends host credentials.
4. Approve the registry in the policy: `pull = true` and its name in
   `registries` (Docker Hub is `docker.io`) for each phase that uses the
   images. `max_pull_mb` needs to cover only what the runs really download.
5. Name each image by the same digest in the task and pull with policy
   `missing` (the plugin's and `rsi-sandbox pull`'s default). With the
   example manifest's first line pre-pulled, a Harbor task names it as
   its `docker_image`:

   ```toml
   [environment]
   docker_image = "alexgshaw/fix-git@sha256:61e431c00c58df652287aadce5457634d9f9330cfdd153ebdf2802df0d540119"
   ```

   and the policy grants the phase that runs it:

   ```toml
   [environments.judge]
   pull = true
   registries = ["docker.io"]
   max_pull_mb = 1024  # what the runs really download; 0 is not allowed
   ```

   A fixed procedure without Harbor pulls the same reference itself:

   ```bash
   rsi-sandbox pull alexgshaw/fix-git@sha256:61e431c00c58df652287aadce5457634d9f9330cfdd153ebdf2802df0d540119
   ```

   The job succeeds with no `pulling` line in its log, and
   `usage.image_bytes` of `rsi-sandbox capabilities --json` does not
   change. For a SWE-bench set each task names its own line, e.g.
   `docker.io/swebench/sweb.eval.x86_64.<instance>@sha256:<digest>`
   ([`sample_tasks/swebench-in-judge/images.manifest`](../sample_tasks/swebench-in-judge/images.manifest)
   is the [SWE-bench sample](#the-swe-bench-sample)'s three).

A `missing` pull of an image already on the host under that reference
downloads nothing: the broker binds the cached image when its `RepoDigests`
prove it came from that repository with exactly that digest (a retagged or
host-built image never binds), and charges nothing to `max_pull_mb`, even
once the budget is spent. A pull that would download is refused (`quota`)
when no budget is left, before the daemon contacts the registry; so is
every `always` pull, at once. A tag reference works the same when the tag
is on the host, but a digest cannot be moved under a run.

`rsi-harness sandbox prune-images` never removes these images: they were
on the host first, so the pull ledger has no entry for them (see
[Pulled images](#pulled-images)). Remove the set yourself while no run uses
it, e.g. `sed 's/#.*//' MANIFEST | xargs -r docker image rm`.

### Many envs at once

The hard caps per session (the Work run, or one Judge round) are 128 live
envs, 512 live service containers and 512 running execs, and 1024 waiters
per run; the policy chooses the actual values within them. Work's envs stay
alive (paused) during a Judge round, so a run holds up to two sessions'
envs. What grows with them:

- CPU and memory: each service is limited by Docker to its own `cpus` and
  `memory_mb`, and a run reserves the full `max_cpus_live` and
  `max_memory_mb_live` of each granted phase from `pool_cpus` and
  `pool_memory_mb`. The pools are the operator's accounting, not a
  measurement of the host: setting them above the host's cores or memory
  oversubscribes it (services run slower, swap or meet the OOM killer).
- Address pools: every env with a bridge (each `public` or `allowlist`
  env, and a `none` env of more than one service) and every builder takes
  one network from Docker's default address pools, as do the Work and Judge
  bridges and everything else on the host. A stock daemon has about 30 (the
  /16s from 172.17 to 172.31 and 192.168/16 cut into /20s); creating one
  more fails. For many envs, set in `/etc/docker/daemon.json`
  `"default-address-pools": [{"base": "172.16.0.0/12", "size": 24}]`
  (4096 networks of 254 addresses, more than 8 services need) and restart
  the daemon; networks that exist keep their subnets. The pools must stay
  inside 10/8, 172.16/12 or 192.168/16 (the broker refuses a bridge outside
  the ranges its firewall rejects), must not overlap the host's own routes,
  and must never be an allowlist `private_cidrs`. Check with
  `docker info --format '{{json .DefaultAddressPools}}'`. Plan for the
  concurrent runs' Σ phase (`max_envs_live` + 1 builder) plus their parents.
- Firewall: each env and builder bridge adds a jump in `DOCKER-USER` and
  one in `INPUT`, matched in order for every new packet, and about twenty
  rules in its own chains; Docker adds its own per bridge.
- Journal: every env change (create, start, each removed object) rewrites
  and syncs the run's lease under the broker lock (about 1.4 KiB per
  two-service env, at most about 10 KiB per env), so env changes are
  serialized. As a guide, 128 public single-service envs took about 55 s to
  create and start, 11 s to pause for a Judge round and 43 s to remove
  (fake firewall; the real firewall adds a few `iptables` calls per env).
- Image handles and broker memory: a session holds max(64, 2 ×
  `max_envs_live`) image handles, live and pending pulls or builds
  together, so each env can run an image of its own; a handle in use by a
  live env cannot be released. The run's broker metadata reservation grows
  with the live limits (see the reservation above; 1496 MiB at every cap).
- Session end: a round's envs are killed and removed on 8 threads at once.
  Nothing of the round runs after 6 s plus 0.1 s per live service container,
  and every object is removed within 60 s plus 1 s per env (57 s and 188 s
  at the caps), or the run fails closed. Work resumes only after that.
- Broker descriptors: each running exec holds three descriptors (its
  stream socket and two output files) and is inspected twice a second; each
  waiter holds a connection. At the caps (1024 execs over two sessions,
  1024 waiters, every request slot) a run needs about 5000 descriptors, so
  the shell that runs `sudo -E rsi-harness run` needs `ulimit -n` of at
  least 8192 (`sudo sh -c 'ulimit -n'`; systemd's default soft limit is
  1024), and the daemon must keep up with 2 inspects a second per running
  exec.
- Retained output: each exec reserves `max_exec_output_bytes` per stream of
  `max_log_bytes` until it ends; give `max_log_bytes` at least
  `max_execs_running × 2 × max_exec_output_bytes` (the harbor-in-judge
  policy's 2 GiB and 16 MiB keep 64 execs at full retention; later ones
  retain less until none is left), and `waiters` at least the execs
  awaited at once.

### tmux for offline envs

Harbor's `terminus-2` drives the agent through tmux in the task's `main`
service and, when the image has none, installs it from the network (apt,
dnf, yum, apk or a source build). In an env with network `none` (Harbor
`no-network`), or an allowlist without the distribution mirrors, that fails
in agent setup. The operator can instead provide one statically linked
tmux, which the broker copies into such an env without changing the image.

1. Build it, as a user that may use Docker:
   `scripts/operator/build_static_tmux.sh ./tmux-3.5a`. The script builds
   the pinned tmux 3.5a against musl, libevent and ncurses 6.5 (source
   tarballs checked by SHA-256) in a throwaway container of the
   digest-pinned `alpine:3.21` (which is pulled if absent and kept) and
   prints the `sha256  path` line. The static binary runs on glibc and musl
   images alike, and its compiled-in terminfo (`xterm-256color`, `xterm`,
   `tmux-256color`, `tmux`, `screen-256color`, `screen`, `vt100`, `linux`,
   `dumb`; an image's own database is read first) lets it run in images
   without one, such as busybox.
2. Install it where only root can write:
   `sudo install -o root -g root -m 0755 tmux-3.5a /opt/rsi/tools/tmux-3.5a`.
3. Approve it in the policy:

   ```toml
   [environments.host.tmux]
   path = "/opt/rsi/tools/tmux-3.5a"
   sha256 = "<the printed sha256>"
   ```

   Run setup reads the file and fails with a `SetupError` when it is
   missing or its SHA-256 differs.
4. Optionally check it on the host, as a docker-group user:
   `RSI_STATIC_TMUX=/opt/rsi/tools/tmux-3.5a pytest tests/integration/test_sandbox_offline_tmux.py`
   copies it into `busybox:1.37.0` and `python:3.13-slim-bookworm` envs
   with network `none`, runs a tmux session there (new-session, send-keys,
   capture-pane), and Harbor's own terminus-2 `TmuxSession` through the
   plugin. A missing `RSI_STATIC_TMUX` file is built there first.

How it is used:

- Two keys, as everywhere: the policy's table offers the tool
  (`capabilities` then lists `"tools": ["tmux"]` for each phase with
  environments), and a caller asks `tool_install` with `tool = "tmux"` for
  one service of an env its own session made.
- The broker, never Work or Judge code, reads the file (a regular file of
  at most 15 MiB), checks its SHA-256 and copies the bytes it hashed through
  the same archive copy as `copy_in`, to `/usr/local/bin/tmux` (root, mode
  0755; missing directories are made 0755). No host path, mount or socket
  reaches the env or the phase. A file that is missing, unreadable or
  differs from `sha256` is refused with `infrastructure` (field `tool`) and
  nothing is copied; the broker's log, not the answer, names the path and
  the hash it found.
- An existing `/usr/local/bin/tmux` is never replaced: the answer is
  `{"tool": "tmux", "path": "/usr/local/bin/tmux", "installed": false}`.
  Whether tmux is elsewhere on the image's PATH is the caller's check
  (`command -v tmux`).
- It costs one operation and no upload budget; the file lands in the
  service's writable layer and counts toward its soft disk.
- The Harbor plugin asks on its own: when the grant offers tmux, `start()`
  runs `command -v tmux` as root in `main` once the env is ready and, only
  when that fails, calls `tool_install` and then `tmux -V` (a warning when
  `/usr/local/bin` is not on the image's PATH). Harbor's agent setup then
  finds tmux and installs nothing. A refused install fails `start()` (the
  env is destroyed). `--ek inject_tmux=off` turns it off for a run.
  Sidecars are never touched.

terminus-2 also records the session with asciinema by default, which it
installs from the network too; offline, task authors pass
`record_terminal_session=false` (see
[Offline agents](harbor-task-authoring/sandboxes.md#offline-agents)).

### Root checks

`scripts/operator/sandbox_root_check.sh` runs, as root, with the real
firewall and loop-ext4 builders (`RSI_SANDBOX_ROOT_MODE=1`):

| Check | What |
|---|---|
| R1 | A public env reaches the internet and DNS; metadata, RFC1918, the LAN gateway, its bridge gateway (host INPUT), a sibling env and a Work-style bridge are blocked; each probe also runs from an unfirewalled container, which shows the targets this host blocks anyway |
| R2 | Services reach each other by alias (public and none envs), another env never; without the intra-bridge ACCEPT peers fail |
| R3 | A builder RUN step reaches `https://pypi.org`, but not metadata, RFC1918, the LAN, its gateway or a child; BuildKit's own fetches (`ADD <url>`, a git source, `FROM` a registry) reach `https://example.com` and none of those |
| R4 | The build suite on loop-ext4: ENOSPC is `disk` and the builder survives; kill -9 at builder, build, load and loaded, then recover detaches the loop device and removes the file, bridge, rule, tags and dangling images |
| R5 | Exec kills through pidfd (interrupt, timeout escalation, OOM), a group kill spares a `setsid` process and other execs, a paused env is killed through `cgroup.kill` with no write after the freeze; the table names the interpreter and its pidfd path (production's Anaconda 3.13 has no `os.pidfd_open`: the libc fallback) |
| R6 | kill -9 mid env_create or pull, or with the Work group paused, then recover, with the real firewall (the paused env through `cgroup.kill`, no write after the freeze); recovery sees every process in `/proc` |
| R7 | The env backend and the broker's env operations as root, with the real firewall: every rule installed has neither jump nor chain left |
| R8 | Harbor through the plugin with the real firewall: prebuilt, Compose sidecar, in a container with only the endpoint, the sample's fixed procedure, a 1 GiB directory round trip |
| R11 | An allowlist env reaches a listed hostname on its listed port (resolved by the embedded DNS) and a listed IP; the hostname's other port, an unlisted host, a public resolver, listed metadata, RFC1918 and LAN addresses and its gateway (INPUT) are blocked; an operator-approved private CIDR is reached; after a refresh the new address is reached on its port only and the rule attests exactly |
| R10 | The acceptance below (`--skip-acceptance`: SKIPPED) |
| R9 | Nothing new of the checks' runs is left on the host: rsi containers, volumes, networks, built tags, `RSI_` chains and `rsi-` jumps, rsi bridges, builder loop devices; objects of other runs started meanwhile are listed apart, and an object no run names (a chain without its jump, a bridge without its network) always counts |

With `RSI_ROOT_CHECK_SELFTEST=1`, a non-root user runs R1-R3 and R11
against a fake firewall: every probe runs and shows what the firewall must
block. Targets that the unfirewalled control also cannot reach prove
nothing about the firewall on that host; the control column says which
rows do.
On hosts whose `iptables` is the nf_tables variant, `/run/xtables.lock` does
not serialize rule changes, so a create cannot be held at `iptables --wait`
by that lock; the settlement of such a create is covered by the recovery
tests.

### Acceptance

`scripts/operator/sandbox_acceptance.sh` runs `rsi-harness run` and
`rsi-harness recover` on `sample_tasks/harbor-in-judge`. Only the agent's
model CLI is replaced by a script (`tests/acceptance/scripted_cli.py`,
`tests/acceptance/work`); the runs are otherwise production runs.
`--only` selects scenarios: `a1` (with A6), `a2` (with A3), `a4`,
`a7-env-create`, `a7-build`, `a7-load`, `a7-paused`, `a8` and `swebench`;
every scenario ends with its A5 audit.

| Item | Scenario |
|---|---|
| A1 | The TB2 subset (fix-git, regex-log, adaptive-rejection-sampler, kv-store-grpc, nginx-request-logging, git-multibranch; 1 CPU, 2 GiB each) in a real Judge: every oracle trial 1.0, every nop 0.0; every oracle env destroyed within 60 s of its verifier, and no quarantine, unknown outcome or fail-closed in the run |
| A2 | The same subset run in Work by the scripted agent before it submits |
| A3 | The Compose sidecar task in Work and in the Judge |
| A4 | fix-git built from its Dockerfile (public build network) in the Judge |
| A5 | After every scenario: no container, network or volume of the run (the Work WORKDIR volume the Harness keeps until `rsi-harness cleanup` is listed apart, and nothing labelled with the run may remain after cleanup), no built image or dangling image of its digests, no bridge, rule or loop device of the run, no `<data_root>/<run>/sb`; pulled images are listed apart |
| A6 | No Docker, containerd or BuildKit socket and no `DOCKER_HOST` in any Work, Judge, env or builder container; Work and Judge see only the endpoint socket, children and a RUN step none (no `/run/buildkit`); the endpoint is no Engine API |
| A7 | kill -9 of the harness at env_create, mid-build, mid-load and with a paused env, then `recover`; A5 holds |
| A8 | A Judge activated 60 s before the Work deadline (verifier 900 s) completes with a reward: the submit lands 30-90 s before the deadline, the Judge round starts before it and ends after it |

`--only swebench` adds the [SWE-bench sample](#the-swe-bench-sample) as row
SWE: `sample_tasks/swebench-in-judge` with its own policy, its three tasks
offline in a real Judge (oracle 1.0 and nop 0.0, a normal run as in A1),
then its A5. It is opt-in: neither the default run nor the root check's R10
includes it, and only its preflight needs its images.

### The vLLM-in-Judge demo

`scripts/operator/vllm_demo.sh` runs the reference scenario through
`rsi-harness run` as root, on `sample_tasks/vllm-in-judge` with its
`operator-policy.toml` (Judge environments only, public egress, pulls, no
builds):

```bash
sudo scripts/operator/vllm_demo.sh --gpus 4      # --dry-run: the plan only
```

- Work is CPU only (`gpus = 0`) with public egress: its scripted agent
  (`work/agent.sh`, through `tests/acceptance/scripted_cli.py`) downloads
  `Qwen/Qwen2.5-1.5B-Instruct` at a pinned commit into
  `/workspace/checkpoint` and submits. `rsi-submit` waits for the Judge
  round, so Work's 5400 s cover the download and the round's 3600 s.
- Each Judge round gets one GPU of the `--gpus` pool
  (`[metadata.rsi_harness.verifier] gpus = 1`; disjoint mode, since Work
  holds none), no network of its own and the WORKDIR as a read-only
  snapshot. Its `tests/test.sh` starts vLLM 0.11.0 on loopback serving the
  checkpoint, waits for `/health`, starts the metering proxy (below), then
  runs Harbor 0.21.0's `terminus-2` with `--model
  hosted_vllm/rsi-checkpoint --ak api_base=http://127.0.0.1:8001/t/<task>/v1
  --ak model_info=...` over a trivial task and TB2 `regex-log`, both
  through `--env rsi_sandbox_harbor:ManagedSandboxEnvironment`. The envs
  need public egress because terminus-2 installs tmux (not with the
  operator's static tmux, see [tmux for offline envs](#tmux-for-offline-envs)).
  Harbor gets the round's remaining budget less 300 s, so the report is
  written even if it hangs.
- The reward is the fraction of trials solved; `vllm-demo-summary.json`
  adds vLLM's GPUs and model root, the completions it served (aborted ones
  apart), each trial's outcome and environment, the tokens its agent
  used and the tokens metered, and `accuracy_vs_tokens` (solved
  fraction against prompt and completion tokens, per task and in total). A
  verifier timeout or missing reward file is completed and unsolved; an
  environment or broker exception is an infrastructure error.
- The script's table: V0 the checkpoint and the submission; V1 vLLM
  healthy, serving exactly `/workspace/checkpoint`, only on GPUs the
  Harness named for the Judge; V2 completions served and every trial's
  agent asked the model; V3 every trial completed in a broker env, the
  Judge seeing the endpoint and no host control socket, no infrastructure
  error and the round completed (any score); A5 nothing of the run left,
  before and after `rsi-harness cleanup`.
- Preflight refuses a `--gpus` device already holding more than 1024 MiB,
  and less than 20 GiB free under Docker's root (15 GiB while the sample's
  image is cached). Interrupted, the script stops its run and prints the
  `rsi-harness recover` and `cleanup` commands for it. The Work image
  (about 10 GB) stays cached as the run's Base image, and the checkpoint
  stays in the WORKDIR volume until cleanup.
- Without root, `RSI_RUN_VLLM=1 RSI_TEST_GPUS=4,5,6,7 pytest
  tests/integration/test_vllm_in_judge_sample.py` runs the same agent and
  procedure in Work- and Judge-shaped containers against a live broker
  with the fake firewall and the shipped policy.

Per-request token accounting belongs to a task's fixed procedure, not to
the Harness; the sample's `tests/metering_proxy.py` (standard library only)
is a copyable reference. It listens on loopback and forwards
`/v1/chat/completions`, `/v1/completions` and `/v1/models` to an
OpenAI-compatible server under a per-trial base path `/t/<trial>/v1` or
bare `/v1`, passes streams through event by event (requesting
`stream_options.include_usage` and stripping it when the client did not ask)
and appends one JSON line per request to `/logs/verifier/usage.jsonl`:
`trial`, `endpoint`, `model`, `stream`, `status` (502 when the upstream
cannot be reached), `prompt_tokens`, `completion_tokens`, `latency_ms` and
`error`. A line is written before the request's last bytes reach the
client; the proxy's own health is `/metering/health`. Harbor's agent kwargs
(`api_base` among them) are per job, not per trial, so the procedure runs
each task as a Harbor job of its own (one trial, `--jobs-dir
harbor-jobs/terminus-2 --job-name <task>`, the name's `/` as `-`), at most
`RSI_VLLM_CONCURRENCY` at once, each agent given
`api_base=.../t/<task>/v1`. A trial whose requests were not metered, or a
request outside every trial's base path, is an infrastructure error.

### The SWE-bench sample

`sample_tasks/swebench-in-judge` judges three SWE-bench Verified tasks
entirely offline: its Judge's `tests/test.sh` (through `tests/run_tasks.sh`,
which Work may run too) runs Harbor 0.21.0's `oracle` and `nop` agents over
`/tests/swebench-verified` with `--env
rsi_sandbox_harbor:ManagedSandboxEnvironment`, every task env with network
`none`, and scores 1 only if every oracle trial scored 1 and every nop
trial 0 (`tests/harbor_reward.py`, shared with harbor-in-judge). Work and
the Judge have no network either.

The tasks are Harbor's registry dataset `swebench-verified@1.0`
(`datasets/swebench-verified/<instance>` of
`laude-institute/harbor-datasets` at commit `86723674`), which an offline
Judge cannot fetch, so the sample vendors three: `pallets__flask-5014`,
`pytest-dev__pytest-7205` and `pytest-dev__pytest-7236` (small images that
share layers, about 1.5 GB to download and 4 GB unpacked, and tests that
run offline in seconds). Tasks whose tests reach the network, such as
`psf/requests` against httpbin.org, cannot be judged offline. The copies
differ from upstream only where it needs the network:

1. The image: upstream builds `FROM
   swebench/sweb.eval.x86_64.<instance>:latest` plus `curl` installing uv.
   Each copy names that image by digest as its `[environment]
   docker_image` (the `latest` digests of 2026-10-01), with `workdir =
   "/testbed"` and `network_mode = "no-network"`; its Dockerfile is only
   that `FROM` and `WORKDIR /testbed`.
2. The reinstall: upstream's `tests/test.sh` reruns `python -m pip install
   -e .`, which fetches the build backend from PyPI. The image already
   holds that editable install, so the copy drops the step.
3. The grading: upstream runs `uv run parser.py`, which installs
   `swebench==4.0.3` and `datasets` from PyPI. The copy carries a
   standard-library port of SWE-bench's pytest log parser and its
   PASS_AND_FAIL resolution (a FAIL_TO_PASS or PASS_TO_PASS test that is
   missing, FAILED or ERROR fails the task; SKIPPED counts as neither),
   run by the testbed's own Python, writing the same `report.json` and
   `reward.txt`.

More tasks of the dataset take the same three changes and one manifest
line each. Operator steps:

1. Pre-pull the images (see
   [Pre-pulling large image sets](#pre-pulling-large-image-sets)):
   `scripts/operator/prepull_images.sh sample_tasks/swebench-in-judge/images.manifest`.
   They stay cached, also after `sandbox prune-images`.
2. Review and sign
   [`sample_tasks/swebench-in-judge/operator-policy.toml`](../sample_tasks/swebench-in-judge/operator-policy.toml):
   both phases get envs with network `none` only, `docker.io` pulls with
   `max_pull_mb = 1` (cached images bind for free; any download is refused
   as soon as its layers are announced), no builds, and three envs of the
   tasks' 1 CPU, 4096 MiB and 10240 MiB of soft disk at once. It reserves
   16 CPUs, 35072 MiB and 61440 MiB of disk.
3. For an agent that needs tmux, such as `terminus-2` (oracle and nop do
   not), build the static tmux and uncomment the policy's
   `[environments.host.tmux]` (see [tmux for offline envs](#tmux-for-offline-envs));
   the plugin then copies it into each env, as the SWE-bench images have
   none.
4. Run it through `rsi-harness run` as root:
   `sudo scripts/operator/sandbox_acceptance.sh --only swebench`
   (`--dry-run` as any user), see [Acceptance](#acceptance).

Without root, `RSI_RUN_SWEBENCH=1 pytest
tests/integration/test_swebench_in_judge_sample.py` runs the fixed
procedure in a Judge-shaped container against a live broker with the fake
firewall and the shipped policy: six trials, every env `none` and bound
from the cache. With `RSI_STATIC_TMUX` it runs once more with tmux offered
(the plugin installs it in all six envs) and runs a tmux session in an
offline env of the flask image.
