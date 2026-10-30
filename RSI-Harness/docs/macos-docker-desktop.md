# Native macOS controller (experimental)

This development backend runs Python and `rsi-harness` on macOS and uses the
existing Docker Desktop daemon. It does not provision another VM or require an
SSH Linux host. Docker Desktop itself still runs Linux containers in its own
Linux VM; this is not native execution of Linux containers by the macOS kernel.

The supported scope is CPU-only, non-sandbox Harbor tasks with a split WORKDIR
below `/`, including public and hidden Judges, interrupted-round recovery and
resume, and the Judge API transport. GPU requests, sandbox tasks, full-rootfs
workspaces, and non-Desktop Docker daemons are refused before execution. Select
your local Docker Desktop Unix socket; TCP and SSH endpoints are explicitly
unsupported, even when the remote daemon advertises itself as Docker Desktop.

Standalone `--judge-llm-auth codex-login` relies on Linux host token-renewal
facilities and is rejected before container creation. Use an API credential or
a separately validated Codex Work Agent's lent login. Validate the chosen Work
user and credential route; do not waive auth ownership checks or silently change
the task user when a non-root authentication setup fails.

## Boundaries

- Work and Judge retain their normal unprivileged execution/isolation rules.
- Private `/logs/verifier` uses a separate anonymous Linux Docker volume for
  each Judge, not a macOS filesystem bind. It is never mounted into Work.
  Before tests or model launch, the controller resolves the configured Judge
  UID/GID and initializes only that volume root with matching ownership and
  mode 0700. Named and non-root Judges retain their configured identity; failed
  ownership or mount checks stop the round.
- Before reading rewards, the controller exports regular files from the volume
  using the Docker archive API. Links, devices, traversal, duplicate or
  case/Unicode-colliding paths, more than 1 GiB of file contents, and more than
  100,000 members fail the round without a score. A separate transport bound
  includes archive overhead. The controller records content rejection outside
  the archive destination and quarantines earlier exported files so an old
  reward cannot be read as the current result, then releases the Judge and volume.
  Transient transport/export failures retain them for recovery; recovery exports before
  removal. Linux recovery ignores image-supplied Desktop labels.
- A trusted, digest-pinned Debian helper enters the Docker daemon's
  Linux mount and network namespaces to operate its iptables and lock. It has
  SYS_ADMIN, NET_ADMIN, SYS_PTRACE and SYS_CHROOT, host PID visibility, and an
  unconfined AppArmor profile. It has no host bind, Docker socket or credentials;
  Work and Judge never receive these privileges. The helper image is resolved
  to its immutable local image ID for use. Operators must trust that image.
  Private, locked records under `~/.cache/rsi-harness/desktop-helpers` track
  creation and removal. Recovery reclaims only inactive helpers with matching
  daemon, image, name and ownership label; incomplete creation remains pending.
  It never removes live operations or unrelated containers by age or name prefix.
- The existing firewall rules and ownership checks remain in effect. The helper
  changes the execution location of iptables, not the task's network policy.
- The controller callback address is resolved through Docker Desktop's
  `host.docker.internal`, rather than the Linux bridge gateway.

Do not disable these checks to accommodate a restrictive Docker Desktop setup.
Linux hosts retain their existing firewall, bind-mount and recovery paths.

## Acceptance

Run the focused tests with `pytest tests/runtime/test_desktop*.py`, and run the
normal Docker, network, Judge, recovery and production-composition tests on Linux.
On an actual Mac, use the real CLI with a bounded deterministic Oracle fixture,
without a fake firewall. Verify repeated submissions, actual low-UID denial of
private file access, exported evidence/rewards and container cleanup. Repeat for
public Judges. No formal research or paid model calls are required for these
runtime checks. Task-specific validation is still required.

Enable the real Mac tests explicitly:

```bash
RSI_TEST_MAC_DESKTOP=1 python -m pytest tests/integration/test_desktop_docker.py -q
```

These tests install real run-scoped firewall rules, create disposable containers
and volumes, and deliberately kill their own controller process for recovery.
The API test uses a synthetic credential and local endpoint; it blocks a second
unapproved host port from Work and Judge and checks output redaction. No raw
login files or real-provider logs belong in the source tree.
