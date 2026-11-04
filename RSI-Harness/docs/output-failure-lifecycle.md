# Required output failures across backends

A failed required log must not be reported as a complete capture or a successful
submission. Establish the sink before starting execution, observe reader errors
while execution is active, contain the affected execution, and surface an
infrastructure failure. Preserve previously committed artifacts; do not silently
restart research or change its budget. Optional console/UI callbacks keep their
existing best-effort contract; they are not a replacement for required artifacts.

| Execution path | Output owner | Failure handling |
| --- | --- | --- |
| Linux Docker, root or Docker-group mode | Docker runtime | Stops and checks the exact container through the daemon; unrelated containers are untouched. |
| macOS controller with Docker Desktop | Same Docker runtime | Same daemon-side containment; no host PID or Linux signal assumptions. |
| Slurm / LSF, single-node Apptainer | Native runtime | Checks the drain thread while waiting, terminates/reaps its process group, and rejects late persistence failures. |
| Multi-node Judge controller | Native host-controller runner | Captures scheduler output through a pipe instead of an unchecked file descriptor; invokes the frozen remote stop command and reaps the local launcher on failure. |
| Multi-node rank output | Phase broker | Publishes bounded rank evidence atomically; persistence failures are visible to the main runtime in memory even if no error/cancellation file can be written. Sibling ranks are cancelled on reader failure. |

Docker raw verifier output retains its existing atomic publication format;
an unpublished partial temporary file can be discarded, while earlier committed
output remains. Agent append logs retain their existing prefix. Cluster logs
retain partial output and failed-round evidence. These formats need not be
identical to enforce the same completion and containment rules.

The snapshot lifecycle is described in [cluster-snapshot-lifecycle.md](cluster-snapshot-lifecycle.md).
No new retention policy, archive format, cluster path, quota command, or automatic
scientific retry is introduced. A failed remote stop or unavailable filesystem
can still require operator recovery; failure reporting is not proof that every
remote host is reachable or has been cleaned successfully.

## Validation

- `tests/runtime/test_docker.py`: required output, deadlines, atomic publication,
  error redaction, and containment of the exact container.
- `tests/cluster/bluevela/test_output_failure.py`: storage errors, child lifetimes,
  and remote-controller stop invocation using real local subprocesses.
- `tests/cluster/bluevela/test_multinode.py`: rank evidence errors, in-memory
  cancellation without writable markers, and cancellation of sibling ranks.
- `tests/acceptance/test_docker_output_failure.py`: opt in with
  `RSI_ACCEPTANCE=1 RSI_DOCKER_TEST_IMAGE=<existing-linux-image>`. Requires a live
  Docker daemon and an image containing `/bin/sh` and `sleep`; exercises real
  containers and checks that a second container remains running.
- `tests/acceptance/test_apptainer_snapshot_lifecycle.py`: real Apptainer scoring,
  cancellation through the official signal handler, and injected log quota error.

Real acceptance has covered macOS controlling Docker Desktop and Slurm running
Apptainer. LSF and multi-node scheduler deployments have automated component
coverage, not a claim of real multi-host acceptance. Provider-specific cloud
sandbox SDK logs and arbitrary task-owned log files are outside this contract.
