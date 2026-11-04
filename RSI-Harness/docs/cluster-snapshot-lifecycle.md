# Snapshot lifecycle across execution backends

The common contract is: Judge execution has ended, the required scoring artifacts
are committed, and teardown is proven before the acquired temporary resource is
released. Record release before resuming Work. An unproven teardown or release requires recovery rather than
a successful release claim. This does not require every backend to store the
snapshot in the same format.

| Backend | Disposable snapshot | Implementation |
| --- | --- | --- |
| Linux Docker, including Docker-group mode | Attested Docker image and/or managed workspace layer | Existing `JudgeRunner` and snapshot backends enforce teardown, persistence and release ordering. |
| macOS controller with Docker Desktop | Daemon-managed image/volume | Uses the Docker lifecycle; host directories are not recursively deleted as a substitute for Docker resource release. |
| Slurm / LSF with Apptainer, single or multi-node | Acquired workspace directory | `NativeJudgeEvaluator` and `DirectorySnapshot` enforce the same ordering and delete only the controller-owned copy. |

Docker's existing implementation is retained. The cluster change fills its
previous no-op release; it does not add a second cleanup mechanism to Docker or
macOS. Backend regression suites cover successful release, teardown failure,
artifact-write failure and retained recovery authority. Real LSF and multi-node
allocations remain outside the current acceptance evidence.

Slurm and LSF share the same directory snapshot implementation. Each submission
pauses Work and copies its workspace for the Judge. Once the Judge has returned,
its remote environments have closed and the scoring artifacts are committed,
the controller removes that round's temporary workspace and its empty parent.
Only then does it report `snapshot_released` and resume Work. This applies to
successful evaluations, verifier errors, invalid rewards and verifier timeouts.

This matches the lifecycle of disposable local Docker snapshots: release is
an actual cleanup operation, not just an event. There are no retention modes,
archives, compression settings, best-round selection rules or new profile keys.
The live Work workspace and all scoring/trajectory artifacts remain available.
A disposable Judge snapshot is not a permanent best-candidate export; candidate
publication continues to use the task's normal submission workflow.

The controller deletes only a directory it acquired in the current process.
It does not scan old runs or identify disposable files by name, age or size.
The acquisition's directory identity is checked locally at release; these device
and inode values are never persisted as cross-node authority. Removal uses file
descriptors and does not follow symlinks. Read-only copied directories can be
removed without changing source directories or shared linked files. Neither
Work nor Judge may modify the controller-owned snapshot parent directory.

If copying, scoring-record persistence, Judge teardown or cleanup fails, the
remaining evidence is kept, release is not claimed, and recovery is required.
The controller ends its Work execution without resuming a paused Agent, even
when it cannot write a recovery record. Partial and old unmanaged directories
require explicit operator inspection. A successfully removed snapshot is not
reconstructed merely because a later event or filesystem synchronization fails.

This uses ordinary POSIX directory operations, symlink-safe Python removal and
filesystem synchronization. Paths come from the existing run configuration;
there is no dependency on a cluster name, a particular mount path, Docker,
root privileges, a quota command or an archive service. Shared storage must
still be correctly configured by the operator. Space for the live workspace,
the current Judge copy and normal logs is still required. This change does not
bound candidate size or log growth and does not alter already-frozen runtimes.

## Real Apptainer acceptance

Inside a CPU allocation, provide an existing SIF containing bash and the Python
named by your site profile, and run:

```sh
RSI_ACCEPTANCE=1 \
RSI_APPTAINER_TEST_PROFILE=/absolute/path/to/site.toml \
RSI_APPTAINER_TEST_IMAGE=/absolute/path/to/task.sif \
python -m pytest tests/acceptance/test_apptainer_snapshot_lifecycle.py -v
```

These opt-in tests use real Work/Judge containers and the native evaluator,
including site isolation checks, Work pause/resume, repeated successful and
failed evaluations, timeout, and controller interruption during a Judge. They
check both snapshot removal and retention of committed reports, output and
interrupted evidence. They need no model credentials and do not launch research.
Image building, model CLI integration and the Engine's resume command are
outside this test's scope. Normal unit tests also exercise copy, artifact-write,
teardown and cleanup failures without filling the host filesystem.
