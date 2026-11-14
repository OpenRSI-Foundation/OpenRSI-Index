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
| Slurm / LSF with Apptainer, single or multi-node | None: Work's paused workspace, bound read-only | `NativeJudgeEvaluator` pauses Work, runs the Judge and resumes Work in the same order; there is no copy to release. |

Docker's existing implementation is retained. Backend regression suites cover
successful rounds, teardown failure, artifact-write failure and retained
recovery authority. Real LSF and multi-node allocations remain outside the
current acceptance evidence.

On every cluster backend, single or multi-node and in both `process_namespaces`
modes, each submission stops every Work process on the node and binds its
workspace read-only at `WORKDIR` in the Judge. No copy of the workspace is
made, so a round needs no extra space and costs no copy time. The pause
stops each Work thread. A thread that waits in the kernel (D) with
that SIGSTOP pending counts as stopped: it stops before it runs user code
again, but a system call it is in can finish during the round. If Work does
not stop within 10 s, the pause continues it and the round is refused, not
counted. On several nodes the workspace is on the shared file system and
remote Work ranks must have finished before the round starts. On NFS a Judge
node can miss a name Work changed within the client's attribute-cache time;
for multi-node runs use a coherent file system or mount the workspace with
`lookupcache=none` (or `actimeo=0`).

While Work is paused, before the Judge starts, the controller walks the whole
workspace without following symlinks. A FIFO, socket or device there could pass
data between the Judge and Work, so the round is refused, not counted, and the
paths are listed in the `rsi-submit` reply. A walk that cannot finish (a
directory it cannot open, or more than ten million entries) refuses the round
the same way. A final round after the Agent has exited is not checked. The
check runs once, before the Judge starts: a Work process that resumes itself
during the round can still change files the Judge reads.

Once the Judge has returned, its remote environments have closed and the
scoring artifacts are committed, the controller reports `judge_removed` and
resumes Work. This applies to successful evaluations, verifier errors, invalid
rewards and verifier timeouts. The live Work workspace and all
scoring/trajectory artifacts remain available.

If scoring-record persistence or Judge teardown fails, recovery is required.
The controller ends its Work execution without
resuming a paused Agent, even when it cannot write a recovery record. A failed
round keeps no copy of the workspace; the workspace itself, the round's logs and
its feedback remain. A run started with an earlier Harness may have left a
`judge-rounds/` directory in its run directory. Nothing uses it now; delete it
with `chmod -R u+w judge-rounds && rm -rf judge-rounds`.

Paths come from the existing run configuration; there is no dependency on a
cluster name, a particular mount path, Docker, root privileges, a quota command
or an archive service. Shared storage must still be correctly configured by the
operator. This change does not bound candidate size or log growth and does not
alter already-frozen runtimes.

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
including site isolation checks, Work pause/resume, a round refused over a FIFO,
repeated successful and failed evaluations, timeout, and controller
interruption during a Judge. They check that the Judge reads Work's workspace
read-only, that no copy is made and that committed reports, output and
interrupted evidence are kept. They need no model credentials and do not launch
research. Image building, model CLI integration and the Engine's resume command
are outside this test's scope. Normal unit tests also exercise the special-file
walk, artifact-write and teardown failures without filling the host filesystem.
