# Native Work/Judge execution on Modal

`rsi-harness modal run` runs the native `RunCoordinator` on a single Modal
Function allocation. It uses the existing submission service, round accounting,
feedback, best-score selection, trajectory capture, leases, and native artifacts.
This is separate from the existing Modal child-sandbox backend.

The supported contract is Linux, a Debian-compatible image with Python 3.12,
`WORKDIR /workspace`, zero Work GPUs, and a clean hidden Judge with zero or one
GPU. Codex and Oracle are supported. Other contracts fail preflight. A task must
implement its verifier and any local inference broker; the runner does not
substitute a remote model for a task's fixed local model.

## Isolation and handoff

The trusted controller owns the Modal credentials, task tests, state volume and
submission service. Work runs as uid 60000 in a private mount, PID, IPC, UTS and
network namespace. Only explicitly selected system directories and phase mounts
enter its filesystem. Its workspace, home and scratch survive between rounds.
It cannot see the controller, hidden tests, state volume, or GPU device nodes.
Seccomp prevents new namespaces, tracing and keyring access.

On submission the controller stops and checks all Work-uid processes, including
processes that started new sessions. It copies bounded regular files, refusing
links, special files and oversized candidates. Judge receives that copy read-only
in a fresh namespace with fresh scratch, read-only tests and separate output.
The candidate copy and its content digest are retained per round.

Judge-only assets sit behind a root-only directory even when the kernel has no
Landlock support. The verifier can explicitly grant a candidate its public input
through `rsi-isolate`; private datasets and model-broker state stay protected.

Judge owns the allocation's GPU. GPU devices and hidden tests are root-only;
task verifiers must use native `rsi-isolate` to execute candidate code. A verifier
can pass a connected Unix-socket descriptor to the candidate to expose its local
model broker. The candidate receives no cloud credentials or provider proxy.
The Work provider broker permits only operator-pinned HTTPS endpoints; Oracle
gets only its authenticated submission socket. Judge has no external model route.

A verifier timeout, nonzero exit, or invalid reward has **no score**, even if a
reward file was already written. Unsupported isolation fails closed.

## Operator configuration

Install the harness with its Modal extra (`pip install -e '.[modal]'`). Use a
dedicated Modal environment and a unique state volume. This JSON is an example;
set the profile, environment, volume and secret to your own resources:

```json
{
  "profile": "research",
  "environment_name": "openrsi",
  "volume_name": "coding-evolution-state",
  "gpu": "H100",
  "cpus": 16,
  "memory_mb": 131072,
  "segment_seconds": 82800,
  "cleanup_seconds": 120,
  "max_segments": 3,
  "max_compute_seconds": 90000,
  "provider_urls": ["https://api.openai.com"],
  "agent_secret_name": "openrsi-agent",
  "asset_volumes": {"/asset-store/model": "pinned-coding-model"},
  "assets": [
    {"source": "/asset-store/model/qwen3-coder", "target": "/models/coder", "phase": "judge"}
  ]
}
```

The named Secret supplies `RSI_AGENT_API_KEY` and optionally
`RSI_AGENT_API_BASE_URL`. Only the Agent-specific fields enter Work; the secret
is not mounted into Judge or written into a launch receipt. Set provider URLs to
match the configured API base. Model files must already exist in the explicitly
bound asset volume, pinned and verified by the task's model manifest. Asset
volumes are read-only. No task-time model download or external inference is
needed. The existing account/environment budget remains authoritative for cost.

Modal's default disk allocation is used unless `disk_mb` is supplied. Explicit
disk requests currently start at 524288 MiB; this field is an allocation request,
not a filesystem quota. The candidate snapshot limit is 512 MiB and 20000 files.

```bash
rsi-harness modal run /absolute/path/to/task \
  --config /absolute/path/to/modal.json \
  --seconds 86400 --max-submissions 100 \
  --agent codex --model YOUR_MODEL \
  --receipt /absolute/path/to/launch-receipt.json
```

Keep the foreground launcher running. It performs serial segment handoffs and
never requests more than one GPU at a time. Retries are disabled. Native state
and artifacts are committed to the volume before a handoff. Each new segment
starts from retained files using native resume, with no extra time or submission
grant. The original wall-clock deadline persists, including handoff delays.
This permits a 24-hour experiment despite Modal's per-Function 24-hour maximum.
The compute ceiling reserves segment execution and cleanup time conservatively;
it is not a dollar cap and does not include image-building or storage charges.
An unclean provider termination stops automatic continuation and requires
inspection; partial evaluations are never promoted to successful submissions.

Native artifacts are under `logs/runs/RUN_ID/TASK_ID/` in the state volume,
including `final_result.json`, `evolve_state.json`, the native round records and
captured Agent output. Workspace copies and launch accounting live under
`data/RUN_ID/`. Export with Modal's volume download command; retain originals for
OpenRSI submission. A JSON launch receipt is not a replacement for those records.
Because Modal Volumes reject hard links, the adapter publishes a complete,
fsynced copy for the native `test_output.txt` alias, preserving its bytes.

## Bounded acceptance checks

Offline contracts run with `pytest tests/test_modal_native.py`. Cloud checks
require explicit credentials and spend compute:

```bash
PYTHONPATH=src python scripts/validate_modal_native.py \
  --config /absolute/path/to/probe.json \
  --case two-rounds --receipt /absolute/path/to/cpu-receipt.json
```

Use `gpu` to check one-GPU Judge access, `verifier-error` to check unscored
failures, and `resume` with `segment_seconds=30` to check a real two-allocation
continuation. Probe configs need no provider endpoints or secret, use a short
compute ceiling, and use `gpu: null` except for the GPU case. All fixtures use
synthetic 0/1 scores: they establish runtime behavior, not recursive improvement.
