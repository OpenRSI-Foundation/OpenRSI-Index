#!/usr/bin/env python3
"""Async run wrapper for the gpic-generation task.

Contract (mirrors the dinov3-imagenet-semdense harness): the agent submits a
training/generation command and ends its turn; the harness polls the
attempt's ``status.json`` every ``--poll-seconds`` without an LLM call and
resumes the agent session when the attempt reaches a terminal state.
Interrupted attempts resume in place from the trainer's own checkpointing
WITHOUT rewinding the data stream (the one-pass budget counts consumed
samples, not wall restarts).

Budget enforcement: the task budget is wall-clock GPU-hours on the released
1x8 H100 topology. ``charged_gpu_hours = wall_seconds * GPIC_TOPOLOGY_GPUS / 3600``.
A detached watchdog terminates the process group when the per-attempt cap
would be exceeded and marks the attempt ``killed_budget``.

One-pass data telemetry: every poll, the watchdog walks the attempt's
process tree and sums ``/proc/<pid>/io`` ``read_bytes``/``rchar``. Reading
substantially more bytes than one pass over the staged training set
(``GPIC_TRAIN_BYTES``, recorded at staging) is physically inconsistent with
a single epoch — the attempt is marked ``data_overread_flag`` in
``status.json``, which the frozen verifier treats as a hard failure. This
is an upper-bound tripwire, not sample-exact accounting; sample uniqueness
itself is a policy rule (RH-007).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT_ROOT = Path(os.environ.get("GPIC_OUTPUT_ROOT", "/app/output"))
ATTEMPTS = OUTPUT_ROOT / "attempts"
CONTROL = OUTPUT_ROOT / "control"
TOPOLOGY_GPUS = int(os.environ.get("GPIC_TOPOLOGY_GPUS", "8"))
DEFAULT_CAP = float(os.environ.get("GPIC_GPU_HOURS_CAP", "1000.0"))
OVERREAD_FACTOR = 1.15
POLL_GRAIN_SECONDS = 30


def _resolve_train_bytes() -> float:
    """One-pass over-read tripwire size. Prefer the explicit env var; fall
    back to the staging manifest so the tripwire is never silently disarmed
    (GPIC_TRAIN_BYTES ships as 0)."""
    val = float(os.environ.get("GPIC_TRAIN_BYTES", "0") or 0)
    if val > 0:
        return val
    train_dir = Path(os.environ.get("GPIC_TRAIN_DIR", "/datasets/gpic/train"))
    manifest = train_dir.parent / "GPIC_STAGING_MANIFEST.json"
    try:
        return float(json.loads(manifest.read_text())["train_total_bytes"])
    except Exception:  # noqa: BLE001
        return 0.0


TRAIN_BYTES = _resolve_train_bytes()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_status(attempt_dir: Path) -> dict:
    path = attempt_dir / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_status(attempt_dir: Path, status: dict) -> None:
    path = attempt_dir / "status.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def charged_gpu_hours(start_epoch: float, end_epoch: float) -> float:
    return max(0.0, end_epoch - start_epoch) * TOPOLOGY_GPUS / 3600.0


def process_tree_read_bytes(root_pid: int) -> float:
    """Best-effort sum of read_bytes over the session's process tree."""
    total = 0.0
    try:
        pids = subprocess.run(
            ["ps", "-o", "pid=", "-g", str(root_pid)],
            capture_output=True, text=True, timeout=10).stdout.split()
    except Exception:  # noqa: BLE001
        pids = [str(root_pid)]
    for pid in pids or [str(root_pid)]:
        try:
            for line in Path(f"/proc/{pid.strip()}/io").read_text().splitlines():
                if line.startswith("read_bytes:"):
                    total += float(line.split(":")[1])
        except (OSError, ValueError):
            continue
    return total


def cmd_submit(args: argparse.Namespace, command: list[str]) -> int:
    if not command:
        print("error: no command after '--'", file=sys.stderr)
        return 2
    if not Path(args.workdir).is_dir():
        print(f"error: workdir does not exist: {args.workdir}", file=sys.stderr)
        return 2
    attempt_dir = ATTEMPTS / args.attempt_id
    status = read_status(attempt_dir)
    if status.get("state") == "running":
        print(f"error: attempt {args.attempt_id} is already running", file=sys.stderr)
        return 2
    (attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    (attempt_dir / "outputs").mkdir(parents=True, exist_ok=True)

    start_epoch = time.time()
    log_path = attempt_dir / "logs" / "train.log"
    exit_code_path = attempt_dir / "exit_code"
    exit_code_path.unlink(missing_ok=True)
    # Shell shim records the command's exit code so the detached watchdog can
    # distinguish completed from failed after this CLI has exited.
    shim = ['/bin/bash', '-c', '"$@"; rc=$?; echo "$rc" > "$0"; exit "$rc"',
            str(exit_code_path)] + command
    with open(log_path, "ab") as log:
        proc = subprocess.Popen(
            shim,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=args.workdir,
            start_new_session=True,
        )

    write_status(attempt_dir, {
        "attempt_id": args.attempt_id,
        "state": "running",
        "pid": proc.pid,
        "command": command,
        "workdir": args.workdir,
        "start_utc": utc_now(),
        "start_epoch": start_epoch,
        "gpu_hours_cap": args.gpu_hours_cap,
        "topology_gpus": TOPOLOGY_GPUS,
        "poll_seconds": args.poll_seconds,
        "run_kind": args.run_kind,
        "cumulative_read_bytes": 0.0,
        "data_overread_flag": False,
    })

    watchdog = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "watch",
         "--attempt-id", args.attempt_id],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(json.dumps({
        "submitted": args.attempt_id,
        "pid": proc.pid,
        "watchdog_pid": watchdog.pid,
        "gpu_hours_cap": args.gpu_hours_cap,
        "log": str(log_path),
        "status": str(attempt_dir / "status.json"),
    }, indent=2))
    print("Attempt submitted. End your turn now; the harness resumes this "
          "session when status.json reaches a terminal state.")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    attempt_dir = ATTEMPTS / args.attempt_id
    status = read_status(attempt_dir)
    if status.get("state") != "running":
        return 0
    pid = int(status["pid"])
    start_epoch = float(status["start_epoch"])
    cap = float(status["gpu_hours_cap"])
    peak_read = 0.0

    while True:
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        now = time.time()
        charged = charged_gpu_hours(start_epoch, now)
        peak_read = max(peak_read, process_tree_read_bytes(pid))
        overread = bool(TRAIN_BYTES > 0 and peak_read > TRAIN_BYTES * OVERREAD_FACTOR)

        if not alive:
            exit_code_path = attempt_dir / "exit_code"
            exit_code = None
            end_epoch = now
            if exit_code_path.exists():
                try:
                    exit_code = int(exit_code_path.read_text().strip())
                except ValueError:
                    exit_code = None
                # true end time: the shim writes exit_code at process exit,
                # so its mtime avoids charging watchdog poll granularity
                end_epoch = min(now, exit_code_path.stat().st_mtime)
            state = "completed" if exit_code == 0 else "failed"
            status.update({
                "state": state,
                "exit_code": exit_code,
                "end_utc": utc_now(),
                "end_epoch": end_epoch,
                "charged_gpu_hours": round(charged_gpu_hours(start_epoch, end_epoch), 4),
                "cumulative_read_bytes": peak_read,
                "data_overread_flag": overread,
            })
            write_status(attempt_dir, status)
            return 0

        if charged >= cap:
            try:
                os.killpg(pid, signal.SIGTERM)
                time.sleep(60)
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            now = time.time()
            status.update({
                "state": "killed_budget",
                "end_utc": utc_now(),
                "end_epoch": now,
                "charged_gpu_hours": round(charged_gpu_hours(start_epoch, now), 4),
                "cumulative_read_bytes": peak_read,
                "data_overread_flag": overread,
                "note": "terminated by watchdog at the per-attempt GPU-hour cap",
            })
            write_status(attempt_dir, status)
            return 0

        status.update({
            "heartbeat_utc": utc_now(),
            "charged_gpu_hours": round(charged, 4),
            "cumulative_read_bytes": peak_read,
            "data_overread_flag": overread,
        })
        write_status(attempt_dir, status)
        time.sleep(POLL_GRAIN_SECONDS)


def cmd_status(args: argparse.Namespace) -> int:
    attempt_dir = ATTEMPTS / args.attempt_id
    status = read_status(attempt_dir)
    if not status:
        print(f"error: no status for attempt {args.attempt_id}", file=sys.stderr)
        return 1
    print(json.dumps(status, indent=2))
    return 0


def cmd_complete_control(_args: argparse.Namespace) -> int:
    CONTROL.mkdir(parents=True, exist_ok=True)
    (CONTROL / "complete").write_text(utc_now() + "\n", encoding="utf-8")
    print("control marked complete; the harness will finalize this session")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="launch an attempt asynchronously")
    p_submit.add_argument("--attempt-id", required=True)
    p_submit.add_argument("--gpu-hours-cap", type=float, default=DEFAULT_CAP)
    p_submit.add_argument("--poll-seconds", type=int, default=1800)
    p_submit.add_argument("--run-kind",
                          choices=["full_budget", "screening", "generation"],
                          default="full_budget")
    p_submit.add_argument("--workdir", default="/app/project/gpic/baselines/PixelGen")

    p_watch = sub.add_parser("watch", help="(internal) budget watchdog")
    p_watch.add_argument("--attempt-id", required=True)

    p_status = sub.add_parser("status", help="print an attempt's status.json")
    p_status.add_argument("--attempt-id", required=True)

    sub.add_parser("complete-control", help="signal final submission is staged")

    argv = sys.argv[1:]
    command: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, command = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)

    if args.cmd == "submit":
        return cmd_submit(args, command)
    if args.cmd == "watch":
        return cmd_watch(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "complete-control":
        return cmd_complete_control(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
