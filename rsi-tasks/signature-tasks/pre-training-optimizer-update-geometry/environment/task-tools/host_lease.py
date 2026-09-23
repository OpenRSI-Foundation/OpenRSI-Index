#!/usr/bin/env python3
"""Atomically lease whole Blue Vela nodes from one enclosing LSF allocation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import uuid
from pathlib import Path
from typing import Any


GPUS_PER_NODE = 8
# This node reproduced a CUDA illegal-memory-access failure twice at the same
# local rank during the scaling-ladder run.  Keep it out of subsequent leases;
# HOST_LEASE_EXCLUDED_HOSTS can add more comma- or whitespace-separated nodes.
# p2-r18-n1 failed three consecutive pre-update torch rendezvous attempts on
# distinct ports (39190, 35350, and 37086) with EADDRINUSE on job 560303.
# p4-r03-n3 produced repeatable cost-invalid steady-state slowdowns on two E1
# screens and one E2 screen (roughly 1.5-2x healthy-node step time).
# p5-r27-n4 entered a resumed allocation while a preempted foreign job still
# retained 40-73 GiB on every GPU, so it is unsafe for nested training.
DEFAULT_EXCLUDED_HOSTS = {
    "p1-r10-n4",
    "p2-r18-n1",
    "p3-r26-n4",
    "p4-r03-n3",
    "p5-r27-n4",
}


def excluded_hosts() -> set[str]:
    configured = os.environ.get("HOST_LEASE_EXCLUDED_HOSTS", "")
    return DEFAULT_EXCLUDED_HOSTS | set(configured.replace(",", " ").split())


def parse_hosts(value: str) -> list[tuple[str, int]]:
    fields = value.split()
    if not fields or len(fields) % 2:
        raise ValueError("LSB_MCPU_HOSTS must contain HOST SLOTS pairs")
    result: list[tuple[str, int]] = []
    positions: dict[str, int] = {}
    for index in range(0, len(fields), 2):
        host = fields[index]
        slots = int(fields[index + 1])
        if slots <= 0:
            raise ValueError(f"host {host} has non-positive slots: {slots}")
        if host in positions:
            position = positions[host]
            old_host, old_slots = result[position]
            result[position] = (old_host, old_slots + slots)
        else:
            positions[host] = len(result)
            result.append((host, slots))
    return result


def read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "leases": {}}
    payload = json.loads(path.read_text())
    if payload.get("version") != 1 or not isinstance(payload.get("leases"), dict):
        raise ValueError(f"invalid host-lease state: {path}")
    if not all(isinstance(lease, dict) for lease in payload["leases"].values()):
        raise ValueError(f"invalid lease entry in {path}")
    return payload


def write_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def with_locked_state(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    handle = lock_path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def prune_stale_local_leases(state: dict[str, Any]) -> None:
    """Drop leases whose owning wrapper has exited in this PID namespace."""

    local_host = socket.gethostname()
    stale: list[str] = []
    for token, lease in state["leases"].items():
        if lease.get("owner_host") != local_host:
            continue
        pid = lease.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            stale.append(token)
        except PermissionError:
            # A live process owned by another identity is not ours to reclaim.
            continue
    for token in stale:
        del state["leases"][token]


def acquire(args: argparse.Namespace) -> None:
    if args.slots <= 0 or args.slots % GPUS_PER_NODE:
        raise SystemExit(f"--slots must be a positive multiple of {GPUS_PER_NODE}")
    hosts = parse_hosts(args.lsb_hosts)
    excluded = excluded_hosts()
    eligible = [
        host for host, slots in hosts
        if slots >= GPUS_PER_NODE and host not in excluded
    ]
    required_nodes = args.slots // GPUS_PER_NODE
    if len(eligible) < required_nodes:
        raise SystemExit(
            f"allocation exposes only {len(eligible)} whole GPU nodes; need {required_nodes}"
        )
    handle = with_locked_state(args.state)
    try:
        state = read_state(args.state)
        prune_stale_local_leases(state)
        used = {
            host
            for lease in state["leases"].values()
            for host in lease.get("hosts", [])
        }
        selected = [host for host in eligible if host not in used][:required_nodes]
        if len(selected) != required_nodes:
            free = len([host for host in eligible if host not in used])
            raise SystemExit(
                f"insufficient free nodes: need {required_nodes}, currently free {free}; "
                "wait for another attempt to finish"
            )
        token = uuid.uuid4().hex
        state["leases"][token] = {
            "attempt_id": args.attempt_id,
            "pid": args.pid,
            "owner_host": socket.gethostname(),
            "slots": args.slots,
            "hosts": selected,
        }
        write_state(args.state, state)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
    lsb_hosts = " ".join(f"{host} {GPUS_PER_NODE}" for host in selected)
    print(f"{token}\t{lsb_hosts}")


def release(args: argparse.Namespace) -> None:
    handle = with_locked_state(args.state)
    try:
        state = read_state(args.state)
        lease = state["leases"].get(args.token)
        if lease is None:
            return
        if lease.get("attempt_id") != args.attempt_id:
            raise SystemExit("lease token belongs to a different attempt")
        del state["leases"][args.token]
        write_state(args.state, state)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def list_leases(args: argparse.Namespace) -> None:
    handle = with_locked_state(args.state)
    try:
        state = read_state(args.state)
        write_state(args.state, state)
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
    print(json.dumps(state, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--state", type=Path, required=True)
    acquire_parser.add_argument("--attempt-id", required=True)
    acquire_parser.add_argument("--slots", type=int, required=True)
    acquire_parser.add_argument("--pid", type=int, default=os.getpid())
    acquire_parser.add_argument("--lsb-hosts", required=True)
    acquire_parser.set_defaults(handler=acquire)
    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--state", type=Path, required=True)
    release_parser.add_argument("--attempt-id", required=True)
    release_parser.add_argument("--token", required=True)
    release_parser.set_defaults(handler=release)
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--state", type=Path, required=True)
    list_parser.set_defaults(handler=list_leases)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
