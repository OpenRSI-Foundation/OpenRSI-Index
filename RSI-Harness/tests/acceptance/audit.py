"""Leftover (A5) and socket (A6) audits, the kill trigger (A7) and verdicts.

scripts/operator/sandbox_acceptance.sh runs, as root, per scenario:

    python -m tests.acceptance.audit watch --data-root D --pid PID --out W \\
        [--kill-at env_create|build|load|paused]
    python -m tests.acceptance.audit leftovers --data-root D --observed W \\
        --run-id RUN [--strict]
    python -m tests.acceptance.audit verdict SCENARIO --dir SCENARIO_DIR

``watch`` follows one run (the only lease under ``<D>/leases``) while the
harness runs: it inspects every container labelled with the run (Work,
Judge, env services, builders) for Docker/containerd/BuildKit sockets and
``DOCKER_HOST``, lists the sockets the Judge and the first children see,
remembers every env/builder network, rule and built image the run had (the
lease forgets removed ones), and for A7 SIGKILLs the harness at the chosen
moment. ``leftovers`` is spec A5 for that run; ``verdict`` turns a
scenario's run log, Agent notes, Judge summary, watch record and leftovers
into PASS/FAIL rows. Only objects labelled with, or derived from, the run
are ever looked at; nothing here removes anything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rsi_harness.runtime.network import firewall_rule_chains, managed_bridge_interface
from rsi_harness.runtime.sandbox_env_contracts import (
    BUILT_IMAGE_REPOSITORY,
    built_image_tag,
)

ENDPOINT_SOCKET = "/run/rsi-harness/sandbox/s"
SOCKET_FIND = (
    "find / \\( -path /proc -o -path /sys \\) -prune -o -type s -print 2>/dev/null"
)
# Host control sockets that must never reach a Work, Judge, child or builder.
FORBIDDEN = ("docker.sock", "containerd", "buildkitd.sock", "/run/buildkit", "podman")
KILL_MOMENTS = ("env_create", "build", "load", "paused")
ENV_NETWORK = re.compile(r"^rsi-sbnet-([0-9a-f]{16})$")
BUILDER_NETWORK = re.compile(r"^rsi-sbbnet-([0-9a-f]{16})$")
Command = Callable[[list[str]], str]


def _command(argv: list[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)}: {result.stderr.strip()}")
    return result.stdout


def run_label(run_id: str) -> dict[str, str]:
    return {"label": f"rsi-harness.run-id={run_id}"}


def built_tag_prefix(run_id: str) -> str:
    """``rsi-sbx-img:<sha256(run)[:12]>-``, the run's part of every built tag."""
    return built_image_tag(run_id, "i" + "0" * 32).removesuffix("0" * 32)


def rule_ids(run_id: str, networks: Iterable[str]) -> list[str]:
    """Each env/builder network's firewall rule (spec 3.2 naming)."""
    rules = []
    for name in networks:
        if match := ENV_NETWORK.match(name):
            rules.append(f"rsi-{run_id}-sbx-{match[1]}")
        elif match := BUILDER_NETWORK.match(name):
            rules.append(f"rsi-{run_id}-sbb-{match[1]}")
    return sorted(rules)


def container_findings(attrs: dict[str, Any]) -> list[str]:
    """A6 for one ``docker inspect``: no control socket mounted, no
    ``DOCKER_HOST``."""
    findings = []
    host = attrs.get("HostConfig") or {}
    sources = [
        f"{item.get('Source', '')} -> {item.get('Destination', '')}"
        for item in attrs.get("Mounts") or []
    ] + list(host.get("Binds") or [])
    for text in sources:
        if any(name in text for name in FORBIDDEN):
            findings.append(f"mount {text}")
    for item in (attrs.get("Config") or {}).get("Env") or []:
        if item.startswith("DOCKER_HOST="):
            findings.append("env DOCKER_HOST")
    return findings


def lease_record(data_root: Path, run_id: str) -> dict[str, Any] | None:
    """The run's lease as JSON (never mutated; the harness owns it)."""
    path = Path(data_root) / "leases" / f"{run_id}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def discover_run(data_root: Path) -> str | None:
    """The one run of a scenario's fresh data root."""
    leases = sorted((Path(data_root) / "leases").glob("*.json"))
    return leases[0].stem if len(leases) == 1 else None


@dataclass
class Observed:
    """What one run had, recorded while it ran (the lease compacts removed
    envs and drops released images)."""

    run_id: str | None = None
    networks: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    loop_devices: list[str] = field(default_factory=list)
    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    sockets: dict[str, dict[str, Any]] = field(default_factory=dict)
    killed_at: str | None = None
    kill_detail: str | None = None

    @classmethod
    def load(cls, path: Path | None) -> Observed:
        if path is None or not Path(path).exists():
            return cls()
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        temporary = Path(f"{path}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=1, sort_keys=True))
        temporary.replace(path)

    def note_lease(self, lease: dict[str, Any]) -> None:
        for image in lease.get("sandbox_images") or []:
            if image.get("kind") == "built" and image.get("image_id"):
                _add(self.images, image["image_id"])
        for env in lease.get("sandbox_envs") or []:
            if env.get("network_name"):
                _add(self.networks, env["network_name"])
        for builder in lease.get("sandbox_builders") or []:
            _add(self.networks, builder["network_name"])
            if builder.get("loop_device"):
                _add(self.loop_devices, builder["loop_device"])


def _add(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def leftovers(
    client: Any,
    run_id: str,
    data_root: Path,
    observed: Observed,
    *,
    strict: bool = False,
    command: Command = _command,
) -> dict[str, Any]:
    """Spec A5: every list must be empty. Pulled images are the host's cache,
    listed apart and never a finding. Before ``rsi-harness cleanup`` the
    Harness still retains the run's Work WORKDIR volume by design: it is
    listed apart unless ``strict`` (the audit after cleanup)."""
    label = run_label(run_id)
    lease = lease_record(data_root, run_id) or {}
    observed.note_lease(lease)
    volumes = client.volumes.list(filters=label)
    retained = [
        item.name
        for item in volumes
        if not strict and item.name.startswith("rsi-harness-workdir-")
    ]
    findings: dict[str, list[str]] = {
        "containers": [
            item.name for item in client.containers.list(all=True, filters=label)
        ],
        "networks": [item.name for item in client.networks.list(filters=label)],
        "volumes": [item.name for item in volumes if item.name not in retained],
    }
    prefix = built_tag_prefix(run_id)
    built = {
        item.id
        for item in client.images.list(
            all=True,
            filters={"label": [label["label"], "rsi-harness.role=sandbox-build"]},
        )
    }
    tagged = [
        tag
        for item in client.images.list(name=BUILT_IMAGE_REPOSITORY)
        for tag in item.tags
        if tag.startswith(prefix)
    ]
    for image_id in observed.images:
        try:
            client.images.get(image_id)
        except Exception:
            continue
        built.add(image_id)
    findings["built images"] = sorted(built) + sorted(tagged)
    bridges = {managed_bridge_interface(name) for name in observed.networks}
    links = command(["ip", "-br", "link"]).splitlines()
    findings["bridges"] = [
        line.split()[0].split("@")[0]
        for line in links
        if line.split() and line.split()[0].split("@")[0] in bridges
    ]
    chains = {
        chain
        for rule in rule_ids(run_id, observed.networks)
        for chain in firewall_rule_chains(rule)
    }
    rules = command(["iptables", "--wait", "-S"]).splitlines()
    findings["iptables"] = [
        line
        for line in rules
        if f"rsi-{run_id}" in line or any(chain in line.split() for chain in chains)
    ]
    # By backing file only: a freed /dev/loopN may already serve someone else.
    run_root = f"{Path(data_root).resolve()}/{run_id}/"
    findings["loop devices"] = [
        line for line in command(["losetup", "-a"]).splitlines() if run_root in line
    ]
    sb = Path(data_root) / run_id / "sb"
    findings["sb"] = [str(sb)] if sb.exists() else []
    pulled = [
        f"{item.get('image_id')} pre_existing={item.get('pre_existing')}"
        for item in lease.get("sandbox_images") or []
        if item.get("kind") == "pulled"
    ]
    return {
        "run_id": run_id,
        "clean": not any(findings.values()),
        "findings": findings,
        "pulled (cache, not leftovers)": pulled,
        "retained until cleanup": retained,
    }


# The host-wide view of the root check (R9): read only, before and after.
# Each object is kept with its owner: the run label of a Docker object or a
# built image, the rule id (``rule:rsi-<run>-...``) of a firewall line, the
# run directory of a loop file; None when nothing names one.
BRIDGE = re.compile(r"^rsi[0-9a-f]{12}$")
RUN_LABEL = "rsi-harness.run-id"
COMMENT = re.compile(r"(?:^|\s)--comment \"?(rsi-[^\s\"]+)")
JUMP = re.compile(r"\s-j (RSI_[FI]_\w+)")
LOOP_RUN = re.compile(r"/([^/\s]+)/sb/build/[^/\s]+\.img")
# The run ids of the D tests and root checks the root check runs
# (m2-net-..., m8-build-..., m9-root-...); the acceptance runs are named by
# their scratch directories instead.
TEST_RUN = re.compile(r"^m\d+-")
Owners = dict[str, dict[str, "str | None"]]


def _labels(item: Any) -> dict[str, str]:
    labels = getattr(item, "labels", None)
    if labels is None:
        labels = (getattr(item, "attrs", None) or {}).get("Labels")
    return labels or {}


def host_snapshot(client: Any, *, command: Command = _command) -> Owners:
    """Every rsi object on the host with its owner: labelled containers,
    volumes and networks, built image tags, RSI_ chains and rsi- jumps,
    managed bridges and builder loop devices."""
    label = {"label": RUN_LABEL}
    networks = {
        item.name: _labels(item).get(RUN_LABEL)
        for item in client.networks.list(filters=label)
    }
    bridges = {managed_bridge_interface(name): run for name, run in networks.items()}
    links = [
        line.split()[0].split("@")[0] for line in _lines(command, "ip", "-br", "link")
    ]
    rules = [
        line
        for line in _lines(command, "iptables", "--wait", "-S")
        if "RSI_" in line or " rsi-" in line
    ]
    # A chain belongs to the rule whose jump names it.
    chains = {
        jump[1]: f"rule:{comment[1]}"
        for line in rules
        if (jump := JUMP.search(line)) and (comment := COMMENT.search(line))
    }

    def rule_owner(line: str) -> str | None:
        if comment := COMMENT.search(line):
            return f"rule:{comment[1]}"
        named = [chains[word] for word in line.split() if word in chains]
        return named[0] if named else None

    loops = [line for line in _lines(command, "losetup", "-a") if "/sb/build/" in line]
    return {
        "containers": {
            item.name: _labels(item).get(RUN_LABEL)
            for item in client.containers.list(all=True, filters=label)
        },
        "volumes": {
            item.name: _labels(item).get(RUN_LABEL)
            for item in client.volumes.list(filters=label)
        },
        "networks": networks,
        "built images": {
            tag: _labels(item).get(RUN_LABEL)
            for item in client.images.list(name=BUILT_IMAGE_REPOSITORY)
            for tag in item.tags
        },
        "iptables": {line: rule_owner(line) for line in rules},
        "bridges": {name: bridges.get(name) for name in links if BRIDGE.match(name)},
        "loop devices": {
            line: (match[1] if (match := LOOP_RUN.search(line)) else None)
            for line in loops
        },
    }


def _lines(command: Command, *argv: str) -> list[str]:
    return [line for line in command(list(argv)).splitlines() if line.split()]


def runs_under(directory: Path) -> set[str]:
    """Run ids with a lease or a log directory under ``directory`` (the
    acceptance runs of the root check's scratch)."""
    root = Path(directory)
    found = {path.stem for path in root.rglob("leases/*.json")}
    found |= {path.name for path in root.rglob("runs/*") if path.is_dir()}
    return found


def is_ours(owner: str | None, runs: Iterable[str] = ()) -> bool:
    """Whether an object belongs to the checks: a test run id, one of
    ``runs``, or no owner at all (a leaked chain or bridge names none, so
    it counts, never excuses)."""
    if owner is None:
        return True
    runs = set(runs)
    if owner.startswith("rule:"):
        rule = owner.removeprefix("rule:")
        return bool(re.match(r"rsi-m\d+-", rule)) or any(
            rule.startswith(f"rsi-{run}-") for run in runs
        )
    return bool(TEST_RUN.match(owner)) or owner in runs


def host_diff(
    before: Owners, after: Owners, runs: Iterable[str] = ()
) -> dict[str, dict[str, list[str]]]:
    """What appeared since ``before`` (and is still there): the checks'
    own objects (R9 fails on any), and other runs' apart (a shared host's
    other runs started meanwhile; informational)."""
    runs = set(runs)
    split: dict[str, dict[str, list[str]]] = {"ours": {}, "other runs": {}}
    for key, value in after.items():
        added = sorted(set(value) - set(before.get(key, {})))
        split["ours"][key] = [item for item in added if is_ours(value[item], runs)]
        split["other runs"][key] = [
            f"{item} (owner {value[item]})"
            for item in added
            if not is_ours(value[item], runs)
        ]
    return split


# -- watch ---------------------------------------------------------------------


def _alive(pid: int) -> bool:
    """Running and not a zombie waiting for its parent to reap it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rindex(")") + 2] not in "ZX"
    except (OSError, ValueError, IndexError):
        return True


def sockets_in(container: Any) -> list[str]:
    result = container.exec_run(["sh", "-c", SOCKET_FIND], user="root")
    return sorted(result.output.decode(errors="replace").split())


def kill_moment(
    moment: str, containers: list[Any], lease: dict[str, Any], top: Callable
) -> str | None:
    """A description when the run is at ``moment`` (spec A7), else None."""
    images = lease.get("sandbox_images") or []
    role = {item.id: item.labels.get("rsi-harness.role") for item in containers}
    if moment == "env_create":
        found = [item.name for item in containers if role[item.id] == "sandbox-env"]
        return f"env container {found[0]} exists" if found else None
    if moment == "build":
        planned = [
            item
            for item in images
            if item.get("kind") == "built" and item.get("state") == "planned"
        ]
        for item in containers:
            if role[item.id] != "sandbox-builder" or item.status != "running":
                continue
            processes = top(item.id).get("Processes") or []
            if planned and any("buildctl" in " ".join(row) for row in processes):
                return f"buildctl running in {item.name}"
        return None
    if moment == "load":
        loading = [
            item["handle"]
            for item in images
            if item.get("kind") == "built" and item.get("state") == "loading"
        ]
        return f"image {loading[0]} loading" if loading else None
    if moment == "paused":
        # freeze_work journals the group once Docker proves it paused.
        journaled = [
            item.get("env_id")
            for item in lease.get("sandbox_envs") or []
            if (item.get("owner") or {}).get("phase") == "work"
            and item.get("state") == "paused"
        ]
        if journaled:
            return f"Work env {journaled[0]} paused"
        paused = [
            item.name
            for item in containers
            if role[item.id] == "sandbox-env"
            and item.labels.get("rsi-harness.sandbox-phase") == "work"
            and item.status == "paused"
        ]
        return f"Work env {', '.join(paused)} paused" if paused else None
    raise ValueError(f"unknown kill moment {moment}")


def watch(
    client: Any,
    data_root: Path,
    pid: int,
    out: Path,
    *,
    kill_at: str | None = None,
    give_up_sec: float = 1800.0,
    interval: float = 0.25,
    children: int = 3,
    clock: Callable[[], float] = time.monotonic,
) -> Observed:
    """Follow the harness ``pid`` until it exits; with ``kill_at``, SIGKILL
    it at that moment, or after ``give_up_sec`` if the moment never comes
    (``killed_at`` stays None: the scenario fails, but still recovers).

    A kill watcher never execs into a container: an exec that a freeze
    catches blocks until the thaw, past the whole paused window (A6 probes
    belong to the a1 watch)."""
    observed = Observed()
    began = clock()
    while _alive(pid):
        observed.run_id = observed.run_id or discover_run(data_root)
        if observed.run_id is None:
            time.sleep(interval)
            continue
        lease = lease_record(data_root, observed.run_id) or {}
        observed.note_lease(lease)
        detail = None
        if kill_at in ("load", "paused"):
            # The load window can be short: the lease alone decides it, as
            # it does the journaled pause.
            detail = kill_moment(kill_at, [], lease, client.api.top)
        label = run_label(observed.run_id)
        try:
            containers = (
                [] if detail else client.containers.list(all=True, filters=label)
            )
            for network in client.networks.list(filters=label):
                if ENV_NETWORK.match(network.name) or BUILDER_NETWORK.match(
                    network.name
                ):
                    _add(observed.networks, network.name)
        except Exception:
            time.sleep(interval)
            continue
        if kill_at is not None and detail is None:
            try:
                detail = kill_moment(kill_at, containers, lease, client.api.top)
            except Exception:
                detail = None
        if kill_at is not None and (detail or clock() - began > give_up_sec):
            os.kill(pid, signal.SIGKILL)
            if detail:
                observed.killed_at, observed.kill_detail = kill_at, detail
            else:
                observed.kill_detail = f"{kill_at} never reached in {give_up_sec:.0f}s"
            observed.save(out)
            return observed
        for item in containers:
            _inspect(observed, item, children if kill_at is None else 0)
        observed.save(out)
        time.sleep(interval)
    observed.save(out)
    return observed


def _inspect(observed: Observed, item: Any, children: int) -> None:
    """Record the container's A6 findings and, while fewer than ``children``
    of its role were, the sockets it sees (0: no probe at all)."""
    role = item.labels.get("rsi-harness.role", "")
    record = observed.containers.setdefault(
        item.id[:12], {"name": item.name, "role": role, "findings": []}
    )
    try:
        for finding in container_findings(item.attrs):
            _add(record["findings"], finding)
        if not children or item.status != "running" or item.name in observed.sockets:
            return
        probed = [value for value in observed.sockets.values() if value["role"] == role]
        if role == "judge" or (role == "sandbox-env" and len(probed) < children):
            observed.sockets[item.name] = {"role": role, "sockets": sockets_in(item)}
    except Exception as error:
        record.setdefault("errors", []).append(str(error)[:200])


# -- verdicts ------------------------------------------------------------------


def run_result(log: str) -> dict[str, Any]:
    """Status and Judge rounds from ``rsi-harness run``'s console output."""
    status = re.search(r"^\s*Status:\s+(\S+)", log, re.MULTILINE)
    rounds = re.findall(r"^\s+(\S+): (\S+) score=(\S+)$", log, re.MULTILINE)
    return {
        "status": status[1] if status else None,
        "rounds": [
            {"round": name, "status": state, "score": score}
            for name, state, score in rounds
        ],
    }


def notes(text: str) -> dict[str, list[str]]:
    """``RSI-ACCEPTANCE <key> <value>`` lines of the Agent output."""
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("RSI-ACCEPTANCE "):
            key, _, value = line.removeprefix("RSI-ACCEPTANCE ").partition(" ")
            found.setdefault(key, []).append(value.strip())
    return found


def _read(directory: Path, pattern: str) -> str:
    found = sorted(directory.rglob(pattern))
    return found[-1].read_text(errors="replace") if found else ""


def _judged(result: dict[str, Any]) -> tuple[bool, str]:
    rounds = result["rounds"]
    ok = bool(rounds) and all(
        item["status"] == "completed" and item["score"] == "1.0" for item in rounds
    )
    detail = ", ".join(f"{r['round']} {r['status']} {r['score']}" for r in rounds)
    return ok, detail or f"no Judge round (status {result['status']})"


def _suites(summary_text: str) -> str:
    try:
        summary = json.loads(summary_text)
    except ValueError:
        return "no Judge summary"
    return "; ".join(
        f"{suite}: oracle {report['oracle']['trials']}/{report['oracle']['expected']}"
        f" {'ok' if report['oracle']['ok'] else 'FAIL'},"
        f" nop {'ok' if report['nop']['ok'] else 'FAIL'}"
        for suite, report in summary.get("suites", {}).items()
    ) or str(summary.get("error", "empty summary"))


def _a6(observed: Observed, found: dict[str, list[str]]) -> tuple[bool, str]:
    problems = [
        f"{record['name']}: {finding}"
        for record in observed.containers.values()
        for finding in record["findings"]
    ]
    judges = [v for v in observed.sockets.values() if v["role"] == "judge"]
    children = [v for v in observed.sockets.values() if v["role"] == "sandbox-env"]
    if not judges or any(v["sockets"] != [ENDPOINT_SOCKET] for v in judges):
        problems.append(f"Judge sockets {[v['sockets'] for v in judges]}")
    if not children or any(v["sockets"] for v in children):
        problems.append(f"child sockets {[v['sockets'] for v in children]}")
    try:
        work = json.loads(found.get("a6-work", ["{}"])[-1])
    except ValueError:
        work = {}
    if not work.get("ok"):
        problems.append(f"Work probe {work or 'missing'}")
    if found.get("a6-run-step") != ["pass"]:
        problems.append(f"RUN step {found.get('a6-run-step')}")
    if found.get("a6-child-sockets") != ["none"]:
        problems.append(f"Work child sockets {found.get('a6-child-sockets')}")
    # Spec A6: the Engine path answers 400 unsupported, whatever the method.
    engine = work.get("engine_paths") or {}
    for path in ("POST /v1/containers/json", "GET /v1/containers/json"):
        if engine.get(path) != 400:
            problems.append(f"{path} answered {engine.get(path)}, not 400")
    inspected = len(observed.containers)
    detail = "; ".join(problems) or (
        f"{inspected} containers inspected; Judge sees only {ENDPOINT_SOCKET}; "
        f"{len(children)} children and a RUN step see none"
    )
    return not problems, detail


# The Harness failing closed or losing an outcome, in its console (the
# broker's warnings) or a Harbor trial log.
FAULT = re.compile(
    r"quarantined|unknown-outcome|recovery_required|requires recovery"
    r"|fail(?:s|ed)?[ -]closed",
    re.IGNORECASE,
)
TEARDOWN_SEC = 60.0


def faults(directory: Path) -> list[str]:
    """Up to five lines of the run showing a quarantine, an unknown outcome
    or a fail-closed recovery (spec M7: none in a normal run)."""
    directory = Path(directory)
    files = [directory / "run.log", *sorted((directory / "logs").rglob("*.log"))]
    found = []
    for path in files:
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        found += [
            f"{path.name}: {line.strip()[:160]}"
            for line in text.splitlines()
            if FAULT.search(line)
        ]
    return found[:5]


def _moment(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def teardowns(directory: Path) -> dict[str, float]:
    """Per oracle trial of the Judge's Harbor jobs: seconds from the end of
    its verifier to the end of the trial, which is when Harbor has stopped
    (destroyed) its environment."""
    found = {}
    for path in sorted(Path(directory).rglob("harbor-jobs/*-oracle/*/result.json")):
        try:
            result = json.loads(path.read_text())
            ended = (result.get("verifier") or {}).get("finished_at")
            found[result["trial_name"]] = (
                _moment(result["finished_at"]) - _moment(ended)
            ).total_seconds()
        except (OSError, ValueError, KeyError, TypeError):
            found[path.parent.name] = float("inf")
    return found


def _normal_run(directory: Path) -> tuple[bool, str]:
    """No fault anywhere, and every oracle env gone within 60 s of its
    verifier (spec M7's A1 path)."""
    problems = faults(directory)
    ended = teardowns(directory)
    if not ended:
        problems.append("no oracle trial result")
    slow = {name: round(sec, 1) for name, sec in ended.items() if sec > TEARDOWN_SEC}
    if slow:
        problems.append(f"env teardown over {TEARDOWN_SEC:.0f} s: {slow}")
    if problems:
        return False, "; ".join(problems)
    return True, (
        f"{len(ended)} oracle envs destroyed within {max(ended.values()):.1f} s; "
        "no quarantine or unknown outcome"
    )


def _submitted(directory: Path) -> list[dict[str, Any]]:
    """The Judge rounds' submission reports (the Harness's own record)."""
    reports = []
    for path in sorted(Path(directory).rglob("submissions/*/report.json")):
        try:
            reports.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return reports


def _a8(directory: Path, found: dict[str, list[str]]) -> tuple[bool, str]:
    """The submit lands about 60 s before the Work deadline, the Judge
    round starts before that deadline and ends after it."""
    try:
        deadline = float(found["work-deadline"][-1])
        submit = float(found["late-submit"][-1])
    except (KeyError, IndexError, ValueError):
        return False, f"no Work deadline or submit note ({sorted(found)})"
    reports = _submitted(directory)
    if not reports:
        return False, "no Judge round report"
    report = reports[-1]
    ended = float(report.get("submitted_at") or 0.0)
    began = ended - float(report.get("runtime_seconds") or 0.0)
    lead = deadline - submit
    ok = 30.0 <= lead <= 90.0 and began < deadline < ended
    return ok, (
        f"submitted {lead:.0f} s before the Work deadline; Judge round "
        f"{began - deadline:+.0f} s .. {ended - deadline:+.0f} s around it"
    )


# Where the demo's Work leaves the checkpoint: the Judge serves it from its
# read-only WORKDIR snapshot (sample_tasks/vllm-in-judge).
VLLM_CHECKPOINT = "/workspace/checkpoint"
VLLM_PLUGIN = "rsi_sandbox_harbor:ManagedSandboxEnvironment"


def _vllm(
    directory: Path,
    result: dict[str, Any],
    found: dict[str, list[str]],
    observed: Observed,
) -> list[tuple[str, bool, str]]:
    """The vLLM-in-Judge demo (scripts/operator/vllm_demo.sh): Work placed
    the checkpoint and submitted; vLLM served that checkpoint on a GPU the
    Harness gave the Judge; every trial's agent made requests; every trial
    ran to completion in a broker-created environment without an
    infrastructure error. The score itself does not matter."""
    try:
        summary = json.loads(_read(Path(directory) / "logs", "vllm-demo-summary.json"))
    except ValueError:
        summary = {}
    downloads = found.get("checkpoint-download", [])
    # The Harness's own record of the submission stands in for the note
    # agent.sh prints after `rsi-submit` returns (it never does when the
    # Work deadline passes during the round).
    reports = _submitted(directory)
    submitted = found.get("submit-exit") == ["0"] or bool(reports)
    work = bool(downloads) and downloads[-1].startswith("0 ") and submitted
    rows = [
        (
            "V0 Work",
            work,
            f"download {downloads[-1] if downloads else 'missing'}; "
            f"{(found.get('checkpoint-bytes') or ['?'])[-1]} bytes; "
            f"submit exit {found.get('submit-exit')}, "
            f"{len(reports)} submission report(s)",
        )
    ]
    vllm = summary.get("vllm") or {}
    served = {
        gpu["uuid"]: gpu for gpu in vllm.get("gpus") or [] if isinstance(gpu, dict)
    }
    serving = list(vllm.get("serving_gpus") or [])
    expected = list(vllm.get("expected_gpus") or [])
    gpus = ", ".join(
        f"{uuid} {served.get(uuid, {}).get('name')} "
        f"+{served.get(uuid, {}).get('vllm_mib')} MiB"
        for uuid in serving
    )
    roots = [item.get("root") for item in vllm.get("models") or []]
    rows.append(
        (
            "V1 vLLM",
            bool(vllm.get("on_judge_gpu"))
            and bool(serving)
            and bool(expected)
            and set(serving) <= set(expected)
            and vllm.get("checkpoint") == VLLM_CHECKPOINT
            and roots == [VLLM_CHECKPOINT],
            f"healthy {vllm.get('healthy')}; serving on {gpus or 'no GPU'}; "
            f"Judge GPUs {expected}; model roots {roots}",
        )
    )
    counted = summary.get("completions") or {}
    requests = counted.get("requests_served") or 0
    trials = summary.get("trials") or []
    silent = [item.get("task") for item in trials if not item.get("requested")]
    rows.append(
        (
            "V2 requests",
            requests > 0 and not silent,
            f"{requests} completions served (metrics; "
            f"{counted.get('requests_aborted')} aborted apart), "
            f"{counted.get('access_log_200')} in the access log; "
            f"tokens in {counted.get('prompt_tokens')}, "
            f"out {counted.get('generation_tokens')}; per trial in "
            + (
                ", ".join(
                    f"{item.get('task')} {item.get('input_tokens')}" for item in trials
                )
                or "none"
            )
            + (f"; no request from {silent}" if silent else ""),
        )
    )
    rounds = result["rounds"]
    finished = bool(rounds) and all(item["status"] == "completed" for item in rounds)
    problems = list(summary.get("infra_errors") or []) + faults(directory)
    if not trials:
        problems.append("no trial result")
    if not finished:
        problems.append(f"Judge round {rounds or result['status']}")
    # Through the sandbox: every trial in the plugin's environment, the
    # broker created an environment container for each, and the Judge had
    # the endpoint and no host control socket (vLLM's own IPC sockets in
    # its /tmp are its business) nor any A6 finding.
    outside = [
        item.get("task") for item in trials if item.get("environment") != VLLM_PLUGIN
    ]
    if outside:
        problems.append(f"not through {VLLM_PLUGIN}: {outside}")
    envs = sorted(
        record["name"]
        for record in observed.containers.values()
        if record["role"] == "sandbox-env"
    )
    if len(envs) < len(trials):
        problems.append(
            f"{len(envs)} sandbox env container(s) for {len(trials)} trials"
        )
    judges = [v["sockets"] for v in observed.sockets.values() if v["role"] == "judge"]
    if not judges or any(
        ENDPOINT_SOCKET not in sockets
        or any(name in path for path in sockets for name in FORBIDDEN)
        for sockets in judges
    ):
        problems.append(f"Judge sockets {judges}")
    problems += [
        f"{record['name']}: {finding}"
        for record in observed.containers.values()
        for finding in record["findings"]
    ]
    outcomes = "; ".join(
        f"{item.get('task')} {item.get('outcome')} reward {item.get('reward')}"
        + (f" ({item['exception']})" if item.get("exception") else "")
        for item in trials
    )
    rows.append(
        (
            "V3 trials",
            not problems and all(item.get("completed") for item in trials),
            f"{outcomes or 'none'}; reward {summary.get('reward')}; "
            f"{len(envs)} broker env container(s); Judge sees {ENDPOINT_SOCKET}"
            + (
                f"; {'; '.join(str(item) for item in problems)[:300]}"
                if problems
                else "; no infrastructure error"
            ),
        )
    )
    return rows


def verdict(scenario: str, directory: Path) -> list[tuple[str, bool, str]]:
    """PASS/FAIL rows of one scenario of scripts/operator/sandbox_acceptance.sh."""
    directory = Path(directory)
    log = (directory / "run.log").read_text(errors="replace")
    result = run_result(log)
    found = notes(_read(directory / "logs", "agent_output.txt"))
    summary = _read(directory / "logs", "harbor-summary.json")
    observed = Observed.load(directory / "watch.json")
    try:
        audit = json.loads((directory / "leftovers.json").read_text())
    except (OSError, ValueError):
        audit = {"clean": False, "findings": {"audit": ["missing"]}}
    dirty = {key: value for key, value in audit["findings"].items() if value}
    detail = "no leftovers" if audit["clean"] else json.dumps(dirty)[:300]
    clean = audit["clean"]
    after = directory / "leftovers-after-cleanup.json"
    if after.exists():
        # After `rsi-harness cleanup`: nothing labelled with the run at all.
        try:
            final = json.loads(after.read_text())
        except ValueError:
            final = {"clean": False, "findings": {"audit": ["unreadable"]}}
        if not final["clean"]:
            left = {key: value for key, value in final["findings"].items() if value}
            detail += "; after cleanup: " + json.dumps(left)[:200]
        else:
            detail += "; none after cleanup"
        clean = clean and final["clean"]
    a5 = (f"A5 {scenario}", clean, detail)
    judged, rounds = _judged(result)
    rows: list[tuple[str, bool, str]] = []
    if scenario == "a1":
        normal, how = _normal_run(directory)
        rows.append(("A1", judged and normal, f"{rounds}; {_suites(summary)}; {how}"))
        rows.append(("A6", *_a6(observed, found)))
    elif scenario == "a2":
        work = dict(item.split(" ", 1) for item in found.get("work-suite", []))
        tb2 = work.get("tb2") == "oracle=True nop=True"
        compose = work.get("compose") == "oracle=True nop=True"
        rows.append(("A2", tb2, f"Work tb2: {work.get('tb2', 'not run')}"))
        rows.append(
            (
                "A3",
                compose and judged,
                f"Work compose: {work.get('compose', 'not run')}; Judge {rounds}; "
                f"{_suites(summary)}",
            )
        )
    elif scenario == "a4":
        rows.append(("A4", judged, f"{rounds}; {_suites(summary)}"))
    elif scenario == "swebench":
        # The SWE-bench sample: its tasks offline, oracle 1 and nop 0.
        normal, how = _normal_run(directory)
        rows.append(("SWE", judged and normal, f"{rounds}; {_suites(summary)}; {how}"))
    elif scenario.startswith("a7-"):
        moment = scenario.removeprefix("a7-").replace("-", "_")
        recovered = (directory / "recover.rc").read_text().strip() == "0"
        killed = observed.killed_at == moment
        rows.append(
            (
                f"A7 {moment}",
                killed and recovered and audit["clean"],
                f"killed: {observed.kill_detail or 'never'}; "
                f"recover {'ok' if recovered else 'FAILED'}",
            )
        )
    elif scenario == "vllm":
        rows.extend(_vllm(directory, result, found, observed))
    elif scenario == "a8":
        timed, when = _a8(directory, found)
        rows.append(("A8", judged and timed, f"{when}; {rounds}"))
    else:
        raise ValueError(f"unknown scenario {scenario}")
    rows.append(a5)
    return rows


# -- CLI -----------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    """The CLI of every command the operator scripts run."""
    parser = argparse.ArgumentParser(prog="python -m tests.acceptance.audit")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("watch")
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--pid", type=int, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--kill-at", choices=KILL_MOMENTS)
    command.add_argument("--give-up-sec", type=float, default=1800.0)
    command = commands.add_parser("run-id")
    command.add_argument("--data-root", type=Path, required=True)
    command = commands.add_parser("leftovers")
    command.add_argument("--data-root", type=Path, required=True)
    command.add_argument("--observed", type=Path)
    command.add_argument("--run-id")
    # After `rsi-harness cleanup`: the retained Work WORKDIR volume counts.
    command.add_argument("--strict", action="store_true")
    commands.add_parser("host-snapshot")
    command = commands.add_parser("host-diff")
    command.add_argument("before", type=Path)
    command.add_argument("--runs-under", type=Path)
    command = commands.add_parser("verdict")
    command.add_argument("scenario")
    command.add_argument("--dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run-id":
        run_id = discover_run(args.data_root)
        if run_id is None:
            return 1
        print(run_id)
        return 0
    if args.command == "verdict":
        for name, ok, detail in verdict(args.scenario, args.dir):
            print(f"{name}|{'PASS' if ok else 'FAIL'}|{detail}")
        return 0
    import docker

    client = docker.from_env(timeout=120)
    if args.command == "host-snapshot":
        snapshot = host_snapshot(client, command=_command)
        print(json.dumps(snapshot, indent=1, sort_keys=True))
        return 0
    if args.command == "host-diff":
        before = json.loads(args.before.read_text())
        runs = runs_under(args.runs_under) if args.runs_under else set()
        added = host_diff(before, host_snapshot(client, command=_command), runs)
        print(json.dumps(added, indent=1))
        return 1 if any(added["ours"].values()) else 0
    if args.command == "watch":
        watch(
            client,
            args.data_root,
            args.pid,
            args.out,
            kill_at=args.kill_at,
            give_up_sec=args.give_up_sec,
        )
        return 0
    observed = Observed.load(args.observed)
    run_id = args.run_id or observed.run_id or discover_run(args.data_root)
    if run_id is None:
        print(json.dumps({"clean": False, "findings": {"run": ["no run found"]}}))
        return 1
    report = leftovers(
        client, run_id, args.data_root, observed, strict=args.strict, command=_command
    )
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    sys.exit(main())
