"""Probe vLLM and report the demo (the sample's own logic, not the Harness's).

    demo_report.py wait --base URL --pid PID --timeout SEC [--path /health]
    demo_report.py fetch --base URL --path /metrics --out FILE
    demo_report.py tasks --tasks DIR [--only TASK,...]
    demo_report.py report --evidence DIR --jobs DIR --job NAME --tasks DIR
        --checkpoint PATH --environment IMPORT_PATH [--only TASK,...]
        [--harbor-exit N] [--usage FILE] --reward-json FILE --summary FILE

``wait`` returns once ``--path`` (vLLM's ``/health``, or the metering
proxy's ``/metering/health``) answers 200, and fails when the server
process ends first or the time is up. ``tasks`` prints the names of the
tasks to run, a line each, and their metering keys (``slug``). ``report``
turns what the fixed procedure
collected (``nvidia-smi`` before and after vLLM came up, its model list,
its Prometheus metrics, its log, a Harbor job per task under ``--jobs``/
``--job``, the metering proxy's usage log) into the reward, the fraction of
the expected trials that scored 1, and a summary: vLLM came up on a GPU the
Harness gave the Judge and served the checkpoint, how many completions it
served (aborted ones apart), each trial's outcome, the environment it ran
in, the model requests it made and the tokens metered for it, accuracy
against tokens, and whether any infrastructure error happened. Standard
library only: it runs in the Judge (and in Work) with the system Python,
next to Harbor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

# A trial exception that is the agent's (or its model's) outcome, not the
# infrastructure's: Harbor still verifies after the first two.
AGENT_OUTCOMES = frozenset(
    {
        "AgentTimeoutError",
        "NonZeroAgentExitCodeError",
        "ContextLengthExceededError",
        "OutputLengthExceededError",
    }
)
# Harbor 0.21's verifier outcomes that what the agent left in its
# environment can cause (a blocking process, a broken pip or uv): completed,
# unsolved, and listed apart. A failure to copy the tests in or the
# verifier's logs out stays the infrastructure's.
VERIFIER_OUTCOMES = frozenset(
    {
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
    }
)
# What Harbor exits with when test.sh's time limit stops it (timeout's
# TERM, then its KILL).
HARBOR_TIMED_OUT = frozenset({124, 137})
# GPU memory vLLM must hold on a Judge GPU to count as up there (weights and
# KV cache of even a 0.5B model are far above it).
GPU_MIN_MIB = 1024
ACCESS_LOG = re.compile(r'"POST /v1/(?:chat/)?completions HTTP/1\.1" 200')


def _get(url: str, timeout: float = 10.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _alive(pid: int) -> bool:
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


def wait(
    base: str, pid: int, timeout: float, interval: float = 2.0, path: str = "/health"
) -> int:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if not _alive(pid):
            print(f"{base} (pid {pid}) exited before it was healthy", file=sys.stderr)
            return 1
        try:
            status, _ = _get(f"{base}{path}", timeout=5.0)
        except (OSError, ValueError):
            status = None
        if status == 200:
            print(f"{base} healthy after {time.monotonic() - started:.0f} s")
            return 0
        time.sleep(interval)
    print(f"{base} not healthy within {timeout:.0f} s", file=sys.stderr)
    return 1


def fetch(base: str, path: str, out: Path) -> int:
    try:
        status, body = _get(f"{base}{path}")
    except (OSError, ValueError) as error:
        print(f"GET {path}: {error}", file=sys.stderr)
        return 1
    out.write_bytes(body)
    return 0 if status == 200 else 1


# -- report ----------------------------------------------------------------------


def task_name(task_dir: Path) -> str:
    """Harbor's name of a task: ``[task] name`` in task.toml, else the
    directory's name."""
    try:
        config = tomllib.loads((task_dir / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return task_dir.name
    name = (config.get("task") or {}).get("name")
    return name if isinstance(name, str) and name else task_dir.name


def slug(name: str) -> str:
    """A task's metering key: its name as one URL path segment (the base
    path ``/t/<slug>/v1`` its agent's requests are attributed by)."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)


def expected_tasks(tasks: Path, only: list[str]) -> list[str]:
    names = sorted(
        task_name(child)
        for child in tasks.iterdir()
        if child.is_dir() and (child / "task.toml").is_file()
    )
    if only:
        missing = sorted(set(only) - set(names))
        if missing:
            raise ValueError(f"{tasks}: no task named {', '.join(missing)}")
        names = [name for name in names if name in only]
    return names


def gpu_table(path: Path) -> dict[str, dict]:
    """``nvidia-smi --query-gpu=index,uuid,name,memory.used
    --format=csv,noheader,nounits`` by UUID."""
    table = {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return table
    for line in lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            continue
        try:
            used = int(float(fields[3]))
        except ValueError:
            continue
        table[fields[1]] = {"index": fields[0], "name": fields[2], "used_mib": used}
    return table


def metric_sum(
    text: str, name: str, *, exclude: str | None = None, only: str | None = None
) -> float | None:
    """Sum of the series of one Prometheus metric, None if absent; without
    the series whose labels contain ``exclude``, or only those that contain
    ``only`` (e.g. ``finished_reason="abort"``)."""
    total, seen = 0.0, False
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        match = re.match(rf"^{re.escape(name)}(\{{[^}}]*\}})?\s+(\S+)", line)
        if not match:
            continue
        labels = match[1] or ""
        if (exclude and exclude in labels) or (only and only not in labels):
            seen = True
            continue
        try:
            total += float(match[2])
        except ValueError:
            continue
        seen = True
    return total if seen else None


def _read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def vllm_evidence(evidence: Path, checkpoint: str) -> dict:
    before = gpu_table(evidence / "gpu-before.csv")
    after = gpu_table(evidence / "gpu-after.csv")
    expected = [
        uuid
        for uuid in os.environ.get("RSI_HARNESS_EXPECTED_GPU_UUIDS", "").split(",")
        if uuid
    ]
    gpus = []
    for uuid, row in after.items():
        grown = row["used_mib"] - before.get(uuid, {}).get("used_mib", 0)
        gpus.append({"uuid": uuid, **row, "vllm_mib": grown})
    serving = [gpu["uuid"] for gpu in gpus if gpu["vllm_mib"] >= GPU_MIN_MIB]
    try:
        models = json.loads(_read(evidence / "models.json"))["data"]
    except (ValueError, KeyError, TypeError):
        models = []
    healthy = (evidence / "healthy").exists()
    served = [
        {"id": item.get("id"), "root": item.get("root")}
        for item in models
        if isinstance(item, dict)
    ]
    return {
        "healthy": healthy,
        "checkpoint": checkpoint,
        "models": served,
        # vLLM serves exactly the checkpoint Work left (its read-only
        # WORKDIR snapshot here).
        "served_checkpoint": [item["root"] for item in served] == [checkpoint],
        "gpus": gpus,
        "expected_gpus": expected,
        "serving_gpus": serving,
        # Only on GPUs the Harness named for this Judge.
        "on_judge_gpu": healthy
        and bool(serving)
        and bool(expected)
        and all(uuid in expected for uuid in serving),
    }


ABORTED = 'finished_reason="abort"'


def completions(evidence: Path) -> dict:
    metrics = _read(evidence / "metrics.txt")
    # Finished with stop or length: an aborted request served nothing.
    served = metric_sum(metrics, "vllm:request_success_total", exclude=ABORTED)
    aborted = metric_sum(metrics, "vllm:request_success_total", only=ABORTED)
    return {
        "requests_served": int(served) if served is not None else None,
        "requests_aborted": int(aborted) if aborted is not None else None,
        "prompt_tokens": metric_sum(metrics, "vllm:prompt_tokens_total"),
        "generation_tokens": metric_sum(metrics, "vllm:generation_tokens_total"),
        "access_log_200": len(ACCESS_LOG.findall(_read(evidence / "vllm.log"))),
    }


def _environment(data: dict) -> str | None:
    """The environment Harbor ran the trial in: the import path of a
    plugin, else its built-in type."""
    config = (data.get("config") or {}).get("environment") or {}
    return config.get("import_path") or config.get("type")


def _count(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def trial_outcome(data: dict, environment: str) -> dict:
    """One trial: ``outcome`` is ``verified``, ``agent:<exception>`` or
    ``verifier:<exception>`` (the model's doing, unsolved),
    ``infra:<exception>``, or ``unverified`` (no exception, no reward). A
    trial completed when it ran in ``environment`` (the sandbox plugin), its
    agent asked the model at least once, and it was verified or stopped by
    a model outcome."""
    error = data.get("exception_info") or None
    kind = error.get("exception_type") if isinstance(error, dict) else None
    rewards = (data.get("verifier_result") or {}).get("rewards") or {}
    try:
        reward = float(rewards.get("reward"))
    except (TypeError, ValueError):
        reward = None
    agent = data.get("agent_result") or {}
    ran_in = _environment(data)
    if error is None:
        outcome = "verified" if reward is not None else "unverified"
    elif kind in AGENT_OUTCOMES:
        outcome = f"agent:{kind}"
    elif kind in VERIFIER_OUTCOMES:
        outcome = f"verifier:{kind}"
    else:
        outcome = f"infra:{kind}"
    infra = outcome.startswith("infra:")
    requested = _count(agent.get("n_input_tokens")) > 0
    return {
        "trial": data.get("trial_name"),
        "task": data.get("task_name"),
        "environment": ran_in,
        "through_sandbox": ran_in == environment,
        "reward": reward,
        "solved": reward is not None and reward >= 1.0,
        "outcome": outcome,
        "exception": (
            f"{kind}: {str(error.get('exception_message'))[:300]}" if error else None
        ),
        "infra_error": infra,
        "requested": requested,
        "completed": ran_in == environment
        and requested
        and (outcome == "verified" or outcome.startswith(("agent:", "verifier:"))),
        "input_tokens": agent.get("n_input_tokens"),
        "output_tokens": agent.get("n_output_tokens"),
    }


def trials(job: Path, environment: str) -> list[dict]:
    """The trials of the Harbor jobs under ``job``: one job per task (named
    by its slug), each with its trial directories."""
    found = []
    for path in sorted(job.glob("*/*/result.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            found.append(
                {
                    "trial": path.parent.name,
                    "task": None,
                    "environment": None,
                    "through_sandbox": False,
                    "reward": None,
                    "solved": False,
                    "outcome": "infra:unreadable result.json",
                    "exception": "unreadable result.json",
                    "infra_error": True,
                    "requested": False,
                    "completed": False,
                }
            )
            continue
        if isinstance(data, dict) and "trial_name" in data:
            found.append(trial_outcome(data, environment))
    return found


def metering(path: Path | None) -> dict[str | None, dict]:
    """The metering proxy's usage log summed by trial key (None for requests
    outside any per-trial base path): requests, those that did not end in a
    clean 200, prompt and completion tokens, and their latency."""
    totals: dict[str | None, dict] = {}
    lines = _read(path).splitlines() if path else []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        total = totals.setdefault(
            record.get("trial"),
            {
                "requests": 0,
                "failed": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "latency_ms": 0,
            },
        )
        total["requests"] += 1
        total["failed"] += record.get("status") != 200 or bool(record.get("error"))
        for key in ("prompt_tokens", "completion_tokens", "latency_ms"):
            total[key] += _count(record.get(key))
    return totals


def accuracy_vs_tokens(outcomes: list[dict], expected: list[str]) -> dict:
    """The score against the tokens the model spent on it, over the
    expected trials (metered at the proxy)."""
    counted = [item for item in outcomes if item["task"] in expected]
    solved = sum(1 for item in counted if item["solved"])
    prompt = sum(item["metered"]["prompt_tokens"] for item in counted)
    completion = sum(item["metered"]["completion_tokens"] for item in counted)
    return {
        "trials": len(expected),
        "solved": solved,
        "solved_fraction": solved / len(expected) if expected else 0.0,
        "requests": sum(item["metered"]["requests"] for item in counted),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "completion_tokens_per_solved": completion / solved if solved else None,
        "per_task": {
            str(item["task"]): {
                "solved": item["solved"],
                "completion_tokens": item["metered"]["completion_tokens"],
            }
            for item in counted
        },
    }


def report(
    evidence: Path,
    job: Path,
    tasks: Path,
    only: list[str],
    harbor_exit: int | None,
    *,
    checkpoint: str,
    environment: str,
    usage: Path | None = None,
) -> dict:
    summary: dict = {"harbor_exit": harbor_exit, "environment": environment}
    try:
        expected = expected_tasks(tasks, only)
    except (OSError, ValueError) as error:
        return {**summary, "reward": 0.0, "demo_ok": False, "error": str(error)}
    vllm = vllm_evidence(evidence, checkpoint)
    served = completions(evidence)
    outcomes = trials(job, environment)
    metered = metering(usage)
    empty = {
        "requests": 0,
        "failed": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0,
    }
    for item in outcomes:
        key = slug(item["task"]) if item["task"] else None
        item["metered"] = metered.get(key, empty) if key else empty
    ran = sorted(str(item["task"]) for item in outcomes)
    infra = [
        f"{item['trial']}: {item['exception']}"
        for item in outcomes
        if item["infra_error"]
    ]
    infra += [
        f"{item['trial']}: ran in {item['environment']}, not {environment}"
        for item in outcomes
        if not item["through_sandbox"]
    ]
    # Every agent's requests through its own base path at the proxy.
    infra += [
        f"{item['trial']}: its agent's requests were not metered"
        for item in outcomes
        if item["requested"] and not item["metered"]["requests"]
    ]
    keys = {slug(task) for task in expected}
    stray = sum(total["requests"] for key, total in metered.items() if key not in keys)
    if stray:
        infra.append(f"metering: {stray} request(s) outside every trial's base path")
    if harbor_exit in HARBOR_TIMED_OUT:
        infra.append(f"harbor run stopped by its time limit (exit {harbor_exit})")
    if ran != sorted(expected):
        infra.append(f"trials ran {ran}, expected one each of {expected}")
    solved = sum(1 for item in outcomes if item["solved"] and item["task"] in expected)
    summary.update(
        expected_tasks=expected,
        vllm=vllm,
        completions=served,
        trials=outcomes,
        infra_errors=infra,
        # Completed, but stopped by the model's doing (an agent timeout, a
        # verifier that found nothing to score), a reward or not: apart
        # from a clean finish.
        model_outcomes=[
            f"{item['trial']}: {item['outcome']}"
            for item in outcomes
            if item["outcome"].startswith(("agent:", "verifier:"))
        ],
        solved=solved,
        reward=solved / len(expected) if expected else 0.0,
        accuracy_vs_tokens=accuracy_vs_tokens(outcomes, expected),
        demo_ok=bool(
            expected
            and vllm["on_judge_gpu"]
            and vllm["served_checkpoint"]
            and served["requests_served"]
            and not infra
            and all(item["completed"] for item in outcomes)
        ),
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("wait")
    command.add_argument("--base", required=True)
    command.add_argument("--pid", type=int, required=True)
    command.add_argument("--timeout", type=float, default=900.0)
    command.add_argument("--path", default="/health")
    command = commands.add_parser("fetch")
    command.add_argument("--base", required=True)
    command.add_argument("--path", required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser("tasks")
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--only", default="")
    command = commands.add_parser("report")
    command.add_argument("--evidence", type=Path, required=True)
    command.add_argument("--jobs", type=Path, required=True)
    command.add_argument("--job", required=True)
    command.add_argument("--tasks", type=Path, required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--environment", required=True)
    command.add_argument("--only", default="")
    command.add_argument("--harbor-exit", type=int)
    command.add_argument("--usage", type=Path)
    command.add_argument("--reward-json", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "wait":
        return wait(args.base, args.pid, args.timeout, path=args.path)
    if args.command == "fetch":
        return fetch(args.base, args.path, args.out)
    only = [task for task in args.only.split(",") if task]
    if args.command == "tasks":
        try:
            names = expected_tasks(args.tasks, only)
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return 1
        for name in names:
            print(name, slug(name))
        return 0
    summary = report(
        args.evidence,
        args.jobs / args.job,
        args.tasks,
        only,
        args.harbor_exit,
        checkpoint=args.checkpoint,
        environment=args.environment,
        usage=args.usage,
    )
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.reward_json.write_text(json.dumps({"reward": summary["reward"]}) + "\n")
    print(
        json.dumps(
            {
                "reward": summary["reward"],
                "demo_ok": summary["demo_ok"],
                "requests_served": (summary.get("completions") or {}).get(
                    "requests_served"
                ),
                "infra_errors": summary.get("infra_errors"),
                "completion_tokens": (summary.get("accuracy_vs_tokens") or {}).get(
                    "completion_tokens"
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
