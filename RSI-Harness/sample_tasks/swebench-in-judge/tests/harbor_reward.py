"""Score this sample's Harbor jobs (the sample's own logic, not the Harness's).

Each suite ran twice, as the jobs ``<suite>-oracle`` and ``<suite>-nop``.
The reward is 1 only if every expected task has exactly one trial in both
jobs, every oracle trial scored 1.0, every nop trial scored 0.0 and no trial
raised; anything else (a missing job, an error, an extra trial) scores 0.
Standard library only: it runs in Work and in the Judge.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

WANT = {"oracle": 1.0, "nop": 0.0}


def task_name(task_dir: Path) -> str:
    """Harbor's name of a task (its trials' ``task_name``): ``[task] name``
    in task.toml, else the directory's name."""
    try:
        config = tomllib.loads((task_dir / "task.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return task_dir.name
    name = (config.get("task") or {}).get("name")
    return name if isinstance(name, str) and name else task_dir.name


def suite_tasks(path: Path, only: list[str] | None = None) -> list[str]:
    """Harbor task names of a suite: one task, or a dataset of tasks."""
    if (path / "task.toml").is_file():
        names = [task_name(path)]
    else:
        names = sorted(
            task_name(child)
            for child in path.iterdir()
            if child.is_dir() and (child / "task.toml").is_file()
        )
    if only:
        missing = sorted(set(only) - set(names))
        if missing:
            raise ValueError(f"{path}: no task named {', '.join(missing)}")
        names = [name for name in names if name in only]
    return names


def job_trials(job: Path) -> list[dict]:
    """The trial results of one Harbor job directory (``<trial>/result.json``;
    the job's own ``result.json`` sits beside them)."""
    trials = []
    for path in sorted(job.glob("*/result.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            trials.append({"trial_name": path.parent.name, "unreadable": True})
            continue
        if isinstance(data, dict) and "trial_name" in data:
            trials.append(data)
    return trials


def trial_verdict(trial: dict, want: float) -> tuple[bool, str]:
    if trial.get("unreadable"):
        return False, "unreadable result.json"
    error = trial.get("exception_info")
    if error:
        return False, f"{error.get('exception_type')}: {error.get('exception_message')}"
    rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    try:
        scored = float(reward)
    except (TypeError, ValueError):
        return False, f"reward {reward!r}, want {want}"
    if scored != want:
        return False, f"reward {reward!r}, want {want}"
    return True, "ok"


def score(jobs: Path, suites: dict[str, list[str]]) -> dict:
    """``suites`` maps a suite to its expected task names."""
    summary: dict = {"suites": {}}
    passed = True
    for suite, tasks in suites.items():
        report: dict = {"tasks": tasks}
        for agent, want in WANT.items():
            trials = job_trials(jobs / f"{suite}-{agent}")
            verdicts = {}
            for trial in trials:
                ok, detail = trial_verdict(trial, want)
                verdicts[str(trial.get("trial_name"))] = {
                    "task": trial.get("task_name"),
                    "ok": ok,
                    "detail": detail,
                }
            # Exactly the expected tasks, each once: not merely as many.
            names = sorted(str(item["task"]) for item in verdicts.values())
            complete = names == sorted(tasks) and len(set(names)) == len(names)
            ok = complete and all(item["ok"] for item in verdicts.values())
            report[agent] = {
                "ok": ok,
                "trials": len(trials),
                "expected": len(tasks),
                "verdicts": verdicts,
            }
            passed = passed and ok
        summary["suites"][suite] = report
    summary["reward"] = 1.0 if passed and suites else 0.0
    return summary


def _suite_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--suite", type=_suite_argument, action="append", default=[])
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="SUITE=TASK,...",
        help="restrict a suite to these task names",
    )
    parser.add_argument("--reward-json", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)
    only = {}
    for item in args.only:
        name, _, tasks = item.partition("=")
        only[name] = [task for task in tasks.split(",") if task]
    try:
        suites = {name: suite_tasks(path, only.get(name)) for name, path in args.suite}
        summary = score(args.jobs, suites)
    except (OSError, ValueError) as error:
        summary = {"reward": 0.0, "error": str(error)}
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.reward_json.write_text(json.dumps({"reward": summary["reward"]}) + "\n")
    print(json.dumps({"reward": summary["reward"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
