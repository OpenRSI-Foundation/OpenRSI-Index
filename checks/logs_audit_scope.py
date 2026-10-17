"""Decide whether a push or pull request needs the Logs repository audit.

Besides the trusted checker, its policies and the workflow, the audit reads one
thing from this repository: its task roots (check_publication.task_roots). A
change confined to rsi-tasks/ that keeps that set, such as a README edit,
cannot change the result. Anything uncertain runs the audit.

Usage: logs_audit_scope.py --repo DIR --base REF --head REF [--github-output FILE]
Prints a notice and writes ``audit=true|false`` to FILE (or stdout).
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib.util
import subprocess
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_publication", Path(__file__).with_name("check_publication.py")
)
publication = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publication)

# The workflow's other trigger paths (tested to match): always audit.
ALWAYS = (
    "checks/check_publication.py",
    "checks/publication-*.toml",
    "checks/publish_logs.py",
    "checks/logs_audit_scope.py",
    ".github/workflows/hf-logs-publication-checks.yml",
)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def roots(repo: Path, ref: str) -> set[str]:
    listed = git(repo, "ls-tree", "-r", "--name-only", "-z", ref, "--", "rsi-tasks")
    return publication.task_roots(listed.split("\0"))


def decide(repo: Path, base: str, head: str) -> tuple[bool, str]:
    """Return (audit needed, reason)."""
    try:
        base = git(repo, "rev-parse", "--verify", "--quiet", base + "^{commit}").strip()
        head = git(repo, "rev-parse", "--verify", "--quiet", head + "^{commit}").strip()
        # What this change touches, as GitHub's path filters see it.
        start = git(repo, "merge-base", base, head).strip()
    except subprocess.CalledProcessError:
        return True, "no usable base revision"
    changed = [
        p
        for p in git(repo, "diff", "--name-only", "--no-renames", "-z", start, head).split("\0")
        if p
    ]
    for path in changed:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in ALWAYS):
            return True, f"{path} changed"
    # The base tip, not the merge base: a task main gained since a branch
    # forked counts as a change.
    old, new = roots(repo, base), roots(repo, head)
    if old != new:
        added = ", ".join(sorted(new - old)) or "none"
        removed = ", ".join(sorted(old - new)) or "none"
        return True, f"the task list changed (added: {added}; removed: {removed})"
    return False, "the task list is unchanged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    audit, reason = decide(args.repo, args.base, args.head)
    # Task paths are contributor input: annotate escapes workflow commands.
    if audit:
        publication.annotate("notice", f"Running the Logs repository audit: {reason}.")
    else:
        publication.annotate(
            "notice", f"Logs repository audit skipped: {reason}, so its result cannot change."
        )
    line = f"audit={'true' if audit else 'false'}\n"
    if args.github_output:
        with args.github_output.open("a") as output:
            output.write(line)
    else:
        print(line, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
