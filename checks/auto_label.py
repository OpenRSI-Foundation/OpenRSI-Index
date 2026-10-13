"""Path-based PR area labels.

The label table is the single source of truth. A PR carries a managed label
exactly when one of its changed files (or, for a rename, the file's previous
path) matches that label's paths. Labels outside the table are never touched.

The workflow runs this file from the base revision and reads PR metadata only
through the GitHub API; PR contents are never checked out or executed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AreaLabel:
    name: str
    color: str
    description: str
    # An entry ending in "/" matches everything below that directory; any
    # other entry matches exactly one repository-relative file.
    paths: tuple[str, ...]
    # Also match top-level dotfiles such as .gitignore and .gitattributes.
    root_dotfiles: bool = False


LABELS = (
    AreaLabel("tasks", "0e8a16", "Changes Harbor tasks under rsi-tasks/", ("rsi-tasks/",)),
    AreaLabel("logs", "c5def5", "Changes run logs under rsi-logs/", ("rsi-logs/",)),
    AreaLabel("harness", "5319e7", "Changes RSI-Harness", ("RSI-Harness/",)),
    AreaLabel(
        "CI",
        "fbca04",
        "Changes workflows, repository checks or top-level config files",
        (".github/workflows/", "checks/", ".github/WORKFLOW_API.md"),
        root_dotfiles=True,
    ),
    AreaLabel(
        "docs",
        "0075ca",
        "Changes repository documentation",
        (
            "README.md",
            "CONTRIBUTING.md",
            "LICENSE",
            ".github/pull_request_template.md",
            "assets/",
        ),
    ),
    AreaLabel("skills", "d4c5f9", "Changes agent skills", (".agents/", ".claude/", ".codex/")),
)
MANAGED = frozenset(label.name for label in LABELS)
_PR_NUMBER = re.compile(r"[1-9][0-9]{0,9}\Z")


def _matches(path: str, pattern: str) -> bool:
    return path.startswith(pattern) if pattern.endswith("/") else path == pattern


def _is_root_dotfile(path: str) -> bool:
    return path.startswith(".") and "/" not in path


def labels_for(files: Iterable[Mapping[str, object]]) -> set[str]:
    """Managed labels for a PR's changed files (REST `pulls/{n}/files` items)."""
    paths = set()
    for item in files:
        for key in ("filename", "previous_filename"):
            value = item.get(key)
            if isinstance(value, str) and value:
                paths.add(value)
    return {
        label.name
        for label in LABELS
        if any(_matches(path, pattern) for path in paths for pattern in label.paths)
        or (label.root_dotfiles and any(_is_root_dotfile(path) for path in paths))
    }


def plan(
    desired: set[str], current: Iterable[str], *, complete: bool = True
) -> tuple[list[str], list[str]]:
    """Managed labels to add and remove. Removal waits for a complete file list."""
    current = set(current)
    add = sorted(desired - current)
    remove = sorted((current & MANAGED) - desired) if complete else []
    return add, remove


class GitHubError(RuntimeError):
    pass


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess]


def _run_gh(args: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["gh", *args], capture_output=True, text=True, timeout=120, check=False
    )


class GitHub:
    def __init__(self, repo: str, run: Runner = _run_gh) -> None:
        self.repo, self.run = repo, run

    def _api(self, *args: str, ok_status: tuple[int, ...] = ()) -> str:
        result = self.run(["api", *args])
        if result.returncode == 0:
            return result.stdout
        match = re.search(r"\(HTTP (\d{3})\)", result.stderr)
        if match and int(match.group(1)) in ok_status:
            return result.stdout
        endpoint = next(arg for arg in args if arg.startswith("repos/"))
        raise GitHubError(
            f"{endpoint} failed: {result.stderr.strip() or result.stdout.strip()}"
        )

    def _json_lines(self, *args: str) -> list:
        output = self._api("--paginate", *args)
        return [json.loads(line) for line in output.splitlines() if line.strip()]

    def pull(self, number: int) -> dict:
        return json.loads(self._api(f"repos/{self.repo}/pulls/{number}"))

    def pull_files(self, number: int) -> list[dict]:
        return self._json_lines(
            f"repos/{self.repo}/pulls/{number}/files?per_page=100",
            "--jq",
            ".[] | {filename, previous_filename}",
        )

    def open_pulls(self) -> list[int]:
        return self._json_lines(
            f"repos/{self.repo}/pulls?state=open&per_page=100", "--jq", ".[].number"
        )

    def issue_labels(self, number: int) -> list[str]:
        return self._json_lines(
            f"repos/{self.repo}/issues/{number}/labels?per_page=100",
            "--jq",
            ".[].name | tojson",
        )

    def ensure_label(self, label: AreaLabel) -> None:
        result = self.run(
            [
                "api",
                f"repos/{self.repo}/labels",
                "-f",
                f"name={label.name}",
                "-f",
                f"color={label.color}",
                "-f",
                f"description={label.description}",
            ]
        )
        if result.returncode == 0:
            print(f"Created label {label.name!r}")
            return
        # 422 already_exists is the normal case once the label is in place.
        if "(HTTP 422)" in result.stderr and "already_exists" in result.stdout:
            return
        raise GitHubError(
            f"creating label {label.name!r} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )

    def add_labels(self, number: int, names: Sequence[str]) -> None:
        fields = [arg for name in names for arg in ("-f", f"labels[]={name}")]
        self._api(f"repos/{self.repo}/issues/{number}/labels", *fields)

    def remove_label(self, number: int, name: str) -> None:
        # 404: the label is already gone (for example, removed by hand meanwhile).
        quoted = urllib.parse.quote(name, safe="")
        self._api(
            "-X",
            "DELETE",
            f"repos/{self.repo}/issues/{number}/labels/{quoted}",
            ok_status=(404,),
        )


def sync_pull(github: GitHub, number: int) -> tuple[list[str], list[str]]:
    pull = github.pull(number)
    files = github.pull_files(number)
    desired = labels_for(files)
    # The files endpoint lists at most 3000 files; never remove on a partial view.
    complete = len(files) >= int(pull.get("changed_files") or 0)
    add, remove = plan(desired, github.issue_labels(number), complete=complete)
    if not complete:
        print(
            f"::warning::PR #{number}: GitHub listed {len(files)} of "
            f"{pull.get('changed_files')} changed files; labels are only added."
        )
    by_name = {label.name: label for label in LABELS}
    for name in add:
        github.ensure_label(by_name[name])
    if add:
        github.add_labels(number, add)
    for name in remove:
        github.remove_label(number, name)
    print(
        f"PR #{number}: areas={sorted(desired) or ['none']} "
        f"add={add or ['none']} remove={remove or ['none']}"
    )
    return add, remove


def targets(github: GitHub, value: str) -> list[int]:
    value = value.strip()
    if value == "open":
        return github.open_pulls()
    if _PR_NUMBER.fullmatch(value):
        return [int(value)]
    raise ValueError(f"expected a PR number or 'open', got {value!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="a PR number, or 'open' for every open PR")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"))
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("--repo or GITHUB_REPOSITORY is required")
    github = GitHub(args.repo)
    failed = []
    for number in targets(github, args.target):
        try:
            sync_pull(github, number)
        except GitHubError as error:
            print(f"::error::PR #{number}: {error}")
            failed.append(number)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
