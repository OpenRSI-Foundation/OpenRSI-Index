"""Choose which Harness Agents' Docker tests a pull request needs.

A change to one Agent's own files, or to its line in the version table,
tests that Agent only. Anything else the Harness Agent tests watch (the
driver, the shared Agent code, rsi-agent, the tests themselves, node's
version) can break every Agent, so it tests all of them. Codex's and Claude
Code's Judge one-shot calls are covered by the model-access job, which runs
when either is selected.

Usage: harness_agent_selection.py BASE HEAD | --all
Writes ``agents=<JSON list>`` and ``model_access=true|false`` to
$GITHUB_OUTPUT (or stdout).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/harness-agent-tests.yml"
AGENT_DIR = "RSI-Harness/src/rsi_loop/harness/agent/"
VERSIONS = AGENT_DIR + "versions.py"

# Each Agent's own files, relative to AGENT_DIR (a trailing / is a directory).
AGENT_FILES = {
    "codex": ("codex.py",),
    "claude-code": ("claude_code.py",),
    "qwen-code": ("qwen_code.py",),
    "kimi-code": ("kimi_code.py",),
    "opencode": ("opencode.py",),
    "zcode": ("zcode.py", "zcode/"),
    "pi": ("pi.py", "pi/"),
}
AGENTS = tuple(AGENT_FILES)
# Version-table constants that belong to one Agent; any other change to the
# table (node's version included) tests every Agent.
VERSION_OWNERS = {
    "CODEX": "codex",
    "CLAUDE_CODE": "claude-code",
    "QWEN_CODE": "qwen-code",
    "KIMI_CODE": "kimi-code",
    "OPENCODE": "opencode",
    "RIPGREP": "opencode",
    "ZCODE": "zcode",
    "PI": "pi",
}
MODEL_ACCESS_AGENTS = {"codex", "claude-code"}
_VERSION_LINE = re.compile(r'^[+-]([A-Z_]+) = "[^"]*"$')


def watched_paths() -> list[str]:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    return list(workflow["on"]["pull_request"]["paths"])


def _agent_of(path: str) -> str | None:
    relative = path[len(AGENT_DIR) :] if path.startswith(AGENT_DIR) else None
    if relative is None:
        return None
    for agent, owned in AGENT_FILES.items():
        for entry in owned:
            if relative == entry or (entry.endswith("/") and relative.startswith(entry)):
                return agent
    return None


def _version_agents(versions_diff: str) -> set[str] | None:
    """Agents whose version lines changed, or None when every Agent is due."""
    agents: set[str] = set()
    for line in versions_diff.splitlines():
        if line.startswith(("+++", "---")) or not line.startswith(("+", "-")):
            continue
        match = _VERSION_LINE.match(line)
        if match is None or match.group(1) not in VERSION_OWNERS:
            return None
        agents.add(VERSION_OWNERS[match.group(1)])
    return agents


def select(changed: list[str], versions_diff: str, watched: list[str]) -> list[str]:
    """Return the Agents to test, in AGENTS order."""
    selected: set[str] = set()
    for path in changed:
        if not any(fnmatch.fnmatchcase(path, pattern) for pattern in watched):
            continue
        if path == VERSIONS:
            agents = _version_agents(versions_diff)
            if agents is None:
                return list(AGENTS)
            selected |= agents
            continue
        agent = _agent_of(path)
        if agent is None:
            return list(AGENTS)
        selected.add(agent)
    return [agent for agent in AGENTS if agent in selected]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str]) -> int:
    if argv[1:] == ["--all"]:
        agents = list(AGENTS)
    elif len(argv) == 3:
        base, head = argv[1], argv[2]
        changed = _git("diff", "--name-only", f"{base}...{head}").splitlines()
        versions_diff = _git("diff", "-U0", f"{base}...{head}", "--", VERSIONS)
        agents = select(changed, versions_diff, watched_paths())
    else:
        print(__doc__, file=sys.stderr)
        return 2
    lines = [
        f"agents={json.dumps(agents)}",
        f"model_access={'true' if MODEL_ACCESS_AGENTS & set(agents) else 'false'}",
    ]
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as handle:
            handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
