#!/usr/bin/env python3
"""Upgrade one pinned Agent CLI, or node, and run its Docker tests.

    uv run python scripts/bump_agent.py <agent|node|ripgrep> <version>
                                        [--no-test] [--no-check]

Rewrites the one constant in src/rsi_loop/harness/agent/versions.py, prints
the diff and, unless --no-test, runs the Docker test cases of the Agents that
use it (every Agent's for node); a case that cannot run, for want of Docker,
fails instead of skipping. Before editing it checks that the version is
published, unless --no-check. It exits non-zero when anything fails; a failed
test leaves the edit in place for a look, and ``git checkout`` on versions.py
reverts it.

Standard library only, so it runs before the project's environment is synced.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ROOT / "src" / "rsi_loop" / "harness" / "agent" / "versions.py"
# Every case in these is parametrized by its Agent's name, so ``-k [name]``
# selects that Agent's cases.
DOCKER_TESTS = ("tests/integration/test_driven_agents_docker.py",)
# Makes a Docker case that cannot run (no Docker, no cached base image) fail
# instead of skip: pytest exits 0 when every selected case skipped, which
# would pass a bump that tested nothing.
REQUIRE_ENV = "RSI_REQUIRE_DOCKER_TESTS"
# Exact versions only: the install commands and their tests expect X.Y.Z.
_VERSION = re.compile(r"\d+\.\d+\.\d+")


@dataclass(frozen=True)
class Target:
    constant: str
    # Where the release is published, to check it exists before editing: an
    # npm package, or a URL template with {v} for the version.
    npm_package: str | None = None
    url: str | None = None
    # The Agents whose Docker cases to run; empty means every Agent.
    agents: tuple[str, ...] = ()


TARGETS = {
    "node": Target(
        "NODE_VERSION", url="https://nodejs.org/dist/v{v}/node-v{v}-linux-x64.tar.xz"
    ),
    "codex": Target("CODEX", npm_package="@openai/codex", agents=("codex",)),
    "claude-code": Target(
        "CLAUDE_CODE", npm_package="@anthropic-ai/claude-code", agents=("claude-code",)
    ),
    "qwen-code": Target(
        "QWEN_CODE", npm_package="@qwen-code/qwen-code", agents=("qwen-code",)
    ),
    "kimi-code": Target(
        "KIMI_CODE", npm_package="@moonshot-ai/kimi-code", agents=("kimi-code",)
    ),
    "opencode": Target(
        "OPENCODE", npm_package="opencode-linux-x64", agents=("opencode",)
    ),
    "zcode": Target(
        "ZCODE",
        url="https://cdn-zcode.z.ai/zcode/electron/releases/{v}/linux-x64/"
        "ZCode-{v}-linux-x64.deb",
        agents=("zcode",),
    ),
    "pi": Target("PI", npm_package="@earendil-works/pi-coding-agent", agents=("pi",)),
    # Not an Agent: the ripgrep OpenCode's image carries.
    "ripgrep": Target(
        "RIPGREP",
        url="https://github.com/BurntSushi/ripgrep/releases/download/{v}/"
        "ripgrep-{v}-x86_64-unknown-linux-musl.tar.gz",
        agents=("opencode",),
    ),
}


class BumpError(Exception):
    pass


def rewrite(text: str, constant: str, version: str) -> tuple[str, str]:
    """Return *text* with *constant* set to *version*, and the old version.

    Refuses unless the assignment appears exactly once, so a renamed or
    duplicated constant is never half-updated.
    """
    if not _VERSION.fullmatch(version):
        raise BumpError(f"version must be X.Y.Z, got {version!r}")
    pattern = re.compile(rf'^({re.escape(constant)} = )"([^"\n]*)"$', re.MULTILINE)
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise BumpError(
            f'expected one `{constant} = "..."` line in {VERSIONS.name}, '
            f"found {len(matches)}"
        )
    old = matches[0][1]
    return pattern.sub(lambda m: f'{m.group(1)}"{version}"', text), old


def pytest_args(target: Target) -> list[str]:
    """Return the pytest arguments that run *target*'s Docker cases."""
    args = [*DOCKER_TESTS]
    if target.agents:
        args += ["-k", " or ".join(f"[{agent}]" for agent in target.agents)]
    return args


def check_published(target: Target, version: str) -> None:
    if target.npm_package:
        npm = shutil.which("npm")
        if npm is None:
            raise BumpError("npm is not on PATH; pass --no-check to skip the check")
        spec = f"{target.npm_package}@{version}"
        result = subprocess.run(
            [npm, "view", spec, "version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or result.stdout.strip() != version:
            raise BumpError(f"{spec} is not published on the npm registry")
    elif target.url:
        url = target.url.format(v=version)
        request = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=30):
                pass
        except (urllib.error.URLError, TimeoutError) as error:
            raise BumpError(f"{url} is not available: {error}") from error


def run_tests(target: Target) -> int:
    uv = shutil.which("uv")
    runner = [uv, "run", "--group", "test"] if uv else [sys.executable, "-m"]
    command = [*runner, "pytest", "-v", *pytest_args(target)]
    print("+", " ".join(command), flush=True)
    env = {**os.environ, REQUIRE_ENV: "1"}
    returncode = subprocess.run(command, cwd=ROOT, env=env, check=False).returncode
    if returncode == 5:
        print("no Docker test case selected for this bump", file=sys.stderr)
    return returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upgrade one pinned Agent CLI (or node) and run its Docker tests."
    )
    parser.add_argument("name", choices=sorted(TARGETS))
    parser.add_argument("version", help="the exact new version, X.Y.Z")
    parser.add_argument(
        "--no-test", action="store_true", help="only edit; skip the Docker tests"
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip checking that the version is published",
    )
    args = parser.parse_args(argv)
    target = TARGETS[args.name]

    try:
        text = VERSIONS.read_text()
        new_text, old = rewrite(text, target.constant, args.version)
        if old == args.version:
            print(f"{args.name} is already {args.version}")
            return 0
        if not args.no_check:
            check_published(target, args.version)
    except BumpError as error:
        print(f"bump-agent: {error}", file=sys.stderr)
        return 2

    VERSIONS.write_text(new_text)
    relative = str(VERSIONS.relative_to(ROOT))
    sys.stdout.writelines(
        difflib.unified_diff(
            text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            f"a/{relative}",
            f"b/{relative}",
        )
    )
    if args.no_test:
        return 0
    returncode = run_tests(target)
    if returncode != 0:
        print(
            f"bump-agent: {args.name} {args.version} failed its Docker tests; "
            f"`git checkout {relative}` reverts the edit",
            file=sys.stderr,
        )
    return returncode


if __name__ == "__main__":
    sys.exit(main())
