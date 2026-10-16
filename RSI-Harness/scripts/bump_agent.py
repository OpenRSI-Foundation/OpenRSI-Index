#!/usr/bin/env python3
"""Upgrade one pinned Agent CLI, or node, and run its Docker tests.

    uv run python scripts/bump_agent.py <agent|node|ripgrep> <version>
                                        [--no-test] [--no-check]

Rewrites the one constant in src/rsi_loop/harness/agent/versions.py (for
codex also the SHA-256 of its native Linux binaries, taken from the npm
registry's packages after their registry integrity check), prints the diff
and, unless --no-test, runs the Docker test cases of the Agents that
use it (every Agent's for node); a case that cannot run, for want of Docker,
fails instead of skipping. Before editing it checks that the version is
published, unless --no-check. It exits non-zero when anything fails; a failed
test leaves the edit in place for a look, and ``git checkout`` on versions.py
reverts it.

Standard library only, so it runs before the project's environment is synced.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSIONS = ROOT / "src" / "rsi_loop" / "harness" / "agent" / "versions.py"
# An Agent's cases in these: a full run (id ``[name]``), the Judge's one-shot
# call through rsi-agent (``[name]``, or ``[name-...]`` for Codex and Claude
# Code), so ``-k "[name] or [name-"`` selects exactly that Agent's cases.
DOCKER_TESTS = (
    "tests/integration/test_driven_agents_docker.py",
    "tests/integration/test_agent_one_shot_docker.py",
    "tests/integration/test_judge_model_access_docker.py",
)
# Makes a Docker case that cannot run (no Docker, no cached base image) fail
# instead of skip: pytest exits 0 when every selected case skipped, which
# would pass a bump that tested nothing.
REQUIRE_ENV = "RSI_REQUIRE_DOCKER_TESTS"
# Exact versions only: the install commands and their tests expect X.Y.Z.
_VERSION = re.compile(r"\d+\.\d+\.\d+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Target:
    constant: str
    # Where the release is published, to check it exists before editing: an
    # npm package, or a URL template with {v} for the version. Either may
    # hold {arch}, one release per name in ``arches``; the install picks the
    # one for the image's architecture, so each must be published.
    npm_package: str | None = None
    url: str | None = None
    # The Agents whose Docker cases to run; empty means every Agent.
    agents: tuple[str, ...] = ()
    # The x86_64 and aarch64 release names, for an {arch} template.
    arches: tuple[str, ...] = ()
    # Pinned SHA-256 constants of a native binary: (constant, npm package
    # version suffix, path in its tarball with {v}).
    digests: tuple[tuple[str, str, str], ...] = ()

    def npm_specs(self, version: str) -> list[str]:
        """Return the npm package specs to check, one per architecture."""
        assert self.npm_package
        return [
            f"{self.npm_package.format(arch=arch)}@{version}"
            for arch in self.arches or ("",)
        ]

    def urls(self, version: str) -> list[str]:
        """Return the release URLs to check, one per architecture."""
        assert self.url
        return [self.url.format(v=version, arch=arch) for arch in self.arches or ("",)]


TARGETS = {
    "node": Target(
        "NODE_VERSION",
        url="https://nodejs.org/dist/v{v}/node-v{v}-linux-{arch}.tar.xz",
        arches=("x64", "arm64"),
    ),
    "codex": Target(
        "CODEX",
        npm_package="@openai/codex",
        agents=("codex",),
        digests=(
            (
                "CODEX_SHA256_X86_64",
                "linux-x64",
                "package/vendor/x86_64-unknown-linux-musl/bin/codex",
            ),
            (
                "CODEX_SHA256_AARCH64",
                "linux-arm64",
                "package/vendor/aarch64-unknown-linux-musl/bin/codex",
            ),
        ),
    ),
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
        "OPENCODE",
        npm_package="opencode-linux-{arch}",
        agents=("opencode",),
        arches=("x64", "arm64"),
    ),
    "zcode": Target(
        "ZCODE",
        url="https://cdn-zcode.z.ai/zcode/electron/releases/{v}/linux-{arch}/"
        "ZCode-{v}-linux-{arch}.deb",
        agents=("zcode",),
        arches=("x64", "arm64"),
    ),
    "pi": Target("PI", npm_package="@earendil-works/pi-coding-agent", agents=("pi",)),
    # Not an Agent: the ripgrep OpenCode's image carries.
    "ripgrep": Target(
        "RIPGREP",
        url="https://github.com/BurntSushi/ripgrep/releases/download/{v}/"
        "ripgrep-{v}-{arch}.tar.gz",
        agents=("opencode",),
        arches=("x86_64-unknown-linux-musl", "aarch64-unknown-linux-gnu"),
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


def rewrite_digest(text: str, constant: str, digest: str) -> str:
    """Return *text* with the one ``constant = "<sha256>"`` set to *digest*."""
    if not _SHA256.fullmatch(digest):
        raise BumpError(f"not a SHA-256: {digest!r}")
    pattern = re.compile(rf'^({re.escape(constant)} = )"[0-9a-f]*"$', re.MULTILINE)
    if len(pattern.findall(text)) != 1:
        raise BumpError(f'expected one `{constant} = "..."` line in {VERSIONS.name}')
    return pattern.sub(lambda m: f'{m.group(1)}"{digest}"', text)


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=300) as response:
        return response.read()


def release_digests(target: Target, version: str, fetch=_fetch) -> dict[str, str]:
    """The SHA-256 of each pinned native binary of *version*, by constant.

    Each comes from the npm registry's package for that platform, after its
    tarball matched the registry's sha512 integrity.
    """
    assert target.npm_package
    found = {}
    for constant, suffix, member in target.digests:
        name = target.npm_package.replace("/", "%2f")
        try:
            metadata = json.loads(
                fetch(f"https://registry.npmjs.org/{name}/{version}-{suffix}")
            )
            integrity = metadata["dist"]["integrity"]
            tarball = fetch(metadata["dist"]["tarball"])
        except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
            raise BumpError(
                f"cannot fetch {target.npm_package}@{version}-{suffix}: {error}"
            ) from error
        algorithm, _, expected = str(integrity).partition("-")
        actual = base64.b64encode(hashlib.sha512(tarball).digest()).decode()
        if algorithm != "sha512" or actual != expected:
            raise BumpError(f"{target.npm_package}@{version}-{suffix} failed integrity")
        try:
            with tarfile.open(fileobj=io.BytesIO(tarball)) as bundle:
                stream = bundle.extractfile(member)
                if stream is None:
                    raise KeyError(member)
                found[constant] = hashlib.sha256(stream.read()).hexdigest()
        except (tarfile.TarError, KeyError) as error:
            raise BumpError(
                f"{target.npm_package}@{version}-{suffix} has no {member}"
            ) from error
    return found


def pytest_args(target: Target) -> list[str]:
    """Return the pytest arguments that run *target*'s Docker cases."""
    args = [*DOCKER_TESTS]
    if target.agents:
        args += [
            "-k",
            " or ".join(f"[{agent}] or [{agent}-" for agent in target.agents),
        ]
    return args


def check_published(target: Target, version: str) -> None:
    if target.npm_package:
        npm = shutil.which("npm")
        if npm is None:
            raise BumpError("npm is not on PATH; pass --no-check to skip the check")
        for spec in target.npm_specs(version):
            result = subprocess.run(
                [npm, "view", spec, "version"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or result.stdout.strip() != version:
                raise BumpError(f"{spec} is not published on the npm registry")
    elif target.url:
        for url in target.urls(version):
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
        if target.digests:
            # Always: the pins must name the new release's bytes.
            for constant, digest in release_digests(target, args.version).items():
                new_text = rewrite_digest(new_text, constant, digest)
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
