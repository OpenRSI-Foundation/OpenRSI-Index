"""The pinned-version table and scripts/bump_agent.py, which edits it."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

import pytest

from rsi_loop.harness.agent import create_agent, list_agent_classes, versions
from rsi_loop.harness.agent.base import NODE_INSTALL_CMD
from rsi_loop.harness.config import RSILoopConfig

ROOT = Path(__file__).resolve().parents[2]
AGENT_NAMES = sorted(cls.name for cls in list_agent_classes())


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "bump_agent", ROOT / "scripts" / "bump_agent.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Dataclasses look their module up while the class is created.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bump = _load_script()


def _install_text(name: str) -> str:
    return "\n".join(create_agent(name, RSILoopConfig()).install_cmds)


# The install commands of the two native-hook Agents, as written before the
# versions moved into one table. Their Work image tags hash the Dockerfile,
# so any other byte would rebuild every cached Codex and Claude Code image.
_NODE_INSTALL = (
    "sudo -E bash -c 'NODE_MIRROR=${{RSI_NODEJS_MIRROR_URL:-https://nodejs.org/dist}}"
    " && curl -fsSL $NODE_MIRROR/v{node}/node-v{node}-linux-x64.tar.xz"
    " | tar -xJ -C /usr/local --strip-components=1'"
)
_CODEX_CONFIG = """if [ -n "$OPENAI_BASE_URL" ]; then
    mkdir -p ~/.codex
    cat > ~/.codex/config.toml << EOF
model_provider = "rsi-loop-proxy"
model_verbosity = "medium"
model = "${CODEX_MODEL:-gpt-5.5}"

[model_providers.rsi-loop-proxy]
name = "rsi-loop-proxy"
base_url = "${OPENAI_BASE_URL}"
env_key = "OPENAI_API_KEY"
EOF
fi"""


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "codex",
            [
                _NODE_INSTALL,
                "sudo -E npm install -g @openai/codex@{codex}",
                _CODEX_CONFIG,
            ],
        ),
        (
            "claude-code",
            [
                _NODE_INSTALL,
                "sudo -E npm install -g @anthropic-ai/claude-code@{claude}",
            ],
        ),
    ],
)
def test_native_hook_agents_keep_their_install_commands_byte_for_byte(
    name: str, expected: list[str]
) -> None:
    pins = {
        "node": versions.NODE_VERSION,
        "codex": versions.CODEX,
        "claude": versions.CLAUDE_CODE,
    }
    rendered = [
        command if command is _CODEX_CONFIG else command.format(**pins)
        for command in expected
    ]

    assert create_agent(name, RSILoopConfig()).install_cmds == rendered


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_every_agent_installs_the_shared_pinned_node_first(name: str) -> None:
    agent = create_agent(name, RSILoopConfig())

    assert agent.install_cmds[0] == NODE_INSTALL_CMD
    assert f"/v{versions.NODE_VERSION}/" in NODE_INSTALL_CMD


def test_bump_script_covers_every_agent_and_constant() -> None:
    assert set(AGENT_NAMES) < set(bump.TARGETS)
    for name, target in bump.TARGETS.items():
        assert isinstance(getattr(versions, target.constant), str), name
        assert set(target.agents) <= set(AGENT_NAMES), name
        if name in AGENT_NAMES:
            assert target.agents == (name,)


def test_every_agent_has_the_docker_cases_the_bump_script_selects() -> None:
    from tests.integration import test_driven_agents_docker as docker_tests

    assert bump.DOCKER_TESTS[0] == str(Path(docker_tests.__file__).relative_to(ROOT))
    assert all((ROOT / path).is_file() for path in bump.DOCKER_TESTS)
    # ``-k [name]`` selects a case by its id, the Agent's name.
    assert sorted(str(case) for case in docker_tests.CASES) == AGENT_NAMES


@pytest.mark.parametrize("name", sorted(bump.TARGETS))
def test_bump_script_checks_the_release_the_image_installs(name: str) -> None:
    target = bump.TARGETS[name]
    version = getattr(versions, target.constant)
    installs = [_install_text(agent) for agent in target.agents or AGENT_NAMES]

    for text in installs:
        if target.npm_package:
            assert f"{target.npm_package}@{version}" in text
        else:
            url = urlparse(target.url.format(v=version))
            assert url.netloc in text
            assert url.path.rsplit("/", 1)[-1] in text


_TABLE = '''"""Pins."""

NODE_VERSION = "22.23.3"

CODEX = "0.159.2"
CLAUDE_CODE = "2.1.285"
# CODEX = "0.1.0" in a comment is not an assignment.
'''


def test_rewrite_changes_only_the_named_constant() -> None:
    text, old = bump.rewrite(_TABLE, "CODEX", "0.160.0")

    assert old == "0.159.2"
    assert text == _TABLE.replace('CODEX = "0.159.2"', 'CODEX = "0.160.0"')
    assert 'CLAUDE_CODE = "2.1.285"' in text


def test_rewrite_sets_node() -> None:
    text, old = bump.rewrite(_TABLE, "NODE_VERSION", "22.24.0")

    assert old == "22.23.3"
    assert 'NODE_VERSION = "22.24.0"' in text


@pytest.mark.parametrize(
    ("table", "found"),
    [
        (_TABLE.replace('CODEX = "0.159.2"\n', ""), 0),
        (_TABLE + 'CODEX = "0.159.2"\n', 2),
    ],
)
def test_rewrite_refuses_unless_the_constant_appears_once(
    table: str, found: int
) -> None:
    with pytest.raises(bump.BumpError, match=f"found {found}"):
        bump.rewrite(table, "CODEX", "0.160.0")


@pytest.mark.parametrize(
    "version", ["v0.160.0", "0.160", "0.160.0-beta.1", "", "1.2.3 "]
)
def test_rewrite_refuses_anything_but_an_exact_version(version: str) -> None:
    with pytest.raises(bump.BumpError, match="X.Y.Z"):
        bump.rewrite(_TABLE, "CODEX", version)


def test_pytest_args_select_the_agents_cases_or_all_for_node() -> None:
    assert bump.pytest_args(bump.TARGETS["claude-code"]) == [
        *bump.DOCKER_TESTS,
        "-k",
        "[claude-code] or [claude-code-",
    ]
    assert bump.pytest_args(bump.TARGETS["ripgrep"])[-1] == "[opencode] or [opencode-"
    assert bump.pytest_args(bump.TARGETS["node"]) == list(bump.DOCKER_TESTS)


def test_run_tests_makes_a_case_that_cannot_run_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration import test_driven_agents_docker as docker_tests

    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv(bump.REQUIRE_ENV, raising=False)
    monkeypatch.setattr(bump.subprocess, "run", run)

    assert bump.run_tests(bump.TARGETS["pi"]) == 0
    [(command, kwargs)] = calls
    args = bump.pytest_args(bump.TARGETS["pi"])
    assert command[-len(args) :] == args
    assert kwargs["env"][bump.REQUIRE_ENV] == "1"
    assert docker_tests.REQUIRE_ENV == bump.REQUIRE_ENV


def test_docker_cases_fail_without_docker_when_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.integration import test_driven_agents_docker as docker_tests

    def no_docker():
        raise ConnectionError("no Docker daemon")

    monkeypatch.setattr(docker_tests.docker, "from_env", no_docker)
    monkeypatch.delenv(docker_tests.REQUIRE_ENV, raising=False)
    # Skipped on a machine without Docker, as any other Docker test.
    with pytest.raises(pytest.skip.Exception):
        docker_tests._docker()

    # Failed under the bump script: a bump never passes on skipped cases.
    monkeypatch.setenv(docker_tests.REQUIRE_ENV, "1")
    with pytest.raises(pytest.fail.Exception, match="no Docker daemon"):
        docker_tests._docker()


@pytest.fixture
def table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "src" / "versions.py"
    path.parent.mkdir()
    path.write_text(_TABLE)
    monkeypatch.setattr(bump, "ROOT", tmp_path)
    monkeypatch.setattr(bump, "VERSIONS", path)
    return path


def test_main_edits_prints_the_diff_and_skips_tests_on_request(
    table: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(bump, "run_tests", pytest.fail)

    assert bump.main(["codex", "0.160.0", "--no-test", "--no-check"]) == 0

    assert 'CODEX = "0.160.0"' in table.read_text()
    out = capsys.readouterr().out
    assert '-CODEX = "0.159.2"\n+CODEX = "0.160.0"\n' in out
    assert "+++ b/src/versions.py" in out


def test_main_returns_the_test_failure(
    table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = []
    monkeypatch.setattr(bump, "run_tests", lambda target: ran.append(target) or 1)

    assert bump.main(["node", "22.24.0", "--no-check"]) == 1
    assert ran == [bump.TARGETS["node"]]


def test_main_checks_the_release_before_editing(
    table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unpublished(target, version):
        raise bump.BumpError(f"{version} is not published")

    monkeypatch.setattr(bump, "check_published", unpublished)

    assert bump.main(["codex", "9.9.9", "--no-test"]) == 2
    assert table.read_text() == _TABLE


def test_main_leaves_an_unchanged_version_alone(
    table: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bump, "run_tests", pytest.fail)
    monkeypatch.setattr(bump, "check_published", pytest.fail)

    assert bump.main(["claude-code", "2.1.285"]) == 0
    assert table.read_text() == _TABLE


def test_main_refuses_an_unknown_name(table: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        bump.main(["not-an-agent", "1.2.3", "--no-test"])

    assert raised.value.code == 2
