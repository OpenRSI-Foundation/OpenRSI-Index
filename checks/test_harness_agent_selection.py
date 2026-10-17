import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_agent_selection as selection

AGENT_DIR = selection.AGENT_DIR
WATCHED = selection.watched_paths()
ALL = list(selection.AGENTS)
MODEL_AGENTS = [agent for agent in ALL if agent != selection.ORACLE]


def _bump(*constants):
    lines = ["--- a/versions.py", "+++ b/versions.py", "@@ -1 +1 @@"]
    for constant in constants:
        lines += [f'-{constant} = "1.0.0"', f'+{constant} = "1.0.1"']
    return "\n".join(lines) + "\n"


def _registered_agents():
    factory = (selection.ROOT / AGENT_DIR / "factory.py").read_text()
    registry = factory[factory.index("_REGISTRY") :]
    registry = registry[: registry.index("}")]
    return sorted(re.findall(r'^\s*"([a-z0-9-]+)":', registry, re.MULTILINE))


def test_selection_knows_every_registered_agent_and_version():
    assert sorted(selection.AGENTS) == _registered_agents()
    versions = (selection.ROOT / selection.VERSIONS).read_text()
    constants = set(re.findall(r'^([A-Z_]+) = "', versions, re.MULTILINE))
    assert constants - {"NODE_VERSION"} == set(selection.VERSION_OWNERS)
    assert set(selection.VERSION_OWNERS.values()) <= set(selection.AGENTS)
    for agent, owned in selection.AGENT_FILES.items():
        for entry in owned:
            assert (selection.ROOT / AGENT_DIR / entry).exists(), (agent, entry)


@pytest.mark.parametrize(
    ("changed", "versions_diff", "expected"),
    [
        ([AGENT_DIR + "pi.py"], "", ["pi"]),
        ([AGENT_DIR + "pi/rsi-never-stop.ts"], "", ["pi"]),
        ([AGENT_DIR + "zcode/launcher.sh", AGENT_DIR + "kimi_code.py"], "", ["kimi-code", "zcode"]),
        ([selection.VERSIONS], _bump("CODEX"), ["codex"]),
        ([selection.VERSIONS], _bump("CODEX", "CLAUDE_CODE"), ["codex", "claude-code"]),
        ([selection.VERSIONS], _bump("RIPGREP"), ["opencode"]),
        ([AGENT_DIR + "oracle.py", AGENT_DIR + "oracle/rsi-oracle"], "", ["oracle"]),
        ([AGENT_DIR + "codex.py", AGENT_DIR + "oracle.py"], "", ["codex", "oracle"]),
        ([selection.VERSIONS], _bump("NODE_VERSION"), ALL),
        ([AGENT_DIR + "driven.py"], "", ALL),
        ([AGENT_DIR + "driver/rsi-agent-driver.mjs"], "", ALL),
        (["RSI-Harness/src/rsi_harness/judge_agent/rsi_agent.py"], "", ALL),
        (["RSI-Harness/tests/fakes/fake_model_server.py"], "", ALL),
        (["checks/harness_agent_selection.py"], "", ALL),
        # Outside what the workflow watches: no Agent is due for it.
        (["README.md", "RSI-Harness/src/rsi_harness/runtime/network.py"], "", []),
    ],
)
def test_select(changed, versions_diff, expected):
    assert selection.select(changed, versions_diff, WATCHED) == expected


def test_a_version_table_edit_that_is_not_a_version_line_tests_every_agent():
    diff = "@@ -1 +1 @@\n-# old comment\n+# new comment\n"
    assert selection.select([selection.VERSIONS], diff, WATCHED) == ALL


def test_the_command_writes_the_job_outputs(tmp_path):
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(Path(selection.__file__)), "--all"],
        env={"GITHUB_OUTPUT": str(output), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    lines = output.read_text().splitlines()
    assert lines == [
        f"agents={selection.json.dumps(MODEL_AGENTS)}",
        "model_access=true",
        "oracle=true",
    ]


def test_the_command_selects_from_a_real_git_diff(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / AGENT_DIR).mkdir(parents=True)
    versions = repo / selection.VERSIONS
    versions.write_text('NODE_VERSION = "22.0.0"\nPI = "1.0.0"\n')
    env = {
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_AUTHOR_NAME": "CI", "GIT_AUTHOR_EMAIL": "ci@example.invalid",
        "GIT_COMMITTER_NAME": "CI", "GIT_COMMITTER_EMAIL": "ci@example.invalid",
        "PATH": "/usr/bin:/bin",
    }

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, env=env, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    versions.write_text('NODE_VERSION = "22.0.0"\nPI = "1.0.1"\n')
    git("commit", "-qam", "bump pi")
    head = git("rev-parse", "HEAD")
    monkeypatch.setattr(selection, "ROOT", repo)
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    assert selection.main(["selection", base, head]) == 0
    assert output.read_text().splitlines() == [
        'agents=["pi"]',
        "model_access=false",
        "oracle=false",
    ]
