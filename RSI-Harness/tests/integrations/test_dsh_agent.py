"""DeepSeek Harness (dsh): environment, patch layer, reasoning effort, refusals."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    _apply_reasoning_effort,
    rsi_loop_agent_environment,
    rsi_loop_runtime_secret_values,
    validate_agent_config,
    validate_agent_reasoning_effort,
)
from rsi_harness.models import AgentAuthSource
from rsi_harness.runtime.images import DockerImageBuilder
from rsi_harness.runtime.local_auth import resolve_agent_auth
from rsi_loop.harness.agent import create_agent, versions
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.harness.agent.dsh import SHELL_ENV_PATH, DshAgent
from rsi_loop.harness.config import RSILoopConfig

API_KEY = "sk-dsh-unit-test-key-5b21"
BASE_URL = "http://203.0.113.12:8000/v1"
PROMPT = "/tmp/rsi-agent-prompt.md"
BASE_URL_JS = '!!js "process.env.RSI_AGENT_API_BASE_URL"'
SHELL_ENV = Path(__file__).parents[2] / "src/rsi_loop/harness/agent/dsh/shell-env.sh"


def _config(**overrides: object) -> RSILoopConfig:
    settings: dict[str, object] = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": BASE_URL,
        "agent_context_window": "131072",
        "agent_max_output_tokens": "16384",
        **overrides,
    }
    return RSILoopConfig(**settings)


def _agent(**overrides: object) -> DshAgent:
    agent = create_agent("dsh", _config(**overrides))
    assert isinstance(agent, DshAgent)
    return agent


def _rows(text: str) -> dict[str, dict[str, Any]]:
    """Parse the patch layer, one JSON row per line, by entry id."""
    assert text.count(BASE_URL_JS) == 1
    rows = [
        json.loads(line.removeprefix("- ").replace(BASE_URL_JS, '"<base-url>"'))
        for line in text.splitlines()
    ]
    by_id = {row["id"]: row for row in rows}
    assert len(by_id) == len(rows)
    return by_id


def _disabled(rows: dict[str, dict[str, Any]]) -> set[str]:
    return {key for key, row in rows.items() if row.get("disabled") is True}


def _run_prelude(command: str, home: Path) -> subprocess.CompletedProcess[str]:
    """Run everything before the CLI starts, then print DSH_HOME.

    With the Agent's real environment, so a prelude that wrote the key's
    value anywhere under *home* would be caught.
    """
    prelude, _, rest = command.partition(" &&\n")
    assert rest.startswith(("exec rsi-agent-driver ", "set -o pipefail")), rest
    agent = _agent()
    return subprocess.run(
        ["/bin/bash", "-c", f'{prelude}\nprintf %s "$DSH_HOME"'],
        env={
            **rsi_loop_agent_environment(agent._config, agent, "dsh-test"),
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )


def test_environment_names_the_rsi_route_and_stays_unattended() -> None:
    agent = _agent()

    environment = rsi_loop_agent_environment(agent._config, agent, "dsh-test")

    assert environment["RSI_AGENT_API_KEY"] == API_KEY
    assert environment["RSI_AGENT_API_BASE_URL"] == BASE_URL
    assert [k for k, v in environment.items() if v == API_KEY] == ["RSI_AGENT_API_KEY"]
    # Other routes read these as their own.
    assert not {"OPENAI_API_KEY", "DEEPSEEK_API_KEY"} & set(environment)
    assert environment["DSH_PERMISSION_MODE"] == "danger-full-access"
    assert environment["DSH_TELEMETRY_DISABLED"] == "1"
    assert environment["NARB_DISABLE_NATIVE_CACHE"] == "1"
    assert API_KEY in rsi_loop_runtime_secret_values(agent._config)


def test_command_drives_the_acp_profile() -> None:
    command = _agent().format_run_cmd(
        PROMPT, model="dsh-test", cwd="/workspace", continuation=STOP_HOOK_REASON
    )

    driver = command.splitlines()[-1]
    assert driver.startswith("exec rsi-agent-driver --protocol acp --agent-name dsh ")
    assert f"--agent-version {versions.DSH}" in driver
    assert driver.endswith(" -- dsh --profile acp")


@pytest.mark.parametrize("internet", [True, False])
def test_patch_routes_every_agent_to_the_run_model(internet: bool) -> None:
    text = _agent().patch_yaml(model="dsh-test", internet=internet)
    rows = _rows(text)

    # Neither the key nor the endpoint is on disk.
    assert API_KEY not in text and BASE_URL not in text
    assert rows["llm-pi-ai"]["config"] == {
        "providers": {
            "rsi": {
                "apiKeyEnv": "RSI_AGENT_API_KEY",
                "api": "openai-completions",
                "baseURL": "<base-url>",
                "compat": {"supportsDeveloperRole": False, "supportsStore": False},
                "retryPolicy": {"mode": "always"},
                "models": [
                    {
                        "id": "dsh-test",
                        "contextWindow": 131072,
                        "maxTokens": 16384,
                        "reasoningEfforts": False,
                    }
                ],
            }
        }
    }
    selection = {"provider": "rsi", "model": "dsh-test"}
    assert rows["acp"]["config"] == selection
    assert rows["agent-default-model"]["config"] == selection
    disabled = _disabled(rows)
    assert {
        "llm-deepseek",
        "llm-deepseek-account",
        "deepseek-account",
        "credentials",
        "web-search-deepseek",
        "session-log-deepseek",
        "session-title-llm",
        "goal",
        "goal-round-driver",
        "tool-goal",
        "hmr",
    } <= disabled
    # Work keeps its instructions, skills and subagents.
    assert not {"agent-instructions", "tool-skill", "tool-subagent"} & disabled
    assert ("tool-web" in disabled) is not internet


def test_oneshot_patch_keeps_the_workspace_and_subagents_out() -> None:
    rows = _rows(_agent().patch_yaml(model="dsh-test", internet=True, oneshot=True))

    assert "acp" not in rows
    assert "retryPolicy" not in rows["llm-pi-ai"]["config"]["providers"]["rsi"]
    assert {
        "agent-instructions",
        "skill-filesystem",
        "tool-skill",
        "tool-subagent",
        "tool-subagent-fork",
        "tool-workflow",
        "tool-agent-team",
    } <= _disabled(rows)


def test_prelude_writes_the_patch_and_resets_the_profile(tmp_path: Path) -> None:
    home = tmp_path / "home"
    stale = home / ".dsh" / "profiles" / "acp"
    stale.mkdir(parents=True)
    (stale / "cordis.patch.yml").write_text("- {id: hmr, disabled: false}\n")
    (home / ".dsh" / "settings.yaml").write_text("x: 1\n")
    session = home / ".dsh" / "sessions" / "s1"
    session.mkdir(parents=True)
    model = "m 'x' \"y\" $HOME `true` \\ \U0001f600"
    agent = _agent()

    completed = _run_prelude(
        agent.format_run_cmd(PROMPT, model=model, internet=False), home
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{home}/.dsh"
    written = (home / ".dsh" / "cordis.patch.yml").read_text()
    assert written == agent.patch_yaml(model=model, internet=False)
    assert _rows(written)["acp"]["config"]["model"] == model
    assert not (home / ".dsh" / "profiles").exists()
    assert not (home / ".dsh" / "settings.yaml").exists()
    # Sessions stay for a later attempt to resume.
    assert session.is_dir()
    for path in home.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text()


@pytest.mark.parametrize("previous", ["", "own"])
def test_shell_commands_get_back_the_variables_dsh_drops(
    tmp_path: Path, previous: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    own = tmp_path / "own-bash-env.sh"
    own.write_text("export OWN_BASH_ENV=ran\n")
    command = _agent().format_run_cmd(PROMPT, model="dsh-test")
    prelude = command.rpartition("\n")[0].replace(SHELL_ENV_PATH, str(SHELL_ENV))
    token = 'a b"c $d `e`\nline two'
    # What dsh's shell tool runs: bash -c, without the variables it drops.
    tool = (
        "env -u RSI_TOKEN -u RSI_AGENT_API_KEY -u PRIVATE_KEY_FILE bash -c "
        '\'printf "%s|%s|%s|%s" "$RSI_TOKEN" "$RSI_AGENT_API_KEY" '
        '"$PRIVATE_KEY_FILE" "${OWN_BASH_ENV-}"\''
    )

    completed = subprocess.run(
        ["/bin/bash", "-c", f"{prelude}\n{tool}"],
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "RSI_TOKEN": token,
            "RSI_AGENT_API_KEY": API_KEY,
            "PRIVATE_KEY_FILE": "/k",
            **({"BASH_ENV": str(own)} if previous else {}),
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{token}|{API_KEY}|/k|{'ran' if previous else ''}"
    assert completed.stderr == ""


@pytest.mark.parametrize("effort", sorted(DshAgent.reasoning_efforts))
@pytest.mark.parametrize("oneshot", [False, True])
def test_reasoning_effort_declares_the_level_and_makes_it_the_default(
    effort: str, oneshot: bool, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    agent = _agent()
    command = (
        agent.format_oneshot_cmd(model="dsh-test").command
        if oneshot
        else agent.format_run_cmd(PROMPT, model="dsh-test")
    )

    completed = _run_prelude(_apply_reasoning_effort(agent, command, effort), home)

    assert completed.returncode == 0, completed.stderr
    rows = _rows((home / ".dsh" / "cordis.patch.yml").read_text())
    route = rows["llm-pi-ai"]["config"]["providers"]["rsi"]
    assert route["models"][0]["reasoningEfforts"] == {effort: effort}
    assert route["reasoning"] == effort


def test_reasoning_effort_outside_the_declared_set_is_refused() -> None:
    validate_agent_reasoning_effort("dsh", "max")
    supported = "choose one of: high, low, max, medium, minimal, xhigh"
    with pytest.raises(SetupError, match=supported):
        validate_agent_reasoning_effort("dsh", "ultra")
    with pytest.raises(ValueError, match="cannot accept reasoning effort"):
        _agent().apply_reasoning_effort("exec dsh --profile acp", "high")


def test_oneshot_answers_with_the_headless_final_event(tmp_path: Path) -> None:
    oneshot = _agent().format_oneshot_cmd(model="dsh-test", internet=False)

    assert oneshot.output == "stdout-json"
    assert oneshot.output_fields == ("text",)
    assert oneshot.schema_option is None
    assert "{ dsh=$(command -v dsh) || exit 127; }" in oneshot.command
    # A stand-in for dsh printing its --json stream.
    stub = tmp_path / "bin" / "dsh"
    stub.parent.mkdir()
    stub.write_text(
        "#!/bin/sh\n"
        'test "$*" = "--profile headless --json" || exit 9\n'
        'echo \'{"type":"session","sessionId":"s"}\'\n'
        'echo \'{"type":"text","text":"draft"}\'\n'
        'echo \'{"type":"final","sessionId":"s","text":"done"}\'\n'
    )
    stub.chmod(0o755)
    command = oneshot.command.replace("{schema_option}", "")
    home = tmp_path / "home"
    home.mkdir()

    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        env={"PATH": f"{stub.parent}:/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "type": "final",
        "sessionId": "s",
        "text": "done",
    }


def test_work_image_installs_pinned_dsh_with_the_driver() -> None:
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=_agent(),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert f"npm install -g --ignore-scripts @deepseek-ai/dsh@{versions.DSH}" in (
        dockerfile
    )
    assert "/opt/rsi-harness/agent-driver.mjs" in dockerfile
    assert SHELL_ENV_PATH in dockerfile
    for launcher in ("rsi-agent-driver", "dsh"):
        assert f"command -v {launcher}" in dockerfile


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"agent_api_base_url": None}, "set RSI_AGENT_API_BASE_URL"),
        ({"agent_api_key": None}, "set RSI_AGENT_API_KEY"),
    ],
)
def test_check_config_requires_an_endpoint_and_key(
    overrides: dict[str, object], missing: str
) -> None:
    with pytest.raises(SetupError, match=missing):
        validate_agent_config(_config(**overrides), "dsh")


def test_check_config_requires_the_model_limits() -> None:
    with pytest.raises(ValueError, match="RSI_AGENT_CONTEXT_WINDOW is not set"):
        _agent(agent_context_window=None).check_config()


def test_a_run_without_a_model_is_refused_before_launch() -> None:
    with pytest.raises(ValueError, match="needs a model"):
        _agent().format_run_cmd(PROMPT)


def test_local_login_is_refused() -> None:
    with pytest.raises(SetupError, match="not supported for Agent 'dsh'"):
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL, agent_name="dsh", agent_api_key=None
        )
