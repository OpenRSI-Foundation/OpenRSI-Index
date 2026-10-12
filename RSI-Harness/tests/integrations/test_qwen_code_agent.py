"""Qwen Code: provider environment, settings and refusals."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
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
from rsi_loop.harness.agent.qwen_code import QwenCodeAgent
from rsi_loop.harness.config import RSILoopConfig

API_KEY = "sk-qwen-unit-test-key-90ab"
BASE_URL = "http://203.0.113.9:8000/v1"


def _config(**overrides: object) -> RSILoopConfig:
    settings: dict[str, object] = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": BASE_URL,
        "agent_context_window": "131072",
        "agent_max_output_tokens": "16384",
        **overrides,
    }
    return RSILoopConfig(**settings)


def _agent(**overrides: object) -> QwenCodeAgent:
    agent = create_agent("qwen-code", _config(**overrides))
    assert isinstance(agent, QwenCodeAgent)
    return agent


def test_environment_uses_qwen_provider_names_and_stays_offline() -> None:
    agent = _agent()

    environment = rsi_loop_agent_environment(agent._config, agent, "qwen-test")

    assert environment["OPENAI_API_KEY"] == API_KEY
    assert environment["OPENAI_BASE_URL"] == BASE_URL
    assert environment["OPENAI_MODEL"] == "qwen-test"
    # The key is in the environment once, under the name Qwen reads.
    assert [k for k, v in environment.items() if v == API_KEY] == ["OPENAI_API_KEY"]
    assert environment["QWEN_CODE_MODELS_DEV"] == "off"
    assert environment["QWEN_TELEMETRY_ENABLED"] == "false"
    assert environment["QWEN_DISABLE_AUTO_TITLE"] == "1"
    assert environment["QWEN_CODE_SUPPRESS_YOLO_WARNING"] == "1"
    assert API_KEY in rsi_loop_runtime_secret_values(agent._config)


def test_command_drives_qwen_over_acp_in_yolo_mode() -> None:
    command = _agent().format_run_cmd(
        "/tmp/rsi-agent-prompt.md",
        model="qwen-test",
        cwd="/workspace",
        continuation=STOP_HOOK_REASON,
    )

    driver = command.splitlines()[-1]
    assert driver.startswith("exec rsi-agent-driver --protocol acp ")
    assert "--permission-prefer restore_previous,allow_always,allow_once" in driver
    assert f"--agent-name qwen-code --agent-version {versions.QWEN_CODE}" in driver
    assert driver.endswith(" -- qwen --acp --approval-mode yolo")


@pytest.mark.parametrize("internet", [True, False])
def test_settings_carry_limits_and_keep_the_run_unattended(internet: bool) -> None:
    settings = _agent().settings(internet=internet)

    assert settings["security"] == {"auth": {"selectedType": "openai"}}
    assert settings["model"] == {
        "generationConfig": {
            "contextWindowSize": 131072,
            "samplingParams": {"max_tokens": 16384},
        }
    }
    denied = settings["permissions"]["deny"]
    assert {"enter_plan_mode", "exit_plan_mode", "ask_user_question"} <= set(denied)
    assert ({"web_fetch", "web_search"} <= set(denied)) is not internet
    assert settings["experimental"] == {"cron": False}
    assert settings["general"] == {"enableAutoUpdate": False}
    assert settings["telemetry"] == {"enabled": False}
    assert settings["privacy"] == {"usageStatisticsEnabled": False}
    assert settings["memory"] == {"enableManagedAutoMemory": False}
    assert settings["ui"] == {"enableFollowupSuggestions": False}
    # Session-wide budgets would end the driven session early.
    text = json.dumps(settings)
    for budget in (
        "maxSessionTurns",
        "maxWallTimeSeconds",
        "maxToolCalls",
        "sessionTokenLimit",
    ):
        assert budget not in text


def test_prelude_writes_settings_under_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    agent = _agent()

    prelude = agent.config_prelude(model="qwen-test", internet=False)
    completed = subprocess.run(
        ["/bin/bash", "-c", prelude],
        # The Agent's real environment: a prelude writing the key's value
        # anywhere under home would be caught below.
        env={
            **rsi_loop_agent_environment(agent._config, agent, "qwen-test"),
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    written = json.loads((home / ".qwen" / "settings.json").read_text())
    assert written == agent.settings(internet=False)
    assert API_KEY not in prelude
    for path in home.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text()


def test_work_image_installs_pinned_qwen_with_the_driver() -> None:
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=_agent(),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert (
        f"npm install -g --ignore-scripts @qwen-code/qwen-code@{versions.QWEN_CODE}"
        in dockerfile
    )
    assert "sudo -E" not in dockerfile
    assert "/opt/rsi-harness/agent-driver.mjs" in dockerfile
    for launcher in ("rsi-agent-driver", "qwen"):
        assert f"command -v {launcher}" in dockerfile


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"agent_api_base_url": None}, "set RSI_AGENT_API_BASE_URL"),
        ({"agent_api_key": None}, "set RSI_AGENT_API_KEY"),
        (
            {"agent_api_base_url": None, "agent_api_key": None},
            "set RSI_AGENT_API_BASE_URL and RSI_AGENT_API_KEY",
        ),
    ],
)
def test_check_config_requires_an_endpoint_and_key(
    overrides: dict[str, object], missing: str
) -> None:
    with pytest.raises(ValueError, match=missing):
        _agent(**overrides).check_config()
    with pytest.raises(SetupError, match=missing):
        validate_agent_config(_config(**overrides), "qwen-code")


def test_check_config_accepts_qwen_variables_from_extra_env() -> None:
    _agent(
        agent_api_base_url=None,
        agent_api_key=None,
        agent_extra_env={"OPENAI_BASE_URL": BASE_URL, "OPENAI_API_KEY": API_KEY},
    ).check_config()


def test_check_config_requires_the_model_limits() -> None:
    with pytest.raises(ValueError, match="RSI_AGENT_MAX_OUTPUT_TOKENS is not set"):
        _agent(agent_max_output_tokens=None).check_config()


def test_reasoning_effort_is_refused_until_verified() -> None:
    with pytest.raises(SetupError, match="does not support reasoning effort"):
        validate_agent_reasoning_effort("qwen-code", "high")


def test_local_login_is_refused() -> None:
    with pytest.raises(SetupError, match="not supported for Agent 'qwen-code'"):
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL, agent_name="qwen-code", agent_api_key=None
        )
