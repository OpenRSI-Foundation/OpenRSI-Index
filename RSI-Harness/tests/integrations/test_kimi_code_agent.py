"""Kimi Code: provider environment, config.toml, reasoning effort and refusals."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    RSILoopAgentAdapter,
    _apply_reasoning_effort,
    rsi_loop_agent_environment,
    rsi_loop_runtime_secret_values,
    validate_agent_config,
    validate_agent_reasoning_effort,
)
from rsi_harness.models import AgentAuthSource, AgentPrepareRequest
from rsi_harness.runtime.images import DockerImageBuilder
from rsi_harness.runtime.local_auth import resolve_agent_auth
from rsi_loop.harness.agent import create_agent, versions
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.harness.agent.kimi_code import KimiCodeAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan

API_KEY = "sk-kimi-unit-test-key-3f07"
BASE_URL = "http://203.0.113.11:8000/v1"
PROMPT = "/tmp/rsi-agent-prompt.md"


def _config(**overrides: object) -> RSILoopConfig:
    settings: dict[str, object] = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": BASE_URL,
        "agent_context_window": "131072",
        "agent_max_output_tokens": "16384",
        **overrides,
    }
    return RSILoopConfig(**settings)


def _agent(**overrides: object) -> KimiCodeAgent:
    agent = create_agent("kimi-code", _config(**overrides))
    assert isinstance(agent, KimiCodeAgent)
    return agent


def _run_prelude(command: str, home: Path) -> subprocess.CompletedProcess[str]:
    """Run everything before the driver's exec, then print KIMI_CODE_HOME.

    With the Agent's real environment, so a prelude that wrote the key's
    value anywhere under *home* would be caught.
    """
    prelude, _, driver = command.rpartition("\n")
    assert driver.startswith("exec rsi-agent-driver ")
    agent = _agent()
    return subprocess.run(
        ["/bin/bash", "-c", f'{prelude}\nprintf %s "$KIMI_CODE_HOME"'],
        env={
            **rsi_loop_agent_environment(agent._config, agent, "kimi-test"),
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )


def test_environment_uses_kimi_provider_fallbacks_and_stays_offline() -> None:
    agent = _agent()

    environment = rsi_loop_agent_environment(agent._config, agent, "kimi-test")

    # Kimi's OpenAI provider reads these when its config names no base URL.
    assert environment["OPENAI_API_KEY"] == API_KEY
    assert environment["OPENAI_BASE_URL"] == BASE_URL
    assert [k for k, v in environment.items() if v == API_KEY] == ["OPENAI_API_KEY"]
    # KIMI_MODEL_NAME would replace the configured model with Kimi's own
    # environment model, which needs its own key variable.
    assert not any(name.startswith("KIMI_MODEL_") for name in environment)
    assert "kimi-test" not in environment.values()
    assert environment["KIMI_DISABLE_TELEMETRY"] == "1"
    assert environment["KIMI_CODE_NO_AUTO_UPDATE"] == "1"
    assert environment["KIMI_CODE_MODEL_CATALOG_REFRESH_ON_START"] == "0"
    assert environment["KIMI_CODE_MODEL_CATALOG_REFRESH_INTERVAL_MS"] == "0"
    assert environment["KIMI_DISABLE_CRON"] == "1"
    assert environment["KIMI_CODE_INFINITE_RETRY"] == "1"
    assert API_KEY in rsi_loop_runtime_secret_values(agent._config)


def test_command_drives_kimi_over_acp_with_the_idle_probe() -> None:
    command = _agent().format_run_cmd(
        PROMPT, model="kimi-test", cwd="/workspace", continuation=STOP_HOOK_REASON
    )

    driver = command.splitlines()[-1]
    assert driver.startswith("exec rsi-agent-driver --protocol acp --acp-mode yolo ")
    # Quoted: the driver expands the variable, after the prelude exported it.
    assert (
        "--idle-probe 'kimi-wire:$KIMI_CODE_HOME/sessions/*/{session}/agents/main/"
        "wire.jsonl'"
    ) in driver
    assert f"--agent-name kimi-code --agent-version {versions.KIMI_CODE}" in driver
    assert driver.endswith(" -- kimi acp")


@pytest.mark.parametrize("internet", [True, False])
def test_config_names_the_key_variable_and_keeps_the_run_unattended(
    internet: bool,
) -> None:
    text = _agent().config_toml(model="kimi-test", internet=internet)
    config = tomllib.loads(text)

    assert config["default_model"] == "rsi"
    assert config["default_permission_mode"] == "yolo"
    assert config["telemetry"] is False
    assert config["auto_session_title"] is False
    # Neither the key nor the endpoint is on disk.
    assert config["providers"] == {
        "rsi": {"type": "openai", "api_key_env": "OPENAI_API_KEY"}
    }
    assert API_KEY not in text and BASE_URL not in text
    assert config["models"] == {
        "rsi": {
            "provider": "rsi",
            "model": "kimi-test",
            "max_context_size": 131072,
            "max_output_size": 16384,
            "capabilities": ["tool_use"],
        }
    }
    disabled = set(config["tools"]["disabled"])
    assert {
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
        "CronCreate",
        "CreateGoal",
        "UpdateGoal",
    } <= disabled
    # Background Bash needs the task tools.
    assert not {"TaskList", "TaskOutput", "TaskStop", "Bash"} & disabled
    assert ({"FetchURL", "WebSearch"} <= disabled) is not internet
    # A per-turn step limit would end turns the session needs.
    assert "loop_control" not in config


def test_prelude_writes_config_under_kimi_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    model = "m 'x' \"y\" $HOME `true` \\ \U0001f600"
    agent = _agent()

    completed = _run_prelude(
        agent.format_run_cmd(PROMPT, model=model, internet=False), home
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == f"{home}/.kimi-code"
    written = (home / ".kimi-code" / "config.toml").read_text()
    assert written == agent.config_toml(model=model, internet=False)
    assert tomllib.loads(written)["models"]["rsi"]["model"] == model
    for path in home.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text()


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_makes_the_model_think_at_that_effort(
    effort: str, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    agent = _agent()
    command = agent.format_run_cmd(PROMPT, model="kimi-test")

    completed = _run_prelude(_apply_reasoning_effort(agent, command, effort), home)

    assert completed.returncode == 0, completed.stderr
    config = tomllib.loads((home / ".kimi-code" / "config.toml").read_text())
    # Kimi sends ``reasoning_effort`` only for a model declaring thinking.
    model = config["models"]["rsi"]
    assert model["capabilities"] == ["tool_use", "thinking"]
    assert model["support_efforts"] == [effort]
    assert model["default_effort"] == effort


def test_reasoning_effort_outside_the_declared_set_is_refused() -> None:
    validate_agent_reasoning_effort("kimi-code", "high")
    with pytest.raises(SetupError, match="choose one of: high, low, medium"):
        validate_agent_reasoning_effort("kimi-code", "xhigh")
    with pytest.raises(ValueError, match="cannot accept reasoning effort"):
        _agent().apply_reasoning_effort("exec kimi acp", "high")


def test_work_image_installs_pinned_kimi_with_the_driver() -> None:
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=_agent(),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert (
        f"npm install -g --ignore-scripts @moonshot-ai/kimi-code@{versions.KIMI_CODE}"
        in dockerfile
    )
    assert "sudo -E" not in dockerfile
    assert "/opt/rsi-harness/agent-driver.mjs" in dockerfile
    for launcher in ("rsi-agent-driver", "kimi"):
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
        validate_agent_config(_config(**overrides), "kimi-code")


def test_check_config_accepts_the_provider_variables_from_extra_env() -> None:
    _agent(
        agent_api_base_url=None,
        agent_api_key=None,
        agent_extra_env={"OPENAI_BASE_URL": BASE_URL, "OPENAI_API_KEY": API_KEY},
    ).check_config()


def test_check_config_requires_the_model_limits() -> None:
    with pytest.raises(ValueError, match="RSI_AGENT_CONTEXT_WINDOW is not set"):
        _agent(agent_context_window=None).check_config()


def test_a_run_without_a_model_is_refused_before_launch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a model"):
        _agent().format_run_cmd(PROMPT)

    plan = make_run_plan(tmp_path)
    agent_plan = plan.task.agent.model_copy(update={"name": "kimi-code"})
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"agent": agent_plan})}
    )
    with pytest.raises(SetupError, match="set RSI_AGENT_MODEL or --model"):
        RSILoopAgentAdapter(_config()).prepare(
            AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path / "prompt.md")
        )


def test_local_login_is_refused() -> None:
    with pytest.raises(SetupError, match="not supported for Agent 'kimi-code'"):
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL, agent_name="kimi-code", agent_api_key=None
        )
