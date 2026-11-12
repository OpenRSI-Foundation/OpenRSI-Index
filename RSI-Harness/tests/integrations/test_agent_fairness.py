"""Every Agent: subagents run on the run's model; no account-level tools.

The exact flags, environment and configuration each Agent's launches carry,
for Work (new session and resume) and a Judge's one-shot call alike. The
Docker tests in tests/integration/test_agent_fairness_docker.py check what
the pinned CLIs then send to the model.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest

from rsi_harness.integrations.rsi_loop import (
    rsi_loop_agent_environment,
    rsi_loop_oneshot,
)
from rsi_harness.models import AgentPlan, NetworkPolicy
from rsi_loop.harness.agent import create_agent
from rsi_loop.harness.agent.base import STOP_HOOK_REASON, remove_managed_config
from rsi_loop.harness.agent.claude_code import _MANAGED_CONFIG as CLAUDE_MANAGED
from rsi_loop.harness.agent.codex import _MANAGED_CONFIG as CODEX_MANAGED
from rsi_loop.harness.config import RSILoopConfig

API_KEY = "sk-fairness-unit-key-41c7"
BASE_URL = "http://203.0.113.9:8000/v1"
MODEL = "run-model"

CLAUDE_ACCOUNT_TOOLS = [
    "RemoteTrigger",
    "Projects",
    "ClaudeDesign",
    "DesignSync",
    "PushNotification",
    "Artifact",
    "ArtifactComments",
    "ArtifactData",
    "ArtifactCheck",
    "AppifactRepl",
    "ReadNotifications",
    "ListConnectors",
    "SearchMcpRegistry",
    "SendFile",
    "SendUserFile",
    "OfferChromeSetup",
    "ListAgents",
    "SendMessage",
    "CronCreate",
    "CronDelete",
    "CronList",
    "ScheduleWakeup",
    "Workflow",
    "EnterWorktree",
    "ExitWorktree",
    "ShareOnboardingGuide",
    "SearchPlugins",
    "SearchSkills",
    "ListPlugins",
    "ListSkills",
    "SuggestPluginInstall",
    "SuggestSkills",
    "PublishPlugin",
    "SuggestConnectors",
    "FetchInboxMessage",
    "SendFeedback",
]
CLAUDE_MODEL_ENV = (
    "ANTHROPIC_DEFAULT_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
)
CLAUDE_ACCOUNT_ENV = {
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    "CLAUDE_CODE_DISABLE_ORG_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_ARTIFACT": "1",
    "CLAUDE_CODE_DISABLE_CRON": "1",
    "CLAUDE_CODE_DISABLE_WORKFLOWS": "1",
}
CLAUDE_ACCOUNT_SETTINGS = {
    "disableClaudeAiConnectors": True,
    "disableRemoteControl": True,
    "remoteControlAtStartup": False,
    "disableWorkflows": True,
    "disableArtifact": True,
}
CODEX_GUARD = [
    "agents.enabled=false",
    "features.multi_agent_v2=false",
    "features.apps=false",
    "features.plugins=false",
    "features.remote_plugin=false",
    "features.tool_suggest=false",
    "features.image_generation=false",
    "features.memories=false",
]
QWEN_GUARD_TOOLS = [
    "agent",
    "workflow",
    "team_create",
    "send_message",
    "create_sub_session",
    "advisor",
    "image_gen",
    "web_search",
]


def _config(**overrides: object) -> RSILoopConfig:
    settings: dict[str, object] = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": BASE_URL,
        "agent_context_window": "131072",
        "agent_max_output_tokens": "16384",
        **overrides,
    }
    return RSILoopConfig(**settings)


def _oneshot(agent: str, *, internet: bool, model: str | None = MODEL, **config):
    return rsi_loop_oneshot(
        _config(**config),
        AgentPlan(
            name=agent,
            model=model,
            network=NetworkPolicy(mode="public" if internet else "no-network"),
        ),
    )


def _claude_same_model_env(model: str) -> dict[str, str]:
    return {
        "CLAUDE_CODE_SUBAGENT_MODEL": model,
        "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
        "CLAUDE_CODE_DISABLE_EXPLORE_INHERIT_CAP": "1",
        "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
        "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",
    }


def _launch_argv(command: str) -> list[str]:
    """The claude launch of a command, as argv (placeholders kept)."""
    return shlex.split(command.splitlines()[-1].removeprefix("exec "))


def _claude_model_env(main: str, **aliases: str) -> dict[str, str]:
    return {
        "ANTHROPIC_MODEL": main,
        **{key: aliases.get(key, "") for key in CLAUDE_MODEL_ENV},
    }


def _assert_claude_guard(
    argv: list[str],
    model: str,
    *,
    internet: bool,
    settings_extra=None,
    main: str | None = None,
    aliases: dict[str, str] | None = None,
) -> None:
    """*model* is the subagent model; *main* the pinned main one (*model*)."""
    main = model if main is None else main
    assert argv.count("--strict-mcp-config") == 1
    assert argv.count("--disallowedTools") == 1
    denied = argv[argv.index("--disallowedTools") + 1].split(",")
    web = [] if internet else ["WebSearch", "WebFetch"]
    assert denied == CLAUDE_ACCOUNT_TOOLS + web
    assert argv.count("--settings") == 1
    settings = json.loads(argv[argv.index("--settings") + 1])
    assert settings == {
        "model": main,
        "env": {
            **_claude_model_env(main, **(aliases or {})),
            **_claude_same_model_env(model),
            **CLAUDE_ACCOUNT_ENV,
        },
        **CLAUDE_ACCOUNT_SETTINGS,
        **(settings_extra or {}),
    }
    assert "--mcp-config" not in argv


# -- Claude Code ------------------------------------------------------------


@pytest.mark.parametrize("internet", [True, False])
@pytest.mark.parametrize("resume", ["new", "session", "last"])
def test_claude_work_launches_pin_subagents_and_deny_account_tools(
    internet: bool, resume: str
) -> None:
    agent = create_agent("claude-code", _config())
    command = agent.format_run_cmd(
        "/tmp/p.md",
        model=MODEL,
        internet=internet,
        resume=resume != "new",
        session_id="0b8f" if resume == "session" else None,
    )
    argv = _launch_argv(command)
    assert argv[argv.index("--model") + 1] == MODEL
    _assert_claude_guard(argv, MODEL, internet=internet)
    assert command.splitlines()[0].endswith(f" && {CLAUDE_MANAGED} &&")


def test_claude_cache_option_shares_the_one_settings_flag() -> None:
    agent = create_agent("claude-code", _config(claude_cache_opt=True))
    argv = _launch_argv(agent.format_run_cmd("/tmp/p.md", model=MODEL))
    assert "--exclude-dynamic-system-prompt-sections" in argv
    _assert_claude_guard(
        argv, MODEL, internet=True, settings_extra={"includeGitInstructions": False}
    )


def test_claude_without_a_model_lets_subagents_inherit_the_main_one() -> None:
    agent = create_agent("claude-code", _config())
    argv = _launch_argv(agent.format_run_cmd("/tmp/p.md"))
    assert "--model" not in argv
    # "" is the CLI's default for the login, over a settings file's model.
    _assert_claude_guard(argv, "inherit", internet=True, main="")
    environment = rsi_loop_agent_environment(agent._config, agent, None)
    assert environment["CLAUDE_CODE_SUBAGENT_MODEL"] == "inherit"
    assert environment["ANTHROPIC_MODEL"] == ""


def test_claude_keeps_the_operators_model_and_alias_environment() -> None:
    aliases = {
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "served-model",
        "ANTHROPIC_SMALL_FAST_MODEL": "served-small",
    }
    agent = create_agent(
        "claude-code",
        _config(agent_extra_env={**aliases, "ANTHROPIC_MODEL": "operator-model"}),
    )
    argv = _launch_argv(agent.format_run_cmd("/tmp/p.md", model="sonnet"))
    _assert_claude_guard(argv, "sonnet", internet=True, aliases=aliases)
    # Without a run model, the operator's ANTHROPIC_MODEL is the main one.
    argv = _launch_argv(agent.format_run_cmd("/tmp/p.md"))
    _assert_claude_guard(
        argv, "inherit", internet=True, main="operator-model", aliases=aliases
    )
    environment = rsi_loop_agent_environment(agent._config, agent, "sonnet")
    assert environment["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "served-model"
    assert environment["ANTHROPIC_DEFAULT_OPUS_MODEL"] == ""
    assert environment["ANTHROPIC_MODEL"] == "sonnet"


def test_claude_configured_model_is_the_subagent_model() -> None:
    agent = create_agent("claude-code", _config(agent_model="configured"))
    argv = _launch_argv(agent.format_run_cmd("/tmp/p.md"))
    assert argv[argv.index("--model") + 1] == "configured"
    _assert_claude_guard(argv, "configured", internet=True)


def test_claude_environment_carries_the_guard_over_extra_env() -> None:
    agent = create_agent(
        "claude-code",
        _config(
            agent_extra_env={
                "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "0",
                "CLAUDE_CODE_SUBAGENT_MODEL": "stronger",
                "ENABLE_CLAUDEAI_MCP_SERVERS": "true",
            }
        ),
    )
    environment = rsi_loop_agent_environment(agent._config, agent, MODEL)
    expected = {**_claude_same_model_env(MODEL), **CLAUDE_ACCOUNT_ENV}
    assert {key: environment[key] for key in expected} == expected


@pytest.mark.parametrize("internet", [True, False])
def test_claude_one_shot_pins_subagents_and_denies_account_tools(
    internet: bool,
) -> None:
    oneshot = _oneshot("claude-code", internet=internet)
    assert oneshot["command"].startswith(f"{CLAUDE_MANAGED} &&\nexec claude -p ")
    argv = _launch_argv(oneshot["command"])
    assert argv[:3] == ["claude", "-p", "--output-format"]
    _assert_claude_guard(argv, MODEL, internet=internet)
    environment = oneshot["environment"]
    assert environment["CLAUDE_CODE_SUBAGENT_MODEL"] == MODEL
    assert environment["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"


def test_claude_reasoning_effort_still_applies_to_the_guarded_launch() -> None:
    agent = create_agent("claude-code", _config())
    command = agent.apply_reasoning_effort(
        agent.format_run_cmd("/tmp/p.md", model=MODEL), "max"
    )
    argv = _launch_argv(command)
    assert argv[:3] == ["claude", "--effort", "max"]
    _assert_claude_guard(argv, MODEL, internet=True)


# -- Codex ------------------------------------------------------------------


def _codex_overrides(command: str, marker: str) -> list[str]:
    after = command.split(marker, 1)[1]
    argv = shlex.split(after)
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == "-c"]


@pytest.mark.parametrize("internet", [True, False])
@pytest.mark.parametrize("model", [MODEL, None])
@pytest.mark.parametrize("base_url", [BASE_URL, None])
@pytest.mark.parametrize("resume", ["new", "session", "last"])
def test_codex_launches_run_without_subagents_apps_or_plugins(
    internet: bool, model: str | None, base_url: str | None, resume: str
) -> None:
    agent = create_agent("codex", _config(agent_api_base_url=base_url))
    command = agent.format_run_cmd(
        "/tmp/p.md",
        model=model,
        internet=internet,
        resume=resume != "new",
        session_id="0b8f" if resume == "session" else None,
    )
    marker = "codex exec resume" if resume != "new" else "codex exec"
    assert command.splitlines()[0].endswith(f" && {CODEX_MANAGED} &&")
    launch = command.splitlines()[-1]
    assert launch.startswith(f"exec {marker} -c agents.enabled=false ")
    overrides = _codex_overrides(launch, marker)
    for flag in CODEX_GUARD:
        assert overrides.count(flag) == 1, flag
    assert ("web_search=disabled" in overrides) is not internet


@pytest.mark.parametrize("internet", [True, False])
def test_codex_one_shot_runs_without_subagents_apps_or_plugins(internet: bool) -> None:
    oneshot = _oneshot("codex", internet=internet)
    assert oneshot["command"].startswith(f"{CODEX_MANAGED} &&\nexec codex exec ")
    overrides = _codex_overrides(oneshot["command"], "codex exec")
    for flag in CODEX_GUARD:
        assert overrides.count(flag) == 1, flag


def test_codex_reasoning_effort_still_applies_to_the_guarded_launch() -> None:
    agent = create_agent("codex", _config())
    command = agent.apply_reasoning_effort(
        agent.format_run_cmd("/tmp/p.md", model=MODEL), "high"
    )
    overrides = _codex_overrides(command, "codex exec")
    assert overrides[0] == 'model_reasoning_effort="high"'
    assert set(CODEX_GUARD) <= set(overrides)


# -- Kimi Code --------------------------------------------------------------


@pytest.mark.parametrize("internet", [True, False])
def test_kimi_forces_subagents_onto_the_run_model(internet: bool) -> None:
    agent = create_agent("kimi-code", _config())
    config = tomllib.loads(agent.config_toml(model=MODEL, internet=internet))
    assert config["default_model"] == "rsi"
    assert config["models"] == {
        "rsi": {
            "provider": "rsi",
            "model": MODEL,
            "max_context_size": 131072,
            "max_output_size": 16384,
            "capabilities": ["tool_use"],
        }
    }
    assert config["secondary_model"] == {"model": "rsi", "force": True}
    # The marker apply_reasoning_effort rewrites is still there once.
    command = agent.format_run_cmd("/tmp/p.md", model=MODEL, internet=internet)
    assert "effort" in agent.apply_reasoning_effort(command, "high")


def test_kimi_reads_its_config_only_at_launch() -> None:
    agent = create_agent("kimi-code", _config(agent_extra_env={"KIMI_CODE_WATCH": "1"}))
    environment = rsi_loop_agent_environment(agent._config, agent, MODEL)
    assert environment["KIMI_CODE_WATCH"] == "0"
    oneshot = _oneshot("kimi-code", internet=False)
    assert oneshot["environment"]["KIMI_CODE_WATCH"] == "0"
    assert "[secondary_model]" in oneshot["command"]
    assert "force = true" in oneshot["command"]
    assert "disallowedTools: Agent, AgentSwarm" in oneshot["command"]


# -- OpenCode ---------------------------------------------------------------


@pytest.mark.parametrize("internet", [True, False])
def test_opencode_provider_offers_only_the_run_model(internet: bool) -> None:
    agent = create_agent("opencode", _config())
    config = agent.config(model=MODEL, internet=internet)
    provider = config["provider"]["rsi"]
    assert provider["whitelist"] == [MODEL]
    assert list(provider["models"]) == [MODEL]
    # A working directory's opencode.json cannot rename what is sent.
    assert provider["models"][MODEL]["id"] == MODEL
    assert config["enabled_providers"] == ["rsi"]
    assert config["model"] == config["small_model"] == f"rsi/{MODEL}"
    for command in (
        agent.format_run_cmd("/tmp/p.md", model=MODEL, internet=internet),
        _oneshot("opencode", internet=internet)["command"],
    ):
        assert f'"whitelist": ["{MODEL}"]' in command
        assert remove_managed_config("/etc/opencode") in command


# -- Qwen Code --------------------------------------------------------------


@pytest.mark.parametrize("internet", [True, False])
def test_qwen_denies_delegation_and_other_model_tools(internet: bool) -> None:
    agent = create_agent("qwen-code", _config())
    denied = agent.settings(internet=internet)["permissions"]["deny"]
    assert set(QWEN_GUARD_TOOLS) <= set(denied)
    assert ("web_fetch" in denied) is not internet
    argv = shlex.split(
        _oneshot("qwen-code", internet=internet)["command"].splitlines()[-1]
    )
    excluded = argv[argv.index("--exclude-tools") + 1].split(",")
    allowed = argv[argv.index("--allowed-tools") + 1].split(",")
    assert set(QWEN_GUARD_TOOLS) <= set(excluded)
    assert not set(QWEN_GUARD_TOOLS) & set(allowed)


@pytest.mark.parametrize("oneshot", [False, True])
def test_qwen_pins_model_settings_in_the_system_layer(
    tmp_path: Path, oneshot: bool
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    agent = create_agent("qwen-code", _config())
    if oneshot:
        prelude = _oneshot("qwen-code", internet=False)["command"].split(" && set -o")[
            0
        ]
    else:
        prelude = agent.config_prelude(model=MODEL, internet=False)
    completed = subprocess.run(
        ["/bin/bash", "-c", f'{prelude} && echo "$QWEN_CODE_SYSTEM_SETTINGS_PATH"'],
        env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    path = Path(completed.stdout.strip())
    assert path == home / ".qwen" / "rsi-system-settings.json"
    assert json.loads(path.read_text()) == {
        "model": {"name": MODEL},
        # Arrays replace, where an object would merge with the working
        # directory's providers: none to route a vision turn to.
        "modelProviders": [],
        "providerProtocol": [],
        "visionModel": "",
        "voiceModel": "",
        "fastModel": MODEL,
        "compactionModel": MODEL,
        "modelFallbacks": "",
        "advisorModel": "",
        "imageModel": "",
        "agents": {"builtin": {"exploreModel": "inherit"}},
        "experimental": {"agentTeam": False},
        "tools": {"workflowsEnabled": False},
        "omni": {"enabled": False},
    }


# -- pi ---------------------------------------------------------------------


@pytest.mark.parametrize("resume", [False, True])
def test_pi_work_loads_no_extension_but_its_own(resume: bool) -> None:
    agent = create_agent("pi", _config())
    command = agent.format_run_cmd(
        "/tmp/p.md",
        model=MODEL,
        resume=resume,
        session_id="0b8f" if resume else None,
        continuation=STOP_HOOK_REASON,
    )
    argv = shlex.split(command.splitlines()[-1])
    assert argv[:7] == [
        "exec",
        "pi",
        "--offline",
        "-p",
        "--no-extensions",
        "--no-approve",
        "-e",
    ]
    assert argv[7] == "/opt/rsi-harness/pi/rsi-never-stop.ts"
    assert ("--session" in argv) is resume
    assert "--thinking" in agent.apply_reasoning_effort(command, "high")


# -- ZCode ------------------------------------------------------------------


@pytest.mark.parametrize("internet", [True, False])
def test_zcode_denies_account_tools(internet: bool) -> None:
    agent = create_agent("zcode", _config())
    args = agent.driver_args(internet=internet)
    denied = args[args.index("--zcode-tool-deny") + 1].split(",")
    assert {"OffPeakList", "ReplyToChannel"} <= set(denied)
    command = _oneshot("zcode", internet=internet)["command"]
    disallowed = command.split("--disallowed-tools ", 1)[1].split()[0].split(",")
    assert {"OffPeakList", "ReplyToChannel", "OffPeakCreate", "Agent"} <= set(
        disallowed
    )


def test_zcode_provider_registry_holds_only_the_run_model() -> None:
    # ZCode resolves a subagent definition's model against this registry and
    # refuses one it lacks (model-not-found); the run's is the only one.
    agent = create_agent("zcode", _config())
    config = agent.provider_config(model=MODEL)["config"]
    [rule] = config["providerConfigRules"]["providerRules"]
    assert rule["config"]["personalModelIds"] == [MODEL]
    assert config["providerOrder"] == ["rsi"]
    [model_rule] = config["modelConfigRules"]["manualProviderModelRules"]
    assert model_rule["modelId"] == MODEL


# -- Managed config ---------------------------------------------------------


def test_managed_config_is_removed_or_the_launch_fails(tmp_path: Path) -> None:
    managed = tmp_path / "etc" / "cli"
    (managed / "managed-settings.d").mkdir(parents=True)
    (managed / "managed-settings.json").write_text("{}")
    link = tmp_path / "etc" / "link"
    link.symlink_to(tmp_path / "missing")
    step = remove_managed_config(str(managed), str(link), str(tmp_path / "none"))
    completed = subprocess.run(
        ["/bin/bash", "-c", f"{step} && echo launched"],
        capture_output=True,
        check=False,
        text=True,
    )
    assert completed.stdout == "launched\n", completed.stderr
    assert not managed.exists() and not link.is_symlink()

    # One that stays (no sudo here: PATH has none) stops the launch.
    managed.mkdir()
    (managed / "managed-settings.json").write_text("{}")
    (tmp_path / "etc").chmod(0o555)
    try:
        completed = subprocess.run(
            ["/bin/bash", "-c", f"{step} && echo launched"],
            env={"PATH": str(tmp_path / "no-bin")},
            capture_output=True,
            check=False,
            text=True,
        )
    finally:
        (tmp_path / "etc").chmod(0o755)
    assert completed.returncode == 1
    assert "launched" not in completed.stdout
    assert f"cannot remove {managed}" in completed.stderr
