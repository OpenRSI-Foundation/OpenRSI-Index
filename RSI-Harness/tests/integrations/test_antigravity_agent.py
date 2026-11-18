"""Antigravity: environment, launch, settings, install, effort and refusals."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from pathlib import Path, PurePosixPath

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
from rsi_loop.harness.agent.antigravity import AntigravityAgent
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.harness.config import RSILoopConfig

API_KEY = "sk-antigravity-unit-test-key-5e21"
BASE_URL = "http://203.0.113.12:8000"
PROMPT = "/tmp/rsi-agent-prompt.md"
MODEL = "gemini-4-carbon-low"


def _config(**overrides: object) -> RSILoopConfig:
    settings: dict[str, object] = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": BASE_URL,
        **overrides,
    }
    return RSILoopConfig(**settings)


def _agent(**overrides: object) -> AntigravityAgent:
    agent = create_agent("antigravity", _config(**overrides))
    assert isinstance(agent, AntigravityAgent)
    return agent


def _driver_argv(command: str) -> list[str]:
    driver = command.splitlines()[-1]
    assert driver.startswith("exec rsi-agent-driver ")
    return shlex.split(driver.removeprefix("exec "))


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_environment_uses_the_gemini_key_and_endpoint_and_stays_offline() -> None:
    agent = _agent()

    environment = rsi_loop_agent_environment(agent._config, agent, MODEL)

    assert environment["GEMINI_API_KEY"] == API_KEY
    assert environment["GOOGLE_GEMINI_BASE_URL"] == BASE_URL
    assert [k for k, v in environment.items() if v == API_KEY] == ["GEMINI_API_KEY"]
    assert environment["AGY_CLI_DISABLE_AUTO_UPDATE"] == "true"
    assert API_KEY in rsi_loop_runtime_secret_values(agent._config)


def test_default_endpoint_is_the_gemini_api() -> None:
    agent = _agent(agent_api_base_url=None)

    environment = rsi_loop_agent_environment(agent._config, agent, MODEL)

    assert "GOOGLE_GEMINI_BASE_URL" not in environment
    assert agent.default_api_base_url == "https://generativelanguage.googleapis.com"
    validate_agent_config(agent._config, "antigravity", model=MODEL)


@pytest.mark.parametrize("resume", [False, True])
def test_command_drives_agy_over_acp_and_sets_the_model(resume: bool) -> None:
    command = _agent().format_run_cmd(
        PROMPT,
        model=MODEL,
        cwd="/workspace",
        continuation=STOP_HOOK_REASON,
        resume=resume,
        session_id="earlier-session" if resume else None,
    )

    argv = _driver_argv(command)
    assert argv[:3] == ["rsi-agent-driver", "--protocol", "acp"]
    # agy reports a failed model call as the turn's reply.
    assert _flag(argv, "--acp-error-text") == "⚠ agent executor error:"
    assert _flag(argv, "--agent-name") == "antigravity"
    assert _flag(argv, "--agent-version") == versions.ANTIGRAVITY
    # The transcript's label, and the model agy is told over ACP.
    assert _flag(argv, "--model") == _flag(argv, "--acp-model") == MODEL
    assert ("--resume-session" in argv) is resume
    assert argv[argv.index("--") + 1 :] == [
        "agy",
        "acp",
        "--dangerously-skip-permissions",
        "--disable-telemetry",
    ]


def test_prelude_writes_settings_and_hooks_and_drops_other_endpoints(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    settings = home / ".gemini" / "antigravity-cli" / "settings.json"
    hooks = home / ".gemini" / "config" / "hooks.json"
    # What an earlier attempt, or the Agent, may have left.
    settings.parent.mkdir(parents=True)
    settings.write_text(json.dumps({"customModelsConfig": {"customModels": {}}}))
    hooks.parent.mkdir(parents=True)
    hooks.write_text("{}")
    agent = _agent()
    command = agent.format_run_cmd(PROMPT, model=MODEL)
    prelude = command.rpartition("\n")[0]
    others = ["AGY_LLM_GATEWAY_URL", "AGY_ADC_AUTH", "GOOGLE_API_KEY"]

    completed = subprocess.run(
        [
            "/bin/bash",
            "-c",
            prelude + "\n" + "".join(f'echo "{n}=${{{n}-unset}}";' for n in others),
        ],
        env={
            **rsi_loop_agent_environment(agent._config, agent, MODEL),
            **{name: "http://203.0.113.99" for name in others},
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == [f"{name}=unset" for name in others]
    assert json.loads(settings.read_text()) == {
        "modelProvider": "gemini",
        "agentMode": "",
        "enableTerminalSandbox": False,
    }
    assert list(json.loads(hooks.read_text())) == ["rsi-harness"]
    mcp = home / ".gemini" / "config" / "mcp_config.json"
    assert json.loads(mcp.read_text()) == {"mcpServers": {}}
    for path in home.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text()


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gemini-4-carbon-low", "high", "gemini-4-carbon-high"),
        ("gemini-4-hydrogen-high", "xhigh", "gemini-4-hydrogen-xhigh"),
        ("gemini-3.8-flash-medium", "low", "gemini-3.8-flash-low"),
        # A catalog id without its suffix takes one.
        ("gemini-3.1-pro", "high", "gemini-3.1-pro-high"),
    ],
)
def test_reasoning_effort_picks_the_catalog_id_of_that_effort(
    model: str, effort: str, expected: str
) -> None:
    agent = _agent()
    command = agent.format_run_cmd(PROMPT, model=model)

    argv = _driver_argv(_apply_reasoning_effort(agent, command, effort))

    assert _flag(argv, "--model") == _flag(argv, "--acp-model") == expected
    oneshot = agent.format_oneshot_cmd(model=model).command
    applied = agent.apply_reasoning_effort(oneshot, effort).splitlines()[-1]
    assert _flag(shlex.split(applied), "--model") == expected


def test_reasoning_effort_outside_the_declared_set_is_refused() -> None:
    validate_agent_reasoning_effort("antigravity", "xhigh")
    with pytest.raises(SetupError, match="choose one of: high, low, medium, xhigh"):
        validate_agent_reasoning_effort("antigravity", "max")
    with pytest.raises(ValueError, match="cannot accept reasoning effort"):
        _agent().apply_reasoning_effort("exec agy acp", "high")


def _install(tmp_path: Path, machine: str) -> tuple[int, str, str]:
    """Run the Antigravity install step with stubs; return status, URL, stderr."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    fetched = tmp_path / "fetched"
    for name, script in (
        ("uname", f"echo {machine}"),
        ("curl", f'echo "$@" > {fetched}'),
        ("tar", 'test "$*" = "-xz antigravity" && cat >/dev/null'),
        ("install", f'echo "$@" > {tmp_path}/installed'),
    ):
        (stubs / name).write_text(f"#!/bin/sh\n{script}\n")
        (stubs / name).chmod(0o755)
    command = _agent().install_cmds[1]
    completed = subprocess.run(
        ["/bin/bash", "-c", command.removeprefix("sudo -E ")],
        env={"PATH": f"{stubs}:/usr/bin:/bin"},
        capture_output=True,
        check=False,
        text=True,
    )
    url = fetched.read_text().split()[-1] if fetched.exists() else ""
    return completed.returncode, url, completed.stderr


@pytest.mark.parametrize(
    ("machine", "release"),
    [
        ("x86_64", "linux-x64/cli_linux_x64"),
        ("aarch64", "linux-arm/cli_linux_arm64"),
        ("arm64", "linux-arm/cli_linux_arm64"),
    ],
)
def test_install_fetches_the_pinned_release_for_the_image_architecture(
    tmp_path: Path, machine: str, release: str
) -> None:
    status, url, stderr = _install(tmp_path, machine)

    assert status == 0, stderr
    assert url == (
        "https://storage.googleapis.com/antigravity-public/antigravity-cli/"
        f"{versions.ANTIGRAVITY}-{versions.ANTIGRAVITY_BUILD}/{release}.tar.gz"
    )
    assert (tmp_path / "installed").read_text().split() == [
        "-m",
        "0755",
        "antigravity",
        "/usr/local/bin/agy",
    ]


def test_install_refuses_an_unsupported_architecture(tmp_path: Path) -> None:
    status, url, stderr = _install(tmp_path, "riscv64")

    assert status == 1 and not url
    assert "Antigravity: no release for architecture riscv64" in stderr


def test_work_image_installs_pinned_agy_with_the_driver() -> None:
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=_agent(),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert f"{versions.ANTIGRAVITY}-{versions.ANTIGRAVITY_BUILD}" in dockerfile
    assert "sudo -E" not in dockerfile
    assert "/opt/rsi-harness/agent-driver.mjs" in dockerfile
    for launcher in ("rsi-agent-driver", "agy"):
        assert f"command -v {launcher}" in dockerfile


def test_check_config_requires_a_key_but_no_endpoint() -> None:
    with pytest.raises(SetupError, match="set RSI_AGENT_API_KEY"):
        validate_agent_config(_config(agent_api_key=None), "antigravity", model=MODEL)
    _agent(
        agent_api_key=None,
        agent_api_base_url=None,
        agent_extra_env={"GEMINI_API_KEY": API_KEY},
    ).check_config()


def test_a_run_without_a_model_is_refused_before_launch() -> None:
    with pytest.raises(ValueError, match="needs a model"):
        _agent().format_run_cmd(PROMPT)
    with pytest.raises(SetupError, match="set RSI_AGENT_MODEL or --model"):
        validate_agent_config(_config(), "antigravity")


def _reply(index: int, text: str) -> dict:
    step = {"step_index": index, "step_type": "agent_response", "text_delta": text}
    return {"step_update": step}


# agy's print-mode stream-json for a turn: a reply with a tool call, then
# the answer streamed in deltas, each reply ending in agy's newline.
_STREAM = [
    {"event": "init", "init": {"tools": ["run_command"]}},
    {"step_update": {"step_index": 0, "step_type": "user_input"}},
    _reply(1, "Let me check.\n"),
    {"step_update": {"step_index": 2, "step_type": "tool", "tool_name": "run_command"}},
    _reply(3, '{"verdict":'),
    _reply(3, '"pass"}'),
    _reply(3, "\n"),
]


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize(
    ("result", "status", "expected"),
    [
        # Only the last reply of the turn, without agy's newline.
        ({"status": "SUCCESS"}, 0, {"response": '{"verdict":"pass"}'}),
        # A failed model call, and agy's status.
        (
            {"status": "ERROR", "error": "agent executor error: Error 429"},
            3,
            {"error": True, "response": "agent executor error: Error 429"},
        ),
        # No result at all.
        (None, 0, {"error": True, "response": "no result"}),
    ],
)
def test_one_shot_prints_the_last_reply_of_the_turn(
    tmp_path: Path, result: dict | None, status: int, expected: dict
) -> None:
    agent = _agent()

    oneshot = agent.format_oneshot_cmd(model=MODEL, internet=False)

    prelude, save, check, launch = oneshot.command.split("\n")
    assert prelude == agent.config_prelude(model=MODEL, internet=False) + " &&"
    # No prompt argument: print mode reads stdin, here the saved prompt.
    assert launch.startswith(
        "set -o pipefail && agy --dangerously-skip-permissions"
        f" --disable-slash-commands --model {MODEL} --output-format stream-json"
        ' {schema_option} <"$prompt" | /usr/local/bin/node -e '
    )
    assert (oneshot.output, oneshot.output_fields) == ("stdout-json", ("response",))
    assert oneshot.error_field == "error"
    assert oneshot.schema_option is None
    stubs = tmp_path / "bin"
    stubs.mkdir()
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(json.dumps(event) for event in _STREAM)
        + ("\n" + json.dumps({"event": "result", "result": result}) if result else "")
        + "\n"
    )
    record = tmp_path / "prompt.txt"
    agy = stubs / "agy"
    agy.write_text(f"#!/bin/sh\ncat > {record}\ncat {events}\nexit {status}\n")
    agy.chmod(0o755)
    script = "\n".join([save, check, launch]).replace("{schema_option}", "")
    script = script.replace("/usr/local/bin/node", shutil.which("node") or "node")
    home = tmp_path / "home"
    home.mkdir()

    completed = subprocess.run(
        ["/bin/bash", "-c", script],
        input="the prompt\n",
        env={"PATH": f"{stubs}:/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == status, completed.stderr
    assert json.loads(completed.stdout) == expected
    assert record.read_text() == "the prompt\n"


def test_one_shot_refuses_a_prompt_agy_would_cut(tmp_path: Path) -> None:
    oneshot = _agent().format_oneshot_cmd(model=MODEL, internet=False)
    _, save, check, _ = oneshot.command.split("\n")
    home = tmp_path / "home"
    home.mkdir()

    for size, status in ((180_000, 0), (180_001, 2)):
        completed = subprocess.run(
            ["/bin/bash", "-c", "\n".join([save, check, "true"])],
            input="x" * size,
            env={"PATH": "/usr/bin:/bin", "HOME": str(home)},
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == status
    assert "at most 180000 bytes" in completed.stderr


def test_a_local_login_needs_no_key_and_an_api_key_wins() -> None:
    login = _config(agent_api_key=None, agent_local_auth=True)
    validate_agent_config(login, "antigravity", model=MODEL)
    with pytest.raises(SetupError, match="or pass --agent-auth local"):
        validate_agent_config(_config(agent_api_key=None), "antigravity", model=MODEL)
    # A key: no login is read.
    assert (
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL,
            agent_name="antigravity",
            agent_api_key=API_KEY,
            user_lookup=lambda _uid: pytest.fail("the login must not be read"),
        )
        is None
    )


def test_login_prelude_links_the_login_and_drops_the_key_and_endpoint(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    login = home / ".gemini" / "antigravity-login" / "antigravity-oauth-token"
    login.parent.mkdir(parents=True)
    login.write_text("{}")
    agent = _agent(agent_api_key=None, agent_local_auth=True)
    prelude = agent.format_run_cmd(PROMPT, model=MODEL).rpartition("\n")[0]
    names = [
        "GEMINI_API_KEY",
        "GOOGLE_GEMINI_BASE_URL",
        "CLOUD_CODE_URL",
        "AGY_ACCOUNT",
        "GOOGLE_API_KEY",
    ]

    completed = subprocess.run(
        [
            "/bin/bash",
            "-c",
            prelude + "\n" + "".join(f'echo "{n}=${{{n}-unset}}";' for n in names),
        ],
        env={
            **rsi_loop_agent_environment(agent._config, agent, MODEL),
            **{name: "from-the-operator" for name in names},
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == [f"{name}=unset" for name in names]
    agy_dir = home / ".gemini" / "antigravity-cli"
    # No modelProvider: agy authenticates with its token file.
    assert json.loads((agy_dir / "settings.json").read_text()) == {
        "agentMode": "",
        "enableTerminalSandbox": False,
    }
    assert (agy_dir / "antigravity-oauth-token").resolve() == login.resolve()
    # agy's logs, which name the account, go to the login's directory.
    assert (agy_dir / "log").resolve() == (login.parent / "log").resolve()
    assert (login.parent / "log").is_dir()
    environment = rsi_loop_agent_environment(agent._config, agent, MODEL)
    assert "GEMINI_API_KEY" not in environment


@pytest.mark.parametrize("name", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
def test_a_local_login_with_a_key_in_the_extra_env_is_refused(name: str) -> None:
    config = _config(
        agent_api_key=None, agent_local_auth=True, agent_extra_env={name: API_KEY}
    )
    with pytest.raises(SetupError, match="not both"):
        validate_agent_config(config, "antigravity", model=MODEL)
