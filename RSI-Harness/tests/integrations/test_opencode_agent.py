"""OpenCode: provider environment, config, install and refusals."""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
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
from rsi_loop.harness.agent.opencode import RIPGREP_PATH, OpenCodeAgent
from rsi_loop.harness.config import RSILoopConfig
from tests.factories import make_run_plan

API_KEY = "sk-opencode-unit-test-key-5b2e"
BASE_URL = "http://203.0.113.13:8000/v1"
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


def _agent(**overrides: object) -> OpenCodeAgent:
    agent = create_agent("opencode", _config(**overrides))
    assert isinstance(agent, OpenCodeAgent)
    return agent


def _run_prelude(
    command: str, home: Path, **env: str
) -> subprocess.CompletedProcess[str]:
    """Run everything before the driver's exec, then print the config.

    With the Agent's real environment, so a prelude that wrote the key's
    value anywhere under *home* would be caught.
    """
    prelude, _, driver = command.rpartition("\n")
    assert driver.startswith("exec rsi-agent-driver ")
    agent = _agent()
    return subprocess.run(
        ["/bin/bash", "-c", f'{prelude}\nprintf %s "$OPENCODE_CONFIG_CONTENT"'],
        env={
            **rsi_loop_agent_environment(agent._config, agent, "oc-test"),
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            **env,
        },
        capture_output=True,
        check=False,
        text=True,
    )


def test_environment_names_the_endpoint_for_the_config_and_stays_offline() -> None:
    agent = _agent()

    environment = rsi_loop_agent_environment(agent._config, agent, "oc-test")

    assert environment["RSI_AGENT_API_KEY"] == API_KEY
    assert environment["RSI_AGENT_API_BASE_URL"] == BASE_URL
    assert [k for k, v in environment.items() if v == API_KEY] == ["RSI_AGENT_API_KEY"]
    # OPENAI_* would also enable OpenCode's built-in provider.
    assert not any(name.startswith("OPENAI_") for name in environment)
    assert environment["OPENCODE_DISABLE_AUTOUPDATE"] == "1"
    assert environment["OPENCODE_DISABLE_MODELS_FETCH"] == "1"
    assert environment["OPENCODE_DISABLE_SHARE"] == "1"
    assert environment["OPENCODE_DISABLE_LSP_DOWNLOAD"] == "1"
    assert API_KEY in rsi_loop_runtime_secret_values(agent._config)


def test_command_drives_opencode_over_acp() -> None:
    command = _agent().format_run_cmd(
        PROMPT, model="oc-test", cwd="/workspace", continuation=STOP_HOOK_REASON
    )

    driver = command.splitlines()[-1]
    # Permissions come from the config, so there is no mode to set.
    assert driver.startswith(
        "exec rsi-agent-driver --protocol acp --agent-name opencode "
        f"--agent-version {versions.OPENCODE} --model oc-test "
    )
    assert driver.endswith(" -- opencode acp")


@pytest.mark.parametrize("internet", [True, False])
def test_config_names_the_key_and_endpoint_and_keeps_the_run_unattended(
    internet: bool,
) -> None:
    config = _agent().config(model="oc-test", internet=internet)
    text = json.dumps(config)

    assert config["model"] == config["small_model"] == "rsi/oc-test"
    assert config["enabled_providers"] == ["rsi"]
    assert config["autoupdate"] is False and config["share"] == "disabled"
    assert config["snapshot"] is False
    # Neither the key nor the endpoint is in the config.
    assert API_KEY not in text and BASE_URL not in text
    assert config["provider"] == {
        "rsi": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "rsi",
            "options": {
                "baseURL": "{env:RSI_AGENT_API_BASE_URL}",
                "apiKey": "{env:RSI_AGENT_API_KEY}",
                # Above the 5 minute defaults, for slow reasoning models.
                "headerTimeout": 900_000,
                "chunkTimeout": 900_000,
            },
            "models": {
                "oc-test": {
                    "name": "oc-test",
                    "tool_call": True,
                    "limit": {"context": 131072, "output": 16384},
                    "options": {},
                }
            },
        }
    }
    permission = config["permission"]
    assert permission["*"] == "allow"
    assert permission["doom_loop"] == "allow"
    assert permission["question"] == "deny"
    web = {permission.get("webfetch"), permission.get("websearch")}
    assert web == ({None} if internet else {"deny"})
    assert config["agent"] == {"title": {"disable": True}}
    # A step limit would end turns the session needs; compaction stays on.
    assert "steps" not in text and "compaction" not in config


def test_prelude_prepares_the_config_dir_and_exports_the_config(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    model = "m 'x' \"y\" $HOME `true` \\ \U0001f600"
    agent = _agent()

    completed = _run_prelude(
        agent.format_run_cmd(PROMPT, model=model, internet=False), home
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == agent.config(model=model, internet=False)
    config_dir = home / ".config" / "opencode"
    # OpenCode skips its plugin SDK install when the lock names the package.
    assert (config_dir / "node_modules").is_dir()
    lock = json.loads((config_dir / "package-lock.json").read_text())
    assert "@opencode-ai/plugin" in lock["packages"][""]["dependencies"]
    rg = home / ".cache" / "opencode" / "bin" / "rg"
    assert rg.is_symlink() and str(rg.readlink()) == RIPGREP_PATH
    for path in home.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text()


def test_prelude_follows_the_xdg_directories(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()

    completed = _run_prelude(
        _agent().format_run_cmd(PROMPT, model="oc-test"),
        home,
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_CACHE_HOME=str(tmp_path / "cache"),
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "config" / "opencode" / "package-lock.json").is_file()
    assert (tmp_path / "cache" / "opencode" / "bin" / "rg").is_symlink()
    assert not (home / ".config").exists()


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_sets_the_model_option(effort: str, tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    agent = _agent()
    command = agent.format_run_cmd(PROMPT, model="oc-test")

    completed = _run_prelude(_apply_reasoning_effort(agent, command, effort), home)

    assert completed.returncode == 0, completed.stderr
    model = json.loads(completed.stdout)["provider"]["rsi"]["models"]["oc-test"]
    assert model["options"] == {"reasoningEffort": effort}


def test_reasoning_effort_outside_the_declared_set_is_refused() -> None:
    validate_agent_reasoning_effort("opencode", "high")
    with pytest.raises(SetupError, match="choose one of: high, low, medium"):
        validate_agent_reasoning_effort("opencode", "xhigh")
    with pytest.raises(ValueError, match="cannot accept reasoning effort"):
        _agent().apply_reasoning_effort("exec opencode acp", "high")


def test_work_image_installs_the_pinned_binaries_with_the_driver() -> None:
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=_agent(),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert "npm pack --silent --cache" in dockerfile
    assert f"opencode-linux-${{arch}}@{versions.OPENCODE}" in dockerfile
    assert "x86_64) arch=x64 ;; aarch64|arm64) arch=arm64 ;;" in dockerfile
    assert (
        f"https://github.com/BurntSushi/ripgrep/releases/download/{versions.RIPGREP}/"
    ) in dockerfile
    assert "sudo -E" not in dockerfile
    assert "/opt/rsi-harness/agent-driver.mjs" in dockerfile
    for launcher in ("rsi-agent-driver", "opencode"):
        assert f"command -v {launcher}" in dockerfile


def _tar_gz(members: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            archive.addfile(info, io.BytesIO(data))
    return payload.getvalue()


# ``uname -m`` -> the name the package and the ripgrep release use for it.
_ARCHES = {
    "x86_64": ("x64", "x86_64-unknown-linux-musl"),
    "aarch64": ("arm64", "aarch64-unknown-linux-gnu"),
    "arm64": ("arm64", "aarch64-unknown-linux-gnu"),
}


def _run_install(
    tmp_path: Path,
    marker: str,
    fetcher: str,
    archive: bytes,
    replace: dict[str, str],
    machine: str = "x86_64",
) -> subprocess.CompletedProcess[str]:
    """Run the install command containing *marker*, *fetcher* serving *archive*.

    ``uname -m`` reports *machine*; the fetcher's arguments go to
    ``tmp_path / "fetched"``.
    """
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    served = tmp_path / "served.tgz"
    served.write_bytes(archive)
    fetched = tmp_path / "fetched"
    stub = stub_bin / fetcher
    # npm pack leaves the tarball, named after the package, in the working
    # directory; curl streams it.
    package = f"opencode-linux-{_ARCHES.get(machine, ('none',))[0]}"
    stub.write_text(
        f'#!/bin/sh\necho "$@" > {fetched}\n'
        + (
            f"cp {served} {package}-{versions.OPENCODE}.tgz\n"
            if fetcher == "npm"
            else f"cat {served}\n"
        )
    )
    stub.chmod(0o755)
    uname = stub_bin / "uname"
    uname.write_text(f"#!/bin/sh\necho {machine}\n")
    uname.chmod(0o755)
    [install] = [c for c in _agent().install_cmds if marker in c]
    command = install.removeprefix("sudo -E ")
    for old, new in replace.items():
        command = command.replace(old, new)
    return subprocess.run(
        ["/bin/bash", "-c", command],
        env={"PATH": f"{stub_bin}:/usr/bin:/bin"},
        capture_output=True,
        check=False,
        text=True,
    )


@pytest.mark.parametrize("machine", sorted(_ARCHES))
def test_install_takes_only_the_binary_from_the_package(
    tmp_path: Path, machine: str
) -> None:
    binary = b"#!/bin/sh\necho opencode\n"
    archive = _tar_gz({"package/bin/opencode": binary, "package/package.json": b"{}"})
    target = tmp_path / "opencode"

    completed = _run_install(
        tmp_path,
        "npm pack",
        "npm",
        archive,
        {"/usr/local/bin/opencode": str(target)},
        machine,
    )

    assert completed.returncode == 0, completed.stderr
    package = f"opencode-linux-{_ARCHES[machine][0]}@{versions.OPENCODE}"
    assert package in (tmp_path / "fetched").read_text().split()
    assert target.read_bytes() == binary
    assert target.stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("machine", sorted(_ARCHES))
def test_install_takes_only_rg_from_the_ripgrep_release(
    tmp_path: Path, machine: str
) -> None:
    release = f"ripgrep-{versions.RIPGREP}-{_ARCHES[machine][1]}"
    archive = _tar_gz({f"{release}/rg": b"rg", f"{release}/README.md": b"r"})
    target = tmp_path / "ripgrep"

    completed = _run_install(
        tmp_path,
        "ripgrep",
        "curl",
        archive,
        {str(Path(RIPGREP_PATH).parent): str(target)},
        machine,
    )

    assert completed.returncode == 0, completed.stderr
    url = (
        "https://github.com/BurntSushi/ripgrep/releases/download/"
        f"{versions.RIPGREP}/{release}.tar.gz"
    )
    assert url in (tmp_path / "fetched").read_text().split()
    assert [p.name for p in target.iterdir()] == ["rg"]
    assert (target / "rg").stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize(
    ("marker", "fetcher"), [("npm pack", "npm"), ("ripgrep", "curl")]
)
def test_install_refuses_an_unsupported_architecture(
    tmp_path: Path, marker: str, fetcher: str
) -> None:
    completed = _run_install(tmp_path, marker, fetcher, b"", {}, "riscv64")

    assert completed.returncode != 0
    assert "no release for architecture riscv64" in completed.stderr
    assert not (tmp_path / "fetched").exists()


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
        validate_agent_config(_config(**overrides), "opencode")


def test_check_config_requires_the_model_limits() -> None:
    with pytest.raises(ValueError, match="RSI_AGENT_MAX_OUTPUT_TOKENS is not set"):
        _agent(agent_max_output_tokens=None).check_config()


def test_a_run_without_a_model_is_refused_before_launch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a model"):
        _agent().format_run_cmd(PROMPT)

    plan = make_run_plan(tmp_path)
    agent_plan = plan.task.agent.model_copy(update={"name": "opencode"})
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"agent": agent_plan})}
    )
    with pytest.raises(SetupError, match="set RSI_AGENT_MODEL or --model"):
        RSILoopAgentAdapter(_config()).prepare(
            AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path / "prompt.md")
        )


def test_local_login_is_refused() -> None:
    with pytest.raises(SetupError, match="not supported for Agent 'opencode'"):
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL, agent_name="opencode", agent_api_key=None
        )
