"""Contract every registered RSI Loop Agent must meet."""

from __future__ import annotations

import json
import logging
import re
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from importlib import resources
from pathlib import Path, PurePosixPath

import pytest

from rsi_harness.errors import SetupError
from rsi_harness.integrations.rsi_loop import (
    _apply_reasoning_effort,
    rsi_loop_agent_environment,
    validate_agent_config,
    validate_agent_reasoning_effort,
)
from rsi_harness.runtime.images import _registered_agent_launcher
from rsi_loop.harness.agent import create_agent, list_agent_classes, zcode
from rsi_loop.harness.agent.base import STOP_HOOK_REASON
from rsi_loop.harness.agent.claude_code import _generate_stop_hook
from rsi_loop.harness.agent.codex import _generate_codex_stop_hook
from rsi_loop.harness.agent.driven import DrivenAgent
from rsi_loop.harness.backend import ExecResult
from rsi_loop.harness.config import RSILoopConfig
from tests.fakes import FakeDrivenAgent, register_fake_driven_agent

AGENT_NAMES = sorted(cls.name for cls in list_agent_classes())
DRIVEN_NAMES = sorted(
    cls.name for cls in list_agent_classes() if issubclass(cls, DrivenAgent)
)
_LAUNCHER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
# Synthetic, distinctive values: what every Agent needs to render a command.
API_KEY = "sk-contract-test-key-4c1e"
_NPM_INSTALL = re.compile(r"\bnpm install -g ([^;&|]+)")
_PINNED_PACKAGE = re.compile(r"(@?[a-z0-9][a-z0-9._/-]*)@\d+\.\d+\.\d+")


@pytest.fixture(autouse=True)
def _private_tmpfs_secrets(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the key file ZCode's prelude writes in a tmpfs directory of its own.

    Removing only that directory leaves alone the ones that tests running at
    the same time are still using.
    """
    root = tempfile.mkdtemp(prefix="rsi-test-", dir="/dev/shm")
    monkeypatch.setattr(zcode, "_SECRET_ROOT", root)
    yield
    shutil.rmtree(root, ignore_errors=True)


def _config(**overrides: str) -> RSILoopConfig:
    settings = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": "http://203.0.113.7:8000/v1",
        "agent_context_window": "128000",
        "agent_max_output_tokens": "8192",
        **overrides,
    }
    return RSILoopConfig(**settings)


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_agent_declares_a_bare_launcher(name: str) -> None:
    agent = create_agent(name, _config())

    command = agent.format_run_cmd("/tmp/rsi-agent-prompt.md", model="m", cwd="/w")

    assert _LAUNCHER.fullmatch(shlex.split(command)[0])


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_every_declared_reasoning_effort_applies(name: str) -> None:
    agent = create_agent(name, _config())
    command = agent.format_run_cmd(
        "/tmp/rsi-agent-prompt.md", model="m", cwd="/workspace"
    )

    for effort in sorted(agent.reasoning_efforts):
        applied = _apply_reasoning_effort(agent, command, effort)
        assert applied != command
        assert effort in applied


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_undeclared_reasoning_effort_is_refused_before_launch(name: str) -> None:
    with pytest.raises(SetupError, match="reasoning effort"):
        validate_agent_reasoning_effort(name, "not-an-effort")


def test_unknown_agent_has_no_reasoning_effort() -> None:
    with pytest.raises(SetupError, match="does not support reasoning effort"):
        validate_agent_reasoning_effort("no-such-agent", "high")


@pytest.mark.parametrize("generate", [_generate_stop_hook, _generate_codex_stop_hook])
def test_stop_hooks_send_the_shared_reason(generate) -> None:
    output = shlex.split(generate().splitlines()[-1])[-1]

    assert json.loads(output) == {"decision": "block", "reason": STOP_HOOK_REASON}


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_native_hook_agents_keep_the_command_without_a_continuation(name: str) -> None:
    agent = create_agent(name, _config())
    if agent.stop_hook is None:
        # Driven Agents and pi carry the continuation in the command.
        pytest.skip(f"{name} has no stop hook file of its own")

    plain = agent.format_run_cmd("/tmp/rsi-agent-prompt.md", model="m", cwd="/w")
    continued = agent.format_run_cmd(
        "/tmp/rsi-agent-prompt.md", model="m", cwd="/w", continuation=STOP_HOOK_REASON
    )

    assert continued == plain


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_npm_installs_pin_exact_versions(name: str) -> None:
    agent = create_agent(name, _config())
    installs = [m for c in agent.install_cmds for m in _NPM_INSTALL.findall(c)]

    for install in installs:
        packages = [arg for arg in shlex.split(install) if not arg.startswith("-")]
        assert packages
        assert all(_PINNED_PACKAGE.fullmatch(package) for package in packages)
        # A driven Agent's CLI needs no install scripts; skipping them keeps a
        # build from fetching anything beyond the pinned package.
        if isinstance(agent, DrivenAgent):
            assert "--ignore-scripts" in install.split()


@pytest.mark.parametrize("name", AGENT_NAMES)
def test_downloads_outside_npm_use_https(name: str) -> None:
    agent = create_agent(name, _config())

    for command in agent.install_cmds:
        assert "http://" not in command, command
        if "curl " in command:
            assert "https://" in command, command


@pytest.mark.parametrize("name", DRIVEN_NAMES)
def test_driven_agent_pins_the_version_it_reports(name: str) -> None:
    agent = create_agent(name, _config())

    assert agent.agent_version
    # As an npm package version or a download path segment.
    pinned = (f"@{agent.agent_version}", f"/{agent.agent_version}/")
    assert any(p in c for c in agent.install_cmds for p in pinned)


@pytest.mark.parametrize("name", DRIVEN_NAMES)
def test_driven_agent_cli_launcher_is_quarantined_and_checked(name: str) -> None:
    agent = create_agent(name, _config())
    launcher = agent.cli_argv[0]

    assert _LAUNCHER.fullmatch(launcher)
    if launcher != "node":
        assert launcher in agent.extra_launchers
    assert _registered_agent_launcher(agent) == "rsi-agent-driver"


@pytest.mark.parametrize("name", DRIVEN_NAMES)
@pytest.mark.parametrize("internet", [True, False])
def test_driven_agent_command_never_carries_the_api_key(
    name: str, internet: bool
) -> None:
    agent = create_agent(name, _config())

    command = agent.format_run_cmd(
        "/tmp/rsi-agent-prompt.md",
        model="m",
        cwd="/w",
        internet=internet,
        continuation=STOP_HOOK_REASON,
    )

    assert API_KEY not in command
    assert API_KEY not in agent.config_prelude(model="m", internet=internet)


@pytest.mark.parametrize("name", DRIVEN_NAMES)
def test_driven_agent_command_runs_with_a_hostile_model_name(
    name: str, tmp_path: Path
) -> None:
    """The prelude and driver argv survive shell quoting as one value each."""
    argv_file = tmp_path / "argv.json"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    stub = stub_bin / "rsi-agent-driver"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    model = "m 'x' \"y\" $HOME `true`"
    agent = create_agent(name, _config())

    command = agent.format_run_cmd(
        "/tmp/rsi-agent-prompt.md", model=model, cwd="/w", continuation=STOP_HOOK_REASON
    )
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        # The Agent's real environment, key included: a prelude that wrote the
        # key's value under home would be caught below.
        env={
            **rsi_loop_agent_environment(_config(), agent, model),
            "PATH": f"{stub_bin}:/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    argv = json.loads(argv_file.read_text())
    assert argv[argv.index("--model") + 1] == model
    assert argv[argv.index("--continuation-reason") + 1] == STOP_HOOK_REASON
    separator = argv.index("--")
    assert argv[separator + 1 :] == agent.agent_argv(model=model, internet=True)
    assert "/tmp/rsi-agent-prompt.md" in argv[:separator]
    for path in home.rglob("*"):
        if path.is_file():
            assert API_KEY not in path.read_text()


@pytest.mark.parametrize("name", [*DRIVEN_NAMES, FakeDrivenAgent.name])
def test_driven_agent_never_launches_after_a_failed_prelude(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without its configuration the CLI would run on its own defaults."""
    register_fake_driven_agent(monkeypatch)
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    launched = tmp_path / "launched"
    stub = stub_bin / "rsi-agent-driver"
    stub.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(launched))}\n")
    stub.chmod(0o755)
    # A file as HOME: every prelude's first mkdir under it fails.
    home = tmp_path / "home"
    home.write_text("")
    agent = create_agent(name, _config())
    command = agent.format_run_cmd(
        "/tmp/rsi-agent-prompt.md", model="m", cwd="/w", continuation=STOP_HOOK_REASON
    )

    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        env={"PATH": f"{stub_bin}:/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    assert not launched.exists()


@pytest.mark.parametrize("name", DRIVEN_NAMES)
def test_driven_agent_refuses_missing_context_limits_when_required(name: str) -> None:
    agent = create_agent(name, _config(agent_context_window=""))
    if not agent.requires_context_limits:
        pytest.skip(f"{name} does not need the model's limits")

    with pytest.raises(ValueError, match="RSI_AGENT_CONTEXT_WINDOW is not set"):
        agent.check_config()


@pytest.mark.parametrize("name", DRIVEN_NAMES)
def test_driven_agent_without_a_default_endpoint_requires_a_model(name: str) -> None:
    agent = create_agent(name, _config())

    # A CLI's own default model name means nothing to an endpoint the run
    # configures, so the run must name one.
    assert agent.requires_model or agent.default_api_base_url


@pytest.mark.parametrize("name", DRIVEN_NAMES)
def test_driven_agent_refuses_a_run_without_a_model_when_required(name: str) -> None:
    agent = create_agent(name, _config())
    if not agent.requires_model:
        pytest.skip(f"{name} has a model of its own")

    with pytest.raises(ValueError, match="needs a model"):
        agent.format_run_cmd("/tmp/rsi-agent-prompt.md", cwd="/w")
    with pytest.raises(SetupError, match="needs a model"):
        validate_agent_config(_config(), name)
    validate_agent_config(_config(), name, model="m")
    validate_agent_config(_config(agent_model="m"), name)


@pytest.mark.parametrize(
    "agent_class", [*list_agent_classes(), FakeDrivenAgent], ids=lambda c: c.name
)
def test_image_files_are_packaged_with_safe_targets(agent_class) -> None:
    package = resources.files("rsi_loop")
    for target, (resource, mode) in agent_class.image_files.items():
        assert PurePosixPath(target).is_absolute()
        assert ".." not in PurePosixPath(target).parts
        assert package.joinpath(resource).is_file()
        assert 0 <= mode <= 0o777


def _driven(**config: str) -> FakeDrivenAgent:
    return FakeDrivenAgent(
        RSILoopConfig(
            agent_context_window=config.get("window", "128000"),
            agent_max_output_tokens=config.get("output", "8192"),
        )
    )


def test_driven_agent_launches_the_driver_with_its_cli(tmp_path: Path) -> None:
    """Run the rendered command against a stub driver to check its quoting."""
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    argv_file = tmp_path / "argv.json"
    stub = stub_bin / "rsi-agent-driver"
    stub.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"open({str(argv_file)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    stub.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    reason = 'Don\'t stop.\nKeep "working" $HOME `true`.'

    command = _driven().format_run_cmd(
        "/tmp/rsi-agent-prompt.md", model="m 1", cwd="/w", continuation=reason
    )
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        env={"PATH": f"{stub_bin}:/usr/bin:/bin", "HOME": str(home)},
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(argv_file.read_text()) == [
        "--protocol",
        "acp",
        "--acp-mode",
        "yolo",
        "--agent-name",
        "fake-driven",
        "--agent-version",
        "1.2.3",
        "--model",
        "m 1",
        "--prompt-file",
        "/tmp/rsi-agent-prompt.md",
        "--continuation-reason",
        reason,
        "--",
        "fake-cli",
        "acp",
    ]
    config = json.loads((home / ".fake-cli" / "config.json").read_text())
    assert config == {
        "model": "m 1",
        "apiKeyEnv": "RSI_AGENT_API_KEY",
        "contextWindow": 128000,
        "maxOutputTokens": 8192,
        "web": True,
    }
    assert "\nexec rsi-agent-driver --protocol acp " in command
    assert "rsi-agent-prompt.md" in command


def test_driven_agent_without_continuation_has_no_reason_flag() -> None:
    command = _driven().format_run_cmd("/tmp/rsi-agent-prompt.md", cwd="/w")

    assert "--continuation-reason" not in command
    assert STOP_HOOK_REASON not in command


def test_driven_agent_launcher_is_the_driver() -> None:
    assert _registered_agent_launcher(_driven()) == "rsi-agent-driver"


@pytest.mark.parametrize(
    ("window", "output", "message"),
    [
        ("", "8192", "RSI_AGENT_CONTEXT_WINDOW is not set"),
        ("128000", "", "RSI_AGENT_MAX_OUTPUT_TOKENS is not set"),
        ("128k", "8192", "RSI_AGENT_CONTEXT_WINDOW is '128k'"),
        ("128000", "0", "RSI_AGENT_MAX_OUTPUT_TOKENS is '0'"),
        ("-5", "8192", "RSI_AGENT_CONTEXT_WINDOW is '-5'"),
        ("8192", "8192", "must be smaller than"),
    ],
)
def test_driven_agent_refuses_missing_or_invalid_context_limits(
    window: str, output: str, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        _driven(window=window, output=output).check_config()


def test_driven_agent_accepts_context_limits() -> None:
    agent = _driven(window=" 200000 ", output="16384")

    agent.check_config()

    assert agent.context_limits() == (200000, 16384)


class _RecordingBackend:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.calls: list[tuple[object, dict[str, object]]] = []

    def exec_run(self, handle, cmd, **options):
        self.calls.append((cmd, options))
        return ExecResult(output="rsi-agent-driver 1.0.0\n", exit_code=self.exit_code)

    def copy_to_container(self, *_args, **_options):
        raise AssertionError("a driven Agent installs no hook file")


def test_driven_stop_hook_only_checks_the_driver_as_the_agent(tmp_path: Path) -> None:
    backend = _RecordingBackend(0)

    _driven().install_stop_hook(backend, object(), tmp_path, logging.getLogger("t"))

    [(cmd, options)] = backend.calls
    assert options == {}
    assert "/usr/local/bin/rsi-agent-driver --version" in cmd[-1]
    assert list(tmp_path.iterdir()) == []


def test_driven_stop_hook_fails_when_the_driver_cannot_run(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="rsi-agent-driver is not runnable"):
        _driven().install_stop_hook(
            _RecordingBackend(126), object(), tmp_path, logging.getLogger("t")
        )


def test_driven_agent_needs_explicit_registration(monkeypatch) -> None:
    assert FakeDrivenAgent.name not in AGENT_NAMES

    register_fake_driven_agent(monkeypatch)

    assert isinstance(create_agent(FakeDrivenAgent.name, RSILoopConfig()), DrivenAgent)
