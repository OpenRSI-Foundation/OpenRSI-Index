"""ZCode: provider config on tmpfs, CLI config, install and refusals."""

from __future__ import annotations

import io
import json
import shutil
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
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
from rsi_loop.harness.agent import create_agent, versions, zcode
from rsi_loop.harness.agent.base import NODE_BIN, STOP_HOOK_REASON
from rsi_loop.harness.agent.zcode import ZCODE_DIR, ZCodeAgent
from rsi_loop.harness.config import RSILoopConfig, load_config
from tests.factories import make_run_plan

API_KEY = "sk-zcode-unit-test-key-8d3a"
BASE_URL = "http://203.0.113.17:8000/v1"
PROMPT = "/tmp/rsi-agent-prompt.md"
SHM = Path("/dev/shm")
NODE = shutil.which("node")
DENIED = "AskUserQuestion,EnterPlanMode,ExitPlanMode,CronList,CronUpdate"

requires_node = pytest.mark.skipif(NODE is None, reason="node is not installed")
# _fake_deb packs the .deb with ar.
requires_ar = pytest.mark.skipif(
    shutil.which("ar") is None, reason="ar is not installed"
)


def _config(**overrides: object) -> RSILoopConfig:
    settings: dict[str, object] = {
        "agent_api_key": API_KEY,
        "agent_api_base_url": BASE_URL,
        "agent_context_window": "131072",
        "agent_max_output_tokens": "16384",
        **overrides,
    }
    return RSILoopConfig(**settings)


def _agent(**overrides: object) -> ZCodeAgent:
    agent = create_agent("zcode", _config(**overrides))
    assert isinstance(agent, ZCodeAgent)
    return agent


@pytest.fixture
def secret_root(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A tmpfs directory of the test's own for the provider config.

    Removing only that directory leaves alone the ones that tests running at
    the same time are still using.
    """
    root = Path(tempfile.mkdtemp(prefix="rsi-test-", dir=SHM))
    monkeypatch.setattr(zcode, "_SECRET_ROOT", str(root))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _run_prelude(
    command: str, home: Path, *, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run everything before the driver's exec, then print the config path.

    With the Agent's real environment, so a prelude that wrote the key's
    value anywhere under *home* would be caught.
    """
    prelude, _, driver = command.rpartition("\n")
    assert driver.startswith("exec rsi-agent-driver ")
    agent = _agent()
    if environment is None:
        environment = rsi_loop_agent_environment(agent._config, agent, "zc-test")
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            # The pinned node an Agent image installs is this host's node.
            f"{prelude.replace(NODE_BIN, str(NODE))}\n"
            'printf %s "$ZCODE_PERSONAL_PROVIDER_CONFIG_FILE"',
        ],
        env={
            **environment,
            "PATH": f"{Path(str(NODE)).parent}:/usr/bin:/bin",
            "HOME": str(home),
        },
        capture_output=True,
        check=False,
        text=True,
    )


def test_environment_names_the_endpoint_for_the_prelude_and_stays_offline() -> None:
    agent = _agent()

    environment = rsi_loop_agent_environment(agent._config, agent, "zc-test")

    assert environment["RSI_AGENT_API_KEY"] == API_KEY
    assert environment["RSI_AGENT_API_BASE_URL"] == BASE_URL
    assert [k for k, v in environment.items() if v == API_KEY] == ["RSI_AGENT_API_KEY"]
    assert environment["ZCODE_MODEL_TELEMETRY_ENABLED"] == "false"
    # ZCode's own origin, from which it would fetch an endpoint routing table
    # before each https model request, is a dead loopback port.
    assert environment["ZCODE_BASE_URL"] == "http://127.0.0.1:9"
    assert API_KEY in rsi_loop_runtime_secret_values(agent._config)


@pytest.mark.parametrize("internet", [True, False])
def test_command_drives_zcode_app_server(internet: bool) -> None:
    command = _agent().format_run_cmd(
        PROMPT,
        model="zc-test",
        cwd="/workspace",
        internet=internet,
        continuation=STOP_HOOK_REASON,
    )

    driver = command.splitlines()[-1]
    denied = DENIED if internet else f"{DENIED},WebFetch"
    assert driver.startswith(
        f"exec rsi-agent-driver --protocol zcode --zcode-tool-deny {denied} "
        f"--agent-name zcode --agent-version {versions.ZCODE} --model zc-test "
    )
    assert driver.endswith(" -- zcode app-server")


def test_cli_config_keeps_the_run_unattended_and_offline() -> None:
    config = _agent().cli_config()

    assert config["permission"] == {"mode": "yolo"}
    assert config["memory"] == {"use": False}
    assert config["plugins"] == {"enabled": False}
    assert config["modelStream"] == {"idleTimeoutMs": 600_000}
    # Hooks stay off: the driver keeps the session working.
    assert "hooks" not in config


def test_provider_config_carries_the_limits_but_not_the_key_or_endpoint() -> None:
    config = _agent().provider_config(model="zc-test")
    text = json.dumps(config)

    assert API_KEY not in text and BASE_URL not in text
    body = config["config"]
    [provider] = body["providerConfigRules"]["providerRules"]
    assert provider["providerId"] == "rsi"
    assert provider["config"]["access"] == {"type": "api-key", "apiKey": None}
    assert provider["config"]["api"] == {
        "type": "openai-chat-completions",
        "baseUrl": None,
        "headers": {},
    }
    assert provider["config"]["personalModelIds"] == ["zc-test"]
    [rule] = body["modelConfigRules"]["manualProviderModelRules"]
    assert rule["modelId"] == "zc-test"
    assert rule["config"]["properties"]["contextWindow"] == 131072
    specs = rule["config"]["optionSpecs"]
    assert specs["maxOutputTokens"] == {"max": 16384}
    # Without an effort the level adds nothing to the request.
    assert specs["reasoningLevel"] == {"values": ["default"], "map": "{}"}
    assert body["defaultModelSelection"] == {
        "providerId": "rsi",
        "modelId": "zc-test",
        "options": {"reasoningLevel": "default"},
    }


@requires_node
def test_prelude_keeps_the_key_in_a_private_file_on_tmpfs(
    tmp_path: Path, secret_root: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    model = "m 'x' \"y\" $HOME `true` \\ \U0001f600"
    agent = _agent()

    completed = _run_prelude(agent.format_run_cmd(PROMPT, model=model), home)

    assert completed.returncode == 0, completed.stderr
    path = Path(completed.stdout)
    assert path.parent.parent == secret_root and path.name == "provider_config.json"
    assert path.parent.name.startswith("rsi-zcode.")
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    written = json.loads(path.read_text())
    [provider] = written["config"]["providerConfigRules"]["providerRules"]
    assert provider["config"]["access"]["apiKey"] == API_KEY
    assert provider["config"]["api"]["baseUrl"] == BASE_URL
    provider["config"]["access"]["apiKey"] = None
    provider["config"]["api"]["baseUrl"] = None
    assert written == agent.provider_config(model=model)
    cli_config = home / ".zcode" / "cli" / "config.json"
    assert json.loads(cli_config.read_text()) == agent.cli_config()
    for file in home.rglob("*"):
        if file.is_file():
            assert API_KEY not in file.read_text()


@requires_node
@pytest.mark.usefixtures("secret_root")
def test_prelude_fails_without_the_key(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    agent = _agent()
    environment = rsi_loop_agent_environment(agent._config, agent, "zc-test")
    del environment["RSI_AGENT_API_KEY"]

    completed = _run_prelude(
        agent.format_run_cmd(PROMPT, model="zc-test"), home, environment=environment
    )

    assert completed.returncode != 0
    assert "RSI_AGENT_API_KEY and RSI_AGENT_API_BASE_URL must be set" in (
        completed.stderr
    )
    assert "rsi-zcode." not in completed.stdout


def test_prelude_refuses_to_write_the_key_outside_a_tmpfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # tmp_path is on the disk here; a tmpfs would not test the refusal.
    root = tmp_path / "disk"
    root.mkdir()
    probe = subprocess.run(
        ["stat", "-f", "-c", "%T", str(root)], capture_output=True, text=True
    )
    if probe.stdout.strip() == "tmpfs":
        pytest.skip("the test directory is itself on a tmpfs")
    monkeypatch.setattr(zcode, "_SECRET_ROOT", str(root))
    home = tmp_path / "home"
    home.mkdir()

    completed = _run_prelude(_agent().format_run_cmd(PROMPT, model="zc-test"), home)

    assert completed.returncode != 0
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("effort", ["low", "medium", "high"])
def test_reasoning_effort_selects_a_level_sent_as_reasoning_effort(
    effort: str,
) -> None:
    agent = _agent()
    command = agent.format_run_cmd(PROMPT, model="zc-test")

    applied = _apply_reasoning_effort(agent, command, effort)

    expected = agent.provider_config(model="zc-test")
    [rule] = expected["config"]["modelConfigRules"]["manualProviderModelRules"]
    rule["config"]["optionSpecs"]["reasoningLevel"] = {
        "values": [effort],
        "map": '{"reasoning_effort": reasoningLevel}',
    }
    expected["config"]["defaultModelSelection"]["options"] = {"reasoningLevel": effort}
    assert json.dumps(expected) in applied
    assert applied.replace(json.dumps(expected), "") == command.replace(
        json.dumps(agent.provider_config(model="zc-test")), ""
    )


def test_reasoning_effort_outside_the_declared_set_is_refused() -> None:
    validate_agent_reasoning_effort("zcode", "high")
    with pytest.raises(SetupError, match="choose one of: high, low, medium"):
        validate_agent_reasoning_effort("zcode", "xhigh")
    with pytest.raises(ValueError, match="cannot accept reasoning effort"):
        _agent().apply_reasoning_effort("exec zcode app-server", "high")


def test_work_image_installs_the_pinned_bundle_with_its_launcher() -> None:
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=_agent(),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert (
        "${RSI_ZCODE_MIRROR_URL:-https://cdn-zcode.z.ai/zcode/electron/releases}"
        f"/{versions.ZCODE}/linux-${{arch}}/ZCode-{versions.ZCODE}-linux-${{arch}}.deb"
    ) in dockerfile
    assert "x86_64) arch=x64 ;; aarch64|arm64) arch=arm64 ;;" in dockerfile
    assert "rm zcode.deb" in dockerfile
    assert "sudo -E" not in dockerfile
    assert "/opt/rsi-harness/agent-driver.mjs" in dockerfile
    assert '"/usr/local/bin/zcode"' in dockerfile
    for launcher in ("rsi-agent-driver", "zcode"):
        assert f"command -v {launcher}" in dockerfile


@pytest.mark.parametrize(
    ("machine", "arch"),
    [("x86_64", "x64"), ("aarch64", "arm64"), ("arm64", "arm64"), ("riscv64", None)],
)
def test_package_download_matches_the_image_architecture(
    tmp_path: Path, machine: str, arch: str | None
) -> None:
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    fetched = tmp_path / "fetched"
    # The download is all this checks: curl records its arguments and fails.
    for name, script in (
        ("uname", f"echo {machine}"),
        ("curl", f'echo "$@" > {fetched}; exit 22'),
        ("dpkg-deb", "exit 1"),
    ):
        (stub_bin / name).write_text(f"#!/bin/sh\n{script}\n")
        (stub_bin / name).chmod(0o755)
    [install] = [c for c in _agent().install_cmds if "zcode.deb" in c]

    completed = subprocess.run(
        ["/bin/bash", "-c", install.removeprefix("sudo -E ")],
        env={"PATH": f"{stub_bin}:/usr/bin:/bin"},
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode != 0
    if arch is None:
        assert "no release for architecture riscv64" in completed.stderr
        assert not fetched.exists()
    else:
        assert (
            "https://cdn-zcode.z.ai/zcode/electron/releases"
            f"/{versions.ZCODE}/linux-{arch}/ZCode-{versions.ZCODE}-linux-{arch}.deb"
        ) in fetched.read_text().split()


def test_mirror_reaches_the_package_download_as_a_build_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mirror = "https://mirror.example.test/zcode"
    monkeypatch.setenv("RSI_ZCODE_MIRROR_URL", mirror)
    config = load_config()
    dockerfile = DockerImageBuilder(None)._work_dockerfile(
        agent=create_agent("zcode", config),
        base_ref="ubuntu:24.04",
        workdir=PurePosixPath("/workspace"),
        final_user="1001",
    )

    assert create_agent("zcode", config).build_args() == {
        "RSI_ZCODE_MIRROR_URL": mirror
    }
    assert _agent().build_args() == {"RSI_ZCODE_MIRROR_URL": None}
    lines = dockerfile.splitlines()
    download = next(i for i, line in enumerate(lines) if "zcode.deb" in line)
    assert lines.index("ARG RSI_ZCODE_MIRROR_URL") < download


def test_launcher_runs_the_bundle_with_the_pinned_node() -> None:
    resource, mode = _agent().image_files["/usr/local/bin/zcode"]
    text = (
        Path(__file__).resolve().parents[2] / "src" / "rsi_loop" / resource
    ).read_text()

    assert mode == 0o755
    assert text.splitlines()[-1] == (
        f'exec /usr/local/bin/node {ZCODE_DIR}/zcode.cjs "$@"'
    )


def _fake_deb(root: Path, bundle: bytes) -> Path:
    """Build a .deb laid out like the vendor package, with *bundle* in it."""
    members = {
        "./opt/ZCode/zcode": b"desktop app, not installed",
        "./opt/ZCode/resources/glm/zcode.cjs": bundle,
        "./opt/ZCode/resources/glm/packages/p/plugin.json": b"{}",
        "./opt/ZCode/resources/config/provider/zcode-builtin.json": b'{"x": 1}',
    }
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:xz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o640
            archive.addfile(info, io.BytesIO(content))
    control = io.BytesIO()
    with tarfile.open(fileobj=control, mode="w:xz") as archive:
        text = b"Package: zcode\nVersion: 3.15.1\nArchitecture: amd64\n"
        text += b"Maintainer: test\nDescription: test\n"
        info = tarfile.TarInfo("./control")
        info.size = len(text)
        archive.addfile(info, io.BytesIO(text))
    parts = root / "parts"
    parts.mkdir()
    (parts / "debian-binary").write_bytes(b"2.0\n")
    (parts / "control.tar.xz").write_bytes(control.getvalue())
    (parts / "data.tar.xz").write_bytes(data.getvalue())
    deb = root / "fake.deb"
    subprocess.run(
        ["ar", "rc", str(deb), "debian-binary", "control.tar.xz", "data.tar.xz"],
        cwd=parts,
        check=True,
    )
    return deb


def _install(
    tmp_path: Path, deb: Path, *, tools: list[str]
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run the install command with curl serving *deb*, into a temp dir."""
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for tool in tools:
        found = shutil.which(tool)
        assert found, tool
        (stub_bin / tool).symlink_to(found)
    curl = stub_bin / "curl"
    curl.write_text(f'#!/bin/sh\ncp {deb} "$3"\n')
    curl.chmod(0o755)
    target = tmp_path / "opt-zcode"
    [install] = [c for c in _agent().install_cmds if "zcode.deb" in c]
    command = install.removeprefix("sudo -E ").replace(ZCODE_DIR, str(target))
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        env={"PATH": str(stub_bin)},
        capture_output=True,
        check=False,
        text=True,
    )
    return completed, target


# What the install command runs, besides curl and the .deb extractor.
_TOOLS = [
    "bash",
    "uname",
    "mktemp",
    "tar",
    "xz",
    "rm",
    "mkdir",
    "cp",
    "install",
    "chmod",
]


@requires_ar
@pytest.mark.parametrize("extractor", ["dpkg-deb", "ar"])
def test_install_extracts_only_the_bundle_and_the_provider_catalog(
    tmp_path: Path, extractor: str
) -> None:
    if not shutil.which(extractor):
        pytest.skip(f"{extractor} is not installed")
    bundle = b"// fake zcode bundle\n"
    deb = _fake_deb(tmp_path, bundle)

    completed, target = _install(tmp_path, deb, tools=[*_TOOLS, extractor])

    assert completed.returncode == 0, completed.stdout + completed.stderr
    files = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file())
    assert files == [
        "packages/p/plugin.json",
        "provider/zcode-builtin.json",
        "zcode.cjs",
    ]
    assert (target / "zcode.cjs").read_bytes() == bundle
    for path in target.rglob("*"):
        mode = stat.S_IMODE(path.stat().st_mode)
        assert mode & 0o004 and not mode & 0o022, (path, oct(mode))


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
        validate_agent_config(_config(**overrides), "zcode")


def test_check_config_requires_the_model_limits() -> None:
    with pytest.raises(ValueError, match="RSI_AGENT_CONTEXT_WINDOW is not set"):
        _agent(agent_context_window=None).check_config()


def test_a_run_without_a_model_is_refused_before_launch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a model"):
        _agent().format_run_cmd(PROMPT)

    plan = make_run_plan(tmp_path)
    agent_plan = plan.task.agent.model_copy(update={"name": "zcode"})
    plan = plan.model_copy(
        update={"task": plan.task.model_copy(update={"agent": agent_plan})}
    )
    with pytest.raises(SetupError, match="set RSI_AGENT_MODEL or --model"):
        RSILoopAgentAdapter(_config()).prepare(
            AgentPrepareRequest(run_plan=plan, prompt_path=tmp_path / "prompt.md")
        )


def test_local_login_is_refused() -> None:
    with pytest.raises(SetupError, match="not supported for Agent 'zcode'"):
        resolve_agent_auth(
            source=AgentAuthSource.LOCAL, agent_name="zcode", agent_api_key=None
        )
